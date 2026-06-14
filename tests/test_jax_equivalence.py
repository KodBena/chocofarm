#!/usr/bin/env python3
"""
test_jax_equivalence.py — THE load-bearing safeguard for the split forward implementation.

The codebase keeps TWO forward implementations on purpose: numpy float32 for INFERENCE (the search
leaf eval; jax-jit batch-1 is ~10× slower) and JAX for TRAINING (autodiff). The risk of two
forwards diverging is bounded by this test: it asserts

    numpy_forward(W, X)  ≈  jax.jit(jax_forward)(W, X)        to FLOAT32 precision

for BOTH the value (standardized scalar) and the policy logits, on random W and X, residual ON,
batched and single-row. It reports the actual max abs/rel differences.

Why the JIT'd forward and NOT eager jax: XLA fuses/reorders ops, so jit numerics differ from eager
jax — and the weights are TRAINED under the jit'd forward (`mlp_jax_train._az_update`'s loss
differentiates `_forward_jax`, which jit compiles). So numpy inference must match THAT (the jit'd
forward), not eager, or the search would run a subtly different net than training optimized. The
test compares `forward_jax_jit` (the public `jax.jit(_forward_jax)`).

It also benchmarks batch-1 inference latency (numpy vs jax.jit) on the real-shaped model and
reports the ratio — the measurement that justifies keeping inference in numpy.

Run pinned + bounded:
    taskset -c 3 timeout 240 /home/bork/w/vdc/venvs/generic/bin/python -m pytest \
        tests/test_jax_equivalence.py -q -s
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import jax.numpy as jnp

from chocofarm.model.env import Environment
from chocofarm.az.features import feature_dim
from chocofarm.az.actions import n_action_slots
from chocofarm.az.mlp import ValueMLP
from chocofarm.az.mlp_jax_train import forward_jax_jit, _JDTYPE


# float32 roundoff bar. The numpy `_forward` runs float64; the jit'd jax forward runs float32 (the
# inference precision). The gap is float32 representable-precision over a ~241→256→256→head matmul
# chain — abs differences land ~1e-6 (value) / ~1e-6 (logits) and rel differences are dominated by
# near-zero entries (hence abs is the meaningful bar). 1e-4 abs is comfortably above the observed
# float32 floor while still catching any real algebraic divergence (a wrong residual skip, a
# missing ReLU, a transposed weight) by orders of magnitude.
ABS_TOL = 1e-4


def _random_net(seed, hidden=256, residual=True):
    env = Environment()
    in_dim, na = feature_dim(env), n_action_slots(env)
    net = ValueMLP(in_dim, hidden=hidden, n_actions=na, seed=seed, residual=residual)
    # give the value head a non-trivial scale so v_std is not ~0 (else every diff is trivially tiny)
    rng = np.random.default_rng(seed + 100)
    net.set_value_scale(float(rng.standard_normal()), float(abs(rng.standard_normal()) + 0.5))
    return net, in_dim, na


def _jax_params(net):
    return {k: jnp.asarray(v, dtype=_JDTYPE) for k, v in net._params().items()}


def _numpy_forward(net, X):
    """The numpy float64 `_forward`: returns (v_std, logits) — the same two outputs the jax forward
    returns (value STANDARDIZED, raw policy logits; no de-standardization, no softmax — the apples-
    to-apples comparison of the two forward implementations)."""
    _, v_std, logits = net._forward(X.astype(np.float64))
    return v_std, logits


def _compare(net, X):
    v_np, lg_np = _numpy_forward(net, X)
    v_jx, lg_jx = forward_jax_jit(_jax_params(net), jnp.asarray(X, dtype=_JDTYPE))
    v_jx = np.asarray(v_jx, dtype=np.float64)
    lg_jx = np.asarray(lg_jx, dtype=np.float64)
    v_abs = float(np.max(np.abs(v_np - v_jx)))
    lg_abs = float(np.max(np.abs(lg_np - lg_jx)))
    v_rel = float(np.max(np.abs(v_np - v_jx) / (np.abs(v_np) + 1e-6)))
    lg_rel = float(np.max(np.abs(lg_np - lg_jx) / (np.abs(lg_np) + 1e-6)))
    return v_abs, lg_abs, v_rel, lg_rel


def test_numpy_jax_jit_forward_equivalence_batched():
    """numpy float64 `_forward` ≈ jax.jit float32 forward, residual ON, on a batched random X.
    BOTH value (standardized) and policy logits within float32 roundoff."""
    net, in_dim, na = _random_net(seed=0, residual=True)
    rng = np.random.default_rng(0)
    X = rng.standard_normal((37, in_dim)).astype(np.float32)
    v_abs, lg_abs, v_rel, lg_rel = _compare(net, X)
    print(f"\n[equivalence batched B=37, residual ON] "
          f"value max|Δ|={v_abs:.3e} (rel {v_rel:.3e})  "
          f"logits max|Δ|={lg_abs:.3e} (rel {lg_rel:.3e})", flush=True)
    assert v_abs < ABS_TOL, f"value abs diff {v_abs:.3e} exceeds {ABS_TOL}"
    assert lg_abs < ABS_TOL, f"logits abs diff {lg_abs:.3e} exceeds {ABS_TOL}"


def test_numpy_jax_jit_forward_equivalence_single_row():
    """Same, single-row X (the search's actual leaf-eval shape) — the row reshaped to (1, in_dim)
    so the jit kernel signature matches the batched one; the value is one scalar."""
    net, in_dim, na = _random_net(seed=1, residual=True)
    rng = np.random.default_rng(1)
    X = rng.standard_normal((1, in_dim)).astype(np.float32)
    v_abs, lg_abs, v_rel, lg_rel = _compare(net, X)
    print(f"[equivalence single-row, residual ON]    "
          f"value max|Δ|={v_abs:.3e} (rel {v_rel:.3e})  "
          f"logits max|Δ|={lg_abs:.3e} (rel {lg_rel:.3e})", flush=True)
    assert v_abs < ABS_TOL
    assert lg_abs < ABS_TOL


def test_numpy_jax_jit_forward_equivalence_across_seeds():
    """The equivalence is not a single-seed accident: hold over several random W and X, residual ON
    AND OFF (the OFF path skips the block — both must match)."""
    worst_v = worst_lg = 0.0
    for seed in range(5):
        for residual in (True, False):
            net, in_dim, na = _random_net(seed=seed, residual=residual)
            rng = np.random.default_rng(seed + 50)
            X = rng.standard_normal((23, in_dim)).astype(np.float32)
            v_abs, lg_abs, _, _ = _compare(net, X)
            worst_v = max(worst_v, v_abs)
            worst_lg = max(worst_lg, lg_abs)
            assert v_abs < ABS_TOL, (seed, residual, v_abs)
            assert lg_abs < ABS_TOL, (seed, residual, lg_abs)
    print(f"[equivalence 5 seeds × residual ON/OFF]   "
          f"worst value max|Δ|={worst_v:.3e}  worst logits max|Δ|={worst_lg:.3e}", flush=True)


def test_production_f32_forward_matches_jax_jit():
    """Pin the ACTUAL inference path against the jit'd training forward. The other equivalence
    tests compare the numpy float64 `_forward` against `forward_jax_jit` — but PRODUCTION inference
    runs `_predict_both_f32` (the hand-written float32 forward, the search's hot path), a THIRD
    forward implementation. `_forward` ↔ `_predict_both_f32` is trusted by inspection; this test
    closes that gap by pinning the float32 path the search actually runs against the jit'd forward
    the weights were trained under. (Out-of-frame-audit finding: the load-bearing safeguard
    certified numpy-f64 ≈ jax-f32 while the live path is numpy-f32; this pins numpy-f32 ≈ jax-f32.)

    Compared quantities: the DE-STANDARDIZED value and the MASKED-SOFTMAX policy (what
    `_predict_both_f32` returns), reconstructed identically from the jit'd forward's (v_std,
    logits). Residual ON, single-row (the leaf shape) and batched."""
    from chocofarm.az.dtypes import is_float32
    if not is_float32():
        # the f32 path only runs under the float32 DTYPE; under float64 predict_both takes the
        # f64 branch which IS `_forward` (already covered). Skip rather than assert vacuously.
        print("\n[f32-forward pin] CHOCO_AZ_DTYPE=float64 — f32 path inactive, skipping", flush=True)
        return
    net, in_dim, na = _random_net(seed=3, hidden=256, residual=True)
    params = _jax_params(net)
    rng = np.random.default_rng(3)
    for B in (1, 29):
        X = rng.standard_normal((B, in_dim)).astype(np.float32)
        # random legal masks (at least the terminate-ish slot legal so softmax has support)
        LM = (rng.random((B, na)) > 0.3).astype(np.float32)
        LM[:, -1] = 1.0
        # production f32 forward (de-standardized value + masked softmax)
        v_prod, p_prod = net._predict_both_f32(X if B > 1 else X[0], LM if B > 1 else LM[0])
        v_prod = np.atleast_1d(np.asarray(v_prod, dtype=np.float64))
        p_prod = np.atleast_2d(np.asarray(p_prod, dtype=np.float64))
        # jit'd forward → de-standardize value + masked softmax (mirror the f32 path's tail)
        v_std_jx, lg_jx = forward_jax_jit(params, jnp.asarray(X, dtype=_JDTYPE))
        v_jx = np.asarray(v_std_jx, dtype=np.float64) * net.y_std + net.y_mean
        lg_jx = np.asarray(lg_jx, dtype=np.float64)
        masked = np.where(LM > 0, lg_jx, -1e30)
        masked = masked - masked.max(axis=1, keepdims=True)
        e = np.exp(masked) * (LM > 0)
        p_jx = e / np.where(e.sum(1, keepdims=True) > 0, e.sum(1, keepdims=True), 1.0)
        v_abs = float(np.max(np.abs(v_prod - v_jx)))
        p_abs = float(np.max(np.abs(p_prod - p_jx)))
        print(f"[f32-forward pin B={B}] value max|Δ|={v_abs:.3e}  policy max|Δ|={p_abs:.3e}",
              flush=True)
        assert v_abs < ABS_TOL, f"production f32 value diverges from jit forward: {v_abs:.3e}"
        assert p_abs < ABS_TOL, f"production f32 policy diverges from jit forward: {p_abs:.3e}"


def test_batch1_inference_latency_numpy_vs_jax():
    """Benchmark batch-1 inference latency: numpy float32 `predict_both` vs the jax.jit forward in
    the SEARCH'S ACTUAL DISPATCH SHAPE — fresh numpy arrays in, numpy arrays out (the host↔device
    round-trip the per-leaf eval pays). This is the measurement that justifies keeping inference in
    numpy. It is the realistic case the perf doc names: a pre-built on-device jnp array reused every
    call is ~1× (the misleading microbench), but the search hands the MLP fresh numpy arrays one
    leaf at a time, and THAT path pays the dispatch + transfer tax.

    The jax side here mirrors `mlp_jax.MlpJaxForward.predict_both` exactly: `jnp.asarray` the fresh
    numpy input, run the jit'd forward, `np.asarray` the outputs back. Reports the ratio. The hard
    assertion is soft-bounded (>1.5×) because a single shared-host core is noisy around the ~6–10×
    the design measured; the direction (jax materially slower at batch-1) is what matters."""
    net, in_dim, na = _random_net(seed=2, hidden=256, residual=True)
    x1 = np.random.default_rng(2).standard_normal(in_dim).astype(np.float32)
    lm = np.ones(na, dtype=np.float32)
    params = _jax_params(net)

    def jax_predict_fresh(x_np, lm_np):
        """The realistic per-leaf jax dispatch: fresh numpy -> device -> compute -> host."""
        xj = jnp.asarray(x_np[None, :], dtype=_JDTYPE)
        v, lg = forward_jax_jit(params, xj)
        return np.asarray(v), np.asarray(lg)

    # warm both paths (numpy f32 cache build; jax trace+compile) — excluded from timing
    net.predict_both(x1, lm)
    jax_predict_fresh(x1, lm)

    REP = 400
    t0 = time.perf_counter()
    for _ in range(REP):
        net.predict_both(x1, lm)
    t_np = (time.perf_counter() - t0) / REP

    t0 = time.perf_counter()
    for _ in range(REP):
        jax_predict_fresh(x1, lm)
    t_jx = (time.perf_counter() - t0) / REP

    ratio = t_jx / t_np if t_np > 0 else float("inf")
    print(f"\n[batch-1 latency, fresh-array per-leaf dispatch] "
          f"numpy f32 predict_both = {t_np*1e6:.1f} µs/call  |  "
          f"jax.jit forward = {t_jx*1e6:.1f} µs/call  |  jax/numpy ratio = {ratio:.1f}×", flush=True)
    # the design premise: jax-jit batch-1 is materially slower than numpy at single-row dispatch.
    # A soft bound (>1.5×) confirms the direction without flaking on host-noise around a hard 6–10×.
    assert ratio > 1.5, (f"expected jax-jit batch-1 materially slower than numpy "
                         f"(the premise for keeping numpy inference); got {ratio:.1f}×")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("all jax-equivalence checks passed")
