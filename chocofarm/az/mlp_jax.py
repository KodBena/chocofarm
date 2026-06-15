#!/usr/bin/env python3
"""
chocofarm AZ — JAX-jit inference forward for the value+policy MLP (the search leaf eval).

The per-leaf `predict_both` (one trunk eval feeding both the leaf value and the PUCT prior) is
the search's NN hotspot (~64µs/call float64 numpy). The trunk is two 256×256 matmuls + heads —
XLA's fused, float32 codegen beats numpy's separate BLAS calls even at single-row dispatch
(34µs vs 64µs measured; see docs/results/az-jax-perf.md). Training stays in `mlp.py` (manual
float64 backprop, off the hot path — once per iteration, not per leaf); this module is inference
only.

The jitted function takes the weights as a params-dict ARGUMENT (not a closure) so it does not
recompile when the net is retrained between iterations — XLA caches one compiled kernel per
(shape, dtype) signature, which is constant across the whole run. `MlpJaxForward` wraps a
`ValueMLP`, holds its weights as a device-array params dict at the chosen precision, and exposes a
numpy-in / numpy-out `predict_both` matching the `ValueMLP` signature so it is a drop-in for the
search.

The forward graph is the ONE `forward.forward_core` (audit R11) — the SAME function the numpy net
and the jax trainer run — so this wrapper HONORS the net's residual block (applied iff the params
dict carries `Wr1`) instead of silently dropping it. Before R11 this module hand-transcribed a
residual-BLIND trunk-only forward; routing it through the shared core makes that drop structurally
impossible.

float32 by default (the parametric DTYPE). The masked softmax matches `ValueMLP._masked_softmax`
in form; XLA + float32 will differ in the last bits and may flip near-tied argmax/SH choices —
expected and acceptable (the brief's behavioral-equivalence bar, not bit-equivalence).
"""
from __future__ import annotations

import os

# Keep XLA single-threaded — the bench is core-pinned (taskset -c 2) and the loop runs one core,
# so multi-threaded Eigen would only add contention. Set before jax imports.
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import jax
import jax.numpy as jnp

from chocofarm.az.dtypes import DTYPE
from chocofarm.az.forward import forward_core

_JDTYPE = jnp.float32 if np.dtype(DTYPE) == np.dtype(np.float32) else jnp.float64


@jax.jit
def _forward_both(params, x, lm, ym, ys):
    """Value head (de-standardized) + masked-softmax policy over the ONE `forward.forward_core`
    (audit R11). `params` is the flat weight dict keyed like `ValueMLP._params()`, so the residual
    block is applied iff `"Wr1"` is present — exactly as in the numpy/jax-train forwards. There is
    NO separate residual-blind graph here anymore: this wrapper cannot drop the residual block (the
    R11 bug is structurally impossible). `x`: (B,in); `lm`: matching legal mask. Returns (v, p)."""
    v_std, logits = forward_core(params, x, jnp)
    v = v_std * ys + ym
    neg = jnp.asarray(-1e30, dtype=logits.dtype)
    legal = lm > 0
    masked = jnp.where(legal, logits, neg)
    masked = masked - masked.max(axis=-1, keepdims=True)
    e = jnp.exp(masked) * legal
    denom = e.sum(axis=-1, keepdims=True)
    denom = jnp.where(denom > 0, denom, 1.0)
    p = e / denom
    return v, p


class MlpJaxForward:
    """JAX-jit inference wrapper over a `ValueMLP`. Drop-in `predict_both(X, legal_mask)` for the
    search leaf eval. Holds the net's weights as a device-array params dict at `DTYPE`; rebuild via
    `refresh()` if the underlying net is retrained (the ExIt loop builds a fresh search per
    iteration, so a fresh wrapper per iteration is the natural seam — but `refresh()` is provided
    for reuse). The forward is `forward.forward_core` (the same graph the numpy net and the jax
    trainer run), so it honors the net's residual block instead of silently dropping it."""

    def __init__(self, net):
        if net.n_actions is None:
            raise ValueError("MlpJaxForward needs a net with a policy head (n_actions set)")
        self.net = net
        self.refresh()

    def refresh(self):
        net = self.net
        d = _JDTYPE
        # the full params dict keyed like `_params()` — INCLUDING the residual block when the net
        # has it, so `forward_core` applies it (the residual-drop fix). Single source of which
        # params exist: the net's own `_params()`.
        self.params = {k: jnp.asarray(v, dtype=d) for k, v in net._params().items()}
        self.ym = jnp.asarray(net.y_mean, dtype=d)
        self.ys = jnp.asarray(net.y_std, dtype=d)

    def predict_both(self, X, legal_mask):
        """Match `ValueMLP.predict_both`: 1-D X -> (float value, (n_actions,) numpy policy);
        2-D X -> ((B,) numpy values, (B,n_actions) numpy policy)."""
        single = (X.ndim == 1)
        x = jnp.asarray(X, dtype=_JDTYPE)
        lm = jnp.asarray(legal_mask, dtype=_JDTYPE)
        if single:
            x = x[None, :]
            lm = lm[None, :]
        v, p = _forward_both(self.params, x, lm, self.ym, self.ys)
        v = np.asarray(v)
        p = np.asarray(p)
        if single:
            return float(v[0]), p[0]
        return v, p

    def warmup(self, in_dim, n_actions):
        """Compile the kernel on a representative single-row input so the first real leaf eval
        doesn't pay the trace+compile cost."""
        x = np.zeros(in_dim, dtype=np.float32)
        lm = np.ones(n_actions, dtype=np.float32)
        self.predict_both(x, lm)
