#!/usr/bin/env python3
"""
chocofarm AZ — the ONE precision-agnostic forward core for the value+policy MLP (audit R11).

Public Domain (The Unlicense).

There used to be FOUR hand-transcribed copies of the same trunk+residual forward graph, kept
bit-compatible by hand and pinned (only partially) by `tests/test_jax_equivalence.py`: the numpy
float64 `mlp._forward`, the numpy float32 hot path `mlp._predict_both_f32`, the jax training
forward `mlp_jax_train._forward_jax`, and the residual-DROPPING jax inference forward
`mlp_jax._forward_both`. Four transcriptions of one graph is three too many — and the fourth was
silently wrong (it read the trunk output straight into the heads, omitting the residual block).

This module is the single source of truth. `forward_core(params, X, xp)` runs the graph once,
parameterized on the array module `xp` (pass `numpy` for the numpy paths, `jax.numpy` for the jax
paths). `@`, `.maximum`, and `.ravel()` are spelled identically on numpy and jax arrays, so ONE
function body serves every backend and every precision; precision is whatever dtype `X`/`params`
carry. The residual block is keyed by the presence of `"Wr1"` in `params` (the same toggle the
numpy net's `self.residual` controls via `_params()`), and the policy head by `"Wp"` — so a
value-only Stage-1 net (no policy head) returns `logits=None`. The graph is, exactly:

    z1 = X @ W1 + b1;  a1 = ReLU(z1)
    z2 = a1 @ W2 + b2;  a2 = ReLU(z2)
    if residual:  head_in = a2 + (ReLU(a2@Wr1+br1) @ Wr2 + br2)   # pre-activation skip, NO outer ReLU
    else:         head_in = a2
    v_std  = (head_in @ Wv + bv).ravel()        # STANDARDIZED scalar value (de-std at the callers)
    logits = (head_in @ Wp + bp) if Wp else None

Because every forward now routes through this one body, there is structurally NO path that can run
a residual net through a residual-dropping graph — the R11 bug is impossible by construction. The
numpy↔jit equivalence test (`tests/test_jax_equivalence.py`) now guards NUMERICS, not transcription.
"""
from __future__ import annotations

from typing import Any


def forward_core(
        params: dict[str, Any],
        X: Any,
        xp: Any,  # numpy or jax.numpy — the backend-polymorphic module seam (ADR-0012 P8 commented use-site Any)
) -> tuple[Any, Any | None]:
    """Run the value+policy forward graph once, in whatever precision `X`/`params` carry.

    `params` is a flat dict keyed exactly like `ValueMLP._params()`: W1 b1 W2 b2 [Wr1 br1 Wr2 br2]
    Wv bv [Wp bp]. `xp` is the array module (`numpy` or `jax.numpy`) — `@`, `xp.maximum`, and
    `.ravel()` are spelled identically on both, so this one body is the numpy-f64, numpy-f32, and
    jax forwards at once. Returns `(v_std, logits)` where `v_std` is the STANDARDIZED scalar value
    (shape (B,), the callers de-standardize) and `logits` is the (B, n_actions) policy logits, or
    `None` when the net has no policy head (`"Wp"` absent — the value-only Stage-1 net).

    The residual block is applied iff `"Wr1"` is present in `params` (the numpy net's `self.residual`
    toggle, surfaced through `_params()`); it is the no-outer-ReLU pre-activation skip
    `head_in = a2 + (ReLU(a2@Wr1+br1)@Wr2+br2)`. This is a pure function of `(params, X)` so
    `jax.value_and_grad(loss)` differentiates it exactly."""
    z1 = X @ params["W1"] + params["b1"]
    a1 = xp.maximum(z1, 0.0)
    z2 = a1 @ params["W2"] + params["b2"]
    a2 = xp.maximum(z2, 0.0)
    if "Wr1" in params:
        zr1 = a2 @ params["Wr1"] + params["br1"]
        ar1 = xp.maximum(zr1, 0.0)
        zr2 = ar1 @ params["Wr2"] + params["br2"]
        head_in = a2 + zr2                       # pre-activation skip, NO outer ReLU (firewall A/B: best CE)
    else:
        head_in = a2
    v_std = (head_in @ params["Wv"] + params["bv"]).ravel()
    logits = (head_in @ params["Wp"] + params["bp"]) if "Wp" in params else None
    return v_std, logits
