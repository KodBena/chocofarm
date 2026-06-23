#!/usr/bin/env python3
"""
throughput-lab/server/lifted/forward.py — the value+policy MLP forward graph, COPIED VERBATIM from
chocofarm/az/forward.py (the `forward_core` SSOT, audit R11). This is one of the ONLY two pieces
lifted from chocofarm into the clean-room testbed (the other is the dtype pin in this same package);
everything else in throughput-lab is re-implemented fresh.

PROVENANCE: copied from chocofarm/az/forward.py @ d30fe8e (verified byte-identical executable graph;
only a trailing inline comment was dropped). Keep it byte-faithful to the original so the testbed's
server runs the SAME forward the production server runs (the throughput is then a throughput of the
same compute). If the parent's forward_core changes, re-copy and note it here.

The phantom-typed jax/numpy ACL is the `xp` parameter: `forward_core(params, X, xp)` runs ONE graph
body parameterized on the array module (`numpy` for the numpy paths, `jax.numpy` for the jax paths)
— `@`, `.maximum`, and `.ravel()` are spelled identically on both, so one body serves every backend
and precision (precision is whatever dtype X/params carry). The residual block is keyed by the
presence of "Wr1" in params; the policy head by "Wp" (absent => logits=None, the value-only net).

The graph, exactly:

    z1 = X @ W1 + b1;  a1 = ReLU(z1)
    z2 = a1 @ W2 + b2;  a2 = ReLU(z2)
    if residual:  head_in = a2 + (ReLU(a2@Wr1+br1) @ Wr2 + br2)   # pre-activation skip, NO outer ReLU
    else:         head_in = a2
    v_std  = (head_in @ Wv + bv).ravel()        # STANDARDIZED scalar value (de-std at the callers)
    logits = (head_in @ Wp + bp) if Wp else None

Public Domain (The Unlicense).
"""
from __future__ import annotations

from typing import Any


def forward_core(
        params: dict[str, Any],
        X: Any,
        xp: Any,  # numpy or jax.numpy — the backend-polymorphic module seam (the phantom-typed ACL)
) -> tuple[Any, Any | None]:
    """Run the value+policy forward graph once, in whatever precision X/params carry.

    `params` is a flat dict keyed like the parent's ValueMLP._params(): W1 b1 W2 b2 [Wr1 br1 Wr2 br2]
    Wv bv [Wp bp]. `xp` is the array module (numpy or jax.numpy). Returns (v_std, logits) where
    v_std is the STANDARDIZED scalar value (shape (B,), the callers de-standardize) and logits is the
    (B, n_actions) policy logits, or None when the net has no policy head ("Wp" absent — value-only).

    The residual block is applied iff "Wr1" is present (the no-outer-ReLU pre-activation skip
    head_in = a2 + (ReLU(a2@Wr1+br1)@Wr2+br2)). A pure function of (params, X)."""
    z1 = X @ params["W1"] + params["b1"]
    a1 = xp.maximum(z1, 0.0)
    z2 = a1 @ params["W2"] + params["b2"]
    a2 = xp.maximum(z2, 0.0)
    if "Wr1" in params:
        zr1 = a2 @ params["Wr1"] + params["br1"]
        ar1 = xp.maximum(zr1, 0.0)
        zr2 = ar1 @ params["Wr2"] + params["br2"]
        head_in = a2 + zr2                       # pre-activation skip, NO outer ReLU
    else:
        head_in = a2
    v_std = (head_in @ params["Wv"] + params["bv"]).ravel()
    logits = (head_in @ params["Wp"] + params["bp"]) if "Wp" in params else None
    return v_std, logits
