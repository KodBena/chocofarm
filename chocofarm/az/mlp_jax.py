#!/usr/bin/env python3
"""
chocofarm AZ — JAX-jit inference forward for the value+policy MLP (the search leaf eval).

The per-leaf `predict_both` (one trunk eval feeding both the leaf value and the PUCT prior) is
the search's NN hotspot (~64µs/call float64 numpy). The trunk is two 256×256 matmuls + heads —
XLA's fused, float32 codegen beats numpy's separate BLAS calls even at single-row dispatch
(34µs vs 64µs measured; see docs/results/az-jax-perf.md). Training stays in `mlp.py` (manual
float64 backprop, off the hot path — once per iteration, not per leaf); this module is inference
only.

The jitted function takes the weights as ARGUMENTS (not a closure) so it does not recompile when
the net is retrained between iterations — XLA caches one compiled kernel per (shape, dtype)
signature, which is constant across the whole run. `MlpJaxForward` wraps a `ValueMLP`, holds its
weights as device arrays at the chosen precision, and exposes a numpy-in / numpy-out
`predict_both` matching the `ValueMLP` signature so it is a drop-in for the search.

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

_JDTYPE = jnp.float32 if np.dtype(DTYPE) == np.dtype(np.float32) else jnp.float64


@jax.jit
def _forward_both(x, lm, W1, b1, W2, b2, Wv, bv, Wp, bp, ym, ys):
    """Trunk in→H→ReLU→H→ReLU, value head (de-standardized) + masked-softmax policy.
    `x`: (in,) or (B,in); `lm`: matching legal mask. Returns (v, p) with the leading shape."""
    a1 = jnp.maximum(x @ W1 + b1, 0.0)
    a2 = jnp.maximum(a1 @ W2 + b2, 0.0)
    v = (a2 @ Wv + bv).ravel() * ys + ym
    logits = a2 @ Wp + bp
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
    search leaf eval. Holds the net's weights as device arrays at `DTYPE`; rebuild via `refresh()`
    if the underlying net is retrained (the ExIt loop builds a fresh search per iteration, so a
    fresh wrapper per iteration is the natural seam — but `refresh()` is provided for reuse)."""

    def __init__(self, net):
        if net.n_actions is None:
            raise ValueError("MlpJaxForward needs a net with a policy head (n_actions set)")
        self.net = net
        self.refresh()

    def refresh(self):
        net = self.net
        d = _JDTYPE
        self.W1 = jnp.asarray(net.W1, dtype=d); self.b1 = jnp.asarray(net.b1, dtype=d)
        self.W2 = jnp.asarray(net.W2, dtype=d); self.b2 = jnp.asarray(net.b2, dtype=d)
        self.Wv = jnp.asarray(net.Wv, dtype=d); self.bv = jnp.asarray(net.bv, dtype=d)
        self.Wp = jnp.asarray(net.Wp, dtype=d); self.bp = jnp.asarray(net.bp, dtype=d)
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
        v, p = _forward_both(x, lm, self.W1, self.b1, self.W2, self.b2,
                             self.Wv, self.bv, self.Wp, self.bp, self.ym, self.ys)
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
