#!/usr/bin/env python3
"""
chocofarm AZ — JAX/optax TRAINING for the value+policy MLP (autodiff replaces manual backprop).

The companion to `mlp.py` (numpy float32 inference) and `mlp_jax.py` (the rejected jax-jit
inference experiment). The split is deliberate and load-bearing:

  * INFERENCE stays numpy float32 (`mlp.ValueMLP.predict_both` / `predict_value`). jax-jit batch-1
    inference is ~10× slower than numpy at single-row dispatch (the dispatch tax; measured, see
    docs/results/az-jax-perf.md and the equivalence-test bench below). The search calls the leaf
    forward one belief at a time, so numpy wins decisively there.
  * TRAINING moves to JAX here. The manual `_residual_backward` + hand-rolled Adam in `mlp.py` were
    fragile (a whole investigation went into the hand-derived residual backward + finite-diff
    gradient-check). `jax.value_and_grad` makes the gradient correct-by-construction and an
    architecture change (a second residual block, a different head) a one-line forward edit with no
    backward to re-derive.

The two forward implementations (numpy `_forward` and `_forward_jax` here) are a duplication risk.
That risk is bounded by THE EQUIVALENCE TEST (`tests/test_jax_equivalence.py`): it asserts the
numpy forward matches `jax.jit(_forward_jax)` to float32 precision for BOTH value and policy
logits, residual ON, batched and single-row. It compares against the JIT'd forward (not eager
jax) on purpose — XLA fuses/reorders, so jit numerics differ from eager, and the weights are
trained under the jit'd forward, so numpy inference must match *that* (not eager) or the search
would run a subtly different net than training optimized.

Params are a flat dict pytree keyed exactly like `ValueMLP._params()` (W1 b1 W2 b2 [Wr1 br1 Wr2
br2] Wv bv Wp bp) so weights are interchangeable with the numpy net — jax trains them, numpy reads
them. The forward mirrors `ValueMLP._forward` EXACTLY: trunk in→H→ReLU→H→ReLU, the no-outer-ReLU
pre-activation residual block `head_in = a2 + (ReLU(a2@Wr1+br1)@Wr2+br2)`, a linear scalar value
head over the STANDARDIZED target, and a policy head of n_actions logits.
"""
from __future__ import annotations

import os

# Match mlp_jax.py: keep XLA single-threaded (the loop / bench is core-pinned). Set before jax
# imports. Training runs once per iteration over a batch — off the per-leaf hot path — so this is
# about not fighting the taskset pin, not about training throughput.
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import jax
import jax.numpy as jnp
import optax

from chocofarm.az.dtypes import DTYPE
from chocofarm.az.forward import forward_core
from chocofarm.az.mlp import is_weight

# The equivalence safeguard is a FLOAT32 contract (the inference precision the search runs at and
# the precision the equivalence test pins). Train in float32 so the weights numpy inference reads
# are exactly the weights the jit'd forward optimized — no f64→f32 truncation gap between training
# and inference. (The old manual path trained in float64; that gap is precisely what the
# equivalence test would otherwise have to absorb. Training in f32 closes it.)
_JDTYPE = jnp.float32 if np.dtype(DTYPE) == np.dtype(np.float32) else jnp.float64


# ---------------------------------------------------------------------------
# Functional forward (params pytree -> (value_standardized, policy_logits))
# ---------------------------------------------------------------------------
def _forward_jax(params, X):
    """The jax forward = the ONE `forward.forward_core` evaluated under `jax.numpy` (audit R11).

    Returns (v_std, logits) with `v_std` shape (B,) and `logits` shape (B, n_actions). `params` is
    a flat dict: W1 b1 W2 b2 [Wr1 br1 Wr2 br2] Wv bv [Wp bp]. The residual block is applied iff
    "Wr1" is in `params` (the same toggle as the numpy net's `self.residual`); the policy head iff
    "Wp" is present (a value-only Stage-1 net has none → `logits` is None). It is the SAME function
    `forward_jax_jit` jit-compiles and `_az_loss`/`_value_loss` differentiate, so the weights are
    trained under exactly the forward numpy inference reads (the equivalence-test contract)."""
    return forward_core(params, X, jnp)


def _masked_softmax_jax(logits, legal_mask):
    """Mirror of `ValueMLP._masked_softmax`: softmax over legal slots only; illegal slots get
    exactly zero mass (masked with -1e30 in log-space, then zeroed). Numerically stable."""
    neg = jnp.asarray(-1e30, dtype=logits.dtype)
    legal = legal_mask > 0
    masked = jnp.where(legal, logits, neg)
    masked = masked - masked.max(axis=1, keepdims=True)
    e = jnp.exp(masked) * legal
    denom = e.sum(axis=1, keepdims=True)
    denom = jnp.where(denom > 0, denom, 1.0)
    return e / denom


# Public jit'd forward — THE reference the equivalence test compares numpy against. It is the same
# function the loss differentiates, so "numpy matches jit'd forward" certifies inference reads the
# net training optimized.
forward_jax_jit = jax.jit(_forward_jax)


def _l2_sumsq(params):
    """Σ‖W‖² over WEIGHT MATRICES only — the L2 scope is `mlp.is_weight` (audit R11: ONE definition,
    imported here, not a re-derived `name.startswith('W')`). The `l2` coefficient is applied by the
    loss (closed over at trace time as a Python float, so a `l2==0` short-circuit need not be a
    traced branch).

    NB this is COUPLED L2 (the penalty is in the loss, so its gradient flows through Adam's
    preconditioner) — NOT optax's decoupled `add_decayed_weights`. That is deliberate: it
    reproduces the numpy path's `g + l2·W` gradient EXACTLY (same coefficient, weights-only scope).
    The numpy Adam added `l2·W` to the raw gradient, which is the gradient of `0.5·l2·‖W‖²`; adding
    that penalty to the loss gives autodiff the identical contribution."""
    s = jnp.asarray(0.0, dtype=_JDTYPE)
    for name, arr in params.items():
        if is_weight(name):
            s = s + jnp.sum(arr * arr)
    return s


def _az_loss(params, X, target_pi, legal_mask, y_std_target, alpha, beta, l2):
    """The AlphaZero loss (design §6, Silver et al. 2017), the EXACT scalar the numpy train_step
    descends:

        L = alpha · CE(p_net, target_pi)  +  beta · MSE(v_std, y_std_target)  +  0.5·l2·||W||²

    where CE is over the masked softmax (illegal slots carry zero probability in both p and π′ so
    they contribute nothing and receive no gradient), MSE is the mean over the batch of
    (v_std − y_std_target)², and y_std_target is the PRE-standardized value target (the caller
    standardizes with the net's y_mean/y_std, matching the numpy path). Returns (loss, (ce, vmse))
    so `value_and_grad(..., has_aux=True)` reports the components for logging."""
    v_std, logits = _forward_jax(params, X)
    resid = v_std - y_std_target
    vmse = jnp.mean(resid ** 2)
    p = _masked_softmax_jax(logits, legal_mask)
    logp = jnp.where(p > 0, jnp.log(jnp.clip(p, 1e-12, 1.0)), 0.0)
    ce = -jnp.mean(jnp.sum(target_pi * logp, axis=1))
    loss = alpha * ce + beta * vmse + 0.5 * l2 * _l2_sumsq(params)
    return loss, (ce, vmse)


def _value_loss(params, X, y_std_target, l2):
    """Value-only loss (MSE on the standardized target + L2), the scalar `train_step_value`
    descends — used by `train_value.py`'s Stage-1 Gate (value head, no policy head)."""
    v_std, _ = _forward_jax(params, X)
    resid = v_std - y_std_target
    vmse = jnp.mean(resid ** 2)
    loss = vmse + 0.5 * l2 * _l2_sumsq(params)
    return loss, vmse


def _make_az_update(opt, l2):
    """Build the jit'd AZ optax step closing over the optax transformation `opt` and the L2
    coefficient `l2` (both compile-time constants for the run). `opt` (a GradientTransformation,
    tuple-of-functions) is not a jax type so it must be a closure; `l2` is closed over too so its
    `0.5*l2*||W||²` term carries no traced branch. jit'ing the whole value_and_grad + optax step is
    the point: XLA fuses the forward, the backward, and the Adam update into one compiled kernel.
    The weights are trained under THIS jit'd forward — which is why the equivalence test pins numpy
    against the jit'd (not eager) forward."""
    @jax.jit
    def _az_update(params, opt_state, X, target_pi, legal_mask, y_std_target, alpha, beta):
        (loss, (ce, vmse)), grads = jax.value_and_grad(_az_loss, has_aux=True)(
            params, X, target_pi, legal_mask, y_std_target, alpha, beta, l2)
        updates, opt_state = opt.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, ce, vmse
    return _az_update


def _make_value_update(opt, l2):
    """The value-only counterpart to `_make_az_update` (no policy head)."""
    @jax.jit
    def _value_update(params, opt_state, X, y_std_target):
        (loss, vmse), grads = jax.value_and_grad(_value_loss, has_aux=True)(
            params, X, y_std_target, l2)
        updates, opt_state = opt.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, vmse
    return _value_update


# ---------------------------------------------------------------------------
# Trainer — wraps a ValueMLP; optax-Adam over the jit'd forward; syncs jax→numpy
# ---------------------------------------------------------------------------
class JaxTrainer:
    """JAX/optax trainer bound to a `ValueMLP`. The net's params are the single source of truth:
    this trainer reads them into a jax pytree at construction, runs optax-Adam steps over the jit'd
    forward, and writes the updated weights BACK into the net after every step (so numpy inference
    reads the trained weights).

    Adam configuration matches the manual optimizer it replaces: betas (0.9, 0.999), eps 1e-8,
    COUPLED L2 on weight matrices only (the `0.5·l2·‖W‖²` penalty folded into the loss, so its
    gradient flows through Adam's preconditioner — exactly the numpy path's `g + l2·W`, NOT optax's
    decoupled `add_decayed_weights` which would also decay biases). `lr` and `l2` are
    passed per-step (the loop varies neither today, but the manual path took them per-step, so the
    signature is preserved). The optimizer state lives across steps — Adam's running moments need
    to persist, exactly as the numpy `self.m/self.v/self.t` did.

    Cache-coherence invariant (preserved): writing weights back REBINDS the net's arrays
    (`net.W1 = np.asarray(...)`), so the float32 inference cache's identity check (`c["_W1"] is
    self.W1`) sees fresh objects and rebuilds — no per-writer invalidation gate needed, the same
    invariant the numpy path relies on (see `ValueMLP._f32_weights`)."""

    def __init__(self, net, lr, l2=0.0, betas=(0.9, 0.999), eps=1e-8):
        self.net = net
        self.lr = float(lr)
        self.l2 = float(l2)
        self.has_policy = net.n_actions is not None
        b1, b2 = betas
        # optax Adam with decoupled L2 in the loss (NOT optax weight_decay, which decays ALL params
        # incl. biases — the numpy path decays weight matrices only, so we keep L2 in the loss and
        # leave optax as plain Adam). This makes the gradient exactly the numpy path's `g + l2*W`.
        self.opt = optax.adam(learning_rate=self.lr, b1=b1, b2=b2, eps=eps)
        self.params = self._read_params()
        self.opt_state = self.opt.init(self.params)
        # jit'd update steps closing over this instance's optax transformation + L2 coefficient
        self._az_update = _make_az_update(self.opt, self.l2)
        self._value_update = _make_value_update(self.opt, self.l2)

    def _read_params(self):
        """Read the net's numpy weights into a jax pytree (float32 = the training/inference
        precision). Keyed exactly like `ValueMLP._params()`."""
        return {k: jnp.asarray(v, dtype=_JDTYPE) for k, v in self.net._params().items()}

    def _write_params(self):
        """Write the jax params back into the net as numpy arrays — REBIND (new objects) so the
        f32 inference cache invalidates via its identity check. Stores at the net's float64 dtype
        for the params (the net's source-of-truth precision); the f32 inference cache re-casts.
        The arrays are np.float64 so predict_value (the float64 path) and load/save round-trip
        cleanly, and the f32 cache casts down for the hot path."""
        net = self.net
        for k, v in self.params.items():
            arr = np.asarray(v, dtype=np.float64)
            # preserve the original numpy shape (Wv/bv are (H,1)/(1,) etc.); jax keeps shape, so a
            # direct rebind is shape-correct. setattr by the registry key.
            setattr(net, k, arr)

    def sync_from_net(self):
        """Re-read the net's weights into the jax params (e.g. after a load/warm-start replaced
        them outside the trainer). Resets the optimizer state — the moments no longer correspond to
        the new weights, exactly as `_init_adam()` reset them in the numpy path on load."""
        self.params = self._read_params()
        self.opt_state = self.opt.init(self.params)

    def train_step(self, X, target_pi, legal_mask, target_v, alpha=1.0, beta=1.0):
        """One Adam step on the AZ loss. X: (B, in_dim); target_pi: (B, n_actions) prob rows;
        legal_mask: (B, n_actions) {0,1}; target_v: (B,) RAW value targets (standardized here with
        the net's y_mean/y_std, matching the numpy path). Returns (ce, vmse) floats for logging,
        and writes the updated weights back into the net."""
        if not self.has_policy:
            raise ValueError("net has no policy head (n_actions=None) — use train_step_value")
        ys = np.float32(self.net.y_std) if _JDTYPE == jnp.float32 else np.float64(self.net.y_std)
        ym = np.float32(self.net.y_mean) if _JDTYPE == jnp.float32 else np.float64(self.net.y_mean)
        X = jnp.asarray(X, dtype=_JDTYPE)
        target_pi = jnp.asarray(target_pi, dtype=_JDTYPE)
        legal_mask = jnp.asarray(legal_mask, dtype=_JDTYPE)
        y_std_target = (jnp.asarray(target_v, dtype=_JDTYPE) - ym) / ys
        self.params, self.opt_state, ce, vmse = self._az_update(
            self.params, self.opt_state, X, target_pi, legal_mask, y_std_target,
            jnp.asarray(alpha, _JDTYPE), jnp.asarray(beta, _JDTYPE))
        self._write_params()
        return float(ce), float(vmse)

    def train_step_value(self, X, target_v):
        """One Adam step on the value-only loss (for the no-policy Stage-1 net). Returns vmse."""
        ys = np.float32(self.net.y_std) if _JDTYPE == jnp.float32 else np.float64(self.net.y_std)
        ym = np.float32(self.net.y_mean) if _JDTYPE == jnp.float32 else np.float64(self.net.y_mean)
        X = jnp.asarray(X, dtype=_JDTYPE)
        y_std_target = (jnp.asarray(target_v, dtype=_JDTYPE) - ym) / ys
        self.params, self.opt_state, vmse = self._value_update(
            self.params, self.opt_state, X, y_std_target)
        self._write_params()
        return float(vmse)
