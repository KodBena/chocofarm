#!/usr/bin/env python3
"""
chocofarm AZ — a pure-numpy 2-layer MLP with value + policy heads (design §3, §8).

No torch (design F7 / §8: "try numpy-MLP first", zero new deps). The architecture is the
CPU-shaped baseline: trunk `in → H → ReLU → H → ReLU`, then a linear scalar value head over
STANDARDIZED targets (design §3: returns are not in [-1,1], so standardize + linear, no tanh)
and an OPTIONAL policy head (logits over the fixed `n_actions`-slot action space, design §3).

The value head alone served the E-DECIDE probe (design §1, §9 — the F4 calibration cure rides
on the value). The full Gumbel ExIt loop (design §5/§6) needs BOTH heads: the policy head is the
search prior `P(s,a)` and the apprentice target, the value head is the leaf evaluation. So the
policy head is now completed: masked softmax + a combined AlphaZero loss `CE(masked) + MSE +
L2` (Silver et al. 2017, design §6).

Manual forward, manual backprop, manual Adam, L2 on weights (not biases). Save/load is npz.

Standardization: the value head regresses the z-scored target `(y - mu) / sigma`; `mu`, `sigma`
are stored in the npz so inference de-standardizes back to the λ-penalized return scale. This
keeps the value loss O(1) regardless of the return magnitude (design §3).

The action↔slot mapping (fixed, env-derived; same scheme the loop and search use) is:
  slot 0 .. N-1            -> collect treasure i           = ("t", i)
  slot N .. N+nD-1         -> sense face (slot - N)        = ("d", slot - N)
  slot N+nD                -> TERMINATE
with n_actions = N + nD + 1 (on the honest env: 20 + 44 + 1 = 65). The legal mask comes from the
FeatureBuilder's `available` (collect) / `informative` (sense) indicators; TERMINATE is always
legal. This file only stores `n_actions` and operates on (logits, legal_mask) — the mapping
itself lives with the caller (`actions.py` / `gumbel_search.py`), kept in one place.
"""
from __future__ import annotations

import numpy as np

from chocofarm.az.dtypes import DTYPE, is_float32


def _he_init(rng, fan_in, fan_out):
    return rng.standard_normal((fan_in, fan_out)) * np.sqrt(2.0 / fan_in)


class ValueMLP:
    """Trunk (in→H→H, ReLU) + linear value head over standardized targets.

    Optional policy head (logits over `n_actions`) is built only if `n_actions` is given; the
    E-DECIDE probe leaves it None. Adam state is per-parameter; `l2` decays weight matrices.
    """

    def __init__(self, in_dim, hidden=256, n_actions=None, seed=0,
                 y_mean=0.0, y_std=1.0):
        rng = np.random.default_rng(seed)
        self.in_dim = int(in_dim)
        self.H = int(hidden)
        self.n_actions = int(n_actions) if n_actions is not None else None
        # trunk
        self.W1 = _he_init(rng, in_dim, hidden); self.b1 = np.zeros(hidden)
        self.W2 = _he_init(rng, hidden, hidden); self.b2 = np.zeros(hidden)
        # value head (linear scalar)
        self.Wv = _he_init(rng, hidden, 1) * 0.1; self.bv = np.zeros(1)
        # optional policy head
        if self.n_actions is not None:
            self.Wp = _he_init(rng, hidden, self.n_actions) * 0.1
            self.bp = np.zeros(self.n_actions)
        # target standardization (stored, applied at train, inverted at predict)
        self.y_mean = float(y_mean)
        self.y_std = float(y_std) if y_std > 1e-8 else 1.0
        # inference-precision cache: when DTYPE is float32, predict_both serves the forward from
        # float32 copies of the weights (BLAS sgemm is ~1.8× the f64 path at single-row dispatch,
        # and float32 is the parametric hot-path precision). Training itself stays float64.
        #
        # CACHE COHERENCE — the invariant over ALL writers, not a per-writer gate. The float32
        # cache must never serve weights that no longer match the float64 source. Rather than ask
        # every weight-mutating site (Adam step, load(), warm-start copy, any future EMA/restart)
        # to remember to invalidate — the fragile per-producer shape an out-of-frame audit caught
        # as a latent stale-serve bug — the cache validates against the *identity AND in-place
        # revision* of the source arrays at every read: it stores `id(W)` of each weight it was
        # built from and the float64 arrays' `.ctypes.data` (buffer address). Any writer that
        # REBINDS a weight (`self.W1 = ...` in load/warm-start) changes `id`; the Adam step
        # mutates IN PLACE so `id` is stable — that one in-place writer bumps `_w_revision`, the
        # single explicit signal still needed. A rebind needs no cooperation: a fresh id forces a
        # rebuild. This closes the audit's hazard (a rebinding writer that forgot to invalidate).
        self._w_revision = 0
        self._f32_cache = None
        self._f32_cache_sig = None
        self._init_adam()

    # ---- parameter registry (drives Adam + L2) ----
    def _params(self):
        p = {"W1": self.W1, "b1": self.b1, "W2": self.W2, "b2": self.b2,
             "Wv": self.Wv, "bv": self.bv}
        if self.n_actions is not None:
            p["Wp"] = self.Wp; p["bp"] = self.bp
        return p

    def _is_weight(self, name):
        return name.startswith("W")  # L2 on weight matrices only, not biases

    def _init_adam(self):
        self.m = {k: np.zeros_like(v) for k, v in self._params().items()}
        self.v = {k: np.zeros_like(v) for k, v in self._params().items()}
        self.t = 0

    # ---- forward ----
    def _forward(self, X):
        """Returns (cache, value_standardized, policy_logits_or_None)."""
        z1 = X @ self.W1 + self.b1
        a1 = np.maximum(z1, 0.0)
        z2 = a1 @ self.W2 + self.b2
        a2 = np.maximum(z2, 0.0)
        v_std = (a2 @ self.Wv + self.bv).ravel()
        logits = (a2 @ self.Wp + self.bp) if self.n_actions is not None else None
        cache = (X, z1, a1, z2, a2)
        return cache, v_std, logits

    # ---- float32 inference fast path (the parametric hot-path precision) ----
    def _f32_weights(self):
        """Float32 copies of the weights, cached and rebuilt whenever the source weights change —
        a REBIND (load/warm-start replace the array OBJECT), an in-place Adam mutation
        (`_w_revision` bump), or a y-scale change. Used by the float32 `predict_both` forward —
        sgemm in float32 is ~1.8× the float64 path at single-row dispatch, no per-dispatch overhead.

        Coherence is an INVARIANT over every writer, not a per-writer gate: the cache-validity
        check compares the source array OBJECTS by identity (`is`) — a rebind yields a new object
        and forces a rebuild with no cooperation from the writer — plus the in-place-mutation
        revision int and the y-scales. The check is inline `is`-comparisons (no per-call tuple
        allocation) because this runs on every leaf eval of an episode; the weights are frozen
        during generation so the hit path is the overwhelming common case."""
        c = self._f32_cache
        if (c is not None
                and c["_rev"] == self._w_revision
                and c["_W1"] is self.W1 and c["_b1"] is self.b1
                and c["_W2"] is self.W2 and c["_b2"] is self.b2
                and c["_Wv"] is self.Wv and c["_bv"] is self.bv
                and (self.n_actions is None
                     or (c["_Wp"] is self.Wp and c["_bp"] is self.bp))
                and c["ym"] == self.y_mean and c["ys"] == self.y_std):
            return c
        return self._rebuild_f32_cache()

    def _rebuild_f32_cache(self):
        c = {
            "W1": self.W1.astype(np.float32), "b1": self.b1.astype(np.float32),
            "W2": self.W2.astype(np.float32), "b2": self.b2.astype(np.float32),
            "Wv": self.Wv.astype(np.float32), "bv": self.bv.astype(np.float32),
            "ym": np.float32(self.y_mean), "ys": np.float32(self.y_std),
            # validity keys: the source array objects (compared by `is`) + the in-place revision
            "_rev": self._w_revision,
            "_W1": self.W1, "_b1": self.b1, "_W2": self.W2, "_b2": self.b2,
            "_Wv": self.Wv, "_bv": self.bv,
        }
        if self.n_actions is not None:
            c["Wp"] = self.Wp.astype(np.float32)
            c["bp"] = self.bp.astype(np.float32)
            c["_Wp"] = self.Wp; c["_bp"] = self.bp
        self._f32_cache = c
        return c

    def _predict_both_f32(self, X, legal_mask):
        """float32-numpy forward (trunk + both heads + masked softmax). Same shape contract as
        `predict_both`; the cast to float32 changes the last bits (acceptable — behavioral, not
        bit, equivalence is the bar)."""
        c = self._f32_weights()
        single = (X.ndim == 1)
        x = np.asarray(X, dtype=np.float32)
        lm = np.asarray(legal_mask, dtype=np.float32)
        if single:
            x = x[None, :]
            lm = lm[None, :]
        a1 = np.maximum(x @ c["W1"] + c["b1"], np.float32(0.0))
        a2 = np.maximum(a1 @ c["W2"] + c["b2"], np.float32(0.0))
        v = (a2 @ c["Wv"] + c["bv"]).ravel() * c["ys"] + c["ym"]
        logits = a2 @ c["Wp"] + c["bp"]
        legal = lm > 0
        masked = np.where(legal, logits, np.float32(-1e30))
        masked = masked - masked.max(axis=1, keepdims=True)
        e = np.exp(masked) * legal
        denom = e.sum(axis=1, keepdims=True)
        denom = np.where(denom > 0, denom, np.float32(1.0))
        p = e / denom
        if single:
            return float(v[0]), p[0]
        return v, p

    def predict_value(self, X):
        """De-standardized value (the λ-penalized return scale). `X` may be 1-D or 2-D."""
        single = (X.ndim == 1)
        if single:
            X = X[None, :]
        _, v_std, _ = self._forward(X)
        v = v_std * self.y_std + self.y_mean
        return float(v[0]) if single else v

    # ---- policy-head inference (masked softmax over the fixed slot space) ----
    @staticmethod
    def _masked_softmax(logits, legal_mask):
        """Softmax over the legal slots only. `logits`,`legal_mask`: (B, n_actions); mask is
        {0,1}. Illegal slots get probability exactly 0 (masked in log-space with -inf), legal
        slots normalize among themselves. Numerically stable (subtract per-row legal max)."""
        neg_inf = np.float64(-1e30)
        masked = np.where(legal_mask > 0, logits, neg_inf)
        masked = masked - masked.max(axis=1, keepdims=True)
        exp = np.exp(masked) * (legal_mask > 0)
        denom = exp.sum(axis=1, keepdims=True)
        denom = np.where(denom > 0, denom, 1.0)
        return exp / denom

    def predict_policy(self, X, legal_mask):
        """Masked-softmax policy P(s,·) over the fixed slot space. `X`: (in_dim,) or (B,in_dim);
        `legal_mask`: matching {0,1} over n_actions slots. Returns the same leading shape."""
        if self.n_actions is None:
            raise ValueError("net has no policy head (n_actions=None)")
        single = (X.ndim == 1)
        if single:
            X = X[None, :]
            legal_mask = legal_mask[None, :]
        _, _, logits = self._forward(X)
        p = self._masked_softmax(logits, legal_mask.astype(np.float64))
        return p[0] if single else p

    def predict_both(self, X, legal_mask):
        """One forward pass -> (de-standardized value, masked policy). The search's hot path:
        a single trunk evaluation feeds both the leaf value and the prior. `X` 1-D or 2-D.

        When the parametric DTYPE is float32 (the default), the float32-numpy fast path serves
        this — ~1.8× the float64 BLAS path at single-row dispatch, the precision the rest of the
        hot path runs at. Set CHOCO_AZ_DTYPE=float64 to take the float64 path below."""
        if self.n_actions is None:
            raise ValueError("net has no policy head (n_actions=None)")
        if is_float32():
            return self._predict_both_f32(X, legal_mask)
        single = (X.ndim == 1)
        if single:
            X = X[None, :]
            lm = legal_mask[None, :]
        else:
            lm = legal_mask
        _, v_std, logits = self._forward(X)
        v = v_std * self.y_std + self.y_mean
        p = self._masked_softmax(logits, lm.astype(np.float64))
        if single:
            return float(v[0]), p[0]
        return v, p

    # ---- one Adam step on the value loss (MSE on standardized target + L2) ----
    def train_step_value(self, X, y, lr, l2, betas=(0.9, 0.999), eps=1e-8):
        """X: (B, in_dim); y: (B,) RAW (un-standardized) targets. Returns standardized-MSE."""
        n = X.shape[0]
        y_std_target = (y - self.y_mean) / self.y_std
        cache, v_std, _ = self._forward(X)
        X_, z1, a1, z2, a2 = cache
        resid = v_std - y_std_target            # (B,)
        loss = float(np.mean(resid ** 2))

        # backprop (MSE = mean over batch of resid^2 → dL/dv_std = 2*resid/n)
        dv = (2.0 / n) * resid[:, None]         # (B,1)
        dWv = a2.T @ dv
        dbv = dv.sum(0)
        da2 = dv @ self.Wv.T
        dz2 = da2 * (z2 > 0)
        dW2 = a1.T @ dz2
        db2 = dz2.sum(0)
        da1 = dz2 @ self.W2.T
        dz1 = da1 * (z1 > 0)
        dW1 = X_.T @ dz1
        db1 = dz1.sum(0)

        grads = {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2, "Wv": dWv, "bv": dbv}
        # the value step does not touch the policy head (no gradient flows to it)
        self._adam_apply(grads, lr, l2, betas, eps)
        return loss

    # ---- combined AlphaZero loss step (CE(masked) + value MSE + L2) ----
    def train_step(self, X, target_pi, legal_mask, target_v, lr, l2,
                   alpha=1.0, beta=1.0, betas=(0.9, 0.999), eps=1e-8):
        """One Adam step on the AlphaZero loss (design §6, Silver et al. 2017):

            L = alpha · CE(p_net, target_pi)  +  beta · (v_net − v_target)²  +  l2 · ‖W‖²

        X: (B, in_dim). target_pi: (B, n_actions) — the Gumbel improved policy π′, a probability
        row over the LEGAL slots (zero on illegal). legal_mask: (B, n_actions) {0,1}. target_v:
        (B,) RAW (un-standardized) value targets. Returns (ce, value_std_mse) for logging.

        CE is over the masked softmax; illegal slots carry zero probability in both p and π′ so
        they contribute nothing to the cross-entropy and receive no gradient (the masked-softmax
        Jacobian leaves illegal logits untouched). This is the standard masked-policy gradient:
        for the softmax+CE pair the logit gradient is `(p − π′)` on legal slots, 0 on illegal."""
        if self.n_actions is None:
            raise ValueError("net has no policy head (n_actions=None) — cannot train policy")
        n = X.shape[0]
        lm = legal_mask.astype(np.float64)
        # --- forward (shared trunk) ---
        cache, v_std, logits = self._forward(X)
        X_, z1, a1, z2, a2 = cache

        # --- value branch (standardized MSE) ---
        y_std_target = (target_v - self.y_mean) / self.y_std
        resid = v_std - y_std_target                         # (B,)
        value_loss = float(np.mean(resid ** 2))
        dv = (2.0 / n) * (beta * resid)[:, None]             # (B,1), dL/dv_std

        # --- policy branch (masked CE) ---
        p = self._masked_softmax(logits, lm)                 # (B, n_actions)
        # CE = -mean_b Σ_a π'_ba log p_ba   (illegal slots zero in both → no contribution)
        logp = np.where(p > 0, np.log(np.clip(p, 1e-12, 1.0)), 0.0)
        ce = float(-np.mean(np.sum(target_pi * logp, axis=1)))
        # softmax+CE logit gradient: (p − π') on legal slots, masked to legal, mean over batch.
        dlogits = (alpha / n) * ((p - target_pi) * lm)       # (B, n_actions)

        # --- backprop both heads into the trunk ---
        dWv = a2.T @ dv
        dbv = dv.sum(0)
        dWp = a2.T @ dlogits
        dbp = dlogits.sum(0)
        da2 = dv @ self.Wv.T + dlogits @ self.Wp.T
        dz2 = da2 * (z2 > 0)
        dW2 = a1.T @ dz2
        db2 = dz2.sum(0)
        da1 = dz2 @ self.W2.T
        dz1 = da1 * (z1 > 0)
        dW1 = X_.T @ dz1
        db1 = dz1.sum(0)

        grads = {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2,
                 "Wv": dWv, "bv": dbv, "Wp": dWp, "bp": dbp}
        self._adam_apply(grads, lr, l2, betas, eps)
        return ce, value_loss

    def set_value_scale(self, y_mean, y_std):
        """Re-pin the value-target standardization (the ExIt loop sets it from the replay
        buffer's running target statistics; design §3 standardize-targets)."""
        self.y_mean = float(y_mean)
        self.y_std = float(y_std) if y_std > 1e-8 else 1.0

    def _adam_apply(self, grads, lr, l2, betas, eps):
        b1, b2 = betas
        self.t += 1
        params = self._params()
        for name, g in grads.items():
            if self._is_weight(name) and l2 > 0:
                g = g + l2 * params[name]
            self.m[name] = b1 * self.m[name] + (1 - b1) * g
            self.v[name] = b2 * self.v[name] + (1 - b2) * (g * g)
            mhat = self.m[name] / (1 - b1 ** self.t)
            vhat = self.v[name] / (1 - b2 ** self.t)
            params[name] -= lr * mhat / (np.sqrt(vhat) + eps)
        # Adam mutates the weight arrays IN PLACE (`params[name] -= ...`), so their id/buffer
        # address is unchanged — the identity check can't see it. Bump the explicit revision so
        # the float32 cache rebuilds. (Rebinding writers — load/warm-start — are caught by the
        # identity check and need no bump; this is the one in-place writer.)
        self._w_revision += 1

    # ---- persistence (npz) ----
    def save(self, path):
        d = {k: v for k, v in self._params().items()}
        d["_meta"] = np.array([self.in_dim, self.H,
                               self.n_actions if self.n_actions is not None else -1],
                              dtype=np.int64)
        d["_yscale"] = np.array([self.y_mean, self.y_std], dtype=np.float64)
        np.savez(path, **d)

    @classmethod
    def load(cls, path):
        z = np.load(path, allow_pickle=False)
        in_dim, H, na = (int(x) for x in z["_meta"])
        y_mean, y_std = (float(x) for x in z["_yscale"])
        n_actions = None if na < 0 else na
        net = cls(in_dim, hidden=H, n_actions=n_actions, y_mean=y_mean, y_std=y_std)
        net.W1, net.b1 = z["W1"], z["b1"]
        net.W2, net.b2 = z["W2"], z["b2"]
        net.Wv, net.bv = z["Wv"], z["bv"]
        if n_actions is not None:
            net.Wp, net.bp = z["Wp"], z["bp"]
        net._init_adam()
        return net
