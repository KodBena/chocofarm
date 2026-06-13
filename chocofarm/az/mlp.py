#!/usr/bin/env python3
"""
chocofarm AZ — a pure-numpy 2-layer MLP with a value head (design §3, §8).

No torch (design F7 / §8: "try numpy-MLP first", zero new deps). The architecture is the
CPU-shaped baseline: trunk `in → H → ReLU → H → ReLU`, then a linear scalar value head over
STANDARDIZED targets (design §3: returns are not in [-1,1], so standardize + linear, no tanh).
A policy head is OPTIONAL and off by default — for E-DECIDE the value head is the load-bearing
one (design §1, §9: the F4 calibration cure rides on the value, not the policy).

Manual forward, manual backprop, manual Adam, L2 on weights (not biases). Save/load is npz.

Standardization: the value head regresses the z-scored target `(y - mu) / sigma`; `mu`, `sigma`
are stored in the npz so inference de-standardizes back to the λ-penalized return scale. This
keeps the value loss O(1) regardless of the return magnitude (design §3).
"""
from __future__ import annotations

import numpy as np


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

    def predict_value(self, X):
        """De-standardized value (the λ-penalized return scale). `X` may be 1-D or 2-D."""
        single = (X.ndim == 1)
        if single:
            X = X[None, :]
        _, v_std, _ = self._forward(X)
        v = v_std * self.y_std + self.y_mean
        return float(v[0]) if single else v

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
