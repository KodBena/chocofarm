#!/usr/bin/env python3
"""
chocofarm AZ — a pure-numpy 2-layer MLP with value + policy heads (design §3, §8).

No torch (design F7 / §8: "try numpy-MLP first", zero new deps). The architecture is the
CPU-shaped baseline: trunk `in → H → ReLU → H → ReLU`, then a linear scalar value head over
STANDARDIZED targets (design §3: returns are not in [-1,1], so standardize + linear, no tanh)
and an OPTIONAL policy head (logits over the fixed `n_actions`-slot action space, design §3).

The value head alone served the E-DECIDE probe (design §1, §9 — the F4 calibration cure rides
on the value). The full Gumbel ExIt loop (design §5/§6) needs BOTH heads: the policy head is the
search prior `P(s,a)` and the apprentice target, the value head is the leaf evaluation. The
policy head serves masked-softmax inference here; the AlphaZero training loss `CE(masked) + MSE +
L2` (Silver et al. 2017, design §6) lives with the JaxTrainer (see below).

This net is INFERENCE-ONLY: manual forward, masked softmax, a float32 inference fast path, and npz
save/load. TRAINING moved to JAX/optax autodiff in `mlp_jax_train.JaxTrainer` (which reads and
writes these weights); the hand-rolled Adam + manual backprop that used to live here are gone.
L2 (on weights, not biases) is now applied by the JaxTrainer's loss, reproducing the prior scope.

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
    E-DECIDE probe leaves it None. Inference-only — the JaxTrainer owns the optimizer.
    """

    def __init__(self, in_dim, hidden=256, n_actions=None, seed=0,
                 y_mean=0.0, y_std=1.0, residual=False):
        rng = np.random.default_rng(seed)
        self.in_dim = int(in_dim)
        self.H = int(hidden)
        self.n_actions = int(n_actions) if n_actions is not None else None
        # trunk
        self.W1 = _he_init(rng, in_dim, hidden); self.b1 = np.zeros(hidden)
        self.W2 = _he_init(rng, hidden, hidden); self.b2 = np.zeros(hidden)
        # OPTIONAL residual block between the trunk output and the two heads (toggle, default OFF →
        # the net is numerically identical to the pre-residual net, so `residual` is a clean
        # ablation axis). Block: z = ReLU(h @ Wr1 + br1); z = z @ Wr2 + br2; out = h + z
        # (pre-activation skip, NO outer ReLU — firewall A/B found this the best CE variant).
        # Wr1/Wr2 are H×H so the skip dimension matches; the heads read `out` instead of `h`.
        # The rng is consumed AFTER the heads below precisely so that the trunk + heads draw the
        # same numbers whether or not the block is built — `residual=False` is bit-identical.
        self.residual = bool(residual)
        # value head (linear scalar)
        self.Wv = _he_init(rng, hidden, 1) * 0.1; self.bv = np.zeros(1)
        # optional policy head
        if self.n_actions is not None:
            self.Wp = _he_init(rng, hidden, self.n_actions) * 0.1
            self.bp = np.zeros(self.n_actions)
        # residual-block params (drawn last so they don't perturb the trunk/head rng stream)
        if self.residual:
            self.Wr1 = _he_init(rng, hidden, hidden); self.br1 = np.zeros(hidden)
            self.Wr2 = _he_init(rng, hidden, hidden); self.br2 = np.zeros(hidden)
        # target standardization (stored, applied at train, inverted at predict)
        self.y_mean = float(y_mean)
        self.y_std = float(y_std) if y_std > 1e-8 else 1.0
        # inference-precision cache: when DTYPE is float32, predict_both serves the forward from
        # float32 copies of the weights (BLAS sgemm is ~1.8× the f64 path at single-row dispatch,
        # and float32 is the parametric hot-path precision). Training itself stays float64.
        #
        # CACHE COHERENCE — the invariant over ALL writers, not a per-writer gate. The float32
        # cache must never serve weights that no longer match the float64 source. Rather than ask
        # every weight-mutating site (load(), warm-start copy, the JaxTrainer's write-back, any
        # future EMA/restart) to remember to invalidate — the fragile per-producer shape an
        # out-of-frame audit caught as a latent stale-serve bug — the cache validates against the
        # *identity* of the source arrays at every read: it stores the source array objects and
        # compares them by `is`. Every current weight writer REBINDS (`self.Wx = ...` in
        # load/warm-start, `setattr` in the JaxTrainer write-back), so a fresh object id forces a
        # rebuild with no cooperation from the writer. This closes the audit's hazard (a rebinding
        # writer that forgot to invalidate). `_w_revision` is retained as the explicit signal an
        # IN-PLACE weight writer would bump (the numpy Adam step was the one such writer; it moved
        # to the JaxTrainer, which rebinds, so nothing bumps it today — it stays at 0).
        self._w_revision = 0
        self._f32_cache = None
        self._f32_cache_sig = None

    # ---- parameter registry (drives L2; consumed by save/load + the JaxTrainer) ----
    def _params(self):
        p = {"W1": self.W1, "b1": self.b1, "W2": self.W2, "b2": self.b2,
             "Wv": self.Wv, "bv": self.bv}
        if self.residual:
            p["Wr1"] = self.Wr1; p["br1"] = self.br1
            p["Wr2"] = self.Wr2; p["br2"] = self.br2
        if self.n_actions is not None:
            p["Wp"] = self.Wp; p["bp"] = self.bp
        return p

    def _is_weight(self, name):
        return name.startswith("W")  # L2 on weight matrices only, not biases

    # ---- forward ----
    def _forward(self, X):
        """Returns (cache, value_standardized, policy_logits_or_None).

        With `residual` ON, a residual block sits between the trunk output `a2` and the heads:
            zr1 = a2 @ Wr1 + br1;  ar1 = ReLU(zr1)
            zr2 = ar1 @ Wr2 + br2
            head_in = a2 + zr2                  # pre-activation skip, no outer ReLU (Wr*: H×H)
        The heads read `head_in`. With `residual` OFF, `head_in is a2` and the math is the
        pre-residual net exactly. `res_cache` is always None — it packaged the residual-block
        intermediates for the (removed) manual backward; inference reads only `v_std`/`logits`."""
        z1 = X @ self.W1 + self.b1
        a1 = np.maximum(z1, 0.0)
        z2 = a1 @ self.W2 + self.b2
        a2 = np.maximum(z2, 0.0)
        if self.residual:
            zr1 = a2 @ self.Wr1 + self.br1
            ar1 = np.maximum(zr1, 0.0)
            zr2 = ar1 @ self.Wr2 + self.br2
            head_in = a2 + zr2                     # pre-activation skip, NO outer ReLU (firewall A/B: best CE)
        else:
            head_in = a2
        v_std = (head_in @ self.Wv + self.bv).ravel()
        logits = (head_in @ self.Wp + self.bp) if self.n_actions is not None else None
        cache = (X, z1, a1, z2, a2, head_in, None)
        return cache, v_std, logits

    # Training moved to JAX/optax autodiff in `mlp_jax_train.JaxTrainer`: `jax.value_and_grad`
    # makes the gradient correct-by-construction (no hand-derived residual backward, no finite-diff
    # gradient-check), and an architecture change is a one-line forward edit with no backward to
    # re-derive. The numpy net below is INFERENCE-ONLY (forward + masked softmax + the float32
    # inference cache) plus npz serialization; the exit_loop / train_value harnesses train via the
    # JaxTrainer. The numpy `_forward` survives because the numpy↔jax-jit FLOAT32 EQUIVALENCE TEST
    # (tests/test_jax_equivalence.py) pins it as the reference. Do not re-add a numpy training
    # path — use `JaxTrainer`.

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
                and (not self.residual
                     or (c["_Wr1"] is self.Wr1 and c["_br1"] is self.br1
                         and c["_Wr2"] is self.Wr2 and c["_br2"] is self.br2))
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
        if self.residual:
            c["Wr1"] = self.Wr1.astype(np.float32); c["br1"] = self.br1.astype(np.float32)
            c["Wr2"] = self.Wr2.astype(np.float32); c["br2"] = self.br2.astype(np.float32)
            c["_Wr1"] = self.Wr1; c["_br1"] = self.br1
            c["_Wr2"] = self.Wr2; c["_br2"] = self.br2
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
        if self.residual:
            ar1 = np.maximum(a2 @ c["Wr1"] + c["br1"], np.float32(0.0))
            zr2 = ar1 @ c["Wr2"] + c["br2"]
            head_in = a2 + zr2                            # NO outer ReLU (matches _forward)
        else:
            head_in = a2
        v = (head_in @ c["Wv"] + c["bv"]).ravel() * c["ys"] + c["ym"]
        logits = head_in @ c["Wp"] + c["bp"]
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

    def set_value_scale(self, y_mean, y_std):
        """Re-pin the value-target standardization (the ExIt loop sets it from the replay
        buffer's running target statistics; design §3 standardize-targets)."""
        self.y_mean = float(y_mean)
        self.y_std = float(y_std) if y_std > 1e-8 else 1.0

    # ---- persistence (npz) ----
    def save(self, path):
        d = {k: v for k, v in self._params().items()}
        # _meta carries a 4th field — the residual flag (0/1). Old npz files have only 3 fields;
        # load() handles that length explicitly (treats absent → residual OFF).
        d["_meta"] = np.array([self.in_dim, self.H,
                               self.n_actions if self.n_actions is not None else -1,
                               1 if self.residual else 0],
                              dtype=np.int64)
        d["_yscale"] = np.array([self.y_mean, self.y_std], dtype=np.float64)
        np.savez(path, **d)

    @classmethod
    def load(cls, path):
        z = np.load(path, allow_pickle=False)
        meta = [int(x) for x in z["_meta"]]
        in_dim, H, na = meta[0], meta[1], meta[2]
        # 4th meta field is the residual flag; absent in pre-residual npz files (length 3).
        meta_residual = bool(meta[3]) if len(meta) >= 4 else False
        y_mean, y_std = (float(x) for x in z["_yscale"])
        n_actions = None if na < 0 else na
        # A net saved WITH the block carries the Wr*/br* arrays; one saved WITHOUT does not. Build
        # the net with the block only if BOTH the flag says so AND the params are present — mirrors
        # the --init-weights dim-mismatch handling (fail informative, not opaque): an old npz loaded
        # against a residual-meta mismatch keeps the block OFF (random/absent) with a clear log line
        # rather than crashing deep in the first forward (ADR-0002).
        have_res_params = all(k in z.files for k in ("Wr1", "br1", "Wr2", "br2"))
        residual = meta_residual and have_res_params
        if meta_residual and not have_res_params:
            print(f"[ValueMLP.load] {path}: _meta says residual=ON but block params "
                  f"(Wr1/br1/Wr2/br2) are absent — loading with residual OFF", flush=True)
        net = cls(in_dim, hidden=H, n_actions=n_actions, y_mean=y_mean, y_std=y_std,
                  residual=residual)
        net.W1, net.b1 = z["W1"], z["b1"]
        net.W2, net.b2 = z["W2"], z["b2"]
        net.Wv, net.bv = z["Wv"], z["bv"]
        if residual:
            # validate the block-param shapes at setup (ADR-0002: fail informative HERE, not deep
            # in the first forward as a raw matmul ValueError). Both Wr* are H×H, both br* are (H,).
            for k, want in (("Wr1", (H, H)), ("Wr2", (H, H)), ("br1", (H,)), ("br2", (H,))):
                if z[k].shape != want:
                    raise ValueError(
                        f"ValueMLP.load {path}: residual param {k} has shape {z[k].shape}, "
                        f"expected {want} (hidden={H}) — corrupt/incompatible npz")
            net.Wr1, net.br1 = z["Wr1"], z["br1"]
            net.Wr2, net.br2 = z["Wr2"], z["br2"]
        if n_actions is not None:
            net.Wp, net.bp = z["Wp"], z["bp"]
        return net
