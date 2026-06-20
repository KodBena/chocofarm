#!/usr/bin/env python3
"""
cpp/stage_a/control_lab/methods/offline_awr.py — an ADVANTAGE-WEIGHTED-REGRESSION (AWR) OFFLINE-RL
issue-gate (REINFORCEMENT-LEARNING family) candidate for the issue-gate control lab.

THE OFFLINE COUNTERPART to the online RL gates (reinforce / a2c). Where those learn ON-POLICY within a
single ~4 s run (Monte-Carlo policy gradient / actor-critic on the trajectory they themselves generate),
AWR learns OFFLINE from a FIXED corpus of trajectories logged by OTHER behavior policies — the 16 methods'
(s, a, r) streams in the depth>1 / S_min=1 convoy regime (postgres session lab-20260620-190846). It never
acts during training; the fitted policy is then DEPLOYED as a frozen runtime Controller. This is the
honest tool for "we already have a corpus, learn the best gate from it" — off-policy, multi-behavior.

WHY AWR FOR THIS CORPUS (the convoy signal). In the S_min=1 regime a thread that already has work in flight
should be DENIED (held back) so its `ready` rows pile up and the next forced flush coalesces a fat batch;
allowing an active thread keeps inflight saturated and each message tiny (the convoy collapse — all_allow's
dps≈11, rows/fwd≈16). The corpus makes this separable: pooled over the behaviors, an ACTIVE (inflight>0)
sample that was DENIED earned ~2x the per-forward reward of one that was ALLOWED. AWR is precisely the
estimator that turns that into a policy: weight each observed (state, action) by the EXPONENTIATED ADVANTAGE
of its outcome and REGRESS the policy onto the high-weight actions, so the deny-when-active decisions
dominate the fit and the allow-when-active ones wash out — WITHOUT importance weights or bootstrapping
(the brittle parts of off-policy RL in a short, multi-behavior corpus). The classic AWR objective
(Peng et al. 2019), specialized to a per-thread Bernoulli gate:

    V_psi(phi)      <- regress to the per-sample return R                      (the value baseline)
    A               =  R - V_psi(phi)                                          (the advantage)
    w               =  clip( exp( A_std / temp ),  0,  w_max )                 (the AWR weight; A_std standardized)
    theta*          =  argmin_theta  - E[ w * log pi_theta(a_obs | phi) ]      (advantage-weighted policy regression)

R is the per-forward reward (forward_rows — the coalescing achieved; HIGHER IS BETTER), the SAME scalar the
online gates optimize; there is no cross-forward discounting (the lab's credit is per-forward, the reward
column IS forward_rows), so the Monte-Carlo return collapses to that forward's reward. The advantage is
STANDARDIZED (zero-mean / unit-std over the corpus) before the exp so `temp` is a scale-free knob and the
weights neither collapse to 0 nor blow up; w_max clips the residual right tail.

PER-THREAD SAMPLES, ACTIVE-ONLY (the credit mask, identical to reinforce/a2c). PARAMETER-SHARING: one shared
policy + one shared value net see EVERY per-thread row, so a forward yields up to T training samples (the
gate is homogeneous across threads). A thread with inflight==0 is force-allowed by the runner's DENY-ONLY
liveness floor (a deny there is an UNGATED no-op), so its observed action carries NO causal credit; those
samples are MASKED OUT of BOTH the value fit and the policy regression (training on a forced no-op injects
pure noise — the faithful handling the online gates also apply). Only active (inflight>0) per-thread samples
train the nets.

JAX/OPTAX FOR THE FIT, NUMPY FOR THE HOT DECISION (the lab-server reality, mirrored from reinforce/a2c). The
offline fit is full-batch jax/optax (adam) over the pooled corpus — off any latency path, it runs in the
trainer (offline_awr_train.py), not in the server. The DEPLOYED policy's per-forward act() is a NUMPY matvec
from a NUMPY MIRROR of the policy params (O(T*d), d=5, one tanh layer) — no jax on the hot tick at all, so it
cannot cold-compile, re-trace on a changing T, or contend the inference device. The runtime gate is
DETERMINISTIC (allow iff sigmoid(logit) >= 0.5 = argmax over {deny, allow}); the offline corpus already
supplies the exploration (multiple behavior policies), so the deployed policy exploits, it does not sample.
The same pure-numpy hot path every method in this package uses; only the (offline) fit is jax.

THE TWO CLASSES (the frozen adapter's two seams):
  * AWRRecipe — a TrainableRecipe: `fit(corpus) -> Controller`. The corpus is an AWRCorpus (pooled per-thread
    active samples: phi (N, d_in), observed action (N,), return (N,)); fit runs the value regression then the
    advantage-weighted policy regression and returns a ready AWRGate carrying the fitted numpy params. This is
    the offline-RL training entry point (the trainer builds the corpus from the trajectory blobs and calls it).
  * AWRGate — the runtime Controller: a frozen policy (numpy mirror) with the cheap deterministic act(). It is
    constructed EITHER from fitted params (recipe.fit) OR from a checkpoint .npz (the registered factory, so
    the harness's make_controller — which rejects a TrainableRecipe — gets a real Controller). The checkpoint
    path is read from the CHOCOFARM_AWR_CKPT env var (default the trainer's output), so deploying the trained
    policy is one env var + the registry name, no harness edit (P2 seam discipline).

Feature surface + wire subtlety (IDENTICAL to reinforce/a2c — train/serve parity is the whole point). The 5-d
phi[t] = [submit_pressure, ready_backlog_norm, inflight_saturation, coalesce_degree_inst, bias] with
D=ctx.d_ceiling, K=ctx.k_per_thread. coalesce_degree_inst first-differences the CUMULATIVE counters
msgs/leaves over SERVED threads only (an absent thread's sentinel-0 would fake a negative delta; an
un-baselined/quiet thread reads the neutral 1.0). The OFFLINE trainer reconstructs this EXACT phi from the
decoded trajectory columns with the identical served-diff logic, so the policy is fitted on, and deployed
against, the same feature the online gates use.

Live-T robustness + non-throwing hot path (identical to reinforce/a2c). Every per-thread array is grown to
the live T on demand at the top of act() (the lab server can serve a tid beyond reset-time n_threads and
calls reset() outside its lock), and defaulted reads keep a malformed/short frame safe (ADR-0002: the
watchdog owns loudness; the hot path stays well-defined and never throws). The numpy mirror is always present.

RL family: act() runs the frozen numpy policy (deterministic gate + the inflight==0 liveness override).
observe() is a no-op (offline — nothing is learned at runtime). reset() captures the trial geometry (T, D, K)
and re-sizes the served-diff baselines; the policy params are NOT touched by reset (they are the fitted, frozen
weights — the trial only re-zeros the per-thread coalescing baselines). metrics() exposes the deployed
policy's last-forward mean allow probability + allow fraction + the fit's recorded AWR diagnostics.

Run the unit gate pinned + bounded:
    PYTHONPATH=cpp/stage_a /home/bork/w/vdc/venvs/generic/bin/python -m pytest tests/test_method_offline_awr.py -q

Public Domain (The Unlicense).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np
import optax

from control_lab.adapter import REGISTRY, Family, Observation, TrialContext

# The fixed feature layout (bias last), IDENTICAL to reinforce/a2c — train/serve parity. The realized input
# dimension d is DERIVED from this tuple, never hardcoded (ADR-0002 single-source-of-truth). The order here is
# the order both _build_phi (runtime) and the offline trainer assemble the columns in.
_FEATURES: tuple[str, ...] = (
    "submit_pressure",
    "ready_backlog_norm",
    "inflight_saturation",
    "coalesce_degree_inst",
    "bias",
)
_D_IN = len(_FEATURES)  # 5; the input dimension shared by the policy and value nets, derived from the layout

# The default checkpoint the registered factory loads (the trainer's output). Overridable via the
# CHOCOFARM_AWR_CKPT env var so a freshly-trained policy is deployed with one env var + the registry name
# (no harness edit — P2 seam discipline). The directory is under ~/w/vdc (NEVER /tmp; experiment output).
DEFAULT_CKPT = os.path.join(
    os.path.expanduser("~"), "w", "vdc", "chocobo", "runs", "control_lab", "offline-rl", "awr_policy.npz"
)


# ============================================================================================
# The pooled offline corpus (the TrainableRecipe input) — per-thread ACTIVE samples.
# ============================================================================================
@dataclass(frozen=True)
class AWRCorpus:
    """The pooled, per-thread, ACTIVE-only offline training set the AWR recipe fits. Each row is one
    (state, observed-action, return) sample from some behavior policy's trajectory, restricted to threads
    that actually acted (inflight>0 — a forced no-op carries no credit). The trainer (offline_awr_train.py)
    builds this from the decoded trajectory blobs, reconstructing phi with the SAME served-diff coalescing
    the runtime act() uses (feature parity).

      phi    : (N, d_in) float32 — the per-thread feature rows (the _FEATURES layout, bias last)
      action : (N,)      float32 — the OBSERVED gate (0 deny / 1 allow) the behavior policy emitted
      ret    : (N,)      float32 — the per-sample return (the forward's reward = forward_rows)
    """
    phi: np.ndarray
    action: np.ndarray
    ret: np.ndarray

    def __post_init__(self) -> None:
        # fail loud (ADR-0002): a malformed corpus is a construction error, surfaced before any fit work.
        if self.phi.ndim != 2 or self.phi.shape[1] != _D_IN:
            raise ValueError(f"AWRCorpus: phi must be (N, {_D_IN}), got {self.phi.shape}")
        n = self.phi.shape[0]
        if self.action.shape != (n,) or self.ret.shape != (n,):
            raise ValueError(f"AWRCorpus: action/ret must be (N,)={ (n,) }, got "
                             f"{self.action.shape}/{self.ret.shape}")
        if n == 0:
            raise ValueError("AWRCorpus: empty corpus (no active per-thread samples to fit)")


# ============================================================================================
# The tiny shared net (policy logit head OR value head — same shape) — jax, fit-only.
# ============================================================================================
def _init_head(key: "jax.Array", hidden: int, out_bias: float) -> dict[str, "jax.Array"]:
    """Initialize ONE tiny scalar-output head pytree (the policy logit head OR the value head — identical
    shape). Linear (hidden==0): weight vector `w` (d_in,) + scalar bias `b`. One hidden tanh layer (hidden>0):
    Glorot-ish small-random W1/b1 (*0.1) into the hidden layer and a small-random read-out (W2 *0.1) + bias.
    The bias is `out_bias` (the policy head's allow-lean cold start; 0 for the value head)."""
    if hidden <= 0:
        return {"w": jnp.zeros((_D_IN,), dtype=jnp.float32),
                "b": jnp.asarray(out_bias, dtype=jnp.float32)}
    k1, k2 = jax.random.split(key)
    w1 = jax.random.normal(k1, (_D_IN, hidden), dtype=jnp.float32) * 0.1
    b1 = jnp.zeros((hidden,), dtype=jnp.float32)
    w2 = jax.random.normal(k2, (hidden,), dtype=jnp.float32) * 0.1
    b2 = jnp.asarray(out_bias, dtype=jnp.float32)
    return {"w1": w1, "b1": b1, "w2": w2, "b2": b2}


def _head(params: dict[str, "jax.Array"], phi: "jax.Array", hidden: int) -> "jax.Array":
    """Scalar-per-row read-out of a tiny head on phi (N, d_in). Linear: phi @ w + b. Hidden: one tanh layer
    then a linear read-out. Returns (N,) — the policy's allow LOGIT or the value V, by which params it is
    called with. Pure; used in the (jax) fit only — the runtime forward is the numpy twin below."""
    if hidden <= 0:
        return phi @ params["w"] + params["b"]
    h = jnp.tanh(phi @ params["w1"] + params["b1"])  # (N, hidden)
    return h @ params["w2"] + params["b2"]           # (N,)


# ============================================================================================
# AWRRecipe — the TrainableRecipe: offline fit (value regression -> advantage -> weighted policy regression).
# ============================================================================================
class AWRRecipe:
    """The OFFLINE-RL training entry point (a TrainableRecipe): `fit(corpus) -> AWRGate`. Runs the two AWR
    stages with jax/optax (adam) full-batch over the pooled per-thread active corpus:

      1) ACTION-CONDITIONAL VALUE BASELINE (a Q-baseline — the load-bearing choice for THIS corpus). Fit a
         tiny Q_psi(phi, a) by MSE regression to the per-sample return R, with the binary action a appended as
         a feature (value_steps adam steps). The advantage is then the proper A = Q_psi(phi, a) - V_psi(phi)
         with V_psi(phi) = mean over the two arms = (Q(phi,0) + Q(phi,1)) / 2 (the behavior-marginalized
         value). This is the DEFINITION of advantage (A = Q - V), and it is action-conditional: it isolates how
         much THIS thread's chosen action beat the alternative IN THE SAME STATE.

         WHY a state-only V FAILS here (the empirically-found trap, ADR-0009). The per-forward reward is SHARED
         across all T threads, and in the convoy regime ~99.6% of active per-thread samples sit at the SAME
         feature value (inflight_saturation in [0.8, 1.0] — inflight pinned near D). So a state-only V_psi(phi)
         is near-CONSTANT (it cannot separate states that don't differ), making A = R - V_psi(phi) track the
         FORWARD'S reward, not the ACTION'S contribution — the exp-weighting then upweights high-reward forwards
         indiscriminately and the policy regresses onto the behavior mixture's marginal allow rate (a measured
         null result: learned allow_frac == observed allow_frac, no convoy-taming learned). The Q-baseline fixes
         exactly this: with phi ~constant, Q(phi, deny) ~ 43 and Q(phi, allow) ~ 21, so A|deny ~ +14 and
         A|allow ~ -7 — the action separation the state-only baseline could not see.
      2) ADVANTAGE-WEIGHTED POLICY REGRESSION. Compute A = Q(phi,a) - V(phi), STANDARDIZE it (zero-mean /
         unit-std over the corpus so `temp` is scale-free), form the AWR weight w = clip(exp(A_std / temp), 0,
         w_max), and fit the policy pi_theta by MINIMIZING the advantage-weighted binary cross-entropy between
         pi and the OBSERVED action (policy_steps adam steps). High-weight (high-advantage = deny-when-active)
         samples dominate the weight mass, so the policy regresses onto the convoy-taming decisions.

    Returns an AWRGate carrying the fitted numpy params (the deployable Controller). Knobs: hidden width,
    temp, w_max, the two step counts, the two learning rates, the policy cold-start allow logit, and the
    seed. fit() also stashes the AWR diagnostics (mean/quantiles of A and w, the learned active allow
    fraction) onto the returned gate's metrics for the trainer to report + verify."""

    name = "offline_awr"

    def __init__(
        self,
        hidden: int = 8,
        temp: float = 0.5,
        w_max: float = 20.0,
        value_steps: int = 3000,
        policy_steps: int = 3000,
        lr_value: float = 0.01,
        lr_policy: float = 0.01,
        init_allow_logit: float = 0.0,
        seed: int = 0,
    ) -> None:
        # fail loud (ADR-0002): degenerate hyperparameters are a CONSTRUCTION error, surfaced at build time.
        if hidden < 0:
            raise ValueError(f"AWRRecipe: hidden must be >= 0 (0 = linear), got {hidden}")
        if temp <= 0.0:
            raise ValueError(f"AWRRecipe: temp must be > 0, got {temp}")
        if w_max <= 0.0:
            raise ValueError(f"AWRRecipe: w_max must be > 0, got {w_max}")
        if value_steps < 1 or policy_steps < 1:
            raise ValueError(f"AWRRecipe: step counts must be >= 1, got {value_steps}/{policy_steps}")
        if lr_value <= 0.0 or lr_policy <= 0.0:
            raise ValueError(f"AWRRecipe: learning rates must be > 0, got {lr_value}/{lr_policy}")
        if not np.isfinite(init_allow_logit):
            raise ValueError(f"AWRRecipe: init_allow_logit must be finite, got {init_allow_logit}")
        self._hidden = int(hidden)
        self._temp = float(temp)
        self._w_max = float(w_max)
        self._value_steps = int(value_steps)
        self._policy_steps = int(policy_steps)
        self._lr_v = float(lr_value)
        self._lr_p = float(lr_policy)
        self._init_logit = float(init_allow_logit)
        self._seed = int(seed)
        # the per-fit training curves (value MSE, policy weighted-BCE) the trainer logs; populated by fit().
        self.curves: dict[str, list[float]] = {"value_mse": [], "policy_wbce": []}

    def fit(self, data: AWRCorpus) -> "AWRGate":  # type: ignore[override]
        """Run the two-stage AWR fit and return a deployable AWRGate. data is an AWRCorpus (pooled per-thread
        active samples). The whole fit is full-batch jax/optax — off any latency path (it runs in the trainer,
        not the server)."""
        if not isinstance(data, AWRCorpus):
            raise TypeError(f"AWRRecipe.fit: data must be an AWRCorpus, got {type(data).__name__}")
        phi = jnp.asarray(data.phi, dtype=jnp.float32)        # (N, d_in)
        act = jnp.asarray(data.action, dtype=jnp.float32)     # (N,)
        ret = jnp.asarray(data.ret, dtype=jnp.float32)        # (N,)
        hidden = self._hidden
        key = jax.random.PRNGKey(self._seed)
        kv, kp = jax.random.split(key)

        # ---------------- stage 1: action-conditional Q-baseline (MSE regression Q(phi, a) -> R) -------
        # Append the binary action as a feature so the value net is a Q-FUNCTION Q(phi, a), not a state-only
        # V(phi). This is the load-bearing fix (see the class docstring): with phi near-constant on active
        # samples, a state-only V cannot separate actions, so the advantage must come from Q(phi,a) - V(phi).
        # The Q-net input is the d_in feature row with the action column appended -> a (d_in+1)-wide head.
        phi_q0 = jnp.concatenate([phi, jnp.zeros((phi.shape[0], 1), dtype=jnp.float32)], axis=1)  # a=deny
        phi_q1 = jnp.concatenate([phi, jnp.ones((phi.shape[0], 1), dtype=jnp.float32)], axis=1)   # a=allow
        phi_qa = jnp.where(act[:, None] > 0.5, phi_q1, phi_q0)   # the OBSERVED (phi, a) Q-input

        def _q_init(key: "jax.Array", out_bias: float) -> dict[str, "jax.Array"]:
            """A Q-head of the same architecture but with a (d_in+1)-wide input (phi + action column)."""
            if hidden <= 0:
                return {"w": jnp.zeros((_D_IN + 1,), dtype=jnp.float32),
                        "b": jnp.asarray(out_bias, dtype=jnp.float32)}
            k1, k2 = jax.random.split(key)
            return {"w1": jax.random.normal(k1, (_D_IN + 1, hidden), dtype=jnp.float32) * 0.1,
                    "b1": jnp.zeros((hidden,), dtype=jnp.float32),
                    "w2": jax.random.normal(k2, (hidden,), dtype=jnp.float32) * 0.1,
                    "b2": jnp.asarray(out_bias, dtype=jnp.float32)}

        q_params = _q_init(kv, out_bias=float(np.mean(data.ret)))  # warm the Q bias at mean R
        tx_v = optax.adam(self._lr_v)
        v_opt = tx_v.init(q_params)

        def _q_loss(p: dict[str, "jax.Array"]) -> "jax.Array":
            q = _head(p, phi_qa, hidden)                      # Q(phi, a_obs) over the d_in+1 input
            return jnp.mean((q - ret) ** 2)

        @jax.jit
        def _value_step(p: dict[str, "jax.Array"], o: optax.OptState):
            loss, g = jax.value_and_grad(_q_loss)(p)
            upd, o2 = tx_v.update(g, o, p)
            return optax.apply_updates(p, upd), o2, loss

        self.curves["value_mse"] = []
        for i in range(self._value_steps):
            q_params, v_opt, vloss = _value_step(q_params, v_opt)
            if i % max(1, self._value_steps // 100) == 0 or i == self._value_steps - 1:
                self.curves["value_mse"].append(float(vloss))

        # ---------------- advantage A = Q(phi,a) - V(phi), V = mean over arms (behavior-marginalized) -----
        q_deny = _head(q_params, phi_q0, hidden)             # (N,) Q(phi, deny)
        q_allow = _head(q_params, phi_q1, hidden)            # (N,) Q(phi, allow)
        v_phi = 0.5 * (q_deny + q_allow)                     # (N,) V(phi) marginalized over the two arms
        q_obs = jnp.where(act > 0.5, q_allow, q_deny)        # (N,) Q(phi, a_obs)
        adv = q_obs - v_phi                                  # (N,) the ACTION-CONDITIONAL advantage
        adv_np = np.asarray(adv)
        a_mean = float(adv_np.mean())
        a_std = float(adv_np.std()) if float(adv_np.std()) > 1e-6 else 1.0
        adv_std = (adv - a_mean) / a_std                      # zero-mean / unit-std
        weights = jnp.clip(jnp.exp(adv_std / self._temp), 0.0, self._w_max)  # (N,) the AWR weight

        # ---------------- stage 2: advantage-weighted policy regression ----------------
        # MINIMIZE  - mean( w * log pi(a_obs | phi) ), log pi = Bernoulli(sigmoid(logit)) log-likelihood in
        # the stable log-sigmoid form. High-weight (high-advantage) samples dominate -> the policy regresses
        # onto the convoy-taming actions.
        p_params = _init_head(kp, hidden, out_bias=self._init_logit)
        tx_p = optax.adam(self._lr_p)
        p_opt = tx_p.init(p_params)

        def _policy_loss(p: dict[str, "jax.Array"]) -> "jax.Array":
            logit = _head(p, phi, hidden)                     # (N,)
            log_pi = act * jax.nn.log_sigmoid(logit) + (1.0 - act) * jax.nn.log_sigmoid(-logit)
            # normalize by the weight mass (a weighted MEAN log-likelihood — scale-stable across temp/w_max).
            return -jnp.sum(weights * log_pi) / jnp.maximum(1.0, jnp.sum(weights))

        @jax.jit
        def _policy_step(p: dict[str, "jax.Array"], o: optax.OptState):
            loss, g = jax.value_and_grad(_policy_loss)(p)
            upd, o2 = tx_p.update(g, o, p)
            return optax.apply_updates(p, upd), o2, loss

        self.curves["policy_wbce"] = []
        for i in range(self._policy_steps):
            p_params, p_opt, ploss = _policy_step(p_params, p_opt)
            if i % max(1, self._policy_steps // 100) == 0 or i == self._policy_steps - 1:
                self.curves["policy_wbce"].append(float(ploss))

        # ---------------- diagnostics: the learned policy's gating behavior on the corpus ----------------
        np_params = _params_to_numpy(p_params, hidden)
        probs = _np_policy_probs(np_params, data.phi.astype(np.float32), hidden)  # (N,)
        learned_allow_frac = float((probs >= 0.5).mean())   # deterministic-gate allow fraction over actives
        w_np = np.asarray(weights)
        act_np = np.asarray(act)
        # the AWR weight MASS on deny vs allow samples — the direct evidence the exp-weighting reweights the
        # corpus toward the high-advantage (deny-when-active) action despite the off-policy behavior mixture.
        w_mass_deny = float(w_np[act_np < 0.5].sum() / max(1e-9, w_np.sum()))
        diag = {
            "n_samples": float(data.phi.shape[0]),
            "adv_mean": a_mean, "adv_std": a_std,
            "adv_p10": float(np.percentile(adv_np, 10)), "adv_p90": float(np.percentile(adv_np, 90)),
            # the action-conditional Q-baseline's per-arm values (the action separation a state-only V missed).
            "q_deny_mean": float(np.asarray(q_deny).mean()), "q_allow_mean": float(np.asarray(q_allow).mean()),
            "adv_deny_mean": float(adv_np[act_np < 0.5].mean()),
            "adv_allow_mean": float(adv_np[act_np > 0.5].mean()),
            "obs_deny_share": float((act_np < 0.5).mean()),
            "weight_mass_deny": w_mass_deny,
            "weight_mean": float(w_np.mean()), "weight_max": float(w_np.max()),
            "weight_frac_clipped": float((w_np >= self._w_max - 1e-6).mean()),
            "value_mse_final": float(self.curves["value_mse"][-1]) if self.curves["value_mse"] else 0.0,
            "policy_wbce_final": float(self.curves["policy_wbce"][-1]) if self.curves["policy_wbce"] else 0.0,
            "learned_active_allow_frac": learned_allow_frac,
            "hidden": float(hidden), "temp": self._temp, "w_max": self._w_max,
        }
        return AWRGate(params=np_params, hidden=hidden, fit_diag=diag)


# ============================================================================================
# AWRGate — the runtime Controller (frozen numpy policy, cheap deterministic act).
# ============================================================================================
class AWRGate:
    """The DEPLOYED AWR policy (a frozen runtime Controller, REINFORCEMENT-LEARNING family). Carries the
    fitted numpy policy params (one shared tanh head over the 5-d per-thread phi) and a CHEAP DETERMINISTIC
    act(): a per-thread numpy matvec -> sigmoid -> allow iff prob >= 0.5 (argmax over {deny, allow}), then the
    inflight==0 liveness override (force-allow — a deny there is an UNGATED no-op). No jax, no sampling, no
    learning at runtime (offline-trained). Constructed from fitted params (AWRRecipe.fit) OR from a checkpoint
    .npz (the registered factory). The hot path is O(T*d), non-throwing, with grown-on-demand per-thread
    arrays (the lab-server live-T / concurrent-reset robustness — identical to reinforce/a2c)."""

    family: Family = "rl"

    def __init__(self, params: dict[str, np.ndarray], hidden: int,
                 fit_diag: Mapping[str, float] | None = None) -> None:
        self._hidden = int(hidden)
        self._np_params = {k: (np.asarray(v, dtype=np.float32) if np.ndim(v) else np.float32(v))
                           for k, v in params.items()}
        self._fit_diag = dict(fit_diag) if fit_diag else {}
        self.name = f"offline_awr_h{self._hidden}"
        # --- per-run geometry + the served-diff coalescing baselines (the only per-trial state) ---
        self._t = 1
        self._d_ceil = 1
        self._k = 1
        self._msgs_prev = np.zeros(1, dtype=np.int64)
        self._leaves_prev = np.zeros(1, dtype=np.int64)
        self._seen = np.zeros(1, dtype=bool)
        self._last_mean_prob = 0.0
        self._last_allow_frac = 1.0

    def reset(self, ctx: TrialContext) -> None:
        """Begin a fresh trial: capture the geometry (T, D, K) the features need and RE-ZERO the per-thread
        served-diff coalescing baselines. The POLICY PARAMS are NOT touched — they are the fitted, frozen
        weights (offline-trained); a trial only re-initializes the per-thread feature baselines. D / K guard
        their max(1, .) divisors so a degenerate trial never divides by < 1."""
        self._t = int(ctx.n_threads)
        self._d_ceil = max(1, int(ctx.d_ceiling))
        self._k = max(1, int(ctx.k_per_thread))
        self._msgs_prev = np.zeros(self._t, dtype=np.int64)
        self._leaves_prev = np.zeros(self._t, dtype=np.int64)
        self._seen = np.zeros(self._t, dtype=bool)
        self._last_mean_prob = 0.0
        self._last_allow_frac = 1.0

    def observe(self, reward: float, info: Mapping[str, Any]) -> None:
        """No-op: AWR is OFFLINE — nothing is learned at runtime (the policy is frozen). Present to satisfy
        the frozen Controller contract."""

    def act(self, obs: Observation) -> Sequence[int]:
        """Advance the served-thread first-difference baselines, build the per-thread feature rows phi, run
        the NUMPY policy forward (logits -> probs), emit the DETERMINISTIC gate (allow iff prob >= 0.5), and
        apply the inflight==0 liveness override. Cheap: one O(T*d) numpy matvec, NO jax, NO sampling, NO
        gradient. Non-throwing — the per-thread arrays are grown to the live T on demand (the lab server can
        grow T past reset, and reset runs outside its lock), and defaulted reads keep a malformed/short
        feature frame safe (the watchdog owns loudness on the hot path, ADR-0002)."""
        T = self._t
        feats = obs.features
        self._ensure_capacity(T)
        inflight = _fit(np.asarray(feats.get("inflight", ()), dtype=np.float64), T)
        ready = _fit(np.asarray(feats.get("ready", ()), dtype=np.float64), T)
        msgs = _fit(np.asarray(feats.get("msgs", ()), dtype=np.float64), T).astype(np.int64)
        leaves = _fit(np.asarray(feats.get("leaves", ()), dtype=np.float64), T).astype(np.int64)
        served = [i for i in obs.served if 0 <= i < T]

        coalesce = self._coalesce(msgs, leaves, served, T)
        phi = self._build_phi(inflight, ready, coalesce)        # (T, d_in) float32

        probs = _np_policy_probs(self._np_params, phi, self._hidden)   # (T,) float64
        self._last_mean_prob = float(probs.mean()) if T else 0.0
        gate = (probs >= 0.5).astype(np.int64)                  # deterministic argmax over {deny, allow}

        # liveness override (DENY-ONLY semantics): inflight==0 is an UNGATED forced flush -> force allow.
        gate[inflight <= 0.0] = 1
        self._last_allow_frac = float(np.count_nonzero(gate == 1)) / float(T) if T else 1.0
        return gate.tolist()

    def metrics(self) -> Mapping[str, float]:
        """Dashboard scalars: the deployed policy's last-forward mean allow probability + allow fraction, and
        the AWR fit's recorded diagnostics (the advantage/weight stats + the learned active allow fraction the
        fit produced), so the dashboard shows the offline-trained policy's behavior with zero schema change."""
        out = {"mean_allow_prob": float(self._last_mean_prob), "allow_frac": float(self._last_allow_frac)}
        out.update({f"fit_{k}": float(v) for k, v in self._fit_diag.items()})
        return out

    # ---------------------------------------------------------------- internals (identical to reinforce/a2c)

    def _ensure_capacity(self, T: int) -> None:
        """Grow the per-thread served-diff baselines to at least length T (the lab server can serve a tid
        beyond the reset-time n_threads, and calls reset() outside its lock so act() may run on a not-yet-sized
        array). New slots are un-seen with zero baselines — exactly the cold first-difference state. Idempotent
        + cheap (a no-op once sized)."""
        if self._seen.shape[0] >= T:
            return
        grow = T - self._seen.shape[0]
        self._msgs_prev = np.concatenate([self._msgs_prev, np.zeros(grow, dtype=np.int64)])
        self._leaves_prev = np.concatenate([self._leaves_prev, np.zeros(grow, dtype=np.int64)])
        self._seen = np.concatenate([self._seen, np.zeros(grow, dtype=bool)])

    def _coalesce(self, msgs: np.ndarray, leaves: np.ndarray, served: list[int], T: int) -> np.ndarray:
        """Served-thread first-difference of the CUMULATIVE counters -> instantaneous coalescing degree
        (Δleaves/Δmsgs). Only served & previously-seen threads carry a real delta; everyone else gets the
        neutral 1.0. Absent threads are NEVER differenced (their sentinel-0 reading would fake a negative
        delta) and their baselines are NOT advanced — the wire subtlety, honored (identical to reinforce/a2c
        AND to the offline trainer's reconstruction, so train/serve features match exactly)."""
        coalesce = np.ones(T, dtype=np.float64)
        for i in served:
            if self._seen[i]:
                d_msgs = int(msgs[i] - self._msgs_prev[i])
                d_leaves = int(leaves[i] - self._leaves_prev[i])
                if d_msgs > 0 and d_leaves >= 0:
                    coalesce[i] = d_leaves / float(d_msgs)
            self._msgs_prev[i] = msgs[i]
            self._leaves_prev[i] = leaves[i]
            self._seen[i] = True
        return coalesce

    def _build_phi(self, inflight: np.ndarray, ready: np.ndarray, coalesce: np.ndarray) -> np.ndarray:
        """Assemble the (T, d_in) float32 feature matrix in the canonical _FEATURES order (identical to
        reinforce/a2c). All divisors are max(1, .)-guarded so phi is always finite (ADR-0002: the hot path
        stays total)."""
        D = float(self._d_ceil)
        headroom = np.maximum(1.0, D - inflight)
        submit_pressure = ready / headroom
        ready_backlog_norm = ready / float(self._k)
        inflight_saturation = inflight / D
        bias = np.ones_like(inflight)
        phi = np.stack(
            [submit_pressure, ready_backlog_norm, inflight_saturation, coalesce, bias], axis=0
        ).T
        return phi.astype(np.float32)


# ============================================================================================
# numpy mirror helpers (the hot-path forward) — the numpy twins of _head, shared with the trainer's diag.
# ============================================================================================
def _params_to_numpy(params: dict[str, "jax.Array"], hidden: int) -> dict[str, np.ndarray]:
    """Export the jax fitted params to a fresh numpy dict (the hot-path mirror + the checkpoint payload). One
    device->host copy at the end of the fit — off any latency path."""
    if hidden <= 0:
        return {"w": np.asarray(params["w"], dtype=np.float32),
                "b": np.float32(np.asarray(params["b"], dtype=np.float32))}
    return {"w1": np.asarray(params["w1"], dtype=np.float32), "b1": np.asarray(params["b1"], dtype=np.float32),
            "w2": np.asarray(params["w2"], dtype=np.float32), "b2": np.asarray(params["b2"], dtype=np.float32)}


def _sigmoid_np(z: np.ndarray) -> np.ndarray:
    """Vectorized numerically-stable sigmoid on the hot path (no jax). The branchless-by-mask stable form
    avoids overflow for large |z| (exp of a positive argument only)."""
    out = np.empty_like(z)
    pos = z >= 0.0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def _np_policy_probs(np_params: dict[str, np.ndarray], phi: np.ndarray, hidden: int) -> np.ndarray:
    """The per-forward NUMPY policy forward: per-thread allow PROBABILITY p[t] = sigmoid(logit[t]) from the
    numpy params mirror. phi is (T, d_in) (or (N, d_in) in the trainer diag); returns float64. Linear:
    phi @ w + b. Hidden: one tanh layer then a linear read-out — the numpy twin of _head. O(T*d), no jax (so
    no cold compile, no re-trace, no device contention on the synchronous per-forward path)."""
    if hidden <= 0:
        logit = phi.astype(np.float64) @ np_params["w"].astype(np.float64) + float(np_params["b"])
    else:
        h = np.tanh(phi.astype(np.float64) @ np_params["w1"].astype(np.float64) + np_params["b1"].astype(np.float64))
        logit = h @ np_params["w2"].astype(np.float64) + float(np_params["b2"])
    return _sigmoid_np(logit)


def _fit(x: np.ndarray, t: int) -> np.ndarray:
    """Coerce a feature array to length T: truncate if long, zero-pad if short. Defensive so act() never
    throws on a malformed/empty feature list (ADR-0002: the per-forward path stays cheap and total; a
    zero-padded slot lands in the un-baselined / inflight==0 liveness path, i.e. neutral + force-allow)."""
    if x.shape[0] == t:
        return x
    out = np.zeros(t, dtype=np.float64)
    n = min(x.shape[0], t)
    if n:
        out[:n] = x[:n]
    return out


# ============================================================================================
# checkpoint I/O + the registered factory (so the harness's make_controller gets a real Controller).
# ============================================================================================
def save_checkpoint(path: str, gate: "AWRGate") -> None:
    """Persist a fitted AWRGate to a .npz checkpoint (np.savez, allow_pickle=False on load — the codebase's
    weights convention). Stores the policy params + the hidden width + the fit diagnostics so the deployed
    factory reconstructs the exact policy AND can surface the fit metrics. Creates the parent dir."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    payload: dict[str, Any] = {"hidden": np.int64(gate._hidden)}
    for k, v in gate._np_params.items():
        payload[f"param_{k}"] = np.asarray(v, dtype=np.float32)
    # fit diagnostics as two parallel arrays (keys + values) so the load is allow_pickle=False safe.
    if gate._fit_diag:
        payload["diag_keys"] = np.array(list(gate._fit_diag.keys()))
        payload["diag_vals"] = np.array([float(v) for v in gate._fit_diag.values()], dtype=np.float64)
    np.savez(path, **payload)


def load_checkpoint(path: str) -> "AWRGate":
    """Reconstruct an AWRGate from a .npz checkpoint (allow_pickle=False — the codebase's weights load
    convention). Fail loud (ADR-0002) if the file is absent or missing the params, never a silent default
    policy (a missing checkpoint at deploy time is a wiring error, surfaced before the trial)."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"AWRGate.load_checkpoint: no checkpoint at {path!r} — train one first "
            f"(offline_awr_train.py) or set CHOCOFARM_AWR_CKPT (ADR-0002: refuse to deploy an unfitted policy)"
        )
    z = np.load(path, allow_pickle=False)
    hidden = int(z["hidden"])
    params = {k[len("param_"):]: z[k] for k in z.files if k.startswith("param_")}
    if not params:
        raise ValueError(f"AWRGate.load_checkpoint: {path!r} carries no policy params (corrupt checkpoint)")
    diag: dict[str, float] = {}
    if "diag_keys" in z.files and "diag_vals" in z.files:
        diag = {str(k): float(v) for k, v in zip(z["diag_keys"], z["diag_vals"])}
    return AWRGate(params=params, hidden=hidden, fit_diag=diag)


def _awr_factory() -> "AWRGate":
    """The registry factory: load the deployed AWR policy from the checkpoint (CHOCOFARM_AWR_CKPT, default
    DEFAULT_CKPT) and return a READY Controller — NOT a TrainableRecipe (the harness's make_controller rejects
    a recipe). So deploying the trained policy is one env var + the registry name `offline_awr`, no harness
    edit (P2 seam discipline). Fail loud if the checkpoint is missing (ADR-0002)."""
    path = os.environ.get("CHOCOFARM_AWR_CKPT", DEFAULT_CKPT)
    return load_checkpoint(path)


# Register additively into the FROZEN adapter.REGISTRY (one entry — the harness + dashboard discover methods
# here). The factory returns a ready Controller (loaded from the checkpoint), so make_controller accepts it;
# the offline TRAINING is the AWRRecipe.fit path the trainer drives. setdefault so a re-import or a name clash
# never silently clobbers an existing registration.
REGISTRY.setdefault("offline_awr", _awr_factory)
