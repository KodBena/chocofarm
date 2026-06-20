#!/usr/bin/env python3
"""
cpp/stage_a/control_lab/methods/a2c.py — an advantage actor-critic (A2C) issue-gate
(REINFORCEMENT-LEARNING family) candidate for the issue-gate control lab.

The sample-efficiency upgrade over REINFORCE (methods/reinforce.py). Where REINFORCE baselines the
policy-gradient with a SCALAR running-mean reward, A2C carries a SHARED CRITIC V_psi(phi[t]) — a tiny learned
value function over the SAME per-thread feature row phi[t] reinforce uses — and baselines with the
bootstrapped one-step ADVANTAGE

    A[t] = r + gamma * V_psi(phi'[t]) - V_psi(phi[t])

so the gradient reinforces a thread's sampled action only insofar as the realized return BEAT THE CRITIC's
estimate of that thread's state, not merely the run-wide average. A state-conditioned baseline cuts the
policy-gradient variance harder than a scalar one, which is the whole point in a run that is only
hundreds-to-thousands of forwards over a ~4s box (the central challenge — convergence in the budget). The
critic is trained alongside the actor by regressing V toward the bootstrapped TD(0) target, the standard
actor-critic coupling.

JAX FOR THE GRADIENT, NUMPY FOR THE HOT DECISION (the lab-server reality). This controller runs IN the
eval-server process, alongside that process's own JAX inference server, under an XLA single-thread pin, and the
gate decision is SYNCHRONOUS on the per-forward boundary with a hard 50ms watchdog. A jax forward on THAT path
is the wrong tool: its first call cold-compiles (blowing the deadline), it re-traces on a changing T, and it
shares the device with the inference server. So the per-forward act() ACTOR FORWARD is done in NUMPY from a
NUMPY MIRROR of the actor params (a cheap O(T*d) matvec, d=5) — no jax on the hot tick at all. The CRITIC is
not consulted on the hot path anyway (it baselines only in the periodic update), so only the actor needs a
mirror. The LEARNING (the combined actor+critic optax adam step) keeps jax, runs DECIMATED (once per N
forwards, off the hot path), and the new actor params are EXPORTED back to the numpy mirror after each step.
jax for the gradient; numpy for the decision — the same pure-numpy hot path every non-RL method here uses.

PARAMETER-SHARING is the lever that makes both nets trainable in-budget: ONE shared actor and ONE shared
critic see EVERY per-thread row, so a single forward yields T (state, action, reward, next-state) transitions
and both gradients are well-conditioned. Every thread is one sample of the same pi_theta / one evaluation of
the same V_psi; the per-thread credit collapses into the pool's per-forward reward (the harness feeds one
scalar per forward — the forward's real row count, the coalescing achieved; HIGHER IS BETTER), which is the
shared RL reward for every thread that acted this forward.

Mechanism (features -> stochastic gate). IDENTICAL feature surface to reinforce — the brief's "same phi as
reinforce". Each forward, per thread t, a 5-d row phi[t] is read off the wire (D = ctx.d_ceiling, K =
ctx.k_per_thread, the capacity normalizers the feature wire omits):

    phi[t] = [ submit_pressure      = ready / max(1, D - inflight),   # queued work vs. headroom to release it
               ready_backlog_norm   = ready / max(1, K),              # K-normalized parked-at-leaf backlog
               inflight_saturation  = inflight / max(1, D),           # proximity to the DENY-ONLY no-op (j -> D)
               coalesce_degree_inst = (Δleaves) / max(1, Δmsgs),      # rows-per-message achieved (windowed, served-diff)
               1.0 ]                                                  # bias

The SHARED ACTOR is a tiny stochastic policy (linear-logistic, or one hidden tanh layer): a per-thread allow
logit, a sigmoid prob, a Bernoulli SAMPLE (sampling IS the exploration — there is no separate epsilon, but an
ENTROPY BONUS keeps the policy from collapsing too fast):

    logit[t] = actor(phi[t])      p[t] = sigmoid(logit[t])      a[t] ~ Bernoulli(p[t])   (a[t]=1 => ALLOW)

The SHARED CRITIC is a second tiny net of the same shape with a SCALAR read-out: V_psi(phi[t]) -> R. The actor
forward on the per-forward critical path is a NUMPY matvec + a numpy Bernoulli draw (O(T*d), d=5, well inside
the deadline, NEVER touching jax); the critic and both gradients live only in the periodic batched update.

Cold start = allow-leaning baseline (the RL analog of the bandit's all-allow anchor arm). The actor's bias
weight is initialized to `init_allow_logit` (>0) with everything else zero, so the cold policy samples allow
with high probability (near the all-allow control arm — the safety floor for the un-warmed warm-up forwards)
while still being STOCHASTIC. The critic's output is zero-initialized, so the cold value is 0 everywhere (the
advantage starts as the raw reward and the critic learns the baseline from there).

Liveness override (DENY-ONLY gate semantics — the runner's effective gate is `inflight < D && allow` and the
forced flush at inflight==0 is UNGATED, so a deny is a NO-OP there): inflight[t]==0 -> force allow. The SAMPLED
action there is OVERRIDDEN. Crucially this also gates CREDIT: a thread whose gate was a forced no-op did not
causally affect the reward, so its sampled log-prob is MASKED OUT of the actor gradient AND its row is dropped
from the critic regression (training on a no-op sample injects pure noise). Only threads that actually acted
(inflight>0 at sample time) carry credit/blame — the faithful handling of "the sampled action there is
overridden".

The learner (observe(reward) -> periodic batched A2C update). The reward fed to observe() is the pool's
PER-FORWARD THROUGHPUT CONTRIBUTION (forward_rows; HIGHER IS BETTER) — the RL reward. Per the FROZEN contract,
observe(r) delivers the reward of the PREVIOUS act. A bootstrapped transition needs BOTH the realized reward
AND the NEXT state phi', which arrive one forward apart: act_i stashes (phi_i, a_i, mask_i) PENDING; observe at
forward i+1 attaches r_i (becoming AWAITING-NEXT — it has the reward but not phi'); act_{i+1} computes phi_{i+1}
which IS phi' for transition i, finalizing (phi_i, a_i, mask_i, r_i, phi_{i+1}) into the trajectory buffer. So
each transition lands one forward after its reward (its next-state's forward). EVERY N forwards (the efficiency
lever — never a backward every forward) ONE optax (adam) step is taken on the accumulated batch:

    A[t]          = r + gamma * V(phi'[t]) - V(phi[t])            # one-step bootstrapped advantage (per thread)
    actor_loss    = -mean_active( stop_grad(A[t]) * log pi(a[t]|phi[t]) ) - entropy_coef * mean_active(H[t])
    critic_loss   =  mean_active( ( stop_grad(r + gamma*V(phi'[t])) - V(phi[t]) )^2 )   # regress V to the TD target
    loss          =  actor_loss + value_coef * critic_loss

The advantage is STOP-GRADIENT'd into the actor term (the actor sees A as a fixed weight, the textbook A2C
decoupling), and the bootstrap TARGET r + gamma*V(phi') is STOP-GRADIENT'd in the critic term so the critic
regresses V(phi) toward the target by gradient through -V(phi) only (TD(0) semi-gradient — the stable reading
of "regress V toward the bootstrapped return"; differentiating through the bootstrap couples target and
estimate and destabilizes the short run). H[t] is the Bernoulli entropy -(p log p + (1-p) log(1-p)); subtracting
it from the loss MAXIMIZES entropy (the exploration bonus). The actor and critic keep SEPARATE params and
SEPARATE adam transforms (so lr_actor / lr_critic are honest independent knobs), but ONE jit'd step takes the
grad of the single combined loss w.r.t. BOTH pytrees and applies each transform — the periodic, DECIMATED batch
update, NOT on the per-forward critical path. After the step the new ACTOR params are EXPORTED to the numpy
mirror the hot path reads. The buffer clears after each step (on-policy A2C: the recent trajectory is the right
horizon for a short non-stationary run). gradient ASCENT on the objective = descend this loss (optax minimizes),
the sign already folded into the loss above.

JIT warmup (reset, off the timed path). The combined optax step is JIT-compiled on a DUMMY fixed-shape batch
during reset() — BEFORE the wall box opens and BEFORE the first real update — so the first real learning step
pays no cold-compile spike on the serve thread. The hot act() is numpy and never compiles at all.

Wire subtlety (honored, identical to reinforce). coalesce_degree_inst first-differences the CUMULATIVE counters
msgs and leaves. lab_server builds each length-T feature list fresh as [0]*T and fills ONLY the served tids, so
a thread ABSENT from a forward reads a SENTINEL 0, not its true cumulative. The per-thread msgs/leaves baselines
advance ONLY for obs.served (against a per-thread baseline, and only once a thread has been SEEN before) — an
absent thread is never first-differenced (its sentinel 0 would manufacture a spurious negative delta) and its
baseline is not touched. An un-baselined or quiet thread reads coalesce_degree_inst = 1.0 (the neutral "no
coalescing measured yet" value). The other features are INSTANTANEOUS gauges (ready, inflight), never
differenced, so a sentinel-0 there is a harmless zero. The reward is the harness's per-forward row count and
needs no wire decode.

Live-T robustness (the lab server grows T past the trial ctx). The server LAZILY grows its gate-vector length
when a served tid exceeds the reset-time n_threads (lab_server._serve_batch), and it calls reset() OUTSIDE its
lock while the serve thread is already acting — so act() can be entered on a thread whose tid exceeds the
per-thread arrays' current length, or even mid-reset. Every per-thread array is therefore GROWN-ON-DEMAND to
the live T at the top of act() (and the numpy actor mirror is always present, set in __init__), so the hot path
is total under both a grown T and a concurrent reset (ADR-0002: the watchdog owns loudness; the hot path stays
well-defined and never throws).

RL family: reset() COLD-STARTS the learner (re-initializes BOTH actor + critic params and their adam moment
states, refreshes the actor numpy mirror, JIT-warms the combined step, clears the trajectory buffer, the
pending + awaiting-next transitions, and the first-difference baselines) — nothing learned survives a trial.
observe() attaches the realized reward and, every N forwards, drives the optax step. act() samples the gate from
the CURRENT actor via the numpy mirror (and stashes the bootstrapping transition). metrics() exposes the
critic's mean value, the policy entropy, the mean advantage (the three the brief names), the last gradient norm,
and the update count. Knobs: lr_actor, lr_critic, gamma, entropy_coef, update period N, hidden size (0 = linear;
>0 = one hidden tanh layer), value loss weight, the cold-start allow logit, and the optional advantage
standardization.

Run the unit gate pinned + bounded:
    PYTHONPATH=cpp/stage_a /home/bork/w/vdc/venvs/generic/bin/python -m pytest tests/test_method_a2c.py -q

Public Domain (The Unlicense).
"""
from __future__ import annotations

from functools import partial
from typing import Any, Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np
import optax

from control_lab.adapter import REGISTRY, Family, Observation, TrialContext

# The fixed feature layout (bias last), IDENTICAL to reinforce's — the brief's "same phi as reinforce". The
# realized input dimension d is DERIVED from this tuple, never hardcoded (ADR-0002 single-source-of-truth: the
# nets' input width is len(_FEATURES); the consumers derive their view). The order here is the order _build_phi
# assembles the columns in.
_FEATURES: tuple[str, ...] = (
    "submit_pressure",
    "ready_backlog_norm",
    "inflight_saturation",
    "coalesce_degree_inst",
    "bias",
)
_D_IN = len(_FEATURES)  # 5; the input dimension shared by actor and critic, derived from the layout


# ----------------------------------------------------------------------------- JIT'd actor-critic core (pure)
# The LEARNING path only (jax.grad + optax over the combined actor+critic loss). Module-level pure functions so
# JAX caches ONE compiled artifact across instances of the same (hidden,) static shape rather than re-tracing
# per controller. `hidden` is a static argument (it changes a net's structure / parameter pytree), so a trace
# is keyed on it. NB: NONE of these run on the per-forward critical path — the hot decision is the numpy actor
# forward below; the critic is consulted only here, in the periodic update.


def _init_head(key: "jax.Array", hidden: int, out_bias: float, zero_readout: bool) -> dict[str, "jax.Array"]:
    """Initialize ONE tiny scalar-output head pytree (actor logit head OR critic value head — same shape).
    Linear (hidden==0): a weight vector `w` (d_in,) and scalar bias `b`, zero EXCEPT b = out_bias. One hidden
    tanh layer (hidden>0): small-random W1/b1 (Glorot-ish, *0.1) and an output layer. When `zero_readout` the
    output weights are zero so the cold head emits the constant `out_bias` regardless of the random first layer
    (the actor's allow-leaning cold start; the critic's zero cold value) — it departs from the constant only as
    the read-out learns."""
    if hidden <= 0:
        w = jnp.zeros((_D_IN,), dtype=jnp.float32)
        b = jnp.asarray(out_bias, dtype=jnp.float32)
        return {"w": w, "b": b}
    k1, _ = jax.random.split(key)
    w1 = jax.random.normal(k1, (_D_IN, hidden), dtype=jnp.float32) * 0.1
    b1 = jnp.zeros((hidden,), dtype=jnp.float32)
    w2 = jnp.zeros((hidden,), dtype=jnp.float32) if zero_readout \
        else jax.random.normal(k1, (hidden,), dtype=jnp.float32) * 0.1
    b2 = jnp.asarray(out_bias, dtype=jnp.float32)
    return {"w1": w1, "b1": b1, "w2": w2, "b2": b2}


def _head(params: dict[str, "jax.Array"], phi: "jax.Array", hidden: int) -> "jax.Array":
    """Scalar-per-thread read-out of a tiny head on phi (T, d_in). Linear: phi @ w + b. Hidden: one tanh layer
    then a linear read-out. Returns (T,) — the actor's allow LOGIT, or the critic's VALUE, by which params it
    is called with (the two heads have the same shape; the meaning is the caller's). Used INSIDE the loss only."""
    if hidden <= 0:
        return phi @ params["w"] + params["b"]
    h = jnp.tanh(phi @ params["w1"] + params["b1"])  # (T, hidden)
    return h @ params["w2"] + params["b2"]           # (T,)


def _bernoulli_entropy(logit: "jax.Array") -> "jax.Array":
    """Bernoulli entropy H(p) = -(p log p + (1-p) log(1-p)) from the logit, in a numerically-stable form via
    log-sigmoid (p log p uses p = sigmoid(z), log p = logsig(z); the (1-p) term uses logsig(-z)). Returns (T,),
    >= 0, peaking at logit 0 (p=0.5). Subtracting its mean from the loss is the exploration bonus."""
    log_p = jax.nn.log_sigmoid(logit)      # log sigmoid(z)
    log_q = jax.nn.log_sigmoid(-logit)     # log (1 - sigmoid(z))
    p = jax.nn.sigmoid(logit)
    return -(p * log_p + (1.0 - p) * log_q)


def _a2c_loss(
    params: tuple[dict[str, "jax.Array"], dict[str, "jax.Array"]],  # (actor, critic)
    phi: "jax.Array",        # (B, T, d_in)   the state rows
    phi_next: "jax.Array",   # (B, T, d_in)   the next-state rows (bootstrap)
    act: "jax.Array",        # (B, T)         sampled actions in {0,1}
    mask: "jax.Array",       # (B, T)         1.0 where the thread actually acted (inflight>0), else 0.0
    rew: "jax.Array",        # (B,)           per-forward shared reward, broadcast to every active thread
    gamma: float,
    entropy_coef: float,
    value_coef: float,
    hidden: int,
) -> tuple["jax.Array", tuple["jax.Array", "jax.Array", "jax.Array"]]:
    """The combined A2C loss (optax minimizes; the ascent sign is folded in). Per active thread:

        A         = r + gamma * V(phi') - V(phi)                         (one-step bootstrapped advantage)
        actor     = - stop_grad(A) * log pi(a | phi)  -  entropy_coef * H(phi)
        critic    =   ( stop_grad(r + gamma*V(phi')) - V(phi) )^2        (TD(0) semi-gradient regression)
        loss      =   mean_active(actor) + value_coef * mean_active(critic)

    The advantage is stop-grad'd into the actor (the textbook decoupling) and the bootstrap target is stop-grad'd
    into the critic (TD(0): differentiate only -V(phi)). The per-forward reward r is the SHARED credit broadcast
    to every active thread (parameter-sharing). MASKED (no-op / inflight==0) threads contribute 0 to BOTH terms.
    Returns (loss, (mean_value, mean_entropy, mean_advantage)) — the (loss, aux) pair `jax.grad(..,
    has_aux=True)` consumes, the aux feeding the dashboard metrics."""
    actor, critic = params
    B, T, d = phi.shape
    flat = phi.reshape(-1, d)            # (B*T, d_in)
    flat_n = phi_next.reshape(-1, d)     # (B*T, d_in)

    logit = _head(actor, flat, hidden).reshape(act.shape)          # (B, T) allow logits
    v = _head(critic, flat, hidden).reshape(act.shape)             # (B, T) V(phi)
    v_next = _head(critic, flat_n, hidden).reshape(act.shape)      # (B, T) V(phi')

    r = rew.reshape(B, 1)                                          # (B, 1) -> broadcast over threads
    td_target = r + gamma * v_next                                # (B, T) bootstrapped return
    adv = td_target - v                                           # (B, T) advantage (grad flows here for critic)

    # actor: REINFORCE-with-baseline score, advantage held fixed (stop-grad). log pi for Bernoulli(sigmoid(z))
    # is the stable log-sigmoid form: a*logsig(z) + (1-a)*logsig(-z).
    log_pi = act * jax.nn.log_sigmoid(logit) + (1.0 - act) * jax.nn.log_sigmoid(-logit)  # (B, T)
    ent = _bernoulli_entropy(logit)                              # (B, T)
    adv_sg = jax.lax.stop_gradient(adv)
    actor_term = -(adv_sg * log_pi) - entropy_coef * ent         # (B, T)

    # critic: TD(0) semi-gradient — regress V(phi) toward the stop-grad'd bootstrap target (grad through -V(phi)).
    critic_term = (jax.lax.stop_gradient(td_target) - v) ** 2    # (B, T)

    # mask to ACTIVE threads only (a forced no-op carries no credit) and mean over the active count (param-share).
    denom = jnp.maximum(1.0, jnp.sum(mask))
    actor_loss = jnp.sum(actor_term * mask) / denom
    critic_loss = jnp.sum(critic_term * mask) / denom
    loss = actor_loss + value_coef * critic_loss

    mean_value = jnp.sum(v * mask) / denom
    mean_entropy = jnp.sum(ent * mask) / denom
    mean_adv = jnp.sum(adv * mask) / denom
    return loss, (mean_value, mean_entropy, mean_adv)


def _make_update_step(
    tx_actor: optax.GradientTransformation,
    tx_critic: optax.GradientTransformation,
    gamma: float,
    entropy_coef: float,
    value_coef: float,
    hidden: int,
) -> Any:
    """Build the JIT'd combined A2C optax step, closing over the two transforms (one for the actor params, one
    for the critic params — so lr_actor / lr_critic stay independent) and the static scalars (mirrors
    az/optimizer.make_update: the transforms are pytrees of FUNCTIONS, captured in the closure, not traced
    arguments). The returned closure takes (actor, critic, opt_a, opt_c, phi, phi', act, mask, rew) and returns
    (actor', critic', opt_a', opt_c', grad_norm, (mean_value, mean_entropy, mean_adv)): one grad of the single
    combined loss w.r.t. BOTH pytrees, each transform applied to its own grads. This whole step is the periodic,
    DECIMATED batch update; it is NOT on the per-forward critical path."""

    @jax.jit
    def _step(
        actor: dict[str, "jax.Array"],
        critic: dict[str, "jax.Array"],
        opt_a: optax.OptState,
        opt_c: optax.OptState,
        phi: "jax.Array",
        phi_next: "jax.Array",
        act: "jax.Array",
        mask: "jax.Array",
        rew: "jax.Array",
    ) -> tuple[
        dict[str, "jax.Array"], dict[str, "jax.Array"], optax.OptState, optax.OptState,
        "jax.Array", tuple["jax.Array", "jax.Array", "jax.Array"],
    ]:
        (loss, aux), grads = jax.value_and_grad(_a2c_loss, has_aux=True)(
            (actor, critic), phi, phi_next, act, mask, rew, gamma, entropy_coef, value_coef, hidden
        )
        g_actor, g_critic = grads
        upd_a, opt_a2 = tx_actor.update(g_actor, opt_a, actor)
        upd_c, opt_c2 = tx_critic.update(g_critic, opt_c, critic)
        actor2 = optax.apply_updates(actor, upd_a)
        critic2 = optax.apply_updates(critic, upd_c)
        gnorm = optax.tree.norm(grads)  # global L2 grad norm over BOTH heads (learning-health metric)
        return actor2, critic2, opt_a2, opt_c2, gnorm, aux

    return _step


class A2CGate:
    """An advantage actor-critic (A2C) issue gate (REINFORCEMENT-LEARNING family). A TINY SHARED actor
    pi_theta (stochastic, Bernoulli-sampled allow bit) and a TINY SHARED critic V_psi over the SAME per-thread
    feature row phi[t] (reinforce's phi). Each forward is T (state, action, reward, next-state) transitions
    (parameter-sharing); the realized PER-FORWARD reward (forward_rows, higher is better) is the shared credit.
    Every N forwards an optax adam step minimizes actor_loss + value_coef*critic_loss, with the one-step
    bootstrapped advantage A = r + gamma*V(phi') - V(phi) baselining the actor (stop-grad'd) and the critic
    regressing V toward the TD(0) target; an entropy bonus sustains exploration. The per-forward actor forward is
    a NUMPY matvec from a numpy mirror of the actor (O(T*d), d=5) — jax never touches the hot path; the critic +
    both gradients live only in the periodic, decimated optax step (jax), which exports the new actor to the
    mirror. inflight==0 force-allows AND masks that thread out of BOTH gradients. Cold-started each trial."""

    family: Family = "rl"

    def __init__(
        self,
        lr_actor: float = 0.05,
        lr_critic: float = 0.1,
        gamma: float = 0.9,
        entropy_coef: float = 0.01,
        update_period: int = 16,
        hidden: int = 0,
        value_coef: float = 0.5,
        init_allow_logit: float = 2.0,
        standardize_adv: bool = True,
        max_batch: int = 256,
    ) -> None:
        # fail loud (ADR-0002): degenerate hyperparameters are a CONSTRUCTION error, surfaced at build time on
        # the ctor, never a silent surprise on the per-forward hot path.
        if lr_actor <= 0.0:
            raise ValueError(f"A2CGate: lr_actor must be > 0, got {lr_actor}")
        if lr_critic <= 0.0:
            raise ValueError(f"A2CGate: lr_critic must be > 0, got {lr_critic}")
        if not (0.0 <= gamma <= 1.0):
            raise ValueError(f"A2CGate: gamma must be in [0, 1], got {gamma}")
        if entropy_coef < 0.0:
            raise ValueError(f"A2CGate: entropy_coef must be >= 0, got {entropy_coef}")
        if update_period < 1:
            raise ValueError(f"A2CGate: update_period N must be >= 1, got {update_period}")
        if hidden < 0:
            raise ValueError(f"A2CGate: hidden must be >= 0 (0 = linear nets), got {hidden}")
        if value_coef < 0.0:
            raise ValueError(f"A2CGate: value_coef must be >= 0, got {value_coef}")
        if not np.isfinite(init_allow_logit):
            raise ValueError(f"A2CGate: init_allow_logit must be finite, got {init_allow_logit}")
        if max_batch < 1:
            raise ValueError(f"A2CGate: max_batch must be >= 1, got {max_batch}")

        self._lr_a = float(lr_actor)
        self._lr_c = float(lr_critic)
        self._gamma = float(gamma)
        self._ent_coef = float(entropy_coef)
        self._n = int(update_period)
        self._hidden = int(hidden)
        self._value_coef = float(value_coef)
        self._init_logit = float(init_allow_logit)
        self._standardize = bool(standardize_adv)
        self._max_batch = int(max_batch)
        self.name = f"a2c_la{self._lr_a:g}_lc{self._lr_c:g}_g{self._gamma:g}_N{self._n}_h{self._hidden}"

        # the optax transforms are built once (the moment pytrees live in self._opt_*, re-init in reset()); the
        # JIT'd combined step closes over BOTH transforms + the static scalars, built once here (not per update).
        self._tx_a = optax.adam(self._lr_a)
        self._tx_c = optax.adam(self._lr_c)
        self._update_step = _make_update_step(
            self._tx_a, self._tx_c, self._gamma, self._ent_coef, self._value_coef, self._hidden
        )

        # --- per-run learner state (sized/cleared in reset) ---
        self._t = 1
        self._d_ceil = 1
        self._k = 1
        self._actor: dict[str, "jax.Array"] = {}
        self._critic: dict[str, "jax.Array"] = {}
        self._opt_a: optax.OptState = None  # type: ignore[assignment]
        self._opt_c: optax.OptState = None  # type: ignore[assignment]
        # NUMPY MIRROR of the ACTOR params — the ONLY thing the hot path reads (the critic is off the hot path).
        # Always present (set here AND at reset()/each update), so a concurrent reset never leaves act() reading
        # a missing key. Cold-init to the allow-leaning constant logit (the safety floor) even pre-reset.
        self._np_actor: dict[str, np.ndarray] = _init_np_actor(self._hidden, self._init_logit)
        # the hot-path Bernoulli RNG is a plain numpy Generator (reseeded from ctx.seed in reset); the jax key
        # below is used ONLY for the (off-hot-path) param init.
        self._rng = np.random.default_rng(0)
        self._key = jax.random.PRNGKey(0)
        # trajectory buffer of COMPLETED transitions (reward AND next-state attached), drained every N forwards.
        self._phi_buf: list[np.ndarray] = []       # each (T, d_in)   state
        self._phin_buf: list[np.ndarray] = []      # each (T, d_in)   next state (bootstrap)
        self._act_buf: list[np.ndarray] = []       # each (T,) in {0,1}
        self._mask_buf: list[np.ndarray] = []      # each (T,) active mask (inflight>0)
        self._rew_buf: list[float] = []            # each scalar per-forward reward
        # the PENDING transition: act() sampled it; the NEXT observe() attaches its reward (the contract's
        # "reward of the PREVIOUS act"). _awaiting holds a reward-attached transition still missing phi'; the
        # NEXT act() supplies phi' and finalizes it into the buffer.
        self._pending: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None       # (phi, a, mask)
        self._awaiting: tuple[np.ndarray, np.ndarray, np.ndarray, float] | None = None  # (phi, a, mask, r)
        # running-mean reward (dashboard only; the critic — not this scalar — is the baseline).
        self._b_sum = 0.0
        self._b_cnt = 0
        # cumulative-counter baselines for the served-thread first-difference (the wire subtlety).
        self._msgs_prev = np.zeros(1, dtype=np.int64)
        self._leaves_prev = np.zeros(1, dtype=np.int64)
        self._seen = np.zeros(1, dtype=bool)
        # dashboard scalars (the brief's three + learning health).
        self._updates = 0
        self._last_grad_norm = 0.0
        self._last_mean_value = 0.0
        self._last_entropy = float(_bernoulli_entropy_scalar(self._init_logit))
        self._last_mean_adv = 0.0
        self._last_mean_prob = float(_sigmoid(self._init_logit))

    def reset(self, ctx: TrialContext) -> None:
        """COLD-START a fresh trial: capture the geometry (T, D, K) the features need, RE-INITIALIZE BOTH the
        actor and critic params + their adam moment states (nothing learned survives a trial), refresh the ACTOR
        NUMPY MIRROR the hot path reads, JIT-WARM the combined optax step on a dummy batch (so the first real
        update pays no cold compile on the serve thread), and clear the trajectory buffer, the pending +
        awaiting-next transitions, the running-mean accumulator, and the first-difference baselines. The RNG is
        reseeded from ctx.seed so a trial's sampling is reproducible per the lab's seed.

        Order note (ADR-0002, the lab-server race): set_trial calls this OUTSIDE its lock while the serve thread
        may already be acting, so the per-thread arrays + the actor mirror are published as the LAST writes,
        each a single atomic rebind; an act() that interleaves reads either the prior trial's consistent state
        or the freshly-published one, never a torn half. (act() also grows the arrays on demand, so even a
        smaller prior length is safe.)"""
        self._t = int(ctx.n_threads)
        self._d_ceil = max(1, int(ctx.d_ceiling))
        self._k = max(1, int(ctx.k_per_thread))
        seed = int(ctx.seed) & 0x7FFFFFFF
        self._key = jax.random.PRNGKey(seed)
        self._key, ka, kc = jax.random.split(self._key, 3)
        # actor: allow-leaning cold start (bias = init_allow_logit, zero read-out). critic: zero cold value
        # (bias = 0, zero read-out) so the advantage starts as the raw reward and the critic learns from there.
        actor = _init_head(ka, self._hidden, self._init_logit, zero_readout=True)
        critic = _init_head(kc, self._hidden, 0.0, zero_readout=True)
        opt_a = self._tx_a.init(actor)
        opt_c = self._tx_c.init(critic)
        # PUBLISH the valid initialized params + an EMPTY buffer FIRST, BEFORE the (GIL-releasing) warmup compile
        # below. set_trial calls reset() outside its lock while the serve thread runs observe()/act(); act() can
        # fire _train_on_batch() which reads self._actor/self._critic — so both must be valid pytrees (never the
        # stale {} or a torn half) the instant the trial goes active, and the buffer must be empty so no stale
        # transition from the prior trial drives a step against the fresh params (ADR-0002: the learner stays
        # well-defined under the concurrent reset).
        self._actor = actor
        self._critic = critic
        self._opt_a = opt_a
        self._opt_c = opt_c
        self._rng = np.random.default_rng(seed)
        self._phi_buf = []
        self._phin_buf = []
        self._act_buf = []
        self._mask_buf = []
        self._rew_buf = []
        self._pending = None
        self._awaiting = None
        self._b_sum = 0.0
        self._b_cnt = 0
        self._updates = 0
        self._last_grad_norm = 0.0
        self._last_mean_value = 0.0
        self._last_entropy = float(_bernoulli_entropy_scalar(self._init_logit))
        self._last_mean_adv = 0.0
        self._last_mean_prob = float(_sigmoid(self._init_logit))
        # PUBLISH the actor mirror + the sized per-thread arrays LAST (each an atomic single rebind).
        self._np_actor = _actor_to_numpy(self._actor, self._hidden)
        self._msgs_prev = np.zeros(self._t, dtype=np.int64)
        self._leaves_prev = np.zeros(self._t, dtype=np.int64)
        self._seen = np.zeros(self._t, dtype=bool)
        # JIT-warm the combined step at the EXACT (N, T, d_in) batch shape every real update uses (the batch is a
        # FIXED window of the last N transitions — see _train_on_batch), off the timed path, AFTER all valid
        # state is published, so the first real update on the serve thread hits the cached executable instead of
        # cold-compiling (the slow_act cause) or re-tracing on a changing B. A zero-reward dummy yields a defined
        # small step; its result is DISCARDED so the published params/opt_state stay pristine.
        Tw = max(1, self._t)
        Bw = self._n
        z3 = jnp.zeros((Bw, Tw, _D_IN), dtype=jnp.float32)
        z2 = jnp.zeros((Bw, Tw), dtype=jnp.float32)
        z1 = jnp.zeros((Bw,), dtype=jnp.float32)
        _wa, _wc, _woa, _woc, _wg, _waux = self._update_step(actor, critic, opt_a, opt_c, z3, z3, z2, z2, z1)
        jax.block_until_ready(_wg)   # force the compile to complete now, not lazily on the first real step.

    def observe(self, reward: float, info: Mapping[str, Any]) -> None:
        """Attach the realized PER-FORWARD reward to the PENDING transition (the contract's reward-of-previous-
        act), promoting it to AWAITING-NEXT (it now has the reward but still needs phi', which the NEXT act
        supplies). Update the running-mean (dashboard only). Before the first act there is no pending transition
        (the harness may observe ahead of the first act) -> ignored. A non-finite reward drops the pending
        transition rather than poisoning the gradient (ADR-0002: the watchdog owns loudness; the learner stays
        well-defined). The batched optax step fires from act() once enough COMPLETE transitions exist, so this
        method never touches the gradient."""
        if self._pending is None:
            return  # no sampled transition to credit yet.
        r = float(reward)
        if not np.isfinite(r):
            self._pending = None  # drop the dangling transition (no reward -> no usable sample).
            return
        phi, act, mask = self._pending
        self._pending = None
        self._awaiting = (phi, act, mask, r)  # has (phi, a, mask, r); the next act() attaches phi' and buffers it.
        self._b_sum += r
        self._b_cnt += 1

    def act(self, obs: Observation) -> Sequence[int]:
        """Advance the served-thread first-difference baselines, build the per-thread feature rows phi, FINALIZE
        any reward-attached AWAITING transition by using THIS forward's phi as its next-state phi' (appending the
        complete (phi, a, mask, r, phi') to the trajectory buffer and, every N forwards, firing the batched optax
        step), then run the NUMPY ACTOR forward (logits -> probs -> Bernoulli SAMPLE) from the actor mirror, apply
        the inflight==0 liveness override, and STASH the new sampled transition as pending. Cheap on the hot path:
        one O(T*d) numpy matvec, NO jax, no gradient (the optax step only fires on the period). Non-throwing —
        the per-thread arrays are grown to the live T on demand (the lab server can grow T past reset, and reset
        runs outside its lock), and defaulted reads keep a malformed/short feature frame safe (the watchdog owns
        loudness on the hot path, ADR-0002)."""
        T = self._t
        feats = obs.features
        self._ensure_capacity(T)   # live-T robustness: grow the per-thread arrays before any indexed read.
        inflight = _fit(np.asarray(feats.get("inflight", ()), dtype=np.float64), T)
        ready = _fit(np.asarray(feats.get("ready", ()), dtype=np.float64), T)
        msgs = _fit(np.asarray(feats.get("msgs", ()), dtype=np.float64), T).astype(np.int64)
        leaves = _fit(np.asarray(feats.get("leaves", ()), dtype=np.float64), T).astype(np.int64)
        served = [i for i in obs.served if 0 <= i < T]

        coalesce = self._coalesce(msgs, leaves, served, T)
        phi = self._build_phi(inflight, ready, coalesce)  # (T, d_in) float32 — THIS forward's state

        # finalize the awaiting transition: THIS phi is its next-state phi'. Append the complete transition and,
        # on the period, run the batched A2C step. (Done BEFORE sampling so the new sample becomes the next
        # transition cleanly.)
        if self._awaiting is not None:
            a_phi, a_act, a_mask, a_r = self._awaiting
            self._awaiting = None
            self._phi_buf.append(a_phi)
            self._phin_buf.append(phi)        # bootstrap next-state = this forward's phi
            self._act_buf.append(a_act)
            self._mask_buf.append(a_mask)
            self._rew_buf.append(a_r)
            if len(self._rew_buf) > self._max_batch:
                self._drop_oldest()
            if len(self._rew_buf) >= self._n:
                self._train_on_batch()

        # NUMPY ACTOR forward: probabilities + a Bernoulli sample per thread (the exploration). The mirror is
        # snapshotted by reference once (an atomic read) so a concurrent update/reset cannot tear it mid-matvec.
        probs = _np_actor_probs(self._np_actor, phi, self._hidden)   # (T,) float64
        u = self._rng.random(T)
        sampled_np = (u < probs).astype(np.float64)                  # Bernoulli(p): allow with probability p
        self._last_mean_prob = float(probs.mean()) if T else 0.0

        # liveness override (DENY-ONLY semantics): inflight==0 is an UNGATED forced flush -> a deny is a no-op,
        # force allow. active[t] = the thread actually acted (inflight>0) -> only those carry credit (the sampled
        # action at a no-op thread is overridden AND masked out of BOTH the actor and critic gradients).
        active = inflight > 0.0
        decision = np.where(active, sampled_np, 1.0)

        # stash the PENDING transition (state phi + sampled action + active mask) for the next observe to reward.
        self._pending = (
            phi,
            sampled_np.astype(np.float32),
            active.astype(np.float32),
        )
        return decision.astype(np.int64).tolist()

    def metrics(self) -> Mapping[str, float]:
        """Dashboard scalars exposing the LEARNED state. The three the brief names — the critic's mean value, the
        policy entropy, and the mean advantage — plus learning health: the last batch's gradient L2 norm, the
        number of optax updates, the current mean allow probability, the running-mean reward, and the buffer
        fill. Empty-safe."""
        b = self._b_sum / self._b_cnt if self._b_cnt else 0.0
        return {
            "mean_value": float(self._last_mean_value),
            "policy_entropy": float(self._last_entropy),
            "mean_advantage": float(self._last_mean_adv),
            "grad_norm": float(self._last_grad_norm),
            "updates": float(self._updates),
            "mean_allow_prob": float(self._last_mean_prob),
            "baseline": float(b),
            "buffer": float(len(self._rew_buf)),
        }

    # ---------------------------------------------------------------- internals

    def _ensure_capacity(self, T: int) -> None:
        """Grow the per-thread first-difference baselines to at least length T (the lab server can serve a tid
        beyond the reset-time n_threads, and calls reset() outside its lock so act() may run on a not-yet-sized
        array). New slots are un-seen with zero baselines — exactly the cold first-difference state, so a grown
        thread is treated as never-baselined (its first delta is the neutral 1.0). Idempotent + cheap (a no-op
        once sized)."""
        if self._seen.shape[0] >= T:
            return
        grow = T - self._seen.shape[0]
        self._msgs_prev = np.concatenate([self._msgs_prev, np.zeros(grow, dtype=np.int64)])
        self._leaves_prev = np.concatenate([self._leaves_prev, np.zeros(grow, dtype=np.int64)])
        self._seen = np.concatenate([self._seen, np.zeros(grow, dtype=bool)])

    def _coalesce(self, msgs: np.ndarray, leaves: np.ndarray, served: list[int], T: int) -> np.ndarray:
        """Served-thread first-difference of the CUMULATIVE counters -> instantaneous coalescing degree
        (Δleaves/Δmsgs). Only served & previously-seen threads carry a real delta; everyone else gets the
        neutral 1.0. Absent threads are NEVER differenced (their sentinel-0 reading would fake a negative delta)
        and their baselines are NOT advanced — the wire subtlety, honored (identical to reinforce)."""
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
        reinforce). All divisors are max(1, .)-guarded so phi is always finite (ADR-0002: the hot path stays
        total)."""
        D = float(self._d_ceil)
        headroom = np.maximum(1.0, D - inflight)               # room to release (>= 1)
        submit_pressure = ready / headroom                     # queued work vs. headroom to release it
        ready_backlog_norm = ready / float(self._k)            # K-normalized parked backlog
        inflight_saturation = inflight / D                     # proximity to the DENY-ONLY no-op
        bias = np.ones_like(inflight)
        phi = np.stack(                                        # (d_in, T) in canonical order, then transpose
            [submit_pressure, ready_backlog_norm, inflight_saturation, coalesce, bias], axis=0
        ).T
        return phi.astype(np.float32)                          # (T, d_in)

    def _train_on_batch(self) -> None:
        """The PERIODIC (decimated) optax (adam) A2C step on a FIXED window of the last N transitions: stack
        those N (phi, phi', a, mask, r), optionally standardize the reward for scale-stability, run the JIT'd
        combined step (jax — off the per-forward hot path; the advantage A = r + gamma*V(phi') - V(phi) is formed
        INSIDE the loss from the critic, so the critic's improvement feeds back each step), record the grad norm
        + the aux metrics, EXPORT the new actor to the numpy mirror the hot path reads, and CLEAR the buffer
        (on-policy: the recent trajectory is the right horizon). The batch is pinned to EXACTLY N rows (the most
        recent N — the buffer is drained each step so it holds ~N, but a concurrent-reset race could leave it
        short or long) so the jit'd step sees ONE fixed (N, T, d_in) shape and never re-traces on the serve
        thread (the slow_act cause); the reset() warmup compiles that exact shape. Total and defensive — too-few
        transitions to fill the window is a no-op (the step needs its fixed batch); a degenerate batch yields a
        finite gradient, never a throw."""
        N = self._n
        if len(self._rew_buf) < N:
            return   # not a full fixed-N window yet (a concurrent reset drained it) — skip; never a torn stack.
        phi = jnp.asarray(np.stack(self._phi_buf[-N:], axis=0))      # (N, T, d_in)  fixed-N window
        phin = jnp.asarray(np.stack(self._phin_buf[-N:], axis=0))    # (N, T, d_in)
        act = jnp.asarray(np.stack(self._act_buf[-N:], axis=0))      # (N, T)
        mask = jnp.asarray(np.stack(self._mask_buf[-N:], axis=0))    # (N, T)
        rew = np.asarray(self._rew_buf[-N:], dtype=np.float32)       # (N,)
        if self._standardize:
            mu = float(rew.mean())
            std = float(rew.std())
            if std > 1e-6:
                rew = (rew - np.float32(mu)) / np.float32(std)  # scale-stable reward -> stable advantage/critic
        rew_j = jnp.asarray(rew)

        self._actor, self._critic, self._opt_a, self._opt_c, gnorm, aux = self._update_step(
            self._actor, self._critic, self._opt_a, self._opt_c, phi, phin, act, mask, rew_j
        )
        mean_value, mean_entropy, mean_adv = aux
        self._last_grad_norm = float(gnorm)
        self._last_mean_value = float(mean_value)
        self._last_entropy = float(mean_entropy)
        self._last_mean_adv = float(mean_adv)
        self._updates += 1
        # EXPORT the actor to the numpy mirror (a single atomic rebind) so the next act() forward sees the new
        # policy. (The critic stays jax-only — it is never read on the hot path.)
        self._np_actor = _actor_to_numpy(self._actor, self._hidden)
        # clear the buffer: on-policy A2C trains on the just-collected trajectory only.
        self._phi_buf.clear()
        self._phin_buf.clear()
        self._act_buf.clear()
        self._mask_buf.clear()
        self._rew_buf.clear()

    def _drop_oldest(self) -> None:
        """Defensive buffer cap: drop the oldest completed transition if the buffer somehow exceeds max_batch
        before an update fires (keeps memory bounded on a pathological run; never throws)."""
        self._phi_buf.pop(0)
        self._phin_buf.pop(0)
        self._act_buf.pop(0)
        self._mask_buf.pop(0)
        self._rew_buf.pop(0)


def _sigmoid(z: float) -> float:
    """Plain scalar sigmoid for the cold-start metric (no JAX round-trip for one float)."""
    return 1.0 / (1.0 + np.exp(-z))


def _sigmoid_np(z: np.ndarray) -> np.ndarray:
    """Vectorized numerically-stable sigmoid on the hot path (no jax). The branchless-by-mask stable form
    avoids overflow for large |z| (exp of a positive argument only)."""
    out = np.empty_like(z)
    pos = z >= 0.0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def _bernoulli_entropy_scalar(z: float) -> float:
    """Scalar Bernoulli entropy from a logit, for the cold-start metric (no JAX round-trip for one float)."""
    p = _sigmoid(z)
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -(p * float(np.log(p)) + (1.0 - p) * float(np.log(1.0 - p)))


def _init_np_actor(hidden: int, init_allow_logit: float) -> dict[str, np.ndarray]:
    """The cold numpy ACTOR mirror (matching _init_head(.., zero_readout=True)) — present from __init__ so the
    hot path always has actor params to read even before the first reset. Linear: w=0, b=init_logit. Hidden:
    zero output read-out so the cold logit is the constant init_logit regardless of the (here zeroed) first
    layer."""
    if hidden <= 0:
        return {"w": np.zeros(_D_IN, dtype=np.float32),
                "b": np.float32(init_allow_logit)}
    return {"w1": np.zeros((_D_IN, hidden), dtype=np.float32), "b1": np.zeros(hidden, dtype=np.float32),
            "w2": np.zeros(hidden, dtype=np.float32), "b2": np.float32(init_allow_logit)}


def _actor_to_numpy(actor: dict[str, "jax.Array"], hidden: int) -> dict[str, np.ndarray]:
    """Export the jax ACTOR params to a fresh numpy dict (the hot-path mirror). One device->host copy per optax
    step / reset — off the per-forward path. The matvec on these arrays is the per-forward actor forward."""
    if hidden <= 0:
        return {"w": np.asarray(actor["w"], dtype=np.float32),
                "b": np.float32(np.asarray(actor["b"], dtype=np.float32))}
    return {"w1": np.asarray(actor["w1"], dtype=np.float32), "b1": np.asarray(actor["b1"], dtype=np.float32),
            "w2": np.asarray(actor["w2"], dtype=np.float32), "b2": np.asarray(actor["b2"], dtype=np.float32)}


def _np_actor_probs(np_actor: dict[str, np.ndarray], phi: np.ndarray, hidden: int) -> np.ndarray:
    """The per-forward NUMPY actor forward: per-thread allow PROBABILITY p[t] = sigmoid(logit[t]) from the actor
    numpy mirror. phi is (T, d_in); returns (T,) float64. Linear: phi @ w + b. Hidden: one tanh layer then a
    linear read-out — the numpy twin of _head over the actor params. O(T*d), no jax (so no cold compile, no
    re-trace, no device contention on the synchronous per-forward path)."""
    if hidden <= 0:
        logit = phi.astype(np.float64) @ np_actor["w"].astype(np.float64) + float(np_actor["b"])
    else:
        h = np.tanh(phi.astype(np.float64) @ np_actor["w1"].astype(np.float64) + np_actor["b1"].astype(np.float64))
        logit = h @ np_actor["w2"].astype(np.float64) + float(np_actor["b2"])
    return _sigmoid_np(logit)


def _fit(x: np.ndarray, t: int) -> np.ndarray:
    """Coerce a feature array to length T: truncate if long, zero-pad if short. Defensive so act() never throws
    on a malformed/empty feature list (ADR-0002: the per-forward path stays cheap and total; a zero-padded slot
    lands in the un-baselined / inflight==0 liveness path, i.e. neutral coalescing + force-allow)."""
    if x.shape[0] == t:
        return x
    out = np.zeros(t, dtype=np.float64)
    n = min(x.shape[0], t)
    if n:
        out[:n] = x[:n]
    return out


# Register additively into the FROZEN adapter.REGISTRY (one entry + one class — P2 seam discipline; the harness
# + dashboard discover methods here). setdefault so a re-import or a name clash never silently clobbers an
# existing registration.
REGISTRY.setdefault("a2c", A2CGate)
