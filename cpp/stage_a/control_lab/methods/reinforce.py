#!/usr/bin/env python3
"""
cpp/stage_a/control_lab/methods/reinforce.py — a REINFORCE policy-gradient issue-gate
(REINFORCEMENT-LEARNING family) candidate for the issue-gate control lab.

The RL counterpart to the bandit gates (threshold_bandit / contextual_bandit). Where a bandit selects from a
fixed arm set, this controller carries a small SHARED stochastic policy pi_theta and improves it by Monte-Carlo
policy-gradient (Williams' REINFORCE) within the single run, treating EACH FORWARD as one (state, action,
reward) transition. The policy is a TINY linear map from a per-thread feature row to a per-thread allow LOGIT;
a sigmoid turns the logit into an allow PROBABILITY and the gate is a Bernoulli SAMPLE of it (sampling IS the
exploration — no separate epsilon/UCB term). PARAMETER-SHARING collapses the per-thread credit into ONE theta:
every thread is one sample of the same policy, so a single forward yields T training examples and the gradient
is well-conditioned in a run that is only hundreds-to-thousands of forwards over a ~4s box (the central
challenge — convergence in the budget — is why the model is kept linear and the optax step is BATCHED, never
a backward every forward).

JAX FOR THE GRADIENT, NUMPY FOR THE HOT DECISION (the lab-server reality). This controller runs IN the
eval-server process, alongside that process's own JAX inference server, under an XLA single-thread pin, and the
gate decision is SYNCHRONOUS on the per-forward boundary with a hard 50ms watchdog. A jax forward on THAT path
is the wrong tool: its first call cold-compiles (blowing the deadline), it re-traces on a changing T, and it
shares the device with the inference server. So the per-forward act() POLICY FORWARD is done in NUMPY from a
NUMPY MIRROR of the policy params (a cheap O(T*d) matvec, d=5) — no jax on the hot tick at all — while the
LEARNING (the optax adam step) keeps jax, runs DECIMATED (once per N forwards, off the hot path), and the new
params are EXPORTED back to the numpy mirror after each step. jax for the gradient; numpy for the decision.
This is the same pure-numpy hot path every non-RL method in this package uses; only the gradient is jax.

Mechanism (features -> stochastic gate). Each forward, per thread t, a 5-d feature row phi[t] is read off the
wire (D = ctx.d_ceiling, K = ctx.k_per_thread — the capacity normalizers the feature wire omits):

    phi[t] = [ submit_pressure      = ready / max(1, D - inflight),   # queued work vs. headroom to release it
               ready_backlog_norm   = ready / max(1, K),              # K-normalized parked-at-leaf backlog
               inflight_saturation  = inflight / max(1, D),           # proximity to the DENY-ONLY no-op (j -> D)
               coalesce_degree_inst = (Δleaves) / max(1, Δmsgs),      # rows-per-message achieved (windowed, served-diff)
               1.0 ]                                                  # bias

The policy is the shared linear-logistic map (theta is a 5-vector, the bias weight is theta[-1]):

    logit[t] = theta · phi[t]            p[t] = sigmoid(logit[t])            a[t] ~ Bernoulli(p[t])

so a[t]=1 means ALLOW. The forward (logit -> prob -> sampled action) is a NUMPY matvec + a numpy Bernoulli draw
and is the ONLY work on the per-forward critical path; it is O(T*d) with d=5, well inside the deadline, and
NEVER touches jax (so it cannot cold-compile, re-trace, or contend the inference device).

Cold start = allow-leaning baseline (the RL analog of the bandit's all-allow anchor arm). theta is zero EXCEPT
the bias weight, initialized to `init_allow_logit` (>0), so the cold policy samples allow with high probability
(sigmoid(init_allow_logit)) — near the all-allow control arm, the safety floor for the un-warmed warm-up
forwards — while still being STOCHASTIC (genuine exploration from forward one). REINFORCE then moves theta
wherever the reward leads. (Initializing flat at 0.5 would explore harder but risk throttling the cold plant
before any reward is seen; the positive bias is the honest short-run compromise, exposed as a knob.)

Liveness override (DENY-ONLY gate semantics — the runner's effective gate is `inflight < D && allow` and the
forced flush at inflight==0 is UNGATED, so a deny is a NO-OP there): inflight[t]==0 -> force allow. The SAMPLED
action at such a thread is OVERRIDDEN. Crucially this also gates CREDIT: a thread whose gate was a forced no-op
did not causally affect the reward, so its sampled log-prob is MASKED OUT of the policy-gradient sum (only
threads that actually acted — inflight>0 at sample time — carry credit/blame). Training on a no-op sample would
inject pure noise; masking it is the faithful handling of "the sampled action there is overridden."

The learner (observe(reward) -> periodic batch ascent). The reward fed to observe() is the pool's PER-FORWARD
THROUGHPUT CONTRIBUTION (forward_rows — the coalescing achieved; HIGHER IS BETTER) — the RL reward. Per the
FROZEN contract, observe(r) delivers the reward of the PREVIOUS act, so the just-sampled transition is held
PENDING (its phi, sampled action, active-mask) until the next observe attaches its realized reward; the
completed (phi, a, mask, r) transition then joins the trajectory buffer. EVERY N forwards (not every forward —
that is the efficiency lever) one optax (adam) gradient ASCENT step is taken on the accumulated batch,
maximizing the REINFORCE objective with a BASELINE b for variance reduction:

    J(theta) = E_f[ (R_f - b) * sum_{t active} log pi_theta(a_{f,t} | phi_{f,t}) ]

b is the RUNNING-MEAN per-forward reward (updated online in observe), subtracted as the advantage A_f = R_f - b
so the gradient reinforces an action only insofar as its forward beat the average — the standard REINFORCE
baseline, which removes the reward's magnitude/scale from the gradient and sharply lowers its variance without
biasing it. The optax step (the only backward) is JIT-compiled and runs once per N forwards on the small batch;
the buffer is cleared after each step (Monte-Carlo on the recent trajectory, the appropriate horizon for a
short non-stationary run). gradient ASCENT = descend the NEGATED objective (optax minimizes). After each step
the updated jax params are EXPORTED to the numpy mirror the hot path reads.

JIT warmup (reset, off the timed path). The optax step is JIT-compiled on a DUMMY fixed-shape batch during
reset() — BEFORE the wall box opens and BEFORE the first real update — so the first real learning step pays no
cold-compile spike on the serve thread. The hot act() is numpy and never compiles at all.

Wire subtlety (honored). coalesce_degree_inst first-differences the CUMULATIVE counters msgs and leaves.
lab_server builds each length-T feature list fresh as [0]*T and fills ONLY the served tids, so a thread ABSENT
from a forward reads a SENTINEL 0, not its true cumulative. The per-thread msgs/leaves baselines are therefore
advanced ONLY for obs.served (against a per-thread baseline, and only once a thread has been SEEN before) — an
absent thread is never first-differenced (its sentinel 0 would manufacture a spurious negative delta) and its
baseline is not touched. An un-baselined or quiet thread reads coalesce_degree_inst = 1.0 (one leaf per
message — the neutral "no coalescing measured yet" value). The other features are read from INSTANTANEOUS
gauges (ready, inflight), never differenced, so a sentinel-0 there is a harmless zero, not a fabricated delta.

Live-T robustness (the lab server grows T past the trial ctx). The server LAZILY grows its gate-vector length
when a served tid exceeds the reset-time n_threads (lab_server._serve_batch), and it calls reset() OUTSIDE its
lock while the serve thread is already acting — so act() can be entered on a thread whose tid exceeds the
per-thread arrays' current length, or even mid-reset. Every per-thread array is therefore GROWN-ON-DEMAND to
the live T at the top of act() (and the numpy params mirror is always present, set in __init__), so the hot
path is total under both a grown T and a concurrent reset (ADR-0002: the watchdog owns loudness; the hot path
stays well-defined and never throws).

RL family: reset() COLD-STARTS the learner (re-initializes theta + the adam moment state, refreshes the numpy
mirror, JIT-warms the update step, clears the trajectory buffer, the pending transition, the running-mean
baseline, and the first-difference baselines) — nothing survives a trial. observe() attaches the realized
reward and, every N forwards, drives the optax ascent step. act() samples the gate from the CURRENT policy via
the numpy mirror. metrics() exposes the mean allow probability, the last gradient norm, the baseline b, and the
update count. Knobs: lr, update period N, hidden size (0 = linear; >0 = one hidden tanh layer of that width),
the cold-start allow logit, and the optional advantage standardization.

Run the unit gate pinned + bounded:
    PYTHONPATH=cpp/stage_a /home/bork/w/vdc/venvs/generic/bin/python -m pytest tests/test_method_reinforce.py -q

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

# The fixed feature layout (bias last). The realized input dimension d is DERIVED from this tuple, never
# hardcoded (ADR-0002 single-source-of-truth: the policy net's input width is len(_FEATURES), the consumers
# derive their view). The order here is the order _build_phi assembles the columns in.
_FEATURES: tuple[str, ...] = (
    "submit_pressure",
    "ready_backlog_norm",
    "inflight_saturation",
    "coalesce_degree_inst",
    "bias",
)
_D_IN = len(_FEATURES)  # 5; the policy's input dimension, derived from the layout


# ----------------------------------------------------------------------------- JIT'd policy core (pure)
# The LEARNING path only (jax.grad + optax). These are module-level pure functions so JAX caches ONE compiled
# artifact across instances of the same (hidden,) static shape, rather than re-tracing per controller. `hidden`
# is a static argument (it changes the net's structure / parameter pytree), so a trace is keyed on it. NB: NONE
# of these run on the per-forward critical path — the hot decision is the numpy forward below.


def _init_params(key: "jax.Array", hidden: int, init_allow_logit: float) -> dict[str, "jax.Array"]:
    """Initialize the tiny policy pytree. Linear (hidden==0): a single weight vector `w` (d_in,) and scalar
    bias `b`, both zero EXCEPT b = init_allow_logit (the allow-leaning cold start = the safety floor). One
    hidden tanh layer (hidden>0): small-random W1/b1 (Glorot-ish, *0.1) into the hidden layer, and a
    zero-initialized output (w2=0, b2=init_allow_logit) so the cold policy is STILL the allow-leaning constant
    logit regardless of the random first layer (the net only departs from baseline as the output learns)."""
    if hidden <= 0:
        # linear-logistic policy: logit = w·phi (phi already carries the bias column, but we keep an explicit
        # scalar bias `b` so the cold-start allow-lean is a clean single initializer rather than buried in w).
        w = jnp.zeros((_D_IN,), dtype=jnp.float32)
        b = jnp.asarray(init_allow_logit, dtype=jnp.float32)
        return {"w": w, "b": b}
    # one-hidden-layer tanh MLP. Output layer zero so the cold logit is exactly the bias (allow-leaning).
    k1, _ = jax.random.split(key)
    w1 = jax.random.normal(k1, (_D_IN, hidden), dtype=jnp.float32) * 0.1
    b1 = jnp.zeros((hidden,), dtype=jnp.float32)
    w2 = jnp.zeros((hidden,), dtype=jnp.float32)
    b2 = jnp.asarray(init_allow_logit, dtype=jnp.float32)
    return {"w1": w1, "b1": b1, "w2": w2, "b2": b2}


def _logits(params: dict[str, "jax.Array"], phi: "jax.Array", hidden: int) -> "jax.Array":
    """Per-thread allow logit from the shared policy. phi is (T, d_in). Linear: phi @ w + b. Hidden: a single
    tanh layer then a linear read-out. Returns (T,). Used INSIDE the loss (the gradient path) only."""
    if hidden <= 0:
        return phi @ params["w"] + params["b"]
    h = jnp.tanh(phi @ params["w1"] + params["b1"])  # (T, hidden)
    return h @ params["w2"] + params["b2"]            # (T,)


def _neg_reinforce_loss(
    params: dict[str, "jax.Array"],
    phi: "jax.Array",       # (B, T, d_in)
    act: "jax.Array",       # (B, T) sampled actions in {0,1}
    mask: "jax.Array",      # (B, T) 1.0 where the thread actually acted (inflight>0), else 0.0
    adv: "jax.Array",       # (B,) per-forward advantage R_f - b
    hidden: int,
) -> "jax.Array":
    """The NEGATED REINFORCE objective (optax minimizes, so minimizing -J ascends J). For each forward f the
    score is adv[f] * sum_{t active} log pi(a[f,t] | phi[f,t]); the loss is the negative mean over the batch.
    log pi for a Bernoulli(sigmoid(logit)) is the negative binary-cross-entropy between the sampled action and
    the policy prob — computed in a numerically-stable log-sigmoid form. Masked (no-op) threads contribute 0."""
    flat = phi.reshape(-1, phi.shape[-1])                  # (B*T, d_in)
    logit = _logits(params, flat, hidden).reshape(act.shape)  # (B, T)
    # log pi(a) = a*log(sigmoid(z)) + (1-a)*log(1-sigmoid(z)) = a*logsig(z) + (1-a)*logsig(-z), stable form.
    log_p_allow = jax.nn.log_sigmoid(logit)
    log_p_deny = jax.nn.log_sigmoid(-logit)
    log_pi = act * log_p_allow + (1.0 - act) * log_p_deny  # (B, T)
    per_forward = jnp.sum(log_pi * mask, axis=1)           # (B,) sum over ACTIVE threads (param-sharing)
    return -jnp.mean(adv * per_forward)                    # negate -> ascent under a minimizer


def _make_update_step(tx: optax.GradientTransformation, hidden: int) -> Any:
    """Build the JIT'd optax (adam) ASCENT step, closing over the transform `tx` and the static `hidden` width
    (mirrors az/optimizer.make_update: the transform is a pytree of FUNCTIONS, not arrays, so it is CAPTURED in
    the closure rather than passed as a traced argument — passing it positionally would make JAX try to trace
    `tx.init` as an abstract array). The returned closure (params, opt_state, phi, act, mask, adv) ->
    (new_params, new_opt_state, grad_norm) takes the grad of the negated objective and applies one update; the
    global grad norm is surfaced as a learning-health metric. The whole step is the periodic batch update; it
    is NOT on the per-forward critical path (it runs decimated, off the hot tick)."""

    @jax.jit
    def _step(
        params: dict[str, "jax.Array"],
        opt_state: optax.OptState,
        phi: "jax.Array",
        act: "jax.Array",
        mask: "jax.Array",
        adv: "jax.Array",
    ) -> tuple[dict[str, "jax.Array"], optax.OptState, "jax.Array"]:
        grads = jax.grad(_neg_reinforce_loss)(params, phi, act, mask, adv, hidden)
        updates, new_opt_state = tx.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        gnorm = optax.tree.norm(grads)  # global L2 grad norm (the non-deprecated optax.global_norm)
        return new_params, new_opt_state, gnorm

    return _step


class ReinforceGate:
    """A REINFORCE policy-gradient issue gate (REINFORCEMENT-LEARNING family). A TINY SHARED stochastic policy
    pi_theta (linear-logistic, or one hidden tanh layer) maps a per-thread feature row to an allow logit;
    sigmoid -> Bernoulli SAMPLE is the per-thread gate (sampling = exploration). Each forward is one (s, a, r)
    transition (parameter-sharing: T samples / forward); the realized PER-FORWARD reward (forward_rows, higher
    is better) drives an optax adam ASCENT step taken EVERY N forwards on the accumulated batch, maximizing
    E[(R - b) * sum log pi] with a running-mean baseline b. The per-forward forward is a NUMPY matvec from a
    numpy mirror of theta (O(T*d), d=5) — jax never touches the hot path; the batched optax step (jax) is the
    periodic, decimated update and exports its result to the mirror. inflight==0 force-allows AND masks that
    thread's log-prob out of the gradient (a forced no-op carries no credit). Cold-started each trial."""

    family: Family = "rl"

    def __init__(
        self,
        lr: float = 0.05,
        update_period: int = 16,
        hidden: int = 0,
        init_allow_logit: float = 2.0,
        standardize_adv: bool = True,
        max_batch: int = 256,
    ) -> None:
        # fail loud (ADR-0002): degenerate hyperparameters are a CONSTRUCTION error, surfaced at build time on
        # the ctor, never a silent surprise on the per-forward hot path.
        if lr <= 0.0:
            raise ValueError(f"ReinforceGate: lr must be > 0, got {lr}")
        if update_period < 1:
            raise ValueError(f"ReinforceGate: update_period N must be >= 1, got {update_period}")
        if hidden < 0:
            raise ValueError(f"ReinforceGate: hidden must be >= 0 (0 = linear policy), got {hidden}")
        if not np.isfinite(init_allow_logit):
            raise ValueError(f"ReinforceGate: init_allow_logit must be finite, got {init_allow_logit}")
        if max_batch < 1:
            raise ValueError(f"ReinforceGate: max_batch must be >= 1, got {max_batch}")

        self._lr = float(lr)
        self._n = int(update_period)
        self._hidden = int(hidden)
        self._init_logit = float(init_allow_logit)
        self._standardize = bool(standardize_adv)
        self._max_batch = int(max_batch)
        self.name = f"reinforce_lr{self._lr:g}_N{self._n}_h{self._hidden}"

        # the optax transform is built once (the moment pytree lives in self._opt_state, re-init in reset());
        # the JIT'd ascent step closes over it + the static hidden width, built once here (not per update).
        self._tx = optax.adam(self._lr)
        self._update_step = _make_update_step(self._tx, self._hidden)

        # --- per-run learner state (sized/cleared in reset) ---
        self._t = 1
        self._d_ceil = 1
        self._k = 1
        self._params: dict[str, "jax.Array"] = {}
        self._opt_state: optax.OptState = None  # type: ignore[assignment]
        # NUMPY MIRROR of the policy params — the ONLY thing the hot path reads. Always present (set here AND
        # at reset()/each update), so a concurrent reset never leaves act() reading a missing key. Cold-init
        # to the allow-leaning constant logit (w=0, b=init_logit) so even a pre-reset act is the safety floor.
        self._np_params: dict[str, np.ndarray] = _init_np_params(self._hidden, self._init_logit)
        # the hot-path Bernoulli RNG is a plain numpy Generator (reseeded from ctx.seed in reset); the jax key
        # below is used ONLY for the (off-hot-path) param init.
        self._rng = np.random.default_rng(0)
        self._key = jax.random.PRNGKey(0)
        # trajectory buffer of COMPLETED transitions (reward attached), drained every N forwards.
        self._phi_buf: list[np.ndarray] = []     # each (T, d_in)
        self._act_buf: list[np.ndarray] = []     # each (T,) in {0,1}
        self._mask_buf: list[np.ndarray] = []    # each (T,) active mask (inflight>0)
        self._rew_buf: list[float] = []          # each scalar per-forward reward
        # the PENDING transition: act() sampled it; the NEXT observe() attaches its reward (the contract's
        # "reward of the PREVIOUS act") and moves it into the buffer.
        self._pending: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
        # running-mean reward baseline b (count-based; the brief's "running-mean reward").
        self._b_sum = 0.0
        self._b_cnt = 0
        # cumulative-counter baselines for the served-thread first-difference (the wire subtlety).
        self._msgs_prev = np.zeros(1, dtype=np.int64)
        self._leaves_prev = np.zeros(1, dtype=np.int64)
        self._seen = np.zeros(1, dtype=bool)
        # dashboard scalars.
        self._updates = 0
        self._last_grad_norm = 0.0
        self._last_mean_prob = float(_sigmoid(self._init_logit))

    def reset(self, ctx: TrialContext) -> None:
        """COLD-START a fresh trial: capture the geometry (T, D, K) the features need, RE-INITIALIZE the policy
        params + the adam moment state (nothing learned survives a trial), refresh the NUMPY MIRROR the hot path
        reads, JIT-WARM the optax step on a dummy batch (so the first real update pays no cold compile on the
        serve thread), and clear the trajectory buffer, the pending transition, the baseline accumulator, and
        the first-difference baselines. The RNG is reseeded from ctx.seed so a trial's sampling is reproducible
        per the lab's seed.

        Order note (ADR-0002, the lab-server race): set_trial calls this OUTSIDE its lock while the serve thread
        may already be acting, so the per-thread arrays + the numpy mirror are published as the LAST writes,
        each a single atomic rebind; an act() that interleaves reads either the prior trial's consistent state
        or the freshly-published one, never a torn half. (act() also grows the arrays on demand, so even a
        smaller prior length is safe.)"""
        self._t = int(ctx.n_threads)
        self._d_ceil = max(1, int(ctx.d_ceiling))
        self._k = max(1, int(ctx.k_per_thread))
        # reseed both RNGs from the trial seed (reproducible exploration + reproducible param init).
        seed = int(ctx.seed) & 0x7FFFFFFF
        self._key = jax.random.PRNGKey(seed)
        self._key, init_key = jax.random.split(self._key)
        params = _init_params(init_key, self._hidden, self._init_logit)
        opt_state = self._tx.init(params)
        # PUBLISH the valid initialized params + an EMPTY buffer FIRST, BEFORE the (GIL-releasing) warmup
        # compile below. set_trial calls reset() outside its lock while the serve thread runs observe()/act();
        # observe() can fire _train_on_batch() which reads self._params — so self._params must be a valid pytree
        # (never the stale {} or a torn half) the instant the trial goes active, and the buffer must be empty so
        # no stale transition from the prior trial drives a step against the fresh params (ADR-0002: the learner
        # stays well-defined under the concurrent reset).
        self._params = params
        self._opt_state = opt_state
        self._rng = np.random.default_rng(seed)
        self._phi_buf = []
        self._act_buf = []
        self._mask_buf = []
        self._rew_buf = []
        self._pending = None
        self._b_sum = 0.0
        self._b_cnt = 0
        self._updates = 0
        self._last_grad_norm = 0.0
        self._last_mean_prob = float(_sigmoid(self._init_logit))
        # JIT-warm the optax step at the EXACT (N, T, d_in) batch shape every real update uses (the batch is a
        # FIXED window of the last N transitions — see _train_on_batch), off the timed path, so the first real
        # update on the serve thread hits the cached executable instead of cold-compiling (the slow_act cause)
        # or re-tracing on a changing B. A zero-advantage dummy yields a zero-gradient no-op step; its result is
        # DISCARDED so the published params/opt_state stay pristine.
        Tw = max(1, self._t)
        Bw = self._n
        _wp, _ws, _wg = self._update_step(
            params, opt_state,
            jnp.zeros((Bw, Tw, _D_IN), dtype=jnp.float32),
            jnp.zeros((Bw, Tw), dtype=jnp.float32),
            jnp.zeros((Bw, Tw), dtype=jnp.float32),
            jnp.zeros((Bw,), dtype=jnp.float32),
        )
        jax.block_until_ready(_wg)   # force the compile to complete now, not lazily on the first real step.
        # PUBLISH the numpy mirror + the sized per-thread arrays LAST (each an atomic single rebind).
        self._np_params = _params_to_numpy(self._params, self._hidden)
        self._msgs_prev = np.zeros(self._t, dtype=np.int64)
        self._leaves_prev = np.zeros(self._t, dtype=np.int64)
        self._seen = np.zeros(self._t, dtype=bool)

    def observe(self, reward: float, info: Mapping[str, Any]) -> None:
        """Attach the realized PER-FORWARD reward to the PENDING transition (the contract's reward-of-previous-
        act), update the running-mean baseline b, and append the completed (phi, a, mask, r) to the trajectory
        buffer. Every N completed forwards, take the batched optax ASCENT step and clear the buffer. Before the
        first act there is no pending transition (the harness may observe ahead of the first act) -> ignored. A
        non-finite reward is dropped rather than poisoning the gradient (ADR-0002: the watchdog owns loudness;
        the learner stays well-defined)."""
        if self._pending is None:
            return  # no sampled transition to credit yet.
        r = float(reward)
        if not np.isfinite(r):
            self._pending = None  # drop the dangling transition with it (no reward -> no usable sample).
            return
        phi, act, mask = self._pending
        self._pending = None
        self._phi_buf.append(phi)
        self._act_buf.append(act)
        self._mask_buf.append(mask)
        self._rew_buf.append(r)
        # running-mean baseline (count-based, parameter-free — the brief's running-mean reward).
        self._b_sum += r
        self._b_cnt += 1
        # cap the buffer defensively so a pathological run (no/sparse updates) can't grow it unbounded.
        if len(self._rew_buf) > self._max_batch:
            self._drop_oldest()
        if len(self._rew_buf) >= self._n:
            self._train_on_batch()

    def act(self, obs: Observation) -> Sequence[int]:
        """Advance the served-thread first-difference baselines, build the per-thread feature rows phi, run the
        NUMPY policy forward (logits -> probs -> Bernoulli SAMPLE) from the numpy params mirror, apply the
        inflight==0 liveness override, and STASH the sampled transition as pending (the next observe attaches its
        reward). Cheap: one O(T*d) numpy matvec, NO jax, NO gradient. Non-throwing — the per-thread arrays are
        grown to the live T on demand (the lab server can grow T past reset, and reset runs outside its lock),
        and defaulted reads keep a malformed/short feature frame safe (the watchdog owns loudness on the hot
        path, ADR-0002)."""
        T = self._t
        feats = obs.features
        self._ensure_capacity(T)   # live-T robustness: grow the per-thread arrays before any indexed read.
        inflight = _fit(np.asarray(feats.get("inflight", ()), dtype=np.float64), T)
        ready = _fit(np.asarray(feats.get("ready", ()), dtype=np.float64), T)
        msgs = _fit(np.asarray(feats.get("msgs", ()), dtype=np.float64), T).astype(np.int64)
        leaves = _fit(np.asarray(feats.get("leaves", ()), dtype=np.float64), T).astype(np.int64)
        served = [i for i in obs.served if 0 <= i < T]

        coalesce = self._coalesce(msgs, leaves, served, T)
        phi = self._build_phi(inflight, ready, coalesce)  # (T, d_in) float32

        # NUMPY policy forward: probabilities + a Bernoulli sample per thread (the exploration). The mirror is
        # snapshotted by reference once (an atomic read) so a concurrent update/reset cannot tear it mid-matvec.
        probs = _np_policy_probs(self._np_params, phi, self._hidden)   # (T,) float64
        u = self._rng.random(T)
        sampled_np = (u < probs).astype(np.float64)                    # Bernoulli(p): allow with probability p
        self._last_mean_prob = float(probs.mean()) if T else 0.0

        # liveness override (DENY-ONLY semantics): inflight==0 is an UNGATED forced flush -> a deny is a no-op,
        # force allow. active[t] = the thread actually acted (inflight>0) -> only those carry credit (the
        # sampled action at a no-op thread is overridden AND masked out of the gradient).
        active = inflight > 0.0
        decision = np.where(active, sampled_np, 1.0)

        # stash the PENDING transition (sampled action + active mask) for the next observe to reward.
        self._pending = (
            phi,
            sampled_np.astype(np.float32),
            active.astype(np.float32),
        )
        return decision.astype(np.int64).tolist()

    def metrics(self) -> Mapping[str, float]:
        """Dashboard scalars exposing the LEARNED state: the last forward's mean allow probability (the policy's
        current behavior), the last batch's gradient L2 norm (learning health), the running-mean baseline b, the
        number of optax updates taken, and the pending trajectory-buffer fill. Empty-safe."""
        b = self._b_sum / self._b_cnt if self._b_cnt else 0.0
        return {
            "mean_allow_prob": float(self._last_mean_prob),
            "grad_norm": float(self._last_grad_norm),
            "baseline": float(b),
            "updates": float(self._updates),
            "buffer": float(len(self._rew_buf)),
        }

    # ---------------------------------------------------------------- internals

    def _ensure_capacity(self, T: int) -> None:
        """Grow the per-thread first-difference baselines to at least length T (the lab server can serve a tid
        beyond the reset-time n_threads, and calls reset() outside its lock so act() may run on a not-yet-sized
        array). New slots are un-seen with zero baselines — exactly the cold first-difference state, so a
        grown thread is treated as never-baselined (its first delta is the neutral 1.0). Idempotent + cheap
        (a no-op once sized)."""
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
        delta) and their baselines are NOT advanced — the wire subtlety, honored."""
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
        """Assemble the (T, d_in) float32 feature matrix in the canonical _FEATURES order. All divisors are
        max(1, .)-guarded so phi is always finite (ADR-0002: the hot path stays total)."""
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
        """The PERIODIC (decimated) optax (adam) ASCENT step on a FIXED window of the last N transitions: stack
        those N, form the advantage A_f = R_f - b (running-mean baseline; optionally standardized for
        scale-stability), run the JIT'd update step (jax — off the per-forward hot path), record the grad norm,
        EXPORT the new params to the numpy mirror the hot path reads, and CLEAR the buffer (Monte-Carlo on the
        recent trajectory). The batch is pinned to EXACTLY N rows (the most recent N — the buffer is drained
        each step so it holds ~N, but a concurrent-reset race could leave it short or long) so the jit'd step
        sees ONE fixed (N, T, d_in) shape and never re-traces on the serve thread (the slow_act cause); the
        reset() warmup compiles that exact shape. Total and defensive — too-few transitions to fill the window
        is a no-op (the step needs its fixed batch); an all-zero advantage yields a zero gradient, never a
        throw."""
        N = self._n
        if len(self._rew_buf) < N:
            return   # not a full fixed-N window yet (a concurrent reset drained it) — skip; never a torn stack.
        phi = jnp.asarray(np.stack(self._phi_buf[-N:], axis=0))     # (N, T, d_in)  fixed-N window
        act = jnp.asarray(np.stack(self._act_buf[-N:], axis=0))     # (N, T)
        mask = jnp.asarray(np.stack(self._mask_buf[-N:], axis=0))   # (N, T)
        rew = np.asarray(self._rew_buf[-N:], dtype=np.float32)      # (N,)
        b = self._b_sum / self._b_cnt if self._b_cnt else 0.0
        adv = rew - np.float32(b)
        if self._standardize:
            std = float(adv.std())
            if std > 1e-6:
                adv = adv / np.float32(std)                    # scale-stable gradient (variance reduction)
        adv_j = jnp.asarray(adv)

        self._params, self._opt_state, gnorm = self._update_step(
            self._params, self._opt_state, phi, act, mask, adv_j
        )
        self._last_grad_norm = float(gnorm)
        self._updates += 1
        # EXPORT to the numpy mirror (a single atomic rebind) so the next act() forward sees the new policy.
        self._np_params = _params_to_numpy(self._params, self._hidden)
        # clear the buffer: REINFORCE is Monte-Carlo on the just-collected trajectory.
        self._phi_buf.clear()
        self._act_buf.clear()
        self._mask_buf.clear()
        self._rew_buf.clear()

    def _drop_oldest(self) -> None:
        """Defensive buffer cap: drop the oldest pending transition if the buffer somehow exceeds max_batch
        before an update fires (keeps memory bounded on a pathological run; never throws)."""
        self._phi_buf.pop(0)
        self._act_buf.pop(0)
        self._mask_buf.pop(0)
        self._rew_buf.pop(0)


def _sigmoid(z: float) -> float:
    """Plain scalar sigmoid for the cold-start metric (no JAX round-trip needed for one float)."""
    return 1.0 / (1.0 + np.exp(-z))


def _sigmoid_np(z: np.ndarray) -> np.ndarray:
    """Vectorized numerically-stable sigmoid on the hot path (no jax). The branchless stable form avoids
    overflow for large |z| (exp of a positive argument only)."""
    out = np.empty_like(z)
    pos = z >= 0.0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def _init_np_params(hidden: int, init_allow_logit: float) -> dict[str, np.ndarray]:
    """The cold numpy mirror (matching _init_params' cold state) — present from __init__ so the hot path always
    has params to read even before the first reset. Linear: w=0, b=init_logit. Hidden: zero output read-out so
    the cold logit is the constant init_logit regardless of the (here zeroed) first layer."""
    if hidden <= 0:
        return {"w": np.zeros(_D_IN, dtype=np.float32),
                "b": np.float32(init_allow_logit)}
    return {"w1": np.zeros((_D_IN, hidden), dtype=np.float32), "b1": np.zeros(hidden, dtype=np.float32),
            "w2": np.zeros(hidden, dtype=np.float32), "b2": np.float32(init_allow_logit)}


def _params_to_numpy(params: dict[str, "jax.Array"], hidden: int) -> dict[str, np.ndarray]:
    """Export the jax policy params to a fresh numpy dict (the hot-path mirror). One device->host copy per
    optax step / reset — off the per-forward path. The matvec on these arrays is the per-forward forward."""
    if hidden <= 0:
        return {"w": np.asarray(params["w"], dtype=np.float32),
                "b": np.float32(np.asarray(params["b"], dtype=np.float32))}
    return {"w1": np.asarray(params["w1"], dtype=np.float32), "b1": np.asarray(params["b1"], dtype=np.float32),
            "w2": np.asarray(params["w2"], dtype=np.float32), "b2": np.asarray(params["b2"], dtype=np.float32)}


def _np_policy_probs(np_params: dict[str, np.ndarray], phi: np.ndarray, hidden: int) -> np.ndarray:
    """The per-forward NUMPY policy forward: per-thread allow PROBABILITY p[t] = sigmoid(logit[t]) from the
    numpy params mirror. phi is (T, d_in); returns (T,) float64. Linear: phi @ w + b. Hidden: one tanh layer
    then a linear read-out — the numpy twin of _logits. O(T*d), no jax (so no cold compile, no re-trace, no
    device contention on the synchronous per-forward path)."""
    if hidden <= 0:
        logit = phi.astype(np.float64) @ np_params["w"].astype(np.float64) + float(np_params["b"])
    else:
        h = np.tanh(phi.astype(np.float64) @ np_params["w1"].astype(np.float64) + np_params["b1"].astype(np.float64))
        logit = h @ np_params["w2"].astype(np.float64) + float(np_params["b2"])
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
REGISTRY.setdefault("reinforce", ReinforceGate)
