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

Mechanism (features -> stochastic gate). Each forward, per thread t, a 5-d feature row phi[t] is read off the
wire (D = ctx.d_ceiling, K = ctx.k_per_thread — the capacity normalizers the feature wire omits):

    phi[t] = [ submit_pressure      = ready / max(1, D - inflight),   # queued work vs. headroom to release it
               ready_backlog_norm   = ready / max(1, K),              # K-normalized parked-at-leaf backlog
               inflight_saturation  = inflight / max(1, D),           # proximity to the DENY-ONLY no-op (j -> D)
               coalesce_degree_inst = (Δleaves) / max(1, Δmsgs),      # rows-per-message achieved (windowed, served-diff)
               1.0 ]                                                  # bias

The policy is the shared linear-logistic map (theta is a 5-vector, the bias weight is theta[-1]):

    logit[t] = theta · phi[t]            p[t] = sigmoid(logit[t])            a[t] ~ Bernoulli(p[t])

so a[t]=1 means ALLOW. The forward (logit -> prob -> sampled action -> log-prob) is JIT-compiled and is the
ONLY work on the per-forward critical path; it is O(T*d) with d=5, well inside the 50ms per-decision deadline.

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
short non-stationary run). gradient ASCENT = descend the NEGATED objective (optax minimizes).

Wire subtlety (honored). coalesce_degree_inst first-differences the CUMULATIVE counters msgs and leaves.
lab_server builds each length-T feature list fresh as [0]*T and fills ONLY the served tids, so a thread ABSENT
from a forward reads a SENTINEL 0, not its true cumulative. The per-thread msgs/leaves baselines are therefore
advanced ONLY for obs.served (against a per-thread baseline, and only once a thread has been SEEN before) — an
absent thread is never first-differenced (its sentinel 0 would manufacture a spurious negative delta) and its
baseline is not touched. An un-baselined or quiet thread reads coalesce_degree_inst = 1.0 (one leaf per
message — the neutral "no coalescing measured yet" value). The other features are read from INSTANTANEOUS
gauges (ready, inflight), never differenced, so a sentinel-0 there is a harmless zero, not a fabricated delta.

RL family: reset() COLD-STARTS the learner (re-initializes theta + the adam moment state, clears the trajectory
buffer, the pending transition, the running-mean baseline, and the first-difference baselines) — nothing
survives a trial. observe() attaches the realized reward and, every N forwards, drives the optax ascent step.
act() samples the gate from the CURRENT policy. metrics() exposes the mean allow probability, the last gradient
norm, the baseline b, and the update count. Knobs: lr, update period N, hidden size (0 = linear; >0 = one
hidden tanh layer of that width), the cold-start allow logit, and the optional advantage standardization.

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
# The hot path. These are module-level pure functions so JAX caches ONE compiled artifact across instances of
# the same (hidden,) static shape, rather than re-tracing per controller. `hidden` is a static argument (it
# changes the net's structure / parameter pytree), so a trace is keyed on it.


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
    tanh layer then a linear read-out. Returns (T,)."""
    if hidden <= 0:
        return phi @ params["w"] + params["b"]
    h = jnp.tanh(phi @ params["w1"] + params["b1"])  # (T, hidden)
    return h @ params["w2"] + params["b2"]            # (T,)


@partial(jax.jit, static_argnames=("hidden",))
def _act_forward(
    params: dict[str, "jax.Array"], phi: "jax.Array", key: "jax.Array", hidden: int
) -> tuple["jax.Array", "jax.Array"]:
    """JIT'd policy forward on the per-forward critical path: logits -> probs -> Bernoulli SAMPLE. Returns
    (probs (T,), sampled_actions (T,) in {0,1}). Sampling here IS the exploration; the heavy optax step is the
    separate periodic batch update, never on this path."""
    p = jax.nn.sigmoid(_logits(params, phi, hidden))
    u = jax.random.uniform(key, p.shape, dtype=p.dtype)
    a = (u < p).astype(jnp.float32)  # Bernoulli(p): allow with probability p
    return p, a


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
    is NOT on the per-forward critical path."""

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
    E[(R - b) * sum log pi] with a running-mean baseline b. The JIT'd policy forward is the only per-forward
    work (O(T*d), d=5); the batched optax step is the periodic update. inflight==0 force-allows AND masks that
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
        params + the adam moment state (nothing learned survives a trial), and clear the trajectory buffer, the
        pending transition, the baseline accumulator, and the first-difference baselines. The RNG is reseeded
        from ctx.seed so a trial's sampling is reproducible per the lab's seed."""
        self._t = int(ctx.n_threads)
        self._d_ceil = max(1, int(ctx.d_ceiling))
        self._k = max(1, int(ctx.k_per_thread))
        # reseed the policy/sampling RNG from the trial seed (reproducible exploration).
        self._key = jax.random.PRNGKey(int(ctx.seed) & 0x7FFFFFFF)
        self._key, init_key = jax.random.split(self._key)
        self._params = _init_params(init_key, self._hidden, self._init_logit)
        self._opt_state = self._tx.init(self._params)
        self._phi_buf = []
        self._act_buf = []
        self._mask_buf = []
        self._rew_buf = []
        self._pending = None
        self._b_sum = 0.0
        self._b_cnt = 0
        self._msgs_prev = np.zeros(self._t, dtype=np.int64)
        self._leaves_prev = np.zeros(self._t, dtype=np.int64)
        self._seen = np.zeros(self._t, dtype=bool)
        self._updates = 0
        self._last_grad_norm = 0.0
        self._last_mean_prob = float(_sigmoid(self._init_logit))

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
        JIT'd policy forward (logits -> probs -> Bernoulli SAMPLE), apply the inflight==0 liveness override, and
        STASH the sampled transition as pending (the next observe attaches its reward). Cheap: one JIT'd O(T*d)
        forward, no gradient. Non-throwing — defaulted reads keep a malformed/short feature frame safe (the
        watchdog owns loudness on the hot path, ADR-0002)."""
        T = self._t
        feats = obs.features
        inflight = _fit(np.asarray(feats.get("inflight", ()), dtype=np.float64), T)
        ready = _fit(np.asarray(feats.get("ready", ()), dtype=np.float64), T)
        msgs = _fit(np.asarray(feats.get("msgs", ()), dtype=np.float64), T).astype(np.int64)
        leaves = _fit(np.asarray(feats.get("leaves", ()), dtype=np.float64), T).astype(np.int64)
        served = [i for i in obs.served if 0 <= i < T]

        coalesce = self._coalesce(msgs, leaves, served, T)
        phi = self._build_phi(inflight, ready, coalesce)  # (T, d_in) float32

        # JIT'd policy forward: probabilities + a Bernoulli sample per thread (the exploration).
        self._key, sub = jax.random.split(self._key)
        probs, sampled = _act_forward(self._params, jnp.asarray(phi), sub, self._hidden)
        sampled_np = np.asarray(sampled, dtype=np.float64)   # (T,) in {0,1}
        self._last_mean_prob = float(np.asarray(probs, dtype=np.float64).mean()) if T else 0.0

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
        """The PERIODIC optax (adam) ASCENT step: stack the buffered transitions, form the advantage
        A_f = R_f - b (running-mean baseline; optionally standardized for scale-stability), run the JIT'd
        update step, record the grad norm, and CLEAR the buffer (Monte-Carlo on the recent trajectory). Total
        and defensive — a degenerate batch (all-zero advantage) yields a zero gradient, never a throw."""
        phi = jnp.asarray(np.stack(self._phi_buf, axis=0))     # (B, T, d_in)
        act = jnp.asarray(np.stack(self._act_buf, axis=0))     # (B, T)
        mask = jnp.asarray(np.stack(self._mask_buf, axis=0))   # (B, T)
        rew = np.asarray(self._rew_buf, dtype=np.float32)      # (B,)
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
