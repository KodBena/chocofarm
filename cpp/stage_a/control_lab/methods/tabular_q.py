#!/usr/bin/env python3
"""
cpp/stage_a/control_lab/methods/tabular_q.py — a tabular Q-learning issue-gate (REINFORCEMENT-LEARNING
family) candidate for the issue-gate control lab.

The RL counterpart to the bandit gates (threshold_bandit / contextual_bandit): where a bandit treats each
forward as a context-free (or contextual-but-myopic) arm pull, this controller treats each forward as one
TEMPORAL-DIFFERENCE transition (s, a, r, s') and BOOTSTRAPS — Q(s,a) is updated toward r + gamma*max_a'
Q(s',a'), so a gate's value folds in the discounted value of the state it LEADS TO, not only its immediate
reward. The lab synthesis flags temporal credit as possibly ILLUSORY on this near-horizon-1, self-clocking
plant (a forward's row count is dominated by the gate set THIS forward, with little carryover); this method
is the instrument that MEASURES whether the bootstrap (gamma > 0) buys anything over a myopic bandit, by
being a clean Q-learner whose only structural difference from a one-step bandit is the bootstrap term.

PARAMETER SHARING (the lever that makes RL trainable in a ~4s box). One SHARED tabular Q(s, a) over a small
per-thread state space — every thread's transition reads and updates the SAME table, so a single forward
yields up to T learning updates (one per served thread). The per-thread reward is the SAME pool scalar the
harness feeds (the forward's real row count, the coalescing achieved — HIGHER IS BETTER), credited to every
served thread's pending transition. One shared low-dimensional table is what converges in hundreds of
forwards; a per-thread or high-dimensional table would not.

STATE (a coarse 3x3 discretization -> 9 states). Per thread t, two instantaneous signals binned:

  * submit_pressure[t] = ready[t] / max(1, D - inflight[t])   — backlog against remaining in-flight headroom
    (D = ctx.d_ceiling), binned into {low, med, high} at two fixed cut points. Rises with parked-at-leaf
    backlog AND as a thread nears the in-flight ceiling (a near-saturated thread has little headroom, so each
    remaining slot is precious). This is the SAME submit-pressure signal the threshold bandit gates on.
  * ready_velocity_sign[t] = sign(ready[t] - ready_prev[t])   — {draining, flat, filling}: whether the
    backlog is growing or being worked off, the one bit of temporal context a tabular state can afford.

The cross product is the 9-state index s[t] = 3*pressure_bin + velocity_bin.

ACTIONS {deny=0, allow=1}. act() picks per thread, epsilon-greedy: argmax_a Q(s[t], a) with prob 1-eps, a
uniform random action with prob eps (the exploration that fills the table). Then the LIVENESS OVERRIDE:
inflight[t]==0 is an UNGATED forced flush (DENY-ONLY gate semantics — the runner's effective gate is
`inflight < D && allow`, and the forced flush at inflight==0 is ungated, so a deny is a NO-OP there), so a
thread with nothing in flight is FORCE-ALLOWED regardless of the policy. (The override is applied to the
emitted gate only; the action STORED for the TD update is the policy's chosen action, so the learner is never
credited for a forced flush it did not choose.)

THE LEARNER (the TD timing). The harness drives one transition per forward: observe(reward_of_PREVIOUS act)
then act(obs). The action a taken last forward was chosen in state s (from the previous obs); its reward r
arrives at THIS observe(); its next state s' is only knowable from THIS forward's obs. So:
  * observe(r) STASHES r as the pending reward for the last action (it cannot yet bootstrap — s' is unknown).
  * act(obs) computes the now-current state s' per thread, and for every thread that has a stored (s, a) AND
    is present in obs.served (so s' is a REAL reading, not the absent-thread sentinel) applies the Q-learning
    bootstrap  Q(s,a) += alpha * (r + gamma * max_a' Q(s',a') - Q(s,a)),  then epsilon-greedily selects the
    new action a' in s' and stores (s', a') as that thread's new pending transition.
This is textbook online Q-learning: the update for (s,a,r,s') fires exactly when s' is observed.

WIRE SUBTLETY honored. lab_server fills each length-T feature list as [0]*T and writes ONLY the served tids,
so a thread ABSENT from this forward reads SENTINEL 0 across the board. submit_pressure is read from the
INSTANTANEOUS ready/inflight gauges (never a first-difference of a cumulative counter — msgs/leaves are never
touched here), so a sentinel-0 merely yields pressure=0 for an absent thread, the honest "no observed
pressure". ready_velocity IS a first difference, but of the instantaneous `ready` gauge — and the sentinel-0
hazard still applies (an absent thread would read ready=0 and fabricate a huge spurious "drain"), so the
velocity baseline AND the TD update are computed ONLY for threads in obs.served, against a per-thread baseline
held across the threads' absences. An absent thread holds its stored transition and its baseline untouched
until it reappears; it is never credited with a fabricated state or a phantom delta.

RL family: the per-run state is the shared Q table, the per-thread pending transition (state, action), the
per-thread ready baseline, the pending reward, and the decaying epsilon — ALL cleared in reset() (COLD START
each trial). O(T) dict/array lookups per decision (one fancy-index into a tiny 9x2 table) and O(T) updates per
observe-folded-into-act, non-throwing on the per-forward path. Tabular -> NUMPY backend, no NN, no jax.

Run the unit gate pinned + bounded:
    PYTHONPATH=cpp/stage_a /home/bork/w/vdc/venvs/generic/bin/python -m pytest tests/test_method_tabular_q.py -q

Public Domain (The Unlicense).
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from control_lab.adapter import REGISTRY, Family, Observation, TrialContext

# State-space geometry: a 3 (submit-pressure level) x 3 (ready-velocity sign) discretization -> 9 states.
_N_PRESSURE_BINS = 3   # {low, med, high}
_N_VELOCITY_BINS = 3   # {draining, flat, filling}
_N_STATES = _N_PRESSURE_BINS * _N_VELOCITY_BINS
_N_ACTIONS = 2         # {deny=0, allow=1}

# Default submit-pressure cut points: pressure < 0.5 -> low, [0.5, 1.0) -> med, >= 1.0 -> high. x = 1.0 is the
# "backlog exactly fills the remaining headroom" knee (ready == D - inflight), the natural med/high boundary.
_DEFAULT_PRESSURE_CUTS: tuple[float, float] = (0.5, 1.0)


class TabularQGate:
    """A tabular Q-learning issue gate (REINFORCEMENT-LEARNING). One SHARED 9x2 table Q(s, a) over the coarse
    per-thread state s = (submit-pressure level {low,med,high}) x (ready-velocity sign {draining,flat,filling})
    and actions {deny, allow}. act() is epsilon-greedy argmax_a Q(s[t], a) per thread with the inflight==0
    force-allow liveness override; observe(reward) stashes the pool reward and act() then applies the
    bootstrap TD update Q(s,a) += alpha*(r + gamma*max_a' Q(s',a') - Q(s,a)) for every served thread's stored
    (s, a). Parameter sharing: T updates per forward into one tiny table. Epsilon decays over the run. O(T)
    numpy on the decision path, non-throwing. Tabular (numpy), no NN."""

    family: Family = "rl"

    def __init__(
        self,
        alpha: float = 0.2,
        gamma: float = 0.6,
        eps_start: float = 0.5,
        eps_end: float = 0.02,
        eps_decay_epochs: int = 400,
        pressure_cuts: tuple[float, float] = _DEFAULT_PRESSURE_CUTS,
        q_init: float = 0.0,
    ) -> None:
        # fail loud (ADR-0002): a degenerate learning rate / discount / epsilon schedule / bin geometry is a
        # CONSTRUCTION error, surfaced at build time on the strongest applicable surface (the ctor), never a
        # silent runtime surprise on the per-forward hot path.
        if not (0.0 < alpha <= 1.0):
            raise ValueError(f"TabularQGate: alpha must be in (0, 1], got {alpha}")
        if not (0.0 <= gamma < 1.0):
            raise ValueError(f"TabularQGate: gamma must be in [0, 1), got {gamma}")
        if not (0.0 <= eps_end <= eps_start <= 1.0):
            raise ValueError(
                f"TabularQGate: require 0 <= eps_end <= eps_start <= 1, got eps_start={eps_start}, "
                f"eps_end={eps_end}"
            )
        if eps_decay_epochs < 1:
            raise ValueError(f"TabularQGate: eps_decay_epochs must be >= 1, got {eps_decay_epochs}")
        lo, hi = pressure_cuts
        if not (0.0 <= lo < hi):
            raise ValueError(f"TabularQGate: pressure_cuts must satisfy 0 <= lo < hi, got {pressure_cuts}")

        self._alpha = float(alpha)
        self._gamma = float(gamma)
        self._eps_start = float(eps_start)
        self._eps_end = float(eps_end)
        self._eps_decay_epochs = int(eps_decay_epochs)
        self._cuts = (float(lo), float(hi))
        self._q_init = float(q_init)
        self.name = f"tabular_q_a{self._alpha:g}_g{self._gamma:g}"

        # per-run learner state (all sized/cleared in reset). The shared Q table is the ONLY learned object;
        # the rest is per-thread bookkeeping for the TD timing (the pending transition + ready baseline) plus
        # the pending pool reward and the epoch counter that drives the epsilon schedule.
        self._t = 1
        self._d = 1
        self._rng = np.random.default_rng(0)
        self._Q = np.full((_N_STATES, _N_ACTIONS), self._q_init, dtype=np.float64)
        self._last_state = np.full(self._t, -1, dtype=np.int64)    # stored s per thread (-1 = no pending)
        self._last_action = np.full(self._t, -1, dtype=np.int64)   # stored a per thread (the POLICY action)
        self._ready_prev = np.zeros(self._t, dtype=np.float64)     # per-thread ready baseline (served-only)
        self._ready_seen = np.zeros(self._t, dtype=bool)           # has this thread ever been served?
        self._pending_reward = 0.0                                 # reward of the last act, awaiting its s'
        self._has_pending_reward = False                           # was an observe() seen since the last act?
        self._epochs = 0                                           # decision epochs (forwards acted), for eps
        self._updates = 0                                          # total TD updates applied (dashboard)
        self._last_reward = 0.0

    def reset(self, ctx: TrialContext) -> None:
        """Begin a fresh trial: capture the geometry (T, D) the state signal needs, seed the exploration RNG
        from the trial seed (reproducible epsilon-exploration), and CLEAR the shared table and all per-thread
        bookkeeping (COLD START — the RL family learns within the single run, nothing carries across trials)."""
        self._t = int(ctx.n_threads)
        self._d = max(1, int(ctx.d_ceiling))                       # headroom denominator guard: max(1, D-inflight)
        self._rng = np.random.default_rng(int(ctx.seed))
        self._Q = np.full((_N_STATES, _N_ACTIONS), self._q_init, dtype=np.float64)
        self._last_state = np.full(self._t, -1, dtype=np.int64)
        self._last_action = np.full(self._t, -1, dtype=np.int64)
        self._ready_prev = np.zeros(self._t, dtype=np.float64)
        self._ready_seen = np.zeros(self._t, dtype=bool)
        self._pending_reward = 0.0
        self._has_pending_reward = False
        self._epochs = 0
        self._updates = 0
        self._last_reward = 0.0

    @property
    def _epsilon(self) -> float:
        """Linearly-annealed exploration rate: eps_start -> eps_end over eps_decay_epochs decision epochs, then
        held at eps_end. The short run means exploration must front-load and decay quickly so the back half of
        the box exploits the learned table."""
        frac = min(1.0, self._epochs / float(self._eps_decay_epochs))
        return self._eps_start + (self._eps_end - self._eps_start) * frac

    def observe(self, reward: float, info: Mapping[str, Any]) -> None:
        """Stash the realized PER-FORWARD reward (the forward's row count; higher is better) as the pending
        credit for the last action. It CANNOT be applied yet — the bootstrap needs s', which is only knowable
        from the next forward's obs — so act() folds it into the Q update. Cheap and total: a non-finite reward
        is dropped rather than poisoning the table (ADR-0002: the watchdog owns loudness; the learner stays
        well-defined)."""
        r = float(reward)
        if not np.isfinite(r):
            return  # never let a NaN/inf reward poison the Q table
        self._last_reward = r
        self._pending_reward = r
        self._has_pending_reward = True

    def _states_for(self, ready: np.ndarray, inflight: np.ndarray) -> np.ndarray:
        """Map the instantaneous per-thread gauges to the 9-state index s = 3*pressure_bin + velocity_bin.
        submit_pressure = ready / max(1, D - inflight) binned at the two cut points; velocity sign from
        ready - ready_prev (the per-thread baseline, only meaningful for previously-served threads — an
        unseen thread reads velocity 'flat')."""
        headroom = np.maximum(1.0, self._d - inflight)
        pressure = ready / headroom
        lo, hi = self._cuts
        # pressure bin: 0 (low) if < lo, 1 (med) if in [lo, hi), 2 (high) if >= hi.
        p_bin = (pressure >= lo).astype(np.int64) + (pressure >= hi).astype(np.int64)
        # velocity bin: 1 (flat) baseline; 0 (draining) if ready fell, 2 (filling) if it rose. Only threads
        # seen before have a meaningful baseline; an unseen thread is 'flat' (delta against a 0 baseline it
        # never set is not trustworthy, so we neither drain nor fill it).
        delta = ready - self._ready_prev
        v_bin = np.ones(self._t, dtype=np.int64)
        v_bin[self._ready_seen & (delta < 0.0)] = 0   # draining
        v_bin[self._ready_seen & (delta > 0.0)] = 2   # filling
        return _N_VELOCITY_BINS * p_bin + v_bin

    def act(self, obs: Observation) -> Sequence[int]:
        """Compute the now-current state s' per thread, apply the deferred Q-learning bootstrap for every
        served thread's stored (s, a) using the pending pool reward, then epsilon-greedily select the new
        per-thread action and emit the gate (with the inflight==0 force-allow liveness override). Cheap (O(T)
        numpy into a 9x2 table) and non-throwing — defaulted reads keep a malformed/short feature frame safe
        (the watchdog owns loudness on the hot path, ADR-0002)."""
        t = self._t
        feats = obs.features
        # length-T INSTANTANEOUS gauges (submit-pressure reads these directly; never first-differences a
        # cumulative counter). Tolerate a short/absent list defensively so act() never throws.
        ready = _fit(np.asarray(feats.get("ready", ()), dtype=np.float64), t)
        inflight = _fit(np.asarray(feats.get("inflight", ()), dtype=np.float64), t)

        # served mask: which threads' readings are REAL this forward (the rest are the [0]*T sentinel). Only
        # served threads get a state update / TD update / baseline refresh; absent threads hold their pending
        # transition and baseline untouched (the wire subtlety — an absent thread's sentinel-0 must never
        # fabricate a state or a velocity delta).
        served = np.zeros(t, dtype=bool)
        for tid in obs.served:
            if 0 <= tid < t:
                served[tid] = True

        s_next = self._states_for(ready, inflight)

        # --- deferred TD update: fold the pending reward into the stored (s, a) of every served thread ------
        if self._has_pending_reward:
            r = self._pending_reward
            have_prev = self._last_state >= 0
            upd = served & have_prev
            if upd.any():
                idx = np.flatnonzero(upd)
                s = self._last_state[idx]
                a = self._last_action[idx]
                bootstrap = r + self._gamma * self._Q[s_next[idx]].max(axis=1)
                td_error = bootstrap - self._Q[s, a]
                # parameter sharing: every served thread's transition updates the SAME table. np.add.at applies
                # the per-thread updates with correct accumulation when two threads land on the same (s, a).
                np.add.at(self._Q, (s, a), self._alpha * td_error)
                self._updates += int(idx.size)
            self._has_pending_reward = False

        # --- epsilon-greedy action selection in the now-current state s' -------------------------------------
        eps = self._epsilon
        greedy = np.argmax(self._Q[s_next], axis=1).astype(np.int64)   # argmax_a Q(s', a) per thread
        explore = self._rng.random(t) < eps
        rand_act = self._rng.integers(0, _N_ACTIONS, size=t)
        action = np.where(explore, rand_act, greedy).astype(np.int64)  # the POLICY action (pre-override)

        # store this forward's (s', a') as the new pending transition for SERVED threads, and refresh their
        # ready baseline. An absent thread keeps its prior pending (s, a) and baseline so the next time it is
        # served its transition closes against a real s' (no phantom credit across the gap).
        served_idx = np.flatnonzero(served)
        self._last_state[served_idx] = s_next[served_idx]
        self._last_action[served_idx] = action[served_idx]
        self._ready_prev[served_idx] = ready[served_idx]
        self._ready_seen[served_idx] = True

        self._epochs += 1

        # --- emit the gate: the policy action, then the liveness override ------------------------------------
        # inflight==0 is an UNGATED forced flush (DENY-ONLY semantics) -> a deny is a no-op there, so force
        # allow. The override touches the EMITTED gate only; the STORED action above is the policy's choice, so
        # the learner is never credited for a forced flush it did not select.
        allow = action == 1
        allow = allow | (inflight <= 0.0)
        return [1 if a else 0 for a in allow.tolist()]

    def metrics(self) -> Mapping[str, float]:
        """Dashboard scalars: the current exploration rate, the max Q value (the most-valued state-action), how
        many of the 9 states have ever been visited (a coverage signal — did the run explore the table?), the
        decision-epoch and TD-update counts, and the last realized reward. Empty-safe."""
        # a state is "visited" if either of its actions has moved off the q_init prior.
        visited = int(np.any(self._Q != self._q_init, axis=1).sum())
        return {
            "epsilon": float(self._epsilon),
            "max_q": float(self._Q.max()),
            "min_q": float(self._Q.min()),
            "n_states_visited": float(visited),
            "epochs": float(self._epochs),
            "updates": float(self._updates),
            "last_reward": float(self._last_reward),
        }


def _fit(x: np.ndarray, t: int) -> np.ndarray:
    """Coerce a feature array to length T: truncate if long, zero-pad if short. Defensive so act() never throws
    on a malformed/empty feature list (ADR-0002: the per-forward path stays cheap and total; a zero-pad lands a
    thread at ready=0/inflight=0, i.e. the inflight==0 liveness force-allow — never a fabricated denial)."""
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
REGISTRY.setdefault("tabular_q", TabularQGate)
