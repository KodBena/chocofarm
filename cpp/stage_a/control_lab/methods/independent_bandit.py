#!/usr/bin/env python3
"""
cpp/stage_a/control_lab/methods/independent_bandit.py — T INDEPENDENT per-thread 2-arm bandits
(ONLINE-LEARNING family) candidate for the issue-gate control lab.

The DECOUPLED contrast / ablation to the parameter-SHARING online methods (threshold_bandit's one shared
theta, contextual_bandit's one shared ridge model). Where those collapse the run's evidence into a single
pool-priced learner, this method runs T INDEPENDENT 2-arm (deny / allow) bandits — one per producer thread —
each selected by per-thread DISCOUNTED-UCB and each scored from a per-thread reward. It exists to
SUBSTANTIATE parameter-sharing: by splitting the short run's evidence T ways it is the expected
SAMPLE-INEFFICIENT arm, and that inefficiency is the measurement that makes the case for sharing.

Mechanism (features -> gate). Two arms per thread, a = {deny(0), allow(1)}, each carrying a discounted value
Q[t, a] and a discounted pull count N[t, a]. The act for thread t is the per-thread D-UCB argmax

    arm[t] = argmax_a  Q[t, a] + c * sqrt( ln( sum_b N[t, b] ) / N[t, a] )      (N[t,a]==0 -> +inf, an initial pull)

allow[t] = (arm[t] == 1). A tie (cold start, both N==0) breaks toward ALLOW so an un-pulled thread starts at
the baseline rather than throttling blind. The chosen arm is HELD per thread for a reward-hold window W
(below). No feature THRESHOLD gates the action — unlike the threshold/contextual siblings the action IS the
bandit arm; the only feature read on the action path is inflight (for the liveness override) and, for
SCORING, the per-thread own-rate (below). submit_pressure is read only to optionally CONTEXTUALIZE each
per-thread bandit on a coarse bin (a tiny tabular contextual bandit, knob `context_bins`); with one bin
(default) it is the plain per-thread bandit.

The reward (the per-thread signal). The ONLY per-thread throughput signal on the wire is a thread's own leaf
production. So the PRIMARY per-thread reward is that thread's own LEAF RATE

    own_leaf_rate[t] = (leaves[t] delta) / dt           (served-diffed; dt from obs.t_monotonic)

— first-differenced against a per-thread leaves baseline ONLY for threads in obs.served, never for an absent
thread (whose sentinel-0 would manufacture a spurious negative delta — the WIRE SUBTLETY). This own-rate is
computed in act() from the features. A thread maximizing its OWN rate can flood the shared server (a tragedy
of the commons: T selfish bandits over-issuing into one queue), so the per-thread reward is BLENDED with a
small weight beta on the POOL reward (the harness's per-forward row count, fed to observe()):

    reward[t] = own_leaf_rate[t]  +  beta * pool_reward

beta couples the otherwise-independent learners just enough to price the shared congestion they create,
without collapsing them into a shared model (that is the sibling's job). beta=0 is the pure decoupled bandit.

Credit assignment under pipeline lag (reward-hold window W). A gate decision shows up in throughput only a
few forwards later, so a choice is HELD for W forwards. Over the window act() accumulates, per SERVED thread,
that thread's own-rate; observe() accumulates the pool reward. At the next window OPEN each thread's held arm
is scored by reward[t] = (window-MEAN own-rate of t, over the forwards in the window where t was served) +
beta*(window-mean pool reward); a thread served zero times in the window gets NO update that epoch (its prior
Q is preserved — an absent thread is never credited with a fabricated 0). The per-arm (Q, N) are discounted
once per epoch (D-UCB non-stationarity: stale evidence fades so a regime shift — warm-up -> steady state,
depth>1 turning D live — can re-open a dominated arm), the held arm credited, and each thread re-selects. W
is the one timing knob: large enough to clear the pipeline lag, small enough not to starve the short
(hundreds-to-thousands of forwards over a ~4s box) run of arm switches. Default W=8.

Liveness override (DENY-ONLY gate semantics: the runner's effective gate is `inflight < D && allow`, and the
forced flush at inflight==0 is UNGATED): inflight[t]==0 -> force allow, applied FRESH every forward (a deny
is a no-op there, so a thread with nothing in flight is never starved by a held deny-arm).

ONLINE family: reset() clears ALL learner state (the per-thread/per-context (Q, N), the held arms, the
leaves baseline, the hold-window accumulators); observe() folds the realized pool reward into the window
(for the beta blend); act() computes the per-thread own-rate, accumulates it, closes the window + re-selects
on the W boundary, and applies the held arm + liveness; metrics() exposes the per-thread chosen-arm value
and the blend beta for the dashboard. O(T) numpy on the decision path (the per-context bin select is a
gather), non-throwing — it rides the per-forward critical path. Knobs: D-UCB constant c, blend beta, hold
window W, discount gamma, exploration mode (ucb / epsilon) + epsilon, optional context_bins.

Run the unit gate pinned + bounded:
    PYTHONPATH=cpp/stage_a /home/bork/w/vdc/venvs/generic/bin/python -m pytest tests/test_method_independent_bandit.py -q

Public Domain (The Unlicense).
"""
from __future__ import annotations

from typing import Any, Literal, Mapping, Sequence

import numpy as np

from control_lab.adapter import REGISTRY, Family, Observation, TrialContext

# the two arms of every per-thread bandit: index 0 = deny, index 1 = allow. allow being arm 1 lets the cold
# tie break toward allow (argmax over equal +inf indices picks the LOWEST index — so the tie is steered to
# allow explicitly in _select rather than implicitly to deny).
_DENY, _ALLOW = 0, 1
_N_ARMS = 2

ExploreMode = Literal["ucb", "epsilon"]


class IndependentBanditGate:
    """T INDEPENDENT per-thread 2-arm (deny/allow) bandits (ONLINE). Each thread runs its own discounted-UCB
    bandit over {deny, allow}, scored by that thread's own leaf rate (served-diffed, the only per-thread
    signal) blended with weight beta on the pool reward (a congestion price against the tragedy of the
    commons). The chosen arm is held per thread for W forwards, then scored by the window-mean per-thread
    reward and re-selected. Optionally each per-thread bandit is contextualized on a coarse submit_pressure
    bin (a tiny tabular contextual bandit). inflight[t]==0 force-allows (a deny is a no-op there). O(T) numpy
    on the decision path, non-throwing. The decoupled ablation to the parameter-sharing online methods."""

    family: Family = "online"

    def __init__(
        self,
        c: float = 1.0,
        beta: float = 0.05,
        hold_window: int = 8,
        gamma: float = 0.95,
        explore: ExploreMode = "ucb",
        epsilon: float = 0.10,
        context_bins: int = 1,
    ) -> None:
        # fail loud (ADR-0002): a degenerate exploration constant / blend / hold window / discount / bin count
        # is a CONSTRUCTION error, surfaced at build time on the strongest applicable surface (the ctor),
        # never a silent runtime surprise on the per-forward hot path.
        if c < 0.0:
            raise ValueError(f"IndependentBanditGate: UCB constant c must be >= 0, got {c}")
        if not np.isfinite(beta):
            raise ValueError(f"IndependentBanditGate: blend beta must be finite, got {beta}")
        if hold_window < 1:
            raise ValueError(f"IndependentBanditGate: hold_window W must be >= 1, got {hold_window}")
        if not (0.0 < gamma <= 1.0):
            raise ValueError(f"IndependentBanditGate: gamma must be in (0, 1], got {gamma}")
        if explore not in ("ucb", "epsilon"):
            raise ValueError(f"IndependentBanditGate: explore must be 'ucb' or 'epsilon', got {explore!r}")
        if not (0.0 <= epsilon <= 1.0):
            raise ValueError(f"IndependentBanditGate: epsilon must be in [0, 1], got {epsilon}")
        if context_bins < 1:
            raise ValueError(f"IndependentBanditGate: context_bins must be >= 1, got {context_bins}")

        self._c = float(c)
        self._beta = float(beta)
        self._W = int(hold_window)
        self._gamma = float(gamma)
        self._explore: ExploreMode = explore
        self._epsilon = float(epsilon)
        self._bins = int(context_bins)
        self.name = f"independent_bandit_W{self._W}_b{self._beta:g}"

        # per-run learner state (all sized/cleared in reset). The bandit tables are indexed
        # [thread, context_bin, arm]; with bins==1 the middle axis is a singleton (the plain per-thread
        # bandit). Q is the discounted mean value, N the discounted pull count.
        self._t = 1
        self._d = 1
        self._k = 1
        self._Q = np.zeros((1, self._bins, _N_ARMS), dtype=np.float64)   # discounted per-arm value
        self._N = np.zeros((1, self._bins, _N_ARMS), dtype=np.float64)   # discounted per-arm pull count
        self._arm = np.full(1, _ALLOW, dtype=np.int64)                   # currently-held arm per thread
        self._arm_bin = np.zeros(1, dtype=np.int64)                      # context bin the held arm was chosen in
        # per-thread leaves baseline for the served-diff own-rate (NaN = not yet seen -> no first-difference).
        self._leaf_base = np.full(1, np.nan, dtype=np.float64)
        self._t_last = np.full(1, np.nan, dtype=np.float64)              # last-served monotonic time per thread
        # hold-window accumulators: per-thread own-rate sum + served-count, and the pool reward sum/count.
        self._own_sum = np.zeros(1, dtype=np.float64)
        self._own_n = np.zeros(1, dtype=np.int64)
        self._pool_sum = 0.0
        self._pool_n = 0
        self._win_forwards = 0                                           # forwards held in the current window
        self._epochs = 0                                                 # decision epochs closed (windows)
        self._last_pool_reward = 0.0
        self._rng = np.random.default_rng(0)

    def reset(self, ctx: TrialContext) -> None:
        """Begin a fresh trial: capture the geometry (T, D, K) the own-rate + optional context binning need
        and clear ALL learner state. Every thread starts on the ALLOW arm (so the un-warmed warm-up forwards
        default to the baseline gate) with empty (Q, N); D-UCB then pulls the untried DENY arm (N==0 -> +inf
        index) as the plant warms. The epsilon-path rng is seeded from ctx.seed for reproducibility."""
        self._t = int(ctx.n_threads)
        self._d = max(1, int(ctx.d_ceiling))                # headroom denominator guard: max(1, D - inflight)
        self._k = max(1, int(ctx.k_per_thread))             # backlog normalizer for the optional context bin
        self._Q = np.zeros((self._t, self._bins, _N_ARMS), dtype=np.float64)
        self._N = np.zeros((self._t, self._bins, _N_ARMS), dtype=np.float64)
        self._arm = np.full(self._t, _ALLOW, dtype=np.int64)
        self._arm_bin = np.zeros(self._t, dtype=np.int64)
        self._leaf_base = np.full(self._t, np.nan, dtype=np.float64)
        self._t_last = np.full(self._t, np.nan, dtype=np.float64)
        self._own_sum = np.zeros(self._t, dtype=np.float64)
        self._own_n = np.zeros(self._t, dtype=np.int64)
        self._pool_sum = 0.0
        self._pool_n = 0
        self._win_forwards = 0
        self._epochs = 0
        self._last_pool_reward = 0.0
        self._rng = np.random.default_rng(int(ctx.seed) & 0xFFFFFFFF)

    def observe(self, reward: float, info: Mapping[str, Any]) -> None:
        """Fold the realized PER-FORWARD POOL reward (the forward's row count; higher is better) into the
        current hold window — it feeds the optional beta-blend congestion term at window close. The per-thread
        own-rate is NOT here (it needs the features, so it is accumulated in act). Cheap and total: a
        non-finite reward is ignored rather than poisoning the learner (ADR-0002: the watchdog owns loudness;
        the learner stays well-defined)."""
        r = float(reward)
        if not np.isfinite(r):
            return  # never let a NaN/inf pool reward poison the blended statistics
        self._last_pool_reward = r
        self._pool_sum += r
        self._pool_n += 1

    def act(self, obs: Observation) -> Sequence[int]:
        """Compute each SERVED thread's own leaf rate (served-diffed against its leaves baseline, dt from
        obs.t_monotonic), accumulate it into the hold window, close the window + re-select on the W boundary,
        then apply the held arm + the inflight==0 liveness force-allow. Cheap (O(T) numpy) and non-throwing —
        defaulted reads keep a malformed/short feature frame safe (the watchdog owns loudness on the hot path,
        ADR-0002)."""
        t = self._t
        feats = obs.features
        # length-T INSTANTANEOUS gauges + the CUMULATIVE leaves counter; tolerate a short/absent list
        # defensively so act() never throws.
        ready = _fit(np.asarray(feats.get("ready", ()), dtype=np.float64), t)
        inflight = _fit(np.asarray(feats.get("inflight", ()), dtype=np.float64), t)
        leaves = _fit(np.asarray(feats.get("leaves", ()), dtype=np.float64), t)

        # served mask: ONLY these threads' cumulative counters are real this forward (the WIRE SUBTLETY — an
        # absent thread reads the [0]*T sentinel, never its true cumulative). Build it from obs.served.
        served = np.zeros(t, dtype=bool)
        for tid in obs.served:
            if 0 <= int(tid) < t:
                served[int(tid)] = True

        now = float(obs.t_monotonic)
        # per-thread own leaf rate, computed ONLY for served threads that already carry a baseline (so the
        # first observation of a thread seeds its baseline without a first-difference). dt is the per-thread
        # gap since that thread was last served (its own-rate is its leaf growth over the interval the held
        # arm was in effect for it). max(dt, tiny) guards the divisor.
        have_base = served & np.isfinite(self._leaf_base) & np.isfinite(self._t_last)
        if have_base.any():
            dleaves = leaves[have_base] - self._leaf_base[have_base]
            dt = np.maximum(now - self._t_last[have_base], 1e-9)
            rate = np.maximum(dleaves, 0.0) / dt          # clamp >=0: a cumulative counter never decreases
            idx = np.flatnonzero(have_base)
            # accumulate this forward's own-rate into the current window for the threads we measured.
            self._own_sum[idx] += rate
            self._own_n[idx] += 1

        # advance the served threads' baselines (only the served — never an absent thread, per the wire
        # subtlety) so the NEXT served forward differences against a real prior value.
        if served.any():
            self._leaf_base[served] = leaves[served]
            self._t_last[served] = now

        # refresh each thread's CONTEXT BIN from the live submit_pressure (a no-op singleton bin when
        # context_bins==1). The held arm is recorded against the bin in force NOW, and the next window-close
        # re-selects against it — so a thread's bandit learns per submit-pressure regime.
        self._arm_bin = self._context_bin(ready, inflight)

        # hold-window bookkeeping: close on the W boundary (re-selects every thread's arm), but never on the
        # very first decision (no held arm has been scored yet) — _win_forwards counts forwards since the last
        # open, so the first window opens lazily after W forwards.
        self._win_forwards += 1
        if self._win_forwards >= self._W:
            self._close_window()

        # apply the held arm per thread, then the liveness override fresh.
        allow = self._arm == _ALLOW
        allow = allow | (inflight <= 0.0)                  # inflight==0: UNGATED forced flush -> deny is a no-op
        return [1 if a else 0 for a in allow.tolist()]

    def _close_window(self) -> None:
        """Score each thread's held arm by its window-mean reward and re-select (one D-UCB decision epoch).

        reward[t] = (window-mean own-rate of t over the forwards it was served) + beta*(window-mean pool
        reward). A thread served zero times this window gets NO update (its prior Q/N preserved — never
        credited with a fabricated 0). Discount every (Q, N) once (non-stationarity), credit the held arm in
        the bin it was chosen in, then re-select each thread's arm in the bin it is CURRENTLY in (recomputed
        at window close from the live submit_pressure -> the chosen arm is held against the context that will
        be in force next window)."""
        pool_mean = self._pool_sum / self._pool_n if self._pool_n else 0.0
        measured = self._own_n > 0                                       # threads with >=1 served forward
        own_mean = np.divide(self._own_sum, self._own_n,
                             out=np.zeros(self._t, dtype=np.float64),
                             where=measured)
        reward_t = own_mean + self._beta * pool_mean                     # blended per-thread reward

        # discount ALL arms once per decision epoch (Garivier-Moulines D-UCB) so stale evidence fades.
        self._Q *= self._gamma
        self._N *= self._gamma
        # credit the held (thread, bin, arm) for measured threads only. The discounted-mean update folds the
        # window reward into Q with the incremental-mean form Q <- Q + (r - Q)/N after N is bumped, which on a
        # discounted count yields the standard D-UCB running mean.
        rows = np.flatnonzero(measured)
        for tt in rows.tolist():
            b = int(self._arm_bin[tt])
            a = int(self._arm[tt])
            self._N[tt, b, a] += 1.0
            self._Q[tt, b, a] += (reward_t[tt] - self._Q[tt, b, a]) / self._N[tt, b, a]
        self._epochs += 1

        # re-select every thread's arm. The current context bin is recomputed lazily by the next act via the
        # live features; we select against the bin each thread is currently recorded in (arm_bin), which the
        # next act may refine. Selection is per-thread, vectorized where possible.
        self._reselect()

        # reset the window accumulators for the next hold.
        self._own_sum[:] = 0.0
        self._own_n[:] = 0
        self._pool_sum = 0.0
        self._pool_n = 0
        self._win_forwards = 0

    def _reselect(self) -> None:
        """Per-thread arm selection in each thread's current context bin. D-UCB: any unpulled arm
        (discounted N==0) is taken first (optimism -> +inf index, tie broken toward ALLOW); otherwise argmax
        of discounted-mean + c*sqrt(ln(sum_b N)/N). The epsilon mode instead exploits the argmax with prob
        1-epsilon and explores uniformly otherwise (rng seeded from ctx.seed). Vectorized over threads."""
        rows = np.arange(self._t)
        b = self._arm_bin                                               # current per-thread bin
        Q = self._Q[rows, b]                                           # (T, 2) discounted values in-bin
        N = self._N[rows, b]                                           # (T, 2) discounted counts in-bin

        if self._explore == "epsilon":
            greedy = _argmax_allow_tiebreak(Q)                        # exploit the best arm (tie -> allow)
            explore_arm = self._rng.integers(0, _N_ARMS, size=self._t)
            do_explore = self._rng.random(self._t) < self._epsilon
            self._arm = np.where(do_explore, explore_arm, greedy).astype(np.int64)
            return

        # D-UCB index. Unpulled arms (N==0) get +inf so they are taken first; pulled arms get mean + bonus.
        total = N.sum(axis=1, keepdims=True)                          # (T, 1) per-thread discounted total
        with np.errstate(divide="ignore", invalid="ignore"):
            bonus = self._c * np.sqrt(np.log(np.maximum(total, 1.0)) / N)
            index = Q + bonus
        index = np.where(N <= 0.0, np.inf, index)                     # force an initial pull of each arm
        self._arm = _argmax_allow_tiebreak(index).astype(np.int64)

    def _context_bin(self, ready: np.ndarray, inflight: np.ndarray) -> np.ndarray:
        """Discretize each thread's submit_pressure x[t] = ready[t]/max(1, D - inflight[t]) into one of
        `context_bins` coarse bins (a tiny tabular contextual bandit). With bins==1 every thread is bin 0
        (the plain per-thread bandit). Bins partition [0, hi] uniformly with the top bin catching the tail;
        cheap and total."""
        if self._bins == 1:
            return np.zeros(self._t, dtype=np.int64)
        headroom = np.maximum(1.0, self._d - inflight)
        x = ready / headroom
        # uniform bins over [0, _BIN_HI] with the last bin absorbing the tail (x can exceed _BIN_HI).
        edges = np.linspace(0.0, _BIN_HI, self._bins, endpoint=False)[1:]   # _bins-1 interior edges
        return np.asarray(np.digitize(x, edges), dtype=np.int64)

    def metrics(self) -> Mapping[str, float]:
        """Dashboard scalars: the blend beta, the count of threads currently on the ALLOW arm, decision epochs
        closed, the last pool reward, and the per-thread CHOSEN-arm discounted value (keyed by thread so the
        dashboard can render which threads the independent learners have throttled and how confidently).
        Empty-safe."""
        rows = np.arange(self._t)
        b = self._arm_bin
        chosen_val = self._Q[rows, b, self._arm]                       # value of each thread's held arm in-bin
        out: dict[str, float] = {
            "beta": self._beta,
            "n_allow": float(int(np.count_nonzero(self._arm == _ALLOW))),
            "epochs": float(self._epochs),
            "last_pool_reward": float(self._last_pool_reward),
            "mean_chosen_value": float(chosen_val.mean()) if self._t else 0.0,
        }
        for tt in range(self._t):
            out[f"thread{tt}_arm"] = float(int(self._arm[tt]))         # 0=deny, 1=allow
            out[f"thread{tt}_value"] = float(chosen_val[tt])
        return out


# the submit_pressure upper edge for the optional context binning: pressures above this land in the top bin.
_BIN_HI = 2.0


def _argmax_allow_tiebreak(score: np.ndarray) -> np.ndarray:
    """Row-wise argmax over the 2-arm axis with ties broken toward ALLOW (arm 1). np.argmax picks the lowest
    index on a tie (-> deny); we want a cold tie to default to the baseline (allow), so prefer ALLOW whenever
    its score is >= DENY's. (T, 2) -> (T,)."""
    return np.where(score[:, _ALLOW] >= score[:, _DENY], _ALLOW, _DENY).astype(np.int64)


def _fit(x: np.ndarray, t: int) -> np.ndarray:
    """Coerce a feature array to length T: truncate if long, zero-pad if short. Defensive so act() never
    throws on a malformed/empty feature list (ADR-0002: the per-forward path stays cheap and total). A
    zero-pad lands a thread at ready=0/inflight=0/leaves=0 — the inflight==0 liveness force-allow, and (since
    such a thread is not in obs.served) never first-differenced — so the pad never fabricates a denial or a
    spurious own-rate."""
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
REGISTRY.setdefault("independent_bandit", IndependentBanditGate)
