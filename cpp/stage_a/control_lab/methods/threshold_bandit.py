#!/usr/bin/env python3
"""
cpp/stage_a/control_lab/methods/threshold_bandit.py — a homogeneous threshold bandit issue-gate
(ONLINE-LEARNING family) candidate for the issue-gate control lab.

The online counterpart to the static threshold gates (bang_bang / whittle_rmab): instead of FIXING the gate
threshold at construction, this controller LEARNS ONE shared threshold theta within the single run by a small
non-stationary MULTI-ARMED BANDIT over a fixed arm set of candidate thresholds. Parameter-sharing collapses
the per-thread credit into one pool reward (the harness feeds one scalar per forward — the forward's real
row count, the coalescing achieved — and HIGHER IS BETTER), so the learning problem is one-dimensional: pick
the theta that maximizes throughput on THIS plant, this run.

Mechanism (features -> gate). Each forward, the per-thread saturation signal is the SUBMIT PRESSURE

    x[t] = ready[t] / max(1, D - inflight[t])

— normalized available backlog against the remaining in-flight headroom (D = ctx.d_ceiling). It rises with
parked-at-leaf backlog (more to release) AND as a thread nears the in-flight ceiling (a near-saturated thread
has little headroom, so each remaining slot is precious). The gate is the SAME learned theta applied to every
thread (the parameter-sharing that makes one pool reward legitimate):

    allow[t] = 1  iff  x[t] >= theta

Liveness override (DENY-ONLY gate semantics — the runner's effective gate is `inflight < D && allow` and the
forced flush at inflight==0 is UNGATED, so a deny is a NO-OP there): inflight[t]==0 -> force allow. The arm
set spans the two degenerate extremes so the bandit is sandwiched by the baseline: theta=0 makes x>=0 trivially
true everywhere = ALL-ALLOW (byte-identical to the AllAllow control arm), and theta=+inf denies until the
forced flush = DENY-UNTIL-FORCED. Because all-allow IS one arm, a converged bandit can never do much worse
than baseline — it only ever trades up from it.

The learner (observe(reward) -> update). The plant RAMPS (cold-compile, warm-up, then steady state), so a
STATIONARY bandit would over-trust an arm scored before the regime settled. We use DISCOUNTED-UCB (D-UCB,
Garivier-Moulines / Kocsis-Szepesvari): per-arm discounted reward sum S_a and discounted count N_a, both
decayed by gamma at every DECISION EPOCH so stale evidence fades and a late regime shift can re-open a
previously-dominated arm. The selection index is the discounted mean plus an exploration bonus

    index[a] = S_a / N_a + c * sqrt(ln(sum_b N_b) / N_a)         (N_a == 0 -> +inf, forces an initial pull)

REWARD-HOLD WINDOW W. A throughput change lags its gate by the pipeline depth, so scoring an arm on the very
next forward would credit it with the PREVIOUS arm's in-flight work. The controller HOLDS the chosen arm for
W forwards, accumulates the per-forward reward over the hold, and only on window close scores that arm by its
WINDOW-MEAN reward, then discounts + folds it into (S_a, N_a) and re-selects. W is the one knob that must clear
the pipeline lag without starving the short run of arm-switches; the default is small (W=8) because the run is
only hundreds-to-thousands of forwards over a ~4s box.

WIRE SUBTLETY honored: x[t] is read from the INSTANTANEOUS ready/inflight gauges, never a first-difference of
a cumulative counter (msgs, leaves) — so the [0]*T sentinel a thread absent from this forward reads (lab_server
fills only served tids) cannot manufacture a spurious negative delta. A sentinel-0 ready merely yields x=0 for
that thread (it is denied unless inflight==0 force-allows it), the honest "no observed pressure" reading; it is
never differenced. The reward is the harness's per-forward row count, so the learning signal needs no wire
decode at all.

ONLINE family: the per-run state is the D-UCB arm statistics, the current arm index, and the reward-window
accumulator — ALL cleared in reset(). observe() updates the learner from the realized reward; act() applies
the current learned theta; metrics() exposes the learned theta and per-arm discounted means for the dashboard.
O(A) per update and O(T) per decision (one numpy expression), non-throwing on the per-forward path.

Run the unit gate pinned + bounded:
    PYTHONPATH=cpp/stage_a /home/bork/w/vdc/venvs/generic/bin/python -m pytest tests/test_method_threshold_bandit.py -q

Public Domain (The Unlicense).
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from control_lab.adapter import REGISTRY, Family, Observation, TrialContext

# The default arm set: candidate submit-pressure thresholds. The two extremes anchor the bandit to the
# baseline — 0.0 is ALL-ALLOW (x >= 0 is always true), +inf is DENY-UNTIL-FORCED (no finite x clears it, so
# only the inflight==0 liveness flush issues). The interior arms sweep increasing selectivity. all-allow being
# an arm is the safety floor the brief names: the bandit can never converge far below baseline.
_DEFAULT_ARMS: tuple[float, ...] = (0.0, 0.25, 0.5, 1.0, 2.0, float("inf"))


class ThresholdBanditGate:
    """A homogeneous threshold bandit issue gate (ONLINE). One shared gate threshold theta is selected by a
    non-stationary discounted-UCB bandit over a fixed arm set of candidate thresholds and applied to EVERY
    thread on the submit-pressure signal x[t] = ready[t] / max(1, D - inflight[t]): allow[t] = 1 iff
    x[t] >= theta. The chosen arm is HELD for W forwards; the accumulated reward's window-mean scores it, then
    D-UCB discounts + re-selects. theta=0 (=all-allow) and theta=+inf (=deny-until-forced) bound the arm set so
    the bandit stays sandwiched by the baseline. inflight[t]==0 force-allows (a deny is a no-op there). O(T)
    numpy on the decision path, non-throwing."""

    family: Family = "online"

    def __init__(
        self,
        arms: Sequence[float] = _DEFAULT_ARMS,
        c: float = 1.0,
        hold_window: int = 8,
        gamma: float = 0.95,
    ) -> None:
        # fail loud (ADR-0002): a degenerate arm set / exploration constant / hold window / discount is a
        # CONSTRUCTION error, surfaced at build time on the strongest applicable surface (the ctor), never a
        # silent runtime surprise on the per-forward hot path.
        arms_arr = np.asarray(list(arms), dtype=np.float64)
        if arms_arr.size < 1:
            raise ValueError("ThresholdBanditGate: arms must be non-empty")
        if np.isnan(arms_arr).any():
            raise ValueError(f"ThresholdBanditGate: arms must not contain NaN, got {list(arms)}")
        if (arms_arr < 0.0).any():
            raise ValueError(f"ThresholdBanditGate: thresholds must be >= 0, got {list(arms)}")
        if c < 0.0:
            raise ValueError(f"ThresholdBanditGate: exploration constant c must be >= 0, got {c}")
        if hold_window < 1:
            raise ValueError(f"ThresholdBanditGate: hold_window W must be >= 1, got {hold_window}")
        if not (0.0 < gamma <= 1.0):
            raise ValueError(f"ThresholdBanditGate: gamma must be in (0, 1], got {gamma}")

        self._arms = arms_arr
        self._a = int(arms_arr.size)
        self._c = float(c)
        self._W = int(hold_window)
        self._gamma = float(gamma)
        self.name = f"threshold_bandit_W{self._W}_c{self._c:g}"

        # per-run learner state (all sized/cleared in reset). D-UCB carries a DISCOUNTED reward sum and a
        # DISCOUNTED count per arm; the current arm index and the reward-hold-window accumulator complete the
        # online state. _last_reward is kept only for the dashboard.
        self._t = 1
        self._d = 1
        self._S = np.zeros(self._a, dtype=np.float64)      # discounted reward sum per arm
        self._N = np.zeros(self._a, dtype=np.float64)      # discounted pull count per arm
        self._cur = 0                                      # currently-selected arm index (the held theta)
        self._win_sum = 0.0                                # reward accumulated over the current hold window
        self._win_n = 0                                    # forwards observed in the current hold window
        self._epochs = 0                                   # completed decision epochs (windows closed)
        self._last_reward = 0.0

    def reset(self, ctx: TrialContext) -> None:
        """Begin a fresh trial: capture the geometry (T, D) the submit-pressure signal needs and clear ALL
        learner state. The first held arm is the all-allow / lowest threshold so the un-warmed warm-up forwards
        default to the baseline gate (and theta=0, if present, is byte-identical to AllAllow there); D-UCB then
        explores the untried arms (N_a==0 -> +inf index) as the plant warms."""
        self._t = int(ctx.n_threads)
        self._d = max(1, int(ctx.d_ceiling))               # headroom denominator guard: max(1, D - inflight)
        self._S = np.zeros(self._a, dtype=np.float64)
        self._N = np.zeros(self._a, dtype=np.float64)
        # start on the arm with the SMALLEST threshold (the most-permissive / closest-to-baseline gate) so the
        # cold start cannot throttle before any reward has been seen.
        self._cur = int(np.argmin(self._arms))
        self._win_sum = 0.0
        self._win_n = 0
        self._epochs = 0
        self._last_reward = 0.0

    def observe(self, reward: float, info: Mapping[str, Any]) -> None:
        """Fold the realized PER-FORWARD reward (the forward's row count; higher is better) into the current
        hold window. On window close (W forwards held) score the held arm by its WINDOW-MEAN reward: discount
        every arm's (S, N) once (non-stationarity — stale evidence fades), credit the held arm, then re-select
        the next arm by the D-UCB index. Cheap (O(A)) and total — a non-finite reward is ignored rather than
        poisoning the learner (ADR-0002: the watchdog owns loudness; the learner stays well-defined)."""
        r = float(reward)
        if not np.isfinite(r):
            return  # never let a NaN/inf reward poison the discounted statistics
        self._last_reward = r
        self._win_sum += r
        self._win_n += 1
        if self._win_n >= self._W:
            self._close_window()

    def _close_window(self) -> None:
        """Score the just-held arm and pick the next (one D-UCB decision epoch)."""
        mean_r = self._win_sum / float(self._win_n) if self._win_n else 0.0
        # discount ALL arms once per decision epoch (Garivier-Moulines D-UCB): old evidence decays so a later
        # regime shift can re-open a previously-dominated arm. Then add this window's evidence to the held arm.
        self._S *= self._gamma
        self._N *= self._gamma
        self._S[self._cur] += mean_r
        self._N[self._cur] += 1.0
        self._epochs += 1
        self._cur = self._select()
        self._win_sum = 0.0
        self._win_n = 0

    def _select(self) -> int:
        """D-UCB arm selection: any unpulled arm (discounted N == 0) is taken first (optimism -> +inf index);
        otherwise argmax of discounted-mean + c*sqrt(ln(sum N)/N). numpy does the reduction; ties break low."""
        untried = self._N <= 0.0
        if untried.any():
            return int(np.argmax(untried))  # first untried arm (argmax of a bool picks the lowest True index)
        total = float(self._N.sum())
        mean = self._S / self._N
        bonus = self._c * np.sqrt(np.log(total) / self._N)
        return int(np.argmax(mean + bonus))

    def act(self, obs: Observation) -> Sequence[int]:
        """Apply the CURRENTLY-selected learned threshold theta to every thread on the submit-pressure signal
        x[t] = ready[t] / max(1, D - inflight[t]): allow iff x[t] >= theta. Then force-allow any thread with
        nothing in flight (inflight==0 is an UNGATED forced flush — a deny is a no-op there). Cheap (O(T) numpy)
        and non-throwing — defaulted reads keep a malformed/short feature frame safe (the watchdog owns loudness
        on the hot path, ADR-0002)."""
        t = self._t
        feats = obs.features
        # length-T INSTANTANEOUS gauges (never first-differenced, so the [0]*T sentinel for an absent thread is
        # a harmless x=0, not a fabricated delta); tolerate a short/absent list defensively so act() never throws.
        ready = _fit(np.asarray(feats.get("ready", ()), dtype=np.float64), t)
        inflight = _fit(np.asarray(feats.get("inflight", ()), dtype=np.float64), t)

        theta = float(self._arms[self._cur])
        # submit pressure: backlog against remaining in-flight headroom. max(1, D - inflight) guards the divisor
        # (a saturated thread, inflight >= D, clamps to 1, so x = ready — a large pressure, the right shape: an
        # issue there is already a no-op under inflight<D, but the signal still says "this thread wants to fly").
        headroom = np.maximum(1.0, self._d - inflight)
        x = ready / headroom

        if theta == float("inf"):
            allow = np.zeros(t, dtype=bool)   # deny-until-forced: no finite pressure clears +inf
        else:
            allow = x >= theta                # the same learned theta gates every thread (parameter sharing)

        # liveness override (DENY-ONLY semantics): inflight==0 is an UNGATED forced flush -> a deny is a no-op,
        # so force allow (keep the override explicit rather than implicit).
        allow = allow | (inflight <= 0.0)

        return [1 if a else 0 for a in allow.tolist()]

    def metrics(self) -> Mapping[str, float]:
        """Dashboard scalars: the currently-learned theta, the best arm's discounted mean reward, how many
        decision epochs have closed, the last realized reward, and the per-arm discounted mean (keyed by arm
        index so the dashboard can render the learned value of each candidate threshold). Empty-safe."""
        cur_theta = float(self._arms[self._cur])
        pulled = self._N > 0.0
        means = np.divide(self._S, self._N, out=np.zeros_like(self._S), where=pulled)
        out: dict[str, float] = {
            "theta": cur_theta,
            "cur_arm": float(self._cur),
            "best_mean_reward": float(means[pulled].max()) if pulled.any() else 0.0,
            "epochs": float(self._epochs),
            "last_reward": float(self._last_reward),
        }
        # expose each arm's learned discounted mean (un-pulled arms report 0.0) for the dashboard sweep view.
        for i in range(self._a):
            out[f"arm{i}_mean"] = float(means[i]) if pulled[i] else 0.0
        return out


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
REGISTRY.setdefault("threshold_bandit", ThresholdBanditGate)
