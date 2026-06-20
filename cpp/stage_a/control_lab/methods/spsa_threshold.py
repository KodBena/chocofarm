#!/usr/bin/env python3
"""
cpp/stage_a/control_lab/methods/spsa_threshold.py — an SPSA-tuned shared-threshold issue-gate
(ONLINE-LEARNING family) candidate for the issue-gate control lab.

The CONTINUOUS twin of the threshold bandit (methods/threshold_bandit.py). It keeps that method's gate
FORM exactly — one shared scalar threshold theta applied to every thread on the SUBMIT-PRESSURE signal

    x[t] = ready[t] / max(1, D - inflight[t])          (D = ctx.d_ceiling)
    allow[t] = 1  iff  x[t] >= theta

— but where the bandit picks theta from a fixed discrete arm set by discounted-UCB, this controller treats
theta as a CONTINUOUS parameter and optimizes it online by SPSA (Simultaneous-Perturbation Stochastic
Approximation, Spall 1992). One pool reward per forward (the forward's real row count — the coalescing
achieved, HIGHER IS BETTER) is the objective J(theta); parameter-sharing collapses the per-thread credit into
that one scalar, so the search is a one-dimensional stochastic maximization on THIS plant, this run.

MECHANISM (features -> gate). Identical to the bandit: the per-thread saturation signal is the submit
pressure x[t] above — instantaneous parked-at-leaf backlog against the remaining in-flight headroom (it rises
with backlog AND as a thread nears the in-flight ceiling, where each remaining slot is precious). The SAME
learned theta gates every thread (the parameter-sharing that makes one pool reward legitimate). Liveness
override (DENY-ONLY semantics — the runner's effective gate is `inflight < D && allow` and the forced flush at
inflight==0 is UNGATED, so a deny is a NO-OP there): inflight[t]==0 -> force allow. The theta box
[theta_min, theta_max] spans the two degenerate extremes that anchor the search to the baseline, exactly as
the bandit's arm extremes do: theta_min=0.0 makes x>=0 trivially true everywhere = ALL-ALLOW (byte-identical
to the AllAllow control arm), and theta_max is a large finite cap that denies all but the very highest-pressure
threads until the forced flush = (effectively) DENY-UNTIL-FORCED. Clipping theta into the box keeps every
perturbed evaluation a real, baseline-sandwiched gate.

THE LEARNER (observe(reward) -> SPSA update). SPSA estimates the gradient of a noisy scalar objective from
just TWO function evaluations per iteration, regardless of dimension — here, the two perturbed thresholds
theta +/- c_k along a single random sign. A measurement CYCLE (one SPSA iteration k) is a three-phase state
machine driven by window closes:

  * PLUS  : hold theta_plus  = clip(theta + c_k*delta, theta_min, theta_max) for a W-forward window; the
            window-mean reward is J_plus.
  * MINUS : hold theta_minus = clip(theta - c_k*delta, theta_min, theta_max) for a W-forward window; J_minus.
  * STEP  : form the simultaneous-perturbation gradient estimate and ascend (we MAXIMIZE throughput):

                g = (J_plus - J_minus) / (2 * c_k * delta)
                theta <- clip(theta + a_k * g, theta_min, theta_max)

            then k <- k+1, draw a fresh sign delta in {+1,-1}, and restart at PLUS.

delta is a Bernoulli +/-1 perturbation (the canonical SPSA distribution — bounded inverse moments, unlike a
Gaussian); for delta in {+1,-1} the identity 1/delta == delta holds, so g = (J_plus - J_minus)*delta/(2*c_k).
The CLASSIC Spall gain sequences decay the perturbation and step:

                a_k = a / (k + 1 + A)^alpha          c_k = c / (k + 1)^gamma

with the asymptotically-optimal exponents alpha=0.602, gamma=0.101 as the defaults, and the stability offset A
a small fraction of the run's iteration budget (so early steps are not wildly large). CRUCIALLY, act() applies
the CURRENT PHASE's perturbed theta (theta_plus during PLUS, theta_minus during MINUS, the running theta only
before the first cycle / during STEP), so each J +/- is measured under the gate actually in force — the
correctness crux of a finite-difference gradient.

REWARD-HOLD WINDOW W. A throughput change lags its gate by the pipeline depth, so scoring a perturbed theta on
the very next forward would credit it with the PREVIOUS theta's in-flight work. Each phase HOLDS its perturbed
theta for W forwards and scores it by the WINDOW-MEAN reward; W is the one knob that must clear the pipeline lag
without starving the short run (a full SPSA iteration costs 2*W forwards). The default is small (W=8) because
the run is only hundreds-to-thousands of forwards over a ~4s box.

WIRE SUBTLETY honored (same as the bandit). x[t] is read from the INSTANTANEOUS ready/inflight gauges, NEVER a
first-difference of a cumulative counter (msgs, leaves) — so the [0]*T sentinel a thread absent from this
forward reads (lab_server fills only served tids) cannot manufacture a spurious negative delta. A sentinel-0
ready merely yields x=0 for that thread (denied unless inflight==0 force-allows it), the honest "no observed
pressure" reading; it is never differenced. The reward is the harness's per-forward row count, so the learning
signal needs no wire decode at all, and rtt_us==0 (un-warmed) is never consulted by this gate.

ONLINE family: the per-run state is the running theta, the SPSA iteration index k, the current phase + drawn
sign + the J_plus it banks across the MINUS window, and the reward-hold-window accumulator — ALL cleared in
reset(). observe() folds the realized reward into the current window and advances the state machine on close;
act() applies the current (possibly perturbed) theta; metrics() exposes theta and the last gradient estimate
for the dashboard. O(1) per update and O(T) per decision (one numpy expression), non-throwing on the hot path.

Run the unit gate pinned + bounded:
    PYTHONPATH=cpp/stage_a /home/bork/w/vdc/venvs/generic/bin/python -m pytest tests/test_method_spsa_threshold.py -q

Public Domain (The Unlicense).
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from control_lab.adapter import REGISTRY, Family, Observation, TrialContext

# The three SPSA phases of one measurement cycle: evaluate theta+ , evaluate theta- , then step theta.
_PHASE_PLUS = 0
_PHASE_MINUS = 1


class SpsaThresholdGate:
    """An SPSA-tuned shared-threshold issue gate (ONLINE). One CONTINUOUS shared threshold theta is optimized
    online by SPSA (Spall) and applied to EVERY thread on the submit-pressure signal
    x[t] = ready[t] / max(1, D - inflight[t]): allow[t] = 1 iff x[t] >= theta. Each measurement cycle holds
    theta +/- c_k for a W-forward window, scores each by its window-mean reward, forms the finite-difference
    gradient g = (J+ - J-)/(2 c_k delta), and ascends theta <- clip(theta + a_k g, theta_min, theta_max) with
    the classic gains a_k = a/(k+1+A)^alpha, c_k = c/(k+1)^gamma. theta is clipped into
    [theta_min, theta_max] = [all-allow .. deny-until-forced]; inflight[t]==0 force-allows (a deny is a no-op
    there). The continuous twin of ThresholdBanditGate. O(T) numpy on the decision path, non-throwing."""

    family: Family = "online"

    def __init__(
        self,
        a: float = 0.30,
        c: float = 0.50,
        A: float = 10.0,
        alpha: float = 0.602,
        gamma: float = 0.101,
        hold_window: int = 8,
        theta0: float = 0.5,
        theta_min: float = 0.0,
        theta_max: float = 8.0,
        seed: int = 0,
    ) -> None:
        # fail loud (ADR-0002): a degenerate gain / window / box is a CONSTRUCTION error, surfaced at build
        # time on the strongest applicable surface (the ctor), never a silent runtime surprise on the
        # per-forward hot path. The classic SPSA gains require a, c > 0 and a non-negative stability offset A;
        # the asymptotic theory wants alpha, gamma in (0, 1]; the box must be a real interval the start sits in.
        if not (a > 0.0):
            raise ValueError(f"SpsaThresholdGate: step gain a must be > 0, got {a}")
        if not (c > 0.0):
            raise ValueError(f"SpsaThresholdGate: perturbation gain c must be > 0, got {c}")
        if A < 0.0:
            raise ValueError(f"SpsaThresholdGate: stability offset A must be >= 0, got {A}")
        if not (0.0 < alpha <= 1.0):
            raise ValueError(f"SpsaThresholdGate: alpha must be in (0, 1], got {alpha}")
        if not (0.0 < gamma <= 1.0):
            raise ValueError(f"SpsaThresholdGate: gamma must be in (0, 1], got {gamma}")
        if hold_window < 1:
            raise ValueError(f"SpsaThresholdGate: hold_window W must be >= 1, got {hold_window}")
        if not (theta_min <= theta_max):
            # an inverted box is a construction error; a zero-width box (theta_min == theta_max) is the
            # legitimate degenerate "frozen theta" config (clipping pins the applied threshold exactly there),
            # so it is allowed — only theta_min > theta_max is refused.
            raise ValueError(
                f"SpsaThresholdGate: need theta_min <= theta_max, got [{theta_min}, {theta_max}]"
            )
        if not (theta_min <= theta0 <= theta_max):
            raise ValueError(
                f"SpsaThresholdGate: theta0={theta0} must lie in [{theta_min}, {theta_max}]"
            )

        self._a = float(a)
        self._c = float(c)
        self._A = float(A)
        self._alpha = float(alpha)
        self._gamma = float(gamma)
        self._W = int(hold_window)
        self._theta0 = float(theta0)
        self._theta_min = float(theta_min)
        self._theta_max = float(theta_max)
        self._seed = int(seed)
        self.name = f"spsa_threshold_W{self._W}_a{self._a:g}_c{self._c:g}"

        # per-run learner state (all sized/cleared in reset). The running theta, the SPSA iteration index k,
        # the phase state machine (current phase + the sign delta drawn for THIS cycle + the J_plus banked
        # across the MINUS window), the reward-hold-window accumulator, and dashboard-only scalars.
        self._t = 1
        self._d = 1
        self._rng = np.random.default_rng(self._seed)
        self._theta = self._theta0
        self._k = 0                     # SPSA iteration index (one per completed +/- /step cycle)
        self._phase = _PHASE_PLUS       # which perturbation this window is measuring
        self._delta = 1.0               # the +/-1 sign drawn for the current cycle
        self._j_plus = 0.0              # J(theta+) banked at the PLUS->MINUS transition
        self._win_sum = 0.0             # reward accumulated over the current hold window
        self._win_n = 0                 # forwards observed in the current hold window
        self._last_grad = 0.0           # last finite-difference gradient estimate (metrics)
        self._last_reward = 0.0

    def reset(self, ctx: TrialContext) -> None:
        """Begin a fresh trial: capture the geometry (T, D) the submit-pressure signal needs and clear ALL SPSA
        state. theta starts at theta0 (near the all-allow / low-threshold end) and the first cycle's sign is
        drawn fresh, so the un-warmed warm-up forwards default to a near-baseline gate before any gradient is
        formed. The RNG is re-seeded from ctx.seed so a trial is reproducible (the perturbation signs are the
        only stochastic input to the learner)."""
        self._t = int(ctx.n_threads)
        self._d = max(1, int(ctx.d_ceiling))               # headroom denominator guard: max(1, D - inflight)
        # ctx.seed mixes with the construction seed so distinct trials of the same controller differ, yet each
        # is reproducible (SPSA's only randomness is the +/-1 perturbation sign sequence).
        self._rng = np.random.default_rng((self._seed, int(ctx.seed)))
        self._theta = self._theta0
        self._k = 0
        self._phase = _PHASE_PLUS
        self._delta = self._draw_sign()
        self._j_plus = 0.0
        self._win_sum = 0.0
        self._win_n = 0
        self._last_grad = 0.0
        self._last_reward = 0.0

    def _draw_sign(self) -> float:
        """A Bernoulli +/-1 perturbation sign (the canonical SPSA distribution — symmetric, bounded inverse
        moments). delta in {+1,-1} also gives the 1/delta == delta identity the step exploits."""
        return 1.0 if self._rng.random() < 0.5 else -1.0

    def _c_k(self) -> float:
        """Perturbation gain c_k = c/(k+1)^gamma (classic Spall). Strictly positive, so the gradient divisor
        2*c_k*delta is never zero (delta in {+1,-1})."""
        return float(self._c / float(self._k + 1) ** self._gamma)

    def _a_k(self) -> float:
        """Step gain a_k = a/(k+1+A)^alpha (classic Spall). The stability offset A damps the early steps."""
        return float(self._a / float(self._k + 1 + self._A) ** self._alpha)

    def observe(self, reward: float, info: Mapping[str, Any]) -> None:
        """Fold the realized PER-FORWARD reward (the forward's row count; higher is better) into the current
        SPSA measurement window. On window close (W forwards held) advance the state machine: PLUS banks J_plus
        and switches to MINUS; MINUS forms the finite-difference gradient and ascends theta, then opens the next
        cycle's PLUS. Cheap (O(1)) and total — a non-finite reward is ignored rather than poisoning the
        estimate (ADR-0002: the watchdog owns loudness; the learner stays well-defined)."""
        r = float(reward)
        if not np.isfinite(r):
            return  # never let a NaN/inf reward poison the gradient estimate
        self._last_reward = r
        self._win_sum += r
        self._win_n += 1
        if self._win_n >= self._W:
            self._close_window()

    def _close_window(self) -> None:
        """Score the just-held perturbed theta by its window-mean reward and advance the SPSA state machine."""
        mean_r = self._win_sum / float(self._win_n) if self._win_n else 0.0
        self._win_sum = 0.0
        self._win_n = 0
        if self._phase == _PHASE_PLUS:
            # finished measuring J(theta+): bank it and move on to measure J(theta-) at the same sign.
            self._j_plus = mean_r
            self._phase = _PHASE_MINUS
            return
        # finished measuring J(theta-): form the simultaneous-perturbation gradient and ascend (MAXIMIZE).
        ck = self._c_k()
        ak = self._a_k()
        # g = (J+ - J-)/(2 c_k delta); delta in {+1,-1} so 1/delta == delta (a divide-by-zero-free form).
        grad = (self._j_plus - mean_r) * self._delta / (2.0 * ck)
        self._theta = float(np.clip(self._theta + ak * grad, self._theta_min, self._theta_max))
        self._last_grad = float(grad)
        # close the cycle: advance k, draw a fresh sign, reopen at PLUS.
        self._k += 1
        self._phase = _PHASE_PLUS
        self._delta = self._draw_sign()
        self._j_plus = 0.0

    def _current_theta(self) -> float:
        """The threshold the gate APPLIES this forward: the perturbed theta of the phase being measured
        (theta+ in PLUS, theta- in MINUS), clipped into the box. Measuring J +/- under exactly the gate in
        force is the correctness crux of the finite-difference estimate."""
        ck = self._c_k()
        if self._phase == _PHASE_PLUS:
            theta = self._theta + ck * self._delta
        else:
            theta = self._theta - ck * self._delta
        return float(np.clip(theta, self._theta_min, self._theta_max))

    def act(self, obs: Observation) -> Sequence[int]:
        """Apply the CURRENT (perturbed) learned threshold theta to every thread on the submit-pressure signal
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

        theta = self._current_theta()
        # submit pressure: backlog against remaining in-flight headroom. max(1, D - inflight) guards the divisor
        # (a saturated thread, inflight >= D, clamps to 1, so x = ready — a large pressure, the right shape: an
        # issue there is already a no-op under inflight<D, but the signal still says "this thread wants to fly").
        headroom = np.maximum(1.0, self._d - inflight)
        x = ready / headroom

        allow = x >= theta                # the same learned theta gates every thread (parameter sharing)
        # liveness override (DENY-ONLY semantics): inflight==0 is an UNGATED forced flush -> a deny is a no-op,
        # so force allow (keep the override explicit rather than implicit).
        allow = allow | (inflight <= 0.0)

        return [1 if v else 0 for v in allow.tolist()]

    def metrics(self) -> Mapping[str, float]:
        """Dashboard scalars: the running learned theta, the threshold the gate is APPLYING this forward (the
        perturbed value), the last finite-difference gradient estimate, the SPSA iteration index k, the current
        phase and perturbation sign, the live gains, and the last realized reward. Empty-safe."""
        return {
            "theta": float(self._theta),
            "theta_applied": self._current_theta(),
            "last_grad": float(self._last_grad),
            "k": float(self._k),
            "phase": float(self._phase),
            "delta": float(self._delta),
            "a_k": self._a_k(),
            "c_k": self._c_k(),
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
REGISTRY.setdefault("spsa_threshold", SpsaThresholdGate)
