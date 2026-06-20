#!/usr/bin/env python3
"""
cpp/stage_a/control_lab/methods/dual_coalescing.py — a dual / Lagrangian coalescing issue-gate
(online family) candidate for the issue-gate control lab.

Mechanism (two-timescale dual ascent on a soft coalescing constraint). The gate enforces a soft
per-thread constraint: the instantaneous coalescing degree S_inst[t] >= S*, where S_inst[t] is the
windowed (leaves-increment) / max(1, msgs-increment) of thread t — leaves coalesced per message, the
real coalescing the producer achieved. Both leaves and msgs are CUMULATIVE counters, so they are
first-differenced; and because the lab_server builds each length-T feature list fresh as [0]*T and fills
ONLY the threads SERVED this forward, an absent thread reads a SENTINEL 0 (not its true cumulative), so
the controller first-differences ONLY served threads against a per-thread baseline guarded by a `seen`
flag — an absent thread is never differenced against a stale baseline (which would manufacture a spurious
negative delta and a phantom violation).

  * FAST inner loop — the dual PRICE p. A shared scalar Lagrange multiplier on the coalescing constraint,
    adapted by dual ASCENT every forward: p <- clip(p + eta * mean_over_served(S* - S_inst[t]), 0, p_max).
    The price RISES when coalescing falls short of S* (the constraint is violated) and DECAYS toward 0 when
    threads over-coalesce (slack) — standard projected dual ascent (the multiplier is >= 0 for an
    inequality constraint).
  * The GATE. DENY[t] iff (p > cutoff) AND (S_inst[t] < S*): only when the price has risen past the cutoff
    AND this thread is itself under-coalescing do we hold its rows to fatten its next batch. ALLOW
    otherwise. The price is the global "is coalescing scarce right now" signal; the per-thread S_inst<S*
    test keeps the deny targeted at the threads actually short of the floor (a well-coalescing thread is
    never held just because a sibling lags).
  * SLOW outer loop — the setpoint S*. A scalar coordinate hill-climb driven by the POOL reward (the
    per-forward real row count fed to observe(); HIGHER IS BETTER). S* is HELD for a reward-hold window W
    forwards (W chosen to clear the pipeline lag so a setpoint change is scored on settled throughput, not
    on the transient); the mean reward over the window is compared to the previous window's mean. If
    raising S* improved the mean we keep climbing (step in the same direction); if it stalled/regressed we
    back off (reverse direction and shrink the step). Timescales are SEPARATED: the price relaxes EVERY
    forward (fast), the setpoint moves only every W forwards (slow), so the inner dual has time to
    equilibrate at each setpoint before the outer loop judges it.

S* corresponds to the runner's S_min producer coalescing floor (ctx.s_min): the outer loop is searching
for the in-band coalescing setpoint that maximizes pool throughput, initialized at / clamped around S_min.

Liveness override (DENY-ONLY gate semantics): the runner's effective gate is `inflight < D && allow` and
the forced flush at inflight==0 is UNGATED, so a deny is a NO-OP whenever inflight[t]==0 — we make that
explicit and force-allow there, so the dual gate never appears to starve a thread that has nothing in
flight. The first decision of a trial (no prior counters, no clock) allows every thread (the AllAllow
baseline) and seeds the per-thread baselines.

Online family: reset() clears all learner state (price, setpoint, window accumulators, baselines);
observe(reward, info) drives the SLOW setpoint search; act(obs) runs the fast price ascent + the gate.
O(T) per decision with numpy, non-throwing on the per-forward critical path (the watchdog owns loudness).
metrics() exposes p, S*, and the mean S_inst for the dashboard.

Knobs: eta (dual step), cutoff (price threshold above which the gate engages), s_star_init / s_star_step /
s_star_min / s_star_max (the setpoint search), W (reward-hold window), p_max (price clip), per_thread
(shared scalar price vs a per-thread price vector). s_min / K come from TrialContext.

Public Domain (The Unlicense).
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from control_lab.adapter import REGISTRY, Family, Observation, TrialContext


class DualCoalescingGate:
    """A two-timescale dual / Lagrangian coalescing issue gate (online). A fast dual price p does projected
    ascent on the soft constraint S_inst[t] >= S* every forward; a slow coordinate hill-climb adapts the
    setpoint S* by the pool reward over a hold window W. DENY[t] iff p > cutoff and S_inst[t] < S*; ALLOW
    otherwise; inflight[t]==0 force-allows (a deny is a no-op there). O(T) numpy, non-throwing."""

    family: Family = "online"

    def __init__(
        self,
        eta: float = 0.05,
        cutoff: float = 0.5,
        s_star_init: float | None = None,
        s_star_step: float = 0.5,
        s_star_min: float = 1.0,
        s_star_max: float = 8.0,
        window: int = 32,
        p_max: float = 10.0,
        per_thread: bool = False,
    ) -> None:
        # Construction-time validation: a degenerate step / window / bound is a wiring error, not a
        # per-forward surprise -> fail loud (ADR-0002) at the strongest surface (the ctor).
        if eta <= 0.0:
            raise ValueError(f"DualCoalescingGate: eta must be > 0, got {eta}")
        if window < 1:
            raise ValueError(f"DualCoalescingGate: window must be >= 1, got {window}")
        if s_star_step <= 0.0:
            raise ValueError(f"DualCoalescingGate: s_star_step must be > 0, got {s_star_step}")
        if not (s_star_min <= s_star_max):
            raise ValueError(
                f"DualCoalescingGate: need s_star_min <= s_star_max, got {s_star_min}, {s_star_max}"
            )
        if p_max <= 0.0:
            raise ValueError(f"DualCoalescingGate: p_max must be > 0, got {p_max}")
        self.name = f"dual_coalescing_e{eta:g}_c{cutoff:g}"
        self._eta = float(eta)
        self._cutoff = float(cutoff)
        self._s_star_init = s_star_init  # None -> seed from ctx.s_min at reset
        self._s_star_step0 = float(s_star_step)
        self._s_star_min = float(s_star_min)
        self._s_star_max = float(s_star_max)
        self._w = int(window)
        self._p_max = float(p_max)
        self._per_thread = bool(per_thread)

        # ---- per-run learner state (all sized / cleared in reset) ----
        self._t = 1
        self._s_min = 1
        self._k = 1
        # fast inner loop: the dual price (scalar broadcast, or a per-thread vector).
        self._p = np.zeros(1, dtype=np.float64)
        # slow outer loop: the setpoint S* and its signed search step.
        self._s_star = 1.0
        self._s_star_step = self._s_star_step0   # signed: +climb up, -climb down
        # reward-hold window accumulators for the slow loop.
        self._win_reward_sum = 0.0
        self._win_count = 0
        self._prev_win_mean: float | None = None
        # per-thread first-difference baselines (wire subtlety: difference only served threads).
        self._leaves_prev = np.zeros(1, dtype=np.int64)
        self._msgs_prev = np.zeros(1, dtype=np.int64)
        self._seen = np.zeros(1, dtype=bool)
        # last computed per-thread S_inst (held for absent threads; metrics + the gate read it).
        self._s_inst = np.zeros(1, dtype=np.float64)
        self._started = False   # has the first decision seeded the baselines?

    def reset(self, ctx: TrialContext) -> None:
        """Begin a fresh trial: size everything to T, clear every learner accumulator, and seed S* from the
        runner's S_min coalescing floor (clamped into the search band). s_min / K come from the out-of-band
        context (the feature wire omits them)."""
        self._t = int(ctx.n_threads)
        self._s_min = max(1, int(ctx.s_min))       # the coalescing floor S* is searched around
        self._k = max(1, int(ctx.k_per_thread))    # capacity normalizer (recorded; not on the gate path)

        init = float(self._s_min) if self._s_star_init is None else float(self._s_star_init)
        self._s_star = float(np.clip(init, self._s_star_min, self._s_star_max))
        self._s_star_step = self._s_star_step0     # restart climbing upward

        width = self._t if self._per_thread else 1
        self._p = np.zeros(width, dtype=np.float64)

        self._win_reward_sum = 0.0
        self._win_count = 0
        self._prev_win_mean = None

        self._leaves_prev = np.zeros(self._t, dtype=np.int64)
        self._msgs_prev = np.zeros(self._t, dtype=np.int64)
        self._seen = np.zeros(self._t, dtype=bool)
        self._s_inst = np.zeros(self._t, dtype=np.float64)
        self._started = False

    def observe(self, reward: float, info: Mapping[str, Any]) -> None:
        """SLOW outer loop. Accumulate the pool reward (per-forward real row count; higher is better) over a
        hold window of W forwards; when the window closes, compare its mean to the previous window's mean and
        take one coordinate hill-climb step on S* — keep the current direction if the mean improved, reverse
        and shrink if it stalled/regressed. Held W forwards so the change is scored on settled throughput,
        not on the pipeline transient. Non-throwing (defensive against a non-finite reward)."""
        r = float(reward)
        if not np.isfinite(r):
            return
        self._win_reward_sum += r
        self._win_count += 1
        if self._win_count < self._w:
            return

        win_mean = self._win_reward_sum / float(self._win_count)
        self._win_reward_sum = 0.0
        self._win_count = 0

        if self._prev_win_mean is None:
            # First closed window: no baseline to compare -> just record it and keep the initial direction.
            self._prev_win_mean = win_mean
            return

        if win_mean >= self._prev_win_mean:
            # The last setpoint move helped (or held): keep climbing in the same direction, same step.
            pass
        else:
            # Throughput regressed under the last move: reverse direction and shrink the step (damped
            # coordinate descent so the setpoint settles rather than oscillating).
            self._s_star_step = -0.5 * self._s_star_step

        self._s_star = float(
            np.clip(self._s_star + self._s_star_step, self._s_star_min, self._s_star_max)
        )
        self._prev_win_mean = win_mean

    def act(self, obs: Observation) -> Sequence[int]:
        """FAST inner loop + gate. First-difference leaves/msgs for the SERVED threads only -> per-thread
        S_inst; do one projected dual-ascent step on the price p from the mean served violation; then deny a
        thread iff (p > cutoff) and (S_inst[t] < S*). inflight[t]==0 force-allows (a deny is a no-op there).
        Cheap (O(T) numpy) and non-throwing — defaulted reads keep a malformed/short frame safe."""
        feats = obs.features
        T = self._t

        inflight = np.asarray(self._vec(feats.get("inflight"), T), dtype=np.int64)
        leaves = np.asarray(self._vec(feats.get("leaves"), T), dtype=np.int64)
        msgs = np.asarray(self._vec(feats.get("msgs"), T), dtype=np.int64)

        # served threads carry a TRUE cumulative reading this forward; absent threads read sentinel 0.
        served = [i for i in obs.served if 0 <= i < T]

        if not self._started:
            # First decision of the trial: no prior counters -> allow everyone (the AllAllow baseline) and
            # seed the per-thread leaves/msgs baselines for the threads we can see, so the FIRST real
            # increment next forward is measured from here, not from zero.
            for i in served:
                self._leaves_prev[i] = leaves[i]
                self._msgs_prev[i] = msgs[i]
                self._seen[i] = True
            self._started = True
            return [1] * T

        # --- S_inst[t] = (leaves delta) / max(1, msgs delta), SERVED threads only (wire first-difference) ---
        s_inst_now = self._s_inst.copy()   # absent threads hold their last S_inst (no spurious update)
        for i in served:
            if self._seen[i]:
                d_leaves = int(leaves[i] - self._leaves_prev[i])
                d_msgs = int(msgs[i] - self._msgs_prev[i])
                if d_msgs > 0:
                    # leaves coalesced per message this window = the coalescing degree achieved.
                    s_inst_now[i] = float(d_leaves) / float(d_msgs)
                # d_msgs == 0: thread sent nothing this window -> no fresh coalescing reading, hold prior.
            self._leaves_prev[i] = leaves[i]
            self._msgs_prev[i] = msgs[i]
            self._seen[i] = True
        self._s_inst = s_inst_now

        # --- FAST inner loop: projected dual ascent on the coalescing constraint S_inst >= S* ---
        # The subgradient of the (negated) Lagrangian w.r.t. the multiplier is (S* - S_inst): the price rises
        # on a shortfall, decays on slack, clipped to [0, p_max] (an inequality multiplier is >= 0).
        served_idx = np.asarray(served, dtype=np.int64)
        if served_idx.size:
            violation = self._s_star - self._s_inst[served_idx]   # >0 short of floor, <0 over-coalesced
            if self._per_thread:
                step = np.zeros(T, dtype=np.float64)
                step[served_idx] = violation
                self._p = np.clip(self._p + self._eta * step, 0.0, self._p_max)
            else:
                self._p = np.clip(
                    self._p + self._eta * float(violation.mean()), 0.0, self._p_max
                )

        # --- the gate: DENY iff price engaged (p > cutoff) AND this thread is under-coalescing (S_inst < S*) ---
        p_vec = self._p if self._per_thread else np.full(T, float(self._p[0]))
        under = self._s_inst < self._s_star
        engaged = p_vec > self._cutoff
        deny = under & engaged

        # liveness override (DENY-ONLY semantics): inflight==0 is an UNGATED forced flush, a deny is a no-op
        # there -> force allow so the dual gate never appears to starve a thread with nothing in flight.
        deny &= inflight > 0

        allow = ~deny
        return [1 if a else 0 for a in allow.tolist()]

    def metrics(self) -> Mapping[str, float]:
        """Dashboard scalars: the learned dual price (mean if per-thread), the current setpoint S*, the
        signed search step, and the mean per-thread coalescing degree S_inst."""
        return {
            "price": float(self._p.mean()) if self._p.size else 0.0,
            "s_star": float(self._s_star),
            "s_star_step": float(self._s_star_step),
            "mean_s_inst": float(self._s_inst.mean()) if self._s_inst.size else 0.0,
        }

    @staticmethod
    def _vec(v: Any, t: int) -> list[int]:
        """Coerce a feature entry to a length-T int list (truncate/zero-pad), tolerating None/short frames so
        act() never throws on a malformed observation (ADR-0002: the watchdog owns loudness on the hot path)."""
        if v is None:
            return [0] * t
        out = [int(x) for x in v[:t]]
        if len(out) < t:
            out.extend([0] * (t - len(out)))
        return out


# Register additively into the FROZEN adapter.REGISTRY (one entry + one class — P2 seam discipline; the
# harness + dashboard discover methods here). setdefault so a re-import or a name clash never silently
# clobbers an existing registration.
REGISTRY.setdefault("dual_coalescing", DualCoalescingGate)
