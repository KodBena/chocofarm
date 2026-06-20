#!/usr/bin/env python3
"""
cpp/stage_a/control_lab/methods/vegas.py — an RTT-driven, delay-based congestion controller
(TCP-Vegas style) issue-gate (static family) candidate for the issue-gate control lab.

The cleanest single-thread fit for the single-threaded M/G/1 coalescing eval server: the reply RTT is a
direct read of how backed-up that server is, so a per-thread delay-based controller throttles exactly when
the queue builds and releases when it drains — no row-counting proxy, just the latency the wire now reports.

Mechanism (per producer thread t, mirroring TCP Vegas's BaseRTT / queueing-delay band):

  * RTT_min[t] — the zero-queue baseline, the analog of Vegas's BaseRTT. Estimated ROBUSTLY as a low
    percentile (default 10th) of a rolling window of that thread's recent rtt_us samples, NOT a single raw
    min. A raw min is anchored permanently by the first transient that dips low; a one-time COLD-COMPILE
    service spike inflates a *sample* but a low percentile of the window ignores it (it is a high outlier),
    and conversely a single spuriously-low sample cannot drag the floor down because the percentile needs a
    mass of low samples, not one. The window also lets the baseline TRACK a genuine regime shift (e.g. after
    warm-up the steady-state floor settles) instead of being pinned to a cold-start artifact forever.
  * queueing q[t] = rtt_us[t] - RTT_min[t] — the in-server delay above the zero-queue baseline (Vegas's
    Diff). Optionally scaled by inflight[t] as a work-in-system / Little's-law proxy (use_inflight knob):
    rtt_us[t] * inflight[t] vs RTT_min[t] * inflight[t], i.e. q[t] = (rtt_us[t] - RTT_min[t]) * inflight[t],
    so a thread with several messages outstanding is weighted by the work it represents.
  * the band [alpha, beta] (microseconds), with hysteresis. q > beta -> the server is backing up for this
    thread -> DENY (throttle, let it drain). q <= alpha -> slack -> ALLOW. Between alpha and beta the prior
    per-thread gate is HELD (the deadband that kills per-forward chatter, the same hysteresis Vegas's stable
    region gives between its two thresholds).

Two liveness overrides force allow (DENY-ONLY gate semantics: the runner's effective gate is
`inflight < D && allow`, and the forced flush at inflight==0 is UNGATED):
  (i)  inflight[t]==0 — a deny is a NO-OP there, so force-allow to keep the override explicit, and it is also
       a clean drain point: nothing is queued, so the queueing estimate is meaningless anyway;
  (ii) rtt_us[t]==0 — the un-warmed sentinel. The feature wire reports rtt_us only for the threads SERVED in
       this forward (an absent thread, and a thread on its genuine first forward, both read 0). Acting on
       that zero would compute q = -RTT_min < alpha and spuriously ALLOW on a fabricated reading — and worse,
       feeding the 0 into the RTT_min window would poison the baseline. So a 0 is NEVER sampled into the
       window and NEVER decided on: the thread holds its prior gate (defaulting to allow), the silent-failure
       trap the brief names.

Static / deterministic / O(T) with numpy on the decision path; observe() is a no-op (no reward learning).
The only per-run state is the per-thread RTT window, the hysteresis bit, and the last-seen estimates; all
sized and cleared in reset(). Knobs: alpha, beta (the band, microseconds), the RTT_min window length and
percentile, and use_inflight (the Little's-law weighting). metrics() reports the mean RTT_min, mean
queueing, and the band for the dashboard.

Run the unit gate pinned + bounded:
    PYTHONPATH=cpp/stage_a /home/bork/w/vdc/venvs/generic/bin/python -m pytest tests/test_method_vegas.py -q

Public Domain (The Unlicense).
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from control_lab.adapter import REGISTRY, Family, Observation, TrialContext


class VegasDelayGate:
    """A per-thread, delay-based (TCP-Vegas style) issue gate (static). Tracks a robust zero-queue baseline
    RTT_min[t] (a low percentile of a rolling rtt_us window, cold-compile-aware), estimates in-server
    queueing q[t] = rtt_us[t] - RTT_min[t] (optionally * inflight[t], Little's law), and keeps it in a band
    [alpha, beta] with hysteresis: q > beta -> deny, q <= alpha -> allow, in-between -> hold prior. Force-allows
    on the two liveness sentinels (inflight==0; rtt_us==0 un-warmed). O(T) numpy, non-throwing on the
    per-forward path."""

    family: Family = "static"

    def __init__(
        self,
        alpha: float = 150.0,
        beta: float = 600.0,
        window: int = 64,
        quantile: float = 0.10,
        use_inflight: bool = False,
    ) -> None:
        # fail loud (ADR-0002): a degenerate band / window / quantile is a construction error, surfaced at
        # build time, not a silent runtime surprise on the hot path.
        if not (0.0 <= alpha < beta):
            raise ValueError(f"VegasDelayGate: need 0 <= alpha < beta (microseconds), got alpha={alpha}, beta={beta}")
        if window < 1:
            raise ValueError(f"VegasDelayGate: window must be >= 1, got {window}")
        if not (0.0 <= quantile <= 1.0):
            raise ValueError(f"VegasDelayGate: quantile must be in [0, 1], got {quantile}")
        self.name = f"vegas_a{alpha:g}_b{beta:g}"
        self._alpha = float(alpha)
        self._beta = float(beta)
        self._window = int(window)
        self._quantile = float(quantile)          # numpy.percentile takes 0..100
        self._pct = 100.0 * float(quantile)
        self._use_inflight = bool(use_inflight)
        # per-run state (all sized/cleared in reset).
        self._t = 1
        self._rtt_win: list[list[float]] = [[]]   # per-thread rolling window of warmed rtt_us samples
        self._allow = np.ones(1, dtype=np.int8)    # sticky hysteresis bit (1 = allow), start = allow
        self._rtt_min = np.zeros(1, dtype=np.float64)  # last-computed baseline (for metrics)
        self._q = np.zeros(1, dtype=np.float64)        # last-computed queueing estimate (for metrics)

    def reset(self, ctx: TrialContext) -> None:
        """Begin a fresh trial: size the per-thread RTT windows + hysteresis state to T and clear every
        accumulator. The Vegas gate reads only the feature wire (rtt_us, inflight); no out-of-band geometry
        is needed beyond the thread count."""
        self._t = int(ctx.n_threads)
        self._rtt_win = [[] for _ in range(self._t)]
        self._allow = np.ones(self._t, dtype=np.int8)        # every thread starts allowing (reproduces baseline)
        self._rtt_min = np.zeros(self._t, dtype=np.float64)
        self._q = np.zeros(self._t, dtype=np.float64)

    def observe(self, reward: float, info: Mapping[str, Any]) -> None:
        """No-op: a static delay-based gate does not learn from the realized reward."""
        return None

    def act(self, obs: Observation) -> Sequence[int]:
        """Sample the warmed rtt_us into each thread's window, recompute RTT_min as a low percentile, estimate
        queueing, and apply the banded hysteresis with the liveness overrides. Cheap (O(T) numpy) and
        non-throwing — defaulted reads keep a malformed/short feature frame safe (ADR-0002: the watchdog owns
        loudness on the hot path)."""
        t = self._t
        feats = obs.features
        rtt = _fit(np.asarray(feats.get("rtt_us", ()), dtype=np.float64), t)
        inflight = _fit(np.asarray(feats.get("inflight", ()), dtype=np.float64), t)

        # warmed = a real (served, post-warm-up) RTT reading: strictly positive. A 0 is the un-warmed/absent
        # sentinel — it is NEVER sampled into the window (would poison the baseline) and NEVER decided on.
        warmed = rtt > 0.0

        # --- update the per-thread RTT window + recompute the robust baseline (low percentile) ---
        rtt_min = self._rtt_min.copy()
        for i in range(t):
            if warmed[i]:
                w = self._rtt_win[i]
                w.append(float(rtt[i]))
                if len(w) > self._window:
                    del w[0]                       # ring-buffer: drop the oldest sample
            w = self._rtt_win[i]
            if w:
                # low percentile of the window: robust to a one-time cold-compile high spike (a high outlier)
                # and to a lone low transient (needs a mass of low samples, not one) — np does the reduction.
                rtt_min[i] = float(np.percentile(w, self._pct))
            # else: no warmed sample seen yet -> keep the prior baseline (0); the thread is liveness-forced below.
        self._rtt_min = rtt_min

        # --- queueing estimate q[t] = rtt - RTT_min, optionally * inflight (Little's-law work-in-system) ---
        q = rtt - rtt_min
        if self._use_inflight:
            q = q * inflight
        # only the warmed threads carry a real q; an un-warmed thread's q is a fabricated -RTT_min and is
        # recorded as NaN so metrics() and the band update both ignore it (never a decided-on value).
        q = np.where(warmed, q, np.nan)
        self._q = q

        # --- banded hysteresis on the WARMED threads only: q > beta -> deny, q <= alpha -> allow, between ->
        # hold prior. The sticky bit (self._allow) is updated ONLY where a real reading exists, so an
        # un-warmed thread's fabricated q can never poison the hysteresis state carried into the next forward.
        gate = self._allow.copy()
        gate[warmed & (q > self._beta)] = 0         # server backing up for this thread -> throttle
        gate[warmed & (q <= self._alpha)] = 1       # slack -> release
        # inside (alpha, beta], AND every un-warmed thread: gate stays == prior (the deadband / held baseline).
        self._allow = gate
        decision = gate.copy()

        # --- liveness override (i): inflight==0 is an UNGATED forced flush — a deny is a no-op there, so
        # force-allow explicitly. (Override (ii), the un-warmed sentinel, is already honored above: un-warmed
        # threads were excluded from the band update and simply hold their prior gate, which starts at allow.)
        decision[inflight <= 0.0] = 1

        return decision.astype(np.int64).tolist()

    def metrics(self) -> Mapping[str, float]:
        """Dashboard scalars: the mean robust baseline RTT_min, the mean queueing estimate (over the WARMED
        threads only — un-warmed slots are NaN and excluded), and the band."""
        # _q is NaN on un-warmed threads; nanmean over an all-NaN array is meaningless, so report 0.0 there.
        any_warmed = bool(self._q.size) and bool(np.isfinite(self._q).any())
        return {
            "mean_rtt_min": float(self._rtt_min.mean()) if self._rtt_min.size else 0.0,
            "mean_queueing": float(np.nanmean(self._q)) if any_warmed else 0.0,
            "alpha": self._alpha,
            "beta": self._beta,
        }


def _fit(x: np.ndarray, t: int) -> np.ndarray:
    """Coerce a feature array to length T: truncate if long, zero-pad if short. Defensive so act() never
    throws on a malformed/empty feature list (ADR-0002: the per-forward path stays cheap and total; the
    zero-pad lands a thread in the un-warmed/inflight==0 liveness path, i.e. force-allow)."""
    if x.shape[0] == t:
        return x
    out = np.zeros(t, dtype=np.float64)
    n = min(x.shape[0], t)
    if n:
        out[:n] = x[:n]
    return out


# Register additively into the FROZEN adapter.REGISTRY (one entry + one class — P2 seam discipline; the
# harness + dashboard discover methods here). setdefault so a re-import or a name clash never silently
# clobbers an existing registration.
REGISTRY.setdefault("vegas", VegasDelayGate)
