#!/usr/bin/env python3
"""
cpp/stage_a/control_lab/methods/bang_bang.py — the bang-bang / (s,S) hysteresis issue-gate (static family).

The in-band, per-thread, software mirror of the static S_min producer coalescing floor (the lead static
pick). Each thread carries a 1-bit hysteresis state (allow/deny) driven by its ready-backlog fraction
r[t] = ready[t] / max(1, K), where K = ctx.k_per_thread is the capacity normalizer the feature wire omits.
Two thresholds r_lo < r_hi bound a deadband: r[t] >= r_hi banks enough backlog to let the fat batch fly
(set allow); r[t] <= r_lo holds to accumulate (set deny); INSIDE the deadband the prior per-thread state is
kept — the hysteresis that kills the per-forward chatter a single threshold would produce.

Two liveness overrides force allow (DENY-ONLY gate semantics: the runner's effective gate is
`inflight < D && allow`, and the forced flush at inflight==0 is UNGATED): (i) inflight[t]==0 — a deny is a
no-op there, so always force-allow to keep the override explicit rather than implicit; (ii) a hold-timeout
H — a thread denied for more than H consecutive decisions is force-allowed once (and its deny streak reset),
bounding the worst-case hold so a thread starved of ready growth cannot wedge forever.

Static / deterministic / O(T) with numpy: observe() is a no-op (no learning); the only per-run state is the
per-thread hysteresis bit and the consecutive-deny counter, both cleared in reset().

Public Domain (The Unlicense).
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from control_lab.adapter import REGISTRY, Family, Observation, TrialContext


class BangBangHysteresisGate:
    """Per-thread bang-bang / (s,S) hysteresis gate on the ready-backlog fraction r[t]=ready[t]/max(1,K).

    Deadband r_lo < r_hi with a sticky 1-bit per-thread state (start=allow): r>=r_hi -> allow, r<=r_lo ->
    deny, in-between -> hold prior. Liveness overrides force allow at inflight==0 (deny is a no-op there)
    and after H consecutive denies (a hold-timeout that bounds the worst-case hold). Static family; cheap,
    non-throwing act on the per-forward path."""

    family: Family = "static"

    def __init__(self, r_lo: float = 0.25, r_hi: float = 0.60, hold_timeout: int = 8) -> None:
        if not (0.0 <= r_lo < r_hi):
            # fail loud (ADR-0002): a degenerate / inverted deadband is a construction error, not a runtime
            # surprise — r_lo must be a real lower threshold strictly below r_hi.
            raise ValueError(f"BangBangHysteresisGate: need 0 <= r_lo < r_hi, got r_lo={r_lo}, r_hi={r_hi}")
        if hold_timeout < 1:
            raise ValueError(f"BangBangHysteresisGate: hold_timeout must be >= 1, got {hold_timeout}")
        self.name = f"bang_bang_{r_lo:g}_{r_hi:g}"
        self._r_lo = float(r_lo)
        self._r_hi = float(r_hi)
        self._hold_timeout = int(hold_timeout)
        # per-run state (cleared/sized in reset): hysteresis bit (1=allow) + consecutive-deny streak.
        self._t = 1
        self._k = 1
        self._allow = np.ones(1, dtype=np.int8)        # start = allow
        self._deny_streak = np.zeros(1, dtype=np.int64)
        self._denied = 0                               # cumulative deny count emitted (metrics)

    def reset(self, ctx: TrialContext) -> None:
        self._t = int(ctx.n_threads)
        self._k = max(1, int(ctx.k_per_thread))        # capacity normalizer; guard the max(1,K) divisor
        self._allow = np.ones(self._t, dtype=np.int8)  # every thread starts allowing (reproduces baseline)
        self._deny_streak = np.zeros(self._t, dtype=np.int64)
        self._denied = 0

    def observe(self, reward: float, info: Mapping[str, Any]) -> None:
        pass  # static: no learning from the (s, a, r) transition

    def act(self, obs: Observation) -> Sequence[int]:
        t = self._t
        feats = obs.features
        # length-T features; tolerate a short/absent list defensively (cheap, non-throwing per the contract).
        ready = np.asarray(feats.get("ready", ()), dtype=np.float64)
        inflight = np.asarray(feats.get("inflight", ()), dtype=np.float64)
        ready = _fit(ready, t)
        inflight = _fit(inflight, t)

        # ready-backlog fraction and the two-threshold hysteresis update (vectorized over threads).
        r = ready / float(self._k)
        prior = self._allow
        gate = prior.copy()
        gate[r >= self._r_hi] = 1          # enough banked: let the fat batch fly
        gate[r <= self._r_lo] = 0          # too little: hold to accumulate
        # inside (r_lo, r_hi): gate stays == prior (the deadband that kills chatter).
        self._allow = gate

        decision = gate.copy()

        # Liveness override (ii): hold-timeout. A thread denied for > H consecutive decisions is force-allowed
        # once and its streak reset, bounding the worst-case hold. Update the streak from THIS forward's
        # hysteresis decision first, then fire the timeout where it is now exceeded.
        denied_now = decision == 0
        self._deny_streak = np.where(denied_now, self._deny_streak + 1, 0)
        timeout_fire = self._deny_streak > self._hold_timeout
        decision[timeout_fire] = 1
        self._deny_streak[timeout_fire] = 0

        # Liveness override (i): inflight==0 is an UNGATED forced flush — a deny is a no-op there, so make it
        # explicit by force-allowing (and clear the streak so the flush counts as drain progress).
        flush = inflight <= 0.0
        decision[flush] = 1
        self._deny_streak[flush] = 0

        self._denied += int(np.count_nonzero(decision == 0))
        return decision.astype(np.int64).tolist()

    def metrics(self) -> Mapping[str, float]:
        return {"r_lo": self._r_lo, "r_hi": self._r_hi, "denied": float(self._denied)}


def _fit(x: np.ndarray, t: int) -> np.ndarray:
    """Coerce a feature array to length T: truncate if long, zero-pad if short. Defensive so act() never
    throws on a malformed/empty feature list (ADR-0002: the per-forward path stays cheap and total)."""
    if x.shape[0] == t:
        return x
    out = np.zeros(t, dtype=np.float64)
    n = min(x.shape[0], t)
    if n:
        out[:n] = x[:n]
    return out


# Register additively (P2 seam: one entry + one class, no edit elsewhere). setdefault so a re-import or a
# harness-side override never clobbers an existing registration.
REGISTRY.setdefault("bang_bang", BangBangHysteresisGate)
