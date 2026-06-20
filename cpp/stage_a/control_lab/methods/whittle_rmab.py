#!/usr/bin/env python3
"""
cpp/stage_a/control_lab/methods/whittle_rmab.py — a Whittle-index restless-bandit issue-gate (static family)
candidate for the issue-gate control lab.

Mechanism. Treat each producer thread t as a restless-bandit ARM with instantaneous state (j, W) read off
the feature wire: j = inflight[t] (in-flight messages — the active work) and W = ready[t] (parked-at-leaf
backlog awaiting issue — the queued reward). ONE shared index function prices every arm (parameter-sharing,
not a per-arm table): a tractable Whittle-STYLE index = the marginal value of ACTIVATING (allowing) arm t
this forward,

    index[t] = (W / max(1, K)) * ((D - j) / max(1, D))

with K = ctx.k_per_thread (the capacity normalizer the feature wire omits) and D = ctx.d_ceiling (the
per-thread in-flight cap). The first factor is normalized available backlog (more ready work -> more value
in releasing it); the second is normalized headroom that vanishes as j -> D, since under the DENY-ONLY gate
(`inflight < D && allow`) an issue from a saturated arm is already a no-op, so its activation value is ~0.
The product makes the index vanish when EITHER there is no backlog OR no headroom — the honest "do not bother
activating" signal. (Indexability is heuristic in the depth>1 regime: this is a candidate to MEASURE, not a
proved index.)

Coordinated gate (the coupling the per-thread methods lack). The one shared eval server couples the arms:
only so many should push the shared forward at once. So we ALLOW the top ceil(p*T) arms by index and DENY the
rest — a cross-thread-coordinated activation budget that prices the shared-server contention. An arm whose
index is 0 (no backlog, or saturated) is never activated just to fill the quota: the effective allow set is
(top ceil(p*T)) intersect {index > 0}. A threshold mode is also available (p=None, theta=...): allow iff
index >= theta. Liveness override (DENY-ONLY semantics — the forced flush at inflight==0 is UNGATED, so a
deny is a no-op there): inflight[t]==0 -> force allow, so a thread with nothing in flight is never starved.

STATIC family: the index is recomputed fresh from the live (j, W) each forward and parameter-sharing means
there is no per-arm learned state, so observe() is a no-op. The only per-run state is the geometry (T, D, K)
and the last indices/activation count for the dashboard, all cleared in reset(). O(T) per decision (one numpy
expression + an argpartition for the top-p set), non-throwing — it rides the per-forward critical path.

Public Domain (The Unlicense).
"""
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np

from control_lab.adapter import REGISTRY, Family, Observation, TrialContext


class WhittleIndexGate:
    """Whittle-index restless-bandit issue gate (static). One shared index function prices every thread-arm
    from its instantaneous state (j=inflight, W=ready): index = (W/max(1,K)) * ((D-j)/max(1,D)). Allows the
    top ceil(p*T) arms with a positive index (the cross-thread activation budget pricing the shared-server
    coupling), or — in threshold mode — every arm with index >= theta. inflight[t]==0 force-allows (a deny is
    a no-op there). O(T) numpy, non-throwing on the per-forward path."""

    family: Family = "static"

    def __init__(self, p: float | None = 0.6, theta: float | None = None) -> None:
        # Exactly one selection rule must be live: a top-p activation budget OR an absolute index threshold.
        # A degenerate config is a CONSTRUCTION error, not a per-forward surprise (ADR-0002: fail loud at the
        # strongest applicable surface — the ctor — keeping act() itself cheap and total).
        if p is not None and theta is not None:
            raise ValueError(f"WhittleIndexGate: pass p OR theta, not both (got p={p}, theta={theta})")
        if p is None and theta is None:
            raise ValueError("WhittleIndexGate: one of p (top-fraction) or theta (index threshold) is required")
        if p is not None and not (0.0 < p <= 1.0):
            raise ValueError(f"WhittleIndexGate: p must be in (0, 1], got {p}")
        if theta is not None and theta < 0.0:
            raise ValueError(f"WhittleIndexGate: theta must be >= 0, got {theta}")

        self._p = None if p is None else float(p)
        self._theta = None if theta is None else float(theta)
        mode = f"p{self._p:g}" if self._p is not None else f"th{self._theta:g}"
        self.name = f"whittle_rmab_{mode}"

        # Per-run state (sized/cleared in reset). The arm geometry D, K and T come from TrialContext; the last
        # index vector + activation count are kept only for the dashboard (metrics), never read back as policy.
        self._t = 1
        self._d = 1
        self._k = 1
        self._last_index = np.zeros(1, dtype=np.float64)
        self._last_active = 0

    def reset(self, ctx: TrialContext) -> None:
        """Begin a fresh trial: capture the arm geometry (T, D, K) from the out-of-band context and clear the
        dashboard state. D / K guard their max(1, .) divisors so a degenerate trial never divides by < 1."""
        self._t = int(ctx.n_threads)
        self._d = max(1, int(ctx.d_ceiling))       # headroom denominator (D - j) / max(1, D)
        self._k = max(1, int(ctx.k_per_thread))     # backlog normalizer W / max(1, K)
        self._last_index = np.zeros(self._t, dtype=np.float64)
        self._last_active = 0

    def observe(self, reward: float, info: Mapping[str, Any]) -> None:
        """No-op: the index is recomputed from the live (j, W) each forward and the shared index function has
        no per-arm learned parameters, so there is nothing to update from the realized reward (static)."""
        return None

    def act(self, obs: Observation) -> Sequence[int]:
        """Price every arm with the shared Whittle-style index, activate the coordinated allow set (top-p with
        a positive index, or index >= theta), then force-allow any arm with nothing in flight. Cheap (O(T)
        numpy) and non-throwing — defaulted reads keep a malformed/short feature frame safe (the watchdog owns
        loudness on the hot path, ADR-0002)."""
        feats = obs.features
        T = self._t

        # length-T instantaneous gauges; tolerate a short/absent list defensively so act() never throws.
        inflight = _fit(np.asarray(feats.get("inflight", ()), dtype=np.float64), T)
        ready = _fit(np.asarray(feats.get("ready", ()), dtype=np.float64), T)

        # --- the shared Whittle-style index: normalized backlog * normalized headroom (vectorized) ---
        headroom = np.clip((self._d - inflight) / float(self._d), 0.0, 1.0)  # vanishes as j -> D (saturated)
        backlog = ready / float(self._k)                                     # available work, K-normalized
        index = backlog * headroom
        self._last_index = index

        # --- the coordinated allow set: top ceil(p*T) by index (positive only), or index >= theta ---
        allow = np.zeros(T, dtype=bool)
        if self._theta is not None:
            allow = index >= self._theta
        else:
            assert self._p is not None  # one rule is always live (enforced in __init__)
            budget = min(T, max(1, math.ceil(self._p * T)))   # at least one activation slot
            positive = index > 0.0
            n_pos = int(positive.sum())
            if n_pos <= budget:
                # fewer positive-index arms than the budget: activate exactly those (never pad with zeros).
                allow = positive
            else:
                # more candidates than slots: take the `budget` highest indices among the positive arms.
                # argpartition is O(T) (no full sort) — only the cut, not the order, matters for a set.
                top = np.argpartition(index, T - budget)[T - budget:]
                allow[top] = True
                allow &= positive   # a tie at exactly 0 must never sneak in via the partition boundary

        # --- liveness override: inflight==0 is an UNGATED forced flush -> a deny is a no-op, so force allow ---
        allow |= inflight <= 0.0

        self._last_active = int(np.count_nonzero(allow))
        return [1 if a else 0 for a in allow.tolist()]

    def metrics(self) -> Mapping[str, float]:
        """Dashboard scalars: the mean/max of the last forward's per-thread indices and the activation count
        (how many arms the coordinated gate allowed). Empty-safe."""
        idx = self._last_index
        return {
            "index_mean": float(idx.mean()) if idx.size else 0.0,
            "index_max": float(idx.max()) if idx.size else 0.0,
            "n_active": float(self._last_active),
        }


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


# Register additively into the FROZEN adapter.REGISTRY (one entry + one class — P2 seam discipline; the
# harness + dashboard discover methods here). setdefault so a re-import or a name clash never silently
# clobbers an existing registration.
REGISTRY.setdefault("whittle_rmab", WhittleIndexGate)
