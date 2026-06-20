#!/usr/bin/env python3
"""
cpp/stage_a/control_lab/methods/token_bucket.py — a per-thread ROW-METERED token bucket (static family)
candidate for the issue-gate control lab.

Mechanism. Each producer thread t carries a token level tok[t] (a float, capped at C_burst) refilled at a
constant rate rho rows/sec. At each per-forward decision the bucket refills by rho * dt, where dt is the
delta of the HARNESS clock obs.t_monotonic since this controller's last decision (clamped to C_burst).
A thread is ALLOWED iff tok[t] >= 1. When a thread ACTUALLY issues, it spends tokens equal to the chunk
count it offered — meter the ROWS, not the messages (the design's correction): the chunk count is the
leaves[t] increment since the last decision divided by max(1, s_min) (the producer coalescing floor / chunk
size). leaves is a CUMULATIVE counter, so it is first-differenced; and because the feature wire reports
leaves only for the threads SERVED in this forward (an absent thread reads a sentinel 0, not its true
cumulative), the bucket consumes + advances its per-thread leaves baseline ONLY for served threads — an
absent thread offered no rows this forward, so it neither refills-against-rows nor first-differences against
a stale baseline.

This is a STATIC controller: no learning, observe() is a no-op. The gate is DENY-ONLY (the runner's
effective gate is `inflight < D && allow`, and the forced flush at inflight==0 is UNGATED), so a deny is a
NO-OP whenever inflight[t]==0; we make that explicit as a liveness override (inflight[t]==0 -> force allow)
so the bucket never starves a thread that has nothing in flight. The first decision of a trial (no prior
clock) allows every thread (the baseline), matching AllAllow until the bucket has a dt to meter against.

Knobs: rho (refill rate, rows/sec) and C_burst (token cap). s_min and K come from TrialContext. metrics()
reports the mean token level and rho for the dashboard.

Public Domain (The Unlicense).
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from control_lab.adapter import REGISTRY, Family, Observation, TrialContext


class TokenBucketGate:
    """A per-thread row-metered token bucket issue gate (static). Refills tok[t] at rho rows/sec against the
    harness clock, caps at C_burst, allows iff tok[t] >= 1, and SPENDS tokens equal to the rows offered
    (leaves-increment / max(1, s_min)) by the threads served this forward. inflight[t]==0 force-allows
    (a deny is a no-op there). O(T) per decision, non-throwing — it rides the per-forward critical path."""

    family: Family = "static"

    def __init__(self, rho: float = 8.0, c_burst: float = 16.0) -> None:
        self.name = f"token_bucket_r{rho:g}_b{c_burst:g}"
        self._rho = float(rho)
        self._c_burst = float(c_burst)
        # Per-run state (all sized at reset, cleared there too).
        self._t = 1
        self._s_min = 1
        self._k = 1
        self._tok = np.zeros(1, dtype=np.float64)        # token levels, one per thread
        self._leaves_prev = np.zeros(1, dtype=np.int64)  # per-thread cumulative-leaves baseline
        self._seen = np.zeros(1, dtype=bool)             # has this thread been served before (baseline valid)?
        self._last_t: float | None = None               # harness clock at the previous decision

    def reset(self, ctx: TrialContext) -> None:
        """Begin a fresh trial: size the per-thread buckets to T, fill to the burst cap, and clear every
        per-run accumulator (baseline, seen-flags, clock). s_min / K come from the out-of-band context."""
        self._t = int(ctx.n_threads)
        self._s_min = max(1, int(ctx.s_min))      # row->chunk normalizer; never divide by < 1
        self._k = max(1, int(ctx.k_per_thread))   # capacity normalizer (recorded; not in the static gate path)
        self._tok = np.full(self._t, self._c_burst, dtype=np.float64)  # start full -> the baseline is all-allow
        self._leaves_prev = np.zeros(self._t, dtype=np.int64)
        self._seen = np.zeros(self._t, dtype=bool)
        self._last_t = None

    def observe(self, reward: float, info: Mapping[str, Any]) -> None:
        """No-op: a static bucket does not learn from the realized reward."""
        return None

    def act(self, obs: Observation) -> Sequence[int]:
        """Refill against dt, meter rows offered by the served threads, and return the per-thread allow bits.
        Cheap (O(T) numpy) and non-throwing — defaulted reads keep a malformed/short feature frame safe."""
        feats = obs.features
        T = self._t

        # --- pull the length-T feature vectors, defaulting defensively (act must not throw) ---
        inflight = np.asarray(self._vec(feats.get("inflight"), T), dtype=np.int64)
        leaves = np.asarray(self._vec(feats.get("leaves"), T), dtype=np.int64)

        # served threads only carry a TRUE cumulative leaves reading this forward; absent threads read a
        # sentinel 0 (lab_server builds a fresh [0]*T and fills served tids), so meter only the served set.
        served = [i for i in obs.served if 0 <= i < T]

        now = float(obs.t_monotonic)
        if self._last_t is None:
            # First decision of the trial: no dt to meter against -> allow everyone (the AllAllow baseline),
            # but seed the per-thread leaves baseline for the threads we can see so the FIRST real increment
            # next forward is measured from here, not from zero.
            for i in served:
                self._leaves_prev[i] = leaves[i]
                self._seen[i] = True
            self._last_t = now
            return [1] * T

        # --- refill: every bucket gains rho * dt rows of credit, clamped to the burst cap ---
        dt = now - self._last_t
        if dt > 0.0:
            self._tok += self._rho * dt
            np.minimum(self._tok, self._c_burst, out=self._tok)
        self._last_t = now

        # --- consume: spend tokens equal to the ROWS offered (leaves increment / s_min) by served threads ---
        for i in served:
            if self._seen[i]:
                d_leaves = int(leaves[i] - self._leaves_prev[i])
                if d_leaves > 0:                       # an actual issue happened -> it offered rows
                    chunks = d_leaves / float(self._s_min)
                    self._tok[i] -= chunks
            self._leaves_prev[i] = leaves[i]
            self._seen[i] = True

        # --- decide: allow iff the bucket holds a whole token; force-allow where nothing is in flight ---
        allow = self._tok >= 1.0
        allow |= (inflight <= 0)                       # liveness override: a deny is a no-op at inflight==0
        return [1 if a else 0 for a in allow.tolist()]

    def metrics(self) -> Mapping[str, float]:
        """Dashboard scalars: the mean token level across threads and the configured refill rate."""
        return {"mean_token": float(self._tok.mean()) if self._tok.size else 0.0, "rho": self._rho}

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
REGISTRY.setdefault("token_bucket", TokenBucketGate)
