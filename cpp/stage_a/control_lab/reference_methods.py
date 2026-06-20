#!/usr/bin/env python3
"""
cpp/stage_a/control_lab/reference_methods.py — the lab's REFERENCE controllers: trivial methods that
exercise the harness + the watchdog, NOT the real candidate families (static/online/supervised/rl —
the later fan-out). They implement the FROZEN adapter.Controller contract.

  * ReadyThresholdGate (static) — a fixed ready-backlog threshold gate: allow a thread iff its ready
    (parked-at-leaf, unsubmitted) slot count is at or below a threshold. A deterministic, cheap,
    well-behaved reference the harness can score against AllAllow.
  * MalfunctioningController (static) — a DELIBERATELY-misbehaving method for the watchdog smoke test:
    it sleeps past the per-decision deadline (and on a chosen tick raises), so the harness proves it
    FLAGS the malfunction, FALLS BACK to all-allow, and the fixture SURVIVES (no hang, no teardown).

These register into adapter.REGISTRY additively (the harness discovers methods there). They are NOT
imported by the production path.

Public Domain (The Unlicense).
"""
from __future__ import annotations

import time
from typing import Any, Mapping, Sequence

from control_lab.adapter import REGISTRY, Family, Observation, TrialContext


class ReadyThresholdGate:
    """A fixed ready-backlog threshold gate (a trivial static reference, NOT a candidate family): deny a
    thread's next discretionary issue while its ready backlog EXCEEDS `threshold` (let the in-flight
    drain), allow otherwise. Holds the gate per-thread; observe() is a no-op (no learning). Deterministic
    and O(T) — a clean baseline the harness scores against AllAllow."""
    family: Family = "static"

    def __init__(self, threshold: int = 2) -> None:
        self.name = f"ready_threshold{threshold}"
        self._threshold = int(threshold)
        self._t = 1

    def reset(self, ctx: TrialContext) -> None:
        self._t = int(ctx.n_threads)

    def observe(self, reward: float, info: Mapping[str, Any]) -> None:
        pass

    def act(self, obs: Observation) -> Sequence[int]:
        ready = obs.features.get("ready", [0] * self._t)
        # allow iff this thread's ready backlog is within the threshold; deny to let it drain otherwise.
        return [1 if (i < len(ready) and ready[i] <= self._threshold) else 0 for i in range(self._t)]

    def metrics(self) -> Mapping[str, float]:
        return {"threshold": float(self._threshold)}


class MalfunctioningController:
    """A DELIBERATELY-malfunctioning controller for the watchdog smoke test (NOT a candidate family). Its
    act() sleeps past the harness's per-decision deadline (so the watchdog flags 'slow' and falls back to
    all-allow), and on the `raise_on` decision it ALSO throws (so the watchdog's exception arm is
    exercised). The harness must keep the fixture alive and move on. `sleep_s` defaults above the lab's
    default 50ms deadline."""
    family: Family = "static"

    def __init__(self, sleep_s: float = 0.150, raise_on: int = 3) -> None:
        self.name = "malfunctioning"
        self._sleep_s = float(sleep_s)
        self._raise_on = int(raise_on)
        self._t = 1
        self._calls = 0

    def reset(self, ctx: TrialContext) -> None:
        self._t = int(ctx.n_threads)
        self._calls = 0

    def observe(self, reward: float, info: Mapping[str, Any]) -> None:
        pass

    def act(self, obs: Observation) -> Sequence[int]:
        self._calls += 1
        time.sleep(self._sleep_s)   # blow the per-decision deadline (the watchdog 'slow' arm)
        if self._calls == self._raise_on:
            raise RuntimeError("malfunctioning controller: deliberate failure (watchdog smoke test)")
        return [1] * self._t

    def metrics(self) -> Mapping[str, float]:
        return {"calls": float(self._calls)}


# Register the reference methods additively (the harness + dashboard discover methods in REGISTRY; a new
# method is one entry + one class — P2 seam discipline). AllAllow is already registered in adapter.py.
REGISTRY.setdefault("ready_threshold2", lambda: ReadyThresholdGate(threshold=2))
REGISTRY.setdefault("malfunctioning", MalfunctioningController)
