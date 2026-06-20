#!/usr/bin/env python3
"""
cpp/stage_a/control_lab/adapter.py — the FROZEN controller contract for the issue-gate control lab.

Every candidate method implements `Controller`; supervised methods add a `TrainableRecipe` whose
offline `fit()` returns a runtime `Controller`. The lab harness drives a `Controller` SYNCHRONOUSLY
on the eval server's forward boundary: after each batch is evaluated it calls `observe(reward, info)`
(the outcome of the previous `act`) then `act(obs)` (the new per-thread gate), which rides back on
the reply wire. `Decimate` wraps a heavy controller to decide only every k-th forward (holding its
last gates between decisions) so a slow policy stays off the per-forward critical path WITHOUT
baking a stride counter into the model's own state.

This file is the ONE authoritative interface the parallel method-implementation workflow targets:
treat it as FROZEN. Additive changes only, surfaced explicitly (ADR-0012 P8 typed-contract / P2 seam).

Public Domain (The Unlicense).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping, Protocol, Sequence, runtime_checkable

Family = Literal["static", "online", "supervised", "rl"]


@dataclass(frozen=True)
class TrialContext:
    """Out-of-band geometry + run config a controller needs at reset (the feature wire omits these)."""
    n_threads: int            # T producer threads
    d_ceiling: int            # D, the per-thread in-flight-message cap (live only under chunk_floor)
    k_per_thread: int         # K = trees_per_thread * ceil(pool_batch / n_threads) — the capacity normalizer
    s_min: int                # the producer coalescing floor (chunk size) when chunk_floor is on
    chunk_floor: bool         # whether depth>1 / D-live is reachable this trial
    seed: int
    cadence: Literal["per_forward"] = "per_forward"   # the decision epoch (reserved; per-forward for now)


@dataclass(frozen=True)
class Observation:
    """One decision epoch (one forward): the decoded feature surface + this forward's context."""
    features: Mapping[str, Any]   # n_threads, d_ceiling, server_rows_per_forward + length-T inflight/ready/msgs/leaves/rtt_us
    served: Sequence[int]         # thread ids whose messages were in the just-evaluated forward
    forward_rows: int             # real rows in the just-evaluated forward (the coalescing achieved)
    t_monotonic: float            # harness clock at this epoch (so controllers need not keep their own)


class Dataset(Protocol):
    """Opaque handle to the collected trajectories (exploratory + counterfactual). Declared here so
    recipes can type `fit`; the concrete shape lands with the data-collection batch."""
    ...


@runtime_checkable
class Controller(Protocol):
    """Maps the feature stream to the per-thread binary issue gate, decided once per forward."""
    name: str
    family: Family

    def reset(self, ctx: TrialContext) -> None:
        """Begin a fresh trial; clear all per-run state."""
        ...

    def observe(self, reward: float, info: Mapping[str, Any]) -> None:
        """Outcome of the PREVIOUS act — one (s, a, r) transition. No-op for static/supervised."""
        ...

    def act(self, obs: Observation) -> Sequence[int]:
        """Return the per-thread allow bits, a length-T sequence of {0,1}. Cheap, non-throwing."""
        ...

    def metrics(self) -> Mapping[str, float]:
        """Scalar introspection for the dashboard (learned threshold, arm values, loss, ...); may be empty."""
        ...


@runtime_checkable
class TrainableRecipe(Protocol):
    """Supervised method: offline `fit` over collected data -> a runtime Controller. Backend is the
    method's own choice (sklearn / lightgbm / torch / jax); only the returned Controller is on the hot path."""
    name: str

    def fit(self, data: Dataset) -> Controller:
        ...


class Decimate:
    """Meta-controller: run `inner` only every k-th forward, holding its last gates between decisions and
    delivering the reward ACCUMULATED across the held window to inner.observe at the next decision (so the
    inner controller still sees one clean (s, a, r) per its own epoch). Keeps a heavy policy off the
    per-forward critical path without a stride counter inside the model's state."""

    def __init__(self, inner: Controller, k: int) -> None:
        if k < 1:
            raise ValueError(f"Decimate: k must be >= 1, got {k}")   # fail loud (ADR-0002)
        self.inner = inner
        self.name = f"decimate{k}:{inner.name}"
        self.family: Family = inner.family
        self._k = k
        self._i = 0
        self._last: list[int] | None = None
        self._reward_acc = 0.0

    def reset(self, ctx: TrialContext) -> None:
        self.inner.reset(ctx)
        self._i = 0
        self._last = None
        self._reward_acc = 0.0

    def observe(self, reward: float, info: Mapping[str, Any]) -> None:
        self._reward_acc += reward   # accumulate over the held window; delivered at the next decision

    def act(self, obs: Observation) -> Sequence[int]:
        if self._last is None or self._i % self._k == 0:
            if self._last is not None:
                self.inner.observe(self._reward_acc, {})   # one transition per inner epoch
                self._reward_acc = 0.0
            self._last = list(self.inner.act(obs))
        self._i += 1
        return self._last

    def metrics(self) -> Mapping[str, float]:
        return self.inner.metrics()


class AllAllow:
    """The reference baseline: allow every thread every forward — byte-identical to the fixed-D runner,
    the A/B control arm. The seam where a real policy plugs in."""
    name = "all_allow"
    family: Family = "static"

    def __init__(self) -> None:
        self._t = 1

    def reset(self, ctx: TrialContext) -> None:
        self._t = ctx.n_threads

    def observe(self, reward: float, info: Mapping[str, Any]) -> None:
        pass

    def act(self, obs: Observation) -> Sequence[int]:
        return [1] * self._t

    def metrics(self) -> Mapping[str, float]:
        return {}


# name -> zero-arg factory returning a Controller OR a TrainableRecipe. The harness + dashboard discover
# methods here; a new method is one registry entry + one class, no edits elsewhere (P2 seam discipline).
Factory = Any  # Callable[[], Controller | TrainableRecipe]
REGISTRY: dict[str, Factory] = {"all_allow": AllAllow}
