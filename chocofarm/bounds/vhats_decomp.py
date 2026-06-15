#!/usr/bin/env python3
"""
vhats_decomp.py — the decomp-backed V̂ strategy for the dual bound (the strongest
TRUSTED V̂; chocofarm/bounds/info_relaxation.py; design+proofs docs/design/dual-bound.md).

Split out of `info_relaxation` BY DEPENDENCY: `DecompVhat` is the one V̂ that reaches
into chocofarm.solvers.decomp. It used to import decomp via a LAZY in-method import
purely so the bounds module stayed importable without decomp on hand — a self-inflicted
import cycle dodged at call time. With the strategy in its own module, decomp is a
NORMAL module-top dependency (this file is only imported when a caller actually wants
the decomp V̂), and the lazy import is dissolved. The math is unchanged.

Public Domain (The Unlicense).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from chocofarm.solvers import decomp as D

if TYPE_CHECKING:
    from chocofarm.model.env import Collected, Environment, Loc, WorldSet
    from chocofarm.solvers.decomp import Cluster, MacroPlanner, MicroSolution

    # The per-λ cache payload `_build` returns: the macro planner, the sense clusters,
    # and the δ-singleton treasure ids.
    _Built = tuple[MacroPlanner, list[Cluster], list[int]]


class DecompVhat:
    """Decomp belief value function V̂_D (dual-bound.md §2.4(2)): the macro's λ-value
    of the live state, reusing chocofarm.solvers.decomp's exact per-cluster
    continuation values + the live occupancy posterior. This is the SAME object the
    decomp policy acts on (the 0.094-achievable belief value), reused as the penalty's
    value approximation — the strongest TRUSTED V̂ here.

    V̂_D(loc, bw, collected) = MacroPlanner.value(loc, live_posterior, visited∅,
    delta_done(from collected), horizon)[0], i.e. the expectimax λ-value of the
    macro state, which already includes the exit toll. Built lazily per λ and cached.

    Note: this is a DECISION value function (it steers the decomp policy), reused as a
    state-value estimate. It is accurate but sub-optimal, so the resulting bound is
    tight-ish, not exact (dual-bound.md §6)."""

    def __init__(self, horizon: int = 1) -> None:
        self.horizon = horizon
        self._built: dict[float, "_Built"] = {}   # round(lam,6) -> (macro, sense, delta_ids)

    def _build(self, env: "Environment", lam: float) -> "_Built":
        key = round(lam, 6)
        if key in self._built:
            return self._built[key]
        clusters = D.discover_clusters(env)
        sense = [c for c in clusters if c.size > 1]
        anchors = {c.name: min(c.tres, key=lambda t: env.d(("w", env.entry), ("t", t)))
                   for c in sense}
        micro: dict[tuple[str, int], "MicroSolution"] = {}
        for c in sense:
            entry: "Loc" = ("t", anchors[c.name])
            for k in range(1, c.size + 1):
                micro[(c.name, k)] = D.build_cluster_micro(env, c, k, lam, entry)
        macro = D.MacroPlanner(env, clusters, micro, lam, horizon=self.horizon)
        delta_ids = [c.tres[0] for c in clusters if c.size == 1]
        built = (macro, sense, delta_ids)
        self._built[key] = built
        return built

    def __call__(self, env: "Environment", loc: "Loc", bw: "WorldSet",
                 collected: "Collected", lam: float) -> float:
        if len(bw) == 0:
            return -lam * env.exit_cost(loc)
        macro, sense, delta_ids = self._build(env, lam)
        post = D._live_occupancy_posterior(env, bw, macro)
        # visited: clusters already fully collected (all members collected) count as
        # visited so the macro does not re-enter them; conservative — an unvisited but
        # partly-collected cluster is left enterable (the macro re-values it).
        visited: set[int] = set()
        for ci, c in enumerate(sense):
            if all(t in collected for t in c.tres):
                visited.add(ci)
        delta_done = frozenset(t for t in delta_ids if t in collected)
        v, _ = macro.value(loc, post, visited, delta_done, self.horizon)
        # macro.value returns the λ-value of CONTINUING (it includes the exit toll on
        # its 'exit' leaf). Add the already-collected reward? No: V̂ is value-TO-GO,
        # the continuation value from this state, which is exactly what macro.value
        # returns. Reward already banked is not part of value-to-go.
        return v
