#!/usr/bin/env python3
"""
chocofarm/references.py — the env-derived %VoI reference lines (floor / ceiling / anchor).

The NEUTRAL home for the three reference quantities the project plots %VoI against:
a realizable static floor (`realizable_static`), a clairvoyant rate ceiling
(`clairvoyant_rate`), the documented decomposition anchor (`DECOMP_ANCHOR`), and the
`BeliefRefs` SSOT that bundles them. These are functions of the env geometry (numpy +
the passed `Environment`) — nothing more — so they are a FOUNDATION both training (`az`)
and evaluation (`eval`) can depend on, not a piece of the eval harness.

Moved here (verbatim) from `chocofarm/eval/harness.py` to cut the backwards
`az → eval` import edge: `eval` is a consumer of training, not a foundation for it, so
`az/exit_loop.py` must not reach into `eval` for these env-derived numbers. `eval.harness`
re-exports these names for back-compat. This module imports ONLY numpy + itertools — it
depends on neither `chocofarm.eval.*` nor `chocofarm.az.*` (the cycle it exists to break).

Public Domain (Unlicense).
"""
from __future__ import annotations

import itertools
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from chocofarm.model.env import Environment, Loc

# The documented exact-decomposition rate (decomp exact, h=1) — the empirical decomp anchor
# reference line. Source: docs/agents/decomp-solver-report.md ("decomp (exact, h=1) 0.0941")
# and docs/results/decomp-rate.md. Hardcoded by maintainer decision (NOT env-derived): unlike the
# floor/ceiling it is a measured policy rate, not a function of the env geometry.
DECOMP_ANCHOR = 0.0941


def realizable_static(env: Environment) -> float:
    loc: Loc = ("w", env.entry)
    unv = set(range(env.N))
    route: list[int] = []
    t = 0.0
    best: tuple[float, int] = (-1.0, 0)
    while unv:
        i = max(unv, key=lambda j: env.value[j] / (env.d(loc, ("t", j)) + 1e-9))
        t += env.d(loc, ("t", i)); loc = ("t", i); route.append(i); unv.discard(i)
        rate = (env.K / env.N) * sum(env.value[r] for r in route) / (t + env.exit_cost(loc))
        if rate > best[0]:
            best = (rate, len(route))
    return best[0]


def clairvoyant_rate(env: Environment) -> float:
    def ev(lam: float, runs: int, seed: int) -> float:
        rng = np.random.default_rng(seed)
        totR = totT = 0.0
        for _ in range(runs):
            w = int(rng.choice(env.worlds))
            present = [t for t in range(env.N) if (w >> t) & 1]
            base = env.exit_cost(("w", env.entry))
            bv, bR, bT = -lam * base, 0.0, base
            for s in range(1, len(present) + 1):
                for sub in itertools.combinations(present, s):
                    R = sum(env.value[i] for i in sub)
                    bt = min(env.route_time(("w", env.entry), list(p))
                             for p in itertools.permutations(sub))
                    v = R - lam * bt
                    if v > bv:
                        bv, bR, bT = v, R, bt
            totR += bR; totT += bT
        return totR / totT
    lam = 0.0
    for _ in range(5):
        lam = ev(lam, 1000, 1)
    return ev(lam, 3000, 7)


class BeliefRefs:
    """Single source for the three %VoI reference lines and the %VoI map itself.

    These are the Tier-4 DERIVED reference lines the project plots %VoI against:
      - `static_floor`        = realizable_static(env)  — DERIVED from the env (the floor).
      - `clairvoyant_ceiling` = clairvoyant_rate(env)   — DERIVED from the env (the ceiling).
      - `decomp_anchor`       = DECOMP_ANCHOR            — the ONE documented constant (anchor),
                                                          the exact-decomposition rate (not env-derived).

    The floor and ceiling are a few seconds each to compute, so they are computed LAZILY on first
    access and MEMOIZED (never recomputed per call). This is the single source for %VoI: route every
    display reference-line site and every (rate → %VoI) conversion through here so they cannot drift.
    """

    def __init__(self, env: Environment):
        self.env = env
        self._static_floor: float | None = None
        self._clairvoyant_ceiling: float | None = None
        self.decomp_anchor = DECOMP_ANCHOR

    @property
    def static_floor(self) -> float:
        if self._static_floor is None:
            self._static_floor = realizable_static(self.env)
        return self._static_floor

    @property
    def clairvoyant_ceiling(self) -> float:
        if self._clairvoyant_ceiling is None:
            self._clairvoyant_ceiling = clairvoyant_rate(self.env)
        return self._clairvoyant_ceiling

    def voi_pct(self, rate: float) -> float:
        """% of the clairvoyant value-of-information gap a `rate` claws back over the static floor."""
        return (rate - self.static_floor) / (self.clairvoyant_ceiling - self.static_floor) * 100
