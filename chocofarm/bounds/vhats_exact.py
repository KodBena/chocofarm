#!/usr/bin/env python3
"""
vhats_exact.py — the EXACT belief-MDP V̂ strategy for the dual bound (the definitive
tightening test; chocofarm/bounds/info_relaxation.py; design+proofs
docs/design/dual-bound.md).

Split out of `info_relaxation` BY DEPENDENCY: `ExactBeliefVhat` is the enumeration V̂ —
backward induction over the belief semilattice, tractable only on small sub-instances.
It carries its own numpy dependency for the belief arrays. Kept in its own module so a
caller wanting only the cheap analytic V̂ does not pull in the enumeration path.

Public Domain (The Unlicense).
"""
from __future__ import annotations

import numpy as np


class ExactBeliefVhat:
    """The EXACT optimal value-to-go V*(loc, belief, collected) of the (small) belief-
    MDP at a fixed λ, by backward induction over the belief semilattice. Tractable ONLY
    on small sub-instances (`env.restrict(keep, k_local)`) — it enumerates reachable
    beliefs, which is the full 15,504-world intractability on the real env (do NOT use on
    the full env).

    Its purpose is the DEFINITIVE tightening test (dual-bound.md §2.3 / §6): BSS
    strong duality (Thm 3.4) says V̂ = V* makes the penalty OPTIMAL and the bound TIGHT
    — λ̄ = ρ*_subinstance exactly. So on a restricted sub-instance this V̂ should drive the dual bound
    down to the sub-instance's achievable optimum, well below its clairvoyant value —
    a direct demonstration that the machinery TIGHTENS when handed a good V̂ (the
    decomp / analytic V̂ are merely weaker approximations, not a failure of the
    construction)."""

    def __init__(self):
        self._memo = {}     # (lam, loc, belief, collected) -> V*

    def __call__(self, env, loc, bw, collected, lam):
        return self._solve(env, lam, loc,
                           tuple(int(x) for x in bw), frozenset(collected))

    def _solve(self, env, lam, loc, bw_key, collected):
        key = (round(lam, 9), loc, bw_key, collected)
        if key in self._memo:
            return self._memo[key]
        bw = np.array(bw_key, dtype=np.int64)
        if len(bw) == 0:
            self._memo[key] = -lam * env.exit_cost(loc)
            return self._memo[key]
        best = -lam * env.exit_cost(loc)                   # TERMINATE
        marg = env.marginals(bw)
        # collect a possibly-present uncollected treasure
        for i in range(env.N):
            if i in collected or marg[i] <= 0:
                continue
            dt = env.d(loc, ("t", i))
            q = float(marg[i])
            pres_b = env.filter_treasure(bw, i, True)
            abs_b = env.filter_treasure(bw, i, False)
            vp = env.value[i] + self._solve(env, lam, ("t", i),
                                            tuple(int(x) for x in pres_b),
                                            collected | {i}) if len(pres_b) else 0.0
            va = self._solve(env, lam, ("t", i), tuple(int(x) for x in abs_b),
                             collected) if len(abs_b) else 0.0
            q_val = -lam * dt + q * vp + (1.0 - q) * va
            best = max(best, q_val)
        # read an informative face
        for j in env.detectors:
            cm = env.cover_mask[j]
            hit = (bw & cm) != 0
            if not (hit.any() and (~hit).any()):
                continue
            dt = env.d(loc, ("d", j))
            p = float(hit.mean())
            vpos = self._solve(env, lam, ("d", j),
                               tuple(int(x) for x in bw[hit]), collected)
            vneg = self._solve(env, lam, ("d", j),
                               tuple(int(x) for x in bw[~hit]), collected)
            q_val = -lam * dt + p * vpos + (1.0 - p) * vneg
            best = max(best, q_val)
        self._memo[key] = best
        return best
