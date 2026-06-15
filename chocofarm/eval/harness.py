#!/usr/bin/env python3
"""
Wire the pluggable policies against the decoupled environment and measure each one's
unbiased rate, reported as % of the clairvoyant value-of-information ceiling it claws back.

References (bounds, not deployable policies):
  - realizable static : a fixed value-aware NN route, best expected-rate prefix (the floor).
  - clairvoyant       : free perfect knowledge of the present set (the ceiling / max VoI).
"""
import itertools
import time
import numpy as np
from chocofarm.model.env import Environment
from chocofarm.solvers.base import GreedyPolicy, CertaintyEquivalentPolicy, RolloutPolicy, SparseSamplingPolicy

# The documented exact-decomposition rate (decomp exact, h=1) — the empirical decomp anchor
# reference line. Source: docs/agents/decomp-solver-report.md ("decomp (exact, h=1) 0.0941")
# and docs/results/decomp-rate.md. Hardcoded by maintainer decision (NOT env-derived): unlike the
# floor/ceiling it is a measured policy rate, not a function of the env geometry.
DECOMP_ANCHOR = 0.0941


def realizable_static(env):
    loc, unv, route, t, best = ("w", env.entry), set(range(env.N)), [], 0.0, (-1.0, 0)
    while unv:
        i = max(unv, key=lambda j: env.value[j] / (env.d(loc, ("t", j)) + 1e-9))
        t += env.d(loc, ("t", i)); loc = ("t", i); route.append(i); unv.discard(i)
        rate = (env.K / env.N) * sum(env.value[r] for r in route) / (t + env.exit_cost(loc))
        if rate > best[0]:
            best = (rate, len(route))
    return best[0]


def clairvoyant_rate(env):
    def ev(lam, runs, seed):
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

    def __init__(self, env):
        self.env = env
        self._static_floor = None
        self._clairvoyant_ceiling = None
        self.decomp_anchor = DECOMP_ANCHOR

    @property
    def static_floor(self):
        if self._static_floor is None:
            self._static_floor = realizable_static(self.env)
        return self._static_floor

    @property
    def clairvoyant_ceiling(self):
        if self._clairvoyant_ceiling is None:
            self._clairvoyant_ceiling = clairvoyant_rate(self.env)
        return self._clairvoyant_ceiling

    def voi_pct(self, rate):
        """% of the clairvoyant value-of-information gap a `rate` claws back over the static floor."""
        return (rate - self.static_floor) / (self.clairvoyant_ceiling - self.static_floor) * 100


def main():
    env = Environment()                      # unit values
    refs = BeliefRefs(env)                    # the floor/ceiling/%VoI SSOT (route the metric through it)
    ceil = refs.clairvoyant_ceiling
    print(f"static floor        : {refs.static_floor:.4f}")
    print(f"clairvoyant ceiling : {ceil:.4f}   "
          f"(VoI headroom +{(ceil-refs.static_floor)/refs.static_floor*100:.0f}%)\n",
          flush=True)

    greedy, ce = GreedyPolicy(), CertaintyEquivalentPolicy()
    plan = [
        ("greedy",            greedy,                                  dict(iters=4, warm_runs=600, final_runs=3000)),
        ("certainty-equiv",   ce,                                      dict(iters=4, warm_runs=600, final_runs=3000)),
        ("rollout(greedy)",   RolloutPolicy(greedy, n_samples=10),     dict(iters=2, warm_runs=40, final_runs=150)),
        ("rollout(CE)",       RolloutPolicy(ce, n_samples=10),         dict(iters=2, warm_runs=40, final_runs=150)),
        ("sparse(d2,leaf=CE)", SparseSamplingPolicy(2, 4, ce),         dict(iters=1, warm_runs=15, final_runs=40)),
    ]

    print(f"{'policy':>20} {'rate':>8} {'%ceiling':>9} {'VoI clawed':>11} {'sec':>6}", flush=True)
    for name, pol, budget in plan:
        t0 = time.time()
        r = env.dinkelbach_rate(pol, **budget)["rate"]
        claw = refs.voi_pct(r)
        print(f"{name:>20} {r:>8.4f} {r/ceil*100:>8.0f}% {claw:>10.0f}% {time.time()-t0:>6.0f}",
              flush=True)


if __name__ == "__main__":
    main()
