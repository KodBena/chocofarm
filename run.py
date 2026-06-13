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
from env import Environment
from policies import GreedyPolicy, CertaintyEquivalentPolicy, RolloutPolicy, SparseSamplingPolicy


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


def main():
    env = Environment()                      # unit values
    static = realizable_static(env)
    ceil = clairvoyant_rate(env)
    print(f"static floor        : {static:.4f}")
    print(f"clairvoyant ceiling : {ceil:.4f}   (VoI headroom +{(ceil-static)/static*100:.0f}%)\n",
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
        claw = (r - static) / (ceil - static) * 100
        print(f"{name:>20} {r:>8.4f} {r/ceil*100:>8.0f}% {claw:>10.0f}% {time.time()-t0:>6.0f}",
              flush=True)


if __name__ == "__main__":
    main()
