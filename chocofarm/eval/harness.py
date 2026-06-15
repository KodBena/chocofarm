#!/usr/bin/env python3
"""
Wire the pluggable policies against the decoupled environment and measure each one's
unbiased rate, reported as % of the clairvoyant value-of-information ceiling it claws back.

References (bounds, not deployable policies):
  - realizable static : a fixed value-aware NN route, best expected-rate prefix (the floor).
  - clairvoyant       : free perfect knowledge of the present set (the ceiling / max VoI).
"""
import time
from chocofarm.model.env import Environment
from chocofarm.solvers.base import GreedyPolicy, CertaintyEquivalentPolicy, RolloutPolicy, SparseSamplingPolicy

# The env-derived %VoI reference lines (floor/ceiling/anchor + the BeliefRefs SSOT) now live in the
# NEUTRAL module `chocofarm/references.py` so both `az` (training) and `eval` can depend on them
# without `az` reaching backwards into the eval harness (roadmap item F). Re-exported here verbatim
# for back-compat: existing `from chocofarm.eval.harness import ...` sites keep working unchanged.
from chocofarm.references import BeliefRefs, realizable_static, clairvoyant_rate, DECOMP_ANCHOR


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
