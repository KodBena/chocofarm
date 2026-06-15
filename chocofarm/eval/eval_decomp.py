#!/usr/bin/env python3
"""
eval_decomp.py — measure the exact hierarchical-decomposition policy on the honest
env, reported against the static floor and the clairvoyant ceiling (the +70% VoI
headroom), alongside the shallow/search pack (all of which sit below the floor).

Every number is an UNBIASED Monte-Carlo estimate produced by env.dinkelbach_rate /
env.rate against `decomp_solver.DecompPolicy` — the env charges exact travel for
every action, so the rate is honest regardless of any internal table approximation.

Bounded by design.  Pin to CPU core 3 under timeout, e.g.:

    timeout 600 taskset -c 3 /home/bork/w/vdc/venvs/generic/bin/python eval_decomp.py

Flags:
    --runs N        final-eval episodes for the decomp Dinkelbach (default 1500)
    --horizon-sweep also run the macro horizon sweep (1..4)
    --search        also run the shallow/search pack (slow; bounded budgets)

Public Domain (The Unlicense).
"""
import argparse
import time
from typing import Any

from chocofarm.model.env import Environment
from chocofarm.references import BeliefRefs
from chocofarm.eval.report import references, print_reference_header, dink_float
from chocofarm.solvers.decomp import DecompPolicy


def measure_decomp(env: Environment, refs: BeliefRefs, runs: int, horizon: int) -> dict[str, Any]:
    pol = DecompPolicy(horizon=horizon)
    t0 = time.time()
    res = env.dinkelbach_rate(pol, iters=4, warm_runs=400, final_runs=runs, seed=7)
    r = dink_float(res, "rate")
    return {
        "horizon": horizon, "rate": r, "ER": res["ER"], "ET": res["ET"],
        "lambda": res["lambda"], "exits": res["exits"],
        "pct_ceiling": r / refs.clairvoyant_ceiling * 100,
        "pct_voi": refs.voi_pct(r),
        "on_demand_solves": pol.fallbacks,
        "sec": time.time() - t0,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=1500)
    ap.add_argument("--horizon-sweep", action="store_true")
    ap.add_argument("--search", action="store_true")
    args = ap.parse_args()

    env = Environment()
    refs = references(env)
    print_reference_header(
        refs,
        extra_lines=("clairvoyant per-excursion: knows the present set, takes the tight route",))

    # the headline: the exact-decomposition policy (myopic macro)
    d = measure_decomp(env, refs, args.runs, horizon=1)
    print(f"{'policy':>22} {'rate':>8} {'%ceiling':>9} {'%of+70%VoI':>11} "
          f"{'ER':>5} {'ET':>6} {'sec':>5}", flush=True)
    print(f"{'decomp (exact, h=1)':>22} {d['rate']:>8.4f} {d['pct_ceiling']:>8.0f}% "
          f"{d['pct_voi']:>10.0f}% {d['ER']:>5.2f} {d['ET']:>6.1f} {d['sec']:>5.0f}",
          flush=True)
    print(f"   exits={d['exits']}  on-demand micro solves={d['on_demand_solves']}\n",
          flush=True)

    if args.horizon_sweep:
        print("macro horizon sweep (h=1 is exact + simplest; deeper is within MC noise):",
              flush=True)
        for h in (2, 3, 4):
            dh = measure_decomp(env, refs, max(600, args.runs // 2), horizon=h)
            print(f"   h={h}: rate={dh['rate']:.4f} %VoI={dh['pct_voi']:.0f}% "
                  f"ER={dh['ER']:.2f} ET={dh['ET']:.1f}", flush=True)
        print(flush=True)

    if args.search:
        # the shallow/search pack — all below the floor (the project's prior finding)
        from chocofarm.solvers.base import (GreedyPolicy, CertaintyEquivalentPolicy,
                              RolloutPolicy, SparseSamplingPolicy)
        greedy, ce = GreedyPolicy(), CertaintyEquivalentPolicy()
        pack = [
            ("greedy",            greedy,                             dict(iters=4, warm_runs=600, final_runs=3000)),
            ("certainty-equiv",   ce,                                 dict(iters=4, warm_runs=600, final_runs=3000)),
            ("rollout(CE)",       RolloutPolicy(ce, n_samples=10),    dict(iters=2, warm_runs=40, final_runs=150)),
            ("sparse(d2,leaf=CE)", SparseSamplingPolicy(2, 4, ce),    dict(iters=1, warm_runs=15, final_runs=40)),
        ]
        print("shallow / search pack (prior finding: all below the floor):", flush=True)
        for name, pol, budget in pack:
            t0 = time.time()
            r = dink_float(env.dinkelbach_rate(pol, **budget), "rate")
            claw = refs.voi_pct(r)
            print(f"   {name:>20} rate={r:.4f} %VoI={claw:>4.0f}% ({time.time()-t0:.0f}s)",
                  flush=True)


if __name__ == "__main__":
    main()
