#!/usr/bin/env python3
"""
Measure NMCSPolicy's unbiased long-run rate against the project's reference lines.

Reports, for NMCS level 1 and level 2:
  - rate        : the policy's own Dinkelbach fixed point (env.dinkelbach_rate).
  - % of ceiling: rate / clairvoyant ceiling (the +70% value-of-information bound).
  - VoI clawed  : (rate - static) / (ceiling - static) -- the fraction of the gap between
                  the realizable-static floor and the clairvoyant ceiling that the policy
                  recovers. This is the headline number the project tracks (existing
                  policies claw back at most ~+6%).

Budgets are deliberately small (bounded-safety): level 2's per-episode search cost is
large, so it runs at a tight branching/sample/run budget. The numbers are honest unbiased
rates at those small N -- the Monte-Carlo standard error is non-trivial at level 2, which
the printed run count makes explicit.

Run a single level to bound wall-time:  python eval_nmcs.py 1   /   python eval_nmcs.py 2
No argument runs both (longer).
"""
import sys

from chocofarm.model.env import Environment
from chocofarm.solvers.nmcs import NMCSPolicy
from chocofarm.eval.report import references, print_reference_header, run_plan


# (label, policy, dinkelbach budget) -- tuned so each level finishes within a bounded
# wall-time. Level 2 uses tighter branching/sampling and fewer runs than level 1.
def make_plan(env):
    return {
        1: ("nmcs(level=1)",
            NMCSPolicy(level=1, playout_samples=3, step_samples=2,
                       cand_det=1, cand_tre=4),
            dict(iters=2, warm_runs=30, final_runs=120)),
        2: ("nmcs(level=2)",
            NMCSPolicy(level=2, playout_samples=2, step_samples=1,
                       cand_det=1, cand_tre=3, max_steps=18),
            dict(iters=1, warm_runs=12, final_runs=40)),
    }


def main():
    levels = [int(a) for a in sys.argv[1:]] or [1, 2]

    env = Environment()                      # unit values, the project's reference regime
    refs = references(env)
    print_reference_header(refs)

    plan = make_plan(env)
    run_plan(env, refs, [plan[lvl] for lvl in levels], seed=7, columns="nmcs")


if __name__ == "__main__":
    main()
