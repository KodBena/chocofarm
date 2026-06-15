#!/usr/bin/env python3
"""
Measure vanilla single-tree UCT's unbiased long-run rate against the VoI ceiling, at budgets
MATCHED to the live ISMCTS sweep (it=200, 400, 800, 1600), so the two are directly comparable.

Mirrors eval_ismcts.py: each policy's rate is its own Dinkelbach fixed point via
`env.dinkelbach_rate`; references are `realizable_static` (floor) and `clairvoyant_rate`
(ceiling). Per budget we report rate, % of the clairvoyant ceiling, % of the VoI gap
(ceiling − static) clawed back, E[R], E[T], and sec/episode.

UCT is the NO-DETERMINIZATION baseline: a single belief-MDP tree whose nodes are
action–observation histories (exact belief via `filter_*`), UCB1 over legal actions + exit,
explicit binary chance nodes on each observation, `GreedyStopBase` rollout, λ-penalised
differential return. See `chocofarm/solvers/uct.py` for the precise variant and how it differs
from SO-ISMCTS (no information-set aggregation, no per-iteration tree determinization).

Budgets are deliberately small (bounded-safety): UCT spends `iterations` belief simulations per
decision and a dozen-ish decisions per episode, and the explicit observation branching makes its
per-episode cost comparable to (often above) ISMCTS, so final-eval N is kept small and every
measurement is meant to be driven under a wall-clock `timeout`. Each row prints its run count so
the Monte-Carlo standard error is explicit. Run subsets to bound wall-time, e.g.:

    python -m chocofarm.eval.eval_uct 200            # just it=200
    python -m chocofarm.eval.eval_uct 200 400        # two budgets
    python -m chocofarm.eval.eval_uct                # the full matched sweep (longest)

An optional trailing `N=<int>` overrides every row's final_runs (handy for shrinking under a
tight timeout): e.g. `python -m chocofarm.eval.eval_uct 1600 N=20`.
"""
import sys

import numpy as np

from chocofarm.model.env import Environment
from chocofarm.solvers.uct import UCTPolicy
from chocofarm.eval.report import references, print_reference_header, run_plan


# (budget label, UCT iterations, Dinkelbach schedule). Matched to the live ISMCTS sweep.
# Schedules shrink as per-decision cost climbs with the budget (calibrated at ~5.1 / 12.1 /
# 19.4 / ~38 s/episode for it=200/400/800/1600 on core 3), so each budget stays inside a single
# ~600 s `timeout` when run solo: total episodes ≈ iters·warm_runs + final_runs. Override the
# final count with N=<int>; run ONE budget per command to keep each measurement bounded.
PLAN = [
    ("uct(it=200)",  200,  dict(iters=2, warm_runs=8, final_runs=40)),
    ("uct(it=400)",  400,  dict(iters=2, warm_runs=4, final_runs=20)),
    ("uct(it=800)",  800,  dict(iters=2, warm_runs=3, final_runs=14)),
    ("uct(it=1600)", 1600, dict(iters=2, warm_runs=1, final_runs=8)),
]


def main():
    args = sys.argv[1:]
    n_override = None
    wanted = set()
    for tok in args:
        if tok.startswith("N="):
            n_override = int(tok[2:])
        else:
            wanted.add(int(tok))

    env = Environment()                                  # unit values, honest face detectors
    refs = references(env)
    print_reference_header(refs)

    plan = []
    for name, iters, budget in PLAN:
        if (not wanted) or (iters in wanted):
            if n_override is not None:
                budget = dict(budget, final_runs=n_override)
            plan.append((name, UCTPolicy(iterations=iters), budget))
    run_plan(env, refs, plan, seed=7, columns="uct")


if __name__ == "__main__":
    main()
