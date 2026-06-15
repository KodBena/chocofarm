#!/usr/bin/env python3
"""
Measure SO-ISMCTS's unbiased long-run rate against the value-of-information ceiling.

Reuses the harness references: `realizable_static` (the floor — best fixed value-aware NN
route) and `clairvoyant_rate` (the ceiling — free perfect info, ~+70% over static). Each
policy's rate is its own Dinkelbach fixed point via `env.dinkelbach_rate`. We report, per
iteration budget: the rate, the % of the clairvoyant ceiling reached, and the % of the VoI
gap (ceiling − static) clawed back. Budgets are deliberately small — ISMCTS spends
`iterations` belief playouts per decision and a dozen-ish decisions per episode, so evaluation
run counts are kept modest and every measurement sits under a wall-clock timeout in the driver.
"""
import numpy as np
from chocofarm.model.env import Environment
from chocofarm.solvers.ismcts import ISMCTSPolicy
from chocofarm.eval.report import references, print_reference_header, run_plan


def main():
    env = Environment()                                  # unit values
    refs = references(env)
    print_reference_header(refs)

    # (budget label, ISMCTS iterations, Dinkelbach schedule). Small on purpose; bounded memory.
    # ~2.5 s/episode at it=150 on the dev machine, so these two rows run in a few minutes total.
    # Bump warm_runs/final_runs for tighter confidence intervals when you have the wall-clock.
    plan = [
        ("ismcts(it=150)", 150, dict(iters=2, warm_runs=12, final_runs=40)),
        ("ismcts(it=400)", 400, dict(iters=2, warm_runs=10, final_runs=30)),
    ]

    run_plan(env, refs, [(name, ISMCTSPolicy(iterations=iters), budget)
                         for name, iters, budget in plan],
             seed=7, columns="ismcts")


if __name__ == "__main__":
    main()
