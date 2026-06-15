#!/usr/bin/env python3
"""
eval_faces.py — re-measure the Monte-Carlo solvers on the HONEST detector model.

The environment now keys its sense actions to the planar arrangement FACES
(arrangement.py + chocobo_faces.json), not the old `cover_mask[i] = {i} ∪
overlap-neighbours` over-approximation (docs/consults/consult-002-detector-misspec-report.md
§(4)). Standing in face F
reveals the disjunction over exactly F.cover — cover and position consistent by
construction. The old model handed out information no real sensor could (70% of
sensing area is singleton-cover; the lone k=4 face is a 0.05-area sliver), so the
honest detectors are WEAKER and the measured rates are expected to drop.

This script reproduces run.py's comparison on the honest env: the static floor and
clairvoyant ceiling (both detector-independent — they should be unmoved at 0.0855 /
0.1454), then every pluggable policy, reporting each one's Dinkelbach rate, % of the
ceiling, and % of the VoI gap (ceiling − static) clawed back.

Budgets are deliberately small (bounded-safety): the action set widened from 16
detectors to 44 faces, so per-episode cost rose and final-eval N is kept ≤150 (much
less for the deep searches). Every measurement carries its run count so the
Monte-Carlo standard error is explicit. Run subsets to bound wall-time:

    python eval_faces.py refs                  # floor + ceiling only
    python eval_faces.py shallow               # greedy / CE / rollout / sparse
    python eval_faces.py nmcs                   # NMCS L1, L2
    python eval_faces.py ismcts                 # ISMCTS it=200, it=400
    python eval_faces.py                        # all (longest)
"""
import sys

import numpy as np

from chocofarm.model.env import Environment
from chocofarm.solvers.base import (GreedyPolicy, CertaintyEquivalentPolicy, RolloutPolicy,
                      SparseSamplingPolicy)
from chocofarm.solvers.nmcs import NMCSPolicy
from chocofarm.solvers.ismcts import ISMCTSPolicy
from chocofarm.eval.report import references, print_reference_header, run_plan


def build_plan(env):
    """(group, label, policy, dinkelbach budget). Budgets shrink with per-episode cost;
    the wider face action set makes every search slower than the 16-detector runs."""
    greedy, ce = GreedyPolicy(), CertaintyEquivalentPolicy()
    return [
        ("shallow", "greedy",             greedy,
            dict(iters=4, warm_runs=600, final_runs=3000)),
        ("shallow", "certainty-equiv",    ce,
            dict(iters=4, warm_runs=600, final_runs=3000)),
        ("shallow", "rollout(greedy)",    RolloutPolicy(greedy, n_samples=10),
            dict(iters=2, warm_runs=40, final_runs=150)),
        ("shallow", "sparse(d2,leaf=CE)", SparseSamplingPolicy(2, 4, ce),
            dict(iters=1, warm_runs=12, final_runs=30)),
        ("nmcs",    "nmcs(level=1)",
            NMCSPolicy(level=1, playout_samples=3, step_samples=2,
                       cand_det=1, cand_tre=4),
            dict(iters=2, warm_runs=30, final_runs=120)),
        ("nmcs",    "nmcs(level=2)",
            NMCSPolicy(level=2, playout_samples=2, step_samples=1,
                       cand_det=1, cand_tre=3, max_steps=18),
            dict(iters=1, warm_runs=12, final_runs=40)),
        ("ismcts",  "ismcts(it=200)",     ISMCTSPolicy(iterations=200),
            dict(iters=2, warm_runs=12, final_runs=40)),
        # it=400 at ~9 s/episode on the 64-face root: run this group's two rows
        # back-to-back blows a ~590 s bound (it=200 alone is ~358 s), so it=400 sits
        # at final_runs=20 and is best run solo. See honest-rates-faces.md caveats.
        ("ismcts",  "ismcts(it=400)",     ISMCTSPolicy(iterations=400),
            dict(iters=2, warm_runs=6, final_runs=20)),
    ]


def main():
    groups = set(sys.argv[1:])
    want_refs = (not groups) or ("refs" in groups)
    env = Environment()                      # unit values, honest face detectors

    refs = references(env)
    print_reference_header(refs, faces=True)
    if groups == {"refs"}:
        return

    plan = [row for row in build_plan(env)
            if (not groups) or (row[0] in groups)]
    run_plan(env, refs, [(name, pol, budget) for _group, name, pol, budget in plan],
             seed=7, columns="faces")


if __name__ == "__main__":
    main()
