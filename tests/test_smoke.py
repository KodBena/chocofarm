#!/usr/bin/env python3
"""
test_smoke.py — the post-restructure verification gate.

A minimal, bounded smoke test that the package restructure preserved behaviour:
imports resolve, the data files load package-relatively, the model constructs to
its known shape, every solver imports and decides, and the detector-independent
reference lines (static floor / clairvoyant ceiling) are unmoved.

Run pinned + bounded, e.g.:
    taskset -c 3 timeout 120 /home/bork/w/vdc/venvs/generic/bin/python -m pytest tests/test_smoke.py -q

This is NOT a numerical-accuracy battery (the eval_*.py harnesses are); it only
asserts the move did not change the model, the wiring, or the reference lines.
"""
import os
import sys

# The package is importable from the repo root (the maintainer's run convention:
# repo root on sys.path, no pyproject). Under pytest / `PYTHONPATH=.` that is
# already satisfied; when this file is invoked as a bare script `tests/` is on
# the path instead, so put the repo root on it before the package imports resolve.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from chocofarm.model.env import Environment, TERMINATE
from chocofarm.model import arrangement, facemodel
from chocofarm.analysis.analyzer import analyze, real_instance
from chocofarm.solvers.base import (
    Policy, GreedyPolicy, CertaintyEquivalentPolicy,
    RolloutPolicy, SparseSamplingPolicy,
)
from chocofarm.solvers.nmcs import NMCSPolicy
from chocofarm.solvers.ismcts import ISMCTSPolicy
from chocofarm.solvers.decomp import DecompPolicy
from chocofarm.eval.harness import realizable_static, clairvoyant_rate


def test_environment_shape():
    """Environment constructs from the package-relative instance.json + faces.json
    and has the frozen instance's shape: 20 treasures, 44 arrangement faces."""
    env = Environment()
    assert env.N == 20
    assert env.K == 5
    assert len(env.detectors) == 44          # the 44 atomic arrangement faces


def test_data_files_load_package_relative():
    """arrangement.load() / a fresh analyze() resolve the moved data files."""
    faces = arrangement.load()
    assert len(faces) == 44
    senses = facemodel.sense_actions()
    assert len(senses) == 44


def test_analyze_runs():
    """The structural analyzer runs on the real instance and reports its shape."""
    rep = analyze(real_instance())
    assert rep.n_faces == 44
    assert rep.n_treasures == 20
    assert rep.K == 5
    assert rep.n_worlds == 15504             # C(20,5)


def test_greedy_decide():
    """GreedyPolicy decides a legal action (or TERMINATE) from the entry state."""
    env = Environment()
    rng = np.random.default_rng(0)
    a = GreedyPolicy().decide(env, ("w", env.entry), env.worlds, set(), 0.08, rng)
    assert a == TERMINATE or (isinstance(a, tuple) and a[0] in ("t", "d"))


def test_decomp_decide():
    """DecompPolicy decides a legal action (or TERMINATE) from the entry state."""
    env = Environment()
    rng = np.random.default_rng(0)
    a = DecompPolicy(horizon=1).decide(env, ("w", env.entry), env.worlds, set(), 0.08, rng)
    assert a == TERMINATE or (isinstance(a, tuple) and a[0] in ("t", "d"))


def test_search_solvers_construct():
    """The remaining pluggable solvers import and construct (NMCS/ISMCTS/rollout/sparse)."""
    greedy, ce = GreedyPolicy(), CertaintyEquivalentPolicy()
    assert isinstance(NMCSPolicy(level=1), Policy)
    assert isinstance(ISMCTSPolicy(iterations=10), Policy)
    assert isinstance(RolloutPolicy(greedy, n_samples=4), Policy)
    assert isinstance(SparseSamplingPolicy(1, 2, ce), Policy)


def test_reference_lines_unmoved():
    """The detector-independent floor / ceiling are unchanged by the restructure."""
    env = Environment()
    assert abs(realizable_static(env) - 0.0855) < 1e-3
    assert abs(clairvoyant_rate(env) - 0.1454) < 1e-3


if __name__ == "__main__":
    # plain-runnable (no pytest needed) — bounded, prints a one-line PASS per check.
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("all smoke checks passed")
