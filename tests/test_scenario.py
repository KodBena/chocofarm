#!/usr/bin/env python3
"""
test_scenario.py — the R7 verification gate for `Scenario` + copy-on-write.

`Environment.with_scenario(s)` must be EXACTLY equivalent to a fresh
`Environment(value=…, entry=…, teleport_overhead=…)` (VERIFY-1), must SHARE the
expensive Tier-1 geometry by reference rather than rebuild it and must not mutate
the base env (VERIFY-2), and must FAIL LOUD on a wrong-length value vector
(VERIFY-3, ADR-0002). This is an ADDITIVE step: no existing behaviour, call site,
or test is touched.

Run pinned + bounded, e.g.:
    taskset -c 3 timeout 120 /home/bork/w/vdc/venvs/generic/bin/python -m pytest tests/test_scenario.py -q

Public Domain (The Unlicense).
"""
import os
import sys

# Repo root on sys.path (the maintainer's run convention; mirrors test_smoke.py)
# so the package imports resolve both under pytest/`PYTHONPATH=.` and as a bare
# script.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from chocofarm.model.env import Environment
from chocofarm.model.instance import Scenario
from chocofarm.solvers.base import GreedyPolicy


def _scenarios(N):
    """The VERIFY-1 scenario battery: defaults, a heterogeneous value vector, a
    non-default entry, and a non-default teleport overhead."""
    return [
        Scenario(),
        Scenario(value=[10.0 if i in {3, 4, 16, 19} else 1.0 for i in range(N)]),
        Scenario(entry="CSCE"),
        Scenario(teleport_overhead=20.0),
    ]


def test_with_scenario_equivalence():
    """VERIFY-1: `base.with_scenario(s)` matches a freshly-constructed reference
    `Environment(value=s.value, entry=s.entry, teleport_overhead=s.teleport_overhead)`
    on the scenario knobs, the shape, the world set, a fixed-seed simulate triple,
    a marginals call, and an exit_cost (which must reflect the new `tp`)."""
    base = Environment()
    for s in _scenarios(base.N):
        e = base.with_scenario(s)
        ref = Environment(value=s.value, entry=s.entry,
                          teleport_overhead=s.teleport_overhead)

        assert e.value == ref.value
        assert e.entry == ref.entry
        assert e.tp == ref.tp
        assert e.N == ref.N
        assert e.K == ref.K
        assert np.array_equal(e.worlds, ref.worlds)

        # A fixed-seed episode triple must match exactly (same policy, world, lam,
        # seed) — the reward read (value), the entry (entry) and the exit (tp) all
        # flow through here.
        lam = 0.08
        world = int(np.random.default_rng(123).choice(base.worlds))
        e_out = e.simulate(GreedyPolicy(), world, lam, np.random.default_rng(7))
        ref_out = ref.simulate(GreedyPolicy(), world, lam, np.random.default_rng(7))
        assert e_out == ref_out

        # A marginals call and an exit_cost (tp-dependent) match.
        assert np.array_equal(e.marginals(e.worlds), ref.marginals(ref.worlds))
        loc = ("t", 0)
        assert e.exit_cost(loc) == ref.exit_cost(loc)


def test_with_scenario_shares_geometry_and_no_mutation():
    """VERIFY-2: the Tier-1 geometry (`_dist`, `worlds`, `coord`) is SHARED by
    reference (copy-on-write, not rebuilt), and `base` is unmutated by the call."""
    base = Environment()
    base_value_before = list(base.value)

    e = base.with_scenario(
        Scenario(value=[10.0 if i in {3, 4, 16, 19} else 1.0 for i in range(base.N)]))

    # Shared by reference — the expensive distance table is NOT rebuilt.
    assert e._dist is base._dist
    assert e.worlds is base.worlds
    assert e.coord is base.coord

    # The override actually took effect (so the sharing is not masking a no-op).
    assert e.value != base.value
    assert e.value[3] == 10.0

    # `base` is untouched: still its original unit-value vector.
    assert base.value == base_value_before
    assert base.value == [1.0] * base.N


def test_with_scenario_wrong_length_value_fails_loud():
    """VERIFY-3 (ADR-0002): a wrong-length value vector is a config error and must
    raise loudly — not be silently broadcast or truncated."""
    base = Environment()
    with pytest.raises(ValueError) as ei:
        base.with_scenario(Scenario(value=[1.0] * 3))
    msg = str(ei.value)
    assert "length 3" in msg
    assert f"N={base.N}" in msg


if __name__ == "__main__":
    # plain-runnable (no pytest needed) — bounded, prints a one-line PASS per check.
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("all scenario checks passed")
