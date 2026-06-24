#!/usr/bin/env python3
"""
throughput-lab/hp/tests/test_oracles.py — both oracles pass on every shipped Target, and an INJECTED
fault fails loud (DESIGN.md §4.2 / §7 / ADR-0002).

  - Oracle A (orbit non-isomorphism), Oracle B (grid cross-check), and the inertness self-check must
    all pass for TOPOLOGY and OVERCOMMIT and a pinned sub-space.
  - A deliberately-corrupted emitted set (an over-collapse: drop a real config; an under-collapse:
    duplicate an orbit member) must be CAUGHT — proving the oracles are not vacuous.
  - The two backends (CP-SAT enumerate vs the itertools grid) agree on the canonical-rep count.

Run:
    PYTHONPATH=throughput-lab /home/bork/w/vdc/venvs/generic/bin/python -m pytest \
        throughput-lab/hp/tests/test_oracles.py -q

Public Domain (The Unlicense).
"""
from __future__ import annotations

import pytest

from hp import compile as cc
from hp import spec
from hp import verify
from hp.backends import cpsat, grid
from hp.relations import TopologyParams
from hp.spec import Surface


def _space(surface: Surface):
    reg = spec.registry()
    if surface is Surface.TOPOLOGY:
        return cc.compile(reg, cc.Target(surfaces=frozenset({surface}), topo=TopologyParams()))
    return cc.compile(reg, cc.Target(surfaces=frozenset({surface})))


@pytest.mark.parametrize("surface", [Surface.TOPOLOGY, Surface.OVERCOMMIT])
def test_both_oracles_pass(surface):
    cs = _space(surface)
    em = cpsat.enumerate_configs(cs)
    ok, msgs = verify.verify_all(cs, em)
    assert ok, "oracle divergence:\n" + "\n".join(msgs)


@pytest.mark.parametrize("surface", [Surface.TOPOLOGY, Surface.OVERCOMMIT])
def test_cpsat_and_grid_agree_on_count(surface):
    cs = _space(surface)
    cpsat_keys = {cpsat.canonical_key(r, cs) for r in cpsat.enumerate_configs(cs)}
    grid_keys = {cpsat.canonical_key(r, cs) for r in grid.feasible_projections(cs)}
    assert cpsat_keys == grid_keys, (
        f"cpsat={len(cpsat_keys)} grid={len(grid_keys)} differ: "
        f"missing={list(grid_keys - cpsat_keys)[:3]} extra={list(cpsat_keys - grid_keys)[:3]}")


def test_pinned_subspace_oracles_pass():
    reg = spec.registry()
    cs = cc.compile(reg, cc.Target(surfaces=frozenset({Surface.OVERCOMMIT}),
                                   pin={"chunk_floor": True}))
    em = cpsat.enumerate_configs(cs)
    ok, msgs = verify.verify_all(cs, em)
    assert ok, "\n".join(msgs)


def test_injected_overcollapse_is_caught():
    # Drop a real config from the emitted set: Oracle B must report it (the grid still finds it).
    cs = _space(Surface.OVERCOMMIT)
    em = cpsat.enumerate_configs(cs)
    corrupted = em[:-1]   # one real candidate silently dropped
    ok_b, msg_b = verify.oracle_b(corrupted, cs)
    assert not ok_b, "Oracle B failed to catch an over-collapse (dropped config)"
    assert "over-collapse" in msg_b or "grid found" in msg_b


def test_injected_undercollapse_is_caught():
    # Add a SECOND member of an existing orbit (topology has non-trivial orbits): Oracle A must catch
    # it as two emitted configs sharing one orbit.
    reg = spec.registry()
    p = TopologyParams()
    cs = cc.compile(reg, cc.Target(surfaces=frozenset({Surface.TOPOLOGY}), topo=p))
    em = cpsat.enumerate_configs(cs)
    # find a config with a non-singleton orbit and append a distinct orbit-member.
    injected = None
    for rec in em:
        orb = verify._orbit(rec, cs)
        if len(orb) > 1:
            # build a sibling by applying a non-identity isolated-core relabel to the raw projection.
            sibling = _relabel_sibling(rec, cs)
            if sibling is not None and sibling != rec:
                injected = sibling
                break
    assert injected is not None, "no non-singleton orbit found to inject an under-collapse"
    ok_a, msg_a = verify.oracle_a(em + [injected], cs)
    assert not ok_a, "Oracle A failed to catch an under-collapse (orbit-duplicate)"
    assert "under-collapse" in msg_a


def _relabel_sibling(rec: dict, cs) -> dict:
    """Produce a genuinely-distinct same-orbit projection by swapping two isolated core VALUES across
    all core vars (and re-deriving so it is still a valid raw projection)."""
    perm = cs.sym.permutations[0]
    mov = perm.movable_values
    if len(mov) < 2:
        return rec
    a, b = mov[0], mov[1]
    swap = {a: b, b: a}
    sib = dict(rec)
    for v in perm.relabelable_vars:
        if v in sib:
            sib[v] = swap.get(sib[v], sib[v])
    return sib
