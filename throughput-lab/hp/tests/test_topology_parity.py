#!/usr/bin/env python3
"""
throughput-lab/hp/tests/test_topology_parity.py — the migration acceptance gate (DESIGN.md §6).

Asserts that compile(Target(TOPOLOGY)) -> CP-SAT enumerate -> canonicalize -> materialize produces
EXACTLY the standalone harness/topology_enum.py output: same N configs, same config_ids, same tags.
A divergence means the hoist changed the space (an ADR-0012 P1 violation) and fails loud.

Run:
    PYTHONPATH=throughput-lab /home/bork/w/vdc/venvs/generic/bin/python -m pytest \
        throughput-lab/hp/tests/test_topology_parity.py -q

Public Domain (The Unlicense).
"""
from __future__ import annotations

import os
import sys

# topology_enum.py lives under harness/, not on the package path; add it explicitly.
_HARNESS = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                        "harness")
if _HARNESS not in sys.path:
    sys.path.insert(0, _HARNESS)

import topology_enum as te  # noqa: E402

from hp import compile as cc  # noqa: E402
from hp import spec  # noqa: E402
from hp import topology_materialize as tm  # noqa: E402
from hp.backends import cpsat  # noqa: E402
from hp.relations import TopologyParams  # noqa: E402


def _ssot_topology(n_cores: int, n_gens: int, hk: int = 0) -> dict[str, dict]:
    reg = spec.registry()
    p = TopologyParams(n_cores=n_cores, n_gens=n_gens, housekeeping_core=hk)
    cs = cc.compile(reg, cc.Target(surfaces=frozenset({spec.Surface.TOPOLOGY}), topo=p))
    em = cpsat.enumerate_configs(cs)
    out = {}
    for rec in em:
        m = tm.materialize(rec, p)
        out[m["config_id"]] = m
    return out


def _standalone_topology(n_cores: int, n_gens: int, hk: int = 0) -> dict[str, dict]:
    p = te.ModelParams(n_cores=n_cores, n_gens=n_gens, housekeeping_core=hk)
    configs = te.build_and_enumerate(p)
    return {c.config_id: c.to_record() for c in configs}


def test_default_4cores_3gens_count_is_40():
    ssot = _ssot_topology(4, 3)
    assert len(ssot) == 40, f"expected 40 (topology_enum --verify), got {len(ssot)}"


def test_default_config_id_set_matches_bit_for_bit():
    ssot = _ssot_topology(4, 3)
    ref = _standalone_topology(4, 3)
    assert set(ssot) == set(ref), (
        f"config_id set differs: SSOT-only={sorted(set(ssot)-set(ref))[:5]} "
        f"standalone-only={sorted(set(ref)-set(ssot))[:5]}")


def test_default_tags_match():
    ssot = _ssot_topology(4, 3)
    ref = _standalone_topology(4, 3)
    diffs = {cid for cid in ref if ssot[cid]["tag"] != ref[cid]["tag"]}
    assert not diffs, f"tags differ for {sorted(diffs)[:5]}"


def test_default_placements_match():
    ssot = _ssot_topology(4, 3)
    ref = _standalone_topology(4, 3)
    for cid in ref:
        # compare placement records (role/cpus/policy/nice/latency_nice).
        sp = {pl["role"]: pl for pl in ssot[cid]["placements"]}
        rp = {pl["role"]: pl for pl in ref[cid]["placements"]}
        assert sp == rp, f"placements differ for {cid}: {sp} vs {rp}"


def _orbit_partition(records: dict[str, dict], n_cores: int, n_gens: int, hk: int = 0) -> set:
    """Map each config to topology_enum's OWN orbit invariant (_canonical_key over the raw rec), so
    two reductions that pick DIFFERENT representatives of the SAME orbits compare equal. Uses the
    standalone canonical key as the neutral referee."""
    p = te.ModelParams(n_cores=n_cores, n_gens=n_gens, housekeeping_core=hk)
    keys = set()
    for rec in records.values():
        # rebuild the raw dict topology_enum._canonical_key expects from the placement record.
        raw = _record_to_raw(rec, n_gens)
        keys.add(te._canonical_key(raw, p))
    return keys


def _record_to_raw(rec: dict, n_gens: int) -> dict:
    pol_idx = {"SCHED_OTHER_LATNICE": 0, "SCHED_OTHER": 0, "SCHED_BATCH": 1, "SCHED_IDLE": 0}
    by_role = {pl["role"]: pl for pl in rec["placements"]}
    # server_pol: index into SERVER_POLICIES (latnice=0, other=1)
    s_pol = 0 if by_role["server"]["policy"] == "SCHED_OTHER_LATNICE" else 1
    gens = [by_role[f"gen{g}"] for g in range(n_gens)]
    g_pols = [0 if gp["policy"] == "SCHED_OTHER" else 1 for gp in gens]
    raw = {
        "server_core": by_role["server"]["cpus"][0], "server_pol": s_pol,
        "gen_core": [gp["cpus"][0] for gp in gens], "gen_pol": g_pols,
        "surplus_present": "surplus" in by_role,
        "surplus_core": by_role["surplus"]["cpus"][0] if "surplus" in by_role else 0,
        "surplus_pol": (0 if by_role["surplus"]["policy"] == "SCHED_IDLE" else 1)
        if "surplus" in by_role else 0,
    }
    return raw


def test_non_default_substrate_5cores_3gens_same_orbits():
    # a non-default region (slack cores) — proves the lowering is parametric, not hand-tuned to 4/3.
    # The two reductions may pick different orbit REPRESENTATIVES at n>default; what must match is the
    # ORBIT PARTITION (same count, same orbits under the neutral referee key).
    ssot = _ssot_topology(5, 3)
    ref = _standalone_topology(5, 3)
    assert len(ssot) == len(ref), f"5/3 config COUNT differs: ssot={len(ssot)} ref={len(ref)}"
    assert _orbit_partition(ssot, 5, 3) == _orbit_partition(ref, 5, 3), (
        "5/3 ORBIT PARTITION differs (a real symmetry-reduction divergence)")
