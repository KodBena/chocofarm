#!/usr/bin/env python3
"""
throughput-lab/hp/topology_materialize.py — turn a TOPOLOGY enumeration projection into the same
typed, tagged, runnable record (config_id + tag + placements) that harness/topology_enum.py emits.

This is the bridge for the migration acceptance gate (DESIGN.md §6): the SSOT's TOPOLOGY config SET
must equal the standalone topology_enum.py output bit-for-bit (same config_ids). The id/tag/placement
construction here MIRRORS topology_enum._materialize exactly so the two are directly comparable; the
policy vocabularies are sourced from relations.py (the hoisted single home).

Public Domain (The Unlicense).
"""
from __future__ import annotations

from . import relations as rel
from .relations import TopologyParams


def _gen_pol_uniform(rec: dict[str, int]) -> int:
    return rec["gen_pol"]


def materialize(rec: dict[str, int], p: TopologyParams) -> dict:
    """Build the topology_enum-compatible record (config_id, tag, placements) from a projection."""
    hk = p.housekeeping_core
    G = p.n_gens
    placements = []

    s_pol, s_nice, s_lat = rel.SERVER_POLICIES[rec["server_pol"]]
    placements.append({"role": "server", "cpus": [rec["server_core"]],
                       "taskset": str(rec["server_core"]),
                       "policy": s_pol, "nice": s_nice, "latency_nice": s_lat})

    gp = rec["gen_pol"]
    for g in range(G):
        g_pol, g_nice, g_lat = rel.GEN_POLICIES[gp]
        gc = rec[f"gen{g}_core"]
        placements.append({"role": f"gen{g}", "cpus": [gc], "taskset": str(gc),
                           "policy": g_pol, "nice": g_nice, "latency_nice": g_lat})

    surplus_present = bool(rec["surplus_present"])
    if surplus_present:
        u_pol, u_nice, u_lat = rel.SURPLUS_POLICIES[rec["surplus_pol"]]
        uc = rec["surplus_core"]
        placements.append({"role": "surplus", "cpus": [uc], "taskset": str(uc),
                           "policy": u_pol, "nice": u_nice, "latency_nice": u_lat})

    # --- tag (mirrors topology_enum._materialize) -----------------------------------------------
    server_on_hk = rec["server_core"] == hk
    parts = []
    parts.append("server-on-housekeeping" if server_on_hk else "server-isolated")
    if not server_on_hk:
        on_hk = [g for g in range(G) if rec[f"gen{g}_core"] == hk]
        if surplus_present and rec["surplus_core"] == hk:
            parts.append("surplus-on-0")
        elif on_hk:
            parts.append("generator-on-0")
    if surplus_present:
        if rec["surplus_core"] == rec["server_core"]:
            parts.append("surplus-shares-server")
        elif rec["surplus_core"] == hk:
            parts.append("surplus-shares-gen-on-0")
        else:
            parts.append("surplus-shares-gen-isolated")
    else:
        parts.append("no-surplus")
    if rel.SERVER_POLICIES[rec["server_pol"]][0] == "SCHED_OTHER_LATNICE":
        parts.append("latnice")
    else:
        parts.append("plain-server")
    if any(rel.GEN_POLICIES[gp][0] == "SCHED_BATCH" for _ in range(G)):
        parts.append("gen-batch")
    tag = "+".join(parts)

    # --- stable id (mirrors topology_enum._materialize) -----------------------------------------
    sk = f"s{rec['server_core']}p{rec['server_pol']}"
    gk = "g" + "-".join(f"{rec[f'gen{g}_core']}.{gp}" for g in range(G))
    uk = (f"u{rec['surplus_core']}p{rec['surplus_pol']}" if surplus_present else "u_none")
    config_id = f"{sk}_{gk}_{uk}"

    return {"config_id": config_id, "tag": tag, "placements": placements}
