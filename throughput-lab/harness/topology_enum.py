#!/usr/bin/env python3
"""
throughput-lab/harness/topology_enum.py — the standalone CLI / runnable tool for the throughput
lab's process-topology config space. It CONSUMES the SSOT (ADR-0012 P1: one home); it no longer
re-authors the space.

WHAT THIS IS (and what changed)
-------------------------------
Historically this file was BOTH the home of the topology config space (its own policy vocabularies +
its own OR-Tools CP-SAT model + its own canonical-key orbit dedup) AND the runnable tool. That made
the space a TWO-home fact (the `hp/` package hand-mirrored these tables) — the ADR-0012 P1 defect
this refactor closes. The single home is now `throughput-lab/hp/`:
  - the (policy, nice, latency_nice) vocabularies live in `hp/relations.py` (SERVER_POLICIES /
    GEN_POLICIES / SURPLUS_POLICIES);
  - the R1-R4 placement model + the joint-orbit symmetry live in `hp/compile.py` (the TOPOLOGY
    lowering) + `hp/backends/cpsat.py` (the enumerator + canonicalizer);
  - the typed runnable record (config_id / tag / placements) is built by `hp/topology_materialize.py`.
This file is now a thin CONSUMER: it selects the TOPOLOGY surface, enumerates via the `hp/` compiler,
and renders the SAME `configs.json` + table it always did. Its CLI surface (`--verify`, `--json`,
`--gens/--cores/--housekeeping-core`) is preserved so existing callers (e.g. topology_sweep.py, which
reads the JSON) keep working bit-for-bit.

THE CLASSES PLACED (the lab's processes)
----------------------------------------
  - SERVER   : 1 INFERENCE consumer. Bursty + mildly latency-sensitive (matmul forwards interleaved
               with socket I/O; ~0.58 core busy). Wants prompt scheduling of its bursts.
  - GEN_k    : G GENERATOR producers (default 3). CPU-bound tree-search, always runnable. The work.
  - SURPLUS  : 0 or 1 extra generator — the low-priority slack-filler that soaks fragmented idle on
               a core it SHARES. Present iff it is the only class permitted to co-locate.

THE PLACEMENT SUBSTRATE (a HYPOTHESIS, not a hard truth)
--------------------------------------------------------
Host boots `rcu_nocbs=1-3 isolcpus=managed_irq,nohz,1-3 irqaffinity=0`:
  - core 0  : housekeeping — IRQs affined here, RCU callbacks offloaded ONTO it, general system tasks
              → contended.
  - cores 1-3: isolated — no managed IRQs, nohz_full, RCU-offloaded → clean compute.
CAVEAT (do NOT hard-code as truth): the lab runs in a guest that may not fully see host isolation, and
a prior in-guest A/B refuted a core-3 benefit. So core 0's contention is treated as a HYPOTHESIS the
enumeration must let us TEST — the config set deliberately SPANS "server on housekeeping core 0" and
"server on an isolated core with a generator/surplus on 0", and the co-locations between. We do not
prune by the isolation assumption; we span it.

THE SCHED-POLICY VOCABULARY (per kernel consult; EEVDF, 6.19) — now sourced from hp/relations.py
------------------------------------------------------------------------------------------------
Per-task knobs, carried as a typed attribute of each placement (we model WHICH policy each class may
take; we do NOT model the kernel). The authoritative (policy, nice, latency_nice) triples live in
`hp/relations.py`; `SchedPolicy` below is the enum the typed record renders them as:
  - OTHER(nice)    : SCHED_OTHER at a given nice. The generators' default (nice 0).
  - BATCH          : SCHED_BATCH — throughput-friendly, no wakeup preemption. A generator alt.
  - IDLE           : SCHED_IDLE — runs only in true idle, yields instantly. The textbook surplus fit.
  - OTHER_LATNICE  : SCHED_OTHER with EEVDF latency-nice (-20) — the latency-sensitive server fit.

OUTPUT CONTRACT (the SSOT the harness consumes)
-----------------------------------------------
`python topology_enum.py` writes a JSON array (default ./configs.json) AND prints a human table. Each
record is a typed self-describing `TopologyConfig` (ADR-0012: typed signature is the contract) — a
stable id/name, a one-line tag, and a per-class `Placement` = (cpus: list[int] for `taskset -c`,
policy, nice, latency_nice). Directly translatable to "how to launch this run".

Public Domain (The Unlicense).
"""
from __future__ import annotations

import argparse
import dataclasses
import itertools
import json
import sys
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from hp import compile as cc
from hp import spec
from hp import topology_materialize as tm
from hp.backends import cpsat
from hp.relations import TopologyParams


# ----------------------------------------------------------------------------------------------------
# Typed config records (ADR-0012: the typed signature IS the contract the harness reads).
# These are the rendering types; the (policy, nice, latency_nice) VOCABULARY is sourced from
# hp/relations.py (the single home), not authored here.
# ----------------------------------------------------------------------------------------------------
class SchedPolicy(str, Enum):
    """The per-task scheduling-policy choices. Values match hp/relations.py's policy strings so a
    materialized record's policy string round-trips to this enum (SchedPolicy(value))."""
    OTHER = "SCHED_OTHER"          # plain CFS/EEVDF, weighted by nice
    BATCH = "SCHED_BATCH"          # throughput-friendly, suppresses wakeup preemption
    IDLE = "SCHED_IDLE"            # runs only in true idle, yields instantly (slack-filler)
    OTHER_LATNICE = "SCHED_OTHER_LATNICE"  # SCHED_OTHER + EEVDF latency-nice (low-latency wakeups)


@dataclass(frozen=True)
class Placement:
    """One class pinned to a cpu-list with a scheduling policy. `cpus` feeds `taskset -c`."""
    role: str                 # "server" | "gen" | "surplus"
    cpus: tuple[int, ...]     # taskset -c cpu list
    policy: SchedPolicy
    nice: Optional[int] = None          # for OTHER / BATCH
    latency_nice: Optional[int] = None  # for OTHER_LATNICE

    def to_record(self) -> dict:
        return {
            "role": self.role,
            "cpus": list(self.cpus),
            "taskset": ",".join(str(c) for c in self.cpus),
            "policy": self.policy.value,
            "nice": self.nice,
            "latency_nice": self.latency_nice,
        }


@dataclass(frozen=True)
class TopologyConfig:
    """A full enumerated candidate: every class placed. The unit the sweep harness A/B-tests."""
    config_id: str
    tag: str
    placements: tuple[Placement, ...]

    def to_record(self) -> dict:
        return {
            "config_id": self.config_id,
            "tag": self.tag,
            "placements": [p.to_record() for p in self.placements],
        }


# ----------------------------------------------------------------------------------------------------
# ModelParams — kept as the standalone tool's parameter record. It is a thin alias-by-value of the
# SSOT's hp.relations.TopologyParams (the substrate axes are one fact, defined there); we re-declare
# it here only so the CLI's argparse + JSON `params` block keep their historical shape, and convert
# to the SSOT type at the boundary.
# ----------------------------------------------------------------------------------------------------
@dataclass
class ModelParams:
    n_cores: int = 4
    n_gens: int = 3
    housekeeping_core: int = 0   # the contended IRQ/RCU core; cores 1..n-1 are the isolated set

    def to_ssot(self) -> TopologyParams:
        return TopologyParams(n_cores=self.n_cores, n_gens=self.n_gens,
                              housekeeping_core=self.housekeeping_core)


def _config_from_materialized(rec: dict) -> TopologyConfig:
    """Rebuild the typed TopologyConfig from hp.topology_materialize's dict (the single home of the
    id/tag/placement construction). The policy string round-trips to the SchedPolicy enum."""
    placements = tuple(
        Placement(
            role=pl["role"],
            cpus=tuple(pl["cpus"]),
            policy=SchedPolicy(pl["policy"]),
            nice=pl["nice"],
            latency_nice=pl["latency_nice"],
        )
        for pl in rec["placements"]
    )
    return TopologyConfig(config_id=rec["config_id"], tag=rec["tag"], placements=placements)


def build_and_enumerate(p: ModelParams) -> list[TopologyConfig]:
    """Enumerate the feasible, symmetry-reduced topology config set by CONSUMING the hp/ compiler.

    This is the consumer seam (ADR-0012 P1): the TOPOLOGY surface is selected from the SSOT registry,
    lowered to the backend-neutral ConfigSpace (hp.compile), enumerated + orbit-canonicalized by the
    CP-SAT backend (hp.backends.cpsat), and materialized into the typed record (hp.topology_materialize)
    — the SAME path the hp CLI uses. No CP-SAT model or canonical key is authored here anymore.
    """
    sp = p.to_ssot()
    cs = cc.compile(spec.registry(),
                    cc.Target(surfaces=frozenset({spec.Surface.TOPOLOGY}), topo=sp))
    emitted = cpsat.enumerate_configs(cs)
    configs = [_config_from_materialized(tm.materialize(rec, sp)) for rec in emitted]
    # Preserve this tool's historical emission ORDER (sorted by _canonical_key over the raw rec) so the
    # JSON is bit-for-bit identical to the pre-consumer tool, not merely set-equal: downstream readers
    # (topology_sweep.py) index configs by position. The hp enumerator's canonical-key order is a
    # different (also valid) representative ordering; we re-sort to the tool's original convention.
    return sorted(configs, key=lambda c: _canonical_key(_raw_from_config(c, p), p))


# A policy-index lookup over the single home's tables: a (policy, nice, latency_nice) triple -> its
# index. Built once from hp.relations (the home), so a config's enum policy round-trips to the index
# the config_id / canonical key encode.
def _pol_index_tables() -> tuple[dict, dict, dict]:
    from hp import relations as rel
    return (
        {tuple(t): i for i, t in enumerate(rel.SERVER_POLICIES)},
        {tuple(t): i for i, t in enumerate(rel.GEN_POLICIES)},
        {tuple(t): i for i, t in enumerate(rel.SURPLUS_POLICIES)},
    )


def _raw_from_config(cfg: TopologyConfig, p: ModelParams) -> dict:
    """Reconstruct the raw solver-style rec (cores + policy INDICES) from a typed TopologyConfig, for
    the neutral-referee _canonical_key. Policy indices come from the single home's tables (hp.relations)."""
    s_tbl, g_tbl, u_tbl = _pol_index_tables()
    by_role = {pl.role: pl for pl in cfg.placements}

    def triple(pl: Placement) -> tuple:
        return (pl.policy.value, pl.nice, pl.latency_nice)

    srv = by_role["server"]
    gens = [by_role[f"gen{g}"] for g in range(p.n_gens)]
    raw = {
        "server_core": srv.cpus[0], "server_pol": s_tbl[triple(srv)],
        "gen_core": [gp.cpus[0] for gp in gens],
        "gen_pol": [g_tbl[triple(gp)] for gp in gens],
        "surplus_present": "surplus" in by_role,
        "surplus_core": by_role["surplus"].cpus[0] if "surplus" in by_role else p.housekeeping_core,
        "surplus_pol": u_tbl[triple(by_role["surplus"])] if "surplus" in by_role else 0,
    }
    return raw


# ----------------------------------------------------------------------------------------------------
# Rendering
# ----------------------------------------------------------------------------------------------------
def _fmt_placement(pl: Placement) -> str:
    pol = pl.policy.value.replace("SCHED_", "")
    extra = ""
    if pl.nice is not None:
        extra = f" nice={pl.nice}"
    if pl.latency_nice is not None:
        extra = f" latnice={pl.latency_nice}"
    return f"{pl.role}@cpu[{','.join(map(str, pl.cpus))}] {pol}{extra}"


def render_table(configs: list[TopologyConfig]) -> str:
    lines = []
    lines.append(f"{len(configs)} feasible topology configs\n")
    for i, cfg in enumerate(configs):
        lines.append(f"[{i:>3}] {cfg.config_id}")
        lines.append(f"      tag: {cfg.tag}")
        for pl in cfg.placements:
            lines.append(f"        {_fmt_placement(pl)}")
        lines.append("")
    return "\n".join(lines)


# ----------------------------------------------------------------------------------------------------
# The orbit invariant (neutral referee). PURE logic — it does NOT author the policy tables or the
# CP-SAT model (those are the SSOT's, in hp/). It maps a RAW solver-style rec to the lex-min image
# under the joint group G = Sym(isolated cores) x Sym(generators); core 0 is never permuted. Kept here
# because the migration parity test (hp/tests/test_topology_parity.py) uses it as a neutral referee to
# compare orbit PARTITIONS across reductions that may pick different representatives.
# ----------------------------------------------------------------------------------------------------
def _canonical_key(rec: dict, p: ModelParams) -> tuple:
    """The orbit invariant: the lexicographically smallest image of `rec` under the JOINT group
    G = Sym(isolated cores {1..n-1}) x Sym(generators). Core 0 (housekeeping) is NEVER permuted.
    Two raw solutions are isomorphic IFF they share this key. `rec` is the raw projection dict
    (server_core/server_pol/gen_core[list]/gen_pol[list]/surplus_present/surplus_core/surplus_pol)."""
    hk = p.housekeeping_core
    isolated = [c for c in range(p.n_cores) if c != hk]

    base: list[tuple] = [("s", rec["server_core"], rec["server_pol"])]
    for g in range(p.n_gens):
        base.append(("g", rec["gen_core"][g], rec["gen_pol"][g]))
    if rec["surplus_present"]:
        base.append(("u", rec["surplus_core"], rec["surplus_pol"]))

    best = None
    for sigma in itertools.permutations(isolated):
        relabel = dict(zip(isolated, sigma))  # bijection on isolated cores; hk fixed
        imaged = tuple(sorted(
            (role, relabel.get(core, core), pol) for (role, core, pol) in base
        ))
        if best is None or imaged < best:
            best = imaged
    return best


# ----------------------------------------------------------------------------------------------------
# Oracle A — the orbit self-check. The generalized implementation lives in hp/verify.py (oracle_a);
# this standalone fingerprint-based check is the original verify_orbits, kept here as the tool's own
# independent referee (it does NOT trust the hp canonicalizer — it recomputes each emitted config's
# orbit by brute force over isolated-core permutations, generator permutation quotiented by the
# anonymized-and-sorted fingerprint). DESIGN.md §6 step 4: "Keep verify_orbits as Oracle A."
# ----------------------------------------------------------------------------------------------------
def verify_orbits(configs: list[TopologyConfig], p: ModelParams) -> tuple[bool, str]:
    """Self-check the orbit reduction with an INDEPENDENT oracle: confirm the emitted configs are
    pairwise non-isomorphic under the JOINT group G = Sym(isolated cores) x Sym(generators). This
    recomputes the orbit by brute force over isolated-core permutations (generator permutation is
    quotiented by the anonymized-and-sorted fingerprint), so it does not trust the canonicalizer —
    it cross-checks the RESULT. Returns (ok, message)."""
    hk = p.housekeeping_core
    isolated = [c for c in range(p.n_cores) if c != hk]

    def fingerprint(cfg: TopologyConfig) -> tuple:
        items = []
        for pl in cfg.placements:
            role = "g" if pl.role.startswith("gen") else pl.role
            items.append((role, pl.cpus[0], pl.policy.value, pl.nice, pl.latency_nice))
        return tuple(sorted(items))

    def orbit(fp: tuple) -> set:
        out = set()
        for sigma in itertools.permutations(isolated):
            relabel = dict(zip(isolated, sigma))
            out.add(tuple(sorted(
                (role, relabel.get(core, core), pol, nice, lat) for (role, core, pol, nice, lat) in fp
            )))
        return out

    seen_orbit_rep: dict[tuple, str] = {}
    collisions = []
    for cfg in configs:
        fp = fingerprint(cfg)
        hit = next((seen_orbit_rep[img] for img in orbit(fp) if img in seen_orbit_rep), None)
        if hit is not None:
            collisions.append((hit, cfg.config_id))
        else:
            rep = min(orbit(fp))
            seen_orbit_rep[rep] = cfg.config_id
    n_orbits = len(set(seen_orbit_rep.values()))
    ok = not collisions and n_orbits == len(configs)
    msg = (f"orbit self-check: {len(configs)} configs, {n_orbits} distinct orbits, "
           f"{len(collisions)} under-collapse collision(s)")
    if collisions:
        msg += " -> FAIL: " + ", ".join(f"{a}~{b}" for a, b in collisions[:5])
    return ok, msg


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cores", type=int, default=4, help="number of vCPUs (default 4)")
    ap.add_argument("--gens", type=int, default=3, help="number of generators (default 3)")
    ap.add_argument("--housekeeping-core", type=int, default=0,
                    help="the contended IRQ/RCU core (default 0); cores != this are isolated")
    ap.add_argument("--json", default="configs.json", help="output JSON path (default ./configs.json)")
    ap.add_argument("--no-table", action="store_true", help="suppress the human-readable table")
    ap.add_argument("--verify", action="store_true",
                    help="run the independent orbit self-check and fail loudly if the reduction is "
                         "not exactly one-representative-per-orbit (ADR-0002)")
    args = ap.parse_args(argv)

    if args.gens + 1 > args.cores:
        print(f"error: {args.gens} generators + 1 server need >= {args.gens + 1} cores, have {args.cores}",
              file=sys.stderr)
        return 2

    params = ModelParams(n_cores=args.cores, n_gens=args.gens, housekeeping_core=args.housekeeping_core)
    configs = build_and_enumerate(params)

    if args.verify:
        ok, msg = verify_orbits(configs, params)
        print(msg, file=sys.stderr)
        if not ok:
            print("ERROR: orbit reduction is unsound — refusing to emit (ADR-0002)", file=sys.stderr)
            return 3

    payload = {
        "schema": "throughput-lab/topology_enum/v1",
        "params": dataclasses.asdict(params),
        "n_configs": len(configs),
        "configs": [c.to_record() for c in configs],
    }
    with open(args.json, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"wrote {len(configs)} configs -> {args.json}", file=sys.stderr)

    if not args.no_table:
        print(render_table(configs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
