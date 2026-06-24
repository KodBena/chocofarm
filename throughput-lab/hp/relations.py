#!/usr/bin/env python3
"""
throughput-lab/hp/relations.py — cross-HP structure that is NOT per-HP.

The topology orbit group, the 1:3-pin partition, and the feasibility predicates (the placement
reasonableness R1-R4) live here as declarative objects, each carrying its PROSE RATIONALE so the
"every constraint carries its why" discipline of topology_enum.py survives the hoist (DESIGN.md
§1.5). compile.py reads these to build the IR; spec.py owns per-HP metadata, relations.py owns the
inter-HP structure.

The topology placement substrate is parameterized by (n_cores, n_gens, housekeeping_core) — the
same ModelParams the standalone topology_enum.py takes — so compile(Target(TOPOLOGY)) reproduces
its space for the default (4 cores / 3 generators).

Public Domain (The Unlicense).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


# ==================================================================================================
# Declarative cross-HP structure (each carries its rationale, preserving topology_enum's R1-R4 why).
# ==================================================================================================
@dataclass(frozen=True)
class Feasibility:
    expr: str               # a human-readable predicate (also realized in compile.py as IR)
    rationale: str


@dataclass(frozen=True)
class Partition:
    whole: str
    parts: tuple[str, ...]
    rationale: str


# ==================================================================================================
# TOPOLOGY placement model — the substrate + the per-role policy vocabularies + R1-R4.
# These mirror topology_enum.py EXACTLY (the hoist must not change the space — DESIGN.md §6).
# The policy vocabularies are (SchedPolicy, nice, latency_nice) triples, index i <-> the i-th entry,
# matching topology_enum.SERVER_POLICIES / GEN_POLICIES / SURPLUS_POLICIES order so the enumerated
# config_ids match bit-for-bit.
# ==================================================================================================
SERVER_POLICIES: tuple[tuple[str, Optional[int], Optional[int]], ...] = (
    ("SCHED_OTHER_LATNICE", None, -20),
    ("SCHED_OTHER", 0, None),
)
GEN_POLICIES: tuple[tuple[str, Optional[int], Optional[int]], ...] = (
    ("SCHED_OTHER", 0, None),
    ("SCHED_BATCH", 0, None),
)
SURPLUS_POLICIES: tuple[tuple[str, Optional[int], Optional[int]], ...] = (
    ("SCHED_IDLE", None, None),
    ("SCHED_BATCH", 10, None),
)


@dataclass(frozen=True)
class TopologyParams:
    """The placement substrate — the same axes as topology_enum.ModelParams."""
    n_cores: int = 4
    n_gens: int = 3
    housekeeping_core: int = 0

    def __post_init__(self) -> None:
        if self.n_gens + 1 > self.n_cores:
            raise ValueError(
                f"{self.n_gens} generators + 1 server need >= {self.n_gens + 1} cores, "
                f"have {self.n_cores}")
        if not (0 <= self.housekeeping_core < self.n_cores):
            raise ValueError(f"housekeeping_core {self.housekeeping_core} out of range")


@dataclass(frozen=True)
class PlacementConstraints:
    """The R1-R4 reasonableness constraints, with rationale (topology_enum.py)."""
    rules: tuple[Feasibility, ...] = field(default_factory=lambda: (
        Feasibility(
            "all_different(gen_core) AND no server on a generator's core",
            "R1: each of the n_gens generators gets a CLEAN, EXCLUSIVE core; generators are the "
            "bottleneck work and a generator never shares with the server."),
        Feasibility(
            "if n_cores == n_gens+1 then every core occupied",
            "R2/R3: no clean isolated core sits idle while generators contend (full occupancy when "
            "cores == gens+1)."),
        Feasibility(
            "surplus (if present) MUST co-locate (sum of other occupants on its core >= 1)",
            "R4: a surplus on its own private core is just a 4th generator; co-location is its "
            "defining purpose."),
        Feasibility(
            "if surplus shares the server's core then surplus_pol == SCHED_IDLE (index 0)",
            "R4: co-location is the surplus's exclusive privilege and only via a low-priority "
            "policy; sharing the latency-sensitive server demands the yield-instantly policy."),
        Feasibility(
            "when surplus absent: surplus_core pinned to housekeeping, surplus_pol pinned to 0",
            "canonicalize unused vars so enumeration does not multiply phantom absent-surplus "
            "solutions (gated on presence so it cannot collide with a server also on hk)."),
    ))


@dataclass(frozen=True)
class TopologySymmetry:
    """The joint orbit group G = Sym(isolated cores) x Sym(generators); core 0 (housekeeping) is an
    asymmetric anchor never permuted. Mirrors topology_enum._canonical_key / verify_orbits."""
    rationale: str = (
        "the isolated cores {1..n-1} permute jointly with the generator relabeling; core 0 is an "
        "asymmetric resource (IRQ/RCU) and is NEVER permuted (the orbit boundary).")


# ==================================================================================================
# OVERCOMMIT structure — the 1:3 pin partition + the activation-gate collapses (CanonInert).
# The collapses themselves are derived from the per-HP `activation` guards in spec.py; this records
# the partition and the rationale for the gate structure as cross-HP relations.
# ==================================================================================================
OVERCOMMIT_PARTITION = Partition(
    whole="cores", parts=("server_core", "producer_cores"),
    rationale="the 1:3 split: 1 server core : 3 producer cores on the 4-vCPU box.")

OVERCOMMIT_FEASIBILITY: tuple[Feasibility, ...] = (
    Feasibility(
        "under wire_mode=strict-barrier: trees_per_thread, chunk_floor, min_coalesce, "
        "max_inflight_msgs are INERT (pinned to default)",
        "wire_mode is the master gate; strict-barrier is structurally D=1, N=1 (the production path "
        "untouched), so those axes carry no signal and must not multiply phantom configs."),
    Feasibility(
        "under chunk_floor=0: min_coalesce, max_inflight_msgs are INERT (pinned to default)",
        "the drain-all path emits a single message, depth=1; S_min/D only BIND when chunk_floor=1 "
        "(measured: identical B/dps at S_min=1 vs 32 on the drain-all path)."),
    Feasibility(
        "min_coalesce in [1, fibers_per_thread] (= ceil(pool_batch/pool_threads))",
        "min_coalesce is clamped to [1,K]; K is a derived dimension, never a free flag."),
)
