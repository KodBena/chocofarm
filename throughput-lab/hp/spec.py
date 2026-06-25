#!/usr/bin/env python3
"""
throughput-lab/hp/spec.py — the HP SSOT registry and its descriptor algebra.

ONE Python module is the single home (ADR-0012 P1) of every throughput-affecting hyperparameter's
*metadata*. It does NOT re-author the domains/defaults that already live in a C++ struct, an
argparse block, or a dataclass; each descriptor carries a `home: SourceRef` pointing at the real
home, and the drift lint (tests/test_ssot_drift.py) enforces agreement (DESIGN.md §1.4). The only
descriptors that may carry a literal default are those with `home = NoCodeHome(reason)`.

The descriptor algebra makes illegal configs unrepresentable (ADR-0000): a Domain validates its
default at construction; a `Measured` effect cannot be built without an `EvidenceRef` (ADR-0009 as
a construction-time type); an `HParam` name must be unique across the registry (P1 uniqueness).

Surfaces (the SELECTION axis, DESIGN.md §5):
  - TOPOLOGY   : the process/scheduling-topology space hoisted from harness/topology_enum.py.
  - OVERCOMMIT : the cpp/stage_a/overcommit_sweep.py parameter region (wire_mode x N x S_min x
                 chunk_floor x D x theta x the 1:3 placement).
  - STATIC_LAB : the throughput-lab/ producer + server + scheduling + build HPs.

Public Domain (The Unlicense).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Union


# ==================================================================================================
# Surfaces, kinds
# ==================================================================================================
class Surface(str, Enum):
    TOPOLOGY = "topology"
    OVERCOMMIT = "overcommit"
    STATIC_LAB = "static_lab"


class Kind(str, Enum):
    PRODUCER_FLAG = "producer_flag"
    SERVER_FLAG = "server_flag"
    SCHEDULING = "scheduling"
    BUILD = "build"
    HARNESS_MEASURE = "harness_measure"


# ==================================================================================================
# Domains — a closed union; construction validates (default in domain, lo<=hi). ADR-0000/ADR-0002.
# ==================================================================================================
@dataclass(frozen=True)
class IntRange:
    lo: int
    hi: int

    def __post_init__(self) -> None:
        if self.lo > self.hi:
            raise ValueError(f"IntRange lo={self.lo} > hi={self.hi}")

    def contains(self, v: object) -> bool:
        return isinstance(v, int) and self.lo <= v <= self.hi

    def materialize(self) -> tuple[int, ...]:
        return tuple(range(self.lo, self.hi + 1))


@dataclass(frozen=True)
class IntSet:
    """An explicit swept ladder (e.g. fibers {0,1,8,32,64,128,256})."""
    values: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.values:
            raise ValueError("IntSet is empty")
        object.__setattr__(self, "values", tuple(sorted(set(self.values))))

    def contains(self, v: object) -> bool:
        return v in self.values

    def materialize(self) -> tuple[int, ...]:
        return self.values


@dataclass(frozen=True)
class FloatRange:
    lo: float
    hi: float

    def __post_init__(self) -> None:
        if self.lo > self.hi:
            raise ValueError(f"FloatRange lo={self.lo} > hi={self.hi}")

    def contains(self, v: object) -> bool:
        return isinstance(v, (int, float)) and self.lo <= v <= self.hi


@dataclass(frozen=True)
class EnumSet:
    """A small string enum, e.g. {"round-sync","greedy"}."""
    values: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.values:
            raise ValueError("EnumSet is empty")
        if len(set(self.values)) != len(self.values):
            raise ValueError(f"EnumSet has duplicates: {self.values}")

    def contains(self, v: object) -> bool:
        return v in self.values

    def materialize(self) -> tuple[str, ...]:
        return self.values


@dataclass(frozen=True)
class Bool:
    def contains(self, v: object) -> bool:
        return isinstance(v, bool)

    def materialize(self) -> tuple[bool, ...]:
        return (False, True)


@dataclass(frozen=True)
class Categorical:
    """A finite set of opaque (hashable) values, e.g. cpu-lists, sorted bucket tuples."""
    values: tuple[object, ...]

    def __post_init__(self) -> None:
        if not self.values:
            raise ValueError("Categorical is empty")

    def contains(self, v: object) -> bool:
        return v in self.values


@dataclass(frozen=True)
class DerivedFrom:
    """A dependent dimension computed from others — NEVER a free var (K = ceil(batch/threads))."""
    fn_name: str                 # human label of the rule (e.g. "ceil(pool_batch/pool_threads)")
    deps: tuple[str, ...]        # the HP names it derives from

    def contains(self, v: object) -> bool:   # derived vars are not free; membership is vacuous
        return True


Domain = Union[IntRange, IntSet, FloatRange, EnumSet, Bool, Categorical, DerivedFrom]


# ==================================================================================================
# SourceRef — where the authoritative definition lives. The SSOT derives, never copies (§1.4).
# ==================================================================================================
@dataclass(frozen=True)
class CppField:
    file: str
    symbol: str            # e.g. "WireRunnerConfig::min_coalesce"


@dataclass(frozen=True)
class CppFlag:
    file: str
    flag: str              # e.g. "--msg-rows"


@dataclass(frozen=True)
class PyArg:
    file: str
    dest: str              # an argparse dest


@dataclass(frozen=True)
class PyField:
    file: str
    symbol: str            # a dataclass/ServerConfig field


@dataclass(frozen=True)
class NoCodeHome:
    """The ONLY case where a literal default is permitted in the SSOT (a genuine runtime-only fact,
    a substrate constant with no static line). Named, not buried (§1.4)."""
    reason: str


SourceRef = Union[CppField, CppFlag, PyArg, PyField, NoCodeHome]


# ==================================================================================================
# Effect — ADR-0009 made a construction-time type. MEASURED REQUIRES an EvidenceRef.
# Effects ANNOTATE; they NEVER prune (DESIGN.md §0 invariant). Effect != Constraint.
# ==================================================================================================
@dataclass(frozen=True)
class EvidenceRef:
    doc: str
    locus: str             # a section/anchor inside the doc


@dataclass(frozen=True)
class Measured:
    sign: str              # "+", "-", "0"
    note: str
    evidence: EvidenceRef

    def __post_init__(self) -> None:
        if self.sign not in ("+", "-", "0"):
            raise ValueError(f"Measured sign must be one of +,-,0; got {self.sign!r}")
        if not isinstance(self.evidence, EvidenceRef):
            raise ValueError("Measured REQUIRES an EvidenceRef (ADR-0009)")


@dataclass(frozen=True)
class Hypothesized:
    rationale: str


@dataclass(frozen=True)
class Unknown:
    note: str = ""


Effect = Union[Measured, Hypothesized, Unknown]


# ==================================================================================================
# Symmetry class of an HP.
# ==================================================================================================
@dataclass(frozen=True)
class Free:
    pass


@dataclass(frozen=True)
class Interchangeable:
    group_id: str          # this HP is one of a permutable set (replicas, threads)


@dataclass(frozen=True)
class Asymmetric:
    reason: str            # never permuted (housekeeping core 0)


@dataclass(frozen=True)
class OrderInsensitive:
    pass                   # a set/tuple HP; stored sorted (bucket set)


SymmetryClass = Union[Free, Interchangeable, Asymmetric, OrderInsensitive]


# ==================================================================================================
# Guard — a predicate over OTHER HPs' values. The conditional-feature ("staged configuration") gate.
# A deselected parent's children are not free dimensions (DESIGN.md §1.1).
# ==================================================================================================
@dataclass(frozen=True)
class Eq:
    hp: str
    value: object


@dataclass(frozen=True)
class IsTrue:
    hp: str


@dataclass(frozen=True)
class And:
    parts: tuple["Guard", ...]


@dataclass(frozen=True)
class Or:
    parts: tuple["Guard", ...]


Guard = Union[Eq, IsTrue, And, Or]


def eq(hp: str, value: object) -> Eq:
    return Eq(hp, value)


def is_true(hp: str) -> IsTrue:
    return IsTrue(hp)


def and_(*parts: Guard) -> And:
    return And(tuple(parts))


def or_(*parts: Guard) -> Or:
    return Or(tuple(parts))


def guard_hps(g: Guard) -> set[str]:
    """The HP names a guard depends on (for the fail-loud selection rule, §5)."""
    if isinstance(g, Eq):
        return {g.hp}
    if isinstance(g, IsTrue):
        return {g.hp}
    if isinstance(g, (And, Or)):
        out: set[str] = set()
        for p in g.parts:
            out |= guard_hps(p)
        return out
    raise TypeError(f"unknown Guard {g!r}")


# ==================================================================================================
# Binding — one concept, many per-surface/per-variant bindings (DESIGN.md §1.3).
# ==================================================================================================
@dataclass(frozen=True)
class Clamp:
    """A typed [lo, hi] clamp where hi may reference another HP by name (the [1,K] clamp)."""
    lo: int
    hi_ref: Optional[str] = None    # another HP name; None => no upper clamp
    hi_const: Optional[int] = None


@dataclass(frozen=True)
class Binding:
    surface: Surface
    flag: str                       # the surface-specific flag/arg name
    home: SourceRef
    variant: Optional[str] = None   # "real" | "synthetic" | None (variant-agnostic)
    clamp: Optional[Clamp] = None


# ==================================================================================================
# HParam — the descriptor. One canonical name; uniqueness enforced by the registry.
# ==================================================================================================
@dataclass(frozen=True)
class HParam:
    name: str
    concept: str
    surfaces: frozenset[Surface]
    kind: Kind
    home: SourceRef
    domain: Domain
    default: object | None        # None iff a SourceRef supplies it (derived; §1.4)
    symmetry: SymmetryClass = field(default_factory=Free)
    effect: Effect = field(default_factory=Unknown)
    activation: Optional[Guard] = None
    bindings: tuple[Binding, ...] = ()

    def __post_init__(self) -> None:
        # NoCodeHome is the ONLY case permitted to carry a literal default; everything else derives.
        if isinstance(self.home, NoCodeHome):
            if self.default is None and not isinstance(self.domain, DerivedFrom):
                raise ValueError(f"HParam {self.name!r}: NoCodeHome requires a literal default")
        # If a literal default is given, it MUST be in the domain (ADR-0000/ADR-0002).
        if self.default is not None and not self.domain.contains(self.default):
            raise ValueError(
                f"HParam {self.name!r}: default {self.default!r} not in its domain {self.domain!r}")
        # Bindings must be for declared surfaces.
        for b in self.bindings:
            if b.surface not in self.surfaces:
                raise ValueError(
                    f"HParam {self.name!r}: binding for {b.surface} but it is not in surfaces "
                    f"{self.surfaces}")

    def binding_for(self, surface: Surface, variant: Optional[str]) -> Optional[Binding]:
        """The binding that resolves for (surface, variant). A variant-specific binding wins over a
        variant-agnostic one; a None-variant request accepts the agnostic binding."""
        cands = [b for b in self.bindings if b.surface == surface]
        if variant is not None:
            exact = [b for b in cands if b.variant == variant]
            if exact:
                return exact[0]
        agnostic = [b for b in cands if b.variant is None]
        if agnostic:
            return agnostic[0]
        return cands[0] if cands else None


# ==================================================================================================
# The registry — the single home. Construction enforces name uniqueness (P1).
# ==================================================================================================
class Registry:
    def __init__(self, params: list[HParam]):
        self._by_name: dict[str, HParam] = {}
        for p in params:
            if p.name in self._by_name:
                raise ValueError(f"duplicate HParam name {p.name!r} (ADR-0012 P1: one home)")
            self._by_name[p.name] = p

    def __getitem__(self, name: str) -> HParam:
        return self._by_name[name]

    def __contains__(self, name: str) -> bool:
        return name in self._by_name

    def all(self) -> list[HParam]:
        return list(self._by_name.values())

    def for_surface(self, surface: Surface) -> list[HParam]:
        return [p for p in self._by_name.values() if surface in p.surfaces]

    def names(self) -> list[str]:
        return list(self._by_name.keys())


# --------------------------------------------------------------------------------------------------
# Cross-surface concepts (DESIGN.md §1.3). Each concept has ONE descriptor with per-surface bindings.
# --------------------------------------------------------------------------------------------------
JOURNEY = "docs/notes/tlab-performance-journey-2026-06-24.md"
ADAPTER = "docs/design/cpp-eval-transport-adapter.md"
FINDINGS = "throughput_research.tlab_finding (exp_db)"   # evidence pointer to the deliberate findings layer


# ==================================================================================================
# THE SSOT REGISTRY
# ==================================================================================================
# --- TOPOLOGY surface: the scheduling/placement vocabulary (single home: hp/relations.py) ----------
# The topology placement/policy space is structural (a joint permutation group over isolated cores x
# generators) and lives in relations.py (PlacementConstraints + the per-role policy vocabularies).
# The per-class SCHED-policy vocabularies are the HParams here: each names the enum domain and its
# measured effect, and cites its `home` = the (policy, nice, latency_nice) table in relations.py (the
# single home — ADR-0012 P1). The standalone harness/topology_enum.py CONSUMES this home; it no longer
# re-authors the tables, so the citation points at the real producer of the values, not at the tool.
POLICY_HOME = "throughput-lab/hp/relations.py"  # the single home of the policy triples

_TOPOLOGY: list[HParam] = [
    HParam(
        name="server_policy", concept="sched_policy",
        surfaces=frozenset({Surface.TOPOLOGY}), kind=Kind.SCHEDULING,
        home=PyField(POLICY_HOME, "SERVER_POLICIES"),
        domain=EnumSet(("SCHED_OTHER_LATNICE", "SCHED_OTHER")),
        default="SCHED_OTHER_LATNICE",
        symmetry=Free(),
        effect=Unknown("OTHER_LATNICE modeled as a null-testable server option; no measured verdict"),
        bindings=(Binding(Surface.TOPOLOGY, flag="server_pol",
                          home=PyField(POLICY_HOME, "SERVER_POLICIES")),),
    ),
    HParam(
        name="gen_policy", concept="sched_policy",
        surfaces=frozenset({Surface.TOPOLOGY}), kind=Kind.SCHEDULING,
        home=PyField(POLICY_HOME, "GEN_POLICIES"),
        domain=EnumSet(("SCHED_OTHER", "SCHED_BATCH")),
        default="SCHED_OTHER",
        # A SINGLE uniform knob applied to ALL generators: the generators are interchangeable, so
        # "which generator runs BATCH" carries no signal (topology_enum.py comment).
        symmetry=Interchangeable("generators"),
        effect=Measured("-", "SCHED_BATCH -1% vs OTHER (surplus_policy_control A/B)",
                        EvidenceRef(JOURNEY, "1e/2")),
        bindings=(Binding(Surface.TOPOLOGY, flag="gen_pol",
                          home=PyField(POLICY_HOME, "GEN_POLICIES")),),
    ),
    HParam(
        name="surplus_policy", concept="sched_policy",
        surfaces=frozenset({Surface.TOPOLOGY}), kind=Kind.SCHEDULING,
        home=PyField(POLICY_HOME, "SURPLUS_POLICIES"),
        domain=EnumSet(("SCHED_IDLE", "SCHED_BATCH")),
        default="SCHED_IDLE",
        symmetry=Free(),
        effect=Measured("+", "SCHED_IDLE surplus on server core +18-25%; nice/BATCH weaker",
                        EvidenceRef(JOURNEY, "1e/2")),
        bindings=(Binding(Surface.TOPOLOGY, flag="surplus_pol",
                          home=PyField(POLICY_HOME, "SURPLUS_POLICIES")),),
    ),
    HParam(
        name="surplus_present", concept="surplus_present",
        surfaces=frozenset({Surface.TOPOLOGY}), kind=Kind.SCHEDULING,
        # surplus_present is a structural ENUMERATION axis (the 0/1 bool the placement model ranges
        # over), not a value with a code-home literal: no struct field / argparse default authors it.
        # Its default False is the "absent" baseline. NoCodeHome is the sanctioned case for a literal
        # default with no static line (DESIGN.md §1.4); named, not buried.
        home=NoCodeHome("surplus_present is an enumeration axis (0/1); no code-home literal — the "
                        "default False is the absent baseline the placement model spans"),
        domain=Bool(), default=False,
        symmetry=Free(),
        effect=Measured("+", "present + SCHED_IDLE on server core = +18-25%",
                        EvidenceRef(JOURNEY, "1e/2")),
        bindings=(Binding(Surface.TOPOLOGY, flag="surplus_present",
                          home=NoCodeHome("enumeration axis; no code-home literal")),),
    ),
]

# --- OVERCOMMIT surface: the overcommit parameter region ------------------------------------------
# NB: overcommit_sweep.py / server_gen_floor_grid.py were MOLTED 2026-06-25 (subsumed by throughput-lab;
# git history). The two operating-point values that homed on overcommit_sweep.py's argparse (pool_threads=3,
# pool_batch=64) are now NoCodeHome literals in THIS hp/ SSOT — their code-home is gone, so hp/ owns them
# (ADR-0012 P1: one home, here a literal-with-reason). GEN_FLOOR_GRID survives only as an EVIDENCE-provenance
# label (EvidenceRef; the drift lint never OPENS it) pointing at the molted file in git history. The live
# runner header (RUNNER_HPP) stays — it is the kept binding for the C++ struct defaults.
RUNNER_HPP = "cpp/include/chocofarm/runner_wire_batched.hpp"
GEN_FLOOR_GRID = "cpp/stage_a/server_gen_floor_grid.py"   # MOLTED — historical evidence source (git history)
INFER_SERVER = "chocofarm/az/inference_server.py"

_OVERCOMMIT: list[HParam] = [
    HParam(
        name="wire_mode", concept="inflight_depth",
        surfaces=frozenset({Surface.OVERCOMMIT}), kind=Kind.PRODUCER_FLAG,
        home=CppField(RUNNER_HPP, "WireRunnerConfig::mode"),
        domain=EnumSet(("strict-barrier", "pipelined-bucket")),
        default="strict-barrier",
        symmetry=Free(),
        effect=Measured("+", "pipelined-bucket beats strict-barrier (stage_b_ab arm3 vs arm1)",
                        EvidenceRef(ADAPTER, "§4/§6")),
        activation=None,   # the master gate; always live
        bindings=(Binding(Surface.OVERCOMMIT, flag="--wire-mode",
                          home=CppField(RUNNER_HPP, "WireMode")),),
    ),
    HParam(
        name="trees_per_thread", concept="overcommit_multiplier",
        surfaces=frozenset({Surface.OVERCOMMIT}), kind=Kind.PRODUCER_FLAG,
        home=CppField(RUNNER_HPP, "WireRunnerConfig::trees_per_thread"),
        # overcommit_sweep default sweep {1,2,3} (N=4 has a separate stall bug, capped out).
        domain=IntSet((1, 2, 3)), default=1,
        symmetry=Free(),
        effect=Hypothesized("N multiplies in-flight leaves toward server fast region B~192"),
        activation=eq("wire_mode", "pipelined-bucket"),   # inert under strict-barrier (D=1,N=1)
        bindings=(Binding(Surface.OVERCOMMIT, flag="--trees-per-thread",
                          home=CppField(RUNNER_HPP, "WireRunnerConfig::trees_per_thread")),),
    ),
    HParam(
        name="chunk_floor", concept="chunk_floor",
        surfaces=frozenset({Surface.OVERCOMMIT}), kind=Kind.PRODUCER_FLAG,
        home=CppField(RUNNER_HPP, "WireRunnerConfig::chunk_floor"),
        domain=Bool(), default=False,
        symmetry=Free(),
        effect=Measured("0", "winning region gen=ON & theta=0 at N=9 (regime-specific)",
                        EvidenceRef(GEN_FLOOR_GRID, "refine_configs")),
        activation=eq("wire_mode", "pipelined-bucket"),   # inert under strict-barrier
        bindings=(Binding(Surface.OVERCOMMIT, flag="--gen-chunk-floor",
                          home=CppField(RUNNER_HPP, "WireRunnerConfig::chunk_floor")),),
    ),
    HParam(
        name="min_coalesce", concept="coalesce_degree",
        surfaces=frozenset({Surface.OVERCOMMIT}), kind=Kind.PRODUCER_FLAG,
        home=CppField(RUNNER_HPP, "WireRunnerConfig::min_coalesce"),
        # grid levels {16,32,64,128}; clamped to [1,K]. We model the swept ladder.
        domain=IntSet((16, 32, 64, 128)), default=32,
        symmetry=Free(),
        effect=Measured("0", "identical B/dps at S_min=1 vs 32 on the drain-all (chunk_floor=0) path",
                        EvidenceRef(GEN_FLOOR_GRID, "docstring")),
        # S_min BINDS only when chunk_floor=1 AND wire_mode=pipelined-bucket; otherwise inert.
        activation=and_(eq("wire_mode", "pipelined-bucket"), is_true("chunk_floor")),
        bindings=(Binding(Surface.OVERCOMMIT, flag="--min-coalesce",
                          home=CppField(RUNNER_HPP, "WireRunnerConfig::min_coalesce"),
                          clamp=Clamp(lo=1, hi_ref="fibers_per_thread")),),
    ),
    HParam(
        name="max_inflight_msgs", concept="inflight_depth_magnitude",
        surfaces=frozenset({Surface.OVERCOMMIT}), kind=Kind.PRODUCER_FLAG,
        home=CppField(RUNNER_HPP, "WireRunnerConfig::max_inflight_msgs"),
        # grid levels {4,8,16,32}.
        domain=IntSet((4, 8, 16, 32)), default=8,
        symmetry=Free(),
        effect=Hypothesized("caps the pipeline depth chunk_floor creates; inert without chunk_floor"),
        activation=and_(eq("wire_mode", "pipelined-bucket"), is_true("chunk_floor")),
        bindings=(Binding(Surface.OVERCOMMIT, flag="--inflight-msgs",
                          home=CppField(RUNNER_HPP, "WireRunnerConfig::max_inflight_msgs")),),
    ),
    HParam(
        name="min_forward_rows", concept="server_drain_floor",
        surfaces=frozenset({Surface.OVERCOMMIT}), kind=Kind.SERVER_FLAG,
        home=PyField(INFER_SERVER, "min_forward_rows"),
        # swept THETA {0,128,256,384,512,768} in the grid; 0 disables.
        domain=IntSet((0, 128, 256, 384, 512, 768)), default=0,
        symmetry=Free(),
        effect=Measured("0", "theta>0 neutral-to-harmful at N=9 (regime-specific; NOT a prune)",
                        EvidenceRef(GEN_FLOOR_GRID, "refine_configs")),
        activation=None,
        bindings=(Binding(Surface.OVERCOMMIT, flag="--min-forward-rows",
                          home=PyArg(INFER_SERVER, "min_forward_rows")),),
    ),
    # pool_threads / pool_batch are the free axes; fibers_per_thread K is DERIVED (never free).
    # pool_threads / pool_batch: the C++ struct (pool_threads=4, pool_batch=32) is the logical definition
    # (the kept CppField binding below), but the OVERCOMMIT *operating point* (3 / 64) was set by the
    # overcommit_sweep.py argparse. That harness was MOLTED 2026-06-25, so the operating-point value now homes
    # on a NoCodeHome literal in THIS SSOT (its code-home is gone; the drift lint reads the literal, not a
    # file). The C++ struct default stays as the named binding divergence. (DESIGN.md §1.3/§1.4.)
    HParam(
        name="pool_threads", concept="producer_threads",
        surfaces=frozenset({Surface.OVERCOMMIT}), kind=Kind.PRODUCER_FLAG,
        home=NoCodeHome("OVERCOMMIT operating point: 3 producer threads (the 1:3 core pin). Its argparse "
                        "code-home overcommit_sweep.py --threads=3 was MOLTED 2026-06-25 (subsumed by "
                        "throughput-lab); the value is now a literal owned by this SSOT. The C++ struct "
                        "default WireRunnerConfig::pool_threads=4 stays as the kept binding (named divergence)."),
        domain=IntSet((3,)), default=3,
        symmetry=Interchangeable("producer_threads"),
        effect=Hypothesized("the 1:3 pin (3 producer cores) sets the natural ceiling"),
        bindings=(Binding(Surface.OVERCOMMIT, flag="--pool-threads",
                          home=CppField(RUNNER_HPP, "WireRunnerConfig::pool_threads")),),
    ),
    HParam(
        name="pool_batch", concept="pool_batch",
        surfaces=frozenset({Surface.OVERCOMMIT}), kind=Kind.PRODUCER_FLAG,
        home=NoCodeHome("OVERCOMMIT operating point: pool_batch=64. Its argparse code-home overcommit_sweep.py "
                        "--pool-batch=64 was MOLTED 2026-06-25; the value is now a literal owned by this SSOT. "
                        "The C++ struct default WireRunnerConfig::pool_batch=32 stays as the kept binding."),
        domain=IntSet((64,)), default=64,
        symmetry=Free(),
        effect=Hypothesized("sets per-thread fiber count K (concurrency width)"),
        bindings=(Binding(Surface.OVERCOMMIT, flag="--pool-batch",
                          home=CppField(RUNNER_HPP, "WireRunnerConfig::pool_batch")),),
    ),
    HParam(
        name="fibers_per_thread", concept="fibers_per_thread",
        surfaces=frozenset({Surface.OVERCOMMIT}), kind=Kind.PRODUCER_FLAG,
        home=NoCodeHome("K = ceil(pool_batch/pool_threads); a DERIVED dimension, never a free flag"),
        domain=DerivedFrom("ceil(pool_batch/pool_threads)", ("pool_batch", "pool_threads")),
        default=None,
        symmetry=Free(),
        effect=Unknown("derived width; same concept as static-lab --fibers"),
        bindings=(),
    ),
]

# --- STATIC_LAB surface: the throughput-lab producer + server operating-point knobs ----------------
# The LIVE tlab stack (tlab-real-producer + the throughput-lab server), distinct from the molted OVERCOMMIT
# runner. Each descriptor's `default` is the CODE-HOME default (drift-linted: producer flags home on the
# real_producer.cpp ternary defaults via CppFlag; n_sims/m on the GumbelConfig struct via CppField; the
# server knobs on server/__main__.py argparse via PyArg). The BANKED operating point (the tuned values
# episodic_dps.sh actually runs) is the SEPARATE BANKED_STATIC below — the pool_threads/pool_batch pattern
# (a tuned value diverging from the code default). Names are unique; `concept` is shared with OVERCOMMIT
# where the knob is the same idea (the sched_policy precedent: many HParams, one concept label).
PRODUCER_CPP = "throughput-lab/cpp/real_producer.cpp"
GUMBEL_HPP = "cpp/include/chocofarm/gumbel.hpp"
SERVER_MAIN = "throughput-lab/server/__main__.py"

_STATIC_LAB: list[HParam] = [
    HParam(
        name="fibers", concept="fibers_per_thread",
        surfaces=frozenset({Surface.STATIC_LAB}), kind=Kind.PRODUCER_FLAG,
        home=CppFlag(PRODUCER_CPP, "--fibers"),
        domain=IntSet((0, 64, 256, 1024, 2048, 4096)), default=0,
        symmetry=Free(),
        effect=Measured("+", "K=1024 is the knee: 256->1024 ~+25% (~100k->~125k); 2048/4096 plateau",
                        EvidenceRef(FINDINGS, "#17 HP-sweep optimum")),
        bindings=(Binding(Surface.STATIC_LAB, flag="--fibers", home=CppFlag(PRODUCER_CPP, "--fibers")),),
    ),
    HParam(
        name="msg_rows", concept="msg_coalesce",
        surfaces=frozenset({Surface.STATIC_LAB}), kind=Kind.PRODUCER_FLAG,
        home=CppFlag(PRODUCER_CPP, "--msg-rows"),
        domain=IntSet((1, 64, 128, 256)), default=1,
        symmetry=Free(),
        effect=Measured("+", "MSG=256 fills the 256 max-batch; beats 128 but ONLY with K>=1024 to supply it",
                        EvidenceRef(FINDINGS, "#17")),
        bindings=(Binding(Surface.STATIC_LAB, flag="--msg-rows", home=CppFlag(PRODUCER_CPP, "--msg-rows")),),
    ),
    HParam(
        name="inflight_msgs", concept="inflight_depth_magnitude",
        surfaces=frozenset({Surface.STATIC_LAB}), kind=Kind.PRODUCER_FLAG,
        home=CppFlag(PRODUCER_CPP, "--inflight-msgs"),
        domain=IntSet((4, 8, 10, 16, 32)), default=8,
        symmetry=Free(),
        effect=Measured("0", "inflight 6-12 is null at the banked point (paired nonparametric, p=0.65)",
                        EvidenceRef(FINDINGS, "#19 inflight-10-vs-8")),
        bindings=(Binding(Surface.STATIC_LAB, flag="--inflight-msgs",
                          home=CppFlag(PRODUCER_CPP, "--inflight-msgs")),),
    ),
    HParam(
        name="driver", concept="producer_pipe_shape",
        surfaces=frozenset({Surface.STATIC_LAB}), kind=Kind.PRODUCER_FLAG,
        home=CppFlag(PRODUCER_CPP, "--driver"),
        domain=EnumSet(("round-sync", "greedy")), default="round-sync",
        symmetry=Free(),
        effect=Measured("+", "greedy wins-or-ties everywhere; clean +15% at the lean ladder (regime-dependent)",
                        EvidenceRef(JOURNEY, "driver A/B")),
        bindings=(Binding(Surface.STATIC_LAB, flag="--driver", home=CppFlag(PRODUCER_CPP, "--driver")),),
    ),
    HParam(
        name="seconds", concept="measure_window",
        surfaces=frozenset({Surface.STATIC_LAB}), kind=Kind.HARNESS_MEASURE,
        home=CppFlag(PRODUCER_CPP, "--seconds"),
        domain=FloatRange(1.0, 120.0), default=5.0,
        symmetry=Free(),
        effect=Measured("0", "run length CONFOUNDS the reading ~6% (non-monotone, supply-side, NOT a "
                             "compute throttle); s>=30 is producer-unstable. Bank 10s.",
                        EvidenceRef(FINDINGS, "#21 runlen / #22 instability")),
        bindings=(Binding(Surface.STATIC_LAB, flag="--seconds", home=CppFlag(PRODUCER_CPP, "--seconds")),),
    ),
    HParam(
        name="n_sims", concept="search_sims",
        surfaces=frozenset({Surface.STATIC_LAB}), kind=Kind.PRODUCER_FLAG,
        home=CppField(GUMBEL_HPP, "GumbelConfig::n_sims"),
        domain=IntSet((48, 128, 256)), default=48,
        symmetry=Free(),
        effect=Hypothesized("search budget per decision; 256 is the production-shape config"),
        bindings=(Binding(Surface.STATIC_LAB, flag="--n-sims",
                          home=CppField(GUMBEL_HPP, "GumbelConfig::n_sims")),),
    ),
    HParam(
        name="m", concept="search_m",
        surfaces=frozenset({Surface.STATIC_LAB}), kind=Kind.PRODUCER_FLAG,
        home=CppField(GUMBEL_HPP, "GumbelConfig::m"),
        domain=IntSet((12, 24)), default=12,
        symmetry=Free(),
        effect=Hypothesized("root actions sampled by Gumbel-Top-k; 24 is the production config"),
        bindings=(Binding(Surface.STATIC_LAB, flag="--m", home=CppField(GUMBEL_HPP, "GumbelConfig::m")),),
    ),
    HParam(
        name="max_batch", concept="server_drain_cap",
        surfaces=frozenset({Surface.STATIC_LAB}), kind=Kind.SERVER_FLAG,
        home=PyArg(SERVER_MAIN, "max_batch"),
        domain=IntSet((256, 512, 1024, 4096)), default=4096,
        symmetry=Free(),
        effect=Measured("+", "256 = the single-pull point (cap==MSG_ROWS, clean 256 bucket); 4096 = the "
                             "legacy overcommit regime (~37% pad util, the topology-screen confound)",
                        EvidenceRef(FINDINGS, "#17 / #20")),
        bindings=(Binding(Surface.STATIC_LAB, flag="--max-batch", home=PyArg(SERVER_MAIN, "max_batch")),),
    ),
    HParam(
        name="warmup_ladder", concept="bucket_ladder",
        surfaces=frozenset({Surface.STATIC_LAB}), kind=Kind.SERVER_FLAG,
        # The banked bucket ladder (64,256). The server's DEFAULT ladder is a ServerConfig list literal
        # (1,8,64,512,4096); a list-valued drift-lint is the filed P7 deferral (DESIGN.md §9), so this homes
        # as a NoCodeHome literal carrying the banked value directly (the sanctioned literal-with-reason).
        home=NoCodeHome("banked server bucket ladder (64,256); the server's default-ladder list-literal "
                        "drift-lint is the P7 deferral (DESIGN.md §9) — list-valued homes not yet extracted"),
        domain=Categorical(((64, 256), (64, 256, 512), (1, 8, 64, 512, 4096))), default=(64, 256),
        symmetry=OrderInsensitive(),
        effect=Measured("+", "cap at 256 (no 512 spill) is the well-filled efficient regime at K=1024/MSG=256",
                        EvidenceRef(FINDINGS, "#17")),
        bindings=(Binding(Surface.STATIC_LAB, flag="--warmup",
                          home=NoCodeHome("banked ladder literal; list-default lint is the P7 deferral")),),
    ),
]

_REGISTRY = Registry(_TOPOLOGY + _OVERCOMMIT + _STATIC_LAB)


def registry() -> Registry:
    return _REGISTRY


# ==================================================================================================
# BANKED OPERATING POINTS — a SELECTED point within an enumerated surface, NOT a sweep axis.
# An HParam models an AXIS (a domain the compiler enumerates over); a banked operating point names the
# JOINT config the lab actually RUNS at — the tuned winner of a sweep. It is deliberately NOT an HParam:
# forcing a 40-way config_id into a TOPOLOGY-surface axis would add an enumeration variable and break the
# bit-for-bit parity gate (DESIGN.md §6). And the per-axis enumeration defaults above are NOT the bank
# (server_policy defaults to LATNICE, surplus_present to False) — those are the axis baselines the space
# spans, not the chosen point. Owned by this SSOT as a NoCodeHome operating point (the same status as the
# OVERCOMMIT pool_threads=3 / pool_batch=64 literals): its only prior home was a hand-pinned literal
# smeared across episodic_dps.sh's taskset args plus a duplicated provenance string — exactly the drift
# this hoist closes. The config_id is validated against the live enumeration at resolve time (consumers
# fail loud on an unknown id — ADR-0002); harness/topology_enum.py is the resolver (config_by_id).
@dataclass(frozen=True)
class OperatingPoint:
    """A SELECTED config within an enumerated surface (the tuned winner the lab runs), with provenance.
    Distinct from an HParam (a sweep axis): this names a joint point, not a domain to enumerate."""
    surface: Surface
    value: object
    home: SourceRef
    effect: Effect
    note: str = ""


BANKED_TOPOLOGY = OperatingPoint(
    surface=Surface.TOPOLOGY,
    value="s2p1_g0.0-1.0-3.0_u2p0",   # server isolated@2 + gens@0,1,3 + SCHED_IDLE surplus@2
    home=NoCodeHome("banked topology operating point: tuned across the 40-config quiet-box screen + a "
                    "paired test; no code-home literal — owned by this SSOT the way pool_threads/pool_batch "
                    "are. Resolved to placements by harness/topology_enum.config_by_id (validated)."),
    effect=Measured("+", "server off the housekeeping core 0: +0.68% (paired, one-sided p=0.045, "
                         "two-sided p=0.090, bootstrap CI straddles 0) — a LOW-REGRET adoption, NOT a "
                         "clean win; tlab_finding #20 (status provisional)",
                    EvidenceRef(FINDINGS, "#20 / topo-pair-20260625T080841Z")),
    note="server isolated@2 + gens@0,1,3 + surplus@2 IDLE. The generalizable lever is 'server NOT on the "
         "housekeeping core 0'; core 2 is one representative (the server-isolated family was tied).",
)


def banked_topology_config_id() -> str:
    """The single home of the banked process-topology choice (a config_id into the TOPOLOGY surface).
    Consumers resolve it to placements via harness.topology_enum.config_by_id (which validates membership
    against the live enumeration — ADR-0002). Replaces the hand-pinned taskset literals in episodic_dps.sh."""
    return BANKED_TOPOLOGY.value


# The banked STATIC_LAB operating point: the tuned producer/server values episodic_dps.sh runs at, the ONE
# home harnesses derive their DEFAULTS from (override args stay, for sweeps). Each value diverges from the
# code-home default (the HParam.default the drift lint guards) the way pool_threads=3/pool_batch=64 do — a
# tuned point, not the binary's out-of-box default. seconds=10 is the banked measurement window (finding #21:
# run length confounds the reading; 10s is stable + on the clean plateau, avoiding the s=14 dip / s>=30 break).
BANKED_STATIC = OperatingPoint(
    surface=Surface.STATIC_LAB,
    value={
        "fibers": 1024, "msg_rows": 256, "inflight_msgs": 8, "driver": "greedy", "seconds": 10,
        "n_sims": 256, "m": 24, "max_batch": 256, "warmup_ladder": (64, 256),
    },
    home=NoCodeHome("banked tlab producer/server operating point (HP sweep #17 + the inflight/topology/"
                    "run-length findings); owned by this SSOT, the pool_threads/pool_batch pattern."),
    effect=Measured("+", "K=1024/MSG=256/greedy/single-pull (max-batch 256) ~+25% over the prior banked "
                         "point; the fine knobs (inflight, exact run length) are noise-floor-limited",
                    EvidenceRef(FINDINGS, "#17 / #19 / #20 / #21")),
    note="harnesses derive their defaults from this; --seconds standardized at 10 to kill the cross-harness "
         "run-length confound (episodic ran 14, topology_sweep 5/10 — the gap that triggered this).",
)


def banked_static() -> "dict[str, object]":
    """The banked STATIC_LAB operating point as a plain dict, VALIDATED against the registry domains at call
    time (ADR-0000: a banked value outside its HParam domain is unrepresentable — a loud error, not a silent
    bad default). The single home episodic_dps.sh / topology_sweep derive their defaults from."""
    reg = registry()
    out = dict(BANKED_STATIC.value)
    for name, val in out.items():
        if name not in reg:
            raise ValueError(f"banked_static: {name!r} is not a registered HParam (ADR-0002)")
        dom = reg[name].domain
        if not dom.contains(val):
            raise ValueError(f"banked_static: {name}={val!r} is outside its domain {dom!r} (ADR-0000)")
    return out


def banked_static_env() -> str:
    """Emit the banked STATIC_LAB operating point as shell `eval`-able BANKED_* assignments — the single
    home episodic_dps.sh / topology_sweep derive their DEFAULTS from (override args stay). Validated by
    banked_static() (each value in its HParam domain). warmup_ladder is comma-joined for the --warmup flag."""
    b = banked_static()
    warmup = ",".join(str(x) for x in b["warmup_ladder"])
    return "\n".join([
        f"BANKED_FIBERS={b['fibers']}",
        f"BANKED_MSG_ROWS={b['msg_rows']}",
        f"BANKED_INFLIGHT={b['inflight_msgs']}",
        f"BANKED_DRIVER={b['driver']}",
        f"BANKED_SECONDS={b['seconds']}",
        f"BANKED_N_SIMS={b['n_sims']}",
        f"BANKED_M={b['m']}",
        f"BANKED_MAX_BATCH={b['max_batch']}",
        f"BANKED_WARMUP={warmup}",
    ])


def ceil_div(num: int, den: int) -> int:
    return -(-num // den)
