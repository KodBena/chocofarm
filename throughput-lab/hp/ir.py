#!/usr/bin/env python3
"""
throughput-lab/hp/ir.py — the backend-neutral compiler IR for the HP config-space compiler.

A deliberately small, typed dataclass tree sitting between the selected SSOT descriptors
(spec.py / relations.py, lowered by compile.py) and a solver invocation. Two lowerings consume a
`ConfigSpace`: `backends/cpsat.py` (the CP-SAT enumerator) and `backends/grid.py` (the itertools
oracle). The two-lowering design IS the verification architecture (DESIGN.md §3/§4): you cannot
check CP-SAT against CP-SAT.

Three IR commitments (DESIGN.md §3), each falsifiable:
  - everything lowers to finite-domain integers (enums -> 0..n-1); no real arithmetic / quantifiers
    is needed by any throughput HP in the inventory, which is also why SMT's theory power is unused
    and CP-SAT is the fit;
  - `projection` is explicit and load-bearing: the config-DEFINING vars, deduped on; aux
    reification vars are never distinct-config keys;
  - `CanonInert` is an IR node (not a post-filter): both a feasibility-correctness mechanism (an
    inert flag at a non-default is not a real config) and the dominant symmetry-reduction mechanism
    (under strict-barrier all {D,N,S_min,chunk_floor} collapse to one).

Constructed objects validate at build time (ADR-0002: fail loudly) — a Var with lo>hi, a Linear
referencing an unknown var, etc. are construction errors, not silent garbage.

Public Domain (The Unlicense).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional, Union


class VarKind(str, Enum):
    """The finite-domain kind of an IR variable. All lower to a bounded integer."""
    INT = "int"
    BOOL = "bool"        # 0/1
    ENUM_IDX = "enum"    # 0..n-1 index into a value table


@dataclass(frozen=True)
class IRVar:
    """A finite-domain integer variable. Enums are indices; their value table is carried in
    `enum_values` so a lowering can map an index back to the human-facing string."""
    id: str
    kind: VarKind
    lo: int
    hi: int
    enum_values: tuple[str, ...] = ()   # only for ENUM_IDX; index i <-> enum_values[i]

    def __post_init__(self) -> None:
        if self.lo > self.hi:
            raise ValueError(f"IRVar {self.id!r}: lo={self.lo} > hi={self.hi} (empty domain)")
        if self.kind is VarKind.BOOL and (self.lo, self.hi) != (0, 1):
            raise ValueError(f"IRVar {self.id!r}: BOOL must have domain [0,1], got [{self.lo},{self.hi}]")
        if self.kind is VarKind.ENUM_IDX:
            if not self.enum_values:
                raise ValueError(f"IRVar {self.id!r}: ENUM_IDX needs enum_values")
            if (self.lo, self.hi) != (0, len(self.enum_values) - 1):
                raise ValueError(
                    f"IRVar {self.id!r}: ENUM_IDX domain must be [0,{len(self.enum_values)-1}], "
                    f"got [{self.lo},{self.hi}]")
        elif self.enum_values:
            raise ValueError(f"IRVar {self.id!r}: enum_values only valid for ENUM_IDX")


# --- expressions over vars (a tiny affine language; enough for the inventory's constraints) -------
@dataclass(frozen=True)
class VarRef:
    """A reference to a var's value (used in Clamp/Derived bounds)."""
    var: str


@dataclass(frozen=True)
class Const:
    value: int


@dataclass(frozen=True)
class CeilDiv:
    """ceil(num / den) — the only nonlinear shape in the inventory (K = ceil(pool_batch/threads)).
    Realized as integer arithmetic at lowering time; never a free var."""
    num: "Expr"
    den: "Expr"


Expr = Union[VarRef, Const, CeilDiv]


# --- constraints ----------------------------------------------------------------------------------
class Op(str, Enum):
    EQ = "=="
    NE = "!="
    LE = "<="
    GE = ">="
    LT = "<"


@dataclass(frozen=True)
class Linear:
    """sum(coeffs[v]*v) <op> rhs. A pure linear (in)equality over integer vars."""
    coeffs: dict[str, int]
    op: Op
    rhs: int


@dataclass(frozen=True)
class Implies:
    """If the literal `lit` is true (a BOOL var; prefix '~' negates), enforce `body`."""
    lit: str
    body: "IRConstr"


@dataclass(frozen=True)
class ReifyEq:
    """boolvar <=> (var == value). Materializes an occupancy/equality indicator."""
    boolvar: str
    var: str
    value: int


@dataclass(frozen=True)
class BoolAnd:
    """target <=> AND(members) (members are BOOL var ids, '~' negates)."""
    target: str
    members: tuple[str, ...]


@dataclass(frozen=True)
class AllDifferent:
    vars: tuple[str, ...]


@dataclass(frozen=True)
class MaxEquality:
    """target == max(members) — used for per-core occupancy booleans (R2/R3)."""
    target: str
    members: tuple[str, ...]


@dataclass(frozen=True)
class Clamp:
    """var in [lo, hi] where lo/hi may be Exprs over other vars (min_coalesce in [1, K])."""
    var: str
    lo: Expr
    hi: Expr


@dataclass(frozen=True)
class Derived:
    """var == expr(other vars). The dependent var is never free (K = ceil(batch/threads))."""
    var: str
    expr: Expr


@dataclass(frozen=True)
class CanonInert:
    """When the guard literal `when` is FALSE, pin `var` to `default_value`. THE key generalization
    (DESIGN.md §3/§4): an inert flag (its activation gate unsatisfied) is pinned to its default so
    inert combinations never multiply phantom configs. Both feasibility-correctness and the dominant
    symmetry-reduction mechanism. `when` is a BOOL var id ('~' negates)."""
    var: str
    default_value: int
    when: str


@dataclass(frozen=True)
class Table:
    """An explicit allowed-tuple constraint over `vars` (the escape hatch for awkward feasibility)."""
    vars: tuple[str, ...]
    allowed: tuple[tuple[int, ...], ...]


IRConstr = Union[Linear, Implies, ReifyEq, BoolAnd, AllDifferent, MaxEquality,
                 Clamp, Derived, CanonInert, Table]


# --- symmetry -------------------------------------------------------------------------------------
@dataclass(frozen=True)
class ValuePermutation:
    """A symmetry generator acting on var VALUES by a value-relabeling, applied jointly to a set of
    'role' var groups (DESIGN.md §4.2). E.g. permuting the isolated cores {1,2,3} relabels every
    placement var's core value. `relabelable_vars` is the set of vars whose value is a core id (so
    the relabel applies to them); `anchors` (e.g. core 0) are values never permuted.

    For the joint group, the canonicalizer (cpsat.py) takes the lex-min image over all permutations
    of `movable_values`, where any var in `relabelable_vars` has its value relabeled, AND any
    'interchangeable replica' group is re-sorted (handled by `replica_groups`)."""
    name: str
    movable_values: tuple[int, ...]            # the value set permuted (e.g. isolated cores 1,2,3)
    relabelable_vars: tuple[str, ...]          # vars whose VALUE is one of movable_values
    rationale: str = ""


@dataclass(frozen=True)
class ReplicaGroup:
    """A set of interchangeable replica 'slots', each described by a tuple of vars (its per-slot
    fields). Permuting the slots yields an equivalent config; the canonicalizer sorts the slots by
    their packed value tuple. E.g. the 3 generators, each (core, pol)."""
    name: str
    slots: tuple[tuple[str, ...], ...]         # each slot = the ordered tuple of its field var ids
    rationale: str = ""


@dataclass(frozen=True)
class IRSym:
    """The declared symmetry group acting on var-tuples by value (DESIGN.md §3)."""
    permutations: tuple[ValuePermutation, ...] = ()
    replica_groups: tuple[ReplicaGroup, ...] = ()
    anchors: tuple[int, ...] = ()              # values never permuted (housekeeping core 0)


# --- the config space -----------------------------------------------------------------------------
@dataclass(frozen=True)
class ConfigSpace:
    """The backend-neutral target the two lowerings consume."""
    vars: tuple[IRVar, ...]
    constrs: tuple[IRConstr, ...]
    sym: IRSym
    projection: tuple[str, ...]                # the CONFIG-DEFINING vars (dedup key); aux excluded
    provenance: dict = field(default_factory=dict)   # name -> Effect record, for the ledger

    def __post_init__(self) -> None:
        ids = {v.id for v in self.vars}
        if len(ids) != len(self.vars):
            dupes = [v.id for v in self.vars if [w.id for w in self.vars].count(v.id) > 1]
            raise ValueError(f"ConfigSpace: duplicate var ids {sorted(set(dupes))}")
        # Validate every constraint references known vars (fail loud — ADR-0002).
        for c in self.constrs:
            for ref in _constr_var_refs(c):
                bare = ref.lstrip("~")
                if bare not in ids:
                    raise ValueError(f"ConfigSpace: constraint {type(c).__name__} references "
                                     f"unknown var {bare!r}")
        for pv in self.projection:
            if pv not in ids:
                raise ValueError(f"ConfigSpace: projection var {pv!r} not in vars")
        for perm in self.sym.permutations:
            for v in perm.relabelable_vars:
                if v not in ids:
                    raise ValueError(f"ConfigSpace: permutation {perm.name} references unknown "
                                     f"var {v!r}")
        for rg in self.sym.replica_groups:
            for slot in rg.slots:
                for v in slot:
                    if v not in ids:
                        raise ValueError(f"ConfigSpace: replica group {rg.name} references unknown "
                                         f"var {v!r}")

    def var(self, vid: str) -> IRVar:
        for v in self.vars:
            if v.id == vid:
                return v
        raise KeyError(vid)


def _expr_var_refs(e: Expr) -> list[str]:
    if isinstance(e, VarRef):
        return [e.var]
    if isinstance(e, Const):
        return []
    if isinstance(e, CeilDiv):
        return _expr_var_refs(e.num) + _expr_var_refs(e.den)
    raise TypeError(f"unknown Expr {e!r}")


def _constr_var_refs(c: IRConstr) -> list[str]:
    """Every var id a constraint touches (literals keep their '~' so callers can strip it)."""
    if isinstance(c, Linear):
        return list(c.coeffs.keys())
    if isinstance(c, Implies):
        return [c.lit] + _constr_var_refs(c.body)
    if isinstance(c, ReifyEq):
        return [c.boolvar, c.var]
    if isinstance(c, BoolAnd):
        return [c.target, *c.members]
    if isinstance(c, AllDifferent):
        return list(c.vars)
    if isinstance(c, MaxEquality):
        return [c.target, *c.members]
    if isinstance(c, Clamp):
        return [c.var] + _expr_var_refs(c.lo) + _expr_var_refs(c.hi)
    if isinstance(c, Derived):
        return [c.var] + _expr_var_refs(c.expr)
    if isinstance(c, CanonInert):
        return [c.var, c.when]
    if isinstance(c, Table):
        return list(c.vars)
    raise TypeError(f"unknown IRConstr {c!r}")
