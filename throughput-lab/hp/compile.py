#!/usr/bin/env python3
"""
throughput-lab/hp/compile.py — the SELECTION affordance + the SSOT -> IR lowering.

`Target` projects the SSOT (spec.py registry + relations.py structure) to a sub-space; `compile`
selects the descriptors whose surfaces intersect the target (or are explicitly included), resolves
per-surface/per-variant bindings, applies pins, and lowers to a backend-neutral `ConfigSpace`
(ir.py) that backends/cpsat.py enumerates and backends/grid.py independently re-derives.

Two worked selections, each a regression test (DESIGN.md §5):
  - Target(surfaces={TOPOLOGY}) MUST reproduce topology_enum.py --gens 3 --cores 4 bit-for-bit
    (the migration acceptance gate, §6) — so the topology lowering reconstructs that exact model.
  - Target(surfaces={OVERCOMMIT}) compiles the overcommit_sweep.py region, with CanonInert killing
    the strict-barrier and chunk_floor=0 phantoms for free.

Fail-loud selection rule (ADR-0002, §5): a Target naming an HP whose activation depends on an
UNSELECTED HP refuses (it cannot honestly emit a one-value axis the receiver thinks is live).

Public Domain (The Unlicense).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from . import relations as rel
from . import spec
from .ir import (
    AllDifferent, BoolAnd, CanonInert, ConfigSpace, Const, Derived, Implies,
    IRSym, IRVar, Linear, MaxEquality, Op, ReifyEq, ReplicaGroup, Table, VarKind,
    ValuePermutation,
)
from .spec import (
    And, Bool, Categorical, DerivedFrom, Eq, EnumSet, FloatRange, Guard, HParam, IntRange,
    IntSet, IsTrue, Measured, Or, Surface, guard_hps,
)


# ==================================================================================================
# Target — the typed selection (DESIGN.md §5).
# ==================================================================================================
@dataclass(frozen=True)
class Target:
    surfaces: Optional[frozenset[Surface]] = None     # None => all
    include: Optional[frozenset[str]] = None          # explicit HP names (union with surfaces)
    pin: dict[str, object] = field(default_factory=dict)
    variant: Optional[str] = None                     # real | synthetic (producer-variant axis)
    # topology substrate (only used when TOPOLOGY is selected)
    topo: rel.TopologyParams = field(default_factory=rel.TopologyParams)


# ==================================================================================================
# compile — dispatch on which surface(s) are selected. TOPOLOGY has a bespoke placement lowering
# (it reconstructs topology_enum's exact CP-SAT model); the flag/enum surfaces use a uniform
# product+gate lowering.
# ==================================================================================================
def compile(reg: spec.Registry, target: Target) -> ConfigSpace:
    sel = _selected_surfaces(reg, target)
    # Run the fail-loud scope/refusal check on the FULL chosen set up front, before dispatch, so the
    # topology fast path cannot smuggle past it (DESIGN.md §5).
    chosen = _selected_hps(reg, target, sel)
    chosen_names = {p.name for p in chosen}
    # the flag-surface HPs in scope (TOPOLOGY's placement HPs are handled by the bespoke lowering).
    flag_surfaces = sel - {Surface.TOPOLOGY}
    has_flag_hps = any(p.surfaces & flag_surfaces for p in chosen) or bool(
        target.include and any(reg[n].surfaces & flag_surfaces for n in target.include))

    if Surface.TOPOLOGY in sel and not has_flag_hps:
        return _compile_topology(reg, target)
    if Surface.TOPOLOGY in sel:
        # Composition of TOPOLOGY with a flag surface: combine the two ConfigSpaces by a Cartesian
        # union of vars/constrs (disjoint var namespaces). Kept simple and explicit.
        topo_cs = _compile_topology(reg, target)
        flat_cs = _compile_flag_surfaces(reg, target, flag_surfaces)
        return _combine(topo_cs, flat_cs)
    return _compile_flag_surfaces(reg, target, sel)


def _selected_surfaces(reg: spec.Registry, target: Target) -> set[Surface]:
    if target.surfaces is not None:
        return set(target.surfaces)
    # include-only selection: the union of the included HPs' surfaces
    if target.include is not None:
        out: set[Surface] = set()
        for n in target.include:
            out |= set(reg[n].surfaces)
        return out
    return set(Surface)


def _selected_hps(reg: spec.Registry, target: Target, sel: set[Surface]) -> list[HParam]:
    chosen: dict[str, HParam] = {}
    for p in reg.all():
        if (p.surfaces & sel) or (target.include and p.name in target.include):
            chosen[p.name] = p
    # Fail-loud selection rule (§5): every HP whose activation depends on an HP NOT in scope refuses.
    in_scope = set(chosen)
    for p in chosen.values():
        if p.activation is not None:
            deps = guard_hps(p.activation)
            missing = deps - in_scope
            if missing:
                raise ValueError(
                    f"selection refused (ADR-0002): {p.name!r} is gated on {sorted(missing)} which "
                    f"is/are not in scope — include it/them or this axis would be silently inert")
    return list(chosen.values())


# ==================================================================================================
# Flag-surface lowering: a finite product over the selected HPs, with activation gates realized as
# CanonInert (pin an inert HP to its default), per-binding clamps, and derived dims.
# ==================================================================================================
def _effective_default(p: HParam):
    """The literal default; for derived/code-home params with default=None we use the inventory's
    documented default. (The drift lint enforces agreement with the code home; here we need a value
    to pin inert axes to.)"""
    if p.default is not None:
        return p.default
    return _DOCUMENTED_DEFAULT[p.name]


# The documented code-home defaults (the lint in tests/test_ssot_drift.py is what enforces these
# against the cited SourceRef; we keep them in ONE place here so CanonInert has a value to pin to).
_DOCUMENTED_DEFAULT: dict[str, object] = {
    "wire_mode": "strict-barrier",
    "trees_per_thread": 1,
    "chunk_floor": False,
    "min_coalesce": 32,
    "max_inflight_msgs": 8,
    "min_forward_rows": 0,
    "pool_threads": 3,
    "pool_batch": 64,
}


def _var_for(p: HParam) -> IRVar:
    """One IR var per selected free HP. Enums/bools -> index/0-1; IntSet -> a 0..n-1 ladder index
    plus a value table carried via enum_values is not used (we keep ints as the value directly for
    IntRange but use a contiguous index for IntSet so domains stay compact). To keep the oracle and
    solver in lockstep we materialize IntSet as an ENUM_IDX over its string-rendered values."""
    d = p.domain
    if isinstance(d, Bool):
        return IRVar(p.name, VarKind.BOOL, 0, 1)
    if isinstance(d, EnumSet):
        return IRVar(p.name, VarKind.ENUM_IDX, 0, len(d.values) - 1, enum_values=d.values)
    if isinstance(d, IntSet):
        vals = tuple(str(v) for v in d.values)
        return IRVar(p.name, VarKind.ENUM_IDX, 0, len(vals) - 1, enum_values=vals)
    if isinstance(d, IntRange):
        return IRVar(p.name, VarKind.INT, d.lo, d.hi)
    if isinstance(d, DerivedFrom):
        # derived var: a wide INT bound; the Derived constraint pins its value.
        return IRVar(p.name, VarKind.INT, 0, 1 << 20)
    if isinstance(d, (FloatRange, Categorical)):
        raise ValueError(f"HP {p.name!r}: domain {type(d).__name__} not lowerable to finite int "
                         f"(no inter-HP arithmetic constraint uses it; pin it or exclude it)")
    raise TypeError(f"unknown domain {d!r}")


def _index_of(p: HParam, value: object) -> int:
    """The IR index for a value of HP p (enum/intset are index-coded; bool/int are direct)."""
    d = p.domain
    if isinstance(d, Bool):
        return 1 if value else 0
    if isinstance(d, EnumSet):
        return d.values.index(value)
    if isinstance(d, IntSet):
        return d.values.index(value)
    if isinstance(d, IntRange):
        return int(value)
    raise TypeError(f"cannot index {value!r} in {d!r}")


def _guard_to_lits(g: Guard, reg: spec.Registry) -> list[tuple[str, int]]:
    """Flatten a guard to a conjunction of (var, required_index) atoms. Only AND/Eq/IsTrue are
    supported for the flag surface (the inventory's gates are all conjunctive); Or raises."""
    if isinstance(g, Eq):
        return [(g.hp, _index_of(reg[g.hp], g.value))]
    if isinstance(g, IsTrue):
        return [(g.hp, 1)]
    if isinstance(g, And):
        out: list[tuple[str, int]] = []
        for p in g.parts:
            out += _guard_to_lits(p, reg)
        return out
    if isinstance(g, Or):
        raise NotImplementedError("Or-guards on the flag surface are not used by the inventory")
    raise TypeError(f"unknown guard {g!r}")


def _compile_flag_surfaces(reg: spec.Registry, target: Target, sel: set[Surface]) -> ConfigSpace:
    hps = _selected_hps(reg, target, sel)
    # Stable order: registry order.
    order = {p.name: i for i, p in enumerate(reg.all())}
    hps.sort(key=lambda p: order[p.name])

    vars_: list[IRVar] = []
    constrs: list = []
    projection: list[str] = []
    provenance: dict = {}

    derived: list[HParam] = []
    free: list[HParam] = []
    for p in hps:
        if isinstance(p.domain, DerivedFrom):
            derived.append(p)
        else:
            free.append(p)

    for p in free:
        vars_.append(_var_for(p))
        projection.append(p.name)
        provenance[p.name] = _provenance_entry(p)

    # derived dims (K = ceil(pool_batch/pool_threads)) — NOT in projection (not config-defining).
    # Deps are VALUE-coded in spec but lowered as index-coded vars; arithmetic must use VALUES. When
    # the deps are single-valued (the overcommit operating point pool_batch=64, pool_threads=3), we
    # resolve K to the concrete value and pin it (still DERIVED from the one home, not copied). A
    # genuinely-multi-valued derivation would need a value<->index Table; none exists in the
    # inventory, so we fail loud rather than emit a wrong index-arithmetic constraint.
    for p in derived:
        assert isinstance(p.domain, DerivedFrom)
        vars_.append(_var_for(p))
        provenance[p.name] = _provenance_entry(p)
        deps = p.domain.deps
        if p.domain.fn_name.startswith("ceil(") and len(deps) == 2:
            num_hp, den_hp = (reg[deps[0]], reg[deps[1]])
            nv = _single_value(num_hp, target)
            dv = _single_value(den_hp, target)
            if nv is not None and dv is not None:
                constrs.append(Derived(p.name, Const(spec.ceil_div(int(nv), int(dv)))))
            else:
                raise NotImplementedError(
                    f"derived {p.name}: deps {deps} are multi-valued; index-arithmetic CeilDiv is "
                    f"unsound — add a value<->index Table lowering (none needed by the inventory)")
        else:
            raise NotImplementedError(f"derived rule {p.domain.fn_name!r} not lowerable")

    # pins (finalize axes) as equalities.
    for name, value in target.pin.items():
        if name not in {p.name for p in hps}:
            raise ValueError(f"pin {name!r} not in the selected sub-space")
        idx = _index_of(reg[name], value)
        constrs.append(Linear({name: 1}, Op.EQ, idx))

    # activation gates -> CanonInert: when the gate is unsatisfied, pin the child to its default idx.
    # A multi-atom gate is satisfied only when ALL atoms hold; "unsatisfied" = at least one atom
    # false. CanonInert pins on a single boolean `when`. We synthesize an aux boolean = AND(atoms)
    # and pin the child when ~aux.
    aux_counter = 0
    activation_aux: dict[str, str] = {}   # HP name -> its activation AND-bool var id
    for p in free:
        if p.activation is None:
            continue
        atoms = _guard_to_lits(p.activation, reg)
        # build a boolean per atom (var == required idx), AND them, CanonInert the child on ~aux.
        atom_bools: list[str] = []
        for (gv, gidx) in atoms:
            if gv not in {q.name for q in hps}:
                # guarded by an out-of-scope var — already refused by _selected_hps.
                continue
            ab = f"__act_{p.name}_{gv}_{gidx}"
            vars_.append(IRVar(ab, VarKind.BOOL, 0, 1))
            constrs.append(ReifyEq(ab, gv, gidx))
            atom_bools.append(ab)
        if not atom_bools:
            continue
        aux = f"__act_all_{p.name}_{aux_counter}"
        aux_counter += 1
        vars_.append(IRVar(aux, VarKind.BOOL, 0, 1))
        constrs.append(BoolAnd(aux, tuple(atom_bools)))
        activation_aux[p.name] = aux
        # CanonInert pins `var` when `when` is FALSE. The gate is SATISFIED iff aux==1, so the child
        # is INERT iff aux==0 (aux FALSE) — pass `when=aux` so the pin fires exactly when inert.
        constrs.append(CanonInert(p.name, _index_of(p, _effective_default(p)), aux))

    # per-binding clamps (min_coalesce in [1,K]). The clamp BINDS only when the HP is ACTIVE; when
    # inert, CanonInert already pins it to its default (which may itself lie OUTSIDE [1,K] — that is
    # fine, it is a don't-care). So the clamp is realized as a JOINT Table over (activation_aux,
    # min_coalesce): aux=1 -> only clamp-satisfying ladder indices; aux=0 -> only the default index.
    for p in free:
        clamp_b = next((bb for bb in p.bindings if bb.clamp is not None
                        and bb.surface in sel), None)
        if clamp_b is not None and clamp_b.clamp is not None:
            _apply_value_clamp(p, reg, clamp_b.clamp, constrs, target,
                               activation_aux.get(p.name))

    return ConfigSpace(
        vars=tuple(vars_), constrs=tuple(constrs), sym=IRSym(),
        projection=tuple(projection), provenance=provenance)


def _apply_value_clamp(p: HParam, reg: spec.Registry, cl, constrs: list, target: Target,
                       act_aux: Optional[str]) -> None:
    """Realize a value clamp min_coalesce in [lo, K] as a Table of allowed indices. K resolves to a
    concrete int when pool_batch/pool_threads are single-valued (they are, in the overcommit region:
    pool_batch=64, pool_threads=3 -> K=ceil(64/3)=22). When the HP is gated (act_aux given) the clamp
    BINDS only when active; when inert it is the CanonInert default (which may exceed K — a
    don't-care)."""
    assert isinstance(p.domain, IntSet)
    hi_val: Optional[int] = None
    if cl.hi_const is not None:
        hi_val = cl.hi_const
    elif cl.hi_ref is not None:
        ref = reg[cl.hi_ref]
        if isinstance(ref.domain, DerivedFrom) and ref.domain.fn_name.startswith("ceil("):
            num_hp, den_hp = ref.domain.deps
            num = _single_value(reg[num_hp], target)
            den = _single_value(reg[den_hp], target)
            if num is not None and den is not None:
                hi_val = spec.ceil_div(int(num), int(den))
    if hi_val is None:
        return  # cannot resolve a concrete K; leave unclamped (the swept ladder stands)
    clamp_ok = [p.domain.values.index(v) for v in p.domain.values if cl.lo <= v <= hi_val]
    if not clamp_ok:
        raise ValueError(f"clamp on {p.name}: no ladder value in [{cl.lo},{hi_val}]")
    default_idx = p.domain.values.index(_effective_default(p))
    if act_aux is None:
        # unconditional clamp (the HP is always active in this selection).
        if len(clamp_ok) < len(p.domain.values):
            constrs.append(Table((p.name,), tuple((i,) for i in clamp_ok)))
        return
    # JOINT (aux, p) Table: active (aux=1) -> clamp-satisfying indices; inert (aux=0) -> default idx.
    rows: list[tuple[int, int]] = []
    for i in clamp_ok:
        rows.append((1, i))
    rows.append((0, default_idx))
    constrs.append(Table((act_aux, p.name), tuple(rows)))


def _single_value(p: HParam, target: Target) -> Optional[int]:
    if p.name in target.pin:
        return int(target.pin[p.name])  # type: ignore[arg-type]
    if isinstance(p.domain, IntSet) and len(p.domain.values) == 1:
        return p.domain.values[0]
    if p.default is not None and isinstance(p.default, int):
        return p.default
    return None


def _provenance_entry(p: HParam) -> dict:
    e = p.effect
    if isinstance(e, Measured):
        return {"effect": "measured", "sign": e.sign, "note": e.note,
                "evidence": {"doc": e.evidence.doc, "locus": e.evidence.locus}}
    if isinstance(e, spec.Hypothesized):
        return {"effect": "hypothesized", "note": e.rationale}
    return {"effect": "unknown", "note": getattr(e, "note", "")}


# ==================================================================================================
# TOPOLOGY lowering — reconstruct topology_enum.py's exact CP-SAT model in IR form so the enumerator
# reproduces it bit-for-bit (DESIGN.md §6). Var ids match topology_enum so the materialized
# config_ids are identical.
# ==================================================================================================
def _compile_topology(reg: spec.Registry, target: Target) -> ConfigSpace:
    p = target.topo
    cores = list(range(p.n_cores))
    hk = p.housekeeping_core
    isolated = [c for c in cores if c != hk]
    G = p.n_gens

    server_pol_hp = reg["server_policy"]
    gen_pol_hp = reg["gen_policy"]
    surplus_pol_hp = reg["surplus_policy"]
    n_server_pol = len(rel.SERVER_POLICIES)
    n_gen_pol = len(rel.GEN_POLICIES)
    n_surplus_pol = len(rel.SURPLUS_POLICIES)

    vars_: list[IRVar] = []
    constrs: list = []

    # placement vars (same ids as topology_enum).
    vars_.append(IRVar("server_core", VarKind.INT, 0, p.n_cores - 1))
    for g in range(G):
        vars_.append(IRVar(f"gen{g}_core", VarKind.INT, 0, p.n_cores - 1))
    vars_.append(IRVar("surplus_present", VarKind.BOOL, 0, 1))
    vars_.append(IRVar("surplus_core", VarKind.INT, 0, p.n_cores - 1))

    # policy vars.
    vars_.append(IRVar("server_pol", VarKind.INT, 0, n_server_pol - 1))
    vars_.append(IRVar("gen_pol", VarKind.INT, 0, n_gen_pol - 1))   # uniform across generators
    vars_.append(IRVar("surplus_pol", VarKind.INT, 0, n_surplus_pol - 1))

    # occupancy reifications.
    for c in cores:
        vars_.append(IRVar(f"server_on_{c}", VarKind.BOOL, 0, 1))
        constrs.append(ReifyEq(f"server_on_{c}", "server_core", c))
    for g in range(G):
        for c in cores:
            vars_.append(IRVar(f"gen{g}_on_{c}", VarKind.BOOL, 0, 1))
            constrs.append(ReifyEq(f"gen{g}_on_{c}", f"gen{g}_core", c))
    for c in cores:
        vars_.append(IRVar(f"surplus_at_{c}", VarKind.BOOL, 0, 1))
        constrs.append(ReifyEq(f"surplus_at_{c}", "surplus_core", c))
    for c in cores:
        vars_.append(IRVar(f"surplus_on_{c}", VarKind.BOOL, 0, 1))
        # surplus_on[c] <=> surplus_present AND surplus_at[c]
        constrs.append(BoolAnd(f"surplus_on_{c}", ("surplus_present", f"surplus_at_{c}")))

    # R1: generators on distinct cores; no server on a generator's core.
    constrs.append(AllDifferent(tuple(f"gen{g}_core" for g in range(G))))
    for g in range(G):
        for c in cores:
            # NOT(gen_on AND server_on): at most one true.
            constrs.append(Linear({f"gen{g}_on_{c}": 1, f"server_on_{c}": 1}, Op.LE, 1))

    # R2/R3: per-core occupancy + full occupancy when n_cores == n_gens+1.
    for c in cores:
        occ = f"occupied_{c}"
        vars_.append(IRVar(occ, VarKind.BOOL, 0, 1))
        members = [f"server_on_{c}"] + [f"gen{g}_on_{c}" for g in range(G)] + [f"surplus_on_{c}"]
        constrs.append(MaxEquality(occ, tuple(members)))
    if p.n_cores == p.n_gens + 1:
        for c in cores:
            constrs.append(Linear({f"occupied_{c}": 1}, Op.EQ, 1))

    # R4: surplus (if present) co-locates: surplus_on[c] implies sum(other occupants) >= 1.
    for c in cores:
        others = [f"server_on_{c}"] + [f"gen{g}_on_{c}" for g in range(G)]
        constrs.append(Implies(f"surplus_on_{c}", Linear({v: 1 for v in others}, Op.GE, 1)))

    # surplus_eq_server (raw, presence-agnostic) + surplus_with_server (AND presence).
    vars_.append(IRVar("surplus_eq_server", VarKind.BOOL, 0, 1))
    # surplus_eq_server <=> (surplus_core == server_core): a reified equality of two vars. Realized
    # via a Linear difference reification through an aux. The IR has ReifyEq(var==const) only, so we
    # encode it with a Table over (surplus_core, server_core, surplus_eq_server).
    eq_allowed = []
    for sc in range(p.n_cores):
        for vc in range(p.n_cores):
            eq_allowed.append((sc, vc, 1 if sc == vc else 0))
    constrs.append(Table(("surplus_core", "server_core", "surplus_eq_server"), tuple(eq_allowed)))

    vars_.append(IRVar("surplus_with_server", VarKind.BOOL, 0, 1))
    constrs.append(BoolAnd("surplus_with_server", ("surplus_present", "surplus_eq_server")))
    # sharing the server's core demands IDLE (index 0).
    constrs.append(Implies("surplus_with_server", Linear({"surplus_pol": 1}, Op.EQ, 0)))

    # absent-surplus canonicalization: pin surplus_core to hk, surplus_pol to 0 when absent.
    constrs.append(CanonInert("surplus_core", hk, "surplus_present"))
    constrs.append(CanonInert("surplus_pol", 0, "surplus_present"))

    # S1: generator within-class lex break on packed key core*K + pol (uniform pol => order by core).
    K = n_gen_pol
    for g in range(G):
        key = f"gen{g}_key"
        vars_.append(IRVar(key, VarKind.INT, 0, (p.n_cores - 1) * K + (K - 1)))
        # key == gen_core*K + gen_pol
        constrs.append(Linear({key: 1, f"gen{g}_core": -K, "gen_pol": -1}, Op.EQ, 0))
    for g in range(G - 1):
        constrs.append(Linear({f"gen{g}_key": 1, f"gen{g+1}_key": -1}, Op.LT, 0))

    # --- symmetry: the joint orbit group, declared for the canonicalizer + oracle A ----------------
    core_vars = (["server_core"] + [f"gen{g}_core" for g in range(G)] + ["surplus_core"])
    perm = ValuePermutation(
        name="isolated_cores",
        movable_values=tuple(isolated),
        relabelable_vars=tuple(core_vars),
        rationale=rel.TopologySymmetry().rationale)
    replicas = ReplicaGroup(
        name="generators",
        slots=tuple((f"gen{g}_core", "gen_pol") for g in range(G)),  # gen_pol uniform; sort by core
        rationale="the generators are interchangeable CPU-bound workers")
    sym = IRSym(permutations=(perm,), replica_groups=(replicas,), anchors=(hk,))

    # projection: the CONFIG-DEFINING vars (not the aux reifications/keys).
    projection = (["server_core", "server_pol"]
                  + [f"gen{g}_core" for g in range(G)]
                  + ["gen_pol", "surplus_present", "surplus_core", "surplus_pol"])

    provenance = {h.name: _provenance_entry(h)
                  for h in (server_pol_hp, gen_pol_hp, surplus_pol_hp, reg["surplus_present"])}

    return ConfigSpace(vars=tuple(vars_), constrs=tuple(constrs), sym=sym,
                       projection=tuple(projection), provenance=provenance)


def _combine(a: ConfigSpace, b: ConfigSpace) -> ConfigSpace:
    """Cartesian union of two disjoint-namespace ConfigSpaces (composition of surfaces)."""
    overlap = {v.id for v in a.vars} & {v.id for v in b.vars}
    if overlap:
        raise ValueError(f"cannot combine: overlapping var ids {sorted(overlap)}")
    prov = dict(a.provenance)
    prov.update(b.provenance)
    return ConfigSpace(
        vars=tuple(a.vars) + tuple(b.vars),
        constrs=tuple(a.constrs) + tuple(b.constrs),
        sym=IRSym(permutations=a.sym.permutations + b.sym.permutations,
                  replica_groups=a.sym.replica_groups + b.sym.replica_groups,
                  anchors=tuple(set(a.sym.anchors) | set(b.sym.anchors))),
        projection=tuple(a.projection) + tuple(b.projection),
        provenance=prov)
