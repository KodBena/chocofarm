#!/usr/bin/env python3
"""
throughput-lab/hp/backends/cpsat.py — the CP-SAT lowering, the enumerate-all-solutions driver, and
the orbit canonicalizer.

`to_cpsat(ConfigSpace)` builds an ortools CpModel mirroring the IR (DESIGN.md §3); `enumerate`
drives `enumerate_all_solutions=True`, projects each solution onto the config-defining vars, and
collapses the joint symmetry orbit by mapping each projection to its canonical key (the lex-min
image under the declared `IRSym`) — generalizing topology_enum._canonical_key (DESIGN.md §4.1
mechanism 2). The canonicalizer is the SHARED orbit invariant the grid oracle (Oracle B) and the
orbit self-check (Oracle A) reuse, so all three agree on what "the same config" means.

CP-SAT is the primary enumerator (DESIGN.md §0): native enumerate_all_solutions, a validated
reference in-tree, and no need for SMT's theory power (the whole space is finite-domain int).

Public Domain (The Unlicense).
"""
from __future__ import annotations

import itertools
from typing import Optional

from ortools.sat.python import cp_model

from ..ir import (
    AllDifferent, BoolAnd, CanonInert, CeilDiv, Clamp, ConfigSpace, Const, Derived, Expr, Implies,
    IRConstr, IRSym, IRVar, Linear, MaxEquality, Op, ReifyEq, Table, VarKind, VarRef,
)


# ==================================================================================================
# Lowering ir.ConfigSpace -> CpModel
# ==================================================================================================
def to_cpsat(cs: ConfigSpace) -> tuple[cp_model.CpModel, dict[str, cp_model.IntVar]]:
    m = cp_model.CpModel()
    handles: dict[str, cp_model.IntVar] = {}
    for v in cs.vars:
        if v.kind is VarKind.BOOL:
            handles[v.id] = m.new_bool_var(v.id)
        else:
            handles[v.id] = m.new_int_var(v.lo, v.hi, v.id)
    for c in cs.constrs:
        _lower_constr(m, handles, c)
    return m, handles


def _lit(handles: dict[str, cp_model.IntVar], name: str):
    """Resolve a literal that may be negated ('~x')."""
    if name.startswith("~"):
        return handles[name[1:]].Not()
    return handles[name]


def _expr_to_linear(handles: dict[str, cp_model.IntVar], e: Expr):
    """Evaluate an Expr to a CP-SAT linear expression (CeilDiv handled by Derived directly)."""
    if isinstance(e, VarRef):
        return handles[e.var]
    if isinstance(e, Const):
        return e.value
    raise TypeError(f"_expr_to_linear cannot handle {e!r} (CeilDiv only via Derived)")


def _lower_constr(m: cp_model.CpModel, h: dict[str, cp_model.IntVar], c: IRConstr) -> None:
    if isinstance(c, Linear):
        expr = sum(coef * h[v] for v, coef in c.coeffs.items())
        if c.op is Op.EQ:
            m.add(expr == c.rhs)
        elif c.op is Op.NE:
            m.add(expr != c.rhs)
        elif c.op is Op.LE:
            m.add(expr <= c.rhs)
        elif c.op is Op.GE:
            m.add(expr >= c.rhs)
        elif c.op is Op.LT:
            m.add(expr < c.rhs)
        else:
            raise ValueError(c.op)
    elif isinstance(c, Implies):
        # if lit then body
        lit = _lit(h, c.lit)
        body = c.body
        if isinstance(body, Linear):
            expr = sum(coef * h[v] for v, coef in body.coeffs.items())
            if body.op is Op.GE:
                m.add(expr >= body.rhs).only_enforce_if(lit)
            elif body.op is Op.EQ:
                m.add(expr == body.rhs).only_enforce_if(lit)
            elif body.op is Op.LE:
                m.add(expr <= body.rhs).only_enforce_if(lit)
            elif body.op is Op.LT:
                m.add(expr < body.rhs).only_enforce_if(lit)
            elif body.op is Op.NE:
                m.add(expr != body.rhs).only_enforce_if(lit)
            else:
                raise ValueError(body.op)
        else:
            raise NotImplementedError(f"Implies body {type(body).__name__}")
    elif isinstance(c, ReifyEq):
        b = h[c.boolvar]
        m.add(h[c.var] == c.value).only_enforce_if(b)
        m.add(h[c.var] != c.value).only_enforce_if(b.Not())
    elif isinstance(c, BoolAnd):
        members = [_lit(h, x) for x in c.members]
        t = h[c.target]
        m.add_bool_and(members).only_enforce_if(t)
        m.add_bool_or([x.Not() for x in members]).only_enforce_if(t.Not())
    elif isinstance(c, AllDifferent):
        m.add_all_different([h[v] for v in c.vars])
    elif isinstance(c, MaxEquality):
        m.add_max_equality(h[c.target], [h[v] for v in c.members])
    elif isinstance(c, Clamp):
        lo = _expr_to_linear(h, c.lo)
        hi = _expr_to_linear(h, c.hi)
        m.add(h[c.var] >= lo)
        m.add(h[c.var] <= hi)
    elif isinstance(c, Derived):
        if isinstance(c.expr, CeilDiv):
            num = _expr_to_linear(h, c.expr.num)
            den = _expr_to_linear(h, c.expr.den)
            # ceil(num/den): for constant den, target == (num + den - 1) // den. Here both may be
            # vars but in practice den is a small fixed value; we require den to be a Const-resolved
            # value range. Use add_division_equality on (num+den-1).
            if isinstance(c.expr.den, Const):
                d = c.expr.den.value
                m.add(h[c.var] * d >= num)
                m.add(h[c.var] * d < num + d)
            else:
                raise NotImplementedError("CeilDiv with variable denominator")
        else:
            m.add(h[c.var] == _expr_to_linear(h, c.expr))
    elif isinstance(c, CanonInert):
        # when `when` is FALSE, pin var to default. `when` is a lit ('~x' negates).
        when = _lit(h, c.when)
        m.add(h[c.var] == c.default_value).only_enforce_if(when.Not())
    elif isinstance(c, Table):
        m.add_allowed_assignments([h[v] for v in c.vars], list(c.allowed))
    else:
        raise TypeError(f"unknown IRConstr {c!r}")


# ==================================================================================================
# Enumerate + project + canonicalize
# ==================================================================================================
def enumerate_raw(cs: ConfigSpace) -> list[dict[str, int]]:
    """Every feasible PROJECTION (config-defining vars only), before orbit canonicalization."""
    m, h = to_cpsat(cs)
    solver = cp_model.CpSolver()
    solver.parameters.enumerate_all_solutions = True
    proj = list(cs.projection)
    out: list[dict[str, int]] = []

    class _Collector(cp_model.CpSolverSolutionCallback):
        def __init__(self):
            super().__init__()

        def on_solution_callback(self):
            out.append({v: int(self.value(h[v])) for v in proj})

    solver.solve(m, _Collector())
    return out


def enumerate_configs(cs: ConfigSpace) -> list[dict[str, int]]:
    """The symmetry-reduced config set: one representative per orbit, in canonical-key order."""
    raw = enumerate_raw(cs)
    by_key: dict[tuple, dict[str, int]] = {}
    for rec in raw:
        key = canonical_key(rec, cs)
        if key not in by_key:
            by_key[key] = rec
    return [by_key[k] for k in sorted(by_key.keys())]


# ==================================================================================================
# The canonicalizer — the orbit invariant (DESIGN.md §4.1 mechanism 2), shared by both oracles.
# It maps a projection dict to the lex-min image under the declared IRSym: relabel the movable
# core values under each permutation, AND re-sort the interchangeable replica slots. The min over
# all permutations is the canonical form. For disjoint-acting factors this is sound; Oracle B is the
# backstop for any factor interaction (DESIGN.md §4.1 caveat).
# ==================================================================================================
def canonical_key(rec: dict[str, int], cs: ConfigSpace) -> tuple:
    sym = cs.sym
    if not sym.permutations and not sym.replica_groups:
        # no symmetry: the projection itself (in stable order) is the key.
        return tuple(sorted(rec.items()))

    proj = list(cs.projection)
    best: Optional[tuple] = None

    # the set of values each permutation moves; the anchors are never moved.
    perms = sym.permutations
    movable_for: list[tuple[int, ...]] = [pm.movable_values for pm in perms]
    relabel_targets: list[frozenset[str]] = [frozenset(pm.relabelable_vars) for pm in perms]

    # generate the Cartesian product of per-permutation value-relabelings.
    perm_choices = [list(itertools.permutations(mv)) for mv in movable_for]
    if not perm_choices:
        perm_choices = [[()]]

    for combo in itertools.product(*perm_choices):
        relabeled = dict(rec)
        for pi, sigma in enumerate(combo):
            if not sigma:
                continue
            mapping = dict(zip(movable_for[pi], sigma))
            for var in relabel_targets[pi]:
                if var in relabeled:
                    relabeled[var] = mapping.get(relabeled[var], relabeled[var])
        img = _replica_canonical_image(relabeled, cs)
        if best is None or img < best:
            best = img
    assert best is not None
    return best


def _replica_canonical_image(rec: dict[str, int], cs: ConfigSpace) -> tuple:
    """Quotient out the interchangeable replica groups by SORTING their slots by packed value, then
    render the projection as a stable tuple. Replica field vars are emitted as a sorted multiset so
    any slot permutation maps to the same image."""
    proj = list(cs.projection)
    replica_field_vars: set[str] = set()
    slot_renders: list[tuple] = []
    for rg in cs.sym.replica_groups:
        slots = []
        for slot in rg.slots:
            slots.append(tuple(rec[v] for v in slot))
            replica_field_vars.update(slot)
        slots.sort()
        slot_renders.append((rg.name, tuple(slots)))

    # the non-replica projection vars, in stable order.
    rest = tuple((v, rec[v]) for v in proj if v not in replica_field_vars)
    return (rest, tuple(slot_renders))
