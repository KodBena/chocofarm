#!/usr/bin/env python3
"""
throughput-lab/hp/backends/grid.py — the itertools oracle (Oracle B, DESIGN.md §4.2).

`to_grid(ConfigSpace)` independently generates the FULL feasible set via itertools.product over the
FREE vars + an IMPERATIVE feasibility filter — a DIFFERENT implementation of the same `constrs`
than backends/cpsat.py's declarative CP-SAT encoding. AUX vars (occupancy/equality reifications,
derived dims, the AND/MAX indicators) are FUNCTIONALLY DETERMINED by the free vars, so the oracle
*computes* them (it does not product over them — that would be exponential) and then checks the
remaining hard constraints. Quotienting this set by the SAME canonicalizer (cpsat.canonical_key)
and comparing to CP-SAT's enumeration catches over-collapse, the per-factor-canonicalizer
interaction bug, and any feasibility-encoding mismatch (DESIGN.md §0/§4.2).

This shares NO code path with CP-SAT — a genuine differential oracle (itertools, not a 2nd solver).

Public Domain (The Unlicense).
"""
from __future__ import annotations

import itertools
from typing import Iterator

from ..ir import (
    AllDifferent, BoolAnd, CanonInert, CeilDiv, Clamp, ConfigSpace, Const, Derived, Expr, Implies,
    IRConstr, IRVar, Linear, MaxEquality, Op, ReifyEq, Table, VarKind, VarRef,
)


def _eval_expr(e: Expr, assign: dict[str, int]) -> int:
    if isinstance(e, VarRef):
        return assign[e.var]
    if isinstance(e, Const):
        return e.value
    if isinstance(e, CeilDiv):
        return -(-_eval_expr(e.num, assign) // _eval_expr(e.den, assign))
    raise TypeError(f"unknown Expr {e!r}")


def _lit_val(name: str, assign: dict[str, int]) -> bool:
    if name.startswith("~"):
        return assign[name[1:]] == 0
    return assign[name] == 1


def _linear_holds(c: Linear, assign: dict[str, int]) -> bool:
    lhs = sum(coef * assign[v] for v, coef in c.coeffs.items())
    return {Op.EQ: lhs == c.rhs, Op.NE: lhs != c.rhs, Op.LE: lhs <= c.rhs,
            Op.GE: lhs >= c.rhs, Op.LT: lhs < c.rhs}[c.op]


def _constr_holds(c: IRConstr, assign: dict[str, int]) -> bool:
    if isinstance(c, Linear):
        return _linear_holds(c, assign)
    if isinstance(c, Implies):
        return (not _lit_val(c.lit, assign)) or _constr_holds(c.body, assign)
    if isinstance(c, ReifyEq):
        return assign[c.boolvar] == (1 if assign[c.var] == c.value else 0)
    if isinstance(c, BoolAnd):
        return assign[c.target] == (1 if all(_lit_val(x, assign) for x in c.members) else 0)
    if isinstance(c, AllDifferent):
        vals = [assign[v] for v in c.vars]
        return len(set(vals)) == len(vals)
    if isinstance(c, MaxEquality):
        return assign[c.target] == max(assign[v] for v in c.members)
    if isinstance(c, Clamp):
        return _eval_expr(c.lo, assign) <= assign[c.var] <= _eval_expr(c.hi, assign)
    if isinstance(c, Derived):
        return assign[c.var] == _eval_expr(c.expr, assign)
    if isinstance(c, CanonInert):
        return _lit_val(c.when, assign) or assign[c.var] == c.default_value
    if isinstance(c, Table):
        return tuple(assign[v] for v in c.vars) in set(c.allowed)
    raise TypeError(f"unknown IRConstr {c!r}")


def _expr_deps(e: Expr) -> list[str]:
    if isinstance(e, VarRef):
        return [e.var]
    if isinstance(e, Const):
        return []
    if isinstance(e, CeilDiv):
        return _expr_deps(e.num) + _expr_deps(e.den)
    return []


def _try_determine(c: IRConstr, assign: dict[str, int]) -> bool:
    """If `c` can compute exactly one currently-UNKNOWN var from known ones, set it and return True.
    Covers the determining constraint kinds AND single-unknown Linear-EQ / Table — so the oracle
    products ONLY over the genuinely-free vars (the projection) and derives all the rest.

    This is an INDEPENDENT solver from CP-SAT (a simple imperative fixpoint), so it remains a
    differential check (DESIGN.md §4.2)."""
    if isinstance(c, ReifyEq):
        if c.boolvar not in assign and c.var in assign:
            assign[c.boolvar] = 1 if assign[c.var] == c.value else 0
            return True
    elif isinstance(c, BoolAnd):
        deps = [x.lstrip("~") for x in c.members]
        if c.target not in assign and all(d in assign for d in deps):
            assign[c.target] = 1 if all(_lit_val(x, assign) for x in c.members) else 0
            return True
    elif isinstance(c, MaxEquality):
        if c.target not in assign and all(d in assign for d in c.members):
            assign[c.target] = max(assign[v] for v in c.members)
            return True
    elif isinstance(c, Derived):
        if c.var not in assign and all(d in assign for d in _expr_deps(c.expr)):
            assign[c.var] = _eval_expr(c.expr, assign)
            return True
    elif isinstance(c, Linear) and c.op is Op.EQ:
        unknown = [v for v in c.coeffs if v not in assign]
        if len(unknown) == 1 and abs(c.coeffs[unknown[0]]) == 1:
            u = unknown[0]
            known = sum(c.coeffs[v] * assign[v] for v in c.coeffs if v != u)
            assign[u] = (c.rhs - known) // c.coeffs[u]
            return True
    elif isinstance(c, Table):
        unknown = [v for v in c.vars if v not in assign]
        if len(unknown) == 1:
            u = unknown[0]
            ui = c.vars.index(u)
            matches = {row[ui] for row in c.allowed
                       if all(assign[v] == row[i] for i, v in enumerate(c.vars) if v != u)}
            if len(matches) == 1:
                assign[u] = next(iter(matches))
                return True
    return False


def to_grid(cs: ConfigSpace) -> Iterator[dict[str, int]]:
    """Yield every feasible FULL assignment. Products ONLY over the config-defining (projection)
    vars; derives every determinable var via a fixpoint; products over any residual free var; then
    applies the hard feasibility filter (ALL constraints — an independent re-check)."""
    proj = set(cs.projection)
    # seed-free vars: projection vars are always producted; any non-projection var we cannot derive
    # is also producted (residual).
    base_product = [v for v in cs.vars if v.id in proj]

    domains = [list(range(v.lo, v.hi + 1)) for v in base_product]
    names = [v.id for v in base_product]
    all_ids = {v.id for v in cs.vars}

    for combo in itertools.product(*domains):
        assign = dict(zip(names, combo))
        # fixpoint: derive everything determinable from the projection assignment.
        changed = True
        while changed:
            changed = False
            for c in cs.constrs:
                if _try_determine(c, assign):
                    changed = True
        residual = [v for v in cs.vars if v.id not in assign]
        if residual:
            # product over any residual (rare; small bool domains).
            rnames = [v.id for v in residual]
            rdoms = [list(range(v.lo, v.hi + 1)) for v in residual]
            for rcombo in itertools.product(*rdoms):
                full = dict(assign)
                full.update(zip(rnames, rcombo))
                if all(_constr_holds(c, full) for c in cs.constrs):
                    yield full
        else:
            if all(_constr_holds(c, assign) for c in cs.constrs):
                yield assign


def feasible_projections(cs: ConfigSpace) -> list[dict[str, int]]:
    proj = list(cs.projection)
    return [{v: assign[v] for v in proj} for assign in to_grid(cs)]
