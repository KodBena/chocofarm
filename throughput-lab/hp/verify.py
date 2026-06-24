#!/usr/bin/env python3
"""
throughput-lab/hp/verify.py — the two independent oracles + the inertness self-check (DESIGN.md §4.2).

All fail loud (ADR-0002): any divergence returns (ok=False, message) and the CLI turns that into a
non-zero exit, refusing to emit — exactly topology_enum.py's `--verify` returning 3.

  - Oracle A (orbit non-isomorphism, the verify_orbits pattern generalized): brute-force recompute
    each emitted config's orbit under IRSym WITHOUT trusting canonical_key; assert the emitted set is
    pairwise non-isomorphic / orbit-disjoint. Catches UNDER-collapse.
  - Oracle B (the grid cross-check): independently generate the full feasible set via the itertools
    oracle (a different feasibility implementation), quotient it by the SAME canonicalizer, and
    assert its canonical-rep set equals CP-SAT's emitted set. Catches OVER-collapse, the per-factor
    canonicalizer interaction bug, and any encoding mismatch between the two backends.
  - Inertness self-check: assert that configs differing only in a guard-FALSE axis canonicalize to
    the same emitted config — i.e. the CanonInert collapses actually happened.

Public Domain (The Unlicense).
"""
from __future__ import annotations

import itertools
from typing import Optional

from .backends import cpsat, grid
from .ir import CanonInert, ConfigSpace


def _proj_tuple(rec: dict[str, int], cs: ConfigSpace) -> tuple:
    return tuple((v, rec[v]) for v in cs.projection)


# ==================================================================================================
# Oracle A — orbit non-isomorphism (does NOT trust canonical_key; recomputes the orbit by brute
# force over the declared permutations + replica relabelings, like verify_orbits).
# ==================================================================================================
def _orbit(rec: dict[str, int], cs: ConfigSpace) -> set[tuple]:
    """The full orbit of a projection under IRSym (every permuted+replica-sorted image), as a set of
    canonical-form tuples computed independently of canonical_key (it reuses the same group
    DEFINITION but enumerates the orbit rather than taking a min — so it cross-checks the result)."""
    sym = cs.sym
    proj = list(cs.projection)
    images: set[tuple] = set()

    perms = sym.permutations
    movable_for = [pm.movable_values for pm in perms]
    relabel_targets = [frozenset(pm.relabelable_vars) for pm in perms]
    perm_choices = [list(itertools.permutations(mv)) for mv in movable_for] or [[()]]

    for combo in itertools.product(*perm_choices):
        relabeled = dict(rec)
        for pi, sigma in enumerate(combo):
            if not sigma:
                continue
            mapping = dict(zip(movable_for[pi], sigma))
            for var in relabel_targets[pi]:
                if var in relabeled:
                    relabeled[var] = mapping.get(relabeled[var], relabeled[var])
        # replica-sorted image (independent re-implementation of the multiset render)
        replica_field_vars: set[str] = set()
        slot_renders = []
        for rg in sym.replica_groups:
            slots = sorted(tuple(relabeled[v] for v in slot) for slot in rg.slots)
            for slot in rg.slots:
                replica_field_vars.update(slot)
            slot_renders.append((rg.name, tuple(slots)))
        rest = tuple((v, relabeled[v]) for v in proj if v not in replica_field_vars)
        images.add((rest, tuple(slot_renders)))
    return images


def oracle_a(emitted: list[dict[str, int]], cs: ConfigSpace) -> tuple[bool, str]:
    seen_rep: dict[tuple, int] = {}     # an orbit image -> index of the emitted config owning it
    collisions: list[tuple[int, int]] = []
    for i, rec in enumerate(emitted):
        orb = _orbit(rec, cs)
        hit = next((seen_rep[img] for img in orb if img in seen_rep), None)
        if hit is not None:
            collisions.append((hit, i))
        else:
            for img in orb:
                seen_rep[img] = i
    n_orbits = len(set(seen_rep.values()))
    ok = not collisions and n_orbits == len(emitted)
    msg = (f"Oracle A (orbit non-isomorphism): {len(emitted)} configs, {n_orbits} distinct orbits, "
           f"{len(collisions)} under-collapse collision(s)")
    if collisions:
        msg += " -> FAIL: " + ", ".join(f"#{a}~#{b}" for a, b in collisions[:5])
    return ok, msg


# ==================================================================================================
# Oracle B — the grid cross-check (independent feasibility + the SAME canonicalizer).
# ==================================================================================================
def oracle_b(emitted: list[dict[str, int]], cs: ConfigSpace) -> tuple[bool, str]:
    # canonical-rep set from CP-SAT's emitted configs.
    cpsat_keys = {cpsat.canonical_key(rec, cs) for rec in emitted}
    # canonical-rep set from the independent itertools feasibility filter.
    grid_keys: set[tuple] = set()
    for rec in grid.feasible_projections(cs):
        grid_keys.add(cpsat.canonical_key(rec, cs))
    missing = grid_keys - cpsat_keys   # CP-SAT dropped a real candidate (OVER-collapse)
    extra = cpsat_keys - grid_keys     # CP-SAT emitted something the grid says is infeasible
    ok = not missing and not extra
    msg = (f"Oracle B (grid cross-check): cpsat={len(cpsat_keys)} grid={len(grid_keys)} "
           f"canonical reps")
    if missing:
        msg += f" -> FAIL: {len(missing)} config(s) the grid found but CP-SAT dropped (over-collapse)"
    if extra:
        msg += f" -> FAIL: {len(extra)} config(s) CP-SAT emitted but the grid rejects (encoding mismatch)"
    return ok, msg


# ==================================================================================================
# Inertness self-check — the CanonInert collapses the activation types generate actually happened.
# For each CanonInert(var, default, ~aux): in every emitted config where the gate is unsatisfied,
# `var` must equal `default`. (If the collapse did NOT happen, two configs differing only in `var`
# would both survive — caught by Oracle B as over/under, but this names it directly.)
# ==================================================================================================
def inertness_check(emitted: list[dict[str, int]], cs: ConfigSpace) -> tuple[bool, str]:
    canon_inerts = [c for c in cs.constrs if isinstance(c, CanonInert)]
    if not canon_inerts:
        return True, "inertness self-check: no CanonInert nodes (n/a)"
    # The CanonInert `when` literal may be an AUX bool not carried in the projection (the activation
    # AND-bool). We reconstruct the FULL assignment from each emitted projection via the SAME
    # independent fixpoint the grid oracle uses (a different implementation from CP-SAT), then check
    # each CanonInert directly: when the guard is FALSE, the var MUST equal its default. This proves
    # the activation-gate collapse the SSOT generates from the activation types actually happened.
    violations: list[str] = []
    for rec in emitted:
        full = _reconstruct_full(rec, cs)
        if full is None:
            violations.append(f"could not reconstruct aux vars for {rec}")
            continue
        for c in canon_inerts:
            inert = not grid._lit_val(c.when, full)   # when FALSE => inert
            if inert and full.get(c.var) != c.default_value:
                violations.append(
                    f"{c.var}={full.get(c.var)} (expected default {c.default_value}) under inert "
                    f"gate {c.when}")
    ok = not violations
    msg = f"inertness self-check: {len(canon_inerts)} CanonInert node(s), {len(violations)} violation(s)"
    if violations:
        msg += " -> FAIL: " + "; ".join(violations[:5])
    return ok, msg


def _reconstruct_full(rec: dict[str, int], cs: ConfigSpace) -> Optional[dict[str, int]]:
    """Derive every aux var from a projection via the grid's independent fixpoint determination."""
    full = dict(rec)
    changed = True
    while changed:
        changed = False
        for c in cs.constrs:
            if grid._try_determine(c, full):
                changed = True
    # everything determined?
    if all(v.id in full for v in cs.vars):
        return full
    return None


def verify_all(cs: ConfigSpace, emitted: Optional[list[dict[str, int]]] = None
               ) -> tuple[bool, list[str]]:
    if emitted is None:
        emitted = cpsat.enumerate_configs(cs)
    msgs: list[str] = []
    ok = True
    for fn in (oracle_a, oracle_b, inertness_check):
        o, m = fn(emitted, cs)
        ok = ok and o
        msgs.append(m)
    return ok, msgs
