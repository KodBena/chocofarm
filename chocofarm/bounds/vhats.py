#!/usr/bin/env python3
"""
vhats.py — the no-heavy-deps V̂ value-function strategies for the dual bound, plus the
Vhat Protocol they all satisfy (chocofarm/bounds/info_relaxation.py; design+proofs
docs/design/dual-bound.md).

A V̂ estimates the fixed-λ value-to-go E[ΣR − λΣT | state] of near-optimal
continuation. ANY V̂ yields a VALID bound (dual feasibility is automatic); a good V̂
yields a TIGHT one. The penalty / inner solve (PenalizedClairvoyant) treat V̂ as an
injected STRATEGY: a (belief, λ) → value callable, invoked as
`vhat(env, loc, bw, collected, lam) -> float`.

This module holds the two strategies with no heavy dependencies — `vhat_zero` and
`vhat_analytic`. The heavier ones live beside their dependency: `DecompVhat` in
`vhats_decomp.py` (needs chocofarm.solvers.decomp) and `ExactBeliefVhat` in
`vhats_exact.py` (the belief-semilattice enumeration). Splitting BY DEPENDENCY is why
`info_relaxation` no longer carries a lazy `import decomp`: a bounds user pulls in only
the V̂ they actually pass.

Public Domain (The Unlicense).
"""
from __future__ import annotations

from typing import TYPE_CHECKING, AbstractSet, Protocol

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from chocofarm.model.env import Environment, Loc


class Vhat(Protocol):
    """The V̂ strategy interface: a (belief, λ) → value approximation of the fixed-λ
    value-to-go. PenalizedClairvoyant injects one and calls it as

        vhat(env, loc, bw, collected, lam) -> float

    `loc` is the current location key (("w"|"t"|"d", id)), `bw` the belief world-set
    array, `collected` the set/frozenset of already-collected treasure ids, `lam` the
    fixed reference λ. Both the plain-function strategies (vhat_zero, vhat_analytic) and
    the class strategies (DecompVhat, ExactBeliefVhat — their `__call__`) satisfy this.

    ADR-0002 honest-signature fix (Stage 3): `loc` is typed as the seam `Loc` alias
    (env.py), NOT the old `Tuple[str, int]`. The latter was a lying signature — the
    `("w", entry)` start location carries a STR id (the teleport name), so a real call
    site passes a tuple `Tuple[str, int]` cannot spell, and every body forwards `loc`
    straight into `env.exit_cost`/`env.d`, which consume `Loc`. `Loc` is the type the
    callers (info_relaxation, eval_bound) actually pass and the env actually consumes."""

    def __call__(self, env: Environment, loc: Loc,
                 bw: npt.NDArray[np.int64], collected: AbstractSet[int],
                 lam: float) -> float:
        ...


def vhat_zero(env: "Environment", loc: "Loc", bw: npt.NDArray[np.int64],
              collected: AbstractSet[int], lam: float) -> float:
    """V̂ ≡ 0 — but NOTE this is NOT the z≡0 clairvoyant baseline. With V̂≡0 the
    value-function penalty is z_t = r_t − E[r_t | F_t, a_t] (the REWARD-DEVIATION
    martingale), which is dual-feasible and nonzero. It is a (mild) valid penalty, not
    the pure relaxation. The TRUE z≡0 regression baseline is `vhat=None` (the
    no-penalty mode in PenalizedClairvoyant), which uses the realized r − λ·dt and
    reproduces clairvoyant_rate exactly. Kept only as a curiosity / extra valid V̂."""
    return 0.0


def vhat_analytic(env: "Environment", loc: "Loc", bw: npt.NDArray[np.int64],
                  collected: AbstractSet[int], lam: float) -> float:
    """Trivial analytic V̂₀ (sanity baseline, dual-bound.md §2.4(1)): expected
    still-collectable reward if grabbable for free, minus the cost to leave.

        V̂₀ = Σ_i marginals(b)[i]·value[i]·1[i∉c]  −  λ·exit_cost(loc)

    Crude but a genuine value estimate; it makes the penalty CHARGE for resolving
    marginals, so B(λ, V̂₀) is a valid bound that should sit modestly below 0.1454."""
    if len(bw) == 0:
        return -lam * env.exit_cost(loc)
    marg = env.marginals(bw)
    # `marg[i]` is a numpy float64 (marginals returns NDArray[float64]); `sum(...)` over
    # them yields a numpy scalar, so the difference is np.float64-typed (Any-leakage under
    # --strict). float(...) states the declared float contract — a lossless scalar unwrap,
    # not a behavior change (np.float64 == its python-float value).
    er = sum(marg[i] * env.value[i] for i in range(env.N) if i not in collected)
    return float(er - lam * env.exit_cost(loc))
