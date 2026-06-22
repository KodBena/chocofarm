"""
tools/analysis/leaf_eval_bound/alloc/report.py
==========================================

`alloc.report` -- the allocation engine's RESULT CONTAINERS (the §2.3-E presentation
concern, lifted out of `AllocationDriver` per the responsibility-refactor §3 / the
2026-06-22 audit's F1). `PrimitiveState` is the per-input bookkeeping for one iteration;
`Recommendation` is the output of one `AllocationDriver.step()` -- the estimate, the CI,
the §4.1 kink diagnostics, the §7.D irreducible-prior-floor lines, and `report()`.

Pure presentation/data: depends only on `math` + stdlib dataclasses/typing -- it imports
NOTHING from the package (the clean §3 import DAG; `alloc.driver` imports THIS, never the
reverse), so the formatter is unit-testable on a synthetic `Recommendation` without a step.

Public Domain (The Unlicense).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class PrimitiveState:
    """Per-input bookkeeping for one iteration."""
    index: int
    name: str
    n: int                       # current pool size
    mean: float                  # mu_hat_i
    sigma: float                 # sample std of the pool
    grad: float                  # df/dx_i at current mean point
    a: float                     # (df/dx_i)^2 * sigma_i^2
    cost: float                  # c_i, cost per sample
    var_contribution: float      # a_i / n_i  (this input's share of Var(E[f]))
    recommend: int               # additional samples suggested this round


@dataclass
class Recommendation:
    """The output of one driver step().

    §6 Phase 2 adds, beside the legacy fields (all preserved — every existing reader of
    `estimate`/`var_estimate`/`ci_halfwidth`/`primitives` is unchanged): the per-`family` CI
    multiplier actually applied (`ci_multiplier` + its `ci_multiplier_label`, §4.3), and the
    `min()`-kink diagnostics (§4.1) — `kink_regime` (the binding-margin flag), the Clark-1961
    de-biased point estimate (`estimate_kink`, the `−a·φ(t)` Jensen correction), and
    `p_nonbinding_max` (the largest arg-min-flip probability `Φ(−t)` over the non-binding arms,
    the quantity the convergence guard refuses to converge on while it exceeds α)."""
    iteration: int
    converged: bool
    estimate: float              # current f(mu_hat)  (the hard-min point value)
    estimate_second_order: Optional[float]  # curvature-corrected mean, if available
    var_estimate: float          # Var(E[f]) = g^T Σ g (or Clark Var[min] in the kink regime)
    ci_halfwidth: float          # mult * sqrt(var_estimate)
    target_halfwidth: float      # requested tolerance h
    shadow_price: float          # d(cost)/dV*, marginal price of variance
    primitives: List[PrimitiveState] = field(default_factory=list)
    ci_multiplier: float = 0.0           # the multiplier actually applied (z / t_dof / conservative)
    ci_multiplier_label: str = "z"       # how it was chosen (§4.3): 'z' | 't(dof=…)' | 'conservative t(dof=…)'
    kink_regime: bool = False            # a non-binding arm is within a plausible tie (§4.1)
    estimate_kink: Optional[float] = None  # Clark-1961 de-biased E[min] (None outside the kink regime)
    p_nonbinding_max: float = 0.0        # max Φ(−t) over non-binding arms (the convergence-guard quantity)
    # §7.D — the irreducible-prior floor surfaced as its OWN line, DISTINCT from the shrinkable sampling
    # variance. `var_floor` is the sum of the declared-spread `Fixed` inputs' `a_i` (an engineering-
    # judgement prior no sampling reduces — `R_gen` σ=8, `B_op` σ=64, …); `var_shrinkable = var_estimate −
    # var_floor` is the part sampling CAN tighten. When `var_floor > V_target` the loop CANNOT meet the CI
    # target by sampling (the §2.3 honest edge) — `converged` correctly stays False (the CI honestly rests
    # on the prior), and these two lines say WHY rather than letting the loop spin against an irreducible
    # floor it cannot cross. A true-constant DEGENERATE pin is in NEITHER (it is ~0 in `var_estimate` —
    # §3 / `_assemble_sigma`); only a NORMAL declared-spread `Fixed` floors the bound.
    var_floor: float = 0.0               # Σ a_i over declared-spread Fixed inputs — the irreducible prior floor
    var_shrinkable: float = 0.0          # var_estimate − var_floor — the part sampling can reduce
    floor_blocks_target: bool = False    # var_floor > V_target: the CI target is unreachable by sampling (§2.3 edge)

    def where_to_spend(self) -> List[PrimitiveState]:
        """Primitives ranked by recommended additional samples (desc)."""
        return sorted(self.primitives, key=lambda p: p.recommend, reverse=True)

    def report(self) -> str:
        lines = []
        status = "CONVERGED" if self.converged else "continue"
        lines.append(
            f"[iter {self.iteration}] {status}  "
            f"E[f]={self.estimate:.4g}"
            + (
                f" (2nd-order {self.estimate_second_order:.4g})"
                if self.estimate_second_order is not None
                else ""
            )
            + (
                f"  [KINK: Clark E[min]={self.estimate_kink:.4g}, "
                f"P(non-binding flips)={self.p_nonbinding_max:.3g}]"
                if self.kink_regime
                else ""
            )
        )
        lines.append(
            f"  CI half-width = {self.ci_halfwidth:.4g}  "
            f"(target {self.target_halfwidth:.4g})   "
            f"shadow price lambda = {self.shadow_price:.4g}   "
            f"mult = {self.ci_multiplier:.4g} ({self.ci_multiplier_label})"
        )
        # §7.D — the irreducible-prior floor on its OWN line, distinct from the shrinkable variance, shown
        # only when a declared-spread `Fixed` prior actually floors the bound (an all-mean report is
        # visually unchanged). When the floor alone exceeds the target the line says the CI rests on the
        # prior — the honest "why it cannot converge by sampling" (§2.3 edge), not a silent spin.
        if self.var_floor > 0.0:
            floor_ci = self.ci_multiplier * math.sqrt(max(self.var_floor, 0.0))
            shrink_ci = self.ci_multiplier * math.sqrt(max(self.var_shrinkable, 0.0))
            note = ("  <- the CI rests on this prior; sampling cannot reach the target"
                    if self.floor_blocks_target else "")
            lines.append(
                f"  irreducible prior floor: var={self.var_floor:.4g} (CI {floor_ci:.4g})   "
                f"shrinkable: var={self.var_shrinkable:.4g} (CI {shrink_ci:.4g}){note}"
            )
        header = (
            f"  {'primitive':<16}{'n':>8}{'sigma':>12}"
            f"{'|df/dx|':>12}{'a_i':>12}{'a_i/n_i':>12}{'+samples':>10}"
        )
        lines.append(header)
        lines.append("  " + "-" * (len(header) - 2))
        for p in self.where_to_spend():
            lines.append(
                f"  {p.name:<16}{p.n:>8d}{p.sigma:>12.4g}"
                f"{abs(p.grad):>12.4g}{p.a:>12.4g}"
                f"{p.var_contribution:>12.4g}{p.recommend:>10d}"
            )
        return "\n".join(lines)
