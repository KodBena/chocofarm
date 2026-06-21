"""
tools/analysis/OpenTURNS/neyman_driver.py
=========================================

The GENERIC, model-agnostic Neyman optimal-allocation driver — a benchmark-budget
allocator for functional uncertainty-propagation models. It owns NO specific model
(ADR-0012 P1 single-home / P2 separation of the allocator-transport from the thing
allocated): a concrete throughput model is a SEPARATE module (e.g.
`examples/demo_msgpass.py`, `model_capacity.py`, `model_cycletime.py`) that builds an
`ot.Function` and hands it to `NeymanDriver`. The synthetic demo that previously lived
in this file (the `_demo()` impurity) was extracted to
`tools/analysis/OpenTURNS/examples/demo_msgpass.py` per the ADR-0012 purification.

Public Domain (The Unlicense).

A benchmark-budget allocator for functional uncertainty-propagation models.

You have a deterministic expression  y = f(x_1, ..., x_d)  (e.g. a crude
throughput model) whose inputs x_i are physical quantities you can *sample*
from a real system at some per-sample cost c_i (message-passing latency,
context-switch cost, inference throughput, ...). You estimate E[f] by f(mu_hat),
where each input mean mu_hat_i is the average of n_i collected samples.

The variance of that estimate, by the delta method (inputs independent), is

        Var(f(mu_hat)) ~= sum_i (df/dx_i)^2 * sigma_i^2 / n_i
                        =  sum_i a_i / n_i,      a_i := (df/dx_i)^2 sigma_i^2.

Minimising total cost  sum_i c_i n_i  subject to  sum_i a_i / n_i <= V*
gives Neyman allocation:

        n_i*  proportional to  sqrt(a_i / c_i),

scaled to hit V*. Because a_i depends on sigma_i and the gradient at mu --
which you only know after sampling -- the procedure is iterative: pilot,
estimate a_i, allocate a top-up, repeat. This module runs that loop.

OpenTURNS supplies f and its gradient (with a finite-difference fallback), and
an optional TaylorExpansionMoments diagnostic flags when the linearisation the
allocation rests on is no longer trustworthy (Jensen / curvature).

Dependencies: numpy, openturns (>=1.17 or so). TaylorExpansionMoments and the
JointDistribution/ComposedDistribution naming are version-sensitive and are
both guarded.

Author's note: this allocates effort to shrink the CI on E[f] (the expected
ceiling). It does NOT shrink sigma_f = sqrt(sum_i a_i), the genuine run-to-run
spread of f itself, which is physical and irreducible by input sampling.

§6 PHASE 2 — THE DRIVER CONSUMES THE `Estimate` (the behavior-CHANGING phase;
docs/design/harmonized-estimator-interface.md §2/§4/§6). The diagonal exposition
above is now the SPECIAL CASE; the driver generalizes it, additively, so an
all-mean / diagonal model is unchanged (the no-regression fixed point):

  * DUAL-MODE INPUT. `set_estimate(i, Estimate)` / `set_estimates_by_name` BESIDE
    `add_samples`. `step()` PREFERS the Estimate (reading its already-divided
    sampling variance off `cov`, the eval point off `theta_hat`, the CI
    multiplier off `family`, cross-input coupling off `cross`); else it WRAPS the
    raw pool as a `Poolwise` Estimate — so a pool-fed and an Estimate-fed driver
    AGREE on the mean case. `run()` takes `measurers[i](budget) -> Estimate`
    (the Phase-2 form) OR the legacy `samplers[i](k) -> array`.
  * THE QUADRATIC FORM (§2.2). `Var(E[f]) = gᵀΣg` replaces `sum a_i/n_i`. Σ is
    block-diagonal across inputs, carrying a within-/cross-bench off-diagonal an
    `Estimate.cross` declares (the slope/intercept −0.81). Equals the old sum
    bit-for-bit on a diagonal Σ.
  * THE ALLOCATION (§2.3). The cost-constrained c-optimal SOCP (the sign-safe
    `Q = diag(g)·R·diag(g)` form, CLARABEL with an SCS retry) — reduces to the
    closed form `n_i* ∝ √(a_i/c_i)` on the diagonal (where the driver uses the
    closed form directly: exact, sign-safe, and robust to the scaling a numerical
    SOCP chokes on), and on a NON-diagonal Σ hits `gᵀΣ(n*)g = V*` exactly with a
    fail-loud `gᵀΣ(n*)g ≈ V*` assertion (a solver `optimal` status does NOT catch
    a mixed-sign sign-fold — the §8 correction-3 trap, which `model_capacity` has).
    FUNDABILITY is the typed D2 marginal (`ShrinkLaw.marginal_dvar_deffort`, §1 D2),
    NOT `A_i = Σ_ii·len(self.pools[i])` (the prior conflation that funded a fit as
    if 1/n pool-shrinkage applied regardless of its `ShrinkLaw`): the per-sample
    `A_i = −marginal·n_eff²` recovers `Σ_ii·n_eff` byte-for-byte for a 1/n law
    (Poolwise mean, QuantileLaw median — the mean case unchanged), while a Fixed
    pin (marginal=0) AND a leverage/misfit-floored `RegressionLaw` fit (marginal≈0,
    more iters never cross `1/Sxx`) are un-fundable and drop out (`_fundability`,
    the one home the §2.3 SOCP and the forward-progress nudge share).
  * THE min()-KINK (§4.1). When a model exposes its min() arms (via `self.arms_fn`)
    and a non-binding arm is within a plausible tie, the driver enters `kink_regime`:
    it replaces `gᵀΣg` with the Clark-1961 closed-form `E[min]`/`Var[min]`
    (deterministic, O(1), NO Monte-Carlo), funds BOTH contending arms by the
    `Φ(±t)` criticality weights, and REFUSES convergence while the arg-min-flip
    probability `Φ(−t) > α`. Each arm's σ is propagated from the Estimate covs.
  * THE CI MULTIPLIER (§4.3). Per-family: NORMAL→z, STUDENT_T(dof)→t_dof,
    mixed→the most-conservative t (labelled conservative); DEGENERATE pins add none.

Dependencies grow by `estimate` (the contract; numpy-only), `cvxpy` (CLARABEL,
the non-diagonal SOCP), and `scipy.stats` (the Clark Φ/φ) — all lazily imported
so the diagonal/closed-form path runs without cvxpy and the import stays cheap.
"""

from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

try:
    import openturns as ot
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "neyman_driver requires openturns: pip install openturns"
    ) from exc

# estimate.py is the harmonized-estimator TYPE SSOT (the `Estimate` contract + ShrinkLaw /
# Support / CIFamily). §6 Phase 2 consumes it: the driver now accepts one `Estimate` per input
# beside the raw pool (set_estimate), reads its already-divided sampling variance off `cov`, and
# wraps a raw pool as a `Poolwise` Estimate so a pool-fed and an Estimate-fed driver AGREE on the
# mean case (the confirmed fixed point). It lives in this directory (no package), imported by
# sys.path the way manifest.py imports it — so adding our directory keeps the import working
# whether the driver is imported as a sibling module or from the repo root.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import estimate as _est  # noqa: E402  — the Estimate contract (numpy-only; touches no DB)

# cvxpy (CLARABEL) backs the §2.3 cost-constrained c-optimal SOCP allocation (the sign-safe
# Q = diag(g)·R·diag(g) form, native to mixed-sign gradients). scipy.stats.norm backs the §4.1
# Clark-1961 closed-form min-moments (Φ, φ — deterministic, no Monte-Carlo). Both are imported
# lazily inside the methods that use them so importing this module stays cheap and so a host
# without cvxpy still runs the diagonal/closed-form path (which needs neither) — the SOCP and the
# Clark path raise loudly (ADR-0002) if their library is genuinely absent when they are reached.


# --------------------------------------------------------------------------- #
# Result containers
# --------------------------------------------------------------------------- #
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


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
class NeymanDriver:
    """
    Iterative optimal-allocation driver for an OpenTURNS scalar function f.

    Parameters
    ----------
    f : ot.Function
        Scalar-output function of d inputs. Built however you like
        (SymbolicFunction, PythonFunction, composite). Its .gradient() is used
        if available; otherwise central finite differences on f are used.
    costs : sequence of float
        Per-sample cost c_i of benchmarking each primitive. Relative scale is
        all that matters. Use wall-clock seconds, or 1.0 for all if uniform.
    tolerance : float
        Target CI half-width h on E[f]. The loop stops when z*sqrt(Var(E[f])) <= h.
    names : sequence of str, optional
        Human-readable primitive names (defaults to x0, x1, ...).
    confidence : float
        Two-sided confidence level for the CI (default 0.95 -> z=1.96).
    growth_cap : float
        Max multiplicative growth of any pool per round (damping; default 3.0).
    max_batch : int, optional
        Hard cap on additional samples per primitive per round.
    fd_rel_step : float
        Relative step for the finite-difference gradient fallback.
    """

    def __init__(
        self,
        f: "ot.Function",
        costs: Sequence[float],
        tolerance: float,
        names: Optional[Sequence[str]] = None,
        confidence: float = 0.95,
        growth_cap: float = 3.0,
        max_batch: Optional[int] = None,
        fd_rel_step: float = 1e-5,
    ):
        self.f = f
        self.d = f.getInputDimension()
        if f.getOutputDimension() != 1:
            raise ValueError("f must have scalar output (output dimension 1).")
        self.costs = np.asarray(costs, dtype=float)
        if self.costs.shape != (self.d,):
            raise ValueError(f"costs must have length {self.d}.")
        if np.any(self.costs <= 0):
            raise ValueError("costs must be positive.")
        self.tolerance = float(tolerance)
        self.names = list(names) if names is not None else [f"x{i}" for i in range(self.d)]
        self.confidence = float(confidence)
        self.z = float(_z_from_confidence(confidence))
        self.growth_cap = float(growth_cap)
        self.max_batch = max_batch
        self.fd_rel_step = float(fd_rel_step)

        # pools[i] is a 1-D numpy array of collected samples for primitive i (the legacy raw-pool
        # path; add_samples appends to it). estimates[i] is the §6 Phase-2 harmonized Estimate for
        # input i when set via set_estimate — preferred over the pool in step(). When neither is set
        # a pilot is required (the >=2-sample gate). When only the pool is set, step() wraps it as a
        # Poolwise Estimate, so the pool-fed and Estimate-fed paths agree on the mean (the fixed point).
        self.pools: List[np.ndarray] = [np.empty(0) for _ in range(self.d)]
        self.estimates: List[Optional["_est.Estimate"]] = [None for _ in range(self.d)]
        self.iteration = 0

    # ------------------------------------------------------------------ #
    # Pool management
    # ------------------------------------------------------------------ #
    def add_samples(self, new: Dict[int, np.ndarray]) -> None:
        """Merge freshly benchmarked samples. Keys are primitive indices."""
        for i, vals in new.items():
            vals = np.asarray(vals, dtype=float).ravel()
            self.pools[i] = np.concatenate([self.pools[i], vals])

    def add_samples_by_name(self, new: Dict[str, np.ndarray]) -> None:
        idx = {name: i for i, name in enumerate(self.names)}
        self.add_samples({idx[k]: v for k, v in new.items()})

    # ------------------------------------------------------------------ #
    # §6 Phase 2 — the Estimate seam: set one harmonized Estimate per input (BESIDE add_samples).
    # ------------------------------------------------------------------ #
    def set_estimate(self, i: int, est: "_est.Estimate") -> None:
        """Set the harmonized `Estimate` for input `i` (the §6 Phase-2 input path, beside the legacy
        `add_samples`). `step()` PREFERS this over the raw pool: it reads the already-divided sampling
        variance off `est.cov`, the evaluation point off `est.theta_hat`, the CI multiplier off
        `est.family`, and any cross-input coupling off `est.cross` (§2.2/§4.3). The Estimate is
        validated at construction (ADR-0002 — `estimate.py`'s `__post_init__` gate); we re-assert
        `is_valid()` here so a value that reached us through a ctor-bypassing path (a hand-built mock,
        a deserialization) fails loudly at the seam rather than producing a malformed bound. A
        multi-component (k>1) Estimate is accepted and its FIRST component is the input's marginal
        (theta_hat[0], cov[0,0]); the within-fit off-diagonal of a k=2 fit is carried by PAIRING two
        inputs through `cross` (Phase 3's bench work — §4.2), not by a single input owning two
        components, because this driver is keyed by one scalar model input per index."""
        if not (0 <= i < self.d):
            raise IndexError(f"set_estimate: input index {i} out of range [0, {self.d}).")
        if not isinstance(est, _est.Estimate):
            raise TypeError(
                f"set_estimate({i}): expected an estimate.Estimate; got {type(est).__name__} "
                f"(ADR-0002: the driver consumes only the typed contract, never a bespoke dict).")
        if not est.is_valid():
            raise ValueError(
                f"set_estimate({i}, name={self.names[i]!r}): the Estimate fails its is_valid() gate — "
                f"refusing a malformed estimate at the seam (ADR-0002 / P2 reject-don't-coerce).")
        self.estimates[i] = est

    def set_estimates_by_name(self, new: Dict[str, "_est.Estimate"]) -> None:
        """`set_estimate` keyed by registry/input name (the form a model wires from `manifest.estimate`
        — `{model_input_name: Estimate}`). An unknown name is a loud KeyError (ADR-0002)."""
        idx = {name: i for i, name in enumerate(self.names)}
        for nm, est in new.items():
            if nm not in idx:
                raise KeyError(
                    f"set_estimates_by_name: {nm!r} is not one of this driver's inputs {self.names}.")
            self.set_estimate(idx[nm], est)

    def _estimate_for(self, i: int) -> "_est.Estimate":
        """The Estimate the loop uses for input `i`: the one set via `set_estimate` if present, else the
        raw pool WRAPPED as a k=1 `Poolwise` Estimate (mean → theta_hat, s²/n → cov, s² → per_sample_var,
        family NORMAL). The wrap is the EXACT same `(mean, s²/n, s²)` the legacy `step()` recombined as
        `a_i/n_i = g_i²·s_i²/n_i`, so a pool-fed driver and an Estimate-fed driver produce byte-for-byte
        the same `g^T Σ g` on the all-means case (the confirmed fixed point; the §2.2 exactness claim).
        Requires a pilot of ≥2 samples when wrapping a pool (the std needs ddof=1) — the same gate the
        legacy code enforced, surfaced here per ADR-0002."""
        est = self.estimates[i]
        if est is not None:
            return est
        pool = self.pools[i]
        n = int(pool.shape[0])
        if n < 2:
            raise RuntimeError(
                f"_estimate_for({i}, name={self.names[i]!r}): no Estimate set and the raw pool has "
                f"n={n} < 2 — need a pilot of ≥2 samples to wrap a pool as a Poolwise Estimate "
                f"(or call set_estimate). (ADR-0002: a variance with n<2 is undefined, not defaulted.)")
        mean = float(pool.mean())
        s2 = float(pool.var(ddof=1))  # the per-sample variance s² (NOT divided by n)
        return _est.Estimate(
            theta_hat=np.array([mean], dtype=np.float64),
            cov=np.array([[s2 / n]], dtype=np.float64),       # the already-divided SAMPLING variance
            names=(self.names[i],),
            shrink=_est.Poolwise(per_sample_var=np.array([s2], dtype=np.float64)),
            support=(_est.Support.POSITIVE,),
            family=(_est.CIFamily.NORMAL,),
            kind="mean",
        )

    # ------------------------------------------------------------------ #
    # Gradient (OT analytic if present, else central FD on f)
    # ------------------------------------------------------------------ #
    def _gradient(self, point: np.ndarray) -> np.ndarray:
        try:
            g = self.f.gradient(ot.Point(point))
            return np.array([g[i, 0] for i in range(self.d)], dtype=float)
        except Exception:
            return self._fd_gradient(point)

    def _fd_gradient(self, point: np.ndarray) -> np.ndarray:
        g = np.empty(self.d)
        f0_pt = point.copy()
        for i in range(self.d):
            h = self.fd_rel_step * max(abs(point[i]), 1.0)
            xp = f0_pt.copy(); xp[i] += h
            xm = f0_pt.copy(); xm[i] -= h
            yp = float(self.f(ot.Point(xp))[0])
            ym = float(self.f(ot.Point(xm))[0])
            g[i] = (yp - ym) / (2.0 * h)
        return g

    # ------------------------------------------------------------------ #
    # Optional curvature diagnostic via TaylorExpansionMoments
    # ------------------------------------------------------------------ #
    def _second_order_mean(self) -> Optional[float]:
        """
        Build a kernel-smoothed joint distribution from the current pools and
        ask OpenTURNS for the second-order (curvature-corrected) mean. Returns
        None if anything in this version-sensitive path is unavailable.
        """
        try:
            marginals = []
            for i in range(self.d):
                sample = ot.Sample(self.pools[i].reshape(-1, 1))
                marginals.append(ot.KernelSmoothing().build(sample))
            dist = _make_joint(marginals)
            inputRV = ot.RandomVector(dist)
            outputRV = ot.CompositeRandomVector(self.f, inputRV)
            taylor = ot.TaylorExpansionMoments(outputRV)
            return float(taylor.getMeanSecondOrder()[0])
        except Exception:
            return None

    # ------------------------------------------------------------------ #
    # The allocation step
    # ------------------------------------------------------------------ #
    def step(self, second_order_check: bool = True) -> Recommendation:
        """
        Re-estimate each input's contribution from its Estimate (or wrapped pool) and recommend
        additional samples (§6 Phase 2).

        The §2.2 quadratic form `Var(E[f]) = gᵀΣg` replaces the diagonal `sum a_i/n_i`; it is
        bit-for-bit today's sum on an all-means / diagonal Σ (the §2.2 exactness claim, asserted on
        the no-regression tests). Σ is assembled block-diagonal across inputs (each input's `cov[0,0]`
        on the diagonal), carrying any cross-input off-diagonal an `Estimate.cross` declares (§4.2). The
        allocation is the §2.3 cost-constrained c-optimal SOCP (the sign-safe `Q = diag(g)·R·diag(g)`
        form, CLARABEL) — it reduces to the closed form `n_i* ∝ √(a_i/c_i)` on the diagonal, and the
        returned `n*` is checked against `gᵀΣ(n*)g ≈ V*` (ADR-0002, §8 correction 3: a solver `optimal`
        status does NOT catch a mixed-sign sign-fold). When a non-binding `min()` arm is within a
        plausible tie the driver enters the §4.1 `kink_regime`: it replaces `gᵀΣg` with the Clark-1961
        closed-form `E[min]`/`Var[min]` (deterministic, O(1), no MC), funds BOTH contending arms by the
        `Φ(±t)` criticality weights, and REFUSES convergence while `P(a non-binding arm is the min) =
        Φ(−t) > α` (the over-permissive false-SAT the guard forbids).

        Each input needs an Estimate set (`set_estimate`) or a pilot of ≥2 raw samples (the wrap gate).
        Returns a Recommendation; collect its suggested samples / re-set the Estimates, and call step()
        again until `.converged` is True.
        """
        self.iteration += 1

        # --- gather each input's Estimate (set, or pool wrapped) and its marginal (θ̂_i, Σ_ii) --- #
        ests = [self._estimate_for(i) for i in range(self.d)]
        mu = np.array([float(e.theta_hat[0]) for e in ests])
        # n_i: the effective sample count behind input i, for the topup damping. From a Poolwise law
        # it is per_sample_var/cov (= the wrapped pool's n); a Fixed/Regression/Quantile law carries
        # no shrinkable sample n in the legacy sense, so we read the live pool length (or 1) — the
        # damping is a heuristic approach to the SOCP target and only needs a positive base.
        n = np.array([self._effective_n(i, ests[i]) for i in range(self.d)], dtype=float)
        # The per-sample spread (the legacy report's `sigma` column AND the legacy `a_i = (g·σ)²`,
        # which transport_sweep/throughput_bound rank on — kept with its legacy semantics so the
        # ranking is byte-for-byte today's on the diagonal). `cov[0,0]` is the already-divided SE²
        # (Σ_ii), used by the quadratic form below; `_report_sigma` is the per-sample stddev.
        sigma_report = np.array([_report_sigma(e) for e in ests])  # per-sample spread

        grad = self._gradient(mu)

        # --- the joint input covariance Σ (block-diagonal across inputs + declared cross terms) --- #
        Sigma = self._assemble_sigma(ests)

        # --- the §2.2 quadratic form (Cov-aware) and the per-input marginal contribution --- #
        Sg = Sigma @ grad
        var_quad = float(grad @ Sg)               # gᵀΣg — replaces sum a_i/n_i (equal on diagonal Σ)
        # a_persample: the legacy per-sample a_i = (g_i·σ_per,i)² (NOT divided by n) — the field
        # transport_sweep/throughput_bound rank `p.a` on; byte-for-byte today's on the diagonal.
        a = (grad * sigma_report) ** 2
        # a_contrib: the §2.2 Cov-aware per-input variance CONTRIBUTION g_i·(Σg)_i (already divided;
        # = g_i²·Σ_ii on the diagonal = the legacy `var_contribution`). This is what `var_contribution`
        # carries (the divided share of Var(E[f])), the off-diagonal cross-terms now folded in.
        a_contrib = grad * Sg

        estimate = float(self.f(ot.Point(mu))[0])  # the hard-min point value f(μ̂)

        # --- §4.1 the min()-kink path: binding-margin diagnostic + Clark closed form (no MC) --- #
        kink = self._kink_assessment(mu, grad, Sigma, estimate)
        kink_regime = kink is not None
        if kink_regime:
            var_est = kink["var_min"]             # Clark Var[min] supersedes the single-arm gᵀΣg
            estimate_kink = kink["E_min"]         # Clark de-biased E[min] (the −a·φ(t) Jensen fix)
            p_nonbind = kink["p_nonbinding_max"]  # max Φ(−t) over the non-binding arm(s)
            # The allocation gradient in the kink regime is the Φ(±t)-weighted both-arm gradient, so
            # the previously-zero-weighted non-binding arm's inputs get funded (§4.1 mechanism 3).
            grad_alloc = kink["grad_weighted"]
        else:
            var_est = var_quad
            estimate_kink = None
            p_nonbind = 0.0
            grad_alloc = grad

        # --- §4.3 the per-family CI multiplier (z / t_dof / conservative), honest about the law --- #
        # Pass the Cov-aware contribution: an input "contributes to the bound" iff its share of
        # Var(E[f]) is nonzero, so a pin (a_contrib≈0) does not drag the multiplier.
        ci_mult, ci_mult_label = self._family_multiplier(ests, a_contrib)
        ci_half = ci_mult * math.sqrt(max(var_est, 0.0))

        # The variance budget: V_target = (h/mult)². The multiplier is now per-family, so the budget
        # is computed against the SAME multiplier the CI uses (a tighter t-multiplier shrinks V_target,
        # demanding a tighter variance — the honest small-n widening of §4.3, not a fixed z).
        V_target = (self.tolerance / ci_mult) ** 2 if ci_mult > 0 else float("inf")

        # --- §7.D the irreducible-prior floor, SEPARATED from the shrinkable sampling variance --- #
        # `var_floor` = Σ a_i over the DECLARED-SPREAD `Fixed` inputs (family=NORMAL — an engineering-
        # judgement prior no sampling reduces: `R_gen` σ=8, `B_op` σ=64). It is the part of `gᵀΣg` that
        # is structurally un-shrinkable, distinct from `var_shrinkable` (the means/medians/funded fits
        # sampling CAN tighten). §7.D: surface it as its OWN line, not conflated into one number — the
        # same conflation-of-quantities shape ADR-0012 forbids (don't merge two distinct quantities into
        # one), one level up from the `len(pools)`-as-`n` fix. A true-constant DEGENERATE pin is ALREADY
        # ~0 in `var_est` (Fix in `_assemble_sigma`), so it is in NEITHER bucket. Computed off `a_contrib`
        # (the smooth-regime per-input `gᵀΣg` split); in the kink regime `var_est` is Clark's Var[min], a
        # min-moment that is NOT a per-input sum, so the split does not apply there (the kink guard owns it).
        if kink_regime:
            var_floor = 0.0
            var_shrinkable = float(var_est)
        else:
            var_floor = float(sum(
                float(a_contrib[i]) for i in range(self.d)
                if isinstance(ests[i].shrink, _est.Fixed)
                and _est._family_tag(ests[i].family[0]) is _est.CIFamily.NORMAL))
            var_shrinkable = float(var_est) - var_floor
        # The §2.3 honest edge: the declared-prior floor ALONE exceeds the CI target, so NO amount of
        # sampling meets it — the CI honestly rests on the prior. We do NOT fire `converged` on the
        # shrinkable part alone (that would be the false-SAT §2.3 forbids — claiming a 1.0-dps CI while
        # the R_gen prior alone is ±√var_floor); instead this flag SURFACES the unreachability so the
        # report says why, and run()'s stall-stop terminates rather than spinning against it.
        floor_blocks_target = bool(var_floor > V_target)

        # --- convergence: variance budget met AND the §4.1 guard (no live arg-min flip) passes --- #
        alpha = 1.0 - self.confidence  # the arg-min-flip tolerance (two-sided level's tail)
        guard_ok = (not kink_regime) or (p_nonbind <= alpha)
        converged = (var_est <= V_target) and guard_ok

        # --- §2.3 the allocation: the cost-constrained c-optimal SOCP (sign-safe Q-form, CLARABEL) --- #
        # The SOCP's per-component A_i is DERIVED from each input's typed D2 marginal (`_fundability`,
        # `A_i = −marginal·n_eff²`), NOT the `Σ_ii·len(pools)`-as-`n` conflation — built from the ALLOCATION
        # gradient (the Φ(±t)-weighted one in the kink regime, the analytic one otherwise) and Σ. On a
        # diagonal Σ this reproduces the closed-form n_i* ∝ √(a_i/c_i) (rel diff ~1e-5; §8(b)). An input
        # whose variance does NOT respond to effort drops out (passed via `ests`): a Fixed pin (marginal=0,
        # irreducible) AND a leverage/misfit-floored fit (RegressionLaw marginal≈0 — more iters never cross
        # 1/Sxx, §4.3) — both keep their current n and contribute their a_i to the bound, but get no funding.
        n_star = self._socp_allocation(grad_alloc, Sigma, self.costs, V_target, n_current=n, ests=ests)

        # Incremental top-up toward the optimal totals. UNCHANGED from the legacy driver: damp the
        # *whole vector* by one scalar so the Neyman proportions are preserved (clipping each input
        # independently flattens toward uniform — exactly wrong). The SOCP supplies the target n_star;
        # this greedy per-round damping is the heuristic approach to it (correct on the diagonal,
        # the SOCP is what is exact once Σ is non-diagonal — §2.3).
        # var_contribution: the §2.2 Cov-aware divided share of Var(E[f]) (= legacy a/n on the diagonal,
        # now with off-diagonal cross-terms folded in). The reported field AND the forward-progress
        # ranking (where the residual variance concentrates).
        var_contrib = np.where(np.isfinite(a_contrib), a_contrib, np.inf)
        topup = np.maximum(n_star - n, 0.0)
        if not converged:
            scale = 1.0
            # Damp 1: no pool grows by more than growth_cap * n in one round.
            grow_lim = self.growth_cap * np.maximum(n, 1.0)
            binding = topup > grow_lim
            if np.any(binding):
                scale = min(scale, float(np.min(grow_lim[binding] / topup[binding])))
            # Damp 2: respect an absolute per-round batch ceiling.
            if self.max_batch is not None:
                over = topup > self.max_batch
                if np.any(over):
                    scale = min(scale, float(self.max_batch) / float(topup.max()))
            topup = np.floor(topup * scale)
            # Forward progress if rounding zeroed the vector — but ONLY onto a FUNDABLE input (one whose
            # variance actually responds to effort, the same `_fundability` mask the allocation uses). A
            # floored fit / a pin is the WORST variance contributor yet un-fundable: nudging IT makes no
            # progress (its variance does not move, so it stays worst and the loop pours the whole — often
            # expensive — fit/pin budget into it every round, the very over-funding the conflation removal
            # forbids, §4.1/§4.3). Restricting the nudge to fundable inputs closes that leak; if NOTHING
            # is fundable the variance is irreducible and no nudge can help (the loop will not converge —
            # the §2.3 honest-edge the convergence/guard surfaces, not papered over by a futile nudge).
            if topup.sum() == 0:
                _m, _A, fundable_nudge = self._fundability(grad_alloc, Sigma, n, ests)
                cand = np.where(fundable_nudge & np.isfinite(var_contrib), var_contrib, -np.inf)
                if np.any(np.isfinite(cand)):
                    worst = int(np.argmax(cand))
                    topup[worst] = max(1.0, math.ceil(0.5 * n[worst]))
        else:
            topup[:] = 0.0
        topup = topup.astype(int)

        # Shadow price of variance (the marginal cost of tightening V*). Kept as the closed-form
        # diagonal expression d(cost)/dV* = (Σ √(a c))² / V*² — a scalar diagnostic, not the
        # allocation (which is the SOCP); honest on the diagonal, an approximation off it.
        sqrt_ac = np.sqrt(np.maximum(a, 0.0) * self.costs)
        S = float(sqrt_ac.sum())
        shadow = (S ** 2) / (V_target ** 2) if (V_target > 0 and math.isfinite(V_target)) else float("inf")

        second_order = (
            self._second_order_mean() if (second_order_check and not converged) else None
        )

        prims = [
            PrimitiveState(
                index=i, name=self.names[i], n=int(n[i]), mean=float(mu[i]),
                sigma=float(sigma_report[i]), grad=float(grad[i]), a=float(a[i]),
                cost=float(self.costs[i]), var_contribution=float(var_contrib[i]),
                recommend=int(topup[i]),
            )
            for i in range(self.d)
        ]

        return Recommendation(
            iteration=self.iteration, converged=converged, estimate=estimate,
            estimate_second_order=second_order, var_estimate=var_est,
            ci_halfwidth=ci_half, target_halfwidth=self.tolerance,
            shadow_price=shadow, primitives=prims,
            ci_multiplier=ci_mult, ci_multiplier_label=ci_mult_label,
            kink_regime=kink_regime, estimate_kink=estimate_kink, p_nonbinding_max=p_nonbind,
            var_floor=var_floor, var_shrinkable=var_shrinkable, floor_blocks_target=floor_blocks_target,
        )

    # ------------------------------------------------------------------ #
    # §6 Phase-2 step() helpers — each single-homes one piece of the spec.
    # ------------------------------------------------------------------ #
    def _effective_n(self, i: int, est: "_est.Estimate") -> float:
        """The effective sample count behind input `i` (for the topup damping base — a positive
        scale, not a statistical quantity the bound reads). A `Poolwise` law carries it as
        `per_sample_var/cov` (= the wrapped pool's n); a `QuantileLaw` carries `n` explicitly; a
        `Fixed`/`RegressionLaw`/`Composed` carries no shrinkable sample n, so we read the live pool
        length, falling back to 1 (a pin/seed never grows, so its base is immaterial)."""
        shrink = est.shrink
        if isinstance(shrink, _est.Poolwise):
            cov00 = float(est.cov[0, 0])
            psv0 = float(shrink.per_sample_var[0])
            return float(round(psv0 / cov00)) if cov00 > 0.0 else 1.0
        if isinstance(shrink, _est.QuantileLaw):
            return float(shrink.n)
        return float(max(len(self.pools[i]), 1))

    def _marginal_for(
        self, i: int, ests: Optional[Sequence["_est.Estimate"]], sigma_ii: float, n_eff: float
    ) -> float:
        """The typed D2 marginal `dΣ_ii/d(effort)` (≤ 0) for input `i` — the local rate at which one more
        unit of THIS input's bench effort lowers its sampling variance, read from its `ShrinkLaw` (the SSOT
        of the shrink law, P1/P8 — §1 D2). This is the quantity the §2.3 allocator equalizes per unit cost,
        and it is what REPLACES the `A = Σ_ii·len(pools)`-as-`n` conflation (`_socp_allocation`): the law
        owns the form of the derivative; the driver supplies only the live operating point (`sigma_ii`, the
        already-divided `cov[0,0]` the driver holds; `n_eff`, the effective count from `_effective_n`).

        When no Estimate is available for the input (a pure raw-pool path that did not wrap), fall back to
        the Poolwise 1/n marginal `−Σ_ii/n_eff` — the legacy behavior for a mean, so a pool-fed and an
        Estimate-fed driver agree (the confirmed fixed point). A `RegressionLaw`/`QuantileLaw` carries its
        own currency-conversion default (iters_per_point=n_eff for the fit's floor; readings_per_effort=1.0
        for the median's one-reading-per-tick) — the honest least-bad until a bench wires its real ratio."""
        if ests is None or ests[i] is None:
            n = float(n_eff)
            return -float(sigma_ii) / n if n > 0.0 else 0.0
        return float(ests[i].shrink.marginal_dvar_deffort(float(sigma_ii), float(n_eff)))

    def _fundability(
        self, grad: np.ndarray, Sigma: np.ndarray, n_eff: np.ndarray,
        ests: Optional[Sequence["_est.Estimate"]],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The ONE home (P1) of "which inputs the allocation can fund, and at what per-sample variance" —
        consulted by BOTH `_socp_allocation` (the SOCP/closed-form target) AND `step()`'s forward-progress
        nudge, so the conflation removal is consistent across the two (a floored fit / a pin the allocator
        de-funds is also one the nudge will not fund). Returns `(marginal, A, fundable)`:

          * `marginal_i = dΣ_ii/d(effort)` (≤ 0), the typed D2 from input i's `ShrinkLaw` (§1 D2). REPLACES
            the `A = Σ_ii·len(pools)`-as-`n` uniform-1/n conflation: the law decides its own shrink rate.
          * `A_i = −marginal_i · n_eff²` — the per-sample variance s.t. `Σ_ii(n)=A_i/n` at the law's true
            rate. For a 1/n law (Poolwise/QuantileLaw, marginal=−Σ_ii/n) this is `Σ_ii·n_eff` BYTE-FOR-BYTE
            (the mean/median case unchanged); for a floored fit / a pin (marginal≈0) it is ≈0.
          * `fundable_i = (g_i≠0) AND (A_i>0)` — moves f AND its variance responds to effort. A pin or a
            leverage/misfit-floored fit (marginal≈0) is NOT fundable: no finite iter budget reduces it
            (§2.3/§4.3); it keeps its current n and still contributes its a_i to the bound via gᵀΣg."""
        d = self.d
        g = np.asarray(grad, dtype=float)
        Sigma_diag = np.array([float(Sigma[i, i]) for i in range(d)])
        ne = np.maximum(n_eff, 1.0)
        marg = np.array([self._marginal_for(i, ests, float(Sigma_diag[i]), float(ne[i]))
                         for i in range(d)])
        A = -marg * (ne ** 2)
        fundable = (np.abs(g) > 0.0) & (A > 0.0) & np.isfinite(A)
        return marg, A, fundable

    def _assemble_sigma(self, ests: Sequence["_est.Estimate"]) -> np.ndarray:
        """The joint input covariance Σ (d×d) the §2.2 `gᵀΣg` consumes. BLOCK-DIAGONAL across inputs
        (each input's own sampling variance `cov[0,0]` on the diagonal — the `cross == {}` independence
        default, correct for the leaf-eval suite as it stands), carrying an off-diagonal `Σ_ij` ONLY
        where input i's `Estimate.cross` declares a coupling keyed by input j's registry name (§4.2 —
        the slope/intercept pairing a Phase-3 bench populates). A declared cross term is symmetrized
        (and cross-checked when BOTH sides declare it — ADR-0002: two homes for one number that
        disagree is a loud fault). The diagonal-only case (every `cross == {}`) yields a diagonal Σ,
        on which `gᵀΣg == Σ g_i² Σ_ii` is bit-for-bit the legacy `sum a_i/n_i` (the no-regression fact).

        A `DEGENERATE`-family component is a TRUE CONSTANT (a deployment/layout fact — `n_gen` = 3 cores;
        §3 PIN-true-constant row), NOT a quantity with CI-bearing uncertainty: the spec gives it `a_i ≈ 0`,
        "~0 bound contribution", and §4.3 gives it "no sampling interval". So its diagonal `Σ_ii` (and any
        cross row/column — a constant covaries with nothing) is ZEROED here: the bound must NOT rest on a
        constant's frozen display σ (the ADR-0008 "a derived value frozen as a literal" slip — `n_gen`'s
        σ=0.05 is a placeholder on an integer core count, not a real spread). This is the bound-side twin of
        `_family_multiplier` already excluding DEGENERATE from the CI multiplier: the `family` field is the
        contract's SSOT of how a component enters the CI (ADR-0012 P8), and the driver — the one home that
        assembles the bound (P1) — honors it for the VARIANCE exactly as it does for the multiplier. A
        DECLARED-SPREAD `Fixed` (family=NORMAL, e.g. `R_gen` σ=8) is UNAFFECTED: it still contributes its
        `a_i` to the bound (the CI honestly rests on the prior — §2.3 / §7.D); only a true constant drops out.
        All-mean models carry no DEGENERATE input, so Σ is unchanged → the §2.2 byte-for-byte fact holds."""
        d = self.d
        Sigma = np.zeros((d, d), dtype=float)
        degenerate = [_est._family_tag(ests[i].family[0]) is _est.CIFamily.DEGENERATE for i in range(d)]
        for i in range(d):
            # A true constant (DEGENERATE) contributes ZERO CI-bearing variance (§3 ~0 bound contribution);
            # a NORMAL declared-spread prior contributes its declared σ² (the CI honestly rests on it).
            Sigma[i, i] = 0.0 if degenerate[i] else float(ests[i].cov[0, 0])
        # Map a registry/component name -> input index, so a cross entry keyed by the OTHER input's
        # name resolves to a matrix position. (An input's own name maps to itself; a cross entry keyed
        # by an unknown name — a coupling to a quantity not in THIS model — is ignored, since it is not
        # a position in this Σ; the bound over this model's inputs cannot carry it.)
        name_to_idx: Dict[str, int] = {}
        for i in range(d):
            name_to_idx.setdefault(self.names[i], i)
        for i in range(d):
            # A true constant covaries with nothing — skip its declared cross entries (and below, any
            # cross entry keyed TO it is dropped by the degenerate-j guard), so the row/column stays zero.
            if degenerate[i]:
                continue
            for other_name, cov_ij in dict(ests[i].cross).items():
                j = name_to_idx.get(str(other_name))
                if j is None or j == i or degenerate[j]:
                    continue
                val = float(cov_ij)
                # If the mirror side also declared it, the two MUST agree (one number, two homes).
                if Sigma[i, j] != 0.0 and not math.isclose(Sigma[i, j], val, rel_tol=1e-9, abs_tol=1e-12):
                    raise ValueError(
                        f"_assemble_sigma: inputs {self.names[i]!r} and {self.names[j]!r} declare "
                        f"DIFFERENT cross-covariances ({Sigma[i, j]} vs {val}) — one coupling has two "
                        f"disagreeing homes (ADR-0002 / P1 single-source-of-truth).")
                Sigma[i, j] = val
                Sigma[j, i] = val
        return Sigma

    def _family_multiplier(
        self, ests: Sequence["_est.Estimate"], a: np.ndarray
    ) -> tuple[float, str]:
        """The §4.3 per-family CI multiplier actually applied, and a label of how it was chosen.
        Each input's first component carries a `family`: NORMAL→z, STUDENT_T(dof)→t_{dof,1−α/2},
        EMPIRICAL→(the bench owns its interval; we fall back to z here and LABEL it, since the driver
        has no bootstrap sample), DEGENERATE→no sampling interval (a pin contributes no multiplier).

        Over inputs of DIFFERING family the combined pivot is NOT an exact Student-t (a Behrens-Fisher
        object; §4.3): we take the MOST-CONSERVATIVE multiplier among the inputs that actually CONTRIBUTE
        to the bound (a_i > 0) — the smallest-dof Student-t if any contributing input is a small-n fit,
        else z — and LABEL it 'conservative …' so 'converged' is announced as *variance budget met under
        a conservative multiplier*, never a false exactness claim. A pin (DEGENERATE, a_i≈0) does not
        widen the interval; an all-pin / all-Fixed contribution would leave only z (the prior's Normal)."""
        contributing = [i for i in range(self.d) if float(a[i]) > 0.0]
        if not contributing:
            return self.z, "z"
        best_mult = self.z
        best_label = "z"
        any_t = False
        min_dof: Optional[int] = None
        for i in contributing:
            fm = ests[i].family[0]
            tag = _est._family_tag(fm)
            if tag is _est.CIFamily.STUDENT_T:
                dof = int(fm.dof)  # type: ignore[union-attr]  — a StudentT entry carries dof
                t_mult = _t_multiplier(dof, self.confidence)
                any_t = True
                min_dof = dof if min_dof is None else min(min_dof, dof)
                if t_mult > best_mult:
                    best_mult = t_mult
                    best_label = f"t(dof={dof})"
            # NORMAL / EMPIRICAL / DEGENERATE -> z is the multiplier we can defend here; EMPIRICAL's
            # bench-owned interval is not available to the driver, NORMAL is z, DEGENERATE adds nothing.
        # When the contributing inputs mix families, mark the multiplier conservative (it is the
        # most-conservative single-family multiplier, not the exact combined pivot — §4.3).
        families = {_est._family_tag(ests[i].family[0]) for i in contributing}
        if any_t and len(families) > 1:
            dof_used = min_dof if min_dof is not None else 1
            best_mult = max(best_mult, _t_multiplier(dof_used, self.confidence))
            best_label = f"conservative t(dof={dof_used})"
        return best_mult, best_label

    def _kink_assessment(
        self, mu: np.ndarray, grad: np.ndarray, Sigma: np.ndarray, hard_min: float
    ) -> Optional[dict]:
        """§4.1 — the `min()`-kink binding-margin diagnostic + the Clark-1961 closed form. Returns None
        (the smooth regime — today's behavior) unless `f` is a `min()` whose SECOND-tightest arm is
        within a statistically-plausible tie of the binding arm, in which case it returns the Clark
        moments (deterministic, O(1), NO Monte-Carlo) the kink path uses.

        Mechanism. The model exposes its `min()` arms via the `self.arms_fn` hook (set by a model's
        `build_driver`) as `[(capacity, {input_name: ∂capacity/∂input}), …]` (a Phase-3 model surface;
        absent it, the driver cannot see the arms and stays in the smooth regime — the honest default,
        never a fabricated tie). Each arm is linearized to `Normal(μ_k, σ_k²)` with `μ_k = capacity_k(μ̂)`,
        `σ_k² = ∇capacity_kᵀ Σ ∇capacity_k`, cross-covariance `∇c_aᵀ Σ ∇c_b`. The two tightest arms drive
        Clark's exact `min`-moments: `a = SD(c_bind − c_contender)`, `t = (μ_bind − μ_contender)/a`,
        `E[min] = μ_bind·Φ(−t) + μ_contender·Φ(t) − a·φ(t)`, and `Var[min]` from the second moment. An
        arm is the realized min iff it is the smaller draw, so `P(binding is min) = Φ(−t)` (the larger
        weight) and `P(contender is min) = Φ(t)` (the arg-min-flip probability). The `kink_regime` fires
        when `P(contender is min) = Φ(t)` exceeds a small floor (a live arg-min flip). The both-arm
        allocation gradient is `Φ(±t)`-weighted (the SSTA criticality weights, summing to 1).
        """
        arms = self._model_arms(mu)
        if arms is None or len(arms) < 2:
            return None  # not a visible-min model -> smooth regime (today's behavior)

        # Each arm: (capacity μ_k, σ_k = sqrt(∇c_kᵀ Σ ∇c_k)).  Cross-cov via the same Σ.
        caps = np.array([c for (c, _g) in arms], dtype=float)
        arm_grads = [np.asarray(g, dtype=float) for (_c, g) in arms]
        arm_var = np.array([float(g @ Sigma @ g) for g in arm_grads])
        arm_sd = np.sqrt(np.maximum(arm_var, 0.0))

        # The binding arm is the realized min; the contender is the next-tightest capacity.
        order = np.argsort(caps)
        b, s = int(order[0]), int(order[1])  # binding, second
        mu_b, mu_s = float(caps[b]), float(caps[s])
        sd_b, sd_s = float(arm_sd[b]), float(arm_sd[s])
        cov_bs = float(arm_grads[b] @ Sigma @ arm_grads[s])

        # Degenerate / measure-zero guards (ADR-0002 honest about the one pathological case): if the
        # spread of (binding − contender) is ~0 there is no resolvable tie scale; treat as smooth.
        a_kink = math.sqrt(max(sd_b ** 2 + sd_s ** 2 - 2.0 * cov_bs, 0.0))
        if a_kink <= 1e-12:
            return None

        from scipy.stats import norm  # deterministic Φ, φ — no draws (ADR-0002 loud if scipy absent)
        # Clark's min-moments via min(X,Y) = −max(−X,−Y), arms (binding=b, contender=s) standardized
        # by the SD of their difference. t = (μ_b − μ_s)/a. An arm is the realized min iff it is the
        # SMALLER draw, so P(b is min) = P(b < s) = Φ((μ_s−μ_b)/a) = Φ(−t) (the binding arm, smaller
        # mean, usually wins); P(s is min) = Φ(t) (the contender's arg-min-flip probability). The
        # criticality weights Φ(−t), Φ(t) sum to 1.
        t = (mu_b - mu_s) / a_kink
        P_b_min = float(norm.cdf(-t))   # P(binding arm is the min)   = Φ(−t)   (the larger weight)
        P_s_min = float(norm.cdf(t))    # P(contender is the min)     = Φ(t)    (the flip probability)
        phi_t = float(norm.pdf(t))
        E_min = mu_b * P_b_min + mu_s * P_s_min - a_kink * phi_t
        E_sq = ((mu_b ** 2 + sd_b ** 2) * P_b_min
                + (mu_s ** 2 + sd_s ** 2) * P_s_min
                - (mu_b + mu_s) * a_kink * phi_t)
        var_min = max(E_sq - E_min ** 2, 0.0)

        # The binding-margin trigger: enter the kink regime only when the contender has a live chance
        # of being the realized min (Φ(t) = P(contender is min) above a small floor — a statistically-
        # plausible tie). Far from a tie this →0 and we return None (smooth; the analytic single-arm
        # gradient is honest, the non-binding arm's df/dx = 0 is correct, behavior is exactly today's).
        p_nonbinding_max = P_s_min
        if p_nonbinding_max < _KINK_PFLOOR:
            return None

        # The Φ(±t)-weighted both-arm allocation gradient (SSTA criticality; weights sum to 1). Only
        # the two tightest arms carry weight; the rest are far and drop out (Φ→0). This funds the
        # previously-zero-weighted contender arm's inputs near the tie, curing the dead-gradient.
        grad_weighted = P_b_min * arm_grads[b] + P_s_min * arm_grads[s]
        return {
            "E_min": float(E_min),
            "var_min": float(var_min),
            "p_nonbinding_max": float(p_nonbinding_max),
            "t": float(t),
            "a": float(a_kink),
            "grad_weighted": grad_weighted,
            "binding": b,
            "contender": s,
        }

    def _model_arms(self, point: np.ndarray) -> Optional[list]:
        """The `min()` arms of `f` at `point`, as `[(capacity, ∇capacity_ndarray), …]`, IF the model
        exposes them via an `arms(point_dict)` hook (a Phase-3 model surface — `model_*.stage_capacities`
        plus per-arm gradients). Absent the hook the driver cannot see the arms and returns None (the
        smooth regime — it never fabricates a min structure from the symbolic `f`, which OT cannot
        differentiate through `min()` anyway). The hook is attached to the driver as `self.arms_fn`
        (set by a model's `build_driver`), keeping `f` the single symbolic SSOT and the arm decomposition
        the model's concern (P1/P2)."""
        fn = getattr(self, "arms_fn", None)
        if fn is None:
            return None
        point_dict = {self.names[i]: float(point[i]) for i in range(self.d)}
        arms = fn(point_dict)
        if not arms:
            return None
        out = []
        for cap, grad_map in arms:
            g = np.array([float(grad_map.get(self.names[i], 0.0)) for i in range(self.d)], dtype=float)
            out.append((float(cap), g))
        return out

    def _socp_allocation(
        self, grad: np.ndarray, Sigma: np.ndarray, costs: np.ndarray, V_target: float,
        n_current: np.ndarray, ests: Optional[Sequence["_est.Estimate"]] = None,
    ) -> np.ndarray:
        """§2.3 — the cost-constrained c-optimal allocation as a SOCP (the sign-safe
        `Q = diag(g)·R·diag(g)` form, CLARABEL), reducing to the closed form `n_i* ∝ √(a_i/c_i)` on the
        diagonal. Returns the TOTAL target `n*` per input to hit `Var(E[f]) = gᵀΣ(n*)g ≤ V_target`.

        The sign-safe form (§8 correction 3): optimize over the genuinely-positive per-component SE
        `w_i = √(A_i/n_i) > 0` (`A_i = g_i²·s²_per,i`, the per-sample variance contribution), with the
        gradient sign ABSORBED into `Q` so `wᵀQw = gᵀΣg` survives MIXED-SIGN gradients (which
        `model_capacity` has). `min Σ c_i A_i w_i^{-2}` (convex) s.t. `‖L_Qᵀ w‖₂² ≤ V*` (convex). After
        the solve, ADR-0002 REQUIRES `gᵀΣ(n*)g ≈ V*` on the returned `n*` (the solver's `optimal`
        status does NOT catch a sign-fold) — a violation raises.

        FUNDABILITY IS THE TYPED D2 MARGINAL, NOT `len(pools)`-as-`n` (the conflation removal, §1 D2/
        §2.3/§4.3). The per-component shrink-rate is read from each input's `ShrinkLaw.marginal_dvar_deffort`
        (the SSOT of HOW its variance responds to effort — P1/P8), NOT by assuming `Σ_ii(n) = Σ_ii·n_cur/n`
        uniformly. An input is fundable iff it moves `f` (g≠0) AND its variance actually RESPONDS to effort
        (`marginal < 0`). A pin (`marginal = 0`, irreducible) OR a leverage/misfit-FLOORED fit
        (`RegressionLaw.marginal ≈ 0` — more iters never cross `1/Sxx`) drops out: it keeps its current n
        and still contributes its `a_i` to the bound (via `gᵀΣg` upstream), but gets NO allocation. The
        per-sample variance `A_i` the closed form / SOCP consume is then DERIVED from the marginal —
        `A_i = −marginal_i · n_eff²` — which for a 1/n law (`marginal = −Σ_ii/n`) recovers `Σ_ii·n_eff`
        BYTE-FOR-BYTE (the mean/median case is unchanged), while a floored fit is simply un-fundable.
        Falls back to the closed-form ratio when cvxpy is unavailable AND Σ is diagonal (the special
        case that needs no solver); a non-diagonal Σ with cvxpy absent is a loud ADR-0002 error."""
        d = self.d
        g = np.asarray(grad, dtype=float)
        # The typed D2 marginal, derived A_i, and the fundable mask — the ONE home (P1, `_fundability`)
        # the nudge in step() shares, so the conflation removal is consistent: `A_i = −marginal·n_eff²`
        # (the law's true shrink rate, NOT `Σ_ii·len(pools)`-as-`n`), and a pin / a floored fit (marginal
        # ≈0) is un-fundable here exactly as it is in the nudge.
        _marg, A, fundable = self._fundability(g, Sigma, n_current, ests)
        if not np.any(fundable) or not (V_target > 0 and math.isfinite(V_target)):
            # Nothing to fund (all pins / floored fits / converged target) -> keep current n.
            return n_current.astype(float).copy()

        # The off-diagonal coupling of the FUNDABLE block (a pin's off-diagonals are irrelevant — it is
        # not allocated). Σ is "diagonal" for allocation iff the fundable inputs are mutually uncorrelated.
        idx = np.where(fundable)[0]
        Sig_sub = Sigma[np.ix_(idx, idx)]
        Sub_diag = np.diag(np.diag(Sig_sub))
        off_diag = float(np.max(np.abs(Sig_sub - Sub_diag))) if len(idx) > 1 else 0.0
        is_diagonal = off_diag <= 1e-12 * max(1.0, float(np.max(np.abs(np.diag(Sig_sub)))))

        # DIAGONAL Σ → the closed form `n_i* ∝ √(a_i/c_i)` (the SOCP's exact diagonal special case, §2.3).
        # We use it DIRECTLY here, not the solver: it is exact, sign-safe (gᵀΣg = Σ g_i²Σ_ii is squared,
        # so the cross-term sign-fold the SOCP guards cannot arise on the diagonal), and ROBUST to the
        # extreme scaling a tight V_target / a wide-σ pilot produces (where a numerical SOCP solver chokes
        # on `cp.power(w,-2)` with w spanning many orders of magnitude — the conditioning the §8 SCS-retry
        # note anticipates). The SOCP is reserved for the NON-diagonal case it is actually needed for
        # (a correlated / fit Σ the closed form cannot express). The two agree to ~1e-5 on the diagonal
        # (§8(b)), so this dispatch is the exact answer either way — not a degraded fallback.
        if is_diagonal:
            return self._closed_form_allocation(g, A, costs, V_target, n_current, fundable)

        # NON-DIAGONAL Σ → the §2.3 SOCP (sign-safe Q-form, CLARABEL/SCS). The correlation R of the
        # fundable block: R = D^{-1} Σ D^{-1}.
        d_sub = np.sqrt(np.maximum(np.diag(Sig_sub), 0.0))
        with np.errstate(divide="ignore", invalid="ignore"):
            Dinv = np.diag(np.where(d_sub > 0, 1.0 / d_sub, 0.0))
        R_sub = Dinv @ Sig_sub @ Dinv
        g_sub = g[idx]
        A_sub = A[idx]
        c_sub = np.asarray(costs, dtype=float)[idx]

        try:
            import cvxpy as cp
        except ImportError as exc:
            raise ImportError(
                "neyman_driver: the §2.3 SOCP allocation needs cvxpy (CLARABEL) for a non-diagonal Σ "
                "(a correlated/fit input). `pip install cvxpy`. (ADR-0002: a non-diagonal Σ cannot be "
                "allocated by the diagonal closed form, so this is a loud requirement, not a fallback.)"
            ) from exc

        # Sign-safe Q = diag(g)·R·diag(g) (PSD by congruence of the PSD correlation R), Q = L_Q L_Qᵀ.
        Q = np.diag(g_sub) @ R_sub @ np.diag(g_sub)
        Q = 0.5 * (Q + Q.T)
        evals, evecs = np.linalg.eigh(Q)
        L_Q = evecs @ np.diag(np.sqrt(np.clip(evals, 0.0, None)))

        m = len(idx)
        w = cp.Variable(m, pos=True)
        objective = cp.Minimize(cp.sum(cp.multiply(c_sub * A_sub, cp.power(w, -2))))
        constraints = [cp.sum_squares(L_Q.T @ w) <= V_target]
        prob = cp.Problem(objective, constraints)

        # CLARABEL is the clean default; on a harsh / ill-conditioned (g, A) instance it can itself
        # SolverError (§8 correction 2 — a documented fact for a few instances, NOT a property of the
        # program), so we RETRY on SCS, which agrees with CLARABEL on the well-posed form. Whichever
        # returns, the gᵀΣ(n*)g ≈ V* assertion below (§8 correction 3) is the real gate — the solver's
        # `optimal` status alone never decides correctness here. We keep the FIRST solution that passes
        # the assertion; if neither solver yields one we raise (ADR-0002: never a silent misallocation).
        def _try(solver) -> Optional[np.ndarray]:
            try:
                prob.solve(solver=solver)
            except cp.error.SolverError:
                return None
            if prob.status not in ("optimal", "optimal_inaccurate") or w.value is None:
                return None
            w_val = np.asarray(w.value, dtype=float)
            if not np.all(np.isfinite(w_val)) or np.any(w_val <= 0.0):
                return None
            n_try = A_sub / np.maximum(w_val ** 2, 1e-300)      # w = √(A/n) -> n = A/w²
            sigma2 = A_sub / np.maximum(n_try, 1e-300)
            Sig_star = np.outer(np.sqrt(sigma2), np.sqrt(sigma2)) * R_sub
            var_star = float(g_sub @ Sig_star @ g_sub)          # the ADR-0002 sign-fold check (§8 corr 3)
            if math.isclose(var_star, V_target, rel_tol=1e-4, abs_tol=1e-9):
                return n_try
            return None

        n_sub: Optional[np.ndarray] = None
        for solver in (cp.CLARABEL, cp.SCS):
            n_sub = _try(solver)
            if n_sub is not None:
                break
        if n_sub is None:
            raise RuntimeError(
                f"neyman_driver: the §2.3 SOCP did not yield an allocation satisfying gᵀΣ(n*)g ≈ V* "
                f"(V_target={V_target:.6g}) on either CLARABEL or SCS — a solver failure or a silent "
                f"sign-fold / ill-conditioning (ADR-0002 / §8 correction 3, the assertion the `optimal` "
                f"status cannot replace). The bound is NOT trusted; surfaced rather than swallowed.")

        n_star = n_current.astype(float).copy()
        n_star[idx] = n_sub
        return n_star

    def _closed_form_allocation(
        self, g: np.ndarray, A: np.ndarray, costs: np.ndarray, V_target: float,
        n_current: np.ndarray, fundable: np.ndarray,
    ) -> np.ndarray:
        """The diagonal closed-form Neyman allocation `n_i* ∝ √(a_i/c_i)` scaled to hit V_target — the
        SOCP's diagonal special case, used as the cvxpy-absent fallback ONLY when Σ is diagonal (where
        it is exact and equals the SOCP). `a_i = g_i²·A_i` (the per-sample variance contribution). This
        is the legacy `neyman_driver` allocation line, preserved for the no-cvxpy diagonal path."""
        a = (g ** 2) * A
        costs = np.asarray(costs, dtype=float)
        sqrt_ac = np.sqrt(np.where(fundable, a * costs, 0.0))
        S = float(sqrt_ac.sum())
        with np.errstate(divide="ignore", invalid="ignore"):
            n_star = np.sqrt(np.where(fundable, a / costs, 0.0)) * (S / V_target)
        return np.where(fundable, n_star, n_current).astype(float)

    # ------------------------------------------------------------------ #
    # Autonomous loop (when you can sample programmatically)
    # ------------------------------------------------------------------ #
    def run(
        self,
        samplers: Optional[Dict[int, Callable[[int], np.ndarray]]] = None,
        pilot: int = 256,
        max_rounds: int = 25,
        verbose: bool = True,
        measurers: Optional[Dict[int, Callable[[int], "_est.Estimate"]]] = None,
    ) -> Recommendation:
        """
        Drive the whole loop when each input can be measured programmatically. TWO modes (exactly one
        of `measurers` / `samplers` must be given — passing both, or neither, is a loud ADR-0002 error):

          * `measurers[i](budget) -> Estimate` (§6 Phase 2): the harmonized form. Each call spends
            `budget` units of this input's bench effort and returns the input's CURRENT `Estimate`
            (the bench owns what 'budget' buys — iters, design points; D2). The driver `set_estimate`s
            each one and steps. This is the form the migrated runners (throughput_bound, transport_sweep,
            untrusted_drive) move to (Phase 4).
          * `samplers[i](k) -> array of k samples` (the legacy raw-pool form): kept so a mid-migration
            caller still works. The driver `add_samples` each draw and steps, wrapping each pool as a
            `Poolwise` Estimate inside step() (the confirmed fixed point — a pool-fed and an
            Estimate-fed run AGREE on the mean case).

        For manual benchmarking, ignore this and use step()/set_estimate()/add_samples() by hand.
        """
        if (measurers is None) == (samplers is None):
            raise ValueError(
                "NeymanDriver.run: pass EXACTLY ONE of `measurers` (the §6 Phase-2 Estimate form) or "
                "`samplers` (the legacy raw-pool form) — got "
                f"{'both' if measurers is not None else 'neither'} (ADR-0002: an ambiguous input "
                "contract is a loud error).")

        def _pilot_and_step() -> "Recommendation":
            if measurers is not None:
                for i, m in measurers.items():
                    self.set_estimate(i, m(pilot))
            else:
                assert samplers is not None
                self.add_samples({i: samplers[i](pilot) for i in samplers})
            return self.step()

        rec = _pilot_and_step()
        if verbose:
            print(rec.report(), "\n")
        rounds = 0
        while not rec.converged and rounds < max_rounds:
            to_fund = [p for p in rec.primitives if p.recommend > 0]
            if not to_fund:
                # No fundable input: every primitive's recommend is 0 — the bound's CI rests ENTIRELY on
                # un-shrinkable variance (declared-spread pins, leverage-floored fits) that sampling cannot
                # reduce. Re-stepping adds no data, so `rec` cannot change: this is a FIXED POINT, not
                # convergence to V_target. Stop rather than spin `max_rounds` identical rounds (ADR-0002:
                # surface the stall loudly; do not churn silently in a way that mimics progress).
                if verbose:
                    print("  STALLED — no input is fundable: the CI is dominated by un-shrinkable variance "
                          "(declared-spread pins and/or leverage-floored fits), which sampling cannot reduce. "
                          "Stopping (a fixed point, NOT convergence to the CI target).\n")
                break
            if measurers is not None:
                for p in to_fund:
                    self.set_estimate(p.index, measurers[p.index](p.recommend))
            else:
                assert samplers is not None
                self.add_samples({p.index: samplers[p.index](p.recommend) for p in to_fund})
            rec = self.step()
            rounds += 1
            if verbose:
                print(rec.report(), "\n")
        return rec


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
# §4.1: the kink regime fires only when a non-binding min()-arm has at least this probability of being
# the realized min (Φ(−t) ≥ floor) — a statistically-plausible tie, not numerical noise. Below it the
# contender is effectively never the min and the analytic single-arm gradient is honest (today's
# behavior). 1e-3 keeps the seed 8.6%-margin tie (Φ(−t)≈0.136) firing while not triggering on a
# comfortably-bound arm.
_KINK_PFLOOR = 1e-3


def _z_from_confidence(confidence: float) -> float:
    try:
        return float(ot.Normal().computeQuantile(0.5 + confidence / 2.0)[0])
    except Exception:
        return 1.959963984540054  # 95% fallback


def _t_multiplier(dof: int, confidence: float) -> float:
    """The two-sided Student-t CI multiplier `t_{dof, 1−α/2}` for `confidence = 1−α` (§4.3). Uses
    OpenTURNS' Student quantile (the project's already-present stats surface, matching `_z_from_confidence`'s
    use of `ot.Normal`); falls back to scipy then to z if the OT path is unavailable. dof≥1 enforced
    upstream (StudentT's ctor); a 7-point fit is dof=5 → t≈2.571 vs z=1.96 (the honest 31% widening)."""
    p = 0.5 + confidence / 2.0
    try:
        return float(ot.Student(float(dof)).computeQuantile(p)[0])
    except Exception:
        try:
            from scipy.stats import t as _student_t
            return float(_student_t.ppf(p, df=dof))
        except Exception:
            return _z_from_confidence(confidence)


def _report_sigma(est: "_est.Estimate") -> float:
    """The per-sample spread to show in the report's `sigma` column (NOT the bound math — that reads
    `cov`). For a `Poolwise` mean it is `sqrt(per_sample_var[0])` (the stddev_samp the legacy report
    showed, NOT the already-divided SE sqrt(cov[0,0])); for any other law it is the first component's
    marginal SE `sqrt(cov[0,0])` (a Fixed seed's declared spread, a fit's coefficient SE)."""
    shrink = est.shrink
    if isinstance(shrink, _est.Poolwise):
        return float(math.sqrt(max(float(shrink.per_sample_var[0]), 0.0)))
    return float(math.sqrt(max(float(est.cov[0, 0]), 0.0)))


def _make_joint(marginals):
    """JointDistribution (newer OT) or ComposedDistribution (older)."""
    if hasattr(ot, "JointDistribution"):
        return ot.JointDistribution(marginals)
    return ot.ComposedDistribution(marginals)


# This module owns NO model and has no __main__ (ADR-0012 P1/P2): the synthetic
# message-passing demo that previously lived here was extracted to
# tools/analysis/OpenTURNS/examples/demo_msgpass.py. Run a concrete model via its own
# module (examples/demo_msgpass.py, model_capacity.py, model_cycletime.py) or the
# throughput_bound.py runner.
