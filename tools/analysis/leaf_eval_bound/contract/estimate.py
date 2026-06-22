"""
tools/analysis/leaf_eval_bound/contract/estimate.py
====================================

The harmonized statistical contract every leaf-eval benchmark exposes: ONE frozen,
typed `Estimate` value per measurable quantity, so the Neyman allocation driver
consumes them uniformly whatever the estimator (a mean of timings, a regression
slope/intercept, a config pin, a ratio, a quantile). This module is the type SSOT
(ADR-0012 P8: the typed signature IS the contract) — it owns ONLY the contract and
its (de)serialization; it never allocates, never measures, never touches SQL.

This is Phase 0 of the §6 migration in
`docs/design/harmonized-estimator-interface.md`: the contract + its store, ZERO
behavior change. Nothing consumes an `Estimate` yet (the driver, the benches, the
manifest's TRUST/SEED projection are later phases).

The five load-bearing fields, each derived from what the allocation loop actually
touches (spec §1, decisions D1–D5):

  * `theta_hat` + `cov`  replace `(mean, sigma, n)`: `cov` is the SAMPLING
    covariance of `theta_hat`, ALREADY divided / already an SE^2 (NOT a per-sample
    s^2 the driver re-divides by an `n` whose meaning differs per estimator). A
    MATRIX, not a scalar, because one bench can emit correlated components (an OLS
    fit emits slope AND intercept with their off-diagonal — §4.2, the −0.81).
  * `shrink` (a `ShrinkLaw` sum type) replaces the scalar `n`: how `cov` responds
    to one more unit of THIS bench's effort (the allocation hook). The scalar `n`
    is demoted into one parameter of one variant — it is the field that means
    three incompatible things across the suite, and `Var = V/n` is false for a fit
    or a quantile (§1 D2).
  * `support` clips the reported CI to the feasible set (a positive latency's CI
    never crosses 0; a fraction's never exceeds 1 — §1 D3).
  * `family` carries the CI multiplier honestly, PER COMPONENT (a 7-point fit is
    STUDENT_T(dof=5), not Normal; a pin is DEGENERATE; a large-n mean is NORMAL;
    a bootstrap quantile is EMPIRICAL — §1 D4, §4.3). `dof`→Student-t is coherent
    ONLY for the mean (n−1) and the OLS coefficient (n_pts−2).
  * `cross` is reserved for composites and is empty by default (the honest
    "independent of every other bench" — block-diagonal Σ; §1 D5, §4.2).

FAIL LOUD (ADR-0002 / P2 reject-don't-coerce). `__post_init__` is the construction
gate: it VALIDATES (cov is (k,k), finite, symmetric, PSD; |names| == |support| ==
|family| == k; support/family/shrink internally consistent) and RAISES on any
violation — it never coerces (no silent symmetrization, no clamp, no pad). A bench
that cannot honor the contract raises here rather than returning a padded
zero-spread pool. `is_valid()` is the same check as a boolean (the §1-named gate
the driver and the jsonb-read path call).

Public Domain (The Unlicense).
"""
from __future__ import annotations

import enum
import math
from dataclasses import dataclass, field
from typing import Mapping, Optional, Union

import numpy as np

# Numerical tolerances for the construction gate. Symmetry is checked to a tight
# absolute+relative bound (a cov a bench BUILT should be symmetric to round-off);
# PSD is checked on the symmetric part's smallest eigenvalue against a tolerance
# that scales with the matrix magnitude (so a large-variance cov is not rejected
# by a fixed epsilon). These are the ADR-0002 gate's thresholds, named once here.
_SYM_ATOL = 1e-9
_SYM_RTOL = 1e-7
_PSD_EIG_RTOL = 1e-8


# ============================================================================================
# Support — the per-component feasible domain (§1 D3). A component is EITHER one of the three
# named domains OR a concrete (lo, hi) interval. The enum is the named-domain SSOT; a bounded
# interval is the `(lo, hi)` tuple alternative the spec's "REAL | POSITIVE | UNIT | (lo,hi)"
# admits. `support[i]` therefore has type `Support | tuple[float, float]` (see SupportSpec).
# ============================================================================================
class Support(enum.Enum):
    """A component's feasible set, used to clip an otherwise-symmetric CI to a physical bound
    (ADR-0002 honesty: a latency CI that would cross 0, or a fraction CI that would exceed 1,
    is clipped and the boundary proximity surfaced, not printed as an impossible value)."""
    REAL = "real"          # (-inf, +inf): no clipping
    POSITIVE = "positive"  # (0, +inf):    lower edge never crosses 0 (a latency, a rate)
    UNIT = "unit"          # [0, 1]:       a fraction / probability


# A per-component support is a named domain OR an explicit closed interval (lo, hi).
SupportSpec = Union[Support, tuple[float, float]]


# ============================================================================================
# CIFamily — the per-component sampling-law family (§1 D4, §4.3). The enum is the family
# vocabulary SSOT. STUDENT_T additionally carries a `dof`, so a per-component family entry is
# `CIFamily | StudentT` (see FamilySpec): a bare enum member for the parameter-free families,
# a `StudentT(dof)` for the one family that carries the multiplier's degrees of freedom.
# ============================================================================================
class CIFamily(enum.Enum):
    """The sampling-law family of one `theta_hat` component — it selects the CI multiplier the
    driver applies (§4.3): NORMAL→z, STUDENT_T(dof)→t_{dof}, EMPIRICAL→the bench's own interval
    (a bootstrap percentile, NOT a t-interval), DEGENERATE→no sampling interval (a pin)."""
    NORMAL = "normal"          # large-n mean / quantile → multiplier z
    STUDENT_T = "student_t"    # mean (dof=n−1) or OLS coef (dof=n_pts−2) → multiplier t_{dof}
    EMPIRICAL = "empirical"    # sample quantile / bootstrap → the bench's interval, not a t
    DEGENERATE = "degenerate"  # a pin → no sampling interval


@dataclass(frozen=True)
class StudentT:
    """A STUDENT_T family entry carrying its degrees of freedom (the one family with a payload;
    the multiplier is t_{dof, 1−α/2}). `dof >= 1` (ADR-0002: a non-positive dof is a loud
    error, not a silent fallback to z). Legitimate ONLY for a mean (dof=n−1) or an OLS
    coefficient (dof=n_pts−2) — §4.3; the bench is responsible for using it only there."""
    dof: int

    def __post_init__(self) -> None:
        if not isinstance(self.dof, (int, np.integer)) or int(self.dof) < 1:
            raise ValueError(f"StudentT.dof must be an int >= 1; got {self.dof!r}")
        object.__setattr__(self, "dof", int(self.dof))

    @property
    def family(self) -> CIFamily:
        return CIFamily.STUDENT_T


# A per-component family is one of the parameter-free enum members, or a StudentT(dof).
FamilySpec = Union[CIFamily, StudentT]


def _family_tag(f: FamilySpec) -> CIFamily:
    """The CIFamily tag of a per-component family entry (the bare member, or StudentT's tag)."""
    return f.family if isinstance(f, StudentT) else f


# ============================================================================================
# ShrinkLaw — the sum type that replaces the scalar `n` (§1 D2). Each variant carries the data
# its estimator's variance-reduction law needs AND owns the typed D2 marginal — the local
# `dΣ_ii/d(effort)` (≤ 0) that the Neyman allocator equalizes per unit cost (§2.3). The
# `ShrinkLaw` TYPE is the single home (P1) and SSOT (P8) for HOW each estimator's variance
# responds to effort, so the marginal FORM lives here, on the law, not re-derived at the
# allocator: a `Poolwise` mean shrinks `−V/n²`, a `QuantileLaw` median by its order-statistic
# law, a `RegressionLaw` fit is FLOORED by x-leverage (more iters never cross `1/Sxx`; ~0 at the
# floor), a `Fixed` pin is 0. The driver supplies the live operating point (Σ_ii, n_eff — P4
# live-not-frozen) and the law returns the derivative's form; the allocator NEVER substitutes a
# uniform `Σ_ii·len(pools)`-as-`n` for the type (the ADR-0008 "derived value frozen as a literal"
# / ADR-0012 P1/P2 conflation this closes). The variants are frozen dataclasses; `ShrinkLaw` is
# their union. Structural validation (shapes/sign) is per ADR-0002; the marginal is pure math.
# ============================================================================================
@dataclass(frozen=True)
class Poolwise:
    """MEAN. `cov(n) = diag(per_sample_var) / n`: more samples shrink the variance ~1/n. Carries
    the PER-SAMPLE variance vector (s^2, NOT s^2/n — the already-divided value lives in
    `Estimate.cov`). `per_sample_var` is (k,), finite, non-negative."""
    per_sample_var: np.ndarray

    def __post_init__(self) -> None:
        v = _as_f64_array(self.per_sample_var, "Poolwise.per_sample_var")
        if v.ndim != 1:
            raise ValueError(f"Poolwise.per_sample_var must be 1-D (k,); got shape {v.shape}")
        if not np.all(np.isfinite(v)):
            raise ValueError("Poolwise.per_sample_var must be finite")
        if np.any(v < 0.0):
            raise ValueError("Poolwise.per_sample_var must be non-negative (a variance)")
        object.__setattr__(self, "per_sample_var", v)

    def marginal_dvar_deffort(self, sigma_ii: float, n_eff: float, component: int = 0) -> float:
        """The typed D2 marginal `dΣ_ii/dn` for the MEAN at its current operating point (§1 D2/§2.3).
        `cov(n) = s²/n`, so `dΣ_ii/dn = −s²/n² = −Σ_ii/n` (one more sample). This is the EXACT
        derivative the closed-form Neyman `n_i* ∝ √(a_i/c_i)` is the KKT solution of — so equalizing
        `g_i²·|marginal_i|/c_i` reproduces today's allocation byte-for-byte on the mean case. `sigma_ii`
        is the already-divided sampling variance `s²/n` the driver holds; `n_eff` the pool size."""
        n = float(n_eff)
        if n <= 0.0:
            return 0.0
        return -float(sigma_ii) / n


@dataclass(frozen=True)
class QuantileLaw:
    """QUANTILE / MEDIAN. `cov(n) = p(1−p) / (n · f_at_q^2)`: the order-statistic law, NOT s^2/n.
    `p` in (0, 1) (the median is p=0.5); `f_at_q` is the density-at-quantile the bench estimates
    (a kernel density or a bootstrap), (k,), finite, strictly positive; `n` the reading count."""
    p: float
    f_at_q: np.ndarray
    n: int

    def __post_init__(self) -> None:
        p = float(self.p)
        if not (0.0 < p < 1.0):
            raise ValueError(f"QuantileLaw.p must be in (0, 1); got {p!r}")
        fq = _as_f64_array(self.f_at_q, "QuantileLaw.f_at_q")
        if fq.ndim != 1:
            raise ValueError(f"QuantileLaw.f_at_q must be 1-D (k,); got shape {fq.shape}")
        if not np.all(np.isfinite(fq)) or np.any(fq <= 0.0):
            raise ValueError("QuantileLaw.f_at_q must be finite and strictly positive (a density)")
        if not isinstance(self.n, (int, np.integer)) or int(self.n) < 1:
            raise ValueError(f"QuantileLaw.n must be an int >= 1; got {self.n!r}")
        object.__setattr__(self, "p", p)
        object.__setattr__(self, "f_at_q", fq)
        object.__setattr__(self, "n", int(self.n))

    def marginal_dvar_deffort(
        self, sigma_ii: float, n_eff: float, component: int = 0, readings_per_effort: float = 1.0
    ) -> float:
        """The typed D2 marginal `dΣ_ii/d(effort)` for the MEDIAN/QUANTILE at its current operating
        point (§1 D2/§2.3). The order-statistic law is `cov(n) = p(1−p)/(n·f̂²)`, so `dcov/dn = −cov/n`
        PER READING. Where one unit of the bench's effort (a `trials` tick) yields only a FRACTION of a
        pool reading — the producer/consumer benches catch ~0.1% of `trials` as readings — the
        chain rule scales it by `readings_per_effort = dn/d(effort)`: `dcov/d(effort) = (−cov/n)·(dn/d effort)`
        (default 1.0 = one reading per effort unit, the latency microbenches). `sigma_ii` is the shipped
        `cov[0,0]` the driver holds; `n_eff` the reading count (= `self.n` projected by the driver)."""
        n = float(n_eff)
        if n <= 0.0:
            return 0.0
        return -(float(sigma_ii) / n) * float(readings_per_effort)


@dataclass(frozen=True)
class RegressionLaw:
    """FIT. `cov(effort) = resid_var · XtX_inv`: more iters shrink `resid_var`, floored by the
    x-leverage in `XtX_inv` (more iters never cross it; only widening the x-design lowers
    1/Sxx). `resid_var >= 0`; `XtX_inv` is (k,k) symmetric; `design` is the x-design matrix
    (m, k); `per_point_var` is the optional per-design-point sampling SE^2 ((m,) or None) for a
    weighted-LS SE (§4.3). Structural validation only — the marginal is the driver's (deferred)."""
    resid_var: float
    XtX_inv: np.ndarray
    design: np.ndarray
    per_point_var: Union[np.ndarray, None] = None

    def __post_init__(self) -> None:
        rv = float(self.resid_var)
        if not math.isfinite(rv) or rv < 0.0:
            raise ValueError(f"RegressionLaw.resid_var must be finite and non-negative; got {rv!r}")
        xi = _as_f64_array(self.XtX_inv, "RegressionLaw.XtX_inv")
        if xi.ndim != 2 or xi.shape[0] != xi.shape[1]:
            raise ValueError(f"RegressionLaw.XtX_inv must be square (k,k); got shape {xi.shape}")
        if not np.all(np.isfinite(xi)):
            raise ValueError("RegressionLaw.XtX_inv must be finite")
        if not np.allclose(xi, xi.T, atol=_SYM_ATOL, rtol=_SYM_RTOL):
            raise ValueError("RegressionLaw.XtX_inv must be symmetric")
        dsg = _as_f64_array(self.design, "RegressionLaw.design")
        if dsg.ndim != 2:
            raise ValueError(f"RegressionLaw.design must be 2-D (m, k); got shape {dsg.shape}")
        if not np.all(np.isfinite(dsg)):
            raise ValueError("RegressionLaw.design must be finite")
        if dsg.shape[1] != xi.shape[0]:
            raise ValueError(
                f"RegressionLaw.design has {dsg.shape[1]} columns but XtX_inv is "
                f"{xi.shape[0]}x{xi.shape[0]} — the design's column count must equal k")
        ppv: Union[np.ndarray, None] = None
        if self.per_point_var is not None:
            ppv = _as_f64_array(self.per_point_var, "RegressionLaw.per_point_var")
            if ppv.ndim != 1:
                raise ValueError(
                    f"RegressionLaw.per_point_var must be 1-D (m,); got shape {ppv.shape}")
            if ppv.shape[0] != dsg.shape[0]:
                raise ValueError(
                    f"RegressionLaw.per_point_var has length {ppv.shape[0]} but the design has "
                    f"{dsg.shape[0]} points — they must match")
            if not np.all(np.isfinite(ppv)) or np.any(ppv < 0.0):
                raise ValueError("RegressionLaw.per_point_var must be finite and non-negative")
        object.__setattr__(self, "resid_var", rv)
        object.__setattr__(self, "XtX_inv", xi)
        object.__setattr__(self, "design", dsg)
        object.__setattr__(self, "per_point_var", ppv)

    def marginal_dvar_deffort(
        self, sigma_ii: float, n_eff: float, component: int = 0, iters_per_point: Optional[float] = None
    ) -> float:
        """The typed D2 marginal `dΣ_ii/d(effort)` for a FIT coefficient — the LEVERAGE FLOOR that
        DISSOLVES the conflation (§1 D2/§2.3/§4.3). The coefficient variance is
        `Σ_cc = resid_var · XtX_inv[c,c]`. More iters shrink ONLY the per-design-point MEASUREMENT-noise
        share of `resid_var`; the LACK-OF-FIT share is a fixed bias, and the x-leverage `XtX_inv[c,c]`
        (∝ 1/Sxx for the slope) is FIXED unless the bench WIDENS the x-design — a different knob the
        iter budget does not buy. So `Σ_cc` is FLOORED: it does NOT shrink as `A/n` (the false
        assumption the allocator's `Σ_ii·len(pools)`-as-`n` makes — the conflation).

          * `per_point_var` PRESENT (a weighted-LS bench): the measurement-noise share is KNOWN —
            `resid_var ≈ lack_of_fit + mean(per_point_var)`, the second term shrinking ~`1/iters`. The
            marginal is the derivative of that shrinkable share only: at `iters_per_point` iters/point,
            `d resid_var/d(iters) ≈ −mean(per_point_var)/iters²`, scaled by `XtX_inv[c,c]`. If
            lack-of-fit dominates (the measurement-noise share is a small fraction of `resid_var`), this
            is correctly ~0. The default `iters_per_point` is `n_eff` (the design-point count the driver
            holds) when not given — a conservative unit step.
          * `per_point_var` ABSENT (the common case — the bench computes only `resid_var`): we CANNOT
            split measurement noise from lack-of-fit, so the honest, ADR-0008-disciplined answer is the
            FLOOR — `~0` (un-fundable by iters). We REFUSE to assume `resid_var` is all-shrinkable
            (that re-introduces the 1/n conflation); the spec's lever for lowering `Σ_cc` is widening the
            x-design (§4.3), not an iter budget. `plain resid_var/Sxx` is a LOWER BOUND on the true slope
            variance and the marginal must not promise variance the iter-work cannot buy (§4.3)."""
        ppv = self.per_point_var
        if ppv is None:
            # Floor: an unknown measurement-noise/lack-of-fit split is treated as at-its-floor. The
            # allocator must not pour iter budget into a fit it cannot shrink (the conflation removal).
            return 0.0
        c = int(component)
        leverage = float(self.XtX_inv[c, c])  # XtX_inv[c,c] (∝ 1/Sxx for the slope) — FIXED by the design
        meas_noise = float(np.mean(ppv))      # the per-point sampling-variance share that DOES shrink
        iters = float(iters_per_point) if iters_per_point is not None else float(n_eff)
        if iters <= 0.0 or leverage <= 0.0 or meas_noise <= 0.0:
            return 0.0
        # d resid_var/d(iters) for the shrinkable share ≈ −meas_noise/iters² (the share ~ meas_noise/iters
        # at the current iters), times the FIXED leverage XtX_inv[c,c]. Lack-of-fit contributes 0.
        return -leverage * meas_noise / (iters * iters)


@dataclass(frozen=True)
class Fixed:
    """PIN / declared spread. `cov(effort) = cov` (un-shrinkable): no finite budget reduces it.
    A true constant (σ tiny) drops out of allocation (a_i≈0); a declared-spread prior (B_op's
    σ=64) still CONTRIBUTES a_i to the bound (the CI honestly rests on the prior) but gets no
    allocation (§2.3). Carries no parameters — the irreducible variance is `Estimate.cov`."""

    def marginal_dvar_deffort(self, sigma_ii: float, n_eff: float, component: int = 0) -> float:
        """The typed D2 marginal for a PIN: `dΣ_ii/d(effort) = 0` — irreducible (§2.3). No finite budget
        reduces a declared-spread prior or a deployment-fact constant, so the pin drops out of allocation
        (it still contributes its `a_i` to the bound via `gᵀΣg`, but gets no funding). This is the right
        reason the legacy `a<=0 ⇒ n*=n` branch fired — irreducible, not merely `a==0`."""
        return 0.0


@dataclass(frozen=True)
class Composed:
    """RATIO / composite. The shrink law of a delta-method composition `h(constituents)`: a
    tuple of the constituents' shrink laws. The driver recurses to the steepest constituent
    (§1 D2). `parts` must be non-empty."""
    parts: tuple["ShrinkLaw", ...]

    def __post_init__(self) -> None:
        if not isinstance(self.parts, tuple):
            raise TypeError(f"Composed.parts must be a tuple; got {type(self.parts).__name__}")
        if len(self.parts) == 0:
            raise ValueError("Composed.parts must be non-empty")
        for i, p in enumerate(self.parts):
            if not isinstance(p, _SHRINK_VARIANTS):
                raise TypeError(
                    f"Composed.parts[{i}] is not a ShrinkLaw; got {type(p).__name__}")

    def marginal_dvar_deffort(self, sigma_ii: float, n_eff: float, component: int = 0) -> float:
        """The typed D2 marginal for a RATIO/composite: recurse to the STEEPEST constituent (§1 D2) —
        the part whose variance responds MOST to effort (the most-negative `dΣ/d effort`), since funding
        the composite's effort flows to its steepest-shrinking constituent.

        APPROXIMATION (surfaced per ADR-0008 Rule 3 — a misfit named, not silent): `Composed` carries
        only the constituents' LAWS, not each constituent's own `(Σ_ii, n)` operating point, so this
        evaluates every part at the SHARED `(sigma_ii, n_eff)` the driver passes for the composite. That
        is exact only when the constituents share that operating point; for genuinely heterogeneous
        constituents it is a least-bad estimate. No current bench registers a `Composed` law (the §3
        ratio row is 'for completeness' — `dps` itself is `f`, not a registered input), so this path is
        untriggered today; a future composite-input bench should pass each constituent's own point (extend
        the signature) rather than rely on the shared-point fallback. Until then this keeps the recursion
        total and honest about its own limit."""
        marginals = [p.marginal_dvar_deffort(sigma_ii, n_eff, component) for p in self.parts]
        return min(marginals) if marginals else 0.0


# The ShrinkLaw sum type (§1 D2). A union, not a base class — the variants are independent
# frozen dataclasses (the project's sum-type idiom; the driver branches on isinstance later).
ShrinkLaw = Union[Poolwise, QuantileLaw, RegressionLaw, Fixed, Composed]
_SHRINK_VARIANTS = (Poolwise, QuantileLaw, RegressionLaw, Fixed, Composed)


# ============================================================================================
# The contract: one frozen, typed Estimate per measurable quantity (§1).
# ============================================================================================
@dataclass(frozen=True)
class Estimate:
    """ONE bench's `measure()` output: the point(s) `f` is evaluated at, their full sampling
    covariance, and the metadata the driver needs to allocate, bound, and report — uniformly
    across every estimator kind (§1). Frozen and validated at construction (ADR-0002): a
    malformed estimate RAISES in `__post_init__`, it is never coerced into a plausible-looking
    lie. The typed signature IS the contract's SSOT (ADR-0012 P8).

    Fields (all per-component fields are length k = len(theta_hat), k >= 1):
      theta_hat : (k,) float64   — the point(s) `f` is evaluated at.
      cov       : (k,k) float64  — SAMPLING covariance of theta_hat (already SE^2, NOT s^2/n);
                                    symmetric and PSD (the §1 D1 within-bench off-diagonal lives
                                    here — e.g. the OLS slope/intercept −0.81).
      names     : (k,) str       — the registry quantity each component estimates.
      shrink    : ShrinkLaw      — how cov responds to more of THIS bench's effort (replaces n).
      support   : (k,) SupportSpec — per-component feasible domain (Support or (lo, hi)).
      family    : (k,) FamilySpec  — per-component sampling-law family (CIFamily or StudentT).
      cross     : {other_name: cov} — OPTIONAL cross-bench covariance; {} = independent (default).
      kind      : str            — provenance label ('mean'|'median'|'ols_fit'|'pin'|
                                    'declared_spread'|'quantile'|'ratio'); the driver branches on
                                    NONE of it — it is for the store and the report.
    """
    theta_hat: np.ndarray
    cov: np.ndarray
    names: tuple[str, ...]
    shrink: ShrinkLaw
    support: tuple[SupportSpec, ...]
    family: tuple[FamilySpec, ...]
    cross: Mapping[str, float] = field(default_factory=dict)
    kind: str = ""

    def __post_init__(self) -> None:
        # --- theta_hat: 1-D (k,), k>=1, finite ---
        th = _as_f64_array(self.theta_hat, "Estimate.theta_hat")
        if th.ndim != 1:
            raise ValueError(f"Estimate.theta_hat must be 1-D (k,); got shape {th.shape}")
        k = th.shape[0]
        if k < 1:
            raise ValueError("Estimate.theta_hat must have k >= 1 components")
        if not np.all(np.isfinite(th)):
            raise ValueError("Estimate.theta_hat must be finite")

        # --- cov: (k,k), finite, symmetric, PSD ---
        cv = _as_f64_array(self.cov, "Estimate.cov")
        if cv.shape != (k, k):
            raise ValueError(
                f"Estimate.cov must be ({k},{k}) to match theta_hat's k={k}; got shape {cv.shape}")
        if not np.all(np.isfinite(cv)):
            raise ValueError("Estimate.cov must be finite")
        if not np.allclose(cv, cv.T, atol=_SYM_ATOL, rtol=_SYM_RTOL):
            raise ValueError("Estimate.cov must be symmetric (a sampling covariance)")
        # PSD on the symmetric part; tolerance scales with the matrix magnitude so a
        # large-variance cov is not rejected by a fixed epsilon (ADR-0002 gate, measure-first).
        sym = 0.5 * (cv + cv.T)
        eigmin = float(np.linalg.eigvalsh(sym)[0])
        psd_tol = _PSD_EIG_RTOL * max(1.0, float(np.max(np.abs(sym))))
        if eigmin < -psd_tol:
            raise ValueError(
                f"Estimate.cov must be PSD; smallest eigenvalue {eigmin:.3e} < -{psd_tol:.3e}")

        # --- names / support / family: each length k ---
        if not isinstance(self.names, tuple):
            raise TypeError(f"Estimate.names must be a tuple; got {type(self.names).__name__}")
        if len(self.names) != k:
            raise ValueError(f"Estimate.names has {len(self.names)} entries but k={k}")
        for nm in self.names:
            if not isinstance(nm, str):
                raise TypeError(f"Estimate.names entries must be str; got {type(nm).__name__}")

        if not isinstance(self.support, tuple):
            raise TypeError(f"Estimate.support must be a tuple; got {type(self.support).__name__}")
        if len(self.support) != k:
            raise ValueError(f"Estimate.support has {len(self.support)} entries but k={k}")
        for i, sp in enumerate(self.support):
            _validate_support(sp, i)

        if not isinstance(self.family, tuple):
            raise TypeError(f"Estimate.family must be a tuple; got {type(self.family).__name__}")
        if len(self.family) != k:
            raise ValueError(f"Estimate.family has {len(self.family)} entries but k={k}")
        for i, fm in enumerate(self.family):
            if not isinstance(fm, (CIFamily, StudentT)):
                raise TypeError(
                    f"Estimate.family[{i}] must be a CIFamily or StudentT; got "
                    f"{type(fm).__name__}")
            # A Student-t family MUST carry its dof — the bare enum member yields no CI
            # multiplier (ADR-0002 fail-loud; matches the from_jsonb read gate which already
            # requires 'dof' for STUDENT_T, §4.3). Use StudentT(dof), not CIFamily.STUDENT_T.
            if fm is CIFamily.STUDENT_T:
                raise ValueError(
                    f"Estimate.family[{i}] is the bare CIFamily.STUDENT_T without a dof; a "
                    f"Student-t family must carry its degrees of freedom — use StudentT(dof) "
                    f"(legitimate only for a mean dof=n-1 or an OLS coef dof=n_pts-2, §4.3)")

        # --- shrink: a ShrinkLaw variant ---
        if not isinstance(self.shrink, _SHRINK_VARIANTS):
            raise TypeError(
                f"Estimate.shrink must be a ShrinkLaw variant "
                f"({'|'.join(c.__name__ for c in _SHRINK_VARIANTS)}); "
                f"got {type(self.shrink).__name__}")
        _validate_shrink_arity(self.shrink, k)

        # --- cross: a {str: float} mapping ---
        if not isinstance(self.cross, Mapping):
            raise TypeError(f"Estimate.cross must be a Mapping; got {type(self.cross).__name__}")
        for ck, cvv in self.cross.items():
            if not isinstance(ck, str):
                raise TypeError(f"Estimate.cross keys must be str; got {type(ck).__name__}")
            if not math.isfinite(float(cvv)):
                raise ValueError(f"Estimate.cross[{ck!r}] must be finite; got {cvv!r}")

        if not isinstance(self.kind, str):
            raise TypeError(f"Estimate.kind must be a str; got {type(self.kind).__name__}")

        # Re-bind the normalized arrays (frozen dataclass → object.__setattr__).
        object.__setattr__(self, "theta_hat", th)
        object.__setattr__(self, "cov", cv)

    @property
    def k(self) -> int:
        """The number of components (= len(theta_hat))."""
        return int(self.theta_hat.shape[0])

    def is_valid(self) -> bool:
        """The §1 ADR-0002 gate as a boolean: True iff the estimate satisfies every invariant
        `__post_init__` enforces. Since construction already raises on a violation, a constructed
        `Estimate` is always valid — this re-runs the check (e.g. on a value deserialized through
        a path that bypassed the ctor) and never raises. The driver and the jsonb-read path call
        it as the explicit gate the spec names."""
        try:
            Estimate(
                theta_hat=self.theta_hat, cov=self.cov, names=self.names, shrink=self.shrink,
                support=self.support, family=self.family, cross=dict(self.cross), kind=self.kind)
            return True
        except (ValueError, TypeError):
            return False


# ============================================================================================
# Internal validators (shared by the ShrinkLaw variants and the Estimate gate).
# ============================================================================================
def _as_f64_array(x: object, label: str) -> np.ndarray:
    """Coerce an array-like to a contiguous float64 ndarray for validation/storage. This is the
    ONE permitted normalization (dtype + a defensive copy so a frozen field can't be mutated
    through an alias) — it is NOT a coercion of a malformed VALUE (a bad shape/NaN/asymmetry
    still raises downstream); ADR-0002's "validate, don't coerce" forbids fixing a wrong number,
    not casting a list to the array dtype the contract is typed over."""
    try:
        arr = np.array(x, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{label} must be array-like of float64; got {type(x).__name__}: {exc}")
    return arr


def _validate_support(sp: SupportSpec, i: int) -> None:
    if isinstance(sp, Support):
        return
    if isinstance(sp, tuple):
        if len(sp) != 2:
            raise ValueError(
                f"Estimate.support[{i}] interval must be (lo, hi); got {len(sp)} entries")
        lo, hi = float(sp[0]), float(sp[1])
        if not (math.isfinite(lo) and math.isfinite(hi)):
            raise ValueError(f"Estimate.support[{i}] interval bounds must be finite; got {sp!r}")
        if not (lo < hi):
            raise ValueError(f"Estimate.support[{i}] interval needs lo < hi; got ({lo}, {hi})")
        return
    raise TypeError(
        f"Estimate.support[{i}] must be a Support or a (lo, hi) tuple; got {type(sp).__name__}")


def _validate_shrink_arity(shrink: ShrinkLaw, k: int) -> None:
    """Cross-check a ShrinkLaw's per-component arity against the estimate's k where the law
    carries a length-k vector. (The variant's own __post_init__ has already validated the law
    in isolation; this ties it to THIS estimate's component count — ADR-0002, P1: the k has one
    home, theta_hat, and the law must agree with it.)"""
    if isinstance(shrink, Poolwise):
        if shrink.per_sample_var.shape[0] != k:
            raise ValueError(
                f"Poolwise.per_sample_var has length {shrink.per_sample_var.shape[0]} but the "
                f"estimate has k={k} components")
    elif isinstance(shrink, QuantileLaw):
        if shrink.f_at_q.shape[0] != k:
            raise ValueError(
                f"QuantileLaw.f_at_q has length {shrink.f_at_q.shape[0]} but the estimate has "
                f"k={k} components")
    elif isinstance(shrink, RegressionLaw):
        if shrink.XtX_inv.shape[0] != k:
            raise ValueError(
                f"RegressionLaw.XtX_inv is {shrink.XtX_inv.shape[0]}x{shrink.XtX_inv.shape[0]} "
                f"but the estimate has k={k} components")
    # Fixed and Composed carry no per-component vector keyed to k (a Composed's parts are the
    # constituents' laws, whose own arity is the constituent estimate's concern), so nothing to tie.


# ============================================================================================
# (De)serialization — Estimate <-> a plain JSON-able dict (the jsonb the store round-trips).
# The dict is the SERIALIZATION; the typed Estimate (P8) is the SSOT of the shape. `to_jsonb`
# and `from_jsonb` are exact inverses on a valid Estimate (the Phase-0 round-trip test). The
# schema is `{theta_hat, cov, names, shrink:{law, params}, support, family, cross, kind}` (§5).
# ============================================================================================
def to_jsonb(est: Estimate) -> dict[str, object]:
    """Serialize an `Estimate` to a plain JSON-able dict (the `benchmark_instance.estimate`
    jsonb payload, §5). Inverse of `from_jsonb`. Arrays become nested lists; the ShrinkLaw and
    the per-component support/family become tagged dicts so the sum types round-trip."""
    return {
        "theta_hat": est.theta_hat.tolist(),
        "cov": est.cov.tolist(),
        "names": list(est.names),
        "shrink": _shrink_to_dict(est.shrink),
        "support": [_support_to_obj(sp) for sp in est.support],
        "family": [_family_to_obj(fm) for fm in est.family],
        "cross": {str(kk): float(vv) for kk, vv in est.cross.items()},
        "kind": est.kind,
    }


def from_jsonb(obj: Mapping[str, object]) -> Estimate:
    """Deserialize a jsonb payload back into a validated `Estimate` (inverse of `to_jsonb`). The
    ctor re-runs the full ADR-0002 gate, so a corrupt/hand-edited payload raises here rather than
    flowing on as a malformed estimate (P2: the read boundary validates, it does not trust)."""
    if not isinstance(obj, Mapping):
        raise TypeError(f"from_jsonb expects a Mapping; got {type(obj).__name__}")
    missing = {"theta_hat", "cov", "names", "shrink", "support", "family"} - set(obj)
    if missing:
        raise ValueError(f"from_jsonb: payload missing required keys {sorted(missing)}")
    support = tuple(_support_from_obj(sp, i) for i, sp in enumerate(_as_list(obj["support"], "support")))
    family = tuple(_family_from_obj(fm, i) for i, fm in enumerate(_as_list(obj["family"], "family")))
    cross_obj = obj.get("cross", {})
    if not isinstance(cross_obj, Mapping):
        raise TypeError(f"from_jsonb: 'cross' must be a mapping; got {type(cross_obj).__name__}")
    return Estimate(
        theta_hat=np.array(obj["theta_hat"], dtype=np.float64),
        cov=np.array(obj["cov"], dtype=np.float64),
        names=tuple(str(n) for n in _as_list(obj["names"], "names")),
        shrink=_shrink_from_dict(obj["shrink"]),
        support=support,
        family=family,
        cross={str(kk): float(vv) for kk, vv in cross_obj.items()},
        kind=str(obj.get("kind", "")),
    )


def _as_list(x: object, label: str) -> list[object]:
    if not isinstance(x, (list, tuple)):
        raise TypeError(f"from_jsonb: '{label}' must be a list; got {type(x).__name__}")
    return list(x)


def _support_to_obj(sp: SupportSpec) -> object:
    if isinstance(sp, Support):
        return sp.value
    return [float(sp[0]), float(sp[1])]  # an (lo, hi) interval


def _support_from_obj(o: object, i: int) -> SupportSpec:
    if isinstance(o, str):
        try:
            return Support(o)
        except ValueError:
            raise ValueError(f"from_jsonb: support[{i}] unknown domain {o!r}")
    if isinstance(o, (list, tuple)) and len(o) == 2:
        return (float(o[0]), float(o[1]))
    raise ValueError(f"from_jsonb: support[{i}] must be a domain string or [lo, hi]; got {o!r}")


def _family_to_obj(fm: FamilySpec) -> object:
    if isinstance(fm, StudentT):
        return {"family": CIFamily.STUDENT_T.value, "dof": fm.dof}
    return {"family": fm.value}


def _family_from_obj(o: object, i: int) -> FamilySpec:
    if not isinstance(o, Mapping) or "family" not in o:
        raise ValueError(f"from_jsonb: family[{i}] must be a dict with a 'family' key; got {o!r}")
    try:
        tag = CIFamily(o["family"])
    except ValueError:
        raise ValueError(f"from_jsonb: family[{i}] unknown family {o['family']!r}")
    if tag is CIFamily.STUDENT_T:
        if "dof" not in o:
            raise ValueError(f"from_jsonb: family[{i}] STUDENT_T requires a 'dof'")
        return StudentT(dof=int(o["dof"]))
    return tag


# --- ShrinkLaw (de)serialization: a tagged dict {law, params} per §5 ---
def _shrink_to_dict(s: ShrinkLaw) -> dict[str, object]:
    if isinstance(s, Poolwise):
        return {"law": "Poolwise", "per_sample_var": s.per_sample_var.tolist()}
    if isinstance(s, QuantileLaw):
        return {"law": "QuantileLaw", "p": s.p, "f_at_q": s.f_at_q.tolist(), "n": s.n}
    if isinstance(s, RegressionLaw):
        return {
            "law": "RegressionLaw",
            "resid_var": s.resid_var,
            "XtX_inv": s.XtX_inv.tolist(),
            "design": s.design.tolist(),
            "per_point_var": (None if s.per_point_var is None else s.per_point_var.tolist()),
        }
    if isinstance(s, Fixed):
        return {"law": "Fixed"}
    if isinstance(s, Composed):
        return {"law": "Composed", "parts": [_shrink_to_dict(p) for p in s.parts]}
    raise TypeError(f"_shrink_to_dict: not a ShrinkLaw; got {type(s).__name__}")


def _shrink_from_dict(o: object) -> ShrinkLaw:
    if not isinstance(o, Mapping) or "law" not in o:
        raise ValueError(f"from_jsonb: 'shrink' must be a dict with a 'law' key; got {o!r}")
    law = o["law"]
    if law == "Poolwise":
        return Poolwise(per_sample_var=np.array(o["per_sample_var"], dtype=np.float64))
    if law == "QuantileLaw":
        return QuantileLaw(
            p=float(o["p"]), f_at_q=np.array(o["f_at_q"], dtype=np.float64), n=int(o["n"]))
    if law == "RegressionLaw":
        ppv = o.get("per_point_var", None)
        return RegressionLaw(
            resid_var=float(o["resid_var"]),
            XtX_inv=np.array(o["XtX_inv"], dtype=np.float64),
            design=np.array(o["design"], dtype=np.float64),
            per_point_var=(None if ppv is None else np.array(ppv, dtype=np.float64)),
        )
    if law == "Fixed":
        return Fixed()
    if law == "Composed":
        parts = o.get("parts", [])
        if not isinstance(parts, (list, tuple)):
            raise ValueError(f"from_jsonb: Composed 'parts' must be a list; got {parts!r}")
        return Composed(parts=tuple(_shrink_from_dict(p) for p in parts))
    raise ValueError(f"from_jsonb: unknown shrink law {law!r}")
