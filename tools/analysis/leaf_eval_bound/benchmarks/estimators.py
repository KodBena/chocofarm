"""
tools/analysis/leaf_eval_bound/benchmarks/estimators.py
======================================================

The §6 Phase-3 ESTIMATOR FACTORIES — pure-numpy builders that turn a bench's raw measurement
(an OLS fit / a latency pool / a declared pin) into ONE harmonized `Estimate` (the
`docs/design/harmonized-estimator-interface.md` contract). Split out of `bench_common` (the
responsibility-refactor note's move 1, ADR-0012 P3 one-owner): the Band-1/2 numpy partner of
`estimate.py`, touching NO SQL — a bench that only computes an `Estimate` (the `untrusted_drive`
`measure()` path) imports this without dragging the Postgres surface (the store glue is `harness.py`,
the pool builders `pools.py`).

  * `fit_estimate`    — k=2 OLS `time = intercept + slope·rows` fit (the −0.81-correlated 2×2 cov).
  * `median_estimate` — k=1 median of a raw pool, BOOTSTRAP median SE (§7.A), `QuantileLaw`.
  * `pin_estimate`    — k=1 `Fixed` pin, `cov=[[σ²]]` UN-DIVIDED (DEGENERATE constant / NORMAL prior).

FAIL LOUD (ADR-0002): a degenerate fit / a sub-2 or zero-spread pool / a bad pin RAISES — never a
padded low-information estimate.

Public Domain (The Unlicense).
"""
from __future__ import annotations

import os
import sys
from typing import Sequence

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)  # the leaf_eval_bound dir (holds bench_store, estimate, leaf_eval_grounding)
for _p in (_PARENT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import estimate as _est  # noqa: E402  — the harmonized Estimate contract (the type SSOT, ADR-0012 P8)


# ============================================================================================
# The §6 Phase-3 FIT slice: turn a `time = intercept + slope·rows` OLS fit over per-width medians
# into ONE k=2 harmonized `Estimate` (docs/design/harmonized-estimator-interface.md §4.2/§5).
#
# WHY THE COVARIANCE IS COMPUTED HERE (not in `_fit_line`). `_fit_line`
# (chocofarm/az/bench/bench_mlp_lowlatency.py) returns only (intercept, slope, r2) and DISCARDS
# the design matrix A and the residuals — so `Cov(slope, intercept)` (the −0.81 off-diagonal the
# whole §4.2 slice is about) is not even computed today (§5). Per ADR-0004 minimal-touch, the
# LESS-INVASIVE route the spec names (§4.2/§5) is to recompute `(AᵀA)⁻¹` and `resid_var` HERE from
# the bench's OWN `(rows, medians)` — no shared-helper change, so NO base-method-override audit and
# NO cpp/ behavioral re-verification. The fit numbers (intercept/slope) are byte-identical to
# `_fit_line`'s lstsq because the design `A = [1, rows]` is the same; this adds only the cov/SE the
# `lstsq` threw away.
#
# THE COMPONENT ORDERING (the §4.2 + driver contract, the load-bearing subtlety). The driver
# (alloc.driver.set_estimate / _assemble_sigma / reconstruct._project_estimate) reads a multi-component
# estimate's FIRST component as the input's marginal (theta_hat[0], cov[0,0]) and the off-diagonal
# via `cross[partner_name]`. The 8 live `manifest.value("t_row_us")` consumers read the SLOPE mean,
# and `value("T_disp_us")` reads the INTERCEPT mean — so the Estimate a fit bench logs must order its
# OWN quantity as component 0 (else the first-component projection hands the slope-reader the
# intercept — a silent wrong number, ADR-0002). So `own_role` selects which of {intercept, slope} is
# component 0; the full 2×2 cov (carrying the −0.81) is row/col-permuted to match, and the SAME
# off-diagonal is ALSO placed in `cross={partner_name: off}` (the form `_assemble_sigma` reads — §4.2
# "the slope/intercept pairing a Phase-3 bench populates"). Both fit partners thus log the SAME fit
# (same correlation, same information), each ordered so ITS quantity projects correctly.
# ============================================================================================
def fit_estimate(
    rows: Sequence[float],
    medians_us: Sequence[float],
    *,
    own_name: str,
    own_role: str,
    partner_name: str,
) -> "_est.Estimate":
    """Build the k=2 OLS-fit `Estimate` for a `time = intercept + slope·rows` fit, with `own_role`'s
    component (∈ {'intercept', 'slope'}) placed FIRST so the manifest's first-component projection
    gives THIS quantity its right mean (§4.2). Carries the full 2×2 `cov = resid_var·(AᵀA)⁻¹` (the
    −0.81 off-diagonal), `shrink=RegressionLaw(resid_var, XtX_inv, design)`, `family=StudentT(dof=
    n_pts−2)` per component, POSITIVE support, `cross={partner_name: off_diagonal}`, `kind='ols_fit'`.

    FAIL LOUD (ADR-0002): a degenerate fit RAISES — fewer than 3 design points (dof = n−2 < 1, no
    Student-t), a collinear/zero-spread x-design (`Sxx == 0`, `(AᵀA)` singular), or any non-finite
    input. It NEVER returns a padded low-information estimate."""
    if own_role not in ("intercept", "slope"):
        raise ValueError(f"fit_estimate: own_role must be 'intercept' or 'slope'; got {own_role!r}")
    x = np.asarray(rows, dtype=np.float64)
    y = np.asarray(medians_us, dtype=np.float64)
    if x.ndim != 1 or y.ndim != 1 or x.shape != y.shape:
        raise ValueError(
            f"fit_estimate: rows {x.shape} and medians_us {y.shape} must be 1-D and the same length")
    n = int(x.shape[0])
    if n < 3:
        raise ValueError(
            f"fit_estimate({own_name!r}): need >= 3 design points for an OLS fit with a Student-t SE "
            f"(dof = n_pts - 2 >= 1); got n={n} (ADR-0002: a degenerate fit raises, it is not padded).")
    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
        raise ValueError(f"fit_estimate({own_name!r}): rows/medians must be finite (ADR-0002).")
    Sxx = float(np.sum((x - x.mean()) ** 2))
    if not (Sxx > 0.0):
        raise ValueError(
            f"fit_estimate({own_name!r}): the x-design has zero spread (Sxx={Sxx}); the slope SE is "
            f"undefined and (AᵀA) is singular (ADR-0002: a collinear design is a loud fault).")

    # The SAME design `_fit_line` uses: columns [1, rows] -> coeffs [intercept, slope]. So intercept is
    # index 0 and slope is index 1 in the NATURAL (fit) ordering. theta_hat/cov are computed here, then
    # permuted to put `own_role` first.
    A = np.vstack([np.ones_like(x), x]).T            # (n, 2): [1, x]
    AtA = A.T @ A
    XtX_inv_nat = np.linalg.inv(AtA)                 # (2, 2) symmetric — (AᵀA)⁻¹, the unscaled cov
    (intercept, slope), *_ = np.linalg.lstsq(A, y, rcond=None)
    yhat = intercept + slope * x
    ss_res = float(np.sum((y - yhat) ** 2))
    # The unbiased residual variance s² = SS_res / (n − p), p = 2 (intercept + slope). cov = s²·(AᵀA)⁻¹
    # is the textbook OLS sampling covariance (already an SE², the contract's invariant — §1 D1).
    resid_var = ss_res / float(n - 2)
    cov_nat = resid_var * XtX_inv_nat                # (2, 2): the FULL sampling cov, off-diagonal carried

    theta_nat = np.array([intercept, slope], dtype=np.float64)
    names_nat = ("intercept", "slope")               # role labels for the permutation, not registry names

    # Permute so `own_role` is component 0 (the marginal the manifest/driver project). A swap when the
    # bench's own quantity is the SLOPE (t_row / t_row_bare); identity when it is the INTERCEPT
    # (iota / T_disp). The permutation reorders theta_hat, cov (rows AND cols), the design columns, and
    # the per-component (name, units, family) together so every per-component field stays aligned.
    perm = (0, 1) if names_nat[0] == own_role else (1, 0)
    theta = theta_nat[list(perm)]
    cov = cov_nat[np.ix_(perm, perm)]
    XtX_inv = XtX_inv_nat[np.ix_(perm, perm)]
    design = A[:, list(perm)]                         # design columns match the (now-permuted) coeff order
    off = float(cov[0, 1])                            # Cov(own, partner) — the −0.81-correlated cross-term
    dof = n - 2                                       # the OLS-coefficient Student-t dof (§4.3)
    # `own` is component 0 by construction (the permutation above), the partner is component 1; so
    # `names = (own_name, partner_name)` regardless of perm.
    return _est.Estimate(
        theta_hat=theta,
        cov=cov,
        names=(own_name, partner_name),
        shrink=_est.RegressionLaw(resid_var=resid_var, XtX_inv=XtX_inv, design=design),
        support=(_est.Support.POSITIVE, _est.Support.POSITIVE),
        family=(_est.StudentT(dof=dof), _est.StudentT(dof=dof)),
        cross={partner_name: off},
        kind="ols_fit",
    )


# The §6 Phase-3 MEDIAN/LATENCY slice: turn a raw pool of per-cycle/per-trial/per-window readings
# (whose headline is `np.median(pool)`) into ONE k=1 harmonized `Estimate`
# (docs/design/harmonized-estimator-interface.md §3 MEDIAN/QUANTILE row, §7.A).
#
# WHY THE SE IS A BOOTSTRAP, NOT THE ASYMPTOTIC `p(1−p)/(n·f̂²)` (§7.A, the review refinement).
# The order-statistic variance of a sample median IS `p(1−p)/(n·f̂(median)²)`, but `f̂(median)` — the
# density AT the quantile — is small-sample-FRAGILE (a kernel-density estimate at a single point is
# the noisiest thing one can read off a finite latency pool, and timing data is right-skewed/heavy-
# tailed: the benches compute a median precisely BECAUSE the mean is tail-poisoned). So the spec
# §7.A says PREFER a bootstrap median SE — steadier at the bench `n` — and declare `family=EMPIRICAL`
# (a sample quantile's CI is NOT a t-interval, §4.3; fabricating an asymptotic Normal SE here would
# be the over-claim ADR-0009 forbids). The bootstrap SE is the VARIANCE AUTHORITY in `cov`.
#
# WHY `QuantileLaw.f_at_q` IS BACK-DERIVED FROM THE BOOTSTRAP SE (not an independent KDE). The
# contract's `shrink=QuantileLaw(p, f_at_q, n)` carries `f_at_q` for the driver's allocation
# MARGINAL hook (`cov(n) = p(1−p)/(n·f_at_q²)` — how one more reading shrinks the variance). To keep
# the law SELF-CONSISTENT with the `cov` it ships beside (P1 single-home: the variance has ONE home,
# the bootstrap, and the law must agree with it rather than assert a second number), `f_at_q` is the
# value that makes the law's implied variance EQUAL the bootstrap variance at the current `n`:
# `f̂ = sqrt(p(1−p)/(n·Var_boot))`. So `cov` is the bootstrap SSOT and `QuantileLaw` reproduces it
# (NOT a fabricated independent asymptotic) — and the marginal `d cov/d n = −cov/n` is the honest
# 1/n shrink the order-statistic law gives, with the bootstrap-calibrated density. (If a bench ever
# prefers a true KDE `f̂`, it passes it; absent one, the bootstrap-calibrated value is the least-bad
# honest choice and is labelled EMPIRICAL, never NORMAL.)
# ============================================================================================
def median_estimate(
    pool: Sequence[float],
    *,
    name: str,
    n_boot: int = 2000,
    boot_seed: int = 0,
) -> "_est.Estimate":
    """Build the k=1 median `Estimate` for a raw latency/cost pool: `theta_hat=[median(pool)]`,
    `cov=[[median_SE²]]` with a BOOTSTRAP median SE (§7.A — NOT the small-sample-fragile asymptotic
    `p(1−p)/(n·f̂²)`), `shrink=QuantileLaw(p=0.5, f_at_q=[f̂], n=len(pool))` (the `f̂` back-derived from
    the bootstrap SE so the law's implied variance equals `cov`), `family=(EMPIRICAL,)`, POSITIVE
    support, `kind='median'`. The bootstrap resamples the pool WITH REPLACEMENT `n_boot` times and
    takes the std (ddof=1) of the resampled medians — a deterministic SE under the fixed `boot_seed`.

    FAIL LOUD (ADR-0002): a pool that cannot yield a defensible variance RAISES, never pads — an empty
    pool, fewer than 2 readings (no bootstrap spread), any non-finite reading, or a DEGENERATE pool
    whose bootstrap medians are all identical (SE == 0, so the order-statistic density `f̂` would be
    infinite — a real fault: a zero-spread latency pool is not a measured median, it is a constant
    masquerading as one, which the spec's MEDIAN row does not cover; surface it, do not fabricate a
    QuantileLaw the data cannot support)."""
    p = np.asarray(pool, dtype=np.float64)
    if p.ndim != 1:
        raise ValueError(f"median_estimate({name!r}): pool must be 1-D; got shape {p.shape}")
    n = int(p.shape[0])
    if n < 2:
        raise ValueError(
            f"median_estimate({name!r}): need >= 2 readings for a bootstrap median SE; got n={n} "
            f"(ADR-0002: a 1-sample 'median' has no defensible variance — it raises, it is not padded).")
    if not np.all(np.isfinite(p)):
        raise ValueError(f"median_estimate({name!r}): pool readings must be finite (ADR-0002).")
    med = float(np.median(p))
    # Bootstrap the median SE: n_boot resamples of size n, with replacement; the SE is the std of the
    # resampled medians. Steadier than the asymptotic density estimate at the bench n (§7.A).
    rng = np.random.default_rng(boot_seed)
    idx = rng.integers(0, n, size=(int(n_boot), n))
    boot_meds = np.median(p[idx], axis=1)
    median_se = float(np.std(boot_meds, ddof=1))
    if not (median_se > 0.0) or not np.isfinite(median_se):
        raise ValueError(
            f"median_estimate({name!r}): the bootstrap median SE is {median_se} (the pool's bootstrap "
            f"medians are degenerate — a zero-spread pool is a constant, not a measured median). "
            f"(ADR-0002: a median with no defensible spread raises; use a Fixed pin for a true constant.)")
    var_med = median_se * median_se
    # Back-derive the density-at-median f̂ so QuantileLaw's implied variance `p(1−p)/(n·f̂²)` EQUALS the
    # bootstrap var (the law agrees with the cov SSOT, it does not assert an independent asymptotic).
    f_at_q = float(np.sqrt(0.25 / (n * var_med)))  # p(1−p) = 0.25 at p=0.5
    return _est.Estimate(
        theta_hat=np.array([med], dtype=np.float64),
        cov=np.array([[var_med]], dtype=np.float64),
        names=(name,),
        shrink=_est.QuantileLaw(p=0.5, f_at_q=np.array([f_at_q], dtype=np.float64), n=n),
        support=(_est.Support.POSITIVE,),
        family=(_est.CIFamily.EMPIRICAL,),
        kind="median",
    )


# ============================================================================================
# The §6 Phase-3 PIN slice: a declared constant / declared engineering-judgement spread into ONE k=1
# `Fixed` `Estimate` (docs/design/harmonized-estimator-interface.md §3 PIN rows, §1 D2/D4).
#
# THE TWO PIN FLAVORS (the §3 distinction the family encodes). A TRUE CONSTANT (n_gen, a deployment
# pinning fact, σ tiny) is `family=DEGENERATE` (no sampling interval; a_i≈0, drops out of allocation)
# and `kind='pin'`. A DECLARED-SPREAD PRIOR (B_op's σ=64, LPD's σ=25, R_gen's σ=8 — an engineering-
# judgement 1-sigma the manifest seeds) is `family=NORMAL` (a prior the models consume as
# `Normal(mean, sigma)` — matching `reconstruct._estimate_from_seed`) and `kind='declared_spread'`: it
# CONTRIBUTES its a_i to the bound (so the CI honestly rests on the prior) but is un-shrinkable by
# sampling (`shrink=Fixed()`; no finite budget reduces an engineering-judgement prior). EITHER way the
# variance is the declared spread UN-DIVIDED — `cov=[[σ²]]`, NOT σ²/n (a prior has no n) — which is
# the §5 store-bug fix: `stddev_samp` over one logged value returns NULL→0, DISCARDING the declared σ
# (B_op's 64 today lives only in `Grounded`/the seed path, never reaching the instance row's variance).
# ============================================================================================
def pin_estimate(
    value: float,
    sigma: float,
    *,
    name: str,
    constant: bool = False,
) -> "_est.Estimate":
    """Build the k=1 `Fixed` `Estimate` for a pin: `theta_hat=[value]`, `cov=[[sigma²]]` (the declared
    spread UN-DIVIDED — un-shrinkable, a prior has no n), `shrink=Fixed()`, POSITIVE support. A TRUE
    CONSTANT (`constant=True`, e.g. n_gen's layout fact) is `family=DEGENERATE` / `kind='pin'`; a
    DECLARED-SPREAD PRIOR (`constant=False`, the default — B_op σ=64, LPD σ=25, …) is `family=NORMAL`
    / `kind='declared_spread'` (the prior the models treat as `Normal(mean, sigma)`).

    FAIL LOUD (ADR-0002): a non-finite value/sigma, or a negative sigma, RAISES. A declared-spread pin
    with `sigma == 0` is also a fault (`family=NORMAL` with zero variance is a contradiction — a prior
    with no spread is a true constant, so pass `constant=True`); a true constant MAY carry `sigma == 0`
    (a `DEGENERATE` zero-variance point is valid). It NEVER coerces a bad spread into a plausible one."""
    v, s = float(value), float(sigma)
    if not (np.isfinite(v) and np.isfinite(s)):
        raise ValueError(f"pin_estimate({name!r}): value and sigma must be finite; got ({v}, {s}) (ADR-0002).")
    if s < 0.0:
        raise ValueError(f"pin_estimate({name!r}): sigma must be >= 0 (a spread); got {s} (ADR-0002).")
    if not constant and not (s > 0.0):
        raise ValueError(
            f"pin_estimate({name!r}): a declared-spread prior (family=NORMAL) needs sigma > 0; got {s}. "
            f"A spread-less prior is a true constant — pass constant=True for family=DEGENERATE (ADR-0002).")
    return _est.Estimate(
        theta_hat=np.array([v], dtype=np.float64),
        cov=np.array([[s * s]], dtype=np.float64),
        names=(name,),
        shrink=_est.Fixed(),
        support=(_est.Support.POSITIVE,),
        family=(_est.CIFamily.DEGENERATE if constant else _est.CIFamily.NORMAL,),
        kind=("pin" if constant else "declared_spread"),
    )
