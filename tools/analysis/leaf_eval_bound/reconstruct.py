"""
tools/analysis/leaf_eval_bound/reconstruct.py
=============================================

The Estimate RECONSTRUCTION / PROJECTION glue — the seed/aggregate -> `Estimate` builders and the
`Estimate` -> legacy `(mean, sigma, n)` projection, lifted out of `manifest.py` (the
responsibility-refactor note's move 2). These are NOT the manifest's private internals: the runners
reach into them directly (`throughput_bound._bound` feeds each input `_estimate_from_seed`), the tell
that they are a SHARED responsibility (ADR-0012 P1/P2 — a thing two callers reach into across a
boundary is not private). Pure transforms over the `Estimate` contract: numpy + `estimate` only; they
touch NO registry, NO postgres, NO manifest state — so `manifest` imports THIS, never the reverse (the
§3 import DAG stays acyclic).

  * `_estimate_from_aggregate` — a `latest_aggregate` (mean, sigma, n) -> k=1 `Poolwise` Estimate (the
    legacy TRUST fall-back; sigma is per-sample, cov = sigma^2/n).
  * `_estimate_from_seed`      — a bench seed (mean, sigma) -> k=1 `Fixed` Estimate (cov = sigma^2
    UN-DIVIDED; DEGENERATE constant / NORMAL declared-spread prior).
  * `_project_estimate`        — an `Estimate` -> legacy `(mean, sigma, n)` (the inverse on the
    mean/seed cases; the first-component marginal for a stored fit).

FAIL LOUD (ADR-0002): a measured aggregate with n<1 RAISES (a contract violation, not a silent default).

Public Domain (The Unlicense).
"""
from __future__ import annotations

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import estimate as _est  # noqa: E402  — the harmonized Estimate contract (the type SSOT, ADR-0012 P8)


def _estimate_from_aggregate(name: str, mean: float, sigma: float, n: int, kind: str) -> "_est.Estimate":
    """Reconstruct a k=1 legacy `Estimate` from a `latest_aggregate` (mean, sigma, n) — the TRUST
    fall-back for a legacy instance that carries no stored `estimate` jsonb (spec §5/§6 Phase 1).

    The aggregate's `sigma` is the PER-SAMPLE spread (`stddev_samp`), so the SAMPLING variance is
    `sigma^2 / n` and that is what goes on `cov`'s diagonal (already divided — an SE^2, the
    contract's invariant). The shrink law is `Poolwise(per_sample_var=[sigma^2])`: the mean's
    `cov(n) = per_sample_var / n` law, with the per-sample variance (NOT the divided SE^2) carried
    so the projection recovers the 4-tuple's per-sample sigma AND n byte-for-byte.

    `kind` is the definition's declared estimator label (carried onto the Estimate for the store /
    report; the driver branches on none of it). NOTE (spec ambiguity, flagged in the Phase-1
    report): even when a quantity's declared estimator is `median`/`quantile`, the legacy aggregate
    supplies NO density-at-quantile (`f_at_q`), so a faithful `QuantileLaw` CANNOT be reconstructed
    from `(mean, sigma, n)` alone — the legacy reconstruction is ALWAYS a `Poolwise` (the order-
    statistic variance is the migrated bench's job, Phase 3). `support=POSITIVE` (every physical
    quantity in this suite is a positive latency/rate/count); `family=NORMAL` (a measured aggregate
    over n samples)."""
    nn = int(n)
    if nn < 1:
        raise ValueError(
            f"_estimate_from_aggregate({name!r}): n must be >= 1 for a measured aggregate; got {n!r} "
            f"(ADR-0002: a measured value with n<1 is a contract violation, not a silent default).")
    s2 = float(sigma) ** 2
    cov00 = s2 / nn  # the already-divided SAMPLING variance (SE^2); 0.0 when sigma==0 (the n==1 case)
    return _est.Estimate(
        theta_hat=np.array([float(mean)], dtype=np.float64),
        cov=np.array([[cov00]], dtype=np.float64),
        names=(name,),
        shrink=_est.Poolwise(per_sample_var=np.array([s2], dtype=np.float64)),
        support=(_est.Support.POSITIVE,),
        family=(_est.CIFamily.NORMAL,),
        kind=(kind or "mean"),
    )


def _estimate_from_seed(
    name: str, mean: float, sigma: float, units: str, constant: bool = False
) -> "_est.Estimate":
    """Build a `Fixed`-law k=1 `Estimate` from a bench's `get_seed()` Grounded (mean, sigma, units) —
    the SEED path (DISTRUST, or a TRUST read that fell back to the seed; spec §5/§6 Phase 1).

    A seed is a DECLARED 1-sigma spread (an engineering-judgement prior), un-shrinkable by sampling:
    the spread IS the variance, so `cov=[[sigma^2]]` directly (NOT divided by any n — a prior has no
    n). The shrink law is `Fixed()` (no finite budget reduces it; the §2.3 "drops out of allocation"
    case). `support=POSITIVE`; the projection of this Estimate is `(mean, sigma, n=0)` — exactly
    today's seed 4-tuple (a seed carries n=0).

    `constant` (the `Grounded.constant` flag, threaded from `_seed_from_module`) selects the §3 PIN
    flavor — the DEGENERATE-vs-declared-spread classification, single-homed on `Grounded.constant`:
      * `constant=False` (the default — a DECLARED-SPREAD prior, e.g. R_gen/B_op/LPD): `family=NORMAL`,
        `kind='declared_spread'` — the prior the models treat as `Normal(mean, sigma)`; it CONTRIBUTES
        its `a_i` to the bound (the CI honestly rests on it — §2.3 / §7.D).
      * `constant=True` (a TRUE CONSTANT — n_gen, a layout fact): `family=DEGENERATE`, `kind='pin'` —
        the bound treats it as ~0 (`a_i ≈ 0`, §3), exactly as the bench's `pin_estimate(constant=True)`
        `measure()` path does, so the SEED and MEASURE paths agree (P1; the σ is a display placeholder
        on an integer/fixed value, never a CI-bearing spread)."""
    s = float(sigma)
    return _est.Estimate(
        theta_hat=np.array([float(mean)], dtype=np.float64),
        cov=np.array([[s * s]], dtype=np.float64),
        names=(name,),
        shrink=_est.Fixed(),
        support=(_est.Support.POSITIVE,),
        family=(_est.CIFamily.DEGENERATE if constant else _est.CIFamily.NORMAL,),
        kind=("pin" if constant else "declared_spread"),
    )


def _project_estimate(est: "_est.Estimate") -> tuple[float, float, int]:
    """Project an `Estimate` onto the legacy `(mean, sigma, n)` — the 4-tuple's first three fields
    (the fourth, `trusted`, is the resolution path's, not the estimate's). This is the inverse of
    the two reconstructions above on the mean/seed cases (the confirmed byte-for-byte fixed point),
    and the marginal of the first component for a multi-component stored estimate (a fit):

      * mean  = theta_hat[0]                              (always — the §5 projection rule).
      * sigma : the PER-SAMPLE spread the 4-tuple carries —
          - Poolwise  -> sqrt(per_sample_var[0])         (recovers the aggregate's stddev_samp; this
                          is NOT sqrt(cov[0,0]), which is the already-divided SE — the 4-tuple's
                          sigma is the per-sample stddev every model consumes as Normal(mean, sigma)).
          - otherwise -> sqrt(cov[0,0])                  (Fixed: the declared spread, since cov=sigma^2;
                          a stored fit/quantile component: its marginal SE — the honest first-component
                          spread for a 4-tuple caller).
      * n     :
          - Poolwise  -> round(per_sample_var[0]/cov[0,0]) when cov[0,0]>0, else 1 (the sigma==0,
                          n==1 degenerate aggregate; per_sample_var==cov==0 carries no n, and n==1 is
                          the only aggregate that yields sigma==0).
          - QuantileLaw -> shrink.n                      (carried explicitly by the law).
          - otherwise (Fixed/RegressionLaw/Composed) -> 0 (a seed/pin/fit carries no sample n in the
                          legacy 4-tuple; today's seed path already returns n=0).
    """
    mean = float(est.theta_hat[0])
    cov00 = float(est.cov[0, 0])
    shrink = est.shrink
    if isinstance(shrink, _est.Poolwise):
        psv0 = float(shrink.per_sample_var[0])
        sigma = float(np.sqrt(psv0))
        n = int(round(psv0 / cov00)) if cov00 > 0.0 else 1
        return mean, sigma, n
    if isinstance(shrink, _est.QuantileLaw):
        return mean, float(np.sqrt(cov00)), int(shrink.n)
    # Fixed / RegressionLaw / Composed: the spread is the first component's marginal SE; no sample n.
    return mean, float(np.sqrt(cov00)), 0
