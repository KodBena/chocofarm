"""
tests/test_bench_iota_t_row_grounding_classification.py
=======================================================

The ADR-0008 RECLASSIFICATION of `iota_us` (the SERVE per-forward fixed cost, the staged
`run_microbatch` fit INTERCEPT) and `slope_us`/`t_row` (the per-row marginal, the fit SLOPE):
a measured-but-punted-pin labelling defect on the STATIC throughput-bound path.

THE DEFECT (the labelling register — distinct from the R_gen STALL). `bench_iota` / `bench_t_row`
already RUN the live k=2 staged-forward OLS fit and return a SHRINKABLE `RegressionLaw` Estimate
(the Phase-3 fit slice; its shape is pinned by tests/test_bench_fit_estimate_phase3.py). But their
grounding (`leaf_eval_grounding.SERVE_INTERCEPT_US` / `SERVE_SLOPE_US`) set NEITHER
`needs_measurement` (defaulting False), so the STATIC models (model_capacity / model_cycletime, which
read `Grounded.needs_measurement`) flagged iota/slope `grounded` — telling the operator NOT to measure
a runnable fit — while the MANIFEST models (model_zmq_baseline / model_cpp_inproc_port, which derive
`needs_measurement = not trusted`) correctly flagged the same physics NEEDS-SOLE-WORKLOAD. A
path-dependent classification of one quantity, and an ADR-0012 P1/P8 double-home of the
`needs_measurement` semantics (literal-on-Grounded vs not-trusted-on-manifest).

THE FIX, AS TESTS (the classification correction — NOT a Fixed->QuantileLaw flip; iota/slope are
ALREADY a live `RegressionLaw`, the design's §3 REGRESSION-fit row, not the R_gen PIN-now row):
  * `SERVE_INTERCEPT_US` / `SERVE_SLOPE_US` now carry `needs_measurement=True` and stay
    `constant=False` (a MEASURED fit, not a true constant), so the static models classify them as
    NEEDS-SOLE-WORKLOAD — single-homing the flag the manifest path already derives;
  * the slope is single-homed across BOTH static models: model_cycletime's locally-re-homed `t_row`
    DERIVES `needs_measurement` from the `G.SERVE_SLOPE_US` SSOT (closing the P1 leak where the fresh
    Grounded defaulted it False), so iota/slope classify identically on every path;
  * the benches' `_estimate_from_raw` builds a SHRINKABLE k=2 `RegressionLaw` fit Estimate (NOT a
    `Fixed` pin), the own quantity component 0, carrying the slope/intercept off-diagonal — the
    variance authority is the fit `cov`, not the seed's hand-literal sigma.

These run WITHOUT the live timed JAX forward by exercising `_estimate_from_raw` on a synthesized
per-width-median design (the §8 discipline — the Estimate SHAPE + cov are design-driven; the live
timed numerics are pinned by test_bench_fit_estimate_phase3.py and the binary/JAX-gated benches).
The estimate/bench modules live under tools/analysis/OpenTURNS/ (no __init__.py — imported by
sys.path), so this test prepends those directories.

Public Domain (The Unlicense).
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

_OT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "tools", "analysis", "OpenTURNS",
)
_BENCH = os.path.join(_OT, "benchmarks")
for _p in (_OT, _BENCH):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import estimate as E  # noqa: E402  — the contract
import leaf_eval_grounding as G  # noqa: E402  — the grounding SSOT under test
import bench_iota  # noqa: E402  — the intercept bench
import bench_t_row  # noqa: E402  — the slope bench

# The real staged-forward width sweep the fit times (bench_t_row/_measure_raw default).
DESIGN = [32, 64, 128, 192, 256, 384, 512]


def _raw(intercept: float = 94.58, slope: float = 4.317, noise: float = 2.0, seed: int = 7) -> dict:
    """A `_measure_raw()`-shaped dict over a synthesized per-width-median design (no live JAX): the
    keys bench_iota/bench_t_row `_estimate_from_raw` reads (`per_width_median_us`, `batches`). A small
    residual keeps resid_var > 0 so the full cov path is exercised (§8: the correlation is
    design-determined, so a synthesized residual is sound)."""
    rng = np.random.default_rng(seed)
    x = np.asarray(DESIGN, dtype=float)
    med = {int(B): float(intercept + slope * B + rng.normal(0.0, noise)) for B in DESIGN}
    return {"slope_us_per_row": slope, "intercept_us": intercept, "r2": 0.998,
            "per_width_median_us": med, "batches": list(DESIGN)}


# --------------------------------------------------------------------------- #
# 1. The classification fix: iota/slope are needs_measurement=True (NOT grounded), NOT constants.
# --------------------------------------------------------------------------- #
def test_iota_and_slope_grounding_is_needs_measurement_not_grounded() -> None:
    """THE DEFECT, FIXED: the grounding SSOT flags iota/slope as MEASURED quantities awaiting a
    sole-workload run (`needs_measurement=True`), NOT `grounded`/`needs_measurement=False`. They stay
    `constant=False` — a measured fit, NOT a true constant (so they do NOT get the n_gen DEGENERATE
    ~0-bound treatment; they are declared-spread priors on the seed path, a live RegressionLaw when run)."""
    for q in (G.SERVE_INTERCEPT_US, G.SERVE_SLOPE_US):
        assert q.needs_measurement is True   # the runnable-bench fix (was the default False)
        assert q.constant is False           # a measured fit, not a layout/pinning constant


def test_iota_slope_match_the_other_benchable_quantities_classification() -> None:
    """Consistency (the defect's heart): every BENCHABLE grounded quantity is needs_measurement=True —
    including g_core/R_gen whose provenance literally says MEASURED. iota/slope are no longer the lone
    runnable-bench quantities flagged grounded; only the TRUE CONSTANT n_gen is needs_measurement=False."""
    benchable = (G.GEN_PER_CORE_LEAVES, G.LEAVES_PER_DECISION, G.GEN_PER_CORE_DPS,
                 G.SERVE_IO_US, G.SERVE_FULL_BUCKET, G.MSG_PER_LEAF_US,
                 G.SERVE_INTERCEPT_US, G.SERVE_SLOPE_US)
    assert all(q.needs_measurement for q in benchable)
    assert G.N_GEN_CORES.needs_measurement is False and G.N_GEN_CORES.constant is True


def test_needs_measurement_is_single_homed_across_static_and_manifest_paths() -> None:
    """ADR-0012 P1/P8: the `needs_measurement` semantics are single-homed, so iota/slope classify
    IDENTICALLY on the STATIC path (model_capacity / model_cycletime read `Grounded.needs_measurement`)
    and would on the MANIFEST path (model_zmq_baseline / model_cpp_inproc_port derive `not trusted` — a
    seed is always not-trusted, hence True). No more path-dependent 'grounded here / NEEDS-WORKLOAD there'."""
    import model_capacity as MA
    import model_cycletime as MB
    # Static path: both quantities now flag NEEDS-SOLE-WORKLOAD (was iota/slope=False -> grounded).
    assert MA.NEEDS_MEASUREMENT["iota_us"] is True
    assert MA.NEEDS_MEASUREMENT["slope_us"] is True
    # The slope is single-homed across BOTH static models: model_cycletime's re-homed t_row DERIVES
    # its flag from the G.SERVE_SLOPE_US SSOT (the P1 leak the fix closes — a fresh Grounded defaulted
    # it False), so the slope physics is NEEDS-SOLE-WORKLOAD on the cycle-time path too.
    assert MB.NEEDS_MEASUREMENT["t_row"] is True
    assert MB.NEEDS_MEASUREMENT["t_row"] == MA.NEEDS_MEASUREMENT["slope_us"]
    # The manifest path's rule (not trusted) — a seed is always not-trusted; the same answer as static.
    assert (not False) is MA.NEEDS_MEASUREMENT["iota_us"]   # documents the equivalence


# --------------------------------------------------------------------------- #
# 2. The Estimate is a SHRINKABLE k=2 RegressionLaw fit (NOT a Fixed pin), own quantity component 0.
# --------------------------------------------------------------------------- #
def test_iota_estimate_is_a_regression_fit_not_a_fixed_pin() -> None:
    """iota's `_estimate_from_raw` builds the k=2 staged-fit `RegressionLaw` Estimate with the INTERCEPT
    component 0 (the marginal `manifest.value('iota_us')` projects) — NOT a `Fixed`/declared-spread pin
    of the seed's 12.0us hand-literal. The variance authority is the fit `cov` (resid_var + x-design)."""
    est = bench_iota._estimate_from_raw(_raw())
    assert isinstance(est.shrink, E.RegressionLaw)        # SHRINKABLE fit law — not E.Fixed
    assert not isinstance(est.shrink, E.Fixed)
    assert est.kind == "ols_fit" and est.k == 2
    assert est.names[0] == "iota_us"                      # OWN quantity component 0 (the projection)
    assert "t_row_us" in est.cross                        # the co-fit off-diagonal partner
    assert est.support == (E.Support.POSITIVE, E.Support.POSITIVE)
    assert est.is_valid()
    # the intercept reads ~94.6 (component 0), the seed's 12.0us σ is NOT the SE — the fit cov is.
    assert abs(float(est.theta_hat[0]) - 94.58) < 5.0
    assert float(np.sqrt(est.cov[0, 0])) != pytest.approx(12.0)


def test_t_row_estimate_is_a_regression_fit_with_the_slope_first() -> None:
    """t_row's `_estimate_from_raw` builds the SAME staged fit with the SLOPE component 0 (the 8 live
    `value('t_row_us')` consumers read this), NOT a `Fixed` pin of the seed's 0.5us/row hand-literal."""
    est = bench_t_row._estimate_from_raw(_raw())
    assert isinstance(est.shrink, E.RegressionLaw) and not isinstance(est.shrink, E.Fixed)
    assert est.kind == "ols_fit" and est.k == 2
    assert est.names == ("t_row_us", "iota_us")           # OWN (slope) first, partner second
    assert "iota_us" in est.cross
    assert abs(float(est.theta_hat[0]) - 4.317) < 1.0     # the slope (component 0)
    assert est.is_valid()


def test_iota_t_row_are_one_fit_with_the_same_off_diagonal() -> None:
    """iota (intercept) and t_row (slope) are literally ONE staged fit read two ways: the SAME
    off-diagonal number, each estimate ordering ITS quantity component 0 (so the first-component
    projection hands the slope-reader the slope and the intercept-reader the intercept — §4.2)."""
    raw = _raw()
    ei = bench_iota._estimate_from_raw(raw)
    et = bench_t_row._estimate_from_raw(raw)
    # the cross-term (Cov(own, partner)) is the SAME number in both orderings — one fit, two read-offs.
    assert ei.cross["t_row_us"] == pytest.approx(et.cross["iota_us"], rel=1e-12)


def test_iota_t_row_estimate_jsonb_round_trips() -> None:
    """The produced fit Estimate is an exact jsonb round-trip (the §5 SSOT serialization) — the store
    persists the RegressionLaw (resid_var/XtX_inv/design) + the off-diagonal cov losslessly."""
    for B in (bench_iota, bench_t_row):
        est = B._estimate_from_raw(_raw())
        rt = E.from_jsonb(E.to_jsonb(est))
        assert isinstance(rt.shrink, E.RegressionLaw)
        assert np.allclose(rt.theta_hat, est.theta_hat)
        assert np.allclose(rt.cov, est.cov)
        assert rt.names == est.names and rt.kind == est.kind
        assert dict(rt.cross) == pytest.approx(dict(est.cross), rel=1e-12)


# --------------------------------------------------------------------------- #
# 3. The fit marginal is the LEVERAGE FLOOR (~0), NOT a Fixed pin's 0 — the design's §4.3 posture.
# --------------------------------------------------------------------------- #
def test_fit_marginal_is_the_leverage_floor_not_the_fixed_pin_punt() -> None:
    """The classification subtlety (why this is NOT the R_gen Fixed->QuantileLaw flip): a `RegressionLaw`
    WITHOUT a weighted-LS `per_point_var` is leverage-FLOORED — its `marginal_dvar_deffort` is ~0 by the
    §4.3/§7.E conservative posture (a fit is funded by WIDENING the x-design, not by pouring iters into
    it), which is the intended honest floor, NOT the un-shrinkable `Fixed`-pin punt R_gen had. The fit
    still carries the full resid_var/XtX_inv/design + the −0.81 cov (a real fit object, not a pin)."""
    est = bench_t_row._estimate_from_raw(_raw())
    rl = est.shrink
    assert isinstance(rl, E.RegressionLaw)
    assert rl.per_point_var is None                       # no weighted-LS SE -> leverage floor
    marg = rl.marginal_dvar_deffort(float(est.cov[0, 0]), float(rl.design.shape[0]))
    assert marg == 0.0                                    # the §4.3 floor (NOT promising un-buyable variance)
    # but it is a genuine fit, not a Fixed pin: it carries resid_var + a square XtX_inv + the design.
    assert rl.resid_var > 0.0
    assert rl.XtX_inv.shape == (2, 2)
    assert rl.design.shape == (len(DESIGN), 2)


# --------------------------------------------------------------------------- #
# 4. BEHAVIORAL: RUN the live staged-forward fit (JAX-gated) — the right quantity/units, shrinkable.
# --------------------------------------------------------------------------- #
def _jax_available() -> bool:
    try:
        import jax  # noqa: F401
        import chocofarm.az.bench.bench_mlp_lowlatency  # noqa: F401
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _jax_available(), reason="jax / bench_mlp_lowlatency unavailable")
def test_live_staged_forward_fit_runs_and_is_a_valid_regression_estimate() -> None:
    """BEHAVIORAL verification (ADR-0009 — RUN it; from the test's CWD, catching a relative-path fault):
    the live `measure()` runs the production staged `run_microbatch` forward across a width sweep and
    returns a VALID k=2 `RegressionLaw` fit Estimate, the own quantity component 0, POSITIVE support. A
    tiny budget (the SHAPE is the contract; the tight ~4.317/94.58 numerics are pinned at full budget by
    test_bench_fit_estimate_phase3.py). Pin `taskset -c 0` and run sole-workload for a faithful timing."""
    et = bench_t_row.measure(batches=[32, 64, 128, 192, 256], iters=8, repeat=4)
    assert isinstance(et.shrink, E.RegressionLaw) and et.kind == "ols_fit" and et.k == 2
    assert et.names[0] == "t_row_us"
    assert et.support == (E.Support.POSITIVE, E.Support.POSITIVE)
    assert et.is_valid()
    ei = bench_iota.measure(batches=[32, 64, 128, 192, 256], iters=8, repeat=4)
    assert isinstance(ei.shrink, E.RegressionLaw) and ei.names[0] == "iota_us" and ei.is_valid()
