"""
tests/test_bench_fit_estimate_phase3.py
=======================================

§6 Phase 3 (the FIT slice) of the harmonized-estimator migration
(docs/design/harmonized-estimator-interface.md §4.2/§5/§8): the fit benches return a
real k=2 `Estimate`. The two co-fit pairs are

  * `bench_t_row` (slope) + `bench_iota` (intercept) — ONE staged
    `time = intercept + slope·rows` fit, the −0.81 slope/intercept off-diagonal,
  * `bench_cpp_inproc_port_t_row_bare_us` (slope) + `bench_t_disp` (intercept) — the
    `fully_device` fit, the second slope/intercept pair.

These tests cover the Phase-3 deliverables WITHOUT the live timed measurement (the benches
are timing-sensitive; the Estimate SHAPE + cov are exercised on the recorded 7-point design
and synthesized per-width medians — the correlation is design-determined, §8):

  * `estimators.fit_estimate` builds the k=2 fit Estimate: `cov = resid_var·(AᵀA)⁻¹` with
    `Corr(slope, intercept) = −0.8114` on the real design `[32,64,128,192,256,384,512]`, the
    three `(AᵀA)⁻¹` entries matching their closed forms, `RegressionLaw`/`StudentT(dof=5)`,
    `kind='ols_fit'`, and the partner off-diagonal mirrored into `cross`;
  * the COMPONENT-ORDERING contract — each bench orders ITS quantity as component 0 (so the
    manifest's first-component projection hands the slope-reader the slope and the
    intercept-reader the intercept — the 8 live `value("t_row_us")` consumers), the SAME fit
    in both orderings (same correlation, same off-diagonal number);
  * FAIL LOUD (ADR-0002) — a degenerate fit (too few / collinear design points) RAISES;
  * the jsonb round-trip identity of the produced fit Estimate.

A DB-gated tail exercises the full `run()` -> `set_estimate` -> `manifest.estimate()` TRUST
stored-estimate path against the live control_research store (skipped when unreachable, and
self-cleaning of its synthetic instances), proving `manifest.estimate('t_row_us')` returns the
k=2 fit Estimate (not the Phase-1 legacy Poolwise reconstruction) and the §5.2 DE-DUP holds
(only the per-width medians as provenance rows, the headline scalar NOT double-logged).

The `estimate`/`estimators`/`bench_<name>` modules live under tools/analysis/leaf_eval_bound/
(no __init__.py — imported by sys.path, the way manifest.py imports bench_store), so this
test prepends those directories to sys.path.

Public Domain (The Unlicense).
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest

_OT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "tools", "analysis", "leaf_eval_bound",
)
_BENCH = os.path.join(_OT, "benchmarks")
for _p in (_OT, _BENCH):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import estimators as BC  # noqa: E402  — the fit_estimate helper under test
import estimate as E  # noqa: E402  — the contract

# The real 7-point design the spec §8 corroborating check fixes the −0.8114 to.
DESIGN = [32, 64, 128, 192, 256, 384, 512]


def _medians(intercept: float, slope: float, noise: float = 3.0, seed: int = 7) -> list[float]:
    """Synthesize plausible per-width medians from a seed fit (intercept + slope·rows + small residual)
    so resid_var > 0 and the full cov path is exercised. The CORRELATION is design-determined (it does
    not depend on the residual), so a synthesized residual is sound for the −0.8114 target (§8)."""
    rng = np.random.default_rng(seed)
    x = np.asarray(DESIGN, dtype=float)
    return [float(v) for v in (intercept + slope * x + rng.normal(0.0, noise, x.shape))]


def _corr(cov: np.ndarray) -> float:
    return float(cov[0, 1] / math.sqrt(cov[0, 0] * cov[1, 1]))


# --------------------------------------------------------------------------- #
# 1. fit_estimate: the −0.8114 cov, the closed forms, the contract fields.
# --------------------------------------------------------------------------- #
def test_fit_estimate_correlation_is_minus_point_8114_on_the_real_design() -> None:
    """§4.2/§8: `Corr(slope, intercept) = −0.8114` on `[32,64,128,192,256,384,512]`, in BOTH co-fit
    orderings (the correlation is a property of the design, not the residual)."""
    med = _medians(94.58, 4.317)
    ei = BC.fit_estimate(DESIGN, med, own_name="iota_us", own_role="intercept", partner_name="t_row_us")
    et = BC.fit_estimate(DESIGN, med, own_name="t_row_us", own_role="slope", partner_name="iota_us")
    assert round(_corr(ei.cov), 4) == -0.8114
    assert round(_corr(et.cov), 4) == -0.8114
    # the off-diagonal (the cross-term) is the SAME number both ways — one fit, two read-offs.
    assert math.isclose(ei.cross["t_row_us"], et.cross["iota_us"], rel_tol=1e-12)


def test_fit_estimate_xtx_inv_matches_closed_forms() -> None:
    """§8 corroborating: the three `(AᵀA)⁻¹` entries match `Var(slope)=1/Sxx`, `Var(intercept)=1/n+
    x̄²/Sxx`, `Cov=−x̄/Sxx` exactly (intercept-first ordering), and `cov == resid_var·(AᵀA)⁻¹`."""
    x = np.asarray(DESIGN, dtype=float)
    n = x.shape[0]
    xbar = float(x.mean())
    Sxx = float(np.sum((x - xbar) ** 2))
    med = _medians(94.58, 4.317)
    est = BC.fit_estimate(DESIGN, med, own_name="iota_us", own_role="intercept", partner_name="t_row_us")
    rl = est.shrink
    assert isinstance(rl, E.RegressionLaw)
    # intercept-first ordering: [0,0]=Var(int)/rv, [1,1]=Var(slope)/rv, [0,1]=Cov/rv.
    assert math.isclose(rl.XtX_inv[0, 0], 1.0 / n + xbar ** 2 / Sxx, rel_tol=1e-12)
    assert math.isclose(rl.XtX_inv[1, 1], 1.0 / Sxx, rel_tol=1e-12)
    assert math.isclose(rl.XtX_inv[0, 1], -xbar / Sxx, rel_tol=1e-12)
    assert np.allclose(est.cov, rl.resid_var * rl.XtX_inv)


def test_fit_estimate_contract_fields() -> None:
    """The produced Estimate honors the §4.2 fit contract: k=2, RegressionLaw, STUDENT_T(dof=n_pts−2),
    POSITIVE support, kind='ols_fit', a PSD cov, and a valid is_valid() gate."""
    est = BC.fit_estimate(DESIGN, _medians(94.58, 4.317), own_name="iota_us", own_role="intercept",
                          partner_name="t_row_us")
    assert est.k == 2
    assert est.kind == "ols_fit"
    assert isinstance(est.shrink, E.RegressionLaw)
    assert est.shrink.design.shape == (len(DESIGN), 2)
    assert all(isinstance(f, E.StudentT) and f.dof == len(DESIGN) - 2 for f in est.family)
    assert est.support == (E.Support.POSITIVE, E.Support.POSITIVE)
    assert est.is_valid()


def test_fit_estimate_orders_own_quantity_first() -> None:
    """The component-ordering contract (§4.2 + the driver/manifest first-component projection): the
    bench's OWN quantity is component 0. So iota's estimate has the intercept first (~94.6) and t_row's
    has the slope first (~4.3) — the marginal `manifest.value` projects for each. The names tuple leads
    with the own name; the partner (and its off-diagonal in cross) is component 1."""
    med = _medians(94.58, 4.317)
    ei = BC.fit_estimate(DESIGN, med, own_name="iota_us", own_role="intercept", partner_name="t_row_us")
    et = BC.fit_estimate(DESIGN, med, own_name="t_row_us", own_role="slope", partner_name="iota_us")
    # iota: intercept first.
    assert ei.names == ("iota_us", "t_row_us")
    assert abs(ei.theta_hat[0] - 94.58) < 5.0
    assert "t_row_us" in ei.cross
    # t_row: slope first (the 8 live value("t_row_us") consumers read THIS as the mean).
    assert et.names == ("t_row_us", "iota_us")
    assert abs(et.theta_hat[0] - 4.317) < 1.0
    assert "iota_us" in et.cross


def test_fit_estimate_fails_loud_on_degenerate_fit() -> None:
    """ADR-0002: a degenerate fit RAISES, never a padded low-information estimate — fewer than 3 design
    points (no Student-t dof), a collinear x-design (Sxx=0, (AᵀA) singular), a length mismatch, a
    non-finite reading, and a bad role."""
    with pytest.raises(ValueError):  # < 3 points -> dof = n-2 < 1
        BC.fit_estimate([32, 64], [100.0, 200.0], own_name="q", own_role="slope", partner_name="p")
    with pytest.raises(ValueError):  # zero x-spread -> Sxx = 0, singular
        BC.fit_estimate([64, 64, 64], [1.0, 2.0, 3.0], own_name="q", own_role="slope", partner_name="p")
    with pytest.raises(ValueError):  # length mismatch
        BC.fit_estimate([32, 64, 128], [1.0, 2.0], own_name="q", own_role="slope", partner_name="p")
    with pytest.raises(ValueError):  # non-finite reading
        BC.fit_estimate([32, 64, 128], [1.0, float("nan"), 3.0], own_name="q", own_role="slope",
                        partner_name="p")
    with pytest.raises(ValueError):  # bad role
        BC.fit_estimate(DESIGN, _medians(94.58, 4.317), own_name="q", own_role="curvature", partner_name="p")


def test_fit_estimate_jsonb_round_trips() -> None:
    """The produced fit Estimate is an exact jsonb round-trip (the §5 SSOT serialization): every field
    — theta_hat, the full 2×2 cov, RegressionLaw (resid_var/XtX_inv/design), names, cross, kind — comes
    back identical, so the store persists the fit losslessly."""
    est = BC.fit_estimate(DESIGN, _medians(94.58, 4.317), own_name="t_row_us", own_role="slope",
                          partner_name="iota_us")
    rt = E.from_jsonb(E.to_jsonb(est))
    assert np.allclose(rt.theta_hat, est.theta_hat)
    assert np.allclose(rt.cov, est.cov)
    assert isinstance(rt.shrink, E.RegressionLaw)
    assert math.isclose(rt.shrink.resid_var, est.shrink.resid_var, rel_tol=1e-12)
    assert np.allclose(rt.shrink.XtX_inv, est.shrink.XtX_inv)
    assert np.allclose(rt.shrink.design, est.shrink.design)
    assert rt.names == est.names
    assert dict(rt.cross) == dict(est.cross)
    assert rt.kind == est.kind


# --------------------------------------------------------------------------- #
# 2. End-to-end through the live store: run() -> set_estimate -> manifest.estimate() (DB-gated).
# --------------------------------------------------------------------------- #
def _db_available() -> bool:
    try:
        import bench_store
        with bench_store.connect() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                cur.fetchone()
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _db_available(), reason="control_research postgres not reachable")
def test_run_logs_fit_estimate_and_manifest_reads_it_back() -> None:
    """End-to-end through the real store, WITHOUT the live timed measurement (`_measure_raw()` — the §6
    Phase-4 dict provenance producer `run()` consumes — monkeypatched to return the recorded fit shape;
    run() then builds the real k=2 fit Estimate from it via the un-patched `_estimate_from_raw`): each fit
    bench's run() logs a k=2 fit Estimate via set_estimate, and
    `manifest.estimate(name)` reads it back through the TRUST stored-estimate path (source
    'postgres(estimate)', NOT the Phase-1 legacy reconstruction) — with the −0.8114 cov and the OWN
    quantity as component 0. The §5.2 DE-DUP is asserted: the instance carries exactly the 7 per-width
    medians as provenance, the headline scalar is NOT double-logged. Self-cleaning of its synthetic rows."""
    import bench_store
    import manifest as M
    import bench_t_row
    import bench_iota
    import bench_t_disp
    import bench_cpp_inproc_port_t_row_bare_us as bcpp

    bench_store.ensure_schema()
    staged = {B: float(94.58 + 4.317 * B) for B in DESIGN}
    fulldev = {B: float(68.84 + 3.092 * B) for B in DESIGN}
    # add a tiny deterministic residual so resid_var > 0 (a perfect line gives resid_var = 0, a valid but
    # degenerate-variance fit; a real bench always has measurement noise).
    rng = np.random.default_rng(3)
    staged = {B: v + float(rng.normal(0, 2.0)) for B, v in staged.items()}
    fulldev = {B: v + float(rng.normal(0, 2.0)) for B, v in fulldev.items()}

    # Patch each bench's `_measure_raw` (the §6 Phase-4 dict producer run() consumes); run() then builds the
    # real fit Estimate from it via the un-patched `_estimate_from_raw` (the path under test).
    saved = (bench_t_row._measure_raw, bench_iota._measure_raw, bench_t_disp._measure_raw, bcpp._measure_raw)
    bench_t_row._measure_raw = lambda **k: {
        "slope_us_per_row": 4.317, "intercept_us": 94.58, "r2": 0.998,
        "per_width_median_us": dict(staged), "batches": list(DESIGN)}
    bench_iota._measure_raw = lambda **k: bench_t_row._measure_raw(**k)
    bench_t_disp._measure_raw = lambda **k: {
        "t_disp_us": 68.84, "intercept_us": 68.84, "slope_us_per_row": 3.092, "r2": 0.997,
        "per_width_median_us": dict(fulldev), "batches": list(DESIGN),
        "decomposition": {"dispatch_floor_us": 68.84}}
    bcpp._measure_raw = lambda **k: {
        "slope_us_per_row": 3.092, "intercept_us": 68.84, "r2": 0.997,
        "per_width_median_us": dict(fulldev), "batches": list(DESIGN), "decomposition": {}}

    inst_ids: list = []
    try:
        for mod in (bench_t_row, bench_iota, bench_t_disp, bcpp):
            mod.run()
        M.discover(force=True)

        for name, role_mean in [("t_row_us", 4.317), ("iota_us", 94.58),
                                ("T_disp_us", 68.84), ("cpp_inproc_port_t_row_bare_us", 3.092)]:
            q = M.quantity(name, trust=True)
            est = M.estimate(name, trust=True)
            assert q.source == "postgres(estimate)"      # the TRUST stored-estimate path, not legacy
            assert est.k == 2 and est.kind == "ols_fit"
            assert isinstance(est.shrink, E.RegressionLaw)
            assert round(_corr(est.cov), 4) == -0.8114    # the −0.81 carried in the stored cov
            assert abs(est.theta_hat[0] - role_mean) < 5.0       # OWN quantity is component 0
            assert abs(M.value(name, trust=True)[0] - role_mean) < 5.0   # value() projects the OWN mean

            # §5.2 DE-DUP: the latest instance has EXACTLY the 7 per-width medians, no headline scalar.
            with bench_store.connect() as c:
                with c.cursor() as cur:
                    cur.execute(
                        """SELECT i.id, i.estimate IS NOT NULL,
                                  (SELECT count(*) FROM benchmark_sample s WHERE s.instance_id = i.id),
                                  (SELECT min(value) FROM benchmark_sample s WHERE s.instance_id = i.id)
                           FROM benchmark_instance i JOIN benchmark_definition d ON d.id = i.definition_id
                           WHERE d.name = %s ORDER BY i.started_at DESC LIMIT 1""", (name,))
                    iid, has_est, nsamp, vmin = cur.fetchone()
            inst_ids.append(iid)
            assert has_est is True
            assert nsamp == len(DESIGN)        # only the design points, not the design points + the scalar
            assert vmin > 50.0                 # a slope (~3-4) would leak in as a tiny sample value
    finally:
        # restore the patched `_measure_raw`s and delete the synthetic instances (keep the definitions).
        (bench_t_row._measure_raw, bench_iota._measure_raw, bench_t_disp._measure_raw, bcpp._measure_raw) = saved
        try:
            import bench_store as _bs
            with _bs.connect() as c:
                with c.cursor() as cur:
                    for name in ("t_row_us", "iota_us", "T_disp_us", "cpp_inproc_port_t_row_bare_us"):
                        cur.execute(
                            """SELECT i.id FROM benchmark_instance i
                               JOIN benchmark_definition d ON d.id = i.definition_id
                               WHERE d.name = %s""", (name,))
                        for (iid,) in cur.fetchall():
                            cur.execute("DELETE FROM benchmark_sample WHERE instance_id = %s", (iid,))
                            cur.execute("DELETE FROM benchmark_instance WHERE id = %s", (iid,))
                c.commit()
            M.discover(force=True)
        except Exception:
            pass
