"""
tests/test_estimate_contract.py
===============================

Phase 0 of the §6 migration (docs/design/harmonized-estimator-interface.md): the
harmonized `Estimate` contract + its store. These tests cover

  * the ADR-0002 fail-loud construction gate (bad cov shape / non-PSD / asymmetric
    cov / names-length mismatch / support+family length / unknown shrink — all RAISE,
    never coerce),
  * each `ShrinkLaw` variant constructing (Poolwise | QuantileLaw | RegressionLaw |
    Fixed | Composed) and each rejecting its own malformed input,
  * the jsonb round-trip identity (`Estimate` -> to_jsonb -> from_jsonb -> `Estimate`),
    in-process AND, when the live control_research store is reachable, through the
    postgres `estimate` jsonb column (set_estimate / latest_estimate).

The `estimate`/`bench_store` modules live under tools/analysis/OpenTURNS/ (no
__init__.py — imported by sys.path, the same way manifest.py imports bench_store),
so this test prepends that directory to sys.path.

Public Domain (The Unlicense).
"""
from __future__ import annotations

import math
import os
import sys
import uuid

import numpy as np
import pytest

_OT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "tools", "analysis", "OpenTURNS",
)
if _OT not in sys.path:
    sys.path.insert(0, _OT)

import estimate as E  # noqa: E402  — the contract under test


# --------------------------------------------------------------------------- #
# Builders — a valid k=1 mean estimate and a valid k=2 OLS-fit estimate.
# --------------------------------------------------------------------------- #
def _mean_estimate() -> E.Estimate:
    """A k=1 mean: theta_hat=[10.0], cov=[[s^2/n]] (already divided), Poolwise, POSITIVE/NORMAL —
    the degenerate case the (theta_hat, V/n, n) sketch reduces to."""
    return E.Estimate(
        theta_hat=np.array([10.0]),
        cov=np.array([[0.25]]),
        names=("tau_io",),
        shrink=E.Poolwise(per_sample_var=np.array([4.0])),
        support=(E.Support.POSITIVE,),
        family=(E.CIFamily.NORMAL,),
        kind="mean",
    )


def _ols_estimate() -> E.Estimate:
    """A k=2 OLS fit (intercept, slope) with a genuinely PSD, strongly-negatively-correlated 2x2
    cov (the §4.2 −0.81 shape), a RegressionLaw, STUDENT_T(dof=5) families, and a non-empty cross
    + a (lo, hi) interval support — exercises every field including the optional ones."""
    var_int, var_slope = 9.0, 4.0
    corr = -0.8114
    cov_off = corr * math.sqrt(var_int * var_slope)
    cov = np.array([[var_int, cov_off], [cov_off, var_slope]])
    xs = np.array([32.0, 64.0, 128.0, 192.0, 256.0, 384.0, 512.0])
    design = np.column_stack([np.ones_like(xs), xs])      # (7, 2): [1, x]
    xtx_inv = np.linalg.inv(design.T @ design)            # (2, 2), symmetric PSD
    return E.Estimate(
        theta_hat=np.array([94.58, 4.317]),
        cov=cov,
        names=("iota", "t_row"),
        shrink=E.RegressionLaw(
            resid_var=0.5, XtX_inv=xtx_inv, design=design,
            per_point_var=np.full(xs.shape[0], 0.01)),
        support=((0.0, 1000.0), E.Support.POSITIVE),
        family=(E.StudentT(dof=5), E.StudentT(dof=5)),
        cross={"T_disp": 0.123},
        kind="ols_fit",
    )


# --------------------------------------------------------------------------- #
# 1. Valid construction + the is_valid() gate.
# --------------------------------------------------------------------------- #
def test_valid_mean_constructs_and_is_valid() -> None:
    est = _mean_estimate()
    assert est.is_valid()
    assert est.k == 1
    assert est.theta_hat.dtype == np.float64
    assert est.cov.shape == (1, 1)


def test_valid_ols_constructs_and_is_valid() -> None:
    est = _ols_estimate()
    assert est.is_valid()
    assert est.k == 2
    # the within-bench off-diagonal is carried, not dropped
    assert est.cov[0, 1] < 0.0
    assert est.cov[0, 1] == pytest.approx(est.cov[1, 0])


# --------------------------------------------------------------------------- #
# 2. Fail-loud construction gate (ADR-0002): each malformed estimate RAISES.
# --------------------------------------------------------------------------- #
def test_bad_cov_shape_raises() -> None:
    with pytest.raises((ValueError, TypeError)):
        E.Estimate(
            theta_hat=np.array([1.0, 2.0]),         # k = 2
            cov=np.array([[1.0]]),                   # but cov is 1x1
            names=("a", "b"),
            shrink=E.Fixed(),
            support=(E.Support.REAL, E.Support.REAL),
            family=(E.CIFamily.NORMAL, E.CIFamily.NORMAL),
        )


def test_non_psd_cov_raises() -> None:
    # symmetric but indefinite: eigenvalues ±1 -> not PSD.
    with pytest.raises(ValueError):
        E.Estimate(
            theta_hat=np.array([0.0, 0.0]),
            cov=np.array([[0.0, 1.0], [1.0, 0.0]]),
            names=("a", "b"),
            shrink=E.Fixed(),
            support=(E.Support.REAL, E.Support.REAL),
            family=(E.CIFamily.NORMAL, E.CIFamily.NORMAL),
        )


def test_asymmetric_cov_raises() -> None:
    with pytest.raises(ValueError):
        E.Estimate(
            theta_hat=np.array([0.0, 0.0]),
            cov=np.array([[1.0, 0.5], [0.4, 1.0]]),   # 0.5 != 0.4 -> asymmetric
            names=("a", "b"),
            shrink=E.Fixed(),
            support=(E.Support.REAL, E.Support.REAL),
            family=(E.CIFamily.NORMAL, E.CIFamily.NORMAL),
        )


def test_names_length_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        E.Estimate(
            theta_hat=np.array([1.0, 2.0]),
            cov=np.eye(2),
            names=("only_one",),                      # len 1 != k=2
            shrink=E.Fixed(),
            support=(E.Support.REAL, E.Support.REAL),
            family=(E.CIFamily.NORMAL, E.CIFamily.NORMAL),
        )


def test_support_length_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        E.Estimate(
            theta_hat=np.array([1.0, 2.0]),
            cov=np.eye(2),
            names=("a", "b"),
            shrink=E.Fixed(),
            support=(E.Support.REAL,),                # len 1 != k=2
            family=(E.CIFamily.NORMAL, E.CIFamily.NORMAL),
        )


def test_family_length_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        E.Estimate(
            theta_hat=np.array([1.0, 2.0]),
            cov=np.eye(2),
            names=("a", "b"),
            shrink=E.Fixed(),
            support=(E.Support.REAL, E.Support.REAL),
            family=(E.CIFamily.NORMAL,),              # len 1 != k=2
        )


def test_non_finite_theta_raises() -> None:
    with pytest.raises(ValueError):
        E.Estimate(
            theta_hat=np.array([np.inf]),
            cov=np.array([[1.0]]),
            names=("a",),
            shrink=E.Fixed(),
            support=(E.Support.REAL,),
            family=(E.CIFamily.NORMAL,),
        )


def test_empty_theta_raises() -> None:
    with pytest.raises(ValueError):
        E.Estimate(
            theta_hat=np.array([]),                   # k = 0
            cov=np.zeros((0, 0)),
            names=(),
            shrink=E.Fixed(),
            support=(),
            family=(),
        )


def test_bad_shrink_type_raises() -> None:
    with pytest.raises(TypeError):
        E.Estimate(
            theta_hat=np.array([1.0]),
            cov=np.array([[1.0]]),
            names=("a",),
            shrink="not_a_shrink_law",               # type: ignore[arg-type]
            support=(E.Support.REAL,),
            family=(E.CIFamily.NORMAL,),
        )


def test_shrink_arity_tied_to_k_raises() -> None:
    # a Poolwise carrying 2 per-sample-vars on a k=1 estimate is a loud mismatch.
    with pytest.raises(ValueError):
        E.Estimate(
            theta_hat=np.array([1.0]),
            cov=np.array([[1.0]]),
            names=("a",),
            shrink=E.Poolwise(per_sample_var=np.array([1.0, 2.0])),
            support=(E.Support.REAL,),
            family=(E.CIFamily.NORMAL,),
        )


def test_bad_support_interval_raises() -> None:
    with pytest.raises(ValueError):
        E.Estimate(
            theta_hat=np.array([1.0]),
            cov=np.array([[1.0]]),
            names=("a",),
            shrink=E.Fixed(),
            support=((5.0, 1.0),),                    # lo >= hi
            family=(E.CIFamily.NORMAL,),
        )


def test_bad_family_type_raises() -> None:
    with pytest.raises(TypeError):
        E.Estimate(
            theta_hat=np.array([1.0]),
            cov=np.array([[1.0]]),
            names=("a",),
            shrink=E.Fixed(),
            support=(E.Support.REAL,),
            family=("normal",),                       # type: ignore[arg-type] — a bare str, not a CIFamily
        )


def test_studentt_bad_dof_raises() -> None:
    with pytest.raises(ValueError):
        E.StudentT(dof=0)


def test_bare_studentt_family_without_dof_raises() -> None:
    # A bare CIFamily.STUDENT_T (no dof) yields no CI multiplier — the construction gate must
    # reject it and require StudentT(dof), consistent with the from_jsonb read gate (ADR-0002).
    with pytest.raises(ValueError):
        E.Estimate(
            theta_hat=np.array([1.0]),
            cov=np.array([[1.0]]),
            names=("a",),
            shrink=E.Fixed(),
            support=(E.Support.REAL,),
            family=(E.CIFamily.STUDENT_T,),           # bare enum, no dof -> must raise
        )
    # the dof-carrying form is the accepted way to declare a Student-t family.
    est = E.Estimate(
        theta_hat=np.array([1.0]),
        cov=np.array([[1.0]]),
        names=("a",),
        shrink=E.Fixed(),
        support=(E.Support.REAL,),
        family=(E.StudentT(dof=5),),
    )
    assert est.is_valid()


# --------------------------------------------------------------------------- #
# 3. Each ShrinkLaw variant constructs (and rejects its own malformed input).
# --------------------------------------------------------------------------- #
def test_poolwise_constructs() -> None:
    s = E.Poolwise(per_sample_var=np.array([4.0, 9.0]))
    assert s.per_sample_var.shape == (2,)


def test_poolwise_negative_var_raises() -> None:
    with pytest.raises(ValueError):
        E.Poolwise(per_sample_var=np.array([-1.0]))


def test_quantilelaw_constructs() -> None:
    s = E.QuantileLaw(p=0.5, f_at_q=np.array([0.02]), n=2000)
    assert s.p == 0.5 and s.n == 2000


def test_quantilelaw_bad_p_raises() -> None:
    with pytest.raises(ValueError):
        E.QuantileLaw(p=1.5, f_at_q=np.array([0.02]), n=10)


def test_quantilelaw_nonpositive_density_raises() -> None:
    with pytest.raises(ValueError):
        E.QuantileLaw(p=0.5, f_at_q=np.array([0.0]), n=10)


def test_regressionlaw_constructs() -> None:
    xs = np.array([1.0, 2.0, 3.0])
    design = np.column_stack([np.ones_like(xs), xs])
    s = E.RegressionLaw(
        resid_var=1.0, XtX_inv=np.linalg.inv(design.T @ design), design=design)
    assert s.per_point_var is None
    assert s.XtX_inv.shape == (2, 2)


def test_regressionlaw_design_column_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        E.RegressionLaw(
            resid_var=1.0,
            XtX_inv=np.eye(2),
            design=np.ones((3, 3)),                   # 3 columns but XtX_inv is 2x2
        )


def test_fixed_constructs() -> None:
    assert isinstance(E.Fixed(), E.Fixed)


def test_composed_constructs() -> None:
    s = E.Composed(parts=(E.Fixed(), E.Poolwise(per_sample_var=np.array([1.0]))))
    assert len(s.parts) == 2


def test_composed_empty_raises() -> None:
    with pytest.raises(ValueError):
        E.Composed(parts=())


def test_composed_non_shrink_part_raises() -> None:
    with pytest.raises(TypeError):
        E.Composed(parts=(E.Fixed(), "nope"))         # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# 3b. The typed D2 marginal `dΣ_ii/d(effort)` (§1 D2/§2.3) — the per-ShrinkLaw shrink rate the
# allocator consults INSTEAD of `Σ_ii·len(pools)`-as-`n` (the conflation removal). Each variant
# owns its marginal (P1/P8 the SSOT of how its variance responds to effort): a mean shrinks −V/n²,
# a median by its order-statistic 1/n law, a FLOORED fit ~0 (more iters never cross the leverage
# floor), a pin 0, a composite the steepest constituent.
# --------------------------------------------------------------------------- #
def test_poolwise_marginal_is_minus_v_over_n2() -> None:
    """MEAN (§1 D2): `dΣ_ii/dn = −s²/n² = −Σ_ii/n` — the EXACT derivative the closed-form Neyman
    `n_i* ∝ √(a_i/c_i)` is the KKT solution of (so the mean allocation is byte-for-byte). At the
    operating point Σ_ii = s²/n = 4/50, the marginal is −Σ_ii/n = −(4/50)/50 = −s²/n²."""
    s = E.Poolwise(per_sample_var=np.array([4.0]))
    n, s2 = 50.0, 4.0
    sigma_ii = s2 / n                                  # the already-divided sampling variance the driver holds
    m = s.marginal_dvar_deffort(sigma_ii, n)
    assert math.isclose(m, -sigma_ii / n, rel_tol=1e-15)
    assert math.isclose(m, -s2 / (n * n), rel_tol=1e-15)   # = −V/n² in raw terms
    assert m < 0.0                                     # a mean's variance DOES respond to effort (fundable)


def test_quantilelaw_marginal_is_order_statistic_1_over_n() -> None:
    """MEDIAN (§1 D2): the order-statistic law `cov(n)=p(1−p)/(n·f̂²)` gives `dcov/dn = −cov/n` per
    reading; with `readings_per_effort` the chain rule scales it by dn/d(effort) (the ~0.1%-of-trials
    capture). Default ratio 1.0 (a latency microbench) -> −cov/n; a 0.1% ratio scales it down 1000×."""
    s = E.QuantileLaw(p=0.5, f_at_q=np.array([0.02]), n=2000)
    cov_ii, n = 0.01, 2000.0
    m = s.marginal_dvar_deffort(cov_ii, n)
    assert math.isclose(m, -cov_ii / n, rel_tol=1e-15)
    assert m < 0.0                                     # a median responds to readings (fundable)
    # the effort->readings currency: catching 0.1% of trials as readings scales the marginal by 1e-3.
    m_frac = s.marginal_dvar_deffort(cov_ii, n, readings_per_effort=0.001)
    assert math.isclose(m_frac, (-cov_ii / n) * 0.001, rel_tol=1e-15)


def test_regressionlaw_marginal_is_floored_when_lack_of_fit_dominates() -> None:
    """FIT (§1 D2/§4.3 — the conflation's core): a `RegressionLaw` with NO `per_point_var` (the common
    case — the bench computes only `resid_var`, which mixes measurement noise with lack-of-fit) returns
    ~0: we REFUSE to assume `resid_var` is all-shrinkable (that re-introduces the 1/n conflation). The
    leverage floor `1/Sxx` is fixed; only widening the x-design lowers it, not the iter budget. So the
    fit is UN-shrinkable by iters — `marginal == 0` — exactly the de-funding the fix delivers."""
    xs = np.array([32.0, 64.0, 128.0, 192.0, 256.0, 384.0, 512.0])
    design = np.column_stack([np.ones_like(xs), xs])
    XtX_inv = np.linalg.inv(design.T @ design)
    s = E.RegressionLaw(resid_var=566.8, XtX_inv=XtX_inv, design=design)   # no per_point_var -> floored
    sigma_ii = 566.8 * float(XtX_inv[0, 0])
    m = s.marginal_dvar_deffort(sigma_ii, 1.0)
    assert m == 0.0                                    # floored: more iters never cross the leverage floor


def test_regressionlaw_marginal_is_nonzero_with_per_point_var() -> None:
    """FIT, weighted-LS branch (§4.3): WITH `per_point_var` (the bench knows the measurement-noise
    share), the marginal is the derivative of the shrinkable share only — nonzero (so a residual-limited
    fit IS fundable), but scaled by the FIXED leverage `XtX_inv[c,c]` and shrinking ~1/iters²."""
    xs = np.array([32.0, 64.0, 128.0, 192.0, 256.0, 384.0, 512.0])
    design = np.column_stack([np.ones_like(xs), xs])
    XtX_inv = np.linalg.inv(design.T @ design)
    ppv = np.array([2.0] * 7)
    s = E.RegressionLaw(resid_var=10.0, XtX_inv=XtX_inv, design=design, per_point_var=ppv)
    m = s.marginal_dvar_deffort(5e-5, 200.0, 0, 200.0)
    leverage = float(XtX_inv[0, 0])
    assert math.isclose(m, -leverage * 2.0 / (200.0 * 200.0), rel_tol=1e-12)
    assert m < 0.0                                     # a residual-limited fit responds to iters (fundable)


def test_fixed_marginal_is_zero() -> None:
    """PIN (§2.3): `dΣ_ii/d(effort) = 0` — irreducible; no finite budget reduces it, so it drops out of
    allocation (it still contributes its a_i to the bound via gᵀΣg, but gets no funding)."""
    assert E.Fixed().marginal_dvar_deffort(4096.0, 1.0) == 0.0
    assert E.Fixed().marginal_dvar_deffort(0.0, 100.0) == 0.0


def test_composed_marginal_is_the_steepest_constituent() -> None:
    """RATIO/composite (§1 D2): recurse to the STEEPEST (most-negative) constituent marginal. A Composed
    of {Fixed (0), Poolwise (−Σ/n)} returns the Poolwise marginal (the steepest)."""
    pw = E.Poolwise(per_sample_var=np.array([4.0]))
    comp = E.Composed(parts=(E.Fixed(), pw))
    sigma_ii, n = 0.08, 50.0
    m_comp = comp.marginal_dvar_deffort(sigma_ii, n)
    m_pw = pw.marginal_dvar_deffort(sigma_ii, n)
    assert m_comp == m_pw                              # the steepest (the Fixed's 0 is not the min)
    assert m_comp < 0.0


# --------------------------------------------------------------------------- #
# 4. jsonb round-trip identity (in-process, no DB).
# --------------------------------------------------------------------------- #
def _assert_estimate_equal(a: E.Estimate, b: E.Estimate) -> None:
    np.testing.assert_array_equal(a.theta_hat, b.theta_hat)
    np.testing.assert_array_equal(a.cov, b.cov)
    assert a.names == b.names
    assert a.support == b.support
    assert a.family == b.family
    assert dict(a.cross) == dict(b.cross)
    assert a.kind == b.kind
    # ShrinkLaw equality: re-serialize both laws and compare the canonical dicts.
    assert E._shrink_to_dict(a.shrink) == E._shrink_to_dict(b.shrink)


@pytest.mark.parametrize("builder", [_mean_estimate, _ols_estimate])
def test_jsonb_round_trip_identity(builder) -> None:
    est = builder()
    payload = E.to_jsonb(est)
    # payload must be plain JSON-able (no numpy types leaking through)
    import json
    json.dumps(payload)
    back = E.from_jsonb(payload)
    assert back.is_valid()
    _assert_estimate_equal(est, back)


def test_jsonb_round_trip_each_shrink_law() -> None:
    """Each ShrinkLaw variant round-trips through jsonb. A RegressionLaw is a k=2 estimate (its
    XtX_inv is 2x2, tied to k by the arity gate); the rest are k=1, so each law is paired with a
    k-matched estimate."""
    xs = np.array([1.0, 2.0, 3.0, 4.0])
    design = np.column_stack([np.ones_like(xs), xs])     # (4, 2)
    reg = E.RegressionLaw(resid_var=0.5, XtX_inv=np.linalg.inv(design.T @ design), design=design)
    # (law, k) pairs — RegressionLaw needs k=2, the others k=1.
    k1_laws = [
        E.Poolwise(per_sample_var=np.array([4.0])),
        E.QuantileLaw(p=0.5, f_at_q=np.array([0.02]), n=2000),
        E.Fixed(),
        E.Composed(parts=(E.Fixed(), E.Poolwise(per_sample_var=np.array([1.0])))),
    ]
    for law in k1_laws:
        est = E.Estimate(
            theta_hat=np.array([1.0]),
            cov=np.array([[1.0]]),
            names=("a",),
            shrink=law,
            support=(E.Support.POSITIVE,),
            family=(E.CIFamily.EMPIRICAL,),
            kind="probe",
        )
        back = E.from_jsonb(E.to_jsonb(est))
        assert E._shrink_to_dict(est.shrink) == E._shrink_to_dict(back.shrink)

    reg_est = E.Estimate(
        theta_hat=np.array([1.0, 2.0]),
        cov=np.eye(2),
        names=("a", "b"),
        shrink=reg,
        support=(E.Support.POSITIVE, E.Support.POSITIVE),
        family=(E.StudentT(dof=2), E.StudentT(dof=2)),
        kind="ols_fit",
    )
    back = E.from_jsonb(E.to_jsonb(reg_est))
    assert E._shrink_to_dict(reg_est.shrink) == E._shrink_to_dict(back.shrink)


def test_from_jsonb_rejects_corrupt_payload() -> None:
    est = _mean_estimate()
    payload = E.to_jsonb(est)
    # corrupt the cov into a non-PSD matrix; the read boundary must re-validate and raise.
    payload["cov"] = [[0.0, 1.0], [1.0, 0.0]]
    payload["theta_hat"] = [0.0, 0.0]
    payload["names"] = ["a", "b"]
    payload["support"] = ["real", "real"]
    payload["family"] = [{"family": "normal"}, {"family": "normal"}]
    with pytest.raises((ValueError, TypeError)):
        E.from_jsonb(payload)


def test_from_jsonb_missing_key_raises() -> None:
    est = _mean_estimate()
    payload = E.to_jsonb(est)
    del payload["cov"]
    with pytest.raises(ValueError):
        E.from_jsonb(payload)


# --------------------------------------------------------------------------- #
# 5. Postgres round-trip through the estimate jsonb column (skipped if DB is down).
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
def test_store_round_trip_through_jsonb_column() -> None:
    import bench_store
    bench_store.ensure_schema()  # idempotent; also applies the Phase-0 ALTERs
    name = f"_test_estimate_rt_{uuid.uuid4().hex[:12]}"
    est = _ols_estimate()
    with bench_store.connect() as conn:
        def_id = bench_store.register_definition(
            name, quantity="test", units="us", description="phase-0 round-trip probe",
            module_path="benchmarks/bench_does_not_exist.py", conn=conn)
        inst_id = bench_store.open_instance(def_id, config={"probe": True}, conn=conn)
        bench_store.set_estimate(inst_id, est, conn=conn)
        back = bench_store.latest_estimate(name, conn=conn)
        assert back is not None
        _assert_estimate_equal(est, back)
        # also pin: the estimator-text column accepts the kind without error.
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE benchmark_definition SET estimator = %s WHERE id = %s",
                (est.kind, def_id))
        conn.commit()
        # clean up the probe rows so the test leaves no residue.
        with conn.cursor() as cur:
            cur.execute("DELETE FROM benchmark_sample WHERE instance_id = %s", (inst_id,))
            cur.execute("DELETE FROM benchmark_instance WHERE id = %s", (inst_id,))
            cur.execute("DELETE FROM benchmark_definition WHERE id = %s", (def_id,))
        conn.commit()
