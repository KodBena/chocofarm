"""
tests/test_manifest_estimate_seam.py
====================================

§6 Phase 1 of the harmonized-estimator migration
(docs/design/harmonized-estimator-interface.md §5/§6): the MANIFEST as the
`Estimate` seam. `manifest.quantity()` now carries an `estimate.Estimate` ALONGSIDE
the legacy (mean, sigma, n, trusted) 4-tuple, and the 4-tuple is a PROJECTION of that
estimate — additive, ZERO behavior change to the existing 4-tuple callers.

These tests cover the three Phase-1 deliverables:

  * the LEGACY reconstruction is the EXACT inverse of the 4-tuple projection on the
    mean case — `_project_estimate(_estimate_from_aggregate(mean, sigma, n)) ==
    (mean, sigma, n)` byte-for-byte (the confirmed fixed point), AND the full
    `quantity()` TRUST path returns the same 4-tuple today's code did (a pool-fed
    caller and an Estimate-fed caller agree),
  * the SEED `Fixed`-law path — `get_seed()` -> a Fixed-law Estimate whose cov is the
    declared spread^2 and whose projection is (mean, sigma, n=0),
  * the round-trip on a quantity that carries a STORED Estimate — the manifest reads
    `bench_store.latest_estimate` and projects its first component to the 4-tuple.

The pure-function and monkeypatched-`quantity()` tests run WITHOUT a DB (the
deterministic core of the seam). An optional tail exercises the real
`bench_store.latest_estimate`/`latest_aggregate` through `quantity()` against the live
control_research store (skipped when it is unreachable), mirroring the Phase-0 test.

The `estimate`/`manifest`/`bench_store` modules live under tools/analysis/OpenTURNS/
(no __init__.py — imported by sys.path, the way manifest.py imports bench_store), so
this test prepends that directory to sys.path.

Public Domain (The Unlicense).
"""
from __future__ import annotations

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

import estimate as E  # noqa: E402  — the contract
import manifest as M  # noqa: E402  — the Phase-1 seam under test


# The real seed numbers + a spread of plausible measured aggregates (mean, sigma, n). The n==1
# cases exercise the degenerate stddev_samp==0 reconstruction (cov00==0 -> projection recovers n=1).
_AGG_CASES = [
    (428.28, 12.0, 200),
    (20.0, 12.0, 2000),
    (4.317, 0.5, 7),
    (152.0, 8.0, 30),
    (68.84, 2.0, 50),
    (99.9, 0.0, 1),     # degenerate: a single reading -> sigma 0, n 1
    (256.0, 64.0, 1),   # a single reading with a (here large) value; cov00 = 4096
]

_SEED_CASES = [
    ("B_op", 256.0, 64.0, "rows/forward"),
    ("tau_io_us", 20.0, 12.0, "us"),
    ("n_gen", 3.0, 0.05, "cores"),
    ("LPD", 500.0, 25.0, "leaves/decision"),
]


# --------------------------------------------------------------------------- #
# 1. The legacy reconstruction <-> 4-tuple projection is an EXACT inverse (the fixed point).
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("mean,sigma,n", _AGG_CASES)
def test_legacy_reconstruction_projects_back_byte_for_byte(mean, sigma, n) -> None:
    """`_estimate_from_aggregate` -> `_project_estimate` recovers (mean, sigma, n) byte-for-byte:
    the per-sample sigma comes back from `Poolwise.per_sample_var` (NOT sqrt(cov00), which is the
    already-divided SE), and n from `round(per_sample_var/cov00)`. This is the confirmed fixed point
    the §6 Phase-1 4-tuple must reduce to."""
    est = M._estimate_from_aggregate("q", mean, sigma, n, "mean")
    # the reconstruction's invariants: a k=1 Poolwise mean, cov = sigma^2/n (already divided).
    assert est.k == 1
    assert isinstance(est.shrink, E.Poolwise)
    assert est.shrink.per_sample_var[0] == sigma ** 2
    assert est.cov[0, 0] == (sigma ** 2) / n
    assert est.family == (E.CIFamily.NORMAL,)
    # the projection is the EXACT inverse (byte-for-byte == on every field).
    pm, ps, pn = M._project_estimate(est)
    assert pm == mean
    assert ps == sigma
    assert pn == n


def test_legacy_reconstruction_rejects_bad_n() -> None:
    """n < 1 on a 'measured' aggregate is a loud ADR-0002 violation (a measured value has at least one
    reading), not a silent default."""
    with pytest.raises(ValueError):
        M._estimate_from_aggregate("q", 1.0, 1.0, 0, "mean")


@pytest.mark.parametrize("mean,sigma,n", _AGG_CASES)
def test_quantity_trust_legacy_path_matches_todays_4tuple(monkeypatch, mean, sigma, n) -> None:
    """The FULL `quantity()` TRUST path, when an instance has samples but no stored estimate,
    reconstructs from the aggregate and returns the SAME (mean, sigma, n, trusted=True) 4-tuple
    today's code returned — and carries the Estimate whose projection IS that 4-tuple. DB-free via
    monkeypatch (the deterministic core of the seam)."""
    # postgres "up", no stored estimate, a legacy aggregate present, a definition with units + kind.
    monkeypatch.setattr(M, "postgres_available", lambda: True)
    monkeypatch.setattr(
        M, "_definition", lambda nm: {"module_path": "benchmarks/bench_tau_io.py",
                                      "units": "us", "estimator": "mean"})

    class _FakeStore:
        @staticmethod
        def latest_estimate(nm, *a, **k):
            return None  # no stored estimate -> the legacy reconstruction branch

        @staticmethod
        def latest_aggregate(nm, *a, **k):
            return (mean, sigma, n)

    monkeypatch.setitem(sys.modules, "bench_store", _FakeStore)

    q = M.quantity("tau_io_us", trust=True)
    # the legacy 4-tuple is byte-for-byte what the pre-Phase-1 code returned.
    assert q.as_tuple() == (mean, sigma, n, True)
    assert q.source == "postgres"
    # and the carried Estimate projects to exactly that 4-tuple (the pool-fed / Estimate-fed agreement).
    assert q.estimate is not None
    assert M._project_estimate(q.estimate) == (mean, sigma, n)
    # value() (unchanged signature) is the projection-backed 4-tuple.
    assert M.value("tau_io_us", trust=True) == (mean, sigma, n, True)
    # estimate() exposes the same object.
    assert M.estimate("tau_io_us", trust=True).theta_hat[0] == mean


# --------------------------------------------------------------------------- #
# 2. The SEED Fixed-law path.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name,mean,sigma,units", _SEED_CASES)
def test_seed_fixed_law_estimate_and_projection(name, mean, sigma, units) -> None:
    """`_estimate_from_seed` builds a Fixed-law k=1 Estimate: cov = sigma^2 (the declared spread IS
    the variance, un-divided — a prior has no n), family NORMAL. Its projection is (mean, sigma, 0)
    — exactly today's seed 4-tuple."""
    est = M._estimate_from_seed(name, mean, sigma, units)
    assert est.k == 1
    assert isinstance(est.shrink, E.Fixed)
    assert est.cov[0, 0] == sigma ** 2            # the declared spread^2, NOT divided by any n
    assert est.theta_hat[0] == mean
    assert est.family == (E.CIFamily.NORMAL,)
    assert est.kind == "declared_spread"
    assert M._project_estimate(est) == (mean, sigma, 0)


@pytest.mark.parametrize("name,mp,mean,sigma", [
    ("B_op", "benchmarks/bench_b_op.py", 256.0, 64.0),
    ("tau_io_us", "benchmarks/bench_tau_io.py", 20.0, 12.0),
])
def test_quantity_seed_path_via_module_path(name, mp, mean, sigma) -> None:
    """The DISTRUST (trust=False) path through a real bench module's get_seed() (resolved by
    module_path, no DB registry): the 4-tuple is the unchanged seed (mean, sigma, 0, False), and the
    carried Estimate is the Fixed-law one whose projection IS that 4-tuple."""
    q = M.quantity(name, trust=False, module_path=mp)
    assert q.as_tuple() == (mean, sigma, 0, False)
    assert q.source == "seed"
    assert isinstance(q.estimate.shrink, E.Fixed)
    assert q.estimate.cov[0, 0] == sigma ** 2
    assert M._project_estimate(q.estimate) == (mean, sigma, 0)
    # value() / estimate() agree with quantity() (all three resolve the same object).
    assert M.value(name, trust=False, module_path=mp) == (mean, sigma, 0, False)
    assert M.estimate(name, trust=False, module_path=mp).theta_hat[0] == mean


# --------------------------------------------------------------------------- #
# 3. A quantity that carries a STORED Estimate — the manifest reads it and projects.
# --------------------------------------------------------------------------- #
def _stored_mean_estimate() -> E.Estimate:
    """A k=1 stored Poolwise mean Estimate (theta_hat=[10], cov=[[s^2/n]], per_sample_var=[s^2])."""
    s, n = 2.0, 16
    return E.Estimate(
        theta_hat=np.array([10.0]),
        cov=np.array([[s * s / n]]),
        names=("q",),
        shrink=E.Poolwise(per_sample_var=np.array([s * s])),
        support=(E.Support.POSITIVE,),
        family=(E.CIFamily.NORMAL,),
        kind="mean",
    )


def _stored_fit_estimate() -> E.Estimate:
    """A k=2 stored OLS-fit Estimate (the §4.2 slope/intercept shape) — its 4-tuple projection is the
    FIRST component's marginal (mean = theta_hat[0], sigma = sqrt(cov[0,0]), n = 0)."""
    import math
    var_int, var_slope, corr = 9.0, 4.0, -0.8114
    off = corr * math.sqrt(var_int * var_slope)
    xs = np.array([32.0, 64.0, 128.0, 192.0, 256.0, 384.0, 512.0])
    design = np.column_stack([np.ones_like(xs), xs])
    return E.Estimate(
        theta_hat=np.array([94.58, 4.317]),
        cov=np.array([[var_int, off], [off, var_slope]]),
        names=("iota", "t_row"),
        shrink=E.RegressionLaw(
            resid_var=0.5, XtX_inv=np.linalg.inv(design.T @ design), design=design),
        support=(E.Support.POSITIVE, E.Support.POSITIVE),
        family=(E.StudentT(dof=5), E.StudentT(dof=5)),
        kind="ols_fit",
    )


def test_quantity_trust_prefers_stored_estimate_mean(monkeypatch) -> None:
    """When an instance carries a stored Estimate, the TRUST path PREFERS it (over the aggregate) and
    projects it to the 4-tuple. For a stored Poolwise mean the projection recovers (10, 2, 16)."""
    stored = _stored_mean_estimate()
    monkeypatch.setattr(M, "postgres_available", lambda: True)
    monkeypatch.setattr(M, "_definition", lambda nm: {"units": "us", "estimator": "mean"})

    class _FakeStore:
        @staticmethod
        def latest_estimate(nm, *a, **k):
            return stored

        @staticmethod
        def latest_aggregate(nm, *a, **k):
            raise AssertionError("latest_aggregate must NOT be consulted when a stored estimate exists")

    monkeypatch.setitem(sys.modules, "bench_store", _FakeStore)

    q = M.quantity("q", trust=True)
    assert q.source == "postgres(estimate)"
    assert q.trusted is True
    assert q.estimate is stored                       # the exact stored object is carried through
    assert q.as_tuple() == (10.0, 2.0, 16, True)      # the projection of the stored Poolwise mean
    assert M.estimate("q", trust=True) is stored


def test_quantity_trust_stored_fit_projects_first_component(monkeypatch) -> None:
    """A stored multi-component (fit) Estimate projects its FIRST component's marginal to the 4-tuple:
    mean = theta_hat[0] = 94.58, sigma = sqrt(cov[0,0]) = 3.0, n = 0 (a fit carries no sample n). The
    full k=2 Estimate (with the −0.81 off-diagonal) is still carried for an Estimate-capable caller."""
    stored = _stored_fit_estimate()
    monkeypatch.setattr(M, "postgres_available", lambda: True)
    monkeypatch.setattr(M, "_definition", lambda nm: {"units": "us", "estimator": "ols_fit"})

    class _FakeStore:
        @staticmethod
        def latest_estimate(nm, *a, **k):
            return stored

    monkeypatch.setitem(sys.modules, "bench_store", _FakeStore)

    q = M.quantity("iota", trust=True)
    assert q.mean == 94.58
    assert q.sigma == pytest.approx(3.0)               # sqrt(var_int=9)
    assert q.n == 0
    assert q.trusted is True
    # the carried Estimate retains the full 2x2 cov (the off-diagonal the 4-tuple cannot express).
    assert q.estimate.k == 2
    assert q.estimate.cov[0, 1] < 0.0


# --------------------------------------------------------------------------- #
# 4. The Estimate seam round-trips through the LIVE store (skipped when DB is down).
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
def test_manifest_reads_stored_estimate_through_postgres() -> None:
    """End-to-end through the real store: register a probe quantity, open an instance, set a stored
    Estimate, and confirm `manifest.quantity()` reads it back (source 'postgres(estimate)') and
    projects it to the 4-tuple. Cleans up its probe rows."""
    import bench_store
    bench_store.ensure_schema()
    name = f"_test_manifest_seam_{uuid.uuid4().hex[:12]}"
    stored = _stored_mean_estimate()
    # make manifest resolve THIS name without the discover() cache fighting us.
    M.discover(force=True)
    with bench_store.connect() as conn:
        def_id = bench_store.register_definition(
            name, quantity="test", units="us", description="phase-1 manifest seam probe",
            module_path="benchmarks/bench_does_not_exist.py", conn=conn)
        inst_id = bench_store.open_instance(def_id, config={"probe": True}, conn=conn)
        bench_store.set_estimate(inst_id, stored, conn=conn)
    try:
        M.discover(force=True)  # pick up the freshly-registered definition
        q = M.quantity(name, trust=True)
        assert q.source == "postgres(estimate)"
        assert q.trusted is True
        assert q.estimate is not None
        assert q.as_tuple() == (10.0, 2.0, 16, True)
    finally:
        with bench_store.connect() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM benchmark_sample WHERE instance_id = %s", (inst_id,))
                cur.execute("DELETE FROM benchmark_instance WHERE id = %s", (inst_id,))
                cur.execute("DELETE FROM benchmark_definition WHERE id = %s", (def_id,))
            conn.commit()
        M.discover(force=True)


@pytest.mark.skipif(not _db_available(), reason="control_research postgres not reachable")
def test_manifest_reconstructs_legacy_aggregate_through_postgres() -> None:
    """End-to-end legacy path: a probe instance with raw SAMPLES but NO stored estimate. The manifest's
    TRUST path reconstructs a Poolwise Estimate from the aggregate, and its 4-tuple equals the live
    aggregate byte-for-byte (the pool-fed / Estimate-fed fixed point, through real SQL)."""
    import bench_store
    bench_store.ensure_schema()
    name = f"_test_manifest_legacy_{uuid.uuid4().hex[:12]}"
    values = [10.0, 12.0, 14.0, 16.0, 18.0]  # mean 14.0, a real spread, n 5
    with bench_store.connect() as conn:
        def_id = bench_store.register_definition(
            name, quantity="test", units="us", description="phase-1 legacy-reconstruction probe",
            module_path="benchmarks/bench_does_not_exist.py", conn=conn)
        inst_id = bench_store.open_instance(def_id, config={"probe": True}, conn=conn)
        bench_store.log_samples(inst_id, values, sample_size=1, conn=conn)
        agg = bench_store.latest_aggregate(name, conn=conn)
    try:
        M.discover(force=True)
        q = M.quantity(name, trust=True)
        assert q.source == "postgres"
        assert q.trusted is True
        assert agg is not None
        # the 4-tuple matches the live aggregate exactly; the carried Estimate projects to it.
        assert q.as_tuple() == (agg[0], agg[1], agg[2], True)
        assert isinstance(q.estimate.shrink, E.Poolwise)
        assert M._project_estimate(q.estimate) == (agg[0], agg[1], agg[2])
    finally:
        with bench_store.connect() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM benchmark_sample WHERE instance_id = %s", (inst_id,))
                cur.execute("DELETE FROM benchmark_instance WHERE id = %s", (inst_id,))
                cur.execute("DELETE FROM benchmark_definition WHERE id = %s", (def_id,))
            conn.commit()
        M.discover(force=True)
