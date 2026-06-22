"""
tests/test_bench_median_pin_estimate_phase3.py
==============================================

§6 Phase 3 (the MEDIAN/PIN slice — the remainder after the fit slice) of the
harmonized-estimator migration (docs/design/harmonized-estimator-interface.md
§6 the Phase-3 bullet, §1, §7.A, §5.2):

  * the LATENCY/MEDIAN benches (`tau_io`, `wakeup`, `gather`, `req_drain`, the
    median `tmsg`) return a k=1 `QuantileLaw` `Estimate` (p=0.5) whose `cov` is a
    BOOTSTRAP median SE (§7.A — NOT the small-sample-fragile asymptotic
    `p(1−p)/(n·f̂²)`, NOT a fabricated `s²/n`), `family=EMPIRICAL`, `kind='median'`;
  * the PINS (`B_op`, `LPD`, `n_gen`, `R_gen`, `g_core`, the single-scalar `tmsg`
    variants) return a k=1 `Fixed` `Estimate` whose `cov=[[σ²]]` is the declared
    spread UN-DIVIDED (recovering e.g. B_op's σ=64 that the §5 `stddev_samp`-over-
    one-value bug discards), `family=DEGENERATE` for a true constant (n_gen) or
    `NORMAL` for a declared-spread prior, `kind='pin'`/'declared_spread'.

These tests cover the Phase-3 deliverables WITHOUT a live timed measurement (the
benches are timing-sensitive; the median's Estimate SHAPE + bootstrap SE are
exercised on a SYNTHESIZED pool, the pin's on the recorded seed — §8 discipline):

  * `bench_common.median_estimate` — the bootstrap median SE is a REAL bootstrap
    (it differs from `s²/n` on a skewed pool), the `QuantileLaw` is self-consistent
    (its implied `p(1−p)/(n·f̂²)` equals the bootstrap `cov`), the contract fields,
    the jsonb round-trip, and FAIL-LOUD on a pool with no defensible variance;
  * `bench_common.pin_estimate` — the declared σ is recovered un-divided, the
    DEGENERATE/NORMAL family split, the round-trip, FAIL-LOUD;
  * the §5.2 DE-DUP — a median bench's `run()` logs ONLY the raw pool as provenance
    (the headline median is NOT double-logged as a sample row);
  * a `QuantileLaw` and a `Fixed` Estimate flow through the driver's `step()`.

A DB-gated tail exercises the full `run()` -> `set_estimate` -> `manifest.estimate()`
TRUST stored-estimate path against the live control_research store (skipped when
unreachable, self-cleaning), proving `manifest.estimate('tau_io_us')` returns the
`QuantileLaw` Estimate (source 'postgres(estimate)', NOT the Phase-1 legacy Poolwise
reconstruction) and `manifest.estimate('B_op')` the `Fixed` one with σ=64 in `cov`.

The `estimate`/`bench_common`/`bench_<name>` modules live under
tools/analysis/OpenTURNS/ (no __init__.py — imported by sys.path, the way manifest.py
imports bench_store), so this test prepends those directories to sys.path.

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
    "tools", "analysis", "OpenTURNS",
)
_BENCH = os.path.join(_OT, "benchmarks")
for _p in (_OT, _BENCH):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import bench_common as BC  # noqa: E402  — the median_estimate/pin_estimate helpers under test
import estimate as E  # noqa: E402  — the contract


def _skewed_pool(n: int = 1500, base: float = 20.0, sigma: float = 0.3, seed: int = 7) -> list[float]:
    """A plausible right-skewed latency pool (the shape the benches median precisely because the mean is
    tail-poisoned): base + lognormal tail. A real spread, a well-defined median."""
    rng = np.random.default_rng(seed)
    return [float(v) for v in (base + rng.lognormal(0.0, sigma, n) - 1.0)]


# --------------------------------------------------------------------------- #
# 1. median_estimate: the bootstrap SE, the QuantileLaw self-consistency, the fields.
# --------------------------------------------------------------------------- #
def test_median_estimate_se_is_a_real_bootstrap_not_s2_over_n() -> None:
    """§7.A: the median SE in `cov` is a BOOTSTRAP, not the fabricated asymptotic `s²/n`. On a skewed
    pool the median is far more robust than the mean, so the bootstrap median SE is MATERIALLY smaller
    than `sqrt(s²/n)` — proving the bench did not just re-label a mean's SE."""
    pool = _skewed_pool()
    est = BC.median_estimate(pool, name="tau_io_us")
    assert est.theta_hat[0] == pytest.approx(float(np.median(pool)))
    boot_se = math.sqrt(float(est.cov[0, 0]))
    s2_over_n_se = math.sqrt(float(np.var(pool, ddof=1)) / len(pool))
    assert boot_se > 0.0
    # the two SEs are NOT the same number (the bootstrap median SE is the order-statistic spread, not s²/n).
    assert not math.isclose(boot_se, s2_over_n_se, rel_tol=0.05)
    assert boot_se < s2_over_n_se   # for a right-skewed pool the median SE is the smaller one


def test_median_estimate_quantile_law_is_self_consistent_with_cov() -> None:
    """The shipped `QuantileLaw(p=0.5, f_at_q, n)` is SELF-CONSISTENT with the `cov` it ships beside
    (P1 single-home — the variance has one home, the bootstrap, and the law agrees rather than asserting
    a second number): the law's implied `p(1−p)/(n·f̂²)` equals the bootstrap `cov[0,0]` exactly."""
    pool = _skewed_pool()
    est = BC.median_estimate(pool, name="tau_io_us")
    ql = est.shrink
    assert isinstance(ql, E.QuantileLaw)
    assert ql.p == 0.5
    assert ql.n == len(pool)
    implied_var = 0.25 / (ql.n * float(ql.f_at_q[0]) ** 2)  # p(1−p)=0.25 at p=0.5
    assert math.isclose(implied_var, float(est.cov[0, 0]), rel_tol=1e-12)


def test_median_estimate_is_deterministic_under_fixed_boot_seed() -> None:
    """The bootstrap SE is reproducible (a fixed `boot_seed`) — so a re-logged estimate is stable, not a
    different number every run (ADR-0009 reproducibility)."""
    pool = _skewed_pool()
    a = BC.median_estimate(pool, name="q", boot_seed=0)
    b = BC.median_estimate(pool, name="q", boot_seed=0)
    assert float(a.cov[0, 0]) == float(b.cov[0, 0])


def test_median_estimate_contract_fields() -> None:
    """The produced Estimate honors the §3 MEDIAN/QUANTILE contract: k=1, QuantileLaw(p=0.5),
    family=EMPIRICAL, POSITIVE support, kind='median', a PSD cov, a valid is_valid() gate."""
    est = BC.median_estimate(_skewed_pool(), name="tau_io_us")
    assert est.k == 1
    assert est.kind == "median"
    assert est.family == (E.CIFamily.EMPIRICAL,)
    assert est.support == (E.Support.POSITIVE,)
    assert isinstance(est.shrink, E.QuantileLaw)
    assert est.is_valid()


def test_median_estimate_jsonb_round_trips() -> None:
    """The median Estimate is an exact jsonb round-trip (the §5 SSOT serialization): theta_hat, cov, the
    QuantileLaw (p/f_at_q/n), family, kind come back identical, so the store persists it losslessly."""
    est = BC.median_estimate(_skewed_pool(), name="tau_io_us")
    rt = E.from_jsonb(E.to_jsonb(est))
    assert np.allclose(rt.theta_hat, est.theta_hat)
    assert np.allclose(rt.cov, est.cov)
    assert isinstance(rt.shrink, E.QuantileLaw)
    assert rt.shrink.p == est.shrink.p
    assert rt.shrink.n == est.shrink.n
    assert np.allclose(rt.shrink.f_at_q, est.shrink.f_at_q)
    assert rt.family == est.family
    assert rt.kind == est.kind


def test_median_estimate_fails_loud_on_no_defensible_variance() -> None:
    """ADR-0002: a pool with no defensible variance RAISES, never pads — an empty pool, a 1-sample
    'median' (no bootstrap spread), a non-finite reading, and a ZERO-SPREAD pool (a constant
    masquerading as a measured median — the bootstrap SE is 0, so the density f̂ would be infinite)."""
    with pytest.raises(ValueError):  # empty
        BC.median_estimate([], name="q")
    with pytest.raises(ValueError):  # 1 sample
        BC.median_estimate([5.0], name="q")
    with pytest.raises(ValueError):  # non-finite
        BC.median_estimate([1.0, float("nan"), 3.0], name="q")
    with pytest.raises(ValueError):  # zero-spread pool -> bootstrap SE == 0
        BC.median_estimate([7.0] * 64, name="q")


# --------------------------------------------------------------------------- #
# 2. pin_estimate: the declared σ recovered un-divided, the family split.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("value,sigma,name", [
    (256.0, 64.0, "B_op"),       # the worked pin instance — σ=64 recovered un-divided
    (500.0, 25.0, "LPD"),
    (152.0, 8.0, "R_gen"),
    (76000.0, 9000.0, "g_core"),
    (1.0, 0.5, "tmsg_us_leaf"),
])
def test_pin_estimate_declared_spread_recovers_sigma_un_divided(value, sigma, name) -> None:
    """§5/§3 PIN-declared-spread: `cov=[[σ²]]` is the declared spread UN-DIVIDED (a prior has no n), so
    `sqrt(cov[0,0])` recovers the declared σ exactly — the latent store bug fix (B_op's 64 reaches the
    instance row's variance, where `stddev_samp` over one logged value returns NULL→0 and discards it).
    family=NORMAL (a prior the models consume as Normal(mean, σ)), shrink=Fixed (un-shrinkable),
    kind='declared_spread'."""
    est = BC.pin_estimate(value, sigma, name=name)
    assert est.k == 1
    assert est.theta_hat[0] == value
    assert est.cov[0, 0] == sigma ** 2                 # un-divided: σ², NOT σ²/n
    assert math.sqrt(float(est.cov[0, 0])) == pytest.approx(sigma)
    assert isinstance(est.shrink, E.Fixed)
    assert est.family == (E.CIFamily.NORMAL,)
    assert est.kind == "declared_spread"
    assert est.is_valid()


def test_pin_estimate_true_constant_is_degenerate() -> None:
    """§3 PIN-true-constant (n_gen — a deployment/layout fact): family=DEGENERATE (no sampling interval),
    kind='pin'. The declared σ=0.05 is still carried in cov (so the projection recovers the 4-tuple
    σ=0.05), but the family says 'a constant, do not draw a sampling CI'."""
    est = BC.pin_estimate(3.0, 0.05, name="n_gen", constant=True)
    assert est.theta_hat[0] == 3.0
    assert est.cov[0, 0] == 0.05 ** 2
    assert est.family == (E.CIFamily.DEGENERATE,)
    assert est.kind == "pin"
    assert isinstance(est.shrink, E.Fixed)
    assert est.is_valid()
    # a true constant MAY carry σ==0 (a valid DEGENERATE zero-variance point).
    z = BC.pin_estimate(3.0, 0.0, name="n_gen", constant=True)
    assert z.cov[0, 0] == 0.0
    assert z.is_valid()


def test_n_gen_constant_flag_is_the_single_home_of_the_degenerate_classification() -> None:
    """P1 single-home (ADR-0012): the DEGENERATE-vs-declared-spread call has ONE source — the
    `Grounded.constant` flag — and `bench_n_gen` DERIVES its `pin_estimate(constant=…)` from it (it does
    NOT hardcode True). So n_gen is DEGENERATE (~0 bound contribution, §3) and R_gen's SEED is NORMAL,
    both flowing from the grounding, so the bench's measure() path and the manifest's seed path cannot
    disagree (the root of the ~/run_output stall: a true constant leaking its frozen σ into the bound).

    NOTE (ADR-0008 reclassification, 2026-06-21): `bench_r_gen.measure()` no longer returns a `Fixed`
    NORMAL pin — R_gen is a MEASURED quantity (the C++ gen-ceiling bench), so its MEASURED Estimate is a
    SHRINKABLE `QuantileLaw` (EMPIRICAL). The declared-spread NORMAL `Fixed` classification R_gen's
    `constant=False` still drives is the SEED/DISTRUST path (`manifest._estimate_from_seed`), checked
    here DB-free — the actual single-home contrast (and it does not invoke the live, timing-sensitive
    binary). See tests/test_bench_r_gen_cpp_gen_ceiling.py for the measured-path shrinkability."""
    import leaf_eval_grounding as G
    import bench_n_gen
    import manifest as M
    assert G.N_GEN_CORES.constant is True                  # the SSOT marks n_gen a true constant
    assert G.GEN_PER_CORE_DPS.constant is False            # R_gen is a measured declared-spread prior
    # the bench derives the family from the SSOT (not a literal) — change the flag, change the family.
    assert bench_n_gen.measure().family == (E.CIFamily.DEGENERATE,)
    assert bench_n_gen.measure().kind == "pin"
    # R_gen's SEED path: constant=False -> a NORMAL declared-spread `Fixed` prior (the un-shrinkable
    # DISTRUST fallback), single-homed on Grounded.constant exactly as n_gen's DEGENERATE is.
    seed_rgen = M._estimate_from_seed(
        "R_gen", G.GEN_PER_CORE_DPS.mean, G.GEN_PER_CORE_DPS.sigma, G.GEN_PER_CORE_DPS.unit,
        constant=G.GEN_PER_CORE_DPS.constant)
    assert seed_rgen.family == (E.CIFamily.NORMAL,)
    assert isinstance(seed_rgen.shrink, E.Fixed)


def test_pin_estimate_fails_loud() -> None:
    """ADR-0002: a non-finite value/sigma, a negative sigma, and a declared-spread prior with σ==0 (a
    NORMAL family with no spread is a contradiction — that is a true constant) all RAISE, never coerce."""
    with pytest.raises(ValueError):
        BC.pin_estimate(1.0, -1.0, name="q")              # negative spread
    with pytest.raises(ValueError):
        BC.pin_estimate(float("inf"), 1.0, name="q")      # non-finite value
    with pytest.raises(ValueError):
        BC.pin_estimate(1.0, float("nan"), name="q")      # non-finite sigma
    with pytest.raises(ValueError):
        BC.pin_estimate(1.0, 0.0, name="q")               # declared-spread prior with σ==0


def test_pin_estimate_jsonb_round_trips() -> None:
    """Both pin flavors round-trip through jsonb exactly (the §5 SSOT serialization)."""
    for est in (BC.pin_estimate(256.0, 64.0, name="B_op"),
                BC.pin_estimate(3.0, 0.05, name="n_gen", constant=True)):
        rt = E.from_jsonb(E.to_jsonb(est))
        assert np.allclose(rt.theta_hat, est.theta_hat)
        assert np.allclose(rt.cov, est.cov)
        assert isinstance(rt.shrink, E.Fixed)
        assert rt.family == est.family
        assert rt.kind == est.kind


# --------------------------------------------------------------------------- #
# 3. The §5.2 DE-DUP: a median bench's run() logs ONLY the raw pool (no headline scalar).
# --------------------------------------------------------------------------- #
def test_median_bench_run_logs_only_the_pool_not_the_headline(monkeypatch) -> None:
    """§5.2 de-dup: a migrated median bench (`bench_tau_io`) logs the harmonized `QuantileLaw` Estimate
    via `logged_run(estimate=…)` and the raw per-cycle pool as the SOLE provenance — the headline median
    scalar is NOT re-logged as a sample row (which would corrupt `latest_aggregate`'s count). Verified
    DB-free by capturing the logged_run calls; the bench's `_measure_raw()` (the §6 Phase-4 dict provenance
    producer `run()` consumes for BOTH the Estimate and the raw rows) is monkeypatched to a synthesized pool
    (no live timing)."""
    import contextlib
    import bench_tau_io as B

    pool = _skewed_pool()
    med = float(np.median(pool))
    monkeypatch.setattr(B, "_measure_raw", lambda *a, **k: {
        "tau_io_us_median": med, "per_cycle_us": pool,
        "n_msgs": 8, "rows_per_msg": 32, "rows_per_forward": 256})

    captured: dict = {"estimate": None, "logs": []}

    @contextlib.contextmanager
    def fake_logged_run(name, *, quantity, units, description, module_path, config=None, estimate=None):
        captured["estimate"] = estimate
        captured["config"] = config

        def log(values, sample_size=None, seq=None):
            if isinstance(values, (list, tuple)):
                captured["logs"].append(("pool", len(values)))
            else:
                captured["logs"].append(("scalar", float(values)))
        yield log

    monkeypatch.setattr(B, "logged_run", fake_logged_run)
    B.run()

    est = captured["estimate"]
    assert isinstance(est, E.Estimate)
    assert isinstance(est.shrink, E.QuantileLaw)
    assert est.kind == "median"
    assert est.theta_hat[0] == pytest.approx(med)
    log_kinds = [t for (t, _) in captured["logs"]]
    assert "pool" in log_kinds            # the raw provenance pool IS logged
    assert "scalar" not in log_kinds      # the headline median is NOT double-logged as a sample row
    # the headline median is preserved as config provenance (not lost — it just is not a sample row).
    assert captured["config"]["tau_io_us_median"] == pytest.approx(med)


# --------------------------------------------------------------------------- #
# 4. The Estimates flow through the driver's step().
# --------------------------------------------------------------------------- #
def test_quantile_and_fixed_estimates_flow_through_the_driver_step() -> None:
    """A `QuantileLaw` (median) Estimate and a `Fixed` (pin) Estimate both feed the Phase-2 driver's
    `step()`: the step completes, the variance is finite/positive, and BOTH inputs contribute their a_i
    to the bound. This is the §6 Phase-3 integration — the median/pin benches' outputs are consumed by
    the allocator unchanged from the fit/mean cases (a QuantileLaw projects its median + bootstrap SE; a
    Fixed contributes its declared-spread a_i). The pin's NON-funding by the ALLOCATOR (the §2.3 'a pin
    drops out, its variance is irreducible' branch) is asserted directly on `_socp_allocation` below,
    separately from `step()`'s forward-progress nudge (which can fund any input when the reducible ones
    round to zero — a distinct mechanism, not the allocator)."""
    pytest.importorskip("scipy")

    from neyman_driver import NeymanDriver
    f = lambda x: 2.0 * x[0] + 3.0 * x[1]
    d = NeymanDriver(f, costs=[1.0, 1.0], tolerance=0.5, names=["x0", "x1"])
    med_est = BC.median_estimate(_skewed_pool(base=10.0), name="x0")   # a QuantileLaw input
    pin_est = BC.pin_estimate(5.0, 2.0, name="x1")                     # a Fixed declared-spread input
    d.set_estimate(0, med_est)
    d.set_estimate(1, pin_est)
    rec = d.step(second_order_check=False)
    prims = {p.name: p for p in rec.primitives}
    assert math.isfinite(rec.var_estimate) and rec.var_estimate > 0.0
    assert math.isfinite(rec.ci_halfwidth)
    assert prims["x0"].a > 0 and prims["x1"].a > 0   # both contribute their a_i to the bound

    # The §2.3 allocator drops the Fixed pin (irreducible variance) — its n_star equals its current n,
    # regardless of the forward-progress nudge that step() may apply on top. Driven directly on the
    # allocation so the assertion is about the ALLOCATOR, not the nudge.
    ests = [med_est, pin_est]
    Sigma = d._assemble_sigma(ests)
    mu = np.array([float(e.theta_hat[0]) for e in ests])
    grad = d._gradient(mu)
    n_cur = np.array([d._effective_n(i, ests[i]) for i in range(2)], dtype=float)
    V_target = (0.5 / d.z) ** 2
    n_star = d._socp_allocation(grad, Sigma, d.costs, V_target, n_cur, ests=ests)
    assert n_star[1] == n_cur[1]   # the pin keeps its current n — the allocator never funds it


# --------------------------------------------------------------------------- #
# 5. End-to-end through the live store: run() -> set_estimate -> manifest.estimate() (DB-gated).
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
def test_run_logs_median_and_pin_estimates_and_manifest_reads_them_back(monkeypatch) -> None:
    """End-to-end through the real store, WITHOUT a live timed measurement (`_measure_raw()` — the §6
    Phase-4 dict provenance producer `run()` consumes — monkeypatched to a synthesized pool / the recorded
    seed): a MEDIAN bench's run() logs a `QuantileLaw` Estimate and a PIN bench's run() a `Fixed` Estimate
    via set_estimate, and `manifest.estimate(name)` reads each back through the TRUST stored-estimate path
    (source 'postgres(estimate)', NOT the Phase-1 legacy Poolwise reconstruction). The median's de-dup is
    asserted (only the pool as provenance rows). The pin recovers σ=64 in the stored cov. Self-cleaning of
    its synthetic instances."""
    import bench_store
    import manifest as M
    import bench_tau_io
    import bench_b_op

    bench_store.ensure_schema()
    pool = _skewed_pool()
    med = float(np.median(pool))
    # MEDIAN bench: a synthesized per-cycle pool (no live timing). Patch `_measure_raw` (the dict producer
    # run() consumes); run() then builds the real QuantileLaw Estimate from it via the un-patched helper.
    monkeypatch.setattr(bench_tau_io, "_measure_raw", lambda *a, **k: {
        "tau_io_us_median": med, "per_cycle_us": pool,
        "n_msgs": 8, "rows_per_msg": 32, "rows_per_forward": 256})
    # PIN bench: its `_measure_raw()` already returns the seed (256, σ=64 via get_seed()); no patch needed.

    try:
        bench_tau_io.run()
        bench_b_op.run()
        M.discover(force=True)

        # --- the MEDIAN reads back as a QuantileLaw via the stored-estimate path ---
        q_med = M.quantity("tau_io_us", trust=True)
        est_med = M.estimate("tau_io_us", trust=True)
        assert q_med.source == "postgres(estimate)"          # the stored-estimate path, not legacy
        assert isinstance(est_med.shrink, E.QuantileLaw)
        assert est_med.kind == "median"
        assert est_med.theta_hat[0] == pytest.approx(med, abs=1e-6)
        assert math.sqrt(float(est_med.cov[0, 0])) > 0.0     # the bootstrap median SE survives the round-trip
        # the 4-tuple projects the median + the QuantileLaw n (the §5/Phase-1 projection rule).
        assert M.value("tau_io_us", trust=True)[0] == pytest.approx(med, abs=1e-6)
        assert M.value("tau_io_us", trust=True)[2] == len(pool)   # n = QuantileLaw.n

        # §5.2 DE-DUP: the latest instance carries the raw pool as provenance, NOT pool + the headline.
        with bench_store.connect() as c:
            with c.cursor() as cur:
                cur.execute(
                    """SELECT (SELECT count(*) FROM benchmark_sample s WHERE s.instance_id = i.id)
                       FROM benchmark_instance i JOIN benchmark_definition d ON d.id = i.definition_id
                       WHERE d.name = %s AND i.estimate IS NOT NULL
                       ORDER BY i.started_at DESC LIMIT 1""", ("tau_io_us",))
                (nsamp,) = cur.fetchone()
        assert nsamp == len(pool)   # exactly the pool, not the pool + 1 headline scalar

        # --- the PIN reads back as a Fixed with σ=64 recovered in cov ---
        q_pin = M.quantity("B_op", trust=True)
        est_pin = M.estimate("B_op", trust=True)
        assert q_pin.source == "postgres(estimate)"
        assert isinstance(est_pin.shrink, E.Fixed)
        assert est_pin.kind == "declared_spread"
        assert est_pin.theta_hat[0] == pytest.approx(256.0)
        assert math.sqrt(float(est_pin.cov[0, 0])) == pytest.approx(64.0)   # σ=64 reaches the DB
        assert est_pin.family == (E.CIFamily.NORMAL,)
    finally:
        with bench_store.connect() as c:
            with c.cursor() as cur:
                for name in ("tau_io_us", "B_op"):
                    cur.execute(
                        """SELECT i.id FROM benchmark_instance i
                           JOIN benchmark_definition d ON d.id = i.definition_id
                           WHERE d.name = %s""", (name,))
                    for (iid,) in cur.fetchall():
                        cur.execute("DELETE FROM benchmark_sample WHERE instance_id = %s", (iid,))
                        cur.execute("DELETE FROM benchmark_instance WHERE id = %s", (iid,))
            c.commit()
        M.discover(force=True)
