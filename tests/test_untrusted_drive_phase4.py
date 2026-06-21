"""
tests/test_untrusted_drive_phase4.py
====================================

§6 Phase 4 (the FINAL phase) of the harmonized-estimator migration
(docs/design/harmonized-estimator-interface.md §6, the Phase-4 bullet): DELETE the
coercion + unify on `measure() -> Estimate`. The three coupled deliverables:

  1. THE `measure() -> Estimate` LIFT. Each bench's `measure()` returns the harmonized
     `Estimate` it DECLARES (built by an internal `_estimate_from_raw`, single-homed with
     `run()` — P1), so the driver consumes it directly with NO guessing which list is the
     estimate. The raw-pool provenance is preserved (the bench's `_measure_raw()` produces
     the dict `run()` logs); the §5.2 de-dup is unchanged.
  2. THE COERCION DELETION. `untrusted_drive._per_sample`'s longest-numeric-list heuristic
     AND the 2-sample zero-spread pad are GONE; `_make_measurer(budget)` returns the bench's
     `Estimate` directly (P2 reject-don't-guess). A bench returning a non-Estimate, or an
     invalid one, is a loud failure (ADR-0002) — never a coerced pool. The EXECUTED proof:
     the un-trusted drive now produces a SANE `E[f]` (the original cratered `E[f]≈11.9`
     symptom — the `t_row` fit mis-read as ~224 by the longest-list heuristic — is GONE).
  3. THE FABRICATED 2-POINT PILOT MIGRATION. `throughput_bound._ot_bound` and
     `transport_sweep`'s CI / variance-ranking now feed each input as its manifest `Estimate`
     via `set_estimate(s_by_name)`, REPLACING the `{mean−sigma, mean+sigma}` 2-point pool fed
     to `add_samples`. The bound is byte-for-byte the old pilot's (the spec EXECUTED and
     REFUTED a claimed `/2` bug: the 2-point set's sample-std is √2·σ, so `a_i/n_i =
     grad²·σ²` exactly — the `/2` is cancelled by the √2). A declared-spread prior is a
     `Fixed` Estimate that drops out of allocation (the §2.3 branch), un-shrinkable.

The `estimate`/`neyman_driver`/`manifest`/`untrusted_drive`/`throughput_bound`/
`transport_sweep`/`bench_*` modules live under tools/analysis/OpenTURNS/ (no __init__.py —
imported by sys.path the way manifest.py imports bench_store), so this test prepends those
directories. openturns + scipy + cvxpy are the driver's deps; the tests skip loudly if one
is genuinely absent. The bench-import tests do NOT run a live timed measurement (the benches
are timing-sensitive — Estimate SHAPE is exercised via the bench's `_estimate_from_raw` on a
recorded/synthesized `_measure_raw` dict, per the §8 / Phase-3 discipline).

Public Domain (The Unlicense).
"""
from __future__ import annotations

import glob
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

import estimate as E  # noqa: E402  — the contract

# The 7-point design the fit benches use; the recorded staged / fully_device fit shapes.
DESIGN = [32, 64, 128, 192, 256, 384, 512]


def _staged_dict() -> dict:
    """A recorded staged-fit `_measure_raw()` dict (slope ~4.317, intercept ~94.58) with a tiny residual
    so resid_var > 0 — the shape `bench_t_row._measure_raw` / `bench_iota._measure_raw` return."""
    medians = {B: float(94.58 + 4.317 * B + (0.4 if B % 2 else -0.4)) for B in DESIGN}
    return {"slope_us_per_row": 4.317, "intercept_us": 94.58, "r2": 0.998,
            "per_width_median_us": medians, "batches": list(DESIGN)}


def _fulldev_dict() -> dict:
    """A recorded fully_device-fit `_measure_raw()` dict (slope ~3.092, intercept ~68.84) — the shape
    `bench_t_disp._measure_raw` / `bench_cpp_inproc_port_t_row_bare_us._measure_raw` return."""
    medians = {B: float(68.84 + 3.092 * B + (0.3 if B % 2 else -0.3)) for B in DESIGN}
    return {"slope_us_per_row": 3.092, "intercept_us": 68.84, "r2": 0.997,
            "per_width_median_us": medians, "batches": list(DESIGN),
            "decomposition": {"dispatch_floor_us": 68.84}}


def _pool_dict(med: float, key: str = "per_cycle_us") -> dict:
    """A recorded median-bench `_measure_raw()` dict: a right-skewed latency pool under `key`, plus the
    cfg fields `bench_tau_io.run()` reads (n_msgs/rows_per_msg/rows_per_forward/tau_io_us_median) so the
    run()-path is exercised faithfully (the dict `_measure_raw` returns is what run() consumes)."""
    rng = np.random.default_rng(7)
    pool = [float(v) for v in (med + rng.lognormal(0.0, 0.25, 1500) - 1.0)]
    return {key: pool, "tau_io_us_median": float(np.median(pool)),
            "n_msgs": 8, "rows_per_msg": 32, "rows_per_forward": 256}


# --------------------------------------------------------------------------- #
# 1. The measure() -> Estimate lift: every bench's measure() declares an Estimate.
# --------------------------------------------------------------------------- #
def _all_bench_modules() -> list:
    """Every bench module (importable jax-free — the jax import is lazy inside `_measure_raw`)."""
    mods = []
    for path in sorted(glob.glob(os.path.join(_BENCH, "bench_*.py"))):
        name = os.path.splitext(os.path.basename(path))[0]
        if name == "bench_common":
            continue
        mods.append(__import__(name))
    return mods


def test_every_bench_measure_returns_an_estimate_annotation() -> None:
    """§6 Phase-4 deliverable 1: EVERY bench exposes `measure`, `_measure_raw`, `_estimate_from_raw`, and
    `measure()` is annotated to return an `Estimate` (the typed contract the driver consumes). The
    annotation is the contract's SSOT (ADR-0012 P8); the lift is uniform across all ~30 benches."""
    import inspect
    mods = _all_bench_modules()
    assert len(mods) >= 25, f"expected the full bench suite, found {len(mods)}"
    for mod in mods:
        assert hasattr(mod, "measure"), f"{mod.__name__} has no measure()"
        assert hasattr(mod, "_measure_raw"), f"{mod.__name__} has no _measure_raw() (the provenance helper)"
        assert hasattr(mod, "_estimate_from_raw"), f"{mod.__name__} has no _estimate_from_raw() (the P1 builder)"
        ret = str(inspect.signature(mod.measure).return_annotation)
        assert "Estimate" in ret and "dict" not in ret, f"{mod.__name__}.measure() returns {ret}, not an Estimate"


def test_pin_benches_measure_returns_valid_fixed_estimate_live() -> None:
    """§6 Phase-4 deliverable 1 (PIN): a pin bench's `measure()` is timing-FREE (it reads the seed), so it
    runs fully — and returns a valid k=1 `Fixed` Estimate (a pin is a `Fixed`/declared-spread Estimate, NOT
    a faked 2-sample pool). The declared σ is recovered un-divided in `cov`."""
    import bench_b_op
    import bench_n_gen
    est = bench_b_op.measure()
    assert isinstance(est, E.Estimate) and est.is_valid()
    assert isinstance(est.shrink, E.Fixed)
    assert est.theta_hat[0] == 256.0
    assert math.sqrt(float(est.cov[0, 0])) == 64.0      # B_op's σ=64 recovered un-divided
    assert est.kind == "declared_spread"
    # a true constant is DEGENERATE, kind='pin'
    n_est = bench_n_gen.measure()
    assert isinstance(n_est.shrink, E.Fixed) and n_est.family == (E.CIFamily.DEGENERATE,)
    assert n_est.kind == "pin"


def test_fit_bench_measure_declares_slope_first_estimate_via_measure_raw(monkeypatch) -> None:
    """§6 Phase-4 deliverable 1 (FIT): `bench_t_row.measure()` returns its k=2 fit Estimate with the SLOPE
    as component 0 (theta_hat[0] ≈ 4.317 — the marginal the driver/manifest project), built by
    `_estimate_from_raw` from `_measure_raw`'s recorded dict. THIS is what cures the longest-list mis-read:
    the bench DECLARES the slope, not the row-count x-axis."""
    import bench_t_row
    monkeypatch.setattr(bench_t_row, "_measure_raw", lambda **k: _staged_dict())
    est = bench_t_row.measure()
    assert isinstance(est, E.Estimate) and est.is_valid()
    assert est.k == 2 and est.kind == "ols_fit"
    assert isinstance(est.shrink, E.RegressionLaw)
    assert est.names[0] == "t_row_us"
    assert abs(float(est.theta_hat[0]) - 4.317) < 1.0   # the SLOPE is component 0, NOT ~224 (the row-count)


def test_iota_bench_measure_declares_intercept_first_via_delegated_measure_raw(monkeypatch) -> None:
    """§6 Phase-4 deliverable 1 (FIT, delegating): `bench_iota.measure()` returns the SAME staged fit with
    iota's INTERCEPT as component 0 (~94.58). Its `_measure_raw` delegates to `bench_t_row._measure_raw`
    (one measurement grounds both); patching the delegated source flows through."""
    import bench_iota
    import bench_t_row
    monkeypatch.setattr(bench_t_row, "_measure_raw", lambda **k: _staged_dict())
    est = bench_iota.measure()
    assert est.k == 2 and est.names[0] == "iota_us"
    assert abs(float(est.theta_hat[0]) - 94.58) < 5.0   # the INTERCEPT is component 0


def test_median_bench_measure_declares_quantile_estimate_via_measure_raw(monkeypatch) -> None:
    """§6 Phase-4 deliverable 1 (MEDIAN): `bench_tau_io.measure()` returns a k=1 `QuantileLaw` median
    Estimate over the per-cycle pool, built by `_estimate_from_raw` from `_measure_raw`'s dict."""
    import bench_tau_io
    d = _pool_dict(20.0, key="per_cycle_us")
    monkeypatch.setattr(bench_tau_io, "_measure_raw", lambda **k: d)
    est = bench_tau_io.measure()
    assert isinstance(est, E.Estimate) and est.is_valid()
    assert isinstance(est.shrink, E.QuantileLaw) and est.kind == "median"
    assert est.theta_hat[0] == pytest.approx(float(np.median(d["per_cycle_us"])))


def test_measure_and_run_share_one_estimate_builder(monkeypatch) -> None:
    """P1 single-home: `measure()` and `run()` build the Estimate via the SAME `_estimate_from_raw`, so
    they cannot disagree. We capture run()'s logged Estimate and assert it equals measure()'s — same
    theta_hat, same cov — on a shared recorded dict. (DB-free: logged_run is faked.)"""
    import contextlib
    import bench_tau_io
    d = _pool_dict(20.0, key="per_cycle_us")
    monkeypatch.setattr(bench_tau_io, "_measure_raw", lambda **k: d)

    captured: dict = {}

    @contextlib.contextmanager
    def fake_logged_run(name, *, quantity, units, description, module_path, config=None, estimate=None):
        captured["estimate"] = estimate

        def log(values, sample_size=None, seq=None):
            captured.setdefault("logs", []).append(("pool" if isinstance(values, (list, tuple)) else "scalar"))
        yield log

    monkeypatch.setattr(bench_tau_io, "logged_run", fake_logged_run)
    bench_tau_io.run()
    measure_est = bench_tau_io.measure()
    run_est = captured["estimate"]
    assert isinstance(run_est, E.Estimate)
    assert np.allclose(run_est.theta_hat, measure_est.theta_hat)
    assert np.allclose(run_est.cov, measure_est.cov)
    # §5.2 de-dup preserved: the raw pool is logged, the headline median is NOT a sample row.
    assert "pool" in captured["logs"] and "scalar" not in captured["logs"]


# --------------------------------------------------------------------------- #
# 2. The coercion deletion + untrusted_drive._make_measurer.
# --------------------------------------------------------------------------- #
def test_untrusted_drive_coercion_is_deleted() -> None:
    """§6 Phase-4 deliverable 2: the `_per_sample` longest-numeric-list heuristic and the `_make_sampler`
    2-sample pad are GONE (the silent failure that cratered the bound). `_make_measurer` replaces them —
    it returns the bench's Estimate directly, nothing to guess."""
    import untrusted_drive as U
    assert not hasattr(U, "_per_sample"), "the _per_sample coercion must be deleted"
    assert not hasattr(U, "_make_sampler"), "the _make_sampler 2-sample-pad path must be deleted"
    assert hasattr(U, "_make_measurer"), "the §6 Phase-4 _make_measurer must replace them"


def test_registry_qname_bridges_both_model_map_shapes() -> None:
    """`untrusted_drive` must resolve a model input -> its registry quantity for BOTH model map shapes:
    `model_zmq_baseline` exposes `INPUT_QUANTITIES[nm]=(qname,cost)`; the other four variants expose
    `_MANIFEST_NAME[nm]=qname`. `python untrusted_drive.py lockfree_mpsc` regressed here with an
    AttributeError (no INPUT_QUANTITIES) — the `_registry_qname` bridge closes it."""
    import importlib
    import untrusted_drive as U
    for slug in ("zmq_baseline", "lockfree_mpsc", "shm_spin_poll", "futex_wake", "cpp_inproc_port"):
        model = importlib.import_module("model_" + slug)
        qs = [U._registry_qname(model, nm) for nm in model.INPUT_NAMES]  # raises if any input is unmapped
        assert len(qs) == len(model.INPUT_NAMES)
        assert all(isinstance(q, str) and q for q in qs), f"{slug}: an input resolved to an empty qname"


def test_make_measurer_returns_estimate_and_rejects_non_estimate(monkeypatch) -> None:
    """§6 Phase-4 deliverable 2: `_make_measurer(qname)(budget)` returns the bench's `Estimate` directly
    (P2). A bench whose measure() returns a non-Estimate (a bespoke dict — exactly the old failure) is a
    loud TypeError at the seam; an invalid Estimate a loud ValueError. NEVER a coerced pool."""
    import types
    import untrusted_drive as U
    import bench_b_op

    # the happy path: a real pin bench measure() -> a valid Fixed Estimate (B_op resolves to bench_b_op).
    m = U._make_measurer("B_op", iters_cap=10)
    est = m(4)
    assert isinstance(est, E.Estimate) and isinstance(est.shrink, E.Fixed)

    # a bench whose measure() returns a DICT (the pre-Phase-4 shape the heuristic guessed a pool out of).
    # `_make_measurer` resolves the bench via `_bench_module` (a fresh import), so we patch THAT to a fake
    # module returning the dict — proving the seam rejects it loudly rather than coercing a pool out of it.
    bad_mod = types.SimpleNamespace(
        NAME="bad", measure=lambda **k: {"per_cycle_us": [1.0, 2.0], "batches": [32, 64]})
    monkeypatch.setattr(U, "_bench_module", lambda qname: bad_mod)
    m2 = U._make_measurer("bad", iters_cap=10)
    with pytest.raises(TypeError):
        m2(4)   # a dict is not an Estimate -> loud, never coerced into a pool (the deleted behavior)

    # a bench whose measure() returns an INVALID Estimate -> loud ValueError (not a padded estimate).
    class _BadEst:
        def is_valid(self):
            return False
    # subclass-free: a real Estimate that we corrupt to fail is_valid would not construct; instead feed a
    # non-Estimate-but-Estimate-typed sentinel via a fake whose measure() returns it.
    bad_est_mod = types.SimpleNamespace(NAME="bad2", measure=lambda **k: _make_invalid_estimate())
    monkeypatch.setattr(U, "_bench_module", lambda qname: bad_est_mod)
    m3 = U._make_measurer("bad2", iters_cap=10)
    with pytest.raises(ValueError):
        m3(4)   # is_valid() False -> loud, never padded


def _make_invalid_estimate() -> "E.Estimate":
    """A constructed-valid Estimate whose `is_valid()` we force False by mutating a field through the frozen
    dataclass's __dict__ (a value that reached the seam through a ctor-bypassing path — the case is_valid()
    re-checks). Used only to drive the seam's invalid-Estimate rejection branch."""
    est = E.Estimate(
        theta_hat=np.array([1.0]), cov=np.array([[1.0]]), names=("q",),
        shrink=E.Fixed(), support=(E.Support.POSITIVE,), family=(E.CIFamily.NORMAL,), kind="declared_spread")
    object.__setattr__(est, "cov", np.array([[-5.0]]))   # a non-PSD cov -> is_valid() returns False
    return est


def test_make_measurer_requires_measure_not_run(monkeypatch) -> None:
    """§6 Phase-4: the test-drive path is `measure()` (in-memory). A bench exposing no measure() is a loud
    AttributeError (run() persists to postgres and is not the un-trusted path). The pre-Phase-4 fallback to
    run() is removed (it returned a dict and would have logged)."""
    import types
    import untrusted_drive as U
    fake = types.SimpleNamespace(NAME="q", run=lambda **k: {})   # has run(), no measure()
    monkeypatch.setattr(U, "_bench_module", lambda qname: fake)
    with pytest.raises(AttributeError):
        U._make_measurer("q", iters_cap=10)


# --------------------------------------------------------------------------- #
# 3. THE EXECUTED PROOF: the un-trusted drive now produces a SANE E[f] (11.9 is GONE).
# --------------------------------------------------------------------------- #
def _ot():
    ot = pytest.importorskip("openturns", reason="the driver requires openturns")
    pytest.importorskip("scipy", reason="the Clark closed form needs scipy.stats")
    return ot


def test_old_longest_list_heuristic_craters_the_bound() -> None:
    """The DOCUMENTED symptom, reproduced: the deleted `_per_sample` grabbed the LONGEST numeric list in
    the t_row dict — the row-count x-axis `[32…512]` (mean ~224) — and fed it as the `t_row` pool, so the
    driver evaluated `f` with `t_row ≈ 224` instead of the slope `4.317`. The bound CRATERS (E[f] ≪ the
    sane ~428). This is the failure Phase 4 removes; we reproduce it to anchor the contrast."""
    ot = _ot()
    import manifest as M
    import model_zmq_baseline as model
    from neyman_driver import NeymanDriver  # noqa: F401  (import gate)

    names = model.INPUT_NAMES
    x0 = model.initial_point(trust=True)
    sig = model.sigmas(trust=True)
    # the longest numeric list the t_row dict carried = the row-count axis
    row_axis = np.array(DESIGN, dtype=float)
    driver, _ = model.build_driver(tolerance=5.0, trust=True)
    for i, nm in enumerate(names):
        if nm == "t_row":
            m, s2, n = float(row_axis.mean()), float(row_axis.var(ddof=1)), len(row_axis)
            driver.set_estimate(i, E.Estimate(
                theta_hat=np.array([m]), cov=np.array([[s2 / n]]), names=(nm,),
                shrink=E.Poolwise(per_sample_var=np.array([s2])),
                support=(E.Support.POSITIVE,), family=(E.CIFamily.NORMAL,), kind="mean"))
        else:
            driver.set_estimate(i, M._estimate_from_seed(nm, x0[nm], sig[nm], ""))
    rec = driver.step(second_order_check=False)
    assert rec.estimate < 50.0   # the cratered nonsense (the ~11.9 family) — t_row read as ~224


def test_untrusted_drive_estimate_path_is_sane(monkeypatch) -> None:
    """§6 Phase-4 EXECUTED PROOF (ADR-0009): the un-trusted drive — every input fed LIVE from its bench as
    an `Estimate`, through `untrusted_drive._make_measurer` + `driver.run(measurers=…)` — now produces a
    SANE `E[f]` (the binding serve capacity ~428, NOT the cratered ~11.9). The benches' measure() are
    mocked to their DECLARED Estimates (timing-free), so this exercises the real Phase-4 loop without a
    live timed run. The contrast vs the old heuristic (the test above) is the deliverable."""
    ot = _ot()
    import manifest as M
    import untrusted_drive as U
    import bench_common as BC
    import model_zmq_baseline as model

    qof = {nm: model.INPUT_QUANTITIES[nm][0] for nm in model.INPUT_NAMES}

    def fit_slope_est():
        medians = [94.58 + 4.317 * B + (0.4 if B % 2 else -0.4) for B in DESIGN]
        return BC.fit_estimate(DESIGN, medians, own_name="t_row_us", own_role="slope", partner_name="iota_us")

    def median_est(name, base):
        rng = np.random.default_rng(7)
        return BC.median_estimate(list(base + rng.lognormal(0, 0.25, 1500) - 1.0), name=name)

    factories = {
        "t_row_us": lambda **k: fit_slope_est(),
        "zmq_baseline_tau_io_us": lambda **k: median_est("zmq_baseline_tau_io_us", 20.0),
        "zmq_baseline_wakeup_us": lambda **k: median_est("zmq_baseline_wakeup_us", 1.5),
        "zmq_baseline_tmsg_us_leaf": lambda **k: BC.pin_estimate(1.0, 0.5, name="zmq_baseline_tmsg_us_leaf"),
        "n_gen": lambda **k: BC.pin_estimate(3.0, 0.05, name="n_gen", constant=True),
        "R_gen": lambda **k: BC.pin_estimate(152.0, 8.0, name="R_gen"),
        "B_op": lambda **k: BC.pin_estimate(256.0, 64.0, name="B_op"),
        "T_disp_us": lambda **k: BC.pin_estimate(68.84, 2.0, name="T_disp_us"),
        "LPD": lambda **k: BC.pin_estimate(500.0, 25.0, name="LPD"),
    }
    # patch the actual bench modules untrusted_drive resolves, and disable the timing-sensitive warmup.
    for qname, fac in factories.items():
        mod = M._import_bench_module(M.discover()[qname]["module_path"])
        monkeypatch.setattr(mod, "measure", fac, raising=False)
        if hasattr(mod, "WARMUP"):
            monkeypatch.setattr(mod, "WARMUP", 0, raising=False)
        if hasattr(mod, "warmup"):
            monkeypatch.delattr(mod, "warmup", raising=False)

    driver, _ = model.build_driver(tolerance=5.0, trust=True)
    names = list(model.INPUT_NAMES)
    measurers = {i: U._make_measurer(qof[nm], iters_cap=50) for i, nm in enumerate(names)}
    final = driver.run(measurers=measurers, pilot=8, max_rounds=2, verbose=False)
    assert 100.0 < final.estimate < 700.0, f"E[f]={final.estimate} is NOT sane (the 11.9 crater is back!)"
    assert final.estimate == pytest.approx(428.28, abs=2.0)   # the binding serve capacity at the seed


# --------------------------------------------------------------------------- #
# 4. The fabricated 2-point pilot migration (throughput_bound + transport_sweep).
# --------------------------------------------------------------------------- #
def _build_driver(model, tolerance):
    """`model.build_driver` — the variant models take a `trust` kwarg, the v1 models (model_capacity /
    model_cycletime) do not. Accept either (uniform across the family)."""
    try:
        return model.build_driver(tolerance=tolerance, trust=True)
    except TypeError:
        return model.build_driver(tolerance=tolerance)


def _two_point_pilot_bound(model, names, x0, sig):
    """The PRE-Phase-4 path, recomputed standalone: feed each input a `{mean−σ, mean+σ}` 2-sample pool via
    add_samples, step once. Returns (var_estimate, ci_halfwidth, {name: p.a}). This is what the migration
    must reproduce (the bound) — the spec's no-`/2`-bug fixed point."""
    driver, _ = _build_driver(model, 0.1)
    pilot = {i: np.array([x0[nm] - max(sig[nm], 1e-9), x0[nm] + max(sig[nm], 1e-9)])
             for i, nm in enumerate(names)}
    driver.add_samples(pilot)
    rec = driver.step(second_order_check=False)
    return rec.var_estimate, rec.ci_halfwidth, {p.name: p.a for p in rec.primitives}


def test_transport_sweep_estimate_feed_reproduces_the_2point_pilot_bound() -> None:
    """§6 Phase-4 deliverable 3: `transport_sweep._model_estimates` feeds each input its manifest `Estimate`
    via `set_estimates_by_name`, REPLACING the 2-point pilot — and the bound (var, ci) is byte-for-byte the
    old pilot's (the spec's no-`/2`-bug fixed point), and the variance ranking (by a_i) is identical."""
    _ot()
    import transport_sweep as TS
    import model_zmq_baseline as model

    names = model.INPUT_NAMES
    x0 = model.initial_point(trust=True)
    sig = model.sigmas(trust=True)
    var_old, ci_old, a_old = _two_point_pilot_bound(model, names, x0, sig)

    driver, _ = model.build_driver(tolerance=0.1, trust=True)
    driver.set_estimates_by_name(TS._model_estimates(model))   # the Phase-4 feed (manifest Estimates)
    rec = driver.step(second_order_check=False)

    assert math.isclose(rec.var_estimate, var_old, rel_tol=1e-9)   # byte-for-byte the old bound (1-ULP reorder)
    assert math.isclose(rec.ci_halfwidth, ci_old, rel_tol=1e-9)
    rank_old = sorted(a_old, key=lambda nm: a_old[nm], reverse=True)
    rank_new = [p.name for p in sorted(rec.primitives, key=lambda p: p.a, reverse=True)]
    assert rank_new == rank_old                                    # the next-benchmark ranking is preserved


def test_throughput_bound_ot_path_feeds_fixed_estimates_no_pilot() -> None:
    """§6 Phase-4 deliverable 3: `throughput_bound._ot_bound` drives via `set_estimate` (Fixed/declared-
    spread Estimates), NOT the 2-point `add_samples` pilot, and reproduces the model's f(μ̂) and the
    grounded-uncertainty CI. The grounded inputs are declared-spread priors, so the allocator funds none
    (un-shrinkable — the §2.3 branch)."""
    _ot()
    import throughput_bound as TB
    import model_capacity

    driver, rec, f_mu, x0 = TB._ot_bound(model_capacity)
    # the bound is the model's f at the grounded mean; the CI is the grounded-uncertainty spread.
    assert f_mu == pytest.approx(model_capacity.throughput_numpy(x0), rel=1e-9)
    assert rec.var_estimate > 0.0 and math.isfinite(rec.ci_halfwidth)
    # every input is a Fixed declared-spread prior -> the allocator funds NONE (recommend collapses);
    # the forward-progress nudge may touch ONE contending input, so total recommend is small, not "samples".
    assert sum(p.recommend for p in rec.primitives) <= 1
    # the bound matches the standalone 2-point pilot's (the no-`/2`-bug fixed point).
    names = model_capacity.INPUT_NAMES
    var_old, ci_old, _ = _two_point_pilot_bound(model_capacity, names, x0, model_capacity.SIGMAS)
    assert math.isclose(rec.var_estimate, var_old, rel_tol=1e-9)
    assert math.isclose(rec.ci_halfwidth, ci_old, rel_tol=1e-9)


def test_two_point_pilot_has_no_over_2_bug_sample_std_is_root2_sigma() -> None:
    """§6/§7/§8 the REFUTED `/2` bug, re-executed (ADR-0009): the 2-point set `{μ−σ, μ+σ}` has sample
    `std(ddof=1) = √2·σ` exactly, so `a_i/n_i = grad²·(√2σ)²/2 = grad²·σ²` — the `/2` is cancelled by the
    √2 inflation. So the migration to a `Fixed` Estimate (`cov=[[σ²]]`, the direct `g²σ²` contribution)
    reproduces the SAME bound. This test pins the invariant so the refuted `/2` claim is never reintroduced."""
    sigma = 3.7
    mu = 42.0
    pool = np.array([mu - sigma, mu + sigma])
    assert float(pool.std(ddof=1)) == pytest.approx(math.sqrt(2.0) * sigma, rel=1e-12)
    grad = 5.0
    a_over_n_2point = (grad * float(pool.std(ddof=1))) ** 2 / len(pool)   # (g·√2σ)²/2
    a_over_n_fixed = grad ** 2 * sigma ** 2                               # g²·σ² (the Fixed cov contribution)
    assert math.isclose(a_over_n_2point, a_over_n_fixed, rel_tol=1e-12)   # identical -> no /2 bug
