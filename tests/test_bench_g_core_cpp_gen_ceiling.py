"""
tests/test_bench_g_core_cpp_gen_ceiling.py
==========================================

The ADR-0008 RECLASSIFICATION of `g_core` (the per-core generation LEAF rate, leaves/s/core):
a MEASURED quantity that was MIS-WIRED as an un-shrinkable `Fixed` pin, now wired to the live C++
gen-ceiling bench (`cpp/build/chocofarm-search-runtime-bench`, source cpp/src/search_runtime_bench.cpp)
so it returns a SHRINKABLE Estimate the Neyman loop can sample (docs/design/harmonized-estimator-
interface.md §3 the PIN-now/measurable-later row + §7.A; §0 status names `g_core` as the SAME
measured-but-punted class as R_gen, the leaf-unit twin). The defect: the bench PUNTED — `_measure_raw()`
returned the seed 76000 and `_estimate_from_raw()` wrapped it in `pin_estimate` -> a
`Fixed`/`marginal=0`/un-fundable Estimate -> `untrusted_drive`'s generation arm STALLED (and the dict
labelled itself `is_cpp_bench=True` while running NO bench — the lying signature, ADR-0012 P8/P1).

This mirrors `tests/test_bench_r_gen_cpp_gen_ceiling.py` (g_core IS R_gen's leaf-unit twin, from the
IDENTICAL binary/config/run): g_core = R_gen * LPD = leaf_requests_total/(n_tasks*serial) per rep.

THE FIX, AS TESTS:
  * `_estimate_from_raw` over a per-rep leaves/s/core pool builds a SHRINKABLE `QuantileLaw` (median)
    Estimate — a REAL bootstrap median SE, `family=EMPIRICAL`, `kind='median'` — NOT a `Fixed` pin; and
    its typed `ShrinkLaw.marginal_dvar_deffort` is < 0 (so the driver's `A_i = −marginal·n²` is > 0 ->
    FUNDABLE), where the old `Fixed.marginal` is 0 (un-fundable).
  * the budget (`reps`) is the shrink lever: more reps -> a tighter median SE.
  * FAIL LOUD (ADR-0002): the C++ bench absent/unbuilt RAISES, a non-PASS RAISES, and a run with NO
    `leaf_requests_total` RAISES (g_core is a LEAF rate — no leaf count, no measurement) — never a
    silent fall-back to the seed-as-if-measured; `get_seed()` stays the DISTRUST fallback.

These run WITHOUT the live binary by exercising `_estimate_from_raw` on a synthesized per-rep pool
(the §8 discipline — the Estimate SHAPE + bootstrap SE + marginal are pool-driven). A binary-gated
tail RUNS the real bench (sole-workload, taskset -c 0): it must report ~76k leaves/s/core and produce
a shrinkable Estimate. The estimate/bench modules live under tools/analysis/leaf_eval_bound/ (no __init__.py
— imported by sys.path), so this test prepends those directories.

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
    "tools", "analysis",
)
_BENCH = os.path.join(_OT, "leaf_eval_bound", "benchmarks")
for _p in (_OT, _BENCH):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from leaf_eval_bound.benchmarks import bench_g_core as B  # noqa: E402  — the module under test
from leaf_eval_bound.contract import estimate as E  # noqa: E402  — the contract


def _leaf_pool(median: float = 76000.0, sigma: float = 500.0, n: int = 8, seed: int = 3) -> list[float]:
    """A plausible per-rep leaves/s/core pool: a tight near-symmetric cluster around `median` (the
    shape the C++ bench's per-rep leaf_requests_total/(n_tasks*serial) readings actually have —
    verified ~75.9k–77.2k leaves/s/core across reps). A real spread, a well-defined median."""
    rng = np.random.default_rng(seed)
    return [float(v) for v in (median + rng.normal(0.0, sigma, n))]


def _raw(pool: list[float]) -> dict:
    """A `_measure_raw()`-shaped dict over a synthesized per-rep pool (no live binary)."""
    return {
        "g_core_leaves_per_core": float(np.median(pool)),
        "per_rep_leaves_per_sec": pool,
        "leaf_requests_total": 16122,
        "lpd": 16122 / 32.0,
        "reps": len(pool),
        "n_tasks": 32,
        "config": "synthetic",
    }


# --------------------------------------------------------------------------- #
# 1. The reclassification: _estimate_from_raw builds a SHRINKABLE QuantileLaw, not a Fixed pin.
# --------------------------------------------------------------------------- #
def test_g_core_estimate_is_a_shrinkable_quantile_law_not_a_fixed_pin() -> None:
    """The defect was a `Fixed` (un-shrinkable) Estimate; the fix is a `QuantileLaw` (median) with a
    BOOTSTRAP median SE. This is the ADR-0008 reclassification: g_core is a MEASURED quantity, not a pin."""
    est = B._estimate_from_raw(_raw(_leaf_pool()))
    assert isinstance(est.shrink, E.QuantileLaw)      # SHRINKABLE — not E.Fixed
    assert not isinstance(est.shrink, E.Fixed)
    assert est.kind == "median"
    assert est.family == (E.CIFamily.EMPIRICAL,)      # a sample-quantile CI, not a NORMAL prior
    assert est.support == (E.Support.POSITIVE,)
    assert est.k == 1
    assert est.theta_hat[0] == pytest.approx(float(np.median(_leaf_pool())))
    assert est.is_valid()


def test_g_core_cov_is_a_real_bootstrap_se_not_a_fixed_sigma() -> None:
    """The variance authority is a REAL bootstrap median SE in `cov` (a positive, finite spread the loop
    can shrink), NOT the seed's frozen declared σ=9000 wrapped un-divided (the old `Fixed` pin's
    cov=[[9000²]])."""
    est = B._estimate_from_raw(_raw(_leaf_pool(sigma=500.0, n=8)))
    boot_se = math.sqrt(float(est.cov[0, 0]))
    assert boot_se > 0.0 and math.isfinite(boot_se)
    # it is the per-rep pool's order-statistic spread (sub-σ for a tight gen-ceiling pool), NOT σ=9000.
    assert boot_se < 9000.0


def test_g_core_marginal_is_negative_so_the_loop_can_fund_it() -> None:
    """THE PAYOFF (§1 D2 / §2.3): the typed `ShrinkLaw.marginal_dvar_deffort` is < 0 for the QuantileLaw
    (so the driver's `_fundability` `A_i = −marginal·n²` is > 0 -> g_core is FUNDABLE -> the loop samples
    it), where the OLD `Fixed` pin's marginal is exactly 0 (`A_i = 0` -> never fundable -> the STALL)."""
    est = B._estimate_from_raw(_raw(_leaf_pool()))
    sigma_ii = float(est.cov[0, 0])
    n_eff = float(est.shrink.n)
    marg_new = est.shrink.marginal_dvar_deffort(sigma_ii, n_eff)
    marg_old = E.Fixed().marginal_dvar_deffort(sigma_ii, n_eff)
    assert marg_new < 0.0          # shrinkable: one more reading lowers the variance
    assert marg_old == 0.0         # the punt: irreducible, the un-fundable stall
    # the driver's fundability mask: A_i = −marginal·n² (>0 fundable, ==0 not).
    A_new = -marg_new * n_eff ** 2
    A_old = -marg_old * n_eff ** 2
    assert A_new > 0.0             # FUNDED
    assert A_old == 0.0            # un-fundable (the stall the fix removes)


def test_g_core_quantile_law_is_self_consistent_with_cov() -> None:
    """The shipped `QuantileLaw(p=0.5, f_at_q, n)` is SELF-CONSISTENT with the `cov` it ships beside (P1
    single-home): its implied `p(1−p)/(n·f̂²)` equals the bootstrap `cov[0,0]` exactly."""
    est = B._estimate_from_raw(_raw(_leaf_pool()))
    ql = est.shrink
    assert isinstance(ql, E.QuantileLaw) and ql.p == 0.5
    implied = 0.25 / (ql.n * float(ql.f_at_q[0]) ** 2)
    assert math.isclose(implied, float(est.cov[0, 0]), rel_tol=1e-12)


def test_g_core_estimate_jsonb_round_trips() -> None:
    """The measured g_core Estimate is an exact jsonb round-trip (the §5 SSOT serialization)."""
    est = B._estimate_from_raw(_raw(_leaf_pool()))
    rt = E.from_jsonb(E.to_jsonb(est))
    assert np.allclose(rt.theta_hat, est.theta_hat)
    assert np.allclose(rt.cov, est.cov)
    assert isinstance(rt.shrink, E.QuantileLaw)
    assert rt.shrink.n == est.shrink.n
    assert rt.family == est.family and rt.kind == est.kind


# --------------------------------------------------------------------------- #
# 2. The budget (reps) is the shrink lever: more reps -> a tighter median SE.
# --------------------------------------------------------------------------- #
def test_more_reps_tightens_the_se() -> None:
    """The Neyman loop sizes g_core's measurement by `reps`; a bigger pool gives a tighter median SE (so
    more/longer runs -> a tighter g_core CI -> a tighter generation-arm CI — the loop's lever)."""
    se_small = math.sqrt(float(B._estimate_from_raw(_raw(_leaf_pool(n=4, seed=1))).cov[0, 0]))
    se_large = math.sqrt(float(B._estimate_from_raw(_raw(_leaf_pool(n=64, seed=1))).cov[0, 0]))
    assert se_large < se_small


# --------------------------------------------------------------------------- #
# 3. FAIL LOUD (ADR-0002): the C++ bench absent / non-PASS / no-leaf-count RAISES — never a silent seed.
# --------------------------------------------------------------------------- #
def test_missing_cpp_bench_raises_never_silently_seeds(monkeypatch) -> None:
    """ADR-0002: if the C++ gen-ceiling bench is not built, `_measure_raw()` RAISES — it NEVER falls back
    to the 76000 seed as if measured (the punt this module removes). The seed stays the DISTRUST fallback
    (`get_seed()`), not a measured-result substitute."""
    monkeypatch.setattr(B, "_BENCH_BIN", os.path.join(B._REPO_ROOT, "cpp", "build", "DOES-NOT-EXIST"))
    with pytest.raises(FileNotFoundError, match="not built"):
        B._measure_raw(reps=4)


def test_get_seed_is_the_distrust_fallback_and_unchanged() -> None:
    """`get_seed()` stays the v1 76000 leaves/s/core grounding (the DISTRUST fallback the manifest seed
    path uses) — the fix changes the MEASURED path, not the seed."""
    seed = B.get_seed()
    assert seed.name == "g_core"
    assert seed.mean == 76000.0
    assert seed.unit == "leaves/s/core"
    # a MEASURED quantity (the §3 PIN-now/measurable-later class), NOT a true constant.
    assert seed.needs_measurement is True
    assert seed.constant is False


def test_non_pass_cpp_output_raises(monkeypatch) -> None:
    """ADR-0002: a bench run that does not report `RESULT: PASS` (e.g. a serial/parallel mismatch FAIL,
    or truncated output) RAISES rather than fabricating a rate. Driven by stubbing the subprocess so no
    live binary is needed."""
    class _Proc:
        returncode = 0
        stdout = "config: ...\nRESULT: FAIL (2 mismatches between serial and parallel)\n"
        stderr = ""

    monkeypatch.setattr(B.os.path, "isfile", lambda p: True)
    monkeypatch.setattr(B.os, "access", lambda p, m: True)
    monkeypatch.setattr(B.shutil, "which", lambda _x: None)        # skip taskset in the stub
    monkeypatch.setattr(B.subprocess, "run", lambda *a, **k: _Proc())
    with pytest.raises(RuntimeError, match="RESULT: PASS"):
        B._measure_raw(reps=4)


def test_missing_leaf_count_raises(monkeypatch) -> None:
    """ADR-0002 (g_core-specific): g_core is a LEAF rate, so a PASS run with per-rep timings but NO
    `leaf_requests_total` has no numerator to scale by — it RAISES rather than fabricating a leaf rate
    or silently seeding. (R_gen could report a dps rate without the leaf count; g_core cannot.)"""
    class _Proc:
        returncode = 0
        stdout = ("config: ...\n"
                  "rep 0: serial=0.2124s parallel=0.2106s\n"
                  "rep 1: serial=0.2119s parallel=0.2105s\n"
                  "RESULT: PASS speedup=0.9927 serial_dps=153.2 parallel_dps=152\n")  # no leaf_requests_total=
        stderr = ""

    monkeypatch.setattr(B.os.path, "isfile", lambda p: True)
    monkeypatch.setattr(B.os, "access", lambda p, m: True)
    monkeypatch.setattr(B.shutil, "which", lambda _x: None)
    monkeypatch.setattr(B.subprocess, "run", lambda *a, **k: _Proc())
    with pytest.raises(RuntimeError, match="leaf_requests_total"):
        B._measure_raw(reps=4)


# --------------------------------------------------------------------------- #
# 4. BINARY-GATED: RUN the real C++ gen-ceiling bench (sole-workload) — ~76k leaves/s/core, shrinkable.
# --------------------------------------------------------------------------- #
def _bench_built() -> bool:
    return os.path.isfile(B._BENCH_BIN) and os.access(B._BENCH_BIN, os.X_OK) \
        and os.path.isfile(B._INSTANCE) and os.path.isfile(B._FACES)


@pytest.mark.skipif(not _bench_built(), reason="the C++ gen-ceiling bench is not built")
def test_live_cpp_bench_reports_about_76k_leaves_per_core_and_is_shrinkable() -> None:
    """BEHAVIORAL verification of the C++ bench (ADR-0009 — RUN it; mypy/lint are blind to cpp/). The
    live `measure()` runs the gen-ceiling bench (eval mocked by the bench's DetNet, sole-workload) and
    must (a) report ~76k leaves/s/core at the sims256/m24 config (the seed grounding) and (b) produce a
    SHRINKABLE `QuantileLaw` Estimate whose marginal is < 0 (FUNDABLE — the loop samples it). Pin
    `taskset -c 0` and run sole-workload."""
    est = B.measure(reps=6)
    assert isinstance(est.shrink, E.QuantileLaw)
    assert est.kind == "median" and est.family == (E.CIFamily.EMPIRICAL,)
    # ~76k leaves/s/core (the gen-ceiling grounding). A generous band — this is a behavioral sanity gate,
    # not a tight equivalence pin (the rate carries scheduler jitter; verified ~75.9k–77.2k across reps).
    assert 65000.0 <= float(est.theta_hat[0]) <= 87500.0
    se = math.sqrt(float(est.cov[0, 0]))
    assert se > 0.0 and math.isfinite(se)
    marg = est.shrink.marginal_dvar_deffort(float(est.cov[0, 0]), float(est.shrink.n))
    assert marg < 0.0           # SHRINKABLE -> A_i > 0 -> the loop funds it (the payoff)


@pytest.mark.skipif(not _bench_built(), reason="the C++ gen-ceiling bench is not built")
def test_live_cpp_bench_cross_reads_lpd_about_500() -> None:
    """The SAME bench's `leaf_requests_total/n_tasks` is the per-decision distinct-node count — the LPD
    grounding (500). A behavioral cross-read confirming the gen-ceiling config is the sims256/m24 tree
    the LPD=500 design pin cites (so g_core and R_gen/LPD come from ONE measurement — the §4 finding)."""
    res = B._measure_raw(reps=4)
    assert res["lpd"] is not None
    assert 440.0 <= float(res["lpd"]) <= 560.0   # leaf_requests_total/32 ≈ 504 (verified)
