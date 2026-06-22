"""
tests/test_bench_lpd_cpp_leaf_count.py
======================================

The ADR-0008 RECLASSIFICATION of `LPD` (leaves per recorded decision, leaves/decision): a MEASURED
quantity that was MIS-WIRED as an un-shrinkable `Fixed` pin, now wired to the live C++ gen-ceiling
bench (`cpp/build/chocofarm-search-runtime-bench`, source cpp/src/search_runtime_bench.cpp) so it
returns a SHRINKABLE Estimate the Neyman loop can sample (docs/design/harmonized-estimator-interface.md
§3 the PIN-now/measurable-later row + §7.A; the defect: the bench PUNTED — `_measure_raw()` returned
the seed 500, `_estimate_from_raw()` wrapped it in `pin_estimate` -> a `Fixed`/`marginal=0`/un-fundable
Estimate -> `untrusted_drive`'s generation arm STALLED — the SAME punt @d5f84b7 removed for R_gen).

The pool is the PER-DECISION leaf-count distribution: each task the bench runs is ONE independent
Gumbel-AZ decision, so its `leaf_requests` IS that decision's leaf count (an LPD reading). The SAME
instrumented run grounds R_gen, g_core, AND LPD (ADR-0012 P1 single-home); the C++ bench prints the
per-decision pool on a `leaf_requests_per_task=` line (additive output, no search edit).

CLASSIFICATION (ADR-0008): LPD is a MEDIAN, NOT a true constant — it is deliberately NOT marked
`constant=True`. A constant would be DEGENERATE (`a_i≈0`, un-fundable) and re-introduce the stall;
LPD is a measured per-decision count that varies tree-to-tree (verified non-degenerate). This test
pins that the seed keeps `constant=False` / `needs_measurement=True` and the measured path is a
shrinkable QuantileLaw.

THE FIX, AS TESTS:
  * `_estimate_from_raw` over a per-decision leaf-count pool builds a SHRINKABLE `QuantileLaw`
    (median) Estimate — a REAL bootstrap median SE, `family=EMPIRICAL`, `kind='median'` — NOT a
    `Fixed` pin; and its typed `ShrinkLaw.marginal_dvar_deffort` is < 0 (so the driver's
    `A_i = −marginal·n²` is > 0 -> FUNDABLE), where the old `Fixed.marginal` is 0 (un-fundable).
  * the budget (`trials`) is the shrink lever: more decisions -> a tighter median SE.
  * FAIL LOUD (ADR-0002): the C++ bench absent/unbuilt RAISES — never a silent fall-back to the
    seed-as-if-measured (the punt this module removes); `get_seed()` stays the DISTRUST fallback.

These run WITHOUT the live binary by exercising `_estimate_from_raw` on a synthesized per-decision
pool (the §8 discipline — the Estimate SHAPE + bootstrap SE + marginal are pool-driven). A binary-gated
tail RUNS the real bench (sole-workload, taskset -c 0): it must report ~500 leaves/decision and produce
a shrinkable Estimate. The estimate/bench modules live under tools/analysis/leaf_eval_bound/ (no
__init__.py — imported by sys.path), so this test prepends those directories.

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

from leaf_eval_bound.benchmarks import bench_lpd as B  # noqa: E402  — the module under test
from leaf_eval_bound.contract import estimate as E  # noqa: E402  — the contract


def _leaf_pool(center: float = 503.0, spread: float = 3.0, n: int = 8, seed: int = 3) -> list[float]:
    """A plausible per-decision leaf-count pool: a tight integer-ish cluster around `center` (the shape
    the C++ bench's per-task `leaf_requests` actually have — verified [504,500,503,500,505,502,508,506]
    at sims256/m24). A real spread, a well-defined median, non-degenerate."""
    rng = np.random.default_rng(seed)
    return [float(round(v)) for v in (center + rng.normal(0.0, spread, n))]


def _raw(pool: list[float]) -> dict:
    """A `_measure_raw()`-shaped dict over a synthesized per-decision pool (no live binary)."""
    total = int(sum(pool))
    return {
        "lpd": float(np.median(pool)),
        "per_decision_leaves": pool,
        "leaf_requests_total": total,
        "lpd_mean_cross_read": total / float(len(pool)),
        "n_tasks": len(pool),
        "config": "synthetic",
    }


# --------------------------------------------------------------------------- #
# 1. The reclassification: _estimate_from_raw builds a SHRINKABLE QuantileLaw, not a Fixed pin.
# --------------------------------------------------------------------------- #
def test_lpd_estimate_is_a_shrinkable_quantile_law_not_a_fixed_pin() -> None:
    """The defect was a `Fixed` (un-shrinkable) Estimate; the fix is a `QuantileLaw` (median) with a
    BOOTSTRAP median SE. This is the ADR-0008 reclassification: LPD is a MEASURED quantity, not a pin."""
    est = B._estimate_from_raw(_raw(_leaf_pool()))
    assert isinstance(est.shrink, E.QuantileLaw)      # SHRINKABLE — not E.Fixed
    assert not isinstance(est.shrink, E.Fixed)
    assert est.kind == "median"
    assert est.family == (E.CIFamily.EMPIRICAL,)      # a sample-quantile CI, not a NORMAL prior
    assert est.support == (E.Support.POSITIVE,)
    assert est.k == 1
    assert est.theta_hat[0] == pytest.approx(float(np.median(_leaf_pool())))
    assert est.is_valid()


def test_lpd_cov_is_a_real_bootstrap_se_not_the_frozen_design_sigma() -> None:
    """The variance authority is a REAL bootstrap median SE in `cov` (a positive, finite spread the loop
    can shrink), NOT the seed's frozen declared σ=25 wrapped un-divided (the old `Fixed` pin's
    cov=[[625]])."""
    est = B._estimate_from_raw(_raw(_leaf_pool(spread=3.0, n=8)))
    boot_se = math.sqrt(float(est.cov[0, 0]))
    assert boot_se > 0.0 and math.isfinite(boot_se)
    # it is the per-decision pool's order-statistic spread (a few leaves for a tight gen-ceiling pool),
    # NOT the design-pin σ=25.
    assert boot_se < 25.0


def test_lpd_marginal_is_negative_so_the_loop_can_fund_it() -> None:
    """THE PAYOFF (§1 D2 / §2.3): the typed `ShrinkLaw.marginal_dvar_deffort` is < 0 for the QuantileLaw
    (so the driver's `A_i = −marginal·n²` is > 0 -> LPD is FUNDABLE -> the loop samples it), where the
    OLD `Fixed` pin's marginal is exactly 0 (`A_i = 0` -> never fundable -> the STALL)."""
    est = B._estimate_from_raw(_raw(_leaf_pool()))
    sigma_ii = float(est.cov[0, 0])
    n_eff = float(est.shrink.n)
    marg_new = est.shrink.marginal_dvar_deffort(sigma_ii, n_eff)
    marg_old = E.Fixed().marginal_dvar_deffort(sigma_ii, n_eff)
    assert marg_new < 0.0          # shrinkable: one more decision lowers the variance
    assert marg_old == 0.0         # the punt: irreducible, the un-fundable stall
    # the driver's fundability mask: A_i = −marginal·n² (>0 fundable, ==0 not).
    A_new = -marg_new * n_eff ** 2
    A_old = -marg_old * n_eff ** 2
    assert A_new > 0.0             # FUNDED
    assert A_old == 0.0            # un-fundable (the stall the fix removes)


def test_lpd_quantile_law_is_self_consistent_with_cov() -> None:
    """The shipped `QuantileLaw(p=0.5, f_at_q, n)` is SELF-CONSISTENT with the `cov` it ships beside (P1
    single-home): its implied `p(1−p)/(n·f̂²)` equals the bootstrap `cov[0,0]` exactly."""
    est = B._estimate_from_raw(_raw(_leaf_pool()))
    ql = est.shrink
    assert isinstance(ql, E.QuantileLaw) and ql.p == 0.5
    implied = 0.25 / (ql.n * float(ql.f_at_q[0]) ** 2)
    assert math.isclose(implied, float(est.cov[0, 0]), rel_tol=1e-12)


def test_lpd_estimate_jsonb_round_trips() -> None:
    """The measured LPD Estimate is an exact jsonb round-trip (the §5 SSOT serialization)."""
    est = B._estimate_from_raw(_raw(_leaf_pool()))
    rt = E.from_jsonb(E.to_jsonb(est))
    assert np.allclose(rt.theta_hat, est.theta_hat)
    assert np.allclose(rt.cov, est.cov)
    assert isinstance(rt.shrink, E.QuantileLaw)
    assert rt.shrink.n == est.shrink.n
    assert rt.family == est.family and rt.kind == est.kind


# --------------------------------------------------------------------------- #
# 2. The budget (trials) is the shrink lever: more decisions -> a tighter median SE.
# --------------------------------------------------------------------------- #
def test_more_decisions_tighten_the_se() -> None:
    """The Neyman loop sizes LPD's measurement by `trials` (decisions sampled); a bigger pool gives a
    tighter median SE (so more decisions -> a tighter LPD CI -> a tighter generation-arm CI)."""
    se_small = math.sqrt(float(B._estimate_from_raw(_raw(_leaf_pool(n=4, seed=1))).cov[0, 0]))
    se_large = math.sqrt(float(B._estimate_from_raw(_raw(_leaf_pool(n=64, seed=1))).cov[0, 0]))
    assert se_large < se_small


# --------------------------------------------------------------------------- #
# 3. ADR-0008 classification: LPD is a MEASURED median, NOT a true constant (would re-stall).
# --------------------------------------------------------------------------- #
def test_lpd_seed_is_a_measured_quantity_not_a_true_constant() -> None:
    """LPD must NOT be marked `constant=True` (a DEGENERATE true constant is un-fundable — it would
    re-introduce the stall the fix removes). The seed stays a MEASURED quantity (`constant=False`,
    `needs_measurement=True`): the ADR-0008 honest classification refuses the constant vocabulary."""
    seed = B.get_seed()
    assert seed.name == "LPD"
    assert seed.constant is False         # NOT a true constant — measured, fundable
    assert seed.needs_measurement is True  # the signature the measured path now honors (P8)


# --------------------------------------------------------------------------- #
# 4. FAIL LOUD (ADR-0002): the C++ bench absent/non-PASS RAISES — never a silent seed fall-back.
# --------------------------------------------------------------------------- #
def test_missing_cpp_bench_raises_never_silently_seeds(monkeypatch) -> None:
    """ADR-0002: if the C++ gen-ceiling bench is not built, `_measure_raw()` RAISES — it NEVER falls back
    to the 500 design pin as if measured (the punt this module removes). The seed stays the DISTRUST
    fallback (`get_seed()`), not a measured-result substitute."""
    monkeypatch.setattr(B, "_BENCH_BIN", os.path.join(B._REPO_ROOT, "cpp", "build", "DOES-NOT-EXIST"))
    with pytest.raises(FileNotFoundError, match="not built"):
        B._measure_raw(trials=4)


def test_get_seed_is_the_distrust_fallback_and_unchanged() -> None:
    """`get_seed()` stays the v1 500 leaves/decision grounding (the DISTRUST fallback the manifest seed
    path uses) — the fix changes the MEASURED path, not the seed."""
    seed = B.get_seed()
    assert seed.name == "LPD"
    assert seed.mean == 500.0
    assert seed.unit == "leaves/decision"


def test_non_pass_cpp_output_raises(monkeypatch) -> None:
    """ADR-0002: a bench run that does not report `RESULT: PASS` (e.g. a serial/parallel mismatch FAIL,
    or truncated output) RAISES rather than fabricating a pool. Driven by stubbing the subprocess so no
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
        B._measure_raw(trials=4)


def test_missing_per_task_line_raises(monkeypatch) -> None:
    """ADR-0002: a PASS run that does NOT print the `leaf_requests_per_task=` line (e.g. an old binary
    without the additive per-decision output) RAISES — the per-decision pool is the measurement, and a
    1-reading aggregate is not a fundable median. Never a fabricated pool."""
    class _Proc:
        returncode = 0
        stdout = ("config: ...\nrep 0: serial=0.05s parallel=0.05s\n"
                  "leaf_requests_total=4028 best_serial=0.05s best_parallel=0.05s\n"
                  "RESULT: PASS speedup=1.0 serial_dps=153 parallel_dps=153\n")
        stderr = ""

    monkeypatch.setattr(B.os.path, "isfile", lambda p: True)
    monkeypatch.setattr(B.os, "access", lambda p, m: True)
    monkeypatch.setattr(B.shutil, "which", lambda _x: None)
    monkeypatch.setattr(B.subprocess, "run", lambda *a, **k: _Proc())
    with pytest.raises(RuntimeError, match="leaf_requests_per_task"):
        B._measure_raw(trials=4)


# --------------------------------------------------------------------------- #
# 5. BINARY-GATED: RUN the real C++ gen-ceiling bench (sole-workload) — ~500 leaves/decision, shrinkable.
# --------------------------------------------------------------------------- #
def _bench_built() -> bool:
    return os.path.isfile(B._BENCH_BIN) and os.access(B._BENCH_BIN, os.X_OK) \
        and os.path.isfile(B._INSTANCE) and os.path.isfile(B._FACES)


@pytest.mark.skipif(not _bench_built(), reason="the C++ gen-ceiling bench is not built")
def test_live_cpp_bench_reports_about_500_lpd_and_is_shrinkable() -> None:
    """BEHAVIORAL verification of the C++ bench (ADR-0009 — RUN it; mypy/lint are blind to cpp/). The
    live `measure()` runs the gen-ceiling bench (eval mocked by the bench's DetNet, sole-workload) and
    must (a) report ~500 leaves/decision at the sims256/m24 config (the seed grounding) and (b) produce a
    SHRINKABLE `QuantileLaw` Estimate whose marginal is < 0 (FUNDABLE — the loop samples it). Pin
    `taskset -c 0` and run sole-workload."""
    est = B.measure(trials=16)
    assert isinstance(est.shrink, E.QuantileLaw)
    assert est.kind == "median" and est.family == (E.CIFamily.EMPIRICAL,)
    # ~500 leaves/decision (the gen-ceiling grounding). A generous band — a behavioral sanity gate, not a
    # tight equivalence pin (the per-decision count varies a few leaves tree-to-tree; verified ~500-508).
    assert 440.0 <= float(est.theta_hat[0]) <= 560.0
    se = math.sqrt(float(est.cov[0, 0]))
    assert se > 0.0 and math.isfinite(se)
    marg = est.shrink.marginal_dvar_deffort(float(est.cov[0, 0]), float(est.shrink.n))
    assert marg < 0.0           # SHRINKABLE -> A_i > 0 -> the loop funds it (the payoff)


@pytest.mark.skipif(not _bench_built(), reason="the C++ gen-ceiling bench is not built")
def test_live_cpp_bench_cross_read_matches_aggregate_lpd() -> None:
    """The per-decision pool's median and the aggregate cross-read `leaf_requests_total/n_tasks` are the
    SAME ~500 LPD grounding off ONE run (so R_gen, g_core, and LPD come from one measurement — the §4
    finding / ADR-0012 P1 single-home)."""
    res = B._measure_raw(trials=8)
    assert 440.0 <= float(res["lpd"]) <= 560.0                  # the per-decision median
    assert 440.0 <= float(res["lpd_mean_cross_read"]) <= 560.0  # leaf_requests_total/n_tasks ≈ 503.5
    assert len(res["per_decision_leaves"]) == res["n_tasks"]    # one reading per decision
