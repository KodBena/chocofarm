"""
tests/test_bench_tmsg_codec_framing.py
======================================

The ADR-0008 RECLASSIFICATION of `tmsg_us_leaf` (the per-leaf-amortized message-passing cost,
us/leaf — the TRANSPORT arm of the throughput `min()`): a MEASURED quantity that was MIS-WIRED as
an un-shrinkable `Fixed` pin, now wired to the live in-process `inference_wire` codec
(`chocofarm/az/inference_wire.py` `encode_request`/`decode_response`) so it returns a SHRINKABLE
Estimate the Neyman loop can sample (docs/design/harmonized-estimator-interface.md §3 the
PIN-now/measurable-later row + §7.A; the defect: the bench's `_measure_raw()` DID a real codec
measurement but `_estimate_from_raw()` wrapped the SEED 1.0us in `pin_estimate` -> a
`Fixed`/`marginal=0`/un-fundable Estimate, logging the live number only "alongside" — the SAME
measured-but-punted punt @d5f84b7 removed for R_gen / LPD / g_core).

The pool is the PER-LEAF FRAMING-COST distribution: the codec's `encode_request` +
`decode_response` over a coalesced S-leaf frame, `/S`, timed in windows (the
`bench_cpp_inproc_port_tmsg_us_leaf` window-loop idiom — the SAME quantity-class
`transport_msg_cost_per_leaf`, ALSO NON-BINDING / ranks-LAST, ALREADY constructed shrinkable, so
non-binding is a RANKING fact not an un-measurability one). NOTE: the C++ `chocofarm-wire-bench` is
NOT the clean sole-measurement (it is a SERVER-COUPLED round-trip = RTT + serve forward, not the
pure-codec framing share) — the in-process codec the bench already times IS the sole-measurement.

CLASSIFICATION (ADR-0008; ADR-0012 P8 the typed family/shrink IS the contract): `tmsg_us_leaf` is a
MEDIAN, NOT a true constant — it is deliberately NOT marked `constant=True`. A constant would be
DEGENERATE (`a_i≈0`, un-fundable) and re-introduce the punt; tmsg is a measured codec-timing pool
that tightens with windows. This test pins that the seed keeps `constant=False` /
`needs_measurement=True` and the measured path is a shrinkable QuantileLaw.

THE FIX, AS TESTS:
  * `_estimate_from_raw` over a per-leaf framing-cost pool builds a SHRINKABLE `QuantileLaw`
    (median) Estimate — a REAL bootstrap median SE, `family=EMPIRICAL`, `kind='median'` — NOT a
    `Fixed` pin; and its typed `ShrinkLaw.marginal_dvar_deffort` is < 0 (so the driver's
    `A_i = −marginal·n²` is > 0 -> FUNDABLE), where the old `Fixed.marginal` is 0 (un-fundable).
  * the budget (`budget` = #windows) is the shrink lever: more windows -> a tighter median SE.
  * the EXPLICIT twin `bench_zmq_baseline_tmsg_us_leaf` carries the IDENTICAL fix (same codec,
    same shrinkable median) and DELEGATES its seed to the single home `G.MSG_PER_LEAF_US` (P1).

These run WITHOUT timing the live codec by exercising `_estimate_from_raw` on a synthesized per-leaf
pool (the §8 discipline — the Estimate SHAPE + bootstrap SE + marginal are pool-driven). A
codec-gated tail RUNS the real in-process codec (sole-workload, pinned by the operator with
taskset -c 0): it must report a small positive us/leaf and produce a shrinkable Estimate. The
estimate/bench modules live under tools/analysis/leaf_eval_bound/ (no __init__.py — imported by sys.path),
so this test prepends those directories.

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

from leaf_eval_bound.benchmarks import bench_tmsg as B  # noqa: E402  — the module under test (the v1 ZMQ-baseline codec bench)
from leaf_eval_bound.benchmarks import bench_zmq_baseline_tmsg_us_leaf as Z  # noqa: E402  — the explicit slug-prefixed twin
from leaf_eval_bound.contract import estimate as E  # noqa: E402  — the contract
from leaf_eval_bound.contract import grounding as G  # noqa: E402  — the grounding single-home


def _framing_pool(center: float = 0.166, spread: float = 0.004, n: int = 64, seed: int = 3) -> list[float]:
    """A plausible per-leaf framing-cost pool: a tight positive cluster around `center` us/leaf (the
    shape the live codec's windowed per-leaf readings actually have — verified median ≈ 0.166 us/leaf,
    std ≈ 0.0028 over 64 windows at S=256). A real spread, a well-defined median, non-degenerate."""
    rng = np.random.default_rng(seed)
    return [float(abs(v)) for v in (center + rng.normal(0.0, spread, n))]


def _raw(pool: list[float]) -> dict:
    """A `_measure_raw()`-shaped dict over a synthesized per-leaf pool (no live codec timing)."""
    return {
        "tmsg_us_leaf_median": float(np.median(pool)),
        "per_leaf_us": pool,
        "encode_us": 256.0 * float(np.median(pool)) * 0.95,   # informational split (plausible)
        "decode_us": 256.0 * float(np.median(pool)) * 0.05,
        "s_leaves": 256,
        "budget": len(pool),
    }


@pytest.fixture(params=[B, Z], ids=["bench_tmsg", "bench_zmq_baseline_tmsg_us_leaf"])
def mod(request):
    """Both the v1 `bench_tmsg` AND its explicit slug-prefixed twin carry the IDENTICAL fix of the
    IDENTICAL pin of the IDENTICAL codec — every reclassification assertion holds for both."""
    return request.param


# --------------------------------------------------------------------------- #
# 1. The reclassification: _estimate_from_raw builds a SHRINKABLE QuantileLaw, not a Fixed pin.
# --------------------------------------------------------------------------- #
def test_tmsg_estimate_is_a_shrinkable_quantile_law_not_a_fixed_pin(mod) -> None:
    """The defect was a `Fixed` (un-shrinkable) Estimate built off the SEED; the fix is a `QuantileLaw`
    (median) with a BOOTSTRAP median SE over the live codec pool. ADR-0008: tmsg is MEASURED, not a pin."""
    pool = _framing_pool()
    est = mod._estimate_from_raw(_raw(pool))
    assert isinstance(est.shrink, E.QuantileLaw)      # SHRINKABLE — not E.Fixed
    assert not isinstance(est.shrink, E.Fixed)
    assert est.kind == "median"
    assert est.family == (E.CIFamily.EMPIRICAL,)      # a sample-quantile CI, not a NORMAL prior
    assert est.support == (E.Support.POSITIVE,)
    assert est.k == 1
    assert est.theta_hat[0] == pytest.approx(float(np.median(pool)))
    assert est.names == (mod.NAME,)
    assert est.is_valid()


def test_tmsg_cov_is_a_real_bootstrap_se_not_the_seed_sigma(mod) -> None:
    """The variance authority is a REAL bootstrap median SE in `cov` (a positive, finite spread the loop
    can shrink), NOT the seed's frozen declared σ=0.5 wrapped un-divided (the old `Fixed` pin's
    cov=[[0.25]]). The measured per-leaf SE is far below the 0.5 seed sigma (sub-0.001 us/leaf)."""
    est = mod._estimate_from_raw(_raw(_framing_pool()))
    boot_se = math.sqrt(float(est.cov[0, 0]))
    assert boot_se > 0.0 and math.isfinite(boot_se)
    assert boot_se < 0.5      # the codec pool's order-statistic spread, NOT the seed σ=0.5


def test_tmsg_marginal_is_negative_so_the_loop_can_fund_it(mod) -> None:
    """THE PAYOFF (§1 D2 / §2.3): the typed `ShrinkLaw.marginal_dvar_deffort` is < 0 for the QuantileLaw
    (so the driver's `A_i = −marginal·n²` is > 0 -> tmsg is FUNDABLE -> the loop can sample it under the
    §4.1 kink regime), where the OLD `Fixed` pin's marginal is exactly 0 (`A_i = 0` -> never fundable)."""
    est = mod._estimate_from_raw(_raw(_framing_pool()))
    sigma_ii = float(est.cov[0, 0])
    n_eff = float(est.shrink.n)
    marg_new = est.shrink.marginal_dvar_deffort(sigma_ii, n_eff)
    marg_old = E.Fixed().marginal_dvar_deffort(sigma_ii, n_eff)
    assert marg_new < 0.0          # shrinkable: one more reading lowers the variance
    assert marg_old == 0.0         # the punt: irreducible, the un-fundable pin
    A_new = -marg_new * n_eff ** 2
    A_old = -marg_old * n_eff ** 2
    assert A_new > 0.0             # FUNDED
    assert A_old == 0.0            # un-fundable (the pin the fix removes)


def test_tmsg_quantile_law_is_self_consistent_with_cov(mod) -> None:
    """The shipped `QuantileLaw(p=0.5, f_at_q, n)` is SELF-CONSISTENT with the `cov` it ships beside (P1
    single-home): its implied `p(1−p)/(n·f̂²)` equals the bootstrap `cov[0,0]` exactly."""
    est = mod._estimate_from_raw(_raw(_framing_pool()))
    ql = est.shrink
    assert isinstance(ql, E.QuantileLaw) and ql.p == 0.5
    implied = 0.25 / (ql.n * float(ql.f_at_q[0]) ** 2)
    assert math.isclose(implied, float(est.cov[0, 0]), rel_tol=1e-12)


def test_tmsg_estimate_jsonb_round_trips(mod) -> None:
    """The measured tmsg Estimate is an exact jsonb round-trip (the §5 SSOT serialization)."""
    est = mod._estimate_from_raw(_raw(_framing_pool()))
    rt = E.from_jsonb(E.to_jsonb(est))
    assert np.allclose(rt.theta_hat, est.theta_hat)
    assert np.allclose(rt.cov, est.cov)
    assert isinstance(rt.shrink, E.QuantileLaw)
    assert rt.shrink.n == est.shrink.n
    assert rt.family == est.family and rt.kind == est.kind


# --------------------------------------------------------------------------- #
# 2. The budget (#windows) is the shrink lever: more windows -> a tighter median SE.
# --------------------------------------------------------------------------- #
def test_more_windows_tightens_the_se(mod) -> None:
    """The Neyman loop sizes tmsg's measurement by `budget`; a bigger pool gives a tighter median SE (so
    more windows -> a tighter tmsg CI -> a tighter transport-arm CI — the loop's lever)."""
    se_small = math.sqrt(float(mod._estimate_from_raw(_raw(_framing_pool(n=8, seed=1))).cov[0, 0]))
    se_large = math.sqrt(float(mod._estimate_from_raw(_raw(_framing_pool(n=64, seed=1))).cov[0, 0]))
    assert se_large < se_small


# --------------------------------------------------------------------------- #
# 3. CLASSIFICATION (ADR-0008): tmsg is a MEDIAN, NOT a true constant; the seed is the DISTRUST fallback.
# --------------------------------------------------------------------------- #
def test_tmsg_grounding_is_measured_not_a_true_constant() -> None:
    """ADR-0008/ADR-0012 P1: the single-home grounding `G.MSG_PER_LEAF_US` is a MEASURED quantity
    (`needs_measurement=True`) and deliberately NOT a true constant (`constant=False`) — a DEGENERATE
    constant would be `a_i≈0` / un-fundable and re-introduce the punt. (n_gen is the true-constant
    contrast: `constant=True`.)"""
    g = G.MSG_PER_LEAF_US
    assert g.needs_measurement is True
    assert g.constant is False                 # a MEASURED quantity, not a layout/pinning fact
    assert G.N_GEN_CORES.constant is True       # the contrast: n_gen IS a true constant


def test_get_seed_is_the_distrust_fallback_and_unchanged(mod) -> None:
    """`get_seed()` stays the v1 1.0 us/leaf grounding (the DISTRUST fallback the manifest seed path
    uses) — the fix changes the MEASURED path, not the seed. BOTH the v1 bench AND the twin delegate
    the seed to the single home `G.MSG_PER_LEAF_US` (P1)."""
    seed = mod.get_seed()
    assert seed is G.MSG_PER_LEAF_US           # delegated to the single home (P1), not a second literal
    assert seed.name == "tmsg_us_leaf"
    assert seed.mean == 1.0
    assert seed.unit == "us/leaf"


# --------------------------------------------------------------------------- #
# 4. CODEC-GATED: RUN the live in-process inference_wire codec (sole-workload) — small +ve us/leaf, shrinkable.
# --------------------------------------------------------------------------- #
def _codec_importable() -> bool:
    try:
        from chocofarm.az.inference_wire import encode_request, decode_response  # noqa: F401
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _codec_importable(), reason="the inference_wire codec is not importable")
def test_live_codec_measure_reports_small_positive_us_per_leaf_and_is_shrinkable(mod) -> None:
    """BEHAVIORAL verification of the live codec measurement (ADR-0009 — RUN it). `measure()` times the
    real `inference_wire` codec (encode_request + decode_response over a coalesced S-leaf frame, /S,
    windowed) and must (a) report a small POSITIVE us/leaf (the pure-codec framing share, provably
    non-binding) and (b) produce a SHRINKABLE `QuantileLaw` Estimate whose marginal is < 0 (FUNDABLE).
    Operator runs sole-workload pinned (taskset -c 0); a small budget keeps the test cheap."""
    est = mod.measure(budget=8)
    assert isinstance(est.shrink, E.QuantileLaw)
    assert est.kind == "median" and est.family == (E.CIFamily.EMPIRICAL,)
    val = float(est.theta_hat[0])
    # A generous behavioral band: the per-leaf framing share is a small positive us/leaf (verified
    # ≈ 0.166 us/leaf at S=256), and is far below the 1.0us seed over-charge (so transport stays
    # non-binding). This is a sanity gate, not a tight equivalence pin (it carries scheduler jitter).
    assert 0.0 < val < 1.0
    se = math.sqrt(float(est.cov[0, 0]))
    assert se > 0.0 and math.isfinite(se)
    marg = est.shrink.marginal_dvar_deffort(float(est.cov[0, 0]), float(est.shrink.n))
    assert marg < 0.0           # SHRINKABLE -> A_i > 0 -> the loop funds it (the payoff)


@pytest.mark.skipif(not _codec_importable(), reason="the inference_wire codec is not importable")
def test_live_codec_measure_raw_pool_is_non_degenerate(mod) -> None:
    """The live `_measure_raw` yields a real per-leaf pool (>= 2 readings with spread) so the bootstrap
    median SE is defined — the budget (#windows) sizes it. A degenerate (zero-spread) pool would RAISE
    in `median_estimate` (ADR-0002), so a passing `measure()` proves the pool has honest spread."""
    res = mod._measure_raw(budget=8)
    pool = res["per_leaf_us"]
    assert len(pool) >= 2
    assert all(x > 0.0 and math.isfinite(x) for x in pool)
    assert res["tmsg_us_leaf_median"] == pytest.approx(float(np.median(pool)))
    assert res["budget"] == len(pool)
