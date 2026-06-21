"""
tests/test_bench_futex_wake_tmsg_reclassification.py
====================================================

The ADR-0008 RECLASSIFICATION of `futex_wake_tmsg_us_leaf` (the per-leaf-amortized MESSAGE cost,
us/leaf, for the FUTEX-WAKE transport: the in-ring memcpy of one request row in + one reply row
out — registered quantity `transport_msg_cost_per_leaf_futex_wake`): a MEASURED quantity that was
MIS-WIRED as an un-shrinkable `Fixed` declared-spread pin, now wired to its OWN live windowed
ring-memcpy measurement so it returns a SHRINKABLE Estimate the Neyman loop can sample
(docs/design/harmonized-estimator-interface.md §3 MEDIAN row + §7.A).

The defect (verified live before the fix): `_measure_raw()` PERFORMED a real measurement of the
per-leaf ring traffic (~0.5 us/leaf), but `_estimate_from_raw()` DISCARDED it and returned
`pin_estimate(get_seed())` — a `Fixed`/`marginal=0`/un-fundable Estimate built off the v1 SEED
(0.1535 us/leaf), so the manifest TRUST path held a re-declared seed the bench's own measurement
contradicts, and the Neyman loop could never tighten it (`A_i = 0` -> the stall `bench_r_gen.py`
removed). This is the SAME measure-then-pin punt `bench_tmsg`/`bench_zmq_baseline` already had
fixed; the runnable per-leaf pool was in hand (the bench's own windowed ring memcpy, the same
windowing `bench_cpp_inproc_port_tmsg_us_leaf` uses), so the correct class is `median`.

CLASSIFICATION (ADR-0008; ADR-0012 P8 the typed family/shrink IS the contract; P1 single-home —
the measured per_leaf_us and the Estimate's theta_hat now share ONE home): the per-leaf ring memcpy
is a MEASURED, shrinkable median, NOT a declared-spread pin.

THE FIX, AS TESTS:
  * `_estimate_from_raw` over a per-window per-leaf pool builds a SHRINKABLE `QuantileLaw` (median)
    Estimate — a REAL bootstrap median SE, `family=EMPIRICAL`, `kind='median'` — NOT a `Fixed` pin;
    and its typed `ShrinkLaw.marginal_dvar_deffort` is < 0 (so the driver's `A_i = −marginal·n²` is
    > 0 -> FUNDABLE), where the old `Fixed.marginal` is 0 (un-fundable -> the stall).
  * the budget (`iters`) is the shrink lever: more leaves -> more windows -> a tighter median SE.
  * `get_seed()` stays the v1 ~0.15 us/leaf DISTRUST fallback (the SEED path), UNCHANGED — the fix
    changes the MEASURED path, not the seed.

Most assertions run WITHOUT timing the live ring by exercising `_estimate_from_raw` on a synthesized
per-leaf pool (the §8 discipline — the Estimate SHAPE + bootstrap SE + marginal are pool-driven). A
LIVE tail RUNS the bench's own in-process windowed measurement (NO external binary, NO geometry to
gate on — numpy + multiprocessing.shared_memory): it must report a positive us/leaf and a shrinkable
Estimate (ADR-0009 — run it; pin `taskset -c 0` is the operator's job, this is a behavioral gate).

The estimate/bench modules live under tools/analysis/OpenTURNS/ (no __init__.py — imported by
sys.path), so this test prepends those directories.

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

import bench_futex_wake_tmsg_us_leaf as B  # noqa: E402  — the module under test
import estimate as E  # noqa: E402  — the contract


def _framing_pool(center: float = 0.53, spread: float = 0.02, n: int = 40, seed: int = 3) -> list[float]:
    """A plausible per-window per-leaf ring-memcpy pool: a tight positive cluster around `center` us/leaf
    (the shape the live windowed ring readings actually have — verified median ≈ 0.53 us/leaf over ~40
    windows). A real spread, a well-defined median, non-degenerate."""
    rng = np.random.default_rng(seed)
    return [float(abs(v)) for v in (center + rng.normal(0.0, spread, n))]


def _raw(pool: list[float]) -> dict:
    """A `_measure_raw()`-shaped dict over a synthesized per-window per-leaf pool (no live ring timing)."""
    return {
        "tmsg_us_leaf_median": float(np.median(pool)),
        "per_leaf_us": pool,
        "iters": len(pool) * B._WINDOW,
    }


# --------------------------------------------------------------------------- #
# 1. The reclassification: _estimate_from_raw builds a SHRINKABLE QuantileLaw, not a Fixed pin.
# --------------------------------------------------------------------------- #
def test_futex_tmsg_estimate_is_a_shrinkable_quantile_law_not_a_fixed_pin() -> None:
    """The defect was a `Fixed` (un-shrinkable) Estimate built off the SEED; the fix is a `QuantileLaw`
    (median) with a BOOTSTRAP median SE over the bench's OWN per-leaf pool. This is the ADR-0008
    reclassification: the per-leaf ring memcpy is a MEASURED quantity, not a declared-spread pin."""
    est = B._estimate_from_raw(_raw(_framing_pool()))
    assert isinstance(est.shrink, E.QuantileLaw)      # SHRINKABLE — not E.Fixed
    assert not isinstance(est.shrink, E.Fixed)
    assert est.kind == "median"
    assert est.family == (E.CIFamily.EMPIRICAL,)      # a sample-quantile CI, not a NORMAL prior
    assert est.support == (E.Support.POSITIVE,)
    assert est.k == 1
    assert est.theta_hat[0] == pytest.approx(float(np.median(_framing_pool())))
    assert est.is_valid()


def test_futex_tmsg_cov_is_a_real_bootstrap_se_not_the_seed_sigma() -> None:
    """The variance authority is a REAL bootstrap median SE in `cov` (a positive, finite spread the loop
    can shrink), NOT the seed's frozen declared σ=0.08 wrapped un-divided (the old `Fixed` pin's
    cov=[[0.0064]])."""
    est = B._estimate_from_raw(_raw(_framing_pool()))
    boot_se = math.sqrt(float(est.cov[0, 0]))
    assert boot_se > 0.0 and math.isfinite(boot_se)
    # it is the per-window pool's order-statistic spread (small for a tight ring-copy pool), NOT σ=0.08.
    assert boot_se < 0.08


def test_futex_tmsg_marginal_is_negative_so_the_loop_can_fund_it() -> None:
    """THE PAYOFF (§1 D2 / §2.3): the typed `ShrinkLaw.marginal_dvar_deffort` is < 0 for the QuantileLaw
    (so the driver's `_fundability` `A_i = −marginal·n²` is > 0 -> the term is FUNDABLE -> the loop can
    tighten it), where the OLD `Fixed` pin's marginal is exactly 0 (`A_i = 0` -> never fundable -> the
    STALL, identical in shape to the R_gen stall on a binding term)."""
    est = B._estimate_from_raw(_raw(_framing_pool()))
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


def test_futex_tmsg_quantile_law_is_self_consistent_with_cov() -> None:
    """The shipped `QuantileLaw(p=0.5, f_at_q, n)` is SELF-CONSISTENT with the `cov` it ships beside (P1
    single-home): its implied `p(1−p)/(n·f̂²)` equals the bootstrap `cov[0,0]` exactly."""
    est = B._estimate_from_raw(_raw(_framing_pool()))
    ql = est.shrink
    assert isinstance(ql, E.QuantileLaw) and ql.p == 0.5
    implied = 0.25 / (ql.n * float(ql.f_at_q[0]) ** 2)
    assert math.isclose(implied, float(est.cov[0, 0]), rel_tol=1e-12)


def test_futex_tmsg_estimate_jsonb_round_trips() -> None:
    """The measured Estimate is an exact jsonb round-trip (the §5 SSOT serialization)."""
    est = B._estimate_from_raw(_raw(_framing_pool()))
    rt = E.from_jsonb(E.to_jsonb(est))
    assert np.allclose(rt.theta_hat, est.theta_hat)
    assert np.allclose(rt.cov, est.cov)
    assert isinstance(rt.shrink, E.QuantileLaw)
    assert rt.shrink.n == est.shrink.n
    assert rt.family == est.family and rt.kind == est.kind


# --------------------------------------------------------------------------- #
# 2. The budget (iters) is the shrink lever: more leaves -> more windows -> a tighter median SE.
# --------------------------------------------------------------------------- #
def test_more_windows_tightens_the_se() -> None:
    """The Neyman loop sizes this bench's measurement by `iters` (-> windows); a bigger pool gives a
    tighter median SE (so more/longer runs -> a tighter futex-tmsg CI — the loop's lever)."""
    se_small = math.sqrt(float(B._estimate_from_raw(_raw(_framing_pool(n=8, seed=1))).cov[0, 0]))
    se_large = math.sqrt(float(B._estimate_from_raw(_raw(_framing_pool(n=64, seed=1))).cov[0, 0]))
    assert se_large < se_small


# --------------------------------------------------------------------------- #
# 3. FAIL LOUD (ADR-0002): a degenerate zero-spread pool RAISES (a constant masquerading as a median).
# --------------------------------------------------------------------------- #
def test_degenerate_pool_raises_never_fabricates_a_quantile_law() -> None:
    """ADR-0002: a zero-spread per-leaf pool is a constant, not a measured median — `_estimate_from_raw`
    RAISES (via `median_estimate`), it does NOT fabricate a QuantileLaw the data cannot support. So a
    passing live `measure()` proves the windowed ring pool has honest spread."""
    with pytest.raises(ValueError):
        B._estimate_from_raw(_raw([0.5] * 40))   # bootstrap median SE == 0 -> raises


def test_get_seed_is_the_distrust_fallback_and_unchanged() -> None:
    """`get_seed()` stays the v1 ~0.15 us/leaf first-principles grounding (the DISTRUST fallback the
    manifest seed path uses) — the fix changes the MEASURED path, not the seed. Returns (mean, σ, unit)."""
    mean, sigma, unit = B.get_seed()
    assert unit == "us"
    assert sigma == pytest.approx(0.08)
    # the bare (req_row + rep_row) memcpy at 8 B/ns: (964 + 264)/8/1000 = 0.1535 us/leaf.
    assert mean == pytest.approx((B._REQ_ROW_B + B._REP_ROW_B) / B._MEMCPY_BW_BYTES_PER_NS / 1000.0)


# --------------------------------------------------------------------------- #
# 4. LIVE: RUN the bench's OWN in-process windowed ring measurement (no binary) — positive, shrinkable.
# --------------------------------------------------------------------------- #
def test_live_measure_runs_the_ring_and_is_shrinkable() -> None:
    """BEHAVIORAL verification (ADR-0009 — RUN it). The live `measure()` times the per-leaf ring memcpy
    in windows (in-process numpy + multiprocessing.shared_memory; NO external binary, NO geometry) and
    must (a) report a positive us/leaf and (b) produce a SHRINKABLE `QuantileLaw` Estimate whose marginal
    is < 0 (FUNDABLE — the loop can tighten it). A small `iters` keeps the test fast; the operator runs it
    pinned (taskset -c 0), sole-workload, at a larger budget for a defensible value."""
    est = B.measure(iters=20000)
    assert isinstance(est.shrink, E.QuantileLaw)
    assert est.kind == "median" and est.family == (E.CIFamily.EMPIRICAL,)
    assert float(est.theta_hat[0]) > 0.0 and math.isfinite(float(est.theta_hat[0]))
    se = math.sqrt(float(est.cov[0, 0]))
    assert se > 0.0 and math.isfinite(se)
    marg = est.shrink.marginal_dvar_deffort(float(est.cov[0, 0]), float(est.shrink.n))
    assert marg < 0.0           # SHRINKABLE -> A_i > 0 -> the loop funds it (the payoff)


def test_live_measure_raw_returns_a_windowed_pool_not_a_scalar() -> None:
    """The fix re-shaped `_measure_raw()` from a single scalar to a windowed per-leaf POOL (the median
    over N windows of `_WINDOW` leaves, the cpp_inproc_port windowing) — the structural change that makes
    `_estimate_from_raw` able to build a shrinkable median. The headline median agrees with the pool."""
    res = B._measure_raw(iters=20000)
    assert set(res) == {"tmsg_us_leaf_median", "per_leaf_us", "iters"}
    assert isinstance(res["per_leaf_us"], list)
    assert len(res["per_leaf_us"]) >= 2          # >= 2 windows so the bootstrap median SE is defined
    assert all(v > 0.0 and math.isfinite(v) for v in res["per_leaf_us"])
    assert res["tmsg_us_leaf_median"] == pytest.approx(float(np.median(res["per_leaf_us"])))
