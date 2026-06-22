"""
tests/test_bench_mpsc_shm_tmsg_reclassification.py
==================================================

The ADR-0008 RECLASSIFICATION of the two remaining punt'd per-leaf MESSAGE-cost benches —
`lockfree_mpsc_tmsg_us_leaf` (a tail-CAS enqueue + slot write/read; registered quantity
`transport_msg_cost_per_leaf_lockfree_mpsc`) and `shm_spin_poll_tmsg_us_leaf` (the bare in-ring
memcpy of one request row in + one reply row out; `transport_msg_cost_per_leaf_shm_spin_poll`):
both were MEASURED latencies MIS-WIRED as un-shrinkable `Fixed` declared-spread pins, now wired to
their OWN live windowed measurement so each returns a SHRINKABLE Estimate the Neyman loop can sample
(docs/design/harmonized-estimator-interface.md §3 MEDIAN row + §7.A).

The defect (verified live before the fix — both reproduced this session): `_measure_raw()` PERFORMED
a real measurement (lockfree_mpsc ~0.8–1.5 us/leaf; shm_spin_poll ~0.5–0.9 us/leaf), but
`_estimate_from_raw()` DISCARDED it and returned `pin_estimate(get_seed())` — a `Fixed`/`marginal=0`/
un-fundable Estimate built off the v1 SEED (0.1835 / 0.1535 us/leaf, a >4x measured-vs-seed gap), so
the manifest TRUST path held a re-declared seed the bench's own measurement contradicts, and the
Neyman loop could never tighten it (`A_i = 0` -> the stall `bench_r_gen.py` removed). This is the SAME
measure-then-pin punt the IN-COMMIT sibling `bench_futex_wake_tmsg_us_leaf` (identical bare-ring
physics to shm_spin_poll) and `bench_cpp_inproc_port_tmsg_us_leaf` already had FIXED; the runnable
per-leaf pool was in hand (each bench's own windowed loop, the cpp_inproc_port/futex windowing), so
the correct class is `median`.

CLASSIFICATION (ADR-0008 — refuse the closest-fit pin, the sibling shows the honest median; ADR-0012
P8 the typed family/shrink IS the contract; P1 single-home — the measured per_leaf_us and the
Estimate's theta_hat now share ONE home): a MEASURED, shrinkable median, NOT a declared-spread pin.
The CONSEQUENCE is benign (tmsg enters the model only as the non-binding min() arm `1/(L*tmsg*1e-6)`
~2000+ dps while SERVE binds ~430 dps -> df/dtmsg=0, a_i ~ 0 — verified the transport stays
non-binding even at the measured tmsg), so this is a CLASSIFICATION/honesty fix, not a wrong-number-
on-the-bound fix; the fix makes the term fundable-when-asked as the DESIGN-PRIORITY transport DOF the
sweep's `_TRANSPORT_MOVED_TERMS` names, while the variance ranking still (correctly) buries it.

THE FIX, AS TESTS (parametrized over both modules):
  * `_estimate_from_raw` over a per-window per-leaf pool builds a SHRINKABLE `QuantileLaw` (median)
    Estimate — a REAL bootstrap median SE, `family=EMPIRICAL`, `kind='median'` — NOT a `Fixed` pin;
    and its typed `ShrinkLaw.marginal_dvar_deffort` is < 0 (so the driver's `A_i = −marginal·n²` is
    > 0 -> FUNDABLE), where the old `Fixed.marginal` is 0 (un-fundable -> the stall).
  * the budget (`iters`) is the shrink lever: more leaves -> more windows -> a tighter median SE.
  * `pin_estimate` is GONE from the import (the punt's tell); `get_seed()` stays the v1 DISTRUST
    fallback, UNCHANGED — the fix changes the MEASURED path, not the seed.

Most assertions run WITHOUT timing the live loop by exercising `_estimate_from_raw` on a synthesized
per-leaf pool (the §8 discipline — the Estimate SHAPE + bootstrap SE + marginal are pool-driven). A
LIVE tail RUNS each bench's own in-process windowed measurement (NO external binary, NO geometry to
gate on — numpy + multiprocessing.shared_memory): it must report a positive us/leaf and a shrinkable
Estimate (ADR-0009 — run it; pin `taskset -c 0` is the operator's job, this is a behavioral gate).

The estimate/bench modules live under tools/analysis/leaf_eval_bound/ (no __init__.py — imported by
sys.path), so this test prepends those directories.

Public Domain (The Unlicense).
"""
from __future__ import annotations

import importlib
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

import estimate as E  # noqa: E402  — the contract

# The two reclassified modules + the seed (mean, sigma) each get_seed() declares, for the
# DISTRUST-fallback-unchanged check. Parametrized so each bench is exercised independently.
_MODULES = ("bench_lockfree_mpsc_tmsg", "bench_shm_spin_poll_tmsg")


def _mod(name: str):
    """Import the bench module under test (the OpenTURNS dirs are on sys.path above)."""
    return importlib.import_module(name)


def _framing_pool(center: float = 0.6, spread: float = 0.03, n: int = 40, seed: int = 3) -> list[float]:
    """A plausible per-window per-leaf pool: a tight positive cluster around `center` us/leaf (the shape
    the live windowed readings actually have — verified ~0.5–0.8 us/leaf over ~20 windows). A real
    spread, a well-defined median, non-degenerate."""
    rng = np.random.default_rng(seed)
    return [float(abs(v)) for v in (center + rng.normal(0.0, spread, n))]


def _raw(mod, pool: list[float]) -> dict:
    """A `_measure_raw()`-shaped dict over a synthesized per-window per-leaf pool (no live timing)."""
    return {
        "tmsg_us_leaf_median": float(np.median(pool)),
        "per_leaf_us": pool,
        "iters": len(pool) * mod._WINDOW,
    }


# --------------------------------------------------------------------------- #
# 1. The reclassification: _estimate_from_raw builds a SHRINKABLE QuantileLaw, not a Fixed pin.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("modname", _MODULES)
def test_tmsg_estimate_is_a_shrinkable_quantile_law_not_a_fixed_pin(modname: str) -> None:
    """The defect was a `Fixed` (un-shrinkable) Estimate built off the SEED; the fix is a `QuantileLaw`
    (median) with a BOOTSTRAP median SE over the bench's OWN per-leaf pool. The ADR-0008 reclassification:
    a MEASURED latency, not a declared-spread pin."""
    B = _mod(modname)
    est = B._estimate_from_raw(_raw(B, _framing_pool()))
    assert isinstance(est.shrink, E.QuantileLaw)      # SHRINKABLE — not E.Fixed
    assert not isinstance(est.shrink, E.Fixed)
    assert est.kind == "median"
    assert est.family == (E.CIFamily.EMPIRICAL,)      # a sample-quantile CI, not a NORMAL prior
    assert est.support == (E.Support.POSITIVE,)
    assert est.k == 1
    assert est.theta_hat[0] == pytest.approx(float(np.median(_framing_pool())))
    assert est.is_valid()


@pytest.mark.parametrize("modname", _MODULES)
def test_tmsg_cov_is_a_real_bootstrap_se_not_the_seed_sigma(modname: str) -> None:
    """The variance authority is a REAL bootstrap median SE in `cov` (a positive, finite spread the loop
    can shrink), NOT the seed's frozen declared σ=0.08 wrapped un-divided (the old `Fixed` pin's
    cov=[[0.0064]])."""
    B = _mod(modname)
    est = B._estimate_from_raw(_raw(B, _framing_pool()))
    boot_se = math.sqrt(float(est.cov[0, 0]))
    assert boot_se > 0.0 and math.isfinite(boot_se)
    # it is the per-window pool's order-statistic spread (small for a tight pool), NOT the seed σ=0.08.
    assert boot_se < 0.08


@pytest.mark.parametrize("modname", _MODULES)
def test_tmsg_marginal_is_negative_so_the_loop_can_fund_it(modname: str) -> None:
    """THE PAYOFF (§1 D2 / §2.3): the typed `ShrinkLaw.marginal_dvar_deffort` is < 0 for the QuantileLaw
    (so the driver's `_fundability` `A_i = −marginal·n²` is > 0 -> the term is FUNDABLE -> the loop can
    tighten it), where the OLD `Fixed` pin's marginal is exactly 0 (`A_i = 0` -> never fundable -> the
    STALL, identical in shape to the R_gen stall on a binding term — ADR-0012's substitution test)."""
    B = _mod(modname)
    est = B._estimate_from_raw(_raw(B, _framing_pool()))
    sigma_ii = float(est.cov[0, 0])
    n_eff = float(est.shrink.n)
    marg_new = est.shrink.marginal_dvar_deffort(sigma_ii, n_eff)
    marg_old = E.Fixed().marginal_dvar_deffort(sigma_ii, n_eff)
    assert marg_new < 0.0          # shrinkable: one more reading lowers the variance
    assert marg_old == 0.0         # the punt: irreducible, the un-fundable stall
    A_new = -marg_new * n_eff ** 2
    A_old = -marg_old * n_eff ** 2
    assert A_new > 0.0             # FUNDED
    assert A_old == 0.0            # un-fundable (the stall the fix removes)


@pytest.mark.parametrize("modname", _MODULES)
def test_tmsg_quantile_law_is_self_consistent_with_cov(modname: str) -> None:
    """The shipped `QuantileLaw(p=0.5, f_at_q, n)` is SELF-CONSISTENT with the `cov` it ships beside (P1
    single-home): its implied `p(1−p)/(n·f̂²)` equals the bootstrap `cov[0,0]` exactly."""
    B = _mod(modname)
    est = B._estimate_from_raw(_raw(B, _framing_pool()))
    ql = est.shrink
    assert isinstance(ql, E.QuantileLaw) and ql.p == 0.5
    implied = 0.25 / (ql.n * float(ql.f_at_q[0]) ** 2)
    assert math.isclose(implied, float(est.cov[0, 0]), rel_tol=1e-12)


@pytest.mark.parametrize("modname", _MODULES)
def test_tmsg_estimate_jsonb_round_trips(modname: str) -> None:
    """The measured Estimate is an exact jsonb round-trip (the §5 SSOT serialization)."""
    B = _mod(modname)
    est = B._estimate_from_raw(_raw(B, _framing_pool()))
    rt = E.from_jsonb(E.to_jsonb(est))
    assert np.allclose(rt.theta_hat, est.theta_hat)
    assert np.allclose(rt.cov, est.cov)
    assert isinstance(rt.shrink, E.QuantileLaw)
    assert rt.shrink.n == est.shrink.n
    assert rt.family == est.family and rt.kind == est.kind


# --------------------------------------------------------------------------- #
# 2. The budget (iters) is the shrink lever: more leaves -> more windows -> a tighter median SE.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("modname", _MODULES)
def test_more_windows_tightens_the_se(modname: str) -> None:
    """The Neyman loop sizes this bench's measurement by `iters` (-> windows); a bigger pool gives a
    tighter median SE (so more/longer runs -> a tighter tmsg CI — the loop's lever)."""
    B = _mod(modname)
    se_small = math.sqrt(float(B._estimate_from_raw(_raw(B, _framing_pool(n=8, seed=1))).cov[0, 0]))
    se_large = math.sqrt(float(B._estimate_from_raw(_raw(B, _framing_pool(n=64, seed=1))).cov[0, 0]))
    assert se_large < se_small


# --------------------------------------------------------------------------- #
# 3. FAIL LOUD (ADR-0002): a degenerate zero-spread pool RAISES; the punt's import is GONE.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("modname", _MODULES)
def test_degenerate_pool_raises_never_fabricates_a_quantile_law(modname: str) -> None:
    """ADR-0002: a zero-spread per-leaf pool is a constant, not a measured median — `_estimate_from_raw`
    RAISES (via `median_estimate`), it does NOT fabricate a QuantileLaw the data cannot support. So a
    passing live `measure()` proves the windowed pool has honest spread."""
    B = _mod(modname)
    with pytest.raises(ValueError):
        B._estimate_from_raw(_raw(B, [0.5] * 40))   # bootstrap median SE == 0 -> raises


@pytest.mark.parametrize("modname", _MODULES)
def test_get_seed_is_the_distrust_fallback_and_unchanged(modname: str) -> None:
    """`get_seed()` stays the v1 first-principles grounding (the DISTRUST fallback the manifest seed path
    uses) — the fix changes the MEASURED path, not the seed. Returns (mean, σ, unit)."""
    B = _mod(modname)
    mean, sigma, unit = B.get_seed()
    assert unit == "us"
    assert sigma == pytest.approx(0.08)
    # the first-principles bare-row memcpy mean: (964 + 264)/8/1000 (+ CAS for mpsc) — the bench's own constants.
    base = (B._REQ_ROW_B + B._REP_ROW_B) / B._MEMCPY_BW_BYTES_PER_NS / 1000.0
    extra = getattr(B, "_CAS_NS", 0.0) / 1000.0     # the MPSC enqueue CAS; 0 for the bare shm ring
    assert mean == pytest.approx(base + extra)


@pytest.mark.parametrize("modname", _MODULES)
def test_pin_estimate_no_longer_imported(modname: str) -> None:
    """The punt's structural tell was `from bench_common import ... pin_estimate`; the fix drops it (the
    measured path is a `median_estimate`). A regression that re-introduces the pin would re-bind the name."""
    B = _mod(modname)
    assert not hasattr(B, "pin_estimate")
    assert hasattr(B, "median_estimate")


# --------------------------------------------------------------------------- #
# 4. LIVE: RUN each bench's OWN in-process windowed measurement (no binary) — positive, shrinkable.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("modname", _MODULES)
def test_live_measure_runs_and_is_shrinkable(modname: str) -> None:
    """BEHAVIORAL verification (ADR-0009 — RUN it). The live `measure()` times the per-leaf traffic in
    windows (in-process numpy + multiprocessing.shared_memory; NO external binary, NO geometry) and must
    (a) report a positive us/leaf and (b) produce a SHRINKABLE `QuantileLaw` Estimate whose marginal is
    < 0 (FUNDABLE — the loop can tighten it). A small `iters` keeps the test fast; the operator runs it
    pinned (taskset -c 0), sole-workload, at a larger budget for a defensible value."""
    B = _mod(modname)
    est = B.measure(iters=20000)
    assert isinstance(est.shrink, E.QuantileLaw)
    assert est.kind == "median" and est.family == (E.CIFamily.EMPIRICAL,)
    assert float(est.theta_hat[0]) > 0.0 and math.isfinite(float(est.theta_hat[0]))
    se = math.sqrt(float(est.cov[0, 0]))
    assert se > 0.0 and math.isfinite(se)
    marg = est.shrink.marginal_dvar_deffort(float(est.cov[0, 0]), float(est.shrink.n))
    assert marg < 0.0           # SHRINKABLE -> A_i > 0 -> the loop funds it (the payoff)


@pytest.mark.parametrize("modname", _MODULES)
def test_live_measure_raw_returns_a_windowed_pool_not_a_scalar(modname: str) -> None:
    """The fix re-shaped `_measure_raw()` from a single scalar (`{'tmsg_us_leaf', 'iters'}`) to a windowed
    per-leaf POOL (the median over N windows of `_WINDOW` leaves, the cpp_inproc_port/futex windowing) —
    the structural change that makes `_estimate_from_raw` able to build a shrinkable median. The headline
    median agrees with the pool."""
    B = _mod(modname)
    res = B._measure_raw(iters=20000)
    assert set(res) == {"tmsg_us_leaf_median", "per_leaf_us", "iters"}
    assert isinstance(res["per_leaf_us"], list)
    assert len(res["per_leaf_us"]) >= 2          # >= 2 windows so the bootstrap median SE is defined
    assert all(v > 0.0 and math.isfinite(v) for v in res["per_leaf_us"])
    assert res["tmsg_us_leaf_median"] == pytest.approx(float(np.median(res["per_leaf_us"])))
