"""
tests/test_bench_common_window_pool.py
======================================

`bench_common.window_pool` — the shared DETERMINISTIC WINDOW-LOOP pool builder (RCA fix #2,
the DRY half; `docs/notes/leaf-eval-estimator-pin-cascade-rca.md` §5.1/§5.2c): the ONE home of
the `for _ in range(N): pool.append(measure_one_window())` idiom the leaf-eval median benches
(the tau_io family, gather, req_drain, zmq_baseline_wakeup, the tmsg family) hand-copied ≈12
times. It is the deterministic COUNTERPART to `collect_pool`: a window loop's reading count is
KNOWN (= the budget, one reading per window), so there is nothing to retry — instead the helper
owns the `>= 2` floor STRUCTURALLY (`len(pool) >= min_windows` by construction), making every
deterministic bench explicitly safe at a tiny budget (a 1-reading pool RAISES in
`median_estimate`, ADR-0002), symmetric with `collect_pool`'s floor for the race family.

These tests drive `window_pool` with FAKE `measure_window` closures (no live timed bench), so
they are deterministic and fast — the §8 / Phase-3 discipline (exercise the harness logic, not
the timing-sensitive measurement). At count >= 2 the helper is a pure refactor of the inline
loop (same readings, same count); the only intended change is the >= 2 floor at a tiny count.

Public Domain (The Unlicense).
"""
from __future__ import annotations

import os
import sys

import pytest

_OT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "tools", "analysis", "leaf_eval_bound",
)
_BENCH = os.path.join(_OT, "benchmarks")
for _p in (_OT, _BENCH):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import bench_common as BC  # noqa: E402


def test_runs_one_call_per_window_in_order() -> None:
    """The NORMAL path (count >= the floor): `window_pool` calls `measure_window()` EXACTLY `count`
    times — one reading per window — and returns them in order. This is the pure-refactor regime (a
    real allocator budget is hundreds): the floor never binds, so the loop runs `count` times, the
    behavioral-equivalence guarantee (ADR-0009) the migration rests on."""
    calls = {"n": 0}

    def measure_window() -> float:
        calls["n"] += 1
        return float(calls["n"])           # 1.0, 2.0, 3.0, ... — proves order + one call per window

    pool = BC.window_pool(measure_window, name="seq", count=5)
    assert calls["n"] == 5                 # exactly `count` calls (no retry, no extra)
    assert pool == [1.0, 2.0, 3.0, 4.0, 5.0]


def test_floor_lifts_a_tiny_count_to_min_windows() -> None:
    """The FLOOR path: a `count` below `min_windows` is lifted to `min_windows` — the ONLY intended
    change vs the inline loop (the >= 2 safety improvement, like collect_pool's floor). A budget of 1
    yields the 2-reading minimum the bootstrap median SE needs (a 1-reading pool RAISES in
    median_estimate, ADR-0002), so a binding-arm bench cannot underflow at a tiny allocator budget."""
    calls = {"n": 0}

    def measure_window() -> float:
        calls["n"] += 1
        return 7.0

    pool = BC.window_pool(measure_window, name="tiny", count=1)
    assert len(pool) == 2                   # floored UP to the default min_windows=2
    assert calls["n"] == 2
    assert pool == [7.0, 7.0]


def test_count_at_or_above_floor_is_unchanged() -> None:
    """At `count == min_windows` (the boundary) the floor is a no-op: `max(2, 2) == 2`, so the loop
    runs `count` times unchanged. This pins that the floor reproduces the benches' existing
    `max(2, …)` byte-for-byte — it never INFLATES a count already at/above the floor."""
    for count in (2, 3, 64):
        calls = {"n": 0}

        def tick() -> float:
            calls["n"] += 1
            return 1.0

        pool = BC.window_pool(tick, name="boundary", count=count)
        assert calls["n"] == count          # ran exactly `count` times (floor a no-op at/above 2)
        assert len(pool) == count


def test_custom_min_windows_floor() -> None:
    """`min_windows` is tunable above 2 (a bench wanting a wider structural floor). A `count` below it
    is lifted to it; `len(pool) == max(min_windows, count)`."""
    pool = BC.window_pool(lambda: 3.0, name="wide", count=3, min_windows=8)
    assert len(pool) == 8                   # count=3 floored up to min_windows=8


def test_rejects_floor_below_two() -> None:
    """`min_windows < 2` is itself a contract violation (a bootstrap median SE needs >= 2 readings) —
    fail loud, symmetric with collect_pool, never a silent 1-sample 'median'."""
    with pytest.raises(ValueError, match="min_windows must be >= 2"):
        BC.window_pool(lambda: 1.0, name="bad", count=10, min_windows=1)


def test_readings_are_coerced_to_float() -> None:
    """Each window's value is coerced to `float` (the pool is a `list[float]` the estimator consumes),
    matching the inline benches' `dt / 1000.0` float readings. `window_pool` does NOT gate finiteness
    or spread — that stays single-homed in `median_estimate` (this helper owns only count/floor)."""
    pool = BC.window_pool(lambda: 5, name="intval", count=3)   # an int reading -> coerced
    assert pool == [5.0, 5.0, 5.0]
    assert all(isinstance(v, float) for v in pool)
