"""
tests/test_bench_common_collect_pool.py
=======================================

`bench_common.collect_pool` — the shared race-collector POOL FLOOR (RCA fix #2,
`docs/notes/leaf-eval-estimator-pin-cascade-rca.md`): a `len(pool) >= min_readings`
guarantee for a producer/consumer wakeup bench whose realized reading count is
DECOUPLED from the requested effort (it coalesces edges + drops torn reads, so a small
allocator budget yields < 2 readings -> `median_estimate` RAISES -> the
`~/shm_spin_poll_fail` crash). `collect_pool` floors on readings COLLECTED (re-run the
batch at growing effort), and NEVER fabricates a reading — an un-yielding collector
RAISES at the cap (ADR-0002).

These tests drive `collect_pool` with FAKE collectors (no live timed bench), so they
are deterministic and fast — the §8 / Phase-3 discipline (exercise the harness logic,
not the timing-sensitive measurement).

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


def test_first_batch_meets_floor_no_retry() -> None:
    """The NORMAL path: a collector yielding >= min_readings in ONE batch returns at once (a real allocator
    budget yields hundreds; the floor never binds). The first effort is `max(min_readings, budget)` — the
    budget is floored UP to the min, never below it."""
    efforts = []

    def plenty(effort: int) -> list[float]:
        efforts.append(effort)
        return [1.0] * effort

    pool = BC.collect_pool(plenty, name="plenty", budget=3)
    assert len(pool) >= 8                 # the default floor
    assert efforts == [8]                 # one batch, at effort=max(8,3)=8 (budget floored up to the min)


def test_accumulates_until_floor_on_underyield() -> None:
    """The RACE path: a collector that under-yields (fewer readings than the effort asked) is RE-RUN at
    DOUBLED effort until the accumulated pool reaches the floor — the floor binds on readings COLLECTED, not
    on the requested effort (the shm_spin_poll wakeup shape: the count is not promised by `trials`)."""
    efforts = []

    def drip(effort: int) -> list[float]:
        efforts.append(effort)
        return [1.0]                      # exactly 1 reading per batch, regardless of effort

    pool = BC.collect_pool(drip, name="drip", budget=2, min_readings=5)
    assert len(pool) == 5                 # accumulated to exactly the floor
    assert efforts == [5, 10, 20, 40, 80]  # 5 batches; effort doubles each retry (start max(5,2)=5)


def test_fail_loud_on_unyielding_collector() -> None:
    """ADR-0002: a collector that NEVER reaches the floor RAISES at `max_attempts` — a wedged / over-
    coalescing producer is a real fault, never a sub-floor pool padded into a fake median."""
    with pytest.raises(ValueError, match="under-yields below the floor"):
        BC.collect_pool(lambda effort: [], name="wedged", budget=2, max_attempts=4)


def test_rejects_floor_below_two() -> None:
    """`min_readings < 2` is itself a contract violation (a bootstrap median SE needs >= 2 readings) — fail
    loud, not a silent 1-sample 'median'."""
    with pytest.raises(ValueError, match="min_readings must be >= 2"):
        BC.collect_pool(lambda effort: [1.0, 2.0], name="bad", budget=2, min_readings=1)
