"""
tools/analysis/leaf_eval_bound/benchmarks/pools.py
=================================================

The POOL BUILDERS (RCA fix #2) — the two shared homes of "accumulate a reading pool to a floor",
split out of `bench_common` (the responsibility-refactor note's move 1). Pure typing-only Python;
they touch no SQL and no `Estimate` — a bench injects its per-batch / per-window measurement as a
closure and gets back a floored pool that `estimators.median_estimate` consumes.

  * `collect_pool` — the RACE-collector floor: re-run a batch at growing effort until
    `len(pool) >= min_readings` (the floor binds on readings COLLECTED, not effort — the count a
    race collector cannot promise).
  * `window_pool`  — the DETERMINISTIC counterpart: run a per-window closure `max(min_windows, count)`
    times (the count IS the budget, so the floor is owned STRUCTURALLY — `len >= 2` by construction).

FAIL LOUD (ADR-0002): a race collector that under-yields below the floor RAISES; `min_* < 2` is a
contract violation and RAISES. Neither fabricates a reading.

Public Domain (The Unlicense).
"""
from __future__ import annotations

from typing import Callable, Sequence


# ============================================================================================
# ============================================================================================
# The race-collector POOL FLOOR (RCA fix #2, docs/notes/leaf-eval-estimator-pin-cascade-rca.md): a
# SHARED `len(pool) >= min_readings` guarantee for a RACE-BASED collector whose realized reading count
# is DECOUPLED from the requested effort. A producer/consumer wakeup bench (shm_spin_poll, futex_wake,
# lockfree_mpsc, cpp_inproc_port) coalesces edges it polls past and drops torn reads, so a batch returns
# FEWER readings than the effort asked — and at a small allocator budget can return < 2, which
# `median_estimate` then RAISES on (the shm_spin_poll wakeup crash: budget 6 -> 1 reading -> raise). The
# floor binds on READINGS COLLECTED, not on the requested effort: re-run the batch at growing effort
# until the accumulated pool reaches the floor. ONE home for the guarantee (ADR-0012 P1) so a new race
# bench inherits it by calling this, not by re-deriving a retry loop (ADR-0011 Rule 4: a structural net
# over the class, not a per-bench patch). It NEVER fabricates a reading (ADR-0002): an un-yielding
# collector RAISES at the attempt cap.
# ============================================================================================
def collect_pool(
    collect_batch: Callable[[int], Sequence[float]],
    *,
    name: str,
    budget: int,
    min_readings: int = 8,
    max_attempts: int = 12,
) -> list[float]:
    """Accumulate a latency/cost pool to a floor of `min_readings` readings from a RACE-BASED collector.
    `collect_batch(effort) -> Sequence[float]` runs ONE batch at the given effort and returns its
    (possibly short) pool; `collect_pool` runs it at `effort = max(min_readings, budget)` and, while the
    accumulated pool is under the floor, RE-RUNS it at doubled effort — so the floor binds on readings
    COLLECTED, not on the requested effort (the count a race collector cannot promise). Returns the
    accumulated pool (`len >= min_readings`).

    `min_readings` defaults to 8 — comfortably above `median_estimate`'s HARD minimum of 2 so the bootstrap
    median SE is non-degenerate (2 readings risk a zero-spread pool, the OTHER median_estimate raise). The
    normal path (a real allocator budget) yields hundreds in the first batch and never retries, so the
    floor binds only at the pathological tiny budget that produced the crash.

    FAIL LOUD (ADR-0002): if `max_attempts` batches (effort up to `budget·2^(max_attempts-1)`) still
    under-yield, RAISE — a collector that cannot reach `min_readings` is a real fault (a wedged producer, a
    pathological over-coalescing), never a sub-floor pool padded into a fake median."""
    if min_readings < 2:
        raise ValueError(
            f"collect_pool({name!r}): min_readings must be >= 2 (a bootstrap median SE needs >= 2 "
            f"readings); got {min_readings} (ADR-0002).")
    pool: list[float] = []
    effort = max(int(min_readings), int(budget))
    for _ in range(int(max_attempts)):
        pool.extend(float(x) for x in collect_batch(effort))
        if len(pool) >= min_readings:
            return pool
        effort *= 2
    raise ValueError(
        f"collect_pool({name!r}): only {len(pool)} reading(s) after {max_attempts} batches (final effort "
        f"{effort}) — the race collector under-yields below the floor {min_readings}; a real fault (a "
        f"wedged/over-coalescing producer), not a sub-floor pool to pad (ADR-0002).")


# ============================================================================================
# The DETERMINISTIC WINDOW-LOOP pool builder (RCA fix #2, the DRY half;
# docs/notes/leaf-eval-estimator-pin-cascade-rca.md §5.1 "factor the window-loop idiom" / §5.2c):
# the deterministic COUNTERPART to `collect_pool`. The leaf-eval median benches whose `_measure_raw`
# runs a `for _ in range(N): pool.append(measure_one_window())` loop (the tau_io family, gather,
# req_drain, zmq_baseline_wakeup, the tmsg family) hand-copied that loop ≈12 times, each differing
# only in the per-window measurement + setup — the audit's cancer D (copy-paste) / P1 (no single
# home). `window_pool` is the ONE home (ADR-0012 P1/P3: a parameterized collaborator, the per-window
# measurement injected as a closure), so a new deterministic window bench inherits the loop by
# calling this, never re-deriving it.
#
# WHY A SEPARATE HELPER FROM `collect_pool` (the deterministic↔race asymmetry). A window loop has a
# KNOWN reading count — exactly the budget, ONE reading per window iteration — because the loop body
# is timed deterministically (no edge-coalescing, no torn-read drops), unlike a race collector whose
# realized count is decoupled from the effort. So `collect_pool`'s RETRY-until-floor machinery
# (re-run the batch at growing effort) is the wrong shape here: there is nothing to retry, the count
# is the loop bound. `window_pool` instead OWNS THE FLOOR STRUCTURALLY — it runs the loop
# `max(min_windows, count)` times, so `len(pool) >= min_windows >= 2` BY CONSTRUCTION. This makes the
# deterministic benches EXPLICITLY safe (a 1-window pool RAISES in `median_estimate`, ADR-0002):
# before this, each bench leaned on the driver's `max(2, …)` budget floor (untrusted_drive
# `_make_measurer`) plus its own ad-hoc `n_windows = max(2, …)` — a per-bench guard the audit names
# as the right instinct applied per-bench (RCA §5.2c); making the floor a PROPERTY OF THE CONTRACT
# closes the gap symmetrically with `collect_pool`.
#
# BEHAVIORAL EQUIVALENCE (ADR-0009). At any `count >= min_windows` (the normal operating regime — a
# real allocator budget is hundreds), `max(min_windows, count) == count`, so the loop runs EXACTLY
# `count` times: the migration is a pure refactor (same closure body, same readings, same pool), the
# `min_windows` default of 2 reproducing the benches' existing `max(2, …)` floor byte-for-byte. The
# ONLY intended change is at a tiny `count < 2` (the floor lifts it to 2) — the same safety
# improvement `collect_pool` made for the race family. `window_pool` owns ONLY the count/floor
# guarantee (the finiteness / zero-spread checks stay single-homed in `median_estimate`, its sole
# gate — this helper does not duplicate them).
# ============================================================================================
def window_pool(
    measure_window: Callable[[], float],
    *,
    name: str,
    count: int,
    min_windows: int = 2,
) -> list[float]:
    """Build a deterministic latency/cost pool by calling `measure_window()` once per window — the
    shared home of the `for _ in range(N): pool.append(one_window())` idiom. `measure_window() ->
    float` times ONE window (the per-window measurement the bench injects; its setup/warmup/teardown
    stay in the bench, around the call) and returns that window's reading; `window_pool` runs it
    `n = max(min_windows, count)` times and returns the `n` readings (so `len(pool) >= min_windows`,
    the >= 2 the bootstrap median SE needs — the floor is structural, not per-bench).

    `min_windows` defaults to 2 — `median_estimate`'s HARD minimum (a 1-reading pool has no bootstrap
    spread and RAISES, ADR-0002), and exactly the floor the deterministic benches carried inline as
    `max(2, …)`. The normal path (a real allocator budget) passes `count` in the hundreds, so the
    floor never binds and the loop runs `count` times unchanged; the floor binds only at the
    pathological tiny budget, lifting it to 2.

    FAIL LOUD (ADR-0002): `min_windows < 2` is itself a contract violation (a bootstrap median SE
    needs >= 2 readings) and RAISES — symmetric with `collect_pool`. It NEVER fabricates a reading;
    each window's value is whatever `measure_window()` returns (the finiteness / zero-spread gate is
    `median_estimate`'s, the single home for that check)."""
    if min_windows < 2:
        raise ValueError(
            f"window_pool({name!r}): min_windows must be >= 2 (a bootstrap median SE needs >= 2 "
            f"readings); got {min_windows} (ADR-0002).")
    n = max(int(min_windows), int(count))
    return [float(measure_window()) for _ in range(n)]
