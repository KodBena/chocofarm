"""
tools/analysis/leaf_eval_bound/benchmarks/bench_cpp_inproc_port_tmsg_us_leaf.py
=========================================================================

LIVE benchmark for `cpp_inproc_port_tmsg_us_leaf` — the per-leaf-amortized message-passing cost
(us/leaf) for the C++ in-process queue-port transport: the per-leaf handoff a producer pays to
submit one leaf-eval into the in-process queue. With NO wire there is no frame, no codec, no
corr-id, no syscall — a producer enqueues one leaf by (a) writing its feature row into its stripe
of the staging arena and (b) pushing a slot index onto a single-producer/single-consumer (or
lock-free MPSC) ready-queue (one relaxed atomic store + a fence). So this is the CHEAPEST tmsg of
any transport variant — the inproc-port endpoint of the message-cost axis.

NON-BINDING (the brief). The transport request/reply CAPACITY (`1/(LPD*tmsg*1e-6)`) is provably far
above the binding serve cycle for every variant; the inproc port makes it the LEAST binding of all.
This term is REPORTED for completeness and ranks LAST for the Neyman allocator. It is the
transport-capacity arm of the model's min(), kept so the bound is honest that transport never binds.

WHAT run() MEASURES (1:1 with the model input, NO JAX, NO host->device). A sole-workload microbench
of the per-leaf enqueue ALONE: write one feature row into an arena stripe + push a slot index onto a
ready-queue (a numpy int ring + an index advance, standing for the relaxed-atomic SPSC/MPSC push).
The per-leaf tmsg is that time. The SEED is a first-principles estimate (a row write + an atomic
push, ~0.05 us/leaf — well below the ZMQ baseline's 1.0 us/leaf and the MPSC's ~0.18 us/leaf,
because there is no CAS-on-tail contention for an arena-stripe SPSC push).

TIMING-SENSITIVE — DO NOT run() during the parallel fan-out. Pin: `taskset -c 0`.

Public Domain (The Unlicense).
"""
from __future__ import annotations

import time
from typing import Any


from leaf_eval_bound.contract import estimate as _est  # noqa: E402  — the harmonized Estimate contract (measure() returns one — §6 Phase 4)
from leaf_eval_bound.benchmarks.estimators import median_estimate  # noqa: E402
from leaf_eval_bound.benchmarks.scaffold import bench as _scaffold  # noqa: E402  — move 6 wiring

NAME = "cpp_inproc_port_tmsg_us_leaf"
MODULE_PATH = "leaf_eval_bound.benchmarks.bench_cpp_inproc_port_tmsg_us_leaf"
_DESC = ("Per-leaf-amortized message cost (us/leaf) for the C++ inproc-port: a producer writes one feature row "
         "into its arena stripe + pushes a slot index onto a ready-queue (one relaxed-atomic SPSC/MPSC push) — "
         "no frame, no codec, no corr-id, no syscall. The cheapest tmsg of any variant; NON-BINDING, ranks "
         "LAST for the allocator (the transport-capacity arm of the bound's min()).")

_IN_DIM = 241
_SEED_US_LEAF = 0.05    # first-principles: a 964 B row write (~0.12us at 8 B/ns) amortizes with the atomic push;
                        # the headline is the per-leaf enqueue HANDOFF (the row write overlaps the producer's own
                        # feature compute), well below ZMQ's 1.0 and MPSC's ~0.18 (no CAS-tail contention, no frame).
_SEED_SIGMA = 0.04      # wide relative spread (a contended MPSC push vs an uncontended SPSC push)


def get_seed() -> tuple[float, float, str]:
    """The v1 seed (DISTRUST fallback): the per-leaf inproc enqueue handoff, ~0.05 us/leaf (a relaxed-atomic
    slot-index push; the arena row write overlaps the producer's feature compute). NON-BINDING. Returns
    (mean, sigma, unit)."""
    return (_SEED_US_LEAF, _SEED_SIGMA, "us/leaf")


def _measure_raw(leaves: int = 200000) -> dict[str, Any]:
    """The raw-pool PROVENANCE producer (the §6 Phase-4 internal helper): measure the per-leaf inproc enqueue: over `leaves` iterations, write one feature row into an arena
    stripe + push a slot index onto a ready ring (an index advance standing for the relaxed-atomic push).
    Returns {'tmsg_us_leaf_median', 'per_leaf_us' (a sampled subset), 'leaves'}. Imports numpy lazily. Pin
    the process (taskset -c 0). `measure()` wraps the per-cycle pool into a median `Estimate`; `run()` uses it for BOTH the Estimate and the raw provenance rows (ONE measurement, two consumers — P1)."""
    import numpy as np

    arena = np.zeros((1024, _IN_DIM), dtype=np.float32)
    ready = np.zeros(1024, dtype=np.int64)
    row = np.ones((_IN_DIM,), dtype=np.float32)

    for i in range(min(2000, leaves)):                 # warm
        arena[i % 1024] = row
        ready[i % 1024] = i

    # Time per-leaf in small windows (a single-leaf perf_counter call would be clock-dominated); the headline
    # is the median per-leaf over the windows.
    window = 1000
    n_windows = max(1, leaves // window)
    per_leaf_us: list[float] = []
    for w in range(n_windows):
        t0 = time.perf_counter_ns()
        for j in range(window):
            slot = (w * window + j) % 1024
            arena[slot] = row                           # the arena-stripe row write
            ready[slot] = w * window + j                # the ready-queue slot-index push (atomic stand-in)
        dt = time.perf_counter_ns() - t0
        per_leaf_us.append(dt / 1000.0 / window)

    return {"tmsg_us_leaf_median": float(np.median(per_leaf_us)), "per_leaf_us": per_leaf_us, "leaves": leaves}


def _estimate_from_raw(res: dict[str, Any]) -> "_est.Estimate":
    """Build this bench's harmonized `Estimate` from a `_measure_raw()` dict — the SINGLE home of the
    Estimate construction (P1), called by BOTH `measure()` and `run()`. A k=1 median `QuantileLaw(p=0.5)`
    with a BOOTSTRAP median SE over the pool (§7.A — the order-statistic variance, NOT s²/n),
    family=EMPIRICAL, kind='median'."""
    return median_estimate(res["per_leaf_us"], name=NAME)   # bootstrap median SE over the per-leaf pool


# Move 6: the shared scaffold wires register_self / measure / run from the bench-specific parts above.
# TUPLE seed (no .unit) — the explicit registered unit is passed via units="us/leaf".
_B = _scaffold(
    name=NAME, quantity="transport_msg_cost_per_leaf_cpp_inproc_port", module_path=MODULE_PATH, description=_DESC,
    units="us/leaf",
    seed=get_seed, measure_raw=_measure_raw, estimate_from_raw=_estimate_from_raw,
    run_config=lambda res, **kw: {"leaves": res["leaves"], "transport": "cpp_inproc_port_direct_call",
           "tmsg_us_leaf_median": res["tmsg_us_leaf_median"],
           "note": "per-leaf enqueue handoff (arena row write + ready-queue slot-index push); NON-BINDING"},
    run_log=lambda res, log, **kw: log(res["per_leaf_us"], sample_size=1),
)
register_self, measure, run = _B.register_self, _B.measure, _B.run


if __name__ == "__main__":
    _m, _s, _u = get_seed()
    print(f"[bench_cpp_inproc_port_tmsg_us_leaf] seed: {_m} {_u} (sigma {_s}) — FIRST-PRINCIPLES "
          f"(arena row write + relaxed-atomic slot-index push; the cheapest tmsg of any variant; NON-BINDING)")
    register_self()
    print("[bench_cpp_inproc_port_tmsg_us_leaf] registered. NOT running the live measurement (timing-sensitive); "
          "invoke run() pinned. NON-BINDING — ranks LAST for the Neyman allocator.")
