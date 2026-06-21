"""
tools/analysis/OpenTURNS/benchmarks/bench_cpp_inproc_port_tmsg_us_leaf.py
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

import os
import sys
import time
from typing import Any

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.dirname(_HERE), _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from bench_common import logged_run, median_estimate  # noqa: E402

NAME = "cpp_inproc_port_tmsg_us_leaf"
MODULE_PATH = "benchmarks.bench_cpp_inproc_port_tmsg_us_leaf"
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


def register_self() -> Any:
    from bench_common import register_quantity
    return register_quantity(NAME, quantity="transport_msg_cost_per_leaf_cpp_inproc_port", units="us/leaf",
                             description=_DESC, module_path=MODULE_PATH)


def measure(leaves: int = 200000) -> dict[str, Any]:
    """Measure the per-leaf inproc enqueue: over `leaves` iterations, write one feature row into an arena
    stripe + push a slot index onto a ready ring (an index advance standing for the relaxed-atomic push).
    Returns {'tmsg_us_leaf_median', 'per_leaf_us' (a sampled subset), 'leaves'}. Imports numpy lazily. Pin
    the process (taskset -c 0)."""
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


def run(leaves: int = 200000) -> dict[str, Any]:
    """Measure the per-leaf inproc enqueue and LOG it as a harmonized k=1 median Estimate (QuantileLaw p=0.5,
    bootstrap median SE, §6 Phase 3, §5.2 de-dup). TIMING-SENSITIVE — operator-invoked, pinned (taskset -c 0),
    NEVER during the fan-out."""
    res = measure(leaves=leaves)
    est = median_estimate(res["per_leaf_us"], name=NAME)
    cfg = {"leaves": res["leaves"], "transport": "cpp_inproc_port_direct_call",
           "tmsg_us_leaf_median": res["tmsg_us_leaf_median"],
           "note": "per-leaf enqueue handoff (arena row write + ready-queue slot-index push); NON-BINDING"}
    with logged_run(NAME, quantity="transport_msg_cost_per_leaf_cpp_inproc_port", units="us/leaf",
                    description=_DESC, module_path=MODULE_PATH, config=cfg, estimate=est) as log:
        # PROVENANCE only (§5.2 de-dup): the headline median lives in estimate.theta_hat[0], not a sample row.
        log(res["per_leaf_us"], sample_size=1)
    return res


if __name__ == "__main__":
    _m, _s, _u = get_seed()
    print(f"[bench_cpp_inproc_port_tmsg_us_leaf] seed: {_m} {_u} (sigma {_s}) — FIRST-PRINCIPLES "
          f"(arena row write + relaxed-atomic slot-index push; the cheapest tmsg of any variant; NON-BINDING)")
    register_self()
    print("[bench_cpp_inproc_port_tmsg_us_leaf] registered. NOT running the live measurement (timing-sensitive); "
          "invoke run() pinned. NON-BINDING — ranks LAST for the Neyman allocator.")
