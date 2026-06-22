"""
tools/analysis/leaf_eval_bound/benchmarks/bench_lockfree_mpsc_wakeup.py
================================================================

LIVE benchmark for `lockfree_mpsc_wakeup_us` — the WAKEUP latency (us) of the LOCK-FREE MPSC
transport's HYBRID SPIN-THEN-PARK consumer: the time from a producer ENQUEUEing a node (a CAS
that publishes the new tail) to the serve core OBSERVING it and entering the batch-dequeue. The
consumer is a CHOSEN point on the spin<->futex axis: it SPINS the queue's atomic head for a
bounded window, and only PARKS on a futex/condvar if the spin window expires empty.

WHY A HYBRID, AND WHY THIS BOUND USES THE SPIN-PHASE VALUE. The pure-spin extreme (shm_spin_poll)
burns a core for the bare cache-coherence wakeup (~0.1us) but wastes the core when idle; the
pure-futex extreme sleeps (0 idle waste) but pays a syscall + scheduler wakeup (~microseconds)
on every wake. The hybrid spins first (cheap when busy) then parks (cheap when idle). THIS bound
models the SATURATION regime (regime R2 — model_cycletime.py): at saturation a node is essentially
always already enqueued when the consumer finishes a forward, so the consumer NEVER exhausts its
spin window and NEVER parks — it pays the SPIN-phase wakeup, the cross-core cache-line coherence
floor (the producer's enqueue store invalidates the consumer's cached head; the consumer's next
load takes a snoop/transfer — tens of ns). So the seed is the spin-phase coherence floor.

THE PARK COST IS REAL BUT OFF-REGIME (honestly surfaced). The futex-park + wake (a syscall each
side, ~1-5us) is paid ONLY when the queue drains below the spin window — i.e. NOT at saturation,
the regime this bound is about. run() measures the spin-phase wakeup (the in-regime value); the
docstring + config note record that the park path exists for the off-saturation regime, so the
operator can separately characterize the park cost if the realized feed is bursty rather than
saturated. Charging the park cost into the saturation bound would OVERSTATE tau_io for a regime
in which the hybrid provably never parks.

WHAT run() MEASURES (1:1). A producer thread bumps an atomic enqueue counter (the MPSC tail) at a
randomized in-spin-window delay; a consumer thread SPIN-POLLS the head and timestamps the moment
it observes the new value. The wakeup latency is (observe_ns - enqueue_ns) over many trials — the
cross-core cache-line transfer the spin pays, NO syscall in the measured path (the in-regime
hybrid stays in its spin phase). (A same-process two-thread form measures the SAME coherence floor
as two processes on two cores; the operator pins the two threads with taskset for the faithful
cross-core read.)

TIMING-SENSITIVE — DO NOT run() during the parallel fan-out. Pin: `taskset -c 0,1` (two cores).

Public Domain (The Unlicense).
"""
from __future__ import annotations

import os
import sys
import threading
import time
from typing import Any

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.dirname(_HERE), _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import estimate as _est  # noqa: E402  — the harmonized Estimate contract (measure() returns one — §6 Phase 4)
from bench_common import collect_pool, logged_run, median_estimate  # noqa: E402

NAME = "lockfree_mpsc_wakeup_us"
MODULE_PATH = "benchmarks.bench_lockfree_mpsc_wakeup"
_DESC = ("LOCK-FREE MPSC HYBRID spin-then-park wakeup latency (us): producer enqueues (CAS publishes the "
         "tail) -> the consumer, SPINNING the atomic head in its bounded spin window, observes it. At "
         "SATURATION (regime R2) the consumer never exhausts the spin window, so it pays the cross-core "
         "cache-line coherence floor (~0.1us), NOT the off-regime futex-park syscall. The wakeup lever on "
         "the spin<->futex axis; ~0 vs the per-forward cycle in the modelled saturation regime.")

_WAKEUP_SEED_US = 0.10   # cross-core cache-line snoop/transfer (a coherence miss); the spin-phase wakeup floor at saturation


def get_seed() -> tuple[float, float, str]:
    """The v1 SEED (DISTRUST fallback) — first-principles: in the saturation regime the hybrid consumer
    stays in its spin phase, so the wakeup is a single cross-core cache-line transfer (the producer's
    enqueue store -> the consumer's head load is one coherence miss), ~100 ns on a modern core. sigma
    0.05us (the snoop latency varies with the coherence state + topology). The off-regime futex-park cost
    (~1-5us) is NOT folded in (provably not paid at saturation). Returns (mean, sigma, unit)."""
    return (_WAKEUP_SEED_US, 0.05, "us")


def register_self() -> Any:
    from bench_common import register_quantity
    return register_quantity(NAME, quantity="wakeup_latency_lockfree_mpsc", units="us",
                             description=_DESC, module_path=MODULE_PATH)


def _measure_raw(trials: int = 20000) -> dict[str, Any]:
    """The raw-pool PROVENANCE producer (the §6 Phase-4 internal helper): measure the hybrid spin-phase wakeup latency: a producer thread bumps an atomic enqueue counter (a
    numpy int64 in shared memory) after a brief in-spin-window delay; a consumer thread spin-polls the head
    and records the observe time. The wakeup is (observe_ns - enqueue_ns) over `trials`. NO syscall in the
    measured spin path (the saturation regime keeps the hybrid spinning). Returns {'wakeup_us_median',
    'per_trial_us', 'trials'}. Imports numpy + shared_memory lazily. Pin two cores (taskset -c 0,1) for the
    faithful cross-core coherence read. `measure()` wraps the per-cycle pool into a median `Estimate`; `run()` uses it for BOTH the Estimate and the raw provenance rows (ONE measurement, two consumers — P1)."""
    import numpy as np
    from multiprocessing import shared_memory

    def _collect(effort: int) -> list[float]:
        """ONE producer/consumer spin batch of `effort` enqueues -> the per-trial wakeup pool (a RACE
        count <= effort; collect_pool re-runs this until the >= min_readings floor is met)."""
        shm_ctr = shared_memory.SharedMemory(create=True, size=16)
        try:
            ctr = np.ndarray((1,), dtype=np.int64, buffer=shm_ctr.buf)     # the MPSC tail (enqueue) counter
            enq_ns = np.ndarray((1,), dtype=np.int64, buffer=shm_ctr.buf, offset=8)  # the producer's enqueue stamp
            ctr[0] = 0
            per_trial_us: list[float] = []
            done = threading.Event()

            def producer() -> None:
                for k in range(1, effort + 1):
                    # brief randomized spacing within the consumer's spin window so the consumer is mid-spin when
                    # the enqueue lands (a real in-regime wakeup), not synchronized to the loop edge.
                    spin = 200 + (k * 2654435761) % 800       # ~200-1000 busy iters between enqueues
                    for _ in range(spin):
                        pass
                    enq_ns[0] = time.perf_counter_ns()        # stamp, then publish the enqueue (the CAS tail bump)
                    ctr[0] = k
                done.set()

            prod = threading.Thread(target=producer, daemon=True)
            prod.start()
            last = 0
            # CONSUMER spin: poll the head; on each new enqueue record (now - producer_stamp).
            while last < effort:
                if ctr[0] != last:
                    obs = time.perf_counter_ns()
                    last = int(ctr[0])
                    dt_us = (obs - int(enq_ns[0])) / 1000.0
                    if dt_us >= 0:                             # guard a torn read on the 64-bit stamp
                        per_trial_us.append(dt_us)
                if done.is_set() and last >= effort:
                    break
            prod.join(timeout=5.0)
            return per_trial_us
        finally:
            shm_ctr.close()
            shm_ctr.unlink()

    pool = collect_pool(_collect, name=NAME, budget=trials)   # floors the RACE count at min_readings (>= 2)
    return {"wakeup_us_median": float(np.median(pool)), "per_trial_us": pool, "trials": len(pool)}


def _estimate_from_raw(res: dict[str, Any]) -> "_est.Estimate":
    """Build this bench's harmonized `Estimate` from a `_measure_raw()` dict — the SINGLE home of the
    Estimate construction (P1), called by BOTH `measure()` and `run()`. A k=1 median `QuantileLaw(p=0.5)`
    with a BOOTSTRAP median SE over the pool (§7.A — the order-statistic variance, NOT s²/n),
    family=EMPIRICAL, kind='median'."""
    return median_estimate(res["per_trial_us"], name=NAME)   # bootstrap median SE over the per-trial pool


def measure(trials: int = 20000) -> "_est.Estimate":
    """Measure the hybrid spin-phase wakeup latency and return its harmonized k=1 median `Estimate` (§6 Phase 4: `measure()`
    returns the `Estimate` the bench DECLARES — the driver/untrusted_drive consume it directly, no
    guessing which list is the pool). The raw pool is the bench's internal `_measure_raw()` provenance.
    TIMING-SENSITIVE — pin the process (taskset -c 0)."""
    return _estimate_from_raw(_measure_raw(trials=trials))


def run(trials: int = 20000) -> dict[str, Any]:
    """Measure the hybrid spin-phase wakeup latency and LOG it as a harmonized k=1 median Estimate
    (QuantileLaw p=0.5, bootstrap median SE, §6 Phase 3, §5.2 de-dup). TIMING-SENSITIVE — operator-invoked,
    pinned (taskset -c 0,1, two cores), never during the fan-out."""
    res = _measure_raw(trials=trials)  # ONE measurement (Estimate + provenance)
    est = _estimate_from_raw(res)  # the SAME Estimate measure() returns (P1)
    cfg = {"trials": res["trials"], "transport": "lockfree_mpsc_queue", "kind": "wakeup_latency",
           "wakeup_policy": "hybrid_spin_then_park",
           "wakeup_us_median": res["wakeup_us_median"],
           "note": "saturation-regime spin-phase wakeup (cross-core cache-line coherence floor); the "
                   "off-regime futex-park syscall (~1-5us) is NOT measured here (provably not paid at saturation)"}
    with logged_run(NAME, quantity="wakeup_latency_lockfree_mpsc", units="us", description=_DESC,
                    module_path=MODULE_PATH, config=cfg, estimate=est) as log:
        # PROVENANCE only (§5.2 de-dup): the headline median lives in estimate.theta_hat[0], not a sample row.
        log(res["per_trial_us"], sample_size=1)
    return res


if __name__ == "__main__":
    _m, _s, _u = get_seed()
    print(f"[bench_lockfree_mpsc_wakeup] seed: {_m:.3f} {_u} (sigma {_s:.3f}) — first-principles "
          f"(saturation-regime spin-phase cross-core cache-line snoop; ~0 vs the per-forward cycle)")
    register_self()
    print("[bench_lockfree_mpsc_wakeup] registered. NOT running the live measurement (timing-sensitive); "
          "invoke run() pinned (taskset -c 0,1) and sole-workload.")
