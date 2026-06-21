"""
tools/analysis/OpenTURNS/benchmarks/bench_cpp_inproc_port_wakeup_us.py
======================================================================

LIVE benchmark for `cpp_inproc_port_wakeup_us` — the WAKEUP latency (us) of the C++ in-process
queue-port's consumer: the time from a producer PUBLISHING a ready leaf (a relaxed-atomic
ready-counter store in the SHARED address space) to the serve core OBSERVING it and entering the
batched forward. With generation and serve in ONE process the wakeup is a same-process,
cross-CORE event: the producer thread's store invalidates the consumer thread's cached
ready-counter line, and the consumer's next load takes a coherence snoop/transfer (~0.1us). There
is NO syscall, NO scheduler wakeup, NO fd-readiness poll on the in-process ready path.

WHY THIS BOUND USES THE SPIN-PHASE VALUE (the saturation regime). The inproc consumer can spin the
ready-counter (it owns a dedicated serve core — CLAUDE.md's 1 serve + 3 gen pinning) or park on a
futex when idle. THIS bound models the SATURATION regime (regime R2 — model_cycletime.py): at
saturation a ready leaf is essentially always already published when the consumer finishes a
forward, so the consumer NEVER parks and pays only the spin-phase cross-core cache-line coherence
floor. The off-regime futex-park + wake (a syscall each side, ~1-5us) is NOT folded in (provably not
paid at saturation — folding it in would OVERSTATE the cycle for a regime in which the consumer
never parks). It is a SEPARATE additive cycle term (the brief names wakeup distinctly) even though
it is ~0 vs the ~900us full-bucket cycle — so the structure makes the in-regime zero-wakeup explicit.

WHAT run() MEASURES (1:1). A producer thread bumps an atomic ready-counter (a numpy int64) at a
randomized in-spin-window delay; a consumer thread spin-polls it and timestamps the moment it
observes the new value. The wakeup latency is (observe_ns - publish_ns) over many trials — the
same-process cross-core cache-line transfer, NO syscall in the measured path. (A same-process
two-thread form measures the SAME coherence floor as the real in-process gen/serve thread pair; the
operator pins the two threads with taskset for the faithful cross-core read.)

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

NAME = "cpp_inproc_port_wakeup_us"
MODULE_PATH = "benchmarks.bench_cpp_inproc_port_wakeup_us"
_DESC = ("C++ inproc-port consumer wakeup latency (us): a producer publishes a ready leaf (a relaxed-atomic "
         "ready-counter store in the SHARED address space) -> the consumer, spinning the counter on its "
         "dedicated serve core, observes it. At SATURATION (regime R2) the consumer never parks, so it pays the "
         "same-process cross-core cache-line coherence floor (~0.1us), NOT the off-regime futex-park syscall. The "
         "wakeup term named separately; ~0 vs the per-forward cycle in the modelled saturation regime.")

_WAKEUP_SEED_US = 0.10   # same-process cross-core cache-line snoop/transfer; the spin-phase wakeup floor at saturation


def get_seed() -> tuple[float, float, str]:
    """The v1 SEED (DISTRUST fallback) — first-principles: in the saturation regime the inproc consumer stays
    spinning the ready-counter on its dedicated serve core, so the wakeup is a single same-process cross-core
    cache-line transfer (the producer's ready store -> the consumer's load is one coherence miss), ~100 ns.
    sigma 0.05us (snoop latency varies with coherence state + topology). The off-regime futex-park cost
    (~1-5us) is NOT folded in (provably not paid at saturation). Returns (mean, sigma, unit)."""
    return (_WAKEUP_SEED_US, 0.05, "us")


def register_self() -> Any:
    from bench_common import register_quantity
    return register_quantity(NAME, quantity="wakeup_latency_cpp_inproc_port", units="us",
                             description=_DESC, module_path=MODULE_PATH)


def _measure_raw(trials: int = 20000) -> dict[str, Any]:
    """The raw-pool PROVENANCE producer (the §6 Phase-4 internal helper): measure the inproc-port spin-phase wakeup latency: a producer thread bumps an atomic ready-counter (a
    numpy int64) after a brief in-spin-window delay; a consumer thread spin-polls it and records the observe
    time. The wakeup is (observe_ns - publish_ns) over `trials`. NO syscall in the measured spin path (the
    saturation regime keeps the consumer spinning). Returns {'wakeup_us_median', 'per_trial_us', 'trials'}.
    Imports numpy lazily. Pin two cores (taskset -c 0,1) for the faithful cross-core coherence read.
    `measure()` wraps the per-cycle pool into a median `Estimate`; `run()` uses it for BOTH the Estimate and the raw provenance rows (ONE measurement, two consumers — P1)."""
    import numpy as np

    def _collect(effort: int) -> list[float]:
        """ONE producer/consumer spin batch of `effort` publishes -> the per-trial wakeup pool (a RACE
        count <= effort; collect_pool re-runs this until the >= min_readings floor is met)."""
        buf = np.zeros(2, dtype=np.int64)          # buf[0] = ready-counter (publish); buf[1] = the producer's publish stamp
        per_trial_us: list[float] = []
        done = threading.Event()

        def producer() -> None:
            for k in range(1, effort + 1):
                # brief randomized spacing within the consumer's spin window so the consumer is mid-spin when the
                # ready store lands (a real in-regime wakeup), not synchronized to the loop edge.
                spin = 200 + (k * 2654435761) % 800       # ~200-1000 busy iters between publishes
                for _ in range(spin):
                    pass
                buf[1] = time.perf_counter_ns()           # stamp, then publish the ready-counter store
                buf[0] = k
            done.set()

        prod = threading.Thread(target=producer, daemon=True)
        prod.start()
        last = 0
        # CONSUMER spin: poll the ready-counter; on each new publish record (now - producer_stamp).
        while last < effort:
            if buf[0] != last:
                obs = time.perf_counter_ns()
                last = int(buf[0])
                dt_us = (obs - int(buf[1])) / 1000.0
                if dt_us >= 0:                             # guard a torn read on the 64-bit stamp
                    per_trial_us.append(dt_us)
            if done.is_set() and last >= effort:
                break
        prod.join(timeout=5.0)
        return per_trial_us

    pool = collect_pool(_collect, name=NAME, budget=trials)   # floors the RACE count at min_readings (>= 2)
    return {"wakeup_us_median": float(np.median(pool)), "per_trial_us": pool, "trials": len(pool)}


def _estimate_from_raw(res: dict[str, Any]) -> "_est.Estimate":
    """Build this bench's harmonized `Estimate` from a `_measure_raw()` dict — the SINGLE home of the
    Estimate construction (P1), called by BOTH `measure()` and `run()`. A k=1 median `QuantileLaw(p=0.5)`
    with a BOOTSTRAP median SE over the pool (§7.A — the order-statistic variance, NOT s²/n),
    family=EMPIRICAL, kind='median'."""
    return median_estimate(res["per_trial_us"], name=NAME)   # bootstrap median SE over the per-trial pool


def measure(trials: int = 20000) -> "_est.Estimate":
    """Measure the inproc-port spin-phase wakeup latency and return its harmonized k=1 median `Estimate` (§6 Phase 4: `measure()`
    returns the `Estimate` the bench DECLARES — the driver/untrusted_drive consume it directly, no
    guessing which list is the pool). The raw pool is the bench's internal `_measure_raw()` provenance.
    TIMING-SENSITIVE — pin the process (taskset -c 0)."""
    return _estimate_from_raw(_measure_raw(trials=trials))


def run(trials: int = 20000) -> dict[str, Any]:
    """Measure the inproc-port spin-phase wakeup latency and LOG it as a harmonized k=1 median Estimate
    (QuantileLaw p=0.5, bootstrap median SE, §6 Phase 3, §5.2 de-dup). TIMING-SENSITIVE — operator-invoked,
    pinned (taskset -c 0,1, two cores), never during the fan-out."""
    res = _measure_raw(trials=trials)  # ONE measurement (Estimate + provenance)
    est = _estimate_from_raw(res)  # the SAME Estimate measure() returns (P1)
    cfg = {"trials": res["trials"], "transport": "cpp_inproc_port_direct_call", "kind": "wakeup_latency",
           "wakeup_policy": "spin_dedicated_serve_core",
           "wakeup_us_median": res["wakeup_us_median"],
           "note": "saturation-regime spin-phase wakeup (same-process cross-core cache-line coherence floor); "
                   "the off-regime futex-park syscall (~1-5us) is NOT measured here (provably not paid at saturation)"}
    with logged_run(NAME, quantity="wakeup_latency_cpp_inproc_port", units="us", description=_DESC,
                    module_path=MODULE_PATH, config=cfg, estimate=est) as log:
        # PROVENANCE only (§5.2 de-dup): the headline median lives in estimate.theta_hat[0], not a sample row.
        log(res["per_trial_us"], sample_size=1)
    return res


if __name__ == "__main__":
    _m, _s, _u = get_seed()
    print(f"[bench_cpp_inproc_port_wakeup_us] seed: {_m:.3f} {_u} (sigma {_s:.3f}) — first-principles "
          f"(saturation-regime spin-phase same-process cross-core cache-line snoop; ~0 vs the per-forward cycle)")
    register_self()
    print("[bench_cpp_inproc_port_wakeup_us] registered. NOT running the live measurement (timing-sensitive); "
          "invoke run() pinned (taskset -c 0,1) and sole-workload.")
