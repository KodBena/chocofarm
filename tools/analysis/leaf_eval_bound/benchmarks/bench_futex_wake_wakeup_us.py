"""
tools/analysis/leaf_eval_bound/benchmarks/bench_futex_wake_wakeup_us.py
================================================================

LIVE benchmark for `futex_wake_wakeup_us` — the WAKEUP latency (us) of the FUTEX-WAKE
transport: the time from a producer issuing `FUTEX_WAKE` on the ring's empty->nonempty edge
to the parked serve thread RETURNING from its `FUTEX_WAIT` and resuming its drain. This is
the lever that distinguishes futex_wake from its sibling shm transports:

  * shm_spin_poll pays ~0 wakeup (a cached cross-core cache-line load, ~0.1us) but BURNS a
    core spinning a counter that never sleeps.
  * futex_wake pays ONE futex syscall round-trip on the empty->nonempty edge — the producer's
    `FUTEX_WAKE` syscall + the kernel scheduler waking the parked serve thread + the serve
    thread's `FUTEX_WAIT` syscall-return — in exchange for NOT burning a core (the serve core
    sleeps when the ring is empty). On a modern Linux that handoff is ~1-3us (a futex wake +
    a context switch onto the woken thread).
  * zmq_baseline pays the `zmq.Poller.poll()` syscall + libzmq's signaler readiness path
    (~1.5us seed) — a comparable syscall-class cost, but through the broker/eventfd machinery
    rather than the bare kernel futex.

This is the "group-wakeup the convoy work studied": when the ring goes empty->nonempty, the
producer wakes the one parked consumer. (A FUTEX_WAKE that wakes N waiters is the convoy
thundering-herd case; here the serve side is a SINGLE consumer, so it is a 1-waiter wake —
the cheap end of the futex-wake spectrum.)

WHERE IT SITS IN THE BOUND (the honest saturation subtlety). The bound models the SATURATED
regime R2. At saturation the ring is rarely empty: the serve core finishes forward k and
requests for forward k+1 are ALREADY queued, so it does NOT FUTEX_WAIT — it sees the ring
nonempty and drains directly, paying ~0 wakeup (like the spin case). The futex syscall is
paid ONLY on the empty->nonempty EDGE, a FRACTION of forwards at saturation. A strict LOWER
bound takes the pessimistic arm (the serve core parks after EVERY forward and must be woken),
so the model charges this wakeup PER FORWARD; `model_futex_wake.saturation_wakeup_contrast`
then shows the amortized (edge-fraction) arm honestly. THIS bench measures the per-edge
handoff cost — the quantity both arms scale.

WHAT run() MEASURES (1:1, NO JAX forward). Two threads sharing a futex word (a 4-byte int in
shared memory). The consumer FUTEX_WAITs on the word; the producer, after a brief spacing so
the consumer is genuinely parked (not racing the wait), stamps `perf_counter_ns`, stores the
new value, and FUTEX_WAKEs one waiter. The consumer returns from FUTEX_WAIT and stamps the
observe time. The wakeup is (observe_ns - wake_ns) over many trials — the FUTEX_WAKE syscall
+ scheduler context-switch the parked serve thread pays. The futex syscalls go through
`ctypes.CDLL(None).syscall(SYS_futex, ...)` (the bare kernel futex, no pthread/glibc condvar
wrapper) so the measured path is the kernel handoff itself. (A same-process two-thread form
measures the SAME kernel futex-wake + context-switch cost as two processes on two cores; the
operator pins the two threads to two cores with taskset for the faithful cross-core read.)

SEED PROVENANCE (a NEW first-principles quantity — NOT in the v1 grounding). A FUTEX_WAKE of
one waiter is one `futex(2)` syscall on the producer + a scheduler wakeup that context-switches
onto the parked consumer + the consumer's `futex(2)` return. A hot syscall round-trip is
~0.5-1us and a context switch onto an already-runnable thread on another core is ~1-2us, so the
edge handoff is ~2us (sigma 1.0us — it spans a ~1us hot path to a ~3us cold/loaded one). It
sits in the BINDING serve cycle (added to tau_io) under the pessimistic per-forward arm, so the
Neyman allocator ranks it — though at a full bucket the cycle is compute-bound (B*t_row ~=
1105us at B_op=256), so even the pessimistic 3us moves the serve bound by <1 dps; it ranks
WELL BEHIND tau_io. A sole-workload run of this bench grounds it directly.

TIMING-SENSITIVE — DO NOT run() during the parallel fan-out. Pin: `taskset -c 0,1` (two cores).

Public Domain (The Unlicense).
"""
from __future__ import annotations

import os
import sys
from typing import Any

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.dirname(_HERE), _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import estimate as _est  # noqa: E402  — the harmonized Estimate contract (measure() returns one — §6 Phase 4)
from bench_common import collect_pool, logged_run, median_estimate  # noqa: E402

NAME = "futex_wake_wakeup_us"
MODULE_PATH = "benchmarks.bench_futex_wake_wakeup_us"
_DESC = ("FUTEX-WAKE wakeup latency (us): a producer FUTEX_WAKEs the parked serve thread on the ring's "
         "empty->nonempty edge -> the serve thread RETURNS from FUTEX_WAIT and resumes its drain. ONE futex "
         "syscall + a scheduler context-switch (no burnt core, vs shm_spin_poll's ~0.1us spin; comparable "
         "to zmq's poll path but bare-kernel). NEW first-principles quantity (seed ~2us). Charged per-forward "
         "in the pessimistic lower-bound arm; amortized by the empty-edge fraction at saturation.")

# First-principles seed (a NEW quantity — this module is the seed's single home, provenance in the docstring).
_SEED_MEAN_US = 2.0    # FUTEX_WAKE syscall + scheduler context-switch onto the parked one-waiter consumer
_SEED_SIGMA_US = 1.0   # ~1us hot path to ~3us cold/loaded; spans the futex-wake + ctx-switch spread
_SEED_UNIT = "us"


def get_seed() -> tuple[float, float, str]:
    """The v1-style SEED (DISTRUST fallback) for this NEW quantity: (mean, sigma, unit) = (2.0, 1.0, 'us').
    A one-waiter FUTEX_WAKE edge handoff = a futex(2) syscall + a context switch onto the parked serve thread
    (provenance in the module docstring). Returned as a (mean, sigma, unit) tuple (the manifest accepts a
    Grounded-like OR a tuple)."""
    return (_SEED_MEAN_US, _SEED_SIGMA_US, _SEED_UNIT)


def register_self() -> Any:
    from bench_common import register_quantity
    return register_quantity(NAME, quantity="transport_wakeup_latency_futex_wake", units=_SEED_UNIT,
                             description=_DESC, module_path=MODULE_PATH)


def _measure_raw(trials: int = 20000) -> dict[str, Any]:
    """The raw-pool PROVENANCE producer (the §6 Phase-4 internal helper): measure the futex-wake edge handoff: a consumer thread FUTEX_WAITs on a shared 4-byte word; a producer
    thread (after a brief spacing so the consumer is genuinely parked) stamps perf_counter_ns, stores the new
    value, and FUTEX_WAKEs one waiter; the consumer returns from FUTEX_WAIT and stamps the observe time. The
    wakeup is (observe_ns - wake_ns) over `trials`. The futex syscalls are the bare kernel `futex(2)` via
    ctypes (no pthread/condvar wrapper). Returns {'wakeup_us_median', 'per_trial_us', 'trials'}. Imports
    numpy + ctypes + shared_memory lazily. Pin two cores (taskset -c 0,1) for the faithful cross-core read.
    `measure()` wraps the per-cycle pool into a median `Estimate`; `run()` uses it for BOTH the Estimate and the raw provenance rows (ONE measurement, two consumers — P1)."""
    import ctypes
    import threading
    import time
    from multiprocessing import shared_memory

    import numpy as np

    # Bare kernel futex(2). FUTEX_WAIT=0, FUTEX_WAKE=1 (private variants add FUTEX_PRIVATE_FLAG=128 — we use the
    # shared variants since the futex word lives in shared memory, the faithful cross-process layout). On x86-64
    # __NR_futex = 202. A futex syscall failing for a reason OTHER than the expected EAGAIN (the value already
    # changed before WAIT) raises (ADR-0002 — a real syscall fault is surfaced, not swallowed).
    _SYS_futex = 202
    _FUTEX_WAIT = 0
    _FUTEX_WAKE = 1
    libc = ctypes.CDLL(None, use_errno=True)

    def _collect(effort: int) -> list[float]:
        """ONE producer/consumer futex-edge batch of `effort` wakes -> the per-trial wakeup pool (a RACE
        count <= effort; collect_pool re-runs this until the >= min_readings floor is met)."""
        shm = shared_memory.SharedMemory(create=True, size=8)
        try:
            word = np.ndarray((1,), dtype=np.int32, buffer=shm.buf)            # the futex word (consumer waits on it)
            word[0] = 0
            # The producer's wake stamp lives in its OWN small buffer (a separate 64-bit int) so it never overlaps
            # the 4-byte futex word — a torn/overlapping write would corrupt the latency reading.
            shm_stamp = shared_memory.SharedMemory(create=True, size=8)
            stamp = np.ndarray((1,), dtype=np.int64, buffer=shm_stamp.buf)
            stamp[0] = 0

            addr = ctypes.addressof(ctypes.c_char.from_buffer(shm.buf))         # &word[0] for the futex syscall
            per_trial_us: list[float] = []
            stop = threading.Event()

            def _futex_wait(expected: int) -> int:
                # FUTEX_WAIT(addr, expected): sleep if *addr == expected; return 0 on wake, -EAGAIN if it already
                # changed. NULL timeout (block). errno EAGAIN/EINTR are the benign "value moved / spurious" cases.
                r = libc.syscall(_SYS_futex, ctypes.c_void_p(addr), _FUTEX_WAIT,
                                 ctypes.c_int(expected), ctypes.c_void_p(0), ctypes.c_void_p(0), ctypes.c_int(0))
                if r != 0:
                    err = ctypes.get_errno()
                    if err not in (11, 4):   # EAGAIN=11 (value changed), EINTR=4 (spurious) — both benign
                        raise OSError(err, f"FUTEX_WAIT failed: errno {err}")
                return r

            def _futex_wake(n: int = 1) -> int:
                r = libc.syscall(_SYS_futex, ctypes.c_void_p(addr), _FUTEX_WAKE,
                                 ctypes.c_int(n), ctypes.c_void_p(0), ctypes.c_void_p(0), ctypes.c_int(0))
                if r < 0:
                    raise OSError(ctypes.get_errno(), "FUTEX_WAKE failed")
                return r   # number of waiters woken

            def consumer() -> None:
                seen = 0
                while seen < effort and not stop.is_set():
                    expected = seen          # park while the word still reads the value we last consumed
                    if word[0] == expected:
                        _futex_wait(expected)     # sleep until the producer bumps the word and wakes us
                    obs = time.perf_counter_ns()
                    if word[0] != expected:
                        dt_us = (obs - int(stamp[0])) / 1000.0
                        if 0 <= dt_us < 1e6:      # guard a torn/racing stamp read
                            per_trial_us.append(dt_us)
                        seen = int(word[0])

            cons = threading.Thread(target=consumer, daemon=True)
            cons.start()

            # PRODUCER: bump the word + FUTEX_WAKE the parked consumer, with spacing so the consumer is parked.
            for k in range(1, effort + 1):
                # Busy-spin a short gap so the consumer reaches FUTEX_WAIT before we wake it (a real edge handoff,
                # not a wake of a not-yet-parked thread).
                for _ in range(2000):
                    pass
                stamp[0] = time.perf_counter_ns()    # stamp, then publish + wake
                word[0] = k                          # the empty->nonempty edge value
                _futex_wake(1)                       # wake the one parked waiter
            stop.set()
            word[0] = effort + 1
            _futex_wake(1)
            cons.join(timeout=10.0)

            shm_stamp.close()
            shm_stamp.unlink()
            return per_trial_us
        finally:
            shm.close()
            shm.unlink()

    pool = collect_pool(_collect, name=NAME, budget=trials)   # floors the RACE count at min_readings (>= 2)
    return {"wakeup_us_median": float(np.median(pool)), "per_trial_us": pool, "trials": len(pool)}


def _estimate_from_raw(res: dict[str, Any]) -> "_est.Estimate":
    """Build this bench's harmonized `Estimate` from a `_measure_raw()` dict — the SINGLE home of the
    Estimate construction (P1), called by BOTH `measure()` and `run()`. A k=1 median `QuantileLaw(p=0.5)`
    with a BOOTSTRAP median SE over the pool (§7.A — the order-statistic variance, NOT s²/n),
    family=EMPIRICAL, kind='median'."""
    return median_estimate(res["per_trial_us"], name=NAME)   # bootstrap median SE over the per-trial pool


def measure(trials: int = 20000) -> "_est.Estimate":
    """Measure the futex-wake edge handoff and return its harmonized k=1 median `Estimate` (§6 Phase 4: `measure()`
    returns the `Estimate` the bench DECLARES — the driver/untrusted_drive consume it directly, no
    guessing which list is the pool). The raw pool is the bench's internal `_measure_raw()` provenance.
    TIMING-SENSITIVE — pin the process (taskset -c 0)."""
    return _estimate_from_raw(_measure_raw(trials=trials))


def run(trials: int = 20000) -> dict[str, Any]:
    """Measure the futex-wake edge handoff and LOG it as a harmonized k=1 median Estimate (QuantileLaw p=0.5,
    bootstrap median SE, §6 Phase 3, §5.2 de-dup). TIMING-SENSITIVE — operator-invoked, pinned (taskset -c 0,1,
    two cores), NEVER during the fan-out."""
    res = _measure_raw(trials=trials)  # ONE measurement (Estimate + provenance)
    est = _estimate_from_raw(res)  # the SAME Estimate measure() returns (P1)
    cfg = {"trials": res["trials"], "transport": "shm_ring_futex_wake", "kind": "wakeup_latency",
           "mechanism": "bare_kernel_FUTEX_WAKE_one_waiter + scheduler_context_switch",
           "regime": "per_edge_handoff",
           "wakeup_us_median": res["wakeup_us_median"],
           "note": "one-waiter futex wake on the ring empty->nonempty edge; no burnt core (serve sleeps); "
                   "charged per-forward in the pessimistic bound arm, amortized by the edge fraction at saturation"}
    with logged_run(NAME, quantity="transport_wakeup_latency_futex_wake", units=_SEED_UNIT, description=_DESC,
                    module_path=MODULE_PATH, config=cfg, estimate=est) as log:
        # PROVENANCE only (§5.2 de-dup): the headline median lives in estimate.theta_hat[0], not a sample row.
        log(res["per_trial_us"], sample_size=1)                     # raw per-trial readings
    return res


if __name__ == "__main__":
    m, s, u = get_seed()
    print(f"[bench_futex_wake_wakeup_us] seed: {m} {u} (sigma {s}; NEW first-principles quantity — "
          f"one-waiter FUTEX_WAKE syscall + scheduler context-switch; provenance in module docstring)")
    register_self()
    print("[bench_futex_wake_wakeup_us] registered. NOT running the live measurement (timing-sensitive); "
          "invoke run() pinned (taskset -c 0,1, two cores) and sole-workload. Ranks BEHIND tau_io for the "
          "Neyman loop (at a full bucket the cycle is compute-bound, so even 3us moves serve <1 dps).")
