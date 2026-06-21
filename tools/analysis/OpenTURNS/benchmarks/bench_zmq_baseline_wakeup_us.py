"""
tools/analysis/OpenTURNS/benchmarks/bench_zmq_baseline_wakeup_us.py
==================================================================

LIVE benchmark for `zmq_baseline_wakeup_us` — the ZMQ BASELINE transport's per-forward FIRST-
REQUEST WAKEUP latency (us): the cost of the `zmq.Poller.poll()` syscall returning POLLIN once a
request is readiness-notified on the ROUTER fd (`inference_server.py _drain`: the bounded
`self._poller.poll(timeout=self._POLL_INTERVAL_MS)` that blocks until ≥1 request is queued, then
breaks to the greedy NOBLOCK drain). This is the WAKEUP term of the zmq_baseline transport
profile — NAMED SEPARATELY from `zmq_baseline_tau_io_us` so the design sweep can contrast wakeup
MECHANISMS across variants (a futex_wake's FUTEX_WAKE→FUTEX_WAIT handoff, a shm_spin_poll's
busy-poll cache-line read, a lockfree_mpsc's notify, a cpp_inproc_port's in-process enqueue all
register their own `<slug>_wakeup_us`).

WHAT IS BEING MEASURED (the SATURATION wakeup, not the idle re-check). `_POLL_INTERVAL_MS = 100`
is the IDLE re-check cadence — the wakeup-to-recheck so a flipped `_stop` is observed (an idle
server parks at ~0 CPU). That 100ms is NOT in the binding cycle: at SATURATION (the regime R2
this bound models) a request is ALREADY queued when the loop polls, so `poll()` returns at once
and the per-forward wakeup is the POLL-ON-READY cost — the syscall + the libzmq I/O-thread's fd
readiness signal (an eventfd/mailbox notify). That ready-poll latency is what this bench times:
prime the socket with a queued frame, then time `poller.poll(0)` (or `poll(timeout)`) returning
POLLIN. (The full idle-wakeup tail — a poll that BLOCKS until a frame arrives — is reported as a
supporting reading for honesty, but the model input is the saturation ready-poll cost, since the
bound is a saturated-throughput floor.)

SEED PROVENANCE (a NEW first-principles quantity — NOT in the v1 grounding, so it carries its own
estimate, not a delegated literal). A ZMQ `poll()` on a ready ROUTER fd is one `poll(2)`/`epoll`
syscall plus libzmq's signaler readiness path (the I/O thread writes the mailbox eventfd; the
app thread's poller reads readiness). On a modern Linux a hot syscall round-trip is ~0.5–1us and
the signaler adds ~0.5–1us, so a saturation ready-poll wakeup is ~1.5us (sigma 1.0us — it spans
a sub-us hot path to a few-us cold one). It sits in the BINDING serve cycle (added to tau_io), so
the Neyman allocator ranks it — though it is ~10x below tau_io, so it ranks well behind it. A
sole-workload run of this bench grounds it directly.

TIMING-SENSITIVE — DO NOT run() during the parallel fan-out. Pin: `taskset -c 0`, sole-workload.

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

import estimate as _est  # noqa: E402  — the harmonized Estimate contract (measure() returns one — §6 Phase 4)
import leaf_eval_grounding as G  # noqa: E402
from bench_common import logged_run, median_estimate  # noqa: E402

NAME = "zmq_baseline_wakeup_us"
MODULE_PATH = "benchmarks.bench_zmq_baseline_wakeup_us"
_DESC = ("ZMQ BASELINE per-forward first-request WAKEUP latency (us): zmq.Poller.poll() returning POLLIN "
         "on a readiness-notified ROUTER fd at saturation (inference_server.py _drain bounded poll). The "
         "wakeup term of the zmq_baseline profile; NAMED separately so the sweep contrasts wakeup "
         "mechanisms across variants. NEW first-principles quantity (seed 1.5us: poll(2) syscall + libzmq "
         "signaler readiness; not in the v1 grounding).")

# First-principles seed (a NEW quantity — this module is the seed's single home, with provenance above).
_SEED_MEAN_US = 1.5
_SEED_SIGMA_US = 1.0
_SEED_UNIT = "us"
_IN_DIM = 241


def get_seed() -> tuple[float, float, str]:
    """The v1-style seed (DISTRUST fallback) for this NEW quantity: (mean, sigma, unit) = (1.5, 1.0, 'us').
    A saturation ready-poll wakeup = a poll(2) syscall + libzmq signaler readiness (provenance in the module
    docstring). Returned as a tuple (the manifest accepts a Grounded-like OR a (mean, sigma, unit) tuple)."""
    return (_SEED_MEAN_US, _SEED_SIGMA_US, _SEED_UNIT)


def register_self() -> Any:
    from bench_common import register_quantity
    return register_quantity(NAME, quantity="transport_wakeup_latency", units=_SEED_UNIT,
                             description=_DESC, module_path=MODULE_PATH)


def _measure_raw(cycles: int = 20000) -> dict[str, Any]:
    """The raw-pool PROVENANCE producer (the §6 Phase-4 internal helper): measure the ZMQ-baseline saturation wakeup: prime the ROUTER with a queued frame, then time
    `poller.poll(0)` returning POLLIN (the ready-poll cost the saturated _drain pays), draining the frame
    each cycle to re-prime. Also records the BLOCKING-poll tail (poll that waits for the frame to arrive)
    as a supporting reading. NO forward, NO codec beyond a 1-row frame. Returns {'wakeup_us_median',
    'ready_poll_us': [...], 'blocking_poll_us_median'}. Imports zmq + numpy lazily. Pin (taskset -c 0).
    `measure()` wraps the per-cycle pool into a median `Estimate`; `run()` uses it for BOTH the Estimate and the raw provenance rows (ONE measurement, two consumers — P1)."""
    import numpy as np
    import zmq
    from chocofarm.az.inference_wire import encode_request

    ctx = zmq.Context.instance()
    router = ctx.socket(zmq.ROUTER)
    ep = "inproc://zmq_baseline_wakeup_bench"
    router.bind(ep)
    dealer = ctx.socket(zmq.DEALER)
    dealer.setsockopt(zmq.IDENTITY, b"w0")
    dealer.connect(ep)
    poller = zmq.Poller()
    poller.register(router, zmq.POLLIN)

    one_row = encode_request(np.zeros((1, _IN_DIM), dtype=np.float32))
    ready_poll_us: list[float] = []
    blocking_poll_us: list[float] = []
    try:
        # SATURATION ready-poll: a frame is already queued; time poll(0) returning POLLIN, then drain.
        for _ in range(cycles):
            dealer.send(one_row)
            # Spin until the inproc frame is actually queued (inproc delivery is near-instant but not
            # synchronous), so we time a READY poll, not an arrival wait — the saturation case.
            while not poller.poll(0):
                pass
            t0 = time.perf_counter_ns()
            events = poller.poll(0)
            ready_poll_us.append((time.perf_counter_ns() - t0) / 1000.0)
            if events:
                router.recv_multipart(flags=zmq.NOBLOCK)   # drain to re-prime next cycle

        # BLOCKING-poll tail (supporting, honesty): time poll(timeout) that WAITS for the arrival.
        tail = min(2000, cycles)
        for _ in range(tail):
            t0 = time.perf_counter_ns()
            dealer.send(one_row)
            poller.poll(timeout=100)   # blocks until the just-sent frame is readiness-notified
            blocking_poll_us.append((time.perf_counter_ns() - t0) / 1000.0)
            router.recv_multipart(flags=zmq.NOBLOCK)
    finally:
        dealer.close(linger=0)
        router.close(linger=0)
    med = float(np.median(ready_poll_us))
    return {"wakeup_us_median": med, "ready_poll_us": ready_poll_us,
            "blocking_poll_us_median": float(np.median(blocking_poll_us)) if blocking_poll_us else med,
            "cycles": cycles}


def _estimate_from_raw(res: dict[str, Any]) -> "_est.Estimate":
    """Build this bench's harmonized `Estimate` from a `_measure_raw()` dict — the SINGLE home of the
    Estimate construction (P1), called by BOTH `measure()` and `run()`. A k=1 median `QuantileLaw(p=0.5)`
    with a BOOTSTRAP median SE over the pool (§7.A — the order-statistic variance, NOT s²/n),
    family=EMPIRICAL, kind='median'."""
    return median_estimate(res["ready_poll_us"], name=NAME)   # bootstrap median SE over the ready_poll pool


def measure(cycles: int = 20000) -> "_est.Estimate":
    """Measure the ZMQ-baseline saturation wakeup and return its harmonized k=1 median `Estimate` (§6 Phase 4: `measure()`
    returns the `Estimate` the bench DECLARES — the driver/untrusted_drive consume it directly, no
    guessing which list is the pool). The raw pool is the bench's internal `_measure_raw()` provenance.
    TIMING-SENSITIVE — pin the process (taskset -c 0)."""
    return _estimate_from_raw(_measure_raw(cycles=cycles))


def run(cycles: int = 20000) -> dict[str, Any]:
    """Measure the ZMQ-baseline wakeup and LOG it as a harmonized k=1 median Estimate (QuantileLaw p=0.5,
    bootstrap median SE, §6 Phase 3, §5.2 de-dup). TIMING-SENSITIVE — operator-invoked, pinned, never
    during the fan-out."""
    res = _measure_raw(cycles=cycles)  # ONE measurement (Estimate + provenance)
    est = _estimate_from_raw(res)  # the SAME Estimate measure() returns (P1)
    cfg = {"cycles": res["cycles"], "transport": "zmq_baseline_router_dealer_inproc",
           "mechanism": "poll(2)+libzmq_signaler_readiness", "regime": "saturation_ready_poll",
           "blocking_poll_us_median": res["blocking_poll_us_median"],
           "wakeup_us_median": res["wakeup_us_median"]}
    with logged_run(NAME, quantity="transport_wakeup_latency", units=_SEED_UNIT, description=_DESC,
                    module_path=MODULE_PATH, config=cfg, estimate=est) as log:
        # PROVENANCE only (§5.2 de-dup): the headline median lives in estimate.theta_hat[0], not a sample row.
        log(res["ready_poll_us"], sample_size=1)                    # raw per-cycle readings
    return res


if __name__ == "__main__":
    m, s, u = get_seed()
    print(f"[bench_zmq_baseline_wakeup_us] seed: {m} {u} (sigma {s}; NEW first-principles quantity — "
          f"poll(2) syscall + libzmq signaler readiness; provenance in module docstring)")
    register_self()
    print("[bench_zmq_baseline_wakeup_us] registered. NOT running the live measurement (timing-sensitive); "
          "invoke run() pinned and sole-workload. Ranks BEHIND tau_io (~10x smaller) for the Neyman loop.")
