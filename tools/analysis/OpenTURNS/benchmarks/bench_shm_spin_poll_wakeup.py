"""
tools/analysis/OpenTURNS/benchmarks/bench_shm_spin_poll_wakeup.py
================================================================

LIVE benchmark for `shm_spin_poll_wakeup_us` — the WAKEUP latency (us) of the SHM SPIN-POLL
transport: the time from a producer bumping the shared atomic tail counter to the spinning
serve core OBSERVING the bumped value and breaking out of its busy-poll loop. There is NO
syscall (no `zmq.poll`, no futex), NO context switch, NO scheduler involvement — the serve
core never sleeps; it spins a cached counter on a DEDICATED burnt poll core. So the wakeup
is just the cross-core CACHE-LINE coherence latency: the producer's store invalidates the
server's cached copy, and the server's next load takes a snoop/transfer from the producer's
cache (tens of ns on a modern machine).

WHY IT MATTERS (and why it is ~0). In the ZMQ baseline the per-forward wakeup is folded into
the blocking `recv`/`poll` syscall path — a syscall + a scheduler wakeup is microseconds. The
SHM spin-poll's defining trade is to BURN one core to drive that wakeup to the bare cache-
coherence floor. The fixed pinning layout (1 serve core + 3 gen cores, isolcpus 1-3) ALREADY
dedicates the serve core, so spinning it costs no extra core. This quantity is reported
SEPARATELY from tau_io (the brief names it as a distinct lever); it folds into the per-forward
cycle as an additive wakeup term, but at ~0.1us it is negligible vs the ~900us+ full-bucket
cycle — its value is to MAKE EXPLICIT that the spin transport pays ~0 wakeup, the property
that distinguishes it from the syscall-wakeup transports.

WHAT run() MEASURES (1:1). A producer thread bumps an atomic counter in shared memory at a
random delay; a server thread spin-polls it and timestamps the moment it observes the new
value. The wakeup latency is (observe_ns - bump_ns) over many trials — the cross-core
cache-line transfer the spin pays. NO syscall in the measured path. (A same-process two-thread
form measures the SAME coherence floor as two processes on two cores; the operator pins the
two threads to two cores with taskset for the faithful cross-core read.)

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

from bench_common import logged_run  # noqa: E402

NAME = "shm_spin_poll_wakeup_us"
MODULE_PATH = "benchmarks.bench_shm_spin_poll_wakeup"
_DESC = ("SHM SPIN-POLL wakeup latency (us): producer bumps an atomic tail counter -> the spinning serve "
         "core observes it. NO syscall, NO context switch (a dedicated burnt poll core) — just the cross-core "
         "cache-line coherence floor (~0.1us). The lever distinguishing the spin transport from syscall-wakeup "
         "transports; ~0 vs the per-forward cycle.")

_WAKEUP_SEED_US = 0.10   # cross-core cache-line snoop/transfer (a coherence miss); the spin-wakeup floor


def get_seed() -> tuple[float, float, str]:
    """The v1 SEED (DISTRUST fallback) — first-principles: a single cross-core cache-line transfer (the
    producer's store -> the server's load is one coherence miss), ~100 ns on a modern core. sigma 0.05us
    (the snoop latency varies with the coherence state + the topology). Returns (mean, sigma, unit)."""
    return (_WAKEUP_SEED_US, 0.05, "us")


def register_self() -> Any:
    from bench_common import register_quantity
    return register_quantity(NAME, quantity="wakeup_latency_shm_spin_poll", units="us",
                             description=_DESC, module_path=MODULE_PATH)


def measure(trials: int = 20000) -> dict[str, Any]:
    """Measure the spin-poll wakeup latency: a producer thread bumps an atomic counter (a numpy int64 in
    shared memory) after a brief spin-delay; a server thread spin-polls it and records the observe time.
    The wakeup is (observe_ns - bump_ns) over `trials`. NO syscall in the measured spin path. Returns
    {'wakeup_us_median', 'per_trial_us', 'trials'}. Imports numpy + shared_memory lazily. Pin two cores
    (taskset -c 0,1) for the faithful cross-core coherence read."""
    import numpy as np
    from multiprocessing import shared_memory

    shm_ctr = shared_memory.SharedMemory(create=True, size=16)
    try:
        ctr = np.ndarray((1,), dtype=np.int64, buffer=shm_ctr.buf)     # the atomic tail counter
        bump_ns = np.ndarray((1,), dtype=np.int64, buffer=shm_ctr.buf, offset=8)  # the producer's stamp
        ctr[0] = 0
        per_trial_us: list[float] = []
        done = threading.Event()

        def producer() -> None:
            for k in range(1, trials + 1):
                # brief randomized spacing so the server is mid-spin when the bump lands (a real wakeup),
                # not synchronized to the loop edge.
                spin = 200 + (k * 2654435761) % 800       # ~200-1000 busy iters between bumps
                for _ in range(spin):
                    pass
                bump_ns[0] = time.perf_counter_ns()       # stamp, then publish
                ctr[0] = k                                 # the bump the server spins for
            done.set()

        prod = threading.Thread(target=producer, daemon=True)
        prod.start()
        last = 0
        # SERVER spin: poll the counter; on each new value record (now - producer_stamp).
        while last < trials:
            if ctr[0] != last:
                obs = time.perf_counter_ns()
                last = int(ctr[0])
                dt_us = (obs - int(bump_ns[0])) / 1000.0
                if dt_us >= 0:                             # guard a torn read on the 64-bit stamp
                    per_trial_us.append(dt_us)
            if done.is_set() and last >= trials:
                break
        prod.join(timeout=5.0)
        med = float(np.median(per_trial_us)) if per_trial_us else float("nan")
        return {"wakeup_us_median": med, "per_trial_us": per_trial_us, "trials": len(per_trial_us)}
    finally:
        shm_ctr.close()
        shm_ctr.unlink()


def run(trials: int = 20000) -> dict[str, Any]:
    """Measure the spin-poll wakeup latency and LOG it. TIMING-SENSITIVE — operator-invoked, pinned
    (taskset -c 0,1, two cores), never during the fan-out."""
    res = measure(trials=trials)
    cfg = {"trials": res["trials"], "transport": "shm_ring_spin_poll", "kind": "wakeup_latency",
           "note": "cross-core cache-line coherence floor; no syscall, no context switch (dedicated poll core)"}
    with logged_run(NAME, quantity="wakeup_latency_shm_spin_poll", units="us", description=_DESC,
                    module_path=MODULE_PATH, config=cfg) as log:
        log(res["wakeup_us_median"], sample_size=res["trials"])
        log(res["per_trial_us"], sample_size=1)
    return res


if __name__ == "__main__":
    _m, _s, _u = get_seed()
    print(f"[bench_shm_spin_poll_wakeup] seed: {_m:.3f} {_u} (sigma {_s:.3f}) — first-principles "
          f"(cross-core cache-line snoop; ~0 vs the per-forward cycle)")
    register_self()
    print("[bench_shm_spin_poll_wakeup] registered. NOT running the live measurement (timing-sensitive); "
          "invoke run() pinned (taskset -c 0,1) and sole-workload.")
