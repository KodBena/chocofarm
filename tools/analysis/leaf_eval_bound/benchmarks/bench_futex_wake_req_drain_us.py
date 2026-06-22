"""
tools/analysis/leaf_eval_bound/benchmarks/bench_futex_wake_req_drain_us.py
===================================================================

LIVE benchmark for `futex_wake_req_drain_us` — the per-forward cost (us) of COPYING the B
request rows OUT of the shared-memory request ring into a contiguous `(B, in_dim)` input
buffer, for the FUTEX-WAKE transport. This is the cost the design AVOIDS by handing a
zero-copy ring SPAN to the host->device staging ("drains rows straight out of the ring") — so
it is charged into `futex_wake_tau_io_us` ONLY IF that zero-copy elision is NOT realized (e.g.
the staging needs a contiguous, page-aligned, non-ring-wrapping buffer).

The ring layout is IDENTICAL to shm_spin_poll (the futex_wake transport differs ONLY in its
wakeup mechanism — it parks the serve core on FUTEX_WAIT instead of busy-spinning a counter),
so the request-drain copy physics is the same; it is registered under the futex slug so the
UNIQUE-name constraint never collides across the fan-out (each variant owns its own prefixed
quantities — ADR-0012 one-home).

WHY A SEPARATE QUANTITY. The dominant uncertainty in the futex tau_io bound is precisely "does
the request drain copy collapse to zero?". Splitting that copy into its OWN measurable quantity
lets the Neyman allocator rank it as an independent question: the tau_io seed uses the zero-copy
arm (the design intent), and this quantity quantifies the PENALTY of the copy-both fallback
(~31us at B_op=256). The model's pessimistic arm (`copy_both_contrast`) adds this term; the
design-faithful headline omits it.

WHAT run() MEASURES (1:1). Time a single `contiguous[:B] = req_ring[:B]` memcpy of B rows of
in_dim f32 out of a shared-memory ring into a freshly-allocated contiguous buffer, over
`cycles`, at the operating-point B. NO JAX, NO syscall. Returns the per-forward copy us.

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

import estimate as _est  # noqa: E402  — the harmonized Estimate contract (measure() returns one — §6 Phase 4)
from estimators import median_estimate  # noqa: E402
from pools import window_pool  # noqa: E402
from harness import logged_run  # noqa: E402

NAME = "futex_wake_req_drain_us"
MODULE_PATH = "benchmarks.bench_futex_wake_req_drain_us"
_DESC = ("FUTEX-WAKE per-forward request-drain copy cost (us): memcpy B request rows out of the request ring "
         "into a contiguous (B,in_dim) input. The cost the zero-copy ring-span drain AVOIDS — charged into "
         "futex_wake_tau_io_us ONLY if zero-copy is not realized. Same ring as shm_spin_poll; the copy-both "
         "penalty.")

_IN_DIM = 241
_REQ_ROW_B = _IN_DIM * 4               # 964 B/row
_B_OP_SEED = 256
_MEMCPY_BW_BYTES_PER_NS = 8.0          # CONSERVATIVE single-thread sequential memcpy (matches the tau_io bench)


def get_seed() -> tuple[float, float, str]:
    """The v1 SEED (DISTRUST fallback) — first-principles: B_op request rows memcpy'd out of the ring at a
    conservative 8 B/ns. At B_op=256: 256 * 964 B / 8 B/ns ~= 30.85 us. sigma is the bandwidth spread (a
    faster L2-resident copy at ~16 B/ns halves it). Returns (mean, sigma, unit)."""
    mean = _B_OP_SEED * _REQ_ROW_B / _MEMCPY_BW_BYTES_PER_NS / 1000.0
    sigma = 0.5 * mean   # bandwidth uncertainty (8 B/ns conservative vs ~16 B/ns L2-resident)
    return (mean, sigma, "us")


def register_self() -> Any:
    from harness import register_quantity
    return register_quantity(NAME, quantity="serve_req_drain_copy_futex_wake", units="us",
                             description=_DESC, module_path=MODULE_PATH)


def _measure_raw(rows: int = 256, cycles: int = 5000) -> dict[str, Any]:
    """The raw-pool PROVENANCE producer (the §6 Phase-4 internal helper): measure the request-drain copy: time `contiguous[:rows] = req_ring[:rows]` (a memcpy of `rows` rows of
    in_dim f32 out of a shared-memory ring) over `cycles`. NO JAX, NO syscall. Returns
    {'req_drain_us_median', 'per_cycle_us', 'rows'}. Imports numpy + shared_memory lazily.
    `measure()` wraps the per-cycle pool into a median `Estimate`; `run()` uses it for BOTH the Estimate and the raw provenance rows (ONE measurement, two consumers — P1)."""
    import numpy as np
    from multiprocessing import shared_memory

    req_bytes = rows * _IN_DIM * 4
    shm_req = shared_memory.SharedMemory(create=True, size=req_bytes)
    try:
        req_ring = np.ndarray((rows, _IN_DIM), dtype=np.float32, buffer=shm_req.buf)
        req_ring[:] = np.ones((rows, _IN_DIM), dtype=np.float32)
        contiguous = np.empty((rows, _IN_DIM), dtype=np.float32)
        for _ in range(min(200, cycles)):           # warm caches + page table
            contiguous[:] = req_ring
        def _one_cycle() -> float:
            """One charged request-drain memcpy (rows of in_dim f32 out of the ring) -> its us reading
            (the per-window measurement window_pool calls once per window)."""
            t0 = time.perf_counter_ns()
            contiguous[:rows] = req_ring[:rows]     # the charged request-drain memcpy
            return (time.perf_counter_ns() - t0) / 1000.0

        # window_pool owns the loop + the >= 2 floor (RCA fix #2): one reading per cycle, count == cycles.
        per_cycle_us = window_pool(_one_cycle, name=NAME, count=cycles)
        med = float(np.median(per_cycle_us))
        return {"req_drain_us_median": med, "per_cycle_us": per_cycle_us, "rows": rows}
    finally:
        shm_req.close()
        shm_req.unlink()


def _estimate_from_raw(res: dict[str, Any]) -> "_est.Estimate":
    """Build this bench's harmonized `Estimate` from a `_measure_raw()` dict — the SINGLE home of the
    Estimate construction (P1), called by BOTH `measure()` and `run()`. A k=1 median `QuantileLaw(p=0.5)`
    with a BOOTSTRAP median SE over the pool (§7.A — the order-statistic variance, NOT s²/n),
    family=EMPIRICAL, kind='median'."""
    return median_estimate(res["per_cycle_us"], name=NAME)   # bootstrap median SE over the per-cycle pool


def measure(rows: int = 256, cycles: int = 5000) -> "_est.Estimate":
    """Measure the request-drain copy and return its harmonized k=1 median `Estimate` (§6 Phase 4: `measure()`
    returns the `Estimate` the bench DECLARES — the driver/untrusted_drive consume it directly, no
    guessing which list is the pool). The raw pool is the bench's internal `_measure_raw()` provenance.
    TIMING-SENSITIVE — pin the process (taskset -c 0)."""
    return _estimate_from_raw(_measure_raw(rows=rows, cycles=cycles))


def run(rows: int = 256, cycles: int = 5000) -> dict[str, Any]:
    """Measure the request-drain copy and LOG it as a harmonized k=1 median Estimate (QuantileLaw p=0.5,
    bootstrap median SE, §6 Phase 3, §5.2 de-dup). TIMING-SENSITIVE — operator-invoked, pinned, never during
    the fan-out."""
    res = _measure_raw(rows=rows, cycles=cycles)  # ONE measurement (Estimate + provenance)
    est = _estimate_from_raw(res)  # the SAME Estimate measure() returns (P1)
    cfg = {"rows": rows, "cycles": cycles, "transport": "shm_ring_futex_wake",
           "kind": "request_drain_copy_fallback",
           "req_drain_us_median": res["req_drain_us_median"],
           "note": "the cost the zero-copy ring-span drain avoids; charged only in the copy-both arm"}
    with logged_run(NAME, quantity="serve_req_drain_copy_futex_wake", units="us", description=_DESC,
                    module_path=MODULE_PATH, config=cfg, estimate=est) as log:
        # PROVENANCE only (§5.2 de-dup): the headline median lives in estimate.theta_hat[0], not a sample row.
        log(res["per_cycle_us"], sample_size=1)
    return res


if __name__ == "__main__":
    _m, _s, _u = get_seed()
    print(f"[bench_futex_wake_req_drain_us] seed: {_m:.2f} {_u} (sigma {_s:.2f}) — first-principles "
          f"({_B_OP_SEED} req rows memcpy out of ring @ {_MEMCPY_BW_BYTES_PER_NS} B/ns; same ring as shm)")
    register_self()
    print("[bench_futex_wake_req_drain_us] registered. NOT running the live measurement (timing-sensitive); "
          "invoke run() pinned (taskset -c 0) and sole-workload.")
