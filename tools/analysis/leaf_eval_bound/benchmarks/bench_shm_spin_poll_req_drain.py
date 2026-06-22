"""
tools/analysis/leaf_eval_bound/benchmarks/bench_shm_spin_poll_req_drain.py
===================================================================

LIVE benchmark for `shm_spin_poll_req_drain_us` — the per-forward cost (us) of COPYING the B
request rows OUT of the shared-memory request ring into a contiguous `(B, in_dim)` input
buffer, for the SHM SPIN-POLL transport. This is the cost the design AVOIDS by handing a
zero-copy ring SPAN to the host->device staging ("drains rows straight out of the ring") —
so it is charged into `shm_spin_poll_tau_io_us` ONLY IF that zero-copy elision is NOT
realized (e.g. the staging needs a contiguous, page-aligned, non-ring-wrapping buffer).

WHY A SEPARATE QUANTITY. The dominant uncertainty in the shm tau_io bound is precisely
"does the request drain copy collapse to zero?". Splitting that copy into its OWN measurable
quantity lets the Neyman allocator rank it as an independent question: the tau_io seed uses
the zero-copy arm (the design intent), and this quantity quantifies the PENALTY of the
copy-both fallback (~31us at B_op=256). A model that wants the pessimistic bound adds this
term; the optimistic (design-faithful) bound omits it.

WHAT run() MEASURES (1:1). Time a single `contiguous[:B] = req_ring[:B]` memcpy of B rows of
in_dim f32 out of a shared-memory ring into a freshly-allocated contiguous buffer, over
`cycles`, at the operating-point B. NO JAX, NO syscall. Returns the per-forward copy us.

TIMING-SENSITIVE — DO NOT run() during the parallel fan-out. Pin: `taskset -c 0`.

Public Domain (The Unlicense).
"""
from __future__ import annotations

import time
from typing import Any


from leaf_eval_bound.contract import estimate as _est  # noqa: E402  — the harmonized Estimate contract (measure() returns one — §6 Phase 4)
from leaf_eval_bound.benchmarks.estimators import median_estimate  # noqa: E402
from leaf_eval_bound.benchmarks.pools import window_pool  # noqa: E402
from leaf_eval_bound.benchmarks.scaffold import bench as _scaffold  # noqa: E402  — move 6 wiring

NAME = "shm_spin_poll_req_drain_us"
MODULE_PATH = "leaf_eval_bound.benchmarks.bench_shm_spin_poll_req_drain"
_DESC = ("SHM SPIN-POLL per-forward request-drain copy cost (us): memcpy B request rows out of the request "
         "ring into a contiguous (B,in_dim) input. The cost the zero-copy ring-span drain AVOIDS — charged "
         "into shm_spin_poll_tau_io_us ONLY if zero-copy is not realized. Quantifies the copy-both penalty.")

_IN_DIM = 241
_REQ_ROW_B = _IN_DIM * 4               # 964 B/row
_B_OP_SEED = 256
_MEMCPY_BW_BYTES_PER_NS = 8.0          # CONSERVATIVE single-thread sequential memcpy (matches the tau_io bench)


def get_seed() -> tuple[float, float, str]:
    """The v1 SEED (DISTRUST fallback) — first-principles: B_op request rows memcpy'd out of the ring at a
    conservative 8 B/ns. At B_op=256: 256 * 964 B / 8 B/ns ~= 30.85 us. sigma is the bandwidth spread
    (a faster L2-resident copy at ~16 B/ns halves it). Returns (mean, sigma, unit)."""
    mean = _B_OP_SEED * _REQ_ROW_B / _MEMCPY_BW_BYTES_PER_NS / 1000.0
    sigma = 0.5 * mean   # bandwidth uncertainty (8 B/ns conservative vs ~16 B/ns L2-resident)
    return (mean, sigma, "us")


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


# Move 6: the shared scaffold wires register_self / measure / run from the bench-specific parts above. The
# seed is a bare (mean, sigma, unit) tuple (no .unit), so the registered unit is passed explicitly (units="us").
_B = _scaffold(
    name=NAME, quantity="serve_req_drain_copy_shm_spin_poll", module_path=MODULE_PATH, description=_DESC,
    units="us", seed=get_seed, measure_raw=_measure_raw, estimate_from_raw=_estimate_from_raw,
    run_config=lambda res, **kw: {"rows": kw["rows"], "cycles": kw["cycles"], "transport": "shm_ring_spin_poll",
                                  "kind": "request_drain_copy_fallback",
                                  "req_drain_us_median": res["req_drain_us_median"],
                                  "note": "the cost the zero-copy ring-span drain avoids; charged only in the copy-both arm"},
    run_log=lambda res, log, **kw: log(res["per_cycle_us"], sample_size=1),
)
register_self, measure, run = _B.register_self, _B.measure, _B.run


if __name__ == "__main__":
    _m, _s, _u = get_seed()
    print(f"[bench_shm_spin_poll_req_drain] seed: {_m:.2f} {_u} (sigma {_s:.2f}) — first-principles "
          f"({_B_OP_SEED} req rows memcpy out of ring @ {_MEMCPY_BW_BYTES_PER_NS} B/ns)")
    register_self()
    print("[bench_shm_spin_poll_req_drain] registered. NOT running the live measurement (timing-sensitive); "
          "invoke run() pinned (taskset -c 0) and sole-workload.")
