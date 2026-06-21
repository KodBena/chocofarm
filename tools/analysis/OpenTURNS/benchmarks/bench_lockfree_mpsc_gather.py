"""
tools/analysis/OpenTURNS/benchmarks/bench_lockfree_mpsc_gather.py
================================================================

LIVE benchmark for `lockfree_mpsc_gather_us` — the per-forward cost (us) of GATHERING the B
request rows out of the lock-free MPSC queue's SCATTERED node payloads into a contiguous
`(B, in_dim)` input buffer, for the LOCK-FREE MPSC transport. Because N producer cores enqueue
nodes INDEPENDENTLY (each CAS-reserves its slot; the enqueues interleave), the request rows are
NOT a single contiguous span — so a batched forward needs a GATHER. This is the intrinsic-to-
coalescing cost the existing wire path already pays (cpp/include/chocofarm/wire_leaf_pool.hpp:
`submit_batch` "the STRICT GATHER-BARRIER: gather B parked rows into ONE encode_request").

WHY A SEPARATE QUANTITY (the dominant uncertainty in the MPSC bound). The MPSC headline tau_io
CHARGES this gather (the honest default — scattered nodes). The one way to ELIDE it is a
staging path that consumes a SCATTER/GATHER iovec list (B per-node views) instead of a
materialized contiguous buffer — handing the device-transfer a gather descriptor. Whether the
host->device staging can do that (vs needing a contiguous, page-aligned, non-fragmented input)
is the dominant measurable question. Splitting the gather into its OWN quantity lets the Neyman
allocator rank "can the gather be elided?" as an independent question: the headline tau_io
INCLUDES this term, and the OPTIMISTIC (gather-elided) bound SUBTRACTS it.

CONTRAST WITH shm_spin_poll_req_drain_us (the honest asymmetry). The shm ring is CONTIGUOUS, so
its req-drain copy is the PESSIMISTIC arm (the headline elides it via a zero-copy span). The
MPSC nodes are SCATTERED, so the gather is the DEFAULT arm (the headline charges it; elision is
the optimistic arm). Same physical memcpy magnitude (~31us at B_op=256), opposite default sign —
because gather-elision is less plausible for an MPSC queue than span-elision is for a ring.

WHAT run() MEASURES (1:1). Time a single GATHER `contiguous[:] = slab[src_idx]` of B rows of
in_dim f32 out of a backing slab at SCATTERED row indices (the MPSC enqueue order) into a
freshly-allocated contiguous buffer, over `cycles`, at the operating-point B. A scattered
fancy-index gather is SLOWER than a flat memcpy (non-sequential reads); this measures that real
gather cost. NO JAX, NO syscall. Returns the per-forward gather us.

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
from bench_common import logged_run, median_estimate, window_pool  # noqa: E402

NAME = "lockfree_mpsc_gather_us"
MODULE_PATH = "benchmarks.bench_lockfree_mpsc_gather"
_DESC = ("LOCK-FREE MPSC per-forward request-GATHER cost (us): gather B request rows out of the MPSC "
         "queue's SCATTERED node payloads (independent multi-producer enqueues are not contiguous) into a "
         "contiguous (B,in_dim) input. Intrinsic to coalescing (the existing WireLeafPool submit_batch pays "
         "it); CHARGED in the headline tau_io, ELIDED only if the staging consumes a scatter/gather iovec "
         "list. The dominant uncertainty in the MPSC bound (rank: is the gather elidable?).")

_IN_DIM = 241
_REQ_ROW_B = _IN_DIM * 4               # 964 B/row
_B_OP_SEED = 256
_ROWS_PER_NODE = 32                    # one enqueued node carries rows_per_node rows; T = B_op/rows_per_node nodes
_MEMCPY_BW_BYTES_PER_NS = 8.0          # CONSERVATIVE single-thread sequential memcpy (matches the tau_io / shm benches)


def get_seed() -> tuple[float, float, str]:
    """The v1 SEED (DISTRUST fallback) — first-principles: B_op request rows gathered out of the slab. A
    scattered gather is bounded BELOW by a contiguous memcpy of the same bytes (B_op * 964 B / 8 B/ns ~=
    30.85 us at B_op=256) and is in practice SLOWER (non-sequential reads cost up to ~2-3x). The seed uses
    the contiguous-memcpy LOWER bound on the gather (so it is a conservative lower bound on the *penalty*,
    keeping the headline tau_io from overstating it); sigma is wide and skewed up (a fully scattered
    fancy-index gather at ~BW/2.5 is ~77us). Returns (mean, sigma, unit)."""
    mean = _B_OP_SEED * _REQ_ROW_B / _MEMCPY_BW_BYTES_PER_NS / 1000.0
    # sigma up to ~the scattered-gather penalty (a non-sequential gather at ~BW/2.5 ~= 77us, ~1.5x the mean).
    sigma = 1.5 * mean
    return (mean, sigma, "us")


def register_self() -> Any:
    from bench_common import register_quantity
    return register_quantity(NAME, quantity="serve_req_gather_lockfree_mpsc", units="us",
                             description=_DESC, module_path=MODULE_PATH)


def _measure_raw(rows: int = 256, rows_per_node: int = 32, cycles: int = 5000) -> dict[str, Any]:
    """The raw-pool PROVENANCE producer (the §6 Phase-4 internal helper): measure the request-gather: time `contiguous[:] = slab[src_idx]` (a gather of `rows` rows of in_dim
    f32 out of a backing slab at SCATTERED node-ordered indices) into a freshly-allocated contiguous buffer
    over `cycles`. The scattered indices model the MPSC enqueue interleave (a node's rows are a strided
    block at a permuted offset). NO JAX, NO syscall. Returns {'gather_us_median', 'per_cycle_us', 'rows'}.
    Imports numpy + shared_memory lazily. `measure()` wraps the per-cycle pool into a median `Estimate`; `run()` uses it for BOTH the Estimate and the raw provenance rows (ONE measurement, two consumers — P1)."""
    import numpy as np
    from multiprocessing import shared_memory

    n_nodes = rows // rows_per_node
    slab_rows = rows * 4                                   # headroom -> scattered, non-contiguous node offsets
    slab_bytes = slab_rows * _IN_DIM * 4
    shm_slab = shared_memory.SharedMemory(create=True, size=slab_bytes)
    try:
        slab = np.ndarray((slab_rows, _IN_DIM), dtype=np.float32, buffer=shm_slab.buf)
        slab[:] = np.ones((slab_rows, _IN_DIM), dtype=np.float32)
        rng = np.random.default_rng(0)
        node_starts = (rng.permutation(slab_rows // rows_per_node)[:n_nodes]) * rows_per_node
        src_idx = np.concatenate([np.arange(s, s + rows_per_node) for s in node_starts])  # scattered row indices
        contiguous = np.empty((rows, _IN_DIM), dtype=np.float32)
        for _ in range(min(200, cycles)):           # warm caches + page table
            contiguous[:] = slab[src_idx]
        def _one_cycle() -> float:
            """One charged request gather (scattered node-ordered rows -> contiguous) -> its us reading
            (the per-window measurement window_pool calls once per window)."""
            t0 = time.perf_counter_ns()
            contiguous[:] = slab[src_idx]            # the charged request GATHER (scattered -> contiguous)
            return (time.perf_counter_ns() - t0) / 1000.0

        # window_pool owns the loop + the >= 2 floor (RCA fix #2): one reading per cycle, count == cycles.
        per_cycle_us = window_pool(_one_cycle, name=NAME, count=cycles)
        med = float(np.median(per_cycle_us))
        return {"gather_us_median": med, "per_cycle_us": per_cycle_us, "rows": rows}
    finally:
        shm_slab.close()
        shm_slab.unlink()


def _estimate_from_raw(res: dict[str, Any]) -> "_est.Estimate":
    """Build this bench's harmonized `Estimate` from a `_measure_raw()` dict — the SINGLE home of the
    Estimate construction (P1), called by BOTH `measure()` and `run()`. A k=1 median `QuantileLaw(p=0.5)`
    with a BOOTSTRAP median SE over the pool (§7.A — the order-statistic variance, NOT s²/n),
    family=EMPIRICAL, kind='median'."""
    return median_estimate(res["per_cycle_us"], name=NAME)   # bootstrap median SE over the per-cycle pool


def measure(rows: int = 256, rows_per_node: int = 32, cycles: int = 5000) -> "_est.Estimate":
    """Measure the request-gather and return its harmonized k=1 median `Estimate` (§6 Phase 4: `measure()`
    returns the `Estimate` the bench DECLARES — the driver/untrusted_drive consume it directly, no
    guessing which list is the pool). The raw pool is the bench's internal `_measure_raw()` provenance.
    TIMING-SENSITIVE — pin the process (taskset -c 0)."""
    return _estimate_from_raw(_measure_raw(rows=rows, rows_per_node=rows_per_node, cycles=cycles))


def run(rows: int = 256, rows_per_node: int = 32, cycles: int = 5000) -> dict[str, Any]:
    """Measure the request-gather and LOG it as a harmonized k=1 median Estimate (QuantileLaw p=0.5, bootstrap
    median SE, §6 Phase 3, §5.2 de-dup). TIMING-SENSITIVE — operator-invoked, pinned, never during the fan-out."""
    res = _measure_raw(rows=rows, rows_per_node=rows_per_node, cycles=cycles)  # ONE measurement (Estimate + provenance)
    est = _estimate_from_raw(res)  # the SAME Estimate measure() returns (P1)
    cfg = {"rows": rows, "rows_per_node": rows_per_node, "cycles": cycles,
           "transport": "lockfree_mpsc_queue", "kind": "request_gather_scattered",
           "gather_us_median": res["gather_us_median"],
           "note": "gather B scattered node rows -> contiguous; charged in the headline tau_io, "
                   "elided only by a scatter/gather-aware staging path"}
    with logged_run(NAME, quantity="serve_req_gather_lockfree_mpsc", units="us", description=_DESC,
                    module_path=MODULE_PATH, config=cfg, estimate=est) as log:
        # PROVENANCE only (§5.2 de-dup): the headline median lives in estimate.theta_hat[0], not a sample row.
        log(res["per_cycle_us"], sample_size=1)
    return res


if __name__ == "__main__":
    _m, _s, _u = get_seed()
    print(f"[bench_lockfree_mpsc_gather] seed: {_m:.2f} {_u} (sigma {_s:.2f}) — first-principles "
          f"({_B_OP_SEED} req rows gathered out of slab @ {_MEMCPY_BW_BYTES_PER_NS} B/ns lower bound; scattered is slower)")
    register_self()
    print("[bench_lockfree_mpsc_gather] registered. NOT running the live measurement (timing-sensitive); "
          "invoke run() pinned (taskset -c 0) and sole-workload. This is the MPSC variant's dominant uncertainty.")
