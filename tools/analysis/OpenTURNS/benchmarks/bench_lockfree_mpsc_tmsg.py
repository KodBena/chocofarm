"""
tools/analysis/OpenTURNS/benchmarks/bench_lockfree_mpsc_tmsg.py
==============================================================

LIVE benchmark for `lockfree_mpsc_tmsg_us_leaf` — the per-leaf-amortized MESSAGE cost (us/leaf)
for the LOCK-FREE MPSC transport: NOT a wire encode/decode (there is no frame envelope, no
corr-id, no `send_multipart`), but the ENQUEUE of one leaf's request node (a CAS that publishes
the slot + writes the feature row) plus the consumer-side read of its reply slot. This is the
TRANSPORT-stage term (request/reply CAPACITY), which is NON-BINDING by a wide margin (the binding
stage is the serialized serve), so it is reported but ranks LAST for the Neyman allocator — the
MPSC variant of the baseline `tmsg_us_leaf`, with the ZMQ memcpy-codec framing replaced by a
CAS-enqueue + a slot write/read.

WHAT run() MEASURES (1:1). Time a single leaf's queue traffic: an atomic tail-CAS (the enqueue
publish) + write one request row (in_dim f32) into its reserved slot + read one reply row
((1+n_actions) f32) out of the reply slab, over `iters` — the per-leaf framing share with no
envelope, no syscall, no codec. The SEED is the bare (req_row + rep_row) memcpy at a conservative
bandwidth + one CAS (~0.18 us/leaf), far below the per-forward budget, so transport never binds.

WHY IT IS NON-BINDING (and still reported). One coalesced FORWARD serves B leaves, so the
transport per-leaf cost is amortized B-fold against the binding serve cycle; even a deliberate
over-charge leaves the transport CAPACITY (1/(LPD*tmsg)) far above the serve ceiling. Reporting
it confirms the wire request/reply CAPACITY (~2000+ dps) never binds — exactly the brief's
"wire request/reply CAPACITY is non-binding" — so it ranks LAST for the allocator.

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

from bench_common import logged_run, pin_estimate  # noqa: E402

NAME = "lockfree_mpsc_tmsg_us_leaf"
MODULE_PATH = "benchmarks.bench_lockfree_mpsc_tmsg"
_DESC = ("LOCK-FREE MPSC per-leaf message cost (us/leaf): a tail-CAS enqueue + write one request row into "
         "the reserved slot + read one reply row out (no frame envelope, no corr-id, no syscall, no codec). "
         "Transport stage; NON-BINDING by a wide margin. The MPSC variant of tmsg_us_leaf (CAS-enqueue + "
         "slot write/read, no ZMQ multipart codec).")

_IN_DIM = 241
_N_ACTIONS = 65
_REQ_ROW_B = _IN_DIM * 4               # 964 B/row
_REP_ROW_B = (1 + _N_ACTIONS) * 4      # 264 B/row
_MEMCPY_BW_BYTES_PER_NS = 8.0          # CONSERVATIVE single-thread sequential memcpy (matches the tau_io bench)
_CAS_NS = 30.0                         # one uncontended enqueue CAS (cache-hot)


def get_seed() -> tuple[float, float, str]:
    """The v1 SEED (DISTRUST fallback) — first-principles: one leaf's queue traffic is (req_row + rep_row)
    bytes memcpy'd at a conservative 8 B/ns plus one enqueue CAS: (964 + 264)/8/1000 + 30/1000 ~= 0.18
    us/leaf. sigma 0.08us (bandwidth + CAS-contention spread). Non-binding by a wide margin. Returns
    (mean, sigma, unit)."""
    mean = (_REQ_ROW_B + _REP_ROW_B) / _MEMCPY_BW_BYTES_PER_NS / 1000.0 + _CAS_NS / 1000.0
    return (mean, 0.08, "us")


def register_self() -> Any:
    from bench_common import register_quantity
    return register_quantity(NAME, quantity="transport_msg_cost_per_leaf_lockfree_mpsc", units="us",
                             description=_DESC, module_path=MODULE_PATH)


def measure(iters: int = 200000) -> dict[str, Any]:
    """Measure lockfree_mpsc_tmsg_us_leaf: time one leaf's queue traffic — an atomic tail bump (the enqueue
    publish, a numpy int64 increment standing for the CAS) + write one request row into its slot + read one
    reply row out of the reply slab — over `iters`. NO envelope, NO syscall, NO codec. Returns
    {'tmsg_us_leaf', 'iters'}. Imports numpy + shared_memory lazily."""
    import numpy as np
    from multiprocessing import shared_memory

    shm_req = shared_memory.SharedMemory(create=True, size=_REQ_ROW_B)
    shm_rep = shared_memory.SharedMemory(create=True, size=_REP_ROW_B)
    shm_ctr = shared_memory.SharedMemory(create=True, size=8)
    try:
        req_slot = np.ndarray((_IN_DIM,), dtype=np.float32, buffer=shm_req.buf)
        rep_slot = np.ndarray((1 + _N_ACTIONS,), dtype=np.float32, buffer=shm_rep.buf)
        ctr = np.ndarray((1,), dtype=np.int64, buffer=shm_ctr.buf)     # the MPSC tail (the enqueue CAS target)
        one_req = np.ones((_IN_DIM,), dtype=np.float32)
        out_rep = np.empty((1 + _N_ACTIONS,), dtype=np.float32)
        ctr[0] = 0
        for _ in range(min(2000, iters)):           # warm
            req_slot[:] = one_req
            ctr[0] += 1
            out_rep[:] = rep_slot
        t0 = time.perf_counter_ns()
        for _ in range(iters):
            req_slot[:] = one_req                    # producer writes one request row into its reserved slot
            ctr[0] += 1                              # the enqueue publish (a tail bump; the CAS in the real queue)
            out_rep[:] = rep_slot                    # consumer reads one reply row out of the reply slab
        per_leaf_us = (time.perf_counter_ns() - t0) / 1000.0 / iters
        return {"tmsg_us_leaf": per_leaf_us, "iters": iters}
    finally:
        for shm in (shm_req, shm_rep, shm_ctr):
            shm.close()
            shm.unlink()


def run(iters: int = 200000) -> dict[str, Any]:
    """Logs a harmonized k=1 Fixed Estimate (§6 Phase 3) recovering the declared spread un-divided, alongside the live measurement. TIMING-SENSITIVE — operator-invoked, pinned, never
    during the fan-out."""
    res = measure(iters=iters)
    _sm, _ss, _ = get_seed()
    est = pin_estimate(_sm, _ss, name=NAME)
    cfg = {"iters": iters, "transport": "lockfree_mpsc_queue", "codec": "cas_enqueue_slot_write",
           "note": "tail-CAS enqueue + slot write of one request row + reply-slot read; no envelope, no syscall"}
    with logged_run(NAME, quantity="transport_msg_cost_per_leaf_lockfree_mpsc", units="us", description=_DESC,
                    module_path=MODULE_PATH, config=cfg, estimate=est) as log:
        log(res["tmsg_us_leaf"], sample_size=iters)
    return res


if __name__ == "__main__":
    _m, _s, _u = get_seed()
    print(f"[bench_lockfree_mpsc_tmsg] seed: {_m:.3f} {_u} (sigma {_s:.3f}) — first-principles "
          f"(bare slot memcpy of one req row + one reply row + one enqueue CAS; non-binding)")
    register_self()
    print("[bench_lockfree_mpsc_tmsg] registered. NOT running the live measurement (timing-sensitive); "
          "invoke run() pinned (taskset -c 0) and sole-workload.")
