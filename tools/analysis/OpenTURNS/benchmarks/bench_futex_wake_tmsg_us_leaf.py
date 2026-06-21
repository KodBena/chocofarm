"""
tools/analysis/OpenTURNS/benchmarks/bench_futex_wake_tmsg_us_leaf.py
===================================================================

LIVE benchmark for `futex_wake_tmsg_us_leaf` — the per-leaf-amortized MESSAGE cost (us/leaf)
for the FUTEX-WAKE transport: NOT a wire encode/decode (there is no frame envelope), but the
in-RING memcpy of one leaf's request row IN + one reply row OUT. This is the TRANSPORT-stage
term (request/reply CAPACITY), which is NON-BINDING by a wide margin (the binding stage is the
serialized serve), so it is reported but ranks LAST for the Neyman allocator.

The per-leaf ring traffic is IDENTICAL to shm_spin_poll (the futex_wake transport differs ONLY
in its wakeup mechanism — it parks the serve core on FUTEX_WAIT instead of busy-spinning), so
this is the same bare-ring-copy physics, registered under the futex slug so the UNIQUE-name
constraint never collides across the fan-out (each variant owns its own prefixed quantities —
ADR-0012 one-home). The futex WAIT/WAKE handoff is NOT per-leaf (it is the empty->nonempty edge
wakeup, the separate futex_wake_wakeup_us term), so it does not enter this per-leaf cost.

WHAT run() MEASURES (1:1). Time a single leaf's ring traffic: copy one request row (in_dim f32)
into the request ring + copy one reply row ((1+n_actions) f32) out of the reply ring, over
`iters` — the per-leaf framing share with no envelope, no syscall. The SEED is the bare memcpy
of (req_row + rep_row) bytes at a conservative bandwidth (~0.15 us/leaf), far below the
per-forward budget, so transport never binds.

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

NAME = "futex_wake_tmsg_us_leaf"
MODULE_PATH = "benchmarks.bench_futex_wake_tmsg_us_leaf"
_DESC = ("FUTEX-WAKE per-leaf message cost (us/leaf): the in-ring memcpy of one request row in + one reply "
         "row out (no frame envelope, no syscall). Transport stage; NON-BINDING by a wide margin. Same ring "
         "copy as shm_spin_poll (the futex wakeup is the separate futex_wake_wakeup_us term, not per-leaf).")

_IN_DIM = 241
_N_ACTIONS = 65
_REQ_ROW_B = _IN_DIM * 4               # 964 B/row
_REP_ROW_B = (1 + _N_ACTIONS) * 4      # 264 B/row
_MEMCPY_BW_BYTES_PER_NS = 8.0          # CONSERVATIVE single-thread sequential memcpy (matches the tau_io bench)


def get_seed() -> tuple[float, float, str]:
    """The v1 SEED (DISTRUST fallback) — first-principles: one leaf's ring traffic is (req_row + rep_row)
    bytes memcpy'd at a conservative 8 B/ns: (964 + 264)/8/1000 ~= 0.15 us/leaf. sigma 0.08us (bandwidth
    spread). Non-binding by a wide margin. Returns (mean, sigma, unit)."""
    mean = (_REQ_ROW_B + _REP_ROW_B) / _MEMCPY_BW_BYTES_PER_NS / 1000.0
    return (mean, 0.08, "us")


def register_self() -> Any:
    from bench_common import register_quantity
    return register_quantity(NAME, quantity="transport_msg_cost_per_leaf_futex_wake", units="us",
                             description=_DESC, module_path=MODULE_PATH)


def measure(iters: int = 200000) -> dict[str, Any]:
    """Measure futex_wake_tmsg_us_leaf: time one leaf's ring traffic — copy one request row into the request
    ring + one reply row out of the reply ring — over `iters`. NO envelope, NO syscall. Returns
    {'tmsg_us_leaf', 'iters'}. Imports numpy + shared_memory lazily."""
    import numpy as np
    from multiprocessing import shared_memory

    shm_req = shared_memory.SharedMemory(create=True, size=_REQ_ROW_B)
    shm_rep = shared_memory.SharedMemory(create=True, size=_REP_ROW_B)
    try:
        req_slot = np.ndarray((_IN_DIM,), dtype=np.float32, buffer=shm_req.buf)
        rep_slot = np.ndarray((1 + _N_ACTIONS,), dtype=np.float32, buffer=shm_rep.buf)
        one_req = np.ones((_IN_DIM,), dtype=np.float32)
        out_rep = np.empty((1 + _N_ACTIONS,), dtype=np.float32)
        for _ in range(min(2000, iters)):           # warm
            req_slot[:] = one_req
            out_rep[:] = rep_slot
        t0 = time.perf_counter_ns()
        for _ in range(iters):
            req_slot[:] = one_req                    # producer writes one request row into the ring
            out_rep[:] = rep_slot                    # consumer reads one reply row out of the ring
        per_leaf_us = (time.perf_counter_ns() - t0) / 1000.0 / iters
        return {"tmsg_us_leaf": per_leaf_us, "iters": iters}
    finally:
        for shm in (shm_req, shm_rep):
            shm.close()
            shm.unlink()


def run(iters: int = 200000) -> dict[str, Any]:
    """Logs a harmonized k=1 Fixed Estimate (§6 Phase 3) recovering the declared spread un-divided, alongside the live measurement. TIMING-SENSITIVE — operator-invoked, pinned, never during
    the fan-out."""
    res = measure(iters=iters)
    _sm, _ss, _ = get_seed()
    est = pin_estimate(_sm, _ss, name=NAME)
    cfg = {"iters": iters, "transport": "shm_ring_futex_wake", "codec": "bare_ring_memcpy",
           "note": "in-ring memcpy of one request row in + one reply row out; no envelope, no syscall"}
    with logged_run(NAME, quantity="transport_msg_cost_per_leaf_futex_wake", units="us", description=_DESC,
                    module_path=MODULE_PATH, config=cfg, estimate=est) as log:
        log(res["tmsg_us_leaf"], sample_size=iters)
    return res


if __name__ == "__main__":
    _m, _s, _u = get_seed()
    print(f"[bench_futex_wake_tmsg_us_leaf] seed: {_m:.3f} {_u} (sigma {_s:.3f}) — first-principles "
          f"(bare ring memcpy of one req row in + one reply row out; non-binding; same ring as shm)")
    register_self()
    print("[bench_futex_wake_tmsg_us_leaf] registered. NOT running the live measurement (timing-sensitive); "
          "invoke run() pinned (taskset -c 0) and sole-workload.")
