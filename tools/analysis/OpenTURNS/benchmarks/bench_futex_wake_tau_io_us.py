"""
tools/analysis/OpenTURNS/benchmarks/bench_futex_wake_tau_io_us.py
================================================================

LIVE benchmark for `futex_wake_tau_io_us` — the SERVER-side per-forward serial TRANSPORT cost
(us) for the FUTEX-WAKE transport: the serve core drains the queued request rows out of a
shared-memory RING and writes the forward's replies into a reply ring. The RING DRAIN is
IDENTICAL to the shm_spin_poll transport — same shared-memory ring, same ZERO-COPY request
span handed to the staging, same reply-ring memcpy + counter bookkeeping. The ONLY difference
between futex_wake and shm_spin_poll is the WAKEUP mechanism (futex_wake parks the serve core
on `FUTEX_WAIT` when the ring is empty and is woken by a producer `FUTEX_WAKE`, instead of
busy-spinning a counter), and that wakeup cost is a SEPARATE quantity (`futex_wake_wakeup_us`),
NOT part of this drain term. So this `tau_io` is the SAME ring-drain physics as
`shm_spin_poll_tau_io_us`, registered under the futex slug so the UNIQUE-name constraint never
collides across the fan-out (each transport variant owns its own prefixed quantities — ADR-0012
one-home: the futex model reads `futex_wake_tau_io_us`, never the shm name).

THE FIRST-PRINCIPLES tau_io RESIDUAL (the term this bench measures), per forward of B rows
assembled from T producer messages:
      tau_io ~= reply-ring memcpy   (B * rep_row_B / bw)
              + per-message counter bookkeeping (T * c_idx — advance head, slot->producer)
              + drain wakeup-poll    (~0; the futex wakeup is the SEPARATE futex_wake_wakeup_us
                                       term, paid on the empty->nonempty edge, not in the drain)
  with the request drain ZERO-COPY (a ring span handed to the staging) — the design intent. The
SEED is computed from that decomposition at the operating point B_op=256 (a conservative single-
thread memcpy bandwidth so the bound is a LOWER bound on throughput / an UPPER bound on tau_io).

WHAT run() MEASURES (1:1 with the model input, NO JAX forward). A sole-workload microbench of
the futex serve loop's I/O ONLY: a producer deposits T coalesced request frames of `rows_per_msg`
rows ROW-MAJOR into a shared-memory request ring + bumps the tail counter; the server drains the
B rows ZERO-COPY (a numpy view over the ring slice), then memcpy's a B-row reply block into a
reply ring + advances the counter. The per-forward tau_io is that I/O time over one drain cycle.
run() ALSO records the copy-both arm (charging the request drain as a real memcpy) so the
operator sees how much zero-copy buys. The futex WAIT/WAKE handoff is NOT in this measured path
(it is the empty-edge wakeup, measured by bench_futex_wake_wakeup_us); at saturation the ring is
nonempty when the drain starts, so the drain pays ~0 wakeup, exactly as measured here.

TIMING-SENSITIVE — DO NOT run() during the parallel fan-out (co-scheduling corrupts the memcpy-
bandwidth timing). Pin: `taskset -c 0`.

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

from bench_common import logged_run  # noqa: E402

NAME = "futex_wake_tau_io_us"
MODULE_PATH = "benchmarks.bench_futex_wake_tau_io_us"
_DESC = ("FUTEX-WAKE server per-forward serial transport cost (us): drain request rows ZERO-COPY out of a "
         "shared-memory ring, memcpy the B reply rows into a reply ring + advance the counter. The transport-"
         "design term in the binding serve cycle; SAME ring drain as shm_spin_poll (the futex wakeup is the "
         "SEPARATE futex_wake_wakeup_us term, not the drain). Registered under the futex slug.")

# Production geometry (matches bench_tau_io / bench_t_row): the request feature width and the reply width.
# A full bucket B_op rows is split across T producer messages of rows_per_msg each.
_IN_DIM = 241
_N_ACTIONS = 65
_REQ_ROW_B = _IN_DIM * 4               # 964 B/row (241 f32 features)
_REP_ROW_B = (1 + _N_ACTIONS) * 4      # 264 B/row (value + n_actions logits)

# --- seed parameters (first-principles; the SEED arithmetic homes here) ---------------------
_B_OP_SEED = 256          # the operating-point full bucket the seed is evaluated at (leaf_eval_grounding B_op)
_ROWS_PER_MSG = 32        # producer coalescing degree (bench_tau_io geometry); T = B_op/rows_per_msg
_MEMCPY_BW_BYTES_PER_NS = 8.0    # CONSERVATIVE single-thread sequential memcpy (slow -> larger tau_io -> LOWER throughput bound)
_COUNTER_NS = 40.0        # per-message counter bookkeeping (atomic head advance + slot->producer map write)


def get_seed() -> tuple[float, float, str]:
    """The v1 SEED (DISTRUST fallback) for futex_wake_tau_io — a FIRST-PRINCIPLES estimate (no v1
    measurement exists; this is a NEW transport quantity, sharing the shm ring-drain physics). Decomposition
    at the operating point B_op=256, T=B_op/32=8 producer messages, ZERO-COPY request drain (the design
    intent):

        tau_io ~= reply-ring memcpy  (B_op * 264 B / 8 B/ns)            ~= 8.45 us
                + counter bookkeeping (T * 40 ns)                       ~= 0.32 us
                + drain wakeup-poll                                      ~  0    us  (the futex wakeup is separate)
                = ~8.8 us

    A CONSERVATIVE memcpy bandwidth (8 B/ns) makes the seed an UPPER bound on tau_io (a LOWER bound on
    throughput). sigma is wide (the bandwidth + the realized zero-copy fraction are both unmeasured): the
    request-drain copy, if NOT elided, adds ~31us (see futex_wake_req_drain_us), so the spread between the
    zero-copy and copy-both arms is the dominant uncertainty. Returns (mean, sigma, unit)."""
    reply_mc_us = _B_OP_SEED * _REP_ROW_B / _MEMCPY_BW_BYTES_PER_NS / 1000.0
    counter_us = (_B_OP_SEED // _ROWS_PER_MSG) * _COUNTER_NS / 1000.0
    mean = reply_mc_us + counter_us
    # sigma: ~half the request-drain copy (the zero-copy-vs-copy ambiguity dominates the spread).
    req_drain_us = _B_OP_SEED * _REQ_ROW_B / _MEMCPY_BW_BYTES_PER_NS / 1000.0
    sigma = 0.5 * req_drain_us
    return (mean, sigma, "us")


def register_self() -> Any:
    from bench_common import register_quantity
    return register_quantity(NAME, quantity="serve_transport_io_cost_futex_wake", units="us",
                             description=_DESC, module_path=MODULE_PATH)


def measure(n_msgs: int = 8, rows_per_msg: int = 32, cycles: int = 5000) -> dict[str, Any]:
    """Measure futex_wake tau_io: time ONE drain+reply cycle over `n_msgs` coalesced producer messages of
    `rows_per_msg` rows each (forward sees n_msgs*rows_per_msg rows). NO JAX forward — the forward is
    iota+t_row*B (separate). Uses a shared-memory RING (a numpy-backed `multiprocessing.shared_memory`
    buffer) + a tail counter; at saturation the ring is nonempty when the drain starts (so NO futex wait in
    this measured path — the futex wakeup is the separate futex_wake_wakeup_us term). Drains the request rows
    ZERO-COPY (a numpy view over the ring slice) and memcpy's the reply block into a reply ring. Returns
    {'tau_io_us_median', 'tau_io_copyboth_us_median', 'per_cycle_us', ...}. Imports numpy + shared_memory
    lazily. Pin the process (taskset -c 0)."""
    import numpy as np
    from multiprocessing import shared_memory

    B = n_msgs * rows_per_msg
    req_bytes = B * _IN_DIM * 4
    rep_bytes = B * (1 + _N_ACTIONS) * 4
    shm_req = shared_memory.SharedMemory(create=True, size=req_bytes)
    shm_rep = shared_memory.SharedMemory(create=True, size=rep_bytes)
    shm_ctr = shared_memory.SharedMemory(create=True, size=8)
    try:
        req_ring = np.ndarray((B, _IN_DIM), dtype=np.float32, buffer=shm_req.buf)
        rep_ring = np.ndarray((B, 1 + _N_ACTIONS), dtype=np.float32, buffer=shm_rep.buf)
        ctr = np.ndarray((1,), dtype=np.int64, buffer=shm_ctr.buf)
        # The forward's output the server copies into the reply ring (a (B, 1+n_actions) block — the shape
        # run_microbatch returns). Pre-allocated; its CONTENT is irrelevant, the memcpy COST is what the
        # per-forward reply scatter costs.
        fwd_out = np.zeros((B, 1 + _N_ACTIONS), dtype=np.float32)
        producer_rows = np.ones((B, _IN_DIM), dtype=np.float32)   # what the producers would deposit

        per_cycle_us: list[float] = []
        per_cycle_copyboth_us: list[float] = []
        contiguous_in = np.empty((B, _IN_DIM), dtype=np.float32)  # the copy-both arm's gather target

        # Warm the caches + the page table.
        for _ in range(min(200, cycles)):
            req_ring[:] = producer_rows
            ctr[0] = n_msgs
            _ = req_ring[: min(int(ctr[0]) * rows_per_msg, B)]
            rep_ring[:] = fwd_out
            ctr[0] = 0

        for _ in range(cycles):
            # PRODUCER side: deposit n_msgs frames row-major into the ring + bump the tail counter by n_msgs.
            req_ring[:] = producer_rows         # the producers' writes (off the server's critical path; the
                                                #   server's tau_io is the DRAIN+REPLY, timed below)
            ctr[0] = n_msgs                     # publish: tail counter says n_msgs messages are ready (at
                                                #   saturation the ring is ALREADY nonempty -> no futex wait)

            t0 = time.perf_counter_ns()
            # SERVER side (the tau_io critical section): the ring is nonempty (saturation), so drain directly.
            drained_rows = min(int(ctr[0]) * rows_per_msg, B)
            # DRAIN: ZERO-COPY — a numpy VIEW over the ring's filled rows handed to the (notional) staging.
            req_view = req_ring[:drained_rows]  # noqa: F841 — a view, no copy (the design's zero-copy drain)
            # REPLY: memcpy the forward's B reply rows into the reply ring (the only mandatory copy).
            rep_ring[:drained_rows] = fwd_out[:drained_rows]
            # per-message counter bookkeeping (advance head; reset for the next cycle).
            ctr[0] = 0
            per_cycle_us.append((time.perf_counter_ns() - t0) / 1000.0)

            # COPY-BOTH arm: the SAME cycle but charging the request drain as a real memcpy (the cost IF
            # zero-copy is NOT realized — measured here so the operator sees what zero-copy buys).
            ctr[0] = n_msgs
            t0 = time.perf_counter_ns()
            drained_rows = min(int(ctr[0]) * rows_per_msg, B)
            contiguous_in[:drained_rows] = req_ring[:drained_rows]   # the charged request-drain memcpy
            rep_ring[:drained_rows] = fwd_out[:drained_rows]
            ctr[0] = 0
            per_cycle_copyboth_us.append((time.perf_counter_ns() - t0) / 1000.0)

        med = float(np.median(per_cycle_us))
        med_cb = float(np.median(per_cycle_copyboth_us))
        return {"tau_io_us_median": med, "tau_io_copyboth_us_median": med_cb,
                "per_cycle_us": per_cycle_us, "per_cycle_copyboth_us": per_cycle_copyboth_us,
                "n_msgs": n_msgs, "rows_per_msg": rows_per_msg, "rows_per_forward": B}
    finally:
        for shm in (shm_req, shm_rep, shm_ctr):
            shm.close()
            shm.unlink()


def run(n_msgs: int = 8, rows_per_msg: int = 32, cycles: int = 5000) -> dict[str, Any]:
    """Measure futex_wake tau_io and LOG it to postgres (per-cycle us readings + the zero-copy headline
    median; the copy-both arm is logged as a config note). TIMING-SENSITIVE — operator-invoked, pinned
    (taskset -c 0), NEVER during the fan-out."""
    res = measure(n_msgs=n_msgs, rows_per_msg=rows_per_msg, cycles=cycles)
    cfg = {"n_msgs": res["n_msgs"], "rows_per_msg": res["rows_per_msg"],
           "rows_per_forward": res["rows_per_forward"], "cycles": cycles,
           "transport": "shm_ring_futex_wake", "request_drain": "zero_copy_view",
           "tau_io_copyboth_us_median": res["tau_io_copyboth_us_median"],
           "note": "same ring drain as shm_spin_poll (futex wakeup is the separate futex_wake_wakeup_us term); "
                   "headline = zero-copy request drain (design intent); copy-both arm charges the req memcpy"}
    with logged_run(NAME, quantity="serve_transport_io_cost_futex_wake", units="us", description=_DESC,
                    module_path=MODULE_PATH, config=cfg) as log:
        log(res["tau_io_us_median"], sample_size=cycles)        # headline median (zero-copy drain)
        log(res["per_cycle_us"], sample_size=1)                  # raw per-cycle readings
    return res


if __name__ == "__main__":
    _m, _s, _u = get_seed()
    print(f"[bench_futex_wake_tau_io_us] seed: {_m:.2f} {_u} (sigma {_s:.2f}) — FIRST-PRINCIPLES "
          f"(zero-copy drain + reply memcpy at B_op={_B_OP_SEED}; conservative {_MEMCPY_BW_BYTES_PER_NS} B/ns; "
          f"same ring drain as shm_spin_poll)")
    register_self()
    print("[bench_futex_wake_tau_io_us] registered. NOT running the live measurement (timing-sensitive); "
          "invoke run() pinned (taskset -c 0) and sole-workload. This is the futex variant's TOP Neyman target.")
