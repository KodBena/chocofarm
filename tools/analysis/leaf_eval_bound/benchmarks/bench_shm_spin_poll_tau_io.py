"""
tools/analysis/leaf_eval_bound/benchmarks/bench_shm_spin_poll_tau_io.py
================================================================

LIVE benchmark for `shm_spin_poll_tau_io_us` — the SERVER-side per-forward serial
TRANSPORT cost (us) for the SHARED-MEMORY SPIN-POLL transport: the serve core BUSY-POLL-
SPINS an atomic head/tail counter (no syscall, no broker hop, no `recv_multipart`), drains
the queued request rows straight out of a shared-memory RING, and writes the forward's
replies into a reply ring. It is the SAME binding-stage term the ZMQ baseline `tau_io_us`
occupies, but with the mechanism swapped — so a transport variant registers its OWN
prefixed quantity (`shm_spin_poll_tau_io_us`) and a model substitutes it into the
invariant serve cycle `cycle = T_disp + tau_io + B_eff*t_row` (model_cycletime.py).

WHAT THE SHM TRANSPORT MOVES (vs the ZMQ baseline's ~20us seed):
  * WAKEUP -> ~0: a cached atomic LOAD of the producer-bumped tail counter, no `zmq.poll`
    syscall, no context switch (it costs one DEDICATED burnt poll core — the 1-serve-core
    layout already burns that core, so spinning it is free in the fixed pinning).
  * DRAIN -> a ring memcpy, no multipart envelope, no broker: the producers write feature
    rows ROW-MAJOR directly into the request ring, so the rows are already contiguous in
    the ring and the host->device staging can read them IN PLACE (zero-copy request drain —
    the design's defining property: "drains rows straight out of the ring"). The
    request-side copy that the ZMQ `decode_request`+`np.concatenate` pays therefore
    COLLAPSES; whether it truly collapses to zero is a SEPARATE measurable question
    (`shm_spin_poll_req_drain_us`), so this bench measures BOTH arms.
  * REPLY -> an in-ring memcpy: the server copies the forward's B reply rows
    ([value][logits] per row) into the reply ring; no frame envelope, no `send_multipart`.

SO THE FIRST-PRINCIPLES tau_io RESIDUAL (the term this bench measures) is, per forward of B
rows assembled from T producer messages:
      tau_io ~= reply-ring memcpy (B * rep_row_B / bw)
              + per-message counter bookkeeping (T * c_idx — advance head, slot->producer)
              + spin (~0)
  with the request drain ZERO-COPY (a ring span handed to the staging) — the design intent.
The SEED is computed from that decomposition at the operating point B_op=256 (a conservative
single-thread memcpy bandwidth so the bound is a LOWER bound on throughput / an UPPER bound
on tau_io). See get_seed() for the arithmetic + provenance.

WHAT run() MEASURES (1:1 with the model input, NO JAX forward). A sole-workload microbench of
the shm serve loop's I/O ONLY: a producer process (or thread) deposits T coalesced request
frames of `rows_per_msg` rows ROW-MAJOR into a shared-memory request ring + bumps an atomic
tail counter; the server thread SPIN-POLLS that counter (no syscall), drains the B rows
(zero-copy: a numpy view over the ring slice), then memcpy's a B-row reply block into a reply
ring + bumps the reply counter. The per-forward tau_io is that I/O time amortized over one
drain cycle. run() ALSO records the copy-both arm (charging the request drain) as a config
note so the operator sees how much zero-copy buys. It logs per-cycle us readings.

TIMING-SENSITIVE — DO NOT run() during the parallel fan-out (co-scheduling corrupts the
memcpy-bandwidth + spin timing). Pin: `taskset -c 0`.

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
from bench_common import logged_run, median_estimate  # noqa: E402

NAME = "shm_spin_poll_tau_io_us"
MODULE_PATH = "benchmarks.bench_shm_spin_poll_tau_io"
_DESC = ("SHM SPIN-POLL server per-forward serial transport cost (us): spin an atomic tail counter "
         "(no syscall), drain request rows ZERO-COPY out of a shared-memory ring, memcpy the B reply rows "
         "into a reply ring. The transport-design term in the binding serve cycle; the SHM variant of "
         "tau_io_us (collapses the ZMQ multipart/syscall/broker cost to a reply memcpy + counter bookkeeping).")

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
    """The v1 SEED (DISTRUST fallback) for shm_spin_poll_tau_io — a FIRST-PRINCIPLES estimate (no v1
    measurement exists; this is a NEW transport quantity). Decomposition at the operating point
    B_op=256, T=B_op/32=8 producer messages, ZERO-COPY request drain (the design intent):

        tau_io ~= reply-ring memcpy  (B_op * 264 B / 8 B/ns)            ~= 8.45 us
                + counter bookkeeping (T * 40 ns)                       ~= 0.32 us
                + spin                                                   ~  0    us
                = ~8.8 us

    A CONSERVATIVE memcpy bandwidth (8 B/ns) is used so the seed is an UPPER bound on tau_io (a LOWER
    bound on throughput). sigma is wide (the bandwidth + the realized zero-copy fraction are both
    unmeasured): the request-drain copy, if NOT elided, adds ~31us (see shm_spin_poll_req_drain_us), so
    the spread between the zero-copy and copy-both arms is the dominant uncertainty. Returns
    (mean, sigma, unit)."""
    reply_mc_us = _B_OP_SEED * _REP_ROW_B / _MEMCPY_BW_BYTES_PER_NS / 1000.0
    counter_us = (_B_OP_SEED // _ROWS_PER_MSG) * _COUNTER_NS / 1000.0
    mean = reply_mc_us + counter_us
    # sigma: ~half the request-drain copy (the zero-copy-vs-copy ambiguity dominates the spread).
    req_drain_us = _B_OP_SEED * _REQ_ROW_B / _MEMCPY_BW_BYTES_PER_NS / 1000.0
    sigma = 0.5 * req_drain_us
    return (mean, sigma, "us")


def register_self() -> Any:
    from bench_common import register_quantity
    return register_quantity(NAME, quantity="serve_transport_io_cost_shm_spin_poll", units="us",
                             description=_DESC, module_path=MODULE_PATH)


def _measure_raw(n_msgs: int = 8, rows_per_msg: int = 32, cycles: int = 5000) -> dict[str, Any]:
    """The raw-pool PROVENANCE producer (the §6 Phase-4 internal helper): measure shm_spin_poll tau_io: time ONE drain+reply cycle over `n_msgs` coalesced producer messages of
    `rows_per_msg` rows each (forward sees n_msgs*rows_per_msg rows). NO JAX forward — the forward is
    iota+t_row*B (separate). Uses a shared-memory RING (a numpy-backed `multiprocessing.shared_memory`
    buffer) + an atomic-style tail counter the SERVER SPIN-POLLS — no zmq, no syscall, no broker. Drains the
    request rows ZERO-COPY (a numpy view over the ring slice) and memcpy's the reply block into a reply ring.
    Returns {'tau_io_us_median', 'tau_io_copyboth_us_median', 'per_cycle_us', ...}. Imports numpy +
    shared_memory lazily. Pin the process (taskset -c 0). `measure()` wraps the per-cycle pool into a median `Estimate`; `run()` uses it for BOTH the Estimate and the raw provenance rows (ONE measurement, two consumers — P1)."""
    import numpy as np
    from multiprocessing import shared_memory

    B = n_msgs * rows_per_msg
    # Request ring: B rows of in_dim f32, row-major (producers write here; server drains it).
    req_bytes = B * _IN_DIM * 4
    rep_bytes = B * (1 + _N_ACTIONS) * 4
    shm_req = shared_memory.SharedMemory(create=True, size=req_bytes)
    shm_rep = shared_memory.SharedMemory(create=True, size=rep_bytes)
    # The tail counter is a single int in its own tiny shared buffer (the atomic the server spins on).
    shm_ctr = shared_memory.SharedMemory(create=True, size=8)
    try:
        req_ring = np.ndarray((B, _IN_DIM), dtype=np.float32, buffer=shm_req.buf)
        rep_ring = np.ndarray((B, 1 + _N_ACTIONS), dtype=np.float32, buffer=shm_rep.buf)
        ctr = np.ndarray((1,), dtype=np.int64, buffer=shm_ctr.buf)
        # The forward's output the server will copy into the reply ring (a (B, 1+n_actions) block — the
        # shape run_microbatch returns). Pre-allocated; its CONTENT is irrelevant, the memcpy COST is what
        # the per-forward reply scatter costs.
        fwd_out = np.zeros((B, 1 + _N_ACTIONS), dtype=np.float32)
        producer_rows = np.ones((B, _IN_DIM), dtype=np.float32)   # what the producers would deposit

        per_cycle_us: list[float] = []
        per_cycle_copyboth_us: list[float] = []
        contiguous_in = np.empty((B, _IN_DIM), dtype=np.float32)  # the copy-both arm's gather target

        # Warm the caches + the page table.
        for _ in range(min(200, cycles)):
            req_ring[:] = producer_rows
            ctr[0] += n_msgs
            _ = req_ring[: ctr_to_rows(ctr[0], rows_per_msg, B)]
            rep_ring[:] = fwd_out
            ctr[0] = 0

        for _ in range(cycles):
            # PRODUCER side: deposit n_msgs frames row-major into the ring + bump the tail counter by n_msgs.
            req_ring[:] = producer_rows         # the producers' writes (off the server's critical path; the
                                                #   server's tau_io is the DRAIN+REPLY, timed below)
            ctr[0] = n_msgs                     # publish: tail counter now says n_msgs messages are ready

            t0 = time.perf_counter_ns()
            # SERVER side (the tau_io critical section): SPIN-POLL the counter (no syscall), then drain.
            while ctr[0] < n_msgs:              # the busy-poll spin (here it is already satisfied — ~0)
                pass
            drained_rows = ctr[0] * rows_per_msg
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
            while ctr[0] < n_msgs:
                pass
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


def _estimate_from_raw(res: dict[str, Any]) -> "_est.Estimate":
    """Build this bench's harmonized `Estimate` from a `_measure_raw()` dict — the SINGLE home of the
    Estimate construction (P1), called by BOTH `measure()` and `run()`. A k=1 median `QuantileLaw(p=0.5)`
    with a BOOTSTRAP median SE over the pool (§7.A — the order-statistic variance, NOT s²/n),
    family=EMPIRICAL, kind='median'."""
    return median_estimate(res["per_cycle_us"], name=NAME)   # bootstrap median SE over the per-cycle pool


def measure(n_msgs: int = 8, rows_per_msg: int = 32, cycles: int = 5000) -> "_est.Estimate":
    """Measure shm_spin_poll tau_io and return its harmonized k=1 median `Estimate` (§6 Phase 4: `measure()`
    returns the `Estimate` the bench DECLARES — the driver/untrusted_drive consume it directly, no
    guessing which list is the pool). The raw pool is the bench's internal `_measure_raw()` provenance.
    TIMING-SENSITIVE — pin the process (taskset -c 0)."""
    return _estimate_from_raw(_measure_raw(n_msgs=n_msgs, rows_per_msg=rows_per_msg, cycles=cycles))


def ctr_to_rows(c: int, rows_per_msg: int, cap: int) -> int:
    """The drained row count for a tail-counter value `c` (messages), capped at the ring width — the trivial
    head/tail arithmetic the server does to know how many rows are ready."""
    return min(int(c) * rows_per_msg, cap)


def run(n_msgs: int = 8, rows_per_msg: int = 32, cycles: int = 5000) -> dict[str, Any]:
    """Measure shm_spin_poll tau_io and LOG it as a harmonized k=1 median Estimate (QuantileLaw p=0.5,
    bootstrap median SE, §6 Phase 3, §5.2 de-dup); the copy-both arm is logged as a config note + supporting
    readings. TIMING-SENSITIVE — operator-invoked, pinned (taskset -c 0), NEVER during the fan-out."""
    res = _measure_raw(n_msgs=n_msgs, rows_per_msg=rows_per_msg, cycles=cycles)  # ONE measurement (Estimate + provenance)
    est = _estimate_from_raw(res)  # the SAME Estimate measure() returns (P1)
    cfg = {"n_msgs": res["n_msgs"], "rows_per_msg": res["rows_per_msg"],
           "rows_per_forward": res["rows_per_forward"], "cycles": cycles,
           "transport": "shm_ring_spin_poll", "request_drain": "zero_copy_view",
           "tau_io_us_median": res["tau_io_us_median"],
           "tau_io_copyboth_us_median": res["tau_io_copyboth_us_median"],
           "note": "headline = zero-copy request drain (design intent); copy-both arm charges the req memcpy"}
    with logged_run(NAME, quantity="serve_transport_io_cost_shm_spin_poll", units="us", description=_DESC,
                    module_path=MODULE_PATH, config=cfg, estimate=est) as log:
        # PROVENANCE only (§5.2 de-dup): the headline median lives in estimate.theta_hat[0], not a sample row.
        log(res["per_cycle_us"], sample_size=1)                  # raw per-cycle readings
    return res


if __name__ == "__main__":
    _m, _s, _u = get_seed()
    print(f"[bench_shm_spin_poll_tau_io] seed: {_m:.2f} {_u} (sigma {_s:.2f}) — FIRST-PRINCIPLES "
          f"(zero-copy drain + reply memcpy at B_op={_B_OP_SEED}; conservative {_MEMCPY_BW_BYTES_PER_NS} B/ns)")
    register_self()
    print("[bench_shm_spin_poll_tau_io] registered. NOT running the live measurement (timing-sensitive); "
          "invoke run() pinned (taskset -c 0) and sole-workload. This is the SHM variant's TOP Neyman target.")
