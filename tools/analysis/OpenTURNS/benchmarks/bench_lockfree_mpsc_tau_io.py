"""
tools/analysis/OpenTURNS/benchmarks/bench_lockfree_mpsc_tau_io.py
================================================================

LIVE benchmark for `lockfree_mpsc_tau_io_us` — the SERVER-side per-forward serial TRANSPORT
cost (us) for the LOCK-FREE MPSC transport: N producer cores ENQUEUE leaf-eval request nodes
into a single multi-producer/single-consumer queue (a CAS on the tail, no per-message mutex,
no broker), and the 1 serve core BATCH-DEQUEUES all ready nodes into ONE forward. It is the
SAME binding-stage term the ZMQ baseline `tau_io_us` occupies (model_cycletime.py
`cycle = T_disp + tau_io + B_eff*t_row`), with the mechanism swapped — so this variant
registers its OWN prefixed quantity (`lockfree_mpsc_tau_io_us`).

WHAT THE MPSC TRANSPORT MOVES (vs the ZMQ baseline's ~20us seed):
  * msg-cost / dequeue -> a CAS-pop per node (no `recv_multipart` syscall, no broker hop, no
    per-message `inference_wire.decode_request`/`encode`/`send_multipart` envelope). The serve
    core pops T ready nodes by following the queue's atomic head — ~tens of ns per pop, not
    the microseconds-per-message ZMQ multipart path. This is the term that COLLAPSES.
  * GATHER (the HONEST cost MPSC does NOT eliminate). A batched forward needs B feature rows
    CONTIGUOUS in one `(B, in_dim)` input. The existing wire path already pays this — the C++
    `WireLeafPool::submit_batch` is "the STRICT GATHER-BARRIER: gather B parked rows into ONE
    encode_request(flat, B, in_dim)" (cpp/include/chocofarm/wire_leaf_pool.hpp). An MPSC queue's
    nodes are enqueued INDEPENDENTLY by N producers, so they are NOT contiguous in memory; the
    consumer must GATHER the B rows out of the scattered node payloads into the contiguous input.
    So the gather is INTRINSIC to coalescing — it is NOT what MPSC removes (MPSC removes the
    ZMQ ENVELOPE around each message, not the row gather). The headline tau_io CHARGES the
    gather; whether it can be ELIDED (the staging accepting a scatter/gather iovec list instead
    of a materialized contiguous buffer) is a SEPARATE measurable question
    (`lockfree_mpsc_gather_us`, the dominant uncertainty — see that bench), so this bench
    measures BOTH arms.
  * REPLY -> an in-slot memcpy: the server copies the forward's B reply rows
    ([value][logits] per row) into per-producer reply slots + signals completion; no frame
    envelope, no `send_multipart`.

SO THE FIRST-PRINCIPLES tau_io for a forward of B rows assembled from T enqueued nodes is
      tau_io ~= gather (B * req_row_B / bw)          [INTRINSIC — charged in the headline]
              + reply-slot memcpy (B * rep_row_B / bw)
              + dequeue CAS pops (T * c_cas)
  The SEED is computed from that decomposition at the operating point B_op=256 (a conservative
single-thread memcpy bandwidth so the seed is an UPPER bound on tau_io / a LOWER bound on
throughput). See get_seed() for the arithmetic + provenance.

WHY THE HEADLINE CHARGES THE GATHER (the honest asymmetry vs shm_spin_poll). The shm-ring
transport's contiguous ring lets the staging read a zero-copy SPAN, so its headline ELIDES the
request copy and the copy-both arm is the pessimistic contrast. For MPSC the nodes are
genuinely scattered (independent enqueues), so gather-elision is LESS plausible (it needs a
scatter/gather-aware staging path); the honest default is therefore the gather-CHARGED arm,
with elision as the OPTIMISTIC contrast. Reporting it the other way would understate the bound.

WHAT run() MEASURES (1:1 with the model input, NO JAX forward). A sole-workload microbench of
the MPSC serve loop's I/O ONLY: T producer "nodes" hold feature rows at SCATTERED offsets in a
backing slab (modelling independent enqueues — the rows are NOT contiguous); the consumer
DEQUEUES them (an index pop per node) and GATHERS the B rows into one contiguous input buffer,
then memcpy's a B-row reply block into a reply slab. The per-forward tau_io is that I/O time.
run() ALSO records the gather-elided arm (no contiguous materialization — a view-list stand-in)
so the operator sees what gather-elision buys. It logs per-cycle us readings.

TIMING-SENSITIVE — DO NOT run() during the parallel fan-out (co-scheduling corrupts the
memcpy-bandwidth + CAS timing). Pin: `taskset -c 0`.

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

NAME = "lockfree_mpsc_tau_io_us"
MODULE_PATH = "benchmarks.bench_lockfree_mpsc_tau_io"
_DESC = ("LOCK-FREE MPSC server per-forward serial transport cost (us): batch-dequeue T ready nodes "
         "(CAS pops, no syscall/broker/codec envelope), GATHER the B scattered request rows into a "
         "contiguous (B,in_dim) input, memcpy the B reply rows into per-producer reply slots. The "
         "transport-design term in the binding serve cycle; the MPSC variant of tau_io_us (collapses the "
         "ZMQ multipart/syscall/per-message-codec to CAS pops, but the GATHER is intrinsic to coalescing).")

# Production geometry (matches bench_tau_io / inference_wire.hpp): request feature width + reply width.
# A full bucket B_op rows is assembled from T enqueued nodes of rows_per_node each.
_IN_DIM = 241
_N_ACTIONS = 65
_REQ_ROW_B = _IN_DIM * 4               # 964 B/row (241 f32 features)
_REP_ROW_B = (1 + _N_ACTIONS) * 4      # 264 B/row (value + n_actions logits)

# --- seed parameters (first-principles; the SEED arithmetic homes here) ---------------------
_B_OP_SEED = 256          # the operating-point full bucket the seed is evaluated at (leaf_eval_grounding B_op)
_ROWS_PER_NODE = 32       # producer coalescing degree (one enqueued node carries rows_per_node rows); T = B_op/rows_per_node
_MEMCPY_BW_BYTES_PER_NS = 8.0    # CONSERVATIVE single-thread sequential memcpy (slow -> larger tau_io -> LOWER throughput bound)
_CAS_NS = 30.0            # one uncontended dequeue CAS / atomic-head pop (cache-hot; a contended pop is ~3x, surfaced in sigma)


def get_seed() -> tuple[float, float, str]:
    """The v1 SEED (DISTRUST fallback) for lockfree_mpsc_tau_io — a FIRST-PRINCIPLES estimate (no v1
    measurement exists; this is a NEW transport quantity). Decomposition at the operating point
    B_op=256, T=B_op/32=8 enqueued nodes, with the GATHER CHARGED (the honest MPSC default — the
    nodes are scattered, so the B rows must be gathered contiguous; the existing WireLeafPool
    submit_batch already pays this gather):

        tau_io ~= gather       (B_op * 964 B / 8 B/ns)            ~= 30.85 us   [INTRINSIC]
                + reply memcpy  (B_op * 264 B / 8 B/ns)            ~=  8.45 us
                + dequeue CAS   (T * 30 ns)                        ~=  0.24 us
                = ~39.5 us

    A CONSERVATIVE memcpy bandwidth (8 B/ns) is used so the seed is an UPPER bound on tau_io (a LOWER
    bound on throughput). sigma is wide (the gather-elision question dominates the spread): if the
    staging accepts a scatter/gather list the gather (~30.85us) drops out (see lockfree_mpsc_gather_us),
    so the spread between the gather-charged headline and the gather-elided contrast is the dominant
    uncertainty. Returns (mean, sigma, unit)."""
    T = _B_OP_SEED // _ROWS_PER_NODE
    gather_us = _B_OP_SEED * _REQ_ROW_B / _MEMCPY_BW_BYTES_PER_NS / 1000.0
    reply_us = _B_OP_SEED * _REP_ROW_B / _MEMCPY_BW_BYTES_PER_NS / 1000.0
    dequeue_us = T * _CAS_NS / 1000.0
    mean = gather_us + reply_us + dequeue_us
    # sigma: ~half the gather copy (the gather-elided-vs-charged ambiguity dominates the spread).
    sigma = 0.5 * gather_us
    return (mean, sigma, "us")


def register_self() -> Any:
    from bench_common import register_quantity
    return register_quantity(NAME, quantity="serve_transport_io_cost_lockfree_mpsc", units="us",
                             description=_DESC, module_path=MODULE_PATH)


def measure(n_nodes: int = 8, rows_per_node: int = 32, cycles: int = 5000) -> dict[str, Any]:
    """Measure lockfree_mpsc tau_io: time ONE batch-dequeue+gather+reply cycle over `n_nodes` enqueued
    nodes of `rows_per_node` rows each (forward sees n_nodes*rows_per_node rows). NO JAX forward — the
    forward is iota+t_row*B (separate). Models the MPSC mechanism: producer rows live at SCATTERED
    offsets in a backing slab (independent enqueues are NOT contiguous), a node-index queue (a numpy
    int array the consumer pops) carries the enqueue order, and the consumer GATHERS the B rows out of
    the scattered slab into one contiguous input + memcpy's a B-row reply block into a reply slab.
    Returns {'tau_io_us_median', 'tau_io_gather_elided_us_median', 'per_cycle_us', ...}. Imports numpy +
    shared_memory lazily. Pin the process (taskset -c 0)."""
    import numpy as np
    from multiprocessing import shared_memory

    B = n_nodes * rows_per_node
    # Backing slab with HEADROOM so node payloads land at SCATTERED, non-contiguous offsets (the MPSC
    # property: N producers reserve slots independently, interleaved — the rows are not a single span).
    slab_rows = B * 4
    slab_bytes = slab_rows * _IN_DIM * 4
    rep_bytes = B * (1 + _N_ACTIONS) * 4
    shm_slab = shared_memory.SharedMemory(create=True, size=slab_bytes)
    shm_rep = shared_memory.SharedMemory(create=True, size=rep_bytes)
    try:
        slab = np.ndarray((slab_rows, _IN_DIM), dtype=np.float32, buffer=shm_slab.buf)
        rep_ring = np.ndarray((B, 1 + _N_ACTIONS), dtype=np.float32, buffer=shm_rep.buf)
        slab[:] = np.ones((slab_rows, _IN_DIM), dtype=np.float32)
        # The enqueue ORDER: each node's rows occupy a scattered, strided block of the slab (a stand-in for
        # independent multi-producer enqueues). The per-row source indices the gather follows.
        rng = np.random.default_rng(0)
        node_starts = (rng.permutation(slab_rows // rows_per_node)[:n_nodes]) * rows_per_node
        src_idx = np.concatenate([np.arange(s, s + rows_per_node) for s in node_starts])  # B scattered row indices
        node_queue = np.array(node_starts, dtype=np.int64)  # the index queue the consumer pops (T pops)

        contiguous_in = np.empty((B, _IN_DIM), dtype=np.float32)  # the gather target (materialized)
        fwd_out = np.zeros((B, 1 + _N_ACTIONS), dtype=np.float32)  # the forward's reply block (content irrelevant; copy cost is what matters)

        # Warm caches + page table.
        for _ in range(min(200, cycles)):
            _ = int(node_queue.sum())
            contiguous_in[:] = slab[src_idx]
            rep_ring[:] = fwd_out

        per_cycle_us: list[float] = []
        per_cycle_elided_us: list[float] = []
        for _ in range(cycles):
            # GATHER-CHARGED arm (the honest headline): pop T nodes (sum stands for the CAS-pop bookkeeping),
            # GATHER the B scattered rows into a contiguous input, memcpy the B reply rows out.
            t0 = time.perf_counter_ns()
            _popped = int(node_queue.sum())            # the dequeue-pop bookkeeping (T atomic head advances)
            contiguous_in[:] = slab[src_idx]            # the GATHER (B scattered rows -> contiguous) — intrinsic
            rep_ring[:] = fwd_out                        # the reply-slot memcpy (the only other mandatory copy)
            per_cycle_us.append((time.perf_counter_ns() - t0) / 1000.0)

            # GATHER-ELIDED arm (the OPTIMISTIC contrast): the staging accepts a scatter/gather list, so the
            # B rows are NOT materialized — only the dequeue bookkeeping + the reply memcpy remain. (A
            # view-list stand-in: no contiguous copy of the request rows.)
            t0 = time.perf_counter_ns()
            _popped = int(node_queue.sum())             # same dequeue bookkeeping
            _view_list = [slab[s:s + rows_per_node] for s in node_queue]  # noqa: F841 — views, no row copy (the iovec the staging would consume)
            rep_ring[:] = fwd_out                         # the reply memcpy still happens
            per_cycle_elided_us.append((time.perf_counter_ns() - t0) / 1000.0)

        med = float(np.median(per_cycle_us))
        med_el = float(np.median(per_cycle_elided_us))
        return {"tau_io_us_median": med, "tau_io_gather_elided_us_median": med_el,
                "per_cycle_us": per_cycle_us, "per_cycle_gather_elided_us": per_cycle_elided_us,
                "n_nodes": n_nodes, "rows_per_node": rows_per_node, "rows_per_forward": B}
    finally:
        for shm in (shm_slab, shm_rep):
            shm.close()
            shm.unlink()


def run(n_nodes: int = 8, rows_per_node: int = 32, cycles: int = 5000) -> dict[str, Any]:
    """Measure lockfree_mpsc tau_io and LOG it to postgres (per-cycle us readings + the gather-charged
    headline median; the gather-elided arm is logged as a config note + supporting readings).
    TIMING-SENSITIVE — operator-invoked, pinned (taskset -c 0), NEVER during the fan-out."""
    res = measure(n_nodes=n_nodes, rows_per_node=rows_per_node, cycles=cycles)
    cfg = {"n_nodes": res["n_nodes"], "rows_per_node": res["rows_per_node"],
           "rows_per_forward": res["rows_per_forward"], "cycles": cycles,
           "transport": "lockfree_mpsc_queue", "gather": "charged_contiguous_materialize",
           "tau_io_gather_elided_us_median": res["tau_io_gather_elided_us_median"],
           "note": "headline = gather CHARGED (intrinsic to coalescing; scattered nodes); "
                   "gather-elided arm = staging accepts a scatter/gather iovec list (optimistic)"}
    with logged_run(NAME, quantity="serve_transport_io_cost_lockfree_mpsc", units="us", description=_DESC,
                    module_path=MODULE_PATH, config=cfg) as log:
        log(res["tau_io_us_median"], sample_size=cycles)        # headline median (gather charged)
        log(res["per_cycle_us"], sample_size=1)                  # raw per-cycle readings
    return res


if __name__ == "__main__":
    _m, _s, _u = get_seed()
    print(f"[bench_lockfree_mpsc_tau_io] seed: {_m:.2f} {_u} (sigma {_s:.2f}) — FIRST-PRINCIPLES "
          f"(gather CHARGED + reply memcpy + CAS dequeue at B_op={_B_OP_SEED}; conservative {_MEMCPY_BW_BYTES_PER_NS} B/ns)")
    register_self()
    print("[bench_lockfree_mpsc_tau_io] registered. NOT running the live measurement (timing-sensitive); "
          "invoke run() pinned (taskset -c 0) and sole-workload. This is the MPSC variant's TOP Neyman target.")
