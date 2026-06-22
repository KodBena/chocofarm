"""
tools/analysis/leaf_eval_bound/benchmarks/bench_cpp_inproc_port_tau_io_us.py
======================================================================

LIVE benchmark for `cpp_inproc_port_tau_io_us` — the SERVER-side per-forward serial TRANSPORT cost
(us) for the C++ IN-PROCESS QUEUE-PORT transport: generation and serve run in ONE process, a
leaf-eval is a DIRECT function call into the batched forward — NO wire at all (no ZMQ ROUTER/DEALER,
no `recv_multipart`/`send_multipart` syscall, no broker hop, no `encode_request`/`decode_request`
codec, no corr-id envelope). It is the SAME binding-stage term the ZMQ baseline `tau_io_us` occupies
(model_cycletime.py `cycle = T_disp + tau_io + B_eff*t_row`), with the mechanism removed entirely —
so this variant registers its OWN prefixed quantity (`cpp_inproc_port_tau_io_us`).

WHAT THE INPROC PORT REMOVES vs WHAT REMAINS (the honest residual).
  REMOVED (the ZMQ baseline's ~20us seed): the multipart recv/send syscalls, the broker hop, the
  per-message wire codec (`inference_wire` encode/decode), the corr-id framing, AND the
  device->host pull-to-host-then-reframe (the in-process caller reads the forward's reply
  device-resident — see why t_row is the fully_device slope in
  bench_cpp_inproc_port_t_row_bare_us). These ALL collapse to ~0: there is no wire.

  REMAINS (intrinsic to coalescing a batched forward, NOT what a wire adds): the forward needs B
  feature rows as ONE `(B, in_dim)` block, and those rows ORIGINATE ON HOST (the CPU gen cores) and
  must cross host->device ONCE per forward. The production path already names this exact crossing —
  "ONE host->device crossing (the cast, folded in)" (inference_server.jit_forward_core) — and the
  fully_device slope (the inproc-port t_row) DELIBERATELY EXCLUDES it (fully_device feeds
  device-resident input). So the host->device stage of the gathered B-row block is the honest
  residual tau_io, charged HERE so t_row (bare slope) + tau_io (this crossing) carry the full
  per-forward cost with no double-count and no gap.

THE ARENA / GATHER ASYMMETRY (vs lockfree_mpsc — the inverse). The MPSC queue's nodes are enqueued
INDEPENDENTLY by N producers, so they are scattered and the MPSC headline CHARGES a gather (elision
is the optimistic arm). The inproc port lives in ONE address space, so the producers can write their
leaf feature rows DIRECTLY into the consumer's contiguous staging ARENA (one cache-line-aligned slab,
each producer owning a row stripe) — so the headline ELIDES the gather (no scattered-node
materialization) and pays ONLY the single host->device crossing of the contiguous block; the
gather-CHARGED arm (producers writing scattered, a same-process gather added) is the PESSIMISTIC
contrast. This is the honest default BECAUSE a contiguous arena is plausible in one process (it is
NOT plausible across the MPSC queue's independent enqueues). `copy_contrast()` in the model shows
both arms; the Neyman allocator ranks "is the staging arena contiguous (gather elidable)?" as the
dominant transport question for this variant.

SO THE FIRST-PRINCIPLES tau_io for a forward of B rows is
      tau_io ~= h2d_crossing (B * req_row_B / h2d_bw)        [INTRINSIC — charged in the headline]
              + arena gather  (B * req_row_B / memcpy_bw)    [ELIDED in the headline; the pessimistic arm]
  The SEED is computed from that decomposition at the operating point B_op=256 (a conservative
  host->device PCIe bandwidth so the seed is an UPPER bound on tau_io / a LOWER bound on throughput).
  See get_seed() for the arithmetic + provenance. The reply is read device-resident (no host reply
  copy — that is the device->host pull the fully_device slope and the wire-elision already account
  for), so there is NO reply-memcpy term here (unlike MPSC, which copies replies into wire slots).

WHAT run() MEASURES (1:1 with the model input, NO JAX forward). A sole-workload microbench of the
inproc serve loop's staging ONLY: B producer feature rows written into a contiguous staging arena;
time (a) the host->device crossing of the B-row block (a real `jax.device_put` of the contiguous
arena, blocked ready) — the headline; and (b) the gather-charged arm (rows at SCATTERED arena
offsets gathered into a contiguous buffer, then the same device_put). The per-forward tau_io is that
staging time. run() logs per-cycle us readings for both arms. Imports jax + numpy LAZILY (so
importing the module for get_seed() stays jax-free).

TIMING-SENSITIVE — DO NOT run() during the parallel fan-out (co-scheduling corrupts the
host->device-bandwidth + memcpy timing). Pin: `taskset -c 0`.

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

NAME = "cpp_inproc_port_tau_io_us"
MODULE_PATH = "benchmarks.bench_cpp_inproc_port_tau_io_us"
_DESC = ("C++ IN-PROCESS queue-port server per-forward serial transport cost (us): NO wire (no syscall/broker/"
         "codec/envelope, no device->host reframe). The residual is the INTRINSIC host->device crossing of the "
         "gathered B-row input block (the one transfer the fully_device t_row slope excludes); headline ELIDES "
         "the arena gather (one address space -> producers write contiguous), gather-charged is the pessimistic "
         "arm. The transport-design term in the binding serve cycle; the inproc-port variant of tau_io_us.")

# Production geometry (matches bench_tau_io / inference_wire.hpp): request feature width. A full bucket B_op
# rows is assembled from N producers' row stripes in the staging arena.
_IN_DIM = 241
_REQ_ROW_B = _IN_DIM * 4               # 964 B/row (241 f32 features)

# --- seed parameters (first-principles; the SEED arithmetic homes here) ---------------------
_B_OP_SEED = 256              # the operating-point full bucket the seed is evaluated at (leaf_eval_grounding B_op)
# CONSERVATIVE host->device transfer bandwidth (a LOWER bw -> a LARGER tau_io -> a LOWER throughput bound).
# This host is a 4-vCPU libvirt VM (CLAUDE.md) — no discrete GPU; the JAX "device" is the CPU backend, so the
# host->device "crossing" is a same-RAM device_put copy + the XLA buffer handoff, NOT a PCIe DMA. 6 B/ns is a
# conservative sustained single-thread copy+handoff bandwidth (slower than the ~8 B/ns the MPSC bench charges
# its in-RAM gather, because device_put adds the XLA buffer-management overhead on top of the raw copy).
_H2D_BW_BYTES_PER_NS = 6.0
# The arena gather bandwidth (the pessimistic arm): a same-RAM strided memcpy of the scattered rows.
_GATHER_BW_BYTES_PER_NS = 8.0


def get_seed() -> tuple[float, float, str]:
    """The v1 SEED (DISTRUST fallback) for cpp_inproc_port_tau_io — a FIRST-PRINCIPLES estimate (no v1
    measurement exists; this is a NEW transport quantity). The HEADLINE (gather ELIDED, the honest inproc
    default — one address space lets producers write a contiguous staging arena) is the host->device crossing
    of the contiguous B-row block ALONE:

        tau_io_headline ~= h2d_crossing (B_op * 964 B / 6 B/ns)   ~= 41.1 us   [INTRINSIC]

    A CONSERVATIVE host->device bandwidth (6 B/ns) is used so the seed is an UPPER bound on tau_io (a LOWER
    bound on throughput). sigma is wide: the gather-elision question dominates the spread — if the staging
    arena is NOT contiguous (producers write scattered) a same-RAM gather (~30.85us at 8 B/ns) is ADDED (the
    pessimistic arm, see the model's copy_contrast()), so the spread between the gather-elided headline and the
    gather-charged contrast is the dominant uncertainty. Returns (mean, sigma, unit)."""
    h2d_us = _B_OP_SEED * _REQ_ROW_B / _H2D_BW_BYTES_PER_NS / 1000.0
    gather_us = _B_OP_SEED * _REQ_ROW_B / _GATHER_BW_BYTES_PER_NS / 1000.0
    mean = h2d_us                      # headline: gather ELIDED (contiguous arena); only the h2d crossing
    sigma = 0.5 * gather_us            # the gather-charged-vs-elided ambiguity dominates the spread
    return (mean, sigma, "us")


def register_self() -> Any:
    from bench_common import register_quantity
    return register_quantity(NAME, quantity="serve_transport_io_cost_cpp_inproc_port", units="us",
                             description=_DESC, module_path=MODULE_PATH)


def _measure_raw(b_rows: int = 256, n_producers: int = 3, cycles: int = 5000) -> dict[str, Any]:
    """The raw-pool PROVENANCE producer (the §6 Phase-4 internal helper): measure cpp_inproc_port tau_io: time ONE per-forward staging cycle for a forward of `b_rows` rows
    assembled from `n_producers` row stripes. NO JAX forward — the forward is T_disp+t_row*B (separate).
    Models the inproc-port mechanism:
      * HEADLINE (gather ELIDED): the B rows already live CONTIGUOUS in a staging arena (one address space,
        producers wrote their stripes in place); time the host->device crossing of the contiguous block (a
        real `jax.device_put(arena).block_until_ready()`).
      * GATHER-CHARGED (pessimistic): the B rows live at SCATTERED arena offsets (producers wrote
        non-contiguous); time the same-RAM gather into a contiguous buffer THEN the device_put.
    Returns {'tau_io_us_median', 'tau_io_gather_charged_us_median', 'per_cycle_us', ...}. Imports jax + numpy
    lazily. Pin the process (taskset -c 0). `measure()` wraps the per-cycle pool into a median `Estimate`; `run()` uses it for BOTH the Estimate and the raw provenance rows (ONE measurement, two consumers — P1)."""
    import numpy as np
    import jax

    B = b_rows
    # The contiguous staging arena (the headline): B rows in place. The scattered slab (the pessimistic arm):
    # B rows at strided, non-contiguous offsets in a slab with headroom (a stand-in for non-arena writes).
    arena = np.ones((B, _IN_DIM), dtype=np.float32)
    slab_rows = B * 4
    slab = np.ones((slab_rows, _IN_DIM), dtype=np.float32)
    rng = np.random.default_rng(0)
    rows_per_stripe = max(1, B // n_producers)
    stripe_starts = (rng.permutation(slab_rows // rows_per_stripe)[:n_producers]) * rows_per_stripe
    src_idx = np.concatenate([np.arange(s, s + rows_per_stripe) for s in stripe_starts])
    src_idx = src_idx[:B] if src_idx.shape[0] >= B else np.resize(src_idx, B)  # exactly B scattered indices
    gather_target = np.empty((B, _IN_DIM), dtype=np.float32)

    # Warm: JIT/dispatch the device_put path + page-in the buffers.
    for _ in range(min(200, cycles)):
        jax.device_put(arena).block_until_ready()
        gather_target[:] = slab[src_idx]

    per_cycle_us: list[float] = []
    per_cycle_gather_us: list[float] = []
    for _ in range(cycles):
        # HEADLINE arm (gather ELIDED): host->device crossing of the contiguous arena ONLY.
        t0 = time.perf_counter_ns()
        d = jax.device_put(arena)
        d.block_until_ready()
        per_cycle_us.append((time.perf_counter_ns() - t0) / 1000.0)

        # GATHER-CHARGED arm (pessimistic): same-RAM gather of scattered rows THEN the device_put.
        t0 = time.perf_counter_ns()
        gather_target[:] = slab[src_idx]               # the same-process gather (B scattered rows -> contiguous)
        d2 = jax.device_put(gather_target)
        d2.block_until_ready()
        per_cycle_gather_us.append((time.perf_counter_ns() - t0) / 1000.0)

    med = float(np.median(per_cycle_us))
    med_g = float(np.median(per_cycle_gather_us))
    return {"tau_io_us_median": med, "tau_io_gather_charged_us_median": med_g,
            "per_cycle_us": per_cycle_us, "per_cycle_gather_charged_us": per_cycle_gather_us,
            "b_rows": B, "n_producers": n_producers}


def _estimate_from_raw(res: dict[str, Any]) -> "_est.Estimate":
    """Build this bench's harmonized `Estimate` from a `_measure_raw()` dict — the SINGLE home of the
    Estimate construction (P1), called by BOTH `measure()` and `run()`. A k=1 median `QuantileLaw(p=0.5)`
    with a BOOTSTRAP median SE over the pool (§7.A — the order-statistic variance, NOT s²/n),
    family=EMPIRICAL, kind='median'."""
    return median_estimate(res["per_cycle_us"], name=NAME)   # bootstrap median SE over the per-cycle pool


def measure(b_rows: int = 256, n_producers: int = 3, cycles: int = 5000) -> "_est.Estimate":
    """Measure cpp_inproc_port tau_io and return its harmonized k=1 median `Estimate` (§6 Phase 4: `measure()`
    returns the `Estimate` the bench DECLARES — the driver/untrusted_drive consume it directly, no
    guessing which list is the pool). The raw pool is the bench's internal `_measure_raw()` provenance.
    TIMING-SENSITIVE — pin the process (taskset -c 0)."""
    return _estimate_from_raw(_measure_raw(b_rows=b_rows, n_producers=n_producers, cycles=cycles))


def run(b_rows: int = 256, n_producers: int = 3, cycles: int = 5000) -> dict[str, Any]:
    """Measure cpp_inproc_port tau_io and LOG it as a harmonized k=1 median Estimate (QuantileLaw p=0.5,
    bootstrap median SE, §6 Phase 3, §5.2 de-dup). TIMING-SENSITIVE — operator-invoked, pinned (taskset -c 0),
    NEVER during the fan-out."""
    res = _measure_raw(b_rows=b_rows, n_producers=n_producers, cycles=cycles)  # ONE measurement (Estimate + provenance)
    est = _estimate_from_raw(res)  # the SAME Estimate measure() returns (P1)
    cfg = {"b_rows": res["b_rows"], "n_producers": res["n_producers"], "cycles": cycles,
           "transport": "cpp_inproc_port_direct_call", "gather": "elided_contiguous_arena",
           "tau_io_gather_charged_us_median": res["tau_io_gather_charged_us_median"],
           "tau_io_us_median": res["tau_io_us_median"],
           "note": "headline = gather ELIDED (one address space -> contiguous staging arena; only the "
                   "host->device crossing of the B-row block remains); gather-charged arm = scattered writes "
                   "+ a same-process gather (pessimistic). No reply-memcpy (the reply is read device-resident)."}
    with logged_run(NAME, quantity="serve_transport_io_cost_cpp_inproc_port", units="us", description=_DESC,
                    module_path=MODULE_PATH, config=cfg, estimate=est) as log:
        # PROVENANCE only (§5.2 de-dup): the headline median lives in estimate.theta_hat[0], not a sample row.
        log(res["per_cycle_us"], sample_size=1)                  # raw per-cycle readings
    return res


if __name__ == "__main__":
    _m, _s, _u = get_seed()
    print(f"[bench_cpp_inproc_port_tau_io_us] seed: {_m:.2f} {_u} (sigma {_s:.2f}) — FIRST-PRINCIPLES "
          f"(host->device crossing of the contiguous B-row block at B_op={_B_OP_SEED}, conservative "
          f"{_H2D_BW_BYTES_PER_NS} B/ns; gather ELIDED — the honest one-address-space default)")
    register_self()
    print("[bench_cpp_inproc_port_tau_io_us] registered. NOT running the live measurement (timing-sensitive); "
          "invoke run() pinned (taskset -c 0) and sole-workload. This is the inproc-port variant's TOP Neyman "
          "target (alongside: is the staging arena contiguous?).")
