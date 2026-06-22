"""
tools/analysis/leaf_eval_bound/benchmarks/bench_cpp_inproc_port_gather_us.py
======================================================================

LIVE benchmark for `cpp_inproc_port_gather_us` — the same-process ARENA GATHER cost (us) the C++
in-process queue-port AVOIDS when the staging arena is contiguous: the cost of materializing B
SCATTERED producer feature rows into one contiguous `(B, in_dim)` host block BEFORE the
host->device crossing. This is NOT a separate cycle term — it is the CONTRAST DOF that swings the
headline `cpp_inproc_port_tau_io_us` between its two arms (the dominant transport uncertainty for
this variant, the Neyman allocator's top DESIGN-priority question: "is the staging arena
contiguous, i.e. is the gather elidable?").

WHY IT IS THE DOMINANT TRANSPORT QUESTION. The inproc port lives in ONE address space, so the
producers CAN write their leaf feature rows directly into the consumer's contiguous staging arena
(each producer owning a row stripe) — eliding the gather; the tau_io headline assumes this (the
honest one-address-space default). But if the producers write SCATTERED (independent slot
reservations, a non-arena allocator), the consumer must GATHER the B rows contiguous first. The
gather-charged arm ADDS this term to tau_io. Whether the arena is contiguous is an
IMPLEMENTATION CHOICE measurable directly — hence this quantity. (Mirror of
`lockfree_mpsc_gather_us`, but INVERTED in disposition: the MPSC nodes are genuinely scattered so
MPSC CHARGES the gather and elision is optimistic; the inproc port can arrange a contiguous arena
so it ELIDES the gather and the charge is pessimistic.)

WHAT run() MEASURES (1:1 with the model input, NO JAX, NO host->device). A sole-workload microbench
of the same-RAM gather ALONE: B rows at SCATTERED offsets in a backing slab gathered (`out[:] =
slab[src_idx]`) into one contiguous `(B, in_dim)` buffer. The per-forward gather cost is that copy
time. The SEED is the first-principles `B_op * req_row_B / gather_bw` at the operating point
(matches the gather term inside bench_cpp_inproc_port_tau_io_us's pessimistic arm).

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

NAME = "cpp_inproc_port_gather_us"
MODULE_PATH = "leaf_eval_bound.benchmarks.bench_cpp_inproc_port_gather_us"
_DESC = ("Same-process ARENA GATHER cost (us) the C++ inproc-port avoids when the staging arena is contiguous: "
         "materialize B scattered producer rows into one contiguous (B,in_dim) host block before the host->device "
         "crossing. NOT a separate cycle term — the contrast DOF that swings cpp_inproc_port_tau_io_us between "
         "the gather-elided headline and the gather-charged arm (the dominant transport uncertainty; 'is the "
         "staging arena contiguous?').")

_IN_DIM = 241
_REQ_ROW_B = _IN_DIM * 4               # 964 B/row

_B_OP_SEED = 256
_GATHER_BW_BYTES_PER_NS = 8.0          # same-RAM strided memcpy (matches the gather term in bench_..._tau_io_us)


def get_seed() -> tuple[float, float, str]:
    """The v1 SEED (DISTRUST fallback): the same-RAM gather of B_op=256 scattered rows at 8 B/ns
    (B_op * 964 B / 8 B/ns ~= 30.85 us). sigma ~half (a contended/cache-cold gather is slower). Returns
    (mean, sigma, unit). This is the term SUBTRACTED from the gather-charged tau_io to get the elided headline
    (and ADDED to the elided headline to get the charged arm) in the model's copy_contrast()."""
    gather_us = _B_OP_SEED * _REQ_ROW_B / _GATHER_BW_BYTES_PER_NS / 1000.0
    return (gather_us, 0.5 * gather_us, "us")


def _measure_raw(b_rows: int = 256, cycles: int = 5000) -> dict[str, Any]:
    """The raw-pool PROVENANCE producer (the §6 Phase-4 internal helper): measure the same-RAM gather of `b_rows` scattered rows into one contiguous buffer. NO JAX, NO
    host->device (that is bench_..._tau_io_us). Returns {'gather_us_median', 'per_cycle_us', 'b_rows'}.
    Imports numpy lazily. Pin the process (taskset -c 0). `measure()` wraps the per-cycle pool into a median `Estimate`; `run()` uses it for BOTH the Estimate and the raw provenance rows (ONE measurement, two consumers — P1)."""
    import numpy as np

    B = b_rows
    slab_rows = B * 4
    slab = np.ones((slab_rows, _IN_DIM), dtype=np.float32)
    rng = np.random.default_rng(0)
    src_idx = rng.permutation(slab_rows)[:B]           # B scattered source rows
    out = np.empty((B, _IN_DIM), dtype=np.float32)

    for _ in range(min(200, cycles)):                  # warm caches + page table
        out[:] = slab[src_idx]

    def _one_cycle() -> float:
        """One same-RAM gather (B scattered rows -> contiguous) -> its us reading (the per-window
        measurement window_pool calls once per window)."""
        t0 = time.perf_counter_ns()
        out[:] = slab[src_idx]                          # the same-RAM gather (B scattered -> contiguous)
        return (time.perf_counter_ns() - t0) / 1000.0

    # window_pool owns the loop + the >= 2 floor (RCA fix #2): one reading per cycle, count == cycles.
    per_cycle_us = window_pool(_one_cycle, name=NAME, count=cycles)

    return {"gather_us_median": float(np.median(per_cycle_us)), "per_cycle_us": per_cycle_us, "b_rows": B}


def _estimate_from_raw(res: dict[str, Any]) -> "_est.Estimate":
    """Build this bench's harmonized `Estimate` from a `_measure_raw()` dict — the SINGLE home of the
    Estimate construction (P1), called by BOTH `measure()` and `run()`. A k=1 median `QuantileLaw(p=0.5)`
    with a BOOTSTRAP median SE over the pool (§7.A — the order-statistic variance, NOT s²/n),
    family=EMPIRICAL, kind='median'."""
    return median_estimate(res["per_cycle_us"], name=NAME)   # bootstrap median SE over the per-cycle pool


# Move 6: the shared scaffold wires register_self / measure / run from the bench-specific parts above.
# TUPLE seed (no .unit) — the explicit registered unit is passed via units="us". The cfg references the
# `cycles` run-knob, read from the scaffold-supplied kw (defaults applied) per pattern (b).
_B = _scaffold(
    name=NAME, quantity="serve_arena_gather_cost_cpp_inproc_port", module_path=MODULE_PATH, description=_DESC,
    units="us",
    seed=get_seed, measure_raw=_measure_raw, estimate_from_raw=_estimate_from_raw,
    run_config=lambda res, **kw: {"b_rows": res["b_rows"], "cycles": kw["cycles"], "transport": "cpp_inproc_port_direct_call",
           "kind": "arena_gather_contrast",
           "note": "the gather the contiguous-arena headline ELIDES; the swing term of tau_io's two arms",
           "gather_us_median": res["gather_us_median"]},
    run_log=lambda res, log, **kw: log(res["per_cycle_us"], sample_size=1),
)
register_self, measure, run = _B.register_self, _B.measure, _B.run


if __name__ == "__main__":
    _m, _s, _u = get_seed()
    print(f"[bench_cpp_inproc_port_gather_us] seed: {_m:.2f} {_u} (sigma {_s:.2f}) — FIRST-PRINCIPLES "
          f"(B_op={_B_OP_SEED} scattered rows at {_GATHER_BW_BYTES_PER_NS} B/ns)")
    register_self()
    print("[bench_cpp_inproc_port_gather_us] registered. NOT running the live measurement (timing-sensitive); "
          "invoke run() pinned (taskset -c 0). This is the DOMINANT transport DESIGN question for the inproc "
          "port: is the staging arena contiguous (gather elidable)?")
