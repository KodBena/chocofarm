"""
tools/analysis/leaf_eval_bound/benchmarks/bench_lockfree_mpsc_tmsg.py
==============================================================

LIVE benchmark for `lockfree_mpsc_tmsg_us_leaf` — the per-leaf-amortized MESSAGE cost (us/leaf)
for the LOCK-FREE MPSC transport: NOT a wire encode/decode (there is no frame envelope, no
corr-id, no `send_multipart`), but the ENQUEUE of one leaf's request node (a CAS that publishes
the slot + writes the feature row) plus the consumer-side read of its reply slot. This is the
TRANSPORT-stage term (request/reply CAPACITY), which is NON-BINDING by a wide margin (the binding
stage is the serialized serve), so it is reported but ranks LAST for the Neyman allocator — the
MPSC variant of the baseline `tmsg_us_leaf`, with the ZMQ memcpy-codec framing replaced by a
CAS-enqueue + a slot write/read.

WHAT run()/measure() MEASURES (1:1 with the model input). Time a single leaf's queue traffic: an
atomic tail-CAS (the enqueue publish) + write one request row (in_dim f32) into its reserved slot
+ read one reply row ((1+n_actions) f32) out of the reply slab — the per-leaf framing share with
no envelope, no syscall, no codec. Timed in WINDOWS (a single per-leaf perf_counter call is
clock-dominated), so the read is a POOL of per-window per-leaf us/leaf, whose headline is the pool
MEDIAN (the same windowing `bench_cpp_inproc_port_tmsg_us_leaf` / `bench_futex_wake_tmsg_us_leaf`
use). The SEED is the bare (req_row + rep_row) memcpy at a conservative bandwidth + one CAS
(~0.18 us/leaf), far below the per-forward budget, so transport never binds.

WHY SHRINKABLE NOW (the ADR-0008 reclassification this edit IS). The per-leaf queue traffic is a
MEASURED latency, not a config pin. The prior version of this module PUNTED — `_measure_raw()`
timed the real traffic but `_estimate_from_raw()` DISCARDED it and wrapped the v1 SEED (0.18
us/leaf) in `pin_estimate(...)` → an un-shrinkable `Fixed` Estimate — so the manifest TRUST path
held a re-declared seed the bench's own measurement contradicts (the measured ~0.9–1.5 us/leaf is
a >4x gap), and the Neyman loop could not sample it (a `Fixed` law's `marginal_dvar_deffort` is
0 → `A_i = 0` → never funded; the same stall `bench_r_gen.py` removed). This module now RUNS the
measurement and returns a SHRINKABLE `QuantileLaw` (median) Estimate over the per-window per-leaf
pool (a real bootstrap median SE — docs/design/harmonized-estimator-interface.md §7.A, §3 MEDIAN
row), so a longer `iters` budget → a tighter pool → a tighter SE the loop FUNDS it as the
DESIGN-PRIORITY transport DOF `_TRANSPORT_MOVED_TERMS['lockfree_mpsc']` names. It STILL ranks LAST
for the allocator's VARIANCE ranking (NON-BINDING by a wide margin — tmsg enters the model only as
the min() arm `1/(L*tmsg*1e-6)` ~2000+ dps while SERVE binds ~430 dps, so `df/dtmsg=0` and the
variance-contribution `a_i ~ 0` at the operating point regardless): both orderings stay honest —
the term is now FUNDABLE-when-asked yet correctly contributes ~0 variance to the binding CI.

WHY MEDIAN AND NOT A CONSTANT PIN (the classification call — ADR-0008). tmsg is a MEASURED latency
(demonstrably non-constant across runs — scheduler jitter), so its honest kind is `median`
(`QuantileLaw`/`EMPIRICAL`), NOT a `Fixed` pin: a measured latency with a runnable shrinkable bench
is the MEDIAN row of the harmonized interface, not the PIN row. A layout/deployment FACT (n_gen)
would be a `DEGENERATE` constant; a latency is not. ADR-0012 P8 (typed-signature-is-SSOT): the
Estimate family/shrink IS this function's contract, and declaring `Fixed`/`declared_spread` for a
quantity the bench actually TIMES is a lying signature against its own measurement (P1 single-home:
the live `_measure_raw` reading and the declared theta_hat are two disagreeing homes of one value).

`get_seed()` stays the DISTRUST fallback (the v1 ~0.18 us/leaf first-principles prior, a `Fixed`
declared-spread Estimate on the SEED path — the manifest's `trust=False` / pg-down route; only the
MEASURED path is shrinkable). The measurement is in-process (numpy + multiprocessing.shared_memory),
so it has NO external binary to gate on; a degenerate zero-spread pool fails LOUD in
`median_estimate` (ADR-0002), never a fabricated QuantileLaw. (A higher-fidelity number is the C++
`cpp/build/chocofarm-stage-a-transport-bench`; this in-process loop is a runnable measurement of the
quantity at the SAME fidelity the seed assumes — the CAS is a numpy int64 increment standing for a
real atomic tail-CAS, exactly the stand-in the seed's `_CAS_NS` already models.)

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

NAME = "lockfree_mpsc_tmsg_us_leaf"
MODULE_PATH = "leaf_eval_bound.benchmarks.bench_lockfree_mpsc_tmsg"
_DESC = ("LOCK-FREE MPSC per-leaf message cost (us/leaf): a tail-CAS enqueue + write one request row into "
         "the reserved slot + read one reply row out (no frame envelope, no corr-id, no syscall, no codec). "
         "Transport stage; NON-BINDING by a wide margin. The MPSC variant of tmsg_us_leaf (CAS-enqueue + "
         "slot write/read, no ZMQ multipart codec). MEASURED median over a per-window pool (shrinkable).")

_IN_DIM = 241
_N_ACTIONS = 65
_REQ_ROW_B = _IN_DIM * 4               # 964 B/row
_REP_ROW_B = (1 + _N_ACTIONS) * 4      # 264 B/row
_MEMCPY_BW_BYTES_PER_NS = 8.0          # CONSERVATIVE single-thread sequential memcpy (matches the tau_io bench)
_CAS_NS = 30.0                         # one uncontended enqueue CAS (cache-hot)
_WINDOW = 1000                         # leaves per timing window (the pool reading; same as cpp_inproc_port/futex_wake)


def get_seed() -> tuple[float, float, str]:
    """The v1 SEED (DISTRUST fallback) — first-principles: one leaf's queue traffic is (req_row + rep_row)
    bytes memcpy'd at a conservative 8 B/ns plus one enqueue CAS: (964 + 264)/8/1000 + 30/1000 ~= 0.18
    us/leaf. sigma 0.08us (bandwidth + CAS-contention spread). Non-binding by a wide margin. The DISTRUST
    fallback only (the SEED path); the MEASURED path (measure()/run()) is the shrinkable median. Returns
    (mean, sigma, unit)."""
    mean = (_REQ_ROW_B + _REP_ROW_B) / _MEMCPY_BW_BYTES_PER_NS / 1000.0 + _CAS_NS / 1000.0
    return (mean, 0.08, "us")


def _measure_raw(iters: int = 200000) -> dict[str, Any]:
    """The raw-pool PROVENANCE producer (the §6 Phase-4 internal helper): measure lockfree_mpsc_tmsg_us_leaf:
    over `iters` leaves, time one leaf's queue traffic — an atomic tail bump (the enqueue publish, a numpy
    int64 increment standing for the CAS) + write one request row into its slot + read one reply row out of
    the reply slab — NO envelope, NO syscall, NO codec. Timed in WINDOWS of `_WINDOW` leaves (a single-leaf
    `perf_counter` call is clock-dominated), so the read is a POOL of per-window per-leaf us/leaf (the same
    windowing `bench_cpp_inproc_port_tmsg_us_leaf._measure_raw` / `bench_futex_wake_tmsg_us_leaf` use).
    Returns {'tmsg_us_leaf_median' (the headline pool median), 'per_leaf_us' (the pool the Estimate is built
    over), 'iters'}. Imports numpy + shared_memory lazily. `iters` IS the shrink budget — more leaves → more
    windows → a tighter median SE. `measure()`/`run()` BOTH consume this ONE measurement (P1)."""
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
        # Time per-leaf in small windows; the headline is the median per-leaf over the windows (a single
        # per-leaf perf_counter call would be clock-dominated). window_pool owns the loop + the >= 2
        # window floor (RCA fix #2 — the >= 2 readings the bootstrap median SE needs; median_estimate
        # RAISES on a 1-reading pool, ADR-0002), count == iters // _WINDOW windows.
        def _one_window() -> float:
            """One window of `_WINDOW` per-leaf queue traffic iterations (tail-bump enqueue + slot write +
            reply read) -> the per-leaf us dt/_WINDOW (the per-window measurement window_pool calls once
            per window)."""
            t0 = time.perf_counter_ns()
            for _ in range(_WINDOW):
                req_slot[:] = one_req                # producer writes one request row into its reserved slot
                ctr[0] += 1                          # the enqueue publish (a tail bump; the CAS in the real queue)
                out_rep[:] = rep_slot                # consumer reads one reply row out of the reply slab
            dt = time.perf_counter_ns() - t0
            return dt / 1000.0 / _WINDOW

        per_leaf_us = window_pool(_one_window, name=NAME, count=iters // _WINDOW)
        return {"tmsg_us_leaf_median": float(np.median(per_leaf_us)),
                "per_leaf_us": per_leaf_us, "iters": iters}
    finally:
        for shm in (shm_req, shm_rep, shm_ctr):
            shm.close()
            shm.unlink()


def _estimate_from_raw(res: dict[str, Any]) -> "_est.Estimate":
    """Build this bench's harmonized SHRINKABLE `Estimate` from a `_measure_raw()` dict — the SINGLE home
    of the Estimate construction (P1), called by BOTH `measure()` and `run()`. A k=1 median
    `QuantileLaw(p=0.5)` with a BOOTSTRAP median SE over the per-window per-leaf pool (§7.A — the
    order-statistic variance, NOT s²/n), `family=EMPIRICAL`, `kind='median'`, POSITIVE support. This is the
    ADR-0008 reclassification: the per-leaf queue traffic is a MEASURED latency whose variance RESPONDS to
    effort (the median's `marginal_dvar_deffort` is `−cov/n < 0`), so the Neyman loop can FUND it, where the
    prior `Fixed` pin (marginal=0) made it un-fundable and held a re-declared seed. `get_seed()` stays the
    DISTRUST fallback (the SEED path), not the trusted Estimate."""
    return median_estimate(res["per_leaf_us"], name=NAME)   # bootstrap median SE over the per-leaf pool


# Move 6: the shared scaffold wires register_self / measure / run from the bench-specific parts above.
# `iters` sizes the measurement pool (the Neyman budget); TIMING-SENSITIVE — measure()/run() are operator-
# invoked, pinned (taskset -c 0), NEVER during the fan-out.
_B = _scaffold(
    name=NAME, quantity="transport_msg_cost_per_leaf_lockfree_mpsc", module_path=MODULE_PATH, description=_DESC, units="us",
    seed=get_seed, measure_raw=_measure_raw, estimate_from_raw=_estimate_from_raw,
    run_config=lambda res, **kw: {"iters": kw["iters"], "transport": "lockfree_mpsc_queue", "codec": "cas_enqueue_slot_write",
           "tmsg_us_leaf_median": res["tmsg_us_leaf_median"],
           "note": "tail-CAS enqueue + slot write of one request row + reply-slot read; no envelope, no syscall"},
    # PROVENANCE only (§5.2 de-dup): the headline median lives in estimate.theta_hat[0], not a sample row.
    run_log=lambda res, log, **kw: log(res["per_leaf_us"], sample_size=1),
)
register_self, measure, run = _B.register_self, _B.measure, _B.run


if __name__ == "__main__":
    _m, _s, _u = get_seed()
    print(f"[bench_lockfree_mpsc_tmsg] seed (DISTRUST fallback): {_m:.3f} {_u} (sigma {_s:.3f}) — "
          f"first-principles (bare slot memcpy of one req row + one reply row + one enqueue CAS; non-binding)")
    register_self()
    print("[bench_lockfree_mpsc_tmsg] registered. measure()/run() RUN the per-leaf CAS-enqueue + slot "
          "write/read (windowed) -> a SHRINKABLE median Estimate. get_seed() is the DISTRUST fallback. NOT "
          "running the live measurement here (timing-sensitive); invoke run() pinned (taskset -c 0), sole-workload.")
