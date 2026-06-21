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

WHAT run()/measure() MEASURES (1:1 with the model input). Time a single leaf's ring traffic: copy
one request row (in_dim f32) into the request ring + copy one reply row ((1+n_actions) f32) out of
the reply ring — the per-leaf framing share with no envelope, no syscall. Timed in WINDOWS (a single
per-leaf perf_counter call is clock-dominated), so the read is a POOL of per-window per-leaf
us/leaf, whose headline is the pool MEDIAN (the same windowing cpp_inproc_port_tmsg_us_leaf uses).

WHY SHRINKABLE NOW (the ADR-0008 reclassification this edit IS). The per-leaf ring memcpy is a
MEASURED quantity, not a config pin. The prior version of this module PUNTED — `_measure_raw()`
timed the real ring traffic but `_estimate_from_raw()` DISCARDED it and wrapped the v1 SEED (0.15
us/leaf) in `pin_estimate(...)` → an un-shrinkable `Fixed` Estimate — so the manifest TRUST path
held a re-declared seed the bench's own measurement contradicts (the measured ~0.5 us/leaf), and the
Neyman loop could not sample it (a `Fixed` law's `marginal_dvar_deffort` is 0 → `A_i = 0` → never
funded; the same stall `bench_r_gen.py` removed). This module now RUNS the measurement and returns a
SHRINKABLE `QuantileLaw` (median) Estimate over the per-window per-leaf pool (a real bootstrap median
SE — docs/design/harmonized-estimator-interface.md §7.A, §3 MEDIAN row), so a longer `iters` budget
→ a tighter pool → a tighter SE the loop FUNDS. It still ranks LAST for the allocator (NON-BINDING by
a wide margin — the binding stage is the serialized serve), so the practical bound impact is small;
the fix is that the design's operator-run → fundable contract is now satisfiable for this term.

`get_seed()` stays the DISTRUST fallback (the v1 ~0.15 us/leaf first-principles prior, a `Fixed`
declared-spread Estimate on the SEED path — the manifest's `trust=False` / pg-down route; only the
MEASURED path is shrinkable). The measurement is in-process (numpy + multiprocessing.shared_memory),
so it has NO external binary to gate on; a degenerate zero-spread pool fails LOUD in
`median_estimate` (ADR-0002), never a fabricated QuantileLaw.

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
_WINDOW = 1000                         # leaves per timing window (the pool reading; same as cpp_inproc_port)


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


def _measure_raw(iters: int = 200000) -> dict[str, Any]:
    """The raw-pool PROVENANCE producer (the §6 Phase-4 internal helper): measure futex_wake_tmsg_us_leaf:
    over `iters` leaves, time one leaf's ring traffic — copy one request row into the request ring + one
    reply row out of the reply ring — NO envelope, NO syscall. Timed in WINDOWS of `_WINDOW` leaves (a
    single-leaf `perf_counter` call is clock-dominated), so the read is a POOL of per-window per-leaf
    us/leaf (the same windowing `bench_cpp_inproc_port_tmsg_us_leaf._measure_raw` uses). Returns
    {'tmsg_us_leaf_median' (the headline pool median), 'per_leaf_us' (the pool the Estimate is built
    over), 'iters'}. Imports numpy + shared_memory lazily. `iters` IS the shrink budget — more leaves →
    more windows → a tighter median SE. `measure()`/`run()` BOTH consume this ONE measurement (P1)."""
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
        # Time per-leaf in small windows; the headline is the median per-leaf over the windows (a single
        # per-leaf perf_counter call would be clock-dominated). window_pool owns the loop + the >= 2
        # window floor (RCA fix #2 — the >= 2 readings the bootstrap median SE needs; median_estimate
        # RAISES on a 1-reading pool, ADR-0002), count == iters // _WINDOW windows.
        def _one_window() -> float:
            """One window of `_WINDOW` per-leaf ring memcpies -> the per-leaf us dt/_WINDOW (the per-window
            measurement window_pool calls once per window)."""
            t0 = time.perf_counter_ns()
            for _ in range(_WINDOW):
                req_slot[:] = one_req                # producer writes one request row into the ring
                out_rep[:] = rep_slot                # consumer reads one reply row out of the ring
            dt = time.perf_counter_ns() - t0
            return dt / 1000.0 / _WINDOW

        per_leaf_us = window_pool(_one_window, name=NAME, count=iters // _WINDOW)
        return {"tmsg_us_leaf_median": float(np.median(per_leaf_us)),
                "per_leaf_us": per_leaf_us, "iters": iters}
    finally:
        for shm in (shm_req, shm_rep):
            shm.close()
            shm.unlink()


def _estimate_from_raw(res: dict[str, Any]) -> "_est.Estimate":
    """Build this bench's harmonized SHRINKABLE `Estimate` from a `_measure_raw()` dict — the SINGLE home
    of the Estimate construction (P1), called by BOTH `measure()` and `run()`. A k=1 median
    `QuantileLaw(p=0.5)` with a BOOTSTRAP median SE over the per-window per-leaf pool (§7.A — the
    order-statistic variance, NOT s²/n), `family=EMPIRICAL`, `kind='median'`, POSITIVE support. This is
    the ADR-0008 reclassification: the per-leaf ring memcpy is a MEASURED quantity whose variance RESPONDS
    to effort (the median's `marginal_dvar_deffort` is `−cov/n < 0`), so the Neyman loop can FUND it, where
    the prior `Fixed` pin (marginal=0) made it un-fundable and held a re-declared seed. `get_seed()` stays
    the DISTRUST fallback (the SEED path), not the trusted Estimate."""
    return median_estimate(res["per_leaf_us"], name=NAME)   # bootstrap median SE over the per-leaf pool


def measure(iters: int = 200000) -> "_est.Estimate":
    """Measure futex_wake_tmsg_us_leaf (time the per-leaf ring memcpy in windows) and return its harmonized
    k=1 SHRINKABLE median `Estimate` (§6 Phase 4: `measure()` returns the `Estimate` the bench DECLARES —
    a `QuantileLaw(p=0.5)` over the per-window per-leaf pool, consumed directly by the driver/untrusted_drive).
    `iters` sizes the measurement pool (the budget the Neyman loop passes — more leaves → more windows → a
    tighter SE). The raw pool is the bench's internal `_measure_raw()` provenance. TIMING-SENSITIVE — pin
    the process (taskset -c 0)."""
    return _estimate_from_raw(_measure_raw(iters=iters))


def run(iters: int = 200000) -> dict[str, Any]:
    """Measure futex_wake_tmsg_us_leaf and LOG it as a harmonized k=1 SHRINKABLE median Estimate
    (`QuantileLaw(p=0.5)`, BOOTSTRAP median SE over the per-window per-leaf pool, §6 Phase 3, §5.2 de-dup).
    TIMING-SENSITIVE — operator-invoked, pinned (taskset -c 0), NEVER during the fan-out."""
    res = _measure_raw(iters=iters)  # ONE measurement (Estimate + provenance pool)
    est = _estimate_from_raw(res)  # the SAME Estimate measure() returns (P1)
    cfg = {"iters": iters, "transport": "shm_ring_futex_wake", "codec": "bare_ring_memcpy",
           "tmsg_us_leaf_median": res["tmsg_us_leaf_median"],
           "note": "in-ring memcpy of one request row in + one reply row out; no envelope, no syscall"}
    with logged_run(NAME, quantity="transport_msg_cost_per_leaf_futex_wake", units="us", description=_DESC,
                    module_path=MODULE_PATH, config=cfg, estimate=est) as log:
        # PROVENANCE only (§5.2 de-dup): the headline median lives in estimate.theta_hat[0], not a sample row.
        log(res["per_leaf_us"], sample_size=1)
    return res


if __name__ == "__main__":
    _m, _s, _u = get_seed()
    print(f"[bench_futex_wake_tmsg_us_leaf] seed (DISTRUST fallback): {_m:.3f} {_u} (sigma {_s:.3f}) — "
          f"first-principles (bare ring memcpy of one req row in + one reply row out; non-binding; same "
          f"ring as shm)")
    register_self()
    print("[bench_futex_wake_tmsg_us_leaf] registered. measure()/run() RUN the per-leaf ring memcpy "
          "(windowed) -> a SHRINKABLE median Estimate. get_seed() is the DISTRUST fallback. NOT running "
          "the live measurement here (timing-sensitive); invoke run() pinned (taskset -c 0), sole-workload.")
