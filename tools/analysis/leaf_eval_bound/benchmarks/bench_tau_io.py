"""
tools/analysis/leaf_eval_bound/benchmarks/bench_tau_io.py
===================================================

LIVE benchmark for `tau_io_us` — the SERVER-side per-forward serial TRANSPORT cost (us): the
drain (recv_multipart × T) + decode (× T) + encode (× T) + scatter (send_multipart × T) the
single-threaded inference server runs BETWEEN forwards (inference_server.py `_drain`/`_scatter`;
SYNTHESIS v2 §3.3 "drain k+1 cannot begin until scatter k completes"). It is the term the
TRANSPORT DESIGN MOVES — the ZMQ baseline pays the full multipart recv/send + codec per
coalesced frame; a shared-memory / futex / inproc-port variant pays a different (often far
smaller) tau_io. THIS module measures the ZMQ BASELINE; each transport variant registers its
OWN `<slug>_tau_io_us` (the prefixed convention) measuring its mechanism.

UNMEASURED in v1 (the brief's "missing Stage-4 term"; seed = 20us with a wide sigma, flagged
needs-measurement). It sits in the BINDING serve stage, so the Neyman allocator ranks it FIRST
— making a real measurement the highest-value bench.

WHAT run() MEASURES (1:1 with the model input). A sole-workload microbench of the serve loop's
I/O ONLY: bind a ROUTER, have a driver send a coalesced T-message request frame, then time the
server's recv_multipart(NOBLOCK) drain + inference_wire decode + inference_wire encode +
send_multipart scatter — WITHOUT a forward (the forward is iota+t_row*B, measured separately).
The per-forward tau_io is that I/O time amortized over one drain cycle. run() logs per-cycle
us readings.

TIMING-SENSITIVE — DO NOT run() during the parallel fan-out. Pin: `taskset -c 0`.

Public Domain (The Unlicense).
"""
from __future__ import annotations

import os
import sys
import time
from typing import Any, Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.dirname(_HERE), _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import estimate as _est  # noqa: E402  — the harmonized Estimate contract (measure() returns one — §6 Phase 4)
import leaf_eval_grounding as G  # noqa: E402
from estimators import median_estimate  # noqa: E402
from pools import window_pool  # noqa: E402
from harness import logged_run  # noqa: E402

NAME = "tau_io_us"
MODULE_PATH = "benchmarks.bench_tau_io"
_DESC = ("SERVER per-forward serial transport cost (us): drain(recv x T)+decode(x T)+encode(x T)+"
         "scatter(send x T) the single-threaded server runs between forwards (ZMQ BASELINE). The term "
         "the transport design moves; binding-stage, top Neyman priority. UNMEASURED in v1.")

# Production geometry (matches bench_t_row): the coalesced-frame width (rows/forward) the drain sees and
# the feature width. A full bucket is B_op rows split across T producer messages.
_IN_DIM = 241
_N_ACTIONS = 65


def get_seed() -> G.Grounded:
    """The v1 seed (DISTRUST fallback): the UNMEASURED tau_io estimate, 20us (wide sigma, needs-measurement)."""
    return G.SERVE_IO_US


def register_self() -> Any:
    from harness import register_quantity
    return register_quantity(NAME, quantity="serve_transport_io_cost", units=get_seed().unit,
                             description=_DESC, module_path=MODULE_PATH)


def _measure_raw(n_msgs: int = 8, rows_per_msg: int = 32, cycles: int = 2000) -> dict[str, Any]:
    """The raw-pool PROVENANCE producer (the §6 Phase-4 internal helper): measure tau_io on the ZMQ baseline
    — time one drain+decode+encode+scatter cycle over `n_msgs` coalesced producer messages of `rows_per_msg`
    rows each (so a forward sees n_msgs*rows_per_msg rows). NO JAX forward — the forward is iota+t_row*B
    (separate). Uses inproc ZMQ (the codec + multipart cost without a real NIC, isolating the serial-serve
    I/O). Returns {'tau_io_us_median', 'per_cycle_us': [...]}. `measure()` wraps the per-cycle pool into a
    median `Estimate`; `run()` uses it for BOTH the Estimate and the raw provenance rows (ONE measurement,
    two consumers — P1). Imports zmq + numpy lazily. Pin the process (taskset -c 0)."""
    import numpy as np
    import zmq
    from chocofarm.az.inference_wire import encode_request, decode_request  # the batched codec SSOT

    ctx = zmq.Context.instance()
    router = ctx.socket(zmq.ROUTER)
    ep = "inproc://tau_io_bench"
    router.bind(ep)
    dealers = []
    for i in range(n_msgs):
        d = ctx.socket(zmq.DEALER)
        d.setsockopt(zmq.IDENTITY, f"p{i}".encode())
        d.connect(ep)
        dealers.append(d)

    # Pre-encode each producer's coalesced request frame (rows_per_msg leaves of in_dim features).
    feats = np.zeros((rows_per_msg, _IN_DIM), dtype=np.float32)
    req_payload = encode_request(feats)

    def _one_cycle() -> float:
        """One drain+decode+encode+scatter cycle over the n_msgs coalesced frames -> its us reading
        (the per-window measurement window_pool calls once per window)."""
        for d in dealers:                       # producers send their parked frame
            d.send(req_payload)
        t0 = time.perf_counter_ns()
        # DRAIN: recv_multipart(NOBLOCK) every queued frame; DECODE each; ENCODE a reply; SCATTER it.
        drained = 0
        while drained < n_msgs:
            try:
                frames = router.recv_multipart(flags=zmq.NOBLOCK)
            except zmq.Again:
                continue
            ident, payload = frames[0], frames[-1]
            req = decode_request(payload)           # decode x 1 (of T)
            b = req.shape[0] if hasattr(req, "shape") else rows_per_msg
            reply = _encode_reply(b)                 # encode x 1 (of T)
            router.send_multipart([ident, reply])    # scatter x 1 (of T)
            drained += 1
        return (time.perf_counter_ns() - t0) / 1000.0

    try:
        # window_pool owns the loop + the >= 2 floor (RCA fix #2): one reading per cycle, count == cycles.
        per_cycle_us = window_pool(_one_cycle, name=NAME, count=cycles)
    finally:
        for d in dealers:
            d.close(linger=0)
        router.close(linger=0)
    med = float(np.median(per_cycle_us))
    return {"tau_io_us_median": med, "per_cycle_us": per_cycle_us,
            "n_msgs": n_msgs, "rows_per_msg": rows_per_msg, "rows_per_forward": n_msgs * rows_per_msg}


def _encode_reply(b: int) -> bytes:
    """Encode a batched response frame for b rows (value + logits per row), matching the wire codec shape
    [ver][B][n_actions][b x (value, logits)]. Falls back to a raw byte frame of the right LENGTH if the
    Python codec's reply encoder is not importable — the LENGTH (not the field values) is what the I/O
    cost depends on, so a length-faithful frame measures the same send cost."""
    import numpy as np
    try:
        from chocofarm.az.inference_wire import encode_response
        return encode_response(np.zeros((b,), dtype=np.float32),
                               np.zeros((b, _N_ACTIONS), dtype=np.float32))
    except Exception:
        # Length-faithful fallback: 1 ver byte + 8 count bytes + b*(1+n_actions)*4 f32 bytes.
        return bytes(1 + 8 + b * (1 + _N_ACTIONS) * 4)


def _estimate_from_raw(res: dict[str, Any]) -> "_est.Estimate":
    """Build tau_io's harmonized `Estimate` from a `_measure_raw()` dict — the SINGLE home of the Estimate
    construction (P1), called by BOTH `measure()` and `run()`. A k=1 median `QuantileLaw(p=0.5)` with a
    BOOTSTRAP median SE over the per-cycle pool (§7.A — the order-statistic variance, NOT s²/n),
    `family=EMPIRICAL`, `kind='median'`."""
    return median_estimate(res["per_cycle_us"], name=NAME)   # bootstrap median SE over the per-cycle pool


def measure(n_msgs: int = 8, rows_per_msg: int = 32, cycles: int = 2000) -> "_est.Estimate":
    """Measure tau_io (ZMQ baseline) and return its harmonized k=1 median `Estimate` (§6 Phase 4:
    `measure()` returns the `Estimate` the bench DECLARES — the driver/untrusted_drive consume it directly,
    no guessing which list is the per-cycle pool). The raw per-cycle pool is the bench's internal
    `_measure_raw()` provenance. TIMING-SENSITIVE — pin the process (taskset -c 0)."""
    return _estimate_from_raw(_measure_raw(n_msgs=n_msgs, rows_per_msg=rows_per_msg, cycles=cycles))


def run(n_msgs: int = 8, rows_per_msg: int = 32, cycles: int = 2000) -> dict[str, Any]:
    """Measure tau_io (ZMQ baseline) and LOG it to postgres as a harmonized k=1 median `Estimate` (§6
    Phase 3): `QuantileLaw(p=0.5)` with a BOOTSTRAP median SE over the per-cycle pool (§7.A — the
    order-statistic variance, NOT s²/n), `family=EMPIRICAL`, `kind='median'`. The per-cycle readings are
    logged as raw PROVENANCE — the variance authority is now `estimate.cov`, so the headline median
    scalar is NO LONGER double-logged as a sample row (the §5.2 de-dup obligation: tau_io previously
    wrote the median AND ~2000 readings into one instance, corrupting `latest_aggregate`'s count).
    TIMING-SENSITIVE — operator-invoked, pinned, never during the fan-out."""
    res = _measure_raw(n_msgs=n_msgs, rows_per_msg=rows_per_msg, cycles=cycles)  # ONE measurement (Est + prov)
    est = _estimate_from_raw(res)                           # the SAME Estimate measure() returns (P1)
    cfg = {"n_msgs": res["n_msgs"], "rows_per_msg": res["rows_per_msg"],
           "rows_per_forward": res["rows_per_forward"], "cycles": cycles, "transport": "zmq_inproc_router",
           "tau_io_us_median": res["tau_io_us_median"]}
    with logged_run(NAME, quantity="serve_transport_io_cost", units=get_seed().unit, description=_DESC,
                    module_path=MODULE_PATH, config=cfg, estimate=est) as log:
        # PROVENANCE only (§5.2): the raw per-cycle readings. The headline median is NOT logged as a
        # sample — it lives in estimate.theta_hat[0] (the SSOT), the median SE in estimate.cov.
        log(res["per_cycle_us"], sample_size=1)
    return res


if __name__ == "__main__":
    print(f"[bench_tau_io] seed: {get_seed().mean} {get_seed().unit} (UNMEASURED — {get_seed().provenance})")
    register_self()
    print("[bench_tau_io] registered. NOT running the live measurement (timing-sensitive); "
          "invoke run() pinned and sole-workload. This is the TOP Neyman target.")
