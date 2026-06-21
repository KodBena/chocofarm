"""
tools/analysis/OpenTURNS/benchmarks/bench_zmq_baseline_tau_io_us.py
==================================================================

LIVE benchmark for `zmq_baseline_tau_io_us` — the ZMQ BASELINE transport's per-forward serial
serve-side I/O cost (us): the drain (recv_multipart(NOBLOCK) × T) + decode (inference_wire ×
T) + encode (× T) + scatter (send_multipart × T) the single-threaded inference server runs
SERIALLY BETWEEN forwards (`inference_server.py` `_drain` / `_scatter`; SYNTHESIS v2 §3.3
"forward k+1 cannot begin until scatter k completes — this is the coalescing engine"). It is
the DOMINANT term the transport design moves; for the zmq_baseline variant it is the reference
value every other transport (shm_spin_poll, futex_wake, lockfree_mpsc, cpp_inproc_port) is
measured AGAINST.

WHY THIS IS A zmq_baseline-PREFIXED PEER OF THE v1 `tau_io_us`. The v1 `tau_io_us`
(`bench_tau_io.py`) measures this same ZMQ-baseline physics — it IS the baseline. This module
is its transport-slug-prefixed twin so the design sweep is UNIFORM: every variant resolves a
`<slug>_tau_io_us` (no UNIQUE-constraint collision across the fan-out), and the comparison
table has a zmq_baseline row of the same shape as the others. There is ONE physical home for
the SEED (the v1 grounding `G.SERVE_IO_US`, 20us) — this module DELEGATES to it rather than
re-littering a second literal (ADR-0012 P1 single-home). The MEASUREMENT is the ZMQ-baseline
serve loop, identical in mechanism to `bench_tau_io.py` (so a real run of either populates the
same physical quantity; they differ only in the registry name the sweep keys on).

WAKEUP NOTE. The ZMQ first-request wakeup (poll-syscall + fd readiness-notify) is a SEPARATE
named quantity `zmq_baseline_wakeup_us` (`bench_zmq_baseline_wakeup_us.py`) so the sweep can
contrast wakeup mechanisms across variants. At SATURATION (the regime R2 this bound models) a
request is already queued when the loop polls, so the per-forward wakeup is the poll-on-ready
cost (~1-2us), naturally exercised inside this drain loop's `recv_multipart(NOBLOCK)` path; the
separate quantity isolates it for the cross-variant comparison without double-charging the
cycle (the model adds tau_io + wakeup; this bench's tau_io is the I/O proper, the readiness
cost the wakeup bench measures is the first-poll latency BEFORE the drain).

WHAT run() MEASURES (1:1 with the model input). A sole-workload microbench of the serve loop's
I/O ONLY: bind a ROUTER, have T DEALER drivers each send a coalesced request frame, then time
the server's recv_multipart(NOBLOCK) drain + inference_wire decode + inference_wire encode +
send_multipart scatter over all T messages — WITHOUT a forward (the forward is iota+t_row*B,
measured separately by bench_iota / bench_t_row). The per-forward tau_io is that I/O time for
one drain cycle. run() logs per-cycle us readings.

TIMING-SENSITIVE — DO NOT run() during the parallel fan-out. Pin: `taskset -c 0`, sole-workload.

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
import leaf_eval_grounding as G  # noqa: E402
from bench_common import logged_run, median_estimate  # noqa: E402

NAME = "zmq_baseline_tau_io_us"
MODULE_PATH = "benchmarks.bench_zmq_baseline_tau_io_us"
_DESC = ("ZMQ BASELINE per-forward serial serve-side I/O (us): drain(recv_multipart(NOBLOCK) x T)+"
         "decode(x T)+encode(x T)+scatter(send_multipart x T) the single-threaded server runs between "
         "forwards (inference_server.py _drain/_scatter). The DOMINANT transport lever; the reference "
         "every other transport variant is measured against. Seed delegates to v1 G.SERVE_IO_US (20us).")

# Production geometry (matches bench_tau_io / bench_t_row): the coalesced-frame width (rows/forward) the
# drain sees split across T producer messages, and the feature / action widths the wire codec frames.
_IN_DIM = 241
_N_ACTIONS = 65


def get_seed() -> G.Grounded:
    """The v1 seed (DISTRUST fallback) — DELEGATED to the single home (ADR-0012 P1): the UNMEASURED
    baseline tau_io estimate `G.SERVE_IO_US` (20us, wide sigma, flagged needs-measurement). zmq_baseline
    MOVES NOTHING off the reference, so its tau_io seed IS the v1 grounding seed."""
    return G.SERVE_IO_US


def register_self() -> Any:
    from bench_common import register_quantity
    return register_quantity(NAME, quantity="serve_transport_io_cost", units=get_seed().unit,
                             description=_DESC, module_path=MODULE_PATH)


def _measure_raw(n_msgs: int = 8, rows_per_msg: int = 32, cycles: int = 2000) -> dict[str, Any]:
    """The raw-pool PROVENANCE producer (the §6 Phase-4 internal helper): measure the ZMQ-baseline tau_io —
    time one drain+decode+encode+scatter cycle over `n_msgs` coalesced producer messages of `rows_per_msg`
    rows each (a forward sees n_msgs*rows_per_msg rows). NO JAX forward (the forward is iota+t_row*B,
    separate). Uses inproc ZMQ (the multipart + codec cost without a NIC, isolating the serial serve I/O the
    production loop pays). Mirrors `inference_server._drain` (greedy NOBLOCK drain) + `_scatter`
    (send_multipart per drained request). Returns {'tau_io_us_median', 'per_cycle_us': [...], ...}.
    `measure()` wraps the per-cycle pool into a median `Estimate`; `run()` uses it for BOTH the Estimate and
    the raw provenance rows (ONE measurement, two consumers — P1). Imports zmq + numpy lazily. Pin (taskset -c 0)."""
    import numpy as np
    import zmq
    from chocofarm.az.inference_wire import encode_request, decode_request  # the batched codec SSOT

    ctx = zmq.Context.instance()
    router = ctx.socket(zmq.ROUTER)
    ep = "inproc://zmq_baseline_tau_io_bench"
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
    per_cycle_us: list[float] = []
    try:
        for _ in range(cycles):
            for d in dealers:                       # producers send their parked frame
                d.send(req_payload)
            t0 = time.perf_counter_ns()
            # DRAIN: recv_multipart(NOBLOCK) every queued frame; DECODE each; ENCODE a reply; SCATTER it —
            # the exact _drain + _scatter mechanism (greedy NOBLOCK pass; send_multipart per request).
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
            per_cycle_us.append((time.perf_counter_ns() - t0) / 1000.0)
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
    """Build this bench's harmonized `Estimate` from a `_measure_raw()` dict — the SINGLE home of the
    Estimate construction (P1), called by BOTH `measure()` and `run()`. A k=1 median `QuantileLaw(p=0.5)`
    with a BOOTSTRAP median SE over the per-cycle pool (§7.A — the order-statistic variance, NOT s²/n),
    family=EMPIRICAL, kind='median'."""
    return median_estimate(res["per_cycle_us"], name=NAME)


def measure(n_msgs: int = 8, rows_per_msg: int = 32, cycles: int = 2000) -> "_est.Estimate":
    """Measure the ZMQ-baseline tau_io and return its harmonized k=1 median `Estimate` (§6 Phase 4:
    `measure()` returns the `Estimate` the bench DECLARES — the driver/untrusted_drive consume it directly,
    no guessing which list is the pool). The raw per-cycle pool is the bench's internal `_measure_raw()`
    provenance. TIMING-SENSITIVE — pin the process (taskset -c 0)."""
    return _estimate_from_raw(_measure_raw(n_msgs=n_msgs, rows_per_msg=rows_per_msg, cycles=cycles))


def run(n_msgs: int = 8, rows_per_msg: int = 32, cycles: int = 2000) -> dict[str, Any]:
    """Measure the ZMQ-baseline tau_io and LOG it to postgres as a harmonized k=1 median Estimate (QuantileLaw
    p=0.5, bootstrap median SE, §6 Phase 3, §5.2 de-dup). TIMING-SENSITIVE — operator-invoked, pinned, never
    during the fan-out."""
    res = _measure_raw(n_msgs=n_msgs, rows_per_msg=rows_per_msg, cycles=cycles)  # ONE measurement (Est + prov)
    est = _estimate_from_raw(res)                          # the SAME Estimate measure() returns (P1)
    cfg = {"n_msgs": res["n_msgs"], "rows_per_msg": res["rows_per_msg"],
           "rows_per_forward": res["rows_per_forward"], "cycles": cycles,
           "transport": "zmq_baseline_router_dealer_inproc", "mechanism": "poll+recv_multipart(NOBLOCK)+send_multipart",
           "tau_io_us_median": res["tau_io_us_median"]}
    with logged_run(NAME, quantity="serve_transport_io_cost", units=get_seed().unit, description=_DESC,
                    module_path=MODULE_PATH, config=cfg, estimate=est) as log:
        # PROVENANCE only (§5.2 de-dup): the headline median lives in estimate.theta_hat[0], not a sample row.
        log(res["per_cycle_us"], sample_size=1)                 # raw per-cycle readings
    return res


if __name__ == "__main__":
    print(f"[bench_zmq_baseline_tau_io_us] seed: {get_seed().mean} {get_seed().unit} "
          f"(UNMEASURED — {get_seed().provenance}; delegated to v1 G.SERVE_IO_US)")
    register_self()
    print("[bench_zmq_baseline_tau_io_us] registered. NOT running the live measurement (timing-sensitive); "
          "invoke run() pinned and sole-workload. This is the TOP Neyman target for zmq_baseline.")
