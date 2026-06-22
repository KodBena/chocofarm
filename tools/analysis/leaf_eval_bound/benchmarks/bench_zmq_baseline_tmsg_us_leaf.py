"""
tools/analysis/leaf_eval_bound/benchmarks/bench_zmq_baseline_tmsg_us_leaf.py
=====================================================================

LIVE benchmark for `zmq_baseline_tmsg_us_leaf` — the ZMQ BASELINE transport's per-leaf-amortized
MESSAGE-PASSING cost (us/leaf): the `inference_wire` request encode + reply decode (the pure-
memcpy codec, `chocofarm/az/inference_wire.py`; `cpp/include/chocofarm/inference_wire.hpp`)
amortized over a coalesced S-leaf frame. This is the TRANSPORT-stage term (request/reply
CAPACITY), NON-BINDING by a wide margin (~2000 dps ceiling vs the ~430 dps serve binding), so it
ranks LAST for the Neyman allocator — but the sweep reports it for every variant (a shm ring / an
inproc port frames at ~0), so zmq_baseline registers its own slug-prefixed `<slug>_tmsg_us_leaf`.

WHY THIS IS A zmq_baseline-PREFIXED PEER OF THE v1 `tmsg_us_leaf`. The v1 `tmsg_us_leaf`
(`bench_tmsg.py`) measures this same ZMQ-baseline memcpy codec — it IS the baseline. This module
is its transport-slug-prefixed twin so the design sweep is UNIFORM (every variant resolves a
`<slug>_tmsg_us_leaf`, no UNIQUE collision). ONE physical home for the SEED (the v1 grounding
`G.MSG_PER_LEAF_US`, 1.0us/leaf — a deliberate over-charge that still leaves transport non-
binding) — this module DELEGATES to it (ADR-0012 P1 single-home), never a second literal.

WHAT run()/measure() MEASURES (1:1 with the model input). Time `encode_request` over an S-row
feature matrix + `decode_response` over the matching reply, divide by S — the per-leaf framing
share of the ZMQ codec — in N windows, pooling a per-leaf reading per window so the headline is the
pool MEDIAN. NO socket (the framing COST is the codec memcpy; the multipart send/recv cost is in
`zmq_baseline_tau_io_us`, the serve-side I/O term — keeping the two terms one-home each).

WHY SHRINKABLE NOW (the ADR-0008 reclassification, mirroring the v1 bench_tmsg / R_gen fix; ADR-0012
P8 the typed signature IS the contract). `tmsg_us_leaf` is a MEASURED quantity
(`G.MSG_PER_LEAF_US.needs_measurement=True`, NOT a true constant — `Grounded.constant` defaults
False), not a config pin. The prior version PUNTED — it DID a real codec measurement but wrapped the
SEED (1.0us) in `pin_estimate(...)` → an un-shrinkable `Fixed`, discarding the live number. The
SAME-quantity-class sibling bench_cpp_inproc_port_tmsg_us_leaf (ALSO NON-BINDING / ranks-LAST) is
constructed SHRINKABLE — so non-binding is a RANKING fact, not an un-measurability fact. This module
now RUNS the codec and returns a SHRINKABLE `QuantileLaw` (median) Estimate over a per-leaf pool (a
real bootstrap median SE — docs/design/harmonized-estimator-interface.md §7.A, §3 the MEDIAN row +
the PIN-now/measurable-later row). `get_seed()` stays the DISTRUST fallback (the v1 1.0us/leaf prior
on the SEED path only). FAIL LOUD (ADR-0002): an import/codec failure propagates — never the
seed-as-if-measured.

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
from estimators import median_estimate  # noqa: E402
from pools import window_pool  # noqa: E402
from harness import logged_run  # noqa: E402

NAME = "zmq_baseline_tmsg_us_leaf"
MODULE_PATH = "benchmarks.bench_zmq_baseline_tmsg_us_leaf"
_DESC = ("ZMQ BASELINE per-leaf-amortized message-passing cost (us/leaf): inference_wire request encode + "
         "reply decode (pure-memcpy codec) over a coalesced S-leaf frame, /S (the pool MEDIAN over "
         "windows). Transport stage; NON-BINDING by a wide margin (ranks LAST for Neyman). Seed delegates "
         "to v1 G.MSG_PER_LEAF_US (1.0us/leaf).")

_IN_DIM = 241
_N_ACTIONS = 65
_S_LEAVES = 256        # the coalesced frame width (rows per encode_request / decode_response)
_FRAMES_PER_WINDOW = 200  # coalesced frames timed per window (a single-frame perf_counter call is clock-dominated)


def get_seed() -> G.Grounded:
    """The v1 seed (DISTRUST fallback) — DELEGATED to the single home (ADR-0012 P1): `G.MSG_PER_LEAF_US`
    (1.0us/leaf, a deliberate over-charge; provably non-binding). zmq_baseline's tmsg IS the v1 grounding
    seed (it moves nothing off the reference codec). Used by the SEED path (manifest `trust=False`); the
    MEASURED path is the shrinkable `QuantileLaw`."""
    return G.MSG_PER_LEAF_US


def register_self() -> Any:
    from harness import register_quantity
    return register_quantity(NAME, quantity="transport_msg_cost_per_leaf", units=get_seed().unit,
                             description=_DESC, module_path=MODULE_PATH)


def _measure_raw(budget: int = 64, s_leaves: int = _S_LEAVES) -> dict[str, Any]:
    """The raw-pool PROVENANCE producer (the §6 Phase-4 internal helper): RUN the live inference-wire
    codec sized by `budget`, and return the per-leaf ZMQ-baseline framing-cost pool plus provenance.
    Over `budget` windows, time `encode_request(S x in_dim) + decode_response(reply)` for
    `_FRAMES_PER_WINDOW` coalesced frames and record ONE per-leaf reading per window (dt/frames/S — the
    per-leaf framing share of the inference_wire memcpy codec), mirroring the v1 bench_tmsg /
    bench_cpp_inproc_port_tmsg_us_leaf window loop so the SAME quantity-class is constructed the SAME
    way. `budget` IS the shrink budget — more windows → a tighter median SE. Returns
    {'tmsg_us_leaf_median' (the pool MEDIAN), 'per_leaf_us' (the pool the Estimate is built over),
    'encode_us', 'decode_us' (informational split), 's_leaves', 'budget'}. Imports the codec + numpy
    lazily. `measure()`/`run()` both consume this ONE measurement (P1).

    FAIL LOUD (ADR-0002): an import/codec failure propagates — never the seed-as-if-measured (the punt
    this module removes). The seed is the DISTRUST fallback path (get_seed()), not a measured-result
    substitute."""
    import numpy as np
    from chocofarm.az.inference_wire import encode_request, encode_response, decode_response

    feats = np.zeros((s_leaves, _IN_DIM), dtype=np.float32)
    reply = encode_response(np.zeros((s_leaves,), dtype=np.float32),
                            np.zeros((s_leaves, _N_ACTIONS), dtype=np.float32))
    # Warm.
    for _ in range(min(200, _FRAMES_PER_WINDOW)):
        encode_request(feats); decode_response(reply)

    def _one_window() -> float:
        """One window: time `_FRAMES_PER_WINDOW` coalesced encode+decode frames -> the per-leaf framing
        share dt/frames/S (the per-window measurement window_pool calls once per window)."""
        t0 = time.perf_counter_ns()
        for _ in range(_FRAMES_PER_WINDOW):
            encode_request(feats)               # the request encode (S x in_dim -> frame)
            decode_response(reply)              # the reply decode (frame -> S values + logits)
        dt = time.perf_counter_ns() - t0
        return dt / 1000.0 / _FRAMES_PER_WINDOW / s_leaves

    # window_pool owns the loop + the >= 2 floor (RCA fix #2 — the >= 2 readings the bootstrap median SE
    # needs): one reading per window, count == budget windows.
    per_leaf_us = window_pool(_one_window, name=NAME, count=budget)
    # A per-frame enc/dec split (informational provenance only — the headline + the pool above are the
    # codec's full per-leaf framing share; this attributes it to encode vs decode at the same point).
    t0 = time.perf_counter_ns()
    for _ in range(_FRAMES_PER_WINDOW):
        encode_request(feats)
    enc_us = (time.perf_counter_ns() - t0) / 1000.0 / _FRAMES_PER_WINDOW
    t0 = time.perf_counter_ns()
    for _ in range(_FRAMES_PER_WINDOW):
        decode_response(reply)
    dec_us = (time.perf_counter_ns() - t0) / 1000.0 / _FRAMES_PER_WINDOW
    return {
        "tmsg_us_leaf_median": float(np.median(per_leaf_us)),
        "per_leaf_us": per_leaf_us,
        "encode_us": enc_us,
        "decode_us": dec_us,
        "s_leaves": s_leaves,
        "budget": len(per_leaf_us),   # the realized window count (== the floored max(2, budget) window_pool ran)
    }


def _estimate_from_raw(res: dict[str, Any]) -> "_est.Estimate":
    """Build this bench's harmonized SHRINKABLE `Estimate` — the SINGLE home of the Estimate
    construction (P1), called by BOTH `measure()` and `run()`. A k=1 median `QuantileLaw(p=0.5)` with a
    BOOTSTRAP median SE over the per-leaf ZMQ-baseline framing-cost pool (§7.A — a real order-statistic
    SE, NOT a `Fixed` pin), `family=EMPIRICAL`, `kind='median'`, POSITIVE support. The ADR-0008
    reclassification (mirroring the v1 bench_tmsg / R_gen fix): the median's `marginal_dvar_deffort` is
    `−cov/n < 0`, so the Neyman loop can FUND it, where the prior `Fixed` pin (marginal=0) made it
    un-fundable. SAME construction as the same-quantity-class sibling bench_cpp_inproc_port_tmsg_us_leaf
    (ADR-0012 P8: one quantity-class, one typed shrink signature)."""
    return median_estimate(res["per_leaf_us"], name=NAME)   # bootstrap median SE over the per-leaf pool


def measure(budget: int = 64, s_leaves: int = _S_LEAVES) -> "_est.Estimate":
    """Measure zmq_baseline_tmsg_us_leaf (RUN the live inference-wire codec) and return its harmonized
    k=1 SHRINKABLE median `Estimate` (§6 Phase 4: `measure()` returns the `Estimate` the bench DECLARES —
    the driver/untrusted_drive consume it directly). `budget` sizes the measurement pool (the Neyman
    loop's lever — more windows tightens the SE). TIMING-SENSITIVE — pinned (taskset -c 0); never during
    the fan-out."""
    return _estimate_from_raw(_measure_raw(budget=budget, s_leaves=s_leaves))


def run(budget: int = 64, s_leaves: int = _S_LEAVES) -> dict[str, Any]:
    """Measure zmq_baseline_tmsg_us_leaf (RUN the live inference-wire codec) and LOG it as a harmonized
    k=1 SHRINKABLE median Estimate (QuantileLaw p=0.5, bootstrap median SE, §6 Phase 3, §5.2 de-dup).
    Returns the raw provenance dict. TIMING-SENSITIVE — operator-invoked, pinned (taskset -c 0), NEVER
    during the fan-out."""
    res = _measure_raw(budget=budget, s_leaves=s_leaves)  # ONE measurement (Estimate + provenance)
    est = _estimate_from_raw(res)  # the SAME Estimate measure() returns (P1)
    cfg = {"kind": "inference_wire_codec_measured", "codec": "inference_wire_memcpy",
           "transport": "zmq_baseline", "s_leaves": res["s_leaves"], "budget": res["budget"],
           "frames_per_window": _FRAMES_PER_WINDOW, "encode_us": res["encode_us"],
           "decode_us": res["decode_us"], "tmsg_us_leaf_median": res["tmsg_us_leaf_median"]}
    with logged_run(NAME, quantity="transport_msg_cost_per_leaf", units=get_seed().unit, description=_DESC,
                    module_path=MODULE_PATH, config=cfg, estimate=est) as log:
        # PROVENANCE only (§5.2 de-dup): the headline median lives in estimate.theta_hat[0], not a sample row.
        log(res["per_leaf_us"], sample_size=1)
    return res


if __name__ == "__main__":
    print(f"[bench_zmq_baseline_tmsg_us_leaf] seed: {get_seed().mean} {get_seed().unit} "
          f"(DISTRUST fallback; provenance: {get_seed().provenance}; delegated to v1 G.MSG_PER_LEAF_US)")
    register_self()
    print("[bench_zmq_baseline_tmsg_us_leaf] registered. measure()/run() RUN the live inference-wire "
          "codec (taskset -c 0) -> a SHRINKABLE median Estimate. NON-BINDING — ranks LAST for Neyman.")
