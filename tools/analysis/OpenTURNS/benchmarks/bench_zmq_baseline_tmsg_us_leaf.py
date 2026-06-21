"""
tools/analysis/OpenTURNS/benchmarks/bench_zmq_baseline_tmsg_us_leaf.py
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

WHAT run() MEASURES (1:1 with the model input). Time `encode_request` over an S-row feature matrix
+ `decode_response` over the matching reply, divide by S — the per-leaf framing share of the ZMQ
codec. NO socket (the framing COST is the codec memcpy; the multipart send/recv cost is in
`zmq_baseline_tau_io_us`, the serve-side I/O term — keeping the two terms one-home each).

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

import leaf_eval_grounding as G  # noqa: E402
from bench_common import logged_run  # noqa: E402

NAME = "zmq_baseline_tmsg_us_leaf"
MODULE_PATH = "benchmarks.bench_zmq_baseline_tmsg_us_leaf"
_DESC = ("ZMQ BASELINE per-leaf-amortized message-passing cost (us/leaf): inference_wire request encode + "
         "reply decode (pure-memcpy codec) over a coalesced S-leaf frame, /S. Transport stage; NON-BINDING "
         "by a wide margin (ranks LAST for Neyman). Seed delegates to v1 G.MSG_PER_LEAF_US (1.0us/leaf).")

_IN_DIM = 241
_N_ACTIONS = 65


def get_seed() -> G.Grounded:
    """The v1 seed (DISTRUST fallback) — DELEGATED to the single home (ADR-0012 P1): `G.MSG_PER_LEAF_US`
    (1.0us/leaf, a deliberate over-charge; provably non-binding). zmq_baseline's tmsg IS the v1 grounding
    seed (it moves nothing off the reference codec)."""
    return G.MSG_PER_LEAF_US


def register_self() -> Any:
    from bench_common import register_quantity
    return register_quantity(NAME, quantity="transport_msg_cost_per_leaf", units=get_seed().unit,
                             description=_DESC, module_path=MODULE_PATH)


def measure(s_leaves: int = 256, iters: int = 5000) -> dict[str, Any]:
    """Measure the ZMQ-baseline tmsg_us_leaf: time encode_request(S x in_dim) + decode_response(reply) over
    `iters`, /S — the per-leaf framing share of the inference_wire memcpy codec. Returns {'tmsg_us_leaf',
    'encode_us', 'decode_us', 's_leaves'}. Imports the codec + numpy lazily. Pin (taskset -c 0)."""
    import numpy as np
    from chocofarm.az.inference_wire import encode_request, encode_response, decode_response

    feats = np.zeros((s_leaves, _IN_DIM), dtype=np.float32)
    reply = encode_response(np.zeros((s_leaves,), dtype=np.float32),
                            np.zeros((s_leaves, _N_ACTIONS), dtype=np.float32))
    # Warm.
    for _ in range(min(200, iters)):
        encode_request(feats); decode_response(reply)
    t0 = time.perf_counter_ns()
    for _ in range(iters):
        encode_request(feats)
    enc_us = (time.perf_counter_ns() - t0) / 1000.0 / iters
    t0 = time.perf_counter_ns()
    for _ in range(iters):
        decode_response(reply)
    dec_us = (time.perf_counter_ns() - t0) / 1000.0 / iters
    per_leaf = (enc_us + dec_us) / s_leaves
    return {"tmsg_us_leaf": per_leaf, "encode_us": enc_us, "decode_us": dec_us, "s_leaves": s_leaves}


def run(s_leaves: int = 256, iters: int = 5000) -> dict[str, Any]:
    """Measure the ZMQ-baseline tmsg_us_leaf and LOG it (the per-leaf headline + the encode/decode
    components). TIMING-SENSITIVE — operator-invoked, pinned, never during the fan-out."""
    res = measure(s_leaves=s_leaves, iters=iters)
    cfg = {"s_leaves": s_leaves, "iters": iters, "encode_us": res["encode_us"],
           "decode_us": res["decode_us"], "codec": "inference_wire_memcpy", "transport": "zmq_baseline"}
    with logged_run(NAME, quantity="transport_msg_cost_per_leaf", units=get_seed().unit, description=_DESC,
                    module_path=MODULE_PATH, config=cfg) as log:
        log(res["tmsg_us_leaf"], sample_size=iters)
    return res


if __name__ == "__main__":
    print(f"[bench_zmq_baseline_tmsg_us_leaf] seed: {get_seed().mean} {get_seed().unit} "
          f"(provenance: {get_seed().provenance}; delegated to v1 G.MSG_PER_LEAF_US)")
    register_self()
    print("[bench_zmq_baseline_tmsg_us_leaf] registered. NOT running the live measurement (timing-"
          "sensitive); invoke run() pinned and sole-workload. NON-BINDING — ranks LAST for Neyman.")
