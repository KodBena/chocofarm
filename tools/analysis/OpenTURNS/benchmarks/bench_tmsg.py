"""
tools/analysis/OpenTURNS/benchmarks/bench_tmsg.py
=================================================

LIVE benchmark for `tmsg_us_leaf` — the per-leaf-amortized MESSAGE-PASSING cost (us/leaf):
the request encode + reply decode of the inference wire codec, amortized over a coalesced
S-leaf frame. This is the TRANSPORT-stage term (request/reply CAPACITY), which is NON-BINDING
by a wide margin (the binding stage is the serialized serve). A transport variant changes the
per-leaf framing cost (a shared-memory ring or inproc port frames at ~0), so each variant MAY
register its own `<slug>_tmsg_us_leaf`; the baseline measures the ZMQ memcpy codec.

WHAT run() MEASURES (1:1 with the model input). Time `encode_request` over an S-row feature
matrix + `decode_response` over the matching reply, divide by S — the per-leaf framing share.
The SEED is the v1 placeholder (1.0 us/leaf, a deliberate over-charge that still leaves
transport non-binding — inference_wire.hpp pure-memcpy codec).

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

import leaf_eval_grounding as G  # noqa: E402
from bench_common import logged_run, pin_estimate  # noqa: E402

NAME = "tmsg_us_leaf"
MODULE_PATH = "benchmarks.bench_tmsg"
_DESC = ("Per-leaf-amortized message-passing cost (us/leaf): request encode + reply decode of the "
         "inference wire codec over a coalesced S-leaf frame, /S. Transport stage; NON-BINDING by a wide "
         "margin. Baseline = the ZMQ memcpy codec.")

_IN_DIM = 241
_N_ACTIONS = 65


def get_seed() -> G.Grounded:
    """The v1 seed (DISTRUST fallback): 1.0 us/leaf (deliberate over-charge; non-binding)."""
    return G.MSG_PER_LEAF_US


def register_self() -> Any:
    from bench_common import register_quantity
    return register_quantity(NAME, quantity="transport_msg_cost_per_leaf", units=get_seed().unit,
                             description=_DESC, module_path=MODULE_PATH)


def measure(s_leaves: int = 256, iters: int = 5000) -> dict[str, Any]:
    """Measure tmsg_us_leaf: time encode_request(S x in_dim) + decode_response(reply) over `iters`, /S.
    Returns {'tmsg_us_leaf', 'encode_us', 'decode_us', 's_leaves'}. Imports the codec + numpy lazily."""
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
    """Logs a harmonized k=1 Fixed Estimate (§6 Phase 3) recovering the declared spread un-divided, alongside the live measurement. TIMING-
    SENSITIVE — operator-invoked, pinned, never during the fan-out."""
    res = measure(s_leaves=s_leaves, iters=iters)
    est = pin_estimate(get_seed().mean, get_seed().sigma, name=NAME)
    cfg = {"s_leaves": s_leaves, "iters": iters, "encode_us": res["encode_us"],
           "decode_us": res["decode_us"], "codec": "inference_wire_memcpy"}
    with logged_run(NAME, quantity="transport_msg_cost_per_leaf", units=get_seed().unit, description=_DESC,
                    module_path=MODULE_PATH, config=cfg, estimate=est) as log:
        log(res["tmsg_us_leaf"], sample_size=iters)
    return res


if __name__ == "__main__":
    print(f"[bench_tmsg] seed: {get_seed().mean} {get_seed().unit} (provenance: {get_seed().provenance})")
    register_self()
    print("[bench_tmsg] registered. NOT running the live measurement (timing-sensitive); "
          "invoke run() pinned and sole-workload.")
