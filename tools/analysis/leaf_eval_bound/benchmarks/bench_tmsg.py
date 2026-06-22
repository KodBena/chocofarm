"""
tools/analysis/leaf_eval_bound/benchmarks/bench_tmsg.py
=================================================

LIVE benchmark for `tmsg_us_leaf` — the per-leaf-amortized MESSAGE-PASSING cost (us/leaf):
the request encode + reply decode of the inference wire codec (`chocofarm/az/inference_wire.py`,
the pure-memcpy codec), amortized over a coalesced S-leaf frame. This is the TRANSPORT-stage term
(request/reply CAPACITY), which is NON-BINDING by a wide margin (the binding stage is the
serialized serve). A transport variant changes the per-leaf framing cost (a shared-memory ring or
inproc port frames at ~0), so each variant MAY register its own `<slug>_tmsg_us_leaf`; the
baseline measures the ZMQ memcpy codec.

WHAT run()/measure() MEASURES (1:1 with the model input). Time `encode_request` over an S-row
feature matrix + `decode_response` over the matching reply, divide by S — the per-leaf framing
share — in N windows, pooling a per-leaf reading per window so the headline is the pool MEDIAN.

WHY SHRINKABLE NOW (the ADR-0008 reclassification this module IS; ADR-0012 P8 the typed signature
IS the contract). `tmsg_us_leaf` is a MEASURED quantity (`G.MSG_PER_LEAF_US.needs_measurement=True`,
NOT a true constant: `Grounded.constant` defaults False — the single-home SSOT of the
true-constant-vs-declared-spread split, leaf_eval_grounding.py), not a config pin. The prior version
of this module PUNTED — `_measure_raw()` DID a real codec measurement but `_estimate_from_raw()`
wrapped the SEED (1.0us) in `pin_estimate(...)` → an un-shrinkable `Fixed`/declared-spread Estimate,
and the live number was logged only "alongside" and DISCARDED — so the Neyman loop could not sample
it (a `Fixed` law's `marginal_dvar_deffort` is 0 → `A_i = 0` → never funded; it is the TRANSPORT arm
of the model's `min()`, funded only under the §4.1 kink regime). This is EXACTLY the R_gen defect
(bench_r_gen.py) and the SAME-quantity-class sibling bench_cpp_inproc_port_tmsg_us_leaf.py — which
ALSO measures `transport_msg_cost_per_leaf` and is ALSO NON-BINDING / ranks-LAST, yet constructs it
SHRINKABLE (a `median_estimate` QuantileLaw): so non-binding is a RANKING fact, not an
un-measurability fact. This module now RUNS the real codec measurement and returns a SHRINKABLE
`QuantileLaw` (median) Estimate over a pool of per-leaf readings (a real bootstrap median SE —
docs/design/harmonized-estimator-interface.md §7.A, §3 the MEDIAN row + the PIN-now/measurable-later
row: `tmsg_us_leaf` flips `Fixed` → `QuantileLaw` once instrumented). More/wider windows (a bigger
`budget`) → a tighter pool → a tighter SE, so the loop FUNDS it instead of leaving it irreducible.

`get_seed()` stays the DISTRUST fallback (the v1 1.0 us/leaf placeholder, a `Fixed` declared-spread
prior on the SEED path — the manifest's `trust=False` / pg-down route; only the MEASURED path is
shrinkable). FAIL LOUD (ADR-0002): if the codec cannot be imported/run, `_measure_raw()` RAISES — it
NEVER silently falls back to the seed-as-if-measured (that silent fallback is exactly the punt this
module removes).

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
from bench_common import logged_run, median_estimate, window_pool  # noqa: E402

NAME = "tmsg_us_leaf"
MODULE_PATH = "benchmarks.bench_tmsg"
_DESC = ("Per-leaf-amortized message-passing cost (us/leaf): request encode + reply decode of the "
         "inference wire codec over a coalesced S-leaf frame, /S (the pool MEDIAN over windows). "
         "Transport stage; NON-BINDING by a wide margin (ranks LAST for Neyman). Baseline = the ZMQ "
         "memcpy codec; v1 seed 1.0us/leaf (a deliberate over-charge).")

_IN_DIM = 241
_N_ACTIONS = 65
_S_LEAVES = 256        # the coalesced frame width (rows per encode_request / decode_response)
_FRAMES_PER_WINDOW = 200  # coalesced frames timed per window (a single-frame perf_counter call is clock-dominated)


def get_seed() -> G.Grounded:
    """The v1 seed (DISTRUST fallback): 1.0 us/leaf (a deliberate over-charge; non-binding). Used by the
    SEED path (manifest `trust=False` / pg-down) as a `Fixed` declared-spread prior; the MEASURED path
    (measure()/run() timing the live codec) is the shrinkable `QuantileLaw`."""
    return G.MSG_PER_LEAF_US


def register_self() -> Any:
    from bench_common import register_quantity
    return register_quantity(NAME, quantity="transport_msg_cost_per_leaf", units=get_seed().unit,
                             description=_DESC, module_path=MODULE_PATH)


def _measure_raw(budget: int = 64, s_leaves: int = _S_LEAVES) -> dict[str, Any]:
    """The raw-pool PROVENANCE producer (the §6 Phase-4 internal helper): RUN the live inference-wire
    codec sized by `budget`, and return the per-leaf framing-cost pool plus provenance. Over `budget`
    windows, time `encode_request(S x in_dim) + decode_response(reply)` for `_FRAMES_PER_WINDOW`
    coalesced frames and record ONE per-leaf reading per window (dt/frames/S — the per-leaf framing
    share of the codec), mirroring bench_cpp_inproc_port_tmsg_us_leaf's window loop so the SAME
    quantity-class is constructed the SAME way. `budget` IS the shrink budget — more windows → a bigger
    pool → a tighter median SE (the Neyman loop sizes it via the `budget` kwarg). Returns
    {'tmsg_us_leaf_median' (the pool MEDIAN), 'per_leaf_us' (the pool the Estimate is built over),
    'encode_us', 'decode_us' (the per-frame headline split, informational), 's_leaves', 'budget'}.
    Imports the codec + numpy lazily. `measure()`/`run()` both consume this ONE measurement (P1).

    FAIL LOUD (ADR-0002): an import/codec failure propagates (a missing codec is a loud fault, not a
    silent seed substitute) — never the seed-as-if-measured (the punt this module removes). The seed is
    the DISTRUST fallback path (get_seed()), not a measured-result substitute."""
    import numpy as np
    from chocofarm.az.inference_wire import encode_request, encode_response, decode_response

    feats = np.zeros((s_leaves, _IN_DIM), dtype=np.float32)
    reply = encode_response(np.zeros((s_leaves,), dtype=np.float32),
                            np.zeros((s_leaves, _N_ACTIONS), dtype=np.float32))
    # Warm (a first cold codec call / cache fill never poisons the pool).
    for _ in range(min(200, _FRAMES_PER_WINDOW)):
        encode_request(feats); decode_response(reply)

    # The per-leaf framing share, pooled per window: each window times `_FRAMES_PER_WINDOW` coalesced
    # encode+decode frames (the request encode + the reply decode, the full per-leaf framing cost) and
    # records dt/frames/S. The headline is the median over the windows (latency is right-skewed — the
    # mean is tail-poisoned, §7.A); a per-frame enc/dec split is kept for provenance.
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
    # A per-frame enc/dec split (informational provenance only — the headline + the pool are the codec's
    # full per-leaf framing share above; this attributes it to encode vs decode at the same operating point).
    t0 = time.perf_counter_ns()
    for _ in range(_FRAMES_PER_WINDOW):
        encode_request(feats)
    enc_acc_ns = float(time.perf_counter_ns() - t0)
    t0 = time.perf_counter_ns()
    for _ in range(_FRAMES_PER_WINDOW):
        decode_response(reply)
    dec_acc_ns = float(time.perf_counter_ns() - t0)
    enc_us = enc_acc_ns / 1000.0 / _FRAMES_PER_WINDOW
    dec_us = dec_acc_ns / 1000.0 / _FRAMES_PER_WINDOW
    return {
        "tmsg_us_leaf_median": float(np.median(per_leaf_us)),
        "per_leaf_us": per_leaf_us,
        "encode_us": enc_us,
        "decode_us": dec_us,
        "s_leaves": s_leaves,
        "budget": len(per_leaf_us),   # the realized window count (== the floored max(2, budget) window_pool ran)
    }


def _estimate_from_raw(res: dict[str, Any]) -> "_est.Estimate":
    """Build tmsg's harmonized SHRINKABLE `Estimate` — the SINGLE home of the Estimate construction
    (P1), called by BOTH `measure()` and `run()`. A k=1 median `QuantileLaw(p=0.5)` with a BOOTSTRAP
    median SE over the per-leaf framing-cost pool (§7.A — a real order-statistic SE, NOT a `Fixed`
    pin), `family=EMPIRICAL`, `kind='median'`, POSITIVE support. This is the ADR-0008 reclassification:
    tmsg_us_leaf is a MEASURED quantity whose variance RESPONDS to effort (the median's
    `marginal_dvar_deffort` is `−cov/n < 0`), so the Neyman loop can FUND it (more windows → tighter
    SE), where the prior `Fixed` pin (marginal=0) made it un-fundable. SAME construction as the
    same-quantity-class sibling bench_cpp_inproc_port_tmsg_us_leaf (ADR-0012 P8: one quantity-class,
    one typed shrink signature)."""
    return median_estimate(res["per_leaf_us"], name=NAME)   # bootstrap median SE over the per-leaf pool


def measure(budget: int = 64, s_leaves: int = _S_LEAVES) -> "_est.Estimate":
    """Measure tmsg_us_leaf (RUN the live inference-wire codec) and return its harmonized k=1 SHRINKABLE
    median `Estimate` (§6 Phase 4: `measure()` returns the `Estimate` the bench DECLARES — the
    driver/untrusted_drive `set_estimate`s it directly). `budget` sizes the measurement pool (the budget
    the Neyman loop passes — more windows tightens tmsg's SE). TIMING-SENSITIVE — pinned (taskset -c 0);
    never during the fan-out."""
    return _estimate_from_raw(_measure_raw(budget=budget, s_leaves=s_leaves))


def run(budget: int = 64, s_leaves: int = _S_LEAVES) -> dict[str, Any]:
    """Measure tmsg_us_leaf (RUN the live inference-wire codec) and LOG it to postgres as a harmonized
    k=1 SHRINKABLE median `Estimate` (§6 Phase 3): `QuantileLaw(p=0.5)` with a BOOTSTRAP median SE over
    the per-leaf framing-cost pool (§7.A), `family=EMPIRICAL`, `kind='median'`. The per-leaf readings
    are logged as raw PROVENANCE — the variance authority is `estimate.cov`, so the headline is NOT
    double-logged as a sample row (§5.2 de-dup). Returns the raw provenance dict. TIMING-SENSITIVE —
    operator-invoked, pinned (taskset -c 0), never during the fan-out."""
    res = _measure_raw(budget=budget, s_leaves=s_leaves)   # ONE measurement (Est + provenance pool)
    est = _estimate_from_raw(res)            # the SAME Estimate measure() returns (P1)
    cfg = {"kind": "inference_wire_codec_measured", "codec": "inference_wire_memcpy",
           "s_leaves": res["s_leaves"], "budget": res["budget"], "frames_per_window": _FRAMES_PER_WINDOW,
           "encode_us": res["encode_us"], "decode_us": res["decode_us"],
           "tmsg_us_leaf_median": res["tmsg_us_leaf_median"]}
    with logged_run(NAME, quantity="transport_msg_cost_per_leaf", units=get_seed().unit, description=_DESC,
                    module_path=MODULE_PATH, config=cfg, estimate=est) as log:
        # PROVENANCE only (§5.2): the raw per-leaf readings. The headline lives in estimate.theta_hat[0]
        # (the SSOT), the median SE in estimate.cov.
        log(res["per_leaf_us"], sample_size=1)
    return res


if __name__ == "__main__":
    print(f"[bench_tmsg] seed: {get_seed().mean} {get_seed().unit} (DISTRUST fallback; provenance: "
          f"{get_seed().provenance})")
    register_self()
    print("[bench_tmsg] registered. measure()/run() RUN the live inference-wire codec (taskset -c 0) "
          "-> a SHRINKABLE median Estimate. get_seed() is the DISTRUST fallback.")
