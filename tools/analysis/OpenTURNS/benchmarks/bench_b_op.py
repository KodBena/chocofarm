"""
tools/analysis/OpenTURNS/benchmarks/bench_b_op.py
=================================================

LIVE benchmark for `B_op` — the server's sustained FULL-BUCKET operating point (rows/forward):
the real row count at a full bucket, the achievable serve peak (the serve curve is a sawtooth
real/(iota+slope*bucket(real)+tau_io), maximized at full buckets). It is the single quantity
that most moves the serve stage AT the binding point — a top Neyman target alongside tau_io.

WHAT run() MEASURES (1:1 with the model input). B_op is the steady-state full-bucket B the
OPTIMUM sustains under a fed producer set — its faithful measurement is the rows/forward
histogram of a saturated end-to-end run (the server fed by N producers at the full-bucket feed),
read off the server's per-forward batch-size counter. That is an end-to-end harness artifact, not
an isolated microbench; `run()` records the v1 estimate (256 rows/forward — analysis_clean.txt
GLOBAL MAX rows/fwd=511.5, server max_batch=256) with a config note that the saturated rows/fwd
histogram is the outstanding measurement, and keeps the quantity flagged needs-measurement.

A transport variant that changes the coalescing dynamics (a group-wakeup futex, a lock-free
queue) can change the sustained B_op, so a variant MAY register its own `<slug>_B_op`.

Public Domain (The Unlicense).
"""
from __future__ import annotations

import os
import sys
from typing import Any

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.dirname(_HERE), _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import leaf_eval_grounding as G  # noqa: E402
from bench_common import logged_run  # noqa: E402

NAME = "B_op"
MODULE_PATH = "benchmarks.bench_b_op"
_DESC = ("Server sustained FULL-BUCKET operating point (rows/forward): the achievable serve peak (the "
         "serve sawtooth is maximized at full buckets). v1 256 (GLOBAL MAX rows/fwd=511.5, max_batch=256). "
         "Real measurement = saturated end-to-end rows/forward histogram. Top serve-stage Neyman target.")


def get_seed() -> G.Grounded:
    """The v1 seed (DISTRUST fallback): B_op=256 rows/forward (full-bucket operating point)."""
    return G.SERVE_FULL_BUCKET


def register_self() -> Any:
    from bench_common import register_quantity
    return register_quantity(NAME, quantity="serve_full_bucket_rows", units=get_seed().unit,
                             description=_DESC, module_path=MODULE_PATH)


def measure() -> dict[str, Any]:
    """The current B_op estimate (full-bucket rows/forward). A faithful measure is the saturated
    end-to-end rows/forward histogram (the server's per-forward batch-size counter under a fed producer
    set), an e2e harness artifact. Returns {'b_op_rows', 'note'}."""
    return {"b_op_rows": get_seed().mean,
            "note": "full-bucket operating point; saturated end-to-end rows/forward histogram outstanding"}


def run() -> dict[str, Any]:
    """Record the current B_op estimate to postgres as a single sample, flagged as awaiting the saturated
    rows/forward histogram. Returns the estimate dict."""
    res = measure()
    cfg = {"kind": "operating_point",
           "needs_measurement": "saturated end-to-end rows/forward histogram (server batch-size counter)",
           "note": res["note"]}
    with logged_run(NAME, quantity="serve_full_bucket_rows", units=get_seed().unit, description=_DESC,
                    module_path=MODULE_PATH, config=cfg) as log:
        log(res["b_op_rows"], sample_size=None)
    return res


if __name__ == "__main__":
    print(f"[bench_b_op] seed: {get_seed().mean} {get_seed().unit} (provenance: {get_seed().provenance})")
    register_self()
    print("[bench_b_op] registered. The real measurement is the saturated e2e rows/forward histogram.")
