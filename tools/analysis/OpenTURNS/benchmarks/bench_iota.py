"""
tools/analysis/OpenTURNS/benchmarks/bench_iota.py
=================================================

LIVE benchmark for `iota_us` — the SERVE-side fixed per-forward cost (us), the INTERCEPT of
the staged JAX `run_microbatch` forward (time = iota + t_row*rows). This is the JAX-FORWARD-
ONLY floor (dispatch + output pull + input + residual); it contains NO ZMQ drain/recv/scatter
(that is the separate `tau_io_us` term). Baseline, transport-invariant: a transport moves
T_io / wakeup / msg-cost, not the XLA dispatch floor.

WHAT run() MEASURES (1:1 with the model input). The intercept of the SAME staged-forward fit
`bench_t_row.measure()` produces (one measurement grounds both the slope and the intercept).
run() logs the fitted intercept. The SEED is the v1 fit (run_microbatch_staging
fits.staged.intercept_us = 94.58 us).

TIMING-SENSITIVE — DO NOT run() during the parallel fan-out. Pin: `taskset -c 0`.

Public Domain (The Unlicense).
"""
from __future__ import annotations

import os
import sys
from typing import Any, Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.dirname(_HERE), _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import leaf_eval_grounding as G  # noqa: E402
from bench_common import logged_run  # noqa: E402

NAME = "iota_us"
MODULE_PATH = "benchmarks.bench_iota"
_DESC = ("SERVE fixed per-forward cost (us): the intercept of the staged run_microbatch JAX forward "
         "(time = iota + t_row*rows). JAX-forward-only floor (dispatch + output pull + input); contains "
         "NO ZMQ drain/scatter (that is tau_io). Baseline, transport-invariant.")


def get_seed() -> G.Grounded:
    """The v1 seed (DISTRUST fallback): the staged run_microbatch intercept, 94.58 us."""
    return G.SERVE_INTERCEPT_US


def register_self() -> Any:
    from bench_common import register_quantity
    return register_quantity(NAME, quantity="serve_fixed_forward_cost", units=get_seed().unit,
                             description=_DESC, module_path=MODULE_PATH)


def measure(batches: Optional[list[int]] = None, iters: int = 200, repeat: int = 30) -> dict[str, Any]:
    """Measure iota: the intercept of the staged-forward fit (delegates to bench_t_row.measure, which
    fits time = intercept + slope*rows). Returns its dict (intercept_us is the iota reading)."""
    import bench_t_row
    return bench_t_row.measure(batches=batches, iters=iters, repeat=repeat)


def run(batches: Optional[list[int]] = None, iters: int = 200, repeat: int = 30) -> dict[str, Any]:
    """Measure iota and LOG it to postgres (the fitted intercept as the headline reading). TIMING-
    SENSITIVE — operator-invoked, pinned, never during the fan-out."""
    res = measure(batches=batches, iters=iters, repeat=repeat)
    cfg = {"batches": res["batches"], "iters": iters, "repeat": repeat,
           "fit_slope_us_per_row": res["slope_us_per_row"], "fit_r2": res["r2"],
           "bench": "run_microbatch_staged"}
    with logged_run(NAME, quantity="serve_fixed_forward_cost", units=get_seed().unit, description=_DESC,
                    module_path=MODULE_PATH, config=cfg) as log:
        log(res["intercept_us"], sample_size=len(res["batches"]))
    return res


if __name__ == "__main__":
    print(f"[bench_iota] seed: {get_seed().mean} {get_seed().unit} (provenance: {get_seed().provenance})")
    register_self()
    print("[bench_iota] registered. NOT running the live measurement (timing-sensitive); "
          "invoke run() pinned and sole-workload.")
