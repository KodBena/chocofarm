"""
tools/analysis/leaf_eval_bound/benchmarks/bench_iota.py
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


from leaf_eval_bound.contract import estimate as _est  # noqa: E402  — the harmonized Estimate contract (measure() returns one — §6 Phase 4)
from leaf_eval_bound.contract import grounding as G  # noqa: E402
from leaf_eval_bound.benchmarks.estimators import fit_estimate  # noqa: E402
from leaf_eval_bound.benchmarks.harness import logged_run  # noqa: E402

NAME = "iota_us"
# The co-fit PARTNER: iota (the intercept) and t_row (the slope) are the SAME staged fit (measure()
# delegates to bench_t_row.measure). The harmonized Estimate this bench logs is that k=2 fit with iota's
# INTERCEPT as component 0 (the marginal manifest.value("iota_us") projects) and t_row_us the partner
# carrying the −0.81 off-diagonal (§4.2) — the SAME fit t_row logs, only the component order differs.
PARTNER_NAME = "t_row_us"
WARMUP = 8   # harness warmup phase (harness.warm): burn cold-compile forwards before measuring
MODULE_PATH = "leaf_eval_bound.benchmarks.bench_iota"
_DESC = ("SERVE fixed per-forward cost (us): the intercept of the staged run_microbatch JAX forward "
         "(time = iota + t_row*rows). JAX-forward-only floor (dispatch + output pull + input); contains "
         "NO ZMQ drain/scatter (that is tau_io). Baseline, transport-invariant.")


def get_seed() -> G.Grounded:
    """The v1 seed (DISTRUST fallback): the staged run_microbatch intercept, 94.58 us."""
    return G.SERVE_INTERCEPT_US


def register_self() -> Any:
    from leaf_eval_bound.benchmarks.harness import register_quantity
    return register_quantity(NAME, quantity="serve_fixed_forward_cost", units=get_seed().unit,
                             description=_DESC, module_path=MODULE_PATH)


def _measure_raw(batches: Optional[list[int]] = None, iters: int = 200, repeat: int = 30) -> dict[str, Any]:
    """The raw-pool PROVENANCE producer (the §6 Phase-4 internal helper): the staged-forward fit
    (DELEGATES to `bench_t_row._measure_raw`, which fits time = intercept + slope*rows — one measurement
    grounds BOTH the slope and the intercept). Returns its design-point dict (intercept_us is the iota
    reading). `measure()` wraps it into iota's intercept-first Estimate; `run()` uses it for both."""
    from leaf_eval_bound.benchmarks import bench_t_row
    return bench_t_row._measure_raw(batches=batches, iters=iters, repeat=repeat)


def _estimate_from_raw(res: dict[str, Any]) -> "_est.Estimate":
    """Build iota's harmonized `Estimate` from a `_measure_raw()` dict — the SINGLE home of the Estimate
    construction (P1), called by BOTH `measure()` and `run()`. The k=2 staged-fit Estimate with iota's
    INTERCEPT as component 0 (the marginal `manifest.value("iota_us")` projects) and t_row_us the partner
    carrying the −0.81 off-diagonal (§4.2) — the SAME fit bench_t_row logs, only the component order differs."""
    batches_used = res["batches"]
    medians = [res["per_width_median_us"][B] for B in batches_used]
    return fit_estimate(batches_used, medians, own_name=NAME, own_role="intercept", partner_name=PARTNER_NAME)


def measure(batches: Optional[list[int]] = None, iters: int = 200, repeat: int = 30) -> "_est.Estimate":
    """Measure iota and return its harmonized k=2 fit `Estimate` (§6 Phase 4: `measure()` returns the
    `Estimate` the bench DECLARES — iota's INTERCEPT as component 0). The raw design-point dict is the
    bench's internal `_measure_raw()` provenance. TIMING-SENSITIVE — pin the process to one core."""
    return _estimate_from_raw(_measure_raw(batches=batches, iters=iters, repeat=repeat))


def run(batches: Optional[list[int]] = None, iters: int = 200, repeat: int = 30) -> dict[str, Any]:
    """Measure iota and LOG it to postgres as a harmonized k=2 fit `Estimate` (§6 Phase 3): the staged-fit
    intercept/slope with their −0.81 off-diagonal, iota's INTERCEPT as component 0. The per-width medians
    are logged as raw-design-point PROVENANCE — the variance authority is now `estimate.cov`, so the
    headline intercept scalar is NO LONGER double-logged as a sample row (the §5.2 de-dup obligation).
    TIMING-SENSITIVE — operator-invoked, pinned, never during the fan-out."""
    res = _measure_raw(batches=batches, iters=iters, repeat=repeat)   # ONE measurement (Estimate + provenance)
    batches_used = res["batches"]
    medians = [res["per_width_median_us"][B] for B in batches_used]
    # The k=2 fit Estimate, built by the SAME helper measure() returns (P1 single-home).
    est = _estimate_from_raw(res)
    cfg = {"batches": batches_used, "iters": iters, "repeat": repeat,
           "fit_slope_us_per_row": res["slope_us_per_row"], "fit_intercept_us": res["intercept_us"],
           "fit_r2": res["r2"], "bench": "run_microbatch_staged"}
    with logged_run(NAME, quantity="serve_fixed_forward_cost", units=get_seed().unit, description=_DESC,
                    module_path=MODULE_PATH, config=cfg, estimate=est) as log:
        # PROVENANCE only (§5.2): the per-width medians (the raw design points). The headline intercept is
        # NOT logged as a sample — it lives in estimate.theta_hat[0] (the SSOT).
        log(medians, sample_size=iters)
    return res


if __name__ == "__main__":
    print(f"[bench_iota] seed: {get_seed().mean} {get_seed().unit} (provenance: {get_seed().provenance})")
    register_self()
    print("[bench_iota] registered. NOT running the live measurement (timing-sensitive); "
          "invoke run() pinned and sole-workload.")
