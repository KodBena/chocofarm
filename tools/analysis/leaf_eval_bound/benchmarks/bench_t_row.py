"""
tools/analysis/leaf_eval_bound/benchmarks/bench_t_row.py
==================================================

LIVE benchmark for `t_row` — the SERVE-side per-row marginal forward cost (us/row), the
SLOPE of the JAX `run_microbatch` staged forward (time = iota + t_row * rows). This is the
per-row term in the binding (serve) stage of both throughput models; it is INVARIANT across
transport designs (a transport moves T_io / wakeup / msg-cost, not the XLA matmul slope), so
it is a BASELINE quantity (un-prefixed), shared by every transport variant.

WHAT run() MEASURES (1:1 with the model input — condition 1). The production forward graph
(`chocofarm.az.bench.bench_mlp_lowlatency._prod_forward`) staged device-resident, timed at a
SWEEP of batch widths; a least-squares `time = intercept + slope*rows` fit yields the slope.
run() logs the per-width median-us readings AND the fitted slope sample. The SEED is the v1
fit (run_microbatch_staging fits.staged.slope_us_per_row = 4.317 us/row).

TIMING-SENSITIVE — DO NOT run() during the parallel fan-out (the bench must own the cores; a
co-scheduled workflow inflates the slope). Pin: `taskset -c 0`. The manifest gates rerun
behind an explicit operator action.

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

import estimate as _est  # noqa: E402  — the harmonized Estimate contract (measure() returns one — §6 Phase 4)
import leaf_eval_grounding as G  # noqa: E402
from bench_common import fit_estimate, logged_run  # noqa: E402

NAME = "t_row_us"
# The co-fit PARTNER: t_row (the slope) and iota (the intercept) are the SAME staged `time = intercept +
# slope·rows` fit (bench_iota.measure delegates here). So the harmonized Estimate this bench logs is the
# k=2 fit with t_row's SLOPE as component 0 (the marginal manifest.value("t_row_us") projects — 8 live
# model consumers read this slope) and iota_us as the partner carrying the −0.81 off-diagonal (§4.2).
PARTNER_NAME = "iota_us"
WARMUP = 8   # harness warmup phase (bench_common.warm): burn cold-compile forwards before measuring
MODULE_PATH = "benchmarks.bench_t_row"
_DESC = ("SERVE per-row marginal forward cost (us/row): the slope of the staged run_microbatch JAX "
         "forward (time = iota + t_row*rows). Baseline, transport-invariant; the per-row term of the "
         "binding serve stage.")

# The PRODUCTION forward geometry (bench_mlp_lowlatency argparse defaults — the inference server's
# shape: 241 feature dim from features.py, 256 ValueMLP hidden, 65 actions = 20 collect+44 sense+1 term).
# Held here as the bench's own pins so measure() is self-contained (the bench module exports no constants).
_IN_DIM, _HIDDEN, _N_ACTIONS = 241, 256, 65


def get_seed() -> G.Grounded:
    """The v1 seed (the DISTRUST fallback): the staged run_microbatch slope fit, 4.317 us/row."""
    return G.SERVE_SLOPE_US


def register_self() -> Any:
    """Register this quantity's definition row (idempotent). Returns the definition id."""
    from bench_common import register_quantity
    return register_quantity(NAME, quantity="serve_per_row_cost", units=get_seed().unit,
                             description=_DESC, module_path=MODULE_PATH)


def _measure_raw(batches: Optional[list[int]] = None, iters: int = 200, repeat: int = 30) -> dict[str, Any]:
    """The raw-pool PROVENANCE producer (the §6 Phase-4 internal helper): time the staged production forward
    across `batches` widths, fit the slope, and return the design-point dict
    {'slope_us_per_row', 'intercept_us', 'r2', 'per_width_median_us': {B: us}, 'batches'}. This is the
    TIMING-SENSITIVE measurement body; `measure()` wraps it into the harmonized `Estimate` and `run()` uses it
    for BOTH the Estimate and the raw provenance rows (ONE measurement, two consumers — P1). Imports jax lazily
    (so importing this module for get_seed() stays jax-free). Pin the process to one core (taskset -c 0)."""
    import numpy as np
    from chocofarm.az.bench.bench_mlp_lowlatency import (
        _median_iqr_us, _prod_forward, _prod_params, _fit_line,
    )
    from chocofarm.az.lowlatency import compile_lowlatency, run

    batches = batches or [32, 64, 128, 192, 256, 384, 512]
    params = _prod_params(_IN_DIM, _HIDDEN, _N_ACTIONS)
    med_us: list[float] = []
    for B in batches:
        x_host = np.zeros((B, _IN_DIM), dtype=np.float32)
        handle = compile_lowlatency(_prod_forward, params, x_host)  # robust: params staged device-resident
        med, _iqr = _median_iqr_us(lambda: run(handle, x_host), iters, repeat)
        med_us.append(med)
    intercept, slope, r2 = _fit_line(batches, med_us)
    return {"slope_us_per_row": slope, "intercept_us": intercept, "r2": r2,
            "per_width_median_us": dict(zip(batches, med_us)), "batches": batches}


def _estimate_from_raw(res: dict[str, Any]) -> "_est.Estimate":
    """Build THIS bench's harmonized `Estimate` from a `_measure_raw()` dict — the SINGLE home of the
    Estimate construction (P1), called by BOTH `measure()` and `run()` so they cannot disagree. The k=2
    staged-fit Estimate with t_row's SLOPE as component 0 (the marginal `manifest.value("t_row_us")`
    projects — 8 live consumers) and iota_us the partner carrying the −0.81 off-diagonal (§4.2)."""
    batches_used = res["batches"]
    medians = [res["per_width_median_us"][B] for B in batches_used]
    return fit_estimate(batches_used, medians, own_name=NAME, own_role="slope", partner_name=PARTNER_NAME)


def measure(batches: Optional[list[int]] = None, iters: int = 200, repeat: int = 30) -> "_est.Estimate":
    """Measure t_row and return its harmonized k=2 fit `Estimate` (§6 Phase 4: `measure()` returns the
    `Estimate` the bench DECLARES — the driver/untrusted_drive consume it directly, no guessing which list
    is the estimate). The raw design-point dict is the bench's internal `_measure_raw()` provenance (read by
    `run()` for the audit rows). TIMING-SENSITIVE — pin the process to one core (taskset -c 0)."""
    return _estimate_from_raw(_measure_raw(batches=batches, iters=iters, repeat=repeat))


def run(batches: Optional[list[int]] = None, iters: int = 200, repeat: int = 30) -> dict[str, Any]:
    """Measure t_row and LOG it to postgres as a harmonized k=2 fit `Estimate` (§6 Phase 3): the staged-fit
    slope/intercept with their −0.81 off-diagonal, t_row's SLOPE as component 0. The per-width medians are
    logged as raw-design-point PROVENANCE — the variance authority is now `estimate.cov` (the SE comes from
    `resid_var` + the x-design), so the headline slope scalar is NO LONGER double-logged as a sample row
    (the §5.2 de-dup obligation: that double-log corrupts `latest_aggregate`'s count + averages a slope
    with seven 4-digit medians). Returns the measurement dict. TIMING-SENSITIVE — operator-invoked, pinned,
    never during the fan-out."""
    res = _measure_raw(batches=batches, iters=iters, repeat=repeat)   # ONE measurement (Estimate + provenance)
    batches_used = res["batches"]
    medians = [res["per_width_median_us"][B] for B in batches_used]
    # The k=2 fit Estimate, built by the SAME helper measure() returns (P1 single-home).
    est = _estimate_from_raw(res)
    cfg = {"batches": batches_used, "iters": iters, "repeat": repeat,
           "fit_slope_us_per_row": res["slope_us_per_row"], "fit_intercept_us": res["intercept_us"],
           "fit_r2": res["r2"], "bench": "run_microbatch_staged"}
    with logged_run(NAME, quantity="serve_per_row_cost", units=get_seed().unit, description=_DESC,
                    module_path=MODULE_PATH, config=cfg, estimate=est) as log:
        # PROVENANCE only (§5.2): the per-width medians (the raw design points), sample_size = iters behind
        # each. The headline slope is NOT logged as a sample — it lives in estimate.theta_hat[0] (the SSOT).
        log(medians, sample_size=iters)
    return res


if __name__ == "__main__":
    # NOTE: timing-sensitive. Run pinned + sole-workload: taskset -c 0 python benchmarks/bench_t_row.py
    print(f"[bench_t_row] seed: {get_seed().mean} {get_seed().unit} "
          f"(provenance: {get_seed().provenance})")
    print("[bench_t_row] registering definition…")
    register_self()
    print("[bench_t_row] registered. NOT running the live measurement here (timing-sensitive); "
          "invoke run() pinned and sole-workload.")
