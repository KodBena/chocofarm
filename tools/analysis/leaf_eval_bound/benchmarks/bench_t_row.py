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

from typing import Any, Optional


from leaf_eval_bound.contract import estimate as _est  # noqa: E402  — the harmonized Estimate contract (measure() returns one — §6 Phase 4)
from leaf_eval_bound.contract import grounding as G  # noqa: E402
from leaf_eval_bound.benchmarks.estimators import fit_estimate  # noqa: E402
from leaf_eval_bound.benchmarks.scaffold import bench as _scaffold  # noqa: E402  — move 6 wiring

NAME = "t_row_us"
# The co-fit PARTNER: t_row (the slope) and iota (the intercept) are the SAME staged `time = intercept +
# slope·rows` fit (bench_iota.measure delegates here). So the harmonized Estimate this bench logs is the
# k=2 fit with t_row's SLOPE as component 0 (the marginal manifest.value("t_row_us") projects — 8 live
# model consumers read this slope) and iota_us as the partner carrying the −0.81 off-diagonal (§4.2).
PARTNER_NAME = "iota_us"
WARMUP = 8   # harness warmup phase (harness.warm): burn cold-compile forwards before measuring
MODULE_PATH = "leaf_eval_bound.benchmarks.bench_t_row"
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


# Move 6: the shared scaffold wires register_self / measure / run from the bench-specific parts above.
# run()'s body has an INTERMEDIATE (batches_used -> medians) and a run-knob (sample_size=iters), so the
# config + log hooks are DEFs that recompute that intermediate VERBATIM from `res` (and the threaded `kw`,
# defaults applied). A lambda can't hold the statements; the recompute is behavior-identical.
def _run_config(res, **kw):
    batches_used = res["batches"]
    return {"batches": batches_used, "iters": kw["iters"], "repeat": kw["repeat"],
            "fit_slope_us_per_row": res["slope_us_per_row"], "fit_intercept_us": res["intercept_us"],
            "fit_r2": res["r2"], "bench": "run_microbatch_staged"}


def _run_log(res, log, **kw):
    batches_used = res["batches"]
    medians = [res["per_width_median_us"][B] for B in batches_used]
    # PROVENANCE only (§5.2): the per-width medians (the raw design points), sample_size = iters behind
    # each. The headline slope is NOT logged as a sample — it lives in estimate.theta_hat[0] (the SSOT).
    log(medians, sample_size=kw["iters"])


_B = _scaffold(
    name=NAME, quantity="serve_per_row_cost", module_path=MODULE_PATH, description=_DESC,
    seed=get_seed, measure_raw=_measure_raw, estimate_from_raw=_estimate_from_raw,
    run_config=_run_config, run_log=_run_log,
)
register_self, measure, run = _B.register_self, _B.measure, _B.run


if __name__ == "__main__":
    # NOTE: timing-sensitive. Run pinned + sole-workload: taskset -c 0 python benchmarks/bench_t_row.py
    print(f"[bench_t_row] seed: {get_seed().mean} {get_seed().unit} "
          f"(provenance: {get_seed().provenance})")
    print("[bench_t_row] registering definition…")
    register_self()
    print("[bench_t_row] registered. NOT running the live measurement here (timing-sensitive); "
          "invoke run() pinned and sole-workload.")
