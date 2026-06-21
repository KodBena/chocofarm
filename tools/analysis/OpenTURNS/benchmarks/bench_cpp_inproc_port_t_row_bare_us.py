"""
tools/analysis/OpenTURNS/benchmarks/bench_cpp_inproc_port_t_row_bare_us.py
=========================================================================

LIVE benchmark for `cpp_inproc_port_t_row_bare_us` — the BARE-forward per-row marginal cost
(us/row) the C++ IN-PROCESS QUEUE-PORT transport feeds: the SLOPE of the `fully_device` variant
of the JAX forward (params + input staged device-resident, output kept on-device — NO host pull,
NO `run_microbatch` host-block stack/concat). This is the ONE per-row term cpp_inproc_port MOVES
off the transport-invariant baseline (`t_row_us` = the staged `run_microbatch` slope 4.317 us/row);
every OTHER transport variant feeds the staged forward and so shares the baseline t_row. By moving
to the bare-forward slope this variant ISOLATES transport-removed (the run_microbatch concat +
device->host pull the inproc port elides) from irreducible-XLA (the matmul slope no transport can
move). See the model docstring (model_cpp_inproc_port.py) for why this is the right per-row term.

WHY THE FULLY-DEVICE SLOPE IS THE RIGHT t_row FOR THIS VARIANT. The inproc port runs generation
and serve in ONE process; a leaf-eval is a DIRECT function call into the batched forward, with the
forward's output left device-resident (the in-process caller reads it back without the
recv-to-host-then-reframe the wire pays). The `fully_device` variant of the lowlatency
decomposition is exactly that geometry — input device-resident, output not pulled to host — so its
slope is the per-row cost an inproc port pays. The host->device crossing of the gathered input
block (the one transfer fully_device's device-resident input ELIDES) is NOT in this slope; it is
charged SEPARATELY in `cpp_inproc_port_tau_io_us` (the residual staging term), so the two terms
together carry the full per-forward cost with no double-count and no gap.

WHAT run() MEASURES (1:1 with the model input — condition 1). The production forward graph
(`chocofarm.az.bench.bench_mlp_lowlatency.bench`, the `fully_device` variant) at a SWEEP of batch
widths; the least-squares `time = intercept + slope*rows` fit's SLOPE is the headline reading. The
SAME fit yields the dispatch floor (`T_disp_us` = its intercept, 68.84 us); this bench reads the
SLOPE off the SAME `fully_device` fit, so t_row_bare and T_disp are mutually consistent by
construction (one fit, two read-offs). The SEED is the v1 fit
(`mlp_lowlatency/results.json fits.fully_device.slope_us_per_row` = 3.0920 us/row, R^2 0.9972).

TIMING-SENSITIVE — DO NOT run() during the parallel fan-out (a co-scheduled workflow inflates the
slope). Pin: `taskset -c 0`. The manifest gates rerun behind an explicit operator action.

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

from bench_common import fit_estimate, logged_run  # noqa: E402

NAME = "cpp_inproc_port_t_row_bare_us"
# The co-fit PARTNER: this bare-forward SLOPE and T_disp (the dispatch-floor INTERCEPT) are read off the
# SAME `fully_device` fit (one fit, two read-offs — see WHAT run() MEASURES above). So the harmonized
# Estimate this bench logs is that k=2 fit with the bare-forward SLOPE as component 0 (the marginal
# manifest.value("cpp_inproc_port_t_row_bare_us") projects) and T_disp_us the partner carrying the −0.81
# off-diagonal (§4.2). This fit is DISTINCT from the staged (iota/t_row) fit — they must NOT cross-pair.
PARTNER_NAME = "T_disp_us"
MODULE_PATH = "benchmarks.bench_cpp_inproc_port_t_row_bare_us"
_DESC = ("BARE-forward per-row marginal cost (us/row) the C++ in-process queue-port feeds: the slope of the "
         "fully_device JAX forward (params+input staged device-resident, output on-device — no host pull, no "
         "run_microbatch concat). The ONE per-row term cpp_inproc_port moves off the staged-slope baseline "
         "(isolates transport-removed from irreducible-XLA); read off the SAME fully_device fit as T_disp.")

# The PRODUCTION forward geometry (bench_mlp_lowlatency defaults — the inference server's shape: 241 feature
# dim from features.py, 256 ValueMLP hidden, 65 actions). Held as the bench's own pins (self-contained).
_IN_DIM, _HIDDEN, _N_ACTIONS = 241, 256, 65

# The v1 seed: the fully_device slope from the lowlatency decomposition (the bare-forward per-row cost). Homed
# here (NOT in leaf_eval_grounding, which carries the STAGED slope as G.SERVE_SLOPE_US and the bare slope only
# as the inline 3.092 literal in model_cycletime.inproc_port_contrast) because this is THIS variant's own
# quantity. Provenance is the same results.json the staged slope comes from, fits.fully_device.slope_us_per_row.
_SEED_MEAN = 3.0920    # mlp_lowlatency/results.json fits.fully_device.slope_us_per_row (R^2 0.9972)
_SEED_SIGMA = 0.15     # same scale as model_cycletime._T_ROW sigma (a fit-slope spread, not the staged 0.5)


def get_seed() -> tuple[float, float, str]:
    """The v1 seed (DISTRUST fallback): the bare-forward fully_device slope, 3.092 us/row. Returns
    (mean, sigma, unit). A (mean, sigma, unit) tuple (not a Grounded) because this is a NEW per-variant
    quantity with no leaf_eval_grounding Grounded of its own — the manifest accepts either form."""
    return (_SEED_MEAN, _SEED_SIGMA, "us/row")


def register_self() -> Any:
    from bench_common import register_quantity
    return register_quantity(NAME, quantity="serve_per_row_cost_bare_forward", units="us/row",
                             description=_DESC, module_path=MODULE_PATH)


def measure(batches: Optional[list[int]] = None, iters: int = 200, repeat: int = 30) -> dict[str, Any]:
    """Measure the bare-forward slope: run the lowlatency decomposition (`bench_mlp_lowlatency.bench`) across
    `batches` widths and read the `fully_device` variant's fitted slope. Returns
    {'slope_us_per_row', 'intercept_us', 'r2', 'decomposition': {...}}. Imports jax lazily (so importing this
    module for get_seed() stays jax-free). Pin the process to one core (taskset -c 0)."""
    from chocofarm.az.bench.bench_mlp_lowlatency import bench

    batches = batches or [32, 64, 128, 192, 256, 384, 512]
    out = bench(batches, _IN_DIM, _HIDDEN, _N_ACTIONS, iters=iters, repeat=repeat, warmup=200)
    fits = out.get("fits", {})
    fd = fits.get("fully_device", {})
    slope = fd.get("slope_us_per_row")
    if slope is None:  # ADR-0002: surface the shape rather than silently default
        raise RuntimeError(f"bench_cpp_inproc_port_t_row_bare: fits.fully_device.slope_us_per_row absent in "
                           f"bench output keys {list(out)} (fits keys {list(fits)})")
    # The fully_device per-width medians (the design points behind the fit) — the inputs the harmonized k=2
    # Estimate's covariance is computed from (§4.2/§5: the fit's lstsq discards them, so the bench recovers
    # them here from bench()'s per_batch_us[variant] = {str(B): {median_us, iqr_us}}).
    pb_fd = out.get("per_batch_us", {}).get("fully_device", {})
    median_us = {int(B): float(pb_fd[str(B)]["median_us"]) for B in batches if str(B) in pb_fd}
    return {"slope_us_per_row": float(slope), "intercept_us": fd.get("intercept_us"),
            "r2": fd.get("r2"), "per_width_median_us": median_us, "batches": batches,
            "decomposition": out.get("decomposition", {})}


def run(batches: Optional[list[int]] = None, iters: int = 200, repeat: int = 30) -> dict[str, Any]:
    """Measure the bare-forward slope and LOG it to postgres (the fully_device slope as the headline reading,
    sample_size = the fit's #widths). Records the fully_device intercept (== T_disp, for cross-consistency)
    and the decomposition in the run config. TIMING-SENSITIVE — operator-invoked, pinned (taskset -c 0),
    NEVER during the fan-out."""
    res = measure(batches=batches, iters=iters, repeat=repeat)
    batches_used = [B for B in res["batches"] if B in res["per_width_median_us"]]
    medians = [res["per_width_median_us"][B] for B in batches_used]
    # The k=2 fit Estimate, the bare-forward slope as component 0; T_disp_us the partner with the
    # off-diagonal. SAME fully_device fit T_disp logs, only the component order differs (§4.2).
    est = fit_estimate(batches_used, medians, own_name=NAME, own_role="slope", partner_name=PARTNER_NAME)
    cfg = {"batches": batches_used, "iters": iters, "repeat": repeat, "variant": "fully_device",
           "fully_device_intercept_us": res["intercept_us"], "fit_slope_us_per_row": res["slope_us_per_row"],
           "fit_r2": res["r2"], "decomposition": res["decomposition"], "bench": "mlp_lowlatency",
           "note": "bare-forward slope (no run_microbatch concat, output device-resident); the inproc-port "
                   "t_row. T_disp is THIS fit's intercept (one fit, two read-offs)."}
    with logged_run(NAME, quantity="serve_per_row_cost_bare_forward", units="us/row", description=_DESC,
                    module_path=MODULE_PATH, config=cfg, estimate=est) as log:
        # PROVENANCE only (§5.2): the fully_device per-width medians. The headline slope is NOT logged as a
        # sample — it lives in estimate.theta_hat[0] (the SSOT).
        log(medians, sample_size=iters)
    return res


if __name__ == "__main__":
    _m, _s, _u = get_seed()
    print(f"[bench_cpp_inproc_port_t_row_bare_us] seed: {_m} {_u} (sigma {_s}) — "
          f"mlp_lowlatency fits.fully_device.slope_us_per_row (R^2 0.9972)")
    register_self()
    print("[bench_cpp_inproc_port_t_row_bare_us] registered. NOT running the live measurement here "
          "(timing-sensitive); invoke run() pinned (taskset -c 0) and sole-workload. This is the ONE per-row "
          "term cpp_inproc_port moves off the staged-slope baseline.")
