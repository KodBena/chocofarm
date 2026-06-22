"""
tools/analysis/leaf_eval_bound/benchmarks/bench_t_disp.py
===================================================

LIVE benchmark for `T_disp` — the pure pjit/XLA dispatch FLOOR (us): the irreducible per-forward
dispatch cost with params AND input pre-staged device-resident and the output left on-device (the
`fully_device` variant of the lowlatency decomposition). It is the cycle-time model's fixed term
(distinct from `iota_us`, which ADDS the host<->device input/output transfers on top of this
floor). The inproc-port contrast (the most aggressive transport) reduces to T_disp + bare compute,
so T_disp is the floor that bounds even a zero-wire transport. Baseline, transport-invariant.

WHAT run() MEASURES (1:1 with the model input). The intercept of the `fully_device` variant's
`time = intercept + slope*rows` fit (params staged, x pre-placed, output not pulled) — the
mlp_lowlatency decomposition's dispatch_floor_us. run() logs that intercept. The SEED is the v1
decomposition value (68.84 us, R2~0.997).

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

NAME = "T_disp_us"
# The co-fit PARTNER: T_disp (the dispatch-floor INTERCEPT) and cpp_inproc_port_t_row_bare (the bare-forward
# SLOPE) are the SAME `fully_device` fit (one fit, two read-offs — see the cpp-port bench docstring). So the
# harmonized Estimate this bench logs is that k=2 fit with T_disp's INTERCEPT as component 0 (the marginal
# manifest.value("T_disp_us") projects — 2 live model consumers read this floor) and the bare-forward slope
# as the partner carrying the off-diagonal (§4.2). NOTE (§4.2): this fit is a DIFFERENT fit from the staged
# (iota/t_row) one — they must NOT be cross-paired (different variants); only co-fit components pair.
PARTNER_NAME = "cpp_inproc_port_t_row_bare_us"
WARMUP = 8   # harness warmup phase (harness.warm): burn cold-compile forwards before measuring
MODULE_PATH = "leaf_eval_bound.benchmarks.bench_t_disp"
_DESC = ("Pure pjit/XLA dispatch floor (us): the irreducible per-forward dispatch with params+input "
         "staged device-resident and output on-device (fully_device variant). The cycle-time model's "
         "fixed term; the floor even a zero-wire inproc-port transport reduces to. Baseline, "
         "transport-invariant.")

_IN_DIM, _HIDDEN, _N_ACTIONS = 241, 256, 65


def get_seed() -> "G.Grounded":
    """The v1 seed (DISTRUST fallback): T_disp = 68.84 us (mlp_lowlatency decomposition dispatch_floor)."""
    # leaf_eval_grounding holds the dispatch floor as a bare float (DISPATCH_FLOOR_US, informational), not
    # a Grounded; wrap it here so the manifest's get_seed() contract (a Grounded-like) holds, with the
    # cycle-time model's sigma/cost for this term (model_cycletime._T_DISP).
    return G.Grounded(name="T_disp", mean=G.DISPATCH_FLOOR_US, sigma=2.0, cost=1.0, unit="us",
                      provenance="mlp_lowlatency/results.json decomposition.dispatch_floor_us (68.84, R2~0.997)",
                      estimability=G.Estimability.MEASURED, module="bench_t_disp")


def register_self() -> Any:
    from leaf_eval_bound.benchmarks.harness import register_quantity
    return register_quantity(NAME, quantity="dispatch_floor_cost", units=get_seed().unit,
                             description=_DESC, module_path=MODULE_PATH)


def _measure_raw(batches: Optional[list[int]] = None, iters: int = 200, repeat: int = 30) -> dict[str, Any]:
    """The raw-pool PROVENANCE producer (the §6 Phase-4 internal helper): Measure T_disp: the intercept of the `fully_device` variant fit. Delegates to the bench's full
    decomposition (bench_mlp_lowlatency.bench), which fits each variant and reports
    decomposition.dispatch_floor_us. Returns {'t_disp_us', 'r2', 'decomposition': {...}}. Imports jax
    lazily; pin the process (taskset -c 0). `measure()` wraps it into the fit Estimate; `run()` uses it for BOTH the Estimate and the raw provenance rows (ONE measurement, two consumers — P1)."""
    from chocofarm.az.bench.bench_mlp_lowlatency import bench

    batches = batches or [32, 64, 128, 192, 256, 384, 512]
    out = bench(batches, _IN_DIM, _HIDDEN, _N_ACTIONS, iters=iters, repeat=repeat, warmup=200)
    # bench() returns a dict with 'fits' (per variant) and 'decomposition'; the dispatch floor is the
    # fully_device intercept (== decomposition.dispatch_floor_us).
    decomp = out.get("decomposition", {})
    fits = out.get("fits", {})
    t_disp = decomp.get("dispatch_floor_us")
    if t_disp is None:  # fall back to the fully_device intercept directly (ADR-0002: surface the shape)
        t_disp = fits.get("fully_device", {}).get("intercept_us")
    if t_disp is None:
        raise RuntimeError(f"bench_t_disp.measure: neither decomposition.dispatch_floor_us nor "
                           f"fits.fully_device.intercept_us present in bench output keys {list(out)}")
    fd_fit = fits.get("fully_device", {})
    r2 = fd_fit.get("r2")
    # The fully_device per-width medians (the design points behind the fit) — the inputs the harmonized
    # k=2 Estimate's covariance is computed from (§4.2/§5: _fit_line discards them, so the bench recovers
    # them here from bench()'s per-variant record). bench() returns per_batch_us[variant] = {str(B):
    # {median_us, iqr_us}}; key it back to int widths in the swept order.
    pb_fd = out.get("per_batch_us", {}).get("fully_device", {})
    median_us = {int(B): float(pb_fd[str(B)]["median_us"]) for B in batches if str(B) in pb_fd}
    return {"t_disp_us": float(t_disp), "intercept_us": fd_fit.get("intercept_us"),
            "slope_us_per_row": fd_fit.get("slope_us_per_row"), "r2": r2,
            "per_width_median_us": median_us, "batches": batches, "decomposition": decomp}


def _estimate_from_raw(res: dict[str, Any]) -> "_est.Estimate":
    """Build this bench's harmonized `Estimate` from a `_measure_raw()` dict — the SINGLE home of the
    Estimate construction (P1), called by BOTH `measure()` and `run()`. The k2 staged/fully_device-fit
    Estimate with this bench's OWN quantity as component 0 (§4.2), the partner carrying the off-diagonal."""
    batches_used = [B for B in res["batches"] if B in res["per_width_median_us"]]
    medians = [res["per_width_median_us"][B] for B in batches_used]
    return fit_estimate(batches_used, medians, own_name=NAME, own_role="intercept", partner_name=PARTNER_NAME)


def measure(batches: Optional[list[int]] = None, iters: int = 200, repeat: int = 30) -> "_est.Estimate":
    """Measure T_disp and return its harmonized k=2 fit `Estimate` (§6 Phase 4: `measure()` returns
    the `Estimate` the bench DECLARES — the driver/untrusted_drive consume it directly). The raw
    design-point dict is the bench's internal `_measure_raw()` provenance. TIMING-SENSITIVE — pin (taskset -c 0)."""
    return _estimate_from_raw(_measure_raw(batches=batches, iters=iters, repeat=repeat))


def run(batches: Optional[list[int]] = None, iters: int = 200, repeat: int = 30) -> dict[str, Any]:
    """Measure T_disp and LOG it as a harmonized k=2 fit `Estimate` (§6 Phase 3): the fully_device-fit
    intercept/slope with their −0.81 off-diagonal, T_disp's INTERCEPT (== dispatch floor) as component 0.
    The fully_device per-width medians are logged as raw-design-point PROVENANCE — the variance authority
    is now `estimate.cov`, so the headline dispatch-floor scalar is NO LONGER double-logged as a sample row
    (the §5.2 de-dup obligation). TIMING-SENSITIVE — operator-invoked, pinned, never during the fan-out."""
    res = _measure_raw(batches=batches, iters=iters, repeat=repeat)  # ONE measurement (Estimate + provenance)
    batches_used = [B for B in res["batches"] if B in res["per_width_median_us"]]
    medians = [res["per_width_median_us"][B] for B in batches_used]
    # The k=2 fit Estimate, T_disp (the intercept) as component 0; the bare-forward slope the partner.
    est = _estimate_from_raw(res)  # the SAME Estimate measure() returns (P1)
    cfg = {"iters": iters, "repeat": repeat, "batches": batches_used, "fit_r2": res["r2"],
           "fit_intercept_us": res["intercept_us"], "fit_slope_us_per_row": res["slope_us_per_row"],
           "decomposition": res["decomposition"], "variant": "fully_device", "bench": "mlp_lowlatency"}
    with logged_run(NAME, quantity="dispatch_floor_cost", units=get_seed().unit, description=_DESC,
                    module_path=MODULE_PATH, config=cfg, estimate=est) as log:
        # PROVENANCE only (§5.2): the fully_device per-width medians. The headline dispatch floor is NOT
        # logged as a sample — it lives in estimate.theta_hat[0] (the SSOT).
        log(medians, sample_size=iters)
    return res


if __name__ == "__main__":
    print(f"[bench_t_disp] seed: {get_seed().mean} {get_seed().unit} (provenance: {get_seed().provenance})")
    register_self()
    print("[bench_t_disp] registered. NOT running the live measurement (timing-sensitive); "
          "invoke run() pinned and sole-workload.")
