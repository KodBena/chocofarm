"""
tools/analysis/OpenTURNS/benchmarks/bench_t_disp.py
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

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.dirname(_HERE), _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import leaf_eval_grounding as G  # noqa: E402
from bench_common import logged_run  # noqa: E402

NAME = "T_disp_us"
MODULE_PATH = "benchmarks.bench_t_disp"
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
                      needs_measurement=False)


def register_self() -> Any:
    from bench_common import register_quantity
    return register_quantity(NAME, quantity="dispatch_floor_cost", units=get_seed().unit,
                             description=_DESC, module_path=MODULE_PATH)


def measure(batches: Optional[list[int]] = None, iters: int = 200, repeat: int = 30) -> dict[str, Any]:
    """Measure T_disp: the intercept of the `fully_device` variant fit. Delegates to the bench's full
    decomposition (bench_mlp_lowlatency.bench), which fits each variant and reports
    decomposition.dispatch_floor_us. Returns {'t_disp_us', 'r2', 'decomposition': {...}}. Imports jax
    lazily; pin the process (taskset -c 0)."""
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
    r2 = fits.get("fully_device", {}).get("r2")
    return {"t_disp_us": float(t_disp), "r2": r2, "decomposition": decomp}


def run(batches: Optional[list[int]] = None, iters: int = 200, repeat: int = 30) -> dict[str, Any]:
    """Measure T_disp and LOG it (the dispatch floor as the headline reading). TIMING-SENSITIVE —
    operator-invoked, pinned, never during the fan-out."""
    res = measure(batches=batches, iters=iters, repeat=repeat)
    cfg = {"iters": iters, "repeat": repeat, "fit_r2": res["r2"], "decomposition": res["decomposition"],
           "variant": "fully_device", "bench": "mlp_lowlatency"}
    with logged_run(NAME, quantity="dispatch_floor_cost", units=get_seed().unit, description=_DESC,
                    module_path=MODULE_PATH, config=cfg) as log:
        log(res["t_disp_us"], sample_size=iters)
    return res


if __name__ == "__main__":
    print(f"[bench_t_disp] seed: {get_seed().mean} {get_seed().unit} (provenance: {get_seed().provenance})")
    register_self()
    print("[bench_t_disp] registered. NOT running the live measurement (timing-sensitive); "
          "invoke run() pinned and sole-workload.")
