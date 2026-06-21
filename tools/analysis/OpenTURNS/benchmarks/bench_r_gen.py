"""
tools/analysis/OpenTURNS/benchmarks/bench_r_gen.py
==================================================

LIVE benchmark for `R_gen` — the per-core single-generator decision rate (decisions/s/core):
the producer ceiling input. Baseline, transport-invariant (a generator core's search rate is
independent of which transport carries its leaf-eval requests; the transport binds the SERVE
stage, not generation). The aggregate producer cap is `N_gen * R_gen`.

WHAT run() MEASURES (1:1 with the model input). The decisions/s a SINGLE generator core sustains
with the leaf-eval mocked (so the read is the producer's own search throughput, not gated on the
server) — the C++ gen-ceiling sole-workload bench (cpp/src/*gen*/search_runtime_bench). That is a
C++ binary, not a Python microbench; `run()` records the v1 MEASURED value (152 dps/core, 4.0x
linear core scaling — adapter.md §2 line 93) with a config note that a fresh sole-workload C++
read (eval mocked) is the tightening measurement, and keeps the quantity flagged needs-measurement.

NOT a Python microbench (the live re-measure is the C++ gen bench); the recorded reading is the
v1 measured figure.

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

import estimate as _est  # noqa: E402  — the harmonized Estimate contract (measure() returns one — §6 Phase 4)
import leaf_eval_grounding as G  # noqa: E402
from bench_common import logged_run, pin_estimate  # noqa: E402

NAME = "R_gen"
MODULE_PATH = "benchmarks.bench_r_gen"
_DESC = ("Per-core single-generator decision rate (decisions/s/core): the producer ceiling input "
         "(aggregate = N_gen*R_gen). v1 MEASURED 152 dps/core, 4.0x linear core scaling. Live re-measure "
         "= the C++ gen-ceiling sole-workload bench (eval mocked). Baseline, transport-invariant.")


def get_seed() -> G.Grounded:
    """The v1 seed (DISTRUST fallback): R_gen=152 dps/core (MEASURED, 4.0x linear)."""
    return G.GEN_PER_CORE_DPS


def register_self() -> Any:
    from bench_common import register_quantity
    return register_quantity(NAME, quantity="producer_decisions_per_core", units=get_seed().unit,
                             description=_DESC, module_path=MODULE_PATH)


def _measure_raw() -> dict[str, Any]:
    """The raw-pool PROVENANCE producer (the §6 Phase-4 internal helper): the current R_gen estimate. A
    faithful re-measure is the C++ gen-ceiling sole-workload bench (eval mocked) — a binary, not a Python
    microbench — so this returns the v1 measured figure with a note. Returns {'r_gen_dps_per_core',
    'core_scaling', 'is_cpp_bench', 'note'}. `measure()` wraps the seed into a `Fixed` Estimate; `run()`
    uses this dict for the raw provenance row."""
    return {"r_gen_dps_per_core": get_seed().mean, "core_scaling": 4.0, "is_cpp_bench": True,
            "note": "v1 MEASURED (adapter.md §2 line 93); fresh sole-workload C++ read (eval mocked) tightens it"}


def _estimate_from_raw(res: dict[str, Any]) -> "_est.Estimate":
    """Build this bench's harmonized `Estimate` — the SINGLE home of the Estimate construction (P1),
    called by BOTH `measure()` and `run()`. A k=1 `Fixed` Estimate recovering the declared spread
    UN-DIVIDED (`cov=[[σ²]]`, the §5 store-bug fix). A pin has no sample n."""
    return pin_estimate(get_seed().mean, get_seed().sigma, name=NAME)


def measure() -> "_est.Estimate":
    """Measure R_gen and return its harmonized k=1 `Fixed` `Estimate` (§6 Phase 4: `measure()` returns the
    `Estimate` the bench DECLARES — a pin is a `Fixed`/declared-spread Estimate, NOT a faked pool, consumed
    directly by the driver/untrusted_drive). The raw dict is the bench's internal `_measure_raw()` provenance."""
    return _estimate_from_raw(_measure_raw())


def run() -> dict[str, Any]:
    """Logs a harmonized k=1 Fixed Estimate (§6 Phase 3) recovering the declared spread un-divided. Returns the estimate dict."""
    res = _measure_raw()  # the raw provenance dict
    est = _estimate_from_raw(res)  # the SAME Estimate measure() returns (P1)
    cfg = {"kind": "cpp_bench_measured", "core_scaling": res["core_scaling"],
           "needs_measurement": "fresh sole-workload C++ gen-ceiling read (eval mocked)", "note": res["note"]}
    with logged_run(NAME, quantity="producer_decisions_per_core", units=get_seed().unit, description=_DESC,
                    module_path=MODULE_PATH, config=cfg, estimate=est) as log:
        log(res["r_gen_dps_per_core"], sample_size=None)
    return res


if __name__ == "__main__":
    print(f"[bench_r_gen] seed: {get_seed().mean} {get_seed().unit} (MEASURED — {get_seed().provenance})")
    register_self()
    print("[bench_r_gen] registered. The live re-measure is the C++ gen-ceiling bench (eval mocked).")
