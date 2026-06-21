"""
tools/analysis/OpenTURNS/benchmarks/bench_g_core.py
===================================================

LIVE benchmark for `g_core` — the per-core generation LEAF rate (leaves/s/core): the producer
ceiling input in LEAF units (the capacity model uses leaves/s; the cycle model uses dps/core).
g_core = R_gen * LPD; it is the SAME physical generation measurement as `R_gen`, expressed in
leaves/s rather than dps/core. Baseline, transport-invariant.

WHAT run() MEASURES (1:1 with the model input). The same C++ gen-ceiling sole-workload bench that
grounds R_gen (eval mocked); g_core is its leaves/s read (v1: 76000 leaves/s/core = 152 dps/core
* 500 leaves/decision, adapter.md §2 line 93). `run()` records the v1 measured leaves/s figure
with a note that the fresh C++ read tightens it, and keeps the quantity flagged needs-measurement.

NOT a Python microbench (the live re-measure is the C++ gen bench).

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

NAME = "g_core"
MODULE_PATH = "benchmarks.bench_g_core"
_DESC = ("Per-core generation leaf rate (leaves/s/core): producer ceiling input in leaf units "
         "(= R_gen*LPD). Same C++ gen-ceiling measurement as R_gen, in leaves/s. v1 76000 leaves/s/core "
         "(MEASURED). Baseline, transport-invariant.")


def get_seed() -> G.Grounded:
    """The v1 seed (DISTRUST fallback): g_core=76000 leaves/s/core (MEASURED)."""
    return G.GEN_PER_CORE_LEAVES


def register_self() -> Any:
    from bench_common import register_quantity
    return register_quantity(NAME, quantity="producer_leaves_per_core", units=get_seed().unit,
                             description=_DESC, module_path=MODULE_PATH)


def _measure_raw() -> dict[str, Any]:
    """The raw-pool PROVENANCE producer (the §6 Phase-4 internal helper): the current g_core estimate
    (leaves/s/core), the same C++ gen-ceiling measurement as R_gen in leaf units. Returns
    {'g_core_leaves_per_core', 'is_cpp_bench', 'note'}. `measure()` wraps the seed into a `Fixed` Estimate;
    `run()` uses this dict for the raw provenance row."""
    return {"g_core_leaves_per_core": get_seed().mean, "is_cpp_bench": True,
            "note": "v1 MEASURED (adapter.md §2 line 93 = 152 dps/core * 500 LPD); fresh C++ read tightens it"}


def _estimate_from_raw(res: dict[str, Any]) -> "_est.Estimate":
    """Build this bench's harmonized `Estimate` — the SINGLE home of the Estimate construction (P1),
    called by BOTH `measure()` and `run()`. A k=1 `Fixed` Estimate recovering the declared spread
    UN-DIVIDED (`cov=[[σ²]]`, the §5 store-bug fix). A pin has no sample n."""
    return pin_estimate(get_seed().mean, get_seed().sigma, name=NAME)


def measure() -> "_est.Estimate":
    """Measure g_core and return its harmonized k=1 `Fixed` `Estimate` (§6 Phase 4: `measure()` returns the
    `Estimate` the bench DECLARES — a pin is a `Fixed`/declared-spread Estimate, NOT a faked pool, consumed
    directly by the driver/untrusted_drive). The raw dict is the bench's internal `_measure_raw()` provenance."""
    return _estimate_from_raw(_measure_raw())


def run() -> dict[str, Any]:
    """Logs a harmonized k=1 Fixed Estimate (§6 Phase 3) recovering the declared spread un-divided. Returns the estimate dict."""
    res = _measure_raw()  # the raw provenance dict
    est = _estimate_from_raw(res)  # the SAME Estimate measure() returns (P1)
    cfg = {"kind": "cpp_bench_measured",
           "needs_measurement": "fresh sole-workload C++ gen-ceiling read (eval mocked)", "note": res["note"]}
    with logged_run(NAME, quantity="producer_leaves_per_core", units=get_seed().unit, description=_DESC,
                    module_path=MODULE_PATH, config=cfg, estimate=est) as log:
        log(res["g_core_leaves_per_core"], sample_size=None)
    return res


if __name__ == "__main__":
    print(f"[bench_g_core] seed: {get_seed().mean} {get_seed().unit} (MEASURED — {get_seed().provenance})")
    register_self()
    print("[bench_g_core] registered. The live re-measure is the C++ gen-ceiling bench (eval mocked).")
