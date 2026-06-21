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

import leaf_eval_grounding as G  # noqa: E402
from bench_common import logged_run  # noqa: E402

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


def measure() -> dict[str, Any]:
    """The current R_gen estimate. A faithful re-measure is the C++ gen-ceiling sole-workload bench (eval
    mocked) — a binary, not a Python microbench — so this returns the v1 measured figure with a note.
    Returns {'r_gen_dps_per_core', 'core_scaling', 'is_cpp_bench', 'note'}."""
    return {"r_gen_dps_per_core": get_seed().mean, "core_scaling": 4.0, "is_cpp_bench": True,
            "note": "v1 MEASURED (adapter.md §2 line 93); fresh sole-workload C++ read (eval mocked) tightens it"}


def run() -> dict[str, Any]:
    """Record the current R_gen estimate to postgres as a single sample, flagged in config as the v1
    measured value awaiting a fresh sole-workload C++ read. Returns the estimate dict."""
    res = measure()
    cfg = {"kind": "cpp_bench_measured", "core_scaling": res["core_scaling"],
           "needs_measurement": "fresh sole-workload C++ gen-ceiling read (eval mocked)", "note": res["note"]}
    with logged_run(NAME, quantity="producer_decisions_per_core", units=get_seed().unit, description=_DESC,
                    module_path=MODULE_PATH, config=cfg) as log:
        log(res["r_gen_dps_per_core"], sample_size=None)
    return res


if __name__ == "__main__":
    print(f"[bench_r_gen] seed: {get_seed().mean} {get_seed().unit} (MEASURED — {get_seed().provenance})")
    register_self()
    print("[bench_r_gen] registered. The live re-measure is the C++ gen-ceiling bench (eval mocked).")
