"""
tools/analysis/OpenTURNS/benchmarks/bench_n_gen.py
==================================================

LIVE benchmark for `n_gen` — the number of generator cores (cores): the FIXED isolation/pinning
layout fact (1 serve core + 3 gen cores on the 4-vCPU host; isolcpus 1-3). It is a CONFIG
quantity, not a measurement — the producer ceiling multiplier (aggregate = n_gen * R_gen).
Baseline, transport-invariant.

WHAT run() RECORDS. n_gen is decided by the deployment pinning (adapter.md §6 M3 1:3; the host's
4-vCPU isolcpus 1-3 layout in CLAUDE.md), so run() records the pinned value (3) with a config note
of its provenance. There is no microbench — the only way this "changes" is a different pinning
decision, which is a config change, not a measurement.

NOT timing-sensitive (recording a config fact).

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
from bench_common import logged_run, pin_estimate  # noqa: E402

NAME = "n_gen"
MODULE_PATH = "benchmarks.bench_n_gen"
_DESC = ("Generator cores (cores): the FIXED isolation/pinning layout (1 serve + 3 gen, isolcpus 1-3 on "
         "the 4-vCPU host). A config fact (producer ceiling multiplier, aggregate = n_gen*R_gen), not a "
         "measurement. Baseline, transport-invariant.")


def get_seed() -> G.Grounded:
    """The v1 seed (DISTRUST fallback): n_gen=3 cores (the 1:3 serve:gen pinning)."""
    return G.N_GEN_CORES


def register_self() -> Any:
    from bench_common import register_quantity
    return register_quantity(NAME, quantity="generator_cores", units=get_seed().unit,
                             description=_DESC, module_path=MODULE_PATH)


def measure() -> dict[str, Any]:
    """The pinned n_gen (a config fact, not a measurement). Returns {'n_gen', 'pinning', 'note'}."""
    return {"n_gen": get_seed().mean, "pinning": "1 serve + 3 gen (isolcpus 1-3)",
            "note": "config fact (adapter.md §6 M3 1:3; CLAUDE.md 4-vCPU host); no microbench"}


def run() -> dict[str, Any]:
    """Logs a harmonized k=1 Fixed Estimate (§6 Phase 3) recovering the declared spread un-divided. Returns the dict.
    (n_gen is a config fact — this records the deployment decision, not a timing measurement.)"""
    res = measure()
    est = pin_estimate(get_seed().mean, get_seed().sigma, name=NAME, constant=True)
    cfg = {"kind": "config_fact", "pinning": res["pinning"], "note": res["note"]}
    with logged_run(NAME, quantity="generator_cores", units=get_seed().unit, description=_DESC,
                    module_path=MODULE_PATH, config=cfg, estimate=est) as log:
        log(res["n_gen"], sample_size=None)
    return res


if __name__ == "__main__":
    print(f"[bench_n_gen] seed: {get_seed().mean} {get_seed().unit} (CONFIG FACT — {get_seed().provenance})")
    register_self()
    print("[bench_n_gen] registered. n_gen is a pinning/config fact (no microbench).")
