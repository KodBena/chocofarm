"""
tools/analysis/leaf_eval_bound/benchmarks/bench_n_gen.py
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

The uniform measure/register_self/run wiring is the shared `scaffold.bench` (move 6); this module
declares only n_gen's bench-specific parts (seed, _measure_raw, _estimate_from_raw, run config/log).

Public Domain (The Unlicense).
"""
from __future__ import annotations

from typing import Any

from leaf_eval_bound.contract import estimate as _est  # noqa: E402  — the harmonized Estimate contract (measure() returns one — §6 Phase 4)
from leaf_eval_bound.contract import grounding as G  # noqa: E402
from leaf_eval_bound.benchmarks.estimators import pin_estimate  # noqa: E402
from leaf_eval_bound.benchmarks.scaffold import bench as _scaffold  # noqa: E402  — move 6 wiring

NAME = "n_gen"
MODULE_PATH = "leaf_eval_bound.benchmarks.bench_n_gen"
_DESC = ("Generator cores (cores): the FIXED isolation/pinning layout (1 serve + 3 gen, isolcpus 1-3 on "
         "the 4-vCPU host). A config fact (producer ceiling multiplier, aggregate = n_gen*R_gen), not a "
         "measurement. Baseline, transport-invariant.")


def get_seed() -> G.Grounded:
    """The v1 seed (DISTRUST fallback): n_gen=3 cores (the 1:3 serve:gen pinning)."""
    return G.N_GEN_CORES


def _measure_raw() -> dict[str, Any]:
    """The raw-pool PROVENANCE producer (the §6 Phase-4 internal helper): the pinned n_gen (a config fact,
    not a measurement). Returns {'n_gen', 'pinning', 'note'}. `measure()` wraps the seed into a `Fixed`
    Estimate; `run()` uses this dict for the raw provenance row."""
    return {"n_gen": get_seed().mean, "pinning": "1 serve + 3 gen (isolcpus 1-3)",
            "note": "config fact (adapter.md §6 M3 1:3; CLAUDE.md 4-vCPU host); no microbench"}


def _estimate_from_raw(res: dict[str, Any]) -> "_est.Estimate":
    """Build this bench's harmonized `Estimate` — the SINGLE home of the Estimate construction (P1),
    called by BOTH `measure()` and `run()`. A k=1 `Fixed` Estimate recovering the declared spread
    UN-DIVIDED (`cov=[[σ²]]`, the §5 store-bug fix). A pin has no sample n. `constant` is DERIVED from
    the grounding SSOT (`Grounded.constant`, here True — n_gen is a layout fact), not hardcoded: a true
    constant is `family=DEGENERATE` (~0 bound contribution, §3), matching the manifest's seed path (P1)."""
    seed = get_seed()
    return pin_estimate(seed.mean, seed.sigma, name=NAME, constant=seed.constant)


# Move 6: the shared scaffold wires register_self / measure / run from the bench-specific parts above.
# n_gen is a config fact, so run() logs the pinned value (sample_size=None) with a config-provenance dict.
_B = _scaffold(
    name=NAME, quantity="generator_cores", module_path=MODULE_PATH, description=_DESC,
    seed=get_seed, measure_raw=_measure_raw, estimate_from_raw=_estimate_from_raw,
    run_config=lambda res, **kw: {"kind": "config_fact", "pinning": res["pinning"], "note": res["note"]},
    run_log=lambda res, log, **kw: log(res["n_gen"], sample_size=None),
)
register_self, measure, run = _B.register_self, _B.measure, _B.run


if __name__ == "__main__":
    print(f"[bench_n_gen] seed: {get_seed().mean} {get_seed().unit} (CONFIG FACT — {get_seed().provenance})")
    register_self()
    print("[bench_n_gen] registered. n_gen is a pinning/config fact (no microbench).")
