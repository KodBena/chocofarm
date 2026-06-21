"""
tools/analysis/OpenTURNS/benchmarks/bench_lpd.py
================================================

LIVE benchmark for `LPD` — leaves per recorded decision (leaves/decision): the unit-conversion
divisor from leaves/s to decisions/s (dps). Baseline, transport-invariant (a transport moves
I/O cost, not how many leaves a search expands per decision).

WHAT run() MEASURES (1:1 with the model input). LPD is a per-decision distinct-leaf count — its
faithful measurement is a PER-DECISION leaf-count HISTOGRAM from one instrumented
generation/search run (the count of distinct nodes a sims256/m24 Gumbel tree expands per
recorded decision). That instrumented run is a C++/search-harness artifact, not a Python
microbench; `run()` therefore records the v1 DESIGN PIN (500) as the current best estimate with
a config note that the histogram is the outstanding measurement, and leaves the quantity flagged
needs-measurement (so the manifest keeps reporting it untrusted until a histogram populates a
real sample distribution). The v1 seed (500) is explicitly a design pin, NOT a measured
histogram (analysis_clean.txt 76000/152=500 is a tautology, not an independent cross-check).

NOT timing-sensitive (recording a pin), but a real LPD measurement (the histogram) requires an
instrumented sole-workload search run.

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

NAME = "LPD"
MODULE_PATH = "benchmarks.bench_lpd"
_DESC = ("Leaves per recorded decision (leaves/decision): the leaves/s -> dps divisor. v1 = a DESIGN PIN "
         "(500, a sims256/m24 Gumbel tree's distinct-node count), NOT a measured histogram. Real "
         "measurement = a per-decision leaf-count histogram from an instrumented search run.")


def get_seed() -> G.Grounded:
    """The v1 seed (DISTRUST fallback): LPD=500 (DESIGN PIN, not a histogram; provenance in the seed)."""
    return G.LEAVES_PER_DECISION


def register_self() -> Any:
    from bench_common import register_quantity
    return register_quantity(NAME, quantity="leaves_per_decision", units=get_seed().unit,
                             description=_DESC, module_path=MODULE_PATH)


def _measure_raw() -> dict[str, Any]:
    """The raw-pool PROVENANCE producer (the §6 Phase-4 internal helper): the current LPD estimate. A
    faithful measurement is a per-decision leaf-count HISTOGRAM from an instrumented search run (a
    C++/search artifact, not a Python microbench), so this returns the v1 design pin with a note that
    the histogram is outstanding. Returns {'lpd', 'is_pin', 'note'}. `measure()` wraps the seed into a
    `Fixed` Estimate; `run()` uses this dict for the raw provenance row."""
    return {"lpd": get_seed().mean, "is_pin": True,
            "note": "design pin (sims256/m24 distinct-node count); histogram from instrumented run outstanding"}


def _estimate_from_raw(res: dict[str, Any]) -> "_est.Estimate":
    """Build this bench's harmonized `Estimate` — the SINGLE home of the Estimate construction (P1),
    called by BOTH `measure()` and `run()`. A k=1 `Fixed` Estimate recovering the declared spread
    UN-DIVIDED (`cov=[[σ²]]`, the §5 store-bug fix). A pin has no sample n."""
    return pin_estimate(get_seed().mean, get_seed().sigma, name=NAME)


def measure() -> "_est.Estimate":
    """Measure LPD and return its harmonized k=1 `Fixed` `Estimate` (§6 Phase 4: `measure()` returns the
    `Estimate` the bench DECLARES — a pin is a `Fixed`/declared-spread Estimate, NOT a faked pool, consumed
    directly by the driver/untrusted_drive). The raw dict is the bench's internal `_measure_raw()` provenance."""
    return _estimate_from_raw(_measure_raw())


def run() -> dict[str, Any]:
    """Logs a harmonized k=1 Fixed Estimate (§6 Phase 3) recovering the declared spread un-divided. Returns the estimate dict. (Recording a pin is not
    timing-sensitive; the real histogram measurement is the outstanding sole-workload run.)"""
    res = _measure_raw()  # the raw provenance dict
    est = _estimate_from_raw(res)  # the SAME Estimate measure() returns (P1)
    cfg = {"kind": "design_pin", "needs_measurement": "per-decision leaf-count histogram (instrumented search run)",
           "note": res["note"]}
    with logged_run(NAME, quantity="leaves_per_decision", units=get_seed().unit, description=_DESC,
                    module_path=MODULE_PATH, config=cfg, estimate=est) as log:
        log(res["lpd"], sample_size=None)   # a single recorded reading (NULL sample_size — not an aggregate)
    return res


if __name__ == "__main__":
    print(f"[bench_lpd] seed: {get_seed().mean} {get_seed().unit} (DESIGN PIN — {get_seed().provenance})")
    register_self()
    print("[bench_lpd] registered. The real measurement is a per-decision leaf-count histogram from an "
          "instrumented search run (outstanding).")
