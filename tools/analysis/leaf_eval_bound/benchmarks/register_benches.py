"""
tools/analysis/leaf_eval_bound/benchmarks/register_benches.py
============================================================

Seed the POSTGRES-DRIVEN registry with EVERY bench's quantity: DISCOVER every `bench_<name>.py`
module in this directory and call its `register_self()` (an INSERT of the `benchmark_definition` row
ONLY — no timing measurement, so it is SAFE to run during the parallel workflow). The manifest then
auto-discovers every registered quantity.

DISCOVERY-DRIVEN (the responsibility-refactor note's move 7, ADR-0011 Rule 4). The prior version
(`register_baseline.py`) hand-listed the 10 transport-invariant quantities in a `BASELINE_BENCHES`
constant — a fails-OPEN enumeration: a new bench had to be REMEMBERED into the list (the RCA's
smoking-gun shape). Now the bench MODULES on disk ARE the SSOT — every `bench_*.py` is discovered and
registered, so a new bench is registered by EXISTING (no list to edit, nothing to forget). The
variant-specific quantities (`bench_<slug>_*`) register alongside the transport-invariant ones, which
is BEHAVIOR-NEUTRAL for the bound: a registered seed-only quantity resolves to the SAME seed `Estimate`
as an unregistered one (`manifest.estimate` falls back to the seed either way — no samples => untrusted);
registering it just makes the registry COMPLETE + discoverable. (The file is renamed from
`register_baseline.py` — it no longer registers a "baseline" subset; ADR-0008 honest naming.)

The SEEDS stay UNTRUSTED (no samples) until a sole-workload run populates each quantity's
instance+samples — `manifest.value(name, trust=True)` returns trusted=False for a seed-only quantity
by construction. It does NOT run() any benchmark (the timing-sensitive measurements corrupt under
co-scheduling).

CLI: `python benchmarks/register_benches.py`  (idempotent — re-running refreshes the rows).

Public Domain (The Unlicense).
"""
from __future__ import annotations

import glob
import importlib
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.dirname(_HERE), _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def discover_bench_modules() -> list[str]:
    """Every bench module on disk — the `bench_<name>.py` files in this directory (the SSOT, replacing
    the hand-listed `BASELINE_BENCHES` — move 7, ADR-0011 Rule 4). Sorted for a deterministic
    registration order. A `bench_*.py` is the ONLY shape that matches (the split-out `estimators` /
    `pools` / `harness` and `register_benches` itself do not), so the glob IS the registerable set."""
    return [os.path.splitext(os.path.basename(p))[0]
            for p in sorted(glob.glob(os.path.join(_HERE, "bench_*.py")))]


def register_all() -> list[tuple[str, str]]:
    """Import each DISCOVERED bench and register its definition (INSERT only). Returns [(name, def_id), …].
    Idempotent (register_self upserts by UNIQUE name). Loud if postgres is down (registration is the
    registry write), or if a discovered `bench_*.py` lacks `register_self`/`NAME` (ADR-0002 — a malformed
    bench raises here, it is never silently skipped; the conformance test pins the contract run-free)."""
    out: list[tuple[str, str]] = []
    for mod_name in discover_bench_modules():
        mod = importlib.import_module(mod_name)
        def_id = mod.register_self()
        out.append((mod.NAME, str(def_id)))
    return out


def main() -> None:
    mods = discover_bench_modules()
    print(f"[register_benches] discovered {len(mods)} bench modules; registering their quantity "
          f"definitions (INSERT only, no timing)…")
    rows = register_all()
    for name, def_id in rows:
        print(f"  registered {name:<26} -> {def_id}")
    print(f"[register_benches] {len(rows)} definitions registered. Seeds stay UNTRUSTED until a "
          f"sole-workload run populates samples.")


if __name__ == "__main__":
    main()
