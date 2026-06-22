"""
tools/analysis/leaf_eval_bound/benchmarks/register_baseline.py
========================================================

Seed the POSTGRES-DRIVEN registry with the BASELINE (transport-invariant + ZMQ-baseline)
quantities: import each `bench_<name>` module and call its `register_self()` (an INSERT of the
`benchmark_definition` row ONLY — no timing measurement, so it is SAFE to run during the
parallel workflow). The manifest then auto-discovers every registered quantity.

This registers the v1 grounding set (leaf_eval_grounding.py): t_row, iota, tau_io, tmsg, LPD,
R_gen, g_core, B_op, n_gen, T_disp. The SEEDS stay UNTRUSTED (no samples) until a sole-workload
run populates each quantity's instance+samples — `manifest.value(name, trust=True)` returns
trusted=False for a seed-only quantity by construction.

It does NOT run() any benchmark (the timing-sensitive measurements corrupt under co-scheduling).

CLI: `python benchmarks/register_baseline.py`  (idempotent — re-running refreshes the rows).

Public Domain (The Unlicense).
"""
from __future__ import annotations

import importlib
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.dirname(_HERE), _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# The baseline quantity modules (one benchmark_definition per quantity).
BASELINE_BENCHES = [
    "bench_t_row", "bench_iota", "bench_tau_io", "bench_tmsg", "bench_lpd",
    "bench_r_gen", "bench_g_core", "bench_b_op", "bench_n_gen", "bench_t_disp",
]


def register_all() -> list[tuple[str, str]]:
    """Import each baseline bench and register its definition (INSERT only). Returns [(name, def_id), …].
    Idempotent (register_self upserts by UNIQUE name). Loud if postgres is down (registration is the
    registry write)."""
    out: list[tuple[str, str]] = []
    for mod_name in BASELINE_BENCHES:
        mod = importlib.import_module(mod_name)
        def_id = mod.register_self()
        out.append((mod.NAME, str(def_id)))
    return out


def main() -> None:
    print("[register_baseline] registering baseline quantity definitions (INSERT only, no timing)…")
    rows = register_all()
    for name, def_id in rows:
        print(f"  registered {name:<14} -> {def_id}")
    print(f"[register_baseline] {len(rows)} definitions registered. Seeds stay UNTRUSTED until a "
          f"sole-workload run populates samples.")


if __name__ == "__main__":
    main()
