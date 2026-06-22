"""
tests/test_register_benches_discovery.py
========================================

The discovery NET for bench registration (responsibility-refactor move 7, ADR-0011 Rule 4): the
`register_benches` script DISCOVERS every `bench_*.py` on disk instead of a hand-listed
`BASELINE_BENCHES`, so a new bench is registered by existing (no list to forget). This pins:
(1) discovery finds every bench module on disk; (2) each discovered bench is REGISTERABLE (has
`register_self` + `NAME` — the contract `register_all` calls); (3) the old 10 transport-invariant
baseline benches are a SUBSET of the discovery (nothing baseline is lost — behavior-preserving).

Run-free: pure filesystem glob + import + hasattr. No postgres (the registration WRITE needs pg; the
DISCOVERY does not — this test exercises the discovery, not the write).

Public Domain (The Unlicense).
"""
from __future__ import annotations

import importlib
import os
import sys

# The leaf_eval_bound package PARENT goes on sys.path (the §3 layout: modules import each other as
# `leaf_eval_bound.<subpkg>.<module>`); the on-disk bench files live in the benchmarks sub-package.
_PKG_PARENT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "tools", "analysis")
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)
_BENCH_DIR = os.path.join(_PKG_PARENT, "leaf_eval_bound", "benchmarks")

from leaf_eval_bound.benchmarks import register_benches  # noqa: E402

# The pre-move hand-list — the 10 transport-invariant baseline quantities the discovery MUST still
# cover (nothing baseline is lost).
_OLD_BASELINE = {
    "bench_t_row", "bench_iota", "bench_tau_io", "bench_tmsg", "bench_lpd",
    "bench_r_gen", "bench_g_core", "bench_b_op", "bench_n_gen", "bench_t_disp"}


def test_discovery_finds_every_bench_on_disk() -> None:
    """`discover_bench_modules()` == the bench_*.py files on disk (the SSOT). A new bench appears here
    by existing, with no list to edit (the Rule-4 close)."""
    discovered = set(register_benches.discover_bench_modules())
    on_disk = {f[:-3] for f in os.listdir(_BENCH_DIR) if f.startswith("bench_") and f.endswith(".py")}
    assert discovered == on_disk
    assert len(discovered) >= len(_OLD_BASELINE)


def test_old_baseline_is_a_subset_of_discovery() -> None:
    """The 10 transport-invariant baseline benches are all discovered — the discovery is a SUPERSET of
    the old hand-list (behavior-preserving; it never drops one)."""
    missing = _OLD_BASELINE - set(register_benches.discover_bench_modules())
    assert not missing, f"discovery dropped baseline bench(es): {missing}"


def test_every_discovered_bench_is_registerable() -> None:
    """Every discovered bench exposes `register_self` + `NAME` (the contract `register_all` calls) — a
    bench_*.py that does not would raise at register time (ADR-0002); pinned run-free here."""
    for mod_name in register_benches.discover_bench_modules():
        mod = importlib.import_module("leaf_eval_bound.benchmarks." + mod_name)
        assert hasattr(mod, "register_self") and callable(mod.register_self), f"{mod_name}: no register_self"
        assert hasattr(mod, "NAME") and isinstance(mod.NAME, str), f"{mod_name}: no NAME"
