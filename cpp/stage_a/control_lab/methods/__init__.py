#!/usr/bin/env python3
"""
cpp/stage_a/control_lab/methods/ — the candidate controller-method package for the issue-gate control lab.

Each module here implements ONE candidate Controller (or TrainableRecipe) against the FROZEN
adapter.Controller contract and SELF-REGISTERS it into adapter.REGISTRY at import time (a new method is one
new file + one `REGISTRY.setdefault(...)` — no edit to any shared file, so a parallel fan-out of method
authors touches only disjoint files).

Discovery is EXPLICIT via `load_all()`, NOT run at package import — so importing a single submodule
(`control_lab.methods.<name>`, e.g. a method's own unit test) does NOT pull in its siblings. That keeps a
parallel fan-out's per-method tests isolated: a half-written sibling cannot break another method's test.
The harness calls `load_all()` once at startup to register every method.

Fail-loud (ADR-0002): `load_all()` lets a module that fails to import surface its error rather than silently
dropping the method — a broken candidate is a loud failure at the single load point, never an invisible no-show.

Public Domain (The Unlicense).
"""
from __future__ import annotations

import importlib
import pkgutil


def load_all() -> None:
    """Import every non-private sibling module so each self-registers into adapter.REGISTRY. Called once by
    the harness at startup. Fail-loud: a module raising on import propagates (a broken method is loud)."""
    for info in pkgutil.iter_modules(__path__):
        if not info.name.startswith("_"):
            importlib.import_module(f"{__name__}.{info.name}")
