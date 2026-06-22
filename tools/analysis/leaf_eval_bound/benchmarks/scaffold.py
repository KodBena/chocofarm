"""
tools/analysis/leaf_eval_bound/benchmarks/scaffold.py

The shared bench SCAFFOLD (responsibility-refactor move 6, §2.5). Every `bench_<name>.py` repeats one
wiring skeleton: `measure = estimate_from_raw(measure_raw())`, a `register_self` registration, and a
`run()` that measures -> estimates -> `logged_run` -> logs the raw provenance. This module OWNS that
skeleton, so a bench declares only its bench-SPECIFIC parts (its seed, its `_measure_raw`, its estimator
`_estimate_from_raw`, its run config + run log) and gets `register_self`/`measure`/`run` for free -- the
structural close of the templated three-function shape the RCA's cancer-D originated in (the acute
duplication was already fixed; this removes the boilerplate a NEW bench would otherwise hand-copy).

A bench keeps `NAME`, `MODULE_PATH`, `get_seed`, `_measure_raw`, `_estimate_from_raw` as its own
module-level names (the manifest / conformance / discovery contract reads them), and binds the rest:

    _B = scaffold.bench(name=NAME, quantity="...", module_path=MODULE_PATH, description=_DESC,
                        seed=get_seed, measure_raw=_measure_raw, estimate_from_raw=_estimate_from_raw,
                        run_config=lambda res: {...},
                        run_log=lambda res, log: log(res["pool"], sample_size=1))
    register_self, measure, run = _B.register_self, _B.measure, _B.run

Public Domain (The Unlicense).
"""
from __future__ import annotations

import inspect
import sys
import types
from typing import Any, Callable, Mapping

from leaf_eval_bound.benchmarks.harness import logged_run
from leaf_eval_bound.contract import estimate as _est


def _live(fn: Any) -> Any:
    """Resolve a passed hook by its module + name at CALL time (so a bench's `_measure_raw` /
    `_estimate_from_raw` remain overridable via the module attr the tests monkeypatch); fall back
    to the original object for an unnamed callable (a lambda)."""
    return getattr(sys.modules.get(fn.__module__), fn.__name__, fn)


def bench(*, name: str, quantity: str, module_path: str, description: str,
          units: "str | None" = None,  # explicit unit for a bare-tuple seed (no .unit); else seed().unit
          seed: Callable[[], Any],
          measure_raw: Callable[..., Mapping[str, Any]],
          estimate_from_raw: Callable[[Mapping[str, Any]], Any],
          run_config: Callable[..., Mapping[str, Any]],
          run_log: Callable[..., None]) -> types.SimpleNamespace:
    """Wire a bench's `register_self`/`measure`/`run` from its bench-specific parts (move 6).

      * seed()                 -> the Grounded-like v1 seed; its `.unit` is the registered unit (read
                                  lazily, exactly as the hand-written benches did).
      * measure_raw(*a, **kw)  -> the ONE raw-provenance measurement; its own signature carries the
                                  sizing default (e.g. `reps=8`), so measure()/run() pass kwargs through.
      * estimate_from_raw(res) -> the bench's harmonized `Estimate` over that measurement (P1: the single
                                  home, shared by measure() and run()).
      * run_config(res)        -> the per-run config dict logged alongside the Estimate.
      * run_log(res, log)      -> emit the raw-provenance row(s) via the `logged_run` `log` callable
                                  (a callable, so a bench that logs zero / one / many rows all fit).
    """
    _mrsig = inspect.signature(measure_raw)

    def _full_kw(a: tuple, kw: dict) -> dict:
        bound = _mrsig.bind(*a, **kw); bound.apply_defaults()
        return dict(bound.arguments)

    def register_self() -> Any:
        from leaf_eval_bound.benchmarks.harness import register_quantity
        return register_quantity(name, quantity=quantity, units=(units if units is not None else seed().unit),
                                 description=description, module_path=module_path)

    def measure(*a: Any, **kw: Any) -> "_est.Estimate":
        return _live(estimate_from_raw)(_live(measure_raw)(*a, **kw))

    def run(*a: Any, **kw: Any) -> Mapping[str, Any]:
        res = _live(measure_raw)(*a, **kw)
        est = _live(estimate_from_raw)(res)
        fkw = _full_kw(a, kw)
        with logged_run(name, quantity=quantity, units=(units if units is not None else seed().unit), description=description,
                        module_path=module_path, config=run_config(res, **fkw), estimate=est) as log:
            run_log(res, log, **fkw)
        return res

    # Expose the bench's REAL sizing-kwarg signature (reps/iters/...) on measure()/run() -- the driver
    # and the conformance tests inspect.signature() these to find the budget knob; a bare *a,**kw hides it.
    measure.__signature__ = _mrsig.replace(return_annotation="_est.Estimate")
    run.__signature__ = _mrsig
    return types.SimpleNamespace(register_self=register_self, measure=measure, run=run)
