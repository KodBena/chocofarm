"""
tools/analysis/leaf_eval_bound/benchmarks/harness.py
===================================================

The bench↔store GLUE — registration + the logged-run context manager + the warmup phase, split out
of `bench_common` (the responsibility-refactor note's move 1, ADR-0012 P3 one-owner: this owns the
bench<->store wiring; `bench_store.py` owns the SQL; the estimator factories are `estimators.py`, the
pool builders `pools.py`). A bench is a thin object:

    def register_self(): return register_quantity(NAME, quantity=…, units=…, …)
    def run(...): -> measure, then `with logged_run(NAME, …, estimate=…) as log: log(values)`

  * `register_quantity` — idempotently register the bench's `benchmark_definition` row.
  * `logged_run`        — open an instance (git_sha + host + config), OPTIONALLY persist a harmonized
                          `Estimate` jsonb (the §5.1 SSOT), yield a `log(values)` provenance callable.
  * `repo_git_sha` / `SIZING_KWARGS` / `warm` — provenance, the sizing-knob vocabulary, the warmup phase.

FAIL LOUD (ADR-0002): a registration/insert error propagates as a typed psycopg error; the git_sha
read degrades to None (a recorded provenance gap, not a swallowed failure).

Public Domain (The Unlicense).
"""
from __future__ import annotations

import contextlib
import os
import subprocess
import sys
import uuid
from typing import Any, Callable, Iterator, Mapping, Optional

_HERE = os.path.dirname(os.path.abspath(__file__))

from leaf_eval_bound.store import bench_store  # noqa: E402
from leaf_eval_bound.contract import estimate as _est  # noqa: E402  — for the logged_run `estimate` annotation (the harmonized contract)


def repo_git_sha() -> Optional[str]:
    """The repo HEAD short-SHA for sample provenance, or None outside a checkout (best-effort — a missing
    SHA is a recorded gap, not a failure: a sole-workload bench run on a detached tree still logs)."""
    try:
        out = subprocess.run(
            ["git", "-C", _HERE, "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip() or None if out.returncode == 0 else None
    except Exception:
        return None


def register_quantity(
    name: str, *, quantity: str, units: str, description: str, module_path: str
) -> uuid.UUID:
    """Idempotently register this bench's quantity (a `benchmark_definition` row) and return its id. The
    `module_path` SHOULD be the dotted import path of the bench module (e.g. `benchmarks.bench_t_row`) so
    the manifest can re-import it for get_seed()/run(). Loud if postgres is down (registration is the
    registry write)."""
    return bench_store.register_definition(
        name, quantity=quantity, units=units, description=description, module_path=module_path)


@contextlib.contextmanager
def logged_run(
    name: str,
    *,
    quantity: str,
    units: str,
    description: str,
    module_path: str,
    config: Optional[Mapping[str, Any]] = None,
    estimate: Optional["_est.Estimate"] = None,
) -> Iterator[Callable[..., None]]:
    """Open a measurement RUN for `name`: (1) register the definition (idempotent), (2) open an instance
    stamped with the repo git_sha + host + `config`, (3) if an `estimate` is given, persist it as the
    instance's `estimate` jsonb (the §5.1 SSOT of the measured object — the harmonized `Estimate`),
    (4) yield a `log(values, sample_size=None, seq=None)` callable the run() body calls with its raw-reading
    PROVENANCE rows. On normal exit the instance is populated; an exception propagates (ADR-0002 — a
    half-measured run is surfaced, the partial samples already committed stay as provenance of the attempt).

    The `estimate` is the §6 Phase-3 path: a migrated bench computes its `Estimate` in `measure()` and passes
    it here, and the raw `log(...)` rows become PROVENANCE only — the variance authority is the jsonb, so the
    headline scalar must NOT be re-logged as a sample row (the §5.2 de-dup obligation, which corrupts
    `latest_aggregate`'s count). Usage:

        # legacy (mean/median bench): no estimate yet, raw pool is the authority
        with logged_run(NAME, quantity=…, units=…, description=…, module_path=…, config={…}) as log:
            log(per_op_us_list, sample_size=iters)        # bulk readings
            log(single_reading)                            # one reading
        # Phase-3 (fit bench): the Estimate is the SSOT; log only raw-design-point provenance
        with logged_run(NAME, …, estimate=est) as log:
            log(per_width_medians, sample_size=iters)     # provenance (NOT the headline scalar)
    """
    def_id = register_quantity(
        name, quantity=quantity, units=units, description=description, module_path=module_path)
    inst_id = bench_store.open_instance(
        def_id, git_sha=repo_git_sha(), config=dict(config) if config else None)
    if estimate is not None:
        # The harmonized Estimate is the SSOT (§5.1); persist it onto the instance up-front so the jsonb is
        # present even if a later raw-provenance log() raises. Validated at construction (ADR-0002).
        bench_store.set_estimate(inst_id, estimate)

    def log(values: Any, sample_size: Optional[int] = None, seq: Optional[int] = None) -> None:
        if isinstance(values, (list, tuple)):
            bench_store.log_samples(inst_id, list(values), sample_size=sample_size)
        else:
            bench_store.log_sample(inst_id, float(values), seq=seq, sample_size=sample_size)

    yield log


# The recognized "how many units of work" keyword a bench's measure()/run() may expose so a caller can
# SIZE one call: the Neyman drive's _make_measurer passes the allocated budget through it, warm() below
# passes the burn-in count. A bench names its sizing knob ONE of these; the caller introspects measure()'s
# signature and uses the first match (None => no sizing knob, e.g. a pin). SINGLE HOME (ADR-0012 P1):
# untrusted_drive._ITERS_KW ALIASES this — the names are NOT re-listed anywhere else. `budget` is the
# drive's own canonical term for the lever (its measurer wrapper is `def measure(budget)`); `leaves` is the
# cpp-inproc tmsg bench's honest per-leaf-count knob. Both size a SHRINKABLE quantity; omitting them left
# those benches showing budget-kw "None" in the drive (shrinkable-but-un-sizable — silently de-funded).
SIZING_KWARGS = ("cycles", "trials", "iters", "n_trials", "reps", "rounds", "samples", "n",
                 "budget", "leaves")


def warm(mod: Any, **kwargs: Any) -> None:
    """The harness WARMUP PHASE for a registered bench, run ONCE before the measured phase so a cold
    transient (a first cold JAX forward at ~hundreds of us/row vs the ~few-us/row steady state; a cache
    fill; socket setup) never poisons the recorded samples. OPT-IN: a bench advertises EITHER a
    `warmup(**kwargs)` callable (its OWN warmup — the harness calls it and does NOT care what it does)
    OR a module-level `WARMUP` int (the harness runs `measure()` for that many discarded iterations, a
    generic burn-in). A bench advertising NEITHER gets no warmup phase. Everything a warmup produces is
    DISCARDED — never logged, never returned."""
    fn = getattr(mod, "warmup", None)
    if callable(fn):
        fn(**kwargs)
        return
    n = int(getattr(mod, "WARMUP", 0) or 0)
    if n > 0 and hasattr(mod, "measure"):
        import inspect
        params = inspect.signature(mod.measure).parameters
        iters_kw = next((k for k in SIZING_KWARGS if k in params), None)
        mod.measure(**({iters_kw: n} if iters_kw else {}))  # discarded — the generic burn-in
