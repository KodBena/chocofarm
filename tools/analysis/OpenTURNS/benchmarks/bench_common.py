"""
tools/analysis/OpenTURNS/benchmarks/bench_common.py
===================================================

Shared scaffolding for the leaf-eval transport benchmark modules (`bench_<name>.py`), so
each bench owns ONLY its measurement loop + its v1 seed — never the register/connect/log
boilerplate (ADR-0012 P3 one-owner: this module owns the bench<->store glue; bench_store.py
owns the SQL; each bench owns its physics). A bench module is a thin object:

    SEED = ...                       # the v1 Grounded fallback (get_seed() returns it)
    def get_seed(): return SEED
    def register_self(): return register_quantity(NAME, quantity=…, units=…, …)
    def run(...): -> measure, then `with logged_run(NAME, config) as log: log(values)`

`logged_run` is the one helper the run() bodies use: it registers the definition (idempotent),
opens an instance with the repo git_sha + host + config, hands back a `log(values, sample_size)`
callable, and on exit leaves a populated instance. A bench that measures NOTHING during the
parallel workflow (timing-sensitive) still exposes run() — it just must not be CALLED then
(the manifest gates rerun behind an explicit operator action).

FAIL LOUD (ADR-0002). A registration/insert error propagates as a typed psycopg error. The
git_sha read is best-effort (a bench may run outside a checkout) and degrades to None, which is
a recorded provenance gap, not a swallowed failure.

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
_PARENT = os.path.dirname(_HERE)  # the OpenTURNS dir (holds bench_store, leaf_eval_grounding)
for _p in (_PARENT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import bench_store  # noqa: E402


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
) -> Iterator[Callable[..., None]]:
    """Open a measurement RUN for `name`: (1) register the definition (idempotent), (2) open an instance
    stamped with the repo git_sha + host + `config`, (3) yield a `log(values, sample_size=None, seq=None)`
    callable the run() body calls with its readings. On normal exit the instance is populated; an exception
    propagates (ADR-0002 — a half-measured run is surfaced, the partial samples already committed stay as
    provenance of the attempt). Usage:

        with logged_run(NAME, quantity=…, units=…, description=…, module_path=…, config={…}) as log:
            log(per_op_us_list, sample_size=iters)        # bulk readings
            log(single_reading)                            # one reading
    """
    def_id = register_quantity(
        name, quantity=quantity, units=units, description=description, module_path=module_path)
    inst_id = bench_store.open_instance(
        def_id, git_sha=repo_git_sha(), config=dict(config) if config else None)

    def log(values: Any, sample_size: Optional[int] = None, seq: Optional[int] = None) -> None:
        if isinstance(values, (list, tuple)):
            bench_store.log_samples(inst_id, list(values), sample_size=sample_size)
        else:
            bench_store.log_sample(inst_id, float(values), seq=seq, sample_size=sample_size)

    yield log
