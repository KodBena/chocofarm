# tools/analysis — offline structural-analysis tooling

These are **offline exploration / analysis utilities**. They are **NOT part of
the live `chocofarm` pipeline**: no module under `chocofarm/` imports them. They
operate on the abstract instance (treasures, faces, teleports, the exactly-K
prior) to re-derive structural quantities of a chocofarm instance as an
auditable *program* (and to run that same program against controlled synthetic
geometry).

## What lives here

- `analyzer.py` — `analyze(instance) -> StructuralReport`: the structural
  decomposition (cluster partition, reachable-belief sizing, detector-coupled
  quantities re-derived under the corrected face model).
- `synthetic.py` — a controlled-geometry instance generator that pushes random
  treasures + overlapping detection regions through the same
  `chocofarm.model.arrangement` the real map uses, so the analyzer can be
  exercised on geometry we control.
- `__init__.py` — package marker (empty).

## The live cluster partition is owned by `decomp.py`, not by this module

The single LIVE source of truth for the cluster partition is
`chocofarm/solvers/decomp.py`, which **recomputes** its partition independently
from the env's own face `cover_mask`s at runtime (`DecompPolicy`); it does not
import this module. The analyzer's `clusters` and `decomp.py`'s partition agree
by construction, but `decomp.py` owns the live value. Do **not** wire the
analyzer into `decomp.py`; these utilities are downstream/offline of the live
pipeline.

## Running

The package is importable from the repo root (the maintainer's run convention:
repo root on `sys.path`, no `pyproject`). Under `PYTHONPATH=.`:

```
PYTHONPATH=. python -m tools.analysis.synthetic [seed]      # synthetic instance report
PYTHONPATH=. python -c "from tools.analysis.analyzer import analyze, real_instance; \
    print(analyze(real_instance()))"
```

`tests/test_smoke.py` imports `tools.analysis.analyzer` to assert the analyzer
still constructs and runs against the real instance.

## Provenance

Relocated here from `chocofarm/analysis/` on **2026-06-15**, per the 2026-06-15
architectural audit's disposition: the audit verified that the live `chocofarm`
package does not import `analyzer.py` / `synthetic.py` (they were used only by
each other and by one smoke test), so they are offline exploration tooling
rather than live package code, and were moved out of the live package. See
`docs/notes/audit/architectural-audit-2026-06-15.md` (the point-in-time audit
record, left un-retro-edited) for the original finding.

Public Domain (The Unlicense).
