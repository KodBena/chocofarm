"""
tools/analysis/leaf_eval_bound/manifest.py
====================================

The SSOT REGISTRY of measurable physical quantities for the leaf-eval transport-design
sweep — importable by EVERY throughput-model module (a transport variant's
`model_<slug>.py`), so a model trusts/distrusts an input through ONE contract rather than
hand-copying a literal. It resolves a quantity name to:

    (mean, sigma, n, trusted)

via `manifest.value(name, trust=True)`:

  * TRUST  (trust=True, the default): the LATEST measured (mean, sigma, n) from the
    postgres metric store (`bench_store.latest_aggregate` — the most-recent instance's
    sample aggregate). `trusted=True` iff a live measurement exists; if the quantity has no
    samples yet (seeds stay untrusted until a sole-workload run populates them), it FALLS
    BACK to the seed and returns `trusted=False`, so a model never silently consumes an
    unbacked number believing it measured.

  * DISTRUST (trust=False): the v1 SEED estimate from the quantity's bench module's
    `get_seed()`, flagged `trusted=False`. The escape hatch for "I want the grounded
    starting point, not whatever the DB currently holds" — and the way a model runs BEFORE
    any benchmark has populated postgres.

§6 PHASE 1 — THE MANIFEST AS THE `Estimate` SEAM (additive; ZERO 4-tuple behavior change).
Every resolved `Quantity` now CARRIES a harmonized `estimate.Estimate` ALONGSIDE the legacy
(mean, sigma, n, trusted) — so every downstream consumer becomes `Estimate`-capable WITHOUT
being touched (Phase 2/3 wire the driver and the benches). The three resolution paths each
fill the contract (docs/design/harmonized-estimator-interface.md §5/§6):

  * TRUST + a STORED estimate: `bench_store.latest_estimate(name)` (the bench's COMPUTED
    Estimate — carries SE(slope)/Cov(slope,intercept) the sample aggregate provably cannot
    recover). Preferred over the aggregate when an instance carries one.
  * TRUST + NO stored estimate (a legacy instance): reconstruct a k=1 `Poolwise` Estimate
    from `latest_aggregate` — `cov=[[sigma^2/n]]` (the already-divided SE^2),
    `Poolwise(per_sample_var=[sigma^2])`, `family=NORMAL`. So old data still resolves.
  * SEED (DISTRUST, or a TRUST fall-back): a `Fixed`-law Estimate from the bench's
    `get_seed()` Grounded — `theta_hat=[mean]`, `cov=[[sigma^2]]` (the declared 1-sigma
    spread IS the variance; a prior is un-shrinkable), `family=NORMAL`.

THE 4-TUPLE IS PRESERVED AS A PROJECTION OF THE Estimate (the confirmed fixed point). `value()`
keeps its signature and return type; `quantity()` builds its (mean, sigma, n) BY PROJECTING the
carried estimate (`reconstruct._project_estimate`), so a pool-fed caller and an `Estimate`-fed caller agree
byte-for-byte on the mean case. For a `Poolwise` mean the projection recovers the per-sample
spread `sigma = sqrt(per_sample_var[0])` (NOT sqrt(cov[0,0]) — that is the SE; the 4-tuple's
sigma is the per-sample stddev_samp every model consumes as Normal(mean, sigma)) and
`n = round(per_sample_var[0] / cov[0,0])`; the legacy reconstruction and the projection are
exact inverses on the aggregate. The new `estimate(name, …)` accessor exposes the Estimate
directly. Nothing in Phase 1 CONSUMES the estimate yet — it is carried so Phase 2/3 can.

POSTGRES-DRIVEN (no shared-file write contention). The registry is the
`benchmark_definition` TABLE, not a dict in this file: `discover()` enumerates quantities
by SELECTing that table, and each definition's `module_path` names the bench module that
owns the quantity's `get_seed()` (the seed) and `run()` (the live re-measure). So a design
agent REGISTERS a new quantity by INSERTing a definition row + writing its bench module —
NEVER by editing this file. Two agents adding two quantities never touch the same file.

GRACEFUL DEGRADATION (import-clean if postgres is down). Importing this module does NOT
touch postgres. The FIRST call that needs the DB (a TRUST read, or `discover()`) attempts
a connection; on ANY connection failure it logs once to stderr and the manifest runs in
SEED-ONLY mode — every `value()` returns its seed flagged `trusted=False`. This is loud
(ADR-0002: the degradation is announced, never silent) but non-fatal (a model on a host
without the DB still computes its bound from seeds, exactly as the v1 models did). A SQL
error on a connection that DID open is NOT swallowed — that is a real fault, not a
"postgres absent" condition.

RE-RUN ON DEMAND. `value(name, trust=True, rerun=True)` (or `measure(name)`) imports the
quantity's bench module and calls `run()` to populate a fresh instance+samples, THEN reads
the new aggregate — the "re-run the live benchmark" path the brief asks for. Guarded:
`run()` may be timing-sensitive (the parallel workflow corrupts timing), so a model NEVER
passes `rerun=True` during the fan-out; it is an explicit operator action.

Dependencies: numpy + the stdlib + `bench_store` (psycopg3, lazily). openturns is NOT
required (models import this for the numbers; the gradient/CI machinery is the driver's, via jax/scipy).

Public Domain (The Unlicense).
"""
from __future__ import annotations

import importlib
import importlib.util
import math
import os
import sys
from dataclasses import dataclass
from typing import Any, Optional


# bench_store is imported LAZILY (inside the functions that touch the DB) so importing the
# manifest never opens a connection — the import-clean / graceful-degradation contract.
#
# estimate.py is the harmonized-estimator TYPE SSOT (the `Estimate` contract + its ShrinkLaw /
# Support / CIFamily). It touches NO DB and imports only stdlib + numpy, so importing it at
# module top PRESERVES the import-clean contract (it is the same numpy-only surface this module
# already requires). It carries the Phase-1 Estimate seam.
import numpy as np  # noqa: E402
from leaf_eval_bound.contract import estimate as _est  # noqa: E402
from leaf_eval_bound.store import reconstruct  # noqa: E402  — the lifted seed/aggregate->Estimate + projection glue (move 2)


# --------------------------------------------------------------------------- #
# Result container
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Quantity:
    """One resolved quantity. `trusted` is the load-bearing flag: True ONLY when `mean`/`sigma`/`n`
    came from a live postgres measurement; False when it is the v1 seed (DISTRUST, or a TRUST read that
    fell back because no samples exist yet). `source` records which path produced it ('postgres' |
    'postgres(estimate)' | 'seed' | 'seed(no-samples)' | 'seed(pg-down)').

    §6 PHASE 1: `estimate` carries the harmonized `estimate.Estimate` ALONGSIDE the legacy 4-tuple
    (additive — every existing `.mean`/`.sigma`/`.n`/`.trusted` reader is unchanged). The (mean,
    sigma, n) are a PROJECTION of this estimate (`reconstruct._project_estimate`), so the 4-tuple and the
    Estimate cannot disagree. `estimate` defaults to None ONLY so the dataclass stays constructible
    in a degenerate path; every `quantity()` resolution sets it (a None `estimate` on a resolved
    Quantity would be a Phase-1 bug)."""
    name: str
    mean: float
    sigma: float
    n: int
    trusted: bool
    units: str = ""
    source: str = "seed"
    estimate: Optional["_est.Estimate"] = None

    def as_tuple(self) -> tuple[float, float, int, bool]:
        """The (mean, sigma, n, trusted) 4-tuple the brief's `value()` API returns."""
        return (self.mean, self.sigma, self.n, self.trusted)


# --------------------------------------------------------------------------- #
# Postgres-down latch (announce once, then run seed-only)
# --------------------------------------------------------------------------- #
_PG_DOWN: bool = False
_PG_DOWN_ANNOUNCED: bool = False


def _announce_pg_down(exc: Exception) -> None:
    global _PG_DOWN, _PG_DOWN_ANNOUNCED
    _PG_DOWN = True
    if not _PG_DOWN_ANNOUNCED:
        _PG_DOWN_ANNOUNCED = True
        print(f"[manifest] postgres unreachable ({type(exc).__name__}: {exc}); "
              f"running SEED-ONLY — every value() is trusted=False until the DB is back.",
              file=sys.stderr)


def postgres_available() -> bool:
    """Whether the metric store is reachable (a cheap probe). Latches `_PG_DOWN` on failure so the rest
    of the session runs seed-only without re-probing on every call. A successful probe CLEARS the latch
    (the DB came back)."""
    global _PG_DOWN
    try:
        from leaf_eval_bound.store import bench_store
        with bench_store.connect() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                cur.fetchone()
        _PG_DOWN = False
        return True
    except Exception as exc:  # connection failure -> seed-only (announced, not silent)
        _announce_pg_down(exc)
        return False


# --------------------------------------------------------------------------- #
# Bench-module resolution (the module_path in the definition row -> get_seed()/run())
# --------------------------------------------------------------------------- #
def _import_bench_module(module_path: str) -> Any:
    """Import a quantity's bench module by its registered `module_path`. Accepts EITHER a dotted import
    path (`benchmarks.bench_t_row`) or a filesystem path to the .py — so a definition row can carry
    whichever the author finds natural. A missing/broken module is a loud ImportError (ADR-0002 — a
    registered quantity whose bench cannot load is a real fault, surfaced, not swallowed)."""
    if module_path.endswith(".py") or os.path.sep in module_path:
        path = module_path if os.path.isabs(module_path) else os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), module_path)
        name = "bench_" + os.path.splitext(os.path.basename(path))[0]
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"manifest: cannot load bench module from path {path!r}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    # Normalize a pre-§3-rename registry path (`benchmarks.bench_X` / `bench_X`, written before the
    # package nesting) to its home in the `leaf_eval_bound` package, so a registry row that predates
    # the rename still resolves (ADR-0002: resolve by the module's real location, not fail on a
    # stale-but-recoverable path; re-run register_all to rewrite the rows canonical).
    mp = module_path
    if not mp.startswith("leaf_eval_bound."):
        if mp.startswith("benchmarks."):
            mp = "leaf_eval_bound." + mp
        elif mp.startswith("bench_"):
            mp = "leaf_eval_bound.benchmarks." + mp
    return importlib.import_module(mp)


def _seed_from_module(module_path: str, name: str) -> tuple[float, float, str, bool]:
    """Pull `(seed_mean, seed_sigma, units, constant)` from a quantity's bench module `get_seed()`. The
    bench module is the ONE owner of its seed (the v1 fallback), so the manifest never hard-codes a seed.
    A module that does not expose `get_seed()` is a loud contract violation (ADR-0002: every registered
    quantity's bench MUST expose its seed).

    `constant` is the `Grounded.constant` flag (default False if the seed does not carry it — a tuple
    seed, or an older `Grounded`): a TRUE CONSTANT (n_gen — a layout fact) seeds a `family=DEGENERATE`
    Estimate (~0 bound contribution, §3), NOT a declared-spread `NORMAL` prior. Threading it here makes
    the SEED path agree with the bench's `measure()` path (P1 single-home: the DEGENERATE classification
    has ONE source, `Grounded.constant`, so n_gen cannot drop out of the bound on one path and floor it
    on the other)."""
    mod = _import_bench_module(module_path)
    if not hasattr(mod, "get_seed"):
        raise AttributeError(
            f"manifest: bench module for {name!r} ({module_path!r}) exposes no get_seed() — "
            f"every quantity's bench module must expose its v1 seed (the DISTRUST fallback).")
    seed = mod.get_seed()
    # get_seed() returns a Grounded-like with .mean/.sigma/.unit/.constant, or a (mean, sigma, unit) tuple.
    if hasattr(seed, "mean"):
        return (float(seed.mean), float(getattr(seed, "sigma", 0.0)),
                str(getattr(seed, "unit", "")), bool(getattr(seed, "constant", False)))
    mean, sigma, *rest = seed
    return float(mean), float(sigma), (str(rest[0]) if rest else ""), False


# --------------------------------------------------------------------------- #
# §6 Phase 1 — the Estimate seam: reconstruct an Estimate for each resolution path, and
# PROJECT it back to the (mean, sigma, n) the 4-tuple carries. These are PURE functions (no
# DB, no postgres) — the manifest's TRUST/SEED paths call them to attach `Quantity.estimate`
# and to derive the 4-tuple from it, so the two cannot disagree (the confirmed fixed point).
# --------------------------------------------------------------------------- #


def _quantity_from_estimate(
    name: str, est: "_est.Estimate", *, trusted: bool, units: str, source: str
) -> Quantity:
    """Build a `Quantity` whose `(mean, sigma, n)` ARE the projection of `est` and whose `estimate`
    is `est` — so the 4-tuple a caller reads and the Estimate carried alongside cannot disagree
    (Phase 1's whole point). The single place a resolved Quantity is assembled from an Estimate."""
    mean, sigma, n = reconstruct._project_estimate(est)
    return Quantity(
        name=name, mean=mean, sigma=sigma, n=n, trusted=trusted,
        units=units, source=source, estimate=est)


# --------------------------------------------------------------------------- #
# Definition lookup (the postgres-driven registry; cached per-session)
# --------------------------------------------------------------------------- #
_DEF_CACHE: Optional[dict[str, dict[str, Any]]] = None


def discover(force: bool = False) -> dict[str, dict[str, Any]]:
    """The registry as a `{name: definition_row}` map, enumerated from the `benchmark_definition` TABLE
    (so a newly-INSERTed quantity appears with no manifest edit). Cached per-session; `force=True`
    re-reads (after a registration). Seed-only when postgres is down — returns the empty map (a model
    then resolves quantities by their seed module directly via the explicit `register`/`value` path it
    already holds; `discover()` is the convenience enumerator, not the only resolution route)."""
    global _DEF_CACHE
    if _DEF_CACHE is not None and not force:
        return _DEF_CACHE
    if not postgres_available():
        _DEF_CACHE = {}
        return _DEF_CACHE
    try:
        from leaf_eval_bound.store import bench_store
        defs = bench_store.list_definitions()
        _DEF_CACHE = {d["name"]: d for d in defs}
    except Exception as exc:  # a SQL error on an OPEN connection is a real fault — surface it
        raise RuntimeError(f"manifest.discover: registry SELECT failed: {exc!r}") from exc
    return _DEF_CACHE


def _definition(name: str) -> Optional[dict[str, Any]]:
    return discover().get(name)


# --------------------------------------------------------------------------- #
# The model-facing API
# --------------------------------------------------------------------------- #
def value(
    name: str,
    *,
    trust: bool = True,
    module_path: Optional[str] = None,
    rerun: bool = False,
) -> tuple[float, float, int, bool]:
    """Resolve `name` to `(mean, sigma, n, trusted)` — the contract every model calls.

    trust=True  : the latest measured (mean, sigma, n) from postgres; trusted=True iff a measurement
                  exists. If none exists yet (or postgres is down) it falls back to the seed and returns
                  trusted=False (a seed is NEVER reported as trusted).
    trust=False : the v1 seed (mean, sigma, n=0), trusted=False — the grounded starting point.
    rerun=True  : (TRUST only) import the bench module and run() it to populate a fresh measurement,
                  THEN read the new aggregate. TIMING-SENSITIVE — never pass during the parallel fan-out.

    `module_path` lets a caller resolve a quantity WITHOUT the postgres registry (pass the bench module
    directly) — the seed-only / no-DB route. Normally omitted: the manifest looks the module_path up in
    the definition row."""
    return quantity(name, trust=trust, module_path=module_path, rerun=rerun).as_tuple()


def estimate(
    name: str,
    *,
    trust: bool = True,
    module_path: Optional[str] = None,
    rerun: bool = False,
) -> "_est.Estimate":
    """Resolve `name` to its harmonized `estimate.Estimate` (§6 Phase 1) — the Estimate carried
    ALONGSIDE the 4-tuple, exposed directly for an `Estimate`-capable caller (Phase 2's driver). Same
    resolution as `value()`/`quantity()` (TRUST stored-estimate -> legacy reconstruction -> SEED); the
    4-tuple is a projection of exactly this object, so `value(name)` and `estimate(name)` cannot disagree
    on (mean, sigma, n). Never None on a resolved quantity (a None would be a Phase-1 bug — the accessor
    asserts it loudly per ADR-0002)."""
    q = quantity(name, trust=trust, module_path=module_path, rerun=rerun)
    if q.estimate is None:
        raise RuntimeError(
            f"manifest.estimate({name!r}): resolved Quantity carries no Estimate — every Phase-1 "
            f"resolution path must set it (ADR-0002: a missing estimate is a loud bug, not a default).")
    return q.estimate


def quantity(
    name: str,
    *,
    trust: bool = True,
    module_path: Optional[str] = None,
    rerun: bool = False,
) -> Quantity:
    """`value()` but returning the full `Quantity` (with units + source provenance + the §6 Phase-1
    `estimate`). The richer form a report wants; `value()` is the 4-tuple convenience over it.

    §6 Phase 1: every returned `Quantity` carries an `estimate.Estimate`, and its (mean, sigma, n)
    are the PROJECTION of that estimate (`_quantity_from_estimate` -> `reconstruct._project_estimate`), so the
    4-tuple cannot drift from the Estimate. The resolution order is unchanged — only the object
    carried alongside is new:
      * TRUST + a STORED estimate (the bench's COMPUTED Estimate)  -> latest_estimate(name).
      * TRUST + a legacy instance (samples, no stored estimate)    -> reconstruct from latest_aggregate.
      * SEED (DISTRUST / no-samples / pg-down)                      -> a Fixed-law Estimate from get_seed()."""
    mp = module_path or (_definition(name) or {}).get("module_path")

    def _seed_quantity(src: str) -> Quantity:
        if not mp:
            raise KeyError(
                f"manifest: quantity {name!r} is not registered and no module_path was given — "
                f"register it (a benchmark_definition row) or pass module_path=… (ADR-0002: an unknown "
                f"quantity is a loud error, not a silent default).")
        mean, sigma, units, constant = _seed_from_module(mp, name)
        # SEED path -> a Fixed-law Estimate; its projection is (mean, sigma, n=0) = today's seed 4-tuple.
        # `constant` (Grounded.constant) selects the §3 PIN flavor: a true constant -> DEGENERATE (~0 bound
        # contribution), a declared spread -> NORMAL prior (contributes a_i) — single-homed, so the SEED
        # path agrees with the bench's measure() path (n_gen drops out of the bound on BOTH).
        est = reconstruct._estimate_from_seed(name, mean, sigma, units, constant=constant)
        return _quantity_from_estimate(name, est, trusted=False, units=units, source=src)

    if not trust:
        return _seed_quantity("seed")

    # TRUST path. Optionally re-run the live bench first (timing-sensitive — explicit only).
    if rerun:
        measure(name, module_path=mp)

    if not postgres_available():
        return _seed_quantity("seed(pg-down)")

    # Prefer the bench's COMPUTED Estimate (carries SE/Cov the sample aggregate cannot recover); fall
    # back to reconstructing a Poolwise Estimate from the legacy aggregate when no instance carries one.
    try:
        from leaf_eval_bound.store import bench_store
        stored = bench_store.latest_estimate(name)
        if stored is not None:
            units = (_definition(name) or {}).get("units", "") or ""
            return _quantity_from_estimate(
                name, stored, trusted=True, units=units, source="postgres(estimate)")
        agg = bench_store.latest_aggregate(name)
    except Exception as exc:  # SQL error on an open connection is a real fault (ADR-0002)
        raise RuntimeError(f"manifest.value({name!r}): estimate/aggregate read failed: {exc!r}") from exc

    if agg is None:  # registered but no samples yet -> seed, untrusted (seeds stay untrusted)
        q = _seed_quantity("seed(no-samples)")
        return q
    mean, sigma, n = agg
    units = (_definition(name) or {}).get("units", "") or ""
    kind = str((_definition(name) or {}).get("estimator", "") or "")
    est = reconstruct._estimate_from_aggregate(name, mean, sigma, n, kind)
    # ADR-0002 fixed-point guard: the legacy reconstruction and the 4-tuple projection MUST be exact
    # inverses on the mean case — if the projected (mean, sigma, n) ever drifts from the aggregate the
    # seam is broken, and that is a loud fault, not a silent number swap.
    pm, ps, pn = reconstruct._project_estimate(est)
    if not (math.isclose(pm, mean, rel_tol=1e-12, abs_tol=1e-12)
            and math.isclose(ps, sigma, rel_tol=1e-12, abs_tol=1e-12)
            and pn == n):
        raise RuntimeError(
            f"manifest.quantity({name!r}): legacy Estimate reconstruction is not a byte-for-byte "
            f"projection of the aggregate — got projection ({pm}, {ps}, {pn}) vs aggregate "
            f"({mean}, {sigma}, {n}) (ADR-0002: the 4-tuple fixed point must hold).")
    return _quantity_from_estimate(name, est, trusted=True, units=units, source="postgres")


def measure(name: str, *, module_path: Optional[str] = None, **run_kwargs: Any) -> Any:
    """Run the live benchmark for `name` ON DEMAND (import its bench module, call `run(**run_kwargs)`),
    populating a fresh postgres instance+samples. Returns whatever `run()` returns (typically the new
    instance id / aggregate). TIMING-SENSITIVE — the parallel workflow corrupts timing, so this is an
    explicit operator action, never called by a model during the fan-out. A bench module without a
    `run()` is a loud contract violation (ADR-0002)."""
    mp = module_path or (_definition(name) or {}).get("module_path")
    if not mp:
        raise KeyError(f"manifest.measure({name!r}): not registered and no module_path given.")
    mod = _import_bench_module(mp)
    if not hasattr(mod, "run"):
        raise AttributeError(
            f"manifest.measure: bench module for {name!r} ({mp!r}) exposes no run() — every quantity's "
            f"bench module must expose a live run() that logs samples to postgres.")
    return mod.run(**run_kwargs)


def register(
    name: str,
    *,
    quantity: str,
    units: str,
    description: str,
    module_path: str,
) -> Any:
    """Convenience wrapper over `bench_store.register_definition` — insert/refresh a quantity's
    definition row. A design agent CAN call this, but the canonical registration is the bench module's
    own `register_self()` (so the definition and the bench live together). Returns the definition id.
    Loud if postgres is down (registration REQUIRES the DB — it is the registry write)."""
    from leaf_eval_bound.store import bench_store
    return bench_store.register_definition(
        name, quantity=quantity, units=units, description=description, module_path=module_path)


def report(names: Optional[list[str]] = None, *, trust: bool = True) -> str:
    """A human-readable table of resolved quantities (name, mean, sigma, n, trusted, source) — the
    one-glance state of the registry. `names` defaults to every registered quantity (`discover()`); pass
    a subset to report just a model's inputs. Trust=False reports the seeds. Never raises on an
    individual quantity — a failing resolve is shown as an error row (so one broken bench does not blank
    the whole report)."""
    if names is None:
        names = sorted(discover().keys())
    lines = [f"  {'quantity':<22}{'mean':>12}{'sigma':>10}{'n':>6}  {'trusted':<8}{'source'}",
             "  " + "-" * 74]
    for nm in names:
        try:
            q = quantity(nm, trust=trust)
            lines.append(f"  {nm:<22}{q.mean:>12.4g}{q.sigma:>10.4g}{q.n:>6d}  "
                         f"{str(q.trusted):<8}{q.source} [{q.units}]")
        except Exception as exc:  # noqa: BLE001 — report-only; show the fault, don't abort the table
            lines.append(f"  {nm:<22}  <resolve error: {type(exc).__name__}: {exc}>")
    return "\n".join(lines)


if __name__ == "__main__":
    # A one-glance dump of the registry (TRUST view) + the seed-only (DISTRUST) view, so an operator can
    # see what is registered, what is measured vs seeded, and whether postgres is up.
    print("[manifest] postgres available:", postgres_available())
    print("[manifest] registered quantities (postgres-driven registry):")
    print(report(trust=True))
    print("\n[manifest] DISTRUST view (v1 seeds, all trusted=False by construction):")
    print(report(trust=False))
