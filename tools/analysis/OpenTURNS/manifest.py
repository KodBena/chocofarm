"""
tools/analysis/OpenTURNS/manifest.py
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
required (models import this for the numbers; the openturns path is the driver's).

Public Domain (The Unlicense).
"""
from __future__ import annotations

import importlib
import importlib.util
import os
import sys
from dataclasses import dataclass
from typing import Any, Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# bench_store is imported LAZILY (inside the functions that touch the DB) so importing the
# manifest never opens a connection — the import-clean / graceful-degradation contract.


# --------------------------------------------------------------------------- #
# Result container
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Quantity:
    """One resolved quantity. `trusted` is the load-bearing flag: True ONLY when `mean`/`sigma`/`n`
    came from a live postgres measurement; False when it is the v1 seed (DISTRUST, or a TRUST read that
    fell back because no samples exist yet). `source` records which path produced it ('postgres' |
    'seed' | 'seed(no-samples)' | 'seed(pg-down)')."""
    name: str
    mean: float
    sigma: float
    n: int
    trusted: bool
    units: str = ""
    source: str = "seed"

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
        import bench_store
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
        path = module_path if os.path.isabs(module_path) else os.path.join(_HERE, module_path)
        name = "bench_" + os.path.splitext(os.path.basename(path))[0]
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"manifest: cannot load bench module from path {path!r}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    return importlib.import_module(module_path)


def _seed_from_module(module_path: str, name: str) -> tuple[float, float, str]:
    """Pull `(seed_mean, seed_sigma, units)` from a quantity's bench module `get_seed()`. The bench
    module is the ONE owner of its seed (the v1 fallback), so the manifest never hard-codes a seed. A
    module that does not expose `get_seed()` is a loud contract violation (ADR-0002: every registered
    quantity's bench MUST expose its seed)."""
    mod = _import_bench_module(module_path)
    if not hasattr(mod, "get_seed"):
        raise AttributeError(
            f"manifest: bench module for {name!r} ({module_path!r}) exposes no get_seed() — "
            f"every quantity's bench module must expose its v1 seed (the DISTRUST fallback).")
    seed = mod.get_seed()
    # get_seed() returns a Grounded-like with .mean/.sigma/.unit, or a (mean, sigma, unit) tuple.
    if hasattr(seed, "mean"):
        return float(seed.mean), float(getattr(seed, "sigma", 0.0)), str(getattr(seed, "unit", ""))
    mean, sigma, *rest = seed
    return float(mean), float(sigma), (str(rest[0]) if rest else "")


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
        import bench_store
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


def quantity(
    name: str,
    *,
    trust: bool = True,
    module_path: Optional[str] = None,
    rerun: bool = False,
) -> Quantity:
    """`value()` but returning the full `Quantity` (with units + source provenance). The richer form a
    report wants; `value()` is the 4-tuple convenience over it."""
    mp = module_path or (_definition(name) or {}).get("module_path")

    def _seed_quantity(src: str) -> Quantity:
        if not mp:
            raise KeyError(
                f"manifest: quantity {name!r} is not registered and no module_path was given — "
                f"register it (a benchmark_definition row) or pass module_path=… (ADR-0002: an unknown "
                f"quantity is a loud error, not a silent default).")
        mean, sigma, units = _seed_from_module(mp, name)
        return Quantity(name=name, mean=mean, sigma=sigma, n=0, trusted=False, units=units, source=src)

    if not trust:
        return _seed_quantity("seed")

    # TRUST path. Optionally re-run the live bench first (timing-sensitive — explicit only).
    if rerun:
        measure(name, module_path=mp)

    if not postgres_available():
        return _seed_quantity("seed(pg-down)")

    try:
        import bench_store
        agg = bench_store.latest_aggregate(name)
    except Exception as exc:  # SQL error on an open connection is a real fault (ADR-0002)
        raise RuntimeError(f"manifest.value({name!r}): aggregate read failed: {exc!r}") from exc

    if agg is None:  # registered but no samples yet -> seed, untrusted (seeds stay untrusted)
        q = _seed_quantity("seed(no-samples)")
        return q
    mean, sigma, n = agg
    units = (_definition(name) or {}).get("units", "") or ""
    return Quantity(name=name, mean=mean, sigma=sigma, n=n, trusted=True, units=units, source="postgres")


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
    import bench_store
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
