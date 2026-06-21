"""
tools/analysis/OpenTURNS/bench_store.py
=======================================

The PostgreSQL EGRESS for the leaf-eval TRANSPORT-DESIGN benchmark sweep: the ONE owner
of this sub-project's DB I/O (ADR-0012 P3 one-owner — it owns ONLY the SQL/connection +
schema, never a model's math, never a benchmark's measurement loop). It is the metric
store the `manifest.py` registry reads and the `benchmarks/bench_<name>.py` modules write,
sample-by-sample.

WHY A SEPARATE STORE FROM lab_store.py. `cpp/stage_a/control_lab/lab_store.py` owns the
issue-gate LAB's session/trial/blob schema (per-trial RL records). This sub-project is a
DIFFERENT measurement domain — first-principles physical QUANTITIES (t_row, iota, tau_io,
…) feeding throughput LOWER-BOUND models, swept across transport designs. Its records are
(definition, instance, sample), not (session, trial, blob). Same host DB
(`control_research` @ 192.168.122.1), same psycopg3-only / TRUST connection facts
(chocofarm.config.lab_pg_params — the single owner of "which postgres"), distinct tables.
Conflating the two would couple two unrelated schemas under one owner (a P3 violation).

THE SCHEMA (idempotent CREATE TABLE IF NOT EXISTS; the registry is POSTGRES-DRIVEN, so a
new quantity is a new `benchmark_definition` ROW + its bench module — no manifest.py edit,
no shared-file write contention across the fan-out design agents):

  * benchmark_definition — one row per measurable QUANTITY (name UNIQUE; quantity label,
    units, description, the module_path of its LIVE bench). The registry's SSOT: the
    manifest enumerates quantities by SELECTing this table, so registering a quantity is an
    INSERT here, never a code edit.
  * benchmark_instance   — one row per measurement RUN of a definition (git_sha, host,
    config jsonb, started_at). The N samples of one sole-workload bench run share an
    instance; "the latest measured value" is the latest instance's sample aggregate.
  * benchmark_sample     — one row per READING (seq, value, optional sample_size = the n
    behind an aggregate reading, e.g. a throughput over N ops; NULL for a single reading).
    FK to the instance; the instance FKs the definition — so a sample reaches its
    definition through the instance→definition chain.

FAIL LOUD (ADR-0002). A connection failure or a SQL error propagates as a typed psycopg
error — never a silent skip, never a sentinel. `ensure_schema()` is idempotent (CREATE …
IF NOT EXISTS), so a second caller is a no-op, not a failure.

TYPED (ADR-0012 P8). The insert signatures are the contract; a caller builds the typed
fields and this module maps them to columns. psycopg3 ONLY (this project never uses
psycopg2 — project memory experiment-db-postgres).

CLI: `python bench_store.py --ensure-schema` creates/verifies the three tables and prints
the row counts (the VERIFICATION the refiner runs once).

Public Domain (The Unlicense).
"""
from __future__ import annotations

import argparse
import os
import socket
import sys
import uuid
from typing import Any, Mapping, Optional

REPO = "/home/bork/w/vdc/1/chocofarm"
if REPO not in sys.path:
    sys.path.insert(0, REPO)
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:  # so the sibling `estimate` module resolves whatever the entry point
    sys.path.insert(0, _HERE)

import psycopg  # noqa: E402  — psycopg3 ONLY (never psycopg2)
from psycopg.types.json import Jsonb  # noqa: E402

from chocofarm.config import lab_pg_connect_timeout, lab_pg_params  # noqa: E402

# The Estimate contract + its (de)serialization (estimate.py — the type SSOT, ADR-0012 P8). Imported
# as a sibling module (no __init__.py in this dir; same pattern manifest.py uses for bench_store).
from estimate import Estimate, from_jsonb, to_jsonb  # noqa: E402


# ============================================================================================
# Connection — TRUST (no password), psycopg3. Fail loud on a connect error (ADR-0002).
# ============================================================================================
def connect() -> psycopg.Connection:
    """Open ONE psycopg3 connection to the control_research PostgreSQL over TRUST using the facts from
    chocofarm.config (host/db/[user] — the OS login when user is unset; the same single owner of "which
    postgres" lab_store.py uses). A connection failure is a loud psycopg.OperationalError (ADR-0002 —
    never a silent None/sentinel). autocommit=False; the insert helpers commit explicitly."""
    params = lab_pg_params()
    return psycopg.connect(connect_timeout=int(lab_pg_connect_timeout()), **params)


# ============================================================================================
# Schema — idempotent. definition (the quantity registry) / instance (a run) / sample (a reading).
# ============================================================================================
_SCHEMA_SQL = """
-- One row per measurable QUANTITY. The POSTGRES-DRIVEN registry's SSOT: the manifest enumerates
-- quantities by SELECTing this table, so a design agent registers a quantity by INSERTing a row here
-- (name PREFIXED by its transport slug to avoid UNIQUE collisions across the fan-out) + writing its
-- bench module — NEVER by editing manifest.py (no shared-file write contention).
CREATE TABLE IF NOT EXISTS benchmark_definition (
    id          uuid PRIMARY KEY,
    name        text NOT NULL UNIQUE,
    quantity    text,
    units       text,
    description text,
    module_path text,
    estimator   text,            -- the estimator KIND of this quantity (mean|median|ols_fit|pin|
                                 -- declared_spread|quantile|ratio); a definition-level declaration so
                                 -- the registry is self-describing (§6 Phase 0 / §5.4). Metadata: the
                                 -- math reads only benchmark_instance.estimate. A re-measure that
                                 -- changes the kind (a pin promoted to a histogram) is a DEFINITION
                                 -- change, surfaced here, not silent (ADR-0002).
    created_at  timestamptz NOT NULL DEFAULT now()
);

-- One row per measurement RUN of a definition: the provenance (git_sha, host) + the run config (jsonb).
-- "The latest measured value of a quantity" is the latest instance's sample aggregate (manifest reads
-- the most-recent instance per definition).
CREATE TABLE IF NOT EXISTS benchmark_instance (
    id            uuid PRIMARY KEY,
    definition_id uuid NOT NULL REFERENCES benchmark_definition(id),
    git_sha       text,
    host          text,
    config        jsonb,
    estimate      jsonb,         -- the serialized Estimate (estimate.py to_jsonb): the bench's COMPUTED
                                 -- estimate, persisted whole — {theta_hat, cov, names, shrink:{law,
                                 -- params}, support, family, cross, kind}. The SSOT of the measured
                                 -- object (§5.1; ADR-0012 P1 single-home), validated on read by
                                 -- Estimate.is_valid(). Carries SE(slope) / Cov(slope,intercept) that
                                 -- avg/stddev_samp over the sample table provably cannot recover.
                                 -- NULL on a legacy instance (the manifest falls back to latest_aggregate).
    started_at    timestamptz NOT NULL DEFAULT now()
);

-- One row per READING. value is the measured number (units per the definition). seq orders readings
-- within an instance. The FK is to the instance; the instance FKs the definition, so a sample reaches
-- its definition through the instance->definition chain.
--
-- sample_size's meaning is PINNED (§5.3, the harmonized-estimator contract): it is the number of RAW
-- READINGS behind ONE sample (the per-sample n) — and ONLY that — NULL for a single reading and NULL for
-- a pin. It is NO LONGER overloaded to carry a fit's "7 design points" or its SE; under the contract
-- those live in benchmark_instance.estimate (estimate.shrink / estimate.cov), dissolving the old "same
-- column, three meanings" defect. The variance authority is the `estimate` jsonb; this table stays the
-- raw-readings PROVENANCE for audit/re-analysis, not the variance source.
CREATE TABLE IF NOT EXISTS benchmark_sample (
    id          bigserial PRIMARY KEY,
    instance_id uuid NOT NULL REFERENCES benchmark_instance(id),
    seq         integer,
    value       double precision NOT NULL,
    sample_size integer,        -- the per-sample n: COUNT of raw readings behind this one sample; NULL
                                -- for a single reading and for a pin (§5.3, the pinned meaning).
    captured_at timestamptz NOT NULL DEFAULT now()
);

-- The manifest's hot query is "latest instance + its samples, per definition name", so index the FK +
-- the definition name for that lookup (idempotent CREATE INDEX IF NOT EXISTS).
CREATE INDEX IF NOT EXISTS benchmark_instance_def_started
    ON benchmark_instance (definition_id, started_at DESC);
CREATE INDEX IF NOT EXISTS benchmark_sample_instance
    ON benchmark_sample (instance_id);

-- §6 Phase 0 harmonized-estimator additions, applied via idempotent ALTER … ADD COLUMN IF NOT EXISTS so
-- a re-init is SAFE on the live store: CREATE TABLE IF NOT EXISTS above does NOT add a column to an
-- already-existing table, so these ALTERs are what carry the migration onto an existing DB (and are a
-- no-op once the columns exist). Additive only — no existing column is dropped or retyped.
ALTER TABLE benchmark_instance   ADD COLUMN IF NOT EXISTS estimate  jsonb;  -- the serialized Estimate (§5.1)
ALTER TABLE benchmark_definition ADD COLUMN IF NOT EXISTS estimator text;   -- the estimator kind (§5.4)
"""


def ensure_schema(conn: Optional[psycopg.Connection] = None) -> None:
    """Create/verify the three tables + their indexes (idempotent CREATE … IF NOT EXISTS). Opens its own
    connection when none is passed (the CLI / a one-off caller); reuses a caller's connection otherwise.
    A SQL error propagates loudly (ADR-0002)."""
    own = conn is None
    c = conn or connect()
    try:
        with c.cursor() as cur:
            cur.execute(_SCHEMA_SQL)
        c.commit()
    finally:
        if own:
            c.close()


# ============================================================================================
# Registration — upsert a definition by its UNIQUE name. Returns the definition id.
# ============================================================================================
def register_definition(
    name: str,
    *,
    quantity: str,
    units: str,
    description: str,
    module_path: str,
    conn: Optional[psycopg.Connection] = None,
) -> uuid.UUID:
    """Idempotently register (or refresh) the quantity `name` and return its definition id. ON CONFLICT
    (name) the descriptive fields are UPDATEd (so re-running a bench's registration keeps the row current
    without a duplicate) — the id is STABLE across re-registration (returned via RETURNING). This is the
    one write a design agent makes to put a new quantity in the registry; the manifest then auto-discovers
    it. `name` must be PREFIXED by the transport slug to avoid UNIQUE collisions across the fan-out."""
    own = conn is None
    c = conn or connect()
    try:
        new_id = uuid.uuid4()
        with c.cursor() as cur:
            cur.execute(
                """
                INSERT INTO benchmark_definition (id, name, quantity, units, description, module_path)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (name) DO UPDATE SET
                    quantity    = EXCLUDED.quantity,
                    units       = EXCLUDED.units,
                    description = EXCLUDED.description,
                    module_path = EXCLUDED.module_path
                RETURNING id
                """,
                (new_id, name, quantity, units, description, module_path),
            )
            row = cur.fetchone()
        c.commit()
        if row is None:  # RETURNING must yield the row on both INSERT and UPDATE (ADR-0002)
            raise RuntimeError(f"register_definition({name!r}) returned no id row")
        return row[0]
    finally:
        if own:
            c.close()


# ============================================================================================
# Instance + samples — open a run, log readings against it.
# ============================================================================================
def open_instance(
    definition_id: uuid.UUID,
    *,
    git_sha: Optional[str] = None,
    host: Optional[str] = None,
    config: Optional[Mapping[str, Any]] = None,
    conn: Optional[psycopg.Connection] = None,
) -> uuid.UUID:
    """Open a measurement RUN (instance) of a definition and return its id. `host` defaults to the local
    hostname; `git_sha` is the caller's responsibility (a bench can pass the repo HEAD). `config` is the
    run's parameters as jsonb (e.g. iters, pin, batch). A bad definition_id is a loud FK violation
    (ADR-0002)."""
    own = conn is None
    c = conn or connect()
    try:
        new_id = uuid.uuid4()
        with c.cursor() as cur:
            cur.execute(
                """
                INSERT INTO benchmark_instance (id, definition_id, git_sha, host, config)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (new_id, definition_id, git_sha, host or socket.gethostname(),
                 Jsonb(dict(config)) if config is not None else None),
            )
        c.commit()
        return new_id
    finally:
        if own:
            c.close()


def log_sample(
    instance_id: uuid.UUID,
    value: float,
    *,
    seq: Optional[int] = None,
    sample_size: Optional[int] = None,
    conn: Optional[psycopg.Connection] = None,
) -> None:
    """Log ONE reading against an instance. `value` is required (a NULL value is a loud NOT NULL violation,
    ADR-0002). `sample_size` is the n behind an aggregate reading (NULL for a single reading); `seq` orders
    readings within the instance."""
    own = conn is None
    c = conn or connect()
    try:
        with c.cursor() as cur:
            cur.execute(
                "INSERT INTO benchmark_sample (instance_id, seq, value, sample_size) VALUES (%s, %s, %s, %s)",
                (instance_id, seq, float(value), sample_size),
            )
        c.commit()
    finally:
        if own:
            c.close()


def log_samples(
    instance_id: uuid.UUID,
    values: list[float],
    *,
    sample_size: Optional[int] = None,
    conn: Optional[psycopg.Connection] = None,
) -> None:
    """Bulk-log readings (seq = 0..len-1) against an instance in ONE round-trip (executemany). The common
    bench path: a pool of per-op timings or per-window throughputs. `sample_size` applies to every row (the
    n behind each aggregate reading) — pass None for raw single readings."""
    own = conn is None
    c = conn or connect()
    try:
        rows = [(instance_id, i, float(v), sample_size) for i, v in enumerate(values)]
        with c.cursor() as cur:
            cur.executemany(
                "INSERT INTO benchmark_sample (instance_id, seq, value, sample_size) VALUES (%s, %s, %s, %s)",
                rows,
            )
        c.commit()
    finally:
        if own:
            c.close()


# ============================================================================================
# Read — the manifest's lookup: latest instance + its sample aggregate, per definition name.
# ============================================================================================
def latest_aggregate(
    name: str, conn: Optional[psycopg.Connection] = None
) -> Optional[tuple[float, float, int]]:
    """The LATEST measured (mean, sigma, n) for the quantity `name`, or None if it has no instance with
    samples yet. "Latest" = the most-recent instance (by started_at) that HAS samples; the aggregate is
    over that instance's samples (avg, stddev_samp, count). sigma is 0.0 when n==1 (stddev_samp is NULL
    for a single reading — the manifest treats that as a measured point with unknown spread). This is the
    one read the manifest's TRUST path makes; a quantity with no row returns None and the manifest falls
    back to the seed (untrusted)."""
    own = conn is None
    c = conn or connect()
    try:
        with c.cursor() as cur:
            cur.execute(
                """
                SELECT avg(s.value), stddev_samp(s.value), count(s.value)
                FROM benchmark_sample s
                WHERE s.instance_id = (
                    SELECT i.id
                    FROM benchmark_instance i
                    JOIN benchmark_definition d ON d.id = i.definition_id
                    WHERE d.name = %s
                      AND EXISTS (SELECT 1 FROM benchmark_sample s2 WHERE s2.instance_id = i.id)
                    ORDER BY i.started_at DESC
                    LIMIT 1
                )
                """,
                (name,),
            )
            row = cur.fetchone()
        if row is None or row[2] is None or row[2] == 0:
            return None
        mean, sigma, n = float(row[0]), (float(row[1]) if row[1] is not None else 0.0), int(row[2])
        return mean, sigma, n
    finally:
        if own:
            c.close()


# ============================================================================================
# Estimate I/O (§6 Phase 0) — the harmonized contract's store side. `set_estimate` writes one
# `Estimate`'s jsonb onto an instance (serialize via estimate.to_jsonb); `latest_estimate` reads
# the latest instance's jsonb back into a validated `Estimate` (deserialize via from_jsonb). The
# two ROUND-TRIP: latest_estimate(... set_estimate(i, est) ...) == est on a valid estimate.
# Nothing CONSUMES this yet (Phase 0 is store-only; the manifest seam is Phase 1).
# ============================================================================================
def set_estimate(
    instance_id: uuid.UUID,
    estimate: Estimate,
    *,
    conn: Optional[psycopg.Connection] = None,
) -> None:
    """Persist an `Estimate` as the instance's `estimate` jsonb (the SSOT of the measured object, §5.1).
    The estimate is serialized with `estimate.to_jsonb`. It is validated at CONSTRUCTION (ADR-0002), so a
    malformed estimate never reaches here — but a bad `instance_id` is a loud FK/UPDATE-miss check below.
    Idempotent on the instance: re-calling overwrites the column with the latest estimate."""
    own = conn is None
    c = conn or connect()
    try:
        with c.cursor() as cur:
            cur.execute(
                "UPDATE benchmark_instance SET estimate = %s WHERE id = %s",
                (Jsonb(to_jsonb(estimate)), instance_id),
            )
            if cur.rowcount != 1:  # the instance must exist (ADR-0002: a no-op UPDATE is a real fault)
                raise RuntimeError(
                    f"set_estimate: instance {instance_id} not found (UPDATE matched {cur.rowcount} rows)")
        c.commit()
    finally:
        if own:
            c.close()


def latest_estimate(
    name: str, conn: Optional[psycopg.Connection] = None
) -> Optional[Estimate]:
    """The LATEST stored `Estimate` for the quantity `name`, or None if no instance carries one yet.
    "Latest" = the most-recent instance (by started_at) whose `estimate` jsonb is NON-NULL — mirroring
    `latest_aggregate`'s "most-recent instance that has samples". The jsonb is deserialized AND re-validated
    by `estimate.from_jsonb` (the read boundary validates, ADR-0002 / P2 — a corrupt/hand-edited payload
    raises here, it does not flow on). This is the Phase-0 read the manifest's TRUST path will prefer over
    `latest_aggregate` in Phase 1; in Phase 0 nothing consumes it."""
    own = conn is None
    c = conn or connect()
    try:
        with c.cursor() as cur:
            cur.execute(
                """
                SELECT i.estimate
                FROM benchmark_instance i
                JOIN benchmark_definition d ON d.id = i.definition_id
                WHERE d.name = %s
                  AND i.estimate IS NOT NULL
                ORDER BY i.started_at DESC
                LIMIT 1
                """,
                (name,),
            )
            row = cur.fetchone()
        if row is None or row[0] is None:
            return None
        payload = row[0]  # psycopg3 decodes a jsonb column to a Python dict already
        if not isinstance(payload, Mapping):
            raise RuntimeError(
                f"latest_estimate({name!r}): estimate jsonb decoded to "
                f"{type(payload).__name__}, expected a mapping")
        return from_jsonb(payload)
    finally:
        if own:
            c.close()


def list_definitions(conn: Optional[psycopg.Connection] = None) -> list[dict[str, Any]]:
    """All registered quantities (name, quantity, units, description, module_path), name-sorted. The
    manifest enumerates the registry through this — so adding a quantity (an INSERT into
    benchmark_definition) appears here with no manifest edit."""
    own = conn is None
    c = conn or connect()
    try:
        with c.cursor() as cur:
            cur.execute(
                "SELECT id, name, quantity, units, description, module_path "
                "FROM benchmark_definition ORDER BY name"
            )
            cols = [d.name for d in cur.description]
            return [dict(zip(cols, r)) for r in cur.fetchall()]
    finally:
        if own:
            c.close()


def _table_counts(conn: psycopg.Connection) -> dict[str, int]:
    out: dict[str, int] = {}
    with conn.cursor() as cur:
        for t in ("benchmark_definition", "benchmark_instance", "benchmark_sample"):
            cur.execute(f"SELECT count(*) FROM {t}")  # noqa: S608 — fixed identifier list, not user input
            out[t] = int(cur.fetchone()[0])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="leaf-eval transport benchmark metric store (control_research)")
    ap.add_argument("--ensure-schema", action="store_true",
                    help="create/verify the three tables + indexes (idempotent) and print row counts")
    args = ap.parse_args()
    if not args.ensure_schema:
        ap.print_help()
        return
    with connect() as conn:
        ensure_schema(conn)
        counts = _table_counts(conn)
        with conn.cursor() as cur:
            cur.execute("SELECT current_database(), current_user")
            db, usr = cur.fetchone()
    print(f"[bench_store] schema ensured on {db} as {usr}")
    print(f"[bench_store] row counts: {counts}")


if __name__ == "__main__":
    main()
