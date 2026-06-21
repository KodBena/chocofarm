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

import psycopg  # noqa: E402  — psycopg3 ONLY (never psycopg2)
from psycopg.types.json import Jsonb  # noqa: E402

from chocofarm.config import lab_pg_connect_timeout, lab_pg_params  # noqa: E402


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
    started_at    timestamptz NOT NULL DEFAULT now()
);

-- One row per READING. value is the measured number (units per the definition). sample_size is the n
-- behind an AGGREGATE reading where it makes sense (e.g. a throughput averaged over N ops); NULL for a
-- single reading. seq orders readings within an instance. The FK is to the instance; the instance FKs
-- the definition, so a sample reaches its definition through the instance->definition chain.
CREATE TABLE IF NOT EXISTS benchmark_sample (
    id          bigserial PRIMARY KEY,
    instance_id uuid NOT NULL REFERENCES benchmark_instance(id),
    seq         integer,
    value       double precision NOT NULL,
    sample_size integer,
    captured_at timestamptz NOT NULL DEFAULT now()
);

-- The manifest's hot query is "latest instance + its samples, per definition name", so index the FK +
-- the definition name for that lookup (idempotent CREATE INDEX IF NOT EXISTS).
CREATE INDEX IF NOT EXISTS benchmark_instance_def_started
    ON benchmark_instance (definition_id, started_at DESC);
CREATE INDEX IF NOT EXISTS benchmark_sample_instance
    ON benchmark_sample (instance_id);
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
