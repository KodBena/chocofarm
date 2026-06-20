#!/usr/bin/env python3
"""
cpp/stage_a/control_lab/lab_store.py — the PostgreSQL EGRESS for the issue-gate control lab: the ONE
owner of the lab's DB I/O (ADR-0012 P3 one-owner — it owns ONLY the SQL/connection, never the harness's
orchestration/scoring logic, never the codec internals). It mirrors the local-JSON record shapes
(lab_harness.SCHEMA_DOC) onto descriptive SQL columns + compressed bytea blobs, and carries the
flag-gated (s,a,r) trajectory blob (trajectory_codec.py, magic CHTRAJ01) that feeds the RL loop.

WHY A SEPARATE MODULE. The harness owns the per-trial loop + scoring; the codec owns serialization; this
module owns the wire to postgres — the Transport⊥Pool⊥Task split applied to "the lab's persistence is not
the lab's logic" (P3). The harness imports `connect`/`ensure_schema`/`insert_*` and calls them at its
BETWEEN-TRIAL flush seam (off the per-forward critical path); nothing here runs on a forward.

THE SCHEMA (idempotent CREATE TABLE IF NOT EXISTS; descriptive columns + compressed blobs):
  * lab_session  — one row per harness run (the session record's shared config; net/warm_pool as jsonb).
  * lab_trial    — one row per method-trial (the TRIAL_RECORD's structured score, flattened dps, the
                   TrialContext geometry denormalized onto the row so a trial is self-describing for a
                   query). bigserial trial_id PK; session_id FK.
  * lab_blob     — the compressed payloads keyed (trial_id, kind): kind='trajectory' (the CHTRAJ01
                   (s,a,r) stream) | kind='metrics_series' (zstd-JSON of the per-sample timeseries). The
                   bytea is ALTER ... SET STORAGE EXTERNAL so postgres does NOT re-compress an already-
                   zstd'd blob (it would waste CPU for ~no gain on incompressible bytes).

FAIL LOUD (ADR-0002). A connection failure or a SQL error propagates as a typed psycopg error — never a
silent skip, never a sentinel. The ONE deliberate idempotency check is the backfill's "skip a session_id
already present", which is a stated requirement, not a swallowed failure.

TYPED (ADR-0012 P8). The insert signatures are the contract; the harness builds the typed records and
this module maps them to columns. psycopg3 ONLY (this project never uses psycopg2).

CLI: `python lab_store.py --ensure-schema` creates/verifies the tables; `python lab_store.py --backfill`
loads the existing local ~/w/vdc/chocobo/runs/control_lab/lab_session-*.json (+ matching
lab_timeseries-*.jsonl) into the tables (idempotent — skips a session already present).

Public Domain (The Unlicense).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import zstandard as zstd

REPO = "/home/bork/w/vdc/1/chocofarm"
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import psycopg  # noqa: E402  — psycopg3 ONLY (never psycopg2)
from psycopg.types.json import Jsonb  # noqa: E402

from chocofarm.config import lab_pg_connect_timeout, lab_pg_params  # noqa: E402

# The default local artifact directory the backfill scans (the harness's --out default).
DEFAULT_RUNS_DIR = os.path.join(os.path.expanduser("~"), "w", "vdc", "chocobo", "runs", "control_lab")
# zstd level for the metrics_series blob. The timeseries JSON is repetitive (method names, flag lists,
# metric keys), so level 3 compresses it well within the between-trial budget. (The trajectory blob is
# already zstd'd by the codec at its own measured level; this module never re-compresses that.)
METRICS_ZSTD_LEVEL = 3


# ============================================================================================
# Connection — TRUST (no password), psycopg3. Fail loud on a connect error (ADR-0002).
# ============================================================================================
def connect() -> psycopg.Connection:
    """Open ONE psycopg3 connection to the control-lab PostgreSQL over TRUST (no password) using the
    facts from chocofarm.config (host/db/[user], the OS login when user is unset). autocommit=False —
    each insert path commits explicitly so a half-written session never lands. A connection failure is a
    loud psycopg.OperationalError (ADR-0002 — never a silent None/sentinel)."""
    params = lab_pg_params()
    return psycopg.connect(connect_timeout=int(lab_pg_connect_timeout()), **params)


# ============================================================================================
# Schema — idempotent CREATE TABLE IF NOT EXISTS + STORAGE EXTERNAL on the blob bytea.
# ============================================================================================
_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS lab_session (
    session_id   text PRIMARY KEY,
    started_at   timestamptz,
    git_sha      text,
    host         text,
    reward_fn    text,
    net          jsonb,
    warm_pool    jsonb,
    notes        text
);

CREATE TABLE IF NOT EXISTS lab_trial (
    trial_id      bigserial PRIMARY KEY,
    session_id    text NOT NULL REFERENCES lab_session(session_id),
    method        text NOT NULL,
    family        text,
    decimate_k    integer,
    n_threads     integer,
    d_ceiling     integer,
    k_per_thread  integer,
    s_min         integer,
    chunk_floor   boolean,
    seed          bigint,
    inflight_msgs integer,
    pool_batch    integer,
    secs          double precision,
    dps_window    double precision,
    dps_mean      double precision,
    dps_pstdev    double precision,
    dps_min       double precision,
    dps_max       double precision,
    rows_per_fwd  double precision,
    forwards      bigint,
    n_decisions   bigint,
    malfunctions  integer,
    flags         jsonb,
    ok            boolean,
    started_at    timestamptz,
    duration_s    double precision
);

CREATE INDEX IF NOT EXISTS lab_trial_session_idx ON lab_trial (session_id);

CREATE TABLE IF NOT EXISTS lab_blob (
    trial_id          bigint NOT NULL REFERENCES lab_trial(trial_id),
    kind              text NOT NULL CHECK (kind IN ('trajectory', 'metrics_series')),
    codec             text,
    n_decisions       bigint,
    raw_bytes         bigint,
    compressed_bytes  bigint,
    payload           bytea,
    PRIMARY KEY (trial_id, kind)
);
"""

# STORAGE EXTERNAL: keep the already-compressed blob out of TOAST's pglz pass (postgres would otherwise
# try to re-compress an incompressible zstd blob — wasted CPU, ~no gain). Idempotent: ALTER SET STORAGE
# is a no-op if it is already EXTERNAL. Run after the table exists.
_STORAGE_SQL = "ALTER TABLE lab_blob ALTER COLUMN payload SET STORAGE EXTERNAL;"


def ensure_schema(conn: psycopg.Connection) -> None:
    """Create the lab_session/lab_trial/lab_blob tables if absent and set STORAGE EXTERNAL on the blob
    bytea (so postgres does not re-compress the already-zstd'd payload). Idempotent — safe to call on
    every harness start. Commits. A SQL error is a loud psycopg error (ADR-0002)."""
    with conn.cursor() as cur:
        cur.execute(_SCHEMA_SQL)
        cur.execute(_STORAGE_SQL)
    conn.commit()


# ============================================================================================
# Typed insert records + the insert API. The harness builds these; this module maps to columns (P8).
# ============================================================================================
@dataclass(frozen=True)
class SessionRow:
    """One lab_session row — the session record's shared config (lab_harness session dict) + provenance
    (git_sha/host captured by the harness at flush). net/warm_pool are JSON-able mappings (-> jsonb)."""
    session_id: str
    started_at: str | None          # ISO-8601 timestamptz string (or None)
    git_sha: str | None
    host: str | None
    reward_fn: str | None
    net: Mapping[str, Any] | None
    warm_pool: Mapping[str, Any] | None
    notes: str | None = None


@dataclass(frozen=True)
class TrialRow:
    """One lab_trial row — the TRIAL_RECORD's structured score (dps flattened) + the TrialContext geometry
    denormalized onto the row (so a single trial row is self-describing for a query). n_decisions is the
    trajectory decision count when trajectory logging is on, else None (the JSON record carries no separate
    per-trial decision count)."""
    session_id: str
    method: str
    family: str | None
    decimate_k: int | None
    n_threads: int | None
    d_ceiling: int | None
    k_per_thread: int | None
    s_min: int | None
    chunk_floor: bool | None
    seed: int | None
    inflight_msgs: int | None
    pool_batch: int | None
    secs: float | None
    dps_window: float | None
    dps_mean: float | None
    dps_pstdev: float | None
    dps_min: float | None
    dps_max: float | None
    rows_per_fwd: float | None
    forwards: int | None
    n_decisions: int | None
    malfunctions: int | None
    flags: Sequence[str]
    ok: bool | None
    started_at: str | None
    duration_s: float | None


def insert_session(conn: psycopg.Connection, row: SessionRow) -> None:
    """Insert (or no-op-update) one lab_session. ON CONFLICT DO NOTHING so a re-run / backfill of an
    already-present session_id is idempotent (the stated requirement), NOT a swallowed error — a genuine
    SQL fault still raises. Commits."""
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO lab_session
                (session_id, started_at, git_sha, host, reward_fn, net, warm_pool, notes)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (session_id) DO NOTHING
            """,
            (
                row.session_id, row.started_at, row.git_sha, row.host, row.reward_fn,
                Jsonb(row.net) if row.net is not None else None,
                Jsonb(row.warm_pool) if row.warm_pool is not None else None,
                row.notes,
            ),
        )
    conn.commit()


def insert_trial(conn: psycopg.Connection, row: TrialRow) -> int:
    """Insert one lab_trial; return the assigned bigserial trial_id (the FK the blobs hang off). The
    RETURNING trial_id is the one-owner handle the harness passes back to insert_blob. Commits. A SQL
    error raises (ADR-0002)."""
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO lab_trial
                (session_id, method, family, decimate_k, n_threads, d_ceiling, k_per_thread, s_min,
                 chunk_floor, seed, inflight_msgs, pool_batch, secs, dps_window, dps_mean, dps_pstdev,
                 dps_min, dps_max, rows_per_fwd, forwards, n_decisions, malfunctions, flags, ok,
                 started_at, duration_s)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s)
            RETURNING trial_id
            """,
            (
                row.session_id, row.method, row.family, row.decimate_k, row.n_threads, row.d_ceiling,
                row.k_per_thread, row.s_min, row.chunk_floor, row.seed, row.inflight_msgs, row.pool_batch,
                row.secs, row.dps_window, row.dps_mean, row.dps_pstdev, row.dps_min, row.dps_max,
                row.rows_per_fwd, row.forwards, row.n_decisions, row.malfunctions,
                Jsonb(list(row.flags)), row.ok, row.started_at, row.duration_s,
            ),
        )
        out = cur.fetchone()
        if out is None:   # RETURNING must yield the id; its absence is a loud contract break (ADR-0002)
            raise RuntimeError("lab_store.insert_trial: INSERT ... RETURNING yielded no trial_id row")
        trial_id = int(out[0])
    conn.commit()
    return trial_id


def insert_blob(conn: psycopg.Connection, trial_id: int, kind: str, codec: str | None,
                payload: bytes, raw_bytes: int, compressed_bytes: int,
                n_decisions: int | None) -> None:
    """Insert one compressed blob for a trial. `kind` is 'trajectory' (the CHTRAJ01 (s,a,r) stream) or
    'metrics_series' (zstd-JSON of the per-sample timeseries). `payload` is the ALREADY-compressed bytes
    (this module never compresses the trajectory; it does zstd the metrics JSON before calling here).
    ON CONFLICT (trial_id, kind) DO UPDATE so a re-flush replaces rather than duplicates. Commits."""
    if kind not in ("trajectory", "metrics_series"):
        raise ValueError(f"lab_store.insert_blob: kind must be 'trajectory'|'metrics_series', got {kind!r}")
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO lab_blob
                (trial_id, kind, codec, n_decisions, raw_bytes, compressed_bytes, payload)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (trial_id, kind) DO UPDATE SET
                codec = EXCLUDED.codec, n_decisions = EXCLUDED.n_decisions,
                raw_bytes = EXCLUDED.raw_bytes, compressed_bytes = EXCLUDED.compressed_bytes,
                payload = EXCLUDED.payload
            """,
            (trial_id, kind, codec, n_decisions, raw_bytes, compressed_bytes,
             psycopg.Binary(payload)),
        )
    conn.commit()


def compress_metrics_series(samples: Sequence[Mapping[str, Any]]) -> tuple[bytes, int, int]:
    """zstd-compress the per-sample timeseries (a list of the JSONL sample dicts) into a metrics_series
    blob. Returns (payload, raw_bytes, compressed_bytes). The harness owns the samples list; this is the
    one place the metrics JSON encoding+compression lives (so the egress and the backfill agree on the
    metrics_series byte format — P7 one-authoritative-wire for the metrics blob)."""
    raw = json.dumps(list(samples), separators=(",", ":")).encode("utf-8")
    comp = zstd.ZstdCompressor(level=METRICS_ZSTD_LEVEL).compress(raw)
    return comp, len(raw), len(comp)


def session_exists(conn: psycopg.Connection, session_id: str) -> bool:
    """True iff a lab_session with this id is already present (the backfill idempotency check)."""
    with conn.cursor() as cur:
        cur.execute("SELECT 1 FROM lab_session WHERE session_id = %s", (session_id,))
        return cur.fetchone() is not None


# ============================================================================================
# Backfill — load the existing local lab_session-*.json (+ matching lab_timeseries-*.jsonl) into the DB.
# Idempotent: a session_id already present is skipped whole (ADR-0002 — a stated skip, not a swallow).
# ============================================================================================
def _session_id_from_record(rec: Mapping[str, Any], path: str) -> str:
    """The session_id is the session record's `run` (e.g. 'lab-20260620-105718') — the stable per-session
    handle. Fall back to the file stem if `run` is somehow absent (a malformed old file), so the backfill
    still lands SOMETHING keyed, never silently drops it."""
    sess = rec.get("session", {})
    run = sess.get("run")
    if isinstance(run, str) and run:
        return run
    stem = os.path.basename(path)
    return stem[len("lab_session-"):-len(".json")] if stem.startswith("lab_session-") else stem


def _timeseries_for(session_path: str) -> str | None:
    """The lab_timeseries-<stamp>.jsonl that matches a lab_session-<stamp>.json (same stamp). Returns the
    path if it exists, else None (an old session may predate the timeseries artifact)."""
    base = os.path.basename(session_path)
    if not base.startswith("lab_session-") or not base.endswith(".json"):
        return None
    stamp = base[len("lab_session-"):-len(".json")]
    ts = os.path.join(os.path.dirname(session_path), f"lab_timeseries-{stamp}.jsonl")
    return ts if os.path.exists(ts) else None


def _samples_for_method(ts_path: str | None, method_label: str) -> list[dict[str, Any]]:
    """Read the timeseries JSONL and return the samples whose `method` matches this trial's label. The
    JSONL `method` field is the decimate-aware label the harness wrote (run_trial: name or
    'decimate{k}:name'), so this groups the per-method samples for the metrics_series blob."""
    if ts_path is None:
        return []
    out: list[dict[str, Any]] = []
    with open(ts_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if obj.get("method") == method_label:
                out.append(obj)
    return out


def _trial_label(trial: Mapping[str, Any]) -> str:
    """The timeseries `method` label for a trial record: the harness writes 'decimate{k}:name' when k>1,
    else the bare name. The trial record's own `method` field already carries exactly this label."""
    return str(trial.get("method", ""))


def backfill(conn: psycopg.Connection, runs_dir: str = DEFAULT_RUNS_DIR) -> dict[str, int]:
    """Load every local lab_session-*.json under `runs_dir` (with its matching lab_timeseries-*.jsonl)
    into lab_session/lab_trial/lab_blob(metrics_series). Idempotent: a session_id already present is
    skipped whole. Returns a small count summary (sessions_loaded, sessions_skipped, trials, blobs)."""
    ensure_schema(conn)
    summary = {"sessions_loaded": 0, "sessions_skipped": 0, "trials": 0, "blobs": 0}
    for sess_path in sorted(glob.glob(os.path.join(runs_dir, "lab_session-*.json"))):
        with open(sess_path, "r") as f:
            rec = json.load(f)
        session_id = _session_id_from_record(rec, sess_path)
        if session_exists(conn, session_id):
            summary["sessions_skipped"] += 1
            continue
        sess = rec.get("session", {})
        # The session record carries no git_sha/host/net/warm_pool today (those are captured live by the
        # harness egress going forward); for a backfilled old session they are NULL, and the shared config
        # is preserved in warm_pool/net jsonb so a query still sees it. We stash the run config into `net`
        # and the pool/thread shape into `warm_pool` faithfully from what the file HAS.
        net_json = {
            "hidden": sess.get("hidden"), "m": sess.get("m"), "n_sims": sess.get("n_sims"),
        }
        warm_pool_json = {
            "pool_batch": sess.get("pool_batch"), "producer_threads": sess.get("producer_threads"),
            "inflight_msgs": sess.get("inflight_msgs"), "pool_plies": sess.get("pool_plies"),
            "decision_deadline_ms": sess.get("decision_deadline_ms"),
            "server_core": sess.get("server_core"), "producer_cores": sess.get("producer_cores"),
        }
        insert_session(conn, SessionRow(
            session_id=session_id, started_at=stamp_to_timestamp(sess.get("stamp")),
            git_sha=None, host=None, reward_fn=sess.get("reward_fn"),
            net=net_json, warm_pool=warm_pool_json,
            notes=f"backfilled from {os.path.basename(sess_path)}",
        ))
        summary["sessions_loaded"] += 1

        ts_path = _timeseries_for(sess_path)
        T = sess.get("n_threads") or sess.get("producer_threads")
        pool_batch = sess.get("pool_batch")
        k_per_thread = (-(-int(pool_batch) // int(T))) if (pool_batch and T) else None
        for trial in rec.get("trials", []):
            dps = trial.get("dps", {})
            label = _trial_label(trial)
            trow = TrialRow(
                session_id=session_id,
                method=label,
                family=trial.get("family"),
                decimate_k=trial.get("decimate_k"),
                n_threads=T,
                d_ceiling=sess.get("inflight_msgs"),
                k_per_thread=k_per_thread,
                s_min=1,
                chunk_floor=False,
                seed=None,
                inflight_msgs=sess.get("inflight_msgs"),
                pool_batch=pool_batch,
                secs=sess.get("secs"),
                dps_window=trial.get("dps_window"),
                dps_mean=dps.get("mean"),
                dps_pstdev=dps.get("pstdev"),
                dps_min=dps.get("min"),
                dps_max=dps.get("max"),
                rows_per_fwd=trial.get("mean_forward_rows"),
                forwards=trial.get("forwards"),
                n_decisions=None,   # the JSON record carries no trajectory; only forward_rows
                malfunctions=trial.get("malfunctions"),
                flags=trial.get("flags", []),
                ok=trial.get("ok"),
                started_at=stamp_to_timestamp(sess.get("stamp")),
                duration_s=trial.get("window_s"),
            )
            trial_id = insert_trial(conn, trow)
            summary["trials"] += 1
            samples = _samples_for_method(ts_path, label)
            if samples:
                payload, raw_n, comp_n = compress_metrics_series(samples)
                insert_blob(conn, trial_id, kind="metrics_series", codec="zstd-json",
                            payload=payload, raw_bytes=raw_n, compressed_bytes=comp_n,
                            n_decisions=None)
                summary["blobs"] += 1
    return summary


def stamp_to_timestamp(stamp: str | None) -> str | None:
    """Convert a session stamp 'YYYYMMDD-HHMMSS' (the harness's time.strftime stamp) to an ISO-8601
    timestamp string postgres parses as timestamptz. ONE home for the conversion — both the live harness
    egress (started_at) and the backfill call this (P1). Returns None for a missing/malformed stamp (the
    column is nullable)."""
    if not stamp or len(stamp) != 15 or stamp[8] != "-":
        return None
    d, t = stamp[:8], stamp[9:]
    return f"{d[:4]}-{d[4:6]}-{d[6:8]}T{t[:2]}:{t[2:4]}:{t[4:6]}"


# ============================================================================================
# CLI — ensure-schema / backfill.
# ============================================================================================
def main() -> int:
    ap = argparse.ArgumentParser(description="control-lab postgres egress: schema + backfill")
    ap.add_argument("--ensure-schema", action="store_true",
                    help="create the lab_session/lab_trial/lab_blob tables (idempotent) and exit")
    ap.add_argument("--backfill", action="store_true",
                    help="load existing local lab_session-*.json (+ timeseries) into the DB (idempotent)")
    ap.add_argument("--runs-dir", default=DEFAULT_RUNS_DIR,
                    help=f"the local artifact dir to backfill from (default {DEFAULT_RUNS_DIR})")
    a = ap.parse_args()
    if not (a.ensure_schema or a.backfill):
        ap.error("nothing to do: pass --ensure-schema and/or --backfill")
    conn = connect()
    try:
        if a.ensure_schema:
            ensure_schema(conn)
            print("[lab_store] schema ensured (lab_session/lab_trial/lab_blob; payload STORAGE EXTERNAL)")
        if a.backfill:
            summary = backfill(conn, a.runs_dir)
            print(f"[lab_store] backfill from {a.runs_dir}: {summary}")
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
