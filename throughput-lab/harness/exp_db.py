#!/usr/bin/env python3
"""
throughput-lab/harness/exp_db.py — the PostgreSQL EGRESS for the throughput lab's benchmark-results store:
the ONE owner (ADR-0012 P3 one-owner) of the lab's experiment-DB I/O. It persists every attributed reading
(leaf-rows/s, DPS, server util, …) with full provenance (the code_stamp commit/tree + UTC time + host), the
HP configuration that produced it, and the EXACT command string — so a number that today lives only in a
conversation + a markdown journal becomes a structured, queryable record keyed to a git hash.

WHY A SEPARATE MODULE (the Transport⊥Pool⊥Task split applied to "the lab's persistence is not the lab's
logic", P3). A harness imports `connect`/`ensure_schema`/`record_reading` and calls them at its post-run
flush seam (off any per-forward critical path); nothing here runs on a forward. A thin CLI (`--record`,
reading a JSON reading on stdin) lets a SHELL harness (episodic_dps.sh) pipe a result in with the same
contract — one home for the wire, two front doors.

THE SCHEMA (idempotent CREATE TABLE IF NOT EXISTS; load-bearing axes first-class, the long tail jsonb):
  * tlab_config  — one row per DISTINCT HP configuration. The load-bearing throughput axes are first-class,
                   queryable columns; `hp_extra` jsonb carries the long tail. Deduplicated on a content
                   hash (config_key) so N replicates of one config share ONE config row and median/IQR
                   queries are a natural GROUP BY. driver is a CHECK enum (round-sync|greedy) — an illegal
                   driver is unrepresentable at the column (ADR-0000).
  * tlab_reading — one row per SAMPLE/replicate (right-skewed benchmark timings MUST be aggregated, never
                   recorded single; replicate_idx + the FK make median/min/max a GROUP BY). Carries the
                   provenance stamp (commit/tree/recorded_at/host), the metrics + the raw counts they derive
                   from, the EXACT command string, the emitting tool, and a free-text tag/notes.

HP NAME ALIGNMENT (ADR-0012 P1, one home). The first-class HP columns mirror the canonical names declared in
the SSOT (throughput-lab/hp/spec.py + hp/compile.py): driver, pool_batch, msg_rows (concept of `msg-rows`),
fibers (the static-lab `--fibers` = the SSOT `fibers_per_thread` concept), inflight_msgs (=
max_inflight_msgs), n_sims, m, server impl, warmup ladder / max_batch (the server bucket policy), topology
(cores), producer binary. This module does NOT re-author those domains — it stamps the VALUES a run used; the
SSOT owns the names/domains. We align the column vocabulary rather than invent a second naming.

PROVENANCE composes with code_stamp (P1 — never re-author the stamp): record_reading calls
`code_stamp.code_stamp()` for {commit, tree} unless the caller passes an explicit stamp (the shell CLI does,
mirroring the two git invocations episodic_dps.sh already runs inline). A DIRTY tree marks a NON-reproducible
artifact (ADR-0011) and is recorded as-is, never silently 'clean'.

FAIL LOUD, BUT DON'T LOSE A LONG RUN (ADR-0002 weighed against a multi-second benchmark). A connection /
SQL fault is a LOUD psycopg error on the direct `record_reading` path — never a silent skip. But a benchmark
that ran for 14s must not have its number thrown away because the DB blipped: `record_reading_safe(...)`
wraps the insert, and on ANY failure it (1) prints a loud `[exp_db] RECORD FAILED` banner to stderr, (2)
dumps the unsaved reading as JSON to a fallback file under ~/w/vdc (NEVER /tmp — experiment output is never
discarded), and (3) returns the failure rather than raising. A harness uses `_safe` so the run's data lands
SOMEWHERE; a one-shot CLI / test uses the raising form. The choice is the caller's and is named, not buried.

TYPED (ADR-0012 P8/ADR-0000). Reading/ConfigKey are frozen dataclasses; the insert maps them to columns. The
load-bearing axes are NOT NULL; an absent driver/metric cannot become a NULL row by accident. psycopg3 ONLY
(this project never uses psycopg2).

CLI:
  python exp_db.py --ensure-schema                  create/verify the tables (idempotent), exit.
  python exp_db.py --record < reading.json          insert one reading from a JSON object on stdin
                                                     (the shell-harness front door); echoes the new id.
  python exp_db.py --aggregate [--tag T]            print the median/min/max-per-config aggregate query.

Public Domain (The Unlicense).
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import socket
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

REPO = "/home/bork/w/vdc/1/chocofarm"
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import psycopg  # noqa: E402  — psycopg3 ONLY (never psycopg2)
from psycopg.types.json import Jsonb  # noqa: E402

from chocofarm.config import lab_pg_connect_timeout, lab_pg_params  # noqa: E402

# code_stamp is the ONE home for the commit/tree provenance stamp (ADR-0012 P1) — compose, never re-author.
sys.path.insert(0, os.path.join(REPO, "throughput-lab", "harness"))
from code_stamp import code_stamp  # noqa: E402

# The throughput lab's OWN database — distinct from the control lab's `control_research` (a different lab,
# a different store). Default is overridable via CHOCO_LAB_PG_DBNAME (the same env hook config.py exposes),
# but the throughput store's natural home is `throughput_research` per the maintainer's connection facts.
DEFAULT_DBNAME = "throughput_research"

# The drivers — the SSOT EnumSet for the producer pipe shape (episodic_dps.sh $5). A CHECK on the column
# mirrors this; keep the two in step (ADR-0012 — the SSOT owns the names, the CHECK is the DB-side echo).
DRIVERS = ("round-sync", "greedy")

# Where an UNSAVED reading is dumped when a DB write fails — under ~/w/vdc (experiment output is NEVER
# discarded to /tmp). The harness can replay these later.
FALLBACK_DIR = os.path.join(os.path.expanduser("~"), "w", "vdc", "chocobo", "runs", "tlab_exp_db_unsaved")


# ============================================================================================
# Connection — TRUST (no password), psycopg3, the throughput_research DB. Fail loud (ADR-0002).
# ============================================================================================
def connect() -> psycopg.Connection:
    """Open ONE psycopg3 connection to the throughput-lab PostgreSQL over TRUST (no password). Reuses the
    host/port/user facts from chocofarm.config.lab_pg_params() (the OS-login trust map), but overrides the
    dbname to `throughput_research` (this lab's own store) UNLESS the caller set CHOCO_LAB_PG_DBNAME.
    autocommit=False — each insert path commits explicitly so a half-written reading never lands. A connect
    failure is a loud psycopg.OperationalError (ADR-0002 — never a silent None)."""
    params = dict(lab_pg_params())
    if "CHOCO_LAB_PG_DBNAME" not in os.environ:
        params["dbname"] = DEFAULT_DBNAME
    return psycopg.connect(connect_timeout=int(lab_pg_connect_timeout()), **params)


# ============================================================================================
# Schema — idempotent. Load-bearing axes first-class + NOT NULL; driver a CHECK enum (ADR-0000).
# ============================================================================================
_DRIVER_CHECK = "('" + "', '".join(DRIVERS) + "')"

_SCHEMA_SQL = f"""
CREATE TABLE IF NOT EXISTS tlab_config (
    config_id    bigserial PRIMARY KEY,
    -- config_key: a deterministic content hash over the load-bearing + extra HP axes. Two readings of the
    -- SAME configuration map to the SAME config row (so replicates GROUP BY naturally). UNIQUE so an insert
    -- of an already-present config no-ops back to the existing id (the dedupe seam).
    config_key   text NOT NULL UNIQUE,

    -- The load-bearing throughput axes — first-class & queryable. Names mirror the hp/spec.py SSOT.
    driver       text   NOT NULL CHECK (driver IN {_DRIVER_CHECK}),  -- producer pipe shape (SSOT EnumSet)
    server_impl  text   NOT NULL,             -- which server module/impl served the forwards
    producer_bin text   NOT NULL,             -- the producer binary/impl that drove the workload
    pool_batch   integer,                     -- SSOT pool_batch
    msg_rows     integer NOT NULL,            -- the coalescing floor (SSOT `--msg-rows` concept)
    fibers       integer NOT NULL,            -- K, the static-lab --fibers (SSOT fibers_per_thread concept)
    inflight_msgs integer NOT NULL,           -- SSOT max_inflight_msgs
    n_sims       integer NOT NULL,            -- search sims/decision
    m            integer,                     -- search m
    max_batch    integer NOT NULL,            -- server bucket ladder ceiling
    warmup_ladder integer[] NOT NULL,         -- the server bucket ladder (sorted snap-up set)
    topology     text   NOT NULL,             -- cores / placement string (e.g. "srv@0,gens@1,2,3")
    -- The long tail of HP axes that are not load-bearing enough to be columns (still queryable via jsonb ->).
    hp_extra     jsonb  NOT NULL DEFAULT '{{}}'::jsonb
);

CREATE TABLE IF NOT EXISTS tlab_reading (
    reading_id    bigserial PRIMARY KEY,
    config_id     bigint NOT NULL REFERENCES tlab_config(config_id),

    -- Replication: multiple samples per config. replicate_idx orders the samples of one (config, batch).
    replicate_idx integer NOT NULL DEFAULT 0,

    -- Provenance (ADR-0011): the code_stamp + when + where. commit/tree are NOT NULL — an un-pinnable
    -- reading is unattributable by construction (DIRTY is recorded as-is, never silently 'clean').
    git_commit    text        NOT NULL,
    git_tree      text        NOT NULL CHECK (git_tree IN ('clean', 'DIRTY')),
    recorded_at   timestamptz NOT NULL DEFAULT now(),
    host          text        NOT NULL,

    -- Reproducibility: the EXACT command that produced the reading + which tool emitted it + a free tag.
    command       text NOT NULL,             -- the exact invocation string
    tool          text NOT NULL,             -- the emitting harness/tool (episodic_dps.sh, coalesce_sweep…)
    tag           text,                      -- optional free-text label
    notes         text,                      -- optional free-text notes

    -- The metrics. seconds + the raw counts are recorded alongside the derived rates so a reading is
    -- self-checking (leaf_rows_s should ≈ leaves/wall_s). Nullable individually (not every tool emits
    -- every metric), but a reading with NO metric at all is rejected by record_reading (ADR-0002).
    wall_s             double precision,
    decisions          bigint,
    leaves             bigint,
    forwards           bigint,
    leaf_rows_s        double precision,      -- leaf-rows/s
    dps                double precision,      -- decisions/s
    real_rows_per_fwd  double precision,      -- real (un-padded) rows per forward
    server_util_pct    double precision,      -- server compute-busy % of wall
    forwards_s         double precision,      -- forwards/s
    lat_mean_ms        double precision,      -- in-server latency, mean
    lat_max_ms         double precision,      -- in-server latency, max
    lpd                double precision,      -- leaves per decision

    -- The long tail of metrics a tool may emit that have no column yet (queryable via jsonb ->).
    metrics_extra jsonb NOT NULL DEFAULT '{{}}'::jsonb
);

CREATE INDEX IF NOT EXISTS tlab_reading_config_idx ON tlab_reading (config_id);
CREATE INDEX IF NOT EXISTS tlab_reading_commit_idx ON tlab_reading (git_commit);
CREATE INDEX IF NOT EXISTS tlab_reading_tag_idx    ON tlab_reading (tag);
"""


def ensure_schema(conn: psycopg.Connection) -> None:
    """Create the tlab_config/tlab_reading tables + indexes if absent. Idempotent — safe on every harness
    start. Commits. A SQL error is a loud psycopg error (ADR-0002)."""
    with conn.cursor() as cur:
        cur.execute(_SCHEMA_SQL)
    conn.commit()


# ============================================================================================
# Typed records — the contract. The caller builds these; this module maps them to columns (P8/ADR-0000).
# ============================================================================================
@dataclass(frozen=True)
class ConfigKey:
    """The HP configuration that produced a reading — the load-bearing axes first-class (aligned to the
    hp/spec.py SSOT names) + a free `hp_extra` map for the long tail. Two readings of the same config build
    an EQUAL ConfigKey, which hashes to the same config_key, which dedupes to one tlab_config row."""
    driver: str                              # SSOT EnumSet — validated against DRIVERS at construction
    server_impl: str
    producer_bin: str
    msg_rows: int
    fibers: int
    inflight_msgs: int
    n_sims: int
    max_batch: int
    warmup_ladder: Sequence[int]
    topology: str
    pool_batch: Optional[int] = None
    m: Optional[int] = None
    hp_extra: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.driver not in DRIVERS:                       # ADR-0000: illegal driver unrepresentable
            raise ValueError(f"ConfigKey.driver must be one of {DRIVERS}, got {self.driver!r}")
        # store the ladder sorted+deduped so {64,256,512} and {512,64,256} are the SAME config (the SSOT
        # OrderInsensitive symmetry of a bucket set — same physical ladder, one config row).
        object.__setattr__(self, "warmup_ladder", tuple(sorted(set(int(x) for x in self.warmup_ladder))))

    def content_key(self) -> str:
        """A deterministic content hash over every axis (incl. hp_extra). The dedupe identity: equal config
        → equal key → one config row → replicates GROUP BY it."""
        payload = {
            "driver": self.driver, "server_impl": self.server_impl, "producer_bin": self.producer_bin,
            "msg_rows": self.msg_rows, "fibers": self.fibers, "inflight_msgs": self.inflight_msgs,
            "n_sims": self.n_sims, "max_batch": self.max_batch, "warmup_ladder": list(self.warmup_ladder),
            "topology": self.topology, "pool_batch": self.pool_batch, "m": self.m,
            "hp_extra": _canonical(self.hp_extra),
        }
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha1(blob.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Reading:
    """One sample/replicate's metrics + reproducibility fields. Provenance (commit/tree/host/recorded_at) is
    stamped by record_reading (composing code_stamp), NOT carried here, so a harness never re-derives the
    stamp. A reading with NO metric at all is a programming error caught at insert (ADR-0002)."""
    command: str                             # the exact invocation string
    tool: str                                # the emitting harness/tool
    replicate_idx: int = 0
    wall_s: Optional[float] = None
    decisions: Optional[int] = None
    leaves: Optional[int] = None
    forwards: Optional[int] = None
    leaf_rows_s: Optional[float] = None
    dps: Optional[float] = None
    real_rows_per_fwd: Optional[float] = None
    server_util_pct: Optional[float] = None
    forwards_s: Optional[float] = None
    lat_mean_ms: Optional[float] = None
    lat_max_ms: Optional[float] = None
    lpd: Optional[float] = None
    metrics_extra: Mapping[str, Any] = field(default_factory=dict)
    tag: Optional[str] = None
    notes: Optional[str] = None

    _METRIC_FIELDS = ("wall_s", "decisions", "leaves", "forwards", "leaf_rows_s", "dps",
                      "real_rows_per_fwd", "server_util_pct", "forwards_s", "lat_mean_ms",
                      "lat_max_ms", "lpd")

    def has_any_metric(self) -> bool:
        return any(getattr(self, f) is not None for f in self._METRIC_FIELDS) or bool(self.metrics_extra)


def _canonical(m: Mapping[str, Any]) -> Any:
    """Stable representation of a jsonb-able mapping for the content hash (sorted keys, recursively)."""
    if isinstance(m, Mapping):
        return {k: _canonical(m[k]) for k in sorted(m)}
    if isinstance(m, (list, tuple)):
        return [_canonical(x) for x in m]
    return m


# ============================================================================================
# The insert API. upsert_config dedupes; insert_reading hangs a sample off it.
# ============================================================================================
def upsert_config(conn: psycopg.Connection, key: ConfigKey) -> int:
    """Insert the config if its content_key is new, else return the existing config_id (the dedupe seam).
    ON CONFLICT (config_key) DO NOTHING then SELECT — so concurrent replicate inserts converge on one row.
    Commits. A SQL error raises (ADR-0002)."""
    ck = key.content_key()
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO tlab_config
                (config_key, driver, server_impl, producer_bin, pool_batch, msg_rows, fibers,
                 inflight_msgs, n_sims, m, max_batch, warmup_ladder, topology, hp_extra)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (config_key) DO NOTHING
            RETURNING config_id
            """,
            (ck, key.driver, key.server_impl, key.producer_bin, key.pool_batch, key.msg_rows,
             key.fibers, key.inflight_msgs, key.n_sims, key.m, key.max_batch,
             list(key.warmup_ladder), key.topology, Jsonb(dict(key.hp_extra))),
        )
        out = cur.fetchone()
        if out is None:                       # the config already existed — fetch its id
            cur.execute("SELECT config_id FROM tlab_config WHERE config_key = %s", (ck,))
            out = cur.fetchone()
            if out is None:                   # neither inserted nor found is a loud contract break
                raise RuntimeError(f"exp_db.upsert_config: config {ck} neither inserted nor found")
        config_id = int(out[0])
    conn.commit()
    return config_id


def record_reading(conn: psycopg.Connection, key: ConfigKey, reading: Reading,
                   stamp: Optional[Mapping[str, str]] = None,
                   host: Optional[str] = None) -> int:
    """The clean front door: stamp the reading with code_stamp (commit/tree) unless `stamp` is supplied,
    upsert the config, insert the reading, return its reading_id. `stamp` lets the shell CLI pass the two
    git tokens episodic_dps.sh already computes inline (one home for the GIT invocations — P1). RAISES on any
    DB failure (ADR-0002). A harness that must not lose a long run uses record_reading_safe instead.

    A reading with NO metric is rejected (recording an empty reading is a programming error, not data)."""
    if not reading.has_any_metric():
        raise ValueError("exp_db.record_reading: reading carries no metric (ADR-0002 — empty is not data)")
    st = dict(stamp) if stamp is not None else code_stamp()
    commit = st.get("commit", "unknown")
    tree = st.get("tree", "DIRTY")
    if tree not in ("clean", "DIRTY"):
        raise ValueError(f"exp_db.record_reading: tree must be 'clean'|'DIRTY', got {tree!r}")
    hostname = host if host is not None else socket.gethostname()

    config_id = upsert_config(conn, key)
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO tlab_reading
                (config_id, replicate_idx, git_commit, git_tree, host, command, tool, tag, notes,
                 wall_s, decisions, leaves, forwards, leaf_rows_s, dps, real_rows_per_fwd,
                 server_util_pct, forwards_s, lat_mean_ms, lat_max_ms, lpd, metrics_extra)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING reading_id
            """,
            (config_id, reading.replicate_idx, commit, tree, hostname, reading.command, reading.tool,
             reading.tag, reading.notes, reading.wall_s, reading.decisions, reading.leaves,
             reading.forwards, reading.leaf_rows_s, reading.dps, reading.real_rows_per_fwd,
             reading.server_util_pct, reading.forwards_s, reading.lat_mean_ms, reading.lat_max_ms,
             reading.lpd, Jsonb(dict(reading.metrics_extra))),
        )
        out = cur.fetchone()
        if out is None:
            raise RuntimeError("exp_db.record_reading: INSERT ... RETURNING yielded no reading_id")
        reading_id = int(out[0])
    conn.commit()
    return reading_id


def record_readings(conn: psycopg.Connection, key: ConfigKey, readings: Sequence[Reading],
                    stamp: Optional[Mapping[str, str]] = None,
                    host: Optional[str] = None) -> list[int]:
    """The replicate/batch form: record N samples of ONE config (the natural shape of a robust benchmark —
    multiple interleaved replicates of one cell). If a reading carries no replicate_idx (==0 default), it is
    auto-numbered by position so the samples are distinguishable. Returns the new reading_ids."""
    ids: list[int] = []
    auto = all(r.replicate_idx == 0 for r in readings) and len(readings) > 1
    for i, r in enumerate(readings):
        rr = r
        if auto:
            rr = Reading(**{**asdict(r), "replicate_idx": i})
        ids.append(record_reading(conn, key, rr, stamp=stamp, host=host))
    return ids


def record_reading_safe(conn: psycopg.Connection, key: ConfigKey, reading: Reading,
                        stamp: Optional[Mapping[str, str]] = None,
                        host: Optional[str] = None) -> Optional[int]:
    """The DON'T-LOSE-A-LONG-RUN front door (ADR-0002 weighed against a 14s benchmark): record the reading,
    but on ANY failure (1) print a LOUD banner to stderr, (2) dump the unsaved reading as JSON under ~/w/vdc
    (NEVER /tmp), and (3) return None instead of raising — so a DB blip does not throw away a number a
    benchmark spent real seconds producing. The failure is loud and the data is preserved; only the EXCEPTION
    is caught. A harness uses this; a one-shot CLI/test uses record_reading (the raising form)."""
    try:
        return record_reading(conn, key, reading, stamp=stamp, host=host)
    except Exception as exc:                                  # noqa: BLE001 — loud-and-preserve, by design
        try:
            conn.rollback()
        except Exception:                                    # noqa: BLE001
            pass
        path = _dump_unsaved(key, reading, stamp, exc)
        print(f"[exp_db] RECORD FAILED ({type(exc).__name__}: {exc}); reading dumped to {path}",
              file=sys.stderr, flush=True)
        return None


def _dump_unsaved(key: ConfigKey, reading: Reading, stamp: Optional[Mapping[str, str]],
                  exc: Exception) -> str:
    """Write the unsaved reading (+ its config + the error) as JSON under ~/w/vdc so nothing is lost. The
    file is replayable. Returns the path. This itself must not raise (it is the last line of defense)."""
    os.makedirs(FALLBACK_DIR, exist_ok=True)
    ts = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%S_%f")
    path = os.path.join(FALLBACK_DIR, f"unsaved-{ts}.json")
    blob = {
        "error": f"{type(exc).__name__}: {exc}",
        "stamp": dict(stamp) if stamp is not None else None,
        "config": {k: (list(v) if k == "warmup_ladder" else v)
                   for k, v in asdict(key).items()},
        "reading": asdict(reading),
    }
    Path(path).write_text(json.dumps(blob, indent=2, default=str))
    return path


# ============================================================================================
# The aggregate query — median/min/max per config (the robust-benchmark read; never single-reading).
# ============================================================================================
_AGGREGATE_SQL = """
SELECT
    c.config_id,
    c.driver,
    c.warmup_ladder,
    c.max_batch,
    c.msg_rows,
    c.fibers,
    c.n_sims,
    count(*)                                                                AS n_samples,
    percentile_cont(0.5) WITHIN GROUP (ORDER BY r.leaf_rows_s)              AS leaf_rows_s_median,
    min(r.leaf_rows_s)                                                      AS leaf_rows_s_min,
    max(r.leaf_rows_s)                                                      AS leaf_rows_s_max,
    percentile_cont(0.5) WITHIN GROUP (ORDER BY r.dps)                      AS dps_median,
    percentile_cont(0.5) WITHIN GROUP (ORDER BY r.server_util_pct)         AS util_median,
    percentile_cont(0.5) WITHIN GROUP (ORDER BY r.real_rows_per_fwd)       AS rows_per_fwd_median
FROM tlab_reading r
JOIN tlab_config c ON c.config_id = r.config_id
{where}
GROUP BY c.config_id, c.driver, c.warmup_ladder, c.max_batch, c.msg_rows, c.fibers, c.n_sims
ORDER BY leaf_rows_s_median DESC NULLS LAST;
"""


def aggregate(conn: psycopg.Connection, tag: Optional[str] = None) -> list[tuple]:
    """Return the median/min/max-per-config aggregate rows (the robust-benchmark read). Optionally filtered
    to one `tag`. This is the canonical 'what's the throughput of each config' query the journal narrated by
    hand — now a GROUP BY over replicate rows."""
    where = ""
    params: tuple = ()
    if tag is not None:
        where = "WHERE r.tag = %s"
        params = (tag,)
    sql = _AGGREGATE_SQL.format(where=where)
    with conn.cursor() as cur:
        cur.execute(sql, params)
        return cur.fetchall()


# ============================================================================================
# CLI — ensure-schema / record (from stdin JSON) / aggregate. The shell-harness front door.
# ============================================================================================
def _reading_from_json(obj: Mapping[str, Any]) -> tuple[ConfigKey, Reading, Optional[dict], Optional[str]]:
    """Parse a JSON reading object (the shell CLI's stdin contract) into (ConfigKey, Reading, stamp, host).
    The object has a `config` sub-object (the HP axes), a `reading` sub-object (metrics + command/tool/tag),
    and optional top-level `stamp` ({commit,tree}) and `host`. A missing required field is a loud KeyError
    via the dataclass constructors (ADR-0002 — the CLI does not invent defaults for load-bearing axes)."""
    cfg = dict(obj["config"])
    rd = dict(obj["reading"])
    key = ConfigKey(**cfg)
    reading = Reading(**rd)
    stamp = obj.get("stamp")
    host = obj.get("host")
    return key, reading, (dict(stamp) if stamp is not None else None), host


def main() -> int:
    ap = argparse.ArgumentParser(description="throughput-lab postgres egress: schema + record + aggregate")
    ap.add_argument("--ensure-schema", action="store_true",
                    help="create the tlab_config/tlab_reading tables (idempotent) and exit")
    ap.add_argument("--record", action="store_true",
                    help="read ONE JSON reading object from stdin and insert it; print the new reading_id")
    ap.add_argument("--aggregate", action="store_true",
                    help="print the median/min/max-per-config aggregate table")
    ap.add_argument("--tag", default=None, help="filter --aggregate to one tag")
    a = ap.parse_args()
    if not (a.ensure_schema or a.record or a.aggregate):
        ap.error("nothing to do: pass --ensure-schema and/or --record and/or --aggregate")
    conn = connect()
    try:
        if a.ensure_schema:
            ensure_schema(conn)
            print("[exp_db] schema ensured (tlab_config/tlab_reading)")
        if a.record:
            ensure_schema(conn)
            obj = json.load(sys.stdin)
            key, reading, stamp, host = _reading_from_json(obj)
            rid = record_reading(conn, key, reading, stamp=stamp, host=host)
            print(rid)
        if a.aggregate:
            rows = aggregate(conn, tag=a.tag)
            hdr = ("config_id", "driver", "ladder", "max_batch", "msg_rows", "fibers", "n_sims",
                   "n", "leaf_rows_s med", "min", "max", "dps med", "util med", "rows/fwd med")
            print("\t".join(hdr))
            for row in rows:
                print("\t".join("" if v is None else
                                 (f"{v:.1f}" if isinstance(v, float) else str(v)) for v in row))
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
