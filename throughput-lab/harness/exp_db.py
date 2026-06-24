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
                   from, the EXACT command string, the emitting tool, a free-text tag, and `notes` (see below).

THE BELIEF LAYER (tlab_finding — measurement ⊥ interpretation, the discipline made structural). A reading is an
immutable MEASUREMENT; an INTERPRETATION (what a result MEANS — the reading-of-the-result that motivates the
next code change) is a different KIND of thing: mutable, often wrong, and about a SET of readings (a comparison),
not one row. Conflating the two is the failure this project has been burned by (a reading-OF the data recorded
as the data). So interpretations live in their OWN table, NEVER in tlab_reading.notes:
  * tlab_finding — one row per authored belief, APPEND-ONLY (supersede, never rewrite — ADR-0005). Carries the
                   `motivation` (what was tested + expected) and the `interpretation` (what was concluded), a
                   `status` in the CLOSED vocabulary {provisional, confirmed, retracted} (ADR-0008), the commit
                   the belief was formed against (ADR-0011), and a `supersedes` link to the finding it corrects
                   — i.e. the journey-doc Witness/Correction chain, mechanized and queryable. The CURRENT belief
                   on a scope is the finding nothing supersedes; the prior is left immutable (amend-by-append).
   Division of labor: MEASUREMENTS auto-record (the harness, at the post-run flush seam); FINDINGS are
   DELIBERATELY authored at analysis time — an interpretation is a conscious, attributable act, not a side
   effect of a run. And `tlab_reading.notes` is for MEASUREMENT-CONDITION facts (load@end, a hiccuped rep) —
   facts ABOUT a run, NEVER a reading OF it; it must not drift into a half-interpretation field.

THE PRE-REGISTRATION LAYER (tlab_prereg + tlab_prereg_conclusion — criterion-before-data, the accountability
property made structural). A finding records what a result MEANT; but "this result was DECISIVE" is, in a
finding, still prose a reader must believe. The defect this layer closes is the retro-fit: a verdict criterion
invented (or quietly bent) AFTER the data is seen, so any outcome can be narrated as decisive. The fix mirrors
ADR-0012 (the typed signature is the SSOT): the QUANTIFIED decisiveness criterion is a TYPED structure — a
partition of one decision metric's value-line into named outcome bins with numeric thresholds (Criterion/Bin)
— registered and stamped BEFORE any data exists, in an IMMUTABLE row (append-only, like a reading/finding).
  * tlab_prereg            — one row per registered experiment: the question, the SINGLE decision metric, the
                             typed `criterion` jsonb (built+validated by the Criterion dataclass — a true
                             partition, no gaps/overlaps; a 'not-decisive → escalate' band is an EXPLICIT
                             non-decisive bin, never an accidental hole), the `rationale` (the arithmetic that
                             justifies why those thresholds discriminate), the plan, and the code stamp it was
                             DECLARED against. UNIQUE slug — one experiment, one criterion; status pinned to
                             'registered' by CHECK (the verdict is never written back onto the criterion row).
  * tlab_prereg_conclusion — the SEPARATE, later verdict (mirrors finding-supersede: the criterion is read +
                             judged, NEVER rewritten). conclude_prereg loads the frozen criterion and runs
                             Criterion.evaluate(observed) — the code computes WHICH pre-declared bin the value
                             fell in + the margin to the nearest threshold + whether that bin is terminal — and
                             records outcome ∈ {decisive, ambiguous, abandoned} (ADR-0008). UNIQUE prereg_id:
                             a pre-registration concludes AT MOST ONCE (a second verdict on one immutable
                             criterion is a contradiction, not an amendment — re-opening is a NEW prereg).
   So "did the result meet the criterion?" is `criterion.evaluate(measured).decisive` — a mechanical check,
   not a human reading — and the criterion's immutability + earlier-than-the-result code stamp make
   "decisive" a falsifiable claim. The conclusion links the resolving tlab_reading and/or tlab_finding,
   closing the loop prereg → reading → finding.

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
  python exp_db.py --record-prereg < prereg.json    register one pre-registration (criterion BEFORE data);
                                                     echoes the new prereg_id.
  python exp_db.py --conclude-prereg ID --observed V   judge a measured value V against prereg ID's frozen
                                                     criterion (mechanical: prints the bin + margin + verdict).
  python exp_db.py --abandon-prereg ID --note "…"   close a prereg with no verdict (registered → abandoned).
  python exp_db.py --preregs [--open]               print the pre-registration layer (criteria + verdicts).

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
    tag           text,                      -- optional free-text label (the cohort key; findings scope to it)
    notes         text,                      -- MEASUREMENT-CONDITION facts (load@end, a hiccuped rep) — ABOUT
                                             -- the run, NEVER an interpretation OF it (those live in tlab_finding)

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

-- The BELIEF layer (measurement ⊥ interpretation). A finding is an authored INTERPRETATION, append-only:
-- a corrected belief is a NEW row that `supersedes` the one it replaces (the Witness/Correction chain), and
-- the prior row is NEVER mutated (amend-by-append, ADR-0005). Distinct table from tlab_reading precisely so
-- the conflation this project has been burned by is UNREPRESENTABLE (ADR-0000).
CREATE TABLE IF NOT EXISTS tlab_finding (
    finding_id       bigserial PRIMARY KEY,
    created_at       timestamptz NOT NULL DEFAULT now(),
    -- the code state the belief was formed against (ADR-0011): travel back to what was believed at a commit.
    git_commit       text NOT NULL,
    git_tree         text NOT NULL CHECK (git_tree IN ('clean', 'DIRTY')),
    host             text NOT NULL,
    -- scope: what the finding is ABOUT — typically a tlab_reading.tag (the cohort it interprets), or a
    -- described comparison. The common reads are "the findings on this scope" and "the current one".
    scope            text NOT NULL,
    -- the belief, the before/after halves of ONE unit (both authored at finding time):
    motivation       text,                   -- what was tested + expected (the hypothesis/prior); nullable
    interpretation   text NOT NULL,          -- what was concluded — the load-bearing field
    -- status: the belief's self-assessment at authoring, CLOSED vocabulary (ADR-0008). A single-session
    -- reading is provisional until an independent check corroborates; retracted withdraws a prior belief.
    status           text NOT NULL CHECK (status IN ('provisional', 'confirmed', 'retracted')),
    -- the Correction link: this finding supersedes a prior one. NULL = a fresh belief. The prior is left
    -- immutable; "is X current?" == "no finding supersedes X".
    supersedes       bigint REFERENCES tlab_finding(finding_id),
    -- optional: the commit/edit this interpretation MOTIVATED (closes the loop interpretation -> code change).
    motivated_change text,
    refs             jsonb NOT NULL DEFAULT '{{}}'::jsonb,   -- optional explicit config/reading refs (long tail)
    notes            text
);
CREATE INDEX IF NOT EXISTS tlab_finding_scope_idx      ON tlab_finding (scope);
CREATE INDEX IF NOT EXISTS tlab_finding_commit_idx     ON tlab_finding (git_commit);
CREATE INDEX IF NOT EXISTS tlab_finding_supersedes_idx ON tlab_finding (supersedes);

-- The PRE-REGISTRATION layer (criterion-before-data — the discipline made structural). Where tlab_finding
-- mechanizes measurement⊥interpretation, this mechanizes hypothesis⊥verdict-criterion⊥result. A finding can
-- still be retro-fitted to a narrative ("this was decisive" is, in tlab_finding, prose a human must believe).
-- A PRE-REGISTRATION fixes the QUANTIFIED decisiveness criterion — a partition of the decision metric's
-- value-line into named outcome bins with numeric thresholds — BEFORE any data exists, so "did the result
-- meet the criterion?" becomes a MECHANICAL check (which pre-declared bin did the measured value land in,
-- with what margin?) rather than rhetoric. The accountability property is structural: the criterion row is
-- immutable (append-only, like a finding/reading), and its conclusion is a SEPARATE, later row — the result
-- can never edit the criterion it is judged against.
CREATE TABLE IF NOT EXISTS tlab_prereg (
    prereg_id     bigserial PRIMARY KEY,
    -- a human-facing stable key (the experiment's slug), UNIQUE so a re-register of the SAME slug is a loud
    -- conflict, never a silent second criterion for one experiment (one experiment, one immutable criterion).
    prereg_key    text NOT NULL UNIQUE,
    created_at    timestamptz NOT NULL DEFAULT now(),
    -- provenance (ADR-0011): the code state the criterion was DECLARED against — necessarily BEFORE the
    -- result's commit. commit/tree NOT NULL; DIRTY recorded as-is, never silently 'clean'.
    git_commit    text NOT NULL,
    git_tree      text NOT NULL CHECK (git_tree IN ('clean', 'DIRTY')),
    host          text NOT NULL,

    -- the pre-registered content (all authored BEFORE the run):
    question      text NOT NULL,            -- the question / hypothesis the experiment tests
    metric        text NOT NULL,            -- the SINGLE metric the verdict rests on (e.g. 'server_util_pct')
    -- criterion: the TYPED decisiveness structure (ADR-0012) — an ordered partition of `metric`'s value-line
    -- into bins {{name, lo, hi, verdict, decisive}}. Stored as jsonb so the SHAPE is queryable, but it is built
    -- and validated by the Criterion dataclass (partition: no gaps, no overlaps), NOT free prose. The code can
    -- EVALUATE a measured value against it (which bin? what margin? terminal-decisive?) without a human reading.
    criterion     jsonb NOT NULL,
    -- the arithmetic/reasoning that justifies WHY those thresholds discriminate (the load-bearing rationale —
    -- a criterion without its justifying arithmetic is a number pulled from air; ADR-0002 rejects an empty one).
    rationale     text NOT NULL,
    method        text,                     -- the method/plan: how the experiment will be run (nullable)
    -- status lifecycle, CLOSED vocabulary (ADR-0008). 'registered' is the only status WRITTEN here; the
    -- transition to concluded/abandoned is recorded by a tlab_prereg_conclusion row (this row is never
    -- mutated to a verdict — the criterion-before-data immutability), so a CHECK pins it to 'registered'.
    status        text NOT NULL DEFAULT 'registered' CHECK (status IN ('registered')),
    refs          jsonb NOT NULL DEFAULT '{{}}'::jsonb,   -- optional config/reading/finding refs (long tail)
    notes         text
);
CREATE INDEX IF NOT EXISTS tlab_prereg_key_idx    ON tlab_prereg (prereg_key);
CREATE INDEX IF NOT EXISTS tlab_prereg_commit_idx ON tlab_prereg (git_commit);

-- The CONCLUSION row — the SEPARATE, later act that resolves a pre-registration (mirrors the finding-supersede
-- discipline: the criterion is never rewritten; its verdict is a new, append-only row). A pre-registration may
-- be concluded AT MOST ONCE (UNIQUE prereg_id) — a second verdict on the same immutable criterion would be a
-- contradiction, not an amendment; re-opening means a NEW pre-registration (a new criterion). The bin the
-- value landed in (`bin_name`) and the `margin` to the nearest boundary are computed MECHANICALLY by the code
-- (Criterion.evaluate) at conclusion time and stored so the verdict is queryable, not re-derived by a reader.
CREATE TABLE IF NOT EXISTS tlab_prereg_conclusion (
    conclusion_id bigserial PRIMARY KEY,
    prereg_id     bigint NOT NULL UNIQUE REFERENCES tlab_prereg(prereg_id),
    created_at    timestamptz NOT NULL DEFAULT now(),
    git_commit    text NOT NULL,
    git_tree      text NOT NULL CHECK (git_tree IN ('clean', 'DIRTY')),
    host          text NOT NULL,
    -- the outcome, CLOSED vocabulary (ADR-0008): 'decisive' = the value landed in a terminal pre-declared bin;
    -- 'ambiguous' = it landed in a non-decisive bin (criterion not met -> escalate, e.g. ADR-0014); 'abandoned'
    -- = the experiment was called off without a measured verdict (the registered->abandoned lifecycle leg).
    outcome       text NOT NULL CHECK (outcome IN ('decisive', 'ambiguous', 'abandoned')),
    -- the measured value of the pre-registered metric (NULL only when outcome='abandoned' — no data taken).
    observed      double precision,
    -- the bin the value fell in + the margin to the nearest boundary, both computed by Criterion.evaluate at
    -- conclusion time (NULL for 'abandoned'). bin_name is the pre-declared bin's name; margin>=0 is distance
    -- to the closest threshold (how decisively it landed). They are RECORDED, not a reader's re-derivation.
    bin_name      text,
    bin_verdict   text,                     -- the pre-declared verdict prose of that bin (denormalized for read)
    margin        double precision,
    -- the resolving evidence: the reading the verdict rests on and/or the finding that interprets it (the loop
    -- prereg -> reading -> finding closed). Both nullable; a verdict with NEITHER is allowed only for 'abandoned'.
    resolved_by_reading bigint REFERENCES tlab_reading(reading_id),
    resolved_by_finding bigint REFERENCES tlab_finding(finding_id),
    notes         text
);
CREATE INDEX IF NOT EXISTS tlab_prereg_conclusion_prereg_idx ON tlab_prereg_conclusion (prereg_id);
"""


def ensure_schema(conn: psycopg.Connection) -> None:
    """Create the tlab_config/tlab_reading/tlab_finding/tlab_prereg(+_conclusion) tables + indexes if absent.
    Idempotent — safe on every harness start. Commits. A SQL error is a loud psycopg error (ADR-0002)."""
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

    def __post_init__(self) -> None:
        """FORECLOSE the reference-140k defect CLASS (ADR-0000 question (a); finding #12). That artifact was a
        `leaf_rows_s` computed as whole-call-leaves / measure-window-wall and stored with NULL operands — a
        rate divorced from the count and window it came from, so it could be neither recomputed nor caught,
        then chased as a target for weeks. The class (ADR-0008 substitution test, calibrated to the class not
        the instance): *a derived rate without — or inconsistent with — its windowed operands*. Make it
        UNREPRESENTABLE at construction (ADR-0012 illegal-states-unrepresentable / ADR-0002 fail-loud, the
        strongest surface): a rate field may be set only WITH its count and time-span, and must equal their
        quotient. A cross-window rate (numerator and denominator from different intervals) cannot pass — the
        two operands are recorded, so the reader (and this check) can see the mismatch. The fuller foreclosure
        — a `Windowed(count, elapsed)` value the MEASUREMENT SITE produces as a unit, so count and span
        provably share one interval — is filed in BACKLOG.md (ADR-0013 Rule 4: a real type deferred, not
        buried); this guard catches the recorded-artifact class the faceplant actually took."""
        def _check(rate_name: str, rate: "float | None",
                   num_name: str, num: "float | None", den_name: str, den: "float | None") -> None:
            if rate is None:
                return
            if num is None or den is None:
                raise ValueError(
                    f"Reading.{rate_name}={rate} has no auditable provenance — it requires {num_name} and "
                    f"{den_name} (a rate stored without its windowed count+span is the un-recomputable, "
                    f"un-checkable shape that produced the reference-140k artifact; ADR-0000/ADR-0002 — record "
                    f"the operands, or omit the rate). See exp_db finding #12.")
            if float(den) == 0.0:
                raise ValueError(f"Reading.{rate_name}: {den_name} is 0 — cannot derive a rate (ADR-0002)")
            expected = float(num) / float(den)
            tol = max(0.02 * abs(expected), 1.0)   # tolerant of integer-floored harness rates (e.g. dps=dec//S)
            if abs(float(rate) - expected) > tol:
                raise ValueError(
                    f"Reading.{rate_name}={rate} is INCONSISTENT with {num_name}/{den_name}={num}/{den}="
                    f"{expected:.3f} (|Δ|>{tol:.3f}). A rate whose numerator and denominator come from DIFFERENT "
                    f"measurement windows is the reference-140k class (finding #12): record the count and the "
                    f"wall from the SAME window. (ADR-0000)")
        _check("leaf_rows_s", self.leaf_rows_s, "leaves", self.leaves, "wall_s", self.wall_s)
        _check("dps", self.dps, "decisions", self.decisions, "wall_s", self.wall_s)
        _check("forwards_s", self.forwards_s, "forwards", self.forwards, "wall_s", self.wall_s)
        _check("lpd", self.lpd, "leaves", self.leaves, "decisions", self.decisions)


def _canonical(m: Mapping[str, Any]) -> Any:
    """Stable representation of a jsonb-able mapping for the content hash (sorted keys, recursively)."""
    if isinstance(m, Mapping):
        return {k: _canonical(m[k]) for k in sorted(m)}
    if isinstance(m, (list, tuple)):
        return [_canonical(x) for x in m]
    return m


# ============================================================================================
# The TYPED decisiveness criterion (ADR-0012 — the criterion is a structure the code EVALUATES, not prose a
# human interprets). A Criterion is an ordered PARTITION of one metric's value-line into named outcome Bins;
# evaluate(value) returns which bin a measured value fell in + its margin to the nearest boundary + whether
# that bin is terminal-decisive. "Did the result meet the criterion?" == `criterion.evaluate(v).decisive`.
# ============================================================================================
NEG_INF = float("-inf")
POS_INF = float("inf")


@dataclass(frozen=True)
class Bin:
    """One outcome band of a Criterion: the half-open interval [lo, hi) on the metric's value-line, a `name`
    (the slug, e.g. 'producer-bound'), the `verdict` prose it implies, and `decisive` — whether landing here
    is a TERMINAL verdict (True) or a not-decisive / escalate band (False, e.g. the '78-90% → escalate' zone).
    Bounds are inclusive-low / exclusive-high so abutting bins tile without overlap; ±inf bound the open ends."""
    name: str
    lo: float                                # inclusive lower bound (NEG_INF for the open low end)
    hi: float                                # exclusive upper bound (POS_INF for the open high end)
    verdict: str                             # what landing in this bin MEANS (the pre-declared reading)
    decisive: bool = True                    # False = a not-decisive band (criterion not met → escalate)

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise ValueError("Bin.name must be non-empty (ADR-0002)")
        if not self.verdict or not self.verdict.strip():
            raise ValueError(f"Bin[{self.name}].verdict must be non-empty (ADR-0002 — a bin states its meaning)")
        lo, hi = float(self.lo), float(self.hi)
        if not (lo < hi):                    # an empty/inverted band is unrepresentable (ADR-0000)
            raise ValueError(f"Bin[{self.name}] needs lo < hi, got lo={lo}, hi={hi}")
        object.__setattr__(self, "lo", lo)
        object.__setattr__(self, "hi", hi)

    def contains(self, value: float) -> bool:
        return self.lo <= value < self.hi


@dataclass(frozen=True)
class BinHit:
    """The mechanical result of evaluating a measured value against a Criterion: the bin it fell in, the
    verdict that bin implies, whether that verdict is terminal-decisive, and the `margin` — the distance to
    the nearest threshold of that bin (how decisively the value landed; +inf when the bin is open on the
    relevant side). A value in no bin is impossible by construction (a Criterion PARTITIONS the whole line)."""
    value: float
    bin_name: str
    verdict: str
    decisive: bool
    margin: float


@dataclass(frozen=True)
class Criterion:
    """The pre-registered decisiveness rule over ONE metric, as a TYPED structure (ADR-0012): an ordered list
    of Bins that PARTITION the entire value-line (-inf, +inf) — no gaps, no overlaps — validated at
    construction. The partition invariant is the whole point: a 'not-decisive → escalate' outcome must be an
    EXPLICIT `decisive=False` bin, never an accidental hole between two thresholds (a hole would let a value
    fall through to no verdict — the silent ambiguity this layer exists to forbid). evaluate(v) is total."""
    metric: str
    bins: tuple[Bin, ...]

    def __post_init__(self) -> None:
        if not self.metric or not self.metric.strip():
            raise ValueError("Criterion.metric must name the single decision metric (ADR-0002)")
        bins = tuple(self.bins)
        if not bins:
            raise ValueError("Criterion needs at least one bin (ADR-0002)")
        names = [b.name for b in bins]
        if len(set(names)) != len(names):
            raise ValueError(f"Criterion bin names must be unique, got {names}")
        ordered = sorted(bins, key=lambda b: b.lo)
        # PARTITION check: the sorted bins must tile (-inf, +inf) edge-to-edge with no gap and no overlap.
        if ordered[0].lo != NEG_INF:
            raise ValueError(f"Criterion bins must cover the low end: first bin lo={ordered[0].lo}, need -inf "
                             f"(an uncovered low tail is the silent-ambiguity hole this type forbids)")
        if ordered[-1].hi != POS_INF:
            raise ValueError(f"Criterion bins must cover the high end: last bin hi={ordered[-1].hi}, need +inf")
        for a, b in zip(ordered, ordered[1:]):
            if a.hi != b.lo:                 # a gap (a.hi < b.lo) or an overlap (a.hi > b.lo) is unrepresentable
                kind = "gap" if a.hi < b.lo else "overlap"
                raise ValueError(f"Criterion bins must tile without {kind}: bin[{a.name}].hi={a.hi} must equal "
                                 f"bin[{b.name}].lo={b.lo} (ADR-0000 — the partition is the invariant)")
        object.__setattr__(self, "bins", ordered)

    def evaluate(self, value: float) -> BinHit:
        """Return which bin `value` fell in + the margin to the nearest threshold + whether the bin is
        terminal-decisive. Total by the partition invariant — every real value lands in exactly one bin."""
        v = float(value)
        for b in self.bins:
            if b.contains(v):
                # margin: distance to the nearest FINITE boundary of this bin (the open ±inf side contributes
                # +inf, so a value deep in an open-ended bin reports the distance to its one finite threshold).
                lo_d = POS_INF if b.lo == NEG_INF else v - b.lo
                hi_d = POS_INF if b.hi == POS_INF else b.hi - v
                return BinHit(value=v, bin_name=b.name, verdict=b.verdict, decisive=b.decisive,
                              margin=min(lo_d, hi_d))
        # Unreachable given the partition invariant; if it ever fires, the invariant was violated (ADR-0002).
        raise RuntimeError(f"Criterion.evaluate: value {v} fell in no bin — partition invariant broken")

    def to_json(self) -> dict[str, Any]:
        """Serialize to the jsonb stored in tlab_prereg.criterion. ±inf become the JSON tokens 'NEG_INF' /
        'POS_INF' (JSON has no infinity literal); from_json inverts this. Round-trips losslessly."""
        def _b(x: float) -> Any:
            return "NEG_INF" if x == NEG_INF else "POS_INF" if x == POS_INF else x
        return {"metric": self.metric,
                "bins": [{"name": b.name, "lo": _b(b.lo), "hi": _b(b.hi),
                          "verdict": b.verdict, "decisive": b.decisive} for b in self.bins]}

    @staticmethod
    def from_json(obj: Mapping[str, Any]) -> "Criterion":
        """Rebuild a Criterion from its stored jsonb (the Port/ACL decode — re-validates the partition, so a
        hand-edited or corrupted criterion row fails LOUD on read, not silently; ADR-0002)."""
        def _b(x: Any) -> float:
            if x == "NEG_INF":
                return NEG_INF
            if x == "POS_INF":
                return POS_INF
            return float(x)
        bins = tuple(Bin(name=d["name"], lo=_b(d["lo"]), hi=_b(d["hi"]),
                         verdict=d["verdict"], decisive=bool(d.get("decisive", True)))
                     for d in obj["bins"])
        return Criterion(metric=str(obj["metric"]), bins=bins)


def thresholds_criterion(metric: str, cuts: Sequence[tuple[float, str, str, bool]]) -> Criterion:
    """Convenience builder for the common shape: a metric partitioned by a sorted list of cut points into
    contiguous bins. `cuts` is a sequence of (upper_bound, name, verdict, decisive) — the bins are
    [-inf, cut0), [cut0, cut1), …, [cut_last, +inf); the LAST tuple's upper_bound is ignored (it is the
    open-ended top bin) but kept for shape uniformity, so pass +inf or any value. Example mirroring the real
    instance (server-util A/B): thresholds_criterion('server_util_pct', [
        (78.0, 'server-loop-ceiling', 'serve loop is the wall', True),
        (90.0, 'escalate',            'not decisive → ADR-0014',  False),
        (POS_INF, 'producer-bound',   'a faster producer would help', True)])."""
    cuts = list(cuts)
    if not cuts:
        raise ValueError("thresholds_criterion: need at least one cut (ADR-0002)")
    bins: list[Bin] = []
    lo = NEG_INF
    for i, (ub, name, verdict, decisive) in enumerate(cuts):
        hi = POS_INF if i == len(cuts) - 1 else float(ub)
        bins.append(Bin(name=name, lo=lo, hi=hi, verdict=verdict, decisive=decisive))
        lo = hi
    return Criterion(metric=metric, bins=tuple(bins))


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
# THE BELIEF LAYER — tlab_finding. Interpretations: append-only, supersede-don't-rewrite (ADR-0005/0011).
# A reading (tlab_reading) is immutable FACT; a finding is a mutable, supersedable BELIEF about readings.
# Findings are DELIBERATELY authored (an interpretation is a conscious act), never auto-emitted by a run.
# ============================================================================================
STATUSES = ("provisional", "confirmed", "retracted")   # closed vocabulary (ADR-0008); a CHECK mirrors it


def record_finding(conn: psycopg.Connection, scope: str, interpretation: str, *,
                   motivation: Optional[str] = None, status: str = "provisional",
                   supersedes: Optional[int] = None, motivated_change: Optional[str] = None,
                   refs: Optional[Mapping[str, Any]] = None, notes: Optional[str] = None,
                   stamp: Optional[Mapping[str, str]] = None, host: Optional[str] = None) -> int:
    """Append ONE authored belief (an interpretation) about a `scope` (typically a tlab_reading.tag, or a
    described comparison). APPEND-ONLY: a finding is never rewritten — a corrected belief is a NEW finding with
    `supersedes` set to the one it replaces (the Witness/Correction chain; ADR-0005 amend-by-append). `status`
    is the belief's self-assessment at authoring, in the CLOSED vocabulary {provisional, confirmed, retracted}
    (ADR-0008): a single-session reading is PROVISIONAL until an independent check corroborates it (the
    measured-vs-interpreted discipline). Stamps commit/tree via code_stamp — the code state the belief was
    formed against (ADR-0011). RAISES on a bad status / empty interpretation / DB fault (ADR-0002). Returns
    the new finding_id."""
    if status not in STATUSES:
        raise ValueError(f"record_finding: status must be one of {STATUSES}, got {status!r}")
    if not interpretation or not interpretation.strip():
        raise ValueError("record_finding: interpretation is the load-bearing field — must be non-empty (ADR-0002)")
    st = dict(stamp) if stamp is not None else code_stamp()
    commit = st.get("commit", "unknown")
    tree = st.get("tree", "DIRTY")
    if tree not in ("clean", "DIRTY"):
        raise ValueError(f"record_finding: tree must be 'clean'|'DIRTY', got {tree!r}")
    hostname = host if host is not None else socket.gethostname()
    with conn.cursor() as cur:
        if supersedes is not None:
            # the prior finding is NOT mutated (append-only) — we only point back at it. Verify it exists so a
            # dangling supersede link is a loud error, not a silent orphan (ADR-0002).
            cur.execute("SELECT 1 FROM tlab_finding WHERE finding_id = %s", (supersedes,))
            if cur.fetchone() is None:
                raise ValueError(f"record_finding: supersedes={supersedes} does not exist")
        cur.execute(
            """
            INSERT INTO tlab_finding
                (git_commit, git_tree, host, scope, motivation, interpretation, status,
                 supersedes, motivated_change, refs, notes)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING finding_id
            """,
            (commit, tree, hostname, scope, motivation, interpretation, status,
             supersedes, motivated_change, Jsonb(dict(refs or {})), notes),
        )
        out = cur.fetchone()
        if out is None:
            raise RuntimeError("record_finding: INSERT ... RETURNING yielded no finding_id")
        fid = int(out[0])
    conn.commit()
    return fid


def supersede(conn: psycopg.Connection, prior_finding_id: int, scope: str, interpretation: str, *,
              status: str = "confirmed", **kw: Any) -> int:
    """Convenience: append a NEW finding that CORRECTS `prior_finding_id` (the Witness→Correction step). The
    prior finding is left IMMUTABLE (amend-by-append); its replacement is found by following `supersedes`. The
    new belief defaults to status='confirmed' (the corrected reading), overridable. Same kwargs as
    record_finding (motivation, motivated_change, refs, notes, stamp, host)."""
    return record_finding(conn, scope, interpretation, status=status, supersedes=prior_finding_id, **kw)


def findings(conn: psycopg.Connection, scope: Optional[str] = None,
             current_only: bool = False) -> list[tuple]:
    """Read the belief layer (the time-travel query). `scope` filters; `current_only` returns only findings
    nothing supersedes — the head of each correction chain, i.e. the LIVE belief. Newest first. Returns
    (finding_id, created_at, git_commit, git_tree, scope, status, supersedes, motivation, interpretation,
    motivated_change)."""
    clauses: list[str] = []
    params: list[Any] = []
    if scope is not None:
        clauses.append("f.scope = %s")
        params.append(scope)
    if current_only:
        clauses.append("NOT EXISTS (SELECT 1 FROM tlab_finding s WHERE s.supersedes = f.finding_id)")
    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT f.finding_id, f.created_at, f.git_commit, f.git_tree, f.scope, f.status,
                   f.supersedes, f.motivation, f.interpretation, f.motivated_change
            FROM tlab_finding f {where}
            ORDER BY f.created_at DESC, f.finding_id DESC
            """,
            params,
        )
        return cur.fetchall()


# ============================================================================================
# THE PRE-REGISTRATION LAYER — tlab_prereg (+ tlab_prereg_conclusion). Criterion-before-data, mechanized.
# A pre-registration FIXES the typed decisiveness criterion BEFORE any data exists (the registration row is
# immutable, append-only like a reading/finding); the verdict is a SEPARATE, later act (conclude_prereg) that
# JUDGES a measured value against the frozen criterion mechanically (Criterion.evaluate), so "this was
# decisive" is a checkable claim — did the value land in a pre-declared terminal bin, with margin? — not
# rhetoric. The result can never edit the criterion it is judged against (that is the accountability property).
# ============================================================================================
PREREG_OUTCOMES = ("decisive", "ambiguous", "abandoned")   # closed vocabulary (ADR-0008); a CHECK mirrors it


@dataclass(frozen=True)
class PreReg:
    """A pre-registration's authored content (everything fixed BEFORE the run). `prereg_key` is the stable
    human slug (UNIQUE — one experiment, one immutable criterion). The criterion is a TYPED Criterion the code
    can evaluate, NOT prose. `rationale` is the arithmetic justifying why the thresholds discriminate (a
    criterion without it is a number from air — rejected). A PreReg with an empty question/rationale is an
    ADR-0002 programming error caught at record time."""
    prereg_key: str
    question: str
    criterion: Criterion
    rationale: str
    method: Optional[str] = None
    refs: Mapping[str, Any] = field(default_factory=dict)
    notes: Optional[str] = None

    def __post_init__(self) -> None:
        for f in ("prereg_key", "question", "rationale"):
            v = getattr(self, f)
            if not v or not str(v).strip():
                raise ValueError(f"PreReg.{f} must be non-empty (ADR-0002 — a pre-registration without its "
                                 f"{f} is not pre-registered)")
        if not isinstance(self.criterion, Criterion):
            raise ValueError("PreReg.criterion must be a typed Criterion (ADR-0012 — not free prose)")


def record_prereg(conn: psycopg.Connection, prereg: PreReg,
                  stamp: Optional[Mapping[str, str]] = None, host: Optional[str] = None) -> int:
    """Register ONE pre-registration: the question, the single decision metric, the TYPED criterion, the
    justifying arithmetic, and the plan — all stamped with the code state they were DECLARED against (ADR-0011;
    necessarily before the result). The row is IMMUTABLE (append-only) — a verdict is a separate later act
    (conclude_prereg). The slug is UNIQUE: re-registering the same key is a LOUD conflict (one experiment, one
    criterion — never a silent second criterion swapped in after seeing data). RAISES on a duplicate key /
    empty field / DB fault (ADR-0002). Returns the new prereg_id."""
    st = dict(stamp) if stamp is not None else code_stamp()
    commit = st.get("commit", "unknown")
    tree = st.get("tree", "DIRTY")
    if tree not in ("clean", "DIRTY"):
        raise ValueError(f"record_prereg: tree must be 'clean'|'DIRTY', got {tree!r}")
    hostname = host if host is not None else socket.gethostname()
    with conn.cursor() as cur:
        # a duplicate slug is a loud conflict (a second criterion for one experiment is the retro-fit this
        # layer forbids) — INSERT and let the UNIQUE constraint raise rather than ON CONFLICT silently no-op.
        cur.execute(
            """
            INSERT INTO tlab_prereg
                (prereg_key, git_commit, git_tree, host, question, metric, criterion, rationale, method,
                 refs, notes)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING prereg_id
            """,
            (prereg.prereg_key, commit, tree, hostname, prereg.question, prereg.criterion.metric,
             Jsonb(prereg.criterion.to_json()), prereg.rationale, prereg.method,
             Jsonb(dict(prereg.refs)), prereg.notes),
        )
        out = cur.fetchone()
        if out is None:
            raise RuntimeError("record_prereg: INSERT ... RETURNING yielded no prereg_id")
        pid = int(out[0])
    conn.commit()
    return pid


def record_prereg_safe(conn: psycopg.Connection, prereg: PreReg,
                       stamp: Optional[Mapping[str, str]] = None,
                       host: Optional[str] = None) -> Optional[int]:
    """The loud-but-non-fatal front door (mirrors record_reading_safe): register the pre-registration, but on
    ANY failure print a LOUD banner + dump the unsaved pre-registration as JSON under ~/w/vdc and return None
    rather than raise — so a harness that pre-registers at run start does not lose the criterion (the whole
    accountability artifact) to a DB blip. A one-shot CLI/test uses the raising record_prereg."""
    try:
        return record_prereg(conn, prereg, stamp=stamp, host=host)
    except Exception as exc:                                  # noqa: BLE001 — loud-and-preserve, by design
        try:
            conn.rollback()
        except Exception:                                    # noqa: BLE001
            pass
        os.makedirs(FALLBACK_DIR, exist_ok=True)
        ts = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%S_%f")
        path = os.path.join(FALLBACK_DIR, f"unsaved-prereg-{ts}.json")
        blob = {"error": f"{type(exc).__name__}: {exc}",
                "stamp": dict(stamp) if stamp is not None else None,
                "prereg": {"prereg_key": prereg.prereg_key, "question": prereg.question,
                           "criterion": prereg.criterion.to_json(), "rationale": prereg.rationale,
                           "method": prereg.method, "refs": dict(prereg.refs), "notes": prereg.notes}}
        Path(path).write_text(json.dumps(blob, indent=2, default=str))
        print(f"[exp_db] PREREG FAILED ({type(exc).__name__}: {exc}); dumped to {path}",
              file=sys.stderr, flush=True)
        return None


def _load_prereg_criterion(conn: psycopg.Connection, prereg_id: int) -> Criterion:
    """Read back the frozen criterion of a pre-registration (the Port/ACL decode — re-validates the partition,
    so a corrupted criterion row fails loud; ADR-0002). Raises if the prereg_id does not exist."""
    with conn.cursor() as cur:
        cur.execute("SELECT criterion FROM tlab_prereg WHERE prereg_id = %s", (prereg_id,))
        row = cur.fetchone()
    if row is None:
        raise ValueError(f"_load_prereg_criterion: prereg_id={prereg_id} does not exist (ADR-0002)")
    return Criterion.from_json(row[0])


def conclude_prereg(conn: psycopg.Connection, prereg_id: int, observed: float, *,
                    resolved_by_reading: Optional[int] = None,
                    resolved_by_finding: Optional[int] = None,
                    notes: Optional[str] = None,
                    stamp: Optional[Mapping[str, str]] = None, host: Optional[str] = None) -> tuple[int, BinHit]:
    """Conclude a pre-registration by JUDGING a measured `observed` value against its FROZEN criterion — the
    separate, later act the immutability guarantee turns on. The bin + margin are computed MECHANICALLY by
    Criterion.evaluate (never a human's reading); the outcome is 'decisive' iff the value landed in a terminal
    pre-declared bin, else 'ambiguous' (criterion not met → escalate, e.g. ADR-0014). The criterion is NEVER
    re-written — only read and judged. A pre-registration concludes AT MOST ONCE (the UNIQUE prereg_id raises a
    loud conflict on a second verdict — a second verdict on one immutable criterion is a contradiction, not an
    amendment; re-opening means a NEW pre-registration). RAISES on a missing/duplicate prereg / DB fault
    (ADR-0002). Returns (conclusion_id, the BinHit verdict)."""
    criterion = _load_prereg_criterion(conn, prereg_id)
    hit = criterion.evaluate(observed)
    outcome = "decisive" if hit.decisive else "ambiguous"
    st = dict(stamp) if stamp is not None else code_stamp()
    commit = st.get("commit", "unknown")
    tree = st.get("tree", "DIRTY")
    if tree not in ("clean", "DIRTY"):
        raise ValueError(f"conclude_prereg: tree must be 'clean'|'DIRTY', got {tree!r}")
    hostname = host if host is not None else socket.gethostname()
    with conn.cursor() as cur:
        if resolved_by_reading is not None:
            cur.execute("SELECT 1 FROM tlab_reading WHERE reading_id = %s", (resolved_by_reading,))
            if cur.fetchone() is None:
                raise ValueError(f"conclude_prereg: resolved_by_reading={resolved_by_reading} does not exist")
        if resolved_by_finding is not None:
            cur.execute("SELECT 1 FROM tlab_finding WHERE finding_id = %s", (resolved_by_finding,))
            if cur.fetchone() is None:
                raise ValueError(f"conclude_prereg: resolved_by_finding={resolved_by_finding} does not exist")
        cur.execute(
            """
            INSERT INTO tlab_prereg_conclusion
                (prereg_id, git_commit, git_tree, host, outcome, observed, bin_name, bin_verdict, margin,
                 resolved_by_reading, resolved_by_finding, notes)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING conclusion_id
            """,
            (prereg_id, commit, tree, hostname, outcome, observed, hit.bin_name, hit.verdict, hit.margin,
             resolved_by_reading, resolved_by_finding, notes),
        )
        out = cur.fetchone()
        if out is None:
            raise RuntimeError("conclude_prereg: INSERT ... RETURNING yielded no conclusion_id")
        cid = int(out[0])
    conn.commit()
    return cid, hit


def abandon_prereg(conn: psycopg.Connection, prereg_id: int, *, notes: Optional[str] = None,
                   stamp: Optional[Mapping[str, str]] = None, host: Optional[str] = None) -> int:
    """Close a pre-registration WITHOUT a measured verdict (the registered → abandoned lifecycle leg — the
    experiment was called off before data). Records an 'abandoned' conclusion (observed/bin/margin all NULL),
    AT MOST ONCE per prereg (the UNIQUE prereg_id). `notes` should say why (ADR-0002 — an unexplained
    abandonment is suspect). RAISES on a missing/already-concluded prereg (ADR-0002). Returns conclusion_id."""
    # touch the prereg so a missing id is a loud error before we write the conclusion.
    with conn.cursor() as cur:
        cur.execute("SELECT 1 FROM tlab_prereg WHERE prereg_id = %s", (prereg_id,))
        if cur.fetchone() is None:
            raise ValueError(f"abandon_prereg: prereg_id={prereg_id} does not exist (ADR-0002)")
    st = dict(stamp) if stamp is not None else code_stamp()
    commit = st.get("commit", "unknown")
    tree = st.get("tree", "DIRTY")
    if tree not in ("clean", "DIRTY"):
        raise ValueError(f"abandon_prereg: tree must be 'clean'|'DIRTY', got {tree!r}")
    hostname = host if host is not None else socket.gethostname()
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO tlab_prereg_conclusion
                (prereg_id, git_commit, git_tree, host, outcome, observed, bin_name, bin_verdict, margin, notes)
            VALUES (%s, %s, %s, %s, 'abandoned', NULL, NULL, NULL, NULL, %s)
            RETURNING conclusion_id
            """,
            (prereg_id, commit, tree, hostname, notes),
        )
        out = cur.fetchone()
        if out is None:
            raise RuntimeError("abandon_prereg: INSERT ... RETURNING yielded no conclusion_id")
        cid = int(out[0])
    conn.commit()
    return cid


def preregs(conn: psycopg.Connection, *, open_only: bool = False,
            prereg_key: Optional[str] = None) -> list[tuple]:
    """Read the pre-registration layer (the registered criteria + their conclusion, if any) — a LEFT JOIN so
    an un-concluded (still-open) pre-registration shows with NULL verdict columns. `open_only` returns only
    those NOT yet concluded (the live experiments awaiting a verdict); `prereg_key` filters to one. Newest
    first. Returns (prereg_id, created_at, git_commit, git_tree, prereg_key, metric, question, outcome,
    observed, bin_name, bin_verdict, margin)."""
    clauses: list[str] = []
    params: list[Any] = []
    if prereg_key is not None:
        clauses.append("p.prereg_key = %s")
        params.append(prereg_key)
    if open_only:
        clauses.append("c.conclusion_id IS NULL")
    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT p.prereg_id, p.created_at, p.git_commit, p.git_tree, p.prereg_key, p.metric, p.question,
                   c.outcome, c.observed, c.bin_name, c.bin_verdict, c.margin
            FROM tlab_prereg p
            LEFT JOIN tlab_prereg_conclusion c ON c.prereg_id = p.prereg_id
            {where}
            ORDER BY p.created_at DESC, p.prereg_id DESC
            """,
            params,
        )
        return cur.fetchall()


# ============================================================================================
# CLI — ensure-schema / record (from stdin JSON) / aggregate / findings. The shell-harness front door.
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


def _prereg_from_json(obj: Mapping[str, Any]) -> tuple[PreReg, Optional[dict], Optional[str]]:
    """Parse a JSON pre-registration object (the CLI stdin contract) into (PreReg, stamp, host). The criterion
    is given as EITHER an explicit {metric, bins:[{name,lo,hi,verdict,decisive}]} object (lo/hi accept the
    tokens 'NEG_INF'/'POS_INF' or numbers), OR the convenience `cuts` shape: {metric, cuts:[[upper, name,
    verdict, decisive], ...]}. A malformed criterion / missing field is a loud error via the Criterion/PreReg
    constructors (ADR-0002 — the partition is re-validated, not trusted)."""
    crit_obj = dict(obj["criterion"])
    if "cuts" in crit_obj:
        cuts = [(float("inf") if c[0] in ("POS_INF", None) else float(c[0]),
                 str(c[1]), str(c[2]), bool(c[3])) for c in crit_obj["cuts"]]
        criterion = thresholds_criterion(str(crit_obj["metric"]), cuts)
    else:
        criterion = Criterion.from_json(crit_obj)
    prereg = PreReg(prereg_key=obj["prereg_key"], question=obj["question"], criterion=criterion,
                    rationale=obj["rationale"], method=obj.get("method"),
                    refs=obj.get("refs", {}), notes=obj.get("notes"))
    stamp = obj.get("stamp")
    host = obj.get("host")
    return prereg, (dict(stamp) if stamp is not None else None), host


def main() -> int:
    ap = argparse.ArgumentParser(description="throughput-lab postgres egress: schema + record + aggregate")
    ap.add_argument("--ensure-schema", action="store_true",
                    help="create the tlab_config/tlab_reading tables (idempotent) and exit")
    ap.add_argument("--record", action="store_true",
                    help="read ONE JSON reading object from stdin and insert it; print the new reading_id")
    ap.add_argument("--aggregate", action="store_true",
                    help="print the median/min/max-per-config aggregate table")
    ap.add_argument("--tag", default=None, help="filter --aggregate to one tag")
    ap.add_argument("--record-finding", action="store_true",
                    help="read ONE JSON finding object (scope, interpretation, ...) from stdin and append it")
    ap.add_argument("--findings", action="store_true",
                    help="print the belief layer (interpretations); --scope filters, --current = live only")
    ap.add_argument("--scope", default=None, help="filter --findings to one scope")
    ap.add_argument("--current", action="store_true", help="--findings: only findings nothing supersedes")
    ap.add_argument("--record-prereg", action="store_true",
                    help="read ONE JSON pre-registration (prereg_key, question, metric, criterion, rationale, "
                         "[method,...]) from stdin and register it; print the new prereg_id")
    ap.add_argument("--conclude-prereg", type=int, metavar="PREREG_ID", default=None,
                    help="conclude pre-registration PREREG_ID against --observed; prints the mechanical verdict")
    ap.add_argument("--observed", type=float, default=None,
                    help="the measured value of the metric, judged against the frozen criterion (--conclude-prereg)")
    ap.add_argument("--resolved-by-reading", type=int, default=None,
                    help="--conclude-prereg: the reading_id the verdict rests on")
    ap.add_argument("--resolved-by-finding", type=int, default=None,
                    help="--conclude-prereg: the finding_id that interprets the verdict")
    ap.add_argument("--abandon-prereg", type=int, metavar="PREREG_ID", default=None,
                    help="close pre-registration PREREG_ID with no verdict (--note should say why)")
    ap.add_argument("--note", default=None, help="note for --conclude-prereg / --abandon-prereg")
    ap.add_argument("--preregs", action="store_true",
                    help="print the pre-registration layer (criteria + verdicts); --open = un-concluded only")
    ap.add_argument("--open", action="store_true", help="--preregs: only pre-registrations not yet concluded")
    a = ap.parse_args()
    if not (a.ensure_schema or a.record or a.aggregate or a.record_finding or a.findings
            or a.record_prereg or a.conclude_prereg is not None or a.abandon_prereg is not None or a.preregs):
        ap.error("nothing to do: pass --ensure-schema / --record / --aggregate / --record-finding / --findings "
                 "/ --record-prereg / --conclude-prereg / --abandon-prereg / --preregs")
    conn = connect()
    try:
        if a.ensure_schema:
            ensure_schema(conn)
            print("[exp_db] schema ensured (tlab_config/tlab_reading/tlab_finding/tlab_prereg)")
        if a.record:
            ensure_schema(conn)
            obj = json.load(sys.stdin)
            key, reading, stamp, host = _reading_from_json(obj)
            # the shell-harness front door uses the loud-but-non-fatal form: a DB blip dumps the reading under
            # ~/w/vdc + warns, but never fails the benchmark that produced it (ADR-0002 weighed against a run).
            rid = record_reading_safe(conn, key, reading, stamp=stamp, host=host)
            if rid is not None:
                print(rid)
        if a.aggregate:
            rows = aggregate(conn, tag=a.tag)
            hdr = ("config_id", "driver", "ladder", "max_batch", "msg_rows", "fibers", "n_sims",
                   "n", "leaf_rows_s med", "min", "max", "dps med", "util med", "rows/fwd med")
            print("\t".join(hdr))
            for row in rows:
                print("\t".join("" if v is None else
                                 (f"{v:.1f}" if isinstance(v, float) else str(v)) for v in row))
        if a.record_finding:
            ensure_schema(conn)
            obj = json.load(sys.stdin)        # {scope, interpretation, [motivation, status, supersedes, ...]}
            scope = obj.pop("scope")
            interp = obj.pop("interpretation")
            stamp = obj.pop("stamp", None)
            host = obj.pop("host", None)
            fid = record_finding(conn, scope, interp, stamp=stamp, host=host, **obj)
            print(fid)
        if a.findings:
            for row in findings(conn, scope=a.scope, current_only=a.current):
                fid, ts, commit, tree, scope, status, sup, motiv, interp, change = row
                sup_s = f" supersedes={sup}" if sup else ""
                print(f"[{fid}] {ts:%Y-%m-%d} {commit}/{tree} <{scope}> {status.upper()}{sup_s}\n"
                      f"      motivation: {motiv or '—'}\n      interpretation: {interp}"
                      + (f"\n      → {change}" if change else ""))
        if a.record_prereg:
            ensure_schema(conn)
            obj = json.load(sys.stdin)        # {prereg_key, question, criterion:{metric, cuts|bins}, rationale, ...}
            prereg, stamp, host = _prereg_from_json(obj)
            # loud-but-non-fatal: a harness pre-registers at run start; a DB blip dumps the criterion under
            # ~/w/vdc + warns, never silently loses the accountability artifact (ADR-0002 weighed against a run).
            pid = record_prereg_safe(conn, prereg, stamp=stamp, host=host)
            if pid is not None:
                print(pid)
        if a.conclude_prereg is not None:
            ensure_schema(conn)
            if a.observed is None:
                ap.error("--conclude-prereg requires --observed (the measured value to judge)")
            cid, hit = conclude_prereg(conn, a.conclude_prereg, a.observed,
                                       resolved_by_reading=a.resolved_by_reading,
                                       resolved_by_finding=a.resolved_by_finding, notes=a.note)
            verdict = "DECISIVE" if hit.decisive else "AMBIGUOUS (criterion not met → escalate, ADR-0014)"
            print(f"[conclusion {cid}] prereg {a.conclude_prereg}: observed {hit.value:g} → "
                  f"bin '{hit.bin_name}' (margin {hit.margin:g}) → {verdict}\n      verdict: {hit.verdict}")
        if a.abandon_prereg is not None:
            ensure_schema(conn)
            cid = abandon_prereg(conn, a.abandon_prereg, notes=a.note)
            print(f"[conclusion {cid}] prereg {a.abandon_prereg}: ABANDONED")
        if a.preregs:
            for row in preregs(conn, open_only=a.open):
                pid, ts, commit, tree, key, metric, question, outcome, observed, bin_name, bin_verdict, margin = row
                if outcome is None:
                    verdict_s = "OPEN (awaiting verdict)"
                elif outcome == "abandoned":
                    verdict_s = "ABANDONED"
                else:
                    verdict_s = (f"{outcome.upper()}: observed {observed:g} → '{bin_name}' (margin {margin:g})")
                print(f"[{pid}] {ts:%Y-%m-%d} {commit}/{tree} <{key}> metric={metric}\n"
                      f"      Q: {question}\n      {verdict_s}"
                      + (f"\n      → {bin_verdict}" if bin_verdict else ""))
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
