#!/usr/bin/env python3
"""
throughput-lab/harness/sweep_common.py — the shared, typed READING of a sweep cell (a run_lab
CellResult JSON record): the cross-layer reconciliation + ONE health verdict, so no consumer (report,
analysis, live progress) re-derives "is this a valid measurement" ad hoc — and so they cannot disagree.

THE LAYERS — why naive counter-equality is a CATEGORY ERROR
-----------------------------------------------------------
A cell's counters are measured at THREE layers, in DIFFERENT units, and are NOT meant to be equal:

  PRODUCER   batches_sent     logical leaf-batches emitted           (unit: producer batches)
             replies_recv     replies the producer read back         (unit: producer batches)
  WIRE       server_requests  Layer-2 messages the server received   (unit: WIRE messages)
  SERVER     server_rows      leaf rows the server drained           (unit: rows)

The COALESCING topology (B) MERGES many producer batches into fewer, larger wire messages, so
`batches_sent >= server_requests` BY DESIGN — comparing those two as if equal is the category error that
once flagged a healthy coalescing cell as a "drop". The quantities actually CONSERVED end-to-end are:

  * ROWS     — every row the producer sent must reach the server: server_rows == batches_sent*rows/batch.
  * REPLIES  — every send is answered, MODULO a small async tail in DECOUPLED free-run: the producer runs
               ahead and stops reading the last few in-flight replies when its measured window closes.
               That tail is LATENCY (it shows up as p50/p99), not loss.

This module owns those two reconciliations (and the coalescing ratio, a DESCRIPTIVE metric, not a loss)
and exposes one typed `verdict` (valid / degraded-rows / degraded-replies / failed). Consumers read the
verdict and the throughput accessors; they never compare the raw cross-layer counters — so the category
error is unrepresentable at the point it used to occur. The async-tail tolerance lives here ONCE, named,
rather than as a magic number sprinkled at a call site.

Public Domain (The Unlicense).
"""
from __future__ import annotations

import json
import statistics
from dataclasses import dataclass
from pathlib import Path

# The decoupled producer runs ahead and stops reading the last few in-flight replies at window close;
# that tail is LATENCY, not loss. A reconciliation within this fraction counts as COMPLETE. One named,
# documented home — never a bare 0.95 tacked onto a special case.
REPLY_TAIL_TOLERANCE = 0.02   # 2% — reply-completion / row-receipt >= 98% is "complete"


@dataclass(frozen=True)
class CellLedger:
    """The typed reconciliation of one sweep cell. Build via CellLedger(record); read `.verdict` /
    `.is_valid` / `.achieved_hz` / `.health`, NEVER the raw cross-layer counters (that is the category
    error this type exists to forbid). `.get(knob)` passes through the cell's own knob fields (topology,
    threads, ...) which ARE safe to read directly — they are this cell's identity, not a cross-layer
    comparison."""
    record: dict

    def get(self, key: str, default="?"):
        return self.record.get(key, default)

    # ---- run_lab's own success flag + failure note ----
    @property
    def ok(self) -> bool:
        return bool(self.record.get("ok"))

    @property
    def note(self) -> str:
        return (self.record.get("note") or "").strip()

    # ---- throughput (median + spread over replicates) ----
    @property
    def achieved_replicates(self) -> "list[float]":
        reps = [float(v) for v in (self.record.get("replicate_achieved_hz") or []) if v]
        if reps:
            return reps
        a = float(self.record.get("achieved_total_hz") or 0.0)
        return [a] if a else []

    @property
    def achieved_hz(self) -> float:
        reps = self.achieved_replicates
        return statistics.median(reps) if reps else 0.0

    @property
    def achieved_min(self) -> float:
        reps = self.achieved_replicates
        return min(reps) if reps else 0.0

    @property
    def achieved_max(self) -> float:
        reps = self.achieved_replicates
        return max(reps) if reps else 0.0

    # ---- SERVED throughput (completed round-trips) — the HONEST metric, vs the OFFERED send rate above.
    # A flooded server can show a huge SEND rate (achieved_*) while serving ~nothing (replies_recv ~0);
    # the leaderboard / regression rank by SERVED so a flood that completes nothing ranks LAST, not first.
    @property
    def served_replicates(self) -> "list[float]":
        raw = self.record.get("replicate_served_hz")
        if raw:                                       # keep zeros — a served rate of 0 (a wedge) is valid data
            return [float(v) for v in raw]
        recv, secs = self.record.get("replies_recv"), self.record.get("seconds")  # fallback for older records
        if recv is not None and secs:
            return [float(recv) / float(secs)]
        return self.achieved_replicates

    @property
    def served_hz(self) -> float:
        reps = self.served_replicates
        return statistics.median(reps) if reps else 0.0

    @property
    def served_min(self) -> float:
        reps = self.served_replicates
        return min(reps) if reps else 0.0

    @property
    def served_max(self) -> float:
        reps = self.served_replicates
        return max(reps) if reps else 0.0

    @property
    def offered_hz(self) -> float:
        """The producer's median SEND rate. NOT throughput — a flood inflates it while the server serves
        ~0; shown next to served_hz so the send/serve gap is visible (it is the tell of a wedged cell)."""
        return self.achieved_hz

    @property
    def requested_hz(self) -> float:
        return float(self.record.get("requested_total_hz") or 0.0)

    @property
    def saturation_pct(self) -> float:
        return (self.achieved_hz / self.requested_hz * 100.0) if self.requested_hz else 0.0

    # ---- the meaningful cross-layer reconciliations (the ONLY sanctioned comparisons) ----
    @property
    def reply_completion(self) -> float:
        """replies_recv / batches_sent — the in-band reply fraction (1.0 minus the async tail)."""
        sent = int(self.record.get("batches_sent") or 0)
        recv = int(self.record.get("replies_recv") or 0)
        return (recv / sent) if sent else 1.0

    @property
    def rows_received_fraction(self) -> float:
        """server_rows / (batches_sent * rows_per_batch) — did the server receive every row sent? This is
        the real conservation law; < 1 (beyond tolerance) is a genuine transport drop."""
        sent = int(self.record.get("batches_sent") or 0)
        rows_per = int(self.record.get("rows_per_batch") or 1)
        rows_sent = sent * rows_per
        srv_rows = int(self.record.get("server_rows") or 0)
        return (srv_rows / rows_sent) if rows_sent else 1.0

    @property
    def coalescing_ratio(self) -> float:
        """batches_sent / server_requests — producer batches per wire message (>= 1). DESCRIPTIVE: how
        much the coalescing thread merged; NOT a loss."""
        sent = int(self.record.get("batches_sent") or 0)
        srv_req = int(self.record.get("server_requests") or 0)
        return (sent / srv_req) if srv_req else 1.0

    @property
    def mode(self) -> str:
        return str(self.record.get("mode", "?"))

    # ---- the single typed verdict consumers act on ----
    @property
    def verdict(self) -> str:
        if not self.ok:
            return "failed"
        # The reply invariant is MODE-DEPENDENT. COUPLED waits for each reply before sending more, so a
        # materially-incomplete reply set is a real failure. DECOUPLED free-runs and ABANDONS its in-flight
        # replies when the measured window closes — and with the send-queue back-pressure the standing
        # queue makes replies_recv structurally far below batches_sent — so reply-completion is NOT a
        # validity signal in decoupled mode (throughput there is the achieved SEND rate, which back-pressure
        # ties to the server's serve rate). Encoding the invariant per-mode is the point of this type: it
        # stops a healthy back-pressured decoupled cell from being mistaken for a dropped one.
        if self.mode == "coupled" and self.reply_completion < 1.0 - REPLY_TAIL_TOLERANCE:
            return "degraded-replies"
        return "valid"

    @property
    def is_valid(self) -> bool:
        return self.verdict == "valid"

    @property
    def health(self) -> str:
        """A short human tag for the flag column / live feed: '' if pristine, 'recv NN%' as INFO when the
        producer abandoned an in-flight reply tail (expected in decoupled), else the failed reason."""
        v = self.verdict
        if v == "failed":
            return "FAILED" + (f": {self.note[:40]}" if self.note else "")
        if v == "degraded-replies":
            return f"LOSS recv {self.reply_completion * 100:.0f}%"
        if self.reply_completion < 0.995:
            return f"recv {self.reply_completion * 100:.0f}%"   # INFO: decoupled abandoned its in-flight tail
        return ""


def load_ledgers(outdir) -> "list[CellLedger]":
    """Every cell record under <outdir>/cells/*.json, wrapped as a CellLedger."""
    out: "list[CellLedger]" = []
    cdir = Path(outdir) / "cells"
    if not cdir.is_dir():
        return out
    for jf in sorted(cdir.glob("*.json")):
        try:
            recs = json.loads(jf.read_text())
        except Exception:  # noqa: BLE001 — a corrupt/partial json is skipped, surfaced by its absence
            continue
        for r in recs:
            r.setdefault("_src", jf.name)
            out.append(CellLedger(r))
    return out


def load_cell_file(path) -> "list[CellLedger]":
    """The CellLedgers for ONE cells/*.json file (used by the live per-point progress formatter)."""
    try:
        recs = json.loads(Path(path).read_text())
    except Exception:  # noqa: BLE001
        return []
    return [CellLedger(r) for r in recs]
