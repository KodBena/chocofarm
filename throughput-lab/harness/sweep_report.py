#!/usr/bin/env python3
"""
throughput-lab/harness/sweep_report.py — aggregate a sweep's per-grid-point JSON (cells/*.json, each a
run_lab.py record array) into a single ranked markdown report. Each cell is read through the typed
`sweep_common.CellLedger` (the shared cross-layer reconciliation + health verdict), so this report and
sweep_analyze.py agree, by construction, on which cells are valid measurements — and neither compares
raw cross-layer counters (see sweep_common for why that is a category error).

Ranks the DECOUPLED cells (the throughput mode) by median achieved throughput; lists the COUPLED cells
apart (RTT-bound by design, not a throughput failure); surfaces any non-`valid` verdict (a real wedge /
transport drop / errored cell) loudly in its own table rather than letting it average in.

The "best observed" it names is exactly that — the best point IN THIS GRID, ON THIS RUN: a MEASURED
observation with its replicate spread, not a proven optimum and not a ceiling (ADR-0009 — report what
was measured, do not launder it into a proof).

Run:  PYTHONPATH=throughput-lab python harness/sweep_report.py <sweep-outdir>
      PYTHONPATH=throughput-lab python harness/sweep_report.py --cell <cells/one.json>   # one point's lines

Public Domain (The Unlicense).
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

from sweep_common import CellLedger, load_cell_file, load_ledgers


def _leaderboard(ledgers: "list[CellLedger]", title: str) -> "list[str]":
    out = [f"\n## {title}\n"]
    out.append("| rank | topology | mode | thr | rate/thr | rows | max_batch | "
               "leaf-rows/s (median) | min–max | req/s | offered req/s | srv_util% | lat_p50_ms | health |")
    out.append("| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |")
    for i, L in enumerate(ledgers, 1):
        out.append(
            f"| {i} | {L.get('topology')} | {L.get('mode')} | {L.get('threads')} | "
            f"{float(L.get('rate_hz_per_thread', 0)):,.0f} | {L.get('rows_per_batch')} | {L.get('max_batch')} | "
            f"{L.served_rows_hz:,.0f} | {L.served_rows_min:,.0f}–{L.served_rows_max:,.0f} | {L.served_hz:,.0f} | "
            f"{L.offered_hz:,.0f} | {float(L.get('server_compute_util_pct', 0)):.0f} | "
            f"{float(L.get('lat_p50_us', 0)) / 1000.0:.2f} | {L.health or '·'} |"
        )
    return out


def progress_block(cell_json: str) -> str:
    """Compact, human-readable lines for ONE grid point's cells — appended live to the sweep's
    progress.log as each point finishes (so `tail -f progress.log` shows cells stream in)."""
    ledgers = load_cell_file(cell_json)
    if not ledgers:
        return f"[??:??:??] {Path(cell_json).name}: empty/unreadable\n"
    ts = datetime.now().strftime("%H:%M:%S")
    L0 = ledgers[0]
    lines = [f"[{ts}] point  threads={L0.get('threads')}  rate={float(L0.get('rate_hz_per_thread', 0)):.0f}"
             f"  rows={L0.get('rows_per_batch')}  max_batch={L0.get('max_batch')}"]
    for L in ledgers:
        tag = (L.health or "ok") if L.is_valid else f"!!{L.health or L.verdict}"
        lines.append(
            f"    {str(L.get('topology')):<11} {str(L.get('mode')):<10} "
            f"{L.served_rows_hz:>11,.0f} rows/s ({L.served_hz:>8,.0f} req/s)  "
            f"util {float(L.get('server_compute_util_pct', 0)):>3.0f}%  "
            f"batch {float(L.get('server_mean_batch_rows', 0)):>6.1f}r  "
            f"p50 {float(L.get('lat_p50_us', 0)) / 1000.0:>8.2f}ms  {tag}"
        )
    return "\n".join(lines) + "\n"


def main(outdir_s: str) -> int:
    outdir = Path(outdir_s)
    ledgers = load_ledgers(outdir)
    if not ledgers:
        print(f"sweep_report.py: no cell records found under {outdir}/cells", file=sys.stderr)
        return 1

    valid = [L for L in ledgers if L.is_valid]
    dec = sorted([L for L in valid if L.get("mode") == "decoupled"], key=lambda L: -L.served_rows_hz)
    cou = sorted([L for L in valid if L.get("mode") == "coupled"], key=lambda L: -L.served_rows_hz)
    bad = [L for L in ledgers if not L.is_valid]

    out: "list[str]" = ["# throughput-lab sweep report\n"]
    manifest = outdir / "MANIFEST.txt"
    if manifest.exists():
        out.append("```\n" + manifest.read_text().strip() + "\n```\n")
    out.append(f"Cells: **{len(ledgers)}**  (valid decoupled: {len(dec)}, valid coupled: {len(cou)}, "
               f"non-valid: {len(bad)}).  Ranked by **leaf-rows/s** = served req/s × rows (the leaf-eval-"
               f"meaningful work rate — req/s alone favours small rows: more, tinier round-trips for the same "
               f"compute). `req/s` is completed round-trips (`replies_recv/seconds`); `offered req/s` is the "
               f"producer send rate (a flood inflates it while the server serves ~0 — a big offered≫served gap "
               f"at low srv_util is that tell). Coupled is RTT-bound, ranked separately. `health` is the "
               f"reconciliation verdict (see sweep_common.CellLedger).")

    if dec:
        b = dec[0]
        out.append("\n## Best observed throughput (this grid, this run)\n")
        out.append(
            f"**{b.served_rows_hz:,.0f} leaf-rows/s served** ({b.served_hz:,.0f} req/s; min–max "
            f"{b.served_rows_min:,.0f}–{b.served_rows_max:,.0f} over {len(b.served_replicates)} replicates; "
            f"producer offered {b.offered_hz:,.0f} req/s) at:\n\n"
            f"- topology **{b.get('topology')}**, mode decoupled, **{b.get('threads')} threads**, "
            f"rate {float(b.get('rate_hz_per_thread', 0)):,.0f} hz/thread, "
            f"**rows={b.get('rows_per_batch')}**, **max_batch={b.get('max_batch')}**\n"
            f"- server compute-util {float(b.get('server_compute_util_pct', 0)):.0f}%, "
            f"mean batch {float(b.get('server_mean_batch_rows', 0)):.1f} rows, p50 latency "
            f"{float(b.get('lat_p50_us', 0)) / 1000.0:.2f} ms\n"
        )
        out.append(
            "> This is the best **observed** point in this grid on this 4-vCPU guest — a measured "
            "observation with its replicate spread, **not** a proven optimum and **not** a ceiling. A "
            "wider grid or deeper measurement (more replicates / longer seconds) can move it. Treat it as "
            "a lead to confirm, not a settled bound.\n"
        )

    out += _leaderboard(dec, "Decoupled — throughput leaderboard (median achieved, desc)")
    if cou:
        out += _leaderboard(cou, "Coupled — RTT-bound (median achieved, desc; lower by design)")

    out.append("\n## Non-valid cells (surfaced, never averaged in)\n")
    if not bad:
        out.append("None — every cell reconciled clean (rows conserved, replies complete modulo the "
                   "decoupled async tail).")
    else:
        out.append("| topology | mode | thr | rate/thr | rows | max_batch | verdict | detail |")
        out.append("| --- | --- | ---: | ---: | ---: | ---: | --- | --- |")
        for L in bad:
            out.append(
                f"| {L.get('topology')} | {L.get('mode')} | {L.get('threads')} | "
                f"{float(L.get('rate_hz_per_thread', 0)):,.0f} | {L.get('rows_per_batch')} | "
                f"{L.get('max_batch')} | {L.verdict} | {(L.health or L.note or '')[:80]} |"
            )

    out.append("\n## Reading the sweep\n")
    out.append(
        "- **rows** and **max_batch** are the two real throughput levers: bigger producer batches (`rows`) "
        "and a larger server gather cap (`max_batch`) amortise the per-message wire + dispatch cost over "
        "more leaf rows. **coalescing** trades per-thread sockets for one coalescing thread (fewer, larger "
        "wire messages — that is why its `batches_sent > server_requests`, which is NOT a drop).\n"
        "- **server_poll_ms is no longer a throughput lever** — the reply wake pipe flushes a reply the "
        "instant it is ready (it was fixed small here only to keep stop() latency low).\n"
        "- A cell with **sat% << 100** is saturated: the producer asked for more than the server could "
        "serve, so achieved is the serving ceiling for that config (the intended way to read max throughput).\n"
        "- To harden a row before citing it: raise `REPLICATES`/`SECONDS_PER`, and re-run with "
        "`OUTDIR=<this dir>` to extend rather than redo.\n"
    )

    sys.stdout.write("\n".join(out) + "\n")
    return 0


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--cell":
        sys.stdout.write(progress_block(sys.argv[2]))
        raise SystemExit(0)
    if len(sys.argv) != 2:
        print("usage: sweep_report.py <sweep-outdir>   |   sweep_report.py --cell <cell.json>",
              file=sys.stderr)
        raise SystemExit(2)
    raise SystemExit(main(sys.argv[1]))
