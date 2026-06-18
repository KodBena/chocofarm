#!/usr/bin/env python3
"""
cpp/stage_a/stage_a_analyze.py — reduce a Stage A sweep JSONL to the throughput surface + the
pre-registered-prediction verdicts (docs/design/cpp-eval-transport-adapter.md §4). THROWAWAY harness.

Reads a sweep_*.jsonl (one row per (e_policy, wakeup, S, D, rep)), reports per-cell median + min-max
leaves/s, the global-max cell, and HELD/REFUTED for each design prediction (a)-(e). A cell "beats"
another only if its MIN > the other's MAX (the design's non-overlapping-bands bar).

dps conversion: the design pins gen-ceiling = 152 dps/core = 76,000 leaves/s, i.e. ~500 leaves/decision
(a sims256 Gumbel tree's leaf count). implied_dps = leaves_per_s / 500. (The strict-barrier 49 dps and
greedy-async 37 dps reference points are on the same per-core basis.)

Public Domain (The Unlicense).
"""
from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict

LEAVES_PER_DECISION = 500.0   # 76000 leaves/s / 152 dps/core (design gen-ceiling)


def dps(lps: float) -> float:
    return lps / LEAVES_PER_DECISION


def load(path: str) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def cell_key(r: dict) -> tuple:
    return (r["e_policy"], r["wakeup"], r["S"], r["D"])


def main() -> int:
    path = sys.argv[1]
    rows = load(path)
    cells: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        cells[cell_key(r)].append(r)

    agg = {}
    for k, rs in cells.items():
        lps = [r["leaves_per_s"] for r in rs]
        agg[k] = {
            "median": statistics.median(lps), "min": min(lps), "max": max(lps),
            "n": len(lps),
            "mean_rows_per_fwd": statistics.median([r["mean_rows_per_fwd"] for r in rs]),
            "pad_fraction": statistics.median([r["pad_fraction"] for r in rs]),
        }

    def med(e, w, S, D):
        return agg.get((e, w, S, D), {}).get("median", float("nan"))

    def cell(e, w, S, D):
        return agg.get((e, w, S, D), {})

    print(f"=== Stage A throughput surface ({path}) ===")
    print(f"reps/cell (median n): {statistics.median([a['n'] for a in agg.values()])}")
    print()
    # the surface table, per (E, wakeup)
    for e in ("padmax", "bucket"):
        for w in ("group", "leaf"):
            print(f"--- E={e}  wakeup={w} :  leaves/s  median [min..max]  (rows/fwd, pad) ---")
            hdr = "  S\\D " + "".join(f"{D:>22}" for D in (1, 2, 8, 32, 128))
            print(hdr)
            for S in (1, 4, 16, 64):
                cells_s = []
                for D in (1, 2, 8, 32, 128):
                    c = cell(e, w, S, D)
                    if c:
                        cells_s.append(f"{c['median']:>8.0f}[{c['min']:>7.0f}-{c['max']:>7.0f}]")
                    else:
                        cells_s.append(f"{'--':>22}")
                print(f"  {S:>3} " + "".join(f"{x:>22}" for x in cells_s))
            print()

    # global max
    gmax_k = max(agg, key=lambda k: agg[k]["median"])
    gmax = agg[gmax_k]
    print("=== GLOBAL MAX cell ===")
    print(f"  (E={gmax_k[0]}, wakeup={gmax_k[1]}, S={gmax_k[2]}, D={gmax_k[3]})  "
          f"median={gmax['median']:.0f} leaves/s  [{gmax['min']:.0f}..{gmax['max']:.0f}]  "
          f"implied_dps={dps(gmax['median']):.0f}  rows/fwd={gmax['mean_rows_per_fwd']:.1f}  "
          f"pad={gmax['pad_fraction']:.2f}")
    print()

    def beats(ca, cb) -> bool:
        """ca beats cb iff ca.min > cb.max (non-overlapping bands)."""
        return bool(ca) and bool(cb) and ca["min"] > cb["max"]

    print("=== pre-registered predictions ===")

    # (a) S=1 corner frame-bound regardless of D (flat in D), ~37 dps
    s1 = {D: cell("bucket", "group", 1, D) for D in (1, 2, 8, 32, 128)}
    s1_med = {D: c.get("median", float("nan")) for D, c in s1.items()}
    s1_vals = [v for v in s1_med.values() if v == v]
    s1_spread = (max(s1_vals) / min(s1_vals)) if s1_vals else float("nan")
    s1_dps = {D: dps(v) for D, v in s1_med.items()}
    print(f"(a) S=1 frame-bound, flat in D (~37 dps):")
    print(f"    S=1 leaves/s by D (bucket,group): " +
          ", ".join(f"D{D}={s1_med[D]:.0f}({s1_dps[D]:.0f}dps)" for D in (1, 2, 8, 32, 128)))
    print(f"    max/min spread across D = {s1_spread:.1f}x  "
          f"(flat-in-D if ~1x; rises with D if NOT flat)")
    print()

    # (b) D=1 corner serialization-bound regardless of S, ~49 dps
    d1 = {S: cell("bucket", "group", S, 1) for S in (1, 4, 16, 64)}
    d1_med = {S: c.get("median", float("nan")) for S, c in d1.items()}
    d1_dps = {S: dps(v) for S, v in d1_med.items()}
    print(f"(b) D=1 serialization-bound regardless of S (~49 dps):")
    print(f"    D=1 leaves/s by S (bucket,group): " +
          ", ".join(f"S{S}={d1_med[S]:.0f}({d1_dps[S]:.0f}dps)" for S in (1, 4, 16, 64)))
    print()

    # (c) (S>=16, D>=8, bucket) beats both corners and is the global max
    target_cells = [("bucket", "group", S, D) for S in (16, 64) for D in (8, 32, 128)]
    best_target_k = max(target_cells, key=lambda k: agg.get(k, {}).get("median", -1))
    bt = agg.get(best_target_k, {})
    # the best S=1 corner cell and best D=1 corner cell (any E/wakeup)
    s1_best_k = max([k for k in agg if k[2] == 1], key=lambda k: agg[k]["median"])
    d1_best_k = max([k for k in agg if k[3] == 1], key=lambda k: agg[k]["median"])
    print(f"(c) (S>=16,D>=8,bucket) beats both corners AND is the global max:")
    print(f"    best target cell {best_target_k}: median={bt.get('median',0):.0f} "
          f"[{bt.get('min',0):.0f}..{bt.get('max',0):.0f}] implied_dps={dps(bt.get('median',0)):.0f}")
    print(f"    best S=1 corner {s1_best_k}: [{agg[s1_best_k]['min']:.0f}..{agg[s1_best_k]['max']:.0f}]")
    print(f"    best D=1 corner {d1_best_k}: [{agg[d1_best_k]['min']:.0f}..{agg[d1_best_k]['max']:.0f}]")
    print(f"    beats S=1 corner (min>max): {beats(bt, agg[s1_best_k])}")
    print(f"    beats D=1 corner (min>max): {beats(bt, agg[d1_best_k])}")
    print(f"    target cell IS global max: {best_target_k == gmax_k}  (global max = {gmax_k})")
    print()

    # (d) pad-to-max underperforms bucket on partial drains by ~pad ratio
    print(f"(d) pad-to-max underperforms bucket on PARTIAL drains by ~the pad ratio:")
    for (S, D) in [(1, 1), (1, 2), (4, 8), (16, 8)]:
        cp = cell("padmax", "group", S, D)
        cb = cell("bucket", "group", S, D)
        if cp and cb:
            ratio = cb["median"] / cp["median"] if cp["median"] else float("nan")
            print(f"    S={S:>2} D={D:>3}: padmax={cp['median']:>8.0f} (rows/fwd={cp['mean_rows_per_fwd']:.0f}"
                  f",pad={cp['pad_fraction']:.2f})  bucket={cb['median']:>8.0f} "
                  f"(rows/fwd={cb['mean_rows_per_fwd']:.0f},pad={cb['pad_fraction']:.2f})  "
                  f"bucket/padmax={ratio:.2f}x  bucket_beats_padmax={beats(cb, cp)}")
    print()

    # (e) per-group wakeup >= per-leaf, widening as leaves/s rises
    print(f"(e) per-group wakeup >= per-leaf, widening as leaves/s rises:")
    for (S, D) in [(1, 1), (4, 8), (16, 8), (64, 32), (64, 128)]:
        cg = cell("bucket", "group", S, D)
        cl = cell("bucket", "leaf", S, D)
        if cg and cl:
            ratio = cg["median"] / cl["median"] if cl["median"] else float("nan")
            print(f"    S={S:>2} D={D:>3}: group={cg['median']:>8.0f}  leaf={cl['median']:>8.0f}  "
                  f"group/leaf={ratio:.2f}x  group>=leaf(min>max):{beats(cg, cl)}")
    print()

    print("=== reference points (implied dps @ 500 leaves/decision) ===")
    print(f"  strict-barrier ref: 49 dps/core  |  greedy-async ref: 37 dps/core")
    print(f"  gen-ceiling: 152 dps/core (76k leaves/s)  |  serve-ceiling: ~190k-264k leaves/s/core")
    print(f"  GLOBAL MAX implied: {dps(gmax['median']):.0f} dps/core  "
          f"({gmax['median']:.0f} leaves/s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
