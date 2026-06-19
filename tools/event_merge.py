#!/usr/bin/env python3
"""
tools/event_merge.py — merge the C++ producer and Python server protocol event logs into ONE timeline.

Both sides write monotonic-timestamped lines (`<mono_ns> <SIDE> <kind> <fields>`) to SEPARATE files
(CHOCO_EVENTLOG_CPP and CHOCO_EVENTLOG) on the SAME CLOCK_MONOTONIC timebase. This tool merge-sorts them by
timestamp and DEBOUNCES the high-traffic events into per-window rollups, while surfacing every COLD forward
(an un-warmed XLA shape => a real compile) individually — so the leaf-eval protocol is legible in one place
and the XLA churn (if any) pops out as a stream of COLD lines.

Debounce / coalescing: events are bucketed into time windows (--window-ms). Within a window the high-traffic
kinds are collapsed to one rollup per (side, kind) — a coalescing key quotiented under a lax equivalence:
SUBMIT/RECV by side, FWD(warm) by width rounded to --bucket for display. COLD forwards bypass the rollup and
print immediately with their compile time.

Usage:
    event_merge.py [--window-ms 500] [--bucket 1] [--only-cold] <cpp.log> <py.log> [more.log ...]
    # offline, after a run; or tail two logs into it. Reads all files, sorts by timestamp, prints.

Public Domain (The Unlicense).
"""
from __future__ import annotations

import argparse
import sys


def parse(path):
    """Yield (ns:int, side:str, kind:str, fields:str) from one log file; skip malformed lines."""
    try:
        fh = open(path, encoding="utf-8", errors="replace")
    except OSError as e:
        print(f"# cannot open {path}: {e}", file=sys.stderr)
        return
    with fh:
        for line in fh:
            parts = line.rstrip("\n").split(" ", 3)
            if len(parts) < 3 or not parts[0].isdigit():
                continue
            ns = int(parts[0])
            side = parts[1]
            kind = parts[2]
            fields = parts[3] if len(parts) > 3 else ""
            yield ns, side, kind, fields


def kv(fields):
    """`a=1 b=2` -> {'a':'1','b':'2'}."""
    out = {}
    for tok in fields.split():
        if "=" in tok:
            k, v = tok.split("=", 1)
            out[k] = v
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("logs", nargs="+", help="event logs to merge (cpp + py)")
    ap.add_argument("--window-ms", type=float, default=500.0, help="coalescing window for high-traffic events")
    ap.add_argument("--bucket", type=int, default=1, help="round FWD width to this multiple for the display key")
    ap.add_argument("--only-cold", action="store_true", help="print only COLD compiles + their context counts")
    a = ap.parse_args()

    events = []
    for p in a.logs:
        events.extend(parse(p))
    if not events:
        print("# no events parsed", file=sys.stderr)
        return 1
    events.sort(key=lambda e: e[0])
    t0 = events[0][0]
    win_ns = int(a.window_ms * 1e6)

    distinct_cold = 0
    seen_widths = set()
    # rollup accumulators for the current window
    w_start = events[0][0]
    roll = {}          # (side, kind) -> count
    roll_widths = {}   # display-width-bucket -> count (warm forwards)
    roll_B = []        # producer SUBMIT batch sizes this window (coalescing degree per message)
    roll_real = []     # server FWD real-row counts this window (rows/forward — collapses in a phase-lock)

    def bucket(w):
        return (w // a.bucket) * a.bucket if a.bucket > 1 else w

    def flush(at_ns):
        if not roll and not roll_widths:
            return
        t = (w_start - t0) / 1e9
        t2 = (at_ns - t0) / 1e9
        bits = []
        for (side, kind), n in sorted(roll.items()):
            bits.append(f"{side}/{kind}x{n}")
        if roll_B:
            bits.append(f"B[{min(roll_B)}..{max(roll_B)}~{sum(roll_B)//len(roll_B)}]")
        if roll_real:
            bits.append(f"rows/fwd[{min(roll_real)}..{max(roll_real)}~{sum(roll_real)//len(roll_real)}]")
        if roll_widths:
            wd = ",".join(f"{w}x{n}" for w, n in sorted(roll_widths.items()))
            bits.append(f"warmFWD{{{wd}}}")
        if not a.only_cold:
            print(f"[{t:8.3f}..{t2:7.3f}s] " + "  ".join(bits))
        roll.clear()
        roll_widths.clear()
        roll_B.clear()
        roll_real.clear()

    for ns, side, kind, fields in events:
        if ns - w_start >= win_ns:
            flush(ns)
            w_start = ns
        d = kv(fields)
        if kind == "FWD" and d.get("cold") == "1":
            flush(ns)   # close the window so the cold event is in causal position
            w_start = ns
            distinct_cold += 1
            w = int(d.get("width", "0"))
            seen_widths.add(w)
            t = (ns - t0) / 1e9
            print(f"[{t:8.3f}s] >>> COLD COMPILE  width={w} real={d.get('real','?')} "
                  f"dt={int(d.get('dt_us','0')) / 1000:.1f}ms  (distinct_cold={distinct_cold})")
            continue
        # high-traffic: accumulate into the rollup
        roll[(side, kind)] = roll.get((side, kind), 0) + 1
        if kind == "FWD":
            w = int(d.get("width", "0"))
            seen_widths.add(w)
            bw = bucket(w)
            roll_widths[bw] = roll_widths.get(bw, 0) + 1
            try:
                roll_real.append(int(d.get("real", w)))
            except ValueError:
                pass
        elif kind == "SUBMIT" and "B" in d:
            try:
                roll_B.append(int(d["B"]))
            except ValueError:
                pass
    flush(events[-1][0] + 1)

    print(f"# summary: {len(events)} events, {distinct_cold} cold compiles, "
          f"{len(seen_widths)} distinct forward widths, span {(events[-1][0]-t0)/1e9:.3f}s",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
