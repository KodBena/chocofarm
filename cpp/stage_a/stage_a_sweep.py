#!/usr/bin/env python3
"""
cpp/stage_a/stage_a_sweep.py — the Stage A throughput-surface sweep driver for the eval-transport-adapter
design (docs/design/cpp-eval-transport-adapter.md §4). THROWAWAY session harness (NOT a committed
fixture). It owns the bench-scoped StageAServer (one warm server per (E-policy, wakeup) config — design
§7.1 same-warm-server discipline) and runs the C++ stage-a-transport-bench as a fresh subprocess for
each (S, D, rep) cell against it, parsing the RESULT line's leaves/s and the SERVER_STATS line's mean
rows/forward + pad fraction.

Sweep (design §4): S ∈ {1,4,16,64}; D ∈ {1,2,8,32,128}; E-policy ∈ {padmax(512), bucket{64,256,512}};
wakeup ∈ {group, leaf}. >=5 reps/cell. Writes one JSONL row per rep + a final summary to
~/w/vdc/chocobo/runs/stage_a_eval_transport/ (never /tmp; the ipc socket is transient in /tmp, fine).

Pin cores with taskset (the caller wraps this whole driver under `taskset -c 0,1,2,3`); the server JAX
forward runs single-threaded by construction (chocofarm.config XLA pin), the C++ producer is one DEALER
thread — the two share the 4 vCPUs.

Public Domain (The Unlicense).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone

REPO = "/home/bork/w/vdc/1/chocofarm"
sys.path.insert(0, os.path.join(REPO, "cpp", "stage_a"))
sys.path.insert(0, REPO)

import stage_a_server as sas  # noqa: E402

BENCH = os.path.join(REPO, "cpp", "build", "chocofarm-stage-a-transport-bench")
OUTDIR = os.path.expanduser("~/w/vdc/chocobo/runs/stage_a_eval_transport")

S_GRID = [1, 4, 16, 64]
D_GRID = [1, 2, 8, 32, 128]
E_GRID = ["padmax", "bucket"]
WAKE_GRID = ["group", "leaf"]

_RESULT_RE = re.compile(
    r"RESULT: PASS S=(\d+) D=(\d+) in_dim=(\d+) leaves=(\d+) msgs=(\d+) "
    r"wall=([\d.]+) leaves_per_s=([\d.]+) msgs_per_s=([\d.]+)")
_STATS_RE = re.compile(
    r"forwards=(\d+) real_rows=(\d+) padded_rows=(\d+) mean_real_rows_per_fwd=([\d.]+) "
    r"pad_fraction=([\d.]+) server_fwd_per_s=([\d.]+)")


def run_cell(endpoint: str, server: "sas.StageAServer", S: int, D: int,
             secs: float, warmup: float, timeout_ms: int) -> dict:
    """Run ONE measured window against the warm server, resetting the server's per-window counters first
    so mean_rows/forward + pad fraction reflect THIS cell only. Returns the parsed metrics."""
    server.n_forwards = 0
    server.n_real_rows = 0
    server.n_padded_rows = 0
    cmd = [BENCH, "--endpoint", endpoint, "--S", str(S), "--D", str(D),
           "--in-dim", "241", "--secs", str(secs), "--warmup-secs", str(warmup),
           "--timeout-ms", str(timeout_ms)]
    proc = subprocess.run(cmd, cwd=REPO, text=True, capture_output=True,
                          timeout=secs + warmup + 60)
    out = proc.stdout + proc.stderr
    m = _RESULT_RE.search(out)
    if proc.returncode != 0 or m is None:
        raise RuntimeError(f"bench cell S={S} D={D} failed (rc={proc.returncode}):\n{out[-2000:]}")
    leaves_per_s = float(m.group(7))
    # server stats for THIS window (forwards run during the cell, incl. the bench's warmup phase —
    # the mean rows/fwd + pad fraction are stable across warmup+measured, so this is representative).
    fwd = server.n_forwards
    real = server.n_real_rows
    pad = server.n_padded_rows
    mean_rows = (real / fwd) if fwd else 0.0
    pad_frac = (pad / (real + pad)) if (real + pad) else 0.0
    return {
        "S": S, "D": D, "leaves_per_s": leaves_per_s,
        "msgs_per_s": float(m.group(8)), "wall": float(m.group(6)),
        "server_forwards": fwd, "mean_rows_per_fwd": mean_rows, "pad_fraction": pad_frac,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--secs", type=float, default=2.5)
    ap.add_argument("--warmup", type=float, default=1.2)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--timeout-ms", type=int, default=20000)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    os.makedirs(OUTDIR, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    tag = (a.tag + "_") if a.tag else ""
    jsonl_path = os.path.join(OUTDIR, f"sweep_{tag}{stamp}.jsonl")
    print(f"[sweep] writing {jsonl_path}", flush=True)
    print(f"[sweep] grid S={S_GRID} D={D_GRID} E={E_GRID} wakeup={WAKE_GRID} reps={a.reps} "
          f"secs={a.secs} warmup={a.warmup}", flush=True)

    n_done = 0
    with open(jsonl_path, "w") as jf:
        for e_policy in E_GRID:
            for wakeup in WAKE_GRID:
                endpoint = f"ipc:///tmp/choco_stage_a_{e_policy}_{wakeup}.ipc"
                try:
                    os.unlink(endpoint.replace("ipc://", ""))
                except OSError:
                    pass
                server, in_dim, n_actions = sas.build(a.hidden, endpoint, 512, e_policy, wakeup)
                t = threading.Thread(target=server.serve_forever, daemon=True)
                t.start()
                print(f"[sweep] server up e={e_policy} wakeup={wakeup} in_dim={in_dim} "
                      f"endpoint={endpoint}", flush=True)
                try:
                    for S in S_GRID:
                        for D in D_GRID:
                            for rep in range(a.reps):
                                row = run_cell(endpoint, server, S, D, a.secs, a.warmup, a.timeout_ms)
                                row.update({"e_policy": e_policy, "wakeup": wakeup, "rep": rep,
                                            "in_dim": in_dim})
                                jf.write(json.dumps(row) + "\n")
                                jf.flush()
                                n_done += 1
                                print(f"[sweep] e={e_policy} wake={wakeup} S={S:>2} D={D:>3} "
                                      f"rep={rep} -> {row['leaves_per_s']:>10.0f} leaves/s "
                                      f"(rows/fwd={row['mean_rows_per_fwd']:.1f} "
                                      f"pad={row['pad_fraction']:.2f})", flush=True)
                finally:
                    server.stop()
                    t.join(timeout=5.0)
                    server.close()
                    print(f"[sweep] server down e={e_policy} wakeup={wakeup}", flush=True)
                    time.sleep(0.5)
    print(f"[sweep] DONE {n_done} cells -> {jsonl_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
