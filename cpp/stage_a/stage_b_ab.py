#!/usr/bin/env python3
"""
cpp/stage_a/stage_b_ab.py — the Stage B e2e A/B orchestrator for the eval-transport-adapter
(docs/design/cpp-eval-transport-adapter.md §4 Stage B). A THROWAWAY bench harness (NOT a committed
fixture, NOT the production server, does NOT touch the production default path).

It stands up the bucketed-E + group-wakeup InferenceServer (the Stage A StageAServer subclass — a
server FLAG, not the production drain) over the SAME 241->256->65 MLP it publishes to redis at
(run,"gen",version), then runs the C++ wire-ab-bench (the real Gumbel-AZ search, every leaf remote)
under BOTH transport modes:

  * arm 1: --wire-mode strict-barrier   (the production default run_episodes_wire_batched)
  * arm 3: --wire-mode pipelined-bucket  (run_episodes_wire_pipelined — D>1 non-blocking)

at 1 thread AND 2 threads (the per-core-drop test), >=5 iterations/arm, reporting decisions/s/core as
mean +/- stddev AND min-max (ADR-0009). For arm 3 it ALSO records the server's mean rows/FORWARD (the
in-flight depth a single real tree sustains — the key number for the overcommit phase). The server is
torn down per-arm cleanly (server.stop/thread.join/server.close — never a nohup'd foreground sleep).

Output is written under ~/w/vdc/chocobo/runs/ (never /tmp), per the storage discipline.

Usage:
    python stage_b_ab.py [--secs 8 --iters 5 --hidden 256 --m 24 --n-sims 256
                          --pool-batch 64 --inflight-msgs 8 --out <dir>]

Public Domain (The Unlicense).
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import threading
import time
import uuid

REPO = "/home/bork/w/vdc/1/chocofarm"
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # for stage_a_server

import chocofarm.config  # noqa: F401,E402 — XLA/OMP single-thread pin BEFORE jax init (SSOT)

from chocofarm.az.actions import n_action_slots  # noqa: E402
from chocofarm.az.features import feature_dim  # noqa: E402
from chocofarm.az.inference_server import (  # noqa: E402
    StaticParamsSource,
    jit_forward_core,
    params_from_manifest_blob,
)
from chocofarm.az.mlp import ValueMLP  # noqa: E402
from chocofarm.az.transport import RedisTransport, connect, pack_net  # noqa: E402
from chocofarm.model.env import Environment  # noqa: E402

from stage_a_server import BUCKETS, StageAServer  # noqa: E402

AB_BENCH = os.path.join(REPO, "cpp", "build", "chocofarm-wire-ab-bench")
INSTANCE = os.path.join(REPO, "chocofarm", "data", "instance.json")
FACES = os.path.join(REPO, "chocofarm", "data", "faces.json")


def build_and_publish(hidden: int, run: str, version: int):
    """Build ONE 241->H->65 net (the Stage A geometry: seed=17, residual=False), publish it to redis at
    (run,"gen",version) so the C++ bench's weight-read sanity passes, and return a StaticParamsSource over
    the SAME packed bytes so the in-process server serves the identical net (the only cross-arm difference
    is the transport schedule — the ADR-0012 P7 invariant Stage B validates)."""
    env = Environment()
    in_dim, n_actions = feature_dim(env), n_action_slots(env)
    net = ValueMLP(in_dim, hidden=hidden, n_actions=n_actions, seed=17,
                   y_mean=0.0, y_std=1.0, residual=False)
    manifest, blob = pack_net(net)
    # publish to redis (the wire bench reads this to confirm the run/version exists; same key the
    # production weight-read seam uses — P7). RedisTransport.publish_weights packs the SAME net.
    rt = RedisTransport(connect())
    rt.publish_weights(net, phase="gen", version=version, run=run)
    params, y_mean, y_std = params_from_manifest_blob(manifest, blob)
    return StaticParamsSource(params, y_mean, y_std), in_dim, n_actions


def start_server(src, endpoint: str, max_batch: int):
    """Stand up the bucketed-E + group-wakeup StageAServer (the Stage B server flag) over `src`. Warms
    every bucket shape + the max so a partial-drain forward never pays a cold JIT in the timed window."""
    server = StageAServer(src, bind=endpoint, max_batch=max_batch, forward_fn=jit_forward_core,
                          e_policy="bucket", wakeup="group")
    server.warmup(sorted(set(BUCKETS) | {max_batch}))
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    return server, t


def run_bench(wire_mode: str, endpoint: str, run: str, version: int, threads: int,
              secs: float, gc_m: int, n_sims: int, pool_batch: int, inflight: int,
              stats_path: str) -> dict:
    """Run ONE wire-ab-bench pass under taskset -c 0,1,2,3, parse its RESULT line. Returns the parsed
    metrics dict (dps, dps_per_core, episodes, decisions, wall)."""
    tok = f"stageb-{wire_mode}-{threads}t-{uuid.uuid4().hex[:8]}"
    cmd = [
        "taskset", "-c", "0,1,2,3", AB_BENCH,
        "--instance", INSTANCE, "--faces", FACES, "--endpoint", endpoint,
        "--run", run, "--version", str(version), "--res-token", tok,
        "--wire-mode", wire_mode, "--secs", str(secs),
        "--m", str(gc_m), "--n-sims", str(n_sims),
        "--pool-threads", str(threads), "--pool-batch", str(pool_batch),
        "--inflight-msgs", str(inflight), "--parity-stats", stats_path,
    ]
    proc = subprocess.run(cmd, cwd=REPO, text=True, capture_output=True, timeout=secs * 8 + 120)
    out = proc.stdout + proc.stderr
    result = {"raw": out, "rc": proc.returncode}
    for line in proc.stdout.splitlines():
        if line.startswith("RESULT:"):
            for tok2 in line.split():
                if "=" in tok2:
                    k, v = tok2.split("=", 1)
                    result[k] = v
    if proc.returncode != 0 or "dps_per_core" not in result:
        raise RuntimeError(f"wire-ab-bench failed (rc={proc.returncode}):\n{out}")
    return result


def parse_wire_summary(stats_path: str) -> dict:
    """Read the trailing {"wire_summary":1,...} JSON line the pipelined driver writes — mean rows/WIRE-
    MESSAGE (S). The server reports rows/FORWARD separately (its drain coalesces across messages)."""
    summary = {}
    try:
        with open(stats_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if obj.get("wire_summary"):
                    summary = obj
    except (OSError, json.JSONDecodeError):
        pass
    return summary


def agg(vals: list[float]) -> dict:
    return {
        "mean": statistics.mean(vals),
        "std": statistics.pstdev(vals) if len(vals) > 1 else 0.0,
        "min": min(vals),
        "max": max(vals),
        "n": len(vals),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--secs", type=float, default=8.0)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--m", type=int, default=24)
    ap.add_argument("--n-sims", type=int, default=256)
    ap.add_argument("--max-batch", type=int, default=512)
    ap.add_argument("--pool-batch", type=int, default=64)
    ap.add_argument("--inflight-msgs", type=int, default=8)
    ap.add_argument("--out", default=os.path.join(os.path.expanduser("~"), "w", "vdc", "chocobo",
                                                  "runs", "stage_b_ab"))
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    run = f"stageb-{stamp}"
    version = 0
    endpoint = f"ipc:///tmp/choco-stageb-{os.getpid()}.ipc"

    src, in_dim, n_actions = build_and_publish(a.hidden, run, version)
    print(f"[stage_b_ab] net published run={run} v={version} in_dim={in_dim} n_actions={n_actions} "
          f"hidden={a.hidden}", flush=True)

    # arms x thread-counts; >=5 iters each. The server stays warm for the whole sweep (one server, not
    # restarted between timed runs — design §7.1), tracking server stats per (arm, threads, iter).
    server, t = start_server(src, endpoint, a.max_batch)
    print(f"[stage_b_ab] server up (bucket+group) endpoint={endpoint} max_batch={a.max_batch}", flush=True)

    arms = [("strict-barrier", "arm1"), ("pipelined-bucket", "arm3")]
    thread_counts = [1, 2]
    records: list[dict] = []
    try:
        for wire_mode, arm in arms:
            for threads in thread_counts:
                for it in range(a.iters):
                    stats_path = os.path.join(a.out, f"wirestats-{arm}-{threads}t-{it}.jsonl")
                    fwd0, rows0 = server.n_forwards, server.n_real_rows
                    r = run_bench(wire_mode, endpoint, run, version, threads, a.secs, a.m,
                                  a.n_sims, a.pool_batch, a.inflight_msgs, stats_path)
                    fwd1, rows1 = server.n_forwards, server.n_real_rows
                    d_fwd = fwd1 - fwd0
                    d_rows = rows1 - rows0
                    mean_rows_per_fwd = (d_rows / d_fwd) if d_fwd else 0.0
                    wsum = parse_wire_summary(stats_path)
                    rec = {
                        "arm": arm, "wire_mode": wire_mode, "threads": threads, "iter": it,
                        "dps": float(r["dps"]), "dps_per_core": float(r["dps_per_core"]),
                        "episodes": int(r["episodes"]), "decisions": int(r["decisions"]),
                        "wall": float(r["wall"]),
                        "server_forwards": d_fwd, "server_rows": d_rows,
                        "server_mean_rows_per_fwd": mean_rows_per_fwd,
                        "wire_mean_rows_per_msg": wsum.get("mean_rows_per_msg", 0.0),
                    }
                    records.append(rec)
                    print(f"[stage_b_ab] {arm} {wire_mode} {threads}t it={it}: "
                          f"dps/core={rec['dps_per_core']:.2f} dps={rec['dps']:.2f} "
                          f"srv_rows/fwd={mean_rows_per_fwd:.2f} "
                          f"wire_rows/msg={rec['wire_mean_rows_per_msg']:.2f}", flush=True)
    finally:
        server.stop()
        t.join(timeout=5.0)
        server.close()

    # ---- aggregate + report ----
    summary: dict = {"run": run, "secs": a.secs, "iters": a.iters, "hidden": a.hidden, "m": a.m,
                     "n_sims": a.n_sims, "pool_batch": a.pool_batch, "inflight_msgs": a.inflight_msgs,
                     "serve_fast_region_B": 192, "cells": {}}
    for arm in ("arm1", "arm3"):
        for threads in (1, 2):
            cell = [r for r in records if r["arm"] == arm and r["threads"] == threads]
            dpc = [r["dps_per_core"] for r in cell]
            srv_rpf = [r["server_mean_rows_per_fwd"] for r in cell]
            wrpm = [r["wire_mean_rows_per_msg"] for r in cell]
            summary["cells"][f"{arm}-{threads}t"] = {
                "dps_per_core": agg(dpc),
                "server_rows_per_forward": agg(srv_rpf),
                "wire_rows_per_msg": agg(wrpm),
            }

    out_json = os.path.join(a.out, f"stage_b_ab-{stamp}.json")
    with open(out_json, "w") as f:
        json.dump({"summary": summary, "records": records}, f, indent=2)

    print("\n==== STAGE B A/B SUMMARY (dps/core, mean +/- std [min-max]) ====", flush=True)
    for threads in (1, 2):
        for arm in ("arm1", "arm3"):
            c = summary["cells"][f"{arm}-{threads}t"]["dps_per_core"]
            print(f"  {arm} ({'strict-barrier' if arm=='arm1' else 'pipelined-bucket'}) {threads}t: "
                  f"{c['mean']:.2f} +/- {c['std']:.2f}  [{c['min']:.2f}-{c['max']:.2f}]", flush=True)
    a3 = summary["cells"]["arm3-1t"]["server_rows_per_forward"]
    print(f"\n  arm3 server mean rows/forward (in-flight depth, 1t): "
          f"{a3['mean']:.2f} +/- {a3['std']:.2f}  [{a3['min']:.2f}-{a3['max']:.2f}]  "
          f"(serve fast region B~=192)", flush=True)
    # non-overlapping bands check at 1 thread
    a1_1t = summary["cells"]["arm1-1t"]["dps_per_core"]
    a3_1t = summary["cells"]["arm3-1t"]["dps_per_core"]
    nonoverlap = a3_1t["min"] > a1_1t["max"]
    print(f"\n  arm3 beats arm1 at 1t with NON-OVERLAPPING bands: {nonoverlap} "
          f"(arm3 min {a3_1t['min']:.2f} {'>' if nonoverlap else '<='} arm1 max {a1_1t['max']:.2f})",
          flush=True)
    print(f"\n[stage_b_ab] wrote {out_json}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
