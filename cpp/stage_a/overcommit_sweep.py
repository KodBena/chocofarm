#!/usr/bin/env python3
"""
cpp/stage_a/overcommit_sweep.py — the overcommit-phase increment (i) A/B sweep harness
(docs/design/cpp-eval-transport-adapter.md §6). A THROWAWAY bench (NOT a committed fixture, NOT the
production server, does NOT touch the production default path).

It measures M1 (N independent TreeStates per producer thread) + M3 (1:3 affinity pinning) on the
EXISTING greedy bucketed-E drain (M2 unchanged — the Stage A StageAServer bucket {64,256,512}).

Layout (this branch's worktree — the bench builds against the worktree's cpp/build, NEVER the main
checkout): REPO is this file's repo root (../../ from cpp/stage_a/).

Pinning (M3, 1:3): the in-process JAX server thread is pinned to core 0 (os.sched_setaffinity from
inside the server thread); the C++ producer bench subprocess is launched under `taskset -c 1,2,3` (the
3 producer cores). So 1 server core : 3 producer cores, the design's §6 M3.

Sweep: N (trees/thread) ∈ {1,2,3,4} under 3 producer threads + the pipelined-bucket arm. The baseline
arm is the strict-barrier single-tree run (N is structurally 1 there). >=5 iters/arm; the HEADLINE is
the server's mean rows/forward (does it climb 54 -> toward B≈192?), plus dps (aggregate + per-core).
mean +/- stddev + min-max; a cell wins only if its MIN beats the other's MAX (ADR-0009).

Output under ~/w/vdc/chocobo/runs/overcommit_sweep/ (never /tmp).

Usage:
    python overcommit_sweep.py [--secs 8 --iters 5 --hidden 256 --m 24 --n-sims 256
                                --pool-batch 64 --inflight-msgs 8 --threads 3
                                --trees 1,2,3,4 --server-core 0 --producer-cores 1,2,3 --out <dir>]

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

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # worktree root
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # for stage_a_server
sys.path.insert(0, os.path.join(REPO, "throughput-lab", "harness"))  # shared ADR-0011 code_stamp (one home)

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

from code_stamp import code_stamp  # noqa: E402 — ADR-0011: stamp every reading with its code state

AB_BENCH = os.path.join(REPO, "cpp", "build", "chocofarm-wire-ab-bench")
INSTANCE = os.path.join(REPO, "chocofarm", "data", "instance.json")
FACES = os.path.join(REPO, "chocofarm", "data", "faces.json")

# The wire-ab-bench is an HONEST warmup + (measure >= one full-occupancy episode-wave) time-box: BOTH the
# warmup and the measure pass run >= one full wave of `total_slots` episodes, and total_slots scales
# LINEARLY with N (trees/thread): total_slots = threads * N * ceil(pool_batch/threads). A wave runs
# total_slots episodes, and because the compute (the fixed producer cores + the single server core) is the
# bottleneck — not slot concurrency — the wave wall scales ~LINEARLY with total_slots (more slots = more
# episodes time-sliced over the same cores). So the per-iter cost scales with N — a fixed constant
# under-estimates high-N and trips the subprocess timeout (the --trees 8,9 failure). We DERIVE the wave
# wall from the slot geometry instead. (Calibration: N=1, threads=3, batch=64 -> total_slots=66 measured
# ~13.5s/wave => ~0.205s/slot; the constant below is set conservatively high so the timeout never clips.)
EST_SLOT_WAVE_S = 0.30   # est. wall to run one full-occupancy wave, per concurrent slot (conservative)
TIMEOUT_MARGIN_S = 60.0


def est_wave_s(threads: int, trees: int, pool_batch: int) -> float:
    """Estimate one full-occupancy episode-wave's wall (s) from the slot geometry — this scales with N
    (trees/thread), so it does NOT under-estimate high-N runs the way a fixed constant did."""
    k_base = (pool_batch + threads - 1) // max(1, threads)
    total_slots = max(1, threads) * max(1, trees) * max(1, k_base)
    return max(8.0, EST_SLOT_WAVE_S * total_slots)


def build_and_publish(hidden: int, run: str, version: int):
    """Build ONE 241->H->65 net (seed=17, residual=False), publish it to redis at (run,'gen',version)
    so the C++ bench's weight-read sanity passes, and return a StaticParamsSource over the SAME packed
    bytes so the in-process server serves the identical net."""
    env = Environment()
    in_dim, n_actions = feature_dim(env), n_action_slots(env)
    net = ValueMLP(in_dim, hidden=hidden, n_actions=n_actions, seed=17,
                   y_mean=0.0, y_std=1.0, residual=False)
    manifest, blob = pack_net(net)
    rt = RedisTransport(connect())
    rt.publish_weights(net, phase="gen", version=version, run=run)
    params, y_mean, y_std = params_from_manifest_blob(manifest, blob)
    return StaticParamsSource(params, y_mean, y_std), in_dim, n_actions


def start_server(src, endpoint: str, max_batch: int, server_core: int,
                 min_forward_rows: int = 0, max_queue_delay_ms: float = 0.0):
    """Stand up the bucketed-E + group-wakeup StageAServer (M2 unchanged). M3: pin the server thread to
    `server_core` from INSIDE the thread (os.sched_setaffinity(0, ...) binds the calling thread) so the
    forward runs on its isolated core. Warms every bucket shape + the max so a partial-drain forward
    never pays a cold JIT in the timed window.

    `min_forward_rows`/`max_queue_delay_ms` arm the increment-(ii) server floor (default OFF — the greedy
    drain); StageAServer forwards them to the base InferenceServer (server-floor-design.md)."""
    server = StageAServer(src, bind=endpoint, max_batch=max_batch, forward_fn=jit_forward_core,
                          e_policy="bucket", wakeup="group",
                          min_forward_rows=min_forward_rows, max_queue_delay_ms=max_queue_delay_ms)
    server.warmup(sorted(set(BUCKETS) | {max_batch}))

    def _serve():
        try:
            os.sched_setaffinity(0, {server_core})  # M3: pin THIS (server) thread to the server core
        except (OSError, AttributeError):
            pass
        server.serve_forever()

    t = threading.Thread(target=_serve, daemon=True)
    t.start()
    return server, t


def run_bench(wire_mode: str, endpoint: str, run: str, version: int, threads: int, trees: int,
              secs: float, gc_m: int, n_sims: int, pool_batch: int, inflight: int, min_coalesce: int,
              producer_cores: str, stats_path: str) -> dict:
    """Run ONE wire-ab-bench pass under `taskset -c <producer_cores>` (M3: the 3 producer cores), parse
    its RESULT line. Returns the parsed metrics dict."""
    tok = f"oc-{wire_mode}-{threads}t-N{trees}-{uuid.uuid4().hex[:8]}"
    cmd = [
        "taskset", "-c", producer_cores, AB_BENCH,
        "--instance", INSTANCE, "--faces", FACES, "--endpoint", endpoint,
        "--run", run, "--version", str(version), "--res-token", tok,
        "--wire-mode", wire_mode, "--secs", str(secs),
        "--m", str(gc_m), "--n-sims", str(n_sims),
        "--pool-threads", str(threads), "--pool-batch", str(pool_batch),
        "--inflight-msgs", str(inflight), "--trees-per-thread", str(trees),
        "--min-coalesce", str(min_coalesce),  # S_min: the producer-side closed convoy floor
        "--parity-stats", stats_path,
    ]
    # HONEST timeout, N-AWARE: BOTH warmup and measure run >= one full-occupancy wave, and a wave scales
    # LINEARLY with total_slots = threads*N*ceil(batch/threads). So timeout = warmup_wave + measure_wave +
    # margin, where each wave estimate scales with N (NOT the old fixed const that tripped at high N, NOR
    # the old secs*40+300 that papered over the oversize-pass overshoot).
    wave = est_wave_s(threads, trees, pool_batch)
    proc = subprocess.run(cmd, cwd=REPO, text=True, capture_output=True,
                          timeout=wave + max(secs, wave) + TIMEOUT_MARGIN_S)
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
    ap.add_argument("--secs", type=float, default=5.0)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--m", type=int, default=24)
    ap.add_argument("--n-sims", type=int, default=256)
    ap.add_argument("--max-batch", type=int, default=512)
    ap.add_argument("--pool-batch", type=int, default=64)
    ap.add_argument("--inflight-msgs", type=int, default=8)
    # S_min: the producer-side minimum coalescing degree (the closed convoy fix). Default 32 matches the
    # runner default; pass --min-coalesce 1 to reproduce the pre-fix (convoy-prone) behavior for an A/B.
    ap.add_argument("--min-coalesce", type=int, default=32)
    # The increment-(ii) SERVER floor (server-floor-design.md): θ rows / max delay. Default OFF (θ=0) so
    # the A/B's control arm is the byte-unchanged greedy drain; pass e.g. --min-forward-rows 192
    # --max-queue-delay-ms 3 for the treatment arm. (Applies to ALL cells in the run — run the sweep once
    # per θ to compare; candidates θ≈192, delay≈2–5 ms.)
    ap.add_argument("--min-forward-rows", type=int, default=0)
    ap.add_argument("--max-queue-delay-ms", type=float, default=0.0)
    ap.add_argument("--threads", type=int, default=3)  # 3 producer threads (1:3 pinning)
    # N sweep default capped at 3: N=4 has a SEPARATE nondeterministic stall bug (out of scope here) — the
    # operator can pass --trees 1,2,3,4 explicitly to probe it, but the default avoids it.
    ap.add_argument("--trees", default="1,2,3")        # N sweep (N<=3; see note above)
    ap.add_argument("--server-core", type=int, default=0)
    ap.add_argument("--producer-cores", default="1,2,3")
    ap.add_argument("--out", default=os.path.join(os.path.expanduser("~"), "w", "vdc", "chocobo",
                                                  "runs", "overcommit_sweep"))
    a = ap.parse_args()

    trees_list = [int(x) for x in a.trees.split(",") if x.strip()]
    os.makedirs(a.out, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    run = f"oc-{stamp}"
    version = 0
    endpoint = f"ipc:///tmp/choco-oc-{os.getpid()}.ipc"

    # Up-front EXPECTED total wall, N-AWARE: each cell's per-iter cost = warmup_wave + max(secs, wave),
    # and a wave scales with N (trees/thread). The baseline-strict cell is structurally N=1; each
    # overcommit cell carries its own N from trees_list. Summed over cells x iters so the high-N tail is
    # not under-counted (the bug a fixed constant hid). The bench reports the true bench_wall per iter.
    cell_trees = [1] + trees_list  # baseline-strict (N=1) then the overcommit N sweep
    est_total = 0.0
    for tr in cell_trees:
        wave = est_wave_s(a.threads, tr, a.pool_batch)
        est_total += a.iters * (wave + max(a.secs, wave))
    print(f"[overcommit_sweep] EXPECTED total wall ~= {est_total:.0f}s ({est_total/60.0:.1f} min): "
          f"{len(cell_trees)} cells x {a.iters} iters x (~warmup_wave + ~max(secs,wave)); "
          f"per-cell wave ~= {EST_SLOT_WAVE_S:.2f}s x total_slots (scales with N)", flush=True)
    t_sweep0 = time.perf_counter()

    src, in_dim, n_actions = build_and_publish(a.hidden, run, version)
    print(f"[overcommit_sweep] net published run={run} v={version} in_dim={in_dim} "
          f"n_actions={n_actions} hidden={a.hidden}", flush=True)

    # M3: server pinned to server-core (inside its thread); producers to producer-cores via taskset.
    server, t = start_server(src, endpoint, a.max_batch, a.server_core,
                             a.min_forward_rows, a.max_queue_delay_ms)
    print(f"[overcommit_sweep] server up (bucket+group) endpoint={endpoint} max_batch={a.max_batch} "
          f"server_core={a.server_core} producer_cores={a.producer_cores} threads={a.threads} "
          f"floor(theta={a.min_forward_rows},delay_ms={a.max_queue_delay_ms})",
          flush=True)

    records: list[dict] = []

    def one_cell(arm: str, wire_mode: str, threads: int, trees: int) -> None:
        for it in range(a.iters):
            stats_path = os.path.join(a.out, f"wirestats-{arm}-{stamp}-{it}.jsonl")
            fwd0, rows0 = server.n_forwards, server.n_real_rows
            t_iter0 = time.perf_counter()  # END-TO-END iteration wall (spawn + run + teardown)
            r = run_bench(wire_mode, endpoint, run, version, threads, trees, a.secs, a.m,
                          a.n_sims, a.pool_batch, a.inflight_msgs, a.min_coalesce,
                          a.producer_cores, stats_path)
            iter_wall = time.perf_counter() - t_iter0
            fwd1, rows1 = server.n_forwards, server.n_real_rows
            d_fwd = fwd1 - fwd0
            d_rows = rows1 - rows0
            mean_rows_per_fwd = (d_rows / d_fwd) if d_fwd else 0.0
            wsum = parse_wire_summary(stats_path)
            rec = {
                "arm": arm, "wire_mode": wire_mode, "threads": threads, "trees_per_thread": trees,
                "iter": it,
                "dps": float(r["dps"]), "dps_per_core": float(r["dps_per_core"]),
                "episodes": int(r["episodes"]), "decisions": int(r["decisions"]),
                "bench_wall": float(r["wall"]),   # the C++ bench's internal pass-loop budget wall
                "iter_wall": iter_wall,           # the harness-measured END-TO-END iteration wall
                "server_forwards": d_fwd, "server_rows": d_rows,
                "server_mean_rows_per_fwd": mean_rows_per_fwd,
                "wire_mean_rows_per_msg": wsum.get("mean_rows_per_msg", 0.0),
            }
            records.append(rec)
            print(f"[overcommit_sweep] {arm} {wire_mode} {threads}t N={trees} it={it}: "
                  f"srv_rows/fwd={mean_rows_per_fwd:.2f} dps={rec['dps']:.2f} "
                  f"dps/core={rec['dps_per_core']:.2f} "
                  f"wire_rows/msg={rec['wire_mean_rows_per_msg']:.2f} "
                  f"bench_wall={rec['bench_wall']:.2f}s iter_wall={iter_wall:.2f}s", flush=True)

    try:
        # Baseline arm: strict-barrier single tree (the §6 baseline). N is structurally 1 (strict ignores
        # trees_per_thread — production default untouched), but we run it at the 3-producer pinning too.
        one_cell("baseline-strict", "strict-barrier", a.threads, 1)
        # Overcommit arm: pipelined-bucket, sweeping N.
        for trees in trees_list:
            one_cell(f"pipelined-N{trees}", "pipelined-bucket", a.threads, trees)
    finally:
        server.stop()
        t.join(timeout=5.0)
        server.close()

    # ---- aggregate + report ----
    cell_keys = ["baseline-strict"] + [f"pipelined-N{n}" for n in trees_list]
    summary: dict = {
        "run": run, "secs": a.secs, "iters": a.iters, "hidden": a.hidden, "m": a.m,
        "n_sims": a.n_sims, "pool_batch": a.pool_batch, "inflight_msgs": a.inflight_msgs,
        "min_coalesce": a.min_coalesce,
        "min_forward_rows": a.min_forward_rows, "max_queue_delay_ms": a.max_queue_delay_ms,
        "threads": a.threads, "server_core": a.server_core, "producer_cores": a.producer_cores,
        "serve_fast_region_B": 192, "model_optimistic_dps": 456,
        "code_stamp": code_stamp(REPO),   # ADR-0011: pin this reading to the worktree's code state (DIRTY => not reproducible)
        "cells": {},
    }
    for arm in cell_keys:
        cell = [r for r in records if r["arm"] == arm]
        if not cell:
            continue
        summary["cells"][arm] = {
            "trees_per_thread": cell[0]["trees_per_thread"],
            "server_rows_per_forward": agg([r["server_mean_rows_per_fwd"] for r in cell]),
            "dps": agg([r["dps"] for r in cell]),
            "dps_per_core": agg([r["dps_per_core"] for r in cell]),
            "wire_rows_per_msg": agg([r["wire_mean_rows_per_msg"] for r in cell]),
            "bench_wall_s": agg([r["bench_wall"] for r in cell]),
            "iter_wall_s": agg([r["iter_wall"] for r in cell]),
            "cell_total_wall_s": sum(r["iter_wall"] for r in cell),
        }

    out_json = os.path.join(a.out, f"overcommit_sweep-{stamp}.json")
    with open(out_json, "w") as f:
        json.dump({"summary": summary, "records": records}, f, indent=2)

    _st = summary["code_stamp"]
    print("\n==== OVERCOMMIT INCREMENT (i) SWEEP — 1:3 pinning, greedy bucketed drain ====", flush=True)
    print(f"  [code: commit={_st['commit']} tree={_st['tree']}]  "
          f"(server fast region B~=192; model optimistic dps~=456; baseline = strict-barrier)\n",
          flush=True)
    hdr = (f"  {'cell':<18} {'N':>2}  {'rows/forward (mean+/-std [min-max])':<34}  "
           f"{'dps':<22}  {'dps/core':>8}  {'iter_wall(s)':>12}")
    print(hdr, flush=True)
    total_sweep_wall = 0.0
    for arm in cell_keys:
        c = summary["cells"].get(arm)
        if not c:
            continue
        rpf = c["server_rows_per_forward"]
        d = c["dps"]
        dpc = c["dps_per_core"]
        iw = c["iter_wall_s"]
        total_sweep_wall += c["cell_total_wall_s"]
        print(f"  {arm:<18} {c['trees_per_thread']:>2}  "
              f"{rpf['mean']:6.1f} +/- {rpf['std']:5.1f} [{rpf['min']:6.1f}-{rpf['max']:6.1f}]  "
              f"{d['mean']:6.1f} +/- {d['std']:5.1f} [{d['min']:5.1f}-{d['max']:5.1f}]  "
              f"{dpc['mean']:8.1f}  {iw['mean']:6.1f}+/-{iw['std']:4.1f}", flush=True)
    print(f"\n  total sweep wall: {total_sweep_wall:.1f}s "
          f"({total_sweep_wall/60.0:.1f} min) across {len(records)} timed iterations", flush=True)
    actual_total = time.perf_counter() - t_sweep0
    print(f"  ACTUAL end-to-end wall: {actual_total:.0f}s ({actual_total/60.0:.1f} min) "
          f"(expected ~={est_total:.0f}s)", flush=True)
    print(f"\n[overcommit_sweep] wrote {out_json}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
