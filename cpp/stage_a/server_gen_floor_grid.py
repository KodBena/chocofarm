#!/usr/bin/env python3
"""
cpp/stage_a/server_gen_floor_grid.py — a THROWAWAY grid sweep of the SERVER-side coalescing floor
(inference_server min_forward_rows θ, increment ii) CROSSED with the GENERATION-side batch floor (the
runnable "final bolt": wire-ab-bench --gen-chunk-floor + --min-coalesce S_min). NOT a committed fixture,
NOT the production path (it only SUBCLASSES InferenceServer via StageAServer and drives the bench).

The question (server-floor-design.md negative result + the user's hunch): the server floor ALONE lowers
dps because the depth-1 producers idle during the accumulation wait. The gen-side chunk floor supplies
genuine in-flight DEPTH>1 (overcommit on the wire) so the producers DON'T idle — the server re-coalesces
the D×T small chunks into one large forward. Does the COMBINATION lift dps where each alone does not, and
is the server forward WIDTH (the user's "server batch too low") the binding axis?

Design (per the user's direction):
  * N=9 ONLY (the overcommit operating point), 3 producer threads (1:3 pin), pool_batch=64. ONE iteration
    per config (data generation is slow). NO strict-barrier arm (not needed).
  * Wakeup latency is NOT a lever here (not a latency-sensitive target): max_queue_delay is a fixed small
    anti-deadlock backstop (10 ms), not swept. The levers are BATCH sizes + the gen-floor boolean.
  * ONE long-lived server: max_batch=1024, a RICH AOT bucket set so the forward WIDTH floats up to where
    supply+θ push it (the design's "bigger is strictly better past 512"). θ (min_forward_rows) is read
    LIVE per-drain (ADR-0012 P4), so it is swept by setting the attribute — NO server restart, no recompile.

Axes (24 configs, intelligently allocated — a Sobol sub-grid on the meaningful ON space + a clean OFF line):
  * chunk_floor (BOOLEAN): the gen-side batch floor on/off.
  * θ_server   (min_forward_rows): the SERVER batch floor / forward-width target ∈ {0,128,256,384,512,768}.
  * S_min      (--min-coalesce): the GENERATION minimum batch size (rows per producer message when the floor
               binds) ∈ {16,32,64,128}. Inert when chunk_floor=0.
  * D          (--inflight-msgs): the per-thread in-flight message cap ∈ {4,8,16,32}. Inert when chunk_floor=0.
  OFF arm: 6 configs sweeping θ_server (the "server floor alone at N=9" reference; S_min/D inert).
  ON  arm: 18 configs, scrambled-Sobol over (θ_server, S_min, D) — the combined lever.

Per config: set θ live, run ONE wire-ab-bench pipelined pass, parse dps + the SERVER's mean rows/forward
(from its counters) + the wire mean rows/msg (the realized gen coalescing degree). The C++ producer's RSS
is sampled at 1 Hz with a hard ceiling (kills a runaway config — the chunk-flood convoy grew producer RSS,
e6d2c41); a wedge/OOM/timeout is recorded as a FAIL row, never aborts the grid. A table row is emitted +
appended to results.jsonl + table.md as each data point lands. Output under
~/w/vdc/chocobo/runs/server_gen_floor_grid/<stamp>/ (never /tmp).

Public Domain (The Unlicense).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
import uuid

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

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

from stage_a_server import StageAServer  # noqa: E402

AB_BENCH = os.path.join(REPO, "cpp", "build", "chocofarm-wire-ab-bench")
INSTANCE = os.path.join(REPO, "chocofarm", "data", "instance.json")
FACES = os.path.join(REPO, "chocofarm", "data", "faces.json")

# The server's AOT bucket set + cap — rich enough that a big accumulated batch lands on a compiled shape.
BUCKETS = (64, 128, 256, 384, 512, 768, 1024)
MAX_BATCH = 1024

# The swept levels.
THETA = [0, 128, 256, 384, 512, 768]   # server batch floor (min_forward_rows)
SMIN = [16, 32, 64, 128]               # generation minimum batch size (--min-coalesce)
DCAP = [4, 8, 16, 32]                  # per-thread in-flight message cap (--inflight-msgs)


def gen_configs(seed: int) -> list[dict]:
    """24 configs: a 6-point OFF reference line over θ (S_min/D inert) + an 18-point scrambled-Sobol ON
    arm over (θ, S_min, D). Sobol (a low-discrepancy quasi-random sequence) spreads the ON points evenly
    across the 3-axis box far better than a coarse full grid would at the same budget."""
    import warnings

    from scipy.stats import qmc
    off = [{"chunk_floor": 0, "theta": th, "s_min": 32, "d": 8} for th in THETA]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")   # Sobol balance-warning for n≠2^k is fine here (we want exactly 18)
        pts = qmc.Sobol(d=3, scramble=True, seed=seed).random(18)
    on = []
    for u in pts:
        th = THETA[min(len(THETA) - 1, int(u[0] * len(THETA)))]
        sm = SMIN[min(len(SMIN) - 1, int(u[1] * len(SMIN)))]
        dc = DCAP[min(len(DCAP) - 1, int(u[2] * len(DCAP)))]
        on.append({"chunk_floor": 1, "theta": th, "s_min": sm, "d": dc})
    # Interleave (1 OFF : 3 ON) so a --limit smoke covers BOTH arms and an interrupted run spans both.
    merged: list[dict] = []
    oi = ni = 0
    while oi < len(off) or ni < len(on):
        if oi < len(off):
            merged.append(off[oi]); oi += 1
        for _ in range(3):
            if ni < len(on):
                merged.append(on[ni]); ni += 1
    return merged


def refine_configs() -> list[dict]:
    """Curated CONFIRMATION set around the grid's winning region (gen=ON, θ_server=0) + the two relevant
    baselines, run at --iters for variance (ADR-0009: a single-iter dps gap needs MIN-beats-MAX before a
    winner is claimed). θ_server>0 is dropped (the grid showed it neutral-to-harmful); the live lever is
    the gen-side (S_min, D)."""
    cfgs = [
        {"chunk_floor": 0, "theta": 0, "s_min": 32, "d": 8},     # greedy baseline (the number to beat)
        {"chunk_floor": 0, "theta": 128, "s_min": 32, "d": 8},   # OFF-arm best (server floor alone)
    ]
    for sm in (16, 32, 64):
        for dc in (8, 16, 32):
            cfgs.append({"chunk_floor": 1, "theta": 0, "s_min": sm, "d": dc})  # the winning region
    return cfgs


def build_and_publish(hidden: int, run: str, version: int):
    env = Environment()
    in_dim, n_actions = feature_dim(env), n_action_slots(env)
    net = ValueMLP(in_dim, hidden=hidden, n_actions=n_actions, seed=17,
                   y_mean=0.0, y_std=1.0, residual=False)
    manifest, blob = pack_net(net)
    RedisTransport(connect()).publish_weights(net, phase="gen", version=version, run=run)
    params, y_mean, y_std = params_from_manifest_blob(manifest, blob)
    return StaticParamsSource(params, y_mean, y_std), in_dim, n_actions


def start_server(src, endpoint: str, server_core: int, delay_ms: float):
    """ONE StageAServer (bucket+group), max_batch=1024 + rich buckets, θ set live per config, a fixed small
    max_queue_delay backstop. Pinned to server_core from inside its thread (1:3)."""
    server = StageAServer(src, bind=endpoint, max_batch=MAX_BATCH, forward_fn=jit_forward_core,
                          e_policy="bucket", wakeup="group", buckets=BUCKETS,
                          min_forward_rows=0, max_queue_delay_ms=delay_ms)
    server.warmup(sorted(BUCKETS))

    def _serve():
        try:
            os.sched_setaffinity(0, {server_core})
        except (OSError, AttributeError):
            pass
        server.serve_forever()

    t = threading.Thread(target=_serve, daemon=True)
    t.start()
    return server, t


def _rss_kib(pid: int) -> int:
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1])
    except OSError:
        pass
    return 0


def run_one(cfg: dict, server, endpoint: str, run: str, version: int, threads: int, trees: int,
            pool_batch: int, secs: float, producer_cores: str, rss_ceiling_kib: int,
            stats_path: str, rss_path: str, timeout_s: float) -> dict:
    """Run ONE wire-ab-bench pipelined pass for `cfg`. Samples the producer subprocess RSS at 1 Hz to
    `rss_path` and KILLS it if RSS exceeds the ceiling (a runaway chunk-flood convoy). Returns a metrics
    dict (status=OK|FAIL_*). Never raises — a bad config is a FAIL row, not a grid abort."""
    server._min_forward_rows = int(cfg["theta"])   # live θ (ADR-0012 P4 — read per-drain, no restart)
    tok = f"sgf-{uuid.uuid4().hex[:8]}"
    cmd = [
        "taskset", "-c", producer_cores, AB_BENCH,
        "--instance", INSTANCE, "--faces", FACES, "--endpoint", endpoint,
        "--run", run, "--version", str(version), "--res-token", tok,
        "--wire-mode", "pipelined-bucket", "--secs", str(secs),
        "--m", "24", "--n-sims", "256",
        "--pool-threads", str(threads), "--pool-batch", str(pool_batch),
        "--inflight-msgs", str(cfg["d"]), "--trees-per-thread", str(trees),
        "--min-coalesce", str(cfg["s_min"]), "--gen-chunk-floor", str(cfg["chunk_floor"]),
        "--parity-stats", stats_path,
    ]
    fwd0, rows0 = server.n_forwards, server.n_real_rows
    peak = {"kib": 0, "killed": False}
    t0 = time.perf_counter()
    proc = subprocess.Popen(cmd, cwd=REPO, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    def _sample():
        with open(rss_path, "a") as rf:
            while proc.poll() is None:
                k = _rss_kib(proc.pid)
                if k > peak["kib"]:
                    peak["kib"] = k
                rf.write(f"{int(time.time())} {k}\n")
                if rss_ceiling_kib and k > rss_ceiling_kib:
                    peak["killed"] = True
                    proc.kill()
                    break
                time.sleep(1.0)

    sampler = threading.Thread(target=_sample, daemon=True)
    sampler.start()
    status, out = "OK", ""
    try:
        out, _ = proc.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        proc.kill()
        out, _ = proc.communicate()
        status = "FAIL_TIMEOUT"
    sampler.join(timeout=2.0)
    wall = time.perf_counter() - t0
    if peak["killed"]:
        status = "FAIL_RSS"
    elif proc.returncode != 0 and status == "OK":
        status = f"FAIL_RC{proc.returncode}"

    res: dict = {}
    for line in out.splitlines():
        if line.startswith("RESULT:"):
            for t in line.split():
                if "=" in t:
                    k, v = t.split("=", 1)
                    res[k] = v
    fwd1, rows1 = server.n_forwards, server.n_real_rows
    d_fwd, d_rows = fwd1 - fwd0, rows1 - rows0
    srv_rpf = (d_rows / d_fwd) if d_fwd else 0.0

    wire_rpm = 0.0
    try:
        with open(stats_path) as f:
            for line in f:
                obj = json.loads(line)
                if obj.get("wire_summary"):
                    wire_rpm = float(obj.get("mean_rows_per_msg", 0.0))
    except (OSError, json.JSONDecodeError):
        pass

    return {
        **cfg,
        "status": status,
        "dps": float(res.get("dps", 0.0)) if status == "OK" else 0.0,
        "dps_per_core": float(res.get("dps_per_core", 0.0)) if status == "OK" else 0.0,
        "srv_rows_per_fwd": srv_rpf,
        "wire_rows_per_msg": wire_rpm,
        "rss_peak_mib": peak["kib"] / 1024.0,
        "wall_s": wall,
        "raw_tail": "" if status == "OK" else out[-600:],
    }


def _row(rec: dict) -> str:
    cf = "ON " if rec["chunk_floor"] else "off"
    return (f"| {cf} | {rec['theta']:>4} | {rec['s_min']:>4} | {rec['d']:>3} | "
            f"{rec['srv_rows_per_fwd']:>7.1f} | {rec['wire_rows_per_msg']:>7.1f} | "
            f"{rec['dps']:>7.1f} | {rec['dps_per_core']:>6.1f} | {rec['rss_peak_mib']:>6.0f} | "
            f"{rec['wall_s']:>5.0f} | {rec['status']} |")


HDR = ("| gen | θsrv | Smin |  D  | srv r/fwd | wire r/msg |   dps  | dps/c | RSS MiB | wall | status |\n"
       "|-----|------|------|-----|-----------|------------|--------|-------|---------|------|--------|")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--secs", type=float, default=4.0)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--trees", type=int, default=9)        # N=9 only (the user's directive)
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--pool-batch", type=int, default=64)
    ap.add_argument("--delay-ms", type=float, default=2.0)   # small anti-deadlock backstop (NOT a lever):
    # deep overcommit supply reaches θ on the non-blocking drain (delay irrelevant); shallow supply fires
    # small/fast (low score) instead of throttling every forward to 1/delay and timing out.
    ap.add_argument("--server-core", type=int, default=0)
    ap.add_argument("--producer-cores", default="1,2,3")
    ap.add_argument("--rss-ceiling-gib", type=float, default=6.0)
    ap.add_argument("--timeout-s", type=float, default=300.0)   # good N=9 config ~120s; catches wedges
    ap.add_argument("--sobol-seed", type=int, default=12345)
    ap.add_argument("--limit", type=int, default=0)   # >0: run only the first N configs (smoke)
    ap.add_argument("--iters", type=int, default=1)   # repeat each config (variance; aggregated row)
    ap.add_argument("--refine", action="store_true")  # curated confirmation set around the grid winner
    ap.add_argument("--out", default=os.path.join(os.path.expanduser("~"), "w", "vdc", "chocobo",
                                                  "runs", "server_gen_floor_grid"))
    a = ap.parse_args()

    stamp = time.strftime("%Y%m%d-%H%M%S")
    out = os.path.join(a.out, stamp)
    os.makedirs(out, exist_ok=True)
    run = f"sgf-{stamp}"
    version = 0
    endpoint = f"ipc:///tmp/choco-sgf-{os.getpid()}.ipc"
    rss_ceiling_kib = int(a.rss_ceiling_gib * 1024 * 1024)

    configs = refine_configs() if a.refine else gen_configs(a.sobol_seed)
    if a.limit > 0:
        configs = configs[:a.limit]
    print(f"[grid] {len(configs)} configs (N={a.trees}, {a.iters} iter(s) each) → {out}", flush=True)

    src, in_dim, n_actions = build_and_publish(a.hidden, run, version)
    server, t = start_server(src, endpoint, a.server_core, a.delay_ms)
    print(f"[grid] server up max_batch={MAX_BATCH} buckets={BUCKETS} delay={a.delay_ms}ms "
          f"in_dim={in_dim} n_actions={n_actions} endpoint={endpoint}", flush=True)

    results_path = os.path.join(out, "results.jsonl")
    table_path = os.path.join(out, "table.md")
    with open(table_path, "w") as f:
        f.write(f"# server-floor × gen-floor grid (N={a.trees}, 1 iter, secs={a.secs})\n\n{HDR}\n")
    print("\n" + HDR, flush=True)

    records: list[dict] = []
    try:
        for i, cfg in enumerate(configs):
            iters: list[dict] = []
            for it in range(max(1, a.iters)):
                stats_path = os.path.join(out, f"wirestats-{i:02d}-{it}.jsonl")
                rss_path = os.path.join(out, f"rss-{i:02d}-{it}.log")
                rec = run_one(cfg, server, endpoint, run, version, a.threads, a.trees, a.pool_batch,
                              a.secs, a.producer_cores, rss_ceiling_kib, stats_path, rss_path, a.timeout_s)
                iters.append(rec)
                with open(results_path, "a") as f:
                    f.write(json.dumps({**rec, "iter": it}) + "\n")
            # Aggregate the config's iters into ONE row: dps mean + [min-max] spread (the MIN-beats-MAX bar).
            oks = [r for r in iters if r["status"] == "OK"]
            base = oks[-1] if oks else iters[-1]
            dpss = [r["dps"] for r in oks] or [0.0]
            agg = {**base,
                   "dps": sum(dpss) / len(dpss),
                   "dps_min": min(dpss), "dps_max": max(dpss),
                   "srv_rows_per_fwd": sum(r["srv_rows_per_fwd"] for r in oks) / len(oks) if oks else 0.0,
                   "rss_peak_mib": max(r["rss_peak_mib"] for r in iters),
                   "wall_s": sum(r["wall_s"] for r in iters),
                   "status": (f"OK n={len(oks)} [{min(dpss):.0f}-{max(dpss):.0f}]" if oks
                              else iters[-1]["status"])}
            records.append(agg)
            line = _row(agg)
            print(line, flush=True)
            with open(table_path, "a") as f:
                f.write(line + "\n")
    finally:
        server.stop()
        t.join(timeout=5.0)
        server.close()

    summary = {
        "run": run, "stamp": stamp, "trees": a.trees, "threads": a.threads, "pool_batch": a.pool_batch,
        "secs": a.secs, "delay_ms": a.delay_ms, "max_batch": MAX_BATCH, "buckets": list(BUCKETS),
        "theta_levels": THETA, "smin_levels": SMIN, "d_levels": DCAP, "sobol_seed": a.sobol_seed,
        "records": records,
    }
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    ok = [r for r in records if r["status"] == "OK"]
    print(f"\n[grid] DONE {len(ok)}/{len(records)} OK. wrote {out}", flush=True)
    if ok:
        best = max(ok, key=lambda r: r["dps"])
        print(f"[grid] best dps={best['dps']:.1f} @ gen={'ON' if best['chunk_floor'] else 'off'} "
              f"θ={best['theta']} S_min={best['s_min']} D={best['d']} "
              f"(srv rows/fwd={best['srv_rows_per_fwd']:.1f})", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
