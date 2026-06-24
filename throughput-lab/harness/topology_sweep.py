#!/usr/bin/env python3
"""
throughput-lab/harness/topology_sweep.py — run the process-topology config space (topology_enum.py's
configs.json) as an A/B throughput sweep, by COMPOSITION: each placement becomes a launch prefix
(`taskset -c <cpus>` + the right scheduling wrapper) around the EXISTING binaries — no recompiles, and
the scheduling policy is applied externally via sched_wrap (so the surplus's SCHED_IDLE and the server's
EEVDF --slice need no code change and no root). The enumerator is the single home of WHICH configs exist
(ADR-0012 P1); this driver only EXECUTES them and measures, so the two never disagree on the space.

PLACEMENT -> LAUNCH PREFIX (the one mapping that matters):
  taskset -c <cpus>  +
    SCHED_OTHER         -> (plain; `nice -n N` if nice != 0)
    SCHED_BATCH         -> sched_wrap --policy batch [--nice N] --
    SCHED_IDLE          -> sched_wrap --policy idle --
    SCHED_OTHER_LATNICE -> sched_wrap --policy other --slice <SLICE_NS> --   (kernel 6.19 has no
                           latency_nice field; the EEVDF custom time-slice is THE latency lever — a
                           smaller slice => prompter, finer-grained pickup at the same CPU share)

Each config: launch the server (its placement), wait READY; launch every generator+surplus as a
`tlab-real-producer --threads 1 --fibers K` process on its placement; let them run --seconds; sum the
per-process REAL-AGG leaves; leaf-rows/s = total_leaves / wall. Server torn down per config (a fresh
stat window + the per-config core/policy placement demands a relaunch). MEASURED, never assumed (ADR-0009).

Run:
  /home/bork/w/vdc/venvs/generic/bin/python throughput-lab/harness/topology_sweep.py \
      --configs /tmp/cfg.json --fibers 64 --seconds 5 --reps 1 --filter '' \
      --outdir ~/w/vdc/chocobo/runs/tlab/topo-sweep-XXXX

Public Domain (The Unlicense).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import signal
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/home/bork/w/vdc/1/chocofarm")
PYBIN = "/home/bork/w/vdc/venvs/generic/bin/python"
PRODUCER = ROOT / "throughput-lab/cpp/build/tlab-real-producer"
SCHED_WRAP = ROOT / "throughput-lab/cpp/build/sched_wrap"
INSTANCE = ROOT / "chocofarm/data/instance.json"
FACES = ROOT / "chocofarm/data/faces.json"
IN_DIM, N_ACTIONS, HIDDEN, MAX_BATCH = 241, 65, 256, 4096
_LEAVES_RE = re.compile(r"leaves=(\d+)\b")
_WALL_RE = re.compile(r"wall_s=([\d.]+)")


def launch_prefix(p: dict, slice_ns: int) -> list[str]:
    """The taskset + scheduling-wrapper argv prefix for one placement (everything BEFORE the real cmd).
    SCHED_OTHER_LATNICE maps to the EEVDF --slice (this kernel exposes no latency_nice — see header)."""
    pre = ["taskset", "-c", p["taskset"]]
    pol = p["policy"]
    if pol == "SCHED_OTHER":
        if p.get("nice"):
            pre += ["nice", "-n", str(p["nice"])]
    elif pol == "SCHED_OTHER_LATNICE":
        pre += [str(SCHED_WRAP), "--policy", "other", "--slice", str(slice_ns), "--"]
    elif pol == "SCHED_BATCH":
        pre += [str(SCHED_WRAP), "--policy", "batch"]
        if p.get("nice"):
            pre += ["--nice", str(p["nice"])]
        pre += ["--"]
    elif pol == "SCHED_IDLE":
        pre += [str(SCHED_WRAP), "--policy", "idle", "--"]
    else:
        raise ValueError(f"unmapped policy {pol!r}")
    return pre


def run_config(cfg: dict, *, fibers: int, seconds: float, n_sims: int, slice_ns: int, seq: int,
               logdir: Path) -> dict:
    placements = cfg["placements"]
    server = next(p for p in placements if p["role"] == "server")
    gens = [p for p in placements if p["role"] != "server"]

    endpoint = f"ipc:///tmp/tlab-topo-{os.getpid()}-{seq}.sock"
    sock = endpoint[len("ipc://"):]
    try:
        os.unlink(sock)
    except FileNotFoundError:
        pass

    slog = open(logdir / f"server-{cfg['config_id']}.log", "w")
    senv = {**os.environ, "PYTHONPATH": str(ROOT / "throughput-lab"), "PYTHONUNBUFFERED": "1"}
    server_cmd = launch_prefix(server, slice_ns) + [
        PYBIN, "-m", "server", "--bind", endpoint, "--in-dim", str(IN_DIM),
        "--n-actions", str(N_ACTIONS), "--hidden", str(HIDDEN), "--max-batch", str(MAX_BATCH),
        "--poll-timeout-ms", "50"]
    srv = subprocess.Popen(server_cmd, stdout=slog, stderr=subprocess.STDOUT, env=senv)
    logpath = logdir / f"server-{cfg['config_id']}.log"
    ready = False
    for _ in range(240):
        if srv.poll() is not None:
            break
        if logpath.read_text().find("READY") >= 0:
            ready = True
            break
        time.sleep(0.5)
    if not ready:
        srv.kill()
        return {"config_id": cfg["config_id"], "tag": cfg["tag"], "ok": False,
                "note": "server never READY", "leaves_per_sec": 0.0}

    procs = []
    for g in gens:
        cmd = launch_prefix(g, slice_ns) + [
            str(PRODUCER), "--instance", str(INSTANCE), "--faces", str(FACES),
            "--endpoint", endpoint, "--threads", "1", "--fibers", str(fibers),
            "--driver", "round-sync", "--seconds", str(seconds), "--n-sims", str(n_sims),
            "--in-dim", str(IN_DIM)]
        procs.append((g["role"], subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)))

    total_leaves, walls, fails = 0, [], []
    for role, pr in procs:
        out, _ = pr.communicate(timeout=seconds + 120)
        m, w = _LEAVES_RE.search(out or ""), _WALL_RE.search(out or "")
        if pr.returncode != 0 or not m:
            fails.append(f"{role}:rc={pr.returncode}")
            continue
        total_leaves += int(m.group(1))
        if w:
            walls.append(float(w.group(1)))

    if srv.poll() is None:
        srv.send_signal(signal.SIGINT)
        try:
            srv.wait(timeout=10)
        except subprocess.TimeoutExpired:
            srv.kill()
    try:
        os.unlink(sock)
    except FileNotFoundError:
        pass

    wall = max(walls) if walls else seconds
    lps = total_leaves / wall if wall > 0 else 0.0
    return {"config_id": cfg["config_id"], "tag": cfg["tag"], "ok": not fails,
            "note": ";".join(fails), "leaves_per_sec": lps, "total_leaves": total_leaves, "wall_s": wall}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--configs", required=True, help="configs.json from topology_enum.py")
    ap.add_argument("--fibers", type=int, default=64)
    ap.add_argument("--seconds", type=float, default=5.0)
    ap.add_argument("--n-sims", type=int, default=24)
    ap.add_argument("--slice-ns", type=int, default=300000, help="EEVDF slice for SCHED_OTHER_LATNICE (ns)")
    ap.add_argument("--reps", type=int, default=1, help="replicates per config (interleaved); median reported")
    ap.add_argument("--filter", default="", help="only configs whose tag contains this substring")
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    for b in (PRODUCER, SCHED_WRAP):
        if not b.exists():
            print(f"missing binary: {b}", file=sys.stderr)
            return 2
    outdir = Path(os.path.expanduser(args.outdir))
    outdir.mkdir(parents=True, exist_ok=True)

    configs = json.load(open(args.configs))["configs"]
    if args.filter:
        configs = [c for c in configs if args.filter in c["tag"]]
    print(f"running {len(configs)} configs x {args.reps} rep(s), fibers={args.fibers}, "
          f"seconds={args.seconds}", flush=True)

    samples: dict[str, list[float]] = {c["config_id"]: [] for c in configs}
    meta = {c["config_id"]: c for c in configs}
    seq = 0
    for rep in range(args.reps):
        for c in configs:
            seq += 1
            r = run_config(c, fibers=args.fibers, seconds=args.seconds, n_sims=args.n_sims,
                           slice_ns=args.slice_ns, seq=seq, logdir=outdir)
            if r["ok"]:
                samples[c["config_id"]].append(r["leaves_per_sec"])
            tag = r["tag"]
            print(f"  rep{rep} {c['config_id']:34} leaves/s={r['leaves_per_sec']:10.0f} "
                  f"{'' if r['ok'] else '[FAIL '+r['note']+']'}  {tag}", flush=True)

    rows = []
    for cid, xs in samples.items():
        med = statistics.median(xs) if xs else 0.0
        rows.append({"config_id": cid, "tag": meta[cid]["tag"], "leaves_per_sec_median": med,
                     "n": len(xs), "samples": xs})
    rows.sort(key=lambda r: -r["leaves_per_sec_median"])
    (outdir / "results.json").write_text(json.dumps(rows, indent=2))

    lines = ["# Topology sweep — leaf-rows/s by config (median)\n",
             f"fibers={args.fibers}, {args.seconds}s, n_sims={args.n_sims}, reps={args.reps}, "
             f"latnice slice={args.slice_ns}ns\n",
             "| rank | leaves/s | config_id | tag |", "| ---: | ---: | --- | --- |"]
    for i, r in enumerate(rows, 1):
        lines.append(f"| {i} | {r['leaves_per_sec_median']:,.0f} | {r['config_id']} | {r['tag']} |")
    (outdir / "REPORT.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\n-> {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
