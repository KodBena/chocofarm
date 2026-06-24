#!/usr/bin/env python3
"""
throughput-lab/harness/coalesce_sweep.py — the static coalescing-floor sweep: vary --msg-rows (how many
of a fiber round's parked leaves are coalesced into ONE message) over the REAL optimal config (server@0,
3 gens@1,2,3, SCHED_IDLE surplus@0, all coalescing at the cell's M), and report leaf-rows/s + the server's
forward count / mean batch / compute-busy per cell. Coalescing cuts the per-message + per-forward CPython
serve-path overhead (the profile's dominant cost); this maps the curve + the static optimum that any
adaptive (dynamic-control) gate would have to beat. Fresh server per cell (clean per-cell stats). ADR-0009.

Run:  OUTDIR=~/w/vdc/chocobo/runs/tlab/coalesce-XXXX \
      tools/shell/compute-watchdog.sh /home/bork/w/vdc/venvs/generic/bin/python \
      throughput-lab/harness/coalesce_sweep.py
Public Domain (The Unlicense).
"""
import json, os, re, signal, subprocess, sys, time
from pathlib import Path

ROOT = Path("/home/bork/w/vdc/1/chocofarm")
PY = "/home/bork/w/vdc/venvs/generic/bin/python"
PROD = str(ROOT / "throughput-lab/cpp/build/tlab-real-producer")
WRAP = str(ROOT / "throughput-lab/cpp/build/sched_wrap")
INST = str(ROOT / "chocofarm/data/instance.json"); FACES = str(ROOT / "chocofarm/data/faces.json")
OUT = Path(os.environ["OUTDIR"]); OUT.mkdir(parents=True, exist_ok=True)
EP = "ipc:///tmp/tlab-coalsweep.sock"
MSG_ROWS = [1, 4, 16, 64, 128, 256, 512]
K, SECONDS, NSIMS = 128, 6, 24
_LEAVES = re.compile(r"leaves=(\d+)\b")


def run_cell(m: int) -> dict:
    try: os.unlink(EP[len("ipc://"):])
    except FileNotFoundError: pass
    slog_path = OUT / f"server-m{m}.log"
    slog = open(slog_path, "w")
    srv = subprocess.Popen(["taskset", "-c", "0", PY, "-m", "server", "--bind", EP, "--in-dim", "241",
                            "--n-actions", "65", "--hidden", "256", "--max-batch", "4096", "--poll-timeout-ms", "50"],
                           stdout=slog, stderr=subprocess.STDOUT,
                           env={**os.environ, "PYTHONPATH": str(ROOT / "throughput-lab"), "PYTHONUNBUFFERED": "1"})
    for _ in range(240):
        if slog_path.read_text().find("READY") >= 0: break
        time.sleep(0.5)
    else:
        srv.kill(); raise SystemExit("server not READY")

    def gen(core, idle=False):
        pre = ["taskset", "-c", core] + ([WRAP, "--policy", "idle", "--"] if idle else [])
        return subprocess.Popen(pre + [PROD, "--instance", INST, "--faces", FACES, "--endpoint", EP,
                                "--threads", "1", "--fibers", str(K), "--msg-rows", str(m), "--driver",
                                "round-sync", "--seconds", str(SECONDS), "--n-sims", str(NSIMS)],
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    procs = [gen("1"), gen("2"), gen("3"), gen("0", idle=True)]
    leaves = 0
    for p in procs:
        o, _ = p.communicate(timeout=SECONDS + 120)
        mm = _LEAVES.search(o or "")
        if mm: leaves += int(mm.group(1))
    srv.send_signal(signal.SIGINT); time.sleep(1); srv.kill()
    txt = slog_path.read_text()
    def g(rx, d=0.0):
        mo = re.search(rx, txt); return float(mo.group(1)) if mo else d
    return {"msg_rows": m, "leaves_per_sec": leaves / SECONDS,
            "forwards": int(g(r"forwards: (\d+)")), "server_requests": int(g(r"served (\d+) requests")),
            "mean_batch": g(r"mean batch ([\d.]+)"), "compute_busy_pct": g(r"\(([\d.]+)% of wall\)")}


rows = [run_cell(m) for m in MSG_ROWS]
for r in rows:
    print(f"  msg_rows={r['msg_rows']:>4}  {r['leaves_per_sec']:>10,.0f} leaf-rows/s  "
          f"requests={r['server_requests']:>7}  forwards={r['forwards']:>5}  "
          f"mean_batch={r['mean_batch']:>6.1f}  srv_matmul={r['compute_busy_pct']:>4.0f}%", flush=True)
(OUT / "results.json").write_text(json.dumps(rows, indent=2))
base = rows[0]["leaves_per_sec"] or 1
lines = ["# Static coalescing-floor sweep (msg-rows) — real config, surplus on\n",
         f"server@0 + 3 gens@1,2,3 + SCHED_IDLE surplus@0, K={K}, {SECONDS}s, n_sims={NSIMS}, production build\n",
         "| msg-rows | leaf-rows/s | vs M=1 | server requests | forwards | mean batch | srv matmul% |",
         "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
for r in rows:
    lines.append(f"| {r['msg_rows']} | {r['leaves_per_sec']:,.0f} | {(r['leaves_per_sec']/base-1)*100:+.0f}% | "
                 f"{r['server_requests']:,} | {r['forwards']:,} | {r['mean_batch']:.0f} | {r['compute_busy_pct']:.0f}% |")
(OUT / "REPORT.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines)); print(f"\n-> {OUT}")
