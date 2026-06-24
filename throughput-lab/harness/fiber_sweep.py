#!/usr/bin/env python3
"""Robustness sweep: fiber-vs-non-fiber leaf-rate, INTERLEAVED replicates + median/IQR, K up to 128,
both drivers. Warmup replicate (r=0) discarded. One server for the whole sweep. Results -> runs/."""
import json, os, re, signal, statistics, subprocess, sys, time
from pathlib import Path

PYBIN = "/home/bork/w/vdc/venvs/generic/bin/python"
ROOT = Path("/home/bork/w/vdc/1/chocofarm")
PROD = ROOT / "throughput-lab/cpp/build/tlab-real-producer"
EP = "ipc:///tmp/tlab-robust.sock"
OUT = Path(os.environ["OUTDIR"])
OUT.mkdir(parents=True, exist_ok=True)
SECONDS, REPS_TOTAL, NSIMS = 4, 6, 24   # 6 runs/cell; r0 = warmup discarded -> 5 measured

# cells: (label, fibers, driver)
CELLS = [("non-fiber", 0, "round-sync")]
for drv in ("round-sync", "greedy"):
    for k in (1, 4, 16, 64, 128):
        CELLS.append((f"{drv}-K{k}", k, drv))

try: os.unlink(EP[len("ipc://"):])
except FileNotFoundError: pass
slog = open(OUT / "server.log", "w")
srv = subprocess.Popen(
    [PYBIN, "-m", "server", "--bind", EP, "--in-dim", "241", "--n-actions", "65",
     "--hidden", "256", "--max-batch", "4096", "--poll-timeout-ms", "50"],
    stdout=slog, stderr=subprocess.STDOUT,
    env={**os.environ, "PYTHONPATH": str(ROOT / "throughput-lab"), "PYTHONUNBUFFERED": "1"},
    preexec_fn=lambda: os.sched_setaffinity(0, {0}))
# wait READY by tailing the log
ready = False
for _ in range(240):
    if (OUT / "server.log").read_text().find("READY") >= 0: ready = True; break
    time.sleep(0.5)
if not ready:
    print("server failed to start", file=sys.stderr); srv.kill(); sys.exit(1)
print("server READY")

RE = re.compile(r"leaves_per_sec=([\d.]+)")
samples = {c[0]: [] for c in CELLS}   # label -> [leaves/s] (measured reps only)
for r in range(REPS_TOTAL):
    warm = (r == 0)
    for label, k, drv in CELLS:
        cmd = ["taskset", "-c", "1,2,3", str(PROD),
               "--instance", str(ROOT / "chocofarm/data/instance.json"),
               "--faces", str(ROOT / "chocofarm/data/faces.json"),
               "--endpoint", EP, "--threads", "3", "--fibers", str(k),
               "--driver", drv, "--seconds", str(SECONDS), "--n-sims", str(NSIMS)]
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=SECONDS + 60)
        m = RE.search(out.stdout)
        lps = float(m.group(1)) if m else float("nan")
        if not warm: samples[label].append(lps)
        print(f"  r{r}{'(warmup)' if warm else '':9} {label:18} leaves/s={lps:10.0f}", flush=True)

srv.send_signal(signal.SIGINT); time.sleep(1.0); srv.kill()

def stats(xs):
    xs = sorted(x for x in xs if x == x)
    if not xs: return (0, 0, 0)
    med = statistics.median(xs)
    q1 = statistics.median(xs[: len(xs) // 2]) if len(xs) > 1 else xs[0]
    q3 = statistics.median(xs[(len(xs) + 1) // 2:]) if len(xs) > 1 else xs[0]
    return (med, q1, q3)

rows = []
for label, k, drv in CELLS:
    med, q1, q3 = stats(samples[label])
    rows.append({"cell": label, "fibers": k, "driver": drv if k else "n/a",
                 "leaves_per_sec_median": med, "iqr_lo": q1, "iqr_hi": q3,
                 "samples": samples[label]})
(OUT / "results.json").write_text(json.dumps(rows, indent=2))

lines = ["# Fiber robustness sweep (interleaved replicates, median/IQR)\n",
         f"3 threads (cores 1,2,3), server core 0, {SECONDS}s/run, n_sims={NSIMS}, "
         f"{REPS_TOTAL-1} measured replicates (r0 warmup discarded).\n",
         "| cell | fibers | driver | leaves/s median | IQR (p25–p75) |",
         "| --- | ---: | --- | ---: | --- |"]
for row in rows:
    lines.append(f"| {row['cell']} | {row['fibers']} | {row['driver']} | "
                 f"{row['leaves_per_sec_median']:,.0f} | "
                 f"{row['iqr_lo']:,.0f}–{row['iqr_hi']:,.0f} |")
(OUT / "REPORT.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
print(f"\n-> {OUT}")
