#!/usr/bin/env python3
"""Decisive control: does SCHED_IDLE specifically reclaim the server-core slack, or is ANY 4th worker
enough? ONE server on core 0; 3 generators on cores 1,2,3; vary the SURPLUS (4th worker, on core 0 with
the server): none / SCHED_IDLE / nice+19 / SCHED_BATCH. Interleaved replicates, median/IQR. Same
framework throughout (separate tlab-real-producer --threads 1 processes), so the ONLY difference is the
surplus policy -> isolates the policy effect that the cross-experiment comparison can't."""
import json, os, re, signal, statistics, subprocess, sys, time
from pathlib import Path

ROOT = Path("/home/bork/w/vdc/1/chocofarm")
PY = "/home/bork/w/vdc/venvs/generic/bin/python"
PROD = str(ROOT / "throughput-lab/cpp/build/tlab-real-producer")
WRAP = str(ROOT / "throughput-lab/cpp/build/sched_wrap")
INST = str(ROOT / "chocofarm/data/instance.json")
FACES = str(ROOT / "chocofarm/data/faces.json")
OUT = Path(os.environ["OUTDIR"]); OUT.mkdir(parents=True, exist_ok=True)
EP = "ipc:///tmp/tlab-surpctl.sock"
K, SECONDS, REPS, NSIMS = 64, 5, 6, 24   # r0 warmup discarded
_LEAVES = re.compile(r"leaves=(\d+)\b")

# surplus variants: label -> the taskset+wrapper prefix for the 4th worker on core 0 (None = no surplus)
VARIANTS = {
    "none":         None,
    "idle":         ["taskset", "-c", "0", WRAP, "--policy", "idle", "--"],
    "nice19":       ["taskset", "-c", "0", "nice", "-n", "19"],
    "batch":        ["taskset", "-c", "0", WRAP, "--policy", "batch", "--"],
}
def gen_cmd(prefix):
    return prefix + [PROD, "--instance", INST, "--faces", FACES, "--endpoint", EP,
                     "--threads", "1", "--fibers", str(K), "--driver", "round-sync",
                     "--seconds", str(SECONDS), "--n-sims", str(NSIMS)]

try: os.unlink(EP[len("ipc://"):])
except FileNotFoundError: pass
slog = open(OUT / "server.log", "w")
srv = subprocess.Popen(["taskset", "-c", "0", PY, "-m", "server", "--bind", EP, "--in-dim", "241",
                        "--n-actions", "65", "--hidden", "256", "--max-batch", "4096", "--poll-timeout-ms", "50"],
                       stdout=slog, stderr=subprocess.STDOUT,
                       env={**os.environ, "PYTHONPATH": str(ROOT / "throughput-lab"), "PYTHONUNBUFFERED": "1"})
for _ in range(240):
    if (OUT / "server.log").read_text().find("READY") >= 0: break
    time.sleep(0.5)
else:
    print("server not READY", file=sys.stderr); srv.kill(); sys.exit(1)
print("server READY (core 0)")

def run_variant(prefix):
    procs = []
    for c in ("1", "2", "3"):  # 3 base generators, one per isolated core
        procs.append(subprocess.Popen(gen_cmd(["taskset", "-c", c]),
                                      stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True))
    if prefix is not None:  # the 4th (surplus) worker on core 0, under its policy
        procs.append(subprocess.Popen(gen_cmd(prefix), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True))
    leaves = 0
    for p in procs:
        out, _ = p.communicate(timeout=SECONDS + 120)
        m = _LEAVES.search(out or "")
        if m: leaves += int(m.group(1))
    return leaves / SECONDS

samples = {k: [] for k in VARIANTS}
for r in range(REPS):
    for label, prefix in VARIANTS.items():
        lps = run_variant(prefix)
        if r > 0: samples[label].append(lps)
        print(f"  r{r}{'(warm)' if r==0 else '':6} {label:8} leaves/s={lps:10.0f}", flush=True)
srv.send_signal(signal.SIGINT); time.sleep(1); srv.kill()

def iqr(xs):
    xs = sorted(xs); n = len(xs)
    return (statistics.median(xs), xs[n//4], xs[(3*n)//4]) if n else (0, 0, 0)
base = statistics.median(samples["none"]) if samples["none"] else 1
lines = ["# Surplus-policy control (one server@0, 3 gens@1,2,3, surplus@0)\n",
         f"K={K}, {SECONDS}s, {REPS-1} measured reps, n_sims={NSIMS}\n",
         "| surplus policy | leaves/s median | IQR | vs no-surplus |", "| --- | ---: | --- | ---: |"]
for label in VARIANTS:
    med, lo, hi = iqr(samples[label])
    lines.append(f"| {label} | {med:,.0f} | {lo:,.0f}–{hi:,.0f} | {(med/base-1)*100:+.1f}% |")
(OUT / "REPORT.md").write_text("\n".join(lines) + "\n")
(OUT / "results.json").write_text(json.dumps(samples, indent=2))
print("\n".join(lines)); print(f"\n-> {OUT}")
