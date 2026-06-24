#!/usr/bin/env python3
"""
throughput-lab/harness/k_idle_sweep.py — does the SCHED_IDLE-surplus win hold across the fiber count K?
The surplus_policy_control settled the POLICY (SCHED_IDLE +18% at K=64); this sweeps K to confirm the
benefit is K-stable, not a K=64 artifact. One server@core0; 3 generators@cores1,2,3; per K, run the
no-surplus baseline and the SCHED_IDLE-surplus@core0 variant (the run_real_best topology), interleaved
replicates, median/IQR + the IDLE delta per K. MEASURED (ADR-0009). Wrap me in compute-watchdog.sh.

Run:  OUTDIR=~/w/vdc/chocobo/runs/tlab/k-idle-XXXX \
      tools/shell/compute-watchdog.sh /home/bork/w/vdc/venvs/generic/bin/python \
      throughput-lab/harness/k_idle_sweep.py

Public Domain (The Unlicense).
"""
import json, os, re, signal, statistics, subprocess, sys, time
from pathlib import Path

ROOT = Path("/home/bork/w/vdc/1/chocofarm")
PY = "/home/bork/w/vdc/venvs/generic/bin/python"
PROD = str(ROOT / "throughput-lab/cpp/build/tlab-real-producer")
WRAP = str(ROOT / "throughput-lab/cpp/build/sched_wrap")
INST = str(ROOT / "chocofarm/data/instance.json")
FACES = str(ROOT / "chocofarm/data/faces.json")
OUT = Path(os.environ["OUTDIR"]); OUT.mkdir(parents=True, exist_ok=True)
EP = "ipc:///tmp/tlab-kidle.sock"
KS = [16, 64, 128, 256]
SECONDS, REPS, NSIMS = 5, 4, 24   # r0 warmup discarded -> 3 measured
_LEAVES = re.compile(r"leaves=(\d+)\b")


def gen(core, k, idle):
    pre = ["taskset", "-c", core] + ([WRAP, "--policy", "idle", "--"] if idle else [])
    return pre + [PROD, "--instance", INST, "--faces", FACES, "--endpoint", EP, "--threads", "1",
                  "--fibers", str(k), "--driver", "round-sync", "--seconds", str(SECONDS), "--n-sims", str(NSIMS)]


def run_variant(k, with_surplus):
    procs = [subprocess.Popen(gen(c, k, False), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
             for c in ("1", "2", "3")]
    if with_surplus:
        procs.append(subprocess.Popen(gen("0", k, True), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True))
    leaves = 0
    for p in procs:
        out, _ = p.communicate(timeout=SECONDS + 120)
        m = _LEAVES.search(out or "")
        if m:
            leaves += int(m.group(1))
    return leaves / SECONDS


try:
    os.unlink(EP[len("ipc://"):])
except FileNotFoundError:
    pass
slog = open(OUT / "server.log", "w")
srv = subprocess.Popen(["taskset", "-c", "0", PY, "-m", "server", "--bind", EP, "--in-dim", "241",
                        "--n-actions", "65", "--hidden", "256", "--max-batch", "4096", "--poll-timeout-ms", "50"],
                       stdout=slog, stderr=subprocess.STDOUT,
                       env={**os.environ, "PYTHONPATH": str(ROOT / "throughput-lab"), "PYTHONUNBUFFERED": "1"})
for _ in range(240):
    if (OUT / "server.log").read_text().find("READY") >= 0:
        break
    time.sleep(0.5)
else:
    print("server not READY", file=sys.stderr); srv.kill(); sys.exit(1)
print("server READY (core 0)", flush=True)

cells = [(k, s) for k in KS for s in (False, True)]
samples = {(k, s): [] for k, s in cells}
for r in range(REPS):
    for k, s in cells:
        lps = run_variant(k, s)
        if r > 0:
            samples[(k, s)].append(lps)
        print(f"  r{r}{'(warm)' if r == 0 else '':6} K={k:<4} surplus={'idle' if s else 'none':4} "
              f"leaves/s={lps:10.0f}", flush=True)
srv.send_signal(signal.SIGINT); time.sleep(1); srv.kill()


def med(xs):
    return statistics.median(xs) if xs else 0.0


lines = ["# SCHED_IDLE-surplus benefit across K (one server@0, 3 gens@1,2,3, surplus@0=idle)\n",
         f"{SECONDS}s, {REPS-1} measured reps, n_sims={NSIMS}\n",
         "| K | none leaves/s | +idle leaves/s | idle gain |", "| ---: | ---: | ---: | ---: |"]
for k in KS:
    n, i = med(samples[(k, False)]), med(samples[(k, True)])
    gain = (i / n - 1) * 100 if n else 0.0
    lines.append(f"| {k} | {n:,.0f} | {i:,.0f} | {gain:+.1f}% |")
(OUT / "REPORT.md").write_text("\n".join(lines) + "\n")
(OUT / "results.json").write_text(json.dumps({f"K{k}_{'idle' if s else 'none'}": v
                                              for (k, s), v in samples.items()}, indent=2))
print("\n".join(lines))
print(f"\n-> {OUT}")
