#!/usr/bin/env python3
"""
throughput-lab/harness/scenario_audit.py — a fixed, scriptable scenario toggle, so a surprising result
(here: the bottleneck regime flipping generator-bound <-> server-bound) is provably attributable to ONE
isolated variable and never lost to oblivion (the maintainer's audit-proof requirement; ADR-0009 measured
+ ADR-0011 mechanized). Holds EVERYTHING fixed (same Python server, same config, same probe) and toggles
exactly one axis — the producer/search build optimization (-march=native ON vs OFF) — then reports the
per-core utilization, throughput, and the classified regime side by side. If the regime flips when only
this flag flips, the flag IS the cause.

Extensible: add a scenario to SCENARIOS to bisect a different axis (n_sims, fibers, max_batch, ...).

Run:  /home/bork/w/vdc/venvs/generic/bin/python throughput-lab/harness/scenario_audit.py
      [--fibers 128 --seconds 10 --keep-builds]
Public Domain (The Unlicense).
"""
from __future__ import annotations

import argparse, json, os, re, signal, subprocess, sys, time
from pathlib import Path

ROOT = Path("/home/bork/w/vdc/1/chocofarm")
PY = "/home/bork/w/vdc/venvs/generic/bin/python"
INST = str(ROOT / "chocofarm/data/instance.json"); FACES = str(ROOT / "chocofarm/data/faces.json")

# The toggle axis: (scenario name) -> whether -march=native is ON. Everything else is held fixed.
SCENARIOS = [("native", True), ("plain", False)]


def _run(cmd, **kw):
    return subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, **kw)


def build_variant(native: bool) -> str:
    """Build chocofarm_core + tlab-real-producer with -march=native ON/OFF in dedicated dirs (cached).
    Returns the tlab-real-producer path. The native variant reuses the standard build dirs (which also
    hold the cap'd sched_wrap — never rebuilt here); the plain variant gets isolated -plain dirs and links
    the plain core (so the SEARCH itself is non-vectorized, not just the producer TUs)."""
    suffix = "" if native else "-plain"
    core_b = ROOT / f"cpp/build{suffix}"
    lab_b = ROOT / f"throughput-lab/cpp/build{suffix}"
    prod = lab_b / "tlab-real-producer"
    if prod.exists():
        return str(prod)
    on = "ON" if native else "OFF"
    print(f"  building {'native' if native else 'plain'} variant ...", flush=True)
    r = _run(["cmake", "-S", "cpp", "-B", str(core_b), "-DCMAKE_BUILD_TYPE=Release",
              f"-DCHOCO_MARCH_NATIVE={on}"])
    r = _run(["cmake", "--build", str(core_b), "--target", "chocofarm_core", "-j"])
    if not (core_b / "libchocofarm_core.a").exists():
        print(r.stdout[-2000:], r.stderr[-2000:]); raise SystemExit(f"core build failed ({core_b})")
    _run(["cmake", "-S", "throughput-lab/cpp", "-B", str(lab_b), "-DCMAKE_BUILD_TYPE=Release",
          "-DTLAB_REAL_GENERATOR=ON", f"-DTLAB_MARCH_NATIVE={on}",
          f"-DCHOCO_CORE_LIB={core_b}/libchocofarm_core.a"])
    r = _run(["cmake", "--build", str(lab_b), "--target", "tlab-real-producer", "-j"])
    if not prod.exists():
        print(r.stdout[-2000:], r.stderr[-2000:]); raise SystemExit(f"producer build failed ({lab_b})")
    return str(prod)


def _cpu_snap():
    out = {}
    for line in open("/proc/stat"):
        if line.startswith("cpu") and len(line) > 3 and line[3].isdigit():
            p = line.split(); v = list(map(int, p[1:])); out[p[0]] = (v[3] + v[4], sum(v))
    return out


def probe(producer: str, fibers: int, seconds: float) -> dict:
    """One server@0 + 3 gens@1,2,3 using `producer`; per-core util over a steady window + server stats.
    The Python server is identical across scenarios (only the C++ producer build changes)."""
    ep = "ipc:///tmp/tlab-audit.sock"
    try: os.unlink(ep[len("ipc://"):])
    except FileNotFoundError: pass
    log = "/tmp/tlab-audit-server.log"
    slog = open(log, "w")
    srv = subprocess.Popen(["taskset", "-c", "0", PY, "-m", "server", "--bind", ep, "--in-dim", "241",
                            "--n-actions", "65", "--hidden", "256", "--max-batch", "4096", "--poll-timeout-ms", "50"],
                           stdout=slog, stderr=subprocess.STDOUT,
                           env={**os.environ, "PYTHONPATH": str(ROOT / "throughput-lab"), "PYTHONUNBUFFERED": "1"})
    for _ in range(240):
        if Path(log).read_text().find("READY") >= 0: break
        time.sleep(0.5)
    else:
        srv.kill(); raise SystemExit("server not READY")

    def gen(core):
        return subprocess.Popen(["taskset", "-c", core, producer, "--instance", INST, "--faces", FACES,
                                 "--endpoint", ep, "--threads", "1", "--fibers", str(fibers), "--driver",
                                 "round-sync", "--seconds", str(seconds), "--n-sims", "24"],
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    procs = [gen("1"), gen("2"), gen("3")]
    time.sleep(3)
    s0 = _cpu_snap(); time.sleep(max(4.0, seconds - 5)); s1 = _cpu_snap()
    util = {}
    for cpu in sorted(s0):
        i0, t0 = s0[cpu]; i1, t1 = s1[cpu]
        util[cpu] = 100 * (1 - (i1 - i0) / (t1 - t0)) if t1 > t0 else 0.0
    leaves = 0
    for p in procs:
        o, _ = p.communicate(timeout=seconds + 120)
        m = re.search(r"leaves=(\d+)", o or "")
        if m: leaves += int(m.group(1))
    srv.send_signal(signal.SIGINT); time.sleep(1); srv.kill()
    txt = Path(log).read_text()
    busy = float(re.search(r"\(([\d.]+)% of wall\)", txt).group(1)) if re.search(r"% of wall", txt) else 0.0
    mb = float(re.search(r"mean batch ([\d.]+)", txt).group(1)) if re.search(r"mean batch", txt) else 0.0
    gen_avg = sum(util[f"cpu{c}"] for c in (1, 2, 3)) / 3
    regime = ("GENERATOR-bound (gen cores saturated, server fed slowly)" if gen_avg > 70 else
              "SERVER-bound (gens idle/reply-bound, server is the constraint)" if gen_avg < 50 else "MIXED")
    return {"util": util, "leaves_per_sec": leaves / seconds, "server_busy_pct": busy,
            "mean_batch": mb, "gen_core_avg": gen_avg, "regime": regime}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fibers", type=int, default=128)
    ap.add_argument("--seconds", type=float, default=11.0)
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()

    results = {}
    for name, native in SCENARIOS:
        print(f"=== scenario: {name} (-march=native {'ON' if native else 'OFF'}) ===", flush=True)
        prod = build_variant(native)
        results[name] = probe(prod, args.fibers, args.seconds)
        r = results[name]
        print(f"  cores: " + "  ".join(f"{c}={r['util'][f'cpu{c}']:.0f}%" for c in range(4)) +
              f"  | {r['leaves_per_sec']:,.0f} leaf-rows/s | srv_matmul {r['server_busy_pct']:.0f}% | "
              f"mean_batch {r['mean_batch']:.0f} | {r['regime']}", flush=True)

    print("\n############ AUDIT: only -march=native changed between these two ############")
    hdr = f"{'scenario':8} {'gen-core avg':>12} {'leaf-rows/s':>12} {'srv matmul%':>11} {'regime'}"
    print(hdr); print("-" * len(hdr))
    for name, _ in SCENARIOS:
        r = results[name]
        print(f"{name:8} {r['gen_core_avg']:>11.0f}% {r['leaves_per_sec']:>12,.0f} "
              f"{r['server_busy_pct']:>10.0f}% {r['regime']}")
    flipped = results["native"]["regime"].split()[0] != results["plain"]["regime"].split()[0]
    print(f"\nVERDICT: the regime {'FLIPS' if flipped else 'does NOT flip'} when only -march=native is "
          f"toggled -> -march=native {'IS' if flipped else 'is NOT'} the isolated cause.")
    if args.outdir:
        outdir = Path(os.path.expanduser(args.outdir)); outdir.mkdir(parents=True, exist_ok=True)
        (outdir / "audit.json").write_text(json.dumps(results, indent=2))
        print(f"-> {outdir}/audit.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
