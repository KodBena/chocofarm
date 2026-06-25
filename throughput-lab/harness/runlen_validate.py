#!/usr/bin/env python3
"""
throughput-lab/harness/runlen_validate.py — VALIDATE the --seconds (run-length) effect at the banked
  operating point. Tests whether throughput genuinely depends on run length, and — crucially — WHICH
  metric moves, so we don't repeat the mistake of attributing a measurement artifact to a physical cause.

DESIGN (artifact-proof):
  - FRESH server per (rep, seconds) point — each point is an independent S-second burst from a clean
    server (a persistent server would accumulate runtime and confound the very effect we test).
  - Sweep --seconds, INTERLEAVED: the seconds order is ROTATED each rep so slow box-drift cannot
    masquerade as a run-length trend (robust-benchmark-statistics discipline).
  - PRIMARY metric = per-forward COMPUTE time (compute_busy / forwards): server-internal, no denominator,
    no head-time. If it RISES with --seconds, forwards genuinely slow down as the burst lengthens (a real
    server/stack effect). If it is FLAT while rates drop, the rate drop is a denominator/head-time artifact
    (the server's wall starts before the producer connects, penalizing short runs) — NOT a physical effect.
  - Secondary: server rows/wall (head-time-biased AGAINST short runs) and producer leaves/wall.
  NB mechanism, if real, is in the server/XLA/host-frequency-governor path — NOT thermal (the box has
  overspec cooling; that hypothesis is closed).

Reuses topology_sweep.run_config (the validated launch composition; it sets cwd=ROOT) at the SSOT banked
topology (hp/spec.banked_topology_config_id() -> harness.topology_enum.config_by_id). The producer/server
OPERATING POINT is also derived from the hp SSOT (hp/spec.banked_static()) — one home, NOT hardcoded; only
--seconds is swept. Writes results.json under ~/w/vdc; recording to tlab_reading/tlab_finding is done
AFTER, from the JSON (measurement ⊥ recording). RUN HANDS-OFF / QUIET BOX.

Run:  PYTHONPATH=throughput-lab:throughput-lab/harness python <thisfile> [SECONDS_CSV] [REPS]
      (defaults: 5,10,14,20  and  6 reps)

Public Domain (The Unlicense).
"""
from __future__ import annotations
import json
import re
import statistics as st
import sys
import time
from pathlib import Path

ROOT = Path("/home/bork/w/vdc/1/chocofarm")
sys.path.insert(0, str(ROOT / "throughput-lab"))
sys.path.insert(0, str(ROOT / "throughput-lab/harness"))
import topology_sweep as ts          # noqa: E402  the validated launch composition
import topology_enum as te           # noqa: E402  the SSOT topology resolver
from hp import spec                  # noqa: E402  the banked topology config_id + banked static point
from code_stamp import code_stamp, code_stamp_str  # noqa: E402

# Default seconds = the STABLE band only. s>=30 hits a PRODUCER instability (rc=-9 SIGKILL / rc=1, and even
# successful 30s runs collapse to ~half throughput) — NOT memory (RAM is fine); characterized separately.
# Pass a CSV first arg to override (e.g. "5,10,14,20,30" to probe the instability). More reps for resolution.
def _banked_params() -> dict:
    """The banked PRODUCER/SERVER operating point as run_config kwargs, derived from the hp SSOT
    (hp/spec.banked_static()) — one home, not a hardcoded copy. Everything is fixed at the banked point
    EXCEPT --seconds, which this harness sweeps (so banked seconds is intentionally dropped here).
    Maps the registry key names onto run_config's parameter names (inflight_msgs->inflight,
    warmup_ladder->the --warmup CSV string)."""
    b = spec.banked_static()
    return dict(
        fibers=int(b["fibers"]), n_sims=int(b["n_sims"]), msg_rows=int(b["msg_rows"]),
        inflight=int(b["inflight_msgs"]), driver=str(b["driver"]), m=int(b["m"]),
        max_batch=int(b["max_batch"]), warmup=",".join(str(x) for x in b["warmup_ladder"]),
        episodic=True, single_thread=True, slice_ns=300000,
    )


_SERVED = re.compile(r"served (\d+) requests / (\d+) rows in ([\d.]+)s")
_COMPUTE = re.compile(r"compute-busy:\s+([\d.]+)s")            # NB the colon (the bug that zeroed per_fwd)
_FORWARDS = re.compile(r"forwards:\s+(\d+)")
_PERFWD = re.compile(r"per-forward \(us\):.*?\bcompute (\d+)")  # DIRECT per-forward compute (us), cleanest


def parse_server(logpath: Path) -> "dict | None":
    if not logpath.exists():
        return None
    t = logpath.read_text()
    m = _SERVED.search(t)
    if not m:
        return None
    rows, wall = int(m.group(2)), float(m.group(3))
    cb = _COMPUTE.search(t)
    fwd = _FORWARDS.search(t)
    pf = _PERFWD.search(t)
    compute_s = float(cb.group(1)) if cb else None
    forwards = int(fwd.group(1)) if fwd else None
    per_fwd = (float(pf.group(1)) if pf else
               (compute_s / forwards * 1e6 if (compute_s and forwards) else None))
    return {
        "server_rows": rows, "server_wall": wall, "server_rows_s": rows / wall if wall else None,
        "compute_s": compute_s, "forwards": forwards, "per_fwd_compute_us": per_fwd,
    }


def med(xs):
    """Median over the non-None entries (pure helper for the by-seconds summary)."""
    xs = [x for x in xs if x is not None]
    return st.median(xs) if xs else None


def main() -> None:
    """The executable body — kept under the __main__ guard so `import runlen_validate` is side-effect-free
    (no SSOT resolution, no mkdir, no run_config / server / producer spawn on import; ADR-0002 — a repo
    harness must be import-safe)."""
    # Default seconds = the STABLE band only (s>=30 hits a producer instability; see header). Pass a CSV
    # first arg to override; more reps for resolution.
    SECONDS = [int(x) for x in (sys.argv[1].split(",") if len(sys.argv) > 1 else ["5", "10", "14", "20"])]
    REPS = int(sys.argv[2]) if len(sys.argv) > 2 else 6

    PARAMS = _banked_params()
    cfg = te.config_by_id(spec.banked_topology_config_id()).to_record()

    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    OUTDIR = Path.home() / f"w/vdc/chocobo/runs/tlab/runlen-{stamp}"
    OUTDIR.mkdir(parents=True, exist_ok=True)

    print(f"=== run-length validation @ banked point [{code_stamp_str()}] -> {OUTDIR} ===")
    print(f"    config={cfg['config_id']}  seconds={SECONDS}  reps={REPS}  (interleaved, rotated order)")

    seq = 0
    rows: list[dict] = []
    for rep in range(REPS):
        k = rep % len(SECONDS)
        order = SECONDS[k:] + SECONDS[:k]                       # rotate each rep -> interleave
        for S in order:
            seq += 1
            ld = OUTDIR / f"r{rep}_s{S}"
            ld.mkdir(exist_ok=True)
            r = ts.run_config(cfg, seconds=float(S), seq=seq, logdir=ld, **PARAMS)
            srv = parse_server(ld / f"server-{cfg['config_id']}.log")
            srate = srv["server_rows_s"] if srv else None
            pfwd = srv["per_fwd_compute_us"] if srv else None
            prod = r.get("leaves_per_sec") if r.get("ok") else None
            rows.append({"rep": rep, "seconds": S, "ok": bool(r.get("ok")),
                         "server_rows_s": srate, "producer_leaves_s": prod, "per_fwd_compute_us": pfwd,
                         **(srv or {})})
            print(f"  rep{rep} s={S:2d}  per_fwd={(pfwd or 0):6.1f}us  server={(srate or 0):9.0f} rows/s  "
                  f"producer={(prod or 0):9.0f}  {'' if r.get('ok') else '[FAIL '+r.get('note','')+']'}",
                  flush=True)

    # --- analysis: median by seconds; trend on the PRIMARY (per-forward compute) + the rates -----------
    by_s = {}
    for S in SECONDS:
        sub = [r for r in rows if r["seconds"] == S and r["ok"]]        # only stable runs into the medians
        fails = sum(1 for r in rows if r["seconds"] == S and not r["ok"])
        by_s[S] = {"n": len(sub), "fails": fails,
                   "per_fwd_us": med([r["per_fwd_compute_us"] for r in sub]),
                   "server_rows_s": med([r["server_rows_s"] for r in sub]),
                   "producer_leaves_s": med([r["producer_leaves_s"] for r in sub])}

    print("\n=== SUMMARY (median by --seconds; failed runs EXCLUDED from medians) ===")
    print(f"  {'secs':>4} {'per_fwd_us':>11} {'server_rows_s':>14} {'producer_l/s':>13} {'n':>3} {'fail':>4}")
    for S in SECONDS:
        b = by_s[S]
        print(f"  {S:>4} {(b['per_fwd_us'] or 0):>11.1f} {(b['server_rows_s'] or 0):>14.0f} "
              f"{(b['producer_leaves_s'] or 0):>13.0f} {b['n']:>3} {b['fails']:>4}")

    smin, smax = min(SECONDS), max(SECONDS)
    pf0, pf1 = by_s[smin]["per_fwd_us"], by_s[smax]["per_fwd_us"]
    sr0, sr1 = by_s[smin]["server_rows_s"], by_s[smax]["server_rows_s"]
    verdict = "INCONCLUSIVE (missing data)"
    if pf0 and pf1 and sr0 and sr1:
        d_pf = 100 * (pf1 - pf0) / pf0          # + => forwards slow down with run length
        d_sr = 100 * (sr1 - sr0) / sr0          # - => longer runs measure slower
        print(f"\n  per-forward COMPUTE {smin}s -> {smax}s: {d_pf:+.1f}%  (PRIMARY; + = forwards genuinely slow down)")
        print(f"  server rows/s        {smin}s -> {smax}s: {d_sr:+.1f}%  (head-time-biased; reads alongside)")
        if d_pf > 2:
            verdict = (f"RUN-LENGTH EFFECT CONFIRMED, PHYSICAL: per-forward compute rises {d_pf:+.1f}% over "
                       f"{smin}->{smax}s. The forwards genuinely slow as the burst lengthens -- a server/XLA/"
                       f"host-frequency-governor effect (NOT thermal). Next: per-second forward-latency trace to localize.")
        elif d_sr < -2:
            verdict = (f"RATE drops {d_sr:+.1f}% but per-forward compute is FLAT ({d_pf:+.1f}%): the run-length "
                       f"difference is a DENOMINATOR/head-time ARTIFACT, not a physical slowdown. Fix = consistent "
                       f"denominator + fixed --seconds; there is no real effect to chase.")
        else:
            verdict = (f"NO run-length effect (per-forward {d_pf:+.1f}%, rate {d_sr:+.1f}%). The --seconds 'confound' "
                       f"is NOT the cause of the original gap -- reopen the RCA.")
    print(f"\n  VERDICT: {verdict}")

    (OUTDIR / "results.json").write_text(json.dumps({
        "code_stamp": code_stamp(), "config_id": cfg["config_id"], "seconds": SECONDS, "reps": REPS,
        "params": PARAMS, "rows": rows, "summary": {str(k): v for k, v in by_s.items()},
        "verdict": verdict,
    }, indent=2))
    print(f"\n-> {OUTDIR}/results.json\n=== DONE ===")


if __name__ == "__main__":
    main()
