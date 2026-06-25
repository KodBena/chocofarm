#!/usr/bin/env python3
"""
throughput-lab/harness/topology_pair.py — PAIRED topology test: a CHALLENGER topology vs the banked
  INCUMBENT, the infpair.sh design generalized to two full process-topologies. The 40-config screen's
  top cluster is a ~1% dead heat and rank-1's lead is consistent with max-of-40 selection noise; this
  resolves "does the best alternative actually beat the bank?" with the drift-cancelling paired
  permutation test (the sharp tool for a sub-1% effect), defeating the selection bias.

  INCUMBENT  the banked process-topology — resolved from the hp SSOT (hp/spec.banked_topology_config_id();
             currently s2p1_g0.0-1.0-3.0_u2p0 = server-isolated@2 + gens@0,1,3 + surplus@2 IDLE). NOT
             pinned here. CHALLENGER defaults to the prior banked s0p1_g1.0-2.0-3.0_u0p0 (server@0
             housekeeping + gens@1,2,3 + surplus@0 IDLE) but is overridable as argv[3].

The ONE difference between the canonical pair is which core hosts the server+surplus pair — the noisy
housekeeping core 0 vs an isolated core. The producer/server OPERATING POINT (K / msg-rows / max-batch /
driver / inflight / n-sims / m / warmup / seconds) is the banked static point, also resolved from the hp
SSOT (hp/spec.banked_static()), NOT hand-pinned — one home, override args still win.

Reuses topology_sweep.run_config (the VALIDATED placement->launch composition — no bash re-impl; it sets
cwd=ROOT, so this is cwd-independent). Order of the two arms is ALTERNATED per rep to balance within-rep
position against drift; the per-rep difference cancels slow box drift. Two-sided sign-flip permutation +
one-sided (H1: challenger>incumbent) + sign test + bootstrap 95%CI on the median paired diff.
RUN HANDS-OFF / QUIET BOX (no concurrent Claude) — finding #18.

Run:  PYTHONPATH=throughput-lab:throughput-lab/harness <py> <thisfile> [REPS] [CONFIGS_JSON] [CHALLENGER_ID]

Public Domain (The Unlicense).
"""
from __future__ import annotations
import json
import random
import statistics as st
import sys
import time
from pathlib import Path

ROOT = Path("/home/bork/w/vdc/1/chocofarm")
sys.path.insert(0, str(ROOT / "throughput-lab"))
sys.path.insert(0, str(ROOT / "throughput-lab/harness"))
import topology_sweep as ts          # noqa: E402  -- run_config + launch composition (single home)
import topology_enum as te           # noqa: E402  -- the SSOT topology resolver (config_by_id)
from hp import spec                  # noqa: E402  -- the banked topology + banked static operating point
from code_stamp import code_stamp, code_stamp_str   # noqa: E402

def _banked_params() -> dict:
    """The banked PRODUCER/SERVER operating point as run_config kwargs, derived from the hp SSOT
    (hp/spec.banked_static()) — one home, not a hardcoded copy. Maps the registry key names onto
    run_config's parameter names (inflight_msgs->inflight, warmup_ladder->the --warmup CSV string).
    single_thread is a SERVER flag (banked finding #5), orthogonal to the producer HP; slice_ns is the
    sched-policy slice for SCHED_IDLE surplus placements."""
    b = spec.banked_static()
    return dict(
        fibers=int(b["fibers"]), seconds=float(b["seconds"]), n_sims=int(b["n_sims"]),
        msg_rows=int(b["msg_rows"]), inflight=int(b["inflight_msgs"]), driver=str(b["driver"]),
        m=int(b["m"]), max_batch=int(b["max_batch"]),
        warmup=",".join(str(x) for x in b["warmup_ladder"]),
        slice_ns=300000, episodic=True, single_thread=True,
    )


def main() -> None:
    """The executable body — kept under the __main__ guard so `import topology_pair` is side-effect-free
    (no SSOT resolution, no mkdir, no run_config / server / producer spawn on import; ADR-0002 — a repo
    harness must be import-safe)."""
    # INCUMBENT = the banked topology, resolved from the hp SSOT (NOT pinned). CHALLENGER defaults to the
    # prior banked server-on-core-0 topology; override as argv[3] to test a different alternative.
    INCUMBENT  = spec.banked_topology_config_id()
    CHALLENGER = sys.argv[3] if len(sys.argv) > 3 else "s0p1_g1.0-2.0-3.0_u0p0"
    REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 24
    CONFIGS = Path(sys.argv[2]) if len(sys.argv) > 2 else \
        Path.home() / "w/vdc/chocobo/runs/tlab/topo-screen-20260625T073508Z/configs.json"

    PARAMS = _banked_params()

    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    OUTDIR = Path.home() / f"w/vdc/chocobo/runs/tlab/topo-pair-{stamp}"
    OUTDIR.mkdir(parents=True, exist_ok=True)

    cfgs = {c["config_id"]: c for c in json.load(open(CONFIGS))["configs"]}
    for cid in (CHALLENGER, INCUMBENT):
        if cid not in cfgs:                              # ADR-0002: a missing config is a loud error
            sys.exit(f"config {cid!r} not in {CONFIGS} -- regenerate the enum")
    cfg_chal, cfg_inc = cfgs[CHALLENGER], cfgs[INCUMBENT]

    print(f"=== PAIRED topology: CHALLENGER {CHALLENGER} vs INCUMBENT {INCUMBENT} (banked) ===")
    print(f"    {REPS} reps, {PARAMS['seconds']}s/arm, banked point [{code_stamp_str()}] -> {OUTDIR}")
    print(f"    challenger: {cfg_chal['tag']}")
    print(f"    incumbent : {cfg_inc['tag']}")

    seq = 0
    def measure(cfg) -> "float | None":
        """One arm: stand up this config's server+producers, return leaf-rows/s (None on failure)."""
        nonlocal seq
        seq += 1
        r = ts.run_config(cfg, seq=seq, logdir=OUTDIR, **PARAMS)
        if not r["ok"]:
            print(f"    [FAIL {cfg['config_id']}: {r['note']}]", flush=True)
            return None
        return r["leaves_per_sec"]

    data_path = OUTDIR / "pairs.txt"
    pairs: list[tuple[float, float]] = []   # (challenger, incumbent)
    with open(data_path, "w") as fh:
        for rep in range(1, REPS + 1):
            if rep % 2 == 0:                # alternate which arm goes first (position balance)
                inc = measure(cfg_inc); chal = measure(cfg_chal)
            else:
                chal = measure(cfg_chal); inc = measure(cfg_inc)
            if chal is None or inc is None:
                print(f"rep{rep:02d}  DROPPED (an arm failed)", flush=True)
                continue
            pairs.append((chal, inc))
            fh.write(f"{rep} {chal:.0f} {inc:.0f}\n"); fh.flush()
            print(f"rep{rep:02d}  chal={chal:9.0f}  inc={inc:9.0f}  diff(chal-inc)={chal-inc:+8.0f}", flush=True)

    n = len(pairs)
    if n < 3:
        sys.exit(f"only {n} usable pairs -- too few to test")

    chal = [a for a, _ in pairs]; inc = [b for _, b in pairs]
    diffs = [a - b for a, b in pairs]
    obs = st.mean(diffs); med = st.median(diffs)
    pos = sum(d > 0 for d in diffs); neg = sum(d < 0 for d in diffs); tie = sum(d == 0 for d in diffs)

    rng = random.Random(0)
    N = 50000; ge_two = 0; ge_one = 0
    for _ in range(N):
        m = sum((d if rng.random() < 0.5 else -d) for d in diffs) / n
        if abs(m) >= abs(obs): ge_two += 1
        if m >= obs: ge_one += 1
    p_two = (ge_two + 1) / (N + 1)
    p_one = (ge_one + 1) / (N + 1)   # H1: challenger > incumbent

    B = 20000; bs = []
    for _ in range(B):
        s = [rng.choice(diffs) for _ in range(n)]; bs.append(st.median(s))
    bs.sort(); lo, hi = bs[int(.025 * B)], bs[int(.975 * B)]

    inc_med = st.median(inc)
    report = [
        "",
        "=== ANALYSIS (paired nonparametric) ===",
        f"n={n} usable paired reps  [{code_stamp_str()}]",
        f"  challenger median = {st.median(chal):.0f}   incumbent median = {inc_med:.0f}   leaf-rows/s",
        f"  paired diff (chal-inc): median={med:+.0f}  mean={obs:+.1f}  "
        f"({100*med/inc_med:+.2f}% of incumbent)  bootstrap95%CI(median)=[{lo:+.0f},{hi:+.0f}]",
        f"  sign test: challenger>incumbent in {pos}/{n}  (incumbent> in {neg}, ties {tie})",
        f"  permutation: two-sided p={p_two:.4f} | one-sided (H1 chal>inc) p={p_one:.4f}  over {N} perms",
        "  VERDICT: " + (
            f"CHALLENGER beats incumbent (one-sided p={p_one:.3f}<0.05) -- consider re-bank to the challenger"
            if p_one < 0.05 else
            f"NO establishable difference (two-sided p={p_two:.3f}) -- keep incumbent; topology is null in the top cluster"),
        "=== DONE ===",
    ]
    print("\n".join(report))
    (OUTDIR / "VERDICT.txt").write_text("\n".join(report) + "\n")
    (OUTDIR / "result.json").write_text(json.dumps({
        "code_stamp": code_stamp(), "challenger": CHALLENGER, "incumbent": INCUMBENT,
        "n": n, "challenger_median": st.median(chal), "incumbent_median": inc_med,
        "median_diff": med, "mean_diff": obs, "ci_median": [lo, hi],
        "p_two_sided": p_two, "p_one_sided": p_one, "sign_pos": pos, "sign_neg": neg,
        "pairs": pairs, "params": PARAMS,
    }, indent=2))
    print(f"-> {OUTDIR}")


if __name__ == "__main__":
    main()
