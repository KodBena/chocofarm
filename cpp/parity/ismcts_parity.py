#!/usr/bin/env python3
"""
cpp/parity/ismcts_parity.py — the AGGREGATE behavioral-parity harness for the C++ ISMCTS (ADR-0012 P6).

This is the Tier-2 check: run the C++ ISMCTS runner (`--policy ismcts`) and the Python ISMCTSPolicy
over matched-seed episodes and compare AGGREGATE statistics — mean episode length, mean λ-return (the
pure-MC ΣR − λ(ΣT+exit) at a fixed λ₀), the action-type distribution (mean collects / senses /
terminate), and mean belief-shrinkage — within Monte-Carlo CI, with the MC standard error REPORTED.
ISMCTS is a reimplementation with its OWN RNG (std::mt19937_64 != numpy), and it determinizes a world
per iteration AND draws an expansion index per expansion, so the search trees differ episode-for-
episode by design; both are unbiased determinizations from the SAME prior world-set, so the
aggregates are comparable within MC CI. This is NOT byte-identity — it is the ADR-0012 P6 behavioral
bar (the SAME bar the RandomPolicy / NMCS parity harnesses apply, here for ISMCTS).

The exact ISMCTS selection+nesting logic — the part that MUST be exact — is asserted separately,
action-for-action, by the DETERMINISTIC logic check (cpp/parity/ismcts_logic.py), independent of RNG.
This tier is the RNG-driven aggregate. ISMCTS runs many iterations per decision, so N is moderate
(per ADR-0012's search-cost note): a few-hundred episodes across 2 seeds at a modest iteration count.

The Python reference mirrors the C++ runner's run_episode flow EXACTLY (same trailing-TERMINATE
record rule, same pure-MC suffix λ-return), so the two compare the SAME quantity on each side.

Run (from repo root):
    PYTHONPATH=. /home/bork/w/vdc/venvs/generic/bin/python cpp/parity/ismcts_parity.py

Public Domain (The Unlicense).
"""
from __future__ import annotations

import os
import sys
import uuid

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

# reuse the RandomPolicy harness's plumbing (the Python episode flow, the cpp runner shell, the
# MC-SE comparison, the weight publish) — one home for the shared parity scaffolding.
from cpp.parity import parity as P  # noqa: E402
from chocofarm.az.features import feature_dim  # noqa: E402
from chocofarm.az.actions import n_action_slots  # noqa: E402
from chocofarm.model.env import Environment  # noqa: E402
from chocofarm.solvers.ismcts import ISMCTSConfig, ISMCTSPolicy  # noqa: E402


def cpp_run_ismcts(run_id, version, cfg, lam, episodes, seed, max_steps):
    """Run the C++ runner with --policy ismcts (+ the ISMCTS knobs), collect its per-episode stats."""
    import json
    import subprocess
    import tempfile
    res_token = "ismcts-" + uuid.uuid4().hex[:12]
    stats_path = tempfile.mktemp(suffix=".jsonl")
    cmd = [P.CPP_BIN, "--instance", P.INSTANCE, "--faces", P.FACES, "--run", run_id,
           "--phase", "gen", "--version", str(version), "--episodes", str(episodes),
           "--lam", repr(lam), "--max-steps", str(max_steps), "--seed", str(seed),
           "--res-token", res_token, "--parity-stats", stats_path,
           "--policy", "ismcts", "--ismcts-iterations", str(cfg.iterations),
           "--ismcts-c", repr(cfg.c), "--ismcts-max-depth", str(cfg.max_depth)]
    out = subprocess.run(cmd, capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(f"C++ ISMCTS runner failed (rc={out.returncode}): {out.stderr}")
    rows = []
    with open(stats_path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    os.unlink(stats_path)
    return rows, res_token


def py_run_ismcts(env, cfg, lam, episodes, seed, max_steps):
    """Python ISMCTS over `episodes` matched-seed episodes, mirroring the C++ runner's run_episode
    flow (P.py_episode is the SAME record/return rule the C++ runner uses)."""
    policy = ISMCTSPolicy(cfg=cfg)
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(episodes):
        world = int(rng.choice(env.worlds))
        rows.append(P.py_episode(env, policy, world, lam, rng, max_steps))
    return rows


def main():
    if not os.path.exists(P.CPP_BIN):
        print(f"FAIL: C++ binary not built at {P.CPP_BIN}\n"
              f"      build it: cmake -S cpp -B cpp/build && cmake --build cpp/build")
        return 1

    env = Environment()
    fd = feature_dim(env)
    ns = n_action_slots(env)
    lam = 0.10                       # fixed λ₀ for the λ-return comparison
    max_steps = 24                   # episode horizon (episodes terminate early at TERMINATE)
    episodes = 120                   # moderate N (ISMCTS runs many iterations per decision)
    seeds = [11, 23]                 # ≥2 seeds
    # a modest iteration count keeps the aggregate tier affordable; the EXACT selection logic is
    # covered at the full default iterations=300 by the deterministic logic check.
    cfg = ISMCTSConfig(iterations=80, c=0.7, max_depth=24)
    keys = ["length", "lam_return", "n_collect", "n_sense", "n_terminate", "belief_shrinkage"]

    run_id = "ismcts-parity-" + uuid.uuid4().hex[:8]
    version = 0
    P.publish_weights(run_id, version)  # the weight-read seam needs a payload (ISMCTS ignores it)

    print("=== ADR-0012 P6 aggregate behavioral parity: C++ ISMCTS vs Python ISMCTS ===")
    print(f"instance: N={env.N} K={env.K} nDet={len(env.detectors)} |worlds|={len(env.worlds)}  "
          f"feat_dim={fd} n_slots={ns}")
    print(f"ISMCTS: iterations={cfg.iterations} c={cfg.c} max_depth={cfg.max_depth}")
    print(f"λ₀={lam} episode_max_steps={max_steps}  episodes/seed={episodes}  seeds={seeds}  "
          f"(N total per side = {episodes*len(seeds)})")
    print("RNG differs across the language boundary (std::mt19937_64 != numpy) — worlds + expansion "
          "draws differ episode-for-episode by design; aggregates compared within MC CI, NOT "
          "byte-identity.\n")

    py_rows_all, cpp_rows_all = [], []
    for seed in seeds:
        py_rows = py_run_ismcts(env, cfg, lam, episodes, seed, max_steps)
        cpp_rows, _res_token = cpp_run_ismcts(run_id, version, cfg, lam, episodes, seed, max_steps)
        py_rows_all += py_rows
        cpp_rows_all += cpp_rows
        print(f"[seed {seed}] ran {len(py_rows)} Python + {len(cpp_rows)} C++ ISMCTS episodes")

    res = P.compare(py_rows_all, cpp_rows_all, keys)
    agg_ok = True
    print()
    print(f"{'stat':<18}{'py mean±SE':<26}{'cpp mean±SE':<26}{'Δ':<14}{'SE_comb':<12}{'|z|':<8}verdict")
    for k in keys:
        r = res[k]
        py = f"{r['py_mean']:.5f}±{r['py_se']:.5f}"
        cp = f"{r['cpp_mean']:.5f}±{r['cpp_se']:.5f}"
        verdict = "OK" if r["z"] < 3.0 else "DIVERGE"
        if r["z"] >= 3.0:
            agg_ok = False
        print(f"{k:<18}{py:<26}{cp:<26}{r['diff']:<+14.5f}{r['se_combined']:<12.5f}"
              f"{r['z']:<8.2f}{verdict}")

    print()
    print("Bar: |z| = |Δ| / SE_combined < 3.0 (≈99.7% two-sided MC band) for every aggregate.")
    print("     The exact ISMCTS selection+nesting logic is asserted separately, action-for-action, by")
    print("     cpp/parity/ismcts_logic.py (RNG-free). This tier is the RNG-driven aggregate.")
    print()
    if agg_ok:
        print("RESULT: PASS — every ISMCTS aggregate is indistinguishable within MC CI")
        return 0
    print("RESULT: FAIL — an ISMCTS aggregate diverged beyond 3·SE")
    return 1


if __name__ == "__main__":
    sys.exit(main())
