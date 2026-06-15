#!/usr/bin/env python3
"""
cpp/parity/nmcs_parity.py — the AGGREGATE behavioral-parity harness for the C++ NMCS (ADR-0012 P6).

This is the Tier-2 check: run the C++ NMCS runner (`--policy nmcs`) and the Python NMCSPolicy over
matched-seed episodes and compare AGGREGATE statistics — mean episode length, mean λ-return (the
pure-MC ΣR − λ(ΣT+exit) at a fixed λ₀), the action-type distribution (mean collects / senses /
terminate), and mean belief-shrinkage — within Monte-Carlo CI, with the MC standard error REPORTED.
NMCS is a reimplementation with its OWN RNG (std::mt19937_64 != numpy), so the worlds differ
episode-for-episode by design; both are unbiased draws from the SAME prior world-set, so the
aggregates are comparable within MC CI. This is NOT byte-identity — it is the ADR-0012 P6 behavioral
bar (the SAME bar the RandomPolicy parity harness applies, here for the search policy).

NMCS is the slowest solver, so N is moderate (per ADR-0012's NMCS-cost note): a few-hundred episodes
across 2 seeds, level 1 (the default) — the milestone-relevant level-2 case is covered exactly by
the deterministic logic check (cpp/parity/nmcs_logic.py); running level-2 over hundreds of episodes
on both sides would be prohibitively slow for the aggregate tier without adding parity signal the
logic check does not already give exactly.

The Python reference mirrors the C++ runner's run_episode flow EXACTLY (same trailing-TERMINATE
record rule, same pure-MC suffix λ-return), so the two compare the SAME quantity on each side.

Run (from repo root):
    PYTHONPATH=. /home/bork/w/vdc/venvs/generic/bin/python cpp/parity/nmcs_parity.py

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
from chocofarm.solvers.nmcs import NMCSConfig, NMCSPolicy  # noqa: E402


def cpp_run_nmcs(run_id, version, cfg, lam, episodes, seed, max_steps):
    """Run the C++ runner with --policy nmcs (+ the NMCS knobs), collect its per-episode stats."""
    import json
    import subprocess
    import tempfile
    res_token = "nmcs-" + uuid.uuid4().hex[:12]
    stats_path = tempfile.mktemp(suffix=".jsonl")
    cmd = [P.CPP_BIN, "--instance", P.INSTANCE, "--faces", P.FACES, "--run", run_id,
           "--phase", "gen", "--version", str(version), "--episodes", str(episodes),
           "--lam", repr(lam), "--max-steps", str(max_steps), "--seed", str(seed),
           "--res-token", res_token, "--parity-stats", stats_path,
           "--policy", "nmcs", "--nmcs-level", str(cfg.level),
           "--nmcs-playouts", str(cfg.playout_samples), "--nmcs-step-samples", str(cfg.step_samples),
           "--nmcs-cand-det", str(cfg.cand_det), "--nmcs-cand-tre", str(cfg.cand_tre),
           "--nmcs-max-steps", str(cfg.max_steps)]
    out = subprocess.run(cmd, capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(f"C++ NMCS runner failed (rc={out.returncode}): {out.stderr}")
    rows = []
    with open(stats_path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    os.unlink(stats_path)
    return rows, res_token


def py_run_nmcs(env, cfg, lam, episodes, seed, max_steps):
    """Python NMCS over `episodes` matched-seed episodes, mirroring the C++ runner's run_episode flow
    (P.py_episode is the SAME record/return rule the C++ runner uses)."""
    policy = NMCSPolicy(cfg=cfg)
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
    max_steps = 24                   # NMCSConfig.max_steps (the search line cap; episodes terminate early)
    episodes = 150                   # moderate N (NMCS is the slowest solver — ADR-0012 cost note)
    seeds = [11, 23]                 # ≥2 seeds
    cfg = NMCSConfig(level=1, playout_samples=3, step_samples=2, cand_det=4, cand_tre=4, max_steps=24)
    keys = ["length", "lam_return", "n_collect", "n_sense", "n_terminate", "belief_shrinkage"]

    run_id = "nmcs-parity-" + uuid.uuid4().hex[:8]
    version = 0
    P.publish_weights(run_id, version)  # the weight-read seam needs a payload (NMCS ignores it)

    print("=== ADR-0012 P6 aggregate behavioral parity: C++ NMCS vs Python NMCS ===")
    print(f"instance: N={env.N} K={env.K} nDet={len(env.detectors)} |worlds|={len(env.worlds)}  "
          f"feat_dim={fd} n_slots={ns}")
    print(f"NMCS: level={cfg.level} playouts={cfg.playout_samples} step_samples={cfg.step_samples} "
          f"cand=({cfg.cand_det},{cfg.cand_tre}) search_max_steps={cfg.max_steps}")
    print(f"λ₀={lam} episode_max_steps={max_steps}  episodes/seed={episodes}  seeds={seeds}  "
          f"(N total per side = {episodes*len(seeds)})")
    print("RNG differs across the language boundary (std::mt19937_64 != numpy) — worlds differ "
          "episode-for-episode by design; aggregates compared within MC CI, NOT byte-identity.\n")

    py_rows_all, cpp_rows_all = [], []
    for seed in seeds:
        py_rows = py_run_nmcs(env, cfg, lam, episodes, seed, max_steps)
        cpp_rows, _res_token = cpp_run_nmcs(run_id, version, cfg, lam, episodes, seed, max_steps)
        py_rows_all += py_rows
        cpp_rows_all += cpp_rows
        print(f"[seed {seed}] ran {len(py_rows)} Python + {len(cpp_rows)} C++ NMCS episodes")

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
    print("     The exact NMCS nesting+selection logic is asserted separately, action-for-action, by")
    print("     cpp/parity/nmcs_logic.py (level-1 AND level-2). This tier is the RNG-driven aggregate.")
    print()
    if agg_ok:
        print("RESULT: PASS — every NMCS aggregate is indistinguishable within MC CI")
        return 0
    print("RESULT: FAIL — an NMCS aggregate diverged beyond 3·SE")
    return 1


if __name__ == "__main__":
    sys.exit(main())
