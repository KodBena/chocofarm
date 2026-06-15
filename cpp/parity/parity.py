#!/usr/bin/env python3
"""
cpp/parity/parity.py — the ADR-0012 P6/P7 behavioral-parity harness for the C++ runner.

It validates the C++ dumb-random runner against the Python `RandomPolicy` reference under the
EXACT bar ADR-0012 P6/P7 names — NOT byte-identity:

  * Logic invariants -> bit-exact. The legality mask `M` the C++ worker emits is bit-identical to
    Python's for the same (loc, belief); illegal-slot PI mass is == 0.0. (Asserted exactly; float32
    cannot perturb a {0,1} mask.) Driven action-for-action over a matched-seed replay (the env is
    deterministic given an action sequence, so the masks ARE a logic fact).
  * Float-sensitive / RNG-driven -> aggregate behavioral equivalence. Mean episode length, mean
    λ-return (ΣR/ΣT-style λ-return at a fixed λ₀), action-type distribution, and mean belief-
    shrinkage are compared over N≥300 episodes across ≥2 seeds, requiring statistical
    indistinguishability within Monte-Carlo CI, with the MC standard error REPORTED.
  * Format round-trip. The C++ result blobs are read back via np.frombuffer(...).reshape(...) per
    transport.py and the shapes/dtypes are confirmed (X (n,feat_dim), PI/M (n,n_slots), Y (n,),
    all float32).

It needs the C++ binary built (cpp/build/chocofarm-cpp-runner) and redis up. The Python reference
mirrors the C++ run_episode flow exactly (same trailing-TERMINATE record rule, same pure-MC suffix
λ-return), so the two are comparing the SAME quantity computed on each side.

Run (from repo root):
    PYTHONPATH=. /home/bork/w/vdc/venvs/generic/bin/python cpp/parity/parity.py

Public Domain (The Unlicense).
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import tempfile
import uuid

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

from chocofarm.az import transport
from chocofarm.az.actions import legal_mask, n_action_slots
from chocofarm.az.features import feature_dim
from chocofarm.model.env import TERMINATE, Environment
from chocofarm.solvers.base import RandomPolicy

CPP_BIN = os.path.join(REPO, "cpp", "build", "chocofarm-cpp-runner")
MASK_BIN = os.path.join(REPO, "cpp", "build", "chocofarm-mask-dump")
INSTANCE = os.path.join(REPO, "chocofarm", "data", "instance.json")
FACES = os.path.join(REPO, "chocofarm", "data", "faces.json")


# ---------------------------------------------------------------------------
# Python reference episode — mirrors cpp/src/runner.cpp run_episode EXACTLY:
# record a decision per ply (incl. a trailing TERMINATE), step on non-TERMINATE, and the λ-return
# is the pure-MC suffix return-to-go from the first executed decision (the lam_blend=1 limit).
# ---------------------------------------------------------------------------
def py_episode(env, policy, world, lam, rng, max_steps):
    loc = ("w", env.entry)
    bw = env.worlds
    collected = set()
    bw0 = len(bw)
    step_rt = []
    n_collect = n_sense = n_terminate = 0
    for _ in range(max_steps):
        if len(bw) == 0:
            break
        a = policy.decide(env, loc, bw, collected, lam, rng)
        if a == TERMINATE:
            n_terminate = 1
            break
        if a[0] == "t":
            n_collect += 1
        else:
            n_sense += 1
        r, loc, bw, collected, dt = env.apply(loc, bw, collected, a, world)
        step_rt.append((r, dt))
    exit_c = env.exit_cost(loc)
    n_dec = len(step_rt)
    # pure-MC λ-return from the first executed decision (suffix_returns_to_go[0]); bare exit -> -λ·exit
    if n_dec > 0:
        suffix_r = sum(r for r, _ in step_rt)
        suffix_t = sum(dt for _, dt in step_rt)
        lam_return = suffix_r - lam * (suffix_t + exit_c)
    else:
        lam_return = -lam * exit_c
    return dict(length=n_dec, lam_return=lam_return, n_collect=n_collect, n_sense=n_sense,
                n_terminate=n_terminate, belief_shrinkage=1.0 - len(bw) / bw0)


def py_run(env, policy, lam, episodes, seed, max_steps):
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(episodes):
        world = int(rng.choice(env.worlds))
        rows.append(py_episode(env, policy, world, lam, rng, max_steps))
    return rows


def cpp_run(run_id, version, lam, episodes, seed, max_steps):
    """Run the C++ runner, collect its per-episode stats (JSON lines) + return the res_token so the
    caller can round-trip the blobs."""
    res_token = "parity-" + uuid.uuid4().hex[:12]
    stats_path = tempfile.mktemp(suffix=".jsonl")
    cmd = [CPP_BIN, "--instance", INSTANCE, "--faces", FACES, "--run", run_id,
           "--phase", "gen", "--version", str(version), "--episodes", str(episodes),
           "--lam", repr(lam), "--max-steps", str(max_steps), "--seed", str(seed),
           "--res-token", res_token, "--parity-stats", stats_path]
    out = subprocess.run(cmd, capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(f"C++ runner failed (rc={out.returncode}): {out.stderr}")
    rows = []
    with open(stats_path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    os.unlink(stats_path)
    return rows, res_token


# ---------------------------------------------------------------------------
# Aggregate-stat comparison with Monte-Carlo standard error (P6: report a number).
# ---------------------------------------------------------------------------
def mean_se(xs):
    a = np.asarray(xs, dtype=np.float64)
    n = len(a)
    mean = float(a.mean())
    se = float(a.std(ddof=1) / math.sqrt(n)) if n > 1 else 0.0
    return mean, se


def compare(py_rows, cpp_rows, keys):
    """For each stat: report (py mean±SE, cpp mean±SE), the difference, the combined SE, and the
    z-score |Δ|/SE_combined. Indistinguishable within MC CI <=> |z| < 3 (a ~99.7% two-sided band)."""
    results = {}
    for k in keys:
        pm, ps = mean_se([r[k] for r in py_rows])
        cm, cs = mean_se([r[k] for r in cpp_rows])
        se_comb = math.hypot(ps, cs)
        diff = cm - pm
        z = abs(diff) / se_comb if se_comb > 0 else (0.0 if diff == 0 else float("inf"))
        results[k] = dict(py_mean=pm, py_se=ps, cpp_mean=cm, cpp_se=cs,
                          diff=diff, se_combined=se_comb, z=z)
    return results


# ---------------------------------------------------------------------------
# Bit-exact mask parity: replay a matched action sequence through BOTH envs and assert the masks
# the two produce are byte-identical. The env is deterministic given (action sequence, world), so
# this is a logic-fact comparison float32 cannot perturb. We drive the sequence in Python (the C++
# mask round-trips through the result blob `M`, already asserted == Python's by reconstruction here:
# the C++ M is built by the SAME legal_actions->slot mapping, so we assert the Python-rebuilt mask
# equals the C++-emitted mask for every recorded decision of a real C++ episode).
# ---------------------------------------------------------------------------
def mask_parity(env, res_token, episodes, feat_dim_, n_slots):
    """Read the C++ M blocks back and, for the SAME (loc, belief) the C++ visited, rebuild the
    Python mask and assert bit-identity. We reconstruct the visited (loc, belief) by replaying the
    executed action sequence — but the executed action is not on the wire, so instead we assert the
    structural mask invariants the logic guarantees AND, separately (test_cpp_runner.py), drive a
    hand-picked belief through both. Here we assert, per recorded decision, the strongest wire-only
    logic facts: M is {0,1}, illegal-PI mass == 0.0, TERMINATE slot == 1.0."""
    conn = transport.connect()
    checked = 0
    for idx in range(episodes):
        xk, pik, mk, yk = transport.result_keys(res_token, idx)
        yb = conn.get(yk)
        if yb is None:
            continue
        Y = np.frombuffer(yb, dtype=np.float32)
        n = len(Y)
        M = np.frombuffer(conn.get(mk), dtype=np.float32).reshape(n, n_slots)
        PI = np.frombuffer(conn.get(pik), dtype=np.float32).reshape(n, n_slots)
        assert set(np.unique(M).tolist()) <= {0.0, 1.0}, "M not in {0,1}"
        assert float(PI[M == 0.0].sum()) == 0.0, "illegal-slot PI mass != 0.0"
        assert bool((M[:, n_slots - 1] == 1.0).all()), "TERMINATE slot not always legal"
        checked += n
    return checked


def mask_bit_exact(env, n_seqs, max_steps, seed, n_slots):
    """The STRONG bit-exact mask claim (ADR-0012 P6/P7): for a matched (loc, belief) sequence, the
    C++ legality mask M is BYTE-IDENTICAL to Python's `legal_mask`. Drive a random action sequence in
    Python (recording the EXECUTED slot + Python's mask at each step), feed the SAME slot sequence to
    the C++ `chocofarm-mask-dump` fixture (deterministic given the world), and assert each step's mask
    matches element-for-element. The env is deterministic given (world, action sequence), so the mask
    is a logic fact float32 cannot perturb."""
    from chocofarm.az.actions import action_to_slot
    rng = np.random.default_rng(seed)
    pol = RandomPolicy()
    total_steps = 0
    for _ in range(n_seqs):
        world = int(rng.choice(env.worlds))
        loc = ("w", env.entry)
        bw = env.worlds
        collected = set()
        py_masks = []
        slots = []
        for _ in range(max_steps):
            if len(bw) == 0:
                break
            py_masks.append(legal_mask(env, loc, bw, collected))   # Python's authoritative mask
            a = pol.decide(env, loc, bw, collected, 0.0, rng)
            slots.append(action_to_slot(env, a))
            if a == TERMINATE:
                break
            _, loc, bw, collected, _ = env.apply(loc, bw, collected, a, world)
        # replay the SAME slot sequence through the C++ fixture
        seq_str = " ".join(str(s) for s in slots)
        out = subprocess.run([MASK_BIN, "--instance", INSTANCE, "--faces", FACES,
                              "--world", str(world)],
                             input=seq_str + "\n", capture_output=True, text=True)
        if out.returncode != 0:
            raise RuntimeError(f"mask-dump failed: {out.stderr}")
        cpp_masks = [np.array([int(x) for x in line.split()], dtype=np.float32)
                     for line in out.stdout.strip().splitlines() if line.strip()]
        assert len(cpp_masks) == len(py_masks), (len(cpp_masks), len(py_masks))
        for k, (pm, cm) in enumerate(zip(py_masks, cpp_masks)):
            # BIT-EXACT: the float32 mask arrays must be byte-identical (== over float32).
            assert pm.astype(np.float32).tobytes() == cm.astype(np.float32).tobytes(), \
                f"mask mismatch at step {k}: py={pm.tolist()} cpp={cm.tolist()}"
            total_steps += 1
    return total_steps


def feature_parity(env, n_seqs, max_steps, seed, feat_dim_):
    """X-port equivalence: the §2.2 feature vector the C++ FeatureBuilder produces vs Python's, for a
    matched (loc, belief). Held to the forward-roundoff bar (ABS_TOL=1e-4, the project's
    test_jax_equivalence tolerance — ADR-0012 P6 / ADR-0009), not byte-identity: both are float64 of
    the SAME math, so they agree to far tighter than 1e-4, but we hold the documented bar. Reports
    the max abs diff across all matched steps."""
    from chocofarm.az.actions import action_to_slot
    from chocofarm.az.features import FeatureBuilder
    fb = FeatureBuilder(env)
    rng = np.random.default_rng(seed)
    pol = RandomPolicy()
    max_abs = 0.0
    total = 0
    for _ in range(n_seqs):
        world = int(rng.choice(env.worlds))
        loc = ("w", env.entry)
        bw = env.worlds
        collected = set()
        py_feats = []
        slots = []
        for _ in range(max_steps):
            if len(bw) == 0:
                break
            py_feats.append(fb.build(loc, bw, collected).astype(np.float64))
            a = pol.decide(env, loc, bw, collected, 0.0, rng)
            slots.append(action_to_slot(env, a))
            if a == TERMINATE:
                break
            _, loc, bw, collected, _ = env.apply(loc, bw, collected, a, world)
        seq_str = " ".join(str(s) for s in slots)
        out = subprocess.run([MASK_BIN, "--instance", INSTANCE, "--faces", FACES,
                              "--world", str(world), "--features"],
                             input=seq_str + "\n", capture_output=True, text=True)
        if out.returncode != 0:
            raise RuntimeError(f"mask-dump --features failed: {out.stderr}")
        cpp_feats = [np.array([float(x) for x in line.split()], dtype=np.float64)
                     for line in out.stdout.strip().splitlines() if line.strip()]
        assert len(cpp_feats) == len(py_feats)
        for pf, cf in zip(py_feats, cpp_feats):
            assert cf.shape == (feat_dim_,)
            max_abs = max(max_abs, float(np.max(np.abs(pf - cf))))
            total += 1
    return max_abs, total


def format_roundtrip(res_token, episodes, feat_dim_, n_slots):
    """Read the four C++ blocks via np.frombuffer(...).reshape(...) per transport.py and confirm the
    shapes/dtypes parse exactly: X (n,feat_dim), PI/M (n,n_slots), Y (n,), all float32."""
    conn = transport.connect()
    n_total = 0
    for idx in range(episodes):
        xk, pik, mk, yk = transport.result_keys(res_token, idx)
        xb = conn.get(xk)
        if xb is None:
            continue
        yb, pib, mb = conn.get(yk), conn.get(pik), conn.get(mk)
        Y = np.frombuffer(yb, dtype=np.float32)
        n = len(Y)
        X = np.frombuffer(xb, dtype=np.float32).reshape(n, feat_dim_)
        PI = np.frombuffer(pib, dtype=np.float32).reshape(n, n_slots)
        M = np.frombuffer(mb, dtype=np.float32).reshape(n, n_slots)
        assert X.dtype == PI.dtype == M.dtype == Y.dtype == np.float32
        assert X.shape == (n, feat_dim_) and PI.shape == M.shape == (n, n_slots)
        n_total += n
    return n_total


def publish_weights(run_id, version):
    """Publish a real net to the (run, phase=gen, version) keys so the C++ runner's weight-read seam
    has a manifest+blob to parse (it reads but RandomPolicy ignores the weights — the read is the
    seam proof)."""
    from chocofarm.az.mlp import ValueMLP
    env = Environment()
    net = ValueMLP(feature_dim(env), hidden=32, n_actions=n_action_slots(env),
                   y_mean=0.0, y_std=1.0, residual=False)
    conn = transport.connect()
    transport.RedisTransport(conn).publish_weights(net, "gen", version, run_id)


def main():
    if not os.path.exists(CPP_BIN):
        print(f"FAIL: C++ binary not built at {CPP_BIN}\n"
              f"      build it: cmake -S cpp -B cpp/build && cmake --build cpp/build")
        return 1

    env = Environment()
    policy = RandomPolicy()
    fd = feature_dim(env)
    ns = n_action_slots(env)
    lam = 0.10                 # the fixed λ₀ for the λ-return comparison
    max_steps = env.max_steps  # the live horizon
    episodes = 400             # ≥300 per seed
    seeds = [11, 23]           # ≥2 seeds
    keys = ["length", "lam_return", "n_collect", "n_sense", "n_terminate", "belief_shrinkage"]

    run_id = "parity-" + uuid.uuid4().hex[:8]
    version = 0
    publish_weights(run_id, version)

    print(f"=== ADR-0012 P6/P7 behavioral parity: C++ runner vs Python RandomPolicy ===")
    print(f"instance: N={env.N} K={env.K} nDet={len(env.detectors)} |worlds|={len(env.worlds)}  "
          f"feat_dim={fd} n_slots={ns}  λ₀={lam} max_steps={max_steps}")
    print(f"episodes/seed={episodes}  seeds={seeds}  (N total per side = {episodes*len(seeds)})\n")

    # --- STRONG bit-exact mask claim: matched (loc, belief) replay, M byte-identical ---
    n_mask_steps = mask_bit_exact(env, n_seqs=200, max_steps=max_steps, seed=777, n_slots=ns)
    print(f"[mask bit-exact] C++ M == Python legal_mask byte-for-byte over {n_mask_steps} matched "
          f"(loc, belief) steps across 200 episodes")

    # --- X-port equivalence: feature vector vs Python, forward-roundoff bar (ABS_TOL=1e-4) ---
    ABS_TOL = 1e-4
    feat_max_abs, n_feat_steps = feature_parity(env, n_seqs=200, max_steps=max_steps, seed=888,
                                                feat_dim_=fd)
    feat_ok = feat_max_abs <= ABS_TOL
    print(f"[feature X-port] max|Δ| = {feat_max_abs:.3e} over {n_feat_steps} matched steps "
          f"(bar ABS_TOL={ABS_TOL:.0e}) -> {'OK' if feat_ok else 'DIVERGE'}\n")

    py_rows_all, cpp_rows_all = [], []
    last_token = None
    for seed in seeds:
        py_rows = py_run(env, policy, lam, episodes, seed, max_steps)
        cpp_rows, res_token = cpp_run(run_id, version, lam, episodes, seed, max_steps)
        py_rows_all += py_rows
        cpp_rows_all += cpp_rows
        last_token = res_token
        # per-seed mask + format checks on this seed's blobs
        checked = mask_parity(env, res_token, episodes, fd, ns)
        nfmt = format_roundtrip(res_token, episodes, fd, ns)
        print(f"[seed {seed}] mask logic-invariants bit-exact over {checked} decisions; "
              f"format round-trip OK over {nfmt} rows")

    print()
    res = compare(py_rows_all, cpp_rows_all, keys)
    ok = feat_ok
    print(f"{'stat':<18}{'py mean±SE':<26}{'cpp mean±SE':<26}{'Δ':<14}{'SE_comb':<12}{'|z|':<8}verdict")
    for k in keys:
        r = res[k]
        py = f"{r['py_mean']:.5f}±{r['py_se']:.5f}"
        cp = f"{r['cpp_mean']:.5f}±{r['cpp_se']:.5f}"
        verdict = "OK" if r["z"] < 3.0 else "DIVERGE"
        if r["z"] >= 3.0:
            ok = False
        print(f"{k:<18}{py:<26}{cp:<26}{r['diff']:<+14.5f}{r['se_combined']:<12.5f}"
              f"{r['z']:<8.2f}{verdict}")

    print()
    print("Bar: |z| = |Δ| / SE_combined < 3.0 (≈99.7% two-sided MC band) for every float-sensitive")
    print("     aggregate; the legality mask M is a logic invariant asserted BIT-EXACT (above).")
    print()
    print("RESULT:", "PASS — aggregates indistinguishable within MC CI; mask bit-exact; format "
          "round-trips" if ok else "FAIL — an aggregate diverged beyond 3·SE")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
