#!/usr/bin/env python3
"""
chocofarm AZ bench — per-component micro-benchmark of the Gumbel-AZ search hot path.

Each hot-path component is timed IN ISOLATION on the representative captured states
(`states.npz`, |bw| from 15,504 down to 1; see capture_states.py). This is the regression
guard: an optimization is validated one component at a time, before/after, so a speedup claim is
never an artifact of episode-level noise.

Components benched:
  * env.marginals            — belief -> per-treasure marginals (numpy reduction)
  * features.build           — the §2.2 feature vector (the ~40% bucket)
  * belief_reductions        — the (nb x nD) detector hit/p_pos/informative reductions in build
  * _puct_select             — interior PUCT argmax (pure-Python bookkeeping)
  * slot_conversions         — slot_to_action / action_to_slot (3.5M calls in the loop)
  * env.d / distance         — inter-node distance (2.08M calls)
  * filter_treasure/detector — belief filtering on a chance outcome
  * predict_both             — the net forward + masked softmax (the ~15% NN bucket)

Run (pinned + bounded):
    PYTHONPATH=. taskset -c 2 timeout 180 python -m chocofarm.az.bench.bench_hotpath \\
        --net .scratch/net.npz --states chocofarm/az/bench/states.npz

`--json out.json` dumps the per-component median ns/call so before/after diffs are mechanical.
`--repeat` controls timing repeats (median of N reported). Deterministic given the states file.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

from chocofarm.model.env import Environment, TERMINATE
from chocofarm.az.features import FeatureBuilder
from chocofarm.az.mlp import ValueMLP
from chocofarm.az.actions import (n_action_slots, action_to_slot, slot_to_action,
                                  legal_mask_from_features)
from chocofarm.az.bench.capture_states import load_states


def _time(fn, iters, repeat=5):
    """Median over `repeat` runs of the per-call time (seconds) of `fn` run `iters` times."""
    best = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        best.append((time.perf_counter() - t0) / iters)
    best.sort()
    return best[len(best) // 2]


def bench(net_path, states_path, repeat=5):
    env, states = load_states(states_path)
    fb = FeatureBuilder(env)
    net = ValueMLP.load(net_path)
    N, nD = env.N, len(env.detectors)
    results = {}

    # representative working set: cycle through captured states (real |bw| distribution)
    locs = [s[0] for s in states]
    bws = [s[1] for s in states]
    colls = [s[2] for s in states]
    S = len(states)

    # pre-build feats + masks for components that need them
    margs = [env.marginals(bw) for bw in bws]
    feats = [fb.build(locs[k], bws[k], colls[k], marg=margs[k]) for k in range(S)]
    masks = [legal_mask_from_features(env, f) for f in feats]
    # a legal node-state for _puct_select: emulate a partially-visited node
    from chocofarm.az.gumbel_search import _Node, GumbelAZSearch
    search = GumbelAZSearch(net, env, m=12, n_sims=48)

    # --- marginals ---
    i = {"k": 0}

    def f_marg():
        i["k"] = (i["k"] + 1) % S
        env.marginals(bws[i["k"]])
    results["env.marginals"] = _time(f_marg, S, repeat)

    # --- features.build (marg supplied; isolates the build itself) ---
    i["k"] = 0

    def f_build():
        i["k"] = (i["k"] + 1) % S
        k = i["k"]
        fb.build(locs[k], bws[k], colls[k], marg=margs[k])
    results["features.build"] = _time(f_build, S, repeat)

    # --- belief reductions (the (nb x nD) detector block in build) ---
    cover = fb.cover
    i["k"] = 0

    def f_belief_red():
        i["k"] = (i["k"] + 1) % S
        bw = bws[i["k"]]
        if len(bw):
            hit = (bw[:, None] & cover[None, :]) != 0
            _ = hit.mean(0)
            _ = hit.any(0)
            _ = (~hit).any(0)
    results["belief_reductions"] = _time(f_belief_red, S, repeat)

    # --- predict_both (NN forward + masked softmax) ---
    i["k"] = 0

    def f_predict():
        i["k"] = (i["k"] + 1) % S
        k = i["k"]
        net.predict_both(feats[k], masks[k])
    results["predict_both"] = _time(f_predict, S, repeat)

    # --- slot conversions (round-trip per legal slot) ---
    legal_lists = []
    for f, m in zip(feats, masks):
        legal = [slot_to_action(env, s) for s in np.nonzero(m)[0]]
        legal_lists.append(legal)
    i["k"] = 0

    def f_slotconv():
        i["k"] = (i["k"] + 1) % S
        legal = legal_lists[i["k"]]
        for a in legal:
            s = action_to_slot(env, a)
            _ = slot_to_action(env, s)
    n_conv = sum(len(l) for l in legal_lists)
    t = _time(f_slotconv, S, repeat)
    results["slot_conversions"] = t * S / max(1, n_conv)  # per round-trip

    # --- env.d / distance (per legal node from loc) ---
    i["k"] = 0

    def f_dist():
        i["k"] = (i["k"] + 1) % S
        loc = locs[i["k"]]
        for j in range(N):
            env.d(loc, ("t", j))
    results["env.d"] = _time(f_dist, S, repeat) / N  # per call

    # --- filter_treasure / filter_detector ---
    i["k"] = 0

    def f_filt_t():
        i["k"] = (i["k"] + 1) % S
        bw = bws[i["k"]]
        if len(bw):
            env.filter_treasure(bw, 0, True)
    results["filter_treasure"] = _time(f_filt_t, S, repeat)

    i["k"] = 0

    def f_filt_d():
        i["k"] = (i["k"] + 1) % S
        bw = bws[i["k"]]
        if len(bw):
            env.filter_detector(bw, 0, True)
    results["filter_detector"] = _time(f_filt_d, S, repeat)

    # --- _puct_select (build a node with a cached eval + some visits, then select) ---
    nodes = []
    for k in range(S):
        node = _Node()
        node.feat = feats[k]
        node.mask = masks[k]
        node.prior = net.predict_both(feats[k], masks[k])[1]
        node.value = 0.1
        node.legal = legal_lists[k]
        # seed a few visits so the Q/N path is exercised
        for a in node.legal[:3]:
            node.N[a] = 2
            node.W[a] = 0.3
        nodes.append(node)
    i["k"] = 0

    def f_puct():
        i["k"] = (i["k"] + 1) % S
        node = nodes[i["k"]]
        if node.legal:
            search._puct_select(env, node)
    results["_puct_select"] = _time(f_puct, S, repeat)

    # --- _evaluate (the full leaf eval: marginals + build + mask + forward + legal) ---
    i["k"] = 0

    def f_eval():
        i["k"] = (i["k"] + 1) % S
        k = i["k"]
        node = _Node()
        search._evaluate(node, locs[k], bws[k], colls[k])
    results["_evaluate"] = _time(f_eval, S, repeat)

    return results, S


def main():
    ap = argparse.ArgumentParser(description="Hot-path micro-bench.")
    ap.add_argument("--net", required=True)
    ap.add_argument("--states", required=True)
    ap.add_argument("--repeat", type=int, default=5)
    ap.add_argument("--json", default=None)
    ap.add_argument("--label", default="run")
    args = ap.parse_args()
    results, S = bench(args.net, args.states, args.repeat)
    print(f"[{args.label}] {S} states; median per-call (us):")
    width = max(len(k) for k in results)
    for k in sorted(results, key=lambda x: -results[x]):
        print(f"  {k:<{width}}  {results[k] * 1e6:8.3f} us")
    if args.json:
        with open(args.json, "w") as f:
            json.dump({"label": args.label, "S": S, "us": {k: v * 1e6 for k, v in results.items()}}, f, indent=2)
        print(f"-> {args.json}")


if __name__ == "__main__":
    main()
