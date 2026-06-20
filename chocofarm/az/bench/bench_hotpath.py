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
from collections.abc import Callable
from typing import Any

import numpy as np
import numpy.typing as npt

from chocofarm.model.env import Environment, TERMINATE
from chocofarm.az.features import FeatureBuilder
from chocofarm.az.mlp import ValueMLP
from chocofarm.az.actions import (n_action_slots, action_to_slot, slot_to_action,
                                  legal_mask_from_features)
from chocofarm.az.bench.capture_states import load_states


def _time(fn: Callable[[], Any], iters: int, repeat: int = 5) -> float:
    """Median over `repeat` runs of the per-call time (seconds) of `fn` run `iters` times."""
    best = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        best.append((time.perf_counter() - t0) / iters)
    best.sort()
    return best[len(best) // 2]


def bench(net_path: str, states_path: str, repeat: int = 5) -> tuple[dict[str, float], int]:
    env, states = load_states(states_path)
    fb = FeatureBuilder(env)
    net = ValueMLP.load(net_path)
    N, nD = env.N, len(env.detectors)
    results: dict[str, float] = {}

    # representative working set: cycle through captured states (real |bw| distribution)
    locs = [s[0] for s in states]
    bws = [s[1] for s in states]
    colls = [s[2] for s in states]
    S = len(states)

    # pre-build feats + masks for components that need them
    feats = [fb.build(locs[k], bws[k], colls[k]) for k in range(S)]
    masks = [legal_mask_from_features(env, f) for f in feats]
    # a legal node-state for _puct_select: emulate a partially-visited node
    from chocofarm.az.gumbel_search import _Node, GumbelAZSearch
    search = GumbelAZSearch(net, env, m=12, n_sims=48)

    # --- marginals ---
    i = {"k": 0}

    def f_marg() -> None:
        i["k"] = (i["k"] + 1) % S
        env.marginals(bws[i["k"]])
    results["env.marginals"] = _time(f_marg, S, repeat)

    # --- features.build (the fused kernel derives marginals; isolates the build itself) ---
    i["k"] = 0

    def f_build() -> None:
        i["k"] = (i["k"] + 1) % S
        k = i["k"]
        fb.build(locs[k], bws[k], colls[k])
    results["features.build"] = _time(f_build, S, repeat)

    # --- belief reductions (the (nb x nD) detector block in build) ---
    cover = fb.cover
    i["k"] = 0

    def f_belief_red() -> None:
        i["k"] = (i["k"] + 1) % S
        bw = bws[i["k"]]
        if len(bw):
            hit = (bw[:, None] & cover[None, :]) != 0
            _ = hit.mean(0)
            _ = hit.any(0)
            _ = (~hit).any(0)
    results["belief_reductions"] = _time(f_belief_red, S, repeat)

    # --- belief reductions, NUMBA fused kernel (marginals + detector counts in one pass) ---
    from chocofarm.az.kernels import belief_marg_cover, warmup as _kwarm
    _kwarm(N, nD)
    i["k"] = 0

    def f_belief_red_numba() -> None:
        i["k"] = (i["k"] + 1) % S
        bw = bws[i["k"]]
        if len(bw):
            belief_marg_cover(bw, cover, N)
    results["belief_reductions_numba"] = _time(f_belief_red_numba, S, repeat)

    # --- env.marginals, NUMBA kernel (the marginals-only fast path) ---
    from chocofarm.az.kernels import marginals_kernel
    i["k"] = 0

    def f_marg_numba() -> None:
        i["k"] = (i["k"] + 1) % S
        bw = bws[i["k"]]
        if len(bw):
            marginals_kernel(bw, N)
    results["env.marginals_numba"] = _time(f_marg_numba, S, repeat)

    # --- predict_both (NN forward + masked softmax). `net.predict_both` is the float32-numpy
    #     fast path when CHOCO_AZ_DTYPE=float32 (default) and the float64 path otherwise. ---
    i["k"] = 0

    def f_predict() -> None:
        i["k"] = (i["k"] + 1) % S
        k = i["k"]
        net.predict_both(feats[k], masks[k])
    results["predict_both"] = _time(f_predict, S, repeat)

    # --- predict_both, EXPLICIT float64 path (the pre-opt baseline forward) ---
    i["k"] = 0

    def f_predict_f64() -> None:
        i["k"] = (i["k"] + 1) % S
        k = i["k"]
        _, v_std, logits = net._forward(feats[k].astype(np.float64)[None, :])
        v = v_std * net.y_std + net.y_mean
        # logits is NDArray | None from _forward; the policy head is always loaded in this bench
        # (the net is the same one GumbelAZSearch uses, so n_actions is set). Assert loudly.
        assert logits is not None, "net has no policy head — bench requires n_actions set"
        net._masked_softmax(logits, masks[k][None, :].astype(np.float64))
        _ = v
    results["predict_both_f64"] = _time(f_predict_f64, S, repeat)

    # --- predict_both, JAX-jit single-eval (XLA float32). Compiled via warmup; the fresh-numpy
    #     per-call pattern the search actually drives it with — exposes the CPU-JAX per-dispatch
    #     overhead that makes single-eval LOSE to f32-numpy. ---
    try:
        from chocofarm.az.mlp_jax import MlpJaxForward
        jfwd = MlpJaxForward(net)
        # MlpJaxForward.__init__ raises if net.n_actions is None; after construction n_actions is int.
        assert net.n_actions is not None, "MlpJaxForward requires n_actions — invariant from __init__"
        jfwd.warmup(net.in_dim, net.n_actions)
        i["k"] = 0

        def f_predict_jax() -> None:
            i["k"] = (i["k"] + 1) % S
            k = i["k"]
            jfwd.predict_both(feats[k], masks[k])
        results["predict_both_jax_single"] = _time(f_predict_jax, S, repeat)

        # --- JAX BATCHED: stack 48 leaves and eval at once (the regime where XLA amortizes the
        #     dispatch — reported per-item so it is comparable to the single-eval numbers). ---
        B = 48
        Xb = np.stack([feats[k % S] for k in range(B)]).astype(np.float32)
        Mb = np.stack([masks[k % S] for k in range(B)]).astype(np.float32)
        jfwd.predict_both(Xb, Mb)   # compile the batched signature

        def f_predict_jax_batch() -> None:
            jfwd.predict_both(Xb, Mb)
        results["predict_both_jax_batch48_peritem"] = _time(f_predict_jax_batch, 1, repeat) / B
    except Exception as e:  # pragma: no cover - jax optional
        results["predict_both_jax_single"] = float("nan")
        results["predict_both_jax_batch48_peritem"] = float("nan")
        print(f"[bench] jax variants skipped: {e}")

    # --- slot conversions (round-trip per legal slot) ---
    legal_lists: list[list[Any]] = []
    for f, m in zip(feats, masks):
        # slot_to_action takes int; np.nonzero returns signedinteger — cast explicitly.
        legal = [slot_to_action(env, int(s)) for s in np.nonzero(m)[0]]
        legal_lists.append(legal)
    i["k"] = 0

    def f_slotconv() -> None:
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

    def f_dist() -> None:
        i["k"] = (i["k"] + 1) % S
        loc = locs[i["k"]]
        for j in range(N):
            env.d(loc, ("t", j))
    results["env.d"] = _time(f_dist, S, repeat) / N  # per call

    # --- filter_treasure / filter_detector ---
    i["k"] = 0

    def f_filt_t() -> None:
        i["k"] = (i["k"] + 1) % S
        bw = bws[i["k"]]
        if len(bw):
            env.filter_treasure(bw, 0, True)
    results["filter_treasure"] = _time(f_filt_t, S, repeat)

    i["k"] = 0

    def f_filt_d() -> None:
        i["k"] = (i["k"] + 1) % S
        bw = bws[i["k"]]
        if len(bw):
            env.filter_detector(bw, 0, True)
    results["filter_detector"] = _time(f_filt_d, S, repeat)

    # --- _puct_select (build a node with a cached eval + some visits, then select) ---
    nodes: list[_Node] = []
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

    def f_puct() -> None:
        i["k"] = (i["k"] + 1) % S
        node = nodes[i["k"]]
        if node.legal:
            search._puct_select(env, node)
    results["_puct_select"] = _time(f_puct, S, repeat)

    # --- _evaluate (the full leaf eval: marginals + build + mask + forward + legal) ---
    i["k"] = 0

    def f_eval() -> None:
        i["k"] = (i["k"] + 1) % S
        k = i["k"]
        node = _Node()
        search._evaluate(node, locs[k], bws[k], colls[k])
    results["_evaluate"] = _time(f_eval, S, repeat)

    return results, S


def main() -> None:
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
