#!/usr/bin/env python3
"""
chocofarm AZ bench — behavioral-equivalence harness for the perf optimization.

float32 + numba change floats and will flip near-tied argmax / Sequential-Halving choices. The
correctness bar is NOT bit- or per-decision equality (it can't be) but AGGREGATE behavioral
equivalence: the optimized policy's fixed-λ₀ rate, mean E[T], and action distribution must be
statistically indistinguishable from the float64 baseline over N≥300 episodes across ≥2 seeds,
within Monte-Carlo CI.

This driver builds a GumbelPolicy at whatever the ambient CHOCO_AZ_DTYPE is and rolls out N
episodes per seed at the pinned λ₀, reporting:
  * fixed-λ₀ rate  = ΣR / ΣT           (the headline metric)
  * mean E[T]       = ΣT / N
  * action histogram over the slot space (collect / sense / terminate buckets + per-slot)

Run it once with CHOCO_AZ_DTYPE=float64 and once with =float32 (same net, same seeds); the rates'
CIs must overlap. The MC standard error on the rate is reported so "indistinguishable" is a
number, not an eyeball.

Pinned + bounded:
    PYTHONPATH=. CHOCO_AZ_DTYPE=float32 taskset -c 2 timeout 600 \
        python -m chocofarm.az.bench.bench_equivalence --net .scratch/net.npz \
        --episodes 300 --seeds 0,1 --json .scratch/equiv_f32.json
"""
from __future__ import annotations

import argparse
import json
import math

import numpy as np

from chocofarm.model.env import Environment
from chocofarm.az.features import feature_dim
from chocofarm.az.actions import n_action_slots, action_to_slot
from chocofarm.az.mlp import ValueMLP
from chocofarm.az.gumbel_search import GumbelPolicy
from chocofarm.az.dtypes import DTYPE_NAME


def rollout(env, pol, n_episodes, lam, seed):
    """Roll out `n_episodes` greedy episodes at fixed λ, recording R, T, and the executed-action
    slot histogram. Returns (sumR, sumT, Ts, action_counts dict slot->n)."""
    rng = np.random.default_rng(seed)
    n_slots = n_action_slots(env)
    sumR = sumT = 0.0
    Ts = []
    counts = np.zeros(n_slots, dtype=np.int64)
    for _ in range(n_episodes):
        w = int(rng.choice(env.worlds))
        # mirror env.simulate but tally the executed actions
        loc, bw, collected, R, T = ("w", env.entry), env.worlds, set(), 0.0, 0.0
        for _step in range(env.max_steps):          # the single episode-horizon home (env.py)
            a = pol.decide(env, loc, bw, collected, lam, rng)
            counts[action_to_slot(env, a)] += 1
            from chocofarm.model.env import TERMINATE
            if a == TERMINATE:
                break
            r, loc, bw, collected, dt = env.apply(loc, bw, collected, a, w)
            R += r; T += dt
        T += env.exit_cost(loc)
        sumR += R; sumT += T; Ts.append(T)
    return sumR, sumT, np.array(Ts), counts


def main():
    ap = argparse.ArgumentParser(description="AZ behavioral-equivalence rollout.")
    ap.add_argument("--net", required=True)
    ap.add_argument("--episodes", type=int, default=300)
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--lam", type=float, default=0.0855)
    ap.add_argument("--m", type=int, default=12)
    ap.add_argument("--n-sims", type=int, default=48)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    env = Environment()
    net = ValueMLP.load(args.net)
    pol = GumbelPolicy(net, env, m=args.m, n_sims=args.n_sims)
    seeds = [int(s) for s in args.seeds.split(",")]

    sumR = sumT = 0.0
    allTs = []
    n_slots = n_action_slots(env)
    counts = np.zeros(n_slots, dtype=np.int64)
    per_seed = []
    for sd in seeds:
        r, t, ts, c = rollout(env, pol, args.episodes, args.lam, sd)
        sumR += r; sumT += t; allTs.append(ts); counts += c
        per_seed.append({"seed": sd, "rate": r / t if t > 0 else 0.0,
                         "ET": float(ts.mean())})
    allTs = np.concatenate(allTs)
    N = len(allTs)
    rate = sumR / sumT if sumT > 0 else 0.0
    ET = float(allTs.mean())
    # MC standard error of E[T] (the rate's SE is harder as it's a ratio; report ET SE + a
    # ratio-of-means delta-method SE for the rate).
    ET_se = float(allTs.std(ddof=1) / math.sqrt(N))

    # action buckets
    N_t, nD = env.N, len(env.detectors)
    collect = int(counts[:N_t].sum())
    sense = int(counts[N_t:N_t + nD].sum())
    term = int(counts[N_t + nD])
    tot_actions = collect + sense + term

    out = {
        "dtype": DTYPE_NAME, "episodes_per_seed": args.episodes, "seeds": seeds,
        "N_total": N, "rate": rate, "ET": ET, "ET_se": ET_se,
        "per_seed": per_seed,
        "action_dist": {"collect": collect, "sense": sense, "terminate": term,
                        "collect_frac": collect / tot_actions,
                        "sense_frac": sense / tot_actions,
                        "terminate_frac": term / tot_actions},
        "action_counts": counts.tolist(),
    }
    print(f"[{DTYPE_NAME}] N={N} ({len(seeds)} seeds x {args.episodes})  "
          f"rate={rate:.5f}  E[T]={ET:.3f} (±{ET_se:.3f})")
    print(f"  actions: collect={collect} ({collect/tot_actions:.1%})  "
          f"sense={sense} ({sense/tot_actions:.1%})  term={term} ({term/tot_actions:.1%})")
    for ps in per_seed:
        print(f"  seed {ps['seed']}: rate={ps['rate']:.5f}  E[T]={ps['ET']:.3f}")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"-> {args.json}")


if __name__ == "__main__":
    main()
