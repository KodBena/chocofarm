#!/usr/bin/env python3
"""
chocofarm AZ bench — Part B value-target variance probe (MC vs TD(λ)/n-step).

The mechanism claim of Part B is: bootstrapping the value target off the search's ~n_sims-averaged
root value LOWERS the target's variance vs the single-rollout MC return-to-go, so the value head can
fit the geometry/belief-dependent component the high-variance MC target collapsed away from.

This probe makes that variance reduction a NUMBER on real episodes, isolating it from policy drift:
it rolls the SAME set of net-guided episodes ONCE (recording, per decision, the realized (r,dt)
steps AND the search root-value bootstrap), then recomputes the value target under several blends
over the IDENTICAL episodes. Because the episodes are held fixed, any change in target variance is
attributable to the target rule, not to a different trajectory distribution.

Reported per blend:
  * mean / std / variance of the value target over all decisions,
  * the variance RATIO vs pure MC (the headline: <1 is the variance reduction),
  * E[mean target] drift (the bootstrap-optimism watch — a blend that raises the mean target is
    pulling toward the search's optimistic estimate; Part B's honest risk).

Also reports, if a held-out (X, y) is buildable from the same episodes, the value R² a freshly-fit
linear probe on the features achieves under each target — a cheap "can the target be fit" signal.
(The loop's own per-iter `value_R2` is the real fit number; this is a bench-time sanity probe.)

Pinned + bounded:
    PYTHONPATH=. taskset -c 2 timeout 300 python -m chocofarm.az.bench.bench_value_target \\
        --net .scratch/net.npz --episodes 40 --seed 7
"""
from __future__ import annotations

import argparse

import numpy as np

from chocofarm.model.env import Environment, TERMINATE
from chocofarm.az.features import FeatureBuilder
from chocofarm.az.actions import legal_mask_from_features
from chocofarm.az.mlp import ValueMLP
from chocofarm.az.gumbel_search import GumbelAZSearch
from chocofarm.az.value_target import blended_returns_to_go


def roll_episode_raw(env, search, fb, world, lam, rng, max_steps=40):
    """Roll ONE net-guided episode, returning (feats, step_rt, boots, exit_c) — everything the
    value-target rule needs, recorded ONCE so multiple blends reuse the identical trajectory.
    Mirrors generate_episode's recording but does NOT compute the target."""
    loc, bw, collected = ("w", env.entry), env.worlds, set()
    feats, step_rt, boots = [], [], []
    for _ in range(max_steps):
        if len(bw) == 0:
            break
        action, _pi, boot = search.decide_with_value(env, loc, bw, collected, lam, rng,
                                                     temperature=0.0)
        feats.append(fb.build(loc, bw, collected))
        boots.append(boot)
        if action == TERMINATE:
            break
        r, loc, bw, collected, dt = env.apply(loc, bw, collected, action, world)
        step_rt.append((r, dt))
    exit_c = env.exit_cost(loc)
    n_dec = len(step_rt)
    return feats[:n_dec], step_rt, boots[:n_dec], exit_c


def fit_r2(X, y):
    """R² of a ridge-fit linear probe (closed form) — a cheap 'is the target fittable' signal."""
    if X.shape[0] < 5:
        return float("nan")
    Xb = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1).astype(np.float64)
    lam_r = 1e-3 * X.shape[1]
    A = Xb.T @ Xb + lam_r * np.eye(Xb.shape[1])
    w = np.linalg.solve(A, Xb.T @ y)
    pred = Xb @ w
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def main():
    ap = argparse.ArgumentParser(description="Part B value-target variance probe.")
    ap.add_argument("--net", required=True)
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--lam", type=float, default=0.0855)
    ap.add_argument("--m", type=int, default=12)
    ap.add_argument("--n-sims", type=int, default=48)
    ap.add_argument("--blends", default="mc,td0.7,td0.3,n2,n1",
                    help="comma list: mc | td<ell> | n<k>")
    args = ap.parse_args()

    env = Environment()
    fb = FeatureBuilder(env)
    net = ValueMLP.load(args.net)
    search = GumbelAZSearch(net, env, m=args.m, n_sims=args.n_sims)
    rng = np.random.default_rng(args.seed)

    eps = []
    for _ in range(args.episodes):
        w = int(rng.choice(env.worlds))
        eps.append(roll_episode_raw(env, search, fb, w, args.lam, rng))
    n_dec = sum(len(e[0]) for e in eps)
    print(f"rolled {args.episodes} episodes, {n_dec} step-decisions; λ={args.lam}", flush=True)

    def parse_blend(tag):
        if tag == "mc":
            return ("mc", dict(lam_blend=1.0, n_step=None))
        if tag.startswith("td"):
            return (tag, dict(lam_blend=float(tag[2:]), n_step=None))
        if tag.startswith("n"):
            return (tag, dict(lam_blend=1.0, n_step=int(tag[1:])))
        raise ValueError(f"bad blend tag {tag!r}")

    blends = [parse_blend(t) for t in args.blends.split(",")]
    mc_var = None
    rows = []
    for tag, kw in blends:
        Xs, Ys = [], []
        for feats, step_rt, boots, exit_c in eps:
            g = blended_returns_to_go(step_rt, boots, exit_c, args.lam, **kw)
            for f, gv in zip(feats, g):
                Xs.append(f); Ys.append(gv)
        X = np.asarray(Xs, dtype=np.float64)
        Y = np.asarray(Ys, dtype=np.float64)
        var = float(np.var(Y))
        if tag == "mc":
            mc_var = var
        r2 = fit_r2(X, Y)
        rows.append((tag, float(Y.mean()), float(Y.std()), var, r2))

    print(f"\n{'blend':>8} {'mean':>9} {'std':>9} {'var':>9} {'var/mc':>8} {'probe_R2':>9}")
    for tag, mean, std, var, r2 in rows:
        ratio = var / mc_var if mc_var and mc_var > 0 else float("nan")
        print(f"{tag:>8} {mean:>+9.4f} {std:>9.4f} {var:>9.4f} {ratio:>8.3f} {r2:>9.4f}")
    print("\n(var/mc < 1 ⇒ Part B variance reduction; mean drifting UP vs mc ⇒ bootstrap optimism)")


if __name__ == "__main__":
    main()
