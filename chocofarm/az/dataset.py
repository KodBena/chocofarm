#!/usr/bin/env python3
"""
chocofarm AZ — value-target dataset generation (design §4.1, §4.5).

Generates (feature-vector, value-target) pairs at a FIXED λ₀ (default 0.0855, the static-floor
rate — design §4.1's "pin λ for the whole run"). The TEACHER is the **decomp policy**
(`DecompPolicy`), the project's strong exact-decomposition solver (clears the floor at rate
~0.094). The design's Stage-1 recipe names the *ISMCTS* policy as the teacher; we use decomp
instead because it is both stronger AND deterministic-faster, so its honest realized returns are
higher-quality, lower-over-collection labels for the SAME quantity. The substituted teacher is
documented in `docs/results/az-edecide.md` and in the report — it is a deliberate, honest
deviation, not an oversight.

The value target for each decision point is the **honest realized λ-penalized return-to-go of
that decomp episode** under true partial-observation dynamics (design §4.5 — the F4 cure):

    G_j = Σ_{t≥j} r_t  −  λ·( Σ_{t≥j} dt_t  +  exit_cost(final_loc) )

i.e. the actual banked value from decision point j onward, minus λ times the actual remaining
travel plus the single end-of-episode exit toll. This is the same object `_base_value` computes
for a determinized rollout, but here it is what ACTUALLY happened when the policy acted under
uncertainty — never a clairvoyant best-case. We use ONLY this honest-MC label (no analytic
decomp value-to-go), keeping the probe's calibration story clean.

We replicate `env.simulate`'s loop verbatim (fresh full belief at the entry teleport, so the
decomp policy's per-episode reset fires identically) and log features at every `decide` call.

CLI:  python -m chocofarm.az.dataset --episodes N --out path.npz [--lam 0.0855] [--seed S]
Output npz: X (n_transitions, feat_dim) float32, y (n_transitions,) float32, plus meta.
Pin to core 3 under timeout (see docs/results/az-edecide.md for the full-run command).
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from chocofarm.model.env import Environment, TERMINATE
from chocofarm.solvers.decomp import DecompPolicy
from chocofarm.az.features import FeatureBuilder, feature_dim
from chocofarm.az.value_target import suffix_returns_to_go


def _episode_transitions(env, policy, fb, world, lam, rng, max_steps=40):
    """Run ONE decomp episode against `world`, logging (features, per-step (r,dt)) at each
    decision point. Returns a list of (feat, return_to_go) for the visited states.

    Mirrors env.simulate exactly (same loc/bw/collected init and update), so the decomp
    policy's fresh-episode detection (full belief at entry) triggers its per-episode reset."""
    loc, bw, collected = ("w", env.entry), env.worlds, set()
    feats = []          # feature vector logged BEFORE each executed action
    step_rt = []        # (r, dt) of each executed action
    for _ in range(max_steps):
        a = policy.decide(env, loc, bw, collected, lam, rng)
        if a == TERMINATE:
            break
        # log the state we DECIDED from (one cached marginals call per node, F7)
        marg = env.marginals(bw)
        feats.append(fb.build(loc, bw, collected, marg=marg))
        r, loc, bw, collected, dt = env.apply(loc, bw, collected, a, world)
        step_rt.append((r, dt))
    exit_c = env.exit_cost(loc)
    # realized λ-penalized return-to-go from each decision point j (the pure-MC suffix rule, the
    # decomp teacher's honest label — Part B's blend does NOT apply here: the dataset is the
    # un-bootstrapped MC target by design). Routed through the shared value_target module so the
    # suffix rule lives in ONE place (the az-exit-loop §(f) audit's prescription, now that exit_loop
    # adds a TD(λ) blend over the same rule).
    g = suffix_returns_to_go(step_rt, exit_c, lam)
    return list(zip(feats, g))


def generate(env, n_episodes, lam, seed, report_every=50):
    fb = FeatureBuilder(env)
    pol = DecompPolicy(horizon=1)
    rng = np.random.default_rng(seed)
    X, Y = [], []
    t0 = time.time()
    for ep in range(n_episodes):
        w = int(rng.choice(env.worlds))
        for feat, g in _episode_transitions(env, pol, fb, w, lam, rng):
            X.append(feat); Y.append(g)
        if report_every and (ep + 1) % report_every == 0:
            print(f"  ...{ep + 1}/{n_episodes} episodes, {len(X)} transitions "
                  f"({time.time() - t0:.0f}s)", flush=True)
    X = np.asarray(X, dtype=np.float32)
    Y = np.asarray(Y, dtype=np.float32)
    return X, Y, fb.dim


def main():
    ap = argparse.ArgumentParser(description="Generate AZ value-target dataset (decomp teacher).")
    ap.add_argument("--episodes", type=int, default=300, help="decomp episodes to roll out")
    ap.add_argument("--out", type=str, required=True, help="output .npz path")
    ap.add_argument("--lam", type=float, default=0.0855, help="fixed λ₀ (static-floor rate)")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    env = Environment()
    print(f"env: N={env.N} detectors(faces)={len(env.detectors)} "
          f"teleports={len(env.teleports)} feat_dim={feature_dim(env)}", flush=True)
    print(f"generating {args.episodes} decomp episodes at λ={args.lam} ...", flush=True)
    X, Y, dim = generate(env, args.episodes, args.lam, args.seed)
    np.savez(args.out, X=X, y=Y,
             meta=np.array([dim, args.episodes], dtype=np.int64),
             lam=np.array([args.lam], dtype=np.float64))
    print(f"wrote {X.shape[0]} transitions × {dim} feats -> {args.out}", flush=True)
    print(f"target stats: mean={Y.mean():.4f} std={Y.std():.4f} "
          f"min={Y.min():.3f} max={Y.max():.3f}", flush=True)


if __name__ == "__main__":
    main()
