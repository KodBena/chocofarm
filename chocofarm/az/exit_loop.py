#!/usr/bin/env python3
"""
chocofarm AZ — the Expert-Iteration (ExIt) loop (design §6); the full Gumbel-AZ "real run".

ExIt (Anthony, Tian & Barber 2017) decomposed: the Gumbel-AZ search (`gumbel_search.py`) is the
slow "expert" / planner; the policy+value net (`mlp.py`) is the fast "apprentice" / generaliser.
No self-play — a single agent against the stochastic simulator (`env.simulate`). Each iteration:

  1. GENERATE  E episodes via the net-guided Gumbel search, recording per decision
               (features(s), improved-policy π′(s), legal-mask(s)); the value target for each
               decision is the HONEST realized λ-penalized return-to-go of THAT episode
               (design §4.5, the F4 cure — never a determinized best-case).
  2. TRAIN     Adam on the replay buffer (last W iterations) with the AlphaZero loss
               CE(masked) + MSE(value) + L2 (design §6; `mlp.train_step`).
  3. EVALUATE  the greedy (argmax-π′) policy's rate on a held-out seed, at fixed λ₀ (and,
               optionally, its own Dinkelbach fixed point), reported as % of the +70% VoI gap.
  4. CHECKPOINT the net (npz) + a JSON history of per-iter rate/%VoI/losses, EVERY iteration —
               a timeout/restart loses nothing and the rate-per-iteration is inspectable.

λ is PINNED to λ₀ = 0.0855 (the static-floor rate; design §4.1) for the whole run. Streams to
TensorBoard (tensorboardX): eval rate (+floor/ceiling/decomp reference lines), %VoI, policy CE,
value MSE, value R², executed-policy entropy.

Warm start (design §6 init): the value head can be loaded from an E-DECIDE weights npz
(`--init-weights`); the policy head is then randomised (the value net has no policy head). If no
init weights are given, both heads start random.

CLI: python -m chocofarm.az.exit_loop -I 40 -E 300 -W 5 --epochs 2 --m 12 --n-sims 48
       --lam 0.0855 --seed 7 [--init-weights w.npz] --tb-logdir tb/az_exit_loop --ckpt-dir ckpt
Pin to a free core under timeout; numpy only (no new deps). See docs/results/az-exit-loop.md.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from chocofarm.model.env import Environment, TERMINATE
from chocofarm.az.features import FeatureBuilder, feature_dim
from chocofarm.az.actions import n_action_slots, legal_mask_from_features
from chocofarm.az.mlp import ValueMLP
from chocofarm.az.gumbel_search import GumbelAZSearch, GumbelPolicy

# reference lines (docs/results/voi-ceiling-2026-06-13.md, decomp-rate.md)
STATIC_FLOOR = 0.0855
CLAIRVOYANT_CEIL = 0.1454
DECOMP_ANCHOR = 0.0941


def r2_score(y_true, y_pred):
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def generate_episode(env, search, fb, world, lam, rng, n_explore_plies, max_steps=40):
    """One self-play episode driven by the Gumbel search. Records, per decision,
    (features, improved-pi, legal-mask). Value targets are filled by suffix accumulation after
    the episode (the realized λ-penalized return-to-go, design §4.5).

    `n_explore_plies`: sample the EXECUTED action from π′ for the first this-many plies
    (temperature 1), argmax thereafter (design §6: temperature on executed action to diversify
    trajectories). The improved-policy TARGET is always the full π′ regardless."""
    loc, bw, collected = ("w", env.entry), env.worlds, set()
    feats, pis, masks, step_rt = [], [], [], []
    for ply in range(max_steps):
        if len(bw) == 0:
            break
        temp = 1.0 if ply < n_explore_plies else 0.0
        action, pi = search.decide_with_target(env, loc, bw, collected, lam, rng, temperature=temp)
        # the node the search just evaluated cached feat+mask for THIS belief; rebuild cheaply
        # (marginals are served by build's per-belief cache, so we don't pre-compute them here)
        feat = fb.build(loc, bw, collected)
        mask = legal_mask_from_features(env, feat)
        feats.append(feat); pis.append(pi); masks.append(mask)
        if action == TERMINATE:
            break
        r, loc, bw, collected, dt = env.apply(loc, bw, collected, action, world)
        step_rt.append((r, dt))
    exit_c = env.exit_cost(loc)
    # realized λ-penalized return-to-go from each decision j (suffix accumulation; exit charged once)
    out = []
    suffix_r = suffix_t = 0.0
    n_dec = len(step_rt)   # decisions that executed a non-TERMINATE action
    # if the episode ended on TERMINATE, the last feat/pi/mask has no step; align targets to steps
    for j in range(len(feats) - 1, -1, -1):
        if j < n_dec:
            r_j, dt_j = step_rt[j]
            suffix_r += r_j
            suffix_t += dt_j
            g = suffix_r - lam * (suffix_t + exit_c)
        else:
            # the TERMINATE decision: continuation value is just the exit toll
            g = suffix_r - lam * (suffix_t + exit_c)
        out.append((feats[j], pis[j], masks[j], g))
    out.reverse()
    return out


class ReplayBuffer:
    """Last-W-iterations replay (design §6: window 4–6 iters; the policy drifts, drop ancient
    data). Stores per-iteration blocks; `sample_arrays` returns the concatenation."""

    def __init__(self, window):
        self.window = int(window)
        self.blocks = []   # list of (X, PI, M, Y) per iteration

    def add(self, X, PI, M, Y):
        self.blocks.append((X, PI, M, Y))
        if len(self.blocks) > self.window:
            self.blocks.pop(0)

    def arrays(self):
        X = np.concatenate([b[0] for b in self.blocks], 0)
        PI = np.concatenate([b[1] for b in self.blocks], 0)
        M = np.concatenate([b[2] for b in self.blocks], 0)
        Y = np.concatenate([b[3] for b in self.blocks], 0)
        return X, PI, M, Y


def train_epochs(net, X, PI, M, Y, epochs, batch, lr, l2, alpha, beta, rng):
    """Adam epochs over the buffer; re-pins the value standardization to the buffer's Y stats
    (design §3) before training so the MSE stays O(1) as the return distribution drifts. Returns
    (mean_ce, mean_value_std_mse, heldout_R2_on_buffer)."""
    net.set_value_scale(float(Y.mean()), float(Y.std()))
    n = X.shape[0]
    steps = max(1, n // batch)
    ce_tot = v_tot = 0.0; cnt = 0
    for ep in range(epochs):
        idx = rng.permutation(n)
        for s in range(steps):
            b = idx[s * batch:(s + 1) * batch]
            if len(b) == 0:
                continue
            ce, vl = net.train_step(X[b].astype(np.float64), PI[b].astype(np.float64),
                                    M[b].astype(np.float64), Y[b].astype(np.float64),
                                    lr, l2, alpha=alpha, beta=beta)
            ce_tot += ce; v_tot += vl; cnt += 1
    pv = net.predict_value(X.astype(np.float64))
    return ce_tot / max(1, cnt), v_tot / max(1, cnt), r2_score(Y, pv)


def policy_entropy(pis):
    """Mean entropy of the improved-policy targets (executed-policy diversity diagnostic)."""
    ent = 0.0
    for p in pis:
        nz = p[p > 0]
        ent += float(-np.sum(nz * np.log(nz)))
    return ent / max(1, len(pis))


def run(args):
    env = Environment()
    fb = FeatureBuilder(env)
    in_dim = feature_dim(env)
    n_slots = n_action_slots(env)
    print(f"env: N={env.N} faces={len(env.detectors)} teleports={len(env.teleports)} "
          f"feat_dim={in_dim} action_slots={n_slots}", flush=True)

    # --- net: warm-start value head from E-DECIDE weights if given; policy head random ---
    if args.init_weights:
        warm = ValueMLP.load(args.init_weights)
        net = ValueMLP(in_dim, hidden=warm.H, n_actions=n_slots, seed=args.seed,
                       y_mean=warm.y_mean, y_std=warm.y_std)
        # copy trunk + value head; policy head stays random
        net.W1, net.b1 = warm.W1.copy(), warm.b1.copy()
        net.W2, net.b2 = warm.W2.copy(), warm.b2.copy()
        net.Wv, net.bv = warm.Wv.copy(), warm.bv.copy()
        net._init_adam()
        print(f"warm-started value head + trunk from {args.init_weights} "
              f"(hidden={warm.H}); policy head random", flush=True)
    else:
        net = ValueMLP(in_dim, hidden=args.hidden, n_actions=n_slots, seed=args.seed)
        print(f"cold net (hidden={args.hidden}); both heads random", flush=True)

    os.makedirs(args.ckpt_dir, exist_ok=True)
    writer = None
    if args.tb_logdir:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter(args.tb_logdir)
        # reference lines as flat series (drawn at every iteration so TB renders them)
        print(f"streaming to TensorBoard -> {args.tb_logdir}", flush=True)

    buf = ReplayBuffer(args.window)
    gen_rng = np.random.default_rng(args.seed + 1)
    train_rng = np.random.default_rng(args.seed + 2)
    history = []
    lam = args.lam

    for it in range(args.iters):
        t0 = time.time()
        # ---- 1. GENERATE ----
        search = GumbelAZSearch(net, env, m=args.m, n_sims=args.n_sims)
        Xs, PIs, Ms, Ys, all_pis = [], [], [], [], []
        for ep in range(args.episodes):
            world = int(gen_rng.choice(env.worlds))
            recs = generate_episode(env, search, fb, world, lam, gen_rng, args.explore_plies)
            for feat, pi, mask, g in recs:
                Xs.append(feat); PIs.append(pi); Ms.append(mask); Ys.append(g); all_pis.append(pi)
        X = np.asarray(Xs, dtype=np.float32)
        PI = np.asarray(PIs, dtype=np.float32)
        M = np.asarray(Ms, dtype=np.float32)
        Y = np.asarray(Ys, dtype=np.float32)
        buf.add(X, PI, M, Y)
        ent = policy_entropy(all_pis)
        t_gen = time.time() - t0

        # ---- 2. TRAIN ----
        bX, bPI, bM, bY = buf.arrays()
        ce, vmse, r2 = train_epochs(net, bX, bPI, bM, bY, args.epochs, args.batch,
                                    args.lr, args.l2, args.alpha, args.beta, train_rng)
        t_train = time.time() - t0 - t_gen

        # ---- 3. EVALUATE (greedy argmax-π′ policy on a held-out seed, fixed λ₀) ----
        eval_pol = GumbelPolicy(net, env, m=args.m, n_sims=args.n_sims)
        eval_rng = np.random.default_rng(args.eval_seed)
        totR = totT = 0.0; ets = []
        for _ in range(args.eval_n):
            w = int(eval_rng.choice(env.worlds))
            R, T, _ = env.simulate(eval_pol, w, lam, eval_rng)
            totR += R; totT += T; ets.append(T)
        rate = totR / totT if totT > 0 else 0.0
        et = float(np.mean(ets)) if ets else 0.0
        voi = (rate - STATIC_FLOOR) / (CLAIRVOYANT_CEIL - STATIC_FLOOR) * 100
        t_eval = time.time() - t0 - t_gen - t_train

        # ---- 4. CHECKPOINT (every iteration) ----
        ckpt = os.path.join(args.ckpt_dir, f"net_iter{it:03d}.npz")
        net.save(ckpt)
        rec = {"iter": it, "rate": rate, "voi_pct": voi, "ET": et,
               "policy_CE": ce, "value_MSE": vmse, "value_R2": r2, "entropy": ent,
               "n_transitions": int(X.shape[0]), "buffer_size": int(bX.shape[0]),
               "t_gen": t_gen, "t_train": t_train, "t_eval": t_eval, "lam": lam}
        history.append(rec)
        with open(os.path.join(args.ckpt_dir, "history.json"), "w") as f:
            json.dump(history, f, indent=2)
        net.save(os.path.join(args.ckpt_dir, "latest_net.npz"))

        if writer is not None:
            writer.add_scalar("eval/rate", rate, it)
            writer.add_scalar("eval/voi_pct", voi, it)
            writer.add_scalar("eval/ET", et, it)
            writer.add_scalar("ref/static_floor", STATIC_FLOOR, it)
            writer.add_scalar("ref/clairvoyant_ceiling", CLAIRVOYANT_CEIL, it)
            writer.add_scalar("ref/decomp_anchor", DECOMP_ANCHOR, it)
            writer.add_scalar("train/policy_CE", ce, it)
            writer.add_scalar("train/value_MSE", vmse, it)
            writer.add_scalar("train/value_R2", r2, it)
            writer.add_scalar("gen/exec_policy_entropy", ent, it)
            writer.flush()

        print(f"iter {it:>3}/{args.iters}  rate={rate:.4f} (%VoI={voi:+.0f}) ET={et:.1f}  "
              f"CE={ce:.3f} vMSE={vmse:.3f} R²={r2:.3f} H={ent:.2f}  "
              f"[{X.shape[0]} tr | gen {t_gen:.0f}s train {t_train:.0f}s eval {t_eval:.0f}s]",
              flush=True)

    if writer is not None:
        writer.close()
    print(f"\nDONE {args.iters} iters. checkpoints + history.json -> {args.ckpt_dir}", flush=True)
    if history:
        best = max(history, key=lambda r: r["rate"])
        print(f"best eval rate {best['rate']:.4f} (%VoI={best['voi_pct']:+.0f}) at iter {best['iter']}",
              flush=True)


def main():
    ap = argparse.ArgumentParser(description="AZ Gumbel Expert-Iteration loop (design §6).")
    ap.add_argument("-I", "--iters", type=int, default=40, help="outer ExIt iterations")
    ap.add_argument("-E", "--episodes", type=int, default=300, help="self-play episodes/iter")
    ap.add_argument("-W", "--window", type=int, default=5, help="replay window (iterations)")
    ap.add_argument("--epochs", type=int, default=2, help="train epochs over the buffer/iter")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--m", type=int, default=12, help="Gumbel root actions")
    ap.add_argument("--n-sims", type=int, default=48, help="simulations/decision")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--l2", type=float, default=1e-4)
    ap.add_argument("--alpha", type=float, default=1.0, help="policy CE weight")
    ap.add_argument("--beta", type=float, default=1.0, help="value MSE weight (≥1, design §1)")
    ap.add_argument("--lam", type=float, default=0.0855, help="pinned λ₀ (static-floor rate)")
    ap.add_argument("--explore-plies", type=int, default=4,
                    help="sample executed action from π′ for the first this-many plies")
    ap.add_argument("--eval-n", type=int, default=200, help="held-out eval episodes/iter")
    ap.add_argument("--eval-seed", type=int, default=12345)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--init-weights", type=str, default=None,
                    help="E-DECIDE value-net npz to warm-start the value head + trunk")
    ap.add_argument("--tb-logdir", type=str, default=None)
    ap.add_argument("--ckpt-dir", type=str, required=True)
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
