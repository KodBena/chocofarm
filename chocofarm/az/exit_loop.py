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
from chocofarm.az.value_target import blended_returns_to_go

# reference lines (docs/results/voi-ceiling-2026-06-13.md, decomp-rate.md)
STATIC_FLOOR = 0.0855
CLAIRVOYANT_CEIL = 0.1454
DECOMP_ANCHOR = 0.0941


def r2_score(y_true, y_pred):
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def generate_episode(env, search, fb, world, lam, rng, n_explore_plies, max_steps=40,
                     lam_blend=1.0, n_step=None):
    """One self-play episode driven by the Gumbel search. Records, per decision,
    (features, improved-pi, legal-mask, value-target). The value target is the Part B blended
    return-to-go (lower-variance TD(λ)/n-step over the realized λ-return and the search's root-value
    bootstrap); with `lam_blend=1.0, n_step=None` it is the prior pure-MC suffix rule, bit-identical.

    `n_explore_plies`: sample the EXECUTED action from π′ for the first this-many plies
    (temperature 1), argmax thereafter (design §6: temperature on executed action to diversify
    trajectories). The improved-policy TARGET is always the full π′ regardless.

    `lam_blend` / `n_step`: the Part B value-target knobs (mutually exclusive; n_step takes
    precedence if set). λ_blend=1 / n=∞ → pure MC (prior behavior). See value_target.py."""
    loc, bw, collected = ("w", env.entry), env.worlds, set()
    feats, pis, masks, step_rt, boots = [], [], [], [], []
    for ply in range(max_steps):
        if len(bw) == 0:
            break
        temp = 1.0 if ply < n_explore_plies else 0.0
        # decide_with_value also returns the search's ~n_sims-averaged root-value bootstrap for THIS
        # belief (Part B) — the value of the state we decided from, used by the lower-variance target.
        action, pi, boot = search.decide_with_value(env, loc, bw, collected, lam, rng,
                                                    temperature=temp)
        # the node the search just evaluated cached feat+mask for THIS belief; rebuild cheaply
        # (marginals are served by build's per-belief cache, so we don't pre-compute them here)
        feat = fb.build(loc, bw, collected)
        mask = legal_mask_from_features(env, feat)
        if action == TERMINATE:
            # the TERMINATE decision executes no step; its target is the continuation (exit toll),
            # which the blend supplies as boot_term. We record it with the MC tail value below by
            # NOT appending a step — but we DO want a training target for the terminate state. Keep
            # the prior behavior: the terminate decision's target is the pure exit-toll continuation
            # (same as boot for an empty/terminal belief), so append it as a zero-step tail decision.
            feats.append(feat); pis.append(pi); masks.append(mask); boots.append(boot)
            break
        feats.append(feat); pis.append(pi); masks.append(mask); boots.append(boot)
        r, loc, bw, collected, dt = env.apply(loc, bw, collected, action, world)
        step_rt.append((r, dt))
    exit_c = env.exit_cost(loc)
    n_dec = len(step_rt)            # decisions that executed a non-TERMINATE action
    n_rec = len(feats)             # all recorded decisions (incl. a trailing TERMINATE decision)

    # Value targets. The blend operates over the EXECUTED-step decisions (those with a (r,dt) step);
    # a trailing TERMINATE decision (n_rec > n_dec) has no step — its target is the continuation
    # value = exit toll (the boot_term boundary), matching the prior code's last-decision handling.
    g_steps = blended_returns_to_go(step_rt, boots[:n_dec], exit_c, lam,
                                    lam_blend=lam_blend, n_step=n_step)
    out = []
    for j in range(n_rec):
        if j < n_dec:
            g = g_steps[j]
        else:
            # the TERMINATE decision: continuation value is just the exit toll (no step, no boot)
            g = -lam * exit_c
        out.append((feats[j], pis[j], masks[j], g))
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


def train_epochs(trainer, X, PI, M, Y, epochs, batch, lr, l2, alpha, beta, rng):
    """Adam epochs over the buffer via the JAX/optax trainer (`mlp_jax_train.JaxTrainer`);
    autodiff over the jit'd forward replaces the manual numpy backprop. Re-pins the value
    standardization to the buffer's Y stats (design §3) before training so the MSE stays O(1) as
    the return distribution drifts — the trainer reads y_mean/y_std off the net per step, so the
    re-pin propagates. After training, numpy inference (`predict_value`) reads the trained weights
    the trainer wrote back into the net. Returns (mean_ce, mean_value_std_mse, heldout_R2_on_buffer).

    `lr`/`l2` are fixed at trainer construction (the loop varies neither); they are accepted here to
    keep the prior signature shape but the trainer's configured lr/l2 are authoritative."""
    net = trainer.net
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
            ce, vl = trainer.train_step(X[b], PI[b], M[b], Y[b], alpha=alpha, beta=beta)
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

    # --- net: resume full net (--resume), warm-start value head (--init-weights), or cold ---
    if args.resume:
        net = ValueMLP.load(args.resume)
        if net.in_dim != in_dim or net.n_actions != n_slots:
            raise SystemExit(
                f"--resume net dims (in_dim={net.in_dim}, n_actions={net.n_actions}) "
                f"!= env (in_dim={in_dim}, n_slots={n_slots}) — incompatible checkpoint")
        _res_note = "residual block ON" if net.residual else "no residual block"
        print(f"RESUMED full net (trunk + {_res_note} + value + policy heads) from "
              f"{args.resume} (hidden={net.H}); JaxTrainer inits a fresh optax optimizer", flush=True)
    elif args.init_weights:
        warm = ValueMLP.load(args.init_weights)
        net = ValueMLP(in_dim, hidden=warm.H, n_actions=n_slots, seed=args.seed,
                       y_mean=warm.y_mean, y_std=warm.y_std, residual=args.residual)
        # copy the second trunk layer + value head; policy head stays random.
        net.W2, net.b2 = warm.W2.copy(), warm.b2.copy()
        net.Wv, net.bv = warm.Wv.copy(), warm.bv.copy()
        # The INPUT layer (W1) can only be warm-started if the warm net was trained on the SAME
        # feature dimension. Part C grew feature_dim (220 → 241), so an old-dim E-DECIDE net's W1 is
        # shape-incompatible with the current input. Detect it and KEEP the fresh random W1 (Part C
        # explicitly says "no warm-start of the input layer" when the dim changes) rather than
        # crash deep in the first forward (ADR-0002: fail informative at setup, not opaque later).
        if warm.W1.shape == net.W1.shape:
            net.W1, net.b1 = warm.W1.copy(), warm.b1.copy()
            w1_note = "input layer warm-started"
        else:
            w1_note = (f"input layer RANDOM (warm net in_dim={warm.in_dim} ≠ current {in_dim}; "
                       f"Part C feature change — W1 cannot warm-start)")
        res_note = "residual block ON (random init)" if net.residual else "no residual block"
        print(f"warm-started 2nd-trunk + value head from {args.init_weights} "
              f"(hidden={warm.H}); {w1_note}; policy head random; {res_note}", flush=True)
    else:
        net = ValueMLP(in_dim, hidden=args.hidden, n_actions=n_slots, seed=args.seed,
                       residual=args.residual)
        res_note = "residual block ON" if net.residual else "no residual block"
        print(f"cold net (hidden={args.hidden}); both heads random; {res_note}", flush=True)

    # --- JAX/optax trainer: autodiff training over the jit'd forward (replaces mlp.py's manual
    #     backprop + hand-rolled Adam). Built ONCE so Adam's running moments persist across
    #     iterations, exactly as the numpy net's self.m/self.v/self.t did. It reads the (now
    #     fully-initialised, possibly warm-started) net's weights and writes the trained weights
    #     back into the net after each step — numpy inference (generation/eval) reads them. ---
    from chocofarm.az.mlp_jax_train import JaxTrainer
    trainer = JaxTrainer(net, lr=args.lr, l2=args.l2)
    print(f"training: JAX/optax Adam (lr={args.lr} l2={args.l2}); inference: numpy float32",
          flush=True)

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
    n_step = args.n_step
    lam_blend = args.td_lambda
    if n_step is not None and lam_blend < 1.0:
        raise ValueError("set at most one of --n-step / --td-lambda (the other stays at the "
                         "pure-MC default); both were given")  # ADR-0002 fail-loud
    bmode = (f"n-step={n_step}" if n_step is not None
             else (f"TD(λ_blend={lam_blend})" if lam_blend < 1.0 else "pure-MC (λ_blend=1)"))
    print(f"value target: {bmode}", flush=True)

    # --- Part A: persistent core-pinned process pool (parallel actor/learner). workers<=0 keeps
    #     the in-process serial path (the true serial baseline for the A/B). ---
    executor = None
    if args.workers and args.workers > 0:
        from chocofarm.az.parallel import ParallelExecutor
        cores = [int(c) for c in args.cores.split(",")] if args.cores else None
        executor = ParallelExecutor(args.workers, cores, args.seed, args.m, args.n_sims)
        print(f"parallel actor/learner: {args.workers} workers pinned to cores "
              f"{executor.cores}; weights + transitions over redis (raw bytes, no pickle) "
              f"run={executor.run}", flush=True)
    else:
        print("serial (in-process) generation + eval", flush=True)

    try:
      for it in range(args.iters):
        t0 = time.time()
        # the parent draws the per-episode true worlds so the world sequence is reproducible
        # regardless of worker count (parallel≈serial): same worlds → same episodes given seeds.
        gen_worlds = [int(gen_rng.choice(env.worlds)) for _ in range(args.episodes)]
        eval_rng = np.random.default_rng(args.eval_seed)
        eval_worlds = [int(eval_rng.choice(env.worlds)) for _ in range(args.eval_n)]

        # ---- 1. GENERATE ----
        if executor is not None:
            # publishes the frozen weights to redis (raw bytes) + fans the episodes; gathers
            # transitions back over redis (no pickle of the array payloads)
            recs_all = executor.generate(net, it, gen_worlds, lam,
                                         args.explore_plies, lam_blend, n_step)
        else:
            search = GumbelAZSearch(net, env, m=args.m, n_sims=args.n_sims)
            recs_all = []
            for world in gen_worlds:
                recs_all.extend(generate_episode(env, search, fb, world, lam, gen_rng,
                                                 args.explore_plies,
                                                 lam_blend=lam_blend, n_step=n_step))
        Xs, PIs, Ms, Ys, all_pis = [], [], [], [], []
        for feat, pi, mask, g in recs_all:
            Xs.append(feat); PIs.append(pi); Ms.append(mask); Ys.append(g); all_pis.append(pi)
        X = np.asarray(Xs, dtype=np.float32)
        PI = np.asarray(PIs, dtype=np.float32)
        M = np.asarray(Ms, dtype=np.float32)
        Y = np.asarray(Ys, dtype=np.float32)
        buf.add(X, PI, M, Y)
        ent = policy_entropy(all_pis)
        y_var = float(np.var(Y)) if Y.size else 0.0    # Part B: value-target variance watch
        t_gen = time.time() - t0

        # ---- 2. TRAIN ----
        bX, bPI, bM, bY = buf.arrays()
        ce, vmse, r2 = train_epochs(trainer, bX, bPI, bM, bY, args.epochs, args.batch,
                                    args.lr, args.l2, args.alpha, args.beta, train_rng)
        t_train = time.time() - t0 - t_gen

        # ---- 3. EVALUATE (greedy argmax-π′ policy on a held-out seed, fixed λ₀) ----
        if executor is not None:
            # re-publishes the now-trained weights at a distinct version, fans eval episodes
            totR, totT, ets = executor.evaluate(net, it + 1_000_000, eval_worlds, lam)
        else:
            eval_pol = GumbelPolicy(net, env, m=args.m, n_sims=args.n_sims)
            ev_rng = np.random.default_rng(args.eval_seed)
            totR = totT = 0.0; ets = []
            for w in eval_worlds:
                R, T, _ = env.simulate(eval_pol, w, lam, ev_rng)
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
               "target_var": y_var,
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
            writer.add_scalar("gen/target_var", y_var, it)   # Part B: value-target variance watch
            writer.flush()

        print(f"iter {it:>3}/{args.iters}  rate={rate:.4f} (%VoI={voi:+.0f}) ET={et:.1f}  "
              f"CE={ce:.3f} vMSE={vmse:.3f} R²={r2:.3f} H={ent:.2f} yVar={y_var:.3f}  "
              f"[{X.shape[0]} tr | gen {t_gen:.0f}s train {t_train:.0f}s eval {t_eval:.0f}s]",
              flush=True)
    finally:
        if executor is not None:
            executor.close()

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
    ap.add_argument("--residual", action="store_true",
                    help="insert a residual block (H×H → H×H, skip+ReLU) between the trunk output "
                         "and the two heads. Default OFF → numerically identical to the "
                         "pre-residual net (a clean ablation axis).")
    ap.add_argument("--init-weights", type=str, default=None,
                    help="E-DECIDE value-net npz to warm-start the value head + trunk")
    ap.add_argument("--resume", type=str, default=None,
                    help="full-net npz to RESUME: loads ALL weights (trunk, residual block, value "
                         "AND policy head), resets Adam. Unlike --init-weights, keeps nothing random.")
    # --- Part A: 4-core actor/learner parallelism ---
    ap.add_argument("--workers", type=int, default=4,
                    help="process-pool workers for the generation+eval fan-out, each pinned to a "
                         "distinct core (Part A). 0 = serial in-process (the A/B baseline).")
    ap.add_argument("--cores", type=str, default="0,1,2,3",
                    help="comma-separated cores to pin workers to (Part A; default 0,1,2,3)")
    # --- Part B: lower-variance value target (mutually exclusive; default = pure MC) ---
    ap.add_argument("--td-lambda", type=float, default=1.0,
                    help="TD(λ) blend weight on the value target (Part B): 1.0 = pure MC (current "
                         "behavior), →0 = pure search-root bootstrap. Mutually exclusive with --n-step.")
    ap.add_argument("--n-step", type=int, default=None,
                    help="n-step value target (Part B): realized reward for n steps then bootstrap "
                         "off the search root value. None/∞ = pure MC. Mutually exclusive with --td-lambda.")
    ap.add_argument("--tb-logdir", type=str, default=None)
    ap.add_argument("--ckpt-dir", type=str, required=True)
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
