#!/usr/bin/env python3
"""
chocofarm/az/cpp_actor_loop.py — a minimal AZ ExIt loop driven by the C++ Gumbel ACTOR.

This is the goal-2 assembly: it turns the generate -> train -> publish cycle with the **C++** runner as
the self-play actor (vs exit_loop.py, which generates via the Python worker pool). All the pieces it
composes are already built + verified:
  * the C++ actor: `chocofarm-cpp-runner --policy gumbel` reads the published weights (the weight-read
    seam), runs the Gumbel-AZ search, and writes the four (X, PI, M, Y) transition blocks to redis —
    with PI the REAL improved-π (Policy::decide_target), verified end to end;
  * the learner: `JaxTrainer.train_step` (the same JAX/optax trainer exit_loop uses);
  * the weight broadcast: `pack_net` + `weight_keys` to the transport redis (the SAME keys the runner's
    read_weights reads), so the actor at iteration `it` plays against the net trained through `it-1`;
  * TensorBoard (tensorboardX), streaming to tb/az so the run shows up on the daemon (:6006).

It is DELIBERATELY minimal — no held-out eval / replay-window / Part-B value target / parallel
worker-pool (those live in exit_loop.py). For the FULL ExIt run with the C++ actor, prefer the SWAP:
`chocofarm/az/cpp_executor.CppActorExecutor` injects the C++ Gumbel actor into exit_loop's GENERATION
step (`python -m chocofarm.az.exit_loop --cpp-runner cpp/build/chocofarm-cpp-runner`), so the held-out
eval, replay window, JAX training, checkpointing, and hp registry are all inherited unchanged. This
standalone loop remains the minimal, dependency-light demonstration: its job is to show the C++ actor IN
the loop, turning and streaming, so the C++ runtime + the production pool have a real actor-learner home
to plug into. The value target Y is the runner's own λ-return (the pure-MC limit); the net's y_mean/y_std
are re-pinned to the buffer's Y stats each iter (exit_loop's standardization discipline) so the value MSE
stays O(1).

Run:  PYTHONPATH=. python -m chocofarm.az.cpp_actor_loop --runner cpp/build/chocofarm-cpp-runner \
          --instance chocofarm/data/instance.json --faces chocofarm/data/faces.json \
          --iters 6 --episodes 16 --n-sims 16 --hidden 64 --lam 0.0855 \
          --tb-logdir /home/bork/w/vdc/chocobo/tb/az/cpp-actor-loop

Public Domain (The Unlicense).
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time

import numpy as np
import redis

from chocofarm.az.actions import n_action_slots
from chocofarm.az.features import feature_dim
from chocofarm.az.mlp import ValueMLP
from chocofarm.az.mlp_jax_train import JaxTrainer
from chocofarm.az.result_spec import RESULT_DTYPE
from chocofarm.az.transport import pack_net, result_keys, weight_keys
from chocofarm.config import transport_redis_params
from chocofarm.model.env import Environment


def _read_episode(conn, tok: str, idx: int, in_dim: int, ns: int):
    """Read one episode's (X, PI, M, Y) blocks the C++ actor wrote, deriving n from the Y block length
    (every block is contiguous little-endian float32 — result_spec). Returns None if the episode is
    absent (idx ran past the actor's output)."""
    xk, pik, mk, yk = result_keys(tok, idx)
    xb, pib, mb, yb = conn.get(xk), conn.get(pik), conn.get(mk), conn.get(yk)
    if yb is None or xb is None:
        return None
    n = len(yb) // RESULT_DTYPE.itemsize
    if n == 0:
        return None
    X = np.frombuffer(xb, dtype=RESULT_DTYPE).reshape(n, in_dim)
    PI = np.frombuffer(pib, dtype=RESULT_DTYPE).reshape(n, ns)
    M = np.frombuffer(mb, dtype=RESULT_DTYPE).reshape(n, ns)
    Y = np.frombuffer(yb, dtype=RESULT_DTYPE)
    return X, PI, M, Y


def main() -> int:
    ap = argparse.ArgumentParser(description="Minimal AZ ExIt loop driven by the C++ Gumbel actor.")
    ap.add_argument("--runner", required=True, help="path to chocofarm-cpp-runner")
    ap.add_argument("--instance", required=True)
    ap.add_argument("--faces", required=True)
    ap.add_argument("-I", "--iters", type=int, default=6)
    ap.add_argument("-E", "--episodes", type=int, default=16)
    ap.add_argument("--n-sims", type=int, default=16)
    ap.add_argument("--gumbel-max-depth", type=int, default=12)
    ap.add_argument("--epochs", type=int, default=2, help="train_step passes over the buffer/iter")
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--l2", type=float, default=1e-4)
    ap.add_argument("--lam", type=float, default=0.0855)
    ap.add_argument("--max-steps", type=int, default=40)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--run", type=str, default="cpp-actor-loop")
    ap.add_argument("--tb-logdir", type=str, default=None)
    args = ap.parse_args()

    env = Environment()
    in_dim, ns = feature_dim(env), n_action_slots(env)
    net = ValueMLP(in_dim, hidden=args.hidden, n_actions=ns, seed=args.seed,
                   y_mean=0.0, y_std=1.0, residual=False)
    trainer = JaxTrainer(net, lr=args.lr, l2=args.l2)
    conn = redis.Redis(**transport_redis_params())

    writer = None
    if args.tb_logdir:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter(args.tb_logdir)
        print(f"streaming to TensorBoard -> {args.tb_logdir}", flush=True)

    print(f"C++-actor AZ loop: env in_dim={in_dim} n_slots={ns}; net hidden={args.hidden}; "
          f"actor={args.runner}; iters={args.iters} episodes={args.episodes} n_sims={args.n_sims}",
          flush=True)

    for it in range(args.iters):
        t0 = time.time()
        # 1. publish the current net's weights (the actor reads these via the weight-read seam).
        manifest, blob = pack_net(net)
        mk, bk = weight_keys(args.run, "gen", it)
        conn.set(mk, manifest)
        conn.set(bk, blob)

        # 2. the C++ Gumbel ACTOR generates E episodes against those weights.
        tok = f"{args.run}-{it}"
        gen = subprocess.run(
            [args.runner, "--instance", args.instance, "--faces", args.faces,
             "--run", args.run, "--phase", "gen", "--version", str(it), "--res-token", tok,
             "--episodes", str(args.episodes), "--lam", str(args.lam), "--max-steps", str(args.max_steps),
             "--seed", str(args.seed + it), "--policy", "gumbel",
             "--gumbel-n-sims", str(args.n_sims), "--gumbel-max-depth", str(args.gumbel_max_depth)],
            capture_output=True, text=True, timeout=1800)
        if gen.returncode != 0:
            sys.stderr.write(gen.stdout + gen.stderr)
            print(f"iter {it}: RUNNER FAILED (rc={gen.returncode})")
            return 1
        t_gen = time.time() - t0

        # 3. read the transitions the actor wrote (concatenate the episodes into one buffer).
        blocks = [b for idx in range(args.episodes) if (b := _read_episode(conn, tok, idx, in_dim, ns))]
        if not blocks:
            print(f"iter {it}: no transitions produced — aborting")
            return 1
        X = np.concatenate([b[0] for b in blocks]).astype(np.float32)
        PI = np.concatenate([b[1] for b in blocks]).astype(np.float32)
        M = np.concatenate([b[2] for b in blocks]).astype(np.float32)
        Y = np.concatenate([b[3] for b in blocks]).astype(np.float32)

        # 4. train. Re-pin the net's value standardization to this buffer's Y stats (exit_loop's
        #    discipline — the trainer reads y_mean/y_std off the net per step, so the re-pin propagates).
        net.y_mean = float(Y.mean())
        net.y_std = float(max(Y.std(), 1e-3))
        ce = vl = 0.0
        for _ in range(args.epochs):
            ce, vl = trainer.train_step(X, PI, M, Y)
        t_iter = time.time() - t0

        # 5. stream TensorBoard + log.
        if writer is not None:
            writer.add_scalar("gen/transitions", len(Y), it)
            writer.add_scalar("gen/mean_value_target", float(Y.mean()), it)
            writer.add_scalar("gen/value_target_std", float(Y.std()), it)
            writer.add_scalar("train/policy_CE", float(ce), it)
            writer.add_scalar("train/value_MSE", float(vl), it)
            writer.flush()
        print(f"iter {it:2d}/{args.iters}  tr={len(Y):4d}  CE={ce:.3f}  vMSE={vl:.3f}  "
              f"meanY={Y.mean():+.3f}  [gen {t_gen:.1f}s iter {t_iter:.1f}s]", flush=True)

    if writer is not None:
        writer.close()
    print(f"DONE {args.iters} iters (C++ Gumbel actor -> JaxTrainer). weights at az:w:{args.run}:gen:*")
    return 0


if __name__ == "__main__":
    sys.exit(main())
