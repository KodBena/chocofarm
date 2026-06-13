#!/usr/bin/env python3
"""
chocofarm AZ — train the value head; report held-out R² / MAE (design §9 Stage-1, Gate 1).

Loads a (X, y) dataset (from `dataset.py`), splits train/held-out, standardizes the value
target from the TRAIN split only (mean/std stored in the net for inference), and runs manual-
Adam SGD on the value MSE (`mlp.ValueMLP`). Reports held-out R² and MAE — both on the RAW
(de-standardized) return scale, so the numbers are directly interpretable.

  **Decision Gate 1 (design §9):** is V_λ learnable to decent R² from the §2.2 features on the
  honest 44-face env? A clearly-positive held-out R² re-confirms the doc's F6 marginal-
  sufficiency claim ON THE HONEST MODEL (F6 was measured on the stale 16-region model). A near-
  zero / negative R² says the featurization or the premise is wrong — investigate before any
  search-in-the-loop work.

CLI: python -m chocofarm.az.train_value --data d.npz --out w.npz [--epochs E] [--batch B]
     [--lr LR] [--l2 L2] [--val-frac F] [--seed S] [--hidden H]
Pin to core 3 under timeout (see docs/results/az-edecide.md).
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from chocofarm.az.mlp import ValueMLP


def r2_score(y_true, y_pred):
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def train(X, y, epochs, batch, lr, l2, val_frac, seed, hidden, writer=None):
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    perm = rng.permutation(n)
    X, y = X[perm], y[perm]
    n_val = max(1, int(n * val_frac))
    Xtr, ytr = X[n_val:], y[n_val:]
    Xva, yva = X[:n_val], y[:n_val]

    # standardize the target from the TRAIN split only (no leakage)
    y_mean, y_std = float(ytr.mean()), float(ytr.std())
    net = ValueMLP(X.shape[1], hidden=hidden, n_actions=None, seed=seed,
                   y_mean=y_mean, y_std=y_std)

    n_tr = Xtr.shape[0]
    steps_per_epoch = max(1, n_tr // batch)
    t0 = time.time()
    for ep in range(epochs):
        idx = rng.permutation(n_tr)
        ep_loss = 0.0
        for s in range(steps_per_epoch):
            b = idx[s * batch:(s + 1) * batch]
            if len(b) == 0:
                continue
            ep_loss += net.train_step_value(Xtr[b].astype(np.float64),
                                             ytr[b].astype(np.float64), lr, l2)
        do_print = (ep + 1) % max(1, epochs // 5) == 0 or ep == 0
        if writer is not None or do_print:
            pv = net.predict_value(Xva.astype(np.float64))
            r2 = r2_score(yva, pv)
            mae = float(np.mean(np.abs(yva - pv)))
            if writer is not None:
                writer.add_scalar("value/train_std_mse", ep_loss / steps_per_epoch, ep + 1)
                writer.add_scalar("value/heldout_R2", r2, ep + 1)
                writer.add_scalar("value/heldout_MAE", mae, ep + 1)
                writer.flush()
            if do_print:
                print(f"  epoch {ep + 1:>4}/{epochs}  train_std_mse={ep_loss / steps_per_epoch:.4f}"
                      f"  held-out R²={r2:.4f}  MAE={mae:.4f}  ({time.time() - t0:.0f}s)",
                      flush=True)

    pv = net.predict_value(Xva.astype(np.float64))
    return net, r2_score(yva, pv), float(np.mean(np.abs(yva - pv))), (y_mean, y_std)


def main():
    ap = argparse.ArgumentParser(description="Train the AZ value head; report held-out R²/MAE.")
    ap.add_argument("--data", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--l2", type=float, default=1e-4)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--tb-logdir", type=str, default=None,
                    help="if set, stream per-epoch train_mse / held-out R² / MAE to this TB logdir")
    args = ap.parse_args()

    z = np.load(args.data, allow_pickle=False)
    X, y = z["X"], z["y"]
    print(f"dataset: {X.shape[0]} transitions × {X.shape[1]} feats", flush=True)
    writer = None
    if args.tb_logdir:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter(args.tb_logdir)
        print(f"streaming training curve -> {args.tb_logdir}", flush=True)
    net, r2, mae, (ym, ys) = train(X, y, args.epochs, args.batch, args.lr, args.l2,
                                   args.val_frac, args.seed, args.hidden, writer=writer)
    if writer is not None:
        writer.close()
    net.save(args.out)
    print(f"\nFINAL held-out R²={r2:.4f}  MAE={mae:.4f}  (target mean={ym:.4f} std={ys:.4f})",
          flush=True)
    print(f"saved value net -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
