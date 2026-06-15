#!/usr/bin/env python3
"""
chocofarm AZ — feature-response diagnostic: permutation importance + partial dependence.

The maintainer wants to inspect what the trained net's VALUE head actually reads. Two bounded,
numpy-only probes on a held-out feature set:

  * PERMUTATION IMPORTANCE — for each feature, shuffle that column across the held-out set and
    measure the DROP in value R² (Breiman 2001). A large drop = the value head leans on that
    feature. Grouped by the §2.2 blocks (per-treasure marg/collected/available/dist, per-detector
    informative/p_pos/dist, global) so the table is readable rather than 220 raw rows.
  * PARTIAL DEPENDENCE — for the most important handful of features, sweep that one feature
    across its observed range (holding the rest at their held-out values) and report the mean
    predicted value at each grid point (Friedman 2001). A 1-D slice of the value surface.

Held-out set: pass a dataset npz (the same `X` shape the net was trained on — from
`dataset.py` or a fresh roll). Value targets `y` are used only to compute the baseline R²; the
sweep needs the net only. Bounded: importance is one forward pass per feature (220 passes over
the held-out set); partial dependence is `grid` passes per swept feature. All numpy, no deps.

CLI: python -m chocofarm.az.feature_response --weights w.npz --data d.npz [--top K] [--grid G]
       [--out report.json]
Pin to a free core under timeout.
"""
from __future__ import annotations

import argparse
import json
from typing import Any, cast

import numpy as np
import numpy.typing as npt

from chocofarm.model.env import Environment
from chocofarm.az.mlp import ValueMLP
from chocofarm.az.features import feature_dim, FeatureLayout


def feature_names(env: Environment) -> tuple[list[str], list[str]]:
    """Human-readable name + block tag for each of the `feature_dim(env)` features, in layout
    order. Reads both straight from `FeatureLayout` (the one owner of the §2.2 layout, audit R6),
    so the names/tags can no longer drift from the builder's write order."""
    layout = FeatureLayout(env)
    names = layout.element_names()
    blocks = layout.block_tags()
    assert len(names) == feature_dim(env), (len(names), feature_dim(env))
    return names, blocks


def r2_score(y_true: npt.NDArray[Any], y_pred: npt.NDArray[Any]) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def _batch_values(net: ValueMLP, X2d: npt.NDArray[Any]) -> npt.NDArray[Any]:
    """`net.predict_value` over a 2-D batch always returns the (B,) value array (the `float` arm of
    its return is only the 1-D-input case); the cast states that batch contract (no runtime change)."""
    return cast("npt.NDArray[Any]", net.predict_value(X2d))


def permutation_importance(net: ValueMLP, X: npt.NDArray[Any], y: npt.NDArray[Any],
                           rng: np.random.Generator) -> tuple[float, npt.NDArray[np.float64]]:
    """Drop in value R² when each feature column is shuffled. Returns (base_r2, drops[d])."""
    base = r2_score(y, _batch_values(net, X.astype(np.float64)))
    d = X.shape[1]
    drops = np.zeros(d)
    for j in range(d):
        Xp = X.copy()
        Xp[:, j] = Xp[rng.permutation(X.shape[0]), j]
        r2p = r2_score(y, _batch_values(net, Xp.astype(np.float64)))
        drops[j] = base - r2p
    return base, drops


def partial_dependence(net: ValueMLP, X: npt.NDArray[Any], j: int, grid: int
                       ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Mean predicted value as feature `j` sweeps its observed [min,max] over `grid` points,
    holding all other features at their held-out values. Returns (grid_values, mean_preds)."""
    lo, hi = float(X[:, j].min()), float(X[:, j].max())
    if hi <= lo:
        hi = lo + 1e-6
    gv = np.linspace(lo, hi, grid)
    means = np.zeros(grid)
    for k, val in enumerate(gv):
        Xp = X.copy()
        Xp[:, j] = val
        means[k] = float(np.mean(net.predict_value(Xp.astype(np.float64))))
    return gv, means


def run(args: argparse.Namespace) -> None:
    env = Environment()
    net = ValueMLP.load(args.weights)
    z = np.load(args.data, allow_pickle=False)
    X, y = z["X"].astype(np.float64), z["y"].astype(np.float64)
    names, blocks = feature_names(env)
    print(f"held-out set: {X.shape[0]} × {X.shape[1]} feats", flush=True)

    rng = np.random.default_rng(args.seed)
    base_r2, drops = permutation_importance(net, X, y, rng)
    print(f"baseline value R² = {base_r2:.4f}\n", flush=True)

    order = np.argsort(drops)[::-1]
    print(f"{'rank':>4} {'feature':>22} {'block':>22} {'ΔR² (importance)':>18}", flush=True)
    for r, j in enumerate(order[:args.top]):
        print(f"{r + 1:>4} {names[j]:>22} {blocks[j]:>22} {drops[j]:>18.5f}", flush=True)

    # per-block aggregate (sum of drops within each §2.2 block)
    block_sum: dict[str, float] = {}
    for j, b in enumerate(blocks):
        block_sum[b] = block_sum.get(b, 0.0) + drops[j]
    print(f"\n{'block':>26} {'Σ ΔR²':>12}", flush=True)
    for b, s in sorted(block_sum.items(), key=lambda kv: -kv[1]):
        print(f"{b:>26} {s:>12.5f}", flush=True)

    # partial dependence for the top-few
    pd: dict[str, dict[str, list[float]]] = {}
    print(f"\npartial dependence (top {min(args.top, 5)} features):", flush=True)
    for j in order[:min(args.top, 5)]:
        gv, means = partial_dependence(net, X, int(j), args.grid)
        pd[names[j]] = {"grid": gv.tolist(), "mean_value": means.tolist()}
        span = means.max() - means.min()
        print(f"  {names[j]:>22}: value swept {means[0]:+.3f} -> {means[-1]:+.3f} "
              f"(span {span:.3f}) over [{gv[0]:.3f}, {gv[-1]:.3f}]", flush=True)

    if args.out:
        report = {
            "baseline_r2": base_r2,
            "ranked": [{"feature": names[int(j)], "block": blocks[int(j)],
                        "importance": float(drops[int(j)])} for j in order],
            "block_importance": block_sum,
            "partial_dependence": pd,
        }
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nwrote ranked report -> {args.out}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="AZ value-head feature-response diagnostic.")
    ap.add_argument("--weights", type=str, required=True, help="trained net npz")
    ap.add_argument("--data", type=str, required=True, help="held-out dataset npz (X, y)")
    ap.add_argument("--top", type=int, default=20, help="rows to print / sweep")
    ap.add_argument("--grid", type=int, default=11, help="partial-dependence grid points")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default=None, help="optional JSON report path")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
