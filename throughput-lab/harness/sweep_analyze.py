#!/usr/bin/env python3
"""
throughput-lab/harness/sweep_analyze.py — turn a finished sweep (cells/*.json) into a plain,
readable answer to "which serving knob drives throughput, and by how much". Two views, deliberately:

  1. MARGINAL MEANS (assumption-light): the mean served req/s at each level of each knob, averaged
     over everything else. No model — just "rows=16 averaged 95k req/s, rows=1 averaged 40k". This is
     the robust view; trust it most.
  2. OLS REGRESSION (the requested 'regression'): served_req_s ~ the knobs (treatment-coded), with
     coefficients, standard errors, t-stats, p-values, R², and a per-knob share-of-variance. Built by
     hand from numpy + scipy (no black box) so every number is auditable.

It is a FIRST-ORDER summary, NOT a mechanistic model: throughput SATURATES and the knobs INTERACT, so
the coefficients are average tendencies, not laws. The leaderboard in REPORT.md is the WITNESS (the
actual best config measured); this file is the EXPLANATION of the trend behind it (ADR-0009: an
analysis is an interpretation reported next to the measurement, never laundered into a proof).

Only DECOUPLED cells feed the throughput regression (coupled mode is latency-bound — a different
regime; mixing them would muddy every coefficient).

Run:  PYTHONPATH=throughput-lab python harness/sweep_analyze.py <sweep-outdir>

Public Domain (The Unlicense).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from sweep_common import load_ledgers

# The serving-strategy knobs (the columns run_lab records per cell) and their short display names.
FACTORS = ["topology", "threads", "rows_per_batch", "max_batch", "rate_hz_per_thread"]
PRETTY = {
    "topology": "topology", "threads": "threads", "rows_per_batch": "rows",
    "max_batch": "max_batch", "rate_hz_per_thread": "rate",
}


def load_long(outdir: Path) -> pd.DataFrame:
    """One row per (decoupled, VALID cell x replicate): the knobs + that replicate's served req/s.
    Validity is the SHARED `CellLedger.is_valid` verdict (rows conserved, replies complete modulo the
    decoupled async tail) — the SAME rule the leaderboard uses, so the regression and the report can
    never disagree about which cells count. Replicate level (not the per-cell median) gives the
    regression honest within-cell noise to estimate standard errors against."""
    rows: "list[dict]" = []
    for L in load_ledgers(outdir):
        if L.get("mode") != "decoupled" or not L.is_valid:
            continue
        for r in L.served_replicates:                  # SERVED (completed round-trips), not the send rate
            rows.append({**{f: L.get(f) for f in FACTORS}, "served_hz": r})
    return pd.DataFrame(rows)


def _design(df: pd.DataFrame, factors: "list[str]") -> "tuple[np.ndarray, list[str]]":
    """Treatment (dummy) coding: intercept + one column per non-baseline level of each factor that VARIES
    (a constant factor contributes nothing and is dropped). Baseline = the sorted-first level."""
    cols: "dict[str, np.ndarray]" = {"(intercept)": np.ones(len(df))}
    for f in factors:
        levels = sorted(df[f].dropna().unique())
        if len(levels) < 2:
            continue
        base = levels[0]
        for lv in levels[1:]:
            cols[f"{PRETTY[f]}={lv} (vs {base})"] = (df[f] == lv).astype(float).to_numpy()
    return np.column_stack(list(cols.values())), list(cols.keys())


def _ols(X: np.ndarray, y: np.ndarray) -> dict:
    """Plain OLS by hand: beta = lstsq; SE from sigma^2 (X'X)^-1; t = beta/SE; two-sided p from Student-t."""
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    n, p = X.shape
    dof = max(n - p, 1)
    rss = float(resid @ resid)
    tss = float(((y - y.mean()) ** 2).sum())
    sigma2 = rss / dof
    xtx_inv = np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.maximum(np.diag(sigma2 * xtx_inv), 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(se > 0, beta / se, 0.0)
    pval = 2.0 * stats.t.sf(np.abs(t), dof)
    r2 = 1.0 - rss / tss if tss > 0 else 0.0
    adj = 1.0 - (rss / dof) / (tss / (n - 1)) if (tss > 0 and n > 1) else 0.0
    return dict(beta=beta, se=se, t=t, p=pval, r2=r2, adj=adj, n=n, params=p, dof=dof)


def _varexp(df: pd.DataFrame, factors: "list[str]", full: dict) -> "list[tuple[str, float]]":
    """Per-factor share of variance: R^2(full) - R^2(full minus that factor). On a balanced factorial
    this is a clean 'how much does this knob explain on its own' read."""
    y = df["served_hz"].to_numpy(float)
    out: "list[tuple[str, float]]" = []
    for f in factors:
        if df[f].nunique() < 2:
            continue
        Xr, _ = _design(df, [g for g in factors if g != f])
        red = _ols(Xr, y)
        out.append((PRETTY[f], max(full["r2"] - red["r2"], 0.0)))
    out.sort(key=lambda kv: -kv[1])
    return out


def _fmt_pct(x: float) -> str:
    return f"{x * 100:.0f}%"


def build(outdir: Path) -> str:
    df = load_long(outdir)
    L: "list[str]" = ["# throughput-lab sweep — regression analysis (decoupled / throughput)\n"]
    L.append(
        "_What this is:_ a plain **OLS linear regression** of served throughput — completed round-trips/s "
        "(`replies_recv/seconds`), NOT the producer send rate — on the serving "
        "knobs, plus assumption-light **marginal means**. It answers *which knob moves throughput and by "
        "how much, on average*. It is a first-order summary, **not** a mechanistic model — throughput "
        "saturates and knobs interact, so read effects as average tendencies, not laws. The leaderboard "
        "in `REPORT.md` is the witness (the actual best config); this explains the trend behind it.\n"
    )
    if df.empty or len(df) < 4:
        L.append(f"\n**Insufficient data** for a regression (only {len(df)} decoupled replicate-rows "
                 f"found). Run a wider/deeper grid, then re-run this analysis.\n")
        return "\n".join(L) + "\n"

    varying = [f for f in FACTORS if df[f].nunique() >= 2]
    constant = [f for f in FACTORS if df[f].nunique() < 2]
    L.append(f"\nData: **{len(df)} decoupled replicate-observations** across "
             f"{df.groupby(FACTORS).ngroups} cells. Response = served req/s. "
             f"Varying knobs: {', '.join(PRETTY[f] for f in varying) or 'none'}."
             + (f" Held constant: {', '.join(f'{PRETTY[f]}={sorted(df[f].unique())[0]}' for f in constant)}."
                if constant else ""))

    y = df["served_hz"].to_numpy(float)
    X, terms = _design(df, varying)
    fit = _ols(X, y)
    ve = _varexp(df, varying, fit)

    # ---- 1. share of variance (which knob matters most) ----
    L.append("\n## 1. Which knob matters most (share of variance explained)\n")
    if ve:
        L.append("| knob | variance explained |")
        L.append("| --- | ---: |")
        for name, share in ve:
            L.append(f"| {name} | {_fmt_pct(share)} |")
        top = ve[0]
        L.append(f"\n**{top[0]}** explains the most ({_fmt_pct(top[1])} of the spread in throughput). "
                 "Higher = this knob, on its own, accounts for more of why throughput differs across cells.")
    else:
        L.append("_(no varying knobs to attribute variance to)_")

    # ---- 2. marginal means (robust) ----
    L.append("\n## 2. Average throughput by knob level (marginal means — no model assumptions)\n")
    L.append("Mean served req/s at each level of each knob, averaged over all other knobs. This is the "
             "robust view — trust it most.\n")
    L.append("| knob | level | mean req/s | std | n |")
    L.append("| --- | ---: | ---: | ---: | ---: |")
    for f in varying:
        g = df.groupby(f)["served_hz"].agg(["mean", "std", "count"]).reset_index()
        for _, row in g.iterrows():
            std = 0.0 if pd.isna(row["std"]) else row["std"]
            L.append(f"| {PRETTY[f]} | {row[f]} | {row['mean']:,.0f} | {std:,.0f} | {int(row['count'])} |")

    # ---- 3. OLS coefficients (the requested regression) ----
    L.append("\n## 3. OLS coefficients (the regression)\n")
    baselines = []
    for f in varying:
        baselines.append(f"{PRETTY[f]}={sorted(df[f].unique())[0]}")
    L.append(f"Model: `served_req_s ~ {' + '.join(PRETTY[f] for f in varying)}` (treatment-coded). "
             f"Baseline cell = {', '.join(baselines)}. "
             f"**R² = {fit['r2']:.2f}** (the model explains {_fmt_pct(fit['r2'])} of the variation in "
             f"throughput across cells), adjusted R² = {fit['adj']:.2f}, n = {fit['n']}.\n")
    L.append("| term | estimate (req/s) | std err | t | p-value | |")
    L.append("| --- | ---: | ---: | ---: | ---: | --- |")
    for name, b, se, t, p in zip(terms, fit["beta"], fit["se"], fit["t"], fit["p"]):
        sig = "✓ real" if p < 0.05 else "? noise" if name != "(intercept)" else ""
        L.append(f"| {name} | {b:,.0f} | {se:,.0f} | {t:.1f} | {p:.3f} | {sig} |")
    L.append("\n_Reading a row:_ `estimate` = how many req/s that level adds vs the baseline, holding the "
             "other knobs fixed. `(intercept)` = the baseline cell's predicted throughput. `p < 0.05` "
             "(✓) ≈ 'unlikely to be just noise'.")

    # ---- 4. plain-language takeaway ----
    L.append("\n## 4. Plain-language takeaway\n")
    bullets: "list[str]" = []
    for f in varying:
        g = df.groupby(f)["served_hz"].mean()
        best_lvl, worst_lvl = g.idxmax(), g.idxmin()
        delta = g.max() - g.min()
        if best_lvl == worst_lvl:
            continue
        bullets.append(f"- **{PRETTY[f]}**: best at `{best_lvl}` (~{g.max():,.0f} req/s), worst at "
                       f"`{worst_lvl}` (~{g.min():,.0f}) — a ~{delta:,.0f} req/s swing on average.")
    # order bullets by their swing (largest lever first)
    swings = []
    for f in varying:
        g = df.groupby(f)["served_hz"].mean()
        swings.append((f, g.max() - g.min()))
    swings.sort(key=lambda kv: -kv[1])
    L.append(f"The levers, biggest first: " + ", ".join(f"**{PRETTY[f]}**" for f, _ in swings) + ".\n")
    L += bullets
    # best observed cell (echo the leaderboard's #1 for convenience)
    cellmean = df.groupby(varying)["served_hz"].mean().reset_index().sort_values("served_hz", ascending=False)
    if not cellmean.empty:
        top = cellmean.iloc[0]
        cfg = ", ".join(f"{PRETTY[f]}={top[f]}" for f in varying)
        L.append(f"\n- **Best observed config** (highest mean over its replicates): {cfg} "
                 f"→ ~{top['served_hz']:,.0f} req/s. Confirm before citing — this analysis explains the "
                 f"trend, it does not *prove* an optimum (see the caveats).")

    # ---- 5. caveats ----
    L.append("\n## 5. Caveats (so you don't over-read it)\n")
    L.append(
        "- **OLS assumes effects ADD UP and are LINEAR.** Real throughput saturates (a knob stops helping "
        "past a point) and knobs interact (e.g. coalescing may help more at high thread counts). So a "
        "coefficient is an *average* effect, not a guaranteed one — check the marginal means and the "
        "leaderboard for the actual shape.\n"
        "- **`rate` is mostly a saturation flag, not a lever.** Where the requested rate over-saturates "
        "the server, served throughput is the *server's* ceiling and rate barely moves it; a `rate` "
        "coefficient near zero means 'we pushed hard enough to saturate', which is what you want.\n"
        "- **Small n.** Few replicates per cell ⇒ approximate standard errors. Raise `REPLICATES` / "
        "`SECONDS_PER` to tighten them before citing a marginal effect.\n"
        "- **Coupled mode is excluded** — it is RTT-bound (a latency regime), not a throughput one.\n"
        "- This is an **observation from this grid on this 4-vCPU guest**, not a proven optimum or a "
        "universal law. It is a lead to confirm, not a settled bound."
    )
    return "\n".join(L) + "\n"


def main(outdir_s: str) -> int:
    outdir = Path(outdir_s)
    if not (outdir / "cells").is_dir():
        print(f"sweep_analyze.py: no cells/ under {outdir}", file=sys.stderr)
        return 1
    sys.stdout.write(build(outdir))
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: sweep_analyze.py <sweep-outdir>", file=sys.stderr)
        raise SystemExit(2)
    raise SystemExit(main(sys.argv[1]))
