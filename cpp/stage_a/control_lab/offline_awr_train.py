#!/usr/bin/env python3
"""
cpp/stage_a/control_lab/offline_awr_train.py — STEP 1 of the offline-RL pipeline: TRAIN the AWR issue-gate
policy (methods/offline_awr.AWRRecipe) on a fixed corpus of logged controller trajectories, save the fitted
policy checkpoint + the training curves under ~/w/vdc/chocobo/runs/control_lab/offline-rl/, and VERIFY the
fit (the learned policy actually gates — allow_fraction < 1, not collapsed).

WHAT IT DOES (the one-owner offline-fit driver; it owns ONLY the corpus-build + the fit driving — the AWR
math lives in methods/offline_awr, the DB I/O in lab_store, the codec in trajectory_codec):
  1. Read the trajectory blobs from the control_research postgres for a TARGET corpus, filtered by
     (chunk_floor IS TRUE AND s_min = S) — the depth>1 convoy regime where gating earns its keep — and
     decode each (trajectory_codec). The DEFAULT filter is the brief's convoy corpus: chunk_floor + s_min=1.
  2. Pool the PER-THREAD, ACTIVE-ONLY samples across the behaviors: for every decision and thread with
     inflight>0 (a forced no-op at inflight==0 carries no causal credit — masked, identical to the online
     gates), reconstruct the SAME 5-d phi the runtime act() builds (the served-diff coalescing, vectorized)
     and pair it with the OBSERVED gate and the forward's reward. This is the AWRCorpus.
  3. Run AWRRecipe.fit(corpus) -> a deployable AWRGate (value baseline -> advantage -> advantage-weighted
     policy regression; jax/optax full-batch). Save the checkpoint (.npz) + the training curves (.json) +
     a run summary under the offline-rl artifact dir.
  4. VERIFY: report the fit diagnostics (advantage/weight stats, the learned active allow fraction) and the
     fit's recorded value MSE / policy weighted-BCE curves, and ASSERT the policy actually gates (the
     learned active allow fraction is bounded away from 1.0 — a policy that allows everything has not learned
     the convoy-taming deny-when-active behavior; fail loud per ADR-0002/ADR-0009 if so).

EXCLUDES the degenerate `malfunctioning` reference (it deliberately throws past the watchdog deadline and
logged only a handful of decisions — no usable gating signal). Multiple behavior policies remain (the
off-policy, multi-behavior corpus AWR is built for).

Usage (the interpreter with jax/optax + psycopg3):
    /home/bork/w/vdc/venvs/generic/bin/python cpp/stage_a/control_lab/offline_awr_train.py \
        [--session lab-20260620-190846] [--s-min 1] [--hidden 8] [--temp 1.0] [--w-max 20] \
        [--value-steps 3000] [--policy-steps 3000] [--seed 0] [--out <offline-rl dir>] [--ckpt <name>]

psycopg3 ONLY (via lab_store). Artifacts under ~/w/vdc (NEVER /tmp).

Public Domain (The Unlicense).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

REPO = "/home/bork/w/vdc/1/chocofarm"
if REPO not in sys.path:
    sys.path.insert(0, REPO)
_STAGE_A = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _STAGE_A not in sys.path:
    sys.path.insert(0, _STAGE_A)

from control_lab import lab_store  # noqa: E402  — the one-owner postgres egress (psycopg3)
from control_lab.trajectory_codec import DecodedTrajectory, decode  # noqa: E402
from control_lab.methods.offline_awr import (  # noqa: E402
    DEFAULT_CKPT, AWRCorpus, AWRRecipe, AWRGate, _np_policy_probs, save_checkpoint,
)

DEFAULT_OUT = os.path.join(
    os.path.expanduser("~"), "w", "vdc", "chocobo", "runs", "control_lab", "offline-rl"
)
# the degenerate reference excluded from training (deliberately misbehaves; no gating signal).
_EXCLUDE = {"malfunctioning"}


def _vec_coalesce(dt: DecodedTrajectory) -> np.ndarray:
    """Reconstruct the per-thread instantaneous coalescing degree (Δleaves/Δmsgs) over the trajectory with the
    SAME served-diff logic the runtime act() uses (an absent thread's sentinel-0 is never differenced; an
    un-baselined/quiet thread reads the neutral 1.0). VECTORIZED per thread over the served rows (the runtime
    does it scalar per forward; this is the equivalent batched reconstruction for the offline fit — feature
    parity is the contract). Returns (n, T) float64."""
    n, T = dt.n_decisions, dt.n_threads
    msgs = dt.msgs.astype(np.int64)
    leaves = dt.leaves.astype(np.int64)
    served = dt.served.astype(bool)
    coalesce = np.ones((n, T), dtype=np.float64)
    for t in range(T):
        s_idx = np.nonzero(served[:, t])[0]   # the rows where thread t was served (in time order)
        if s_idx.size < 2:
            continue
        m = msgs[s_idx, t]
        l = leaves[s_idx, t]
        dm = np.diff(m)                       # consecutive served-row deltas (the runtime's per-thread baseline)
        dl = np.diff(l)
        c = np.ones(s_idx.size, dtype=np.float64)   # first served row stays 1.0 (un-baselined, never differenced)
        good = (dm > 0) & (dl >= 0)
        c[1:][good] = dl[good] / dm[good].astype(np.float64)
        coalesce[s_idx, t] = c
    return coalesce


def build_corpus(session: str | None, s_min: int) -> tuple[AWRCorpus, dict]:
    """Read the trajectory blobs for the target convoy corpus from postgres (filtered chunk_floor IS TRUE AND
    s_min=S, optionally pinned to one session), decode them, and pool the per-thread ACTIVE-only samples into
    an AWRCorpus. Returns (corpus, provenance) where provenance records which trials contributed and the
    geometry (a fail-loud check that the corpus is the convoy regime: one T/D/K, chunk_floor on, s_min=S)."""
    conn = lab_store.connect()
    cur = conn.cursor()
    where = "t.chunk_floor IS TRUE AND t.s_min = %s"
    params: list = [s_min]
    if session:
        where += " AND t.session_id = %s"
        params.append(session)
    cur.execute(
        f"""
        SELECT t.method, t.session_id, t.n_threads, t.d_ceiling, t.k_per_thread, b.payload
        FROM lab_trial t
        JOIN lab_blob b ON b.trial_id = t.trial_id AND b.kind = 'trajectory'
        WHERE {where}
        ORDER BY t.session_id, t.method
        """,
        params,
    )
    rows = cur.fetchall()
    conn.close()
    if not rows:
        raise RuntimeError(
            f"offline_awr_train: no trajectory blobs for chunk_floor=TRUE s_min={s_min}"
            f"{(' session=' + session) if session else ''} — nothing to train on (ADR-0002)"
        )

    phis: list[np.ndarray] = []
    acts: list[np.ndarray] = []
    rets: list[np.ndarray] = []
    contributors: list[dict] = []
    geoms: set[tuple[int, int, int]] = set()
    for method, sess_id, T_db, D_db, K_db, payload in rows:
        if method in _EXCLUDE:
            continue
        dt = decode(bytes(payload))
        n, T = dt.n_decisions, dt.n_threads
        D = float(dt.d_ceiling)
        K = float(dt.k_per_thread)
        geoms.add((T, int(D), int(K)))
        inflight = dt.inflight.astype(np.float64)
        ready = dt.ready.astype(np.float64)
        coalesce = _vec_coalesce(dt)
        headroom = np.maximum(1.0, D - inflight)
        # the canonical _FEATURES order (bias last) — IDENTICAL to AWRGate._build_phi / reinforce / a2c.
        phi = np.stack(
            [ready / headroom, ready / K, inflight / D, coalesce, np.ones((n, T))], axis=-1
        )  # (n, T, d_in)
        active = inflight > 0.0                              # the credit mask (forced no-ops carry no credit)
        r = np.broadcast_to(dt.reward[:, None], (n, T))     # the per-forward reward, shared across threads
        sel = active
        phis.append(phi[sel].astype(np.float32))
        acts.append(dt.action[sel].astype(np.float32))
        rets.append(r[sel].astype(np.float32))
        contributors.append({
            "method": method, "session": sess_id, "n_decisions": int(n),
            "active_samples": int(sel.sum()),
            "active_allow_frac": float(dt.action[sel].mean()) if sel.sum() else float("nan"),
            "reward_mean": float(dt.reward.mean()),
        })

    if not phis:
        raise RuntimeError("offline_awr_train: every trajectory was excluded — no training samples (ADR-0002)")
    corpus = AWRCorpus(
        phi=np.concatenate(phis, axis=0),
        action=np.concatenate(acts, axis=0),
        ret=np.concatenate(rets, axis=0),
    )
    prov = {
        "n_active_samples": int(corpus.phi.shape[0]),
        "n_contributing_trials": len(contributors),
        "geometries_TDK": sorted(geoms),
        "corpus_active_allow_frac": float(corpus.action.mean()),
        "corpus_reward_mean": float(corpus.ret.mean()),
        "corpus_reward_std": float(corpus.ret.std()),
        "contributors": contributors,
    }
    # fail loud if the corpus is not a single clean convoy geometry (mixed T/D/K would corrupt the K/D
    # normalizers the features divide by — ADR-0002, refuse to fit a mixed corpus silently).
    if len(geoms) != 1:
        raise RuntimeError(f"offline_awr_train: corpus mixes geometries (T,D,K) {sorted(geoms)} — refuse to "
                           f"pool incommensurable feature normalizers (ADR-0002)")
    return corpus, prov


def main() -> int:
    ap = argparse.ArgumentParser(description="offline-RL step 1: train the AWR issue-gate on the convoy corpus")
    ap.add_argument("--session", default="lab-20260620-190846",
                    help="pin the corpus to one lab_session (default the brief's convoy session); "
                         "empty string = every chunk_floor/s_min-matching session")
    ap.add_argument("--s-min", type=int, default=1, help="the convoy floor to train on (default 1 — the "
                    "regime where gating decisively matters)")
    ap.add_argument("--hidden", type=int, default=8, help="policy/value hidden width (0 = linear)")
    ap.add_argument("--temp", type=float, default=1.0, help="AWR temperature (advantage standardized first)")
    ap.add_argument("--w-max", type=float, default=20.0, help="AWR weight clip ceiling")
    ap.add_argument("--value-steps", type=int, default=3000)
    ap.add_argument("--policy-steps", type=int, default=3000)
    ap.add_argument("--lr-value", type=float, default=0.01)
    ap.add_argument("--lr-policy", type=float, default=0.01)
    ap.add_argument("--init-allow-logit", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=DEFAULT_OUT, help="the offline-rl artifact dir (under ~/w/vdc)")
    ap.add_argument("--ckpt", default=None, help="checkpoint filename within --out (default awr_policy.npz, "
                    "the deploy default the registry factory loads)")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    session = a.session if a.session else None
    ckpt_path = os.path.join(a.out, a.ckpt) if a.ckpt else DEFAULT_CKPT

    print(f"[awr-train] building convoy corpus (chunk_floor=TRUE s_min={a.s_min}"
          f"{', session=' + session if session else ''}) ...", flush=True)
    corpus, prov = build_corpus(session, a.s_min)
    print(f"[awr-train] corpus: {prov['n_active_samples']} active per-thread samples from "
          f"{prov['n_contributing_trials']} trials; geometry (T,D,K)={prov['geometries_TDK'][0]}; "
          f"active allow_frac={prov['corpus_active_allow_frac']:.3f}; "
          f"reward mean={prov['corpus_reward_mean']:.1f}+/-{prov['corpus_reward_std']:.1f}", flush=True)

    print(f"[awr-train] fitting AWR (hidden={a.hidden} temp={a.temp} w_max={a.w_max} "
          f"value_steps={a.value_steps} policy_steps={a.policy_steps} seed={a.seed}) ...", flush=True)
    t0 = time.monotonic()
    recipe = AWRRecipe(
        hidden=a.hidden, temp=a.temp, w_max=a.w_max,
        value_steps=a.value_steps, policy_steps=a.policy_steps,
        lr_value=a.lr_value, lr_policy=a.lr_policy,
        init_allow_logit=a.init_allow_logit, seed=a.seed,
    )
    gate = recipe.fit(corpus)
    fit_s = time.monotonic() - t0
    diag = dict(gate.metrics())   # the AWRGate exposes fit_* diagnostics in metrics()
    fit_diag = {k[len("fit_"):]: v for k, v in diag.items() if k.startswith("fit_")}

    # ---- save the checkpoint (deployable) + the training curves + a self-describing run summary ----
    save_checkpoint(ckpt_path, gate)
    curves_path = os.path.join(a.out, f"awr_train_curves-{stamp}.json")
    with open(curves_path, "w") as f:
        json.dump({"value_mse": recipe.curves["value_mse"],
                   "policy_wbce": recipe.curves["policy_wbce"]}, f, indent=2)
    summary = {
        "stamp": stamp, "session": session, "s_min": a.s_min,
        "hyperparams": {"hidden": a.hidden, "temp": a.temp, "w_max": a.w_max,
                        "value_steps": a.value_steps, "policy_steps": a.policy_steps,
                        "lr_value": a.lr_value, "lr_policy": a.lr_policy,
                        "init_allow_logit": a.init_allow_logit, "seed": a.seed},
        "corpus": prov,
        "fit_diagnostics": fit_diag,
        "fit_seconds": round(fit_s, 2),
        "checkpoint": ckpt_path,
        "curves": curves_path,
    }
    summary_path = os.path.join(a.out, f"awr_train_summary-{stamp}.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # ---- VERIFY the fit (the brief's verification: loss/weights sane; the policy actually gates) ----
    print(f"\n[awr-train] === FIT DIAGNOSTICS (fitted in {fit_s:.1f}s) ===", flush=True)
    print(f"  value MSE: {recipe.curves['value_mse'][0]:.1f} -> {recipe.curves['value_mse'][-1]:.1f} "
          f"({len(recipe.curves['value_mse'])} logged pts)", flush=True)
    print(f"  policy weighted-BCE: {recipe.curves['policy_wbce'][0]:.4f} -> "
          f"{recipe.curves['policy_wbce'][-1]:.4f}", flush=True)
    print(f"  Q-baseline (action-conditional): Q|deny={fit_diag['q_deny_mean']:.1f} "
          f"Q|allow={fit_diag['q_allow_mean']:.1f}  ->  A|deny={fit_diag['adv_deny_mean']:+.1f} "
          f"A|allow={fit_diag['adv_allow_mean']:+.1f}", flush=True)
    print(f"  advantage: mean={fit_diag['adv_mean']:.2f} std={fit_diag['adv_std']:.2f} "
          f"p10={fit_diag['adv_p10']:.1f} p90={fit_diag['adv_p90']:.1f}", flush=True)
    print(f"  AWR weight: mean={fit_diag['weight_mean']:.2f} max={fit_diag['weight_max']:.2f} "
          f"frac_clipped={fit_diag['weight_frac_clipped']:.3f}  | weight MASS on DENY samples="
          f"{fit_diag['weight_mass_deny']:.3f} (obs deny share={fit_diag['obs_deny_share']:.3f})", flush=True)
    laf = fit_diag["learned_active_allow_frac"]
    print(f"  LEARNED active allow fraction: {laf:.3f}  "
          f"(corpus observed active allow_frac was {prov['corpus_active_allow_frac']:.3f})", flush=True)

    # The policy MUST gate: a learned active allow fraction at/near 1.0 means it collapsed to all-allow and
    # learned NOTHING of the convoy-taming deny-when-active behavior (ADR-0009: report the real state; a
    # collapsed fit is a failed fit, surfaced loud rather than deployed). The convoy signal is deny-when-active,
    # so a healthy fit has the learned active allow fraction WELL below 1 (the good heuristics sat at 0.0-0.1).
    verified = laf < 0.95
    print(f"\n[awr-train] VERIFY policy-gates (learned active allow_frac < 0.95): "
          f"{'PASS' if verified else 'FAIL — policy collapsed to all-allow'}", flush=True)
    print(f"[awr-train] checkpoint -> {ckpt_path}", flush=True)
    print(f"[awr-train] curves     -> {curves_path}", flush=True)
    print(f"[awr-train] summary    -> {summary_path}", flush=True)
    if not verified:
        # fail loud: a collapsed policy is not deployable as a convoy gate (ADR-0002 / ADR-0009).
        print("[awr-train] ERROR: the fitted policy does not gate (collapsed to all-allow) — not a usable "
              "convoy controller. Re-examine temp/w_max/steps before deploying.", file=sys.stderr, flush=True)
        return 1

    # ---- a quick deploy sanity: reload the checkpoint via the factory path and confirm it round-trips ----
    from control_lab.methods.offline_awr import load_checkpoint
    reloaded = load_checkpoint(ckpt_path)
    # confirm the reloaded numpy params reproduce the same gating on a small probe of the corpus.
    probe = corpus.phi[:2048]
    p_fit = _np_policy_probs(gate._np_params, probe, gate._hidden)
    p_rel = _np_policy_probs(reloaded._np_params, probe, reloaded._hidden)
    if not np.allclose(p_fit, p_rel):
        raise RuntimeError("offline_awr_train: checkpoint round-trip mismatch (saved vs reloaded policy "
                           "disagree) — refuse to claim a deployable checkpoint (ADR-0002)")
    print("[awr-train] checkpoint round-trip OK (reloaded policy reproduces the fitted gating).", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
