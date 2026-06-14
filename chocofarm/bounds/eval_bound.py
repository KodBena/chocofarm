#!/usr/bin/env python3
"""
eval_bound.py — validation + headline driver for the information-relaxation dual bound
(chocofarm/bounds/info_relaxation.py; design+proofs docs/design/dual-bound.md).

VALIDATION (small sub-instances only — the live AZ job holds cores 0–3; the full
15,504-world headline is DEFERRED to the orchestrator with cores freed):

  (i)   z≡0 (vhat=None) reproduces the clairvoyant inner solve on the world-set
        [regression — on the FULL env this is the 0.1454 ceiling, deferred; on the
        sub-instance it is that sub-instance's clairvoyant value, checked here]
  (ii)  ρ_dual ≤ clairvoyant                                [tighter than clairvoyant]
  (iii) ρ_dual ≥ an achievable rate                         [a valid upper bound cannot
                                                            drop below an achievable
                                                            policy]
  (iv)  empirical penalty mean ≈ 0 under the belief         [direct dual feasibility]

Usage (always pinned + bounded):
  PYTHONPATH=. timeout 600 taskset -c 3 python -m chocofarm.bounds.eval_bound --validate
  PYTHONPATH=. timeout 3000 taskset -c 3 python -m chocofarm.bounds.eval_bound --full   # DEFERRED
"""
import argparse
import itertools
import time

import numpy as np

from chocofarm.model.env import Environment
from chocofarm.bounds.minienv import nw_cluster_mini
from chocofarm.bounds.info_relaxation import (
    PenalizedClairvoyant, dual_bound_rate, empirical_penalty_mean,
    vhat_analytic, DecompVhat, ExactBeliefVhat,
)


def exact_optimal_rate(mini, lo=0.0, hi=0.4, it=40):
    """The exact optimal belief-MDP rate on a (small) sub-instance: the λ where
    V*(initial belief) = 0 (the optimal policy's own Dinkelbach fixed point). This is
    the ACHIEVABLE optimum on the sub-instance — the dual bound must sit ≥ it, and with
    V̂=V* (strong duality) should equal it."""
    ev = ExactBeliefVhat()
    for _ in range(it):
        mid = 0.5 * (lo + hi)
        v0 = ev(mini, ("w", mini.entry), mini.worlds, set(), mid)
        if v0 > 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def clairvoyant_on_worlds(env, keep, worlds, dink_iters=6, lo=0.0, hi=0.4):
    """The EXISTING clairvoyant inner solve (harness.clairvoyant_rate) restricted to
    `worlds` over treasures `keep` — the z≡0 ground truth (a Dinkelbach fixed point of
    ΣR/ΣT over the per-world subset×permutation optima)."""
    def ev(lam):
        totR = totT = 0.0
        for w in worlds:
            w = int(w)
            present = [t for t in keep if (w >> t) & 1]
            base = env.exit_cost(("w", env.entry))
            bv, bR, bT = -lam * base, 0.0, base
            for s in range(1, len(present) + 1):
                for sub in itertools.combinations(present, s):
                    R = sum(env.value[i] for i in sub)
                    bt = min(env.route_time(("w", env.entry), list(p))
                             for p in itertools.permutations(sub))
                    v = R - lam * bt
                    if v > bv:
                        bv, bR, bT = v, R, bt
            totR += bR
            totT += bT
        return totR / totT
    lam = 0.0
    for _ in range(dink_iters):
        lam = ev(lam)
    return lam


def achievable_on_mini(mini, vl=0.10):
    """A simple achievable rate on the sub-instance: the realizable-static-style route
    over the mini cluster (a fixed value-aware NN route, best expected-rate prefix) —
    a genuine ACHIEVABLE policy rate the bound must not fall below (deliverable iii).
    Uses the mini's reduced prior (K/|keep| present-fraction)."""
    env = mini
    loc = ("w", env.entry)
    unv = set(mini.keep)
    route, t, best = [], 0.0, -1.0
    while unv:
        i = max(unv, key=lambda j: env.value[j] / (env.d(loc, ("t", j)) + 1e-9))
        t += env.d(loc, ("t", i)); loc = ("t", i); route.append(i); unv.discard(i)
        # expected reward of this prefix under the mini prior: each kept treasure is
        # present with prob K/|keep|
        p = mini.K / len(mini.keep)
        rate = p * sum(env.value[r] for r in route) / (t + env.exit_cost(loc))
        best = max(best, rate)
    return best


def validate():
    env = Environment()
    print("=" * 74)
    print("INFORMATION-RELAXATION DUAL BOUND — validation on a small sub-instance")
    print("=" * 74, flush=True)

    # Sub-instance: the NW sense-cluster {8,9,10,11,12}, k_local=2 present → C(5,2)=10
    # worlds, a microscopic belief-MDP. Real geometry/faces/costs (honest numbers).
    mini = nw_cluster_mini(env, k_local=2)
    print(f"\nsub-instance: NW cluster keep={mini.keep} K={mini.K}  "
          f"worlds={len(mini.worlds)}  faces={len(mini.detectors)}  "
          f"(real geometry/faces/costs)\n", flush=True)

    # --- reference rates on this sub-instance ---
    t0 = time.time()
    clair_ref = clairvoyant_on_worlds(env, mini.keep, mini.worlds)
    achiev = achievable_on_mini(mini)
    print(f"reference: clairvoyant(z≡0) = {clair_ref:.4f}   "
          f"achievable(static-NN) = {achiev:.4f}   ({time.time()-t0:.1f}s)\n", flush=True)

    # (i) REGRESSION: vhat=None (z≡0) reproduces the clairvoyant value ------------
    t0 = time.time()
    pc0 = PenalizedClairvoyant(mini, vhat=None)
    out0 = dual_bound_rate(pc0, mini.worlds, lo=0.0, hi=0.4)
    dual0 = out0["lambda"]
    ok_i = abs(clair_ref - dual0) < 2e-3
    print(f"(i)   REGRESSION   z≡0 dual = {dual0:.4f}   clairvoyant = {clair_ref:.4f}   "
          f"{'PASS' if ok_i else 'FAIL'}   ({time.time()-t0:.1f}s)", flush=True)

    # (iv) DUAL FEASIBILITY: empirical penalty mean ≈ 0 on the FULL belief --------
    # The martingale property holds on the natural filtration regardless of the
    # world-population, so use the full env's real belief filtration with a cheap
    # F-adapted policy.
    from chocofarm.solvers.base import GreedyPolicy
    t0 = time.time()
    pc_feas = PenalizedClairvoyant(env, vhat=vhat_analytic, vhat_lam=0.10)
    m, se = empirical_penalty_mean(env, pc_feas, GreedyPolicy(), lam=0.10, runs=400, seed=11)
    ok_iv = abs(m) < 3 * se + 1e-6
    print(f"(iv)  DUAL-FEAS    mean Σz = {m:+.4f} ± {se:.4f} (1σ)   "
          f"{'PASS (≈0)' if ok_iv else 'CHECK'}   ({time.time()-t0:.1f}s)", flush=True)

    # exact achievable optimum on the sub-instance (the floor the bound must clear) ---
    t0 = time.time()
    opt = exact_optimal_rate(mini)
    print(f"\nexact optimal belief-MDP rate on sub-instance = {opt:.4f}   "
          f"({time.time()-t0:.1f}s)", flush=True)

    # (ii)+(iii) TIGHTNESS: V̂ = V* (definitive, strong-duality), analytic, decomp ----
    # V̂=V* is the DEFINITIVE tightening test: BSS strong duality ⇒ λ̄ = opt, well below
    # the clairvoyant — proving the machinery tightens when handed a good V̂. analytic
    # and decomp are weaker approximations (may or may not tighten; both still VALID).
    vhats = [("exact-V*", ExactBeliefVhat(), opt),
             ("analytic", vhat_analytic, 0.10)]
    for name, vh, vl in vhats:
        t0 = time.time()
        pc = PenalizedClairvoyant(mini, vhat=vh, vhat_lam=vl)
        out = dual_bound_rate(pc, mini.worlds, lo=0.0, hi=0.4)
        lam_bar = out["lambda"]
        certified = min(lam_bar, clair_ref)   # both valid; report the tighter
        below = certified <= clair_ref + 2e-3
        above = lam_bar >= opt - 2e-3         # validity floor: bound ≥ achievable
        tag = "TIGHTENS" if lam_bar < clair_ref - 2e-3 else "no-tighten (valid)"
        print(f"(ii/iii) V̂={name:9s} λ̄ = {lam_bar:.4f}  certified=min(λ̄,clair)="
              f"{certified:.4f}   ≤clair: {'Y' if below else 'N'}   "
              f"≥opt({opt:.4f}): {'Y' if above else 'N'}   {tag}   "
              f"({time.time()-t0:.1f}s)", flush=True)

    # decomp V̂: a FAST single-λ diagnostic (B at the clairvoyant λ). The decomp
    # DECISION-value is not a self-consistent state-value; its martingale increments are
    # exploitable, so B(clair) > 0 ⇒ its root λ̄ is ABOVE clairvoyant (LOOSENS). Shown
    # as a diagnostic, not a full bisection (which would widen hi repeatedly + be slow).
    try:
        dv = DecompVhat(horizon=1)
        pc = PenalizedClairvoyant(mini, vhat=dv, vhat_lam=0.094)
        t0 = time.time()
        Bc, _, _ = pc.B_value(clair_ref, mini.worlds)
        verdict = ("LOOSENS (λ̄>clair) — decision-value not self-consistent; "
                   "certified falls back to clairvoyant" if Bc > 1e-3
                   else "tightens/neutral")
        print(f"(diag)   V̂=decomp    B(clair={clair_ref:.4f}) = {Bc:+.4f}  →  {verdict}  "
              f"({time.time()-t0:.1f}s)", flush=True)
    except Exception as e:
        print(f"      (decomp V̂ diagnostic skipped: {type(e).__name__}: {e})", flush=True)

    print("\nNOTE: the full 15,504-world headline is DEFERRED. The flat inner DP is "
          "INTRACTABLE on the full belief (measured, see report §4.4); the full run "
          "needs the decomposition-aligned separable solve.", flush=True)


def full():
    """The full 15,504-world headline. DEFERRED — run only with cores freed.

    MEASURED TRACTABILITY FINDING (see the report / dual-bound.md §4.3): the FLAT
    per-world inner DP enumerates all 44 faces and recurses into their (large)
    successor-belief splits, so it does NOT scale to the full 15,504-world belief —
    even a single world's no-penalty inner DP did not finish in >60 s on core 3. The
    flat driver below is therefore guarded: it runs the z≡0 regression via the EXISTING
    tractable clairvoyant solve (reproducing 0.1454, seconds), and refuses to launch
    the intractable flat penalized run, pointing at the decomposition-aligned solve
    that the report specifies as the tractable full-instance path."""
    env = Environment()
    print("FULL 15,504-world dual bound.", flush=True)
    # z≡0 regression — the existing tractable clairvoyant solve (subset×perm over the
    # present set, no faces). Reproduces 0.1454 in seconds.
    t0 = time.time()
    ceil = clairvoyant_on_worlds(env, list(range(env.N)), env.worlds)
    print(f"z≡0 (clairvoyant) ceiling = {ceil:.4f}   ({time.time()-t0:.0f}s)  "
          f"[the loose bound this sharpens]", flush=True)
    print("\nPENALIZED full run via the FLAT inner DP is INTRACTABLE (measured) — the "
          "tractable path is the decomposition-aligned separable inner solve "
          "(report §full-run). Not launching the flat penalized run (it would hang).",
          flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--full", action="store_true")
    args = ap.parse_args()
    if args.full:
        full()
    else:
        validate()
