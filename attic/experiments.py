#!/usr/bin/env python3
"""
Decisive experiments after the consult-001 detector fix (unit values).

E1 (the null-detector): a CLAIRVOYANT policy that knows the true present-set for free at
entry and plays the rate-optimal route over it. `rate_clairvoyant − rate_static` is the
ABSOLUTE CEILING on what any adaptive sensing could ever buy. If small, adaptivity cannot
pay much on this instance regardless of detector fidelity — and we KNOW it rather than
assume it. Alongside: the corrected greedy/rollout (detectors now genuinely disjunctive)
measured by unbiased MC, vs a realizable static route.
"""
import sys
import os
import itertools
import time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import chocobo_stage2_approx as M

ENTRY = M.ENTRY


def route_time(seq):
    if not seq:
        return M.exit_cost(("w", ENTRY))
    t = M.d(("w", ENTRY), ("t", seq[0]))
    for a, b in zip(seq, seq[1:]):
        t += M.d(("t", a), ("t", b))
    return t + M.exit_cost(("t", seq[-1]))


def clairvoyant_eval(lam, runs, seed):
    npr = np.random.default_rng(seed)
    totR = totT = 0.0
    for _ in range(runs):
        w = int(npr.choice(M.worlds))
        present = [t for t in range(M.N) if (w >> t) & 1]
        base = M.exit_cost(("w", ENTRY))                # collect nothing, just teleport
        bestv, bR, bT = -lam * base, 0.0, base
        for s in range(1, len(present) + 1):
            for sub in itertools.combinations(present, s):
                R = sum(M.value[i] for i in sub)
                bt = min(route_time(list(p)) for p in itertools.permutations(sub))
                v = R - lam * bt
                if v > bestv:
                    bestv, bR, bT = v, R, bt
        totR += bR
        totT += bT
    return totR / totT, totR / runs, totT / runs


def clairvoyant_rate():
    lam = 0.0
    for _ in range(5):
        lam = clairvoyant_eval(lam, 1000, 1)[0]
    return clairvoyant_eval(lam, 3000, 7)


def realizable_static():
    loc, unv, route, t, best = ("w", ENTRY), set(range(M.N)), [], 0.0, (-1.0, 0)
    while unv:
        i = max(unv, key=lambda j: M.value[j] / (M.d(loc, ("t", j)) + 1e-9))
        t += M.d(loc, ("t", i)); loc = ("t", i); route.append(i); unv.discard(i)
        rate = (M.K / M.N) * sum(M.value[r] for r in route) / (t + M.exit_cost(loc))
        if rate > best[0]:
            best = (rate, len(route))
    return best


def fp(decision, lam0, iters, runs, seed):
    lam = lam0
    for _ in range(iters):
        lam = M.eval_rate(decision, lam, runs, seed)[0]
    return lam


def main():
    print("UNIT values; detectors now disjunctive (cover from the real 17 overlaps).\n", flush=True)

    s_rate, s_L = realizable_static()
    print(f"realizable static : {s_rate:.4f}  (fixed route, {s_L} treasures)", flush=True)

    lam_g = fp(M.greedy_decision, 0.0, 4, 800, 1)
    g_rate = M.eval_rate(M.greedy_decision, lam_g, 5000, 7)[0]
    print(f"greedy            : {g_rate:.4f}", flush=True)

    lam_r = lam_g
    for _ in range(3):
        lam_r = M.eval_rate(M.rollout_decision, lam_r, 50, 3)[0]
    r_rate, rR, rT, r_ex, r_det = M.eval_rate(M.rollout_decision, lam_r, 250, 7)
    print(f"rollout           : {r_rate:.4f}  (det/run={r_det:.2f}, E[R]={rR:.2f}, E[T]={rT:.2f})", flush=True)

    t0 = time.time()
    c_rate, cR, cT = clairvoyant_rate()
    print(f"clairvoyant CEIL  : {c_rate:.4f}  (E[R]={cR:.2f}, E[T]={cT:.2f})  [{time.time()-t0:.0f}s]", flush=True)

    print(f"\nVoI ceiling (clairvoyant vs static) : {(c_rate-s_rate)/s_rate*100:+.1f}%  "
          f"<- max any sensing could buy", flush=True)
    print(f"captured     (rollout vs static)    : {(r_rate-s_rate)/s_rate*100:+.1f}%", flush=True)


if __name__ == "__main__":
    main()
