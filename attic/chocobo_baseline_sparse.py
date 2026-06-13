#!/usr/bin/env python3
"""
The DUMB convergent baseline: online sparse-sampling expectimax (Kearns-Mansour-Ng).

No tricks: for each legal action, sample C consistent worlds, recurse to depth d, average,
take the max; terminate is always an option.  Provably approaches the optimum as (C->inf,
d->horizon) -- the anytime / "run-it-forever-gets-better" guarantee.  d=1 bottoms out in the
greedy rollout we already have; increasing d is the convergence ladder.

We anchor at lambda0 = the static-route rate, so the ROOT value satisfies:
    V_root > 0   <=>   the optimal adaptive policy beats the static baseline.
Reporting V_root and the chosen first action across a budget ladder shows the solver
converging (and whether/where it crosses the static line).
"""
import time
import numpy as np
import chocobo_stage2_approx as M


def legal_actions(bw, collected):
    marg = M.marginals(bw)
    acts = [("t", i) for i in range(M.N) if i not in collected and marg[i] > 0]
    acts += [("d", i) for i in M.regions
             if np.any((bw & M.cover_mask[i]) != 0) and np.any((bw & M.cover_mask[i]) == 0)]
    return acts


def child(a, w, bw, collected):
    kind, i = a
    if kind == "t":
        pres = bool((w >> i) & 1)
        r = M.value[i] if (pres and i not in collected) else 0.0
        return ("t", i), M.filt_treasure(bw, i, pres), (collected | {i} if pres else collected), r
    pos = bool(w & M.cover_mask[i])
    return ("d", i), M.filt_detector(bw, i, pos), collected, 0.0


def sparse_value(loc, bw, collected, lam, depth, C, npr, want_action=False):
    best, best_a = -lam * M.exit_cost(loc), ("term", None)         # terminate
    for a in legal_actions(bw, collected):
        smp = npr.choice(bw, size=min(C, len(bw)), replace=len(bw) < C)
        tot = 0.0
        for w in smp:
            w = int(w)
            cl, bw2, coll2, r = child(a, w, bw, collected)
            step = r - lam * M.d(loc, cl)
            if depth <= 1:
                rb, tb = M.simulate_base(cl, bw2, coll2, w, lam)    # greedy-rollout leaf
                tot += step + (rb - lam * tb)
            else:
                tot += step + sparse_value(cl, bw2, coll2, lam, depth - 1, C, npr)
        q = tot / len(smp)
        if q > best:
            best, best_a = q, a
    return (best, best_a) if want_action else best


def main():
    s_rate, s_L = M.static_rate()
    lam0 = s_rate
    print(f"static rate = lambda0 = {lam0:.4f}  (NN route, {s_L} treasures)")
    print(f"root value V>0  <=>  optimal adaptive beats static\n", flush=True)
    print(f"{'budget':>10} {'V_root (mean)':>15} {'first action':>16} {'beats static?':>14} {'sec':>6}",
          flush=True)
    for depth, C, seeds in [(1, 8, 3), (1, 32, 3), (2, 6, 3), (2, 16, 2)]:
        t0 = time.time()
        vals, acts = [], {}
        for s in range(seeds):
            npr = np.random.default_rng(s)
            v, a = sparse_value(("w", M.ENTRY), M.worlds, set(), lam0, depth, C, npr, True)
            vals.append(v)
            acts[a] = acts.get(a, 0) + 1
        v = float(np.mean(vals))
        aa = max(acts, key=acts.get)
        print(f"  d={depth},C={C:>3} {v:>15.4f} {str(aa):>16} {('YES' if v > 0 else 'no'):>14} "
              f"{time.time()-t0:>6.0f}", flush=True)


if __name__ == "__main__":
    main()
