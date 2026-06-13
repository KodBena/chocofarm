#!/usr/bin/env python3
"""
Stage 2b -- BOUNDED approximate adaptive policy on the real parsed map.

Exact belief is maintained online (a numpy array of surviving world-ints over the
15,504 exactly-5-of-20 worlds); it is cheap to filter and SHRINKS as we observe, so
memory is flat -- no offline enumeration of beliefs, no memo, no blow-up.

Policies:
  * greedy base    -- chase the best expected lambda-adjusted treasure; ignores detectors.
  * one-step rollout (policy improvement over greedy) -- at each decision, for each
    candidate (terminate / nearest undetermined detectors / nearest likely treasures),
    sample worlds from the current belief, simulate the base policy after the action,
    pick the argmax.  This is where detector use (information gathering) appears.
Rate objective via Dinkelbach: lambda <- achieved rate (fixed point of greedy).

UNITS CAVEAT: travel is Euclidean distance in GeoGebra map units (not seconds; real
terrain times are asymmetric and uncollected).  Teleport overhead is a stand-in in the
same units.  So "rate" is treasures per map-distance-unit, swappable when real times land.
"""
import json
import math
import itertools
import os
import random
import sys
import numpy as np
from shapely import wkt

# ---- params (bounded) ----
TELE_OH = 12.0          # teleport overhead, map-distance units (uncalibrated stand-in)
STEP_CAP = 16
ROLL_S = 10             # world-samples per rollout Q-estimate
ROLL_RUNS = 40          # MC runs to evaluate the rollout policy
GREEDY_RUNS = 2500
LAM_RUNS, LAM_ITERS = 800, 4
NEAR_DET, NEAR_TRE = 3, 3
ENTRY = "CSNE"

# ---- load instance ----
HERE = os.path.dirname(os.path.abspath(__file__))
data = json.load(open(os.path.join(HERE, "chocobo_instance.json")))
treasures = {int(i): tuple(xy) for i, xy in data["treasures"].items()}
teleports = {k: tuple(v) for k, v in data["teleports"].items()}
regions = {int(i): wkt.loads(w) for i, w in data["regions_wkt"].items()}
N, K = len(treasures), 5
value = [1.0] * N

# detectors: a region's disjunctive cover = itself + every region it AREA-overlaps with
# (the parsed `overlaps` array, 17 pairs). Entering region i reveals ">=1 present among
# {i} ∪ overlap-neighbours" — the genuinely disjunctive sensor. (Over-approximation: treats
# entry to region i as exposure to all its overlaps; per-arrangement-face reification is a
# later refinement. Fixes consult-001 flaw #1, where a single representative_point() dropped
# 9 of 17 overlaps and left 8/16 detectors as singletons.)
det_pt, cover_mask = {}, {}
overlap_nbrs = {i: {i} for i in regions}
for a, b in data["overlaps"]:
    overlap_nbrs[int(a)].add(int(b))
    overlap_nbrs[int(b)].add(int(a))
for i, reg in regions.items():
    p = reg.representative_point()
    det_pt[i] = (p.x, p.y)
    cover_mask[i] = sum(1 << j for j in overlap_nbrs[i])

coord = {}
for i, xy in treasures.items():
    coord[("t", i)] = xy
for i, xy in det_pt.items():
    coord[("d", i)] = xy
for k, xy in teleports.items():
    coord[("w", k)] = xy


def d(a, b):
    (x1, y1), (x2, y2) = coord[a], coord[b]
    return math.hypot(x1 - x2, y1 - y2)


def exit_cost(loc):
    return min(d(loc, ("w", k)) for k in teleports) + TELE_OH


def nearest_exit(loc):
    return min(teleports, key=lambda k: d(loc, ("w", k)))


worlds = np.array([sum(1 << t for t in c) for c in itertools.combinations(range(N), K)], dtype=np.int64)


def marginals(bw):
    if len(bw) == 0:
        return np.zeros(N)
    return (((bw[:, None] >> np.arange(N)) & 1).mean(0))


def filt_treasure(bw, i, pres):
    bit = (bw >> i) & 1
    return bw[bit == (1 if pres else 0)]


def filt_detector(bw, i, pos):
    hit = (bw & cover_mask[i]) != 0
    return bw[hit if pos else ~hit]


# ---- greedy base ----
def greedy_decision(bw, loc, collected, lam):
    marg = marginals(bw)
    best, act = 0.0, ("term", None)
    for i in range(N):
        if i in collected or marg[i] <= 0:
            continue
        s = marg[i] * value[i] - lam * d(loc, ("t", i))
        if s > best:
            best, act = s, ("t", i)
    return act


def simulate_base(loc, bw, collected, world, lam):
    R = T = 0.0
    collected = set(collected)
    for _ in range(STEP_CAP):
        marg = marginals(bw)
        best, bi = 0.0, None
        for i in range(N):
            if i in collected or marg[i] <= 0:
                continue
            s = marg[i] * value[i] - lam * d(loc, ("t", i))
            if s > best:
                best, bi = s, i
        if bi is None:
            break
        T += d(loc, ("t", bi))
        pres = bool((world >> bi) & 1)
        if pres and bi not in collected:
            R += value[bi]
            collected.add(bi)
        bw = filt_treasure(bw, bi, pres)
        loc = ("t", bi)
    return R, T + exit_cost(loc)


# ---- one-step rollout (policy improvement over greedy) ----
def rollout_decision(bw, loc, collected, lam, rng):
    marg = marginals(bw)
    dets = [i for i in regions
            if np.any((bw & cover_mask[i]) != 0) and np.any((bw & cover_mask[i]) == 0)]
    dets.sort(key=lambda i: d(loc, ("d", i)))
    tres = [i for i in range(N) if i not in collected and marg[i] > 0]
    tres.sort(key=lambda i: d(loc, ("t", i)))
    cands = [("term", None)] + [("d", i) for i in dets[:NEAR_DET]] + [("t", i) for i in tres[:NEAR_TRE]]
    sample = rng.choice(bw, size=min(ROLL_S, len(bw)), replace=len(bw) < ROLL_S)

    term_val = -lam * exit_cost(loc)
    best_q, best_a = term_val, ("term", None)
    for a in cands:
        if a[0] == "term":
            q = term_val
        else:
            kind, i = a
            tot = 0.0
            for w in sample:
                w = int(w)
                if kind == "d":
                    pos = bool(w & cover_mask[i])
                    bw_a, coll_a, r = filt_detector(bw, i, pos), collected, 0.0
                else:
                    pres = bool((w >> i) & 1)
                    r = value[i] if (pres and i not in collected) else 0.0
                    bw_a = filt_treasure(bw, i, pres)
                    coll_a = collected | {i} if pres else collected
                rb, tb = simulate_base((kind, i), bw_a, coll_a, w, lam)
                tot += (r - lam * d(loc, (kind, i))) + (rb - lam * tb)
            q = tot / len(sample)
        if q > best_q:
            best_q, best_a = q, a
    return best_a


# ---- MC rate evaluation ----
def eval_rate(decision, lam, runs, seed):
    rng = random.Random(seed)
    npr = np.random.default_rng(seed)
    totR = totT = 0.0
    exits = {}
    det_visits = 0
    for _ in range(runs):
        w = int(npr.choice(worlds))
        bw, loc, collected = worlds, ("w", ENTRY), set()
        R = T = 0.0
        for _ in range(STEP_CAP):
            a = decision(bw, loc, collected, lam, npr) if decision is rollout_decision \
                else decision(bw, loc, collected, lam)
            if a[0] == "term":
                break
            kind, i = a
            T += d(loc, (kind, i))
            if kind == "t":
                pres = bool((w >> i) & 1)
                if pres and i not in collected:
                    R += value[i]
                    collected.add(i)
                bw = filt_treasure(bw, i, pres)
            else:
                det_visits += 1
                bw = filt_detector(bw, i, bool(w & cover_mask[i]))
            loc = (kind, i)
        e = nearest_exit(loc)
        T += d(loc, ("w", e)) + TELE_OH
        exits[e] = exits.get(e, 0) + 1
        totR += R
        totT += T
    return totR / totT, totR / runs, totT / runs, exits, det_visits / runs


# ---- static baseline: nearest-neighbour route, best prefix ----
def static_rate():
    loc, unv, route, t = ("w", ENTRY), set(range(N)), [], 0.0
    rows = []
    while unv:
        i = min(unv, key=lambda j: d(loc, ("t", j)))
        t += d(loc, ("t", i))
        loc = ("t", i)
        route.append(i)
        unv.discard(i)
        ER = 0.25 * len(route)            # marginal P(present) = K/N
        T = t + exit_cost(loc)
        rows.append((len(route), ER / T))
    L, best = max(rows, key=lambda r: r[1])
    return best, L


def main():
    print(f"real map: {N} treasures, k={K}, {len(regions)} detectors, "
          f"teleports {list(teleports)}; entry={ENTRY}", flush=True)

    import time
    s_rate, s_L = static_rate()
    print(f"static baseline (NN route, best prefix={s_L}) = {s_rate:.4f}\n", flush=True)

    # greedy at its own Dinkelbach fixed point
    lam_g = 0.0
    for _ in range(LAM_ITERS):
        lam_g, *_ = eval_rate(greedy_decision, lam_g, LAM_RUNS, seed=1)
    g_rate, gR, gT, g_ex, _ = eval_rate(greedy_decision, lam_g, GREEDY_RUNS, seed=7)
    print(f"greedy base    : lambda*={lam_g:.4f}  rate={g_rate:.4f}  "
          f"(E[R]={gR:.3f}, E[T]={gT:.2f})  exits={g_ex}", flush=True)

    # rollout at ITS OWN fixed point (Dinkelbach on the rollout policy)
    lam_r = lam_g
    for _ in range(3):
        lam_r, *_ = eval_rate(rollout_decision, lam_r, 30, seed=3)
        print(f"  rollout Dinkelbach: lambda -> {lam_r:.4f}", flush=True)
    t0 = time.time()
    r_rate, rR, rT, r_ex, r_det = eval_rate(rollout_decision, lam_r, 80, seed=7)
    print(f"rollout policy : lambda*={lam_r:.4f}  rate={r_rate:.4f}  "
          f"(E[R]={rR:.3f}, E[T]={rT:.2f})  exits={r_ex}  det/run={r_det:.2f}  "
          f"[{time.time()-t0:.0f}s]", flush=True)

    print(f"\nrollout vs greedy : {(r_rate-g_rate)/g_rate*100:+.1f}%", flush=True)
    print(f"rollout vs static : {(r_rate-s_rate)/s_rate*100:+.1f}%", flush=True)
    used_tau4 = r_ex.get("tau_4", 0) + g_ex.get("tau_4", 0)
    print(f"tau_4 teleport used as exit: {used_tau4} (rollout {r_ex.get('tau_4',0)}/80, "
          f"greedy {g_ex.get('tau_4',0)}/{GREEDY_RUNS})", flush=True)


if __name__ == "__main__":
    main()
