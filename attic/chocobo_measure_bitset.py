#!/usr/bin/env python3
"""
Improved-representation variant: belief as a big-integer BITSET over world INDICES.

Worlds are enumerated 0..W-1.  belief is a Python int with bit i set iff world i
is still consistent.  For each treasure t and each face f we precompute a "presence
mask" PM[t] / PM_face[f] = the big-int with bit i set iff world i satisfies the
clause (treasure t present / >=1 of the face's set present).  Then:

    pres  = belief & PM[t]          # worlds where t present
    absent= belief & ~PM[t] & FULL  # worlds where t absent

are single big-int AND ops (C-speed), instead of Python-level frozenset iteration.
The canonical state key is (location, collected, belief_int) -- belief_int is a
canonical, hashable, compact signature of the surviving world set.

This isolates the representation question: the REACHABLE STATE COUNT is identical
to the frozenset solver (same recursion, same memoisation key semantics); only the
per-state cost and memory change.
"""
import json
import math
import time
import sys
import tracemalloc

sys.path.insert(0, "/home/bork/w/vdc")
from chocobo_measure import build_subinstance


class BitsetSolver:
    def __init__(self, inst, worlds):
        self.inst = inst
        self.worlds = list(worlds)
        W = len(self.worlds)
        self.W = W
        self.FULL = (1 << W) - 1
        # presence masks over world INDICES
        self.PM = [0] * inst.n                       # treasure t present
        face_mask_bits = {f: sum(1 << t for t in S) for f, S in inst.faces.items()}
        self.PMface = {f: 0 for f in inst.faces}
        for i, w in enumerate(self.worlds):
            bit_i = 1 << i
            for t in range(inst.n):
                if w & (1 << t):
                    self.PM[t] |= bit_i
            for f, S in face_mask_bits.items():
                if w & S:
                    self.PMface[f] |= bit_i
        self.belief0 = self.FULL

    def candidates(self, collected, belief):
        inst, cand = self.inst, []
        for t in range(inst.n):
            if collected & (1 << t):
                continue
            if belief & self.PM[t]:                  # some world has t present
                cand.append(t)
        for f in inst.faces:
            pmf = self.PMface[f]
            pos = belief & pmf
            if pos and pos != belief:                # discriminating
                cand.append(f)
        return cand

    def arrive(self, u, collected, belief):
        inst = self.inst
        if u < inst.n:
            pres = belief & self.PM[u]
            absent = belief & ~self.PM[u]
            tot = belief.bit_count()
            out = []
            if pres:
                out.append((pres.bit_count() / tot, inst.value[u], collected | (1 << u), pres))
            if absent:
                out.append((absent.bit_count() / tot, 0.0, collected, absent))
            return out
        pmf = self.PMface[u]
        pos = belief & pmf
        neg = belief & ~pmf
        tot = belief.bit_count()
        out = []
        if pos:
            out.append((pos.bit_count() / tot, 0.0, collected, pos))
        if neg:
            out.append((neg.bit_count() / tot, 0.0, collected, neg))
        return out

    def solve(self, lam):
        inst, memo, acts = self.inst, {}, {}
        distinct_beliefs = set()
        distinct_infostates = set()

        def terminate(loc):
            return -lam * (min(inst.travel(loc, e) for e in inst.exits) + inst.teleport_time)

        def V(loc, collected, belief):
            key = (loc, collected, belief)
            if key in memo:
                return memo[key]
            distinct_beliefs.add(belief)
            distinct_infostates.add((collected, belief))
            best, best_act = terminate(loc), ("teleport", None)
            for u in self.candidates(collected, belief):
                exp = 0.0
                for p, r, c2, b2 in self.arrive(u, collected, belief):
                    exp += p * (r + V(u, c2, b2))
                val = -lam * inst.travel(loc, u) + exp
                if val > best:
                    best, best_act = val, ("go", u)
            memo[key], acts[key] = best, best_act
            return best

        v0 = V(inst.entry, 0, self.belief0)
        stats = dict(reachable_states=len(memo),
                     distinct_beliefs=len(distinct_beliefs),
                     distinct_infostates=len(distinct_infostates))
        return v0, memo, acts, stats


def measure(n, k=5, anchor="CSCE", lam=0.01):
    inst, worlds, nfaces = build_subinstance(n, k, anchor)
    solver = BitsetSolver(inst, worlds)
    tracemalloc.start()
    t0 = time.perf_counter()
    v0, memo, acts, stats = solver.solve(lam)
    dt = time.perf_counter() - t0
    cur, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return dict(n=n, k=k, nfaces=nfaces, nworlds=len(worlds), lam=lam,
                wall_s=dt, peak_mb=peak / 1e6, v0=v0, **stats)


if __name__ == "__main__":
    ns = [int(x) for x in sys.argv[1:]] if len(sys.argv) > 1 else [8, 10, 12]
    print(f"{'n':>3} {'faces':>6} {'worlds':>7} {'reach_states':>13} "
          f"{'distinct_belief':>16} {'infostates':>11} {'wall_s':>9} {'peak_MB':>9}")
    for n in ns:
        r = measure(n)
        print(f"{r['n']:>3} {r['nfaces']:>6} {r['nworlds']:>7} {r['reachable_states']:>13} "
              f"{r['distinct_beliefs']:>16} {r['distinct_infostates']:>11} "
              f"{r['wall_s']:>9.3f} {r['peak_mb']:>9.1f}", flush=True)
