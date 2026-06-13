#!/usr/bin/env python3
"""
Independent feasibility measurement harness for the chocobo exact solver.

Builds growing sub-instances from the REAL parsed instance (chocobo_instance.json):
take the n treasures nearest a chosen teleport, keep their real coords, real
detection regions, and the overlaps among them; build faces (singleton detectors
+ disjunctive overlap faces); k=5 worlds = C(n,5).

Instruments an instrumented COPY of the Stage-1 solver to count:
  * reachable (location, collected, belief) states actually solved,
  * distinct beliefs (frozensets of worlds) reached,
  * distinct (collected, belief) information-states,
  * wall-time and peak memory of solve(lam) at a representative lambda.

Nothing outside /home/bork/w/vdc/ is touched.
"""
import json
import math
import itertools
import time
import sys
import tracemalloc
from dataclasses import dataclass

# reuse the dataclass + helpers from the prototype
sys.path.insert(0, "/home/bork/w/vdc")
from chocobo_stage1 import Instance, worlds_exactly_k


JSON = "/home/bork/w/vdc/chocobo_instance.json"


def load_real():
    d = json.load(open(JSON))
    treasures = {int(k): tuple(v) for k, v in d["treasures"].items()}
    teleports = {k: tuple(v) for k, v in d["teleports"].items()}
    regions = set(int(k) for k in d["regions_wkt"])      # which treasures have a region
    overlaps = [tuple(p) for p in d["overlaps"]]
    return treasures, teleports, regions, overlaps


def build_subinstance(n, k, anchor="CSCE"):
    """n nearest treasures to the anchor teleport (relabelled 0..n-1), k present.

    Faces: every kept treasure with a region gets a singleton detector;
    every overlap pair fully inside the kept set gets a disjunctive face.
    Node id layout:  treasures 0..n-1 ; faces n.. ; teleports last 3.
    Entry/exits = the 3 real teleports.
    """
    treasures, teleports, regions, overlaps = load_real()
    ax, ay = teleports[anchor]
    order = sorted(treasures, key=lambda t: math.hypot(treasures[t][0] - ax,
                                                        treasures[t][1] - ay))
    kept = order[:n]
    relabel = {old: new for new, old in enumerate(kept)}    # old treasure id -> 0..n-1
    keptset = set(kept)

    coords = {}
    value = [1.0] * n
    for old, new in relabel.items():
        coords[new] = treasures[old]

    faces = {}
    fid = n
    # singleton detectors for kept treasures that have a region
    for old in kept:
        if old in regions:
            new = relabel[old]
            faces[fid] = frozenset({new})
            coords[fid] = treasures[old]                    # region surrounds the treasure
            fid += 1
    # disjunctive faces for overlap pairs fully inside the kept set
    for (i, j) in overlaps:
        if i in keptset and j in keptset:
            ni, nj = relabel[i], relabel[j]
            faces[fid] = frozenset({ni, nj})
            mx = (treasures[i][0] + treasures[j][0]) / 2.0  # waypoint ~ overlap midpoint
            my = (treasures[i][1] + treasures[j][1]) / 2.0
            coords[fid] = (mx, my)
            fid += 1

    # teleports as entry/exits
    tele_ids = {}
    base = fid
    for off, (name, xy) in enumerate(teleports.items()):
        tele_ids[name] = base + off
        coords[base + off] = xy
    entry = tele_ids[anchor]
    exits = list(tele_ids.values())

    inst = Instance(n=n, coords=coords, value=value, faces=faces,
                    entry=entry, exits=exits, teleport_time=80.0)
    worlds = worlds_exactly_k(n, k)
    return inst, worlds, len(faces)


# --------------------------------------------------------- instrumented solve ---
class InstrumentedSolver:
    """Same recursion as Stage-1 Solver.solve, with reach counters."""

    def __init__(self, inst, worlds):
        self.inst = inst
        self.worlds0 = frozenset(worlds)
        self.face_mask = {f: sum(1 << t for t in S) for f, S in inst.faces.items()}

    def arrive(self, u, collected, belief):
        inst = self.inst
        if u < inst.n:
            bit = 1 << u
            pres = frozenset(w for w in belief if w & bit)
            absent = belief - pres
            out = []
            if pres:
                out.append((len(pres) / len(belief), inst.value[u], collected | bit, pres))
            if absent:
                out.append((len(absent) / len(belief), 0.0, collected, absent))
            return out
        S = self.face_mask[u]
        pos = frozenset(w for w in belief if w & S)
        neg = belief - pos
        out = []
        if pos:
            out.append((len(pos) / len(belief), 0.0, collected, pos))
        if neg:
            out.append((len(neg) / len(belief), 0.0, collected, neg))
        return out

    def candidates(self, collected, belief):
        inst, cand = self.inst, []
        for t in range(inst.n):
            bit = 1 << t
            if collected & bit:
                continue
            if not any(w & bit for w in belief):
                continue
            cand.append(t)
        for f, S in self.face_mask.items():
            if any(w & S for w in belief) and any(not (w & S) for w in belief):
                cand.append(f)
        return cand

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

        v0 = V(inst.entry, 0, self.worlds0)
        stats = dict(reachable_states=len(memo),
                     distinct_beliefs=len(distinct_beliefs),
                     distinct_infostates=len(distinct_infostates))
        return v0, memo, acts, stats


def measure(n, k=5, anchor="CSCE", lam=None, time_budget_s=900):
    inst, worlds, nfaces = build_subinstance(n, k, anchor)
    solver = InstrumentedSolver(inst, worlds)
    nworlds = len(worlds)
    # a representative lambda: pick the rate of a cheap greedy-ish baseline.
    # Using a fixed small lambda avoids the Dinkelbach outer loop (which would
    # call solve ~60x); one solve at a representative lambda reveals the reach.
    if lam is None:
        lam = 0.01
    tracemalloc.start()
    t0 = time.perf_counter()
    v0, memo, acts, stats = solver.solve(lam)
    dt = time.perf_counter() - t0
    cur, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return dict(n=n, k=k, nfaces=nfaces, nworlds=nworlds, lam=lam,
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
