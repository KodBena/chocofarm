#!/usr/bin/env python3
"""
Representation-independent measurement of the INTRINSIC belief reachability.

Forget locations and the lambda penalty.  The number of distinct BELIEFS reachable
is a function only of the observation structure (which clauses can be tested) and
the prior world set.  Compute the forward closure:

  start  = {FULL world set}
  expand: for each belief B in the frontier, for each clause c (treasure-present[t]
          for t, or face-disjunction[f]), B splits into  B&c  and  B&~c  (the two
          observation outcomes).  Each non-empty child is a reachable belief.

This BOUNDS the VI reachable-belief set from above for the unconstrained operator
(VI only reaches a subset, gated by which actions are non-dominated and which
treasures are already collected).  It is the cleanest intrinsic growth signal and
is cheap enough to push to larger n than full VI.

We also count the (collected, belief) info-states under the SAME closure but with
collected tracked (treasure-present outcome sets collected|=bit), which is the
genuine planning-state count modulo location.
"""
import math
import sys
import time
import tracemalloc

sys.path.insert(0, "/home/bork/w/vdc")
from chocobo_measure import build_subinstance


def belief_closure(n, k=5, anchor="CSCE", track_collected=False, cap=None):
    inst, worlds, nfaces = build_subinstance(n, k, anchor)
    worlds = list(worlds)
    W = len(worlds)
    FULL = (1 << W) - 1
    PM = [0] * inst.n
    face_bits = {f: sum(1 << t for t in S) for f, S in inst.faces.items()}
    PMface = {f: 0 for f in inst.faces}
    for i, w in enumerate(worlds):
        bi = 1 << i
        for t in range(inst.n):
            if w & (1 << t):
                PM[t] |= bi
        for f, S in face_bits.items():
            if w & S:
                PMface[f] |= bi

    # clause masks: each treasure-present and each face-disjunction
    treasure_masks = list(range(inst.n))           # index -> treasure id
    face_ids = list(inst.faces)

    if not track_collected:
        seen = set()
        frontier = [FULL]
        seen.add(FULL)
        while frontier:
            B = frontier.pop()
            # treasure splits
            for t in range(inst.n):
                pm = PM[t]
                pos = B & pm
                neg = B & ~pm
                for child in (pos, neg):
                    if child and child not in seen:
                        seen.add(child)
                        frontier.append(child)
            # face splits
            for f in face_ids:
                pm = PMface[f]
                pos = B & pm
                neg = B & ~pm
                for child in (pos, neg):
                    if child and child not in seen:
                        seen.add(child)
                        frontier.append(child)
            if cap and len(seen) > cap:
                return dict(n=n, beliefs=len(seen), capped=True)
        return dict(n=n, beliefs=len(seen), capped=False)
    else:
        # (collected, belief) reachable set; only present-outcome of a treasure adds to collected
        seen = set()
        start = (0, FULL)
        seen.add(start)
        frontier = [start]
        while frontier:
            c, B = frontier.pop()
            for t in range(inst.n):
                if c & (1 << t):
                    continue
                pm = PM[t]
                pos = B & pm
                neg = B & ~pm
                if pos:
                    st = (c | (1 << t), pos)
                    if st not in seen:
                        seen.add(st); frontier.append(st)
                if neg:
                    st = (c, neg)
                    if st not in seen:
                        seen.add(st); frontier.append(st)
            for f in face_ids:
                pm = PMface[f]
                pos = B & pm
                neg = B & ~pm
                for child in (pos, neg):
                    if child:
                        st = (c, child)
                        if st not in seen:
                            seen.add(st); frontier.append(st)
            if cap and len(seen) > cap:
                return dict(n=n, infostates=len(seen), capped=True)
        return dict(n=n, infostates=len(seen), capped=False)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "belief"
    ns = [int(x) for x in sys.argv[2:]] if len(sys.argv) > 2 else [8, 10, 12]
    cap = 80_000_000
    print(f"# mode={mode}  cap={cap}")
    for n in ns:
        tracemalloc.start()
        t0 = time.perf_counter()
        if mode == "belief":
            r = belief_closure(n, track_collected=False, cap=cap)
            metric = r['beliefs']; label = 'beliefs'
        else:
            r = belief_closure(n, track_collected=True, cap=cap)
            metric = r['infostates']; label = 'infostates'
        dt = time.perf_counter() - t0
        cur, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        print(f"n={n:>3}  {label}={metric:>14,}  capped={r['capped']}  "
              f"{dt:>8.2f}s  {peak/1e6:>8.1f}MB", flush=True)
