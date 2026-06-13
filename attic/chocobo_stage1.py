#!/usr/bin/env python3
"""
Chocobo gil-farming as ADAPTIVE STOCHASTIC ORIENTEERING -- Stage 1 (synthetic).

Model (as agreed in design):
  * n treasure locations; a prior over which are present, given as an explicit world
    set (default: exactly k of n, uniform without replacement -> C(n,k) worlds; bit t
    of a mask = treasure t present).  Always-present "token" nodes = bit set in every world.
  * Detection faces: arriving at a face covering set S reveals only the DISJUNCTION
    "(>=1 of S present)?".  Positive is weak (>=1); negative is strong (all absent).
    A treasure's own node reveals AND collects it (the delta point: observe == collect).
  * Belief = the set of worlds still consistent with everything observed.  Sound and
    complete BY CONSTRUCTION -- keep every consistent world, read entailments/posteriors
    off counts.  No hand-coded refutations.
  * Objective = long-run treasures / second (renewal-reward: i.i.d. re-roll each run).
    DINKELBACH: for a rate lambda, maximise E[ sum value - lambda * sum time ]; the
    lambda* where that optimum hits 0 IS the farming rate.  Under the lambda-penalty,
    elapsed time leaves the state -> backward-induction state is (location, belief, collected).
  * A run starts at a fixed entry waystone and ends by travelling to whichever exit
    waystone is cheapest + an ~80s teleport.  Dominated exits/nodes are never chosen.
"""
import math
import itertools
import random
from dataclasses import dataclass


# ----------------------------------------------------------------- instance ---
@dataclass
class Instance:
    n: int
    coords: dict           # node id -> (x, y), for every nav node
    value: list            # value[t] for treasure t
    faces: dict            # face node id -> frozenset of covered treasures
    entry: int
    exits: list
    teleport_time: float

    def travel(self, a, b):
        (x1, y1), (x2, y2) = self.coords[a], self.coords[b]
        return math.hypot(x1 - x2, y1 - y2)


def worlds_exactly_k(n, k):
    return [sum(1 << t for t in c) for c in itertools.combinations(range(n), k)]


# ------------------------------------------------------------------- solver ---
class Solver:
    def __init__(self, inst: Instance, worlds):
        self.inst = inst
        self.worlds0 = frozenset(worlds)
        self.face_mask = {f: sum(1 << t for t in S) for f, S in inst.faces.items()}

    def arrive(self, u, collected, belief):
        inst = self.inst
        if u < inst.n:                                   # treasure: reveal + collect
            bit = 1 << u
            pres = frozenset(w for w in belief if w & bit)
            absent = belief - pres
            out = []
            if pres:
                out.append((len(pres) / len(belief), inst.value[u], collected | bit, pres))
            if absent:
                out.append((len(absent) / len(belief), 0.0, collected, absent))
            return out
        S = self.face_mask[u]                            # detection face: disjunction only
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
            if not any(w & bit for w in belief):         # determined-absent -> dominated
                continue
            cand.append(t)
        for f, S in self.face_mask.items():
            if any(w & S for w in belief) and any(not (w & S) for w in belief):
                cand.append(f)
        return cand

    def solve(self, lam):
        inst, memo, acts = self.inst, {}, {}

        def terminate(loc):
            return -lam * (min(inst.travel(loc, e) for e in inst.exits) + inst.teleport_time)

        def V(loc, collected, belief):
            key = (loc, collected, belief)
            if key in memo:
                return memo[key]
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
        return v0, memo, acts

    def optimal_rate(self):
        g = lambda L: self.solve(L)[0]
        lo, hi = 0.0, 1.0
        while g(hi) > 0:
            hi *= 2
        for _ in range(60):
            mid = (lo + hi) / 2
            if g(mid) > 0:
                lo = mid
            else:
                hi = mid
        lam = (lo + hi) / 2
        _, _, acts = self.solve(lam)
        return lam, acts

    def eval_policy(self, acts):
        inst, cache = self.inst, {}

        def ev(loc, collected, belief):
            key = (loc, collected, belief)
            if key in cache:
                return cache[key]
            act = acts.get(key, ("teleport", None))
            if act[0] == "teleport":
                e = min(inst.exits, key=lambda x: inst.travel(loc, x))
                res = (0.0, inst.travel(loc, e) + inst.teleport_time)
            else:
                u = act[1]
                ER, ET = 0.0, inst.travel(loc, u)
                for p, r, c2, b2 in self.arrive(u, collected, belief):
                    er, et = ev(u, c2, b2)
                    ER += p * (r + er)
                    ET += p * et
                res = (ER, ET)
            cache[key] = res
            return res

        return ev(inst.entry, 0, self.worlds0)

    def _realize(self, w, u, collected, belief):
        inst = self.inst
        if u < inst.n:
            bit = 1 << u
            if w & bit:
                return inst.value[u], collected | bit, frozenset(x for x in belief if x & bit)
            return 0.0, collected, frozenset(x for x in belief if not (x & bit))
        S = self.face_mask[u]
        if w & S:
            return 0.0, collected, frozenset(x for x in belief if x & S)
        return 0.0, collected, frozenset(x for x in belief if not (x & S))

    def monte_carlo(self, acts, runs=20000, seed=0, far=None):
        rnd = random.Random(seed)
        inst, worlds = self.inst, list(self.worlds0)
        totR = totT = 0.0
        far_runs = 0
        for _ in range(runs):
            w = rnd.choice(worlds)
            loc, collected, belief, R, T = inst.entry, 0, self.worlds0, 0.0, 0.0
            visited_far = False
            while True:
                act = acts.get((loc, collected, belief), ("teleport", None))
                if act[0] == "teleport":
                    e = min(inst.exits, key=lambda x: inst.travel(loc, x))
                    T += inst.travel(loc, e) + inst.teleport_time
                    break
                u = act[1]
                if u == far:
                    visited_far = True
                T += inst.travel(loc, u)
                r, collected, belief = self._realize(w, u, collected, belief)
                R += r
                loc = u
            totR += R; totT += T
            far_runs += visited_far
        return totR / totT, totR / runs, totT / runs, far_runs / runs

    def exploratory_rollout(self, acts, eps=0.15, runs=20000, seed=1, far=None, max_steps=40):
        """epsilon-greedy: with prob eps take a uniformly random legal action (incl. the
        distant node and random teleport).  Shows that EXPLORATION samples the dominated
        node, unlike the optimal policy."""
        rnd = random.Random(seed)
        inst, worlds = self.inst, list(self.worlds0)
        totR = totT = 0.0
        far_runs = 0
        for _ in range(runs):
            w = rnd.choice(worlds)
            loc, collected, belief, R, T = inst.entry, 0, self.worlds0, 0.0, 0.0
            visited_far = False
            for _step in range(max_steps):
                if rnd.random() < eps:
                    options = [("go", u) for u in self.candidates(collected, belief)] + [("teleport", None)]
                    act = rnd.choice(options)
                else:
                    act = acts.get((loc, collected, belief), ("teleport", None))
                if act[0] == "teleport":
                    e = min(inst.exits, key=lambda x: inst.travel(loc, x))
                    T += inst.travel(loc, e) + inst.teleport_time
                    break
                u = act[1]
                if u == far:
                    visited_far = True
                T += inst.travel(loc, u)
                r, collected, belief = self._realize(w, u, collected, belief)
                R += r
                loc = u
            else:
                e = min(inst.exits, key=lambda x: inst.travel(loc, x))
                T += inst.travel(loc, e) + inst.teleport_time
            totR += R; totT += T
            far_runs += visited_far
        return totR / totT, far_runs / runs


# --------------------------------------------------------- static baseline ---
def static_optimal_rate(inst, worlds):
    W = list(worlds); nW = len(W)
    p = [sum(1 for w in W if w & (1 << t)) / nW for t in range(inst.n)]
    best = (-1.0, None)
    for r in range(1, inst.n + 1):
        for subset in itertools.combinations(range(inst.n), r):
            best_t, best_route = math.inf, None
            for perm in itertools.permutations(subset):
                t = inst.travel(inst.entry, perm[0])
                for a, b in zip(perm, perm[1:]):
                    t += inst.travel(a, b)
                e = min(inst.exits, key=lambda x: inst.travel(perm[-1], x))
                t += inst.travel(perm[-1], e) + inst.teleport_time
                if t < best_t:
                    best_t, best_route = t, (perm, e)
            ER = sum(inst.value[t] * p[t] for t in subset)
            rate = ER / best_t
            if rate > best[0]:
                best = (rate, (subset, best_route, best_t, ER))
    return best


# --------------------------------------------------------------- instances ---
def instance_A():
    coords = {
        0: (10, 50), 1: (20, 55), 2: (50, 20), 3: (80, 60), 4: (85, 50), 5: (50, 90),
        6: (15, 52), 7: (48, 22), 8: (82, 55),
        9: (5, 5), 10: (7, 7), 11: (95, 95), 12: (150, 50),
    }
    inst = Instance(n=6, coords=coords, value=[1] * 6,
                    faces={6: frozenset({0, 1}), 7: frozenset({2}), 8: frozenset({3, 4})},
                    entry=9, exits=[10, 11, 12], teleport_time=80.0)
    return inst, worlds_exactly_k(6, 2)


def instance_A_distant():
    """Instance A + treasure 6: ALWAYS present (token), but ~7000 units away.  Renumbered:
    treasures 0..6, faces 7/8/9, entry 10, exits 11/12/13."""
    coords = {
        0: (10, 50), 1: (20, 55), 2: (50, 20), 3: (80, 60), 4: (85, 50), 5: (50, 90),
        6: (5000, 5000),                                  # the distant always-present token
        7: (15, 52), 8: (48, 22), 9: (82, 55),
        10: (5, 5), 11: (7, 7), 12: (95, 95), 13: (150, 50),
    }
    inst = Instance(n=7, coords=coords, value=[1] * 7,
                    faces={7: frozenset({0, 1}), 8: frozenset({2}), 9: frozenset({3, 4})},
                    entry=10, exits=[11, 12, 13], teleport_time=80.0)
    worlds = [w | (1 << 6) for w in worlds_exactly_k(6, 2)]   # 2 of {0..5} present, t6 always
    return inst, worlds


# ------------------------------------------------------------ policy print ---
def print_tree(solver, acts, loc, collected, belief, prefix=""):
    inst = solver.inst
    act = acts.get((loc, collected, belief), ("teleport", None))
    if act[0] == "teleport":
        e = min(inst.exits, key=lambda x: inst.travel(loc, x))
        print(f"{prefix}|_ TELEPORT home via W{e}  (end run)")
        return
    u = act[1]
    if u < inst.n:
        print(f"{prefix}|_ go collect t{u}  (travel {inst.travel(loc, u):.0f})")
    else:
        print(f"{prefix}|_ go to detector D{u} over {set(inst.faces[u])}  (travel {inst.travel(loc, u):.0f})")
    for p, r, c2, b2 in solver.arrive(u, collected, belief):
        if u < inst.n:
            tag = (f"PRESENT  p={p:.2f}  +{inst.value[u]:.0f}" if c2 != collected
                   else f"absent   p={p:.2f}")
        else:
            ispos = all(w & solver.face_mask[u] for w in b2)
            tag = (f"DETECT>=1 p={p:.2f}" if ispos else f"clear     p={p:.2f}")
        print(f"{prefix}    [{tag}]   belief {len(belief)}->{len(b2)}")
        print_tree(solver, acts, u, c2, b2, prefix + "        ")


# ------------------------------------------------------------- entry points ---
def main():
    inst, worlds = instance_A()
    solver = Solver(inst, worlds)
    lam, acts = solver.optimal_rate()
    ER, ET = solver.eval_policy(acts)
    s_rate, _ = static_optimal_rate(inst, worlds)
    print(f"baseline: adaptive lambda*={lam:.5f}  (E[R]/E[T]={ER:.3f}/{ET:.2f}), "
          f"static={s_rate:.5f}, gain {(lam-s_rate)/s_rate*100:+.1f}%")
    print_tree(solver, acts, inst.entry, 0, solver.worlds0)


def verify():
    print("=" * 72)
    print("VERIFICATION: a guaranteed-but-absurdly-distant token node must be rejected")
    print("=" * 72)

    base_inst, base_w = instance_A()
    base = Solver(base_inst, base_w)
    lam0, _ = base.optimal_rate()

    inst, worlds = instance_A_distant()
    far = 6                                               # the distant always-present token
    solver = Solver(inst, worlds)
    lam, acts = solver.optimal_rate()
    ER, ET = solver.eval_policy(acts)

    n_states = len(acts)
    n_go_far = sum(1 for a in acts.values() if a == ("go", far))

    mc_rate, mc_R, mc_T, mc_far = solver.monte_carlo(acts, far=far)
    exp_rate, exp_far = solver.exploratory_rollout(acts, eps=0.15, far=far)

    d_entry = inst.travel(inst.entry, far)
    d_back = min(inst.travel(far, e) for e in inst.exits)
    token_rate = inst.value[far] / (d_entry + d_back + inst.teleport_time)

    print(f"\nThe token node t{far}: ALWAYS present (+{inst.value[far]:.0f}), at distance "
          f"{d_entry:.0f} from entry (round-trip ~{d_entry+d_back:.0f}s).")
    print(f"  Its standalone marginal rate: {inst.value[far]:.0f}/{d_entry+d_back+inst.teleport_time:.0f}"
          f" = {token_rate:.6f}/s  -- about {lam/token_rate:.0f}x below lambda*.\n")

    print("--- does adding t6 change the optimum? ---")
    print(f"  lambda* without t6 : {lam0:.5f}")
    print(f"  lambda* with    t6 : {lam:.5f}   (delta = {lam-lam0:+.2e}  -> unchanged)")
    print(f"  exact E[R]/E[T]    : {ER:.4f}/{ET:.2f} = {ER/ET:.5f}")

    print("\n--- does the OPTIMAL policy ever go to t6? ---")
    print(f"  reachable states solved                 : {n_states}")
    print(f"  states whose optimal action is 'go t6'  : {n_go_far}")
    print(f"  optimal-policy MC (20k): rate={mc_rate:.5f},  runs that visited t6 = {mc_far*100:.2f}%")

    print("\n--- what would EXPLORATION do? (epsilon-greedy, eps=0.15) ---")
    print(f"  exploratory rate = {exp_rate:.5f}  (<< lambda*),  runs that visited t6 = {exp_far*100:.1f}%")

    ok = (n_go_far == 0 and mc_far == 0.0 and abs(lam - lam0) < 1e-6)
    print("\n" + ("VERDICT: PASS -- the planner rejects t6 in every reachable state; "
                   "exploration samples it and pays for it." if ok
                   else "VERDICT: FAIL -- t6 leaked into the optimal policy."))


if __name__ == "__main__":
    verify()
