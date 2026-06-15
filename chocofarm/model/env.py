#!/usr/bin/env python3
"""
chocofarm environment — the SIMULATION MODEL, decoupled from any solver.

Owns: the instance (treasures, arrangement-face sense actions, teleports, travel, values), the exact
belief mechanics (numpy world-set + filtering), the dynamics (legal actions, apply), and the
unbiased simulation/evaluation (simulate one episode; Monte-Carlo rate; Dinkelbach fixed
point). It knows nothing about HOW a decision is made — that is a `Policy` (see
chocofarm/solvers/base.py), passed in. New solution methods (NMCS, ISMCTS, …) are new
Policy subclasses; this file does not change.
"""
import math
import numpy as np

from chocofarm.model import arrangement as A
from chocofarm.model.instance import load_instance, world_array

TERMINATE = ("term", None)


class Environment:
    def __init__(self, instance_path=None, value=None, teleport_overhead=12.0, entry="CSNE"):
        inst = load_instance(instance_path)
        self.treasures = inst.treasures
        self.teleports = inst.teleports
        self.N, self.K = inst.N, inst.K
        self.value = list(value) if value is not None else [1.0] * self.N
        self.entry, self.tp = entry, float(teleport_overhead)

        # detectors: arrangement faces (consult-002 §4 / facemodel.ENV_ADOPTION). A sense action
        # is "stand at face F's representative point and read the disjunction over F's cover" —
        # cover and position are consistent BY CONSTRUCTION (the face is the single carrier of
        # both). This replaces the old `cover_mask[i] = {i} ∪ overlap-neighbours`, which read
        # the union over every face in Δ_i (a k=5 semantics) at one face's rep-point (a k≤2
        # position) — an over-approximation that handed out information no real sensor could.
        # The detector abstraction is re-keyed from regions to faces; the ('d', id) action shape
        # and every method below are UNCHANGED IN FORM — only the underlying data are faces.
        faces = A.load()                                          # 44 atomic arrangement faces
        self.detectors = list(range(len(faces)))                 # face ids 0..43 are the sense actions
        self.det_pt = {k: faces[k].rep_point for k in self.detectors}
        self.cover_mask = {k: faces[k].bitmask for k in self.detectors}

        self.coord = {}
        for i, xy in self.treasures.items():
            self.coord[("t", i)] = xy
        for i, xy in self.det_pt.items():
            self.coord[("d", i)] = xy
        for k, xy in self.teleports.items():
            self.coord[("w", k)] = xy

        self.worlds = world_array(self.N, self.K)

        # Precomputed inter-node distance table (perf). The coordinate set is STATIC for an
        # instance, so `d(a, b)` is a static function of the two coord keys; recomputing
        # `math.hypot` per call cost ~2M hypot calls per generated episode (the hot-path
        # profile). The table is built from the SAME `math.hypot(x1-x2, y1-y2)` inputs, so it is
        # bit-identical to the on-the-fly computation — a structural memoization, not an
        # approximation. 67 coord keys -> ~4.5k entries, a few ms to build, negligible memory.
        self._dist = {}
        coord_items = list(self.coord.items())
        for ka, (x1, y1) in coord_items:
            for kb, (x2, y2) in coord_items:
                self._dist[(ka, kb)] = math.hypot(x1 - x2, y1 - y2)

    # ---- geometry ----
    def d(self, a, b):
        """Distance between two coord keys. Served from the precomputed static table built at
        construction (same `math.hypot` inputs -> bit-identical); falls back to a live compute
        for any key pair not in the table (none arise in normal use, but keeps the contract
        total)."""
        v = self._dist.get((a, b))
        if v is not None:
            return v
        (x1, y1), (x2, y2) = self.coord[a], self.coord[b]
        return math.hypot(x1 - x2, y1 - y2)

    def exit_cost(self, loc):
        return min(self.d(loc, ("w", k)) for k in self.teleports) + self.tp

    def nearest_exit(self, loc):
        return min(self.teleports, key=lambda k: self.d(loc, ("w", k)))

    def route_time(self, start, seq):
        if not seq:
            return self.exit_cost(start)
        t = self.d(start, ("t", seq[0]))
        for a, b in zip(seq, seq[1:]):
            t += self.d(("t", a), ("t", b))
        return t + self.exit_cost(("t", seq[-1]))

    # ---- belief ----
    def marginals(self, bw):
        if len(bw) == 0:
            return np.zeros(self.N)
        return ((bw[:, None] >> np.arange(self.N)) & 1).mean(0)

    def filter_treasure(self, bw, i, present):
        bit = (bw >> i) & 1
        return bw[bit == (1 if present else 0)]

    def filter_detector(self, bw, i, pos):
        hit = (bw & self.cover_mask[i]) != 0
        return bw[hit if pos else ~hit]

    def sample_world(self, bw, rng):
        return int(rng.choice(bw))

    # ---- dynamics ----
    def legal_actions(self, loc, bw, collected):
        marg = self.marginals(bw)
        acts = [("t", i) for i in range(self.N) if i not in collected and marg[i] > 0]
        for i in self.detectors:
            cm = self.cover_mask[i]
            if np.any((bw & cm) != 0) and np.any((bw & cm) == 0):     # outcome still uncertain
                acts.append(("d", i))
        return acts

    def apply(self, loc, bw, collected, action, world):
        """Realise `action` against the true `world`. Returns (reward, loc', bw', collected', dt)."""
        kind, i = action
        dt = self.d(loc, (kind, i))
        if kind == "t":
            pres = bool((world >> i) & 1)
            r = self.value[i] if (pres and i not in collected) else 0.0
            nc = collected | {i} if pres else collected
            return r, (kind, i), self.filter_treasure(bw, i, pres), nc, dt
        pos = bool(world & self.cover_mask[i])
        return 0.0, (kind, i), self.filter_detector(bw, i, pos), collected, dt

    # ---- simulation / evaluation (solver-agnostic) ----
    def simulate(self, policy, world, lam, rng, max_steps=40):
        loc, bw, collected, R, T = ("w", self.entry), self.worlds, set(), 0.0, 0.0
        for _ in range(max_steps):
            a = policy.decide(self, loc, bw, collected, lam, rng)
            if a == TERMINATE:
                break
            r, loc, bw, collected, dt = self.apply(loc, bw, collected, a, world)
            R += r; T += dt
        return R, T + self.exit_cost(loc), self.nearest_exit(loc)

    def rate(self, policy, lam, runs, seed):
        rng = np.random.default_rng(seed)
        totR = totT = 0.0
        exits = {}
        for _ in range(runs):
            w = int(rng.choice(self.worlds))
            R, T, e = self.simulate(policy, w, lam, rng)
            totR += R; totT += T
            exits[e] = exits.get(e, 0) + 1
        return totR / totT, totR / runs, totT / runs, exits

    def dinkelbach_rate(self, policy, iters=4, warm_runs=600, final_runs=3000, seed=7, lam0=0.0):
        """A policy's own long-run rate = its Dinkelbach fixed point (lambda <- achieved rate)."""
        lam = lam0
        for _ in range(iters):
            lam = self.rate(policy, lam, warm_runs, seed=1)[0]
        rate, ER, ET, exits = self.rate(policy, lam, final_runs, seed)
        return {"lambda": lam, "rate": rate, "ER": ER, "ET": ET, "exits": exits}
