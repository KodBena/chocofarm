#!/usr/bin/env python3
"""
minienv.py — genuinely small SUB-INSTANCES of the chocofarm env for VALIDATING the
information-relaxation dual bound (chocofarm/bounds/info_relaxation.py) without the
full 15,504-world compute (the live AZ job holds cores 0–3).

A MiniEnv is a real Environment restricted to a SUBSET of treasures, a reduced
present-count K, and only the faces whose cover lies entirely within the kept set.
Geometry, distances, costs, and cover-disjunctions are the REAL ones (delegated to the
parent env), so every number is honest; only the world-population shrinks — which
makes the inner DP's reachable-belief semilattice small enough to enumerate
EXACTLY (the validity-critical property, dual-bound.md §4).

This is the right shape of sub-instance for the four validation checks: a single
sense-cluster with a small K has a microscopic belief-MDP (decomp-rate.md: per-cluster
reachable beliefs are in the hundreds), so the inner solve is provably complete.
"""
from __future__ import annotations

import numpy as np

from chocofarm.model.instance import world_array


class MiniEnv:
    """A treasure-subset / reduced-K view of a real Environment.

    Exposes the env's `Policy`/bound interface (N, K, value, entry, worlds, marginals,
    filter_*, legal_actions, apply, d, exit_cost, nearest_exit, route_time, detectors,
    cover_mask, ...) but with:
      * treasures restricted to `keep` (a sorted tuple of original treasure ids);
      * K = `k_local` present among them (worlds = C(|keep|, k_local) bitmasks over the
        ORIGINAL bit positions, so distances/covers index correctly);
      * detectors restricted to faces whose cover ⊆ keep AND is non-empty over keep.

    Bit positions are the ORIGINAL treasure ids (not re-indexed), so `cover_mask`,
    `d`, `value`, presence bits all line up with the parent env unchanged. `marginals`
    returns the full-N vector (zeros off `keep`), matching the parent's contract."""

    def __init__(self, env, keep, k_local):
        self._env = env
        self.keep = tuple(sorted(keep))
        self.K = int(k_local)
        self.N = env.N                      # keep full N so bit indexing is unchanged
        self.value = env.value
        self.entry = env.entry
        self.tp = env.tp
        self.teleports = env.teleports
        self.coord = env.coord
        # worlds: K_local-of-keep present-sets, as bitmasks over ORIGINAL bit positions
        self.worlds = world_array(self.N, self.K, support=self.keep)
        # detectors: faces whose cover is non-empty and ⊆ keep
        keepset = set(self.keep)
        self.detectors = []
        self.cover_mask = {}
        self.det_pt = {}
        for fid in env.detectors:
            cm = env.cover_mask[fid]
            cover = [t for t in range(env.N) if (cm >> t) & 1]
            if cover and set(cover) <= keepset:
                self.detectors.append(fid)
                self.cover_mask[fid] = cm
                self.det_pt[fid] = env.det_pt[fid]

    # delegate geometry / belief mechanics to the parent (REAL distances + costs)
    def d(self, a, b):
        return self._env.d(a, b)

    def exit_cost(self, loc):
        return self._env.exit_cost(loc)

    def nearest_exit(self, loc):
        return self._env.nearest_exit(loc)

    def route_time(self, start, seq):
        return self._env.route_time(start, seq)

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

    def legal_actions(self, loc, bw, collected):
        marg = self.marginals(bw)
        acts = [("t", i) for i in self.keep if i not in collected and marg[i] > 0]
        for i in self.detectors:
            cm = self.cover_mask[i]
            if np.any((bw & cm) != 0) and np.any((bw & cm) == 0):
                acts.append(("d", i))
        return acts

    def apply(self, loc, bw, collected, action, world):
        kind, i = action
        dt = self.d(loc, (kind, i))
        if kind == "t":
            pres = bool((world >> i) & 1)
            r = self.value[i] if (pres and i not in collected) else 0.0
            nc = collected | {i} if pres else collected
            return r, (kind, i), self.filter_treasure(bw, i, pres), nc, dt
        pos = bool(world & self.cover_mask[i])
        return 0.0, (kind, i), self.filter_detector(bw, i, pos), collected, dt


def nw_cluster_mini(env, k_local=2):
    """The NW sense-cluster {8,9,10,11,12} with `k_local` present — a microscopic
    sub-instance (worlds = C(5,k_local)) with a tiny belief semilattice, ideal for the
    exact inner-DP validation."""
    return MiniEnv(env, keep=(8, 9, 10, 11, 12), k_local=k_local)
