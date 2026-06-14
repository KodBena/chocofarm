#!/usr/bin/env python3
"""
chocofarm AZ — belief featurization (design §2.2), ADAPTED to the honest 44-face env.

The net must read the belief. The design's §2.2 vector was sized against the SUPERSEDED
16-region detector model (16 detectors, ~90 floats). The honest `env` carries 44 arrangement
FACES as the sense actions (many of them singletons), so we derive EVERY dimension from `env`
at construction — never hardcode 16 / 37 / 90.

The vector is the cheap, near-sufficient statistic F6 endorses: marginals + collected-mask +
the open-clause separators + geometry. It is built from a SINGLE `env.marginals(bw)` call (the
~1.2 ms per-node bottleneck, F7), cached by the caller across a node's action loop.

Layout (all blocks fixed-dimension, derived from env):

  per-treasure  (env.N × 4):     marg[i], collected[i], available[i], dist[i]
  per-detector  (len(env.detectors) × 3): informative[i], p_pos[i], dist[i]
  global        (5 + n_teleports): log|bw|/log Nworlds, n_collected/K, Σmarg/K,
                                    exit_cost(loc)/diag, |bw|>0 flag, {dist to each teleport}

On the live instance (env.N=20, 44 faces, 3 teleports): 20·4 + 44·3 + (5+3) = 80 + 132 + 8 =
220 floats. This is LARGER than the doc's 90 — expected, faces are more numerous than the stale
regions. `feature_dim(env)` reports the exact value; nothing downstream assumes a constant.

Distances are normalized by the coordinate-bbox diagonal (`map_diag`), so geometry the value
head needs (route cost, early-exit option) is O(1) and recoverable; marginals alone cannot
encode it (design §2.2).
"""
from __future__ import annotations

import math

import numpy as np

from chocofarm.model.env import Environment
from chocofarm.solvers.ismcts import _belief_key


def map_diag(env: Environment) -> float:
    """Diagonal of the bounding box over all named coordinates (treasures, faces, teleports).
    The distance normalizer; a stable O(1) scale for the geometry features."""
    xs = [xy[0] for xy in env.coord.values()]
    ys = [xy[1] for xy in env.coord.values()]
    return math.hypot(max(xs) - min(xs), max(ys) - min(ys))


def feature_dim(env: Environment) -> int:
    """Exact feature-vector length for THIS env. Derived, never hardcoded."""
    n_tel = len(env.teleports)
    return env.N * 4 + len(env.detectors) * 3 + (5 + n_tel)


class FeatureBuilder:
    """Builds the §2.2 feature vector for (loc, bw, collected). Pre-caches the env-static
    geometry (per-treasure / per-detector / per-teleport distances are STATIC given `loc` — the
    instance has only a fixed set of coordinate keys — so they are precomputed once per loc and
    served by lookup; the normalizer, teleport keys, and detector cover masks are fixed too).

    `build(loc, bw, collected, marg=None)` — pass a pre-computed `marg = env.marginals(bw)` to
    reuse the single per-node marginals call (the F7 amortization); otherwise it is computed
    here. Returns a float64 vector of length `feature_dim(env)`.

    Two perf caches, both behavior-preserving (structural — same values, computed fewer times):

      * `_loc_cache` — per-loc normalized distance block (dist_t | dist_d | per-teleport dist) and
        exit_cost. Distances are static given the instance, so this is exactly the same array the
        per-loc list comprehensions produced; it eliminates ~2M `env.d` calls / episode.

      * `_belief_cache` — the belief-derived intermediates (marg, p_pos, informative, Σmarg, the
        belief-sharpness scalar) keyed by `_belief_key` with `np.array_equal` collision
        verification. These depend ONLY on `bw`, and within an episode the same belief is reached
        ~3.5× on average (it is the (nb×nD) detector reduction + marginals — the ~40% feature
        bucket — that this reuses). The verification guards against the documented `_belief_key`
        collisions (distinct beliefs of equal size sharing min/max world ids), so a hit returns
        the features of the SAME belief: bit-identical to recomputing them."""

    def __init__(self, env: Environment):
        self.env = env
        self.N = env.N
        self.K = env.K
        self.detectors = list(env.detectors)
        self.nD = len(self.detectors)
        self.tele_keys = list(env.teleports.keys())
        self.diag = map_diag(env)
        self.log_nworlds = math.log(len(env.worlds))
        # cover masks as a contiguous int64 array, for a single vectorized p_pos over detectors.
        self.cover = np.array([env.cover_mask[i] for i in self.detectors], dtype=np.int64)
        self.dim = feature_dim(env)
        self._loc_cache = {}      # loc -> (dist_block (N+nD+n_tel,), exit_cost_norm)
        self._belief_cache = {}   # _belief_key -> list of (bw_ref, BeliefFeat)
        self._belief_cache_n = 0  # entry count, for the safety-net cap below
        # Safety cap: callers that drive the cache as an episode-scoped store (GumbelAZSearch
        # resets it per episode) keep it small, but a caller that never resets (e.g. an ISMCTS
        # leaf evaluator) would otherwise grow it unbounded. Clearing is always correctness-safe
        # (a hit only ever returns equal-belief features), so a generous cap bounds memory for
        # ANY caller — far above one episode's distinct-belief count (hundreds).
        self._belief_cache_cap = 50000

    def reset_belief_cache(self):
        """Drop the per-belief cache (call between unrelated search lifetimes if memory matters;
        correctness does not depend on it — the cache only ever returns features of a belief that
        compared equal)."""
        self._belief_cache.clear()
        self._belief_cache_n = 0

    # ---- per-loc static distance block (structural memo) ----
    def _loc_block(self, loc):
        cached = self._loc_cache.get(loc)
        if cached is not None:
            return cached
        env = self.env
        diag = self.diag
        dist_t = np.fromiter((env.d(loc, ("t", i)) for i in range(self.N)),
                             dtype=np.float64, count=self.N) / diag
        dist_d = np.fromiter((env.d(loc, ("d", i)) for i in self.detectors),
                             dtype=np.float64, count=self.nD) / diag
        dist_w = np.fromiter((env.d(loc, ("w", k)) for k in self.tele_keys),
                             dtype=np.float64, count=len(self.tele_keys)) / diag
        exit_norm = env.exit_cost(loc) / diag
        block = (dist_t, dist_d, dist_w, exit_norm)
        self._loc_cache[loc] = block
        return block

    # ---- belief-derived intermediates (cached by belief, verified on collision) ----
    def _belief_feats(self, bw, marg):
        """Returns (marg, p_pos, informative, marg_sum, sharpness, nb) — all functions of `bw`
        alone. Cached by `_belief_key`; a hit is verified with `np.array_equal` so a key
        collision never returns another belief's features."""
        nb = len(bw)
        key = _belief_key(bw)
        bucket = self._belief_cache.get(key)
        if bucket is not None:
            for bw_ref, feats in bucket:
                if bw_ref is bw or np.array_equal(bw_ref, bw):
                    return feats
        # miss (or collision against a different belief): compute
        if marg is None:
            marg = self.env.marginals(bw)
        if nb:
            hit = (bw[:, None] & self.cover[None, :]) != 0   # (nb, nD) bool
            # Single reduction (count of positive reads) instead of three (mean + any + ~any):
            # p_pos = cnt/nb is exactly hit.mean(0); cnt>0 is exactly hit.any(0); cnt<nb is
            # exactly (~hit).any(0). Same values, one pass over the (nb×nD) bool matrix.
            cnt = np.count_nonzero(hit, axis=0)
            p_pos = cnt / nb
            informative = ((cnt > 0) & (cnt < nb)).astype(np.float64)
            sharpness = math.log(nb) / self.log_nworlds
        else:
            p_pos = np.zeros(self.nD)
            informative = np.zeros(self.nD)
            sharpness = 0.0
        marg_sum = float(marg.sum())
        feats = (marg, p_pos, informative, marg_sum, sharpness, nb)
        # store a reference to bw so a later identity/equality check is sound (beliefs are
        # immutable in the model — every filter returns a fresh array — so a reference is safe)
        if self._belief_cache_n >= self._belief_cache_cap:
            self._belief_cache.clear()
            self._belief_cache_n = 0
        self._belief_cache.setdefault(key, []).append((bw, feats))
        self._belief_cache_n += 1
        return feats

    def build(self, loc, bw, collected, marg=None) -> np.ndarray:
        N, nD = self.N, self.nD
        marg_in, p_pos, informative, marg_sum, sharpness, nb = self._belief_feats(bw, marg)
        marg = marg_in
        dist_t, dist_d, dist_w, exit_norm = self._loc_block(loc)

        out = np.empty(self.dim, dtype=np.float64)
        o = 0

        # --- per-treasure block (N × 4): marg, collected, available, dist ---
        coll = np.zeros(N)
        for i in collected:
            coll[i] = 1.0
        avail = ((marg > 0) & (coll == 0)).astype(np.float64)  # legal-collect mask
        out[o:o + N] = marg; o += N
        out[o:o + N] = coll; o += N
        out[o:o + N] = avail; o += N
        out[o:o + N] = dist_t; o += N

        # --- per-detector block (nD × 3): informative (open-clause), p_pos, dist ---
        out[o:o + nD] = informative; o += nD
        out[o:o + nD] = p_pos; o += nD
        out[o:o + nD] = dist_d; o += nD

        # --- global block (5 + n_teleports) ---
        out[o] = sharpness; o += 1                          # belief sharpness
        out[o] = len(collected) / self.K; o += 1            # n_collected
        out[o] = marg_sum / self.K; o += 1                  # Σmarg (≈ remaining present)
        out[o] = exit_norm; o += 1                          # early-exit geometry
        out[o] = 1.0 if nb else 0.0; o += 1                 # non-empty belief flag
        out[o:o + len(dist_w)] = dist_w; o += len(dist_w)   # per-teleport distances

        assert o == self.dim, (o, self.dim)
        return out
