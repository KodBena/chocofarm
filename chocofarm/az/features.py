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
    geometry (per-treasure / per-detector / per-teleport distances are recomputed per `loc`,
    but the normalizer, teleport keys, and detector cover masks are fixed).

    `build(loc, bw, collected, marg=None)` — pass a pre-computed `marg = env.marginals(bw)` to
    reuse the single per-node marginals call (the F7 amortization); otherwise it is computed
    here. Returns a float64 vector of length `feature_dim(env)`."""

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

    def build(self, loc, bw, collected, marg=None) -> np.ndarray:
        env = self.env
        if marg is None:
            marg = env.marginals(bw)
        nb = len(bw)
        out = np.empty(self.dim, dtype=np.float64)
        o = 0

        # --- per-treasure block (N × 4): marg, collected, available, dist ---
        coll = np.zeros(self.N)
        for i in collected:
            coll[i] = 1.0
        avail = ((marg > 0) & (coll == 0)).astype(np.float64)  # legal-collect mask
        dist_t = np.array([env.d(loc, ("t", i)) for i in range(self.N)]) / self.diag
        out[o:o + self.N] = marg; o += self.N
        out[o:o + self.N] = coll; o += self.N
        out[o:o + self.N] = avail; o += self.N
        out[o:o + self.N] = dist_t; o += self.N

        # --- per-detector block (nD × 3): informative (open-clause), p_pos, dist ---
        if nb:
            hit = (bw[:, None] & self.cover[None, :]) != 0   # (nb, nD) bool
            p_pos = hit.mean(0)                              # P(positive read | belief)
            any_hit = hit.any(0)
            any_miss = (~hit).any(0)
            informative = (any_hit & any_miss).astype(np.float64)  # outcome still uncertain
        else:
            p_pos = np.zeros(self.nD)
            informative = np.zeros(self.nD)
        dist_d = np.array([env.d(loc, ("d", i)) for i in self.detectors]) / self.diag
        out[o:o + self.nD] = informative; o += self.nD
        out[o:o + self.nD] = p_pos; o += self.nD
        out[o:o + self.nD] = dist_d; o += self.nD

        # --- global block (5 + n_teleports) ---
        out[o] = (math.log(nb) / self.log_nworlds) if nb else 0.0; o += 1   # belief sharpness
        out[o] = len(collected) / self.K; o += 1                            # n_collected
        out[o] = float(marg.sum()) / self.K; o += 1                         # Σmarg (≈ remaining present)
        out[o] = env.exit_cost(loc) / self.diag; o += 1                     # early-exit geometry
        out[o] = 1.0 if nb else 0.0; o += 1                                 # non-empty belief flag
        for k in self.tele_keys:
            out[o] = env.d(loc, ("w", k)) / self.diag; o += 1               # per-teleport distance

        assert o == self.dim, (o, self.dim)
        return out
