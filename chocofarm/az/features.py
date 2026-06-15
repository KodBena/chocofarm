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

  per-treasure  (env.N × 5):     marg[i], collected[i], available[i], dist[i], unc[i]
  per-detector  (len(env.detectors) × 3): informative[i], p_pos[i], dist[i]
  global        (6 + n_teleports): log|bw|/log Nworlds, n_collected/K, Σmarg/K,
                                    exit_cost(loc)/diag, |bw|>0 flag, Σ_uncollected unc[i],
                                    {dist to each teleport}

The belief-resolution block (Part C, az-parallel-exp): per-treasure `unc[i] = marg[i]·(1−marg[i])`
is the Bernoulli variance of treasure i's presence — 0 when the treasure is resolved (marg 0 or 1),
0.25 at maximum doubt (marg 0.5). It is the "is this one known or still in question" signal the
prior feature set lacked: the bare marginal cannot distinguish a resolved-absent treasure (marg 0,
unc 0) from one the belief is split on (marg 0.5, unc 0.25). The global `Σ_{uncollected} unc[i]`
is the expected number of still-in-question treasures — a scalar "how much belief structure remains
to resolve" the value head can read directly (the geometry/belief-dependent component the high-
variance MC target collapsed away from; see docs/results/az-parallel-exp.md). It sums only over
UNCOLLECTED treasures (a collected treasure carries no remaining decision-relevant uncertainty).

On the live instance (env.N=20, 44 faces, 3 teleports): 20·5 + 44·3 + (6+3) = 100 + 132 + 9 =
241 floats. This is LARGER than the prior 220 (the +20 per-treasure unc block + 1 global Σunc) and
than the doc's stale 90. `feature_dim(env)` reports the exact value; nothing downstream assumes a
constant — adding the block re-inits the net's input layer (no warm-start of W1; the trunk's other
rows are fine to re-learn from the richer input).

Distances are normalized by the coordinate-bbox diagonal (`map_diag`), so geometry the value
head needs (route cost, early-exit option) is O(1) and recoverable; marginals alone cannot
encode it (design §2.2).
"""
from __future__ import annotations

import math
import weakref

import numpy as np

from chocofarm.model.env import Environment
from chocofarm.solvers.ismcts import _belief_key
from chocofarm.az.dtypes import DTYPE
from chocofarm.az.kernels import belief_marg_cover


def map_diag(env: Environment) -> float:
    """Diagonal of the bounding box over all named coordinates (treasures, faces, teleports).
    The distance normalizer; a stable O(1) scale for the geometry features."""
    xs = [xy[0] for xy in env.coord.values()]
    ys = [xy[1] for xy in env.coord.values()]
    return math.hypot(max(xs) - min(xs), max(ys) - min(ys))


class FeatureLayout:
    """THE single owner of the §2.2 feature-vector layout (audit R6).

    The layout used to live, hand-kept in sync, in THREE independent places: the positional
    `out[o:o+N]=...; o+=N` accumulation in `FeatureBuilder.build`, the literal slice offsets
    `feat[2N:3N]` / `feat[5N:5N+nD]` in `actions.legal_mask_from_features`, and the per-element
    name/tag list in `feature_response.feature_names`. Reordering a block in one and not the
    others silently MISLABELED the vector with no error (and feature_response had zero test
    coverage). This descriptor is now the one owner: it encodes the ordered named blocks ONCE,
    derives every slice/name/tag from that single table, and the three consumers read from it.
    Reorder a block HERE and all consumers follow.

    Built from `env` (widths: N=env.N, nD=len(env.detectors), n_tel=len(env.teleports)). The
    ordered block table below IS the canonical §2.2 layout; the slices/names/tags are derived from
    it. Block order (start offsets in parens) gives `available` at 2N..3N and `informative` at
    5N..5N+nD — the offsets actions.py historically hardcoded — and total dim 5N+3nD+6+n_tel
    (= 241 on the live env).

    API:
      * `self.slices: dict[str, slice]` — keyed by block KEY (the first column of the table).
      * `self.dim: int` — sum of block widths (== `feature_dim(env)`).
      * `self[key] -> slice` — the slice for a named block (also via `self.slices[key]`).
      * `element_names() -> list[str]` — the per-element human-readable name for each of `dim`
        positions, in layout order (reproduces feature_response's historical output exactly).
      * `block_tags() -> list[str]` — the per-element block tag for each of `dim` positions.
    """

    def __init__(self, env: Environment):
        N = env.N
        nD = len(env.detectors)
        n_tel = len(env.teleports)
        # Canonical ordered block table — (key, width, group, display). `group` selects the
        # element-name / block-tag naming convention; `display` is the per-block display token.
        #   treasure : element "t{i}.{display}"            tag "treasure/{display}"
        #   detector : element "d{j}.{display}"            tag "detector/{display}"
        #   global   : element "global.{display}"          tag "global"
        #   teleport : element "global.tele_dist{k}"       tag "global"
        self.blocks: list[tuple[str, int, str, str]] = [
            # per-treasure (width N)
            ("marg",        N,     "treasure", "marg"),
            ("collected",   N,     "treasure", "collected"),
            ("available",   N,     "treasure", "available"),
            ("dist_t",      N,     "treasure", "dist"),
            ("unc",         N,     "treasure", "unc"),
            # per-detector (width nD)
            ("informative", nD,    "detector", "informative"),
            ("p_pos",       nD,    "detector", "p_pos"),
            ("dist_d",      nD,    "detector", "dist"),
            # global scalars (width 1 each)
            ("sharpness",   1,     "global",   "log|bw|"),
            ("n_collected", 1,     "global",   "n_collected"),
            ("marg_sum",    1,     "global",   "sum_marg"),
            ("exit_norm",   1,     "global",   "exit_cost"),
            ("nonempty",    1,     "global",   "nonempty"),
            ("sum_unc",     1,     "global",   "sum_unc"),
            # per-teleport (width n_tel)
            ("dist_w",      n_tel, "teleport", "tele_dist"),
        ]
        self.slices: dict[str, slice] = {}
        o = 0
        for key, width, _group, _display in self.blocks:
            self.slices[key] = slice(o, o + width)
            o += width
        self.dim: int = o
        # Fail-loud (ADR-0002): the blocks must partition [0, dim) contiguously — every position
        # owned by exactly one block, no gap, no overlap. This is the invariant `build`'s old
        # positional `assert o == self.dim` checked at write time; checking it ONCE here makes a
        # write through these slices provably fill the whole `np.empty` vector.
        covered = sorted((s.start, s.stop) for s in self.slices.values())
        cursor = 0
        for start, stop in covered:
            assert start == cursor and stop >= start, ("non-contiguous layout", start, stop, cursor)
            cursor = stop
        assert cursor == self.dim, (cursor, self.dim)

    def __getitem__(self, key: str) -> slice:
        return self.slices[key]

    def element_names(self) -> list[str]:
        """Per-element human-readable name for each of `self.dim` positions, in layout order."""
        names: list[str] = []
        for key, width, group, display in self.blocks:
            if group == "treasure":
                names.extend(f"t{i}.{display}" for i in range(width))
            elif group == "detector":
                names.extend(f"d{j}.{display}" for j in range(width))
            elif group == "global":
                names.extend(f"global.{display}" for _ in range(width))
            elif group == "teleport":
                names.extend(f"global.{display}{k}" for k in range(width))
            else:  # unreachable; fail-loud on an unknown group
                raise ValueError(f"unknown feature group {group!r} for block {key!r}")
        return names

    def block_tags(self) -> list[str]:
        """Per-element block tag for each of `self.dim` positions, in layout order."""
        tags: list[str] = []
        for key, width, group, display in self.blocks:
            if group == "treasure":
                tag = f"treasure/{display}"
            elif group == "detector":
                tag = f"detector/{display}"
            elif group in ("global", "teleport"):
                tag = "global"
            else:  # unreachable; fail-loud on an unknown group
                raise ValueError(f"unknown feature group {group!r} for block {key!r}")
            tags.extend(tag for _ in range(width))
        return tags


# Env-keyed memo for the layout descriptor — a WeakKeyDictionary keyed by the ENV OBJECT itself
# (audit R9). The layout is a fixed env-derived table (same idiom as actions._SLOT_TABLES), so
# build it once per env and serve O(1) — hot-path callers (legal_mask_from_features, run once per
# search node) get the named slices without re-building the 15-block table every call (the path the
# docstrings call "only array slicing — no env calls").
#
# The key is the env object (a weak reference), NOT id(env). Environment instances are
# weak-referenceable (no __slots__) and identity-hashable (Environment defines no __eq__), so each
# distinct env object — including every copy-on-write restrict()/with_scenario view (a restricted
# env has fewer detectors → a SMALLER feature_dim) — gets its OWN correctly-computed layout, not the
# parent's. Weak refs evict the entry on GC (no leak — the old id(env) dict never evicted), and an
# entry tied to the object's lifetime (not its address) cannot alias a different env at a reused
# CPython address (the old id(env) address-reuse hazard). See actions._SLOT_TABLES for the full R9
# rationale, including the deviation from the audit's literal "env attribute" (it would cycle).
_LAYOUTS = weakref.WeakKeyDictionary()


def feature_layout(env: Environment) -> "FeatureLayout":
    """Return the cached `FeatureLayout` for `env`, building+caching on first use (keyed by the env
    OBJECT in a WeakKeyDictionary — audit R9). Same env-derived-table memo idiom as
    actions.slot_action_tables."""
    lay = _LAYOUTS.get(env)
    if lay is None:
        lay = FeatureLayout(env)
        _LAYOUTS[env] = lay
    return lay


def feature_dim(env: Environment) -> int:
    """Exact feature-vector length for THIS env. Derived, never hardcoded.

    Single-sourced through `FeatureLayout` (the one owner of the layout): the per-treasure block
    is N×5 (marg, collected, available, dist, unc — the belief-resolution `unc` is Part C);
    per-detector is nD×3; global is (6 + n_teleports) (the +1 over the prior 5 is the global
    Σ_uncollected unc scalar). On the live env: 20·5 + 44·3 + (6+3) = 241."""
    return feature_layout(env).dim


class FeatureBuilder:
    """Builds the §2.2 feature vector for (loc, bw, collected). Pre-caches the env-static
    geometry (per-treasure / per-detector / per-teleport distances are STATIC given `loc` — the
    instance has only a fixed set of coordinate keys — so they are precomputed once per loc and
    served by lookup; the normalizer, teleport keys, and detector cover masks are fixed too).

    `build(loc, bw, collected)` — the marginals are derived internally by the fused numba kernel
    (a single pass over `bw` that also yields the detector counts), so no caller-supplied `marg`
    is needed. Returns a float64 vector of length `feature_dim(env)`.

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

    # The set of FeatureLayout block KEYs `build` writes — asserted equal to the layout's keys at
    # the end of every build, so a block added to FeatureLayout but not written here fails loud.
    _WRITTEN_KEYS = frozenset({
        "marg", "collected", "available", "dist_t", "unc",
        "informative", "p_pos", "dist_d",
        "sharpness", "n_collected", "marg_sum", "exit_norm", "nonempty", "sum_unc",
        "dist_w",
    })

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
        # The single owner of the §2.2 layout (audit R6): `build` writes THROUGH it by named
        # block, so reordering a block in FeatureLayout moves the write here in lockstep.
        self.layout = FeatureLayout(env)
        self.dim = self.layout.dim
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
    def _belief_feats(self, bw):
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
        # miss (or collision against a different belief): compute. The fused numba kernel
        # (kernels.belief_marg_cover) does the marginals AND the (nb×nD) detector reduction in a
        # SINGLE pass over `bw` — ~12× the numpy form across the whole |bw| distribution, the new
        # #1 hot path. It returns integer-exact `cnt` (== numpy count_nonzero) and the marginals,
        # so it folds the separate `env.marginals(bw)` call into the same pass. The kernel is
        # int/float64 internally (bit-exact); the dtype cast to DTYPE happens in `build`.
        if nb:
            marg, cnt = belief_marg_cover(bw, self.cover, self.N)
            p_pos = cnt / nb
            informative = ((cnt > 0) & (cnt < nb)).astype(np.float64)
            sharpness = math.log(nb) / self.log_nworlds
        else:
            # empty belief: marginals are zero
            marg = np.zeros(self.N)
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

    def build(self, loc, bw, collected) -> np.ndarray:
        """Build the §2.2 feature vector at `(loc, bw, collected)`. The fused numba kernel computes
        the marginals AND the detector counts in one pass (kernels.belief_marg_cover), so the
        builder always derives its own marginals here — the kernel's marg is bit-identical to
        `env.marginals(bw)` (verified) and faster than a separate marginals call, so there is no
        caller-supplied `marg` to consume. Returns a `DTYPE` vector."""
        N = self.N
        lay = self.layout
        marg_in, p_pos, informative, marg_sum, sharpness, nb = self._belief_feats(bw)
        marg = marg_in
        dist_t, dist_d, dist_w, exit_norm = self._loc_block(loc)

        out = np.empty(self.dim, dtype=DTYPE)

        # --- per-treasure block (N × 5): marg, collected, available, dist, unc ---
        coll = np.zeros(N)
        for i in collected:
            coll[i] = 1.0
        avail = ((marg > 0) & (coll == 0)).astype(np.float64)  # legal-collect mask
        # belief-resolution (Part C): Bernoulli variance of treasure-i presence. 0 when resolved
        # (marg 0 or 1), 0.25 at max doubt (marg 0.5). The "known-vs-unknown" signal the bare
        # marginal cannot carry (a resolved-absent marg-0 and a split marg-0.5 both look "not here"
        # to a value head reading marg alone).
        unc = marg * (1.0 - marg)
        # Write THROUGH the layout (audit R6): each block goes to its named slice, so the order is
        # owned by FeatureLayout, not by a positional cursor here. The values keyed below MUST cover
        # every layout block — `_WRITTEN_KEYS` is asserted equal to the layout's keys (so adding a
        # block to FeatureLayout without writing it here fails loudly, ADR-0002).
        out[lay["marg"]] = marg
        out[lay["collected"]] = coll
        out[lay["available"]] = avail
        out[lay["dist_t"]] = dist_t
        out[lay["unc"]] = unc

        # --- per-detector block (nD × 3): informative (open-clause), p_pos, dist ---
        out[lay["informative"]] = informative
        out[lay["p_pos"]] = p_pos
        out[lay["dist_d"]] = dist_d

        # --- global block (6 + n_teleports) ---
        # Σ_uncollected unc[i] — expected number of treasures still in question (Part C). Only
        # uncollected treasures count: a collected one carries no remaining decision-relevant doubt.
        sum_u = float(np.sum(unc * (coll == 0)))
        out[lay["sharpness"]] = sharpness                  # belief sharpness
        out[lay["n_collected"]] = len(collected) / self.K  # n_collected
        out[lay["marg_sum"]] = marg_sum / self.K           # Σmarg (≈ remaining present)
        out[lay["exit_norm"]] = exit_norm                  # early-exit geometry
        out[lay["nonempty"]] = 1.0 if nb else 0.0          # non-empty belief flag
        out[lay["sum_unc"]] = sum_u                        # Σ_uncollected unc (belief-resolution)
        out[lay["dist_w"]] = dist_w                         # per-teleport distances

        # sanity: every layout block was written. The layout's constructor proved its slices
        # partition [0, dim) contiguously, so covering every block KEY here == the whole vector
        # is fully written (no `np.empty` garbage leaks through an unwritten slice).
        assert self._WRITTEN_KEYS == lay.slices.keys(), (self._WRITTEN_KEYS, lay.slices.keys())
        return out
