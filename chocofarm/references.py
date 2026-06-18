#!/usr/bin/env python3
"""
chocofarm/references.py — the env-derived %VoI reference lines (floor / ceiling / anchor).

The NEUTRAL home for the three reference quantities the project plots %VoI against:
a realizable static floor (`realizable_static`), a clairvoyant rate ceiling
(`clairvoyant_rate`), the documented decomposition anchor (`DECOMP_ANCHOR`), and the
`BeliefRefs` SSOT that bundles them. These are functions of the env geometry (numpy +
the passed `Environment`) — nothing more — so they are a FOUNDATION both training (`az`)
and evaluation (`eval`) can depend on, not a piece of the eval harness.

Moved here (verbatim) from `chocofarm/eval/harness.py` to cut the backwards
`az → eval` import edge: `eval` is a consumer of training, not a foundation for it, so
`az/exit_loop.py` must not reach into `eval` for these env-derived numbers. `eval.harness`
re-exports these names for back-compat. This module's HARD deps are numpy + itertools + stdlib
(hashlib/struct/math); the clairvoyant cache lazily imports `redis` + `chocofarm.config` (an OPTIONAL
optimization — a redis outage degrades to a direct compute). It depends on neither `chocofarm.eval.*`
nor `chocofarm.az.*` (the cycle it exists to break).

Public Domain (Unlicense).
"""
from __future__ import annotations

import hashlib
import itertools
import math
import struct
from typing import TYPE_CHECKING, Any, cast

import numpy as np

if TYPE_CHECKING:
    from chocofarm.model.env import Environment, Loc

# The documented exact-decomposition rate (decomp exact, h=1) — the empirical decomp anchor
# reference line. Source: docs/agents/decomp-solver-report.md ("decomp (exact, h=1) 0.0941")
# and docs/results/decomp-rate.md. Hardcoded by maintainer decision (NOT env-derived): unlike the
# floor/ceiling it is a measured policy rate, not a function of the env geometry.
DECOMP_ANCHOR = 0.0941


def realizable_static(env: Environment) -> float:
    loc: Loc = ("w", env.entry)
    unv = set(range(env.N))
    route: list[int] = []
    t = 0.0
    best: tuple[float, int] = (-1.0, 0)
    while unv:
        i = max(unv, key=lambda j: env.value[j] / (env.d(loc, ("t", j)) + 1e-9))
        t += env.d(loc, ("t", i)); loc = ("t", i); route.append(i); unv.discard(i)
        rate = (env.K / env.N) * sum(env.value[r] for r in route) / (t + env.exit_cost(loc))
        if rate > best[0]:
            best = (rate, len(route))
    return best[0]


def clairvoyant_rate(env: Environment) -> float:
    def ev(lam: float, runs: int, seed: int) -> float:
        rng = np.random.default_rng(seed)
        totR = totT = 0.0
        for _ in range(runs):
            w = int(rng.choice(env.worlds))
            present = [t for t in range(env.N) if (w >> t) & 1]
            base = env.exit_cost(("w", env.entry))
            bv, bR, bT = -lam * base, 0.0, base
            for s in range(1, len(present) + 1):
                for sub in itertools.combinations(present, s):
                    R = sum(env.value[i] for i in sub)
                    bt = min(env.route_time(("w", env.entry), list(p))
                             for p in itertools.permutations(sub))
                    v = R - lam * bt
                    if v > bv:
                        bv, bR, bT = v, R, bt
            totR += bR; totT += bT
        return totR / totT
    lam = 0.0
    for _ in range(5):
        lam = ev(lam, 1000, 1)
    return ev(lam, 3000, 7)


# ---- the clairvoyant ceiling, behind an optimistic cache on the persistent (6379) redis ----
# clairvoyant_rate(env) brute-forces itertools.permutations over the worlds — seconds for a real instance,
# and a quarter of a SHORT run's wall (it amortizes to ~0 on a long run). It is a PURE function of the env
# geometry, so it is cached once per env on the disk-persisted noeviction REGISTRY instance
# (config.registry_redis_params(), 6379 — NOT the ephemeral 6380 transport): fetch if present, else
# compute and store. Pre-populate it out-of-band before a measurement run so the compute never lands
# inside the timed loop.
_CLAIRVOYANT_KEY_PREFIX = "choco:clairvoyant:v1:"


def _env_clairvoyant_fingerprint(env: Environment) -> str:
    """A stable hex digest of the env state `clairvoyant_rate` is a pure function of: N, K, value, entry,
    the teleport overhead, the teleport waypoints, and the full pairwise distance geometry (`route_time`/
    `exit_cost` derive from `env.d`). Two envs with the same digest yield the same rate; any geometry/value
    change flips the digest, so a stale ceiling is never served (ADR-0012 P6/P2)."""
    h = hashlib.sha256()
    h.update(repr((int(env.N), int(env.K), str(env.entry), float(env.tp),
                   tuple(float(v) for v in env.value),
                   tuple(sorted(str(k) for k in env.teleports)))).encode())
    locs = cast("list[Loc]",
                [("w", str(k)) for k in sorted(env.teleports)] + [("t", i) for i in range(int(env.N))])
    for a in locs:
        for b in locs:
            h.update(struct.pack("<d", float(env.d(a, b))))
    return h.hexdigest()


def _cache_redis() -> "Any | None":
    """The REGISTRY (6379 noeviction) redis client, or None if redis is unimportable/unreachable — the
    clairvoyant cache is OPTIMISTIC, so a redis outage degrades to a direct compute rather than failing the
    run (a logged warning, not a silent skip; ADR-0002). Bounded timeouts so a stall is loud, not a hang."""
    try:
        import redis
        from chocofarm import config
    except ImportError:
        return None
    params = config.registry_redis_params()
    try:
        r = redis.Redis(host=str(params["host"]), port=int(params["port"]), db=int(params["db"]),
                        socket_timeout=config.redis_socket_timeout(),
                        socket_connect_timeout=config.redis_connect_timeout())
        r.ping()
        return r
    except Exception as e:
        import sys
        print(f"[references] clairvoyant cache: registry redis unreachable ({e}); computing the ceiling "
              f"directly (uncached)", file=sys.stderr, flush=True)
        return None


def cached_clairvoyant_rate(env: Environment) -> float:
    """`clairvoyant_rate(env)` behind the optimistic 6379 cache (fetch → else compute+store). A redis
    outage degrades to a direct compute (logged); a PRESENT-but-malformed cached blob is a LOUD abort
    (ADR-0002 / P2 ACL — never run the %VoI line on a garbage ceiling). No TTL (persistent, like the hp
    registry that shares the instance)."""
    key = _CLAIRVOYANT_KEY_PREFIX + _env_clairvoyant_fingerprint(env)
    r = _cache_redis()
    if r is not None:
        raw = r.get(key)
        if raw is not None:
            try:                                  # redis.get returns bytes — decode before float()
                val = float(raw.decode() if isinstance(raw, (bytes, bytearray)) else raw)
            except (TypeError, ValueError, UnicodeDecodeError) as e:
                raise ValueError(f"clairvoyant cache blob at {key!r} is not a float ({raw!r})") from e
            if not math.isfinite(val):
                raise ValueError(f"clairvoyant cache blob at {key!r} is non-finite ({val})")
            return val
    rate = clairvoyant_rate(env)
    if r is not None:
        r.set(key, repr(float(rate)))
    return rate


class BeliefRefs:
    """Single source for the three %VoI reference lines and the %VoI map itself.

    These are the Tier-4 DERIVED reference lines the project plots %VoI against:
      - `static_floor`        = realizable_static(env)  — DERIVED from the env (the floor).
      - `clairvoyant_ceiling` = clairvoyant_rate(env)   — DERIVED from the env (the ceiling).
      - `decomp_anchor`       = DECOMP_ANCHOR            — the ONE documented constant (anchor),
                                                          the exact-decomposition rate (not env-derived).

    The floor and ceiling are a few seconds each to compute, so they are computed LAZILY on first
    access and MEMOIZED (never recomputed per call). This is the single source for %VoI: route every
    display reference-line site and every (rate → %VoI) conversion through here so they cannot drift.
    """

    def __init__(self, env: Environment):
        self.env = env
        self._static_floor: float | None = None
        self._clairvoyant_ceiling: float | None = None
        self.decomp_anchor = DECOMP_ANCHOR

    @property
    def static_floor(self) -> float:
        if self._static_floor is None:
            self._static_floor = realizable_static(self.env)
        return self._static_floor

    @property
    def clairvoyant_ceiling(self) -> float:
        if self._clairvoyant_ceiling is None:
            self._clairvoyant_ceiling = cached_clairvoyant_rate(self.env)
        return self._clairvoyant_ceiling

    def voi_pct(self, rate: float) -> float:
        """% of the clairvoyant value-of-information gap a `rate` claws back over the static floor."""
        return (rate - self.static_floor) / (self.clairvoyant_ceiling - self.static_floor) * 100
