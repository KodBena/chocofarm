#!/usr/bin/env python3
"""
chocofarm AZ bench — capture representative (loc, bw, collected) states for the hot-path bench.

Runs a handful of net-guided episodes once and snapshots every leaf belief the search reaches,
so the micro-benchmark exercises the SAME state distribution the live loop does: |bw| spanning
from the full world-set (15,504) down to the small post-sense beliefs. Saved to an npz the bench
loads (so the bench itself is deterministic and does NOT depend on a checkpoint being present).

The states are stored as a ragged set: a flat int64 array of all belief world-ids, an offsets
array delimiting each state's slice, a parallel array of loc-encodings, and the collected sets
(as bitmasks over treasures). `load_states(path)` reconstructs the list of (loc, bw, collected).

Run (pinned + bounded), pointing at any policy+value npz:
    PYTHONPATH=. taskset -c 2 timeout 120 python -m chocofarm.az.bench.capture_states \\
        --net .scratch/net.npz --out chocofarm/az/bench/states.npz --episodes 6
"""
from __future__ import annotations

import argparse
from typing import Any

import numpy as np
import numpy.typing as npt

from chocofarm.model.env import Collected, Environment, Loc, WorldSet
from chocofarm.az.features import FeatureBuilder
from chocofarm.az.mlp import ValueMLP
from chocofarm.az.gumbel_search import GumbelAZSearch, _Node
from chocofarm.az.exit_loop import generate_episode


# loc is ("w", key) | ("t", i) | ("d", i). Encode as (kind_code, idx) where kind_code is the
# index into _KINDS and idx is the integer id (teleport keys are mapped to their position).
_KINDS = ("w", "t", "d")

# Snap: (loc_kind_code, loc_idx, collected_bitmask, bw_array)
_Snap = tuple[int, int, int, npt.NDArray[np.int64]]
# State: (loc, bw, collected_set) — the representation load_states returns
_State = tuple[Loc, npt.NDArray[np.int64], set[int]]


def _encode_loc(env: Environment, loc: Loc) -> tuple[int, int]:
    kind, i = loc
    if kind == "w":
        # In this branch Loc is tuple[Literal["w"], str], so i is str — the index() call is typed.
        idx = list(env.teleports.keys()).index(i)  # type: ignore[arg-type]  # i is str|int at union; narrowed to str by kind=="w"
    else:
        idx = int(i)  # i is int for "t"/"d" branches of Loc; int() is the safe coercion here
    return _KINDS.index(kind), idx


def _decode_loc(env: Environment, code: int, idx: int) -> Loc:
    kind = _KINDS[code]
    if kind == "w":
        return ("w", list(env.teleports.keys())[idx])
    return (kind, idx)  # type: ignore[return-value]  # kind is "t"|"d" and idx is int — valid Loc


def capture(net_path: str, episodes: int = 6, seed: int = 2024,
            max_states: int = 4000) -> list[_Snap]:
    env = Environment()
    fb = FeatureBuilder(env)
    net = ValueMLP.load(net_path)
    search = GumbelAZSearch(net, env, m=12, n_sims=48)
    rng = np.random.default_rng(seed)

    snaps: list[_Snap] = []   # (loc_code, loc_idx, collected_mask, bw_array)
    seen: set[tuple[Any, ...]] = set()

    import chocofarm.az.gumbel_search as gs
    orig = gs.GumbelAZSearch._evaluate

    def patched(self: GumbelAZSearch, node: _Node, loc: Loc, bw: WorldSet,
                collected: Collected) -> float:
        key: tuple[Any, ...] = (loc, len(bw),
                                int(bw[0]) if len(bw) else 0,
                                int(bw[-1]) if len(bw) else 0)
        if key not in seen and len(snaps) < max_states:
            seen.add(key)
            cmask = 0
            for c in collected:
                cmask |= (1 << c)
            lc, li = _encode_loc(env, loc)
            snaps.append((lc, li, cmask, np.asarray(bw, dtype=np.int64).copy()))
        return orig(self, node, loc, bw, collected)

    gs.GumbelAZSearch._evaluate = patched  # type: ignore[method-assign]  # bench-only monkeypatch; restored in `finally`
    try:
        for _ in range(episodes):
            w = int(rng.choice(env.worlds))
            generate_episode(env, search, fb, w, 0.0855, rng, 0)
    finally:
        gs.GumbelAZSearch._evaluate = orig  # type: ignore[method-assign]  # restoring original after bench capture

    # also include the full root belief explicitly (|bw| = 15504), the heaviest single state
    lc, li = _encode_loc(env, ("w", env.entry))
    snaps.insert(0, (lc, li, 0, np.asarray(env.worlds, dtype=np.int64).copy()))
    return snaps


def save_states(snaps: list[_Snap], path: str) -> None:
    flat = np.concatenate([s[3] for s in snaps]) if snaps else np.zeros(0, np.int64)
    lens = np.array([len(s[3]) for s in snaps], dtype=np.int64)
    offsets = np.zeros(len(lens) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(lens)
    loc_codes = np.array([s[0] for s in snaps], dtype=np.int64)
    loc_idxs = np.array([s[1] for s in snaps], dtype=np.int64)
    cmasks = np.array([s[2] for s in snaps], dtype=np.int64)
    np.savez(path, flat=flat, offsets=offsets, loc_codes=loc_codes,
             loc_idxs=loc_idxs, cmasks=cmasks)


def load_states(path: str) -> tuple[Environment, list[_State]]:
    """Returns (env, list of (loc, bw, collected_set))."""
    env = Environment()
    z = np.load(path)
    flat: npt.NDArray[np.int64] = z["flat"]
    offsets: npt.NDArray[np.int64] = z["offsets"]
    loc_codes: npt.NDArray[np.int64] = z["loc_codes"]
    loc_idxs: npt.NDArray[np.int64] = z["loc_idxs"]
    cmasks: npt.NDArray[np.int64] = z["cmasks"]
    out: list[_State] = []
    for k in range(len(loc_codes)):
        bw = flat[offsets[k]:offsets[k + 1]].copy()
        loc = _decode_loc(env, int(loc_codes[k]), int(loc_idxs[k]))
        cmask = int(cmasks[k])
        collected: set[int] = {i for i in range(env.N) if (cmask >> i) & 1}
        out.append((loc, bw, collected))
    return env, out


def main() -> None:
    ap = argparse.ArgumentParser(description="Capture hot-path bench states.")
    ap.add_argument("--net", required=True, help="policy+value npz")
    ap.add_argument("--out", required=True)
    ap.add_argument("--episodes", type=int, default=6)
    ap.add_argument("--seed", type=int, default=2024)
    ap.add_argument("--max-states", type=int, default=4000)
    args = ap.parse_args()
    snaps = capture(args.net, args.episodes, args.seed, args.max_states)
    save_states(snaps, args.out)
    sizes = sorted(len(s[3]) for s in snaps)
    print(f"captured {len(snaps)} states -> {args.out}")
    print(f"  |bw| range: min={sizes[0]} median={sizes[len(sizes)//2]} max={sizes[-1]}")
    buckets = [0, 0, 0, 0]
    for n in sizes:
        if n >= 10000:
            buckets[0] += 1
        elif n >= 1000:
            buckets[1] += 1
        elif n >= 100:
            buckets[2] += 1
        else:
            buckets[3] += 1
    print(f"  buckets: >=10k:{buckets[0]} 1k-10k:{buckets[1]} 100-1k:{buckets[2]} <100:{buckets[3]}")


if __name__ == "__main__":
    main()
