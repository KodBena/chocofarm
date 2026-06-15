#!/usr/bin/env python3
"""
chocofarm/model/instance.py — the SINGLE instance loader + the C(N,K) world array.

The one place that resolves and parses `data/instance.json` into an immutable
`Instance` (treasures, teleports, K) and the one home for the exactly-K-of-N
equiprobable-worlds bitmask array. Replaces the per-file inline `json.load` + the
hardcoded `K=5` literal (env.py, analyzer.py) and the verbatim world-array
comprehension duplicated across env.py / bounds/minienv.py / analysis/analyzer.py.

Public Domain (The Unlicense).
"""
from __future__ import annotations

import itertools
import json
import os
from dataclasses import dataclass

import numpy as np

DEFAULT_INSTANCE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "data", "instance.json")


@dataclass(frozen=True)
class Instance:
    """The parsed chocofarm instance: the treasure coordinates, the teleport
    coordinates, and the exactly-K-of-N present-count. `N` is derived from the
    treasure count (never stored), exactly as the model has always computed it."""
    treasures: dict          # {id(int) -> (x, y)}
    teleports: dict          # {name(str) -> (x, y)}
    K: int

    @property
    def N(self) -> int:
        return len(self.treasures)


def load_instance(path: str | None = None) -> Instance:
    """Resolve + parse the instance file into an `Instance`.

    With `path=None` this resolves the SAME package-relative file the model has
    always used (`<this dir>/../data/instance.json`) and parses it identically:
    treasures as `{int(i): tuple(xy)}`, teleports as `{k: tuple(v)}`, with `K`
    read from the data."""
    if path is None:
        path = DEFAULT_INSTANCE
    data = json.load(open(path))
    treasures = {int(i): tuple(xy) for i, xy in data["treasures"].items()}
    teleports = {k: tuple(v) for k, v in data["teleports"].items()}
    return Instance(treasures, teleports, int(data["K"]))


def world_array(N: int, K: int, support=None) -> np.ndarray:
    """The C(N,K) equiprobable worlds as a bitmask array (bit t set = τ_t present).

    `support=None` enumerates K-subsets of `range(N)`; a `support` iterable
    enumerates K-subsets of that subset instead (the minienv case), with the bit
    positions still the original treasure ids. Reproduces exactly
    `np.array([sum(1 << t for t in c) for c in
    itertools.combinations(<support or range(N)>, K)], dtype=np.int64)` — same
    elements, same order, same dtype."""
    items = range(N) if support is None else support
    return np.array(
        [sum(1 << t for t in c) for c in itertools.combinations(items, K)],
        dtype=np.int64)
