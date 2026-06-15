#!/usr/bin/env python3
"""
chocofarm AZ — the action↔slot mapping for the policy head (design §3, staleness-corrected).

The policy head emits logits over a FIXED, env-derived slot space so the output layer is stable
while the LEGAL set shrinks/grows with the belief (masking handles legality; a face that becomes
uninformative is simply masked, no re-indexing — design §3). The mapping is kept HERE, in one
place, so `mlp.py`, `gumbel_search.py`, and `exit_loop.py` all agree on it by construction.

The design doc's "37-slot" space is STALE (it assumed the superseded 16-region detector model).
The honest `env` carries `env.N` collects + `len(env.detectors)` arrangement-FACE senses + 1
TERMINATE. On the live instance that is 20 + 44 + 1 = **65 slots**. Everything is derived from
`env`; nothing is hardcoded.

  slot 0 .. N-1            -> ("t", i)           collect treasure i = slot
  slot N .. N+nD-1         -> ("d", slot - N)    sense face id (slot - N)
  slot N+nD               -> TERMINATE          (always legal)

The legal mask is read straight from the FeatureBuilder's per-treasure `available[i]` block and
per-detector `informative[j]` block (the §2.2 features that ARE the mask, design §3), plus the
always-legal TERMINATE slot — so building the mask costs nothing beyond the feature vector that
was already built. `legal_mask_from_features` does exactly that, by slicing the known blocks.
"""
from __future__ import annotations

import weakref
from typing import Any

import numpy as np
import numpy.typing as npt

from chocofarm.model.env import Action, Collected, Environment, Loc, TERMINATE, WorldSet
from chocofarm.az.features import feature_layout


def n_action_slots(env: Environment) -> int:
    """Fixed action-space size for THIS env. Derived, never hardcoded (= 65 on the live env)."""
    return env.N + len(env.detectors) + 1


def term_slot(env: Environment) -> int:
    """Index of the TERMINATE slot (the last one)."""
    return env.N + len(env.detectors)


# Slot<->action lookup tables, a WeakKeyDictionary keyed by the ENV OBJECT itself (audit R9).
# The mapping is a fixed env-derived bijection (design §3), computed once per env and served by
# O(1) lookup — eliminating the ~3.5M per-element function-body executions the search's edge loops
# incurred (hot-path profile). The tables encode EXACTLY the same bijection the original branch
# logic did (asserted by tests/test_az_loop.py::test_action_slot_bijection), so this is a
# structural memoization, not a behavioral change.
#
# The key is the env object (a weak reference), NOT id(env). Environment instances are
# weak-referenceable (no __slots__) and identity-hashable (Environment defines no __eq__), so each
# distinct env object — including every copy-on-write restrict()/with_scenario view — gets its OWN
# entry. Weak refs mean the entry is dropped automatically when the env is GC'd (no leak — the old
# id(env) dict never evicted), and an entry tied to the object's lifetime (not its address) can
# never alias a different env at a reused CPython address (the old id(env) address-reuse hazard).
# A restrict()-ed env (smaller detector subset → smaller n_action_slots) is a distinct object, so
# it gets its OWN correctly-computed tables, not the parent's — the copy-on-write correctness the
# id() global accidentally had, which a naive env.__init__ attribute would have broken.
#
# DEVIATION (audit R9 literal "env.slot_tables attribute"): an env attribute would require env to
# compute these AZ tables, a features→env→features import cycle; this WeakKeyDictionary keyed by env
# achieves R9's intent (kill the leak + the address-reuse hazard) WITHOUT the cycle.
#
# Hot loops that convert millions of times should hoist the tables once via `slot_action_tables(env)`
# and index them directly, rather than calling the wrapper per element.
_SlotTables = tuple[tuple[Action, ...], dict[Action, int]]
_SLOT_TABLES: "weakref.WeakKeyDictionary[Environment, _SlotTables]" = weakref.WeakKeyDictionary()


def slot_action_tables(env: Environment) -> _SlotTables:
    """Return (slot_to_action_tuple, action_to_slot_dict) for `env`, building+caching on first
    use. `slot_to_action_tuple[s]` is the action for slot s; `action_to_slot_dict[a]` the slot
    for action a (TERMINATE included)."""
    tabs = _SLOT_TABLES.get(env)
    if tabs is None:
        N, nD = env.N, len(env.detectors)
        # spell the tag as the precise Action members so the bijection's static type IS the seam's
        # Action union, not the widened tuple[str, int] a bare ("t", i) literal infers.
        collects: tuple[Action, ...] = tuple(("t", i) for i in range(N))
        senses: tuple[Action, ...] = tuple(("d", j) for j in range(nD))
        s2a: tuple[Action, ...] = collects + senses + (TERMINATE,)
        a2s = {a: s for s, a in enumerate(s2a)}
        tabs = (s2a, a2s)
        _SLOT_TABLES[env] = tabs
    return tabs


def action_to_slot(env: Environment, action: Action) -> int:
    """('t',i) / ('d',j) / TERMINATE  ->  fixed slot id. O(1) via the cached bijection table."""
    s = slot_action_tables(env)[1].get(action)
    if s is not None:
        return s
    raise ValueError(f"unknown action {action!r}")


def slot_to_action(env: Environment, slot: int) -> Action:
    """fixed slot id  ->  ('t',i) / ('d',j) / TERMINATE. O(1) via the cached bijection table."""
    s2a = slot_action_tables(env)[0]
    if 0 <= slot < len(s2a):
        return s2a[slot]
    raise ValueError(f"slot {slot} out of range for action space {n_action_slots(env)}")


def legal_mask(env: Environment, loc: Loc, bw: WorldSet,
               collected: Collected) -> npt.NDArray[np.float64]:
    """{0,1} mask over the fixed slots from `env.legal_actions` (+ always-legal TERMINATE).

    The authoritative legality source is `env.legal_actions` — this maps its output onto slots.
    `legal_mask_from_features` is the cheaper hot-path variant that reuses an already-built
    feature vector; both must agree (a test in tests/ asserts it)."""
    m = np.zeros(n_action_slots(env), dtype=np.float64)
    for a in env.legal_actions(loc, bw, collected):
        m[action_to_slot(env, a)] = 1.0
    m[term_slot(env)] = 1.0   # TERMINATE is always legal
    return m


def legal_mask_from_features(env: Environment,
                             feat: npt.NDArray[Any]) -> npt.NDArray[np.float64]:
    """The hot-path mask: slice the feature blocks that ARE the mask (design §3).

    Reads the named blocks straight from the §2.2 layout owner, `FeatureLayout` (features.py),
    rather than re-deriving the slice offsets as literals: the per-treasure `available[i]` block is
    the legal-collect mask, and the per-detector `informative[j]` block is the open-clause /
    legal-sense mask. TERMINATE is always legal. Asking the layout for the slice (instead of
    hardcoding `2N..3N` / `5N..5N+nD`) means a block reorder in FeatureLayout moves this read in
    lockstep — no silent mislabel. This costs only array slicing — no env calls."""
    N, nD = env.N, len(env.detectors)
    layout = feature_layout(env)   # O(1) env-keyed memo — no per-call table rebuild on this hot path
    m = np.zeros(n_action_slots(env), dtype=np.float64)
    # per-treasure available[i] is the legal-collect mask (named block, not a magic offset).
    avail = feat[layout["available"]]
    m[0:N] = (avail > 0).astype(np.float64)
    # per-detector informative[j] is the open-clause / legal-sense mask (named block).
    informative = feat[layout["informative"]]
    m[N:N + nD] = (informative > 0).astype(np.float64)
    m[N + nD] = 1.0
    return m
