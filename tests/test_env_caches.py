#!/usr/bin/env python3
"""
test_env_caches.py — pins the WeakKeyDictionary env caches (audit R9).

Two module-global memos used to be keyed by `id(env)`: `actions._SLOT_TABLES` (the slot↔action
bijection) and `features._LAYOUTS` (the §2.2 FeatureLayout memo added in R6). `id(env)` is the
least value-stable key — CPython reuses freed addresses after GC (a new env at a freed address
could read a stale cache), and the dict never evicted (one entry per env ever built, a leak). R9
re-keys both by the ENV OBJECT itself in a `weakref.WeakKeyDictionary`: entries evict on GC (no
leak) and are tied to the object's lifetime, not its address (no aliasing across address reuse).

This test pins the behavior-preservation (same cached object on repeat — caching still works) AND
the copy-on-write correctness the weak-key scheme buys: a `restrict()`-ed env is a DISTINCT object
with a smaller action space, so it gets its OWN correctly-sized tables, not the parent's stale ones.
"""
import os
import sys
import weakref

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chocofarm.model.env import Environment
from chocofarm.az import actions, features
from chocofarm.az.actions import n_action_slots, slot_action_tables
from chocofarm.az.features import feature_layout


def test_caches_are_weak_key_dictionaries():
    """Both module-global memos are WeakKeyDictionary (audit R9) — so they evict on GC and key by
    the object's identity/lifetime, not its address. A plain dict here would be the old leak +
    address-reuse hazard."""
    assert isinstance(actions._SLOT_TABLES, weakref.WeakKeyDictionary)
    assert isinstance(features._LAYOUTS, weakref.WeakKeyDictionary)


def test_slot_action_tables_caches_same_object():
    """Behavior-preserving: repeat calls for the same env return the SAME cached object (the memo
    still works — it is not rebuilt per call)."""
    env = Environment()
    assert slot_action_tables(env) is slot_action_tables(env)


def test_feature_layout_caches_same_object():
    """Behavior-preserving: repeat calls for the same env return the SAME cached layout."""
    env = Environment()
    assert feature_layout(env) is feature_layout(env)


def test_copy_on_write_isolation_slot_tables():
    """A restrict()-ed env is a DISTINCT object with a smaller action space (fewer detectors), so it
    gets its OWN slot tables — NOT the parent's stale ones. This is the copy-on-write correctness the
    id() global accidentally had and a naive env attribute would have broken."""
    env = Environment()
    sub = env.restrict((8, 9, 10, 11, 12), 2)

    # sub has a strictly smaller action space (fewer detectors → fewer sense slots).
    assert n_action_slots(sub) != n_action_slots(env)
    assert n_action_slots(sub) < n_action_slots(env)

    s2a_env = slot_action_tables(env)[0]
    s2a_sub = slot_action_tables(sub)[0]

    # Distinct tables — sub did NOT inherit the parent's.
    assert s2a_sub is not s2a_env

    # Each reflects its OWN action space: the slot→action tuple is exactly n_action_slots long, and
    # sub's is shorter (it has its own detector subset).
    assert len(s2a_env) == n_action_slots(env)
    assert len(s2a_sub) == n_action_slots(sub)
    assert len(s2a_sub) < len(s2a_env)

    # sub's detector slots span exactly sub's detector count (one ('d', j) per kept detector),
    # mapping to sub's own positional detector ids — not the parent's 44.
    nD_sub = len(sub.detectors)
    det_slots = s2a_sub[sub.N:sub.N + nD_sub]
    assert det_slots == tuple(("d", j) for j in range(nD_sub))
    assert nD_sub < len(env.detectors)


def test_copy_on_write_isolation_feature_layout():
    """Likewise for the feature layout: a restricted env has fewer detectors → a SMALLER feature_dim,
    so feature_layout(sub) is its own descriptor with a distinct dim, not the parent's."""
    env = Environment()
    sub = env.restrict((8, 9, 10, 11, 12), 2)

    lay_env = feature_layout(env)
    lay_sub = feature_layout(sub)

    assert lay_sub is not lay_env
    # fewer detectors → smaller feature_dim (the dims differ, so this is a real isolation check).
    assert lay_sub.dim != lay_env.dim
    assert lay_sub.dim < lay_env.dim


def test_keep_accessor_full_and_restricted():
    """Environment.keep (audit item H) is the PUBLIC read of the legal-action treasure-id hook
    (`_treasure_ids`), so cross-module readers (bounds/eval_bound.py) stop reaching the private
    name. For a FULL env it is tuple(range(N)) = every treasure; for a restrict()-ed env it is the
    sorted `keep` tuple the restriction stored — byte-identical to what those readers got from the
    private `_treasure_ids` before. The internal hook is untouched (still what legal_actions
    iterates); `keep` is the read-only public alias."""
    env = Environment()
    assert tuple(env.keep) == tuple(range(env.N))     # full env: all N treasures

    sub = env.restrict((8, 9, 10, 11, 12), 2)
    assert sub.keep == (8, 9, 10, 11, 12)             # restricted: the sorted keep tuple

    # restrict sorts its `keep` arg; the accessor reflects that (sorted tuple regardless of input).
    assert env.restrict((12, 8, 11, 9, 10), 2).keep == (8, 9, 10, 11, 12)

    # byte-equivalent to the private hook it reads (tuple of `_treasure_ids`) — the equality the
    # eval_bound repoint relies on (set/len/display/clairvoyant arg all unchanged).
    assert env.keep == tuple(env._treasure_ids)
    assert sub.keep == tuple(sub._treasure_ids)


def test_keep_is_read_only():
    """`keep` is a read-only property: it must not become a second writer of the treasure-id hook
    (the one writer stays restrict/__init__ via `_treasure_ids`). Assigning to it raises."""
    env = Environment()
    try:
        env.keep = (1, 2, 3)
    except AttributeError:
        pass
    else:
        raise AssertionError("Environment.keep should be read-only (no setter)")
