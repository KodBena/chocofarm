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

import numpy as np

from chocofarm.model.env import TERMINATE


def n_action_slots(env) -> int:
    """Fixed action-space size for THIS env. Derived, never hardcoded (= 65 on the live env)."""
    return env.N + len(env.detectors) + 1


def term_slot(env) -> int:
    """Index of the TERMINATE slot (the last one)."""
    return env.N + len(env.detectors)


def action_to_slot(env, action) -> int:
    """('t',i) / ('d',j) / TERMINATE  ->  fixed slot id."""
    if action == TERMINATE:
        return env.N + len(env.detectors)
    kind, i = action
    if kind == "t":
        return i
    if kind == "d":
        return env.N + i
    raise ValueError(f"unknown action {action!r}")


def slot_to_action(env, slot: int):
    """fixed slot id  ->  ('t',i) / ('d',j) / TERMINATE."""
    nD = len(env.detectors)
    if slot < env.N:
        return ("t", slot)
    if slot < env.N + nD:
        return ("d", slot - env.N)
    if slot == env.N + nD:
        return TERMINATE
    raise ValueError(f"slot {slot} out of range for action space {n_action_slots(env)}")


def legal_mask(env, loc, bw, collected) -> np.ndarray:
    """{0,1} mask over the fixed slots from `env.legal_actions` (+ always-legal TERMINATE).

    The authoritative legality source is `env.legal_actions` — this maps its output onto slots.
    `legal_mask_from_features` is the cheaper hot-path variant that reuses an already-built
    feature vector; both must agree (a test in tests/ asserts it)."""
    m = np.zeros(n_action_slots(env), dtype=np.float64)
    for a in env.legal_actions(loc, bw, collected):
        m[action_to_slot(env, a)] = 1.0
    m[term_slot(env)] = 1.0   # TERMINATE is always legal
    return m


def legal_mask_from_features(env, feat: np.ndarray) -> np.ndarray:
    """The hot-path mask: slice the §2.2 feature blocks that ARE the mask (design §3).

    Feature layout (features.py): per-treasure block is [marg | collected | available | dist],
    each env.N wide; the `available[i]` sub-block is the legal-collect mask. The per-detector
    block is [informative | p_pos | dist], each nD wide; `informative[j]` is the open-clause /
    legal-sense mask. TERMINATE is always legal. This costs only array slicing — no env calls."""
    N, nD = env.N, len(env.detectors)
    m = np.zeros(n_action_slots(env), dtype=np.float64)
    # per-treasure block: available[i] is the 3rd sub-block (offset 2N .. 3N)
    avail = feat[2 * N:3 * N]
    m[0:N] = (avail > 0).astype(np.float64)
    # per-detector block starts at 4N; informative[j] is its 1st sub-block (offset 4N .. 4N+nD)
    informative = feat[4 * N:4 * N + nD]
    m[N:N + nD] = (informative > 0).astype(np.float64)
    m[N + nD] = 1.0
    return m
