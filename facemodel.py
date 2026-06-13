#!/usr/bin/env python3
"""
facemodel.py — the arrangement-face sense action.

The corrected detector model (consult-002 §4).  Where the old `cover_mask`
asked "enter Δ_i and read {i} ∪ overlap-neighbours simultaneously" — a
disjunction no single point realises — the face model makes the *position* the
action:

        sense-action  :=  "go to face F"
        travel cost    =  d(loc, F.rep_point)
        observation    =  ( world & F.bitmask ) != 0
                       =  ⋁_{j ∈ F.cover}  τ_j present

Point and cover are consistent *by construction*: the disjunction read is
exactly the cover of the face the rep point sits in.  The old model's defect
was reading cover_mask[i] (the union over *all* faces in Δ_i) at det_pt[i] (one
particular face's rep point) — a k=2 position given k=5 semantics.

This module is deliberately small and effect-free.  It depends only on
arrangement.Face and a metric `d`; it touches no solver.  See ENV_ADOPTION
(bottom) for how Environment would consume it.
"""
from __future__ import annotations
import numpy as np
import arrangement as A


class SenseAction:
    """A single 'go to face F' action: where to stand, and what it tells you."""
    def __init__(self, face: A.Face):
        self.face = face
        self.bitmask = face.bitmask          # cover as bits
        self.cover = face.cover
        self.rep_point = face.rep_point

    # --- cost: travel to the face's interior representative point ---
    def cost(self, loc, dist) -> float:
        return dist(loc, self.rep_point)

    # --- belief update: the disjunction over the cover, against a world-set ---
    def filter(self, bw: np.ndarray, positive: bool) -> np.ndarray:
        hit = (bw & self.bitmask) != 0
        return bw[hit if positive else ~hit]

    def observe(self, world: int) -> bool:
        """The true reading at this face for a concrete world."""
        return bool(world & self.bitmask)

    def informative(self, bw: np.ndarray) -> bool:
        """Outcome still uncertain over the current belief — both polarities live."""
        hit = (bw & self.bitmask) != 0
        return bool(hit.any() and (~hit).any())

    def __repr__(self):
        return f"Sense{{{','.join(map(str, sorted(self.cover)))}}}@{self.rep_point}"


def sense_actions(faces=None):
    """The face sense-action set.  Singleton faces are exact single-treasure
    probes (τ_j? — 70% of area); the genuine disjunctive VoI lives in the k≥2
    faces.  Callers may prune dominated/singleton faces; the model does not
    decide policy."""
    return [SenseAction(f) for f in (faces if faces is not None else A.load())]


# ---------------------------------------------------------------------------
# ENV_ADOPTION — how Environment would consume this (no rewiring done here).
#
# Environment.__init__ replaces the cover_mask block with:
#     self.faces   = arrangement.load()                 # 44 faces
#     self.senses  = facemodel.sense_actions(self.faces)
#     # action coords: a sense action 's' is reached at s.rep_point
#     for k, s in enumerate(self.senses):
#         self.coord[("f", k)] = s.rep_point
#
# legal_actions: collects (unchanged) + informative faces + TERMINATE
#     acts = [("t", i) for i ... ]                       # collects, as today
#     acts += [("f", k) for k, s in enumerate(self.senses) if s.informative(bw)]
#
# apply, for a sense action ("f", k):
#     s   = self.senses[k]
#     dt  = self.d(loc, ("f", k))
#     pos = s.observe(world)
#     return 0.0, ("f", k), s.filter(bw, pos), collected, dt
#
# The action set becomes ~20 collects + |informative faces| sense actions +
# TERMINATE — replacing the fixed 16 detector actions.  det_pt and the cover
# semantics are now the *same* object (the face), so the consult-001 /
# cover_mask inconsistency cannot recur.  Singleton-face pruning is a policy
# choice left to the solver, not baked into the model.
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    senses = sense_actions()
    bits = np.array([8, 9, 11, 12])
    k = max(range(len(senses)), key=lambda i: len(senses[i].cover))
    print(f"{len(senses)} sense actions; richest = {senses[k]!r} "
          f"(cover {sorted(senses[k].cover)})")
