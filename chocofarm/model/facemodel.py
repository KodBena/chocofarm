#!/usr/bin/env python3
"""
facemodel.py — the arrangement-face sense action.

The corrected detector model (docs/consults/consult-002-detector-misspec-report.md
§(4) "The correct model and remedy").  Where the old `cover_mask`
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
arrangement.Face and a metric `d`; it touches no solver.  It is the env's SINGLE
carrier of a face (audit item E — adopted): Environment builds
`self.senses = sense_actions(faces)` and its detector dynamics delegate to the
SenseAction's filter/observe/informative — see ENV_ADOPTION (bottom).

Public Domain (The Unlicense).
"""
from __future__ import annotations
import numpy as np
from chocofarm.model import arrangement as A


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
    decide policy.

    GEOMETRIC DERIVABILITY (maintainer's binding constraint).  Each face is a
    DERIVED object, never a frozen opaque table: it is the intersection-refinement
    of the atomic detectors, computed from the geometric data and reproducible
    end-to-end via
        scripts/chocobo_geometry.py  (parse chocobo.ggb -> the atomic detector
                                      regions Δ_j; data/instance.json regions_wkt)
        arrangement.arrangement(...) (polygonize(unary_union({∂Δ_j})) -> the atomic
                                      faces; each face's cover = {j : Δ_j ⊇ face},
                                      read at an interior rep-point — a REFINEMENT
                                      of the Δ_j, not a change: per region,
                                      ⋃(covers of its faces) == the old cover_mask)
            -> arrangement.persist() -> data/faces.json -> arrangement.load()
        scripts/{build_faces_ggb,verify_faces}.py  (visual round-trip + the
                                      self-check of that per-region union equality;
                                      docs/design/face-model-verification.md).
    A SenseAction MERELY CARRIES that derived face (position + cover + the
    observe/filter/informative reads); it freezes nothing the geometry does not
    already determine.  Re-running the pipeline regenerates faces.json and the
    senses follow — the face stays derived, not baked."""
    return [SenseAction(f) for f in (faces if faces is not None else A.load())]


# ---------------------------------------------------------------------------
# ENV_ADOPTION — how Environment DOES consume this (audit item E: adopted).
#
# `Environment.__init__` builds `self.senses = facemodel.sense_actions(faces)` (faces
# from `arrangement.load()`) and the SenseAction is the env's SINGLE carrier of a
# face — the env no longer reimplements filter/observe/informative inline beside a
# dead copy.  The detector dynamics DELEGATE to the SenseAction:
#     filter_detector(bw, i, pos)  ->  self.senses[i].filter(bw, pos)
#     legal_actions informative    ->  self.senses[i].informative(bw)
#     apply observe                ->  self.senses[i].observe(world)
# `det_pt` / `cover_mask` are served FROM the senses (`senses[i].rep_point` /
# `.bitmask`), so position and cover semantics are the *same* object (the face) and
# the consult-001 / cover_mask inconsistency cannot recur.
#
# ACTION SHAPE is the legacy ('d', id) — id is the face id (0..43), one detector
# action per arrangement face — NOT the ('f', k) this comment once speculatively
# prescribed (that shape never went live; the audit flagged the drift, and this
# note is corrected to the live shape).  The deliberate ('d', id) preservation is
# the honest reuse: re-keying regions->faces happened, but the action key still fit,
# so it was kept (ADR-0008).  Singleton-face pruning (70% of area, exact
# single-treasure probes) is a policy choice left to the solver, not baked here.
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    senses = sense_actions()
    bits = np.array([8, 9, 11, 12])
    k = max(range(len(senses)), key=lambda i: len(senses[i].cover))
    print(f"{len(senses)} sense actions; richest = {senses[k]!r} "
          f"(cover {sorted(senses[k].cover)})")
