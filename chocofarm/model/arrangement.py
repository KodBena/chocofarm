#!/usr/bin/env python3
"""
arrangement.py — the planar arrangement of the detection polygons.

The detector cover is a *position-dependent* fact, not a per-region one.  Standing
at a point p, you read the disjunction over exactly the regions that contain p:

        cover(p) = { j : Δ_j ∋ p }.

This map is constant on each atomic face of the planar arrangement of the {Δ_j}
boundaries, so the faces — not the regions — are the natural carriers of a sense
action.  This module computes that arrangement and its cover function, the
"deferred reification" consult-002 specifies as the fix for the `cover_mask`
over-approximation.

A Face is a value:  (cover, rep_point, area, polygon_wkt).  Nothing here knows
about solvers, beliefs, or travel; see facemodel.py for the sense-action that
consumes a Face.
"""
from __future__ import annotations
import json
import os
from dataclasses import dataclass
from shapely import wkt
from shapely.geometry import Point
from shapely.ops import polygonize, unary_union

INSTANCE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "instance.json")
FACES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "faces.json")


@dataclass(frozen=True)
class Face:
    """An atomic arrangement face: the cover function is constant on it."""
    cover: frozenset       # {j : Δ_j ⊇ this face}
    rep_point: tuple       # an interior point (x, y) — see representative_point()
    area: float
    polygon_wkt: str

    @property
    def bitmask(self) -> int:
        return sum(1 << j for j in self.cover)


def detection_regions(instance_path: str = INSTANCE) -> dict:
    """{j : Δ_j} as shapely geometries, keyed by treasure index."""
    data = json.load(open(instance_path))
    return {int(i): wkt.loads(w) for i, w in data["regions_wkt"].items()}


def cover_of(geom, regions: dict, probe: Point) -> frozenset:
    """The cover set of a face, decided at an interior probe point (works for
    non-convex faces, where the centroid may fall outside)."""
    return frozenset(j for j, Dj in regions.items() if Dj.contains(probe))


def arrangement(regions: dict) -> list:
    """Faces of the planar arrangement of the region boundaries.

    polygonize(unary_union(boundaries)) yields every bounded atomic region the
    boundaries cut the plane into; we keep those that lie inside the detector
    union (the exterior face is dropped) and label each by its cover set, read
    at an interior representative point.  Non-convexity is handled throughout by
    using representative_point() rather than centroid.
    """
    boundaries = unary_union([Dj.boundary for Dj in regions.values()])
    union = unary_union(list(regions.values()))
    faces = []
    for cell in polygonize(boundaries):
        probe = cell.representative_point()
        if not union.contains(probe):           # exterior / holes — not sensing area
            continue
        cover = cover_of(cell, regions, probe)
        if not cover:                            # numerical sliver outside every Δ_j
            continue
        faces.append(Face(cover, (probe.x, probe.y), cell.area, cell.wkt))
    return faces


def persist(faces: list, path: str = FACES) -> None:
    json.dump(
        {"faces": [
            {"cover": sorted(f.cover), "rep_point": list(f.rep_point),
             "area": f.area, "polygon_wkt": f.polygon_wkt}
            for f in faces]},
        open(path, "w"), indent=2)


def load(path: str = FACES) -> list:
    return [Face(frozenset(d["cover"]), tuple(d["rep_point"]), d["area"], d["polygon_wkt"])
            for d in json.load(open(path))["faces"]]


if __name__ == "__main__":
    regions = detection_regions()
    faces = arrangement(regions)
    persist(faces)
    print(f"regions: {len(regions)}   faces: {len(faces)}   "
          f"persisted -> {os.path.basename(FACES)}")
