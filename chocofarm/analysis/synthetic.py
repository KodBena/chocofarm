#!/usr/bin/env python3
"""
synthetic.py — a controlled-geometry instance generator for the analyzer.

PURPOSE
-------
`analyzer.analyze` operates on the abstract `Instance` (treasures, faces,
teleports, exactly-K prior).  To exercise it on geometry we control — and to
show the method is not a one-off fit to the real map — we generate random
treasures + overlapping detection regions, push them through the SAME
`arrangement.py` that produced the real faces, and wrap the result as an
`Instance`.  The analyzer then runs on it unchanged.

The generator is deliberately small and parameterized:

    n_treasures, K          the exactly-K-of-N prior
    n_regions               how many detection polygons to place
    overlap_density         0..1; higher packs regions closer so more overlap
    nonconvex_frac          fraction of regions made NON-convex (the real map has
                            non-convex Δ_1/Δ_6/Δ_7 multipolygons; a generator that
                            only made convex blobs would not stress the
                            representative_point() handling the arrangement relies on)
    n_delta                 treasures with NO region (sense-isolated, observe==collect)

MODEL FIDELITY: on the real map, region Δ_j is the detector for treasure τ_j, and
a face's cover is {j : Δ_j ∋ p} — the treasures whose OWN regions contain the
point.  So "covered" treasures are exactly those with a region, and co-coverage
is region overlap.  The generator mirrors this: the first `n_regions` treasures
each get a region (region j ↔ treasure j); the remaining treasures are
region-less δ-treasures (sense-isolated, observe==collect).  Face covers are thus
emergent from where the regions overlap, not hand-assigned — exactly as on the
real map.

BOUNDEDNESS: geometry only.  `arrangement()` is the same bounded polygonize call
the real pipeline uses.  No solver, no belief enumeration here.
"""
from __future__ import annotations

import math

import numpy as np
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union

from chocofarm.model import arrangement as A
from chocofarm.analysis.analyzer import Instance


def _convex_blob(cx, cy, r, rng, n_vert=7):
    """A roughly-circular convex-ish polygon (sorted angular vertices)."""
    angs = np.sort(rng.uniform(0, 2 * math.pi, n_vert))
    rad = r * rng.uniform(0.7, 1.0, n_vert)
    pts = [(cx + ra * math.cos(a), cy + ra * math.sin(a)) for a, ra in zip(angs, rad)]
    return Polygon(pts).buffer(0)


def _nonconvex_blob(cx, cy, r, rng):
    """A non-convex region: a blob with a wedge bitten out, or an L/crescent.
    Mirrors the real map's multipolygon/concave Δ regions so the arrangement's
    representative_point() (not centroid) handling is genuinely exercised."""
    base = _convex_blob(cx, cy, r * 1.2, rng, n_vert=9)
    # bite a wedge out with a smaller offset blob
    bx = cx + r * rng.uniform(0.3, 0.7) * math.cos(rng.uniform(0, 2 * math.pi))
    by = cy + r * rng.uniform(0.3, 0.7) * math.sin(rng.uniform(0, 2 * math.pi))
    bite = _convex_blob(bx, by, r * 0.6, rng, n_vert=6)
    out = base.difference(bite).buffer(0)
    # if the difference fragmented or vanished, fall back to convex
    if out.is_empty or out.area < 0.1 * base.area:
        return base
    return out


def generate(n_treasures=14, K=4, n_regions=8, overlap_density=0.5,
             nonconvex_frac=0.3, n_delta=2, span=20.0, seed=0) -> Instance:
    """Build a synthetic Instance of the shape analyze() consumes.

    Returns an Instance whose faces are the genuine planar arrangement of the
    generated regions — so clusters, co-coverage, indistinguishability and
    collapse are all emergent from the geometry, not assigned.

    n_regions is clamped to n_treasures - n_delta (a region needs its own
    treasure).  Region j is centered on treasure j; co-coverage arises where two
    regions overlap, which `overlap_density` tunes via packing + radius."""
    rng = np.random.default_rng(seed)
    n_regions = min(n_regions, n_treasures - n_delta)

    # region centers: closer packing for higher overlap_density
    spread = span * (1.0 - 0.55 * overlap_density)
    centers = rng.uniform(-spread / 2, spread / 2, size=(n_regions, 2))
    base_r = span / (2.2 + 2.0 * (1.0 - overlap_density))   # bigger r -> more overlap

    regions = {}
    treasures = {}
    # region j ↔ treasure j (the real-map semantics: Δ_j detects τ_j)
    for j in range(n_regions):
        cx, cy = centers[j]
        r = base_r * rng.uniform(0.8, 1.3)
        if rng.random() < nonconvex_frac:
            poly = _nonconvex_blob(cx, cy, r, rng)
        else:
            poly = _convex_blob(cx, cy, r, rng)
        regions[j] = poly
        treasures[j] = (float(cx), float(cy))

    # δ-treasures: region-less, placed strictly OUTSIDE every region (sense-isolated)
    union = unary_union(list(regions.values()))
    tid = n_regions
    n_delta_target = n_treasures - n_regions
    placed_d, attempts = 0, 0
    while placed_d < n_delta_target and attempts < 8000:
        attempts += 1
        p = (float(rng.uniform(-span / 2, span / 2)),
             float(rng.uniform(-span / 2, span / 2)))
        if not union.contains(Point(p)):
            treasures[tid] = p
            tid += 1
            placed_d += 1

    N = len(treasures)
    # build the arrangement from the region geometries, just like the real pipeline
    faces = A.arrangement(regions)

    # teleports: a few corners of the span (entry-ish) so cluster_geography works
    teleports = {
        "TP_NW": (-span / 2, span / 2),
        "TP_SE": (span / 2, -span / 2),
        "TP_C": (0.0, 0.0),
    }
    return Instance(treasures=treasures, faces=faces, teleports=teleports, N=N, K=K)


if __name__ == "__main__":
    import sys
    from chocofarm.analysis.analyzer import analyze, _print_report

    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    inst = generate(seed=seed)
    print(f"# synthetic instance seed={seed}: N={inst.N} treasures, "
          f"{len(inst.faces)} faces, δ={sorted(inst.delta)}")
    _print_report(analyze(inst))
