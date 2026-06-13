# Face-model verification — the arrangement detector, self-checked

> Research note (point-in-time). Every number below is computed from the **real instance**
> (`chocobo_instance.json`) via `arrangement.py` under `timeout`, on the frozen geometry
> (shapely 2.1.2). It records the corrected detector model consult-002 specifies and the
> self-checks that confirm the extraction. The companion artifacts are `arrangement.py`
> (face extraction), `facemodel.py` (the sense action), `chocobo_faces.json` (persisted
> faces), and `chocobo_faces.ggb` (the visual round-trip).

## 0. What was wrong, in one line

`cover_mask[i] = {i} ∪ {j : Δ_i overlaps Δ_j}` reads "enter Δ_i" as the *simultaneous*
disjunction over i and all its overlap-neighbours. That disjunction is not realisable at
any single point: the cover you actually read is `{j : Δ_j ∋ p}`, which depends on the
**arrangement face** p sits in. `cover_mask[i]` is the *union over all faces in Δ_i* — the
set revealable *somewhere* in Δ_i — passed off as revealed *everywhere*.

## 1. The model

The cover function `cover(p) = {j : Δ_j ∋ p}` is constant on each atomic face of the planar
arrangement of the region boundaries. Faces, not regions, carry sensing:

        faces = polygonize( unary_union( {∂Δ_j} ) ),  keep those inside ⋃Δ_j
        Face  = (cover, rep_point, area, polygon_wkt)
        sense-action "go to F":  cost = d(loc, F.rep_point),
                                 observation = (world & F.bitmask) != 0  =  ⋁_{j∈F.cover} τ_j

Point and cover are consistent *by construction* — the disjunction read is exactly the cover
of the face the rep point lies in. This sits strictly between the two prior errors:
consult-001's under-approximation (the cover of *one* face) and `cover_mask`'s
over-approximation (the union over *all* faces in a region).

**Non-convexity.** Δ_2, Δ_5, Δ_12, Δ_18 are non-convex (and Δ_1, Δ_7 are multi-part). The
cover of a face is decided at `representative_point()` (interior-guaranteed) rather than the
centroid (which can fall outside a non-convex face); `polygonize`/`intersection` handle
non-convex simple polygons directly. No convexity is assumed anywhere.

## 2. Self-check against consult-002 — all matched

All figures from a single bounded run (`arrangement.py` + the self-verification block).

| check | consult-002 | measured | verdict |
|---|---|---|---|
| total faces | ~44 | **44** | match |
| k=1 area share | 70.3% | **70.3%** (36.060) | match |
| k=2 area share | 24.2% | **24.2%** (12.417) | match |
| k=3 area share | 5.4% | **5.4%** (2.765) | match |
| k=4 area share | 0.1%, area 0.052 | **0.1%**, area **0.0516** | match |
| k≥5 faces | 0 | **0** (max cardinality 4) | match |
| tiling: Σ face area = union area | — | 51.2938 = 51.2938 (Δ 7e-15) | match |
| per-region: ⋃(face covers in Δ_i) = old cover_mask[i] | holds ∀i | **holds for all 16** | match |
| unique k=4 face | exactly {8,9,11,12} | **{8,9,11,12}**, the only one | match |

The per-region refinement equality is the load-bearing one: it proves the face model
**refines** the geometry rather than changing it — every treasure `cover_mask[i]` claimed is
still revealable *somewhere* in Δ_i, just not *simultaneously*. The board is overwhelmingly
singleton-cover (70%); the genuine disjunctive structure is the 24% pairs, 5% triples, and the
lone 0.052-area {8,9,11,12} sliver.

No discrepancy against consult-002 was found.

## 3. The GeoGebra round-trip (`chocobo_faces.ggb`)

The primary visual artifact. `build_faces_ggb.py` copies `/home/bork/chocobo.ggb` and, in its
`geogebra.xml`:

- **removes** the 19 Δ detection polygons — each `Polygon` command, its polygon element, and
  its output edge segments (orphaned segments would invalidate the construction);
- **demotes** the 4 helper commands (`Intersect` H_3/I_3/J_3, `Point` C_5) that consumed
  deleted Δ segments to free points at their cached coords — nothing visible moves, and the
  one annotated point among them (C_5 = `δ_{11,B}`) keeps its caption;
- **adds** the 44 faces — each face's vertices as fresh points, a `Polygon` over them, and a
  polygon element captioned by its cover set (`{8,9}`, `{8,9,11,12}`, …);
- **keeps** the 20 treasure points τ_*, the teleports (W=CSNE, Z=CSCE, τ_4), every named map
  annotation, and the background map image `Bild1` (the embedded jpg is copied byte-for-byte).

It re-zips as `chocobo_faces.ggb`. **To verify:** open it in GeoGebra; the 44 translucent
steel-blue faces overlay the real map, each labelled by its cover set. Confirm the non-convex
regions reproduce, and that the tiny `{8,9,11,12}` sliver sits where Δ_8/Δ_9/Δ_11/Δ_12 mutually
overlap near the CSNE teleport.

**One frame caveat (honest disclosure).** The faces are computed from the *frozen instance*
WKT, which is authoritative. For 15 of the 16 regions the instance geometry is identical to the
original `.ggb` drawing to ~1e-16, so the faces sit exactly on the old region outlines. The
exception is Δ_1: the original drawing (poly7, area 7.658) is a strict superset of the instance's
Δ_1 (area 6.558) — the instance was clipped ~1.1 area units smaller upstream (the parser's
`buffer(0)` repair of a self-touching ring). The old poly7 is removed in the round-trip, so
nothing visibly conflicts; but a maintainer reconciling against memory of the original Δ_1
outline near τ_1 will see the faces stop ~1.1 units short of where poly7 reached. This is a
property of the frozen instance, not of this change.

**Machine validation** (we cannot run GeoGebra): the zip passes `testzip`; all non-XML members
(image, thumbnail, defaults, js) are byte-identical to the source; the XML parses; `Bild1` and
its file reference survive; 20 treasures + both teleports are present; 0 Δ polygons remain and
exactly 44 Face polygons exist; every Face polygon references only defined points; 0 orphaned
segments; every remaining command's inputs resolve; and each ggb face polygon, reconstructed
from its referenced point coords, is geometrically identical to the computed face
(symmetric-difference area < 1e-9 for all 44, including the Q sliver and the four non-convex
regions, which tile their parent region to ~1e-16). The face↔instance frame is identity for 15
of 16 regions; the Δ_1 caveat above is the lone exception, and it is upstream of this change.

## 4. Adopting it in `Environment` (not done here)

The change is left contained and reviewable — solvers and `run.py` are untouched. The note in
`facemodel.py` (`ENV_ADOPTION`) gives the shape: `Environment.__init__` replaces the
`cover_mask` block with `self.faces = arrangement.load()` and `self.senses =
facemodel.sense_actions(...)`; `legal_actions` becomes collects + informative faces +
TERMINATE; `apply` for a sense action travels to `s.rep_point` and filters by `s.observe(world)`.
The action set becomes ~20 collects + |informative faces| sense actions + TERMINATE, replacing
the 16 detector actions. `det_pt` and the cover semantics become the same object (the face), so
the `cover_mask`/`consult-001` inconsistency cannot recur. Singleton-face pruning (70% of area,
exact single-treasure probes) is a policy choice for the solver, not baked into the model.
