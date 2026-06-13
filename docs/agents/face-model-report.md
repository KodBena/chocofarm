# Face-model build agent — report (verbatim)

> The agent's complete final message, reproduced verbatim. Commission in
> `face-model-commission.md`. Independently re-verified by the coordinator (44 faces;
> k4={8,9,11,12} area 0.0516; tiling drift −1.4e-14; per-region ⋃face-covers == old
> cover_mask; ggb valid, 44 face polygons, 0 Δ remaining) and visually verified by the
> maintainer in GeoGebra (Δ₁ truncation accepted as a conservative under-approximation).

---

Done. Clean tree, committed locally on `feat/face-model`, not pushed.

## Final report

Replaced the broken `cover_mask` with the true polygon-arrangement-face detector model and built the GeoGebra verification round-trip. All work is committed on `feat/face-model` (1f7db0f) in `/home/bork/w/vdc/chocobo-facemodel`; nothing pushed; `env.py` and the solvers/run.py left untouched per the containment requirement.

### Extraction approach (`arrangement.py`)
The cover function `cover(p) = {j : Δ_j ∋ p}` is constant on each atomic face of the planar arrangement of the region boundaries, so faces — not regions — carry sensing. `faces = polygonize(unary_union({∂Δ_j}))`, keeping cells whose interior probe lies inside `⋃Δ_j`. Each `Face = (cover, rep_point, area, polygon_wkt)` (frozen dataclass), persisted to `chocobo_faces.json`. Yields 44 faces.

### The 4 non-convex regions (Δ_2, Δ_5, Δ_12, Δ_18)
Cover sets are decided at `representative_point()` (interior-guaranteed) rather than `centroid` (which can fall outside a non-convex face); `polygonize`/`intersection` handle non-convex simple polygons directly. No convexity assumed anywhere. Verified each non-convex region is exactly tiled by its faces (area drift ~1e-16).

### GeoGebra file (`chocobo_faces.ggb`, `build_faces_ggb.py`)
Copies `/home/bork/chocobo.ggb`; in `geogebra.xml` removes the 19 Δ Polygon commands + their polygon elements + edge segments, then appends the 44 faces (vertices as fresh points, a Polygon per face, captioned by its cover set e.g. `{8,9}`, `{8,9,11,12}`). Keeps the 20 treasures, teleports (W/Z/τ_4), every map annotation, and the background image `Bild1` (embedded jpg copied byte-for-byte). The 4 helper commands (`Intersect` H_3/I_3/J_3, `Point` C_5) that consumed deleted Δ segments are demoted to free points at their cached coords (C_5 keeps its `δ_{11,B}` caption). **To open/verify:** open in GeoGebra; 44 translucent steel-blue faces overlay the map, each labelled by its cover set; confirm the tiny `{8,9,11,12}` sliver near the CSNE teleport. Machine validation passed: zip well-formed, non-XML members byte-identical, XML parses, 0 Δ polygons remain / 44 Face polygons, 0 undefined inputs, 0 orphaned segments, and every ggb face polygon is geometrically identical to the computed face (symdiff 0.0 for all 44).

### Abstract model interface (`facemodel.py`)
`SenseAction(face)`: `cost(loc, dist)` = travel to `rep_point`; `filter(bw, positive)` = belief update over `(bw & bitmask) != 0`; `observe(world)`; `informative(bw)`. Point and cover are consistent by construction (the disjunction read is exactly the cover of the face the rep point sits in), so the old single-rep-point + union-cover inconsistency cannot recur. `ENV_ADOPTION` note spells out the `Environment` change: action set becomes ~20 collects + informative faces + TERMINATE.

### Self-check vs consult-002 — all matched (`verify_faces.py`)
- 44 faces; max cover cardinality 4 (no k≥5)
- Area distribution: k1 70.3%, k2 24.2%, k3 5.4%, k4 0.1% (0.0516) — exact match
- Tiling: Σ face area = union area (Δ 7e-15)
- Per-region: `⋃(face covers in Δ_i)` == old `cover_mask[i]` for all 16 (proves refinement, not change)
- Unique k=4 face is exactly `{8,9,11,12}`, area 0.0516

No discrepancy against consult-002.

### One honest caveat (surfaced by an out-of-frame hack audit, now disclosed in the doc)
The faces are computed from the frozen instance WKT, which is authoritative. 15 of 16 regions match the original `.ggb` drawing to ~1e-16, but **Δ_1** differs: the original poly7 (area 7.658) is a strict superset of the instance's Δ_1 (area 6.558) — the instance was clipped ~1.1 area units smaller upstream (parser `buffer(0)` repair). The old poly7 is removed in the round-trip so nothing visibly conflicts, but the faces near τ_1 stop ~1.1 units short of where the original outline reached. This is a property of the frozen instance, not of this change; documented in `docs/design/face-model-verification.md`. The audit verdict was *narrower-but-justified* (not a hack); its other two notes — the instance-specific dangling-command handler and the strict-interior margin on the Q sliver — are residual fragility that would fail loudly (the verify check catches undefined inputs), not defects.

Note: `*.ggb` is gitignored, so `chocobo_faces.ggb` was force-added by explicit path (`git add -f`); no `git add -A` was used.
