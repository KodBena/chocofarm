# Consult 002 — detector mis-specification postmortem (report, verbatim)

> Independent postmortem on the `cover_mask` over-approximation. Complete final message,
> reproduced verbatim. Commission in `consult-002-detector-misspec-commission.md`.

---

The commission itself embeds the false premise: "the early nodes are tightly clustered geographically — e.g. {8,9,10,11,12} near the CSNE teleport — so an early 'nothing' there collapses a large chunk." The maintainer's lead *assumed* a single negative read could rule out the whole cluster, and the agent confirmed it against `cover_mask` (which encodes exactly that assumption) without ever independently checking the geometry. I have the complete propagation chain.

I have all four investigation tasks covered with measured numbers. Here is my report.

---

# Independent Postmortem — the `cover_mask` over-approximation in chocofarm

## Verdict in one line

The maintainer is correct on every geometric point. `cover_mask` does not model a detector at all — it models *the union of every treasure a region could co-reveal somewhere*, collapsed into a single fictitious simultaneous disjunction. The headline "5.16× early-negative collapse ruling out all of {8,9,10,11,12}" is **geometrically unrealizable**: no point on the board reveals more than a 4-way disjunction, the maximal 4-way face is a 0.052-area sliver, and Δ_10 is not reachable from it. The error is the exact over-correction of the bug consult-001 caught, and it went uncaught because the verifying agents measured `cover_mask` against itself and against a premise that was baked into the commission.

---

## (1) Geometric verdict — measured

All numbers below are from the frozen instance via shapely 2.1.2, read-only.

### The maintainer's three claims — all confirmed

| Claim | Measured | Verdict |
|---|---|---|
| `Q = Δ_8∩Δ_9∩Δ_11∩Δ_12` nonempty and tiny | area **0.05162**, = **1.4%** of the smallest of the four regions (Δ_11, area 3.63) | **CONFIRMED** |
| No point covered by all of {8,9,10,11,12} | the 5-way intersection `Q∩Δ_10` has area **0.0** | **CONFIRMED — no k=5 face exists anywhere** |
| Δ_10 not close to Q | min gap `Δ_10.distance(Q)` = **1.50 units**; centroid gap 2.25; `area(Δ_10∩Δ_11)=area(Δ_10∩Δ_12)=0.0` | **CONFIRMED** |

Δ_10 overlaps **only** Δ_8 (0.370) and Δ_9 (0.851). The `overlaps` array confirms this structurally: pairs (10,11) and (10,12) are **absent**. So Δ_10's links into the cluster are entirely through Δ_8/Δ_9, never through Δ_11/Δ_12 — yet `cover_mask[8]={8,9,10,11,12}` and `cover_mask[11]={8,9,11,12}` both assert disjunctions that are unreachable as single reads.

The unique max-cardinality nonempty intersection within the NW cluster is `(8,9,11,12)` at k=4, area 0.05162. There is nothing larger.

### The true arrangement (16 polygons → 44 atomic faces)

Globally, area-weighted over all detector area (union area 51.29):

| face cover-set cardinality | area | share of covered area |
|---|---|---|
| **1 (singleton)** | 36.06 | **70.3%** |
| 2 | 12.42 | 24.2% |
| 3 | 2.77 | 5.4% |
| **4 (the {8,9,11,12} sliver, the only one)** | 0.052 | **0.1%** |
| 5+ | 0 | 0% |

So the board is overwhelmingly singleton-cover. The disjunctive structure the whole project is "about" occupies 5.5% of the sensing area; the multi-cover beyond pairs is a rounding error.

### Per-detector contrast: `cover_mask[i]` vs the faces actually inside Δ_i

For **every** detector, the union of the cover sets of the faces inside Δ_i exactly equals `cover_mask[i]`. **That is the smoking gun: `cover_mask[i]` IS the union-over-faces** — the set of treasures revealable *somewhere* in Δ_i — being passed off as the set revealed *simultaneously everywhere* in Δ_i. The worst offenders:

| det | cover_mask claims | true max face inside Δ_i | singleton-cover fraction of Δ_i |
|---|---|---|---|
| **D_8** | {8,9,10,11,12} (size 5) | **{8,9,11,12}** k=4, area 0.052 (0.9% of Δ_8) | 26.2% is bare {8} |
| **D_9** | {8,9,10,11,12} (size 5) | **{8,9,11,12}** k=4, area 0.052 | 32.6% |
| D_11 | {8,9,11,12} (size 4) | {8,9,11,12} k=4, area 0.052 | 20.3% |
| D_12 | {8,9,11,12} (size 4) | {8,9,11,12} k=4, area 0.052 | 27.2% |
| D_7 | {5,6,7} (size 3) | {6,7} k=2 — **{5,6,7} never co-occur** | 67.0% |
| D_15 | {1,13,14,15} (size 4) | {13,14,15} k=3 — **1 never co-occurs with 13/14** | 58.4% |
| D_0 | {0,1,2} (size 3) | {0,2} k=2 — **{0,1,2} never co-occur** | 56.5% |

D_8 is the canonical failure: cover_mask claims a 5-disjunction; the largest face inside Δ_8 is a 4-disjunction occupying 0.9% of the region, and over a quarter of Δ_8 senses nothing but treasure 8 alone.

### What the headline number actually buys

The "5.16× early-negative collapse" requires a single negative read over all five of {8,9,10,11,12} (#neg = 3003, which I reproduce exactly: 15504/3003 = 5.16×). The realizable truth:

| read | rules out | #surviving | collapse |
|---|---|---|---|
| cover_mask claim {8,9,10,11,12} | all 5 | 3003 | **5.16×** (fictitious) |
| best single face inside Δ_8 = {8,9,11,12} (0.9% of region) | 4 | 4368 | 3.55× |
| **area-weighted typical single read inside Δ_8** | varies | — | **1.87×** |

Standing at the *typical* point of Δ_8, a negative read collapses the belief **1.87×**, not 5.16×. The headline overstates the single-read leverage by **~2.8×**. And ruling out all five genuinely requires **multiple** reads at geometrically separated faces: treasure 10 is revealable only on faces {(8,9,10), (9,10), (10,)}, which are disjoint from the {8,9,11,12} sliver, so no single position can rule out 10-and-{11,12} together.

### A second, independent inconsistency (worse than the maintainer flagged)

`env.det_pt[i] = regions[i].representative_point()` is the single point the agent travels to. For **12 of 16 detectors** that point lands on a face that senses *strictly fewer* treasures than `cover_mask[i]` applies. The D_8 action travels to (−6.98, 9.42) — a point that geometrically senses only **{8,9}** — yet `filter_detector` grants it the **{8,9,10,11,12}** reading. The model is internally contradictory: a 2-cover *position* is given a 5-cover *semantics*. (The "CSNE is 2.6 from the NW pack" cheapness argument is also to these rep-points, which sense {8,9}, not the cluster.)

### Second-order issues — moot in this instance

The δ-points (`tau_16-in-Δ_1`) worry the stage script flagged does **not** materialize: `delta_pts` were never persisted to the instance JSON, and all four δ-treasures {3,4,16,19} are geometrically **outside every region**. So `filter_detector` correctly never co-reveals them; that is the one thing the current model gets right. The τ_4/τ_16 boundary is a non-issue here.

---

## (2) Process root-cause — how it arose and why it went uncaught

**The deferred computation.** `env.py`'s own comment names it: cover sets are "an over-approximation; per-arrangement-face reification is a later refinement." The arrangement-face computation was deferred and never done. `chocobo_geometry.py` computes only **pairwise area-overlaps** (lines 109–116) and persists the 17-pair `overlaps` array. `env.py` builds `cover_mask` as the union of pairwise neighbours: `nbr[i] = {i} ∪ {j : (i,j) ∈ overlaps}`. Union of *pairwise* relations silently became a claim of *joint* coverage. Pairwise overlap is an existential ("∃ a point in Δ_i∩Δ_j"); the disjunctive read needs the conjunction ("∃ a point in Δ_i∩Δ_j∩Δ_k∩…"). These are different geometric facts, and the gap between them is precisely the arrangement.

**The over-correction.** This is the exact mirror of consult-001's finding. Consult-001 caught the *under*-approximation: a single `representative_point()` cover that saw only the rep-point's face, dropping ~11 of 17 overlaps and degenerating 8 detectors to singletons. Both errors are the **same missing computation** — the cover set of a *chosen face* — approached from opposite ends. The under-approximation chose one face (the rep-point's) and took its true small cover. The "fix" abandoned the face concept entirely and unioned all of them. The correct answer lies strictly between and is *position-dependent*; neither extreme is it.

**Why neither the static-opt agent nor the coordinator caught it.** Three compounding reasons:

1. **The false premise was in the commission.** The static-opt commission states the lead as fact: "the early nodes are tightly clustered … so an early 'nothing' there collapses a large chunk." The agent was asked to *quantify* a premise, not to falsify it. It filtered the 15,504-world array against `cover_mask[8]` and got 3003/5.16× — correct *as a count*, false *as a model*. The filter validated the mask against the mask.

2. **Self-referential verification.** The static-opt report's reassurance — "the consult's flagged rep-point bug is fixed in this env … the loaded cover sets are the disjunctive unions, not singletons" — checks only that the masks are no longer singletons. It never re-ran shapely to ask whether a *single point* realizes those unions. "Not the old bug" was mistaken for "correct."

3. **The AZ agent amplified it into a clean bill of health.** Its F2: "The detector model is genuinely disjunctive … sizes 2–5, mean 3.12, zero singletons … A negative read on D_8 rules out all five of {8,9,10,11,12} at once." Every clause is the over-approximation restated as vindication. "Zero singletons, mean 3.12" is the *signature of the over-approximation*, not its refutation — the true area-weighted picture is 70% singleton.

The structural cause: **all three agents measured cover statistics by reading `cover_mask`, never by re-deriving from the WKT.** consult-001 explicitly couldn't run shapely and flagged its E0 ("confirm with real `representative_point()`") as the one thing to verify first. E0 was never run; the env was patched to the union, and subsequent agents read the patched mask as ground truth.

---

## (3) Contamination scope — suspect vs surviving

### SUSPECT (depend on `cover_mask`; must be re-derived against the true face model)

- **The entire static-opt headline and §2 collapse table.** The "5.16× free at entry" (true single-read leverage ≈1.87× area-weighted, 3.55× best-case); the §2.1 single-observation prune factors for D_8/D_9 and the size-4/3 rows wherever cover_mask over-claims (D_0, D_7, D_15 especially); the §2.2 multi-step chain numbers.
- **§1.4 indistinguishability floor / E[120.7] worlds / 319 outcome classes.** These partition worlds by the 16-bit cover_mask outcome vector. Under the true model each detector is a *menu of faces*, so the achievable partition differs (richer). The qualitative {8,9}/{11,12} symmetry partly survives; the quantified floor does not.
- **§3 cluster decomposition keystone.** The occupancy factorization `#worlds = ∏C(size,k)` is a property of the *prior* and is **detector-independent — survives.** But the claim it is *operationally* reachable because "a detector covers the NW cluster" is false; the two-level macro/micro hierarchy rests on cluster-level reads that don't exist as single actions.
- **The AZ design's F2/F3/F6** and any feature premised on cover_mask (`p_pos[i]`, the per-detector open-clause feature). "mean 3.12, zero singletons" is wrong; the real detector is multi-action (one per face); the 37-slot action space under-counts the true action set.
- **NMCS and ISMCTS rates.** Both call `env.filter_detector`/`cover_mask`. Their policies and measured rates are against a detector that reveals more than any real detector could. The qualitative conclusions (determinization optimism, deeper-is-worse) are method properties and likely survive; the numeric rates and "% of +70% clawed" must be re-measured, and would likely *drop* (the true sensor is weaker).

### SURVIVING (detector-independent)

- **The clairvoyant +70% ceiling (0.0855 → 0.1454).** The clairvoyant is handed the true present-set for free — it never calls a detector. Structurally immune to the cover_mask error; bounds VoI from above regardless of sensor fidelity.
- **The static floor 0.0855, greedy 0.0806, the C(20,5)=15,504 prior, the multivariate-hypergeometric occupancy factorization** (a prior property), and the teleport geometry / τ_4-dominated routing (pure coordinates).
- **The qualitative diagnosis** that VoI is gated behind depth and that determinized search is optimistically biased — method properties, not sensor properties. (Under the true weaker sensor, "depth" is *more* demanding — you need a face read per treasure to corner it, not one cluster read.)

---

## (4) The correct model and remedy

### The arrangement-face detector model

A **detector action** is not "enter region Δ_i." It is **"go to a representative point of arrangement face F,"** observation = the binary disjunction over **F's cover set** = `{j : Δ_j ⊇ F}`. The instance has **44 atomic faces**; the action set is the distinct (face, cover-set) pairs — the agent picks *where in the overlap structure to stand*, and that choice determines the disjunction tested. This unifies the two prior errors: consult-001's rep-point cover is "the cover of *one particular* face"; the current cover_mask is "the union over *all* faces in Δ_i." The correct model is "the cover of *the face you choose*" — a position-dependent menu, strictly between.

### What implementing it changes

1. **Action set widens** from 16 detector actions to one per useful arrangement face (≤44, fewer after pruning singleton/dominated). The fixed 37-slot AZ action space is wrong; it should be ~(20 collects + N_faces sense-faces + TERMINATE).
2. **`cover_mask: {det → bitmask}` becomes `face_cover: {face → bitmask}`**, each face carrying a representative point (travel cost) and its cover bitmask (`filter_detector`). `det_pt` and the cover semantics become *consistent by construction*.
3. **`filter_detector` unchanged in form** (`(bw & face_mask) != 0`) but keyed by face. `legal_actions` enumerates informative faces.
4. **Pruning:** singleton faces (70% of area) are exact single-treasure probes (a "is τ_j present" query). The handful of multi-cover faces (k=2: 24%, k=3: ~5%, the lone k=4 sliver) are the only sources of genuine disjunctive VoI.

### Build-order remedy (read-only verification first, per consult-001's unrun E0)

1. **Run E0 now** (this report is most of it): regenerate `face_cover` from the WKT via `polygonize(unary_union(boundaries))`, persist it alongside (not replacing) `overlaps`. The deferred reification.
2. **Replace `cover_mask`** with the face-keyed model; make `det_pt` the face representative point so semantics and geometry agree.
3. **Re-measure** static-opt §2 collapse factors, the indistinguishability floor, and the NMCS/ISMCTS/greedy rates. The clairvoyant ceiling needs no re-run.
4. **Re-audit the headline shortcuts:** "5.16× free at entry" → "≈1.87× typical / 3.55× best-case single read; ruling out the full NW cluster needs a multi-face chain" — which *reinforces* the project's central finding that VoI is gated behind depth, since even the flagship cluster cannot be resolved in one read.

### A note on direction of error

This over-approximation makes the simulated detectors **more informative than physically possible**. Every search method was measured against a sensor that hands out information no real sensor could. So the true sensing problem is *harder*, the true VoI-per-read is *lower*, and the gap between the methods and the clairvoyant ceiling is, if anything, *understated* by the contaminated runs. The clairvoyant +70% stands; the path to it through real detectors is narrower than the static-opt note claims.

---

**Computations run (read-only, bounded, shapely + 15,504-world filtering):** overlaps→cover_mask reconstruction (Q area, Δ_10 relations, max-cardinality NW intersection); the 44-face arrangement via `polygonize` (per-detector cover-cardinality area distribution; union-of-realized-faces = cover_mask); negative-read collapse (5.16× claim vs 3.55× best face vs 1.87× area-weighted); rep-point face vs cover_mask mismatch on 12/16 detectors; δ-treasures outside all regions. Nothing in the worktree was modified.
