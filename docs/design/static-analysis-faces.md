# Static structural analysis under the face model — regenerated, honest

> Research note (point-in-time). Every number below is **recomputed by `analyzer.py`**
> (`analyze(instance) -> StructuralReport`) over the abstract instance — treasures, the
> 44 arrangement **faces** (`chocobo_faces.json` via `arrangement.py`), teleports, and the
> exactly-5-of-20 prior (`C(20,5)=15,504` worlds). All runs were bounded under `timeout`,
> counting/structure only; no solver, no global reachable-belief enumeration. The only
> array touched is the 15,504-world bitmask array (cheap to filter).
>
> **This note SUPERSEDES `docs/design/static-shortcuts.md`.** That note was computed on the
> broken `cover_mask` over-approximation (consult-002), so its detector-coupled headline
> numbers are wrong. Each section below marks every quantity as **[DET-DEP]** (re-derived
> under faces — supersedes) or **[DET-IND]** (a property of the prior/geometry — ports
> verbatim). Each quantity is one named function in `analyzer.py`; the function name is
> cited so any figure is re-derivable in isolation.

## 0. What changed, in one paragraph

`cover_mask` modelled a detector as "the union of every treasure region Δ_i could co-reveal
*somewhere*", read as a *simultaneous* disjunction. The corrected face model makes the
**position** the action: standing on an atomic arrangement face `F`, you read exactly the
disjunction over `F.cover = {j : Δ_j ⊇ F}`. The board has **44 faces / 34 distinct
cover-sets**, overwhelmingly **singleton** (21 faces are single-treasure probes). The
two largest corrections that fall out: the headline "5.16× single-read collapse" is
**unrealizable** (no single face covers a whole cluster — best single-face negative collapse
is **3.55×**), and the "120.7-world indistinguishability floor / four indistinguishable pairs"
is **wrong** — under faces every region-covered treasure has a distinguishing face, so the
sensing floor collapses to **≈4 worlds**, gated only by the δ-treasures.

---

## 1. Clusters — co-coverage connected components

`cocoverage_edges` / `clusters` / `cluster_geography`.

**[DET-DEP, but ports unchanged on this map]** Treasures are linked iff *some face's cover
contains both* (a single position reads a disjunction mentioning both). The corrected
definition is the honest one (the old "regions overlap" is an existential over the pair).
On the real map the two edge sets **coincide exactly** — all 17 pairs — because every 2-D
pairwise overlap contains at least one face covering both. So the cluster **partition is the
same** as old §1.2; only the definition is now correct (and on synthetic instances the two
can diverge).

| cluster | treasures | nearest teleport (DET-IND geometry) |
|---|---|---|
| SE+mid (fused) | {0, 1, 2, 13, 14, 15} | bridged by faces `{1,15}` (D_1/D_15 region) |
| NW | {8, 9, 10, 11, 12} | CSNE (entry) |
| N | {5, 6, 7} | CSNE |
| S | {17, 18} | CSCE |
| δ singletons | {3}, {4}, {16}, {19} | visit-only (no face) |

The teleport partition (CSNE → {NW, N}; CSCE → {SE+mid, S, δ16, δ19}; τ_4 → {δ3, δ4}; τ_4
dominated) is **[DET-IND]** — pure coordinates — and ports verbatim from old §1.3.

---

## 2. Belief collapse — honest single-face leverage and face-read chains

`face_collapse` / `best_single_face_collapse` / `cluster_resolution_chain`.

**[DET-DEP — SUPERSEDES old §2.1/§2.2.]** The corrected single-read leverage, by cover
cardinality (every #neg matches the hypergeometric `C(20−k,5)` exactly):

| face cover size | example | prune on NEG | prune on POS |
|---|---|---|---|
| **k=1** (singleton, 21 faces) | `{8}` | 1.33× | **4.00×** |
| k=2 (18 faces) | `{8,9}` | 1.81× | 2.24× |
| k=3 (4 faces) | `{8,9,10}` | 2.51× | 1.66× |
| **k=4 (the lone `{8,9,11,12}` sliver)** | `{8,9,11,12}` | **3.55×** | 1.39× |

The **single largest negative collapse achievable by ANY single face is 3.55×** (the unique
k=4 face), not the old 5.16×. The 5.16× required ruling out all of {8,9,10,11,12} in one
read; no face covers all five (Δ_10's only co-covers are {8,9}, geographically disjoint from
the {8,9,11,12} sliver — consult-002). The richest *routinely available* read is k=3 at
2.51×. Note the singleton faces give a clean **4.00× on a POSITIVE** read (collapse to worlds
containing that treasure) — the sharpest *single-treasure* information on the board, which the
old model could not express because it had no singleton-face concept.

**Resolving a whole cluster needs a CHAIN, not one read** (`cluster_resolution_chain`,
greedy by expected surviving worlds `E[|belief|] = Σ nᵢ²/N`; the length is an *upper bound*
on the reads needed — greedy-maximal, not a proven minimum — and the terminal belief equals
reading every distinct cluster face):

| cluster | greedy resolving-chain length | chain prune |
|---|---|---|
| SE+mid {0,1,2,13,14,15} | 10 reads | 18.2× |
| NW {8,9,10,11,12} | 10 reads | 11.7× |
| N {5,6,7} | 5 reads | 4.2× |
| S {17,18} | 3 reads | 2.6× |

This is the corrected analog of the old §2.2 "one-per-cluster sweep, 18.7×". Under faces a
cluster is resolved only by reading most of its faces — geographically separated positions.
The project's central finding ("VoI is gated behind depth") is **reinforced**: the depth is
deeper than the old note implied, because even the flagship NW cluster takes a 10-read chain,
not one negative probe.

---

## 3. Occupancy factorization — the keystone (ports verbatim)

`occupancy_factorization` / `n_occupancy_partitions`.

**[DET-IND — a property of the exactly-K prior; ports verbatim from old §3.1.]** Given the
per-cell occupancy vector `(k_c)`, the worlds factor as independent within-cell uniform
subsets; the sole coupling is `Σ k_c = K`. Verified exactly on the cluster partition:

```
occupancy (k_SEmid, k_NW, k_N, k_S, k3,k4,k16,k19) = (1,1,1,1,1,0,0,0)
  counted worlds = 180  ==  C(6,1)·C(5,1)·C(3,1)·C(2,1)·C(1,1) = 6·5·3·2·1 = 180   ✓
```

The function is **generalized to any partition** (not just clusters). The macro state — the
count of occupancy vectors with `Σ k_c = 5` over the **8-cell** cluster partition — is
**320** (`n_occupancy_partitions`). This **differs from the old 613** purely because the
partition is now 8 cells (SE and mid are one fused cluster), not the old 9-cell split; it is
the same combinatorial computation on the corrected partition.

---

## 4. Exact-solvable sub-problems — per-cluster reachable beliefs

`reachable_local_beliefs`.

**[DET-DEP — SUPERSEDES old §4.]** BFS over each cluster's local belief lattice (states =
frozensets of in-cluster latent subsets; transitions = every in-cluster face read both
polarities + every in-cluster collect both outcomes), counting distinct reachable beliefs =
the exact backward-induction table size:

| cluster | size | latent subsets | in-cluster faces | **reachable local beliefs** |
|---|---|---|---|---|
| SE+mid {0,1,2,13,14,15} | 6 | 64 | 11 | **1,448** |
| NW {8,9,10,11,12} | 5 | 32 | 15 | **745** |
| N {5,6,7} | 3 | 8 | 5 | 34 |
| S {17,18} | 2 | 4 | 3 | 10 |

These are **larger** than the old §4 counts (NW was 332, merged SE+mid was 1,320) because
the corrected cluster exposes *more* faces — singletons and asymmetric covers the cover_mask
model collapsed away — so more distinct beliefs are reachable. They remain small enough for
exact value iteration. **Honest boundedness caveat:** the reachable-belief count grows much
faster than `2^size`; `reachable_local_beliefs` enforces a hard size cap (default 2^8) and a
reachable-state cap (200k), returning a `skipped`/`truncated` marker rather than running
unbounded. "Exactly solvable" is a property of *small* clusters only — a large fused cluster
must be sub-decomposed, not solved flat (see §6).

---

## 5. Indistinguishability under faces — the floor moved

`face_signature` / `indistinguishability_classes` / `full_sense_floor`.

**[DET-DEP — the sharpest SUPERSESSION.]** Two treasures are face-indistinguishable iff every
face covers both or neither. Under the face model:

> indistinguishability classes = **{3,4,16,19}** (the δ-treasures, no faces) — and *every
> other treasure is its own singleton class.*

The old §1.4 claimed {8,9}, {11,12}, {13,14}, {17,18} were indistinguishable pairs. They are
**all separated** under faces: e.g. face `{8}` separates 8 from 9; face `{18}` separates 18
from 17; face `{13}` + `{13,15}` separate 13 from 14. The cover_mask gave the two members of
each pair an identical 16-bit signature; the faces do not.

Consequently the **full-sense floor collapses**: reading every distinct face leaves expected
belief **≈4.1 worlds** (`full_sense_floor`, prune ≈3,783×), not the old **120.7** (128×). The
residual is governed only by the δ-treasures' free placement and by treasure **14** — the one
region-covered treasure with no literal singleton face (it lives only in `{13,14}` /
`{13,14,15}`), so it is pinned by chain difference rather than directly. The old "breaking a
pair requires a collect" claim is **retired** for region-covered treasures: a face read
suffices; only the δ-treasures still require a visit.

---

## 6. Recommended decomposition — honest operational reachability

`decomposition_assessment`.

**[DET-DEP.]** The two-level macro/micro hierarchy from old §6 still stands in *shape*, but
its operational premise must be corrected:

- **The factorization (macro layer) survives unchanged** — it is DET-IND (§3). The macro
  state is the multivariate-hypergeometric over the 8 cells (320 partitions), coupled only by
  the budget integer `Σ k_c = 5`.
- **The "one cheap cluster probe" the old hierarchy leaned on does not exist.** Each cluster's
  occupancy is resolved only by a **face-read chain** (§2: 3–10 reads of geographically
  separated faces), not a single negative probe. So entering a cluster and resolving it is a
  genuine multi-step sub-episode, not a one-shot collapse. This is *why* the shallow search
  methods clawed back ≈0% of the +70%: the VoI is behind chains deeper than the old note
  implied.
- **The micro layer (exact per-cluster belief-MDP) is still microscopic and exactly
  solvable** for every real cluster (≤1,448 beliefs), giving trusted leaves that remove the
  determinization-optimism the prior NMCS/ISMCTS hit — provided the cluster is under the
  tractability cap (the real map's largest is size 6, comfortably under it).

Net honest verdict: per-cluster decomposition is **still the right structure and still
operationally reachable on the real map** — but each cluster is a small *deep* sub-problem
(a face-read chain interleaved with collects), not a one-read collapse. The decomposition's
value is precisely that it pushes that depth *inside* an exactly-solved cluster.

The **clairvoyant +70% ceiling (0.0855 → 0.1454)** is **[DET-IND]** — the clairvoyant never
calls a sensor — and ports verbatim; it bounds VoI from above regardless of sensor fidelity.
The corrected (weaker) sensor means the true path to that ceiling is *narrower* than the
contaminated note claimed, never wider.

---

## 7. Generalization — the same `analyze` on synthetic instances

`synthetic.generate` builds random treasures + overlapping regions (the first `n_regions`
treasures each get a region; the rest are δ-treasures), including **non-convex** regions
(wedge-bitten blobs, matching the real map's concave/multipolygon Δ_1/Δ_6/Δ_7), pushes them
through the **same `arrangement.py`** that produced the real faces, and yields an `Instance`
of the shape `analyze` consumes. Face covers are emergent from the geometry, not assigned.

Two demonstration runs (`generate(n_treasures=14, K=4, n_regions=9, overlap_density=0.12,
nonconvex_frac=0.4, n_delta=2, seed=…)`), chosen to produce *multiple* non-trivial clusters
(random blobs at high density percolate into one giant cluster — itself an honest observation):

**Synthetic A (seed 9):** N=14, 20 faces, worlds=C(14,4)=1001.
- clusters: {1,2,3,7} and {4,6,8}; δ = {9,10,11,12,13}.
- best single-face negative collapse: `{1,2,3}` → 3.03× (k=3, the richest face — same
  cardinality-driven pattern as the real map).
- occupancy factorization: `counted = product = 12` exact (`C(4,1)·C(3,1)·1·1 = 12`).
- indistinguishability under faces: only the δ-class {9,10,11,12,13}; every region-covered
  treasure distinguished — **the real map's qualitative result reproduces**.
- per-cluster reachable beliefs: 124 (size 4) and 34 (size 3) — both exactly solvable.
- resolution chains: 7 reads (8.8×) and 5 reads (5.0×) — clusters need chains, as on the
  real map.

**Synthetic B (seed 5):** N=14, 29 faces, worlds=1001.
- clusters: {1,2,3,5,8} (size 5) and {0,6,7}; δ = {9,10,11,12,13}.
- best single-face neg collapse `{1,2,8}` → 3.03×; occupancy `15 = C(5,1)·C(3,1) = 15` exact.
- reachable beliefs: 891 (size 5) and 34 (size 3); chains 11 reads (15.4×) and 5 reads (5.0×).

Both reproduce the real map's structural signature without any hand-tuning: singleton-rich
faces ⇒ only δ-treasures stay indistinguishable; the occupancy product holds exactly;
per-cluster local beliefs stay in the hundreds; clusters resolve by chains. The method is not
a one-off fit to the real geometry.

---

## 8. What was computed (reproducibility)

All via `/home/bork/w/vdc/venvs/generic/bin/python` under `timeout`, bounded counting/structure
only; only artifact written by the analyzer is none (it returns a `StructuralReport`). Run
`python -m chocofarm.analysis.analyzer` for the real instance, `python -m chocofarm.analysis.synthetic <seed>` for a synthetic one.
Each figure is one named function in `analyzer.py`; the load-bearing re-derivations: every
single-face #neg equals the hypergeometric `C(20−k,5)`; `occupancy_factorization` confirms
`#worlds == ∏ C(size,k)` exactly; `face_signature` shows only the δ-class is multi-element;
`full_sense_floor` lands at ≈4 worlds (vs the contaminated 120.7); `reachable_local_beliefs`
terminates in hundreds-to-low-thousands for every real cluster and refuses (does not hang) on
over-cap synthetic clusters.

### Supersession ledger (vs `static-shortcuts.md`)

| old claim | status | corrected |
|---|---|---|
| "5.16× single-read NW collapse, free at entry" | **WRONG** | best single face = 3.55× (lone k=4); routine k=3 = 2.51× |
| "a detector covers {8,9,10,11,12}" | **WRONG** | no face covers >4 treasures; max cover is `{8,9,11,12}` |
| "{8,9},{11,12},{13,14},{17,18} indistinguishable" | **WRONG** | all separated by faces; only {3,4,16,19} remain |
| "120.7-world / 319-class sensing floor" | **WRONG** | ≈4.1 worlds; full-face read pins every region-covered treasure |
| "one-per-cluster sweep resolves clusters, 5 reads" | **WRONG (under-counts)** | each cluster needs a 3–10 read chain |
| per-cluster reachable beliefs (NW 332, merged 1320) | **superseded** | NW 745, SE+mid 1448 (more faces ⇒ more beliefs) |
| macro occupancy partitions = 613 | **superseded** | 320 (8-cell partition, SE+mid fused) |
| occupancy factorization `∏ C(size,k)` | **ports verbatim (DET-IND)** | unchanged, generalized to any partition |
| clairvoyant +70% ceiling 0.0855 → 0.1454 | **ports verbatim (DET-IND)** | unchanged |
| teleport 3-region partition / τ_4 dominated | **ports verbatim (DET-IND)** | unchanged |
