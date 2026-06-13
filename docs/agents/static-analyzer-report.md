# Static-analyzer agent — report (verbatim)

> Complete final message, verbatim. Commission in `static-analyzer-commission.md`. Headline
> corrected numbers independently re-derived by the coordinator (single-face collapse 3.55×;
> all-faces floor 4.1 worlds / 6,414 classes; only δ {3,4,16,19} indistinguishable — all matched).

---

Done. Committed on `feat/static-analyzer` (commit `4a2d8a5`), not pushed, only the three deliverables, nothing outside the worktree touched.

## Final report

### Deliverables (committed, not pushed)

- **`analyzer.py`** — `analyze(instance) -> StructuralReport`, operating on an abstract `Instance` (treasures, arrangement faces+covers, teleports, the exactly-K prior). Composed from small, independently-runnable named functions, each docstring-tagged **[DET-DEP]** (re-derived under faces — supersedes the contaminated note) or **[DET-IND]** (a prior/geometry property — ports verbatim). Functions: `cocoverage_edges` / `clusters` / `cluster_geography`; `face_collapse` / `best_single_face_collapse` / `cluster_resolution_chain` / `sweep_collapse`; `occupancy_factorization` / `n_occupancy_partitions`; `reachable_local_beliefs`; `face_signature` / `indistinguishability_classes` / `full_sense_floor`; `decomposition_assessment`. Runs on the real instance in ~0.1s.
- **`synthetic.py`** — `generate(n_treasures, K, n_regions, overlap_density, nonconvex_frac, n_delta, seed)`: random treasures + overlapping regions (region j ↔ treasure j, the rest δ-treasures), including non-convex wedge-bitten blobs, pushed through the **same `arrangement.py`** to yield an `Instance` of the shape `analyze` consumes. Face covers emerge from geometry, not assignment.
- **`docs/design/static-analysis-faces.md`** — regenerated real-instance analysis with a per-section DET-DEP/DET-IND tagging, two synthetic-instance sanity runs, and an explicit supersession ledger vs `static-shortcuts.md`.

### Corrected real-instance numbers (vs the contaminated ones)

- **Single-read leverage:** best single-face negative collapse is **3.55×** (the unique k=4 face `{8,9,11,12}`), routine k=3 is 2.51×. The old **5.16×** is unrealizable — no face covers a whole cluster. Singleton faces (21 of 44) give a clean 4.00× on a *positive* read — single-treasure information the old model couldn't express.
- **Indistinguishability:** under faces, **only the δ-treasures {3,4,16,19}** stay indistinguishable; every region-covered treasure has a distinguishing face. The old "four indistinguishable pairs {8,9}/{11,12}/{13,14}/{17,18}" is wrong — all are face-separated.
- **Sensing floor:** reading all faces leaves **≈4.1 expected worlds** (prune ≈3,783×), not the old **120.7** (128×).
- **Cluster resolution needs a chain, not one read:** NW takes ~10 face-reads (greedy upper bound), SE+mid ~10, N 5, S 3 — reinforcing the project's "VoI gated behind depth" finding.
- **Ports verbatim (DET-IND):** the occupancy factorization `#worlds = ∏ C(size,k)` (verified exact, generalized to any partition); the clairvoyant +70% ceiling (0.0855→0.1454); the teleport 3-region partition / τ_4-dominated routing. Macro occupancy partitions = **320** (8-cell cluster partition, SE+mid fused) — the old 613 differs only because the partition is now 8 cells, same computation.
- **Co-coverage edge set equals the 17 overlap pairs exactly**, so the cluster *partition* is unchanged from the old note even though the *definition* is now the honest one.

### Synthetic generator

Produces controlled-geometry instances of the same shape; the two doc demos (seeds chosen for multiple non-trivial clusters) reproduce the real map's structural signature with no hand-tuning: singleton-rich faces ⇒ only δ-treasures indistinguishable; occupancy product exact; per-cluster local beliefs in the hundreds; clusters resolve by chains. An honest emergent observation: random blobs at high overlap density percolate into one giant cluster.

### Honest caveats

- **`reachable_local_beliefs` was the one unbounded piece** — the reachable-belief count grows far faster than `2^size`, and a 10-treasure synthetic cluster hung the BFS. I added a hard size cap (default 2^8) and a 200k reachable-state abort, returning a `skipped`/`truncated` marker rather than hanging (ADR-0002 fail-loud). "Exactly solvable" is now correctly stated as a property of *small* clusters only; large fused clusters must be sub-decomposed. Every real cluster (largest size 6 → 1,448 beliefs) is comfortably under the cap.
- The per-cluster reachable-belief counts are **larger** than the old §4 (NW 745 vs 332; SE+mid 1,448 vs 1,320) because the corrected faces expose more distinct beliefs — still small enough for exact value iteration.
- The `cluster_resolution_chain` length is a greedy **upper bound**, not a proven minimum (stated in the docstring and doc); the load-bearing point is the order of magnitude (many reads, not one).
- Harness note: backgrounded multi-seed runs returned empty output files (sandbox isolation discarded both stdout and filesystem writes); I verified every number via short synchronous single-seed runs whose stdout displayed inline.
