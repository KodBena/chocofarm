# Static-analyzer agent — commission (verbatim)

> The exact prompt sent to the agent. Its report is in `static-analyzer-report.md`.

---

You are turning a one-off hand-analysis into a reusable, abstract PROGRAMMATIC method: a function `analyze(instance)` that computes the structural decomposition of a chocofarm map, runnable on the real map AND on synthetic ones. Work in your worktree **/home/bork/w/vdc/chocobo-analyzer** (branch `feat/static-analyzer`). Do NOT touch /home/bork/w/omega, the main checkout, or sibling worktrees. Venv: `/home/bork/w/vdc/venvs/generic/bin/python` (numpy, shapely). Keep all runs BOUNDED and under `timeout` — counting/structure only; never run a full solver or enumerate reachable beliefs (filtering the 15,504-world array is fine and cheap).

**READ FIRST (your worktree):**
- `arrangement.py`, `facemodel.py`, `chocobo_faces.json` — the CORRECTED, just-verified detector model (the planar-arrangement faces, each with a `cover` set + `rep_point`; the `SenseAction` semantics). **This is the abstract instance your analyzer operates on. Use it; do not reintroduce `cover_mask`.**
- `docs/agents/consult-002-detector-misspec-report.md` — what was contaminated vs detector-independent, and the correct face model.
- `docs/design/static-shortcuts.md` and `docs/agents/static-opt-report.md` — the ORIGINAL hand-analysis (clusters, belief-collapse, occupancy factorization, decomposition, exact-solvable sub-problems). **It was computed on the broken `cover_mask`, so its detector-coupled conclusions (e.g. "a detector covers {8,9,10,11,12}", "5.16× single-read collapse") are WRONG and must be re-derived under the face model.** Your output supersedes it.
- `env.py` (the prior: exactly-5-of-20 → C(20,5)=15,504 worlds), `chocobo_instance.json`.

**THE PROBLEM:** 20 treasures, exactly 5 present per run (15,504 equiprobable worlds). Sensing is now per-FACE (44 faces): standing in face F reveals the disjunction over `F.cover`. δ-treasures {3,4,16,19} are visit-only. Objective: long-run treasures/time. Belief = surviving-world set. Clairvoyant ceiling +70% (detector-independent, survives). Goal of this analysis: find the structure that makes near-exact solving tractable.

**MAINTAINER REQUIREMENT — ABSTRACTION/AUDITABILITY:** the maintainer (math background) audits by reasoning. Write `analyze` as **composable, math-legible functions over the abstract instance**, not a monolith. Each structural quantity is a small named function with a clear definition. Mark every output as detector-DEPENDENT (re-derived under faces) or detector-INDEPENDENT (a prior property, ports verbatim).

**DELIVERABLES (commit on `feat/static-analyzer`, do NOT push):**

1. **`analyzer.py` — `analyze(instance) -> StructuralReport`**, operating on the abstract instance (treasures, faces+covers, teleports, the exactly-k prior). Compute, under the FACE model:
   - **Clusters** — connected components of the treasure co-coverage hypergraph (treasures linked iff some face's cover contains both), with their geographic/teleport association.
   - **Belief-collapse table** — for each informative face, |surviving worlds| after a positive and a negative read (filter the world array on `F.cover`), as honest prune factors. This is the corrected analog of the bogus "5.16×": report the real single-face leverage, and — since no single face covers a whole cluster — the minimal **face-read chain** to resolve a cluster (and its cost).
   - **Occupancy factorization** (`#worlds = ∏ C(cluster_size, k_c)` over a partition) — detector-INDEPENDENT; port verbatim; generalize to any partition.
   - **Exact-solvable sub-problems** — per-cluster reachable-belief sizing (small enough for exact backward induction?).
   - **Indistinguishability** — treasures sharing identical face signatures (re-defined under faces).
   - **Recommended decomposition** — macro/micro hierarchy, but honestly: is per-cluster decomposition still *operationally reachable* now that a cluster needs a face-read chain (not one read)? Quantify.

2. **`synthetic.py` — a synthetic-instance generator.** Random treasures + overlapping regions (include some NON-convex, since the real map has them), run through `arrangement.py` to produce faces, yielding an `instance` of the same shape `analyze` consumes. So the analyzer can be exercised on controlled geometry. Keep it small and parameterized (n_treasures, k, n_regions, overlap density).

3. **`docs/design/static-analysis-faces.md`** — the honest, regenerated structural analysis of the REAL instance produced by `analyze`, explicitly noting which numbers SUPERSEDE the contaminated `static-shortcuts.md` (and which port unchanged). Include 1–2 synthetic-instance runs as a sanity demonstration that the method generalizes.

Stage by EXPLICIT path (never `git add -A`); commit, message ending `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`. Do NOT push (the orchestrator inspects + merges). RETURN a complete final report: the analyzer's interface, the corrected real-instance numbers (vs the old contaminated ones), what the synthetic generator produces, and honest caveats.
