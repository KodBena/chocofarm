# Env face-adoption + re-measure agent — commission (verbatim)

> The exact prompt sent to the agent. Its report is in `env-faces-report.md`.

---

You are wiring the (already-verified) arrangement-face detector model into the solver `Environment`, then RE-MEASURING the Monte-Carlo solvers on this honest model. Work in your worktree **/home/bork/w/vdc/chocobo-envfaces** (branch `feat/env-faces`). Do NOT touch /home/bork/w/omega, the main checkout, or sibling worktrees (another agent is editing `analyzer.py`/`synthetic.py` on a different branch — you only touch `env.py` + a small eval script). Venv: `/home/bork/w/vdc/venvs/generic/bin/python` (numpy, shapely). Keep ALL runs BOUNDED, under `timeout`, no parallel processes (the action set is bigger now → solvers are slower; use modest budgets).

**CONTEXT.** `env.py` currently models detectors with a BROKEN `cover_mask[i] = {i} ∪ overlap-neighbours` (an over-approximation; see `docs/agents/consult-002-detector-misspec-report.md`). The corrected model is the planar **arrangement faces**: `arrangement.py` + `facemodel.py` + `chocobo_faces.json` (each face = `cover` frozenset, `rep_point`, `area`). `facemodel.py` has an `ENV_ADOPTION` note describing the intended change. The honest sensing primitive: standing in face F reveals the disjunction over `F.cover`.

**TASK 1 — wire faces into `env.py`, MINIMAL-TOUCH (preserve the interface so the existing policies run UNCHANGED).** Re-key the detector abstraction from regions to faces:
- Load faces from `chocobo_faces.json`. Let the sense-actions be the faces (id them e.g. `0..43`).
- `self.detectors` = list of face ids; `self.cover_mask[face_id]` = bitmask of that face's `cover`; `self.det_pt[face_id]` = that face's `rep_point`; `self.coord[("d", face_id)] = rep_point`.
- **Keep the action shape `('d', id)` and the methods `filter_detector`, `legal_actions`, `apply`, `d`, `exit_cost`, etc. unchanged IN FORM** — only their data changes (faces instead of regions). `legal_actions` keeps the same "outcome still uncertain in belief" test, now per face.
- Delete the old `overlaps`→`cover_mask` construction. Treasures (collect actions), teleports, the exactly-5-of-20 prior, travel, the static floor and clairvoyant — all UNCHANGED (they're detector-independent; confirm floor≈0.0855 and ceiling≈0.1454 are unmoved).
- The policies in `policies.py`, `nmcs.py`, `ismcts.py` must run with NO changes (they consume `env.detectors`/`cover_mask`/`det_pt`/`filter_detector`/`legal_actions`/`apply`). Verify by a smoke test: one `decide` from each of greedy, rollout, sparse, NMCS(level=1), ISMCTS(iterations=50) — they must return a legal action without error.

**TASK 2 — re-measure on the honest model (bounded).** Run the `run.py`-style comparison: static floor, clairvoyant ceiling, greedy, certainty-equiv, rollout(greedy), sparse(depth 2), NMCS L1 and L2, ISMCTS at two budgets (e.g. it=200, it=400). Report each policy's Dinkelbach rate, % of the clairvoyant ceiling, and % of the VoI gap clawed back. Use SMALL N (≤150 final eval runs; the bigger action set makes episodes slower) — and a `timeout` on every command. If something is too slow, shrink and say so.

**TASK 3 — honest comparison.** Tabulate the new (honest) rates against the OLD contaminated ones (from `docs/results/*.md` and the agent reports). Note expectation, from consult-002: the real detectors are WEAKER (70% of sensing area is singleton-cover; no single face resolves a cluster — it needs a read-chain), so rates likely DROP and the gap to the +70% ceiling is at least as large. Report truthfully; do not spin a drop as anything else.

**DELIVERABLES (commit on `feat/env-faces`, explicit paths, never `git add -A`, `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`, do NOT push):** the `env.py` change; a small `eval_faces.py` (the comparison); `docs/results/honest-rates-faces.md` (the new-vs-old table + caveats). RETURN a complete report: what you re-keyed (and proof the policies run unchanged), the smoke result, the honest comparison table (new vs old vs floor/ceiling), and honest caveats (budgets, action-set size, any slowdowns).
