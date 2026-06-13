# Env face-adoption + re-measure agent — report (verbatim)

> Complete final message, verbatim. Commission in `env-faces-commission.md`. The honest
> re-measurement on the corrected face model.

---

Done. Clean working tree, commit local on `feat/env-faces`, no upstream tracking (not pushed). Here is the complete report.

---

## Report

### TASK 1 — faces wired into `env.py` (minimal-touch)

Re-keyed the detector abstraction from regions to the 44 planar arrangement faces. The diff is confined to the detector-construction block in `Environment.__init__` (plus the import swap `shapely.wkt` → `arrangement` and one docstring line) — 15 insertions, 11 deletions:

- `self.detectors = list(range(44))` — face ids 0..43 are the sense actions
- `self.cover_mask[k] = faces[k].bitmask`, `self.det_pt[k] = faces[k].rep_point`; `coord[("d", k)]` follows from `det_pt`
- Deleted the old `overlaps`→`nbr`→`cover_mask` union construction

**The `('d', id)` action shape and `filter_detector` / `legal_actions` / `apply` / `d` / `exit_cost` are unchanged in form** — only their data are faces. Treasures, teleports, the exactly-5-of-20 prior, travel, floor and ceiling are untouched. **`policies.py`, `nmcs.py`, `ismcts.py` were not modified.**

**Proof the policies run unchanged (smoke test):** greedy, certainty-equiv, rollout(greedy), sparse(d2), NMCS(level=1), ISMCTS(it=50) each returned a legal action without error. Key-set invariant `cover_mask == det_pt == coord["d"]` holds; face data round-trips into the env. Root legal set widened 36 → **64** (20 collects + 44 sense-faces). Floor **0.0855** and ceiling **0.1454 (+70%)** re-confirmed unmoved (detector-independent).

### TASK 2 / TASK 3 — honest re-measurement (new vs old)

| policy | NEW (faces) | %ceil | VoI clawed | OLD (regions) | OLD VoI |
|---|---|---|---|---|---|
| static floor | 0.0855 | 59% | — | 0.0855 | — |
| greedy | 0.0810 | 56% | −8% | 0.0810 | −8% |
| certainty-equiv | 0.0751 | 52% | −17% | 0.0751 | −17% |
| rollout(greedy) | 0.0898 | 62% | +7% | 0.0892 | +6% |
| sparse(d2,leaf=CE) | *not feasible* | — | — | *timed out* | — |
| nmcs(level=1) | 0.0781 | 54% | −12% | 0.0780 | −13% |
| nmcs(level=2) | 0.0704 | 48% | −25% | 0.0688 | −28% |
| ISMCTS(it=200) | 0.0621 | 43% | −39% | 0.0680 (it=150) | −29% |
| ISMCTS(it=400) | 0.0747 | 51% | −18% | 0.0763 | −15% |
| clairvoyant ceiling | 0.1454 | 100% | +100% | 0.1454 | +100% |

**Honest read:** rates did not improve — the expected and correct result. consult-002 predicted the honest faces are weaker (70% singleton-cover, the lone k=4 face a 0.05-area sliver, no single face resolves a cluster), so every policy stays below the floor where it was below, and the gap to the +70% ceiling is at least as large. I am not spinning the drop: it is the same "VoI is gated behind depth" verdict, now measured against a sensor no longer handing out impossible information. Detector-blind greedy/CE are **bit-for-bit unchanged** (the cleanest invariant check that re-keying touched only sensing data). ISMCTS at a fixed iteration budget claws back slightly *less* than on the contaminated model (it=200: −39% vs old −29%), consistent with weaker per-read leverage.

### Honest caveats
- **Bounded budgets, small N:** greedy/CE 3000 runs, rollout 150, NMCS L1 120 / L2 40, ISMCTS it=200 40 / it=400 20. At N=20–40 the NMCS-L2 and ISMCTS numbers are trends, not tightly-resolved; the monotone directions (NMCS sub-floor, L2<L1, ISMCTS tightening with iterations) are the trustworthy part.
- **sparse(d2) is infeasible at a bounded budget on the 64-face set:** a single depth-2 width-2 episode took **67.6 s** (depth-2 expectimax branches ≈64×width×64 base-playouts per decision). The old region run already timed out; on 44 faces it is ~4× worse. Recorded as infeasible, not measured — it still *runs* (smoke test passes), only the full-rate measurement is out of budget.
- **ISMCTS it=400** was measured in a separate tighter-budget run (final_runs=20, 323 s) because running it=200+it=400 back-to-back blew the 590 s bound.
- All runs were under `timeout` with no parallel solver processes; a sibling worktree's `analyzer.py`/`synthetic.py` work on another branch was left untouched.

### Deliverables (committed `4e83ad8` on `feat/env-faces`, not pushed)
- `env.py` — the face re-keying
- `eval_faces.py` — the comparison harness (runnable in `refs`/`shallow`/`nmcs`/`ismcts` groups to bound wall-time)
- `docs/results/honest-rates-faces.md` — the new-vs-old table, findings, and caveats

Staged by explicit path (no `git add -A`), `Co-Authored-By: Claude Opus 4.8` trailer, not pushed.
