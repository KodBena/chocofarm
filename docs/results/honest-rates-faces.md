# Honest re-measurement on the arrangement-face detector model (point-in-time)

The Monte-Carlo solvers re-measured against the **corrected detector model**: the
planar **arrangement faces** (`arrangement.py` + `chocobo_faces.json`,
`docs/consults/consult-002-detector-misspec-report.md` §(4)), wired into `env.py`. The old model keyed sensing to regions with
`cover_mask[i] = {i} ∪ overlap-neighbours` — an over-approximation that read the
union over *every* face in Δ_i (a k=5 semantics) at one face's representative point
(a k≤2 position). The honest model makes the **face** the sense action: standing at
face F's rep-point reveals the disjunction over exactly F's cover, with cover and
position consistent by construction.

UNIT values throughout. static floor **0.0855**, clairvoyant ceiling **0.1454**
(**+70%** headroom) — both detector-independent and **re-confirmed unmoved** on the
honest model (the static route never senses; the clairvoyant is handed the true
present-set for free).

## What changed in the model

- 16 region-detectors → **44 arrangement faces** as the sense-action set
  (face ids 0..43). Cover-size histogram: **21 singletons, 18 pairs, 4 triples,
  1 quad** (the lone {8,9,11,12} sliver, area 0.052). **70% of sensing area is
  singleton-cover** — most faces are exact single-treasure probes, not disjunctions.
- `self.detectors`, `self.cover_mask[id]`, `self.det_pt[id]`, `coord[("d", id)]`
  re-keyed from regions to faces. The action shape `('d', id)` and the methods
  `filter_detector` / `legal_actions` / `apply` / `d` / `exit_cost` are **unchanged
  in form** — only their data are faces. The policies in `policies.py`, `nmcs.py`,
  `ismcts.py` run with **no changes** (confirmed by smoke test).
- The root legal-action set widened from ~36 (20 collects + 16 detectors) to **64**
  (20 collects + 44 sense-faces), before TERMINATE.

## Honest rates: new (faces) vs old (contaminated regions)

| policy | NEW rate (faces) | %ceil | VoI clawed | OLD rate (regions) | OLD VoI | N (new) |
|---|---|---|---|---|---|---|
| static floor | **0.0855** | 59% | — | 0.0855 | — | — |
| greedy | 0.0810 | 56% | −8% | 0.0810 | −8% | 3000 |
| certainty-equiv | 0.0751 | 52% | −17% | 0.0751 | −17% | 3000 |
| rollout(greedy) | 0.0898 | 62% | **+7%** | 0.0892 | +6% | 150 |
| sparse(d2,leaf=CE) | *not feasible* | — | — | *timed out* | — | — |
| nmcs(level=1) | 0.0781 | 54% | −12% | 0.0780 | −13% | 120 |
| nmcs(level=2) | 0.0704 | 48% | −25% | 0.0688 | −28% | 40 |
| ISMCTS(it=200) | 0.0621 | 43% | −39% | 0.0680 (it=150) | −29% | 40 |
| ISMCTS(it=400) | 0.0747 | 51% | −18% | 0.0763 | −15% | 20 |
| **clairvoyant ceiling** | **0.1454** | 100% | +100% | 0.1454 | +100% | — |

(Old region-model numbers from `docs/results/pluggable-policies-2026-06-13.md`,
`nmcs-result.md`, `ismcts-result.md`. `%ceil` = rate / 0.1454. `VoI clawed` =
(rate − 0.0855) / (0.1454 − 0.0855), the fraction of the +70% gap recovered.)

## Findings (honest)

- **The rates did not improve, and that is the expected and correct result.**
  consult-002 predicted the honest detectors are *weaker* — 70% of sensing area is
  singleton-cover and no single face resolves a cluster; cornering a treasure needs
  a multi-face read-chain. So the headroom to the +70% ceiling is at least as large
  as before, and every policy still sits well below it. We do **not** spin this as
  progress: it is the same "VoI is gated behind depth" verdict, now measured against
  a sensor no longer handing out impossible information.

- **The detector-blind policies are identical old-vs-new, as they must be.**
  `greedy` (0.0810) and `certainty-equiv` (0.0751) never call a detector — they
  respond to belief only through collect-reveals. Their rates being *bit-for-bit
  unchanged* is the cleanest invariant check that the re-keying touched only the
  sensing data and nothing detector-independent (treasures, teleports, the
  exactly-5-of-20 prior, travel, the floor and ceiling).

- **The detector-using policies barely move, and stay below floor where they were
  below floor.** rollout(greedy) holds its marginal +7% (was +6%, within MC noise at
  N=150 — its base is detector-blind greedy, so its one-step lookahead over the now-
  honest weaker faces gives essentially the same small gain). NMCS L1/L2 reproduce
  their sub-floor rates and the "level 2 worse than level 1" (determinization-
  optimism / winner's-curse) signature: L2 = 0.0704 < L1 = 0.0781, both below
  static. The honest faces, being weaker per read, give these searches *less* to
  exploit, consistent with a slightly lower (not higher) clawback.

- **ISMCTS tightens with iterations but starts lower.** it=200 → it=400 climbs
  0.0621 → 0.0747 (−39% → −18%), the same monotone "more iterations route tighter"
  direction the old run showed (0.0680 → 0.0763). But it=200's honest rate (0.0621,
  −39%) is *below* the old it=150 (0.0680, −29%), and it=400 (0.0747) lands just
  under the old it=400 (0.0763) — the weaker faces give the search less leverage per
  determinized read, so at a fixed iteration budget it claws back slightly less.
  Neither budget clears the floor.

- **The qualitative diagnosis survives the model fix.** Determinization optimism
  (NMCS), the deeper-is-worse curse, and the bottleneck being deep contingent
  sensing chains are all *method* properties, not sensor artifacts — and under the
  honest weaker sensor "depth" is *more* demanding (a face read per treasure to
  corner it, not one cluster read), exactly as consult-002 §(4) anticipated.

## Caveats (budgets, action-set size, slowdowns)

- **Small N, deliberately.** The widened 64-action root makes every search slower
  than the 16-detector runs. Final-eval run counts: greedy/CE 3000 (cheap, detector-
  blind), rollout 150, NMCS L1 120, NMCS L2 40, ISMCTS it=200 40 and it=400 20. The
  ratio estimator on R∈{0..5}/episode has a standard error of several percent at
  N=20–40, so the NMCS-L2 and ISMCTS deltas are **trends, not tightly-resolved
  numbers**. The monotone directions (NMCS below floor, L2 below L1; ISMCTS
  tightening with more iterations) are the trustworthy part; the exact rates are not.

- **ISMCTS it=400 was measured in a separate, tighter-budget run.** Running
  it=200 and it=400 back-to-back through `eval_faces.py ismcts` exceeded the 590 s
  wall-clock bound — it=200 alone took 358 s on the 64-action root (~9 s/episode at
  200 determinized tree-walks per decision), so it=400 was cut by the timeout. It was
  then re-run alone at final_runs=20 (323 s); that is the number tabulated. The
  `eval_faces.py ismcts` plan still lists it=400 at final_runs=30 — bump the budget or
  run it solo when more wall-clock is available.

- **sparse(d2,leaf=CE) is not feasible to measure at a bounded budget on the honest
  model.** A *single* depth-2 episode at width 2 takes **67.6 s** on the 64-action
  set (depth-2 expectimax branches ≈64 × width × 64 base-playouts per decision,
  ~10 decisions/episode). A Dinkelbach measurement (warm + final runs) would run well
  over an hour, outside bounded-safety. The old region model already recorded sparse
  as "timed out at budget" on 16 detectors; on 44 faces it is ~4× worse (branching
  squared in the action count). It is recorded here as *infeasible at budget* rather
  than measured. The smoke test confirms it still *runs* — one `decide` returns a
  legal action — it is only the full-rate measurement that is out of budget.

- **All runs were bounded** under `timeout` with no parallel solver processes (a
  sibling worktree's `analyzer.py`/`synthetic.py` work on another branch was left
  untouched). NMCS L2 used ~301 s of its 590 s budget for 40 final runs; ISMCTS
  it=200/it=400 likewise sit near their budgets — bumping `iterations` and
  `final_runs` would tighten the confidence intervals but the direction would not
  change.

## Provenance

- Model change: `env.py` (detector block re-keyed to `arrangement.load()`; action
  shape and methods unchanged in form).
- Harness: `eval_faces.py` (reuses `run.py`'s `realizable_static` /
  `clairvoyant_rate` references and `env.dinkelbach_rate`; runnable in groups
  `refs` / `shallow` / `nmcs` / `ismcts` to bound wall-time).
- Old (contaminated) numbers: `docs/results/pluggable-policies-2026-06-13.md`,
  `docs/results/nmcs-result.md`, `docs/results/ismcts-result.md`,
  `docs/results/voi-ceiling-2026-06-13.md`.
- Predicted direction of change: `docs/consults/consult-002-detector-misspec-report.md`
  §(3)–§(4) ("the true sensor is weaker… the gap to the ceiling is, if anything,
  understated by the contaminated runs").
