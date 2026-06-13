# Consult 001 — prompt (verbatim)

> The exact commission sent to the independent consult agent. Its full report is in
> `consult-001-report.md`. Recorded verbatim per the verbatim-record discipline.

---

You are an INDEPENDENT consult — an adversarial reviewer of someone else's analysis. Your
job is to stress-test a contention, not to agree with it. The author has written their
position precisely (`docs/STATUS.md`) so you can attack it on the merits.

**HARD CONSTRAINT (read first):** This is an ANALYTIC consultation. Do **NOT** run the
solvers or any heavy/looping/parallel computation. A previous agent exhausted the machine's
RAM by running the 15,504-world solver in parallel — do not repeat that. You may read files
and reason; trivial read-only inspection (`wc`, reading a function, `head`) is fine. If a
claim would need computational verification, **recommend a bounded experiment** rather than
running it.

**What to read** (a frozen worktree snapshot at `/home/bork/w/vdc/chocobo-consult`):
- `docs/STATUS.md` — the author's full contention and the questions they want judged.
- `chocobo_stage2_approx.py` — instance loading, exact online belief (numpy mask over the
  15,504 worlds), greedy base policy, one-step rollout, Dinkelbach rate loop, static NN
  baseline, MC evaluation.
- `chocobo_baseline_sparse.py` — sparse-sampling expectimax anchor (and its root-value readout).
- `chocobo_geometry.py` — the `.ggb` parser (detection regions, overlaps, δ treasures, teleports).
- `chocobo_instance.json` — the parsed 20-treasure instance.
- `chocobo_stage1.py` — the original exact solver (small instances), for reference on the model.

**The model in one breath:** adaptive stochastic orienteering / belief-MDP; 20 treasures,
exactly 5 present per run uniform-without-replacement (15,504 worlds, i.i.d. per run); 16
overlapping detection regions giving binary *disjunctive* observations; 4 δ treasures
(observe==collect); objective = long-run treasures/time (renewal-reward) via Dinkelbach;
Euclidean travel in uncalibrated map units; 3 teleports.

**Judge these (the author's contention, condensed — full text in STATUS.md):**
1. "Adaptivity genuinely pays" (proof-by-construction via a human contingent policy).
2. "My numbers don't show it because of two artifacts: (a) unit values mute the margin;
   (b) the max-of-sample-means root value is maximization-biased — use the induced policy's
   measured rate instead."
3. "Next step: unbiased policy-rate Monte-Carlo + heterogeneous gil values; sparse sampling
   stays the convergent anchor."
4. "τ_4 teleport is likely dominated."

**Specifically attack:**
- Is the **unit-value** explanation actually the cause, or is there a **modeling or
  implementation flaw** that better explains why adaptive underperforms static? Look hard
  at: the **greedy base** (it ignores detectors — is rolling out over a detector-blind base
  fair/sufficient?); the **detector model** (a region's *representative point* + which
  regions contain it defines the disjunctive cover — does this faithfully capture
  "entering a region," and does it under/over-state overlaps vs. a true arrangement?); the
  **Dinkelbach/λ** handling and whether rates are being compared at consistent λ; the
  **renewal-rate** formulation (single fixed entry = CSNE, best-exit, ~80-equivalent
  teleport in map units — is the cycle definition sound?); the **static baseline**'s
  fairness (is NN-route-best-prefix a strong enough static opponent, or a strawman?).
- Is the **maximization-bias** diagnosis correct and sufficient, or are there *other*
  reasons the sparse root value is unreliable here?
- Is "**unbiased policy-rate MC + heterogeneous values**" the right next move, or is a
  **cluster decomposition** (exact within geographic neighbourhoods, chained as a
  max-ratio cycle), a **proper POMDP solver**, or a **different anchor** better value?
- Where is the author **over-claiming or missing something**? In particular: **could
  adaptivity genuinely fail to pay much even with heterogeneous values**, and what
  experiment would *detect* that rather than assume it?

**Deliverable — return as your FINAL MESSAGE, complete markdown** (it is saved verbatim and
read in full): (1) a one-paragraph verdict on the contention; (2) per-claim critique
(1–4); (3) a ranked list of the most likely real flaws/risks with the evidence (from the
code/docs) behind each; (4) endorse-or-redirect on the proposed next step, concretely;
(5) what's missing and the cheapest experiment that would tell us if adaptivity does *not*
pay. Be concrete, cite specific functions/lines, and reach your own independent verdict.
