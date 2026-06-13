# chocofarm — status & contention (2026-06-13)

A scratch project: optimal gil farming via chocobo treasure digging in FFXIII, treated
formally as **adaptive stochastic orienteering under partial observation**. This file is
my honest statement of where things stand and what I currently contend — written to be
critiqued by an independent consult.

## The problem

- **20 treasures** at known 2D coords. Each run **exactly 5 are present**, uniform without
  replacement → `C(20,5) = 15,504` equiprobable latent worlds; **i.i.d. re-roll each run**
  (this is what makes it non-trivial; if treasures were fixed it would just be a TSP).
- **16 advance-detection regions** (polygons). Entering one yields a **binary disjunctive**
  observation: "≥1 of the treasures whose regions cover this point present?". Regions
  **overlap** (17 pairs, several near-containment), so a positive reveal in an overlap is
  `τ_i ∨ τ_j` (doesn't say which); a negative reveal is `¬τ_i ∧ ¬τ_j` (rules all out).
- **4 δ-singularity treasures** (3, 4, 16, 19): no advance region → observe == collect.
- **Belief** = the set of latent worlds still consistent with all observations. Sound &
  complete by filtering; cheap to maintain online; **shrinks** as you observe.
- **Objective**: long-run **rate = treasures / time** (renewal-reward), solved via
  **Dinkelbach**: for rate `λ`, maximise `E[Σ value − λ·Σ time]`; the `λ*` where that
  optimum is 0 is the rate. Under the `λ`-penalty, elapsed time leaves the planning state.
- **Movement**: Euclidean travel in **map-distance units** (APPROX — real terrain is
  asymmetric and uncalibrated; teleport overhead is a stand-in). **3 teleports**: CSNE,
  CSCE, τ_4.

## Settled

- **Exact optimal is infeasible.** Backward induction over `(location, belief, collected)`
  enumerates the reachable belief space → memory grows unbounded; empirically even small
  `n` did not finish. (An earlier "firewall" that tried to *measure* this ran the naive
  solver in parallel and exhausted RAM — an operational mishap, not a different verdict.)

## Approximations built (all maintain exact belief online; flat memory)

- `chocobo_stage2_approx.py`: **greedy base** (chase best expected λ-value treasure;
  ignores detectors) and **one-step rollout** (policy improvement over greedy).
- `chocobo_baseline_sparse.py`: **sparse-sampling expectimax** (Kearns–Mansour–Ng) — the
  dumb, provably-convergent anchor (more width/depth → nearer optimal).

## Results so far — under UNIT treasure values

- static NN route (best prefix): **0.0855**
- greedy: 0.0810 · rollout (own Dinkelbach fixed point): 0.0822 → **adaptive does NOT beat static**
- sparse-sampling **root value** at `λ0 = static rate`: `d1/C8 → +0.21`, `d1/C32 → −0.03`,
  `d2/C6 → +0.38`. **More samples gave a worse number** → the positives are
  **maximization bias** (winner's curse on `max` of sample-means), not signal. The
  least-biased point (`d1/C32 ≈ −0.03`) reads ≈ tie.

## My contention (to be critiqued)

1. **Adaptivity genuinely pays.** Proof by construction: a human executes a contingent
   policy (e.g. on the static route 10→9→…→17, if 10/9/8 are present early, beeline to
   CSCE having banked three and exit) that no fixed route can encode. The structure
   (disjunctive detectors + exactly-5 correlation + early exit) carries real adaptive value.
2. **My numbers fail to show it for two artifact reasons, not absence of value:**
   - (a) **Unit values mute the margin** — gain is timing-only; you collect all present
     anyway and there is nothing of differing worth to "chase." (This has recurred ~3×.)
   - (b) My **readout is biased** — `max`-of-sample-means overstates; the induced policy's
     **measured rate** is the honest metric.
3. **Path:** (i) measure the induced policy's **rate by unbiased Monte-Carlo**, not the
   biased value estimate; (ii) introduce **heterogeneous gil values** (synthetic now, real
   redis values later) — the regime where the adaptive margin should clearly exceed static.
   Sparse sampling stays the convergent anchor, evaluated by unbiased policy rollout.
4. **τ_4 teleport** is likely **dominated** (greedy never used it across 2,500 runs;
   rollout's 8/80 reads as its own noise).

## What I want the consult to judge

- Is the diagnosis **sound and complete**, or is there a **modeling/implementation flaw**
  that is the real reason adaptive underperforms — e.g. the **base policy** (greedy
  ignores detectors), the **detector representation** (region *representative point* +
  point-in-polygon for the disjunctive cover set), the **Dinkelbach/λ handling**, the
  **renewal-rate formulation**, or the **fairness of the static baseline**?
- Is the **maximization-bias** diagnosis correct?
- Is "**unbiased policy-rate MC + heterogeneous values**" the right next step, or is there
  a **better/cheaper** route (cluster decomposition into exactly-solvable neighbourhoods; a
  proper POMDP method; a different anchor)?
- What am I **over-claiming or missing**? Could adaptivity genuinely *not* pay much even
  with heterogeneous values, and **how would we detect that**?
