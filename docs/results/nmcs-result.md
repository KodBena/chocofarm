# Nested Monte-Carlo Search vs the VoI ceiling (point-in-time)

NMCS (Cazenave, "Nested Monte-Carlo Search", IJCAI 2009) implemented as a pluggable
`Policy` subclass (`nmcs.py`) against the unchanged `env.py`. Measured by the env's own
unbiased Dinkelbach rate (`eval_nmcs.py`). UNIT values; detectors disjunctive (the real 17
overlaps). static floor **0.0855**; clairvoyant ceiling **0.1454** (**+70%** headroom).

## Algorithm as implemented

Faithful NMCS skeleton: a level-n search, at each step, tries every (pruned) legal move,
runs a level-(n-1) search from each resulting state, keeps the move whose continuation
scored best, plays it, and recurses; **level-0 is a base playout**. The best full line
found so far is memorized and its first action replayed when a fresh nested search regresses
(Cazenave's memorize-best-sequence rule).

Three adaptations to this stochastic, partially-observed, finite-horizon belief-MDP:

1. **Episodic / finite-horizon.** A line ends at TERMINATE, when no informative action
   remains, or at a step cap. TERMINATE is always a candidate, so the search can choose to
   bank-and-exit early.
2. **Determinized, world-averaged scoring.** The latent world is hidden, so every `apply`
   inside the search samples a concrete world from the current belief (`env.sample_world`,
   unbiased by the env contract). A playout samples a world and plays a base policy
   (Greedy) to the end in it; the score is the lambda-penalized return
   `sum(value) - lam*(travel + exit)`. Both the level-0 playout and the per-move level-n
   evaluation are averaged over a few sampled worlds to cut determinization variance.
3. **lambda is the Dinkelbach penalty** passed to `decide`; every score is lambda-penalized,
   so maximizing it maximizes the renewal-reward objective at the current rate target.

`decide` runs a bounded NMCS search from the current observed state and returns only the
first action of the best line; the real belief shrinks as the episode observes, so
re-planning per step is the natural fit. Memory stays flat — one path plus a handful of
sampled worlds; no belief enumeration or caching (bounded-safety).

Branching is pruned to the nearest `cand_det` informative detectors + nearest `cand_tre`
uncollected treasures + TERMINATE (the same rate-aware pruning RolloutPolicy already uses).

## Results

| policy | rate | % of ceiling | VoI clawed | runs | notes |
|---|---|---|---|---|---|
| nmcs(level=1) | 0.0780 | 54% | **-13%** | 120 | base=Greedy, cand_det=1 |
| nmcs(level=2) | 0.0688 | 47% | **-28%** | 40 | leaner branching/samples; noisy at N=40 |

Equal-N cross-check at a common lambda=0.0855 (50 episodes each): L1 = 0.0813
(ER=4.08, ET=50.2, 3.46 det/run), L2 = 0.0747 (ER=4.18, ET=56.0, 3.48 det/run).
Clairvoyant for comparison: ER=4.55 in ET=31.

## Findings (honest)

- **NMCS does not clear the static floor on this instance under unit values.** Both levels
  land below 0.0855, joining the existing shallow policies (greedy -8%, CE -17%,
  rollout(CE) -10%); only rollout(greedy) was marginally positive (+6%). NMCS does **not**
  claw back the +70% headroom; it sits with the rest of the pack.
- **Level 2 is worse than level 1, not better.** At equal N and common lambda, deeper
  search collects marginally *more* treasure (ER 4.18 vs 4.08) but at meaningfully higher
  travel (ET 56 vs 50), netting a lower rate. This is the same signature the project
  recorded for sparse sampling ("more samples gave a worse number") — **maximization /
  winner's-curse bias**: a deeper nest takes `max` over more noisy determinized estimates,
  and the extra optimism shows up as over-collection and over-travel rather than signal.
- **Root cause: determinization optimism.** Scoring a line against a *fully revealed*
  sampled world makes detours and extra collection look risk-free — in any known world the
  greedy base happily grabs all five present treasures and a detector reads as free perfect
  information. World-averaging dampens this but does not remove it. The symptom is constant
  across configs: NMCS earns clairvoyant-like reward (~4.1 of 4.55) but spends 50-80% more
  time to do it (ET ~50-56 vs 31). The over-detouring is what `cand_det=1` partially curbs
  (at `cand_det=4`, L1 over-visits ~6-7 detectors/episode and falls to -22%).
- **The decoupling held.** NMCS dropped in as a `Policy` subclass with no change to
  `env.py`, `policies.py`, or `run.py`; it reuses `_base_value` and the env belief/dynamics
  primitives directly.

## Caveats

- Budgets are small for bounded-safety; level 2 at 40 runs has a non-trivial MC standard
  error (the equal-N=50 cross-check is the more trustworthy comparison and agrees in
  direction). The qualitative conclusion (NMCS below floor, level 2 below level 1) is robust
  to the noise; the exact rates are not.
- This is the **unit-value** regime, where (per the consult and the het-values result)
  detector information can only reorder/skip travel, never change what is worth collecting —
  the channel through which deep contingent search would pay is structurally narrow. NMCS
  may behave differently under heterogeneous values; not tested here.
- The faithful determinized NMCS is, on this problem, a victim of the same optimism that
  the existing rollout dampens by averaging over the belief at a *single* step. A
  variance-aware variant (penalize per-move score spread across worlds, or score against a
  base that may not collect un-confirmed treasures) would be the natural next probe, but
  that departs from textbook NMCS and is left for a follow-up.

## Literature consulted

Cazenave, "Nested Monte-Carlo Search," IJCAI 2009 (the PDF would not parse to text via the
fetch tool); the algorithm structure was confirmed from secondary descriptions of the
level-n recursion and the memorize-best-sequence rule (a Wikipedia MCTS overview and a
worked single-player NMCS implementation walkthrough). What was taken: the level-n / level-0
recursion, per-step "try every move, run a level-(n-1) search, keep the best," and the
memorize-and-replay-the-best-sequence rule.
