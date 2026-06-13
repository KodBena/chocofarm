# NMCS agent — report (verbatim)

> The implementation agent's complete final message, reproduced verbatim. Commission in
> `nmcs-commission.md`. Recorded per the verbatim-record discipline; not retro-edited.

---

Committed on `feat/nmcs` (not pushed, as instructed). Work is complete.

## Summary

I implemented Nested Monte-Carlo Search as a pluggable `Policy` in the chocobo-nmcs worktree and measured it honestly. The result is a genuine negative finding, not a bug.

### Algorithm as implemented (`/home/bork/w/vdc/chocobo-nmcs/nmcs.py`)
Faithful Cazenave-2009 NMCS skeleton — a level-n search, at each step, tries every (pruned) legal move, runs a level-(n−1) search from each resulting state, keeps the best move, plays it, recurses; level-0 is a base playout; the best full line is memorized and its first action replayed when a deeper search regresses. Three adaptations to this stochastic, partially-observed, finite-horizon belief-MDP:
- **Determinized, world-averaged scoring**: every `apply` inside the search samples a concrete world from the current belief (`env.sample_world`, unbiased per the env contract); a playout plays the Greedy base to the end in that world, scored by the λ-penalized return; both level-0 playouts and per-move level-n evaluations are averaged over a few sampled worlds.
- **Episodic**: TERMINATE is always a candidate, so the search can bank-and-exit early.
- `decide` runs a bounded search and returns the first action of the best line; re-planning per real step over the shrinking belief keeps memory flat (no belief enumeration/caching).

### Literature
Cazenave, "Nested Monte-Carlo Search," IJCAI 2009. The PDF would not parse to text via the fetch tool, so I confirmed the level-n/level-0 recursion and the memorize-best-sequence rule from secondary descriptions (MCTS overview + a worked single-player NMCS walkthrough). Took: the recursion structure, per-step "try every move / keep best," and the memorize-and-replay rule.

### Measured results (unit values; static floor 0.0855, clairvoyant ceiling 0.1454, +70% headroom)
- **Level 1: rate 0.0780 = 54% of ceiling, −13% of the VoI gap** (120 runs, ~157s)
- **Level 2: rate 0.0688 = 47% of ceiling, −28%** (40 runs, ~203s; noisy at N=40)
- Equal-N cross-check at common λ=0.0855 (50 episodes): L1 = 0.0813, L2 = 0.0747 — confirms L2 < L1 robustly.

**NMCS does not clear the static floor and claws back none of the +70%.** It joins the existing shallow pack (greedy −8%, CE −17%, rollout(CE) −10%; only rollout(greedy) was +6%).

### Why (honest)
- **Determinization optimism** is the root cause: scoring lines against a *fully revealed* sampled world makes detours and over-collection look risk-free. NMCS earns clairvoyant-like reward (~4.1 of 4.55) but spends 50–80% more time (ET ~50–56 vs the clairvoyant's 31). World-averaging dampens but doesn't remove it.
- **Level 2 is worse than level 1** — deeper nesting takes `max` over more noisy determinized estimates, amplifying the optimism into more over-travel. This is exactly the maximization/winner's-curse signature the project already documented for sparse sampling ("more samples gave a worse number").
- Detector over-visiting is the visible symptom; tightening detector candidates to `cand_det=1` lifted level 1 from −22% to −13% but couldn't get above the floor.

### Caveats
- Budgets are deliberately small (bounded-safety); level-2 at N=40 has real MC noise — the equal-N=50 cross-check is the trustworthy comparison and agrees in direction. Qualitative conclusions are robust; exact rates are not.
- Unit-value regime only: detector info can only reorder/skip travel here, never change what's worth collecting, so the channel deep search would exploit is structurally narrow. A variance-aware NMCS variant (penalize per-move score spread, or a base that won't collect un-confirmed treasures) is the natural follow-up but departs from textbook NMCS.

### Deliverables (committed on `feat/nmcs`, commit `bf6172c`, **not pushed**)
- `/home/bork/w/vdc/chocobo-nmcs/nmcs.py`
- `/home/bork/w/vdc/chocobo-nmcs/eval_nmcs.py` (run a single level via `python eval_nmcs.py 1` to bound wall-time)
- `/home/bork/w/vdc/chocobo-nmcs/docs/results/nmcs-result.md`

`env.py`, `policies.py`, `run.py` are unchanged (verified — no diff). Staged by explicit path, no `git add -A`.
