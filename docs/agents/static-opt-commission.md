# Static-optimization agent — commission (verbatim)

> The exact prompt sent to the agent. Its report is in `static-opt-report.md`.

---

You are a research agent exploring STRUCTURAL SHORTCUTS to shrink the search and enable exact-or-cheap solving of sub-structures in a partially-observed planning problem ("chocofarm"). Systematically explore and QUANTIFY them on the real instance.

Work in your worktree **/home/bork/w/vdc/chocobo-staticopt** (branch `feat/static-opt`). Do NOT touch /home/bork/w/omega, the main checkout, or sibling worktrees. **Three pinned long-running processes occupy CPU cores 0,1,2** — do NOT disturb them; pin ANY python you run to core 3 (`taskset -c 3 …`), keep it BOUNDED and under `timeout`, no heavy solves, no parallel, low memory (counting/structure computations only; never enumerate reachable beliefs or run a full solver). Venv: `/home/bork/w/vdc/venvs/generic/bin/python`.

Read end to end: `env.py`, `policies.py`, `run.py`, `chocobo_geometry.py`, `chocobo_instance.json` (treasure coords; detection-region WKT; the `overlaps` array — 17 pairs; teleports), `docs/STATUS.md`, `docs/results/*.md`, `docs/agents/*-report.md`.

THE PROBLEM: adaptive stochastic orienteering / belief-MDP. 20 treasures, exactly 5 present per run (uniform w/o replacement → 15,504 equiprobable latent "worlds"; bitmask, bit t = treasure t present; i.i.d. re-roll). 16 OVERLAPPING detection regions → binary DISJUNCTIVE observations ("≥1 present among a covered set"); 4 δ treasures (observe==collect). Objective = long-run rate = treasures/time via Dinkelbach. Belief = the surviving-world set. A clairvoyant policy scores +70% over static, but search methods capture ~none — the VoI is gated behind deep sensing chains. Exact full backward induction is infeasible (the reachable belief space blows up).

THE LEAD (maintainer's): the early nodes are tightly clustered geographically — e.g. {8,9,10,11,12} near the CSNE teleport — so an early "nothing" (negative detection) there collapses a large chunk of the belief/search space, enabling hierarchical or conditional models and memoization. There are likely MANY such shortcuts. Find and QUANTIFY them.

INVESTIGATE, with bounded computation on the real instance (the 15,504-world array is cheap to filter — DO the counts):
- **Cluster structure:** geographic + overlap clusters of treasures/detectors (coords + the overlaps graph). Which treasures co-cluster; which detectors cover which clusters.
- **Conditional belief collapse (quantify):** given exactly-5-of-20, count |consistent worlds| after key observations — an early NEGATIVE on a multi-cover detector (rules out all covered treasures), a POSITIVE, and informative combinations. Report the prune factors (how many of 15,504 survive). Identify the highest-information early observations.
- **Decomposition:** does the problem approximately factor into per-cluster sub-problems chained by teleports (a max-mean-ratio cycle over waystones)? Quantify the coupling (exactly-5 global vs per-cluster hypergeometric counts). Identify sub-problems small enough to solve EXACTLY.
- **Conditional independence / hierarchy:** after a cluster-level "nothing", are remaining treasures ~independent across clusters? Could a hierarchical policy (choose cluster to probe → route within) exploit it?
- **Memoization:** what canonical sub-states recur (e.g. (cluster, local-belief-pattern, location))? Estimate the count of distinct canonical sub-states — is memoization viable where full belief-MDP memoization wasn't?
- **Exact-solvable sub-problems:** the largest sub-structures where exact backward induction IS tractable (small belief), as trusted anchors.

DELIVERABLE: write `docs/design/static-shortcuts.md` in your worktree — a structural map of the instance + a RANKED list of concrete, QUANTIFIED shortcut opportunities (each with the measured prune factor / sub-problem size / memoization estimate) + a recommended decomposition/hierarchy that could make near-optimal solving tractable. Back every claim with a number computed from the real instance (cite the filter + count you ran). Commit on `feat/static-opt` (EXPLICIT paths, never `git add -A`), `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`. Do NOT push; do NOT edit env.py/policies.py/run.py. Return a complete final report: structural map, ranked quantified shortcuts, recommended decomposition, which sub-problems are exactly solvable.
