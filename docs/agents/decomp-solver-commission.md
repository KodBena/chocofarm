# Decomposition-solver agent — commission (verbatim)

> The exact prompt sent to the agent (the relaunch, after an accidental stop of the first
> launch which had done no work). Its report is in `decomp-solver-report.md`.

---

You are building the EXACT HIERARCHICAL DECOMPOSITION SOLVER for chocofarm — the path the postmortem + static analysis identify as most likely to actually capture the +70% value-of-information, by replacing intractable flat search with exact solving of small per-cluster sub-problems chained by a macro layer. Work in your worktree **/home/bork/w/vdc/chocobo-decomp** (branch `feat/decomp-solver`) — it already exists and is clean. Do NOT touch /home/bork/w/omega, the main checkout, or sibling worktrees. **Three pinned indefinite runners occupy CPU cores 0,1,2** — pin ANY python you run to **core 3** (`taskset -c 3 …`), keep it BOUNDED, under `timeout`, no parallel. Venv: `/home/bork/w/vdc/venvs/generic/bin/python`.

**READ FIRST (your worktree):**
- `analyzer.py` (`analyze(instance)`) + `docs/design/static-analysis-faces.md` — the HONEST structure: clusters (NW {8,9,10,11,12}; SE+mid {0,1,2,13,14,15}; N {5,6,7}; S {17,18}; δ {3,4,16,19}), per-cluster reachable local-belief counts (NW 745, SE+mid 1448 — small), the occupancy factorization `#worlds = ∏ C(size,k)`, the teleport 3-region partition.
- `facemodel.py`, `arrangement.py`, `chocobo_faces.json` — the corrected face sensing model (sense-action = stand at a face, read disjunction over its cover).
- `env.py` — the honest `Environment` (sense-actions are the 44 faces; `legal_actions`/`apply`/`filter_*`/`rate`/`dinkelbach_rate`; the exactly-5-of-20 prior; static floor 0.0855, clairvoyant ceiling 0.1454 = +70%).
- `run.py` (`realizable_static`, `clairvoyant_rate`), `policies.py` (the Policy interface).

**THE PROBLEM:** 20 treasures, exactly 5 present per run (15,504 worlds). Sensing per-face (44 faces, mostly singleton-cover; a cluster resolves only via a *chain* of face-reads, not one). Objective: long-run treasures/time (renewal-reward, Dinkelbach). Flat exact solving is infeasible; approximate search (NMCS/ISMCTS/rollout) sits below the static floor. But the problem FACTORS: clusters are small enough for exact value iteration, coupled only by the global Σ=5 (handled at the macro).

**BUILD — two layers (abstract & auditable; the maintainer audits by reasoning):**
1. **MICRO — per-cluster exact belief-MDP.** For each cluster, the local problem is: sense via the cluster's faces, collect its treasures, and "leave" (the boundary action). The local belief = the surviving local present-sets (sized by `analyze`'s reachable_local_beliefs — hundreds, tractable). Do EXACT backward induction / value iteration on the λ-penalized objective (`Σ value − λ·time`) over `(position-in-cluster, local belief, collected)`, conditioned on the cluster's occupancy (how many of the 5 are in it). Output: the exact per-cluster value + policy.
2. **MACRO — occupancy chaining.** The latent the macro reasons over is the cluster-occupancy vector (multivariate-hypergeometric, `analyze`'s ~320 partitions); visiting a cluster reveals its occupancy and yields the micro layer's exact (reward, time) for that occupancy. Decide which cluster to enter, when to bank-and-exit, which of the 3 teleports — solving the renewal/average-reward problem (Dinkelbach) over the macro states. The Σ=5 coupling is handled exactly here via the occupancy posterior (note honestly if you approximate clusters as independent given occupancy — `analyze` measured ≤2.8% deviation).

**INTEGRATE + MEASURE (the headline).** Wrap it as a `Policy` (precompute the micro/macro tables, then `decide` executes them) so `env.rate`/`dinkelbach_rate` evaluates it on the honest env. **Report its rate vs the static floor (0.0855) and the clairvoyant ceiling (0.1454): how much of the +70% does exact decomposition capture?** Compare to the shallow/search pack (all below floor). Bounded N for the MC eval; pin to core 3; under timeout.

**DELIVERABLES (commit on `feat/decomp-solver`, explicit paths, never `git add -A`, `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`, do NOT push):** a clean `decomp_solver.py` (micro + macro, abstract), an eval script, and `docs/results/decomp-rate.md` (the rate vs floor/ceiling, % of +70% captured, comparison to the search pack, honest caveats — incl. any independence approximation, budget bounds, and whether the macro occupancy handling is exact). RETURN a complete report: the micro solve (per-cluster belief sizes + that exact VI was tractable), the macro chaining, the measured rate and % of the +70% captured, and honest caveats.
