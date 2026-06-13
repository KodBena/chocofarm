# Decomposition-solver agent — report (verbatim)

> Complete final message, verbatim. Commission in `decomp-solver-commission.md`. The headline
> (rate 0.0941, +14% of the +70% VoI, first method to clear the static floor) was
> independently re-measured by the coordinator at a fixed λ=0.094 / fresh seed: rate 0.0951
> (+11% over floor, +16% of VoI) — confirmed, not noise.

---

Done. The commit is on `feat/decomp-solver`, not pushed, and the only remaining untracked item is the auto-generated `__pycache__` (not staged). The working tree is otherwise clean.

Let me provide the final report.

---

## Report: Exact Hierarchical-Decomposition Solver for chocofarm

Built, integrated, measured, and committed on `feat/decomp-solver` in `/home/bork/w/vdc/chocobo-decomp` (not pushed). All Python pinned to core 3 under `timeout`, no parallelism. An out-of-frame hack-rationalization audit was run and its findings discharged before commit.

### Deliverables
- `decomp_solver.py` — micro + macro + `DecompPolicy` (674 lines, abstract/auditable)
- `eval_decomp.py` — eval harness
- `docs/results/decomp-rate.md` — full writeup with honest caveats

### MICRO — per-cluster exact belief-MDP (confirmed tractable)
Exact backward induction over the local belief lattice (state = `(in-cluster position, surviving-present-set support, locally-collected)`; actions = in-cluster face reads / collects / LEAVE). Two exact constructions:
- **Occupancy-conditioned** (`k` fixed, what the macro chains): uniform within `k` by the `∏ C(size,k)` factorization — exact. Reachable beliefs top out at **558** (SE+mid k=3); VI tables ≤**7,949 states**, milliseconds each. Yields exact `(E[R|k], E[T|k])`; at λ≈0.094 the optimal micro senses-then-collects every present member (`E[R|k]=k` exactly).
- **Joint** (occupancy-marginal, what the runtime executes): the unconditioned lattice the analyzer sized (**745** NW / **1,448** SE+mid beliefs, ≤**34,405 states**), with each present-set weighted by its exact completion count `C(N−size, K−j)` so it is exact under the true env prior.

All four real clusters are well under the tractability cap (largest size 6); a hard `max_states` guard aborts loudly (ADR-0002) on an over-cap synthetic blob. **Exact VI on the sub-problems is confirmed tractable — the decomposition's premise holds.**

### MACRO — exact occupancy chaining
Receding-horizon expectimax over the cluster-visit sequence of one excursion. The occupancy posterior is **recomputed exactly from the live belief** at every decision (multivariate hypergeometric over Σ=5; verified to reduce to the analytic `∏ C(size,k)` prior at 0.0 deviation). The cluster-enter look-ahead conditions the full joint — **exact**. Shipped default **horizon=1** (myopic), which is exact (no recursion) and statistically indistinguishable from horizon=2. The δ-singletons {3,4,16,19} are handled as a symmetric pool (observe==collect), picking the cheapest uncollected one.

### Measured headline (N=2000 Dinkelbach, on the honest env)
| | rate | % of ceiling | **% of +70% VoI** | ER | ET |
|---|---|---|---|---|---|
| static floor | 0.0855 | 59% | 0% | — | — |
| **decomp (exact, h=1)** | **0.0941** | **65%** | **+14%** | 3.27 | 34.7 |
| clairvoyant ceiling | 0.1454 | 100% | +100% | 4.55 | 31.3 |

**Exact decomposition is the first method to clear the static floor**, clawing back **~14% of the +70%** (a clean fixed-λ scan resolves the achievable rate flat at 0.0937–0.0940). The entire search pack sits at/below the floor: greedy −8%, certainty-equiv −17%, rollout(CE) −11%, rollout(greedy) +7% (noise-band), NMCS/ISMCTS −12% to −39%. The gain is modest in absolute terms but a qualitative break — the structure-exploiting solver values the contingent sensing chains shallow search could not pay for. The remaining gap to the ceiling is mostly *reward* (ER 3.27 vs 4.55): at the rate-optimal λ, several clusters aren't worth their face-read chain, so present treasures there go uncollected — the genuine VoI frontier.

### Honest caveats
- **Rate is unbiased** regardless of table approximations — the env charges exact `env.d`; the tables only steer decisions.
- **Macro occupancy handling is exact at the shipped horizon=1** (full joint, recomputed from live belief; enter look-ahead conditions the full joint). The independence approximation the analyzer measured (≤2.8%) is **not** invoked by the realised policy.
- The deeper-horizon (≥2, inspection-only) **δ-dip look-ahead carries a scoped staleness approximation** (δ-pool occupancy held at prior across a chain of dips) — confined to that path, never biasing the rate. This is documented in code and doc (an earlier overclaim of unconditional exactness was caught by the out-of-frame audit and corrected).
- **Re-anchoring** is a decision-only geometric approximation (planner scores entry→anchor; runtime corrects the executed action exactly from the live loc).
- **On-demand micro solves** (coupling-conditioned supports not in the marginal build) are solved exactly and cached — a small, N-dependent warm-up cost, not a correctness signal.

The audit's other findings (stale module-header default, dead `_ep["post"]` slot, δ-pool selection bug that tracked a count not the set) were all fixed: the δ fix is a genuine invariant (cheapest *uncollected* δ), verified with zero cross-cluster action leakage over 150 episodes.
