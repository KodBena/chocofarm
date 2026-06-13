# Exact hierarchical-decomposition solver — measured rate (point-in-time)

The first chocofarm policy to clear the **static floor**: an EXACT two-layer
decomposition (per-cluster belief-MDPs chained by an occupancy macro) measured on
the honest arrangement-face env. UNIT values throughout. Static floor **0.0855**,
clairvoyant ceiling **0.1454** (**+70%** VoI headroom — both detector-independent,
re-confirmed unmoved here).

> Why this path. Flat exact solving of the C(20,5)=15,504-world belief-MDP is
> intractable, and the shallow/search pack (greedy, certainty-equiv, rollout,
> sparse-sampling, NMCS, ISMCTS) all sit at or below the floor — the value of
> information is gated behind face-read *chains* too deep for shallow search to pay
> for (`docs/design/static-analysis-faces.md` §6, `docs/results/honest-rates-faces.md`).
> But the problem FACTORS: treasures partition into small sense-clusters, coupled
> only by the global Σ=5; each cluster's belief-MDP is microscopic and exactly
> solvable; a macro layer reasons over the cluster-occupancy latent. This is the
> §6 "right structure, still operationally reachable" path, built and measured.

## Headline

| policy | rate | % of ceiling | **% of the +70% VoI** | ER | ET | N |
|---|---|---|---|---|---|---|
| realizable static (floor) | 0.0855 | 59% | 0% | — | — | — |
| **decomp (exact, myopic macro h=1)** | **0.0941** | **65%** | **+14%** | 3.27 | 34.7 | 2000 |
| decomp (exact, macro h=2) | 0.0947 | 65% | +15% | 3.33 | 35.2 | 2000 |
| clairvoyant (free perfect info) | 0.1454 | 100% | +100% | 4.55 | 31.3 | — |

`% of the +70% VoI` = (rate − 0.0855) / (0.1454 − 0.0855), the fraction of the
adaptivity headroom recovered. Measured at the policy's own Dinkelbach fixed point
(λ ≈ 0.093). A clean fixed-λ scan at N=2000 puts the achievable rate flat at
**0.0937–0.0940 across λ∈[0.088, 0.100]**, crossing rate=λ at λ≈0.094 — so the
fixed point is well-resolved (±0.001).

**Exact decomposition is the first method above the floor.** It claws back ~+14–15%
of the +70%, where the entire search pack is below it: greedy −8%, certainty-equiv
−17%, rollout(CE) −11%, rollout(greedy) +7% (noise-band), NMCS/ISMCTS −12% to −39%
(`honest-rates-faces.md`). The gain is modest in absolute terms but it is a
*qualitative* break: the structure-exploiting solver values the contingent sensing
chains the search methods could not.

## The micro layer — exact per-cluster belief-MDPs (tractable, confirmed)

For each cluster, `build_cluster_micro` does **exact backward induction** over the
local belief semilattice (state = `(in-cluster position, support of surviving local
present-sets, locally-collected)`; actions = in-cluster face reads / collects /
LEAVE; expectimax over read polarities and presence). Two exact constructions:

- **Occupancy-conditioned** (`k` fixed) — the MDP the MACRO chains. Conditioned on
  occupancy `k`, the env prior is *uniform* over the cluster's `C(size,k)`
  present-sets (the `∏ C(size,k)` factorization, exact), so the per-`k` solve carries
  no weights. Reachable-belief counts (the analyzer's occupancy-conditioned sizing)
  top out at **558** (SE+mid, k=3); the value-iteration tables (× positions ×
  collected) peak at **7,949 states** — milliseconds each. Output: the exact
  `(E[R|k], E[T|k])` per occupancy. At λ≈0.094 the optimal micro collects every
  present member (`E[R|k] = k` exactly) after sensing to localise them — e.g. NW
  k=2 spends ~4.9 time-units to sense-then-collect both present treasures.

- **Joint** (occupancy-marginal) — the MDP the RUNTIME executes, because a priori
  the cluster's occupancy is unknown and the policy must act on the mixed-`k`
  belief. This is the unconditioned local semilattice the analyzer sized (**745** beliefs
  for NW, **1,448** for SE+mid); the VI table peaks at **34,405 states** (SE+mid),
  built in ~1.4 s. Here the cross-occupancy weights are **not** uniform, so each
  local present-set carries its exact env-prior weight `C(N−size, K−j)` (its
  completion count) — this makes the joint VI exact under the true prior.

All four real clusters are comfortably under the tractability cap (the largest is
size 6); `max_states` aborts loudly (ADR-0002) on an over-cap synthetic blob rather
than hanging. **Exact value iteration on the per-cluster sub-problems is confirmed
tractable** — the decomposition's central premise holds. Whole-table build for all
clusters × occupancies × the joint solve is ~3 s per λ, cached.

## The macro layer — exact occupancy chaining

`MacroPlanner` decides, at each cluster boundary of one excursion, which cluster to
enter next, when to bank-and-exit, and (via `env.nearest_exit` at TERMINATE) which
of the three teleports to leave by. The latent is the **cluster-occupancy vector**,
a multivariate hypergeometric over Σ k_c = 5 (the analyzer's ≤320 partitions),
tracked **exactly**:

- At every macro decision the joint occupancy posterior is **recomputed exactly from
  the live belief** (`_live_occupancy_posterior`: project each surviving world onto
  the cells — the 4 sense clusters + the δ pool — and group-count). This reflects
  every reveal so far (cluster chains, δ collects) with no incremental bookkeeping,
  and reduces to the analytic prior `∏ C(size,k)` on the full world set (verified to
  0.0 absolute deviation; NW marginal E[k]=1.25 = 5·5/20).
- Entering a cluster yields the micro layer's exact `(E[R|k], E[T|k])`; the
  realised occupancy is revealed by the micro's own reads (the cluster's face-read
  chain *is* the occupancy resolution). The cluster-ENTER look-ahead conditions the
  full joint posterior (`cond = {v : v[ci]=k}`) — exact at every depth.

**Exactness scope (precise).** At the **shipped default horizon=1** the macro makes
one decision per boundary and re-plans, so the realised policy's occupancy handling
is **exact — no independence approximation** (the posterior is the exact live joint,
the enter look-ahead conditions the full joint, and there is no recursion to
accumulate error). The deeper-horizon expectimax (kept for inspection) is **not**
fully exact: its δ-dip branch at horizon≥2 holds the δ-pool occupancy at its prior
across a chain of δ dips (the `p_present` denominator does not shrink and the
recursion reuses the unconditioned posterior) — an independence/staleness
approximation **confined to the horizon≥2 δ-dip look-ahead**. It never biases the
measured rate (the env recomputes the live posterior and charges exact travel each
step); it only degrades a deeper plan's δ valuation. The cluster-enter branch is
exact at all depths.

| macro horizon | Dinkelbach rate | % of +70% | ER | ET | occupancy handling |
|---|---|---|---|---|---|
| **1 (myopic, default)** | **0.0941** | **+14%** | 3.27 | 34.7 | exact |
| 2 | 0.0947 | +15% | 3.33 | 35.2 | exact enter; approx δ-dip |

The two differ by ~0.0006 (≈1% of rate, within the N=2000 standard error), so
horizon=1 — the simplest, "enter the single best cluster then re-evaluate", and the
one with no approximation — is the default; horizon=2 is marginally higher but
statistically indistinguishable. (An earlier sweep showed deeper look-ahead
*hurting*; that was an artifact of a δ-pool selection bug — the macro tracked only a
*count* of δ collected and could re-pick a collected δ and prematurely exit. Fixed
by tracking `delta_done` as the set of collected δ ids and picking the cheapest
*uncollected* one.)

## Where the remaining gap to the ceiling lives

The decomp banks **ER≈3.3** treasures in **ET≈35**; the clairvoyant banks **4.55**
in **31.3**. The gap is mostly *reward*, not time: being selective enough to keep ET
near the clairvoyant's means leaving present treasures uncollected in clusters the
macro declined to enter. That is the genuine VoI frontier — the clairvoyant pays
nothing to know where all five are, while the decomp must spend a face-read chain
per cluster to find them, and at the rate-optimal λ several clusters are not worth
the chain. The +14% it does claw back is exactly the value of resolving the clusters
it *does* enter, optimally.

## Honest caveats

- **Rate-honesty vs decision-quality.** The env is ground truth: every distance the
  measured rate sees is exact `env.d`, so the rate is **unbiased** regardless of any
  table approximation. The micro/macro tables only *steer* decisions. The
  approximations below therefore cost decision quality (a lower rate), never honesty
  of the reported number.

- **Macro occupancy handling is exact at the shipped horizon=1** (full
  multivariate-hypergeometric joint, recomputed from the live belief, so every
  reveal is reflected exactly; the enter look-ahead conditions the full joint). The
  deeper-horizon δ-dip look-ahead (horizon≥2, inspection-only) carries the scoped
  staleness approximation described above — it does not bias the rate.

- **Micro within-occupancy uniformity is exact** (the `∏ C(size,k)` factorization),
  and the joint micro's completion-count weights make the occupancy-marginal solve
  exact too.

- **Macro re-anchoring (geometric, decision-only).** The planner scores a cluster's
  travel as entry→*anchor* (the cluster member nearest the entry teleport) and the
  micro's `E[T|k]` is measured from that anchor; the realised excursion enters from
  the live position and exits at a world-dependent member. The runtime corrects the
  *executed* action exactly (the joint micro solver re-evaluates from the live loc),
  but the macro's *cluster-choice* value uses the anchor estimate. This can pick a
  slightly sub-optimal next cluster; it does not bias the rate. The δ pool is
  scored by its marginal P(present) and entered one cheapest-uncollected treasure at
  a time (δ are observe==collect, no faces).

- **On-demand micro solves.** A runtime state whose `(loc, support, collected)` the
  marginal joint build never enumerated — the first in-cluster step from a
  non-anchor entry, or a support the global Σ=5 coupling conditioned to a novel
  subset — is solved on demand by the memoised exact solver and cached (bounded by
  the reachable-belief count). Because the per-cluster memo is shared across all
  episodes at a fixed λ, this is a one-time warm-up cost that amortises as the memo
  fills — the count is N-dependent and small (order ~10² over a few-hundred-episode
  run, tailing off thereafter). The solve is **exact either way** — the count is
  purely a precompute-coverage diagnostic, not a correctness signal.

- **Budget.** Headline at N=2000 final Dinkelbach episodes (4 warm iterations × 400),
  ~40 s on one core. The ratio estimator on R∈{0..5}/episode has a standard error of
  ~1% at N=2000, so 0.0941 is resolved to ~±0.001; the h=1/h=2 difference is below
  that. All runs bounded under `timeout`, pinned to CPU core 3, no parallel solvers
  (cores 0–2 held by unrelated pinned runners).

## Reproduce

```
timeout 600 taskset -c 3 /home/bork/w/vdc/venvs/generic/bin/python -m chocofarm.eval.eval_decomp \
    --runs 2000 --search --horizon-sweep
```

Source: `decomp_solver.py` (micro `build_cluster_micro`, macro `MacroPlanner`,
policy `DecompPolicy`), `eval_decomp.py` (harness, reusing `run.py`'s
`realizable_static` / `clairvoyant_rate` and `env.dinkelbach_rate`). Structural
sizing: `analyzer.py` / `docs/design/static-analysis-faces.md`. Search-pack
baselines: `docs/results/honest-rates-faces.md`.
