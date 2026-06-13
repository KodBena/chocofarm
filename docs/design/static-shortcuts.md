# Static structural shortcuts for chocofarm — a quantified map

> Research note (point-in-time). Every number below is computed from the **real instance**
> (`chocobo_instance.json`, loaded through `env.py`) by a bounded filter/count, pinned to
> CPU core 3 under `timeout`. No solver was run; no reachable-belief enumeration of the
> global problem was attempted. The cited "filter + count" is the operation that produced
> each figure, so any claim can be re-derived in seconds.

## 0. The problem in one line

Adaptive stochastic orienteering / belief-MDP: 20 treasures, **exactly 5 present** per run
(uniform without replacement → `C(20,5) = 15,504` equiprobable worlds; bitmask, bit t =
present; i.i.d. re-roll). 16 overlapping detection regions give **binary disjunctive**
observations ("≥1 present in the covered set"); 4 δ-treasures (3, 4, 16, 19) have no region
(observe == collect). Objective = long-run rate (treasures/time) via Dinkelbach. Documented
references: static floor **0.0855**, clairvoyant ceiling **0.1454** (**+70%** VoI headroom);
every search method built so far (greedy, CE, rollout, NMCS L1/L2, SO-ISMCTS) claws back ≈0%
of that +70%. The bottleneck named by the consult and the search agents is **depth**: the VoI
is gated behind multi-step sensing chains the shallow methods cannot reach. This note finds
and quantifies the structure that makes those chains — and the sub-problems behind them —
**cheap to solve exactly**.

---

## 1. Structural map of the instance

### 1.1 The cover map (what each detector senses)

The current `env.py` builds `cover_mask` from the real 17-pair `overlaps` array (the consult's
flagged rep-point bug is **fixed** in this env — verified: the loaded cover sets below are the
disjunctive unions, not singletons). Filter: enumerate `(cover_mask[i] >> t) & 1` per detector.

| detector | cover set | size |
|---|---|---|
| D_0 | {0, 1, 2} | 3 |
| D_1 | {0, 1, 15} | 3 |
| D_2 | {0, 2} | 2 |
| D_5 | {5, 7} | 2 |
| D_6 | {6, 7} | 2 |
| D_7 | {5, 6, 7} | 3 |
| D_8 | {8, 9, 10, 11, 12} | 5 |
| D_9 | {8, 9, 10, 11, 12} | 5 |
| D_10 | {8, 9, 10} | 3 |
| D_11 | {8, 9, 11, 12} | 4 |
| D_12 | {8, 9, 11, 12} | 4 |
| D_13 | {13, 14, 15} | 3 |
| D_14 | {13, 14, 15} | 3 |
| D_15 | {1, 13, 14, 15} | 4 |
| D_17 | {17, 18} | 2 |
| D_18 | {17, 18} | 2 |

δ-treasures **{3, 4, 16, 19}** are covered by no detector — sensing-isolated; the only way to
learn their presence is to visit (observe == collect).

### 1.2 Sensing clusters (detector-cover connected components)

Filter: union-find over treasures linked when they co-occur in any detector's cover set.

- **{0, 1, 2, 13, 14, 15}** — SE pack {0,1,2} and mid pack {13,14,15} are **fused** by two
  bridge detectors: **D_1** covers {0,1,**15**} and **D_15** covers {**1**,13,14,15}. So
  treasures 1 and 15 stitch the two geographic packs into one sensing component.
- **{8, 9, 10, 11, 12}** — the NW pack, fully self-contained (no bridge out).
- **{5, 6, 7}** — the N pack, self-contained.
- **{17, 18}** — the S pair, self-contained.
- **{3}, {4}, {16}, {19}** — δ singletons, each its own component.

### 1.3 Geography and the teleport partition

Filter: cluster centroid → nearest teleport (Euclidean, `env.d`). Teleports: CSNE (−8.0, 11.7),
CSCE (−2.0, 2.0), τ_4 (4.2, 13.7); teleport overhead `tp = 12.0`; inter-teleport
d(CSNE,CSCE)=11.43, d(CSNE,τ_4)=12.28, d(CSCE,τ_4)=13.25.

| cluster | centroid | nearest TP (dist) | second |
|---|---|---|---|
| NW {8,9,10,11,12} | (−7.9, 9.2) | **CSNE** (2.6) | CSCE (9.3) |
| N {5,6,7} | (−4.0, 10.8) | **CSNE** (4.1) | tau_4 (8.6) |
| mid {13,14,15} | (−5.4, 6.0) | **CSCE** (5.2) | CSNE (6.3) |
| SE {0,1,2} | (−0.7, 7.1) | **CSCE** (5.3) | tau_4 (8.2) |
| S {17,18} | (−0.3, 2.6) | **CSCE** (1.8) | tau_4 (12.0) |
| δ16 | (−3.5, 3.7) | **CSCE** (2.3) | CSNE (9.2) |
| δ19 | (−7.8, 1.3) | **CSCE** (5.9) | CSNE (10.4) |
| δ3 | (1.4, 10.5) | **tau_4** (4.2) | CSCE (9.2) |
| δ4 | (4.2, 13.7) | **tau_4** (0.0) | tau_4 (8.6) |

The teleports induce a clean **three-region partition**:

- **CSNE region** (the entry): NW + N — exactly the 8 treasures {5,6,7,8,9,10,11,12}.
- **CSCE region**: mid + SE + S + δ16 + δ19 — {0,1,2,13,14,15,16,17,18,19}.
- **τ_4 region**: only δ3 + δ4 — {3,4}.

This confirms the maintainer's lead **quantitatively**: the entry CSNE sits 2.6 units from the
NW pack's centroid (0.5 from τ_10, 2.1 from τ_9, 2.2 from τ_8), so the cheapest first move is to
sense/collect the NW cluster, and **τ_4 is a degenerate teleport** serving only {3,4} — it is
the most distant exit from everything else (consistent with the project's "τ_4 dominated" note).

### 1.4 The irreducible atoms — detector-indistinguishability classes

Filter: detector-signature `(1 if cover_mask[d]>>t&1 else 0 for all d)` per treasure; group by
identical signature. These are pairs no sequence of detectors can ever separate — the **floor**
on belief sharpness from sensing alone.

- **{3, 4, 16, 19}** — no detector (must be visited).
- **{8, 9}** — identical signature (both in D_8,9,10,11,12).
- **{11, 12}** — identical (both in D_8,9,11,12).
- **{13, 14}** — identical (both in D_13,14,15).
- **{17, 18}** — identical (both in D_17,18).
- singletons (detector-resolvable): {0}, {1}, {2}, {5}, {6}, {7}, {10}, {15}.

Consequence: firing **all 16 detectors** still leaves expected belief **120.7 worlds** (a
128.4× collapse from 15,504), because within each indistinguishable pair "which of the two is
present" is unresolvable. Filter: partition all 15,504 worlds by their full 16-bit detector
outcome pattern → 319 classes, E[|belief|]=Σ(n²)/15504 = 120.7, largest class 498 worlds, only
5 classes are singletons. **Breaking a pair requires a collect/visit, not a detector** — this is
where the δ-style "observe==collect" channel does real work.

---

## 2. Conditional belief collapse — quantified prune factors

All counts are exact filters of the 15,504-world array. P(neg) matches the hypergeometric
`C(20−c,5)/C(20,5)` exactly for every detector (verified, c = cover size).

### 2.1 Single-observation collapse (from the full belief)

| detector | cover | P(neg) | #neg | **prune on NEG** | P(pos) | #pos | prune on POS | bits H(pos) |
|---|---|---|---|---|---|---|---|---|
| **D_8 / D_9** | {8,9,10,11,12} | 0.194 | **3003** | **5.16×** | 0.806 | 12501 | 1.24× | 0.709 |
| D_11 / D_12 | {8,9,11,12} | 0.282 | 4368 | 3.55× | 0.718 | 11136 | 1.39× | 0.858 |
| D_15 | {1,13,14,15} | 0.282 | 4368 | 3.55× | 0.718 | 11136 | 1.39× | 0.858 |
| D_0,1,7,10,13,14 | size-3 | 0.399 | 6188 | 2.51× | 0.601 | 9316 | 1.66× | 0.970 |
| D_2,5,6,17,18 | size-2 | 0.553 | 8568 | 1.81× | 0.447 | 6936 | 2.24× | **0.992** |

Two **different** notions of "informative", and they point opposite ways:

- **Largest world-collapse on a negative**: D_8/D_9 — an early "nothing" over the whole NW pack
  rules out all five of {8,9,10,11,12} in one reading, surviving worlds 15,504 → **3,003**
  (5.16×). This is the single highest-leverage early prune and it is **at the entry** (CSNE is
  2.6 from NW). After D_8-neg the remaining 5 are uniform over the other 15 (every non-NW
  marginal becomes exactly 5/15 = 0.333 — verified via `env.marginals`).
- **Most Shannon bits**: the balanced size-2 detectors (D_2,5,6,17,18) at 0.992 bits, because
  their positive/negative split is near 50/50. D_8 carries fewer bits (0.709) precisely because
  its positive is near-certain (0.806).

**For shrinking the search/belief, the negative-collapse ordering is the relevant one; for
resolving *which* treasure, the balanced detectors do more.** A good early policy fires D_8
first (cheap at entry, biggest collapse), not the highest-entropy detector.

### 2.2 Multi-step sensing-chain collapse (expected surviving worlds)

Filter: partition worlds by joint outcome pattern over a detector set; E[|belief|]=Σ(n²)/15504.

| chain | E[surviving worlds] | prune | #outcome-patterns |
|---|---|---|---|
| D_8 | 10661 | 1.5× | 2 |
| D_8, D_10 | 6834 | 2.3× | 3 |
| D_8, D_10, D_11 (full NW resolution) | 5434 | 2.9× | 4 |
| + mid {D_13, D_15} | 2414 | 6.4× | 12 |
| **one-per-cluster {D_8, D_13, D_0, D_7, D_17}** | **828** | **18.7×** | 31 |
| all 16 detectors | 121 | 128.4× | 319 |

A **5-detector "one probe per cluster" sweep** (≈ one reading per geographic neighbourhood)
already collapses the belief 18.7× to ~828 worlds — small enough that the residual problem is
near-clairvoyant. The marginal value of detectors beyond the first-per-cluster is steep but
diminishing, and bottoms out at the 120.7-world indistinguishability floor (§1.4).

---

## 3. Decomposition: the problem factors by cluster, coupled only by Σ = 5

### 3.1 Exact factorization (the keystone)

Filter: for an occupancy vector `(k_c)` over the 9 cells, count worlds with exactly `k_c`
present in each cell; compare to `∏_c C(size_c, k_c)`.

> Verified: occupancy (k_NW,k_mid,k_SE,k_N,k_S,k3,k4,k16,k19) = (2,1,1,1,0,0,0,0,0) →
> **270 worlds = ∏ C(size,k) = C(5,2)·C(3,1)·C(3,1)·C(3,1) = 10·3·3·3 = 270**, exact match.

**Given the occupancy vector, the worlds factorize exactly as the product of independent
uniform within-cluster subsets. The sole coupling between clusters is the global constraint
Σ k_c = 5.** Equivalently: the prior is a **multivariate hypergeometric** over the 9 cells, and
conditioned on the cell counts each cell is an independent uniform draw.

Per-cluster occupancy distributions (all verified == hypergeometric `C(c,k)C(20−c,5−k)/C(20,5)`):

- NW (size 5): {0:3003, 1:6825, 2:4550, 3:1050, 4:75, 5:1}
- mid / SE / N (size 3): {0:6188, 1:7140, 2:2040, 3:136}
- S (size 2): {0:8568, 1:6120, 2:816}
- each δ (size 1): {0:11628, 1:3876} (P(present)=0.25)

### 3.2 How weak the coupling is (so a decomposed policy loses little)

Filter: compare joint cluster-occupancy to the product of marginals.

- NW × mid, **before** any observation: max cell deviation from independence = **406.6 / 15504
  = 2.6%**.
- mid × N, **after** conditioning on NW empty: max deviation = **82.9 / 3003 = 2.8%**.

The clusters are mildly **negatively** correlated (a present treasure here is one fewer
elsewhere), but the entire coupling is carried by a **single integer** — the remaining budget
`5 − Σ(resolved counts)`. A policy that tracks the cluster-occupancy posterior (the
multivariate-hypergeometric belief) plus each cluster's local-belief pattern represents the
**exact** global belief with no approximation. There is no hidden cross-cluster entanglement to
lose.

### 3.3 Bridge caveat (honest)

The clean 9-cell partition is **not** perfectly detector-separable: **D_1** (covers {0,1,15})
and **D_15** (covers {1,13,14,15}) straddle the SE/mid boundary. Two clean handlings:

- **Merge** SE + mid into one supercluster {0,1,2,13,14,15}: its local belief-MDP has only
  **1,320 reachable local beliefs** (filter: BFS over local-subset-set beliefs under the
  6 in-cell detectors + collects; 2^6=64 latent subsets) — still tiny, still exactly solvable.
- **Keep them split** and forgo only the two cross readings of D_1/D_15 (2 of 16 detectors).
  Cheaper state, marginally weaker sensing.

All other clusters are bridge-free.

---

## 4. Exactly-solvable sub-problems (trusted anchors)

The global belief-MDP blows up (the project's settled "exact is infeasible"). But each
**cluster's local belief-MDP is microscopic.** Filter: BFS over local beliefs = frozensets of
in-cluster latent subsets, expanding by every in-cluster detector split and every in-cluster
collect (both outcomes), counting distinct reachable beliefs.

| cluster | latent subsets (2^size) | **reachable local beliefs** |
|---|---|---|
| NW {8,9,10,11,12} | 32 | **332** |
| SE {0,1,2} | 8 | 36 |
| N {5,6,7} | 8 | 36 |
| mid {13,14,15} | 8 | 31 |
| S {17,18} | 4 | 10 |
| (merged SE+mid {0,1,2,13,14,15}) | 64 | 1320 |
| each δ {3},{4},{16},{19} | 2 | 2 (known / unknown) |
| **sum (5 base clusters)** | — | **445** |

Compare: the **global** reachable belief-MDP is what blew up RAM. Per-cluster it is **445**
canonical local beliefs total (largest single cluster 332). Crossed with a handful of
locations (cluster waystone + each treasure point + entry/exit teleport) and the budget integer
`k ∈ {0..size}`, the per-cluster backward-induction table is on the order of **hundreds to low
thousands of states** — solvable **exactly and instantly** with classic value iteration. These
are the trusted anchors the sparse sampler could never be: an exact value for "given I am at
CSNE with budget k present somewhere in the NW pack, what is the optimal sense-and-collect rate
within NW before I leave?"

The **macro** layer (which cluster to visit, in what order, with which budget) has only **613
occupancy partitions** of 5 across the 9 cells (filter: count vectors with Σ=5, k_c≤size_c) —
also small. So the decomposed state — (macro occupancy posterior) × (current cluster's local
belief) × (location) — is bounded in the low thousands, where flat global belief-MDP
memoization was unbounded.

---

## 5. Ranked, quantified shortcut opportunities

Ranked by leverage (collapse × cheapness × how much it unblocks exact solving).

1. **Entry-NW negative collapse (5.16×, free at entry).** Probe D_8/D_9 from CSNE first.
   P(neg)=0.194 → 15,504 → 3,003 worlds, all of {8,9,10,11,12} ruled out, all other marginals
   snap to 0.333. CSNE is 2.6 units from the NW centroid (0.5 from τ_10), so this is the
   cheapest reading on the board and the largest single-negative collapse. **This is the
   maintainer's lead, confirmed and measured.**

2. **Exact per-cluster sub-solvers as anchors (445 total local beliefs).** Solve each cluster's
   local belief-MDP exactly by backward induction (NW: 332 beliefs; others 10–36). Use them as
   the trusted leaf value inside any search, replacing the determinization-optimistic playouts
   that wreck NMCS/ISMCTS (the agents traced their failure to determinization optimism + max
   over noisy estimates — an exact leaf removes that noise entirely).

3. **Occupancy-posterior macro-state (613 partitions, exact factorization).** Represent the
   global belief as the multivariate-hypergeometric over the 9 cells × per-cluster local
   pattern. Coupling between clusters ≤ 2.8% deviation from independence and is fully captured
   by the single integer "remaining budget". This is the representation that makes the whole
   problem fit in memory where the flat belief did not.

4. **One-per-cluster sensing sweep (18.7× collapse, 5 detectors).** {D_8, D_13, D_0, D_7, D_17}
   → E[828] worlds. A near-clairvoyant belief from one cheap reading per neighbourhood; the
   residual is small enough for a final exact route plan.

5. **Teleport-chained route decomposition (3 regions).** CSNE → {NW, N}; CSCE → {mid, SE, S,
   δ16, δ19}; τ_4 → {δ3, δ4}. Solve each region's sense-collect-exit sub-policy exactly, then
   chain the regions as a max-mean-ratio cycle over the three waystones. τ_4 serves only {3,4}
   and is the most distant exit from everything else — it is **dominated** for any run whose
   present set does not include 3 or 4 (and even then competes with collecting 3/4 on the way
   to CSCE). Treat τ_4 as a special-case detour, not a routing peer.

6. **Pair-breaking via collect (the indistinguishability floor).** Detectors alone bottom out at
   120.7 expected worlds because {8,9},{11,12},{13,14},{17,18} and {3,4,16,19} are
   detector-blind. When the optimal plan needs to know *which* of a pair is present (e.g. to
   pick the nearer of 8 vs 9), it must spend a collect — this is a real, quantified cost the
   policy should budget rather than expecting detectors to resolve it.

---

## 6. Recommended decomposition / hierarchy

A two-level policy that exploits §3's exact factorization:

**Macro level (cheap, ~613 occupancy partitions).** Maintain the multivariate-hypergeometric
posterior over the 9 cells. State = (per-cell occupancy posterior, current location/region,
remaining budget). Decide **which cluster to probe/visit next** and **when to bank-and-exit**
(the early-exit option the consult flagged as the strongest adaptive lever, since it is
contingent on realized cluster occupancy and no fixed prefix can encode it).

**Micro level (exact, ≤332 local beliefs/cluster).** On entering a cluster, run its
**pre-solved exact** local belief-MDP: optimal interleaving of in-cluster detector reads and
collects given the macro-supplied budget posterior, returning the cluster's value and the
posterior update to feed back to the macro layer.

**Coupling.** The only information passed up from micro to macro is the resolved count `k_c` for
the cluster (and which specific treasures, to break pairs). Because §3.1 factorizes exactly, the
macro posterior update is a clean multivariate-hypergeometric conditioning — no approximation.

**Why this can reach the +70% where flat search could not.** The agents' search methods failed
on **depth** (the VoI is behind multi-step chains) and on **determinization optimism** (max over
noisy sampled-world estimates → over-collection/over-travel). The decomposition removes both:
the deep chain is *inside* a cluster whose belief-MDP is solved **exactly** (no sampling noise,
no max-bias), and the macro layer's branching is over ~9 clusters, not the 36-way flat root.
The early-exit and skip-absent decisions that the clairvoyant exploits to bank E[R]=4.55 in
E[T]≈31 (vs static's E[R]=4 in E[T]=47) become first-class moves at the macro layer.

### Build order (suggested, all bounded)

1. Exact per-cluster local belief-MDP solver (NW first — 332 beliefs; validate against a
   brute-force over its 32 latent subsets). This alone gives a trusted leaf for existing search.
2. Multivariate-hypergeometric macro belief + occupancy-conditioned cluster values.
3. Macro routing over the 3 teleport regions with the early-exit option; τ_4 as a special case.
4. Measure the induced policy's **unbiased rate** (`env.dinkelbach_rate`) at large N against the
   0.0855 floor / 0.1454 ceiling — the honest metric the project already settled on.

---

## 7. What was computed (reproducibility)

All via `taskset -c 3 timeout … /home/bork/w/vdc/venvs/generic/bin/python`, reading the real
instance through `env.py`; no solver, no global belief enumeration. Each table cites its filter
inline. The load-bearing checks: every single-detector #neg equals the hypergeometric
`C(20−c,5)`; the occupancy factorization `#worlds == ∏ C(size,k)` holds exactly; per-cluster
reachable-belief BFS terminates in hundreds of states; the full-detector partition leaves 319
classes / E[120.7] worlds with the residue exactly the indistinguishability pairs of §1.4.
