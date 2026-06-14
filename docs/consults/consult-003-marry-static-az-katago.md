# Consult 003 — marrying the static analysis into AZ, KataGo-style auxiliary targets, and whether pushing past the ~0.10 plateau is worth it

> A design consult (analysis + ranked proposals, NOT implementation). The organizing
> insight is `docs/design/dual-bound.md`: **value calibration is a shared lever** — a
> calibrated belief state-value both sharpens the provable ceiling (drives the dual gap
> toward zero) and improves the policy. Everything below is scored against that finding.
> The full commission prompt is reproduced verbatim in Appendix A, per the project's
> consult-record discipline (`consult-001-prompt.md` is the format reference). KataGo
> facts are from David Wu, "Accelerating Self-Play Learning in Go" (arXiv:1902.10565) and
> `lightvector/KataGo/docs/KataGoMethods.md`, fetched 2026-06-14 (Appendix B).

---

## 1. Orienting synthesis — the headroom picture and the central thread

### 1.1 The numbers, assembled in one place

| quantity | rate | % of the +70% VoI gap | what it is |
|---|---|---|---|
| realizable static floor | 0.0855 | 0% | best fixed route, no sensing |
| exact decomp (myopic macro h=1) | 0.0941 | +14% | structure-exploiting; FIRST above the floor |
| best AZ to date (TD0.6 + unc feats, 241-d) | 0.0973–0.0978 | +20% | FIRST past decomp; N=400 confirm 0.0973 |
| **loose clairvoyant ceiling** | **0.1454** | **+100%** | free perfect info — pays nothing to sense |
| **dual-bound tightening (sub-instance demo, V̂=V\*)** | **λ̄ → ρ\*** | — | strong duality: the gap is *closable in principle* |

The plateau the question names ("~0.10 long-run rate") sits at **+20% of the +70%** — AZ
banks ER≈3.3 of the present 5 (decomp's figure; AZ is similar at E[T]≈43). The clairvoyant
banks 4.55/5 in E[T]≈31.3. The naive read is "+80% of the headroom is still on the table."
**The dual bound says that read is wrong, and that is the single most important input to
this consult.**

### 1.2 What the dual bound actually changes about "headroom"

The clairvoyant 0.1454 is a *valid* upper bound but a **loose** one, for one structural
reason (`dual-bound.md` §1): it hands the solver the true present-set for free and the
solver never pays for a single sensing read — it routes straight to the present five. The
entire difficulty of chocofarm is the **costly contingent sensing chains** a real policy
must run to *discover* which five are present (`static-analysis-faces.md` §2: resolving the
NW cluster alone is a **10-read chain**, not one probe). The dual construction *charges for
that free information* with a martingale penalty and lands strictly below 0.1454, toward the
achievable frontier (~0.094–0.10).

Two facts from `dual-bound.md` are load-bearing for the verdict:

- **The exact state-value achieves strong duality (gap → 0).** On the NW sub-instance,
  handing V̂ = V\* (the true fixed-λ optimal belief value) as the penalty drove the bound to
  **λ̄ = 0.0957 ≈ ρ\*_sub = 0.0956** (§5 row (ii)/(iii)). So a *calibrated state-value* both
  (a) sharpens the provable ceiling and (b) — being the optimal value function — *is* the
  optimal policy's value. **Calibration is one lever with two payoffs.**

- **The decomp *decision*-value LOOSENS the bound.** Its one-step martingale increments are
  large and exploitable (measured `E[V̂′|F] − V̂ ≈ −0.39` for a face read at the NW belief);
  as a penalty it makes the inner clairvoyant *game* the penalty, pushing λ̄ ≈ **0.15** —
  looser than even the sub-instance clairvoyant 0.099 (§6 caveat). The decomp value steers a
  good policy but is **not a self-consistent state-value**.

The honest reframing of "how much headroom is left," then, is: **we do not have a tight
upper bound on ρ\* on the full instance yet** — the tight bound is gated behind exactly the
calibrated V̂ that AZ is trying to learn. The real headroom above AZ's 0.097 is bounded
above by 0.1454 but is **probably much smaller**, because 0.1454 pays nothing for the
10-read chains. The qualitative finding from `decomp-rate.md` §"Where the remaining gap to
the ceiling lives" sharpens this: *the gap is mostly REWARD, not time* — the decomp (and AZ)
leave present treasures uncollected in clusters not worth their face-read chain at the
rate-optimal λ. That is the genuine VoI frontier, and it is genuinely expensive to cross.

### 1.3 The central mechanistic thread

Three findings interlock and point at one diagnosis:

1. **The AZ value head is a geometry-blind progress counter** (`az-residual.md`,
   `az-edecide.md`, `az-parallel-exp.md` feature-response): it tracks how-far-along, not
   belief geometry. Extra value capacity (the residual block) did not move the rate. Search
   depth is the lever, not capacity.

2. **The dual bound explains WHY that matters**: an uncalibrated value cannot close the gap.
   A progress-counter value has large, exploitable Bellman residuals — exactly the property
   that made the decomp decision-value *loosen* the dual bound. The same defect that keeps
   the bound loose keeps the policy below optimal. **These are the same defect.**

3. **The one win that broke past decomp was a variance reduction on the value target, not a
   capacity or feature win.** The TD(λ=0.6) blend (`az-parallel-exp.md` Part B) made the
   value "far more predictive (R² 0.62 → 0.90) and modestly de-blinded it to geometry/
   sensing" — and that, alone, bought +5 VoI points past decomp. The `unc` belief feature
   (Part C) free-rode (importance 0.002, near-inert in the value head). Feature-response on
   the winning net: `treasure/dist` rose off zero (0.002 → 0.012), `detector/dist`
   0.003 → 0.028, `informative` 0.027 → 0.060. **The value got better by getting more
   self-consistent (lower-variance target → lower Bellman residual), and as it did, it
   started reading geometry it had been ignoring.**

So the spine of this consult is: **the lever is value calibration — driving the Bellman
residual of the belief state-value toward zero — and the credit-assignment density on the
contingent sensing chains where the VoI lives.** Everything KataGo did that helped in Go
maps onto exactly these two axes (denser/lower-variance training signal; better-calibrated
representation), and everything in the static analysis is a source of *trusted, calibrated*
value and *structured credit* that the net currently has to discover from scratch.

This is why the question "is it even worth pushing" is sharp rather than rhetorical: the
real headroom is small *and* the only lever that both certifies it and captures it is the
one thing the net is currently bad at. The proposals below are ranked by how directly they
attack calibration, scoped by the (small, REWARD-frontier) headroom the dual bound implies.

---

## 2. Ranked proposals — the AZ-side bets

Each proposal: (a) the idea; (b) the mechanism in the language of the findings; (c) the
KataGo move it mirrors; (d) implementability against the current code (file/seam, effort);
(e) expected impact, honestly scoped; (f) risks / how it could fail or mislead.

The ranking is by expected value *given the dual-bound headroom* — i.e. weighted toward the
bets that calibrate the state-value, because that is both the policy lever and the
bound-tightening lever, and against the bets that only add capacity or features (which the
record shows did not move the rate).

---

### Bet 1 (highest EV) — Bellman-consistency auxiliary target: predict the *next-belief value* and minimize the one-step residual

**(a) The idea.** Add a per-step auxiliary loss that directly penalizes the value head's
one-step Bellman inconsistency. At each executed decision the search already computes the
realized step reward `r_j − λ·dt_j` and reaches a successor belief whose net value
`V̂(s_{j+1})` is available (it was evaluated as a child in the tree). Add a loss term that
drives `V̂(s_j)` toward `(r_j − λ·dt_j) + E_{outcome}[V̂(s_{j+1})]`, where the expectation is
over the *belief-conditioned* observation outcome (the same `p⁺ V̂(b⁺) + (1−p⁺) V̂(b⁻)` the
dual penalty uses, computable from the belief without a determinization). This is a TD(0)
consistency target *as an explicit auxiliary head/loss*, not merely as a variance-reduction
blend of the MC return.

**(b) Mechanism.** This is the **most direct possible attack on the diagnosis**. The dual
bound is loose exactly when the value's martingale increments `V̂(s_{j+1}) − E[V̂(s_{j+1})|F_j]`
are large (the measured −0.39 on the decomp value). The Bellman-consistency loss *minimizes
those increments by construction* — it trains the net to be a self-consistent state-value,
which is (i) the property that makes the policy optimal and (ii) the property that drives the
dual gap to zero. The progress-counter pathology is precisely a value with low loss on the
*level* (where-am-I) but high residual on the *step* (what-did-this-action-buy); a level-only
MC target rewards the former and is indifferent to the latter, which is why the value
collapsed to a progress counter. Adding the residual to the loss makes the step the thing
being fit. Note the belief-conditioned expectation is **cheap and exact** here in a way it
never is in Go: the outcome distribution of a sense/collect is a closed-form function of the
belief (`p⁺ = (bw & cover != 0).mean()`), so the consistency target needs no extra rollouts
and no determinization — it is computed from the belief the search already holds.

**(c) KataGo mirror.** Closest to KataGo's **short-term value targets** (the three
exponentially-averaged-future-MCTS-value heads at ~6/16/50-turn horizons): KataGo found
"neural nets train slightly faster and achieve better value/score loss on the main head" by
adding lower-variance auxiliary value targets at controlled horizons. The 1-step Bellman
residual is the shortest-horizon member of that family, and the belief-MDP gives us its
expectation in closed form (Go must Monte-Carlo it). The general principle — *a denser,
lower-variance value signal at a controlled horizon calibrates the main value head* — is the
one KataGo leaned on hardest.

**(d) Implementability.** Seam: `chocofarm/az/value_target.py` (a new target alongside
`blended_returns_to_go`) plus the loss assembly in `chocofarm/az/exit_loop.py` and the head
in `chocofarm/az/mlp.py`. The successor-value expectation needs the per-outcome successor
beliefs and `p⁺`; the search already filters to successor beliefs in `gumbel_search.py`
(`_simulate_root_action`, `_descend`) and the belief-derived `p_pos` is already a feature
(`features.py`). Two implementation shapes: (i) cheapest — add a TD(0)-residual *loss term*
on the existing value head using the search-cached successor values (no new head); (ii)
fuller — a dedicated auxiliary "next-belief value" head trained to the same target so the
gradient is isolated. Start with (i). Effort: **moderate** (a new target function + loss
wiring + one ablation axis), comparable to the Part B TD(λ) build that already landed.

**(e) Expected impact, scoped.** This is the bet most aligned with the lever the dual bound
identifies, so it has the **best expected value of the AZ-side bets** — but scoped honestly:
the headroom above 0.097 is bounded by the loose 0.1454 and is *probably* small and
REWARD-frontier-bound. A realistic target is **closing a meaningful slice of the
0.097 → (true ρ\*) gap**, where true ρ\* is unknown but almost certainly well below 0.1454.
The dual-bound machinery gives a *measurable* read on success that no rate number can: train
the value with this loss, freeze it, feed it as V̂_AZ into `chocofarm/bounds/eval_bound.py`
(the §2.4(3) generator the dual-bound doc already specs), and watch whether λ̄ drops below
0.1454 on the full instance. **A value that tightens the bound is, by the strong-duality
theorem, a better policy value — the two move together.** This is the single cleanest
success metric in the whole program.

**(f) Risks / how it could mislead.** (1) **Bootstrap optimism** — the same risk Part B
flagged: a TD/Bellman target bootstraps off the net's own (initially bad) successor values,
which can inflate on under-sampled deep-sensing beliefs and reintroduce over-collection
(rising E[T]). Mitigation: the belief-conditioned expectation is *exact* (no determinization
optimism, unlike the Go case), so the only optimism is the net's own extrapolation; keep the
`stop_grad` on the successor value (KataGo does this for its error heads) and keep the MC
anchor in the blend (ℓ=1 escape hatch). (2) **It may just re-learn the progress counter with
lower residual** — a value can be Bellman-consistent and still geometry-coarse if the
features can't express geometry; this is why Bet 1 pairs with the static-structure features
(§3). (3) The cleanest read (does λ̄ drop) requires the *separable inner solve* in the dual
code to be built (`dual-bound.md` §4.4 — currently specified, not implemented), so the
full-instance bound read is itself gated on that build.

---

### Bet 2 — per-treasure presence/marginal auxiliary prediction (the belief's "ownership map")

**(a) The idea.** Add an auxiliary head that predicts, per treasure, the *realized* presence
bit `1[i ∈ present]` of the episode's true world (a 20-way independent Bernoulli head),
trained against the world that was actually rolled. Optionally also predict the
*end-of-episode resolved marginal*. This is the belief-MDP's direct analog of KataGo's
ownership map: "which of the 20 cells will turn out to be owned (present)."

**(b) Mechanism.** Two effects, both on the calibration/credit axis. **First, localized
credit assignment** (KataGo's exact stated mechanism for ownership): with only the scalar
λ-return, the value head "can only guess" which part of the belief caused a bad return; a
per-treasure presence target gives "direct feedback, with large errors and gradients
localized to the mispredicted [treasure]." The value's geometry-blindness is partly a
credit-assignment failure — it never gets a gradient that says *this treasure's
present/absent status is what your return hinged on*. **Second, it forces the trunk to encode
the belief's joint structure** rather than just its marginal level. The progress-counter
pathology is a trunk that compresses the belief to a scalar "how resolved am I"; a
per-treasure presence target resists that compression because the 20 outputs cannot be
reconstructed from a scalar. This is the representational pressure the `unc` feature was
*supposed* to supply on the input side but didn't (it free-rode at 0.002) — supplying it as
an *output target* is far stronger, because it shapes the trunk via gradient rather than
hoping the trunk reads an input.

**(c) KataGo mirror.** Directly the **ownership auxiliary target** — KataGo's single
highest-impact auxiliary (removing ownership+score = 1.65× slowdown, the largest individual
effect in the ablation table). The mechanism KataGo names (localized gradients for spatial
credit assignment) transfers almost verbatim: treasures are the "points," presence is
"ownership."

**(d) Implementability.** Seam: a new head in `mlp.py` (20 sigmoid outputs); target =
`[(world >> i) & 1 for i in range(N)]`, available for free in `dataset.py` /
`exit_loop.generate_episode` (the rolled world is known per episode). Loss: BCE, low weight
(KataGo uses `wo = 1.5/b²`, modest; mirror with a small weight ~0.1–0.3). The head is
inference-time-discardable (KataGo never uses ownership at play), so it costs nothing in the
search hot path. Effort: **low-moderate** — a head + a target column + a loss term; no
search changes.

**(e) Expected impact, scoped.** KataGo's ownership was its biggest auxiliary win, and this
is the cleanest structural analog, so on priors this is a strong second bet — *but* the
chocofarm record warns against over-reading the analogy. The presence target is what the
marginal feature *already encodes on-distribution* (F6: marginals are near-sufficient), so
its value is not "new information the net lacks" but "denser gradient + representational
pressure against the progress-counter collapse." Expect a **sample-efficiency / calibration**
gain (faster, more-geometry-aware value) more than a raw-information gain. Scoped by the
headroom: a calibration gain that compounds with Bet 1; not a silver bullet on its own.

**(f) Risks / how it could mislead.** (1) The target is *the true world's bit*, which under
uncertainty is a high-variance label (the belief is genuinely split) — the head learns the
*posterior marginal* in expectation (BCE's minimizer is the conditional mean = the marginal),
so this may converge to predicting `marg[i]` — which is already an input feature. If so, the
gradient pressure on the trunk is the real (and only) benefit; the head's *outputs* are
redundant. That is fine (KataGo's ownership outputs are also discarded) but means the win is
purely representational, and could be small if the trunk was already adequate. (2) Could
mildly *mislead* the headline if someone reads the ownership head's accuracy as "the policy
is good" — it isn't a policy metric. Keep the rate + the dual-bound λ̄ as the verdict.

---

### Bet 3 — per-cluster occupancy-k auxiliary prediction (the factorization as a training target)

**(a) The idea.** Add an auxiliary head predicting, for each of the ~4 sense-clusters + the
δ-pool, the *occupancy* `k_c` = how many of the 5 present treasures lie in that cell — a
small categorical (k_c ∈ {0..size_c}) per cell. Target = the realized occupancy of the
episode's world, projected onto the cells.

**(b) Mechanism.** This injects the **occupancy factorization** (`static-analysis-faces.md`
§3, the DET-IND keystone: #worlds = ∏ C(size,k), coupled only by Σk=5) as a learning signal.
The factorization is *the* structure that makes the problem tractable for the decomp solver,
and it is precisely the geometry the value head is blind to: the rate-relevant question
"is this cluster worth its 10-read chain at the current λ?" is a question about the cluster's
*occupancy posterior*, not about individual marginals. A per-cell occupancy head forces the
trunk to encode the cell-level joint (how many are here) rather than the flat marginal
vector — the abstraction the macro layer reasons over. This is a *denser, more structured*
credit signal than per-treasure presence (Bet 2): it tells the net not just "treasure 8 was
present" but "the NW cluster had 2 of its 5 present" — the quantity that gates the
enter/skip/exit decision where the REWARD-frontier gap lives.

**(c) KataGo mirror.** Between KataGo's **ownership** (spatial credit) and its
**score-distribution** target (predict a *distribution* over an aggregate quantity, not a
scalar — KataGo found the full distribution enables uncertainty-aware utility). Occupancy-k
is an aggregate-over-a-region distribution, exactly the score-distribution register applied
to "how many present in this region."

**(d) Implementability.** Seam: a head in `mlp.py` (one small softmax per cell); the cell
partition is `discover_clusters(env)` from `decomp.py` (reuse it directly — the clusters are
already computed there and in `analyzer.py`); target = project the rolled world onto the
cells and count, exactly `_live_occupancy_posterior`'s projection logic (already written in
`decomp.py`). Loss: per-cell cross-entropy, low weight. Effort: **low-moderate**, and it
reuses the cluster machinery that already exists — the cleanest static-marriage on the
target side (see §3).

**(e) Expected impact, scoped.** Plausibly the **highest-leverage of the three auxiliary
heads** for *this* problem specifically, because occupancy-k is the exact latent the
factorization and the decomp's macro layer use, and the REWARD-frontier gap is an
occupancy-level decision (enter/skip a cluster). But it is the least-validated by analogy
(KataGo had no exact factorization). Scoped: a representational + credit win on the decision
that matters most; pair with Bet 1.

**(f) Risks.** (1) Like Bet 2, the BCE/CE minimizer is the *posterior* occupancy, which the
net could in principle compute from marginals — so the win is again representational pressure,
not new information. (2) The δ-pool occupancy is genuinely hard (δ are observe==collect, no
faces) — the head will be near-prior there; that is honest and not a defect. (3) Modest risk
of head/target plumbing bugs in the projection (mitigated by reusing the tested
`_live_occupancy_posterior` projection).

---

### Bet 4 — short-term-value-error head → uncertainty-weighted search (the deep-sensing-chain credit fix)

**(a) The idea.** Train an auxiliary head to predict the *squared error* of the value head's
own short-term estimate (`(stop_grad(V̂) − realized_short_return)²`), then **downweight
search playouts proportionally to predicted uncertainty** and/or scale cPUCT by predicted
utility variance — KataGo's uncertainty-weighting + dynamic-variance cPUCT, applied to the
Gumbel search.

**(b) Mechanism.** This attacks the *other* axis of the central thread: **VoI is gated behind
depth** (F3; the 10-read NW chain), and the search budget (m=12, n=48) is the established
lever (search depth moved the rate; capacity didn't). KataGo's uncertainty weighting makes
search intensity track position volatility: "calm positions get equal weight; volatile
tactical positions automatically get more playouts." In chocofarm the "volatile" states are
exactly the deep-sensing beliefs mid-chain, where the value is most uncertain and the VoI
lives — the states `az-edecide.md` named as *under-calibrated because the non-exploring
teacher never visited them*. Spending more of the fixed budget on the uncertain
deep-sensing branches is a direct way to get search depth where it pays, without raising the
total budget (which the contended 4-vCPU host can't afford).

**(c) KataGo mirror.** **Uncertainty-Weighted MCTS Playouts** + **Dynamic Variance-Scaled
cPUCT** (KataGoMethods §2, §5; reported ~75 Elo combined). Mechanism transfers cleanly: the
net predicts its own error, search trusts confident estimates more.

**(d) Implementability.** Seam: an error head in `mlp.py` (predict squared short-term value
error, `stop_grad` on the value); the search reweighting in `gumbel_search.py` (`_visit` /
`_descend` accumulation of W/N, and the `_puct_select` cPUCT scaling). This is the **most
invasive** of the bets — it touches the search hot path and the Sequential-Halving
accounting (which assumes integer playout counts; fractional weights need care). Effort:
**moderate-high**, with real risk of subtle bugs in the SH/PUCT math (the code is carefully
faithful to Danihelka et al.; reweighting perturbs that).

**(e) Expected impact, scoped.** Targets the depth lever directly, which has the best track
record (search depth is what moved the rate). But it depends on Bet 1/2/3 first giving a
value calibrated enough that its *error prediction* is meaningful — an uncertainty head on a
progress-counter value predicts the error of a counter, which is not useful. So this is a
**second-wave bet**, contingent on the value first becoming calibrated. Scoped: a search-
efficiency multiplier on whatever calibration the value heads achieve.

**(f) Risks.** (1) Fractional playouts break the clean Sequential-Halving budget accounting —
implementation hazard. (2) On a host that already can't reach 4× parallelism
(`az-parallel-exp.md`: ~1.9× on the contended vCPUs), a more expensive per-node search may
not be affordable. (3) The uncertainty head needs the short-term value head (Bet 1) to exist
first; sequencing matters.

---

### Bet 5 (lowest EV among AZ bets) — more value capacity / richer encoder

**(a) The idea.** Bigger trunk, a transformer/DeepSets-over-clauses encoder, more residual
blocks.

**(b) Mechanism / why it ranks last.** **The record already falsified this.** The residual
block (`az-residual.md`) added 131k params of value capacity and *did not move the rate* —
the value is a progress counter not because it lacks capacity but because the *target* (high-
variance MC) didn't ask it to be anything else, and the *features* (F6: marginals near-
sufficient) already carry the deduction. The design doc's own §2.3/§2.4 mark the
clause-DeepSets encoder as "held in reserve precisely because F6 predicts it buys little."
Adding capacity is the gold-plating the project warns against; it neither calibrates the
value (Bet 1) nor densifies credit (Bets 2/3) nor adds depth (Bet 4).

**(c–f).** KataGo mirror: **global pooling** is the *one* architectural change KataGo found
high-impact (1.60× slowdown when removed) — but it helped because Go's conv net is *locally
blind* and needs global context (winning/losing status, ko). Chocofarm's MLP over a flat
~241-d belief vector is *already global* (every feature sees the whole belief), so the
specific mechanism that made global pooling pay in Go **does not apply** — the chocofarm net
has no locality constraint to relieve. Implement only if an ablation shows the MLP under-
fitting *after* the target/credit bets, which the residual result suggests it won't. Expected
impact: low. Risk: spends the compute budget on the part the evidence says isn't the
bottleneck.

---

## 3. The static-analysis marriage, specifically

The static structure (`static-analysis-faces.md`, `decomp.py`) offers four distinct entry
points into AZ: **input features**, **auxiliary targets**, **architectural prior**, **search
prior**. Scored under the same rubric, and against the dual-bound lever.

### 3.1 As auxiliary targets — the strongest marriage (this is Bets 2 and 3)

The cleanest, highest-EV use of the static structure is **as training targets, not inputs**.
The reason is the central thread: the net's defect is a *trunk that compresses the belief to
a progress scalar*, and the fix is *gradient pressure that resists that compression*. The
static structure supplies exactly the right resisting targets:

- **Per-treasure presence** (Bet 2) = the ownership map; the factorization's atoms.
- **Per-cluster occupancy-k** (Bet 3) = the factorization's keystone latent (∏ C(size,k)),
  reused via `discover_clusters` + `_live_occupancy_posterior`'s projection (already coded,
  already exact). This is the **single most native** static-marriage: the occupancy vector is
  literally the macro layer's state, and it is the latent the REWARD-frontier decision turns
  on.
- **Time-to-resolution / expected-remaining-reward distribution** (a further auxiliary):
  predict the number of reads to resolve the live cluster (the chain length, which
  `static-analysis-faces.md` §2 quantifies: NW=10, SE+mid=10, N=5, S=3) and/or a distribution
  over remaining banked reward. These are KataGo's score-distribution register and directly
  encode the "is the chain worth it" question; medium priority, after Bets 2/3.

Mechanism, restated against the dual bound: an auxiliary target that the net can only predict
by encoding the cluster-occupancy joint is a target that *lowers the value's Bellman residual*
indirectly — a value head reading a trunk that knows occupancy can be self-consistent across
a sense step in a way a progress-counter trunk cannot. Targets and Bet 1's explicit
consistency loss compose.

### 3.2 As a *trusted, calibrated* value target — the deepest marriage (a new lever)

This is the proposal that most directly exploits the dual-bound finding and is **not yet in
the AZ program**: use the decomp's **exact micro values as a low-variance value label**, not
as the determinization-optimistic playout the project rightly fears.

`decomp.py` already computes, per cluster and occupancy, the *exact* `enter_value =
E[R|k] − λ·E[T|k]` (`MicroSolution.enter_value`) — an exact λ-value of the cluster sub-MDP,
with **zero sampling noise and zero determinization optimism** (it is backward induction over
the local semilattice). The `az-edecide.md` caveat explicitly deferred this: "Decomp does
expose an exact λ-value via its micro tables; blending it as a lower-variance label is a
deferred option, not exercised."

The dual bound reframes why this matters: the decomp *macro decision-value* loosens the bound
(large residuals), but the *micro per-cluster value is exact within the cluster* — it is a
genuine, calibrated state-value *for the in-cluster belief*. So:

- **Use the exact micro `enter_value` (composed through the live occupancy posterior) as a
  low-variance value label** blended with the honest MC return — the same TD(λ) blend
  machinery (`value_target.py`) that already won, but bootstrapping off an *exact* anchor
  instead of the noisy search root-value. This is the lowest-variance, least-optimistic value
  label available anywhere in the codebase. It directly attacks the calibration deficit, and
  it composes with Bet 1 (the micro value is the closest thing to V\* we can compute cheaply
  on a cluster).
- **Honest caveat**: the composition across clusters (the macro layer) is where the decomp
  value loses self-consistency (the −0.39 residual is a *macro/face-read* increment, not an
  in-cluster one). So blend the micro value **only on in-cluster beliefs** (where it is
  exact), and fall back to MC / Bet-1-consistency on the macro/boundary beliefs. This is a
  principled split: trust the exact value where it is exact, learn the rest.

Effort: **moderate** — `decomp.py` already exposes `enter_value` and the live occupancy
posterior; the wiring is a new label source in `value_target.py` / `dataset.py`, gated on
"is this an in-cluster belief." Expected impact: this is the **second-best AZ-side bet after
Bet 1**, and arguably part of the same bet — it is the cheapest route to a calibrated value
on the cluster sub-problems, which is exactly where the dual bound's strong-duality demo
(V̂=V\* → tight) says calibration pays.

### 3.3 As input features — low priority (the record says so)

The marginals already carry the deduction (F6: near-sufficient on-distribution); the `unc`
belief feature (Part C) free-rode at importance 0.002. Adding cluster-count input features
(Σmarg per cluster, cluster entropy) is the design doc's own §2.3 "ablation add-on, not
baseline," and the `unc` result is the empirical confirmation that *input* enrichments are
weak here. **Do not lead with input features.** The one input feature worth trying is a
per-cluster occupancy *posterior* summary (E[k_c] and Var[k_c] per cell), because it is the
macro-relevant statistic — but as §3.1 argues, the same information is far more powerful as a
*target* than as an input. Low priority; only if an ablation shows a residual gap after the
target bets.

### 3.4 As an architectural prior — moderate, speculative

A trunk that respects the cluster partition (per-cluster sub-encoders pooled into a macro
representation, mirroring micro/macro) is a defensible inductive bias — it bakes the
factorization into the architecture rather than hoping the trunk learns it. But this is a
larger build than the target bets, the global-pooling lesson (§2 Bet 5) warns that
architecture changes pay only when the net has a *structural blindness* to relieve, and the
flat MLP is already global. Mechanism is plausible (cluster-structured trunk ⇒ occupancy is
easier to encode ⇒ lower residual), but the EV is below the target bets because the targets
(§3.1) get most of the benefit at a fraction of the build. **Defer to a second wave;** prefer
targets first.

### 3.5 As a search prior — moderate, and it composes with the depth lever

Two concrete uses:

- **Decomp policy as the warm-start / prior teacher.** The ExIt loop's first iteration is
  bootstrapped from a teacher; `az-edecide.md` already used decomp (stronger + faster than
  ISMCTS). The deeper use: seed the *policy head's prior* with the decomp's per-state action
  (the micro π\* is exact), so the Gumbel root sampling concentrates the tiny m=12 budget on
  the sense-chain actions the decomp knows are good — directly serving H-amortize (reach the
  deep chains) where the budget is scarce.
- **Decomp value as the search *leaf* on in-cluster beliefs** — the exact micro value as the
  leaf evaluation when the search descends into a cluster, replacing the net value there
  (the §2 trusted-anchor idea from `static-shortcuts.md` §5, but scoped to where the micro is
  exact). This removes leaf noise on exactly the deep-sensing descents, which is where F4's
  determinization optimism and the value's miscalibration both bite. Composes with Bet 4
  (spend search where the value is *not* exact; trust the exact leaf where it is).

Effort: low-moderate (the decomp policy/value are already callable). Expected impact:
moderate, and it is the cheapest way to get the depth lever and the calibration lever to
cooperate. The risk is the anchor-geometry approximation in the macro (`decomp-rate.md`
caveats) — but those are *decision-only* and don't bias a value used as a leaf estimate.

---

## 4. Honest verdict — is envelope-pushing worth it, and which 1–3 bets?

### 4.1 The honest headroom read

The clairvoyant 0.1454 is a mirage as a target: it pays nothing for the 10-read sensing
chains that *are* the problem. The dual bound's whole point is that the real ceiling is much
lower, and the tight ceiling is gated behind the same calibrated state-value the policy
needs. We **do not have a tight upper bound on the full instance yet** — so the most honest
statement of headroom is: *AZ at 0.097 is +20% of a +70% gap whose top 80% is largely
unreachable, and the reachable remainder is REWARD-frontier-bound (leaving present treasures
in clusters not worth their chain at the rate-optimal λ).* The remaining gap is
**structurally hard, not merely un-amortized.**

That said, the program is **not at a dead end**, for one specific reason: the one win that
broke past decomp (TD-blend value calibration, +5 VoI points) came from the *exact lever the
dual bound names*, and it came cheaply. That is direct evidence the lever has more to give
before it saturates — the value went R² 0.62 → 0.90 and started reading geometry, and the
proposals above push the *same* lever harder (explicit consistency loss; exact micro labels;
occupancy targets) rather than betting on capacity (falsified) or input features (free-rode).

### 4.2 The verdict

**Pushing is worth it, narrowly and on a specific basis: the goal should shift from "claw
back rate" to "calibrate the value, and let the calibrated value both tighten the provable
bound and incrementally improve the policy."** The rate gains will be modest (the headroom is
small); the *bound* gain is the under-appreciated payoff. A calibrated V̂_AZ fed into
`eval_bound.py` (§2.4(3)) that drives λ̄ below 0.1454 on the full instance would be a
genuinely new result — the first *tight-ish* certified ceiling on ρ\* — and by strong duality
it is the same artifact as a better policy. That reframing is what makes the effort worth it:
the consult's honest finding is that **the value-calibration work has a guaranteed payoff
(the bound) even if the rate gain is small**, which de-risks the bet in a way a pure rate-
chase is not.

If the goal is *only* the rate and the bound is not valued, the honest answer tilts toward
**"AZ is near a practical ceiling and the remaining gap is structurally hard"** — the +5 VoI
points past decomp is real but small, the host can't afford much more search, and the REWARD-
frontier gap is the genuinely expensive part the clairvoyant bound flatters.

### 4.3 The 1–3 bets with the best expected value

1. **Bet 1 — Bellman-consistency value target / loss** (§2 Bet 1). The most direct attack on
   the lever; success is *measurable via the dual bound* (does λ̄ drop), not just the noisy
   rate. Moderate effort, reuses `value_target.py`. **Do this first.**

2. **The exact-micro-value calibrated label** (§3.2). The cheapest route to a calibrated
   value on the cluster sub-problems — exactly where the strong-duality demo (V̂=V\* → tight)
   says calibration pays. Reuses `decomp.py`'s `enter_value` + the live occupancy posterior;
   blend only on in-cluster beliefs. Composes with Bet 1 (it is the closest cheap proxy for
   the V\* that closed the sub-instance gap). **Do this with Bet 1.**

3. **Per-cluster occupancy-k auxiliary target** (§2 Bet 3 / §3.1). The factorization's
   keystone latent as gradient pressure on the trunk, on the exact decision (enter/skip a
   cluster) where the REWARD-frontier gap lives. Reuses the cluster machinery. **Do this if
   1+2 show the value calibrating but the trunk still geometry-coarse** (feature-response is
   the diagnostic).

Deprioritize: more capacity / encoders (Bet 5 — falsified by the residual result); input-
feature enrichments (the `unc` free-ride is the warning); architectural priors and
uncertainty-weighted search (second wave — both depend on the value first being calibrated,
and the host can't afford more search until parallelism is uncontended).

The unifying thread, one more time: **every recommended bet drives the belief state-value's
Bellman residual toward zero. That is the lever that improves the policy and the lever that
tightens the certified ceiling — they are the same lever, which is the dual bound's central
gift and the reason this work is worth doing even where the rate headroom is thin.**

---

## Appendix A — commission prompt (verbatim)

> [Reproduced verbatim per the consult-record discipline; see `consult-001-prompt.md`.]

You are working on **chocofarm** (`/home/bork/w/vdc/chocobo`, github KodBena/chocofarm) — an
Operations Research exercise (FFXIII gil-farming modeled as adaptive stochastic orienteering /
a belief-MDP), NOT a game tool. Success = OR/solver quality and honest, mechanistic analysis;
the maintainer prefers "this probably won't pay, because X" over optimistic listing, and
provable/mechanistic claims over hand-waving.

This is a **design consult** — analysis + ranked proposals, NOT implementation.

THE QUESTION: How do we push the AlphaZero agent past its ~0.10 long-run-rate plateau on this
belief-MDP, via (1) marrying the **static-analysis structure** (the co-coverage cluster
decomposition, the occupancy factorization, the micro/macro solve) into the AZ agent; (2)
**KataGo-style** richer input features and **auxiliary training targets**; and (3) any other
envelope-pushing direction you find — all **scoped by the dual-bound finding** about where the
real headroom actually is. And, honestly: **is it even worth pushing**, given that headroom?

SET-UP — create your own worktree (it contains every doc and all code you need, including the
just-written dual-bound design):
  git -C /home/bork/w/vdc/chocobo worktree add /home/bork/w/vdc/chocobo-consult -b
  docs/consult-marry-static-az feat/az-dual-bound
Work in `/home/bork/w/vdc/chocobo-consult`. This is read-and-reason; you write ONE doc and
commit it. You generally need not run code, but may (pinned `taskset -c 3` under `timeout`) if
a small check sharpens a claim — a live AZ job holds cores 0-2.

READ END TO END before citing any of them (load-bearing — do not act on grep fragments; read
each fully):
  - `docs/design/dual-bound.md` — THE organizing insight, just established and independently
    verified: the loose clairvoyant ceiling 0.1454 massively overstates real headroom because
    it pays nothing to sense; tightening that ceiling — and, equivalently, closing the gap to
    the true optimum — requires a **calibrated belief STATE-value**. The decomp *decision*-value
    LOOSENS the bound; the exact state-value V\* achieves strong duality (gap → 0). So **value
    calibration is a shared lever**: it both sharpens the provable ceiling and improves the
    policy. Make this the spine of the consult.
  - `docs/results/voi-ceiling-2026-06-13.md` — static floor 0.0855, clairvoyant ceiling 0.1454
    (+70% / 100%-VoI).
  - `docs/results/decomp-rate.md` — exact decomposition 0.094 (~14% of the +70%); read the
    section "Where the remaining gap to the ceiling lives" (the gap is mostly REWARD — present
    treasures left uncollected in clusters not worth their face-read chain at the rate-optimal λ).
  - `docs/design/alphazero-surrogate-design.md` — the AZ surrogate design (value/policy over the
    belief; the Dinkelbach λ-penalty; the feature set).
  - `docs/results/az-residual.md`, `docs/results/az-edecide.md`, `docs/results/az-parallel-exp.md`
    — the AZ frontier findings. THE CRITICAL ONE: the value head behaves as a **geometry-blind
    progress counter** — it tracks how-far-along, not belief geometry; extra value capacity (the
    residual block) did NOT move the rate; search depth is the lever. The dual-bound finding
    explains WHY that matters (an uncalibrated value cannot close the gap).
  - `docs/design/static-analysis-faces.md`, `docs/design/static-shortcuts.md` — the static
    structure: co-coverage clusters, occupancy factorization (#worlds = ∏ C(size,k)), one-per-
    cluster sweeps, the macro cell partition.
  - AZ feature/target/search code: `chocofarm/az/features.py` (current inputs, dim 241),
    `chocofarm/az/value_target.py` (TD(λ) target), `chocofarm/az/feature_response.py` (the probe
    that found value=progress-counter), `chocofarm/az/gumbel_search.py` (the search).
  - `chocofarm/solvers/decomp.py` — the structure that could become features / targets / priors.
Use WebSearch/WebFetch on **KataGo's improvements** — David Wu, "Accelerating Self-Play Learning
in Go" (2019), and the KataGo methods notes: the **auxiliary targets** (ownership map, final-
score distribution, short-term value/score, policy-aux/opponent-reply), the **input-feature**
enrichments, **global pooling**. Extract WHAT made each help (sample efficiency, value
calibration, credit assignment) and HOW it maps onto this problem. If the network is
unavailable, use your own knowledge and say so.

THE ANALOGY TO DRAW (the heart of the consult): KataGo's leap came largely from auxiliary heads
that gave the net a richer, better-calibrated representation and a denser training signal
(ownership per-point, full score distribution, etc.). This problem's analogs include: per-
treasure presence/marginal prediction (the belief's "ownership"); per-cluster occupancy-k
prediction; time-to-resolution; expected-remaining-reward distribution; which-face-resolves-next
(a policy aux); and **value-at-the-next-belief / Bellman-consistency auxiliaries** that directly
attack the calibration deficit the dual bound exposed. For EACH proposal, connect it to the
dual-bound finding: does it calibrate the state-value? does it densify credit on the contingent
sensing chains where the VoI is gated behind depth?

DELIVERABLES — write `docs/consults/consult-003-marry-static-az-katago.md` (and append your full
commission prompt verbatim as an appendix, per the project's consult-record discipline). The doc
contains:
1. A short orienting synthesis: the headroom picture (floor / AZ / decomp / loose ceiling / what
   the dual bound says about *real* headroom) and the central mechanistic thread (value
   calibration + sensing-chain credit assignment).
2. A RANKED set of concrete proposals. For each: (a) the idea; (b) the mechanism — WHY it should
   help, in the language of the findings (calibration, geometry-blindness, VoI-gated-behind-
   depth, the reward-frontier gap); (c) the KataGo move it mirrors; (d) implementability against
   the current AZ code (which file/seam, rough effort); (e) expected impact, honestly scoped by
   the dual-bound headroom; (f) risks / how it could fail or mislead.
3. The static-analysis marriage specifically: can the cluster decomposition / occupancy
   factorization / micro-macro values become AZ input features, auxiliary targets, an
   architectural prior, or a search prior? Concrete proposals under the same rubric.
4. An honest verdict: given the (qualitative) headroom and that the real lever is value
   calibration, IS envelope-pushing worth it — and if so, which 1-3 bets have the best expected
   value? Or is the honest finding that AZ is near a practical ceiling and the remaining gap is
   structurally hard? Say which.

CONSTRAINTS: branch `docs/consult-marry-static-az`; commit with EXPLICIT PATHS only (NEVER
`git add -A`); end commit messages with `Co-Authored-By: Claude Opus 4.8
<noreply@anthropic.com>`; do NOT push. Your final message IS the record — make it a complete,
self-contained rendering of the consult's substance (not a pointer to the file). Be honest and
mechanistic.

---

## Appendix B — KataGo source extraction (fetched 2026-06-14)

From David Wu, "Accelerating Self-Play Learning in Go" (arXiv:1902.10565, via ar5iv) and
`lightvector/KataGo/docs/KataGoMethods.md`. The network was available; numbers below are as
fetched. Ablation impacts are at 2.5G equivalent queries (paper Table; "approximate, from
shorter ablation runs").

**Paper auxiliary targets and architecture (ablation slowdown when removed):**

| component | mechanism KataGo names | slowdown when removed |
|---|---|---|
| **Ownership** (per-point owner prob, {0,0.5,1}) | localized spatial credit assignment: "with an ownership target the net receives direct feedback on which area was mispredicted, with large errors and gradients localized to the mispredicted area" | bundled with score = **1.65×** (largest single effect) |
| **Score distribution** (full pdf+cdf over final score, not scalar) | uncertainty-aware utility; integrate for expected utility, trade win vs margin rationally | (bundled with ownership above) |
| **Auxiliary policy** (opponent's reply next turn) | regularization via predicting future actions; weighted `wopp=0.15` (15% of main policy); never used at play | **1.30×** |
| **Global pooling** (per-channel mean, board-width-scaled mean, max → 3c) | lets a *locally-blind conv net* condition on global context (winning/losing status, ko) | **1.60×** |
| **Game-specific input features** (liberties, ladders, pass-alive, komi parity) | pre-structured domain knowledge | **1.55×** |
| Playout cap randomization | (training-data quality) | 1.37× |
| Forced playouts + policy target pruning | (exploration / target quality) | 1.25× |

Loss weights quoted: ownership `wo=1.5/b²`; score pdf/cdf `0.02` each; opponent policy
`wopp=0.15`.

**Post-paper methods (`KataGoMethods.md`):**

- **Short-term value/score targets**: predict exponentially-averaged future MCTS values,
  `(1−λ)Σ_{t′≥t} MCTS_value(t′)·λ^{t′−t}`, at three λ giving mean horizons ~6/16/50 turns.
  "Nets train slightly faster and achieve better value/score loss on the main head" by
  decreasing the main-head weight and adding these — a bias/variance tradeoff giving "lower-
  variance feedback." Enables the two methods below.
- **Uncertainty-Weighted MCTS Playouts**: an auxiliary head predicts
  `(stop_grad(NN_shortterm) − MCTS_shortterm)²`; at test time playouts are reweighted —
  uncertain playouts count as "a fraction of a playout," confident ones as "more than one" —
  in averaging and PUCT. Aligns search intensity with position uncertainty.
- **Dynamic Variance-Scaled cPUCT**: scale cPUCT ∝ √(utility variance) per node; high-
  variance (fights) ⇒ more exploration. Combined with uncertainty weighting, ~75 Elo.
- **Optimistic Policy**: an auxiliary policy head trained on the same target but reweighted
  toward samples where the player found a surprising value improvement (`z_value`/`z_score`
  via the error head); used for both sides at inference. ~40–90 Elo. Explores tactics the raw
  model missed; reduces horizon-delaying moves.
- **Policy soft target** (v1.12.0+): predict `policy^(1/T)`, T=4, to discriminate low-mass
  moves; weighted 8×.

Mapping to chocofarm (drawn in §2–§3): ownership ↔ per-treasure presence (Bet 2); score
distribution ↔ per-cluster occupancy-k / remaining-reward distribution (Bet 3); short-term
value targets ↔ Bellman-consistency target, with the key advantage that the belief-MDP gives
the one-step successor-value *expectation in closed form* (no Monte-Carlo, unlike Go) — Bet 1;
uncertainty-weighting + variance cPUCT ↔ Bet 4; global pooling ↔ *does not transfer* (the
chocofarm MLP is already global; no locality to relieve) — Bet 5.
