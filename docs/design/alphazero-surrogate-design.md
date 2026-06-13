# AlphaZero-style learned surrogate for chocofarm — design-space map and spec (2026-06-13)

A design investigation, not an implementation. It asks one question: **could a learned
policy/value over the belief claw back a meaningful slice of the +70% value-of-information
gap that every model-based search tried so far has missed?** The honest answer is *plausibly,
but the design space is narrow and the risk is concentrated in two places* — the value
target under the rate objective, and whether the determinization optimism that sank NMCS/
ISMCTS is actually cured by a learned value or merely relocated. The bulk of this document
is a concrete spec an implementer can run, ending in a single decisive first experiment that
costs a few CPU-hours and settles whether the idea is worth a full loop.

The tone throughout is the project's: state the tradeoffs, name the failure modes, and do not
oversell. The literature is encouraging but the prior on this specific instance — six search
methods all below the static floor — is not.

---

## 0. What the prior evidence forces us to design *around*

These are load-bearing facts from the existing record (`docs/results/*`, `docs/agents/*`,
`docs/consults/consult-001-report.md`) plus small read-only measurements run for this study
(pinned to core 3, bounded). They constrain the design more than the literature does.

**F1 — The headroom is real and large.** static floor 0.0855, clairvoyant ceiling 0.1454,
**+70%** (`docs/results/voi-ceiling-2026-06-13.md`). The clairvoyant banks 4.55/5 treasures in
E[T]≈31; static sweeps ~16 of 20 in E[T]≈47. The gap is *structural* (go straight to the
present five, skip absent, exit early), and it survives unit values — so this is not a
"flatten the values" problem.

**F2 — The detector model is genuinely disjunctive (the consult's feared bug did NOT
survive the fix).** Measured cover sets: sizes 2–5, mean 3.12, **zero singleton detectors**.
A negative read on D_8 rules out all five of {8,9,10,11,12} at once; a positive read on a
2-cover like D_2={0,2} lifts each covered marginal from 0.25 to ~0.56 but cannot say which.
This is exactly the regime the whole project is about, and it is intact. (This refutes the
consult's E0 worry on the live instance.)

**F3 — Detectors are individually weak; VoI is gated behind *chains*.** One positive read
yields marginals ~0.35–0.56 on the cover; cornering a single treasure to certainty needs
several overlapping positive/negative reads. The ISMCTS report verified that *after* ~8
detector reads sharpen the belief, base-playout returns flip strongly positive (+1.0 at
λ=0.08). The value is real but **deep and contingent** — credit is sparse, appearing only at
the end of a sensing chain against a 36-way root branching factor.

**F4 — Every determinized search is optimistically biased; deeper made it *worse*.**
NMCS L2 < L1; ISMCTS improved only as iterations rose and it *stopped sweeping*. Root cause
named in both reports: scoring a line against a *fully revealed* sampled world makes detours
and over-collection look risk-free (a detector reads as free perfect info inside a fixed
world). This is the winner's-curse / maximization-bias signature the project documented for
sparse sampling. **This is the failure mode the AlphaZero hypothesis must actually cure, not
the branching factor.**

**F5 — Belief size spans 1 … 15,504, median ≈ 118, p75 ≈ 1,489, p90 ≈ 7,260.** The full
surviving-world set is small mid/late-episode but enormous at the top. Any featurization that
must materialize or pool over worlds is cheap late and infeasible early. This kills
"DeepSets over surviving worlds" as the *primary* encoder (see §2).

**F6 — Marginals(20) + collected-mask(20) are very nearly a sufficient statistic
on-distribution.** Across 4,000 realistic mixed (detector+collect) histories bucketed by
2-dp-rounded marginal vectors: **3,805 distinct buckets, exactly ONE collision**, and that
single collision is resolved the instant you add the collected-mask (the two beliefs were
identical world-sets with collected={8,17} vs {8}). At 2 detector reads there were *zero*
marginal collisions over 210 beliefs. So the naive "marginals are hopelessly lossy" worry is
**empirically overstated for this instance** — marginals carry the deduction once filtering
has made it. The residual loss is real but small, and the cheap separators (`|belief|`, the
open-clause structure) close most of it. This is the single most important featurization
finding and it argues *against* an expensive set encoder.

**F7 — Compute reality.** 4 cores, cores 0–2 pinned by indefinite rate loggers, ~7.3 GiB
free. Venv has numpy 2.4 + shapely only; **no torch** (but torch-CPU 2.x is pip-installable
for py3.13). Measured costs: a base playout from root ≈ 3.1 ms (324/s); `marginals` on the
full 15,504-world belief ≈ 1.2 ms; `filter_detector` ≈ 0.024 ms; an ISMCTS decision at
it=200 ≈ 0.9 s. **A net forward pass must cost far less than a 3 ms playout to be worth
substituting**, and the feature build (dominated by one `marginals` call) is already ~1.2 ms —
so feature construction, not the net, is the per-node bottleneck on CPU. This shapes the
architecture toward "tiny MLP, amortize the marginals call."

---

## 1. The hypothesis, made falsifiable

The AlphaZero pitch is two claims:

- **(H-amortize)** a learned policy prior lets a *small* guided search reach the deep
  contingent sensing chains that a 200–800-iteration unguided ISMCTS could not, by
  concentrating simulations on the few promising detector→detector→collect lines.
- **(H-calibrate)** a learned value, trained on *honest realized returns*, replaces the
  optimistic determinized rollout at the leaf — so the search stops being fooled into
  over-collection (F4).

H-calibrate is the load-bearing one. If a learned value is *also* optimistic (because its
training targets are themselves produced by an optimistic search, or because it never learns
to distrust an un-sharpened belief), AlphaZero inherits F4 and joins the pack. The whole
design must therefore be organized so that the value target is an **honest Monte-Carlo
realized return under the actual partial-observation dynamics**, never a determinized
best-case. §4 makes this concrete; it is the spec's spine.

A useful sharpening: this problem's adaptive value is almost entirely an **information-
gathering + early-exit option**, not a route-quality problem. So the net's most valuable
output is arguably *the value head* (does sensing here pay before I commit?) more than the
policy head. That inverts the usual AlphaZero emphasis and informs the ablations (§7).

---

## 2. Belief featurization — the crux (narrow, opinionated)

The net must read the belief. The design axis runs from *lossy-but-cheap* (marginals) to
*sufficient-but-infeasible* (the world set). F5 and F6 collapse this axis to a clear winner.

### 2.1 What is the sufficient statistic, and do we need it?

The **theoretical** sufficient statistic for the belief-MDP is the surviving-world set `bw`
itself (it is, by construction, the Bayesian posterior — POMDP belief is a sufficient
statistic of history; Kaelbling, Littman & Cassandra 1998). Equivalently and more compactly,
the **set of accumulated observation clauses** plus the exactly-5 constraint determines `bw`
exactly (the clauses are the only thing filtering ever applied). Either is exact; neither is
fixed-dimension, and `bw` is up to 15,504 wide (F5).

The **practical** statistic, from F6, is far cheaper than feared: `marginals(20) ⊕
collected(20) ⊕ location` is nearly injective on-distribution. We are not obliged to feed the
net the sufficient statistic; we are obliged to feed it enough that two beliefs demanding
different actions are distinguishable. F6 says marginals+collected already nearly does that,
and the cheap separators below close the rest.

### 2.2 Recommended feature vector (fixed dimension, ~90 floats)

Lead with a compact, hand-built vector. **Do not** build a DeepSets-over-worlds encoder as
the primary path — it is infeasible at episode top (F5), and F6 shows it is unnecessary
on-distribution. (A DeepSets *over clauses* is a defensible enrichment if §7's ablation shows
a residual gap; spec'd in §2.4 but not in the baseline.)

Per-treasure block (20 treasures × 4 = 80):
- `marg[i]` — posterior marginal P(τ_i present | belief). The workhorse; carries the deduced
  0/1 pins (F6). One `env.marginals(bw)` call.
- `collected[i]` — 1 if already banked.
- `available[i]` — 1 if `i` uncollected and `marg[i] > 0` (legal-collect mask; also drives
  the policy head mask).
- `dist[i]` — normalized travel distance `env.d(loc, ("t", i))` / map-diag. Geometry the
  value head needs to reason about route cost; cheap, and the net cannot recover it from
  marginals.

Per-detector block (16 detectors × 3 = 48):
- `informative[i]` — 1 if the detector's outcome is still uncertain under `bw`
  (`env.legal_actions` criterion); this is the **open-clause indicator**, the F6 separator
  that distinguishes beliefs marginals alone might blur. Also the detector-action mask.
- `p_pos[i]` — P(positive read | belief) = `(bw & cover_mask[i] != 0).mean()`. The expected
  information content; cheap (one bitwise-and + mean per detector).
- `dist[i]` — normalized `env.d(loc, ("d", i))`.

Global block (~8):
- `log|bw| / log 15504` — belief sharpness (F5; the single most informative scalar about
  "how much do I still not know").
- `n_collected / 5`.
- `expected_remaining_present = sum(marg) ` (≈ 5 − collected at start, shrinks as absents are
  ruled out) / 5.
- `exit_cost(loc)` normalized, and a 3-vector of normalized distances to the three teleports
  (the early-exit option is geometry-dependent; the value head must see it).
- current λ — **if** training a λ-conditioned value (§4.3); omit for fixed-λ.

That is **~90 floats**, every component a few numpy ops dominated by the single `marginals`
call (F7: ~1.2 ms, the per-node bottleneck — cache it across the action loop at a node).

### 2.3 Pairwise/cluster enrichment — measured to be *low priority*

The intuition that disjunctive detection needs pairwise statistics is the natural worry. F6
says it earns little on this instance: the joint structure marginals "drop" is reconstructed
by filtering *before* the marginals are read, so the deduction shows up as marginals near 0/1.
The geometry is also strongly clustered into a handful of packs (NW {8,9,10,11,12}; {13,14,15};
SE {0,1,2,17,18}; the consult's clusters), so a *cluster-count* feature — for each of ~4
geographic clusters, the expected number still present `sum(marg[i] for i in cluster)` and the
cluster's belief-entropy — is the cheapest enrichment if needed. Spec it as an **ablation
add-on (§7), not the baseline.**

### 2.4 If (and only if) §7 shows a residual gap: a DeepSets *clause* encoder

Should the value head plateau below clairvoyant in a way that probes attribute to lost joint
structure, add a permutation-invariant encoder *over the accumulated clause set*, not over
worlds. Each clause is `(type ∈ {pos,neg,collect}, cover-bitmask-as-20-vector)`; embed each
with a small φ-MLP, sum-pool, ρ-MLP to a 16-d code, concatenate to §2.2 (DeepSets:
`f(S)=ρ(Σφ(s))`, the universal permutation-invariant form, Zaheer et al. 2017; Wagstaff et al.
2022 on the latent-dimension caveat). The clause set is small (≤ a few dozen reads/episode),
so this is cheap — unlike worlds. **Held in reserve precisely because F6 predicts it buys
little; adding it pre-emptively is the gold-plating the project warns against.**

---

## 3. Architecture (small, CPU-shaped)

Evidence (F6: features near-sufficient; F7: net must be cheap) points to a **plain MLP**, not
a transformer/set encoder, for the baseline.

- **Trunk:** input ~90 → Dense(256) → ReLU → Dense(256) → ReLU (optionally a 3rd block /
  one residual connection). ~90k–140k params. A transformer over the 20+16 entities is
  *possible* and would share weights across treasures/detectors, but it is premature: the
  fixed per-entity blocks already encode identity, and F7 makes every extra matmul a tax paid
  at every search node. Start MLP; revisit only if §7 ablation shows the MLP under-fits.
- **Two heads off the trunk** (§4 for the value-head semantics):
  - **Policy head:** Dense → logits over the **fixed 37-slot action space** =
    {collect τ_0..19} ∪ {sense D_0..D_15 (by detector id, 16 slots)} ∪ {TERMINATE}. Masked to
    legal actions at use time (the `available[i]` and `informative[i]` features *are* the
    mask). Softmax over unmasked slots.
  - **Value head:** Dense → scalar (see §4 for what scalar).
- **Normalization:** all distances ÷ map diagonal (~17.9 units, from the treasure bbox);
  λ-penalized returns are O(1) at the operating λ, so a `tanh`-free linear value head with
  target standardization is fine. (No need for AlphaZero's `tanh` value: returns here are not
  bounded in [−1,1]; standardize targets to unit variance instead and use a linear head.)

Fixed 37-slot action space (not the variable legal set) is deliberate: it gives the policy
head a stable output layout, and masking handles legality. Detectors are addressed by id, so
a detector that becomes uninformative simply gets masked — no re-indexing.

---

## 4. The value target under the rate objective — the question to get right

This is where most learned-RL-on-a-ratio-objective attempts go wrong, so treat it
explicitly. The objective is long-run **rate = ΣR / ΣT** (renewal-reward), solved by
Dinkelbach: at penalty λ, maximize `E[ΣR − λ·ΣT]`; the λ* where that optimum is 0 is the
rate (`env.dinkelbach_rate`). Three candidate targets:

### 4.1 Option A (RECOMMENDED) — learn the λ-penalized (differential) value at a *fixed* λ

Train the value head to predict, for a state `s`, the **expected λ-penalized return-to-go**
under the current (search-improved) policy:

```
V_λ(s) = E[ Σ_{t≥now} r_t − λ·Σ_{t≥now} dt_t  −  λ·exit_cost(final_loc) | s, π ]
```

This is exactly the quantity the existing search already optimizes (`_base_value`, the
ISMCTS/NMCS playout return), and — crucially — it is precisely a **differential / relative
value function** of the average-reward MDP at gain λ (R-learning's differential value;
Sutton & Barto avg-reward; Dinkelbach's λ is the gain/`ρ*` of the average-reward
reformulation — the two are the same object, e.g. the avg-cost Dinkelbach reformulation in
the fractional-programming literature). So Option A is theoretically clean: *the fixed-point
λ is the rate, and V_λ is the standard value function of the problem at that operating
point.*

Why fixed λ and not the live, drifting Dinkelbach λ: a moving target makes the value
regression non-stationary in a second variable. **Pin λ for the whole training run** to a
defensible operating point — start at the **static floor rate, λ₀ = 0.0855** (a strong known
operating point; the policy we must beat lives there), and after the loop converges, *measure*
the learned policy's own Dinkelbach fixed point unbiasedly (re-using `env.dinkelbach_rate`
machinery) for the headline number. If the learned policy's fixed point drifts far from λ₀,
do one **outer** Dinkelbach step: re-pin λ to the achieved rate and retrain (typically 1–2
outer steps suffice; this is Dinkelbach's own iteration wrapped around the whole learner).

This sidesteps the ISMCTS report's "low-λ over-collection is a Dinkelbach artifact" trap: by
pinning λ at the *static* rate (not the weak policy's low self-consistent λ), the value
target already penalizes time at the correct rate from step one, so the learner is not
trained to over-collect.

### 4.2 Option B — learn the rate directly (REJECTED)

A value head predicting `ΣR/ΣT` to-go is tempting but wrong: rate is **not additive** along a
trajectory (you cannot bootstrap `V(s) = r + V(s')` on a ratio), so it breaks the entire
backup/TD machinery and the renewal structure. Dinkelbach exists precisely to linearize this.
Do not.

### 4.3 Option C — a λ-conditioned value V(s, λ) (OPTIONAL, deferred)

Feed λ as an input feature (it is in §2.2) and train across a small band of λ values, so one
net serves the outer Dinkelbach loop without retraining. Attractive for amortizing the outer
loop, but it adds a regression dimension and sample demand for a benefit (avoiding 1–2
retrains) that is small at this scale. **Recommend Option A for the first experiment; promote
to C only if the outer loop proves to need many λ re-pins.**

### 4.4 Policy target

Standard ExIt/AlphaZero: the **search-improved policy** at the root is the policy target. With
Gumbel low-sim search (§5) the target is the Gumbel *improved policy*
`π' = softmax(logits + σ(completedQ))` over the considered actions (Danihelka et al. 2022) —
which is well-defined even when only a handful of root actions were simulated, the property
that makes Gumbel the right fit for our tiny budget. With classical PUCT it would be the
normalized visit counts `π(a) ∝ N(s,a)^{1/τ}`. Use Gumbel's improved policy (see §5 for why).

### 4.5 Value target *value* — honest MC, never determinized best-case

The value-regression target for a visited state is the **realized λ-penalized return of the
actual episode from that state**, played under true partial-observation dynamics (the
true world is fixed for the episode; the agent only ever sees observations). This is the
calibration cure for F4: the target is what *actually happened* when the policy acted under
uncertainty, not what a clairvoyant rollout in a determinized world would have got. Optionally
blend with the search's bootstrapped root value (a λ in [0,1] TD(λ)-style mix) to cut
variance, but **anchor on the realized MC return** — that is the entire point.

---

## 5. The guided search — Gumbel low-simulation MCTS over chance nodes

Single agent, partially observed, stochastic (disjunctive outcomes = chance nodes). Two
families: net-prior+value **ISMCTS** (the running ISMCTS is the natural expert) vs
**Gumbel-AlphaZero** low-simulation MCTS. They are not exclusive — the right answer is
*Gumbel-style root action selection layered on the information-set tree*.

### 5.1 Why Gumbel, concretely

AlphaZero's PUCT needs many simulations to be a sound policy improvement; with few sims it can
*degrade* the prior (Danihelka et al. 2022). Our budget is tiny (F7: even ~50 net-evaluated
nodes per decision is ~tens of ms only if the net is cheap; we want decisions in well under a
second to make an expert-iteration loop affordable on one core). Gumbel AlphaZero **guarantees
policy improvement at low simulation counts** by:
- sampling `m` root actions *without replacement* via Gumbel-Top-k on `logit + g`
  (g ~ Gumbel) — so a small `m` (say 8–16 of 37) is chosen in a principled, exploratory way;
- allocating the `n` simulations across them by **Sequential Halving** (rounds of
  `n/⌈log2 m⌉` sims, dropping the worst half each round);
- forming the improved policy from **completed Q-values**:
  `π' = softmax(logits + σ(completedQ))`, where unvisited actions' Q is *completed* by the
  value net (`v_mix`), and `σ(q) = (c_visit + max_a N(a))·c_scale·q` (Danihelka's monotone
  transform; c_visit≈50). This is exactly the §4.4 policy target, defined even at `n=16`.

This directly attacks F3 (sparse credit / 36-way branching): instead of UCB exploring all 36
root actions, Gumbel commits the budget to a net-chosen handful, going *deeper* on the few
sensing chains the prior likes — H-amortize in action.

### 5.2 Chance-node handling (the partial-observation part)

The tree is over **information sets** (beliefs), exactly as the existing `ismcts.py` already
does it well — keep that scaffold. Each action is followed by a **chance node**: the
observation outcome (treasure present/absent, detector pos/neg) is resolved by the active
determinization `w ~ bw` and routes to the successor-belief child, while the action's
statistics aggregate over the information set (Cowling, Powley & Whitehouse 2012, the SO-
ISMCTS contract `ismcts.py` implements). Two refinements over the current code:
- **Net priors at every node:** replace the UCB1-with-availability selection at *interior*
  nodes with PUCT using the net's masked policy prior P(s,a) and Q from the value net,
  `argmax_a Q(s,a) + c_puct·P(s,a)·√(Σ_b N_b)/(1+N(s,a))` (the standard AlphaZero PUCT;
  Silver et al. 2017). Gumbel governs only the *root*.
- **Net value at leaves** replaces `_base_value`'s determinized playout — this is the F4 cure.
  A leaf is evaluated by `V_λ(belief)` directly. (Keep an option to *blend* a short base
  playout in early iterations before the value net is trained — see schedule §6.)
- **Determinization is still unbiased** straight from `bw` (the env note; no particle
  filter), but with a chance-aware twist for variance: average each leaf's contribution over
  a small `c=2–4` determinizations of the *immediate* observation outcome, so the chance node
  is not collapsed to one sample (light progressive-widening over outcomes; outcomes are
  binary so widening is trivial).

### 5.3 Why not pure net-ISMCTS (PUCT only)

It would work, but it needs more sims for the same improvement, and at our budget that is the
difference between an affordable loop and an unaffordable one. Gumbel-at-root + PUCT-interior
is strictly the lower-budget-robust choice. **Recommend it.**

### 5.4 Budget

Per decision: `m = 12` root actions, `n = 48` simulations (Sequential Halving:
⌈log2 12⌉ = 4 rounds; ≈12 sims/round), `c_puct = 1.25`, leaf = net value (with `c=2`
outcome-averaging). Episodes ≈ 16 steps (F5 measured), so ≈ 16 × 48 ≈ **~770 net leaf
evals per episode** plus ~770 prior evals. With a ~90-float MLP forward at ≪1 ms (batched, see
§8), a self-play episode is dominated by belief filtering, not the net.

---

## 6. Expert-iteration loop (single agent, no self-play)

ExIt (Anthony, Tian & Barber 2017) is the exact frame: decompose into *planning* (the guided
search = the "expert"/slow system) and *generalization* (the net = the "apprentice"/fast
system); the net's improved guidance makes the next round of search stronger. No opponent, no
self-play symmetry — just an agent against the stochastic simulator (`env.simulate`).

```
init: net θ random (or pretrain policy head to imitate GreedyStopBase, value head to its
      base-playout returns — a cheap warm start that gives the first search a non-flat prior).
pin λ = λ₀ = 0.0855 (static-floor rate).
repeat for I outer iterations:
  1. GENERATE: run E episodes of env.simulate driven by the Gumbel-net search (§5).
     Each step records (features(s), improved-policy π'(s), legal-mask).
     Each step's value target = realized λ-penalized return-to-go of THAT episode (§4.5).
     Fresh world re-rolled per episode (the env's i.i.d. re-roll — this is our "data
     augmentation"; 15,504 worlds means negligible repeat).
  2. TRAIN: SGD/Adam on the replay buffer (last W iterations of data):
        L = α · CE(π_net, π') · legal-mask   +   β · (V_net − V_target)^2   +   γ·||θ||^2
     (AlphaZero loss form: cross-entropy policy + MSE value + L2; Silver et al. 2017.)
     α=1.0, β=1.0 (value is the load-bearing head here — consider β=2.0 per §1), γ=1e-4.
  3. EVALUATE: every iteration, measure the greedy-from-π_net policy's unbiased rate via
     env.dinkelbach_rate (or rate at fixed λ₀) on a held-out seed; track % of +70% clawed.
  4. (outer Dinkelbach, occasional): if the measured rate has drifted far from λ₀, re-pin
     λ to it and continue. Expect 0–2 re-pins total.
```

Data/targets/loss/schedule, concretely:
- **E (episodes/iter):** 200–500. At ≈16 steps each → 3.2k–8k training transitions/iter.
- **Replay window W:** keep the last 4–6 iterations (≈15k–40k transitions); the policy is
  changing, so don't keep ancient data.
- **Train steps/iter:** 1–3 epochs over the buffer, batch 256, Adam 1e-3 → 3e-4 decay.
- **I (outer iters):** plan for ~30–60; the decisive *first* experiment (§9) runs far fewer.
- **Exploration:** Gumbel's root sampling *is* the exploration (no Dirichlet noise needed —
  Gumbel-Top-k already injects root randomness; Danihelka et al. note this replaces Dirichlet).
  Add a temperature on the *executed* action during generation (sample from π' for the first
  few plies, argmax later) to diversify trajectories.

A subtlety worth flagging (ADR-0002 honesty): **the value target is bootstrapped through the
net's own search.** If early nets are bad, early value targets are noisy-but-honest (realized
MC returns are unbiased regardless of policy quality — they just have high variance under a
weak policy). This is *safer* than the determinized-optimism failure (F4): a weak policy
produces low, honest values, not inflated ones, so the learner is not pushed toward
over-collection. That asymmetry is the core reason to expect AlphaZero to behave differently
from NMCS/ISMCTS here — and it is the thing the first experiment must confirm.

---

## 7. Ablations (what to vary, in priority order)

1. **Value-head source (the F4 test):** net-value-at-leaf vs determinized-base-playout-at-leaf,
   *same* search budget. If net-value does not beat the playout leaf, H-calibrate is false and
   the whole approach reduces to the search pack. **This is the most important ablation.**
2. **Features:** §2.2 baseline vs +cluster-counts (§2.3) vs +clause-DeepSets (§2.4). F6
   predicts marginal gains; this checks it.
3. **Search:** Gumbel-root vs PUCT-only, at matched low budget (validates §5.1).
4. **Value emphasis β:** 1.0 vs 2.0 (per §1's inversion).
5. **λ handling:** fixed-λ (A) vs λ-conditioned (C).

---

## 8. Feasibility and cost on one 4-core CPU (cores 0–2 are pinned)

**Verdict: tractable on CPU; a GPU is not warranted for the decisive experiment, and
probably not for the full loop at this scale.** Reasoning:

- The net is tiny (~100k params, ~90 inputs). A CPU forward pass batched over the ≤37 root
  actions is sub-millisecond; the per-node cost is dominated by `env.marginals` (~1.2 ms,
  F7), which is *numpy, not the net*. So a GPU accelerates the cheap part and leaves the
  bottleneck untouched — classic small-model-CPU regime. (If anything, optimize the feature
  build: cache `marginals(bw)` per node, vectorize `p_pos` across detectors in one bitwise
  op.)
- Generation is the cost center: ~770 leaf evals/episode × ~400 episodes/iter ≈ 3×10^5 net
  evals/iter, plus the belief filtering. At a few hundred μs amortized per search-node-with-
  filtering, that is order **minutes per iteration on a single core** — and we have exactly
  one free core (core 3). 30–60 iters ⇒ hours-to-low-days, unattended, under `timeout` and
  `taskset -c 3`. Acceptable for a research probe.
- Memory: the replay buffer of ~40k × ~130 floats is < 25 MB; beliefs are int64 arrays,
  largest 15,504×8 B ≈ 124 KB, transient. Nowhere near the 7.3 GiB ceiling. **No risk of the
  RAM-exhaustion mishap the earlier naive solver caused** — we never enumerate or persist the
  belief space; everything is bounded by the iteration/buffer budget.
- **torch-CPU is pip-installable** (2.x for py3.13, confirmed). Alternatively a ~150-line
  pure-numpy 2-layer MLP with manual Adam is entirely viable at this size and avoids the
  dependency — and keeps the whole thing in the existing numpy-only venv. **Recommend: try
  numpy-MLP first** (zero new deps, full control, fast enough), fall back to torch-CPU only if
  autodiff convenience matters once architecture stops changing.
- **GPU would be warranted only if** the architecture grows to a transformer/DeepSets over
  worlds (rejected, §2) or if generation parallelism across many envs becomes the bottleneck —
  at which point batched GPU inference over a large env-pool helps. Not the first experiment.

The hard operational constraints are honored by construction: pin any run to core 3
(`taskset -c 3`), wrap in `timeout`, never touch cores 0–2, never enumerate the belief space.

---

## 9. The decisive first experiment (smallest thing that decides go/no-go)

The question is **not** "does a full AlphaZero loop beat static" (expensive, many moving
parts). It is the §1/§7-#1 question: **does a learned value, used as the ISMCTS leaf, beat
the determinized playout leaf at matched budget — i.e. is H-calibrate true on this instance?**
If no, stop; AlphaZero will not save it. If yes, the full loop is justified.

**E-DECIDE — one-step value-substitution probe (target: < ~4 CPU-hours on core 3).**

Two stages, both bounded:

*Stage 1 — supervised value learnability (no search-in-the-loop; ~1 hour).*
1. Generate a dataset of (belief-state, honest λ-penalized return-to-go) pairs by running the
   **existing ISMCTS policy** (`ismcts.py`, it=200) on `env.simulate` for ~300 episodes at
   λ=0.0855, logging features (§2.2) at every decision point and the *realized* return-to-go
   from each (F4-honest target, §4.5). ~5k transitions.
2. Train the §3 MLP value head (policy head optional here) to regress V_λ. Report held-out
   R² / MAE.
   - **Decision gate 1:** if V_λ is *not* learnable to decent R² from §2.2 features, the
     featurization (or the premise) is wrong — investigate before any loop. (F6 predicts it
     *is* learnable.)

*Stage 2 — drop the learned value into the search and measure (~2–3 hours).*
3. Build `NetValueISMCTS`: identical to `ismcts.py` but the leaf evaluation calls the trained
   `V_λ(belief)` instead of `_base_value`'s determinized playout. *Same iteration budget.*
4. Measure its unbiased rate (`env.dinkelbach_rate`, or rate at fixed λ₀) on a held-out seed,
   N≥300 episodes for a <2% SE (the prior reports' small-N caveat must not recur — budget for
   the runs).
5. Compare three rows at matched search budget and N:
   - ISMCTS with determinized-playout leaf (the F4 baseline, ≈0.068–0.076 from the report),
   - ISMCTS with **learned-value leaf** (the probe),
   - static floor 0.0855 and clairvoyant 0.1454 as the reference lines.

**Read-out / decision:**
- **GO** (full Gumbel ExIt loop justified) if the learned-value leaf *clears the static floor*
  or at minimum *strictly and significantly beats the playout leaf at matched budget* with the
  ET-shrinking (less over-collection) signature — that is direct evidence H-calibrate cures
  F4. Even +5–10% of the VoI gap clawed back *by the value swap alone* (before any policy
  amortization or loop iteration) is a strong signal, since the full loop adds H-amortize on
  top.
- **NO-GO / rethink** if the learned-value leaf ties or trails the playout leaf. Then the
  value is not the lever (it inherited the optimism, or the gate is genuinely the search
  depth, not the leaf), and a full loop is unlikely to pay — redirect to the consult's
  cheaper structural ideas (cluster-exact decomposition as a *trusted* anchor; recalibrated
  time model) before spending days on a learner.

This experiment is decisive because it isolates the single claim the whole approach rests on
(calibrated value > optimistic playout) from the expensive parts (policy amortization,
iteration), reuses the existing ISMCTS scaffold unchanged, and costs a few core-3 hours. It is
also the natural *first iteration* of the real loop (a learned value bootstrapped from the
current expert), so a GO result is not throwaway — it is iteration 0 of §6.

---

## 10. Tradeoffs and honest risks (the part that argues against itself)

- **The value might inherit the optimism after all.** If, after a few loop iterations, the
  net's value is trained mostly on the policy's *own confident* trajectories, it can drift
  optimistic on under-sampled deep-sensing beliefs (the very states VoI lives in) — sparse-
  data extrapolation. Mitigation: Gumbel's root exploration + executed-action temperature keep
  deep-sensing lines in the data; monitor calibration on held-out deep beliefs. But this is a
  real way the approach could quietly become NMCS-with-a-net. E-DECIDE catches the *first-
  iteration* version of this; the loop must keep watching it.
- **The headroom may be genuinely hard, not just un-amortized.** F3 says VoI is behind long
  chains under weak detectors. If the *optimal* contingent policy needs 8+ sensing reads to
  pin the present five, the renewal cycle spends those reads' travel cost, and the net rate
  gain over static may be smaller than the +70% clairvoyant bound (which pays *nothing* for
  information). The clairvoyant is an *unattainable* ceiling; a realistic learned policy might
  top out well below it even if it works. The honest target is "beat static and claw back a
  *meaningful fraction*," not "approach clairvoyant."
- **Single instance, uncalibrated time model.** Everything is conditioned on TELE_OH=12 and
  symmetric Euclidean travel (consult flaw #6). The whole conclusion could move under a
  recalibrated time model. A learned surrogate trained on the current `env` is only as
  meaningful as the env; this is a model-fidelity risk orthogonal to the ML.
- **Effort vs the cheaper alternative.** The consult's **cluster-exact decomposition** (exact
  belief-MDP within geographic packs, chained as a max-ratio cycle) is a *trusted* anchor that
  is plausibly cheaper to build than a full ExIt loop and would tell us the true within-
  neighborhood adaptive value. If E-DECIDE is NO-GO, that is the redirect. If GO, the cluster-
  exact value is still worth building as a *check* on what the learner achieves.
- **Static baseline fairness.** Headline % should be reported against a *realizable* static
  route (and ideally Stage-1's true `static_optimal_rate`, `chocobo_stage1.py:230`), not only
  the oracle-truncated NN best-prefix the consult flagged as slightly optimistic. Use the same
  reference line for all learned numbers.

---

## 11. Recommended spec — one-screen summary

| Element | Decision |
|---|---|
| **Features** | ~90-float fixed vector: per-treasure (marg, collected, available, dist) ×20; per-detector (informative/open-clause, p_pos, dist) ×16; global (log\|bw\|, n_collected, Σmarg, exit_cost, 3 teleport dists). **No DeepSets-over-worlds** (infeasible+unneeded, F5/F6). |
| **Architecture** | MLP trunk 90→256→256 (ReLU); policy head over fixed 37-slot action space (masked); scalar value head (linear, standardized targets). ~100k params. |
| **Value target** | Option A: λ-penalized differential value V_λ at **fixed λ₀ = static-floor rate 0.0855**; target = **honest realized MC return-to-go** under true partial obs (the F4 cure). Outer Dinkelbach re-pin 0–2× if rate drifts. NOT direct-rate (non-additive). |
| **Search** | Gumbel-AlphaZero root (m=12 actions via Gumbel-Top-k, n=48 sims via Sequential Halving, improved-policy target `softmax(logit+σ(completedQ))`) layered on SO-ISMCTS information-set tree (reuse `ismcts.py` scaffold); PUCT interior with net prior+value; net value replaces determinized playout at leaves; chance nodes = observation outcomes, c=2 outcome-averaging. |
| **Loop** | ExIt (no self-play): generate via net-guided search on `env.simulate`, train on (search-improved π′, realized-MC V) targets, AlphaZero loss CE+MSE+L2 (β≥1 on value), replay window 4–6 iters, E=200–500 episodes/iter, fresh world re-roll = augmentation. |
| **Eval** | Unbiased `env.dinkelbach_rate` vs static floor 0.0855 and clairvoyant 0.1454; headline = % of +70% clawed; N≥300 (no small-N caveat). |
| **Compute** | numpy-MLP (no new deps) first, torch-CPU fallback; `taskset -c 3`, `timeout`; minutes/iter, hours-to-days for the loop; **no GPU**; never enumerate the belief space. |
| **First experiment** | **E-DECIDE**: swap learned V_λ for the determinized playout leaf in the existing ISMCTS at matched budget; GO iff it significantly beats the playout leaf (ideally clears static) with the ET-shrinking signature. ~4 core-3 hours. |

---

## Literature consulted

- Silver et al., "Mastering Chess and Shogi by Self-Play..." (AlphaZero), 2017 — loss
  `(z−v)² − πᵀlog p + c‖θ‖²`, PUCT `Q + c_puct·P·√(ΣN)/(1+N)`, visit-count policy target.
  (via trunghng.github.io AlphaZero notes.)
- Danihelka, Guez, Schrittwieser & Silver, "Policy improvement by planning with Gumbel,"
  ICLR 2022 — Gumbel-Top-k root sampling, Sequential Halving, completed-Q improved policy
  `softmax(logit+σ(completedQ))`, `σ(q)=(c_visit+max N)·c_scale·q` (c_visit≈50); guaranteed
  improvement at low simulation counts; replaces Dirichlet noise.
  https://openreview.net/forum?id=bERaNdoegnO (abstract); ReSCALE arXiv:2603.21162 (worked
  formulas); MiniZero arXiv:2310.11305 (comparison).
- Anthony, Tian & Barber, "Thinking Fast and Slow with Deep Learning and Tree Search"
  (Expert Iteration / ExIt), NeurIPS 2017 — the no-self-play plan/generalize decomposition
  our loop instantiates. arXiv:1705.08439.
- Cowling, Powley & Whitehouse, "Information Set Monte Carlo Tree Search," IEEE TCIAIG 2012 —
  SO-ISMCTS, information-set nodes, subset-armed bandit (already implemented in `ismcts.py`).
- Kaelbling, Littman & Cassandra, "Planning and acting in partially observable stochastic
  domains," AIJ 1998 — belief state as sufficient statistic of history.
  https://people.csail.mit.edu/lpk/papers/aij98-pomdp.pdf
- Zaheer et al., "Deep Sets," NeurIPS 2017 — `f(S)=ρ(Σ φ(s))` universal permutation-invariant
  form (the reserved clause-encoder in §2.4); Wagstaff et al., JMLR 2022, on latent-dim
  caveats. https://www.jmlr.org/papers/volume23/21-0730/21-0730.pdf
- Average-reward / differential-value RL (R-learning; RVI Q-learning; Sutton & Barto avg-
  reward) — the λ-penalized value at fixed λ *is* the differential value at gain λ, which is
  the Dinkelbach reformulation of the rate objective; grounds Option A. (Springer
  10.1007/s00521-024-10620-5; arXiv:2304.03729 full-gradient avg-reward.)
- Dinkelbach fractional programming ⟺ average-cost MDP reformulation — the λ* fixed point is
  the gain ρ*; the parametric subproblems are the per-λ value problems
  (ScienceDirect S0098135409001367; the project's own `env.dinkelbach_rate`).

## Measurements run for this study (read-only, core-3, bounded)

All against the unchanged `env.py`; no solver enumerated the belief space; nothing touched
cores 0–2. Findings F2, F3, F5, F6, F7 above are from these. Verbatim numbers: detector cover
sizes 2–5 (mean 3.12, **0 singletons**); belief-size percentiles p50≈118 / p75≈1489 /
p90≈7260 / max 15504; marginal-vector collisions 1 / 3805 buckets over 4000 histories
(resolved by collected-mask); base playout 3.1 ms, marginals 1.2 ms, ISMCTS it=200 ≈ 0.9 s;
torch absent but torch-CPU 2.x pip-installable.
