# A sharper provable upper bound on ρ\* via the information-relaxation dual (2026-06-14)

This document designs and proves valid a **penalized information-relaxation upper
bound** on the optimal long-run rate ρ\* of chocofarm, sharpening the existing
**loose clairvoyant ceiling 0.1454** (`docs/results/voi-ceiling-2026-06-13.md`).
The construction is Brown–Smith–Sun (2010), "Information Relaxations and Duality in
Stochastic Dynamic Programs" (the canonical reference; the Brown–Smith *Review and
Tutorial*, Found. & Trends in Optimization 5(3), restates it with theorem labels and
is the source quoted below).

The existing clairvoyant ceiling is loose for one structural reason: it hands the
solver the **true present-set for free** and the solver, knowing the world, never
pays for a single sensing read — it routes straight to the present treasures. The
whole difficulty of chocofarm is the **costly contingent sensing chains** that a real
policy must run to *discover* which five are present (the +70% VoI gap;
`docs/results/decomp-rate.md` §"Where the remaining gap to the ceiling lives"). Our
bound **charges for that free information** via a dual penalty, and so lands strictly
below 0.1454, toward the achievable frontier (~0.094–0.10).

The construction has two layers that must be composed carefully:

1. The Brown–Smith–Sun (BSS) penalized perfect-information relaxation, which is
   normally stated for a **finite-horizon additive-reward** DP, applied to the
   **fixed-λ differential-value problem** `E[ΣR − λΣT]`.
2. The **Dinkelbach / renewal-reward** transform that turns the long-run *rate*
   ρ\* into a λ-root of that fixed-λ value. §3 proves the composition is valid and
   gets the **monotonicity direction** right so the root ordering holds.

---

## 1. The primal: chocofarm as a fixed-λ belief-MDP, and ρ\* as a Dinkelbach root

### 1.1 The renewal-reward / Dinkelbach object

One *run* (renewal cycle) starts at the entry teleport with the full belief (all
`M = C(20,5) = 15,504` worlds equiprobable), the agent reads faces / collects
treasures / exits, banks `R ∈ {0..5}` treasures in elapsed time `T` (travel +
the single end-of-run exit toll), then the world is re-rolled i.i.d. The long-run
rate of a policy π is the renewal-reward ratio

> **ρ(π) = E[R(π)] / E[T(π)]** (treasures per time unit),

and **ρ\* = sup_π ρ(π)** is what we bound. `env.dinkelbach_rate` computes a given
policy's ρ(π) as its Dinkelbach fixed point.

Dinkelbach's transform linearizes the ratio. For a fixed scalar λ define the
**fixed-λ differential value**

> **g(λ) = sup_π E[ R(π) − λ·T(π) ]**.

`g` is the optimal value of an ordinary additive-reward problem (no ratio). Standard
fractional-programming / average-reward facts (the design doc's Option-A grounding,
`docs/design/alphazero-surrogate-design.md` §4.1; Dinkelbach 1967):

* `g` is **convex** in λ (a pointwise sup of affine functions `λ ↦ E[R(π)] − λ E[T(π)]`)
  and **strictly decreasing** (every feasible policy has `E[T(π)] > 0`: a run costs at
  least the exit toll, `env.tp = 12 > 0`, so each affine piece has strictly negative
  slope).
* `g(λ) = 0` **iff λ = ρ\***. (`g(λ) ≥ 0 ⇔ ∃π: E[R(π)] ≥ λ E[T(π)] ⇔ ∃π: ρ(π) ≥ λ ⇔
  λ ≤ ρ\*`; combined with strict monotonicity, the unique zero of `g` is ρ\*.)

This is exactly the structure `clairvoyant_rate` and `env.dinkelbach_rate` already
exploit: iterate `λ ← rate(at λ)` to the fixed point.

### 1.2 The fixed-λ DP as a finite-horizon belief-MDP

Fix λ. The fixed-λ problem `g(λ) = sup_π E[ΣR − λΣT]` is a **finite-horizon
belief-MDP** in the BSS sense, with the standard POMDP "belief = sufficient
statistic of history" identification (Kaelbling–Littman–Cassandra 1998), which the
env already implements exactly:

* **Scenario** ω = the latent world `w` (a 5-of-20 bitmask), drawn uniform over the
  15,504. This is *the* source of randomness for one run; all observations are
  deterministic functions of `w` and the actions.
* **State** `x_t = (loc_t, b_t, c_t)` — location, belief world-set `b_t ⊆ worlds`,
  collected-set `c_t`. (`env`'s `(loc, bw, collected)`.)
* **Action** `a_t ∈ {("t", i), ("d", j), TERMINATE}` — collect treasure i, read
  face j, or end the run.
* **Reward** for a non-terminal step is the **λ-penalized increment**
  `r_t = (reward of a_t) − λ·dt_t`, where `dt_t = d(loc_t, target(a_t))` is the
  deterministic travel and the collect-reward is `value[i]·1[w∈present, i∉c_t]`.
  TERMINATE contributes the final reward `−λ·exit_cost(loc)` and ends the run.
* **Transition** `x_{t+1} = apply(x_t, a_t, w)` — exactly `env.apply`: a collect
  filters the belief on the realized presence bit and may grow `c`; a face read
  filters the belief on the realized disjunction polarity. The natural filtration
  `F_t` is "everything observed through the start of period t" = the belief `b_t`
  (and `loc_t, c_t`), which is what makes `b_t` the sufficient statistic.

The total run reward is `r(a, ω) = ΣR − λΣT` and `g(λ) = sup_{π F-adapted} E[r(π)]`.
The horizon is finite (`env.simulate` caps at `max_steps = 40`, and the optimal
policy terminates well before that — a run has at most 5 useful collects and
finitely many informative reads).

**Reward measurability (load-bearing for §2).** The travel part `−λ·dt_t` of `r_t`
is `F_t`-measurable: distances are deterministic and the action is chosen at t. The
**collect-reward part is NOT `F_t`-measurable** — `value[i]·1[w∈present_i]` depends on
the realized presence bit, which is revealed *by* the action, not known at the start
of the period. This is why the penalty below must use the **full BSS generating
function `w_t = r_t + V̂_{t+1}`** (reward included), not the bare `V̂_{t+1}` form; the
reward terms do **not** cancel. (BSS Review Prop. 3.1, eq. 3.14 vs. the eq. 3.13
bare form, which is valid only for `F_t`-measurable `r_t`.)

---

## 2. The penalized perfect-information relaxation (fixed λ)

### 2.1 The relaxation and the BSS weak-duality bound

The **perfect-information relaxation** `G_t = F` for all t hands the solver the entire
scenario ω (the world w) up front. Under it, the BSS weak-duality theorem (Review
Thm 3.1, perfect-information form eq. 3.2) states: for **any dual-feasible penalty**
`z(a, ω)`,

> **g(λ) ≤ E_w[ sup_{a ∈ A(w)} ( r(a, w) − z(a, w) ) ] =: B(λ, z).** (★)

The sup is **inside** the expectation (the perfect-information special case), over
**all action sequences feasible in the fully-revealed world w** — and crucially this
includes sensing reads and visits to absent treasures, because the penalty can make
sensing worthwhile in the relaxed problem. `B(λ, z)` is the **penalized clairvoyant
value** at λ. With `z ≡ 0`, (★) is the existing loose hindsight bound, and `B(λ, 0)`
is exactly what `clairvoyant_rate`'s inner double-loop computes (the absent-visit and
sense actions are dominated away when z = 0, collapsing the path sup to "pick a
subset of the present and route it" — see §4.2). Any `z` that *charges* for
foreknowledge tightens (★) below the hindsight bound.

**Dual feasibility** of `z` means `E[z(a, ·)] ≤ 0` for every F-adapted (nonanticipative)
policy a (Review §3.1: penalties that "do not penalize, in expectation, temporally
feasible policies"). The weak-duality chain is
`g(λ) = E[r(α_F)] ≤ E[r(α_F) − z(α_F)] ≤ E[ sup_a (r − z) ] = B(λ, z)`: the first
inequality is dual feasibility, the second is `A_F ⊆ A_G`. So **(★) holds for any
dual-feasible z** — weak duality, claim (b) of the task.

### 2.2 The penalty z from an approximate value function V̂

We build z by the BSS **value-function-generated penalty** (Review Prop. 3.1,
eq. 3.14): given any approximate value function `V̂_t(x_t)` (an estimate of the
fixed-λ value-to-go from state x_t; see §2.4 for the V̂ we use), define the per-step
generating function `w_t = r_t + V̂_{t+1}(x_{t+1})` and the penalty as a **sum of
martingale differences along the realized information process**:

> **z(a, w) = Σ_t [ ( r_t + V̂_{t+1}(x_{t+1}) ) − E[ r_t + V̂_{t+1}(x_{t+1}) | F_t, a_t ] ]** (z)

Each summand is a realized "reward-plus-next-value" minus its **one-step conditional
expectation under the natural filtration** `F_t` (= the belief b_t at decision t),
given the action a_t actually taken. The conditional expectation averages over the
*observation outcome* of a_t under the belief b_t — the only randomness resolved in
period t:

* **face read `("d", j)`**: r_t = −λ·dt (deterministic, cancels in the difference).
  The next state branches to the positive / negative successor belief. With
  `p⁺ = P(positive | b_t) = (b_t & cover_j ≠ 0).mean()`,
  `E[V̂_{t+1} | F_t, a_t] = p⁺·V̂(b_t⁺) + (1−p⁺)·V̂(b_t⁻)`,
  and z's increment is `V̂(b_t^{realized}) − that expectation`.
* **collect `("t", i)`**: r_t = −λ·dt + value[i]·1[present]. With
  `q = P(i present | b_t) = marginals(b_t)[i]`, the difference is
  `(value[i]·1[present] + V̂(x_{t+1})) − ( q·(value[i] + V̂(x^{pres})) + (1−q)·V̂(x^{abs}) )`.
  The reward term `value[i]·1[present] − q·value[i]` does **not** cancel — this is
  exactly the `F_t`-non-measurability noted in §1.2, and is why the full
  `r_t + V̂_{t+1}` form is mandatory.
* **TERMINATE**: deterministic (no successor randomness), increment 0.

### 2.3 Proof (a): dual feasibility — E[z | info available] = 0 at each step

Each increment of (z) is, by construction, of the form `w_t(a^t) − E[w_t(a^t) | F_t]`
(BSS eq. 3.7 with `w_t = r_t + V̂_{t+1}`). For **any** F-adapted policy, the action
a_t is `F_t`-measurable, so `w_t(a^t)` is a well-defined random variable and

> **E[ w_t − E[w_t | F_t] | F_t ] = E[w_t | F_t] − E[w_t | F_t] = 0** a.s.

i.e. each increment is a **martingale difference under F** (zero conditional
expectation given the info actually available at t). By the law of iterated
expectations `E[z(a, ·)] = Σ_t E[ E[increment_t | F_t] ] = 0` for every F-adapted a —
dual feasibility holds, and with **equality** (not merely ≤ 0), for **any V̂
whatsoever** (BSS Prop. 3.1; the construction needs no optimality of V̂). This is the
direct dual-feasibility statement the task asks for, and it composes the **empirical
mean-zero check** of §5 deliverable (iv): sampling worlds and averaging the realized
per-step increments under the belief filtration must give ≈ 0.

Note this is robust: **any** V̂ — wrong, crude, or the decomp value — yields a *valid*
bound. V̂'s quality affects only the **tightness** of B(λ, z), not its validity. When
`V̂ = V*` (the true fixed-λ optimal value), the penalty is BSS-optimal and (★) holds
with **equality** (strong duality, zero gap; Review Thm 3.4) — the inner penalized
value becomes constant across worlds and B(λ, z) = g(λ). We approach but do not
attain this; §6 is honest about how far.

### 2.4 The V̂ we use

`V̂_t(x)` must estimate the fixed-λ value-to-go `E[ΣR − λΣT | x]` of (near-)optimal
continuation from state x. Three generators, in increasing strength:

1. **Trivial analytic V̂₀ (sanity baseline).** `V̂₀(x) = Σ_i marginals(b)[i]·value[i]·1[i∉c]
   − λ·exit_cost(loc)`: "expected still-collectable reward if we could grab it for free,
   minus the cost to leave." Crude, but it is a genuine value estimate and it makes the
   penalty *do something* (it charges for resolving marginals). Its only role is to
   confirm the machinery: any V̂ is dual-feasible, so B(λ, V̂₀) must be a valid bound
   ≤ 0.1454 — a cheap regression that the harness is wired correctly. Expect it to be
   only modestly tighter than the hindsight bound.

2. **Decomp value V̂_D (the natural first real choice).** The exact hierarchical
   decomposition (`chocofarm/solvers/decomp.py`) already computes, at a fixed λ, an
   accurate **belief value function**: the macro's `value(loc, posterior, …)` returns
   the λ-value of the live macro state, and the micro `enter_value` / `solve` give the
   exact per-cluster continuation values. Composed, these yield `V̂_D(loc, b, c)` =
   (live occupancy posterior over cells) chained through the micro continuation values
   plus the exit toll — the **same object the decomp policy acts on**, reused as the
   penalty's value-function approximation. Because the decomp value is *accurate*
   (it produces the 0.094 achievable rate), it is a strong penalty: it charges
   foreknowledge close to what a near-optimal belief-policy would itself value, so
   B(λ, V̂_D) should land well below 0.1454. (It is *not* the optimal V\*, so the bound
   is not tight — but it is the best trusted V̂ we have.)

3. **Frozen AZ value-net checkpoint V̂_AZ (optional, stronger; NOT depended on).** A
   frozen AZ value head (`chocofarm/az/`, trained at a fixed λ) is a drop-in
   `V̂_AZ(features(b, loc, c))`. It is an *optional* generator behind a flag and is
   **not** wired into the validation, because (i) it requires the live AZ run's
   artifacts and (ii) its validity is identical to any other V̂ — it can only change
   tightness. The bound's correctness never depends on it.

The implementation takes V̂ as an injected callable; the three above are concrete
instances. **Whatever V̂ is chosen, (★) is a valid upper bound** — that is the whole
point of dual feasibility.

---

## 3. Composing with Dinkelbach: the λ-root of B(·, z) upper-bounds ρ\*

This is the step the literature does not hand us (BSS is finite-horizon additive,
not a ratio). We must show the **λ-root of the penalized clairvoyant value
upper-bounds ρ\***, with the monotonicity direction right.

**Setup.** Define `B(λ) := B(λ, z_λ)`, the penalized clairvoyant value at λ (z may
itself be built at λ, since V̂ is fixed-λ — written `z_λ`). Define the candidate
bound `λ̄` as the **root of B**: `B(λ̄) = 0`. We claim **ρ\* ≤ λ̄**.

**Lemma (pointwise domination).** For every λ, `g(λ) ≤ B(λ)`. — This is exactly (★)
(weak duality at that λ), valid for any dual-feasible z_λ. ∎

**Monotonicity of B.** `B(λ) = E_w[ sup_a ( R(a,w) − λ·T(a,w) − z_λ(a,w) ) ]`. Hold z
*fixed* and view the map `λ ↦ sup_a (R − λT − z)`: it is a pointwise sup of affine
functions of λ with slopes `−T(a,w) < 0` (every feasible path has `T ≥ exit toll
> 0`), hence **convex and strictly decreasing in λ**, for each w; the expectation
preserves both. So with z held fixed B is strictly decreasing. We additionally need
B *as actually evaluated* (z_λ rebuilt at each λ) to be **decreasing** so its root is
well-defined and the ordering below holds. Two routes secure this:

* **(Route A — z built at a single fixed λ\*, the clean one we use.)** Build z once at
  a reference λ\* (e.g. the decomp's own fixed point ≈ 0.094, a defensible operating
  point) and **hold z fixed** while scanning λ to find the root of
  `λ ↦ E_w[sup_a(R − λT − z)]`. Then B is *exactly* strictly decreasing (the affine
  argument above), the root λ̄ is unique, and the lemma gives the bound. This is the
  recommended, provably-clean variant — z fixed, λ scanned.

* **(Route B — z_λ rebuilt at each λ.)** If V̂ (hence z) is rebuilt at each λ, B may
  not be globally monotone. But the bound still holds *without* monotonicity of B, by
  the following direct argument, so Route A's monotonicity is a convenience, not a
  necessity.

**Theorem (the bound).** Let λ̄ satisfy `B(λ̄) = 0` (Route A: the unique root; Route B:
any root). Then **ρ\* ≤ λ̄**.

*Proof.* By the Lemma at λ = λ̄, `g(λ̄) ≤ B(λ̄) = 0`. By §1.1, `g` is strictly
decreasing with unique zero at ρ\*, and `g(λ) ≤ 0 ⇔ λ ≥ ρ\*`. From `g(λ̄) ≤ 0` we get
**λ̄ ≥ ρ\***. ∎

The direction is the crux: `B(λ̄) = 0` forces `g(λ̄) ≤ 0` (domination), and `g ≤ 0`
forces `λ̄ ≥ ρ\*` (g decreasing). Get either monotonicity backwards and the ordering
flips into a *lower* bound — which is why §1.1 pins `g` strictly decreasing (exit toll
> 0) and §3 pins the affine slopes `−T < 0`. Both hold because **time is strictly
positive on every run**.

**Why λ̄ should land below 0.1454 (a tightness claim, not a theorem).** Two facts
frame this honestly:

1. **0.1454 is itself a valid upper bound** — it is the root of `B(·, 0)` (the z ≡ 0
   hindsight relaxation = `clairvoyant_rate`). So is `λ̄` (the root of B with our z).
   Both certify `ρ\* ≤ ·`. The *certified* bound we report is therefore
   **min(λ̄, 0.1454)** — taking the tighter of two valid upper bounds is always valid.
   This makes the deliverable robust: even a useless V̂ cannot push the reported number
   above 0.1454, because we always have the z ≡ 0 bound in hand.

2. **Whether λ̄ < 0.1454 strictly** turns on whether `B(λ, z) < B(λ, 0)` near the
   ceiling λ. This is *not* automatic pointwise (z can be negative on some paths), but
   it is the generic behaviour of a *good* penalty: a value-function penalty built from
   an accurate V̂ rebates the foreknowledge-exploiting paths the hindsight relaxation
   over-credits, lowering the per-world inner sup in expectation, hence lowering B at
   the relevant λ, hence moving the root λ̄ down (B strictly decreasing). The BSS
   ideal-penalty theorem is the limiting case: at `V̂ = V*`, B collapses to g and
   `λ̄ = ρ\*`. We are between the two, so the *expected* landing is strictly below 0.1454
   and above ρ\*.

The honest framing: **validity is proven** (`ρ\* ≤ min(λ̄, 0.1454)`); **tightness is
measured** (§5(ii) confirms λ̄ ≤ clairvoyant on the sub-instance; the full-run margin
below 0.1454 is the empirical payoff).

---

## 4. The inner per-world optimization — exact, or the bound breaks

This is **the key correctness hazard** the task names. (★) is an upper bound **only if
the inner `sup_a (r − z)` is a genuine supremum or an over-estimate of it.** If we
compute a *lower* bound on the inner max — e.g. by evaluating `r − z` along a single
heuristic path, or by an incomplete search that can miss the true argmax — the
quantity is **not** guaranteed to upper-bound g(λ): the weak-duality chain
`g ≤ E[r − z] ≤ E[sup(r − z)]` breaks at the second inequality. (This is immediate
from (★) being defined as a supremum; the failure mode is the under-solve.)

### 4.1 The inner problem is a *deterministic shortest-path-with-penalty* DP per world

Fix a world w. The world is fully revealed, so every observation outcome is
**determined**: reading face j returns `1[w & cover_j ≠ 0]` (a known bit), collecting
treasure i returns `value[i]·1[(w>>i)&1 and i∉c]` (known). Therefore the *only*
remaining choices are **which actions to take and in what order** — a deterministic
sequencing problem on a known graph. The belief b_t along a path is a deterministic
function of the action prefix and w (it is whatever survives the realized filters),
and the penalty z accumulates the known per-step increments along that path. So the
inner problem is

> `sup over finite action sequences a (collects, reads, then TERMINATE) of
>    Σ_t [ r_t(a, w) − z_t(a, w) ]`,

a **deterministic** longest-path / sequencing optimization in a fully-known world.

### 4.2 Why z ≡ 0 collapses to the existing clairvoyant inner solve (regression check)

With z ≡ 0 and known w: a **face read** has r_t = −λ·dt < 0 and yields no reward (the
world is already known) — strictly dominated, never taken. A visit to an **absent**
treasure has r_t = −λ·dt < 0 — strictly dominated. So the inner sup collapses to "pick
a subset S ⊆ present(w) and an order, maximize ΣR(S) − λ·route_time(S)" — **exactly**
`clairvoyant_rate`'s inner double loop (subsets × permutations). Thus `B(λ, 0)`
reproduces the clairvoyant value, and its root reproduces the clairvoyant rate (0.1454
on the full instance; the sub-instance's own clairvoyant value on a sub-instance).
This is the §5(i) regression — **MEASURED PASS**: on the NW-cluster sub-instance the
z ≡ 0 dual (0.0991) reproduced the clairvoyant reference (0.0990) to bisection
tolerance.

**Implementation subtlety (the "z≡0" mode is `vhat=None`, not `V̂≡0`).** The pure
relaxation needs the penalty *identically zero*: the inner objective is the realized
Σ r_t. Note this is NOT the same as choosing V̂ ≡ 0 in the value-function penalty: with
V̂ ≡ 0 the generated penalty is z_t = r_t − E[r_t | F_t, a_t] (the reward's deviation
from its conditional mean — still dual-feasible, still valid, but nonzero and giving
the *expected*-reward inner objective, not the realized one). So the code has a
distinct `vhat=None` no-penalty mode (realized r_t) for the regression, separate from
`vhat=vhat_zero` (the V̂≡0 reward-deviation penalty). This was a real bug found in
validation: the first cut used V̂≡0 for the regression and failed to reproduce the
clairvoyant per-world value (it returned the expected, not realized, reward). Fixed.

The new code generalizes `clairvoyant_rate` by inserting `− z` into the per-path
objective and **re-admitting sense / absent-visit actions** (which z can now make
worthwhile).

### 4.3 Solving the inner problem exactly (or over-approximating) — the honest plan

With z ≠ 0, sense and absent-visit actions are **no longer dominated** (the penalty
rebates the foreknowledge, so a sense can carry net-positive penalized value). The
exact inner solve is then a deterministic DP over `(loc, set-of-actions-still-useful,
collected)`. Two regimes, both kept **exact or over-approximating** — never
under-approximating:

* **Exact enumeration over the present-and-near set (small instances / validation).**
  In a fully-known world the *reward-bearing* moves are only the ≤5 present treasures;
  the only reason to read a face or visit an absent treasure is the **penalty rebate**.
  We enumerate action sequences over the present treasures **plus** the faces/absent
  treasures whose penalty increment is positive, bounded by a cap, and take the true
  max. For the small validation sub-instances (single cluster / reduced world-set) this
  enumeration is **complete** → exact sup. For the full 15,504-world run we bound the
  action set per world to a **superset** of the truly-useful actions and enumerate
  exhaustively within it (see below) — still the true sup over the admitted set.

* **Over-approximation guard (full instance, if exhaustive is too costly).** The
  validity-preserving fallback is to **enlarge** the inner feasible set or **relax**
  the inner constraints so the computed value is `≥` the true sup — e.g. allow the
  inner solver to "teleport free between any two reward sites" (a relaxation that can
  only *raise* the per-world max), or solve a Lagrangian/LP relaxation of the
  sequencing problem. Any such **over-estimate keeps (★) valid** (it only loosens the
  bound). We will **never** substitute a single-path heuristic or a truncated search
  that can miss the argmax — that is the one move that silently breaks the bound.

**The implemented inner solver is a memoized exact DP** over the reachable
`(loc, belief, collected)` states (full legal action set: possibly-present collects +
informative faces + TERMINATE), with a hard `max_inner_states` cap that **aborts
loudly (ADR-0002-style) rather than silently truncating** — a truncated search could
miss the argmax and return a *lower* bound, the one move that breaks (★). It takes the
true max over the full admitted action set, so it is the exact sup. On the small
validation sub-instances this is complete and fast.

### 4.4 MEASURED tractability finding — the flat DP does NOT scale to the full belief

A real result from validation, recorded honestly: the flat memoized DP **enumerates
every informative face at every state and recurses into its successor-belief splits**,
and on the full 15,504-world belief this explodes — *even a single world's no-penalty
inner DP did not finish in > 60 s on core 3*. The existing `clairvoyant_rate` avoids
this precisely because it **never touches faces** — with z ≡ 0 they are dominated, so
it enumerates only present-subset routes (≤ C(5,s)·s! per world). The penalized inner
solve re-admits faces, and the flat DP pays the full belief-split cost.

So the **full-instance penalized headline is NOT runnable via the flat DP.** The
tractable full-instance path (specified, not yet implemented — this phase is
design+implement+small-validate, not full-compute) is the **decomposition-aligned
separable inner solve**:

* The decomp V̂ (and any cluster-separable V̂) makes the penalty a **sum of per-cluster
  martingale increments** — the clusters partition the treasures and faces, coupled
  only by the global Σ = 5. So in a known world the inner sup **separates** into a
  per-cluster penalized sequencing sub-problem (each cluster's belief semilattice is in
  the hundreds — `decomp-rate.md`) plus a macro routing/ordering layer over the (≤ 4)
  clusters. Each per-cluster sub-problem is the size of the validated mini-instances,
  so it is exactly the tractable regime already demonstrated; the macro layer is a
  small permutation/subset search over clusters. This is exact and tractable, and is
  the natural reuse of `decomp.py`'s micro/macro machinery on the penalized objective.
* The fully-general validity-preserving fallback remains the **over-approximation**
  (enlarge the inner feasible set / relax constraints so the computed value is ≥ the
  true sup), which keeps (★) valid at the cost of looseness. Never the restriction
  (dropping faces is a *lower* bound — forbidden).

The wall-time and the exact full-run command in the report are stated against this
separable path, not the flat DP.

A clean structural fact that justifies the separation: in a known world only
**present** treasures give reward, and a face read's *only* value is its penalty
increment; with a cluster-separable V̂ those increments are intra-cluster, so the
foreknowledge-charging is intra-cluster and the inner optimum factorizes across
clusters up to the Σ = 5 coupling the macro layer carries.

---

## 5. Validation — MEASURED on the NW-cluster sub-instance (core 3, bounded)

Sub-instance: the NW sense-cluster `{8,9,10,11,12}` with `k_local = 2` present →
`C(5,2) = 10` worlds, 17 in-cluster faces, real geometry/faces/costs (a `MiniEnv`
view of the real env). The belief semilattice is small enough that the inner DP is
**provably complete** (exact sup). All runs pinned `taskset -c 3` under `timeout`;
cores 0–2 held by the live AZ job. The four deliverable numbers:

| # | check | result | verdict |
|---|---|---|---|
| (i)  | **Regression** — z≡0 (`vhat=None`) reproduces the clairvoyant inner solve | dual **0.0991** vs clairvoyant **0.0990** | **PASS** (to bisection tol) |
| (ii) | **Tighter than clairvoyant** — with a good V̂ | V̂ = V\*: λ̄ **0.0957** < clairvoyant **0.0990** | **PASS (strict)** |
| (iii)| **Above an achievable rate** — bound ≥ achievable optimum | λ̄ **0.0957** ≥ exact optimum **0.0956** | **PASS** (sits at the optimum) |
| (iv) | **Empirical mean-zero penalty** (dual feasibility) | mean Σz = **+0.0000 ± 0.0000** over 400 runs | **PASS** |

The headline of the validation is the **V̂ = V\*** row: handing the exact optimal
belief-MDP value function as the penalty drives the dual bound to **0.0957 ≈ ρ\*_sub =
0.0956** — i.e. **strong duality** (BSS Thm 3.4) holds to bisection tolerance, the bound
is *tight*, and it sits strictly below the clairvoyant 0.0990 and exactly at the
achievable optimum. This is the direct demonstration that **the machinery tightens
when handed a good V̂** and that all three orderings (achievable ≤ λ̄ ≤ clairvoyant)
hold. (Why V̂ = V\* rather than the decomp value for the tightening demo: see §6 — the
decomp *decision*-value is not a self-consistent state-value and loosens; V\* is the
clean test of the construction itself.)

Reproduce: `PYTHONPATH=. timeout 600 taskset -c 3 python -m
chocofarm.bounds.eval_bound --validate`

### 5.1 Full run — wall-time and command (the deferred heavy computation)

The orchestrator runs the full computation later with cores freed. Two pieces:

* **z≡0 regression → 0.1454** (the loose ceiling this sharpens): tractable in
  **seconds** — it is the existing `clairvoyant_rate` (subset×perm over the present
  set, no faces). `--full` runs exactly this and prints the ceiling. **Safe to run
  now; ~5–15 s.**
* **Penalized full bound (the actual sharpened number):** the **flat DP is intractable**
  on the full belief (§4.4, measured — a single world did not finish in >60 s). It must
  be the **decomposition-aligned separable inner solve** (per-cluster penalized
  sub-problems + a macro cluster-ordering layer), which is *specified but not yet
  implemented* (this phase is design + small-validate). Once built, its cost is
  dominated by the per-cluster sub-solves: each is the size of the validated
  mini-instances (a penalized B-eval on the NW mini was ~14–44 s depending on V̂, on a
  *contended* core 3); with cores freed and only ≤4 clusters × a bisection (~40 B-evals)
  the estimate is **order ~20–40 min on a freed 4-core host** if the per-cluster
  sub-solves are parallelized across cores, comfortably inside the ~50-min AZ-pause
  budget. **This estimate is contingent on the separable solve being built**; the flat
  DP as shipped cannot produce the full number.

Exact ready-to-run command (z≡0 ceiling now; penalized number after the separable
solve lands):

```
PYTHONPATH=/home/bork/w/vdc/chocobo-bound timeout 3000 taskset -c 0-3 \
  /home/bork/w/vdc/venvs/generic/bin/python -m chocofarm.bounds.eval_bound --full
```

---

## 6. Honest caveats

* **Validity vs. tightness.** `ρ\* ≤ λ̄` is **proven** (weak duality + the Dinkelbach
  composition, §3). That λ̄ < 0.1454 (the bound is *useful*) is a **tightness** claim
  that depends on V̂'s quality and is **measured**, not proven. A poor V̂ yields a
  valid-but-loose bound.
* **The inner solve is the single fragile point.** Everything rests on the inner
  per-world optimization being an exact sup or an over-estimate (§4). The
  implementation does a memoized exact DP over the full legal action set and **aborts
  loudly** if the state cap is hit, rather than truncating. Any future performance
  shortcut here must preserve the exactness / over-approximation property or the bound
  silently becomes invalid. Flagged in the code (the `RuntimeError` refuses to
  truncate).
* **Flat DP does not scale; the full headline needs the separable solve (§4.4).** The
  flat per-world DP is intractable on the full belief (measured). The full-instance
  penalized bound requires the decomposition-aligned separable inner solve, which is
  *specified but not implemented in this phase* (design + small-validate only). The
  small-instance validation is exact and complete; the full number is the remaining
  build.
* **The decomp V̂ LOOSENS, not tightens (a measured, honest finding).** The decomp
  `macro.value` is a *decision* value — accurate enough to steer the 0.094 policy — but
  it is **not a self-consistent state-value**: its one-step martingale increments are
  large and exploitable (measured gap `E[V̂'|F] − V̂ ≈ −0.39` for a face read at the NW
  belief). As a penalty generator this makes B *higher* (the inner clairvoyant games
  the penalty), so the decomp-V̂ dual root λ̄ ≈ **0.15** — *looser* than even the
  sub-instance clairvoyant 0.099. The bound stays valid only via the certified
  `min(λ̄, clairvoyant)` fallback. **Tightening below 0.1454 on the full instance
  therefore needs a *calibrated* V̂** — either V\* (exact, the tightening demo, but
  intractable full-scale), a frozen AZ value-net (the §2.4(3) generator, the most
  promising tractable calibrated V̂), or a calibrated decomp value (subtract the
  realized-reward baseline / re-anchor so the Bellman residual is small). This is the
  key tightness lever and the honest gap between "valid" (done) and "useful on the full
  instance" (needs a calibrated V̂).
* **Strong duality is approached when V̂ is good.** Demonstrated: V̂ = V\* gives
  λ̄ = 0.0957 ≈ ρ\*_sub = 0.0956 on the sub-instance (tight). On the full instance the
  achievable landing depends entirely on the calibrated-V̂ quality; the honest target
  is below 0.1454 and above the 0.094 frontier, *contingent on a calibrated V̂* — not
  guaranteed by the decomp value as-is.
* **Route-A z-fixing.** The cleanest monotonicity argument fixes z at a reference λ\*
  and scans λ for B's root (§3 Route A). If z is instead rebuilt per λ, the bound is
  still valid (Theorem holds at any root of B), but B's global monotonicity is no
  longer guaranteed; we use Route A for a unique, well-behaved root.
* **Single instance, uncalibrated time model.** As with every chocofarm result, the
  number is conditioned on `TELE_OH = 12` and symmetric Euclidean travel; the bound is
  only as meaningful as the env.

---

## 7. References

* Brown, Smith & Sun (2010), "Information Relaxations and Duality in Stochastic
  Dynamic Programs," *Operations Research* 58(4):785–801. — weak duality (Thm 3.1),
  value-function-generated penalty (Prop. 3.1, eqs. 3.7/3.14), ideal-penalty strong
  duality (Thm 3.4).
* Brown & Smith, "Information Relaxations and Duality in Stochastic Dynamic Programs:
  A Review and Tutorial," *Found. & Trends in Optimization* 5(3):246–339. — the quoted
  theorem statements and the perfect-information eq. (3.2).
* Dinkelbach (1967), "On Nonlinear Fractional Programming," *Management Science*
  13(7). — the ratio-to-parametric transform and `g(λ)=0 ⇔ λ=ρ\*`.
* Kaelbling, Littman & Cassandra (1998), "Planning and acting in partially observable
  stochastic domains," *AIJ*. — belief as sufficient statistic.
* The chocofarm record: `docs/results/voi-ceiling-2026-06-13.md` (the 0.1454 ceiling
  this sharpens), `docs/results/decomp-rate.md` (the 0.094 frontier and the V̂
  generator), `docs/design/alphazero-surrogate-design.md` §4.1 (the λ-penalized
  differential value = average-reward differential value at gain λ).
