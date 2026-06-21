<!--
docs/design/harmonized-estimator-interface.md — the harmonized statistical
interface every leaf-eval benchmark exposes so the Neyman allocation driver
consumes them uniformly (one contract per measurable quantity, the bench owns
its estimator, the driver stays estimator-agnostic).

Band-1, solver-agnostic: this is a contract over *estimators of physical
quantities feeding an uncertainty-propagation model*, with no FFXIII / OR
content. The worked examples are the leaf-eval transport sweep
(tools/analysis/OpenTURNS/), but the contract is a general delta-method
allocation interface.

Design note (a spec, not code). Implementation is a later phase. ADR-0005
authoring discipline; ADR-0006 header.

Public Domain (The Unlicense).
-->

# The harmonized estimator interface — `Estimate`

## 0. Status, provenance, and what this note is

- **Status:** Design proposal (the requirements/spec the suite never had). Not
  implemented; implementation is a later phase. No code is committed by this
  note.
- **Provenance:** Synthesis of four independent designs and their statistical
  critiques (the 1:N:N:1 fan-out for this task), grounded against the actual
  benches, models, store, and driver in `tools/analysis/OpenTURNS/`. Every
  claim that a particular bench does a particular thing was read from that
  bench; every numerical claim (the slope/intercept correlation, the 2-point
  pilot's variance, the `min()` Jensen bias) was executed and is reported with
  its number per ADR-0009.
- **What it governs:** the **single contract** every measurable quantity (a
  *benchmark*) exposes so the Neyman driver
  (`tools/analysis/OpenTURNS/neyman_driver.py`) consumes them **uniformly**,
  whatever the quantity's estimator — a mean of timings, a regression
  slope/intercept, a config pin, a ratio, a quantile. It is the cure for the
  ad-hoc state the brief names: today each bench's `measure()` returns a
  bespoke dict, and a generic sampler fed a regression bench grabbed its
  row-count x-axis instead of the slope (`untrusted_drive._per_sample`, the
  longest-numeric-list heuristic at lines 80–98), and the bound cratered.

This note states **(1)** the contract, **(2)** the Neyman derivation under it,
**(3)** the case-handling table, **(4)** the two subtleties most likely to bite
(the non-smooth `min()` and the slope/intercept correlation), **(5)** the
schema delta, **(6)** the migration path. It is honest per
claims-measured-vs-interpreted: §7 separates what is rigorous from what is a
modelling choice, and names the one case (the `min()` kink) that **cannot** be
folded into the per-input contract and must be handled at the driver instead.

A scoping note (ADR-0002, read-what-you-cite): ADR-0012 is a long file; the
principles this note leans on — **P1** single-home, **P2** seam/port
translate-validate-reject, **P3** one-owner collaborators, **P4**
live-not-frozen, **P5** fail-loud/remove-root-cause, **P8** the typed signature
is the contract's SSOT — were read in full. The C++-component material (**P7**,
**P9**, the concrete C++ wire contract) was **not** relied on here and is not
cited: this is a Band-1 Python statistical contract, not a compiled component.

---

## 1. The contract every bench exposes

A bench's `measure()` returns **one frozen, typed `Estimate` value** (ADR-0012
P8: the signature *is* the contract's SSOT), not a bespoke dict. The driver
consumes only `Estimate`s and never reaches into a bench's internals (no raw
pool, no fit object, no x-axis): the bench **owns** how it produces the numbers
(warm a JIT via `bench_common.warm`, fit a line, bootstrap, read a constant);
the driver owns allocation; the store owns SQL (P3 one-owner split, already the
project's shape — `neyman_driver` / `model_*` / `bench_store`).

```python
@dataclass(frozen=True)
class Estimate:
    theta_hat: np.ndarray            # (k,) the point(s) f is evaluated at. k>=1.
    cov:       np.ndarray            # (k,k) SAMPLING covariance of theta_hat (already
                                     #   divided / already an SE^2 — NOT a per-sample s^2).
    names:     tuple[str, ...]       # (k,) the registry quantity each component estimates.
    shrink:    ShrinkLaw             # how cov responds to more of THIS bench's effort
                                     #   (the allocation hook; demotes the ambiguous scalar n).
    support:   tuple[Support, ...]   # (k,) per-component domain: REAL | POSITIVE | UNIT | (lo,hi)
    family:    tuple[CIFamily, ...]  # (k,) the sampling-law family of each theta_hat component:
                                     #   NORMAL | STUDENT_T(dof) | EMPIRICAL | DEGENERATE
    cross:     Mapping[str, float] = {}   # OPTIONAL cross-bench covariance, keyed by the OTHER
                                     #   bench's quantity NAME. Empty {} = "independent of all
                                     #   others" (the honest default). For composites only (§3).
    kind:      str = ""              # provenance label: 'mean'|'median'|'ols_fit'|'pin'|
                                     #   'declared_spread'|'quantile'|'ratio'. Driver branches
                                     #   on NONE of it; it is for the store and the report.

    def is_valid(self) -> bool: ...  # ADR-0002 gate: cov PSD, theta_hat finite, family known,
                                     #   |names| == |theta_hat| == k, support/family length k.
```

The five load-bearing decisions, each **derived from what the loop actually
touches**, not inherited from any sketch:

**(D1) `theta_hat` + `cov` replace `(mean, sigma, n)`.** Read `step()`
(`neyman_driver.py:272–290`): the only statistical reach into the data is
`mu=[p.mean()]` (to evaluate `f` and its gradient) and `sigma=[p.std(ddof=1)]`,
used **only ever** recombined as `a_i/n_i = (df/dx_i)^2 · sigma_i^2/n_i =
(df/dx_i)^2 · Var(theta_hat_i)`. So the loop never needs a per-sample `sigma`
it re-divides by an `n` whose meaning differs per estimator; it needs
`Var(theta_hat_i)` **directly** — already divided, already an SE². The bench
delivers exactly that in `cov`. `cov` is a **matrix**, not a scalar, because one
bench can emit multiple correlated components (an OLS fit emits slope **and**
intercept with their off-diagonal — §4.2). This is the first place the contract
beats the maintainer's `(theta_hat, Var, n)` hypothesis: a scalar `Var` per
input structurally **cannot** carry `Cov(slope, intercept)`.

**(D2) `shrink` (a `ShrinkLaw` sum type) replaces the scalar `n`.** The Neyman
top-up needs to know, for input *i*, **how much variance one more unit of this
bench's effort buys** — the function `cov_i(effort)` and its local marginal. The
maintainer's `Var ~= V/n` is one shrink law (the mean's), promoted to a
universal truth it is not: a fit's SE has an x-leverage **floor** (more iters
never crosses it); a pin's variance is constant; a quantile's variance carries a
density-at-quantile factor. The scalar `n` is also the field that **cannot be
defined uniformly** across the suite — in the current schema it already carries
three incompatible meanings (cycles for a latency; `len(batches)=7` for a fit's
slope row **and** `iters=200` for the same fit's per-width rows, written into
*one* instance by `bench_t_row.run`; `NULL` for a pin). Demoting `n` into one
parameter of one `ShrinkLaw` variant dissolves the ambiguity: the driver never
interprets `n`; it asks the bench's shrink law for the marginal.

```python
ShrinkLaw =
    Poolwise(per_sample_var: ndarray)              # cov(n) = diag(per_sample_var)/n   — MEAN
  | QuantileLaw(p, f_at_q: ndarray, n: int)        # cov(n) = p(1-p)/(n f_at_q^2)       — QUANTILE/MEDIAN
  | RegressionLaw(resid_var, XtX_inv, design,      # cov(effort) = resid_var * XtX_inv,
                  per_point_var: ndarray|None)     #   shrinks resid_var w/ iters, floored by leverage — FIT
  | Fixed()                                        # cov(effort) = cov  (un-shrinkable)  — PIN / declared spread
  | Composed(parts: tuple[ShrinkLaw, ...])         # recurse to the steepest constituent — RATIO
```

The driver asks `shrink.marginal(cost)` — the local `d(cov)/d(effort) / cost`
in the bench's own effort currency — to rank where the next batch goes (§2). The
bench owns **which knob** its effort buys (a fit may spend effort on more
iters-per-point *or* a wider x-design; the driver passes a budget, the bench
decides — P2/P4). This is the honest answer to the brief's "what does *add more
samples* mean for a fit vs a pool": it means *spend the effort this bench's
shrink law consumes*, which the bench declares and the driver respects.

**(D3) `support` clips the reported CI to the feasible set.** A positive
latency's CI lower edge never crosses 0; a fraction's CI never exceeds 1. This
is where a symmetric `z·sqrt(Var)` would otherwise print an impossible value —
exactly where the Normal approximation is worst (near a boundary). The driver
clips and *flags* when the unclipped CI crossed a bound (ADR-0002 honesty: the
boundary proximity is surfaced, not hidden).

**(D4) `family` carries the CI multiplier honestly, per component.** A 7-point
fit is `STUDENT_T(dof=5)`, not Normal; a pin is `DEGENERATE`; a large-`n` mean
is `NORMAL`; a quantile/bootstrap is `EMPIRICAL`. This is a **per-component**
field, not a universal `dof` scalar, because — resolving an invalidating
critique — `dof`→Student-t is **only coherent for the mean (n−1) and the OLS
coefficient (n_pts−2)**; a sample quantile's CI is not a t-interval, and a
delta-method ratio is asymptotically Normal with no finite dof. Carrying
`family` per component lets the driver apply the **right** interval per input
and refuse to fabricate a t-distribution where none exists (§4.3).

**(D5) `cross` is reserved for composites and is empty by default.** The
within-bench off-diagonals (slope/intercept) live in `cov` (one bench, one
`Estimate`, `k=2`). `cross` exists only for the rare case where two **distinct**
registered inputs are statistically coupled because one is a composite built
from the other (§3, the ratio case). The honest default is `{}` — "independent
of every other bench" — which is **correct** for the leaf-eval suite as it
stands: distinct benches (a latency microbench, a separate fit) are
independent, so the global covariance is block-diagonal across benches with each
bench's own block (its `cov`) on the diagonal.

A bench that cannot honor the contract — no defensible variance, a fit that
failed — **raises** (ADR-0002 / P2 reject-don't-coerce), never returns a padded
2-sample zero-spread pool. The contract is the Port: it translates the bench's
native estimator output into `(theta_hat, cov, shrink, …)` and rejects what it
cannot honor.

### Where the contract keeps, and where it rejects, the `(theta_hat, Var, n)` hypothesis

The maintainer sketched: *"a quantity is an estimator exposing `(theta_hat,
Var(theta_hat), n)`, and `Var(theta_hat) ~= V/n` keeps the Neyman math
identical."* We **keep its spine and reject its sufficiency**, and the split is
earned, not asserted:

- **KEEP `theta_hat`** verbatim — the point `f` is evaluated at. Forced by what
  the loop does.
- **KEEP "carry `Var(theta_hat)` directly, already divided"** — this is `cov`'s
  diagonal. For a single mean input, `Estimate(theta_hat,
  cov=[[s^2/n]], shrink=Poolwise([s^2]), family=NORMAL)` **is** the sketch's
  `(theta_hat, V/n, n)`, and the driver produces byte-for-byte today's number
  (§2). Earned validation, not obedience: the mean is one consistent estimator
  and the delta method never required more.
- **REJECT the scalar `Var` as sufficient** — it cannot carry the
  slope/intercept covariance the real benches structurally produce (D1).
- **REJECT the scalar `n`** — it is the one field that means three incompatible
  things across the suite and for which `Var = V/n` is false (a fit, a
  quantile). Demoted into `ShrinkLaw` (D2), where each estimator's true
  variance-reduction law lives.
- **ADD `support` and `family`** — the two honesty axes (bounded support,
  non-asymptotic / non-Gaussian sampling law) the sketch has no slot for.

So the contract is the **minimal superset** of the sketch that (a) reduces to it
on the mean case, (b) carries the one within-bench correlation the real benches
produce, (c) names the one shrink-law degree of freedom that separates the
estimator kinds, and (d) keeps the driver estimator-agnostic. No member is
decorative; each earns its place by a thing `step()` does or a real bench's
behavior.

We considered going further — an **influence-function** representation (carry
each estimator's per-sample influence, recover any covariance by an inner
product) or a **posterior** (carry samples, let the driver integrate). Both are
**rejected on IMPLEMENTABLE-1:1 grounds**: every field of `Estimate` maps to a
real number an existing bench already computes (a mean's `s^2/n`; an OLS
`(A^T A)^{-1}` and residuals available at the `lstsq`; a declared `sigma`; an
order-statistic variance the bench can bootstrap). An influence function would
demand the pin and config-constant benches expose an estimating equation they do
not have (a P2 violation — a method the receiver cannot honor). A posterior is
strictly more than the delta-method allocator can consume — it reduces
everything to `g^T Σ g` anyway. The explicit covariance object is the named
alternative the brief floats, and it is the right one here precisely because it
is what the benches can already produce.

---

## 2. The Neyman derivation under the contract

The allocation stays valid and re-derivable. **Only the inputs change** —
from *recomputed-from-a-raw-pool* to *read-from-an `Estimate`*; the optimum's
structure is preserved, and the one place the classical closed form is replaced
is named precisely.

### 2.1 What `step()` does today (the baseline to preserve)

`neyman_driver.py:272–290`, all-means, diagonal:

```
mu          = [p.mean() for p in pools]
sigma       = [p.std(ddof=1) for p in pools]
grad        = f.gradient(mu)                # OT analytic, central-FD fallback
a_i         = (grad_i * sigma_i)^2
var_contrib = a_i / n_i                     # this input's share of Var(E[f])
var_est     = sum_i var_contrib_i           # diagonal only — Cov dropped
ci_half     = z * sqrt(var_est)
n_star_i    = sqrt(a_i / c_i) * (sum_j sqrt(a_j c_j)) / V_target
```

### 2.2 What `step()` does under the contract

The driver holds one `Estimate` per input (replacing `self.pools[i]`). Let
`theta = concat(e_i.theta_hat)` be the full evaluation point and **Σ** the joint
input covariance: **block-diagonal across distinct benches** (each bench's `cov`
is its block — independence is the `cross == {}` default), carrying the
**within-bench off-diagonal** (the slope/intercept 2×2) inside its own block.
Then:

```
g           = grad_f(theta)                 # unchanged (OT analytic / central FD)
Var(E[f])   = g^T Σ g                        # THE quadratic form — replaces sum a_i/n_i
a_i         = g_i * (Σ g)_i                  # per-input marginal contribution, now Cov-aware
ci_half     = mult_i * sqrt(Var(E[f]))      # mult per family (z / t / empirical), §4.3
estimate    = f(theta)
converged   = Var(E[f]) <= V_target  AND  (no contributing input is small-n-unreliable)
                                     AND  (the binding-margin guard of §4.1 passes)
```

**Exactness claim, substantiated.** When every input is a mean, Σ is diagonal
with `Σ_ii = s_i^2/n_i`, and `g^T Σ g = sum_i g_i^2 · s_i^2/n_i`, which is
**bit-for-bit** today's `sum_i a_i/n_i` (substitute `Var(theta_hat_i) =
s_i^2/n_i`). The off-diagonal cross-terms `2 g_i g_j Σ_ij` are **pure addition**
the old code dropped — and they are nonzero exactly for the within-fit
slope/intercept pair (§4.2). So the bound is **at least as valid and strictly
more faithful**; no step assumed the mean.

### 2.3 The allocation, restated — where the closed form holds and where it is replaced

The Neyman program is unchanged: minimize total cost `sum_i c_i · effort_i`
subject to `Var(E[f]) <= V_target = (h/mult)^2`. Its KKT stationarity condition
is the **general** Neyman optimum:

> at the optimum, the marginal variance-reduction **per unit cost** is equal
> across all funded inputs: `d Var(E[f])/d effort_i / c_i = lambda` (a common
> shadow price) for every funded *i*.

This holds for **any** differentiable, monotone-decreasing shrink law — it does
**not** require `Var = V/n`. The contract supplies exactly
`d Var(E[f])/d effort_i = g_i^2 · d(Σ_ii)/d effort_i` (plus the cross-term's
derivative) via `shrink.marginal`. So the optimum is computable for **every**
case, and three regimes fall out:

- **`Poolwise` (mean) — the closed form is preserved exactly.** With
  `Σ_ii(n) = s_i^2/n_i`, `d Σ_ii/d n_i = -s_i^2/n_i^2`, and equalizing the
  per-cost marginal recovers `n_i^* ∝ sqrt(a_i/c_i)` — the exact formula at
  `neyman_driver.py:286–289`. **This is the maintainer's sketch, earned**: for
  every mean input the generalized rule reproduces the current allocation
  line-for-line. *(KKT-point exact; the greedy per-batch top-up is the same
  damped approach to it the current `topup` block implements.)*
- **`RegressionLaw` (fit) — the closed form is replaced by the general
  condition.** A coefficient's SE has an x-leverage floor: `Var(slope) =
  resid_var/Sxx`, where more iters shrink `resid_var` but the `1/Sxx` leverage
  term is fixed unless the bench widens the x-design. There is **no single `n`**
  for which `Var = V/n`. So `n_i^*` is undefined and the driver does **not**
  invert a closed form; it funds by the general per-cost-marginal-equalization,
  which **saturates** when `resid_var` stops dominating — so the loop correctly
  **stops funding a fit whose SE is leverage-limited rather than
  residual-limited** (a fit the `V/n` assumption would fund forever). This is
  the brief's "states precisely where it breaks and what replaces it": it breaks
  for non-`V/n` laws, and explicit marginal allocation replaces the closed form,
  agreeing with it on the `Poolwise` subset.
- **`Fixed` (pin / declared spread) — drops out of allocation, for the right
  reason.** `d Σ_ii/d effort = 0`, so a pin never enters the equalization (no
  finite budget reduces it) — reproducing today's `a<=0 => n_star=n` "don't
  sample dead inputs" branch (`neyman_driver.py:290`), but because its variance
  is **irreducible**, not merely because `a==0`. A `Fixed` input with `cov>0`
  (a declared engineering-judgement spread) **still contributes** `a_i` to the
  bound (so the CI honestly reflects the prior uncertainty) while getting no
  allocation. *(Honest edge: if the `Fixed` inputs' contribution alone exceeds
  `V_target`, **no** allocation converges; the driver must say so loudly —
  ADR-0002 — rather than spin.)*

### 2.4 The stop criterion and its one honest amendment

`var_est <= V_target` is unchanged in form; `var_est` is now the quadratic
form. The amendments, each forced by a critique:

- the multiplier is **per-family** (§4.3), not a fixed `z`, so a small-`n`
  fit's CI is widened by Student-t rather than under-stated by `z=1.96`;
- convergence is **gated** on `family` (no contributing input is
  small-`n`-unreliable) **and** on the binding-margin guard (§4.1) — so
  "converged" cannot fire on a `Var(E[f])` that is itself a lie about the true
  spread (the `min()` kink, §4.1) nor on a variance that is itself badly
  estimated.

This is the strongest honest claim the math earns; it is stated as a *variance
budget on E[f]* that becomes a frequentist CI only when no input is a
`Fixed` declared-prior spread (§7, the category-mixing caveat).

---

## 3. The case-handling table

Every case differs **only** in how the bench fills the contract's fields — never
in the field set. The driver's view is identical for every row.

| Case | `theta_hat`, `cov` | `shrink` | `support` / `family` | Notes & where it beats `(θ̂,V,n)` |
| — | — | — | — | — |
| **MEAN** (a true arithmetic-mean latency) | `[mean]`, `[[s²/n]]` | `Poolwise([s²])` | POSITIVE / NORMAL (n≫30) | The degenerate case; the sketch **is** this. Reduces to today's driver byte-for-byte. **Note:** the leaf-eval latency benches do **not** currently report a mean — see MEDIAN. |
| **MEDIAN / QUANTILE** (`tau_io`, `wakeup`; any p99; the p50 headline today) | `[q_p]`, `[[p(1−p)/(n·f̂(q)²)]]` | `QuantileLaw(p, f̂(q), n)` | POSITIVE (latency) or UNIT (fraction) / EMPIRICAL or NORMAL-asymptotic | Var is the **order-statistic** law, **not** s²/n; `f̂(q)` is a kernel density the bench estimates (or a bootstrap). The median is `p=0.5`. **This is what the latency benches actually are** (they report `np.median`); the contract matches the bench, the sketch's `V/n` does not (§7.A). |
| **REGRESSION fit** (`t_row`+`iota` from one staged fit; `t_row_bare`+`T_disp` from the cpp fit) | `[intercept, slope]` (k=2), `cov = s_resid²·(AᵀA)⁻¹` (full 2×2, off-diagonal carried) | `RegressionLaw(s_resid², (AᵀA)⁻¹, design, per_point_var)` | POSITIVE / STUDENT_T(dof=n_pts−2) | One fit → one `Estimate` with two components and the **−0.81** off-diagonal. The `n` ambiguity dissolves (neither 7 nor 200 is in `cov`; the SE comes from `resid_var` and the x-design). **Beats the sketch:** no single `n` gives `Var=V/n` for a slope. The two registry names map to the two **components** of one estimate (§4.2). |
| **PIN — true constant** (`n_gen`) | `[3]`, `[[σ²]]` (σ tiny, the declared layout spread) | `Fixed()` | POSITIVE / DEGENERATE | `a_i ≈ 0`: no allocation, ~0 bound contribution. |
| **PIN — declared spread** (`T_disp` σ=2, `B_op` σ=64, `LPD` σ=25, `tmsg` σ=0.5) | `[value]`, `[[σ_declared²]]` | `Fixed()` | POSITIVE / NORMAL (a prior) | **Contributes** `a_i` to the bound (the bound honestly rests on the prior) but **un-shrinkable** by sampling (no bench reduces an engineering-judgement prior; the manifest seeds it, a sole-workload run only confirms). Fixes the latent store bug where `stddev_samp` over one logged value returns NULL→0, **discarding** the declared σ (§5). |
| **PIN-now / measurable-later** (`B_op` as a saturated-rows histogram, `LPD` as a leaf-count histogram, `R_gen`/`g_core` as a C++ rate) | `Fixed()` **today** → `QuantileLaw`/`Poolwise` once instrumented | the definition's registered `kind` flips when the bench gains a real `run()` | per the upgraded estimator | Keeps **"cannot be reduced"** (a true `Fixed`) distinct from **"not yet measured"** (a pin awaiting its bench). This is exactly the `Grounded.needs_measurement` / manifest `trusted` distinction the code already tracks, now typed in the shrink law. |
| **RATIO / composite** (a quantity defined as `h(constituents)` — e.g. an aggregate `N_gen·R_gen` were it ever registered as one input) | `[h(θ̂s)]`, `cov = J·Σ_constituents·Jᵀ` (J = h's Jacobian) | `Composed(parts)` — recurse to the steepest constituent | inherited / family of the dominant constituent | The `Estimate` of a composition **is** a delta-method output, so the contract is **closed under composition**: a model output can feed a higher model. Carries `cross` to **every** shared constituent (the §3 invariant) so `g^T Σ g` does not double-count. `dps` itself is `f` and is **not** registered (confirmed: no `dps` definition row), so this row is for completeness + the producer-cap aggregate; the contract survives such a thing appearing as an input. |
| **NON-ASYMPTOTIC small-n** (the 7-point fit; a few-replicate C++ `R_gen`; an `n=1` pin) | as the case above | as the case above | family carries `STUDENT_T(dof)` or an `EMPIRICAL` small-n flag | Not a separate row — an **attribute** any case carries through `family`. The driver uses the family's multiplier and **gates convergence** on it (§4.3). |
| **NON-SMOOTH `f`** (the `min()` kink) | — | — | — | **Not an `Estimate` field** — it is a property of `f`, not of any input estimator. Handled at the driver (§4.1). The contract's role is to carry each input's full `cov` so the driver can compute each arm's capacity variance honestly. |

---

## 4. The two subtleties that bite

### 4.1 The non-smooth `min()` — the highest-risk issue, handled at the driver, not the contract

`f = min(GENERATION, SERVE, TRANSPORT)`. This is a property of `f`, so it is
**not** a per-input contract field; it is the driver's and the model's concern.
The contract's contribution is that by carrying each input's full `cov` (not a
point `sigma`), the driver can compute the variance of **each arm's** capacity
and the binding margin honestly. But the contract alone does **not** make the
bound honest at a tie — and the critique that flagged this as *invalidating* a
contract-only treatment is **correct**. Here is the full, substantiated picture
and the driver-level response.

**What actually happens at the seed point (executed, not asserted).** For
`model_zmq_baseline` at the seed: `SERVE` binds at **428.28 dps**, `GENERATION`
(producer `N_gen·R_gen = 456`) is second at a **6.5%** margin
(`model_capacity` is the same shape: SERVE 419.76, margin 8.6%). The OT analytic
gradient at that point (the live `WRN - Switch to finite difference` is OT
failing to differentiate `min()` and falling back to FD) gives:

```
df/dN_gen = 0     df/dR_gen = 0        (producer arm — ZERO Neyman weight)
df/dtau_io = -0.358   df/dT_disp = -0.358   df/dt_row = -91.7   (serve arm — binding)
```

So the producer arm's inputs get **zero allocation** even though a small upward
revision of any serve input flips the binding arm. Worse, **the reported
variance is a lie**: I ran a 4·10⁶-draw Monte-Carlo of `min(Normal(456, 60),
Normal(428.28, 2))` (producer wide because `R_gen` is a single costly C++
figure; serve tight) and got:

- `E[min] = 415.7` vs the delta-method `E[f] = min(456, 428.28) = 428.28` — a
  **Jensen bias of +12.6 dps, optimistic** (`min` is concave, so `min(E[·])`
  over-states `E[min(·)]`; this is **first-order** O(σ) at a kink, not the
  O(σ²) a smooth `f` gives);
- `sd[min] = 25.6` vs the single-binding-arm delta-method `sd = 2.0` — **12.8×**
  larger, because the realized binding arm switches stochastically;
- `P(producer is the realized min) = 0.32` — a full third of the probability
  mass lives on an arm whose ±(3·60) dps the reported CI entirely ignores.

`g^T Σ g` cannot represent `Var(min(·))`: it is a Clark-1961 mixture moment, a
**functional of `f`**, not a per-input estimate — `theta_hat`/`cov` are
per-input quantities. So the danger is the over-permissive **false-SAT** the
project's faithful-model discipline warns against: `converged := var_est <=
V_target` could fire at `E[f]=428.28` with CI `±3.9` while 32% of the mass is
producer-bound.

**TaylorExpansionMoments does NOT rescue this** (resolving the Design 3/4
critiques against Designs 3 and 4, which proposed it as the kink-validity
signal). `_second_order_mean()` (`neyman_driver.py:233–250`) asks
`getMeanSecondOrder()`, a 2nd-order Taylor expansion assuming `f ∈ C²`. At a
`min()` tie the Hessian is **0 a.e. and a Dirac at the boundary**, so the tool
is undefined exactly where it is summoned. I ran it: it returns `mean 2nd order
= 432.4` — moving the mean the **wrong direction** (+4.2, vs the true Jensen bias
of −12.6) and reporting `sd = 51.4` from an FD-Hessian. It is **blind to the
kink**. It must **not** be cited as the kink-validity signal; the existing
`_second_order_mean` hook stays only as a smooth-region curvature diagnostic
(away from a tie, where it is valid), never as the tie detector.

**The driver-level response (three honest mechanisms):**

1. **Binding-margin diagnostic + a `kink_regime` flag.** The driver computes,
   per arm, `margin = (this_arm_capacity − binding_capacity)/binding_capacity`,
   and the *variance* of each arm's capacity from the carried `cov`s. It raises
   `kink_regime` when any non-binding arm is within a threshold (a
   statistically-plausible tie, e.g. `margin < k · arm-capacity-CI`). The 6.5%
   seed margin **fires** this flag.

2. **In the kink regime, replace `g^T Σ g` with a mixture estimate of `E[f]`
   and `Var(f)`.** When `kink_regime` is set, the driver does **not** report the
   single-arm `g^T Σ g`; it computes `E[f]` and `Var(f)` by a cheap Monte-Carlo
   of `min(arm₁,…,arm_k)` over the input `Estimate`s' distributions (the
   `Estimate`s carry `theta_hat`, `cov`, `family`, `support` — everything needed
   to sample each arm), **or** a stated-temperature soft-min. It reports
   **that** as `var_estimate`/`ci_halfwidth`, and states plainly that the
   delta-method is **suspended** (not corrected) and the point estimate `f(μ̂)`
   is a biased, variance-understated estimator of `E[f]`.

3. **A convergence guard.** Convergence is **refused** while
   `P(arg-min flips) > α` (a small probability, computed from the same joint
   distribution) or while the importance-factor mass on a non-binding arm
   exceeds ε. This makes the over-permissive false-SAT impossible (ADR-0002 loud
   at the strongest surface).

   And the allocation in the kink regime funds **both** contending arms'
   inputs (a sub-gradient convention: the realized `min` can be either arm, so
   both arms' uncertainty is live) — the necessary-but-insufficient piece the
   contract-only designs got right, now paired with the CI/convergence fix that
   is the actual cure.

Away from a tie (`margin` comfortably large) the analytic gradient is honest,
the non-binding arms' `df/dx = 0` is **correct** (those inputs genuinely do not
move the bound), and behavior is exactly today's. The honest statement at the
tie is: *the delta-method Var is suspended; here is the mixture Var and the
arg-min-flip probability* — surfaced loudly, not a silent zero.

**A second, distinct non-smoothness — the `pad(B)` sawtooth — stated for
completeness.** The physical serve curve uses `B_eff = pad(B)`, a step function
with jumps at bucket edges (`serve_sawtooth()` exhibits it), so `d(serve)/dB`
through `pad` is 0 a.e. and undefined at edges. The symbolic `THROUGHPUT_EXPR`
uses bare `B` (smooth), so this is **latent only while `B_op` is pinned to a
full bucket** (256, an edge). It is inert today (`B_op` is a `Fixed` pin at a
bucket edge); if `B` ever becomes a swept/allocated quantity, `df/dB` is
meaningless and a bucket-snap-aware treatment is required. The binding-margin
machinery covers the `min()` kink; the `pad()` sawtooth is a **separate** latent
non-smoothness, named here so the loop's honesty about "delta-method validity at
the non-smoothness" names **both**, not one.

### 4.2 The slope/intercept correlation — carried structurally, attributed to the right model

One OLS fit emits **one** `Estimate` with `k=2` and a full 2×2 `cov` whose
off-diagonal is `Cov(slope, intercept) = −x̄·s_resid²/Sxx`. I verified on the
real 7-point design `[32,64,128,192,256,384,512]`:

```
Var(slope)      = resid_var / Sxx                          (exact)
Var(intercept)  = resid_var · (1/n + x̄²/Sxx)               (exact)
Cov(slope,int)  = −x̄ · resid_var / Sxx                     (exact)
correlation     = −0.8114                                  (strongly negative)
```

Dropping this off-diagonal is **not** safe. The driver assembles the 2×2 as a
**block** of the global Σ (the two registry names map to the two **components**
of one `Estimate`), and `g^T Σ g` picks up the cross-term `2·g_slope·g_int·Cov`
automatically. `bench_iota.measure` already **delegates** to
`bench_t_row.measure` (`bench_iota.py:56–57`) — they are literally one fit — so
the contract just stops splitting one fit into two uncorrelated scalars.

**Where the correlation is LIVE vs INERT — the attribution the source designs
got wrong, corrected against the actual `INPUT_NAMES`:**

- **`model_capacity` — LIVE.** Its serve term is
  `(iota_us + slope_us·B_op + tau_io_us)` (`model_capacity.py:89`), consuming
  **both** `iota_us` and `slope_us` from the **same** staged fit
  (`leaf_eval_grounding`: `results_nopad.json fits.staged`, intercept 94.58 and
  slope 4.317, both R²=0.998). Here `g^T Σ g` **materially** corrects the
  serve-stage variance: with `Cov<0` and both inputs entering the serve
  denominator with same-sign sensitivities, the dropped cross-term is positive,
  so the diagonal-only sum **understates** serve variance. **The cov-matrix
  earns its place on `model_capacity`.**
- **`model_zmq_baseline` — INERT (and the headline several source designs
  asserted is false here).** Its serve cycle is
  `(T_disp + tau_io + wakeup + B·t_row)` (`model_zmq_baseline.py:117/127`).
  **`iota` is not an input to this model.** The intercept term is `T_disp_us`
  (the dispatch floor, 68.84, measured by a **different** fit — the
  `fully_device` variant via `bench_t_disp`, `mlp_lowlatency` decomposition),
  and the slope is `t_row_us` (the staged fit). So for the brief's **named
  primary model**, the slope/intercept cross-term multiplies `df/d_iota = 0` and
  contributes **exactly zero** to `Var(E[f])` regardless of the −0.81
  correlation. The cov-matrix is **harmless** here (block-diagonal,
  off-diagonal unused), not load-bearing. A design that markets the correlation
  as live in `zmq_baseline` is conflating the two models; this note states which
  model each shared-fit claim applies to.

**The decomposed-intercept subtlety (`T_disp` ⊂ `iota`).** Physically,
`iota = dispatch_floor + output_pull + input + residual` (94.58 = 68.84 + 9.14 +
5.52 + residual), so `T_disp` (68.84) is a **sub-component** of the staged
intercept — yet it is **measured by a separate `fully_device` fit**, not read
off the staged fit's intercept. The contract's `cross == {}` block-diagonal rule
therefore treats `T_disp` and `t_row` (and `T_disp` and `iota`) as
**independent**, which is **defensible because they are different fits** — but it
is a *modelling choice*, not a rigorous fact: the two fits time the same
hardware and are physically non-independent. This is named in §7 as a known
approximation. The `cov_group` / shared-`Estimate` mechanism pairs **only**
co-fit components: `(t_row, iota)` from the staged fit, `(t_row_bare, T_disp)`
from the cpp fit — and must **not** pair `t_row` with `T_disp` (different
variants). A migration that grouped them would fabricate a covariance that does
not exist.

### 4.3 Non-asymptotic and non-Gaussian — `family`, not a universal `dof`

`family` carries the CI multiplier per component, and the choice is honest about
what each estimator's sampling law actually is — resolving the critique that a
universal `dof`→Student-t is incoherent off the Gaussian family:

- `family = NORMAL` (a large-`n` mean/quantile) → multiplier `z`.
- `family = STUDENT_T(dof)` → multiplier `t_{dof,1−α/2}`; **legitimate only for
  the mean (dof=n−1) and the OLS coefficient (dof=n_pts−2)**, the two cases
  whose pivot genuinely is Student-t. A 7-point fit is `dof=5` → `t≈2.57` vs
  `z=1.96`, a 31% wider CI **honestly** reported (today's fixed `z` at
  `neyman_driver.py:380–384` under-states it).
- `family = EMPIRICAL` (a sample quantile, a bootstrap) → the bench's own
  interval (e.g. a bootstrap percentile), **not** a t-interval — a sample
  quantile's CI is not in the t-family, so fabricating a `dof` for it would be
  the over-claim ADR-0009 forbids.
- `family = DEGENERATE` (a pin) → no sampling interval.

**The combined bound over inputs of differing family** is stated honestly: when
the inputs carry different families, the pivot
`(f(θ̂)−E[f])/sqrt(g^T Σ̂ g)` is a Normal-combination over a sum of scaled
chi-squares (a Behrens-Fisher-like object), **not** an exact Student-t. The
driver uses a **conservative** multiplier (the most-conservative family among
the binding inputs, e.g. `t_{min-dof}`) and **labels** it as a conservative
heuristic, not an exact CI — so "converged" is announced as *"variance budget
met under a conservative multiplier,"* not a false exactness claim.

Two further honesty points the critiques surfaced, recorded as obligations on
the **bench's** `cov`/`shrink`, not the driver:

- **The fit SE is over per-width medians, not raw timings, and the medians are
  heteroscedastic.** `bench_t_row` fits `_fit_line` on the 7 per-width *medians*
  (each a median of `iters·repeat` timings, with its own width-dependent
  variance). The textbook `resid_var/Sxx` conflates per-median measurement noise
  (shrinks ~1/iters) with XLA-curve **lack-of-fit** (a bias that does **not**
  shrink). So `RegressionLaw` must (a) carry the lack-of-fit / R² gate (the
  bench already computes `r2`) and flag the family when the fit is non-linear,
  and (b) ideally propagate each median's own sampling SE into a
  **weighted-LS** SE (the bench has the per-width iter data) — separating
  measurement noise from misfit, so a bowed curve does not silently report a
  too-small SE. The plain `resid_var/Sxx` is a **lower bound** on the true slope
  variance and must be labelled as such if the weighted form is not implemented.
- **`dvar`/`marginal` for a fit must not promise variance the work cannot buy.**
  If the lack-of-fit term dominates, more iters never crosses the leverage/misfit
  floor; `RegressionLaw.marginal` must return ~0 there (or route effort to
  *design points*, the knob that genuinely lowers `1/Sxx`), so the allocator
  does not pour budget into the dominant-cost fit chasing a floor it cannot
  reach.

---

## 5. The schema delta

**What a stored measurement is under the contract:** one `benchmark_instance`
row **is** one realized `Estimate` — the bench's *computed* estimate, persisted
whole. The store stops being an estimator (today `latest_aggregate` computes
`avg, stddev_samp, count` over an instance's samples) and becomes a **record of
the bench's estimate**. This is the load-bearing schema consequence, and it is
forced, not stylistic:

**The store provably cannot reconstruct the variance for any non-mean
estimator (read, not assumed).** `latest_aggregate` (`bench_store.py:280–316`)
does `SELECT avg(value), stddev_samp(value), count(value)`:

- For a **mean** it recovers `Var = stddev_samp²/count` — fine.
- For a **fit**, `bench_t_row.run` logs the scalar slope (`sample_size=7`) **and
  the 7 per-width medians** (`sample_size=200`) into **one** instance, so
  `avg(value)` averages `4.317` with seven 4-digit medians (a meaningless
  number), `stddev_samp` over the blended set is not `SE(slope)`, and
  `Cov(slope, intercept)` is recoverable from **nothing** in the sample table.
  `_fit_line` itself returns only `(intercept, slope, r2)` and **discards** the
  design matrix `A` and residuals — so the covariance is **not even computed
  today**, let alone stored. (A correction to a source design's wording:
  `bench_t_row.measure` does **not** "already hold" the covariance; it holds the
  *inputs* — `batches`, `per_width_median_us` — to compute it, and the
  conversion **adds** that computation. Either extend `_fit_line` to return
  `(AᵀA)⁻¹` and `resid_var`, or compute the covariance in the bench. **Audit
  obligation:** `_fit_line` lives in the shared AZ helper
  `chocofarm/az/bench/bench_mlp_lowlatency.py:155` and is called by
  `bench_t_row`, `bench_iota`, and `bench_t_disp` — changing its return triggers
  the base-method-override audit; propagate to all three call sites and verify
  behaviorally.)
- For a **declared-spread pin**, `stddev_samp` over one logged value returns
  NULL→0, **discarding** the declared σ (B_op's 64 lives only in `Grounded`,
  never reaches the DB). So even the trusted path is silently wrong for fits and
  spread-pins **independent of the driver**.

**The delta (additive; the three tables and the existing `sample_size` column
survive):**

1. **`benchmark_instance` gains `estimate jsonb`** — the serialized `Estimate`:
   `{theta_hat:[…], cov:[[…]], names:[…], shrink:{law, params}, support:[…],
   family:[…], cross:{…}, kind}`. This is the **SSOT** of the measured object
   (ADR-0012 P1 single-home: the variance/covariance has one home, the instance
   row, never two derivations that must agree). The manifest's TRUST path reads
   this column directly — so `SE(slope)` and `Cov(slope,intercept)` survive (in
   `cov`), which `avg/stddev_samp` provably cannot recover. A **jsonb** column
   (not N typed columns) because `shrink` is a sum type of varying arity
   (`Poolwise` carries a per-sample-var vector, `RegressionLaw` a matrix +
   design) and `support`/`family`/`cross` are a small open set — typing each as
   a column would re-author the sum type in DDL; the typed `Estimate` dataclass
   (P8) is the SSOT of the shape, the jsonb is its serialization, validated on
   read by `Estimate.is_valid()` (ADR-0002).

2. **`benchmark_sample` is structurally unchanged** and stays the **raw-readings
   provenance** (the `per_cycle_us` pool, the per-width medians) for audit and
   re-analysis — but it is **no longer the variance authority**. **De-dup
   obligation (forced by a critique):** today several `run()`s log the headline
   scalar **and** the raw pool into one instance, corrupting `latest_aggregate`'s
   count (tau_io writes the median *and* ~2000 readings; t_row writes the slope
   *and* the medians). Under the contract the `estimate` jsonb is the SSOT, so
   the headline scalar must **not** be double-logged as a sample row — either
   stop logging the headline as a sample (keep it only in the jsonb) or move it
   to `config`, so the legacy `latest_aggregate` fallback path (used during
   migration) stays correct for the inputs that are still genuine means.

3. **`sample_size`'s meaning is PINNED** to **the number of raw readings behind
   one sample** (the per-sample `n`), and **only** that — `NULL` for a pin. The
   fit's "7 design points" and its SE no longer live here at all; they live in
   `estimate.shrink` / `estimate.cov`. This dissolves the "same column, three
   meanings" defect by demoting `sample_size` to its one honest meaning.

4. **`benchmark_definition` gains `estimator text`** (one of
   `mean|median|ols_fit|pin|declared_spread|quantile|ratio`) — the
   definition-level declaration of which estimator kind a quantity is. It is
   metadata (the math reads only `estimate`), but it makes the registry
   self-describing (ADR-0002): a re-measure that changes the estimator kind (a
   pin promoted to a histogram) is a **definition change**, surfaced, not
   silent.

`manifest.quantity()` changes from returning `(mean, sigma, n)` to **also**
carrying the deserialized `Estimate` (reading `instance.estimate` when present,
**falling back** to reconstructing a `Poolwise`/`median` `Estimate` from
`latest_aggregate` for legacy instances — so old data still resolves). The
existing `.mean/.sigma/.n/.trusted` stay as a projection (`mean = theta_hat[0]`,
`sigma = sqrt(cov[0,0])`) so every downstream `.mean`/`.sigma` reader is
unchanged. The seed path returns a `Fixed`-law `Estimate` built from the bench's
`get_seed()` `Grounded` (mean → `theta_hat`, `sigma` → `cov` diagonal, support
from units, `family=NORMAL` as a prior).

---

## 6. The migration path

No flag-day. The contract is additive at every seam; each of ~30 benches
migrates independently because the manifest can reconstruct a legacy `Estimate`
from `latest_aggregate` and from `get_seed()`.

- **Phase 0 — contract + store, zero behavior change.** Add `estimate.py` (the
  `Estimate` dataclass + `ShrinkLaw` + `Support`/`CIFamily` enums; ADR-0006
  header; P1 single-home; P8 typed SSOT). Add the three schema additions
  (`benchmark_instance.estimate jsonb`, `benchmark_definition.estimator text`,
  the pinned `sample_size` meaning) via idempotent `ALTER … ADD COLUMN IF NOT
  EXISTS` in `_SCHEMA_SQL`. Add `latest_estimate(name)` reading the jsonb.
  Nothing consumes it yet.

- **Phase 1 — manifest as the seam.** `manifest.quantity()` carries the
  `Estimate`; the TRUST path reads `instance.estimate` when present, else
  reconstructs a `Poolwise`/`median` `Estimate` from `latest_aggregate`; the
  SEED path builds a `Fixed`-law `Estimate`. The `(mean, sigma, n, trusted)`
  4-tuple is preserved as a projection. This makes every downstream consumer
  `Estimate`-capable without touching them yet.

- **Phase 2 — driver dual-mode.** Add `NeymanDriver.set_estimate(i, Estimate)` /
  `set_estimates_by_name` **beside** `add_samples`. `step()` prefers
  `self.estimates[i]`, else falls back to wrapping the raw pool as a `Poolwise`
  `Estimate` (so a pool-fed driver and an `Estimate`-fed driver agree on the
  mean case — the confirmed fixed point). Replace the diagonal sum with `g^T Σ
  g` (equal to today's sum for diagonal Σ → no regression on existing models).
  Land the binding-margin / `kink_regime` mixture path (§4.1), the per-`family`
  multiplier, and the convergence guard. `run()`'s `samplers[i](k)` becomes
  `measurers[i](budget) -> Estimate`. Keep `add_samples` as a thin
  `Poolwise`-wrapping shim so a mid-migration caller still works.

- **Phase 3 — benches, one at a time, highest-Neyman-priority first.** Each
  `measure()` returns an `Estimate`; each `run()` logs `instance.estimate`
  **alongside** (de-duped, §5) the raw provenance rows. Order: **the fit benches
  first** (`bench_t_row` + `bench_iota` as one shared-fit `Estimate` with the
  −0.81 off-diagonal; `bench_cpp_inproc_port_t_row_bare_us` + `bench_t_disp` as
  the second pair) — they are where the current code is **wrong** and the bound
  craters, so they earn the change; then the **latency/median** benches
  (`QuantileLaw`, p=0.5, with the median's order-statistic variance — **not**
  s²/n); then the **pins** (`Fixed`, recovering B_op's declared spread). An
  un-migrated bench still resolves via the Phase-1 legacy reconstruction.

- **Phase 4 — delete the coercion.** Remove `untrusted_drive._per_sample`'s
  longest-numeric-list heuristic (lines 80–98) and the 2-sample zero-spread pad
  (lines 119–121) — the documented concrete symptom (for `t_row`'s dict it picks
  `batches=[32..512]`, the row-count x-axis, because `per_width_median_us` is a
  dict and skipped, so the driver multiplies `df/dt_row=−91.7` against ~224 and
  the bound craters). `_make_sampler` becomes a one-liner: `measurer(budget) =
  mod.measure() -> Estimate`, consumed directly; no guessing which list is the
  estimate, because the bench **declares** it (P2: reject, don't guess). A pin is
  now a `Fixed`/`DEGENERATE` `Estimate`, not a faked pool; a bench exposing no
  valid `Estimate` is a loud `is_valid()` failure.

  **Also migrate `throughput_bound.py` and `transport_sweep.py`'s fabricated
  2-point pilot** (`{mean−sigma, mean+sigma}` per input, `driver.add_samples`,
  `throughput_bound.py:91–101`, `transport_sweep.py:283/310`) to pass the
  manifest's `Estimate` straight to `set_estimate`. **Migrate it on its true
  merit, not a fabricated bug.** A source design claimed this pilot has a latent
  `/2` bound bug; I executed it and **refute** that: the 2-point set
  `{mean−sigma, mean+sigma}` has sample std `sqrt(2)·sigma` (not `sigma`), so
  `a_i/n_i = grad²·(2σ²)/2 = grad²·σ²` **exactly** — the `/2` is cancelled by the
  `sqrt(2)` std inflation, and the Neyman proportions are preserved (every `a_i`
  carries the identical factor). There is **no `/2` bug**. The real reasons to
  replace it: it is an **opaque hack** smuggling `(mean, sigma)` through the pool
  API, and its inline comment in `throughput_bound.py` is **wrong** — it claims
  *"its sample-std EXACTLY the grounded sigma"* when the actual sample std is
  `sqrt(2)·sigma`. Per ADR-0009, substantiating a real cleanup with a fabricated
  numerical error is exactly the over-claim the project forbids; the cleanup
  stands on the opacity and the wrong comment.

- **Docs (ADR-0005, part of the delivery).** This note is the spec. If the
  contract lands, ADR-0012 P8 ("the typed signature is the contract's SSOT")
  gains a worked instance (the `Estimate` contract) — a dated append, not a
  rewrite. No ADR "Revisit when…" trigger fires from a design note alone;
  implementation phases would (e.g. ADR-0002's mechanization-by-append when the
  `is_valid()` gate lands).

---

## 7. Honest accounting — rigorous vs modelling choice

Per claims-measured-vs-interpreted, what this contract makes **rigorous** and
where it rests on a **modelling choice** or a **least-bad option**:

**Rigorous (proved against the code / executed):**

- The delta-method bound `Var(f(θ̂)) ≈ g^T Σ g` for any consistent estimator
  with sampling covariance Σ — the multivariate delta method; the mean is one
  consistent estimator and no step required it. Reduces to today's diagonal sum
  bit-for-bit on the all-means case.
- The Neyman optimum reduces to `n_i^* ∝ sqrt(a_i/c_i)` for `Poolwise` inputs
  (KKT marginal-per-cost equalization), and the general first-order condition
  replaces the closed form for non-`V/n` laws.
- The slope/intercept covariance: `Cov = −x̄·resid_var/Sxx`, correlation −0.8114
  on the real design — carried in `cov`, not dropped.
- The fabricated 2-point pilot has **no** `/2` bug (its std is `sqrt(2)·σ`).

**Modelling choices / least-bad options (named, not papered over):**

- **`A.` The latency benches report a MEDIAN, so the "MEAN" mapping is a
  modelling choice that must be corrected to QuantileLaw(p=0.5).** The benches
  return `np.median`; `Var(median) ≈ 1/(4 n f(median)²)`, the order-statistic
  law, **not** `s²/n`. The contract handles this cleanly (the MEDIAN/QUANTILE
  row), and the `(θ̂, V/n, n)` sketch is **not** earned for latencies — only for
  a true arithmetic mean, which no timing bench currently produces. The
  least-bad alternative if a bench prefers `Poolwise` is to switch its headline
  to a trimmed mean; that is the bench's choice, declared in `kind`.
- **`B.` The `min()` kink cannot be folded into the per-input contract.** This is
  the one case the harmonized interface **does not** harmonize, and §4.1 says so
  plainly: `Var(min(·))` is a functional of `f`, not a per-input estimate, so it
  is handled at the driver (mixture MC + binding-margin guard), and `f(μ̂)` is a
  biased (Jensen, +12.6 dps at the seed), variance-understated (12.8×) estimate
  of `E[f]` in the tie regime. The least-bad option — suspend the delta-method
  and report the mixture Var with the arg-min-flip probability — is honest, not a
  silent zero, but it is **more machinery than a clean closed form**, and the
  point estimate's bias is a real cost the loop must surface.
- **`C.` `T_disp` ⊥ `t_row` (and `T_disp` ⊥ `iota`) is block-diagonal by the
  different-fits rule, but they are physically non-independent** (different fits,
  same hardware). Treating distinct fits as independent is defensible and is the
  `cross=={}` default, but it is a modelling choice; a future cross-fit
  covariance would tighten it.
- **`D.` The combined CI over differing families is a conservative variance
  budget, not an exact CI** (Behrens-Fisher); and when a `Fixed` declared-prior
  spread contributes, the interval mixes a frequentist sampling variance with a
  declared prior variance — so the honest label is **"variance budget on E[f]"**,
  becoming a frequentist CI only when no input is a declared-prior `Fixed`. The
  driver should surface the irreducible-prior floor (the sum of `Fixed` `a_i`) as
  its own line, distinct from the shrinkable sampling variance.
- **`E.` The fit SE over per-width medians is heteroscedastic and conflates
  measurement noise with lack-of-fit** (§4.3); plain `resid_var/Sxx` is a lower
  bound on the true slope variance unless a weighted-LS SE is used. The contract
  carries the obligation; the bench owns the implementation.

**The one highest-risk open question** (carried forward to implementation): the
`min()`-kink regime needs a *cheap, faithful* `Var(min)` estimator the driver
runs every step — a few-thousand-draw Monte-Carlo over the input `Estimate`s, or
a soft-min with a principled temperature. The Monte-Carlo is straightforward but
adds a stochastic element to a previously-deterministic `step()` (reproducibility
and per-step cost); the soft-min is deterministic but introduces a temperature
that is itself a modelling choice whose bias must be characterized. Which one,
and how its own error is bounded, is the load-bearing decision the contract
**enables** (by carrying each arm's full `cov`) but does **not** settle.
