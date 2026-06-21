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
- **Consolidation (2026-06, dated append per ADR-0005 Rule 8):** this revision
  integrates an external statistical review (a literature-cited critique of the
  allocator and this contract) that broadly endorses the `Estimate` contract and
  adds four material refinements, each **re-executed here, not relayed** (ADR-0009
  — a formula is integrated only with its own reproduced number, never on the
  reviewer's authority): **(1)** the §4.1 `min()` kink resolves to the
  **Clark-1961 closed form** — deterministic, O(1), no Monte-Carlo, no soft-min
  temperature — which **closes the §7 open question** (the MC-vs-soft-min fork was
  a false choice); **(2)** the §2 allocation is a **cost-constrained c-optimal
  experimental design solved as an SOCP** (Sagnol), of which the closed-form
  `sqrt(a_i/c_i)` ratio is the diagonal/independent special case; **(3)** a new
  **robustness axis** (§4.4, Cai–Rafi) the contract did *not* address — the
  interface makes the **variance** faithful, not the **allocation** robust to its
  own small-pilot estimation error; **(4)** `cross` (§4.2/§7.C) is **broadened**
  from compositional-only coupling to also carry the **shared-hardware nuisance**
  covariance, with an empirical residual-cross-correlation gate. Where the review
  was less careful than this note (it presents the producer `σ₁=60` as the actual
  binding pair without flagging it is *stipulated*, not the seed delta-method
  propagation `σ₁≈25.2`; and it shorthands the kink as "nonconvex in n" when the
  precise loss is the SOC-**expressibility of the variance constraint**, the
  decision space staying convex in `n` on every region probed), this note states
  the correction with the executed number. The executed verifications backing
  this consolidation are in §8. **Re-execution pass (this revision):** the SOCP
  numbers were **re-run on the sign-safe `Q = diag(g)·R·diag(g)` formulation**
  (over the genuinely-positive per-component SE `w = √(A/n)`), which **corrected
  three figures** the prior passes relayed — (1) the diagonal formula's miss under
  a *negative* off-diagonal is an **over-spend** (`Var = 2.73 < V*`), not the
  wrong-signed under-statement `6.51 > V*` (the error's sign is `sign(g_i g_j R_ij)`,
  the magnitude instance-specific); (2) **SCS is not "materially inaccurate"** on
  the well-posed program (it agrees with CLARABEL across instances) — the "16×"
  claim was an artifact of a DCP-marginal `q = n^{-1/2}` form and is **withdrawn**;
  and (3) — the deepest fix — the intuitive `v = u/√n` form **silently
  misallocates on mixed-sign gradients** (it returns `status = optimal` with the
  cone reading `V*` while the true `Var = 5.59 ≠ 5.0`, because `cp.power(·,-2)`'s
  `>0` domain folds out the gradient sign); `model_capacity` **has** mixed-sign
  gradients (`d serve/dB_op > 0`, the rest `< 0`), so the general program needs the
  sign-safe `Q`-form and an ADR-0002 `gᵀΣ(n*)g ≈ V*` assertion on the returned
  allocation (§8 corrections 1–3). The same-sign worked numbers (the slope/intercept
  pair, the over-spend example) stand under either form. The convexity claim was
  also **re-anchored** on the Hessian min-eigenvalue (the `(σ₁,σ₂)`-alone
  nonconvexity is real but thin, in the small-`σ₁` corner — a naive whole-box chord
  search misses it; §8(c)). The Clark, correlation, smoothness, ρ-concavity,
  disjointness, and 2-point-pilot numbers all **reproduced** unchanged.
- **What it governs:** the **single contract** every measurable quantity (a
  *benchmark*) exposes so the Neyman driver
  (`tools/analysis/OpenTURNS/neyman_driver.py`) consumes them **uniformly**,
  whatever the quantity's estimator — a mean of timings, a regression
  slope/intercept, a config pin, a ratio, a quantile. It is the cure for the
  ad-hoc state the brief names: today each bench's `measure()` returns a
  bespoke dict, and a generic sampler fed a regression bench grabbed its
  row-count x-axis instead of the slope (`untrusted_drive._per_sample`, the
  longest-numeric-list heuristic at lines 80–98), and the bound cratered.

This note states **(1)** the contract, **(2)** the Neyman derivation under it
(closed-form on the diagonal, **SOCP** on the correlated/constrained general
case), **(3)** the case-handling table, **(4)** the subtleties most likely to
bite (the non-smooth `min()` — resolved to a closed form; the slope/intercept
correlation; the small-pilot robustness axis), **(5)** the schema delta,
**(6)** the migration path, **(8)** the executed verifications backing the
consolidation. It is honest per claims-measured-vs-interpreted: §7 separates
what is rigorous from what is a modelling choice. The `min()` kink — flagged in
the original draft as the one case that **cannot** be folded into the per-input
contract — is now resolved at the driver by the **Clark-1961** closed form
(§4.1), a deterministic per-step computation; the contract's role is unchanged
(carry each arm's full `cov` so the driver can evaluate Clark's min-moments).

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

### 2.3 The allocation, restated — closed form on the diagonal, an SOCP in general

The Neyman program is unchanged: minimize total cost `sum_i c_i · effort_i`
subject to `Var(E[f]) <= V_target = (h/mult)^2`. Its KKT stationarity condition
is the **general** Neyman optimum:

> at the optimum, the marginal variance-reduction **per unit cost** is equal
> across all funded inputs: `d Var(E[f])/d effort_i / c_i = lambda` (a common
> shadow price) for every funded *i*.

**What this program *is*, named (a refinement the external review supplied, and
the recent-literature home of the allocator).** Minimizing a budget subject to
`g^T Σ(n) g ≤ V*` with `c = ∇f` (the delta-method linearization) is precisely a
**cost-constrained c-optimal experimental design**: Neyman allocation is the
c-optimal design for a stratified mean (estimand `cᵀθ`, independent per-stratum
information), and the moment the inputs are **correlated** or are **fit
parameters rather than means**, the information matrix is non-diagonal and one
has left Neyman-proper for c-optimal design with a full cross-covariance. The
§2.3 stationarity above is exactly that design's optimality condition. The payoff
is **machinery**: Sagnol showed c-/A-/T-/D-optimal multiresponse design on a
finite design space — with several linear constraints (the cost budget is one) —
reduces to a **second-order cone program (SOCP)**. So the "small coupled convex
program" the correlated case needs is a named, solvable SOCP, **native to the
cross-covariance** the closed-form ratio drops. (The correlated-estimator
generalization — many models estimating one QoI — is the MLBLUE / approximate-
control-variate strand, where Croci–Willcox–Wright 2023 give an **SDP** yielding
both the sample counts and the model selection; a formulation to borrow, not a
drop-in, since this note's structure is "many inputs each estimated once.")

**Executed (ADR-0009, §8).** A `cvxpy` SOCP — the **sign-safe** form absorbs each
gradient's sign into the quadratic: `Q = diag(g)·R·diag(g)` (PSD by congruence of
the PSD correlation `R`), optimize over the genuinely-positive per-component SE
`w_i = √(A_i/n_i) > 0`, objective `min Σ_i c_i·A_i·w_i^{-2}` s.t. `‖L_Qᵀ w‖₂² ≤ V*`
where `Q = L_Q L_Qᵀ` — **(a)** reproduces the closed-form Neyman `n_i^* ∝ √(a_i/c_i)`
on the **diagonal** case to a max relative difference of **~1.9·10⁻⁵**, hitting
`Var = V*` exactly; **(b)** on a **non-diagonal** Σ (a slope/intercept-style `−0.81`
off-diagonal the closed form **cannot express**) hits `Var = V* = 5.0` exactly,
where the diagonal-optimal allocation evaluated under the true Σ misses `V*` (it
ignores the cross-term — for an instance with the negative off-diagonal and
**same-sign** `g`, it **over-spends**: realized `Var = 2.73 < V*`, cost `30.66` vs
the SOCP's `13.35`; §8 correction 1, magnitudes instance-specific, the **direction**
canonical). **Solver fact (operational):** **CLARABEL** is a clean, exact default;
on the well-posed program **SCS agrees with CLARABEL** (the earlier "SCS off ~16×"
was an artifact of a DCP-marginal `cp.power(n,-0.5)` form that itself errors on the
non-diagonal case — §8 correction 2, the warning withdrawn).

**A sign trap the naive `v`-form hides (an ADR-0002 fail-loud obligation, §8
correction 3).** The intuitive variable choice `v_i = u_i/√n_i` with `u_i = g_i√A_i`
and constraint `‖Lᵀv‖₂² ≤ V*` (`Σ = LLᵀ` the correlation Cholesky) is DCP-clean
**only for same-sign gradients**. Because `cp.power(v,-2)` has DCP domain `v > 0`,
the formulation silently forces `v > 0`, folding out the gradient signs; the cone
constraint `‖Lᵀv‖₂² = vᵀRv` then equals the true `gᵀΣg` **only if `v` carries the
sign of `g`**. On **mixed-sign** gradients the solver returns `status = optimal`
with the cone reading exactly `V*` while the **true** `Var(gᵀΣg)` at the returned
allocation is wrong (executed: `Var = 5.59 ≠ 5.0`, §8(b)). This is precisely the
fail-silently-on-an-`optimal`-status pattern ADR-0002 forbids, and `model_capacity`
**has** mixed-sign gradients — at the seed `d serve/dB_op = +3.08` while
`d serve/d{iota, slope, tau_io, LPD}` are all negative (§8). So **the general
program must use the sign-safe `Q`-form above** (which returns `Var = 5.0` exactly
for mixed signs), and the driver must **assert `gᵀΣ(n*)g ≈ V*` on the returned
`n*` before trusting it** — the solver's `optimal` status does not catch the sign
fold. The LIVE `iota`/`slope` pair is same-sign (`df/diota·df/dslope > 0`,
verified), so the slope/intercept worked numbers stand under either form; it is
the *general* program over all of a model's inputs that needs the `Q`-form.

So the SOCP is a **strict generalization**: the closed-form ratio is its
diagonal/independent special case, and it solves the correlated/constrained case
the 1934 formula cannot.

This holds for **any** differentiable, monotone-decreasing shrink law — it does
**not** require `Var = V/n`. The contract supplies exactly
`d Var(E[f])/d effort_i = g_i^2 · d(Σ_ii)/d effort_i` (plus the cross-term's
derivative) via `shrink.marginal`. So the optimum is computable for **every**
case, and three regimes fall out:

- **`Poolwise` (mean), diagonal Σ — the closed form is the SOCP's special
  case.** With `Σ_ii(n) = s_i^2/n_i`, `d Σ_ii/d n_i = -s_i^2/n_i^2`, and
  equalizing the per-cost marginal recovers `n_i^* ∝ sqrt(a_i/c_i)` — the exact
  formula at `neyman_driver.py:286–289`, **and** the SOCP solution on the
  diagonal (rel diff `~1.9·10⁻⁵`, §8). **This is the maintainer's sketch,
  earned**: for every independent mean input the generalized rule reproduces the
  current allocation line-for-line. *(KKT-point exact; the current greedy
  per-batch `topup` block is a **damped heuristic approach to the same SOCP
  optimum** — correct on the diagonal, but it is the SOCP, not the greedy
  damping, that is exact once Σ is non-diagonal.)*
- **Correlated / fit inputs (non-diagonal Σ) — the SOCP replaces the closed
  form.** A coefficient's SE has an x-leverage floor: `Var(slope) =
  resid_var/Sxx`, where more iters shrink `resid_var` but the `1/Sxx` leverage
  term is fixed unless the bench widens the x-design — so there is **no single
  `n`** for which `Var = V/n`, and the within-fit `−0.81` off-diagonal (§4.2)
  makes Σ non-diagonal. The driver does **not** invert a closed form; it solves
  the **SOCP** (§2.3, executed), which natively eats the cross-covariance and
  **saturates** when `resid_var` stops dominating — so the loop correctly **stops
  funding a fit whose SE is leverage-limited rather than residual-limited** (a fit
  the `V/n` assumption would fund forever). This is the brief's "states precisely
  where it breaks and what replaces it": the closed-form ratio breaks for
  non-`V/n` laws and for correlated inputs, the **SOCP** replaces it, and the two
  agree on the `Poolwise`-diagonal subset. *(The one place even the SOCP structure
  is lost is the `min()` kink, where the variance **constraint** stops being a
  second-order cone — §4.1; that is the principled boundary for the `kink_regime`
  branch.)*
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
| **NON-SMOOTH `f`** (the `min()` kink) | — | — | — | **Not an `Estimate` field** — it is a property of `f`, not of any input estimator. Resolved at the driver by the **Clark-1961 closed form** (§4.1): deterministic min-moments + `P(arg-min flips)=Φ(−t)`, no MC, no temperature. The contract's role is to carry each input's full `cov` so the driver can propagate each arm's capacity variance and feed Clark. |

---

## 4. The subtleties that bite

### 4.1 The non-smooth `min()` — resolved at the driver by the Clark-1961 closed form

`f = min(GENERATION, SERVE, TRANSPORT)`. This is a property of `f`, so it is
**not** a per-input contract field; it is the driver's and the model's concern.
The contract's contribution is that by carrying each input's full `cov` (not a
point `sigma`), the driver can compute the variance of **each arm's** capacity
and the binding margin honestly. The original draft of this note flagged the
`min()` as the one case that **cannot** be folded into the contract and proposed
a per-step Monte-Carlo (or a soft-min with an ad-hoc temperature) at the driver;
it correctly named the object as "a Clark-1961 mixture moment" but did **not**
push to the closed form. **The closed form exists, is deterministic and O(1), and
closes that open question** (§7); the MC-vs-soft-min fork was a false choice. The
full picture and the driver-level response follow.

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
single-arm variance is a lie** — and the honest object is a **closed form**, not
a simulation.

**The operating point, in the real (propagated) numbers — anchor here, not on the
dramatic worked figures below.** The producer spread is **not** a literal `60`: it
is the delta-method propagation through `producer = N_gen·R_gen`, which at the seed
is **`σ₁ ≈ 25.2`** (`√((R_gen·σ_{N_gen})² + (N_gen·σ_{R_gen})²)`; production rule:
*source each arm's σ from its `Estimate` `cov`*, §8 — never a literal). There Clark
gives **`E[min] = 426.5`** (Jensen bias **+1.7 dps**), **`sd = 6.2`**, and
**`P(producer binds) = Φ(−t) = 0.136`** — modest, but enough that the binding margin
is statistically live and the convergence guard funds both arms. **The worked example
below uses `σ₁ = 60`**, a **stipulated stress case** (2.4× wider) chosen to make the
kink's structure legible; read every `σ₁=60`-tagged figure below
(`E[min]=415.68 / sd=25.58 / P=0.322 / 12.79×`) as the **stress reading**, not the
operating point (full propagation in the provenance caveat below).

The two contending arms are **input-disjoint**: producer reads
`{N_gen, R_gen}`, serve reads `{T_disp, tau_io, wakeup, B, t_row, L}`, and the
intersection is **empty** (verified against `model_zmq_baseline.INPUT_NAMES`;
`L` is shared only with the non-binding transport arm). So `ρ_arms = 0` and the
two delta-method-Gaussian arms — producer `Normal(456, 60)`, serve
`Normal(428.28, 2)` — are exactly Clark's **exact-independent** case. The
**Clark-1961** moments of `min(X₁,X₂)` (via `min(x,y) = −max(−x,−y)`), with
`a = √(σ₁²+σ₂²−2ρσ₁σ₂) = SD(g₁−g₂)` and standardized margin `t = (μ₁−μ₂)/a`:

```
a               = 60.033          t = (μ₁−μ₂)/a = 0.4617
E[min]          = μ₁Φ(−t) + μ₂Φ(t) − a·φ(t)                          = 415.68
Var[min]        = (μ₁²+σ₁²)Φ(−t) + (μ₂²+σ₂²)Φ(t) − (μ₁+μ₂)a·φ(t) − E[min]²
                                                          ⇒ sd[min]  = 25.58
P(producer is the min) = Φ(−t)                                       = 0.3221
```

Read against the naive single-arm delta-method:

- `E[min] = 415.68` vs the delta-method `E[f] = min(456, 428.28) = 428.28` — a
  **Jensen bias of +12.60 dps, optimistic** (`min` is concave, so `min(E[·])`
  over-states `E[min(·)]`). This bias **is the `−a·φ(t)` term**, and since
  `a = O(σ)`, it is **first-order** O(σ) at a kink — not the O(σ²) a smooth `f`
  gives. The same closed form that fixes the variance de-biases the mean.
- `sd[min] = 25.58` vs the single-binding-arm delta-method `sd = 2.0` — **12.79×**
  larger, because the realized binding arm switches stochastically.
- `P(producer is the realized min) = Φ(−t) = 0.3221` — a full third of the
  probability mass lives on an arm whose ±(3·60) dps the single-arm CI ignores.

These are **deterministic, no draws**. A 4·10⁶-draw Monte-Carlo cross-check
(reported here only to *validate* the closed form, not as the per-step method)
agrees to ~4 decimals: `E[min]=415.70`, `sd=25.55`, `P=0.3220` (§8). So
`g^T Σ g` cannot represent `Var(min(·))` — it is a **Clark-1961 mixture moment**,
a functional of `f`, not a per-input estimate — but **Clark gives it in closed
form**. The danger the closed form forecloses is the over-permissive **false-SAT**
the project's faithful-model discipline warns against: `converged := var_est <=
V_target` could fire at `E[f]=428.28` with CI `±3.9` while 32% of the mass is
producer-bound.

> **Provenance caveat on `σ₁=60` (a correction the external review did not flag).**
> The worked pair uses producer `σ₁=60`, a **stipulated** wide value (the review
> presents `456/60/428.28/2` as the actual binding pair). The **seed**
> delta-method propagation of `σ₁` from `producer = N_gen·R_gen` is
> `√((R_gen·σ_{N_gen})² + (N_gen·σ_{R_gen})²) = √((152·0.05)² + (3·8)²) = 25.17`
> — a **2.4× difference**. Clark reproduces 415.68/25.58/0.32 for the *given*
> `(456, 60, 428.28, 2)` regardless, but `P(producer binds)` is **σ₁-sensitive**:
> `Φ(−t) = 0.136` at `σ₁=25.17` versus `0.322` at `σ₁=60` (§8). Because the
> `kink_regime` trigger and the funded-both-arms decision turn on this
> probability, **production must source each arm's σ from its `Estimate` `cov`
> (the delta-method propagation through `N_gen·R_gen`), not a literal `60`.** The
> worked numbers below are annotated as `σ₁=60`-stipulated.

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

**The driver-level response (three honest mechanisms, all Clark arithmetic — no
stochastic element added to `step()`):**

1. **Binding-margin diagnostic + a `kink_regime` flag.** The driver computes,
   per arm, `margin = (this_arm_capacity − binding_capacity)/binding_capacity`,
   and the *variance* of each arm's capacity from the carried `cov`s
   (propagated through each arm's `f` by the delta method). It raises
   `kink_regime` when any non-binding arm is within a threshold (a
   statistically-plausible tie, e.g. `margin < k · arm-capacity-CI`). The 6.5%
   seed margin **fires** this flag. *(With `σ₁` sourced from the `Estimate` `cov`
   rather than the stipulated `60`, the trigger sensitivity is `Φ(−t)`; see the
   provenance caveat above.)*

2. **In the kink regime, replace `g^T Σ g` with the Clark-1961 closed-form
   `E[f]` and `Var(f)`.** When `kink_regime` is set, the driver does **not**
   report the single-arm `g^T Σ g`; it linearizes each arm at the operating
   point — `arm_k ≈ Normal(μ_k, σ_k²)` with `μ_k = g_k(θ̂)`, `σ_k² = ∇g_kᵀΣ∇g_k`,
   cross-covariance `∇g₁ᵀΣ∇g₂` (zero here, by input-disjointness) — and evaluates
   **Clark's exact min-moments** for `E[min]` and `Var[min]` (the formulae above).
   This is **deterministic, O(1), parameter-free**: the input noise itself
   supplies the smoothing scale `t = Δ/a`, so there is **no Monte-Carlo
   (reproducibility preserved) and no soft-min temperature to characterize**.
   Both the **de-biased `E[f]`** (the `−a·φ(t)` Jensen correction, mechanism 2's
   point estimate) and the **honest `Var(f)`** are arithmetic on `Φ, φ`, and the
   arms' covs, evaluated each step. This is exactly what block-based statistical
   static timing analysis (SSTA) does: it **never Monte-Carlos a max/min**; it
   propagates Clark moments. *(For ≥3 contending arms — not the case here, where
   transport is non-binding — Clark's recursive pairwise form with its
   correlation-propagation step applies; with 2 arms it is the single closed form
   above.)*

3. **A convergence guard.** Convergence is **refused** while the **arg-min-flip
   probability `P(producer is the min) = Φ(−t) > α`** (a small probability,
   read directly off the same closed form — mechanism 3 is `Φ(−t)`, not a
   separate computation) or while the probability mass on a non-binding arm
   exceeds ε. This makes the over-permissive false-SAT impossible (ADR-0002 loud
   at the strongest surface).

   And the allocation in the kink regime funds **both** contending arms'
   inputs. The principled weight is the **probability-of-binding-weighted
   gradient** `∇E[min] ≈ Φ(−t)∇g₁ + Φ(t)∇g₂` — the weights are
   `P(each arm is the min)`, what SSTA calls the **tightness / criticality
   probability**, and they **sum to 1** (verified: `dE/dμ₁ = Φ(−t) = 0.3221`,
   `dE/dμ₂ = Φ(t) = 0.6779`, §8). This is the rigorous form of the "soft-min"
   the contract-only designs hand-waved: near a tie both arms get nonzero weight,
   curing the zero-gradient-on-the-inactive-arm pathology; away from a tie
   (`|t|→∞`) it collapses to the hard `min`. The necessary-but-insufficient
   "fund both arms" piece the contract-only designs got right is now the exact
   `Φ(±t)` weighting, paired with the CI/convergence fix that is the actual cure.

It is **not** that the delta-method is "suspended" at the tie and falls back to
something worse: the **faithful objective is smooth in closed form**. The kink in
`min` does **not survive the expectation** — the input noise convolves it away,
so `Var[min](n)` is `C^∞` on the positive orthant (verified smooth even at the
exact tie `Δ=0`: bounded `d²Var/dσ₁²`, no spikes, §8). What is lost at the tie is
not smoothness but **convexity**: `Var[min]` is generally **nonconvex** in
`(σ₁, σ₂, ρ)` (verified — a stable negative Hessian eigenvalue `≈ −0.27` in the
small-`σ₁` corner of `(σ₁,σ₂)`, and `Var[min]` concave in `ρ`, §8). So the
cone-program structure of the single-branch SOCP
(§2) is lost exactly at the tie — the **principled boundary for the
`kink_regime` branch** — and the regime becomes smooth **nonconvex** optimal
design, still differentiable (analytic `dVar/dn` from Clark × `dΣ/dn`), so
gradient-based design works there. *(Precision, departing from the review's
shorthand "nonconvex in n": the loss is the **SOC-expressibility of the variance
constraint** — `Var[min](n) ≤ V*` cannot be written as a second-order cone
because `Var[min]` is nonconvex in the `(σ,ρ)` it is built from. The decision
variable `n` itself stayed convex on every region probed, because
`σ(n)=√(A/n)` tends to **restore** convexity in `n`: 0/28 probed
(gap × n-range) regimes showed a nonconvexity witness in `n`-space, §8. The
honest claim is "lose the cone constraint," not "the program in `n` is
nonconvex.")*

Away from a tie (`margin` comfortably large) `t` is large, `Φ(−t)→0`, the
non-binding arm drops out, and Clark collapses to the single binding arm: the
analytic gradient is honest, the non-binding arms' `df/dx = 0` is **correct**,
and behavior is exactly today's. The exact tie `Δ=0` (measure zero) is the one
permanently pathological case — `t≡0` for all `n`, the blend never resolves, and
Clark's normality assumption degrades precisely when the means are equal with
dissimilar variances; there the kink is permanently load-bearing.

**One reconciliation, so a reader is not whipsawed (raised by the review,
resolved here).** It can look contradictory that `min` of two i.i.d. Gaussians
*clips* variance (`Var[min] ≈ 0.68σ²`) while this case shows `Var[min]` **12.79×
larger** than the binding arm. Both are Clark; the sign depends on the
**asymmetry**. The clip shows up against the **wide** arm (here producer,
`σ=60`): `min` truncates its upper tail. The **inflation** shows up against the
**tight** arm (here serve, `σ=2`): the wide producer dips under serve 32% of the
time and **injects its spread** into the realized `min`. The dangerous direction
for the convergence test is the one this case sits in — measured against the
tight binding arm, the honest variance is **larger**, so the naive single-arm CI
is **overconfident**: precisely the false-SAT the §4.1 guard exists to forbid.

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
hardware and are physically non-independent. This is named in §7.C as a known
approximation. The `cov_group` / shared-`Estimate` mechanism pairs **only**
co-fit components: `(t_row, iota)` from the staged fit, `(t_row_bare, T_disp)`
from the cpp fit — and must **not** pair `t_row` with `T_disp` (different
variants). A migration that grouped them would fabricate a covariance that does
not exist.

**`cross` carries TWO kinds of coupling — broadened from the original draft (a
distinction the external review supplied).** As specified by **D5**, `cross` is
scoped to **compositional** coupling: one input a composite of another (the
ratio / Jacobian-of-a-composite case, §3). But the `T_disp` ⊥ `t_row`
independence-by-different-fits is a different animal — it is **nuisance**
coupling: two **distinct** fits sharing a clock, a thermal envelope, and the same
core. A shared-clock/thermal covariance between two distinct fits is **not** a
Jacobian-of-a-composite, so it has **no home in `cross` as D5 specifies it**.
The slot is therefore broadened: `cross` carries **both** the compositional block
(D5) **and** an optional **nuisance / shared-hardware block** keyed by the other
bench's quantity name, defaulting to `{}` (block-diagonal — the honest "no shared
nuisance" assumption). The two are distinguished by a `kind` tag on the cross
entry (`composite` vs `nuisance`) so the store and the report do not conflate a
rigorous Jacobian coupling with a measured nuisance correlation.

**The nuisance block is empirically checkable before it is assumed negligible
(the gate that earns the block-diagonal default).** Whether `T_disp ⊥ t_row` is
benign is not a matter of assertion: **interleave** the two fit benches (alternate
their measurements on the same core, same session) versus running each
**isolated**, and measure the **cross-correlation of their fit residuals**. A
residual cross-correlation `≈ 0` **earns** the block-diagonal default (the
`cross == {}` assumption is then a measured fact, not a hope); a nonzero value is
a **number that needs a slot** — populated into the nuisance block. This makes
the §7.C "a future cross-fit covariance would tighten it" caveat **actionable**:
it names the experiment (interleaved vs isolated), the statistic (residual
cross-correlation), and the threshold (≈0 ⇒ block-diagonal), rather than leaving
the independence an unfalsified convenience.

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
  reach. *(The within-fit x-design is itself a c-optimal design: lowering
  `1/Sxx` by the choice of batch widths is the classical c-optimal design for a
  slope, whose solution puts mass at the x-**extremes**. The even-ish 7-point
  `[32…512]` is deliberately **not** slope-optimal — it trades endpoint
  efficiency for the interior points the lack-of-fit / R² gate needs. State the
  tension as the **reason** the design is what it is, so a future "optimize the
  x-design" impulse does not collapse it to two endpoints and go blind to
  curvature.)*

### 4.4 Robustness of the allocation to its own estimation error — an orthogonal axis (Cai–Rafi)

This sub-section records a refinement the external review supplied and the
original draft did **not** address — and it is deliberately a **separate axis**
from everything above. The contract (`theta_hat`, `cov`, `family`, `support`)
makes the **reported variance faithful**: it carries the right `Var(theta_hat)`,
the right correlation, the honest CI multiplier. It does **not** make the
**allocation robust to the error in its own inputs.** Those are different
properties, and conflating them would be its own dishonesty.

**The mechanism.** The allocation is **fully plug-in**: `a_i = g_i² · V̂ar` is
computed from a small pilot's `σ̂` and `ĝ` (the suite's `UD_PILOT = 32`). Cai &
Rafi's small-pilot result is directly on point: plug-in Neyman allocation can do
**worse than uniform** when the outcome is near-homoskedastic across the split or
**heavy-tailed** — and timing data is precisely right-skewed/heavy-tailed (GIL
handoffs, tail latencies; the benches compute `_median_iqr_us` for exactly that
reason). So a 32-sample pilot can **confidently misallocate**. Note what does and
does **not** help here: `family + STUDENT_T` (§4.3) makes the **CI honest**, and
the median estimators (§7.A) tame the tails at the **estimate** level — but
**neither touches the allocation**, which still chases a noisy `V̂ar` with no
shrinkage or floor.

**Mitigations (cheap → principled), reusing hooks the suite already has:**

- **Robust scale.** Use a robust spread (the **IQR the benches already compute**,
  `_median_iqr_us`) in place of `std(ddof=1)` when forming `a_i`, so a single
  heavy-tail draw does not dominate the allocation.
- **Shrink `a_i` toward uniform.** Convex-combine the plug-in `a_i` with a flat
  allocation; on a noisy small pilot this strictly dominates the raw plug-in in
  the regime Cai–Rafi identify.
- **Floor each allocation.** Keep any input's share from collapsing to zero on a
  noisy early estimate — the same instinct as the driver's existing
  `growth_cap` (which damps *growth*; a floor damps *collapse*).
- **The principled move: Clip-OGD.** Drop fixed-point chasing of a moving `a_i`
  for the **online-convex** view — regret minimization with **clipping** that
  keeps any input's share from collapsing on a noisy early estimate. This is the
  `growth_cap` instinct **with a regret guarantee**, and it is the named
  adaptive-allocation result that *does* transplant to a budget-split problem
  (the adaptive-sequential-Neyman branch — TS-Neyman — does **not** transplant:
  it advances a different reading where the decision is which units go to which
  arm/stratum, not how to split a measurement budget across estimators of several
  inputs).

**Why this is orthogonal, stated plainly (the honest framing).** The interface
makes the **variance** faithful; it does **not** make the **allocation** robust
to its own estimation error. The two literatures the §2.3 SOCP rests on (c-optimal
design, MLBLUE/ACV) **assume the variances and covariances are known or local** —
the small-pilot, heavy-tailed fragility is a **caveat on the input** to those
methods, not something they fix. So this axis is a hook the contract **enables**
(the `cov`, the IQR, the `family` are all present) but does **not** itself
discharge; it is named here so a reader does not mistake "faithful variance" for
"robust allocation."

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
  g` (equal to today's sum for diagonal Σ → no regression on existing models),
  and the closed-form `sqrt(a_i/c_i)` allocation with the **SOCP** (§2.3) on the
  correlated/constrained case (CLARABEL; it reduces to the closed form on the
  diagonal, so no regression on all-mean models). Land the binding-margin /
  `kink_regime` **Clark closed-form** path (§4.1 — deterministic, no MC), the
  per-`family` multiplier, and the convergence guard. `run()`'s `samplers[i](k)`
  becomes `measurers[i](budget) -> Estimate`. Keep `add_samples` as a thin
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

**Rigorous (proved against the code / executed — see §8 for the numbers):**

- The delta-method bound `Var(f(θ̂)) ≈ g^T Σ g` for any consistent estimator
  with sampling covariance Σ — the multivariate delta method; the mean is one
  consistent estimator and no step required it. Reduces to today's diagonal sum
  bit-for-bit on the all-means case.
- The allocation is a **cost-constrained c-optimal design solved as an SOCP** (the
  **sign-safe `Q = diag(g)·R·diag(g)`** form, native to mixed-sign gradients);
  the Neyman closed form `n_i^* ∝ sqrt(a_i/c_i)` is its **diagonal/independent
  special case** (SOCP vs closed form: rel diff `~1.9·10⁻⁵`, §8), and the SOCP
  hits `V*` exactly on the correlated case the closed form cannot express
  (CLARABEL exact default; SCS agrees on the well-posed program — the earlier
  "SCS inaccurate" was a DCP-form artifact, §8 correction 2). The diagonal
  formula's error under a dropped cross-term has the sign of `g_i g_j R_ij` (it
  can over- **or** under-spend; §8 correction 1). The naive `v = u/√n` form
  **silently misallocates on mixed-sign gradients** (which `model_capacity` has),
  so the general program needs the `Q`-form and an ADR-0002 `gᵀΣ(n*)g ≈ V*`
  assertion (§8 correction 3).
- The slope/intercept covariance: `Cov = −x̄·resid_var/Sxx`, correlation −0.8114
  on the real design (and the three `(AᵀA)⁻¹` entries match the closed forms
  exactly, §8) — carried in `cov`, not dropped.
- The `min()`-kink moments are the **Clark-1961 closed form**, deterministic and
  parameter-free: `E[min]=415.68`, `sd[min]=25.58`, `P(producer binds)=Φ(−t)=
  0.322` for the stipulated pair, MC-confirmed to ~4 decimals (§8); the
  arg-min-flip gradient weights are `Φ(±t)` and sum to 1. The arms are
  input-disjoint (`ρ_arms=0`, verified vs `INPUT_NAMES`).
- The fabricated 2-point pilot has **no** `/2` bug (its std is `sqrt(2)·σ`,
  verified — `a_i/n_i = grad²·σ²` exactly, §8).

**Modelling choices / least-bad options (named, not papered over):**

- **`A.` The latency benches report a MEDIAN, so the "MEAN" mapping is a
  modelling choice that must be corrected to QuantileLaw(p=0.5).** The benches
  return `np.median`; `Var(median) ≈ 1/(4 n f(median)²)`, the order-statistic
  law, **not** `s²/n`. The contract handles this cleanly (the MEDIAN/QUANTILE
  row), and the `(θ̂, V/n, n)` sketch is **not** earned for latencies — only for
  a true arithmetic mean, which no timing bench currently produces. The
  least-bad alternative if a bench prefers `Poolwise` is to switch its headline
  to a trimmed mean; that is the bench's choice, declared in `kind`. *(Review
  refinement: `f̂(median)` — the density-at-quantile in the order-statistic law —
  is itself small-sample-fragile, so the asymptotic `p(1−p)/(n f̂²)` is unreliable
  at the bench `n`. Prefer a **bootstrap median SE**, which is steadier at small
  `n`; the bench owns this, declaring `family = EMPIRICAL` with its bootstrap
  interval rather than a fabricated asymptotic one.)*
- **`B.` The `min()` kink is resolved at the driver by the Clark-1961 closed
  form** (this supersedes the original draft's "cannot be folded in / suspend the
  delta-method and Monte-Carlo it" framing — see §4.1). `Var(min(·))` is a
  functional of `f`, not a per-input estimate, so it is **not** a contract field;
  but it is **not** computed by simulation either. The faithful objective
  (delta-method per arm, **Clark to combine**) is **smooth in closed form** — the
  kink is convolved away by the input noise and never reaches the variance
  functional. `f(μ̂)` is a biased (Jensen, **+12.60** dps at the seed, the `−a·φ(t)`
  term), variance-understated (**12.79×**) estimate of `E[f]` in the tie regime,
  and the **de-biased `E[f]` and honest `Var(f)` are Clark arithmetic each step**
  (deterministic, O(1), no temperature, no draws). The remaining modelling content
  is small: Clark's normality of the arms is exact only under the delta-method
  Gaussian linearization, and it **degrades at the exact tie `Δ=0`** (means equal,
  variances dissimilar) — the one permanently load-bearing case (§4.1). **And —
  bridging `B` to `F`/§4.4, the connection the two sections otherwise leave
  implicit:** Clark's arm-normality is also only **leading-order on heavy-tailed
  inputs**. The producer arm `N_gen·R_gen` inherits `R_gen`'s right skew, and
  serve's delta-method-Gaussianity is only as good as its inputs' CoV — so the
  **same heavy-tail fact that makes the allocation fragile (`F`) also makes the
  kink moments approximate**: the `P(flip)` and `Var[min]` the convergence guard
  reads are Gaussian-arm estimates of **skewed** quantities. This is the standard
  leading-order choice and almost certainly fine to start; if the Gaussian-arm
  error ever proves material, the **SSTA** literature §4.1 already borrows carries
  **skew-aware** max/min moment variants (skew-normal arms / higher-moment
  matching) as the escalation path — a documented hand-off, not a blocker.
- **`C.` `T_disp` ⊥ `t_row` (and `T_disp` ⊥ `iota`) is block-diagonal by the
  different-fits rule, but they are physically non-independent** (different fits,
  same hardware). Treating distinct fits as independent is defensible and is the
  `cross=={}` default, but it is a modelling choice. This is a **nuisance**
  coupling (shared clock/thermal/core), **not** the compositional coupling D5
  scopes `cross` to — so the slot is **broadened** to carry a nuisance block
  (§4.2), and the independence is made **falsifiable**: interleave the two fit
  benches vs run them isolated and measure the **residual cross-correlation** —
  `≈0` earns the block-diagonal default, nonzero is a number for the nuisance
  block. The "future cross-fit covariance would tighten it" is now an actionable
  experiment, not an open hand-wave.
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
- **`F.` The interface makes the variance faithful, not the allocation robust to
  its own estimation error** (§4.4, Cai–Rafi). `a_i` is plug-in from a 32-sample
  pilot; on heavy-tailed timing data a small pilot can misallocate worse than
  uniform. This is **orthogonal** to everything above: `family + STUDENT_T` makes
  the CI honest and the medians tame the tails at the *estimate* level, but
  neither makes the *allocation* robust. The mitigations (robust IQR scale, shrink
  `a_i` toward uniform, floor each share; principled = Clip-OGD) are hooks the
  contract enables but does not itself discharge.

**The highest-risk open question is now RESOLVED (was carried forward; closed by
this consolidation).** The original draft ended on a dilemma: the `min()`-kink
regime needs a *cheap, faithful* `Var(min)` estimator the driver runs every step
— a Monte-Carlo (stochastic, costs reproducibility) **or** a soft-min with a
principled temperature (a modelling choice whose bias must be characterized.) The
external review supplied, and §8 executes, the resolution: **neither — the
Clark-1961 closed form.** Because the two contending arms are input-disjoint
(`ρ_arms=0`) delta-method Gaussians, Clark's exact min-moments give `E[min]`,
`Var[min]`, and `P(arg-min flips)=Φ(−t)` **deterministically, O(1),
parameter-free**. The Monte-Carlo was the right way to *validate* (it agrees to
~4 decimals); it is **unnecessary per step**, so `step()` stays deterministic
(the reproducibility the dilemma worried about is preserved) and there is **no
temperature to characterize** (the input noise supplies the smoothing scale
`t=Δ/a`; Clark is its exact form). This is the SSTA discipline: propagate Clark
moments, never simulate a max/min. The one residual caveat is the exact tie
`Δ=0` (§4.1, §7.B), where Clark's normality degrades — a measure-zero case the
guard refuses to call converged, not a method gap. The load-bearing decision the
contract **enables** (by carrying each arm's full `cov`) is now also **settled**.

---

## 8. Executed verifications backing the consolidation (ADR-0009)

Every integrated formula in this consolidation is reported here with the number
it reproduces, executed against the live code and the stated libraries
(`numpy 2.4.6`, `scipy 1.17.1`, `cvxpy 1.9.1` with CLARABEL + SCS, at
`/home/bork/w/vdc/venvs/generic/bin/python`). A formula is integrated **only**
with its own reproduced number — never relayed on the review's authority. (The
verification scripts are throwaway drivers; the numbers, not the scripts, are the
deliverable. ADR-0009: the claim carries its substantiation.)

**(a) The Clark-1961 closed form reproduces the note's `min()` numbers
deterministically.** For `min(Normal(456, 60), Normal(428.28, 2))`, `ρ=0`:

| quantity | Clark closed form (no draws) | MC, 4·10⁶ draws | note's figure |
| — | — | — | — |
| `a = SD(g₁−g₂)` | 60.033 | — | 60.03 |
| `t = (μ₁−μ₂)/a` | 0.4617 | — | 0.462 |
| `E[min]` | **415.681** | 415.697 | 415.7 |
| `sd[min]` | **25.582** | 25.555 | 25.6 |
| `P(producer is min) = Φ(−t)` | **0.3221** | 0.3220 | 0.32 |
| Jensen gap `min(μ)−E[min]` | **+12.599** | — | +12.6 |
| `sd[min] / σ_bind` (=2) | **12.79×** | — | 12.8× |

The smoothed gradient (FD on Clark's `E[min]`) is `dE/dμ₁ = 0.32213 = Φ(−t)` and
`dE/dμ₂ = 0.67787 = Φ(+t)`, **summing to 1.0** (the SSTA criticality property).
**Conclusion:** the closed form reproduces all of 415.7 / 25.6 / 0.32 / +12.6 /
12.8× with no draws; the MC-vs-soft-min fork (§7 open question) is a **false
choice**. *(Provenance caveat: `σ₁=60` is **stipulated**, not the seed
delta-method propagation `√((152·0.05)²+(3·8)²)=25.17`; at `σ₁=25.17`,
`Φ(−t)=0.136` vs `0.322` at `σ₁=60` — production must source `σ₁` from the
`Estimate` `cov`, §4.1.)*

**(b) The cvxpy SOCP expresses the cost-constrained c-optimal allocation and
agrees with the closed form on the diagonal.** The **general, sign-safe**
formulation absorbs the gradient sign into the quadratic so it survives mixed-sign
gradients: `Q = diag(g)·R·diag(g)` (PSD by congruence of the PSD correlation `R`,
`Q = L_Q L_Qᵀ`), optimize over the genuinely-positive per-component SE
`w_i = √(A_i/n_i) > 0`, `min Σ_i c_i·A_i·w_i^{-2}` (`cp.power(w,-2)`, convex) s.t.
`‖L_Qᵀ w‖₂² = wᵀQw ≤ V*` (convex). *(Provenance: an intuitive `v_i = u_i/√n_i`
form with `u_i = g_i√A_i` and constraint `‖Lᵀv‖₂² ≤ V*`, `Σ = LLᵀ`, is DCP-clean
**only for same-sign gradients** — `cp.power(·,-2)`'s `>0` domain forces `v>0` and
folds out the sign, so it silently misallocates on mixed signs, see the trap row
below. A further variant `q = cp.power(n,-0.5)` is **DCP-marginal** and raises a
`DCPError` on the non-diagonal case. The earlier consolidation reported numbers
from a `q`-form variant; this revision re-executes on the sign-safe `Q`-form and
**corrects three figures** — flagged below.)*

| case | solver | status | `Var` achieved (target 5.0) | vs closed form |
| — | — | — | — | — |
| diagonal `R=I` (sign-safe `Q`) | **CLARABEL** | optimal | **5.000000** | max rel diff `n` **1.9·10⁻⁵** |
| diagonal `R=I` (sign-safe `Q`) | SCS | optimal | **5.000000** | agrees (correction 2) |
| non-diagonal `R`(−0.81), same-sign `g` (sign-safe `Q`) | **CLARABEL** | optimal | **5.000000** | (closed form cannot express) |
| non-diagonal `R`(−0.81), same-sign `g` (sign-safe `Q`) | SCS | optimal | **5.000000** | agrees |
| non-diagonal `R`(−0.81), **mixed-sign** `g`, naive `v`-form | CLARABEL | **optimal (a lie)** | **5.585** ≠ 5.0 | the silent sign fold — correction 3 |
| non-diagonal `R`(−0.81), **mixed-sign** `g`, sign-safe `Q`-form | CLARABEL | optimal | **5.000000** | the fix |

At the **diagonal-optimal** allocation evaluated under the true non-diagonal Σ
(`R₁₂ = −0.81`, same-sign `g`), the realized `Var = 2.73 < 5.0` — the closed form
**misses `V*`** because it drops the cross-term, here **over-spending** (cost
`30.66` vs the SOCP's `13.35`). *(The magnitudes `2.73 / 30.66 / 13.35` are
specific to the undisclosed `(g, A, c)` instance; the **load-bearing** fact is the
**direction** — same-sign `g` + negative `R` ⇒ the diagonal formula over-states
variance ⇒ over-spend, `Var < V*` — which is robust across instances, §8 correction
1. A different instance gives different magnitudes preserving the direction.)*
**Conclusion:** the SOCP is a strict generalization (reduces to Neyman on the
diagonal, solves the correlated case the closed form cannot), provided the
sign-safe `Q`-form is used so mixed-sign gradients do not silently misallocate.

> **Correction 1 (sign — supersedes the earlier consolidation's `Var = 6.51`).**
> The effect of dropping the cross-term has the sign of `g_i g_j R_ij`. For the
> stated **negative** correlation `R₁₂ = −0.81` with same-sign gradients, the
> dropped term is negative, so the diagonal formula **over-states** the variance
> (true `Var = 2.73 < V*`), i.e. the closed form is **conservative and
> over-spends** — the executed cost is `30.66` vs the SOCP's `13.35`. The earlier
> `Var = 6.51 > V*` (an *under-statement*) is the **wrong sign** for a negative
> off-diagonal: re-running the same `R = −0.81` instance on the well-posed `v`-form
> gives `2.73`. (`6.51 > V*` would require `g_i g_j R_ij > 0`; verified by
> sign-flipping one gradient — opposite-sign `g`, `R₁₂ = −0.81` gives the
> under-statement `Var = 7.27`.) **The load-bearing claim is unchanged** — the SOCP
> hits `V*` exactly under non-diagonal Σ while the diagonal formula does not — but
> the *direction* of the diagonal formula's error is `sign(g_i g_j R_ij)`, and for
> the real slope/intercept pair it is set by the model's `df/dslope`, `df/dintercept`
> signs (so it can over- **or** under-spend; it is not generically anti-conservative).
>
> **Correction 2 (solvers — supersedes "SCS is materially inaccurate").** On the
> well-posed program (the sign-safe `Q`-form, and the `v`-form on same-sign `g`),
> **SCS and CLARABEL agree** to ~1e-6 (`Var = 5.000000` vs `4.99998`) — across
> same-sign non-diagonal instances, five random diagonal instances, and a
> badly-scaled instance (`n` up to ≈2825, `A` spanning 100–4000). I **could not
> reproduce** the earlier "`SCS` returns `optimal_inaccurate`, off ≈16×, `Var`
> 7.8–9.6"; that was an **artifact of the DCP-marginal `q`-form** (which itself
> errors on the non-diagonal case), not a property of SCS on this allocation. The
> honest operational claim: **CLARABEL is a clean default and exact; SCS is also
> accurate on the well-posed form** (tighten `eps` only if an ill-conditioned
> instance reports `optimal_inaccurate`; a few harsh `(g,A)` instances do make
> CLARABEL itself `SolverError` — verify `Var` and retry on the other solver). The
> "16×" warning is withdrawn.
>
> **Correction 3 (sign — the naive `v`-form silently misallocates on mixed-sign
> gradients; an ADR-0002 fail-loud obligation).** The `v_i = u_i/√n_i` form with
> `u_i = g_i√A_i` and constraint `‖Lᵀv‖₂² ≤ V*` is correct **only for same-sign
> gradients**: `cp.power(v,-2)`'s `v>0` domain folds out the sign, and the cone
> `‖Lᵀv‖₂² = vᵀRv` equals the true `gᵀΣg` only if `v` carries `sign(g)`. On
> mixed-sign `g` the solver returns `status = optimal` with the cone reading
> exactly `V*` while the **true** `Var = 5.59 ≠ 5.0` (executed). The sign-safe fix
> is the `Q = diag(g)·R·diag(g)` form (PSD by congruence) over the genuinely-
> positive `w = √(A/n)`, which returns `Var = 5.0000` for mixed signs (executed).
> This is the **same** sign-indefiniteness of `Lᵀ` the rejected `q`-form already
> flags ("mixes a convex `q` through the sign-indefinite `Lᵀ`") — it **resurfaces
> in the `v`-form** the instant gradients are mixed-sign; canonicalizing on `v>0`
> is exactly what **hides** the fold, so the `v`-form is not automatically safe
> just because it is DCP-clean. `model_capacity`'s gradient **is** mixed-sign — at
> the seed `d serve/dB_op = +3.08` while `d serve/d{iota, slope, tau_io, LPD}` are
> negative (sign set `{+, −}`, §8 corroborating checks) — so the *general* program
> over all its inputs needs the `Q`-form. Because the solver's `optimal` status
> does **not** catch the fold, the driver must **assert `gᵀΣ(n*)g ≈ V*` on the
> returned `n*`** (ADR-0002 loud) before trusting it. *(This does not invalidate any
> reported figure: the worked correlation case and the over-spend example use
> same-sign gradients, and the LIVE `iota`/`slope` pair is same-sign, so their
> numbers stand under either form.)*

**(c) `Var[min](n)` is smooth everywhere but loses convexity (and hence SOC
structure) — in `(σ,ρ)`, not in `n`.**

- **Smooth:** `max|d²Var/dσ₁²|` stays bounded (~0.68–0.73) with no spikes across
  gaps `Δ = 27.72 → 10 → 2 → 0` **including the exact tie** — the kink is
  convolved away (`C^∞` on the positive orthant), confirming the review.
- **Nonconvex in `(σ₁,σ₂)`, located by the Hessian (the robust witness).** The
  nonconvexity is **real** but **thin**, living in the small-`σ₁` corner: the
  Hessian of `Var[min]` w.r.t. `(σ₁,σ₂)` has a **stable negative eigenvalue
  ≈ −0.266 at an interior point `σ₁≈1.0, σ₂≈20.5`** (seed gap `Δ=27.72`, `ρ=0`),
  reproducible across FD steps `h ∈ [10⁻², 10⁻⁴]` (`−0.266 / −0.266 / −0.268`).
  *(Methodological caveat so a reader does not "refute" the claim with a generic
  search: a **naive whole-box random chord search returns 0 witnesses** — the
  positive chord witness `f(mid) > ½(f(A)+f(B))` the earlier pass cited
  (`+0.57…+1.9`) appears **only** for short/medium chords centered near the
  `σ₁≈2, σ₂≈20` corner or directed along the negative eigenvector, so it is a
  chord-**placement** artifact, not the primary evidence; the Hessian eigenvalue
  is.)* `Var[min]` is also **concave in `ρ`** (`d²V/dρ² < 0` throughout:
  re-executed, `d²V/dρ² ∈ [−2.18, −1.62]`), and **strongly nonconvex jointly** in
  `(σ₁,σ₂,ρ)` (min Hessian eigenvalue `≈ −4·10³` over a random scan — the
  nonconvexity is carried chiefly by the `ρ` direction). So the variance
  **constraint** stops being a second-order cone at the tie — the **principled
  boundary for `kink_regime`**.
- **But convex in the decision variable `n`:** sweeping 28 (gap × n-range)
  regimes with `σ_i(n)=√(A_i/n_i)`, a chord-witness nonconvexity in `n`-space
  appears in **0/28** — `σ(n)=√(A/n)` tends to **restore** convexity in `n`.
  Contrast the single-arm `Σ a_i/n_i`: `d²/dn² = 2a/n³ > 0` everywhere (convex,
  SOCP-able).

**Conclusion (the honest disagreement with the review):** the review is **right**
that `Var[min]` is smooth (not non-smooth) and nonconvex in `(σ,ρ)`; its shorthand
"you land in nonconvex optimization [in n]" is **overstated** — the precise loss
is the **SOC-expressibility of the variance constraint**, the program in `n`
staying convex on every region probed.

**Corroborating codebase checks (executed against the live modules):**

- `Corr(slope, intercept) = −0.8114` on the real 7-point design
  `[32,64,128,192,256,384,512]`, and all three `(AᵀA)⁻¹` entries
  (`Var(slope)=1/Sxx`, `Var(intercept)=1/n+x̄²/Sxx`, `Cov=−x̄/Sxx`) match their
  closed forms exactly.
- Producer `{N_gen,R_gen}` ∩ serve `{T_disp,tau_io,wakeup,B,t_row,L}` = **∅**
  against `model_zmq_baseline.INPUT_NAMES` ⇒ `ρ_arms=0` exact for `zmq_baseline`.
  *(For `model_capacity`, `LPD` divides **both** producer and serve as the unit
  conversion — a genuine shared input — so its arms are not strictly disjoint;
  `LPD` is a `Fixed` pin, so its contribution to the cross-arm covariance is the
  small declared-spread term, but the `ρ_arms=0` simplification is exact only for
  `zmq_baseline`, where the producer reads `R_gen` rather than `LPD`.)*
- Stage margins at the seed: `zmq_baseline` SERVE 428.28 / GEN 456 = **6.5%**;
  `model_capacity` SERVE 419.76 / GEN 456 = **8.6%** — both fire `kink_regime`.
- `OpenTURNS getMeanSecondOrder = 432.43` (note: 432.4), **+4.16 the wrong
  direction** (true Jensen for the concave `min` is **−12.6**); the live
  `WRN - Switch to finite difference to compute the hessian` confirms OT cannot
  differentiate `min()`, so it must **not** be the kink-validity signal (§4.1).
- The 2-point pilot `{μ−σ, μ+σ}` has sample `std(ddof=1) = √2·σ` exactly, so
  `a_i/n_i = grad²·(√2σ)²/2 = grad²·σ²` — **no `/2` bug** (§7, the cleanup stands
  on the opacity and the wrong inline comment, not a fabricated error).
