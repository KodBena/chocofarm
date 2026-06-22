<!--
docs/design/leaf-eval-bound-responsibility-refactor.md — an ADVISORY design
note proposing how to refactor tools/analysis/OpenTURNS/ (the leaf-eval
throughput lower-bound tool) BY RESPONSIBILITY: the responsibilities an
independent read of the code finds, the seams between them, where the current
flat module namespace blurs those seams, and a proposed package decomposition
that would also ease the planned OpenTURNS->JAX autodiff swap.

ADVISORY ONLY. It proposes; the maintainer reviews and decides whether to
ratify. No mandate, no code change is made by this note. ADR-0005 authoring
discipline; ADR-0006 header. ADR-0008/0011/0012 are the classification /
mechanization / structural-hygiene lenses applied. The neutrality requirement
was explicit: the decomposition is derived from the code, not from any
pre-supposed boundary.

Public Domain (The Unlicense).
-->

# Refactoring the leaf-eval bound tool by responsibility — an advisory (2026-06-22)

An **advisory** design record, authored at a decision point but **proposing
nothing binding**: it reads `tools/analysis/OpenTURNS/` as it stands, derives —
from the code alone — what each module actually *owns* and what *flows between*
the owners, and recommends a package decomposition along those seams. The
maintainer reviews and decides whether to ratify; **no code is changed by this
note, and nothing here is a mandate** (ADR-0004 no-retroactive-sweep posture is
respected — a ratified version would sequence work, this note only names it).

The neutrality requirement was the whole point of the commission: *do not anchor
on a proposed structure; let the seams fall out of what the code does.* So §2
builds the decomposition bottom-up from an inventory of every module's owned
concern (§1) before §3 names the package boundaries. Where I assert a module
"owns X and also Y," that is a read of its body, cited.

## 0. What I read end to end (ADR-0002 read-before-cite)

Every non-bench module in `tools/analysis/OpenTURNS/` in full: `estimate.py`,
`leaf_eval_grounding.py`, `neyman_driver.py` (all 1174 lines), `manifest.py`,
`bench_store.py`, `model_capacity.py`, `model_cycletime.py`,
`model_zmq_baseline.py`, `model_cpp_inproc_port.py`; the runners
`throughput_bound.py`, `transport_sweep.py`, `untrusted_drive.py`;
`benchmarks/bench_common.py` and `benchmarks/register_baseline.py` in full;
`benchmarks/bench_r_gen.py` in full as the canonical bench, and the skeletons of
`bench_tau_io`, `bench_b_op`, `bench_shm_spin_poll_wakeup`, `bench_t_disp`,
`bench_lockfree_mpsc_tmsg` (the median / pin / race-wakeup / fit / tmsg shapes).
I read the remaining `model_*` and `bench_*` modules by their import graph and
their public surface (the `grep` of every `def`, every `SLUG`/`INPUT_NAMES`/
estimator-helper call), not line by line — flagged here per ADR-0002 so the
claim "every bench has shape Z" is honestly an *enumeration of the public
surface plus five fully-read instances*, not 38 full reads.

Documents read end to end: **ADR-0008** (classification discipline),
**ADR-0011** (mechanization discipline), **ADR-0012** (all 1330 lines —
compositional & structural hygiene; the P1/P2/P3/P5/P7/P8 principles this note
leans on were read in full, and the P9/C++-component material was read but is
not relied on, as this tool is Band-1/Band-3 Python with no compiled component),
the **adr-synopsis** and **STATUS.md**, the post-mortem
**`docs/notes/leaf-eval-estimator-pin-cascade-rca.md`** (the cautionary
instance, including its 2026-06-22 addendum), and
**`docs/design/harmonized-estimator-interface.md`** (all 1629 lines — the
contract whose evolution this tool's seams now carry).

## 1. Inventory — what each module owns (the evidence the decomposition rests on)

The directory is **one flat namespace**: ~16 top-level `.py` modules plus a
`benchmarks/` subpackage of ~31 bench modules + `bench_common.py` +
`register_baseline.py`, and an `examples/` with one demo. There is **no
`__init__.py`** at the top level — modules resolve each other by
`sys.path.insert(_HERE)` (every module repeats this preamble), and `manifest` /
`bench_store` / `untrusted_drive` import siblings as bare names (`import
estimate`, `import manifest`). The flatness is the condition the rest of this
note responds to.

Reading each module for the *one concern it owns* (the ADR-0012 P3 "name it in
one clause; if you need an 'and', it is two collaborators" test):

| Module | The concern it owns (one clause) | "and also…" (the seam strain) |
| — | — | — |
| `estimate.py` | the **typed `Estimate` contract + ShrinkLaw sum type + its (de)serialization** — pure, numpy-only, touches no SQL, no measurement, no allocation (its own header says so) | — clean. The one genuinely single-responsibility module. |
| `leaf_eval_grounding.py` | the **SSOT of the grounded physical constants** (`Grounded` table) | …**and** the `Estimability`/`Grounded` *types*, **and** a block of `REF_*` reference literals (`REF_PLATEAU_DPS`, `REF_GLOBAL_MAX_DPS`, …) that are a *different* concern (display anchors, not model inputs) |
| `bench_store.py` | the **Postgres egress** — connection + schema + the definition/instance/sample tables + `Estimate` jsonb I/O (its header: "owns ONLY the SQL/connection + schema") | — clean, by design. |
| `manifest.py` | the **registry resolver** — name → `(mean,sigma,n,trusted)` / `Estimate`, TRUST/SEED/reconstruct paths, pg-down degradation | …**and** the seed→Estimate / aggregate→Estimate **reconstruction + projection math** (`_estimate_from_seed`, `_estimate_from_aggregate`, `_project_estimate`) — statistical glue that is the manifest's, but is *consumed by the runners directly* (`throughput_bound`, `transport_sweep` call `manifest._estimate_from_seed`) |
| `neyman_driver.py` (1174 L) | the **generic allocation engine** | …a god-cluster: the `Estimate`-seam, the `gᵀΣg` quadratic form + Σ assembly, the SOCP allocation, the **Clark-1961 min-kink machinery**, the per-family CI multiplier, the **`Recommendation` report formatter**, **and** the autonomous `run()` loop — six+ concerns in one class/file (see §2.3) |
| `model_capacity.py`, `model_cycletime.py` | a **static-grounded throughput model** (its `f` + numpy twin + grounded-input wiring + driver factory + model diagnostics) | the *pattern* differs from the variant models (§2.4) |
| `model_zmq_baseline.py` + 4 `model_<slug>.py` | a **manifest-driven transport-variant model** (same, but resolving inputs through the manifest, with a `SLUG`) | two **divergent model dialects** with no shared base (§2.4) |
| `throughput_bound.py` | a **runner**: the two static models' seeded bound + Neyman ranking, OT-or-numpy | …**and** its own `_fd_gradient`, its own numpy delta-method bound (`_numpy_bound`), its own Neyman-table formatter |
| `transport_sweep.py` (640 L) | a **runner**: the 5-variant sweep, three honesty levels, optimum-over-transports | …**and** a *second* `_fd_gradient`, a *second* numpy bound, the per-variant transfer-policy table (`VARIANTS`), the `getattr`-shims that paper over the two model dialects (`_model_sigmas`, `_registry_qname`, `_untrusted`, `_model_estimates`) |
| `untrusted_drive.py` | a **runner**: live-bench mechanism test (every input measured now) | …**and** a *third* copy of `_registry_qname` (its own docstring admits the duplication), the ETA/progress machinery, the measurer-construction (`_make_measurer`) |
| `benchmarks/bench_common.py` (541 L) | the **bench↔store glue** (`logged_run`, `register_quantity`, `warm`, `SIZING_KWARGS`) | …**and** the **estimator factories** (`fit_estimate`, `median_estimate`, `pin_estimate`) — *pure numpy statistics*, a different concern from the SQL-touching glue — **and** the pool builders (`collect_pool`, `window_pool`) |
| `benchmarks/bench_<name>.py` (×31) | **one physical quantity's measurement** (its `_measure_raw`, `_estimate_from_raw`, `measure`, `run`, `get_seed`, `register_self`) | the ~30-fold templated `_measure_raw`/`_estimate_from_raw` shape is the RCA's cancer-D surface (§2.5) |
| `benchmarks/register_baseline.py` | a **one-shot registration script** (a hand-listed `BASELINE_BENCHES`) | the hand-list is an ADR-0011 Rule-4 enumeration (§2.5) |
| `examples/demo_msgpass.py` | a **worked synthetic example** of the driver | — clean, correctly extracted out of the driver already (P1 purification). |

The **directory name itself is a misclassification** (ADR-0008, positive
register): the package is named `OpenTURNS`, after the *library it currently
uses for autodiff* — but its responsibility is "a provable throughput
lower-bound for the leaf-eval serving path," and OpenTURNS is one swappable
dependency of one layer (the gradient), about to be replaced by JAX (§5). A
folder named after a soon-to-be-removed dependency is a vocabulary that does not
fit the territory; this is called out in §4 as the highest-value rename.

## 2. The responsibility decomposition (derived bottom-up)

Reading the inventory for *what flows between modules*, six responsibilities
fall out. I derive each from the data that crosses its boundary, not from a
layering I brought in.

### 2.1 The contract (what every measurement *is*)

`estimate.py` already isolates this perfectly: the frozen typed `Estimate`, the
`ShrinkLaw` sum type, the `Support`/`CIFamily` vocabularies, and the jsonb
(de)serialization — numpy-only, no SQL, no measurement, no allocation. **Every
other layer is phrased over this type.** It is the keystone, and it is already
clean; the decomposition's job is to make the rest *as* clean, not to touch this.

This responsibility's boundary is the type. The thing that flows across it is an
`Estimate` value. Nothing else belongs here (it correctly does not know what a
"bench" or a "model" or "postgres" is).

### 2.2 The grounding (the domain facts the bound rests on)

`leaf_eval_grounding.py` owns the FFXIII-/serving-path-specific physical
constants — this is **Band 3** (ADR-0003: instance-bound facts, isolated). The
seam here is *already slightly overloaded*: the module mixes three things that
read the same but are not the same concern —

1. the `Grounded` **dataclass + `Estimability` enum** (a *type/vocabulary* — the
   single-home measured-vs-pinned axis the RCA's fix #1 introduced);
2. the **grounded-constant table** (`SERVE_INTERCEPT_US`, `GEN_PER_CORE_DPS`, …)
   — the actual Band-3 data;
3. the `REF_*` **reference literals** (`REF_PLATEAU_DPS = 203`,
   `REF_GLOBAL_MAX_DPS = 468`, …) — display/comparison anchors that are *not
   model inputs*, consumed only by the runners' headers and `model_zmq_baseline.
   ref_plateau_dps()`.

These are three responsibilities (a vocabulary type, a data table, a set of
named reference points). The data flowing out of (2) is `Grounded` values that
become `Estimate`s; the data flowing out of (3) is bare floats used only in
print statements. They share a file by history, not by concern.

### 2.3 The allocation engine (the generic OR machinery)

`neyman_driver.py` is the **Band-2 OR-general** core: it owns *no model* (P1/P2,
correctly — the demo was extracted). But at 1174 lines it has become the god-
object ADR-0012 P3 names. Reading its methods, it bundles **six distinct
collaborators** that a `min()`-free all-means model would not all need:

- **A — the input seam.** `set_estimate` / `set_estimates_by_name` /
  `_estimate_for` / `_effective_n` — accept an `Estimate` per input or wrap a
  raw pool. (The dual-mode `Estimate`-or-pool ingestion.)
- **B — the variance form.** `_assemble_sigma` (block-diagonal Σ + declared
  cross-terms + the DEGENERATE-zeroing) and the `gᵀΣg` quadratic in `step()`.
- **C — the allocation solver.** `_fundability` (the typed-D2-marginal home),
  `_socp_allocation` (the sign-safe `Q`-form SOCP + the `gᵀΣ(n*)g ≈ V*`
  assertion), `_closed_form_allocation`.
- **D — the kink machinery.** `_kink_assessment`, `_model_arms`,
  `_KINK_PFLOOR`, the Clark-1961 Φ/φ min-moments. This is the single densest,
  most self-contained block — pure statistics over arm covariances, reachable
  only when a model exposes `arms_fn`.
- **E — the CI report.** `_family_multiplier`, `_report_sigma`, and the whole
  `Recommendation` dataclass with its `report()` / `where_to_spend()` /
  prose-formatting (the `var_floor` / `var_shrinkable` lines).
- **F — the driver loop.** `run()` (the autonomous pilot→step→re-measure loop,
  measurers-or-samplers, the stall-stop).

A, B, C, D are the *generic allocation mathematics*; E is *presentation*; F is
*orchestration*. The audit's own R-series did exactly this kind of split for the
analyzer (presentation split out) and `exit_loop` (`RunConfig` out). The same
move applies: the math is one owner, the `Recommendation`-formatter is another,
the `run()` loop is a third. **D (the kink) is the cleanest candidate for its
own module** — it is parameter-free, pure, and depends on nothing in the driver
but the arm covariances; it reads like a transplant from SSTA (its own comments
say so) and would be unit-testable in isolation the moment it is lifted out.

### 2.4 The models (the things-being-bounded — and their dialect split)

A "model" owns: an expression `f`, its numpy twin, the input wiring, a driver
factory, and model-specific diagnostics (`stage_capacities` / `cycle_breakdown`
/ `serve_sawtooth` / `inproc_port_contrast`). This is **Band-2/Band-3 mixed**:
the *shape* (min-of-stages, a serialized-serve cycle) is OR-general; the
*grounded numbers* are Band-3.

The strain here is a **two-dialect split with no shared base** (ADR-0012 P3 / the
audit's cancer D — "copy-paste programs"):

- **Static-grounded dialect** (`model_capacity`, `model_cycletime`): reads
  `G.Grounded` directly, exposes module-level `SIGMAS`/`COSTS`/`NEEDS_
  MEASUREMENT`, `initial_point()` with **no `trust` arg**, no `SLUG`.
- **Manifest-driven dialect** (`model_zmq_baseline` + 4 variants): resolves
  inputs through `manifest.quantity`, exposes `sigmas(trust=…)` /
  `trusted_flags()` / `bound()`, `initial_point(trust=…)` **with** a `trust`
  arg, a `SLUG`, and an `INPUT_QUANTITIES` **or** `_MANIFEST_NAME` map (the two
  variants of *that* are themselves inconsistent — `zmq_baseline` uses
  `INPUT_QUANTITIES[nm] = (qname, cost)`, the other four use `_MANIFEST_NAME[nm]
  = qname`).

The cost of the divergence is **paid in the runners**: `transport_sweep` and
`untrusted_drive` each carry `getattr`-based shims — `_model_sigmas` (`hasattr
(model,'sigmas')` else `model.SIGMAS`), `_registry_qname` (`INPUT_QUANTITIES`
else `_MANIFEST_NAME`), `_untrusted` (three fallbacks), `_model_estimates`. Each
shim is a Port translating between two model dialects that *should be one*. A
shared model interface (a small base or a typed `Protocol`) is the responsibility
this split is missing — and `_registry_qname` being **physically duplicated**
between the two runners (verbatim, the docstring admitting it) is the P1
violation the missing interface causes.

`THROUGHPUT_EXPR` (a muParser string) and `throughput_numpy` (a Python function)
are **two hand-maintained homes of the same `f`** in every model — a P1
dual-write that the JAX migration dissolves (§5), but is a real present-day
hazard (a model author edits one and not the other and the OT path silently
diverges from the numpy cross-check).

### 2.5 The benches (the measurement leaves)

Each `bench_<name>.py` owns one quantity's measurement. The layering here is
**clean in one direction** (no bench imports the driver or a model — verified;
benches sit strictly below the model/driver layer) and **strained in another**:
the `_measure_raw` → `_estimate_from_raw` → `measure`/`run`/`get_seed`/
`register_self` skeleton is **templated ~30 times** (the verified count: 31
`_measure_raw`, 30 `_estimate_from_raw`). This *is* the RCA's originating
defect: the migration "stamped the same three-function shape into ≈30 files," and
because the measured-vs-pinned decision was re-made per file, it was made wrong
eight times (the `Fixed`-pin cascade). The RCA's fixes have since landed
(`bench_common.collect_pool` / `window_pool` are the shared pool builders;
`Estimability` is the single-home classification — its 2026-06-22 addendum
records this), so the *acute* duplication is reduced. What remains is **structural,
not yet a defect**: there is no shared bench *scaffold* that owns the
`measure = _estimate_from_raw(_measure_raw(...))` wiring, so a new bench still
hand-copies it.

`bench_common.py` itself straddles **two responsibilities**: the *estimator
factories* (`fit_estimate` / `median_estimate` / `pin_estimate` — pure numpy
statistics over a pool, the Band-1/Band-2 partner of `estimate.py`) and the
*store glue* (`logged_run` / `register_quantity` — Postgres-touching, the
partner of `bench_store.py`). A bench that only computes an `Estimate` (the
`untrusted_drive` path, `measure()`) imports the whole module and drags the SQL
surface with it. These are two owners wearing one file.

`register_baseline.py`'s hand-listed `BASELINE_BENCHES` is the ADR-0011 Rule-4
enumeration-that-fails-open: a new baseline bench must be remembered into the
list (the RCA's smoking-gun shape). The registry is *already* discoverable
(`manifest.discover()` enumerates the definition table); a discovery-driven
registration would close it.

### 2.6 The runners (the entry points)

Three top-level programs — `throughput_bound` (static models, seeded),
`transport_sweep` (variant sweep), `untrusted_drive` (live-bench mechanism test)
— each owning *one orchestration*. They share, today by copy: `_fd_gradient`
(three copies), a numpy delta-method bound (two copies), the model-dialect shims
(§2.4), the report formatting. This is the audit's cancer D again at the runner
layer — "the Nth bespoke `main()` differing only in one literal." The shared
parts (the numpy fallback bound, the gradient, the model-dialect adapter) are a
*runner-support* responsibility that has no home, so it lives thrice.

## 3. The proposed package layout (advisory)

Mapping the six responsibilities onto a package. This is a **relocation +
subtraction**, in the audit's idiom — almost nothing is rewritten; modules move
to a directory that names their owner, and the duplicated glue collapses to one
home. A top-level `__init__.py` makes it a real package (closing the
`sys.path.insert` preamble every module repeats).

```
tools/analysis/leaf_eval_bound/          # renamed from OpenTURNS (§4)
  __init__.py                            # NEW: a real package; kills the sys.path preamble
  contract/
    estimate.py                          # unchanged — the Estimate type SSOT (§2.1)
    grounding.py                         # leaf_eval_grounding's Grounded table (§2.2 part 2)
    grounded_types.py                    # Grounded + Estimability (§2.2 part 1) — the vocabulary
    references.py                        # the REF_* anchors (§2.2 part 3) — display points, not inputs
  store/
    bench_store.py                       # unchanged — the Postgres egress (§2.2/2.5)
    manifest.py                          # the registry resolver (§1) …
    reconstruct.py                       # …with _estimate_from_seed / _from_aggregate / _project lifted
                                         #   out (the statistical glue the runners reach into — §2.2)
  alloc/                                 # the generic OR engine, split per §2.3
    driver.py                            # A+B+C+F: the seam, the quadratic form, the SOCP, run()
    kink.py                              # D: the Clark-1961 min-moment machinery (pure, transplantable)
    report.py                            # E: Recommendation + its formatting
    gradient.py                          # the gradient backend (OT today; the JAX swap seam — §5)
  models/
    base.py                              # NEW: the one model interface both dialects satisfy (§2.4)
    capacity.py, cycletime.py            # the static-grounded models
    transport/
      zmq_baseline.py, shm_spin_poll.py, futex_wake.py,
      lockfree_mpsc.py, cpp_inproc_port.py
  benchmarks/
    estimators.py                        # fit_estimate / median_estimate / pin_estimate (pure stats)
    pools.py                             # collect_pool / window_pool
    harness.py                           # logged_run / register_quantity / warm / SIZING_KWARGS (store glue)
    scaffold.py                          # NEW: the measure=_estimate_from_raw(_measure_raw()) wiring (§2.5)
    bench_*.py                           # the ~31 quantity benches (each thinner, scaffold-driven)
    register.py                          # discovery-driven registration (replaces the hand-list — §2.5)
  runners/
    support.py                           # the ONE numpy-fallback bound + _fd_gradient + model adapter
    throughput_bound.py, transport_sweep.py, untrusted_drive.py
  examples/
    demo_msgpass.py                      # unchanged
```

The **load-bearing moves** (each closes a named seam, ranked by leverage):

1. **Split `bench_common` into `estimators.py` (pure stats) + `harness.py`
   (store glue) + `pools.py`** (P3 one-owner). The estimator factories are the
   numpy partner of `estimate.py`; they should not drag the SQL surface. Highest
   leverage because it is the cleanest cut and unblocks (2).
2. **Lift `_estimate_from_seed` / `_estimate_from_aggregate` / `_project_estimate`
   out of `manifest.py` into `reconstruct.py`** — they are reached into *by the
   runners directly* (`throughput_bound._ot_bound`, `transport_sweep.
   _model_estimates` call `manifest._estimate_from_seed`), which is the tell that
   they are a shared responsibility, not the manifest's private internals (P1/P2:
   a thing two callers reach into across a boundary is not private).
3. **Introduce one model interface (`models/base.py`) both dialects satisfy**,
   and delete the runner shims (`_model_sigmas`, `_registry_qname`, `_untrusted`,
   `_model_estimates`). This removes the *verbatim-duplicated* `_registry_qname`
   (P1) and the three-fallback `getattr` chains (P2: a Port should translate one
   declared shape, not sniff for whichever of three a model happens to expose).
4. **Lift the Clark-1961 kink machinery into `alloc/kink.py`** (P3). It is pure,
   self-contained, and the single densest block in the 1174-line driver; lifting
   it makes both it and the driver readable, and it becomes unit-testable on
   synthetic arm covariances without an OT function.
5. **Collapse the three runner `_fd_gradient` + two numpy-bound copies into one
   `runners/support.py`** (P1, cancer D at the runner layer).
6. **A bench `scaffold.py` owning the `measure=_estimate_from_raw(_measure_raw())`
   wiring**, so a new bench declares only `_measure_raw` + the estimator choice
   — the structural close of the RCA's templated-shape origin (the acute form is
   already fixed; this is the prophylactic for the *next* bench).
7. **`register.py` discovery-driven** (ADR-0011 Rule 4): register every bench the
   manifest can discover, not a hand-list.

Moves 1–5 are **subtraction and relocation** in the audit's exact sense (no new
mathematics, behavior-preserving — they would carry the ADR-0009 bar: the bound
numbers must be unchanged, asserted on the existing tests). Moves 6–7 are small
new scaffolds. The whole is incremental: each box can land on its own commit,
and `contract/estimate.py` need never be touched.

A note on **import cycles** (the one real hazard a careless split creates):
`models/` import `alloc/` (the driver) and `store/manifest`; `alloc/` imports
`contract/estimate` only; `store/` imports `contract/`; `benchmarks/` import
`contract/` + `store/harness`; `runners/` import everything below. This is a
clean DAG **iff** `alloc/` does not import `models/` — which it already does not
(the `arms_fn` hook is injected by the model onto the driver, P2, precisely so
the driver never imports a model). The decomposition *preserves* the one
inversion-of-control seam the tool already got right; it does not introduce a
cycle.

## 4. Rename recommendations (agency conferred, exercised sparingly)

The maintainer conferred renaming agency and noted the tool has drifted from
strict "Neyman" allocation. I recommend **three** renames and explicitly decline
to mass-rename.

- **`OpenTURNS/` → `leaf_eval_bound/` (strong recommendation).** The folder is
  named after a *swappable dependency of one layer*, not its responsibility, and
  that dependency is about to be removed (§5). After the JAX swap, a directory
  named `OpenTURNS` that imports no OpenTURNS is the ADR-0008 fossil-label
  failure (a stale categorisation the next reader reads as authoritative). The
  honest name is the tool's job: a leaf-eval throughput lower bound. (The
  substitution test, ADR-0008: the same "named after the current backend" shape
  on a load-bearing surface mis-names the thing for every future reader; the cost
  is low here but the discipline calibrates to the shape.)

- **`neyman_driver` → `alloc/driver` (recommend, with the rename the maintainer
  flagged).** The maintainer is right that the engine is **no longer strict
  Neyman**: the harmonized-estimator doc's own §2.3 establishes it is a
  *cost-constrained c-optimal experimental design solved as a SOCP*, of which
  Neyman's `√(a_i/c_i)` is only the diagonal special case, and the Clark kink
  path is not Neyman at all. The honest module name is its responsibility
  (*allocation*), with the docstring naming the c-optimal-design lineage (it
  already does). "Neyman" is accurate for one branch; the *file* should be named
  for the whole.

- **`leaf_eval_grounding` → `contract/grounding` (+ the `references`/
  `grounded_types` split).** Minor; the split (§2.2) is the substantive part, the
  rename just follows the file to its owner (ADR-0005 Rule 5: location reflects
  content).

I **decline** to rename the per-quantity benches, `SLUG`, `INPUT_NAMES`,
`THROUGHPUT_EXPR`, or the `Estimate` field names: these vocabularies *fit* (the
ADR-0008 honest-reuse test — verify the vocabulary still fits before reaching for
a new name; these do). Renaming agency is a scalpel, not a sweep.

## 5. How the OpenTURNS → JAX migration factors in

The planned swap replaces the OpenTURNS autodiff/gradient machinery with JAX. The
decomposition **eases it, and the eased seam is narrow and already half-isolated**:

The OT/autodiff surface is, by the import map, exactly **three sites**:

1. `neyman_driver._gradient` (OT `f.gradient()`), `_fd_gradient` (the central-FD
   fallback), and `_second_order_mean` (`TaylorExpansionMoments` — already
   demoted by the harmonized-estimator doc §4.1 to "smooth-region diagnostic
   only; blind to the kink," so the JAX swap can *drop* it rather than port it);
2. each `model_*.build_symbolic_function()` (`ot.SymbolicFunction(INPUT_NAMES,
   [THROUGHPUT_EXPR])`) — the OT representation of `f`;
3. the runners' `_fd_gradient` (three copies) — the numpy fallback gradient.

What the decomposition does for the swap:

- **It gives the gradient one home (`alloc/gradient.py`).** Today the gradient is
  obtained three ways (OT analytic, OT-FD fallback inside the driver, runner-FD
  in numpy) at three+ sites. A single gradient-backend seam — a function
  `grad_f(f, point) -> ndarray` the driver and runners both call — is the Port
  (P2) at which OT is swapped for `jax.grad` *once*. The driver's `step()` calls
  `self._gradient(mu)`; if that resolves through one backend module, the swap is
  a one-file change, not a hunt. This is the P7 cross-boundary lesson applied to
  the *autodiff* boundary: one authoritative definition, every caller derives.

- **It collapses the `THROUGHPUT_EXPR` ⊕ `throughput_numpy` dual-home — which is
  the swap's biggest simplification.** Today every model writes `f` **twice** (a
  muParser string for OT + a Python function for the numpy fallback/cross-check),
  a standing P1 dual-write. A JAX `f` is **one** function that is *both*
  evaluable and differentiable (`jax.grad(f)`), so the migration naturally
  *removes* a duplication rather than adding a third representation. The model
  interface (`models/base.py`, move 3) should therefore be designed so a model
  declares `f` **once** as a JAX-traceable callable; the numpy twin and the OT
  string both retire. **This is the single strongest argument for doing the model-
  interface unification (§3 move 3) before/with the JAX swap** — the swap wants a
  one-`f` model, and the dialect unification is what produces one.

- **It quarantines the `min()` non-differentiability decision in `alloc/kink.py`.**
  The kink is the one place autodiff is fundamentally delicate (`min` has no
  honest analytic gradient — OT silently FD-falls-back; JAX's `grad(min)` gives a
  subgradient that picks one arm). The harmonized doc's resolution (Clark-1961
  closed-form min-moments, fed by `arms_fn`) is *independent of the autodiff
  backend* — it consumes arm capacities + their covariances, not `f.gradient()`.
  Lifting it to `alloc/kink.py` (§3 move 4) means the JAX swap touches the
  *smooth-gradient* path only and leaves the kink machinery — the subtle part —
  untouched. That is the migration de-risked: the backend change cannot perturb
  the kink, because the kink does not call the backend.

- **`support`/`family`/`shrink` carry through unchanged.** The `Estimate`
  contract is autodiff-agnostic (it is consumed as Σ, a numpy matrix); JAX does
  not touch `contract/`. So the swap's blast radius is `alloc/gradient.py` + the
  model `f` representation — exactly the two surfaces the decomposition isolates.

**The decomposition does not *impede* the swap and materially eases it**, and the
one piece of advice the swap adds to the refactor is: **unify the model interface
around a single JAX-traceable `f` (retiring the string⊕numpy dual-home) as part
of the same arc** — the two changes want each other.

## 6. Honest accounting (claims-measured-vs-interpreted)

- **Measured / read from the code:** the module inventory (§1), the import graph
  and the three OT/autodiff sites (§5) are read directly (`grep` of every import,
  every `def`, the 5 fully-read modules + the public surface of the rest). The
  two model dialects and the verbatim `_registry_qname` duplication are read from
  the source. The templated-bench count (31/30) is a `grep` count.
- **Interpreted (my reading, offered as advisory, not proven):** that the
  proposed boundaries are *the* right ones is a judgment, not a theorem —
  ADR-0008's negative register says under ambiguity default to the honest flat
  description and name the incompleteness, so: a defensible alternative keeps
  `alloc/` as one module (the driver is internally coherent and the split is for
  readability, not correctness), and a defensible alternative leaves the two
  model dialects as-is if the variant family is considered frozen. I recommend
  the splits because the audit's own R-series made exactly these moves elsewhere
  (analyzer presentation, `exit_loop` RunConfig, Transport⊥Pool⊥Task) and because
  the JAX swap rewards the model-`f` unification — but the maintainer's verdict on
  "is the variant family still growing?" is the input I do not have, and it
  governs whether §3 move 3 is worth its cost.
- **Not claimed:** I did not run the tool, the benches, or the driver (this is a
  read-only-on-code commission); no behavioral or performance claim is made — the
  proposed moves 1–5 are asserted *behavior-preserving by construction* (pure
  relocation), which a ratifying implementation must *verify* under the ADR-0009
  bar (the bound numbers unchanged on the existing tests), not take on this
  note's word.

## 7. What this note is not

It is **not** a mandate, not a commissioned R-series item, and not a scheduled
task — it is one independent read offered for the maintainer to accept, amend, or
reject. Per ADR-0004 it sweeps nothing; per ADR-0005 Rule 8 a ratified version
would be a *new* record (or an append), not a rewrite of this one. If ratified,
the natural enforcement surface (ADR-0011 Rule 1) for the layout is **review +
the import-cycle DAG of §3** (a `tests/` import-direction check is the mechanical
form, if the split recurs into a cycle); until then it is review-only, declared
as such.

---

## Landed — 2026-06-22 (append per ADR-0005 Rule 8)

This advisory was ratified; the work landed on `feat/issue-control-lab`. The body above is
left intact as the point-in-time advisory it was (Rule 8: amend by append, never rewrite) —
this section records what became of each proposal, in the §6 honest register (verifiable in
the git log and on disk, not on this note's word).

**§3's package layout landed in full.** `tools/analysis/leaf_eval_bound/` is now the
`contract/ · store/ · alloc/ · models/ · benchmarks/ · runners/` package §3 drew, with a real
top-level `__init__.py` (the per-module `sys.path.insert` preamble is gone). Move by move:

- **1** — `bench_common` → `estimators.py` / `pools.py` / `harness.py` (`075147f`).
- **2** — `reconstruct.py` lifted out of `manifest.py` (`8d34957`).
- **3** — one model interface: the typed `TransportModel` contract + a variant-family
  conformance net (`7ad7ae7`).
- **4** — the §2.3 driver god-object split into `alloc/{driver,kink,report,gradient}.py`
  (A+B+C+F in `driver`, the Clark-1961 D in `kink`, the `Recommendation`/E in `report`; `c3f9e4f`).
- **5** — landed *in substance, relocated*: the three-copy runner `_fd_gradient` (+ numpy-bound)
  triplication is gone (verified — no `_fd_gradient` remains in any runner), but its home is
  **`alloc/gradient.py`** (+ `alloc/jax_backend.py`), not the advisory's proposed
  `runners/support.py`. The §5 JAX swap made the gradient backend the natural single home, and
  the runners now reach the gradient through the driver's one seam. A deviation in *placement
  only* — the copy-paste defect move 5 named is closed.
- **6** — a bench `scaffold.py` owning the `measure=_estimate_from_raw(_measure_raw())` wiring,
  adopted across all 30 benches (`4e3f089`).
- **7** — discovery-driven registration replacing the hand-list (`register_baseline.py` →
  `register_benches.py`; `9cff51a`).

**§4 renames landed**: `OpenTURNS/` → `leaf_eval_bound/` (`c1d954f`), `neyman_driver` →
`alloc/driver` + `NeymanDriver` → `AllocationDriver` (`948858f`), and the
`grounding`/`grounded_types`/`references` split (`9d713f1`). The declined renames (benches,
`SLUG`, `THROUGHPUT_EXPR`, the `Estimate` fields) were left as recommended.

**§5 is underway**: the `THROUGHPUT_EXPR ⊕ throughput_numpy` dual-write is dissolved (`0dc2769`),
and `alloc/jax_backend.py` is the gradient-backend seam the swap drops into.

**Verification (ADR-0009)**: every stage gated on the full `tests/` suite — **732 passed,
33 skipped** (the baseline), behavior-preserving throughout (relocation + subtraction; the
bound numbers unchanged). The independent before/after audit of this arc is
`docs/notes/leaf-eval-refactor-audit-2026-06-22/`.
