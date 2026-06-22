<!--
tools/analysis/leaf_eval_bound/GLOSSARY.md
Purpose: the stand-alone legend for the leaf-eval throughput-modeling work AND the serving
  implementation it models — every active abbreviation, symbol, and hyperparameter in one place, so
  no reader has to hunt the main text or historical project context to decipher a variable (the
  Stand-Alone Principle). It defines SYMBOLS, their MEANING, and their UNIT; it deliberately does NOT
  carry live numeric values — those live in their SSOT (grounding.py for the grounded constants; the
  C++/Python code for the knobs) — so this legend cannot itself go stale on a value. Where a symbol is
  OVERLOADED or its meaning is CONTESTED / stale, it says so loudly (ADR-0002).
ADR-0005 doc discipline; ADR-0006 header. Public Domain (The Unlicense).
-->

# Leaf-eval modeling & diagnosis — glossary (as of 2026-06-23)

The consolidated legend for the leaf-eval throughput **bound** tool (`tools/analysis/leaf_eval_bound/`,
see `MANUAL.md`), the **diagnostic loop** advisory
(`docs/design/leaf-eval-impl-to-model-diagnostic-loop.md`), and the **serving implementation** they
model. One symbol, one meaning, one unit — read here, do not hunt.

**How to use.** This defines *what a symbol means*, not *what its value is right now*. Live values
have a single home: a grounded constant's value is in `contract/grounding.py`; a knob's default is in
the code that parses it (cited per entry). Look there for a number; look here for a meaning.

---

## ⚠ Contested & overloaded symbols (read first)

These are the ones that have bitten. A symbol below means **different things in different files** —
never carry one file's reading into another.

| Symbol | The trap |
| — | — |
| **`N`** | **Overloaded three ways.** (1) In the **wire serve path** (`runner_wire_batched.hpp`), `N` = `trees_per_thread`, the **overcommit multiplier** (§5). (2) In the **belief/env path** (`collected_set.hpp`, `env.hpp`), `N` = the **world / treasure count**, structurally `≤ 32` (the world-mask packs into a `uint32`). (3) In featurization (`feature_compute.hpp`), `N` is a feature/action dimension. **Stale framing (flagged 2026-06-23):** the blind-model SYNTHESIS's *"deployed default = strict-barrier, N=1"* is the **outdated** formal model — under the current code `N` is a `PipelinedBucket`-only knob and is **ignored under the `StrictBarrier` default**, so "the default is N=1" conflates two modes. Whether `N` is even live in production is the consultation's open question #2 (unresolved). |
| **`B` / `B_op`** | `B` (model symbol) = `B_op` (registry quantity) = **rows per forward** — the serve batch width. The *model* evaluates it at a full bucket; the *implementation*'s realized rows/forward is a **different, measured** number (`mean_rows_per_msg`, §5). Do not assume they are equal — their gap is the contested operating point (consultation §7.4). |
| **`K`** | `K` = `fibers_per_thread` = `ceil(pool_batch / pool_threads)` — the per-thread in-flight **slot count** in the strict path (§5). It is *not* the model's `B`, though under `StrictBarrier` the realized rows/forward is close to `K`. (Beware: `K` also appears in C++ profiling comments as an unrelated profile label, e.g. "the K=16 profile.") |
| **`L` vs `LPD`** | Same quantity: **leaves per decision**. `L` is the model symbol; `LPD` / `leaves_per_decision` is the registry quantity name. |
| **`T`** | In the serve path, `T` = `pool_threads` = OS worker threads (§5). In a model's fit, `T_disp` is a *different* `T` (dispatch, §2). Read the subscript. |

---

## 1. Throughput and the bound

| Term | Meaning |
| — | — |
| **DPS** | **Decisions per second** — the throughput quantity everything targets. (A "decision" is one recorded Gumbel-AZ move.) |
| **the bound** / `f(μ̂)` | The model's throughput **lower bound**: `f` (the cycle model) evaluated at the grounded mean point `μ̂`. Read *"under this model, at least `f(μ̂)` DPS is achievable."* It is a **denotational conjecture**, not a measured fact (MANUAL §1). |
| **lower-bound semantics** | A real well-designed cycle has every cost the model has *plus* the coordination losses the model omits, so it achieves *at least* `f(μ̂)`; a sloppy one reveals its slack as the gap below. Conservative by construction. |
| **trust ladder** | How much to believe a number, by where its inputs sit (MANUAL §6): **seeded** (first-principles priors) → **grounded-in-a-fit** (a few real fit read-offs) → **trusted-measured** (a bench was run sole-workload, the manifest flipped it `trusted=True`) → **untrusted+confounded** (live but through a confounding substrate, e.g. Python/GIL). |
| **REF_PLATEAU_DPS** | A **display reference**, value ≈ 203 — the contested "~200 DPS roof." `references.py` itself disowns it: *"a USER-supplied reference for ONE config family … NOT grounded in any readable repo file."* Never an input to the bound; there to be beaten by a witness, not matched by a model (§7). |

---

## 2. The model's cycle vocabulary

The cycle is `f(inputs) → DPS`. **There are two model dialects** with overlapping but not identical
symbol sets — `model_capacity` (called *Design-A*) and `model_cycletime` / the transport variants
(*Design-B*). **Both models are currently under revision** (the impl→model loop exists because the
formal model is outdated) — treat these symbols as the *current* vocabulary, not settled physics.
"Registry quantity" is the name the manifest resolves (the bench/grounding key).

| Symbol | Meaning | Unit | Registry quantity | Dialect |
| — | — | — | — | — |
| `N_gen` / `n_gen` | generator **cores** (producer parallelism) | cores | `n_gen` | both |
| `R_gen` | one core's decision rate | decisions/s/core | `R_gen` | B |
| `g_core` | one core's **leaf** rate | leaves/s/core | `g_core` (`GEN_PER_CORE_LEAVES`) | A |
| `B` / `B_op` | **rows per forward** (serve batch width / bucket) | rows/forward | `B_op` | both |
| `L` / `LPD` | **leaves per decision** (search tree's distinct-node count per move) | leaves/decision | `LPD` | both |
| `t_row` / `slope_us` | per-row serve **slope** (marginal forward cost per row) | µs/row | `t_row_us` (= `SERVE_SLOPE_US`) | B / A |
| `iota_us` | serve forward **intercept** — Design-A's fixed per-forward cost | µs | `SERVE_INTERCEPT_US` | A |
| `T_disp` | **dispatch floor** — pjit/XLA forward-dispatch fixed cost (Design-B's split of the fixed cost) | µs | `T_disp_us` (≈ `DISPATCH_FLOOR_US`) | B |
| `wakeup` | **first-poll readiness** latency, before `tau_io` (Design-B) | µs | `{slug}_wakeup_us` | B (variants) |
| `tau_io` | **server drain / decode / encode / scatter**, serial between forwards. *Currently UNMEASURED — the top measurement target.* | µs | `tau_io_us` (= `SERVE_IO_US`) | both |
| `tmsg` | per-**leaf** wire-message cost (the TRANSPORT stage; non-binding) | µs/leaf | `tmsg_us_leaf` (= `MSG_PER_LEAF_US`) | both |
| `cycle_us` | one serialized serve **forward** total (Design-B): `T_disp + tau_io + wakeup + B·t_row` | µs | — (derived) | B |
| `fwd_us` | Design-A's forward total: `iota_us + slope_us·B_op + tau_io_us` | µs | — (derived) | A |

**Stages** (what the `min` is over — *one* model shape, not a law; MANUAL §2):

| Stage | Meaning | Form (Design-B) |
| — | — | — |
| **GENERATION** / `producer` | producer cores' aggregate search rate | `N_gen · R_gen` |
| **SERVE** | the serialized serve-forward capacity | `1e6 · B / (cycle_us · L)` |
| **TRANSPORT** | per-leaf wire-framing capacity (ranks last / non-binding) | `1 / (L · tmsg · 1e-6)` |
| **binding stage** | the `min` arm that sets the throughput — the bottleneck. The `min()`-**kink** (`alloc/kink.py`, Clark-1961) handles a statistical tie between arms. |

---

## 3. The estimator & grounding contract

| Term | Meaning |
| — | — |
| **`Estimate`** | The typed value a bench's `measure()` returns (`contract/estimate.py`, frozen). Fields: `theta_hat` (the point(s) `f` is evaluated at), `cov` (the **already-divided** sampling covariance — an SE², not a per-sample variance), `shrink` (a `ShrinkLaw`), `support`, `family`, `kind`. |
| **`theta_hat`** | the estimate's central value(s) (a length-k vector; k=1 for most, k=2 for a fit). |
| **`cov`** | the sampling covariance of `theta_hat` (SE², already divided by n). |
| **`ShrinkLaw`** | how `cov` responds to more effort. Variants: **`Fixed`** (irreducible — a pin/prior), **`QuantileLaw`** (a median, bootstrap SE), **`RegressionLaw`** (an OLS fit, leverage-floored), `Poolwise` (a mean), `Composed` (a ratio). |
| **`Support`** | the feasible domain that clips the CI: `REAL` / `POSITIVE` / `UNIT` (or an explicit `(lo,hi)`). |
| **`CIFamily`** | the CI multiplier family: `NORMAL` (z) / `STUDENT_T(dof)` (t) / `EMPIRICAL` (the bench's own interval) / `DEGENERATE` (a pin — no interval). |
| **`kind`** | a provenance label on an `Estimate`: `pin` / `declared_spread` / `median` / `ols_fit` / `mean` / `quantile` / `ratio`. (Metadata; the math reads the `ShrinkLaw`/`family`, not this string.) |
| **estimator (pin / median / fit)** | the three factories (`benchmarks/estimators.py`): **`pin_estimate`** (a config fact or prior → `Fixed`), **`median_estimate`** (a sampled pool → `QuantileLaw`), **`fit_estimate`** (an OLS `time = intercept + slope·rows` → `RegressionLaw`). |
| **`Grounded`** | a grounded physical constant (`contract/grounded_types.py`): `name, mean, sigma, cost, unit, provenance, estimability, module`. The v1 seed a model uses before any live measurement. |
| **`Estimability`** | the measured-vs-pinned axis (the single home of that decision): **`CONSTANT`** (a true layout fact → `DEGENERATE` pin, ~0 bound contribution), **`MEASURED`** (a runnable bench measures it live → shrinkable), **`PRIOR`** (an engineering-judgement value, no runnable bench yet → `NORMAL` pin). |
| **`cost`** | a `Grounded`'s relative per-sample benchmark cost (the allocation effort price for sampling it). |
| **`SLUG`** | a transport variant's registry prefix + comparison-table key (e.g. `zmq_baseline`). |

---

## 4. The diagnostic loop (the advisory's vocabulary)

From `docs/design/leaf-eval-impl-to-model-diagnostic-loop.md` and the witness-lowering review.

| Term | Meaning |
| — | — |
| **witness** | the model **lowered to a runnable cycle** built from the benched stages, and clocked — a real end-to-end DPS. The *operational* semantics to `f`'s *denotational* one; the **adequacy witness** for `f`. Does not exist yet (the consultation's biggest open question). |
| **`gap_A`** | `f(μ̂) − witness` — the **omitted coordination loss** (RTT idle, convoy, cold-JIT) the model excludes by assumption. A *legitimate* gap; the adequacy gap between the two composition maps. |
| **`gap_B`** | `witness − implementation` — the **implementation's own slack** (it runs a suboptimal config). An *engineering* finding, not a model error. |
| **form fault** | the model's `f` is wrong even though every input it reads is faithful (a missing coupling, an omitted term, or the wrong operating point for an input). |
| **fidelity fault** | a **benchmark misrepresents** its stage — it measures something other than what the running system pays (a mock too cheap, a value biased ×N). |
| **tool-inadequate** | the third verdict (§7a): the harness/DSL **cannot resolve** the question at the granularity the gap demands (the separable `f` can't carry a coupling; `Estimate` can't carry a bimodal stage; a binding stage has no bench/observation point). The honest move is to *name the obstruction and propose the lift*. |
| **`a_i`** | a model input's **sensitivity** = `(∂f/∂x_i)² · σ_i²` — how much its uncertainty constrains the bound. The relevance weight on every verdict (`Recommendation`, Purpose 2). |
| **port** | an **instrumented seam at a carved joint** — a passive counter / ring-buffered timestamp (in the shape of `CHOCO_EVENTLOG`), parsed offline, that reads one stage's live cost without perturbing the hot path. |
| **carve-at-the-joints** | refactoring so each model term is *one callable unit* (typed signature SSOT, ADR-0012), so the witness and the implementation **share the identical stage code** and differ only in composition glue. |
| **the ladder** | building the witness as **rungs from idealization down**: rung 0 = `f(μ̂)` → rung 1 (real stages, idealized composition — a stage-fidelity check) → +transport → +parking/drain → implementation; each delta is one named coordination cost, the probe overhead canceling in the delta. |
| **Purpose 1 / Purpose 2** | the tool's two jobs: **(1)** the theoretical lower bound (explore what's possible); **(2)** the gradient/sensitivity decomposition (diagnose where a real cycle's slack or the bound's softness lives). |

---

## 5. The serving implementation (what we are explaining)

The running actor is a persistent **C++ Gumbel-AZ** process generating episodes; leaf evaluation goes
by one of two paths.

| Term | Meaning | Source |
| — | — | — |
| **serial path** | leaf eval is a *local, in-process* C++ MLP forward (`NetForward`), no wire. Its sole-workload ceiling is what `bench_r_gen` measures (`R_gen`). | — |
| **wire-batched path** | the C++ producer parks leaves; a drain gathers them into one ZMQ multipart frame; a single-threaded Python `inference_server.py` decodes, coalesces across threads, runs **one JAX forward**, scatters replies. This is the path the model's SERVE cycle abstracts. | — |
| **`WireMode`** | the transport scheduling arm: **`StrictBarrier`** (the **production DEFAULT**, untouched: each round gathers ALL parked slots into ONE message — structurally `D=1`) or **`PipelinedBucket`** (arm 3, behind a flag — the overcommit / D-pipeline). | `runner_wire_batched.hpp:59,107` |
| **`N`** (here) | `trees_per_thread`, the **overcommit multiplier**: each `PipelinedBucket` producer thread owns **N×K** independent `EpisodeSlot`s (self-contained search trees), supplying N× the parked-leaf depth per forward. `N=1` = pre-overcommit count. **Ignored under `StrictBarrier`.** (See the ⚠ table — `N` is overloaded and its "default=1" framing is stale.) | `runner_wire_batched.hpp:73`; adapter §6 M1 |
| **`K`** | `fibers_per_thread` = `ceil(pool_batch / pool_threads)` — per-thread in-flight slot count. ~54 in the cited strict-barrier measure (≈ the realized rows/forward there). | `runner_wire_batched.hpp:64` |
| **`D`** (`max_inflight_msgs`) | per-thread in-flight **message** cap for `PipelinedBucket` (`StrictBarrier` is structurally `D=1`). | `runner_wire_batched.hpp:67` |
| **`T`** (`pool_threads`) | OS worker threads in the wire pool. | `runner_wire_batched.hpp:63` |
| **`pool_batch`** | the in-flight **leaf** target across the pool (default 32; env `CHOCO_POOL_BATCH`). | `serve.hpp:36` |
| **`chunk_floor`** | when ON, the producer supplies overcommit DEPTH while the server controls forward WIDTH; when OFF, `issue()` drains ALL ready into ONE message (depth ≈ 1, the production path). | `runner_wire_batched.hpp:94-101` |
| **`IssueController`** | the online overcommit controller fixture for the `PipelinedBucket` arm. | `issue_controller.hpp` |
| **`EpisodeSlot`** | one self-contained search tree (one in-flight episode); the unit `N×K` counts. | `runner_wire_batched.hpp:74` |
| **`mean_rows_per_msg`** | live telemetry: `total_leaves / total_msgs` from the wire driver — **a direct measurement of the implementation's realized `B`** (rows/forward). | wire driver |
| **`CHOCO_EVENTLOG`** | opt-in telemetry timestamping `FWD` (a JAX forward) and `DRAIN` (a batch-drain) events in `inference_server.py` — the live serve-cycle observation. | `inference_server.py` |
| **`FWD` / `DRAIN`** | the two `CHOCO_EVENTLOG` event kinds: a forward, and a batch drain. `FWD` inter-arrival ≈ the live `cycle_us`. | — |
| **`DetNet`** | the **deterministic-net mock** the search bench uses in place of the real eval, so `bench_r_gen` isolates search-core rate without net-forward cost. | search bench |
| **redis-stall** | the producer blocking on redis I/O (the episode source) — an unmodelled coordination cost (`gap_B` candidate; SYNTHESIS open question on source timing). | — |

---

## 6. Hyperparameters & run knobs

| Knob | Meaning | Default / source |
| — | — | — |
| `--n-sims` (`n_sims`) | Gumbel-AZ **simulations per decision** | search config (e.g. `n_sims=256`) |
| `--m` (`m`) | Gumbel **top-m** sampled root actions | search config (e.g. `m=24`) |
| `--max-depth` | search tree **max depth** | search config |
| `--tasks` (`n_tasks`) | number of episode tasks | `32` (`search_runtime_bench`), `8` (others) |
| `--workers` | worker threads | `4` |
| `--reps` | bench repetitions (a `bench_r_gen` sizing knob) | `3`–`8` |
| `--pool-batch` | the wire `pool_batch` (in-flight leaf target) | `32` |
| **`SIZING_KWARGS`** | the bench sizing-knob vocabulary the allocation driver looks for on `measure()` to spend budget: `cycles, trials, iters, n_trials, reps, rounds, samples, n, budget, leaves`. | `benchmarks/harness.py` |
| `tolerance` / `confidence` / `growth_cap` | `AllocationDriver` params: the CI target (DPS), the CI confidence (0.95), and the per-round sample-growth cap. | `alloc/driver.py` |
| `UD_PILOT` / `UD_ROUNDS` / `UD_ITERS_CAP` / `UD_TOL` | `untrusted_drive` env knobs: pilot sample count, max rounds, per-input iter cap, CI target. | `runners/untrusted_drive.py` |

---

## 7. Reference anchors (`references.py` — display only, never inputs)

| Symbol | Meaning |
| — | — |
| `REF_PLATEAU_DPS` (≈ 203) | the contested "~200 roof" — a one-config user reference, code-disowned (§1). |
| `REF_GLOBAL_MAX_DPS` (≈ 468) | the analysis_clean.txt global-max (full bucket, pad=0) — grounded in a readable file. |
| `REF_PRIOR_MODEL_DPS`, `REF_SERVE_CEILING_DPS`, `REF_*_DPS_PER_CORE`, … | other display anchors (per-core ceilings, serve ceilings). Re-derive; do not anchor on them. |

---

## 8. Bands & cross-cutting

| Term | Meaning |
| — | — |
| **Band 1 / 2 / 3** | the domain-classification axis (ADR-0003): **Band 1** solver-agnostic (the `Estimate` contract), **Band 2** OR-general (the allocation driver, the model shapes), **Band 3** FFXIII-/serving-path-bound (the grounded constants). |
| **manifest / registry** | the metric store (`store/`): a name → `(mean, sigma, n, trusted)` / `Estimate` resolver, backed by **host PostgreSQL `control_research` @ 192.168.122.1:5432** (psycopg3) — *not* the `:6379` redis (that backs a different subsystem). |
| **SSOT** | single source of truth (ADR-0012: the typed signature is the SSOT — for a value, its one home). |

---

## 9. Units

| Unit | Meaning |
| — | — |
| `µs` / `us` | microseconds |
| `µs/row` | per-row serve cost (the `t_row` slope) |
| `µs/leaf` | per-leaf message cost (`tmsg`) |
| `decisions/s`, `decisions/s/core` | DPS, and per-core DPS (`R_gen`) |
| `leaves/decision` | `L` / `LPD` |
| `rows/forward` | `B` / `B_op` (the serve batch width) |
| `cores` | `N_gen` / `n_gen` |

---

*If a symbol you hit is not here, it is a gap in this legend — add it (and, per the Stand-Alone
Principle, hoist its definition into the using file's header) rather than leaving the next reader to
hunt. This glossary defines meanings; values live in their SSOT.*
