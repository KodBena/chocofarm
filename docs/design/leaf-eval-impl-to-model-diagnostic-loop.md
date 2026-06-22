<!--
docs/design/leaf-eval-impl-to-model-diagnostic-loop.md
Purpose: an ADVISORY design for an LLM-driven, human-step-gated, mereological diagnostic loop that
  works from the running implementation TOWARD the leaf-eval model, to explain why the running
  control loop (~200 DPS) underperforms the model's throughput lower bound (~420-456 DPS). It
  designs the steps + their order, the discrepancy metric, the form-vs-fidelity attribution method
  (the crux), the human gate, the token economy, and the failure modes. It proposes the loop on top
  of the existing tools/analysis/leaf_eval_bound/ tool (its Estimate/driver/sensitivity machinery)
  and the existing implementation instrumentation surface; it does not build it.
ADVISORY ONLY — it proposes; the maintainer reviews and decides. No mandate, no code change is made
  by this note. ADR-0005 authoring discipline; ADR-0006 header; ADR-0002 fail-loud; ADR-0009
  substantiation; ADR-0012 the typed signature is the SSOT.
Public Domain (The Unlicense).
-->

# An implementation-to-model diagnostic loop for the leaf-eval bound — an advisory (2026-06-22)

An **advisory** design record, authored at a decision point but **proposing nothing binding**. It
designs a loop that iteratively reconciles a *model* of the leaf-eval serving cycle against the
*running implementation*, attributes the discrepancy to a cause, and is gated step-by-step by the
maintainer. The maintainer reviews and decides whether to ratify; **no code is changed by this
note, and nothing here is a mandate.**

This was commissioned as an **independent** consultation with an explicit instruction to interrogate
the framing rather than ratify it. So §4 sets out the maintainer's specified components, §5 names
the one place I believe the frame is incomplete (it has no name for the artifact the tool was built
to produce — the *witness*), and the design that follows honors the components where they are right
and corrects them where I judge they have a gap. Where I assert the tool "does X," it is a read of
its code or its MANUAL, cited.

---

## 0. What I read end to end (ADR-0002 read-before-cite)

Read **in full**: the tool's own `tools/analysis/leaf_eval_bound/MANUAL.md` (all 8 sections — the
two purposes, the bound-vs-witness epistemics §1/§2/§2.1, the model/bench/grounding/runner
contracts, the trust ladder, the traps); `docs/adr-synopsis.md`; this project's `CLAUDE.md` and
`docs/STATUS.md`; and these tool modules end to end: `alloc/driver.py` (the `AllocationDriver` — the
`gᵀΣg` quadratic form, the Neyman allocation, the kink dispatch, `run()`), `alloc/report.py` (the
`PrimitiveState`/`Recommendation` surface and `report()`), `alloc/kink.py` (the Clark-1961
min-moments), `contract/estimate.py` (the `Estimate` keystone + the `ShrinkLaw` sum type),
`contract/grounding.py` (the grounded-constant table), `contract/references.py` (the `REF_*`
anchors), `benchmarks/bench_r_gen.py` (the canonical subprocess bench), `runners/untrusted_drive.py`
(the live-bench loop), `runners/transport_sweep.py`, `models/model_cycletime.py`,
`models/model_cpp_inproc_port.py`, and `models/model_base.py` (the `TransportModel` contract).

Read **in full** on the implementation/diagnosis side: `docs/design/cpp-eval-transport-adapter.md`
(the §5/§6/§7 measured A/B records — the source of the grounding's MEASURED provenances and the
~189-dps-at-N=9 curve), and `docs/design/stall-investigation/blind-model-v2/SYNTHESIS.md` (the prior
blind model-of-the-implementation, all 10 sections — the depth-1 spine, the strict-barrier-default
finding, the negative-feedback batch-size fixed point, and its open questions).

Read **partially, flagged**: `models/model_capacity.py` (the first ~90 lines — its framing and
`throughput_jax`; enough to state its `min(GENERATION, SERVE, TRANSPORT)` shape, not its diagnostics
verbatim); `docs/design/leaf-eval-bound-responsibility-refactor.md` (the first ~120 lines — read for
this directory's house style for an advisory note, not relied on for content).

**Did NOT read directly (a real gap, named per ADR-0002):** the C++ sources themselves —
`cpp/src/serve.cpp`, `cpp/src/runner_wire_batched.cpp`, `cpp/src/runner.cpp`, `cpp/src/gumbel.cpp`,
`cpp/src/search_runtime_bench.cpp`, `cpp/src/main.cpp`, and `chocofarm/az/inference_server.py`,
`chocofarm/az/exit_loop.py`, `chocofarm/az/parallel.py`. The implementation map in §3 below is
sourced from a commissioned exploration agent's read of those files **plus** the parts of them the
blind-model SYNTHESIS and the grounding table quote with `file:line` provenance (which I read in
full). I have marked every claim that rests **only** on the agent's read as *(agent-sourced)* so the
maintainer can tell first-hand evidence from second-hand, and so the loop's Step 0 (§6) re-grounds
them against the live code before any of them is acted on. This is itself an instance of the
discipline the loop enforces: *verify the artifact, not the report of it.*

---

## 1. The problem, stated precisely

The tool computes a **model-contingent throughput lower bound** for the leaf-eval serving cycle:
under the current models, *"at least ~420-456 DPS is achievable"* (MANUAL §1, §5.1). The running
implementation sustains a contested **~200 DPS** (`references.REF_PLATEAU_DPS = 203.0`, with the
in-code comment that it is *"a USER-supplied reference for ONE config family … NOT grounded in any
readable repo file"*).

The MANUAL is emphatic and correct about the epistemics (§1, §2, §8): **a model-bound is a
denotational conjecture, not a refutation.** It motivates; it does not prove ~200 is beatable, and it
does not by itself explain why ~200 is where the system sits. The honest reading is:

> The gap between ~200 (measured, one config) and ~430 (a model's optimistic lower bound) is **not a
> defect report**. It is a *discrepancy between two different objects* — a number the model denotes
> and a number a particular implementation produces — and the gap can live in **either** object: the
> model can be too generous (it omits a coupling, or it trusts a benchmark that lies), or the
> implementation can be leaving real headroom on the table (it runs a config the model assumes away).

The loop's job is to **localize the gap to a cause** while being honest, every round, that the cause
can sit on the model side *or* the measurement side. The maintainer's components (§4) name this
two-sided suspicion exactly; the contribution of this note is the *method* that makes the
localization decisive rather than a vibe.

A concrete instance — already on the table from the prior work — fixes intuition. The model's serve
stage is evaluated at a **full bucket** `B_op = 256` rows/forward (`grounding.py: SERVE_FULL_BUCKET`,
classified `PRIOR`). The blind-model SYNTHESIS proves the deployed default is the **strict-barrier,
N=1** driver, whose measured operating point is **~54 rows/forward at 1 thread** (adapter.md §5), and
that the N-axis (overcommit) is *opt-in*, not the production default. So one strong, pre-existing
hypothesis is: *the model is not wrong about the cycle; it is evaluated at an operating point the
implementation does not occupy.* That is a **form-side** finding about an input's value — and the
loop must be able to *re-derive and quantify it from measurement*, not assume it (the prior work
is a reference to interrogate, §5; it is not this loop's conclusion handed in).

---

## 2. The one idea the design rests on: the benchmark is the shared term

The tool already contains the structural lever this whole loop needs, and the MANUAL names it (§2.1):
**a model is a DSL program, and each primitive's operational semantics already exists as its
benchmark.** `bench_r_gen` *is* "run the generator"; `bench_t_row` *is* "pay the per-row serve cost."
A model is therefore both *evaluable* (substitute benched means into `f` → the bound) and
*executable* (lower it to a runnable cycle built from the benched stages → a **witness**, the real
end-to-end DPS).

The design adds one observation on top of that, and it is the hinge of the attribution method:

> **A benchmark is the one object that is shared between the model and the implementation.** The
> model reads a benchmark's *mean* into `f`. The implementation *contains the very same stage* the
> benchmark claims to measure (the generator core; the per-row serve cost; the server drain). So a
> benchmark sits at a three-way junction:
>
> - the **model** consumes its mean,
> - the **implementation** realizes the stage it purports to measure,
> - and the **benchmark itself** is a *third*, independent claim about that stage's cost.
>
> When the model and the implementation disagree on the top-line number, the disagreement must
> resolve into exactly one of two failure shapes, because the benchmark gives us a third reading to
> triangulate against (§7 is the full method):
>
> 1. **Functional-form fault** — the benchmark faithfully measures its stage *and* the
>    implementation's stage matches it, but the model's `f` still mispredicts the whole. The error is
>    in *how the stages compose* (a missing coupling, an omitted term, the wrong operating point for
>    an input): `f` is wrong even though every primitive it reads is right.
> 2. **Benchmark-fidelity fault** — the benchmark *disagrees with the implementation's own stage*. It
>    measures something other than what the running system pays (a mock that is too cheap, a microbench
>    that omits a real cost, a value biased for no good reason — the MANUAL §4.4 explicitly warns
>    "nothing a-priori prevents a benchmark from returning a value biased a thousand-fold"). Here the
>    model may be a perfectly good function of *wrong inputs*.

Everything below is the machinery for getting those three readings, comparing them at the **same
stage boundary**, and using the model's own gradient/sensitivity decomposition (Purpose 2) to say
which stage's disagreement actually *matters* for the top-line gap.

---

## 3. The implementation, concretely (what we are explaining)

A diagnostic loop that "works from the implementation toward the model" must first know what the
implementation *is* and where it can be observed. The following is the map; *(agent-sourced)* marks
what rests on the commissioned exploration rather than my own file read, and Step 0 (§6) re-grounds
it.

**Two production paths share one search core.** The running actor is a persistent C++ Gumbel-AZ
process *(agent-sourced: `cpp/src/serve.cpp`, `cpp/src/main.cpp`)* driving episode generation, with
two dispatch paths for leaf evaluation:

- **Serial path** — a *local, in-process* C++ MLP forward (`NetForward`), no wire. The sole-workload
  ceiling of this path *is* what `bench_r_gen` measures: ~152 dps/core at the gen-ceiling config
  (`sims256/m24`), the `R_gen` grounding (confirmed first-hand in `bench_r_gen.py` and
  `grounding.py: GEN_PER_CORE_DPS`).
- **Wire-batched path** — the C++ producer parks leaves, a drain gathers them into one ZMQ multipart
  frame, and a single-threaded Python `inference_server.py` decodes, coalesces across threads, runs
  one JAX forward, and scatters replies. This is the path the model's **serve cycle** abstracts:
  `cycle_us = T_disp + tau_io + B·t_row`, `dps = min(N_gen·R_gen, 1e6·B/(cycle·L))`
  (`model_cycletime.py`, read first-hand). The blind-model SYNTHESIS (read first-hand) is a
  faithful, Z3-checked model of *this* boundary's control flow.

**The stage boundaries the model already names** (so the implementation must be observed at these,
to be comparable — §5.2 of the MANUAL's `cycle_breakdown` / `stage_capacities` are the surface):

| Model stage / term | What it denotes | Implementation locus | Benchmark today |
| --- | --- | --- | --- |
| `GENERATION` = `N_gen·R_gen` | producer cores' aggregate search rate | the C++ search core, serial path | `bench_r_gen` (C++ subprocess, eval mocked by `DetNet`) |
| `R_gen` | one core's decisions/s | one producer core in isolation | `bench_r_gen` |
| `L` (LPD) | leaves per recorded decision | the Gumbel tree's distinct-node count per decision | `bench_lpd` (PRIOR/design-pin today) |
| serve cycle: `T_disp` | pjit/XLA dispatch floor | the JAX forward dispatch in `inference_server.py` | `bench_t_disp` (fit) |
| serve cycle: `t_row` | per-row forward slope | `run_microbatch` staged forward | `bench_t_row` (fit) |
| serve cycle: `B` (`B_op`) | rows/forward at the operating point | the server's actual drained batch width | `bench_b_op` (PRIOR) |
| serve cycle: `tau_io` | server drain/decode/encode/scatter, serial between forwards | `inference_server.py` `_drain`/`_scatter` | `bench_tau_io` (UNMEASURED, top target) |
| `TRANSPORT` = `1/(L·tmsg)` | per-leaf wire framing capacity | the `inference_wire` codec | `bench_tmsg` (non-binding) |

**Where throughput is observed today** (the granularity the loop can start from):

- **End-to-end, top line:** episodes × decisions / wall-time, inferred from the gen-loop's
  `[gen Xsec iter Ysec]` logging *(agent-sourced; not a per-second counter)*. This is the ~200
  number's family. It is the *least* decomposable observation — the top line only.
- **Sub-end-to-end, already present:** `inference_server.py` has an opt-in `CHOCO_EVENTLOG` that
  timestamps `FWD` (forward) and `DRAIN` (batch-drain) events *(agent-sourced; the blind-model
  SYNTHESIS confirms the `_drain`/`serve_forever`/`run_microbatch` structure first-hand)*; the wire
  driver emits `mean_rows_per_msg = total_leaves/total_msgs` (the blind-model SYNTHESIS §6 quotes
  `runner_wire_batched.cpp:449-450,494-500` first-hand). **These two telemetry points are gold** —
  `mean_rows_per_msg` is *exactly* a live measurement of the model's `B`, and the `FWD`
  inter-arrival is *exactly* a live measurement of the serve `cycle_us`.
- **Producer, isolatable:** `search_runtime_bench` already isolates the search core sole-workload
  (the `R_gen` bench). The producer stage can be measured cleanly today.

The decisive fact for the design: **the implementation can be instrumented at the same stage
boundaries the model names, and two of those instruments already exist.** That is what makes a
stage-level discrepancy metric (§5) achievable cheaply, rather than a from-scratch instrumentation
project.

---

## 4. The maintainer's specified components (the frame, stated faithfully)

The commission specified the loop's skeleton. Stated in the maintainer's terms, so the design can be
checked against it:

- **C1.** The modellers (LLM agents) first need a *good understanding of the implementation*.
- **C2.** They take the *last model* they have — or, from zero, a *"reasonable" model* like
  `min(producer_throughput, consumer_throughput)` — and *measure it against the implementation*.
- **C3.** If model and implementation *disagree too much*, they *identify the most likely cause*.
- **C4.** The causes are *not limited to denotational questions* (the model's `f`). They must
  *equally* interrogate *how the benchmarks correspond to ground truth* — both the model's form and
  the benchmarks' fidelity are suspects, every round.
- **G (gate).** LLM-driven but *human-step-gated*: the maintainer inspects every step to confirm it
  is moving *constructively, not randomly*.
- **T (token economy).** Not boil-the-ocean; cheap, decisive steps early.

These are right, and the design honors them: C2's `min(producer, consumer)` is *literally*
`model_capacity.py`'s `min(GENERATION, SERVE, TRANSPORT)` and `model_cycletime.py`'s
`min(producer, serve)` — the "reasonable model from zero" already exists and is the natural seed. C4
is the crux and §7 gives it the most care. Two refinements I judge necessary follow in §5.

---

## 5. Interrogating the frame — two corrections and one missing name

Neutrality was required, so this section is where I push back. Three points; the first is the load-
bearing one.

### 5.1 The frame has no name for the bridge — the *witness* (a gap, named)

C2 says "measure the model against the implementation." But the model is a *denotational* object
(a number `f` denotes) and the implementation is an *operational* object (a number a running system
produces). **You cannot directly compare them** — that is precisely the category error the MANUAL
§1/§8 forbids (*"a model-bound is a conjecture, not a refutation"*). Comparing `f(μ̂) = 430` to
`measured = 200` and concluding "the model is 2× too high" is the silent failure the whole tool was
built to prevent: it treats a lower bound on the optimum *over designs* as a prediction of *one
design's* throughput.

The tool already supplies the missing middle term, and the frame should adopt its name: the
**witness** (MANUAL §2.1). A witness is the model *lowered to a runnable cycle built from the
benched stages, and clocked.* It is the object that is legitimately comparable to both: it is
operational like the implementation (a real DPS), but it is built *exactly* from the model's stages,
so a gap between the witness and the implementation is attributable *stage by stage*, and a gap
between the witness and `f(μ̂)` is the **coordination loss the model deliberately omits** (RTT idle,
convoy, cold-JIT — MANUAL §1, the losses `f` excludes by assumption).

So the loop's comparison is **not** model-vs-implementation. It is a **three-point comparison**:

```
   f(μ̂)            the bound        — denotational, the optimistic ceiling (omits coordination loss)
     │
     │  gap_A = coordination loss the model omits  (a LEGITIMATE gap; it is what `f` excludes)
     ▼
   witness DPS     a runnable cycle  — operational, built from the SAME benched stages as `f`
     │
     │  gap_B = the implementation's own slack vs a clean composition of its stages
     ▼
   implementation  the deployed cycle — operational, the ~200 number, runs a real (maybe suboptimal) config
```

This decomposition is what makes the loop honest. `gap_A` and `gap_B` are *different kinds of gap*
and want different responses. `gap_A` is the model's omitted-coupling budget — closing it means
*adding a term to `f`* (a form change) or *accepting it as irreducible coordination cost*. `gap_B` is
the implementation's headroom — closing it is an *engineering* change to the running system (e.g.
turn on overcommit), not a model change. **Collapsing them into one "the model is wrong by 230 dps"
is the comfortable-wrong-answer failure mode (§8).** The maintainer's C2/C3 implicitly fold these
together; the design un-folds them.

A caveat the MANUAL itself flags (§2.1, last paragraph): the tool does **not yet** lower a model to
an end-to-end witness — `untrusted_drive` measures every primitive live and *evaluates the bound*,
but does not *compose* the primitives into a running cycle. **Building the witness is the single
biggest piece of net-new work this loop needs**, and it is the honest first decision for the
maintainer (§11). Until it exists, the loop can still run a *degraded* form (§6, Steps 0-4) that
compares benchmark readings against implementation-stage readings directly — that is enough to find
*benchmark-fidelity* faults, which §7 argues are the cheaper-and-more-likely class to clear first.

### 5.2 C2's "reasonable model from zero" is already in-tree — start from it, do not re-invent

`min(producer_throughput, consumer_throughput)` is not a fresh model to write; it is
`model_capacity.py` / `model_cycletime.py`, which the tool ships, cross-checks against each other to
9 dps (MANUAL §5.1), and decomposes via `cycle_breakdown`/`stage_capacities`. The loop should seed
from these and inherit their sensitivity machinery for free. "From zero" is the wrong default when a
cross-checked, gradient-instrumented model already exists; re-deriving it forfeits the cross-check
and the Purpose-2 decomposition that §7's attribution *depends on*.

### 5.3 "Disagree too much" needs a *threshold owner*, and the threshold is not on the top line

C3's "too much" must be quantified, and — given §5.1 — it must be quantified **per gap, at the stage
level**, not on the end-to-end top line. A 230-dps top-line gap is *expected* and *uninformative*
(it is mostly `gap_A`, the omitted coordination loss). The informative threshold is on `gap_B`'s
*per-stage* residuals: *does the implementation's measured stage cost agree with its benchmark's
claimed cost, within the benchmark's own CI?* That is a question the `Estimate`'s `cov` already
answers (§5, §7). The threshold owner is therefore the per-stage `Estimate` CI, not a hand-picked
"too much" on the top line.

---

## 6. The steps and their order (cheap-and-decisive first)

The loop is a sequence of gated steps. The ordering principle: **spend the first (cheap) steps
ruling out the benchmark-fidelity faults that would invalidate everything downstream, before
spending the expensive steps on functional form.** A form analysis built on a lying benchmark is
worthless (§8), so fidelity is cleared first. Tokens go where a decisive cut is cheapest.

Each step is one human-gated checkpoint (§8 details the gate). "Cheap" / "Expensive" is the token +
wall-clock cost of the *evidence-gathering* the step requires.

### Step 0 — Ground the implementation map (CHEAP; agent + human, no measurement)

*Re-derive §3 against the live code.* An agent reads the C++/Python sources I did not
(`serve.cpp`, `runner_wire_batched.cpp`, `inference_server.py`, the gen loop) and produces a
**stage-to-locus table**: for each model term, the exact file/function that realizes it, and the
exact observation point (a counter, an event log, a timer) that could measure it. **Gate:** the
maintainer confirms each model stage maps to a real implementation locus and a real observation
point — *or* flags a stage the implementation does not have (which is itself a form finding: the
model has a term the system does not realize). Cheap because it is reading, not running. Decisive
because it converts every later "the benchmark should match the implementation" into a *concrete,
checkable* pair of numbers. **This step also discharges my §0 gap** — every *(agent-sourced)* claim
above is verified or corrected here before anything is acted on.

### Step 1 — Pin the top line honestly (CHEAP-MEDIUM; measurement, sole-workload)

Measure the implementation's end-to-end DPS *as a real number with provenance and a CI*, under a
named, frozen config (which path? serial or wire? strict-barrier or overcommit-N? how many cores?
which instance?). **This replaces the contested ~200/~203** — that number is a *one-config user
reference*, never to be treated as the implementation's throughput in general (the
claims-measured-vs-interpreted discipline; the ~203 is the cautionary instance — a one-config number
the project explicitly refuses to anchor on). The output is `D_impl ± CI` for *this* config, logged
like any bench (verify-the-artifact: a measured number, never a claimed one). **Gate:** the
maintainer confirms the config is the one we mean to explain (the deployed default, per the
blind-model's open question #1), and that `D_impl` is measured, not asserted.

### Step 2 — Measure each stage on the implementation, at the benchmark's boundary (MEDIUM; the key data-gathering step)

For each stage the model names, obtain a **live reading from the running implementation**, at the
*same boundary the benchmark measures*. Two of these are nearly free (existing telemetry):

- `B_impl` (rows/forward) ← `mean_rows_per_msg` from the wire driver (already emitted).
- `cycle_impl` (per-forward serve cycle) ← `FWD` inter-arrival from `CHOCO_EVENTLOG` (already
  emitted); `tau_io_impl` ← (`DRAIN`→`FWD` start) minus the dispatch floor.
- `R_gen_impl` ← the producer's live rate under the *deployed* (not mocked) eval, from the same
  search core (a variant of `search_runtime_bench` without the `DetNet` mock).
- `L_impl` (leaves/decision) ← a per-decision distinct-node histogram from one instrumented run (the
  grounding flags `LPD` as needing exactly this; it is a `PRIOR`/design-pin today).

The output is, per stage, a pair: `(bench_reading, impl_reading)` — both `Estimate`s, both with CIs.
**Gate:** the maintainer confirms each `impl_reading` is observed at the boundary that makes it
comparable to its benchmark (not a different cut of the same stage — the §7 trap), and that the two
readings are on the same units. This is where the loop's tokens concentrate, and §7 is what they buy.

### Step 3 — Discriminate form vs fidelity, per stage (MEDIUM tokens; the analysis step — §7 is the method)

For each stage, run the §7 discriminator on `(bench_reading, impl_reading)` plus the model's
sensitivity `a_i`. Classify each stage's contribution to the gap as **fidelity-suspect** (bench ≠
impl), **form-suspect** (bench ≈ impl but the stage is mis-composed / mis-valued in `f`), or
**settled** (bench ≈ impl and the stage is non-binding / low-sensitivity). **Gate:** the maintainer
reviews the per-stage classification and the *single* most-likely cause the agents nominate, with its
evidence, before any model or bench edit is proposed. This is C3/C4 made mechanical.

### Step 4 — Propose the minimal change, and re-measure (MEDIUM-EXPENSIVE; close one loop iteration)

Exactly one change is proposed, of exactly one kind, addressing the nominated cause:

- **fidelity fix** → repair the benchmark so it measures what the implementation pays (then its
  `Estimate` flips trusted, no model edit — the trust ladder, MANUAL §6); **or**
- **form fix** → add the missing term / fix the operating point in `f` (a new model module or a term,
  per the MANUAL §4 workflow); **or**
- **engineering finding** → the gap is `gap_B` (implementation slack), recorded as *"the model is
  right; the implementation runs a suboptimal config X"* — **no model or bench change**, a finding
  handed back to the running system (e.g. the overcommit lever the adapter doc §6 already scoped); **or**
- **instrument lift** (§7a) → the round's verdict is *tool-inadequate*: the proposed change is to the
  apparatus (a richer `f`-shape, a regime-aware estimator, a bench the grounding names, an
  implementation counter), *scoped here and built as its own gated work* — not a form/fidelity edit
  forced through an instrument that cannot honestly carry it.

Then re-measure the affected stage (Step 2) and the top line (Step 1) and confirm the change moved
the discrepancy in the predicted direction and magnitude (ADR-0009: a perf/equivalence claim is
honest only with captured, reproducible substantiation). **Gate:** the maintainer approves the one
change, *predicts* its effect before it is applied, and confirms the re-measurement matched the
prediction (a pre-registered prediction, so the loop can be *wrong loudly* — the adapter doc §4 is
the house instance of this discipline). A change that does not move the gap as predicted is a finding
about the *model of the cause*, not a success — it re-enters at Step 3.

### The witness (the operational refutation — built once, then a fixture)

Independently of the per-round loop, the **witness** (§5.1) is built once: lower the seed model to a
runnable cycle composing the benched stages, and clock it. Once it exists, Step 1 gains a third point
(`witness DPS`) and the §5.1 three-point decomposition becomes available every round — `gap_A`
(witness vs `f(μ̂)`) and `gap_B` (impl vs witness) separate cleanly, which is what lets Step 3 tell a
*form* fault (lives in `gap_A`) from an *engineering* finding (lives in `gap_B`). Building it is the
biggest one-time cost and the §11 open question; the loop runs degraded without it (Steps 0-4 still
find fidelity faults), so the witness is a *fast-follow*, not a blocker.

---

## 7. The attribution method — separating form from fidelity (the crux)

This is the section the commission asked to be given the most care. The instrument is the model's
**gradient/sensitivity decomposition** (Purpose 2 — `Recommendation.primitives`, each carrying
`grad`, `sigma`, `a = (∂f/∂x)²·σ²`, and `var_contribution`) **crossed with** the per-stage
implementation instrumentation from Step 2. The two together discriminate the two fault classes; the
gradient alone cannot, and the instrumentation alone cannot.

### 7.1 The discriminator, stated as a decision

For a model input `x_i` whose stage the implementation realizes, we now hold *three* numbers:

- `bench_i` — the benchmark's `Estimate` for `x_i` (mean + CI).
- `impl_i` — the implementation's live reading of the same stage (mean + CI), from Step 2.
- `a_i` — the model's sensitivity: how much `x_i` moves the bound (`Recommendation`, Purpose 2).

The decision, per input:

```
  if  bench_i  disagrees with  impl_i  (CIs do not overlap, at the stage boundary):
        → FIDELITY FAULT at stage i.
          The benchmark is measuring something other than what the running system pays.
          Severity = a_i: a fidelity fault on a HIGH-a_i (binding) stage is decisive;
          on a low-a_i (non-binding) stage it is real but does not explain the top-line gap.

  elif bench_i  AGREES with  impl_i  (the benchmark is faithful to the live stage)
       AND the stage is a binding / high-a_i term
       AND the top-line gap persists after substituting impl_i for bench_i in f:
        → FORM FAULT.
          Every primitive f reads at this stage is right, yet f still mispredicts.
          The error is in how f composes the stages (a missing coupling, an omitted
          interacting term) OR in the OPERATING POINT f evaluates an input at
          (the B_op=256 vs ~54 rows/forward instance — a form fault about a value).

  else:
        → SETTLED at stage i (faithful bench, low sensitivity, or non-binding).
```

The load-bearing move is the **third number**. The maintainer's C4 says "both `f` and the benchmarks
are suspects" — true, but *suspicion alone does not discriminate*. What discriminates is that the
benchmark and the implementation are **two independent readings of the same physical stage**, so
their *agreement* is a test of fidelity that is logically prior to, and separable from, the test of
form. You cannot indict `f` for a stage until that stage's benchmark is exonerated against the live
system; and once it is exonerated, a persistent gap *is* a form indictment. The gradient `a_i` then
says *which* stage's verdict matters for the top line.

### 7.2 Why this is decisive and not circular

The trap the discriminator avoids is **tuning the model to match a biased benchmark** (§8). If the
loop only ever compared `f`-evaluated-on-benchmarks against the implementation, a benchmark biased by
1000× (the MANUAL §4.4 warning) would be *indistinguishable* from a form error: both show up as
`f(μ̂) ≠ D_impl`, and the agents would happily "fix the form" to absorb the benchmark's lie. The
implementation reading `impl_i` breaks the circularity: it is a reading of the *same stage* that does
**not** pass through `f` or through the benchmark. A benchmark cannot be both wrong about a stage and
agree with an independent reading of that stage. So:

- **fidelity is falsifiable** — `bench_i` vs `impl_i` is a direct, model-free comparison; if they
  disagree, the benchmark is indicted with no appeal to `f`.
- **form is falsifiable only after fidelity is cleared** — and the discriminator enforces that order
  (the `elif` requires `bench_i ≈ impl_i` first). A form fault is *only* declared on a stage whose
  benchmark already matched the live system, so "fixing the form" can never be a disguised absorption
  of a benchmark lie.

### 7.3 The gradient's specific job (and what it does *not* do)

The sensitivity decomposition does three things in the discriminator, and it is worth being precise
that it does *not* do a fourth:

1. **It ranks which stage's gap matters.** A fidelity or form fault on a `var_contribution`-dominant
   stage explains the top line; one on a non-binding stage (e.g. `tmsg`, which the models show ranks
   last) does not. `a_i` is the relevance weight on every verdict.
2. **It exposes the binding stage and its margin** via the `min()`-kink machinery (`alloc/kink.py`).
   If the implementation's binding stage *differs* from the model's binding stage, that is itself a
   first-order finding — and the kink's `p_nonbinding_max` says whether the model's binding stage is
   even *certain* (a near-tie means *two* stages must be reconciled before the bottleneck is trusted,
   MANUAL §5.2). A common form fault is exactly "the model thinks SERVE binds; the implementation is
   GENERATION-bound (or transport-bound) at its real operating point."
3. **It tells fidelity-fix from form-fix economics.** `Recommendation.recommend` (the Neyman
   `+samples`) says whether a stage's *uncertainty* is even reducible by more measurement (a median
   bench) or is a floored prior/fit that only re-engineering moves — so the loop knows whether
   "measure it harder" or "change the model" is the lever (MANUAL §5.2, the `var_floor` /
   `floor_blocks_target` lines).

What it does **not** do: the gradient is a *local linearization at μ̂*. It cannot, by itself, find a
**missing interacting term** — a coupling the separable `f` cannot see (the MANUAL §2 caution: "a
clean separable `f` … quietly asserts that each stage is an independent quantity"). Detecting a
*missing* coupling is the residual that survives after every *present* stage is reconciled: if every
input's `bench_i ≈ impl_i` *and* substituting all `impl_i` into `f` still leaves a gap, the
remainder is a **structural** form fault — `f` omits a term or a coupling that no input it currently
reads can express. That residual is the signal to *add structure* to `f`, and it is the one place the
loop must reason qualitatively about the *physics* of the cycle, not just the numbers (the witness,
§5.1, is what bounds this residual: `f(μ̂)` minus the witness DPS *is* the omitted-coordination
budget, so the structural residual cannot exceed it).

### 7.4 The operating-point fault is a form fault about a value (the worked instance)

The `B_op` case (§1, §5.3) is the worked instance of a form fault that is *not* a missing term but a
**wrong operating point** — and it shows why the discriminator must read `impl_i`, not just
`bench_i`. The benchmark `bench_b_op` could be *perfectly faithful* about "the serve cost at 256
rows" — yet the model is still wrong about the implementation, because the implementation does not
*run* at 256 rows; it runs at ~54 (strict-barrier, N=1). The discriminator catches this precisely
because `B_impl` (from `mean_rows_per_msg`) ≠ the model's `B_op = 256`, even when the *cost function*
`t_row` is faithful. The fix is a form fix (evaluate `f` at the realized `B`, or model the operating
point as an input the producer config sets) — *or* it is reclassified as `gap_B`, an engineering
finding (*the model is right about the optimum; production runs an under-batched config*), which is
the truthful reading the blind-model SYNTHESIS already supports. The discriminator surfaces the
choice; the maintainer's gate (Step 4) decides which it is, because that choice is a judgment about
*what we are trying to explain* — the optimum, or this config.

---

## 7a. The third suspect — when the harness/DSL itself is inadequate

A diagnostic instrument can be the thing that is wrong. The loop's two suspects so far (the model's
form, the benchmarks' fidelity) both presuppose that the **tool** — the DSL `f` is written in, the
`Estimate`/`ShrinkLaw` contract, the benches' resolution, the implementation's observability — is
*capable of representing* the distinction the round needs to draw. When it is not, forcing a verdict
through an inadequate instrument is itself a silent failure (it is ADR-0008's positive register
applied to the diagnostic apparatus: *refuse a fuzzy match against an inadequate vocabulary; revise
the vocabulary instead of picking the closest fit*). So the loop carries a **third verdict** any
round may return, beside form-fault and fidelity-fault:

> **TOOL-INADEQUATE at stage i (or for the gap as a whole)** — the harness/DSL cannot resolve the
> question at the granularity the gap demands. The honest move is to *name the obstruction and propose
> how to lift it*, not to emit a form/fidelity verdict the instrument cannot actually support.

Concrete shapes this takes, each tied to a real limit of the current tool (so the flag is checkable,
not a shrug):

- **The DSL cannot express the coupling.** `throughput_jax` is a separable composition (`min`,
  products, sums over per-stage primitives), and the MANUAL §2 names the hazard outright: "a clean
  separable `f` … quietly asserts that each stage is an independent quantity." If the surviving
  structural residual (§7.3) implies an *interaction* the separable form cannot carry — a stage cost
  that depends on another stage's state, a feedback the blind-model SYNTHESIS shows is real (the
  negative-feedback batch-size fixed point couples `B` to service time `S`, which `B·t_row` with a
  *constant* `t_row` cannot express) — then no edit *within the current DSL shape* is faithful. The
  flag: *"the gap needs a coupling the additive cycle cannot represent; lift = a richer `f`-shape (a
  term that reads two stages), or an explicit state-dependent input."*
- **The `Estimate` contract cannot carry the uncertainty's true shape.** The five `ShrinkLaw`
  variants (`Poolwise`/`QuantileLaw`/`RegressionLaw`/`Fixed`/`Composed`) are the vocabulary of *how a
  variance responds to effort*. If a stage's real behavior is none of these (e.g. a bimodal service
  time — the cold-vs-warm JIT the SYNTHESIS §3.2 names, where a single mean/median is a category
  error), the contract will *accept a faithful-looking but wrong* `Estimate`. The flag: *"this stage's
  cost is not unimodal; a mean/median Estimate misrepresents it; lift = a mixture/regime-aware
  estimator, or split the stage into its regimes."*
- **The benchmark cannot reach the binding stage at all.** `tau_io` is UNMEASURED and is the top
  Neyman target precisely because *no bench measures it yet* (it is the serial drain/scatter that sits
  in no microbench — grounding.py, the `SERVE_IO_US` note). If the gap localizes to a stage that has
  no runnable bench and no clean implementation observation point, the loop is *blind* there. The
  flag: *"the binding stage has no instrument; lift = build the bench the grounding already names (the
  serve-loop microbench), or add the implementation counter."*
- **The implementation is not observable at the boundary the model names.** Step 0/Step 2 can find
  that a stage the model treats as a clean term is, in the running system, *entangled* with another
  (e.g. the producer rate is not separable from redis-write stalls — the SYNTHESIS open question #6
  names exactly this unmodelled source-timing input). Then `impl_i` cannot be read at the boundary
  that makes it comparable to `bench_i`, and the §7 discriminator *cannot run* for that stage. The
  flag: *"the implementation has no observation point at this boundary; lift = add the counter/event,
  or accept the stage as unresolvable and say so (ADR-0002 — a named limitation, not a papered gap)."*
- **The discrepancy itself is below the instrument's floor.** If the gap the round chases is smaller
  than the combined CI of the readings (the `Estimate.cov` floor, or the confounded-substrate noise
  of MANUAL §6 rung 4), the loop is chasing noise. The flag: *"the residual is within measurement
  noise; lift = tighten the dominant CI first (the Neyman `+samples` says whether that is even
  possible — §5.2), or stop, because the gap is not resolvable at this resolution."*

**Why this is a distinct verdict and not a form/fidelity sub-case.** A form fault says *fix `f`*; a
fidelity fault says *fix the bench*; both presume the *apparatus can express the fix*. The
tool-inadequate verdict says *the apparatus cannot represent the answer the evidence points at* — the
fix is to the **instrument**, one meta-level up. Collapsing it into "form fault" would be the very
failure it guards against: editing `f` to *appear* to close a gap the DSL cannot honestly model
(F1/F3, one level up). Surfacing it keeps the loop from manufacturing false precision — and, since
lifting an obstruction is often the highest-value finding (it unblocks every future round, not just
this one), it is frequently the *most* constructive thing a round can produce, not a dead end. This
also composes with the project's own roadmap: a tool-inadequacy that recurs is an ADR-0011 Rule-2
trigger (a describing record → a mechanism), and the named lift is the candidate mechanism.

**The economics (token discipline).** This verdict is *cheap to raise* (it is a recognition, during
Step 3, that the readings do not let the discriminator fire) and its *lift* is scoped, not executed,
in the same round — the agent proposes the instrument change; building it is its own gated work
(Step 4-class, expensive, deferred), exactly like the witness. So flagging inadequacy costs a
sentence; acting on it is a separate, gated decision. The loop must never *silently* absorb an
inadequacy by forcing a verdict — that is the cost it is designed to avoid.

## 8. The human-gate structure

The gate is the part the commission asked to be designed, not just asserted. The maintainer inspects
every step to confirm it is moving *constructively, not randomly*. Each gate is a **checkpoint
artifact** the maintainer reads, and a **binary decision** that either proceeds the loop or sends it
back. The principle: every gate surfaces the *evidence and the prediction*, never just the
conclusion (the project's verify-the-artifact discipline — the maintainer approves a checkable
artifact, never a claim).

| Step | Checkpoint artifact surfaced to the maintainer | The maintainer's decision |
| --- | --- | --- |
| 0 | the stage→locus→observation table (each model term ↔ its real code locus ↔ how to observe it) | every stage maps to a real locus + observation point? (or a stage has no locus = a form finding) → proceed / flag |
| 1 | `D_impl ± CI` under a named frozen config, with the raw measurement provenance | is this the config we mean to explain, and is the number measured (not asserted)? → proceed / re-config |
| 2 | per stage, the pair `(bench_i, impl_i)` as two `Estimate`s with CIs + the boundary each was read at | are the two readings at the *same* boundary, same units, both measured? → proceed / re-measure |
| 3 | the per-stage form/fidelity/**tool-inadequate**/settled classification (§7/§7a) + the *single* nominated most-likely cause + its evidence | is the classification supported, and the nominated cause the most likely (not the most convenient)? is any verdict actually *unsupportable by the instrument* (§7a)? → proceed / re-analyze |
| 4 | the *one* proposed change, of *one* declared kind (fidelity fix / form fix / engineering finding / **instrument lift**), + the *pre-registered prediction* of its effect | approve the change and its predicted magnitude *before* it is applied; then confirm re-measurement matched → ratify / reject |

Two structural gate rules, both anti-self-deception:

- **One change per iteration, of one declared kind.** A round proposes exactly one edit (a fidelity
  fix, *or* a form fix, *or* an engineering finding), so its effect is attributable. Bundling changes
  re-introduces the confounding the loop exists to remove (and mirrors the project's
  phase-checkpoint discipline: one coherent change, persisted, before the next).
- **Pre-registered prediction at Step 4.** The maintainer records the *expected* direction and
  magnitude of the gap-change before the change is applied. A change that lands outside its predicted
  band is a finding about the *model of the cause*, surfaced loudly (ADR-0002), not quietly absorbed
  — this is the single most important guardrail against the loop converging on a comfortable wrong
  answer (§10, F3), because it forces the loop to be *wrong out loud* exactly where it would prefer to
  be quietly right.

**On the LLM agents' role at each gate.** The agents *gather and propose*; the maintainer *decides*.
The agents produce the artifacts (the readings, the classification, the nominated cause, the
prediction); they never apply a change or advance a gate. This keeps the human in the loop *at the
decision*, not merely informed after it — which is what "confirm it is moving constructively, not
randomly" requires.

---

## 9. The token economy of one loop iteration

Where the tokens go, and where they do not:

- **Cheap, up front, decisive (most of the value):** Step 0 (read the loci — a one-time agent read,
  amortized across all later rounds) and Step 3's discriminator (a small, mechanical analysis over
  numbers the tool already computes — the `Recommendation` is *one* `driver.step()`). These two
  carry the loop. The discriminator is essentially free per round: the sensitivity decomposition is a
  single gradient evaluation the tool produces anyway.
- **Medium, the data-gathering body:** Step 2 (the per-stage live readings). Two of the stages are
  *already instrumented* (`mean_rows_per_msg`, `CHOCO_EVENTLOG`), so their cost is "run the system
  and parse a log," not "write instrumentation." The others (`R_gen` un-mocked, `L` histogram) are
  bounded sole-workload runs. This is where wall-clock, not tokens, dominates.
- **Expensive, deferred, gated:** Step 4's re-measurement (a full re-run) and, *once*, the witness
  build (§5.1 — the one genuinely large piece of net-new code). The witness is built **once** and
  becomes a fixture; it is not a per-round cost.
- **Where tokens do *not* go:** re-deriving the seed model (it exists — §5.2); modelling the
  transport faithfully to "regenerate ~200" (the `transport_sweep.py` docstring's own warning:
  *"Modelling ZMQ faithfully just regenerates ~200 dps (a coordination artifact of one config)"* — a
  faithful model of the *current* config is not the goal, and the blind-model corpus already did it);
  exhaustively instrumenting every stage when the gradient says only the binding one matters (Step 3
  prunes to the high-`a_i` stages first — a fidelity check on `tmsg` is wasted tokens, because
  `tmsg`'s `a_i` is ~0).

The economy in one line: **the cheap steps (read the loci, run the discriminator) are where the
decisive cuts are, and they are front-loaded; the expensive steps (re-run, build the witness) are
gated and amortized.** A round that does not need a fresh expensive measurement (a fidelity fix
confirmed against existing telemetry) is *all cheap*.

---

## 10. Failure modes and their guardrails

The commission asked specifically how the loop could move randomly, converge on a comfortable wrong
answer, or fool itself. Each, with its guardrail:

**F1 — Tuning `f` to match a biased benchmark.** The named hazard (MANUAL §4.4). If the loop
compared only `f(benchmarks)` vs the implementation, a biased benchmark and a form error are
indistinguishable, and the agents would "fix the form" to absorb the bias. *Guardrail:* the §7
discriminator requires `bench_i ≈ impl_i` (a model-free check) *before* any form fault is declared.
A benchmark is exonerated against the live stage before `f` is ever indicted — so a form fix cannot
be a disguised benchmark-lie absorption.

**F2 — Matching a confounded implementation measurement.** The dual hazard:
`untrusted_drive`'s own caveat (the Python/GIL substrate confounds the cross-thread benches — MANUAL
§6 rung 4). If `impl_i` is itself measured through a confounded path, the discriminator's "agreement"
is meaningless. *Guardrail:* Step 2's gate requires each `impl_i` to be read at a clean boundary
(sole-workload where the stage is timing-sensitive; the existing `FWD`/`mean_rows_per_msg` telemetry
is *in the running system*, not a microbench, so it is the *least* confounded reading available — a
reason to prefer it). And the trust ladder (MANUAL §6) is the standing honesty register: a reading's
rung is stated, never assumed.

**F3 — Converging on a comfortable wrong answer (the top-line collapse).** The loop "explains" the
230-dps gap by inflating one model term until `f(μ̂)` drops to 200 — the MANUAL §8 forbidden move
("Never widen a stage to 'explain' a slow harness — that hides the very thing Purpose 2 is for").
*Guardrails, three:* (a) the §5.1 three-point decomposition forbids comparing `f(μ̂)` to `D_impl`
directly — the legitimate comparison is per-gap, and `gap_A` (omitted coordination) is *expected*,
not a defect to absorb into a stage; (b) the bound is a *lower* bound on the *optimum*, so a term
inflated past its real cost makes `f` *no longer a lower bound* — a contradiction the maintainer's
Step-4 gate catches; (c) the pre-registered prediction (§8) means a term-inflation that "fixes" the
top line but mispredicts the *next* re-measurement is surfaced as a failure, not a success.

**F4 — Moving randomly (a different cause every round).** The agents nominate a fresh, unrelated
cause each iteration, never converging. *Guardrail:* the §7 discriminator is *deterministic given the
readings* — it does not "guess" a cause, it *classifies* each stage by a fixed rule, and the gradient
*ranks* which classification matters. The nominated cause is the highest-`a_i` indicted stage, not a
free choice. Plus Step 4's one-change-per-round + pre-registered-prediction makes randomness *visible*
(a random cause mispredicts its own re-measurement) rather than invisible.

**F5 — Explaining one config and calling it the answer.** The loop reconciles the model to the
~200 config and declares the gap "explained," when ~200 was only ever one config family (the
`REF_PLATEAU_DPS` cautionary instance — a one-config number the project refuses to treat as a
ceiling). *Guardrail:* Step 1's gate names the config explicitly and Step 4 distinguishes a *form*
finding (about the optimum) from an *engineering* finding (about this config). A gap that resolves to
`gap_B` is recorded as *"this config is suboptimal,"* which is **not** a statement that the optimum is
200 — it is the opposite. The claims-measured-vs-interpreted discipline applies in full: an empirical
plateau of one config is *measured*; "200 is the ceiling" is an *interpretation* that the loop must
never record as proven (the user has been burned by exactly this — a one-config plateau saved as a
proved ceiling).

**F6 — The witness that flatters itself.** If/when the witness is built, it could be tuned (better
batching, warmer JIT, a kinder config) until it clears the implementation, "proving" headroom that
the real system can't reach because the witness quietly assumes away a coordination cost. *Guardrail:*
the witness is built from the *benched stages composed in the order `f` specifies* (MANUAL §2.1 — "a
composition, not a rewrite"); any cost it omits relative to the implementation must show up as a named
`gap_B` stage in Step 2, not as an unexplained witness advantage. A witness that beats the
implementation by an amount no Step-2 stage accounts for is itself a finding (the witness omits a real
cost), not a refutation.

**F7 — Forcing a verdict through an inadequate instrument.** The loop emits a confident form- or
fidelity-fault when the harness/DSL cannot actually resolve the question — editing `f` to *appear* to
close a gap the separable DSL cannot model, or trusting a unimodal `Estimate` of a bimodal stage
(§7a). This is F1/F3 one meta-level up: false precision manufactured by an instrument out of its
depth. *Guardrail:* the **tool-inadequate** verdict (§7a) is a first-class outcome of Step 3, and
Step 3's gate explicitly asks whether any verdict is *unsupportable by the instrument*. The agents are
charged to flag inadequacy and propose the lift rather than force a verdict; raising the flag is cheap
(a sentence at Step 3) and is often the round's highest-value output (it unblocks every future round).
A loop that *never* raises this flag across many rounds on a hard gap is itself suspect — perfect
instrument-adequacy is not the null hypothesis on a problem this under-resolved.

---

## 11. What is uncertain / what I need the maintainer to decide

Honest accounting of where the design rests on a judgment that is the maintainer's, not mine. The
biggest is first.

1. **Build the witness, or run degraded?** The form-vs-fidelity attribution is *strongest* with the
   witness (it is what cleanly separates `gap_A` from `gap_B` — §5.1). But the witness does not exist
   yet (MANUAL §2.1, last paragraph), and building it (lowering a model to a runnable composed cycle)
   is the one large piece of net-new work. The loop *runs without it* (Steps 0-4 find fidelity faults
   and operating-point form faults from the three-point readings, which is most of the value), but it
   cannot cleanly separate "the model omits a coordination cost" from "the implementation is slack"
   until the witness pins the omitted-coordination budget. **Decision:** is the witness in scope as a
   fast-follow, or does the loop run degraded indefinitely? This is the single most consequential
   choice, and it is genuinely a cost/value judgment I cannot make for you.

2. **Which config are we explaining — the deployed default, or the optimum?** The blind-model's open
   question #1 is unresolved in the cleanroom: *does deployed production run the strict-barrier
   default (N=1, ~54 rows/forward) or an out-of-tree launcher that flips it to overcommit?* The
   entire weight of the `B_op` form-vs-engineering verdict (§7.4) hinges on this, and it is a fact
   about a file outside the modelled world. **Decision (cheap, and a prerequisite for Step 1):**
   confirm the deployed config — which path, which `WireMode`, which N. Without it, Step 1's gate
   cannot be passed honestly.

3. **The discrepancy threshold per stage.** §5.3 argues the "too much" threshold belongs on
   `gap_B`'s per-stage residuals (does `impl_i` fall in `bench_i`'s CI?), owned by the `Estimate` CI,
   not a hand-picked top-line number. But the *tolerance* on that residual (how much CI overlap
   counts as "agreement"; how to combine the two CIs) is a calibration choice. The tool's
   `is_valid()`/`cov` machinery supplies the CIs; the *decision rule* over them
   (overlap? Welch-style? a fixed `k·σ`?) is unspecified and is the maintainer's to set
   (measure-first, per ADR-0011: pick it against observed stage-reading spread, not a priori).

4. **The clean-boundary readings (F2).** Two stage readings are nearly free (existing telemetry);
   the others (`R_gen` un-mocked, an `L` histogram) need bounded instrumentation I have only
   *(agent-sourced)* evidence exists cheaply. **Decision/risk:** confirm in Step 0 that each
   high-`a_i` stage actually *can* be read at a clean boundary in the running system; if a binding
   stage cannot be observed without confounding, the loop's verdict on *that* stage is degraded and
   must say so (ADR-0002 — an unobservable binding stage is a named limitation, not a papered gap).

5. **Where the design is right, and why.** To close honestly: the maintainer's components C1-C4 and
   the gate are *correct* — C4's insistence that benchmarks are suspects *equally with* the form is
   the right instinct, and it is precisely what the §7 three-number discriminator operationalizes;
   C2's seed model already exists and is gradient-instrumented; the human-step-gate is the right
   shape for "constructive, not random." The two places I judged the frame incomplete are named, not
   smuggled: it lacked a name for the **witness** (the comparable middle term, §5.1) and a place to
   put the **engineering finding** (the `gap_B` reading where the model is right and the
   *implementation* is what's suboptimal, §7.4/F5). Both are corrections *toward* the tool's existing
   epistemics (the MANUAL already has the witness; the project already refuses to call ~200 a
   ceiling), not a different design imposed over them.

---

## Summary (for the commissioning record)

- The legitimate comparison is **not** model-vs-implementation (a category error the MANUAL forbids)
  but a **three-point** one: `f(μ̂)` (the optimistic bound) → the **witness** (the model lowered to a
  runnable cycle built from the benched stages) → the implementation; the model-vs-witness gap is the
  *omitted coordination loss* (a legitimate gap), the witness-vs-implementation gap is the
  *implementation's own slack* (an engineering finding), and conflating them is the central failure
  mode.
- **Form-vs-fidelity attribution in one sentence:** the benchmark is the one object *shared* between
  the model and the implementation, so for each stage we hold three numbers — the benchmark's reading,
  the *implementation's* live reading of the same stage, and the model's sensitivity `a_i` — and a
  **benchmark ≠ implementation** disagreement is a *fidelity* fault while a **benchmark ≈
  implementation but `f` still mispredicts** is a *form* fault, with `a_i` saying which stage's verdict
  matters for the top line.
- A **third suspect** sits beside form and fidelity: the **harness/DSL itself** (§7a). Any round may
  return *tool-inadequate* — the apparatus cannot resolve the question at the granularity the gap
  demands (the separable `f` cannot express a real coupling; the `Estimate` contract cannot carry a
  bimodal service time; a binding stage has no bench or no clean observation point) — and the honest
  move is to *name the obstruction and propose the lift*, not force a form/fidelity verdict the
  instrument cannot support. Flagging it is cheap and is often the round's highest-value output.
- The loop spends its cheap steps first (read the loci; run the gradient discriminator — both nearly
  free, the discriminator is one `driver.step()`) to clear fidelity faults *before* the expensive
  form work, because a form analysis built on a lying benchmark is worthless; and it gates every step
  on a checkable artifact + a pre-registered prediction so randomness and comfortable-wrong-answers
  are surfaced loudly.
- **The single biggest open question for the maintainer:** is the **witness** in scope (build it as a
  fast-follow — it is the one large piece of net-new work, and it is what cleanly separates "the model
  omits a coordination cost" from "the implementation is slack"), or does the loop run in its degraded
  form (still able to find fidelity faults and operating-point form faults, but not to cleanly
  partition the gap) indefinitely?
