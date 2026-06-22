<!--
tools/analysis/leaf_eval_bound/MANUAL.md — the author/operator manual for the leaf-eval
throughput lower-bound tool: how to model a new cycle, how to read the bound and the
sensitivities (the tool's two purposes), and the contract every model, bench, and grounded
constant must satisfy. Written so a collaborator NEVER has to read an existing bench or model
to write a new one — every contract and a copy-accurate skeleton of each shape is in here.

ADR-0005 authoring discipline; ADR-0006 header; ADR-0002 (fail loud) and ADR-0012 (the typed
signature is the SSOT) are the laws the contracts below rest on.

Public Domain (The Unlicense).
-->

# The leaf-eval throughput-bound tool — modeling manual

This is the instruction manual for `tools/analysis/leaf_eval_bound/` — a tool that computes a
**provable throughput lower bound** for a serving "cycle" (a control loop that generates,
serves, and transports leaf evaluations), and tells you **where the bound's uncertainty lives**.

You should be able to read this manual and then model a new cycle, write its benches, run it,
and interpret the result **without opening a single existing `bench_*.py` or `model_*.py`** — the
contracts and a worked skeleton of every shape are all here. Where a claim is a contract, it
cites the typed signature (ADR-0012: the signature is the single source of truth).

---

## Table of contents

1. [Why this tool exists — and its two purposes](#1-why-this-tool-exists)
2. [The one idea: a cycle is a min-of-stages, and the bound is the slowest stage](#2-the-one-idea)
   - [2.1 The model is a DSL; the benchmarks are its operational semantics — the path to a *witness*](#21-dsl-operational)
3. [The package map](#3-the-package-map)
4. [Modeling a new cycle — the five-step workflow](#4-modeling-a-new-cycle)
   - [4.1 Write `f` — the model](#41-write-f)
   - [4.2 Classify each input — measured, prior, or constant](#42-classify-each-input)
   - [4.3 Write a bench for each measured input](#43-write-a-bench)
   - [4.4 A benchmark is *any process* that emits a throughput-interpretable number](#44-a-benchmark-is-any-process)
   - [4.5 Ground, register, run](#45-ground-register-run)
5. [Reading the output — the two purposes](#5-reading-the-output)
   - [5.1 Purpose 1 — the bound ("we can achieve at least X")](#51-purpose-1-the-bound)
   - [5.2 Purpose 2 — the diagnosis (where the bound is soft / what's wrong)](#52-purpose-2-the-diagnosis)
6. [The trust ladder](#6-the-trust-ladder)
7. [Contract reference (the cheat sheet)](#7-contract-reference)
8. [Discipline and traps](#8-discipline-and-traps)

---

<a name="1-why-this-tool-exists"></a>
## 1. Why this tool exists — and its two purposes

This tool was built to interrogate a specific claim. A collaborator asserted, confidently, that the
**~200 DPS** (decisions/second) the harness was producing was *"the roof of what can be achieved."*
A first-principles calculation suggested otherwise — that we were not even halfway to the achievable
optimum.

What the tool produces is a **model-contingent lower bound**: *"under this model of the cycle, at
least X DPS is achievable."* The current models put X at **~420–456 DPS**, roughly 2× the contested
figure.

**Be precise about what that does and does not establish.** A model-derived bound is a *denotational*
claim — a number computed from a description — and it is only as strong as the model is faithful.
This model is deliberately optimistic: it omits coordination losses (RTT idle, convoy, cold-JIT) by
assumption, and several inputs are seeded engineering priors, not measurements. So the bound does
**not** *refute* "200 is the roof." A skeptic can always answer: *your model is too generous; the
losses it omits are exactly what caps a real system at 200.* The bound's honest role is to
**motivate** — it shows ~200 is not *provably* the roof, it quantifies the headroom the design might
have, and (via the sensitivities, §5.2) it says *where* that headroom would have to come from. It is
a well-founded conjecture, not a witness.

**The refutation is operational — and the tool is built to reach it.** What actually refutes "200 is
the roof" is a *witness*: a real cycle that runs and achieves more than 200. A model here is a
program in a small DSL (a cycle written as a min-of-stages over grounded primitives), and **each
primitive's operational semantics already exists, as its benchmark** — `bench_r_gen` *is* the
generator running; `bench_t_row` *is* the per-row serve cost being timed. So a model is not only
*evaluable* (compose the benchmarked means → the bound) but *executable*: lower it — port it back
into Python/C++ — into a runnable end-to-end cycle built from the benchmarked stages, run it, and
**witness** its DPS. That witnessing run, not the bound, is what refutes the roof; and because the
operational pieces are already the benchmarks, the port should be a composition, not a rewrite (§2.1).

From that origin, the tool has **two purposes**, with the witness as the bridge between them:

- **Purpose 1 — explore what is theoretically possible.** Model a protocol, get its model-contingent
  lower bound, and compare protocols to find the optimum-over-designs — a *candidate* ceiling to then
  witness.
- **Purpose 2 — diagnose an implementation.** The bound comes with a **gradient decomposition**:
  which input's uncertainty (or which stage's cost) most constrains the number. When a witnessing run
  underperforms the bound, this is the decomposition that localizes the gap — which stage binds,
  which term dominates the loss, which quantity is least pinned. *"The witness does 260; the bound
  says 430; the decomposition says the serve cycle is dominated by `t_row` and the wakeup — look
  there."* The model's optimism and the implementation's slack meet here.

The contested number is preserved in the code, neither anchored to nor pretended away: `references.py`
holds `REF_PLATEAU_DPS = 203.0` with the comment that it is *"a USER-supplied reference for ONE config
family… NOT grounded in any readable repo file."* It is a display anchor the bound never consumes
(§7) — there to be *beaten by a witness*, not matched by a model. The discipline that produced this
tool is the discipline it enforces: do not anchor on a plateau, and do not mistake a model for a proof.

---

<a name="2-the-one-idea"></a>
## 2. The one idea: a cycle is a min-of-stages, and the bound is the slowest stage

A serving cycle is a pipeline of **stages** running concurrently. Throughput is set by the
**slowest** stage — the bottleneck. So the model of a cycle is a function

```
f(inputs) = min( stage_1(inputs), stage_2(inputs), ... )      # decisions per second (DPS)
```

and the canonical cycle (the "spine" every transport variant shares) has three stages:

```
producer   = N_gen * R_gen                                    # generation: cores × per-core rate
cycle_us   = T_disp + tau_io + wakeup + B * t_row             # one serialized serve forward (microseconds)
serve      = 1e6 * B / (cycle_us * L)                         # serve: B rows per forward / per-decision leaves
transport  = 1.0 / (L * tmsg * 1e-6)                          # transport: per-leaf message cost
f          = min(producer, serve, transport)
```

Every term is a real, physically-grounded fixed cost or per-row slope. Coordination losses (RTT
idle, convoy effects, cold-JIT) are **deliberately not** put into any stage — those are exactly
what a *well-designed* implementation engineers away, and the bound is over the well-designed
optimum. That is what makes `f(μ̂)` a **lower bound**: a real cycle has every cost the model has,
plus the losses the model omits, so a real well-designed cycle achieves *at least* `f` and a real
sloppy one reveals its sloppiness as the gap below `f`.

Each input carries an **uncertainty**, not just a value. The bound is therefore a *distribution*,
and the tool reports `E[f]` together with a confidence interval whose width comes from propagating
the inputs' uncertainty through the gradient of `f`. Uncertainty enters the bound the conservative
way: a wider input spread widens the interval; a small-sample fit gets a fatter (Student-t)
multiplier than a large-sample mean; a physically-impossible interval edge (a negative latency, a
fraction above 1) is clipped to the feasible set and the clip is surfaced, never printed as a real
value.

This is the mental model. The rest of the manual is how you *express* a cycle in it.

<a name="21-dsl-operational"></a>
### 2.1 The model is a DSL program; the benchmarks are its operational semantics

The min-of-stages formula is *denotational* — it says what number a cycle *denotes*. But a cycle
written this way is also a small **program in a DSL**: `f` composes primitives (a generator rate, a
per-row cost, a per-leaf message cost) under a fixed combinator (the serialized cycle, the min over
stages). And every primitive in that DSL already has an **operational semantics written down — its
benchmark.** `bench_r_gen` is what "run the generator" *means*; `bench_t_row` is what "pay the
per-row serve cost" *means*. A benchmark is not merely a measurement *of* a primitive; it *is* the
primitive's executable behavior.

That duality is the tool's real leverage:

- **Evaluate the program denotationally** — substitute each primitive's benchmarked *mean* into `f` —
  and you get the **bound** (Purpose 1): a number, a *conjecture* about the ceiling.
- **Execute the program operationally** — *lower* it back into Python/C++, composing the primitives'
  actual benchmarked behaviors in the order `f` specifies, and run it — and you get a **witness**: a
  real end-to-end DPS that either clears 200 (refuting the roof *by construction*) or falls short (at
  which point Purpose 2's sensitivities localize the gap between the model's optimism and the running
  cycle).

The port from model to witness *should be cheap* precisely because the operational semantics is not
missing — it is the benchmark suite. Lowering a model is **composing executable pieces you already
have**, per the recipe `f` already wrote, not building a system from scratch. A model and its witness
are two readings of one DSL program: the denotational reading hands you the target, the operational
reading earns it.

This is the half of the loop that turns the tool from a calculator-of-conjectures into an
instrument-of-refutation — and it is the honest answer to *"is 200 the roof?"*: **not the bound, but
the witnessing run the bound tells you is worth attempting.** (Note: even `untrusted_drive`, which
measures every primitive live, still *evaluates the bound* with those live numbers — it does not yet
*compose* them into the end-to-end cycle. That composition — the lowering — is the witness, and it is
the natural next capability the design is shaped for.)

---

<a name="3-the-package-map"></a>
## 3. The package map

You touch three or four of these directories to model a new cycle. The geography:

```
tools/analysis/leaf_eval_bound/
  contract/        # what a measurement IS, and the domain facts the bound rests on
    estimate.py        # the typed Estimate (theta_hat, cov, shrink-law, family, support) — the keystone
    grounded_types.py  # the Grounded dataclass + the Estimability enum (measured / prior / constant)
    grounding.py       # THE TABLE of grounded physical constants (you add a constant here)
    references.py      # REF_* display anchors (incl. REF_PLATEAU_DPS=203) — never inputs to the bound
  benchmarks/      # the measurement leaves
    estimators.py      # pin_estimate / median_estimate / fit_estimate — raw numbers -> an Estimate
    scaffold.py        # scaffold.bench(...) — the wiring every bench reuses (you call this)
    harness.py         # logged_run / register_quantity / SIZING_KWARGS — the postgres glue
    bench_*.py         # one quantity's measurement each (you ADD one of these per measured input)
    register_benches.py# discovery-driven registration (finds every bench_*.py — no hand-list)
  models/          # the cycles being bounded
    model_base.py      # the TransportModel contract (the typed interface a new cycle satisfies)
    model_capacity.py, model_cycletime.py        # the two STATIC-grounded models
    model_zmq_baseline.py, model_shm_spin_poll.py, model_futex_wake.py,
    model_lockfree_mpsc.py, model_cpp_inproc_port.py   # the MANIFEST-driven transport variants
  alloc/           # the generic allocation engine (you do NOT touch this to add a cycle)
    driver.py          # AllocationDriver: one step() yields the bound AND the sensitivities
    gradient.py, jax_backend.py   # the autodiff backend (jax.grad through min())
    kink.py            # Clark-1961 min-moments — the "which stage binds, and is it certain?" math
    report.py          # Recommendation / where_to_spend — the diagnostic presentation
  store/           # the metric registry (postgres)
    manifest.py        # name -> (mean, sigma, n, trusted) / Estimate; the trust/seed/reconstruct paths
    bench_store.py     # the postgres egress (the three tables; the Estimate jsonb I/O)
    reconstruct.py     # seed -> Estimate and aggregate -> Estimate (the math the runners read)
  runners/         # the entry points you actually run
    throughput_bound.py    # the two static models, seeded, cross-checked
    transport_sweep.py     # the 5 transport variants, three honesty levels, optimum-over-transports
    untrusted_drive.py     # the live-bench loop: measure every input now, allocate, re-measure
```

The dependency direction is a clean DAG: `runners/` → `models/` → (`alloc/` + `store/`) →
`contract/`; `benchmarks/` → `contract/` + `store/`. The driver never imports a model (a model
injects its `f` into the driver), which is why you can add a cycle without touching `alloc/`.

---

<a name="4-modeling-a-new-cycle"></a>
## 4. Modeling a new cycle — the five-step workflow

Concretely: a **new cycle** is a new `model_<slug>.py` (its `f` and how its inputs resolve), plus a
`bench_*.py` for any input that does not already exist, plus a grounded seed for each new input.
"Modeling a new transport variant" is the contract case — it satisfies the typed `TransportModel`
interface and joins the sweep. The five steps:

| Step | You write | Where |
| — | — | — |
| 1. Write `f` | the throughput function (min-of-stages) + the input list | `models/model_<slug>.py` |
| 2. Classify inputs | measured / prior / constant, per input | (a decision; recorded in step 4) |
| 3. Write a bench | one `bench_*.py` per *measured* input that doesn't exist yet | `benchmarks/bench_*.py` |
| 4. Ground + register | a `Grounded` seed per new input; run discovery registration | `contract/grounding.py` + a command |
| 5. Run + interpret | nothing — you run a runner and read the output | `runners/` |

<a name="41-write-f"></a>
### 4.1 Write `f` — the model

A model is a **module** (its members are module-level functions, no `self`). To join the transport
sweep it must satisfy the `TransportModel` contract — a `@runtime_checkable` Protocol whose members
are (from `model_base.py`, the single home of the contract):

```python
INPUT_NAMES: Sequence[str]                                    # the input order; f is unpacked positionally from this
SLUG: str                                                     # the variant's registry prefix + comparison key
throughput_jax:   Callable[[Any], Any]                        # the driver's f: an x-array -> scalar DPS (JAX-traceable)
throughput_numpy: Callable[[Mapping[str, float]], float]      # the headline f: a dict -> DPS (DERIVED from throughput_jax)
registry_qname:   Callable[[str], str]                        # input name -> registry quantity name
initial_point:    Callable[..., Mapping[str, float]]          # (trust=...) -> the grounded starting point x0
sigmas:           Callable[..., Mapping[str, float]]          # (trust=...) -> per-input 1-sigma
trusted_flags:    Callable[..., Mapping[str, bool]]           # (trust=...) -> per-input "is this measured yet?"
build_driver:     Callable[..., Any]                          # (tolerance=..., trust=...) -> (AllocationDriver, x0)
cycle_breakdown:  Callable[[Mapping[str, float]], Mapping[str, float]]   # x -> per-forward cost decomposition
serve_sawtooth:   Callable[[int], float]                      # real rows -> bucketed serve DPS
```

**`f` has one home.** The throughput function is the single JAX-traceable callable
`throughput_jax`. (The historical dual-write — a muParser string for OpenTURNS plus a hand-written
numpy twin — is dissolved; do **not** reintroduce a second representation.) The numpy headline is
*derived*, not hand-written, so the two can never drift:

```python
def throughput_numpy(x: dict[str, float]) -> float:
    return float(throughput_jax([x[nm] for nm in INPUT_NAMES]))   # orders x by INPUT_NAMES; evaluates the one f
```

`jax.grad(throughput_jax)` is the analytic gradient the driver uses — exact, including through
`min()` (the arg-min tie is handled separately by `alloc/kink.py`, not by the linearization). Write
`f` as a plain min-of-stages over `jnp`; `INPUT_NAMES` is the one declaration of the signature, and
both `throughput_jax` (unpacking positionally) and `throughput_numpy` (ordering by it) read it, so
they cannot disagree.

The full model skeleton is in §4.5; read it after step 2.

> **Two dialects — pick the manifest one.** There are two historical model shapes: *static-grounded*
> (`model_capacity`, `model_cycletime`) read constants straight from `contract/grounding.py` and do
> **not** take a `trust` argument; *manifest-driven* (the five transport variants) resolve inputs
> through the registry by name and **do** take `trust`. A new cycle in the contract sense is a
> **manifest-driven** variant — it carries a `SLUG`, a `registry_qname`, and `trust`-taking
> resolvers. Copying a static model as your template will silently fail the conformance test (which
> checks that `initial_point`/`sigmas`/`trusted_flags`/`build_driver` each accept `trust`).

<a name="42-classify-each-input"></a>
### 4.2 Classify each input — measured, prior, or constant

Every input to `f` sits on one axis, `Estimability` (the single home of the measured-vs-pinned
decision — `grounded_types.py`):

| `Estimability` | Meaning | Becomes | Funded by the allocator? |
| — | — | — | — |
| `MEASURED` | a runnable bench measures it live | a **shrinkable** Estimate (median or fit) | **yes** — more measurement tightens it |
| `PRIOR` | an engineering-judgement value, no runnable bench yet | a `Fixed` Estimate, `family=NORMAL` | no — its spread is irreducible until you build the bench |
| `CONSTANT` | a true deployment/layout fact (e.g. `n_gen = 3` cores) | a `Fixed` Estimate, `family=DEGENERATE`, ~0 bound contribution | no — it drops out of allocation |

This one axis is *generative*: it drives both the model's per-input flags (`needs_measurement`,
`constant`) **and** the bench's pin-vs-shrinkable body, so a quantity cannot be labelled "measurable"
on one path and pin a constant on another (that contradiction — the "pin cascade" — was the bug a
prior refactor fixed; the axis is what prevents it recurring). Classify honestly: if there is a
runnable bench, it is `MEASURED`; if it is genuinely a layout constant, it is `CONSTANT`; an
educated guess with no bench yet is a `PRIOR` (and `PRIOR` is a standing invitation to build the
bench and promote it).

<a name="43-write-a-bench"></a>
### 4.3 Write a bench for each measured input

A bench is one `bench_<name>.py` module that measures one quantity. It declares **six** of its own
names and binds **three** from the scaffold. You never write `register_self`/`measure`/`run` — the
scaffold wires them.

The six you write:

| Name | Type | Role |
| — | — | — |
| `NAME` | `str` | the registry quantity name (unique). Slug-prefix it if it could collide across variants. |
| `MODULE_PATH` | `str` | this module's dotted import path (the manifest re-imports by it) |
| `_DESC` | `str` | a human description |
| `get_seed()` | `() -> Grounded` *(or `(mean, sigma, unit)` tuple)* | the v1 fallback used before any live measurement |
| `_measure_raw(...)` | `(...) -> dict` | **the measurement** — returns a raw-provenance dict (see §4.4) |
| `_estimate_from_raw(res)` | `(dict) -> Estimate` | turns that dict into one harmonized `Estimate` |

The estimator you call inside `_estimate_from_raw` is the whole classification decision made
concrete. Three factories (`estimators.py`), one per shape:

| Call | Use when the quantity is… | Produces | Shrinkable? |
| — | — | — | — |
| `pin_estimate(value, sigma, *, name, constant)` | a config fact or an engineering prior (no live pool) | `Fixed` (`kind='pin'` if `constant`, else `'declared_spread'`) | no |
| `median_estimate(pool, *, name)` | a sampled latency / cost / rate (a pool of readings) | `QuantileLaw`, bootstrap median SE, `family=EMPIRICAL` | **yes** — more readings → tighter |
| `fit_estimate(rows, medians_us, *, own_name, own_role, partner_name)` | a per-row OLS regression `time = intercept + slope·rows` | `RegressionLaw` (k=2: own coeff + partner), Student-t | only by widening the x-design |

Use **median** for almost every live latency/rate (timing data is right-skewed, so the median is
robust where the mean is tail-poisoned). Use **fit** when the quantity *is* a regression coefficient
(a per-row slope or a fixed-cost intercept). Use **pin** for constants and for priors you have not
yet benched.

The four skeletons in §4.4 and §4.5 are copy-accurate — start from the one whose shape matches.

**The sizing knob.** The allocator decides how much budget to spend on a bench by reading
`measure()`'s signature for a recognized sizing keyword and calling `measure(<knob>=<budget>)`. Name
a parameter of `_measure_raw` after one of `harness.SIZING_KWARGS` — `("cycles", "trials", "iters",
"n_trials", "reps", "rounds", "samples", "n", "budget", "leaves")` — and its default is the natural
size. The scaffold copies `_measure_raw`'s signature onto `measure`/`run` so the knob stays visible
to introspection. A pin bench takes no knob (correctly un-fundable); **a sampled bench that forgets
its knob is silently de-funded** — always expose it.

<a name="44-a-benchmark-is-any-process"></a>
### 4.4 A benchmark is *any process* that emits a throughput-interpretable number

This is the most important idea in the manual, and the one earlier collaborators got wrong: they
assumed every benchmark had to be a Python microbenchmark. **It does not.** The framework's only
requirement on `_measure_raw` is that it returns a `dict`. What happens *inside* it is opaque to the
tool. It can:

- time a Python/numpy loop in-process;
- **shell out to a compiled C++ binary** and parse its stdout;
- in principle, drive **the real running system** and read its emitted throughput;
- read a config file, query a database, call a remote service.

…as long as it returns a dict that `_estimate_from_raw` can turn into an `Estimate`. The
"benchmark" is whatever produces a throughput-interpretable value. The Python is glue around the
measurement, not the measurement itself.

The worked instance already in the codebase is **`bench_r_gen`** (the generator-core rate). Its
`_measure_raw` runs a C++ binary and parses the result:

```python
def _measure_raw(reps: int = 8) -> dict[str, Any]:
    if not (os.path.isfile(_BENCH_BIN) and os.access(_BENCH_BIN, os.X_OK)):
        raise FileNotFoundError(f"bench_r_gen: binary not built at {_BENCH_BIN!r} (ADR-0002).")  # fail LOUD, never seed-fallback
    taskset = ["taskset", "-c", "0"] if shutil.which("taskset") else []                          # sole-workload pin
    cmd = [*taskset, _BENCH_BIN, "--instance", _INSTANCE, "--faces", _FACES,
           "--tasks", str(_N_TASKS), "--workers", "1", "--reps", str(int(reps)), ...]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600, cwd=_REPO_ROOT, env={...})
    if proc.returncode != 0 or "RESULT: PASS" not in proc.stdout:
        raise RuntimeError(f"bench_r_gen: {proc.stdout}\n{proc.stderr}")                          # parse failure is loud
    per_rep_dps = [_N_TASKS / t for t in _RE_REP.findall(proc.stdout) if float(t) > 0]            # the POOL
    return {"r_gen_dps_per_core": float(np.median(per_rep_dps)), "per_rep_dps": per_rep_dps, "reps": int(reps), ...}

def _estimate_from_raw(res):
    return median_estimate(res["per_rep_dps"], name=NAME)                                          # a sampled rate -> median
```

Note the discipline: the binary is **located** (env-overridable), and if it is missing or fails, the
bench **raises** — it never silently substitutes the seed (ADR-0002). The raw per-rep numbers become
the pool; the estimator takes their median. *The same bench could be the real harness instead of a
microbench binary — the contract would not change.*

> Naming caution: `bench_cpp_inproc_port_*` is **not** a subprocess bench despite the `cpp_` prefix —
> that prefix names the *modeled transport* (a C++ in-process queue), and its `_measure_raw` is a
> numpy micro-loop standing in for it. `bench_r_gen` is the bench that actually shells out.

**Skeleton (a) — subprocess bench** (a compiled binary; estimator = median over the parsed pool):

```python
"""
tools/analysis/leaf_eval_bound/benchmarks/bench_example_subproc.py
LIVE benchmark for `example_rate` — measured by a C++ subprocess (median over per-rep readings).
Public Domain (The Unlicense).
"""
from __future__ import annotations
import os, re, shutil, subprocess
from typing import Any
from leaf_eval_bound.contract import estimate as _est                       # noqa: E402
from leaf_eval_bound.benchmarks.estimators import median_estimate           # noqa: E402
from leaf_eval_bound.benchmarks.scaffold import bench as _scaffold          # noqa: E402

NAME = "example_rate"
MODULE_PATH = "leaf_eval_bound.benchmarks.bench_example_subproc"
_DESC = "Example rate measured by a C++ subprocess (median over per-rep readings)."
_BIN = os.environ.get("CHOCO_EXAMPLE_BIN", "/path/to/cpp/build/chocofarm-example-bench")
_RE_REP = re.compile(r"^rep\s+\d+:\s*value=([0-9.eE+-]+)\b", re.MULTILINE)

def get_seed() -> tuple[float, float, str]:
    return (100.0, 10.0, "units/s")                                          # bare tuple -> MUST pass units= below

def _measure_raw(reps: int = 8) -> dict[str, Any]:
    import numpy as np
    n = max(2, int(reps))                                                    # >= 2 so the bootstrap median SE is defined
    if not (os.path.isfile(_BIN) and os.access(_BIN, os.X_OK)):
        raise FileNotFoundError(f"bench_example_subproc: binary not built at {_BIN!r} (ADR-0002).")
    taskset = ["taskset", "-c", "0"] if shutil.which("taskset") else []
    proc = subprocess.run([*taskset, _BIN, "--reps", str(n)], capture_output=True, text=True, timeout=600)
    if proc.returncode != 0:
        raise RuntimeError(f"bench_example_subproc: exit {proc.returncode}\n{proc.stdout}\n{proc.stderr}")
    pool = [float(v) for v in _RE_REP.findall(proc.stdout)]
    if not pool:
        raise RuntimeError(f"bench_example_subproc: no per-rep readings.\n{proc.stdout}")
    return {"value_median": float(np.median(pool)), "pool": pool, "reps": n}

def _estimate_from_raw(res: dict[str, Any]) -> "_est.Estimate":
    return median_estimate(res["pool"], name=NAME)

_B = _scaffold(
    name=NAME, quantity="example_rate_quantity", module_path=MODULE_PATH, description=_DESC,
    units="units/s",                                                         # required: get_seed() is a bare tuple
    seed=get_seed, measure_raw=_measure_raw, estimate_from_raw=_estimate_from_raw,
    run_config=lambda res, **kw: {"kind": "subprocess", "reps": res["reps"], "value_median": res["value_median"]},
    run_log=lambda res, log, **kw: log(res["pool"], sample_size=1),          # provenance pool ONLY; headline lives in the Estimate
)
register_self, measure, run = _B.register_self, _B.measure, _B.run

if __name__ == "__main__":
    print(f"[bench_example_subproc] seed: {get_seed()}"); register_self()
```

**Skeleton (b) — in-process median bench** (the most common shape; a sampled latency):

```python
"""
tools/analysis/leaf_eval_bound/benchmarks/bench_example_median.py
LIVE benchmark for `example_latency_us` — median over per-window readings.
Public Domain (The Unlicense).
"""
from __future__ import annotations
import time
from typing import Any
from leaf_eval_bound.contract import estimate as _est                       # noqa: E402
from leaf_eval_bound.benchmarks.estimators import median_estimate           # noqa: E402
from leaf_eval_bound.benchmarks.scaffold import bench as _scaffold          # noqa: E402

NAME = "example_latency_us"
MODULE_PATH = "leaf_eval_bound.benchmarks.bench_example_median"
_DESC = "Example per-op latency (us): median over per-window readings."

def get_seed() -> tuple[float, float, str]:
    return (0.5, 0.2, "us/op")

def _measure_raw(iters: int = 200000) -> dict[str, Any]:                     # `iters` is the sizing knob
    import numpy as np
    window, per_op_us = 1000, []
    for _ in range(max(2, iters // window)):                                 # >= 2 windows for the bootstrap SE
        t0 = time.perf_counter_ns()
        for _ in range(window):
            pass                                                             # <- the per-op work under test
        per_op_us.append((time.perf_counter_ns() - t0) / 1000.0 / window)
    return {"latency_us_median": float(np.median(per_op_us)), "per_op_us": per_op_us, "iters": iters}

def _estimate_from_raw(res: dict[str, Any]) -> "_est.Estimate":
    return median_estimate(res["per_op_us"], name=NAME)

_B = _scaffold(
    name=NAME, quantity="example_latency_quantity", module_path=MODULE_PATH, description=_DESC,
    units="us/op",
    seed=get_seed, measure_raw=_measure_raw, estimate_from_raw=_estimate_from_raw,
    run_config=lambda res, **kw: {"iters": res["iters"], "latency_us_median": res["latency_us_median"]},
    run_log=lambda res, log, **kw: log(res["per_op_us"], sample_size=1),
)
register_self, measure, run = _B.register_self, _B.measure, _B.run

if __name__ == "__main__":
    print(f"[bench_example_median] seed: {get_seed()}"); register_self()
```

**Skeleton (c) — pin bench** (a config fact, from a `Grounded` seed; no sizing knob = un-fundable):

```python
"""
tools/analysis/leaf_eval_bound/benchmarks/bench_example_pin.py
LIVE benchmark for `example_cores` — a FIXED config/layout fact (no microbench).
Public Domain (The Unlicense).
"""
from __future__ import annotations
from typing import Any
from leaf_eval_bound.contract import estimate as _est                       # noqa: E402
from leaf_eval_bound.contract import grounding as G                         # noqa: E402
from leaf_eval_bound.benchmarks.estimators import pin_estimate              # noqa: E402
from leaf_eval_bound.benchmarks.scaffold import bench as _scaffold          # noqa: E402

NAME = "example_cores"
MODULE_PATH = "leaf_eval_bound.benchmarks.bench_example_pin"
_DESC = "Example config fact (a deployment/layout constant); no microbench."

def get_seed() -> G.Grounded:
    return G.N_GEN_CORES                                                     # a Grounded whose .constant is True

def _measure_raw() -> dict[str, Any]:                                        # NO sizing knob -> un-fundable (correct for a pin)
    return {"value": get_seed().mean, "note": "config fact; no microbench"}

def _estimate_from_raw(res: dict[str, Any]) -> "_est.Estimate":
    seed = get_seed()
    return pin_estimate(seed.mean, seed.sigma, name=NAME, constant=seed.constant)   # `constant` DERIVED from the grounding, never hardcoded

_B = _scaffold(
    name=NAME, quantity="example_cores_quantity", module_path=MODULE_PATH, description=_DESC,
    seed=get_seed, measure_raw=_measure_raw, estimate_from_raw=_estimate_from_raw,   # units come from seed().unit
    run_config=lambda res, **kw: {"kind": "config_fact", "note": res["note"]},
    run_log=lambda res, log, **kw: log(res["value"], sample_size=None),     # sample_size None for a pin
)
register_self, measure, run = _B.register_self, _B.measure, _B.run

if __name__ == "__main__":
    print(f"[bench_example_pin] seed: {get_seed().mean} {get_seed().unit}"); register_self()
```

For a **prior** (not a true constant) use `pin_estimate(mean, sigma, name=NAME, constant=False)`
with `sigma > 0` (it raises on a spread-less prior).

**Skeleton (d) — fit bench** (a per-row OLS slope; k=2 co-fit with a partner):

```python
"""
tools/analysis/leaf_eval_bound/benchmarks/bench_example_slope.py
LIVE benchmark for `example_slope` — the SLOPE of time = intercept + slope*rows.
Public Domain (The Unlicense).
"""
from __future__ import annotations
from typing import Any, Optional
from leaf_eval_bound.contract import estimate as _est                       # noqa: E402
from leaf_eval_bound.benchmarks.estimators import fit_estimate              # noqa: E402
from leaf_eval_bound.benchmarks.scaffold import bench as _scaffold          # noqa: E402

NAME = "example_slope"
PARTNER_NAME = "example_intercept"                                          # the co-fit partner's NAME
WARMUP = 8                                                                  # generic burn-in (harness.warm runs measure() this many discarded times)
MODULE_PATH = "leaf_eval_bound.benchmarks.bench_example_slope"
_DESC = "Example per-row slope (us/row): the slope of a time = intercept + slope*rows fit."

def get_seed() -> tuple[float, float, str]:
    return (4.0, 0.5, "us/row")

def _measure_raw(rows: Optional[list[int]] = None, iters: int = 200) -> dict[str, Any]:
    import numpy as np
    rows = rows or [32, 64, 128, 256, 512]                                  # >= 3 design points (fit_estimate raises on < 3)
    med = []
    for B in rows:
        med.append(float(np.median([_timed_forward_us(B) for _ in range(iters)])))   # <- the per-width timing
    return {"per_width_median_us": dict(zip(rows, med)), "rows": rows}

def _estimate_from_raw(res: dict[str, Any]) -> "_est.Estimate":
    rows = res["rows"]
    medians = [res["per_width_median_us"][B] for B in rows]
    return fit_estimate(rows, medians, own_name=NAME, own_role="slope", partner_name=PARTNER_NAME)   # own_role MUST be this bench's coeff

def _run_config(res, **kw):
    return {"rows": res["rows"], "iters": kw["iters"]}                       # kw carries defaulted call-args (e.g. iters)

def _run_log(res, log, **kw):
    rows = res["rows"]
    log([res["per_width_median_us"][B] for B in rows], sample_size=kw["iters"])      # design points; the slope lives in the Estimate

_B = _scaffold(
    name=NAME, quantity="example_slope_quantity", module_path=MODULE_PATH, description=_DESC,
    units="us/row",
    seed=get_seed, measure_raw=_measure_raw, estimate_from_raw=_estimate_from_raw,
    run_config=_run_config, run_log=_run_log,                               # def hooks (not lambdas): they recompute an intermediate
)
register_self, measure, run = _B.register_self, _B.measure, _B.run

if __name__ == "__main__":
    print(f"[bench_example_slope] seed: {get_seed()}"); register_self()
```

The intercept partner is the same module with `NAME="example_intercept"`,
`PARTNER_NAME="example_slope"`, `own_role="intercept"` — and it may delegate its `_measure_raw` to
the slope bench's (one fit grounds both coefficients).

<a name="45-ground-register-run"></a>
### 4.5 Ground, register, run

**Ground.** Each input needs a `Grounded` seed in `contract/grounding.py`. The dataclass (frozen,
all fields required):

```python
Grounded(name, mean, sigma, cost, unit, provenance, estimability, module)
```

`cost` is the relative per-sample benchmark cost (the allocator's effort price); `provenance` is the
file/line the number came from (be specific — the table is audited); `estimability` is the §4.2
axis; `module` is the owning `bench_*`. Two real entries as the pattern:

```python
# A MEASURED quantity (a runnable bench owns its live measurement):
GEN_PER_CORE_DPS = Grounded(
    name="R_gen", mean=152.0, sigma=8.0, cost=30.0, unit="decisions/s/core",
    provenance="adapter.md §2 line 93 'MEASURED gen 152 dps/core, 4.0x linear'",
    estimability=Estimability.MEASURED, module="bench_r_gen",
)
# A true CONSTANT (a layout fact — DEGENERATE, ~0 bound contribution; sigma is a display placeholder):
N_GEN_CORES = Grounded(
    name="n_gen", mean=3.0, sigma=0.05, cost=0.5, unit="cores",
    provenance="adapter.md §6 M3 1:3 pinning; CLAUDE.md host (4-vCPU, isolcpus 1-3)",
    estimability=Estimability.CONSTANT, module="bench_n_gen",
)
```

A manifest-driven model resolves its inputs through the registry by quantity name, so for a moved
lever (a term *this* variant changes) register a slug-prefixed quantity (`f"{SLUG}_tau_io_us"`) and
seed it; for a transport-invariant pull (a shared physical fact) reference the existing bare name
(`n_gen`, `R_gen`, `B_op`, `t_row_us`, `LPD`, …). An unregistered quantity is a **loud `KeyError`**
at import (the module-level `SIGMAS = sigmas(trust=True)` resolves eagerly) — so the seeds are not
optional polish; the model will not import without them.

**The full model skeleton** (the manifest-driven variant — drop in your `f` and input map):

```python
"""
tools/analysis/leaf_eval_bound/models/model_mycycle.py
Transport variant DESIGN-mycycle: a first-principles leaf-eval throughput LOWER BOUND (DPS) on the
serialized-serve spine, with its own moved-term profile. Satisfies model_base.TransportModel (P8).
Public Domain (The Unlicense).
"""
from __future__ import annotations
from typing import Any
from leaf_eval_bound.store import manifest                                  # noqa: E402
from leaf_eval_bound.alloc.driver import AllocationDriver                    # noqa: E402

SLUG = "mycycle"
INPUT_NAMES = ["N_gen", "R_gen", "B", "T_disp", "tau_io", "wakeup", "t_row", "L", "tmsg"]
INPUT_QUANTITIES: dict[str, tuple[str, float]] = {                          # input -> (registry quantity name, cost)
    "N_gen": ("n_gen", 0.5), "R_gen": ("R_gen", 30.0), "B": ("B_op", 4.0),
    "T_disp": ("T_disp_us", 1.0), "tau_io": (f"{SLUG}_tau_io_us", 8.0),     # SLUG-prefixed = this variant's MOVED levers
    "wakeup": (f"{SLUG}_wakeup_us", 6.0), "t_row": ("t_row_us", 1.0),
    "L": ("LPD", 2.0), "tmsg": (f"{SLUG}_tmsg_us_leaf", 2.0),
}

def throughput_jax(x: Any) -> Any:                                          # the ONE f (x ordered by INPUT_NAMES)
    from leaf_eval_bound.alloc.jax_backend import jnp
    N_gen, R_gen, B, T_disp, tau_io, wakeup, t_row, L, tmsg = x
    producer  = N_gen * R_gen
    cycle_us  = T_disp + tau_io + wakeup + B * t_row
    serve     = 1e6 * B / (cycle_us * L)
    transport = 1.0 / (L * tmsg * 1e-6)
    return jnp.minimum(jnp.minimum(producer, serve), transport)            # min-of-stages: the slowest stage binds

def throughput_numpy(x: dict[str, float]) -> float:
    return float(throughput_jax([x[nm] for nm in INPUT_NAMES]))            # DERIVED — never hand-write

def _resolve(trust: bool = True) -> dict[str, "manifest.Quantity"]:
    return {nm: manifest.quantity(INPUT_QUANTITIES[nm][0], trust=trust) for nm in INPUT_NAMES}
def registry_qname(nm: str) -> str: return INPUT_QUANTITIES[nm][0]
def initial_point(trust: bool = True) -> dict[str, float]: return {nm: q.mean   for nm, q in _resolve(trust).items()}
def sigmas(trust: bool = True) -> dict[str, float]:        return {nm: q.sigma  for nm, q in _resolve(trust).items()}
def trusted_flags(trust: bool = True) -> dict[str, bool]:  return {nm: q.trusted for nm, q in _resolve(trust).items()}
def costs() -> dict[str, float]: return {nm: INPUT_QUANTITIES[nm][1] for nm in INPUT_NAMES}
SIGMAS = sigmas(trust=True); COSTS = costs()
NEEDS_MEASUREMENT = {nm: (not t) for nm, t in trusted_flags(trust=True).items()}

def build_driver(tolerance: float = 5.0, trust: bool = True):
    driver = AllocationDriver(throughput_jax, costs=[COSTS[nm] for nm in INPUT_NAMES],
                              tolerance=tolerance, names=INPUT_NAMES, confidence=0.95, growth_cap=3.0)
    return driver, initial_point(trust=trust)

def cycle_breakdown(x: dict[str, float]) -> dict[str, float]:              # the mereological decomposition (Purpose 2)
    disp, io, wake, comp = x["T_disp"], x["tau_io"], x["wakeup"], x["B"] * x["t_row"]
    cycle = disp + io + wake + comp
    return {"T_disp_us": disp, "tau_io_us": io, "wakeup_us": wake, "compute_us": comp, "cycle_us": cycle,
            "serve_dps": 1e6 * x["B"] / (cycle * x["L"]),
            "producer_dps": x["N_gen"] * x["R_gen"], "transport_dps": 1.0 / (x["L"] * x["tmsg"] * 1e-6)}

def stage_capacities(x: dict[str, float]) -> dict[str, float]:
    cb = cycle_breakdown(x)
    return {"GENERATION": cb["producer_dps"], "SERVE": cb["serve_dps"], "TRANSPORT": cb["transport_dps"]}

def serve_sawtooth(real: int, buckets=(64, 256, 512), max_batch=512, trust: bool = True) -> float:
    x = initial_point(trust=trust)
    pad = real if real > max_batch else next((b for b in buckets if b >= real), real)
    cycle = x["T_disp"] + x["tau_io"] + x["wakeup"] + pad * x["t_row"]
    return 1e6 * real / (cycle * x["L"])
```

Finally, add `"leaf_eval_bound.models.model_mycycle"` to `_VARIANTS` in
`tests/test_transport_model_conformance.py` (the conformance test *is* the type check — the tool is
not mypy-gated), and the variant joins the sweep.

**Register** (discovery-driven — it finds every `bench_*.py` on disk; no hand-list to update). All
commands run from `tools/analysis` with the project interpreter:

```bash
PY=/home/bork/w/vdc/venvs/generic/bin/python
cd /home/bork/w/vdc/1/chocofarm/tools/analysis

# register every bench's quantity DEFINITION (INSERT only, no timing — safe any time)
PYTHONPATH=. $PY -m leaf_eval_bound.benchmarks.register_benches

# confirm the registry sees your quantity (dumps the manifest, TRUST and seed views)
PYTHONPATH=. $PY -m leaf_eval_bound.store.manifest

# run ONE live measurement, pinned & sole-workload (NOT during any parallel run); logs to postgres
taskset -c 0 env PYTHONPATH=. $PY -c \
  'from leaf_eval_bound.store import manifest; print(manifest.measure("example_rate"))'
```

The metric registry is **host PostgreSQL** (`control_research` @ `192.168.122.1:5432`, psycopg3,
configured in `chocofarm/config.py` as `lab_pg_params()`). It is *not* the `127.0.0.1:6379` redis —
that backs a different subsystem (the hp registry). The store degrades gracefully: if postgres is
down it announces once and every quantity reads its seed (`trusted=False`), so the bound still
computes.

**Run** (§5).

---

<a name="5-reading-the-output"></a>
## 5. Reading the output — the two purposes

There are three runners. All run from `tools/analysis`:

| Runner | Command | What it does | Trust |
| — | — | — | — |
| `throughput_bound` | `python -m leaf_eval_bound.runners.throughput_bound` | both static models, seeded, cross-checked | SEEDED |
| `transport_sweep` | `python -m leaf_eval_bound.runners.transport_sweep` | 5 transport variants × 3 honesty levels; optimum-over-transports | SEEDED |
| `untrusted_drive` | `python -m leaf_eval_bound.runners.untrusted_drive [slug]` | the live-bench loop: measure every input now, allocate, re-measure | UNTRUSTED + confounded |

`untrusted_drive` takes env knobs (`UD_PILOT`, `UD_ROUNDS`, `UD_ITERS_CAP`, `UD_TOL`); it runs live
JAX-fit benches and is **order-of-minutes** — lower `UD_PILOT`/`UD_ITERS_CAP` for a fast spin, or it
will look hung.

Every runner is one `AllocationDriver` consuming a model's `f`. A single `driver.step()` produces
**both** purposes from the **same** gradient evaluation, which is why they are reported together.

<a name="51-purpose-1-the-bound"></a>
### 5.1 Purpose 1 — the bound ("we can achieve at least X")

The headline is `E[f] = <X> dps` — `f` evaluated at the grounded mean point — the provable lower
bound, followed by its delta-method confidence half-width `mult·√(gᵀΣg)`. Real values from a current
run:

```
Design-A (capacity)  : 420 +/- 98 dps   (~2.07x the ~203 plateau)
Design-B (cycle-time): 429 +/- 53 dps   (~2.11x the ~203 plateau)
The two routes agree to within 9 dps (the cross-check).
Both lower bounds sit ABOVE the ~203 reference, so ~203 is NOT near the achievable optimum's floor.
```

How to read it: the two *independent* static derivations agreeing to 9 DPS is the cross-check that
the number is real, not an artifact of one model. Both sit ~2× above the contested ~203 — the
finding the tool was built to produce. In `transport_sweep`, the HEADLINE optimum is
`cpp_inproc_port` at **456 DPS**, but note it is *generation*-bound there (the transport stopped
being the bottleneck — the producer cores cap it); under the conservative 1.9× producer worst case
every variant collapses to the same ~289 DPS producer floor (at which point the optimum is a
*producer* question, not a transport one — a finding in itself).

Always read the bound with its **trust level** (§6) and the printed caveats (these are BENCH-DPS
bounds at a full-bucket feed; `tau_io` — server drain/scatter — is currently UNMEASURED and is the
top measurement target). And keep the category straight: this is a *model* bound. The two static
routes agreeing to 9 DPS raises confidence that the *model* is internally consistent — it does not,
by itself, refute the ~200 roof. Only a **witnessing run** (§2.1) — the model lowered to a runnable
cycle and clocked — converts the bound from a defended conjecture into a refutation.

<a name="52-purpose-2-the-diagnosis"></a>
### 5.2 Purpose 2 — the diagnosis (where the bound is soft / what's wrong)

The same step ranks every input by its **sensitivity** `a_i = (∂f/∂x_i)² · σ_i²` — how much that
input's uncertainty inflates the variance of the bound. The largest `a_i` is the quantity whose
imprecision most constrains the number: the place to look when a real harness underperforms, and
the place to spend measurement effort. This is the mereological decomposition — it attributes the
bound's softness to specific terms. A real ranking (Design-B):

```
[iter 1] continue  E[f]=428.8                                                  <- the bound (Purpose 1)
  CI half-width = 53.05  (target 5)   shadow price lambda = 125.7   mult = 1.96 (z)
  irreducible prior floor: var=732.6 (CI 53.05)   shrinkable: var=~0   <- the CI rests on this prior; sampling cannot reach the target
  primitive              n       sigma     |df/dx|         a_i     a_i/n_i  +samples
  ----------------------------------------------------------------------------------
  L                      1          25      0.8576       459.7       459.7         0
  t_row                  1        0.15       91.94       190.2       190.2         0
  B                      1          64      0.1246       63.63       63.63         0
  T_io                   1          12      0.3591       18.57       18.57         0
  T_disp                 1           2      0.3591      0.5159      0.5159         0
  R_gen                  1           8           0           0           0         0
  N_gen                  1        0.05           0           0           0         0
```

Reading this block:

- **`a_i` / `share`** — the ranking. Here `L` (leaves-per-decision) and `t_row` (per-row serve cost)
  carry the bound's uncertainty (`a_i` 460 and 190). If the harness is slow, the model says the
  serve cycle's row cost and the per-decision leaf count are where the leverage is.
- **`+samples`** — the allocator's recommendation: how many more samples to draw from each bench this
  round. It comes from the cost-constrained optimal allocation `n_i* ∝ √(a_i / c_i)`, but **only for
  fundable inputs** — an input is funded only if it moves `f` *and* its variance actually responds to
  effort. On a **seeded** run every `+samples` is 0 even though `a_i` is large: a prior is
  un-shrinkable by sampling. **This is not a broken allocator** — the message is "the way to tighten
  these is to *run their benches* (flip them from prior to measured), not draw more samples."
- **`var_floor` / `var_shrinkable`** — the CI's variance split into the part resting on irreducible
  priors and the part sampling can reduce. When the floor alone exceeds the target (the line "the CI
  rests on this prior; sampling cannot reach the target"), the loop honestly *cannot* converge by
  sampling — the only lever is to build/run the seeded inputs' benches. A true constant (a
  `DEGENERATE` pin like `n_gen`) is in neither bucket; it is zeroed entirely.
- **The min()-kink** — `f = min(stages)`, so the binding stage is the bottleneck, reported in
  `stage_capacities` / `cycle_breakdown`. When a non-binding stage comes within a statistical tie of
  the binding one, the driver enters the **kink regime** (Clark-1961, `alloc/kink.py`): it reports a
  de-biased `E[min]`, the probability the bottleneck flips, funds *both* contending stages, and
  refuses to converge while the flip probability is high. This is the "is the bottleneck even
  certain?" diagnostic — a tie means *two* things need measuring before you trust which stage caps
  you.

So Purpose 2 in one sentence: the bound tells you the ceiling; the sensitivities tell you which term
is holding the ceiling down and whether you can buy it down by measuring or only by re-engineering.

---

<a name="6-the-trust-ladder"></a>
## 6. The trust ladder

How much to believe a bound depends on where its inputs sit on this ladder. A number does not change
how it is *computed* across these — only how much evidence stands behind it:

1. **Seeded (first-principles).** Every input is a `Grounded` prior — an engineering-judgement σ,
   un-shrinkable by sampling. `throughput_bound` and `transport_sweep` are here (the sweep prints
   "0 / 43 inputs trusted"). A defensible estimate, **not** a measured floor.
2. **Grounded-in-a-fit.** A sub-class: a few inputs are real measured read-offs (a fit slope, a
   dispatch floor) while the rest are seeds. The sweep separates these in its "grounded-vs-unmeasured
   split."
3. **Trusted-measured.** What an input *becomes* once you run its bench sole-workload and the manifest
   flips `trusted=True` **automatically** (no model edit — the model resolves through the registry,
   and the registry now has a measurement). This is the goal state for the high-`a_i` inputs.
4. **Untrusted + confounded.** `untrusted_drive` measures every input live *this run*, proving the
   loop runs end-to-end — but the current benches are Python and the cross-thread ones carry GIL
   handoff in the timing path, so the numbers are confounded. It proves the *mechanism*, not the
   floor, until the benches are native.

The trip from rung 1 to rung 3 for a given input is exactly: **build its bench (if `PRIOR`), then run
it sole-workload.** Nothing in the model changes — that is the point of resolving inputs through the
registry.

---

<a name="7-contract-reference"></a>
## 7. Contract reference (the cheat sheet)

**`scaffold.bench(...)`** — keyword-only (`benchmarks/scaffold.py`):

```python
bench(*, name, quantity, module_path, description, units=None,
      seed, measure_raw, estimate_from_raw, run_config, run_log) -> SimpleNamespace(register_self, measure, run)
```
`units=None` reads `seed().unit` (a `Grounded`); pass an explicit string only when `get_seed()`
returns a bare `(mean, sigma, unit)` tuple. `run_config(res, **kw)` → the per-run config jsonb;
`run_log(res, log, **kw)` → emits provenance via `log(values, sample_size=…)`; both receive the
call's kwargs with defaults applied (so `kw["iters"]` works). Keep `_measure_raw`/`_estimate_from_raw`
as module-level *named* functions (the scaffold re-resolves them by name so they stay mockable).

**The estimator factories** (`benchmarks/estimators.py`):

```python
pin_estimate(value, sigma, *, name, constant=False) -> Estimate     # Fixed; constant=True -> DEGENERATE/'pin', else NORMAL/'declared_spread' (sigma>0)
median_estimate(pool, *, name, n_boot=2000, boot_seed=0) -> Estimate# QuantileLaw, bootstrap median SE, EMPIRICAL; pool >= 2, non-degenerate
fit_estimate(rows, medians_us, *, own_name, own_role, partner_name) -> Estimate  # RegressionLaw, k=2; own_role in {'slope','intercept'}; >= 3 points
```

**The `Estimate`** (`contract/estimate.py`, frozen) — `theta_hat` (the point(s) `f` is evaluated at),
`cov` (the **already-divided** sampling covariance — an SE², not a per-sample variance), `names`,
`shrink` (a `ShrinkLaw`: `Fixed`/`QuantileLaw`/`RegressionLaw`/`Poolwise`/`Composed` — how `cov`
responds to effort), `support` (`REAL`/`POSITIVE`/`UNIT` or `(lo,hi)` — clips the CI to the feasible
set), `family` (`NORMAL`/`STUDENT_T(dof)`/`EMPIRICAL`/`DEGENERATE` — the CI multiplier), `kind` (a
provenance label). Construction validates and **raises** on any violation (ADR-0002); it never
coerces.

**`TransportModel`** members — see §4.1. The conformance test (`tests/test_transport_model_conformance.py`)
is the enforcement; add a new variant to its `_VARIANTS` list.

**`Grounded`** (`contract/grounded_types.py`, frozen) — `name, mean, sigma, cost, unit, provenance,
estimability, module`; `.constant` and `.needs_measurement` are derived properties of `estimability`
(never stored, so they cannot disagree with the bench body).

**`manifest.quantity(name, *, trust=True) -> Quantity`** — resolves `name` to `(mean, sigma, n,
trusted)` + the `Estimate`. `trust=True`: the latest measurement (`trusted=True`) or, failing that,
the seed (`trusted=False`). `trust=False`: forces the seed. An unregistered name is a loud `KeyError`.

**Registry** — host PostgreSQL `control_research` @ `192.168.122.1:5432` (psycopg3;
`chocofarm/config.py: lab_pg_params()`). Three tables: `benchmark_definition` (one per quantity),
`benchmark_instance` (one per run, carries the `Estimate` jsonb — the variance authority),
`benchmark_sample` (raw readings). `register_benches.py` discovers and registers every `bench_*.py`.

**Run** — `python -m leaf_eval_bound.runners.{throughput_bound|transport_sweep|untrusted_drive}` from
`tools/analysis` (interpreter `/home/bork/w/vdc/venvs/generic/bin/python`). Live `run()`/`measure()`
is timing-sensitive — pin sole-workload with `taskset -c 0`, never during a parallel job.

---

<a name="8-discipline-and-traps"></a>
## 8. Discipline and traps

- **A model-bound is a conjecture, not a refutation.** State it as *"under this model, at least X"* —
  never as a settled fact or a disproof of an empirical number. The proof is the **witnessing run**
  (§2.1): the model lowered to a runnable cycle, built from the benchmarks (its primitives'
  operational semantics), that clears the number. Until then the bound motivates the search; it does
  not end it.
- **The bound is a *lower* bound — keep it honest.** Put only real staged costs into `f`;
  coordination losses belong to the gap a real implementation reveals, not to the model. Never widen
  a stage to "explain" a slow harness — that hides the very thing Purpose 2 is for.
- **Classify on one axis.** A quantity's `Estimability` drives both the model flags and the bench
  body. Do not label a quantity `MEASURED` and then pin it (the "measured-but-punted" lie); do not
  hardcode a bench's `constant=` — derive it from the seed. An agreement guard test enforces this.
- **The fit `own_role` trap.** A fit logs a k=2 Estimate (slope + intercept); the driver projects
  **component 0** as the bench's own marginal. Set `own_role` to *this* bench's coefficient, or a
  slope-reader silently gets the intercept (a wrong number, not an error).
- **Fits are not funded by `iters`.** A `RegressionLaw`'s variance is leverage-floored — pouring
  iterations in does not tighten the slope SE. The lever is widening the x-design (more, more-spread
  batch sizes) or running the bench to flip it trusted. A median bench *is* funded by more readings.
- **Tuple seed ⇒ pass `units=`.** A `get_seed()` returning a bare `(mean, sigma, unit)` tuple has no
  `.unit` attribute; omitting `units=` from the scaffold call fails at register/run time.
- **Do not re-log the headline.** When an `Estimate` is attached (always, for a live bench), the
  headline lives in `estimate.theta_hat`; `run_log` logs only raw pool members / design points.
  Re-logging the headline corrupts the sample aggregate.
- **Fail loud (ADR-0002).** A missing binary, an unparseable output, an unregistered quantity — raise,
  never silently fall back to a seed. A silent fallback turns a broken measurement into a confident
  wrong bound, which is the one outcome this tool exists to prevent.
- **Do not anchor on `REF_*`.** The references (incl. `REF_PLATEAU_DPS=203`) are display comparisons,
  never inputs. Re-derive the bound, and beat the reference with a *witness* — do not match it with a
  model.
