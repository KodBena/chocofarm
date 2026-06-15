# Whole-project architectural refactoring audit for chocofarm — the abstractions, where they leak, and the target structure (2026-06-15)

A whole-project design-and-audit note, analysis only — no code was changed and no
job was run. The thesis the maintainer handed down: **abstractions are what keep
software maintainable, and this project has accumulated architectural debt across
the whole tree** — conflated responsibilities, missing or leaky boundaries, wrong
or absent abstractions, god-objects, hidden coupling. The brief names one worked
instance (training and optimization conflated in `JaxTrainer`) and is explicit
that it is **one entry among many, not the spine**. This note audits the entire
`chocofarm/` package through one lens, applied uniformly:

> **What is the right abstraction / boundary here, and how does the current code
> depart from it?**

SSOT / DRY / "make-illegal-states-unrepresentable" are used as *diagnostics that
locate a misplaced abstraction* — never as the headline. A duplicated reference
constant, a hand-re-derived `_JDTYPE`, a slot count computed in two files: each is
a *symptom* that points at a boundary in the wrong place. The cure is the right
boundary, from which the specific edits fall out.

The tone is the project's: name what each change costs, name where a boundary is
genuinely fine and should be left alone, and prefer "this dissolves that specific
coupling" over "this is cleaner." The note is structured top-down — the honest map
first (§1), the problems found across the tree ranked by architectural leverage
(§2), the target architecture the problems imply (§3), the sequenced plan (§4),
the C++-sim seam check (§5), and honest caveats with per-area coverage depth (§6).

The companion notes compose, they do not contradict: the narrow
`docs/design/training-optimization-refactor.md`
(branch `docs/training-optimization-refactor`) is folded in as the optimizer/
training entry (§2.1, §3.2) — this general audit supersedes its framing but keeps
its mechanism; and `docs/design/simulation-parallelization-viability.md`
(branch `docs/sim-parallelization-viability`) characterizes the simulation hot
path and the compiled-core seam that §5 verifies the target architecture keeps
clean.

---

## 0. The one fact that organizes everything

chocofarm has exactly **one genuinely clean architectural seam**, and it is the
most important one in the project: `Environment` + `Policy`. `env.py` "knows
nothing about HOW a decision is made — that is a `Policy`, passed in"
(`env.py:8-11`), and `Environment.simulate(policy, world, lam, rng, max_steps)`
(`env.py:138`) is data-in/data-out. Every solver consumes the env *only* through
the documented contract (`legal_actions`/`apply`/`filter_*`/`marginals`/
`exit_cost`/`sample_world`/`d`) and reaches for no internals. That seam is why a
C++ simulation core is even on the table (§5), and it is the model every other
boundary in this audit should be measured against.

The debt is **everything built around that seam**: the learner side (training/
optimization conflated, dtype/forward-selection scattered, value-target logic
split across two files), the search (one class that is both a search engine and a
Policy and a target-synthesizer), the supporting machinery (rollouts, candidate
pruning, the Dinkelbach step, belief fingerprinting, all reimplemented per
solver), the transport (a god-object that fuses pool + redis + task I/O), the
bounds module (five V̂ strategies bundled; a second `Environment` reimplemented in
`MiniEnv`), and the eval/entry-point layer (nine near-identical `main()` scripts,
reference constants hardcoded in ten places, two of which already disagree). The
env↔policy seam is the proof the project *can* draw a clean boundary; the rest is
the work of drawing the others to the same standard.

---

## 1. The honest architecture map

The package as it actually is — responsibilities, the dependency structure, and
where coupling crosses a boundary it should not. Evidence is file:line.

### 1.1 The layers

```
                         ┌─────────────────────────────────────────────┐
   SIMULATION (clean)    │ model/env.py  — Environment (belief mechanics,│
                         │   dynamics, simulate/rate/dinkelbach)        │
                         │ model/arrangement.py — planar faces (geometry)│
                         │ model/facemodel.py   — SenseAction (UN-WIRED) │
                         └───────────────▲──────────────────────────────┘
                                         │ Policy.decide(env, …)  (the seam)
        ┌────────────────────────────────┼────────────────────────────────┐
        │ SOLVERS                         │                                │
        │ solvers/base.py (Policy ABC + 4 base policies + _base_value)     │
        │ solvers/{ismcts,nmcs,uct,decomp}.py  — tree searches             │
        │ az/gumbel_search.py (GumbelAZSearch + GumbelPolicy)              │
        └───────────────▲─────────────────────────────▲──────────────────┘
                        │                              │
        ┌───────────────┴──────────┐    ┌──────────────┴───────────────────┐
        │ LEARNER (az/)            │    │ BOUNDS (bounds/)                 │
        │ mlp.py (inference+persist)│    │ info_relaxation.py (penalty +    │
        │ mlp_jax.py (jit forward) │    │   5 V̂ strategies bundled)        │
        │ mlp_jax_train.py         │    │ eval_bound.py (validation driver)│
        │   (JaxTrainer: TRAIN ⊕   │    │ minienv.py (a 2nd Environment)   │
        │    OPTIMIZE conflated)   │    └──────────────────────────────────┘
        │ features.py, actions.py, │
        │ value_target.py, kernels,│    ┌──────────────────────────────────┐
        │ dtypes.py, dataset.py    │    │ EVAL / ENTRY POINTS (eval/)      │
        │ parallel.py (god-object) │    │ harness.py (2 ref-lines + dead   │
        │ exit_loop.py (the loop)  │    │   main); eval_{az,decomp,ismcts, │
        └───────────────▲──────────┘    │   nmcs,uct,faces}.py — 9 mains   │
                        │               │ tb_runner.py, feature_response.py│
        ┌───────────────┴──────────┐    └──────────────────────────────────┘
        │ CONFIG / REGISTRY        │
        │ config.py (redis facts)  │    ┌──────────────────────────────────┐
        │ hp/schema.py (typed SSOT)│    │ ANALYSIS (analysis/) — ORPHANED  │
        │ hp/registry.py (live read)│   │ analyzer.py, synthetic.py        │
        └──────────────────────────┘    └──────────────────────────────────┘
```

### 1.2 What each layer is, and the one-line verdict

- **`model/` (simulation).** `env.py` is the project's best-factored object: one
  responsibility, a clean Port (`Policy.decide`), no leaks. `arrangement.py` is a
  clean geometry layer the env consumes through `arrangement.load()`
  (`env.py:43`). **`facemodel.py` is un-wired** — `env.py` references it only in a
  comment (`env.py:35`); the live env builds detectors from `arrangement.load()`
  directly (`env.py:43-46`) and `SenseAction`/`sense_actions()` have no caller.
  It is a design artifact living in the source tree.

- **`solvers/`.** The `Policy` ABC (`base.py:16-19`) is real and all six policies
  honor it. But `base.py` conflates the *interface* with *four concrete base
  policies* and the module-level `_base_value` helper (`base.py:121-131`); and the
  tree searches reimplement shared scaffolding (§2.4).

- **`az/` (learner).** The densest debt. `mlp_jax_train.py`'s `JaxTrainer` is two
  objects in one (Trainer ⊕ Optimizer, §2.1). `gumbel_search.py` is three
  responsibilities in one (search ⊕ Policy ⊕ target-synthesis, §2.2). The
  net-shape/feature-layout/action-slot knowledge is spread across `features.py`,
  `actions.py`, `mlp.py`, `feature_response.py`, `gumbel_search.py` (§2.3). The
  dtype/forward-selection decision is scattered (§2.6). `parallel.py`'s
  `ParallelExecutor` is a god-object (§2.5).

- **`bounds/`.** `info_relaxation.py` bundles the penalty machinery with five V̂
  strategies (§2.7); `minienv.py` is a second `Environment` reimplemented by hand
  (§2.7).

- **`eval/`.** Nine near-identical `main()` entry points; the two reference-line
  functions in `harness.py` are shared, but the reference *constants*, the policy
  packs, the Dinkelbach budgets, and the CLI args are copy-pasted per script
  (§2.8). `harness.main()` is dead.

- **`hp/` + `config.py`.** Recently landed and genuinely well-shaped: the schema
  is a typed SSOT, the registry is a strict fail-loud codec, `config.py`
  consolidated the duplicated `_redis_params()`. This layer is the *model* the
  rest of the audit borrows from — it is what "one authority, read live" looks
  like done right. Its one structural soft spot (the `Mut` facet is a
  hand-maintained reading, `schema.py:18`) is addressed in §2.1/§3.2.

- **`analysis/`.** `analyzer.py` (606 lines, ~20 analysis functions) and
  `synthetic.py` are used only by each other (`synthetic.py:46,134`); nothing in
  the live pipeline imports them, and the solver's cluster definitions are
  hardcoded rather than read from the analyzer. Orphaned diagnostic code in the
  package tree.

### 1.3 The coupling that crosses boundaries it should not

| coupling | where | why it is wrong |
|---|---|---|
| Trainer reads + writes the Optimizer's state | `mlp_jax_train.py:215-220, 240-245` | one object owns two lifetimes (§2.1) |
| search owns the value-target rule | `gumbel_search.py` `_v_mix`/`_improved_policy` | target synthesis lives in the engine, not `value_target.py` (§2.2) |
| net shape / feature layout known in 5 files | `features.py`, `actions.py`, `mlp.py`, `feature_response.py`, `gumbel_search.py:104` | no SSOT for the feature/action contract (§2.3) |
| transport knows the net's field layout | `parallel.py:107-109` (`pack_net` manifest) | transport coupled to `ValueMLP` internals (§2.5) |
| `MiniEnv` reimplements env belief mechanics | `minienv.py:81-115` vs `env.py:99-135` | a second simulation model, hand-kept in sync (§2.7) |
| reference rates hardcoded | 10 sites incl. `exit_loop.py:49-51`, `eval_az.py:34,79` | values already disagree (0.094 vs 0.0941) (§2.8) |
| `_JDTYPE` derivation duplicated | `mlp_jax.py:37` == `mlp_jax_train.py:54` | the precision-policy decision has no home (§2.6) |

---

## 2. The architectural problems, ranked by leverage

Ordered by how much maintainability the right abstraction buys, highest first.
Each entry: **where**, **what abstraction is wrong/missing/conflated**, **why it
hurts**, **the diagnostic symptoms**. The optimizer/training conflation is the
first entry because it is the cleanest worked case and the maintainer named it —
but it is *one* of eight, and §2.2–§2.8 carry comparable weight.

### 2.1 Training and optimization are conflated in `JaxTrainer` (one object, two lifetimes)

**Where.** `mlp_jax_train.py:187` (`class JaxTrainer`). Its `__init__`
(`:206-220`) builds the optimizer (`self.opt = optax.adam(learning_rate=self.lr,
…)`, `:215`), captures `l2` and bakes it into the jit closures (`_make_az_update(
self.opt, self.l2)`, `:219`), and reads the net weights + inits Adam's moment
state (`:216-217`) — all in one constructor.

**The wrong abstraction.** One class wears two hats with *different lifetimes and
different mutation semantics*. The **Trainer** owns the loss, the data
marshalling, the y-standardization re-pin, the write-back; its natural lifetime is
*per-run* (built once so the moments persist). The **Optimizer** owns the optax
transform and the coefficients `lr`/`l2`/`betas`/`eps`; conceptually its
coefficients are read *each step*. Because the two share a constructor, the
optimizer's coefficients are captured when the *Trainer* is built — and the
constraint that makes Adam's moments persist (don't rebuild the Trainer) is,
accidentally, the constraint that bakes `lr`.

**Why it hurts.** Fields that *ought* to be live can't be. The handoff's queued
lr-anneal experiment (`handoff-2026-06-15.md` §Pending-3, "resume at `--lr 1e-4`")
is forced through `--resume` because there is no live point-of-use for `lr`. The
registry already reads HOT fields live every iteration (`exit_loop.py:317-336`)
and rebuilds the search at the boundary (`:353`) — the trainer is the *one* object
on the per-iteration path that is built once and therefore stuck.

**Diagnostic symptoms** (each a pointer back to this misplaced boundary):
- *DRY/SSOT:* `lr` lives in the schema (`schema.py:151`) **and** is copied into
  `self.lr` (`mlp_jax_train.py:208`) **and** baked into `optax.adam` (`:215`) —
  three authorities that can silently drift. Same for `l2`/`beta1`/`beta2`/`eps`.
  This is the exact shape `config.py` just fixed for redis params, recurring.
- *Vestigial interface:* `train_step` and `train_epochs` carry `lr`/`l2` in their
  signatures (`mlp_jax_train.py:247`, `exit_loop.py:139`) but use neither — a
  channel that *looks* live and is not (`mlp_jax_train.py:198-199` admits it).
- *Reach-in:* `sync_from_net()` resets the optimizer by calling `self.opt.init(…)`
  (`:240-245`) — the Trainer mutating the Optimizer's state because it owns both.

**The fix is structural** (detailed in §3.2): split the Optimizer out as a small
object whose hyperparameters are *injected runtime state* (`optax.inject_
hyperparams`), supplied per step from the live snapshot. Once that object reads
its coefficients from the call each step, `inject_hyperparams` is not a feature you
add — it is the only way to write the object at all, and `lr`/`l2`/`betas`/`eps`
become HOT because there is nowhere left to bake them. (Full mechanism, the
HOT-ness table, and the SSOT/MISU treatment are in the folded-in
`training-optimization-refactor.md`; §3.2 here states the target shape, §4 sequences
it as one arc among the others.)

### 2.2 `GumbelAZSearch` is three responsibilities fused (search ⊕ Policy ⊕ target rule)

**Where.** `gumbel_search.py`. The class is (a) the **search algorithm**
(Sequential Halving + Gumbel-Top-k root + PUCT interior descent), (b) a **Policy**
(`GumbelPolicy`, `:435-446`, wraps the search as a `decide`), and (c) the
**value-target synthesizer** (`_v_mix` and `_improved_policy`, ~`:388-432`, compute
the Danihelka σ-transform `softmax(logit + σ(completed_q))` — which *is* the §4.4
policy target of the design).

**The wrong abstraction.** Three things with three different reuse profiles are
welded together. The search engine should emit raw `(visited_q, count)` statistics;
the **target rule** (how those become an improved policy) belongs in
`value_target.py` alongside the return-to-go rules it already owns; the **Policy
adapter** (argmax the improved policy) is a thin wrapper that should not require
the engine to know it is being used as a Policy.

**Why it hurts.** The target rule is the project's actual research frontier — the
calibration agenda (`consult-003`, handoff §4) is *about* value/policy targets.
Today a second search (a PUCT-only variant, the ablation the design's §7 calls
for, or a future hierarchical search) cannot reuse the Danihelka improved-policy
rule without either re-implementing `_v_mix`/`_improved_policy` or importing the
whole `GumbelAZSearch`. The target math is also *exactly* the part that should be
unit-testable in isolation (it is pure), and it is buried in a stateful engine.

**Diagnostic symptoms.**
- *Misplaced cohesion:* `value_target.py` owns `suffix_returns_to_go` /
  `blended_returns_to_go` (the value-target rules) but the *policy*-target rule
  lives in `gumbel_search.py` — the two halves of "the AZ target" are in two files.
- *Hidden coupling:* `_improved_policy` side-reads node statistics; it is not a
  function of explicit inputs, so it cannot be called outside a live tree.

**The fix** (§3.3): extract `improved_policy_from_stats(logits, root_q, legal_slots,
root_value, prior, c_visit, c_scale)` and `v_mix(...)` into `value_target.py` as
pure functions; the search calls them; `GumbelPolicy` becomes a trivial adapter.

### 2.3 The feature/action/net-shape contract has no single source of truth

**Where.** The same structural fact — "the belief is encoded as N×5 + nD×3 +
global floats, and the action space is N + nD + 1 slots" — is independently
re-expressed in five places:
- `features.py:195-229` builds the vector with hardcoded block offsets; its layout
  is *also* described in prose (`features.py:14-20`).
- `actions.py:109-119` reads sub-block offsets (`2N..3N` available, `5N..5N+nD`
  informative) to derive the legal mask from a feature vector.
- `feature_response.py:44-58` hardcodes the feature *names* per block for the
  importance diagnostic.
- `gumbel_search.py:104` re-derives `term_slot = env.N + len(env.detectors)` by
  hand instead of calling `actions.term_slot(env)` (`actions.py:36`).
- `mlp.py` takes `n_actions` as a construction parameter with **no validation**
  against the env (`mlp.py:29` documents the expected `N+nD+1` only in a comment).

**The wrong abstraction.** There is no `FeatureLayout` / action-space *object* that
owns the contract. The layout is a convention re-encoded at each consumer. The
action-slot *count* is nearly SSOT (one definition, `actions.py:31-33`) — but its
*structure* (term is last; treasures then detectors then term) is assumed by hand
in `gumbel_search.py:104` and unenforced when a net is loaded.

**Why it hurts.** Changing the featurization (the design's §2.3/§7 ablations
explicitly anticipate adding cluster-count or clause channels) requires synchronized
edits in four files, and a desync is *silent* until a runtime shape assert
(`features.py:231`) — or, worse, until a loaded net's `n_actions` disagrees with the
env and the masked softmax indexes garbage with no guard at all.

**Diagnostic symptoms.** Four-file edit to add one channel; a `term_slot`
re-derivation that duplicates `actions.term_slot`; a net→env compatibility that is
checked for `in_dim`/`n_actions` only on `--resume` (`exit_loop.py:204`) and never
inside `GumbelAZSearch`/`NetValueISMCTS` construction.

**The fix** (§3.4): a `FeatureLayout` value object (built from the env) that owns
the block structure and exports offsets, names, and the legal-index ranges;
`features.py`/`actions.py`/`feature_response.py` read from it. A
`validate_net_against_env(net, env)` that `GumbelAZSearch.__init__` and
`NetValueISMCTS.__init__` call, so a shape mismatch is loud at construction
(ADR-0002) rather than silent in the forward.

### 2.4 The tree searches reimplement shared scaffolding (no rollout / candidate / step abstractions)

**Where.** Across `base.py`, `ismcts.py`, `nmcs.py`, `uct.py`, `decomp.py`:
- **Base-playout** is called four times via the module-level `_base_value`
  (`ismcts.py`, `nmcs.py`, `uct.py`, `decomp.py` all import and call it) — a
  free function, not a composable object, so each search wires it by hand.
- **Candidate-action pruning** (nearest-K informative detectors + treasures) is
  inlined identically in `RolloutPolicy` (`base.py:69-73`) and `NMCSPolicy`
  (`nmcs.py:89-100`), with *different parameter names* for the same knob
  (`near_det` vs `cand_det`).
- **The Dinkelbach step** `r - lam*dt` and the λ-penalized exit `-lam*exit_cost`
  are inlined in every solver (`base.py:110`, `ismcts.py:162`, `nmcs.py:205`,
  `uct.py:189`, `decomp.py:274`).
- **Belief fingerprinting** is reinvented per search: ISMCTS's `_belief_key`
  `(n, min, max)` triplet vs decomp's cluster-bit projection (`decomp.py` local
  support).

**The wrong abstraction.** The *interface* (`Policy`) is right and should not
change. The *machinery a tree search needs* — "play the base policy to the end,"
"the nearest-K candidate actions," "the λ-penalized step value," "a hashable
belief key" — is the missing abstraction. These are not policies; they are the
shared substrate every tree policy reaches for and re-implements.

**Why it hurts.** The objective term `r - lam*dt` is the *definition of what is
being optimized* (the Dinkelbach reformulation); having it inlined in ~20 places
means the objective lives nowhere and everywhere. If the time model is recalibrated
(the standing caveat in every design note — symmetric Euclidean travel is an
approximation), or the base playout must change, the edit fans out across five
files with no compiler to catch a missed one.

**Diagnostic symptoms.** Identical pruning logic with divergent parameter names; a
free `_base_value` function imported four times; the objective formula textually
repeated.

**The fix** (§3.5): a small `rollout` / `search-support` module — `step_value(r,
dt, lam)`, `candidate_actions(env, loc, bw, collected, n_det, n_tre)`,
`RolloutExecutor(base).value_to_end(...)`, and a `belief_key` protocol — that the
tree searches *compose* rather than re-implement. None of this touches the `Policy`
seam.

### 2.5 `ParallelExecutor` is a god-object fusing pool + transport + task I/O

**Where.** `parallel.py:326-451`. One class owns: the process pool
(`__init__`, `:331-342`), the redis connection and weight broadcast (`:336`,
`generate` `:356-374`), task construction as positional tuples (`:368-370`),
result collection/deserialization (`:376-402`), and teardown (`:422-450`). Worker
state is threaded into a module-global dict `_W` (`:132-135`, `:191`) holding
11 fields that tasks read directly (`_W["env"]`, `_W["search"]`, …).

**The wrong abstraction.** Three boundaries are collapsed into one: the **pool**
(workers, core-pinning), the **transport** (raw-bytes weights in, transition
records out — the part that is actually well-designed: pickle-free, `:251-271`),
and the **task contract** (what a generate/eval task takes and returns). They have
independent reasons to change (a different transport; a different pool; a new task
field) and today any of them requires editing the one class plus the global-dict
unpack.

**Why it hurts.** The transport is the *exact* boundary the C++-sim plan and the
sim-parallelization note rely on (§5). It being legible and isolated is what makes
"swap the worker's simulation for a C++ core" a drop-in. Fused into a god-object
with the pool and the positional-tuple task contract, the seam is *true* (raw bytes
cross it) but not *legible* — a reader cannot see that no training/optimizer/registry
type crosses it without tracing the whole class. Adding a HOT knob to the task means
editing the parent tuple (`:368`) and the worker unpack (`:237`) in lockstep.

**Diagnostic symptoms.** An 11-field module global as the dependency-injection
mechanism; positional task tuples with no `TaskSpec` type; the version-bump cadence
coupled to the hot-knob refresh rate (`:204` — hot_search can only change when the
net version changes).

**The fix** (§3.6): split into `WorkerPool` (owns workers + pinning), `Transport`
(owns the raw-bytes weight publish / record collect — keep its good internals), and
a `TaskSpec` dataclass replacing the positional tuple. `ParallelExecutor` becomes a
thin composition. This makes the simulation seam *legible*, which is the §5 win.

### 2.6 The precision/forward-selection policy has no home (scattered dtype decisions)

**Where.** The "which precision, which forward implementation" decision is made in
four places independently:
- `dtypes.py` reads `CHOCO_AZ_DTYPE` and exports `DTYPE`/`is_float32()` (the one
  good part — a parametric constant).
- `mlp_jax.py:37` and `mlp_jax_train.py:54` each recompute the *identical*
  `_JDTYPE = jnp.float32 if np.dtype(DTYPE) == np.dtype(np.float32) else jnp.float64`.
- `mlp.py:280` branches on `is_float32()` to pick the f32-numpy vs f64 forward.
- `gumbel_search.py:117-125` decides whether to wire `MlpJaxForward` based on a
  `use_jax_mlp` constructor flag — i.e. the search picks the net's forward
  implementation.

**The wrong abstraction.** The net should own "which forward am I." The
jax-dtype derivation should have one home, not be copy-pasted between the two jax
modules. The search has no business knowing numpy-vs-jit exists.

**Why it hurts.** It is a low-severity but textbook *misplaced-decision* smell: a
policy (precision/forward) implemented at its consumers instead of at its owner. A
third forward (the design floats torch-CPU; the sim note floats a numba tree core)
adds another flag to the search constructor. The duplicated `_JDTYPE` can silently
diverge if one module's mapping is edited.

**The fix** (§3.4, folded with the net contract): a `jax_dtype()` helper in
`dtypes.py` (one derivation), and a `net.forward()` factory on `ValueMLP` that
returns the right `predict_both` for the net's dtype/heads — the search just calls
`net.forward()`, losing the `use_jax_mlp` flag and the `MlpJaxForward` wiring.

### 2.7 The bounds module bundles five V̂ strategies and reimplements `Environment`

**Where.**
- `info_relaxation.py:54-211` bundles five value-function strategies (`vhat_zero`,
  `vhat_analytic`, `DecompVhat`, `ExactBeliefVhat`, and the conceptual zero) in one
  module with the penalty machinery (`PenalizedClairvoyant`, `dual_bound_rate`).
  `DecompVhat` reaches into decomp internals via a lazy import (`:113`) precisely
  *because* the bundling would otherwise force every bounds user to import decomp.
- `minienv.py:69-116` reimplements `Environment`'s belief mechanics by hand —
  `marginals` (`:81-84`), `filter_treasure` (`:86-88`), `filter_detector`,
  `legal_actions` (`:97-104`), `apply` (`:106-115`) — each a near-copy of the
  `env.py` original (`env.py:99-135`), differing only in the restricted treasure
  set.

**The wrong abstraction.** Two distinct mistakes. First, **V̂ is a Strategy** — a
`V̂: (belief, λ) → value` interface with several implementations — but the
implementations are concrete functions/classes bundled with their one consumer, so
the lazy import is needed to break a cycle the bundling created. Second, `MiniEnv`
is a **second `Environment`** rather than a *restriction of the same one*: the
belief mechanics are identical bitwise operations on a smaller world-set, so the
right shape is a parameterized `Environment` (an optional `kept_treasures` filter)
or a true subclass with `super()` delegation — not a hand-maintained copy.

**Why it hurts.** The V̂ Strategy is the *exact* seam the calibration agenda needs:
the dual-bound's whole point (handoff §5) is to plug a trained `V̂_AZ` into
`eval_bound.py` and see if `λ̄` falls below 0.1454. A clean `V̂` Port makes
"`az-ckpt` is just another strategy" trivial; the bundling makes it a new branch in
a crowded module. `MiniEnv` duplicating belief mechanics means a fix to `apply` or
`filter_detector` must be applied twice and can silently diverge — and the
divergence would corrupt exactly the dual-bound validation that is supposed to be
the *trusted* check on the learner.

**Diagnostic symptoms.** A lazy import to break a self-inflicted cycle
(`info_relaxation.py:113`); five strategies one `import` away from each other;
`minienv.py` methods that are line-for-line `env.py` methods with a mask.

**The fix** (§3.7): a `Vhat` Protocol with implementations split by dependency
(`vhats.py` for zero/analytic, `vhats_decomp.py`, `vhats_exact.py`), so
`dual_bound_rate` imports only what it needs; and fold `MiniEnv` into `Environment`
as a `restrict(keep, k_local)` factory so there is one belief-mechanics
implementation.

### 2.8 Eval/entry-point sprawl: nine `main()`s, ten hardcoded reference constants

**Where.** `eval/` has nine top-level `main()` scripts (`eval_az`, `eval_decomp`,
`eval_ismcts`, `eval_nmcs`, `eval_uct`, `eval_faces`, `eval_bound`, `tb_runner`,
plus a dead `harness.main()`). The two genuinely-shared functions
(`realizable_static`, `clairvoyant_rate`, `harness.py:14-50`) *are* imported by all
— good. But everything around them is copy-pasted:
- the reference *constants* 0.0855 / 0.1454 / 0.094 are hardcoded in ten sites
  (`exit_loop.py:49-51`, `eval_az.py:34,79`, `dataset.py`, `info_relaxation.py`,
  `eval_bound.py`, …) — and **already disagree**: `exit_loop.py:51` says
  `DECOMP_ANCHOR = 0.0941` while `eval_az.py:79` writes `0.094`. That is the
  SSOT-violation-as-bug the brief warns about, live in the tree.
- `clairvoyant_rate` is implemented *twice* — once in `harness.py:28-50`, once in
  `eval_bound.py:52-77` — differing only by a `keep` subset.
- the shallow/search/NMCS **policy packs** are re-declared in `harness.main()`,
  `eval_decomp.py`, `eval_faces.py`, `eval_nmcs.py` with drifting budgets.
- the Dinkelbach budgets and the CLI args (`--seed`/`--n`/`--it`) are redeclared
  per script with inconsistent defaults and parsing styles.

**The wrong abstraction.** Each `eval_*` is a copy-pasted `main()` that should be a
*configuration* of one harness. The missing abstractions are: a `constants` module
(or `harness` module-level constants) for the reference rates; a parameterized
`clairvoyant_rate(env, keep=None)`; a policy *registry*; a shared eval-argument
factory; and one entry point with subcommands instead of nine scripts.

**Why it hurts.** The disagreeing 0.094/0.0941 is the concrete cost — a reader
cannot tell which is authoritative, and a plot's reference line silently depends on
which script drew it. Adding a policy or changing a budget is a 3-file edit.

**The fix** (§3.8): `eval/constants.py` (or `harness` constants) as the one home for
the reference rates; merge the duplicated clairvoyant into one parameterized
function; a `policy_registry.py` for the packs; an `add_eval_args(parser)` factory;
and consolidate the `eval_*` mains into one `run_evals` with subcommands.

### 2.9 Lower-leverage items (named for completeness, not sequenced first)

- **`facemodel.py` un-wired** (`env.py:35` comment-only). Either integrate (replace
  the inline `arrangement.load()` detector build, `env.py:43-46`, with
  `sense_actions()`) or move it to `docs/`. As source-tree code with no caller it
  is a maintenance trap (it *looks* live).
- **`analysis/` orphaned** (`analyzer.py`/`synthetic.py` used only by each other).
  Either wire the cluster discovery into the env (so decomp's hardcoded clusters
  come from the analyzer) or relocate to a `tools/` directory. Today the solver's
  cluster definitions and the analyzer's are two unrelated implementations of the
  same intent.
- **`mlp.py` f32-cache invariant is a comment, not a mechanism** (`mlp.py:86-97`):
  "INVARIANT over ALL writers" relies on every weight-setter rebinding (so the
  identity check invalidates). It holds today (the only writer, `JaxTrainer`,
  rebinds, `mlp_jax_train.py:238`) but is one careless in-place writer from a silent
  stale-cache bug. This is the >1-writer-on-a-slot shape; per the project's own
  discipline it wants an owner-method (`net.set_weights()` that invalidates), not a
  per-writer convention. Low urgency (one writer today), real if a second appears.
- **`kernels.py` bit-width contract implicit** (`kernels.py:26`): the int64 bitmask
  reduction silently assumes `N < 64`; true on the live env (N=20) but unguarded.

---

## 3. The target architecture

Top-down: the boundaries first, the instances second. The organizing principle is
the one clean seam the project already has (env↔Policy) extended to the rest — every
component below is something that can be *reasoned about and replaced
independently*, with the dependency arrows pointing inward toward pure logic.

### 3.1 The boundary inventory (the Ports/ACLs the codebase should have)

| boundary | separates | shape | today |
|---|---|---|---|
| **Simulation Port** | env mechanics ↔ everything | `Environment` + `Policy.decide` | **exists, clean** — the model |
| **Optimizer/Trainer split** | parameter update ↔ loss/data/write-back | two objects; hparams injected per step | conflated (§2.1) |
| **AZ-target module** | search engine ↔ target rule | pure fns in `value_target.py` | split across 2 files (§2.2) |
| **Feature/Action contract** | belief encoding ↔ its consumers | a `FeatureLayout` value object + net→env validator | re-encoded in 5 files (§2.3) |
| **Search-support substrate** | tree machinery ↔ each search | `step_value`/`candidates`/`RolloutExecutor`/`belief_key` | reimplemented per solver (§2.4) |
| **Transport / Pool / Task** | bytes-on-the-wire ↔ pool ↔ task contract | 3 objects + `TaskSpec` | one god-object (§2.5) |
| **Forward/precision policy** | "which forward" ↔ consumers | `net.forward()` factory + `dtypes.jax_dtype()` | scattered (§2.6) |
| **V̂ Strategy Port** | dual-bound penalty ↔ value-fn approx | `Vhat` Protocol, impls split by dep | bundled (§2.7) |
| **Restriction, not re-impl** | full env ↔ sub-instance | `Environment.restrict(keep,k)` | second env (§2.7) |
| **Eval harness vs config** | measurement ↔ what-to-measure | one harness + registry + constants | nine mains (§2.8) |

The two boundaries that carry the most research leverage are the **AZ-target
module** (§3.3) and the **V̂ Strategy Port** (§3.7) — both are *exactly* the seams
the calibration agenda will exercise, so getting them right is not cleanup, it is
unblocking the project's actual frontier.

### 3.2 Optimizer ⊥ Trainer (the folded-in worked instance)

The target is `training-optimization-refactor.md`'s split, restated as the shape
the boundary inventory implies. A small `Optimizer` (`az/optimizer.py`) owns the
optax transform and *nothing else*; its hyperparameters live in the optax state via
`optax.inject_hyperparams` and are supplied as a **required argument** to `step`
each call — so there is no `self.lr` to bake and no callable shape that skips
supplying one. The slimmed `Trainer` keeps the loss, data marshalling,
y-standardization read, and write-back, and *delegates* the update. `l2` becomes a
traced loss argument (it is a loss coefficient, not optimizer state). A
`adam_hparams_from(cfg)` adapter (the ACL translating the registry snapshot into the
optimizer's vocabulary) feeds the live values each iteration off the snapshot the
loop already refreshes (`exit_loop.py:317`).

Consequences: `lr`/`l2`/`betas`/`eps` flip from RESTART to HOT in the schema
(`schema.py:151-155`) because the consuming code now reads them live; the vestigial
`lr`/`l2` signature channel is deleted; the three-authority drift (§2.1) collapses to
one read-site per value. The genuinely-RESTART set (net shapes, env constants, `m`/
`n_sims`, `seed`) stays non-hot *by construction* — the Optimizer's moment pytree is
typed to the params it was built from, so a shape change is rejected loudly by jax,
and there is no `AdamHParams` slot to route a shape through. The full HOT-ness table,
the SSOT/MISU treatment, and the out-of-frame audit that corrected the
"unrepresentable" overclaim live in that note; this audit's contribution is to place
it as *one* boundary in the inventory, not the spine.

### 3.3 The AZ-target module (search emits stats; `value_target.py` owns the rule)

`value_target.py` already owns the return-to-go rules; it should own the *whole* AZ
target. Move `_v_mix` and `_improved_policy` out of `gumbel_search.py` and rewrite
them as pure functions of explicit inputs:

```
# value_target.py  (signatures are the contract)
def v_mix(root_value, visited_q, visited_n, prior, legal_slots) -> float: ...
def improved_policy(logits, visited_q, visited_n, root_value, prior,
                    legal_slots, c_visit, c_scale) -> np.ndarray:
    """Danihelka softmax(logit + σ(completed_q)); completed_q uses v_mix for
    unvisited actions. Pure: no node state, no tree. THE §4.4 policy target,
    reusable by any search and unit-testable in isolation."""
```

`GumbelAZSearch` collects `(visited_q, visited_n)` per root action and calls
`improved_policy`; `GumbelPolicy` becomes `argmax(improved_policy(...))`, a thin
adapter that no longer requires the engine to know it is a Policy. A PUCT-only
ablation or a future search reuses the rule by import, not by copy.

### 3.4 The feature/action contract as a value object + a net→env validator

A `FeatureLayout` built from the env owns the block structure once:

```
class FeatureLayout:
    """Owns the N×5 + nD×3 + global layout. Built from env. The ONE place the
    block offsets, the feature names, and the legal-index ranges live."""
    def offsets(self) -> dict: ...          # 'marg':(0,N), 'available':(2N,3N), …
    def names(self) -> list[str]: ...       # for feature_response.py
    def legal_index_ranges(self) -> dict: ...# 'available', 'informative' (for actions.py)
```

`features.py` builds from it; `actions.legal_mask_from_features` reads
`legal_index_ranges()`; `feature_response.feature_names` reads `names()`;
`gumbel_search.py:104` calls `actions.term_slot(env)` instead of re-deriving. Adding
a channel is a one-place edit. Alongside it, `validate_net_against_env(net, env)`
(asserts `net.in_dim == feature_dim(env)` and `net.n_actions == n_action_slots(env)`)
is called in `GumbelAZSearch.__init__` and `NetValueISMCTS.__init__` so a mismatch is
loud at construction.

The forward/precision policy (§2.6) folds in here: `dtypes.jax_dtype()` is the one
derivation `mlp_jax.py`/`mlp_jax_train.py` share, and `ValueMLP.forward()` is the
factory the search calls (dropping `use_jax_mlp` and the `MlpJaxForward` wiring out
of `gumbel_search.py`).

### 3.5 The search-support substrate (compose, don't reimplement)

A `solvers/rollout.py` (or extend `base.py`) holding the pieces every tree search
reaches for, leaving the `Policy` seam untouched:

```
def step_value(r, dt, lam) -> float: return r - lam * dt   # the objective term, one home
def candidate_actions(env, loc, bw, collected, n_det, n_tre): ...  # the pruning, one home
class RolloutExecutor:
    def __init__(self, base: Policy): self.base = base
    def value_to_end(self, env, loc, bw, collected, world, lam) -> float: ...  # was _base_value
BeliefKey = Protocol  # global (n,min,max) and per-cluster impls
```

`ismcts`/`nmcs`/`uct`/`decomp` compose these. The objective formula now lives in one
function, so a time-model recalibration touches one line. This is the lowest-risk
high-coverage refactor (pure extraction, behavior-identical) and is a good early
step (§4) because it de-risks the others by shrinking the surface.

### 3.6 Transport ⊥ Pool ⊥ Task

Split `ParallelExecutor` into `WorkerPool` (workers + core-pinning + the worker-init
that builds env/fb/net/search), `Transport` (the *good* raw-bytes weight-publish /
record-collect, lifted verbatim), and a `TaskSpec` dataclass replacing the positional
tuples. `ParallelExecutor` becomes `Transport` + `WorkerPool` composition. The win is
that the simulation seam — weights-bytes in, record-bytes out, scalar HOT knobs in —
becomes *legible*: a reader sees that no training/optimizer/registry type crosses it,
which is the §5 C++-readiness property made visible rather than merely true.

### 3.7 The V̂ Strategy Port + `Environment.restrict`

A `Vhat` Protocol (`(belief, λ) → value`) with implementations split by dependency:
`vhats.py` (zero, analytic — no heavy deps), `vhats_decomp.py` (imports decomp),
`vhats_exact.py` (the enumeration). `dual_bound_rate` imports only the strategy it is
given, dissolving the lazy import that exists to break the bundling cycle. The
`az-ckpt` strategy (a trained `V̂_AZ`) is then *just another impl* — the calibration
agenda's plug-in point, clean by construction. Separately, fold `MiniEnv` into
`Environment.restrict(keep, k_local)` so the belief mechanics have one
implementation; the dual-bound validation then exercises the *same* `apply`/`filter`
the learner does, which is what makes it a trustworthy check.

### 3.8 One eval harness, one constants home, one policy registry

`eval/constants.py` (or `harness` module constants) holds `STATIC_FLOOR = 0.0855`,
`CLAIRVOYANT_CEILING = 0.1454`, `DECOMP_RATE` — resolving the 0.094/0.0941
disagreement to one authoritative value. `clairvoyant_rate(env, keep=None)`
absorbs the `eval_bound` duplicate. `policy_registry.py` holds the packs.
`add_eval_args(parser)` is the shared CLI. The `eval_*` mains consolidate into one
`run_evals` with subcommands; `harness.main()` (dead) is deleted.

---

## 4. Sequenced refactor plan (by leverage; each step independently reviewable)

Ordering principle: **pure extractions and contracts first** (they de-risk
everything downstream and are behavior-identical, so the equivalence/AZ-loop tests
pin them at `max|ΔG|`-style fidelity), **effectful splits next**, **research-frontier
Ports last** (they are where new capability lands). Inter-step dependencies named.
Each step's verification reuses the standing immune system — `test_jax_equivalence.py`
(numpy↔jit forward), `test_az_loop.py`, `test_parallel_deadlock.py`,
`test_hp_registry.py`, `test_jax_equivalence.py` — plus a per-step assertion.

**Step 0 — baseline (no code change).** Run the full suite; capture a short fixed-λ₀
`exit_loop` smoke (per-iter CE/vMSE/R²) as the reference the behavior-preserving steps
must reproduce. *Verify:* green; smoke captured.

**Step 1 — the reference-constants SSOT (§3.8, smallest, highest clarity/risk ratio).**
Create `eval/constants.py`; replace the ten hardcoded literals; resolve 0.094 vs
0.0941 to one value (the maintainer picks; `eval_az.py:79`'s 0.094 and
`exit_loop.py:51`'s 0.0941 must agree). *Verify:* grep shows one definition;
reference-line plots unchanged at the chosen value. *Depends on:* nothing. *Why
first:* it is a live SSOT-bug, trivially reviewable, and zero behavioral risk.

**Step 2 — the search-support substrate (§3.5).** Extract `step_value`,
`candidate_actions`, `RolloutExecutor`, `belief_key` into `solvers/rollout.py`; have
the four tree searches compose them. Pure extraction. *Verify:* `test_az_loop` +
each solver's eval reproduce their recorded rates bit-for-bit (same operations, same
order); a regression assert that `RolloutExecutor.value_to_end` equals the old
`_base_value` on a fixed seed. *Depends on:* nothing. *De-risks:* §4 downstream by
shrinking the duplicated surface.

**Step 3 — the FeatureLayout contract + net→env validator (§3.4).** Introduce
`FeatureLayout`; route `features`/`actions`/`feature_response`/`gumbel_search.py:104`
through it; add `validate_net_against_env` to the two search constructors; add
`dtypes.jax_dtype()` and de-duplicate `_JDTYPE`. *Verify:* `test_jax_equivalence`
green (the vector is byte-identical — same offsets); a new test that a mismatched
`n_actions` raises loudly at `GumbelAZSearch` construction. *Depends on:* nothing;
*enables:* §4-Step-6 (the forward factory).

**Step 4 — the AZ-target module (§3.3).** Move `_v_mix`/`_improved_policy` to
`value_target.py` as pure functions; `GumbelAZSearch` calls them; `GumbelPolicy`
becomes the thin adapter. *Verify:* `test_az_loop` reproduces the improved-policy
targets bit-for-bit on a fixed seed (the Danihelka invariants in the suite —
`test_executed_action_is_sh_survivor` etc. — are the guard); a new unit test calls
`improved_policy(...)` in isolation. *Depends on:* nothing structurally, but lands
cleaner after Step 3 (shared `FeatureLayout`/slot helpers).

**Step 5 — the Optimizer ⊥ Trainer split (§3.2; the folded-in arc).** Execute the
`training-optimization-refactor.md` plan: extract `Optimizer` (plain optax first,
then `inject_hyperparams`), move `l2` to a traced loss arg, delete the dead `lr`/`l2`
channel, wire `adam_hparams_from`, flip the schema facets to HOT, add the
facet-consistency + write-site tests. Migrate `train_value.py`'s Stage-1 gate onto the
same `Optimizer`/`AdamHParams` contract. *Verify:* steps 1–3 of *that* note are
bit-reproducible against Step-0's smoke; the behavioral steps gated by an integration
test that `set train.lr 1e-4` lands live. *Depends on:* nothing structurally; sequence
it here so it lands on a tree already cleaned of the feature/target debt (smaller
review surface).

**Step 6 — the forward/precision factory (§2.6/§3.4 tail).** Add `ValueMLP.forward()`;
the search calls it; drop `use_jax_mlp` and the `MlpJaxForward` wiring out of
`gumbel_search.py`. *Verify:* `test_jax_equivalence` green; bench parity (the search
picks the same forward it did via the flag). *Depends on:* Step 3 (`jax_dtype`).

**Step 7 — Transport ⊥ Pool ⊥ Task (§3.6).** Split `ParallelExecutor`; introduce
`TaskSpec`; lift the raw-bytes transport verbatim. *Verify:* `test_parallel_deadlock`
green; a parallel≈serial check (the seed-fold determinism, `az-parallel-exp.md`)
reproduces bit-identical aggregate transitions workers=1 vs 4. *Depends on:* nothing,
but is the highest-effort effectful split — sequence after the pure ones so the diff
is isolated. *Unblocks:* the §5 C++ seam legibility.

**Step 8 — the V̂ Strategy Port + `Environment.restrict` (§3.7).** Split the V̂ impls
by dependency; fold `MiniEnv` into `Environment.restrict`. *Verify:* `eval_bound
--validate` reproduces the recorded sub-instance regression/tightness numbers (the
dual-bound's own check); the restricted env's `apply`/`filter` match `MiniEnv`'s on a
fixed belief. *Depends on:* nothing; lands the calibration plug-in point.

**Step 9 — eval harness consolidation (§3.8 tail) + the facemodel/analysis disposition
(§2.9).** Merge the `eval_*` mains into `run_evals` subcommands; `policy_registry`;
`add_eval_args`; delete `harness.main()`. Decide facemodel (wire or relocate) and
`analysis/` (wire cluster discovery or relocate to `tools/`) — these are
*dispositions*, surfaced for the maintainer, not auto-applied. *Verify:* each
subcommand reproduces its old script's output; the disposition is a maintainer call.

The arc front-loads clarity (Step 1) and pure extractions (Steps 2–4), puts the named
worked instance in the middle on a cleaned tree (Step 5), and ends with the
effectful split and the research Ports (Steps 7–8) where new capability lands. Any
step can ship alone; none requires a long training run to verify (smoke + unit tests
settle each, per the project's `max|ΔG|` discipline).

---

## 5. C++-sim seam — the property the target architecture has

The brief asks this be *verified as a property the boundaries already give*, not
treated as added scope. It composes with `simulation-parallelization-viability.md`
(which ranks: #1 widen the exact cross-episode fan-out, #2 a compiled/columnar tree
core *conditionally for latency*, #3 GPU leaf-batching is gold-plating) and does not
contradict it — every change in this audit is on the *learner/orchestration* side,
which that note holds outside the simulation hot path.

**The seam is `Environment` + `Policy`, and it is already clean (§0).** The surface a
C++ core reimplements is exactly: `apply(loc, bw, collected, action, world)`
(`env.py:125`), `filter_treasure`/`filter_detector`/`marginals` (`env.py:99-135`),
`exit_cost`/`d` (the static memo, `env.py:73-85`), and `simulate(policy, world, lam,
rng, max_steps)` (`env.py:138`). Every one is a pure function of (state, action,
world) → numbers. **No training, optimizer, registry, feature, or target type appears
in any of these signatures, and nothing in this audit's refactors adds one.**

**The target architecture *strengthens* this property in three ways:**
1. **The transport split (§3.6) makes the seam legible.** Today the seam is true
   (raw-bytes weights in `parallel.py:92`, raw-bytes records out `:267`, scalar HOT
   knobs in `:201`) but buried in a god-object. After the split, the learner side is
   visibly `Transport` + `WorkerPool` + the `Optimizer`/`Trainer`/snapshot-adapter,
   and the data crossing the seam is *manifestly* (weights-bytes in, record-bytes out,
   scalar knobs in). A C++ core that produces the same record bytes from the same
   weights+scalars is a drop-in; nothing on the learner side changes a line.
2. **The Optimizer/Trainer split (§3.2) removes a false signal.** The vestigial
   `lr`/`l2` channel through `train_epochs` (§2.1) currently makes it *look* like
   optimizer hyperparameters thread through the per-step path the simulation feeds.
   Deleting it makes "the simulation→records→train pipeline carries no optimizer
   state" legible from the code.
3. **`Environment.restrict` (§3.7) means one belief-mechanics implementation.** A C++
   core reimplements *one* `apply`/`filter`, not two (env + the hand-copied `MiniEnv`),
   so there is no second implementation to keep bit-compatible.

**The scalar interface is already language-agnostic.** The search's HOT knobs cross
the process boundary as a flat `hot_search` dict of scalars (`parallel.py:201`) and
`max_steps`/`lam` as scalars — a key→number map, the canonical FFI-friendly shape,
already serialized pickle-free precisely so it does not depend on Python object
semantics. A `TaskSpec` dataclass (§3.6) only makes that contract explicit; it
serializes to the same flat scalars.

**Verdict.** The target architecture has the C++-swappability property, and the
refactor *increases* it — the boundary is unchanged (the env↔Policy seam was already
right), and the work makes it *legible and singular* rather than buried and
duplicated. This is the sim-parallelization note's #2 (compiled tree core) dropping
in behind a clean Port, at no cost to its #1 (cross-episode fan-out).

---

## 6. Honest caveats, costs, and coverage depth

**Per-area coverage (where this is thorough vs a first pass):**
- *Thorough* (read end-to-end, file:line evidence, fix designed): the `az/` learner
  core (`mlp_jax_train`, `exit_loop`, `gumbel_search`, `features`, `actions`,
  `value_target`, `mlp`, `mlp_jax`, `dtypes`, `parallel`), `model/env`, `config`,
  `hp/{schema,registry}`. §2.1–§2.6 rest on full reads.
- *Solid but one pass* (read in full, but the internal correctness of the algorithms
  was not re-derived): the solvers (`base`, `ismcts`, `nmcs`, `uct`, `decomp`) and
  the bounds (`info_relaxation`, `eval_bound`, `minienv`). §2.4/§2.7 identify the
  *structural* debt confidently; I did **not** audit whether e.g. decomp's micro VI
  or the dual-bound penalty is numerically correct — that is the design notes' job
  and out of this audit's frame.
- *First pass* (read for role and coupling, not line-audited): `eval/*`,
  `feature_response`, `tb_runner`, `analysis/*`, `model/{facemodel,arrangement}`,
  `dataset`, `kernels`, `netvalue_ismcts`. The §2.8/§2.9 claims (duplication,
  orphaning, un-wiring) are coupling facts I verified by grep + targeted reads, not
  deep line audits.

**Costs and risks, named:**
- **Steps 2–4, 6 are bit-reproducible *by argument*, not by certified diff.** The
  "behavior-identical" claim for the pure extractions is an argument from the
  operations being unchanged; the per-step equivalence/AZ-loop verification is what
  *certifies* it. This note structures the steps so exactness is preservable; it does
  not pre-certify any implementation.
- **Step 5 (the Optimizer split) is surgery on a numerically load-bearing file**
  (`mlp_jax_train.py`, pinned by the equivalence test). The folded-in note's
  step-by-step bit-reproduction is load-bearing, not ceremony — and it carries its own
  honest caveats (the `l2==0` short-circuit cost, live-`lr`×Adam-moment interaction,
  `inject_hyperparams` state-shape change). Those stand.
- **Step 7 (transport split) is the highest-effort, highest-risk effectful change.**
  The parallel path has a deadlock history (`jaxtrain-deadlock-rca.md`); the split
  must preserve the bounded-drain / loud-RuntimeError discipline and the seed-fold
  determinism. The verification (parallel≈serial bit-identity) is the guard, but this
  is real surgery on the one concurrent component.
- **The eval consolidation (Step 9) risks subtle output drift** if a subcommand's
  default differs from its old script's. The mitigation is reproducing each old
  script's output, but the nine scripts have *drifting* defaults today (that is the
  debt), so "reproduce the old output" means picking which old default is canonical —
  a maintainer call, not mechanical.
- **The facemodel and `analysis/` dispositions are judgment calls, not refactors.**
  Whether to wire facemodel into the env (a behavior change — the detector model
  switch is consult-002's whole point) or relocate it, and likewise for the analyzer,
  is the maintainer's to decide; this note surfaces them as debt and stops short of
  prescribing.
- **The standing project caveat applies:** everything is conditioned on the single
  instance and the uncalibrated symmetric-Euclidean time model. The architecture work
  is orthogonal to that model-fidelity question, but the `step_value` consolidation
  (§3.5) is *motivated* partly by anticipating a time-model recalibration — which may
  or may not happen.
- **This is design only.** No code was run, no job launched; every claim is from
  reading the tree on `feat/hp-registry` (the worktree branch base). The line numbers
  are from that state and should be re-resolved if the tree moves before
  implementation.

**What I am *not* claiming.** Not that the codebase is badly built — the env↔Policy
seam, the hp registry, and `config.py` are genuinely well-shaped, and several
"problems" (the `Policy` ABC, the residual ablation axis, the f32-cache *mechanism*)
are correct and should be left alone. The thesis is narrower and load-bearing: the
project has *one* clean seam and should extend its standard to the half-dozen
boundaries that are currently conflated, buried, or duplicated — and doing so unblocks
the calibration frontier (the AZ-target module and the V̂ Port are the seams that
agenda will exercise) as a *consequence*, not a side quest.

---

## Appendix A — commission prompt (verbatim)

> Recorded verbatim per the consult-record discipline
> (`docs/consults/consult-001-prompt.md` is the format reference).

---

You are a **refactor auditor for the ENTIRE chocofarm project** (`/home/bork/w/vdc/chocobo`, github KodBena/chocofarm) — an Operations Research exercise (a belief-MDP / adaptive stochastic orienteering problem). Codebase posture: **fail-loudly (ADR-0002)**. Public Domain (Unlicense). The maintainer prefers honest, mechanistic "this costs X, buys Y" over optimism, and **disciplined, abstraction-centered analysis over ceremony/buzzwords**.

This is a **whole-project architectural refactoring audit**. The deliverable is ONE implementation-ready **design note**. You do NOT implement anything, do not modify source, do not run code or any job.

## Scope — the WHOLE project. Do not narrow this.
The maintainer's thesis: **abstractions are what keep software maintainable**, and this project has accumulated **architectural debt all over the place** — conflated responsibilities, missing or leaky boundaries, wrong or absent abstractions, god-objects, hidden coupling. Your job is to audit the **entire codebase** for this and design the refactoring that fixes it.

**One known instance** of the debt: training and optimization are conflated in `JaxTrainer` (optimizer hyperparameters captured once at construction, so fields that ought to be live can't be). **That is ONE example, not the scope.** There is almost certainly architectural debt throughout — the model/env layer, the search, the solvers, the bounds machinery, the eval harness, the parallel transport, the registry, the package boundaries, the entry points. **Find it across the whole project.** Do NOT frame the audit around hyperparameters, or training/optimization, or any single subsystem — that framing is the failure mode this commission exists to avoid. The hyperparameter case is one worked entry among many.

## The lens — abstractions first
Lead every finding with **"what is the right abstraction / boundary here, and how does the current code depart from it?"** The maintainability lever is correct abstractions: clean single-responsibility components, the right Ports / ACLs / seams, parts that can be reasoned about and replaced independently.

**SSOT / DRY / MISU are diagnostics, NOT goals.** A DRY/SSOT violation or a representable-illegal-state is a *symptom that locates a misplaced abstraction* — use them to find and explain debt, but do NOT headline the audit with them, and do NOT reduce it to a checklist of acronyms. Disciplined structural reasoning, not band-aid bookkeeping.

## A robustness property to design for (not added scope)
The maintainer plans to eventually reimplement the **simulation** (the rollout / search engine) in **C++**. A correctly-bounded simulation component makes that a drop-in. So a good target architecture has a clean, language-agnostic seam around the simulation — treat C++-swappability as a property the right boundaries already give you, and verify your target architecture has it. (`docs/sim-parallelization-viability.md` on branch `docs/sim-parallelization-viability` characterizes the sim hot path — read via `git show docs/sim-parallelization-viability:docs/design/simulation-parallelization-viability.md` — compose, don't contradict.)

## Survey — the whole tree (an ARCHITECTURAL read: responsibilities, boundaries, coupling — not every line)
Walk and map the entire `chocofarm/` package:
- `chocofarm/config.py`; `chocofarm/model/{env,facemodel,arrangement}.py`; `chocofarm/analysis/{analyzer,synthetic}.py`
- `chocofarm/az/` (largest surface): `exit_loop.py`, `gumbel_search.py`, `mlp.py`/`mlp_jax.py`/`mlp_jax_train.py`, `features.py`, `value_target.py`, `dataset.py`, `parallel.py`, `actions.py`, `kernels.py`, `dtypes.py`, `feature_response.py`, `netvalue_ismcts.py`
- `chocofarm/solvers/{base,decomp,ismcts,nmcs,uct}.py`
- `chocofarm/bounds/{eval_bound,info_relaxation,minienv}.py`
- `chocofarm/eval/{harness,eval_*,tb_runner}.py`
- `chocofarm/hp/{schema,registry}.py`
Your worktree is on `feat/hp-registry` (the latest state, including the in-flight hp registry + `config.py`), so all of the above are present as files. Orientation: `docs/handoff-2026-06-15.md`, `docs/STATUS.md`, `docs/design/alphazero-surrogate-design.md`, and existing design notes (`dual-bound.md`, `static-*.md`); read any project-level architecture/framework doc if present. The narrow `docs/design/training-optimization-refactor.md` (branch `docs/training-optimization-refactor`, read via `git show`) is **one worked instance** — fold its substance in as the optimizer/training entry; the general audit likely supersedes it.

## Deliverable — `docs/design/architecture-refactor-audit.md` (implementation-ready)
House style of `docs/design/*.md`. Disciplined and mechanistic — no ceremony, no buzzword theatre. Include:
1. **An honest architecture map** — the actual modules, their responsibilities, and the dependency/coupling structure as it IS, with file-level evidence.
2. **The architectural problems found across the whole project** — each: where (file/line), what abstraction is wrong / missing / conflated, why it hurts maintainability or extension, and the diagnostic symptoms. Be comprehensive across the codebase; the optimizer/training conflation is one entry, not the spine.
3. **The target architecture** — the right abstractions / boundaries (Ports / ACLs / module splits), described concretely (responsibilities + interfaces), from which the specific refactors fall out. Top-down: the structure first, the instances second.
4. **A prioritized, sequenced refactor plan** — ordered by architectural leverage, each step independently reviewable with its verification, inter-step dependencies named.
5. **C++-sim seam** — verify the target architecture makes the simulation cleanly swappable.
6. **Honest caveats** — cost, risk, what's uncertain, and your coverage depth per area (where it's a first pass vs thorough).

Prioritize ruthlessly by architectural leverage — comprehensive does not mean flat-and-exhaustive; lead with the highest-leverage structural problems and state coverage honestly.

## Constraints
- Design/analysis ONLY. No code changes, no running anything, no side effects.
- Work in the prepared worktree **`/home/bork/w/vdc/chocobo-audit`** (branch `docs/architecture-refactor-audit`). Do not create another worktree or touch the others.
- Commit on `docs/architecture-refactor-audit` with **EXPLICIT PATH ONLY** (`git add docs/design/architecture-refactor-audit.md`; never `git add -A`/`.`). Commit message ends with exactly: `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`
- Do **NOT** push (the orchestrator pushes after review).
- Append this entire commission prompt verbatim as "Appendix A — commission prompt" (consult-record discipline; `docs/consults/consult-001-prompt.md` is the format reference).
- Honest, mechanistic, disciplined; where uncertain or shallow, say so plainly.

## Final message
Render the audit's substance self-containedly: the architecture map, the prioritized architectural problems found across the project (the optimizer/training case as one among them), the target architecture, the sequenced refactor plan, the C++-seam check, and honest coverage caveats. Report branch, commit SHA, file path. Not a pointer to the file — the substance.
