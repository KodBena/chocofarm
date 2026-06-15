# ADR-0012: Compositional and Structural Hygiene

- **Status:** Proposed
- **Genre:** Tenet (cross-cutting structural-design discipline) — the ninth
  tenet, and the structural counterpart to the *authoring*-discipline family
  (ADR-0002/0005/0007/0009) and the *corrective*-discipline tenet (ADR-0011).
  Where ADR-0011 says *a recurrence converts to a mechanism*, this tenet says
  *new structure is born in the shape the audit's mechanisms enforce* — so the
  conversion ADR-0011 mandates is rarely needed, because the rot never forms.
  It is the **positive inverse of the 2026-06-15 architectural audit's
  "architectural cancer" taxonomy**: each disease the audit named gets the
  structural rule whose presence makes that disease impossible to author.
- **Date:** 2026-06-15
- **Provenance:** Native to chocofarm, not transferred. Its source substrate is
  the 2026-06-15 architectural audit (`docs/notes/audit/architectural-audit-2026-06-15.md`)
  and the forward-looking seam design (`docs/design/scaling-and-cpp-seam.md`).
  The audit's §1 verdict — *"the bones are sound; the connective tissue is
  rotting … the right idea applied once and not propagated"* — is this tenet's
  reason to exist: the disciplines were known (the env↔Policy seam, live λ,
  derived dimensions) and proven, but were not the **default shape new code is
  born in**. This ADR makes them the default. It is written now, ahead of the
  incoming C++ runner and the future async actor-learner loop, precisely so
  that those — the next large bodies of new code — are born clean rather than
  audited dirty.
- **Scope:** All **new** structure across the `chocofarm/` package and any
  new-language component that joins it (the incoming C++ search/sim runner
  first; a future async actor-learner second). It binds at design and
  authoring time. Per ADR-0004's incremental-retrofit posture it mandates **no
  retroactive sweep** of existing code; the audit's R-series roadmap (not this
  ADR) sequences the cleanup of what already exists.

## Context

The 2026-06-15 architectural audit diagnosed eight recurring "architectural
cancers" (§2, anti-patterns A–H), verified line-by-line against
`main@cfce276`, and named the remediation as *"overwhelmingly subtraction and
relocation … the codebase finishing a sentence it started correctly."* The
deepest finding (§1, §14) is that **chocofarm already proved it knows the right
answer** — λ is threaded as a live per-call cell to ~100 sites, owned by one
fixed-point loop; `feature_dim(env)` and `n_action_slots(env)` are derived
from the instance with zero drift; the env↔Policy inversion of control is
honored to the letter — *and then applied that discipline once and stopped.*
The cancers are not wrong ideas; they are the **right idea not propagated.**

This tenet's job is propagation by default. It states, as **checkable rules**,
the compositional and structural hygiene the audit's R-series enforces, so a
contributor (human or LLM) authoring new code can self-check against a closed
list rather than rediscovering each lesson. It is deliberately **anti-pattern-
first**: the cancer is the load-bearing motivation, so each rule is anchored to
the specific disease its absence permits.

This tenet **composes with — and does not restate —** its siblings, which own
adjacent concerns:

- **ADR-0002 (fail loudly)** owns *error/diagnosis surfacing*. Principle 5
  below cites it; it does not re-derive the loudness hierarchy.
- **ADR-0004 (minimal-touch)** owns *editing under partial visibility*. This
  tenet's no-retroactive-sweep scoping defers to it.
- **ADR-0005 (documentation discipline)** owns *how facts are documented*.
  Principle 1's SSOT is the **structural** twin of ADR-0005 Rule 1's
  single-source-of-truth-per-handle (documentation register); they cite each
  other, neither restates the other.
- **ADR-0007 (file size / information density)** owns *file budgets*. Principle
  3 (no god-objects) produces small files as a byproduct but is justified on
  one-owner grounds, not line count; the budget is ADR-0007's.
- **ADR-0009 (perf/equivalence investigation discipline)** owns *substantiating
  perf and equivalence claims*. Principle 6 composes with it directly and
  imports its two-tier (bit-exact vs aggregate-behavioral) bar wholesale rather
  than redefining it.
- **ADR-0011 (mechanization discipline)** owns *converting a recurrence to a
  mechanism*. This tenet is upstream of it: structure born clean is structure
  ADR-0011 never has to convert. The mechanisms ADR-0011 mints (`FeatureLayout`,
  `BeliefRefs`, the equivalence tests) are this tenet's worked examples.

## Decision

We adopt **Compositional and Structural Hygiene** as a codebase-wide tenet for
new structure. It is stated in two registers: first **the anti-pattern
checklist** (each audit cancer → the rule that prevents it — the index a
contributor scans before authoring), then **the seven principles** (each a
checkable rule, with a worked example from this codebase and the cancer it
prevents), then a **dedicated concrete section for a new-language (C++)
component**.

### The anti-pattern checklist (cancer → preventing rule)

This is the audit's §2 disposition table, inverted: read it before authoring
new structure, and again at review. Each row is "if your new code can exhibit
this shape, the named principle forbids it."

| Audit cancer (§2) | The shape to never author | Preventing rule |
| — | — | — |
| **A** — Config frozen at construction; ownership lives nowhere | a tunable swept across a run captured once in `__init__`/`Namespace` with no per-call or per-iteration read | **P4** (live, not frozen) — heat is decided by *where the value lives*; a value that changes within a run is a live cell, not a ctor invariant |
| **B** — SSOT dissolved; same knowledge re-encoded in N places | a second hand-maintained copy of a fact (belief math, the C(N,K) prior, the feature layout, K, the reference rates) | **P1** (single source of truth / derive-don't-duplicate) — every fact has one home; derived quantities are computed, never re-typed |
| **C** — Hidden global state keyed by object identity | a module-global cache keyed on `id(env)` (or any value-less identity) instead of owned on the object | **P2** (seam/port discipline) — derived data lives on the object whose lifetime it shares; no module global keyed by address |
| **D** — Copy-paste programs instead of one parameterized runner | the Nth bespoke `main()`/driver differing only in one literal | **P3** (no god-objects → one parameterized collaborator) + **P1** (one definition of the metric) |
| **E** — Abstraction built then abandoned beside a live inline copy | a fully-built type sitting unused next to the hand-inlined path that is actually live; a parameter the receiver ignores | **P5** (remove the root cause) — adopt or delete; **P2** — a parameter the receiver cannot honor is not in the signature |
| **F** — Magic constants strewn as bare literals | a shared invariant (the episode horizon, UCB `c`, a λ-tolerance) typed at each use site | **P1** — one owner, referenced; not re-typed and trusted to agree |
| **G** — Load-bearing knowledge offloaded to unenforceable prose | a convention that lives only in a comment/doc the code cannot check or that does not resolve | **P5** + **ADR-0011** — encode in code or a real registry; cite the derivation, not volatile prose (ADR-0011 owns the mechanization) |
| **H** — Defensive band-aids stacked against a hostile substrate | a new mitigation layered on an un-diagnosed cause; a reliability strategy that *is* a stack of patches | **P5** (fail loud; remove the root cause) — distinguish a justified guard from a band-aid masking an undiagnosed cause |
| **(new, cross-language)** — a second encoder / shared types across the language boundary | a C++ component sharing Python types, or re-encoding the wire format independently | **P7** (cross-language wire discipline) — the redis raw-bytes protocol is the *only* contract; never a second encoder, never shared types |

### The seven principles

#### P1 — Single source of truth / derive-don't-duplicate

**Rule (checkable).** Every fact has exactly **one** home. A *derived*
quantity — a dimension, a layout, a count, the feature/weight layout, the
"keep" set of a sub-instance, a reference rate — is **computed from its source
at the point of use (or cached on the object that owns the source)**, never
hand-copied as a literal or re-encoded in a second place. The check: *grep the
tree for the value; if it appears as an independent literal in two places that
must agree, the rule is violated.*

**Worked example (this codebase).** `feature_dim(env)` and
`n_action_slots(env)` are derived from the instance with **zero drift** — the
audit's praise (§1, §6 "Seams to preserve"). The mechanism `FeatureLayout`
(`az/features.py`, ADR-0011's worked proof) is the SSOT made structural: one
ordered block table the three former writers (`features.py`, `actions.py`,
`feature_response.py`) now read **by name**, with a fail-loud contiguous-
partition check. `BeliefRefs(env)` (audit R3) is the same move for the three
reference rates: computed once from `harness.realizable_static`/
`clairvoyant_rate`, imported everywhere. `WeightContainer` (audit item J) owns
the weight layout once.

**Cancer prevented: B (SSOT dissolved), and F (magic constants).** The audit
proved the fuse is already lit: `DECOMP_ANCHOR=0.0941` (`exit_loop.py:51`) had
already drifted from `0.094` (`eval_az.py`, and `eval_bound.py:173` where it is
a *numerical input to a provable bound*). The sharpest landmine — the three-
writer feature layout, one writer untested — would *silently mislabel feature-
importance rows* on a reorder. This rule is the structural form of ADR-0005
Rule 1 (single-source-of-truth-per-handle, documentation register); they are
twins, not duplicates.

#### P2 — Seam / port discipline (dependency inversion)

**Rule (checkable).** A boundary between two concerns is an **explicit port
with its dependency injected**, not an import-coupling or a reach into the
other side's internals. The template is the env↔Policy seam: **a new capability
is a new `Policy` subclass with ZERO core edits.** A Port/ACL boundary
**translates-and-validates** — it decodes the foreign representation into the
native one and rejects what it cannot honor; it does **not** coerce a
malformed input into a plausible one (the hp registry's strict decode is the
exemplar). The checks: *(a) does a new method/capability require editing the
core, or only adding a subclass/impl behind the seam? (b) does the boundary
reject what it cannot honor, or silently accept it? (c) is any derived state
owned on the object whose lifetime it shares, or on a module global keyed by
identity?*

**Worked example (this codebase).** `env.py` imports no solver; `Policy.decide(
env, loc, bw, collected, lam, rng)` is the injected contract; adding a solver
is a new subclass (`env.py:8-10`, `base.py:16-19` — the single hardest decision
in the system, made right). The dual bound's injected-callable `V̂` seam lets a
trained AZ net or a decomp decision-value serve interchangeably (audit §3.7).
The hp registry's strict decode translates-and-validates rather than coercing
(refuses a RESTART-field change mid-run, naming both values — ADR-0002). The R9
remediation re-keys the slot-table cache from `id(env)` to a
`WeakKeyDictionary` keyed on the **env object** (`actions.py:67`,
`slot_action_tables`), tying each cached bijection to the env's lifetime rather
than its CPython address — a recorded deviation from the audit's literal
"`env.slot_tables` attribute" (an env attribute would force a
features→env→features import cycle; the WeakKeyDictionary achieves R9's intent
without it). It does not own the table on the env, and the module-level cache
intentionally persists; what changes is the **key** (object-identity, GC-safe)
not the storage location.

**Cancer prevented: C (hidden global state keyed by identity) and the leaky-
boundary half of E.** The pre-R9 `_SLOT_TABLES[id(env)]` cache was
keyed on the *least value-stable key possible* — masked today only because
every env is layout-identical, it would hand back the **wrong bijection with no
error** the moment two envs differ in N (and leak one never-evicted entry per
env). A parameter the receiver cannot honor
(`train_epochs(lr, l2)` ignored; `build(marg)` ignored; `restrict_faces` gates
`pass`) is a *lying signature* — P2 forbids it: **a parameter the receiver
cannot honor is not in the signature.**

#### P3 — No god-objects

**Rule (checkable).** Orthogonal concerns are split into **one-owner
collaborators**, each owning exactly one axis of the problem. The check: *can
you name, in one clause, the single concern this object owns? If naming its
responsibility requires "and," it is two collaborators wearing one class.* This
produces small files, but the justification is single-ownership, not the line
budget (that is ADR-0007's).

**Worked example (this codebase).** The audit's item K — the **Transport ⊥ Pool
⊥ Task** split — is the worked target: `transport.py` owns *everything about
how bytes travel over redis and nothing about the process pool (worker_pool.py)
or what one worker computes (worker.py)* (its own header). `WeightContainer`
(item J) owns the weight layout, split out of the transport's former second
encoder. The optimizer split (item M) separates the precision-agnostic forward
(`ForwardSpec`) from the JAX/optax trainer. The 26-flag argparse `Namespace`
threaded as `args.*` (audit §3.5) is the god-object the `RunConfig` nested
dataclasses (R12) dissolve.

**Cancer prevented: D (copy-paste programs) and the split-brain-encoder half of
B.** A god-object forces every consumer to re-thread its whole state, which is
why the same orchestration was re-typed across eight eval `main()`s and the
weight layout was *split-brained* between `ValueMLP` and `JaxTrainer`. One
parameterized collaborator (`eval/report.run_plan`, a `SOLVERS` registry,
`WeightContainer`) replaces N copies.

#### P4 — Live, not frozen, where it should breathe

**Rule (checkable).** A value that is **tuned mid-run or swept across runs** is
**read at the point of use from the live source**, not baked at construction.
A value's *heat is decided by where it lives, not by intentions* (audit L1):
a knob assigned to `self.X` in `__init__` is cold no matter how often you mean
to sweep it; the same knob arriving as a per-call argument or read from a live
registry is hot for free. The check — the audit's litmus test: *if the value
changes during a run or across a sweep, it is a live cell, not a constructor
invariant.* Apply the hp registry's facet discipline: classify each tunable as
**HOT** (read per-use, e.g. per-iteration), **RESTART** (changed only across a
restart, with a loud drift refusal mid-run — ADR-0002), or **INSTANCE** (a true
Tier-1 geometry invariant), and place it accordingly. Bake only the INSTANCE
facet; never bake what is HOT.

**Worked example (this codebase).** λ is the gold standard — owned by one
fixed-point loop, threaded as a live per-call argument to ~100 sites;
`DecompPolicy` even rebuilds its per-λ tables when λ moves (`base.py:18`,
`env.py:141/159-165`, `decomp.py:546`). The remediation extends exactly this:
live `lr`/`l2` via `optax.inject_hyperparams` (audit R13) to unblock the queued
LR-anneal — *which today must kill the process and `--resume`* because
`optax.adam(learning_rate=self.lr)` bakes the rate into the jit'd update closure
at construction (`mlp_jax_train.py:215`). The hp registry's HOT-per-iteration
snapshot of `n_step`/`td_lambda` with a loud RESTART-drift refusal is the facet
discipline in the tree today.

**Cancer prevented: A (config frozen at construction; ownership lives
nowhere).** The audit's verdict: *"of the project's experimentation levers,
exactly one — λ — is live. Every other dial is welded shut."* The frozen-at-
construction failure is **biting the project in production, on its own
roadmap** — the LR-anneal experiment cannot run without a process restart.

#### P5 — Fail loud; remove the root cause, never band-aid

**Rule (checkable).** A stall or error surfaces as a **loud, diagnosable
failure** (this defers wholesale to **ADR-0002** for the loudness hierarchy —
construction-time raise > test/CI failure > runtime exception > logged
diagnostic > silent fallback-only-when-genuinely-right). And: when a defect's
**root cause** is found, you **remove the cause**, not add another mitigation.
The check distinguishing a *guard* from a *band-aid*: **a justified defensive
guard is re-justified on orthogonal merit and kept; a band-aid masks an
un-diagnosed cause and is one of a growing stack.** Ask: *is this layer fixing a
symptom of the previous layer's fight, and would the whole stack disappear if
the substrate conflict were removed at the root?* If yes, it is a band-aid;
remove the root instead.

**Worked example (this codebase).** Audit R14 removed **JAX-from-the-child** —
the deadlock *root cause* (a tight compiled inner loop sharing a process with
XLA's thread pool) — by giving workers a numpy-only entrypoint, rather than
adding an eighth mitigation to the seven stacked deadlock band-aids
(`parallel.py`'s per-result timeouts, bounded socket timeouts, TTL leak-bounds,
faulthandler+SIGUSR1, the native-thread env-var `setdefault`, the core-pin
process-name scrape). Contrast the **kept** guards: `transport.py`'s bounded
socket timeout is **re-justified on orthogonal merit** — "loopback redis under
no memory pressure never trips 60s, so this is a safety net, not a happy-path
behavior change" (its own docstring) — a guard that turns a stall into a loud
`redis.TimeoutError`, kept because it is sound, not because it patches an
undiagnosed cause. The audit's L8: *"when the reliability strategy becomes a
stack of patches, the substrate is the bug."*

**Cancer prevented: H (defensive band-aids stacked against a hostile
substrate) and the silent-fallback half of A/G.** A subsystem whose correctness
test can only assert "fails loud" rather than "works" is fragile by
construction; the fix is to remove the substrate conflict so the bands become
unnecessary.

#### P6 — Substantiate equivalence/perf claims (composes with ADR-0009)

**Rule (checkable).** A perf, regression, null-result, or equivalence claim is
honest only with its substantiation attached — this **composes with ADR-0009**
and imports its **two-tier bar wholesale**, it does not restate it. The
ML-specific calibration this tenet underlines, because the C++ parity work
(P7) rests on it: **behavioral float32-equivalence is the bar, NOT byte-
identity** — float32 is not associative, so a reordered or cross-language
reimplementation of the same math *will* move the float and may flip a
near-tied argmax / Sequential-Halving choice. The check: *(a) is the quantity a
logic invariant (illegal-slot mass, a legality mask) → assert bit-exactly
(`== 0.0`); (b) is it a float-sensitive numeric (a rate under float32+numba,
or a cross-language forward) → hold to aggregate behavioral equivalence
(statistically indistinguishable rate / E[T] / action distribution over N≥300
episodes, ≥2 seeds, within Monte-Carlo CI); (c) claim bit-identity ONLY where
it is free and proven* (the three bit-exactness contracts the audit names: the
distance memo, the `ABS_TOL=1e-4` forward equivalence test, the value-target
MC-limit identity).

**Worked example (this codebase).** `bench_equivalence.py` holds the float32 +
numba path to aggregate behavioral equivalence; `tests/test_jax_equivalence.py`
holds the f64/f32/jax forwards to `ABS_TOL=1e-4` (the bit-near-identity that
makes the four-forward consolidation R11 *safe to attempt*); the illegal-slot
mass is asserted `== 0.0`. The audit's reproduced `max|Δp| = 0.0082` stale-
weight divergence is exactly the silent equivalence failure an un-run check
misses.

**Cancer prevented: unsubstantiated "equivalent"/"faster" claims** — the
ADR-0008/0009 closed-vocabulary failure in the perf/equivalence register, and
specifically the category error of pinning a float-sensitive quantity bit-
exactly (which forbids a legitimate optimization *and* a legitimate cross-
language port).

#### P7 — Cross-language wire discipline (the new material)

**Rule (checkable).** The language-agnostic boundary is the **redis raw-bytes
protocol owned by `chocofarm/az/transport.py`** — the **only** contract a
new-language component shares with the Python stack. A new-language component:
**(1)** mirrors the env↔Policy seam with a **composable Policy interface** in
its own language (`RandomPolicy` today, a search/MLP policy later); **(2)**
treats **the wire as the contract** — it reads and writes the exact keys and
byte layouts `transport.py` spells, and shares **no types** with Python and
writes **no second encoder**; **(3)** **reimplements the surface behind the
seam** (belief mechanics, `forward_core`) against the wire, not by translating
Python objects; **(4)** is **validated by parity** under the **P6 behavioral-
equivalence bar** (matched-seed aggregate-stat comparison vs the Python
reference). The full concrete contract is the dedicated section below.

**Cancer prevented: the cross-language form of B (a second encoder / split-
brain across the language boundary) and C (shared mutable state across
processes).** A C++ component that shares Python types or re-encodes the wire
format independently re-creates the split-brain encoder *across a language
boundary*, where it is hardest to catch and the drift is silent. The wire is
the one contract; there is no second one.

---

## Concrete guidance for a new-language (C++) component

This section is the actionable contract for the **incoming C++ search/sim
runner** (the audit's and `scaling-and-cpp-seam.md`'s **Shape A**: a worker
that runs the Gumbel-AZ search and belief mechanics in C++/numba, reading
weight bytes from redis and writing transition bytes back). It is deliberately
maximally concrete: a C++ author should be able to implement against it without
reading Python source beyond `transport.py`. It rests on the four already-clean
seams (`scaling-and-cpp-seam.md` §0): env↔Policy, the net-as-injected-port, the
redis raw-bytes transport, and the version-gated weight broadcast.

### 1. Mirror the env↔Policy seam — a composable Policy interface (P2)

The C++ runner reproduces the **shape** of the env↔Policy seam in its own
language, not a binding to the Python objects. Define a C++ `Policy` interface
whose single method mirrors `Policy.decide(env, loc, bw, collected, lam, rng)`:
the env owns all dynamics (belief, simulate, cost), the policy is injected and
decides. A new C++ capability is a new C++ `Policy` implementation with **zero
edits to the C++ core** — the same inversion of control P2 mandates. Start with
the trivial composable instance (a `RandomPolicy`, mirroring the Python
`RandomPolicy`) to validate the seam and the wire end-to-end **before** porting
any search; graduate to a search/MLP policy once parity on the trivial case
holds. `lam` and the budget (`m`, `n_sims`, `max_steps`) arrive as **live
per-decision scalars** (P4), never baked into the C++ object — they cross the
wire as numbers (see §3).

### 2. Treat the redis wire as the contract — cite the actual keys/format (P7)

`chocofarm/az/transport.py` is the **SOLE owner** of the wire protocol (audit
item K). The C++ runner builds its read/write keys and parses/emits bytes to
match it **exactly**. Connection: via the transport's `config.redis_params()`
— default `127.0.0.1:6379` db 0, env-overridable through `CHOCO_REDIS_HOST`/
`CHOCO_REDIS_PORT`/`CHOCO_REDIS_DB`. The C++ runner reads the **same
`CHOCO_REDIS_*` contract**, so it lands on whatever instance the operator points
the Python transport at; `config.py` is the one owner of "which redis" (P1), not
a port re-typed here. The protocol, verbatim from `transport.py`:

**Weight keys (`weight_keys(run, phase, version)`).** Two keys per published
net, namespaced by `run`, `phase ∈ {"gen","eval"}`, and `version`:

```
manifest_key = az:w:<run>:<phase>:<version>:m
blob_key     = az:w:<run>:<phase>:<version>:b
```

The `phase` segment is the R14 namespacing that **replaced the `it + 1_000_000`
hack** (audit item C, ADR-0011 Rule 4): the gen and eval phases of one
iteration `it` publish to **distinct** keys at the **real** `version=it`. The
C++ worker selects `gen` vs `eval` weights at the same real `version`. A
missing payload is a **loud failure** (ADR-0002 / P5), never a silent stale-net
serve — `read_weights` raises `RuntimeError(f"weight payload az:w:{run}:{phase}:
{version} missing from redis")`; the C++ read must do the same (raise/abort,
not serve a stale net).

**Weight payload (manifest + blob).** The `blob` is **contiguous float64**
weight bytes — the raw `tobytes()` of each weight concatenated, *not* float32,
*not* pickle. The `manifest` is JSON: per-weight `(name, shape, dtype, offset,
byte-length)` entries plus the scalar construction meta (`in_dim`, `H`,
`n_actions`, `y_mean`, `y_std`, and `residual: bool`). The C++ side reconstructs
the net by reading the manifest's meta (so an older manifest without `residual`
→ block OFF), then binds each weight as a view/copy at its `(offset, len)` into
the blob. **Do not re-enumerate or re-order the params**: the param order is
the `WeightContainer`'s canonical (historical) order, recorded in the manifest;
the C++ reader follows the manifest, it does not invent a layout. Optional
params (the residual block `Wr*`/`br*`) ride along automatically **iff** the
manifest lists them — exactly the derive-don't-duplicate (P1) the param-registry
serializer already nails.

**Result keys (`result_keys(res_token, idx)`).** Four keys per task, namespaced
by a fresh per-`generate`-call `res_token` (a uuid) and the task `idx`. Result
keys **carry no `phase` segment** — results exist only for the gen phase and the
uuid `res_token` already prevents collision, so adding `phase` would be dead
symmetry (ADR-0008: don't fabricate a dimension a key doesn't need):

```
X  = az:res:<token>:<idx>:X
PI = az:res:<token>:<idx>:PI
M  = az:res:<token>:<idx>:M
Y  = az:res:<token>:<idx>:Y
```

**Result-blob layout (the float32 wire).** Each of the four blocks is the
contiguous `tobytes()` of a **float32** array (note: results are float32,
weights are float64 — match each exactly):

- `X`  — features, reshaped `(n, feat_dim)`
- `PI` — policy targets, reshaped `(n, n_slots)`
- `M`  — legal-action mask, reshaped `(n, n_slots)`
- `Y`  — value targets, shape `(n,)`

where `n` is the number of transitions the task produced, and the parent reads
each block with `np.frombuffer(..., dtype=np.float32).reshape(...)` against a
`(idx, n, feat_dim, n_slots)` meta. The C++ worker emits each block as a
contiguous little-endian float32 buffer in **row-major** order matching those
shapes. Set the result TTL (`CHOCO_RESULT_TTL`, default 3600s) in the same SET
round-trip — the aborted-iteration self-clean safety net (the post-mortem found
~980 leaked `az:res:*` keys with no expiry; P5).

**The hot knobs** (`m`, `n_sims`, `lam`, `max_steps`) cross as **scalars**
(P4) — a key→number map plus the raw weight/result bytes is language-agnostic
**by construction** (`scaling-and-cpp-seam.md` §0.3). There is nothing
Python-specific on the wire.

### 3. Stay SSOT — no second encoder; reimplement *behind* the seam (P1, P7)

The C++ runner **reimplements the surface behind the seam** — the belief
mechanics (`filter_treasure`/`filter_detector`/`sample_world`/`apply`/
`marginals`) and the single `forward_core(params, X)` — against the wire, **not**
by sharing Python types and **never** by adding a second encoder of the weight
or result layout. This is the SSOT rule (P1) applied across the language
boundary: the layout has **one owner** (`WeightContainer`, surfaced on the wire
via the manifest), and the C++ side **reads that manifest** rather than
hardcoding offsets. Two concrete prohibitions:

- **No hardcoded weight offsets in C++.** Read `(offset, len, shape, dtype)`
  from the manifest JSON. A hardcoded offset is the cross-language form of the
  three-writer feature-layout cancer (B) — it drifts silently the first time
  the Python net's param set changes (e.g. the residual block toggles).
- **No second result encoder.** Emit the four float32 blocks in the exact
  shapes `read_and_delete_results` expects; do not invent a packed/struct
  format. The Python parent's `np.frombuffer(...).reshape(n, fd)` *is* the
  decoder contract; the C++ encoder mirrors it byte-for-byte.

R8 collapsed the belief mechanics to **one** implementation
(`Environment.restrict`, no `MiniEnv` copy) and R11 collapsed the forward to
**one** `forward_core` — so there is exactly **one** Python surface to mirror,
not four (`scaling-and-cpp-seam.md` §0.1–0.2). The C++ port mirrors that one
surface. Adding a second C++ encoder of a layout the manifest already owns
would re-create the split-brain encoder the whole SSOT discipline exists to
prevent — across the hardest boundary to audit.

### 4. Validate by parity — matched-seed aggregate-stat comparison (P6)

Parity is the C++ runner's acceptance test, and it takes the **same behavioral-
equivalence bar as P6 / ADR-0009** — **not byte-identity.** A C++
reimplementation of the same math in a different language and compiler **will**
move the float (float32 is not associative across the C++ reorder, just as it
moves across the numba/JAX reorder the project already accepts) and may flip a
near-tied Sequential-Halving choice. So the bar is, exactly:

- **Logic invariants → bit-exact.** Illegal-action-slot mass is `== 0.0`; the
  legality `M` mask the C++ worker emits is bit-identical to the Python one for
  the same `(loc, belief)` — these are logic facts float32 cannot perturb.
- **Float-sensitive numerics → aggregate behavioral equivalence.** Run the C++
  worker and the Python reference on **matched seeds** and compare **aggregate
  statistics** — fixed-λ₀ rate `ΣR/ΣT`, mean E[T], and action distribution —
  over **N≥300 episodes across ≥2 seeds**, requiring statistical
  indistinguishability **within Monte-Carlo CI**, with the MC standard error
  reported so "indistinguishable" is a number, not an eyeball (the
  `bench_equivalence.py` metric set, applied cross-language).
- **Bit-identity only where free and proven.** Where a quantity *is* bit-stable
  (the legality mask above; a pure-integer index computation), assert it
  bit-exactly — but do not extend that to any float-sensitive output.

This is the **cross-episode** equivalence kind (`scaling-and-cpp-seam.md` §2
Axis A / Shape B): it carries only the forward-roundoff non-exactness the
project already accepts (`test_jax_equivalence` `ABS_TOL=1e-4`), **not** the
approximate-search non-exactness the project defers. Begin parity at the
trivial `RandomPolicy` (which removes the search-choice float-sensitivity and
isolates the wire + belief mechanics), then graduate to the search policy under
the full aggregate-stat bar.

> **The single asterisk** (`scaling-and-cpp-seam.md` §3): the C++ worker is a
> composition of seams that already exist — the env↔Policy seam, the redis
> transport, the version-gated weight broadcast — and **falls out for free**.
> The one structure that does *not* fall out is the synchronous
> `generate → train` loop becoming a continuous async actor-learner; that is a
> localized, R12/R14-enabled restructure, and the deliberate trade it records
> (relaxing the parallel≈serial *bit-determinism* of aggregate reproducibility
> for throughput, while keeping per-episode exactness) is itself a P6
> behavioral-equivalence judgment, recorded so a later reader does not mistake
> the relaxation for a regression.

## Self-application (ADR-0011 Rule 1 — enforcement surface)

Per ADR-0011 Rule 1, this tenet declares **how each principle is enforced**,
against ADR-0011's closed vocabulary (construction-time / test-CI gate /
write-time data constraint / run-time invariant / review-only):

- **P1 (SSOT):** mostly **run-time invariant + test/CI gate** where mechanized
  (`FeatureLayout`'s contiguous-partition assertion; the equivalence tests; a
  `feature_names` test); **review-only** for new facts until their mechanism is
  minted (ADR-0011 Rule 2 is the conversion trigger).
- **P2 (seam/port):** **review-only at design**, with the ACL's strict decode a
  **construction/import-time** raise where a boundary exists (the hp registry
  decode).
- **P3 (no god-objects):** **review-only** (a one-clause-responsibility
  judgment), composing with ADR-0007's review-only file budget.
- **P4 (live, not frozen):** **construction-time + run-time** where the registry
  facet discipline applies (the loud RESTART-drift refusal is a construction/
  run-time raise); **review-only** for placing a new tunable in its tier.
- **P5 (fail loud / root cause):** inherits **ADR-0002's full loudness
  hierarchy**; the guard-vs-band-aid distinction is **review-only**.
- **P6 (substantiate):** inherits **ADR-0009's** surface (test/CI gate for the
  bit-exact and forward-`ABS_TOL` parts; review-only-with-explicit-absence for
  the behavioral part).
- **P7 (cross-language wire):** the wire is enforced by **the parity test/CI
  gate** (matched-seed aggregate comparison) plus the **construction-time** loud
  failure on a missing/malformed payload (`read_weights`' `RuntimeError`); the
  no-second-encoder rule is **review-only** until a manifest-round-trip parity
  test mechanizes it.

This tenet's own Rule-1 declaration: **review-and-audit-policed**, with the
architectural audit as the absence-detector — exactly as ADR-0011 declares for
itself. Its protection is the structure it shapes at authoring time, not its
prose.

## Consequences

### Positive

- **New code is born clean.** The incoming C++ runner and the future async loop
  are authored against a closed checklist of the exact diseases the audit found,
  so the audit's "subtraction and relocation" remediation is never needed for
  them — they never accrete the rot. This is the whole point: propagation by
  default of disciplines the codebase already proved (the env↔Policy seam, live
  λ, derived dimensions).
- **The cancer taxonomy becomes a forward-looking checklist, not just a
  diagnosis.** The audit is point-in-time and not retro-edited (ADR-0005 Rule
  8); this ADR carries its lessons forward as authoring rules so the next
  contributor scans a list rather than re-deriving the lessons.
- **The cross-language boundary has exactly one contract.** P7 makes the redis
  wire the sole seam, so "swap the worker for C++" stays a drop-in and the
  split-brain encoder cannot form across the language boundary.

### Negative

- **Per-authoring overhead.** Each new structure carries a checklist pass; most
  principles are review-only (ADR-0011 Rule 1), so they are policed by attention
  until a recurrence mints a mechanism (ADR-0011 Rule 2). This is the same
  policy-vs-mechanism cost ADR-0003–0009 carry.
- **Some rules are judgments, not measurements.** "No god-object" (one-clause
  responsibility) and the guard-vs-band-aid distinction are calibrated at
  review, like ADR-0007's density heuristic and ADR-0008's severity. ADR-0008's
  substitution test (calibrate to the worst case the shape could apply to, not
  the observed instance) calibrates the cost honestly.

### Neutral

- **No retroactive sweep** (ADR-0004's incremental-retrofit posture). Existing
  code is cleaned by the audit's R-series on its own schedule, not by this ADR;
  this ADR binds **new** structure. Existing rules retrofit on touch.
- **No new infrastructure mandated beyond what the R-series already names.** The
  worked mechanisms (`FeatureLayout`, `BeliefRefs`, `WeightContainer`,
  `transport.py`'s wire) are the audit's, surfaced here as this tenet's
  examples — not new builds this ADR commissions.

## Revisit when…

1. **A principle introduces its own failure mode.** Flag the offending rule
   here by dated amendment (ADR-0005 Rule 8).
2. **The C++ runner lands and the wire contract proves incomplete.** If the
   parity work surfaces a wire detail P7 under-specifies (an endianness
   ambiguity, a manifest field the C++ side cannot reconstruct), record the
   clarification here and repoint the contract — `transport.py`'s docstring is
   the live SSOT, this section the rationale.
3. **A new-language component beyond C++ joins** (a Rust core, a GPU service).
   P7 is stated over "a new-language component," not C++ specifically; confirm
   the redis-wire-as-sole-contract rule survives the new component's
   constraints, or amend.
4. **A principle's review-only enforcement recurs into a defect** (ADR-0011
   Rule 2). The recurrence converts the principle to a mechanism at the
   strongest feasible-and-proportionate surface; record the mechanism here.
5. **The async actor-learner restructure lands** (`scaling-and-cpp-seam.md`
   Shape C). It relaxes the aggregate bit-determinism P6/the design note record
   as a deliberate trade; confirm the trade is still the right one and that
   per-episode exactness held.

## Related

- **ADR-0002 (fail loudly).** P5 defers to it wholesale for the loudness
  hierarchy; the missing-weight-payload `RuntimeError` and the RESTART-drift
  refusal are its mechanisms in the wire/registry register.
- **ADR-0004 (minimal-touch).** Owns the no-retroactive-sweep posture this
  tenet's scoping defers to; new structure is born clean, existing structure is
  retrofitted on touch.
- **ADR-0005 (documentation discipline).** Rule 1 (single-source-of-truth-per-
  handle) is P1's documentation twin; this tenet is its structural form. Rule 8
  (amend point-in-time records by append) governs how the audit is cited
  without retro-editing it.
- **ADR-0007 (file size / information density).** P3 (no god-objects) produces
  small files; ADR-0007 owns the budget and the density heuristic. They
  reinforce; neither restates the other.
- **ADR-0009 (perf/equivalence investigation discipline).** P6 composes with it
  directly and imports its two-tier (bit-exact vs aggregate-behavioral) bar;
  the cross-language parity of P7 is that bar applied across the language
  boundary.
- **ADR-0011 (mechanization discipline).** This tenet is upstream of it:
  structure born clean is structure ADR-0011 never converts. ADR-0011's worked
  mechanisms (`FeatureLayout`, `BeliefRefs`, the param-registry serializer) are
  this tenet's worked examples; its Rule 1 governs this tenet's enforcement-
  surface declaration above.
- **The 2026-06-15 architectural audit** (`docs/notes/audit/`). The source
  substrate — every anti-pattern A–H here inverts one of its §2 cancers, and
  the R-series roadmap is the remediation of existing code this ADR's
  forward-looking rules make unnecessary for new code.
- **`docs/design/scaling-and-cpp-seam.md`.** The four-seam composition and the
  three deployment shapes the C++ section operationalizes; P7's concrete wire
  contract is `transport.py` cited against that design's Shape A.

## License

Public Domain (The Unlicense).
