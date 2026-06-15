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
contributor scans before authoring), then **the nine principles** (each a
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
| **(new, cross-language)** — two writers of one cross-boundary truth | a hand-mirrored type, offset, key, or codec on the far side of the language boundary that re-authors a fact the near side already defines (a hardcoded weight offset; a second result-blob codec) | **P7** (cross-language wire discipline) — a cross-boundary fact has exactly one authoritative definition and every side *derives* its view (reads it at runtime or generates it at build time), never re-authors it by hand; mechanically enforced at the strongest feasible level (generate/compile-from-one-source > build-time lint > runtime parity backstop). Schema-driven codegen (one schema → N derived readers) is SSOT and is encouraged, not banned |
| **(new, call-boundary)** — a contract carried only by an unenforced or dishonest signature | an untyped function/method/dataclass signature (the contract lives nowhere checkable), or a *lying* one whose body does not honor its annotation (`hp: AdamHParams = None` whose body accepts `None`; `lr/b1/b2/eps: float` populated with jax `Array`s) | **P8** (typed signatures are the contract's SSOT) — the signature is the single source of truth of the input/output contract, honored by the body, at the **strict-where-achievable** bar, with each relaxation a named stub-gap (not a convenience); mechanically enforced by the mypy `--strict` CI gate ratcheting a monotonically-decreasing baseline (ADR-0011 Rule 1) |
| **(new, compiled-component)** — an untyped-effectful-void / black-box mutation in a compiled (C++/new-language) component | a function taking raw pointers (`const float*`, a `T*, size_t` pair) and returning `void` while writing its result through an output parameter or mutating hidden/global state (`void matvec_bias(const float* in, …, std::vector<float>& out)`; `void require_matrix(…, int& rows, int& cols, std::vector<float>& out)`) — a black box you cannot unit-test (it mutates rather than returns), cannot compose (it is not a value-function to chain), and whose contract is invisible (the `void` + raw-pointer signature names neither the bounds, the const-ness, nor what it mutates); or **signaling failure by throwing an exception** (an untyped control-flow escape absent from the signature, which the caller is not forced to handle); or **signaling a legitimately-absent result with a nullable raw pointer or a sentinel** (`const char* opt(…)` returning `nullptr` for "not found"; a `-1` / `""` magic return) — an **untyped optional** whose absence is invisible in the type, so a missed null-check is undefined behavior, the C++ form of ADR-0002's sentinel-instead-of-raise red flag; or, more generally, **a reliquary anti-pattern where a designed-replacement modern feature exists, used out of habit** (a raw `new`/`delete` where RAII / a smart pointer fits; a C-style cast where a named cast says which conversion; a `#define` constant where `constexpr` does; `strcmp`/`strcpy` where `std::string_view` does; `NULL` where `nullptr` does; a `typedef` where `using` does; an unscoped `enum` where `enum class` does) — the modern feature is the standard answer to that construct's hazard at zero runtime cost, so the legacy form needs a *measured* justification, never a habit one | **P9** (functional core, imperative shell) — a computation is a pure function of typed, bounds-carrying, const-correct inputs **and outputs** (`std::span<const T>` / `std::string_view` over a raw `T*`, in *either* direction) **returning its result by value**; every effect is named in the signature, the only sanctioned hidden mutation is a measured hot-path buffer-reuse routed through an explicitly-typed `Workspace`/`Context&` parameter, **a legitimately-absent result is a `[[nodiscard]] std::optional<T>`** and **a failure is a `[[nodiscard]] std::expected<T, Error>`, never a sentinel, a nullable pointer, or a throw** (a throwing ctor becomes a `create(…) -> std::expected` factory). P9 is the **modern-C++ discipline**: these five rules are the catalog, not the whole — for any reliquary construct, prefer the standard (C++11–23) feature designed to ameliorate it at zero runtime cost, the legacy form forbidden absent a measured reason (a profile or a real, named constraint), never habit. The compiled-component form of B (a second/hidden writer), of P2 (hidden state / a lying signature), and of P8 (an untyped/dishonest contract), enforced by the compiler (`-Wall -Wextra`, the nodiscard warning treated as an error) and a future `clang-tidy` config — its `modernize-*` family the purpose-built net for the reliquary→modern substitutions (ADR-0011 Rule 1) |

### The nine principles

#### P1 — Single source of truth / derive-don't-duplicate

**Rule (checkable).** Every fact has exactly **one** home. A *derived*
quantity — a dimension, a layout, a count, the feature/weight layout, the
"keep" set of a sub-instance, a reference rate — is **computed from its source
at the point of use (or cached on the object that owns the source)**, never
hand-copied as a literal or re-encoded in a second place. The check: *grep the
tree for the value; if it appears as an independent literal in two places that
must agree, the rule is violated.* (P8 is this same single-home rule at the
call boundary: a function's contract has one home — its typed signature.)

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
cannot honor is not in the signature** (and P8 at the call boundary: an
annotation the body does not honor is the same lie surfaced at the type layer).

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

**Rule (checkable).** A **cross-boundary fact** — a layout, a key, a byte
format — has exactly **one authoritative definition**; every side **derives**
its view from that one definition (reading it at runtime, *or* generating it
from it at build time) and **never re-authors it by hand**. The violation is
**two writers of one truth** — a hand-mirrored type or a hand-written codec
that can drift from the authority — *regardless of representation*. This is
**P1 applied across the language boundary**: shared types are not the sin
(schema-driven codegen — one schema → N generated/derived readers — is SSOT
and is **encouraged**); a second hand-author of the same truth is.

The rule is **mechanically enforced at the strongest feasible level**, against
ADR-0011/ADR-0002's own enforcement hierarchy: **generate-or-compile-from-one-
source > build-time lint > runtime parity test.** A runtime parity test is a
**backstop, not the contract** — it catches drift only if it runs, with the
right fixtures, *after* the drift already exists. Where the contract is
**static** (a fixed layout with known dtypes/shapes) it should be generated or
compiled from one schema, or at minimum **build-time linted so a Python/C++
format disagreement fails the build** — not left to two hand-written codecs
joined only by a runtime test. This is cancer **G** (load-bearing knowledge in
unenforceable convention) plus **ADR-0011** (mechanize at the strongest
feasible surface) applied across the language boundary. *Never* justify
settling for a weaker mechanism with a scale / minimality / "one X" / "for now"
/ "unnecessary here" / YAGNI argument — that argument shape is itself the tell
this tenet exists to reject (the discipline applied once at small scale is
exactly how the cancers grew).

Separate the **serialization contract** from the **transport/coordination
mechanism** — the durable rule is mechanism-independent. A shared bytes-store
(redis today) holds **state/payloads** (the current weight snapshot, late-join,
large blobs); a **messaging fabric** (ZeroMQ — `scaling-and-cpp-seam.md`
Shape B — or a broker) carries **events/coordination/streaming**. **Never** use
a shared bytes-store as a synchronization/coordination primitive (polling a
key; pub-sub-on-a-store as the backbone) — that is an architectural smell.
Today the synchronous loop coordinates via the OS process pool with redis as a
**pure bytes-store** (no sync-via-store is committed); the C++ worker and the
async actor-learner introduce an explicit fabric for coordination while the
bytes contract is unchanged. Do not enshrine redis as "the contract," and do
not enshrine ZeroMQ as "the one way" either — they are instances of
(bytes-store) and (messaging fabric).

The asymmetry between the two payloads matters: the **weight** payload is a
**dynamic** layout (residual-block toggles, instance-derived dims), so a
**self-describing manifest read at runtime is a legitimate mechanism** there —
the C++ derives `(offset, len, shape, dtype)` each run, so a layout change is
**absorbed, not drifted**; its residual gap is only that the manifest's *own*
schema is still two hand-written (de)serializers. The **result** format is
**static** (four blocks X/PI/M/Y, known dtypes/shapes) — exactly what a
generated/compiled contract is for, and exactly what is left to hand-codecs
plus a runtime test today; **this is where codegen/lint is warranted.**

A new-language component then: **(1)** mirrors the env↔Policy seam with a
**composable Policy interface** in its own language (`RandomPolicy` today, a
search/MLP policy later); **(2)** **derives** its read/write of the keys and
byte layouts from the one authority `transport.py` spells — reading the manifest
at runtime for the dynamic weight layout, and (the floor) a build-time lint
that fails on a format-constant disagreement for the static result layout —
authoring **no second hand codec**; **(3)** **reimplements the surface behind
the seam** (belief mechanics, `forward_core`) against the wire, not by
translating Python objects; **(4)** is **validated by parity** under the **P6
behavioral-equivalence bar** (matched-seed aggregate-stat comparison vs the
Python reference) — as the **backstop**, not the primary guarantee. The full
concrete contract is the dedicated section below.

**Implementation guidance (examples, not mandate).** For raw float blobs on a
hot path a **zero-copy IDL** (FlatBuffers / Cap'n Proto) fits better than
protobuf's parse-and-copy; the **floor** is a build-time lint that fails on a
format-constant disagreement. The MVP's runtime parity test stays — as the
backstop.

**Cancer prevented: the cross-language form of B (two writers of one truth —
a hand-mirrored type or codec that drifts from its authority, across the
hardest boundary to audit, where the drift is silent) and G (load-bearing
format knowledge left in an unenforceable runtime-only convention instead of
generated/compiled/linted from one source) and C (shared mutable state across
processes).** A cross-boundary fact has one authoritative home; every side
derives its view and none re-authors it.

#### P8 — Typed signatures are the single source of truth of a function's contract

**Rule (checkable).** A function, method, or dataclass **signature is the
single source of truth of its input/output contract** — the call-boundary twin
of P1 (one home per fact) and of ADR-0002's no-lying-signature, P2's "a
parameter the receiver cannot honor is not in the signature." An annotation the
body does not honor is a **lying signature** — the type-layer form of the same
lie: `hp: AdamHParams = None` whose body proves `None` is an accepted value;
`lr/b1/b2/eps: float` fields populated with traced jax `Array`s (the exact two
defects the from-scratch strict run surfaced — assessment §3). The bar is
**strict-where-achievable**: `mypy --strict`-clean at the maximal real
strictness a module can reach, where array internals annotated `NDArray[Any]` /
`Any` *satisfy* strict without any relaxation (they are honest types, not
escapes). The check: *(a) does every function/method/dataclass field carry a
param+return annotation? (b) does the body honor each — no value the annotation
forbids reaches a consumer? (c) does the module pass the strict gate, or is it
a documented backlog entry on the way in (see Self-application)?*

**Named-relaxation posture (constraint, not excuse).** A per-module
`ignore_missing_imports` is legitimate **only** for a genuine stub-gap — a
library that ships **no** `py.typed` and no stubs (numba's `@njit` erases the
decorated signature; optax's `GradientTransformation`; tensorboardX's logging
sink). A library that **is** typed (jax ships `py.typed`) must **not** be
blanket-ignored — silencing a checkable library is a convenience-relaxation, so
its friction is instead a **commented `Any` at the use site**, visible in the
diff, distinguishing constraint from excuse in the source itself. Each escape
stays honest by `warn_unused_ignores` (a relaxation that stops being needed
fails CI). And — **reusing P7's no-scale-excuse rule verbatim** — *never*
justify a weaker bar with a scale / "one maintainer" / "for now" / minimality /
YAGNI argument; that argument shape is the tell P7 already named and rejected
(the discipline applied once at small scale is exactly how the cancers grew). A
weaker bar is justified only by a named, verified stub-gap, never by extent.

**Worked example (this codebase).** The single genuine bug cluster the
from-scratch strict run found is `az/mlp_jax_train.py`'s `AdamHParams` path
(assessment §3): the `hp: AdamHParams = None` default whose body
(`hp = self._default_hp if hp is None else hp`) *proves* `None` is accepted —
a P2 / ADR-0002 **lying signature** surfaced at the type layer (fix:
`AdamHParams | None`) — and the `NamedTuple` declaring `lr/b1/b2/eps: float`
while `_hp_arrays` constructs it with `jnp.asarray(...)` traced Arrays (fix:
widen to `float | jax.Array`, the two forms it genuinely holds). The
contrasting **clean documented `Any`** is `forward_core`'s backend-polymorphic
`xp` (numpy-or-jax module) — an honest, commented use-site `Any` at a real
backend-polymorphism seam, not a relaxation: it is what an annotation looks like
when the type genuinely *is* "either backend," distinguished from the lie by
being honored.

**Cancer prevented: untyped / lying-signature contracts.** A contract carried
only by an unenforced signature lives nowhere checkable (the call-boundary form
of G — load-bearing knowledge in unenforceable convention); a contract carried
by a *dishonest* signature is worse — it asserts a guarantee the body breaks,
the call-boundary form of B's two-writers (the signature says one thing, the
body another) and of ADR-0002's silently-accepted lie. P8 makes the signature
the one honored authority, checked by the gate below.

#### P9 — Functional core, imperative shell (the compiled-component contract)

**Rule (checkable).** In a **compiled (C++/new-language) component**, a
computation is a **pure function of typed inputs that returns its result by
value**; effects (I/O, the redis transport, the episode/inference loop, buffer
lifetimes, absence, and failure) live in a thin **imperative shell** that calls
the pure core. This is **P8's typed-contract rule carried into the compiled
component** and **P2's no-hidden-state rule sharpened by C++**, where a raw `T*`
erases — whichever way it points — the bounds and const an input depends on, or
the nullability an output depends on. The discipline costs **no
performance**: it is built on **zero-cost abstractions** (a `std::span<const T>`
compiles to the same pointer+length a hand-rolled pair would; **guaranteed copy
elision / (N)RVO** makes return-by-value free), so the honest signature is not a
tax paid for cleanliness — it is the same machine code with the contract
restored.

**The general posture (P9 is the modern-C++ discipline).** The five rules below
are specific instances of one general principle, and P9 states that principle
explicitly so the discipline extends past the enumerated five. For any **legacy
/ C-with-classes / pre-modern construct** — a *reliquary* form carried forward
out of habit — there is usually a **standard C++ (11–23) feature designed
precisely to make it safer, clearer, or both at zero runtime cost**, and that
feature is **preferred**; the reliquary form is **forbidden absent a measured
reason**. *Check (general): for the construct under review, is there a standard
modern feature designed to replace it? If so, the legacy form needs a measured
justification — a profile showing the modern form costs runtime here, or a real,
named constraint (a fixed C ABI to interoperate with, a toolchain that genuinely
lacks the feature) — never a habit one.* The carve-out is exact: **only a
profile or a real, named constraint** licenses the reliquary form. *That's how
it's always been done* / *it's how I learned C++* / *the old way is fine here* is
**habit, not a reason** — the same lazy-argument shape as the scale-excuse P7/P8
already reject (the discipline declined "just this once" is precisely how the
cancers grew), and P9 rejects it in the same words: a *measured* reason justifies
the legacy form, a habit one never does.

The five rules below are **the current catalog of this principle, not an
exhaustive list** — the principle is general and extensible, and a reliquary
construct outside the five is governed by the general check above just as the
five are governed by their specific forms. A **representative (explicitly
non-exhaustive) catalog** of the same move beyond the five — in each row the rule
is "use the designed replacement," and the list is only its illustration: raw
`new`/`delete` → RAII / value semantics / smart pointers (`std::unique_ptr`,
`std::make_unique`); C-style casts → named casts (`static_cast` /
`reinterpret_cast`, which say *which* conversion and are greppable); `#define`
constants and function-like macros → `constexpr` / `consteval` / templates
(typed, scoped, debuggable); C-string functions (`strcmp` / `strcpy` / `strlen`)
→ `std::string_view` / `std::string` (bounds-carrying, no manual terminator);
`NULL` → `nullptr` (a typed null, no `int`-conversion ambiguity); `typedef` →
`using` (reads left-to-right, and aliases templates); unscoped `enum` →
`enum class` (scoped, no implicit-int decay); a hand-rolled index loop →
range-based-`for` / `<algorithm>` / ranges where clearer; a manual
acquire/release resource pair → an RAII handle whose destructor releases. In
every row the modern feature is the standard library's or language's answer to
the exact hazard the legacy form leaves open, at no runtime cost — so the
reliquary form, not the modern one, is what carries the burden of a measured
justification.

Five **checkable rules a reviewer enforces yes/no from the signature
alone** — the catalog of the general principle at the call/computation
boundary:

1. **Inputs *and outputs* are typed, bounds-carrying, const-correct — no raw
   or nullable pointer crosses the signature in either direction.** Favor
   `std::span<const T>` (or a typed view) over a raw `T*` or a `T*, size_t`
   pair — the span carries the extent and prevents the out-of-bounds the raw
   pointer silently invites; a non-trivial read-only input is `const&`. The ban
   is **directional-symmetric**: a raw-pointer or nullable-pointer **output** is
   as forbidden as a raw-pointer input — a returned string is a
   `std::string_view`, a returned-or-absent string a
   `std::optional<std::string_view>`, never a `const char*` that may be
   `nullptr`. A raw/nullable pointer erases the same contract whichever way it
   points: as an input it erases the extent and const-ness; as an output it
   erases the **nullability** (the `T*` return says nothing about whether
   `nullptr` is a sanctioned value), so a missed null-check is undefined
   behavior the type did not warn against. *Check: does any signature take **or
   return** a raw `T*` / nullable pointer where a `std::span<const T>` /
   `std::string_view` (always present) or a `std::optional<…>` (legitimately
   absent — rule 5) would carry the contract? Is every read-only input `const`?*
2. **Outputs are returned by value.** A function that computes a value
   **returns** it — exploiting guaranteed copy elision / (N)RVO so the return is
   free — not a `void f(…, Out& out)` that writes through an output parameter.
   *Check: does the function return what it computes, or mutate an
   out-parameter? A primary result delivered through an out-parameter is
   forbidden.*
3. **The signature declares every effect.** A function mutates only what its
   signature names — an explicit non-`const` `&` parameter that **is** the
   declared purpose, or `this`; never a global/static, never a parameter whose
   mutation is not the function's stated job. *Check: can a reviewer name, from
   the signature alone, every piece of state the function mutates? An invisible
   mutation is forbidden.*
4. **The ML hot-path exception is explicit and typed.** When **measured**
   allocation overhead on a hot path (an inference / episode loop) genuinely
   requires reusing buffers, the mutable scratch is isolated into an
   explicitly-typed **`Workspace`/`Context`** struct passed as an explicit
   `Workspace&` parameter — so the reuse is a **declared, typed requirement of
   the computation**, not a hidden side-effect. The core stays otherwise pure:
   it reads its typed inputs, uses the `Workspace` as named scratch, and
   **still returns its result by value**. *Check: is every hot-path
   buffer-reuse mutation routed through an explicitly-typed `Workspace`/
   `Context` parameter, with the result still returned by value, and is that the
   only mutation?* **Reusing the P7/P8 no-excuse posture verbatim:** a hidden
   mutation is justified **only** by a measured, named hot-path requirement
   expressed as a typed `Workspace` — *never* by a scale / minimality / "it's
   faster" / "for now" / "unnecessary here" / YAGNI argument. That argument
   shape is the tell P7 and P8 already named and reject; "faster" with no
   measured allocation profile and no typed `Workspace` is a hand-wave, and the
   hand-wave is exactly how the cancers grew.
5. **Absence and failure are BOTH typed return values — `optional` for the
   legitimately-absent, `expected` for the fallible — never a sentinel or a
   nullable pointer.** A result that may be **legitimately absent** is a
   `[[nodiscard]] std::optional<T>` returned by value; a result that may
   **fail** is a `[[nodiscard]] std::expected<T, Error>` returned by value. The
   distinction is drawn precisely and is **not interchangeable**: `optional` =
   "there might be nothing, and that *is* a valid, expected outcome the caller
   chooses what to do with" (a CLI flag the user did not pass, a lookup with no
   match — no error has occurred); `expected` = "it might **fail**, and the
   caller must handle a named `Error`" (a malformed payload, an unreachable
   redis — something went wrong). Choosing `optional` where the absence is
   actually a failure throws away the diagnosis; choosing `expected` where the
   absence is routine fabricates an error category. What is forbidden in **both**
   cases is the same: a **nullable raw pointer** (`const char*` that may be
   `nullptr`) or any **sentinel** (`nullptr`, `-1`, `""`, an empty-but-valid
   value standing for "not found"). A nullable raw pointer is the **worst of
   both** — the absence is invisible in the type (a `T*` return declares nothing
   about whether `nullptr` is a sanctioned value) so a missed check is
   **undefined behavior**, *and* if the absence was really a failure the error
   carries no diagnosis. This is **ADR-0002's sentinel-instead-of-raise red flag
   in the C++ register** — a nullable pointer or magic return is the C++
   sentinel ADR-0002 names — lifted from convention to **compile-enforcement**:
   `[[nodiscard]]` on the `optional`/`expected` return makes *ignoring* the
   absence-or-error a **compile error** (compile-time > runtime in the loudness
   hierarchy P5 defers to), where a nullable pointer's missed check compiles
   silently. (And — composing with **P8** — a `T*` return that may be `nullptr`
   is a **dishonest contract**: the type does not carry the nullability the body
   relies on, the call-boundary lie P8 forbids, here in the compiled register.)

   **Failure, specifically, is never an exception either.** Where the result is
   the `expected` kind, a fallible computation reports failure as that **typed
   value** — never by throwing. An exception is the purest untyped effect: a
   control-flow escape that appears **nowhere in the signature**, that the
   caller is not forced to handle, and that makes the function no longer a total
   value-function of its inputs (the rule-2/rule-3 violation in the one register
   the other four do not cover) — the **control-flow** twin of the nullable
   pointer's **value** sentinel, both of them absences/failures the signature
   hides. `std::expected` makes the error path a **declared part of the return
   type**, and `[[nodiscard]]` makes *ignoring* it a **compile error** — lifting
   ADR-0002's fail-loud to its strongest surface. The **functional core is
   total** — pure arithmetic over already-validated inputs, it neither throws
   nor returns `expected`; the error surface lives entirely in the **imperative
   shell**, at its boundaries (I/O, parsing, construction). A throwing
   **constructor** (which cannot return a value) becomes a static factory:
   `T::create(…) -> std::expected<T, Error>` with a private `noexcept` ctor.
   *Check: does any function signal an absent result with a sentinel / nullable
   pointer instead of a `[[nodiscard]] std::optional<T>`, or a failure by
   throwing (or by a sentinel) instead of returning a `[[nodiscard]]
   std::expected<T, Error>`? Is the absent-or-error path in the return type and
   forced on the caller? Is the core total (throw-free)?* The one thing
   `std::expected` does **not** absorb:
   a genuine **invariant violation** — a state the code's own logic guarantees
   impossible, i.e. a bug — is an `assert`/contract abort, not an `expected`;
   `expected` is reserved for the *recoverable, expected* boundary conditions
   (a missing redis payload, a malformed manifest) an operator or upstream
   causes and a caller can report. The two are categorically distinct: a
   `std::expected` value is for what the world legitimately hands you and the
   program must handle; an abort is for what your own invariants say can never
   happen and a return value would only paper over. (`std::expected` is C++23;
   the toolchain — GCC 15.2 — provides it, so the compiled components build at
   `-std=c++23`.)

**Worked example (the anchor).** The C++ `NetForward` MLP
(`cpp/include/chocofarm/net.hpp`, `cpp/src/net.cpp`) is the cautionary
instance. Its leaf-evaluator entry point `NetPrediction predict(const float* X)
const` does return by value (rule 2 met) — but it takes a **raw `const float*`
with no length** (rule 1 violated: the caller must already know `in_dim_`, and
the body trusts the pointer addresses that many floats — exactly the
bounds-erasure `std::span<const float>` exists to close; the sibling
`predict(const std::vector<float>&)` overload only re-derives the length to
guard *one* caller, not the raw-pointer path the search will actually use). The
internals are the **untyped-effectful void** in full: `void matvec_bias(const
float* in, const std::vector<float>& W, int rows, int cols, const float* bias,
std::vector<float>& out)` takes two raw pointers and an `int rows, int cols`
pair (no bounds, no const-carrying view) and returns its result by **writing
through `std::vector<float>& out`**; `void require_matrix(…, int& rows, int&
cols, std::vector<float>& out)` returns `void` while writing **three**
out-parameters and `void require_vector(…, int& len, std::vector<float>& out)`
**two**, in place of returning a small typed result; `void
relu_inplace(std::vector<float>& v)` is a void in-place mutation.
None of these can be unit-tested as value-functions or chained, and none
declares its contract in its signature. The **compliant form**: the matmul is a
pure value-function — `std::vector<float> matvec_bias(std::span<const float> in,
std::span<const float> W, int rows, int cols, std::span<const float> bias)`
returning its result by value (free under NRVO), `relu` returning a new vector
(or taking and returning by value), and `require_matrix`/`require_vector`
returning a small typed result struct rather than writing those out-params; the
public entry becomes `predict(std::span<const float> x, const WeightPayload& w)
-> NetPrediction` (value-returned), with the per-layer matmul scratch — **if and
only if** a measured allocation profile on the search's leaf loop shows the
per-`predict` `std::vector` churn matters — moved into a typed
`ForwardWorkspace&` parameter, leaving the core otherwise pure and still
returning `NetPrediction` by value. The as-merged interim `NetForward` predates
P9 (it is the live instance that **motivated** the rule) and is to be brought
into compliance; per the no-retroactive-sweep scoping it is retrofitted on
touch, not by a P9 sweep.

**Worked example (the error axis, rule 5).** Every `throw` in `cpp/src` today is
a `std::runtime_error`, and every one of them is at a **boundary** — none on the
hot path: `transport.cpp` (redis connect/GET/SET, and the missing-weight-payload
abort mirroring `read_weights`), `instance.cpp` (the instance-file/JSON load),
and the `NetForward` **constructor** with its `require_matrix`/`require_vector`
helpers, which validate the manifest at construction. The forward compute itself
(`predict(const float* X)`, `matvec_bias`, `relu`) and the search that will call
it are **throw-free** — the only `throw` reachable from a `predict` overload is
the length guard at the *boundary* of the `vector`-taking entry, not on the raw-
pointer compute path the search uses. So the core is already total; what rule 5
adds is that the boundary's failures should be **typed return values, not
thrown.** The compliant form returns `[[nodiscard]] std::expected<…, Error>`
from those boundary functions (`read_weights`, the instance loader,
`require_matrix`/`require_vector`), so a caller cannot ignore the error path
without a compile error; `NetForward`'s construction — a throwing ctor cannot
return a value — becomes a `NetForward::create(const WeightPayload&) ->
std::expected<NetForward, Error>` factory over a private `noexcept` ctor.
The forward/search core stays total and exception-free, exactly as it is today.
The distinction rule 5 draws against the as-merged code: those manifest-shape
checks are **recoverable boundary conditions** (a malformed payload an upstream
produced, a missing redis key an operator can be told about) — they are
`expected`, not `assert`. A `matvec_bias` indexing past `cols` because the
caller passed an `in_dim_`/`hidden_` the constructor already reconciled would be
the other category — an **invariant violation**, a bug, an `assert`/abort — and
it never becomes an `expected`.

**Worked example (the optionality axis, rules 1 & 5).** The CLI helper in
`cpp/src/main.cpp` is the live **untyped-optional** instance:

```cpp
const char* opt(int argc, char** argv, const char* name) {
    for (int i = 1; i + 1 < argc; ++i)
        if (std::strcmp(argv[i], name) == 0) return argv[i + 1];
    return nullptr;                  // "not found" as a nullable raw pointer
}
```

It parses `--instance`, `--phase`, `--lam`, etc., and returns `nullptr` when
the flag is absent — and absence here **is a legitimate, expected outcome** (an
optional flag the user simply did not pass), not a failure. So this is exactly
the absence rule 5 names, encoded the forbidden way: an **untyped optional** (a
nullable `const char*` whose absence is invisible in the type — a caller that
forgets the null-check dereferences `nullptr`, undefined behavior the type never
warned against) that is **also** a raw-pointer in *and* out (rule 1: raw `char**`
input, raw `const char*` output). It is the C++ sentinel ADR-0002 names, and a
P8 dishonest contract (the type does not carry the nullability the callers
rely on). The **compliant form** makes the absence typed and the pointers views:

```cpp
[[nodiscard]] std::optional<std::string_view>
opt(std::span<const std::string_view> args, std::string_view name);
```

— `std::optional` (not `expected`: a missing flag is routine absence, not an
error) carries the "might be nothing" in the return type, `[[nodiscard]]` makes
ignoring it a compile error, and `std::string_view` replaces the raw pointers in
and out. The **imperative shell** does the one untyped→typed translation at the
boundary, building the typed view once in `main`:

```cpp
std::vector<std::string_view> args{argv, argv + argc};   // the ACL, once
```

This is the boundary acting as the **Port/ACL** (P2) that translates the untyped
`argv` the OS hands it into typed values the core consumes — **not an excuse to
keep the raw pointers** flowing inward. The single `argv`→`string_view` decode
is the sanctioned translate-at-the-edge; every signature downstream of it is
typed. (This `opt` helper predates P9 and is **retrofitted on touch** — it falls
in `#28`'s scope — per the no-retroactive-sweep scoping, not swept for its own
sake.) The reflex to wave this off as "it's just CLI parsing, the absence is
obvious" is **the exact rationalization this tenet rejects**: a nullable
`const char*` is an untyped optional whether it parses argv or a redis payload,
the missed null-check is the same undefined behavior, and "it's just X" is the
scale/minimality tell P7/P8 already named — the discipline declined "just this
once at the edge" is precisely how the cancers grew.

**Cancer prevented: untestable, uncomposable black-box mutations.** A
`void`-returning, raw-pointer-taking, out-parameter-writing function in a
compiled component is the compiled form of three cancers at once — **B** (a
second/hidden writer of state the signature does not name), **P2's hidden state
/ lying signature** (a contract the signature does not carry), and **P8's
untyped contract** (no bounds, no const, no return declared) — sharpened by C++,
where the raw `T*` erases the very bounds and const-ness the contract needs. The
functional core makes each computation a value you can test in isolation and
compose, and confines every effect to the named, typed imperative shell.

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

### 2. Derive from the bytes-store channel — cite the actual keys/format (P7)

`chocofarm/az/transport.py` is the **SOLE authority** of the serialization
contract (audit item K): the keys and byte layouts have **one definition**
there, and the C++ runner **derives** its read/write from it — never re-authors
it by hand. Keep the **serialization contract** distinct from the
**transport/coordination mechanism**: redis here is a **pure bytes-store**
holding state/payloads (the weight snapshot, the per-task result blobs), *not*
a coordination primitive. **Coordination today** is the OS process pool — no
sync-via-store is committed (no key-polling, no pub-sub-on-a-store backbone);
**coordination tomorrow** (the C++ worker, the async actor-learner) is an
explicit messaging fabric (ZeroMQ — `scaling-and-cpp-seam.md` Shape B — or a
broker), introduced for events/coordination/streaming **while the bytes
contract below is unchanged.** Redis is named here as the current instance of
(bytes-store), not enshrined as "the contract"; do not enshrine the fabric as
"the one way" either. The C++ runner is a **transport** component, so its
connection is via the transport role's `config.transport_redis_params()` —
default `127.0.0.1:6380` db 0, the **ephemeral** memory-cache instance
(`allkeys-lru`), env-overridable through `CHOCO_TRANSPORT_REDIS_HOST`/
`CHOCO_TRANSPORT_REDIS_PORT`/`CHOCO_TRANSPORT_REDIS_DB`. This is explicitly
**NOT** the registry's disk-persisted `127.0.0.1:6379` `noeviction` instance
(`config.registry_redis_params()`, the `CHOCO_REGISTRY_REDIS_*` family) — the
two roles are deliberately distinct instances. The C++ runner reads the **same
`CHOCO_TRANSPORT_REDIS_*` contract**, so it lands on whatever instance the
operator points the Python transport at; `config.py` is the one owner of "which
redis" per role (P1), not a port re-typed here. The protocol, verbatim from
`transport.py`:

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

### 3. Stay SSOT — derive, never re-author; reimplement *behind* the seam (P1, P7)

The C++ runner **reimplements the surface behind the seam** — the belief
mechanics (`filter_treasure`/`filter_detector`/`sample_world`/`apply`/
`marginals`) and the single `forward_core(params, X)` — against the wire, and
**derives** its view of every cross-boundary layout from the one authority
rather than **re-authoring it by hand**. This is the SSOT rule (P1) across the
language boundary: a cross-boundary fact has **one authoritative definition**,
and every side reads it (at runtime) or generates it (at build time) — two
writers of one truth is the violation. The two payloads sit at opposite ends of
the static↔dynamic axis and warrant different mechanisms:

- **Dynamic weight layout → derive from the runtime manifest (no hardcoded
  offsets in C++).** The layout has one owner (`WeightContainer`, surfaced on
  the wire via the manifest), and because the layout is **dynamic** (residual-
  block toggles, instance-derived dims), a self-describing manifest **read at
  runtime** is the legitimate mechanism: read `(offset, len, shape, dtype)` per
  weight from the manifest JSON each run, so a layout change is **absorbed, not
  drifted**. A hardcoded offset would be the cross-language form of the
  three-writer feature-layout cancer (B). *Residual gap:* the manifest's **own**
  schema is still two hand-written (de)serializers (Python pack / C++ parse) —
  the one place this payload is not yet generated from a single schema.
- **Static result format → a generated/compiled/linted contract, not two hand
  codecs.** The four float32 blocks X/PI/M/Y have **fixed, known dtypes/shapes**
  — a **static** contract, exactly what codegen exists for. Today it is left to
  two hand-written codecs (the Python `np.frombuffer(...).reshape(n, fd)`
  reader and the C++ emitter) joined only by the runtime parity test — a
  **runtime-only convention** (cancer G). At the strongest feasible level it
  should be **generated/compiled from one schema**; the **floor** is a
  build-time lint that **fails the build** on a Python/C++ format-constant
  disagreement. Whichever mechanism is chosen, the C++ side **derives** the four
  blocks' shapes from the one authority and invents **no** packed/struct format
  of its own. (For raw float blobs on this hot path a zero-copy IDL —
  FlatBuffers / Cap'n Proto — fits better than protobuf's parse-and-copy; this
  is an example, not an ADR mandate.)

R8 collapsed the belief mechanics to **one** implementation
(`Environment.restrict`, no `MiniEnv` copy) and R11 collapsed the forward to
**one** `forward_core` — so there is exactly **one** Python surface to mirror,
not four (`scaling-and-cpp-seam.md` §0.1–0.2). The C++ port mirrors that one
surface. Adding a second C++ encoder of a layout the manifest already owns
would re-create the split-brain encoder the whole SSOT discipline exists to
prevent — across the hardest boundary to audit.

### 4. Validate by parity — the backstop, not the primary guarantee (P6)

Parity is the C++ runner's acceptance test under the **same behavioral-
equivalence bar as P6 / ADR-0009** — **not byte-identity** — but it is the
**backstop, not the contract.** A runtime parity test catches a drift only if
it runs, with the right fixtures, *after* the drift already exists; the primary
guarantee is the generated/compiled/linted serialization contract of §3 that
makes a format disagreement **unable to be authored** in the first place
(strongest-feasible: generate-or-compile-from-one-source > build-time lint >
this runtime parity test). With that floor in place, parity then certifies the
*numerics* — a C++ reimplementation of the same math in a different language and
compiler **will** move the float (float32 is not associative across the C++
reorder, just as it moves across the numba/JAX reorder the project already
accepts) and may flip a near-tied Sequential-Halving choice. So the parity bar
is, exactly:

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
- **P7 (cross-language wire):** enforced at the **strongest feasible level** —
  the **static** result format wants a **generate/compile-from-one-schema or
  build-time-lint gate** (a Python/C++ format-constant disagreement fails the
  build); the **dynamic** weight layout is enforced by the **runtime manifest**
  the C++ derives `(offset, len, shape, dtype)` from (its residual gap: the
  manifest's own schema is still two hand-written codecs). Below those sits the
  **runtime parity test/CI gate** (matched-seed aggregate comparison) as the
  **backstop**, plus the **construction-time** loud failure on a missing/
  malformed payload (`read_weights`' `RuntimeError`). Until the static-format
  codegen/lint is minted, that gap is **review-only** — but settling for the
  runtime-test-only backstop is *not* justified by a scale / minimality / "for
  now" argument (that argument shape is the tell P7 rejects).
- **P8 (typed signatures):** **test/CI gate** — the **mypy `--strict` CI gate**
  (`pyproject.toml` `[tool.mypy]` + `tests/test_mypy_strict.py`) is the
  ADR-0011 Rule-1 mechanism that converts "typed signatures" from review-only
  prose into an enforced contract. It runs `mypy --strict` against an explicit
  `STRICT_CLEAN` set and asserts zero errors, ratcheting a
  **monotonically-decreasing baseline module-by-module** (assessment §5,
  Stages 0–4): a module joins the gated set as it is typed, and a regression in
  any gated module's annotations fails CI. A module is **review-only** until it
  joins that set — and that join is the ADR-0011 Rule-2 conversion trigger (the
  recurrence that converts review-only prose to a mechanism), here a scheduled
  monotonic rollout rather than a defect. `warn_unused_ignores` keeps each
  named relaxation honest at the same gate.
- **P9 (functional core, imperative shell):** a **mix** — the error axis
  (rule 5) is **partly compile-enforced**, the other four (input/output/mutation)
  are **review-only**. `[[nodiscard]]` on every `std::expected`-returning
  boundary function, **with the nodiscard warning treated as an error**, makes an
  **unhandled error a build failure** — a strictly stronger surface than the
  review-only structural rules, and the same compile-time-over-runtime move ADR-
  0011 Rule 1 ranks highest. Rules 1–4 are policed against their checkable form
  at C++ review, with the **compiler (`-Wall -Wextra`)** as the standing floor
  and a **future `clang-tidy` config** as the mechanization surface. The compiler
  already raises some of the relevant signals (an unused parameter, a
  const-violation); the `clang-tidy` config is the ADR-0011 Rule-2 conversion
  trigger — **when a P9 violation recurs after this record, mint the `clang-tidy`
  check** that catches it (e.g. a check against out-parameters or raw-pointer
  arithmetic where a `std::span` belongs, or `-Werror` on a thrown exception
  escaping a function the contract says should return `expected`) rather than
  re-stating the rule in prose. The **general modern-C++ posture** has a
  concrete, purpose-built mechanization surface: clang-tidy's **`modernize-*`
  check family** exists *precisely* to catch the reliquary→modern substitutions,
  so the ADR-0011 Rule-2 trigger for the general principle is to **enable the
  `modernize-*` (or `cppcoreguidelines-*`) check that catches the recurring
  reliquary form** — `modernize-use-nullptr` (`NULL` → `nullptr`),
  `modernize-use-using` (`typedef` → `using`), `modernize-avoid-c-arrays`,
  `modernize-loop-convert`, `modernize-make-unique`/`-make-shared` (raw `new` →
  smart pointers), and `cppcoreguidelines-pro-bounds-pointer-arithmetic` /
  `-pro-type-cstyle-cast` for the bounds/cast rows — each the standing answer to
  one catalog row. Until the check for a given recurrence is enabled, that
  construct is review-policed against the **general check** (is there a designed
  modern replacement?), with the compiler the floor. Until those recurrences
  fire, rules 1–4 and the general posture are review-only — and settling for
  review-only is *not* justified by a scale / "one compiled component" / "for
  now" / "that's how it's always been done" argument (that argument shape is the
  tell P7/P8 reject); it is the honest ADR-0011 Rule-1 level for a discipline
  whose recurrence has not yet fired.

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
- **Every cross-language fact has exactly one authoritative definition.** P7
  gives each cross-boundary layout/key/format one home from which every side
  derives — separating the serialization contract from the transport/
  coordination mechanism (a bytes-store holds state, a messaging fabric carries
  coordination) — so "swap the worker for C++" stays a drop-in and a second
  hand-author of one truth cannot form across the language boundary.
- **The compiled component is value-functional, not a black box.** P9 makes each
  C++ computation a pure function of typed, bounds-carrying inputs returning its
  result by value, so the compiled core is unit-testable and composable rather
  than an untyped-effectful void — at zero performance cost (zero-cost
  abstractions: a `std::span` is a pointer+length, return-by-value is free under
  (N)RVO), with the one hot-path buffer-reuse exception declared as a typed
  `Workspace` parameter rather than hidden. Absence and failure, too, are
  **typed return values** — a legitimately-absent result a `[[nodiscard]]
  std::optional<T>`, a failure a `[[nodiscard]] std::expected<T, Error>`, never
  a nullable pointer or sentinel whose absence is invisible in the type (the C++
  form of ADR-0002's sentinel red flag, a P8 dishonest contract) and never an
  untyped thrown escape — so the absence/error path is declared in the return
  type, ignoring it is a **compile error** (ADR-0002 fail-loud at its strongest
  surface), and the core stays total while that surface lives at the shell's
  boundaries.

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
   P7 is stated over "a new-language component," not C++ specifically — and a
   third reader of a static layout is **exactly** the recurrence at which
   generating/compiling the serialization contract from one schema becomes the
   right move (ADR-0011 Rule 2). Confirm the one-authoritative-definition /
   derive-don't-re-author rule survives the new component's constraints, and
   that the static formats are mechanized at the strongest feasible level
   rather than gaining a third hand codec; amend if not.
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
  mechanisms (`FeatureLayout`, `BeliefRefs`, the param-registry serializer, and
  the **mypy `--strict` CI gate** that backs P8) are this tenet's worked
  examples; its Rule 1 governs this tenet's enforcement-surface declaration
  above, and its Rule 2 (recurrence → mechanism) is the trigger by which a
  module joins P8's gated set.
- **The 2026-06-15 architectural audit** (`docs/notes/audit/`). The source
  substrate — every anti-pattern A–H here inverts one of its §2 cancers, and
  the R-series roadmap is the remediation of existing code this ADR's
  forward-looking rules make unnecessary for new code.
- **`docs/design/scaling-and-cpp-seam.md`.** The four-seam composition and the
  three deployment shapes the C++ section operationalizes; P7's concrete wire
  contract is `transport.py` cited against that design's Shape A.

## License

Public Domain (The Unlicense).
