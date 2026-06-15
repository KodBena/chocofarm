# ADR-0002: Fail Loudly

- **Status:** Accepted
- **Genre:** Tenet (cross-cutting principle) — as distinct from ADR-0001,
  which was a specific technical decision. Tenets guide future decisions;
  decisions resolve specific questions. Both are filed under `docs/adr/`
  for single-location retrieval.
- **Date:** 2026-06-15
- **Provenance:** Transferred from the LengYue ADR corpus (this project forks
  that corpus's authoring discipline). The tenet is universal and transfers
  wholesale; the instance list is re-derived against chocofarm's real
  surfaces (the env/scenario seams, the parallel executor, the hp registry,
  the AZ stack). chocofarm's code **already cites this ADR by number** — 16+
  `ADR-0002` invocations across seven modules and the tests — so this
  document is the registry those citations point at. It must exist and match
  their intent. (Before this ADR existed, the 2026-06-15 audit named those
  citations "a binding convention with no definition"; this ADR closes that
  gap.)
- **Scope:** Codebase-wide — the whole `chocofarm/` package and `tests/`.
  Every module that surfaces a deviation through a loud channel is an
  instance: the env's config validation, the parallel executor's
  bounded-drain RuntimeError, the hp registry's RESTART-drift refusal, the
  AZ block-param shape checks, the dtype/precision guard.

## Context

During the buildup of this project, many small and large decisions have
shared one hidden dependency: they each resolve an ambiguity between "try to
handle this anomaly gracefully" and "make the anomaly visible and stop." In
every such case the project has chosen visibility, and has been better off
for it. The pattern has enough weight to be worth naming, so future decisions
don't have to re-derive it.

Examples of decisions already made under this tenet (some labeled with it in
the code, some not, before this ADR existed):

- **The parallel-executor deadlock band-aids.** The JAX-training parallel
  loop suffered intermittent deadlocks (the parent parking in
  `multiprocessing.imap_unordered` awaiting a worker incapacitated by
  JAX-to-spawn-child thread residue; RCA in
  `docs/notes/jaxtrain-deadlock-rca.md`). The remedy converts a permanent
  hang into a **loud, diagnosable RuntimeError** naming the phase, the run,
  and the iteration (`az/parallel.py`), with bounded socket timeouts and a
  loud-now ping if redis is unreachable. Rationale: a silent permanent park
  looks like progress until someone checks; a loud abort with
  "restart from the last checkpoint" is actionable.
- **The hp registry's RESTART-drift refusal.** Changing a baked field
  (`lr`, `l2`, search width) mid-run is refused **loudly**, naming the field,
  the construction-time value, and the new value (`az`/`hp/registry.py`,
  `RestartRequired`), rather than silently running on a config the net is
  invalid against. The registry also refuses a malformed write at the source
  (`schema.py` strict decode) and never coerces a missing/drifted blob to a
  default — `RegistryDecodeError`, `RegistryKeyMissing`, `RegistryUnavailable`
  are distinct so the operator's mental model stays true.
- **The env's config validation.** `with_scenario` raises `ValueError` on a
  wrong-length value vector; `restrict` raises on an empty/out-of-range
  `keep` or a `k_local` exceeding `|keep|` (`model/env.py`). A wrong-length
  value vector is a config error, not something to silently broadcast or
  truncate to N.
- **The AZ block-param shape checks.** Loading weights with a corrupt or
  dimension-mismatched residual block fails **loudly at load** (`az/mlp.py`,
  `tests/test_az_loop.py`) — "fail informative HERE, not deep in the first
  forward."
- **The dtype/precision guard.** An unrecognised precision request is a
  configuration error that raises, not a silent fallback (`az/dtypes.py`).
- **The decomp boundedness abort.** `decomp.py`'s reachable-state
  enumeration aborts loudly on an over-cap synthetic blob rather than
  hanging or OOMing (`solvers/decomp.py`).

The common thread: **when the system has a choice between "recover quietly"
and "fail audibly," prefer audibly.** Silent failures accumulate into debt
that is discovered late — often as a wrong number in a research result, or a
corrupted checkpoint, or a metric that silently misreports.

## Decision

**We adopt "Fail Loudly" as a codebase-wide tenet.** When the system
encounters a condition that deviates from its stated invariants — unexpected
shapes, timeouts, config drift, missing resources, failed transport,
violated numerical assumptions — it surfaces the deviation through the
loudest appropriate channel, not papers over it.

### The hierarchy of loudness

Loudness is not binary. From strongest to weakest:

1. **Import/construction-time error** (the program refuses to start with a
   bad config). Strongest; the anomaly never reaches a run. Preferred where
   the invariant is knowable at setup — the AZ shape checks at load, the
   dtype guard.
2. **Test/build-time error** (a test fails). Nearly as strong for runtime
   paths whose invariants the type system can't capture — the jax/numpy
   bit-equivalence test, the scenario validation tests.
3. **Runtime exception** (raises and halts the current operation). Strong;
   breaks the offending path clearly rather than continuing in an undefined
   state — the parallel-drain RuntimeError, the env config `ValueError`s,
   the registry refusals.
4. **Logged warning / surfaced diagnostic** (faulthandler dump, a named
   log line). Visible to whoever runs or inspects the process. Appropriate
   for "this shouldn't happen, but the run can continue" — the worker
   faulthandler + SIGUSR1 wedge diagnostics.
5. **Silent fallback or default.** Lowest. Appropriate only when the
   fallback genuinely is the right answer (e.g. `env.d`'s live-compute
   fallback for a coord pair absent from the precomputed table — the
   fallback is bit-identical to the table, so it is not a coercion).

The tenet: **reach for the strongest level that fits the anomaly, not the
weakest that's expedient.**

### Concrete rules

1. **No automatic retry / silent fallback for operations that could
   indicate a genuine problem.** Timeouts, failed transport, a config the
   process could not validate: surface them. (Transient socket-level
   behavior bounded by an explicit timeout is not "automatic retry" in this
   sense; it is the bound that makes a stall loud.)
2. **Validate at boundaries; do not coerce.** The hp registry strict-decodes
   the config blob and refuses a malformed one rather than filling missing
   fields with defaults. `with_scenario`/`restrict` validate shapes and
   ranges. A boundary translates and checks; it does not guess.
3. **Sentinel-return-instead-of-raise is a red flag** and requires
   justification. Prefer raising, or a value whose "absent" case is
   distinguishable from a legitimate empty result. A silently-returned wrong
   number is the worst case on a research codebase, because it surfaces as a
   plausible result.
4. **A config field that the receiver cannot honor must not be silently
   accepted.** The audit's "lying signature" finding (a `train_epochs(lr,
   l2)` that ignored its args; a `build(marg)` ignored; a `restrict_faces`
   gating `pass`) is the same failure in the parameter register: a seam that
   looks configured but is dead. Honor it or delete it (this is the
   subject of ADR-0011 Rule 6's lineage and the audit's L6).
5. **No silent state-mutation that breaks an invariant.** The float32 cache
   coherence (ADR-0001) is a fail-loud-adjacent invariant: a rebind keeps
   the cache honest; an in-place mutation that didn't bump the signature
   would silently serve stale weights, which is exactly the silent failure
   this tenet forbids.
6. **A derived value frozen as a literal that feeds a result is a latent
   silent failure.** The three reference rates (static floor, clairvoyant
   ceiling, decomp anchor) are *derived* from the env. `eval/harness.py`
   computes them live; where they are instead hardcoded as literals
   (`exit_loop.py`, and the `%VoI` divisor), the metric will silently
   misreport the moment the env's value vector moves — and a test that pins
   the literal (`test_smoke.py`) *forbids the legitimate retune that should
   update it*. The fix is to derive, never freeze, and to assert the
   recompute is sane rather than pinning a number. (This is the audit's §4
   trace and L4; the firing is currently latent, not realized.)

### What counts as "loud enough"

A deviation is surfaced loudly enough when a developer running the code, or
an operator inspecting it, sees that something went wrong (a raised
exception, a failed test, a named log line, a refusal to proceed), or when
the anomaly is recorded retrievably. It is **not** loud enough when the
system guesses what the caller "probably meant," retries invisibly, returns
a sentinel indistinguishable from a legitimate result, or logs a warning
nobody will see.

## Consequences

### Positive

- **Failures surface on the timescale of development, not of a wrong
  result.** A shape mismatch that raises at load costs minutes; the same
  mismatch surfacing weeks later as "the residual block was never trained"
  costs a research direction.
- **The codebase becomes self-documenting about its invariants.** Every
  fail-loud raise, every strict decode, every shape check is a tiny record
  of what the code expects. The 16+ `ADR-0002` citations are lane markings.
- **The parallel substrate's correctness story is honest.** Because the
  deadlock path fails loud, `test_parallel_deadlock` can assert the abort
  fires — a smaller but truthful guarantee than a silent hang would allow.

### Negative

- **Slightly more verbose code.** A function that raises on malformed input
  is longer than one that returns a default. The justifying comments are
  lines that wouldn't exist without the tenet.
- **The tenet is a policy, not (mostly) a mechanism.** A lazy bare `except:`
  will run fine; only review catches it. *Partially mechanized:* the env
  config validation, the registry strict decode, the AZ shape checks, and
  the dtype guard are tests/raises at `error`-equivalent strength; the
  judgment calls (is this fallback honest? is this sentinel justified?)
  remain review's. ADR-0011 (mechanization discipline) is where the
  enforcement-surface declaration for each rule lives.

### Neutral

- **The tenet does not prescribe implementation details.** It says "fail
  loud"; it doesn't say "always raise." The right mechanism depends on the
  level in the loudness hierarchy that fits — a construction-time raise for a
  config error, a faulthandler dump for a wedge diagnostic.

## Exceptions

Some places deliberately do not fail loud. Documented so the tenet isn't
misapplied.

### Bit-identical structural fallbacks

`env.d(a, b)` serves from the precomputed distance table and **falls back to
a live `math.hypot` compute** for any coord pair absent from the table. This
is not a coercion: the table was built from the same `math.hypot` inputs, so
the fallback is bit-identical. The fallback keeps the contract total; it
never hides a wrong answer. Rule of thumb: **a fallback that provably
returns the same value as the primary path is not a silent failure.**

### Idempotent / no-op-when-already-done operations

A teardown that runs twice, a `seed_registry` that no-ops when the blob
already exists (a `--resume` re-binds rather than clobbering operator
overrides), a cache rebuild skipped when the signature is unchanged — these
are idempotence guarantees, not failures. Rule of thumb: **idempotence is
not silent failure; it is an invariant being preserved.**

### Bounded, scheduled-for-removal compat shims

A defensive fallback during a bounded transition (e.g. the worker
core-pinning's fail-soft `except: widx = 0` while the
process-name-scraping approach is replaced) is acceptable **if** the
alternative would produce a failure the operator cannot action, and **if**
it is commented as bounded and scheduled. (The audit flags the core-pinning
fail-soft as a band-aid to remove, not a permanent exception — see ADR-0009's
sibling and the audit's §2.H.)

## What this tenet does NOT mean

- **Not "crash on any anomaly."** A missing optional field is not
  crash-worthy; a missing required field might be.
- **Not "refuse to handle edge cases."** Edge cases get handled — visibly,
  not silently.
- **Not "spam the logs."** The loudness hierarchy is graded; most anomalies
  are developer/operator-level, not result-level.
- **Not "fail on a bounded transient."** A socket op bounded by an explicit
  timeout that succeeds on its budget is not a failure at our layer.

## Revisit when…

1. **A rule of this tenet gains a mechanical guard** (a lint, a CI gate, a
   schema constraint). Record the mechanization here by dated append — the
   enforcement level is part of a rule's meaning (ADR-0011 Rule 1). The hp
   registry's strict-decode constraints are the first such instance.
2. **A new surface emerges where silent fallback genuinely is the right
   answer.** Add it to Exceptions alongside the three captured here.
3. **A structured error/telemetry layer is adopted** for the training runs.
   The loudness hierarchy may gain a level between log-line and raise, and
   the rules may need updating.

## Related

- **ADR-0001 (immutability and copy-on-write).** The copy-on-write seams and
  the rebind-not-mutate cache invariant are applications of this tenet — they
  raise at boundaries and keep a coherence invariant that, if violated
  silently, is exactly the failure this tenet forbids.
- **ADR-0004 (minimal-touch).** The authoring-side counterpart: ADR-0002
  says "fail audibly at runtime"; ADR-0004 says "don't introduce changes a
  later run will be the first to discover."
- **ADR-0008 (classification discipline).** The proactive register of the
  same family — refuse fuzzy vocabulary matches before they become the
  silent failure this tenet surfaces.
- **ADR-0009 (performance investigation discipline).** The per-domain
  instance for perf claims — an unsubstantiated "faster" is the silent
  failure this tenet names, in the perf-claim register.
- **ADR-0002 applies to documentation consumption.** The root `CLAUDE.md`
  records the gravest sin against this tenet for an LLM collaborator: citing
  a document one has not read in full. Surfacing the gap audibly is the only
  correct move.

## License

Public Domain (The Unlicense).
