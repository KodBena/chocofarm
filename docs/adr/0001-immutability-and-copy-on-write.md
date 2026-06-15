# ADR-0001: Immutability, Copy-on-Write, and Rebind-not-Mutate

- **Status:** Accepted
- **Genre:** Decision (a specific technical decision, as distinct from the
  cross-cutting tenets that follow) — resolves how shared mutable-looking
  state is handled across the env, the scenario/restriction seams, and the
  inference-weight cache.
- **Date:** 2026-06-15
- **Provenance:** Adapted from the LengYue ADR corpus (this project is a fork
  of that corpus's authoring discipline). LengYue's ADR-0001 was a Vue 3
  `readonly`/reactivity decision; chocofarm has no Vue, no TypeScript, and no
  reactive store, so that decision does not transfer. What transfers is the
  *question* it answered — "where does mutable-looking state actually get
  mutated, and what guarantees does the code rely on?" — re-derived against
  chocofarm's real seams. The general philosophy LengYue's ADR-0001 shares
  with the rest of the corpus (declarations should match actual behavior, no
  aspirational guarantees) is preserved.
- **Decision drivers:** correctness of the belief filter; cheap scenario/
  restriction sweeps without rebuilding expensive geometry; coherence of the
  float32 inference cache against its float64 source; the heterogeneous-value
  experiment the project is building toward.

## Context

chocofarm is a single numpy/JAX/numba Python package. It has no reactive
framework and no language-level immutability primitive — Python objects are
mutable by default and `numpy` arrays are mutable in place. So the question
LengYue's ADR-0001 framed in `readonly`/Proxy terms reappears here in plain
Python terms: **which state is genuinely immutable, which is copy-on-write,
and where does the code depend on a writer rebinding rather than mutating?**

Three concrete seams in chocofarm answer this question, and they were not
designed under a common name until the 2026-06-15 architectural audit
(`docs/notes/audit/architectural-audit-2026-06-15.md`) named the pattern and
its hazards. This ADR is that name.

### The three seams

1. **The belief world-set is immutable; every filter returns a fresh array.**
   The belief is the numpy array `bw` of latent-world bitmasks still
   consistent with all observations (`Environment.worlds` is the prior;
   `filter_treasure` / `filter_detector` shrink it). Each filter
   (`env.py`) returns `bw[mask]` — a **new** array — never an in-place edit
   of the caller's belief. A policy that holds a belief and calls a filter
   gets a fresh, smaller belief back; its own copy is untouched. This is what
   lets the simulator, the solvers, and the dual bound all share belief
   primitives without any of them corrupting another's state.

2. **Scenario and restriction are copy-on-write on the env.** The expensive
   Tier-1 geometry — the ~4.5k-entry distance table `_dist`, the
   `C(20,5)=15,504`-world array, the 44 parsed arrangement faces — depends
   only on the instance, never on the scenario knobs (`value`, `entry`,
   `teleport_overhead`). `env.with_scenario(Scenario)` and
   `env.restrict(keep, k_local)` (`env.py`) therefore do a
   `copy.copy(self)` that **aliases** the geometry and overrides only the
   handful of scenario/restriction fields. `self` is never mutated. A
   value/K sweep is `[env.with_scenario(s) for s in scenarios]` — one
   geometry build, N shallow copies — not N full rebuilds. The contract is
   load-bearing for the dual bound: `restrict` produces the same belief
   mechanics the learner uses, so the bound certifies against the exact same
   dynamics (one implementation, not a copy).

3. **The float32 inference cache is invalidated by rebind, not by writer
   cooperation.** `ValueMLP` (`az/mlp.py`) caches float32 copies of its
   float64 weights for the inference forward. Coherence is an **invariant
   over every writer, not a per-writer gate**: the cache-validity check
   compares the source weight *objects* by identity (`is`), so any writer
   that **rebinds** (`self.Wx = new_array`, which is what load/warm-start and
   the JaxTrainer do) automatically invalidates the cache. The hazard this
   closes — a writer that mutates a weight array in place would leave the
   cache silently serving stale float32 weights — was a real out-of-frame
   audit finding (`az/mlp.py`, the cache-coherence comment block).

### Why this is a decision, not a tenet

It resolves a specific structural question (how shared mutable state is
handled at three named seams) rather than stating a cross-cutting principle.
The tenets are ADR-0002 and onward. But the decision is itself an
*application* of ADR-0002 (fail loudly): `with_scenario` raises on a
wrong-length value vector rather than broadcasting or truncating it, and
`restrict` raises on an empty/over-restricted `keep` rather than clamping it
to a degenerate sub-instance.

## Decision

**Treat the belief world-set as immutable (filters return fresh arrays).
Make scenario and restriction copy-on-write on the env (alias the geometry,
override only the scenario/restriction fields, never mutate `self`). Enforce
inference-cache coherence as a rebind-not-mutate invariant over all writers,
not a per-writer obligation.**

Concretely:

- **Immutable belief:** no method mutates a caller's `bw` in place. A filter
  produces a new array. (If a future hot-path optimization wants in-place
  filtering, it gets its own copy first and the rule is named at the site.)
- **Copy-on-write env:** `with_scenario` / `restrict` shallow-copy and
  override only the Tier-2/restriction fields. **No code may cache a
  structure DERIVED from `value`/`entry`/`tp` on the env** — such a cache
  would be shared stale across scenarios (the contract is stated verbatim in
  `env.__init__`). Value/entry/tp-derived quantities are computed at
  point-of-use (`apply`, `simulate`, `exit_cost`).
- **Rebind-not-mutate cache:** the float32 cache (and any future identity-
  validated cache) is coherent because every weight writer rebinds. A writer
  that wants to mutate weights in place must bump the cache signature
  explicitly; the default and the documented expectation is rebind.

## Consequences

### Positive

- **Belief sharing is safe by construction.** The simulator, eight solvers,
  and the dual bound all call the same belief primitives without any of them
  able to corrupt another's state — because nobody mutates in place.
- **Scenario/restriction sweeps are cheap.** The heterogeneous-value
  experiment (the project's stated next lever) becomes a comprehension over
  `Scenario`s rather than a monkeypatch of a frozen module global (the attic
  precedent) or N rebuilds of the distance table.
- **No `id(env)` cache can silently alias the old config**, because
  `with_scenario`/`restrict` yield fresh objects. (The remaining `id(env)`
  hazard the audit named lives in `az/actions.py`'s slot-table global, not
  in the env itself — see ADR-0002, and the audit's R9.)
- **Cache coherence does not depend on every writer remembering a rule.**
  The invariant is checked at read time against object identity, so a new
  writer that follows the (default, idiomatic) rebind pattern is correct for
  free.

### Negative

- **The copy-on-write contract is a convention, not a mechanism.** Nothing
  stops a future contributor from caching a value-weighted precompute on the
  env; the `env.__init__` comment is the guard, and review is its
  enforcement. This is the same policy-not-mechanism cost every tenet here
  carries (ADR-0011 names the enforcement-surface declaration discipline).
- **A writer that mutates weights in place silently breaks the cache.** The
  invariant catches stale serves only because writers rebind; an in-place
  Adam-style mutation (the historical numpy path, since removed) would need
  to bump the signature. The rule is named in `mlp.py`; a violator is
  review's to catch.

### Neutral

- **No language-level immutability is claimed.** numpy arrays are mutable;
  Python objects are mutable. The guarantees here are conventions the code
  upholds at named seams, policed by review and a small number of tests
  (`tests/test_scenario.py`, the cache-coherence tests in the AZ suite), not
  by a frozen-object runtime.

## Revisit when…

1. **A hot-path optimization wants in-place belief filtering.** If the
   per-episode allocation of fresh belief arrays shows up as a real cost in a
   captured profile (ADR-0009), in-place filtering on a caller-owned copy may
   be warranted — at named sites, with the immutability rule carved out
   explicitly there.
2. **A scenario-derived cache is genuinely wanted on the env.** If a future
   feature needs a value-weighted precompute, the copy-on-write contract must
   be revised (the cache keyed on the scenario, or rebuilt in
   `with_scenario`) rather than silently violated. That revision is this
   ADR's amendment trigger.
3. **A weight writer needs in-place mutation.** If a future training path
   mutates weights in place for performance, the rebind-not-mutate invariant
   needs an explicit signature bump and this ADR records the exception.

## Related

- **ADR-0002 (fail loudly).** `with_scenario` and `restrict` raise on
  malformed config rather than coercing — the copy-on-write seams are
  fail-loud at their boundaries.
- **The 2026-06-15 architectural audit** (`docs/notes/audit/`) — named the
  copy-on-write seam (R7), the `restrict`/`MiniEnv` belief-mechanics unity
  (R8), and the rebind-not-mutate cache coherence (the f32-cache hazard) as
  the load-bearing seams to preserve. This ADR is the standing record of the
  decisions that audit's roadmap targeted.

## Not goals (explicit)

- **Not adopting language-level immutability.** Python and numpy don't
  provide it; the conventions are upheld at named seams.
- **Not freezing the env.** The env is constructed once with mutable
  attributes; the discipline is that *scenario/restriction* go through
  copy-on-write and the *belief* is never mutated in place — not that the
  object is immutable.
- **Not a claim that the conventions are bulletproof.** They rely on review
  and a handful of tests. That is a cost we accept.

## License

Public Domain (The Unlicense).
