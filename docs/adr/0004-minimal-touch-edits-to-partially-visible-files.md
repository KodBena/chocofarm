# ADR-0004: Minimal-Touch Edits to Partially-Visible Files

- **Status:** Accepted
- **Genre:** Tenet (cross-cutting authoring discipline) — the second tenet,
  after ADR-0002 (fail loudly).
- **Date:** 2026-06-15
- **Provenance:** Transferred from the LengYue ADR corpus. The tenet is
  universal and transfers wholesale; LengYue's Vue/TypeScript instance list
  (prop/emit/composable drift) is re-derived against chocofarm's real
  surfaces — large numpy/JAX modules with hand-synced numerical contracts the
  test suite only partially polices. chocofarm's code already cites this ADR
  by number (`az/netvalue_ismcts.py` carries an "ADR-0004 register" note).
- **Scope:** All authoring work on this codebase, especially during the kind
  of mechanical refactor where many files are touched in close succession
  (the audit's R-series consolidations are exactly this).

## Context

A chocofarm source file has multiple distinct contracts that the test suite
only partially polices:

- **Numerical equivalence contracts.** The forward graph exists across
  numpy-f64, numpy-f32, and JAX backends; the jax/numpy bit-equivalence test
  (`tests/test_jax_equivalence.py`) pins that they agree. But an edit that
  changes the *order* of operations in one backend can drift the numerics
  below the test's tolerance only on inputs the test doesn't exercise — silent
  until a different input surfaces it.
- **The feature layout's positional contract.** The feature vector's block
  order is written in `features.py`, sliced by offset in `actions.py`, and
  (historically) listed a third time in `feature_response.py`. A reorder of a
  sub-block produces no error and *silently mislabels feature-importance
  rows* — the audit's sharpest landmine (its §2.B). The only guard is
  order-blind to one of the writers.
- **The belief-mechanics duality contract.** The dual bound certifies against
  the env's belief math via `env.restrict`. A change to `apply`'s semantics
  in the env that isn't reflected in the restriction path would have the
  bound certify against stale dynamics — with no test failure (the audit's
  L5: the worst duplication is the one that validates the original).
- **The episode-horizon agreement contract.** The simulator, the base-policy
  rollout, the info-relaxation bound, and the tree search must agree on the
  horizon for a value estimate to be unbiased. It is now owned in one place
  (`env.max_steps`); a change that reintroduces a bare literal in one of the
  four sites silently disagrees with the other three.

In each case the failure mode is *silent at edit time and audible only when a
specific run or input surfaces it* — exactly the most dangerous tier of
ADR-0002's loudness hierarchy.

The risk concentrates sharply during large mechanical sweeps. chocofarm's
largest files are precisely where partial visibility bites: `decomp.py`
(675 lines), `analyzer.py` (605 lines), `exit_loop.py` (510 lines),
`parallel.py` (451 lines), `registry.py` (715 lines), `mlp.py` (360 lines).
When a tool view truncates such a file, the editor's attention is on the one
issue flagged, but the whole file is nominally "in front of them." The
temptation is to fix the flagged issue and tidy the rest "while I'm in here."
That tidy-up, applied to parts the editor doesn't fully see, is where silent
breakage gets introduced.

## Decision

**When editing a file under conditions where the full source is not in
immediate view, the only changes that go in are the specific lines the tool,
test, or task is about.** A "while I'm in here" full-file rewrite is not
permitted under these conditions.

The discipline has two cases:

- **Files visible in full.** Edit freely. The editor has the context to
  reason about the whole file's contracts — the numerical equivalence, the
  feature layout, the belief-mechanics duality, the horizon agreement.
- **Files visible only in part.** Edit only the specific lines the task or a
  failing test points at. If a broader rewrite seems warranted, read the full
  file first; do not produce one from inference. A 675-line `decomp.py` or a
  715-line `registry.py` is exactly the file where an inferred rewrite drifts
  a numerical or layout contract the editor couldn't see.

## Consequences

### Positive

- **Silent numerical / layout / duality drift is structurally prevented,**
  not merely caught after the fact. A reorder of a feature sub-block, a
  reordered op in one forward backend, or a horizon literal reintroduced in
  one of four sites — none get introduced by an edit that touches only the
  flagged lines.
- **The cost of reading the full file is paid up-front,** in the cheaper
  currency (one read) rather than later in the more expensive currency (a
  silently-wrong research result that requires re-running to diagnose).
- **Bisection stays useful.** When a flagged issue gets fixed, the editor
  has confidence nothing else changed.

### Negative

- **Sweeps take more turns.** A consolidation that touches a file the editor
  hasn't seen in full requires reading it first — slower than a speculative
  rewrite.
- **The discipline is policy, not mechanism.** Like ADR-0002, it lives in
  review and authoring habit. There is no automated check that catches a
  violation. (ADR-0011 Rule 1: this is a declared review-only surface.)

### Neutral

- **No code change today.** This ADR documents a discipline for future
  authoring; it does not trigger any refactoring of existing code.

## Revisit when…

1. **A mechanical guard makes a drift class catchable.** The jax/numpy
   equivalence test already catches *some* numerical drift; a feature-layout
   name-sliced assertion (the audit's FEATURE_LAYOUT + `feature_names` test,
   R6) would catch layout drift. As each contract gains a mechanical guard,
   the policy can relax in proportion to the new guarantee — but only for the
   guarded class.
2. **The largest files are split below the partial-visibility threshold.**
   ADR-0007 (file size) is the prophylactic counterpart: if `decomp.py`,
   `analyzer.py`, and `registry.py` are split so partial visibility becomes
   rare, this tenet's reactive discipline applies less often. The two compose.
3. **The discipline introduces its own unanticipated failure mode.**
   Unlikely, but worth flagging as the trigger for revisit.

## Related

- **ADR-0002 (fail loudly).** The failure mode this tenet prevents — silent
  drift a later run is the first to discover — sits at the most dangerous
  tier of ADR-0002's loudness hierarchy. This tenet is ADR-0002's
  authoring-side counterpart: ADR-0002 says "when in doubt, fail audibly at
  runtime"; this one says "when in doubt about the file you're editing, don't
  introduce changes the run will be the first to discover."
- **ADR-0007 (file size and information density).** The prophylactic sibling:
  keep files small enough that partial visibility is rare, eliminating the
  condition under which this tenet's reactive discipline applies. The
  oversized files named in Context are ADR-0007's refactoring queue.
- **ADR-0001.** The same philosophy at the meta-level: don't write code that
  asserts a contract (a numerical equivalence, a layout) you haven't
  verified.

## Not goals (explicit)

- **Not a prohibition on full-file edits.** Files visible in full are edited
  freely; substantial rewrites are fine when the editor has the file in full
  and the rewrite is the point of the commit.
- **Not a requirement that every edit be tiny.** The tenet targets the
  *incidental* rewrite during a sweep focused on something else, not
  deliberate large refactors done with the file fully read.
- **Not a slowdown for trusted, well-known small files.** When a file is
  stable and small enough to edit blind safely, that's a per-file judgment
  call, not a relaxation of the general policy.

## License

Public Domain (The Unlicense).
