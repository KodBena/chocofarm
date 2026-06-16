<!-- docs/notes/cpp-continuation-refactor-decision-2026-06-16.md -->

# Decision: how to make the Gumbel search resumable for the work-stealing pool — Option A (fiber) over Option B (explicit state machine)

**Status:** Decision record, made autonomously under a standing delegation, **reversible and
vetoable until chunk 2 actually builds the pool.** It resolves the open A-vs-B question
`docs/design/cpp-search-runtime.md` §3 left for "when the code is in view." The code is now in
view — `cpp/{include/chocofarm/gumbel.hpp, src/gumbel.cpp}` landed (1a structure + 1b
float32-prior × float64-Q mixed precision) — and reading it reverses the design memo's lean.

## The decision

For the continuation refactor the work-stealing pool needs (a tree that can park at a leaf and
resume), adopt **Option A: run the unchanged `GumbelAZPolicy::run_search` inside a stackful
fiber, and inject a `YieldingNetEvaluator` — a `NetEvaluator` whose `predict()` yields the fiber
and returns the routed leaf value.** Reject **Option B (rewrite `run_search` into an explicit
`advance/resume` state machine)** as the *default*, on the reasoning below. The design memo
(`cpp-search-runtime.md` §3) recommended B; the landed code reverses that, and the memo's own
§3 hedge ("decided then, with the code in view") is exactly this moment.

## What the landed code actually is (the fact that reverses the lean)

The design memo treated the search as an abstraction. It is, concretely, a **five-level
recursion** with the net call at the bottom:

```
run_search → sequential_halving → visit (×count) → simulate_root_action (×c_outcome)
           → descend → descend → … → evaluate → net_.predict()     (gumbel.cpp)
```

— and it was **just fidelity-validated**: 1a pins the discrete structure
(`gumbel_logic.py`), 1b pins a *deliberate* float32-prior × float64-Q mixed precision at four
documented seams (`gumbel.cpp` seams 1–4, `gumbel_precision.py`, ~34/144 discrimination). The
net is reached only through `evaluate()` (`gumbel.cpp:246`), from the root (`run_search:543`) and
interior leaves (`descend:340,346`); the per-tree RNG draw order is one `src.gumbel(n_slots)` at
root then per-sim `sample_world`, with no-leaf short-circuits (TERMINATE / `bw.empty()` / cached)
that consume a draw but issue no leaf.

Against *that*:

- **Option B reifies the entire recursion's stack by hand** — the SH phase/candidate/budget, the
  `visit` counter, the `c_outcome` counter, the `descend` recursion path and its on-unwind `W/N`
  update — into an explicit reentry cursor, **rewriting correctness-critical code that was
  validated 24 hours ago**, and would force **re-proving the 1b mixed-precision fidelity** the
  rewrite could subtly perturb.
- **Option A changes nothing below `run_search`.** `evaluate()` calls `net_.predict()` exactly as
  today; the injected `YieldingNetEvaluator::predict()` parks the fiber and returns the routed
  value. The fiber's stack holds the recursion state automatically. The 1a/1b-validated search
  runs **byte-for-byte**. And it **reuses the existing `NetEvaluator` port with zero search
  edits** — a refinement of the memo's claim that the port can't express submit-and-yield: at the
  type level `predict→value` can't, but a *fiber makes a value-returning `predict()` yield
  transparently*, which the memo undervalued against a deep recursion.

## The decisive point: Option A dissolves the memo's biggest risk

`cpp-search-runtime.md` §7.1 makes "the continuation refactor did not perturb the per-tree RNG
order / the 1b precision / the Danihelka invariants" **the gating precondition of the whole
benchmark**, because under B *all runtimes would be wrong identically* if the refactor erred.
**Under A there is no refactor** — the search is literally unchanged, the fiber only changes
*when* `predict` returns, not *what* it returns — so that validity risk largely evaporates. The
§7.1 layer-2 test degrades from "prove a risky rewrite preserved fidelity" to "prove
fiber-driven ≡ directly-driven," which is near-guaranteed by construction and cheap to confirm.
Preserving the crown-jewel fidelity *by construction* is worth more than B's purity here.

## The ADR-0012 tension, weighed honestly (not papered over)

This is a genuine tension *inside* the tenet:

- **P9 (functional core, no hidden effects) pulls toward B.** A fiber yield is a hidden
  control-flow transfer — a leaf call that *looks* total but suspends the stack and lets other
  trees run, exactly the invisible effect P9 rule 3 names. B's `advance/resume` is the P9-pure
  shape.
- **Minimal-touch + "the fidelity is the crown jewel, just landed" pulls toward A** — and P9's
  own **"measured reason" carve-out covers A**: the reliquary/hidden-effect form is forbidden
  *absent a measured, named constraint*, and "do not rewrite-and-re-prove the 1b mixed precision"
  is a **named** constraint, not habit. The spirit of ADR-0012 (the right structure preserved,
  not rot introduced by needless rewrites of validated code) lands on A.

The honest weighing: P9-purity (B) loses to fidelity-preservation-by-construction (A) **because
the code is freshly validated and correctness-critical**. Were the search not yet validated, B
would be the cleaner ground-up choice. It is validated, so A.

## The honest costs of A (named, with mitigations)

1. **A stackful-fiber mechanism.** Two realizations, an **open sub-decision for chunk 2** (not
   resolved here — it wants a prototype + measure): **boost.context** (clean, a new system dep
   like hiredis/zmq) vs **POSIX `ucontext`** (no new dep, but finicky — signal-mask/perf
   caveats). NOT C++20 stackless `co_await` — that is *viral*, coloring every function from
   `run_search` to `evaluate` into a coroutine, which defeats the whole "don't touch the search"
   point. Lean boost.context for clarity; `ucontext` is the no-dep fallback. Decide by prototype.
2. **A fiber stack per parked tree.** K parked trees ⇒ K stacks (the §3.3 "each parked tree holds
   a `_Node` heap" already implies per-tree memory; the fiber stack adds to it). Bounded, sized,
   reported — the same measure-first posture as cap (b).
3. **The hidden yield is contained, not eliminated.** It lives *entirely* behind the
   `YieldingNetEvaluator` + the runtime's imperative shell; the search core stays oblivious and
   the effect is named at exactly one boundary (the injected leaf). That containment is the most
   P9 can be honored here without the rewrite — and it is recorded as a known deviation, the
   honest ADR-0011-Rule-1 "review-only, named" level, not a silent one.

## Chunk 2 plan under Option A (build order, each its own worktree/branch, each green-gated)

1. **`YieldingNetEvaluator` + a minimal fiber wrapper, proven in isolation** — drive the
   *unchanged* `run_search` inside one fiber with a yielding leaf that a trivial driver feeds
   synchronously, and assert the executed action + improved-π **equal the direct
   `decide`/`run_search`** (the §7.1 precondition, now near-trivial). This validates Option A
   *before* any pool. (Resolves the boost-vs-`ucontext` sub-decision by prototype here.)
2. **Expose the production `RngGumbelSource`** (move it from `gumbel.cpp`'s anonymous namespace to
   `gumbel.hpp`, behaviour-preserving) so a runtime can drive `run_search` directly and surface
   the full `Decision` (`improved_pi` + `n_spent`); re-run `gumbel_logic.py` + `gumbel_precision.py`
   to confirm no behavioural change. (This also lets SerialRuntime carry the full Decision — the
   chunk-1 follow-up.)
3. **The unified work-stealing pool** over the `{SELECT, BACKPROP, FAIL}` task algebra
   (`cpp-search-runtime.md` §1–§2), workers running trees as fibers; **single-writer-per-tree is
   the load-bearing obligation (§8.2) and gets a TSan concurrency test that must be green before
   the pool is trusted.**
4. **The `DealerRendezvous`** (the non-blocking leaf transport) — *only* after the pool works on a
   blocking rendezvous and the §6-Q5 benchmark justifies its complexity (measure-first; building
   it speculatively is the violation the originating consult named).
5. **The §6-Q5 benchmark.**

## Why this is safe to have decided autonomously

It changes no committed code (chunk 1 stands on its own; it uses `decide()` and is Option-A/B
agnostic). It only *sequences* chunk 2 and picks A's lineage. The first chunk-2 step (item 1
above) is itself the proof that A is sound, and it is cheap — so if the maintainer vetoes A in
favour of B's P9-purity, nothing is lost but that one prototype. The decision is recorded here so
the veto is informed, not a silent override.

*Public Domain (The Unlicense).*
