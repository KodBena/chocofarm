# ADR Synopsis

A condensed reference to the architectural decisions and tenets that govern
this codebase. Each entry summarizes what the ADR decides and why a
contributor would care, in 1–2 paragraphs. For full context, exceptions, and
rationale, read the ADR itself.

This document is a navigational aid — primarily for LLM contributors arriving
cold and needing the codebase's architectural personality in one read — and a
quick refresher for human contributors. It is not authoritative; the ADRs
themselves are. If this synopsis disagrees with an ADR, the ADR wins and the
synopsis needs updating.

**Provenance.** This corpus is a fork of the LengYue ADR corpus's authoring
discipline (LengYue is a spaced-repetition Go study tool; chocofarm is a
single operations-research Python package). The tenets transferred wholesale
with their instance lists re-derived against chocofarm; the frontend-specific
decisions were re-evaluated honestly (one rewritten for chocofarm's real
concern, one recorded as not-applicable). Numbering is preserved across all
eleven because chocofarm's code cites ADR-0002 and ADR-0004 by number. See
"How a fork consumes this corpus" at the end.

## ADR-0001: Immutability, Copy-on-Write, and Rebind-not-Mutate

**Decision.** Three named seams handle mutable-looking state: (1) the belief
world-set is immutable — every `filter_*` returns a fresh array, never an
in-place edit, so the simulator, eight solvers, and the dual bound share
belief primitives safely; (2) scenario and restriction are copy-on-write on
the env — `with_scenario` / `restrict` shallow-copy and alias the expensive
Tier-1 geometry (the ~4.5k-entry distance table, the 15,504-world array, the
44 faces), overriding only scenario/restriction fields and never mutating
`self`, so a value/K sweep is N shallow copies not N rebuilds; (3) the float32
inference cache is coherent by a rebind-not-mutate invariant over *all*
writers (identity-checked at read time), not a per-writer obligation.

**Why care.** This is chocofarm's analog of LengYue's `readonly` decision,
re-derived for plain Python (no reactive framework, no language-level
immutability). It is the decision that makes the heterogeneous-value
experiment — the project's stated next lever — cheap (a `Scenario`
comprehension, not a frozen-global monkeypatch), and it closes the
stale-weight cache hazard the audit reproduced (`max|Δp| = 0.0082`). A
contribution that caches a value-derived structure on the env, or mutates a
weight array in place without bumping the cache signature, fights this
decision.

## ADR-0002: Fail Loudly

**Decision.** When the system hits a deviation from its invariants —
unexpected shapes, timeouts, config drift, failed transport, violated
numerical assumptions — it surfaces it through the strongest applicable
channel: construction-time raise, then test/CI failure, then runtime
exception, then logged diagnostic, and silent fallback only when the fallback
is genuinely right (e.g. `env.d`'s bit-identical live-compute fallback).
Concrete rules: no silent fallback for real problems; validate at boundaries,
don't coerce; sentinel-instead-of-raise is a red flag; a config field the
receiver can't honor must not be silently accepted (the "lying signature"
failure); no silent state-mutation that breaks an invariant; a derived value
frozen as a literal that feeds a result is a latent silent failure (the
reference-rate / `%VoI`-divisor case).

**Why care.** This is the most consequential single tenet, and chocofarm's
code already cites it 16+ times — this ADR is the registry those citations
point at. It is why the parallel deadlock raises a loud diagnosable
RuntimeError instead of hanging, why the hp registry refuses a RESTART-field
change mid-run naming both values, why `with_scenario`/`restrict` raise on
malformed config, why the dtype guard and the AZ block-param shape checks
fail at load. On a research codebase the worst case is a silently-wrong
number that surfaces as a plausible result; this tenet exists to make that
loud instead.

## ADR-0003: Domain-Coupling Bands

**Decision.** A descriptive map of the codebase's coupling in three bands —
Band 1 solver-agnostic (the env/Policy inversion of control: `env` imports no
solver, a new method is a new `Policy` subclass with zero env edits); Band 2
OR-general (the belief mechanics, the Dinkelbach renewal-rate machinery, the
orienteering/routing, the AlphaZero stack, the dual bound, the analyzer — all
phrased over `worlds`/`value`/`N`/`K`, not over FFXIII facts); Band 3
FFXIII-bound (the instance data, the arrangement-face detector geometry, the
game-specific loaders) — plus a forward-looking two-question principle ("what
changes if the game differs but the OR problem is the same? if the OR problem
differs but the machinery is the same?"). Abstractions are extracted only when
a second concrete instance exists, not preemptively.

**Why care.** This is chocofarm's rewrite of LengYue's Go-portability ADR for
an OR package — the load-bearing analog is real. A different OR problem
replaces only Band 3 (instance + geometry) and keeps Bands 1–2 (the
overwhelming majority by line count); a different game with the same OR shape
also replaces only Band 3. The principle shapes how new modules are written —
an FFXIII fact goes in Band 3 and is isolated; a belief/rate concept goes in
Band 2 named for the problem class. The audit's E lesson (an abstraction built
then abandoned beside a live inline copy — `facemodel.SenseAction`) is the
caution against extracting too early.

## ADR-0004: Minimal-Touch Edits to Partially-Visible Files

**Decision.** When editing a file whose full source isn't immediately in view,
only change the specific lines the task or a failing test points at; a "while
I'm in here" full-file rewrite is forbidden under partial visibility. If a
broader rewrite is warranted, read the full file first.

**Why care.** chocofarm's source files carry numerical and structural
contracts the test suite only partially polices: the f64/f32/jax forward
equivalence (a reordered op can drift below tolerance on un-tested inputs); the
feature-layout positional contract (a sub-block reorder silently mislabels
feature-importance rows); the belief-mechanics duality the dual bound certifies
against; the single episode-horizon agreement across four sites. The risk
concentrates in the largest files — `decomp.py` (675L), `analyzer.py` (605L),
`registry.py` (715L) — exactly where a tool view truncates and an inferred
rewrite drifts a contract the editor couldn't see. chocofarm's code cites this
ADR by number (`netvalue_ismcts.py`).

## ADR-0005: Documentation Discipline

**Decision.** Nine rules for authoring documentation: (1) single source of
truth per nominal handle (the reference rates, the feature layout, the
horizon each get one owner, everything else derives); (2) consult/design/
agent/results/audit records live where the directory convention says
(consults in `docs/consults/`, the fix for the `consult-002` misfiling); (3)
descriptions describe relations not content snapshots, and a cited section
must exist (the report's real `§(4)`, never the dangling `§4`); (4) bodies
don't bare-name siblings where a rename breaks them; (5) file location
reflects content, repoint live referrers on a move; (6) author as you decide,
and status docs record slowly-aging decisions not a live task queue (the
24-seconds-stale handoff); (7) transitional docs sunset themselves; (8)
sibling-revisions / dated corrections over silent edits of point-in-time
records (the audit's not-retro-edited posture); (9) commissioned-review
artifacts are recorded verbatim, in-tree (the audit appendix; the consult
report pairs).

**Why care.** The 2026-06-15 architectural audit surfaced the recurring
failure: documentation written reactively decays into low-trust artifacts
faster than the code around it (the cited-but-nonexistent ADR registry, the
dangling `consult-002 §4`, the design-doc specifics STALE in the code, the
handoff stale in 24 seconds). This tenet names the discipline at the moment of
authoring so reconstruction cost stays bounded. Per ADR-0004's spirit, no
retroactive sweep — and point-in-time records (the audit, the agent
commissions) are explicitly NOT retro-edited.

## ADR-0006: Source-File Headers

**Decision.** Every Python source file carries a module-docstring header with
three parts: the module's path/area, a brief purpose statement, and a `Public
Domain (The Unlicense)` declaration. This codifies an already-de-facto
convention (the env, registry, schema, report, parallel headers all follow
it). `__init__.py` and data files are exempt.

**Why care.** A file pasted into a review, diff, or agent-report output
identifies itself — composing directly with ADR-0004's partial-visibility
discipline on the largest files, and the opposite of the audit's "Part A/B/C
as load-bearing identifiers" rot (a header that names path + purpose keeps a
file readable standalone). The per-file Public Domain declaration matters the
moment any single file is vendored or reposted. Per ADR-0004's spirit, no
retroactive sweep — headers accumulate as files cycle through normal editing.

## ADR-0007: File Size and Information Density

**Decision.** Source files target soft Python budgets — ≤ 300 lines typical,
≤ 400 for a single coherent unit where splitting would fragment cross-line
invariants. Density (decisions vs scaffolding) matters as much as size,
assessed qualitatively at review. Content rarely hand-edited (data tables)
may contract; numerical/decision logic never does — code golf in the belief
filter or a forward backend hides bugs behind dense lines and, given ADR-0004,
a dense line in a partially-visible file is exactly where a silent drift hides.

**Why care.** Large files are the condition under which ADR-0004's reactive
partial-visibility discipline has to apply; this tenet prevents the condition.
The oversized files are named as the refactoring queue (`decomp.py` 675L,
`analyzer.py` 605L, `registry.py` 715L, and others), several of which the
audit's own roadmap shrinks via content changes (the analyzer's presentation
split, `exit_loop`'s `RunConfig`). No retroactive sweep; incremental retrofit
composing with ADR-0004/0006. The density numeric thresholds are a review
heuristic, never measured — held qualitative honestly.

## ADR-0008: Classification Discipline

**Decision.** When a choice involves classification — picking from a closed
vocabulary (an enum-like choice, a detector action key, a severity tag),
placing a file in the `docs/` tree, naming a category — the choice is honest
only if the vocabulary precisely fits. Two registers: positive (refuse fuzzy
matches against an inadequate vocabulary; revise the vocabulary instead of
picking the closest fit) and negative (refuse to fabricate categories under
ambiguity; default to flat / named-as-incomplete). Severity is calibrated by
the substitution test — what the same failure shape would cost on a critical
surface, not the observed instance's cost. Four rules; two exceptions
(scheduled-for-revision misfit, deliberately-imprecise tag).

**Why care.** The chocofarm substrate is real. The detector mis-specification
(consult-002) is the positive-register failure: the original model keyed
sensing to *regions* — the closest available encoding, the wrong vocabulary —
and the fix re-keyed to *arrangement faces* (revise the vocabulary, don't pick
the closest fit). The deliberate `('d', i)` action-key preservation is the
honest reuse (the element still fit). The `instance.json` fossil arrays are the
negative-register failure (stale categorisation left standing). The audit's §4
reference-rate trace is the substitution test: a harmless frozen display
literal and a catastrophic frozen bound-input share one failure shape; the
discipline calibrates to the worst case.

## ADR-0009: Performance Investigation Discipline

**Decision.** A perf or equivalence claim — speedup, regression, null result,
or "the optimized path matches the baseline" — is honest only when its
investigation is captured reproducibly. The chocofarm tool surface is the
bench harnesses, not browser DevTools: `bench_hotpath.py` (per-component
timing on captured states, the regression guard), `bench_equivalence.py` (the
behavioral-equivalence harness), the f64/f32/jax forward test at `ABS_TOL =
1e-4`. The distinctive calibration is the **two-tier bar**: logic invariants
(illegal-slot mass) are asserted bit-exactly (`== 0.0`); float-sensitive
numerics (a rate under float32 + numba) are held to *aggregate behavioral
equivalence* (statistically indistinguishable rate / E[T] / action
distribution over N≥300 episodes, ≥2 seeds, within MC CI — never bit-equal,
which float32 can't be).

**Why care.** chocofarm already has this discipline (the `az-perf.md` /
`az-jax-perf.md` write-ups, the benches); this ADR names it as a tenet, swapping
LengYue's browser tool surface for chocofarm's ML/search one while keeping the
tenet. Confusing the two bars is the failure: pinning a float-sensitive rate
bit-exactly forbids a legitimate optimization (the ADR-0008 fuzzy/fossil
failure in the perf register), while relaxing a logic invariant to a tolerance
admits a real bug. The `ABS_TOL = 1e-4` contract is also what makes the audit's
forward-consolidation (R11) safe to attempt.

## ADR-0010: Render Locality and Canvas — Not Applicable (Lineage Entry)

**Decision.** None. LengYue's ADR-0010 is a Vue-SPA frontend tenet (where a
high-frequency reactive value may be read in a component tree; when a
data-dense visual must be a `<canvas>`). chocofarm has no UI — no Vue, no
component tree, no canvas, no DOM — so the tenet has no surface to apply to.
Per ADR-0008, the honest move is to record that it does not transfer rather
than invent a strained analog, and to keep the number stable so corpus
numbering stays aligned with the lineage (the code cites other ADRs by
number).

**Why care.** A contributor looking here for "the rule that prevents a silent
hot-path cost" should read **ADR-0009** — chocofarm's invisible-until-measured
costs live in the search/forward hot path, and ADR-0009 is the tenet that
carries that concern. This slot is a signpost, kept for numbering continuity.

## ADR-0011: Mechanization Discipline

**Decision.** A corrective-design tenet in four rules: (1) disciplines declare
their enforcement surface against a closed vocabulary (construction-time /
test-CI gate / write-time data constraint / run-time invariant / review-only);
review-only is legitimate but presumptively decaying, so declaring it makes
the choice challengeable. (2) Recurrence after a describing record converts to
a mechanism at the strongest feasible-and-proportionate surface, not more
prose. (3) Mechanisms adopt measure-first (a measured baseline before a check
goes to full strength — `ABS_TOL = 1e-4` chosen above observed float error;
the audit running the live env before claiming a metric stale). (4) Nets
quantify over the class, not the instance — an ownership slot, a name/shape
predicate, a derived-from-one-source invariant, never an enumeration that
fails open at the next instance. Status Proposed.

**Why care.** chocofarm's characteristic failure is the
invisible-at-authoring, visible-only-in-aggregate defect, against which policy
policed by one person's memory is structurally weak — the audit's whole
diagnosis. The worked proof is `FeatureLayout`: the three-writer feature-layout
triplication (the audit's sharpest landmine) was converted from a prose hazard
into one ordered block table the consumers read by name, with a fail-loud
partition check — the describing record named the hazard, the mechanism removed
it. Rule 1 is the enforcement register of the ADR-0002 / ADR-0008 / ADR-0009
unsubstantiated-claim family (an enforcement level is itself a claim, so it
must be declared). The audit's R-series roadmap is the chocofarm register of
Rule 2 (recurrence → mechanism).

## ADR-0012: Compositional and Structural Hygiene

**Decision.** A structural-design tenet for all *new* code — the incoming C++
runner first, the future async actor-learner second — stated as the positive
inverse of the 2026-06-15 audit's eight "architectural cancers." Nine checkable
principles: **P1** single-source-of-truth / derive-don't-duplicate; **P2**
seam/port discipline (the env↔Policy inversion of control as template; derived
state owned on the object whose lifetime it shares, never a module global keyed
by `id()`); **P3** no god-objects (one-owner collaborators — the
Transport⊥Pool⊥Task split); **P4** live-not-frozen (a value's heat is decided by
where it lives — a swept tunable is a live cell, not a ctor invariant); **P5**
fail loud / remove-the-root-cause-not-band-aid (distinguish a re-justified guard
from a patch on an undiagnosed substrate); **P6** substantiate equivalence/perf
claims (behavioral float32-equivalence, *not* byte-identity, for ML); **P7**
cross-language wire discipline (a cross-boundary fact — a layout, key, byte
format — has *one* authoritative definition; every side *derives* its view,
never re-authors it: two writers of one truth is the sin, not shared types, so
schema-driven codegen is encouraged; mechanically enforced at the strongest
feasible level — generate/compile-from-one-source > build-time lint > runtime
parity backstop — separating the serialization contract from the transport/
coordination mechanism, a bytes-store for state vs a messaging fabric for
coordination); **P8** typed signatures are the SSOT of a function's contract
(the call-boundary twin of P1 and of ADR-0002's no-lying-signature — an
annotation the body does not honor is a *lying signature*; the bar is
strict-where-achievable, each relaxation a named stub-gap not a convenience,
enforced by the mypy `--strict` CI gate ratcheting a monotonically-decreasing
baseline); **P9** functional core, imperative shell — the compiled-component
(C++) contract framed around *honest function signatures*: a computation is a
pure function of typed, bounds-carrying, const-correct inputs (`std::span<const
T>` over a raw `T*`) *returning its result by value* (free under guaranteed copy
elision / NRVO — the discipline costs no performance), with effects confined to a
thin imperative shell, the signature naming every mutation, and the *only*
sanctioned hidden mutation a *measured* hot-path buffer-reuse routed through an
explicitly-typed `Workspace`/`Context&` parameter; it outlaws the
*untyped-effectful void* (a raw-pointer-taking, `void`-returning,
out-parameter-writing black box — the compiled form of B / P2 / P8) **and the
exception** (the purest untyped effect — a control-flow escape absent from the
signature the caller is not forced to handle): failure is a typed return value,
`[[nodiscard]] std::expected<T, Error>` returned by value, never thrown (a
throwing ctor becomes a `create(…) -> std::expected` factory), so the error path
is declared in the return type and `[[nodiscard]]` makes ignoring it a compile
error (ADR-0002 fail-loud at its strongest surface), while the functional core
stays *total* (throw-free, neither throwing nor returning `expected`) and a
genuine invariant violation (a bug) remains an `assert`/abort, not an `expected`.
The C++ `NetForward` MLP (`predict(const float* X)`, the `void matvec_bias(…,
std::vector<float>& out)` internals, the throwing constructor) is the cautionary
instance — every `cpp/src` throw is at a boundary (redis I/O, instance load, the
manifest-validating ctor), none on the throw-free forward/search core. P9 is now
a mix: the error/`[[nodiscard]]` axis is **compile-enforced** (an unhandled
`std::expected` fails the build), the input/output/mutation rules review-policed
with the compiler `-Wall -Wextra` and a future `clang-tidy` config as the
mechanization surface. It composes with rather than
restates its siblings (0002/0004/0005/0007/0009/0011 cited, not re-derived), and
carries a dedicated concrete C++ wire contract (the `transport.py` keys/dtypes,
parity under the P6 bar).

**Why care.** The audit's deepest finding was that chocofarm *already proved it
knows the right answer* (live λ, derived dimensions, the env↔Policy seam) and
then applied that discipline once and stopped — the cancers are the right idea
not propagated. This tenet is propagation by default, written ahead of the next
large bodies of new code so they are born clean rather than audited dirty; it is
upstream of ADR-0011 (structure born clean is structure no mechanism must later
convert). It binds new structure at authoring time, mandates no retroactive
sweep (the R-series owns existing cleanup, ADR-0004), and gives a C++ author an
implement-against-this contract that keeps the language boundary a single seam.

## How to read these together

The tenets form a coherent posture:

- **ADR-0002** says fail audibly when invariants break.
- **ADR-0004** says don't introduce silent failures by editing blind.
- **ADR-0005** says don't let documentation drift into silent failures of its
  own.
- **ADR-0006** says individual files identify themselves to reduce the cost of
  partial-visibility editing.
- **ADR-0007** says keep files small enough that partial visibility is the
  rare case, not the default.
- **ADR-0008** says refuse fuzzy matches against an inadequate vocabulary and
  refuse synthetic fabrications under ambiguity — classify only when the
  classification is honest.
- **ADR-0009** says perf and equivalence claims are a closed vocabulary that
  must be substantiated by captured, reproducible investigation, not author
  intuition — with the bit-vs-behavioral two-tier bar applied per the
  quantity's kind.
- **ADR-0010** is a lineage signpost (no chocofarm rule); it points to
  ADR-0009 as the nearest concern.
- **ADR-0011** says a discipline declares how it is enforced, and a recurrence
  converts to a mechanism rather than more prose — because policy policed by
  one person's memory decays, and only mechanical nets hold.

ADR-0002, ADR-0008, and ADR-0009 form a family of unsubstantiated-claim
disciplines at three intervention points: ADR-0002 is the reactive register
(when invariants break, surface); ADR-0008 is the proactive classification
register (when categorising, refuse fuzzy matches and synthetic fabrications);
ADR-0009 is the per-domain instance for the performance/equivalence vocabulary
(when asserting perf or equivalence, attach the substantiation). **ADR-0011
Rule 1** is the enforcement register of that family: an enforcement level is
itself a claim about a discipline, so it must be declared rather than implied.

The two structural records — ADR-0001 (a decision) and ADR-0003 (a domain
map) — describe specific structural choices that shape how the tenets get
applied. ADR-0001's copy-on-write and rebind-not-mutate seams are the
discipline ADR-0002 verifies at the env's boundaries; ADR-0003's Band-1
env/Policy seam is the structure the whole solver toolkit and the AZ stack
hang off, and the Band-2 OR machinery is what ADR-0007's file budgets and
ADR-0005's documentation organization ultimately serve.

A contribution against the grain of any one of these will cause friction
wherever it touches the others.

## How a fork consumes this corpus

This corpus is itself a fork — of the LengYue ADR corpus — and the adaptation
that produced it is the worked instance of how a *further* fork would consume
chocofarm's:

- **Tenets transfer wholesale, re-deriving instance lists.** ADR-0002, 0004,
  0005, 0006, 0007, 0008, 0009, 0011 transferred from LengYue as universal
  disciplines; only their Go/Vue/frontend examples were swapped for chocofarm
  ones (the env/Policy seam, the fail-loud cases, the benches, the
  `FeatureLayout` mechanism). A fork of chocofarm does the same: keep the
  decision and rationale, re-instance the examples against the fork's reality.

- **Structural decisions and maps re-evaluate against the fork's context.**
  ADR-0001 (a decision about *this* codebase's seams) and ADR-0003 (a map of
  *this* codebase's coupling) do not transfer as settled — a fork re-derives
  them. LengYue's ADR-0001 (Vue `readonly`) and ADR-0003 (Go-portability)
  did not survive the move to an OR package; they were rewritten. A fork of
  chocofarm should expect the same: ADR-0003's bands are a transfer *map* to
  read once and then supersede with the fork's own.

- **Frontend-specific decisions are re-evaluated honestly, not strained into
  applicability.** LengYue's ADR-0010 (render locality / canvas) has no
  chocofarm surface; it is recorded as not-applicable (a lineage entry
  pointing to ADR-0009) rather than fabricated into a fake OR analog — exactly
  the ADR-0008 synthetic-fabrication refusal. A fork inherits this honesty:
  an inherited ADR with no surface in the fork is recorded as not transferring,
  not bent to fit.

- **Numbering is preserved.** chocofarm's code cites ADR-0002 and ADR-0004 by
  number; the full 0001–0011 numbering stayed aligned with the LengYue lineage
  so those citations (and future ones) remain stable. A fork that inherits
  chocofarm's numbered citations preserves the numbers the same way.

- **Infrastructure named by a tenet is re-instantiated, not inherited.**
  Where LengYue's tenets named umbrella infrastructure chocofarm lacks (a
  Postgres work-status store, a dispatch ledger, a doc-graph validator), the
  rules were re-stated in chocofarm's actual form (the directory convention,
  review-only enforcement) — never inherited as if the infrastructure existed.
  A fork checks each tenet's mechanism (ADR-0011's enforcement-surface
  declarations are the transfer manifest) and re-instantiates what its tree
  does not already provide.

New fork decisions continue the numbering with their own records.
