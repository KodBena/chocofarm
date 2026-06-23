# ADR-0000: The Alpha and the Omega — Type-Driven Design as the Foundational Law

- **Status:** **Provisional (emergency).** Filed under thin executive bandwidth.
  This ADR is numbered `0000` because it is the **root** the corrective trio
  ADR-0011 / ADR-0012 / ADR-0013 descend from — the **Alpha** (the first
  principle every design proceeds from) and the **Omega** (the final court of
  appeal a contested decision returns to). It is filed at provisional strength
  because the *absence* of a stated foundational guideline caused serious harm
  and real financial loss on the substrate below, and the cost of waiting for a
  polished ratification exceeded the cost of an imperfect-but-substantive record
  now. It is deliberately imperfect and **invites refinement**; it is a court of
  first resort, not a finished constitution. (Per ADR-0011 Rule 1 its honest
  enforcement surface is named explicitly, and per ADR-0008 its genre tension
  with the trio it parents is flagged, not resolved away — see "Revisit when…".)
- **Genre:** Tenet (foundational / root — the *zeroth* tenet). It is the meta-frame
  the structural trio instantiate: **ADR-0012** is its *shape* (types — the typed
  signature is the SSOT, illegal states unrepresentable), **ADR-0011** is its
  *operational net* (mechanism — a recurrence converts to a check, not more prose),
  **ADR-0013** is its *integrity* (apply the real fix, to its ratified end, not a
  patch around the symptom). This ADR does not restate those three; it names the
  single root question they are three answers to, and binds the contributor to ask
  it *first*. (The genre overlap with that trio is the ADR-0008 tension flagged below.)
- **Date:** 2026-06-24
- **Provenance:** Native to chocofarm — not a transferred universal. It arises from
  a **named, dated, first-person session** on the `throughput-lab` testbed
  (`throughput-lab/`, the producer↔server↔consumer leaf-evaluation loop), in which
  the maintainer **repeatedly redirected the executor** from the reflex *"how do I
  fix it"* to the question *"what **type** would have made this defect
  unrepresentable?"* — and each time, a refined type was the right answer. The
  episode is the substrate; the four specimens in Context are its evidence. Like
  ADR-0013, this ADR is filed at the strength it is precisely because the substrate
  is not hypothetical: the absence of the rule had a measurable price. It is a
  point-in-time record of that session (ADR-0005 Rule 8) and is not retro-edited.
- **Scope:** Every contributor — human or LLM — at the moment a **defect is
  identified** anywhere in the `chocofarm/` package, its testbeds, its docs corpus,
  or any new-language component that joins it. It binds the *first move* after a
  defect surfaces, before any fix is authored. It governs *posture*, not a specific
  code shape (ADR-0012 owns shape) and not a specific enforcement mechanism
  (ADR-0011 owns mechanization) — the posture it governs is **which question is
  asked first**, and the trio it parents supplies the answers.

A word on register, in ADR-0013's key and for the same reason. The rest of the
corpus is neutral; the trio it parents earns an edge from dated failures, and this
root inherits that edge because its own substrate is a dated failure too — the
omission of this very rule. The disdain such a record carries is for the *conduct*
(the reflex to patch the symptom and move on), never the contributor; it is the
disdain a competent practitioner reserves for their own first, lazy instinct. The
rule exists because that instinct is universal and presents to the agent's own
judgement as efficiency.

## Context

The corpus already contains this law — distributed across three siblings, each
owning one facet, none naming the root they share. ADR-0012 P8 says *the typed
signature is the single source of truth of a function's contract*; the
anti-pattern checklist's spine is *make illegal states unrepresentable*. ADR-0011
Rule 2 says *a recurrence converts to a mechanism, not more prose*. ADR-0013's
amendment (2026-06-24) says *fair dealing runs both ways* — apply the real fix, to
its ratified end, and neither narrow nor maliciously comply. These are three
answers. **This ADR names the one question they answer**, because a session that
does not ask the question reaches for none of the three answers — it reaches for a
patch.

The danger is not the absent discipline in the cartoon sense (a contributor who
does not know about types). It is the **reflex**, present in the most competent
practitioner: when a defect detonates, the mind goes immediately to *"how do I make
this specific failure stop."* That reflex is locally correct and globally
catastrophic — it fixes the instance and leaves the **class representable**, so the
next instance is one edit away, exactly the *enumeration-fails-open-at-the-next-
instance* failure ADR-0011 Rule 4 names. The substrate proves the reflex recurs in
the diagnostician, and proves the alternative — *ask what type prevents the class* —
was right every single time it was asked.

### Specimen 1 — the oversize/wrong-width/wrong-dtype wire frame → `BoundedBatch`

A producer emitted a leaf-evaluation batch onto the wire that was oversize (more
rows than the server's `max_batch`), or the wrong width (a feature dimension the
server did not expect), or the wrong dtype. The frame was structurally legal as
*bytes* and detonated **three layers downstream** — past the wire, past the
coalescing intake, inside the server's forward — where the diagnosis was furthest
from the cause (the ADR-0002 hierarchy's whole point: fail at construction, not deep
in the first forward). The *"how do I fix it"* reflex produces a downstream guard:
a length check at the forward, a clamp, a defensive reshape. The **right** answer
was the question's answer: a refined wire type — **`BoundedBatch`** — whose validator
**makes the illegal shape unrepresentable at the boundary** (the Port/ACL of
ADR-0012 P2: a boundary *translates-and-validates*, it does not coerce). A
`BoundedBatch` that cannot be constructed from an over-`max_batch`, wrong-width, or
wrong-dtype buffer cannot reach the forward at all. The defect class is gone, not
guarded — ADR-0012's *illegal states unrepresentable* made concrete at the wire.

### Specimen 2 — the cross-layer counter category error → `CellLedger`

A health check compared counters drawn from three different layers — producer-batch
counts, wire-message counts, and consumed-row counts — as if they were one currency,
and **mis-flagged healthy cells** because the three are not commensurable (one
producer batch is N wire messages is M rows). This is precisely an **ADR-0008
category error**: a fuzzy match across an inadequate vocabulary, three distinct
units read as one. The *"how do I fix it"* reflex adds a fudge factor, a tolerance,
a special case for the cell that mis-flagged. The **right** answer was a type that
makes the only-meaningful comparisons the only-expressible ones: a **`CellLedger`**
reconciliation type that *owns* the three counters as distinct, typed quantities and
exposes exactly one verdict — so a cross-currency comparison is not a bug to catch
but a sentence you cannot write. The vocabulary was revised (ADR-0008's positive
register), structurally, at the type.

### Specimen 3 — the unbounded producer send queue → a byte-budgeted high-water-mark

A producer's send queue had no bound; under backpressure it grew until the process
was **OOM-killed at ~7 GB**. The *"how do I fix it"* reflex caps the queue at a
round number of *messages* (1000? 10000?) — a magic constant strewn as a bare literal
(ADR-0012 cancer F), arbitrary because the thing that actually exhausts memory is
*bytes*, not message count, and messages vary in size. The **right** answer was a
type whose bound is **derived from the one source that makes it meaningful**: a
**byte-budgeted high-water-mark** computed from the message size (ADR-0012 P1,
derive-don't-duplicate — the bound has one home and is computed, not guessed). The
queue refuses the write that would exceed the byte budget, loudly (ADR-0002), at the
boundary — and the OOM class is unrepresentable, not merely less likely.

### Specimen 4 — the unbounded coalescing intake → a bounded blocking queue

The server's coalescing intake (which gathers producer messages into a microbatch)
was likewise unbounded — the same disease at the consuming end. The **right** answer
was the same *kind* of answer: a **bounded blocking queue** whose capacity is a typed
invariant of the structure, so an intake that would overflow **blocks** (applying
backpressure) rather than growing without bound. The pattern across all four is one
pattern: every defect was a **design signal**, and the durable fix was a **type**,
not a patch.

### The contrast specimen — "fix the one blocking call"

The instructive negative is the reflex itself, caught in the act. Confronted with a
blocking call in the serve loop, the executor asked *"how do I fix this one blocking
call"* — and produced **two successive incomplete patches**, because each patch fixed
the instance in view and left the *shape* that permits the class untouched, so the
class re-surfaced one call over. Had the first move been *"what shape prevents a
blocking call from sitting on this path at all"*, one structural answer would have
closed it once. This is the same root as ADR-0013's attrition specimens — the patch
that asks "how to fix" instead of "what shape prevents this class" is the *execution*
sibling of the cut corner: it does the visible work and forfeits the durable work.

### Why this is the root, and why it is filed now

The four specimens are four instances of one missing reflex. The maintainer had to
inject the question by hand, repeatedly, because no document carried it — the trio
that *answers* it (0011/0012/0013) presumes it has already been asked. The cost of
that omission was the OOM kill, the mis-flagged cells, the three-layer detonation,
and the doubled patch — real time and real money. ADR-0013's edge is "finish what
was ratified"; ADR-0012's is "born in the right shape"; ADR-0011's is "mechanize the
recurrence." **None of them fires unless the contributor first asks the root
question.** This ADR makes the question mandatory and first.

## Decision

We adopt **Type-Driven Design** as the foundational law of the codebase, in three
rules. The spine is one sentence: **all design is determined first by its
types/contracts, and when a defect is identified the FIRST move is not "how do I fix
it" but two prior questions — (a) what type would make this defect class
unrepresentable, and (b) what operational lapse let it recur — answered with the
shape ADR-0012 mandates, the mechanism ADR-0011 mandates, and the integrity ADR-0013
mandates; a patch authored without first asking the two questions is itself a defect.**

Each rule names its enforcement surface in ADR-0011 Rule 1's closed vocabulary
(construction/import-time · test/CI gate · write-time data constraint · run-time
invariant · review-only), honestly — the two-question reflex is largely review-only,
and saying so is the point.

### Rule 1 — All design is type-driven (the Alpha)

The shape of correct code is determined **first** by its types and contracts, not by
its control flow. A function's contract has one home — its **typed signature**
(ADR-0012 P8); a fact has one home — the type that owns it (ADR-0012 P1); a boundary
is a type that *translates-and-validates* and refuses what it cannot honor (ADR-0012
P2). The design question precedes the implementation question: *what are the types
such that the illegal states this code must never enter are unrepresentable?* This
is ADR-0012's domain in full — this rule does not restate it, it **elevates it to
first**: types are not documentation of a design arrived at by other means; they
*are* the design, and the implementation is their consequence.

*Enforcement surface: composes with ADR-0012's declared surfaces — **test/CI gate**
where the type is mypy-checkable (P8's `mypy --strict` ratchet) or compile-enforced
(P9's `[[nodiscard]]`), **construction/import-time** where a boundary's strict decode
raises (the Port/ACL), **run-time invariant** where a derived-partition check fires
(`FeatureLayout`), and **review-only** for the design judgement of *which* type to
reach for before any of those gates exist.* This rule mints no new mechanism; it
binds the contributor to ADR-0012's, applied first.

### Rule 2 — On a defect, ask the two questions before authoring any fix (the reflex)

When a defect is identified, **before** a single line of fix is written, ask, in
order:

**(a) "What type or typing discipline would have made this defect class
unrepresentable?"** Name the failure in its most general form (ADR-0008's
substitution test: calibrate to the class, not the observed instance), and ask what
type forecloses the class. If a robust, well-typed *architectural* answer exists — a
refined wire type, a reconciliation type, a derived-from-one-source bound, a bounded
structure — it is **applied in full**, with the professional integrity ADR-0013
mandates: the real fix, to its ratified end, **not** a slipshod patch tacked onto the
special case, and **not** a downstream guard that leaves the class representable
upstream. This is the constructive composition of ADR-0012 (the *shape* the answer
takes) and ADR-0013 (the *integrity* to carry that shape to completion rather than
patch around it). The four specimens are the worked instances: each (a)-answer was a
type, and each closed a class.

**(b) "What operational lapse let this happen?"** Read *operational* as **executive**:
this question is aimed at the **maintainer/ratifier**, NOT the implementer. It asks what
*the executive* failed to put in place — the guideline, the ADR, the typing discipline,
the mechanism — that would have rendered the class unrepresentable or caught it loudly.
A defect that *recurred* is a signal that the *net* failed: the discipline was
review-only where it should have been mechanized. **When ADR-0011 is violated it is
structurally the executive's to own**, because the executive owns the
enforcement-surface — a recurrence that was never mechanized is a guard the maintainer
did not build, not an implementer who erred. (This composes with the codebase's standing
posture that named failure modes are *organizational, not personal*: question (b) is
self-directed accountability, not a hunt for fault downstream.) Convert the recurrence
into a **mechanism** that makes the class **fail loudly** (ADR-0002), at the strongest
feasible-and-proportionate surface, per **ADR-0011 Rule 2** — a validator at the
boundary, a ratcheting baseline, a build-time lint — quantifying over the *class*, not
the instance (ADR-0011 Rule 4). Question (a) gives the type; question (b) gives the net
that keeps the type honest as the tree grows — and the very act of authoring this ADR,
ADR-0011, and ADR-0014 is the maintainer answering (b) about themselves.

A fix authored **without** first asking (a) and (b) — a patch that stops the symptom
while leaving the bug-class representable — is **itself a defect** under this rule,
and is rejected on the same footing as the bug it papered over. (This is the
contrast specimen's lesson: the doubled patch was two defects, not one fix.)

*Enforcement surface: review-only for the asking; the **answers** inherit mechanized
surfaces.* No mechanism reads intent, so the *act* of asking the two questions is
review-policed, with maximal self-suspicion — exactly ADR-0013 Rule 3's posture
toward the prudent-sounding reflex (the "I'll just guard it downstream" demurral is
the tell, not the argument). But an (a)-answer that is a type lands on ADR-0012's
gates (mypy `--strict`, the boundary raise, the partition invariant), and a
(b)-answer that is a mechanism lands on ADR-0011's surface — so the *outputs* of the
two questions are as strong as the trio's machinery, even though the *trigger* is
attention. Per ADR-0011 Rule 1, this is stated, not hidden: **the reflex is
review-only; its answers are not.**

### Rule 3 — The two questions are facets of one root; the trio are its three answers

The contributor does not choose *between* type, mechanism, and integrity — a complete
disposition of a defect carries all three: the **shape** (ADR-0012 — the type that
makes the class unrepresentable), the **net** (ADR-0011 — the mechanism that makes a
recurrence loud), and the **integrity** (ADR-0013 — the real fix applied to its
ratified end, neither narrowed nor maliciously complied). When the trio appear to
conflict, this root is the court of appeal: the question *"what shape, fully applied,
makes this class impossible"* dissolves most apparent conflicts, because a real type
that forecloses a class is simultaneously the shape (0012), the loud net (0011 — an
unconstructable illegal state is the loudest possible failure, at construction time),
and the complete fix (0013 — a class foreclosed is not a corner cut).

*Enforcement surface: review-only.* This is a framing rule — the recognition that a
defect's disposition is one act with three facets, not three competing options. It is
policed by the reviewer reading a fix against all three, and by the contributor
declining to treat "I typed it" as discharging "I mechanized the recurrence" or "I
finished the work."

### The escape hatch — ADR-0014 (when the reflex itself stalls)

The two-question reflex presumes the contributor *can* see the type that forecloses
the class. Sometimes they cannot — the right type is genuinely not visible, and
grinding produces a guess or a patch. **That inability is itself a trigger**, named by
the sibling **ADR-0014 (request a second opinion when a problem resists resolution),
authored in parallel with this ADR.** When the two-question reflex stalls — when (a)
yields no type and (b) yields no mechanism after honest effort — the disposition is
**not** to ship a patch and move on (that is the very reflex this ADR rejects); it is
to get an **independent second pair of eyes** (ADR-0014), exactly as ADR-0013 Rule 3
leans on the out-of-frame hack-rationalization check rather than self-certification.
A stall is a signal, not a license to patch.

## Consequences

### Positive

- **The bug-class dies, not the bug-instance.** The whole purpose: a defect
  disposed of under Rule 2 closes the *class* (a type that makes it unrepresentable)
  rather than the instance (a guard one edit from the next occurrence). The four
  specimens are four classes foreclosed; the contrast specimen is the doubled cost of
  not asking.
- **The trio gain their missing premise.** ADR-0011/0012/0013 each presume the
  contributor has already asked "what shape prevents this class." This root makes that
  presupposition explicit and first, so the three answers are actually reached for
  rather than skipped past to a patch.
- **The disposition is complete by construction.** Rule 3 makes "I typed it" /
  "I mechanized it" / "I finished it" three facets a reviewer checks together, so a
  fix that satisfies one and silently drops the others is visible as incomplete.

### Negative

- **The reflex is review-only and enforced by the faculty it corrupts.** Exactly
  ADR-0013 Rule 3's honest admission, inherited: the *act* of asking the two questions
  before patching is policed primarily by the contributor's own in-the-moment
  recognition, and the patch-reflex presents to that judgement as efficiency. The only
  external backstops are the reviewer reading a fix for a foreclosed class vs a guarded
  one, and ADR-0014's second-opinion escape hatch. Stated, not hidden (ADR-0011 Rule 1).
- **Higher up-front cost on a defect, on purpose.** Asking "what type forecloses this
  class" and applying it in full is slower than a downstream clamp — the same
  policy-vs-mechanism cost ADR-0011/0012/0013 carry, here paid at the moment of
  triage. It is the cost the four specimens prove is cheaper than the alternative.
- **Risk of weaponization into over-typing.** A bad-faith reading could wield "all
  design is type-driven" to demand a bespoke type for every trivial value, or to block
  a genuinely-right patch. The Exceptions carve the legitimate cases; the discriminator
  is whether a *class* is at stake — a type earns its place by foreclosing a class, not
  by existing.

### Neutral

- **No new mechanism is minted here.** This root binds the contributor to the trio's
  existing machinery (ADR-0012's mypy `--strict` ratchet and boundary raises,
  ADR-0011's mechanization triggers, ADR-0013's artifact-verification) applied in the
  right order; it commissions no gate of its own. Its protection is the *question*, not
  its prose.
- **No retroactive sweep, and no conflict with minimal-touch.** This ADR binds the
  *first move on a newly-identified defect*; it does not license roving the tree for
  defects to re-type (ADR-0004 / scope discipline). "Ask the two questions when a defect
  surfaces" is not "hunt for defects to retype."

## Exceptions

These are the *honest* not-a-type dispositions — distinguished from the patch-reflex
by a single discriminator: **no bug-class is at stake, or the class is genuinely
filed, not silently left.**

- **No class at stake.** A genuine one-off — a trivial typo, a value that cannot
  recur and threatens no class — is fixed directly. The two questions are asked and
  *answered "no class here"*; the discipline applied to itself returns "patch is the
  right disposition." The tell of the *violation* is a recurring shape (ADR-0011 Rule 4)
  dressed as a one-off.
- **The type is real but its blast radius is deferred — filed, not buried.** When the
  (a)-answer is a real type whose introduction is large enough to defer, the deferral is
  filed where deferrals live (`BACKLOG.md`) with the misfit marked at the site
  (ADR-0008 Rule 3 / ADR-0002), per ADR-0013 Rule 4 — a *filed* deferral, never a
  narrated-and-left one. The class is named even when its fix is deferred.
- **The stall, escalated.** Rule 2's reflex genuinely yields no type after honest
  effort. The sanctioned disposition is ADR-0014's second opinion, not a patch shipped
  as if the question were answered.

What is **never** an exception: the in-the-moment sense that a downstream guard is
"good enough" because finding the foreclosing type is "more work than it's worth." That
sense is the patch-reflex this ADR exists to overrule, and it is the same
scale/minimality/"for now" tell ADR-0012 P7/P8/P9 and ADR-0013 Rule 3 already name and
reject.

## Revisit when…

1. **The ADR-0008 genre tension is resolved (the open, honest question).** This ADR
   **may strain ADR-0008 (classification discipline)**: its genre overlaps the existing
   0011/0012/0013 trio, and it is a fair question whether *"the root they descend from"*
   is a **distinct ADR** or merely a **meta-frame** that should live as a synopsis
   section or a preamble rather than a numbered record. Filing it as `0000` is itself a
   classification choice, and an honest reading must concede it is **not obviously
   crisp** — the boundary between "a tenet" and "a frame over three tenets" is exactly
   the kind of vocabulary-fit question ADR-0008's positive register asks. This ADR
   **does not resolve that tension**; it flags it, per ADR-0008 Rule 3 (surface the gap
   visibly) and per the provisional posture (an emergency record names its own
   imperfection rather than papering it). Revisit when executive bandwidth allows a
   considered ruling: ratify `0000` as a genuine root, fold it into the synopsis as a
   frame, or split it. Until then it stands as a provisional, flagged-as-imperfect
   record — which is the honest disposition, not a defect.
2. **A rule introduces its own failure mode** — most plausibly Rule 1 weaponized into
   over-typing (a bespoke type for every triviality), or Rule 2 hardening into a ritual
   that blocks a genuinely-right direct fix. Flag the offending rule here by dated
   amendment (ADR-0005 Rule 8).
3. **A recurrence mints a mechanism for the reflex itself** (ADR-0011 Rule 2). If a
   pattern of patch-first-ask-never recurs and a check can catch it — a review-checklist
   that a defect-fix PR names its foreclosed class, an out-of-frame
   rationalization-detector run on the fix's justification — record it here and tighten
   Rule 2's surface from review-only toward the gate.
4. **ADR-0014 lands and its trigger boundary needs reconciling.** This ADR names
   ADR-0014 as the escape hatch for a stalled reflex; once ADR-0014 is ratified, confirm
   the hand-off (stall → second opinion) is described consistently from both sides, and
   repoint if its number or framing shifts (ADR-0005 Rule 3/5).
5. **A second OR/game instance adopts the corpus** (ADR-0003's trigger). Confirm this
   root transferred as *posture* — its substrate is the dated `throughput-lab` session,
   local and first-person, so a fork re-anchors it to its own dated defects exactly as
   ADR-0013 must.

## Related

- **ADR-0012 (compositional and structural hygiene).** The **shape** facet. P8 (the
  typed signature is the SSOT) and the *illegal-states-unrepresentable* spine are
  Rule 1's content and Rule 2(a)'s answer-form; P1/P2 are the SSOT and Port/ACL the four
  specimens' types instantiate. This ADR elevates ADR-0012 to *first*; it does not
  restate it.
- **ADR-0011 (mechanization discipline).** The **net** facet. Rule 2(b) is ADR-0011
  Rule 2 (recurrence → mechanism) named as the second of the two mandatory questions;
  Rule 4 (quantify over the class) is why a foreclosing *type* beats a guard on an
  *instance*. Rule 1's enforcement-surface vocabulary is the one this ADR declares its
  rules against.
- **ADR-0013 (execution integrity).** The **integrity** facet. Rule 2(a)'s "applied in
  full, not a slipshod patch" is ADR-0013's mandate that the ratified fix is owed to its
  ratified end; the contrast specimen (the doubled patch) is the *type-driven* sibling of
  ADR-0013's attrition specimens. The 2026-06-24 amendment (fair dealing both ways) is
  the integrity that keeps Rule 2 from being either narrowed or maliciously complied.
- **ADR-0002 (fail loudly).** Rule 2(b)'s mechanism makes the foreclosed class fail
  loudly; a type that makes an illegal state unconstructable is fail-loud at its
  strongest surface (construction-time), the top of ADR-0002's hierarchy. The four
  specimens' boundary refusals (the `BoundedBatch` validator, the byte-budgeted queue's
  loud refusal) are ADR-0002 mechanisms.
- **ADR-0008 (classification discipline).** Two ways: Specimen 2's counter category
  error is an ADR-0008 positive-register failure (the `CellLedger` *revises the
  vocabulary*), and this ADR's own genre is an ADR-0008 tension it flags rather than
  resolves (Revisit #1).
- **ADR-0014 (request a second opinion when a problem resists resolution) — authored in
  parallel.** The escape hatch when the two-question reflex stalls: a type that will not
  reveal itself is a trigger for an independent second pair of eyes, not for a patch.
- **The `throughput-lab` testbed** (`throughput-lab/`). This ADR's direct substrate —
  the producer↔server↔consumer leaf-evaluation loop on which the four specimens and the
  contrast specimen occurred.

## What this tenet does NOT mean

- **Not "type everything, always, maximally."** A type earns its place by foreclosing a
  **class**; a bespoke type for a triviality that threatens no class is the over-typing
  weaponization Revisit #2 names, not the discipline. The two questions include the
  honest answer "no class here — a direct fix is right."
- **Not "never patch."** A genuine one-off with no class at stake is patched directly
  (Exceptions). The discipline is to *ask first* whether a class is at stake, not to
  forbid every small fix.
- **Not a license to re-type the whole tree.** This binds the *first move on a
  newly-surfaced defect*, not a roving retype of existing code — ADR-0004 and scope
  discipline are untouched.
- **Not self-certifying.** Per ADR-0011 Rule 1, this ADR expects its own prose to be
  exactly as weak as it says: the two-question *reflex* is review-only and enforced by
  the faculty it corrupts. Its protection is the trio's mechanized *answers*, the
  reviewer's check against all three facets, and ADR-0014's out-of-frame escape — not
  the contributor's good intentions, which (per ADR-0013 Specimen 2) the diagnostician
  had in full and which failed in minutes.
- **Not a finished constitution.** It is filed provisional, under thin bandwidth,
  because its absence cost real money; it invites refinement and flags its own ADR-0008
  genre tension rather than pretending the classification is clean (Revisit #1).

## License

Public Domain (The Unlicense).
