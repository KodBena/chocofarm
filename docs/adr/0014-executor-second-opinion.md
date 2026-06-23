# ADR-0014: Request a Second Opinion When a Problem Resists Resolution

- **Status:** Provisional (emergency). Filed under thin executive bandwidth in
  the immediate wake of a session in which the *absence* of this guideline cost
  real work and real money — the executor locked onto one diagnosis and burned
  successive attempts on the wrong target while the actual fault sat one reframe
  away. It is deliberately imperfect and invites refinement; it is **not** a
  stub. The provisional status is itself honest in the ADR-0011 Rule 1 register:
  the harm was concrete, the mechanism is partial, and saying so plainly beats
  letting the lesson decay into prose nobody enforces.
- **Genre:** Tenet (cross-cutting execution discipline) — **ADR-0011
  (mechanization discipline) turned INWARD on the executor's own process**.
  ADR-0011's thesis is that *a recurrence converts to a mechanism rather than
  more prose, at a declared enforcement surface*. ADR-0013 already instantiated
  that to one failure mode — the *attrition of will to finish*. This tenet
  instantiates it to a **sibling** failure mode: the *anchoring of reasoning that
  will not finish because it cannot escape its own frame*. Where ADR-0013 guards
  the will to carry a mandate to its end, this tenet guards the **ability to see
  the problem freshly** when the executor's own line of attack has demonstrably
  stalled. The recurring thing ADR-0011 keys on is, here, the executor's *own
  repeated mis-targeted attempts at one problem*; the mechanism the recurrence
  mints is the **structured second-opinion request**.
- **Date:** 2026-06-24
- **Provenance:** Native to chocofarm. Not a transferred universal — a response
  to a **named, dated, first-person failure** on the `throughput-lab` testbed,
  recorded in Context below. The substrate is an LLM executor's own session
  record, in the same first-person register ADR-0013 Specimen 2 established as
  the most damning kind of evidence: the failure is in the *diagnostician*, not
  merely narrated about someone else. This ADR is filed at provisional strength
  precisely because the substrate is real and the cost was real, but the
  bandwidth to harden it was not yet available.
- **Scope:** Every executor — human or LLM, but **with LLM-specific force** (see
  Context) — at the moment a problem **resists resolution**: a fix is attempted,
  proves to have addressed the wrong target, and the failure *recurs* under a
  second or third attempt that stays inside the same frame. It binds the conduct
  of *getting unstuck*, not the shape of the code (ADR-0012) nor the will to
  finish (ADR-0013); the conduct it governs is *whether the executor, having
  demonstrably locked onto one line of reasoning, fetches an independent frame
  before burning a third, fourth, fifth attempt on the same wrong target*.

A word on register, so it is not mistaken for tone. This is a **license, not a
mandate** (Decision, Rule 1). It grants permission and encouragement to escalate
to a second pair of eyes when genuinely stuck; it does **not** demand a subagent
for every problem, and it does **not** excuse offloading thought one could have
done. Both abuses — ego-locking onto a failing line *and* reflexively spawning a
fresh agent to avoid thinking — are failures, and the second is exactly the
*malicious compliance* ADR-0013's 2026-06-24 amendment forbids, here in the
opposite direction. The thing this tenet rewards is the unglamorous professional
move: **noticing you are stuck, and asking.**

## Context

### The failure shape

The executor's characteristic stall is not the absence of effort — it is
**anchoring**: locking into a single line of reasoning and failing to escape the
frame even as the evidence accumulates against it. An LLM executor is
*especially* prone to this. Having committed to a diagnosis, it walks the
garden path it has already paved: each new attempt is a refinement *within* the
first frame ("it must be a different blocking call than the one I just fixed"),
never a *reframe* of the problem ("maybe the problem is not a blocking call at
all"). The faculty that would notice the frame is wrong is the same faculty that
built the frame — so it does not notice. This is the structural reason a
*second* opinion has value that a longer *first* opinion does not: its worth is
**independence** — a fresh frame that has not walked the same path and is not
invested in the same diagnosis.

This is the same root ADR-0011 names — *the invisible-at-authoring,
visible-only-in-aggregate defect, against which policy enforced by one
attention is structurally weak.* A locked line of reasoning is invisible from
inside the lock (each step feels like progress) and visible only in
aggregate (three attempts, same wrong target). ADR-0011's remedy is the right
one: a mechanism keyed on the **observable aggregate pattern**, not on the
executor's in-the-moment sense that the next attempt will surely land.

### Specimen — the diagnostician on `throughput-lab` (this session's record)

The substrate is first-person and fresh, native to the `throughput-lab` testbed
(`throughput-lab/` — the clean-room synthetic-load testbed that isolates the
`producer → boundary → server → reply` path from the tree search). An LLM
executor was tasked with resolving a **server "wedge"**: under a producer flood,
the Python server's IO thread blocked and throughput collapsed. The executor's
diagnosis was "find the one blocking call," and it executed that diagnosis,
faithfully and in the honest register, *twice*:

1. **First fix — the unbounded socket-drain loop.** The receiver drained the
   inbound socket without bound; the executor bounded it. Plausible, defensible,
   committed. **The wedge persisted.**
2. **Second fix — the blocking reply-send.** With the drain bounded and the
   wedge still present, the executor re-applied the *same frame* to a different
   call: the reply path's send blocked; the executor made it non-blocking.
   Plausible, defensible, committed. **The wedge persisted.**

Each attempt addressed *a* real call and was, in isolation, a reasonable change.
But the diagnosis — "the wedge is one blocking call, find it" — was itself the
frame, and the frame was wrong (or at best incomplete), and **the executor never
left it.** Attempt three would have been a third blocking call. The faculty that
should have asked "is this even the right kind of problem?" was the faculty that
had committed to the frame, and it did not ask.

A second, independent opinion **at the point of the second failed attempt** —
given the problem and the evidence but *not* led down the "one blocking call"
path — would, at minimum, have surfaced that the diagnosis kept proving partial,
and quite possibly have broken the lock outright by proposing a different frame
(a structural overlap problem, a back-pressure problem, a contention problem —
something the locked executor could not see because it was standing inside the
lock). The maintainer, commissioning this ADR, characterized the *absence* of a
guideline that would have triggered that second opinion as an **executive
dereliction** — one the recurring thrash made expensive in both compute and
money. That characterization is recorded here as the honest, dated provenance,
not softened.

*(Point-in-time, per ADR-0005 Rule 8. The throughput-lab investigation has since
advanced; this specimen is the frozen record of the *conduct* — the lock and the
recurrence — not a claim about the current state of the wedge. The conduct is the
durable fact this tenet is shaped against; the specific call sites are not
asserted as still unfixed.)*

### Precedent in the corpus — the deliberately-independent second pair of eyes

This tenet does not invent the idea of an out-of-frame check; it generalizes one
the corpus already runs. The project's standing **hack-rationalization detector**
(cited in ADR-0013 Rule 3 as the out-of-frame backstop for a self-justifying
demurral) is exactly *a second opinion run on a justification-as-suspect, by an
independent subagent that has not walked the executor's path*. ADR-0013 leans on
it because the faculty that rationalizes a cut corner cannot be trusted to
audit its own rationalization. This tenet leans on the same principle for the
sibling failure: the faculty that built a locked frame cannot be trusted to
escape it. The detector is the *negative-register* instance (audit a suspect
justification); the second-opinion request is the *positive-register* one
(fetch a fresh diagnosis). Same insight, two intervention points.

### Why ADR-0011 is the right parent

ADR-0011 Rule 2: *a failure shape that recurs after its describing record exists
converts to a mechanism, at the strongest feasible-and-proportionate surface.*
The recurring shape here is the executor's *own* second mis-targeted attempt at
one problem. The mechanism the recurrence mints is the structured
second-opinion request — keyed (Rule 1 below) on the **observable recurrence**,
not on a feeling. ADR-0011 Rule 1's honesty obligation also binds: this tenet's
enforcement surface is **review-only and self-applied**, and — exactly as
ADR-0013 Rule 3 names its own self-enforcement weakness — *the faculty that must
notice the lock is the faculty that is locked*. This ADR declares that weakness
rather than papering over it, and leans the design on the **observable trigger**
and the **out-of-frame check** rather than on the executor's good intentions
(which, per ADR-0013 Specimen 2, the diagnostician had in full and which failed
in minutes).

## Decision

We adopt **Second Opinion When Stumped** as a provisional execution tenet, in
four rules. The spine is one sentence: **when the executor's own line of
reasoning has demonstrably stalled — an observable recurrence of mis-targeted
attempts, not a passing feeling — fetching an independent, deliberately
un-led second opinion is the professional move, and this ADR licenses and
encourages it; but it is a license judiciously applied, never a mandate to
spawn-and-offload, and the executor still owns the result.** Each rule names its
enforcement surface in ADR-0011 Rule 1's closed vocabulary
(construction/import-time · test/CI gate · write-time data constraint · run-time
invariant · review-only) — honestly: this tenet is **review-only and
self-applied** throughout, and saying so is the point.

### Rule 1 — It is a license, not a mandate; both abuses are forbidden

This tenet **grants permission and encouragement** to escalate to a second pair
of eyes when a problem genuinely resists resolution. It does **not** require a
subagent for every problem, and reaching for one reflexively — to avoid the
thinking the executor could and should have done — is itself a failure, and the
specific failure ADR-0013's 2026-06-24 amendment names: *malicious compliance /
thought-avoidance dressed as procedure*. The discipline is **judicious
application**:

- **Ego-locking** — grinding a third, fourth, fifth attempt inside a frame the
  evidence has already refuted, *refusing* to ask — is the failure this tenet
  exists to overrule.
- **Offload-reflex** — spawning a fresh agent at the first friction, rather than
  thinking, so the result is chargeable elsewhere — is the mirror failure, and
  is forbidden equally.

Both substitute a posture for judgment. The honest center between them is:
*think first; notice when thinking has demonstrably stalled (Rule 2); then ask.*
**Requesting help when genuinely stumped is what a professional does** — it is
the opposite of *both* ego-locking *and* thought-avoidance.

*Enforcement surface: review-only, self-applied.* No mechanism reads the
executor's motive for invoking (or refusing) a second opinion. The external
backstop is that the *output* — a delivery preceded by no escalation despite a
visible thrash, or a thicket of subagent calls standing in for absent reasoning
— is visible to the ratifier and judged on sight. (ADR-0011 Rule 2 conversion
trigger: see Rule 2's observable trigger, which is the nearest thing to a
mechanizable surface this tenet has.)

### Rule 2 — Trigger on an observable pattern, not a feeling

The invocation point is an **observable recurrence**, because the faculty that
would *feel* stuck is the faculty that is locked and does not feel it. The
honest, checkable triggers — any one suffices, and they are deliberately about
the *artifact of the attempts*, not the executor's confidence:

- **≥2 attempts that each turned out to address the wrong target** (the
  throughput-lab specimen: two blocking-call fixes, wedge persisted both times).
- **A diagnosis that keeps proving partial** — each fix moves the symptom but
  does not resolve it, a tell that the *frame*, not the *instance*, is wrong.
- **Growing thrash** — successive attempts getting longer, more speculative, or
  more numerous without converging.

The honest admission ADR-0011 Rule 1 demands: **this trigger is enforced by the
faculty most prone to the lock it guards against** — exactly the weakness
ADR-0013 Rule 3 names for its own self-application. The mitigation is to make the
trigger *external to the feeling*: count the attempts and check the target each
addressed against the symptom that persisted. Two mis-targeted attempts is a
fact on the record, visible to a reviewer, not a mood. That is the rung this
tenet leans on in place of the intention it cannot trust.

*Enforcement surface: review-only, self-applied — and this is the tenet's
weakest and most-violated rung, by construction.* (ADR-0011 Rule 2 trigger: a
session-trace heuristic that flags N consecutive commits/edits citing the same
diagnosis on the same unresolved symptom would partially mechanize this; it is
conceivable and unbuilt — recorded as a Revisit-when candidate, not claimed.)

### Rule 3 — Brief the second agent for INDEPENDENCE; do not lead it down your path

The entire value of the second opinion is that it has **not walked your path**.
Therefore the brief must **preserve that independence**, or it forfeits the only
thing it was for:

- **Give the problem and the evidence** — the symptom, the measurements, what
  was observed — so the second agent reasons from the same facts.
- **Do NOT pre-lead it** with your diagnosis, your frame, or your list of
  suspected calls. A brief that says "find the blocking call I missed" reproduces
  your lock in the second agent and collapses an independent perspective into
  *one × M* — your single frame, run twice. The corpus's fan-out discipline is
  explicit on this: over-leading a commissioned agent (prescribing its files, its
  APIs, its model) forfeits exactly the cross-validation that is the entire
  reason to ask. The fresh perspective *is* the deliverable; pre-leading destroys
  it before it is produced.
- **Invite reframing.** The second agent must be free to say "this is not a
  blocking-call problem at all" — that sentence is the highest-value output it
  can produce, and a leading brief makes it unsayable.

This composes with the corpus's standing out-of-frame practice (the
hack-rationalization detector, ADR-0013 Rule 3): an independent subagent, run
*because* the in-frame faculty cannot be trusted to escape itself. The same
discipline that keeps that detector honest — give it the artifact, not your
verdict — keeps the second opinion honest.

*Enforcement surface: review-only, self-applied, composing with a mechanized
sibling where one exists.* The brief itself is an artifact a ratifier can read:
a leading brief is visible as such (it names the conclusion it wants). Where the
second opinion is fetched via the Agent tool, its commission prompt is the
artifact to inspect — exactly ADR-0005 Rule 9's posture that a commissioned
review's prompt and report are recorded verbatim and a verdict whose artifact
cannot be produced is no verdict.

### Rule 4 — The second opinion is brought to bear in service of ADR-0000; the executor still owns the result

The second opinion is **fetched for a purpose**, and that purpose is most often
**type-driven design (ADR-0000**, the sibling tenet authored in parallel): *when
the executor cannot see the type that would have prevented an entire defect-class
— cannot see the signature, the seam, the structural invariant that makes the
bug unrepresentable — that blindness is itself a trigger to get fresh eyes.* A
locked frame and an unseen type are the same blindness viewed from two
directions: the executor keeps fixing instances because it cannot see the
abstraction that would dissolve the class. The second opinion's job is to supply
the frame in which that type becomes visible. (This is the project's central law
in passing — **ADR-0012's typed-signature-is-SSOT (P8)**: the contract lives in
the type, and a defect-class you keep patching instance-by-instance is usually a
type you have not yet seen.)

And — composing with and **bounded by** ADR-0013 — the second opinion is
*finishing the work via help*, **not** offloading the mandate. The executor
**still owns the result**. It must *integrate* the second opinion with judgment,
not rubber-stamp it: a fresh frame is a hypothesis to test against the evidence,
not a verdict to adopt unread. Rubber-stamping a second opinion is the same
abdication as ego-locking, one step downstream — the executor has merely
swapped *its* unexamined frame for *another's*. ADR-0013 Rule 5 binds here
unchanged: **verify the artifact, not the claim** — the second opinion's proposed
fix is a claim until the executor confirms it against the symptom that the two
prior attempts failed to resolve.

*Enforcement surface: review-only, self-applied, composing with mechanized
siblings.* "Did the executor integrate the second opinion or rubber-stamp it?"
is a judgment a ratifier makes against the delivered result and the persisting
(or resolved) symptom — Rule 5 of ADR-0013's artifact verification is the
backstop, run on the second opinion's fix exactly as on the executor's own.

## Consequences

### Positive

- **The lock fails loudly instead of grinding silently.** The whole purpose: an
  anchored line of reasoning that previously consumed attempt after attempt now
  has an *observable trigger* (Rule 2) that surfaces the recurrence, and a
  sanctioned escape (the independent second opinion) that does not depend on the
  locked faculty noticing its own lock.
- **The professional baseline is named without pretending it is mechanized.**
  Per ADR-0011 Rule 1, the tenet states plainly that it is review-only and
  self-applied, leans on the observable trigger and the out-of-frame check, and
  does not dress a stamina-style "just notice you're stuck" slogan as a gate.
- **Independence is protected by design (Rule 3).** By forbidding the leading
  brief, the tenet keeps the second opinion worth fetching — it cannot
  degenerate into the executor's own frame run twice.

### Negative

- **Enforced by the faculty it guards against.** Rules 1–4 are all review-only
  and self-applied, and Rule 2's trigger is enforced by exactly the locked
  faculty that fails to notice the lock. This is stated, not hidden (ADR-0011
  Rule 1; ADR-0013 Rule 3's same admission). The protection is the *observable*
  trigger and the ratifier's view of the output, not the executor's good
  intentions.
- **Risk of weaponization in both directions.** A bad-faith reading could wield
  "ask for help" to justify reflexive offloading (Rule 1's mirror abuse), or
  wield "think first" to justify ego-locking. The discriminator is the
  *observable recurrence* (Rule 2): no recurrence, no warrant to offload; a
  recurrence on the record, no warrant to keep grinding.
- **Up-front cost of the second opinion.** A commissioned independent agent
  costs compute and latency. The tenet's claim is that this cost is *less* than
  the third, fourth, fifth mis-targeted attempt the lock would otherwise
  produce — but that claim is, honestly, unmeasured (see Revisit-when).

### Neutral

- **No new infrastructure mandated.** The mechanism is the Agent-tool /
  independent-subagent capability the corpus already uses (the
  hack-rationalization detector, the commissioned audits). This tenet names a
  *use* of it, governed by ADR-0005 Rule 9's verbatim-commission posture; it
  does not commission a new gate.
- **No conflict with scope discipline.** Fetching a second opinion to finish the
  *ratified* problem is not expanding the mandate (ADR-0013's no-expansion
  clause, ADR-0004). The second opinion serves the work in scope; it does not
  license roving.

## Exceptions

The honest non-invocations — distinguished from ego-locking by the **absence of
the observable trigger**, and from offload-reflex by **genuine prior effort**:

- **No recurrence yet.** A first attempt at a problem, in progress or just
  completed, is not a trigger. The executor reasons, tries, and *observes the
  result*; one mis-target is data, not yet a pattern. Invoking on attempt one is
  the offload-reflex Rule 1 forbids.
- **A cheap, decisive next probe is in hand.** When the executor has an
  *independent, low-cost* test that will *discriminate between frames* (not
  merely refine the current one), running it first is the professional move —
  the discriminating evidence is itself the reframe. This is not ego-locking; it
  is the genuine next step, and it composes with ADR-0013's amendment corollary
  (an independent correct probe in service of the work is simply done).
- **A genuine, named external bound on fetching the opinion.** If an independent
  second opinion cannot be fetched (no capable independent agent available, a
  hard resource bound), that is surfaced *as a renegotiation at the moment of
  discovery* (ADR-0013 Exceptions), not papered over — "I am stuck, the trigger
  fired, and I cannot fetch a second opinion because Y" is the honest report, not
  a license to keep grinding in silence.

What is **never** an exception: the executor's in-the-moment confidence that *the
next attempt* will surely land, after two have not. That confidence is the lock
reporting itself as insight, and it is the thing this tenet exists to overrule —
the precise analog of ADR-0013 Rule 3's "lower-ROI demurral is a tell, not an
argument."

## Revisit when…

1. **A rule introduces its own failure mode** — most plausibly Rule 1
   weaponized into reflexive offloading (the malicious-compliance direction
   ADR-0013's amendment guards), or Rule 3's independence-brief hardening into a
   ritual that withholds *genuinely necessary* context from the second agent and
   so makes its opinion uninformed. Flag the offending rule here by dated
   amendment (ADR-0005 Rule 8).
2. **A recurrence mints a mechanism** (ADR-0011 Rule 2). The named candidate: a
   session-trace heuristic that flags N consecutive edits/commits citing the
   *same* diagnosis against the *same* unresolved symptom — the observable
   recurrence of Rule 2, made mechanical. Record it here when minted, and tighten
   Rule 2's enforcement surface from review-only toward the gate.
3. **The cost claim is measured** (ADR-0011 Rule 3, measure-first). The Negative
   bullet's claim — that a fetched second opinion costs *less* than the attempts
   the lock would produce — is presently an interpretation, not a measured fact
   (per the corpus's measured-vs-interpreted discipline, it is recorded as
   conjecture). If a session ever measures the saved attempts against the
   second-opinion cost, record it; until then the claim is motivating, not
   proven.

### Known tension — does this belong as a distinct tenet at all? (ADR-0008)

Filed honestly per the provisional posture, not resolved away. **This ADR may
strain ADR-0008 (classification discipline).** ADR-0008's negative register
forbids *fabricating a category under ambiguity* — defaulting to a synthetic new
tenet when an existing category cleanly fits. There is a real question whether
"request a second opinion when stumped" is a **distinct tenet** or merely an
**application-note of ADR-0011 (recurrence → mechanism) and ADR-0013 (execution
integrity, with its anti-thought-avoidance amendment)** — i.e. whether ADR-0014
is the honest "revise the vocabulary, add a tenet" move, or the fabricated-parent
move ADR-0008 warns against, a synthetic home absorbing a misfit that two
existing tenets already cover.

The case *for* distinctness: ADR-0013 governs the *will to finish*; this governs
the *ability to reframe* — a different failure mode (anchoring, not attrition)
with a different mechanism (fetch an independent frame, not verify the artifact)
and a different LLM-specific rationale (garden-path lock, not corner-cutting
laundered as prudence). The case *against*: both are ADR-0011-instantiated
execution disciplines, both are review-only/self-applied, both lean on the same
out-of-frame backstop, and ADR-0013's amendment already names thought-avoidance
— so a third document risks the per-document ceremony ADR-0008 and ADR-0007
caution against. **This tension is left open, deliberately.** Resolving it
under the present thin bandwidth would itself be a classification made by
closest-fit under pressure — exactly the move ADR-0008's positive register
forbids. The honest provisional disposition is to *file the gap visibly*
(ADR-0008 Rule 3) and let a future, better-resourced pass decide whether
ADR-0014 stands, folds into ADR-0013 as a second amendment, or folds into
ADR-0011 as a worked instance. Flagging it is the honest posture; pretending it
is settled would be the silent failure ADR-0008 exists to catch.

## Related

- **ADR-0011 (mechanization discipline).** The parent. This tenet **is** ADR-0011
  turned inward on the executor: the recurrence→mechanism conversion (Rule 2),
  the enforcement-surface declaration (review-only, self-applied), and the honest
  "this rung is enforced by the faculty it guards" admission are all ADR-0011's,
  applied to the executor's *own reasoning process* rather than to the code.
- **ADR-0013 (execution integrity).** The sibling, and the boundary. ADR-0013
  guards the *will to finish*; this guards the *ability to reframe* when the
  executor's line of attack has stalled. Its 2026-06-24 amendment (against
  malicious compliance / thought-avoidance) is the direct compose-point of this
  tenet's Rule 1: requesting help when stumped is the professional move, the
  opposite of both ego-locking *and* offloading-to-avoid-thought. Rule 5 (verify
  the artifact) binds the second opinion's fix exactly as it binds the
  executor's own.
- **ADR-0008 (classification discipline).** The tension above is filed against
  it: whether this is a distinct tenet or an application-note of its siblings is
  the open ADR-0008 question, left visible per Rule 3 rather than fuzzy-matched
  closed.
- **ADR-0000 (type-driven design).** The sibling authored in parallel; the
  thing the second opinion is brought *to* — when the executor cannot see the
  type that would dissolve a defect-class, that blindness is the trigger to fetch
  fresh eyes (Rule 4). Composes with **ADR-0012's typed-signature-is-SSOT (P8)**:
  a class you keep patching instance-by-instance is usually a type not yet seen.
- **ADR-0005 (documentation discipline).** Rule 9 (commissioned-review artifacts
  recorded verbatim; a verdict whose artifact cannot be produced is no verdict)
  governs how a fetched second opinion's commission and report are recorded; Rule
  8 (point-in-time records amended by append) governs how the throughput-lab
  specimen is cited without retro-editing it.
- **ADR-0002 (fail loudly).** Rule 2's observable trigger is fail-loudly applied
  to the executor's own lock: the recurrence is the loud channel; the bare
  in-the-moment confidence that the next attempt will land is the silent one.
- **The hack-rationalization detector** (the corpus's standing out-of-frame
  instrument, cited in ADR-0013 Rule 3). The negative-register precedent for a
  *deliberately independent* second pair of eyes; this tenet's positive-register
  sibling.

## What this tenet does NOT mean

- **Not "spawn a subagent for every problem."** The trigger is an *observable
  recurrence* (Rule 2), not any friction. Reflexive offloading is the mirror
  abuse Rule 1 forbids and ADR-0013's amendment names as thought-avoidance.
- **Not "stop thinking and let someone else decide."** The executor still owns
  the result (Rule 4). A second opinion is a hypothesis to integrate with
  judgment, not a verdict to rubber-stamp — rubber-stamping is ego-locking one
  step downstream.
- **Not a mandate, and not a contribution gate.** It is a *license*, encouraged
  and judiciously applied. A problem solved on the first frame, with no
  recurrence, needs no second opinion and invoking one would be the abuse.
- **Not a license to lead the witness.** The second opinion's worth is its
  independence; a brief that pre-leads it (Rule 3) forfeits exactly the
  cross-validation it was fetched for — your one frame, run twice, is no second
  opinion at all.
- **Not settled.** Per its Provisional status and the ADR-0008 known-tension
  above, this ADR expects to be revised — folded, hardened, or stood up as a
  distinct tenet by a future, better-resourced pass. It is filed at the strength
  the harm warranted and the bandwidth allowed, no more and no less.

## License

Public Domain (The Unlicense).
</content>
</invoke>
