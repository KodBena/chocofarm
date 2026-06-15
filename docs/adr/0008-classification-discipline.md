# ADR-0008: Classification Discipline

- **Status:** Accepted
- **Genre:** Tenet (cross-cutting authoring discipline) — the sixth tenet,
  after ADR-0002 (fail loudly), ADR-0004 (minimal-touch), ADR-0005
  (documentation discipline), ADR-0006 (source-file headers), and ADR-0007
  (file size and information density). Sibling of ADR-0002: same shape of
  failure (a category error silently propagating), different intervention
  point — fail-loudly is the *reactive* register (surface a deviation after
  it occurs); classification discipline is the *proactive* register (refuse
  fuzzy matches and synthetic fabrications when a choice is being made
  against a vocabulary).
- **Date:** 2026-06-15
- **Provenance:** Transferred from the LengYue ADR corpus. The two-register
  principle (refuse fuzzy matches against an inadequate vocabulary; refuse to
  fabricate categories under ambiguity) is universal and transfers wholesale.
  LengYue's instance substrate (a Vue knob-domain enum, chrome-neighborhood
  mounts) is re-derived against chocofarm's real classification surfaces — the
  detector-model keying decision, the SSOT vocabulary, the audit's
  band/severity classifications, and the consult-record `kind` choice.
- **Scope:** All authoring work involving classification — picking values from
  closed vocabularies (enum-like choices, the detector action keys, severity
  tags), placing files into the `docs/` tree, naming categories, and the
  symmetric act of creating new categories under ambiguity. Applies across
  the whole `chocofarm/` package and the docs corpus.

## Context

A categorisation made by closest-fit when no true fit exists, or by
fabricated-fit when no honest category exists, silently propagates a wrong
vocabulary through every downstream consumer. The 2026-06-15 architectural
audit surfaces both registers in chocofarm.

### Substrate — positive register (fuzzy match against an inadequate vocabulary)

- **The detector mis-specification (consult-002).** The original detector
  model keyed sensing to *regions* (`cover_mask[i] = {i} ∪ overlap-neighbours`)
  — the closest available encoding, the union over every face in a region,
  passed off as a simultaneous disjunction. It was the wrong vocabulary: the
  honest sensing unit is the *arrangement face*, not the region. The mismatch
  propagated through six commits and three agent reports (each measuring
  `cover_mask` against itself) before the consult caught it. The corrected
  model re-keys the vocabulary from regions to faces
  (`docs/consults/consult-002-detector-misspec-report.md` §(4)) — exactly the
  "revise the vocabulary, don't pick the closest fit" move this register
  prescribes.
- **The `('d', i)` action-key preservation.** When the env adopted the face
  model, the action-key shape `('d', i)` was *deliberately preserved* (the env
  is "re-keyed from regions to faces; the action shape … UNCHANGED IN FORM" —
  `model/env.py`). This is the honest move: the vocabulary element (`('d', i)`)
  still fit; only the underlying data changed. The discipline is not "always
  invent new names" — it is "verify the vocabulary still fits before reusing
  it."

### Substrate — negative register (fabricate a category under ambiguity)

- **The fossil arrays in `instance.json`.** The instance file carried
  the superseded 16-region `overlaps` / `delta_treasures` arrays the face
  arrangement replaced (audit §3.1, appendix). A reader cannot tell which
  fields are live and which are fossils; an edit to `overlaps` silently does
  nothing. This is the negative-register failure: stale categorisation left
  in the canonical vocabulary, which the next reader reads as authoritative.
  *(Amended 2026-06-15: the two fossil arrays were subsequently stripped from
  `instance.json` — both are derivable from the live geometry (`overlaps` ==
  the arrangement co-coverage edge set, `delta_treasures` == the treasures no
  face covers), and the one remaining reader, `scripts/verify_faces.py`, now
  re-derives the old cover_mask from `regions_wkt` rather than the frozen array.
  The instance now carries only live, non-derivable facts. The example above is
  preserved as the motivating instance for this register.)*
- **The audit's own band/severity vocabulary.** The audit classifies findings
  `critical`/`major`/`minor` and modules `sound`/`messy`. It is disciplined
  about the failure mode this register names: severity is calibrated by the
  *substitution test* (below), not by the observed instance's cost, and the
  audit's §10 self-critique flags that the `critical`/`major` line "is softer
  than the line between `confirmed` and `refuted`" — naming the vocabulary's
  own imprecision rather than pretending it is crisp.

### Two registers, one principle

The positive register is about consuming a vocabulary; the negative register
is about extending one. Both rest on the same insight: **vocabularies and
taxonomies are honest only when they precisely fit the territory; bridging
gaps with fuzzy-fit or synthetic fabrication is the failure mode.** Both look
legitimate post-hoc and both propagate through every consumer that later reads
the classification as authoritative.

## Decision

We adopt **Classification Discipline** as a codebase-wide tenet. When a choice
involves classification, the choice is honest only if the vocabulary or
taxonomy precisely fits the case. Fuzzy matches and synthetic fabrications are
the failure mode the tenet forbids.

### Positive register — refuse fuzzy matches against an inadequate vocabulary

When choosing from a closed vocabulary and no element is a true match, the
honest move is **revise the vocabulary**, not pick the closest fit. The
detector model is the worked chocofarm instance: when "enter region Δ_i" did
not honestly model a single-point sensor, the fix was to re-derive the
vocabulary (faces, not regions), not to keep using the closest-fitting region
encoding. If vocabulary revision is out of scope for the current arc, the
deviation is filed visibly (a consult record, an inline comment naming the
misfit, an ADR amendment) so the next reader sees the gap rather than reading
the closest-match as a legitimate fit.

### Negative register — refuse to fabricate categories under ambiguity

When CREATING a classification and no existing category cleanly fits, the
honest move is **default to flat / leave it un-categorised and named as such**,
not invent a synthetic parent or force a "least-bad" home. A fabricated
category that descriptively fits nothing absorbs ambiguity into the taxonomy,
where the absorbed wrongness becomes the new baseline. The fossil
`instance.json` arrays are the dual failure: a stale categorisation left
standing is as misleading as a fabricated one — the remedy is to strip the
fossils (mark them dead or remove them), not to leave the reader to guess.

### Severity calibration — the substitution test

The discipline is calibrated by what the failure shape would cost on a
critical surface, not by the observed instance's user-visible cost. The
exercise: name the failure shape in its most general form; list the surfaces
to which the same shape could apply; calibrate to the worst case on that list.

The audit's §4 reference-rate trace is the chocofarm worked example. A frozen
`DECOMP_ANCHOR` literal used *only* as a TensorBoard display line has near-zero
cost. The *same failure shape* — a derived value frozen as a literal — applied
to the `%VoI` divisor (`exit_loop.py`) or to `vhat_lam` (a numerical input to
a provable bound, `eval_bound.py`) has catastrophic cost: a silently wrong
research result or a corrupted certificate. The discipline that catches the
harmless instance must be calibrated to the worst case, not the observed one.

## Concrete rules

1. **Verify vocabulary fit before selecting.** Before picking a value from any
   closed vocabulary (an enum-like choice, a detector action key, a severity
   tag, a directory to file a doc in), check that some element is a true match
   for the case. If none is, name the gap.
2. **Default to flat / named-as-incomplete under ambiguity.** Before creating
   a new classification, ask whether an existing category descriptively fits.
   If yes, use it. If not, leave flat and name the incompleteness. Synthetic
   parents are last resort, not default.
3. **Surface the gap visibly.** When the right move (revise the vocabulary,
   strip the fossil, hold flat) is out of scope for the current arc, file the
   deviation visibly — a consult record (per ADR-0005 Rule 2), an ADR
   amendment, or at minimum an inline comment naming the misfit. Silent
   acceptance is the failure mode this tenet forbids.
4. **Apply the substitution test for severity.** When a category error
   surfaces, calibrate the remediation to what the failure shape would cost on
   the worst-case surface it could apply to, not the observed instance's cost.

## Exceptions

### Temporary, scheduled-for-revision misfit

When the right vocabulary revision is real but its blast radius is large
enough to defer, an inline `# TODO: misfit — see X` plus a follow-up record is
acceptable. The gap is filed visibly; the misfit is bounded; the revision has
a named trigger. This parallels ADR-0002's bounded-compat-shim exception.

### Deliberately-imprecise tag

A tag that *deliberately* admits the classification is incomplete (the
analyzer's "this quantity is detector-coupled and therefore suspect," a STALE
marker on a superseded design-doc specific, the audit's `cited-not-rerun`
evidence tag) is not a closest-match — it is an explicit refusal to classify
until the case firms up, which is the discipline applied to itself. Choosing
one of these is honest; reaching for them to *avoid* choosing an honest fit is
the discipline working as intended.

## Consequences

### Positive

- **Vocabulary integrity over time.** Each addition is forced through "does
  this fit, or does the vocabulary need revising?" — which is exactly the
  question the consult-002 fix answered (re-key to faces) and the fossil-array
  finding flags (strip the dead arrays).
- **Composes with existing tenets.** ADR-0002's reactive register catches the
  silent symptom; this tenet catches the cause before the symptom forms.
  ADR-0005's documentation discipline (Rule 5, file location reflects content)
  is the documentation register of the negative register's file-placement
  case (the consult-002 relocation).
- **Self-evident audit trail.** When the gap is filed visibly, future readers
  see the gap rather than reading the closest-match as authoritative.

### Negative

- **Per-classification authoring overhead.** Each classification choice now
  carries "does this vocabulary fit?". Small per choice, real in aggregate.
- **Refused fits can stall arcs.** When the honest answer is "revise the
  vocabulary" but the revision is itself substantial (the detector re-keying
  was a multi-file change), the arc may stall on the predecessor revision. The
  mitigation is the scheduled-for-revision exception and Rule 3's gap-filing.
- **Discipline is policy, not mechanism.** Like the other tenets, it lives in
  review and audit. There is no automated check (ADR-0011 Rule 1: a declared
  review-only surface; an enum-coverage or fabricated-parent check would be
  the mechanization trigger).

### Neutral

- **No code change today.** This ADR documents a discipline for future
  authoring; ADR-0004 / ADR-0006's incremental-retrofit posture applies — no
  batched rewrite of existing classifications.

## Revisit when…

1. **A specific rule introduces its own failure mode.** Flag as the revisit
   trigger.
2. **A genuinely new register surfaces** the positive/negative split doesn't
   cover. Append a third register here rather than starting a new tenet.
3. **The substitution test produces calibration that fights another tenet**
   (e.g. a worst-case calibration demanding more loudness than ADR-0002's
   exceptions allow). Reconcile then.
4. **Tooling makes part of the discipline mechanical** — an enum-coverage
   check, a fabricated-parent (single-occupant synthetic directory) detector,
   a fossil-field check on `instance.json`. Tighten the corresponding rule
   toward enforcement as the mechanical surface grows.

## Related

- **ADR-0002 (fail loudly).** The reactive sibling. A fuzzy classification
  that slips through becomes the silent symptom ADR-0002 surfaces; this tenet
  prevents the cause. The two compose at different intervention points.
- **ADR-0003 (domain-coupling bands).** The Band 1/2/3 vocabulary is one of
  the classifications this tenet protects against fuzzy-matching — a Band-2
  module that accretes an FFXIII fact is a band-misfit this discipline catches.
- **ADR-0005 (documentation discipline).** Rule 5 (file location reflects
  content) is the documentation-register instance of this tenet's negative
  register applied to file placement (the consult-002 relocation).
- **ADR-0009 (performance investigation discipline).** The per-domain instance
  for the perf-claim vocabulary — "faster"/"regression"/"no change" is a
  closed vocabulary; substantiation is the fit-verification this tenet implies.
- **The 2026-06-15 architectural audit** — the detector-misspec substrate, the
  reference-rate severity calibration (§4/§5), and the fossil-array
  negative-register instance.

## What this tenet does NOT mean

- **Not "every category must be perfect on first pass."** Authoring is
  iterative; the tenet asks for honest "this vocabulary doesn't fit" surfacing
  when it doesn't.
- **Not "all classifications need ceremony."** Trivial one-off names are not
  what this tenet operates on; it applies to classifications that propagate to
  consumers — the detector model, the SSOT vocabulary, severity tags, the
  directory taxonomy.
- **Not a ban on synthetic parents in all cases.** A synthetic parent that
  genuinely captures a real distinction (e.g. an `hp/` package for the
  registry/schema that share a real characteristic) is honest. The tenet bans
  parents that exist *to absorb a misfit*.
- **Not a substitute for fail-loudly.** When the discipline fails in practice,
  ADR-0002's reactive register catches the resulting silent symptom.

## License

Public Domain (The Unlicense).
