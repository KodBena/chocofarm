# ADR-0005: Documentation Discipline

- **Status:** Accepted
- **Genre:** Tenet (cross-cutting authoring discipline) — the third tenet,
  after ADR-0002 (fail loudly) and ADR-0004 (minimal-touch).
- **Date:** 2026-06-15
- **Provenance:** Transferred from the LengYue ADR corpus. The tenet and its
  rules are universal and transfer wholesale. LengYue's instance list named
  monorepo/dispatch-ledger/work-status-store machinery chocofarm does not
  have; the rules are re-derived against chocofarm's real documentation
  corpus — the design notes under `docs/design/`, the consult records under
  `docs/consults/`, the agent commission/report pairs under `docs/agents/`,
  the results under `docs/results/`, and the architectural-audit corpus under
  `docs/notes/audit/`. Rules that presuppose infrastructure chocofarm lacks
  (a Postgres work-status store, a cross-team dispatch ledger) are re-stated
  in the form chocofarm actually uses or marked as not-applicable.
- **Scope:** All authoring of documentation in this repository — ADRs, design
  notes, consult records, agent commission/report pairs, results write-ups,
  STATUS / handoff documents, and the audit corpus.

## Context

chocofarm is a fast-moving research scratch project with an extensive `docs/`
corpus, and the 2026-06-15 architectural audit (`docs/notes/audit/`)
surfaced documentation rot and drift patterns that share a common root:
**documentation written reactively, after-the-fact, or without an explicit
lifecycle decays into low-trust artifacts faster than the code decays around
it.** The audit names several concrete instances:

- **A doc written to cure staleness that was stale in 24 seconds.** The
  2026-06-15 handoff listed a `train_value.py` docstring fix as "pending";
  git shows that exact fix committed 24 seconds later (audit §9, L10). A live
  task queue narrated in immutable prose is stale before it is read.
- **A binding convention with no definition.** `ADR-0002` was cited 16 times
  across the code, and the handoff pointed readers to "the ADR-0002 registry"
  — which did not exist (audit §9, L9). This ADR corpus is the fix; the
  citations now resolve.
- **A dangling pointer to the simulation's heart.** `consult-002 §4`, the
  authority for the env's corrected face model, was filed in the wrong
  directory (`docs/agents/` rather than `docs/consults/`) and its report had
  no literal `§4` anchor (audit §9). Relocated and re-anchored as part of
  establishing this corpus.
- **Load-bearing knowledge offloaded to volatile prose.** 111 `design §N`
  citations make a design doc the de-facto spec, while several of its
  load-bearing specifics (`37-slot` space, `90-float` vector, ISMCTS teacher)
  are marked STALE in the very code implementing their successors (audit
  §2.G).

A working contributor's experience of these patterns: orientation takes
longer than it should because the documentation graph has to be reconstructed
from the code rather than read from the documents. The cost compounds — the
longer the gap between work and its documentation, the more reconstruction
the next reader does, and reconstruction degrades into guessing. The
underlying principle: *documentation is cheaper to write while you remember
why, not when you reconstruct why later.*

## Decision

We adopt **Documentation Discipline** as a codebase-wide tenet. Every
documentation artifact is authored under the following rules.

### Rule 1: Single source of truth per nominal handle

Anything that names a piece of work or a fact — an ADR number, a consult id,
a reference rate, the feature layout, the episode horizon — has exactly one
owning home. Parallel records of the same handle drift silently. The audit's
sharpest SSOT findings are exactly this failure: the three reference rates
hand-copied across ~10 files (one already drifted, `0.0941` vs `0.094`); the
belief mechanics duplicated where the dual bound certifies against them; the
feature layout written in three places. The structural fix is one owner per
fact (the env computes the rates; one `FEATURE_LAYOUT(env)` owns the layout;
`env.max_steps` owns the horizon), with everything else deriving from it —
the discipline ADR-0008 (classification) and the audit's target architecture
(§6) generalize.

### Rule 2: Consult and design records live where the convention says

Records have a predictable home, not an author's-convenience one:

- **Design notes** live under `docs/design/`.
- **Consult records** (an independent review commissioned and treated as
  evidence) live under `docs/consults/`, as `consult-NNN-*`.
- **Agent commission/report pairs** live under `docs/agents/`.
- **Results** live under `docs/results/`; **audit records** under
  `docs/notes/audit/`.

The load-bearing commitment is **one place, known to the next reader**. The
`consult-002` misfiling (it lived under `docs/agents/` though it is a consult)
is the concrete violation this rule fixes; relocating it to `docs/consults/`
was part of establishing this corpus.

*(LengYue's Rule 2 named a cross-team dispatch ledger. chocofarm is a single
package with no sub-projects, so there is no dispatch ledger; this rule is
re-instanced as the consult/design/agent/results/audit directory convention,
which is the chocofarm analog of "one place, both parties know where.")*

### Rule 3: Descriptions describe relations, not content snapshots

A reference's description should describe how the referenced document RELATES
to the referencing one, not what the referenced document SAYS. The latter
goes stale when the target evolves; the former survives most realistic
evolutions. When fixing a relocated reference (the `consult-002` path
repointing, the `honest-rates-faces.md` path fix), the citation must still
accurately describe the real relation: a citation that points at a section
must point at a section that *exists* (the report's `## (4) The correct model
and remedy`, cited as `§(4)` — never the dangling `§4` that resolved
nowhere). Apply this in every "see also," every Related section, every
cross-document link.

### Rule 4: Document bodies don't bare-name their siblings where a rename would break them

Prefer generic descriptors ("the companion report," "the audit appendix")
over bare filenames in running prose where the reference would self-break on
a rename. Exception: filenames in code blocks, in shell commands, and in
load-bearing path citations (where the path *is* the resolvable handle) are
fine — the rule applies to incidental running prose.

### Rule 5: File location reflects content, not authoring history

If a file's content has drifted from its directory's intent, move it before
someone trusts the directory. The `consult-002` relocation is the worked
instance: a consult record filed under `docs/agents/` is exactly the
location-misleads-content trap this rule names. When relocating, repoint the
live referrers (a moved file's links are broken links until repointed — see
Rule 3); leave point-in-time records that *describe* the old location alone
(Rule 8 / Rule 11).

### Rule 6: Documentation lifecycle — author as you decide

Write the record while you remember why. Status updates, deviation notes, and
the context for a decision are captured in the moment, not reconstructed at
the close. The audit's 24-seconds-stale handoff is the failure this rule
prevents: a "pending" item narrated in prose that was done before the prose
was read. The corollary the audit draws (L10): **status documents record
slowly-aging decisions and rationale, never a live task queue** — the queue
belongs in version control / the commit log, not in immutable prose that
rots in seconds.

### Rule 7: Transitional documentation sunsets itself

Sections or documents introduced as transitional carry an explicit retirement
plan named at the moment they are added. Without it, transitional sections
ossify into permanent fixtures that misdescribe the current state. (The
audit's STATUS / handoff documents are the natural home for transitional
orientation; each transitional claim names what retires it.)

### Rule 8: Sibling revisions / dated corrections over silent edits of point-in-time records

When an authoritative record is found wrong in a load-bearing way, preserve
the original as the planning-time record and add a dated correction
(an Amendments-line entry for an ADR, a sibling note, or an in-situ dated
strike) — never silently rewrite a point-in-time artifact. The
architectural-audit corpus is the worked instance of the convention done
right: it is explicitly **point-in-time and not retro-edited** — where a
worker overstated, the original stands verbatim in the appendix and the
correction is made in the audit's §5 (the deflation record), dated. A silent
rewrite of a point-in-time artifact destroys the traceability the record
exists for. (This is why the agent commission records' old `docs/agents/`
references to the relocated consult-002 were left intact — they are frozen
records of what an agent was told, not live links to repoint.)

### Rule 9: Commissioned-review artifacts are recorded verbatim, in-tree

When work leans on a commissioned review — a delegated audit, a consult, an
adversarial pass — whose verdict the citing session treats as evidence, the
commission prompt and the full report are recorded verbatim, in-tree. The
verdict label does not travel without the artifact's substance; **a verdict
whose artifact cannot be produced on demand is treated as no verdict.** The
architectural-audit is the largest worked instance: every one of 35 workers'
raw outputs is reproduced verbatim in
`architectural-audit-2026-06-15-appendix.md`, and the consult records carry
both the commission and the verbatim report (`consult-001`, `consult-002`,
`consult-003`). Verbatim appendices are reference records consumed by
pointer-citation, not read end to end on every consultation — but the digest
that fans out over them is read in full, reconciling this rule with the
read-fully-before-citing discipline (root `CLAUDE.md`).

## Consequences

### Positive

- **Lower reconstruction cost.** A reader walking into the corpus cold spends
  less time guessing which docs are current, which references resolve, and
  which numbers mean what.
- **Friction-aligned with development.** The discipline operates at the
  moment of authoring; it doesn't impose a batched cleanup later.
- **Audit trail.** Each rule corresponds to a concrete pattern the
  architectural audit surfaced; future audits reference the rule rather than
  re-deriving the pattern.

### Negative

- **Per-write authoring overhead.** Each documentation event takes slightly
  longer. Small per write, real in aggregate.
- **Discipline is largely policy, not mechanism.** Like ADR-0002 and
  ADR-0004, this tenet lives mostly in review and authoring habit. chocofarm
  has no doc-graph validator (LengYue's mechanization); cross-reference
  resolution is review's, not a CI gate's — a declared review-only surface
  (ADR-0011 Rule 1).
- **Some rules require judgment.** Rule 4 (bare-naming) is nearly
  mechanical; Rule 3 (relation-vs-content) requires a small evaluation each
  time. Reasonable contributors will sometimes disagree.

### Neutral

- **No retroactive rewrite required.** ADR-0004's spirit applies: incremental
  retrofit when files are touched for other reasons; no blanket rewrite pass.
  The point-in-time records (the audit, the agent commissions, the
  `(point-in-time)`-marked results) are explicitly NOT retro-edited.

## Revisit when…

1. **A specific rule introduces its own failure mode.** Unlikely; flag as the
   revisit trigger.
2. **Documentation tooling matures enough to mechanize part of the
   discipline.** A cross-reference-resolution checker (does every cited path
   resolve?) is the easiest candidate and is *not* soft — a path either
   points at an existing doc or it doesn't. If one is built, it becomes the
   mechanization of Rule 3/5's resolution half. (ADR-0011 Rule 1 records this
   as the open mechanization.)
3. **A genuinely new failure pattern surfaces** not covered by the existing
   rules. Append the rule rather than starting a new tenet — this tenet is
   shaped to absorb additional disciplines.

## Related

- **ADR-0002 (fail loudly).** This tenet is fail-loudly applied to
  documentation: when a documentation gap exists, name it visibly rather than
  papering over it. The dangling `consult-002 §4` and the
  cited-but-nonexistent ADR registry were exactly the silent-doc-failures
  ADR-0002 forbids, surfaced and fixed here.
- **ADR-0004 (minimal-touch).** The incremental-retrofit posture for existing
  documentation directly applies ADR-0004: don't blanket-rewrite docs that
  aren't being touched.
- **ADR-0006 (source-file headers).** The companion tenet governing per-file
  header conventions — a specific instance of this discipline at the file
  level.
- **The 2026-06-15 architectural audit and its appendix** — the worked
  instance of Rule 9 (verbatim records) and Rule 8 (point-in-time,
  not-retro-edited, dated corrections in §5).

## What this tenet does NOT mean

- **Not "all documentation is created equal."** ADRs and the audit are
  higher-stakes; commit messages are lower-stakes. The discipline applies to
  all but the formality scales.
- **Not "no documentation churn."** The goal is to reduce DRIFT
  (unintentional staleness), not to freeze documents.
- **Not "documentation must be exhaustive."** Brevity remains a virtue.
- **Not a contribution gate.** A change with imperfect documentation is not
  blocked by this tenet; reviewers flag specific rules proportionately.

## License

Public Domain (The Unlicense).
