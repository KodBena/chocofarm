# ADR-0007: File Size and Information Density

- **Status:** Accepted
- **Genre:** Tenet (cross-cutting authoring discipline) — the fifth tenet,
  after ADR-0002 (fail loudly), ADR-0004 (minimal-touch), ADR-0005
  (documentation discipline), and ADR-0006 (source-file headers). Sibling of
  ADR-0004: same failure mode, different intervention point.
- **Date:** 2026-06-15
- **Provenance:** Transferred from the LengYue ADR corpus. The tenet (size +
  density together, soft thresholds, no logic golf, content-aware
  contraction) is universal. LengYue's numeric budgets and contraction table
  were TypeScript/Vue-specific; they are re-derived here for Python, and the
  oversized-file instances are chocofarm's real ones.
- **Scope:** Source-code authoring in `chocofarm/`. Documentation is governed
  by ADR-0005.

## Context

ADR-0004 governs editing a partially-visible file: only touch the flagged
lines. This tenet is the prophylactic counterpart — keep files small enough
that partial visibility is rare, eliminating the condition under which
ADR-0004's reactive discipline applies.

Two metrics together do the work. **Size** caps the number of lines a tool
view has to fit. **Density** ensures those lines carry decisions, not
boilerplate. A bloated file (high size, low density) is the worst case: tool
truncations elide as much decision content as boilerplate, and reviewers wade
through noise to find the parts that matter.

chocofarm has real oversized files that the 2026-06-15 architectural audit
named. They are the refactoring queue this tenet's Neutral clause governs:

- `solvers/decomp.py` — **675 lines**. The audit is explicit that it is *not*
  the god-object its length implies — it is three honest layers (cluster
  decomposition, micro-solve, macro-plan). Its sins are elsewhere (frozen λ,
  re-derived env state), but its length still makes it a partial-visibility
  hazard under ADR-0004.
- `analysis/analyzer.py` — **605 lines**. Disciplined internally, but large;
  it also mixes presentation (`_print_report`) into analysis, a clean split
  seam.
- `hp/registry.py` (**715**), `az/exit_loop.py` (**510**), `az/parallel.py`
  (**451**), `hp/schema.py` (**449**), `az/features.py` (**389**),
  `az/mlp.py` (**360**).

## Decision

### Size — soft thresholds (Python)

- **Target ≤ 300 lines** for a typical module; **≤ 400 acceptable** for a
  single coherent unit (one solver family, one schema, one cohesive state
  machine) where splitting would fragment cross-line invariants.

When a file crosses the threshold, the contributor pauses and asks whether a
split is warranted before extending further. Typical refactor moves in
chocofarm: split presentation from analysis (`analyzer.py`'s `_print_report`);
lift a shared helper into the package base (`solvers.base`'s
`candidate_actions`, already done); separate the schema (the typed contract)
from the thin layer over it (`hp/schema.py` vs `hp/registry.py`, already
split this way).

### Density — effective lines / total lines

"Effective" lines carry decisions specific to this file's purpose (function
bodies, non-trivial numerical expressions, the belief/dynamics logic, the
contracts a docstring owns). "Boilerplate" lines do not (imports, trivial
property accessors, repeated scaffolding).

Operational thresholds at review (qualitative — chocofarm has never measured
the ratio, so it is a review heuristic, not a metric):
- **healthy:** the file is mostly decisions.
- **yellow flag:** noticeable scaffolding-to-decision ratio — review for
  splitting next time the file is touched.
- **red flag:** the decisions are buried in scaffolding — refactor before
  further extension.

### Format — content-aware contraction

Format reflects edit cadence. Content rarely hand-edited may contract to
maximize the visible budget for content that is; decision logic does not.

| Content | Rule |
|---|---|
| **Data tables, constant arrays, fixture literals** | Contraction acceptable — pack multi-value rows, keep one logical row per line. |
| **Numerical / decision logic** | No contraction. Standard formatting; multi-line for clarity. |
| **Docstrings / contracts** | Contextual. The rich env/registry headers are decisions-about-the-file and earn their length; pure boilerplate prose does not. |

**Soft column cap:** ~100 characters (chocofarm's existing code runs to
~100–110; beyond that even contracted content goes multi-line).

**The no-go.** Never contract numerical or decision logic to fit a size
budget. Code golf in the belief filter, the Dinkelbach loop, or a forward
backend hides bugs behind dense lines and inflates working-memory cost per
line — and, given ADR-0004, a dense line in a partially-visible file is
exactly where a silent numerical drift hides. If a logic file is over budget,
the answer is structural extraction, never cosmetic compression.

## Exceptions

- **Coherent units** (one solver family, one schema dataclass set) with high
  density may run to ~400 lines if splitting would fragment cross-line
  invariants.
- **Generated artifacts**, if added, are exempt; size is a property of the
  upstream contract.

## Consequences

**Positive.** Partial-visibility risk (ADR-0004) drops at the source. Review
fits in working memory. Single-purpose discipline enforced by gravity.

**Negative.** Some refactors are mandatory work. Discipline is policy, not
mechanism — like the other tenets, it lives in review (ADR-0011 Rule 1: a
declared review-only surface; no `max-lines` check exists). Over-fragmentation
is a real risk if the rules are read too literally — `decomp.py`'s three
honest layers should not be shattered into a dozen files just to hit a line
count.

**Neutral.** No retroactive sweep. The oversized files named in Context enter
a refactoring queue and are addressed when next touched substantively,
composing with ADR-0004's and ADR-0006's incremental-retrofit posture. (Per
the audit's own roadmap, several of those files are slated for content
changes — `analyzer`'s presentation split, `exit_loop`'s `RunConfig` — that
naturally shrink them.)

## Revisit when…

1. A linter or pre-commit hook automates the size rule — soft thresholds can
   become enforced limits (ADR-0011 Rule 1's mechanization trigger).
2. The information-density heuristic proves too judgmental in practice —
   replace with a more mechanical proxy.
3. A specific exception's classification turns out wrong in practice — the
   exception narrows.

## Related

- **ADR-0004** — the reactive sibling; this prevents the situation ADR-0004
  mitigates. The oversized files here are ADR-0004's partial-visibility
  hazards.
- **ADR-0005** — the documentation analog; both compose when a refactor
  relocates and resizes simultaneously.
- **ADR-0006** — file-level companion; smaller files multiplied by per-file
  headers keep overhead bounded.
- **ADR-0001 / ADR-0003** — file structure should match actual responsibility
  and domain band, no aspirational cohabitation.

## What this tenet does NOT mean

- Not a hard line-count limit; the threshold flags, not ceilings.
- Not a mandate to split immediately; existing files retrofit incrementally.
- Not a directory-organization decision.
- Not enforced by tooling today.

## License

Public Domain (The Unlicense).
