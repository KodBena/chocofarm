# ADR-0006: Source-File Headers

- **Status:** Accepted
- **Genre:** Tenet (file-level authoring discipline) — the fourth tenet,
  after ADR-0002 (fail loudly), ADR-0004 (minimal-touch), and ADR-0005
  (documentation discipline).
- **Date:** 2026-06-15
- **Provenance:** Transferred from the LengYue ADR corpus. LengYue's tenet
  unified two divergent header conventions across a TypeScript frontend and a
  Python backend; chocofarm is a single Python package with one already-de-
  facto convention, so this ADR *codifies the existing convention* rather than
  reconciling two. The Python form (module docstring with path + purpose +
  license) is exactly LengYue's "Form for Python files," and chocofarm
  already follows it across the package.
- **Scope:** All Python source files in `chocofarm/`, plus `tests/`,
  `scripts/`, and `probes/`. Data files (`instance.json`, `faces.json`)
  follow their own format conventions.

## Context

chocofarm's source files already converge on a header pattern: a module
docstring whose first content names the module's path/area and purpose, with
a `Public Domain (The Unlicense)` declaration. Examples at HEAD:
`hp/registry.py`, `hp/schema.py`, `eval/report.py`, `model/env.py`,
`solvers/base.py`, `az/parallel.py` all carry it. The convention is good and
in use; this ADR names it so it is a stated tenet rather than an unwritten
habit, and so the few files that lack it have a rule to retrofit against.

The convention earns its weight for two reasons that the architectural audit's
findings reinforce:

1. **Self-locating files.** A file pasted into a review, a diff, or a search
   result identifies itself. This composes directly with ADR-0004
   (minimal-touch): a contributor working with partial visibility into a
   675-line `decomp.py` or a 715-line `registry.py` benefits from the file
   declaring where it lives. The audit's "Part A/B/C as load-bearing
   identifiers" finding (nine modules explaining their behavior by reference
   to ephemeral session tags) is the *opposite* failure — a header that
   names path + purpose, not a session tag, is what keeps a file readable
   standalone.

2. **Per-file license declaration.** chocofarm is Public Domain (The
   Unlicense), and the chocofarm files already declare it individually. This
   matters at the moment any single file is vendored, copied, or reposted —
   without a per-file license, only the project as a whole is identifiably
   Public Domain, and the signal is lost once a file leaves its repo context.

## Decision

**Every Python source file in `chocofarm/` (and `tests/`, `scripts/`,
`probes/`) carries a module-docstring header with three parts:**

1. **The module's path or area**, as the first content of the docstring
   (e.g. `chocofarm/eval/report.py — …` or `chocofarm AZ — Part A: …`).
2. **A brief purpose statement** (one line minimum; multi-section commentary
   fine — the env and registry headers are good examples of rich-but-bounded
   purpose docs).
3. **A `Public Domain (The Unlicense)` declaration**, typically at the end of
   the docstring (newer files state it explicitly; the rule is that it be
   present).

### Form

```python
#!/usr/bin/env python3
"""
chocofarm/<area>/<file>.py — <one-line purpose>.

[optional: design notes, the contracts this file owns, audit references]

Public Domain (The Unlicense).
"""
```

### Why path-first

A file that names its own path is the cheapest insurance against being moved
without its docstring updated, and the most useful thing to have when a
fragment is pasted out of context. Composes with ADR-0004.

### Composition with ADR-0004 — incremental retrofit

ADR-0004 enables incremental retrofit. When a file is touched with full
visibility, the header is added/corrected; when it's touched under partial
visibility, the header is left for next time. No special discipline is
required; headers accumulate as files cycle through normal editing.

### Exceptions

- **`__init__.py`** files: a header is fine but not required (often empty or
  re-exports only).
- **Data files** (`instance.json`, `faces.json`): follow their own format;
  no module docstring.
- **Generated artifacts**, if any are added, do not carry a hand-written
  header (a header would be lost on regeneration); the generator's config is
  the right home for that concern.

## Consequences

### Positive

- **Self-locating files.** Easier to read pasted code, navigate diffs, and
  identify files in tooling and agent-report output (the audit's 35-agent
  fan-out cited files by path; the headers make that unambiguous).
- **Per-file license clarity.** Vendored or extracted files retain the
  Public Domain signal.
- **Consistency across the package.** One shape everywhere reduces friction.

### Negative

- **Per-file ceremony.** Small but real, especially for short utilities.
- **Discipline is policy, not mechanism.** The tenet lives in authoring habit
  and review; there is no header-presence linter. (ADR-0011 Rule 1: a
  declared review-only surface; a path-presence check would be the
  mechanization trigger, as LengYue's `tools/source-headers/check.mjs` was
  for that corpus.)

### Neutral

- **No retroactive sweep.** Per ADR-0004, existing files without a complete
  header are retrofitted incrementally as they're touched, not in a sweep.

## Revisit when…

1. **Tooling exists to auto-verify path headers.** A presence check would
   partially mechanize the discipline, at which point the rule could tighten
   from reviewed toward enforced. (This is the firing LengYue's ADR-0006
   recorded; chocofarm has not built it.)
2. **The license posture changes.** If the project moves away from Public
   Domain, the per-file declaration's specifics need updating; the discipline
   remains.

## Related

- **ADR-0004 (minimal-touch).** Self-locating files reduce the cost of
  partial-visibility editing; the composition pattern is incremental retrofit.
- **ADR-0005 (documentation discipline).** The umbrella tenet of which file
  headers are a file-level instance. ADR-0005 Rule 5 (file location reflects
  content) is harder to violate when the file declares its own location.
- **ADR-0007 (file size and information density).** Smaller files multiplied
  by per-file headers keep header overhead bounded.

## What this tenet does NOT mean

- **Not a requirement for documentation files** (`.md`, the ADRs
  themselves). Markdown has its own conventions; ADR-0005 governs.
- **Not a requirement for data/blob files.** The tenet applies to files
  carrying source code intended for human reading.
- **Not a license-enforcement mechanism.** The declaration is a signal; the
  file's actual license is the project's overall Public Domain status.
- **Not a substitute for git-tracked metadata.** Authorship and change
  history live in git, not in headers.

## License

Public Domain (The Unlicense).
