# ADR-0010: Render Locality and Canvas — Not Applicable (Lineage Entry)

- **Status:** Accepted (as a lineage record; no chocofarm decision is taken
  here)
- **Genre:** Lineage entry — preserves corpus-numbering continuity with the
  LengYue ADR corpus this project forked.
- **Date:** 2026-06-15
- **Provenance:** LengYue's ADR-0010 ("Render Locality and Canvas for
  Data-Dense Visuals") is a Vue-SPA frontend tenet: it governs where a
  high-frequency reactive value may be read in a component tree, and when a
  data-dense visual must be a `<canvas>` rather than a `v-for` of DOM/SVG
  nodes. Both rules are specific to a reactive UI framework rendering to a
  browser.

## Why this slot is kept but empty

**chocofarm has no UI.** It is a single numpy/JAX/numba operations-research
Python package — a simulation environment, a set of solvers, an AlphaZero
stack, a provable bound, an eval suite, and a docs corpus. There is no Vue, no
component tree, no reactive render loop, no `<canvas>`, no DOM. LengYue's
ADR-0010 has no surface to apply to here.

Per the fork-adaptation discipline (ADR-0008: refuse a fuzzy match against an
inadequate vocabulary), the honest move is **not** to invent a strained
chocofarm "render locality" analog that fits nothing — that would be exactly
the synthetic-fabrication failure ADR-0008's negative register forbids. The
honest move is to record that this LengYue-lineage tenet does not transfer,
and to keep the number stable so the corpus numbering stays aligned with its
source (the code cites other ADRs by number; numbering continuity is a real
property to preserve).

## The nearest chocofarm concern

The *spirit* of LengYue's ADR-0010 — a cost that is invisible at authoring
time and surfaces only under measurement, prevented by a name the author
reaches for and a reviewer checks against — does have a chocofarm home, but
it is **ADR-0009 (performance investigation discipline)**, not a render rule.
chocofarm's invisible-until-measured costs live in the search/forward hot path
(the per-component `bench_hotpath` regression guard, the float32/numba
behavioral-equivalence bar, the forward `ABS_TOL = 1e-4` contract). A
contributor looking here for "the rule that prevents a silent hot-path cost"
should read ADR-0009.

## Consequences

- **Numbering is stable.** ADR-0002 and ADR-0004 are cited by number in
  chocofarm's code; keeping the full 0001–0011 numbering aligned with the
  LengYue lineage avoids any renumbering that would invalidate those
  citations or future ones.
- **No discipline is imposed.** This entry adds no chocofarm rule. It is a
  signpost.

## Related

- **ADR-0009 (performance investigation discipline).** The chocofarm tenet
  that carries the nearest concern — invisible-until-measured cost,
  captured-and-reproducible substantiation.
- **The LengYue ADR-0010** — the source tenet, recorded here as not
  transferring, per the fork-consumption discipline in `docs/adr-synopsis.md`
  ("How a fork consumes this corpus").

## License

Public Domain (The Unlicense).
