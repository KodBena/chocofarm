# ADR-0003: Domain-Coupling Bands

- **Status:** Accepted
- **Genre:** Bounded Context Map (structural-descriptive with prescriptive
  elements) — a third genre after the *decision* of ADR-0001 and the *tenet*
  of ADR-0002. Maps the domain coupling of the codebase and gives a principle
  for evaluating future changes against it.
- **Date:** 2026-06-15
- **Provenance:** Adapted from the LengYue ADR corpus. LengYue's ADR-0003
  mapped a Vue frontend's coupling to the game of Go, with a "what would a
  Chess port require?" principle. chocofarm has no frontend and is not a Go
  client, so the instance map does not transfer — but the *structure* of the
  decision does: name the bands of coupling, give a forward-looking question
  that forces honest separation of abstraction from instance, and refuse to
  extract abstractions before a second concrete consumer exists. The bands
  are re-derived for chocofarm's actual axes: how tightly each module is
  coupled to FFXIII-the-game, to the operations-research machinery, and to
  the simulation/solver seam.
- **Scope:** The whole `chocofarm/` package. Cross-references the env/Policy
  inversion of control (the load-bearing seam ADR-0001 and the audit both
  protect).

## Context

chocofarm exists to compute optimal gil farming in FFXIII, formalized as
adaptive stochastic orienteering under partial observation (a belief-state
MDP). That is two facts glued together: a *specific game* (FFXIII chocobo
treasure digging, with concrete treasure coordinates, teleports, and
detection geometry) and a *general OR problem class* (belief-MDP, Dinkelbach
renewal-rate optimization, orienteering). The codebase mixes both, and the
mix is not uniform: some modules know about treasure coordinates and the CSNE
teleport; some speak only in beliefs, worlds, and rates; some don't care
which problem they're solving at all.

Two prospective futures make the coupling worth mapping honestly:

1. **A different OR problem** (a different orienteering/belief-MDP instance —
   not FFXIII at all). What would survive? The belief mechanics, the
   Dinkelbach loop, the orienteering/route machinery, the solvers, the
   AlphaZero stack, the dual bound — all of it is problem-class machinery, not
   FFXIII machinery. Only the instance data and the FFXIII-specific geometry
   would be replaced.

2. **A different game with the same OR shape** (another treasure-hunt-style
   game). The FFXIII coordinates, teleports, and arrangement faces would be
   replaced; the OR machinery and the env/Policy seam would survive.

Without a map, future features can't be honestly designed against the
boundary — and without a principle, the map is just inventory.

## Decision

**Document the current domain coupling of the codebase as three bands, and
adopt a single forward-looking principle for evaluating new modules against
it. Do not preemptively extract abstractions; do design new modules so the
seam is clean.**

The principle, stated plainly:

> When writing a new module, ask: **"what would change if the game were
> different but the OR problem were the same? And what would change if the OR
> problem were different but solved by the same machinery?"** Not because a
> second instance exists, but because the two questions force honest
> separation between the FFXIII instance, the OR abstraction, and the
> solver-agnostic seam. If the answer to the first is "everything in this
> module," the module is FFXIII-bound — isolate it so a different game could
> replace it wholesale. If the answer to both is "nothing," the module is
> solver-agnostic — name its concepts for the problem class, not the
> instance. If the answer is "some of it," that is the seam; design it
> deliberately, even if you don't extract an abstraction today.

This is a design discipline at authoring time, not an extraction mandate.
Existing code stays put; new code is written with the seam in mind.

### Why not extract abstractions preemptively

Sandi Metz's principle applies: *duplication is cheaper than the wrong
abstraction.* An abstraction extracted before a second concrete instance
exists is shaped by speculation and is almost always wrong-shaped. chocofarm
has exactly one game instance and one OR instance today, so the cost-benefit
tilts toward "extract when the second use case exists." The 2026-06-15 audit
makes the same point from the dual direction (its E lesson — abstractions
built then abandoned beside a live inline copy are *worse* than no
abstraction): the `facemodel.SenseAction` object is fully built, documented,
and dead, while the env reimplements it inline. The discipline is not "build
abstractions"; it is "design clean seams and extract only when a second
instance forces it."

## The three bands

The codebase's modules sit on a spectrum.

### Band 1 — Solver-agnostic / the simulation–solver seam

These modules speak in concepts no specific solver and no specific game
needs. The load-bearing instance is the **env/Policy inversion of control**:
`Environment` owns dynamics, belief, and simulation; `Policy` is a thin
injected `decide(env, loc, bw, collected, lam, rng)` seam; `env.py` imports
no solver. A new solution method is a new `Policy` subclass with zero env
edits. This seam is the single hardest architectural decision in the system,
made right (the audit's §1), and it is solver-agnostic by construction: the
env doesn't know whether it's being driven by greedy, ISMCTS, or AlphaZero.

### Band 2 — OR-general (belief-MDP / orienteering machinery)

These modules speak the operations-research problem class, not the FFXIII
instance. They would survive a port to *any* adaptive-stochastic-orienteering
/ belief-MDP problem:

- The **belief mechanics** (the world-set, filtering, marginals) — a
  belief-MDP concept, not an FFXIII one.
- The **Dinkelbach renewal-rate machinery** (`rate`, `dinkelbach_rate`, the
  λ-penalty threaded as a live per-call argument) — the rate-optimization
  abstraction; λ is the OR-general control variable.
- The **orienteering / routing** (`route_time`, `exit_cost`, the
  greedy/CE/rollout/sparse-sampling/UCT/ISMCTS/NMCS solvers) — orienteering
  machinery parameterized by a distance function, not by FFXIII coordinates.
- The **AlphaZero/Gumbel stack** (`az/`), the **provable dual bound**
  (`bounds/`), and the **structural analyzer** (`analysis/`) — all phrased
  over `worlds`, `cover`, `value`, `N`, `K`, never over "treasure 8" or
  "the CSNE teleport."

A different OR instance replaces the data, not this machinery.

### Band 3 — FFXIII-bound

These modules carry FFXIII facts that don't exist outside the game (or carry
FFXIII-specific encodings of general concepts). Porting them is replacement,
not refactoring:

- The **instance data** (`instance.json`, `faces.json`): the 20 treasure
  coordinates, the 3 teleports (CSNE/CSCE/τ_4), the detection-region geometry.
- The **arrangement faces / detector geometry** (`arrangement.py`,
  `facemodel.py`): the planar arrangement of FFXIII's 16 detection polygons
  into 44 atomic faces, the corrected sense model
  (`docs/consults/consult-002-detector-misspec-report.md` §(4)).
- The **instance-loading and geometry tooling** that parses the FFXIII
  GeoGebra/WKT source.

A different game replaces all of this; a different OR problem keeps the
*shape* (an instance file, a distance function, an observation model) but
swaps the contents.

### Band-mixed — the seams

A few modules straddle bands and are where seam-design matters most:

- **`Environment`** is Band 1 in its seam (the env/Policy contract) and Band
  2 in its mechanics (belief, Dinkelbach), but loads Band 3 instance data.
  The copy-on-write `with_scenario`/`restrict` (ADR-0001) are the clean seam
  that keeps the Band-2 machinery sharable across Band-3 instance changes.
- **`features.py` / `actions.py`** are Band 2 (an AZ feature/action encoding
  over a general belief) but their *layout* is derived from the env's
  instance shape (`feature_dim(env)`, `n_action_slots(env)`). The derived-
  dimension discipline is what keeps them instance-agnostic in form while
  instance-sized in fact. (The audit's three-writer FEATURE_LAYOUT finding is
  exactly a case where this seam was not kept clean enough — see ADR-0011.)

## What a different OR problem would actually require

A useful concrete sizing. To retarget chocofarm to a *different*
adaptive-stochastic-orienteering / belief-MDP instance (not FFXIII):

- **Replace** (Band 3): the instance file, the detection geometry, the
  game-specific loader. The observation model would be re-derived for the new
  problem's sensing structure.
- **Keep, parameterized** (Band 2): the belief mechanics, Dinkelbach,
  orienteering, all eight solvers, the AZ stack, the dual bound, the
  analyzer — they are phrased over `worlds`/`value`/`N`/`K`/`cover` and a
  distance function, so a new instance with the same primitives reuses them.
- **No change** (Band 1): the env/Policy seam. A new instance is a new
  `Environment` and the same `Policy` subclasses drive it.

The Band-2 surface is the overwhelming majority of the codebase by line
count, which is *why* this is a research toolkit for the problem class rather
than a single hardcoded solver — and it is so because the env/Policy seam and
the derived-dimension discipline were honored as the code grew.

## What a different game (same OR shape) would require

The inverse partition: replace **only** the Band-3 instance data and
geometry; keep Band 1 and Band 2 entirely. This is the cheaper port, because
the OR machinery is already instance-agnostic — it is the partition the
copy-on-write `with_scenario`/`restrict` seams were built to make cheap.

## Consequences

### Positive

- **New modules are evaluated against the boundary at design time.** The two
  questions are fast to ask and clarify the shape of new code (an FFXIII fact
  goes in Band 3 and is isolated; a belief/rate concept goes in Band 2 and is
  named for the class).
- **Auditability of coupling.** A future maintainer (or a port adopter) has
  this map as a starting point.
- **Explicit seams without premature extraction.** The het-values experiment
  gets the right shape (a `Scenario` sweep over Band-2 machinery) without
  paying for an abstraction nobody needs yet.

### Negative

- **The principle is policy, not mechanism.** A contributor who doesn't ask
  the question won't have a tool catch them. Like ADR-0002, the discipline
  lives in review. (Unlike LengYue, chocofarm has no band-conformance CI
  check; that would be the mechanization trigger — ADR-0011 Rule 1.)
- **The inventory will drift.** As modules change, their band assignments may
  shift. This document carries band *definitions* (which drift slowly), not a
  per-file tag list (which would rot — the audit is the per-file coupling
  evidence, and it is point-in-time).

### Neutral

- **No code change today.** This ADR documents existing structure and a
  discipline for future structure. Existing code is not refactored against it.

## Revisit when…

1. **A second concrete instance materializes** (a different OR problem, or a
   different game with the same shape). At that point, extraction stops being
   premature — the second use case is the trigger that flips the cost-benefit,
   and the seams documented here become the natural extraction points.
2. **The instance data and the machinery drift apart.** If a Band-2 module
   accretes FFXIII facts (a hardcoded treasure id, a teleport name), the band
   boundary has leaked; that is the canary this map exists to catch.
3. **A band classification turns out wrong in practice.** E.g. if a module
   thought Band 2 turns out to be far more FFXIII-coupled once examined, the
   band moves and the principle's application to it changes.
4. **The two-question thought experiment stops being useful.** If the project
   commits to FFXIII-only forever, the principle relaxes — though even then,
   the seam-design discipline produces better code, so it is worth retaining
   as a heuristic.

## Related

- **ADR-0001 (immutability and copy-on-write).** The same philosophy —
  declarations match actual behavior, no aspirational structure — applied to
  the scenario/restriction seams that keep Band-2 machinery sharable across
  Band-3 instance changes.
- **ADR-0002 (fail loudly).** The env's config validation (a Band-1/2
  surface) fails loud at the instance boundary; a Band-3 instance change that
  produces a wrong-length value vector is caught there.
- **ADR-0011 (mechanization discipline).** A band-conformance check would be
  this ADR's mechanization; its absence is a declared review-only enforcement
  surface, not an oversight.
- **The 2026-06-15 architectural audit** — the per-module coupling evidence
  (the env/Policy seam praise, the FEATURE_LAYOUT seam finding, the
  analyzer's orphaned-but-reusable status) is the point-in-time substrate
  this map abstracts.

## Not goals (explicit)

- **Not a refactoring mandate.** Existing code stays put.
- **Not an abstraction-extraction roadmap.** No Ports are being declared or
  planned. The seams are designed; the abstractions are not extracted.
- **Not a portability promise.** We are not committing to ever shipping a
  different instance. The discipline produces better code even in an
  FFXIII-only future, which is the actual justification.

## License

Public Domain (The Unlicense).
