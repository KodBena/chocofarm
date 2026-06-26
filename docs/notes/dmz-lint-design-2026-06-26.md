# DMZ lint — design + policy (2026-06-26)

The capstone that keeps the phantom-type discipline from relapsing: a CI gate that flags raw-integer
**domain leaks** so they can never silently creep back. Built on the lesson that the discipline is only
durable if mechanized (ADR-0000: mechanize the lapse, don't re-fix instances).

## What it flags (and what it deliberately doesn't)
- **Target: struct/class FIELDS typed as a bare builtin integer** (`int`/`long`/`short`/`char`/`size_t`/
  `uNN_t`) — these are where a domain quantity (a count/id/rank) leaks as un-typed state. High-signal:
  fields are few and load-bearing.
- **NOT params/returns** (a first cut matched them → 174 hits, ~all boundaries: `std::pmr` overrides, the
  env public API, generic interfaces, lambda hash-mixers). Function signatures are a *boundary* surface, not
  domain state; policing them blanket is pure noise. (A future stricter pass could police a curated subset.)
- **NOT loop-counter locals.** The measured-hot ones are kept raw by the loop-mod carve-out (typing them
  cost ~1% via frontend bloat — see phantom-typing-2026-06-26.md); the generator-fed ones are caught by
  `modernize-loop-convert` instead.
- **NOT the domain aliases / strong types.** `World`/`world_mask_t`/`count_t` (deliberate domain aliases)
  and `Quantity<Tag,Rep>` are typed; the driver keeps only BARE-int *spellings* (source-verified), so an
  alias-typed field is not a hit.

## How it works
- `tools/lint/dmz.clang-query` — AST matcher: `fieldDecl(hasType(isInteger()), not bool, in our tree)`.
- `tools/lint/dmz-lint.sh` — runs it per-TU over `cpp/build/compile_commands.json`, dedups header hits,
  drops `cpp/build/_deps` (vendored), reads each location's source, keeps only bare-int spellings, then
  clears anything **(a)** in a DMZ file or **(b)** carrying a `// NOLINT(dmz...)` marker. Exit 1 on any
  remaining holdout. AST-anchored, source-verified, two-axis allowlist.
- Companion `.clang-tidy`: `modernize-loop-convert` (the generator-fed loop antipattern) + narrowing checks;
  `-Wsign-compare` belongs in the CI build flags (already clean post-phantom).

## The two-axis allowlist
1. **DMZ files** (raw int is legitimately the representation): `quantity.hpp` (the strong-type machinery's
   `Rep`), `world.hpp`/`domains.hpp`/`proc_domains.hpp` (domain aliases + tags + Rep typedefs),
   `wire_spec.hpp` (the on-wire `count_t` codec), `collected_set.hpp` (the `uint64` bitmask storage).
2. **`// NOLINT(dmz: <reason>)` markers** — greppable, self-justifying, per-field. Two justified kinds:
   - **boundary/seam**: a value that genuinely lives as raw int at a boundary (the `std::pmr` allocator's
     byte sizes; the dual-domain `Action.i` that carries treasure-id XOR face-id pending an Action reshape).
   - **measured-hot (`NOLINT(perf)`)**: a hot-path integer where typing it *measurably regresses* (the
     loop-mod evidence). Policy: a hot field stays raw ONLY with a perf A/B on record; default is to type it.

## The current holdout inventory (18 fields) — scope-completion TODO
| field | domain | disposition |
| --- | --- | --- |
| `env.hpp` `entry_idx_` | TeleportId | TYPE (cold) |
| `gumbel.hpp` `n_spent` | sim count | TYPE (bookkeeping) |
| `gumbel.hpp` `survivor_slot` | ActionSlot (−1) | TYPE → `optional<ActionSlot>` |
| `features.hpp` `belief_cache_n_`/`belief_cache_cap_` | cache counts | TYPE (cache mgmt, cold) |
| `env.hpp` `count_` | WorldCount | A/B (hot compares) → type or `NOLINT(perf)` |
| `env.hpp` `kw64_`/`kW64_` | WordCount | A/B (hot word-loop bounds) |
| `gumbel_cursor.hpp` `node`/`sh_phase_idx_`/`sh_rr_` | indices | A/B (per-step) |
| `gumbel.hpp` `m`/`n_sims`/`c_outcome`/`max_depth` | search budget | A/B (hot loop bounds) — likely `NOLINT(perf)` or a `GumbelConfig` domain |
| `env.hpp` `Action.i` | TreasureId XOR FaceId | `NOLINT(dmz)` — dual-domain seam, Action reshape filed |
| `releasing_arena.hpp` `map_len`/`pad` | byte length | `NOLINT(dmz)` — `std::pmr`/mmap byte-size boundary |

Expectation (from the loop-mod evidence): the cold ones type cleanly; most hot-adjacent ones will regress
and land as `NOLINT(perf)` measured exceptions; the boundaries are `NOLINT(dmz)`. Once every holdout is
typed / DMZ / NOLINT'd, the gate goes green and wires into CI (flag-mode, never blind autofix on the numeric
core — `modernize-loop-convert`'s autofix can reorder float accumulation; conversions are reviewed + pass the
byte-for-byte oracle).

## Amendment 2026-06-26 — final dispositions: all 11 FIELD holdouts resolved, gate GREEN (exit 0)

The 11 raw-int FIELD holdouts the table above projected were resolved (the phantom-typing completion
mandate, branch `feat/tlab-phantom-counts`). The **expectation was WRONG in its key prediction**: the
loop-mod evidence led the table to expect "most hot-adjacent ones will regress → `NOLINT(perf)`." In the
event, **every hot field typed neutral-or-FASTER** — none regressed, so there is not a single `NOLINT(perf)`
on the list. The deep lesson held: the loop-mod regression was the COUNTER IDIOM in the per-WORLD hot loop
(`belief_features`), not stored values read in compares/bounds nor cursors that step per-candidate. The full
per-field evidence (bit-identity + the A/B numbers) is in `docs/notes/phantom-mandate-ledger-2026-06-26.md`.

| field | final disposition (commit) |
| --- | --- |
| `env.hpp` `entry_idx_` | TYPED `TeleportId` (`600dc9b`, cold) |
| `gumbel.hpp` `Decision::n_spent` | TYPED `SimBudget` (`600dc9b`, cold) |
| `gumbel.hpp` `Decision::survivor_slot` | TYPED `std::optional<SlotIndex>` — `-1` sentinel killed (`600dc9b`, cold) |
| `env.hpp` `BitsetBelief::count_` | TYPED `WorldCount` — A/B cursor -0.49% / direct -1.14%, both faster (`8031bd6`) |
| `env.hpp` `BitsetBelief::kw64_` | TYPED `WordCount` (`8031bd6`) |
| `env.hpp` `Environment::kW64_` | TYPED `WordCount` (`8031bd6`) |
| `gumbel.hpp` `GumbelConfig::m`/`n_sims`/`c_outcome`/`max_depth` | TYPED `CandidateCount`/`SimBudget`/`OutcomeIndex`/`PlyDepth` — NOT a merged `SearchBudget` (the existing domains are right-grained); A/B cursor -0.33% / direct neutral (`aa63507`) |
| `gumbel_cursor.hpp` `DescendFrame::node` | TYPED `NodeIndex` (new domain; also `GumbelNode::children` value + run_search locals); A/B neutral (`dac3c5b`) |
| `gumbel_cursor.hpp` `sh_rr_`/`sh_phase_idx_` | TYPED `OutcomeIndex` (the documented affine-cursor domain); A/B neutral (`dac3c5b`) |
| `features.hpp` `belief_cache_n_`/`belief_cache_cap_` | `NOLINT(dmz)` — generic cache-residency count/capacity, no class at stake (ADR-0000 over-typing carve; out-of-frame review confirmed) (`6c0c66c`) |

The two NOLINTs are `NOLINT(dmz: ...)` (no-class / generic-capacity), NOT `NOLINT(perf)`. The earlier
`NOLINT(dmz)` config-knob markers (`375fd51`) were REPLACED by real types (`aa63507`); the
`Action.i` dual-domain seam and `releasing_arena.hpp` byte-size boundaries remain `NOLINT(dmz)` boundaries
as the table above filed them (not FIELD holdouts on the worklist). `tools/lint/dmz-lint.sh` now exits 0.

Public Domain (The Unlicense).
