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

Public Domain (The Unlicense).
