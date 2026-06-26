# Phantom-typing mandate — completion ledger (2026-06-26)

The running record of the DMZ-holdout resolution (the "type-or-carve every remaining raw-int FIELD
holdout so `tools/lint/dmz-lint.sh` exits 0" mandate, branch `feat/tlab-phantom-counts`). Anything that
needs the maintainer's intervention or review is recorded HERE; carved fields, design ambiguities, and any
gate that could not be satisfied are logged with their evidence. This is a point-in-time record (ADR-0005
Rule 8): append, do not retro-edit.

Baseline: HEAD `375fd51` (DMZ lint reported 11 raw-int FIELD holdouts). Toolchain GCC 15.2.1, `-O3
-march=native -std=c++23 -Wall -Wextra`. Bit-identity gates run on the live instance (N=20, nD=44,
|worlds|=15504, kW64=243), core 3, nice -19. A/B: the saved `leaf_cpu_microbench` (`--mode cursor` AND
`--mode direct`, 700 decisions, 12 interleaved reps + warmup discard, median + bootstrap 95% CI on the
paired AFTER-BEFORE diff), BEFORE/AFTER cores each compiled in their matching git state.

## Per-field disposition

### Cold (typed, bit-identity only — not hot) — commit `600dc9b`
| field | domain | disposition |
| --- | --- | --- |
| `env.hpp` `Environment::entry_idx_` | `TeleportId` | TYPED. `.value()` at the `inst_.teleports[...]` index + the ctor std::vector-index wrap. |
| `gumbel.hpp` `Decision::n_spent` | `SimBudget` | TYPED. Producers already computed it as SimBudget and `static_cast<int>`-stored it — typing the field DELETES the casts; `to_decision` becomes a same-domain pass-through; `.value()` at the ostream prints (proto/dump). |
| `gumbel.hpp` `Decision::survivor_slot` | `std::optional<SlotIndex>` | TYPED. The `-1 = no survivor` sentinel → typed absence (ADR-0002); `finalize()`'s `assert(!=-1)`+rewrap collapses to `assert(has_value())`+unwrap. |

Bit-identity for the cold commit: oracle PASS (byte-for-byte + flat-vs-bitset byte-identical), cursor-proto
(m24/n256/co2/d24) PASS, gumbel_logic PASS, gumbel_precision 144/144, quantity-elision PASS; build GREEN
default + `-DCHOCO_BELIEF_ZDD=ON`, `-Wall -Wextra` clean.

### Hot belief-arm word/world counts (typed, A/B KEPT) — commit `8031bd6`
| field | domain | disposition |
| --- | --- | --- |
| `env.hpp` `BitsetBelief::count_` | `WorldCount` | TYPED. `popcount_all()` already returns WorldCount → same-domain store. |
| `env.hpp` `BitsetBelief::kw64_` | `WordCount` | TYPED. The live-word stride; loops already wrapped it. |
| `env.hpp` `Environment::kW64_` | `WordCount` | TYPED. ctor uses the named `words_to_words64()` bridge; `kW64()` stays the raw-int public accessor. |

A/B (700 dec, 12 interleaved reps + warmup, core 3 nice -19, loadavg<0.3; BEFORE=`600dc9b`, AFTER=`8031bd6`):
- **cursor**: BEFORE 6976.1 us, AFTER 6942.1 us, paired **-34.1 us (-0.49%)**, bootstrap 95% CI [-0.85%, -0.44%] — SIGNIFICANT SPEEDUP.
- **direct**: BEFORE 7030.4 us, AFTER 6954.9 us, paired **-80.3 us (-1.14%)**, bootstrap 95% CI [-1.22%, -0.85%] — SIGNIFICANT SPEEDUP.

Both CIs exclude 0 on the FASTER side → KEEP. Mechanism (consistent with the prior increment, not re-measured
at the disasm level here): the field types stop forcing the lying-int64 widenings the stored ints sat behind.
Bit-identity: oracle PASS (byte-for-byte + flat-vs-bitset), cursor-proto PASS, gumbel_logic PASS,
gumbel_precision 144/144, quantity-elision PASS; build GREEN default + ZDD, `-Wall -Wextra` clean.

### GumbelConfig search-budget knobs (typed with existing domains, A/B KEPT) — commit `aa63507`
| field | domain | disposition |
| --- | --- | --- |
| `gumbel.hpp` `GumbelConfig::m` | `CandidateCount` | TYPED. |
| `gumbel.hpp` `GumbelConfig::n_sims` | `SimBudget` | TYPED. |
| `gumbel.hpp` `GumbelConfig::c_outcome` | `OutcomeIndex` | TYPED (the existing shared count/index domain). |
| `gumbel.hpp` `GumbelConfig::max_depth` | `PlyDepth` | TYPED. |

**The maintainer-requested SearchBudget test — RESULT: typed with the existing distinct domains, NOT a new
`SearchBudget` type.** The brief proposed minting one unified `SearchBudget` phantom. I interrogated that
against the codebase and did NOT mint it: `domains.hpp` ALREADY mints these four as DISTINCT, right-grained
domains (`CandidateCount`/`SimBudget`/`OutcomeIndex`/`PlyDepth`), and the search core already wrapped each
knob into its proper domain at every use site. A single merged `SearchBudget` would re-collapse the
count-vs-index discipline ADR-0008/domains.hpp deliberately keep (m is a candidate-set width, n_sims a sim
budget, c_outcome a determinization count, max_depth a depth cap — four different things). Typing the FIELDS
with the existing domains is the faithful ADR-0000 answer and DELETES the per-use `static_cast<SearchRep>`
wraps. The crossings now live only at the true boundaries: the CLI parse (gumbel_cursor_proto,
search_runtime_bench), the JSON Port/ACL (`actor_config_from_json` — the domain minima + `std::to_string`
diagnostics), and the ostream prints. **KEPT** (not carved).

A/B (700 dec, 12 interleaved reps + warmup, core 3 nice -19; BEFORE=`8031bd6`, AFTER=`aa63507`):
- **cursor**: paired **-0.33%**, CI [-0.49%, -0.13%] — SIGNIFICANT SPEEDUP.
- **direct**: paired **+0.03%**, CI [-0.11%, +0.20%] — NEUTRAL.
Neither CI excludes 0 on the slow side → KEEP. Bit-identity all PASS; build GREEN default + ZDD, `-Wall
-Wextra` clean.

### Cursor index/cursor bundle (NodeIndex + the SH cursors) — commit `dac3c5b`
| field | domain | disposition |
| --- | --- | --- |
| `gumbel_cursor.hpp` `DescendFrame::node` | `NodeIndex` (new) | TYPED. A new domain minted in domains.hpp — the per-decision GumbelNode-arena index, DISTINCT from a slot/depth (the foreclosed class). Applied also to `GumbelNode::children`'s map value + the `child`/`node` locals in run_search (gumbel.cpp) so the arena-index domain has one home, not just the lint-flagged field. |
| `gumbel_cursor.hpp` `sh_rr_` | `OutcomeIndex` | TYPED. domains.hpp EXPLICITLY assigns "the round-robin survivor cursor (read modulo considered.size())" to the OutcomeIndex affine domain — leaving it raw would contradict the SSOT. (Caught by the out-of-frame review below.) |
| `gumbel_cursor.hpp` `sh_phase_idx_` | `OutcomeIndex` | TYPED (the same affine container-cursor domain, for consistency with sh_rr_). |

**Out-of-frame review (the user's hack-rationalization posture, ADR-0014):** before disposing the
counter-like cursor fields and the cache fields, I commissioned an independent subagent to pressure-test
whether "leave raw + NOLINT" was sound ADR-0000 reasoning or a regression-dodge. Its load-bearing catch:
`sh_rr_` has a DOCUMENTED domain home (domains.hpp's OutcomeIndex comment names the RR survivor cursor), and
its hotter sibling `cur_k_` (same `+ SearchRep{1}` idiom, same drive() loop) was already typed `OutcomeIndex`
and is part of the prior increment's +1.7% net win — so a "measured-hot carve" of `sh_rr_` would be
unjustified. I acted on that: `sh_rr_`/`sh_phase_idx_` TYPED (not carved). The review also confirmed the
belief-cache fields are a genuine no-class case (NOLINT, below).

A/B (counter-like bundle — the regression-risk per the loop-mod lesson; 700 dec, 12 interleaved reps +
warmup, core 3 nice -19, loadavg<0.2; BEFORE=`aa63507`, AFTER=`dac3c5b`; each binary built in its
matching git state — a first run had a wide CI from load contamination on the BEFORE binary's early reps and
was re-run clean):
- **cursor**: BEFORE 6903.8 us, AFTER 6904.7 us, paired **-0.08%**, CI [-0.31%, +0.16%] — NEUTRAL.
- **direct**: BEFORE 6953.2 us, AFTER 6952.1 us, paired **-0.05%**, CI [-0.26%, +0.32%] — NEUTRAL.
Both CIs span 0 → KEEP. The `+ SearchRep{1}` affine-step idiom on `sh_rr_`/`sh_phase_idx_` (the loop-mod
regression risk) is measurement-neutral here: these cursors step per-candidate / per-leftover-sim, not
per-world, so the frontend-bloat cost the hot belief_features counters paid does not materialize (the AFTER
core is +776 bytes, but it does not move the wall). KEPT. Bit-identity all PASS; build GREEN default + ZDD,
`-Wall -Wextra` clean.

### belief-cache capacity family (NOLINT, no class at stake) — commit `dac3c5b`
| field | disposition |
| --- | --- |
| `features.hpp` `belief_cache_n_` | NOLINT(dmz). A generic memo-residency COUNT, only ever compared to its own cap (no confusable currency in scope — never a world/word count). |
| `features.hpp` `belief_cache_cap_` | NOLINT(dmz). A generic cache-backstop CAPACITY (a `CHOCO_BELIEF_CACHE_CAP` tuning knob), not a modeled domain magnitude. |

These are the ADR-0000 "no class at stake" exception, NOT a regression-dodge: the in-source comment
(features.hpp ~227-230) already argued this on ADR-0000 grounds ("type where it carries meaning; a cache
backstop is a generic capacity, not a domain magnitude"), and the out-of-frame review independently confirmed
it (no minted tag exists for "memo residents"; minting `BeliefCacheCount` would foreclose no representable
confusion → the over-typing weaponization ADR-0000 Revisit #2 names). NOLINT with the recorded reason is the
honest disposition. No A/B owed (not a perf carve; a no-class carve).

## Items needing maintainer attention

### Pre-existing broken non-gate targets on the branch (NOT introduced by this mandate)
At baseline `375fd51`, several **non-gate** executables already fail to compile, left red by the earlier
phantom increment (`7966d2f`) which typed `n_action_slots()`/`term_slot()`/`action_to_slot()`/`fb.dim()`
to phantom domains but did not retrofit these out-of-scope tools' raw-int call sites:
`chocofarm-mask-dump`, `chocofarm-nmcs-dump`, `chocofarm-ismcts-dump`, `chocofarm-fiber-proto`,
`chocofarm-cpp-runner` (`main.cpp` + `runner_wire_batched.cpp`). Verified by building the full tree at
`375fd51` (identical error set, before any of my edits). These are OUT of this mandate's scope (not in the
worklist, not in the named gate-target set: `chocofarm_core` + the gate-proto/oracle/elision targets) and
out of the lint's TU set. I did not expand scope to fix them (ADR-0004 scope discipline) — flagged here for
the maintainer. NOTE: my Decision-field change DID update the in-tree consumers I could
(`gumbel_dump.cpp` — builds clean; `fiber_proto.cpp`/`gumbel_cursor_proto.cpp` — `gumbel_cursor_proto`
builds clean; `fiber_proto` remains blocked by the *pre-existing* `DetNet(SlotCount)` error at line 114,
upstream of my edits), so they stay type-consistent for when the maintainer fixes the upstream breakage.

### SearchBudget design ambiguity (the maintainer-requested test) — see below when it lands.
