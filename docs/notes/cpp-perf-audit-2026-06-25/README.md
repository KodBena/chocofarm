# C++ producer/engine — soundness audit + micro-optimization session (2026-06-25)

Session record of the post-Option-B/prior_d perf+soundness pass on the tlab generator
(`feat/tlab-real-generators`). Measurements are gitignored under `~/w/vdc/chocobo/runs/tlab/`;
these are the readable markdown findings copied in-tree for review. Point-in-time records
(ADR-0005 Rule 8/9 — the audit is recorded verbatim; not retro-edited).

## Contents
1. **[01-soundness-audit.md](01-soundness-audit.md)** — the lean 4-auditor + synthesis soundness audit
   (14 ranked "spends a resource for no gain" findings; no correctness/equivalence hazard found).
2. **[02-microopt-story.md](02-microopt-story.md)** — per-change perf story (throughput / pipeline /
   cache / prefetch) for the micro-opts attempted this session, with the SUMMARY + remaining-tier.
3. **[03-generation-profile.md](03-generation-profile.md)** — the pure-generation profile + topdown
   pipeline diagnostics that motivated the micro-opts (belief_features = 55% hotspot).
4. **[04-cursor-cache-analysis.md](04-cursor-cache-analysis.md)** — cache stats behind the Option-B win.
5. **[05-producer-memory-profile.md](05-producer-memory-profile.md)** — where producer memory lives
   (node arena ~98%; the smaps method, since heaptrack is blind to the mmap arena).

## Headline outcomes (landed this session, on `feat/tlab-real-generators`)
- **Option B (cursor)** replaced the fiber engine: ~1.6% producer CPU, sounder (P9, no boost/mmap), merged.
- **prior_d removed**: −17% producer RSS (smaller `GumbelNode`), bit-identical, merged.
- **Micro-opts** (this batch): `eval_finish` workspace (C) and `collected_features` inline (D) — small
  cycle/cache wins, cumulative prefetch-miss −35%; admission-guard legibility (#8/#13). Candidate A
  (int-types) was **refuted** — the lever was in the dead flat belief arm; production runs the bitset
  popcount, already int32.
- **Remaining tier** (E geometry single-home, #6 boundary set, B ArenaPool — the HIGH memory win): each
  has a design decision on validated core; flagged in 02 for a deliberate pass rather than rushed.

Public Domain (The Unlicense).
