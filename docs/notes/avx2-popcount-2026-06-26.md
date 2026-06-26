# AVX2 vpshufb popcount swap — 2026-06-26

Branch feat/tlab-avx2-popcount off feat/tlab-phantom-counts (0a7dd19). The de-risk's lever #1, landed in
production `popcount_and` + `popcount_all` (cpp/include/chocofarm/belief_bitset_ops.hpp): the scalar
`std::popcount` per-word loop -> an AVX2 vpshufb (nibble-LUT) popcount doing 4 words/instr + scalar tail.
popcount is EXACT + order-independent, so byte-identical counts. Build -march=native (AVX2).

## Bit-identity — ALL PASS (independently re-run on the committed tree)
belief-sweep oracle byte-for-byte + flat-vs-bitset byte-identical; cursor-proto PASS; gumbel_logic PASS;
gumbel_precision 144/144; quantity-elision PASS. Build GREEN default + (the parity now runs against a
FRESHLY-BUILT gumbel-dump — see the false-green finding below).

## Performance — MAJOR win
A/B (core 3, nice -19, 10 interleaved reps, bootstrap CI):
| path | scalar | AVX2 | Δ | 95% CI |
| --- | --- | --- | --- | --- |
| producer cursor | 6921 us/dec | 5155 us/dec | **−25.6%** | [−26.2%, −25.1%] |
| producer direct | 6954 us/dec | 5193 us/dec | **−25.3%** | [−25.7%, −24.8%] |
| belief_features only (bench) | 11.82 ns/world | 10.42 ns/world | −13% | (tight) |

The producer win (−25.5%) >> the belief_features-only win (−13%) because the bench exercises only
`popcount_and`, but the SEARCH also hammers `popcount_all` (the cached `count_` recompute on EVERY belief
filter during descent) — AVX2-ing both is the ~1.34x producer speedup.

## CPU metrics (perf -r 3, cursor, loadavg 0.29)
| | scalar | AVX2 |
| --- | --- | --- |
| cycles | 17.37e9 | 12.94e9 (**−25.5%**) |
| instructions | 45.47e9 | 27.76e9 (**−38.9%**) |
| IPC | 2.62 | 2.14 |
| retiring | 58.7% | 53.8% |
| L1d-miss | 145.7M | 218.5M (256-bit loads) |
| L1i-miss | 141.5M | 123.7M |
Mechanism: an INSTRUCTION-COUNT collapse (−38.9%) — vpshufb does 4 words/instr vs scalar POPCNT's 1.
popcount was a larger share of producer INSTRUCTIONS than its 55% time-share implied. IPC falls (heavier
AVX2 ops) but the instruction cut dominates. disasm: vpshufb x14 present (the kernel); popcnt remains for
the scalar tails. NOT a stall change — pure throughput of the primitive.

## Finding: a fixed FALSE-GREEN in the parity gates
`gumbel-dump` (the gumbel_logic/precision harness binary) was STALE — the SearchBudget typing (aa63507)
broke its `cfg.m = int` writes, but the tool-retrofit (db599b3) MISSED it, so the parity scripts had been
running a stale PRE-typing binary (silent false-green). Retrofitted gumbel_dump.cpp's config writes
(SearchBudget ACL) + rebuilt -> parity now 144/144 against the CURRENT typed+AVX2 core. (The belief-sweep
oracle, which DOES rebuild, was always honest — it's the decisive popcount gate.)

Public Domain (The Unlicense).
