# Pure-generation profile + pipeline diagnostics — 2026-06-25

**Stamp:** `feat/tlab-real-generators` @ `901be39` (post cursor + prior_d-removal). `leaf-gen-prof`
(= leaf_cpu_microbench) -O3 symbols, `nice -n -19 taskset -c 3`, `--mode direct` (pure producer
search compute; the DetNet leaf stands in for the server-side net). Core 3 per the single-core rule.

## Pipeline diagnostics (perf topdown L1)
| category | share | reading |
| --- | --- | --- |
| Retiring | **56.9%** | useful work |
| Backend-bound | **24.5%** | the largest stall — port pressure / data |
| Frontend-bound | 12.8% | i-cache / decode |
| Bad-speculation | ~5.8% | branch mispredicts (modest) |

~43% non-retiring; the backend (execution ports / data) is the headroom, with only modest
bad-speculation (so signed/unsigned *mispredict* effects are not the story).

## Code-level hotspots (perf record, flat self-time)
| function | self% | note |
| --- | --- | --- |
| **`belief_features`** | **55.0%** | the O(nb·(N+nD)) belief sweep — THE hotspot |
| `__sin_fma`/`exp`/`hypot` | ~6.5% | mostly the DetNet stand-in (server-side in prod — discount) + geometry hypot (~1%) |
| `Environment::apply` | 3.4% | env step |
| `belief_key` | 2.6% | the transposition fingerprint (Zobrist candidate) |
| `evaluate` / `puct_select` | 2.4 / 2.3% | |
| `masked_softmax`/`build_into`/`descend` | 1.4 / 0.9 / 0.9% | |
| `GBeliefChildKeyHash` find | 0.55% | children-map hash (Zobrist candidate) |

`belief_features` dominates; everything else is single-digit. (`sin`/`exp` are the DetNet test leaf,
NOT a real producer cost — the real net runs on the server.)

## The two micro-opt candidates, judged against the data
### 1. Lying int types — REAL, and in the 55% hotspot
`belief_features_nonempty` (`features.cpp:189-196`) accumulates the column sums into **`int64_t`
bit_cnt/det_cnt**, but counts ≤ nb ≤ 15504 fit `uint32` (~280,000× headroom). Consequences (confirmed
in the disassembly: `vpand`/`vpcmpeqd`/`vpaddq`/`vpmovzxdq`/`vfmadd`, vectorized):
- `int64 += uint32` forces the **`vpmovzxdq`** 32→64 widening every iteration.
- `int64` packs 4-wide (`vpaddq`); `int32` packs **8-wide** (`vpaddd`) → **2× SIMD lane density** on
  the dominant reduction.
- **Bit-exact** to narrow: the only consumer is `(double)count * inv` (exact for count ≤ 2^53) and
  `0 < cnt < nb` (fine in 32-bit). P6-clean.
- The mechanism is accumulator-over-width (halved throughput), NOT a signed/unsigned mispredict
  (bad-spec is 5.8%; the loop is unsigned zero-extend, no sign-ext). High-leverage; a clean A/B.
- **CAVEAT:** this bench hits the **flat** arm (`belief_features_nonempty`); the `BitsetBelief` arm is a
  separate `popcount_and` kernel where this lever does NOT apply. Confirm production uses the flat arm
  before optimizing it.

### 2. Zobrist hash — LOW ROI
The hash machinery is ~3% total: `belief_key` 2.6% + `GBeliefChildKeyHash` map-find 0.55%. Zobrist
would shave part of `belief_key` (and only if its fingerprint is recomputed rather than incremental).
Not worth it next to the 55% sweep; deprioritize.

## Recommendation (ADR-0009 — measure before/after if pursued)
Pursue the **int64→uint32 accumulator narrowing in `belief_features_nonempty`** (in the 55% hotspot,
2× SIMD density, bit-exact) — measure the function's self-time + the bit-identity parity before/after.
Deprioritize Zobrist (3%). The biggest fish remains the belief sweep itself; the int-narrowing is the
one candidate that lands inside it.

## Files
`gen.data` (perf record). Re-run: `perf stat -M TopdownL1` + `perf record -e cycles` on `leaf-gen-prof
--mode direct`, core 3.

Public Domain (The Unlicense).
