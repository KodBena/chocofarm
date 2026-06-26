# De-risk: batched belief-featurization (idea #3) — 2026-06-26

Branch `feat/tlab-batch-insrc` (off `9324a01`). An **additive PROTOTYPE bench** — it does NOT touch the
production search / feature core. Frame: ADR-0000 / ADR-0012 / ADR-0009 (measure, don't assume).

> Note on orientation inputs: the brief named `docs/notes/batchpredict-throughput-design-2026-06-26.md`
> as the seam + de-risk plan to read first; **that file does not exist on this branch** (only
> `phantom-typing-2026-06-26.md`, read end to end). I proceeded from the de-risk plan stated in the brief
> itself (idea #3, the hypothesis, the bit-identity + A/B gates). Flagging the missing doc per ADR-0002.

> Provenance (ADR-0013 — verify the artifact, not the claim): the first draft of this note concluded a flat
> "NO-GO, ceiling ≤0%" from a SINGLE batched kernel (the scalar loop-transpose). An out-of-frame
> hack-rationalization audit (the artifact is in the branch's session record) flagged that as an
> **UNDISCHARGED-HACK**: the strongest natural batched layout for a popcount-over-masks kernel is word-tiled
> SIMD (vpshufb), not a loop transpose, and that was never tested — the loss of the weak kernel was
> generalized to the whole idea. The auditor built a stronger kernel that won ~2.2x. This note is the
> corrected, fuller version: all four layouts are now in the committed prototype, bit-identity-gated, and
> A/B'd against the live instance masks.

## The question
`belief_features` (the popcount sweep — on the live env the BITSET arm `belief_features_bitset`: masked-AND
+ popcount over the env-static treasure/detector masks) is ~55% of producer compute. The cursor parks B
leaves per RTT. **Hypothesis (idea #3):** featurizing the B parked beliefs *together*, restructured for
locality, amortizes mask loads and improves cache/SIMD. Is the batched sweep faster, and by how much —
BEFORE building the BatchPredict seam?

## The layouts (four kernels, all Phase-2-shared and bit-identity-gated)
The production per-leaf arm, per belief, loops the N+nD mask rows inner
(`popcount_and(belief.live(), mask_row)` = a scalar `std::popcount(b[w]&m[w])` chain over kW64=243 words).
The four candidates (all stage the SAME integer counts — popcount is order-independent — then run the
byte-identical production Phase-2 `* inv` maps):

| kernel | what changes | seam? |
| --- | --- | --- |
| `bat-scalar` | mask-major loop transpose, SAME scalar primitive | (this is idea #3 as literally stated) |
| `sep-avx2` | per-leaf, AVX2 vpshufb popcount primitive, NO batching | **no** — a per-leaf kernel rewrite |
| `bat-avx2` | mask-major + AVX2 primitive | yes (batched shape) |
| `bat-avx2-tile` | mask-major + AVX2 + mask word held resident across a 4-belief register tile | **yes** — the batch-specific locality |

Prototype: `cpp/src/batched_featurization_proto.cpp` (target `chocofarm-batched-featurization-proto`).
Realistic beliefs: the full prior + rank-strided filtered subsets spanning nb ∈ {8 … 15504}, cycled +
shuffled per batch (the belief-sweep oracle/bench style).

## Bit-identity: PASS
2880 byte-for-byte comparisons (batched row == per-leaf `chocofarm::belief_features`) across all THREE
batched kernels × B ∈ {8,16,32,64} × 8 batches — every field byte-equal. `belief_features` (the BITSET arm)
is the reference (the belief-sweep oracle nets IT against an independent naive count). The AVX2/tiled
kernels stage the identical integer counts; the gate proves it.

## A/B: layout-only is NULL, but the SIMD primitive + batch tiling WIN big
Env: N=20, nD=44, |worlds|=15504, kW64=243 → mask matrix **121.5 KiB**. Host: i5-6600 (Skylake), L1d 32K,
**L2 4 MiB/core**, L3 16 MiB, AVX2/BMI2 (`-march=native`). Method: interleaved paired reps (order
alternated), warmup discarded, median per-batch µs, bootstrap-95% CI of the paired ratio (candidate/
separate). `nice -n -19 taskset -c 3`, loadavg < 0.3, 13 reps × 0.25 s/point. Two independent runs agree;
representative (run 2):

| candidate | B=8 speedup | B=16 | B=32 | B=64 | verdict |
| --- | --- | --- | --- | --- | --- |
| `bat-scalar` (idea #3 as stated) | +1.2% | +0.1% | −0.5% | −0.9% | **NULL** (CIs straddle/just below 0) |
| `sep-avx2` (primitive swap, NO seam) | +75.3% | +75.0% | +74.2% | +74.2% | FASTER |
| `bat-avx2` (primitive swap, batched) | +113.9% | +93.5% | +87.4% | +83.7% | FASTER |
| `bat-avx2-tile` (full batch-specific) | +133.9% | +127.0% | +124.4% | +124.6% | FASTER |

(speedup% = (1/ratio − 1)·100 vs the production per-leaf scalar arm; all CIs of the AVX2 arms wholly < 1.0.)

**The decomposition that matters** (the seam-funding question):
- `sep-avx2` vs `separate` = **the primitive swap (scalar POPCNT → AVX2 vpshufb), +~74%, NEEDS NO SEAM** —
  it speeds the existing per-leaf path. This is the dominant win and it is free of the batch refactor.
- `bat-avx2-tile` vs `sep-avx2` = **the genuinely BATCH-SPECIFIC increment, +~29–34%** (B=8: 41.7→31.2 µs;
  B=64: 335.9→260.9 µs) — the part that requires processing B beliefs together (mask word resident across
  the belief tile). Stable across B and across runs.

## Mechanism (attributed, not conjectured)
The sweep is **popcount-THROUGHPUT-bound with the masks L2-resident** — the mask matrix (121.5 KiB) fits
the 4 MiB L2, so almost nothing re-streams from memory (`perf stat`: IPC ≈ 3.3, 71% retiring topdown,
0.69% of loads reach L3). The *first draft's error* was concluding "compute-bound ⇒ nothing to win." On
Skylake, scalar `POPCNT` is **port-1-only, 1 word/cycle**; the AVX2 `vpshufb` nibble-LUT popcount processes
**4 words/instruction across multiple ports**. "Throughput-bound on a suboptimal primitive" is exactly the
regime where a better SIMD primitive wins — hence the +74% from the primitive swap alone. The additional
batch-specific +~30% comes from keeping the mask word resident in registers across a 4-belief tile, so each
mask load is reused 4× before eviction (the locality story idea #3 named — it just needed the SIMD primitive
to expose it; with the scalar primitive, port-1 saturation hides the locality, which is why `bat-scalar` is
null). The loop-transpose-only kernel changes access order but not the primitive, so it cannot win — the
original null result is correct *for that kernel*, and wrong only as a generalization to the whole idea.

## GO / NO-GO
**Idea #3 as literally stated (cache/loop-order restructure with the same scalar primitive): NO-GO** —
measured null (+1% to −1%). The masks already live in L2; reordering loads buys nothing while the scalar
POPCNT primitive is the bottleneck.

**The de-risk QUESTION ("can batching the featurization win?"): conditional GO, with the win decomposed:**
- **First, and independently of any seam: swap the per-leaf belief-features popcount to AVX2 vpshufb.**
  ~+74% on the ~55%-of-producer sweep, no batch refactor, helps the existing per-leaf path today. This is
  the highest-leverage, lowest-risk move and should happen regardless of the BatchPredict decision. (It is
  a production-core change — out of scope for this additive prototype — but the prototype establishes the
  number.)
- **Then, the batch-specific increment is real: ~+29–34%** on top of the SIMD primitive, requiring the
  in-process batched featurizer (mask-resident tiling over the B parked beliefs). **Expected ceiling for the
  #3 seam build, measured against an already-SIMD per-leaf baseline: ~+30%** (NOT the headline +124%, which
  mostly belongs to the no-seam primitive swap). Whether ~+30% of ~55%-of-producer-compute justifies the
  seam's complexity is the build-decision; this de-risk supplies the honest number, not a verdict on the
  refactor's worth.

Caveats (honest scope): (1) the batched bit-identity is exact integer counts, so the seam stays at the
P6 behavioral bar like the scalar arm — no behavior change. (2) Bench-isolated Phase-1 popcount (Phase 2 is
identical across arms and small); a real producer also pays belief-filter + net RTT, so the producer-level
win is this fraction of the ~55% sweep bucket, not 30% of the whole producer. (3) Holds while the mask
matrix fits L2; a much larger |worlds|/kW64 (masks spilling L2) would re-introduce a memory-traffic term
that the batched/tiled layout would help with MORE — re-measure with this prototype if the dims grow.

## Artifacts
- Prototype + CMake target: `cpp/src/batched_featurization_proto.cpp`, `cpp/CMakeLists.txt`.
- Run: `cmake --build cpp/build --target chocofarm-batched-featurization-proto` then
  `nice -n -19 taskset -c 3 ./cpp/build/chocofarm-batched-featurization-proto --instance
  chocofarm/data/instance.json --faces chocofarm/data/faces.json`.

Public Domain (The Unlicense).
