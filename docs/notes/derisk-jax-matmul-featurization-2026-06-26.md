# De-risk: fused-JAX matmul featurization (idea #1) — 2026-06-26

Prototype-stage measurement + parity for de-risk idea #1 of
`docs/notes/batchpredict-throughput-design-2026-06-26.md`: reframe the producer's `belief_features`
hotspot as a matmul shipped to JAX/XLA, riding the net batch. This note records the **parity verdict,
the wire-size tradeoff, the CPU matmul timing, the provability caveat, and a GO/NO-GO** with the open
questions for the real build. It is evidence for a decision, not a build.

Frame: ADR-0000 (the reframe makes the per-world loop unrepresentable as a matmul), ADR-0012 (the
cross-language SSOT — the C++ `belief_features` is the oracle; the JAX impl must agree within tolerance),
ADR-0009 (measure, don't assert).

## The reframe (what was prototyped)

`belief_features` (`cpp/src/features.cpp`, `belief_features_nonempty`) is two down-the-worlds integer
column-sums over the nb×(N+nD) bit matrix, then a pointwise phase 2:

- `bit_cnt[t] = Σ_w bit_t(w)`, `det_cnt[j] = Σ_w [(w & mask_j) != 0]`  (phase 1, the O(nb·(N+nD)) cost)
- `marg[t] = bit_cnt[t] · inv`, `p_pos[j] = det_cnt[j] · inv` where `inv = 1/nb`
- `informative[j] = (0 < det_cnt[j] < nb)`, `marg_sum = Σ_t marg[t]`, `sharpness = log(nb)/log(Nworlds)`,
  `nonempty = 1`

Phase 1 **is** `counts = belief_indicator · world_feature_matrix`, where `belief_indicator` is the
nb-bit live-world (worlds()-rank) mask and `world_feature_matrix` is the env-static `nworlds×(N+nD)` bit
matrix. The env **already builds** that matrix column-major as rank bitsets: `treasure_mask(t)` is
column `t`, `detector_mask(j)` is column `N+j` (each kW64 u64 words). Batched over B beliefs this is a
`(B×nworlds)·(nworlds×(N+nD))` matmul — XLA-native.

## Artifacts (additive prototype — no production path touched)

- **C++ export** — `cpp/src/belief_features_export.cpp` (CMake target
  `chocofarm-belief-features-export`): dumps one JSON blob with the env-static column bitsets
  (`treasure_mask`/`detector_mask`, the world_feature_matrix) + a spread of reference
  `(belief_indicator rank-bitset, C++ belief_features double-precision)` pairs — the parity oracle.
- **JAX parity/bench** — `cpp/parity/jax_matmul_featurization.py`: unpacks the matrix, runs the matmul
  + phase-2 maps in JAX (CPU), diffs against the oracle (f32 **and** f64), and measures wire-size +
  CPU matmul timing for B=8/32/64.

Live instance: N=20, nD=44, nworlds=15504, kW64=243, feat_dim=241. 11 reference beliefs spanning
nb ∈ {1, 2, 3, 5, 16, 100, 1000, 7752, 15504, 2215, 1193}.

## (a) Parity verdict — WITHIN TOLERANCE

Max abs / max rel diff per feature block, JAX vs the C++ double oracle:

| block         | JAX f64 (abs) | JAX f32 (abs) | JAX f32 (rel) |
| ---           | ---           | ---           | ---           |
| marg          | 0             | 4.44e-08      | 8.24e-08      |
| p_pos         | 0             | 5.44e-08      | 1.00e-07      |
| informative   | 0 (exact)     | **0 (exact)** | **0 (exact)** |
| marg_sum      | 0             | 9.54e-07      | 1.91e-07      |
| sharpness     | 1.11e-16      | 7.27e-08      | 7.83e-08      |
| nonempty      | 0             | 0 (exact)     | 0 (exact)     |

- **f64 JAX reproduces the C++ double oracle exactly** (worst 1.1e-16, a single ULP in `log` for
  `sharpness`). This proves the **matmul reframe is denotationally identical** to the C++ per-world
  sweep — there is no reframing error; the only gap is float width.
- **f32 JAX**: worst abs diff **9.5e-7** (`marg_sum`, the accumulation of 20 `marg` terms); per-block
  ~4–5e-8 for `marg`/`p_pos`. This is **two orders of magnitude inside** the ADR-0012 P6 behavioral bar
  the net-forward parity already uses (max|Δ| < 1e-4, `net_dump.cpp`). VERDICT: **within tolerance.**
- **`informative` is BIT-EXACT in f32** (0.0). This is load-bearing: `informative` (and `available`,
  derived from `marg>0`) is the **legal mask** — a logic invariant ADR-0012 holds bit-exact. It survives
  f32 because the matmul produces **exact integer counts**: f32 represents every integer ≤ 2²⁴ =
  16 777 216, and nworlds=15504 ≪ that, so a 0/1·0/1 matmul accumulates with no rounding. The only f32
  error enters at the `· inv` divide (marg/p_pos), never at the count comparison `informative` uses.

## (b) Wire payload tradeoff

| payload                       | size    |
| ---                           | ---     |
| belief rank-bitset (kW64×8)   | 1944 B  |
| feature vector (241 f32 × 4)  | 964 B   |

**Ratio 2.02× (+980 B per leaf).** The bitset is **fixed-size regardless of nb** (it spans the full
rank space). A sparse rank-list encoding beats it only for nb ≪ nworlds/64 (~242 ranks); narrowed
beliefs deep in the search may qualify, but the early-decision beliefs (large nb) do not, so 1944 B is
the planning number. This is the compute-for-bandwidth trade: pay 2× the wire to move the O(nb·(N+nD))
featurization off the producer and onto the (batched, GPU-amortizable) net side.

## (c) CPU JAX matmul timing — feasible

Median of 50 runs after warmup, float32, pinned to one core (`taskset -c 0`):

| B   | median  | per-belief |
| --- | ---     | ---        |
| 8   | 0.45 ms | 56 us      |
| 32  | 1.43 ms | 45 us      |
| 64  | 2.73 ms | 43 us      |

Not absurd — sub-3 ms for B=64 on **CPU**, with the per-belief cost amortizing as B grows (the matmul is
the full dense `nworlds×(N+nD)` regardless of nb). The GPU only helps later (the net forward is the part
that GPU-amortizes); this confirms the featurization itself is not a CPU bottleneck at realistic B.
NB the matmul cost is dominated by **nworlds** (15504), not nb — the indicator is mostly zeros for
narrowed beliefs, so a deep-search belief costs the same as a full one in this dense form.

## Provability caveat (the cost to name)

The C++ belief-sweep has a **bit-exact in-language oracle** (`belief_sweep_oracle_check.cpp`): the
production sweep is netted byte-for-byte against an independent naive count. Moving the featurization to
JAX f32 **forfeits that bit-exact oracle at this boundary**: marg/p_pos become a P6 *behavioral* bar
(≤ ~1e-6 here), not a bit-exact one. What is preserved:

- `informative`/`available` (the legal mask) stay **bit-exact** (exact integer counts, §a) — the logic
  invariant ADR-0012 protects is not weakened. *Contingent on* nworlds < 2²⁴; a larger instance would
  lose count-exactness and must be re-measured before reuse (the f32-count argument is a measured
  witness on **this** instance, not a proof for all N — model-bound-is-conjecture-not-witness).
- f64 JAX is bit-identical to the C++ double oracle, so a **provability fallback exists**: run the JAX
  side in x64 to recover the exact oracle for verification, even if production serves f32.

The honest framing: the 2× wire is justified **iff** the offload (featurization off the producer's hot
path, fused into the net batch) buys throughput exceeding the bandwidth cost — and *that* A/B is exactly
what the design note's seam (#3 in-process vs #1 fused-JAX behind one `BatchPredict`) is built to
measure. This de-risk establishes that #1 is **numerically sound and CPU-feasible**; it does not yet
establish the throughput win (that needs the seam + the real net-side fusion).

## GO / NO-GO

**GO** for the fused-JAX matmul featurization, as a second `BatchPredict` impl behind the seam:

- Parity holds within tolerance (f32 worst 9.5e-7 ≪ 1e-4 bar); the legal-mask logic invariant stays
  bit-exact; f64 gives an exact provability fallback.
- CPU matmul timing is feasible (≤ 2.7 ms at B=64), and the GPU only improves the net side later.
- The reframe is denotationally exact (f64 == C++ double), so the only standing cost is the named 2×
  wire and the f32-vs-double provability downgrade on marg/p_pos.

NOT a green light to wire production: this is parity + feasibility, not the throughput A/B. The 2× wire
is a real cost that only the seam's head-to-head can justify.

## Open questions for the real build

1. **Wire protocol.** Ship the kW64-word rank bitset (1944 B) as-is, or a tighter encoding (sparse
   rank-list for narrowed beliefs; or ship the *delta* from the parent belief since the search narrows
   monotonically)? The fixed 1944 B is the worst case; a delta/sparse scheme could cut deep-search
   leaves substantially.
2. **Python inference extension.** Where does the matmul live relative to the net forward — a single
   fused JAX function `(belief_batch) → prediction` (featurize + net in one XLA program, the design's
   intent), or two stages? The fused form is what amortizes on GPU; it needs the world_feature_matrix
   resident as a device constant (969 KiB dense uint8 / or bit-packed) shared across the batch.
3. **Dense vs sparse matmul on GPU.** The dense `nworlds×(N+nD)` matmul is wasteful for narrow beliefs;
   evaluate a segment-sum / gather over live ranks vs the dense matmul once on-device.
4. **nworlds < 2²⁴ guard.** The bit-exact-`informative` argument is instance-contingent; the real build
   needs a guard (or a re-measure) for instances where nworlds exceeds the f32 exact-integer range.

Public Domain (The Unlicense).
