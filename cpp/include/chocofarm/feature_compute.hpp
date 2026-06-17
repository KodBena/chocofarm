// cpp/include/chocofarm/feature_compute.hpp
// Purpose: expose the PURE belief-sweep compute (belief_features) — the functional core of the §2.2
//   featurization (features.cpp), the O(nb·(N+nD)) popcount-marg + observe-cover loop the K=16 profile
//   measured at ~81% of the single-thread cost. It is declared HERE (not left file-local) for ONE
//   reason: so the isolated microbenchmark (belief_sweep_bench.cpp) and future unit tests can drive the
//   REAL function directly — without the search / wire / cache / geometry / assemble around it — and an
//   optimizer (pospopcount / inline-masks / decision-diagram) iterates against a clean signal. This is
//   the bounded "feature_compute_testonly" exposure the cache-design consult sanctioned: it is a pure
//   value-function (ADR-0012 P9 — no I/O, no state, total over its bounds-carrying inputs), NOT a
//   behavioral wire contract. Its single definition stays in features.cpp.
//
// Public Domain (The Unlicense).
#pragma once

#include "chocofarm/env.hpp"        // Environment + Belief (the seam value type)
#include "chocofarm/features.hpp"   // BeliefFeatures (the returned value type)

namespace chocofarm {

// The belief-derived intermediates for `bw`: mean-bit marg over treasures + per-detector cover counts ->
// p_pos / informative, plus marg_sum / sharpness / nonempty. The signature names its TRUE inputs
// (ADR-0012 P9 honest signature): the env (which carries the per-detector cover bitmasks env.face_masks()
// for the FLAT arm AND the env-static bitset masks env.treasure_mask/detector_mask for the BITSET arm, plus
// the dims N/nD and log|worlds|) and the belief `bw` (the seam value type). STEP 2 (the bitset arm,
// docs/design/cpp-belief-rep-scoping.md §5 L3) makes this VISIT the variant: the flat arm runs the EXISTING
// §A.4 sweep over `.worlds` UNCHANGED; the bitset arm runs the masked-AND + popcount kernel (env-static
// masks), producing the SAME integer bit_cnt/det_cnt then the IDENTICAL Phase-2 `* inv` — byte-identical
// to the flat sweep (§6 risk 6). The signature took `(const Belief&, span<masks>, N, nD, log_nworlds)` in
// Step 1; Step 2 folds those into `const Environment&` because the bitset arm needs the env's bitset masks
// (the span-only signature could not reach them) — every caller already holds an Environment. Pure; the
// single home is features.cpp. (geometry_features / collected_features stay file-local — only the sweep,
// the K=16 profile's ~81%, is exposed for the bench + the bit-exact oracle.)
[[nodiscard]] BeliefFeatures belief_features(const Environment& env, const Belief& bw);

}  // namespace chocofarm
