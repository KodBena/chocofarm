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

#include <cstdint>
#include <span>

#include "chocofarm/env.hpp"
#include "chocofarm/features.hpp"   // BeliefFeatures (the returned value type)

namespace chocofarm {

// The belief-derived intermediates for `bw`: mean-bit marg over treasures + per-detector cover counts ->
// p_pos / informative, plus marg_sum / sharpness / nonempty. Caller supplies the env-derived dims:
// N = env.N(), nD = env.n_detectors(), log_nworlds = log(|env.worlds()|). Pure; the single home is
// features.cpp. (geometry_features / collected_features stay file-local — only the sweep is benched.)
[[nodiscard]] BeliefFeatures belief_features(const Environment& env, std::span<const uint32_t> bw,
                                             int N, int nD, double log_nworlds);

}  // namespace chocofarm
