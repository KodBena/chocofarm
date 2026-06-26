// cpp/include/chocofarm/batch_predict.hpp
// Purpose: the BatchPredict seam (throughput lever #3, IN-PROCESS arm) — the boundary that featurizes a
//   BATCH of B parked leaves together, exploiting the locality the per-leaf path cannot: the producer's
//   cursor parks B leaves per net RTT (the same B batched to the net), so a batch of (loc, belief,
//   collected) is naturally available at the park point. This seam is the C++ twin of the design's
//   BatchPredict boundary; the IN-PROCESS impl (BatchFeaturizer) is the featurize-half (it produces the B
//   feature rows; the net forward is the caller's NetEvaluator, unchanged).
//
//   WHY a batch wins (de-risk-batched-featurization-2026-06-26.md, the productionized form): the belief
//   sweep — masked-AND + popcount over the env-static treasure/detector masks — is ~55% of producer
//   compute. The mask matrix ((N+nD)*kW64*8 ≈ 121.5 KiB on the live instance) is L2-resident. The per-leaf
//   path (FeatureBuilder, even with the AVX2 popcount primitive) re-streams that whole matrix B times; the
//   batched sweep holds each mask WORD resident across a 4-belief register tile, so the mask matrix streams
//   ~once instead of B times. The prototype measured ~+30% OVER the already-AVX2 per-leaf baseline for this
//   batch-specific tiling increment (the headline +124% mostly belongs to the no-seam primitive swap, which
//   helps the per-leaf path too — see the de-risk note's decomposition).
//
//   THE SHAPE (minimal, coherent with the eval seam net_evaluator.hpp / the cursor's eval_build_features):
//   a BatchFeaturizer holds a FeatureBuilder (the production featurizer — the SSOT for the §2.2 layout +
//   Phase-2 + geometry + collected assembly) and adds ONE in-process batched belief sweep (the
//   mask-resident AVX2 tile, productionizing the prototype's bat-avx2-tile Phase-1). It does NOT own the
//   net, the cursor, or the producer: it answers "B leaves -> B feature rows", the featurize-half of
//   BatchPredict. The full feature rows are BYTE-IDENTICAL to B calls of the production per-leaf
//   FeatureBuilder::build (the bit-identity gate proves it): the batched sweep stages the IDENTICAL integer
//   counts (popcount is order-independent), then FeatureBuilder::assemble_into runs the production Phase-2 +
//   assembly (NOT a fork — the ONE assembly body build_into shares, ADR-0012 P1). Float order preserved
//   (P6 behavioral bar, as the per-leaf path); the batched sweep is exact integer counts (no float reorder).
//
//   GATING/SCOPE: this is an ADDITIVE component + bench, NOT wired into the production cursor/producer (that
//   integration is a later maintainer-owned step). The batched sweep requires the env's BITSET arm (the
//   env-static treasure_mask/detector_mask): featurize_batch dispatches on env.use_bitset() — bitset arm =>
//   the tiled AVX2 sweep; otherwise => the per-leaf fallback (FeatureBuilder::build_into per leaf, still
//   byte-identical, no batch win). The AVX2 kernel is [[gnu::target("avx2")]]-attributed (host -march=native
//   gives AVX2 on this Skylake; the attribute keeps it explicit + portable to a non-native build).
//
// Public Domain (The Unlicense).
#pragma once

#include <cstddef>
#include <span>
#include <vector>

#include "chocofarm/collected_set.hpp"
#include "chocofarm/domains.hpp"  // FeatureDim — the typed feature-row length (P1, derived from the env)
#include "chocofarm/env.hpp"      // Environment + Belief (the seam value types)
#include "chocofarm/features.hpp" // FeatureBuilder (the §2.2 SSOT) + BeliefFeatures

namespace chocofarm {

// One parked leaf in a batch: the resolved standing point + the live belief + the collected set — exactly
// the (loc, bw, collected) triple eval_build_features / FeatureBuilder::build consume for ONE leaf. The
// belief is held by REFERENCE (the cursor owns the parked beliefs across the RTT; this view does not copy):
// the batch must outlive the featurize_batch call (the producer's natural lifetime — the leaves are parked).
struct BatchLeaf {
    Point loc;                       // the resolved standing Point (as in Loc::pt / env.coord)
    const Belief* bw = nullptr;      // the live belief at this leaf (non-owning; the cursor owns it)
    const CollectedSet* collected = nullptr;  // the collected set at this leaf (non-owning)
};

// The IN-PROCESS BatchPredict featurize-half: B parked leaves -> B feature rows, with the belief sweep run
// as ONE mask-resident batched pass (the lever-#3 locality) and the rest of the row built by the production
// FeatureBuilder (the SSOT). Holds a FeatureBuilder by value (its own — single-thread-owned, like every
// other FeatureBuilder consumer; the belief/loc memos are this featurizer's). NOT the net, NOT the cursor.
class BatchFeaturizer {
  public:
    explicit BatchFeaturizer(const Environment& env);

    [[nodiscard]] FeatureDim dim() const { return fb_.dim(); }  // the per-row length (derived; P8)

    // Featurize a batch of B leaves into `out_rows` (resized to B; each row resized to dim() then
    // overwritten). out_rows[b] is BYTE-IDENTICAL to fb_.build(leaves[b].loc, *leaves[b].bw,
    // *leaves[b].collected) — the gate proves it. The belief sweep is batched (the bitset arm) or falls
    // back per-leaf (non-bitset env); the assembly is always the production FeatureBuilder::assemble_into.
    // float64 rows (the build() dtype); the float32 narrowing is the caller's (as eval_build_features does).
    void featurize_batch(std::span<const BatchLeaf> leaves,
                         std::vector<std::vector<double>>& out_rows) const;

    // The batched belief sweep ALONE (the lever-#3 kernel under test): B beliefs -> B BeliefFeatures, the
    // mask-resident AVX2 tile over the env-static masks. EXPOSED for the bit-identity gate + the A/B bench
    // (the same role feature_compute.hpp's belief_features plays for the per-leaf sweep). out[b] is
    // byte-identical to belief_features(env, *beliefs[b]) (exact integer counts, the IDENTICAL Phase-2).
    // Requires env.use_bitset() (asserted): the batched arm IS the bitset masked-AND+popcount kernel.
    void belief_features_batch(std::span<const Belief* const> beliefs,
                               std::vector<BeliefFeatures>& out) const;

  private:
    const Environment& env_;
    FeatureBuilder fb_;  // the §2.2 SSOT (layout + Phase-2 + geometry/collected assembly); this featurizer's
};

}  // namespace chocofarm
