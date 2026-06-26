// cpp/src/batch_predict.cpp
// Purpose: the IN-PROCESS BatchPredict featurize-half (batch_predict.hpp) — productionizes the de-risk
//   prototype's bat-avx2-tile Phase-1 (docs/notes/derisk-batched-featurization-2026-06-26.md) as a real
//   component on the production featurizer. featurize_batch(B leaves) -> B feature rows, each BYTE-IDENTICAL
//   to FeatureBuilder::build for that leaf.
//
//   THE KERNEL (belief_features_batch): the env-static treasure/detector masks are swept MASK-MAJOR over the
//   B beliefs — the outer loop walks the N+nD mask rows, the inner loop walks the B beliefs in 4-belief
//   tiles. Each mask WORD is loaded once and AND-popcounted against all 4 beliefs of the tile while it is
//   hot in a register (the locality lever-#3 names). The per-(mask,belief) popcount is the AVX2 vpshufb
//   nibble-LUT count (4 words/instruction, multi-port) — the SAME integer count the production
//   popcount_and produces (popcount is order-independent), so the staged counts are bit-identical to the
//   per-leaf belief_features_bitset arm. Phase 2 (the `* inv` maps + informative/sharpness/nonempty) is the
//   IDENTICAL pointwise map the production arm runs.
//
//   BYTE-IDENTITY: the staged integer counts equal the per-leaf bitset arm's (exact, order-independent);
//   Phase 2 is the same `* inv` over the same counts + the same nb; the full row is then assembled by the
//   production FeatureBuilder::assemble_into (the ONE assembly body build_into shares — geometry + collected
//   + named-block writes in the unchanged float order, P6). So out_rows[b] == fb.build(leaf b) byte-for-byte.
//
//   DISPATCH: featurize_batch / belief_features_batch require the env's BITSET arm for the batched kernel
//   (the env-static masks live only when use_bitset()). A non-bitset env, or a per-leaf belief that is not a
//   non-empty BitsetBelief (an empty belief / a flat/ZDD arm), falls back to the per-leaf belief_features /
//   build_into for THAT leaf — still byte-identical, just no batch win on it. The live instance gates ON.
//
// Public Domain (The Unlicense).
#include "chocofarm/batch_predict.hpp"

#include <bit>
#include <cassert>
#include <cstdint>
#include <variant>

#include <immintrin.h>  // AVX2 vpshufb popcount (the mask-resident tile primitive; -march=native => AVX2)

#include "chocofarm/feature_compute.hpp"  // belief_features (the per-leaf reference / the fallback sweep)

namespace chocofarm {

namespace {

// AVX2 nibble-LUT popcount of 4×uint64 lanes, horizontal-summed per 64-bit lane (sad_epu8). Identical to
// the prototype's popcnt256 (de-risk note). Returns the per-lane partial counts in 4 uint64 lanes.
[[gnu::target("avx2")]] inline __m256i popcnt256(__m256i v) {
    const __m256i lut = _mm256_setr_epi8(
        0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4, 0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4);
    const __m256i lo_mask = _mm256_set1_epi8(0x0f);
    const __m256i lo = _mm256_and_si256(v, lo_mask);
    const __m256i hi = _mm256_and_si256(_mm256_srli_epi16(v, 4), lo_mask);
    const __m256i pc = _mm256_add_epi8(_mm256_shuffle_epi8(lut, lo), _mm256_shuffle_epi8(lut, hi));
    return _mm256_sad_epu8(pc, _mm256_setzero_si256());  // 4 partial counts, one per 64-bit lane
}

// AVX2 popcount(b & m) over W words, 4 words/iter; scalar tail. The SAME integer count as popcount_and
// (popcount is order-independent) — the staged counts stay bit-identical to the per-leaf bitset arm.
[[gnu::target("avx2")]] inline uint64_t pc_and_avx2(const uint64_t* b, const uint64_t* m, size_t W) {
    __m256i acc = _mm256_setzero_si256();
    size_t w = 0;
    for (; w + 4 <= W; w += 4)
        acc = _mm256_add_epi64(acc, popcnt256(_mm256_and_si256(
            _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + w)),
            _mm256_loadu_si256(reinterpret_cast<const __m256i*>(m + w)))));
    alignas(32) uint64_t tmp[4];
    _mm256_store_si256(reinterpret_cast<__m256i*>(tmp), acc);
    uint64_t s = tmp[0] + tmp[1] + tmp[2] + tmp[3];
    for (; w < W; ++w) s += static_cast<uint64_t>(std::popcount(b[w] & m[w]));
    return s;
}

// Stage ONE mask column's counts across the B beliefs into dst[b*stride + col], mask-resident over 4-belief
// tiles (the prototype's Avx2Tiled kernel): the mask word is held in `mv` and reused across the 4 beliefs of
// the tile before the next mask word is loaded. The < 4-belief tail (and a non-multiple-of-4 B) runs the
// per-belief pc_and_avx2. All paths produce the SAME integer count (order-independent popcount).
[[gnu::target("avx2")]] inline void stage_column_tiled(
        const uint64_t* mask, size_t W, const std::vector<const uint64_t*>& belptr, size_t B,
        size_t col, size_t stride, std::vector<WorldCountRep>& dst) {
    size_t b = 0;
    for (; b + 4 <= B; b += 4) {
        const uint64_t* b0 = belptr[b];     const uint64_t* b1 = belptr[b + 1];
        const uint64_t* b2 = belptr[b + 2]; const uint64_t* b3 = belptr[b + 3];
        __m256i a0 = _mm256_setzero_si256(), a1 = a0, a2 = a0, a3 = a0;
        size_t w = 0;
        for (; w + 4 <= W; w += 4) {
            const __m256i mv = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(mask + w));
            a0 = _mm256_add_epi64(a0, popcnt256(_mm256_and_si256(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(b0 + w)), mv)));
            a1 = _mm256_add_epi64(a1, popcnt256(_mm256_and_si256(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(b1 + w)), mv)));
            a2 = _mm256_add_epi64(a2, popcnt256(_mm256_and_si256(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(b2 + w)), mv)));
            a3 = _mm256_add_epi64(a3, popcnt256(_mm256_and_si256(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(b3 + w)), mv)));
        }
        const uint64_t* bs[4] = {b0, b1, b2, b3};
        const __m256i accs[4] = {a0, a1, a2, a3};
        for (int kk = 0; kk < 4; ++kk) {
            alignas(32) uint64_t tmp[4];
            _mm256_store_si256(reinterpret_cast<__m256i*>(tmp), accs[kk]);
            uint64_t s = tmp[0] + tmp[1] + tmp[2] + tmp[3];
            for (size_t ww = w; ww < W; ++ww) s += static_cast<uint64_t>(std::popcount(bs[kk][ww] & mask[ww]));
            dst[(b + static_cast<size_t>(kk)) * stride + col] = static_cast<WorldCountRep>(s);
        }
    }
    for (; b < B; ++b)
        dst[b * stride + col] = static_cast<WorldCountRep>(pc_and_avx2(belptr[b], mask, W));
}

// Phase 2 for ONE belief from the staged B-major counts: the IDENTICAL `* inv` maps the production bitset
// arm (belief_features_bitset) runs — marg/marg_sum in treasure-id order (P6), p_pos via `* inv`,
// informative via (0 < cnt < nb), sharpness, nonempty. Byte-identical to the per-leaf arm by construction.
inline BeliefFeatures phase2_one(const std::vector<WorldCountRep>& cnt_marg,
                                 const std::vector<WorldCountRep>& cnt_det, size_t b, size_t Nn, size_t nDn,
                                 WorldCountRep nb, double log_nworlds) {
    BeliefFeatures bf;
    bf.marg.assign(Nn, 0.0);
    bf.p_pos.assign(nDn, 0.0);
    bf.informative.assign(nDn, 0.0);
    const double inv = 1.0 / static_cast<double>(nb);
    for (size_t t = 0; t < Nn; ++t) {
        bf.marg[t]   = static_cast<double>(cnt_marg[b * Nn + t]) * inv;
        bf.marg_sum += bf.marg[t];  // treasure-id order (P6) — matches the production arm exactly
    }
    for (size_t j = 0; j < nDn; ++j) {
        const WorldCountRep dc = cnt_det[b * nDn + j];
        bf.p_pos[j]       = static_cast<double>(dc) * inv;
        bf.informative[j] = (dc > 0 && dc < nb) ? 1.0 : 0.0;
    }
    bf.sharpness = std::log(static_cast<double>(nb)) / log_nworlds;
    bf.nonempty  = 1.0;
    return bf;
}

}  // namespace

BatchFeaturizer::BatchFeaturizer(const Environment& env) : env_(env), fb_(env) {}

void BatchFeaturizer::belief_features_batch(std::span<const Belief* const> beliefs,
                                            std::vector<BeliefFeatures>& out) const {
    // The batched arm IS the bitset masked-AND+popcount kernel (the env-static masks). A non-bitset env has
    // no such masks: that is a programmer-bug call (the dispatcher in featurize_batch guards it), so fail
    // loud (ADR-0002) rather than silently produce wrong rows.
    assert(env_.use_bitset() && "belief_features_batch: env gates OFF the bitset arm (no masks to sweep)");
    using TR = TreasureRep;
    using GR = GeometryIdRep;
    const TR N = static_cast<TR>(env_.N());
    const GR nD = static_cast<GR>(env_.n_detectors());
    const size_t Nn = static_cast<size_t>(N);
    const size_t nDn = static_cast<size_t>(nD);
    const size_t B = beliefs.size();
    const size_t W = static_cast<size_t>(env_.kW64());
    const double log_nworlds = std::log(static_cast<double>(env_.worlds().size()));
    out.assign(B, BeliefFeatures{});
    if (B == 0) return;

    // Partition the batch: the BATCHABLE leaves (a non-empty BitsetBelief — the kernel's domain) go through
    // the mask-resident tiled sweep; any OTHER arm (empty belief / flat / ZDD) falls back to the per-leaf
    // belief_features for that index (still byte-identical). batch_idx maps a tile position -> the leaf index.
    std::vector<const uint64_t*> belptr;     belptr.reserve(B);
    std::vector<WorldCountRep>   nb_of;      nb_of.reserve(B);
    std::vector<size_t>          batch_idx;  batch_idx.reserve(B);
    for (size_t b = 0; b < B; ++b) {
        const Belief& blf = *beliefs[b];
        if (const auto* bs = std::get_if<BitsetBelief>(&blf); bs && bs->count_ != WorldCount{0}) {
            belptr.push_back(bs->live().data());
            nb_of.push_back(bs->count_.value());
            batch_idx.push_back(b);
        } else {
            out[b] = belief_features(env_, blf);  // empty/flat/ZDD: per-leaf, byte-identical
        }
    }

    const size_t Bb = belptr.size();
    if (Bb == 0) return;  // nothing batchable (all leaves fell back)

    // B-major count scratch over the BATCHABLE leaves (column = treasure/detector, row = tile position).
    std::vector<WorldCountRep> cnt_marg(Bb * Nn, 0), cnt_det(Bb * nDn, 0);

    // Phase 1, MASK-MAJOR + 4-belief tiled: each mask row read once, AND-popcounted against all Bb beliefs.
    for (size_t t = 0; t < Nn; ++t)
        stage_column_tiled(env_.treasure_mask(static_cast<int>(t)).data(), W, belptr, Bb, t, Nn, cnt_marg);
    for (size_t j = 0; j < nDn; ++j)
        stage_column_tiled(env_.detector_mask(static_cast<int>(j)).data(), W, belptr, Bb, j, nDn, cnt_det);

    // Phase 2 per batched leaf -> write back to its original index.
    for (size_t i = 0; i < Bb; ++i)
        out[batch_idx[i]] = phase2_one(cnt_marg, cnt_det, i, Nn, nDn, nb_of[i], log_nworlds);
}

void BatchFeaturizer::featurize_batch(std::span<const BatchLeaf> leaves,
                                      std::vector<std::vector<double>>& out_rows) const {
    const size_t B = leaves.size();
    out_rows.resize(B);

    // Non-bitset env: no batched kernel (no env-static masks) — fall back to the per-leaf production
    // build_into for every leaf (byte-identical, no batch win). The live instance gates ON, so this is the
    // safety arm, not the live path.
    if (!env_.use_bitset()) {
        for (size_t b = 0; b < B; ++b)
            fb_.build_into(leaves[b].loc, *leaves[b].bw, *leaves[b].collected, out_rows[b]);
        return;
    }

    // Bitset env: ONE batched belief sweep over the B beliefs, then the production assembly per leaf.
    std::vector<const Belief*> bptr(B);
    for (size_t b = 0; b < B; ++b) bptr[b] = leaves[b].bw;
    std::vector<BeliefFeatures> bf;
    belief_features_batch(std::span<const Belief* const>(bptr.data(), B), bf);

    // Assemble each row from the batch-computed BeliefFeatures + the per-loc geometry + the collected set —
    // the production FeatureBuilder::assemble_into (the SAME body build_into runs), so the full row is
    // byte-identical to fb_.build(leaf b). No belief recompute here (the batch already did it).
    for (size_t b = 0; b < B; ++b)
        fb_.assemble_into(leaves[b].loc, bf[b], *leaves[b].collected, out_rows[b]);
}

}  // namespace chocofarm
