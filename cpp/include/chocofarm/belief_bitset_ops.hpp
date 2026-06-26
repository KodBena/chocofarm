// cpp/include/chocofarm/belief_bitset_ops.hpp
// Purpose: the BITSET-arm belief kernels — the masked-AND + popcount over kW64 words that are the §A.4
//   belief-sweep's bitset twin (docs/design/cpp-belief-rep-scoping.md §5; the decision-reversal note's
//   masked-AND/popcount/r-th-set-bit kernels). ONE home (ADR-0012 P1: derive-don't-duplicate): the env
//   seam bodies (env.cpp — marginals / informative / the filters / world_at_rank) AND the belief_features
//   bitset arm (features.cpp) both include this, so the masked-popcount kernel is defined ONCE rather than
//   copy-pasted per translation unit. Pure, env-free, header-inline (trivial bodies — `inline` so each TU
//   gets the same definition with no ODR clash). They operate on a kW64-word span; the flat arm is the
//   REFERENCE and these MUST produce byte-identical integer counts (the A/B oracle nets that).
//
//   TYPING (ADR-0000 / ADR-0012 P8): the kernels speak the PHANTOM count/rank + word domains (world.hpp /
//   domains.hpp), not raw
//   ints — popcount_all/popcount_and return a WorldCount, rth_set_bit_index takes + returns a WorldRank.
//   The signed stdlib std::popcount/std::countr_zero are unwrapped through the ONE ACL (popcount_word /
//   the countr_zero crossing here), so no arbitrarily-signed int stands in for a count/rank. The ZERO-COST
//   ACL for the auto-vectorizer: the inner sweep iterates the RAW std::span<const uint64_t> and accumulates
//   a raw rep (the compiler's AVX popcount/AND loop sees plain integers); the result is re-wrapped into the
//   domain type ONCE at the return boundary (ADR-0000 item 4 — expose the raw span to the vectorizer
//   inside, re-wrap at the boundary). The kW64 WORD loop is itself typed: the span's word stride is a
//   WordCount (the .size() ACL), the scan's word position a WordIndex (domains.hpp) — distinct from the
//   WorldRank/WorldCount the bits SELECT (words != worlds); the w*64 step is the named WordIndex -> world-
//   rank-base bridge, with the raw-rep accumulator kept inside the sweep for the auto-vectorizer.
//
// Public Domain (The Unlicense).
#pragma once

#include <bit>
#include <cstddef>
#include <cstdint>
#include <immintrin.h>  // AVX2 vpshufb popcount (the de-risk +74% lever; build is -march=native => AVX2)
#include <optional>
#include <span>

#include "chocofarm/domains.hpp"  // WorldCount/WorldRank (via world.hpp) + WordCount/WordIndex (the word-loop domain, P1)

namespace chocofarm {

// ---- AVX2 vpshufb popcount kernel (the de-risk's +74% lever, throughput-derisk-verdict-2026-06-26): the
// belief sweep is popcount-throughput-bound with masks L2-resident, and scalar POPCNT is port-1-bound at
// 1 word/cyc while vpshufb does 4 words/instr across ports. Build is -march=native (AVX2); the
// [[gnu::target("avx2")]] makes the codegen explicit + inlinable into the (also-native) callers. popcount is
// EXACT + order-independent, so these return the SAME integer count as the old scalar loop — bit-identical
// belief_features / count_ (the belief-sweep oracle nets it byte-for-byte). ----
namespace detail {
// nibble-LUT popcount of 4x uint64 lanes, horizontally summed per 64-bit lane (sad_epu8).
[[gnu::target("avx2")]] inline __m256i popcnt256(__m256i v) {
    const __m256i lut = _mm256_setr_epi8(
        0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4, 0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4);
    const __m256i lo_mask = _mm256_set1_epi8(0x0f);
    const __m256i lo = _mm256_and_si256(v, lo_mask);
    const __m256i hi = _mm256_and_si256(_mm256_srli_epi16(v, 4), lo_mask);
    const __m256i pc = _mm256_add_epi8(_mm256_shuffle_epi8(lut, lo), _mm256_shuffle_epi8(lut, hi));
    return _mm256_sad_epu8(pc, _mm256_setzero_si256());
}
[[gnu::target("avx2")]] inline std::uint64_t hsum256(__m256i acc) {
    std::uint64_t t[4]; _mm256_storeu_si256(reinterpret_cast<__m256i*>(t), acc);
    return t[0] + t[1] + t[2] + t[3];
}
// popcount(b & m) over W words: 4 words/iter (AVX2) + scalar tail.
[[gnu::target("avx2")]] inline std::uint64_t pc_and_avx2(const std::uint64_t* b, const std::uint64_t* m, std::size_t W) {
    __m256i acc = _mm256_setzero_si256();
    std::size_t w = 0;
    for (; w + 4 <= W; w += 4)
        acc = _mm256_add_epi64(acc, popcnt256(_mm256_and_si256(
            _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + w)),
            _mm256_loadu_si256(reinterpret_cast<const __m256i*>(m + w)))));
    std::uint64_t s = hsum256(acc);
    for (; w < W; ++w) s += static_cast<std::uint64_t>(std::popcount(b[w] & m[w]));
    return s;
}
// popcount(b) over W words (no mask) — the count_ recompute.
[[gnu::target("avx2")]] inline std::uint64_t pc_all_avx2(const std::uint64_t* b, std::size_t W) {
    __m256i acc = _mm256_setzero_si256();
    std::size_t w = 0;
    for (; w + 4 <= W; w += 4)
        acc = _mm256_add_epi64(acc, popcnt256(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + w))));
    std::uint64_t s = hsum256(acc);
    for (; w < W; ++w) s += static_cast<std::uint64_t>(std::popcount(b[w]));
    return s;
}
}  // namespace detail

// popcount over the whole belief — the cached count_'s definition (recompute count_ after a filter).
[[nodiscard]] inline WorldCount popcount_all(std::span<const uint64_t> bits) {
    return WorldCount{static_cast<WorldCountRep>(detail::pc_all_avx2(bits.data(), bits.size()))};
}

// popcount(belief & mask): the masked overlap count — marginals[t]=Σ_w bit_t(w), the detector cover cnt[j],
// the informative split test. AVX2 vpshufb kernel (the de-risk +74% lever); bits/mask are exactly kW64
// words by construction; bit-identical to the prior scalar loop (popcount exact + order-independent).
[[nodiscard]] inline WorldCount popcount_and(std::span<const uint64_t> bits, std::span<const uint64_t> mask) {
    return WorldCount{static_cast<WorldCountRep>(detail::pc_and_avx2(bits.data(), mask.data(), bits.size()))};
}

// The r-th set bit's GLOBAL index (0-based world RANK): scan words, early-exit on the containing word,
// clear the r low set bits, tzcnt the rest. Portable (BMI2 pdep/tzcnt would accelerate but is not
// load-bearing for correctness). The CALLER guarantees r < count_ (so the scan always hits); a miss is an
// invariant violation the caller aborts on (ADR-0002 / scoping §6 risk 7). The miss is now a TYPED ABSENCE
// — std::nullopt, not the former `-1` magic sentinel (the untyped-optional ADR-0012 P9 forbids: an
// unsigned rank cannot even carry -1, so the type now makes the sentinel unrepresentable and forces the
// caller to handle the absence). `r`/the returned index are both WorldRank: passing a count where a rank
// is owed (or vice versa) does not compile.
[[nodiscard]] inline std::optional<WorldRank> rth_set_bit_index(std::span<const uint64_t> bits, WorldRank r) {
    WorldCountRep rem = r.value();  // unwrap the rank ONCE for the scan arithmetic (the rep is the index)
    // The word scan indexes the belief in the WordIndex domain (bounded by the span's WordCount); the
    // containing word's index is the typed WordIndex `w`, bridged to the world-rank base via w*64 (the
    // affine WorldRank + raw bit offset — the kW64 word-loop boundary, ADR-0000 item 5).
    const WordCount nwords{static_cast<WordRep>(bits.size())};
    for (WordIndex w{0}; w.value() < nwords.value(); w = w + WordRep{1}) {
        uint64_t x = bits[w.value()];
        const WorldCountRep pc = static_cast<WorldCountRep>(std::popcount(x));
        if (rem < pc) {
            for (WorldCountRep s = 0; s < rem; ++s) x &= x - 1;  // clear the r lowest set bits
            return WorldRank{static_cast<WorldCountRep>(w.value()) * 64u +
                             static_cast<WorldCountRep>(std::countr_zero(x))};
        }
        rem -= pc;
    }
    return std::nullopt;  // invariant: r < count_ guarantees a hit; nullopt => count_ desynced from bits
}

}  // namespace chocofarm
