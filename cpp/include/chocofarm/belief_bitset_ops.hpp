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
#include <cstdint>
#include <optional>
#include <span>

#include "chocofarm/domains.hpp"  // WorldCount/WorldRank (via world.hpp) + WordCount/WordIndex (the word-loop domain, P1)

namespace chocofarm {

// popcount over the whole belief — the cached count_'s definition (recompute count_ after a filter).
// Returns a typed WorldCount. ZERO-COST ACL: the hot loop accumulates a RAW WorldCountRep over the raw
// span (so the AVX popcount auto-vectorizer sees a plain-integer reduction), re-wrapped ONCE at return.
[[nodiscard]] inline WorldCount popcount_all(std::span<const uint64_t> bits) {
    WorldCountRep s = 0;
    for (uint64_t w : bits) s += static_cast<WorldCountRep>(std::popcount(w));
    return WorldCount{s};
}

// popcount(belief & mask): the masked overlap count — the bitset twin of a per-world predicate sum
// (marginals[t] = Σ_w bit_t(w), the detector cover cnt[j], the informative split test). Words stride
// together. Same zero-cost ACL: raw-rep accumulation over the raw spans, re-wrap at the boundary.
[[nodiscard]] inline WorldCount popcount_and(std::span<const uint64_t> bits, std::span<const uint64_t> mask) {
    // The shared word stride as the typed WordCount (the .size() ACL: a container size_t -> a word count;
    // bits/mask are exactly kW64 words by construction). The inner index + accumulator stay RAW inside the
    // sweep (the documented zero-cost vectorizer ACL above — the AVX popcount/AND loop sees plain integers),
    // re-wrapped into WorldCount ONCE at the return boundary.
    const WordCount nwords{static_cast<WordRep>(bits.size())};
    WorldCountRep s = 0;
    for (WordRep w = 0; w < nwords.value(); ++w)
        s += static_cast<WorldCountRep>(std::popcount(bits[w] & mask[w]));
    return WorldCount{s};
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
