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
// Public Domain (The Unlicense).
#pragma once

#include <bit>
#include <cstdint>
#include <span>

namespace chocofarm {

// popcount over the whole belief — the cached count_'s definition (recompute count_ after a filter).
[[nodiscard]] inline int popcount_all(std::span<const uint64_t> bits) {
    int s = 0;
    for (uint64_t w : bits) s += std::popcount(w);
    return s;
}

// popcount(belief & mask): the masked overlap count — the bitset twin of a per-world predicate sum
// (marginals[t] = Σ_w bit_t(w), the detector cover cnt[j], the informative split test). Words stride together.
[[nodiscard]] inline int popcount_and(std::span<const uint64_t> bits, std::span<const uint64_t> mask) {
    int s = 0;
    for (int w = 0; w < static_cast<int>(bits.size()); ++w)
        s += std::popcount(bits[static_cast<size_t>(w)] & mask[static_cast<size_t>(w)]);
    return s;
}

// The r-th set bit's GLOBAL index (0-based world rank): scan words, early-exit on the containing word, clear
// the r low set bits, tzcnt the rest. Portable (BMI2 pdep/tzcnt would accelerate but is not load-bearing for
// correctness). The CALLER guarantees 0 <= r < count_ (so the scan always hits); a miss is an invariant
// violation the caller aborts on (ADR-0002 / scoping §6 risk 7) — this kernel returns -1 on a miss so the
// caller can fail loudly with its own context (it never legitimately returns -1).
[[nodiscard]] inline int rth_set_bit_index(std::span<const uint64_t> bits, int r) {
    for (int w = 0; w < static_cast<int>(bits.size()); ++w) {
        uint64_t x = bits[static_cast<size_t>(w)];
        const int pc = std::popcount(x);
        if (r < pc) {
            for (int s = 0; s < r; ++s) x &= x - 1;  // clear the r lowest set bits
            return w * 64 + std::countr_zero(x);
        }
        r -= pc;
    }
    return -1;  // invariant: 0 <= r < count_ guarantees a hit; -1 => the caller's count_ desynced from bits
}

}  // namespace chocofarm
