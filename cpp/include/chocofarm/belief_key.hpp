// cpp/include/chocofarm/belief_key.hpp
// Purpose: the ONE belief-identity fingerprint (ADR-0012 P1) — a cheap, order-insensitive key for an
//   information-set world-set, shared by the Gumbel node transposition (gumbel.cpp) and the
//   FeatureBuilder belief memo (features.cpp). It is collision-RESISTANT, not collision-free (distinct
//   equal-size beliefs can share min/max world ids — Python documents this), so every consumer verifies
//   a hit by FULL bw-equality; the fingerprint is only a pre-filter. Mirrors Python's _belief_key
//   (chocofarm/solvers/ismcts.py). A leaf header (only <cstdint>/<tuple>/<vector>) so both gumbel.hpp
//   and features.hpp include it with no include cycle.
//
// Public Domain (The Unlicense).
#pragma once

#include <cstdint>
#include <tuple>
#include <vector>

namespace chocofarm {

// (count, first-world-id, last-world-id) — the order-insensitive fingerprint of a belief world-set.
using BeliefKey = std::tuple<int, uint32_t, uint32_t>;

[[nodiscard]] inline BeliefKey belief_key(const std::vector<uint32_t>& bw) {
    if (bw.empty()) return BeliefKey{0, 0u, 0u};
    return BeliefKey{static_cast<int>(bw.size()), bw.front(), bw.back()};
}

}  // namespace chocofarm
