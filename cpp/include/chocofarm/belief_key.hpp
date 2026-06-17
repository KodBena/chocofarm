// cpp/include/chocofarm/belief_key.hpp
// Purpose: the ONE belief-identity fingerprint TYPE (ADR-0012 P1) — a cheap, order-insensitive key for
//   an information-set world-set, shared by the Gumbel node transposition (gumbel.cpp) and the
//   FeatureBuilder belief memo (features.cpp). It is collision-RESISTANT, not collision-free (distinct
//   equal-size beliefs can share min/max world ids — Python documents this), so every consumer verifies
//   a hit by FULL belief-equality; the fingerprint is only a pre-filter. Mirrors Python's _belief_key
//   (chocofarm/solvers/ismcts.py). A leaf header (only <cstdint>/<tuple>) so both gumbel.hpp and
//   features.hpp include it with no include cycle.
//
//   The fingerprint FUNCTION moved to the env (Environment::belief_key(const Belief&), env.hpp) in
//   STEP 1 of the belief-rep cutover (docs/design/cpp-belief-rep-scoping.md §5, poke L2): the env owns
//   the read of the belief's representation, so no caller pokes `.worlds`. Only the TYPE stays here —
//   env.hpp includes THIS header for the BeliefKey return type, so the type cannot move into env.hpp
//   without an include cycle.
//
// Public Domain (The Unlicense).
#pragma once

#include <cstdint>
#include <tuple>

namespace chocofarm {

// (count, first-world-id, last-world-id) — the order-insensitive fingerprint of a belief world-set.
using BeliefKey = std::tuple<int, uint32_t, uint32_t>;

}  // namespace chocofarm
