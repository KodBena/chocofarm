// cpp/include/chocofarm/world.hpp
// Purpose: the ONE home (ADR-0012 P1) for the world-mask type — the per-treasure bitmask that the env's
//   world-set, the bitset/ZDD arms, and collected_set.hpp's treasure-id domain all share. `World` is a
//   uint32_t, so the treasure-id domain it can carry is [0, 32) — the structural treasure-count limit the
//   whole codebase shares. Previously this fact lived in two+ places (env's bare std::vector<uint32_t>
//   world storage AND collected_set.hpp's local `using world_mask_t = std::uint32_t;`), a parallel
//   re-declaration that could silently drift; this header makes it single-sourced.
//
//   A pure leaf header (only <cstdint>) so it stays cycle-free — it is included BY env.hpp and
//   collected_set.hpp, exactly as belief_key.hpp is a leaf type header.
//
// Public Domain (The Unlicense).
#pragma once

#include <cstdint>

namespace chocofarm {

// A world = the per-treasure bitmask, bit t set <=> treasure t present; uint32_t => the treasure-id domain
// is [0, 32) — the structural treasure-count limit the whole codebase shares.
using World = std::uint32_t;

}  // namespace chocofarm
