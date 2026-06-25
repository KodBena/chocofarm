// cpp/include/chocofarm/world.hpp
// Purpose: the ONE home (ADR-0012 P1) for the world-mask type AND the belief world-count/world-rank
//   PHANTOM types — the integer vocabulary of "how many worlds are live in a belief" and "which world,
//   by 0-based rank". `World` is the per-treasure bitmask that the env's world-set, the bitset/ZDD arms,
//   and collected_set.hpp's treasure-id domain all share. `WorldCount` / `WorldRank` are the count and
//   index of those worlds. Previously each fact lived in two+ places (env's bare std::vector<uint32_t>
//   world storage AND collected_set.hpp's local `using world_mask_t`; and a world-COUNT spelled four
//   inconsistent ways — `int count_`, `int64_t cached_count_`, `size_t nb`, `int popcount` — with a
//   static_cast at every seam, the ad-hoc coercion ADR-0008/0012 forbid). This header makes each
//   single-sourced (ADR-0000: the count/rank category error is made unrepresentable, not guarded).
//
//   A pure leaf header (only <bit>/<cstdint> + the Quantity machinery) so it stays cycle-free — it is
//   included BY env.hpp and collected_set.hpp, exactly as belief_key.hpp is a leaf type header. The
//   zero-cost strong-type MACHINERY (Quantity<Tag, Rep>) is single-sourced in quantity.hpp (Band-1,
//   reusable); this header only INSTANTIATES it for the world domain (Band-3, FFXIII-bound).
//
// Public Domain (The Unlicense).
#pragma once

#include <bit>
#include <cstdint>

#include "chocofarm/quantity.hpp"  // Quantity<Tag, Rep> — the zero-cost phantom-type SSOT (P1)

namespace chocofarm {

// A world = the per-treasure bitmask, bit t set <=> treasure t present; uint32_t => the treasure-id domain
// is [0, 32) — the structural treasure-count limit the whole codebase shares.
using World = std::uint32_t;

// ---- the belief world-count / world-rank PHANTOM types (ADR-0000 / ADR-0012 P1/P6/P8) ----
//
// MOTIVATION OF SIGN + WIDTH (ADR-0000 rule 1: width/sign are motivated at the declaration, not "int by
// default"). A belief world-COUNT and a world-RANK are both:
//   * NON-NEGATIVE — a count cannot be < 0, a rank is a 0-based index ⇒ UNSIGNED (never the
//     arbitrarily-signed `int`/`int64_t` the four old spellings mixed).
//   * BOUNDED by |worlds| = C(N,K), the enumerated world-set the bitset/flat arms materialize IN MEMORY.
//     The env already enumerates the full world array (env.cpp build_worlds) and the legacy `int count_`
//     (env.hpp) already assumed |worlds| < 2^31; the bitset arm packs one bit per world into RAM, so a
//     count > 2^32 would need 512 MiB of belief bitset alone — structurally absent here. 32 bits
//     therefore COVERS the domain with headroom (max |worlds| at the live instance is 15504), while
//     std::uint32_t is HALF the width of a reflexive size_t/int64_t — chosen for the count that fits 32
//     bits, not reached for by habit (ADR-0000 mandate item 2). The static_assert below pins the width.
//
// PHANTOM = a zero-cost strong type (the Quantity<Tag, Rep> machinery, quantity.hpp): a one-field struct
// that does NOT implicitly convert to/from the raw integer nor to the SIBLING type, so an illegal mix
// (count used as a rank, a raw int passed where a count is owed, a width/sign mismatch) is UNREPRESENTABLE
// at compile time (ADR-0000: make the illegal state unrepresentable; ADR-0012 P8: the typed signature is
// the SSOT). The two distinct tags below mint two distinct, non-interconvertible types. Zero-cost is proven
// statically (quantity.hpp's elision asserts) + empirically (the objdump A/B oracle).
//
// The underlying representation the two phantom types share. One home (P1) for the width decision.
using WorldCountRep = std::uint32_t;
static_assert(sizeof(WorldCountRep) == 4,
              "WorldCount/WorldRank width is motivated as 32-bit: |worlds|=C(N,K) materializable in RAM "
              "(legacy int count_ already assumed <2^31); a wider type would be unmotivated (ADR-0000).");

// A count of live worlds in a belief (the nb / count_ / cached_count_ / popcount domain).
struct WorldCountTag {};
using WorldCount = Quantity<WorldCountTag, WorldCountRep>;

// A 0-based world rank/index (the r-th set bit, world_at_rank's argument, rth_set_bit_index's domain).
struct WorldRankTag {};
using WorldRank = Quantity<WorldRankTag, WorldCountRep>;

// ARITHMETIC OPT-INS (quantity.hpp's concept-gated traits). A COUNT is additive — a partial popcount
// accumulates into a count, a count+count is a count (the ONLY count arithmetic). A RANK is affine — a
// rank plus a raw word/bit offset is still a rank, and the gap between two ranks is a raw count (the
// rth_set_bit scan's `w*64 + tzcnt` index arithmetic). A count+rank is UNREPRESENTABLE (distinct tags do
// not interoperate); the one MOTIVATED count->rank crossing is the named last_rank() bridge below.
template <> struct quantity_additive<WorldCountTag> : std::true_type {};
template <> struct quantity_affine<WorldRankTag> : std::true_type {};

// The ONE motivated count→rank bridge: the LAST rank of a non-empty belief is count-1 (the belief_key /
// sample-range upper bound). A named crossing (count domain → rank domain) so the only count→rank step is
// visible and centralized, never an ad-hoc int subtraction at a call site (ADR-0000 item 5).
[[nodiscard]] constexpr WorldRank last_rank(WorldCount n) noexcept {
    return WorldRank{static_cast<WorldCountRep>(n.value())} - WorldCountRep{1};  // affine rank - raw offset
}

// popcount of a 64-bit belief word as a typed partial count (the stdlib boundary: std::popcount returns a
// signed int, non-negative by construction; this is the ONE place that signed→unsigned-count crossing is
// done, centralized as an ACL so popcount_all/popcount_and never re-do it).
[[nodiscard]] constexpr WorldCount popcount_word(std::uint64_t w) noexcept {
    return WorldCount{static_cast<WorldCountRep>(std::popcount(w))};  // popcount ∈ [0,64], always fits
}

}  // namespace chocofarm
