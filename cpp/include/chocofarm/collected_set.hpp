// cpp/include/chocofarm/collected_set.hpp
// Purpose: the typed fixed-width bitmask of treasure ids already collected (ADR-0012 P9 — an HONEST
//   signature: a named struct, NOT a bare uint64_t that would be confusable with the env's
//   treasure_mask/detector_mask which range over a DIFFERENT domain, the worlds-by-rank bitvector).
//   It replaces the former `std::set<int> collected` threaded through every Policy/search/env contract:
//   the set node-allocated per element and was COPIED per descent step (collected ∪ {slot} materialized
//   on every treasure-collecting transition, scaling max_depth×n_sims), which the K=512 bitset-arm
//   profile flagged as the #1 self-time client (the malloc/unlink_chunk/cfree family ~30%). A bitmask is
//   O(1), allocation-free, and a copy is a register move — the per-descent set-copy alloc churn is gone.
//
//   DOMAIN (the basis for the fixed width, ADR-0012 P3 derive-don't-hardcode): a member is a TREASURE id
//   in [0, N). The env packs the world-set as `(1<<t)` sums into a uint32_t (env.hpp worlds(): "20 bits
//   fit a uint32"), so the treasure-id domain is BOUNDED by that world-mask width — N <= 32 structurally.
//   `bits` is a uint64_t (64 >= 32 with headroom); the static_assert below DERIVES the cover obligation
//   from the world-mask type rather than hardcoding 20 or 64, and the per-insert assert is the ADR-0002
//   fail-loud run-time net for an out-of-range id (a bug — an id the world-mask could never carry).
//
//   BIT-EXACTNESS vs std::set<int> (the gate, ADR-0012 P6): the set was used for (i) membership
//   (contains — exclude already-collected treasures from the legal/candidate set), (ii) add-on-collect
//   (insert, in env.apply — idempotent, `bits |=` matches std::set::insert's set-union semantics), and
//   (iii) iteration + |collected| (the collected-indicator feature axis, size() via popcount). The
//   bitmask reproduces (i)/(ii)/size() trivially (same membership, same union, same cardinality). The
//   ONE iteration consumer (features.cpp collected_features) writes a position-INDEPENDENT indicator
//   `coll[i]=1`, so its bytes do not actually depend on order — but for_each_ascending() iterates low->high
//   via countr_zero (exactly std::set's sorted order) regardless, so an order-DEPENDENT consumer (none
//   today) would also stay bit-exact without a per-consumer audit. `collected` is NOT a node-map key
//   anywhere (the gumbel/ismcts transposition tables key on (action_slot, belief_key), never collected) —
//   verified — so the change is observable ONLY through these uses, all preserved.
//
//   A leaf header (only <bit>/<cstdint>/<cassert> + the cycle-free domains.hpp/world.hpp phantom-type
//   leaves) so env.hpp / policy.hpp / features.hpp include it with no cycle, exactly as belief_key.hpp is
//   a leaf type header. The member-arg type is now the typed TreasureId (ADR-0012 P8: the typed signature
//   is the SSOT) — a FaceId/SlotIndex passed here is a hard compile error, not a runtime category slip.
//
// Public Domain (The Unlicense).
#pragma once

#include <bit>
#include <cassert>
#include <cstdint>

#include "chocofarm/domains.hpp"  // TreasureId / CollectedCount — the typed treasure-id + count domains (P1)
#include "chocofarm/world.hpp"

namespace chocofarm {

// The fixed-width bitmask of collected treasure ids. Value semantics: a copy is a register move (no heap),
// the O(1) replacement for the former node-allocating std::set<int>. Bit t set <=> treasure t collected.
struct CollectedSet {
    // The treasure-id domain is the world-mask's domain (the env packs worlds as `(1<<t)` into a
    // uint32_t), so a valid member id is < kMaxId = bit-width(world_mask_t) = 32 — both kMaxId (the
    // run-time bound the asserts guard) and the storage width are DERIVED from the world-mask type, not
    // hardcoded 20/32/64 (P1/P3). The static_assert fails the build (ADR-0002, strongest surface) if a
    // future world-mask type ever outgrew the uint64_t storage; the per-id asserts below guard the actual
    // DOMAIN (kMaxId), not the storage width — an id in [kMaxId, 64) is an id the world-mask cannot carry,
    // a bug, and is caught loud rather than silently set in a tail bit.
    using world_mask_t = World;  // IS the shared World SSOT (world.hpp) now, not a local re-declaration:
                                 // the env's per-world bitmask type (env.hpp worlds()), single-sourced (P1)
    std::uint64_t bits = 0;
    // The treasure-id domain ceiling = bit-width(world_mask_t) = 32; a TreasureCount (the [0,N) upper bound,
    // here the structural [0,32) one), DERIVED from the world-mask type, never hardcoded (P1/P3). The
    // TreasureRep static_cast is the ONE width crossing for this constant (sizeof*8 ∈ size_t -> u16).
    static constexpr TreasureCount kMaxId =
        TreasureCount{static_cast<TreasureRep>(sizeof(world_mask_t) * 8)};  // = 32, the treasure-id domain
    static_assert(kMaxId.value() <= static_cast<TreasureRep>(sizeof(std::uint64_t) * 8),
                  "CollectedSet storage must cover the world-mask's treasure-id domain (ADR-0012 P3)");

    bool operator==(const CollectedSet&) const = default;

    // Membership test (the former std::set::count(i) == 1 / contains(i)). `i` is a typed TreasureId — a
    // FaceId/SlotIndex here does NOT compile (the load-bearing ADR-0000 win). The .value() at the shift is
    // the ONE id->raw crossing (the bit position), the assert the ADR-0002 fail-loud range net.
    [[nodiscard]] bool contains(TreasureId i) const {
        assert(i.value() < kMaxId.value() && "CollectedSet::contains id out of range");
        return ((bits >> i.value()) & 1u) != 0;
    }

    // The immutable add (the descent's `collected ∪ {slot}`): returns a NEW set with bit i set, leaving
    // *this unchanged — a register-move value, the allocation-free replacement for the per-descent copy.
    [[nodiscard]] CollectedSet with(TreasureId i) const {
        assert(i.value() < kMaxId.value() && "CollectedSet::with id out of range");
        return CollectedSet{bits | (std::uint64_t{1} << i.value())};
    }

    // The in-place add (the former std::set::insert(i)) — env.apply mutates `collected` in place on a
    // treasure collect. ADR-0002: fail-loud on an out-of-range id (an id the world-mask cannot carry).
    void insert(TreasureId i) {
        assert(i.value() < kMaxId.value() && "CollectedSet::insert id out of range");
        bits |= (std::uint64_t{1} << i.value());
    }

    // |collected| (the former std::set::size()) — popcount, O(1). The stdlib ACL: std::popcount returns a
    // non-negative signed int, crossed to the typed CollectedCount via the explicit ctor (the named .size()
    // boundary; popcount ∈ [0,64], always fits TreasureRep).
    [[nodiscard]] CollectedCount size() const {
        return CollectedCount{static_cast<TreasureRep>(std::popcount(bits))};
    }
    [[nodiscard]] bool empty() const { return bits == 0; }

    // ASCENDING iteration (low bit -> high bit == std::set<int> sorted order). `fn` is called with each
    // collected id (a typed TreasureId), in ascending order. The one current consumer (features.cpp
    // build_into's collected/available/sum_unc axes) writes a position-independent `coll[i]=1`, so order
    // does not affect ITS bytes; the ascending guarantee is conservative — it preserves the former std::set
    // order so an order-DEPENDENT consumer (none today) would also stay bit-exact, no per-consumer audit.
    template <typename Fn>
    void for_each_ascending(Fn&& fn) const {
        std::uint64_t b = bits;
        while (b != 0) {
            // lowest set bit = smallest remaining id; countr_zero ∈ [0,63] crosses to TreasureId (ACL).
            const TreasureId i{static_cast<TreasureRep>(std::countr_zero(b))};
            fn(i);
            b &= b - 1;                          // clear the lowest set bit
        }
    }
};

}  // namespace chocofarm
