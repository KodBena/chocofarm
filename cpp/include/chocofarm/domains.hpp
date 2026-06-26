// cpp/include/chocofarm/domains.hpp
// Purpose: the ONE home (ADR-0012 P1) for the chocofarm-core integer-DOMAIN phantom types beyond the
//   world count/rank pair (which stay in world.hpp, the worked seed). Every distinct integer domain the
//   env / instance / feature / search subsystems speak — a treasure id, a face id, a teleport id, the
//   instance cardinalities N / nD / n_tel / K, an action-slot index + the slot count, a feature-vector
//   dimension, a belief-bitset word count + word index, a search ply/depth, a sim/budget/candidate count,
//   an outcome index, a visit count, a collected-set cardinality — is minted HERE as a zero-cost phantom
//   over the Quantity<Tag, Rep> machinery (quantity.hpp), with its sign + width MOTIVATED at the
//   declaration (ADR-0000 rule 1: never "int by default"). Synonyms that the integer-domain inventory
//   spread across env/features/gumbel/ismcts/search are MERGED into one type each here — e.g. the
//   FaceId/DetectorId pair, the TreasureCount(N) the feature builder and the env both speak, the
//   ActionSlot the gumbel and ismcts trees both index — so an implementer retypes a USAGE, never
//   re-authors a type (ADR-0012 P8: the typed signature is the SSOT; P1: one home).
//
//   ADR-0000 is the frame: the load-bearing wins are the IDs (TreasureId / FaceId / TeleportId / SlotIndex
//   are four DISTINCT non-interconvertible tags, so the classic "a face id used where a treasure id is
//   owed" — both bare `int` today, both reaching the overloaded Action::i field — becomes a hard COMPILE
//   error, not a runtime category error), and the count-vs-index pairs (a TreasureCount is not a
//   TreasureId, a SlotCount is not a SlotIndex, a WordCount is not a WordIndex — the size and the thing it
//   bounds are different domains, ADR-0008 type discipline).
//
//   ACL crossings (ADR-0000 item 5: every raw<->domain crossing is named + visible): the EXPLICIT
//   Quantity(Rep) ctor and .value() are the only crossings; the loaders' std::stoi/get<int> JSON boundary
//   (instance.cpp) is the one signed->unsigned-id ACL, fail-loud on a negative at the boundary (ADR-0002);
//   .size() / std::popcount (stdlib, non-negative by construction) is the size_t/signed->count ACL; the
//   motivated cross-domain bridges (TreasureId/FaceId -> SlotIndex, WorldCount -> WordCount via /64) are
//   the NAMED functions at the foot of this header, mirroring world.hpp's last_rank().
//
//   A leaf header (only <cstdint> + the Quantity machinery) so env.hpp / instance.hpp / features.hpp /
//   gumbel.hpp / ismcts.hpp / search_runtime.hpp include it with no cycle.
//
// Public Domain (The Unlicense).
#pragma once

#include <cstddef>
#include <cstdint>

#include "chocofarm/quantity.hpp"  // Quantity<Tag, Rep> — the zero-cost phantom-type SSOT (P1)
#include "chocofarm/world.hpp"     // World / WorldCount / WorldRank (the seeded world domain) + bridges

namespace chocofarm {

// ============================================================================================
//  MOTIVATED REPRESENTATION WIDTHS (ADR-0000 rule 1: width/sign chosen at the declaration)
// ============================================================================================
// All these domains are NON-NEGATIVE (a count, a cardinality, a 0-based index) => UNSIGNED, never the
// arbitrarily-signed `int` the legacy spellings reached for by habit. Width is motivated, not reflexive.

// The treasure-id domain rep. A treasure id / count is a bit position in the uint32 World mask, so the
// STRUCTURAL ceiling is [0, 32) (CollectedSet::kMaxId = bit-width(World) = 32). 16 bits covers it with
// vast headroom while staying non-reflexive; the width also serves N and K (both <= 32 by the same
// world-mask argument) so the (N, K, treasure-id) family shares one motivated home (P1).
using TreasureRep = std::uint16_t;
static_assert(sizeof(TreasureRep) == 2,
              "treasure-id/N/K width is motivated as 16-bit: the domain is [0,32) (the World mask's "
              "bit-width ceiling); 16 bits covers it with headroom, narrower than a reflexive int.");

// The face/detector + teleport id-and-count rep. A face id indexes Instance::faces (44 live); a teleport
// id indexes Instance::teleports (single digits live). Neither has the World-mask 32-bit structural tie;
// both are bounded by the arrangement geometry, comfortably < 2^16. 16 bits is the motivated cover.
using GeometryIdRep = std::uint16_t;
static_assert(sizeof(GeometryIdRep) == 2,
              "face/teleport id width is motivated as 16-bit: bounded by |faces|/|teleports| (geometry "
              "cardinalities, tens not thousands), well under 2^16; narrower than a reflexive int.");

// The action-slot + feature-dimension rep. n_slots = N + nD + 1 (65 live); dim = 5N+3nD+6+n_tel (241
// live). Both are low-hundreds sums of the cardinalities above; 16 bits covers with headroom.
using LayoutRep = std::uint16_t;
static_assert(sizeof(LayoutRep) == 2,
              "slot-index/slot-count/feature-dim width is motivated as 16-bit: N+nD+1 and 5N+3nD+6+n_tel "
              "are low-hundreds (65/241 live), far under 2^16; narrower than a reflexive int.");

// The belief-bitset word rep (kW64 = ceil(|worlds|/64); word index within [0, kW64)). Bounded by
// kBitsetMaxWords = 256 for the bitset arm to be selected at all (the gate's fits_inline conjunct);
// live kW64 = 243. 16 bits covers it, distinct from the world count it is derived from (words != worlds).
using WordRep = std::uint16_t;
static_assert(sizeof(WordRep) == 2,
              "bitset word-count/word-index width is motivated as 16-bit: kW64 = ceil(|worlds|/64) <= "
              "kBitsetMaxWords = 256; 16 bits covers it, distinct from the 32-bit world count.");

// The search-budget rep (n_sims, m, c_outcome, visit counts, depth, sims spent, candidate counts). All
// bounded by the configured compute budget — n_sims default 48, ~256/~1024 ceiling per the producer-OOM
// RCA; depth <= max_depth = 24; ΣN <= n_sims. 32 bits is vast headroom and matches the slot-index width
// for the N+nD+1 vs slot arithmetic without per-op conversions. (Wider than the layout rep deliberately:
// a visit total can in principle be summed across many decisions; 32 bits removes any overflow worry.)
using SearchRep = std::uint32_t;
static_assert(sizeof(SearchRep) == 4,
              "search-budget/visit/depth width is motivated as 32-bit: bounded by the configured budget "
              "(n_sims, ~hundreds-to-low-thousands), with headroom for accumulated visit totals.");

// ============================================================================================
//  THE IDENTITY DOMAINS — four DISTINCT, non-interconvertible 0-based id tags (ADR-0000)
// ============================================================================================
// IDs are OPAQUE: no id + id (an id is not additive), no implicit cross-tag use. They are NOT opted into
// quantity_additive. They ARE opted into quantity_affine (an id + a raw offset stays the same id — the
// loop/iteration step), which also gives the `id - id -> raw gap` an iteration occasionally wants. A
// TreasureId used where a FaceId is owed does NOT compile (distinct tags), dissolving the overloaded
// Action::i (treasure id OR face id in one bare-int field) category error.

// A treasure identifier / World-mask bit position, in [0, N) (structurally [0, 32)). Action.i for a
// Treasure action, the bit in World/Face cover, the marginals/dist_t feature index, CollectedSet member.
struct TreasureIdTag {};
using TreasureId = Quantity<TreasureIdTag, TreasureRep>;
template <> struct quantity_affine<TreasureIdTag> : std::true_type {};

// A face / arrangement-face / detector / sense-action identifier, in [0, nD). Action.i for a Detector
// action, the observe()/informative()/detector_mask argument, the index into Instance::faces, the
// p_pos/informative/dist_d feature index. DISTINCT tag from TreasureId (the load-bearing win).
struct FaceIdTag {};
using FaceId = Quantity<FaceIdTag, GeometryIdRep>;
template <> struct quantity_affine<FaceIdTag> : std::true_type {};

// A teleport / save-station / waypoint identifier, in [0, n_tel). teleport_pt(k)'s argument, entry_idx_,
// the loc ('w', k), the dist_w feature index. DISTINCT from TreasureId/FaceId (the env's three loc kinds).
struct TeleportIdTag {};
using TeleportId = Quantity<TeleportIdTag, GeometryIdRep>;
template <> struct quantity_affine<TeleportIdTag> : std::true_type {};

// A flat action-slot index over the fixed action space: slot 0..N-1 = Treasure i, N..N+nD-1 = Detector j,
// N+nD = TERMINATE. The dense W/N/prior/legal/policy vector index, the children-key first component, the
// puct/ucb/SH selected slot, the improved-pi index. DISTINCT from TreasureId/FaceId — slot==id only in
// the treasure range; a detector's slot is N+j, not j (the offset bug a phantom prevents). The -1 "no
// survivor / no best slot / Terminate-sentinel" uses become std::optional<SlotIndex> (ADR-0002 typed
// absence), so the slot itself never carries a negative.
struct SlotIndexTag {};
using SlotIndex = Quantity<SlotIndexTag, LayoutRep>;
template <> struct quantity_affine<SlotIndexTag> : std::true_type {};

// A feature-vector dimension / block-start offset / row index, in [0, dim). The build() output length,
// the BlockOffsets fields, the FeatureLayoutSpec start/width, the float64/float32 row index, the net
// input width. DISTINCT from SlotIndex (a different vector space). Affine (offset + block-relative i).
struct FeatureDimTag {};
using FeatureDim = Quantity<FeatureDimTag, LayoutRep>;
template <> struct quantity_affine<FeatureDimTag> : std::true_type {};
template <> struct quantity_additive<FeatureDimTag> : std::true_type {};  // offset + width -> offset

// A 0-based 64-bit-word index within a belief bitset / mask row, in [0, kW64). bits[w], the w*64+tzcnt
// scan, the mask-table row stride index. DISTINCT from a WorldRank (words != worlds). Affine.
struct WordIndexTag {};
using WordIndex = Quantity<WordIndexTag, WordRep>;
template <> struct quantity_affine<WordIndexTag> : std::true_type {};

// ============================================================================================
//  THE CARDINALITY DOMAINS — counts that BOUND the id domains (count-vs-index, ADR-0008)
// ============================================================================================
// COUNTS are additive (a count + a count is a count) — they are opted into quantity_additive, NOT
// quantity_affine. A count is never an index: a TreasureCount is not a TreasureId (the value [N] is the
// exclusive upper bound of the [0,N) id domain, a different domain than a member of it).

// The number of treasures in the instance, N (= treasures.size()); the upper bound of the TreasureId
// domain, the marginals width, the ZDD variable-universe size. Bounded by N <= 32 (the World mask width).
struct TreasureCountTag {};
using TreasureCount = Quantity<TreasureCountTag, TreasureRep>;
template <> struct quantity_additive<TreasureCountTag> : std::true_type {};

// The exactly-K-of-N present treasure count, K (|worlds| = C(N,K)); 0 <= K <= N <= 32. A structural
// instance parameter. DISTINCT tag from TreasureCount(N): N is the universe size, K the chosen subset
// size — conflating them is meaningless (C(N,N) != the instance). Also the n_collected / marg_sum
// normalizer (collected/K is a count/count ratio with both operands treasure-counts).
struct PresentCountTag {};
using PresentCount = Quantity<PresentCountTag, TreasureRep>;
template <> struct quantity_additive<PresentCountTag> : std::true_type {};

// The cardinality |collected| of the collected-treasure set (CollectedSet::size(), popcount of bits);
// 0 <= |collected| <= N <= 32. A count of treasures, DISTINCT from a WorldCount (collected-id
// cardinality, not world cardinality) and from N/K (a runtime subset size, not an instance parameter).
struct CollectedCountTag {};
using CollectedCount = Quantity<CollectedCountTag, TreasureRep>;
template <> struct quantity_additive<CollectedCountTag> : std::true_type {};

// The number of arrangement faces / detectors, nD (= faces.size()); the upper bound of the FaceId
// domain, the p_pos/informative/dist_d block width, the detector_mask row count. Bounds FaceId.
struct FaceCountTag {};
using FaceCount = Quantity<FaceCountTag, GeometryIdRep>;
template <> struct quantity_additive<FaceCountTag> : std::true_type {};

// The number of teleports, n_tel (= teleports.size()); the upper bound of the TeleportId domain, the
// dist_w block width. Bounds TeleportId.
struct TeleportCountTag {};
using TeleportCount = Quantity<TeleportCountTag, GeometryIdRep>;
template <> struct quantity_additive<TeleportCountTag> : std::true_type {};

// The cardinality of the action-slot space, n_slots = N + nD + 1 (65 live); always >= 1 (TERMINATE
// alone). Sizes every dense per-slot vector, bounds the slot loops, is the Gumbel-draw length. The
// count that BOUNDS SlotIndex (count-vs-index). DISTINCT tag from SlotIndex.
struct SlotCountTag {};
using SlotCount = Quantity<SlotCountTag, LayoutRep>;
template <> struct quantity_additive<SlotCountTag> : std::true_type {};

// The number of 64-bit words in the belief bitset, kW64 = ceil(|worlds|/64) (243 live); <= kBitsetMaxWords
// = 256. The runtime word count the bitset arm iterates, the mask-table row stride. The count that BOUNDS
// WordIndex. DISTINCT from a WorldCount (words != worlds) — the /64 is the named bridge below.
struct WordCountTag {};
using WordCount = Quantity<WordCountTag, WordRep>;
template <> struct quantity_additive<WordCountTag> : std::true_type {};

// ============================================================================================
//  THE SEARCH-SHAPE DOMAINS (gumbel / ismcts / search_runtime)
// ============================================================================================

// A per-action selection/visit count N(a) and its sum ΣN; the per-sim visit `count`. Bounded by the
// sim budget (n==0 <=> unvisited is the load-bearing test, which unsigned expresses with no negative
// representable). Additive (visits accumulate). Subsumes the lone `long sum_n` (one home for the width).
struct VisitCountTag {};
using VisitCount = Quantity<VisitCountTag, SearchRep>;
template <> struct quantity_additive<VisitCountTag> : std::true_type {};

// The Sequential-Halving / search compute budget family: n_sims, the running budget, n_spent, per-phase
// and per-action shares, the ISMCTS iterations, leaf_requests. All <= n_sims; the `budget > 0` /
// `v <= 0` guards are guarding a 0, never a real negative — unsigned makes the never-negative invariant
// structural (ADR-0000). Additive (budget draws down / accumulates).
struct SimBudgetTag {};
using SimBudget = Quantity<SimBudgetTag, SearchRep>;
template <> struct quantity_additive<SimBudgetTag> : std::true_type {};

// The Gumbel-Top-k root-sample / candidate-set width: m (sampled), the considered count, n_phases =
// ceil(log2 m), keep = max(1, |keyed|/2) survivors. All non-negative cardinalities, small. Additive.
struct CandidateCountTag {};
using CandidateCount = Quantity<CandidateCountTag, SearchRep>;
template <> struct quantity_additive<CandidateCountTag> : std::true_type {};

// A search-tree descent depth / ply, in [0, max_depth] (max_depth = 24); the iterate/descend depth, the
// DescendFrame::depth, the episode step counter ep_step against max_steps = 40. A level count, never a
// sentinel (depth 0 is the valid root). Affine (depth + 1 is the only arithmetic).
struct PlyDepthTag {};
using PlyDepth = Quantity<PlyDepthTag, SearchRep>;
template <> struct quantity_affine<PlyDepthTag> : std::true_type {};

// The immediate-outcome determinization count c_outcome and its 0..c_outcome-1 index k (k==0 reuses the
// threaded world, k>0 redraws — the load-bearing distinction, expressible on an unsigned). The
// round-robin survivor cursor (read modulo considered.size()) shares this affine index domain. Affine.
struct OutcomeIndexTag {};
using OutcomeIndex = Quantity<OutcomeIndexTag, SearchRep>;
template <> struct quantity_affine<OutcomeIndexTag> : std::true_type {};

// The number of OS worker threads in the task-parallel runtime (PoolRuntime), clamped to [1, #tasks],
// practically <= host vCPUs (4). A positive count (>= 1 invariant dissolves the std::max(...,1) guard).
// DISTINCT from any sim budget or task index.
struct WorkerCountTag {};
using WorkerCount = Quantity<WorkerCountTag, SearchRep>;
template <> struct quantity_additive<WorkerCountTag> : std::true_type {};

// A 0-based index into the per-decision GumbelNode arena (NodePool = std::pmr::vector<GumbelNode>): the
// children-transposition-table value (action-slot,belief_key) -> arena idx, the descend frame's `node`,
// the `child` produced by emplace_back. DISTINCT from a SlotIndex (a node arena position is not an
// action slot) and from a PlyDepth (not a tree level) — the load-bearing foreclosure: a node index used
// where a slot/depth is owed does not compile. Affine (an index into a contiguous arena; child = size-1).
// The root is arena index 0; the former -1 "no node" sentinel on DescendFrame::node is a typed default
// (the frame's `node` is always assigned a real index before it is read — `stepped`/push order guarantee).
struct NodeIndexTag {};
using NodeIndex = Quantity<NodeIndexTag, SearchRep>;
template <> struct quantity_affine<NodeIndexTag> : std::true_type {};

// ============================================================================================
//  THE MEMORY-SIZE + OPAQUE-SEED DOMAINS (where size_t / a wide opaque rep IS the motivated width)
// ============================================================================================

// A non-negative byte extent or buffer offset: the per-decision pmr arena (kArenaInlineBytes = 32 KiB
// inline floor + the mmap-overflow blocks), the bitset-arm mask-storage bytes (mask_bytes = (N+nD)*kW64*8)
// and the L2-residency budget (kTargetMaskCacheBudgetBytes) the gate weighs it against. size_t is the ONE
// motivated reflexive width: it sizes/indexes raw byte buffers and is what the std::array / pmr / mmap
// APIs demand. DISTINCT from a world/word/element COUNT (a byte budget is not a count of things; the
// *8 / *FLOAT_BYTES is the count->bytes ACL). Additive + affine (a size, and an offset into a buffer).
struct ByteCountTag {};
using ByteCount = Quantity<ByteCountTag, std::size_t>;
template <> struct quantity_additive<ByteCountTag> : std::true_type {};
template <> struct quantity_affine<ByteCountTag> : std::true_type {};

// The per-tree / per-task RNG SEED: an OPAQUE 64-bit bit-pattern that seeds std::mt19937_64 (one stream
// per SearchTask). NOT a count or index — a seed; uint64_t MATCHES the mt19937_64 seed width exactly. No
// arithmetic (opted into neither trait): equality only, so it cannot be confused with a byte/word count.
struct RngSeedTag {};
using RngSeed = Quantity<RngSeedTag, std::uint64_t>;

// ============================================================================================
//  THE MOTIVATED CROSS-DOMAIN BRIDGES (named, visible — ADR-0000 item 5; like world.hpp last_rank)
// ============================================================================================

// |worlds| -> the last valid byte-budget is NOT a bridge; the only byte<->count crossing is *8 at the
// allocation site (the named ACL). The bridges below are the id/count <-> slot/word ones.

// A TreasureId IS its own action slot (the bijection's identity leg on [0, N)). The one motivated
// treasure-id -> slot crossing.
[[nodiscard]] constexpr SlotIndex slot_of_treasure(TreasureId t) noexcept {
    return SlotIndex{static_cast<LayoutRep>(t.value())};
}

// A FaceId maps to slot N + j (the bijection's detector leg). The motivated face-id + N -> slot crossing
// (N is the TreasureCount that offsets the detector block).
[[nodiscard]] constexpr SlotIndex slot_of_face(FaceId j, TreasureCount n) noexcept {
    return SlotIndex{static_cast<LayoutRep>(static_cast<LayoutRep>(n.value()) +
                                            static_cast<LayoutRep>(j.value()))};
}

// The TERMINATE slot = N + nD (the bijection's terminal leg). The motivated (N, nD) -> term-slot crossing.
[[nodiscard]] constexpr SlotIndex term_slot(TreasureCount n, FaceCount nd) noexcept {
    return SlotIndex{static_cast<LayoutRep>(static_cast<LayoutRep>(n.value()) +
                                            static_cast<LayoutRep>(nd.value()))};
}

// The action-slot space cardinality = N + nD + 1. The motivated (N, nD) -> slot-count crossing.
[[nodiscard]] constexpr SlotCount n_action_slots(TreasureCount n, FaceCount nd) noexcept {
    return SlotCount{static_cast<LayoutRep>(static_cast<LayoutRep>(n.value()) +
                                            static_cast<LayoutRep>(nd.value()) + LayoutRep{1})};
}

// kW64 = ceil(|worlds|/64): the one motivated world-COUNT -> WORD-count crossing (the bitset stride).
// WorldCount lives in world.hpp; this is the named /64 bridge between the two count domains.
[[nodiscard]] constexpr WordCount words_to_words64(WorldCount n) noexcept {
    return WordCount{static_cast<WordRep>((n.value() + 63u) / 64u)};
}

}  // namespace chocofarm
