// cpp/include/chocofarm/env.hpp
// Purpose: the minimal C++ env port — the SIMULATION MODEL decoupled from any Policy, mirroring
//   chocofarm/model/env.py. It owns the belief world-set (C(N,K) bitmasks), the legal-action set,
//   `apply` (move -> observe -> filter belief -> collect), the geometry distances, and the exact
//   belief filters (filter_treasure / filter_detector / the disjunction over a face's cover). It
//   knows nothing about HOW a decision is made — that is a Policy (policy.hpp), injected. A new
//   capability is a new Policy subclass with ZERO edits here (ADR-0012 P2, the env<->Policy seam).
//
//   NOT ported (deferred to a later slice, per ADR-0012's C++ section and scaling-and-cpp-seam.md
//   Shape A): the Gumbel-AZ search and the MLP forward. This is the dumb-random MVP that proves the
//   wire + the env<->Policy seam + the belief mechanics.
//
// Public Domain (The Unlicense).
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <random>
#include <span>
#include <variant>
#include <vector>

#include "chocofarm/belief_key.hpp"
#include "chocofarm/collected_set.hpp"
#include "chocofarm/instance.hpp"
#include "chocofarm/world.hpp"

// The OPT-IN ZDD belief engine — included at GLOBAL scope (NOT inside namespace chocofarm: a standard-
// header include inside a namespace mis-nests std as chocofarm::std). Present only under the flag; the
// ZddBelief value type + the zdd:: op declarations live in this header (env.hpp) alongside the variant,
// under the same #ifdef — the whole ZDD-arm flag surface is here + env.cpp + features.cpp's one visit.
#ifdef CHOCO_BELIEF_ZDD
#include "chocofarm/belief_zdd_engine.hpp"
#endif

namespace chocofarm {

// The belief value type — the world-set the search reasons over (ADR-0012 P2: the belief seam). STEP 1
// of the belief-rep cutover (docs/design/cpp-belief-rep-scoping.md §5) introduced ONE belief value the
// search names, replacing the bare `std::vector<uint32_t>` every caller used to poke directly. STEP 2
// (this slice, §5 "Add the bitset arm + the gate") swaps the alias to a `std::variant<FlatBelief,
// BitsetBelief>`: the search now names TWO representations behind one opaque value, chosen ONCE per env
// (the gate, §4) and invariant for the env's life. Every caller already speaks `const Belief&` / `Belief`
// / `Belief&`, so this slice touches ONLY this alias + the env op bodies + the few tools that build a
// belief DIRECTLY (the A/B oracle, the benches). The seam ops dispatch COARSELY (one std::visit /
// holds_alternative per env op, §3), then run a pure rep-specific body — NEVER a visit inside a per-world
// loop. The bitset arm is BYTE-IDENTICAL to the flat arm (filters, counts, features, sampling,
// fingerprint); the flat arm is the reference (any divergence is a bug, ADR-0002).

// The FLAT arm: the world-set as an explicit vector of bitmasks, in worlds()-RANK order (the general base
// + the non-enumerable fallback). RANK order is the combinations(range(N), K) emission order of
// build_worlds (env.cpp) — NOT numerically ascending (the bitmask values are not monotone: e.g. world rank
// 15 is the high-bit combination {0,1,2,3,19} = 524303, rank 16 is {0,1,2,4,5} = 55). The flat belief is
// always a SUBSEQUENCE of worlds_ (it starts as worlds_ and erase_if preserves order), so it stays in rank
// order; the bitset arm indexes the SAME worlds_ by rank. THAT shared rank order — not a numeric one — is
// the bit-exactness basis (front()/back() == rank-0/rank-(count-1) world for the same belief). The seam ops
// below are the SOLE readers/mutators.
struct FlatBelief {
    std::vector<World> worlds;
    bool operator==(const FlatBelief&) const = default;
};

// The inline belief-word CAPACITY (NOT the live kW64). The bitset arm's `bits` is a FIXED-CAPACITY inline
// std::array of this many words so a per-node belief copy is an inline value-copy, NOT a ~2 KiB heap
// alloc+free (the profiled per-copy malloc/free cost the inline storage kills). This is a CAPACITY, like a
// fixed buffer's size — NOT the live 243: the gate (env ctor) ADDITIONALLY requires kW64 <= kBitsetMaxWords
// (else the inline buffer cannot hold the belief and the env falls to the flat arm). 256 comfortably covers
// the live kW64=243 (243 worlds-words for |worlds|=15504) with headroom; raise it (and re-measure the
// inline footprint) before an instance whose ceil(|worlds|/64) exceeds 256 can use the bitset arm. The
// runtime word count is `kw64_` below (= env's kW64_, <= kBitsetMaxWords); the ops iterate the first kw64_
// words via a span, NEVER the full array.
inline constexpr int kBitsetMaxWords = 256;

// The BITSET arm (gated fast path): a dense bitvector over the env's enumerated worlds (bit r set <=>
// world of rank r is live), packed into kw64_ = ceil(|worlds|/64) words held in a FIXED-CAPACITY INLINE
// std::array<uint64_t, kBitsetMaxWords>. The inline buffer replaces the former std::vector<uint64_t> (whose
// per-copy heap alloc+free — ~2 KiB on every descent-local belief copy — the K=128 client profile flagged
// as ~14.5% of the malloc/free family): the bitset variant now copies inline, no allocation. The scoping
// report (§1/§3) ORIGINALLY chose `std::array<uint64_t, kW64>` for exactly this reason; the STEP-2 note
// switched it to a vector to keep kW64 runtime-derived (derive-don't-hardcode). This refactor reconciles
// both: the CAPACITY (kBitsetMaxWords) is a named compile-time buffer cap (NOT 243), while the ACTUAL word
// count `kw64_` stays RUNTIME-derived from the env (= env's kW64_) — so no live dimension is hardcoded in
// the TYPE, and the per-copy alloc is gone. The variant grows to ~2 KiB inline (the §3 cost, accepted: the
// per-copy alloc this removes is the worse cost). Per-belief footprint: a fixed ~2 KiB inline, NO heap.
//
// `kw64_` is the runtime word count (= env.kW64() <= kBitsetMaxWords); the ops read ONLY the first kw64_
// words (a std::span(bits.data(), kw64_)), never the full kBitsetMaxWords array. `count_` is the cached
// popcount (the O(1)-nb obligation, §6 risk 2): updated in EVERY filter, NEVER recounted at a guard.
//
// `operator== = default` compares the WHOLE array (all kBitsetMaxWords words) + kw64_ + count_. That is
// CORRECT precisely because the unused tail words [kw64_, kBitsetMaxWords) are ALWAYS zero: the array is
// zero-initialized at construction (the `{}` member initializer), full_belief() writes only words [0,kw64_)
// and the partial-tail mask leaves [kw64_,end) at 0, and the filters only AND existing words (an AND never
// sets a previously-zero tail word). Two beliefs with the same first-kw64_ words therefore have identical
// tails, so the whole-array `= default` compare is equivalent to comparing the first kw64_ words — and
// clearer (no hand-written loop, no kw64_ to thread). The A/B oracle nets this (belief_key / operator== /
// every derived value byte-identical to the flat arm).
struct BitsetBelief {
    std::array<uint64_t, kBitsetMaxWords> bits{};  // first kw64_ words live; tail [kw64_,end) always 0
    int kw64_ = 0;                                 // runtime word count (= env.kW64() <= kBitsetMaxWords)
    int count_ = 0;                                // cached popcount(first kw64_ words) — the O(1) nb
    bool operator==(const BitsetBelief&) const = default;

    // The LIVE words [0, kw64_) as a span — the ONE place the inline-array/runtime-count split is bridged
    // (P1). Every bitset op (the masked-popcount kernels, the filter, world_at_rank/belief_key) reads THIS,
    // never the full kBitsetMaxWords array (which would count the always-zero tail — harmless for popcount
    // but wrong for any size-derived op). Const + mutable overloads so the filter can &= in place.
    [[nodiscard]] std::span<const uint64_t> live() const {
        return {bits.data(), static_cast<size_t>(kw64_)};
    }
    [[nodiscard]] std::span<uint64_t> live() {
        return {bits.data(), static_cast<size_t>(kw64_)};
    }
};

#ifdef CHOCO_BELIEF_ZDD
// The OPT-IN belief-as-diagram (ZDD) arm — the §B.4(b) graduation (belief_features_and_decision_diagram
// _note.md Part B; docs/design/cpp-belief-zdd-onramp.md). A thin value wrapper over the maintained
// BeliefDiagram (the engine owns the per-belief arena; this is the seam's opaque value — copyable,
// value-semantics, == by the diagram's CANONICAL STRUCTURE, O(|Z|): see BeliefDiagram::operator==). The
// diagram is maintained THROUGH the search:
// filter_treasure/detector are RESTRICT ops on `z` in place, not a rebuild. cached_count_ mirrors
// z.count() so nb() is O(1) (the same O(1)-nb obligation the bitset arm's count_ serves); it is
// recomputed in EVERY filter. The EQUIVALENCE ASYMMETRY (design §4 of the B.4(b) task): the ZDD arm is
// BIT-EXACT on filters/counts/features (restrict gives members set-equal to the flat filter;
// all_marginals/all_detector_counts + the IDENTICAL Phase-2 * inv give byte-identical features), BUT the
// sampling/fingerprint trio (sample_world / world_at_rank / belief_key) RE-BASELINES — the ZDD's
// canonical member order != worlds()-rank order, so the r-th ZDD member != the flat bw[r] (an O(nb) rank
// map would defeat |Z|≪nb). The scripted gumbel/ismcts parity diverges on the ZDD arm (a different world
// index), the JAX-reorder behavioral bucket — EXPECTED, not a bug.
struct ZddBelief {
    beliefzdd::BeliefDiagram z;
    int64_t cached_count_ = 0;  // = z.count(); the O(1) nb; recomputed after each filter
    // Value-equality by the diagram's CANONICAL STRUCTURE — z == o.z compares the canonical layout
    // (n_, root_, the byte-identical nodes_ array) in O(|Z|), with NO enumeration and NO allocation. This
    // replaces the former z.members() == o.z.members(), which fully enumerated BOTH world-sets (O(nb) + two
    // heap vectors) on every belief-cache full-equality verify (the belief_key fingerprint pre-filter, then
    // this net — ~40% of the ZDD client self-time post the value-copy fix). compact() canonicalizes the
    // layout at every mutation exit, so two diagrams of the SAME family (reached via any restrict sequence)
    // are byte-identical and structural == is EXACT (a canonical reduced ZDD is its family's unique
    // representation: structural-equal ⟺ family-equal — no false positives, never a wrong cache value; the
    // only conceivable failure is a harmless false-negative cache miss). See BeliefDiagram::operator==.
    bool operator==(const ZddBelief& o) const { return z == o.z; }
};

// The Belief variant — THREE arms under the flag (flat + bitset + ZDD), TWO in the default build.
using Belief = std::variant<FlatBelief, BitsetBelief, ZddBelief>;

// The per-op ZDD `else` arm injector — the WHOLE visit-side flag surface is this one macro (+ the variant
// alias above). A seam-op std::visit reads:
//     if constexpr (FlatBelief) {..} else if constexpr (BitsetBelief) {..} CHOCO_ZDD_ELSE(return zdd::OP;)
// ON  -> `else { return zdd::OP; }` (the third arm). OFF (the #else) -> empty (the variant has only two
// alternatives, so std::visit instantiates only flat+bitset and the macro vanishes — the default build's
// visit is byte-for-byte the current one). Defined in BOTH branches so env.cpp/features.cpp always have it.
#define CHOCO_ZDD_ELSE(...) else { __VA_ARGS__ }
#else
// The DEFAULT build: the live flat + bitset arms ONLY, byte-for-byte unchanged.
using Belief = std::variant<FlatBelief, BitsetBelief>;
#define CHOCO_ZDD_ELSE(...)  // the third arm vanishes in the default build (no ZddBelief alternative)
#endif

#ifdef CHOCO_BELIEF_ZDD
// The ZDD-arm seam-op bodies (the §B.4(b) arm) — free functions the env-op visits' CHOCO_ZDD_ELSE branch
// calls, DEFINED in env_zdd.cpp (which has the full Environment + features.hpp). Declared HERE (after the
// variant + the ZddBelief value type, before the inline nb() and the Environment class that use them) so
// the call sites see the declarations. Environment / BeliefFeatures are forward-declared — the op
// DECLARATIONS need only that (env_zdd.cpp has the full types). Each op MIRRORS its flat/bitset twin
// BYTE-IDENTICALLY on counts/marginals/det-counts/features and gives members set-equal to the flat filter;
// only world_at_rank (the sampling unrank) re-baselines (the canonical-order note on ZddBelief). NB: no
// zdd::legal_actions / zdd::sample_world — Environment::legal_actions composes marginals()+informative()
// and Environment::sample_world composes nb()+world_at_rank(), both already visit-dispatching to the ZDD
// arm (one home, P1) — only the leaf ops need a ZDD body.
class Environment;       // full class body below (the op declarations need only the forward decl)
struct BeliefFeatures;   // full definition in features.hpp (env_zdd.cpp / the A/B include it)
namespace zdd {
[[nodiscard]] ZddBelief full_belief(const Environment& env);                       // Z over every world (C(N,K))
[[nodiscard]] inline int nb(const ZddBelief& b) { return static_cast<int>(b.cached_count_); }  // O(1)
void filter_treasure(ZddBelief& b, int i, bool present);                           // restrict_var in place
void filter_detector(const Environment& env, ZddBelief& b, int i, bool positive);  // restrict_cover
[[nodiscard]] uint32_t world_at_rank(const ZddBelief& b, int r);                   // r-th CANONICAL member (re-baseline)
[[nodiscard]] std::vector<double> marginals(const Environment& env, const ZddBelief& b);
[[nodiscard]] bool informative(const Environment& env, int face_id, const ZddBelief& b);
[[nodiscard]] BeliefFeatures belief_features(const Environment& env, const ZddBelief& b);
}  // namespace zdd
#endif

// The machine-cache budget the derived mask set must fit (the gate's second input, §4). NAMED here as
// what it is — a target-cache fact, not a derived quantity: half of the i5-6600 (Skylake) 256 KiB
// per-core L2, leaving the other half for the live belief + the per-step working set. The live mask set
// is (N + nD) * kW64 * 8 = 64 * 243 * 8 = 124416 B ≈ 121.5 KiB <= 131072 B, so the live instance lands on
// the bitset side. A constexpr (ADR-0012 P9: a typed compile-time constant, not a #define).
inline constexpr std::size_t kTargetMaskCacheBudgetBytes = 128 * 1024;

// An action: a kind tag + an index. Mirrors the env's ("t", i) / ("d", i) / TERMINATE shape.
enum class ActionKind { Treasure, Detector, Terminate };
struct Action {
    ActionKind kind = ActionKind::Terminate;
    int i = -1;  // treasure id (Treasure) or face id (Detector); -1 for Terminate
    bool operator==(const Action& o) const { return kind == o.kind && i == o.i; }
};
inline Action terminate_action() { return Action{ActionKind::Terminate, -1}; }

// Where the agent currently stands. Mirrors the env's loc tuple: a teleport ("w", k), a treasure
// ("t", i), or a detector ("d", i) rep_point. We carry the resolved Point so distance is a pure
// (Point, Point) -> double (std::hypot), mirroring env.d's static distance table (same hypot inputs).
struct Loc {
    Point pt;
};

// The result of apply (mirrors env.apply's (reward, loc', bw', collected', dt)). `bw` is mutated in
// place on the Environment for the running episode (the Python env returns a fresh filtered array;
// here we filter the member vector — same belief, same elements, same order).
struct StepResult {
    double reward = 0.0;
    double dt = 0.0;
};

class Environment {
  public:
    explicit Environment(const Instance& inst);

    // ---- belief world-set ----
    // The full C(N,K) world-set as bitmasks over treasure ids (mirrors world_array): all K-subsets
    // of range(N), in itertools.combinations order, as (1<<t) sums. 20 bits fit a uint32.
    const std::vector<World>& worlds() const { return worlds_; }
    int N() const { return inst_.N; }
    int K() const { return inst_.K; }
    int n_detectors() const { return static_cast<int>(inst_.faces.size()); }

    // ---- geometry ----
    double dist(const Point& a, const Point& b) const;        // std::hypot (mirrors env.d/math.hypot)
    double exit_cost(const Point& loc) const;                 // min teleport dist + tp (env.exit_cost)
    Point entry_point() const { return inst_.teleports[entry_idx_]; }
    Point treasure_pt(int i) const { return inst_.treasures[i]; }
    Point face_pt(int i) const { return inst_.faces[i].rep_point; }
    // Per-treasure reward magnitude (mirrors env.value[i]). The live instance carries unit values
    // (env.py: `self.value = [1.0]*N`), the same constant `apply` already banks as the collect
    // reward — one home for the magnitude, read explicitly here so a value-using policy (the NMCS
    // GreedyPolicy base) mirrors Python's `env.value[i]` rather than hiding the 1.0 in a literal.
    double value(int i) const { (void)i; return 1.0; }
    // The single episode-horizon home (mirrors env.max_steps): the safety-net cap a base playout
    // runs to (solvers.base._base_value loops `range(env.max_steps)`). One source of truth so a
    // playout's horizon cannot silently desync from the Python env's.
    int max_steps() const { return 40; }
    int n_teleports() const { return static_cast<int>(inst_.teleports.size()); }
    Point teleport_pt(int k) const { return inst_.teleports[k]; }

    // ---- belief construction (the seam's entry — replaces `bw = env.worlds()`) ----
    // The full belief: every world (the C(N,K) prior). When the gate is on (use_bitset_) the bitset arm:
    // all-ones over the nb worlds (count_ = |worlds|); else the flat arm (copy `worlds_`, in rank order).
    Belief full_belief() const;

    // ---- belief introspection (the seam ops — COARSE visit per op, NEVER per-world; §3) ----
    // belief size. Coarse visit: the flat arm reads .worlds.size(); the bitset arm returns the CACHED
    // count_ (NOT a recount — the O(1)-nb obligation, §6 risk 2). One predicted branch per call.
    int nb(const Belief& b) const {
        return std::visit([](const auto& a) -> int {
            using T = std::decay_t<decltype(a)>;
            if constexpr (std::is_same_v<T, FlatBelief>) return static_cast<int>(a.worlds.size());
            else if constexpr (std::is_same_v<T, BitsetBelief>) return a.count_;  // cached popcount, never a recount
            CHOCO_ZDD_ELSE(return zdd::nb(a);)  // ZDD arm (opt-in): the cached_count_, O(1) (empty in the default build)
        }, b);
    }
    bool empty(const Belief& b) const { return nb(b) == 0; }  // derives from nb (one home; O(1) both arms)
    // The r-th world by RANK (worlds()-position = combinations order, NOT numeric — see FlatBelief). Flat:
    // worlds[r] (the belief is a rank-ordered subsequence). Bitset: the r-th set bit, unranked through
    // worlds_ (the SAME rank order, so byte-identical to the flat arm for the same r — the bit-exactness
    // basis). The scripted parity sources resolve their `bw[idx]` poke through this (L4), preserving the
    // exact index.
    uint32_t world_at_rank(const Belief& b, int r) const;
    // Sample one concrete world uniformly from the belief (mirrors env.sample_world / rng.choice(bw)).
    // MOVED here from RngWorldSource (L1) so no caller pokes the representation. The uniform draw is the
    // IDENTICAL std::uniform_int_distribution<size_t>(0, nb-1)(rng) on BOTH arms, so the RNG stream is
    // byte-identical; the drawn rank r then unranks via world_at_rank — byte-identical to the flat index.
    uint32_t sample_world(const Belief& b, std::mt19937_64& rng) const;
    // The ONE belief-identity fingerprint (L2), MOVED off belief_key.hpp into the env so the seam owns
    // the read of the representation. (count, first, last); {0,0,0} on the empty belief. Flat: (size,
    // front, back). Bitset: (count_, world_at_rank(0), world_at_rank(count_-1)) — bit-identical to the
    // flat triple (the RANK order is shared: flat front/back ARE the rank-0/rank-(count-1) worlds), so the
    // cache hit-rate / gumbel transposition behaviour
    // is preserved exactly (§6 risk 5). The `using BeliefKey` TYPE stays in belief_key.hpp (a leaf header
    // gumbel.hpp/features.hpp include — moving the TYPE would create an include cycle).
    BeliefKey belief_key(const Belief& b) const;

    // ---- belief marginals (mirrors env.marginals) ----
    std::vector<double> marginals(const Belief& bw) const;

    // ---- dynamics ----
    // Legal action set for (loc, belief, collected): collects with marg>0 and not collected, plus
    // each face whose outcome is still uncertain over the belief (informative). TERMINATE is NOT
    // included here (it is the always-legal extra slot, appended by the Policy / the mask builder),
    // matching env.legal_actions + actions.term_slot exactly.
    std::vector<Action> legal_actions(const Belief& bw,
                                      const CollectedSet& collected) const;

    // Realise `action` against the true `world`. Filters `bw` IN PLACE (move/observe/collect), and
    // returns (reward, dt). The belief filter is the same disjunction/treasure-bit logic as env.py.
    StepResult apply(Loc& loc, Belief& bw, CollectedSet& collected,
                     const Action& action, uint32_t world) const;

    // ---- belief filters (mirror filter_treasure / SenseAction.filter) ----
    void filter_treasure(Belief& bw, int i, bool present) const;
    void filter_detector(Belief& bw, int i, bool positive) const;

    // A face's true reading at a concrete world (mirrors SenseAction.observe).
    bool observe(int face_id, uint32_t world) const {
        return (world & inst_.faces[face_id].bitmask) != 0;
    }
    // The per-detector cover bitmasks as a CONTIGUOUS view (face j's bitmask at index j) — the same
    // masks observe() / filter_detector read one-at-a-time off inst_.faces (an array-of-structs, so
    // `.bitmask` strides by sizeof(Face)). Homed once in the ctor so a per-world sweep over all nD
    // detectors reads them packed, without that stride (ADR-0012 P1: env still owns the masks; this is
    // one contiguous read of them). The belief sweep relies on the identity
    // observe(j, w) == ((w & face_masks()[j]) != 0).
    std::span<const uint32_t> face_masks() const { return face_masks_; }
    // Outcome still uncertain over the belief — both polarities live (SenseAction.informative).
    bool informative(int face_id, const Belief& bw) const;

    // ---- the bitset arm's env-static state + gate (§3/§4) ----
    // Whether the bitset arm is active for THIS env (the gate decision, computed ONCE in the ctor and
    // invariant for the env's life). The full_belief() the search starts from takes the chosen arm.
    bool use_bitset() const { return use_bitset_; }
#ifdef CHOCO_BELIEF_ZDD
    // Whether the OPT-IN ZDD arm is active (the §B.4(b) gate, ON build only). When true the gate SELECTS
    // ZDD over bitset/flat — full_belief() returns a ZddBelief and the whole search runs on the maintained
    // diagram (the head-to-head profile vs the bitset). Gated on worlds enumerable (the diagram is built
    // from worlds()); this is the WHOLE selection surface — no call site decides (the gate does, §4).
    bool use_zdd() const { return use_zdd_; }
#endif
    int kW64() const { return kW64_; }  // ceil(|worlds|/64) — the bitset word count (0 when not enumerable)
    // The env-static masks the bitset bodies AND-against, homed in the ctor like face_masks_ (P1):
    // treasure_mask_[t] = bitvector of worlds (by rank) with bit t set; detector_mask_[j] = bitvector of
    // worlds where (w & face_masks()[j]) != 0 (the identity env.observe rests on). Each is kW64 words.
    // Built only when use_bitset_; empty otherwise.
    std::span<const uint64_t> treasure_mask(int t) const {
        return {treasure_mask_.data() + static_cast<size_t>(t) * static_cast<size_t>(kW64_),
                static_cast<size_t>(kW64_)};
    }
    std::span<const uint64_t> detector_mask(int j) const {
        return {detector_mask_.data() + static_cast<size_t>(j) * static_cast<size_t>(kW64_),
                static_cast<size_t>(kW64_)};
    }

  private:
    Instance inst_;
    std::vector<World> worlds_;
    std::vector<uint32_t> face_masks_;  // contiguous per-detector cover bitmasks (face_masks(); built in ctor)
    int entry_idx_ = 0;

    // ---- bitset arm gate + env-static masks (built in the ctor, P1) ----
    bool use_bitset_ = false;   // the gate decision (§4): worlds enumerable AND mask_bytes <= budget
#ifdef CHOCO_BELIEF_ZDD
    bool use_zdd_ = false;      // the §B.4(b) gate (ON build only): worlds enumerable -> select the ZDD arm
#endif
    int kW64_ = 0;              // ceil(|worlds|/64) — derived, never the literal 243
    // Flattened kW64-word mask tables (row t/j is masks[row*kW64_ .. row*kW64_+kW64_]). std::vector (not
    // std::array): kW64_ is runtime-derived (see BitsetBelief). treasure: N rows; detector: nD rows.
    std::vector<uint64_t> treasure_mask_;
    std::vector<uint64_t> detector_mask_;
};

// One-owner in-place belief compaction (ADR-0012 P1/P3): keep the worlds where ((w & mask) != 0) == want,
// in order; returns the kept count. filter_treasure / filter_detector are thin wrappers differing ONLY by
// the mask (a treasure is the single-bit mask 1<<i; a detector is its cover bitmask) — the SAME operation,
// one compaction + two predicates (the unification is the win). Idiomatic std::erase_if body (see env.cpp:
// a hand-branchless variant was measured slower under -march=native and rejected); bit-exact with the
// former per-method erase(remove_if) — same kept set, same order (P6). Exposed as a free function so the
// belief-filter A/B bench can drive the real compaction directly (the same bounded exposure as belief_features).
std::size_t filter_inplace(std::vector<uint32_t>& bw, uint32_t mask, bool want);

}  // namespace chocofarm
