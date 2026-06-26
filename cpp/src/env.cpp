// cpp/src/env.cpp
// Purpose: the minimal C++ env port (see env.hpp). Mirrors chocofarm/model/env.py's belief
//   mechanics, dynamics, and geometry. The belief filters and the world-set are LOGIC-exact
//   (integer bit ops) — bit-identical to the numpy env; the distances are float-equivalent
//   (std::hypot mirroring math.hypot). ADR-0012 P6/P7.
//
//   PHANTOM-TYPE NOTE (ADR-0000 / ADR-0012 P8): the bitset kernels + this TU's internal word loops speak
//   the typed word/world domains (WordCount/WordIndex/WorldCount/WorldRank — domains.hpp/world.hpp), with
//   the raw<->domain crossings named at the ACL (popcount_all().value(), last_rank(), the .size()/kw64_
//   wrap). The BELIEF MEMBER types (BitsetBelief::count_ / kw64_, ZddBelief::cached_count_) and the env's
//   CROSS-FILE public API (nb()/world_at_rank()/sample_world()/belief_key()/Action.i) stay RAW int here:
//   they are read/written by out-of-frame TUs (features.cpp, feature_compute.hpp, gumbel/ismcts/nmcs/
//   policy, the parity/bench harnesses) whose signatures this slice does not touch — the member/API-type
//   migration is gated on retyping those readers together (ADR-0004 minimal-touch under partial scope).
//
// Public Domain (The Unlicense).
#include "chocofarm/env.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <numeric>   // std::iota (build_worlds combination seed)

#include "chocofarm/belief_bitset_ops.hpp"  // popcount_all / popcount_and / rth_set_bit_index (the ONE home, P1)

namespace chocofarm {

namespace {

// DEBUG-only guard on the tail-zero invariant operator== relies on (the unused words [kw64_, kBitsetMaxWords)
// must always be 0, so the whole-array `= default` compare equals comparing the first kw64_ words). Called
// after each production write of a BitsetBelief; compiles to nothing under NDEBUG (the Release build), so it
// is OFF the hot read path entirely. It is the cheap net for a FUTURE writer that sets a tail word (the one
// way the inline-array operator== could silently diverge — the out-of-frame audit's named residual risk).
inline void assert_tail_zero([[maybe_unused]] const BitsetBelief& b) {
#ifndef NDEBUG
    // GENERATOR-FED: the tail words [kw64_, kBitsetMaxWords) are walked in order and the index is not
    // consumed — range over the tail subspan, no counter (ADR-0000). Debug-only/cold. Bit-identical.
    for (uint64_t tail_word : std::span<const uint64_t>(b.bits).subspan(static_cast<size_t>(b.kw64_)))
        assert(tail_word == 0ull &&
               "BitsetBelief tail word nonzero — operator== invariant broken by a writer past kw64_");
#endif
}

// In-place filter: keep where the mask reads `want` (bits &= want ? mask : ~mask), then recompute count_.
// The one place count_ is written (§6 risk 8). O(kw64_) — the same cost as the AND. Iterates the LIVE words
// [0, kw64_) (b.live()), NOT b.bits.size() (now the inline CAPACITY kBitsetMaxWords): the mask span is
// exactly kw64_ words (env.treasure_mask/detector_mask), so an AND only ever clears bits in the live words —
// it never touches the always-zero tail, preserving the tail-zero invariant operator== relies on.
void filter_bits(BitsetBelief& b, std::span<const uint64_t> mask, bool want) {
    std::span<uint64_t> bits = b.live();
    // The live word stride as a typed WordCount; the AND scan walks the WordIndex domain (domains.hpp).
    // `kw64_` is the still-raw BitsetBelief member (its count/word-domain migration is gated by the
    // cross-file readers of the member, see this file's header note) — wrapped here at the loop boundary.
    const WordCount nwords{static_cast<WordRep>(b.kw64_)};
    if (want) for (WordIndex w{0}; w.value() < nwords.value(); w = w + WordRep{1}) bits[w.value()] &=  mask[w.value()];
    else      for (WordIndex w{0}; w.value() < nwords.value(); w = w + WordRep{1}) bits[w.value()] &= ~mask[w.value()];
    // popcount_all returns a typed WorldCount; count_ is still the raw cached int (the count-domain
    // migration of the Belief members is a follow-on sweep) — unwrap at this seam (ADR-0000 item 5).
    b.count_ = static_cast<int>(popcount_all(b.live()).value());
    assert_tail_zero(b);  // debug-only: an AND never sets a tail word, but net the invariant explicitly
}

// The r-th set bit -> world rank, with the loud-abort invariant arm the seam owns (the kernel returns a
// TYPED ABSENCE — std::nullopt — on a count_/bits desync; here that becomes a FATAL abort, ADR-0002 /
// scoping §6 risk 7). `r` is wrapped into the WorldRank domain at this seam; the returned global index is
// unwrapped back to the raw int the caller still uses (the rank-domain migration is a follow-on sweep).
[[nodiscard]] int rank_or_abort(std::span<const uint64_t> bits, int r) {
    const std::optional<WorldRank> idx = rth_set_bit_index(bits, WorldRank{static_cast<WorldCountRep>(r)});
    if (!idx) {
        assert(false && "rth_set_bit_index: r out of range (count_ desynced from bits?)");
        std::cerr << "chocofarm: FATAL invariant: rth_set_bit_index: r out of range\n";
        std::abort();
    }
    return static_cast<int>(idx->value());
}

}  // namespace

// C(N,K) bitmask world-set in itertools.combinations order (mirrors instance.world_array). Bit t
// set <=> treasure t present. The "next combination" walk reproduces combinations(range(N), K)
// element order exactly.
static std::vector<uint32_t> build_worlds(int N, int K) {
    std::vector<uint32_t> out;
    if (K < 0 || K > N) return out;
    std::vector<int> c(K);
    std::iota(c.begin(), c.end(), 0);  // GENERATOR-FED seed 0..K-1, counter-free (cold; the raw-int "next combination" kernel below is unchanged)
    while (true) {
        uint32_t mask = 0;
        for (int t : c) mask |= (uint32_t{1} << t);
        out.push_back(mask);
        // advance to the next combination (lexicographic, itertools order)
        int i = K - 1;
        while (i >= 0 && c[i] == N - K + i) --i;
        if (i < 0) break;
        ++c[i];
        for (int j = i + 1; j < K; ++j) c[j] = c[j - 1] + 1;
    }
    return out;
}

Environment::Environment(const Instance& inst) : inst_(inst) {
    // build_worlds speaks raw int (the combination walk's loop arithmetic); N()/K() are the env's named
    // raw-int ACL accessors over the typed inst_.N/inst_.K count domains (env.hpp; ADR-0000 item 5).
    worlds_ = build_worlds(N(), K());
    // contiguous per-detector cover bitmasks (face_masks()): hoist faces[j].bitmask out of the
    // array-of-structs into a packed uint32_t[nD] so the belief sweep reads them without the AoS
    // stride (ADR-0012 P1 — one contiguous home, env still owns them). Order = face id (== faces order).
    face_masks_.reserve(inst_.faces.size());
    for (const Face& f : inst_.faces) face_masks_.push_back(f.bitmask);
    // resolve the entry teleport index. The std::vector index k is the raw container-position boundary;
    // wrap it into the TeleportId domain at this seam (ADR-0000 item 5).
    entry_idx_ = TeleportId{0};
    for (size_t k = 0; k < inst_.teleport_names.size(); ++k) {
        if (inst_.teleport_names[k] == inst_.entry) { entry_idx_ = TeleportId{static_cast<GeometryIdRep>(k)}; break; }
    }

    // ---- the bitset-arm gate + env-static masks (§4; built ONCE here, homed like face_masks_, P1) ----
    // The gate has TWO inputs (§4): a DERIVED quantity (the mask-storage bytes, a pure function of N/nD/kW64)
    // and a MACHINE CONSTANT (kTargetMaskCacheBudgetBytes — the L2-residency budget). kW64 is DERIVED from
    // the world count, NEVER the literal 243.
    const std::size_t nworlds = worlds_.size();
    const bool worlds_enumerable = nworlds > 0;  // build_worlds returns the full C(N,K) set; empty only if K out of range
    kW64_ = static_cast<int>((nworlds + 63) / 64);
    const std::size_t mask_bytes =
        static_cast<std::size_t>(N() + n_detectors()) * static_cast<std::size_t>(kW64_) * sizeof(uint64_t);
    // The gate's THIRD conjunct (the inline-buffer fit): the BitsetBelief now holds its words in a
    // FIXED-CAPACITY inline std::array<.., kBitsetMaxWords> (env.hpp — the per-copy heap alloc this kills),
    // so a belief with kW64 > kBitsetMaxWords CANNOT be represented in the bitset arm and MUST fall to the
    // flat arm. Conjoin it here (and print it in the GATE: line) so the inline cap is an EXPLICIT gate
    // input, not a silent overflow (ADR-0002).
    const bool fits_inline = kW64_ <= kBitsetMaxWords;
    use_bitset_ = worlds_enumerable && (mask_bytes <= kTargetMaskCacheBudgetBytes) && fits_inline;

#ifdef CHOCO_BELIEF_ZDD
    // The §B.4(b) gate (OPT-IN build only): SELECT the maintained ZDD arm over bitset/flat whenever the
    // worlds are enumerable (the diagram builds from worlds()). This is the WHOLE selection surface — the
    // search's full_belief() then returns a ZddBelief and every seam op routes to the ZDD bodies. The
    // default build never compiles this (use_zdd_ does not exist there), so the gate is byte-for-byte the
    // current bitset/flat decision OFF. The bitset masks are still built below (use_bitset_ unchanged) so
    // the flat-vs-ZDD A/B can build a bitset arm for the same belief; full_belief prefers ZDD when on.
    use_zdd_ = worlds_enumerable;
#endif

    if (use_bitset_) {
        // treasure_mask_[t] = worlds (by rank) with bit t set; detector_mask_[j] = worlds where
        // (w & face_masks()[j]) != 0 — the SAME enumeration + face masks the env owns, so the masks
        // derive from worlds_/face_masks_ (the identity env.observe rests on; the oracle pins it).
        const size_t kw = static_cast<size_t>(kW64_);
        treasure_mask_.assign(static_cast<size_t>(N()) * kw, 0ull);
        detector_mask_.assign(static_cast<size_t>(n_detectors()) * kw, 0ull);
        for (size_t r = 0; r < nworlds; ++r) {
            const uint32_t w = worlds_[r];
            const size_t word = r >> 6;           // r / 64
            const uint64_t bit = uint64_t{1} << (r & 63u);  // 1 << (r % 64)
            for (int t = 0; t < N(); ++t)
                if ((w >> t) & 1u) treasure_mask_[static_cast<size_t>(t) * kw + word] |= bit;
            for (int j = 0; j < n_detectors(); ++j)
                if ((w & face_masks_[static_cast<size_t>(j)]) != 0) detector_mask_[static_cast<size_t>(j) * kw + word] |= bit;
        }
    }
}

// ---- belief construction + introspection (the seam ops; COARSE visit per op, §3) ----

Belief Environment::full_belief() const {
#ifdef CHOCO_BELIEF_ZDD
    // The §B.4(b) gate selected the ZDD arm: build the diagram over EVERY world (the C(N,K) prior). The
    // search then maintains it through filtering as a ZDD restrict op (no rebuild). Checked FIRST so the
    // ON build runs the head-to-head ZDD profile; the bitset/flat arms below are untouched (the A/B still
    // builds them directly for the same belief).
    if (use_zdd_) return zdd::full_belief(*this);
#endif
    if (use_bitset_) {
        // all-ones over the nb worlds: the first kw64_ words = ~0, the last (partial) word masked to the live
        // bit-tail so trailing bits past |worlds| stay 0 (a popcount over them must NOT count phantom worlds).
        // The inline array is zero-initialized ({}), so the unused tail words [kw64_, kBitsetMaxWords) STAY 0
        // (the tail-zero invariant operator== relies on); we write ONLY the live words [0, kw64_).
        BitsetBelief b;  // bits{} zero-initialized; tail words past kw64_ stay 0
        b.kw64_ = kW64_;
        std::fill_n(b.bits.begin(), kW64_, ~uint64_t{0});  // GENERATOR-FED all-ones over the live words [0,kW64_), counter-free (tail-zero invariant preserved)
        const size_t nworlds = worlds_.size();
        const int tail = static_cast<int>(nworlds & 63u);  // live bits in the final word (0 => the word is full)
        if (tail != 0) b.bits[static_cast<size_t>(kW64_ - 1)] = (uint64_t{1} << tail) - 1;
        b.count_ = static_cast<int>(nworlds);  // = popcount(all-ones over nworlds bits), the C(N,K) prior
        assert_tail_zero(b);  // debug-only: full_belief writes only [0,kW64_); the tail stays zero-init
        return b;
    }
    return FlatBelief{worlds_};  // the flat arm copies worlds_ (rank order = combinations order)
}

uint32_t Environment::world_at_rank(const Belief& b, int r) const {
    return std::visit([&](const auto& a) -> uint32_t {
        using T = std::decay_t<decltype(a)>;
        if constexpr (std::is_same_v<T, FlatBelief>) return a.worlds[static_cast<size_t>(r)];
        else if constexpr (std::is_same_v<T, BitsetBelief>) return worlds_[static_cast<size_t>(rank_or_abort(a.live(), r))];  // r-th set bit -> world
        CHOCO_ZDD_ELSE(return zdd::world_at_rank(a, r);)  // ZDD arm: r-th member in CANONICAL order (the re-baseline)
    }, b);
}

uint32_t Environment::sample_world(const Belief& b, std::mt19937_64& rng) const {
    // The IDENTICAL uniform draw on both arms (so the RNG stream is byte-identical): r in [0, nb-1], then
    // unrank via world_at_rank. The flat arm's nb is .worlds.size(); the bitset arm's is count_.
    const int n = nb(b);
    // The uniform upper bound is the LAST valid world rank — the named count->rank crossing last_rank()
    // (world.hpp), not an ad-hoc `n - 1` int subtraction (ADR-0000 item 5). `n` is the public int nb (the
    // env's cross-file API stays raw); it is wrapped into WorldCount here and unwrapped through WorldRank.
    const WorldRank hi = last_rank(WorldCount{static_cast<WorldCountRep>(n)});
    std::uniform_int_distribution<size_t> pick(0, static_cast<size_t>(hi.value()));
    const int r = static_cast<int>(pick(rng));
    return world_at_rank(b, r);
}

BeliefKey Environment::belief_key(const Belief& b) const {
    const int n = nb(b);
    if (n == 0) return BeliefKey{0, 0u, 0u};
    // (count, first, last) — bit-identical across arms (shared RANK order, NOT numeric: flat front/back are
    // the rank-0/rank-(count-1) worlds). Flat reads front/back directly; bitset unranks rank 0 and count_-1.
    return std::visit([&](const auto& a) -> BeliefKey {
        using T = std::decay_t<decltype(a)>;
        if constexpr (std::is_same_v<T, FlatBelief>)
            return BeliefKey{n, a.worlds.front(), a.worlds.back()};
        else if constexpr (std::is_same_v<T, BitsetBelief>)
            return BeliefKey{n, worlds_[static_cast<size_t>(rank_or_abort(a.live(), 0))],
                                worlds_[static_cast<size_t>(rank_or_abort(a.live(), n - 1))]};
        // ZDD arm: (count, first, last) in the ZDD's CANONICAL member order — a valid, deterministic
        // fingerprint (the cache verifies hits by full ZddBelief::operator== regardless), but NOT the flat
        // rank-order triple (the re-baseline: the same belief fingerprints differently under the ZDD arm).
        CHOCO_ZDD_ELSE(return BeliefKey{n, zdd::world_at_rank(a, 0), zdd::world_at_rank(a, n - 1)};)
    }, b);
}

double Environment::dist(const Point& a, const Point& b) const {
    return std::hypot(a.x - b.x, a.y - b.y);  // mirrors env.d / math.hypot
}

double Environment::exit_cost(const Point& loc) const {
    double best = -1.0;
    for (const Point& w : inst_.teleports) {
        double d = dist(loc, w);
        if (best < 0.0 || d < best) best = d;
    }
    return best + inst_.teleport_overhead;  // min teleport dist + tp (env.exit_cost)
}

std::vector<double> Environment::marginals(const Belief& bw) const {
    // COARSE visit (§3): one dispatch, then a pure rep-specific body. Both produce byte-identical marg —
    // exact integer counts (Σ_w bit_t / popcount_and) times the SAME `* inv` (1/nb), and (double)count is
    // exact for count <= |worlds| (P6).
    return std::visit([&](const auto& a) -> std::vector<double> {
        using T = std::decay_t<decltype(a)>;
        std::vector<double> m(static_cast<size_t>(N()), 0.0);
        if constexpr (std::is_same_v<T, FlatBelief>) {
            if (a.worlds.empty()) return m;
            for (uint32_t w : a.worlds)
                for (int t = 0; t < N(); ++t)
                    if ((w >> t) & 1u) m[static_cast<size_t>(t)] += 1.0;
            const double inv = 1.0 / static_cast<double>(a.worlds.size());
            for (double& v : m) v *= inv;  // mean over the world-set (mirrors env.marginals)
            return m;
        } else if constexpr (std::is_same_v<T, BitsetBelief>) {
            if (a.count_ == 0) return m;
            const double inv = 1.0 / static_cast<double>(a.count_);
            for (int t = 0; t < N(); ++t)
                m[static_cast<size_t>(t)] = static_cast<double>(popcount_and(a.live(), treasure_mask(t)).value()) * inv;
            return m;
        }
        CHOCO_ZDD_ELSE(return zdd::marginals(*this, a);)  // ZDD arm: all_marginals * inv — byte-identical
    }, bw);
}

bool Environment::informative(int face_id, const Belief& bw) const {
    // Outcome still uncertain over the belief — both polarities live (SenseAction.informative). COARSE
    // visit. Bitset: 0 < popcount_and(detector_mask[j]) < count_ (the cover count strictly between empty
    // and full ⇔ a hit AND a miss both exist) — byte-identical to the flat two-polarity scan.
    return std::visit([&](const auto& a) -> bool {
        using T = std::decay_t<decltype(a)>;
        if constexpr (std::is_same_v<T, FlatBelief>) {
            const uint32_t bm = inst_.faces[static_cast<size_t>(face_id)].bitmask;
            bool any_hit = false, any_miss = false;
            for (uint32_t w : a.worlds) {
                if ((w & bm) != 0) any_hit = true; else any_miss = true;
                if (any_hit && any_miss) return true;  // both polarities live
            }
            return false;  // mirrors SenseAction.informative: hit.any() and (~hit).any()
        } else if constexpr (std::is_same_v<T, BitsetBelief>) {
            const int cnt = static_cast<int>(popcount_and(a.live(), detector_mask(face_id)).value());
            return cnt > 0 && cnt < a.count_;
        }
        CHOCO_ZDD_ELSE(return zdd::informative(*this, face_id, a);)  // ZDD arm: 0 < det_cnt < nb — byte-identical
    }, bw);
}

std::vector<Action> Environment::legal_actions(const Belief& bw,
                                               const CollectedSet& collected) const {
    std::vector<Action> acts;
    std::vector<double> marg = marginals(bw);
    // collects: marg>0 and not collected, in treasure-id order (env iterates _treasure_ids = range(N))
    for (int i = 0; i < N(); ++i) {
        // collected.contains takes a typed TreasureId; Action.i / the loop counter stay raw int (the env's
        // documented cross-file ACL boundary). Wrap at this crossing (ADR-0000 item 5).
        if (!collected.contains(TreasureId{static_cast<TreasureRep>(i)}) && marg[i] > 0.0)
            acts.push_back(Action{ActionKind::Treasure, i});
    }
    // senses: each informative face, in face-id order (env iterates self.detectors = range(nD))
    for (int j = 0; j < n_detectors(); ++j) {
        if (informative(j, bw)) acts.push_back(Action{ActionKind::Detector, j});
    }
    return acts;  // TERMINATE NOT included here (it is the always-legal extra slot)
}

// The one-owner belief compaction (env.hpp): keep worlds where ((w & mask) != 0) == want, in order.
// filter_treasure / filter_detector are thin wrappers differing ONLY by the mask — that unification (one
// compaction, two predicates) is the real win of this refactor (ADR-0012 P1/P3).
//
// Body: the IDIOMATIC std::erase_if (the C++20 erase-remove). A hand-written BRANCHLESS stream-compaction
// (unconditional store + advance-by-keep) was tried and MEASURED on realistic observation-filtered beliefs
// (belief_filter_bench): it ran ~1.4-1.5x SLOWER across every belief size on this i5-6600 native build
// (speedup 0.65-0.93x). Two reasons: the belief predicate predicts well in practice (it is structured, not
// random 50/50, so the branch the branchless form removes was rarely mispredicting), and under
// -march=native the idiom's predicate scan auto-vectorizes whereas the branchless serial out-pointer
// defeats vectorization. So the measured profile REFUSES the non-idiomatic deviation (ADR-0009/ADR-0011 —
// the same "a measured reason decides" rule, here landing on keep-the-idiom). Bit-exact with the former
// per-method erase(remove_if): erase_if IS erase(remove_if) — same kept set, same order (P6). If the filter
// ever dominates a future profile, the measured SIMD-compress rung (AVX2 vpcompress) drops in behind this
// signature, re-gated by belief_filter_bench against this idiom floor.
std::size_t filter_inplace(std::vector<uint32_t>& bw, uint32_t mask, bool want) {
    std::erase_if(bw, [&](uint32_t w) { return ((w & mask) != 0) != want; });  // drop where reading != want
    return bw.size();
}

void Environment::filter_treasure(Belief& bw, int i, bool present) const {
    // In-place through the visited variant (§6 risk 8). Flat: erase_if on the single-bit mask 1<<i.
    // Bitset: bits &= ±treasure_mask[i], recompute count_ (the one place count_ is written). Byte-identical
    // kept set (same keep-predicate, same rank order — both filters keep worlds in their worlds_ rank).
    std::visit([&](auto& a) {
        using T = std::decay_t<decltype(a)>;
        if constexpr (std::is_same_v<T, FlatBelief>) filter_inplace(a.worlds, uint32_t{1} << i, present);
        else if constexpr (std::is_same_v<T, BitsetBelief>) filter_bits(a, treasure_mask(i), present);
        CHOCO_ZDD_ELSE(zdd::filter_treasure(a, i, present);)  // ZDD arm: restrict_var in place (no rebuild)
    }, bw);
}

void Environment::filter_detector(Belief& bw, int i, bool positive) const {
    // As filter_treasure, but the mask is the detector's cover bitmask (the disjunction). Flat: erase_if on
    // faces[i].bitmask; bitset: bits &= ±detector_mask[i], recompute count_.
    std::visit([&](auto& a) {
        using T = std::decay_t<decltype(a)>;
        if constexpr (std::is_same_v<T, FlatBelief>) filter_inplace(a.worlds, inst_.faces[static_cast<size_t>(i)].bitmask, positive);
        else if constexpr (std::is_same_v<T, BitsetBelief>) filter_bits(a, detector_mask(i), positive);
        CHOCO_ZDD_ELSE(zdd::filter_detector(*this, a, i, positive);)  // ZDD arm: restrict_cover in place
    }, bw);
}

StepResult Environment::apply(Loc& loc, Belief& bw, CollectedSet& collected,
                              const Action& action, uint32_t world) const {
    StepResult res;
    Point target;
    if (action.kind == ActionKind::Treasure) target = inst_.treasures[action.i];
    else if (action.kind == ActionKind::Detector) target = inst_.faces[action.i].rep_point;
    else {
        // ADR-0012 P9: apply is NEVER legitimately called with TERMINATE — every caller (the
        // episode loop, the NMCS search/eval, the dump fixtures) breaks before applying it. Reaching
        // here is an INVARIANT violation (a programmer bug), so it aborts loudly rather than being a
        // recoverable boundary Error. assert covers debug; the abort makes it loud under NDEBUG too.
        assert(false && "Environment::apply called with TERMINATE");
        std::cerr << "chocofarm: FATAL invariant: Environment::apply called with TERMINATE\n";
        std::abort();
    }

    res.dt = dist(loc.pt, target);  // travel cost (env.apply: d(loc, (kind, i)))

    if (action.kind == ActionKind::Treasure) {
        bool pres = ((world >> action.i) & 1u) != 0;
        // reward = value[i] (all unit values on this instance) iff present and not yet collected
        // action.i is the raw-int treasure id (Action stays raw int — the env's cross-file ACL boundary);
        // CollectedSet contains/insert take a typed TreasureId. Wrap once at this crossing (ADR-0000 item 5).
        const TreasureId tid{static_cast<TreasureRep>(action.i)};
        bool fresh = pres && !collected.contains(tid);
        res.reward = fresh ? 1.0 : 0.0;  // env.value[i] = 1.0 on the live instance (unit values)
        if (pres) collected.insert(tid);
        filter_treasure(bw, action.i, pres);
    } else {
        bool pos = observe(action.i, world);  // the face's reading at this world
        filter_detector(bw, action.i, pos);
        res.reward = 0.0;
    }
    loc.pt = target;
    return res;
}

}  // namespace chocofarm
