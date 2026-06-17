// cpp/include/chocofarm/features.hpp
// Purpose: the C++ port of the AZ belief featurization + the action<->slot mask, mirroring
//   chocofarm/az/features.py (the FeatureLayout / FeatureBuilder) and chocofarm/az/actions.py
//   (n_action_slots / the legal mask). EVERY dimension is DERIVED from the env (ADR-0012 P1) —
//   feat_dim = 5N + 3nD + 6 + n_tel (= 241 on the live env); n_slots = N + nD + 1 (= 65). Nothing
//   is hardcoded.
//
//   The feature vector is float-sensitive (held to the ADR-0012 P6 behavioral bar). The LEGAL MASK
//   is a logic invariant (ADR-0012 P6/P7): it is bit-identical to Python's for the same
//   (loc, belief) — illegal-slot mass == 0.0 exactly — so the parity harness asserts it bit-exact.
//
//   FeatureBuilder memoizes its two pure sub-computations (belief intermediates by belief VALUE,
//   per-loc distances by Point) — the behaviour-preserving cross-leaf memo (ADR-0012 P4 derived data;
//   a hit is P6 bit-identical to a recompute), porting Python's _belief_cache / _loc_cache.
//
// Public Domain (The Unlicense).
#pragma once

#include <bit>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <set>
#include <span>
#include <unordered_map>
#include <utility>
#include <vector>

#include "chocofarm/belief_key.hpp"
#include "chocofarm/env.hpp"
#include "chocofarm/feature_layout.hpp"

namespace chocofarm {

// Fixed action-space size for this env (mirrors actions.n_action_slots): N collects + nD senses +
// 1 TERMINATE. Slot 0..N-1 = ("t", i); N..N+nD-1 = ("d", j); N+nD = TERMINATE (always legal).
int n_action_slots(const Environment& env);
int term_slot(const Environment& env);          // index of the always-legal TERMINATE slot

// The fixed slot for an action (mirrors actions.action_to_slot).
int action_to_slot(const Environment& env, const Action& a);

// The legal-action mask over the fixed slots (mirrors actions.legal_mask): 1.0 on each legal action
// slot + the always-legal TERMINATE slot, 0.0 elsewhere. This is the LOGIC INVARIANT M the runner
// emits — bit-identical to Python's by construction (the same env.legal_actions set mapped onto the
// same slot bijection). Returned as float (the wire dtype) so the comparison is exact. This is the
// non-hot ORACLE: the per-step training-mask emission (runner.cpp) + the parity tool use it; the hot
// search path uses FeatureBuilder::legal_mask_from_features, which the parity harness nets against THIS.
std::vector<float> legal_mask(const Environment& env, const std::vector<uint32_t>& bw,
                              const std::set<int>& collected);

// --- the featurizer's internal value types (the memo's stored shapes; the pure compute functions in
// features.cpp return them, the FeatureBuilder caches below hold them) ---

// The belief-derived intermediates — a PURE function of the world-set `bw`. marg[i] = mean over bw of
// bit i; per detector p_pos = cover-count/nb, informative = (0 < cover-count < nb). The O(nb·(N+nD))
// sweep that produces these is the profile bottleneck (the ~40% feature bucket the belief memo amortizes).
struct BeliefFeatures {
    std::vector<double> marg;         // N  — per-treasure marginal P(present)
    std::vector<double> p_pos;        // nD — detector positive-cover probability
    std::vector<double> informative;  // nD — detector splits the belief (0/1)
    double marg_sum = 0.0;            // Σ marg[t]  (order-fixed — a P6 watch item; do not reorder)
    double sharpness = 0.0;           // log|bw| / log Nworlds
    double nonempty = 0.0;            // nb ? 1.0 : 0.0
};

// The per-loc static distance block — a PURE function of `loc` (geometry is fully separable). Each
// distance normalized by the bbox diagonal. The loc-set is the env's fixed coordinate keys, so the
// per-loc memo below is bounded by that set.
struct GeometryFeatures {
    std::vector<double> dist_t;  // N
    std::vector<double> dist_d;  // nD
    std::vector<double> dist_w;  // n_tel
    double exit_norm = 0.0;
};

// Exact-bit hash/eq for the per-loc memo key. The loc is ALWAYS a named env coordinate (a fixed float
// from instance.json), so exact-bit keying never conflates distinct coordinates (correctness rests on
// PointEq; unordered_map resolves hash collisions by ==). bit_cast EACH double separately — a bit_cast
// over the whole 16-byte Point would not compile. NEVER an epsilon compare (it would conflate distinct
// fixed coordinates).
struct PointHash {
    [[nodiscard]] std::size_t operator()(const Point& p) const noexcept {
        const std::uint64_t hx = std::bit_cast<std::uint64_t>(p.x);
        const std::uint64_t hy = std::bit_cast<std::uint64_t>(p.y);
        return std::hash<std::uint64_t>{}(hx) ^ (std::hash<std::uint64_t>{}(hy) * 0x9e3779b97f4a7c15ULL);
    }
};
struct PointEq {
    [[nodiscard]] bool operator()(const Point& a, const Point& b) const noexcept {
        return std::bit_cast<std::uint64_t>(a.x) == std::bit_cast<std::uint64_t>(b.x)
            && std::bit_cast<std::uint64_t>(a.y) == std::bit_cast<std::uint64_t>(b.y);
    }
};

// The §2.2 feature vector for (loc, bw, collected), mirroring features.FeatureBuilder.build. The
// layout is the canonical ordered block table (per-treasure N×5, per-detector nD×3, global
// 6+n_tel). Returned as float64 internally, cast to float32 at the wire by the runner. Float-
// sensitive (ADR-0012 P6 behavioral bar), not asserted bit-exact.
class FeatureBuilder {
  public:
    explicit FeatureBuilder(const Environment& env);
    int dim() const { return dim_; }
    // `loc` is the current standing point (resolved Point, as in env.coord). `bw`/`collected` are
    // the live belief + collected set. Returns a length-`dim()` float64 vector.
    std::vector<double> build(const Point& loc, const std::vector<uint32_t>& bw,
                              const std::set<int>& collected) const;

    // The legal-action mask sliced from an ALREADY-BUILT feature vector (mirrors
    // actions.legal_mask_from_features): the per-treasure `available` block IS the collect-legal mask,
    // the per-detector `informative` block IS the sense-legal mask, TERMINATE always legal. The hot-path
    // mask — it REUSES build()'s belief sweep instead of recomputing it via env.legal_actions →
    // marginals (ADR-0012 P1: marg has ONE home, build's; the mask consumes it). `feat` MUST be a
    // length-dim() vector THIS builder's build() returned, in THIS layout — a coupling the bare span
    // cannot express, so it is asserted in the impl (fail-loud, ADR-0002). Bit-identical to the free
    // legal_mask(env, bw, collected) ORACLE for the same state: available == (marg>0 ∧ ¬collected) ==
    // the collect test; informative == (0<cnt<nb) == env.informative (the parity harness nets the two).
    [[nodiscard]] std::vector<float> legal_mask_from_features(std::span<const float> feat) const;

    // Drop the per-belief memo (mirrors Python reset_belief_cache). The search calls this at the start
    // of each decision (run_search), scoping the belief cache to one tree's beliefs — which narrow across
    // decisions, so cross-decision reuse is low — and keeping the long-lived serve-path builder from
    // accumulating a process-lifetime cache (so it is never a never-reset cache). The per-loc memo is NOT
    // reset (bounded by the env's fixed coordinate set). Correctness never depends on either cache (a hit
    // returns bit-identical bytes), so clearing is always sound.
    void reset_belief_cache() const;

  private:
    const Environment& env_;
    int N_;
    int nD_;
    int n_tel_;
    int dim_;
    double diag_;          // bounding-box diagonal over all coords (map_diag)
    double log_nworlds_;   // log(|worlds|)
    FeatureLayoutSpec layout_;  // the §2.2 block table, runtime-read from the Python-emitted SSOT

    // The §2.2 block start offsets, resolved ONCE from layout_ in the ctor (derived data, like dim_ /
    // diag_): build() and legal_mask_from_features read these ints directly instead of re-doing a
    // string_view -> offset hash lookup (with a std::string construction) per named block per call —
    // FeatureLayoutSpec::start was 2.3% self-time in the K=32 profile. The layout SSOT still OWNS the
    // offsets (ADR-0012 P1); this caches the resolved values, bit-identical to the lookups. Fields are
    // the written-key set, in declaration order.
    struct BlockOffsets {
        int marg, collected, available, dist_t, unc, informative, p_pos, dist_d,
            sharpness, n_collected, marg_sum, exit_norm, nonempty, sum_unc, dist_w;
    };
    BlockOffsets off_{};

    // ---- behaviour-preserving memos (derived data; a hit is bit-identical to a recompute, P6). `mutable`
    // so build() stays `const` (logical-const: the observable value-for-input is invariant). SINGLE-
    // THREAD-OWNED: every consumer holds its OWN FeatureBuilder (a fresh one per task in the runtimes; the
    // serve builder is touched only by the single serve thread), so the mutation is race-free by ownership
    // — a future restructure that SHARES one builder across threads owes the synchronization analysis at
    // that sharing site. ----
    //
    // Belief memo: belief_key fingerprint (the SAME authority gumbel's node cache uses — no second
    // fingerprint) -> a bucket of (owned bw copy, features); a hit walks the bucket verifying FULL
    // bw-equality (the fingerprint is collision-resistant, not -free), so a collision never returns
    // another belief's features. The bucket owns a COPY of bw (a stored span would dangle); paid per miss
    // only. The cap is a memory backstop mirroring Python's _belief_cache_cap.
    static constexpr int kBeliefCacheCap = 50000;
    mutable std::map<BeliefKey, std::vector<std::pair<std::vector<uint32_t>, BeliefFeatures>>> belief_cache_;
    mutable int belief_cache_n_ = 0;
    mutable std::unordered_map<Point, GeometryFeatures, PointHash, PointEq> loc_cache_;

    // Thin memo wrappers: look up, compute-on-miss via the private pure functions, store, return a ref.
    [[nodiscard]] const BeliefFeatures& belief_feats_(const std::vector<uint32_t>& bw) const;
    [[nodiscard]] const GeometryFeatures& geometry_feats_(const Point& loc) const;
};

}  // namespace chocofarm
