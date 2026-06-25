// cpp/src/features.cpp
// Purpose: the C++ port of the §2.2 featurization + the action<->slot mask (see features.hpp).
//   Mirrors chocofarm/az/features.py (FeatureLayout block order + FeatureBuilder.build) and
//   chocofarm/az/actions.py (n_action_slots + legal_mask). The mask is the LOGIC INVARIANT M
//   (bit-exact, ADR-0012 P6/P7); the feature vector is float-sensitive (behavioral bar).
//
//   build() is DECOMPOSED (docs/design refactor step 1) along the three input groups the dossier's
//   DAG identifies — belief math (the O(nb·(N+nD)) bottleneck), separable geometry, the collected
//   indicator — into three pure value-functions plus a thin assembler. The decomposition is hygiene
//   only (ADR-0012 P3: one axis per function; P9-rule-2: returned by value): the geometry / collected /
//   assembler ops and their order are UNCHANGED, bit-identical to the former monolith (P6). `available`
//   and `sum_unc` are the only belief×collected couplings, so they live in the assembler, which keeps
//   belief_features a pure function of `bw` — the unit later hoisted to a single SSOT the legal mask
//   shares (legal_mask_from_features slices build()'s sweep, not a recompute).
//
//   belief_features (the belief-math unit) was SUBSEQUENTLY rewritten to the §A.4 form
//   (belief_features_and_decision_diagram_note.md): one fused branchless integer sweep over contiguous
//   masks (env.face_masks()), then a pointwise Phase 2 normalizing both marg AND p_pos via `* inv`. The
//   `* inv` for p_pos is a deliberate behavioral RE-BASELINE (was `/ nb`), so the belief block is NO
//   LONGER byte-identical to the pre-rebaseline monolith — it sits at the P6 behavioral bar vs Python,
//   while the legal mask it feeds (informative / available) stays bit-exact. The bit-exact oracle
//   (belief_sweep_oracle_check) pins the rewrite against an independent naive count; see below.
//
// Public Domain (The Unlicense).
#include "chocofarm/features.hpp"
#include "chocofarm/feature_compute.hpp"
#include "chocofarm/belief_bitset_ops.hpp"  // popcount_and (the ONE home, P1 — shared with env.cpp's seam)

#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <span>
#include <string>
#include <string_view>

namespace chocofarm {

// env.N()/env.n_detectors() return raw int (env.hpp out of scope here); the explicit Treasure|FaceCount
// ctor is the .size()/cardinality ACL (the size_t/int->count crossing). The slot arithmetic then flows
// through the NAMED domain bridges (domains.hpp): n_action_slots / term_slot / slot_of_treasure /
// slot_of_face — never an ad-hoc raw N + j sum at a call site (ADR-0000 item 5).
SlotCount n_action_slots(const Environment& env) {
    return n_action_slots(TreasureCount{static_cast<TreasureRep>(env.N())},
                          FaceCount{static_cast<GeometryIdRep>(env.n_detectors())});  // mirrors actions.n_action_slots
}
SlotIndex term_slot(const Environment& env) {
    return term_slot(TreasureCount{static_cast<TreasureRep>(env.N())},
                     FaceCount{static_cast<GeometryIdRep>(env.n_detectors())});  // the last slot (always legal)
}

SlotIndex action_to_slot(const Environment& env, const Action& a) {
    // a.i is the overloaded raw int (treasure id OR face id, env.hpp out of scope): the ActionKind
    // disambiguates which typed id it is, and the bridge maps that id to its slot — the phantom makes the
    // "face id used as a treasure slot" offset bug unrepresentable past this boundary (ADR-0000).
    switch (a.kind) {
        case ActionKind::Treasure:
            return slot_of_treasure(TreasureId{static_cast<TreasureRep>(a.i)});       // slot 0..N-1
        case ActionKind::Detector:
            return slot_of_face(FaceId{static_cast<GeometryIdRep>(a.i)},
                                TreasureCount{static_cast<TreasureRep>(env.N())});     // slot N..N+nD-1
        case ActionKind::Terminate:
            return term_slot(env);                                                    // slot N+nD
    }
    // ADR-0012 P9: the enum class is exhaustive above — reaching here means a corrupted ActionKind,
    // an INVARIANT violation (a programmer bug), so it aborts loudly rather than being a boundary Error.
    assert(false && "action_to_slot: unknown action kind");
    std::cerr << "chocofarm: FATAL invariant: action_to_slot: unknown action kind\n";
    std::abort();
}

std::vector<float> legal_mask(const Environment& env, const Belief& bw,
                              const CollectedSet& collected) {
    // The authoritative legality source is env.legal_actions (mirrors actions.legal_mask): map its
    // output onto slots + the always-legal TERMINATE. Logic-exact -> bit-identical to Python's M. The
    // mask is FLOAT; only its INDEXING is a SlotIndex/SlotCount — .value() is the std::vector size/index
    // ACL (the typed count/index -> size_t crossing at the container boundary).
    std::vector<float> m(static_cast<size_t>(n_action_slots(env).value()), 0.0f);
    for (const Action& a : env.legal_actions(bw, collected))
        m[static_cast<size_t>(action_to_slot(env, a).value())] = 1.0f;
    m[static_cast<size_t>(term_slot(env).value())] = 1.0f;  // TERMINATE always legal
    return m;
}

FeatureBuilder::FeatureBuilder(const Environment& env)
    // ACL: env.N()/n_detectors() return raw int (env.hpp out of scope) -> the typed cardinalities via the
    // explicit ctor (the .size()/count crossing). n_tel_/dim_ are set in the body (env reads sequenced there).
    : env_(env), N_(TreasureCount{static_cast<TreasureRep>(env.N())}),
      nD_(FaceCount{static_cast<GeometryIdRep>(env.n_detectors())}),
      n_tel_(TeleportCount{0}), dim_(FeatureDim{0}), diag_(0.0), log_nworlds_(0.0) {
    // map_diag: bbox diagonal over ALL named coords — treasures, face rep_points, teleports
    // (mirrors features.map_diag, which ranges over env.coord.values()). The loop counters are RAW reps
    // of the typed cardinalities (.value()); env.treasure_pt/face_pt/teleport_pt take a raw int (out of
    // scope), so the loop is over the raw range [0, count) — the index is a transient loop variable, not a
    // stored domain magnitude.
    double xmin = 1e300, xmax = -1e300, ymin = 1e300, ymax = -1e300;
    auto acc = [&](const Point& p) {
        xmin = std::min(xmin, p.x); xmax = std::max(xmax, p.x);
        ymin = std::min(ymin, p.y); ymax = std::max(ymax, p.y);
    };
    for (TreasureRep i = 0; i < N_.value(); ++i) acc(env_.treasure_pt(i));
    for (GeometryIdRep j = 0; j < nD_.value(); ++j) acc(env_.face_pt(j));
    n_tel_ = TeleportCount{static_cast<GeometryIdRep>(env_.n_teleports())};  // .size()/count ACL
    for (GeometryIdRep k = 0; k < n_tel_.value(); ++k) acc(env_.teleport_pt(k));
    diag_ = std::hypot(xmax - xmin, ymax - ymin);

    // dim = 5N + 3nD + 6 + n_tel (derived, never hardcoded; mirrors feature_dim). The formula mixes three
    // DISTINCT count domains by their literal coefficients, so it is computed in the shared LayoutRep then
    // crossed ONCE into FeatureDim — the named derive-the-dimension ACL (the dims are all <= low hundreds,
    // LayoutRep covers them; ADR-0000 the width is motivated, the crossing is visible).
    dim_ = FeatureDim{static_cast<LayoutRep>(
        5 * static_cast<LayoutRep>(N_.value()) + 3 * static_cast<LayoutRep>(nD_.value()) +
        LayoutRep{6} + static_cast<LayoutRep>(n_tel_.value()))};
    log_nworlds_ = std::log(static_cast<double>(env_.worlds().size()));

    // Read the §2.2 block table from the cross-language SSOT (ADR-0012 P7): the order + widths are the
    // Python FeatureLayout's, emitted to feature_layout.json and netted by tests/test_feature_layout.py,
    // so assemble() writes by NAMED block instead of re-encoding the layout as a positional `o += N`
    // ladder. Path: CHOCO_FEATURE_LAYOUT (else the shipped default, resolved from the run cwd). The
    // layout is a shipped STRUCTURAL artifact (like the hp schema), so a missing/inconsistent spec is a
    // broken install — a LOUD abort (ADR-0002), not a boundary the search could limp past. The fallible
    // read is the load() factory (P9 rule 5); the abort here is its invariant arm (the binary cannot
    // featurize without its layout).
    const char* env_path = std::getenv("CHOCO_FEATURE_LAYOUT");
    const std::string spec_path = env_path ? std::string(env_path) : "chocofarm/data/feature_layout.json";
    auto spec = FeatureLayoutSpec::load(spec_path, dim_);
    if (!spec) {
        std::cerr << "chocofarm: FATAL: FeatureBuilder: " << spec.error().message
                  << "\n  (set CHOCO_FEATURE_LAYOUT to feature_layout.json, or run from the repo root; "
                     "regenerate via FeatureLayout.spec() — see tests/test_feature_layout.py)\n";
        std::abort();
    }
    layout_ = std::move(*spec);

    // The keys this builder writes MUST equal the spec's blocks (mirrors Python's
    // _WRITTEN_KEYS == layout.slices.keys()): a spec block we never write would leak a zero into the
    // vector; a key we write that the spec lacks aborts in start(). count == |written| AND every written
    // key present ⇒ the two key-sets are equal.
    static constexpr std::array<std::string_view, 15> kWritten = {
        "marg", "collected", "available", "dist_t", "unc",
        "informative", "p_pos", "dist_d",
        "sharpness", "n_collected", "marg_sum", "exit_norm", "nonempty", "sum_unc",
        "dist_w",
    };
    bool keys_ok = layout_.block_count() == static_cast<int>(kWritten.size());
    for (std::string_view k : kWritten) keys_ok = keys_ok && layout_.contains(k);
    if (!keys_ok) {
        std::cerr << "chocofarm: FATAL invariant: FeatureBuilder layout/writer key-set mismatch (spec has "
                  << layout_.block_count() << " blocks; writer expects " << kWritten.size() << ")\n";
        std::abort();
    }

    // Resolve the block start offsets ONCE (the key-set is now verified present): build() and
    // legal_mask_from_features read these instead of a per-block hash lookup (FeatureLayoutSpec::start,
    // 2.3% in the K=32 profile). Bit-identical — the same offsets start() returns. Designated init in
    // declaration order so the compiler nets each field against its key.
    off_ = BlockOffsets{
        .marg = layout_.start("marg"),               .collected = layout_.start("collected"),
        .available = layout_.start("available"),     .dist_t = layout_.start("dist_t"),
        .unc = layout_.start("unc"),                 .informative = layout_.start("informative"),
        .p_pos = layout_.start("p_pos"),             .dist_d = layout_.start("dist_d"),
        .sharpness = layout_.start("sharpness"),     .n_collected = layout_.start("n_collected"),
        .marg_sum = layout_.start("marg_sum"),       .exit_norm = layout_.start("exit_norm"),
        .nonempty = layout_.start("nonempty"),       .sum_unc = layout_.start("sum_unc"),
        .dist_w = layout_.start("dist_w"),
    };
}

// --- belief-derived intermediates: a PURE, env-FREE function of the world-set `bw` + the per-detector
// cover masks (the §2-decomposition unit; the K=16 profile's ~81%). EXPOSED via feature_compute.hpp (its
// single home stays here) for the isolated belief_sweep_bench + the bit-exact oracle. DICHOTOMIZED by
// belief size: the nb==0 empty case (a trivial zero-return) and the nb>=1 HOT path are separate children,
// so the hot path carries NO per-call empty guard and reads with one belief-size invariant (inv = 1/nb).
//
// The nb>=1 child is the §A.4 rewrite (belief_features_and_decision_diagram_note.md). The unifying
// reframe: both Phase-1 outputs are VERTICAL (down-the-worlds) reductions of the nb×bits matrix —
// bit_cnt[t] = Σ_w bit_t(w) (pre-normalized marg) and det_cnt[j] = Σ_w [(w & mask_j) != 0] (the cover
// count). Both are INTEGER and BRANCHLESS (fixed trip count, no data-dependent branch -> the vectorizer /
// a future pos-popcount can take over) and read the masks from a CONTIGUOUS span (env.face_masks(), no
// array-of-structs stride). Phase 2 is pointwise. Both marg and p_pos normalize via `* inv` — the SETTLED
// convention (note decision 1): a deliberate behavioral RE-BASELINE of p_pos (was `/ nb`), so the belief
// block is no longer byte-identical to the pre-rebaseline C++; from here `* inv` over exact integer counts
// IS the reference, and every later rung (SIMD/pos-popcount, the Part B diagram) matches it bit-for-bit.
// Cross-language vs Python stays at the P6 behavioral bar; the legal mask informative feeds stays bit-
// exact. belief_sweep_oracle_check nets this against an independent naive count (ADR-0011, net the rewrite).
namespace {

// nb==0 (the empty belief): no worlds to average/cover, so every derived quantity is 0 — the struct's
// scalar defaults (marg_sum/sharpness/nonempty), the vectors zero-filled. The search guards the empty
// belief upstream (run_search), so this is rarely (if ever) on the hot path; isolating it keeps the
// nb>=1 path free of the empty check.
[[nodiscard]] BeliefFeatures belief_features_empty(TreasureCount N, FaceCount nD) {
    BeliefFeatures bf;
    // .value() at the std::vector size ACL (the typed count -> size_t container-sizing crossing).
    bf.marg.assign(static_cast<size_t>(N.value()), 0.0);
    bf.p_pos.assign(static_cast<size_t>(nD.value()), 0.0);
    bf.informative.assign(static_cast<size_t>(nD.value()), 0.0);
    return bf;  // marg_sum / sharpness / nonempty default to 0.0 — exactly the former nb==0 values
}

// nb>=1 (the HOT path, the §A.4 rewrite). Phase 1: ONE fused sweep = two down-the-worlds integer
// reductions; read each world ONCE and drive both accumulators (note A.3 — do NOT split: splitting
// doubles the traffic over bw, the dominant memory cost). Phase 2: pointwise maps over the two
// column-sums + nb, `* inv` everywhere.
[[nodiscard]] BeliefFeatures belief_features_nonempty(std::span<const uint32_t> bw,
                                                      std::span<const uint32_t> masks,
                                                      TreasureCount N, FaceCount nD, double log_nworlds) {
    // nb = |worlds in the belief| — a WorldCount (the .size() count ACL). >= 1 (dispatcher guarantees it).
    const WorldCount nb{static_cast<WorldCountRep>(bw.size())};
    const size_t Nn = static_cast<size_t>(N.value());     // typed-count -> size_t at the container ACL
    const size_t nDn = static_cast<size_t>(nD.value());
    BeliefFeatures bf;
    bf.marg.assign(Nn, 0.0);
    bf.p_pos.assign(nDn, 0.0);
    bf.informative.assign(nDn, 0.0);
    // Column sums of the bit / masked-hit matrices: each entry is a COUNT of worlds (a WorldCountRep,
    // bounded by nb), so the per-column accumulator is the raw rep of the WorldCount domain. (Kept as a
    // contiguous WorldCountRep array for the branchless fused sweep; wrapped at the phase-2 read.)
    std::vector<WorldCountRep> bit_cnt(Nn, 0);   // marg_raw: column sums of the bit matrix
    std::vector<WorldCountRep> det_cnt(nDn, 0);  // cnt:      column sums of the masked-hit matrix

    // phase 1: ONE fused sweep, both bodies branchless + integer (the O(nb*(N+nD)) cost). t/j are treasure
    // / face bit positions (.value() the id->bit-position crossing); the column index is the same value.
    for (uint32_t w : bw) {
        for (TreasureRep t = 0; t < N.value(); ++t)  bit_cnt[t] += (w >> t) & 1u;
        for (GeometryIdRep j = 0; j < nD.value(); ++j) det_cnt[j] += (w & masks[j]) != 0;
    }

    // phase 2: pointwise maps. marg uses ·inv exactly as before (bit-exact: (double)count is exact for
    // count <= |worlds|); p_pos now ·inv too (the re-baseline). marg_sum accumulates in treasure-id
    // order, unchanged from the former two-pass ladder — a P6 watch item, do not reorder. nb.value() is
    // the WorldCount -> raw crossing for the floating normalizer + the informative split test.
    const double inv = 1.0 / static_cast<double>(nb.value());
    for (size_t t = 0; t < Nn; ++t) {
        bf.marg[t]   = static_cast<double>(bit_cnt[t]) * inv;
        bf.marg_sum += bf.marg[t];
    }
    for (size_t j = 0; j < nDn; ++j) {
        bf.p_pos[j]       = static_cast<double>(det_cnt[j]) * inv;
        bf.informative[j] = (det_cnt[j] > 0 && det_cnt[j] < nb.value()) ? 1.0 : 0.0;
    }
    bf.sharpness = std::log(static_cast<double>(nb.value())) / log_nworlds;
    bf.nonempty  = 1.0;
    return bf;
}

// nb>=1, the BITSET arm (§5 step 5). Phase 1: the SAME two integer column-sums as the §A.4 flat sweep —
// bit_cnt[t] = Σ_w bit_t(w) = popcount(belief & treasure_mask[t]); det_cnt[j] = Σ_w [(w & mask_j)!=0] =
// popcount(belief & detector_mask[j]) — produced by masked-AND + popcount instead of a per-world loop.
// Phase 2 is the IDENTICAL pointwise `* inv` (NEVER `/ nb`; §6 risk 6), and informative via
// det_cnt>0 && det_cnt<(int64_t)nb — so the result is BYTE-IDENTICAL to belief_features_nonempty for the
// same belief (exact integer counts, same inv). `count_` is the cached nb (>= 1 here, the dispatcher
// guarantees it).
[[nodiscard]] BeliefFeatures belief_features_bitset(const Environment& env, const BitsetBelief& b,
                                                    TreasureCount N, FaceCount nD, double log_nworlds) {
    // b.count_ is the cached nb (raw int, env.hpp out of scope) -> WorldCount via the count ACL; >= 1 here
    // (the dispatcher guarantees it). env.treasure_mask(t)/detector_mask(j) take a raw int (out of scope),
    // so the id .value() crosses at that mask-lookup call.
    const WorldCount nb{static_cast<WorldCountRep>(b.count_)};
    const size_t Nn = static_cast<size_t>(N.value());
    const size_t nDn = static_cast<size_t>(nD.value());
    BeliefFeatures bf;
    bf.marg.assign(Nn, 0.0);
    bf.p_pos.assign(nDn, 0.0);
    bf.informative.assign(nDn, 0.0);

    const double inv = 1.0 / static_cast<double>(nb.value());
    const std::span<const uint64_t> live = b.live();  // the live words [0,kw64_), NOT the inline-array cap
    for (TreasureRep t = 0; t < N.value(); ++t) {
        // popcount_and already returns the typed WorldCount (the centralized stdlib-popcount ACL); .value()
        // is its read into the floating normalizer. bit_cnt = Σ_w bit_t(w).
        const WorldCountRep bit_cnt = popcount_and(live, env.treasure_mask(t)).value();
        bf.marg[t]  = static_cast<double>(bit_cnt) * inv;
        bf.marg_sum += bf.marg[t];                                            // treasure-id order (P6)
    }
    for (GeometryIdRep j = 0; j < nD.value(); ++j) {
        const WorldCountRep det_cnt = popcount_and(live, env.detector_mask(j)).value();  // Σ_w [(w & mask_j)!=0]
        bf.p_pos[j]       = static_cast<double>(det_cnt) * inv;
        bf.informative[j] = (det_cnt > 0 && det_cnt < nb.value()) ? 1.0 : 0.0;
    }
    bf.sharpness = std::log(static_cast<double>(nb.value())) / log_nworlds;
    bf.nonempty  = 1.0;
    return bf;
}

}  // namespace

// The public sweep entry (feature_compute.hpp): COARSE visit on the rep (§3), then dispatch on belief size
// to the empty / hot child. The flat arm reads `.worlds` and runs the EXISTING §A.4 sweep UNCHANGED; the
// bitset arm runs the masked-AND + popcount kernel (byte-identical, §6 risk 6). Dims + masks + log|worlds|
// come from the env (the honest signature, P9): N/nD/log_nworlds are env-static; the empty child takes only
// N/nD (both arms share belief_features_empty — the zero return).
BeliefFeatures belief_features(const Environment& env, const Belief& bw) {
    // ACL: env.N()/n_detectors() return raw int (out of scope) -> the typed cardinalities (count crossing).
    const TreasureCount N{static_cast<TreasureRep>(env.N())};
    const FaceCount nD{static_cast<GeometryIdRep>(env.n_detectors())};
    const double log_nworlds = std::log(static_cast<double>(env.worlds().size()));
    return std::visit([&](const auto& a) -> BeliefFeatures {
        using T = std::decay_t<decltype(a)>;
        if constexpr (std::is_same_v<T, FlatBelief>) {
            return a.worlds.empty()
                       ? belief_features_empty(N, nD)
                       : belief_features_nonempty(std::span<const uint32_t>(a.worlds), env.face_masks(),
                                                  N, nD, log_nworlds);
        } else if constexpr (std::is_same_v<T, BitsetBelief>) {
            return a.count_ == 0
                       ? belief_features_empty(N, nD)
                       : belief_features_bitset(env, a, N, nD, log_nworlds);
        }
        // ZDD arm (opt-in): the whole BeliefFeatures off the maintained diagram (all_marginals +
        // all_detector_counts + the IDENTICAL Phase-2 * inv) — byte-identical to the flat/bitset arms.
        CHOCO_ZDD_ELSE(return zdd::belief_features(env, a);)
    }, bw);
}

namespace {

// --- per-loc static distance block: geometry is FULLY separable (one outgoing edge in the DAG). Each
// distance normalized by the bbox diagonal. `hypot` chains TODAY (a precomputed lookup is the deferred
// §5 move). Same ops + order as the former inline block.
[[nodiscard]] GeometryFeatures geometry_features(const Environment& env, const Point& loc,
                                                 TreasureCount N, FaceCount nD, TeleportCount n_tel,
                                                 double diag) {
    GeometryFeatures gf;
    // .value() at the std::vector size/index ACL; env.*_pt take a raw int (out of scope), so the loop
    // runs over the raw rep range [0, count) — the index is a transient loop variable, not a stored domain.
    gf.dist_t.resize(static_cast<size_t>(N.value()));
    gf.dist_d.resize(static_cast<size_t>(nD.value()));
    gf.dist_w.resize(static_cast<size_t>(n_tel.value()));
    for (TreasureRep i = 0; i < N.value(); ++i) gf.dist_t[i] = std::hypot(loc.x - env.treasure_pt(i).x,
                                                                          loc.y - env.treasure_pt(i).y) / diag;
    for (GeometryIdRep j = 0; j < nD.value(); ++j) gf.dist_d[j] = std::hypot(loc.x - env.face_pt(j).x,
                                                                             loc.y - env.face_pt(j).y) / diag;
    for (GeometryIdRep k = 0; k < n_tel.value(); ++k) gf.dist_w[k] = std::hypot(loc.x - env.teleport_pt(k).x,
                                                                                loc.y - env.teleport_pt(k).y) / diag;
    gf.exit_norm = env.exit_cost(loc) / diag;
    return gf;
}

// --- collected-set indicator: one axis. coll[i] = 1 iff treasure i collected.
}  // namespace

std::vector<double> FeatureBuilder::build(const Point& loc, const Belief& bw,
                                          const CollectedSet& collected) const {
    // Thin value-returning wrapper over build_into (P9 rule 2): the non-hot callers (the runner's
    // record-assembly, the parity harnesses, the cache check) keep the convenient by-value form; the
    // body lives ONCE in build_into. Bit-identical — `out` is constructed exactly as the former
    // monolith (`(dim_, 0.0)`), then build_into overwrites it with the SAME writes.
    std::vector<double> out(static_cast<size_t>(dim_.value()), 0.0);  // FeatureDim -> size_t (container ACL)
    build_into(loc, bw, collected, out);
    return out;
}

void FeatureBuilder::build_into(const Point& loc, const Belief& bw, const CollectedSet& collected,
                                std::vector<double>& out) const {
    // Raw block-extent reps of the typed cardinalities (.value()) — the per-block write loops run over the
    // raw range [0, count); the WRITE position is the typed block-start offset + the raw within-block index.
    const TreasureRep N = N_.value();
    const GeometryIdRep nD = nD_.value();
    const GeometryIdRep n_tel = n_tel_.value();

    // Three input groups that do not interact until assembly (the dossier's DAG): belief math (the
    // O(nb·(N+nD)) bottleneck), separable geometry, the collected indicator. Each is a pure unit.
    const BeliefFeatures& bf = belief_feats_(bw);          // memoized by belief VALUE (P6 bit-identical hit)
    const GeometryFeatures& gf = geometry_feats_(loc);     // memoized by loc
    // The collected indicator is written DIRECTLY into out[off_.collected] below (out is pre-zeroed by
    // assign, so non-collected slots stay 0) and read back for the `available`/`sum_unc` couplings;
    // |collected| comes from collected.size(). No per-leaf CollectedFeatures vector (audit #5 — the one
    // per-leaf temporary that slipped past the FeatureWorkspace amortization). Bit-identical (same values).

    // --- assemble by NAMED block: the layout SSOT (audit R6 / ADR-0012 P7) owns the order + offsets; the
    // ctor resolved them once into off_ (no positional `o += N` ladder re-encoding the layout here, and no
    // per-build string_view->offset hash lookup). `available` and `sum_unc` are the only belief×collected
    // couplings, computed in this step. Slots are independent, so the WRITE order is irrelevant; the
    // float-op order (the sum_unc accumulation, the per-i unc/available) is UNCHANGED from the former
    // ladder — bit-exact (P6). The per-build layout-mismatch assert is GONE: the ctor's load (Σwidth==dim
    // ==env-derived dim, no dup/neg widths ⇒ a contiguous partition) + its key-set check make the desync
    // that assert guarded structurally unauthorable (dossier §3 — mechanize > assert).
    // Reuse the caller's buffer: size to dim_ and zero EVERY slot, identical to the former
    // `std::vector<double> out(dim_, 0.0)` construction (the layout is a contiguous Σwidth==dim
    // partition, so every slot is then overwritten below — the zero-init is the build() basis kept
    // exact). `assign` reuses the existing capacity across leaves (no per-leaf alloc on the steady path).
    out.assign(static_cast<size_t>(dim_.value()), 0.0);  // FeatureDim -> size_t (container ACL)
    // Each block-start `s` crosses the FeatureDim offset to size_t ONCE at the flat-buffer indexing ACL;
    // the within-block raw index then adds to it (pointer arithmetic into the contiguous vector). The
    // `coll` start is reused below for the available/sum_unc collected read-back.
    const size_t coll_s = static_cast<size_t>(off_.collected.value());
    { const size_t s = static_cast<size_t>(off_.marg.value());
      for (TreasureRep i = 0; i < N; ++i) out[s + i] = bf.marg[i]; }
    // for_each_ascending now yields a typed TreasureId; .value() is the id -> bit/index crossing.
    collected.for_each_ascending([&](TreasureId i) { out[coll_s + i.value()] = 1.0; });
    { const size_t s = static_cast<size_t>(off_.available.value());
      for (TreasureRep i = 0; i < N; ++i)
          out[s + i] = ((bf.marg[i] > 0.0) && (out[coll_s + i] == 0.0)) ? 1.0 : 0.0; }
    { const size_t s = static_cast<size_t>(off_.dist_t.value());
      for (TreasureRep i = 0; i < N; ++i) out[s + i] = gf.dist_t[i]; }
    double sum_unc = 0.0;
    { const size_t s = static_cast<size_t>(off_.unc.value());
      for (TreasureRep i = 0; i < N; ++i) {
          double u = bf.marg[i] * (1.0 - bf.marg[i]);                     // unc
          out[s + i] = u;
          if (out[coll_s + i] == 0.0) sum_unc += u;  // Σ over UNCOLLECTED treasures (order preserved)
      } }
    { const size_t s = static_cast<size_t>(off_.informative.value());
      for (GeometryIdRep j = 0; j < nD; ++j) out[s + j] = bf.informative[j]; }
    { const size_t s = static_cast<size_t>(off_.p_pos.value());
      for (GeometryIdRep j = 0; j < nD; ++j) out[s + j] = bf.p_pos[j]; }
    { const size_t s = static_cast<size_t>(off_.dist_d.value());
      for (GeometryIdRep j = 0; j < nD; ++j) out[s + j] = gf.dist_d[j]; }
    out[static_cast<size_t>(off_.sharpness.value())]   = bf.sharpness;        // log|bw|/log Nworlds
    // n_collected/K and Σmarg/K: both numerator and K are TREASURE-COUNTS (collected.size() is a
    // CollectedCount, env.K() the PresentCount K via the count ACL); .value() at the floating-divide.
    const PresentCount K{static_cast<TreasureRep>(env_.K())};
    out[static_cast<size_t>(off_.n_collected.value())] =
        static_cast<double>(collected.size().value()) / static_cast<double>(K.value());  // n_collected/K
    out[static_cast<size_t>(off_.marg_sum.value())]    =
        bf.marg_sum / static_cast<double>(K.value());                        // Σmarg/K
    out[static_cast<size_t>(off_.exit_norm.value())]   = gf.exit_norm;        // exit geometry
    out[static_cast<size_t>(off_.nonempty.value())]    = bf.nonempty;         // non-empty belief flag
    out[static_cast<size_t>(off_.sum_unc.value())]     = sum_unc;             // Σ_uncollected unc
    { const size_t s = static_cast<size_t>(off_.dist_w.value());
      for (GeometryIdRep k = 0; k < n_tel; ++k) out[s + k] = gf.dist_w[k]; }
}

std::vector<float> FeatureBuilder::legal_mask_from_features(std::span<const float> feat) const {
    // Thin value-returning wrapper over legal_mask_into (P9 rule 2): the non-hot callers keep the
    // by-value form; the body lives ONCE in legal_mask_into. Bit-identical — `m` is constructed exactly
    // as the former monolith (`(N+nD+1, 0.0f)`), then legal_mask_into overwrites it with the SAME writes.
    // The mask length is the SlotCount (N+nD+1) via the named bridge; .value() at the std::vector size ACL.
    std::vector<float> m(static_cast<size_t>(n_action_slots(N_, nD_).value()), 0.0f);
    legal_mask_into(feat, m);
    return m;
}

void FeatureBuilder::legal_mask_into(std::span<const float> feat, std::vector<float>& out) const {
    // `feat` MUST be this builder's build()/build_into output (length dim()). The bare span cannot carry
    // that contract, so assert it (fail-loud, ADR-0002) — a mis-laid buffer is a programmer bug, not a
    // recoverable boundary. .value() crosses the FeatureDim to size_t for the span-size compare.
    assert(feat.size() == static_cast<size_t>(dim_.value()) && "legal_mask_into: feat is not a build() vector");
    // Slice the §2.2 blocks that ARE the mask (design §3): the per-treasure `available` block is the
    // legal-collect mask, the per-detector `informative` block is the legal-sense mask, TERMINATE is
    // always legal. No belief recompute — these blocks were just written by build() from the ONE marg
    // sweep (ADR-0012 P1). Offsets via the layout SSOT (named, not magic literals). `assign` reuses the
    // caller buffer's capacity (no per-leaf alloc on the steady path) and zeros it, identical to the
    // former `(N+nD+1, 0.0f)` construction — TERMINATE is then set 1.0f, every other slot written below.
    out.assign(static_cast<size_t>(n_action_slots(N_, nD_).value()), 0.0f);  // SlotCount -> size_t (size ACL)
    // feat-block starts cross FeatureDim -> size_t once (the flat-buffer read ACL); the mask WRITE position
    // is the typed slot index from the bijection bridge (slot_of_treasure / slot_of_face / term_slot),
    // .value() at the std::vector index — the offset bug (a detector written at slot j not N+j) is now
    // structurally prevented by the bridge, not a hand-rolled N_+j sum (ADR-0000).
    const size_t avail = static_cast<size_t>(off_.available.value());
    const size_t info = static_cast<size_t>(off_.informative.value());
    for (TreasureRep i = 0; i < N_.value(); ++i)
        out[static_cast<size_t>(slot_of_treasure(TreasureId{i}).value())] =
            (feat[avail + i] > 0.0f) ? 1.0f : 0.0f;
    for (GeometryIdRep j = 0; j < nD_.value(); ++j)
        out[static_cast<size_t>(slot_of_face(FaceId{j}, N_).value())] =
            (feat[info + j] > 0.0f) ? 1.0f : 0.0f;
    out[static_cast<size_t>(term_slot(N_, nD_).value())] = 1.0f;  // TERMINATE always legal (term_slot = N+nD)
}

// The belief-memo capacity: the measured throughput-neutral default, OR an env override
// (CHOCO_BELIEF_CACHE_CAP) for the ADR-0009 validation sweep. Read once per builder (belief_cache_cap_).
// A non-positive / unparseable env value falls back to the default (fail-soft on a tuning knob; the
// search is correct at ANY cap >= 1, so this never affects results, only the recompute/memory trade).
int FeatureBuilder::belief_cache_cap() {
    if (const char* e = std::getenv("CHOCO_BELIEF_CACHE_CAP")) {
        const int v = std::atoi(e);
        if (v > 0) return v;
    }
    return kDefaultBeliefCacheCap;
}

void FeatureBuilder::reset_belief_cache() const {
    belief_cache_.clear();
    belief_fifo_.clear();
    belief_cache_n_ = 0;
    // the per-loc memo is NOT cleared — it is bounded by the env's fixed coordinate set (mirrors Python:
    // reset_belief_cache clears _belief_cache only; _loc_cache persists).
}

// ---- the memo wrappers: a hit returns the STORED value (bit-identical to a recompute, P6); a miss
// computes via the private pure function above, stores, returns the cached ref. ----
const BeliefFeatures& FeatureBuilder::belief_feats_(const Belief& bw) const {
    const BeliefKey key = env_.belief_key(bw);  // the SAME fingerprint gumbel's node cache uses (P1; L2)
    if (auto it = belief_cache_.find(key); it != belief_cache_.end())
        for (const auto& entry : it->second)
            if (entry.first == bw) return entry.second;  // hit — full belief-equality verified (L5)
    // miss: compute, store an OWNED copy of bw (a stored span would dangle), return the cached ref.
    //
    // BOUNDED-RESIDENT EVICTION (ADR-0000 O(fibers)-resident fix; RCA tlab_finding #26, heaptrack-
    // attributed — ADR-0009). This memo was the DOMINANT per-fiber resident term (~75% of peak at the
    // banked config): the former backstop was kBeliefCacheCap=50000 entries cleared WHOLESALE only on
    // overflow, so a fiber mid-decision held EVERY distinct belief it had reached this decision — an owned
    // BitsetBelief (a fixed std::array<u64,256> = 2 KiB) + its BeliefFeatures, ~113 avg / ~357 max at
    // n_sims=256 (MEASURED). With the driver parking ALL K fibers mid-decision across the RTT, resident =
    // threads*K*memo-high-water -> the unbounded-in-fiber-population OOM. The memo is PURELY a within-
    // decision amortization of the O(nb) belief sweep, and gumbel's NODE transposition table (children,
    // keyed by belief_key) already dedups the repeated-belief reuse, so the memo's marginal hit rate is
    // low; bounding it to a small ring barely moves throughput (MEASURED — see the structural-fix sweep)
    // while making per-fiber resident O(cap), NOT O(tree-size). Correctness is INVARIANT: a hit is bit-
    // identical to a recompute and a miss recomputes (P6), so eviction only trades a recompute for memory.
    // FIFO eviction (insertion order ~= recency under the search's depth-first revisit pattern): on cap,
    // drop the OLDEST single entry, not the whole cache (whole-clear at a small cap would thrash). SAFETY:
    // belief_feats_ returns a ref into the bucket consumed by the caller BEFORE the next belief_feats_
    // call (build_into reads bf then only touches loc_cache_), so an eviction on the NEXT miss never
    // invalidates a live ref (the lifetime contract the former whole-clear already relied on).
    if (belief_cache_n_ >= belief_cache_cap_) evict_oldest_belief_();
    BeliefFeatures feats = belief_features(env_, bw);  // visits the rep (flat sweep / bitset popcount)
    auto& bucket = belief_cache_[key];
    bucket.emplace_back(bw, std::move(feats));
    belief_fifo_.push_back(key);  // record insertion order for FIFO eviction
    ++belief_cache_n_;
    return bucket.back().second;
}

// Evict the oldest cached belief entry (FIFO): pop the front insertion key, drop one entry from its
// bucket (the front-most surviving entry of that key — bucket order is insertion order within the key),
// and erase the bucket if it empties. Bounds per-fiber resident to kBeliefCacheCap entries. The dropped
// entry is never a live reference (see belief_feats_ SAFETY note). O(1) amortized.
void FeatureBuilder::evict_oldest_belief_() const {
    while (!belief_fifo_.empty()) {
        const BeliefKey key = belief_fifo_.front();
        belief_fifo_.pop_front();
        auto it = belief_cache_.find(key);
        if (it == belief_cache_.end()) continue;       // bucket already gone (key re-evicted) — skip
        auto& bucket = it->second;
        if (bucket.empty()) { belief_cache_.erase(it); continue; }
        bucket.erase(bucket.begin());                  // drop the oldest entry of this key
        if (bucket.empty()) belief_cache_.erase(it);
        --belief_cache_n_;
        return;                                        // evicted exactly one
    }
}

const GeometryFeatures& FeatureBuilder::geometry_feats_(const Point& loc) const {
    if (auto it = loc_cache_.find(loc); it != loc_cache_.end()) return it->second;  // hit
    GeometryFeatures feats = geometry_features(env_, loc, N_, nD_, n_tel_, diag_);
    return loc_cache_.emplace(loc, std::move(feats)).first->second;
}

}  // namespace chocofarm
