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
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <span>
#include <string>
#include <string_view>

namespace chocofarm {

int n_action_slots(const Environment& env) {
    return env.N() + env.n_detectors() + 1;  // mirrors actions.n_action_slots
}
int term_slot(const Environment& env) {
    return env.N() + env.n_detectors();      // the last slot (always legal)
}

int action_to_slot(const Environment& env, const Action& a) {
    switch (a.kind) {
        case ActionKind::Treasure: return a.i;                 // slot 0..N-1
        case ActionKind::Detector: return env.N() + a.i;       // slot N..N+nD-1
        case ActionKind::Terminate: return term_slot(env);     // slot N+nD
    }
    // ADR-0012 P9: the enum class is exhaustive above — reaching here means a corrupted ActionKind,
    // an INVARIANT violation (a programmer bug), so it aborts loudly rather than being a boundary Error.
    assert(false && "action_to_slot: unknown action kind");
    std::cerr << "chocofarm: FATAL invariant: action_to_slot: unknown action kind\n";
    std::abort();
}

std::vector<float> legal_mask(const Environment& env, const Belief& bw,
                              const std::set<int>& collected) {
    // The authoritative legality source is env.legal_actions (mirrors actions.legal_mask): map its
    // output onto slots + the always-legal TERMINATE. Logic-exact -> bit-identical to Python's M.
    std::vector<float> m(n_action_slots(env), 0.0f);
    for (const Action& a : env.legal_actions(bw, collected)) m[action_to_slot(env, a)] = 1.0f;
    m[term_slot(env)] = 1.0f;  // TERMINATE always legal
    return m;
}

FeatureBuilder::FeatureBuilder(const Environment& env)
    : env_(env), N_(env.N()), nD_(env.n_detectors()),
      n_tel_(0), dim_(0), diag_(0.0), log_nworlds_(0.0) {
    // map_diag: bbox diagonal over ALL named coords — treasures, face rep_points, teleports
    // (mirrors features.map_diag, which ranges over env.coord.values()).
    double xmin = 1e300, xmax = -1e300, ymin = 1e300, ymax = -1e300;
    auto acc = [&](const Point& p) {
        xmin = std::min(xmin, p.x); xmax = std::max(xmax, p.x);
        ymin = std::min(ymin, p.y); ymax = std::max(ymax, p.y);
    };
    for (int i = 0; i < N_; ++i) acc(env_.treasure_pt(i));
    for (int j = 0; j < nD_; ++j) acc(env_.face_pt(j));
    n_tel_ = env_.n_teleports();
    for (int k = 0; k < n_tel_; ++k) acc(env_.teleport_pt(k));
    diag_ = std::hypot(xmax - xmin, ymax - ymin);

    // dim = 5N + 3nD + 6 + n_tel (derived, never hardcoded; mirrors feature_dim)
    dim_ = 5 * N_ + 3 * nD_ + 6 + n_tel_;
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
[[nodiscard]] BeliefFeatures belief_features_empty(int N, int nD) {
    BeliefFeatures bf;
    bf.marg.assign(N, 0.0);
    bf.p_pos.assign(nD, 0.0);
    bf.informative.assign(nD, 0.0);
    return bf;  // marg_sum / sharpness / nonempty default to 0.0 — exactly the former nb==0 values
}

// nb>=1 (the HOT path, the §A.4 rewrite). Phase 1: ONE fused sweep = two down-the-worlds integer
// reductions; read each world ONCE and drive both accumulators (note A.3 — do NOT split: splitting
// doubles the traffic over bw, the dominant memory cost). Phase 2: pointwise maps over the two
// column-sums + nb, `* inv` everywhere.
[[nodiscard]] BeliefFeatures belief_features_nonempty(std::span<const uint32_t> bw,
                                                      std::span<const uint32_t> masks,
                                                      int N, int nD, double log_nworlds) {
    const size_t nb = bw.size();   // >= 1 (the dispatcher guarantees it)
    BeliefFeatures bf;
    bf.marg.assign(N, 0.0);
    bf.p_pos.assign(nD, 0.0);
    bf.informative.assign(nD, 0.0);
    std::vector<int64_t> bit_cnt(N, 0);   // marg_raw: column sums of the bit matrix
    std::vector<int64_t> det_cnt(nD, 0);  // cnt:      column sums of the masked-hit matrix

    // phase 1: ONE fused sweep, both bodies branchless + integer (the O(nb*(N+nD)) cost).
    for (uint32_t w : bw) {
        for (int t = 0; t < N; ++t)  bit_cnt[t] += (w >> t) & 1u;
        for (int j = 0; j < nD; ++j) det_cnt[j] += (w & masks[j]) != 0;
    }

    // phase 2: pointwise maps. marg uses ·inv exactly as before (bit-exact: (double)count is exact for
    // count <= |worlds|); p_pos now ·inv too (the re-baseline). marg_sum accumulates in treasure-id
    // order, unchanged from the former two-pass ladder — a P6 watch item, do not reorder.
    const double inv = 1.0 / static_cast<double>(nb);
    for (int t = 0; t < N; ++t) {
        bf.marg[t]   = static_cast<double>(bit_cnt[t]) * inv;
        bf.marg_sum += bf.marg[t];
    }
    for (int j = 0; j < nD; ++j) {
        bf.p_pos[j]       = static_cast<double>(det_cnt[j]) * inv;
        bf.informative[j] = (det_cnt[j] > 0 && det_cnt[j] < static_cast<int64_t>(nb)) ? 1.0 : 0.0;
    }
    bf.sharpness = std::log(static_cast<double>(nb)) / log_nworlds;
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
                                                    int N, int nD, double log_nworlds) {
    const size_t nb = static_cast<size_t>(b.count_);  // >= 1 (the dispatcher guarantees it)
    BeliefFeatures bf;
    bf.marg.assign(N, 0.0);
    bf.p_pos.assign(nD, 0.0);
    bf.informative.assign(nD, 0.0);

    const double inv = 1.0 / static_cast<double>(nb);
    for (int t = 0; t < N; ++t) {
        const int64_t bit_cnt = popcount_and(b.bits, env.treasure_mask(t));  // Σ_w bit_t(w)
        bf.marg[static_cast<size_t>(t)]  = static_cast<double>(bit_cnt) * inv;
        bf.marg_sum += bf.marg[static_cast<size_t>(t)];                       // treasure-id order (P6)
    }
    for (int j = 0; j < nD; ++j) {
        const int64_t det_cnt = popcount_and(b.bits, env.detector_mask(j));   // Σ_w [(w & mask_j)!=0]
        bf.p_pos[static_cast<size_t>(j)]       = static_cast<double>(det_cnt) * inv;
        bf.informative[static_cast<size_t>(j)] = (det_cnt > 0 && det_cnt < static_cast<int64_t>(nb)) ? 1.0 : 0.0;
    }
    bf.sharpness = std::log(static_cast<double>(nb)) / log_nworlds;
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
    const int N = env.N();
    const int nD = env.n_detectors();
    const double log_nworlds = std::log(static_cast<double>(env.worlds().size()));
    return std::visit([&](const auto& a) -> BeliefFeatures {
        using T = std::decay_t<decltype(a)>;
        if constexpr (std::is_same_v<T, FlatBelief>) {
            return a.worlds.empty()
                       ? belief_features_empty(N, nD)
                       : belief_features_nonempty(std::span<const uint32_t>(a.worlds), env.face_masks(),
                                                  N, nD, log_nworlds);
        } else {
            return a.count_ == 0
                       ? belief_features_empty(N, nD)
                       : belief_features_bitset(env, a, N, nD, log_nworlds);
        }
    }, bw);
}

namespace {

// --- per-loc static distance block: geometry is FULLY separable (one outgoing edge in the DAG). Each
// distance normalized by the bbox diagonal. `hypot` chains TODAY (a precomputed lookup is the deferred
// §5 move). Same ops + order as the former inline block.
[[nodiscard]] GeometryFeatures geometry_features(const Environment& env, const Point& loc,
                                                 int N, int nD, int n_tel, double diag) {
    GeometryFeatures gf;
    gf.dist_t.resize(N);
    gf.dist_d.resize(nD);
    gf.dist_w.resize(n_tel);
    for (int i = 0; i < N; ++i) gf.dist_t[i] = std::hypot(loc.x - env.treasure_pt(i).x,
                                                          loc.y - env.treasure_pt(i).y) / diag;
    for (int j = 0; j < nD; ++j) gf.dist_d[j] = std::hypot(loc.x - env.face_pt(j).x,
                                                           loc.y - env.face_pt(j).y) / diag;
    for (int k = 0; k < n_tel; ++k) gf.dist_w[k] = std::hypot(loc.x - env.teleport_pt(k).x,
                                                              loc.y - env.teleport_pt(k).y) / diag;
    gf.exit_norm = env.exit_cost(loc) / diag;
    return gf;
}

// --- collected-set indicator: one axis. coll[i] = 1 iff treasure i collected.
struct CollectedFeatures {
    std::vector<double> coll;     // N indicator
    double n_collected = 0.0;     // |collected|
};

[[nodiscard]] CollectedFeatures collected_features(const std::set<int>& collected, int N) {
    CollectedFeatures cf;
    cf.coll.assign(N, 0.0);
    for (int i : collected) cf.coll[i] = 1.0;
    cf.n_collected = static_cast<double>(collected.size());
    return cf;
}

}  // namespace

std::vector<double> FeatureBuilder::build(const Point& loc, const Belief& bw,
                                          const std::set<int>& collected) const {
    const int N = N_, nD = nD_, n_tel = n_tel_;

    // Three input groups that do not interact until assembly (the dossier's DAG): belief math (the
    // O(nb·(N+nD)) bottleneck), separable geometry, the collected indicator. Each is a pure unit.
    const BeliefFeatures& bf = belief_feats_(bw);          // memoized by belief VALUE (P6 bit-identical hit)
    const GeometryFeatures& gf = geometry_feats_(loc);     // memoized by loc
    const CollectedFeatures cf = collected_features(collected, N);

    // --- assemble by NAMED block: the layout SSOT (audit R6 / ADR-0012 P7) owns the order + offsets; the
    // ctor resolved them once into off_ (no positional `o += N` ladder re-encoding the layout here, and no
    // per-build string_view->offset hash lookup). `available` and `sum_unc` are the only belief×collected
    // couplings, computed in this step. Slots are independent, so the WRITE order is irrelevant; the
    // float-op order (the sum_unc accumulation, the per-i unc/available) is UNCHANGED from the former
    // ladder — bit-exact (P6). The per-build layout-mismatch assert is GONE: the ctor's load (Σwidth==dim
    // ==env-derived dim, no dup/neg widths ⇒ a contiguous partition) + its key-set check make the desync
    // that assert guarded structurally unauthorable (dossier §3 — mechanize > assert).
    std::vector<double> out(dim_, 0.0);
    { const int s = off_.marg;        for (int i = 0; i < N; ++i) out[s + i] = bf.marg[i]; }
    { const int s = off_.collected;   for (int i = 0; i < N; ++i) out[s + i] = cf.coll[i]; }
    { const int s = off_.available;   for (int i = 0; i < N; ++i)
          out[s + i] = ((bf.marg[i] > 0.0) && (cf.coll[i] == 0.0)) ? 1.0 : 0.0; }
    { const int s = off_.dist_t;      for (int i = 0; i < N; ++i) out[s + i] = gf.dist_t[i]; }
    double sum_unc = 0.0;
    { const int s = off_.unc;
      for (int i = 0; i < N; ++i) {
          double u = bf.marg[i] * (1.0 - bf.marg[i]);                     // unc
          out[s + i] = u;
          if (cf.coll[i] == 0.0) sum_unc += u;       // Σ over UNCOLLECTED treasures (order preserved)
      } }
    { const int s = off_.informative; for (int j = 0; j < nD; ++j) out[s + j] = bf.informative[j]; }
    { const int s = off_.p_pos;       for (int j = 0; j < nD; ++j) out[s + j] = bf.p_pos[j]; }
    { const int s = off_.dist_d;      for (int j = 0; j < nD; ++j) out[s + j] = gf.dist_d[j]; }
    out[off_.sharpness]   = bf.sharpness;                                 // log|bw|/log Nworlds
    out[off_.n_collected] = cf.n_collected / static_cast<double>(env_.K());  // n_collected/K
    out[off_.marg_sum]    = bf.marg_sum / static_cast<double>(env_.K());     // Σmarg/K
    out[off_.exit_norm]   = gf.exit_norm;                                 // exit geometry
    out[off_.nonempty]    = bf.nonempty;                                  // non-empty belief flag
    out[off_.sum_unc]     = sum_unc;                                      // Σ_uncollected unc
    { const int s = off_.dist_w;      for (int k = 0; k < n_tel; ++k) out[s + k] = gf.dist_w[k]; }
    return out;
}

std::vector<float> FeatureBuilder::legal_mask_from_features(std::span<const float> feat) const {
    // `feat` MUST be this builder's build() output (length dim()). The bare span cannot carry that
    // contract, so assert it (fail-loud, ADR-0002) — a mis-laid buffer is a programmer bug, not a
    // recoverable boundary.
    assert(feat.size() == static_cast<size_t>(dim_) && "legal_mask_from_features: feat is not a build() vector");
    // Slice the §2.2 blocks that ARE the mask (design §3): the per-treasure `available` block is the
    // legal-collect mask, the per-detector `informative` block is the legal-sense mask, TERMINATE is
    // always legal. No belief recompute — these blocks were just written by build() from the ONE marg
    // sweep (ADR-0012 P1). Offsets via the layout SSOT (named, not magic literals).
    std::vector<float> m(static_cast<size_t>(N_ + nD_ + 1), 0.0f);
    const int avail = off_.available;
    const int info = off_.informative;
    for (int i = 0; i < N_; ++i)
        m[static_cast<size_t>(i)] = (feat[static_cast<size_t>(avail + i)] > 0.0f) ? 1.0f : 0.0f;
    for (int j = 0; j < nD_; ++j)
        m[static_cast<size_t>(N_ + j)] = (feat[static_cast<size_t>(info + j)] > 0.0f) ? 1.0f : 0.0f;
    m[static_cast<size_t>(N_ + nD_)] = 1.0f;  // TERMINATE always legal (term_slot = N+nD)
    return m;
}

void FeatureBuilder::reset_belief_cache() const {
    belief_cache_.clear();
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
    // miss: the cap is a memory backstop (mirrors _belief_cache_cap); compute, store an OWNED copy of bw
    // (a stored span would dangle), return the cached ref.
    if (belief_cache_n_ >= kBeliefCacheCap) { belief_cache_.clear(); belief_cache_n_ = 0; }
    BeliefFeatures feats = belief_features(env_, bw);  // visits the rep (flat sweep / bitset popcount)
    auto& bucket = belief_cache_[key];
    bucket.emplace_back(bw, std::move(feats));
    ++belief_cache_n_;
    return bucket.back().second;
}

const GeometryFeatures& FeatureBuilder::geometry_feats_(const Point& loc) const {
    if (auto it = loc_cache_.find(loc); it != loc_cache_.end()) return it->second;  // hit
    GeometryFeatures feats = geometry_features(env_, loc, N_, nD_, n_tel_, diag_);
    return loc_cache_.emplace(loc, std::move(feats)).first->second;
}

}  // namespace chocofarm
