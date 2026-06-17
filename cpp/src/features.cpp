// cpp/src/features.cpp
// Purpose: the C++ port of the §2.2 featurization + the action<->slot mask (see features.hpp).
//   Mirrors chocofarm/az/features.py (FeatureLayout block order + FeatureBuilder.build) and
//   chocofarm/az/actions.py (n_action_slots + legal_mask). The mask is the LOGIC INVARIANT M
//   (bit-exact, ADR-0012 P6/P7); the feature vector is float-sensitive (behavioral bar).
//
//   build() is DECOMPOSED (docs/design refactor step 1) along the three input groups the dossier's
//   DAG identifies — belief math (the O(nb·(N+nD)) bottleneck), separable geometry, the collected
//   indicator — into three pure value-functions plus a thin assembler. This is hygiene only (ADR-0012
//   P3: one axis per function; P9-rule-2: returned by value): the ops and their order are UNCHANGED,
//   so the output is bit-identical to the former monolith (P6). `available` and `sum_unc` are the only
//   belief×collected couplings, so they live in the assembler, which keeps belief_features a pure
//   function of `bw` — the unit a later step hoists to a single SSOT shared with legal_mask /
//   legal_actions (today those recompute the same marginals per leaf).
//
// Public Domain (The Unlicense).
#include "chocofarm/features.hpp"

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

std::vector<float> legal_mask(const Environment& env, const std::vector<uint32_t>& bw,
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
}

namespace {

// --- belief-derived intermediates: a PURE function of the world-set `bw` (the §2-decomposition unit;
// the memoizable / hoistable core). marg[i] = mean over bw of bit i; per detector cnt = #worlds in the
// disjunction's cover, p_pos = cnt/nb, informative = (0 < cnt < nb). Same ops + order as the former
// inline block (bit-exact, ADR-0012 P6). The O(nb·(N+nD)) sweep here is the profile bottleneck.
struct BeliefFeatures {
    std::vector<double> marg;         // N  — per-treasure marginal P(present)
    std::vector<double> p_pos;        // nD — detector positive-cover probability
    std::vector<double> informative;  // nD — detector splits the belief (0/1)
    double marg_sum = 0.0;            // Σ marg[t]  (order-fixed — a P6 watch item; do not reorder)
    double sharpness = 0.0;           // log|bw| / log Nworlds
    double nonempty = 0.0;            // nb ? 1.0 : 0.0
};

[[nodiscard]] BeliefFeatures belief_features(const Environment& env, std::span<const uint32_t> bw,
                                             int N, int nD, double log_nworlds) {
    BeliefFeatures bf;
    bf.marg.assign(N, 0.0);
    std::vector<int64_t> cnt(nD, 0);
    const size_t nb = bw.size();
    for (uint32_t w : bw) {
        for (int t = 0; t < N; ++t) if ((w >> t) & 1u) bf.marg[t] += 1.0;
        for (int j = 0; j < nD; ++j) if (env.observe(j, w)) cnt[j] += 1;
    }
    if (nb) {
        double inv = 1.0 / static_cast<double>(nb);
        for (int t = 0; t < N; ++t) bf.marg[t] *= inv;
    }
    for (int t = 0; t < N; ++t) bf.marg_sum += bf.marg[t];
    bf.p_pos.assign(nD, 0.0);
    bf.informative.assign(nD, 0.0);
    for (int j = 0; j < nD; ++j) {
        if (nb) {
            bf.p_pos[j] = static_cast<double>(cnt[j]) / static_cast<double>(nb);
            bf.informative[j] = (cnt[j] > 0 && cnt[j] < static_cast<int64_t>(nb)) ? 1.0 : 0.0;
        }
    }
    bf.sharpness = nb ? (std::log(static_cast<double>(nb)) / log_nworlds) : 0.0;
    bf.nonempty = nb ? 1.0 : 0.0;
    return bf;
}

// --- per-loc static distance block: geometry is FULLY separable (one outgoing edge in the DAG). Each
// distance normalized by the bbox diagonal. `hypot` chains TODAY (a precomputed lookup is the deferred
// §5 move). Same ops + order as the former inline block.
struct GeometryFeatures {
    std::vector<double> dist_t;  // N
    std::vector<double> dist_d;  // nD
    std::vector<double> dist_w;  // n_tel
    double exit_norm = 0.0;
};

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

std::vector<double> FeatureBuilder::build(const Point& loc, const std::vector<uint32_t>& bw,
                                          const std::set<int>& collected) const {
    const int N = N_, nD = nD_, n_tel = n_tel_;

    // Three input groups that do not interact until assembly (the dossier's DAG): belief math (the
    // O(nb·(N+nD)) bottleneck), separable geometry, the collected indicator. Each is a pure unit.
    const BeliefFeatures bf = belief_features(env_, std::span<const uint32_t>(bw), N, nD, log_nworlds_);
    const GeometryFeatures gf = geometry_features(env_, loc, N, nD, n_tel, diag_);
    const CollectedFeatures cf = collected_features(collected, N);

    // --- assemble by NAMED block: the layout SSOT (audit R6 / ADR-0012 P7) owns the order + offsets,
    // so there is no positional `o += N` ladder re-encoding it here. `available` and `sum_unc` are the
    // only belief×collected couplings, computed in this step. Slots are independent, so the WRITE order
    // is irrelevant; the float-op order (the sum_unc accumulation, the per-i unc/available) is UNCHANGED
    // from the former ladder — bit-exact (P6). The per-build layout-mismatch assert is GONE: the ctor's
    // load (Σwidth==dim==env-derived dim, no dup/neg widths ⇒ a contiguous partition) + its key-set
    // check make the desync that assert guarded structurally unauthorable (dossier §3 — mechanize > assert).
    std::vector<double> out(dim_, 0.0);
    { const int s = layout_.start("marg");        for (int i = 0; i < N; ++i) out[s + i] = bf.marg[i]; }
    { const int s = layout_.start("collected");   for (int i = 0; i < N; ++i) out[s + i] = cf.coll[i]; }
    { const int s = layout_.start("available");   for (int i = 0; i < N; ++i)
          out[s + i] = ((bf.marg[i] > 0.0) && (cf.coll[i] == 0.0)) ? 1.0 : 0.0; }
    { const int s = layout_.start("dist_t");      for (int i = 0; i < N; ++i) out[s + i] = gf.dist_t[i]; }
    double sum_unc = 0.0;
    { const int s = layout_.start("unc");
      for (int i = 0; i < N; ++i) {
          double u = bf.marg[i] * (1.0 - bf.marg[i]);                     // unc
          out[s + i] = u;
          if (cf.coll[i] == 0.0) sum_unc += u;       // Σ over UNCOLLECTED treasures (order preserved)
      } }
    { const int s = layout_.start("informative"); for (int j = 0; j < nD; ++j) out[s + j] = bf.informative[j]; }
    { const int s = layout_.start("p_pos");       for (int j = 0; j < nD; ++j) out[s + j] = bf.p_pos[j]; }
    { const int s = layout_.start("dist_d");      for (int j = 0; j < nD; ++j) out[s + j] = gf.dist_d[j]; }
    out[layout_.start("sharpness")]   = bf.sharpness;                     // log|bw|/log Nworlds
    out[layout_.start("n_collected")] = cf.n_collected / static_cast<double>(env_.K());  // n_collected/K
    out[layout_.start("marg_sum")]    = bf.marg_sum / static_cast<double>(env_.K());     // Σmarg/K
    out[layout_.start("exit_norm")]   = gf.exit_norm;                     // exit geometry
    out[layout_.start("nonempty")]    = bf.nonempty;                      // non-empty belief flag
    out[layout_.start("sum_unc")]     = sum_unc;                          // Σ_uncollected unc
    { const int s = layout_.start("dist_w");      for (int k = 0; k < n_tel; ++k) out[s + k] = gf.dist_w[k]; }
    return out;
}

std::vector<float> FeatureBuilder::legal_mask_from_features(std::span<const float> feat) const {
    // Slice the §2.2 blocks that ARE the mask (design §3): the per-treasure `available` block is the
    // legal-collect mask, the per-detector `informative` block is the legal-sense mask, TERMINATE is
    // always legal. No belief recompute — these blocks were just written by build() from the ONE marg
    // sweep (ADR-0012 P1). Offsets via the layout SSOT (named, not magic literals).
    std::vector<float> m(static_cast<size_t>(N_ + nD_ + 1), 0.0f);
    const int avail = layout_.start("available");
    const int info = layout_.start("informative");
    for (int i = 0; i < N_; ++i)
        m[static_cast<size_t>(i)] = (feat[static_cast<size_t>(avail + i)] > 0.0f) ? 1.0f : 0.0f;
    for (int j = 0; j < nD_; ++j)
        m[static_cast<size_t>(N_ + j)] = (feat[static_cast<size_t>(info + j)] > 0.0f) ? 1.0f : 0.0f;
    m[static_cast<size_t>(N_ + nD_)] = 1.0f;  // TERMINATE always legal (term_slot = N+nD)
    return m;
}

}  // namespace chocofarm
