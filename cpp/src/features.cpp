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
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <span>

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

    // --- assemble the canonical ordered block table (per-treasure N×5, per-detector nD×3, global
    // 6+n_tel). `available` and `sum_unc` are the ONLY belief×collected couplings, so they are computed
    // HERE (not in the pure units). The op order is UNCHANGED from the former monolith — bit-exact
    // (P6). The `o += …` offset ladder is the SSOT target of the next step (dossier §3).
    std::vector<double> out(dim_, 0.0);
    int o = 0;
    for (int i = 0; i < N; ++i) { out[o + i] = bf.marg[i]; }              // marg
    o += N;
    for (int i = 0; i < N; ++i) { out[o + i] = cf.coll[i]; }              // collected
    o += N;
    for (int i = 0; i < N; ++i) { out[o + i] = ((bf.marg[i] > 0.0) && (cf.coll[i] == 0.0)) ? 1.0 : 0.0; }
    o += N;                                                               // available
    for (int i = 0; i < N; ++i) { out[o + i] = gf.dist_t[i]; }            // dist_t
    o += N;
    double sum_unc = 0.0;
    for (int i = 0; i < N; ++i) {
        double u = bf.marg[i] * (1.0 - bf.marg[i]);                       // unc
        out[o + i] = u;
        if (cf.coll[i] == 0.0) sum_unc += u;         // Σ over UNCOLLECTED treasures
    }
    o += N;

    // --- per-detector block (nD × 3): informative, p_pos, dist ---
    for (int j = 0; j < nD; ++j) { out[o + j] = bf.informative[j]; }      // informative
    o += nD;
    for (int j = 0; j < nD; ++j) { out[o + j] = bf.p_pos[j]; }            // p_pos
    o += nD;
    for (int j = 0; j < nD; ++j) { out[o + j] = gf.dist_d[j]; }           // dist_d
    o += nD;

    // --- global block (6 + n_tel) ---
    out[o++] = bf.sharpness;                                              // log|bw|/log Nworlds
    out[o++] = cf.n_collected / static_cast<double>(env_.K());           // n_collected/K
    out[o++] = bf.marg_sum / static_cast<double>(env_.K());              // Σmarg/K
    out[o++] = gf.exit_norm;                                             // exit geometry
    out[o++] = bf.nonempty;                                              // non-empty belief flag
    out[o++] = sum_unc;                                                  // Σ_uncollected unc
    for (int k = 0; k < n_tel; ++k) out[o++] = gf.dist_w[k];             // per-teleport distances

    // ADR-0012 P9: dim_ is derived from the SAME N/nD/n_tel this loop walks (5N+3nD+6+n_tel), so a
    // mismatch is impossible unless the derivation desyncs — an INVARIANT violation (a programmer
    // bug), an assert/abort, not a recoverable boundary Error.
    assert(o == dim_ && "FeatureBuilder::build: layout mismatch");
    return out;
}

}  // namespace chocofarm
