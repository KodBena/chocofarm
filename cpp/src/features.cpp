// cpp/src/features.cpp
// Purpose: the C++ port of the §2.2 featurization + the action<->slot mask (see features.hpp).
//   Mirrors chocofarm/az/features.py (FeatureLayout block order + FeatureBuilder.build) and
//   chocofarm/az/actions.py (n_action_slots + legal_mask). The mask is the LOGIC INVARIANT M
//   (bit-exact, ADR-0012 P6/P7); the feature vector is float-sensitive (behavioral bar).
//
// Public Domain (The Unlicense).
#include "chocofarm/features.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

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
    throw std::runtime_error("action_to_slot: unknown action");
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

std::vector<double> FeatureBuilder::build(const Point& loc, const std::vector<uint32_t>& bw,
                                          const std::set<int>& collected) const {
    std::vector<double> out(dim_, 0.0);
    const int N = N_, nD = nD_, n_tel = n_tel_;
    const size_t nb = bw.size();

    // --- belief-derived intermediates (functions of bw alone) ---
    // marg[i] = mean over bw of bit i; for each detector: cnt = #worlds in cover (the disjunction
    // hit count), p_pos = cnt/nb, informative = (0 < cnt < nb). Mirrors the fused belief_marg_cover.
    std::vector<double> marg(N, 0.0);
    std::vector<int64_t> cnt(nD, 0);
    for (uint32_t w : bw) {
        for (int t = 0; t < N; ++t) if ((w >> t) & 1u) marg[t] += 1.0;
        for (int j = 0; j < nD; ++j) if (env_.observe(j, w)) cnt[j] += 1;
    }
    double marg_sum = 0.0;
    if (nb) {
        double inv = 1.0 / static_cast<double>(nb);
        for (int t = 0; t < N; ++t) marg[t] *= inv;
    }
    for (int t = 0; t < N; ++t) marg_sum += marg[t];
    std::vector<double> p_pos(nD, 0.0), informative(nD, 0.0);
    for (int j = 0; j < nD; ++j) {
        if (nb) {
            p_pos[j] = static_cast<double>(cnt[j]) / static_cast<double>(nb);
            informative[j] = (cnt[j] > 0 && cnt[j] < static_cast<int64_t>(nb)) ? 1.0 : 0.0;
        }
    }
    double sharpness = nb ? (std::log(static_cast<double>(nb)) / log_nworlds_) : 0.0;

    // --- per-loc static distance block (normalized by the bbox diagonal) ---
    std::vector<double> dist_t(N), dist_d(nD), dist_w(n_tel);
    for (int i = 0; i < N; ++i) dist_t[i] = std::hypot(loc.x - env_.treasure_pt(i).x,
                                                       loc.y - env_.treasure_pt(i).y) / diag_;
    for (int j = 0; j < nD; ++j) dist_d[j] = std::hypot(loc.x - env_.face_pt(j).x,
                                                        loc.y - env_.face_pt(j).y) / diag_;
    for (int k = 0; k < n_tel; ++k) dist_w[k] = std::hypot(loc.x - env_.teleport_pt(k).x,
                                                           loc.y - env_.teleport_pt(k).y) / diag_;
    double exit_norm = env_.exit_cost(loc) / diag_;

    // --- per-treasure block (N × 5): marg, collected, available, dist, unc ---
    std::vector<double> coll(N, 0.0);
    for (int i : collected) coll[i] = 1.0;
    int o = 0;
    for (int i = 0; i < N; ++i) { out[o + i] = marg[i]; }                 // marg
    o += N;
    for (int i = 0; i < N; ++i) { out[o + i] = coll[i]; }                 // collected
    o += N;
    for (int i = 0; i < N; ++i) { out[o + i] = ((marg[i] > 0.0) && (coll[i] == 0.0)) ? 1.0 : 0.0; }
    o += N;                                                               // available
    for (int i = 0; i < N; ++i) { out[o + i] = dist_t[i]; }               // dist_t
    o += N;
    double sum_unc = 0.0;
    for (int i = 0; i < N; ++i) {
        double u = marg[i] * (1.0 - marg[i]);                            // unc
        out[o + i] = u;
        if (coll[i] == 0.0) sum_unc += u;            // Σ over UNCOLLECTED treasures
    }
    o += N;

    // --- per-detector block (nD × 3): informative, p_pos, dist ---
    for (int j = 0; j < nD; ++j) { out[o + j] = informative[j]; }         // informative
    o += nD;
    for (int j = 0; j < nD; ++j) { out[o + j] = p_pos[j]; }               // p_pos
    o += nD;
    for (int j = 0; j < nD; ++j) { out[o + j] = dist_d[j]; }              // dist_d
    o += nD;

    // --- global block (6 + n_tel) ---
    out[o++] = sharpness;                                                  // log|bw|/log Nworlds
    out[o++] = static_cast<double>(collected.size()) / static_cast<double>(env_.K());  // n_collected/K
    out[o++] = marg_sum / static_cast<double>(env_.K());                  // Σmarg/K
    out[o++] = exit_norm;                                                  // exit geometry
    out[o++] = nb ? 1.0 : 0.0;                                             // non-empty belief flag
    out[o++] = sum_unc;                                                    // Σ_uncollected unc
    for (int k = 0; k < n_tel; ++k) out[o++] = dist_w[k];                  // per-teleport distances

    if (o != dim_) throw std::runtime_error("FeatureBuilder::build: layout mismatch");
    return out;
}

}  // namespace chocofarm
