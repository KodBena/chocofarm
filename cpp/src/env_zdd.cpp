// cpp/src/env_zdd.cpp
// Purpose: the env-seam-op bodies for the OPT-IN belief-as-diagram (ZDD) arm — namespace chocofarm::zdd
//   (declared in env.hpp under #ifdef CHOCO_BELIEF_ZDD, alongside the ZddBelief value type). The §B.4(b)
//   graduation
//   (belief_features_and_decision_diagram_note.md Part B; docs/design/cpp-belief-zdd-onramp.md;
//   docs/design/cpp-belief-rep-scoping.md §2 op table). Compiled ONLY when CHOCO_BELIEF_ZDD is ON
//   (chocofarm_core's CMake gates this TU on the option), so the DEFAULT build never sees it and is
//   byte-for-byte the current flat+bitset.
//
//   Each op MIRRORS its FlatBelief/BitsetBelief twin: the counts/marginals/det-counts/features are
//   BYTE-IDENTICAL to the flat arm (the §B.3 logic invariant — exact integer counts + the IDENTICAL
//   Phase-2 `* inv`), and the filters give members set-equal to the flat filter's kept world-set. The
//   sampling trio (world_at_rank / sample_world via the env / belief_key) RE-BASELINES: the ZDD's
//   canonical member order != worlds()-rank order, so a scripted/sampled rank resolves to a DIFFERENT
//   world (the deliberate, expected divergence — design §4 of the B.4(b) task).
//
// Public Domain (The Unlicense).
#ifdef CHOCO_BELIEF_ZDD

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <span>
#include <vector>

#include "chocofarm/env.hpp"       // ZddBelief + the zdd:: op declarations + the engine (under the flag)
#include "chocofarm/features.hpp"  // the full BeliefFeatures definition

namespace chocofarm::zdd {

namespace {
// The loud-abort arm the seam owns for the canonical-rank unrank (mirrors env.cpp's rank_or_abort): the
// engine returns the 0xFFFFFFFF sentinel on r out of [0,count) (a cached_count_/diagram desync); here
// that becomes a FATAL abort (ADR-0002 / scoping §6 risk 7). It never legitimately returns the sentinel.
[[nodiscard]] uint32_t member_or_abort(const beliefzdd::BeliefDiagram& z, int64_t r) {
    const uint32_t w = z.member_at_rank(r);
    if (w == 0xFFFFFFFFu) {
        std::cerr << "chocofarm: FATAL invariant: zdd member_at_rank: r out of range "
                     "(cached_count_ desynced from the diagram?)\n";
        std::abort();
    }
    return w;
}
}  // namespace

ZddBelief full_belief(const Environment& env) {
    // The diagram of EVERY world (the C(N,K) prior). worlds() is duplicate-free by construction, so the
    // build's nb := count(Z) == |worlds| (the §13 trap-8 precondition holds).
    const std::vector<uint32_t>& all = env.worlds();
    ZddBelief b;
    b.z = beliefzdd::BeliefDiagram(std::span<const uint32_t>(all), env.N());
    b.cached_count_ = b.z.count();
    return b;
}

void filter_treasure(ZddBelief& b, int i, bool present) {
    // restrict the maintained diagram in place (the B.4(b) op — NOT a rebuild). A treasure is a single
    // variable; keep members that set it (present) / do not (absent). Recompute the cached count.
    b.z.restrict_var(i, present);
    b.cached_count_ = b.z.count();
}

void filter_detector(const Environment& env, ZddBelief& b, int i, bool positive) {
    // restrict by the face's cover DISJUNCTION (the same cover bitmask filter_detector reads). positive:
    // the disjunction holds (≥1 cover bit set); negative: it fails (none set — the disjoint subfamily).
    b.z.restrict_cover(env.face_masks()[static_cast<size_t>(i)], positive);
    b.cached_count_ = b.z.count();
}

uint32_t world_at_rank(const ZddBelief& b, int r) {
    return member_or_abort(b.z, r);  // r-th member in ZDD CANONICAL order (the re-baseline)
}

std::vector<double> marginals(const Environment& env, const ZddBelief& b) {
    // bit_cnt[t] over the diagram, then the SAME `* inv` (1/nb) as the flat arm — byte-identical marg
    // (exact integer counts, exact (double)count for count <= |worlds|). §B.3.
    std::vector<double> m(static_cast<size_t>(env.N()), 0.0);
    if (b.cached_count_ == 0) return m;
    const std::vector<int64_t> bit_cnt = b.z.all_marginals();
    const double inv = 1.0 / static_cast<double>(b.cached_count_);
    for (int t = 0; t < env.N(); ++t) m[static_cast<size_t>(t)] = static_cast<double>(bit_cnt[static_cast<size_t>(t)]) * inv;
    return m;
}

bool informative(const Environment& env, int face_id, const ZddBelief& b) {
    // 0 < det_cnt[j] < nb (the cover count strictly between empty and full <=> a hit AND a miss both
    // exist) — byte-identical to the flat two-polarity scan. det_cnt via the non-constructing disjoint
    // count over the single face's cover mask.
    if (b.cached_count_ == 0) return false;
    const uint32_t mask = env.face_masks()[static_cast<size_t>(face_id)];
    const std::vector<uint32_t> one_mask{mask};
    const int64_t det = b.z.all_detector_counts(std::span<const uint32_t>(one_mask))[0];
    return det > 0 && det < b.cached_count_;
}

// NB: no zdd::legal_actions / zdd::sample_world here — Environment::legal_actions composes
// marginals()+informative() and Environment::sample_world composes nb()+world_at_rank(), both of which
// already visit-dispatch to the ZDD arm (one home, P1). Only the leaf ops need a ZDD body.

BeliefFeatures belief_features(const Environment& env, const ZddBelief& b) {
    const int N = env.N();
    const int nD = env.n_detectors();
    const double log_nworlds = std::log(static_cast<double>(env.worlds().size()));
    const int64_t nb = b.cached_count_;
    BeliefFeatures bf;
    bf.marg.assign(N, 0.0);
    bf.p_pos.assign(nD, 0.0);
    bf.informative.assign(nD, 0.0);
    if (nb == 0) return bf;  // == belief_features_empty (the flat/bitset empty branch)
    // ONE all-marginals sweep + ONE all-detector-counts pass off the diagram (the §A.4 Phase-1 integer
    // outputs), then the IDENTICAL Phase-2 `* inv` (NEVER / nb; §6 risk 6) — so the WHOLE BeliefFeatures
    // is byte-identical to belief_features_nonempty / belief_features_bitset for the same belief.
    const std::vector<int64_t> bit_cnt = b.z.all_marginals();
    const std::vector<int64_t> det_cnt = b.z.all_detector_counts(env.face_masks());
    const double inv = 1.0 / static_cast<double>(nb);
    for (int t = 0; t < N; ++t) {
        bf.marg[static_cast<size_t>(t)] = static_cast<double>(bit_cnt[static_cast<size_t>(t)]) * inv;
        bf.marg_sum += bf.marg[static_cast<size_t>(t)];  // treasure-id order (P6)
    }
    for (int j = 0; j < nD; ++j) {
        bf.p_pos[static_cast<size_t>(j)] = static_cast<double>(det_cnt[static_cast<size_t>(j)]) * inv;
        bf.informative[static_cast<size_t>(j)] =
            (det_cnt[static_cast<size_t>(j)] > 0 && det_cnt[static_cast<size_t>(j)] < nb) ? 1.0 : 0.0;
    }
    bf.sharpness = std::log(static_cast<double>(nb)) / log_nworlds;
    bf.nonempty = 1.0;
    return bf;
}

}  // namespace chocofarm::zdd

#endif  // CHOCO_BELIEF_ZDD
