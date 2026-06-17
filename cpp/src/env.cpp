// cpp/src/env.cpp
// Purpose: the minimal C++ env port (see env.hpp). Mirrors chocofarm/model/env.py's belief
//   mechanics, dynamics, and geometry. The belief filters and the world-set are LOGIC-exact
//   (integer bit ops) — bit-identical to the numpy env; the distances are float-equivalent
//   (std::hypot mirroring math.hypot). ADR-0012 P6/P7.
//
// Public Domain (The Unlicense).
#include "chocofarm/env.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <iostream>

namespace chocofarm {

// C(N,K) bitmask world-set in itertools.combinations order (mirrors instance.world_array). Bit t
// set <=> treasure t present. The "next combination" walk reproduces combinations(range(N), K)
// element order exactly.
static std::vector<uint32_t> build_worlds(int N, int K) {
    std::vector<uint32_t> out;
    if (K < 0 || K > N) return out;
    std::vector<int> c(K);
    for (int i = 0; i < K; ++i) c[i] = i;
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
    worlds_ = build_worlds(inst_.N, inst_.K);
    // contiguous per-detector cover bitmasks (face_masks()): hoist faces[j].bitmask out of the
    // array-of-structs into a packed uint32_t[nD] so the belief sweep reads them without the AoS
    // stride (ADR-0012 P1 — one contiguous home, env still owns them). Order = face id (== faces order).
    face_masks_.reserve(inst_.faces.size());
    for (const Face& f : inst_.faces) face_masks_.push_back(f.bitmask);
    // resolve the entry teleport index
    entry_idx_ = 0;
    for (size_t k = 0; k < inst_.teleport_names.size(); ++k) {
        if (inst_.teleport_names[k] == inst_.entry) { entry_idx_ = static_cast<int>(k); break; }
    }
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
    std::vector<double> m(inst_.N, 0.0);
    if (bw.worlds.empty()) return m;
    for (uint32_t w : bw.worlds) {
        for (int t = 0; t < inst_.N; ++t) {
            if ((w >> t) & 1u) m[t] += 1.0;
        }
    }
    double inv = 1.0 / static_cast<double>(bw.worlds.size());
    for (double& v : m) v *= inv;  // mean over the world-set (mirrors env.marginals)
    return m;
}

bool Environment::informative(int face_id, const Belief& bw) const {
    uint32_t bm = inst_.faces[face_id].bitmask;
    bool any_hit = false, any_miss = false;
    for (uint32_t w : bw.worlds) {
        if ((w & bm) != 0) any_hit = true; else any_miss = true;
        if (any_hit && any_miss) return true;  // both polarities live
    }
    return false;  // mirrors SenseAction.informative: hit.any() and (~hit).any()
}

std::vector<Action> Environment::legal_actions(const Belief& bw,
                                               const std::set<int>& collected) const {
    std::vector<Action> acts;
    std::vector<double> marg = marginals(bw);
    // collects: marg>0 and not collected, in treasure-id order (env iterates _treasure_ids = range(N))
    for (int i = 0; i < inst_.N; ++i) {
        if (collected.count(i) == 0 && marg[i] > 0.0) acts.push_back(Action{ActionKind::Treasure, i});
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
    filter_inplace(bw.worlds, uint32_t{1} << i, present);   // a treasure is the single-bit mask 1<<i
}

void Environment::filter_detector(Belief& bw, int i, bool positive) const {
    filter_inplace(bw.worlds, inst_.faces[i].bitmask, positive);  // a detector is its cover bitmask (disjunction)
}

StepResult Environment::apply(Loc& loc, Belief& bw, std::set<int>& collected,
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
        bool fresh = pres && collected.count(action.i) == 0;
        res.reward = fresh ? 1.0 : 0.0;  // env.value[i] = 1.0 on the live instance (unit values)
        if (pres) collected.insert(action.i);
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
