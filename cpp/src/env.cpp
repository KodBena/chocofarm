// cpp/src/env.cpp
// Purpose: the minimal C++ env port (see env.hpp). Mirrors chocofarm/model/env.py's belief
//   mechanics, dynamics, and geometry. The belief filters and the world-set are LOGIC-exact
//   (integer bit ops) — bit-identical to the numpy env; the distances are float-equivalent
//   (std::hypot mirroring math.hypot). ADR-0012 P6/P7.
//
// Public Domain (The Unlicense).
#include "chocofarm/env.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

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

std::vector<double> Environment::marginals(const std::vector<uint32_t>& bw) const {
    std::vector<double> m(inst_.N, 0.0);
    if (bw.empty()) return m;
    for (uint32_t w : bw) {
        for (int t = 0; t < inst_.N; ++t) {
            if ((w >> t) & 1u) m[t] += 1.0;
        }
    }
    double inv = 1.0 / static_cast<double>(bw.size());
    for (double& v : m) v *= inv;  // mean over the world-set (mirrors env.marginals)
    return m;
}

bool Environment::informative(int face_id, const std::vector<uint32_t>& bw) const {
    uint32_t bm = inst_.faces[face_id].bitmask;
    bool any_hit = false, any_miss = false;
    for (uint32_t w : bw) {
        if ((w & bm) != 0) any_hit = true; else any_miss = true;
        if (any_hit && any_miss) return true;  // both polarities live
    }
    return false;  // mirrors SenseAction.informative: hit.any() and (~hit).any()
}

std::vector<Action> Environment::legal_actions(const std::vector<uint32_t>& bw,
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

void Environment::filter_treasure(std::vector<uint32_t>& bw, int i, bool present) const {
    uint32_t bit = uint32_t{1} << i;
    auto keep = [&](uint32_t w) { return (((w & bit) != 0) == present); };
    bw.erase(std::remove_if(bw.begin(), bw.end(), [&](uint32_t w) { return !keep(w); }), bw.end());
}

void Environment::filter_detector(std::vector<uint32_t>& bw, int i, bool positive) const {
    uint32_t bm = inst_.faces[i].bitmask;
    auto keep = [&](uint32_t w) { return (((w & bm) != 0) == positive); };  // the disjunction filter
    bw.erase(std::remove_if(bw.begin(), bw.end(), [&](uint32_t w) { return !keep(w); }), bw.end());
}

StepResult Environment::apply(Loc& loc, std::vector<uint32_t>& bw, std::set<int>& collected,
                              const Action& action, uint32_t world) const {
    StepResult res;
    Point target;
    if (action.kind == ActionKind::Treasure) target = inst_.treasures[action.i];
    else if (action.kind == ActionKind::Detector) target = inst_.faces[action.i].rep_point;
    else throw std::runtime_error("Environment::apply called with TERMINATE");

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
