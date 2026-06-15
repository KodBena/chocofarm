// cpp/include/chocofarm/env.hpp
// Purpose: the minimal C++ env port — the SIMULATION MODEL decoupled from any Policy, mirroring
//   chocofarm/model/env.py. It owns the belief world-set (C(N,K) bitmasks), the legal-action set,
//   `apply` (move -> observe -> filter belief -> collect), the geometry distances, and the exact
//   belief filters (filter_treasure / filter_detector / the disjunction over a face's cover). It
//   knows nothing about HOW a decision is made — that is a Policy (policy.hpp), injected. A new
//   capability is a new Policy subclass with ZERO edits here (ADR-0012 P2, the env<->Policy seam).
//
//   NOT ported (deferred to a later slice, per ADR-0012's C++ section and scaling-and-cpp-seam.md
//   Shape A): the Gumbel-AZ search and the MLP forward. This is the dumb-random MVP that proves the
//   wire + the env<->Policy seam + the belief mechanics.
//
// Public Domain (The Unlicense).
#pragma once

#include <cstdint>
#include <set>
#include <vector>

#include "chocofarm/instance.hpp"

namespace chocofarm {

// An action: a kind tag + an index. Mirrors the env's ("t", i) / ("d", i) / TERMINATE shape.
enum class ActionKind { Treasure, Detector, Terminate };
struct Action {
    ActionKind kind = ActionKind::Terminate;
    int i = -1;  // treasure id (Treasure) or face id (Detector); -1 for Terminate
    bool operator==(const Action& o) const { return kind == o.kind && i == o.i; }
};
inline Action terminate_action() { return Action{ActionKind::Terminate, -1}; }

// Where the agent currently stands. Mirrors the env's loc tuple: a teleport ("w", k), a treasure
// ("t", i), or a detector ("d", i) rep_point. We carry the resolved Point so distance is a pure
// (Point, Point) -> double (std::hypot), mirroring env.d's static distance table (same hypot inputs).
struct Loc {
    Point pt;
};

// The result of apply (mirrors env.apply's (reward, loc', bw', collected', dt)). `bw` is mutated in
// place on the Environment for the running episode (the Python env returns a fresh filtered array;
// here we filter the member vector — same belief, same elements, same order).
struct StepResult {
    double reward = 0.0;
    double dt = 0.0;
};

class Environment {
  public:
    explicit Environment(const Instance& inst);

    // ---- belief world-set ----
    // The full C(N,K) world-set as bitmasks over treasure ids (mirrors world_array): all K-subsets
    // of range(N), in itertools.combinations order, as (1<<t) sums. 20 bits fit a uint32.
    const std::vector<uint32_t>& worlds() const { return worlds_; }
    int N() const { return inst_.N; }
    int K() const { return inst_.K; }
    int n_detectors() const { return static_cast<int>(inst_.faces.size()); }

    // ---- geometry ----
    double dist(const Point& a, const Point& b) const;        // std::hypot (mirrors env.d/math.hypot)
    double exit_cost(const Point& loc) const;                 // min teleport dist + tp (env.exit_cost)
    Point entry_point() const { return inst_.teleports[entry_idx_]; }
    Point treasure_pt(int i) const { return inst_.treasures[i]; }
    Point face_pt(int i) const { return inst_.faces[i].rep_point; }
    int n_teleports() const { return static_cast<int>(inst_.teleports.size()); }
    Point teleport_pt(int k) const { return inst_.teleports[k]; }

    // ---- belief marginals (mirrors env.marginals) ----
    std::vector<double> marginals(const std::vector<uint32_t>& bw) const;

    // ---- dynamics ----
    // Legal action set for (loc, belief, collected): collects with marg>0 and not collected, plus
    // each face whose outcome is still uncertain over the belief (informative). TERMINATE is NOT
    // included here (it is the always-legal extra slot, appended by the Policy / the mask builder),
    // matching env.legal_actions + actions.term_slot exactly.
    std::vector<Action> legal_actions(const std::vector<uint32_t>& bw,
                                      const std::set<int>& collected) const;

    // Realise `action` against the true `world`. Filters `bw` IN PLACE (move/observe/collect), and
    // returns (reward, dt). The belief filter is the same disjunction/treasure-bit logic as env.py.
    StepResult apply(Loc& loc, std::vector<uint32_t>& bw, std::set<int>& collected,
                     const Action& action, uint32_t world) const;

    // ---- belief filters (mirror filter_treasure / SenseAction.filter) ----
    void filter_treasure(std::vector<uint32_t>& bw, int i, bool present) const;
    void filter_detector(std::vector<uint32_t>& bw, int i, bool positive) const;

    // A face's true reading at a concrete world (mirrors SenseAction.observe).
    bool observe(int face_id, uint32_t world) const {
        return (world & inst_.faces[face_id].bitmask) != 0;
    }
    // Outcome still uncertain over the belief — both polarities live (SenseAction.informative).
    bool informative(int face_id, const std::vector<uint32_t>& bw) const;

  private:
    Instance inst_;
    std::vector<uint32_t> worlds_;
    int entry_idx_ = 0;
};

}  // namespace chocofarm
