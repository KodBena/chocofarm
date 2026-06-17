// cpp/src/ismcts.cpp
// Purpose: the C++ Single-Observer ISMCTS Policy implementation (see ismcts.hpp). A faithful
//   reimplementation of chocofarm/solvers/ismcts.py against the C++ env port — the information-set
//   node with subset-armed-bandit per-action statistics, the determinized per-iteration descent
//   (selection / expansion / simulation / backprop), the subset-armed UCB1 (eq. 7), and the
//   most-visited final — behind the composable Policy seam (ADR-0012 P2/P7: behavioral parity, NOT
//   byte-identity; the env/runner core is untouched).
//
//   DRY against the shared base (ADR-0012 P1): the leaf utility (base_value), the default leaf base
//   (GreedyStopBase), the UCB constant (UCB_C), and the generic world-sampling draw (the shared
//   RngWorldSource) all live in policy.{hpp,cpp} and are REUSED — this file does NOT include nmcs.hpp
//   and re-authors no base/sampling/leaf logic. ISMCTS's own pieces are here: the belief fingerprint,
//   the node arena, the subset-armed UCB select, and the RNG-source extension.
//
//   Parity-critical detail (the same hazard the NMCS strict-`>`/first-wins cleared): both the UCB
//   select and the most-visited final iterate over the per-action statistics in INSERTION order with
//   a strict `>` first-wins tie — Python dicts preserve insertion order and `if v > best_v` /
//   `max(...)` keep the first. C++ std::map iterates in sorted-key order, so we track insertion order
//   explicitly in `visit_order` and iterate THAT, never the map's key order.
//
// Public Domain (The Unlicense).
#include "chocofarm/ismcts.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>

#include "chocofarm/features.hpp"  // action_to_slot / term_slot: the action<->slot bijection

namespace chocofarm {

namespace {
// The fixed slot for an action (the action<->slot bijection, mirrors action_to_slot). ISMCTS keys
// its per-action maps by slot — a faithful stand-in for the Python Action-tuple keys (the mapping is
// a bijection), with insertion order tracked separately so the tie-break stays first-wins.
[[nodiscard]] int slot_of(const Environment& env, const Action& a) { return action_to_slot(env, a); }

// Reconstruct an Action from its slot (the inverse of action_to_slot), so run_search can return the
// selected action. Slot 0..N-1 = ("t", i); N..N+nD-1 = ("d", j); N+nD = TERMINATE.
[[nodiscard]] Action action_of_slot(const Environment& env, int slot) {
    if (slot < env.N()) return Action{ActionKind::Treasure, slot};
    if (slot < env.N() + env.n_detectors()) return Action{ActionKind::Detector, slot - env.N()};
    return terminate_action();
}

// _update(node, a, ret) (mirrors ISMCTSPolicy._update): bump visits + reward for action-slot `a`.
// The FIRST insertion of `a` into visits records its INSERTION ORDER in visit_order — the order both
// the UCB select and the most-visited final iterate (the parity-critical first-wins tie source).
void update_node(ISMCTSNode& nd, int a, double ret) {
    auto vit = nd.visits.find(a);
    if (vit == nd.visits.end()) {
        nd.visits[a] = 1;
        nd.visit_order.push_back(a);  // first visits-insertion -> insertion order
    } else {
        vit->second += 1;
    }
    auto rit = nd.reward.find(a);
    if (rit == nd.reward.end())
        nd.reward[a] = ret;
    else
        rit->second += ret;
}
}  // namespace

ISMCTSPolicy::ISMCTSPolicy(const ISMCTSConfig& cfg, const Policy* base)
    : cfg_(cfg), base_(base ? base : &default_base_) {}

// ---- subset-armed UCB1 selection (eq. 7) (mirrors _ucb_select) ------------------------------------
int ISMCTSPolicy::ucb_select(const ISMCTSNode& node) const {
    int best_a = -1;
    double best_v = -std::numeric_limits<double>::infinity();
    const double c = cfg_.c;
    // iterate in INSERTION order (visit_order), not the map's sorted-key order — the parity-critical
    // first-wins tie is over insertion order (mirrors Python dict iteration in _ucb_select).
    for (int a : node.visit_order) {
        int n_j = node.visits.at(a);
        if (n_j == 0) return a;  // an unselected-but-present arm: pick it immediately (mirrors Python)
        double exploit = node.reward.at(a) / static_cast<double>(n_j);
        auto it = node.avail.find(a);
        int navail = (it != node.avail.end()) ? it->second : n_j;  // avail.get(a, n_j)
        double explore = (navail > 1)
                             ? c * std::sqrt(std::log(static_cast<double>(navail)) /
                                             static_cast<double>(n_j))
                             : c;
        double v = exploit + explore;
        if (v > best_v) {  // strict >: first arm (insertion order) wins a tie (mirrors `if v > best_v`)
            best_v = v;
            best_a = a;
        }
    }
    // _ucb_select is only called on a fully-expanded node (visits non-empty), so a best is always
    // found; ADR-0012 P9 — an empty visit_order here is an invariant violation (a bug), assert/abort.
    assert(best_a != -1 && "ucb_select on an empty (unexpanded) node");
    return best_a;
}

// ---- one determinized iteration (mirrors _iterate) ------------------------------------------------
double ISMCTSPolicy::iterate(const Environment& env, std::vector<ISMCTSNode>& nodes, int node,
                             const Loc& loc, const std::vector<uint32_t>& bw,
                             const std::set<int>& collected, uint32_t world, double lam,
                             ISMCTSSource& src, int depth) const {
    if (depth >= cfg_.max_depth) return -lam * env.exit_cost(loc.pt);

    // Actions compatible with the determinization at this node: every legal action is compatible
    // (its observation is simply resolved by `world`); TERMINATE is always legal. The slot order is
    // legal_actions order (treasures id-order, then detectors id-order) then TERMINATE — the SAME
    // order Python builds `actions = list(legal) + [TERMINATE]`.
    std::vector<Action> actions = env.legal_actions(bw, collected);
    actions.push_back(terminate_action());
    std::vector<int> slots;
    slots.reserve(actions.size());
    for (const Action& a : actions) slots.push_back(slot_of(env, a));

    // Bump availability for every action legal on this visit (subset-armed bandit, §IV-B). This
    // touches ONLY the avail dict; the parity-critical insertion order is the VISITS-insertion order
    // (the order _update first inserts an action into `visits`), which both _ucb_select and the
    // most-visited final iterate — so visit_order is appended at the _update site below, NOT here.
    for (int a : slots) nodes[node].avail[a] = nodes[node].avail.count(a) ? nodes[node].avail[a] + 1 : 1;

    // (3) Expansion: if any compatible action is untried here, expand one uniformly.
    std::vector<int> untried;
    for (int a : slots)
        if (nodes[node].visits.find(a) == nodes[node].visits.end()) untried.push_back(a);
    if (!untried.empty()) {
        int pick = src.expand_index(static_cast<int>(untried.size()));
        int a = untried[static_cast<size_t>(pick)];
        Action act = action_of_slot(env, a);
        double ret;
        if (act.kind == ActionKind::Terminate) {
            ret = -lam * env.exit_cost(loc.pt);  // stop now: only the exit toll remains
        } else {
            Loc nloc = loc;
            std::vector<uint32_t> nbw = bw;
            std::set<int> nc = collected;
            StepResult sr = env.apply(nloc, nbw, nc, act, world);
            double step = sr.reward - lam * sr.dt;
            // register the successor child (still part of this edge's statistics), then the leaf.
            std::tuple<int, BeliefKey> ckey{a, belief_key(nbw)};
            if (nodes[node].children.find(ckey) == nodes[node].children.end()) {
                nodes.emplace_back();
                nodes[node].children[ckey] = static_cast<int>(nodes.size()) - 1;
            }
            double cont = src.leaf_value(nloc, nbw, nc, world, lam);
            ret = step + cont;
        }
        update_node(nodes[node], a, ret);  // _update(node, a, ret)
        return ret;
    }

    // (2) Selection: UCB1 with the availability count in the exploration term.
    int a = ucb_select(nodes[node]);
    Action act = action_of_slot(env, a);
    double ret;
    if (act.kind == ActionKind::Terminate) {
        ret = -lam * env.exit_cost(loc.pt);  // stop now: only the exit toll remains
    } else {
        Loc nloc = loc;
        std::vector<uint32_t> nbw = bw;
        std::set<int> nc = collected;
        StepResult sr = env.apply(nloc, nbw, nc, act, world);
        double step = sr.reward - lam * sr.dt;
        std::tuple<int, BeliefKey> ckey{a, belief_key(nbw)};
        auto cit = nodes[node].children.find(ckey);
        int child;
        if (cit == nodes[node].children.end()) {
            // the action edge exists, but this determinization routes to a successor belief not yet
            // seen — create that child node (still part of the same edge's statistics).
            nodes.emplace_back();
            child = static_cast<int>(nodes.size()) - 1;
            nodes[node].children[ckey] = child;
        } else {
            child = cit->second;
        }
        double cont = iterate(env, nodes, child, nloc, nbw, nc, world, lam, src, depth + 1);
        ret = step + cont;
    }
    update_node(nodes[node], a, ret);  // _update(node, a, ret)
    return ret;
}

// ---- the pure search core (mirrors decide's loop + final) -----------------------------------------
Action ISMCTSPolicy::run_search(const Environment& env, const Loc& loc,
                                const std::vector<uint32_t>& bw, const std::set<int>& collected,
                                double lam, ISMCTSSource& src) const {
    if (bw.empty()) return terminate_action();  // mirrors decide's len(bw)==0 -> TERMINATE
    std::vector<ISMCTSNode> nodes;
    nodes.emplace_back();  // the root (index 0)
    for (int i = 0; i < cfg_.iterations; ++i) {
        uint32_t w = src.sample_world(bw);  // (1) determinize: one world ~ belief
        std::set<int> coll = collected;     // _iterate mutates a fresh collected-set per iteration
        iterate(env, nodes, 0, loc, bw, coll, w, lam, src, 0);
    }
    // (final) most-visited root action; TERMINATE if nothing was tried (visits empty). First-wins tie
    // over INSERTION order (mirrors max(root.visits, key=...) which keeps the first max).
    const ISMCTSNode& root = nodes[0];
    if (root.visits.empty()) return terminate_action();
    int best_slot = -1;
    int best_visits = -1;
    for (int a : root.visit_order) {
        auto it = root.visits.find(a);
        if (it == root.visits.end()) continue;  // an avail-only arm never selected: skip
        if (it->second > best_visits) {  // strict >: first arm (insertion order) wins a tie
            best_visits = it->second;
            best_slot = a;
        }
    }
    assert(best_slot != -1 && "non-empty visits but no most-visited arm");
    return action_of_slot(env, best_slot);
}

namespace {
// The production ISMCTS source: the generic uniform sample_world (reused from the shared
// RngWorldSource), the expansion-index draw (rng.integers(n)), and the GreedyStopBase base_value
// leaf. It DERIVES from RngWorldSource so the world draw is the shared one (ADR-0012 P1); it adds the
// two ISMCTS-specific draws + the real leaf. Production wires off the real rng + the real base.
class RngISMCTSSource final : public ISMCTSSource {
  public:
    RngISMCTSSource(const Environment& env, const Policy& base, std::mt19937_64& rng)
        : env_(env), base_(base), draw_(rng), rng_(rng) {}

    uint32_t sample_world(const std::vector<uint32_t>& bw) override { return draw_.sample_world(bw); }

    int expand_index(int n) override {
        std::uniform_int_distribution<int> pick(0, n - 1);  // mirrors int(rng.integers(n))
        return pick(rng_);
    }

    double leaf_value(const Loc& loc, const std::vector<uint32_t>& bw,
                      const std::set<int>& collected, uint32_t world, double lam) override {
        return base_value(env_, base_, loc, bw, collected, world, lam);  // shared leaf utility (P1)
    }

  private:
    const Environment& env_;
    const Policy& base_;
    RngWorldSource draw_;   // the shared generic uniform-from-belief draw (ADR-0012 P1)
    std::mt19937_64& rng_;  // the SAME stream the draw uses, for the expansion-index draw
};
}  // namespace

Action ISMCTSPolicy::decide(const Environment& env, const Loc& loc, const std::vector<uint32_t>& bw,
                            const std::set<int>& collected, double lam,
                            std::mt19937_64& rng) const {
    RngISMCTSSource src(env, *base_, rng);
    return run_search(env, loc, bw, collected, lam, src);
}

}  // namespace chocofarm
