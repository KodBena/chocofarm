// cpp/include/chocofarm/ismcts.hpp
// Purpose: the C++ Single-Observer ISMCTS Policy — Information Set Monte-Carlo Tree Search (Cowling,
//   Powley & Whitehouse, IEEE TCIAIG 2012; Algorithm 1 §IV-E, subset-armed-bandit UCB §IV-B) ported
//   behind the composable env<->Policy seam, mirroring chocofarm/solvers/ismcts.py EXACTLY (ADR-0012
//   P7: derive from the ONE authority, reimplement, behavioral parity NOT byte-identity). It is a
//   drop-in `Policy` alongside RandomPolicy / NMCSPolicy: the runner takes `const Policy&` and never
//   names this class, so adding it is ZERO edits to the search/env core (the P2 seam).
//
//   DRY against the shared base (ADR-0012 P1): the leaf utility (base_value), the default leaf base
//   (GreedyStopBase), and the generic world-sampling draw (the shared WorldSource sample_world) all
//   live in policy.{hpp,cpp} and are REUSED here, exactly as ismcts.py IMPORTS _base_value, UCB_C and
//   GreedyStopBase from solvers.base. This unit does NOT include nmcs.hpp; it shares only the base.
//   ISMCTS's own pieces are the information-set node, the (count, bw[0], bw[-1]) belief fingerprint
//   (ISMCTS-specific, kept local), the subset-armed UCB1 selection, and its RNG-source extension
//   (the expansion-index draw + the scripted leaf for the logic check).
//
//   Algorithm (mirrors ismcts.py): an information-set _Node keeps per-action reward[a] / visits[a]
//   (n_j) / avail[a] (n'_j) AGGREGATED over the info-set, children keyed by (action, belief_key).
//   Per decide(): `iterations` determinized walks; each samples one world w ~ bw and recurses
//   `iterate` in that fixed world. iterate: depth≥max_depth → −λ·exit_cost; actions =
//   legal_actions + [TERMINATE]; bump avail[a] for every action (subset-armed §IV-B); if any untried,
//   expand one uniformly (the source's expansion-index draw), play the base to the end for the leaf
//   (base_value with GreedyStopBase), update, return; else UCB1-select (eq.7, subset-armed
//   denominator, strict `>` first-wins over insertion order), route the determinization to the
//   (action, belief_key) child, recurse, backprop. TERMINATE edge value = −λ·exit_cost; step =
//   r − λ·dt. Final: the most-visited root action (first-wins tie), TERMINATE if nothing was tried.
//
//   RNG note (ADR-0012 P6): std::mt19937_64 != numpy, so aggregate parity is the behavioral bar, not
//   byte-identity. Both RNG draws (sample_world + the expansion index) AND the leaf value route
//   through the injectable ISMCTSSource so a DETERMINISTIC logic check feeds BOTH languages identical
//   scripted sequences and asserts the SAME selected action — the selection/nesting logic validated
//   independent of RNG, exactly as nmcs_logic.py does for NMCS.
//
// Public Domain (The Unlicense).
#pragma once

#include <cstddef>
#include <cstdint>
#include <random>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "chocofarm/belief_key.hpp"
#include "chocofarm/collected_set.hpp"
#include "chocofarm/domains.hpp"  // SlotIndex/SlotCount/SimBudget/PlyDepth/VisitCount + Treasure|FaceId — typed search domains (P1)
#include "chocofarm/env.hpp"
#include "chocofarm/policy.hpp"

namespace chocofarm {

// The frozen scalar hyperparameters (mirrors solvers.ismcts.ISMCTSConfig — audit item I). The
// simulation `base` (a Policy, not a scalar) stays a separate construction param, exactly as Python
// keeps `base` out of the frozen config. Defaults match ISMCTSConfig's (iterations=300, c=UCB_C,
// max_depth=24).
struct ISMCTSConfig {
    SimBudget iterations{300};  // determinized tree-walks per decision (the search budget — SimBudget)
    double c = UCB_C;           // the UCB1 exploration constant (paper default 0.7; one home: base.UCB_C)
    PlyDepth max_depth{24};     // recursion depth cap (a depth>=max_depth leaf is the bare -lam*exit_cost)
};

// The information-set node identity: the (count, first, last) belief fingerprint — now the ONE shared
// authority in belief_key.hpp (ADR-0012 P1), reused by the Gumbel node cache and the FeatureBuilder
// memo. ISMCTS, Gumbel, and the featurizer each authored this triple before; they now derive from one
// home. BeliefKey + belief_key(bw) come from chocofarm/belief_key.hpp, included above.

// ISMCTS's RNG-source extension of the shared WorldSource. The generic `sample_world` draw lives in
// the base; ISMCTS adds (a) `expand_index(n)` — the uniform draw selecting which untried action to
// expand (mirrors rng.integers(len(untried))) — and (b) `leaf_value(...)` — the leaf estimate from a
// freshly expanded post-action belief (production: base_value with the GreedyStopBase; scripted: the
// next canned number). Routing BOTH the draw and the leaf through one source lets the deterministic
// logic check script identical sequences across both languages (ADR-0012 P6).
struct ISMCTSSource : public WorldSource {
    // Uniform index in [0, n) over the untried-action LIST (mirrors int(rng.integers(n))). This is a
    // CONTAINER POSITION into the local `untried` vector, NOT a domain quantity (the slot it selects is
    // untried[pick]) — so it stays a raw int (the documented container-index ACL), like a std::vector idx.
    virtual int expand_index(int n) = 0;
    // The leaf value at a freshly expanded post-action (loc, bw, collected) in the fixed `world`
    // (mirrors _base_value(env, base, nloc, nbw, ncoll, world, lam)). `world` is a concrete World mask.
    virtual double leaf_value(const Loc& loc, const Belief& bw,
                              const CollectedSet& collected, World world, double lam) = 0;
};

// Hash for the children transposition key (action-slot, belief_key) = tuple<int, tuple<int,u32,u32>>.
// `children` is a find/insert-only TRANSPOSITION TABLE (never iterated in key order — iterate only `find`s
// then inserts), so std::map -> std::unordered_map is bit-exact (the selection iterates visit_order, NOT
// the children map). Same boost hash_combine mix as Gumbel's GBeliefChildKeyHash (shift+golden-ratio, no
// XOR-fold cancellation); correctness rests on the tuple's operator== on a bucket collision, not on the
// hash being injective.
struct ISMCTSChildKeyHash {
    [[nodiscard]] std::size_t operator()(const std::tuple<SlotIndex, BeliefKey>& k) const noexcept {
        auto mix = [](std::size_t& h, std::size_t v) {
            h ^= v + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
        };
        std::size_t h = 0;
        const BeliefKey& bk = std::get<1>(k);
        // std::hash ACL: the per-field static_cast<size_t> mixing is a hash REGISTER, kept raw (the
        // intended avalanche, not a domain magnitude). The action slot crosses out of its SlotIndex domain
        // here via .value() — the one named unwrap into the raw hash word (BeliefKey fields are already raw).
        mix(h, static_cast<std::size_t>(static_cast<uint32_t>(std::get<0>(k).value())));  // action slot
        mix(h, static_cast<std::size_t>(static_cast<uint32_t>(std::get<0>(bk))));         // belief count
        mix(h, static_cast<std::size_t>(std::get<1>(bk)));                                // first world id
        mix(h, static_cast<std::size_t>(std::get<2>(bk)));                                // last world id
        return h;
    }
};

// One information-set node. Per-action statistics (reward sum, selection count n_j, availability
// count n'_j) aggregated over the whole information set — the ISMCTS contract. reward/visits/avail are
// DENSE per-slot vectors (sized n_slots, zero-initialized at node creation), indexed by action SLOT —
// REPLACING the former std::map<int,.> (the same dense conversion as GumbelNode's W/N; the std::map's
// per-action _Rb_tree_increment is the cost). The conversion is BYTE-IDENTICAL because the selection
// (ucb_select) and the most-visited final iterate `visit_order` (insertion order), NEVER the maps in
// key order; a slot's "presence" in a map is exactly "its dense count > 0" (visits/avail only increment
// from the zero-init, so 0 <=> absent — preserving avail.get(a, n_j)'s present?value:default and the
// first-insertion-into-visits test that appends to visit_order). Children are keyed by (action,
// belief_key): an action's observation outcome under the active determinization routes WHICH
// successor-belief child the simulation continues from, but does NOT split the action's bandit statistics
// (mirrors ismcts.py's _Node).
struct ISMCTSNode {
    std::vector<double> reward;              // (n_slots,) action-slot -> summed playout return (0 absent)
    std::vector<VisitCount> visits;          // (n_slots,) action-slot -> times selected n_j (0 absent)
    std::vector<VisitCount> avail;           // (n_slots,) action-slot -> times available n'_j (0 absent)
    std::vector<SlotIndex> visit_order;      // action-slots in INSERTION order (first-wins tie source)
    // (action-slot, belief_key) -> child node-arena index. A find/insert-only transposition table (never
    // iterated in key order), so an unordered_map is bit-exact with the former std::map — see
    // ISMCTSChildKeyHash. The value is a node-ARENA index (a container position into `nodes`), kept raw int.
    std::unordered_map<std::tuple<SlotIndex, BeliefKey>, int, ISMCTSChildKeyHash> children;

    // Construct with reward/visits/avail sized to the slot space, zero-initialized (the absent semantics:
    // dense count == 0 <=> the slot has not been touched). `n_slots` is a SlotCount (the count that BOUNDS
    // the SlotIndex these vectors are indexed by); .value() crosses it into the container .size() ACL.
    ISMCTSNode() = default;
    explicit ISMCTSNode(SlotCount n_slots)
        : reward(static_cast<size_t>(n_slots.value()), 0.0),
          visits(static_cast<size_t>(n_slots.value()), VisitCount{0}),
          avail(static_cast<size_t>(n_slots.value()), VisitCount{0}) {}
};

// Single-Observer ISMCTS as a pluggable Policy. Construction takes the scalar config (iterations,
// c, max_depth) and the simulation/leaf base Policy (defaults to GreedyStopBase), mirroring
// ISMCTSPolicy.__init__.
class ISMCTSPolicy final : public Policy {
  public:
    explicit ISMCTSPolicy(const ISMCTSConfig& cfg = {}, const Policy* base = nullptr);

    // The Policy contract. Builds the production ISMCTSSource off `rng` + the GreedyStopBase
    // base_value leaf and runs `iterations` determinized walks from the current observed state,
    // returning the most-visited root action (mirrors ismcts.py's decide). λ is the live Dinkelbach
    // penalty threaded through every score (P4).
    Action decide(const Environment& env, const Loc& loc, const Belief& bw,
                  const CollectedSet& collected, double lam, std::mt19937_64& rng) const override;

    // The pure search core, parameterized by an injected ISMCTSSource (the seam the logic check
    // exploits). Runs the `iterations` determinized walks and returns the selected root action
    // (most-visited, first-wins tie; TERMINATE if nothing was tried). Mirrors ismcts.py's decide's
    // loop + final, exposed for the logic-check fixture so the selection logic is validated
    // independent of RNG.
    [[nodiscard]] Action run_search(const Environment& env, const Loc& loc,
                                    const Belief& bw, const CollectedSet& collected,
                                    double lam, ISMCTSSource& src) const;

    const ISMCTSConfig& config() const { return cfg_; }

  private:
    // One determinized iteration: selection + expansion + simulation + backprop in fixed `world`.
    // Returns the λ-penalized return from `nodes[node]` onward (mirrors ismcts.py's _iterate).
    // `nodes` is the node arena (a flat vector so child nodes are stable across reallocation; the
    // Python dict-of-_Node is here an index into this arena).
    // `node` is a node-ARENA index (a container position into `nodes`, kept raw int); `world` a concrete
    // World mask; `depth` the descent ply (PlyDepth, [0, max_depth]).
    double iterate(const Environment& env, std::vector<ISMCTSNode>& nodes, int node, const Loc& loc,
                   const Belief& bw, const CollectedSet& collected, World world,
                   double lam, ISMCTSSource& src, PlyDepth depth) const;

    // Subset-armed UCB1 selection (eq. 7): exploit = reward[a]/n_j; explore =
    // c·sqrt(log(navail)/n_j) if navail>1 else c, navail = avail.get(a, n_j); strict `>` first-wins
    // over INSERTION order (mirrors _ucb_select). Returns the selected action SLOT (a SlotIndex).
    [[nodiscard]] SlotIndex ucb_select(const ISMCTSNode& node) const;

    ISMCTSConfig cfg_;
    GreedyStopBase default_base_;
    const Policy* base_;  // the simulation/leaf base (default_base_ unless overridden)
};

}  // namespace chocofarm
