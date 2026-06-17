// cpp/probe/belief_zdd.hpp
// Purpose: the typed BeliefDiagram VALUE-SEAM (ADR-0012 P3 one-owner / P9 — no node ids cross the
//   boundary) for the §B.4(a) belief decision-diagram on-ramp
//   (belief_features_and_decision_diagram_note.md Part B; docs/design/cpp-belief-zdd-onramp.md).
//
//   A belief is a FAMILY OF SUBSETS over the N-treasure universe (each world a K-of-N subset, a
//   uint32 bitmask). This wraps a hand-rolled, zero-suppressed binary decision diagram (ZDD) — the
//   representation specialized for SPARSE sets of subsets (§B.1) — behind a value seam: only the
//   constructor (build Z from `bw`) and value-returning queries are public. No raw node id ever
//   escapes; this is the boundary at which a CUDD/Sylvan/SapporoBDD engine would later be swapped
//   (§B.5) without touching callers. The ZNode struct, the var() sentinel, and `mk` are private —
//   the engine LIVES IN THE PROBE'S OWN TU (this header + belief_zdd.cpp), NOT in chocofarm_core
//   (§11 scope discipline: a WIP engine must not be able to red the runner build).
//
//   Counts are returned as int64_t (matching features.cpp's int64_t accumulators) so the
//   comparison-and-cast path to BeliefFeatures is bit-identical (the §B.2/§B.3 logic invariant).
//
//   Correctness traps honored (design §13): the ZDD reduction rule is hi==BOT zero-suppression (NOT
//   the BDD lo==hi rule); var_of returns the +inf sentinel n_ for BOTH terminals; NO 2^skip factor
//   anywhere (zero-suppression makes a skipped var contribute 1, not 2); a LOSSLESS hash-cons key
//   (struct {var,lo,hi} + mixing hash + ==, never an XOR-folded packed int); one throwaway arena per
//   belief, queries non-constructing.
//
// Public Domain (The Unlicense).
#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <unordered_map>
#include <vector>

namespace chocofarm::beliefzdd {

// The two terminals (the central ZDD-vs-BDD distinction, §2.1):
//   BOT = the empty family {}      (no subsets) — count(BOT) = 0
//   TOP = the family {emptyset}    (exactly the one empty subset) — count(TOP) = 1
// These are NOT a BDD's 0/1; confusing them is the classic ZDD bug.
inline constexpr uint32_t BOT = 0;
inline constexpr uint32_t TOP = 1;

// A ZDD node. `var` is the treasure id (== ZDD variable); along any root->terminal path var strictly
// increases. lo = the 0-edge ("var absent"), hi = the 1-edge ("var present"). nodes_[0]=BOT,
// nodes_[1]=TOP carry var=n_ as a sentinel filler (never read via the stored field — see var_of).
struct ZNode {
    int32_t var;
    uint32_t lo;
    uint32_t hi;
};

// The LOSSLESS hash-cons key (design §13 trap 4): an exact struct, NEVER an XOR-folded packed
// integer (a collision merges distinct nodes, faking compression — and faithful-rep would NOT catch
// an under-count of |Z| on disjoint paths). operator== is the exact field test; the hash is a
// splitmix-style mix of three exact fields (injective inputs, no field cancellation).
struct NodeKey {
    int32_t var;
    uint32_t lo;
    uint32_t hi;
    bool operator==(const NodeKey&) const = default;
};
struct NodeKeyHash {
    [[nodiscard]] std::size_t operator()(const NodeKey& k) const noexcept {
        uint64_t h = static_cast<uint64_t>(static_cast<uint32_t>(k.var));
        h = h * 0x9E3779B97F4A7C15ull + k.lo;
        h = (h ^ (h >> 30)) * 0xBF58476D1CE4E5B9ull + k.hi;
        h ^= h >> 27;
        return static_cast<std::size_t>(h);
    }
};

// The typed value seam (design §3). One BeliefDiagram per belief, constructed then discarded (the
// per-belief throwaway arena, §2.4 — this is what makes the id-monotonic topological order valid and
// safe; the queries never call mk, so they cannot mutate nodes_ mid-sweep).
class BeliefDiagram {
  public:
    // Build Z = the family of EXACTLY the worlds in `bw`, over N variables. `bw` may be unsorted; the
    // family is the SET of distinct worlds. Empty bw -> Z = BOT. PRECONDITION: bw is duplicate-free
    // (design §13 trap 8) — the caller (the probe) asserts it before constructing.
    BeliefDiagram(std::span<const uint32_t> bw, int N);

    // ---- Stage 1: the decision-gate measurements ----
    [[nodiscard]] int64_t count() const;                  // cardinality == nb (§5.1)
    [[nodiscard]] int node_count() const;                 // |Z| : reachable INTERNAL nodes (§5.3)
    [[nodiscard]] std::vector<uint32_t> members() const;  // enumerate Z's worlds (faithful-rep, §5.2)

    // ---- Stage 2: the §B.2 counting queries (the §A.4 Phase-1 integer outputs) ----
    [[nodiscard]] std::vector<int64_t> all_marginals() const;  // bit_cnt[t] for all t (§9)
    [[nodiscard]] std::vector<int64_t> all_detector_counts(    // det_cnt[j] for all j (§8)
        std::span<const uint32_t> masks) const;

  private:
    std::vector<ZNode> nodes_;
    std::unordered_map<NodeKey, uint32_t, NodeKeyHash> unique_;  // build-time; queries don't touch it
    std::unordered_map<uint64_t, uint32_t> umemo_;              // zunion memo (build-time scratch)
    uint32_t root_ = BOT;
    int n_ = 0;

    // var_of(id) returns n_ (= a value strictly greater than every real var 0..n_-1) for BOTH
    // terminals, so a terminal is always "below" every variable in the union var-comparison arms.
    // NEVER read nodes_[id].var for id < 2 directly (design §13 trap 2).
    [[nodiscard]] int32_t var_of(uint32_t id) const { return id < 2 ? n_ : nodes_[id].var; }

    uint32_t mk(int32_t var, uint32_t lo, uint32_t hi);   // the ONLY internal-node creator
    uint32_t single(uint32_t w);                          // one world as a chain
    uint32_t zunion(uint32_t a, uint32_t b);              // memoized union (the only build apply)

    // The non-constructing disjoint-count recursion (§8): members of the subfamily at z that set NONE
    // of mask's bits. Calls no mk; per-call scratch (memo) sized once at call start.
    [[nodiscard]] int64_t disjoint_count(uint32_t z, uint32_t mask,
                                         std::vector<int64_t>& memo) const;
};

}  // namespace chocofarm::beliefzdd
