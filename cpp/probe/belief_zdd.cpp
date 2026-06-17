// cpp/probe/belief_zdd.cpp
// Purpose: the hand-rolled zero-suppressed decision-diagram (ZDD) ENGINE behind the BeliefDiagram
//   value seam (belief_zdd.hpp) — the §B.4(a) on-ramp engine
//   (belief_features_and_decision_diagram_note.md §B.1/§B.2/§B.5; docs/design/cpp-belief-zdd-onramp.md
//   §2/§4/§5/§8/§9). Implements mk / single / zunion (build) and count / members / node_count /
//   all_marginals / all_detector_counts (queries). Linked ONLY by the probe target — NOT compiled
//   into chocofarm_core (§11 scope discipline): a WIP engine must not be able to red the runner build.
//
//   The canonicity facts that make |Z| (node_count) a meaningful structural measure and the integer
//   counts a bit-exact logic invariant (design §13):
//     - fixed variable order 0<1<...<N-1, var strictly increasing root->terminal (a constexpr fact);
//     - zero-suppression: mk suppresses on hi==BOT (the ZDD rule), NEVER on lo==hi (the BDD rule);
//     - lossless hash-cons merge (NodeKey struct + mixing hash + ==), so identical (var,lo,hi) merge;
//     - NO 2^skip factor in count/below/up — a skipped var is UNCONDITIONALLY absent (factor 1);
//     - one throwaway arena per belief (id-monotonic: every child created before its parent), so an
//       ascending-id loop is a valid bottom-up order and a descending-id loop a valid top-down order;
//     - queries are non-constructing (never call mk); scratch sized once from nodes_.size().
//
// Public Domain (The Unlicense).
#include "belief_zdd.hpp"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <unordered_set>
#include <utility>
#include <vector>

namespace chocofarm::beliefzdd {

// The ONLY function that creates internal nodes — both reduction rules + lossless merge funnel here
// (mechanize > assert: canonicity is structural, unauthorable otherwise). §2.3.
uint32_t BeliefDiagram::mk(int32_t var, uint32_t lo, uint32_t hi) {
    if (hi == BOT) return lo;  // (1) ZERO-SUPPRESSION — the ZDD rule (NEVER lo==hi, the BDD rule)
    // ordering invariant: children must be strictly deeper than `var` (var_of(terminal)==n_ > var).
    assert(var_of(lo) > var && var_of(hi) > var && "mk: variable-ordering violation");
    // id-monotonicity invariant: children are created before parents, so both child ids are < the id
    // we are about to assign — what makes an ascending-id loop a valid bottom-up order (§2.4).
    assert(lo < static_cast<uint32_t>(nodes_.size()) && hi < static_cast<uint32_t>(nodes_.size()));
    NodeKey key{var, lo, hi};
    if (auto it = unique_.find(key); it != unique_.end()) return it->second;  // (2) MERGE (hash-cons)
    uint32_t id = static_cast<uint32_t>(nodes_.size());
    nodes_.push_back(ZNode{var, lo, hi});
    unique_.emplace(key, id);
    return id;
}

// One world as a chain (§4.1): the family { set-bits-of-w }. Descending t so children (larger var)
// are made first (id-monotonicity). A K-of-N world yields a K-node chain — the absent vars are
// elided by zero-suppression (the sparse-regime compaction, §B.1).
uint32_t BeliefDiagram::single(uint32_t w) {
    uint32_t cur = TOP;  // {emptyset}: the deepest tail
    for (int t = n_ - 1; t >= 0; --t)
        if ((w >> t) & 1u) cur = mk(t, BOT, cur);  // var t PRESENT: lo=BOT, hi=cur (absent vars elided)
    return cur;
}

// The memoized union — the only apply the build needs (§4.2). Terminal guards come BEFORE any var_of
// dereference; the var-comparison arms rely on var_of(TOP)==n_ (below all vars).
uint32_t BeliefDiagram::zunion(uint32_t a, uint32_t b) {
    if (a == BOT) return b;
    if (b == BOT) return a;
    if (a == b) return a;
    if (a > b) std::swap(a, b);  // commutative -> canonical memo key
    uint64_t key = (static_cast<uint64_t>(a) << 32) | b;
    if (auto it = umemo_.find(key); it != umemo_.end()) return it->second;
    uint32_t r;
    int32_t va = var_of(a), vb = var_of(b);  // TOP -> n_ (below all vars), via the sentinel
    if (va == vb) {  // both internal at the same var (TOP==TOP was caught by a==b above)
        r = mk(va, zunion(nodes_[a].lo, nodes_[b].lo), zunion(nodes_[a].hi, nodes_[b].hi));
    } else if (va < vb) {  // a internal & shallower: b rides a's lo-edge (b has no var va)
        r = mk(va, zunion(nodes_[a].lo, b), nodes_[a].hi);
    } else {  // vb < va, symmetric
        r = mk(vb, zunion(a, nodes_[b].lo), nodes_[b].hi);
    }
    umemo_.emplace(key, r);
    return r;
}

// Build-from-worlds: a simple left fold of singleton chains + memoized union (§4.3). The empty-belief
// edge case (build({})==BOT) is handled by the initial z=BOT and the a==BOT/b==BOT guards.
BeliefDiagram::BeliefDiagram(std::span<const uint32_t> bw, int N) : n_(N) {
    nodes_.push_back(ZNode{N, 0, 0});  // BOT (var=N sentinel, unused)
    nodes_.push_back(ZNode{N, 0, 0});  // TOP
    uint32_t z = BOT;
    for (uint32_t w : bw) z = zunion(z, single(w));
    root_ = z;
    // umemo_/unique_ left as-is; the per-belief arena is discarded after the queries.
}

// count() — cardinality nb (§5.1). Bottom-up via ascending-id loop (valid topo order); NO 2^skip
// factor (zero-suppression makes skipped vars absent, contributing 1, not 2 — the ZDD-vs-BDD trap).
int64_t BeliefDiagram::count() const {
    std::vector<int64_t> card(nodes_.size(), 0);
    card[BOT] = 0;
    card[TOP] = 1;
    for (uint32_t id = 2; id < nodes_.size(); ++id)
        card[id] = card[nodes_[id].lo] + card[nodes_[id].hi];
    return card[root_];  // nb <= |worlds|, fits int64 with vast headroom
}

// members() — enumerate Z's worlds (the faithful-rep witness, §5.2). Canonical -> distinct,
// size == count().
std::vector<uint32_t> BeliefDiagram::members() const {
    std::vector<uint32_t> out;
    auto rec = [&](auto&& self, uint32_t id, uint32_t acc) -> void {
        if (id == BOT) return;
        if (id == TOP) { out.push_back(acc); return; }
        const ZNode& nd = nodes_[id];
        self(self, nd.lo, acc);                                       // var absent
        self(self, nd.hi, acc | (uint32_t{1} << nd.var));            // var present
    };
    rec(rec, root_, 0);
    return out;
}

// node_count() — |Z|: reachable INTERNAL nodes from root_ (terminals excluded so it is
// apples-to-apples with nb). §5.3.
int BeliefDiagram::node_count() const {
    std::unordered_set<uint32_t> seen;
    std::vector<uint32_t> stk{root_};
    while (!stk.empty()) {
        uint32_t u = stk.back();
        stk.pop_back();
        if (u < 2 || !seen.insert(u).second) continue;  // skip terminals + already-seen
        stk.push_back(nodes_[u].lo);
        stk.push_back(nodes_[u].hi);
    }
    return static_cast<int>(seen.size());
}

// all_marginals() — bit_cnt[t] = #{worlds in Z that set bit t}, for all t in one O(|Z|+N) sweep (§9).
//   below[u] = count(u) (subtree cardinality, backward) via ascending-id loop;
//   up[u]    = path-counts to u (forward): up[root]=1; each node pushes up[u] to both children.
// A member sets bit t iff its path takes the hi-edge of the unique node with var==t it passes
// through, so bit_cnt[t] = Σ over nodes u with var(u)==t of up[u] * below[hi(u)]. NO 2^skip factor:
// a var skipped between parent and child is unconditionally absent (factor 1), not a free 0/1 choice.
// Process nodes in DESCENDING id (the natural reverse-topological order for up): up[id] is final when
// reached (every parent has a strictly smaller id, §2.4), so read its contribution THEN push down.
std::vector<int64_t> BeliefDiagram::all_marginals() const {
    const size_t M = nodes_.size();
    std::vector<int64_t> below(M, 0), up(M, 0);
    below[TOP] = 1;
    for (uint32_t id = 2; id < M; ++id)  // backward: children-before-parent (ascending id)
        below[id] = below[nodes_[id].lo] + below[nodes_[id].hi];

    up[root_] = 1;
    std::vector<int64_t> bit(n_, 0);
    for (uint32_t id = static_cast<uint32_t>(M); id-- > 2;) {  // forward + combine, descending id
        if (up[id] == 0) continue;
        const ZNode& nd = nodes_[id];
        bit[nd.var] += up[id] * below[nd.hi];  // (a) read contribution (up[id] final, below[hi] final)
        up[nd.lo] += up[id];                   // (b) push down to children
        up[nd.hi] += up[id];
    }
    return bit;
}

// disjoint_count(z, mask): members of the subfamily at z that set NONE of mask's bits (§8). At a node
// whose var is in mask, drop the hi-branch (taking that var would intersect the mask); sum the
// lo-branch always. Non-constructing (calls no mk); memo keyed by node id (mask fixed within a call).
int64_t BeliefDiagram::disjoint_count(uint32_t z, uint32_t mask, std::vector<int64_t>& memo) const {
    if (z == BOT) return 0;
    if (z == TOP) return 1;  // emptyset is disjoint from everything
    if (memo[z] >= 0) return memo[z];
    int32_t v = nodes_[z].var;
    int64_t r = disjoint_count(nodes_[z].lo, mask, memo);  // omit var: always allowed
    if (((mask >> v) & 1u) == 0)                            // var NOT in mask -> may take it
        r += disjoint_count(nodes_[z].hi, mask, memo);
    return memo[z] = r;
}

// all_detector_counts(masks) — det_cnt[j] = nb - #{worlds disjoint from mask_j} (§8). "disjoint" =
// sets none of the bits in mask_j. Cost O(|Z|) per detector, nb-independent. Per-detector memo sized
// from CURRENT nodes_, no mutation (queries non-constructing).
std::vector<int64_t> BeliefDiagram::all_detector_counts(std::span<const uint32_t> masks) const {
    const int64_t nb = count();
    std::vector<int64_t> det(masks.size(), 0);
    for (size_t j = 0; j < masks.size(); ++j) {
        std::vector<int64_t> memo(nodes_.size(), -1);
        det[j] = nb - disjoint_count(root_, masks[j], memo);
    }
    return det;
}

}  // namespace chocofarm::beliefzdd
