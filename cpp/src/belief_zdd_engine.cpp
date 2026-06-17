// cpp/src/belief_zdd_engine.cpp
// Purpose: the hand-rolled zero-suppressed decision-diagram (ZDD) engine impl behind the maintained
//   belief-as-diagram arm (belief_zdd_engine.hpp; the §B.4(b) graduation of
//   belief_features_and_decision_diagram_note.md Part B; docs/design/cpp-belief-zdd-onramp.md). It is
//   the §B.4(a) probe engine (origin/zdd-onramp-probe cpp/probe/belief_zdd.cpp) PROMOTED into a belief
//   module, with the build apply (mk/single/zunion) and the §5/§8/§9 queries (count/members/
//   node_count/all_marginals/all_detector_counts) UNCHANGED in math, plus the NEW B.4(b) RESTRICT ops
//   (restrict_var / restrict_cover via with_var/without_var/cover_hold/cover_fail) that maintain the
//   belief through filtering as a ZDD op rather than a rebuild.
//
//   Compiled ONLY when CHOCO_BELIEF_ZDD is ON — chocofarm_core's CMake gates this TU on the option, so
//   the default (flat+bitset) build never compiles it (scope discipline: a WIP arm cannot red the
//   default runner build). The build is otherwise self-contained.
//
//   The canonicity facts that make the integer counts a bit-exact logic invariant (design §13): fixed
//   variable order 0<...<N-1, var strictly increasing root->terminal; zero-suppression on hi==BOT (the
//   ZDD rule, NEVER lo==hi the BDD rule); a lossless hash-cons merge; NO 2^skip factor; id-monotonic
//   arena (every child created before its parent), so an ascending-id loop is a valid bottom-up order
//   and a descending-id loop a valid top-down order; queries are non-constructing (never call mk).
//   The restrict ops DO call mk (they ADD nodes), but they read only PRE-EXISTING node ids as input
//   (lo/hi of original nodes) and mk only APPENDS (never modifies/removes), so the input ids stay
//   stable and the memo keyed by input id is valid; the id-monotonic invariant is preserved.
//
// Public Domain (The Unlicense).
#include "chocofarm/belief_zdd_engine.hpp"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace chocofarm::beliefzdd {

void BeliefDiagram::reset(int N) {
    nodes_.clear();
    unique_.clear();
    nodes_.push_back(ZNode{N, 0, 0});  // BOT (var=N sentinel, unused)
    nodes_.push_back(ZNode{N, 0, 0});  // TOP
    root_ = BOT;
    n_ = N;
}

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

// One world as a chain (§4.1): the family { set-bits-of-w }. Descending t so children (larger var) are
// made first (id-monotonicity). A K-of-N world yields a K-node chain — the absent vars are elided by
// zero-suppression (the sparse-regime compaction, §B.1).
uint32_t BeliefDiagram::single(uint32_t w) {
    uint32_t cur = TOP;  // {emptyset}: the deepest tail
    for (int t = n_ - 1; t >= 0; --t)
        if ((w >> t) & 1u) cur = mk(t, BOT, cur);  // var t PRESENT: lo=BOT, hi=cur (absent vars elided)
    return cur;
}

// The memoized union — the only apply the build needs (§4.2). Terminal guards come BEFORE any var_of
// dereference; the var-comparison arms rely on var_of(TOP)==n_ (below all vars). The umemo is build-
// scratch passed in (so it is not retained on the value-type arena across copies).
uint32_t BeliefDiagram::zunion(uint32_t a, uint32_t b, std::unordered_map<uint64_t, uint32_t>& umemo) {
    if (a == BOT) return b;
    if (b == BOT) return a;
    if (a == b) return a;
    if (a > b) std::swap(a, b);  // commutative -> canonical memo key
    uint64_t key = (static_cast<uint64_t>(a) << 32) | b;
    if (auto it = umemo.find(key); it != umemo.end()) return it->second;
    uint32_t r;
    int32_t va = var_of(a), vb = var_of(b);  // TOP -> n_ (below all vars), via the sentinel
    if (va == vb) {  // both internal at the same var (TOP==TOP was caught by a==b above)
        r = mk(va, zunion(nodes_[a].lo, nodes_[b].lo, umemo), zunion(nodes_[a].hi, nodes_[b].hi, umemo));
    } else if (va < vb) {  // a internal & shallower: b rides a's lo-edge (b has no var va)
        r = mk(va, zunion(nodes_[a].lo, b, umemo), nodes_[a].hi);
    } else {  // vb < va, symmetric
        r = mk(vb, zunion(a, nodes_[b].lo, umemo), nodes_[b].hi);
    }
    umemo.emplace(key, r);
    return r;
}

// Build-from-worlds: a simple left fold of singleton chains + memoized union (§4.3). The empty-belief
// edge case (build({})==BOT) is handled by the initial z=BOT and the a==BOT/b==BOT guards.
BeliefDiagram::BeliefDiagram(std::span<const uint32_t> bw, int N) {
    reset(N);
    std::unordered_map<uint64_t, uint32_t> umemo;  // build-scratch (not retained on the arena)
    uint32_t z = BOT;
    for (uint32_t w : bw) z = zunion(z, single(w), umemo);
    root_ = z;
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
// size == count(). The order is the ZDD's CANONICAL member order (a DFS of the diagram), which is NOT
// the env's worlds()-rank order — the source of the ZDD arm's sampling RE-BASELINE (the counts/features
// are bit-exact; the r-th member != the flat bw[r]).
std::vector<uint32_t> BeliefDiagram::members() const {
    std::vector<uint32_t> out;
    auto rec = [&](auto&& self, uint32_t id, uint32_t acc) -> void {
        if (id == BOT) return;
        if (id == TOP) { out.push_back(acc); return; }
        const ZNode& nd = nodes_[id];
        self(self, nd.lo, acc);                              // var absent
        self(self, nd.hi, acc | (uint32_t{1} << nd.var));   // var present
    };
    rec(rec, root_, 0);
    return out;
}

// member_at_rank(r) — the r-th member in the ZDD's CANONICAL DFS order (lo-subtree before hi-subtree),
// via a weighted descent over subtree counts: at a node the lo-subtree's count(lo) members come first
// (var absent), then the hi-subtree's (var present). below[] is the per-node cardinality (ascending-id
// bottom-up, like count()). This order is the SAME as members()'s DFS order but found without
// materializing all members. It is NOT worlds()-rank order — the ZDD arm's deliberate sampling
// re-baseline (the r-th ZDD member != the flat bw[r]); the counts/features stay bit-exact regardless.
uint32_t BeliefDiagram::member_at_rank(int64_t r) const {
    if (r < 0) return 0xFFFFFFFFu;  // NEGATIVE r is out of range too (the bitset twin catches both ends);
                                    // the caller (env_zdd.cpp member_or_abort) fail-loud aborts on the sentinel
    std::vector<int64_t> below(nodes_.size(), 0);
    below[TOP] = 1;
    for (uint32_t id = 2; id < nodes_.size(); ++id)
        below[id] = below[nodes_[id].lo] + below[nodes_[id].hi];
    // PRECONDITION 0 <= r < count(): the loud-abort arm is the caller's (env_zdd.cpp). Here we descend
    // (over-range r descends past every member to BOT and returns the sentinel; under-range is caught above).
    uint32_t z = root_;
    uint32_t acc = 0;
    while (z >= 2) {  // internal node
        const ZNode& nd = nodes_[z];
        const int64_t lo_cnt = below[nd.lo];
        if (r < lo_cnt) {
            z = nd.lo;                              // var absent: the r-th member is in the lo-subtree
        } else {
            r -= lo_cnt;
            acc |= (uint32_t{1} << nd.var);         // var present
            z = nd.hi;
        }
    }
    // z is now a terminal: TOP (acc is the member) if r resolved, else BOT (invariant violation).
    return z == TOP ? acc : 0xFFFFFFFFu;  // sentinel on BOT (r out of range) — caller fail-loud aborts
}

// node_count() — |Z|: reachable INTERNAL nodes from root_ (terminals excluded so it is
// apples-to-apples with nb). §5.3. (Counts only nodes reachable from the LIVE root_ — dead nodes left
// by a restrict are not reachable and so not counted.)
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
// below[u]=count(u) (backward, ascending id); up[u]=path-counts to u (forward, descending id). A member
// sets bit t iff its path takes the hi-edge of the unique node with var==t, so bit_cnt[t]=Σ over nodes
// u with var(u)==t of up[u]*below[hi(u)]. NO 2^skip factor. NB: the backward sweep is over ALL nodes_
// (ascending id) but the forward sweep seeds up[root_]=1 only, so dead nodes (up==0) contribute nothing.
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
// from CURRENT nodes_ (no mutation — queries non-constructing). The memo is sized over ALL nodes_ (dead
// ones included) but only the live sub-DAG from root_ is ever visited; the extra entries are harmless.
std::vector<int64_t> BeliefDiagram::all_detector_counts(std::span<const uint32_t> masks) const {
    const int64_t nb = count();
    std::vector<int64_t> det(masks.size(), 0);
    for (size_t j = 0; j < masks.size(); ++j) {
        std::vector<int64_t> memo(nodes_.size(), -1);
        det[j] = nb - disjoint_count(root_, masks[j], memo);
    }
    return det;
}

// ============================ the B.4(b) RESTRICT ops ============================
// Each restrict is a memoized recursion over the LIVE sub-DAG that funnels node creation through mk.
// The memo is keyed by INPUT node id (the restrict params are fixed within a call); the input ids are
// pre-existing (lo/hi of original nodes) and mk only APPENDS, so they stay stable through the call.

// with_var(z, t): keep members of z that SET treasure t. A member sets t iff its path takes the hi-edge
// of the (unique) node with var==t. v>t means t was SKIPPED (absent) -> drop; v==t keeps only the hi
// branch (t-present), re-wrapping mk(t, BOT, hi) so t stays recorded; v<t recurses both (t may be set
// in either subtree).
uint32_t BeliefDiagram::with_var(uint32_t z, int t, std::unordered_map<uint64_t, uint32_t>& memo) {
    if (z == BOT) return BOT;
    if (z == TOP) return BOT;  // reached terminal without var t -> t absent -> drop
    if (auto it = memo.find(z); it != memo.end()) return it->second;
    int32_t v = nodes_[z].var;
    uint32_t r;
    if (v > t) {
        r = BOT;  // t skipped on this path (zero-suppression) -> t absent -> drop
    } else if (v == t) {
        r = mk(t, BOT, nodes_[z].hi);  // keep ONLY the t-present branch; mk does NOT suppress on lo==BOT
    } else {                            // v < t: recurse both (re-read lo/hi BEFORE mk grows nodes_)
        uint32_t lo = nodes_[z].lo, hi = nodes_[z].hi;
        r = mk(v, with_var(lo, t, memo), with_var(hi, t, memo));
    }
    memo.emplace(z, r);
    return r;
}

// without_var(z, t): keep members of z that do NOT set treasure t. v>t means t was skipped (absent) ->
// keep the whole subtree; v==t keeps the lo branch (t-absent) directly (everything below it is var>t,
// none is t); v<t recurses both.
uint32_t BeliefDiagram::without_var(uint32_t z, int t, std::unordered_map<uint64_t, uint32_t>& memo) {
    if (z == BOT) return BOT;
    if (z == TOP) return TOP;  // emptyset does not set t -> keep
    if (auto it = memo.find(z); it != memo.end()) return it->second;
    int32_t v = nodes_[z].var;
    uint32_t r;
    if (v > t) {
        r = z;  // t skipped -> already absent -> keep whole subtree
    } else if (v == t) {
        r = nodes_[z].lo;  // keep ONLY the t-absent branch (its members already omit t)
    } else {               // v < t: recurse both
        uint32_t lo = nodes_[z].lo, hi = nodes_[z].hi;
        r = mk(v, without_var(lo, t, memo), without_var(hi, t, memo));
    }
    memo.emplace(z, r);
    return r;
}

// cover_hold(z, mask): keep members of z that set ≥1 bit of mask (the disjunction HOLDS). v in mask:
// the hi-edge SETS bit v ∈ mask, so the disjunction holds regardless of the rest -> keep hi WHOLE; the
// lo-edge must satisfy via a deeper mask bit -> recurse. v not in mask: recurse both.
uint32_t BeliefDiagram::cover_hold(uint32_t z, uint32_t mask, std::unordered_map<uint64_t, uint32_t>& memo) {
    if (z == BOT) return BOT;
    if (z == TOP) return BOT;  // emptyset sets no bit -> disjunction fails -> drop
    if (auto it = memo.find(z); it != memo.end()) return it->second;
    int32_t v = nodes_[z].var;
    uint32_t lo = nodes_[z].lo, hi = nodes_[z].hi;
    uint32_t r;
    if (((mask >> v) & 1u) != 0) {
        r = mk(v, cover_hold(lo, mask, memo), hi);  // v satisfies -> keep hi whole
    } else {
        r = mk(v, cover_hold(lo, mask, memo), cover_hold(hi, mask, memo));
    }
    memo.emplace(z, r);
    return r;
}

// cover_fail(z, mask): keep members of z that set NONE of mask's bits (the disjoint subfamily). v in
// mask: the hi-edge sets v ∈ mask -> violates disjointness -> drop hi (mk would zero-suppress hi==BOT
// anyway, so just return the recursed lo). v not in mask: recurse both.
uint32_t BeliefDiagram::cover_fail(uint32_t z, uint32_t mask, std::unordered_map<uint64_t, uint32_t>& memo) {
    if (z == BOT) return BOT;
    if (z == TOP) return TOP;  // emptyset is disjoint from everything -> keep
    if (auto it = memo.find(z); it != memo.end()) return it->second;
    int32_t v = nodes_[z].var;
    uint32_t lo = nodes_[z].lo, hi = nodes_[z].hi;
    uint32_t lo2 = cover_fail(lo, mask, memo);
    uint32_t r = ((mask >> v) & 1u) != 0 ? lo2 : mk(v, lo2, cover_fail(hi, mask, memo));
    memo.emplace(z, r);
    return r;
}

void BeliefDiagram::restrict_var(int t, bool present) {
    std::unordered_map<uint64_t, uint32_t> memo;  // keyed by input node id (params fixed for the call)
    root_ = present ? with_var(root_, t, memo) : without_var(root_, t, memo);
}

void BeliefDiagram::restrict_cover(uint32_t mask, bool positive) {
    std::unordered_map<uint64_t, uint32_t> memo;
    root_ = positive ? cover_hold(root_, mask, memo) : cover_fail(root_, mask, memo);
}

}  // namespace chocofarm::beliefzdd
