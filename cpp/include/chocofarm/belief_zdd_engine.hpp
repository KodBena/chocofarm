// cpp/include/chocofarm/belief_zdd_engine.hpp
// Purpose: the hand-rolled zero-suppressed decision-diagram (ZDD) ENGINE behind the maintained
//   belief-as-diagram arm (the §B.4(b) graduation of belief_features_and_decision_diagram_note.md
//   Part B; docs/design/cpp-belief-zdd-onramp.md). This is the §B.4(a) probe engine
//   (origin/zdd-onramp-probe cpp/probe/belief_zdd.{hpp,cpp}) PROMOTED out of the probe TU into a
//   belief module so it can be the search's third belief representation behind the env seam — the
//   §11 scope note's "promote when B.4(b) graduates" trigger. It is compiled ONLY when the opt-in
//   CHOCO_BELIEF_ZDD CMake option is ON (the default build never sees this header — env.hpp includes it
//   only under #ifdef CHOCO_BELIEF_ZDD, at global scope, and env_zdd.cpp gets it transitively via
//   env.hpp), so a WIP arm can never red the default runner build (the same scope discipline the probe
//   honored).
//
//   WHAT IS NEW vs the FEATURE-TIME probe (§B.4(a)): the probe built Z from an explicit `bw` once,
//   then queried — it never MAINTAINED the belief as a ZDD. B.4(b) maintains it: the env filters
//   (filter_treasure / filter_detector) become ZDD RESTRICT ops on the diagram (a new root in the
//   SAME arena), NOT a rebuild from a materialized world list. So this header adds:
//     - `restrict_var(t, present)`  — the subfamily WITH (present) / WITHOUT (absent) variable t
//                                     (the filter_treasure twin: a treasure's single-bit cover).
//     - `restrict_cover(mask, positive)` — the subfamily where the cover-disjunction over `mask`
//                                     HOLDS (positive: ≥1 bit of mask set) or FAILS (negative: none
//                                     set — the disjoint subfamily the §8 disjoint-count characterizes).
//                                     The filter_detector twin (a face's cover bitmask is the mask).
//   Both are memoized ZDD applies (like zunion), funnel through `mk`, allocate no world list, and
//   leave `members(restrict(Z,...))` set-equal to the flat filter's kept world-set (the A/B nets it).
//
//   LIFETIME / VALUE SEMANTICS (the B.4(b) realization of §2.4's per-belief throwaway arena): the
//   engine is a SELF-CONTAINED, COPYABLE value — `nodes_` + `unique_` + `root_` + `n_` are all member
//   state, so a BeliefDiagram copies cleanly (the per-node-step descent copy the search makes) and is
//   thread-safe by construction (no shared mutable table across threads — each thread's beliefs are
//   independent, exactly like the bitset arm's inline array). The COPIED VALUE is the COMPACTED REDUCED
//   diagram: each mutation (the bw ctor + every restrict_*) ends in compact(), which gc's `nodes_` to
//   the nodes reachable from root_ (a restrict strands the pre-restrict sub-DAG — without the gc those
//   dead nodes accumulate across the descent) and CLEARS `unique_` (the hash-cons is transient build/
//   restrict scratch, seeded from the live nodes at mutation entry, NOT carried in the value — the
//   C(N,K) full_belief() build seeds it huge). So between ops the stored value holds only a small
//   compacted `nodes_` (O(|Z|)) + an empty hash-cons, and a value-copy is O(|Z|), not O(full-build-arena)
//   — the fix for the measured per-descent-copy OOM at the production search config. compact() renumbers
//   in POST-ORDER, so the id-monotonic invariant (child id < parent id) — what makes the ascending/
//   descending-id query loops valid — is preserved, and canonicity + the member ORDER (var/lo/hi-
//   structural, not raw ids) are invariant under the renumber. Through a search descent the arena stays
//   small precisely when |Z| ≪ nb — the B.4(b) win.
//
//   Counts are returned as int64_t (matching features.cpp's int64_t accumulators) so the
//   comparison-and-cast path to BeliefFeatures is bit-identical (the §B.2/§B.3 logic invariant).
//
//   Correctness traps honored (design §13): the ZDD reduction rule is hi==BOT zero-suppression (NOT
//   the BDD lo==hi rule); var_of returns the +inf sentinel n_ for BOTH terminals; NO 2^skip factor
//   anywhere; a LOSSLESS hash-cons key (struct {var,lo,hi} + mixing hash + ==, never an XOR-folded
//   packed int).
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

// The LOSSLESS hash-cons key (design §13 trap 4): an exact struct, NEVER an XOR-folded packed integer
// (a collision merges distinct nodes, faking compression). operator== is the exact field test; the
// hash is a splitmix-style mix of three exact fields (injective inputs, no field cancellation).
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

// The typed value seam (design §3) — now a COPYABLE belief value (B.4(b)). One BeliefDiagram holds the
// full per-belief arena; a copy duplicates the arena (the search's descent-local copy). No node id ever
// crosses the boundary; only the constructor + value-returning queries + the in-place restrict ops are
// public. This is the boundary at which a CUDD/Sylvan/SapporoBDD engine would swap (§B.5).
class BeliefDiagram {
  public:
    // Build Z = the family of EXACTLY the worlds in `bw`, over N variables. `bw` may be unsorted; the
    // family is the SET of distinct worlds. Empty bw -> Z = BOT. PRECONDITION: bw is duplicate-free
    // (design §13 trap 8) — the env's full_belief() (worlds(), distinct by construction) and every
    // filtered descendant satisfy it.
    BeliefDiagram(std::span<const uint32_t> bw, int N);

    // Default-construct an EMPTY diagram over 0 variables (Z = BOT) — present only so the value type is
    // regular (default-constructible / assignable) for storage; the env always uses the bw ctor.
    BeliefDiagram() { reset(0); }

    // Value semantics: the arena (nodes_ + unique_ + root_ + n_) is plain member state, so the
    // compiler-generated copy/move/assign duplicate it correctly. Explicitly defaulted for clarity.
    BeliefDiagram(const BeliefDiagram&) = default;
    BeliefDiagram(BeliefDiagram&&) = default;
    BeliefDiagram& operator=(const BeliefDiagram&) = default;
    BeliefDiagram& operator=(BeliefDiagram&&) = default;

    // ---- the maintained-belief queries (the seam's reads) ----
    [[nodiscard]] int64_t count() const;                  // cardinality == nb (§5.1)
    [[nodiscard]] int node_count() const;                 // |Z| : reachable INTERNAL nodes (§5.3)
    [[nodiscard]] std::vector<uint32_t> members() const;  // enumerate Z's worlds, in ZDD canonical order
    [[nodiscard]] int n() const { return n_; }            // the variable universe size (N)
    // The r-th member in the ZDD's CANONICAL (DFS lo-then-hi) order, via a weighted descent over subtree
    // counts (O(|Z|+depth), no materialization of all members). This order is NOT worlds()-rank order, so
    // member_at_rank(r) != the flat bw[r] — the deliberate SAMPLING RE-BASELINE of the ZDD arm (counts and
    // features stay bit-exact; only the world a scripted/sampled rank resolves to differs). PRECONDITION:
    // 0 <= r < count() (the caller — env.cpp's rank_or_abort twin — fail-loud aborts otherwise).
    [[nodiscard]] uint32_t member_at_rank(int64_t r) const;

    // ---- Stage-2 counting queries (the §A.4 Phase-1 integer outputs) ----
    [[nodiscard]] std::vector<int64_t> all_marginals() const;  // bit_cnt[t] for all t (§9)
    [[nodiscard]] std::vector<int64_t> all_detector_counts(    // det_cnt[j] for all j (§8)
        std::span<const uint32_t> masks) const;

    // ---- the maintained-belief FILTERS (the B.4(b) restrict ops — in place, no rebuild) ----
    // restrict_var(t, present): replace root_ with the subfamily of Z that HAS variable t set
    // (present=true) or does NOT (present=false). The filter_treasure twin. Memoized; funnels through mk.
    void restrict_var(int t, bool present);
    // restrict_cover(mask, positive): replace root_ with the subfamily where the cover-disjunction over
    // `mask` HOLDS (positive=true: the member sets ≥1 bit of mask) or FAILS (positive=false: the member
    // sets NONE of mask's bits — the disjoint subfamily). The filter_detector twin (a face's cover mask).
    void restrict_cover(uint32_t mask, bool positive);

  private:
    // The COMPACTED reduced diagram is the stored/copied value (the per-descent-copy OOM fix). Between
    // operations `nodes_` holds ONLY BOT, TOP and the nodes reachable from `root_` (dead nodes a restrict
    // strands are gc'd), and `unique_` is EMPTY (the hash-cons is transient build/restrict scratch, NOT
    // carried in the value). So a value-copy is O(|Z|) (|Z| ≪ nb), not O(full-build-arena: the C(20,5)
    // build seeds `unique_` huge and `nodes_` never compacted across restricts — the measured OOM). Each
    // mutation (the bw ctor + every restrict_*) seeds `unique_` from the live `nodes_` at entry (so mk
    // still hash-conses canonically), then compact()s + clears `unique_` at exit. Canonicity + the
    // id-monotonic invariant (child id < parent id) are PRESERVED — compact() renumbers in POST-ORDER so
    // the ascending/descending-id query sweeps (count/below/up in member_at_rank/all_marginals) stay valid;
    // the member ORDER (var/lo/hi-structural, lo-before-hi DFS) is invariant under renumbering.
    std::vector<ZNode> nodes_;
    std::unordered_map<NodeKey, uint32_t, NodeKeyHash> unique_;  // hash-cons — TRANSIENT (empty between ops)
    uint32_t root_ = BOT;
    int n_ = 0;

    void reset(int N);  // seed nodes_ with BOT/TOP, clear unique_, root_=BOT, n_=N

    // gc `nodes_` to the reachable-only reduced diagram (BOT, TOP + everything reachable from root_),
    // renumbered in POST-ORDER (children before parents -> id-monotonicity preserved), remapping root_ and
    // every lo/hi edge; then CLEAR `unique_`. Asserts count() is unchanged (ADR-0002 fail-loud — compact
    // must preserve the family). Run at the end of every mutation so the stored/copied value is the small
    // compacted diagram with an empty hash-cons.
    void compact();
    // Repopulate `unique_` from the live `nodes_` (the inverse of compact()'s clear) so mk hash-conses
    // against the existing canonical nodes at the start of a mutation. O(|nodes_|).
    void seed_unique();

    // var_of(id) returns n_ (= a value strictly greater than every real var 0..n_-1) for BOTH terminals,
    // so a terminal is always "below" every variable in the union/restrict var-comparison arms. NEVER
    // read nodes_[id].var for id < 2 directly (design §13 trap 2).
    [[nodiscard]] int32_t var_of(uint32_t id) const { return id < 2 ? n_ : nodes_[id].var; }

    uint32_t mk(int32_t var, uint32_t lo, uint32_t hi);  // the ONLY internal-node creator
    uint32_t single(uint32_t w);                         // one world as a chain
    uint32_t zunion(uint32_t a, uint32_t b,
                    std::unordered_map<uint64_t, uint32_t>& umemo);  // memoized union (build apply)

    // The non-constructing disjoint-count recursion (§8): members of the subfamily at z that set NONE
    // of mask's bits. Calls no mk; per-call scratch (memo) sized once at call start.
    [[nodiscard]] int64_t disjoint_count(uint32_t z, uint32_t mask,
                                         std::vector<int64_t>& memo) const;

    // The restrict applies (the B.4(b) op the probe lacked). Each is a memoized recursion that funnels
    // node creation through mk; the memo is keyed by node id (the restrict parameters are fixed within a
    // call). They RETURN the new sub-root; the public restrict_* set root_ to the result.
    //   with_var(z, t)    -> subfamily of z that SETS variable t (t kept present in every member).
    //   without_var(z, t) -> subfamily of z that does NOT set variable t.
    //   cover_hold(z, mask) -> subfamily of z that sets ≥1 of mask's bits (the disjunction HOLDS).
    //   cover_fail(z, mask) -> subfamily of z that sets NONE of mask's bits (the disjoint subfamily).
    // cover_hold and cover_fail PARTITION z (every member sets ≥1 mask bit or none), so
    // count(cover_hold)+count(cover_fail)==count(z) — an invariant the A/B nets.
    uint32_t with_var(uint32_t z, int t, std::unordered_map<uint64_t, uint32_t>& memo);
    uint32_t without_var(uint32_t z, int t, std::unordered_map<uint64_t, uint32_t>& memo);
    uint32_t cover_hold(uint32_t z, uint32_t mask, std::unordered_map<uint64_t, uint32_t>& memo);
    uint32_t cover_fail(uint32_t z, uint32_t mask, std::unordered_map<uint64_t, uint32_t>& memo);
};

}  // namespace chocofarm::beliefzdd
