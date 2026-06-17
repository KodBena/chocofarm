# ZDD on-ramp design — workflow phases (2026-06-17)

The three independent proposals and the adversarial critique that fed the locked design
(`docs/design/cpp-belief-zdd-onramp.md`). Preserved per the expensive-output convention. Public Domain.

## Proposal 1

Confirmed: K=5, nD=44, C(20,5)=15504. N=20 is implied by treasures coordinates. I have all the context. Here is the complete design.

---

# Design: B.4(a) On-ramp — a minimal hand-rolled ZDD for belief features (Angle A: simplicity-first to the |Z|-vs-nb gate)

This is the complete, directly-implementable design for the B.4(a) feature-time decision diagram, staged measure-first per the note's §B.4 (`docs/../belief_features_and_decision_diagram_note.md`). It leads with the simplest representation and algorithms that get to the **|Z|-vs-nb decision gate** (Stage 1) with a *provably-correct* minimal ZDD, then carries the same primitives through to the **bit-exact all-marginals + disjoint-count harness** (Stage 2, full §B.4a).

Citations are to that note's Part B sections (§B.1 mapping, §B.2 queries-as-counts, §B.3 validation, §B.4 on-ramp + the deciding measurement, §B.5 hand-roll, §B.6 ADR hygiene).

---

## 0. The one design decision that controls everything: variable ordering and the ZDD reduction rule

Two correctness traps named up front (§B.4 "correctness traps"), because they fix every algorithm below.

### Universe and ordering
The universe is the N=20 treasure bits, **variable index = treasure id `t`** (so `x_t` is "treasure t present"). We fix the **canonical order `0 < 1 < … < 19`** (top variable = `var 0`, terminals at the bottom). A world `w` is the set `{t : (w>>t)&1}`; it always has exactly K=5 elements (the env fact — "EXACTLY 5 bits set per world"). The order is global and immutable; every node carries its `var`, and on any root→terminal path, `var` is *strictly increasing*. This is the trap: an algorithm (build, apply, count) that ever produces a child whose `var` is `<=` its parent's `var` has corrupted the diagram. We enforce it as an assertion in `mk`.

### ZDD nodes and the **zero-suppression** reduction rule (the central trap, §B.4)
A ZDD node is `(var, lo, hi)`:
- `hi` = the sub-family of subsets that **contain** `var` (with `var` removed);
- `lo` = the sub-family that **omit** `var`.

Two terminals: **`⊥` (Empty, id 0)** = the empty family `{}` (no subsets at all), and **`⊤` (Base, id 1)** = the family `{∅}` (exactly the one empty subset). These are NOT BDD's 0/1 — confusing them is the classic ZDD bug.

The ZDD reduced form has **two** rules, both mandatory in the hash-cons constructor `mk(var, lo, hi)`:
1. **Zero-suppression (the ZDD-specific rule):** if `hi == ⊥`, the node is redundant — *return `lo` directly*, do not create a node. Meaning: "no subset in this family contains `var`", so `var` carries no information and is skipped. **This is exactly the rule that makes a ZDD compress sparse families** (each world sets only 5 of 20 bits, so 15 bits are absent and get suppressed). Omitting this rule yields a correct-but-unreduced diagram with `|Z|` inflated toward `nb` — it would **falsely kill Part B** by reporting no compression. It must be present and it must be the *only* suppression rule (do not also suppress on `lo`; that is the BDD rule and is wrong for ZDD).
2. **Node sharing (hash-consing):** if a node with identical `(var, lo, hi)` already exists, return the existing id. This is what makes "reduced unique nodes" well-defined and gives `|Z|` its meaning.

A diagram built with exactly these two rules is the **canonical reduced ordered ZDD** (ROZDD) for the family under the fixed order — unique, so `|Z|` (count of reachable non-terminal unique nodes) is a faithful structural measure. The faithful-representation check (Stage 1) independently re-derives the member set and `nb` so we never trust `|Z|` without proof that the diagram *is* the belief.

---

## 1. File / target layout (mirrors `belief_sweep_oracle_check`, §B.5/B.6)

Per ADR-0012 P3 (one-owner) and the existing standalone-tool pattern. **All new code; no edits to `features.cpp` or `env.cpp`.** The ZDD lives behind a typed value seam (`BeliefZdd`, §B.6 P9 — no raw node pointers cross the boundary), header + impl in `chocofarm_core`, with one standalone probe executable and one pytest gate.

```
cpp/include/chocofarm/belief_zdd.hpp     # the BeliefZdd value-seam: the engine + all queries (header)
cpp/src/belief_zdd.cpp                    # impl (build, count, |Z|, enumerate, all-marginals, disjoint-count)
cpp/src/belief_zdd_probe.cpp             # the STANDALONE probe: Stage-1 gate + Stage-2 bit-exact harness
                                          #   (mirrors belief_sweep_oracle_check.cpp's main + arg parsing)
tests/test_cpp_runner.py                  # ADD one gate: test_cpp_belief_zdd_probe (mirrors
                                          #   test_cpp_belief_sweep_oracle exactly)
```

CMake: add `src/belief_zdd.cpp` to the `chocofarm_core` library source list (alongside `src/features.cpp`), and add **one** standalone executable mirroring `chocofarm-belief-sweep-oracle-check`:

```cmake
# The belief-ZDD probe (NOT the runner): the §B.4(a) on-ramp. STAGE 1 — builds a minimal hand-rolled ZDD
# from each belief, asserts the FAITHFUL-REPRESENTATION invariant (enumerate(Z)==bw, count(Z)==nb), and
# logs |Z| vs nb across REALISTIC observation-narrowed beliefs (the §B.4 deciding measurement). STAGE 2 —
# answers all-marginals (bit_cnt[t]) + per-detector disjoint-count (det_cnt[j]) off Z and asserts they
# EQUAL chocofarm::belief_features's integer counts, bit-exact (the §B.3 logic invariant). The decision-
# gate + bit-exact net for Part B (ADR-0011: net the diagram, don't trust it). Separate (P3, one-owner).
add_executable(chocofarm-belief-zdd-probe src/belief_zdd_probe.cpp)
target_link_libraries(chocofarm-belief-zdd-probe PRIVATE chocofarm_core)
target_compile_options(chocofarm-belief-zdd-probe PRIVATE -Wall -Wextra)
```

Why `belief_zdd.cpp` goes into `chocofarm_core` rather than living only in the probe TU: it keeps the engine reusable for the future B.4(b) graduation behind the same seam (§B.4b "behind the seam"), and lets a future unit test or the isolated bench link it the way `feature_compute.hpp` exposes `belief_features`. It pulls in nothing new — pure `<cstdint>/<vector>/<unordered_map>`.

Every file gets the ADR-0006 module-docstring header (path + purpose + Public Domain), matching the existing files.

---

## 2. The ZDD engine — node representation, unique table, terminals (`belief_zdd.hpp`)

Index-based (no pointers, no GC) — the simplest correct arena. Node ids are `uint32_t` into a flat `nodes_` vector. This is the "few hundred lines, unique-node table + memoized apply" hand-roll the note prescribes (§B.5).

```cpp
namespace chocofarm {

class BeliefZdd {                          // the typed value-seam (§B.6 P9): no node ids escape the API
  public:
    using Id = uint32_t;
    static constexpr Id kEmpty = 0;        // ⊥  : the family {}      (no subsets)
    static constexpr Id kBase  = 1;        // ⊤  : the family {∅}     (the single empty subset)

    struct Node { int var; Id lo; Id hi; };   // var-ordered; on any path var strictly increases

    explicit BeliefZdd(int n_vars);        // N = 20; reserves the two terminals

    // ---- construction (§2.1 below) ----
    Id build_from_worlds(std::span<const uint32_t> bw);   // Z = the family of the bw worlds

    // ---- structural queries (Stage 1) ----
    uint64_t count(Id z) const;                           // cardinality = nb (memoized bottom-up)
    size_t   node_count(Id z) const;                      // |Z| : reachable non-terminal unique nodes
    void     enumerate(Id z, std::vector<uint32_t>& out) const;   // every member world, as a bitmask

    // ---- the Stage-2 queries (§B.2) ----
    void all_marginals(Id z, std::span<int64_t> bit_cnt) const;          // bit_cnt[t] for ALL t, ONE sweep
    int64_t disjoint_count(Id z, uint32_t mask) const;                   // #{w in Z : (w & mask)==0}

  private:
    Id mk(int var, Id lo, Id hi);          // hash-cons + zero-suppression (the reduction rule, §0)
    std::vector<Node> nodes_;              // arena; nodes_[0]=⊥ sentinel, nodes_[1]=⊤ sentinel
    std::unordered_map<uint64_t, Id> unique_;   // (var,lo,hi) packed -> id  (node sharing)
    int n_vars_;
};
}  // namespace chocofarm
```

`mk` — the heart of correctness (both reduction rules + the ordering assert):

```cpp
BeliefZdd::Id BeliefZdd::mk(int var, Id lo, Id hi) {
    if (hi == kEmpty) return lo;                       // ZERO-SUPPRESSION: var carries no member -> skip it
    // ordering invariant: children's top var must be deeper than `var` (terminals have no var).
    assert((lo < 2 || nodes_[lo].var > var) && (hi < 2 || nodes_[hi].var > var)
           && "BeliefZdd::mk: variable-ordering violation");
    const uint64_t key = (uint64_t(var) << 56) ^ (uint64_t(lo) << 28) ^ uint64_t(hi);  // 20 vars, ids < 2^28
    if (auto it = unique_.find(key); it != unique_.end()) return it->second;  // NODE SHARING
    const Id id = static_cast<Id>(nodes_.size());
    nodes_.push_back(Node{var, lo, hi});
    unique_.emplace(key, id);
    return id;
}
```

(The key packing assumes node ids `< 2^28` and `var < 256` — true here: |worlds|=15504 bounds any belief's node count far below 2^28. An assert on `nodes_.size() < (1u<<28)` guards it. Angle A keeps the packing trivially correct rather than introducing a tuple-hash; a `<28`-bit id space is abundant for this universe.)

---

## 2.1 Build-from-worlds — the simplest provably-correct algorithm

Angle A choice: **build each world as a singleton ZDD (a top-down chain of its 5 set bits), then `union` them, accumulating into `Z`.** This is O(nb · K) `mk`-calls for the singletons plus the `union` memo cost; it is dead-simple to get right and the build cost is irrelevant to the gate (the gate measures `|Z|` vs `nb`, not build time — §B.4a "Build `Z` from the current explicit `bw` once per belief").

**Singleton of a world `w`** (the family `{ set(w) }`): walk treasure ids **descending** so children are constructed before parents (children must be deeper = larger var, so build from var 19 down to var 0). The chain takes `hi` at each present bit and `lo=⊥` at the bottom is wrong — instead the chain's tail is `⊤`:

```cpp
// singleton: the family containing exactly the one subset 'bits-of-w'.
Id singleton(uint32_t w) {
    Id z = kBase;                                  // ⊤ = {∅}: the deepest tail
    for (int t = n_vars_ - 1; t >= 0; --t)         // descending so var increases toward terminals
        if ((w >> t) & 1u) z = mk(t, kEmpty, z);   // present: hi = (rest), lo = ⊥ (this var must be set)
    return z;                                      // absent vars are simply never mk'd -> zero-suppressed
}
```

Note the absent vars are *not* given a `lo`-branch — zero-suppression means "var not present in any member" is encoded by its *absence* from the chain, which is exactly why a 5-of-20 world produces a 5-node chain, not a 20-node one. This is the structural reason the sparse regime compresses (§B.1 "specialized for sparse sets of subsets").

**Union** (the only `apply` operation the build needs — §B.5 "memoized `apply`"), the standard ZDD recursion, memoized on the unordered id-pair:

```cpp
Id zunion(Id a, Id b) {
    if (a == kEmpty) return b;
    if (b == kEmpty) return a;
    if (a == b)      return a;
    if (a > b) std::swap(a, b);                    // canonicalize the memo key (union is commutative)
    const uint64_t key = (uint64_t(a) << 32) | b;
    if (auto it = union_memo_.find(key); it != union_memo_.end()) return it->second;
    Id r;
    const bool at = a >= 2, bt = b >= 2;           // is non-terminal?
    if (!at) {            // a == ⊤ ({∅}) , b non-terminal: add ∅ into b's lo-most branch
        // ⊤ ∪ b : the empty subset joins b. b's members never include b.var on its lo path;
        // recurse into lo so ∅ lands at b's tail.
        r = mk(nodes_[b].var, zunion(kBase, nodes_[b].lo), nodes_[b].hi);
    } else if (!bt) {
        r = mk(nodes_[a].var, zunion(nodes_[a].lo, kBase), nodes_[a].hi);
    } else {
        const Node& na = nodes_[a]; const Node& nb = nodes_[b];
        if (na.var < nb.var)        r = mk(na.var, zunion(na.lo, b), na.hi);   // b has no na.var -> goes lo
        else if (na.var > nb.var)   r = mk(nb.var, zunion(a, nb.lo), nb.hi);
        else                        r = mk(na.var, zunion(na.lo, nb.lo), zunion(na.hi, nb.hi));
    }
    union_memo_.emplace(key, r);
    return r;
}

Id build_from_worlds(std::span<const uint32_t> bw) {
    Id z = kEmpty;                                 // ⊥ : empty family
    for (uint32_t w : bw) z = zunion(z, singleton(w));
    return z;
}
```

Correctness traps handled: the `⊤ ∪ b` arm must recurse into `b`'s `lo` chain so `∅` is added as a member of the family without disturbing `b`'s members (a naive `mk(var, ⊤, hi)` would be wrong). Because every world has the same K=5 cardinality, in practice `⊤` only meets the other operand at the shared tail, but the general arm is written for total correctness and is exercised by the empty/degenerate edge cases below.

(`union_memo_` is cleared between beliefs in the probe — it is keyed by node id and ids are per-belief-build. Simplest: construct a fresh `BeliefZdd` per belief in Stage 1's loop. That also makes `|Z| = node_count(z)` count *only this belief's* nodes, which is what the gate wants. Angle A: one `BeliefZdd` per belief, thrown away after — no cross-belief node-table reuse to reason about.)

---

## 3. Structural queries (Stage 1: the decision gate)

### Cardinality `count(Z) = nb` (§B.2 row 1)
One bottom-up pass, memoized per node id:
```cpp
uint64_t count(Id z) const {                       // memo: vector<int64_t> sized nodes_.size(), -1 = unset
    if (z == kEmpty) return 0;
    if (z == kBase)  return 1;
    if (memo[z] >= 0) return memo[z];
    return memo[z] = count(nodes_[z].lo) + count(nodes_[z].hi);   // members = (omit var) + (contain var)
}
```
The recursion is the definition: a node's family = its lo-family ∪ its hi-family, disjoint by construction, so cardinalities add.

### Node count `|Z|` (the gate number)
Reachable **non-terminal** unique nodes from `z`:
```cpp
size_t node_count(Id z) const {                    // DFS with a visited set, exclude ⊥/⊤
    std::unordered_set<Id> seen;
    std::function<void(Id)> go = [&](Id u){ if (u<2 || !seen.insert(u).second) return;
                                            go(nodes_[u].lo); go(nodes_[u].hi); };
    go(z); return seen.size();
}
```
This is `|Z|` — well-defined because `mk` canonicalizes (§0). It is the quantity the whole on-ramp exists to measure.

### Member enumeration (for the faithful-rep check)
```cpp
void enumerate(Id z, std::vector<uint32_t>& out) const {   // accumulate the set-bits along hi-edges
    if (z == kEmpty) return;
    if (z == kBase)  { out.push_back(acc); return; }        // acc threaded as a recursion arg, starts 0
    // lo: var absent;  hi: var present -> set its bit
    enumerate(nodes_[z].lo, out /*acc*/);
    enumerate(nodes_[z].hi, out /*acc | (1u<<nodes_[z].var)*/);
}
```
(Implemented with `acc` as an explicit parameter; sketch elides it for brevity.) Returns every member world as its bitmask.

### The FAITHFUL-REPRESENTATION check (Stage 1 — what makes `|Z|` trustworthy)
For each belief `bw`:
1. `Id z = build_from_worlds(bw);`
2. `assert(count(z) == bw.size());`  — cardinality matches.
3. `enumerate(z, members); sort(members); sort(bw_copy); assert(members == bw_copy);` — the *member set equals bw exactly* (the env's worlds are distinct, so set-equality is multiset-equality here). This proves `z` *is* the belief, so `|Z| = node_count(z)` is a faithful structural measure, not a number for some other family.

Only after this passes do we record `(nb, |Z|)`.

---

## 4. The realistic-belief generator (Stage 1 — the CRITICAL part, §B.4)

The note is explicit and the prompt re-stresses it: **realistic beliefs are the full world-set narrowed by RANDOM OBSERVATION SEQUENCES, not random subsets.** Random subsets have `|Z| ≈ nb` and would *falsely kill Part B* (§B.4 "ISMCTS information sets … defined by observation histories (conjunctions of detector constraints) — precisely what diagrams represent compactly"). The generator must therefore produce *observation-consistent* beliefs.

Algorithm (uses only the existing env filters — `filter_detector`, `filter_treasure`, both reading the live instance):

```cpp
// One realistic belief at a target depth D, RNG-seeded for reproducibility.
std::vector<uint32_t> realistic_belief(const Environment& env, std::mt19937_64& rng, int depth) {
    const std::vector<uint32_t>& all = env.worlds();
    // 1. sample a TRUE world uniformly from the full set (the ground truth the observations are consistent with)
    uint32_t truth = all[std::uniform_int_distribution<size_t>(0, all.size()-1)(rng)];
    std::vector<uint32_t> bw(all.begin(), all.end());      // start from the full belief
    // 2. apply `depth` random observations, each outcome CONSISTENT with `truth` (so bw never empties).
    const int N = env.N(), nD = env.n_detectors();
    for (int s = 0; s < depth && bw.size() > 1; ++s) {
        if (std::uniform_int_distribution<int>(0,1)(rng)) {          // a detector observation
            int j = std::uniform_int_distribution<int>(0, nD-1)(rng);
            bool positive = env.observe(j, truth);                  // the TRUE reading at the true world
            env.filter_detector(bw, j, positive);                   // keep worlds agreeing with truth
        } else {                                                    // a treasure observation
            int i = std::uniform_int_distribution<int>(0, N-1)(rng);
            bool present = ((truth >> i) & 1u) != 0;
            env.filter_treasure(bw, i, present);
        }
    }
    return bw;   // never empty: `truth` itself always survives every consistent filter (>=1 world)
}
```

Why this is the right generator (and not random subsets):
- The belief is exactly an **information set**: the set of worlds consistent with an observation history — a *conjunction of detector/treasure constraints* (§B.4). That is the structured regime where `|Z| ≪ nb` is plausible.
- Outcomes are drawn against a fixed sampled `truth`, so the belief is always **non-empty and self-consistent** (mirrors how the live search narrows belief in `env.apply` → `filter_detector`/`filter_treasure`). `truth` always survives, so `nb ≥ 1`.
- It reuses the env's exact filters — no re-derivation of belief mechanics, so the beliefs are precisely the ones the search produces.

The probe sweeps **depths** `D ∈ {0, 1, 2, 4, 8, 16, 24, 32}` (0 = full world-set; deep = sharply narrowed) and **samples per depth** (e.g. 64), seeded deterministically so the run is reproducible. For each it records `(depth, seed, nb, |Z|, |Z|/nb)`.

---

## 5. The |Z|-vs-nb experiment (Stage 1 output — the deciding measurement, §B.4)

For each `(depth, sample)`:
1. generate the realistic belief;
2. build `Z`, run the faithful-rep check (abort loudly on failure — ADR-0002; a broken diagram must not silently report a `|Z|`);
3. record `nb = count(Z)`, `|Z| = node_count(Z)`, ratio.

Report (stdout, parseable, mirroring the oracle's `RESULT:` line discipline):
- per-depth aggregate: `min/median/max nb`, `min/median/max |Z|`, `median |Z|/nb`, and `median nb/|Z|` (the *compression factor* — the §B.3 win is exactly `|Z| ≪ nb`);
- a final verdict line, e.g.
  `RESULT: PASS belief-zdd gate (depths 0..32, 64 samples each; median |Z|/nb at depth 16 = <r>; faithful-rep verified on <m> beliefs)`.

The **interpretation rubric** (the note's §B.4 decision, restated so the probe's output is actionable): median `|Z|/nb ≪ 1` at realistic search depths → structured → graduate toward §B.4(b) (belief-as-ZDD). Median `|Z|/nb ≈ 1` → no structure → the SIMD sweep wins; shelve the diagram for features (the exploration still taught the tool, §B.4). The probe **does not decide**; it produces the number. (Angle A: the gate is the deliverable — get a trustworthy `|Z|/nb` table on observation-narrowed beliefs as cheaply as possible.)

---

## 6. Stage 2 — the two queries, bit-exact against the sweep (full §B.4a)

These are the §B.2 queries, and the §B.3 validation is what makes the diagram safe to stand up beside the sweep: `bit_cnt` and `det_cnt` from the diagram are **exact integers that must equal the sweep's** — a logic invariant.

### 6.1 All-marginals `bit_cnt[t]` for all t — ONE forward × backward sweep, O(|Z|) (§B.2 row 2)
The note is explicit: this is *not* N independent queries. The standard "marginals from a decision diagram" computation:

- **`below[u]` = count(u)** (subtree cardinality, the §3 `count` memo) — # members in `u`'s family. (the "backward / sub-counts below each node" sweep.)
- **`above[u]` = # of partial paths from the root that reach node `u` taking lo/hi edges**, i.e. the number of distinct *prefix choices* (which earlier vars were present/absent) that land at `u`. (the "forward / path-counts to each node" sweep.)

`above` is computed top-down: `above[root] = 1`; each node `u` pushes its `above[u]` to both children: `above[lo[u]] += above[u]`, `above[hi[u]] += above[u]`. **Trap (§B.4 "the all-marginals sweep"):** because ZDD *suppresses* absent vars, an edge from `u` (var `a`) to a child `c` (var `b > a`) *skips* vars `a+1 … b-1`, and every skipped var is **absent** on that path — it contributes nothing to `bit_cnt` of the skipped vars, which is correct (those vars are not present in any member reached via that edge). So **no level-skipping correction is needed for the marginal count of var `t`**: a world is counted in `bit_cnt[t]` iff its path takes the **hi-edge of a node whose var == t**.

Therefore:
```
bit_cnt[t] = Σ over nodes u with nodes_[u].var == t  of  ( above[u] * below[hi[u]] )
```
i.e. for each node, the number of full members whose path goes (some prefix reaching u) × (u takes its hi-edge, contributing the var) × (any completion below hi). One pass over all nodes accumulates this into `bit_cnt[var]`. Total cost O(|Z|) shared across all t — exactly the note's "single forward × backward sweep, not N independent queries."

```cpp
void all_marginals(Id z, std::span<int64_t> bit_cnt) const {  // bit_cnt sized N, zero-filled by caller
    // 1. below[] = count() memo (backward).  2. above[] via top-down accumulation (forward).
    // toposort by var ascending is automatic: process nodes in DECREASING id is not safe; instead
    // do a DFS from z pushing above to children (a node's above is final once all parents processed —
    // guaranteed because every parent has a strictly-smaller var, so a var-ascending order works).
    std::vector<int64_t> above(nodes_.size(), 0), below(nodes_.size(), -1);
    // collect reachable nodes, order by var ascending (parents before children)
    std::vector<Id> order = reachable_sorted_by_var(z);   // DFS-collect, stable-sort by nodes_[u].var
    above[z] = 1;
    for (Id u : order) {                                  // u's above is now final
        below_of(u, below);                               // memoized count
        if (above[u] == 0) continue;
        Id lo = nodes_[u].lo, hi = nodes_[u].hi;
        bit_cnt[nodes_[u].var] += above[u] * count_via(hi, below);   // hi-edge contributes var present
        if (lo >= 2) above[lo] += above[u];
        if (hi >= 2) above[hi] += above[u];
    }
}
```
(`count_via(c, below)` returns 0 for ⊥, 1 for ⊤, else `below[c]`.) Reachability + var-ascending ordering guarantees every parent is processed before its children (strict ordering, §0), so `above[u]` is final when `u` is visited.

**Sanity identity to assert in-probe:** `Σ_t bit_cnt[t] == K * nb` (every world has exactly K=5 present bits — the env fact). A cheap extra net on the marginals sweep beyond the sweep-equality, catching an ordering/level-skip bug even if it somehow agreed coincidentally on a sample. (Angle A bonus check, near-free.)

### 6.2 Per-detector disjoint-count `det_cnt[j] = nb − #{worlds disjoint from mask_j}` (§B.2 row 3)
`disjoint_count(z, mask)` = members of `Z` that **avoid every bit in `mask`**. The ZDD operation (§B.2: "chain `offset(b)` over `b ∈ mask_j`, then count") simplifies to a single memoized recursion that, at a node whose var is in `mask`, **discards the hi-branch** (taking that var would intersect the mask):

```cpp
int64_t disjoint_count(Id z, uint32_t mask) const {     // memo keyed by node id (mask is fixed per call)
    if (z == kEmpty) return 0;
    if (z == kBase)  return 1;                           // ∅ is disjoint from everything
    if (memo[z] >= 0) return memo[z];
    const Node& u = nodes_[z];
    int64_t r = disjoint_count(u.lo, mask);              // omit var: always allowed
    if (((mask >> u.var) & 1u) == 0)                     // var NOT in mask -> may take it
        r += disjoint_count(u.hi, mask);
    return memo[z] = r;
}
// det_cnt[j] = count(z) - disjoint_count(z, masks[j]);
```
Cost O(|Z|) per detector (the memo is per-call; `Σ|mask_j|·|Z|` in the note collapses to `nD·|Z|` here since each detector is one disjoint-count pass over the whole node set). `nb`-independent.

This is exactly the §B.2 through-line: the popcount-1 shortcut `cnt[j]=bit_cnt[bit_j]` is the `|mask_j|=1` special case (`nb − #{worlds without bit b} = #{worlds with bit b}`). The disjoint-count generalizes it to arbitrary multi-bit masks — and the multi-bit edge case below verifies precisely that.

### 6.3 The bit-exact harness (§B.3, mirrors `belief_sweep_oracle_check`)
For each belief (the **realistic** generated set, plus the explicit edge cases of §7):
1. `BeliefFeatures sweep = chocofarm::belief_features(bw, env.face_masks(), N, nD, log_nworlds);` — the production sweep, the oracle.
2. Build `Z`, run the faithful-rep check.
3. Compute the diagram's integer counts:
   - `std::vector<int64_t> zbit(N,0); zdd.all_marginals(z, zbit);`
   - `for j: zdet[j] = count(z) - disjoint_count(z, masks[j]);`
4. **Reconstruct the sweep's integer counts** from `sweep` to compare integer-to-integer (the safety net is the *logic invariant*, §B.3): `sweep`'s `marg[t] = bit_cnt[t]*inv` and `p_pos[j] = det_cnt[j]*inv`, so `sweep_bit[t] = llround(sweep.marg[t] * nb)` and `sweep_det[j] = llround(sweep.p_pos[j] * nb)`. Assert `zbit == sweep_bit` and `zdet == sweep_det`, integer-exact. (Equivalently — and more directly faithful to "the sweep's counts" — the probe can call a tiny inline naive count, identical to the oracle's `reference()`, and compare integers without the float round-trip. Angle A prefers the inline naive count: it removes the `llround` reasoning entirely and is the same independent path the existing oracle already trusts. Use that.)
5. Then assert the **feature-vector byte-identity** (§B.3 "the identical Phase 2 (`* inv`) makes the feature vector byte-identical"): run the diagram counts through the *identical* Phase-2 map (`* inv`, same `informative`/`marg_sum`/`sharpness`/`nonempty` lines copied from `belief_features_nonempty`) into a `BeliefFeatures`, and reuse the oracle's `equal_features`-style byte comparison against `sweep`. This closes the loop: the diagram path is byte-for-byte the §A golden.

On any mismatch: `RESULT: FAIL` naming the belief (depth/seed) and the field/index, exactly as the oracle's `fail()` does. On success: `RESULT: PASS`.

---

## 7. Edge cases (all required; each is a one-liner in the probe's fixture list)

The probe's belief fixture is the realistic-generator sweep **plus** these explicit hand-built cases, mirroring how `belief_sweep_oracle_check` seeds prefixes + a strided set:

| Edge case | Belief | What it pins |
| — | — | — |
| **Empty belief** (nb=0) | `bw = {}` | `build_from_worlds → kEmpty`; `count==0`, `node_count==0`, `enumerate→∅`, `all_marginals→ all 0`, `disjoint_count==0`. Feature path must equal `belief_features_empty` (all-zero BeliefFeatures). The `⊥` terminal is exercised. |
| **Single world** (nb=1) | `bw = {all[0]}` | `Z` = one singleton chain (K=5 nodes); `count==1`, `node_count==5`, `enumerate=={all[0]}`. `bit_cnt[t]==1` for the 5 present bits, 0 else; `det_cnt[j]==observe(j,all[0])?1:0`. The `⊤` tail is exercised. |
| **Full world-set** (nb=15504) | `bw = env.worlds()` | the largest `nb`; faithful-rep must still enumerate all 15504 in `combinations` order (after sort). `Σ bit_cnt = 5·15504`. Stresses the union/memo and the worst-case `|Z|` (this is the depth-0 generator case too). |
| **Popcount-1 detector mask** | any belief; pick a `j` with `popcount(masks[j])==1` | verifies `det_cnt[j] == bit_cnt[bit_j]` — the §B.2 through-line / the Part-A popcount-1 shortcut as the `|mask|=1` special case of disjoint-count. The probe asserts this identity *in addition* to the sweep-equality. (If no nD has popcount 1 on the live instance, the probe synthesizes a single-bit mask `1<<t` and checks `disjoint_count` against `nb - bit_cnt[t]` directly — not all detectors need be real; this exercises the algorithm.) |
| **Multi-bit detector mask** | any belief; pick a `j` with `popcount(masks[j])>1` | the general disjoint-count: `det_cnt[j] == nb - #{w : (w&mask_j)==0}`, the arbitrary-disjunction case the diagram generalizes the popcount-1 shortcut to. Asserts against the sweep's `det_cnt[j]` and against an independent brute `#{w∈bw : (w&mask)==0}` count over `bw`. |

The empty and single-world cases are the terminal-correctness net for `⊥`/`⊤` — the ZDD bug surface most likely to be wrong (the §0 trap).

---

## 8. The pytest gate (mirrors `test_cpp_belief_sweep_oracle` exactly)

Add to `tests/test_cpp_runner.py` next to the oracle gate (same opt-in `CHOCO_RUN_CPP` mechanism, same `RESULT: PASS` contract, same `cwd=REPO`):

```python
BELIEF_ZDD_BIN = os.path.join(REPO, "cpp", "build", "chocofarm-belief-zdd-probe")

@pytest.mark.skipif(not (_RUN_CPP and os.path.exists(BELIEF_ZDD_BIN)), reason=_CPP_SKIP)
def test_cpp_belief_zdd_probe():
    """The §B.4(a) ZDD on-ramp. STAGE 1: a minimal hand-rolled ZDD built per belief, with a FAITHFUL-REP
    invariant (enumerate(Z)==bw, count(Z)==nb) so |Z| is trustworthy, measured on REALISTIC
    observation-narrowed beliefs (NOT random subsets) -> the |Z|-vs-nb decision number (note §B.4). STAGE 2:
    the all-marginals (bit_cnt) + per-detector disjoint-count (det_cnt) off Z, asserted EQUAL to
    chocofarm::belief_features's integer counts bit-exact (the §B.3 logic invariant), then the identical
    Phase-2 *inv makes the feature vector byte-identical to the sweep. ADR-0011: net the diagram, don't
    trust it. Pure compute (no FeatureBuilder, no redis)."""
    out = subprocess.run([BELIEF_ZDD_BIN, "--instance", DATA_INSTANCE, "--faces", DATA_FACES],
                         cwd=REPO, capture_output=True, text=True, timeout=120,
                         env={**os.environ, "PYTHONPATH": REPO})
    sys.stdout.write(out.stdout); sys.stderr.write(out.stderr)
    assert out.returncode == 0 and "RESULT: PASS" in out.stdout
```

The probe's `main` mirrors `belief_sweep_oracle_check.cpp`: parse `--instance`/`--faces` via the same `opt()` helper, `load_instance` → `Environment`, read `N`/`nD`/`face_masks()`/`worlds()`/`log_nworlds`, run Stage 1 (gate + faithful-rep) then Stage 2 (bit-exact harness), emit the `|Z|`-vs-`nb` table to stdout, and a single terminal `RESULT: PASS …` / `RESULT: FAIL …` line. Returns 0/1.

---

## 9. ADR hygiene (§B.6) and documentation

- **`BeliefZdd` is a one-owner collaborator (P3) behind a typed value seam (P9):** node ids never escape its API — callers pass beliefs/masks in and get counts/vectors out. The future B.4(b) graduation swaps the engine behind this same seam (§B.4b "behind the seam, not a new cross-boundary wire fact" — P7 unaffected). The probe is a separate executable, one-owner of the gate+harness fixture (P3).
- **The diagram counts are a logic invariant → bit-exact assert vs the sweep (P6 strongest tier):** the sweep (`chocofarm::belief_features`) stays the oracle during bring-up; the probe nets the diagram against it, never trusts it (ADR-0011 — exactly as `belief_sweep_oracle_check` nets the §A.4 rewrite).
- **Fail loudly (ADR-0002):** the faithful-rep check `abort()`s with a named belief on any divergence — a diagram that misrepresents the belief must never reach the `|Z|` report.
- **ADR-0006:** all three new files carry the path + purpose + Public Domain module-docstring header.
- **Documentation (ADR-0005 / "documentation is part of the work"):** the driving note's §B.4 is a point-in-time design record — *do not retro-edit it*; append the gate result (the measured `|Z|/nb` table and the (a)→(b) verdict) where the live state lives (the commit log / a dated handoff entry), per the CLAUDE.md "live queue belongs in the commit log, not immutable prose." If the gate graduates Part B, that is the §B.4 "Revisit when…" trigger firing — record it by dated amendment, not silent rewrite.

---

## 10. Why this is the simplest path to a *trustworthy* gate (Angle A summary)

- **One reduction rule that matters** (zero-suppression on `hi==⊥`) + hash-consing — nothing else. No `apply` beyond `union`. No level-padding, no complement edges, no garbage collection (per-belief throwaway arena). This is the minimum that yields a canonical reduced ZDD, so `|Z|` is meaningful.
- **Build = singletons + union** — the easiest-to-verify construction; build cost is irrelevant to the gate.
- **`|Z|` is never trusted without the faithful-rep check** (`enumerate==bw` and `count==nb`) — so a buggy diagram fails loudly rather than producing a misleadingly small or large `|Z|`.
- **Realistic = observation-narrowed, never random subsets** — the single methodological point that decides whether the gate is honest; reuses the env's own `filter_detector`/`filter_treasure` so the measured beliefs are exactly the search's.
- **Stage 2 reuses the same `count`/recursion primitives** and the existing oracle's independent-naive-count pattern, so the bit-exact net is a small extension of an already-trusted fixture.

Relevant absolute paths:
- New: `/home/bork/w/vdc/1/chocofarm/cpp/include/chocofarm/belief_zdd.hpp`, `/home/bork/w/vdc/1/chocofarm/cpp/src/belief_zdd.cpp`, `/home/bork/w/vdc/1/chocofarm/cpp/src/belief_zdd_probe.cpp`
- Edited: `/home/bork/w/vdc/1/chocofarm/cpp/CMakeLists.txt` (add `src/belief_zdd.cpp` to `chocofarm_core`; add the `chocofarm-belief-zdd-probe` target), `/home/bork/w/vdc/1/chocofarm/tests/test_cpp_runner.py` (add `test_cpp_belief_zdd_probe`)
- Mirrored/reused unchanged: `/home/bork/w/vdc/1/chocofarm/cpp/src/belief_sweep_oracle_check.cpp` (harness pattern + `reference()` naive count), `/home/bork/w/vdc/1/chocofarm/cpp/src/features.cpp` (`belief_features`, the oracle), `/home/bork/w/vdc/1/chocofarm/cpp/src/env.cpp` (`filter_detector`/`filter_treasure`/`observe`/`face_masks`/`worlds`)


---

## Proposal 2

I have all context needed. I have read all six required documents end to end (the driving note, features.cpp, belief_sweep_oracle_check.cpp, env.hpp, env.cpp, CMakeLists.txt), plus the supporting headers (feature_compute.hpp, features.hpp BeliefFeatures, instance.hpp Face/load_instance), the pytest gate pattern, and confirmed the ADR-0012 P-principle definitions I cite. Now I'll produce the complete design.

# Belief ZDD On-Ramp — Complete Design (Angle B: the typed `BeliefDiagram` seam + the B.2 query algorithms done right)

This is the B.4(a) on-ramp from `belief_features_and_decision_diagram_note.md` Part B, staged measure-first. I lead with my emphasis — a clean typed `BeliefDiagram` value-seam (note B.6) and the full B.2 query algorithms with the subtle forward/backward all-marginals sweep nailed — and cover every requested component. All counts the diagram produces are the **logic invariant** the §A.4 sweep already computes (`bit_cnt`, `det_cnt`, `nb`); the diagram path is brought up *beside* the sweep and diffed against it bit-exact (note B.3), exactly as `belief_sweep_oracle_check.cpp` nets the §A.4 rewrite.

Citations to the note are to its B.x sections.

---

## 0. The decision this on-ramp informs (note B.4, the spectrum)

Part B's payoff is conditional on belief structure (note opening, B.3). The measurement that decides (a)→(b) is `|Z|` vs `nb` on **realistic** beliefs (B.4 final paragraph): `|Z| ≪ nb` ⇒ graduate to belief-as-diagram (b); `|Z| ≈ nb` ⇒ the SIMD sweep wins, shelve the diagram for features. So the deliverable is staged:

- **STAGE 1 — the decision gate.** Minimal hand-rolled ZDD: build-from-worlds, `count(Z)=nb`, `|Z|` (reduced unique nodes), a faithful-representation check (enumerate Z == bw, count==nb) so `|Z|` is trustworthy, and the `|Z|`-vs-`nb` measurement on realistic beliefs. This is the smallest thing that produces the deciding number honestly.
- **STAGE 2 — full B.4(a).** The all-marginals query (forward×backward sweep, `O(|Z|)`) and the per-detector disjoint-count, asserted bit-exact equal to the sweep's `bit_cnt`/`det_cnt` per belief, then the identical Phase-2 `* inv` makes features byte-identical (B.2, B.3).

Both stages live behind one typed seam so Stage 2 is *additive* to Stage 1 — no rework.

---

## 1. The `BeliefDiagram` seam (note B.5, B.6)

### 1.1 The value-seam contract (B.6: one-owner P3, typed value seam P9)

`BeliefDiagram` is a **one-owner collaborator** (note B.6, ADR-0012 P3) wrapping a hand-rolled ZDD engine, exposing a **typed value seam** — no raw node pointers / node ids leak across the boundary (note B.6, P9). The only types crossing the boundary are: a constructor taking the world-set, and value-returning queries (`uint64_t`, `BeliefFeatures`, `std::vector<uint32_t>`). This is the "swap the engine later if it pays" boundary of B.5: a CUDD/Sylvan/SapporoBDD wrap would implement the *same* public surface.

```cpp
// cpp/include/chocofarm/belief_diagram.hpp
// Purpose: the typed value-seam over a hand-rolled ZDD of a belief (a family of 5-of-20 worlds).
//   It is the B.4(a) on-ramp from belief_features_and_decision_diagram_note.md Part B: build Z from
//   the explicit world-set `bw`, then answer the Phase-1 counting queries (nb, all bit_cnt[t],
//   per-detector det_cnt[j]) in O(|Z|)-ish time, independent of nb (B.2). The integer counts are a
//   LOGIC INVARIANT that must EQUAL the §A.4 sweep's bit-for-bit (B.3) — netted by the oracle.
//
//   One-owner collaborator (ADR-0012 P3); typed value seam (P9) — no ZDD node ids leak across the
//   boundary, so the engine (hand-rolled here; CUDD/Sylvan/SapporoBDD later, B.5) is swappable.
//
// Public Domain (The Unlicense).
#pragma once

#include <cstdint>
#include <span>
#include <vector>

#include "chocofarm/features.hpp"   // BeliefFeatures (Stage 2 returns this, byte-identical to the sweep)

namespace chocofarm {

class BeliefDiagram {
  public:
    // Build Z = the ZDD family of exactly the worlds in `bw`, over N variables (treasure ids 0..N-1).
    // N is the variable-count (env.N()); the worlds carry K=5 set bits each but the diagram does NOT
    // assume that — it is general over subsets of the N-element universe (note B.1). Worlds may repeat
    // or be unsorted in `bw`; the family is the SET of distinct worlds (build is set-union by
    // construction — see §3). The empty `bw` builds Z = {} (the false/empty terminal).
    BeliefDiagram(std::span<const uint32_t> bw, int N);

    // ---- STAGE 1: the decision-gate measurements ----
    [[nodiscard]] uint64_t count() const;        // cardinality |Z| of the FAMILY == nb (note B.2 row 1)
    [[nodiscard]] std::size_t node_count() const; // |Z|: number of REDUCED, hash-consed internal nodes
    [[nodiscard]] std::vector<uint32_t> members() const;  // enumerate Z's worlds (faithful-rep check)

    // ---- STAGE 2: the B.2 counting queries (the §A.4 Phase-1 outputs) ----
    // all bit_cnt[t] for t in 0..N-1 from ONE forward(path-counts) x backward(subtree-counts) sweep,
    // O(|Z|) shared work — NOT N independent queries (note B.2 row 2). The subtle one (§5).
    [[nodiscard]] std::vector<uint64_t> all_marginals() const;
    // det_cnt[j] = nb - #{worlds disjoint from mask_j} for each j (note B.2 row 3). The disjoint count
    // is count() of the subfamily of Z avoiding every bit in mask_j (§6).
    [[nodiscard]] std::vector<uint64_t> all_detector_counts(std::span<const uint32_t> masks) const;

  private:
    // ---- the hand-rolled ZDD engine (opaque; never leaks) ----
    struct Node { int var; uint32_t lo; uint32_t hi; };  // §2
    std::vector<Node> nodes_;     // node table; ids 0,1 reserved for terminals (§2.2)
    uint32_t root_ = 0;           // the family's root node id (0=⊥ empty family, 1=⊤ {∅})
    int n_ = 0;                   // variable count N
    // hash-cons + reduction live in the .cpp (build-time only); not part of the value seam.
};

}  // namespace chocofarm
```

The seam is deliberately *thin*: Stage 1 needs only the first three methods; Stage 2 adds the two query methods. No method takes or returns a node id. `all_marginals()` returns `bit_cnt`; `all_detector_counts(masks)` returns `det_cnt`; `count()` returns `nb`. The caller (the probe, later the feature builder) reconstructs `BeliefFeatures` with the **identical Phase-2 `* inv`** of features.cpp — that is what makes the feature vector byte-identical (B.3). (For Stage 2 I provide a free helper `belief_features_from_diagram(...)` that does exactly Phase-2, §7, so the byte-identity claim has one home.)

---

## 2. ZDD node representation, terminals, hash-cons, reduction (note B.1, B.5)

### 2.1 Why ZDD, not BDD (note B.1)

A belief is a family of subsets over the N=20-element treasure universe; each world is a 5-of-20 subset (sparse — the regime ZDDs compress, ENV FACTS). The ZDD is "the representation specialized for sparse sets of subsets — more compact when worlds carry few treasures, which is the common case" (note B.1). BDD model-counting is "the cleaner first mental model" (B.1), but the queries are stated structure-agnostically (B.1) and the **production target if structure exists** is the ZDD (B.1). I hand-roll the ZDD directly because the zero-suppression rule is *the whole point* of the `|Z| ≪ nb` measurement on sparse worlds.

### 2.2 Nodes and terminals

A ZDD over variables `0..N-1` with a **fixed variable order** `0 < 1 < ... < N-1` (treasure-id order; §"correctness traps" on ordering). Two terminals and internal nodes:

- **Terminal `⊥`** = id `0` = the empty family `{}` (the FALSE sink — no sets). `count(⊥)=0`.
- **Terminal `⊤`** = id `1` = the family `{∅}` containing exactly the empty set (the TRUE sink). `count(⊤)=1`.
- **Internal node** = `{var, lo, hi}` at variable `var`, with `lo` = the subfamily where `var` is **absent**, `hi` = the subfamily where `var` is **present** (the hi-edge "selects" variable `var` into every set below it). Both `lo`,`hi` are ids of nodes with `node[lo].var > var` and `node[hi].var > var` (strictly increasing down both edges; terminals are treated as `var == N`, i.e. greater than every variable).

```cpp
struct Node { int var; uint32_t lo; uint32_t hi; };
// nodes_[0] = ⊥ sentinel, nodes_[1] = ⊤ sentinel (var = N for both, lo=hi=0 unused).
// internal nodes appended at ids >= 2.
```

### 2.3 The zero-suppression reduction rule — the correctness trap (note: §B "the ZDD reduction rule")

Two reduction rules, applied at every node creation (this is *not* the BDD rule — getting this wrong silently miscounts):

1. **Zero-suppression (the ZDD-specific rule):** if `hi == ⊥` (id 0), the node is redundant — return `lo` directly, do NOT create the node. (A node whose hi-edge leads to the empty family contributes nothing by selecting `var`; the variable is suppressed.) **This is the rule that gives `|Z| ≪ nb` on sparse families** and the one a BDD-trained reflex gets wrong (the BDD rule eliminates a node when `lo == hi`, which is *invalid* for a ZDD and would corrupt the count).
2. **Merging (hash-consing / unique-table):** two nodes with identical `(var, lo, hi)` are the same node — return the existing id. This is the *unique* in "reduced unique nodes" — `|Z|` counts these merged nodes.

A node is created only via the factory `mk(var, lo, hi)` that applies both rules:

```cpp
// the ZDD reduce-and-hash-cons factory: returns the id of the (var,lo,hi) node, applying
// zero-suppression then unique-table merge. The ONLY way internal nodes are made (P1: one home).
uint32_t mk(int var, uint32_t lo, uint32_t hi) {
    if (hi == 0) return lo;                       // (1) zero-suppression: hi==⊥ ⇒ node is redundant
    const Key key{var, lo, hi};
    if (auto it = unique_.find(key); it != unique_.end()) return it->second;  // (2) merge
    const uint32_t id = static_cast<uint32_t>(nodes_.size());
    nodes_.push_back(Node{var, lo, hi});
    unique_.emplace(key, id);
    return id;
}
```

`unique_` is `std::unordered_map<Key, uint32_t>` with `Key{int var; uint32_t lo; uint32_t hi}` and a hand-written hash (mix of the three fields). It is **build-time scratch** (it lives in the build helper, not as a member — the seam holds only the reduced `nodes_` + `root_`). `|Z| = nodes_.size() - 2` (subtract the two terminals).

**Canonicity:** with a fixed variable order, both reduction rules applied at every `mk`, and unique-table merging, the ZDD is **canonical** — two equal families produce structurally identical diagrams and the same `|Z|`. This is what makes `node_count()` a meaningful measurement (an un-reduced or non-canonical diagram would inflate `|Z|` and falsely kill or falsely bless Part B).

---

## 3. Build-from-worlds (note B.4a: "build Z from the current explicit `bw`")

Build `Z` as the **union** of the singleton families `{w}` for each distinct `w ∈ bw`. Two correct algorithms; I specify the **bottom-up radix/merge build** (it is `O(nb · N)`-ish and naturally canonical), with a note on the alternative.

### 3.1 Single-world family `{w}` as a ZDD chain

The family `{w}` (one world, the subset whose set bits are `w`) is a chain: for each variable `t` in **increasing** order, if bit `t` is set in `w`, a node `mk(t, lo=⊥, hi=below)` selecting `t`; bits not set in `w` are simply skipped (zero-suppression already gives this — a variable absent from every set in the subfamily never appears). The chain terminates at `⊤`. Concretely, build from the top variable down by recursion, or iteratively from the **highest** set bit down to the lowest:

```cpp
// the ZDD for the single set whose elements are the set bits of `w`.
uint32_t single(uint32_t w) {
    uint32_t cur = 1;                          // ⊤ = {∅}
    for (int t = N - 1; t >= 0; --t)           // build the chain bottom-up (high var first)
        if ((w >> t) & 1u) cur = mk(t, /*lo=*/0, /*hi=*/cur);  // select t: lo=⊥, hi=cur
    return cur;                                // a chain of exactly popcount(w) nodes
}
```

For our worlds `popcount(w) == K == 5`, so each single is a 5-node chain.

### 3.2 The union of all singletons (the family build)

The build is the ZDD `union` (set-union `∪`) folded over all worlds. Union is the standard memoized binary ZDD apply:

```cpp
// ZDD union: the family-union of subfamilies a and b. Memoized on (a,b) (canonicalized a<=b).
uint32_t zunion(uint32_t a, uint32_t b) {
    if (a == b) return a;
    if (a == 0) return b;                      // {} ∪ b = b
    if (b == 0) return a;                      // a ∪ {} = a
    if (a > b) std::swap(a, b);                // commutative ⇒ canonicalize the memo key
    const PairKey k{a, b};
    if (auto it = union_memo_.find(k); it != union_memo_.end()) return it->second;
    uint32_t res;
    const Node& na = nodes_[a]; const Node& nb = nodes_[b];
    // both ⊤? handled by a==b. One ⊤ (var==N) and one internal: ⊤ = {∅}.
    if (a == 1) { res = include_empty(b); }    // {∅} ∪ b  (see §3.3)
    else if (b == 1) { res = include_empty(a); }
    else if (na.var == nb.var)
        res = mk(na.var, zunion(na.lo, nb.lo), zunion(na.hi, nb.hi));
    else if (na.var < nb.var)                  // b has no node at na.var ⇒ b's whole family is in lo
        res = mk(na.var, zunion(na.lo, b), na.hi);
    else                                       // nb.var < na.var
        res = mk(nb.var, zunion(a, nb.lo), nb.hi);
    union_memo_.emplace(k, res);
    return res;
}
```

The fold: `root_ = ⊥; for (w : bw) root_ = zunion(root_, single(w));`.

**The pairwise fold is `O(nb · |Z| · ...)`** in the worst case. For a clean, predictably-`O(nb · K)` build with low constant, I use the **balanced (tournament) reduction** instead of a left fold — pair up the `nb` singletons, union each pair, recurse — which keeps intermediate diagrams small and is the standard `O(nb log nb · width)` union build. Either is correct (canonicity guarantees the same `root_`); the tournament is the default. (This matches note B.4a's "`O(nb log)`" build-cost estimate.)

### 3.3 `include_empty` (the `{∅} ∪ family` case)

`{∅} ∪ F` adds the empty set to family `F` if not present. A ZDD encodes "`∅ ∈ F`" by the leftmost all-lo path reaching `⊤`. `include_empty(f)`: if `f` already contains `∅` (walk all-lo to a terminal; `⊤` ⇒ yes) return `f`; else set the all-lo terminal from `⊥` to `⊤`. In our build the worlds are 5-of-20 so `∅` never appears, but the union apply must handle it for generality and for the empty-belief edge case. Cleanest: a tiny recursive `include_empty(f)` = `if f==⊥ then ⊤ elif f==⊤ then ⊤ else mk(node.var, include_empty(node.lo), node.hi)`.

### 3.4 Alternative build (noted, not chosen)

A **direct trie-merge bottom-up**: bucket worlds by their lowest set bit, recurse. Correct and slightly faster, but the union-apply build reuses the *same* memoized-apply machinery the disjoint-count query (§6) needs, so building it once and reusing it is the DRY choice (ADR-0012 P1: one home for the apply). I build on the union-apply.

---

## 4. Stage-1 primitives: cardinality, node-count, member-enumeration

### 4.1 `count()` — cardinality `|Z| == nb` (note B.2 row 1: "one bottom-up pass, memoized per node")

The cardinality of the family at node `n` is `card(lo) + card(hi)` with `card(⊥)=0`, `card(⊤)=1`. Memoized per node (computed once, `O(|Z|)`):

```cpp
uint64_t count() const {  // == nb
    std::vector<uint64_t> card(nodes_.size(), 0);
    card[0] = 0; card[1] = 1;                            // ⊥, ⊤
    for (uint32_t id = 2; id < nodes_.size(); ++id)      // ids are topologically increasing by
        card[id] = card[nodes_[id].lo] + card[nodes_[id].hi];  // construction (children made first)
    return card[root_];
}
```

**Topological order for free:** because `mk` appends a node only after its children exist (children are constructed first in `single`/`zunion`), node ids are a valid bottom-up order — a plain forward loop over ids is a correct memoized post-order, no recursion/visited-set needed. This same observation drives the all-marginals sweep (§5).

`card` fits `uint64_t`: `nb ≤ C(20,5) = 15504`, far under `2^64`. (Kept `uint64_t` so the same code is correct under belief-as-diagram (b), where `nb` can be astronomical — B.3 — while `|Z|` stays small.)

### 4.2 `node_count()` — `|Z|`

`return nodes_.size() - 2;` (the two terminals are not part of `|Z|`). This is the **decision-gate number**: log it against `nb=count()`.

### 4.3 `members()` — enumerate Z's worlds (the faithful-rep check input)

Depth-first over the diagram, accumulating the selected variables along each hi-edge into a world bitmask; every path that reaches `⊤` emits one world:

```cpp
std::vector<uint32_t> members() const {
    std::vector<uint32_t> out;
    std::function<void(uint32_t,uint32_t)> rec = [&](uint32_t id, uint32_t acc) {
        if (id == 0) return;                 // ⊥: dead path, no members
        if (id == 1) { out.push_back(acc); return; }   // ⊤: emit the accumulated set
        const Node& nd = nodes_[id];
        rec(nd.lo, acc);                                 // var absent
        rec(nd.hi, acc | (uint32_t{1} << nd.var));       // var present
    };
    rec(root_, 0);
    return out;
}
```

By canonicity this emits each member exactly once; its size equals `count()`. Order is the diagram's natural DFS order (ascending-by-construction), which is **not** `bw`'s order — so the faithful-rep check compares as *sets* (§8.2).

---

## 5. STAGE 2 — `all_marginals()`: the forward×backward sweep (MY EMPHASIS — note B.2 row 2)

This is the subtle part and the one the note flags ("This is the standard 'literal-count / marginals from a decision diagram' computation"; B.2 row 2). The naive way is N independent queries (`det_cnt`-style per variable), `O(N·|Z|)`. The right way is **one forward (path-counts to each node) × one backward (subtree-counts below each node) sweep**, `O(|Z| + N)` shared work (B.2 row 2: "from a single forward × backward sweep — `O(|Z|)` shared work, **not** N independent queries"). Getting the ZDD bookkeeping exactly right is the whole task.

### 5.1 The exact quantity and the ZDD subtlety

`bit_cnt[t] = #{worlds w ∈ Z : bit t set in w}` — the §A.4 `bit_cnt`. On a ZDD, "bit `t` set in `w`" means the path realizing `w` takes the **hi-edge at a node whose `var == t`**. So:

> `bit_cnt[t] = Σ over hi-edges e=(node u with u.var==t) of  ( #paths root→u ) × ( #members in the subfamily at u.hi )`.

Two per-node quantities make this `O(|Z|)`:

- **`down[u]` (backward / subtree-counts):** the cardinality of the subfamily rooted at `u` — i.e. `card[u]` from §4.1. `down[⊤]=1`, `down[⊥]=0`, `down[u]=down[u.lo]+down[u.hi]`. (Computed bottom-up, the §4.1 pass — reused, P1.)
- **`up[u]` (forward / path-counts):** the number of partial paths from `root_` down to `u`, where each step that *skips* a variable (because the ZDD suppressed it) still counts as exactly **one** way (zero-suppression means a skipped variable is unconditionally absent — there is one assignment for it, not two). So `up` does **not** multiply by skipped variables. `up[root_]=1`; every other `up[u]` accumulates `up[parent]` over each edge (lo or hi) into `u`.

### 5.2 The ZDD trap in the forward pass (the correctness trap on the all-marginals sweep)

The trap, and the reason a BDD-marginals recipe is wrong here: **in a ZDD, a variable skipped between a parent at `var=a` and a child at `var=b>a+1` is UNCONDITIONALLY ABSENT, not a free 0/1 choice.** In a *BDD*, a skipped variable doubles the path count (it can be 0 or 1). In a *ZDD*, a variable not on the path is absent in every member of that subfamily — so it contributes a factor of **1**, not 2, to both `up` and `down`. Because we never multiply by `2^(skipped)`, the ZDD path-count *equals* the member count directly, and `Σ_t bit_cnt[t]` will equal `Σ_w popcount(w) = K·nb = 5·nb` exactly (a built-in cross-check, §8.1). If you accidentally use the BDD doubling rule, `down[root_]` would be `2^N`-scaled and every count would be wrong — this is the single most likely bug, and the bit-exact oracle (§8) catches it instantly.

### 5.3 The algorithm

```cpp
std::vector<uint64_t> all_marginals() const {       // returns bit_cnt[0..N-1]
    const size_t M = nodes_.size();
    // backward (subtree-counts) — bottom-up; ids are topological (children-first), §4.1.
    std::vector<uint64_t> down(M, 0);
    down[1] = 1;                                     // ⊤; down[0]=0 (⊥) already
    for (uint32_t id = 2; id < M; ++id)
        down[id] = down[nodes_[id].lo] + down[nodes_[id].hi];

    // forward (path-counts to each node) — TOP-DOWN; process parents before children. Because ids are
    // assigned children-first (mk appends after recursion), a DESCENDING id loop visits every node
    // after all its parents — the reverse-topological order the forward pass needs.
    std::vector<uint64_t> up(M, 0);
    up[root_] = 1;                                   // one (empty) path arrives at the root
    for (uint32_t id = M; id-- > 2; ) {              // descending; terminals 0,1 are leaves (no children)
        if (up[id] == 0) continue;                   // unreachable from root_ (e.g. a shared subnode
        //                                              that is not under THIS root — possible if the
        //                                              engine is reused; here Z has one root so reachable)
        const Node& nd = nodes_[id];
        up[nd.lo] += up[id];                          // lo-edge: var absent, one way
        up[nd.hi] += up[id];                          // hi-edge: var present, one way
    }

    // combine: each hi-edge at a node with var==t contributes up[node] * down[hi] worlds with bit t set.
    std::vector<uint64_t> bit_cnt(static_cast<size_t>(n_), 0);
    for (uint32_t id = 2; id < M; ++id) {
        const Node& nd = nodes_[id];
        bit_cnt[static_cast<size_t>(nd.var)] += up[id] * down[nd.hi];
    }
    return bit_cnt;
}
```

**Why this is exactly `bit_cnt`:** every member `w ∈ Z` corresponds to exactly one root→⊤ path. `w` has bit `t` set iff that path takes the hi-edge at the (unique, by canonicity) node with `var==t` it passes through. `up[node]` counts the distinct prefixes reaching `node`; `down[node.hi]` counts the distinct suffixes from `node.hi` to `⊤`; their product over the hi-edge counts exactly the members whose path goes `prefix → (hi at var t) → suffix`. Summing over all `var==t` nodes (a member passes through at most one such node by the strict ordering, and through exactly one iff bit t is set) gives `#{w ∈ Z : bit t set} = bit_cnt[t]`. **No `2^skipped` factor** (§5.2) — the ZDD's zero-suppression makes skipped variables unconditionally absent.

**Cost:** two `O(|Z|)` passes + one `O(|Z|)` combine + an `O(N)` alloc = `O(|Z| + N)`, *one* sweep for all N marginals (B.2 row 2). `up[id]*down[hi]` is `uint64_t`; products are bounded by `nb ≤ 15504` per term and `Σ = K·nb`, no overflow at N=20/K=5; `uint64_t` keeps it correct under (b).

---

## 6. STAGE 2 — `all_detector_counts(masks)`: the disjoint-count query (note B.2 row 3)

`det_cnt[j] = nb − #{worlds disjoint from mask_j}` (note B.2 row 3 and the popcount-1 through-line B.2). The disjoint count is `count()` of the **subfamily of Z avoiding every bit in `mask_j`** — the ZDD `offset`/`subset0` chained over `b ∈ mask_j` (B.2 row 3: "ZDD: chain `offset(b)` over `b ∈ mask_j`, then `count`").

### 6.1 `offset` (subset0): the subfamily not containing variable `b`

`offset(f, b)` = the subfamily of `f` whose members do **not** contain `b` (drop the hi-branch at `b`). Standard ZDD `subset0`, memoized:

```cpp
uint32_t offset(uint32_t f, int b) {   // members of f that do NOT contain variable b
    if (f == 0 || f == 1) return f;    // ⊥,⊤ contain no variables ⇒ unchanged
    const Node& nd = nodes_[f];
    if (nd.var > b)  return f;          // b is below this node's var ⇒ b absent everywhere already
    if (nd.var == b) return nd.lo;      // drop the hi (var-present) branch — keep only var-absent
    // nd.var < b: recurse into both children (b may appear deeper)
    const Key k{... offset-memo on (f,b) ...};
    if (cached) return it->second;
    uint32_t res = mk(nd.var, offset(nd.lo, b), offset(nd.hi, b));
    memoize; return res;
}
```

### 6.2 The per-detector disjoint count and `det_cnt[j]`

For detector `j` with cover mask `mask_j`, the subfamily disjoint from `mask_j` is `offset` chained over every set bit `b` of `mask_j`; its cardinality is the disjoint count:

```cpp
std::vector<uint64_t> all_detector_counts(std::span<const uint32_t> masks) const {
    const uint64_t nb = count();
    std::vector<uint64_t> det_cnt(masks.size(), 0);
    for (size_t j = 0; j < masks.size(); ++j) {
        uint32_t sub = root_;
        for (int b = 0; b < n_; ++b)
            if ((masks[j] >> b) & 1u) sub = offset(sub, b);   // chain offset over b ∈ mask_j
        const uint64_t disjoint = subfamily_card(sub);        // count() restricted to `sub`
        det_cnt[j] = nb - disjoint;                            // note B.2 row 3
    }
    return det_cnt;
}
```

`subfamily_card(sub)` is the §4.1 cardinality of an arbitrary node (the same bottom-up `card` array, indexed at `sub`; compute the `card` array once and reuse across all j — P1). **Through-line to Part A (B.2):** for a popcount-1 mask (`mask_j = 1<<b`), `det_cnt[j] = nb − #{worlds without bit b} = #{worlds with bit b} = bit_cnt[b]` — the exact §A.4 popcount-1 shortcut, now a special case of the general disjoint query (B.2 "Through-line"). The edge-case section (§9) tests both a popcount-1 and a multi-bit detector mask to pin this.

**Cost:** `O(|mask_j| · |Z|)` per detector (B.2 row 3), `nb`-independent. The `card` array is shared, so only the `offset` chain is per-j.

---

## 7. Phase 2: features byte-identical to the sweep (note B.3, A.5)

The diagram produces the **exact integer counts** `nb`, `bit_cnt`, `det_cnt`. The feature vector becomes byte-identical to the §A.4 golden by running the *identical* Phase-2 `* inv` of features.cpp (note B.3: "the identical Phase 2 (`* inv`) makes the feature vector byte-identical"). One home for Phase-2 so the byte-identity is structural:

```cpp
// belief_diagram.cpp — Phase 2, COPIED VERBATIM from features.cpp belief_features_nonempty's phase 2
// (same float ops, same order: marg_sum in treasure-id order — the P6 watch item). The diagram is the
// ONLY difference (it produced bit_cnt/det_cnt without the per-world sweep); Phase 2 is shared math.
[[nodiscard]] BeliefFeatures belief_features_from_diagram(const BeliefDiagram& z,
                                                         std::span<const uint32_t> masks,
                                                         int N, int nD, double log_nworlds) {
    const uint64_t nb = z.count();
    BeliefFeatures bf;
    bf.marg.assign(N, 0.0); bf.p_pos.assign(nD, 0.0); bf.informative.assign(nD, 0.0);
    if (nb == 0) return bf;                                  // empty: matches belief_features_empty
    const std::vector<uint64_t> bit_cnt = z.all_marginals();
    const std::vector<uint64_t> det_cnt = z.all_detector_counts(masks);
    const double inv = 1.0 / static_cast<double>(nb);
    for (int t = 0; t < N; ++t) { bf.marg[t] = static_cast<double>(bit_cnt[t]) * inv; bf.marg_sum += bf.marg[t]; }
    for (int j = 0; j < nD; ++j) {
        bf.p_pos[j]       = static_cast<double>(det_cnt[j]) * inv;
        bf.informative[j] = (det_cnt[j] > 0 && det_cnt[j] < nb) ? 1.0 : 0.0;
    }
    bf.sharpness = std::log(static_cast<double>(nb)) / log_nworlds;
    bf.nonempty  = 1.0;
    return bf;
}
```

This is byte-identical to `belief_features_nonempty` iff `bit_cnt`/`det_cnt`/`nb` match — which the oracle (§8) asserts as a logic invariant (B.3). Note: `cast<double>(uint64_t count ≤ 15504)` is exact, matching the `int64_t` casts in features.cpp (counts are non-negative, well under 2^53). **One subtlety to honor for P6:** features.cpp's `informative` uses `det_cnt[j] < static_cast<int64_t>(nb)`; here `nb` is `uint64_t`, so the comparison is unsigned but numerically identical for these magnitudes — I keep the comparison forms aligned to avoid any doubt.

---

## 8. Bit-exact validation + faithful-rep check (mirroring `belief_sweep_oracle_check.cpp`)

A standalone probe `chocofarm-belief-diagram-check` mirrors `belief_sweep_oracle_check.cpp` exactly: same `--instance/--faces` CLI, same `opt()` arg parse, same `RESULT: PASS/FAIL` protocol, no redis/no net (note B.6: net the diagram against the sweep, P6 strongest tier). It does **three** independent checks per belief, and ALSO runs the Stage-1 measurement (§10) so one binary serves both stages.

### 8.1 The bit-exact count check (the logic-invariant net — B.3)

For each sample belief, build `Z`, then assert **bit-exact equality of the integer counts** against the production sweep (the existing oracle's reference path, reused):

- `Z.count() == nb` (== `bw` distinct size).
- `Z.all_marginals() == bit_cnt` from `chocofarm::belief_features` (compare the integer `bit_cnt` directly — recover it as `marg[t]*nb` is lossy, so instead the probe recomputes the sweep's `bit_cnt` with the same naive `(w>>t)&1` loop `belief_sweep_oracle_check.cpp` uses, the independent reference path, and compares integer-to-integer).
- `Z.all_detector_counts(masks) == det_cnt` from the naive `env.observe` reference loop (the independent path).
- The derived `BeliefFeatures` (via §7) is **byte-equal** to `chocofarm::belief_features(bw, masks, ...)` using the existing `equal_features` predicate verbatim — this is the end-to-end B.3 claim.

Plus the built-in cross-checks: `Σ_t bit_cnt[t] == K · nb` (catches the §5.2 ZDD-doubling trap), and `count() == members().size()`.

### 8.2 The faithful-representation check (so `|Z|` is trustworthy — Stage 1 CRITICAL)

`|Z|` is only a meaningful decision number if `Z` faithfully represents `bw`. Assert, per belief:

- `set(Z.members()) == set(bw)` (enumerate Z's members, compare as sets — `bw` may be unsorted/duplicated; canonical `members()` is duplicate-free, so compare sorted-unique). **This is the gate that makes the `|Z|` number trustworthy** (Stage 1 requirement: "enumerate Z's members == bw exactly, and count(Z)==nb").
- `Z.count() == |set(bw)|` (cardinality matches the distinct-world count).

If either fails, the `|Z|` measurement is meaningless and the probe `RESULT: FAIL`s loudly before reporting any `|Z|`-vs-`nb` numbers (ADR-0002: fail loud, never report a number off an unfaithful diagram).

### 8.3 Sample beliefs

Reuse `belief_sweep_oracle_check.cpp`'s sample set (the empty belief, prefixes 1/2/3/5/16/100/1000/half/full, a strided every-13th subset) for the bit-exact + faithful-rep checks (those test correctness across cover mixes), AND ADD the **realistic** observation-narrowed beliefs (§10) for the same checks plus the `|Z|`-vs-`nb` measurement.

---

## 9. Edge cases (all five required, each a probe assertion)

| Edge case | Expected ZDD behavior | Assertion |
|---|---|---|
| **Empty belief** (`bw = {}`) | `root_ = ⊥` (id 0); `count()=0`, `node_count()=0`, `members()=[]`. `belief_features_from_diagram` returns the empty struct (matches `belief_features_empty`). | `count()==0 && node_count()==0 && members().empty()`; features byte-equal the empty sweep. |
| **Single world** (`bw = {w}`, popcount 5) | `Z` is a 5-node chain; `count()=1`; `bit_cnt[t]=1` for `t∈w` else 0; `det_cnt[j]=observe(j,w)?1:0`. | `node_count()==5`; counts match the sweep; `members()=={w}`. |
| **Full world-set** (`bw = env.worlds()`, nb=15504) | `Z` = the family of all 5-of-20 subsets — `|Z|` is small and fixed (a "symmetric" K-of-N ZDD has `O(N·K)` nodes); `count()=15504`; `bit_cnt[t] = C(19,4) = 3876` for every t; `det_cnt[j] = nb − #{disjoint}`. **This is the canonical `|Z| ≪ nb` demonstrator** (15504 worlds, ~`O(N·K)≈100` nodes). | counts byte-equal the sweep; log `|Z|` (should be ~tens-to-hundreds, ≪ 15504). |
| **Popcount-1 detector mask** (`mask_j = 1<<b`) | `det_cnt[j] = bit_cnt[b]` exactly (the §A.4 popcount-1 shortcut as the `|mask_j|=1` disjoint-query special case, B.2). | `all_detector_counts()[j] == all_marginals()[b]` for any single-bit mask, AND both == the sweep. |
| **Multi-bit detector mask** (`mask_j`, popcount ≥ 2 — the live faces) | `det_cnt[j] = nb − card(offset chained over all bits of mask_j)`; the chained `offset` exercises the recursive subset0 on overlapping bits. | `all_detector_counts()[j] == sweep det_cnt[j]` for the real `env.face_masks()` (these are the production multi-bit covers). |

The full-world-set and multi-bit-mask cases are the ones that exercise sharing/canonicity and the `offset` recursion respectively — the parts a naive impl gets wrong.

---

## 10. The realistic-belief generator + the `|Z|`-vs-`nb` experiment (Stage 1, CRITICAL — note B.3, B.4)

### 10.1 Why realistic beliefs, not random subsets (the note's regime claim, B.3)

The deciding measurement is `|Z|` vs `nb` (note B.4 final paragraph). **Random subsets have `|Z|≈nb`** (no shared substructure) — they would falsely show no compression and falsely kill Part B. The note is explicit that the win is "exactly when the belief has structure (shared substructure / low width)" and that "ISMCTS information sets often do, because they're defined by observation histories (conjunctions of detector constraints) — precisely what diagrams represent compactly" (B.3). So the realistic generator must produce **observation-narrowed** beliefs, matching how the search actually forms information sets (env.cpp `filter_detector`/`filter_treasure`).

### 10.2 The generator (consistent observation sequences)

Mirror the env's actual belief evolution:

```
generate_realistic(env, depth, rng):
    bw = env.worlds()                            # start from the full world-set
    true_world = bw[rng.uniform(0, bw.size())]   # sample a hidden true world (a real 5-of-20 world)
    for step in 1..depth:
        if rng.coin():                            # a detector observation
            j = rng.uniform(0, nD)
            positive = env.observe(j, true_world) # the OUTCOME consistent with the true world (CRITICAL)
            env.filter_detector(bw, j, positive)  # narrow bw exactly as the search does
        else:                                     # a treasure observation
            i = rng.uniform(0, N)
            present = ((true_world >> i) & 1) != 0
            env.filter_treasure(bw, i, present)
        if bw.empty(): break                      # cannot happen — true_world always survives
    return bw                                     # a realistic information set
```

The outcomes are drawn from a **sampled true world** so the constraints are mutually consistent (the belief is exactly the set of worlds agreeing with the observation history) — `true_world` always survives, so `bw` is never empty, and `bw` is a genuine conjunction-of-constraints information set, the B.3 regime. This is the Stage-1 CRITICAL requirement verbatim ("the full world-set narrowed by RANDOM OBSERVATION SEQUENCES ... with outcomes consistent with a sampled true world, NOT random subsets").

### 10.3 The experiment + report

Across a grid of `depth ∈ {1,2,3,5,8,12}` and `samples` (e.g. 200 per depth) with a fixed seed (reproducibility, ADR-0009):

1. `bw = generate_realistic(env, depth, rng)`.
2. Build `Z`; run the §8 faithful-rep + bit-exact checks (every realistic belief is also a correctness sample — fail loud if any diverges).
3. Record `nb = count()`, `|Z| = node_count()`, and the ratio `|Z|/nb`.
4. Report per depth: `nb` distribution (min/median/max), `|Z|` distribution, median `|Z|/nb`, and the fraction with `|Z| < nb/10` (the "structured" tally). Emit as a small table on stdout (the probe prints it before `RESULT: PASS`).

The interpretation (note B.4): median `|Z|/nb ≪ 1` and growing-`nb`-with-flat-`|Z|` as depth shrinks the belief slowly ⇒ structure exists ⇒ recommend (b). `|Z|/nb ≈ 1` ⇒ shelve the diagram for features, the SIMD sweep wins. The probe prints a one-line verdict echoing this, but does NOT itself decide (a)→(b) — that is the human's call on the reported number (note B.4: "let that number decide").

**Control (non-vacuous, ADR-0011):** also generate a matched set of **random subsets** of the same `nb` sizes and report their `|Z|/nb` — expected ≈ 1. Showing realistic ≪ random on the same axis proves the compression is the *structure*, not an artifact (the 1a/1b mutation-control posture the gumbel parity uses). This guards against a happy-looking number that any belief would produce.

---

## 11. File / target layout (mirroring `belief_sweep_oracle_check`)

### 11.1 Files (each with the ADR-0006 module-docstring header: path + purpose + Public Domain)

- **`cpp/include/chocofarm/belief_diagram.hpp`** — the typed `BeliefDiagram` value-seam (§1.1) + the `belief_features_from_diagram` free declaration (§7). The only header that crosses the boundary; no node ids exposed (B.6, P9).
- **`cpp/src/belief_diagram.cpp`** — the hand-rolled ZDD engine: `mk` (reduce+hash-cons, §2.3), `single`/`zunion`/`include_empty` (build, §3), `count`/`node_count`/`members` (§4), `all_marginals` (§5), `offset`/`all_detector_counts` (§6), `belief_features_from_diagram` (§7). One owner of the engine (P3). Added to the `chocofarm_core` library `add_library(...)` list in `cpp/CMakeLists.txt` alongside `src/features.cpp`.
- **`cpp/src/belief_diagram_check.cpp`** — the standalone probe (§8, §9, §10), structured exactly like `belief_sweep_oracle_check.cpp`: `opt()`/`fail()` helpers, the independent naive `reference()` count (reuse the existing one's body — `bit_cnt` via `(w>>t)&1`, `det_cnt` via `env.observe`), `equal_features` verbatim, `--instance/--faces` CLI, `RESULT: PASS/FAIL`. Prints the §10 `|Z|`-vs-`nb` table.

### 11.2 CMake target (append to `cpp/CMakeLists.txt`, mirroring the oracle target block at lines 203–209)

```cmake
# The belief DECISION-DIAGRAM (ZDD) probe (NOT the runner): the B.4(a) on-ramp from
# belief_features_and_decision_diagram_note.md Part B. Builds a hand-rolled ZDD Z from each sample
# belief and (Stage 1) asserts a FAITHFUL representation (members(Z)==bw, count(Z)==nb) then MEASURES
# |Z| vs nb on REALISTIC observation-narrowed beliefs (the (a)->(b) decision number); (Stage 2) answers
# the all-marginals (forward x backward sweep) + per-detector disjoint-count queries and asserts they
# EQUAL the §A.4 sweep's integer counts bit-exact (B.2/B.3, the logic invariant) -> features byte-
# identical. Pure compute (no redis/net), cwd-independent. Separate from the runner (P3, one-owner);
# mirrors chocofarm-belief-sweep-oracle-check.
add_executable(chocofarm-belief-diagram-check src/belief_diagram_check.cpp)
target_link_libraries(chocofarm-belief-diagram-check PRIVATE chocofarm_core)
target_compile_options(chocofarm-belief-diagram-check PRIVATE -Wall -Wextra)
```

And add `src/belief_diagram.cpp` to the `chocofarm_core` source list (line 114–130 block).

### 11.3 The pytest gate (append to `tests/test_cpp_runner.py`, mirroring `test_cpp_belief_sweep_oracle`)

```python
BELIEF_DIAGRAM_BIN = os.path.join(REPO, "cpp", "build", "chocofarm-belief-diagram-check")

@pytest.mark.skipif(not (_RUN_CPP and os.path.exists(BELIEF_DIAGRAM_BIN)), reason=_CPP_SKIP)
def test_cpp_belief_diagram_zdd():
    """The B.4(a) belief-ZDD on-ramp (belief_features_and_decision_diagram_note.md Part B). The probe
    builds a hand-rolled ZDD Z from each belief and asserts (1) FAITHFUL representation
    (members(Z)==bw, count(Z)==nb) so |Z| is trustworthy, (2) the B.2 queries — all-marginals (the
    forward x backward sweep) and per-detector disjoint-count — EQUAL the §A.4 sweep's integer counts
    bit-exact (the logic invariant, B.3), and the derived features are byte-identical, and (3) it
    reports |Z| vs nb on REALISTIC observation-narrowed beliefs (the (a)->(b) decision number, NOT
    random subsets which have |Z|~nb). ADR-0011: net the diagram against the sweep, don't trust it.
    Pure compute (no redis); cwd=REPO/PYTHONPATH kept only for parity with the other cpp gates."""
    out = subprocess.run([BELIEF_DIAGRAM_BIN, "--instance", DATA_INSTANCE, "--faces", DATA_FACES],
                         cwd=REPO, capture_output=True, text=True, timeout=120,
                         env={**os.environ, "PYTHONPATH": REPO})
    sys.stdout.write(out.stdout)
    sys.stderr.write(out.stderr)
    assert out.returncode == 0 and "RESULT: PASS" in out.stdout
```

It is OPT-IN under `CHOCO_RUN_CPP=1` like every other cpp gate (the default suite stays green without the C++ build).

---

## 12. Correctness traps (consolidated — note's "Note correctness traps")

1. **The ZDD reduction rule is NOT the BDD rule (§2.3).** Zero-suppress when `hi == ⊥` (return `lo`); do **not** eliminate when `lo == hi` (that is BDD's rule and corrupts the count). This is the rule that produces `|Z| ≪ nb` and the one a BDD reflex gets backwards. The faithful-rep check (§8.2) and `count()==members().size()` catch a wrong rule instantly.
2. **Variable ordering must be fixed and consistent (§2.2).** Treasure-id order `0<...<N-1`, strictly increasing down both edges, terminals as `var=N`. Canonicity (hence a meaningful `|Z|`) depends on it. `single`, `zunion`, `offset`, and the sweeps all assume it; a single off-order `mk` breaks canonicity silently (counts may still pass, `|Z|` becomes meaningless). Assert in `mk` (debug): `node[lo].var > var && node[hi].var > var`.
3. **The all-marginals ZDD doubling trap (§5.2).** A variable skipped between parent and child is *unconditionally absent* in a ZDD (factor 1), not a free 0/1 choice (factor 2 — the BDD rule). Never multiply `up`/`down` by `2^skipped`. The cross-check `Σ_t bit_cnt[t] == K·nb` (§8.1) is the canary.
4. **Forward-pass order (§5.3).** `up` must be computed parents-before-children (descending id loop — valid because `mk` appends children-first); `down`/`card` children-before-parents (ascending). Swapping them silently zeros the counts.
5. **`members()`/`bw` order mismatch (§4.3, §8.2).** Compare as sets, never element-wise — `bw` is unsorted/possibly-duplicated; `members()` is canonical-ordered and duplicate-free.
6. **Distinct-world semantics (§3.2).** The family is the *set* of worlds; if `bw` carries duplicates, `count()` = distinct count, so the faithful-rep check compares against `set(bw)`, and the sweep's `nb` (which counts `bw.size()`) only matches when `bw` is already duplicate-free (it is, in production — the env filters never duplicate). The probe asserts `bw` is duplicate-free on the realistic/sample beliefs so the `nb` definitions coincide; if not, it fails loud rather than papering over it (ADR-0002).

---

## 13. ADR hygiene (note B.6, the documentation-is-part-of-the-work CLAUDE.md rule)

- **B.6 satisfied:** `BeliefDiagram` is a one-owner collaborator (P3), a typed value seam with no node ids leaking (P9); the diagram counts are a logic invariant netted bit-exact vs the sweep, which stays the oracle during bring-up (P6 strongest tier); this is the feature-time prototype (a), behind the existing belief surface — if it graduates to (b) it is "the belief surface reimplemented behind the seam, not a new wire fact" (P7/§3 untouched).
- **Documentation deliverables** (required to call this complete per CLAUDE.md): a short design note `docs/design/belief-zdd-onramp.md` recording this design and the eventual `|Z|`-vs-`nb` numbers; and when the measurement lands, an append-only dated record of the (a)→(b) verdict. The §A.4/B.3 comments in `features.cpp` and `belief_sweep_oracle_check.cpp` already *anticipate* the Part B diagram ("every later rung ... the Part B diagram ... matches it bit-for-bit") — no edit needed there, the prediction is now realized by the new probe; the new files cite the note's B.x sections in their headers (ADR-0006).

---

## Summary of deliverables

| Component | Where | Note section |
|---|---|---|
| ZDD node rep + terminals | §2.2 | B.1, B.5 |
| Hash-cons unique table | §2.3 (`mk`, `unique_`) | B.5 |
| Zero-suppression reduction rule | §2.3 (the trap) | B.5 |
| Build-from-worlds | §3 (`single`+`zunion` tournament fold) | B.4a |
| Cardinality `count()==nb` | §4.1 | B.2 row 1 |
| Node count `|Z|` | §4.2 | B.4 |
| Member enumeration | §4.3 | Stage-1 faithful-rep |
| **All-marginals (fwd×bwd sweep)** | **§5 (my emphasis)** | **B.2 row 2** |
| Disjoint-count `det_cnt[j]` | §6 (`offset` chain) | B.2 row 3 |
| Realistic-belief generator | §10.2 | B.3, B.4 |
| `|Z|`-vs-`nb` experiment + control | §10.3 | B.4 |
| Bit-exact validation + faithful-rep | §8 | B.3 |
| Edge cases (5) | §9 | — |
| Correctness traps | §12 | — |
| Typed `BeliefDiagram` seam | §1 (my emphasis) | B.5, B.6 |
| File/target/pytest layout | §11 | mirrors oracle |

Files to create (all absolute): `/home/bork/w/vdc/1/chocofarm/cpp/include/chocofarm/belief_diagram.hpp`, `/home/bork/w/vdc/1/chocofarm/cpp/src/belief_diagram.cpp`, `/home/bork/w/vdc/1/chocofarm/cpp/src/belief_diagram_check.cpp`, `/home/bork/w/vdc/1/chocofarm/docs/design/belief-zdd-onramp.md`. Files to edit: `/home/bork/w/vdc/1/chocofarm/cpp/CMakeLists.txt` (core source list + new target), `/home/bork/w/vdc/1/chocofarm/tests/test_cpp_runner.py` (the gate).


---

## Proposal 3

I have everything. Here is the complete design.

---

# Belief Decision Diagram (ZDD) On-Ramp — B.4(a) Staged Design

**Emphasis: measurement rigor (angle C).** The whole `(a)→(b)` decision in the note (§B.4: "*The measurement that decides (a)→(b): build `Z` for a sample of real beliefs and log `|Z|` vs `nb`*") rests on one number — `|Z|`. A `|Z|`-vs-`nb` plot is worthless if (1) the beliefs are not faithful to the search, or (2) the `|Z|` count is not a trustworthy count of a structure that actually represents `bw`. So this design leads with the **realistic-belief generator** and an **airtight faithful-representation + bit-exact validation**, then fills in the data structure and algorithms those rest on. It cites §B.1 (mapping), §B.2 (queries-as-counts), §B.3 (validation), §B.4 (on-ramp + the deciding measurement), §B.5 (hand-roll), §B.6 (ADR hygiene).

---

## 0. The two trust failures this design is built to prevent

Both are silent — they produce a clean number that is wrong.

**Trust failure 1 — measuring the wrong beliefs (the note's explicit trap, §B.4).** The note's `(a)→(b)` payoff is conditional on ISMCTS information sets having structure: "*defined by observation histories (conjunctions of detector constraints) — precisely what diagrams represent compactly, and precisely the regime where `nb` can be astronomical while `|Z|` stays small*" (§B.3). A **random subset** of `worlds()` has *no* such structure: it is an arbitrary family with `|Z| ≈ nb` (every member needs its own path). Measuring random subsets would report `|Z| ≈ nb` and **falsely kill Part B** — concluding "no structure, shelve the diagram" when the real search beliefs (conjunctions of filters) are exactly the structured regime. The prompt names this explicitly. **The realistic generator (`§3`) is therefore the single load-bearing component of the measurement.**

**Trust failure 2 — trusting `|Z|` from a structure that does not faithfully represent `bw`.** A ZDD with a bug (wrong reduction rule, wrong variable handling) can have a small node count while enumerating a *different* family. Its `|Z|` is then meaninglessly small. So before any `|Z|` is logged, we prove the diagram **is** the belief: enumerate `Z`'s members and assert set-equality with `bw`, and assert `count(Z) == nb` (`§5`). This is the **faithful-representation check** the prompt demands; `|Z|` is only reported for diagrams that pass it.

These map onto the note's safety net (§B.3): "*`bit_cnt` and `det_cnt` from the diagram are exact integers that must equal the sweep's — a logic invariant, so the test is `assert(diagram_counts == sweep_counts)` per belief, bit-exact.*" The faithful-rep check is the strictly-stronger version (set-equality of members ⊇ count-equality), run as the Stage-1 gate; the count-equality vs the sweep is the Stage-2 gate.

---

## 1. The mapping and the variable ordering (§B.1)

A belief is a set of worlds; each world is an `N=20`-bit subset with **exactly K=5 bits set** (env fact; `build_worlds` in `env.cpp` enumerates `C(20,5)=15504`). That is "*a family of subsets over an N-element universe — the canonical object a decision diagram represents*" (§B.1). We take the **ZDD** branch (§B.1: "*the representation specialized for sparse sets of subsets — more compact when worlds carry few treasures, which is the common case*"). 5-of-20 is exactly the sparse regime ZDDs compress.

**Universe and variables.** The universe is the `N=20` treasure bits `x_0 .. x_{N-1}`. A world `w` (a `uint32_t`) is the subset `{ t : (w>>t)&1 }`. A ZDD over these 20 variables represents a family of such subsets.

**Variable ordering (a correctness trap — fixed once, globally).** The ZDD invariant is that along any path, variable indices are **strictly increasing** from root to terminal. We fix the order to be **the natural treasure-id order `0 < 1 < ... < 19`**, top = variable 0. This choice is not arbitrary and must be honored everywhere:
- `build_from_worlds` must insert each world's bits in increasing id order.
- The all-marginals sweep (`§7`) and the disjoint-count (`§8`) both assume this order.
- Mismatched orderings between build and query silently corrupt counts. The ordering is a single `constexpr` fact (variable index == treasure id); there is no reordering heuristic (sifting) in the hand-roll — it would be a second writer of canonical form and is out of scope.

A node tests one variable `var ∈ [0,N)`; the **0-edge (`lo`)** is "this treasure absent", the **1-edge (`hi`)** is "this treasure present".

---

## 2. ZDD node representation, terminals, reduction rule, hash-cons (§B.5 hand-roll)

Hand-rolled, "*a few hundred lines, behind a typed `BeliefDiagram` value-seam*" (§B.5), one-owner collaborator wrapping the engine (§B.6, P3). No CUDD/Sylvan dependency for Stage 1 — the measurement does not need it, and a library adds a build dep before the `|Z|` number justifies it (§B.5 sequencing: hand-roll to *understand*, swap engine "*if it pays*").

### 2.1 Terminals

Two canonical terminals, by convention the first two node ids:

- **`BOT = 0`** (the ∅ family, "empty set of subsets" — *no* members). The ZDD false terminal.
- **`TOP = 1`** (the {∅} family — *one* member, the empty subset). The ZDD true terminal.

`count(BOT)=0`, `count(TOP)=1`. (Standard ZDD terminal semantics — distinct from BDD: a ZDD's `1`-terminal denotes the family containing the empty set, not "all assignments".)

### 2.2 Internal node

```cpp
struct ZNode {                 // one node = one decision on variable `var`
    int32_t var;               // tested treasure id in [0, N); -1 reserved for terminals
    int32_t lo;                // 0-edge child id (treasure `var` ABSENT)
    int32_t hi;                // 1-edge child id (treasure `var` PRESENT)
};
```

Nodes live in a single `std::vector<ZNode> nodes_` indexed by `int32_t` id. Ids 0 and 1 are the terminals (their `var=-1`, children unused). `|Z|` = **number of distinct internal nodes reachable from the root** (terminals are shared and excluded from the structural count, matching the note's "*node count |Z| (reduced unique nodes)*"). We report both "internal nodes reachable from root" (the headline `|Z|`) and "total unique nodes in the table" if useful; the headline is reachable-internal so it is comparable to `nb` apples-to-apples.

### 2.3 The two canonicity rules (the central correctness trap — §B.5, "the ZDD reduction rule")

A ZDD is *reduced* iff both hold for every internal node:

1. **Zero-suppression rule (the ZDD-specific one).** If a node's **1-edge points to `BOT`** (`hi == BOT`), the node is *eliminated* — it is replaced by its `lo` child. Rationale: a node whose "present" branch leads nowhere contributes no member that sets `var`, so testing `var` is redundant; suppressing it is what makes the diagram compact on sparse families. **This is the rule a BDD does NOT have, and getting it wrong (e.g. applying the BDD rule "eliminate if `lo==hi`") yields a structure that is small but represents the wrong family.** The mk-node helper (`§2.4`) enforces exactly this and nothing else.

2. **Merge rule (hash-consing / unique-table).** No two distinct nodes have identical `(var, lo, hi)`. Enforced by the unique table.

Worked trap statement, to bolt onto the code: in a ZDD you suppress `hi==BOT` (zero-suppression), **not** `lo==hi`; in a BDD you suppress `lo==hi`, **not** `hi==BOT`. Mixing them is the classic silent corruption. The faithful-rep check (`§5`) is precisely the net that catches a mis-implemented rule, because a wrong rule changes the *member set*, and member-set equality with `bw` would fail.

### 2.4 Unique table + the `mk` (make-or-find canonical node) primitive

```cpp
// key = (var, lo, hi) -> node id. The hash-cons table.
std::unordered_map<uint64_t, int32_t> unique_;   // pack(var,lo,hi) -> id

static uint64_t pack(int32_t var, int32_t lo, int32_t hi) {
    // var < 32 (< 64 to be safe), lo/hi are vector indices. Pack into 64 bits.
    return (uint64_t(uint32_t(var)) << 48) ^ (uint64_t(uint32_t(lo)) << 24) ^ uint64_t(uint32_t(hi));
    // NOTE: with |Z| small this is fine; for safety against collision use a struct key + std::hash,
    //       or a 3-field tuple key. (See trap note below — prefer the collision-free tuple key.)
}

int32_t mk(int32_t var, int32_t lo, int32_t hi) {
    if (hi == BOT) return lo;                 // (1) ZERO-SUPPRESSION rule — THE ZDD rule
    auto key = key_of(var, lo, hi);           // (2) MERGE: canonical (var,lo,hi)
    if (auto it = unique_.find(key); it != unique_.end()) return it->second;
    int32_t id = static_cast<int32_t>(nodes_.size());
    nodes_.push_back(ZNode{var, lo, hi});
    unique_.emplace(key, id);
    return id;
}
```

**`mk` is the only function that creates internal nodes.** Every construction path (build, offset/disjoint-subfamily) goes through it, so canonicity is structural — there is no way to author a non-reduced node. (Mechanize > assert, the codebase's posture in `features.cpp` ctor.)

**Collision trap.** `unordered_map` keyed on a *lossy* `pack` would silently merge distinct nodes and shrink `|Z|` artificially (trust failure 2). Use a **lossless** key: either a `struct {int32_t var,lo,hi;}` with a custom `std::hash` and `operator==`, or `std::map<std::tuple<int,int,int>,int>`. With `|Z|` expected small (the whole hypothesis), a tuple-keyed `std::map` is fast enough and provably collision-free. **Choose the lossless key** — measurement rigor over micro-perf at Stage 1.

---

## 3. The realistic-belief generator (LEAD COMPONENT — §B.4, faithful to the search)

This is the component the entire `|Z|`-vs-`nb` conclusion stands on. The note: "*realistic beliefs = the full world-set narrowed by RANDOM OBSERVATION SEQUENCES (filter_detector / filter_treasure with outcomes consistent with a sampled true world), NOT random subsets.*"

### 3.1 Why a sampled true world (the consistency requirement)

A belief in the real search is **always non-empty and always consistent with how it was formed**: it is `worlds()` filtered by a sequence of observations, each observation's outcome being what the *true world* would yield. If we filtered by *random* outcomes we could (a) drive `bw` empty (an unobserved, off-distribution state) or (b) produce inconsistent constraint sets the search never visits. Sampling one true world `w*` up front and taking every observation's outcome *from `w*`* guarantees `w* ∈ bw` at all times — so `nb ≥ 1` always, and the belief is exactly an information-set the search could be in. This mirrors the env's own `apply` (`env.cpp:117`), where the observation outcome is read from the passed `world` (`observe(action.i, world)`, `((world>>i)&1)` for treasures).

### 3.2 The generator (exact algorithm)

Inputs: the `Environment env`, a depth `D` (number of observations), a `std::mt19937_64 rng`. Outputs: a belief `bw` (sorted `std::vector<uint32_t>`) that is `worlds()` narrowed by `D` consistent observations, with metadata `(depth, true_world, applied_ops)`.

```
generate_realistic_belief(env, D, rng):
    worlds = env.worlds()                       # the full C(20,5) set
    w_star = worlds[ uniform(0, worlds.size()) ] (rng)   # the sampled TRUE world (consistency anchor)
    bw = worlds                                 # start from the full belief (the search's t=0 state)
    applied = []
    for step in 1..D:
        # choose an action that is still INFORMATIVE over bw (mirrors env.legal_actions:
        #   a detector is offered iff env.informative(j,bw); a treasure-probe is meaningful
        #   iff 0 < marg < 1, i.e. the bit is uncertain over bw). Off-distribution otherwise.
        cands = []
        for j in 0..nD-1: if env.informative(j, bw): cands.push( ("d", j) )
        for i in 0..N-1:  if 0 < bitcount_i(bw) < bw.size(): cands.push( ("t", i) )
        if cands empty: break                   # belief fully determined; stop early (record actual depth)
        a = cands[ uniform(0, cands.size()) ](rng)
        if a is ("d", j):
            pos = env.observe(j, w_star)        # outcome consistent with the TRUE world
            env.filter_detector(bw, j, pos)     # the SAME filter the search uses (env.cpp:111)
            applied.push( ("d", j, pos) )
        else:                                    # ("t", i)
            present = ((w_star >> i) & 1) != 0
            env.filter_treasure(bw, i, present)  # env.cpp:105
            applied.push( ("t", i, present) )
    sort(bw)                                     # canonical order for set-compares / determinism
    return Belief{ bw, w_star, D_actual = applied.size(), applied }
```

Key fidelity properties, each load-bearing:

1. **Consistency invariant (asserted):** `w_star ∈ bw` at every step and at return ⇒ `nb ≥ 1`. We `assert` it in the probe; a violation means a filter disagrees with `observe` (a real bug, caught loudly per ADR-0002).
2. **Same filters as the search.** `filter_detector` / `filter_treasure` are `env`'s own methods — the belief is byte-identical to a search information-set, not a synthetic approximation. (No reimplementation of the filter in the generator: one home, the env, per ADR-0012 P1.)
3. **Informative-only action choice.** Restricting candidates to informative detectors and uncertain treasures (the env's own legality test, `env.cpp:80,90`) keeps every step a *real* search move — a redundant observation (one that filters nothing) would inflate the "depth" without changing the belief and is exactly what the search never does.
4. **Two probe families.** Detector observations are *disjunctive* constraints (the structured case the note bets on); treasure observations are *single-bit* constraints. The note (§B.3) says ISMCTS sets are "*conjunctions of detector constraints*", so the **detector-only** sweep is the headline; we also run a **mixed** sweep (detectors + treasure probes) and a **detector-only deep** sweep to show the trend across constraint type.

### 3.3 The sampling grid (so the `|Z|`-vs-`nb` plot is not cherry-picked)

For each depth `D ∈ {0, 1, 2, 3, 5, 8, 12, 20}` (D=0 is the full world-set; large D drives `nb` small), draw `S` independent samples (e.g. `S=64`) with distinct rng seeds. For each sample log a row: `(D, seed, nb, |Z|, ratio=|Z|/nb, count_nodes_total, faithful_ok)`. Report per-depth aggregates: `nb` range, `|Z|` range, **min/median/max of `|Z|/nb`**, and the fraction with `|Z| < nb/2` (a coarse "structured" flag). The median ratio across the grid **is the `(a)→(b)` decision number** (§B.4).

### 3.4 The control arm (proves the generator matters)

Critically — and this is the measurement-rigor payoff — for **the same `nb`**, also build `Z` from a **random subset** of `worlds()` of size `nb` and log its `|Z|`. The expected, validating outcome (the note's premise, §B.3/§B.4): the realistic belief at that `nb` has `|Z| ≪ nb` while the random subset has `|Z| ≈ nb`. Reporting the two side by side **demonstrates** that the conclusion is driven by structure, not by `nb` — and pre-empts the exact misread ("you only got a small `|Z|` because `nb` is small") that would otherwise make the plot unconvincing. If the realistic arm did *not* beat the random arm, that is itself the honest finding (no structure → shelve, §B.4) — but we would *know* it rather than guess it.

---

## 4. build-from-worlds (Z = the family of the bw worlds)

Build `Z` so it represents **exactly** the set `bw` (each world a member subset). Two equivalent strategies; we specify the simple, order-correct one and note the faster one.

### 4.1 `single_world(w)` — the ZDD for one world (a chain)

A world `w` with set bits `t_1 < t_2 < ... < t_5` is the family `{ {t_1,...,t_5} }` — a single member. Its reduced ZDD is the chain that, for each set bit, takes the 1-edge, and for unset variables is *suppressed* (zero-suppression handles the absent variables automatically — we only build nodes for the **set** bits):

```
single_world(w):
    node = TOP                                  # base: the family {∅}
    for t = N-1 down to 0:                       # build bottom-up so children exist first
        if (w>>t)&1:
            node = mk(t, BOT, node)              # var t PRESENT: lo=BOT (absent ⇒ not a member), hi=node
        # else: variable t is suppressed (mk would zero-suppress hi==BOT anyway, but we skip it)
    return node
```

Note `lo=BOT`: if treasure `t` is *absent*, this world is not a member, so the 0-edge is the empty family. `hi=node` continues the chain. Building top variable last (loop high→low) means each `mk`'s children already exist. This yields a chain of exactly **5 internal nodes** for a 5-of-20 world (zero-suppression elides the 15 absent variables — the sparse-regime compaction in action).

### 4.2 union of two ZDDs (`apply ∪`, memoized — §B.5 "memoized apply")

```
union(a, b):                                     # ZDD set-union
    if a == BOT: return b
    if b == BOT: return a
    if a == b:   return a
    key = (UNION, min(a,b), max(a,b))            # commutative ⇒ canonicalize operand order
    if memo has key: return memo[key]
    (va, vb) = (var(a), var(b))                  # terminals: TOP has var = +inf (treat as below all)
    if va == vb:
        r = mk(va, union(lo(a),lo(b)), union(hi(a),hi(b)))
    elif va < vb:                                # a tests the higher (smaller-index) variable
        r = mk(va, union(lo(a), b), hi(a))       # b is constant w.r.t. var va ⇒ rides the 0-edge
    else:                                        # vb < va, symmetric
        r = mk(vb, union(a, lo(b)), hi(b))
    memo[key] = r
    return r
```

The `var(TOP)` / `var(BOT)` sentinel: terminals are "below" all variables, so set their effective var to `N` (or `INT_MAX`) in `var()`. The "`b` rides the 0-edge" case is correct because in a ZDD, a node absent for variable `va` means `va` is *not in any member of `b`* — so `b` contributes only to the 0-edge (var absent), matching zero-suppression semantics. (This is the standard ZDD-union and is the one place ordering and the rule interact — another correctness trap; it is netted by `§5`.)

### 4.3 build-from-worlds

```
build_from_worlds(bw):
    root = BOT                                   # empty family
    for w in bw:
        root = union(root, single_world(w))
    return root
```

Cost `O(nb · log)`-ish (note §B.4(a): "*build Z from the current explicit bw once per belief (O(nb log))*"). Correctness does not depend on `bw` being sorted or unique (union is idempotent and commutative) — but the generator sorts/dedups, and `bw` from filters is already unique. **Faster alternative (noted, not required for Stage 1):** a direct radix/trie build that inserts worlds in sorted order and hash-conses bottom-up in one pass — but `union` is simpler to validate and the build is not the measured cost (the *query* is). Keep `union` for Stage 1.

---

## 5. Cardinality, member-enumeration, node-count, and the FAITHFUL-REPRESENTATION check (the Stage-1 gate)

### 5.1 `count(Z)` — cardinality (`nb`) (§B.2 row 1)

"*one bottom-up pass, memoized per node*":

```
count(node):                                    # number of members (subsets) in the family
    if node == BOT: return 0
    if node == TOP: return 1
    if memo has node: return memo[node]
    r = count(lo(node)) + count(hi(node))        # 0-edge members + 1-edge members
    memo[node] = r
    return r                                      # int64 — max nb = 15504, no overflow
```

ZDD count is `count(lo)+count(hi)` (no `2^skip` factor — zero-suppression means skipped variables are *absent*, contributing exactly one way, not two). This is the ZDD-vs-BDD count trap; getting it wrong is caught by `count(Z) != nb` in the gate.

### 5.2 `enumerate(Z)` — materialize the member set (the faithful-rep witness)

```
enumerate(node, acc_bits, out):
    if node == BOT: return
    if node == TOP: out.push_back(acc_bits); return    # acc_bits is one complete member world
    enumerate(lo(node), acc_bits, out)                  # var absent
    enumerate(hi(node), acc_bits | (1<<var(node)), out) # var present
```

Returns every member as a `uint32_t` world. `enumerate(root)` yields exactly the family's members.

### 5.3 `node_count(Z)` — `|Z|` (reachable internal nodes)

```
node_count(node):                               # DFS marking reachable internal node ids
    visited = {}; stack = [node]
    while stack:
        n = stack.pop()
        if n == BOT or n == TOP or n in visited: continue
        visited.add(n); stack.push(lo(n)); stack.push(hi(n))
    return |visited|                             # |Z| — the headline measurement
```

### 5.4 The faithful-representation check (THE Stage-1 trust gate — prevents trust failure 2)

For a belief `bw`, build `Z`, then assert **all** of:

```
faithful(Z, bw):
    members = enumerate(Z)                       # as a set
    bw_set  = set(bw)
    assert set(members) == bw_set                # (i) MEMBER-SET EQUALITY — the strongest witness
    assert members.size() == bw.size()           # (ii) no duplicate members (ZDD members are distinct)
    assert count(Z) == bw.size()                 # (iii) cardinality equals nb (the §B.2 count query)
    # (i) ⊇ (iii), but assert both: a count bug that happens to total right but enumerates wrong is caught by (i);
    # a count recursion bug is caught by (iii) directly. Cross-checking the two is the airtight part.
    return true (else abort loudly, ADR-0002)
```

Only after `faithful(Z, bw)` passes is `|Z| = node_count(Z)` logged. **`|Z|` from a non-faithful diagram is never reported** — that is the discipline that makes the `|Z|`-vs-`nb` number trustworthy (the prompt's "*so the |Z| number is trustworthy*").

This is strictly stronger than the note's §B.3 count-assert (which compares integer counts to the sweep); §B.3's count-assert is the Stage-2 gate (`§9`), the faithful-rep check is the Stage-1 gate. Both run.

### 5.5 Edge cases (each an explicit faithful-rep case in the probe)

| Case | belief | expected ZDD | expected `count` / `|Z|` |
| — | — | — | — |
| **empty belief** | `bw = {}` | `Z = BOT` | `count=0`, `|Z|=0` |
| **single world** | `bw = {w}` (5 bits) | a 5-node chain | `count=1`, `|Z|=5` |
| **full world-set** | `bw = worlds()` | the ZDD of *all* 5-of-20 subsets | `count=15504`, `|Z|` small (high sharing) |
| **popcount-1 detector mask** | belief filtered by a 1-bit detector | (used in `§8` disjoint test) | `count` = `nb − #without that bit` |
| **multi-bit detector mask** | belief filtered by a real disjunctive face | (used in `§8`) | the disjunction test |

The empty belief is the dispatcher's `nb==0` branch (mirrors `belief_features_empty`, `features.cpp:166`): `build_from_worlds({})` returns `BOT`, every query returns 0 — exactly the sweep's all-zero `BeliefFeatures`. The single-world chain pins the zero-suppression rule (a chain of exactly K=5 nodes; if the rule were the BDD rule you'd get a different shape). The full world-set is the upper-bound stress (largest `nb`, must still enumerate to exactly `worlds()`).

---

## 6. Stage 1 deliverable summary (the decision gate)

Stage 1 ships: `mk`/unique-table/zero-suppression (`§2`), `single_world`+`union`+`build_from_worlds` (`§4`), `count`/`enumerate`/`node_count` (`§5`), `faithful` (`§5.4`), the realistic-belief generator + control arm (`§3`), the edge cases (`§5.5`), and the `|Z|`-vs-`nb` experiment that emits the per-depth table and the headline median ratio (`§3.3`). It answers the note's deciding question (§B.4) with a *trustworthy* number, because every reported `|Z|` passed faithful-rep and the realistic generator guarantees the beliefs are search-faithful. Stage 1 does **not** yet implement the marginal/disjoint queries — that is Stage 2.

---

## 7. Stage 2 — all-marginals `bit_cnt[t]` via one forward × backward sweep (§B.2 row 2)

The note: "*the all-marginals query: `#{worlds with bit t set}` for every t from a single forward (path-counts to each node) × backward (sub-counts below each node) sweep — O(|Z|) shared work, not N independent queries.*" Concretely, for treasure `t`, `bit_cnt[t] = #{ members of Z whose subset contains t }`. The standard decision-diagram marginal computation:

Define for each node `n`:
- **`below[n] = count(n)`** = number of members in the sub-family rooted at `n` (the `§5.1` count, memoized — the *backward / subtree-counts* sweep).
- **`above[n]`** = number of *partial paths from the root to `n`* (the *forward / path-counts* sweep). `above[root]=1`; for an edge `n --(lo or hi)--> c`, that edge contributes `above[n]` paths to `c`.

Then a member contains treasure `t` iff its path takes a **1-edge out of a node with `var=t`**. The number of such members is summed over all nodes testing `t`:

```
all_marginals(Z, N) -> bit_cnt[0..N-1]:
    # backward: below[n] = count(n) for every node (memoized count, §5.1)
    # forward:  above[n] = sum of path-counts from root (topological, parents before children)
    above[root] = 1; above[everything else] = 0
    for n in topological order (root first, by increasing... see ordering note):
        if n is terminal: continue
        above[lo(n)] += above[n]
        above[hi(n)] += above[n]
    bit_cnt = [0]*N
    for each internal node n:
        # members through n's 1-edge = above[n] (ways to reach n) * below[hi(n)] (ways to finish below the 1-edge)
        bit_cnt[ var(n) ] += above[n] * below[ hi(n) ]
    return bit_cnt
```

**Why this is exactly `bit_cnt[t]`.** Every member of `Z` corresponds to exactly one root→TOP path. That member contains `t` iff its path leaves *some* node with `var=t` via the 1-edge (zero-suppression guarantees: if a node for `t` is absent on a path, `t` is absent from that member). The members passing through a given node `n` via its 1-edge number `above[n] · below[hi(n)]` (paths-in × completions-below). Summing over all `var(n)==t` nodes counts each `t`-containing member exactly once (a path leaves at most one `t`-node, since each variable appears at most once per path under strict ordering). Cost: `O(|Z|)` for both sweeps + `O(|Z|)` accumulation = **`O(|Z|)` total, independent of `nb`** — the note's "*not N independent queries*". This is the through-line to Part A's `bit_cnt[t] = Σ_w bit_t(w)` (`features.cpp:191`).

**Topological-order trap (the all-marginals sweep trap, §B.2).** The forward sweep requires **all parents of `n` processed before `n`** (so `above[n]` is final before it is pushed down). Because the variable ordering is strict-increasing root→leaf, a valid topological order is **by increasing `var`** (with terminals last). Process nodes grouped by `var` ascending; within a var the order is free. If you push `above` in the wrong order you double-count or undercount — caught by the bit-exact assert vs the sweep (`§9`). Implementation: bucket internal nodes by `var` (a `vector<vector<int>>` of size N) during a reachability DFS, then iterate `var = 0..N-1`.

---

## 8. Stage 2 — per-detector disjoint-count `det_cnt[j]` (§B.2 row 3)

The note: "*`det_cnt[j] = nb − #{worlds disjoint from mask_j}`. The disjoint count = the sub-family of Z avoiding every bit in mask_j (ZDD: chain `offset(b)` over `b ∈ mask_j`, then count).*" `det_cnt[j] = #{ worlds w ∈ bw : (w & mask_j) ≠ 0 }` (= `env.observe(j,w)`, matching `features.cpp:192`). A world is *disjoint* from `mask_j` iff it sets **none** of the bits in `mask_j`. So:

```
det_cnt[j] = nb - count( subfamily of Z avoiding every bit in mask_j )
```

The "subfamily avoiding bit `b`" is the ZDD operation that **removes every member containing `b`**, then keeps the rest — i.e. project onto members where `b` is absent. For a single bit it is exactly the **0-edge restriction** (in ZDD terms, the family `{S ∈ Z : b ∉ S}`):

```
avoid_bit(node, b):                             # subfamily of `node` whose members do NOT contain b
    if node == BOT or node == TOP: return node   # terminals contain no var ⇒ unchanged
    if var(node) == b: return lo(node)           # drop the 1-edge (members with b): keep only b-absent
    if var(node) >  b: return node               # b would have appeared above; absent ⇒ no member has b here on
    # var(node) < b: recurse into both children, rebuild via mk (preserves canonicity)
    key = (AVOID, node, b); if memo has key: return memo[key]
    r = mk(var(node), avoid_bit(lo(node), b), avoid_bit(hi(node), b))
    memo[key] = r; memo[key]=r; return r

avoid_mask(node, mask):                          # subfamily avoiding EVERY bit in mask (chained)
    r = node
    for b in bits_of(mask): r = avoid_bit(r, b)
    return r

det_cnt[j] = nb - count( avoid_mask(Z, mask_j) )
```

Cost `O(|mask_j| · |Z|)` per detector (note §B.2), `nb`-independent. **Through-line (§B.2):** the popcount-1 case `|mask_j|=1` gives `det_cnt[j] = nb − #{without bit b} = #{with bit b} = bit_cnt[b]` — literally Part A's popcount-1 shortcut (`note §A.4`, `features.cpp` popcount-1 harvesting) as the single-bit special case. The probe asserts this identity directly for popcount-1 faces as an extra net.

**Detector trap.** `mask_j` here is over **treasure-id bits** (the same `face_masks()` the sweep reads, `env.hpp:109`); `observe(j,w) == ((w & face_masks()[j]) != 0)` (env's documented identity). The ZDD variables *are* the treasure bits, so `avoid_bit(_, b)` for `b ∈ mask_j` is well-defined directly — no index translation. Multi-bit faces chain `avoid_bit` over all set bits of the mask; order of chaining is irrelevant (commutes), but each step must go through `mk` to stay reduced.

---

## 9. Stage 2 validation — bit-exact vs the sweep (§B.3, the note's safety net)

Stage 2's gate is the note's `assert(diagram_counts == sweep_counts)` per belief, bit-exact (§B.3, P6 strongest tier, §B.6). Mirror `belief_sweep_oracle_check.cpp` exactly:

1. For each realistic belief `bw` (the `§3` generator + the edge cases `§5.5`):
   - **Integer-count assert (the core invariant):** compute `bit_cnt[]` via `all_marginals(Z)` (`§7`) and `det_cnt[]` via `avoid_mask`+`count` (`§8`); compute the **reference** `bit_cnt`/`det_cnt` the exact way the oracle's `reference()` does (`belief_sweep_oracle_check.cpp:54-58`: `bc[t] += (w>>t)&1`, `dc[j] += env.observe(j,w)`). Assert `bit_cnt_zdd == bit_cnt_ref` and `det_cnt_zdd == det_cnt_ref` element-wise (`int64` equality — exact, no float).
   - **Feature byte-identity (closing the loop, §B.3):** feed the ZDD's `bit_cnt`/`det_cnt` through the *identical* Phase-2 `* inv` map (`features.cpp:198-208`) to produce a `BeliefFeatures`, and assert it is **byte-equal** to `chocofarm::belief_features(bw, masks, N, nD, log_nworlds)` using the oracle's `equal_features` (`belief_sweep_oracle_check.cpp:73`). This is the note's "*the identical Phase 2 (\*inv) makes the feature vector byte-identical to the §A golden*". Because the only float op is `* inv` over exact integer counts, `==` is the exact bit test (same justification as the oracle, `belief_sweep_oracle_check.cpp:70-72`).
2. Also assert the faithful-rep check (`§5.4`) — Stage 2 keeps the Stage-1 gate active (members == `bw`, `count == nb`).
3. Print `RESULT: PASS ...` / `RESULT: FAIL ...` exactly like the oracle (`belief_sweep_oracle_check.cpp:130`) so the pytest gate matches on the same string.

Sharing only the math spec (not the code path) between ZDD and reference is the oracle's discipline (`belief_sweep_oracle_check.cpp:8`): matching counts then prove the ZDD recursion (`all_marginals`, `avoid_mask`) is exact, not just internally consistent.

---

## 10. File / target / test layout (mirrors `belief_sweep_oracle_check`, §B.6 P3/P9)

### 10.1 The typed value-seam header (§B.6 P9 — "no raw diagram pointers across the boundary")

**`cpp/include/chocofarm/belief_diagram.hpp`** — declares the `BeliefDiagram` value type and the query free-functions. The header per ADR-0006 carries the path + purpose + Public Domain docstring. The seam exposes *values* (`int64_t` counts, `std::vector<uint32_t>` members, `std::vector<int64_t>` marginals), never node ids / pointers:

```cpp
namespace chocofarm {
class BeliefDiagram {                            // one-owner collaborator wrapping the hand-rolled ZDD (P3)
  public:
    static BeliefDiagram from_worlds(std::span<const uint32_t> bw, int N);  // build (§4)
    int64_t              count() const;                          // nb (§5.1)
    int                  node_count() const;                     // |Z| (§5.3)
    std::vector<uint32_t> members() const;                       // enumerate (§5.2) — for faithful-rep
    std::vector<int64_t> all_marginals() const;                  // bit_cnt[0..N) (§7)
    int64_t              det_count(uint32_t mask_j) const;        // det_cnt for one face mask (§8)
  private:
    // nodes_, unique_, root_, N_ — the engine, fully encapsulated (no leak across the seam, P9)
};
}  // namespace chocofarm
```

`BeliefDiagram` owns its `nodes_`/`unique_` (one engine instance per diagram, so the seam is value-clean; a shared-table variant is a later perf option behind the same seam, §B.5 "swap the engine if it pays"). The `.cpp` definition (`cpp/src/belief_diagram.cpp`) is added to the `chocofarm_core` library `add_library` list (alongside `src/features.cpp`, `CMakeLists.txt:117`) so tools/tests link it. Header + impl mirror `feature_compute.hpp`/`features.cpp`'s "single home" discipline.

### 10.2 The standalone probe executable (mirrors `belief_sweep_oracle_check.cpp`)

**`cpp/src/belief_diagram_probe.cpp`** — a single self-contained tool, no redis/net, pure compute (like the oracle, `belief_sweep_oracle_check.cpp:16`). It runs **both stages** and emits one `RESULT:` line:

- **Protocol:** `belief-diagram-probe --instance <p> --faces <p> [--samples 64] [--seed 0]` (same arg-parsing `opt()` helper as the oracle/bench, `belief_sweep_oracle_check.cpp:34`, `belief_sweep_bench.cpp:31`).
- **Stage 1 section:** runs the edge cases (`§5.5`) + the realistic generator grid (`§3.3`) + the random-subset control (`§3.4`); for every belief runs `faithful()`; prints the per-depth `|Z|`-vs-`nb` table (realistic vs control) and the headline median ratio. Aborts loudly on any faithful-rep failure (ADR-0002).
- **Stage 2 section:** for every belief runs the bit-exact asserts (`§9`) — `bit_cnt`/`det_cnt` integer equality vs the naive reference, and `BeliefFeatures` byte-identity vs `chocofarm::belief_features`.
- **Output:** `RESULT: PASS belief-diagram (N=.. nD=.. |worlds|=..; <k> beliefs faithful + bit-exact; median |Z|/nb = ..)` or `RESULT: FAIL <field/belief>`, matching the oracle's stdout contract (`belief_sweep_oracle_check.cpp:130`) so the same pytest string-match works. Return 0 on PASS, 1 on FAIL (oracle convention).

### 10.3 The CMake target (mirrors `chocofarm-belief-sweep-oracle-check`, `CMakeLists.txt:203-209`)

Append to `cpp/CMakeLists.txt`:

```cmake
# The belief DECISION DIAGRAM (ZDD) on-ramp probe (NOT the runner): the B.4(a) staged measure-first gate.
# STAGE 1 — build Z from realistic beliefs (worlds() narrowed by random CONSISTENT observation sequences,
# NOT random subsets), assert faithful-rep (enumerate(Z)==bw, count(Z)==nb), and report |Z| vs nb (+ a
# random-subset control). STAGE 2 — answer bit_cnt (all-marginals sweep) + det_cnt (disjoint count) from Z
# and assert byte-equality vs chocofarm::belief_features (the §B.3 logic-invariant net). Pure compute, no
# redis/net (like the sweep oracle). Separate executable (ADR-0012 P3, one-owner). Public Domain.
add_executable(chocofarm-belief-diagram-probe src/belief_diagram_probe.cpp)
target_link_libraries(chocofarm-belief-diagram-probe PRIVATE chocofarm_core)
target_compile_options(chocofarm-belief-diagram-probe PRIVATE -Wall -Wextra)
```

(`belief_diagram.cpp` goes into the `chocofarm_core` source list at `CMakeLists.txt:117-130`, so the probe links it via `chocofarm_core` exactly as the oracle does.)

### 10.4 The pytest gate (mirrors `test_cpp_belief_sweep_oracle`, `test_cpp_runner.py:116-130`)

Append to `tests/test_cpp_runner.py`, with the same opt-in `_RUN_CPP` skip + `RESULT: PASS` match + `cwd=REPO`/`PYTHONPATH` convention:

```python
BELIEF_DIAGRAM_BIN = os.path.join(REPO, "cpp", "build", "chocofarm-belief-diagram-probe")

@pytest.mark.skipif(not (_RUN_CPP and os.path.exists(BELIEF_DIAGRAM_BIN)), reason=_CPP_SKIP)
def test_cpp_belief_diagram_onramp():
    """B.4(a) ZDD on-ramp (belief_features_and_decision_diagram_note.md Part B). STAGE 1: build a ZDD
    from REALISTIC beliefs (worlds() narrowed by random CONSISTENT observation sequences, the search's
    information sets — NOT random subsets), prove faithful representation (enumerate(Z)==bw, count(Z)==nb),
    and measure |Z| vs nb (the §B.4 (a)->(b) decision number; a random-subset control shows the win is
    structure, not nb). STAGE 2: answer bit_cnt (all-marginals sweep) + det_cnt (disjoint count) from Z and
    assert byte-equality vs chocofarm::belief_features (the §B.3 logic-invariant net). RESULT: PASS gates it."""
    out = subprocess.run([BELIEF_DIAGRAM_BIN, "--instance", DATA_INSTANCE, "--faces", DATA_FACES],
                         cwd=REPO, capture_output=True, text=True, timeout=120,
                         env={**os.environ, "PYTHONPATH": REPO})
    sys.stdout.write(out.stdout); sys.stderr.write(out.stderr)
    assert out.returncode == 0 and "RESULT: PASS" in out.stdout
```

The probe prints the `|Z|`-vs-`nb` table to stdout, which the gate echoes (`sys.stdout.write`), so the **measurement is captured in CI logs**, not just the pass/fail — per the user's storage preference (keep the expensive output), the table is the deliverable, the PASS is the safety net.

---

## 11. Correctness-trap checklist (consolidated)

1. **ZDD reduction rule** (`§2.3`): suppress `hi==BOT` (zero-suppression), **not** `lo==hi` (that is the BDD rule). All node creation funnels through `mk` so the rule cannot be bypassed. Netted by faithful-rep (a wrong rule changes the member set).
2. **Variable ordering** (`§1`): strict-increasing treasure-id order, top=var 0, honored identically in build, `union`, `all_marginals` topological order, and `avoid_bit`. A mismatch silently corrupts counts; netted by bit-exact vs the sweep.
3. **All-marginals sweep** (`§7`): forward `above` must be processed in topological (increasing-`var`) order so each node's path-count is final before it is pushed to children; the per-`t` sum is `above[n]·below[hi(n)]` over `var(n)==t`. Netted by `bit_cnt_zdd == bit_cnt_ref`.
4. **ZDD count** (`§5.1`): `count(lo)+count(hi)` with **no** `2^skip` factor — zero-suppression makes skipped variables absent, not free. Netted by `count(Z)==nb`.
5. **Hash-cons collision** (`§2.4`): use a lossless `(var,lo,hi)` key, never a lossy packed hash, or `|Z|` shrinks artificially (trust failure 2).
6. **Realistic vs random beliefs** (`§3`): the headline beliefs are `worlds()` filtered by *consistent* (true-world-derived) observation sequences; the random-subset arm is a *control*, never the measurement. Conflating them is the note's named trap that would falsely kill Part B.
7. **Empty belief** (`§5.5`): `build_from_worlds({})==BOT`; every query returns 0; equals `belief_features_empty`. Keep it off the hot recursion the way `features.cpp:217` dispatches.
8. **`det_cnt` mask domain** (`§8`): `mask_j` bits are treasure ids = ZDD variables = `face_masks()[j]` (`env.hpp:109`); no translation. Popcount-1 faces double as a check of the `det_cnt[j]==bit_cnt[b]` identity (Part A's shortcut).

---

## 12. ADR / documentation hygiene (§B.6, and CLAUDE.md "documentation is part of the work")

- **§B.6 mapped:** `BeliefDiagram` is the one-owner collaborator + typed value seam (P3/P9) — `belief_diagram.hpp` exposes only values. The counts are a logic invariant → bit-exact assert vs the sweep, sweep kept as oracle during bring-up (P6 strongest tier). Nothing here is a wire fact (P7 untouched) — it is a feature-time prototype beside the sweep (§B.4(a)), not a belief-surface replacement (that is §B.4(b), out of scope until the `|Z|` number says go).
- **ADR-0006:** all new files (`belief_diagram.hpp`, `belief_diagram.cpp`, `belief_diagram_probe.cpp`) carry the path + purpose + Public Domain module-docstring header, like `belief_sweep_oracle_check.cpp:1-16`.
- **Docs to touch when implementing (not now — flagged per "documentation is part of the work"):** `docs/STATUS.md` / the current handoff if they describe the belief-feature surface as "sweep only"; and a short design note recording the `|Z|`-vs-`nb` result is the natural home for the deciding number (the note's §B.4 measurement is what graduates `(a)→(b)`). The note itself (`belief_features_and_decision_diagram_note.md`) is a point-in-time record — do **not** retro-edit it; record the firing of its §B.4 measurement in a new dated note or STATUS amendment (ADR-0005 append-don't-rewrite).

---

**Files referenced (all absolute):** `/home/bork/belief_features_and_decision_diagram_note.md` (Part A, §B.0-B.6); `/home/bork/w/vdc/1/chocofarm/cpp/src/features.cpp` (the §A.4 sweep to mirror in Phase 2 + dispatch); `/home/bork/w/vdc/1/chocofarm/cpp/src/belief_sweep_oracle_check.cpp` (the harness/`RESULT:` pattern to mirror); `/home/bork/w/vdc/1/chocofarm/cpp/include/chocofarm/env.hpp` + `/home/bork/w/vdc/1/chocofarm/cpp/src/env.cpp` (`worlds()`, `face_masks()`, `observe`, `filter_detector`, `filter_treasure`, `informative`); `/home/bork/w/vdc/1/chocofarm/cpp/include/chocofarm/feature_compute.hpp` + `/home/bork/w/vdc/1/chocofarm/cpp/include/chocofarm/features.hpp` (`belief_features`, `BeliefFeatures`); `/home/bork/w/vdc/1/chocofarm/cpp/CMakeLists.txt` (target pattern, `chocofarm_core` source list); `/home/bork/w/vdc/1/chocofarm/tests/test_cpp_runner.py` (pytest gate pattern). **New files to create:** `/home/bork/w/vdc/1/chocofarm/cpp/include/chocofarm/belief_diagram.hpp`, `/home/bork/w/vdc/1/chocofarm/cpp/src/belief_diagram.cpp`, `/home/bork/w/vdc/1/chocofarm/cpp/src/belief_diagram_probe.cpp`.


---

## Adversarial critique

```json
{
  "issues": [
    {
      "design": "all",
      "severity": "major",
      "area": "queries",
      "problem": "All three designs assert that node-ids assigned by mk are in valid topological order (children created before parents, so a plain ascending-id loop is a correct bottom-up `count`/`down` pass and a descending-id loop is a correct top-down `up`/forward pass). This is TRUE for a single throwaway-arena build (singleton+union, ids 0,1 reserved for terminals, every child constructed before its parent), but it is silently FALSE the moment any node table is reused across beliefs OR a query (offset/avoid_bit) appends new nodes whose children are pre-existing OLD ids smaller than the new id. Design 1 explicitly says 'one BeliefZdd per belief, thrown away' (safe). Design 2 says `up` is computed by a DESCENDING id loop AND elsewhere proposes the seam may later share the table; Design 3's `avoid_bit` calls `mk(var, avoid(lo), avoid(hi))` which appends a NEW node whose `var` is SMALLER than its just-built parent only if recursion is bottom-up — but the new node's id is LARGER than its children's ids, so the id-monotonic invariant still holds for those. The real trap: `det_cnt` via offset/avoid_mask MUTATES the node table (new nodes appended) between the `count` array sizing and the forward/backward sweep arrays sized by `nodes_.size()`. If `all_marginals` caches `card`/`below` in a vector sized at call time and a later `avoid_bit` grows `nodes_`, a stale-sized array indexes out of bounds.",
      "fix": "Make the id-monotonic-topological-order invariant EXPLICIT and assert it (in mk: assert children ids < new id, which holds by construction since children are made first). Size every per-query scratch array (`card`, `up`, `down`, disjoint memo) at the START of that query from the CURRENT `nodes_.size()`, and forbid table mutation during a sweep — or, cleanest, give `all_marginals`/`disjoint_count` their own recursion+memo that does not call `mk` at all (count-only queries never need to build nodes). Design 3's `avoid_bit` is the only query that builds nodes; prefer a non-constructing disjoint-count recursion (Design 1's `disjoint_count` does exactly this: it never calls mk, just memoized count with the hi-branch dropped when var in mask) over Design 2/3's offset-chain that materializes subfamilies."
    },
    {
      "design": "1",
      "severity": "major",
      "area": "zdd-correctness",
      "problem": "Design 1's hash-cons key packs (var,lo,hi) LOSSILY into 64 bits: `(uint64(var)<<56) ^ (uint64(lo)<<28) ^ uint64(hi)`. XOR-folding of shifted fields is NOT injective: lo occupies bits 28..(28+27)=55, var occupies 56.., and hi occupies 0..27, but lo<<28 overlaps var<<56 only above bit 56 (lo can be up to 2^28 so lo<<28 reaches bit 55 — no overlap with var at 56). However XOR of two distinct (lo,hi) pairs can collide because there is no separator and the fields are combined by XOR not concatenation — e.g. differing bits in lo and hi can cancel. A collision silently merges two structurally-distinct nodes, shrinking |Z| ARTIFICIALLY — exactly the trust-failure that fakes compression and could FALSELY BLESS Part B. Design 3 explicitly flags this same trap and chooses a lossless tuple/struct key; Design 1 hand-waves it as 'trivially correct'.",
      "fix": "Adopt Design 3's lossless key: a `struct{int32 var; uint32 lo,hi;}` with a custom std::hash and operator==, or pack into a wider key with NON-overlapping bit-concatenation (var:8 | lo:28 | hi:28 = 64 bits, using OR with masked fields, not XOR). The faithful-rep check (enumerate==bw) would catch a collision that changes membership, but a collision that merges two nodes reachable on disjoint paths can leave membership intact while still under-counting |Z| — so the lossless key is mandatory, not optional."
    },
    {
      "design": "2",
      "severity": "major",
      "area": "zdd-correctness",
      "problem": "Design 2's `zunion` handles the `a==1 (TOP)` case via `include_empty(b)`, and `include_empty` walks/creates the all-lo path to TOP. But `zunion`'s memo is keyed on the unordered (a,b) pair AND the recursion descends `na.var < nb.var => mk(na.var, zunion(na.lo,b), na.hi)`. When one operand is a deeper-tail TOP meeting an internal node, the var-comparison arm treats TOP as var==N (sentinel) — Design 2 says 'terminals: TOP has var=+inf' for the union but the WRITTEN code special-cases `a==1`/`b==1` BEFORE the var arms, which is correct. The actual bug risk: Design 2 also says it uses a TOURNAMENT (balanced) fold of singletons rather than a left fold, claiming 'O(nb log nb)'. A tournament fold unions independently-built subfamilies, which is fine for correctness, but Design 2 ALSO proposes sharing the apply machinery between build (union) and the det_cnt query (offset) AND a possibly-shared node table — combined with the descending-id forward pass this re-opens the topological-order hazard above. The narrower correctness concern: `include_empty` is never actually needed for 5-of-20 worlds (∅ is never a member) yet is on the union hot path for EVERY union where one side bottoms out at TOP — and the sketch `include_empty(f)=if f==TOP then TOP` returns TOP without recursing into f's structure when f is itself TOP, but when f is internal it rebuilds the entire lo-spine, which for two 5-chains is correct but O(depth) per union.",
      "fix": "Keep build = singleton + union but use the SIMPLE left fold (Design 1/3), not the tournament — the build cost is irrelevant to the gate (all three correctly note |Z| not build-time is measured), and the tournament adds an unforced shared-table temptation. Verify the `zunion` var-sentinel for terminals is handled by explicit `a==1`/`b==1`/`a==0`/`b==0` guards BEFORE any `var()` dereference (Design 2's written code does this; Design 3's `union` relies on a `var(TOP)=N` sentinel in `var()` — ensure that sentinel exists and is not read off `nodes_[1].var` which is -1 in Design 3's struct). Keep det_cnt non-constructing (see the all-designs topo issue)."
    },
    {
      "design": "3",
      "severity": "major",
      "area": "zdd-correctness",
      "problem": "Design 3 sets terminal `var=-1` (ZNode reserves -1 for terminals) but its `union` recursion uses `var(TOP)`/`var(BOT)` as '+inf (treat as below all)' via a `var()` accessor. If `var()` naively returns `nodes_[id].var` it returns -1 for terminals, which is SMALLER than every real var (0..19) — the exact opposite of the intended +inf sentinel. The union arm `va < vb => b rides the 0-edge` would then fire incorrectly when one operand is a terminal, corrupting the union. Design 3's prose says 'terminals: TOP has var = +inf' but the struct field is -1; the accessor must special-case terminals, and the design does not show that accessor.",
      "fix": "Make `var(id)` return `N` (or INT_MAX) for id<2 (both terminals), never the stored -1, everywhere it is used (union, avoid_bit, the marginals topological bucketing). This is the same sentinel Design 2 names explicitly. Add an assert in the union that the explicit terminal guards (`a==BOT/b==BOT/a==b`) catch TOP before any var-comparison arm — Design 3's `union` only guards BOT and `a==b`, NOT a lone TOP meeting an internal node, so TOP-vs-internal falls through to the var arms and MUST rely on the +inf sentinel being correct."
    },
    {
      "design": "all",
      "severity": "minor",
      "area": "queries",
      "problem": "All three give the all-marginals identity `bit_cnt[t] = Σ over nodes u with var(u)==t of up[u]*below[hi(u)]` and all three correctly argue NO 2^skipped factor is needed because ZDD zero-suppression makes skipped vars unconditionally absent. This is correct. But the forward `up[]` accumulation in Design 2 (descending id) and Design 3 (bucket by var ascending) both rely on processing every PARENT before the node. Design 1 builds an explicit `reachable_sorted_by_var` order. The subtle correctness point none state crisply: a node u may be reached from the root by MULTIPLE distinct edges (lo of one parent, hi of another, or hi-edges from two different parents at different vars) — `up[u]` must SUM all incoming path-counts, and the combine step must run AFTER all incoming contributions land. The descending-id loop (Design 2) is correct ONLY because every parent has a strictly smaller var hence (by the build) a strictly smaller id; but a shared subnode reachable from two parents at the same var is impossible under canonicity, so the per-var bucket order (Design 3) is also safe. The risk is an implementer combining (reading up[u]*below[hi]) inside the same pass that is still accumulating up into children — Design 1's code does the combine in the SAME loop as the push-down, which is correct since up[u] is final when u is visited, but only if the order guarantee holds.",
      "fix": "State the invariant as a one-line assertion: process nodes in any order where every node precedes its lo/hi children (a reverse-topological / var-ascending / ascending-id order); `up[root]=1`; for each node in that order, (a) read its contribution `bit_cnt[var]+=up[u]*below[hi(u)]` THEN (b) push `up[u]` to children. Assert `up` of an already-combined node is never mutated again (debug: mark combined). Cross-check `Σ_t bit_cnt[t]==K*nb` (all three propose this — keep it; it is the cheap canary for the doubling/order bug)."
    },
    {
      "design": "2",
      "severity": "minor",
      "area": "measurement-realism",
      "problem": "Design 2's realistic generator picks the observation action UNIFORMLY at random over ALL detectors/treasures (`j = uniform(0,nD)`, coin for detector-vs-treasure), without checking the action is informative over the CURRENT bw. A non-informative detector (all worlds agree) filters nothing — it inflates the nominal 'depth' without narrowing the belief, so Design 2's depth axis is a NOISY proxy for belief size, weakening the |Z|-vs-nb-vs-depth reading. Design 3 explicitly restricts candidates to env.informative(j,bw) and uncertain treasures (mirroring env.legal_actions), which is the faithful search move-set. Design 1 also picks uniformly without the informative filter.",
      "fix": "Adopt Design 3's informative-only candidate set (env.informative(j,bw) for detectors; 0<bit_cnt_i<nb for treasures) so each step is a REAL search move and the belief actually narrows — this both matches how the search forms information sets (env.legal_actions, env.cpp:90) and makes the depth axis monotone. Cheap and strictly more faithful."
    },
    {
      "design": "1",
      "severity": "minor",
      "area": "measurement-realism",
      "problem": "Design 1 (and Design 2) lack the RANDOM-SUBSET CONTROL ARM that Design 3 includes. Without it, a reader of the |Z|-vs-nb table cannot distinguish 'small |Z| because the belief is structured' from 'small |Z| because nb is small' — the exact misread that makes the (a)->(b) decision number unconvincing. The note's whole premise (§B.3) is that structured beliefs compress while arbitrary families do not; demonstrating realistic ≪ random at matched nb is what proves the compression is structure, not an artifact.",
      "fix": "Carry Design 3's control arm: for each realistic belief of size nb, also build Z for a RANDOM subset of worlds() of the same nb and log its |Z| side by side. Expected/validating: realistic |Z| ≪ nb while random |Z| ≈ nb. If they coincide, that is the honest 'no structure -> shelve' finding (note §B.4) — known, not guessed. This matches the codebase's non-vacuous-control posture (the gumbel 1a/1b mutation controls)."
    },
    {
      "design": "2",
      "severity": "minor",
      "area": "harness",
      "problem": "Design 2 proposes reconstructing the sweep's integer bit_cnt by `llround(sweep.marg[t]*nb)` as one option, then (in §8.1) correctly pivots to recomputing the sweep's bit_cnt with a naive `(w>>t)&1` loop for integer-to-integer comparison. The llround round-trip is a real trap: marg[t]=bit_cnt[t]*inv is a float, and llround(marg[t]*nb) can differ from bit_cnt[t] by 1 for large nb due to double rounding (the value was rounded once into marg, rounded again multiplying by nb). Design 1 explicitly rejects the llround path for the same reason and uses the inline naive count. Design 3 uses the naive reference directly.",
      "fix": "Drop the llround option entirely (Design 2 already prefers the naive recompute in §8.1 — make that the ONLY path). Reuse belief_sweep_oracle_check.cpp's `reference()` naive integer count (bc[t] via (w>>t)&1, dc[j] via env.observe) as the independent integer oracle and compare ZDD integer counts to it directly; never round-trip through the float marg/p_pos. Then run the ZDD counts through the IDENTICAL Phase-2 *inv and byte-compare the resulting BeliefFeatures to chocofarm::belief_features via equal_features (all three correctly specify this final byte-identity step)."
    },
    {
      "design": "all",
      "severity": "minor",
      "area": "harness",
      "problem": "All three reuse the oracle's `equal_features`, which compares marg, p_pos, informative, marg_sum, sharpness, nonempty — but NONE of the three's belief_features_from_diagram explicitly recomputes `informative[j]` from det_cnt with the SAME unsigned-vs-signed comparison as features.cpp (`det_cnt[j] > 0 && det_cnt[j] < static_cast<int64_t>(nb)`). Design 2 notes the unsigned/signed subtlety but stores det_cnt as uint64; comparing `det_cnt < nb` unsigned vs the production's `(int64)det_cnt < (int64)nb` is numerically identical for these magnitudes (all < 15504) so it passes, but the design should pin the comparison form to avoid a future int-width drift silently changing informative. Also: equal_features does NOT compare the `available`/`unc`/`sum_unc` blocks (those are assembled in FeatureBuilder::build, not in BeliefFeatures), so the harness validates BeliefFeatures byte-identity but NOT the full feature vector; that is fine for the on-ramp (the diagram only replaces the belief sweep) but should be stated.",
      "fix": "In belief_features_from_diagram, compute informative with the EXACT features.cpp form (cast both sides to int64_t or keep both int64_t) so equal_features.informative passes structurally, not coincidentally. Store bit_cnt/det_cnt as int64_t (matching features.cpp's accumulators) rather than uint64_t to keep the comparison identical and the casts to double bit-identical. State explicitly that the harness pins BeliefFeatures (the belief sweep's output) byte-for-byte — which IS the unit the diagram replaces — and that the downstream available/unc/sum_unc assembly is unchanged and already covered by the existing parity harness."
    },
    {
      "design": "3",
      "severity": "minor",
      "area": "feasibility",
      "problem": "Design 3's det_cnt via `avoid_mask` chains `avoid_bit` over each set bit of mask_j, and `avoid_bit` CONSTRUCTS new nodes via mk (subset0/offset). For nD=44 detectors, multi-bit masks, run over many beliefs, this materializes many intermediate subfamilies and grows the node table per query — interacting with the topological-order hazard (issue 1) and adding allocation churn. Design 1's `disjoint_count` is strictly simpler and non-constructing: a single memoized recursion that, at a node whose var is in mask, drops the hi-branch and sums lo (no mk, no new nodes), giving the disjoint count directly in O(|Z|) per detector. Both are correct; Design 1's avoids the table mutation entirely.",
      "fix": "Use Design 1's non-constructing disjoint_count(z, mask): memoized count where at each node `r = disjoint_count(lo)` and `+= disjoint_count(hi)` only if `var not in mask`. det_cnt[j] = count(z) - disjoint_count(z, masks[j]). This is O(|Z|) per detector, allocates nothing, cannot perturb the node table, and the per-detector memo is keyed by node id with mask fixed. Drop the offset/avoid_bit subfamily-materialization."
    },
    {
      "design": "all",
      "severity": "minor",
      "area": "integration",
      "problem": "All three add belief_diagram/belief_zdd.cpp to the chocofarm_core library source list (CMakeLists.txt:114-130). That is heavier than needed for a Stage-1 probe and couples the core lib (which every runner/parity binary links) to the new TU — a compile-error in the ZDD engine would break the entire build including the runner. The existing belief_sweep_oracle_check is a SINGLE-TU standalone (its source is only in its own add_executable, the reference() naive count is local to the TU; it links chocofarm_core only for env/features). Putting the engine in core is justified ONLY by the future B.4(b) reuse, which is out of scope until the |Z| number says go.",
      "fix": "For the Stage-1/Stage-2 on-ramp, keep the ZDD engine in the PROBE's own translation unit (or its own small object linked only by the probe target), exactly as belief_sweep_oracle_check is self-contained, so a WIP engine cannot red the runner build. Promote belief_diagram.cpp into chocofarm_core only when B.4(b) graduates and the engine becomes a shared collaborator behind the seam (the note's B.6/§3 trigger). This honors scope discipline (don't pre-build the (b) plumbing) and keeps the core build green."
    },
    {
      "design": "all",
      "severity": "minor",
      "area": "measurement-realism",
      "problem": "All three include the FULL world-set (nb=15504) as an edge case / depth-0 sample and Design 2 calls it 'the canonical |Z|≪nb demonstrator (~O(N*K) nodes)'. The full 5-of-20 family is a SYMMETRIC family with a famously small ZDD, so it will report a spectacularly small |Z|/nb — but it is NOT a realistic search belief (the search never sits on the unfiltered world-set except at t=0 before any observation). Presenting depth-0 / full-set |Z|/nb alongside the realistic-depth numbers risks an over-optimistic headline if the aggregate mixes it in. It is correct as a CORRECTNESS edge case (stresses union/sharing) but misleading as a MEASUREMENT sample.",
      "fix": "Keep the full world-set as a CORRECTNESS edge case (faithful-rep + bit-exact must hold on it). EXCLUDE depth-0 / full-set from the headline |Z|-vs-nb MEASUREMENT aggregate, or report it in a clearly separate row labelled 't=0 unfiltered (not a decision point)'. The decision number is the median |Z|/nb at realistic search depths (Design 3's grid of D>=1 with the informative-move generator), not the symmetric-family outlier."
    }
  ],
  "best_elements": "Carry forward a SYNTHESIS, taking the soundest component from each design.\n\nZDD ENGINE (node rep, terminals, reduction, mk): take the core shared by all three — BOT=0 (empty family {}), TOP=1 (family {∅}), node=(var,lo,hi), the SINGLE reduction rule `mk(var,lo,hi){ if(hi==BOT) return lo; ...hash-cons... }` (zero-suppression only, NOT the BDD lo==hi rule), strict-increasing var order = treasure-id order. Take DESIGN 3's LOSSLESS hash-cons key (struct/tuple key with custom hash, never an XOR-folded packed key) — Design 1's lossy XOR packing is a genuine artificial-compression hazard. Add an explicit ordering assert in mk (children ids < new id; children var > var).\n\nBUILD: take the SIMPLE left fold of singleton-chains + memoized zunion (Design 1/3), NOT Design 2's tournament fold (build cost is not the measured quantity, and the tournament invites a shared-table that re-opens the topological-order hazard).\n\nSTAGE-1 PRIMITIVES: count (memoized lo+hi, no 2^skip), node_count (reachable internal), enumerate (acc along hi-edges) — identical across all three, all correct.\n\nFAITHFUL-REP CHECK: take DESIGN 3's airtight triple — set(enumerate(Z))==set(bw) AND members.size()==bw.size() (no dup members) AND count(Z)==bw distinct size — run as the Stage-1 GATE before any |Z| is logged, fail-loud (ADR-0002). This is strictly stronger than the count-only check and is what makes |Z| trustworthy.\n\nALL-MARGINALS: take the identity bit_cnt[t]=Σ_{var(u)==t} up[u]*below[hi(u)] with NO 2^skip factor (all three agree, all correct), processed in a var-ascending / reverse-topological order with up[root]=1, combine-after-all-incoming. Take DESIGN 3's explicit 'bucket nodes by var ascending' as the clearest correct order (avoids the id-monotonicity assumption Design 2's descending-id loop hides). Keep the Σ_t bit_cnt[t]==K*nb canary all three propose.\n\nDISJOINT-COUNT: take DESIGN 1's NON-CONSTRUCTING memoized recursion (drop hi-branch when var in mask, sum lo; det_cnt[j]=count(z)-disjoint_count(z,mask_j)) over Design 2/3's offset/avoid_bit subfamily-materialization — it is O(|Z|) per detector, allocates nothing, and cannot mutate the node table mid-sweep. Keep the popcount-1 identity det_cnt[j]==bit_cnt[b] as an extra net (all three propose it).\n\nREALISTIC-BELIEF GENERATOR: take DESIGN 3's generator — sample one true world w*, start from worlds(), apply observations whose outcomes are read from w* (so w* always survives, nb>=1, the belief is a genuine information set), AND restrict each step's candidate action to env.informative(j,bw) detectors / uncertain treasures (mirroring env.legal_actions) so every step is a real search move and the depth axis is monotone. Take DESIGN 3's RANDOM-SUBSET CONTROL ARM (realistic vs random at matched nb) — it is what makes the |Z|-vs-nb conclusion non-vacuous and pre-empts the 'small because nb small' misread.\n\nEXPERIMENT: depth grid with S samples/depth, seeded; report per-depth nb/|Z| distributions, median |Z|/nb, realistic-vs-control; EXCLUDE depth-0/full-set from the headline aggregate (it is a symmetric-family outlier, keep it only as a correctness edge case).\n\nBIT-EXACT HARNESS: mirror belief_sweep_oracle_check exactly — reuse its reference() naive integer count (bc via (w>>t)&1, dc via env.observe) as the independent integer oracle, compare ZDD integer counts to it directly (NEVER the llround round-trip Design 2 floats), then run ZDD counts through the IDENTICAL Phase-2 *inv and equal_features byte-compare vs chocofarm::belief_features. Store bit_cnt/det_cnt as int64_t (matching features.cpp accumulators) and compute informative with the exact `det_cnt>0 && det_cnt<(int64)nb` form.\n\nEDGE CASES: take the union of all three's tables — empty (Z=BOT, all-zero, ==belief_features_empty), single world (5-node chain, pins zero-suppression), full world-set (correctness stress, NOT a measurement sample), popcount-1 mask (det_cnt==bit_cnt identity), multi-bit mask (general disjoint-count). All three cover these; Design 3's mapping to the env's actual nb==0 dispatch is the clearest.\n\nFILE/TARGET LAYOUT: take the standalone-probe shape from all three (header + impl + probe main + CMake add_executable mirroring chocofarm-belief-sweep-oracle-check + the opt-in pytest gate keyed on RESULT: PASS). BUT keep the engine in the PROBE's own TU for the Stage-1/2 on-ramp (Design's all-put-it-in-core is premature) — promote to chocofarm_core only when B.4(b) graduates. Take DESIGN 2/3's typed BeliefDiagram value-seam (no node ids cross the boundary, P9) as the API shape, with Design 1's per-belief throwaway arena as the lifetime model (sidesteps cross-belief table-reuse hazards for the gate).\n\nADR HYGIENE: all three correctly cite B.6 (P3 one-owner, P9 typed value seam, P6 bit-exact net, P7 untouched), ADR-0006 headers, ADR-0002 fail-loud on faithful-rep failure, and ADR-0005 append-don't-rewrite for recording the |Z|/nb verdict (do NOT retro-edit the note). Carry all of this; Design 3's note-is-point-in-time framing is the crispest.\""
}

```
