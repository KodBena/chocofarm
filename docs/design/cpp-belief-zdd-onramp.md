<!-- docs/design/cpp-belief-zdd-onramp.md
     The LOCKED design for the Part B (ZDD) belief-diagram on-ramp (B.4a): the implementation contract.
     Produced by the zdd-onramp-design workflow (3 independent proposals -> adversarial critique -> synthesis),
     2026-06-17, on branch cpp-actor-online-reconfig. The phases (proposals + critique) are preserved in
     docs/notes/zdd-onramp-design-2026-06-17-phases.md. Public Domain (The Unlicense). -->

All env facts confirmed: N=20, K=5, nD=44 detectors, 21 popcount-1 covers and 23 multi-bit covers (so both edge cases exist as real detectors on the live instance — no synthesis needed). |worlds| = C(20,5) = 15504.

I have read every required document end to end and verified the environment facts directly. I now have everything needed to produce the locked design doc. Here it is.

---

# Belief Decision Diagram (ZDD) On-Ramp — LOCKED Design (B.4(a))

**Status:** implementation contract. Synthesizes the soundest components of Designs 1/2/3 and resolves every fatal/major/minor issue the adversarial critique raised. Staged measure-first per the driving note `/home/bork/belief_features_and_decision_diagram_note.md` §B.4. Citations are to that note's Part B sections (§B.1 mapping, §B.2 queries-as-counts, §B.3 validation, §B.4 on-ramp + the deciding measurement, §B.5 hand-roll, §B.6 ADR hygiene).

**Verified env facts (read directly from the live instance, not assumed):** N=20 treasures, K=5 (exactly 5 bits per world), nD=44 detectors, |worlds|=C(20,5)=15504. Cover-popcount histogram on `faces.json`: `{1: 21, 2: 18, 3: 4, 4: 1}` — so **both** the popcount-1 and the multi-bit detector edge cases are real detectors on the live instance (no synthesized mask needed). Max treasure id appearing in any cover is 18 (< N).

---

## 0. The two trust failures this design exists to prevent (from Design 3 §0, the framing)

Both are silent — they produce a clean `|Z|` number that is wrong, and either would corrupt the (a)→(b) decision.

1. **Measuring the wrong beliefs.** A *random subset* of `worlds()` has `|Z| ≈ nb` (no shared substructure). Measuring random subsets reports "no compression" and **falsely kills Part B** — when the real search beliefs (information sets = conjunctions of observation constraints, §B.3) are exactly the structured regime. The realistic generator (§6) is therefore the single load-bearing measurement component, and a **random-subset control arm** (§7) is carried to prove the win is structure, not small `nb`.

2. **Trusting `|Z|` from a structure that does not faithfully represent `bw`.** A buggy ZDD (wrong reduction rule, lossy hash key, wrong ordering) can have a small node count while enumerating a *different* family. Before any `|Z|` is logged, the **faithful-representation check** (§5.4) proves the diagram *is* the belief: `set(enumerate(Z)) == set(bw)` AND `members.size() == |bw|` AND `count(Z) == |distinct bw|`, fail-loud on any divergence (ADR-0002).

---

## 1. The mapping and variable ordering (§B.1)

A belief is a **family of subsets** over the N=20-element treasure universe; each world is a 5-of-20 subset (the sparse regime ZDDs compress, §B.1). We take the **ZDD** branch (zero-suppressed — "specialized for sparse sets of subsets," §B.1).

**Universe and variables.** Variable index == treasure id `t`. A world `w` (`uint32_t`) is the subset `{ t : (w>>t)&1 }`. The ZDD represents a family of such subsets.

**Variable ordering (correctness trap, fixed once globally).** Canonical order `0 < 1 < … < 19`, top = variable 0, terminals at the bottom. Along any root→terminal path, `var` is **strictly increasing**. The order is a `constexpr` fact (no sifting/reordering heuristic — that would be a second writer of canonical form and is out of scope). The `lo` (0-)edge means "treasure `var` absent"; the `hi` (1-)edge means "treasure `var` present". Build, all-marginals, and disjoint-count all assume this order; a single off-order node breaks canonicity silently.

---

## 2. The ZDD engine: nodes, terminals, the reduction rule, hash-cons

### 2.1 Terminals (the central ZDD-vs-BDD distinction)

- **`BOT = 0`** = the empty family `{}` (no subsets). `count(BOT)=0`.
- **`TOP = 1`** = the family `{∅}` (exactly the one empty subset). `count(TOP)=1`.

These are NOT a BDD's 0/1; confusing them is the classic ZDD bug.

### 2.2 Node representation + the `var()` accessor sentinel (resolves Design-3 Major)

```cpp
struct ZNode { int32_t var; uint32_t lo; uint32_t hi; };
// nodes_[0]=BOT, nodes_[1]=TOP (their var field is unused; see var() below).
// internal nodes appended at ids >= 2. On any path, var strictly increases.
```

**Terminal `var()` MUST return a +∞ sentinel, never the stored field** (Design 3's fatal trap: a stored `-1` is *smaller* than every real var, the opposite of the intended sentinel, and would corrupt the union/disjoint var-comparison arms):

```cpp
// var(id) returns N (= a value strictly greater than every real var 0..N-1) for BOTH terminals,
// so a terminal is always "below" every variable in the comparison arms. NEVER read nodes_[id].var
// for id<2.
[[nodiscard]] int32_t var_of(uint32_t id) const { return id < 2 ? n_ : nodes_[id].var; }
```

### 2.3 The zero-suppression reduction rule + lossless hash-cons (`mk`)

Two reduction rules, both applied at **every** node creation; `mk` is the **only** function that creates internal nodes (mechanize > assert — canonicity is structural, unauthorable otherwise):

1. **Zero-suppression (the ZDD-specific rule):** if `hi == BOT`, the node is redundant — return `lo` directly. ("No member contains `var`," so `var` carries no information.) This is what produces `|Z| ≪ nb` on sparse families. It is the ONLY suppression — do **not** suppress on `lo==hi` (that is the BDD rule and corrupts the count).
2. **Merge (hash-consing):** identical `(var, lo, hi)` returns the existing id.

**Lossless key (resolves Design-1 Major — the XOR-folded packed key is a genuine artificial-compression hazard).** Use a struct key with a custom hash and `operator==`, never an XOR-folded packed integer:

```cpp
struct NodeKey {
    int32_t var; uint32_t lo, hi;
    bool operator==(const NodeKey&) const = default;
};
struct NodeKeyHash {
    size_t operator()(const NodeKey& k) const noexcept {
        // splitmix-style mix of three exact fields — injective inputs, no field cancellation.
        uint64_t h = (uint64_t)(uint32_t)k.var;
        h = h * 0x9E3779B97F4A7C15ull + k.lo;
        h = (h ^ (h >> 30)) * 0xBF58476D1CE4E5B9ull + k.hi;
        h ^= h >> 27;
        return (size_t)h;
    }
};

uint32_t mk(int32_t var, uint32_t lo, uint32_t hi) {
    if (hi == BOT) return lo;                          // (1) ZERO-SUPPRESSION — the ZDD rule
    // ordering invariant (debug assert): children must be strictly deeper than `var`.
    assert(var_of(lo) > var && var_of(hi) > var && "mk: variable-ordering violation");
    // id-monotonicity invariant (resolves the all-designs topo Major): children are created before
    // parents, so both child ids are < the id we are about to assign. This is what makes an
    // ascending-id loop a valid bottom-up order and a var-bucket order a valid top-down order.
    assert((lo < (uint32_t)nodes_.size()) && (hi < (uint32_t)nodes_.size()));
    NodeKey key{var, lo, hi};
    if (auto it = unique_.find(key); it != unique_.end()) return it->second;  // (2) MERGE
    uint32_t id = (uint32_t)nodes_.size();
    nodes_.push_back(ZNode{var, lo, hi});
    unique_.emplace(key, id);
    return id;
}
```

`unique_` is `std::unordered_map<NodeKey, uint32_t, NodeKeyHash>`. With a fixed order + both rules + lossless merge, the diagram is the **canonical reduced ordered ZDD**, so `|Z|` is a meaningful structural measure.

### 2.4 Lifetime model: one throwaway arena per belief (from Design 1; resolves the all-designs topo Major)

**One `BeliefDiagram` per belief, constructed then discarded.** No cross-belief node-table reuse. This is what makes the critique's "id-monotonic topological order" invariant true and safe: ids 0,1 are terminals, every child is created before its parent, so for *this* belief's table:
- an **ascending-id loop** is a valid bottom-up (children-before-parent) order for `count`/`below`;
- a **var-ascending bucket order** is a valid top-down (parents-before-children) order for the forward `up` sweep.

The disjoint-count and all-marginals queries are **count-only and non-constructing** (§8, §9) — they never call `mk`, so they cannot mutate `nodes_` mid-sweep. Scratch arrays (`card`, `up`, `below`) are sized once from `nodes_.size()` at query start. This closes the critique's "stale-sized array / table-mutation-during-sweep" hazard completely.

---

## 3. The typed `BeliefDiagram` value-seam (§B.5, §B.6 — P3/P9)

From Designs 2/3: a one-owner collaborator (P3) wrapping the hand-rolled engine behind a typed value seam (P9) — **no node ids cross the boundary**. Only the constructor and value-returning queries are public. This is the boundary at which a CUDD/Sylvan/SapporoBDD engine would later be swapped (§B.5) without touching callers.

```cpp
// cpp/probe/belief_zdd.hpp  (engine lives in the PROBE's own TU — see §11; NOT in chocofarm_core)
namespace chocofarm::beliefzdd {

class BeliefDiagram {
  public:
    // Build Z = the family of exactly the worlds in `bw`, over N variables. `bw` may be unsorted;
    // the family is the SET of distinct worlds. Empty bw -> Z = BOT.
    BeliefDiagram(std::span<const uint32_t> bw, int N);

    // ---- Stage 1: the decision-gate measurements ----
    [[nodiscard]] int64_t  count() const;                 // cardinality == nb (§5.1)
    [[nodiscard]] int      node_count() const;            // |Z| : reachable internal nodes (§5.3)
    [[nodiscard]] std::vector<uint32_t> members() const;  // enumerate Z's worlds (faithful-rep, §5.2)

    // ---- Stage 2: the B.2 counting queries (the §A.4 Phase-1 integer outputs) ----
    [[nodiscard]] std::vector<int64_t> all_marginals() const;            // bit_cnt[t] for all t (§9)
    [[nodiscard]] std::vector<int64_t> all_detector_counts(             // det_cnt[j] for all j (§8)
                      std::span<const uint32_t> masks) const;

  private:
    std::vector<ZNode> nodes_;
    std::unordered_map<NodeKey, uint32_t, NodeKeyHash> unique_;  // build-time; queries don't touch it
    uint32_t root_ = BOT;
    int n_ = 0;
    int32_t var_of(uint32_t id) const { return id < 2 ? n_ : nodes_[id].var; }
    uint32_t mk(int32_t var, uint32_t lo, uint32_t hi);
    uint32_t single(uint32_t w);
    uint32_t zunion(uint32_t a, uint32_t b);
};

}  // namespace chocofarm::beliefzdd
```

Counts are returned as **`int64_t`** (matching features.cpp's `int64_t` accumulators) so the comparison-and-cast path to `BeliefFeatures` is bit-identical (resolves the all-designs `informative`/int-width minor).

---

## 4. Build-from-worlds (Z = the family of `bw`)

**Simple left fold of singleton chains + memoized union** (Design 1/3; reject Design 2's tournament fold — build cost is not the measured quantity and the tournament invites a shared table that re-opens the topo hazard, per critique).

### 4.1 Singleton `single(w)` — one world as a chain

```cpp
uint32_t single(uint32_t w) {                       // family { set-bits-of-w }
    uint32_t cur = TOP;                             // {∅}: the deepest tail
    for (int t = n_ - 1; t >= 0; --t)               // descending: children (larger var) made first
        if ((w >> t) & 1u) cur = mk(t, BOT, cur);   // var t PRESENT: lo=BOT, hi=cur
    return cur;                                      // absent vars are zero-suppressed (never mk'd)
}
```

A 5-of-20 world yields a **5-node chain** — the 15 absent variables are elided by zero-suppression (the sparse-regime compaction, §B.1).

### 4.2 Union (`zunion`) — the only `apply` the build needs (§B.5 memoized apply)

Terminal guards come **before** any `var_of` dereference (resolves Design-3 Major: a lone TOP meeting an internal node must not fall through to a var-comparison arm relying solely on the sentinel; we guard it explicitly *and* keep the sentinel correct):

```cpp
uint32_t zunion(uint32_t a, uint32_t b) {
    if (a == BOT) return b;
    if (b == BOT) return a;
    if (a == b)   return a;
    if (a > b) std::swap(a, b);                      // commutative -> canonical memo key
    uint64_t key = ((uint64_t)a << 32) | b;
    if (auto it = umemo_.find(key); it != umemo_.end()) return it->second;
    uint32_t r;
    int32_t va = var_of(a), vb = var_of(b);          // TOP -> n_ (below all vars), via the sentinel
    if (va == vb) {                                   // both internal at the same var (TOP==TOP hit a==b)
        r = mk(va, zunion(nodes_[a].lo, nodes_[b].lo), zunion(nodes_[a].hi, nodes_[b].hi));
    } else if (va < vb) {                             // a internal & shallower: b rides a's lo-edge
        r = mk(va, zunion(nodes_[a].lo, b), nodes_[a].hi);   // (b has no var va -> var va absent in b)
    } else {                                          // vb < va, symmetric
        r = mk(vb, zunion(a, nodes_[b].lo), nodes_[b].hi);
    }
    umemo_.emplace(key, r);
    return r;
}
```

`umemo_` (`std::unordered_map<uint64_t,uint32_t>`) is build-time scratch. Note: for our worlds `∅` is never a member, so the lone-TOP-vs-internal case only arises at the shared tail and is handled by the `va==vb` arm at the chain bottom (both reach TOP, caught by `a==b`); the `va<vb` arm correctly puts the deeper operand on the shallower's lo-edge. The empty-belief edge case (`build({})==BOT`) is handled by the fold's initial `BOT` and the `a==BOT/b==BOT` guards.

### 4.3 The fold

```cpp
BeliefDiagram::BeliefDiagram(std::span<const uint32_t> bw, int N) : n_(N) {
    nodes_.push_back(ZNode{N, 0, 0});               // BOT  (var=N sentinel, unused)
    nodes_.push_back(ZNode{N, 0, 0});               // TOP
    uint32_t z = BOT;
    for (uint32_t w : bw) z = zunion(z, single(w));
    root_ = z;
    // umemo_/unique_ may be left as-is; the per-belief arena is discarded after the queries.
}
```

Build cost is `O(nb·K + union memo)` ≈ the note's `O(nb log)`; irrelevant to the gate (which measures `|Z|`, not build time).

---

## 5. Stage-1 primitives + the faithful-representation gate

### 5.1 `count()` — cardinality `nb` (§B.2 row 1)

Bottom-up via ascending-id loop (valid topo order, §2.4); **no `2^skip` factor** (zero-suppression makes skipped vars absent, contributing 1, not 2 — the ZDD-vs-BDD count trap):

```cpp
int64_t count() const {
    std::vector<int64_t> card(nodes_.size(), 0);
    card[BOT] = 0; card[TOP] = 1;
    for (uint32_t id = 2; id < nodes_.size(); ++id)
        card[id] = card[nodes_[id].lo] + card[nodes_[id].hi];
    return card[root_];                              // nb <= 15504, fits int64 with vast headroom
}
```

### 5.2 `members()` — enumerate (the faithful-rep witness)

```cpp
std::vector<uint32_t> members() const {
    std::vector<uint32_t> out;
    auto rec = [&](auto&& self, uint32_t id, uint32_t acc) -> void {
        if (id == BOT) return;
        if (id == TOP) { out.push_back(acc); return; }
        self(self, nodes_[id].lo, acc);                                  // var absent
        self(self, nodes_[id].hi, acc | (uint32_t{1} << nodes_[id].var)); // var present
    };
    rec(rec, root_, 0);
    return out;                                       // canonical -> distinct, size == count()
}
```

### 5.3 `node_count()` — `|Z|`

Reachable **internal** nodes from `root_` (terminals excluded so it is apples-to-apples with `nb`):

```cpp
int node_count() const {
    std::unordered_set<uint32_t> seen;
    std::vector<uint32_t> stk{root_};
    while (!stk.empty()) {
        uint32_t u = stk.back(); stk.pop_back();
        if (u < 2 || !seen.insert(u).second) continue;
        stk.push_back(nodes_[u].lo); stk.push_back(nodes_[u].hi);
    }
    return (int)seen.size();
}
```

### 5.4 The FAITHFUL-REPRESENTATION check (Design 3's airtight triple — the Stage-1 gate)

For each belief, build `Z`, then assert ALL of (fail-loud, ADR-0002):

```
m = members(Z) as a multiset
assert  set(m) == set(bw)               # (i) member-set equality — the strongest witness
assert  m.size() == distinct_count(bw)  # (ii) no duplicate members (ZDD members are distinct)
assert  count(Z) == distinct_count(bw)  # (iii) cardinality equals nb (the §B.2 count query)
```

`bw` from the env filters is already duplicate-free; the probe **asserts `bw` is duplicate-free** on every belief so the `nb := count(Z) == bw.size()` identity holds (else fail-loud, never paper over — ADR-0002). Only after the triple passes is `|Z| = node_count(Z)` recorded. `|Z|` from a non-faithful diagram is **never reported**.

---

## 6. The realistic-belief generator (Design 3 — the LEAD measurement component, §B.3/§B.4)

Realistic beliefs = `worlds()` narrowed by **random observation sequences with outcomes consistent with a sampled true world** — NOT random subsets. Two fidelity points beyond a naive generator:

- **Outcomes read from a sampled true world `w*`** so the belief is always non-empty (`w* ∈ bw` at every step ⇒ `nb ≥ 1`) and is a genuine information set the search could occupy.
- **Informative-only candidate actions** (resolves Design-2 minor): each step picks only from detectors with `env.informative(j, bw)` and treasures with `0 < (count of bit i over bw) < nb` — mirroring `env.legal_actions`, so every step is a *real* search move and the depth axis is monotone (a non-informative observation filters nothing and would inflate depth without narrowing).

```
generate_realistic_belief(env, depth D, rng) -> (bw, w*):
    all = env.worlds()
    w* = all[ uniform(0, |all|) ]                          # the consistency anchor
    bw = all (copy)                                        # the search's t=0 state
    for step in 1..D:
        cands = []
        for j in 0..nD-1: if env.informative(j, bw): cands += ("d", j)
        for i in 0..N-1:
            ci = #{ w in bw : (w>>i)&1 }
            if 0 < ci < |bw|: cands += ("t", i)
        if cands empty: break                              # belief fully determined; stop (record actual depth)
        a = cands[ uniform(0, |cands|) ]
        if a == ("d", j): env.filter_detector(bw, j, env.observe(j, w*))
        else:             env.filter_treasure(bw, i, ((w*>>i)&1) != 0)
    sort(bw)                                               # canonical order for set-compares
    assert w* in bw                                        # consistency invariant (fail-loud)
    return (bw, w*)
```

Reuses the env's own `informative` / `filter_detector` / `filter_treasure` / `observe` (one home, P1) — the beliefs are byte-identical to search information sets. Detector observations are the headline (disjunctive constraints, the §B.3 structured regime); a mixed detector+treasure sweep is also run.

---

## 7. The |Z|-vs-nb experiment + the control arm (§B.4)

**Grid.** Depths `D ∈ {1, 2, 3, 5, 8, 12, 20}` (D=0 / full-set is excluded from the headline aggregate — see below), `S` samples per depth (e.g. 64), seeded deterministically (reproducible, ADR-0009). Per sample: generate, build, run §5.4 faithful-rep, then record `(D, seed, nb, |Z|, |Z|/nb)`.

**Control arm (Design 3 — non-vacuous, resolves Design-1 minor).** For each realistic belief of size `nb`, also build `Z` for a **random subset** of `worlds()` of the same size `nb`, and record its `|Z|`. Expected/validating: realistic `|Z| ≪ nb` while random `|Z| ≈ nb`. Reporting them side by side proves the compression is *structure*, not small `nb` — and pre-empts the exact misread that would make the number unconvincing. If they coincide, that is the honest "no structure → shelve" finding (§B.4), known rather than guessed. (Mirrors the codebase's gumbel 1a/1b mutation-control posture.)

**Full-world-set handling (resolves the all-designs minor).** The full 5-of-20 family is a *symmetric* family with a famously tiny ZDD, so its `|Z|/nb` is spectacularly small but it is NOT a realistic search belief (the search sits on the unfiltered world-set only at t=0). It is kept as a **correctness edge case** (§10) but **excluded from the headline measurement aggregate**, or reported in a clearly separate row labeled `t=0 unfiltered (not a decision point)`.

**Report (stdout, parseable).** Per depth: `nb` distribution (min/median/max), `|Z|` distribution, **median `|Z|/nb`** (realistic vs control), and the fraction with `|Z| < nb/2`. The median `|Z|/nb` at realistic depths **is** the (a)→(b) decision number. The probe **does not decide** — it produces the number; the human reads it (§B.4: "let that number decide"). The table is echoed by the pytest gate into CI logs (the measurement is the deliverable; the PASS is the safety net).

---

## 8. Stage 2 — `all_detector_counts(masks)`: the non-constructing disjoint-count (§B.2 row 3)

`det_cnt[j] = nb − #{worlds disjoint from mask_j}`, where "disjoint" = sets none of the bits in `mask_j`.

**Use Design 1's non-constructing memoized recursion** (resolves Design-3 feasibility minor + the all-designs topo Major): at a node whose `var` is in `mask`, drop the hi-branch (taking that var would intersect the mask); sum the lo-branch always. It calls **no `mk`**, allocates nothing, and cannot mutate the node table:

```cpp
// disjoint_count(z, mask): members of the subfamily at z that set NONE of mask's bits.
// memo keyed by node id (mask is fixed within a call). Per-call scratch sized at call start.
int64_t disjoint_count(uint32_t z, uint32_t mask, std::vector<int64_t>& memo) const {
    if (z == BOT) return 0;
    if (z == TOP) return 1;                          // ∅ is disjoint from everything
    if (memo[z] >= 0) return memo[z];
    int32_t v = nodes_[z].var;
    int64_t r = disjoint_count(nodes_[z].lo, mask, memo);   // omit var: always allowed
    if (((mask >> v) & 1u) == 0)                            // var NOT in mask -> may take it
        r += disjoint_count(nodes_[z].hi, mask, memo);
    return memo[z] = r;
}

std::vector<int64_t> all_detector_counts(std::span<const uint32_t> masks) const {
    const int64_t nb = count();
    std::vector<int64_t> det(masks.size(), 0);
    for (size_t j = 0; j < masks.size(); ++j) {
        std::vector<int64_t> memo(nodes_.size(), -1);       // sized from CURRENT nodes_, no mutation
        det[j] = nb - disjoint_count(root_, masks[j], memo);
    }
    return det;
}
```

Cost `O(|Z|)` per detector, `nb`-independent (the note's `Σ|mask_j|·|Z|` collapses to `nD·|Z|`). **Through-line (§B.2):** the popcount-1 mask `1<<b` gives `det_cnt = nb − #{without b} = #{with b} = bit_cnt[b]` — Part A's popcount-1 shortcut as the special case (asserted as an extra net, §10).

---

## 9. Stage 2 — `all_marginals()`: the forward × backward sweep (§B.2 row 2)

`bit_cnt[t] = #{worlds in Z that set bit t}`, computed for **all t in one O(|Z|+N) sweep**, not N independent queries.

- **`below[u] = count(u)`** (subtree cardinality, backward): bottom-up via ascending-id loop.
- **`up[u]`** (path-counts to `u`, forward): `up[root]=1`; each node pushes `up[u]` to both children. **No `2^skip` factor** — in a ZDD a variable skipped between parent and a deeper child is unconditionally absent (factor 1), NOT a free 0/1 choice (the BDD doubling rule — the single most likely bug; §B.2 trap).

A member sets bit `t` iff its path takes the hi-edge of the (unique, by canonicity) node with `var==t` it passes through. So:

```
bit_cnt[t] = Σ over nodes u with var(u)==t  of  up[u] * below[hi(u)]
```

**Order (resolves the all-designs minor — state the invariant crisply):** process nodes in any order where every node precedes both its children — a var-ascending / reverse-topological / ascending-id order. For each node, **(a) read its contribution then (b) push `up` to children**, so `up[u]` is final before it is read. Use the **ascending-id loop** (valid by §2.4's id-monotonicity, which we assert in `mk`):

```cpp
std::vector<int64_t> all_marginals() const {
    const size_t M = nodes_.size();
    std::vector<int64_t> below(M, 0), up(M, 0);
    below[TOP] = 1;
    for (uint32_t id = 2; id < M; ++id)              // backward: children-before-parent (ascending id)
        below[id] = below[nodes_[id].lo] + below[nodes_[id].hi];

    up[root_] = 1;
    std::vector<int64_t> bit(n_, 0);
    // forward + combine in one descending-id pass: up[id] is final when we reach id (every parent has
    // a strictly smaller id by §2.4), so we read its contribution, THEN push to children.
    for (uint32_t id = M; id-- > 2; ) {
        if (up[id] == 0) continue;
        const ZNode& nd = nodes_[id];
        bit[nd.var] += up[id] * below[nd.hi];        // (a) read contribution
        up[nd.lo] += up[id];                         // (b) push down
        up[nd.hi] += up[id];
    }
    return bit;
}
```

(Descending-id is the natural reverse-topological order for `up`; an explicit `auto contribution-then-push` keeps `up[id]` final when read because every parent has a smaller id, and the combine reads `up[id]` before mutating any child's `up`.)

**Cheap canary (all three propose; keep it):** `Σ_t bit_cnt[t] == K·nb == 5·nb` — every world has exactly K=5 present bits. Catches a doubling/order bug even on a sample that coincidentally agreed elsewhere.

---

## 10. Bit-exact validation + edge cases (mirrors `belief_sweep_oracle_check.cpp`)

### 10.1 Phase-2 features, byte-identical (§B.3)

A free helper produces `BeliefFeatures` from the diagram's integer counts via the **identical** Phase-2 `* inv` of `features.cpp` (one home for the byte-identity claim). Counts stored as `int64_t`, and `informative` computed with the **exact** features.cpp form `det_cnt[j] > 0 && det_cnt[j] < static_cast<int64_t>(nb)` (resolves the all-designs `informative`/int-width minor):

```cpp
BeliefFeatures belief_features_from_diagram(const BeliefDiagram& z,
                                            std::span<const uint32_t> masks,
                                            int N, int nD, double log_nworlds) {
    const int64_t nb = z.count();
    BeliefFeatures bf;
    bf.marg.assign(N, 0.0); bf.p_pos.assign(nD, 0.0); bf.informative.assign(nD, 0.0);
    if (nb == 0) return bf;                                  // == belief_features_empty
    const std::vector<int64_t> bit_cnt = z.all_marginals();
    const std::vector<int64_t> det_cnt = z.all_detector_counts(masks);
    const double inv = 1.0 / static_cast<double>(nb);
    for (int t = 0; t < N; ++t) { bf.marg[t] = static_cast<double>(bit_cnt[t]) * inv; bf.marg_sum += bf.marg[t]; }
    for (int j = 0; j < nD; ++j) {
        bf.p_pos[j]       = static_cast<double>(det_cnt[j]) * inv;
        bf.informative[j] = (det_cnt[j] > 0 && det_cnt[j] < static_cast<int64_t>(nb)) ? 1.0 : 0.0;
    }
    bf.sharpness = std::log(static_cast<double>(nb)) / log_nworlds;
    bf.nonempty  = 1.0;
    return bf;
}
```

### 10.2 The harness (the §B.3 logic-invariant net)

For each belief (the realistic-generator grid + the control arm + the explicit edge cases below), assert, integer-to-integer:

1. **Integer counts vs the independent naive reference** (reuse `belief_sweep_oracle_check.cpp`'s `reference()` shape: `bc[t]` via `(w>>t)&1`, `dc[j]` via `env.observe(j,w)`). Assert `all_marginals() == bc` and `all_detector_counts(masks) == dc`. **Never the `llround(marg[t]*nb)` round-trip** (resolves Design-2 minor — double rounding can differ by 1). The naive integer reference is the independent oracle.
2. **The §9 canary:** `Σ_t bit_cnt[t] == 5·nb`, and `count() == members().size()`.
3. **Feature byte-identity:** `belief_features_from_diagram(...)` byte-equal to `chocofarm::belief_features(bw, masks, N, nD, log_nworlds)` via the oracle's `equal_features` verbatim. The only float op is `* inv` over exact integers, so `==` is the exact bit test (same justification as the oracle).
4. **Faithful-rep (§5.4)** stays active in Stage 2.

**Scope statement (resolves the all-designs minor):** the harness pins **`BeliefFeatures` byte-for-byte** — which IS the unit the diagram replaces (the belief sweep). The downstream `available`/`unc`/`sum_unc` assembly in `FeatureBuilder::build` is unchanged and already covered by the existing parity harness.

### 10.3 Edge cases (union of all three; mapped to env's actual dispatch)

| Edge case | Belief | What it pins |
|---|---|---|
| **Empty belief** (nb=0) | `bw = {}` | `build → BOT`; `count==0`, `node_count==0`, `members==∅`, `all_marginals→all 0`, every `det_cnt==0`. `belief_features_from_diagram` == `belief_features_empty` (features.cpp nb==0 branch). Exercises BOT. |
| **Single world** (nb=1) | `bw = {all[0]}` | 5-node chain; `count==1`, `node_count==5` (pins zero-suppression — wrong rule gives a different shape); `bit_cnt[t]==1` on the 5 present bits else 0; `det_cnt[j]==observe(j,all[0])?1:0`. Exercises TOP + the chain. |
| **Full world-set** (nb=15504) | `bw = env.worlds()` | correctness stress: faithful-rep must enumerate all 15504; `Σ bit_cnt==5·15504`; `bit_cnt[t]==C(19,4)==3876` for every t. **Correctness edge case only — excluded from the §7 headline aggregate** (symmetric outlier). |
| **Popcount-1 detector mask** | a real face with `popcount(mask_j)==1` (21 exist on the live instance) | asserts `det_cnt[j] == bit_cnt[b]` (Part A's shortcut as the `\|mask\|=1` disjoint-count special case), in addition to the sweep equality. |
| **Multi-bit detector mask** | a real face with `popcount(mask_j)>1` (23 exist) | the general disjoint-count; asserts vs the sweep's `det_cnt[j]` and vs a brute `#{w∈bw:(w&mask)==0}` over `bw`. |

On mismatch: `RESULT: FAIL` naming the belief (depth/seed) + the field/index. On success: `RESULT: PASS`.

---

## 11. File / target / test layout

**Engine in the PROBE's own TU, NOT in `chocofarm_core`** (resolves the all-designs integration minor + honors scope discipline): a WIP ZDD engine must not be able to red the runner build, and `belief_sweep_oracle_check` is itself a self-contained standalone. Promote `belief_zdd` into `chocofarm_core` only when B.4(b) graduates and the engine becomes a shared collaborator behind the seam.

```
cpp/probe/belief_zdd.hpp        # the BeliefDiagram value-seam + the engine (header-only OR paired .cpp,
                                #   both compiled into the probe TU only)
cpp/probe/belief_zdd.cpp        # engine impl (mk/single/zunion/count/members/node_count/
                                #   all_marginals/all_detector_counts) — linked ONLY by the probe target
cpp/src/belief_zdd_probe.cpp    # the standalone probe main: Stage-1 gate + Stage-2 bit-exact harness;
                                #   mirrors belief_sweep_oracle_check.cpp (opt()/fail()/reference()/
                                #   equal_features, --instance/--faces, RESULT: PASS|FAIL)
```

(If a header-only engine is preferred, fold `belief_zdd.cpp` into `belief_zdd.hpp` and compile only `belief_zdd_probe.cpp` — either keeps the engine out of `chocofarm_core`. The probe still links `chocofarm_core` for `Environment` / `belief_features` / `BeliefFeatures` / `load_instance`, exactly as the oracle does.)

All three files carry the ADR-0006 module-docstring header (path + purpose + Public Domain), each citing the note's §B.x sections.

### CMake (append to `cpp/CMakeLists.txt`, mirroring the oracle block at lines 203–209)

```cmake
# The belief DECISION DIAGRAM (ZDD) on-ramp probe (NOT the runner): the §B.4(a) staged measure-first gate
# (belief_features_and_decision_diagram_note.md Part B). STAGE 1 — build a hand-rolled ZDD Z from REALISTIC
# beliefs (worlds() narrowed by random CONSISTENT observation sequences, NOT random subsets), assert
# faithful-rep (set(enumerate(Z))==set(bw), count(Z)==nb), and report |Z| vs nb (+ a random-subset control;
# excl. the t=0 full-set outlier from the headline) — the (a)->(b) decision number. STAGE 2 — answer
# all-marginals (forward x backward sweep) + per-detector NON-CONSTRUCTING disjoint-count and assert they
# EQUAL chocofarm::belief_features's integer counts bit-exact (§B.2/§B.3 logic invariant), then the
# identical Phase-2 *inv makes the feature vector byte-identical. Pure compute (no redis/net). The ZDD
# engine lives in THIS TU (not chocofarm_core) until B.4(b) graduates (scope discipline). Separate
# executable (ADR-0012 P3, one-owner). Public Domain.
add_executable(chocofarm-belief-zdd-probe probe/belief_zdd_probe.cpp probe/belief_zdd.cpp)
target_include_directories(chocofarm-belief-zdd-probe PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/probe)
target_link_libraries(chocofarm-belief-zdd-probe PRIVATE chocofarm_core)
target_compile_options(chocofarm-belief-zdd-probe PRIVATE -Wall -Wextra)
```

### pytest gate (append to `tests/test_cpp_runner.py`, mirroring `test_cpp_belief_sweep_oracle`)

```python
BELIEF_ZDD_BIN = os.path.join(REPO, "cpp", "build", "chocofarm-belief-zdd-probe")

@pytest.mark.skipif(not (_RUN_CPP and os.path.exists(BELIEF_ZDD_BIN)), reason=_CPP_SKIP)
def test_cpp_belief_zdd_probe():
    """The §B.4(a) ZDD on-ramp (belief_features_and_decision_diagram_note.md Part B). STAGE 1: build a
    hand-rolled ZDD from REALISTIC beliefs (worlds() narrowed by random CONSISTENT observation sequences
    — the search's information sets, NOT random subsets which have |Z|~nb), prove faithful representation
    (set(enumerate(Z))==set(bw), count(Z)==nb) so |Z| is trustworthy, and measure |Z| vs nb (the (a)->(b)
    decision number; a random-subset control shows the win is structure, not small nb). STAGE 2: answer
    bit_cnt (all-marginals sweep) + det_cnt (non-constructing disjoint count) off Z and assert they EQUAL
    chocofarm::belief_features's integer counts bit-exact (the §B.3 logic invariant), then the identical
    Phase-2 *inv makes the feature vector byte-identical. ADR-0011: net the diagram, don't trust it. Pure
    compute (no FeatureBuilder, no layout file, no redis); cwd=REPO/PYTHONPATH for parity with other cpp gates."""
    out = subprocess.run([BELIEF_ZDD_BIN, "--instance", DATA_INSTANCE, "--faces", DATA_FACES],
                         cwd=REPO, capture_output=True, text=True, timeout=120,
                         env={**os.environ, "PYTHONPATH": REPO})
    sys.stdout.write(out.stdout); sys.stderr.write(out.stderr)
    assert out.returncode == 0 and "RESULT: PASS" in out.stdout
```

Opt-in under `CHOCO_RUN_CPP=1` like every other cpp gate; `RESULT: PASS` contract; the `|Z|`-vs-`nb` table is echoed into CI logs (the measurement deliverable).

---

## 12. ADR hygiene + documentation (§B.6; "documentation is part of the work")

- **§B.6:** `BeliefDiagram` is a one-owner collaborator (P3) behind a typed value seam (P9 — no node ids escape). Counts are a logic invariant → bit-exact assert vs the sweep, which stays the oracle during bring-up (P6 strongest tier, ADR-0011: net the diagram, don't trust it). Nothing here is a wire fact (P7 untouched) — it is the feature-time prototype (a), not the belief-surface replacement (b).
- **ADR-0002 (fail loudly):** the faithful-rep triple and the `bw`-duplicate-free assert `abort`/`RESULT: FAIL` on any divergence — a misrepresenting diagram never reaches the `|Z|` report.
- **ADR-0006:** all three new files carry the path + purpose + Public Domain module-docstring header.
- **ADR-0005 (append-don't-rewrite):** the driving note is a point-in-time record — **do not retro-edit it**. Record the firing of its §B.4 measurement (the measured `|Z|/nb` table + the realistic-vs-control delta + the (a)→(b) verdict) in the commit log / a dated handoff entry where the live state lives (the CLAUDE.md "live queue belongs in the commit log, not immutable prose"). If the gate graduates Part B, that is the §B.4 "Revisit when…" trigger, recorded by dated amendment.

---

## 13. Open risks / correctness traps to watch in implementation

1. **The ZDD reduction rule is NOT the BDD rule.** Suppress on `hi==BOT` (zero-suppression); never on `lo==hi`. All node creation funnels through `mk` so the rule cannot be bypassed. Netted by faithful-rep (a wrong rule changes the member set).
2. **Terminal `var()` sentinel.** `var_of(id)` returns `n_` (+∞-ish) for both terminals, never the stored field. The union's var-comparison arms and the ordering assert depend on it. (Design 3's fatal trap — guard terminals explicitly *and* keep the sentinel correct.)
3. **No `2^skip` factor anywhere.** `count`, `below`, and `up` all add children directly — zero-suppression makes skipped vars absent, contributing 1, not 2. The `Σ_t bit_cnt[t] == 5·nb` canary is the cheap detector for a doubling/order bug.
4. **Lossless hash-cons key.** Struct key + mixing hash, never an XOR-folded packed integer (Design-1 hazard: a collision merges distinct nodes, faking compression and falsely blessing Part B). A collision on disjoint paths can leave membership intact while under-counting `|Z|`, so faithful-rep would NOT catch it — the lossless key is mandatory.
5. **One throwaway arena per belief; queries are non-constructing.** This is what makes the id-monotonic topo order valid and forbids table mutation mid-sweep. The disjoint-count is the non-constructing recursion (Design 1), not an `offset`/`avoid_bit` that materializes subfamilies. Scratch arrays sized once from `nodes_.size()` at query start.
6. **All-marginals order invariant.** Process each node before its children (ascending-id / var-ascending), and **read the contribution `up[u]*below[hi(u)]` before pushing `up` to children**, so `up[u]` is final when read. Multiple parents sum into `up[u]`; canonicity forbids two parents at the same var reaching the same child, so a single combine pass is correct.
7. **Realistic ≠ random; control arm is mandatory.** The headline beliefs are observation-narrowed (informative-only steps, consistent outcomes); the random subset is the control, never the measurement. Exclude the t=0 full-set symmetric outlier from the headline aggregate (correctness edge case only).
8. **`bw` duplicate-free precondition.** `count(Z)` = distinct count; faithful-rep compares against `set(bw)`; the probe asserts `bw` is duplicate-free so `nb := count(Z) == bw.size()` (fail-loud otherwise). True for env-filtered beliefs.
9. **Integer types + `informative` form.** Store `bit_cnt`/`det_cnt` as `int64_t` (matching features.cpp accumulators) and compute `informative` with the exact `det_cnt>0 && det_cnt<(int64_t)nb` form, so `equal_features` passes structurally, not coincidentally. Never the `llround` float round-trip for the integer comparison.
10. **Mask domain.** `mask_j` bits are treasure ids = ZDD variables = `env.face_masks()[j]`; no index translation. `observe(j,w) == ((w & face_masks()[j]) != 0)` is the env's documented identity.

---

**New files (absolute):** `/home/bork/w/vdc/1/chocofarm/cpp/probe/belief_zdd.hpp`, `/home/bork/w/vdc/1/chocofarm/cpp/probe/belief_zdd.cpp`, `/home/bork/w/vdc/1/chocofarm/cpp/src/belief_zdd_probe.cpp`.
**Edited files (absolute):** `/home/bork/w/vdc/1/chocofarm/cpp/CMakeLists.txt` (add the `chocofarm-belief-zdd-probe` target — engine in the probe TU, NOT in `chocofarm_core`), `/home/bork/w/vdc/1/chocofarm/tests/test_cpp_runner.py` (add `test_cpp_belief_zdd_probe` + the `BELIEF_ZDD_BIN` path).
**Mirrored/reused unchanged (absolute):** `/home/bork/w/vdc/1/chocofarm/cpp/src/belief_sweep_oracle_check.cpp` (harness pattern, `reference()`, `equal_features`), `/home/bork/w/vdc/1/chocofarm/cpp/src/features.cpp` (`belief_features` oracle + the Phase-2 spec), `/home/bork/w/vdc/1/chocofarm/cpp/src/env.cpp` + `/home/bork/w/vdc/1/chocofarm/cpp/include/chocofarm/env.hpp` (`worlds`/`face_masks`/`observe`/`informative`/`filter_detector`/`filter_treasure`), `/home/bork/w/vdc/1/chocofarm/cpp/include/chocofarm/feature_compute.hpp` + `/home/bork/w/vdc/1/chocofarm/cpp/include/chocofarm/features.hpp` (`belief_features` decl + `BeliefFeatures`).

---

## B.4(b) GRADUATION NOTE (2026-06-17 — amend-by-append, ADR-0005 Rule 8; the §B.4 "let that number decide" trigger fired)

The §B.4(a) probe (this document's contract) graduated to **§B.4(b) belief-as-diagram**: the belief is now MAINTAINED as a ZDD through the search behind the env seam, as an OPT-IN third belief arm. This is a point-in-time record of that firing; the contract above (the B.4(a) probe) is NOT retro-edited.

**What landed (on a worktree branch off `main`):**

1. **The engine PROMOTED out of the probe TU** into a belief module: `cpp/include/chocofarm/belief_zdd_engine.hpp` + `cpp/src/belief_zdd_engine.cpp` (the §11 "promote when B.4(b) graduates" trigger). The build apply (`mk`/`single`/`zunion`) + the §5/§8/§9 queries (`count`/`members`/`node_count`/`all_marginals`/`all_detector_counts`) are the probe's, UNCHANGED in math. The `BeliefDiagram` is now a COPYABLE value (the per-belief arena is member state) so it is a per-descent-step belief value, thread-safe by construction (no shared mutable table — each thread's beliefs are independent, like the bitset arm's inline array).

2. **NEW: ZDD RESTRICT ops** — the maintenance the probe lacked (the probe REBUILT Z from an explicit `bw`; B.4(b) FILTERS the diagram as a ZDD op). `restrict_var(t, present)` (filter_treasure twin: `with_var`/`without_var`) and `restrict_cover(mask, positive)` (filter_detector twin: `cover_hold` for the disjunction-holds subfamily / `cover_fail` for the disjoint subfamily the §8 disjoint-count already characterizes). Each is a memoized recursion funnelling node creation through `mk`. **Validated bit-exact:** `members(restrict(Z,…))` is SET-EQUAL the flat filter's kept world-set, asserted after each of a 44-detector + 20-treasure filter SEQUENCE over 12 beliefs (+ a restrict→BOT empty-transition net), in the flat-vs-ZDD A/B (`belief_sweep_oracle_check.cpp` Part 3, under `#ifdef CHOCO_BELIEF_ZDD`).

3. **The OPT-IN arm:** a `CHOCO_BELIEF_ZDD` CMake option, DEFAULT OFF. OFF the Belief variant is `variant<FlatBelief, BitsetBelief>` and the engine/ZDD-op TUs are not compiled — the default build is byte-for-byte the live flat+bitset (the flat-vs-bitset A/B is unchanged). ON it is `variant<Flat, Bitset, ZddBelief>`, the engine + `cpp/src/env_zdd.cpp` (the zdd:: seam bodies) compile into `chocofarm_core`, and the gate SELECTS the ZDD arm (`full_belief()` returns a `ZddBelief`) so the whole search runs on the maintained diagram — for the head-to-head profile vs the bitset (and the large-N hedge). The flag surface is minimized: ONE `#ifdef` for the variant alias + the `ZddBelief` value type (env.hpp), ONE per-op `else`-arm macro `CHOCO_ZDD_ELSE(...)` injecting the third visit arm (env.cpp/features.cpp), and the gate's `use_zdd_` field — the existing flat/bitset arm bodies are UNCHANGED (the bare `else` became `else if constexpr (BitsetBelief)`).

4. **The EQUIVALENCE ASYMMETRY (deliberate, design-§4):** the ZDD arm is BIT-EXACT on filters/counts/features (the restrict ops + `all_marginals`/`all_detector_counts` + the IDENTICAL Phase-2 `* inv` give byte-identical nb/marginals/informative/legal_actions/belief_features to flat — the §B.3 logic invariant). BUT `sample_world`/`world_at_rank`/`belief_key` RE-BASELINE: the ZDD's canonical member order ≠ `worlds()`-rank order, so the r-th ZDD member ≠ the flat `bw[r]` (an O(nb) rank map would defeat |Z|≪nb). The ZDD arm's SAMPLING is therefore a search-level behavioral re-baseline (the JAX-reorder bucket) — the scripted gumbel/ismcts-dump parity WILL diverge on the ZDD arm (a scripted world index resolves to a different world). This is EXPECTED, NOT a bug; the flat-vs-ZDD A/B does NOT assert sampling equal.

**New files (absolute):** `/home/bork/w/vdc/1/chocofarm/cpp/include/chocofarm/belief_zdd_engine.hpp`, `/home/bork/w/vdc/1/chocofarm/cpp/src/belief_zdd_engine.cpp`, `/home/bork/w/vdc/1/chocofarm/cpp/src/env_zdd.cpp`.
**Edited (absolute):** `cpp/include/chocofarm/env.hpp` (the `ZddBelief` value type + the gated 3-arm variant + the `CHOCO_ZDD_ELSE` macro + the `use_zdd_` gate + the zdd:: op decls), `cpp/src/env.cpp` (the gate selection + the per-op ZDD visit arms), `cpp/src/features.cpp` (the `belief_features` ZDD visit arm), `cpp/src/belief_sweep_oracle_check.cpp` (the flat-vs-ZDD FEATURE A/B, gated), `cpp/CMakeLists.txt` (the `CHOCO_BELIEF_ZDD` option + the conditional TUs + the PUBLIC compile-definition).

---

## REVISIT-WHEN: DYNAMIC SELECTION (2026-06-17 — amend-by-append, ADR-0005 Rule 8)

§12 records the graduation trigger as "Part B graduates → the belief-surface replacement (b)"; the B.4(b)
graduation note above records that firing. A *distinct*, nearer trigger is now recorded: the
**head-to-head ZDD-vs-bitset profile** (planned now that the opt-in ZDD arm has landed — see the B.4(b)
note above) feeds a follow-on design question — **dynamic, per-belief representation selection keyed on
the support `nb`** — written up in **`docs/design/cpp-belief-dynamic-rep-selection.md`**. The `|Z|`-vs-`nb`
table this probe already produces (§7, with its random-subset control) is exactly one of the three inputs
that decision needs. Two framing notes for that decision:

1. ZDD here is an **opt-in build flag** (an all-or-nothing alternative). If the dynamic measurement shows
   a real win, the three reps are reconceived as a **standing portfolio** the runtime draws from per
   belief — at which point ZddBelief becomes a *permanent* variant arm and the opt-in build-flag framing
   is **superseded, not violated**. Until that measurement fires, the opt-in flag stands as specified here.
2. The ZDD crossover is **not purely `nb`** — `|Z|` is a *structure* measure (the whole point of the §7
   control arm), and `|Z|` is not cheaply knowable without building the diagram. So `nb` is a *proxy*;
   the dynamic note's §5b must establish that `nb` (or depth) predicts the ZDD winner before any
   `nb`-keyed ZDD selector is built.
