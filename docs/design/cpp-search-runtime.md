<!-- docs/design/cpp-search-runtime.md -->

# The C++ search runtime: a unified work-stealing pool over the Gumbel-AZ leaf-eval task algebra

**Status:** Design record (forward-looking, contracts-first). No code is committed; this is the
artifact the maintainer reviews before implementation. It makes concrete the **work-stealing pool**
that `docs/design/cpp-batched-search.md` §3.2 settled on (a *unified* pool over heterogeneous task
types, **not** a split selection/backprop partition, **not** a backprop-only pool), and it answers
the ADR-0009 measure-first gate `cpp-batched-search.md` §6-Q5 names. It does **not** re-litigate the
settled Axis-A / serial-per-tree exactness regime (`cpp-batched-search.md` §2), and it does **not**
design the Gumbel-AZ search itself — that port is in progress separately; this spec designs the
runtime **around** it (§3). Read end to end before implementation.

## On reading ADR-0012 in spirit, not by the letter (the frame this whole spec is held to)

The originating consult (`docs/notes/consult/opus-consult-2026-06-16-zmq-net-client-blocking-req.md`)
found a blocking-REQ client defended with a **proportionality argument** — "build the simpler thing
now, the structurally-right thing only if a benchmark forces it." ADR-0012 names *that argument
shape* — scale / minimality / "one X" / "for now" / "unnecessary here" / YAGNI / proportionate — as
**itself the tell** the discipline exists to reject (P7, repeated verbatim in P8 and P9). The myopic
reading of ADR-0012 is "P7 is about wire codecs." The **spirit** is broader and is the load-bearing
constraint on this design:

1. **Separate the durable contract from the swappable mechanism, and enshrine none as "the one
   way"** (P7's transport/serialization separation, generalized). Here the **durable contract** is
   the *task algebra + the per-tree state word + the leaf rendezvous*; the **swappable mechanisms**
   are the *transport* (blocking REQ ↔ non-blocking DEALER) **and** the *scheduling discipline*
   (pinned-thread-per-tree ↔ work-stealing pool). Neither transport nor scheduling is "the one way";
   both sit behind the contract as injected impls (P2).
2. **The structurally-correct thing is not demoted by a "simpler / for-now" argument.** The unified
   work-stealing pool is the correct CPU structure — `cpp-batched-search.md` §3.2 *settled* this on
   utilization and single-invariant grounds, not as an optimization to be earned later. So this spec
   is **born around the pool**; the pinned-thread-per-tree case is its **degenerate baseline** (the
   pool with stealing disabled and a blocking rendezvous), not a co-equal "embodiment." The thing
   ADR-0009 measure-first *legitimately* gates is the **transport sophistication** (does the DEALER
   multiplexer beat the dumb blocking rendezvous on a 4-vCPU host, §6-Q5) — a sub-knob *under* the
   pool, not the pool's right to exist.
3. **One home per fact** (P1). A tree's progress has exactly one home — the per-tree **state word**;
   the pool *derives* readiness from it and stores no second copy. The wire stays one codec; the one
   amendment (an echoed correlation id, §5) is a P7-disciplined SSOT bump, not a second hand codec.
4. **Functional core, imperative shell** (P9). The search step is a pure typed value-function
   (`advance`/`resume`); the pool, the sockets, the waits, the stealing are the imperative shell.
5. **Born clean, not built ahead of its consumer.** The Gumbel-AZ search the pool drives is landing
   separately. This spec is the design; the *implementation* sequences **after** the search lands
   (§3, build order) — building the runtime against a search that does not yet exist would be the
   build-ahead-of-consumer shape the originating consult flagged, in the other direction.

Where a rule below cites a P-number, it is the **spirit** of that principle, applied to scheduling
and transport as readily as to a wire layout.

---

## 0. The question this runtime answers

The maintainer's idea (`cpp-batched-search.md` §0): a pool of workers, each advancing an independent
tree, that descends to a leaf, pushes the leaf over ZeroMQ, and fetches the next work unit up to a
max in-flight cap; the Python side batches the leaves and routes results back to a **work-stealing
pool** that applies them. The design note's one correction (§3.2), which the maintainer has accepted:
**do not split the threads into a selection population and a backprop population, and do not make the
pool backprop-only** — use **one unified work-stealing pool over a heterogeneous task set**, so a
worker runs whatever ready task is next (a descent, a backup, or a failure), and no core idles on an
empty backprop queue while selection work piles up. "Backprop workers do only that" is preserved as a
*per-task* property (a worker running a backprop task does only that backprop), not a *partition* of
threads.

So the runtime's job is to **run that unified work-stealing pool** while keeping the leaf rendezvous
(transport) and the scheduling discipline swappable behind a stable contract, so the §6-Q5
benchmark — *does the work-stealing pool with a multiplexed non-blocking rendezvous actually beat the
dumb pinned-blocking baseline on the 4-vCPU host?* — can be run with **one search, one scenario, and
only the mechanism under test differing**.

The regime is settled and not re-opened here (`cpp-batched-search.md` §2): independent trees, each
**strictly serial** (at most one outstanding leaf; re-select only after that leaf backprops — the
exactness mechanism, §1.2 there), batched cross-tree at the central evaluator (Axis A, exact). Within-
tree leaf batching / virtual loss (Axis C) is out of scope and is *unreachable by construction* here
(§4).

---

## 1. The task algebra and the per-tree state word — the durable contract (P1, P2)

The center of the design is **what work exists** and **where a tree is**, not how bytes travel. Both
have exactly one home.

### 1.1 The per-tree state word (the SSOT of a tree's progress)

Each tree owns one atomic state word, the §3.1 enumeration:

```
READY ──▶ SELECTING ──(descend to a leaf, build the feature row, park the cursor)──▶
AWAITING_LEAF ──(its leaf reply lands)──▶ BACKPROP ──(apply value, W/N backup)──▶
   READY (sims/phases remain)  |  DECIDED (emit survivor + improved-π)
FAILED ◀── (a typed leaf-RPC failure on this tree)
```

This word is the **single source of truth** for the tree's progress. The pool does not keep a second
"is this tree ready" set — readiness *is* `state == READY`. The whole serial-per-tree guarantee is
one assertion over this word (§3.1): **a tree is enqueued in the pool at most once at any instant,
and only when READY.** Per-tree in-flight is strictly 1 because a tree at AWAITING_LEAF physically
cannot issue a second leaf (§3 makes this structural in the tree's interface) — but this is the
in-flight==1 invariant *only*; the distinct single-writer-per-tree invariant *across workers* is the
§8.2 obligation, **not** provided by the cursor.

### 1.2 The task algebra (the unified, heterogeneous work set)

A unit of work the pool schedules is one of three, keyed on a tree and its state — the §3.2 union:

```cpp
// cpp/include/chocofarm/search_task.hpp  (forward-looking — not built)

// A heterogeneous work unit. The pool steals these; a worker runs ONE to its next yield point.
// `Tree*` is a borrowed handle to a pool-owned TreeSearch (the pool owns tree lifetime; the handle
// is non-owning and never null — a typed view, not an optional). The union is closed (a typed sum,
// P9: no sentinel, the variant carries which-kind in the type).
struct SelectTask   { Tree* tree; };                              // advance a READY tree to its next leaf/decision
struct BackpropTask { Tree* tree; NetPrediction prediction; };    // apply a returned leaf value, then continue
struct FailTask     { Tree* tree; Error error; };                 // route a typed leaf-RPC failure to its owner
using Task = std::variant<SelectTask, BackpropTask, FailTask>;
```

- **`SelectTask`** runs the descent: it calls the tree's `advance()` (§3), which descends to a leaf
  and yields either a leaf request or the finished decision.
- **`BackpropTask`** is the work the maintainer originally (correctly) wanted a pool for, now *unioned
  with selection*: it calls the tree's `resume(prediction)` (§3), applying the client-side
  float32 masked-softmax + de-standardized value, the `v_mix`/`improved_policy`/`W-N` backup, and
  then continuing the search to its next leaf or decision.
- **`FailTask`** is fail-loud as a first-class task (P5 / ADR-0002): a timed-out / server-down /
  malformed leaf RPC for a tree becomes a `FailTask` that aborts *that tree's* episode loudly, never a
  silent drop or a stale/zero-value substitution (§5).

The union is the point: a worker pulls the next ready task of **any** kind, so a core never parks on an
empty backprop queue while select work waits, and the threads are **not** partitioned (§3.2). The
invariant that protects exactness lives in the **state word**, not in which worker runs the task —
which is exactly why the unified pool needs no cross-population handoff that the split model would.

---

## 2. The runtime seam and the work-stealing pool (P2, P3)

The runtime is a `SearchRuntime`: it turns a batch of independent decisions into their results, owning
the pool, the tree lifetimes, and the leaf rendezvous. The benchmark harness and any future self-play
driver hold a `const SearchRuntime&` and never name a concrete impl (a new scheduling discipline or a
new transport is a new impl behind the seam — the P2 inversion of control the `NetEvaluator` port
already realizes).

```cpp
// cpp/include/chocofarm/search_runtime.hpp  (forward-looking — not built)

// One decision to make: one fully-independent tree (own RNG stream, own _Node graph) for one problem
// instance. The env is borrowed ONCE by run() (not held per task — a reference member would fight the
// std::vector<SearchTask> backing the span). Live per-decision scalars (λ, budget, seed) ride the
// task (P4), never baked into the runtime.
struct SearchTask {
    Loc loc;                           // observed agent location
    std::vector<uint32_t> bw;          // observed belief world-set (bitmasks over treasure ids)
    std::uint64_t collected = 0;       // collected-treasure BITMASK (a small fixed universe — not a
                                       //   per-node-allocating std::set; bw is already bitmask-encoded)
    double lam = 0.0;                  // the live Dinkelbach penalty (per-decision, not frozen)
    std::uint64_t seed = 0;            // the per-tree RNG seed (the _fold_seed discipline)
    GumbelConfig cfg{};                // m, n_sims, c_puct, c_visit, c_scale, c_outcome, max_depth, temperature
};

// One decision result, returned by value. For the §7 benchmark only `executed` and `leaf_requests`
// are load-bearing; `improved_pi` is the self-play trainer's target, carried so Decision is the
// production shape (a Decision omitting it would force the self-play consumer to re-derive the target
// outside the seam). The runtime takes NO position on the float32-prior/float64-Q math that produces
// it — that is the Gumbel search's (§3, §7).
struct Decision {
    Action executed{};
    std::vector<float> improved_pi;    // π′ over the full slot space (the trainer's policy target)
    int leaf_requests = 0;             // net forwards this decision issued (== the structural seq len)
};

// The runtime seam: own {the work-stealing pool + tree lifetimes + the leaf rendezvous} and turn a
// batch of independent SearchTasks into their Decisions, IN INPUT ORDER. A failure on SOME tree
// aborts the whole batch loudly (no partial vector with a silent hole — §5). Polymorphic (held by
// base reference — the zero-cost ACL); impls are `final`.
class SearchRuntime {
  public:
    virtual ~SearchRuntime() = default;
    // ORDERING NOTE: impls that complete trees out of order (the pool finishes a tree when its last
    // leaf lands, i.e. in non-deterministic order) buffer into a position-indexed slot vector and emit
    // only when all slots are filled; a mid-batch failure discards the partial buffer (§5).
    [[nodiscard]] virtual std::expected<std::vector<Decision>, Error>
    run(const Environment& env, std::span<const SearchTask> tasks) const = 0;
};
```

### 2.1 The canonical impl — `WorkStealingRuntime` (the spine)

```cpp
// cpp/include/chocofarm/work_stealing_runtime.hpp  (forward-looking — not built)

// N workers, per-worker task deques with work-stealing (a worker drains its own deque LIFO, steals
// from a victim's deque FIFO when empty). It owns: the pool of TreeSearch objects (one per task, §3);
// a per-tree state word (§1.1); and a LeafRendezvous (the injected transport sub-mechanism, §4). The
// scheduling discipline (this pool) and the transport are SEPARATE injected concerns (P3, P7-spirit):
// the same pool runs over a blocking rendezvous OR a DEALER rendezvous unchanged.
//
// The worker loop, in one breath: steal a Task; if SelectTask -> advance(tree); if BackpropTask ->
// resume(tree, prediction); if FailTask -> mark the tree FAILED and abort the batch. On a step
// yielding NeedsLeaf, hand the LeafRequest to the rendezvous (which will, when the reply lands,
// enqueue a BackpropTask for this tree); on Decided, record the Decision in the tree's output slot;
// then loop. A tree moves READY -> (SELECTING|BACKPROP) -> AWAITING_LEAF|DECIDED and is re-enqueued
// (as a SelectTask) ONLY when it returns to READY — at most once (§1.1). The single-writer-per-tree
// guarantee (§8) is that a tree is in exactly ONE of {a deque, the rendezvous' outstanding-set, a
// done slot}, never two.
class WorkStealingRuntime final : public SearchRuntime {
  public:
    [[nodiscard]] static std::expected<WorkStealingRuntime, Error>
    create(int n_workers, std::unique_ptr<LeafRendezvous> rendezvous, int max_inflight);
    [[nodiscard]] std::expected<std::vector<Decision>, Error>
    run(const Environment& env, std::span<const SearchTask> tasks) const override;
    // ... RAII members; move-only ...
};
```

The scheduling discipline is a knob of this one impl: **stealing enabled** is the pool; **stealing
disabled + one tree pinned per worker thread + a blocking rendezvous** is the `cpp-batched-search.md`
§3.4 "thread-per-inflight" case — the **degenerate baseline** the benchmark measures the pool
against, expressed as the same runtime with the mechanism dialed down, *not* a separate parallel
design. (A `SerialRuntime` — `n_workers == 1`, blocking, no stealing — is the further degenerate B==1
fidelity reference, §7.) This is the P7-spirit separation made concrete: one durable pool, the
*scheduling discipline* and the *transport* both swappable beneath it, neither enshrined.

---

## 3. The tree as the task executor — `TreeSearch::advance/resume` (P9 functional core)

A task runs by stepping one tree. The tree is a **value-returning resumable state machine**: it does
not call the net — a `SelectTask`/`BackpropTask` *runs* it and it *yields* a leaf request or a
decision. This is `cpp-batched-search.md` §3.1's state word lifted into a pure interface, and it is
the substrate that makes the task algebra (§1.2) first-class: `advance()` *is* the SELECT step,
`resume(pred)` *is* the BACKPROP step.

```cpp
// cpp/include/chocofarm/tree_search.hpp  (forward-looking — not built)

struct LeafRequest { std::uint64_t corr_id = 0; std::vector<float> features; };  // the row the rendezvous evaluates
struct NeedsLeaf   { LeafRequest request; };
struct Decided     { Decision decision; };
using Step = std::variant<NeedsLeaf, Decided>;   // a typed sum, returned by value (P9: no sentinel)

class TreeSearch {
  public:
    // Fallible only if the task is itself malformed (empty belief / zero-legal root) — a boundary
    // condition, hence expected (P9 rule 5); a throwing ctor cannot return a value, so create() factory.
    [[nodiscard]] static std::expected<TreeSearch, Error> create(const Environment& env, const SearchTask& task);

    // The SELECT step. Advance the search — issuing per-tree RNG draws and running any NO-LEAF sims
    // inline — until it reaches a leaf needing a forward (NeedsLeaf, cursor parked) or the decision is
    // done (Decided). The FIRST call does the root leaf-eval (the whole decision waits on the root
    // forward). LEGAL only when READY; advance() while AWAITING_LEAF is a driver bug (assert/abort).
    [[nodiscard]] Step advance();

    // The BACKPROP step. Resume from AWAITING_LEAF with the leaf's prediction:
    //   * apply the CLIENT-SIDE float32 masked-softmax to the RAW logits (the prior is computed
    //     in-search, float32, not on the wire); the value is already de-standardized by the wire;
    //   * run v_mix / improved_policy / W-N backup of THIS leaf;
    //   * CONTINUE — consuming ZERO OR MORE further per-tree RNG draws (the next sims' sample_world)
    //     and running intervening NO-LEAF sims inline (a drawn TERMINATE world short-circuits with
    //     -λ·exit_cost and NO net call — it consumes an RNG draw but yields no NeedsLeaf);
    //   * until the NEXT leaf (NeedsLeaf) or the decision (Decided).
    // So resume() LEGITIMATELY ADVANCES THE RNG STREAM and absorbs intervening no-leaf sims. There is
    // deliberately no Step arm for "drew, terminated, no leaf yet": those draws live inside the
    // resume()->next-NeedsLeaf transition, and the §7 layer-2 structural-determinism test is the proof
    // they happen in the SAME ORDER as the synchronous reference. Resuming a non-AWAITING tree is an
    // invariant violation (assert/abort), not an expected.
    [[nodiscard]] Step resume(const NetPrediction& prediction);

  private:
    explicit TreeSearch(/* moved-in state */) noexcept;
    // owns: the _Node graph, the std::mt19937_64 RNG, the state word, and the SINGLE reentry cursor
    // (the explicit value the parked descent re-enters at — which SH phase, which surviving candidate,
    // which sim, which c_outcome determinization, the node-path being backed up). At-most-one cursor,
    // non-empty IFF AWAITING_LEAF: per-tree in-flight==1 is the statement "the cursor is singular" —
    // there is structurally nowhere to park a second leaf (Axis C unreachable, §4). No net, no socket,
    // no thread.
};
```

**This is the P9 functional core:** `advance`/`resume` are total value-functions of typed inputs (the
tree's own validated state + an already-decoded `NetPrediction`) returning a typed `Step` by value;
no I/O, no throw. Every effect (the socket, the threads, the stealing, the waits) lives in the
`WorkStealingRuntime` imperative shell. It is also the chief safety dividend: per-tree in-flight==1 is
**structural** (the singular cursor), so no scheduling bug can issue a second leaf and slide into Axis
C — and the same interface is the parity-test seam (§7 layer 2: feed canned predictions to `resume`,
assert the `NeedsLeaf` sequence).

**The honest cost (not papered over).** Making the search resumable is a real **continuation
refactor** of the Gumbel descent: `_descend`'s recursion and `_sequential_halving`'s phase/candidate/
sim loops must become the explicit reentry cursor, and it is load-bearing that this not perturb the
three Danihelka invariants or the per-tree RNG draw order. **This refactor is this spec's commissioned
step, sequenced after the synchronous Gumbel port lands** (build order below) — not the port's job and
not done speculatively before the search exists.

**Build order (load-bearing, P9 / don't-build-ahead-of-consumer):**
(i) the Gumbel-AZ search lands as a *synchronous recursive* search and passes its own parity (the
three Danihelka invariants, the float32 masked-softmax + weak-promotion kernel seams);
(ii) the `TreeSearch` continuation refactor restructures it into `advance/resume`;
(iii) `SerialRuntime` (n=1, blocking) lands and the §7 layer-2 structural-determinism test passes
**against the un-refactored synchronous Gumbel reference** — the gating precondition of the whole
benchmark (§7.1), proving the refactor reordered nothing;
(iv) the `WorkStealingRuntime` + the blocking rendezvous land (the pinned baseline falls out as a
dialed-down config);
(v) the DEALER rendezvous lands;
(vi) the §6-Q5 benchmark runs.

---

## 4. The leaf rendezvous — the swappable transport sub-mechanism (P7-spirit: not "the one way")

A `SelectTask` yields a `NeedsLeaf`; the rendezvous is **how that leaf becomes a future
`BackpropTask`**. It is a port the pool injects — *the* place transport lives, and the *only* place it
lives. Two impls; the pool runs over either unchanged.

```cpp
// cpp/include/chocofarm/leaf_rendezvous.hpp  (forward-looking — not built)

// Turn a tree's NeedsLeaf into an eventual BackpropTask. The pool calls submit() when a SELECT yields
// a leaf; the pool's completion path calls drain() to harvest finished leaves and enqueue their
// BackpropTasks. The codec is the SHARED inference_wire (P7 — re-author nothing on the wire).
class LeafRendezvous {
  public:
    virtual ~LeafRendezvous() = default;
    // Dispatch one tree's leaf. The corr_id keys the eventual completion back to the tree (§5).
    // PRE-DISPATCH VALIDATION: reject an empty / non-finite feature row HERE (mirroring the blocking
    // client) so a malformed row fails loudly at submit, not via a downstream timeout.
    [[nodiscard]] virtual std::expected<void, Error>
    submit(std::uint64_t corr_id, std::span<const float> features) = 0;
    // Harvest currently-finished leaves (0..N) as (corr_id, prediction); the pool enqueues a
    // BackpropTask per completion. A transport failure / malformed reply for a known corr_id is a
    // FailTask for that tree; a failure with no corr_id aborts the batch (§5).
    [[nodiscard]] virtual std::expected<std::vector<Completion>, Error> drain() = 0;
};
struct Completion { std::uint64_t corr_id = 0; NetPrediction prediction; };
```

- **`BlockingRendezvous`** wraps the existing blocking `ZmqNetClient` (`cpp/src/zmq_net_client.cpp`):
  `submit` does a synchronous send→recv on a per-worker REQ socket and `drain` returns that one
  completion. A worker thus **holds its core during the network wait** — so this rendezvous only keeps
  the batch full when workers ≫ cores (the §3.4 "thread-per-inflight" economics), and it pairs with
  the *pinned, stealing-disabled* config. It needs N REQ sockets (the REQ client is not thread-safe —
  "give each worker its own", `zmq_net_client.hpp:25-26`), i.e. N peers on the one ROUTER — a stated,
  reportable resource axis of the result, not a hidden asymmetry.
- **`DealerRendezvous`** is the one genuinely new component: a non-blocking DEALER client (many
  outstanding sends on one socket) that `submit`s without waiting and `drain`s replies by polling. A
  worker **never blocks on the network** — it submits, the tree parks at AWAITING_LEAF *off the
  deque*, and the worker steals the next ready task; when the reply lands, `drain` produces a
  `BackpropTask`. **This is why the work-stealing pool wants the non-blocking rendezvous:** stealing
  only pays when a worker freed by a parked leaf can immediately do other trees' work. (`DealerRendezvous`
  reuses the `inference_wire` codec verbatim; it owns only the DEALER socket lifecycle and the submit/
  poll/route — RAII, move-only, `create()`-factory, `void*` handles in the header.)

### 4.1 Correlation-id routing — an echoed id (P1: one home, P5: fail loud)

The server is a single-threaded greedy-drain ROUTER (`zmq-inference-service.md` §3). ROUTER↔DEALER
preserves per-peer ordering of *sent* frames, but the greedy drain does **not** reply in submit order
*across* drains: a request that just missed batch K waits for K+1, so a DEALER that submitted A then B
can receive B before A. And the server's malformed-request path (`_reject`, `inference_server.py`)
**logs and drops with no reply frame** (verified) — so a positional "i-th reply is the i-th submit"
FIFO is one dropped request away from mis-routing *every* subsequent reply on that peer. Therefore the
`DealerRendezvous` carries an **explicit echoed `u64` corr_id** in the frame, matched at `drain()`.
This is a real wire amendment, made the P7 way — **one home**: a new field in the `wire_spec` SSOT,
*derived* on both the Python and C++ sides, covered by the `tests/test_wire_drift.py` net, with the
server echoing the request's id into its reply. The routing keys on the echoed id, so a reordered
drain or a dropped request cannot mis-route. (Open question §8.1: the echoed-id bump vs. constraining
the server to a barrier/deterministic drain — the latter freezes the wire but couples routing to a
server timing discipline, the more brittle coupling. This spec leans echoed-id; not silently resolved.)

The §3.4 source leaned on a positional FIFO ("no wire change") as the *first cut*. Adopting that
because it is *simpler / for now* would be the proportionality argument ADR-0012 rejects; the echoed id
is the structurally-correct router and the one-field bump is its honest, mechanized cost.

---

## 5. Failure routing — `FailTask`, typed, owned, loud (ADR-0002 / P5)

`cpp-batched-search.md` §3.6: a timed-out / server-down / malformed leaf RPC is a **typed Error routed
to the owning tree**, aborting that tree's episode loudly — never a silent drop, never a stale/zero
value (which would corrupt that tree's backup and the aggregate-equivalence bar invisibly). In the
task algebra this is first-class: a failed leaf becomes a `FailTask{tree, error}`, the worker marks the
tree FAILED and propagates the Error out of `run()`, discarding the partial position-indexed output
buffer (no partial vector escapes, §2). The two completion-path holes the corr_id router introduces are
both closed: (1) a `drain()` reply whose corr_id matches no parked tree is a typed Error (a desync) —
loud abort, never slid; (2) a parked tree whose corr_id never returns (the server's silent-drop path,
or a lost reply) is caught by a **per-corr_id submit-timestamp deadline** — the *sole* backstop for the
silent-drop path, sized ≥ the server's worst-case forward latency under full `max_batch` (256) load so
it does not false-fire behind a full drain. The §4 pre-dispatch validation keeps malformed rows off the
wire so this deadline is reserved for genuine server/transport faults.

---

## 6. The two in-flight caps

- **Cap (a) — per-tree in-flight == 1, structural.** The §1.2 exactness mechanism, enforced by the
  `TreeSearch` interface shape (the singular reentry cursor, §3): the only way to get the next leaf is
  to `resume()` the previous one, so a tree physically cannot have two leaves outstanding. This is the
  in-flight==1 invariant **only** — no scheduling bug can issue a *second leaf*, and Axis C is
  unreachable by construction. It does **not** by itself provide the *separate* single-writer-per-tree
  invariant (that two workers never mutate one tree's `_Node` graph concurrently): that is the §8.2
  obligation the cursor does not give for free — TSan-gated, not structural. The proven claim (one
  outstanding leaf, from the cursor) must not vouch for the unproven one (one writer, from §8.2).
- **Cap (b) — global in-flight (`max_inflight`), a live knob.** The number of trees with a leaf
  outstanding at once; it sizes the server's achieved batch `B`. Bounded above by the host wall (the
  ~1.9× ceiling on the 4-vCPU VM, CLAUDE.md) and by the server `max_batch` (256). At this scale (tiny
  MLP, 4 vCPUs) the realistic parked count is **far below `max_batch`**, so the regime is expected to
  be **demand-limited, not cap-limited** — itself a §6-Q5 finding (if cap (b) never approaches
  `max_batch`, the multiplexer's many-in-flight advantage is moot at this scale). Under the blocking
  rendezvous cap (b) is effectively the worker count N (each blocked worker holds one outstanding
  leaf); under the DEALER rendezvous cap (b) is decoupled from N (one worker can have many trees parked).

---

## 7. Benchmark and parity — work-stealing pool vs. the pinned baseline

### 7.1 What §6-Q5 actually compares, and the validity precondition

The gate is **work-stealing pool + `DealerRendezvous`** vs. the **pinned, stealing-disabled, blocking
baseline** — i.e. is the multiplexed non-blocking pool worth its complexity over M dumb blocking
workers on this host. Work-stealing is the thing under test, not a footnote.

**Gating precondition (the validity bar, stated first).** All configs drive the *same refactored*
`TreeSearch` (§3). If the continuation refactor perturbed the per-tree RNG order or the SH bookkeeping,
**every config is wrong identically** and a config-vs-config aggregate bar would pass while all of them
diverge from the real search. So the benchmark is **meaningless until the §7 layer-2
structural-determinism test passes against the un-refactored synchronous Gumbel reference** (build-step
iii), **and** `SerialRuntime` matches that reference in aggregate. The shared substrate must be proven
faithful to the real search before any timing is trusted.

**Held fixed across the timed runs** (so a throughput delta is attributable to scheduling+transport,
nothing else): the same `TreeSearch`; the same captured scenario — the tuple **`(states corpus +
frozen net params + seeds + cfg)`** (without pinned weights "same scenario" is not reproducible, since
the leaf values determine how many leaves each tree issues); the same `InferenceServer` with
`StaticParamsSource` (one net version — a version straddle would change the leaf-request count and void
the structural cross-check), warm (no cold-XLA on the first batch of one run only); the same
`--cores 0,1,2,3` pinning.

**The fairness metric is achieved server batch `B`, not matched cap count.** "Sweep both at matched
cap (b)" is *not* a fairness mechanism: the blocking baseline's cap (b) is OS threads (real scheduler
entities thrashing 4 cores at high N) and the pool's is parked continuations (near-free), so equal
numbers hold a quantity of *different meaning* equal. The honest comparable is **throughput at matched
achieved `B`** and **throughput vs. core-utilization vs. the 1.9× wall**, with the cap sweep demoted to
*tracing each config's frontier*. A throughput win with no `B` increase, or a win that is really one
side under-subscribing the host, must be visible and explained.

### 7.2 Metric set and parity layers

Metrics (the ADR-0009 vocabulary, extended): **throughput (decisions/s)**, warmed, ≥3 reps with
variance; **achieved `B`** (mean + distribution — the mechanism *and* the fairness anchor); **core
utilization vs. 1.9×**; **leaf-request count per decision** (`Decision::leaf_requests` — a structural
cross-check: identical across all configs for matched seeds, else scheduling changed the search = a
bug); **cap-(b) frontier** (throughput vs. `B` and vs. utilization).

Parity, the four `cpp-batched-search.md` §5 layers: **(1)** net-forward parity (inherited, ~e-7,
unchanged); **(2)** single-tree **structural-determinism** — feed one `TreeSearch` canned byte-identical
predictions via `resume`, assert the `NeedsLeaf` sequence (including the no-leaf-sim effects, §3) and the
final `Decision` are identical to the **un-refactored synchronous Gumbel reference** (this is Option B's
dividend *and* the §7.1 gating precondition); **(3)** the three Danihelka invariants
(`test_executed_action_is_sh_survivor`, `test_vmix_prior_weighted`,
`test_sequential_halving_spends_full_budget`) under every config; **(4)** aggregate behavioral
equivalence across configs — N≥300 decisions / ≥2 seeds, action-distribution + improved-π statistically
indistinguishable within Monte-Carlo CI (because batch-composition roundoff, §2.2/`zmq §4`, legitimately
flips near-ties, so per-decision cross-config comparison would red a *correct* impl), **plus** a
`SerialRuntime`-vs-original-`gumbel_search` aggregate check (substrate faithful to the *real* search,
not merely self-consistent across configs), **plus** a batch-composition stress test (vary cap (b) +
inject arrival jitter, assert aggregate stays in CI). The two per-tree-consumer obligations the runtime
*calls* but does not *own* — the in-search float32 masked-softmax and the float32-prior/float64-Q
weak-promotion seam — belong to the Gumbel port (§3); the runtime must merely invoke them faithfully in
`resume`.

---

## 8. Open questions, boundaries, non-goals

### 8.1 Correlation routing (§4.1)
Echoed `u64` corr_id (this spec's lean — robust, couples nothing to server timing) vs. a
barrier/deterministic-drain server that preserves a positional FIFO (wire frozen, routing coupled to a
server discipline — the more brittle coupling). Not silently resolved. **Decision criterion:** prefer
the echoed id unless a consumer needs the barrier drain's *exact aggregate reproducibility* (e.g. a
single-tree determinism harness) as the **default** — which the throughput regime explicitly trades
away (`cpp-batched-search.md` §2.2). No such default consumer exists today, so the echoed id wins
unless the maintainer names one.

### 8.2 The pool's load-bearing concurrency obligation — single-writer-per-tree (§6-Q3)
This is the one real correctness cost the work-stealing pool adds that the pinned scheme gets for free,
and it is **front-and-center, not a footnote**: a tree's `_Node` graph must be touched by exactly one
worker at a time. The invariant: **a tree is in exactly one of {a worker deque, the rendezvous'
outstanding-set, a done slot} at any instant, never two**, and it is enqueued (as a `SelectTask`) only
on the READY transition (§1.1). `cpp-batched-search.md` §6-Q3 flags that the Phase-1 map does not
*prove* a multi-thread pool is race-free per-tree; this needs an **implementation guarantee + a targeted
concurrency test** (a TSan run over the pool driving a recording stub), else `W/N` corruption is a
correctness bug, not accepted slack. The benchmark must not run until that test is green.

### 8.3 Net-version-per-decision (inherited, §3.7 — not resolved here)
A tree's ~48 leaves can straddle a version reload today. The harness pins one version
(`StaticParamsSource`) as a **benchmark control** — both to avoid numeric confound and to keep the
leaf-request count stable for the structural cross-check (§7). Whether production self-play pins one
frozen version per tree-decision is an open decision orthogonal to this seam.

### 8.4 Failure-batch granularity
Whole-batch-abort `expected<vector<Decision>>` now (loudest, simplest §5 routing, right at benchmark
scope). Per-task `vector<expected<Decision, Error>>` is a self-play-consumer concern that does not exist
yet (designing it now is building ahead of a consumer); promote when that consumer lands. Deferred, not
litigated.

### 8.5 Boundaries / non-goals
NOT the Gumbel search math (the in-progress separate port — §3; this spec wraps it, commissions the
continuation refactor after it lands, must not conflict with its mixed-precision work). NOT within-tree
leaf batching / virtual loss (Axis C — unreachable by construction, §3/§6). NOT a re-authoring of the
wire (the runtimes compose the `inference_wire` codec and `NetEvaluator` port; the one echoed-id field
is a P7-disciplined SSOT bump). NOT the server (it stays the single-threaded greedy-drain ROUTER — no
XLA in a worker thread; the §4.1 echoed-id and the §8.1 barrier alternative are the only server touches
contemplated, mutually exclusive open questions). NOT building the `DealerRendezvous` / the pool
*before* the Gumbel search lands and *before* the §6-Q5 benchmark justifies the multiplexer's transport
complexity — building the transport sophistication speculatively is the measure-first violation in the
opposite direction. **(Note the asymmetry the ADR-0012 spirit demands: the *work-stealing pool* is the
settled-correct structure (§3.2) and is born clean now; only the *DEALER transport* is the
measure-first-gated optimization. Do not conflate "defer the fancy transport" with "defer the pool" —
that conflation would be the proportionality argument demoting the structurally-right thing.)**

---

## 9. ADR-0012 conformance — by spirit

The two conformances that are *not* obvious — and that the prior transport-centric draft got wrong —
are **P7 generalized to scheduling** (separate the durable contract from *both* swappable mechanisms,
transport and scheduling) and the **measure-first asymmetry** (the pool is born-clean settled
structure; only the DEALER transport is the gated optimization). The rest follow the standard reading:

- **P1 (one home per fact).** A tree's progress has one home: the **state word** (§1.1); the pool
  derives readiness from it, storing no second "ready set." The wire stays one codec; the echoed-id is
  an SSOT field both sides derive (§4.1). Derived quantities (`leaf_requests`, the dims) are computed,
  not re-typed.
- **P2 / P3 (seam, one-owner collaborators).** Four separated owners: the `WorkStealingRuntime` owns
  scheduling; `TreeSearch` owns one tree's search; the `LeafRendezvous` owns the transport; the codec
  owns the wire. A new scheduling discipline (pinned/stealing) or transport (REQ/DEALER) is a new impl/
  config behind the seam — zero edits to the search or the task algebra. **Neither transport nor
  scheduling is enshrined as "the one way"** (the P7-spirit separation generalized to mechanisms).
- **P4 (live, not frozen).** λ, the budget, the seed ride the `SearchTask`; `max_inflight` is a live
  knob — none baked into the runtime.
- **P5 / ADR-0002 (fail loud, root cause).** `FailTask` is a first-class task type; a leaf failure
  aborts the owning tree (and the batch) loudly with the typed Error's diagnostic; no silent stale
  value. The pre-dispatch validation removes the root cause of the server's silent-drop stall rather
  than band-aiding the timeout.
- **P6 (substantiate, behavioral bar).** The §7 parity is the two-tier bar: bit-exact on the structural
  `NeedsLeaf` sequence (a logic invariant), aggregate-behavioral within MC CI on the float-sensitive
  numerics. The benchmark is gated on the substrate proving faithful to the *real* search.
- **P7 (the spirit, non-myopic — the load-bearing one).** (a) The wire keeps one authoritative codec;
  the echoed-id is derived on both sides + drift-net-covered. (b) The **durable contract** (task
  algebra + state-word SSOT + leaf rendezvous) is separated from the **swappable mechanisms** (transport
  *and* scheduling), neither enshrined — which is exactly what makes the §6-Q5 benchmark *coherent*. (c)
  The structurally-correct work-stealing pool is **not demoted** by a "simpler / for-now" argument; the
  proportionality shape ADR-0012 names as the tell is refused (the positional-FIFO-because-simpler and
  the blocking-because-easier temptations are both declined). The measure-first gate applies only to the
  *transport sophistication*, never to the pool's existence.
- **P9 (functional core, imperative shell).** `TreeSearch::advance/resume` are total typed value-
  functions returning a typed `Step` by value — no I/O, no throw; the `Task` is a typed sum (no
  sentinel); per-tree in-flight==1 is a structural invariant, not a remembered check; a driver-misuse is
  an assert/abort, a leaf failure is an `expected`. All effect (sockets, threads, stealing, waits) lives
  in the `WorkStealingRuntime` shell, RAII / move-only / `create()`-factory throughout.
- **ADR-0009 / ADR-0011 (measure-first, mechanize over the class).** The seam exists to run the §6-Q5
  gate against a captured reproducible scenario with the ADR-0009 metric vocabulary; the echoed-id is
  mechanized by the existing drift net (a net over the class, not an instance); the single-writer-per-
  tree invariant gets a concurrency test before the benchmark runs (§8.2); per-tree in-flight==1 is
  mechanized *structurally* (the interface makes the violation un-authorable) — the strongest feasible
  surface.

---

*Public Domain (The Unlicense).*
