<!-- docs/design/cpp-search-runtime.md -->

# The C++ SearchRuntime seam: one interface over two leaf-evaluation concurrency embodiments

**Status:** Design record (forward-looking, contracts-first). No code is committed; this is the
artifact the maintainer reviews before any implementation begins. It deepens and tests the
established direction of `docs/design/cpp-batched-search.md` §3.4 / §6 open-question 5 (the ADR-0009
measure-first benchmark gate) and the consult
`docs/notes/consult/opus-consult-2026-06-16-zmq-net-client-blocking-req.md` (the matched-pair
finding). It does **not** re-litigate the settled Axis-A / serial-per-tree exactness regime
(`cpp-batched-search.md` §2), and it does **not** design the Gumbel-AZ search itself — that port is
in progress elsewhere; this spec designs the runtime seam **around** it (§2). Read end to end before
implementation.

---

## 0. The question this seam exists to answer

`cpp-batched-search.md` §3.4 names two embodiments of the same exact regime (cross-tree fan-out,
each tree strictly serial, at most one outstanding leaf), and §6-Q5 makes the choice between them an
**explicit ADR-0009 measure-first gate**: *"on the 4-vCPU host with a tiny-MLP cheap forward, does
the DEALER+fiber multiplexer actually beat M dumb blocking-REQ workers? Benchmark before committing
to the multiplexer's complexity."* That benchmark cannot be run until both embodiments exist behind
**one interface**, fed **one search** and **one scenario**, so the only thing that differs between
the two timed runs is the scheduling/transport mechanism. This spec is that interface.

The two embodiments, restated faithfully (`cpp-batched-search.md` §3.4):

- **Embodiment 1 — thread-per-inflight.** N OS threads, each advancing one independent tree; at a
  leaf the thread **blocks** in its own blocking `ZMQ_REQ` client (`cpp/src/zmq_net_client.cpp`).
  Many-in-flight comes from many threads; the server's greedy-drain ROUTER batches whatever REQ
  requests are concurrently in flight. Acceptable only when threads ≫ cores (a blocked worker is a
  parked core).
- **Embodiment 2 — DEALER submit/poll multiplexer.** One (or few) threads, K parked tree
  continuations, **one DEALER socket** (many outstanding sends); at a leaf the tree **submits**
  `(corr_id, features)` and **yields**; a completion path routes the reply back to the owning tree.
  Many-in-flight comes from many parked trees on one thread.

The settled finding the consult established, and this spec adopts as its load-bearing premise: the
swappable seam is **not** the `NetEvaluator` leaf port. Both embodiments are **identical at the
leaf** — same `inference_wire` codec, same decode, same de-standardized `NetPrediction`. They differ
in **how a tree awaits a leaf** (OS-thread-block vs continuation-yield/completion-route), which lives
*above* the leaf port. A blocking `predict` **cannot** run inside Embodiment 2's continuation — it
freezes the whole multiplexer thread and stalls every other parked tree. So `{scheduler, transport}`
are a **matched pair**: the seam must own **both**, not just the leaf call. The `NetEvaluator` port
(`cpp/include/chocofarm/net_evaluator.hpp`: `predict(span<const float>) -> expected<NetPrediction,
Error>`) makes local↔remote-blocking swappable; it does **not** make blocking↔async swappable (the
async leaf call is *submit-and-yield*, a shape `predict(span)->expected<NetPrediction>` cannot
express). That is the §2 finding of the consult, and it is why the seam is a `SearchRuntime`, not a
second net impl.

---

## 1. The seam — `SearchRuntime`

The seam owns `{tree scheduling + leaf dispatch}` and exposes a single batch-of-decisions entry
point. The benchmark harness, and any future self-play driver, hold a `const SearchRuntime&` and
never name a concrete impl (a new scheduling embodiment is a new `SearchRuntime` subclass with zero
edits to the search or to its callers — the same zero-cost-ACL inversion-of-control the
`NetEvaluator` port already realizes).

```cpp
// cpp/include/chocofarm/search_runtime.hpp  (forward-looking — not built)

// One unit of work the runtime schedules: "make ONE Gumbel-AZ decision for this independent problem
// instance, from this observed state, under this live λ." Each SearchTask is a fully independent
// tree (its own RNG stream, its own _Node graph) — the cross-tree independence that makes the batch
// row-independent (Axis A, cpp-batched-search.md §2.1).
//
// The env is borrowed ONCE by run() (see SearchRuntime::run), NOT held per-task: all tasks of a
// batch share one Environment and one server (§7.1). This keeps SearchTask trivially copyable /
// vector-storable (a struct with a reference member is non-assignable, which fights the
// std::vector<SearchTask> backing the span — FIX from review). The live per-decision scalars (λ,
// budget, seed) ride the task, never baked into a runtime object.
struct SearchTask {
    Loc loc;                           // observed agent location
    std::vector<uint32_t> bw;          // observed belief world-set (bitmasks over treasure ids)
    std::uint64_t collected = 0;       // treasures already collected — a BITMASK over the fixed
                                       //   treasure universe (NOT a std::set: the universe is small
                                       //   and fixed, bw is already bitmask-encoded — a per-node
                                       //   allocating std::set is the reliquary P9 flags; FIX)
    double lam = 0.0;                  // the live Dinkelbach penalty (per-decision, not frozen)
    std::uint64_t seed = 0;            // the per-tree RNG seed (the _fold_seed discipline, §6)
    GumbelConfig cfg{};                // m, n_sims, c_puct, c_visit, c_scale, c_outcome, max_depth,
                                       //   temperature — the frozen INSTANCE budget for this decision
};

// One decision result, returned by value. For the BENCHMARK this spec exists to enable, only
// `executed` and `leaf_requests` are load-bearing (the executed action and the structural
// cross-check, §7.2). `improved_pi` is the SELF-PLAY trainer's policy target; it is carried here so
// the runtime's Decision is the production shape (not silently under-specified) and so the parity
// harness can compare it (§7.2 layer 2), but it is NOT a quantity the scheduling benchmark measures
// and the runtime seam takes NO position on the float32-prior/float64-Q weak-promotion math that
// produces it — that math is the Gumbel port's (§2, §7.3). (Review CUT-2: kept, with this scoping
// note, rather than dropped — a Decision that omitted the trainer target would force the self-play
// consumer to re-derive it outside the seam, re-introducing the search-math coupling at the call
// site.)
struct Decision {
    Action executed{};                 // the SH survivor (temperature==0) or the sampled action
    std::vector<float> improved_pi;    // π′ over the full slot space (the trainer's policy target)
    int leaf_requests = 0;             // net forwards this decision issued (== the structural seq len)
};

// The runtime seam: own {tree scheduling + leaf dispatch} and turn a batch of independent
// SearchTasks into their Decisions. The search the runtime drives is IDENTICAL across impls (§3);
// only the await-mechanism differs. Polymorphic — held by base reference at the call site (the
// zero-cost ACL), so it carries a virtual destructor; impls are `final`.
class SearchRuntime {
  public:
    virtual ~SearchRuntime() = default;

    // Drive `tasks` to completion against `env` and return one Decision per task, IN INPUT ORDER, or
    // a typed boundary failure. The result vector is positionally aligned with `tasks`. A failure on
    // SOME tree aborts the whole batch loudly (it does NOT return a partial vector with a silent
    // hole — §5). The env is borrowed for the call's duration and shared by every task. The input is
    // a bounds-carrying view; the result is returned by value (P9 rules 1, 2, 5).
    //
    // ORDERING NOTE (FIX — GAP-2): impls that complete trees out of order (FiberMux finishes a tree
    // when its last leaf arrives, i.e. in non-deterministic completion order) MUST buffer into a
    // position-indexed slot vector and emit only when all slots are filled; a mid-batch failure
    // discards the partial buffer (§5). "Positionally aligned" is a buffering obligation on the
    // driver, not a free property.
    [[nodiscard]] virtual std::expected<std::vector<Decision>, Error>
    run(const Environment& env, std::span<const SearchTask> tasks) const = 0;
};
```

Three impls satisfy it, all driving the same `TreeSearch` (§3): `SerialRuntime` (the fidelity
reference + B==1 floor), `ThreadPerTreeRuntime` (Embodiment 1), `FiberMuxRuntime` (Embodiment 2).
Construction-site choice; the search, the env, and the `Decision` contract do not change a line.

**Why batch-of-tasks, not one-task-at-a-time.** A `run(span<const SearchTask>)` signature is what
makes the fan-out *available* to the runtime: `ThreadPerTreeRuntime` spreads the tasks across N
threads, `FiberMuxRuntime` parks them as K continuations on one thread, `SerialRuntime` loops. A
one-task entry point would force the fan-out up into the caller and re-introduce the very scheduling
concern the seam exists to own. The batch is the unit of concurrency.

**Failure-batch granularity (whole-batch-abort; one-line rationale, not a fork).** `run` returns
`expected<vector<Decision>>` for the whole batch: one bad leaf aborts the batch. This is the loudest
shape (ADR-0002) and keeps the §5 routing identical across all three impls, and at the benchmark
scope this spec is *for*, a single batch is one timed scenario that a leaf failure invalidates
anyway — so per-task fallibility is a self-play-consumer concern that does not exist yet (the Gumbel
search is not merged; self-play on it is further still). Per ADR-0012 anti-pattern E / P5
(adopt-or-delete) and the consult's symmetric measure-first warning, designing a per-task
`vector<expected<Decision, Error>>` fork now is building ahead of a consumer. Deferred, not litigated
(§8.2). (Review CUT-1: accepted — collapsed from a §1 paragraph + open question to this one-liner.)

---

## 2. The search is the thing being wrapped — the Gumbel-port dependency

The runtime drives the C++ Gumbel-AZ search, which is **being ported separately and is not yet
merged**. `cpp-batched-search.md` §1 is explicit: only NMCS is in C++ today (a forward recursion
with in-stack memorize-and-replay, no per-node backprop), and NMCS uses the `WorldSource`
determinization seam, **not** the `NetEvaluator` leaf port. The Gumbel-AZ search — Gumbel-Top-k →
Sequential Halving → improved-π, the `_decide_root`/`_descend`/`_puct_select`/`_sequential_halving`
structure of `chocofarm/az/gumbel_search.py` — is what this runtime's `TreeSearch` (§3) *wraps and
restructures*.

**This sequences the work hard, and the spec resolves who owns each step (FIX — the proposer's text
attributed the continuation refactor to the port in one place and to a follow-on step in another;
this is the disambiguation):**

1. **The Gumbel port lands as a *synchronous recursive* search.** Its decision logic — the
   `m`-candidate SH loop, the PUCT descent, the `v_mix`/`improved_policy` weak-promotion seam, the
   three float32 hazards (`cpp-batched-search.md` §1.3) — is the maintainer's in-progress
   mixed-precision port. This spec **treats it as a black box** and **must not modify its decision
   math**.
2. **THIS spec's seam then commissions the continuation refactor** (§3, Option B) as a distinct,
   later step that restructures that *landed synchronous* search into advance/resume. The refactor
   is **this spec's downstream work, not the port's** — the port finishes its mixed-precision work
   elsewhere as a synchronous search, and the runtime seam wraps the result.
3. **Build order (load-bearing):**
   (i) the Gumbel port lands as a synchronous recursive search and passes its own parity (the three
   Danihelka invariants, the float32 masked-softmax and weak-promotion kernel seams — §7);
   (ii) the `TreeSearch` continuation refactor restructures it into advance/resume (§3, Option B);
   (iii) `SerialRuntime` lands and the §7.2-layer-2 structural-determinism test passes **against the
   un-refactored synchronous Gumbel reference** — this is the gating precondition of the whole
   benchmark (§7.1, FIX-1), proving the refactor did not perturb the search;
   (iv) `ThreadPerTreeRuntime` lands (it needs no new transport);
   (v) the DEALER client + `FiberMuxRuntime` land;
   (vi) the §6-Q5 benchmark runs.

The runtime seam designs the `SearchRuntime` / `SearchTask` / `Decision` / DEALER-client /
failure-routing / benchmark-harness contracts against the *shape* of a Gumbel decision (one root
leaf, then a serial chain of interior leaves, with intervening no-leaf sims, then a survivor +
improved-π), not against any specific line of the search.

---

## 3. Keeping the search shared across runtimes — Option A vs Option B

For the benchmark to compare **scheduling**, all three runtimes must drive the **same search**, so a
flip in throughput is attributable to the await-mechanism and not to two different searches. There
are two ways to make one search drivable by both a blocking driver and a yielding driver.

### 3.1 Option A — synchronous-looking search + stackful fibers

Keep the search synchronously coded (`leaf = net.predict(features)` inline in `_descend`), and run
each tree inside a **stackful fiber** (`boost.context` / `boost.fiber`). At the leaf, the call
blocks-or-yields depending on the runtime: Embodiment 1 lets it block the OS thread; Embodiment 2
swaps the fiber out and resumes it when the reply arrives.

- **For:** smallest change to a synchronously-ported search — the search reads as straight-line
  recursion; the yield is invisible at the call site.
- **Against:** (a) adds a fiber dependency (`boost.context`) the project does not have, against a
  4-vCPU scratch host where every dependency is a real cost; (b) the yield is a **hidden
  control-flow effect** — a P9 concern: a leaf call that *looks* like a total function but actually
  suspends the stack and lets arbitrary other trees run is exactly the kind of invisible effect P9
  rule 3 ("the signature declares every effect") exists to forbid; (c) stackful fibers carry their
  own correctness surface (stack sizing, the interaction with thread-local state in the matmul/codec
  path) that the benchmark would have to control for — it muddies the very comparison it exists to
  enable.

### 3.2 Option B — the tree as an explicit resumable state machine (the recommendation)

Make the tree a **value-returning resumable state machine**: it does not *call* the net — it
*returns* a request for a leaf and is *resumed* with the answer. This is `cpp-batched-search.md`
§3.1's per-tree state word (`READY → SELECTING → AWAITING_LEAF → BACKPROP → …`) lifted into a
value-returning interface.

```cpp
// cpp/include/chocofarm/tree_search.hpp  (forward-looking — not built)

// One leaf request the tree has parked on: a correlation id (assigned by the driver, §4) and the
// feature row the net must forward. The features are the FeatureBuilder float32 vector — the SAME
// row both embodiments put on the wire (the leaf is identical across runtimes, §0).
struct LeafRequest {
    std::uint64_t corr_id = 0;         // the routing key (driver-assigned; §4)
    std::vector<float> features;       // length in_dim — the row predict()/the DEALER submits
};

// The reentry cursor: the EXPLICIT state a parked descent re-enters at (FIX — GAP/FIX-4). The
// synchronous search's implicit C++ call stack at the leaf becomes this owned value: which SH phase,
// which surviving candidate, which sim within that candidate, which c_outcome determinization, and
// the node-path being backed up. It is opaque at the contract level (its fields are the search's
// business), but its EXISTENCE and the invariant it carries are part of the contract: AT MOST ONE
// reentry cursor exists, and it is non-empty IFF the tree is AWAITING_LEAF. Per-tree-in-flight==1 is
// the statement "the cursor is singular" — there is structurally nowhere to park a second leaf.
class ReentryCursor;                   // owned by TreeSearch; not exposed by value.

// The result of one advance/resume: the tree either NEEDS a leaf (it has parked at AWAITING_LEAF
// with exactly one outstanding request), or it has DECIDED (the survivor + improved-π). A FAILED
// invariant (an impossible state) is NOT an arm here — it is an assert/abort, distinct from a leaf
// RPC failure (the driver's Error, §5). A typed sum, returned by value (P9 rules 2, 5);
// std::variant carries the "which arm" in the type — no sentinel, no nullable pointer.
struct NeedsLeaf { LeafRequest request; };
struct Decided   { Decision decision; };
using Step = std::variant<NeedsLeaf, Decided>;

// The resumable Gumbel-AZ tree. It OWNS its _Node graph, its RNG stream, its state word, and its
// single ReentryCursor; it does NOT own a net, a socket, or a thread. The driver (a SearchRuntime
// impl) calls advance() to start and resume(prediction) to feed each leaf back, alternating until
// Decided.
//
// INVARIANT (the serial-per-tree exactness mechanism, cpp-batched-search.md §1.2 / §3.1): between an
// advance()/resume() returning NeedsLeaf and the matching resume(), the tree has EXACTLY ONE
// outstanding leaf and CANNOT issue a second. This is structural — the only way to get the next leaf
// is to resume() with the previous one's value. Per-tree in-flight == 1 falls out of the interface
// shape (the singular ReentryCursor); it is not a runtime check the driver must remember to make.
//
// This is a FUNCTIONAL-CORE shape (ADR-0012 P9): advance/resume are total over already-validated
// inputs (the tree's own state + a NetPrediction the driver already decoded); they neither throw nor
// do I/O. All transport/effect lives in the driver. The Step is returned by value.
class TreeSearch {
  public:
    // Create a tree for one decision against a borrowed env. Fallible only if the task is itself
    // malformed (e.g. an empty belief / a zero-legal root) — a boundary condition, hence expected
    // (P9 rule 5). A throwing ctor cannot return a value (rule 5), so construction is a create()
    // factory.
    [[nodiscard]] static std::expected<TreeSearch, Error>
    create(const Environment& env, const SearchTask& task);

    // Advance the search forward — issuing per-tree RNG draws and running any NO-LEAF sims inline —
    // until it reaches a leaf that needs a net forward (returns NeedsLeaf, cursor non-empty) OR the
    // decision is done (returns Decided). The FIRST call performs the root leaf-eval request
    // (gumbel_search.py:235 — the whole decision waits on the root forward).
    //
    // LEGAL only when the tree is READY (no outstanding leaf). Calling advance() while AWAITING_LEAF
    // is a driver bug — an assert/abort, not an expected (FIX-4: the symmetric guard to resume()'s).
    [[nodiscard]] Step advance();

    // Resume from AWAITING_LEAF with the leaf's prediction. THE NO-LEAF-SIM CONTRACT (FIX-3, the
    // exact point the stream is most easily reordered, cpp-batched-search.md §1.3/§4):
    //   * apply the CLIENT-SIDE float32 masked-softmax to the returned RAW logits (the prior is
    //     computed IN-SEARCH, float32, not on the wire; cpp-batched-search.md §1.3); the value is
    //     already de-standardized by the wire;
    //   * run v_mix / improved_policy / W-N backup of THIS leaf;
    //   * then CONTINUE the search — which may consume ZERO OR MORE further per-tree RNG draws
    //     (the next sims' sample_world determinizations) and may run intervening NO-LEAF sims to
    //     completion inline (a drawn TERMINATE world short-circuits with -λ·exit_cost and NO net
    //     call, gumbel_search.py:345-346 — it consumes an RNG draw but issues no NeedsLeaf);
    //   * until it reaches the NEXT leaf (return NeedsLeaf) or finishes (return Decided).
    // So resume() LEGITIMATELY ADVANCES THE RNG STREAM and returns the next NeedsLeaf/Decided after
    // absorbing all intervening no-leaf sims. There is deliberately NO Step arm for "I drew, it
    // terminated, no leaf yet": those draws live inside the resume()→next-NeedsLeaf transition, and
    // the §7.2-layer-2 structural-determinism test is precisely the proof that they happen in the
    // SAME ORDER as the synchronous reference (it asserts the exact NeedsLeaf request sequence under
    // canned leaves). The "preserved by construction" claim is EARNED only by that test passing.
    //
    // Resuming a tree that is not AWAITING_LEAF is an INVARIANT violation (a driver bug) — an
    // assert/abort, not an expected (P9: expected is for the world's boundary conditions, assert for
    // one's own impossible states).
    [[nodiscard]] Step resume(const NetPrediction& prediction);

  private:
    explicit TreeSearch(/* moved-in state */) noexcept;
    // owns: _Node graph, std::mt19937_64 rng, the single ReentryCursor (state word + parked path +
    // SH bookkeeping), borrowed const Environment&. No net, no socket, no thread.
};
```

**The choice: Option B.** Defended on the merits, not on the established lean:

1. **No fiber dependency.** On a 4-vCPU scratch host, a `boost.context` dependency is a real cost
   that B avoids entirely. The state machine is plain C++23.
2. **It is the P9 functional-core shape.** advance/resume are total value-functions of typed inputs
   returning a typed `Step` by value; every effect (the socket, the thread, the wait) lives in the
   driver's imperative shell. Option A's hidden-yield call site is the P9-rule-3 invisible-effect
   anti-pattern. This is the single strongest argument: B makes the search *unit-testable in
   isolation* (feed canned predictions, assert the Step sequence — §7.2 layer 2) precisely *because*
   it does not block or yield.
3. **The same interface is the structural-determinism parity-test seam.** §7.2 layer 2 (the
   single-tree structural-determinism test) needs to feed canned `NetPrediction`s to the search and
   assert the `NeedsLeaf` request sequence — which is *exactly* `resume()` fed scripted values. With
   Option A there is no such seam without instrumenting the fiber; with Option B the test seam **is**
   the production interface. One interface serves the runtime *and* the parity harness.
4. **Embodiment 2 needs the continuation anyway.** `cpp-batched-search.md` §3.4 calls the
   continuation refactor "the one structural gap" Embodiment 2 needs regardless. Option B pays that
   cost once, as a clean state machine, and gets the parity seam free; Option A pays it as a fiber
   runtime and still lacks the clean test seam.

**The honest cost of B (the genuine hard part — not papered over).** The continuation refactor
**reaches into the search's recursive descent.** `gumbel_search.py`'s `_descend` is a recursion that
calls the net at the bottom and reads the running `W/N` on the way back up; `_sequential_halving`
loops over phases, each phase over surviving candidates, each candidate over `per_action` sims, each
sim a `_simulate_root_action` that averages `c_outcome` determinizations, each of which descends to
one leaf. To make this *resumable*, the descent's call stack must become the **explicit
`ReentryCursor`** the tree parks on and re-enters: where the recursion would block on `net.predict`,
it instead records "I am at this node, mid-this-sim, mid-this-candidate, mid-this-phase, awaiting
this leaf" and returns. This is a real restructure of the search's control flow, and it is
**load-bearing that it not perturb the three Danihelka invariants or the per-tree RNG draw order**
(§7). The restructure preserves them *by construction* only if the reentry resumes at exactly the
draw the recursion was about to make — which is why §7.2 layer 2 is **the** gating precondition of
the benchmark (§7.1, FIX-1), not an optional test: it is the proof the refactor did not reorder the
stream. **This refactor is this spec's commissioned step (§2.2), sequenced after the synchronous
Gumbel port lands** — the spec names the reach honestly so the maintainer prices it.

---

## 4. The DEALER submit/poll client — the one genuinely new component

Embodiment 1 reuses the existing blocking `ZmqNetClient` unchanged (it *is* the thread-per-inflight
transport — the consult's keepers/superseded decomposition). Embodiment 2 needs a **new non-blocking
DEALER client** that the consult names as "the only genuinely new component." It **reuses the
`inference_wire` codec verbatim** (P7: do not re-author the wire) and is otherwise a new fail-loud
typed component.

```cpp
// cpp/include/chocofarm/zmq_dealer_client.hpp  (forward-looking — not built)

// A non-blocking DEALER submit/poll client: many outstanding sends on ONE socket, replies polled and
// routed by correlation id. The codec is the SHARED inference_wire (encode_request /
// decode_response) — this client RE-AUTHORS NOTHING on the wire (P7); it owns only the DEALER socket
// lifecycle and the submit/poll/route mechanics the blocking REQ client cannot express.
//
// Lifetime (P9 RAII): the zmq context + DEALER socket are RAII members; the type is MOVE-ONLY.
// Construction can fail (ctx/socket/connect), so it is a create() factory over a private ctor (a
// throwing ctor cannot return a value — P9 rule 5). void* handles in the header (no zmq.h), the same
// discipline as ZmqNetClient.
class ZmqDealerClient final {
  public:
    [[nodiscard]] static std::expected<ZmqDealerClient, Error>
    create(const std::string& endpoint, int poll_timeout_ms = 5000);

    ~ZmqDealerClient();
    ZmqDealerClient(const ZmqDealerClient&) = delete;
    ZmqDealerClient& operator=(const ZmqDealerClient&) = delete;
    ZmqDealerClient(ZmqDealerClient&&) noexcept;
    ZmqDealerClient& operator=(ZmqDealerClient&&) noexcept;

    // SUBMIT: encode `features` (the shared codec) and send on the DEALER socket WITHOUT waiting for
    // a reply — many of these can be outstanding at once (the property the blocking REQ lacks). The
    // corr_id is the routing key the caller will match the reply against (§4.1).
    //
    // PRE-SEND VALIDATION (FIX-5a — load-bearing): submit() MUST perform the SAME empty / non-finite
    // (NaN/Inf) feature-row rejection the blocking client already does (zmq_net_client.cpp:121-126),
    // returning a typed Error at submit time. This is not cosmetic: the server's malformed-request
    // path (_reject in inference_server.py) LOGS AND DROPS WITH NO REPLY FRAME, so a malformed row
    // that reached the wire would NEVER produce a reply — its corr_id would sit in the outstanding
    // set until the per-corr_id deadline fires (§5), a 5s stall instead of an instant loud reject.
    // Validating before the wire turns that latent timeout into an immediate typed failure.
    [[nodiscard]] std::expected<void, Error>
    submit(std::uint64_t corr_id, std::span<const float> features) const;

    // POLL: wait up to poll_timeout_ms for the NEXT available reply, decode it (the shared codec),
    // and return its (corr_id, NetPrediction). Returns:
    //   * a value  — one reply is ready, routed by its ECHOED corr_id (§4.1);
    //   * an Error — a transport failure or a malformed reply (the codec's typed rejection), OR a
    //                poll timeout (server-down / overloaded) — the loud non-hang path (§5).
    // The completion loop calls poll() repeatedly to drain replies and dispatch each to its tree.
    [[nodiscard]] std::expected<Completion, Error> poll() const;

  private:
    // ... void* ctx_, sock_ (no zmq.h in the header) ...
};

// One polled reply: which tree it belongs to (the echoed corr_id) + the decoded prediction.
struct Completion {
    std::uint64_t corr_id = 0;
    NetPrediction prediction;
};
```

### 4.1 Correlation-id routing and its ordering guarantee

The server is a single-threaded ROUTER that scatters replies by ZMQ identity frame
(`zmq-inference-service.md` §3). ROUTER↔DEALER is the natural pair and **preserves per-peer
ordering**: for one DEALER peer, the ROUTER delivers that peer's replies in the order it *processed*
that peer's requests. `cpp-batched-search.md` §3.4 leans on this: with **one DEALER per multiplexer
thread**, replies for that peer arrive in arrival order, and a per-thread FIFO queue of `(corr_id →
tree)` would route correctly with **no wire change** — adding the explicit echoed id only "if replies
could ever arrive out of order (they cannot with one single-threaded ROUTER)."

**This spec tests that lean and finds the positional FIFO too fragile to be the *primary* mechanism,
and adopts an echoed corr_id instead — here is the honest reasoning.** The greedy-drain server
(`inference_server.py`) does **not** reply strictly in submit order across drains: it blocks for ≥1,
drains *all* queued requests up to `max_batch`, runs one forward, then scatters by identity. Within
one drained batch the scatter preserves the drained (arrival) order — so two requests in the *same*
drain are fine — but a request that just missed batch K waits for batch K+1, so a DEALER that
submitted A then B can legitimately receive B's reply before A's when A and B land in different
drains. A pure positional FIFO (`the i-th reply is the i-th submit`) is therefore **not** guaranteed:
per-peer ordering guarantees the *frames* arrive in the order the ROUTER *sent* them, not that the
ROUTER *sent* them in the order the DEALER *submitted* them, once the greedy drain reorders across
batches. The §3.4 note half-acknowledges this ("Only if replies could ever arrive out of order... add
an explicit echoed u32 request-id") but files it as can't-happen; the greedy drain makes it
can-happen.

There is a second, sharper reason the positional FIFO is unsafe under *this* server: the
malformed-request `_reject` path **drops the frame and sends no reply** (`inference_server.py`). A
positional FIFO assumes one reply per submit; a silently-dropped request breaks the positional
correspondence for *every* subsequent reply on that peer, not just its own. (The C++ submit-side
validation of §4 keeps a malformed row off the wire, but a positional FIFO that depends on
"exactly one reply per submit" is one server-side change away from a silent mis-route — exactly the
brittleness an explicit id removes.)

**The mechanism, therefore:** an **explicit echoed `u64` corr_id** carried in the frame, matched at
`poll()`. This is a **real codec amendment** — a new field in the `wire_spec` SSOT, derived on both
sides (P7: the C++ mirror header and the Python `wire_spec.py` both derive the new field; the
`tests/test_wire_drift.py` net covers it; the server echoes the request's corr_id into its reply).
The routing then keys on the echoed id, not on arrival position, so a reordered drain or a dropped
request cannot mis-route. Fail-loud: a `poll()` that returns a corr_id the driver has no parked tree
for is a typed Error (a desync — never a silent drop), and a parked tree whose corr_id never returns
within its deadline is a per-tree timeout routed as that tree's failure (§5).

**The cost, named:** this is a wire amendment, so it touches the P7 SSOT and the drift net, and the
server must echo the field — it is **not** "no wire change" as §3.4 hoped. The trade is correctness
(a reorder-proof, drop-tolerant router) for a one-field codec bump. This spec recommends paying it.
**Open question (§8.1):** confirm whether the maintainer prefers the echoed-id codec bump now vs.
constraining the server to a deterministic per-peer reply order (a barrier drain) that would preserve
the positional FIFO — the latter keeps the wire frozen but couples the routing to a server
discipline, which is the more brittle coupling.

---

## 5. Failure routing — typed, owned, loud (ADR-0002 / P9)

`cpp-batched-search.md` §3.6 is the contract: a timed-out / server-down / malformed leaf RPC is a
**typed Error routed to the owning tree**, aborting that tree's episode loudly — never a silent drop
and never a stale/zero-value substitution (which would corrupt that tree's backup and its
distribution). The three impls realize it differently but to the **same observable**:

- **`SerialRuntime` / `ThreadPerTreeRuntime` (Embodiment 1):** the blocking `ZmqNetClient::predict`
  already returns `expected<NetPrediction, Error>`. The driver, between `advance()` and `resume()`,
  calls `predict`; on the Error arm it does **not** call `resume()` — it abandons that tree and
  propagates the Error out of `run()`. The tree's parked state is dropped with it. Embodiment 1 gets
  the routing nearly free (the Error unwinds the one thread that owns that tree).

- **`FiberMuxRuntime` (Embodiment 2):** the completion loop's `poll()` returns either a
  `Completion{corr_id, prediction}` (→ look up the parked tree by echoed corr_id, `resume(prediction)`)
  or an Error. An Error from `poll()` carrying a corr_id (a malformed reply for a known request)
  routes to *that* tree as its failure. An Error without a corr_id (a transport/poll failure) is
  ambiguous — it cannot be attributed to one tree — and aborts the batch.

  **The two completion-path holes the corr_id router introduces, both closed (FIX-5b):**
    1. **A reply's corr_id matches no parked tree** (a desync) — a typed Error, a loud abort, never
       silently slid (§4.1).
    2. **A parked tree's corr_id never returns** — the server's silent-drop path (`_reject`) or a
       lost reply. The driver tracks a **submit timestamp per outstanding corr_id**; a corr_id past
       its **per-tree deadline** is that tree's failure. This deadline is the **sole backstop** for
       the silent-drop path, so it must be sized explicitly: **>= the server's worst-case forward
       latency under full `max_batch` load** (so it does not false-fire when the tree is merely
       waiting behind a full drain), and the §4 submit-side validation keeps malformed rows off the
       wire so this 5s-class backstop is reserved for genuine server/transport faults, not for
       self-inflicted drops.

**The invariant the failure path must not break:** a failed tree's leaf is never substituted with a
zero or stale value to "keep going." That is the silent-failure ADR-0002 forbids and would corrupt
the aggregate-equivalence bar (§7.3 layer 4) invisibly. Fail the tree (and, under the §1
whole-batch contract, the batch), loudly, with the diagnostic the `Error` carries. A mid-batch
failure discards the partial position-indexed buffer (§1, GAP-2) — no partial vector escapes.

---

## 6. The two in-flight caps

`cpp-batched-search.md` §3.3 names two caps; both are load-bearing and they live at different layers.

- **Cap (a) — per-tree in-flight == 1, structural.** This is the §1.2 exactness mechanism and it is
  enforced **by the `TreeSearch` interface shape** (the singular `ReentryCursor`, §3.2), not by a
  runtime check: the only way to get the next leaf is to `resume()` the previous one, so a tree
  physically cannot have two leaves outstanding. No driver can violate it by accident — the type
  system holds it. This is the defining invariant of the whole Axis-A regime (the moment a second
  leaf issues from one tree before backprop, it is virtual-loss / Axis C, `cpp-batched-search.md`
  §2.1) and making it structural rather than a remembered assertion is the chief safety dividend of
  Option B.

- **Cap (b) — global concurrency, sized vs the host wall AND the server `max_batch`.** The number of
  trees with a leaf in flight at once. It sizes the server's achieved batch `B` and must be large
  enough that the greedy drain stays near-full, but it is bounded by the **~1.9× host-contention
  ceiling on the 4-vCPU VM** (CLAUDE.md) and capped above by the server's **`max_batch` (256, the
  `inference_server.py` default)**. On this host with a tiny MLP the realistic parked-tree count is
  **far below `max_batch`** — so the operating regime is expected to be **demand-limited, not
  cap-limited** (cap (b) ≪ 256). That is itself a §6-Q5 finding: if cap (b) never approaches
  `max_batch`, the multiplexer's many-in-flight advantage is moot at this scale, which is precisely
  what §6-Q5 asks. The sweep range is therefore **[1 .. the host knee], expecting B ≪ max_batch**
  (GAP-4). Concretely:
    - `ThreadPerTreeRuntime`: cap (b) **is** the thread count N. The §3.4 caveat ("acceptable only
      when threads ≫ cores") means N must exceed cores for the IO-blocked threads to overlap and
      keep `B > 1` — but each thread is a real OS thread on a 4-vCPU host, so N is bounded by the
      memory of N parked `_Node` heaps and the scheduler overhead of N ≫ 4 threads. This is the
      embodiment whose cap is the *least* comfortable on this host, which is exactly why the
      benchmark exists. It also needs **N blocking `ZmqNetClient` instances** (the REQ client is not
      thread-safe — "give each worker its own", `zmq_net_client.hpp:25-26`), i.e. N REQ peers on the
      one ROUTER; their lifecycle and the N-sockets-vs-1-socket transport footprint asymmetry vs
      Emb2 are a stated, reportable axis of the result (GAP-5), not a hidden difference.
    - `FiberMuxRuntime`: cap (b) is the number of parked continuations K on the multiplexer thread —
      many K per OS thread, so K can be large for little thread-scheduling cost; one DEALER socket;
      the bound is again memory (K `_Node` heaps) and the point of diminishing return where `B`
      already saturates demand (which, at this scale, is well below `max_batch`).
    - `SerialRuntime`: cap (b) == 1 by definition (one tree at a time); `B` is always 1.

---

## 7. Benchmark harness and parity plan

### 7.1 What makes the Emb1-vs-Emb2 benchmark FAIR (and what would secretly rig it)

**Gating precondition (FIX-1 — the validity bar, stated first because the whole comparison rests on
it).** All three runtimes drive the *same refactored* `TreeSearch` (§3). If the continuation refactor
perturbed the per-tree RNG draw order or the SH bookkeeping, **all three runtimes are wrong
identically**, and the cross-runtime aggregate-equivalence bar (§7.3 layer 4) would pass while every
runtime diverges from the real Gumbel search. So the benchmark is **meaningless until the
§7.2-layer-2 structural-determinism test passes against the un-refactored synchronous Gumbel
reference** (the maintainer's landed port, §2 build-step iii), AND `SerialRuntime` matches that
reference in aggregate (a `SerialRuntime`-vs-original-`gumbel_search` aggregate check over the parity
corpus, not only the canned-leaf structural check). The shared substrate must be **proven faithful to
the real search** before any Emb1/Emb2 timing is trusted. This is a precondition, not a follow-up
test.

The benchmark compares **scheduling**, so everything below the scheduler must be **identical and held
fixed** across the two timed runs:

- **Same search, same `TreeSearch`.** Both drivers drive the identical advance/resume state machine
  (§3) — the reason for Option B. A throughput delta is then attributable to scheduling, not to two
  different searches.
- **Same scenario, captured and reproducible (ADR-0009).** The captured scenario is the tuple
  **`(states corpus + frozen net params + seeds + cfg)`** (FIX — GAP-3): a pinned set of
  `SearchTask`s — `(loc, bw, collected, λ, seed, cfg)` — captured to a file (the `states.npz` analog
  for the C++ runtime, the ADR-0009 `capture_states.py` discipline) **plus the frozen net params**
  the server serves (without the pinned weights, "same scenario" is not reproducible — the leaf
  values, hence how many leaves each tree issues, depend on the net). The C++ harness loads the
  states corpus from the captured file and the net params via the existing manifest/blob path the
  server already reconstructs. Same belief widths (full C(N,K) down to singleton), same λ, same
  budget (`m=12, n_sims=48`, the gumbel_search defaults), same matched seeds.
- **Same server, same params, same `max_batch`.** One `InferenceServer` instance with
  `StaticParamsSource` (no redis, no version churn — net-version-per-decision held constant, §8.3)
  and the same `forward_core` weights, so both embodiments hit the same greedy-drain ROUTER. The
  server must **not** be restarted between the two runs (warm XLA).
- **Same host pinning.** `--cores 0,1,2,3` on the 4-vCPU VM for both.

**The fairness metric is achieved-B, not matched cap count (FIX-2).** It is tempting to assert
"sweep both over matched cap-(b) points, Emb1's N == Emb2's K" as the fairness mechanism. **It is
not one.** Emb1's cap (b) is **OS threads** (each a real scheduler entity contending for the 4 vCPUs)
and Emb2's cap (b) is **parked continuations on one thread** (near-free). At N==K==64, Emb1 has 64 OS
threads thrashing 4 cores while Emb2 has 64 cheap continuations on 1–2 threads — holding the *number*
equal does not hold "offered concurrency to the server" equal in any way that isolates *scheduling*;
it holds a number equal whose *meaning differs per side*. So a reader could not tell whether an Emb2
win is scheduling or merely "64 threads is past the 4-vCPU thrash knee" — the exact confound the
benchmark exists to resolve. The honest comparable is therefore **throughput at matched achieved
server batch size `B`** and **throughput vs core-utilization**, with the cap-(b) sweep demoted to
*one parameterization that traces each runtime's frontier* (sweep each runtime over its own cap-(b)
range, plot throughput-vs-B and throughput-vs-utilization, and compare the **frontiers**), not a
point-for-point N==K equality claim.

**What else would secretly advantage one side (name it to forbid it):**
- Letting Embodiment 2's single multiplexer thread monopolize a core while Embodiment 1's blocked
  threads are descheduled — the host-pinning controls for the wall, but the harness must report
  **core utilization** so a "win" that is really one side under-subscribing the host is visible.
- A cold server for one run and a warm one for the other (XLA compile on the first batch). Warm both
  before timing.
- Different RNG seeds per side — matched seeds, so the per-tree trajectories (which determine how
  many leaves each tree issues, hence the offered load) are identical inputs to both schedulers.

### 7.2 The metric set

Per the ADR-0009 metric vocabulary, extended for this scheduling axis:

- **Throughput — decisions/s** (the headline comparable), over the fixed task corpus, wall-clock,
  warmed, ≥3 repetitions with reported variance.
- **Achieved server batch size `B`** (mean and distribution) — the server reports its per-drain batch
  size; this is the *mechanism* by which a scheduler wins (one that keeps `B` higher hides more
  latency) **and the fairness anchor (§7.1)**: the headline comparison is throughput **at matched
  B**. A throughput win with no `B` increase is suspicious and must be explained.
- **Core utilization vs the ~1.9× ceiling** — the four-vCPU wall is the real bound (CLAUDE.md); the
  harness reports achieved parallel speedup so "Emb2 is faster" is read against "but neither beats
  1.9×," and so an under-subscription win is visible.
- **Leaf-request count per decision** (from `Decision::leaf_requests`) — an invariant cross-check:
  for matched seeds and canned-equivalent leaves the *structural* request count must be identical
  across all three runtimes (a different count means the scheduling changed the search — a bug, not
  a speedup). This is also why the benchmark pins one net version (§8.3): a version straddle would
  change this count and void the cross-check.
- **Cap-(b) sweep, traced as a frontier** — throughput vs achieved `B` and vs core-utilization for
  each runtime, so the comparison is curves over the host's operating regime, not matched-cap points
  (§7.1).

### 7.3 The parity bar — aggregate, not per-decision

This is the load-bearing fidelity subtlety and the spec states it plainly. Because of
**batch-composition roundoff** (`cpp-batched-search.md` §2.2 / `zmq-inference-service.md` §4): *which*
other trees co-batch a given tree's leaf depends on arrival timing under the greedy drain, so the low
bits of that leaf's value depend on its batch neighbours, and at a near-tie (§1.3) a ≤1e-4
perturbation can legitimately flip which SH survivor or PUCT child is chosen. **So per-decision
results legitimately differ across runtimes** (Serial vs ThreadPerTree vs FiberMux schedule different
batch compositions) — comparing decisions byte-for-byte across runtimes would red on a *correct*
implementation. The bar is therefore **aggregate behavioral equivalence** (ADR-0009 / ADR-0011 tier 2):
over **N≥300 decisions across ≥2 seeds**, the action distribution and improved-π statistics must be
statistically indistinguishable across the three runtimes within Monte-Carlo CI, with the MC standard
error reported.

The parity/test plan, in the four composing layers `cpp-batched-search.md` §5 names, instanced for
this seam:

1. **Net-forward parity (inherited, unchanged).** The wire path is already pinned at max|Δ| < 1e-4
   (`tests/test_zmq_net_cpp.py`, measured ~e-7). The runtimes do not touch it.

2. **Single-tree structural-determinism test (the advance/resume test — Option B's dividend AND the
   benchmark's gating precondition, §7.1).** Drive **one** `TreeSearch` with a **recording stub**
   that returns **canned, byte-identical** `NetPrediction`s, and assert that the **sequence of
   `NeedsLeaf` requests** (the `(loc, bw, collected, λ)` leaf states, in order, including the
   intervening no-leaf sims' effect on *which* leaf comes next — §3.2 no-leaf-sim contract) and the
   final `Decision` are **identical** to the **un-refactored synchronous serial Gumbel reference**
   fed the same canned leaves. This isolates *search structure* ("which leaf is requested next") from
   leaf numerics and **directly proves the continuation refactor did not reorder the per-tree RNG
   stream or perturb the descent** (§3's hard part). It is the same test for all three runtimes
   (canned leaves remove batch-composition variation), so it also proves the three drivers drive the
   *same* search. This test seam **is** the `resume()` interface — the chief reason Option B was
   chosen (§3.2).

3. **The three Danihelka invariants, per-decision, unchanged.**
   `test_executed_action_is_sh_survivor`, `test_vmix_prior_weighted`,
   `test_sequential_halving_spends_full_budget` (`cpp-batched-search.md` §1.1/§5). Each tree runs its
   own private SH budget regardless of how it is scheduled, so these must still hold under every
   runtime; a failure means a tree's budget got coupled across the scheduler — i.e. it slid into Axis
   C. The continuation refactor (§3) is the most likely place to break these (the SH bookkeeping
   becomes the explicit `ReentryCursor`), so they re-run against the wrapped search.

4. **Aggregate behavioral equivalence (the cross-runtime bar + the fidelity reference).** The N≥300 /
   ≥2-seed action-distribution + improved-π comparison across {Serial, ThreadPerTree, FiberMux}
   within MC CI (above). Plus the **`SerialRuntime`-vs-original-`gumbel_search` aggregate check** (the
   §7.1 precondition: the shared substrate is faithful to the *real* search, not merely
   self-consistent across the three drivers). Plus a **batch-composition stress test**: vary cap (b)
   and inject arrival jitter, and assert the aggregate stays inside CI — pinning that
   batch-composition roundoff stays inside the ADR-0011 tier-2 envelope and does not, in aggregate,
   change the policy.

**The two parity obligations the seam inherits but does not itself satisfy** (`cpp-batched-search.md`
§5, both at the per-tree consumer, both orthogonal to scheduling): the **in-search float32
masked-softmax** (the prior computed client-side in `resume()`, in float32, not on the wire) and the
**float32-prior / float64-Q weak-promotion seam** in `v_mix`/`improved_policy`. These belong to the
Gumbel port (§2), not to the runtime; the runtime must merely call them in `resume()` faithfully. A
near-tied-logit kernel test on those is the Gumbel port's obligation; this spec names them so a
"looks exact" runtime does not get blamed for (or credited with) a divergence that lives in the
search math.

---

## 8. Open questions, boundaries, and non-goals

### 8.1 Correlation routing: echoed-id codec bump vs. deterministic-drain server
The §4.1 analysis recommends an **echoed `u64` corr_id** (a wire amendment) over the §3.4 positional
FIFO, because the greedy drain can reorder per-peer replies and the `_reject` drop breaks the
one-reply-per-submit assumption. The open decision for the maintainer: pay the one-field codec bump
(robust, couples nothing to server timing), or constrain the server to a barrier/deterministic drain
that preserves the positional FIFO (wire frozen, but the routing now depends on a server
discipline — the more brittle coupling). This spec leans echoed-id; not silently resolved.

### 8.2 Failure-batch granularity (§1)
Whole-batch-abort `expected<vector<Decision>>` (this spec's pick — loudest, simplest §5 routing, and
right at benchmark scope) is adopted now. Per-task `vector<expected<Decision, Error>>` (one tree's RPC
failure does not lose its 63 healthy siblings) is a self-play-consumer concern that does not exist
yet; promote to it **when** the self-play consumer lands, if losing one tree's episode is preferable
to losing the batch. Deferred, not litigated (§1 rationale).

### 8.3 Net-version consistency per tree-decision (inherited open question, §3.7 — NOT resolved here)
`cpp-batched-search.md` §3.7 / §6-Q1: a single tree's ~48 leaves can straddle a version reload at
batch boundaries today, perturbing that tree's distribution. The benchmark harness pins one version
(`StaticParamsSource`, §7.1) — **not only** to avoid the numeric confound but because a mid-decision
version straddle would change a tree's **leaf-request count** (and which actions survive), voiding the
structural cross-check (§7.2) that proves the three runtimes drive the *same* search. That pin is a
**benchmark control, not a resolution.** Whether production self-play pins one frozen net version per
tree-decision (freeze weights during a generation phase) or accepts the straddle is an open decision
orthogonal to the runtime seam. Surfaced, not silently resolved.

### 8.4 Boundaries — what this spec is NOT
- **Not the Gumbel search.** Its decision math (SH, PUCT, v_mix, the float32 hazards) is the
  in-progress separate port (§2); this spec wraps it, commissions the continuation refactor *after*
  it lands as a synchronous search, and must not conflict with the in-progress mixed-precision work.
- **Not within-tree leaf batching / virtual loss (Axis C).** Per-tree in-flight stays structurally 1
  (§6 cap a, the singular `ReentryCursor`). The seam *cannot* express a second outstanding leaf per
  tree (§3) — Axis C is unreachable by construction, exactly as `cpp-batched-search.md` §2.1
  requires.
- **Not a re-authoring of the wire.** The runtimes compose the existing `inference_wire` codec and
  the existing `NetEvaluator` port (P7); the DEALER client adds a new socket lifecycle and submit/
  poll/route, reusing the codec — except for the one echoed-id field amendment of §4.1, which is a
  P7-disciplined SSOT change, not a second hand codec.
- **Not the server.** The server stays the single-threaded greedy-drain ROUTER
  (`zmq-inference-service.md` §3/§8 — workers stay dumb, the server batches; no XLA in a worker
  thread). The runtimes are entirely worker-side. (The §4.1 echoed-id and the §8.1 barrier-drain
  alternative are the *only* server touches this spec contemplates, and they are mutually exclusive
  open questions, not adopted changes.)

### 8.5 Non-goals
Virtual loss; a shared tree; root parallelization; any within-tree leaf batching; moving masking
server-side; adding XLA-bearing threads to the server; resolving net-version-per-decision; building
the DEALER client / `FiberMuxRuntime` **before** the Gumbel port lands and before the §6-Q5 benchmark
justifies the multiplexer's complexity (building it speculatively is the ADR-0011 Rule-3
measure-first violation in the opposite direction — the consult's §6 symmetric warning).

**On `SerialRuntime`'s status (review CUT-3, resolved — kept as a production-shaped runtime, with
rationale).** `SerialRuntime` is a real `SearchRuntime` subclass, not merely a test fixture. It earns
that status three ways: (i) it is the **fidelity reference** the §7.1 precondition checks the shared
substrate against (the `SerialRuntime`-vs-original-Gumbel aggregate check, §7.2 layer 4); (ii) it is
the **B==1 throughput floor** every cap-(b) frontier is read against; (iii) it is the **clean driver**
of the §7.2-layer-2 structural-determinism test (feed canned leaves, assert the `NeedsLeaf`
sequence). A `ThreadPerTreeRuntime` with N==1 would also produce B==1, but it would do so *through the
threading machinery* — so it could not serve as the machinery-free reference that isolates a substrate
fault from a scheduler fault. Keeping `SerialRuntime` as the one driver with **no concurrency
apparatus at all** is what makes "is the bug in the search or in the scheduler?" answerable; that is a
production-shaped role, not an adopt-or-delete-E ornament.

---

## 9. ADR conformance mapping

- **P7 (cross-language wire discipline).** The runtimes **reuse** the one `inference_wire` codec and
  the one `NetEvaluator` port — no second hand codec. The single wire change (the echoed `u64`
  corr_id, §4.1) is made the P7 way: a new field in the `wire_spec` SSOT, derived on both the Python
  and C++ sides, covered by the `tests/test_wire_drift.py` net, with the server echoing it — one
  authoritative definition, every side derives. The transport fence P7 draws is honored: `{REQ,
  DEALER}` are two instances of a messaging fabric behind the codec, neither enshrined as "the one
  way" — which is exactly what makes a benchmark between them coherent.
- **P9 (functional core / imperative shell).** `TreeSearch::advance/resume` are the functional core:
  total value-functions of typed inputs returning a typed `Step` by value, no I/O, no throw — the
  `Step = variant<NeedsLeaf, Decided>` carries which-arm in the type (no sentinel). All effect (the
  socket, the threads, the poll loop, the waits) lives in the `SearchRuntime` imperative shells. Every
  signature takes bounds-carrying inputs (`std::span<const SearchTask>`, `std::span<const float>`) and
  returns by value; failure is `[[nodiscard]] std::expected<…, Error>`, absence (where it arises)
  `std::optional`, never a sentinel or nullable pointer; `SearchTask` holds no reference member and
  no per-node-allocating `std::set` (the env is borrowed once by `run`, `collected` is a bitmask), so
  it is trivially vector-storable behind the `std::span`; the DEALER client and `TreeSearch` are RAII,
  move-only, `create()`-factory-constructed (a throwing ctor cannot return a value). The
  per-tree-in-flight-==1 invariant is structural (the singular `ReentryCursor`), and a driver-misuse
  (resume of a non-awaiting tree, or advance of an awaiting tree) is an assert/abort (an invariant
  violation, a bug), not an `expected` (a boundary condition) — the P9 rule-5 distinction held.
- **ADR-0009 (measure-first).** The seam exists *to run the §6-Q5 gate*: no embodiment is preferred
  before the benchmark; `SerialRuntime` is the captured reproducible baseline and the fidelity
  reference; the harness is a pinned, reproducible `(states + frozen net params + seeds + cfg)`
  scenario with the ADR-0009 metric vocabulary (decisions/s, achieved `B`, core utilization vs the
  1.9× wall, matched seeds), the fairness anchor is **matched achieved-B not matched cap count**, and
  the benchmark is gated on the substrate proving faithful to the real search (§7.1). Building
  `FiberMuxRuntime` before the benchmark justifies it is the measure-first violation §8.5 forbids.
- **ADR-0011 (mechanization / two-tier equivalence).** The parity bar is the two-tier bar applied
  across runtimes: the structural-determinism layer-2 test is **bit-exact** on the `NeedsLeaf` request
  sequence (a logic invariant — the search structure under canned leaves, asserted against the
  un-refactored reference); the cross-runtime layer-4 comparison is **aggregate-behavioral within MC
  CI** (the float-sensitive numerics, where batch-composition roundoff legitimately moves the float).
  The echoed-id wire change is mechanized by the existing drift net (Rule 4: a net over the class, not
  an enumerated instance). The per-tree-in-flight-==1 invariant is mechanized structurally (the
  interface makes the violation un-authorable), the strongest feasible surface (Rule 1).

---

*Public Domain (The Unlicense).*
