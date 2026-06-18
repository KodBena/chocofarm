<!-- docs/design/cpp-search-runtime.md -->

# The C++ SearchRuntime seam: one interface over two leaf-evaluation concurrency embodiments

**Status:** Design record (forward-looking, contracts-first). No code is committed; this is the
artifact the maintainer reviews before any implementation begins. It deepens and tests the
established direction of `docs/design/cpp-batched-search.md` §3.4 open-question 5 (the ADR-0009
measure-first benchmark gate) and the consult
`docs/notes/consult/opus-consult-2026-06-16-zmq-net-client-blocking-req.md` (the matched-pair
finding). It does **not** re-litigate the settled Axis-A / serial-per-tree exactness regime
(`cpp-batched-search.md` §2), and it does **not** design the Gumbel-AZ search itself — that port is
in progress elsewhere; this spec designs the runtime seam **around** it. Read end to end before
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
makes local↔remote-blocking swappable; it does **not** make blocking↔async swappable (the async leaf
call is *submit-and-yield*, a shape `predict(span)->expected<NetPrediction>` cannot express). That is
the §2 finding of the consult, and it is why the seam is a `SearchRuntime`, not a second net impl.

---

## 1. The seam — `SearchRuntime`

The seam owns `{tree scheduling + leaf dispatch}` and exposes a single batch-of-decisions entry
point. The benchmark harness, the self-play loop, and any future driver hold a `const
SearchRuntime&` and never name a concrete impl (P2: a new scheduling embodiment is a new
`SearchRuntime` subclass with zero edits to the search or to its callers).

```cpp
// cpp/include/chocofarm/search_runtime.hpp  (forward-looking — not built)

// One unit of work the runtime schedules: "make ONE Gumbel-AZ decision for this independent
// problem instance, from this observed state, under this live λ." Each SearchTask is a fully
// independent tree (its own RNG stream, its own _Node graph) — the cross-tree independence that
// makes the batch row-independent (Axis A, cpp-batched-search.md §2.1). All inputs are by value /
// const-ref bounds-carrying views (P9 rule 1); the live per-decision scalars (λ, budget) ride the
// task, never baked into a runtime object (P4).
struct SearchTask {
    const Environment& env;            // the dynamics owner (belief, simulate, cost) — borrowed
    Loc loc;                           // observed agent location
    std::vector<uint32_t> bw;          // observed belief world-set (bitmasks over treasure ids)
    std::set<int> collected;           // treasures already collected
    double lam = 0.0;                  // the live Dinkelbach penalty (P4 — per-decision, not frozen)
    std::uint64_t seed = 0;            // the per-tree RNG seed (the _fold_seed discipline, §6)
    GumbelConfig cfg{};                // m, n_sims, c_puct, c_visit, c_scale, c_outcome, max_depth,
                                       //   temperature — the frozen INSTANCE budget for this decision
};

// One decision result: the executed action + the improved-policy target the trainer consumes, and
// the leaf-request count (a fidelity/throughput observable the harness reads — §7). Returned by
// value (P9 rule 2).
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

    // Drive `tasks` to completion and return one Decision per task, IN INPUT ORDER, or a typed
    // boundary failure (a leaf RPC that timed out / a server-down / a malformed reply on SOME tree
    // — §5). The result vector is positionally aligned with `tasks`. A failure aborts the whole
    // batch loudly (it does NOT return a partial vector with a silent hole — §5). The input is a
    // bounds-carrying view; the result is returned by value (P9 rules 1, 2, 5).
    [[nodiscard]] virtual std::expected<std::vector<Decision>, Error>
    run(std::span<const SearchTask> tasks) const = 0;
};
```

Three impls satisfy it, all driving the same `TreeSearch` (§3): `SerialRuntime` (the reference
baseline, one tree at a time), `ThreadPerTreeRuntime` (Embodiment 1), `FiberMuxRuntime` (Embodiment
2). Construction-site choice; the search, the env, and the `Decision` contract do not change a line.

**Why batch-of-tasks, not one-task-at-a-time.** A `run(span<const SearchTask>)` signature is what
makes the fan-out *available* to the runtime: `ThreadPerTreeRuntime` spreads the tasks across N
threads, `FiberMuxRuntime` parks them as K continuations on one thread, `SerialRuntime` loops. A
one-task entry point would force the fan-out up into the caller and re-introduce the very
scheduling concern the seam exists to own. The batch is the unit of concurrency.

**Failure-batch granularity (named trade).** `run` returns `expected<vector<Decision>>` for the
*whole batch* — one bad leaf aborts the batch, not just its tree. The alternative —
`vector<expected<Decision, Error>>`, one fallible slot per tree, so a single tree's RPC failure does
not abort its 63 healthy siblings — is **more permissive and arguably more useful for a long
self-play run**, but it is a real design fork, not an obviously-correct default. This spec picks
whole-batch-abort because (a) it is the loudest shape (ADR-0002), (b) at benchmark scope a single
batch is one timed scenario and a leaf failure invalidates the measurement anyway, and (c) it keeps
the §5 routing identical across all three impls. **Open question (§8.2):** promote to per-task
`expected` when the self-play consumer lands, if losing one tree's episode is preferable to losing
the batch. Flagged, not silently resolved.

---

## 2. The search is the thing being wrapped — the Gumbel-port dependency

The runtime drives the C++ Gumbel-AZ search, which is **being ported separately and is not yet
merged**. `cpp-batched-search.md` §1 is explicit: only NMCS is in C++ today (a forward recursion
with in-stack memorize-and-replay, no per-node backprop), and NMCS uses the `WorldSource`
determinization seam, **not** the `NetEvaluator` leaf port. The Gumbel-AZ search — Gumbel-Top-k →
Sequential Halving → improved-π, the `_decide_root`/`_descend`/`_puct_select`/`_sequential_halving`
structure of `chocofarm/az/gumbel_search.py` — is what this runtime's `TreeSearch` (§3) *wraps and
restructures*.

**This sequences the work hard:** the continuation refactor of §3 **cannot land before the Gumbel
port lands**, because it restructures that search's recursive descent. This spec therefore:

1. **Does not design the Gumbel search.** Its decision logic — the `m`-candidate SH loop, the PUCT
   descent, the `v_mix`/`improved_policy` weak-promotion seam, the three float32 hazards
   (`cpp-batched-search.md` §1.3) — is the maintainer's in-progress mixed-precision port. This spec
   treats it as the black box `TreeSearch` wraps.
2. **Designs the runtime seam *around* it.** The `SearchRuntime` / `SearchTask` / `Decision` /
   DEALER-client / failure-routing / benchmark-harness contracts below are stated against the
   *shape* of a Gumbel decision (one root leaf, then a serial chain of interior leaves, then a
   survivor + improved-π), not against any specific line of the search.
3. **States the ordering dependency as load-bearing.** Build order: (i) the Gumbel port lands and
   passes its own parity (the three Danihelka invariants, the float32 masked-softmax and
   weak-promotion seams — §7); (ii) the `TreeSearch` continuation refactor restructures it into
   advance/resume (§3, Option B); (iii) `SerialRuntime` + `ThreadPerTreeRuntime` land first (they
   need no new transport); (iv) the DEALER client + `FiberMuxRuntime` land; (v) the §6-Q5 benchmark
   runs. The runtime seam must not conflict with the in-progress port — it consumes the search, it
   does not modify its decision math.

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
  rule 3 ("the signature declares every effect") exists to forbid. The effect is real (the leaf
  *does* suspend); hiding it in a synchronous-looking call site is the lie P9 names. (c) Stackful
  fibers carry their own correctness surface (stack sizing, the interaction with thread-local state
  in the matmul/codec path) that the benchmark would have to control for — it muddies the very
  comparison it exists to enable.

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

// The result of one advance/resume: the tree either NEEDS a leaf (it has parked at AWAITING_LEAF
// with exactly one outstanding request), or it has DECIDED (the survivor + improved-π), or it has
// FAILED its own invariant (an impossible state — distinct from a leaf RPC failure, which is the
// driver's Error, §5). A typed sum, returned by value (P9 rules 2, 5). std::variant carries the
// "which arm" in the type — no sentinel, no nullable pointer.
struct NeedsLeaf { LeafRequest request; };
struct Decided   { Decision decision; };
using Step = std::variant<NeedsLeaf, Decided>;

// The resumable Gumbel-AZ tree. It OWNS its _Node graph, its RNG stream, and its per-tree state
// word; it does NOT own a net, a socket, or a thread. The driver (a SearchRuntime impl) calls
// advance() to start and resume(prediction) to feed each leaf back, alternating until Decided.
//
// INVARIANT (the serial-per-tree exactness mechanism, cpp-batched-search.md §1.2 / §3.1): between
// an advance()/resume() returning NeedsLeaf and the matching resume(), the tree has EXACTLY ONE
// outstanding leaf and CANNOT issue a second. This is structural — the only way to get the next
// leaf is to resume() with the previous one's value. Per-tree in-flight == 1 falls out of the
// interface shape; it is not a runtime check the driver must remember to make.
//
// This is a FUNCTIONAL-CORE shape (ADR-0012 P9): advance/resume are total over already-validated
// inputs (the tree's own state + a NetPrediction the driver already decoded); they neither throw
// nor do I/O. All transport/effect lives in the driver. The Step is returned by value.
class TreeSearch {
  public:
    // Create a tree for one decision. The factory is fallible only if the task is itself malformed
    // (e.g. an empty belief) — a boundary condition, hence expected (P9 rule 5). A throwing ctor
    // cannot return a value (rule 5), so construction is a create() factory.
    [[nodiscard]] static std::expected<TreeSearch, Error> create(const SearchTask& task);

    // Advance from READY: run the search forward (root eval if not yet done, else PUCT descent /
    // the SH loop) until it reaches a leaf that needs a net forward, OR until the decision is done.
    // Returns NeedsLeaf{corr_id-less request} when it parks at a leaf (the driver stamps the
    // corr_id, §4), or Decided when the budget is spent. The FIRST call does the root leaf eval
    // request (gumbel_search.py:235 — the whole decision waits on the root forward).
    [[nodiscard]] Step advance();

    // Resume from AWAITING_LEAF with the leaf's prediction: apply the CLIENT-SIDE float32
    // masked-softmax (cpp-batched-search.md §1.3 — the prior is computed IN-SEARCH, not on the
    // wire), de-standardize is already done (the wire returns de-std value), run v_mix /
    // improved_policy / W-N backup, then either continue (return the next NeedsLeaf) or finish
    // (Decided). The NetPrediction is the SAME decoded value both embodiments produce. Resuming a
    // tree that is not AWAITING_LEAF is an INVARIANT violation (a driver bug) — an assert/abort,
    // not an expected (P9: expected is for the world's boundary conditions, assert for one's own
    // impossible states).
    [[nodiscard]] Step resume(const NetPrediction& prediction);

  private:
    explicit TreeSearch(/* moved-in state */) noexcept;
    // owns: _Node graph, std::mt19937_64 rng, the parked-path pointer, the state word, the SH
    // bookkeeping. No net, no socket, no thread.
};
```

**The choice: Option B.** Defended on the merits, not on the established lean:

1. **No fiber dependency.** On a 4-vCPU scratch host, a `boost.context` dependency is a real cost
   that B avoids entirely. The state machine is plain C++23.
2. **It is the P9 functional-core shape.** advance/resume are total value-functions of typed inputs
   returning a typed `Step` by value; every effect (the socket, the thread, the wait) lives in the
   driver's imperative shell. Option A's hidden-yield call site is the P9-rule-3 invisible-effect
   anti-pattern. This is the single strongest argument: B makes the search *unit-testable in
   isolation* (feed canned predictions, assert the Step sequence — §7 layer 2) precisely *because*
   it does not block or yield.
3. **The same interface is the structural-determinism parity-test seam.** §7 layer 2 (the
   single-tree structural-determinism test) needs to feed canned `NetPrediction`s to the search and
   assert the `NeedsLeaf` request sequence — which is *exactly* `resume()` fed scripted values. With
   Option A there is no such seam without instrumenting the fiber; with Option B the test seam **is**
   the production interface. One interface serves the runtime *and* the parity harness.
4. **Embodiment 2 needs the continuation anyway.** `cpp-batched-search.md` §3.4 calls the
   continuation refactor "the one structural gap" Embodiment 2 needs regardless. Option B pays that
   cost once, as a clean state machine, and gets the parity seam free; Option A pays it as a fiber
   runtime and still lacks the clean test seam.

**The honest cost of B (the genuine hard part — not papered over).** The continuation refactor
**reaches into the search's recursive descent**. `gumbel_search.py`'s `_descend` is a recursion that
calls the net at the bottom and reads the running `W/N` on the way back up; `_sequential_halving`
loops over phases, each phase over surviving candidates, each candidate over `per_action` sims, each
sim a `_simulate_root_action` that averages `c_outcome` determinizations, each of which descends to
one leaf. To make this *resumable*, the descent's call stack must become **explicit state** the tree
can park on and re-enter: where the recursion would block on `net.predict`, it must instead record
"I am at this node, mid-this-sim, mid-this-candidate, mid-this-phase, awaiting this leaf" and return.
This is a real restructure of the search's control flow — turning the implicit C++ call stack at the
leaf into an explicit reentrant cursor (the parked-path pointer + the SH bookkeeping). It is the
bigger one-time cost, and it is **load-bearing that it not perturb the three Danihelka invariants or
the per-tree RNG draw order** (§7). The restructure preserves them *by construction* only if the
reentry resumes at exactly the draw the recursion was about to make — which is why §7 layer 2 (the
structural-determinism test) is not optional: it is the proof the refactor did not reorder the
stream. **Crucially, this refactor is the Gumbel port's job, sequenced after it (§2), not this
spec's** — but the spec names the reach honestly so the maintainer prices it.

---

## 4. The DEALER submit/poll client — the one genuinely new component

Embodiment 1 reuses the existing blocking `ZmqNetClient` unchanged (it *is* the thread-per-inflight
transport — the consult's keepers/superseded decomposition). Embodiment 2 needs a **new
non-blocking DEALER client** that the consult names as "the only genuinely new component." It
**reuses the `inference_wire` codec verbatim** (P7: do not re-author the wire) and is otherwise a
new fail-loud typed component.

```cpp
// cpp/include/chocofarm/zmq_dealer_client.hpp  (forward-looking — not built)

// A non-blocking DEALER submit/poll client: many outstanding sends on ONE socket, replies polled
// and routed by correlation id. The codec is the SHARED inference_wire (encode_request /
// decode_response) — this client RE-AUTHORS NOTHING on the wire (P7); it owns only the DEALER
// socket lifecycle and the submit/poll/route mechanics the blocking REQ client cannot express.
//
// Lifetime (P9 RAII): the zmq context + DEALER socket are RAII members; the type is MOVE-ONLY.
// Construction can fail (ctx/socket/connect), so it is a create() factory over a private ctor (a
// throwing ctor cannot return a value — P9 rule 5).
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
    // corr_id is the routing key the caller will match the reply against (§4.1). Returns a typed
    // Error on a transport send failure (NOT a silent drop — §5). Does not block on a reply.
    [[nodiscard]] std::expected<void, Error>
    submit(std::uint64_t corr_id, std::span<const float> features) const;

    // POLL: wait up to poll_timeout_ms for the NEXT available reply, decode it (the shared codec),
    // and return (corr_id, NetPrediction). Returns:
    //   * a value  — one reply is ready, routed by its corr_id;
    //   * an Error — a transport failure or a malformed reply (the codec's typed rejection), OR a
    //                poll timeout (server-down / overloaded) — the loud non-hang path (§5).
    // The completion loop calls poll() repeatedly to drain replies and dispatch each to its tree.
    [[nodiscard]] std::expected<Completion, Error> poll() const;

  private:
    // ... void* ctx_, sock_ (no zmq.h in the header — same discipline as ZmqNetClient) ...
};

// One polled reply: which tree it belongs to (corr_id) + the decoded prediction.
struct Completion {
    std::uint64_t corr_id = 0;
    NetPrediction prediction;
};
```

### 4.1 Correlation-id routing and its ordering guarantee

The server is a single-threaded ROUTER that scatters replies by ZMQ identity frame
(`zmq-inference-service.md` §3). ROUTER↔DEALER is the natural pair and **preserves per-peer
ordering**: for one DEALER peer, the ROUTER delivers that peer's replies in the order it processed
that peer's requests. `cpp-batched-search.md` §3.4 leans on this: with **one DEALER per multiplexer
thread**, replies for that peer arrive in arrival order, and a per-thread FIFO queue of `(corr_id →
tree)` would route correctly with **no wire change**.

**This spec tests that lean and finds it too fragile to rely on as the *primary* mechanism, and
adopts an echoed corr_id instead — here is the honest reasoning.** The greedy-drain server does
**not** reply in request order: it blocks for ≥1, drains *all* queued requests up to `max_batch`,
runs one forward, then scatters by identity. Within one drained batch the per-identity scatter order
is an implementation detail of the scatter loop, and across batches a request that just missed batch
K waits for batch K+1 — so a DEALER that submitted A then B can legitimately receive B's reply
before A's if A and B landed in different drains and the scatter order differs. A pure positional
FIFO (`the i-th reply is the i-th submit`) is therefore **not** guaranteed by ROUTER↔DEALER per-peer
ordering the way the §3.4 lean reads it: per-peer ordering guarantees the *frames* arrive in the
order the ROUTER *sent* them, not that the ROUTER *sent* them in the order the DEALER *submitted*
them, once the greedy drain reorders within/across batches. The §3.4 note half-acknowledges this
("Only if replies could ever arrive out of order... add an explicit echoed u32 request-id") but
files it as a can't-happen; the greedy drain makes it a can-happen.

**The mechanism, therefore:** an **explicit echoed `u64` corr_id** carried in the frame, matched at
`poll()`. This is a **real codec amendment** — a new field in the `wire_spec` SSOT, derived on both
sides (P7: the C++ mirror header and the Python `wire_spec.py` both derive the new field; the
`tests/test_wire_drift.py` net covers it; the server echoes the request's corr_id into its reply).
The routing then keys on the echoed id, not on arrival position, so a reordered drain cannot
mis-route. Fail-loud: a `poll()` that returns a corr_id the driver has no parked tree for is a typed
Error (a desync — never a silent drop), and a parked tree whose corr_id never returns within
poll_timeout is a per-tree timeout routed as that tree's failure (§5).

**The cost, named:** this is a wire amendment, so it touches the P7 SSOT and the drift net, and the
server must echo the field — it is **not** "no wire change" as §3.4 hoped. The trade is correctness
(a reorder-proof router) for a one-field codec bump. This spec recommends paying it: relying on a
positional FIFO under a *greedy-drain* server is precisely the kind of "it can't reorder" assumption
that a near-tie float32 hazard's worth of debugging teaches you to distrust, and the echoed id makes
the routing robust to any future server-side batching change for one field's cost. **Open question
(§8.1):** confirm whether the maintainer prefers the echoed-id codec bump now vs. constraining the
server to a deterministic per-peer reply order (a barrier drain) that would preserve the positional
FIFO — the latter keeps the wire frozen but couples the routing to a server discipline, which is the
more brittle coupling.

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
  the routing nearly free (the raise/Error unwinds the one thread that owns that tree).

- **`FiberMuxRuntime` (Embodiment 2):** the completion loop's `poll()` returns either a
  `Completion{corr_id, prediction}` (→ look up the parked tree, `resume(prediction)`) or an Error.
  An Error from `poll()` that carries a corr_id (a malformed reply for a known request) routes to
  *that* tree as its failure. An Error without a corr_id (a transport/poll failure) is ambiguous —
  it cannot be attributed to one tree — and aborts the batch. A parked tree whose corr_id has not
  returned within its deadline is a **per-tree timeout** (the driver tracks a submit timestamp per
  outstanding corr_id) and routes as that tree's failure. The one real new failure mode the
  corr_id-routing introduces — a desync where a reply's corr_id matches no parked tree — is itself a
  typed Error (a loud abort), **never** allowed to silently slide the router (§4.1).

**The invariant the failure path must not break:** a failed tree's leaf is never substituted with a
zero or stale value to "keep going." That is the silent-failure ADR-0002 forbids and would corrupt
the aggregate-equivalence bar (§7 layer 4) invisibly. Fail the tree (and, under the §1 whole-batch
contract, the batch), loudly, with the diagnostic the `Error` carries.

---

## 6. The two in-flight caps

`cpp-batched-search.md` §3.3 names two caps; both are load-bearing and they live at different layers.

- **Cap (a) — per-tree in-flight == 1, structural.** This is the §1.2 exactness mechanism and it is
  enforced **by the `TreeSearch` interface shape**, not by a runtime check: the only way to get the
  next leaf is to `resume()` the previous one, so a tree physically cannot have two leaves
  outstanding (§3). No driver can violate it by accident — the type system holds it. This is the
  defining invariant of the whole Axis-A regime (the moment a second leaf issues from one tree
  before backprop, it is virtual-loss / Axis C, `cpp-batched-search.md` §2.1) and making it
  structural rather than a remembered assertion is the chief safety dividend of Option B.

- **Cap (b) — global concurrency, sized vs the host wall.** The number of trees with a leaf in
  flight at once. This sizes the server's achieved batch `B` and must be large enough that the
  greedy drain stays near-full (`B ≈ #trees parked at a leaf per drain`), but it is bounded by the
  **~1.9× host-contention ceiling on the 4-vCPU VM** (CLAUDE.md), not by the cap — over-provisioning
  just holds more `_Node` heaps for no throughput. Concretely:
    - `ThreadPerTreeRuntime`: cap (b) **is** the thread count N. The §3.4 caveat ("acceptable only
      when threads ≫ cores") means N must exceed cores for the IO-blocked threads to overlap and
      keep `B > 1` — but each thread is a real OS thread on a 4-vCPU host, so N is bounded by the
      memory of N parked `_Node` heaps and the scheduler overhead of N ≫ 4 threads. This is the
      embodiment whose cap is the *least* comfortable on this host, which is exactly why the
      benchmark exists.
    - `FiberMuxRuntime`: cap (b) is the number of parked continuations K on the multiplexer thread —
      many K per OS thread, so K can be large for little thread-scheduling cost; the bound is again
      memory (K `_Node` heaps) and the point of diminishing return where `B` already saturates
      `max_batch`.
    - `SerialRuntime`: cap (b) == 1 by definition (one tree at a time); `B` is always 1. It exists
      to measure the no-fan-out floor.

  The harness sweeps cap (b) (§7) — that sweep *is* the measurement of where the host wall, not the
  cap, becomes the binding constraint.

---

## 7. Benchmark harness and parity plan

### 7.1 What makes the Emb1-vs-Emb2 benchmark FAIR (and what would secretly rig it)

The benchmark compares **scheduling**, so everything below the scheduler must be **identical and
held fixed** across the two timed runs:

- **Same search, same `TreeSearch`.** Both drivers drive the identical advance/resume state machine
  (§3). This is the whole reason for Option B — if Emb1 ran a synchronous search and Emb2 a
  continuation one, a throughput delta would confound scheduling with two different searches.
- **Same scenario, captured and reproducible (ADR-0009).** A fixed corpus of `SearchTask`s — a
  pinned set of `(env, loc, bw, collected, λ, seed, cfg)` tuples captured to a file (the
  `states.npz` analog for the C++ runtime), so the run is reproducible and not dependent on ambient
  state. Same belief widths (full C(N,K) down to singleton), same λ, same budget (`m=12, n_sims=48`,
  the gumbel_search defaults), same matched seeds.
- **Same server, same params, same `max_batch`.** One `InferenceServer` instance with
  `StaticParamsSource` (no redis, no version churn — net-version-per-decision held constant so it
  does not confound, §8.3) and the same `forward_core` weights, so both embodiments hit the same
  greedy-drain ROUTER. The server must **not** be restarted between the two runs (warm XLA).
- **Same host pinning.** `--cores 0,1,2,3` on the 4-vCPU VM for both.

**What would secretly advantage one side (name it to forbid it):**
- Giving Embodiment 1 more threads than Embodiment 2 has parked continuations — that is comparing
  different cap-(b) values, not different schedulers. **Both must be swept over matched cap-(b)
  points** (Emb1's N == Emb2's K at each point), and the *frontier* compared, not one point each.
- Letting Embodiment 2's single multiplexer thread monopolize a core while Embodiment 1's blocked
  threads are descheduled — the host-pinning + the cap-(b) sweep control for this, but the harness
  must report **core utilization** so a "win" that is really one side under-subscribing the host is
  visible.
- A cold server for one run and a warm one for the other (XLA compile on the first batch). Warm both
  before timing.
- Different RNG seeds per side — matched seeds, so the per-tree trajectories (which determine how
  many leaves each tree issues, hence the offered load) are identical inputs to both schedulers.

### 7.2 The metric set

Per the ADR-0009 metric vocabulary, extended for this scheduling axis:

- **Throughput — decisions/s** (the headline comparable), over the fixed task corpus, wall-clock,
  warmed, ≥3 repetitions with reported variance.
- **Achieved server batch size `B`** (mean and distribution) — the server reports its per-drain
  batch size; this is the *mechanism* by which a scheduler wins (a scheduler that keeps `B` near
  `max_batch` hides more latency). A throughput win with no `B` increase is suspicious and must be
  explained.
- **Core utilization vs the ~1.9× ceiling** — the four-vCPU wall is the real bound (CLAUDE.md); the
  harness reports achieved parallel speedup so "Emb2 is faster" is read against "but neither beats
  1.9×."
- **Leaf-request count per decision** (from `Decision::leaf_requests`) — an invariant cross-check:
  for matched seeds and canned-equivalent leaves the *structural* request count must be identical
  across all three runtimes (a different count means the scheduling changed the search — a bug, not
  a speedup).
- **Cap-(b) sweep frontier** — throughput vs cap (b) for each runtime, so the comparison is curves,
  not points (§7.1).

### 7.3 The parity bar — aggregate, not per-decision

This is the load-bearing fidelity subtlety and the spec states it plainly. Because of
**batch-composition roundoff** (`cpp-batched-search.md` §2.2 / `zmq-inference-service.md` §4): *which*
other trees co-batch a given tree's leaf depends on arrival timing under the greedy drain, so the low
bits of that leaf's value depend on its batch neighbours, and at a near-tie (§1.3) a ≤1e-4
perturbation can legitimately flip which SH survivor or PUCT child is chosen. **So per-decision
results legitimately differ across runtimes** (Serial vs ThreadPerTree vs FiberMux schedule
different batch compositions) — comparing decisions byte-for-byte across runtimes would red on a
*correct* implementation. The bar is therefore **aggregate behavioral equivalence** (ADR-0009 /
P6 tier 2): over **N≥300 decisions across ≥2 seeds**, the action distribution and improved-π
statistics must be statistically indistinguishable across the three runtimes within Monte-Carlo CI,
with the MC standard error reported.

The parity/test plan, in the four composing layers `cpp-batched-search.md` §5 names, instanced for
this seam:

1. **Net-forward parity (inherited, unchanged).** The wire path is already pinned at max|Δ| < 1e-4
   (`tests/test_zmq_net_cpp.py`, measured ~e-7). The runtimes do not touch it.

2. **Single-tree structural-determinism test (the advance/resume test — Option B's dividend).**
   Drive **one** `TreeSearch` with a **recording stub** that returns **canned, byte-identical**
   `NetPrediction`s, and assert that the **sequence of `NeedsLeaf` requests** (the `(loc, bw,
   collected, λ)` leaf states, in order) and the final `Decision` are **identical** to the
   in-process serial Gumbel reference fed the same canned leaves. This isolates *search structure*
   ("which leaf is requested next") from leaf numerics and **directly proves the continuation
   refactor did not reorder the per-tree RNG stream or perturb the descent** (§3's hard part). It is
   the same test for all three runtimes (canned leaves remove batch-composition variation), so it
   also proves the three drivers drive the *same* search. This test seam **is** the `resume()`
   interface — the chief reason Option B was chosen (§3.2).

3. **The three Danihelka invariants, per-decision, unchanged.**
   `test_executed_action_is_sh_survivor`, `test_vmix_prior_weighted`,
   `test_sequential_halving_spends_full_budget` (`cpp-batched-search.md` §1.1/§5). Each tree runs its
   own private SH budget regardless of how it is scheduled, so these must still hold under every
   runtime; a failure means a tree's budget got coupled across the scheduler — i.e. it slid into
   Axis C. The continuation refactor (§3) is the most likely place to break these (the SH
   bookkeeping becomes explicit reentrant state), so they re-run against the wrapped search.

4. **Aggregate behavioral equivalence (the cross-runtime bar).** The N≥300 / ≥2-seed
   action-distribution + improved-π comparison across {Serial, ThreadPerTree, FiberMux} within MC CI
   (above). Plus a **batch-composition stress test**: vary cap (b) and inject arrival jitter, and
   assert the aggregate stays inside CI — pinning that batch-composition roundoff stays inside the
   P6 envelope and does not, in aggregate, change the policy.

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
FIFO, because the greedy drain can reorder per-peer replies. The open decision for the maintainer: pay
the one-field codec bump (robust, couples nothing to server timing), or constrain the server to a
barrier/deterministic drain that preserves the positional FIFO (wire frozen, but the routing now
depends on a server discipline — the more brittle coupling). This spec leans echoed-id; not silently
resolved.

### 8.2 Failure-batch granularity (§1)
Whole-batch-abort `expected<vector<Decision>>` (this spec's pick — loudest, simplest §5 routing) vs.
per-task `vector<expected<Decision, Error>>` (one tree's RPC failure does not lose its 63 healthy
siblings — better for a long self-play run). Promote to per-task when the self-play consumer lands, if
that is the maintainer's preference. Flagged.

### 8.3 Net-version consistency per tree-decision (inherited open question, §3.7 — NOT resolved here)
`cpp-batched-search.md` §3.7 / §6-Q1: a single tree's ~48 leaves can straddle a version reload at
batch boundaries today, perturbing that tree's distribution. The benchmark harness side-steps it by
pinning one version (`StaticParamsSource`, §7.1) so it does not confound the scheduling measurement —
**but that is a benchmark control, not a resolution.** Whether production self-play pins one frozen
net version per tree-decision (freeze weights during a generation phase) or accepts the straddle is
an open decision orthogonal to the runtime seam. Surfaced, not silently resolved.

### 8.4 Boundaries — what this spec is NOT
- **Not the Gumbel search.** Its decision math (SH, PUCT, v_mix, the float32 hazards) is the
  in-progress separate port (§2); this spec wraps it, sequences after it, and must not conflict with
  it.
- **Not within-tree leaf batching / virtual loss (Axis C).** Per-tree in-flight stays structurally 1
  (§6 cap a). The seam *cannot* express a second outstanding leaf per tree (§3) — Axis C is
  unreachable by construction, exactly as `cpp-batched-search.md` §2.1 requires.
- **Not a re-authoring of the wire.** The runtimes compose the existing `inference_wire` codec and
  the existing `NetEvaluator` port (P7); the DEALER client adds a new socket lifecycle and submit/
  poll/route, reusing the codec — except for the one echoed-id field amendment of §4.1, which is a
  P7-disciplined SSOT change, not a second hand codec.
- **Not the server.** The server stays the single-threaded greedy-drain ROUTER
  (`zmq-inference-service.md` §3/§8 — workers stay dumb, the server batches; no XLA in a worker
  thread). The runtimes are entirely worker-side.

### 8.5 Non-goals
Virtual loss; a shared tree; root parallelization; any within-tree leaf batching; moving masking
server-side; adding XLA-bearing threads to the server; resolving net-version-per-decision; building
the DEALER client / `FiberMuxRuntime` **before** the Gumbel port lands and before the §6-Q5 benchmark
justifies the multiplexer's complexity (building it speculatively is the ADR-0011 Rule-3
measure-first violation in the opposite direction — the consult's §6 symmetric warning).

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
  socket, the threads, the poll loop, the waits) lives in the `SearchRuntime` imperative shells.
  Every signature takes bounds-carrying inputs (`std::span<const SearchTask>`, `std::span<const
  float>`) and returns by value; failure is `[[nodiscard]] std::expected<…, Error>`, absence (where
  it arises) `std::optional`, never a sentinel or nullable pointer; the DEALER client and
  `TreeSearch` are RAII, move-only, `create()`-factory-constructed (a throwing ctor cannot return a
  value). The per-tree-in-flight-==1 invariant is structural (the interface shape), and a
  driver-misuse (resume of a non-awaiting tree) is an assert/abort (an invariant violation, a bug),
  not an `expected` (a boundary condition) — the P9 rule-5 distinction held.
- **ADR-0009 (measure-first).** The seam exists *to run the §6-Q5 gate*: no embodiment is preferred
  before the benchmark; `SerialRuntime` is the captured reproducible baseline; the harness is a
  pinned, reproducible scenario with the ADR-0009 metric vocabulary (decisions/s, achieved `B`, core
  utilization vs the 1.9× wall, matched seeds). Building `FiberMuxRuntime` before the benchmark
  justifies it is the measure-first violation §8.5 forbids.
- **ADR-0011 (mechanization / two-tier equivalence).** The parity bar is the two-tier ADR-0009/P6
  bar applied across runtimes: the structural-determinism layer-2 test is **bit-exact** on the
  `NeedsLeaf` request sequence (a logic invariant — the search structure under canned leaves);
  the cross-runtime layer-4 comparison is **aggregate-behavioral within MC CI** (the float-sensitive
  numerics, where batch-composition roundoff legitimately moves the float). The echoed-id wire change
  is mechanized by the existing drift net (Rule 4: a net over the class, not an enumerated instance).
  The per-tree-in-flight-==1 invariant is mechanized structurally (the interface makes the violation
  un-authorable), the strongest feasible surface (Rule 1).

---

## Amendment — 2026-06-18: the WIRE generation path lands (ADR-0005 Rule 8, dated append)

The wire-batched `--serve` GENERATION path (`docs/design/cpp-wire-generation-roadmap.md`,
`run_episodes_wire_batched`) is implemented and measured. It resolves the open questions this spec
flagged, and where it diverges from a recommended embodiment it is stated precisely (it does NOT
silently rewrite the point-in-time record above):

- **§8.1 (correlation routing) is CLOSED — via a TRANSPORT ENVELOPE, NOT the §4.1 codec-field
  embodiment.** The implementation **diverged from §4.1's recommended echoed-`u64`-corr_id codec field**
  (a `wire_spec` SSOT field covered by `test_wire_drift.py`) and instead carries the `u64` corr-id as a
  leading **ZMQ transport-envelope frame** (`wire_pool_bench.cpp` / `WireLeafPool::submit`), round-tripped
  **opaquely** by the server (`inference_server.py`, `frames[1:-1]` — never parsed, never decoded). This is
  strictly better on P7 grounds (zero value-codec surface; `wire_spec`, `inference_wire`, and
  `test_wire_drift.py` are UNCHANGED) — serialization⊥transport (P7). Read this as "closed §8.1 with a
  transport envelope, diverging from §4.1's recommended codec-field," NOT "reached §4.1" (it didn't) nor
  "supersedes §4.1" (imprecise).
- **§8.2 (failure-batch granularity): whole-generate-abort taken; per-task DEFERRED.** A failed/timed-out/
  unknown-corr-id leaf sets a shared `failed` flag and the whole pass returns `std::unexpected` ->
  `ERR_GENERATE_FAILED` (never a partial write, never a zero/stale leaf — ADR-0002). This keeps the
  executor's written-vs-read reconciliation all-or-nothing. The per-task promotion stays flagged.
- **§8.3 (net-version straddle): tightened, NOT a benchmark control anymore.** Production uses a live
  `RedisParamsSource` (version-gated, not the `StaticParamsSource` benchmark control); the actor's generate
  is lock-step on the control channel and the server reloads only between generates, so there is one reload
  boundary per generate and no in-flight straddle. The weights are published **publish-THEN-bump** (the
  blob is written FIRST, then the version the server's reload poll wants advances) — the reverse order opens
  a missing-blob reload-abort window. The residual hazard is weight-blob LRU eviction on the 6380 transport
  instance under T×K pressure (roadmap OR-3), watched at the measurement.
- **Option A (stackful fiber + `YieldingNetEvaluator`) is the realized leaf-park mechanism**; the §3.2
  Option-B continuation refactor was NOT taken (the search core `run_search` is byte-untouched — the fiber
  changes only WHEN predict returns, not WHAT). Layers 1-3 stay exact; layer 4 is aggregate-within-MC-CI.
- **Server standup lives in `cpp_executor.py`** (an in-process JAX `InferenceServer` daemon thread over the
  live net on an `ipc://` endpoint, built in `_ensure_actor`, torn down in `close()`); eval stays in-process
  Python (ADR-0008) — the server serves GENERATION only.
- **Pool knobs ride `--serve` STARTUP args, not `ActorConfig`** (their ONE home is `RuntimeConfig` — P1).
  The §8.x assumption that the runtime knobs could ride a HOT `ActorConfig` field is superseded: `--infer-
  endpoint`/`--pool-threads`/`--pool-batch` are startup args; the config_epoch gate and the 7 HOT search
  knobs (`m/n_sims/c_*`) ride `ActorConfig` exactly as before. Online-K retuning stays deferred (ADR-0009).
- **The layer-2 structural cross-check uses the PRODUCTION RNG `TreeState` ctor** (the arm the wire driver
  runs), NOT `fiber_proto`'s scripted `CyclicGumbelSource` arm; the driver-side per-decision leaf count is
  the structural discriminator (the wire driver owns the submit loop and counts directly — there is no
  `leaf_requests` field on `GumbelAZPolicy::Decision`).

---

*Public Domain (The Unlicense).*
