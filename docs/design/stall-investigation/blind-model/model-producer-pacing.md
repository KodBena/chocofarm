# A faithful model of the C++ producer side of the leaf-evaluation transport boundary

**Scope of this document.** The PRODUCER side: the search source, the per-thread DEALER
`WireLeafPool`, and the pipelined driver `run_episodes_wire_pipelined` (with its strict-barrier
sibling `run_episodes_wire_batched`). The model is derived FORWARD from the operational semantics
of the code; the peer (the Python `InferenceServer`) appears only through the RELY assumptions,
each checkable against `chocofarm/az/inference_server.py`.

All claims below are grounded in files read end to end:

- `cpp/include/chocofarm/wire_leaf_pool.hpp` (the DEALER wrapper; socket options; corr-id map)
- `cpp/src/runner_wire_batched.cpp` (both drivers; slot lifecycle; `issue_one`; the D-pipeline)
- `cpp/include/chocofarm/inference_wire.hpp` (the byte codec; transport-free)
- `cpp/include/chocofarm/wire_spec.hpp` (the layout SSOT mirror; `HEADER_BYTES=9`, f32, u32, u8)
- `cpp/include/chocofarm/fiber_tree.hpp` (`TreeState::start/resume_with/running`)
- `cpp/include/chocofarm/fiber_leaf.hpp` (`FiberLeafChannel`, `YieldingNetEvaluator::predict`)
- `cpp/include/chocofarm/runner_wire_batched.hpp` (`WireRunnerConfig`, `WireMode`)
- `cpp/include/chocofarm/runtime_config.hpp` (T, K = `fibers_per_thread`)
- `chocofarm/az/inference_server.py` (the peer — RELY grounding)
- `docs/design/zmq-inference-service.md` (design intent)

---

## 0. The boundary in one paragraph (the operational frame)

The producer is `T` independent OS worker threads (`run_episodes_wire_pipelined`,
runner_wire_batched.cpp:420, `threads.emplace_back(worker, t)` at :604). Each thread owns
`K = ceil(pool_batch / pool_threads)` `EpisodeSlot`s (`fibers_per_thread()`,
runtime_config.hpp:29; `slots(K)` at :432), its OWN DEALER `WireLeafPool` (:421-423), its OWN
`FeatureBuilder` and its OWN `RedisClient`. Each slot holds at most one `TreeState` — a Gumbel-AZ
search running inside a boost.context stackful fiber that PARKS at each leaf
(fiber_tree.hpp:42-108). A slot becomes *eligible* (`running == true`, parked at a leaf with a
feature row in `ts->ch.features`) when its search reaches a leaf needing a net evaluation; the
DRIVER gathers eligible-and-unsubmitted slots into coalesced DEALER messages, holds up to `D`
messages outstanding, and on each reply routes predictions back to their slots by correlation id,
resumes those fibers, and refills. The whole interface is: WHEN a slot becomes eligible (the
search's own progress — nondeterministic), HOW MANY are eligible at the instant `issue_one` runs
(coalescing degree S), and HOW the D-deep pipeline interleaves outstanding messages with
out-of-order replies.

Two drivers exist on the same per-slot state machine; only the OUTER drain differs:

- `run_episodes_wire_batched` (StrictBarrier, the production default): D=1 structurally —
  gather ALL parked → one submit → block on the one reply → resume all. Lines 293-337.
- `run_episodes_wire_pipelined` (PipelinedBucket, arm 3, behind `wcfg.mode`): up to
  `D = max(1, wcfg.max_inflight_msgs)` coalesced messages outstanding; resume per reply
  (out of order by corr-id), refill to D. Lines 377-627. Dispatch at :66-67.

The pipelined driver is the focus (the strict barrier is its D=1 specialization, made explicit
in §7-A).

---

## 1. The ZeroMQ socket facts (determined from code, not assumed)

These fix the BLOCKING behavior of the boundary and so must be read off the code, not the OS
defaults in general.

| option | value | code_ref | consequence |
| --- | --- | --- | --- |
| socket type | `ZMQ_DEALER` | wire_leaf_pool.hpp:77 | async, round-robins sends, fair-queues recvs; no REQ lockstep |
| `ZMQ_LINGER` | `0` | wire_leaf_pool.hpp:81-82 | close/dtor discards unsent frames immediately; no shutdown stall |
| `ZMQ_RCVTIMEO` | `timeout_ms` (`wcfg.timeout_ms`, default 15000) | wire_leaf_pool.hpp:83; WireRunnerConfig default :69 | `zmq_msg_recv` (recv_corr_payload :217) returns `EAGAIN` after this many ms → a LOUD recv-timeout Error, NOT a hang |
| `ZMQ_SNDHWM` | OS default (1000) — **not set** | absence in create() :75-91 | a `zmq_send` (submit_batch :139,:142) blocks only if 1000 messages are queued unsent; in practice never reached (D ≤ a few; the server drains) |
| `ZMQ_SNDTIMEO` | OS default (-1, block) — **not set** | absence in create() | a send blocks if HWM is hit; with HWM unreached it returns immediately |
| `ZMQ_RCVHWM` | OS default (1000) — **not set** | absence in create() | inbound queue bound; not reached at these depths |
| context-level options | none set | runner_wire_batched.cpp `zmq_ctx_new()` :394, no `zmq_ctx_set` | default I/O threads (1) |

Connect is **lazy** (wire_leaf_pool.hpp:67-70): `zmq_connect` over a not-yet-bound `ipc://`
endpoint does NOT fail; a dead endpoint surfaces only at the first `recv` after `RCVTIMEO`, as a
loud timeout. So the producer can `submit_batch` (which only enqueues into the DEALER's outgoing
pipe) before the server has bound, and the failure mode is "the recv times out," never "the send
errors at connect time."

**Load-bearing consequence.** The ONLY blocking call in the producer's hot loop is the recv inside
`recv_batch` (`pool.recv_batch()` at :580 → `recv_corr_payload` → `zmq_msg_recv` at :217). `submit_batch`
is effectively non-blocking (HWM never reached). So in the pipelined driver a thread blocks at most
ONE place — `recv_batch` — and only when it has chosen to (when `inflight_msgs > 0` and it has
issued all it can). This is the timing knob the model must represent faithfully (§3).

---

## 2. The state machine

### 2.1 Per-slot states (one slot, per thread)

The slot is single-writer-per-thread (no migration — wire_leaf_pool.hpp:23-27; the corr-id atomic
is process-global but the inflight map is per-pool). So each slot is an independent automaton; a
thread runs `K` of them concurrently (cooperatively, via fibers) plus a `D`-deep message pipeline.

| state | meaning | code predicate |
| --- | --- | --- |
| `IDLE` | slot has no active episode (subset exhausted) | `!sl.active` (after `fill` returns false) |
| `PARKED` | active, parked at a leaf, NOT outstanding — *eligible to send* | `sl.active && sl.ts && sl.ts->running && !submitted[s]` = `is_ready(s)` (:541-544) |
| `OUTSTANDING` | its leaf is in a sent message awaiting a reply | `submitted[s] == 1` (set at :564, cleared at :588) |
| `ADVANCING` | reply arrived; fiber resumed; running the search/episode logic between leaves | transient: inside the `for (Completion c : *reply)` body :584-594 |
| `FINALIZED` | the episode ended this ply (TERMINATE / horizon / empty belief); slot will refill | `apply_decision` returned false → `finalize_and_write` ran (:484-499) |

A `TreeState` sub-automaton lives inside the slot (fiber_tree.hpp):

- `start()` builds the fiber and `resume()`s it to the first leaf-yield or to finish; sets
  `running = ch.at_leaf` (:98-99). The yield happens inside `YieldingNetEvaluator::predict`
  (fiber_leaf.hpp:45-50): it writes `ch.features = x`, sets `ch.at_leaf = true`, and resumes the
  caller. So "PARKED with a feature row ready" is exactly `ts->running == true` with
  `ts->ch.features` valid.
- `resume_with(pred)` writes `ch.value = pred` and `resume()`s; the search's `predict` returns
  `ch.value` (fiber_leaf.hpp:49) and runs on, either to the next leaf (`running` stays true) or to
  the Decision (`ch.at_leaf=false` at fiber_tree.hpp:96 → `running=false`).

### 2.2 Transitions (each with guard, action, code_ref, free/determined)

Let `s` be a slot index in `[0, K)` of thread `tid`.

| # | from → to | guard | action | code_ref | free? |
| --- | --- | --- | --- | --- | --- |
| T1 | (none) → PARKED or FINALIZED | priming, `s < K` | `fill(s)`: seed `fold_seed(seed, idx)`, world draw, `spawn_ply`; if it parks → PARKED, else `advance` chains, else next idx; degenerate empty-belief → finalize & retry | :572 calls fill :511-538 | **determined** given the idx and RNG (the search's draws are RNG-exact); the SUCCESSION of which idx lands in which slot is determined by `next_idx = tid; +=T` (:514-515) |
| T2 | PARKED → OUTSTANDING | `is_ready(s)` AND this slot is gathered by the current `issue_one` AND `inflight_msgs < D` | `gather` its `ts->ch.features` into the coalesced row block; after submit, `submitted[s]=1` | issue_one :551-569 (gather :554-560, mark :564) | **partly free**: WHETHER this slot is included is determined (every ready slot at the gather instant is included, :555); but WHICH slots are ready *at that instant* is the free source-timing choice (§3) |
| T3 | (eligible set) → one message sent | `gathered` non-empty AND `inflight_msgs < D` | `pool.submit_batch(gathered, gather, in_dim)` → stamp one corr-id, two-frame send `[corr][payload]`; `++inflight_msgs` | issue_one :562-568; submit_batch wire_leaf_pool.hpp:129-147 | **free in S**: S = `gathered.size()` = #ready at that instant (1..K). The number is nondeterministic |
| T4 | OUTSTANDING → ADVANCING | a reply with THIS message's corr-id arrives at the recv | `recv_batch()` decodes; `--inflight_msgs`; for each Completion: `submitted[s]=0`, `ts->resume_with(c.pred)` | drain :580-589; recv_batch wire_leaf_pool.hpp:170-196 | **free in ARRIVAL ORDER**: which outstanding message's reply lands first is the peer's scheduling choice (RELY R3); routed by corr-id, so out-of-order is correct |
| T5a | ADVANCING → PARKED | `ts->running` after resume (search hit its NEXT leaf) | re-park: slot rejoins the ready set on the next `issue_one` | drain :590 `continue` | determined by the search (RNG-exact) |
| T5b | ADVANCING → PARKED (down a chain) | `!ts->running` after resume, `advance(s)` returns true | `advance`: `apply_decision` (record/step/`env.apply`), `spawn_ply`, loop until parked or finalized | drain :591 `advance(s)`; advance :502-510 | determined by the search + env |
| T5c | ADVANCING → FINALIZED → (refill) | `!ts->running`, `advance(s)` returns false (episode ended) | `fill(s)`: start the next episode in the subset (→ PARKED) or IDLE if subset exhausted | drain :593 `fill(s)` | determined by the episode logic |
| T6 | (drain) refill to D | `inflight_msgs < D` AND `issue_one()` finds ≥1 ready | issue another coalesced message | drain :596 `while (inflight_msgs < D && !failed && issue_one()){}` | **free**: how many of the resumed/re-parked slots are ready NOW vs still ADVANCING determines S of the refill messages |
| T7 | any → ABORT | a recv/decode/corr-id/count error, OR a submit error, OR a redis write error (`failed` set) | `set_error`, break; whole pass returns `std::unexpected` | drain :581 `set_error`; issue_one :563; finalize_and_write :474 | determined (it is the loud-failure arm, ADR-0002) |

Priming and main loop (the OUTER structure, runner_wire_batched.cpp):

```
for s in [0,K): fill(s)                          # :572  T1 — park K slots (or fewer if subset small)
while inflight_msgs < D and issue_one(): pass    # :578  T3 — prime the pipe to depth D
while inflight_msgs > 0 and !failed:             # :579
    reply = recv_batch()                         # :580  T4  (BLOCKS up to RCVTIMEO)
    --inflight_msgs                               # :582
    for c in reply:                              # :584
        submitted[c.slot] = 0                    # :588
        ts.resume_with(c.pred)                   # :589  T4 cont.
        if ts.running: continue                  # :590  T5a
        if advance(s): continue                  # :591  T5b
        if failed: break
        fill(s)                                  # :593  T5c
    while inflight_msgs < D and issue_one(): pass # :596  T6 refill
```

The loop terminates when `inflight_msgs == 0` — i.e. no message outstanding AND none could be
issued (every slot is IDLE: the thread's whole subset is consumed). Because `issue_one` returns
false when nothing is ready (:561) and `recv_batch` strictly decrements `inflight_msgs`, the loop
makes monotone progress to drain.

---

## 3. The timing model (the heart of faithfulness)

Two clocks are nondeterministic; the model represents each as bounded nondeterminism and pins the
causal constraints only.

### 3.1 SOURCE-emission timing — when a slot next becomes PARKED

A slot becomes PARKED at the instant the search's `predict` is called inside the fiber
(fiber_leaf.hpp:46-48). WHEN that happens is set by the search's own internal work between leaves —
expansions, the Sequential-Halving schedule, env stepping in `advance`/`apply_decision` — none of
which this code fixes. The driver never sets a leaf interval; it only *reacts* to `ts->running`.

**Representation.** For each slot `s`, after it is resumed at logical step `r`, the time until it
next becomes PARKED (or FINALIZED) is a nondeterministic positive interval `δ_s^r ∈ (0, ∞)`. Constraints:

- **Positivity.** `δ_s^r > 0`. (A resume runs real fiber code; not instantaneous.) Code: every
  resume is `fib = std::move(fib).resume()` (fiber_tree.hpp:106) — real work.
- **Reply-dependence.** A slot cannot become PARKED at its *next* leaf before its *current* leaf's
  reply has resumed it: the fiber is suspended inside `predict` (fiber_leaf.hpp:48) until
  `resume_with` runs (fiber_tree.hpp:103-107). So the (r+1)-th PARKED of slot s causally follows
  the r-th reply for slot s. This is the single most important producer causal constraint: **a
  reply-dependent next request cannot precede its reply.**
- **No upper bound from the producer.** The code imposes no max think-time; `δ_s^r` may be
  arbitrarily large (a long search ply). The ONLY thing that bounds the *observable consequence* is
  RCVTIMEO on the recv side — but that bounds the SERVER's latency, not the producer's think-time.
- **Independence across slots is NOT assumed.** Slots on one thread are cooperatively scheduled on
  one OS thread: only one fiber runs at a time (boost.context is cooperative; the resume loop
  :584-594 resumes them one at a time). So within a thread the `δ`'s are not simultaneous wall-clock
  intervals but *interleaved* segments of one CPU. The model treats "slot s becomes PARKED" as an
  event whose ORDER relative to other slots' events is free, subject only to the per-slot
  reply-dependence and the single-runner serialization within a thread.

**Why this must NOT be collapsed to a constant (DOF-2).** If `δ` were a fixed constant, the
coalescing degree S (§3.3) would be deterministic — every slot would arrive in lockstep and S would
be pinned. The real code lets searches finish leaves at data-dependent times, so the count ready at
any `issue_one` instant ranges over `1..K`. Collapsing `δ` forbids the real executions where S
varies. **Not collapsed.**

### 3.2 SINK-service timing — the forward (modeled as a RELY, observed at the recv)

The producer does not run the forward; it WAITS for it at `recv_batch`. But the model must
represent the service time faithfully because the transport behavior is a function of it. From
`inference_server.py` (read end to end):

- The server is **single-threaded** (class docstring :291-300; comment :35) — **ONE forward at a
  time**. While batch K's forward runs, requests for K+1 queue in the ROUTER (`serve_forever`
  docstring :428-432). This serializes service: a reply cannot be produced before the forward that
  produced it completes, and forwards do not overlap.
- The forward time **depends on batch size only through padding**: `run_microbatch` pads every
  batch UP to `pad_to = self._max_batch` (:171-172, :385) so XLA compiles ONE executable
  (docstring :166-176; warmup :389-426). **Consequence: the service time is, to first order,
  INDEPENDENT of the instantaneous batch B (always the padded `(max_batch, in_dim)` shape) once
  warmed.** Before warmup, a cold B triggers a per-B JIT compile (warmup docstring :393-400) — a
  large one-time latency. The model represents service time as a nondeterministic positive interval
  `σ_k ∈ [σ_min, σ_max]` per forward k, with `σ_max` allowing a cold-compile spike on the first few
  forwards and `σ_min > 0` always.
- The **drain** that forms a forward's batch (`_drain` :322-363) blocks for ≥1 request then drains
  ALL currently-queued up to `max_batch` rows (`total_rows < max_batch`, :348). So one forward
  coalesces across the producer's D outstanding messages AND across threads. The producer cannot
  observe this directly; it only observes that a reply for a given corr-id eventually arrives.

**Causal constraints linking the two clocks:**

1. `reply(corr) ` time `>` `forward_complete(k)` time `>` `forward_start(k)` time `>` `submit(corr)`
   time, where corr's rows were drained into forward k. (A reply cannot precede the forward that
   produced it; the forward cannot start before its rows arrived.)
2. Forwards are totally ordered (single-threaded server): `forward_complete(k) ≤
   forward_start(k+1)`.
3. `submit(corr)` (T3) precedes `recv(corr)` (T4) on the producer's own DEALER (the round-trip).
4. Per-slot reply-dependence (§3.1): `submit` of slot s's (r+1)-th leaf `>` `recv` of its r-th.

The model leaves `σ_k` and `δ_s^r` free within these constraints; it pins NOTHING to a constant
(no instantaneous forward, no zero think-time, no fixed batch-to-time map). **The one place the
code DOES make the service time batch-independent is the padding** (run_microbatch :171-172) — and
the model honors that by making `σ` not a function of B, which is a *code-justified* removal of one
degree of freedom (it would be UNFAITHFUL to make `σ` grow with B, because the padded shape is
constant). This is the only timing the code justifies collapsing, and it is collapsed to
"independent of B," not to a constant value.

### 3.3 Coalescing degree S (a derived nondeterministic quantity)

`S = gathered.size()` at the instant `issue_one` runs (:558). It equals the number of slots that
are `is_ready` — PARKED and not OUTSTANDING — at that instant. Because the source-timing (§3.1)
makes "which slots are PARKED now" nondeterministic, S ranges over `1..K` per message
(issue_one returns false if 0, :561). The driver's `my_leaves += gathered.size()`, `++my_msgs`
(:566-567) and the trailing `mean_rows_per_msg` telemetry (:620-624) are exactly the observable
of this nondeterminism. The strict-barrier driver has the same S but only ever issues when ALL
parked are gathered (it gathers every running slot each round, :313-320), so its S = #parked-this-round.

---

## 4. Assume-Guarantee contract

### RELY (what the producer assumes about the peer, each checkable against inference_server.py)

- **R1 — opaque corr-id round-trip.** The server echoes the transport envelope (`frames[1:-1]`)
  verbatim, never parsing it. So the producer's leading 8-byte corr-id frame returns unchanged.
  Check: `_drain` captures `envelope = frames[1:-1]` (:354), `_serve_batch` sends
  `[ident, *envelope, resp]` (:387). The producer relies on this to route replies (recv_batch matches
  `inflight_.find(corr)`, wire_leaf_pool.hpp:179).
- **R2 — one batched reply per request, B-exact.** For a request carrying `B_i` rows, the reply
  carries exactly `B_i` predictions in the SAME row order. Check: `run_microbatch` splits by
  per-request `counts` and `encode_response(v_rows, l_rows)` per identity (:184-189); the
  scatter is in drained order. The producer relies on this: `recv_batch` errors if
  `decoded->size() != slots.size()` (wire_leaf_pool.hpp:185-188) and scatters in submit order
  (:190-194). If the server ever returned a different count, the producer aborts loudly (it does
  NOT silently mis-apply).
- **R3 — replies may arrive in any order across the producer's outstanding messages.** The server
  drains whatever is queued; which of the producer's D messages lands in which forward, and which
  forward completes first, is the server's scheduling. The producer relies only on per-corr-id
  matching, not on FIFO. Check: ROUTER fair-queues; `_drain` is greedy and order-of-arrival
  dependent (:348-362). The pipelined driver explicitly tolerates out-of-order (header comment
  runner_wire_batched.hpp:88-90; corr-id routing).
- **R4 — bounded service, eventual reply.** Every well-formed request eventually gets a reply
  (the server loops forever, `serve_forever` :436-439), within a time the producer bounds with
  RCVTIMEO. The producer relies on this for liveness: if no reply arrives within `timeout_ms`, the
  recv returns EAGAIN and the producer ABORTS loudly (wire_leaf_pool.hpp:217-221) rather than
  hanging. (A malformed request is *dropped* server-side with a log, `_reject` :365-370 — so the
  producer's safety net for a dropped request is also the RCVTIMEO.)
- **R5 — wire codec agreement.** The server decodes the v2 batched frame
  `[ver=2][B][in_dim][f32×B·in_dim]` and encodes `[ver=2][B][n_actions][B×(value,logits)]`. Check:
  `decode_request`/`encode_response` (inference_wire.py) derive from `wire_spec.py`, drift-checked
  against the C++ mirror (`wire_spec.hpp` PROTOCOL_VERSION=2, HEADER_BYTES=9). The producer relies
  on this so `wire::decode_response` (inference_wire.hpp:191) reads real floats.

### GUARANTEE (what the producer guarantees to the peer, each grounded in producer code)

- **G1 — every request is a well-formed v2 batched frame.** `submit_batch` sends
  `[corr:8B][encode_request(flat,B,in_dim)]` (wire_leaf_pool.hpp:139-144). `encode_request` is
  total on well-typed input and rejects `B=0`, `in_dim=0`, or `flat.size() != B·in_dim`
  (inference_wire.hpp:100-109) — a ragged or empty batch is the typed error arm, never sent. So the
  server never sees a malformed length from a non-erroring producer.
- **G2 — unique correlation ids, process-global.** Each submit stamps
  `corr_seq_->fetch_add(1, relaxed)` (wire_leaf_pool.hpp:137) from a process-global atomic
  (runner_wire_batched.cpp:403, passed by reference). So no two outstanding requests across ALL
  threads share a corr-id; the server's opaque echo cannot alias two producers' replies.
- **G3 — a submitted slot's feature row stays alive and unmodified until its reply.** The slot is
  single-writer-per-thread; `submitted[s]` prevents re-gathering an outstanding slot
  (`is_ready` excludes `submitted[s]`, wire_leaf_pool/issue_one :543). The row lives in
  `sl.ts->ch.features` (a span into the parked fiber's stack, fiber_leaf.hpp:32) and the fiber stays
  suspended until `resume_with` — so the bytes the server reads are stable. (The producer copies
  the row into `gather` at submit time anyway, :557, so the server reads a stable copy regardless.)
- **G4 — at most D messages outstanding per thread; the producer always eventually recvs.** The
  pipeline holds `inflight_msgs ≤ D` (:578, :596) and the loop recvs whenever
  `inflight_msgs > 0` (:579-580). So the producer does not pile unbounded work on the server
  beyond D·T messages, and it always drains its own replies (it does not wedge the ROUTER's
  send queue).
- **G5 — loud abort, never a silent wrong-slot apply.** Any recv/decode/corr-id/count mismatch is
  the typed error arm (wire_leaf_pool.hpp:154-196) → `set_error` → whole-pass `std::unexpected`
  (:609-614). The producer never applies a reply to the wrong slot and never substitutes a
  zero/stale leaf (ADR-0002).

---

## 5. Degrees of freedom (each with code_ref and admitted behaviors)

- **DOF-1 — coalescing degree S per message (1..K).** issue_one gathers every ready slot at the
  gather instant (:554-560). Admits: a message of S=1 (only one slot ready) through S=K (all
  slots ready and unsubmitted). The mean is reported as `mean_rows_per_msg` (:620). Behaviors: the
  server's per-forward batch — and thus its amortization — depends on the distribution of S across
  threads.
- **DOF-2 — source think-time `δ_s^r` (positive, unbounded).** No code fixes the leaf interval
  (§3.1). Admits: lockstep arrivals (all slots park near-simultaneously → large S), staggered
  arrivals (slots park one at a time → many S=1 messages), or any mix; a slow ply (large δ) leaving
  a slot OUTSTANDING-absent while others cycle.
- **DOF-3 — reply arrival order across the D outstanding messages.** recv_batch returns whichever
  reply the ROUTER delivers next (wire_leaf_pool.hpp:170-173); the drain `for c in *reply`
  processes that message's slots (:584). Admits: replies in submit order, fully reversed, or any
  permutation — all correct via corr-id routing (RELY R3). Behaviors: a later-submitted message's
  reply can resume-and-refill before an earlier one's.
- **DOF-4 — pipeline depth D (= `max(1, wcfg.max_inflight_msgs)`, default 8).** :392. Admits:
  D=1 (the strict-barrier specialization — the search idles the whole RTT each round) up to D
  large (the search rarely idles, more messages overlap the forward). Behaviors: D bounds how many
  messages a thread keeps in flight, hence how much of the forward's latency is hidden behind
  search progress.
- **DOF-5 — slot/episode geometry T, K.** T = `max(1, pool_threads)` (:390), K =
  `ceil(pool_batch/pool_threads)` (:391, runtime_config.hpp:29). Admits: K=1 (one tree per thread,
  S always 1) up to large K (deep per-thread coalescing). T scales the number of independent
  producers feeding the one server.
- **DOF-6 — degenerate non-parking chains.** `fill`/`advance` drain a chain of plies that finish
  without parking a leaf (empty-belief guard inside the search returns without parking;
  fill :527-535, advance :502-510). Admits: a slot that advances several plies (env stepping,
  records) between two PARKED states, or an episode that finalizes without EVER parking a leaf
  (fill skips to the next idx, :513-537). Behaviors: a slot can be silent on the wire for a stretch
  while its episode logic runs locally.
- **DOF-7 — which idx lands in which slot, and subset exhaustion timing.** `next_idx = tid; +=T`
  (:514-515): the thread's subset is `{tid, tid+T, ...}` consumed in order, but WHICH slot picks up
  the next idx depends on which slot finalized first (data-dependent, DOF-2/DOF-6). Admits:
  slots draining their shared subset in any interleaving; a thread finishing while others still run.

---

## 6. Representative executions

Each is a concrete trace of genuinely-enabled transitions (code_refs in §2.2). Stability =
whether, once in this kind of execution, the system stays there (self-reinforcing) or leaves it
(transient).

### E1 — Full coalescing then a synchronized barrier (S=K, D drives overlap)

Geometry: T=1, K=4, D=2. All four slots park near-simultaneously (DOF-2 lockstep).

1. T1×4: `fill(0..3)` → all PARKED, each `is_ready`. (`fill` :572)
2. Prime to D: `issue_one()` gathers all 4 ready → S=4, one message, corr=c0, `inflight=1`,
   all `submitted`. (:578, issue_one :551-568). Second `issue_one()`: nothing ready (all
   submitted) → returns false (:561). So `inflight=1 < D` but the pipe cannot fill past 1 here.
3. `recv_batch()` blocks (T4). The server drains c0's 4 rows (padded to max_batch), one forward
   (`σ_0`), replies. recv returns 4 Completions for corr=c0. `--inflight=0`. (:580-582)
4. For each of the 4: `submitted[s]=0`, `resume_with` (T4). Say all 4 hit their next leaf
   (`running` true) → all re-PARKED (T5a, :590).
5. Refill: `issue_one()` gathers all 4 again → S=4, corr=c1, `inflight=1`. (:596) Loop to step 3.

**Exercises:** DOF-1 (S=K), DOF-2 (lockstep), DOF-4 (D unreachable because all slots submit in one
message — D>1 buys nothing when one message exhausts the ready set). **Stability:**
*self-reinforcing while the searches stay phase-locked* — if every slot takes the same number of
leaves per ply they re-synchronize each round; this is exactly the strict-barrier behavior (§7-A),
showing the pipelined driver DEGENERATES to the barrier when S=K saturates the ready set. Transient
the moment one search's δ diverges (→ E2).

### E2 — Staggered arrivals, many small messages, real pipeline overlap (S small, D bites)

Geometry: T=1, K=4, D=3. Slots park at staggered times (DOF-2): slot 0 parks first, then 1, etc.

1. T1×4 → 4 PARKED.
2. Prime to D: first `issue_one` — suppose only slots {0,1} are ready at that instant (2 still
   ADVANCING from `fill`'s internal `advance`) → S=2, corr=c0, `inflight=1`. (:578) Second
   `issue_one` — slot 2 just parked → S=1, corr=c1, `inflight=2`. Third — slot 3 parked → S=1,
   corr=c2, `inflight=3 = D`. Pipe primed with THREE messages, total 4 leaves. (DOF-1 mix)
3. `recv_batch` (T4). Out-of-order (DOF-3): c1's reply lands first. `--inflight=2`. Resume slot 2;
   it re-parks (T5a). (:584-590)
4. Refill: `inflight=2 < D`, `issue_one` gathers slot 2 (now ready) → corr=c3, `inflight=3`. (:596)
5. Next recv: c0's reply (2 slots). Resume slots 0,1; slot 0 finalizes its episode (`advance`
   returns false, T5c) → `fill(0)` starts the next idx → PARKED; slot 1 re-parks. (:591-593)
6. Refill issues over {0,1}. Loop.

**Exercises:** DOF-1 (S varies 1..2), DOF-3 (out-of-order c1 before c0), DOF-4 (D=3 keeps the pipe
busy so the thread rarely blocks with nothing outstanding), DOF-7 (slot 0 picks up a new idx).
**Stability:** *self-reinforcing* — staggered think-times keep the ready set partial, so messages
stay small and the pipeline stays full; this is the regime the pipelined driver exists for
(runner_wire_batched.hpp:88-90: "the search never idles the full RTT").

### E3 — RCVTIMEO loud abort (the dead/slow peer)

Geometry: any. The server is down (lazy connect, §1) or its forward exceeds `timeout_ms`.

1. Prime the pipe (≥1 message OUTSTANDING). (:578)
2. `recv_batch` → `recv_corr_payload` → `zmq_msg_recv` blocks; after `RCVTIMEO` it returns < 0 with
   EAGAIN. (wire_leaf_pool.hpp:217-221) → typed Error "zmq_msg_recv failed".
3. `recv_batch` returns the error (:173-174); drain `set_error(reply.error())`, `break` (:581).
4. After join, `failed` true → whole pass `std::unexpected` (:609-614). No partial write.

**Exercises:** the RCVTIMEO determinacy (§1), G5, R4 violated by the peer. **Stability:**
*terminal* (the pass ends). It is the ONLY way the producer exits a stuck recv — it never hangs
(the bounded RCVTIMEO is what makes a dead `ipc://` a loud timeout, not a deadlock).

### E4 — Degenerate non-parking episode (a slot silent on the wire)

Geometry: K≥1. An episode's first ply has an immediately-empty belief, or a search returns its
Decision without ever parking a leaf.

1. `fill(s)`: `env.empty(sl.bw)` true → `finalize_and_write` (n==0, no write), `continue` to next
   idx. (:527-531) OR `spawn_ply`; `!ts->running`; `advance(s)` drains the chain
   (`apply_decision`→`spawn_ply` loop, :502-510). If `advance` parks → PARKED; if it finalizes the
   whole episode without parking, `fill` tries the next idx (:534-535).
2. So a slot can consume several idx's, run record/step/`env.apply` logic, and produce ZERO wire
   messages before it either parks a leaf or exhausts its subset (→ IDLE).

**Exercises:** DOF-6, DOF-7. **Stability:** *transient* per occurrence (a degenerate episode is
finite), but the IDLE end-state (subset exhausted) is *terminal* for that slot — once IDLE it is
never re-filled (the drain only `fill`s after a finalize, :593; a slot that exhausted its subset in
the priming `fill` is permanently IDLE and contributes nothing to the ready set).

### E5 — Mixed-S coalescing across the refill (the typical steady state)

Geometry: T=4, K=8, D=8 (defaults-ish: pool_batch=32, pool_threads=4 → K=8; max_inflight=8). One
thread's view:

1. Prime: ready slots trickle in as `fill`'s internal advances complete; `issue_one` issues
   messages of S = {3,2,1,1,1} (5 messages) until `inflight=8` is hit OR the ready set empties.
   (DOF-1, :578)
2. Steady loop: each recv resolves one corr-id's S slots; resume → some re-park, some advance/refill,
   some finalize+fill. Refill issues over the now-ready set (mixed S). Across the 4 threads the
   server's `_drain` coalesces these into forwards of up to `max_batch=256` rows (padded), one at a
   time. (R2/R3; server :322-363)

**Exercises:** DOF-1, DOF-3, DOF-4, DOF-5, DOF-7 simultaneously — the full latitude. **Stability:**
*self-reinforcing* — this is the design's intended operating point (telemetry `mean_rows_per_msg`
and the server's rows/forward are the measured observables, :620-624).

---

## 7. DOF-control notes (the constraint that removes each latitude)

- **DOF-1 (S).** Remove by: forcing one leaf per message (submit each ready slot separately instead
  of gathering, i.e. replace issue_one's loop with a single-slot submit). Then S≡1 always;
  executions E1/E5 (S>1) become unrepresentable; the server loses cross-message coalescing within a
  thread (only cross-thread `_drain` coalescing remains).
- **DOF-2 (think-time δ).** Remove by: a synchronous, fixed-cost mock search (constant leaf
  interval) — e.g. the `CyclicGumbelSource` scripted path (fiber_tree.hpp:55-56) with a fixed
  number of leaves per ply. Then arrivals are deterministic; the staggered-arrival executions E2 and
  the mixed-S E5 collapse to the lockstep E1; S becomes a fixed function of K and the round.
- **DOF-3 (reply order).** Remove by: a FIFO/REQ socket (REQ enforces one-outstanding lockstep) OR
  a server that replies strictly in receive order. Then out-of-order E2-step-3 is unrepresentable;
  the corr-id routing becomes redundant (a positional match would suffice).
- **DOF-4 (D).** Remove by: hardwiring D=1 (the StrictBarrier driver, `wcfg.mode` default). Then no
  message overlaps the forward; the pipeline-overlap executions (E2, E5 with multiple outstanding)
  become unrepresentable; only the barrier E1/§7-A regime remains.
- **DOF-5 (T, K).** Remove by: pinning `pool_threads=1, pool_batch=1` → T=1, K=1. Then S≡1, no
  per-thread coalescing, one tree at a time; E1/E5 unrepresentable.
- **DOF-6 (non-parking chains).** Remove by: an env with no empty-belief/early-terminate guard, so
  every ply parks exactly one leaf. Then E4's silent-on-the-wire stretches are unrepresentable;
  every advance produces exactly one wire message.
- **DOF-7 (idx assignment).** Remove by: assigning a fixed idx-to-slot mapping instead of the
  shared `next_idx` pool. Then the data-dependent E2-step-5 / E5 interleaving of idx pickup is
  unrepresentable; each slot runs a predetermined idx sequence.

### 7-A. The strict-barrier specialization (the production default, for completeness)

`run_episodes_wire_batched` (runner_wire_batched.cpp:293-337) is the D=1, gather-ALL specialization:
`any_parked` → gather every running slot into ONE submit (S = #parked) → `recv_batch` (block) →
resume all → re-park/advance/fill → repeat. There is no `submitted[]` flag and no refill loop: the
thread blocks the entire round-trip each round (the comment :363-364: "the search idles the whole
round-trip each round"). It is exactly DOF-4 pinned to D=1 with S forced to #parked-this-round. Its
state machine is a strict subset of §2.2 (T2/T3 fire once per round with S=#parked; T4 always
resolves the single outstanding message; T6 never fires). Every strict-barrier execution is a
pipelined execution with D=1; the converse fails (the pipelined driver admits D>1 overlap the
barrier cannot).

---

## 8. Fidelity self-audit

### Possible over-permissions (admitting executions the code cannot produce) — checked and excluded

- **Could the model admit two outstanding messages with the SAME corr-id?** No — `corr_seq` is a
  process-global atomic incremented per submit (wire_leaf_pool.hpp:137); the model's G2 forbids
  aliasing. Excluded.
- **Could it admit a reply applied to a slot that wasn't submitted?** No — `recv_batch` looks up
  `inflight_.find(corr)` and erases it; an unknown corr-id is a loud abort (wire_leaf_pool.hpp:179-182).
  The model routes strictly by the recorded slot list. Excluded.
- **Could it admit a send blocking forever and a reply never arriving, with the thread silently
  stuck?** No — RCVTIMEO bounds the recv (§1); SNDHWM is never reached at these depths so the send
  does not block. A stuck peer is E3 (loud abort), not a silent hang. Excluded. (If SNDHWM *were*
  reached — D·T ≥ 1000 outstanding unsent — the send would block with SNDTIMEO=-1; but D·T is a few
  dozen, far below 1000, so this is not a reachable execution under the code's geometry. Noted as a
  boundary, not admitted.)
- **Could it admit S=0 messages (empty sends)?** No — issue_one returns false on empty `gathered`
  (:561); submit_batch rejects B=0 (inference_wire.hpp:100). Excluded.

### Possible over-constraints (forbidding executions the code can produce) — checked and excluded

- **Did I pin think-time or service time to a constant?** No — both are bounded-nondeterministic
  intervals (§3.1, §3.2). The ONLY collapse is "σ independent of B," which the padding code
  (run_microbatch :171-172) *justifies* — making σ grow with B would be the over-constraint here, so
  the collapse is in the faithful direction.
- **Did I forbid out-of-order replies?** No — DOF-3 explicitly admits any permutation (E2).
- **Did I forbid the degenerate non-parking and immediately-empty paths?** No — DOF-6/E4 cover them
  (fill :527-535, advance :502-510).
- **Did I assume FIFO on the DEALER?** No — DEALER fair-queues and the model relies only on corr-id
  matching (R3). A model that assumed FIFO would forbid E2; this one does not.
- **One residual asymmetry worth naming:** the model treats per-thread fibers as cooperatively
  serialized (one runs at a time) — this is FAITHFUL (boost.context is cooperative;
  the resume loop processes Completions one at a time, :584). It would be an OVER-PERMISSION to let
  two fibers of one thread run simultaneously; the model does not. Across threads, true parallelism
  IS admitted (T independent OS threads, :604), bounded by the 4-vCPU host but the code imposes no
  such bound itself, so the model leaves T-way parallelism free.

---

## 9. Code-derivation attestation

This model is derived purely from the operational semantics of the listed source files, each read
end to end. Every state, guard, transition, free choice, and timing assumption is mapped to a
specific line (or to a named causal necessity — positivity of durations, reply-after-forward,
request-after-reply, per-slot reply-dependence, single-threaded-server serialization). No outside
expectation of how the system "ought" to behave was introduced: the coalescing degree, the
think-time, the service time, the reply ordering, and the pipeline depth are left exactly as free
as the code leaves them, and pinned exactly where the code pins them (S = #ready at the gather
instant; σ independent of B because of padding; D = `max(1, max_inflight_msgs)`; corr-id routing;
RCVTIMEO-bounded recv as the sole blocking point and sole liveness backstop). The system was NOT
run; no sweep was executed. A single bounded Z3 check (below) confirms one representative execution
(E2's out-of-order pipelined interleaving) is admissible under the derived constraints — confirmation
of the theory, not its source.

*Public Domain (The Unlicense).*
