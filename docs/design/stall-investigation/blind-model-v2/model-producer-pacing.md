# Faithful Parametric Model — C++ Producer Side (N independent trees/thread + DEALER + pipelined driver)

Role: **producer** (the C++ worker side of the leaf-evaluation transport boundary).
Driver under model: `run_episodes_wire_pipelined` (`cpp/src/runner_wire_batched.cpp:270-503`),
DEALER wrapper `WireLeafPool` (`cpp/include/chocofarm/wire_leaf_pool.hpp`).

All file:line references are to the cleanroom tree
`/home/bork/w/vdc/chocobo/runs/leaf-eval-model-2/cleanroom`. Every state, transition, guard, free
choice, and timing assumption below is mapped to a code line or to a named causal necessity. Comments and
docstrings were stripped from the source; everything here is derived FORWARD from operational semantics.

The orchestrator selects this driver only when `wcfg.mode == WireMode::PipelinedBucket`
(`runner_wire_batched.cpp:44-45`); the StrictBarrier sibling `run_episodes_wire_batched` is out of focus
but referenced where it disambiguates a behavior.

---

## 0. Parameters (the model is parametric in all of these)

| Symbol | Source | Meaning |
|---|---|---|
| `T` | `pool_threads`, `runner_wire_batched.cpp:283` `T = max(1, pool_threads)` | number of C++ worker threads; each runs `worker(tid)` |
| `base` | `RuntimeConfig::fibers_per_thread()`, `runtime_config.hpp:12-15` = `max(1, ceil(max(1,pool_batch)/max(1,T)))` | per-thread fiber base (StrictBarrier's slot count) |
| `N` | `trees_per_thread`, `runner_wire_batched.cpp:285` `N = max(1, trees_per_thread)` | **trees per thread (the overcommit factor)** |
| `K` | `runner_wire_batched.cpp:286` `K = N * base` | **per-thread slot count** = number of `EpisodeSlot` per worker |
| `D` | `max_inflight_msgs`, `runner_wire_batched.cpp:287` `D = max(1, max_inflight_msgs)` | **in-flight message cap** per thread (count of outstanding ZMQ *messages*, not rows) |
| `max_batch` | server `InferenceServer.__init__`, `inference_server.py:149`; default 256; bench `--max-batch` default 512 | server-side drain row cap and pad target (peer parameter; affects RELY) |
| drain variant | `inference_server.py` (greedy) vs `stage_a_server.py` (bench: `e_policy`×`wakeup`) | peer's gather+forward policy (peer parameter; affects RELY/service timing) |
| `timeout_ms` | `WireRunnerConfig::timeout_ms`, default 15000 (`runner_wire_batched.hpp:22`) | DEALER `ZMQ_RCVTIMEO`; the only producer-side timeout |

This producer model is **per-thread**: `T` worker threads run the identical lambda
(`runner_wire_batched.cpp:312-478`) over disjoint `EpisodeSlot` arrays and disjoint episode indices
(`next_idx = tid`, stride `T`, line 323/402-403). The threads share only: the ZMQ context `zctx`
(`runner_wire_batched.cpp:289`, passed to every `WireLeafPool::create`), the atomic `corr_seq`
(`runner_wire_batched.cpp:298`, the global correlation-id allocator), the `failed`/`first_error`
machinery, and two stat atomics. **There is no per-thread shared mutable search state**; the only
cross-thread coupling on the hot path is (a) the shared ZMQ context (threads multiplex independent DEALER
sockets through it; this is the documented-safe ZMQ usage — one socket per thread) and (b) the global
`corr_seq` atomic. So the producer model is `T` independent copies of the single-thread state machine
below, coupled only at the server (they contend for the single ROUTER's attention — that contention lives
in the RELY about service timing, §4, not in producer state).

---

## 1. The slot lifecycle (per `EpisodeSlot`, K of them per thread)

An `EpisodeSlot` (`runner_wire_batched.cpp:21-35`) carries one in-progress episode and its fiber-suspended
tree search `ts` (`TreeState`, `fiber_tree.hpp:19-63`). The search is a **coroutine**: `ts->start(...)`
runs the search until the policy calls `ynet.predict(x)`, which (`fiber_leaf.hpp:24-29`) stashes the
feature span into `ch.features`, sets `ch.at_leaf = true`, and `resume()`s back to the caller. `running`
mirrors `ch.at_leaf` (`fiber_tree.hpp:55, 61`). So:

- `ts->running == true`  ⇔  the search is **parked at a leaf**, `ch.features` is a live span of the
  feature vector that needs one NN evaluation. (`fiber_tree.hpp:55,61`; `fiber_leaf.hpp:26`.)
- `ts->running == false` after a `start`/`resume_with`/`advance` ⇔ the search **finished** (returned a
  `Decision`); `ch.at_leaf` set false at `fiber_tree.hpp:51`.

The producer never sees the search internals; it sees only the **park/finish coroutine boundary**. The
interval between two consecutive parks of one slot — search progress between leaves — is **set by the
search's own work and is NOT fixed by this code**. It is the source-emission nondeterminism (§3).

### Per-slot state (the product state per slot)

Two orthogonal booleans fully determine a slot's transport-relevant state:
`active` (`EpisodeSlot::active`, line 22; set true in `fill` line 413, false in `finalize_and_write` line
356) and the pair (`ts!=null && ts->running`) = "parked at a leaf", plus the driver-owned flag
`submitted[s]` (`runner_wire_batched.cpp:327`, a `char` per slot).

| Slot state | Predicate (code) | Meaning |
|---|---|---|
| **EMPTY** | `!active` (after `fill` exhausts `next_idx>=episodes`) | no more episodes for this slot; permanently idle |
| **ELIGIBLE** | `is_ready(s)` = `active && ts && ts->running && !submitted[s]` (`runner_wire_batched.cpp:427-430`) | parked at a leaf, its eval not yet sent — a candidate for the next `issue_one` |
| **OUTSTANDING** | `active && ts && ts->running && submitted[s]` | its eval has been sent in some in-flight message; awaiting reply |
| **ADVANCING** | inside `advance(s)`/`fill(s)`/`resume_with` between recv and next park | the reply arrived; the search is being driven forward to its next park (or to finalize) — a transient computational state, never observed by `issue_one` because it is synchronous within the recv loop body |
| **FINALIZED→refilled or EMPTY** | `finalize_and_write` then `fill` (`runner_wire_batched.cpp:336-363, 398-425`) | episode ended (Terminate, max_steps, or empty belief); slot is rebound to a new episode (`fill` succeeds) or goes EMPTY (`fill` fails) |

Crucial slot-count fact: **all K slots exist for the whole run**; they are reused across episodes
(`fill` rebinds `sl.idx`, reseeds, resets, `runner_wire_batched.cpp:400-413`). A slot oscillates
ELIGIBLE → OUTSTANDING → ADVANCING → (ELIGIBLE | EMPTY) many times.

---

## 2. Operational state machine

The state machine has two layers: a **per-slot** layer (above) and a **per-thread driver** layer (the
pipeline controller). The transport-relevant nondeterminism lives at the driver layer; I give the driver
state machine, with per-slot states as the data it scans.

### Driver state (per thread)
Driver carries `inflight_msgs ∈ [0, D]` (`runner_wire_batched.cpp:328`) and, implicitly, the multiset of
per-slot states and the DEALER's `inflight_` map (`wire_leaf_pool.hpp:169`, corr-id → slot-list).

States:

- **INIT**: before any slot is filled.
- **FILLING**: running the initial `for s in 0..K: fill(s)` (`runner_wire_batched.cpp:454`).
- **PRIMING**: the prime loop `while inflight_msgs<D && issue_one()` (`runner_wire_batched.cpp:456`).
- **PUMPING**: the steady-state recv/resume/refill loop (`runner_wire_batched.cpp:457-475`).
- **DRAINING/DONE**: `inflight_msgs==0` and no eligible slots; loop exits (line 457 guard fails), thread
  joins (line 483).
- **FAILED**: `failed.load()` true anywhere (`set_error`, line 303-310); every loop guard
  `!failed.load()` short-circuits; thread returns.

The transitions (guard / action / code_ref / free-choice flag) are enumerated in the structured object's
`state_machine`. The load-bearing ones:

1. **issue (coalesce)** — `issue_one` (`runner_wire_batched.cpp:434-452`). Guard: at least one
   `is_ready(s)` AND `inflight_msgs<D`. Action: scan **all K slots in index order** (line 437); append
   every ready slot's `ts->ch.features` into one flat `gather` buffer and its index into `gathered`
   (lines 438-442); `submit_batch(gathered, gather, in_dim)` — **ONE ZMQ message carrying B=|gathered|
   rows** (line 445; `wire_leaf_pool.hpp:76-94`); mark each `submitted[s]=1` (line 447); `++inflight_msgs`
   (line 448). The **coalescing degree B = number of slots simultaneously ELIGIBLE at the instant
   `issue_one` runs**, `1 ≤ B ≤ K`. This is the producer's single most important free choice surface, and
   it is *not* free in the scheduling sense — it is a deterministic function of the eligible-set at that
   instant, but the eligible-set is itself shaped by source-emission timing (§3) and by reply arrival
   order (§4), both nondeterministic. **Not a free choice of the driver; a function of nondeterministic
   inputs.**

2. **send (two-frame)** — `submit_batch` (`wire_leaf_pool.hpp:76-94`): `zmq_send(corr, SNDMORE)` then
   `zmq_send(payload, 0)`; record `inflight_[corr] = gathered` (line 92). corr from
   `corr_seq->fetch_add(1)` (line 84, `memory_order_relaxed`, global across threads). The DEALER prepends
   no identity frame of its own on send; ZMQ delivers `[corr][payload]` to the ROUTER, which prepends the
   DEALER's identity (RELY §4).

3. **recv (blocking, ordered by arrival)** — `recv_batch` (`wire_leaf_pool.hpp:106-132`) →
   `recv_corr_payload` (`wire_leaf_pool.hpp:140-165`): loop `zmq_msg_recv` over all frames of one
   multipart message until `!more`; the **first call blocks up to `ZMQ_RCVTIMEO=timeout_ms`** (set
   line 41). On the wire the reply is `[corr][resp-payload]` (server strips its own routing-identity and
   echoes `[corr, resp]`, RELY §4). Match `inflight_.find(corr)` (line 115); **unknown corr → hard error**
   (line 116-118), recover the slot-list, erase, decode, scatter into `Completion{slot, pred}` (line
   125-131). The producer consumes replies **in ZMQ arrival order on its DEALER**, which for a single
   DEALER↔ROUTER pair is the server's send order — **not** the producer's send order. This is the
   out-of-order-by-corr-id latitude (§5, DOF-4).

4. **resume + dispatch** — for each `Completion` (`runner_wire_batched.cpp:462-472`): `submitted[s]=0`
   (line 466, slot leaves OUTSTANDING); `sl.ts->resume_with(c.pred)` (line 467; `fiber_tree.hpp:58-62`
   feeds the prediction back into the coroutine, runs to the next park or finish). If still running → slot
   is now **ELIGIBLE again** (continue, line 468). Else `advance(s)` (drive plies until next leaf-park or
   episode end, line 469; `runner_wire_batched.cpp:389-397`) → if it parks, ELIGIBLE; if episode ends,
   `finalize_and_write` inside advance/apply_decision, then `fill(s)` rebinds or EMPTY (line 471).

5. **refill** — after processing a whole reply: `while inflight_msgs<D && !failed && issue_one()`
   (line 474). Tops the pipeline back up to D messages if enough slots are eligible.

### The K=base, D≥? degenerate and the N≥2 regime
- With `N=1` (`K=base`) the slot count equals StrictBarrier's. With `N≥2`, `K` grows linearly in `N`
  (`K=N·base`, line 286), so the *pool of potentially-eligible slots* grows linearly in `N`. This is the
  central N-dependence; everything in §3/§5/§6 is a corollary.

---

## 3. Source-emission timing (the park interval) — first-class nondeterminism

**What the code fixes.** Causally, for a single slot:
- A slot can only *become* ELIGIBLE by parking, which happens inside `spawn_ply`/`start` (the very first
  leaf of an episode, `runner_wire_batched.cpp:419` / `fiber_tree.hpp:44-56`) or inside `resume_with`
  after a reply (`fiber_tree.hpp:58-62`) — i.e., a slot's *k-th* park causally requires the reply to its
  *(k−1)-th* leaf (the search cannot reach leaf k without the value at leaf k−1). This is a hard causal
  edge: **a thread cannot emit a reply-dependent request before that reply arrives** (the reply feeds
  `ch.value` at `fiber_leaf.hpp:28`, which is the return of `predict`, which the search consumes before
  descending further).
- The *first* park of a fresh episode (in `fill`→`spawn_ply`, line 419) is **not** reply-dependent; it can
  happen as soon as the slot is filled. So at PRIMING (line 456), up to K slots can already be ELIGIBLE
  with zero replies received — the prime can coalesce up to `min(K, ...)`-row first messages.

**What the code leaves free (the nondeterministic interval).** The wall-clock duration of search work
between one park and the next — the gap between `resume_with` returning and `ts->running` becoming true
again — is set entirely by `policy.run_search`'s internal progress (`fiber_tree.hpp:50`), which this code
does not constrain. Model it as a **positive, bounded-but-otherwise-free random duration** `δ_park(s,k) >
0` per slot per ply. The code constrains it only by:
- (C1) **positivity** — a coroutine cannot park instantaneously twice with no work; `δ_park > 0`.
- (C2) **reply-causality** — `δ_park(s,k)` is measured from the instant `resume_with(reply_{k-1})` is
  invoked, which is *after* the reply is received (`runner_wire_batched.cpp:467`); so the source emission
  of leaf k of slot s causally trails the reply to leaf k−1 of slot s.
- (C3) **finalization absorbs some parks** — `advance` (`runner_wire_batched.cpp:389-397`) loops
  `apply_decision; spawn_ply` until either a park (`ts->running`) or episode-end; a search may produce
  **multiple plies with no leaf** if `spawn_ply` immediately finishes (the `while(!ts->running)` loop body
  at line 391-394 runs apply_decision again). So *one reply can yield zero new ELIGIBLE for that slot* (the
  episode finalized) **or** *one new ELIGIBLE for that slot* (it re-parked) — never more than one new
  eligible *from that one slot* per reply, because a slot holds exactly one fiber.

This is **not** collapsed to a constant. The whole pacing behavior of §5/§6 is a function of the *joint*
distribution of `{δ_park(s,k)}` across the K slots; the model leaves that joint distribution free subject
only to C1–C3. (If it were collapsed to a constant, the eligible-set would evolve in lockstep and the
coalescing degree would be an artifact, not a real degree of freedom — that would be unfaithful.)

---

## 4. Sink-service timing (RELY about the peer) — first-class, not an instant

The producer's forward progress is gated by **when replies arrive**, which is set by the server's service
time. I model the server as the peer and state it as RELY (§7). Grounding it in the peer code so each
clause is checkable:

- The server is **single-threaded** (`inference_server.py:219-225` `serve_forever`: `drain` then
  `serve_batch`, sequential; bench `stage_a_server.py:97` runs one `serve_forever` thread). So **all T
  producer threads' requests are serviced by one forward-at-a-time engine** — replies to different threads
  are serialized through one ROUTER/forward. This couples the T producer copies at the sink.
- **Greedy drain** (`inference_server.py:160-186`): block on the poller until ≥1 message
  (`_POLL_INTERVAL_MS=100` ms poll loop, line 165), then `recv_multipart(NOBLOCK)` every currently-queued
  message until `total_rows >= max_batch` or `zmq.Again` (lines 171-186). So the server **coalesces across
  producers and across messages**: one forward can serve many DEALERs' messages. Then ONE forward padded
  to `max_batch` (`run_microbatch(..., pad_to=self._max_batch)`, line 198; `forward.py`), then scatter one
  reply per drained message (line 200, `send_multipart([ident, *envelope, resp])`).
- **Service time** = forward time. The forward (`forward.py:3-18`) is a fixed 2–3 layer MLP; in the greedy
  server it is **JIT-compiled and padded to a fixed shape `max_batch`** (`jit_forward_core`,
  `inference_server.py:22-34`; `pad_to=self._max_batch` line 198). Padding to a fixed shape means the XLA
  executable is compiled **once per shape**; service time is therefore ≈ **constant in the number of real
  rows** (it always computes `max_batch` rows) — call it `S_pad`. The bench server's `padmax` E-policy is
  identical (`stage_a_server.py:61-62`, `pad_to=max_batch`); its `bucket` E-policy snaps to the smallest
  of `{64,256,512}` ≥ real rows (`stage_a_server.py:30-37,63-64`), giving a **step function** service
  time `S(bucket(real))` with three plateaus (three compiled shapes) — smaller real batches get the cheap
  64-shape. The `wakeup` knob (`stage_a_server.py:57`) chooses ONE forward per drained group (`group`) vs
  ONE forward per queued message (`leaf`): `leaf` multiplies forward count by the number of messages and
  shrinks each batch, so under `bucket`+`leaf` most forwards hit the 64-bucket.
- **Service-time model (the RELY the producer assumes):** a positive duration `S_fwd > 0` per forward,
  with `S_fwd = S_pad` (constant) under padmax/greedy, or `S_fwd ∈ {S_64, S_256, S_512}` (a step in real
  rows) under bucket. **Not collapsed to an instant** — a reply *cannot* precede the forward that produced
  it (causal), and the reply to corr c cannot arrive at the producer before the server has (a) drained c's
  message, (b) run a forward covering it, (c) sent c's reply. Across producers the server processes drains
  FIFO-ish by poller readiness, so a thread's reply latency includes queueing behind other threads' drains
  — bounded only by `max_batch` per forward and the single-thread serialization.

The producer's `ZMQ_RCVTIMEO=timeout_ms` (default 15000 ms, `wire_leaf_pool.hpp:41`) caps how long
`recv_batch` blocks; if the server takes longer than `timeout_ms` to reply, `zmq_msg_recv` returns EAGAIN,
`recv_corr_payload` returns an error (`wire_leaf_pool.hpp:147-150`), `recv_batch` propagates it, the recv
loop calls `set_error` and breaks (`runner_wire_batched.cpp:459`) → FAILED. So the producer's liveness
RELY is "server replies to every corr within `timeout_ms`"; violation is a loud failure, not a silent
hang (ADR-0002 shape).

---

## 5. Degrees of freedom (each with code_ref, behaviors admitted, N-dependence)

See the structured object's `degrees_of_freedom`. Summary of the load-bearing ones:

- **DOF-1: park-interval nondeterminism** `{δ_park(s,k)>0}` (`fiber_tree.hpp:50`,
  `runner_wire_batched.cpp:467`). Admits any interleaving of which slots are ELIGIBLE at any instant,
  subject to C1–C3. **N-dependence:** more slots (K=N·base) → more independent `δ_park` streams → the
  eligible-set size at any instant is a sum of more Bernoulli-ish indicators → by concentration, the
  *expected* number simultaneously eligible grows ~linearly in N while its *relative* variance shrinks
  (√N/N). So coalescing becomes both larger and steadier as N grows.

- **DOF-2: coalescing degree B per message** (`runner_wire_batched.cpp:437-444`). `1 ≤ B ≤ K`. **Not a
  free choice** — deterministic in the eligible-set at the `issue_one` instant — but the eligible-set is a
  DOF-1×DOF-4 function. **N-dependence:** B's reachable maximum is K=N·base (linear in N); its typical
  value rises with N because PRIMING (line 456) and refill (line 474) each gather *all* currently-eligible
  slots, and with more slots more are eligible per gather. As N→∞ (with D, base fixed), a single message
  can carry up to N·base rows — the producer's batch grows linearly in N. This is the mechanism by which
  overcommit raises server batch size.

- **DOF-3: in-flight message depth `inflight_msgs ∈ [0,D]`** (`runner_wire_batched.cpp:328,448,460`).
  The pipeline holds up to D *messages* concurrently outstanding. **N-independent in cap** (D is its own
  knob) but **N-coupled in content**: with K≫D, each of the D messages can be a large coalesced batch, so
  the number of *rows* in flight is up to D·(typical B) which rises with N. If K ≤ D the pipeline can
  never fill (fewer eligible slots than message slots); if K ≫ D (large N) the refill loop (line 474)
  keeps all D message-slots full and the leftover eligible slots wait for the next issue_one.

- **DOF-4: reply arrival order ≠ send order** (`wire_leaf_pool.hpp:106-132`,
  `runner_wire_batched.cpp:458`). The DEALER receives replies in the server's send order; with one server
  draining/coalescing across messages and replying per-message, the order is **whatever order the server
  sends** — for the greedy server, the scatter order is `zip(run_microbatch(...), drained)` = drain order
  (`inference_server.py:197-200`), and drains coalesce multiple producer messages, so a later-sent message
  can be replied-to before an earlier-sent one across threads, and within one thread the D outstanding
  messages can be answered in any order the server chooses. The producer is **order-agnostic**: it matches
  by corr-id (`wire_leaf_pool.hpp:115`), not by FIFO. **N-dependence:** larger N → more rows per message →
  fewer messages per thread for the same work, but the *per-message* corr-id matching is unaffected; the
  out-of-order latitude is structurally constant in N (it is a property of the corr-id design), though
  with more in-flight content the *opportunity* for reorder per unit work rises.

- **DOF-5: prime vs refill coalescing asymmetry** (`runner_wire_batched.cpp:456` vs `474`). PRIMING runs
  before any reply, so it gathers only first-leaf parks (DOF-1 C3 note: every fresh-filled slot parks once
  with no reply needed); refill runs after each reply and re-gathers slots that re-parked plus any that
  were waiting. **N-dependence:** with large N the prime can already issue D large messages from first
  leaves alone; with N=1 the prime may only fill a fraction of D.

- **DOF-6: episode-boundary slot churn** (`fill`/`finalize_and_write`, `runner_wire_batched.cpp:336-363,
  398-425`). A reply can finalize an episode (no re-park) and `fill` rebinds the slot, whose new first leaf
  is immediately eligible — or `fill` fails (`next_idx>=episodes`) and the slot goes EMPTY. **N-dependence:**
  with K=N·base slots and a fixed total `episodes`, larger N exhausts `next_idx` per thread sooner *per
  slot* but the EMPTY slots simply drop out of the eligible scan; the active pool shrinks toward the end,
  reducing coalescing in the tail (a ramp-down whose length is ~N·base episodes).

---

## 6. N-dependence summary (derived, not assumed)

As `N = trees_per_thread` grows (T, base, D, max_batch fixed):

1. **Slot pool grows linearly:** K = N·base (`runner_wire_batched.cpp:286`).
2. **Coalescing degree grows linearly in its ceiling and rises typically:** B ∈ [1, K]; `issue_one`
   gathers *all* currently-eligible slots (`runner_wire_batched.cpp:437-444`), and the expected
   eligible-set size grows ~linearly in N (DOF-1/DOF-2). Mean rows/msg (reported at
   `runner_wire_batched.cpp:496`) is monotone increasing in N until clipped by server `max_batch`.
3. **Messages per unit work fall:** more rows per message ⇒ fewer messages for the same number of leaves;
   `total_msgs` (line 477,498) falls, `mean_rows_per_msg` (line 496) rises with N.
4. **Pipeline depth in rows rises, in messages is capped at D:** `inflight_msgs ≤ D` always (line 456,
   474); but rows-in-flight ≈ D·B rises with N. For K ≤ D the pipeline underfills regardless of N; the
   knee is at N·base ≈ D.
5. **Steadier batches:** relative variance of the eligible-set ~1/√N, so the coalescing degree (hence
   server batch utilization and pad fraction) stabilizes as N grows — the server's `pad_fraction`
   (`stage_a_server.py:113-114`) falls toward 0 under padmax as N pushes B toward max_batch.
6. **Tail ramp-down lengthens:** ~N·base episodes of declining active slots at the end (DOF-6).
7. **Memory/throughput tradeoff is the only thing N buys the producer:** N does not change the *protocol*
   (corr-id matching, two-frame send, D-cap) at all; it changes the *statistics* of the batch a producer
   hands the server. The transport state machine is invariant in N; only the data feeding its guards
   scales.

---

## 7. Assume–Guarantee contract

### RELY (what the producer assumes about the peer/server, each checkable against `inference_server.py` / `stage_a_server.py`)
- **R1 (reply shape):** every reply is a multipart message whose **first frame is the 8-byte corr-id**
  echoed unchanged and whose **last frame is a valid wire-v2 response payload**
  (`wire_leaf_pool.hpp:157-163` requires ≥2 frames, leading 8 bytes; `inference_server.py:200`
  `send_multipart([ident, *envelope, resp])` where `envelope`=the producer's corr frame, line 177,183).
  Checkable: the greedy server preserves `envelope = frames[1:-1]` (line 177) and re-emits it (line 200).
- **R2 (corr echo & uniqueness):** the corr in the reply equals a corr the producer sent and not yet
  retired (`wire_leaf_pool.hpp:115-118` errors otherwise). Checkable: server never invents corr-ids;
  echoes envelope verbatim.
- **R3 (row count match):** the reply's prediction count equals the request's row count B
  (`wire_leaf_pool.hpp:121-124` errors otherwise). Checkable: `run_microbatch` slices `v[off:off+n]` per
  request using the request's own `counts` (`inference_server.py:50-72`) — exactly B per ident.
- **R4 (eventual reply within timeout):** every sent corr is answered within `timeout_ms`
  (`wire_leaf_pool.hpp:41`). Checkable: server's drain blocks ≤100 ms then forwards
  (`inference_server.py:165`); forward time `S_fwd` is bounded by the fixed `max_batch` shape; under
  load, latency ≤ (queue-ahead drains)·S_fwd, which the producer assumes < timeout_ms.
- **R5 (no spurious frames):** replies are not partial/multipart-corrupt;
  `recv_corr_payload` consumes exactly one logical reply per `more`-loop (`wire_leaf_pool.hpp:142-156`).
- **R6 (service-time character):** `S_fwd>0`, constant in real rows under padmax, a 3-step function under
  bucket (§4). The producer does not depend on `S_fwd`'s value for correctness, only for liveness (R4)
  and for the *statistics* that make large-N coalescing worthwhile.

### GUARANTEE (what the producer provides to the server)
- **G1 (well-formed request):** every sent message is exactly two frames `[corr(8 bytes)][payload]`
  (`wire_leaf_pool.hpp:86-91`), payload = wire-v2 header (version=2, B, in_dim) + B·in_dim f32, with
  B≥1, in_dim≥1, and `flat.size()==B·in_dim` enforced before send (`inference_wire.hpp:51-70`).
- **G2 (corr uniqueness, monotone allocation):** every corr is a distinct value from a single global
  atomic `fetch_add` (`wire_leaf_pool.hpp:84`, `runner_wire_batched.cpp:298`), so no two outstanding
  requests across all T threads ever share a corr (the server may safely use it as the reply key).
- **G3 (in-flight bound):** at most D messages per thread are outstanding at once
  (`runner_wire_batched.cpp:456,474` gate on `inflight_msgs<D`), so the producer offers bounded backlog;
  combined with default `ZMQ_SNDHWM=1000` (unset → default) the producer never blocks on send under any
  realistic D (D default 8 ≪ 1000).
- **G4 (reply-causal pacing):** the producer never sends a request whose features depend on a not-yet-
  received reply (the coroutine cannot reach leaf k without reply k−1, §3 C2).
- **G5 (one consumer per socket):** exactly one thread owns each DEALER (`WireLeafPool` is move-only,
  non-copyable, `wire_leaf_pool.hpp:55-69`; constructed once per worker, `runner_wire_batched.cpp:313`),
  so ZMQ's single-thread-per-socket rule holds.
- **G6 (eventual recv / liveness):** while `inflight_msgs>0` the producer is in a blocking `recv_batch`
  (`runner_wire_batched.cpp:458`), so it always eventually consumes the server's replies (it does not
  send-only and stall the server's HWM).

### Socket options (determined from code; blocking depends on exactly these)
- `ZMQ_DEALER` (`wire_leaf_pool.hpp:35`).
- `ZMQ_LINGER = 0` set (`wire_leaf_pool.hpp:39-40`): on close, discard unsent — no shutdown hang.
- `ZMQ_RCVTIMEO = timeout_ms` set (`wire_leaf_pool.hpp:41`): `recv_batch` blocks at most timeout_ms then
  EAGAIN → loud error.
- `ZMQ_SNDTIMEO`: **not set** → default −1 (block forever on send if SNDHWM reached). Reachable only if
  >1000 messages queue unsent (SNDHWM default), which G3 (D≪1000) prevents.
- `ZMQ_SNDHWM` / `ZMQ_RCVHWM`: **not set** → default 1000 each. With D≪1000 and per-message rows bounded,
  HWM back-pressure is not on the modeled path.
- `ZMQ_ROUTER_MANDATORY`: N/A (DEALER side).
- No `zmq_ctx_set` options (`runner_wire_batched.cpp:289` plain `zmq_ctx_new`): default IO threads = 1,
  default max sockets. The single context is shared by all T DEALERs (safe: distinct sockets).

---

## 8. DOF-control notes (what removing each constraint would wrongly admit)

See `dof_controls` in the structured object. Each control names a constraint that, if dropped, makes the
model **over-permissive** (admits executions the code cannot produce) or, if added, makes it
**over-constrained** (forbids ones it can).

---

## 9. Fidelity self-audit

See `fidelity_self_audit`. The two sharp edges: (a) the coalescing degree B is *not* a free scheduler
choice — modeling it as free would over-permit (it is pinned to the eligible-set at the issue instant);
(b) the park interval δ_park *is* free (positive, bounded) — collapsing it to a constant would
over-constrain and destroy the very N-dependence the task asks for.

---

## 10. Z3 confirmation (confirmation only, not the source of trust)

A small bounded encoding in `out/producer_check.py` confirms that a representative execution is admissible:
two slots with nondeterministic park intervals, D=1, exhibiting (i) a coalesced B=2 prime message, (ii) an
out-of-order reply, (iii) reply-causal pacing (no leaf-k request before reply k−1). The check is bounded
and minimal; it confirms admissibility, it does not establish the model.
