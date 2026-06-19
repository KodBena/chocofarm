# Faithful parametric model — C++ PRODUCER side of the leaf-eval transport boundary

**Role:** producer (the C++ worker thread: N independent trees/thread + ZeroMQ DEALER wrapper
`WireLeafPool` + the pipelined driver `run_episodes_wire_pipelined`).

**Method:** derived FORWARD from the cleanroom code's operational semantics. Every state,
transition, guard, free choice, and timing assumption is mapped to a cleanroom `file:line` (the line
numbers correspond to the real source) or to a named causal necessity. No outside expectation of how
the system "ought" to behave is introduced.

All files read end to end (per ADR-0002), all under
`/home/bork/w/vdc/chocobo/runs/leaf-eval-model-2/cleanroom`:

- `cpp/include/chocofarm/wire_leaf_pool.hpp` (DEALER wrapper) — full
- `cpp/src/runner_wire_batched.cpp` (driver) — full
- `cpp/include/chocofarm/runner_wire_batched.hpp` (WireRunnerConfig) — full
- `cpp/include/chocofarm/inference_wire.hpp` (codec) — full
- `cpp/include/chocofarm/wire_spec.hpp` (frame layout) — full
- `cpp/include/chocofarm/fiber_tree.hpp`, `fiber_leaf.hpp` — full
- `cpp/include/chocofarm/runtime_config.hpp` — full
- `cpp/include/chocofarm/error.hpp`, `net_evaluator.hpp` — full
- `chocofarm/az/inference_server.py` (the peer; grounds the RELY) — full
- `chocofarm/az/inference_wire.py`, `chocofarm/az/forward.py`,
  `cpp/stage_a/stage_a_server.py`, `chocofarm/config.py` (peer context) — full

Verified by `grep` over the whole cleanroom that the ONLY ZMQ options set in the producer path are at
`wire_leaf_pool.hpp:40` (`ZMQ_LINGER=0`) and `:41` (`ZMQ_RCVTIMEO=timeout_ms`); the only `ZMQ_SNDMORE`
flag is at `:86`. No `zmq_ctx_set`, no `SNDHWM`/`RCVHWM`, no `SNDTIMEO`, no `ROUTER_MANDATORY` anywhere
in producer or server. (The server Python sets no socket options either.)

---

## 0. Parameters

From `WireRunnerConfig` (`runner_wire_batched.hpp:18-26`) and the driver
(`runner_wire_batched.cpp:280-298`):

| symbol | source | meaning |
|---|---|---|
| `T` | `pool_threads`, clamped `max(1,·)` (`:283`) | number of worker threads, each = one DEALER, one inflight map |
| `N` | `trees_per_thread`, clamped `max(1,·)` (`:285`) | independent searches multiplexed per thread (the focus parameter) |
| `base` | `RuntimeConfig::fibers_per_thread()` = `ceil(max(1,pool_batch)/max(1,T))` (`runtime_config.hpp:12-15`) | per-thread slot base |
| `K` | `N * base` (`:286`) | **per-thread slot count** (`std::vector<EpisodeSlot> slots(K)` `:324`) |
| `D` | `max_inflight_msgs`, clamped `max(1,·)` (`:287`) | **in-flight MESSAGE cap** per thread (NOT a leaf/row cap) |
| `timeout_ms` | `WireRunnerConfig::timeout_ms` default 15000 (`runner_wire_batched.hpp:22`) | DEALER `ZMQ_RCVTIMEO` |
| `endpoint` | `WireRunnerConfig::endpoint` | the ROUTER bind (`tcp://…`) |
| `max_batch`, `E-policy`, `wakeup` | server-side knobs (peer) | shape the RELY on service timing, not the producer's local state |

`feat_dim = fb.dim()` (`:275`) is the per-leaf feature width; `in_dim = feat_dim` (`:325`). The wire
protocol version is `2` (`wire_spec.hpp:8`).

This model is the `WireMode::PipelinedBucket` path only: `run_episodes_wire_batched` dispatches to
`run_episodes_wire_pipelined` when `wcfg.mode == WireMode::PipelinedBucket`
(`runner_wire_batched.cpp:44-45`). (The `StrictBarrier` path — same file, `:215-251` — is the
all-ready / submit-one / recv-one degenerate of `D=1` with `K=base` and no `submitted[]` book; noted
where it differs, but not the assigned model.)

Each of the `T` worker threads runs `worker(tid)` (`:312-478`) independently; they share only
`zctx`, the atomics (`written`, `failed`, `corr_seq`, `total_*`), and `err_mu`. **The transport state
is entirely per-thread**: each thread constructs its own `WireLeafPool` (`:313-315`), so its own
DEALER socket and its own `inflight_` correlation map. The corr-id counter `corr_seq` is the ONE
shared transport-relevant atomic (`:298`), fetched-add relaxed (`wire_leaf_pool.hpp:84`); it makes
correlation ids globally unique across threads, which matters for the RELY (a reply for thread A's
corr can never be routed to thread B because DEALER↔ROUTER routing is per-connection, but global
uniqueness means even a hypothetical mis-route would fail the `inflight_.find` check rather than
silently alias). I model ONE thread's transport in full and treat the other `T-1` as peers competing
only for the single-threaded server's service time.

---

## 1. The slot lifecycle (the SOURCE behavior at the interface)

A slot is an `EpisodeSlot` (`:21-35`, declared `slots(K)` at `:324`). Its transport-relevant life:

A search runs as a **boost::context fiber** (`fiber_tree.hpp:19-63`). The fiber executes
`policy.run_search(...)` (`:50`). When the search needs a net evaluation it calls
`YieldingNetEvaluator::predict(x)` (`fiber_leaf.hpp:24-29`), which sets `ch.features = x`,
`ch.at_leaf = true`, and **resumes the caller** (`:27`) — i.e. yields back to the driver thread,
PARKED at a leaf. `TreeState::start`/`resume_with` set `running = ch.at_leaf` (`fiber_tree.hpp:55,62`):
`running == true` means "parked at a leaf, one feature row `ch.features` awaiting a prediction";
`running == false` means the fiber returned (search finished, `ch.at_leaf=false` at
`fiber_tree.hpp:51`), i.e. a decision is ready.

**`ch.features` is a `std::span<const float>` (`fiber_leaf.hpp:16`) — a non-owning view into the
fiber's stack.** It is valid only while the fiber is parked and unchanged; this is why the driver
copies it immediately into `gather` (`:439-440`, `gather.insert(... feats.begin(), feats.end())`)
before any resume could invalidate it. (Causal necessity: a thread must copy a parked slot's features
before resuming that slot.)

**Source emission timing.** "When a slot next parks" is set by the search's own internal progress
between two `predict` calls — the number of simulations, tree shape, RNG. The code FIXES NONE of this.
So at the interface, a parked slot's feature row is emitted at a **nondeterministic positive interval**
after the slot was last resumed; one parked slot per leaf, exactly one row (`B=1` per slot — the gather
appends `feats` = exactly `in_dim` floats per gathered slot, `:439`). Each slot parks/finishes some
**bounded but unfixed** number of times per episode; the model leaves this latitude open (DOF-T1).

Per-episode book (not transport, but it gates when a slot is `active`/`ready`): `fill` (`:398-425`)
seeds a new episode index `next_idx += T` (so thread `tid` owns indices `tid, tid+T, tid+2T, …`,
`:401-402`), runs `spawn_ply` and possibly `advance` synchronously, and returns `true` iff the slot is
left parked (`sl.ts->running`, `:420`). `advance` (`:389-397`) loops applying decisions and respawning
plies **synchronously until either the slot parks (`running`, return true) or the episode ends**. A
slot can therefore traverse many plies (each a synchronous CPU burst) between two transport events,
all invisible to the wire — only the *parked* moments touch transport.

---

## 2. The transport state machine (one DEALER, one thread)

### States

The per-thread driver state is the triple **(submission book, inflight count, socket phase)**:

- `submitted[s] ∈ {0,1}` for `s ∈ [0,K)` (`:327`): slot `s` has a request on the wire awaiting reply.
- `inflight_msgs ∈ [0,D]` (`:328`): number of MESSAGES sent but not yet recv'd by this thread.
- `pool.inflight_` : `corr → vector<int> slots` (`wire_leaf_pool.hpp:169`): the correlation→slots map;
  `|inflight_| == inflight_msgs` is an invariant (one map entry per outstanding message — proved in
  §6).
- socket phase ∈ {idle, mid-send (between the two `zmq_send`s `wire_leaf_pool.hpp:86/89`), blocked-in-recv}.

| state | meaning |
|---|---|
| `FILL` | startup: `for s in [0,K): fill(s)` (`:454`); slots being seeded synchronously, no transport yet |
| `PRIME` | `while inflight_msgs<D && issue_one()` (`:456`): issue messages until cap or no ready slot |
| `GATHER` | inside `issue_one`: scanning `is_ready(s)` over all K, building one `gather`/`gathered` (`:434-444`) |
| `SEND` | inside `submit_batch`: `zmq_send(corr,SNDMORE)` then `zmq_send(payload,0)` (`wire_leaf_pool.hpp:86-91`) |
| `RECV_BLOCK` | `pool.recv_batch()` → blocking `zmq_msg_recv` loop, bounded by `RCVTIMEO` (`wire_leaf_pool.hpp:147`) |
| `MATCH` | decode reply, look up corr in `inflight_`, scatter `Completion`s (`wire_leaf_pool.hpp:106-132`) |
| `RESUME` | per completion: `submitted[s]=0`, `resume_with`, then `advance`/`fill` (`:462-472`) |
| `REFILL` | `while inflight_msgs<D && !failed && issue_one()` (`:474`): re-prime after a drain |
| `DONE` | `inflight_msgs==0` and loop exits (`:457`); or `failed` set; thread joins |
| `ERR` | any transport/codec/desync error → `set_error`, break (`:446,459`; `wire_leaf_pool` errors) |

The main loop is **`while (inflight_msgs > 0 && !failed)`** (`:457`): drain one reply, resume its
slots, then refill toward `D`. It is a **strict request/reply pump bounded by D**: the thread is never
blocked in recv with `inflight_msgs==0`, and never holds more than `D` outstanding messages.

### Transitions (guard / action / code_ref / free-choice)

| # | from→to | guard | action | code_ref | free? |
|---|---|---|---|---|---|
| t0 | start→FILL | thread spawned | construct DEALER (`create`), set LINGER=0, RCVTIMEO, connect | `wire_leaf_pool.hpp:35-48`; `runner…:313-315` | no |
| t1 | FILL→FILL | `s<K` | `fill(s)`: seed episode, run synchronous plies; leaves slot parked or finished | `:454,398-425` | yes (source timing: which slots end up parked) |
| t2 | FILL→PRIME | all K filled | enter prime loop | `:456` | no |
| t3 | PRIME/REFILL→GATHER | `inflight_msgs<D` | call `issue_one()` | `:456,474` | no |
| t4 | GATHER→(no-op) | `gathered.empty()` | `issue_one` returns false; prime/refill loop stops | `:444` | no |
| t5 | GATHER→SEND | `∃ ready slot` | `submit_batch(gathered,gather,in_dim)`: encode, `corr=fetch_add` | `:445`; `wire_leaf_pool.hpp:76-84` | no |
| t6 | SEND→SEND | corr frame queued | `zmq_send(payload,0)` (2nd frame) | `wire_leaf_pool.hpp:89` | no |
| t6b | SEND→ERR | `zmq_send<0` (e.g. ETERM on ctx term) | `set_error`, return false | `wire_leaf_pool.hpp:86-91`; `:446` | no |
| t7 | SEND→PRIME/REFILL | both sends ok | `inflight_.emplace(corr,slots)`; mark `submitted[s]=1 ∀ gathered`; `++inflight_msgs`; `++my_msgs` | `wire_leaf_pool.hpp:92`; `:447-451` | no |
| t8 | PRIME→RECV_BLOCK | prime loop done, `inflight_msgs>0` | enter main loop, `pool.recv_batch()` | `:457-458` | no |
| t9 | RECV_BLOCK→RECV_BLOCK | frame recv'd, `more==1` | accumulate frame, loop | `wire_leaf_pool.hpp:144-156` | no |
| t10 | RECV_BLOCK→MATCH | last frame (`more==0`), ≥2 frames, leading 8B | memcpy corr, take payload | `wire_leaf_pool.hpp:157-164` | no |
| t11 | RECV_BLOCK→ERR | `zmq_msg_recv<0` (EAGAIN after `RCVTIMEO`, or ETERM) | error "zmq_msg_recv failed" | `wire_leaf_pool.hpp:147-151` | **yes (timeout: when the reply is later than `timeout_ms`)** |
| t12 | MATCH→ERR | malformed payload / unknown corr / size≠slots | corresponding error | `wire_leaf_pool.hpp:112-124` | no (RELY-guarded; see §5) |
| t13 | MATCH→RESUME | reply decoded, corr found, sizes match | build `Completion[]` (slot↦pred), `--inflight_msgs` | `wire_leaf_pool.hpp:125-132`; `:460` | no |
| t14 | RESUME→RESUME | per completion `c`, `sl.ts->running` after resume | `submitted[s]=0`; `resume_with`; still parked → continue | `:462-468` | yes (source: a fresh leaf appears for `s`) |
| t15 | RESUME→(advance) | resume left slot not running | `advance(s)`: synchronous plies; if parks again, continue; else `fill(s)` | `:468-471` | yes (source timing + episode end) |
| t16 | RESUME→REFILL | completions exhausted | re-enter `while inflight_msgs<D && issue_one()` | `:474` | no |
| t17 | REFILL→RECV_BLOCK | `inflight_msgs>0` after refill | loop back to `recv_batch` | `:457` | no |
| t18 | any→DONE | `inflight_msgs==0` (all work drained, no new ready) | loop exits; accumulate `total_*`; thread returns | `:457,476-477` | no |
| t19 | any→ERR | `failed.load()` observed | break out, thread returns | `:457,462,470` | yes (another thread may set `failed`) |

`is_ready(s)` (`:427-430`) = `sl.active && sl.ts && sl.ts->running && !submitted[s]`: a slot is
**eligible for the NEXT message** iff it is parked at a leaf AND not already on the wire. `submitted[]`
is the de-dup that keeps a parked slot from being gathered twice while its reply is outstanding.

---

## 3. The producer's complete BLOCKING SURFACE

This is the heart of the assigned focus. Enumerated exhaustively from the code; for each: where, how
long, what unblocks.

### 3.1 `SEND` — the two-frame `zmq_send` (`wire_leaf_pool.hpp:86-91`)

- `ZMQ_SNDMORE` corr frame (8 bytes) then payload frame (`HEADER_BYTES + B*in_dim*4` bytes,
  `inference_wire.hpp:64`). **No `ZMQ_DONTWAIT`** → these are *blocking* sends.
- **`ZMQ_SNDTIMEO` is NOT set** → OS default `-1` (infinite). **`ZMQ_SNDHWM` is NOT set** → default
  `1000` messages.
- A DEALER `zmq_send` blocks **only when the outbound pipe to the connected ROUTER is at SNDHWM**
  (1000 queued messages for that peer) AND the peer is not draining. On a healthy connected
  DEALER↔ROUTER it returns immediately (copies into the pipe). **Can this thread reach SNDHWM?**
  No: the thread holds at most `D` messages outstanding before it must `recv_batch` (`:457` guard),
  and `D = max_inflight_msgs` with default 8 — far below 1000. The thread cannot enqueue a
  `D+1`-th send without first draining one. **So `SEND` never blocks on HWM in any reachable state
  for any `D ≤ 1000`.** (If a deployment set `D > 1000` AND the server stalled, the 1001st send would
  block indefinitely — `SNDTIMEO=-1` — but `D≤1000` is the operative regime; see DOF-T4.)
- Failure mode: `zmq_send < 0` only on a hard error — `ETERM` (context terminated, e.g. another path
  called `zmq_ctx_term`, but here `zmq_ctx_term` is after all joins `:485`), `EFAULT`, `EINTR`. Then
  t6b → `set_error`. **LINGER=0** (`wire_leaf_pool.hpp:39-40`) means at socket close, any still-queued
  unsent message is discarded immediately rather than blocking the close.
- **N-dependence:** the per-message payload size grows with the number of gathered ready slots, which
  scales with `K = N·base` (a single `issue_one` can gather up to all K ready slots into ONE message
  → up to `K * in_dim * 4` payload bytes). More N ⇒ fatter single messages, but the *number* of
  in-flight messages is still capped at `D` independent of N. So the outbound BYTE pressure grows ∝ N
  while the outbound MESSAGE-COUNT pressure is constant in N. HWM is counted in **messages**, so HWM
  proximity does NOT grow with N (it stays `≤ D ≪ 1000`); the byte queue per message grows ∝ N but
  ZMQ HWM does not bound bytes. (Practical TCP backpressure on enormous single frames is below the ZMQ
  model and not represented as a state.)

### 3.2 `RECV_BLOCK` — `recv_batch` → `recv_corr_payload` → `zmq_msg_recv` (`wire_leaf_pool.hpp:140-165`)

- `zmq_msg_recv(&m, sock_, 0)` — flags `0`, **blocking** (`:147`).
- **`ZMQ_RCVTIMEO = timeout_ms`** IS set (`:41`, default 15000). So the FIRST-frame recv blocks at
  most `timeout_ms`; on expiry `zmq_msg_recv` returns `-1` with `errno=EAGAIN` → t11 → error
  `"WireLeafPool::poll: zmq_msg_recv failed: …"`. **This is the producer's only bounded wait and its
  liveness backstop:** if the server never replies (dead/slow), the thread fails loudly after
  `timeout_ms` rather than hanging forever (ADR-0002 fail-loudly, realized as the one finite timeout).
- **Subtlety — multi-frame atomicity.** `RCVTIMEO` applies to *each* `zmq_msg_recv` call. The loop
  (`:144-156`) reads frames while `more`. A reply is `[corr | payload]` (2 frames, see RELY §5). After
  the first frame arrives, ZMQ delivers a multipart message atomically, so the subsequent
  `zmq_msg_recv` for the payload frame returns essentially immediately (the whole message is already
  in the pipe). The timeout is therefore effectively a bound on *waiting for a reply to begin*, not on
  inter-frame gaps.
- What unblocks it: **a reply message from the server for this DEALER's connection** — and because the
  loop is the SAME `pool` (DEALER) the thread submitted on, ZMQ FIFO-orders replies on that pipe. The
  thread will receive replies for ITS corr-ids only (per-connection routing on the ROUTER side, RELY
  §5). It blocks here even if many of its own slots are parked-and-ready: the loop structure
  (`:457-475`) requires draining at least one outstanding message before issuing more, so a thread
  with `inflight_msgs==D` and more ready slots will WAIT in recv rather than send — the D cap converts
  ready-slot pressure into a blocking-recv wait.
- **N-dependence:** the *time* spent blocked here is governed by sink service timing (RELY §5), which
  depends on aggregate offered load. As N grows, each message can carry more rows and each `recv` can
  return more `Completion`s (a `recv_batch` returns `decoded->size()` predictions = the gathered batch
  size, up to K), so the thread does MORE work per unblock and unblocks LESS often per leaf — fewer,
  fatter round-trips. The number of *distinct* RECV_BLOCK episodes per episode-batch falls roughly ∝
  1/(rows-per-msg), i.e. ∝ 1/N in the all-ready regime. The `timeout_ms` bound is constant in N, but
  the risk of *approaching* it rises with N because a larger server batch has longer service time
  (RELY §5: service time grows with batch rows / bucket), and because more competing threads (large T)
  lengthen the queue.

### 3.3 No other blocking points in the transport

- `issue_one` (`:434-452`), `is_ready` (`:427-430`), `submitted[]` updates, `inflight_msgs`
  arithmetic, the `inflight_.emplace`/`find`/`erase` (`wire_leaf_pool.hpp:92,115,120`): all pure local
  CPU, **no wait**.
- `recv_batch`'s `inflight_.find` is an `unordered_map` lookup, not a wait (`wire_leaf_pool.hpp:115`).
- There is **no `zmq_poll`** in the producer path (only the server uses a `Poller`). The producer never
  does an event-loop wait; it does exactly one blocking primitive (`zmq_msg_recv`) and the rest is
  busy synchronous work. So the producer cannot livelock on a poll; it either has work, is sending
  (non-blocking under `D≤1000`), or is blocked in the single bounded recv.
- `RedisClient::create()` and `wredis.write_results` (`:319-321,359`) are off-path for the transport
  boundary (a different socket to a different service); they are episode-completion side effects, not
  part of the DEALER↔ROUTER protocol. They are not modeled as transport states (they cannot block the
  leaf-eval round-trip; they happen only on `RESUME`→episode-end).

---

## 4. The correlation-id inflight bookkeeping (exact)

`corr_seq` is a single shared `std::atomic<uint64_t>` (`:298`). Every `submit_batch` does
`corr = corr_seq->fetch_add(1, relaxed)` (`wire_leaf_pool.hpp:84`) → **globally unique, monotone
correlation ids across all T threads.** Per send, the thread `inflight_.emplace(corr, slots)`
(`:92`) recording the EXACT vector of slot indices that went into that message (in gather order
`:441`).

On reply: `recv_corr_payload` extracts `corr` from frame 0 (`wire_leaf_pool.hpp:162`);
`recv_batch` does `inflight_.find(corr)` (`:115`):
- **not found** → error "unknown correlation id … (a desynchronized wire)" (`:116-118`) → ERR. This is
  the producer's protection against a misrouted/duplicated/stale reply.
- **found** → `slots = move(it->second); inflight_.erase(it)` (`:119-120`). Then it REQUIRES
  `decoded->size() == slots.size()` (`:121`) else error "reply carried X predictions for a batch of Y"
  → ERR. So the producer enforces that the server returns **exactly as many predictions as the
  request carried rows**, in order: `out[i].slot = slots[i]` pairs the i-th prediction with the i-th
  gathered slot (`:126-128`).

**Invariant `|inflight_| == inflight_msgs` (per thread):** `submit_batch` does exactly one `emplace`
and the driver does exactly one `++inflight_msgs` per successful send (`:448`); `recv_batch` does
exactly one `erase` and the driver one `--inflight_msgs` per reply (`:460`). The only ways out of the
loop are error (ERR) or `inflight_msgs==0` (DONE). So in every non-error reachable state the map size
equals the counter. (Proof obligation discharged for the Z3 check in §9.) **Corollary:** `inflight_`
holds ≤ D entries; the per-thread correlation-map memory is O(D), independent of N. (N inflates the
`vector<int> slots` per entry up to K, so per-entry size is O(K)=O(N·base), but entry COUNT is O(D).)

**Cross-thread aliasing is impossible** because (a) each thread's `inflight_` is private and (b) corr
ids are globally unique, so even an erroneously delivered foreign reply would `find`-miss → loud ERR,
never a silent slot mismatch.

---

## 5. ASSUME–GUARANTEE contract

### RELY (what the producer assumes about the peer over the wire — each checkable against the server code)

R1. **Reply envelope = `[corr | payload]` (≥2 frames, leading frame exactly 8 bytes).** The producer
needs `frames.size()>=2 && frames.front().size()==sizeof(uint64_t)` (`wire_leaf_pool.hpp:157`).
*Checkable:* the ROUTER receives `recv_multipart` → `frames=[ident, *envelope, payload]`
(`inference_server.py:177-181`); the DEALER's send was `[corr | reqpayload]`, so on the ROUTER
`ident` is the DEALER routing id (added by ROUTER), `envelope = frames[1:-1] = [corr]`, `payload =
frames[-1] = reqpayload`. The server replies `send_multipart([ident, *envelope, resp])`
(`inference_server.py:200`; stage_a `:70`) = `[ident, corr, resp]`; the ROUTER strips `ident` on the
wire, so the DEALER receives `[corr, resp]` — exactly R1. ✓

R2. **`corr` echoed unchanged and uniquely.** The server treats `envelope` opaquely and echoes it
(`:197-200`). It never invents or drops a corr. *Checkable:* envelope flows untouched from drain to
send. ✓ (Producer enforces via `find`/erase; a violation → loud ERR, not corruption.)

R3. **`resp` is a valid wire-v2 response with `B == request's B` predictions in request row order.**
Producer enforces `decode_response` validity (`wire_leaf_pool.hpp:111`) and `decoded->size() ==
slots.size()` (`:121`). *Checkable:* `run_microbatch` slices `v[off:off+n]` per request using
`counts` = each request's row count and emits one `encode_response` per request
(`inference_server.py:55-72`), pad rows (`:58-59`) are discarded by `off:off+n`. So each reply has
exactly that request's B rows, in order. ✓ For the stage_a `wakeup="leaf"` variant, each request is
its own group/forward (`stage_a_server.py:57`) but still one response per request ident → still
B-matched. ✓

R4. **One reply per request, FIFO on the DEALER pipe, no spontaneous messages.** The server only ever
sends in response to a drained request (`_serve_batch` iterates `drained`); it sends exactly one
multipart per request ident. *Checkable:* `serve_forever` → `_drain` → `_serve_batch`
(`inference_server.py:219-225`); one `send_multipart` per drained request (`:197-200`). The producer's
`while inflight_msgs>0` loop assumes each `recv_batch` consumes exactly one outstanding message; R4
guarantees the bijection. ✓ (Ordering: ROUTER may reply to drained requests in drain order, possibly
batching several of THIS thread's corr-ids into one forward but still one reply-message each; the
DEALER receives them FIFO; the producer matches by corr regardless of order — so even out-of-order
replies are fine, `find` handles it.)

R5 (**SINK SERVICE TIMING — the timing RELY**). After the server has ≥1 of this thread's requests
queued, it produces the reply after a **nondeterministic positive service time** `S` with this
derivable structure:
- The server BLOCKS until ≥1 request is queued: `_drain` polls with `_POLL_INTERVAL_MS=100`
  (`inference_server.py:142,163-166`) and returns only when `poll()` is truthy. So a reply cannot
  precede the request that triggers the drain — **causal: reply-after-request**. Empty polls add
  multiples of ~100 ms of *latency before the forward*, but never below the arrival of a request.
- It drains **all currently-queued requests up to `max_batch` rows** in ONE forward
  (`:171-186`): `total_rows < self._max_batch` accumulates `X.shape[0]`. So one forward can serve many
  requests (from this thread and the other T−1) — coalescing. The producer cannot observe how many
  peers were coalesced; it only sees its own reply.
- The forward is **`run_microbatch` padded to a fixed shape** (`:198`, `pad_to=self._max_batch`).
  Production drain: ONE forward per drained group padded to `max_batch` rows — so service time is
  **roughly constant in the real row count** (fixed-shape compile: `jax.jit` over a fixed
  `(max_batch,in_dim)` shape, `inference_server.py:22-34`); the JITted function is compiled once per
  shape and cached (`_jit_forward_cache`), so the first forward of a new shape pays compile latency
  (warmup at `stage_a_server.py:82` pre-compiles the buckets+max_batch to amortize it).
- Stage_a variant: service time depends on `E-policy`/`wakeup` (`stage_a_server.py:54-70`):
  `padmax` → `pad_to=max_batch` (constant-shape, like production); `bucket` → `pad_to =
  _bucket_for(real) ∈ {64,256,512}` (`:32-37,61-64`) → service time is a STEP function of the real row
  count (one of three compiled shapes); `wakeup="leaf"` → one forward PER request (B≈1 each, smallest
  bucket) → service time per reply is the small-shape forward but there are more forwards.
- So `S = S_drain_wait + S_forward(shape) + S_scatter`, with `S_forward` a function of the
  **padded/bucketed shape**, NOT a constant and NOT an instant. Bounds: `S > 0` always; `S_forward`
  monotone non-decreasing in the chosen shape; under fixed-shape (`padmax`) `S_forward` is
  shape-invariant (one compiled size). The producer models `S` as bounded nondeterminism: positive,
  finite (else its own `RCVTIMEO` fires), otherwise unconstrained except by the shape-dependence above.
- **N-dependence of `S` as seen by the producer:** larger N ⇒ this thread's single message carries
  more rows (up to K) ⇒ pushes the server's per-forward row count up ⇒ more likely to hit a larger
  bucket / fill `max_batch` / split across forwards if rows exceed `max_batch` *within one drain* (the
  drain stops accepting new requests once `total_rows >= max_batch`, but a single already-accepted
  request is never split — so a single message with `>max_batch` rows is forwarded whole, `pad_to >
  B` is false so no padding, `:58`). Thus as N grows the server runs at a *larger* effective batch,
  raising `S_forward` toward its max-batch value but lowering forwards-per-leaf (better amortization).
  The producer cannot control this; it only relies on `0 < S < ∞`.

### GUARANTEE (what the producer provides to the server)

G1. **Every request is a valid wire-v2 request: `[corr(8B) | payload]`, payload =
`version(=2) | B(u32) | in_dim(u32) | B*in_dim f32`, with `B≥1`, `in_dim≥1`, exact length.** Enforced
by `encode_request` (`inference_wire.hpp:51-70`) which refuses `B==0`/`in_dim==0`/size-mismatch, and
`submit_batch` always passes `B=gathered.size()≥1` (guarded by `issue_one`'s `gathered.empty()→return`
`:444`) and `in_dim=feat_dim≥1`. The corr frame is exactly `sizeof(uint64_t)=8` bytes
(`wire_leaf_pool.hpp:86`). *Server-checkable:* `decode_request` requires these exact invariants
(`inference_wire.py:42-61`) and would reject otherwise — but the producer guarantees they hold, so the
server never `_reject`s a producer request in normal operation.

G2. **Globally unique, never-reused corr ids; ≤ D outstanding messages per DEALER.** `fetch_add`
monotone (`:84`); the producer never reuses an id (it erases on reply and never re-emplaces the same
key — keys are strictly increasing). ≤ D from the loop guards (`:456,474`). So the server's echoed
corr is always fresh; no aliasing.

G3. **The producer will eventually `recv` every reply it is owed (it does not abandon outstanding
requests except on hard error/timeout).** The `while inflight_msgs>0` loop (`:457`) keeps recv-ing
until the count hits 0. On error/`failed`, LINGER=0 lets it tear down without flushing — so the
guarantee is "drains to completion OR fails loudly", which matches R4's bijection (the server is never
left with a request whose requester silently vanished while healthy).

G4. **One feature row per slot per message; the same slot is never double-submitted while
outstanding.** `submitted[s]` de-dup (`:447,466`) and `is_ready`'s `!submitted[s]` (`:429`). So the
server never sees two concurrent requests for the same logical leaf from one thread.

---

## 6. DEGREES OF FREEDOM (each with code_ref, behaviors admitted, N-dependence)

**DOF-T1 — Source emission interval (when a parked slot next yields a leaf).**
`code_ref:` `fiber_leaf.hpp:24-29` (predict yields), `:175,420,468` (`running` after spawn/resume),
`advance` `:389-397`. *Admits:* any positive, unfixed interval between a slot's resume and its next
park; a slot may park 0 times (episode ends in a synchronous `advance` burst, `:469`) or many times
per episode. *N-dependence:* with K=N·base slots, the **set** of simultaneously-parked slots at any
instant is a nondeterministic subset of size 0..K; larger N widens this set, so `issue_one` can gather
more rows per message. The *individual* slot's interval is N-independent; the aggregate arrival
process per thread scales ∝ K.

**DOF-T2 — Gather composition / batch size of a message.**
`code_ref:` `issue_one` `:434-444` (gathers ALL currently-ready slots), `is_ready` `:427-430`.
*Admits:* a single message carries any subset of the ready slots from 1..K rows — exactly those that
are `active && running && !submitted` at the scan instant. Because the scan is synchronous over all K,
it captures a *snapshot*; slots that park a microsecond later wait for the next `issue_one`.
*N-dependence:* max rows-per-message = K = N·base, so message fatness scales **linearly in N**. This
is the dominant N effect: bigger N → fewer, fatter messages (mean_rows_per_msg rises, `:496`).

**DOF-T3 — Pipelining depth actually used (1..D messages in flight).**
`code_ref:` prime loop `:456`, refill `:474`, cap `D` `:287`. *Admits:* between 1 and D outstanding
messages. **Crucial interaction with DOF-T2:** if at PRIME time ALL ready slots are gathered into ONE
message (the all-ready regime), the very next `issue_one` finds `gathered.empty()` → returns false
(`:444`), so **only 1 message is in flight** even though `D>1`. Multiple messages in flight arise ONLY
when ready slots appear in *waves* (some parked now, others park while the first message is on the
wire) — i.e. when source timing is staggered. *N-dependence:* larger N ⇒ more slots ⇒ at startup a
larger first wave (so the first message is huge and D-utilization stays near 1); but during steady
state more slots are mid-search at staggered times, so more partial waves → D-utilization rises toward
D as N grows. Net: small N tends to under-use D (one big message); large N tends to fill D with
medium messages. The cap D bounds in-flight messages regardless of N.

**DOF-T4 — Send-blocking regime (HWM).**
`code_ref:` `wire_leaf_pool.hpp:86-91` (no DONTWAIT, no SNDTIMEO, no SNDHWM set → default 1000).
*Admits:* a non-blocking send in every reachable state when `D ≤ 1000` (the operative regime); an
indefinitely-blocking send (`SNDTIMEO=-1`) only in the unreachable-by-default regime `D > 1000` with a
stalled server. *N-dependence:* **none in message count** — HWM is messages and the cap is D, constant
in N. Byte-queue per message grows ∝ N but ZMQ HWM does not count bytes, so no HWM approach as N grows.

**DOF-T5 — Recv timeout firing (the liveness backstop).**
`code_ref:` `wire_leaf_pool.hpp:41,147-151` (`RCVTIMEO`, blocking recv → EAGAIN → ERR). *Admits:* the
thread aborts loudly iff a reply takes longer than `timeout_ms`. *N-dependence:* the *probability* of
firing rises with N and T (bigger server batches → longer `S_forward`; more threads → longer queue),
but the *bound itself* (`timeout_ms`) is N-independent. This is the one place N can flip a
correct execution into an ERR execution (a too-aggressive N/T against a slow forward).

**DOF-T6 — Reply arrival order vs. submit order.**
`code_ref:` corr-matching `wire_leaf_pool.hpp:115` (lookup by id, order-agnostic). *Admits:* replies
for this thread's outstanding messages arriving in ANY order relative to submit order; the producer
matches by corr regardless. (Within one DEALER pipe ZMQ is FIFO, and the server replies one message
per request, so in practice order follows the server's forward schedule; the producer is robust to any
permutation.) *N-dependence:* with more in-flight messages (larger N→higher D-utilization, DOF-T3),
more reorderings are *possible*; the producer's tolerance is unchanged.

**DOF-T7 — Drain variant on the peer (affects RELY R5 shape only).**
`code_ref:` `inference_server.py:192-200` (production padmax/group) vs `stage_a_server.py:54-70`
(E-policy ∈ {padmax,bucket}, wakeup ∈ {group,leaf}). *Admits:* service time `S` as constant-shape
(padmax), 3-step (bucket), or per-request (leaf). The producer's local state machine is IDENTICAL
across variants (it only ever sees one matched reply per message); only the *timing* `S` differs.
*N-dependence:* under `bucket`, larger N pushes `real` row count into higher buckets → larger `S`
steps; under `padmax`, `S` is N-flat (always max_batch shape); under `leaf`, N multiplies forwards
(one per row) so server throughput, not per-reply `S`, degrades with N.

---

## 7. TIMING MODEL (summary)

- **Source emission** (per parked slot): nondeterministic positive interval `δ_s > 0` between resume
  and next park, set by the search's internal progress (`fiber_leaf.hpp:24-29`), unfixed by code. Per
  thread the offered arrival process is the superposition of up to K=N·base such streams.
- **Sink service** (per message): `S = S_wait + S_forward(shape) + S_scatter`, `S>0`, finite. Shape ∈
  {pad-to-max (constant), bucket {64,256,512} (3-step), per-leaf small} per DOF-T7; `S_forward`
  monotone non-decreasing in shape; first-of-shape pays JIT compile unless warmed
  (`stage_a_server.py:82`). Modeled as bounded nondeterminism.
- **Causal constraints (hard):** (a) a reply cannot precede the request that produced it — the server
  blocks on `_drain` until a request is queued (`inference_server.py:163-166`) and only forwards
  drained rows; (b) a thread cannot emit a reply-dependent request for slot `s` before that slot's
  reply: `resume_with(c.pred)` (`:467`) must run before the resumed search can reach its next
  `predict` and re-become `is_ready` — `submitted[s]` stays 1 until the reply clears it (`:466`); (c)
  `inflight_msgs` never exceeds D and a recv strictly precedes the matching `--inflight_msgs`
  (`:458-460`); (d) feature span must be copied (`:439-440`) before the slot is resumed (the span is
  stack-backed, `fiber_leaf.hpp:16`).
- **Nothing collapsed to a constant.** Both source `δ_s` and sink `S` are kept as bounded
  nondeterministic positive durations. The ONE finite *constant* in the producer is the RECV bound
  `timeout_ms` (`RCVTIMEO`), which is a config constant (DOF-T5), not a collapse of a nondeterministic
  duration. The server poll granularity `_POLL_INTERVAL_MS=100` is a server constant folded into
  `S_wait`'s lower-bound quantization, not into the producer state.

---

## 8. REPRESENTATIVE EXECUTIONS (concrete traces; see structured object for full step lists)

1. **All-ready single-shot (small staggering):** FILL leaves all K slots parked → first `issue_one`
   gathers all K into one message (B=K) → only 1 in flight despite D>1 → block in recv → one reply of
   K predictions → resume all → refill. Self-reinforcing at large N (the all-ready snapshot grows with
   K). Exercises DOF-T2 (max fatness) + DOF-T3 (D under-used).
2. **Staggered waves fill D:** slots park in waves; PRIME issues message 1 (subset), more slots park,
   REFILL issues 2..D before the first reply returns → D messages in flight → drain one, refill one
   (steady pump at depth D). Transient→self-reinforcing as N grows (more partial waves). Exercises
   DOF-T3 (full D) + DOF-T6 (reorder possible).
3. **Timeout abort:** D messages in flight, server batch huge/slow (large N·T), `S > timeout_ms` →
   `zmq_msg_recv` EAGAIN → ERR → `set_error` → all threads observe `failed` and abort. Transient;
   more reachable as N (and T) grow (DOF-T5).
4. **Desync guard:** a reply carries a corr not in `inflight_` (a RELY-R2 violation) → "unknown
   correlation id" ERR; or a reply size ≠ slots size (RELY-R3 violation) → size-mismatch ERR. Not
   producible by the conformant server; included to show the producer's fail-loud boundary (DOF-T6
   matching is the guard).

---

## 9. Z3 confirmation (confirmation only, never the source of trust)

A minimal bounded encoding of one thread's pump (`out/producer_check.py`) asserts the
representative steady-state execution (#2): K slots, cap D, the invariant `|inflight_| ==
inflight_msgs ≤ D`, reply-after-request causality, and `submitted[]` de-dup, and asks Z3 for a
satisfying interleaving of a few send/recv steps. SAT ⇒ the trace is admissible under the modeled
constraints.

**Run (`out/producer_transport_check.py`, `nice -n 19 timeout 90`, z3 4.16):** `RESULT: sat`. The
returned trace reaches full depth D=3 and drains fully:

```
step 0: SEND corr=0 inflight=1
step 1: RECV corr=0 inflight=0
step 2: SEND corr=1 inflight=1
step 3: SEND corr=2 inflight=2
step 4: SEND corr=3 inflight=3   <- pipeline at cap D
step 5: RECV corr=1 inflight=2
step 6: RECV corr=3 inflight=1   <- reply out of submit order (DOF-T6 admissible)
step 7: RECV corr=2 inflight=0   <- fully drained
```

This confirms a pipelined send/recv interleaving up to depth D, fully drained, respecting cap-D,
`|inflight|==inflight_msgs`, monotone-unique corr, and reply-after-request — and that out-of-order
reply matching (corr 1,3,2) is admissible, as DOF-T6 claims. Confirmation only; trust is in §1–§7.

---

## 10. DOF-CONTROL notes (what each constraint forbids; what removing it would wrongly admit)

- **Cap D (`:456,457,474`)** — removing it (unbounded in-flight) would wrongly admit executions where
  a thread sends all K messages then blocks forever in recv with no backpressure, and would let
  `inflight_msgs` exceed any HWM. Keeping it makes >D-in-flight executions unrepresentable.
- **`submitted[]` de-dup (`:447,466,429`)** — removing it would wrongly admit two concurrent requests
  for the same parked leaf (double-counting a slot, violating G4 and breaking the B-match RELY R3 on
  resume). Keeping it makes double-submit-while-outstanding unrepresentable.
- **corr-find/erase + size check (`wire_leaf_pool.hpp:115-124`)** — removing it would wrongly admit
  silently mismatched slot↦prediction pairings. Keeping it makes "accept a reply whose B≠request B or
  whose corr is unknown" unrepresentable (forces ERR).
- **`RCVTIMEO` bound (`:41`)** — removing it (infinite recv) would wrongly admit an execution that
  hangs forever on a dead server; keeping it makes "wait > timeout_ms for a reply" unrepresentable
  (forces ERR). This is the deliberate liveness/fail-loud boundary.
- **reply-after-request + copy-before-resume causality (server `:163-166`; `:439-440`)** — removing
  either admits acausal traces (a prediction before its features were sent; a stale span). Keeping
  them makes those unrepresentable.

---

## 11. FIDELITY SELF-AUDIT

**Possible over-permissions (places I may admit more than the code can do):**
- I model reply arrival order as a free permutation (DOF-T6). The producer code is order-agnostic, so
  this is faithful for the producer's STATE; but on a single DEALER↔ROUTER pipe ZMQ is FIFO, so the
  *actual* reachable orderings are a subset. I deliberately left the latitude on the producer side
  (it cannot distinguish), flagging that the realized order is constrained by the peer/transport.
- I treat `SEND` as potentially indefinitely-blocking only in the `D>1000` regime. If a real
  deployment never sets `D>1000`, that blocking transition is dead; I keep it because the code's
  `SNDTIMEO=-1`/`SNDHWM=1000` defaults permit it for some configuration.
- `S` as bounded-but-otherwise-free nondeterminism may admit timing combinations the JIT/CPU cannot
  physically realize; I constrained only what the code/causality fix (positivity, shape-monotonicity,
  finiteness), per the assignment's instruction to leave exactly the latitude the code leaves.

**Possible over-constraints (places I may forbid something the code can do):**
- I asserted `|inflight_| == inflight_msgs` as an invariant. It holds on every non-error path, but on
  the ERR path between a failed mid-loop `set_error` and thread teardown the counter and map can be
  momentarily inconsistent (we break without reconciling). I treat ERR as terminal, so this transient
  is outside the modeled steady states — if a downstream consumer needs the exact ERR-teardown
  micro-states, this is the one place to relax.
- I model the producer as seeing **one matched reply per outstanding message**. If the server ever
  coalesced two of this thread's messages into one reply frame, the producer would mis-decode — but
  R4 (server sends one reply per request ident) forbids that, so I did not add a "merged-reply" state.
  If a future server variant batched replies, this model would be over-constrained and need a new
  transition.
- I do not model `RedisClient` write blocking as a transport state. It can block (a Redis stall) and
  thereby delay when a finished slot's successor becomes `is_ready`, indirectly affecting arrival
  timing. I judged it off the leaf-eval transport boundary; if the boundary is taken to include
  "time until the next leaf is offered," this omission slightly under-models a source-timing input.

---

## 12. n_dependence_summary

The single dominant N-effect: **N inflates the per-message ROW count (up to K = N·base rows per
message) while leaving the in-flight MESSAGE count capped at D, constant in N.** Consequently, as N
grows the producer makes fewer, fatter round-trips: outbound BYTES per message scale ∝ N, the
correlation-map ENTRY count stays O(D) (each entry's slot-vector grows O(K)), HWM proximity does not
grow (HWM counts messages, and messages ≤ D ≪ 1000), and the blocking surface concentrates into the
single `RCVTIMEO`-bounded recv — whose firing probability (not its bound) rises with N and T because
larger server batches lengthen the sink service time `S_forward`. Pipelining depth utilization
(messages actually in flight, 1..D) rises from ~1 (one giant all-ready message) toward D as N grows
and source timing staggers the waves.
