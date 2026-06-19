# A faithful model of the C++ producer side of the leaf-evaluation transport boundary

**Scope:** the producer (source) side — the search source that emits leaf rows, the per-thread DEALER
`WireLeafPool`, and the pipelined driver `run_episodes_wire_pipelined` (and its strict-barrier twin
`run_episodes_wire_batched`). Derived FORWARD from the operational semantics of the code, end-to-end-read
files only. No outside expectation of behavior is introduced.

**Files read end to end (ADR-0002):**

- `cpp/include/chocofarm/wire_leaf_pool.hpp` (1–243, full)
- `cpp/src/runner_wire_batched.cpp` (1–630, full — both drivers)
- `cpp/include/chocofarm/inference_wire.hpp` (1–226, full)
- `cpp/include/chocofarm/wire_spec.hpp` (1–58, full — the layout the codec derives)
- `chocofarm/az/inference_server.py` (1–457, full — the peer, for RELY)
- `chocofarm/az/inference_wire.py` (1–185, full — the peer codec, for RELY)
- `chocofarm/az/forward.py` (1–63, full — the sink service body, for the service-time model)
- `docs/design/zmq-inference-service.md` (1–367, full — protocol intent + amendments)
- `cpp/include/chocofarm/fiber_tree.hpp` (1–111, full — `start`/`resume_with`/`ch.features`/`running`,
  the source-emission generator)
- `cpp/include/chocofarm/runtime_config.hpp` (1–46, full — `fibers_per_thread = ceil(batch/threads)`)
- `cpp/include/chocofarm/runner_wire_batched.hpp` (1–100, full — `WireRunnerConfig` defaults, `WireMode`)
- grep-confirmed (then read at the hit) the COMPLETE set of ZMQ socket/context option call sites in
  `cpp/include` + `cpp/src` (the blocking-surface determinant).

---

## 0. The one fact that fixes the whole blocking surface: which socket options are set

The producer's DEALER socket is opened in exactly one place, `WireLeafPool::create`
(`wire_leaf_pool.hpp:71–91`). The COMPLETE set of options touched, verified by grepping every
`zmq_setsockopt` / `zmq_ctx_set` / `*HWM` / `*TIMEO` / `LINGER` / `IMMEDIATE` / `RECONNECT` call site in
`cpp/include` and `cpp/src`:

```
zmq_setsockopt(sock, ZMQ_LINGER, &linger=0, …)         // wire_leaf_pool.hpp:82
zmq_setsockopt(sock, ZMQ_RCVTIMEO, &timeout_ms, …)     // wire_leaf_pool.hpp:83
```

and the context is `zmq_ctx_new()` with **no** `zmq_ctx_set` anywhere
(`runner_wire_batched.cpp:87,394`). Therefore, **every** other option is at its libzmq default:

| option | producer DEALER value | source |
| --- | --- | --- |
| `ZMQ_LINGER` | 0 (set) | `wire_leaf_pool.hpp:82` |
| `ZMQ_RCVTIMEO` | `timeout_ms` (set; default 15000 from `WireRunnerConfig::timeout_ms`) | `wire_leaf_pool.hpp:83`, `runner_wire_batched.hpp:69` |
| `ZMQ_SNDTIMEO` | **−1 (default, infinite)** — NOT set | no call site |
| `ZMQ_SNDHWM` | **1000 (default)** — NOT set anywhere (socket or context) | no call site |
| `ZMQ_RCVHWM` | **1000 (default)** — NOT set | no call site |
| `ZMQ_IMMEDIATE` | 0 (default — queue while connecting) | no call site |
| `ZMQ_RECONNECT_IVL` | 100 ms (default — auto-reconnect on) | no call site |
| `ZMQ_IO_THREADS` | 1 (context default) | `zmq_ctx_new` with no `zmq_ctx_set` |
| `ZMQ_MAX_SOCKETS` | 1023 (context default) | same |

This determines the entire blocking analysis below, and it is the single most fidelity-critical fact:

- **`zmq_send` (submit_batch) blocks only when the send queue is full** (SNDHWM=1000 messages), and then
  blocks **indefinitely** (SNDTIMEO=−1). It does not time out. Against the live server this practically
  never fills (the server drains continuously and a single thread holds ≤ D=8 messages outstanding, far
  below 1000), so `submit_batch` is effectively non-blocking — but a model that *forbids* the send from
  blocking would be too permissive in the corner where the server stalls. I model the send as
  `may-block-unboundedly-when-SNDHWM-reached`, with the queue-full predicate derived from causality.
- **`recv` (recv_batch/poll) blocks up to `timeout_ms`** then returns EAGAIN, which the codec turns into a
  loud typed error (`recv_corr_payload` → `WireLeafPool::poll: zmq_msg_recv failed`, `wire_leaf_pool.hpp:217–220`).
- **There is no `ZMQ_CONNECT` failure at create time** for `ipc://`: the connect is lazy
  (`wire_leaf_pool.hpp:65–70` NB; the header states a dead endpoint surfaces only as a recv timeout, not a
  hang). So the producer can enter the loop, submit, and only discover a dead sink at the recv timeout.

---

## 1. The system the producer participates in

The producer is `T` OS worker threads (`run_episodes_wire_pipelined`, `runner_wire_batched.cpp:390`,
`T = max(1, pool_threads)`, default 4). Each thread is independent: its own `WireLeafPool` (one DEALER
socket, `:421–423`), its own `RedisClient`, its own `FeatureBuilder`, its own disjoint episode subset
`{tid, tid+T, …}` (`:431`), and `K = ceil(pool_batch/pool_threads)` `EpisodeSlot`s (`:391`,
`runtime_config.hpp:29–32`; default `pool_batch=32, pool_threads=4 ⇒ K=8`). The threads share only the
process-global `corr_seq` atomic (`:403`, borrowed by reference into each pool) and the failure flags
(`failed`/`have_error`/`first_error`/`err_mu`).

Each slot holds one `TreeState` fiber (`fiber_tree.hpp:42`). A fiber, when resumed, runs the Gumbel search
until it next needs a leaf evaluation, at which point it yields with `ch.at_leaf=true` and its feature row
in `ch.features` (`fiber_tree.hpp:88–107`, `running = ch.at_leaf`). The driver feeds the prediction back
via `resume_with` (`:103–107`) and the fiber runs to its next leaf or returns a `Decision`.

The two drivers differ ONLY in transport schedule (the per-slot episode state machine — `spawn_ply`,
`finalize_and_write`, `apply_decision`, `advance`, `fill` — is line-for-line identical, `:441–443`
comment). This model treats the **pipelined** driver as the general case (`D ≥ 1`), and the strict-barrier
driver as the special case `D = 1` with a structural "submit-ALL-parked-then-block-for-the-one-reply" round
(`:310–337`). Where they diverge I flag it.

---

## 2. The producer state machine

I model ONE worker thread (the threads are independent modulo the shared `corr_seq` and the fail flag; §6
DOF-7 covers cross-thread interleaving). Within a thread there are two nested levels of state: (a) the
**per-slot** episode/fiber state, and (b) the **per-thread** transport pipeline state. The transport
boundary lives at level (b); level (a) is the *source-emission generator* feeding it.

### 2a. Per-slot fiber state (the source of leaf rows)

| state | meaning |
| --- | --- |
| `IDLE` | slot not active: `sl.active=false` (exhausted subset, or between episodes). `is_ready` false. |
| `PARKED` | `sl.active && sl.ts && sl.ts->running && !submitted[s]`: parked at a leaf, row in `ts->ch.features`, NOT yet sent. This is the only `is_ready`-true state (`:541–544`). |
| `INFLIGHT` | `submitted[s]=1`: its row is out on the wire under some corr-id, awaiting a reply. Not re-gatherable. |
| `RUNNING` | transiently inside the fiber (between `resume_with`/`start` and the next `running` read) — the source's internal search work. Not observable as a stable driver state; it is the *duration* the source consumes before re-entering PARKED or finalizing. |

Slot transitions (all in the per-slot lambdas, shared by both drivers):

| from | to | guard | action | code_ref | free? |
| --- | --- | --- | --- | --- | --- |
| IDLE | PARKED | `next_idx < cfg.episodes`, episode not immediately-finalizing | `fill`: seed-fold, world-draw, `spawn_ply`; if `ts->running` parked | `:511–538` | det. (RNG-exact draws) |
| IDLE | IDLE | `next_idx ≥ cfg.episodes` | `fill` returns false; slot stays idle | `:513,537` | det. |
| PARKED | INFLIGHT | `is_ready(s)` and this slot chosen into the coalesced gather | `issue_one`: `submitted[s]=1` | `:551–569` | **det. given which slots issue_one gathers**; the *set* is "ALL currently-ready" |
| INFLIGHT | PARKED | reply for this slot's corr-id arrives; `resume_with`; `ts->running` still true (next leaf) | `:588–590` | det. given the reply |
| INFLIGHT | (apply) | reply arrives; `ts->running` false (Decision) → `advance` | `:591` | det. |
| (apply) | PARKED | `advance` spawns next ply that parks | `:502–510,591` | det. |
| (apply) | IDLE | `advance`/`apply_decision` finalizes (TERMINATE/horizon/empty); then `fill` | `:592–593` | det. |

The source-emission nondeterminism is entirely in the **duration** of the INFLIGHT→PARKED and IDLE→PARKED
and (apply)→PARKED edges: how long the fiber runs before its next leaf yield. The code fixes the *order* of
RNG draws (byte-identical to serial `run_episode`) but NOT the wall-clock interval. See §3.

### 2b. Per-thread transport pipeline state (the boundary)

State variable `inflight_msgs ∈ {0…D}` (`:438`), plus the per-slot `submitted[]` vector, plus the pool's
`inflight_` map (corr-id → ordered slot list, `wire_leaf_pool.hpp:239`). Note `inflight_msgs` counts
MESSAGES (corr-ids), each coalescing `S ≥ 1` slots; `inflight_.size() == inflight_msgs`.

| state | meaning |
| --- | --- |
| `PRIMING` | initial fill of K slots then `while (inflight_msgs < D && issue_one())` (`:572,578`). |
| `PIPE_FULL_OR_DRAINING` | `inflight_msgs > 0`: the main loop. Blocks in `recv_batch` for one reply. (`:579`) |
| `BLOCKED_RECV` | inside `pool.recv_batch()` → `zmq_msg_recv` (`wire_leaf_pool.hpp:217`): blocked up to RCVTIMEO. |
| `BLOCKED_SEND` | inside `pool.submit_batch` → `zmq_send` when SNDHWM reached (SNDTIMEO=−1): blocked indefinitely. Practically unreachable against a live server; kept for fidelity. |
| `SCATTERING` | iterating `*reply` completions, resume/advance/fill each (`:584–594`). |
| `REFILLING` | `while (inflight_msgs < D && issue_one())` after a reply (`:596`). |
| `DRAINED` | `inflight_msgs == 0` and loop predicate `inflight_msgs > 0` fails → worker exits drain, writes telemetry (`:598–599`). |

Transport transitions:

| from | to | guard | action | code_ref | free? |
| --- | --- | --- | --- | --- | --- |
| PRIMING | PRIMING | `inflight_msgs < D` and some slot ready | `issue_one()` → +1 inflight_msgs | `:578,565` | det. given ready set |
| PRIMING | PIPE | `inflight_msgs == D` OR no slot ready | enter `while inflight_msgs>0` | `:578–579` | det. |
| PIPE | BLOCKED_RECV | `inflight_msgs>0 && !failed` | call `recv_batch` | `:580` | det. |
| BLOCKED_RECV | SCATTERING | a reply frame arrives within RCVTIMEO, envelope valid, corr-id known, count matches | decode, `--inflight_msgs`, loop completions | `:580–582`, `wire_leaf_pool.hpp:170–196` | det. given the reply (but WHICH corr-id replies first is a **free choice** the sink owns — §RELY) |
| BLOCKED_RECV | ABORT | RCVTIMEO fires (EAGAIN) / malformed envelope / unknown corr-id / count≠ | `set_error`, break | `:581`, `wire_leaf_pool.hpp:180–188,227–231` | det. given the bad/absent reply |
| SCATTERING | (per slot) PARKED/apply/IDLE | per-completion `resume_with` then re-park/advance/fill; `submitted[s]=0` | `:584–594` | det. given preds |
| SCATTERING | REFILLING | completions exhausted | `while inflight_msgs<D && issue_one()` | `:596` | det. |
| REFILLING | BLOCKED_SEND | a coalesced gather is sent and SNDHWM is reached | `submit_batch` blocks | `wire_leaf_pool.hpp:139–144` | det. given queue full (free: when sink stalls) |
| REFILLING | PIPE | refilled back to D, or nothing ready | continue outer loop | `:596–597` | det. |
| PIPE | DRAINED | `inflight_msgs == 0` (all replies consumed, nothing re-issued because no slot ready / subset exhausted) | exit loop | `:579,598` | det. |
| any | ABORT | `failed.load()` set by THIS or ANOTHER thread | break out, return unexpected | `:579,581,592`, `:609–614` | det. given the flag (free: another thread's timing) |

**Strict-barrier specialization (`run_episodes_wire_batched`, `:310–337`).** `D` is structurally 1 and the
"message" coalesces *all* currently-parked slots (`any_parked` / the gather over all `ts->running` slots,
`:303–320`). The transport states collapse to: gather-all → `submit_batch` (one corr-id) → `recv_batch`
(block for the one reply) → scatter-all → re-gather. So BLOCKED_RECV is entered exactly once per round and
the thread idles the full round-trip each round (`runner_wire_batched.hpp:50–53` "D=1 outstanding/thread").

---

## 3. The timing model (source emission + sink service)

Timing is modeled as **bounded nondeterminism over a partial order of events**, constrained only by
positivity and the causal necessities the code/transport impose. No interval is collapsed to a constant.

### 3.1 Source emission (when the next request is ready)

A slot's next leaf row becomes available when its fiber, resumed at time `t_resume`, runs the search to its
next `ch.at_leaf` yield. Call that duration `δ_search(slot, ply)`.

- **Representation:** `δ_search > 0`, otherwise unconstrained (a *nondeterministic positive interval*, NOT
  a constant). The code fixes the RNG draw *order* (byte-identical to serial `run_episode`,
  `runner_wire_batched.cpp:8–9` purpose) but says nothing about wall-clock duration; the search's work per
  leaf varies with belief size, legal-action count, Sequential-Halving budget, and OS scheduling on the
  shared 4-vCPU host. Modeling it as a constant would forbid the real executions where two slots' rows
  become ready at different, data-dependent times — which is exactly what makes coalescing-degree `S` vary.
- **Causal constraints (necessities):**
  - `δ_search > 0` (positivity; a fiber resume is real CPU work).
  - **A reply-dependent row cannot be emitted before its reply.** A slot enters INFLIGHT; its row for the
    *next* leaf is computed inside `resume_with(c.pred)` (`:589`), which only runs after `recv_batch`
    delivered `c.pred`. So `ready(slot, ply+1) > recv_time(slot, ply)`. This is the central producer-side
    causal law: the source's emission of its next request is gated on its previous reply. (The FIRST leaf
    of a fresh episode, via `fill`→`spawn_ply`, is NOT reply-gated — it is ready after the seed/world draw,
    `:516–532`.)
  - **Coalescing is instantaneous-snapshot, not timed.** `issue_one` gathers exactly the set of slots that
    are `is_ready` *at the instant it runs* (`:554–560`). It is single-threaded within a worker, so "which
    slots are ready together" is fixed by the relative ordering of their `δ_search` completions vs. the
    points where `issue_one` is called (after PRIMING and after each scatter, `:578,596`). There is no
    timer, no threshold (`:298–300` records the rejected "flush at threshold" alternative). Hence `S`
    (rows per message) is `1 ≤ S ≤ K`, data-and-scheduling-determined, NOT a tunable.

### 3.2 Sink service (the forward, as the producer observes it)

The producer does not run the forward; it WAITS on the reply. But the wait duration is a function of the
sink's service time, which I model faithfully (in scope) so the producer's blocking is not abstracted to an
instant.

- **The forward is one `forward_core` matmul over a padded `(pad_to, in_dim)` matrix**
  (`inference_server.py:164–177`, `forward.py:50–63`). `pad_to = self._max_batch` (`:385`), so EVERY batch
  the server runs is padded UP to the SAME fixed shape `(max_batch, in_dim)` — `run_microbatch:171–172`.
  This is the load-bearing service-time fact: **because of fixed-shape padding the per-forward service time
  is ESSENTIALLY BATCH-SIZE-INDEPENDENT** for `B ≤ max_batch` — XLA compiles ONE executable
  (`jit_forward_core:100–104`, "pads every batch to one shape, so it compiles a SINGLE executable"), and
  the matmul is always `(max_batch, in_dim) @ …`. So service time `τ_fwd ≈ τ(max_batch)` regardless of how
  many real rows `B` the drain gathered.
- **Representation:** `τ_fwd > 0`, drawn nondeterministically from a bounded interval `[τ_min, τ_max]` that
  is *constant in B* (the padded-shape consequence) but NOT a single constant across calls (XLA/OS jitter,
  warmup state). The model leaves `τ_fwd` as a positive bounded nondeterministic duration. I do NOT collapse
  it to a constant: `MEMORY.md` records the real measure (~50 dps JAX-batched), and the design (§4) records
  run-to-run roundoff/jitter — both say the duration is real and variable.
- **Warmup removes the cold-compile confound but not the per-call variability.** `warmup(batch_sizes)`
  (`inference_server.py:389–426`) forces XLA to compile shape `(max_batch, in_dim)` before serving, so the
  first real forward is at steady state. With padding to one shape there is exactly one executable, so one
  warmup call suffices; the design comment that warmup exists "per exact batch size B" (`:397`) is the
  pre-padding rationale — under the current `pad_to=max_batch` code there is one B. The producer RELY does
  not depend on warmup having run (it only bounds `τ_fwd` by RCVTIMEO; a cold first forward that exceeds
  RCVTIMEO would surface as a loud timeout, which is admissible in my model).
- **One forward at a time (single-threaded server).** `InferenceServer` is single-threaded by construction
  (`inference_server.py:33–35,291–300,428–439`): `serve_forever` is a sequential loop, `_drain` then
  `_serve_batch`. So forwards are SERIALIZED: while batch K's forward runs, K+1's requests only QUEUE; they
  are not serviced concurrently. From the producer's side this means: replies to messages that landed in the
  same drain come back together (one `send_multipart` per request, `:387, in drain order`), and a message
  that arrived during a forward waits at least the remainder of that forward plus the next drain+forward.

- **Causal constraints (necessities) binding source and sink:**
  - `τ_fwd > 0` (a real matmul; `forward.py:50–62`, `run_microbatch:177` `np.asarray` forces completion).
  - **A reply cannot precede the forward that produced it.** `_serve_batch` sends responses only after
    `run_microbatch` returns (`:384–387`); `run_microbatch` returns only after the `np.asarray(forward_fn…)`
    device→host pull completes (`:177`). So `reply_send_time(msg) ≥ drain_time(msg) + τ_fwd_of_its_batch`.
  - **A reply cannot precede its request.** The server can only batch a request after `recv_multipart`
    dequeued it (`:350`), which is after the producer's `submit_batch` two-frame send arrived.
  - **The producer's reply-matching is by corr-id, order-agnostic.** The pipelined driver makes NO
    assumption that replies arrive in submit order — `recv_batch` routes by the echoed leading frame
    (`wire_leaf_pool.hpp:179–196`) and the comment at `runner_wire_batched.cpp:368–371` states out-of-order
    replies route to the right slot by corr-id "with no extra bookkeeping." So the model leaves the sink
    FREE to reply to outstanding corr-ids in any order consistent with the causal laws above.

---

## 4. Assume–Guarantee contract

### RELY (what the producer assumes about the sink, each checked against `inference_server.py`)

R1. **Every well-formed request gets exactly one reply, eventually, OR the producer times out loudly.**
The producer blocks in `recv_batch` up to RCVTIMEO and treats EAGAIN as a fatal typed error
(`wire_leaf_pool.hpp:217–220`). Check: `serve_forever`→`_drain`→`_serve_batch` sends one
`send_multipart([ident, *envelope, resp])` per drained request (`inference_server.py:384–387`); a malformed
request is `_reject`ed (dropped + logged, `:358–359,365–370`) with NO reply — so a malformed request yields
the producer's RCVTIMEO path. **Caveat (fidelity):** the producer's encode is total over finite features
(`inference_wire.hpp:98–117`); the rows come from the live search (real finite features), so in normal
operation the request is never malformed and R1's "exactly one reply" holds. A NaN/Inf row WOULD be
rejected by the server's decode (`inference_wire.py:90–91,125–126`) → producer timeout. Admissible.

R2. **The reply echoes the corr-id verbatim as the leading frame.** Check: the server captures
`envelope = frames[1:-1]` opaquely and echoes it verbatim in `send_multipart([ident, *envelope, resp])`
(`inference_server.py:353–354,383–387`; design §"DrainedRequest" `:128–131`). For a DEALER carrying an
8-byte corr-id, `frames` at the ROUTER are `[identity][corr-id][payload]`, so `envelope=[corr-id]` and the
reply is `[identity][corr-id][resp]`; the producer's DEALER strips the ROUTER identity and receives
`[corr-id][resp]` — exactly what `recv_corr_payload` expects (≥2 frames, 8-byte leading, `:227–232`). ✓

R3. **The reply payload is a valid batched response frame with `B == #slots in the submitted batch`.**
Check: `run_microbatch` scatters each request its OWN `counts[i]` rows (`inference_server.py:182–189`,
`counts` from each request's matrix row count `:158–163`); `encode_response` emits `B = #values`
(`inference_wire.py:138–139,153`). The producer's request for corr-id carried `S` rows
(`submit_batch`, `wire_leaf_pool.hpp:129–145`), and the server's `decode_request` recovers that `(S,in_dim)`
matrix (`inference_wire.py:124`), so the response carries `S` predictions. The producer enforces `B == S`
(`wire_leaf_pool.hpp:185–188`) — a violation is a loud abort. ✓ (So R3 is not just assumed; it is checked.)

R4. **Service time is positive, padded-shape-bounded, and bounded by RCVTIMEO in the common case.** The
producer's only timing assumption is that a reply arrives within `timeout_ms` (default 15000 ms,
`runner_wire_batched.hpp:69`). Check: §3.2 — one forward, padded to `max_batch`, single-threaded. The
producer does NOT assume any particular service time below RCVTIMEO; if the sink is slow/stalled/down, the
producer times out loudly (admissible). So R4 is the weakest possible: `0 < τ_fwd` and the round-trip is
either `< RCVTIMEO` (reply) or `≥ RCVTIMEO` (loud abort).

R5. **The sink replies to outstanding corr-ids in SOME order consistent with causality, possibly reordered
across messages.** The producer's pipelined matching is order-agnostic (`:368–371`); the strict-barrier
driver issues only one corr-id at a time so reordering is moot there. Check: the server batches across
concurrently-queued requests from one OR several DEALER threads (`_drain` loops `recv_multipart(NOBLOCK)`
draining ALL queued, `:348–363`) and sends them back in drain order (`:384`), but the producer makes no
order assumption, so any sink ordering is within RELY. ✓

R6. **The sink does not coerce a bad frame into a plausible reply.** Check: `_reject` drops + logs, never
sends a zero-filled response (`inference_server.py:358–370`); decode raises loudly
(`inference_wire.py:WireError`). So the producer never receives a *valid-looking* wrong reply — a desync is
either a timeout (no reply) or a structurally-detectable mismatch (unknown corr-id / count≠). ✓

### GUARANTEE (what the producer guarantees to the sink, grounded in producer code)

G1. **Every message is exactly two frames: `[corr-id:8 bytes][payload]`.** `submit_batch` sends
`zmq_send(&corr, 8, ZMQ_SNDMORE)` then `zmq_send(req, size, 0)` (`wire_leaf_pool.hpp:139–144`). The DEALER
prepends no identity (DEALER, not REQ), so the ROUTER sees `[identity][corr-id][payload]` — matching the
server's `ident=frames[0]`, `envelope=frames[1:-1]=[corr-id]`, `payload=frames[-1]`
(`inference_server.py:353–354`). ✓

G2. **The payload is a well-formed v2 batched request frame** `[ver=2][B:u32][in_dim:u32][f32×B·in_dim]`,
row-major, with `B = #slots gathered`, `in_dim` consistent, `B·in_dim` floats exactly
(`inference_wire.hpp:98–117`, `wire_spec.hpp:33` PROTOCOL_VERSION=2). `encode_request` refuses `B==0`,
`in_dim==0`, or `flat.size()≠B·in_dim` as a typed error (`:100–109`) — so the producer never emits a ragged
or empty frame. ✓ The server's `decode_request` validates the same invariants (`inference_wire.py:110–127`).

G3. **The producer holds at most `D` messages outstanding per thread (`D=8` default), each ≤ K rows; and
each slot is single-writer.** `inflight_msgs ≤ D` is enforced by the issue guards (`:578,596`). A slot's row
stays alive in the slot until its reply resumes it; `submitted[s]` prevents re-gathering an in-flight slot
(`:541–544,564,588`). So the producer never has two outstanding messages claiming the same slot. This is
what lets the sink reply out of order safely (it is the dual of R5). ✓ The total rows one thread can put on
the wire at once is `≤ K` (= `ceil(batch/threads)`, default 8); across T threads `≤ pool_batch` (default
32) — the server's `max_batch=256` default comfortably caps above this, so the server's drain never has to
split a producer's message.

G4. **The producer consumes replies (frees the wire) — it does not stall the sink indefinitely.** Each loop
iteration recvs one reply and decrements `inflight_msgs` (`:580–582`), then refills. The producer is never
the party that lets the sink's send queue back up unboundedly, because it is actively recving. (If the
producer aborts on `failed`, it stops recving and closes the socket via the pool dtor `zmq_close`
+ `LINGER=0` `wire_leaf_pool.hpp:82,93–95` — discarding unsent/unrecv'd frames immediately, ADR-0002 loud
whole-pass abort. The sink then sees a vanished peer; its `send_multipart` to that identity is dropped by
the ROUTER. No deadlock.) ✓

---

## 5. Degrees of freedom the code leaves (each with code_ref + admitted behaviors)

**DOF-1 — Source-emission interval `δ_search`.** The fiber's per-leaf search duration is unconstrained
positive. `code_ref:` `fiber_tree.hpp:88–107` (resume runs to next yield), `runner_wire_batched.cpp:589`.
*Admits:* any relative ordering of when different slots' rows become ready ⇒ any coalescing degree `S∈[1,K]`
per message; a slot can re-park "fast" (immediately ready again) or "slow."

**DOF-2 — Coalescing degree `S` per message.** `issue_one` snapshots the ready set at call time
(`:554–560`). *Admits:* `S=1` (only one ready) up to `S=K` (all ready at once) — and within a pass, a
mixture. The strict barrier forces `S = #all-parked` each round (`:313–320`); the pipelined driver lets `S`
vary message-to-message.

**DOF-3 — In-flight message depth actually held (`0 ≤ inflight_msgs ≤ D`).** `:578,596`. *Admits:* the pipe
may run BELOW D when the source can't produce enough ready slots to refill (source-bound) — so D is a cap,
not a floor. At `D=1` (strict) the pipe is exactly the barrier.

**DOF-4 — Sink service time `τ_fwd` and reply ordering.** `inference_server.py:372–387`, §3.2. *Admits:*
the sink may reply to the producer's outstanding corr-ids in any causally-consistent order; service time
varies per call within a padded-shape-bounded interval; a stall/down manifests as a producer RCVTIMEO.

**DOF-5 — Which other producers' rows share a sink forward.** The sink drains ALL concurrently-queued
requests across threads (`inference_server.py:348–363`). *Admits:* a given thread's message may be serviced
alone (B=S) or batched with other threads' messages (B up to `min(total queued rows, max_batch)`), changing
the low-bit f32 a leaf receives (design §4 batch-composition roundoff) — but NOT the producer's transport
behavior (it gets its own S predictions back regardless, R3).

**DOF-6 — `submit_batch` send blocking.** SNDHWM=1000, SNDTIMEO=−1 (§0). *Admits:* in the corner where the
sink stops draining and a thread's send queue reaches 1000 messages, `submit_batch` blocks indefinitely
(no timeout). With `D ≤ 8` per thread this requires the sink to accept-but-not-reply ~1000 messages while
the producer keeps issuing — unreachable in the resume-gated pipeline (the producer can't issue past D
without a reply), so this is reachable only under the strict-barrier's single outstanding message never
filling 1000, i.e. effectively never. Kept in the model for completeness; a constraint that forbade it
would be too permissive.

**DOF-7 — Cross-thread interleaving and the shared `corr_seq`.** `corr_seq.fetch_add(…,relaxed)`
(`wire_leaf_pool.hpp:137`) and the shared `failed` flag. *Admits:* corr-ids are globally unique but
interleaved arbitrarily across threads (relaxed atomic ⇒ no ordering guarantee beyond uniqueness); any
thread's loud abort aborts the whole pass at the next flag check in EVERY thread (`:579,581,592,609`),
truncating the others' in-flight work.

---

## 6. Representative executions (genuinely-enabled traces, with code_refs)

**E1 — Steady pipelined drain, D=8, source fast (self-reinforcing).** Prime: `fill` 8 slots
(`:572`), `issue_one` coalesces all-ready into messages until `inflight_msgs=D=8` (`:578`). Then loop: recv
reply for corr `c` (`:580`) → `--inflight_msgs=7` → scatter its S preds, each slot re-parks at next leaf
(`resume_with`, `ts->running` true, `:589–590`) → REFILL `issue_one` brings a new message → back to 8
(`:596`). *Stability:* **self-reinforcing** while the source keeps producing ready slots faster than replies
drain — the pipe stays at depth D, the sink stays busy, `S` ≈ K/D per message. Exercises DOF-2,3,4.

**E2 — Source-bound underflow (transient, recurring).** Same prime, but `δ_search` is large (heavy search):
after a reply resumes a slot, the fiber takes long to reach its next leaf, so at REFILL time `issue_one`
finds `gathered.empty()` and returns false (`:561`) — `inflight_msgs` stays below D. The pipe runs at depth
1–2. *Stability:* **transient** per-occurrence (it resolves as soon as a slot re-parks), but **recurrent**
under a heavy-search workload — i.e. the system spends time source-bound. Exercises DOF-1,3.

**E3 — Out-of-order reply.** Two messages outstanding: corr `c1` (slots {2,5}), corr `c2` (slot {3}). The
sink batches both queued requests into ONE forward and replies `c2` first then `c1` (causally legal — both
forwards done, send order is the sink's drain order). Producer `recv_batch` gets `c2`, `inflight_.find(c2)`
hits, routes pred to slot 3 (`wire_leaf_pool.hpp:179–196`), `--inflight_msgs`; next `recv_batch` gets `c1`,
routes to {2,5} in order. No mis-route (corr-id keyed). *Stability:* **transient** (a single reordering).
Exercises DOF-4, validates G3/R5.

**E4 — Strict-barrier round (D=1).** `run_episodes_wire_batched`: `any_parked` true (`:310`) → gather ALL 8
parked rows into ONE message (`:313–320`) → `submit_batch` (one corr-id, `:321`) → `recv_batch` blocks the
full round-trip (`:323`) → scatter all 8 (`:326–336`) → re-gather. *Stability:* **self-reinforcing** as the
production default — each round the thread idles the entire `τ_fwd + RTT` (the design's stated D=1 cost,
`runner_wire_batched.hpp:50–53`). Exercises DOF-2 (S=#parked always), contrasts DOF-3 (pinned to 1).

**E5 — Loud recv timeout (sink down/slow).** Producer submits, blocks in `recv_batch` (`:580`); no reply
within `timeout_ms` → `zmq_msg_recv` returns −1/EAGAIN → typed error (`wire_leaf_pool.hpp:217–220`) →
`set_error`, break (`:581`), worker returns; the join + `failed` check returns `std::unexpected`
(`:609–614`). *Stability:* **terminal** (whole-pass abort). Exercises DOF-4 (the timeout arm) and the
lazy-connect NB (`wire_leaf_pool.hpp:65–70` — a never-bound endpoint reaches HERE, not a create failure).

**E6 — Unknown corr-id / count mismatch (desync abort).** Producer receives a well-framed reply whose
leading 8 bytes decode to a corr-id not in `inflight_` (`wire_leaf_pool.hpp:180–182`) OR whose decoded
`B ≠ slots.size()` (`:185–188`) → typed error → whole-pass abort. *Stability:* **terminal.** This is the
guard that makes R3/R6 checked, not merely assumed; reachable only if the sink violated its guarantee (it
does not — `run_microbatch` scatters exactly `counts[i]`), so in a correct pairing this transition is
unreachable, present as the fail-loud net.

---

## 7. DOF-control notes (what design constraint removes each latitude, what becomes unrepresentable)

- **DOF-1/DOF-2 (δ_search, S):** A *fixed-B barrier with a fixed gather cadence* (e.g. always wait until all
  K slots are parked before any submit — the design's "deterministic drain," zmq-inference-service.md §4)
  removes the per-message variability of `S`: every message would carry exactly K rows. Then E2
  (source-bound underflow with S<K) and any mixed-S pipeline trace become unrepresentable; only E4-shape
  rounds remain. Cost: throughput (no RTT overlap), as the design notes.
- **DOF-3 (inflight depth):** Setting `D=1` (the StrictBarrier mode, `runner_wire_batched.hpp:54,70`)
  removes all depth>1 executions — E1 and E3 (out-of-order, needs ≥2 outstanding) become unrepresentable.
  Conversely `D=∞` would remove the "pipe runs below cap because capped" sub-case of DOF-3 but not the
  source-bound underflow.
- **DOF-4 (sink ordering/service):** A sink that replied strictly in submit order (e.g. a per-corr-id FIFO
  forward, one message per forward, no cross-message batching) removes E3's reordering and pins reply order;
  it would also remove the cross-thread batch composition (DOF-5) entirely. The design explicitly chose the
  greedy cross-request drain instead (zmq-inference-service.md §3), so this latitude is intentional.
- **DOF-5 (which rows co-batch):** A *barrier drain at fixed B with a single submitting thread* removes
  cross-thread co-batching — the batch-composition roundoff (design §4) vanishes and per-leaf values become
  run-to-run reproducible, at a throughput cost (design §4 "deterministic drain available… not the default").
- **DOF-6 (send blocking):** Setting `ZMQ_SNDTIMEO` to a finite value (it is currently unset/−1) would
  convert the indefinite send-block into a loud timeout, removing the (practically-unreachable) infinite
  send-block execution. The code chooses not to (only RCVTIMEO is set), so the latitude exists.
- **DOF-7 (cross-thread interleaving):** A single worker thread (`pool_threads=1`) removes all cross-thread
  interleaving and makes `corr_seq` effectively single-threaded; the shared-flag abort still exists but only
  one writer. Removing the shared `corr_seq` (per-thread counters) would be representable only if no future
  cross-thread reply routing is wanted (the header's CRITIQUE C1/D2 note, `wire_leaf_pool.hpp:23–27`).

---

## 8. Fidelity self-audit

**Possible over-permissions (admitting executions the real code cannot produce):**
- I leave `δ_search` and `τ_fwd` fully unconstrained-positive. The real search has a finite max work per
  leaf and the forward a finite padded-shape time, so the true intervals are bounded above. I model
  "bounded nondeterminism" but did not pin numeric bounds (the project is explicitly uncalibrated-time,
  zmq-inference-service.md §9, MEMORY.md ~50 dps). This is faithful to the code (no constant is in the code)
  but is the place a too-permissive critic would point: I allow arbitrarily-large `δ_search`/`τ_fwd`,
  including ones exceeding RCVTIMEO. That is actually *representable* by the code (it produces E5), so it is
  not an over-permission — but a tighter model with a calibrated `τ_max < RCVTIMEO` would forbid spurious
  timeouts. I deliberately did not add that bound (it is not in the code).
- I allow DOF-6's infinite send-block. With the resume-gated pipeline (`D≤8`, send-then-recv) the producer
  cannot enqueue 1000 messages, so the SNDHWM is unreachable in practice. Keeping it is mildly
  over-permissive *operationally* but exactly faithful to the option settings (SNDTIMEO=−1, SNDHWM=1000
  unset) — removing it would assume a server liveness the code does not assume.
- I allow the sink to reply in any causally-consistent order (R5). The actual single-threaded server replies
  in drain order within a batch (`inference_server.py:384`), which is more constrained. But the producer
  imposes NO order assumption, so for the *producer's* representable executions any order is admissible; the
  over-permission is on the sink's side, outside my modeled party, and harmless to the producer (corr-id
  routing, G3/R5).

**Possible over-constraints (forbidding executions the code can produce):**
- I assert `submit_batch` is "effectively non-blocking against a live server." If the server's ROUTER
  RCVHWM (1000, default) backed up while the producer kept sending, a send could block sooner than the
  producer's own SNDHWM — but DEALER↔ROUTER HWM is split across both ends and the producer holds ≤D
  outstanding, so this is not reachable in the pipeline; I keep the send-block transition (BLOCKED_SEND) so
  I am NOT forbidding it. No over-constraint here.
- I model the strict-barrier as `D=1`. That is exactly its structure (`:310–337`), not a simplification —
  the barrier gathers all-parked into one corr-id and blocks for one reply. Faithful.
- I treat the per-slot episode state machine as identical across the two drivers. The code asserts this
  line-for-line (`:441–443`), and I read both copies. Not an over-constraint.

**One subtlety I resolved by reading, not assuming:** the warmup comment says "compiles ONCE PER EXACT
BATCH SIZE B" (`inference_server.py:397`), which would imply per-B service-time steps. But `_serve_batch`
passes `pad_to=self._max_batch` (`:385`) and `run_microbatch` pads up to that single shape (`:171–172`), so
under the *current* code there is exactly one compiled shape and τ_fwd is B-independent. I modeled the
current code (one shape), and flagged the warmup comment as the pre-padding rationale — NOT a contradiction
introduced from outside.

---

## 9. Code-derivation attestation

Every state, transition, guard, free choice, RELY, GUARANTEE, and timing bound in this model is grounded in
a specific line of the read code (cited inline) or in a named causal/transport necessity (positivity;
reply-after-forward; request-before-reply; resume-after-reply; the libzmq default-option semantics implied
by the *absence* of a `zmq_setsockopt`/`zmq_ctx_set` call, which I verified by grepping the complete option
call-site set). No behavior was introduced from an outside expectation of how an inference service "ought"
to" behave. Where the code determines a choice (RNG-exact draws, corr-id routing, the two-frame send, the
fail-loud arms) I determined it; where the code leaves latitude (search duration, forward duration, reply
ordering, coalescing degree, in-flight depth, cross-thread interleaving) I left exactly that latitude and
named the design constraint that would remove it.

---

## 10. Z3 confirmation (theory-confirmation only, minimal)

A small bounded SMT encoding (`producer_transport_check.smt2.py`, in this dir) confirms ONE representative
execution — E1/E3 combined: a single thread, D=2, two messages outstanding, the sink replying OUT OF ORDER,
with the causal laws (positivity, reply-after-forward, resume-after-reply, inflight_msgs∈[0,D]) — is
SATISFIABLE, i.e. admissible. The check is confirmation of the derivation, never its source.
