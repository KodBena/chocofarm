# SYNTHESIS — the faithful composed model of the leaf-evaluation transport boundary

**Synthesizer role.** This document composes ONE faithful model of the leaf-evaluation
transport boundary from the four submitted side-models (two producer, two server) and their
four adversarial fidelity audits (too-permissive + too-constrained per side). For each model
element it adopts the version that survives BOTH fidelity lenses, corrects every fatal/major
defect a verifier found, then composes the two sides via assume-guarantee and characterizes
the global behavior of the result.

Faithfulness is the whole objective: the model's set of representable executions must equal
the system's set of real executions — no more (too-permissive), no fewer (too-constrained).

Every claim is grounded in a line of the code read end to end for this synthesis:
`cpp/src/runner_wire_batched.cpp` (1-630), `cpp/include/chocofarm/wire_leaf_pool.hpp` (1-243),
`cpp/include/chocofarm/inference_wire.hpp` (1-226), `cpp/include/chocofarm/fiber_tree.hpp`
(1-111), `cpp/include/chocofarm/fiber_leaf.hpp` (1-57), `cpp/include/chocofarm/runner_wire_batched.hpp`
(1-100), `cpp/include/chocofarm/runtime_config.hpp` (1-46), `cpp/include/chocofarm/wire_spec.hpp`
(1-58), `chocofarm/az/inference_server.py` (1-457), `chocofarm/az/inference_wire.py` (1-185),
`chocofarm/az/forward.py` (1-64), `chocofarm/config.py` (30-59), and the design record
`docs/design/zmq-inference-service.md` (1-367, used only to CONFIRM operational claims, never
to source them).

---

## 0. Headline

**The boundary is a closed-loop, single-server batch-service system in which N independent
C++ producer threads each run a strictly serial request/reply cycle (per-thread in-flight
depth identically ONE), and a single-threaded Python server self-clocks its batch size to the
instantaneous cross-thread arrival count via a greedy drain over a fixed-shape padded forward.**
The producer's apparent "pipeline depth D" knob is, under the actual coalesce-all + synchronous
drain control flow, a dead knob for a single thread: each thread emits exactly one coalesced
message of S = (number of its slots parked at that instant), blocks for that one reply, and
re-issues. All concurrency the server sees, and all batch-size growth, is **cross-thread**: up
to T messages can be outstanding at once (one per producer thread), and the server's batch B is
the number of those that arrived during the previous forward.

---

## 1. The two correction passes (what the synthesis adopts and fixes)

### 1.1 Producer side — the load-bearing correction (FATAL defect, both producer models)

**Both producer models (M1 = `model-producer-pacing.md`, M2 = `model-producer-transport.md`)
commit the identical fatal too-permissive defect:** they admit a per-thread pipeline of D>1
coalesced messages outstanding with producer-observed out-of-order reply routing. The
too-permissive audit identified this; I have re-verified it line-by-line against
`run_episodes_wire_pipelined` (runner_wire_batched.cpp:540-597) and it is correct.

**The control-flow proof that per-thread depth is identically 1:**

1. `is_ready(s)` requires `!submitted[s]` (line 543).
2. Prime (line 572): `for (s<K) fill(s)` parks all K slots **synchronously** — no fiber runs
   between fills, and crucially none runs between the subsequent `issue_one` calls.
3. Prime (line 578): `while (inflight_msgs < D && issue_one()) {}`. The **first** `issue_one`
   (551-569) gathers EVERY ready slot into ONE message (one corr-id), marks them all
   `submitted=1` (564), and `++inflight_msgs` → 1. The **second** `issue_one` finds
   `is_ready` false for every slot (all submitted, none re-parked — no fiber ran), so
   `gathered.empty()` → returns false (561). The prime loop ends at `inflight_msgs == 1`.
4. Main loop (579-597): `recv_batch()` (580) returns the one outstanding corr-id's
   completions; `--inflight_msgs` → 0 (582). The Completion loop (584-594) resumes each slot
   **one at a time** (boost.context is cooperative, fiber_leaf.hpp:48; the loop is sequential),
   re-parking it (`running` → `continue`, 590) or advancing/filling it. A slot becomes ready
   again only **inside** this loop — i.e. only AFTER the recv that already decremented the lone
   outstanding message.
5. Refill (596): `while (inflight_msgs < D && issue_one()) {}`. `inflight_msgs == 0 < D`, so
   the first `issue_one` gathers the now-ready slots into ONE message → `inflight_msgs = 1`;
   the second finds nothing ready → false. Loop ends at `inflight_msgs == 1`.

So **`inflight_msgs ∈ {0, 1}` for a single thread at every observable point**, and a thread
holds at most ONE coalesced message outstanding. `D = max(1, wcfg.max_inflight_msgs)` (392)
constrains nothing reachable for a single thread. The out-of-order-tolerant corr-id routing
(`inflight_.find(corr)`, wire_leaf_pool.hpp:179) is a **correctness guarantee offered to the
peer, never a behavior the single-socket producer exercises** — `recv_batch` deterministically
returns the one outstanding corr-id, and `inflight_.find` always hits the single map entry.

**Adopted producer state machine.** I adopt M1's per-slot automaton states (the cleaner names),
M1's FINALIZED→PARKED / FINALIZED→IDLE post-finalize fork (the too-constrained audit found M2
collapses this — M2-1, minor; M1 is faithful here), and M2's explicit grounding of the blocking
surface in the libzmq option settings. I CORRECT both models' D>1 / out-of-order-per-thread
latitude to per-thread depth-1. The strict-barrier driver (`run_episodes_wire_batched`,
60-354) and the pipelined driver (`run_episodes_wire_pipelined`, 377-627) are therefore **both
per-thread depth-1**: the strict barrier by construction (one gather-all submit, one recv,
resume-all, repeat — 310-337), the pipelined driver because coalesce-all `issue_one` caps
`inflight_msgs` at 1. They coincide in their per-thread message-count behavior — but NOT via a
tunable D, and the strict barrier is the **production default** (`WireMode::StrictBarrier`,
runner_wire_batched.hpp:70; the pipelined arm is reachable only behind the mode flag, 66-67).

The per-slot episode lambdas (spawn_ply / finalize_and_write / apply_decision / advance / fill)
are **line-for-line identical** across the two drivers (the pipelined header comment 441-443
asserts this and the code confirms it), so the per-slot state machine is shared.

### 1.2 Server side — the load-bearing correction (FATAL defect, ONE server model)

**Server model B (`model-server-transport.md`) commits a fatal too-permissive defect** in the
scatter path: it asserts that `send_multipart` (inference_server.py:387) to a KNOWN peer whose
per-peer outbound pipe is at `SNDHWM` BLOCKS indefinitely (`SNDTIMEO=-1`), and builds an entire
representative execution E7 (an absorbing server-wide wedge) and a guarantee mechanism (G5) on
it. **Both server audits settle this against the installed libzmq-4.3.5 and the code:** the
ROUTER sets **no socket options at all** (verified directly — inference_server.py:315-318 is
`socket(zmq.ROUTER)` + `bind` + poller-register, with no `setsockopt` anywhere in the file), so
`ROUTER_MANDATORY = 0` (default). Under `ROUTER_MANDATORY = 0` the libzmq-4.3.5 ROUTER
**DROPS** a message — silently — when it cannot be routed OR when the destination peer's HWM is
reached, for KNOWN and unknown peers alike. It NEVER blocks. Blocking-until-`SNDTIMEO` is the
`ROUTER_MANDATORY = 1` behavior, which the code never enables, so `SNDTIMEO = -1` is irrelevant
(the send never enters the blocking path).

**Adopted server scatter semantics = Model A's (`model-server-drain.md`):** a scatter send to a
full OR vanished peer is **silently dropped** (DOF-7), the scatter loop continues, and the
peer's finite `RCVTIMEO` surfaces the lost reply as a loud whole-pass abort (the peer's R6 /
the producer's E3). The indefinite-block SCATTER→SCATTER transition, DOF-7-as-block, and E7 are
removed. The corrected liveness statement is **stronger** than Model B's: the server's scatter
is unconditionally non-blocking, independent of the peer's recv behavior — the ONLY server
waits are the bounded 100ms first-request poll (339-342) and the between-batch reload (381).

**Adopted server state machine = Model A's** (it had `fidelity_verdict: faithful` under both
lenses), with two additions the audits flagged:

- **The aggregate batch-size bound** (too-permissive audit, minor, against Model A's R4): the
  free range of one drain's `total_rows` is the **Σ-over-threads** outstanding count
  (≈ T · mean-rows-per-message), capped by `max_batch` + soft overrun — NOT a single thread's
  bound. With per-thread depth-1 (§1.1) the aggregate is ≤ T · K = `pool_batch` rows offered at
  once (defaults: 4 · 8 = 32), well under `max_batch = 256`.
- **An exceptional-termination terminal** (too-constrained audit, minor, BOTH server models):
  `run_microbatch` raises an **uncaught** `ValueError` on a ragged-`in_dim` batch (162), on a
  bad forward-output shape (178-179), and `RedisParamsSource.poll`→`read_weights` can raise
  (284). None is caught — each propagates through `_serve_batch` (385) → `serve_forever`
  (437-439) and **kills the server thread**. The model must carry a reachable
  EXCEPTIONAL-TERMINATION terminal (distinct from the clean `_stop` STOPPED), gated on RELY
  R1/R5 violations exactly as the reject path is gated on a malformed frame.

### 1.3 The one collapse both sides got right (a fidelity SUBTLETY, not a defect)

Every one of the four models, and all four audits, agree on the single legitimate timing
collapse, and I adopt it: **forward service time `σ` is INDEPENDENT of the instantaneous batch
B**, because `run_microbatch` pads every batch UP to the fixed `(max_batch, in_dim)` shape
(inference_server.py:171-172, `pad_to=self._max_batch` at 385) so XLA compiles and runs ONE
executable (jit_forward_core caches one compiled graph, 105-115). Padded rows are real zeros and
`forward_core` is row-independent (forward.py:50-63), so real-row outputs are byte-identical to
the unpadded forward. Making `σ` GROW with B would be the over-constraint (it would forbid the
real constant-shape forward). `σ` is NOT pinned to a numeric constant — it stays a positive
bounded-nondeterministic duration (host/VM jitter on the shared 4-vCPU host; XLA pinned
single-threaded via `XLA_FLAGS=--xla_cpu_multi_thread_eigen=false`, `OMP_NUM_THREADS=1`,
config.py:41-42), with a **one-time cold-XLA-compile tail** on the first forward of the padded
shape when `warmup` was not pre-run (the too-constrained audit's M2-2 correction: `serve_forever`
does NOT call `warmup` itself, 437-439, so the cold spike EXISTS for an un-warmed server). A
**second compiled shape** appears only under the soft-cap overrun (§3, DOF-2): a single request
whose `B_i` pushes the post-batch total over `max_batch` is appended whole (348/362), so
`pad_to > B` is false (171) and the forward runs unpadded at a larger shape (with its own
first-sight compile). Under the default geometry (T·K = 32 ≤ max_batch = 256) the overrun is
unreachable; it becomes reachable only if a single thread's K exceeds `max_batch`.

---

## 2. The composed state machine

The composed model is two automata (one per producer thread; one server) plus the wire as a
causal-order relation. The producer-thread automaton drives **K per-slot sub-automata** over
one DEALER socket; the server automaton is a single greedy-drain loop over the ROUTER.

### 2.1 Producer-thread automaton (per thread t; default T = 4)

The per-thread automaton is a strict cycle, NOT a pipeline. At each thread there is at most one
coalesced message outstanding (§1.1).

```
                 fill all K slots (prime)
   [PRIME] ───────────────────────────────────► [GATHER]
                                                    │  issue_one: gather ALL ready slots
                                                    │  (S = #ready, 1..K) into ONE corr-id
                                                    │  message; mark submitted; inflight=1
                                                    ▼
                                                 [BLOCKED_RECV]   ← the SOLE blocking point
                                                    │  recv_batch: block up to RCVTIMEO=15000ms
                                  ┌─────────────────┼─────────────────┐
                       reply arrives                       RCVTIMEO fires / malformed /
                       (the one corr-id)                   unknown corr-id / count mismatch
                                  │                                   │
                                  ▼                                   ▼
                              [SCATTER]                           [ABORT]
                    resume each of the S slots one              set_error; break;
                    at a time → re-park / advance / fill        whole-pass std::unexpected
                                  │
                                  ▼
                              [REFILL]  inflight 0→ (issue_one over now-ready slots) →1
                                  │         (one message again; second issue_one finds nothing)
                       ┌──────────┴──────────┐
              inflight>0 (issued)      no slot ready AND inflight==0
                       │                      │
                       ▼                      ▼
                  [BLOCKED_RECV]          [DRAINED]  (subset exhausted; telemetry; return)
```

**Per-slot sub-automaton** (states a slot occupies inside the thread cycle):

| state | meaning | code |
|---|---|---|
| IDLE | `!sl.active`; subset exhausted or between episodes; not `is_ready` | 543, 537 |
| PARKED | `sl.active && ts->running && !submitted[s]`; leaf row live in `ts->ch.features`; the ONLY ready state | 541-544, fiber_leaf.hpp:31 |
| OUTSTANDING | `submitted[s]==1`; the slot's row is in the thread's one coalesced message, awaiting the reply | 564 |
| ADVANCING | transient: reply arrived, `resume_with` ran, the search runs between leaves (or `advance` drains a non-parking chain) | 589-591, 502-510 |
| FINALIZED | `apply_decision`/`advance` returned false (TERMINATE / `ply>=max_steps` / `env.empty`); `finalize_and_write` ran | 484-499, 449-476 |

Slot transitions (each grounded; **free-choice flags corrected per §1.1**):

| from → to | guard | free? | note |
|---|---|---|---|
| IDLE→PARKED | prime/refill `fill(s)`, `next_idx<episodes`, spawns a ply that parks | det | RNG-exact draws (seed fold + slot mt19937_64); 511-538 |
| PARKED→OUTSTANDING | `is_ready(s)` AND this slot is in the current `issue_one` gather | **det given the ready set** | `issue_one` includes EVERY ready slot (554-561); S = #ready is data/timing-set, NOT a free per-edge choice (corrects M1's "free 1..K at prime") |
| OUTSTANDING→ADVANCING | the (one) outstanding corr-id's reply is delivered | **det in WHICH corr-id** | only one is outstanding (§1.1); the free input is the service DURATION and reply TIMING, not arrival ORDER (corrects M1/M2) |
| ADVANCING→PARKED | `ts->running` after `resume_with`, or `advance` parked down a chain | det | search/env are RNG-exact; the next think-time is the free input (DOF-2) |
| ADVANCING→FINALIZED | `!ts->running` and `apply_decision` returned false | det | episode/env logic |
| FINALIZED→PARKED | `fill(s)`, `next_idx<episodes`: next episode parks its first leaf | det | M1's fork; the common steady-state outcome (593→511-533) |
| FINALIZED→IDLE | `fill(s)` returns false, `next_idx>=episodes` | det | subset exhausted (537) |
| OUTSTANDING→ABORT | recv/decode/corr-id/count error (RCVTIMEO, malformed, unknown id, B mismatch) | det given trigger | the loud arm (ADR-0002); TRIGGER is the peer's free choice (R4/R6) |

### 2.2 Server automaton (single, single-threaded; default `max_batch = 256`)

```
   [STANDUP] ──current() asserted──► [POLL_WAIT]
                                         │  poll(100ms) loop (339-342); ~0 CPU idle
                          ┌──────────────┼──────────────┐
                  poll readable                  _stop observed
                          │                              │
                          ▼                              ▼
                      [DRAINING]                      [STOPPED]  (clean; 344-345/436)
        recv_multipart(NOBLOCK) loop (348-362):
          take frames while total_rows < max_batch;
          decode_request each; malformed → _reject (drop+log, 358-360);
          stop on zmq.Again (queue momentarily empty) OR cap reached
                          │
            ┌─────────────┴───────────────┐
     drained non-empty              drained == [] (all rejected, or woke on stop)
            │                              │
            ▼                              ▼
      [RELOAD_CHECK]                   [POLL_WAIT]   (no forward; re-poll; 438)
   params_source.poll() (381):
     StaticParamsSource → None;
     RedisParamsSource → may read new weights (can RAISE → EXCEPTIONAL_TERMINATION)
            │
            ▼
        [FORWARD]   run_microbatch (134-189): concat → pad to (max_batch,in_dim) →
            │       ONE forward_fn; np.asarray BLOCKS until XLA done (177) = SERVICE TIME σ.
            │       ragged in_dim (162) / bad shape (178-179) → uncaught → EXCEPTIONAL_TERMINATION
            ▼
        [SCATTER]   send_multipart([ident,*envelope,resp]) per drained request, in drained
            │       order (384-387). NON-BLOCKING: ROUTER_MANDATORY=0 ⇒ drop (never block) on a
            │       full/vanished peer (corrected per §1.2). Dropped reply → peer's RCVTIMEO abort.
            ▼
        [POLL_WAIT]  (loop) ... or [STOPPED] if _stop
```

`[EXCEPTIONAL_TERMINATION]` is an absorbing terminal reachable from RELOAD_CHECK / FORWARD on a
RELY-violating input (ragged `in_dim`, bad forward shape, reload raise): the uncaught exception
unwinds `serve_forever` and the thread dies. From the peer's standpoint every in-flight and
future request silently stops being answered → each peer's `RCVTIMEO` fires → loud whole-pass
abort. (RELY-gated; minor.)

---

## 3. The timing model (bounded nondeterminism, derived)

Two free inputs; everything else is a deterministic function of them and the causal order.

**Source emission `δ_{s}^{r}` (per slot s, per logical step r).** The interval from when slot
s's fiber is resumed (`start`/`resume_with`, fiber_tree.hpp:88-107) until it next parks at a
leaf (`ch.at_leaf=true`, fiber_leaf.hpp:46-48) or finalizes. Set by the search's own internal
work, which the code never fixes. Constraints DERIVED from code:

- **(δ-1) positivity:** `δ > 0` (every resume runs real fiber code, fiber_tree.hpp:106).
- **(δ-2) per-slot reply-dependence:** slot s's (r+1)-th park cannot precede its r-th reply —
  the fiber is suspended inside `YieldingNetEvaluator::predict` until `resume_with`
  (fiber_leaf.hpp:48, fiber_tree.hpp:103-107). The single causal link from server outputs to
  future producer inputs — it makes the boundary a **closed loop**.
- **(δ-3) no producer-imposed upper bound** (no timer/threshold anywhere; the "rejected
  alternative" of a partial-parked flush threshold is explicitly NOT taken, runner_wire_batched.cpp:298-300).
- **(δ-4) within-thread serialization:** at most one of a thread's K fibers runs at a wall-clock
  instant (boost.context cooperative; the Completion loop processes one at a time). Across the T
  OS threads true parallelism is admitted and **code-unbounded** (the 4-vCPU host wall is an
  operational fact, not a code constraint — `T` independent `std::thread`s, 604).

The coalescing degree **S = #ready-at-the-`issue_one`-instant ∈ [1, K]** is a deterministic
read of the ready set at the gather instant; WHICH slots are ready then is the free emission
timing (δ). NOT collapsed: pinning δ would pin S and forbid the staggered-arrival executions.

**Sink service `σ` (per forward).** The wall-clock duration of FORWARD (entering `forward_fn`
to `np.asarray` returning, 177). Constraints:

- **(σ-1) positivity:** `σ > 0` (a real MLP matmul chain; `np.asarray` blocks the thread until
  XLA finishes — never instantaneous).
- **(σ-2) shape-invariance (the one collapse, §1.3):** `σ` independent of live B because every
  batch is padded to `(max_batch, in_dim)` (171-172) → one compiled executable. A bounded
  nondeterministic positive duration around a shape-determined center, NOT a constant.
- **(σ-3) cold-compile tail:** the first forward of the padded shape on an un-warmed server adds
  a one-time `σ_compile ≫ σ_steady` (`serve_forever` does not call `warmup`).
- **(σ-4) overrun regime:** a B>max_batch batch (DOF-2) runs unpadded at a larger shape with its
  own first-sight compile; reachable only when a single thread's K > max_batch.
- **(σ-5) one-at-a-time:** the server is single-threaded; forwards are TOTALLY ORDERED and
  NON-OVERLAPPING; requests arriving during a forward accumulate in the ROUTER pipes for the
  NEXT drain. This is the engine of the self-clocking batch-size feedback loop.

**Cross-boundary causal order (the binding laws):**

1. `δ, σ > 0` (positivity; no instant transition).
2. **round-trip:** `submit(corr) < recv(corr)` on the DEALER.
3. **reply-after-forward:** `reply(corr) ≥ forward_complete(k) > forward_start(k) ≥
   arrival(corr)` for the forward k that drained corr's rows.
4. **forwards totally ordered:** `forward_complete(k) ≤ forward_start(k+1)` (σ-5).
5. **per-slot reply-dependence:** δ-2.
6. **RCVTIMEO bound:** a producer recv that waits longer than `timeout_ms = 15000` returns
   EAGAIN → loud abort (the sole liveness backstop, wire_leaf_pool.hpp:217-221).
7. **per-thread depth-1:** at most one coalesced message outstanding per producer thread (§1.1).

---

## 4. Assume-guarantee composition

Each side is one party. RELY = what it assumes about the peer over the wire (checkable against
the peer's code); GUARANTEE = what it provides. Composition checks whether each RELY is
discharged by the peer's GUARANTEE.

### 4.1 Producer GUARANTEEs (grounded in the C++ driver)

- **PG1 — well-formed v2 batched frame.** Every message is `[corr:8B (SNDMORE)][payload]`
  (wire_leaf_pool.hpp:139-144); the payload is `encode_request(flat, B, in_dim)` which rejects
  B=0 / in_dim=0 / `flat.size()!=B*in_dim` as a typed error (inference_wire.hpp:100-109) — a
  ragged/empty batch is never sent. `PROTOCOL_VERSION = 2` (wire_spec.hpp:33).
- **PG2 — process-global unique corr-ids.** Each submit stamps `corr_seq->fetch_add(1, relaxed)`
  from a process-global atomic passed by reference (wire_leaf_pool.hpp:137; runner_wire_batched.cpp:403),
  so no two outstanding requests across ALL threads alias under the server's opaque echo.
- **PG3 — stable feature row until reply.** Single-writer-per-thread; `submitted[s]` excludes a
  slot from re-gather (543); the row lives in the suspended fiber's stack (fiber_leaf.hpp:31)
  and is copied into the gather buffer at submit (557).
- **PG4 — at most ONE message outstanding per thread; always eventually recvs.** Per-thread
  depth-1 (§1.1); the thread recvs whenever `inflight_msgs>0` (579-580), so it never wedges the
  ROUTER by failing to drain its replies, and the total rows offered across T threads is ≤
  `pool_batch`.
- **PG5 — loud abort, never a silent wrong-slot apply.** Any recv/decode/corr-id/count mismatch
  is the typed error arm (wire_leaf_pool.hpp:154-196) → `set_error` → whole-pass
  `std::unexpected` (609-614); never a wrong-slot apply, never a zero/stale leaf.

### 4.2 Server GUARANTEEs (grounded in the Python server)

- **SG1 — one B-exact reply per accepted request, envelope echoed verbatim.** `run_microbatch`
  scatters each request its own `counts[i]` rows (158-163, 184-189); `_serve_batch` sends
  `[ident, *envelope, resp]` 1:1 in drained order (384-387). Envelope = `frames[1:-1]` captured
  opaquely (354), never parsed.
- **SG2 — no silent coercion of a bad request.** A malformed frame is `_reject`ed (drop+log,
  358-360, 365-370), never zero-filled into the forward; the bad corr-id gets no reply.
- **SG3 — one consistent net version per batch.** Params read once per batch (381-382); reload
  is strictly between batches (G5 in design §3).
- **SG4 — single forward per batch; padded fixed shape.** Exactly one `forward_fn` call per
  `_serve_batch` (177 inside one `run_microbatch`, 385); padded to one shape.
- **SG5 — scatter is non-blocking.** ROUTER_MANDATORY=0 ⇒ a send to a full/vanished peer is
  DROPPED, not blocked (§1.2); the server's only waits are the bounded poll and the reload.
- **SG6 — replies may be coalesced/reordered across peers and corr-ids, but each corr-id's reply
  carries exactly that request's B predictions.** The server drains across all queued requests
  (348-363) and replies in drain order; it makes no per-peer ordering promise beyond per-corr-id
  B-exactness.

### 4.3 Discharge table

| # | RELY | held by | discharged? | note |
|---|---|---|---|---|
| Producer R1 | opaque corr-id round-trip (8-byte leading frame returns unchanged) | SG1 (envelope echo, 354/387) | **yes** | server never parses `frames[1:-1]`; producer routes via `inflight_.find(corr)` (179) |
| Producer R2 | one B-exact reply per request, same row order | SG1 (per-count scatter, 184-189) | **yes** | producer enforces `decoded->size()==slots.size()` (185) and aborts otherwise — defense in depth |
| Producer R3 | replies may arrive in any order across outstanding messages | SG6 (drain-order replies; ROUTER fair-queue) | **yes, but VACUOUS per-thread** | with per-thread depth-1 (§1.1) a thread never has 2 outstanding, so it never observes reordering; the rely is real but never load-bearing on one socket. Across threads, each thread's single reply is independent — no cross-thread ordering assumption is made. |
| Producer R4 | bounded service, eventual reply within RCVTIMEO | SG4 (one forward) + SG5 (non-block scatter) + serve_forever loop (436-439) | **yes**, MODULO the open gaps below | a reply arrives unless the server EXCEPTIONALLY TERMINATES or DROPS the reply; both surface as the producer's loud RCVTIMEO abort, not a hang |
| Producer R5 | wire codec agreement (v2 frame) | SG1 + drift-checked `wire_spec` (test_wire_drift) | **yes** | one SSOT, mechanically drift-checked |
| Server R1 | well-formed v2 value frames | PG1 (encode_request validation) | **yes**; breach degrades to `_reject` drop, not a server fault | |
| Server R2 | envelope `[corr:8B][payload]` after the identity | PG1 (SNDMORE corr frame) | **yes** | for a DEALER the ROUTER sees `[identity][corr][payload]`; `frames[1:-1]=[corr]` |
| Server R3 | the peer eventually recvs its reply (so the per-peer pipe stays ≪ SNDHWM) | PG4 (depth-1: ≤1 unanswered message/thread; thread recvs whenever inflight>0) | **yes** | per-thread depth-1 bounds the per-peer outbound queue at ≤1 reply; SNDHWM=1000 is never approached |
| Server R4 | bounded outstanding per peer | PG4 (depth-1) | **yes, STRONGER than assumed** | the server models assumed ≤ T·D; the real bound is ≤1 per thread |
| Server R5 | peer tolerates coalescing + out-of-order corr-id replies | PG5 (corr-id routing, B-exact check) | **yes** | producer routes by echoed corr-id, accepts B==submitted-slot-count |
| Server R6 | a dropped reply surfaces as a loud peer timeout, not a hang | PG5 + RCVTIMEO (the SOLE producer blocking point) | **yes** | a SG5 drop or an EXCEPTIONAL_TERMINATION → producer RCVTIMEO → loud whole-pass abort |

### 4.4 Composition gaps (undischarged or partially-discharged relies)

1. **Server EXCEPTIONAL_TERMINATION partially discharges Producer R4.** Producer R4 ("a
   well-formed request eventually gets a reply") is NOT discharged as "reply" when the server
   thread dies on a ragged-`in_dim` / bad-shape / reload-raise (§1.2/§2.2). It IS discharged as
   "the producer does not hang" — every outstanding and future request times out at
   `RCVTIMEO = 15000ms` → loud abort. So the composed guarantee is **"reply OR loud bilateral
   abort within RCVTIMEO," never a silent hang** — but NOT "reply." Naming this is the point: the
   server's uncaught-exception terminal is a real reachable state (RELY-gated), and the only
   thing that keeps it from being a silent wedge is the producer's RCVTIMEO. If `RCVTIMEO` were
   unset (it is set — 83), this gap would become a deadlock; it is closed only by that one
   socket option.

2. **Producer R3 (out-of-order tolerance) is offered but never exercised per-thread.** This is
   not a gap that breaks composition — it is a guarantee the producer makes that the current
   single-socket-depth-1 producer never needs. It is recorded because it is the seam where a
   FUTURE cross-thread work-stealing registry (the `corr_seq` is process-global precisely to
   enable this, wire_leaf_pool.hpp:23-27) would start exercising real per-socket reordering, at
   which point R3 becomes load-bearing. Today it is slack, not a gap.

3. **Scatter-drop ↔ peer-RCVTIMEO is a system-level liveness contract, not a transport
   guarantee.** SG5 (drop) discharges Server R6 only because the producer side independently sets
   `RCVTIMEO`. Neither side's code alone makes a dropped reply observable; the contract is
   discharged only by the COMPOSITION (server drops loudly-at-the-system-level iff the peer
   times out). A producer with `RCVTIMEO` unset would turn a server drop into a permanent
   single-thread stall (the other T−1 threads keep running; the pass never returns). The code
   closes this by always setting `RCVTIMEO=timeout_ms` (83).

---

## 5. Global behavior — the regimes the composed model admits

All regimes below are DERIVED from the composed model (§2-§4), not assumed. They are
characterized by the timing/scheduling relation between source emission `δ` and service `σ`,
under per-thread depth-1 and single-threaded serialized forwards.

### R1 — Idle / light-load, B≈1 self-clocking (low latency)

- **Conditions.** The aggregate offered rate ≪ 1/σ: at most one producer thread has a parked
  slot when the server drains. Server self-loops in POLL_WAIT until one request arrives.
- **Reachable:** yes. **Representative schedule:** T=1 or a slow search (large δ); one slot
  parks, server polls readable (342), drains one request (`zmq.Again` on the next recv, 351),
  one padded forward, one reply; the reply-gated producer re-parks ~1 deep; repeat.
- **Stability:** self-reinforcing while δ stays large relative to σ (each reply re-arms exactly
  one request that the next forward consumes).
- **Progress.** Latency-optimal (B≈1, no queueing delay); throughput = 1/σ leaves/sec — the
  forward is un-amortized. Both sides make progress every round.

### R2 — Cross-thread coalescing, B tracks demand (the design's intended operating point)

- **Conditions.** Several producer threads have messages outstanding concurrently (one per
  thread, depth-1). During forward k, some subset of the other threads' single messages arrive
  and buffer; the next drain coalesces them into ONE forward. B at drain k = #threads whose
  message arrived during forward k−1, capped by `max_batch`.
- **Reachable:** yes — **confirmed SAT** by the bounded Z3 check (`check_composed_admissible.py`,
  part A): two threads each at depth-1, thread 1's message buffered during thread 0's forward,
  forwards serialized and non-overlapping, replies after forwards, re-issue after reply. This is
  the regime the cross-thread `_drain` coalescing (inference_server.py:348-363) exists to serve.
- **Representative schedule.** T=4, K=8, default geometry. Threads park slots on independent δ
  schedules; each thread emits one coalesced message of S∈[1,K]; the server drains whatever of
  the ≤T messages are queued into one padded forward; scatters; each thread re-issues on its
  reply. Mean rows/forward grows with the cross-thread arrival overlap.
- **Stability.** Self-reinforcing AROUND a load-determined fixed point: a slower forward lets
  MORE threads' messages buffer → a larger next B → (same σ, padding) the SAME per-forward cost
  amortized over more rows → higher throughput, which drains the backlog → smaller next B. The
  feedback is negative (self-correcting), so B settles where the aggregate arrival rate equals
  the drain rate. **No batch-size explosion** (the soft cap + padding bound the cost).
- **Progress.** Throughput = B/σ leaves/sec, with B between R1's ≈1 and R3's `max_batch`. Both
  sides make progress; the forward is amortized across threads. This is the **only** regime where
  the design's batching lever actually pays off, and it is **cross-thread** — a single thread's
  depth-1 cycle contributes at most S rows per forward and idles a full RTT per its own message.

### R3 — Saturation, B pinned at the cap (high throughput, cap-bounded)

- **Conditions.** Aggregate offered rows/σ ≥ max_batch: during each forward, ≥ max_batch rows
  buffer across threads. Reachable only if `T·K = pool_batch ≥ max_batch` (default 32 < 256, so
  the default geometry CANNOT reach this — it is reachable only under a config that raises
  pool_batch above max_batch, e.g. many threads × large K).
- **Reachable:** yes, under a high-T/high-K config; **not** under the defaults.
- **Representative schedule.** The drain hits `total_rows >= max_batch` every iteration (348);
  B_k = max_batch (or just over via the soft overrun, DOF-2); same padded shape, same σ.
- **Stability.** Self-reinforcing: each forward consumes max_batch and lets ≥ max_batch more
  pile up — the cap is a fixed point. A standing backlog forms (latency rises) but throughput is
  maximal and stable.
- **Progress.** Throughput = max_batch/σ (the ceiling). Per-request latency = queueing delay +
  σ, bounded by the producers' depth-1 backpressure (a thread cannot offer a second message until
  its first is answered, so the total backlog is ≤ pool_batch rows — the system cannot build an
  unbounded queue).

### R4 — Per-thread strict-barrier idle (the depth-1 RTT-bound regime)

- **Conditions.** Always present at the per-thread level: every producer thread idles the full
  round-trip (σ + transit) each of its own cycles, because depth-1 means it blocks on its one
  reply before issuing the next message. With T=1 this is the WHOLE system behavior (no
  cross-thread overlap to hide the RTT).
- **Reachable:** yes; it is the strict-barrier driver's defining behavior (310-337) and the
  pipelined driver's per-thread behavior too (§1.1).
- **Representative schedule.** T=1, K=8, strict barrier: gather all parked → one submit → block
  the full RTT → resume all → re-gather. The single thread's search idles σ+RTT every round; the
  server sees exactly one message at a time → R1-like B = S.
- **Stability.** Self-reinforcing for T=1. For T>1 it is the per-thread substrate that R2
  overlaps across threads — the more threads, the more each thread's idle RTT is hidden behind
  others' forwards (this is why throughput scales sub-linearly toward the ~1.9× ceiling on the
  4-vCPU host, an operational fact).
- **Progress.** Per thread: one S-row forward per RTT. The "pipeline depth D" does NOT improve
  this for a single thread — the dead knob.

### R5 — Soft-cap overrun, second compiled shape (transient service-time spike)

- **Conditions.** A single request's `B_i` (a thread's S up to K) crosses the cap: the drain
  appends it whole (348/362), pushing `total_rows` into `[max_batch, max_batch + B_i − 1]`;
  `pad_to > B` is false (171) → unpadded forward at a NEW shape → first-sight XLA compile.
  Reachable only if a single thread's K > max_batch (default K=8 ≪ 256: unreachable).
- **Reachable:** yes under K > max_batch; not under defaults.
- **Stability.** Transient: the first overrun of a given shape pays the compile; subsequent
  overruns at that shape are warm. If overruns recur at varied B, each new shape compiles once.
- **Progress.** A one-time σ spike per never-seen shape; throughput otherwise unchanged.

### R6 — Loud bilateral abort (the liveness backstop)

- **Conditions.** Any of: (a) the server is down / its forward exceeds the producer's
  `RCVTIMEO=15000ms` → producer recv EAGAIN; (b) the server EXCEPTIONALLY TERMINATES
  (ragged in_dim / bad shape / reload raise) and stops answering → producer RCVTIMEO; (c) the
  server DROPS a reply (full/vanished peer, SG5) → producer RCVTIMEO; (d) a desync (unknown
  corr-id / count mismatch / malformed envelope) → producer typed error directly.
- **Reachable:** yes (the failure edge). RELY-gated against a correct, live pairing — but a real
  reachable state.
- **Representative schedule.** Producer primes, blocks in `recv_batch`, no/late/dropped reply →
  EAGAIN at `timeout_ms` → `set_error`, break → after `join`, `failed` → whole-pass
  `std::unexpected` (609-614). No partial write, no wrong-slot apply.
- **Stability.** Terminal (transient → absorbing abort). The whole pass fails loudly.
- **Progress.** No throughput; the system makes NEGATIVE progress (aborts) but **does not hang** —
  the RCVTIMEO is the sole liveness backstop. The one regime where a composition gap (§4.4-1)
  would, absent RCVTIMEO, become a deadlock.

### R7 — Degenerate non-parking / empty-belief stretches (silent on the wire)

- **Conditions.** `fill`/`advance` drain plies that finish without parking a leaf
  (empty-belief guard inside the search returns without parking, fill:527-535, advance:502-510),
  or an episode finalizes without ever parking a leaf.
- **Reachable:** yes (env/search-dependent). **Representative schedule.** A slot advances several
  plies (env stepping, record-assembly) between two PARKED states, or an episode that produces
  zero wire messages. **Stability.** Transient (per-episode). **Progress.** The slot makes search
  progress but emits nothing on the wire; the server sees fewer rows from that thread that round.

---

## 6. Degrees of freedom and the constraints that remove them

For each DOF: the latitude the code leaves, the side, the behaviors it admits, the concrete
design change that removes it, the behaviors that become unrepresentable, and the cost. (Fully
in the structured object; summarized here.)

| DOF | side | latitude | removing constraint | becomes unrepresentable |
|---|---|---|---|---|
| DOF-1 source think-time δ | producer | when a slot next parks (search-paced; δ>0, else free) | a fixed-cost scripted source (CyclicGumbelSource + fixed leaf count/ply, fiber_tree.hpp:55-56) | R1↔R2↔R3 transitions; S becomes a fixed function of K and the round; all staggered-arrival and B-tracks-demand executions |
| DOF-2 coalescing degree S∈[1,K] | producer | #ready at the `issue_one` instant (no timer/threshold) | submit each ready slot in its own message (replace the gather loop with single-slot submit) | all S>1 messages; only cross-thread `_drain` coalescing survives |
| DOF-3 per-thread depth (corrected: identically 1) | producer | NONE per thread — the dead knob; D is unreachable latitude | (already pinned by coalesce-all + synchronous drain) | nothing further — depth>1 is ALREADY unrepresentable; this row records that D removes no real latitude |
| DOF-4 cross-thread overlap / which threads co-batch | both | which of the ≤T outstanding messages arrived during the prior forward | a fixed-B barrier drain (block until exactly B queued — design §4 deterministic drain) OR T=1 | R2/R3 batch-composition nondeterminism; per-leaf f32 becomes run-to-run reproducible (design §4 roundoff), at a throughput cost |
| DOF-5 geometry T, K, max_batch | both | T=max(1,pool_threads), K=ceil(pool_batch/pool_threads), max_batch=256 | pin pool_threads=1, pool_batch=1 → T=1,K=1 | S≡1, one tree at a time; R2/R3 (cross-thread coalescing) unrepresentable; collapses to R1/R4 |
| DOF-6 service σ + cold-compile tail + overrun shape | server | σ>0, shape-invariant in B (padding), with a cold-JIT tail and an overrun branch | a real-time scheduler + isolated core + fully AOT fixed-shape forward, no JIT | the cold spike (R5/σ-3), host-contention σ spread, the σ↔next-B coupling |
| DOF-7 soft-cap overrun (B>max_batch) | server | cap tested on PRE-request total; request appended whole | test the POST-request total (peek B_i; defer over-cap to next drain) — a HARD cap | R5 entirely; the second compiled shape; every forward stays at the single padded shape |
| DOF-8 between-batch weight reload | server | poll() may return new params iff the version advanced | construct with StaticParamsSource (poll≡None) and never republish | the inter-batch version step (no version straddle is possible either way, SG3) |
| DOF-9 scatter drop (ROUTER_MANDATORY off) | server | a full/vanished-peer send is silently dropped | set ROUTER_MANDATORY=1 and handle EHOSTUNREACH loudly | the silent-drop→peer-RCVTIMEO path; the drop becomes a loud server-side error at scatter time |
| DOF-10 reject-only empty drain | server | an all-malformed wakeup does work (rejects) but issues no forward/reply | make RELY R1 a hard invariant (verified-codegen wire both sides) | the reject path + empty-drain re-poll (R7-server) |
| DOF-11 idx-to-slot / subset-exhaustion | producer | which slot picks up the next shared `next_idx` (data-dependent) | a fixed idx-to-slot mapping instead of the shared `next_idx` | the data-dependent interleaving of which slot runs which idx |

The single most consequential design change: **removing DOF-3's non-latitude is already done by
the code** (the dead D knob); the change that would make the pipelined driver actually use D>1
is the inverse — to make a slot ready BETWEEN `issue_one` calls during prime/refill (e.g. issue
one message per ready slot rather than coalesce-all, OR run fibers asynchronously off the recv
thread). That change would move R2's coalescing from cross-thread-only to also within-thread,
and is the design's stated overcommit direction (the §6 restructure on the current branch). The
model faithfully represents the CURRENT code, where that change has not landed.

---

## 7. Minimal fidelity requirements (any faithful model of this boundary MUST satisfy)

1. **Per-thread in-flight depth is identically 1** (coalesce-all `issue_one` + synchronous
   recv→resume→refill). A model that admits D>1 messages outstanding per producer thread, or
   producer-observed out-of-order reply routing on one socket, is TOO PERMISSIVE.
2. **All batch-size growth and all concurrency the server sees is cross-thread** (≤ T messages
   outstanding, one per thread). A model that attributes batching to per-thread pipelining is
   unfaithful.
3. **Forward service time is positive, never instantaneous, and shape-invariant in B** (padding
   to one `(max_batch,in_dim)` shape), with a one-time cold-compile tail on an un-warmed server
   and an unpadded overrun branch for B>max_batch. A model that grows σ with B (over-constraint)
   or collapses σ to a constant/instant (over-constraint) or zero-cost (over-permissive) is
   unfaithful.
4. **Source emission δ is a positive bounded-nondeterministic interval**, reply-gated per slot
   (δ-2), with no producer-imposed upper bound. Pinning δ to a constant pins S and forbids the
   staggered-arrival regimes (over-constraint).
5. **Forwards are totally ordered and non-overlapping** (single-threaded server). Concurrent
   forwards are unrepresentable.
6. **The ROUTER scatter is non-blocking** (ROUTER_MANDATORY=0 ⇒ drop, never block). A
   send-blocks-on-full wedge is a phantom (too-permissive); forbidding the silent known-peer drop
   is too-constrained.
7. **The sole producer blocking point and sole liveness backstop is the RCVTIMEO-bounded recv**;
   the SNDHWM send-block is unreachable under depth-1 (D·T ≪ 1000). A faithful model has exactly
   one place a producer thread can block, and it is bounded.
8. **Loud bilateral abort, never a silent hang or a wrong-slot apply** on any
   recv/decode/corr-id/count error or server termination; never a partial write.
9. **A reachable EXCEPTIONAL_TERMINATION** server terminal (ragged in_dim / bad shape / reload
   raise), RELY-gated, distinct from the clean `_stop` shutdown.
10. **The soft cap is tested on the PRE-request total**, so a multi-row request crosses it whole
    (post-batch total ∈ [max_batch, max_batch + B_last − 1]).

---

## 8. Code-derivation attestation

The composed model is derived purely FORWARD from the operational semantics of the thirteen
files listed in the preamble, each read end to end for this synthesis (ADR-0002, special force
for LLM collaborators). Every state, transition, guard, free-choice flag, RELY, GUARANTEE, DOF,
and timing constraint maps to a specific cited line or to a named causal necessity (positivity
of durations; per-slot reply-dependence via the suspended fiber; reply-after-forward; total
ordering of forwards on the single-threaded server; per-thread depth-1 via coalesce-all +
synchronous drain; RCVTIMEO as the sole bounded blocking point).

The two fidelity-critical socket facts were read directly from the code, not assumed:
- The **DEALER** sets ONLY `ZMQ_LINGER=0` and `ZMQ_RCVTIMEO=timeout_ms` (wire_leaf_pool.hpp:81-83);
  SNDHWM/RCVHWM/SNDTIMEO and all context-level options are at libzmq-4.3.5 defaults
  (1000/1000/−1), so the recv is the only place a producer thread can block at these depths.
- The **ROUTER** sets NO socket options at all (inference_server.py:315-318), so
  ROUTER_MANDATORY=0 → scatter DROPS, never blocks; this is what settles the server models'
  one disagreement (Model B's E7 block-wedge is a phantom).

No outside expectation of how an inference/transport service "ought" to behave was introduced.
Where the code DETERMINES a choice it is determined (one forward per drained batch; scatter
strictly after the forward; pad to one shape; reload only between batches; coalesce-all gather;
per-thread depth-1); where the code LEAVES latitude it is left exactly (δ, σ, cross-thread
arrival overlap, S∈[1,K], reject/accept, reload timing, soft-cap overrun, idx-to-slot, the
scatter-drop outcome). The system was NOT run and no solver sweep was executed.

**Bounded confirmation (theory's check, not its source).** One Z3 4.16 script
(`check_composed_admissible.py`) was run under `nice -n 19 timeout 90`. It encodes the composed
seam's two load-bearing facts and confirms BOTH:
- **(A) SAT** — the cross-thread depth-1 coalescing overlap (R2): two producer threads each at
  depth-1, the second's message buffered during the first's forward, forwards serialized and
  non-overlapping, replies after forwards, re-issue after reply. The real operating point is
  admissible (the model is not vacuously over-constrained at the seam).
- **(B) UNSAT** — a single producer thread holding two messages simultaneously outstanding, under
  the control-flow law `sub2 > rcv1` (a thread's second submit follows its first reply). Depth is
  identically 1 (the model is not over-permissive at the seam — it forbids the phantom per-thread
  pipeline both audits flagged).

---

## 9. Open questions the model cannot settle

1. **Numeric timing.** δ and σ are bounded-nondeterministic positives; the model does not pin
   numbers. The project is explicitly uncalibrated-time (design §9). Whether the system sits in
   R1, R2, or R3 in practice depends on the actual δ/σ ratio and T — which the MEMORY note's
   "~50 dps JAX-batched-over-ZMQ" hints at but the code does not fix. The model says which
   regimes EXIST and how they transition, not which one a given run occupies.
2. **The 4-vCPU parallelism ceiling.** The code imposes no bound on T-way cross-thread
   parallelism (T independent `std::thread`s); the ~1.9× ceiling is a host operational fact, not
   a code constraint, so the model leaves T-way parallelism free. Whether R2's cross-thread
   overlap actually amortizes the forward depends on the host scheduler, which is outside the
   code.
3. **Whether EXCEPTIONAL_TERMINATION is reachable in production.** It is reachable only on a
   RELY-violating input (ragged `in_dim` requires two requests with different feature dims in one
   drain; a bad forward shape requires a forward bug; a reload raise requires a malformed
   manifest). Under the drift-checked single-net wire (every leaf of one net shares in_dim) the
   ragged path is unreachable; the model cannot rule it out from the transport code alone (it
   depends on whether two different nets' workers can ever address one server — outside this
   boundary).
4. **The overcommit restructure on the current branch.** The git log shows an in-progress §6
   overcommit design (N trees/thread + 1:3 pinning) that would make a slot ready between
   `issue_one` calls and thus make D>1 reachable within a thread. This model is faithful to the
   CURRENT code (depth-1); it cannot speak to the restructured driver, which would move R2's
   coalescing partly within-thread and is explicitly out of scope.
5. **Server-side fairness across peers under saturation.** The ROUTER fair-queues across peer
   pipes, but the drain order among simultaneously-queued frames is libzmq's, not pinned by the
   server code (DOF-4). Whether a slow producer thread can be starved of forward slots under
   sustained R3 saturation depends on libzmq's fair-queue internals, which the server code does
   not constrain and this model leaves free.

---

*Public Domain (The Unlicense).*
