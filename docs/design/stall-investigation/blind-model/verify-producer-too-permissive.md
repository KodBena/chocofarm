# Adversarial fidelity audit — C++ producer side, TOO-PERMISSIVE lens

Verifier role: find executions a model ADMITS that the real code cannot produce.
Side: producer (search source + per-thread DEALER `WireLeafPool` + pipelined / strict-barrier driver).
Reading: all cited files read end to end (ADR-0002). Socket-option call sites enumerated by grep over
`cpp/` + the Python server/client; the only `zmq_setsockopt` on the producer DEALER are
`ZMQ_LINGER=0` and `ZMQ_RCVTIMEO=timeout_ms` (`wire_leaf_pool.hpp:82-83`); the server ROUTER sets no
HWM/option (`inference_server.py:315-316`). Both models' blocking-surface premise is therefore correct.

The two models are referred to as **M1** (`model-producer-pacing.md`) and **M2**
(`model-producer-transport.md`).

---

## The load-bearing defect (shared by both models): D>1 / pipeline overlap / out-of-order replies are UNREACHABLE in this driver

This is the single most important too-permissive finding, and it is structural, not a corner case.

### The code fact

`issue_one` (`runner_wire_batched.cpp:551-569`) gathers **every** currently-ready slot into **ONE**
`submit_batch` (one corr-id), then `++inflight_msgs`. It does not issue one-message-per-slot; it
coalesces the entire ready set into a single message each call.

The prime loop is `while (inflight_msgs < D && issue_one()) {}` (`:578`). Crucially, before this loop,
`fill(s)` has run **synchronously** for all `s` in `0..K-1` (`:572`), leaving every fillable slot
PARKED with `submitted[s]==0`. No fiber runs between the `fill` loop and the prime loop, and no fiber
runs between successive `issue_one()` calls inside the prime loop (there is no `recv`, no
`resume_with`). Therefore:

- The **first** `issue_one()` gathers ALL parked slots into ONE message → `inflight_msgs == 1`.
- The **second** `issue_one()` finds `is_ready(s)` false for every slot (all now `submitted[s]==1`,
  `:564`) → `gathered.empty()` → returns false (`:561`).
- The prime loop exits at **`inflight_msgs == 1`**, regardless of `D`.

In steady state the same coalescing holds. After `recv_batch` resolves the one outstanding message
(`--inflight_msgs` → 0, `:582`), the completion loop resumes those slots **synchronously** to a re-park
or finalize (`advance`/`fill` run to completion, `:589-593`). Only then does refill run
(`while (inflight_msgs < D && issue_one())`, `:596`): its first `issue_one` gathers ALL the now-ready
slots into ONE message (`inflight_msgs` → 1), and its second finds nothing ready → false. Depth returns
to 1.

There is no interleaving in which `issue_one` runs while another message is genuinely outstanding **and**
a fresh slot is ready that was not in that message: a slot only becomes ready by a fiber running
(`resume_with`/`fill`/`advance`), which only happens inside the completion loop **after** the recv that
already decremented the outstanding message. Hence **within a single worker thread `inflight_msgs` never
exceeds 1**; `D = max(1, wcfg.max_inflight_msgs)` is a dead knob in this code path.

Consequence chain, all unreachable within a thread:

1. **D>1 / multiple messages outstanding per thread.** Never happens.
2. **Out-of-order reply routing exercised by the producer.** `recv_batch` always returns the single
   outstanding corr-id; `inflight_.find(corr)` (`wire_leaf_pool.hpp:179`) always finds the one entry.
   The corr-id-keyed routing is *correct* and *necessary for safety*, but the **reordering latitude it
   tolerates is never realized** because the producer never has two replies pending on one socket.
   (Each thread owns its own DEALER, so there is no cross-thread reordering on a single socket either.)
3. **Real RTT overlap / "the search never idles the full RTT".** Within a thread the search idles the
   full round-trip every round, exactly like the strict barrier — because there is only ever one message
   in flight and the loop blocks on its reply.

### What the models claim instead

- **M1** stance: "holds up to D outstanding, routing out-of-order replies by corr-id"; DOF-3 "reply
  arrival order across the D outstanding messages"; DOF-4 "D=1 … up to D large (many messages overlap
  the forward)"; representative execution **E2** ("the pipe primes with three messages (S=2,1,1); c1's
  reply lands before c0's … CONFIRMED ADMISSIBLE by the bounded Z3 check") and **E5** ("messages of
  mixed S coalesce … D=8"). M1's E2 step 2 explicitly issues three messages during prime and step 3
  delivers c1 before c0.
- **M2** stance: "keep up to D = wcfg.max_inflight_msgs COALESCED messages outstanding"; DOF-3
  "0<=inflight_msgs<=D … up to full depth D=8"; DOF-4 reply reordering; representative execution **E1**
  ("prime to depth D=8") and **E3** ("two messages outstanding (c1 over {2,5}, c2 over {3}); sink
  replies c2 then c1").

Every one of these admits a producer execution the code cannot produce **within a single worker
thread**: the prime depth is 1, not D; only one message is ever outstanding per thread; the producer
never observes an out-of-order reply on its socket.

### Why the Z3 checks did not catch it

Both validation runs encoded the *causal* constraints (positivity, request-before-reply,
reply-after-forward, single-server forward ordering, out-of-order send/recv) and found them SAT. That is
true but irrelevant to fidelity: causality permits out-of-order replies; the **producer's own control
flow** (`issue_one` coalesce-all + the synchronous recv→resume→refill structure) forbids ever having two
messages outstanding to be reordered. The Z3 encodings modeled the wire, not the driver's
message-issuing structure, so they confirmed an admissibility the actual `issue_one`/drain code removes.
This is the classic "the model relied on a freedom the code removes" failure the lens targets.

### Correction

The faithful producer model has **per-thread in-flight depth identically 1** for this code: each round
issues exactly one coalesced message of S = #ready slots, blocks on its single reply, resumes, refills.
`D` constrains nothing reachable. The "pipeline overlap", "out-of-order reply", and "D>1" behaviors
must be removed from the set of admissible producer executions (they are reachable only by a *different*
issue policy — one message per slot, or a non-coalescing `issue_one` — which this code does not
implement). The corr-id routing should be modeled as *capable of* reordering but *never exercised* by
this producer (a guarantee the producer offers the peer, not a behavior it produces). The cross-thread
"co-batching" at the SERVER is real (multiple threads' single messages drain together); that is a SINK
behavior, not producer pipeline depth, and must not be conflated with per-thread D.

Severity: **fatal** for both models — the central distinguishing behavior each model advertises
(pipelined, multi-outstanding, out-of-order) is not in the code's reachable set, so each model's
representable-execution set strictly exceeds the code's.

---

## Per-model findings

### M1 (`model-producer-pacing.md`)

1. **`is_free_choice` on PARKED→OUTSTANDING inclusion / DOF-1 coalescing degree at prime.**
   `expected_code_ref`: `runner_wire_batched.cpp:554-561, :572, :578`.
   M1 says S "ranges 1..K nondeterministically" and E2 prime issues S=2,1,1. At **prime**, S is
   determined, not free: it equals the number of slots `fill` parked, and the first message takes ALL of
   them in one gather. S can be <K only post-prime, when a strict subset of slots is ready (others still
   `submitted`). M1 over-states the latitude by letting prime stagger into several small-S messages.
   Defect type: timing-over-permissive / unjustified-free-choice. Severity: major.

2. **DOF-3 / DOF-4 / E2 / E5 — the D>1 pipeline (see the load-bearing defect).** Severity: fatal.

3. **OUTSTANDING→ADVANCING `is_free_choice: true`, "any of the D outstanding may land first".**
   `expected_code_ref`: `wire_leaf_pool.hpp:170-196`, `runner_wire_batched.cpp:580-584`.
   With depth always 1, "which of the D lands first" is vacuous; the recv is deterministic in *which*
   corr-id returns (the only one). The genuine free choice on this edge is the SINK service duration and
   the reply *timing*, not arrival *order*. Defect type: unfaithful-rely / timing-over-permissive.
   Severity: major.

4. **Strict barrier described as "the D=1 specialization" of the pipelined driver.**
   `expected_code_ref`: `runner_wire_batched.cpp:66-67, 310-337`.
   The production default is `run_episodes_wire_batched`'s own body (selected when
   `wcfg.mode != PipelinedBucket`, `:66`), which gathers all parked into one message and blocks — and
   since the pipelined driver is *also* depth-1 (finding above), the two are behaviorally the same, but
   M1's framing implies the pipelined driver's D-knob produces the strict barrier as a special case when
   in fact the pipelined driver never leaves depth 1 either. The claim is accidentally true (both are
   depth-1) but for the wrong reason; it should be restated. Defect type: mis-mapped-transition.
   Severity: minor.

5. **Within-thread serialization correctly modeled; cross-thread parallelism correctly left free.**
   `code_ref`: `runner_wire_batched.cpp:604` (T independent `std::thread`s), boost.context cooperative
   fibers. No defect — this latitude is real and the model leaves it. (Recorded as a non-finding so the
   audit is not all-negative.)

### M2 (`model-producer-transport.md`)

1. **PARKED→INFLIGHT `is_free_choice: false` with "WHICH slots are ready together is fixed by … relative
   completion order".** `expected_code_ref`: `runner_wire_batched.cpp:551-569`.
   This edge is modeled more carefully than M1 (it does NOT call the inclusion a free choice, and it
   correctly notes the gather is determined given the ready set). Good. But M2 still inherits the prime-
   depth error in E1 (below). Defect type: none on this edge specifically.

2. **`PRIMING` "issue_one until inflight_msgs==D" + E1 "fill K=8 slots, issue_one until
   inflight_msgs=D=8".** `expected_code_ref`: `runner_wire_batched.cpp:572, 578`.
   The prime reaches depth **1** (one coalesced message of S=K), not D=8 — the synchronous fill leaves
   all K parked, the first `issue_one` takes all of them, the second finds nothing ready. Defect type:
   timing-over-permissive / mis-mapped-transition. Severity: fatal (this is the load-bearing defect as
   it lands in M2's flagship steady-state trace E1).

3. **DOF-3 "0<=inflight_msgs<=D … up to full depth D=8" and DOF-4 / E3 out-of-order reply.**
   `expected_code_ref`: `runner_wire_batched.cpp:578, 596` (the coalesce-all `issue_one`).
   E3 posits two messages (c1 over {2,5}, c2 over {3}) outstanding on one thread's socket and the sink
   replying c2-then-c1. This thread never has two messages outstanding (depth-1 finding), so E3 is not a
   producible producer execution. Defect type: missing-blocking-semantics / timing-over-permissive.
   Severity: fatal.

4. **`BLOCKED_SEND` state + `REFILLING→BLOCKED_SEND` `is_free_choice: true`.**
   `expected_code_ref`: `wire_leaf_pool.hpp:139-144` (no SNDHWM/SNDTIMEO set anywhere).
   This is **faithful, not over-permissive.** SNDHWM=1000 and SNDTIMEO=-1 are genuinely left at libzmq
   defaults, so a send-block IS in the code's representable set even if operationally unreachable under
   D-cap; with the depth-1 finding the producer holds at most one message per socket, so it is even
   further from SNDHWM — but the *option settings* permit the block, and M2 keeps it as a state without
   asserting it in a representative trace. No defect; recorded to pre-empt a false positive. (The depth-1
   finding makes it *more* unreachable, not less faithful.) Defect type: none. Severity: n/a.

5. **`tau_fwd` modeled batch-independent due to `pad_to=max_batch`.**
   `expected_code_ref`: `inference_server.py:171-172, 385`; `forward.py:36-63`.
   Verified against the code: `_serve_batch` passes `pad_to=self._max_batch` and `run_microbatch` pads
   UP to one `(max_batch,in_dim)` shape, so `jit_forward_core` compiles one executable. This is the
   correct faithful collapse (constant in B, NOT constant across calls). No defect. (Same for M1.)

6. **R5 "the sink may reply to outstanding corr-ids in any causally-consistent order."**
   `expected_code_ref`: `inference_server.py:348-363, 384-387`.
   As a RELY this is *weaker than the peer guarantees* (the single-threaded server replies in drain
   order within a batch), which is the safe direction for a rely (assuming less). But because the
   producer never has two messages outstanding per socket, the rely is also never load-bearing. M2 even
   self-audits this as "over-permission on the sink's side … harmless to the producer due to corr-id
   routing." Acceptable; not a producer-side over-permission. Defect type: none. Severity: minor (note
   only).

---

## Trace admissibility

### M1 traces
- **E1** (S=K barrier, T=1,K=4,D=2): admissible in shape, and it correctly notes "D>1 buys nothing".
  But it presents this as a *degeneration* of a pipeline that can do better, when in fact E1 is the ONLY
  regime this driver produces per thread. The trace steps are individually enabled. **Admissible.**
- **E2** (staggered prime into 3 messages, out-of-order c1<c0): step 2 (three messages at prime) and
  step 3 (out-of-order reply) are **NOT enabled** — prime issues one message; one message is ever
  outstanding. **Broken at step 2.**
- **E3** (RCVTIMEO loud abort): every step enabled (`wire_leaf_pool.hpp:217-221` → `:581` → `:609-614`).
  **Admissible.**
- **E4** (degenerate non-parking episode): enabled (`fill:527-535`, `advance:502-510`). **Admissible.**
- **E5** (mixed-S, D=8, multi-outstanding): the "D=8 … messages of S={3,2,1,1,1} until inflight=8"
  prime is **NOT enabled** (depth-1). The cross-thread server coalescing it describes is a SINK fact,
  not per-thread depth. **Broken at step 1.**

Overall M1: `traces_admissible = false` (E2 and E5, its two pipeline showcases, contain an
unenabled step; E1/E3/E4 are fine).

### M2 traces
- **E1** (steady pipelined drain, prime to depth D=8): **NOT enabled** at step 1 (prime reaches depth
  1). **Broken at step 1.**
- **E2** (source-bound underflow, depth 1-2): the depth "1-2" overstates; depth is exactly 1, and the
  underflow it describes (issue_one finds nothing ready, runs below D) is real but the framing "below D"
  presumes D>1 was reachable. Step 2 (`gathered.empty()→false`) is genuinely enabled; the *depth* label
  is wrong. **Admissible in steps, mislabeled in depth.**
- **E3** (out-of-order reply, two outstanding): **NOT enabled** (depth-1). **Broken at step 1.**
- **E4** (strict-barrier round, D=1): correctly maps to `run_episodes_wire_batched:310-337`. Every step
  enabled. **Admissible.**
- **E5** (RCVTIMEO loud abort): enabled. **Admissible.**
- **E6** (desync abort, unknown corr-id / B mismatch): enabled (`wire_leaf_pool.hpp:180-188`).
  **Admissible.**

Overall M2: `traces_admissible = false` (E1 and E3 contain an unenabled step).

---

## Timing fidelity

Both models model source think-time `delta` and sink service `tau_fwd` as positive bounded
nondeterministic intervals (not constants, not instants), correctly grounded:
- source emission free (`fiber_tree.hpp:99,106-107`; no driver-imposed interval) — faithful;
- sink service positive and **batch-independent** due to `pad_to=max_batch` one-shape compile
  (`inference_server.py:171-172`; `jit_forward_core` one executable) — the correct faithful collapse
  (growing tau with B would be the over-constraint);
- reply-after-forward, request-after-reply, single-threaded-server total order — all correctly stated.

The timing **values** are faithful. The timing-driven **structural** consequence is where both models
fail: they let staggered think-time produce *multiple outstanding messages per thread and out-of-order
replies*, which the coalesce-all `issue_one` + synchronous drain forbid. So the timing model is
internally faithful but is wired to a control-flow model that is too permissive (timing-over-permissive
at the structural level: real latitude in *durations*, invented latitude in *how many messages that
latitude can put in flight*).

---

## Verdicts

- **M1**: too-permissive. Faithful on the strict-barrier shape, the abort path, the degenerate path, the
  timing intervals, and the assume-guarantee codec/corr-id reasoning; but it admits a per-thread
  pipeline (D>1, multi-outstanding, out-of-order replies) the `issue_one` coalesce-all structure removes.
- **M2**: too-permissive, for the same structural reason; marginally more careful on the
  PARKED→INFLIGHT free-choice flag and on the send-block fidelity, but its flagship steady-state trace
  E1 and its out-of-order E3 both assert prime/maintain depth D>1, which is unreachable.

Both models are too-permissive in the SAME load-bearing way; neither is too-constrained in any way I
found (their abort, degenerate, and codec-rejection coverage is complete and the timing intervals are
left appropriately free).
