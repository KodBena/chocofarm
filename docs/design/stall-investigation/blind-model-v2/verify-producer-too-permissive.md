# Adversarial fidelity verification — C++ producer side — lens: TOO PERMISSIVE

Verifier role: adversarial fidelity VERIFIER for the C++ producer side
(`run_episodes_wire_pipelined` + `WireLeafPool` DEALER). Lens: find executions a
model ADMITS that the cleanroom code cannot produce.

All file:line references are to the cleanroom tree
`/home/bork/w/vdc/chocobo/runs/leaf-eval-model-2/cleanroom`. I read end to end:
`cpp/include/chocofarm/wire_leaf_pool.hpp`, `cpp/src/runner_wire_batched.cpp`,
`cpp/include/chocofarm/runner_wire_batched.hpp`, `inference_wire.hpp`, `wire_spec.hpp`,
`fiber_tree.hpp`, `fiber_leaf.hpp`, `runtime_config.hpp`, `error.hpp`, `net_evaluator.hpp`,
and the peer `chocofarm/az/inference_server.py`, `cpp/stage_a/stage_a_server.py`,
`chocofarm/az/inference_wire.py`, `chocofarm/az/forward.py`, `chocofarm/config.py`.

I also read the two models' Z3 confirmation scripts already on disk
(`out/producer_check.py`, `out/producer_transport_check.py`) because both models cite
them as the substantiation for their headline representative executions.

---

## Headline (both models share the same fatal over-permission)

The single decisive too-permissive defect is shared by **both** producer models:

> Both treat **reply arrival order vs send order as a free permutation for ONE
> thread's own D outstanding messages** (model 1 DOF-4; model 2 DOF-T6), and both
> stake their flagship representative execution + Z3 "confirmation" on a
> **single-DEALER (T=1)** out-of-order reply (`fwd["c1"] < fwd["c0"]`,
> `recv(c1) < recv(c0)`).

This admits executions the code cannot produce. A single `WireLeafPool` is one
`ZMQ_DEALER` socket (`wire_leaf_pool.hpp:35`) connected to one `ZMQ_ROUTER`
(`inference_server.py:153`). ZeroMQ preserves message order **per pipe** in both
directions. For a single DEALER:

1. The producer sends multipart messages `[corr|payload]` in `corr_seq` order
   (`wire_leaf_pool.hpp:84,86,89`).
2. The ROUTER receives that one DEALER's messages **FIFO** and `_drain` pulls them
   with `recv_multipart` in arrival order (`inference_server.py:173`, appended to
   `drained` in that order at `:184`).
3. `_serve_batch` zips `run_microbatch(...)` output with `drained` and
   `send_multipart` **in drain order** (`inference_server.py:197-200`);
   `run_microbatch` preserves request order (`identities`/`counts` built in order
   at `:46,:54,:68`). The stage_a variants preserve it too: `wakeup="group"` is one
   ordered group, `wakeup="leaf"` is one forward per request still emitted in drain
   order (`stage_a_server.py:57,65,69-70`).
4. The reply pipe ROUTER→DEALER is FIFO, so the producer receives replies to its
   own messages **in send order**.

Therefore a thread's own outstanding replies arrive **strictly FIFO (= send
order)**. Reorder is real **only across distinct DEALERs** (different threads),
because those are different pipes the single-threaded server interleaves in drain
order. The models grant the reorder latitude to *one thread's* stream, which the
transport forbids. Both models' flagship representative execution is a **T=1**
reorder, and both Z3 scripts assert `fwd["c1"] < fwd["c0"]`
(`producer_check.py:82`) / drain order `1,3,2` for one socket
(`producer_transport_check.py:63-65`) — i.e. the "confirmation" certifies the very
inadmissible execution, so it cannot catch the defect.

Model 1's own self-audit half-sees this ("on a single DEALER<->ROUTER pipe ZMQ is
FIFO ... I left the latitude on the producer side") but then keeps the latitude AND
ships rep-exec #1 (T=1, K=2) asserting the server "replies c1 before c0". Seeing
the constraint and declining to impose it is precisely the admit-too-much this lens
targets.

**Confirmation (minimal Z3, `out/verify_producer_fifo_check.py`, run under
`nice -n 19 timeout 90`).** Part A encodes the models' claim (one DEALER, send order
`send0 < send1`, reply-after-forward, and the reversed `recv1 < recv0` from
`producer_check.py:87`): **sat** — the model admits it. Part B adds only the code's
real constraint (FIFO per pipe ⇒ `recv0 < recv1` for one DEALER): **unsat**. The
SAT/UNSAT gap is the over-permission itself. (Confirmation only, not the source of
trust; the FIFO derivation above is.)

---

## Model 1 (`model-producer-pacing.md`) — findings

### F1.1 [FATAL] Single-thread out-of-order reply admitted (DOF-4 + rep-exec #1 + Z3)
- model_element: DOF-4 "reply arrival order != send order"; representative
  execution #1 steps 2,4; validation_run.
- expected_code_ref: `wire_leaf_pool.hpp:35` (one DEALER), `inference_server.py:173,197-200`
  (FIFO drain + in-order send), `producer_check.py:82,87`.
- defect: timing-over-permissive / admits-too-much.
- For T=1 the model admits `recv(c1) < recv(c0)` with `send(c0) < send(c1)`. The
  single DEALER↔ROUTER pipe is FIFO both ways; the single-threaded server drains and
  replies in arrival order, so this thread's replies are FIFO. The execution is not
  realizable.
- correction: Restrict reorder to **across DEALERs only** (across threads). Within
  one thread, `recv` order == `send` order (FIFO). Replace rep-exec #1 with a T≥2
  cross-thread reorder, or drop the reorder and keep only the genuinely-free
  within-reply completion order (one coalesced reply scatters its rows in the
  server's row order, `wire_leaf_pool.hpp:126-130` — that IS free and N-relevant).

### F1.2 [MAJOR] Rep-exec #1 issues two single-slot messages from a simultaneously-eligible state
- model_element: representative execution #1, step 0 ("both SLOT_ELIGIBLE") → step 1
  ("issue_one sends msg c0 carrying slot0 ... a second issue sends c1 carrying slot1").
- expected_code_ref: `runner_wire_batched.cpp:437-444` (gather ALL is_ready into ONE
  message), `:456` (the prime loop's second `issue_one` finds nothing ready).
- defect: unjustified-free-choice / admits-too-much.
- `issue_one` coalesces **every** `is_ready` slot into a single message (no subset
  selection, no early break, `:437-444`). If both slots are eligible at the first
  `issue_one`, they go into ONE message (B=2, inflight=1); the second `issue_one`
  finds all `submitted` → returns false (`:444`). Two separate single-slot messages
  from a simultaneously-eligible pair is impossible. The model's own DOF-2 control
  states this ("issue_one is NOT free to pick a subset"), so the rep-exec
  contradicts the model's own invariant.
- This is *why* `producer_check.py` silently switches (docstring `:7-16`) to a
  staggered "2-message variant" without asserting both slots parked before the first
  send — the Z3 confirms a different (admissible) staggered scenario than the prose
  (simultaneous) claims. The prose execution is over-permissive.
- correction: Either (a) both slots eligible ⇒ one B=2 message (inflight reaches 1,
  not 2); or (b) state that slot1 parks *after* the first `issue_one` so its message
  is a separate refill message — then the two-message shape is admissible, but it is
  no longer the "both eligible at PRIMING" scenario the trace describes.

### F1.3 [MINOR] "B up to K = N*base ... overcommit's mechanism for raising server batch size, until clipped by server max_batch" (DOF-2 n_dependence)
- model_element: DOF-2 n_dependence sentence.
- expected_code_ref: `runner_wire_batched.cpp:444-451` (producer never clips),
  `inference_server.py:171,184-185` (drain row cap is checked *before* reading a
  whole message; one message is never split).
- defect: minor over-statement (timing/throughput characterization).
- The **producer** never clips B to `max_batch`; it can send a single message of
  B = K rows regardless of `max_batch`. The server's `max_batch` bounds the *drain's
  row total across messages* (`total_rows < self._max_batch` checked at loop top,
  `:171`, then a whole message is appended, `:184-185`), so one producer message of
  B > max_batch is accepted whole, not clipped. The phrase "until clipped by server
  max_batch" mis-locates a clip that does not happen to B; it is a peer-side
  throughput effect, not a cap on the producer's message size. Not a state-machine
  error; flagged so the N-dependence story is not read as "B is capped at
  max_batch".
- correction: "B's ceiling is K = N·base; the server never splits a single producer
  message, so B is uncapped on the wire — `max_batch` bounds only how many *messages*
  the server coalesces per forward, a peer effect."

Everything else in model 1 verified faithful under this lens: the gather-all
coalescing (B pinned to the eligible set, DOF-2 / control), the D-message cap
(`:456,:474`), reply-causal pacing (fiber blocks in `predict`, `fiber_leaf.hpp:27`,
until `resume_with`, `fiber_tree.hpp:58`), the at-most-one-new-eligible-per-reply
(one fiber/slot, `advance` reaches one park, `:389-397`), the fail-loud RCVTIMEO
(`wire_leaf_pool.hpp:41,147-150`), the unknown-corr / size-mismatch hard errors
(`:116-124`). The socket-option census (LINGER=0, RCVTIMEO set; SNDHWM/RCVHWM/
SNDTIMEO/ROUTER_MANDATORY default) matches the grep. The RELY set does not exceed
the peer's guarantees (R4 liveness is correctly a rely whose violation → FAILED).

---

## Model 2 (`model-producer-transport.md`) — findings

### F2.1 [FATAL] Single-thread reply order modeled as a free permutation (DOF-T6 + rep-exec #2 + Z3)
- model_element: DOF-T6 "reply arrival order vs submit order — any permutation";
  representative execution #2 (out-of-order drain corr 1,3,2); validation_run.
- expected_code_ref: `wire_leaf_pool.hpp:35`, `inference_server.py:173,197-200`,
  `producer_transport_check.py:63-65`.
- defect: timing-over-permissive / admits-too-much.
- Identical to F1.1: DOF-T6 says "Any permutation of outstanding-reply arrivals;
  producer tolerates all" and the Z3 certifies corr drain order `1,3,2` on **one**
  socket. The producer's *tolerance* (corr-keyed match, order-agnostic,
  `wire_leaf_pool.hpp:115`) is faithfully modeled, but the **set of realizable
  orders** for one DEALER is FIFO, not all permutations. The model's representable
  executions therefore exceed the system's.
- Model 2's self-audit explicitly notes "ZMQ is FIFO, so the actually-reachable
  orderings are a subset; I left the latitude on the producer side" — and then keeps
  it. The lens does not accept a knowingly-retained superset.
- correction: DOF-T6 should read "across DEALERs (threads) any interleaving; within
  one DEALER, reply order == send order (FIFO per pipe)". Within a single coalesced
  reply, the row scatter order is the only intra-thread free order
  (`wire_leaf_pool.hpp:126-130`).

### F2.2 [MAJOR] |inflight_| == inflight_msgs asserted as an invariant the code does not maintain in lockstep
- model_element: timing_model.causal_constraints "|inflight_| == inflight_msgs
  invariant on every non-error path"; state RECV_BLOCK/MATCH/RESUME.
- expected_code_ref: `wire_leaf_pool.hpp:92,120` (map emplace on send / erase on
  recv), `runner_wire_batched.cpp:448` (`++inflight_msgs`), `:460` (`--inflight_msgs`).
- defect: missing-blocking-semantics / over-permissive equality claim (it *over*-
  states what holds, which the model leans on for liveness reasoning).
- `inflight_msgs` is incremented in `issue_one` at `:448` after a successful
  `submit_batch`. The map entry is `emplace`d *inside* `submit_batch` at
  `wire_leaf_pool.hpp:92` — i.e. the map is updated before the counter. On the recv
  side the map `erase` happens inside `recv_batch` (`wire_leaf_pool.hpp:120`) and
  the counter decrements at `runner_wire_batched.cpp:460` *after* `recv_batch`
  returns. So at the instants between `:92`↔`:448` and `:120`↔`:460` the two are
  transiently unequal even on the success path, not only on ERR. More importantly,
  the producer **never reads `inflight_.size()`** to make a decision — it gates only
  on the integer `inflight_msgs` (`:456,:457,:474`). The map's job is corr→slots
  lookup, not depth accounting. Asserting an always-equal invariant is a claim the
  code does not enforce and does not need; a reader who relies on it could justify a
  step the code cannot take (e.g. reasoning "recv is enabled because the map is
  non-empty" when the integer is what gates).
- correction: State the actual invariant: the recv loop is gated solely by the
  integer `inflight_msgs ∈ [0,D]` (`:457`); `inflight_` is a corr→slots dictionary
  whose size equals `inflight_msgs` only between completed send/recv operations, and
  is never consulted for control flow.

### F2.3 [MINOR] SEND modeled as potentially indefinitely-blocking (DOF-T4) — a dead transition under the producer's own cap
- model_element: DOF-T4 "an infinite send-block (only if D>1000 and the peer never
  drains)"; state-machine transition SEND→ERR / SEND blocking regime.
- expected_code_ref: `wire_leaf_pool.hpp:86-91` (two `zmq_send`, no DONTWAIT, no
  SNDTIMEO), `runner_wire_batched.cpp:456,474` (cap D), `runner_wire_batched.hpp:24`
  (`max_inflight_msgs = 8` default).
- defect: over-permissive (admits a blocking SEND the cap makes unreachable for
  realistic D), but the model already fences it as "unreachable-by-default".
- A DEALER's outbound queue can hold at most D unacked messages because the producer
  blocks in recv once `inflight_msgs == D` and only sends while `< D`. With default
  SNDHWM=1000, send blocks on HWM only if outstanding > 1000, i.e. only if D > 1000.
  The model keeps the transition "because the code's defaults permit it for some
  configuration" — that is defensible, but the transition is dead for every
  shipping config (default D=8). Flagged minor: a model whose state graph includes a
  transition that no in-range parameter reaches is mildly over-permissive about the
  reachable state set; it should be marked dead-unless-D>SNDHWM rather than a live
  blocking regime.
- correction: Mark the SEND-blocks branch reachable **iff D > SNDHWM (default 1000)**;
  for the parameter range the system uses (D ≤ ~1000) SEND is non-blocking in every
  reachable state, so the only live bounded wait is RECV_BLOCK under RCVTIMEO.

### F2.4 [MINOR] Rep-exec "Staggered waves fill D" implies separate messages per wave-slot without stating the staggering precisely
- model_element: representative execution #2, steps 1-2 ("issue_one #1 gathers
  wave-1 ready slots; ... issue_one #2..#D gather later-parked slots").
- expected_code_ref: `runner_wire_batched.cpp:437-444,456`.
- defect: minor under-specification that risks admitting the F1.2 shape.
- This trace is admissible **only** because it explicitly says each `issue_one`
  gathers a *different* wave (slots that parked between issues), so each message
  carries the then-eligible set. That is correct (unlike model 1's rep-exec #1).
  Flagged only to contrast: the D messages in flight arise from *temporal*
  staggering of parks, never from splitting a simultaneously-eligible set. The model
  states this adequately; no correction required beyond keeping the "later-parked"
  qualifier load-bearing.

Everything else in model 2 verified faithful: the socket-option census (verified by
grep — only LINGER=0 + RCVTIMEO; server sets none), G1-G4 guarantees, the corr
uniqueness via shared atomic `fetch_add` (`wire_leaf_pool.hpp:84`,
`runner_wire_batched.cpp:298`), the fail-loud desync guards, the copy-before-resume
dangling-span constraint (gather copies `ts->ch.features` at `:439-440` before
`resume_with`), the RCVTIMEO liveness backstop. The RELY set (R1-R5) is checkable
against and does not exceed the peer's guarantees.

---

## Trace check (per model)

Model 1: representative executions are NOT all admissible. Rep-exec #1 breaks at
step 1 (two single-slot messages from a both-eligible state — forbidden by gather-all
`:437-444`) and at step 2 (single-thread out-of-order reply — forbidden by FIFO per
pipe). Rep-exec #2 (underfilled pipeline) and #3 (RCVTIMEO failure) are admissible.

Model 2: rep-exec #2 ("staggered waves fill D") breaks at the out-of-order-drain
claim (single socket, corr `1,3,2` — forbidden by FIFO). Rep-exec #1 (all-ready
single-shot), #3 (timeout abort), #4 (desync guard) are admissible. The single-shot
trace is the *correct* depiction of the both-eligible case (one B=K message,
inflight=1, D under-used) — and it is exactly the shape model 1's rep-exec #1 should
have used.

## Timing fidelity (per model)

Both keep the park interval `delta_park > 0` as positive bounded nondeterminism (not
collapsed) and the forward `S_fwd > 0` with code-derived structure (constant-in-rows
under padmax pad-to-`max_batch` `inference_server.py:198`; 3-step bucket
`stage_a_server.py:30-37,63-64`; per-leaf under `wakeup=leaf`). No duration is
collapsed to a constant or instant — faithful. The single defect in the timing
*ordering* is the reorder over-permission (F1.1/F2.1): the timing model lets a
single thread's `recv` times reorder relative to `send`, which FIFO-per-pipe forbids.

## N-dependence check (per model)

Both derive K = N·`fibers_per_thread()` from `runner_wire_batched.cpp:286`, where
`fibers_per_thread() = ceil(max(1,pool_batch)/max(1,pool_threads))`
(`runtime_config.hpp:12-15`) = "base". This is faithfully derived, not asserted. The
downstream corollaries (B's ceiling K, in-flight rows ~ D·B, longer tail) follow.
Two N-dependence over-statements, both minor: model 1 F1.3 ("clipped by server
max_batch" — the producer's B is not clipped); and both models' implicit claim that
larger N makes single-thread reorder "more reachable per unit work" (DOF-4/DOF-T6
n_dependence) — single-thread reorder is not reachable at any N, so its
N-dependence is vacuous, not merely "structurally constant". The protocol-invariance
claim (transport protocol invariant in N; N moves only the statistics) is otherwise
correct.

## Cross-model note

The two models are the same producer derived twice (pacing-centric vs
transport-centric) and share one fatal over-permission: **single-thread reply
reorder** (F1.1 ≡ F2.1), each propped up by a Z3 script that certifies the
inadmissible execution rather than catching it (`producer_check.py:82,87`;
`producer_transport_check.py:63-65`). The correct, FIFO-respecting depiction already
exists *inside the pair* — model 2's "all-ready single-shot" rep-exec — which both
should adopt for the both-eligible case, reserving reorder for cross-DEALER (T≥2)
interleavings. Model 1 additionally over-permits message *splitting* of a
simultaneously-eligible set (F1.2); model 2 gets the coalescing right but
over-asserts a map/counter lockstep invariant the code neither maintains nor reads
(F2.2). Net: both are too-permissive, dominated by the reorder defect; model 1 is
slightly worse (it also splits coalesced sends in its flagship trace).
