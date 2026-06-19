# Adversarial fidelity verification — C++ producer side — lens: TOO-CONSTRAINED

Role: adversarial verifier for the C++ producer (`run_episodes_wire_pipelined` + `WireLeafPool`
DEALER). Lens: find executions the code CAN produce that a model FORBIDS (forced orderings,
pinned counts, lockstep, dropped blocking/timeout outcomes, timing collapsed to a constant, an
N-dependence that understates/mis-states what is fixed-vs-growing).

All file:line refs are into `/home/bork/w/vdc/chocobo/runs/leaf-eval-model-2/cleanroom`.
I read all 15 cleanroom files end to end before judging
(`runner_wire_batched.cpp`, `wire_leaf_pool.hpp`, `runner_wire_batched.hpp`, `inference_wire.hpp`,
`wire_spec.hpp`, `fiber_tree.hpp`, `fiber_leaf.hpp`, `runtime_config.hpp`, `error.hpp`,
`net_evaluator.hpp`; `chocofarm/az/inference_server.py`, `inference_wire.py`, `forward.py`,
`config.py`; `cpp/stage_a/stage_a_server.py`).

## 0. The load-bearing operational fact both models mis-handle: inflight_msgs ∈ {0,1}

This governs the two strongest findings, so I establish it first, purely from the code.

`issue_one` (runner_wire_batched.cpp:434-452) gathers **every** `is_ready(s)` slot into **one**
message (loop 437-444, no early break, no subset selection), then `++inflight_msgs` once
(line 448). A slot becomes `is_ready` only via `running` being set true inside
`resume_with`/`advance`/`fill` (fiber_tree.hpp:55,61), and `submitted[s]` is cleared only at
line 466 inside the completion loop.

The worker thread is single-threaded over its own slots. Between two consecutive `issue_one`
calls **with no intervening `recv_batch`**, no slot can change state:

- PRIME loop, line 456 `while (inflight_msgs < D && issue_one()) {}`: the first `issue_one`
  submits **all** ready slots (`submitted[s]=1`, line 447). The second `issue_one` scans
  `is_ready` (line 438): every previously-ready slot now has `submitted[s]==1` → not ready;
  nothing re-parked (no recv ran). So `gathered.empty()` → `issue_one` returns false (line 444).
  **PRIME issues exactly one message and exits with `inflight_msgs == 1`, for every K, every N,
  every D ≥ 1.**
- REFILL loop, line 474 `while (inflight_msgs < D && !failed && issue_one()) {}`: identical
  argument. After one `recv` (inflight→0) the completion loop (462-472) re-parks ≤ B slots; the
  first refill `issue_one` gathers all of them into one message (inflight→1); the second finds
  none ready → false. **REFILL issues exactly one message per drain; inflight returns to 1.**

Therefore **`inflight_msgs ∈ {0,1}` in every reachable state, for all N, T, and all D ≥ 1.** The
cap D (runner_wire_batched.cpp:287, gates 456/474) is a **dead parameter**: it never binds because
the per-`issue_one` single-coalesced-gather can never leave a second message-worth of ready slots
without an intervening recv. Multi-message in-flight (depth 2..D) is **unreachable**.

Both models build their large-N story on inflight climbing toward D and frame inflight=1 as a
small-N / synchronized corner that is *escaped* as N grows. That inverts the truth: inflight=1 is
**universal**. Concretely this **forbids** the inflight=1-at-large-N executions the code always
produces — a too-constrained defect (recorded as findings P1/Q1 below). (The dual error — admitting
the unreachable depth-2..D states — is over-permissive and out of my lens; I record it in the
cross-model note for the producer's benefit.)

## 1. Verified socket-option facts (both models correct here)

DEALER sets exactly `ZMQ_LINGER=0` (wire_leaf_pool.hpp:40) and `ZMQ_RCVTIMEO=timeout_ms`
(wire_leaf_pool.hpp:41). Nothing else: no SNDHWM, RCVHWM, SNDTIMEO, ROUTER_MANDATORY, context
option (grep over the whole cleanroom confirms; only those two `setsockopt` calls exist). Server
ROUTER (inference_server.py:153-154) sets no socket options. So:

- Send (wire_leaf_pool.hpp:86,89) has no DONTWAIT, no SNDTIMEO (default -1 = block forever),
  SNDHWM default 1000. Because inflight ≤ 1 (§0), the outbound queue holds ≤ 1 message at any
  time, so send **never** blocks **regardless of D** — even D > 1000. Model 1's G3 conclusion
  ("never blocks on send") is therefore faithful, though its *stated reason* ("SNDHWM 1000 >> D=8")
  is wrong (the real reason is inflight ≤ 1, not the HWM headroom); model 2's DOF-T4 send-block
  transition at D>1000 is in fact **unreachable** (over-permissive, not my lens). Neither is a
  too-constrained finding, so I raise none here.
- `corr_seq` is one shared atomic `fetch_add` (runner_wire_batched.cpp:298; wire_leaf_pool.hpp:84)
  → globally unique monotone corr ids across all T threads. Both models correct.
- recv error of ANY kind (EAGAIN/ETERM/<2 frames/bad leading size) → set_error → break
  (runner_wire_batched.cpp:459; wire_leaf_pool.hpp:147-161). Both models correct.

## 2. Findings — Model 1 (model-producer-pacing.md)

### P1 [major, wrong-n-dependence] inflight=1 is universal, not a small-N corner that "disappears"
- model_element: representative_executions[1] ("Underfilled pipeline K≤D"), n_dependence:
  *"Disappears as N grows; once N*base>D the prime fills all D message-slots and the pipeline
  saturates ... escaped monotonically as N increases past D/base."* Also DOF-3 ("rows-in-flight
  ~ D*(typical B)") and DOF-5 ("the prime alone can issue D large messages").
- expected_code_ref: runner_wire_batched.cpp:456 and 474 (the two `while (inflight<D && issue_one())`
  loops) crossed with 437-444 (single-coalesced gather) and 447 (`submitted[s]=1`).
- defect_type: wrong-n-dependence.
- severity: major.
- explanation: Per §0, every `issue_one` coalesces ALL ready slots into one message, and no slot
  re-parks between two consecutive `issue_one` calls without an intervening recv. So PRIME and
  every REFILL issue exactly one message; `inflight_msgs ∈ {0,1}` for ALL N. The model asserts the
  inflight=1 regime "disappears as N grows" and the pipeline "saturates" at depth D once N·base>D.
  The code produces inflight=1 at every N; the model FORBIDS the (universal) inflight=1 execution
  at large N and instead REQUIRES depth-D saturation that is unreachable. This is the exact
  too-constrained pattern: a quantity that is FIXED (=1) is modeled as GROWING toward D with N.
- correction: State that in-flight MESSAGE depth is identically 1 for all N (and all D ≥ 1); D is a
  dead parameter. N grows only the ROW count B inside that single message (ceiling K=N·base,
  DOF-2), not the message depth. The "K≤D vs K>D knee" should be deleted: there is no knee, because
  no execution reaches depth 2.

### P2 [minor, missing-blocking-semantics] "never a silent hang" forbids the timeout_ms ≤ 0 indefinite block
- model_element: timing_model.causal_constraints[5]: *"ZMQ_RCVTIMEO=timeout_ms caps recv blocking;
  exceeding it is a loud FAILED transition, never a silent hang"*; and RELY R4 ("answered within
  timeout_ms=15000 default").
- expected_code_ref: wire_leaf_pool.hpp:41 (`zmq_setsockopt(sock, ZMQ_RCVTIMEO, &timeout_ms, ...)`,
  no validation); runner_wire_batched.hpp:22 (`int timeout_ms = 15000;` — a free, unclamped int in
  `WireRunnerConfig`); passed through at runner_wire_batched.cpp:313.
- defect_type: missing-blocking-semantics.
- severity: minor.
- explanation: `timeout_ms` is an unvalidated config int handed straight to ZMQ_RCVTIMEO. With
  `timeout_ms == -1` (or any negative), ZMQ_RCVTIMEO=-1 means **block forever**; a dead/slow peer
  then causes `zmq_msg_recv` (wire_leaf_pool.hpp:147) to block indefinitely with NO FAILED
  transition — a genuine silent hang. The model categorically forbids this ("never a silent hang"),
  so it forbids an execution the code permits. (With timeout_ms==0, ZMQ_RCVTIMEO=0 means
  non-blocking → immediate EAGAIN → FAILED on the first recv even when a reply is in flight, a
  different code-permitted outcome the "blocks up to the bound" framing also omits.)
- correction: Make the recv-blocking duration depend on the sign of `timeout_ms`: for
  `timeout_ms > 0`, bounded block then EAGAIN→FAILED (as modeled); for `timeout_ms < 0`, unbounded
  block (silent hang admissible under a dead peer); for `timeout_ms == 0`, immediate EAGAIN→FAILED.
  The "never a silent hang" guarantee holds only for the positive-timeout configuration.

### P3 [minor, mis-mapped-transition] PRIMING→PRIMING "issue up to D messages" pins a reachable count the code caps at 1
- model_element: state_machine transition PRIMING→PRIMING, guard "inflight_msgs<D && some
  is_ready(s)" presented as repeatable up to D; and representative_executions[0] step 1: *"with
  D=2, a second issue sends c1 carrying slot1."*
- expected_code_ref: runner_wire_batched.cpp:456 (prime loop), 437-444 (gather-all), 444
  (`if (gathered.empty()) return false`).
- defect_type: mis-mapped-transition.
- severity: minor.
- explanation: This is the PRIMING face of P1. The model's PRIMING self-loop and the step-1 "second
  issue sends c1" describe the prime loop issuing ≥ 2 messages before any recv. Per §0 the prime
  loop's second `issue_one` always returns false (all ready slots already submitted), so PRIMING
  issues exactly one message. The model permits a 2-message prime that the code cannot produce —
  and, paired with P1, mis-states the prime as able to reach depth D. (Direction here is
  over-permissive in isolation; I log it because it is the concrete mechanism by which P1's
  too-constrained large-N claim is reached — the model needs the multi-message prime to justify
  "saturates at D", which then forbids inflight=1.)
- correction: Collapse the PRIMING self-loop to a single firing (one message), then unconditional
  PRIMING→PUMPING. Remove the "(or two messages c0,c1)" alternative from rep-exec[0] step 1; with
  two ready slots at prime the code sends ONE B=2 message.

### P4 [minor, n-dependence asserted not derived] 1/sqrt(N) eligible-set "concentration" assumes cross-slot independence the code does not provide
- model_element: DOF-1 n_dependence: *"relative variance shrinks ~1/sqrt(N) (concentration) ...
  the eligible-set is both larger and steadier"*; rep-exec[0] n_dependence ("more stable as N grows").
- expected_code_ref: fiber_tree.hpp:50 (`policy.run_search`, park timing wholly unfixed by this
  code); runner_wire_batched.cpp:404 (`sl.rng.seed(fold_seed(cfg.seed, idx))`) and 406
  (world drawn from the shared `worlds`), i.e. slots share env and a deterministic seeding scheme.
- defect_type: none (statistical-narrative, not an execution-set restriction).
- severity: minor.
- explanation: The 1/√N concentration is a CLT statement that requires the K park-interval streams
  to be independent (or weakly dependent). The code fixes NOTHING about the park-interval
  distribution and provides no independence guarantee; slots can be strongly correlated (same env,
  correlated search costs), so executions where all K slots park in near-lockstep at large N are
  admissible (the model's own DOF-1 behaviors_admitted lists lockstep). The "steadier as N grows"
  claim, IF read as a property executions must satisfy, would forbid those correlated-lockstep
  large-N executions. Model 1's state machine does NOT actually narrow the admissible set (DOF-1
  keeps lockstep↔fully-staggered), so I assign defect_type none, but I flag the n_dependence text
  as asserted, not derived — it should not be read as constraining executions.
- correction: Demote the 1/√N concentration to an explicitly-non-binding statistical aside; keep
  the admissible eligible-set the full {1..K} at every N (no steadiness constraint).

## 3. Findings — Model 2 (model-producer-transport.md)

### Q1 [major, wrong-n-dependence] in-flight depth modeled as rising toward D with N; code fixes it at 1
- model_element: DOF-T3 ("Effective in-flight depth 1..D"; *"large N → ... D-utilization rises
  toward D"*); representative_executions[1] ("Staggered waves fill D", steady pump "at depth D",
  *"the steady pump at depth D is the large-N attractor"*); n_dependence_summary
  (*"Pipelining-depth utilization (messages actually in flight, 1..D) rises ... toward D as N grows"*).
- expected_code_ref: runner_wire_batched.cpp:456 and 474 (the two issue loops) with 437-444
  (gather-all into one message) and 447 (`submitted[s]=1`).
- defect_type: wrong-n-dependence.
- severity: major.
- explanation: Same root as P1. Because every `issue_one` coalesces ALL ready slots into ONE
  message and nothing re-parks between consecutive `issue_one` calls without a recv (§0), the prime
  and every refill issue exactly one message; `inflight_msgs ∈ {0,1}` at all N. Rep-exec[1]
  ("PRIME issues message 1..#D climbing inflight to D") is unreachable, and the claim that the
  large-N attractor is "steady pump at depth D" FORBIDS the actual universal execution
  (single-message, depth 1) at large N. The model treats a FIXED quantity (depth = 1) as N-growing.
  Its Z3 "confirmation" of full depth D=3 confirms an admissibility the C++ driver cannot realize —
  the encoding modeled independent send/recv events without the gather-all + no-reparking-between-
  issues constraint, so it validated a depth the code forecloses.
- correction: Replace DOF-T3 with: in-flight MESSAGE depth is identically 1 for all N (D ≥ 1 dead);
  the N effect is entirely in DOF-T2 (rows per single message, up to K=N·base). Delete rep-exec[1]
  "Staggered waves fill D" as unreachable, or relabel it explicitly as a counterfactual showing
  what the code forecloses.

### Q2 [minor, missing-blocking-semantics] "never a silent hang" forbids the timeout_ms ≤ 0 indefinite block
- model_element: timing_model.causal_constraints[4] (mirrors Model 1's wording: *"ZMQ_RCVTIMEO=
  timeout_ms caps recv blocking; exceeding it is a loud FAILED transition, never a silent hang"*);
  DOF-T5 ("aborts loudly iff a reply takes longer than timeout_ms"); transition RECV_BLOCK→ERR.
  (Wording of causal_constraints[4] appears in Model 1's payload; the substance is asserted by
  Model 2's DOF-T5 and R5 identically — the RCVTIMEO bound treated as always finite and firing.)
- expected_code_ref: wire_leaf_pool.hpp:41 (unvalidated `ZMQ_RCVTIMEO=timeout_ms`);
  runner_wire_batched.hpp:22 (`int timeout_ms = 15000;`, free int).
- defect_type: missing-blocking-semantics.
- severity: minor.
- explanation: Identical mechanism to P2. `timeout_ms` is a free, unclamped int set directly as
  ZMQ_RCVTIMEO. `timeout_ms = -1` → block forever → silent hang under a dead peer, with no FAILED
  transition; `timeout_ms = 0` → non-blocking recv → immediate EAGAIN→ERR even with a reply
  pending. DOF-T5's "aborts loudly iff a reply takes longer than timeout_ms" and the "only bounded
  wait" framing forbid both of these code-permitted outcomes.
- correction: Make RECV_BLOCK's bound a function of sign(timeout_ms): unbounded (silent-hang
  admissible) for negative, zero-block (instant EAGAIN→ERR) for zero, bounded-then-ERR for positive.

### Q3 [minor, mis-mapped-transition] FILL→PRIME→GATHER chain lets PRIME issue multiple messages before recv
- model_element: state_machine transitions POST_SEND→PRIME ("inflight_msgs<D (still priming)") and
  PRIME→GATHER repeated; representative_executions[1] step 2 ("issue_one #2..#D ... inflight_msgs
  climbs to D").
- expected_code_ref: runner_wire_batched.cpp:456 (prime loop), 444 (`gathered.empty() → false`),
  447 (`submitted[s]=1`).
- defect_type: mis-mapped-transition.
- severity: minor.
- explanation: The PRIME face of Q1. The POST_SEND→PRIME→GATHER cycle is drawn as repeatable up to
  D before any RECV_BLOCK; the code's second prime `issue_one` always returns false (all ready
  slots submitted, nothing re-parked without a recv), so PRIME issues exactly one message and
  transitions to RECV_BLOCK. The multi-message prime is the mechanism the model uses to reach depth
  D, which then forbids the universal depth-1 reality.
- correction: After the first SEND/POST_SEND in PRIME, the only enabled transition is
  POST_SEND→RECV_BLOCK (a second PRIME→GATHER yields GATHER→PRIME via `gathered.empty()` and then
  proceeds to recv). Collapse the prime to a single message.

### Q4 [minor, unfaithful-rely] R4 "exactly one reply message per request" — faithful for the producer, but the model should not also assert depth-D draining that depends on it
- model_element: RELY R4 ("Exactly one reply message per request ... the DEALER pipe is FIFO");
  fidelity_self_audit notes DEALER↔ROUTER FIFO already.
- expected_code_ref: wire_leaf_pool.hpp:106-132 (one corr → one Completion-vector per recv_batch);
  runner_wire_batched.cpp:457-460 (one recv per loop iteration, `--inflight_msgs` once).
- defect_type: none.
- severity: minor.
- explanation: R4 itself is faithful (one reply per request). I flag only that the model pairs a
  correct "one reply per outstanding message" with the unreachable multi-outstanding-message
  pipeline (Q1); the producer DOES tolerate out-of-order replies across multiple outstanding
  messages (DOF-T6) — but since inflight ≤ 1, at most one message is ever outstanding, so the
  out-of-order reorder space is actually EMPTY in this code. The model's DOF-T6 ("any permutation
  of outstanding-reply arrivals") is non-vacuous only at depth ≥ 2, which is unreachable. This is
  over-permissive (admits reorderings that cannot occur), recorded for completeness; it is not a
  too-constrained finding, hence defect_type none.
- correction: Note that because inflight ≤ 1, there is never more than one outstanding message, so
  cross-message reply reordering (DOF-T6) is unreachable; the only "ordering" freedom left is the
  WITHIN-message completion order, which is fixed (ascending gathered-slot order, wire_leaf_pool.hpp
  :125-130) — i.e. no freedom at all on the producer's reply-consumption order.

## 4. Trace checks

Model 1's representative traces: rep-exec[0] is admissible EXCEPT step 1's "(with D=2, a second
issue sends c1)" alternative, which the prime loop cannot produce (single-coalesced gather; §0). The
primary "one B=2 message at D=1" reading is admissible. rep-exec[1] (inflight=1) is admissible but
mislabeled as a vanishing small-N corner (P1). rep-exec[2] (RCVTIMEO) is admissible for
timeout_ms>0; admissibility is config-dependent (P2). Net: traces are admissible as concrete event
sequences, but the depth-D / "second prime message" annotations describe unreachable steps.

Model 2's rep-exec[0] (all-ready single message, inflight=1) is admissible and is in fact the
UNIVERSAL behavior. rep-exec[1] ("Staggered waves fill D", inflight climbs to D) is NOT admissible:
the code never reaches inflight ≥ 2 (§0). The Z3 run that "confirmed depth D=3 fully drained"
confirmed an abstraction looser than the code (independent send/recv without the gather-all +
no-reparking-between-issues constraints), so it does not witness a code-reachable execution.

## 5. Timing-fidelity judgment

Source park-interval: both models keep `delta_park > 0` as positive bounded nondeterminism
(fiber_tree.hpp:50) — faithful, NOT collapsed to a constant. Good. The only over-reach is the 1/√N
"steadier" narrative (P4), which is asserted, not derived, and (because the admissible eligible-set
stays {1..K}) does not actually narrow executions.

Sink service: both keep `S_fwd > 0` with code-derived STRUCTURE — padmax ≈ row-count-invariant
(fixed pad to max_batch, inference_server.py:198,58-59; forward.py 2-3-layer MLP) and bucket as a
3-step function over {64,256,512} (stage_a_server.py:32-37,61-64). Faithful; not collapsed to an
instant. The leaf-vs-group wakeup effect (stage_a_server.py:57) is correctly attributed to
cross-producer forward count, not within-message splitting (a producer message = one ROUTER
identity = one drained request = one forward even under leaf wakeup). No too-constrained timing
defect on the service side.

The one missing-blocking-semantics timing defect is P2/Q2: the recv block bound is modeled as always
finite-and-firing, whereas `timeout_ms ≤ 0` makes it unbounded (silent hang) or zero (instant
EAGAIN). That is a dropped blocking outcome, the only timing collapse in the too-constrained
direction.

## 6. N-dependence check

The producer's sole N entry is K = N·base (runner_wire_batched.cpp:286). Both models correctly
derive that the per-message ROW count B scales linearly with N (ceiling K=N·base; gather-all at
437-444). That part is faithfully derived from the code.

Both models, however, ASSERT a second N-dependence — in-flight MESSAGE depth / D-utilization rising
toward D with N — that is NOT derivable from the code and is in fact CONTRADICTED by it: inflight ∈
{0,1} for all N (§0). So the claimed N-dependence is mixed: the row-count growth is derived and
correct; the pipeline-depth growth is asserted and wrong (P1/Q1), and forbids the universal depth-1
executions at large N. The "tail ramp-down lasts ~N·base episodes" (Model 1 DOF-6) is correctly
derived from K=N·base slots draining against fixed total episodes.

## 7. Fidelity verdicts

- Model 1 (model-producer-pacing.md): **mixed** — faithful on the protocol shape, corr-matching,
  RCVTIMEO-as-fail (for positive timeout), and the B=N·base row-count growth; too-constrained on
  the large-N pipeline-depth claim (P1: forbids universal inflight=1) and on the categorical
  "never a silent hang" (P2: forbids timeout_ms≤0). P3/P4 are corollary/narrative.
- Model 2 (model-producer-transport.md): **mixed** — same shape; too-constrained on the
  depth-toward-D large-N attractor (Q1: forbids universal inflight=1) and "never a silent hang"
  (Q2). Q3 corollary; Q4 records the (over-permissive) empty-reorder-space dual.

Both verdicts are mixed rather than too-constrained-overall because each model's dominant deviation
from the code is actually OVER-permissive (admitting the unreachable depth-2..D pipeline); the
too-constrained findings (P1/Q1, P2/Q2) are the consequence of that same depth mis-model forbidding
the real universal depth-1 large-N executions, plus the dropped negative-timeout block.
