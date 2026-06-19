# Adversarial fidelity verification — SERVER side, lens = TOO-CONSTRAINED

Role: verifier for the Python single-threaded ROUTER inference server (production greedy drain
`chocofarm/az/inference_server.py` + bench bucketed/group/leaf drain `cpp/stage_a/stage_a_server.py`).

Lens: **too-constrained** — find any execution the code CAN produce that a model FORBIDS (forced
orderings, pinned counts, lockstep, dropped blocking/timeout outcomes, timing collapsed to a constant,
or an N-dependence that understates what grows). Every finding is mapped to a cleanroom `file:line`.

All paths below are under
`/home/bork/w/vdc/chocobo/runs/leaf-eval-model-2/cleanroom/`.

## Independent read of the side (end to end)

I read end to end: `chocofarm/az/inference_server.py` (1-239), `cpp/stage_a/stage_a_server.py` (1-131),
`chocofarm/az/forward.py` (1-19), `chocofarm/az/inference_wire.py` (1-107), `chocofarm/config.py` (1-44),
`cpp/include/chocofarm/wire_leaf_pool.hpp` (1-173), `cpp/src/runner_wire_batched.cpp` (1-506),
`cpp/include/chocofarm/runtime_config.hpp`, `wire_spec.hpp`, `inference_wire.hpp`, `fiber_leaf.hpp`,
`fiber_tree.hpp`, `net_evaluator.hpp`, `runner_wire_batched.hpp`, `error.hpp`.

Ground facts established by reading (these gate every too-constrained judgment):

- **Server spine is strictly serial.** `serve_forever` (inference_server.py:219-225) does
  `_drain` → (if drained) `_serve_batch`; one server thread; no concurrency. This is the coalescing
  engine and it is faithfully captured by all three models. Not a too-constrained surface (a single
  thread genuinely cannot overlap forwards).
- **Server sets NO socket options.** `grep` confirms the only `setsockopt` calls are on the *producer*
  (`wire_leaf_pool.hpp:40-41`: LINGER=0, RCVTIMEO=timeout_ms). The server binds ROUTER
  (inference_server.py:153-154) and only `close(linger=0)` at shutdown (:236). Hence SNDHWM=RCVHWM=1000
  default, ROUTER_MANDATORY=0 default, recv always NOBLOCK (:173), send blocking but cannot HWM-block
  under MANDATORY-off. All three models state this correctly.
- **Drain membership/order.** `_drain` (:171-185) pulls `recv_multipart(NOBLOCK)` until `zmq.Again` or
  `total_rows >= max_batch` checked at loop TOP. The pull order is ZeroMQ ROUTER **fair-queuing across
  peers**, not a single global FIFO. The drain takes *everything currently queued up to the row cap*, so
  the drained SET is "all queued at the snapshot up to cap"; the only order-sensitive freedom is which
  message straddles the cap (the over-fill split).
- **`pad_to` semantics.** `run_microbatch` pads only when `pad_to > B` (:58). Production always
  `pad_to=max_batch` (:198); over-fill `B>max_batch` ⇒ no pad ⇒ fed shape = `B` (unwarmed). Bench
  `_bucket_for` snaps to {64,256,512} clamped 512 (stage_a:32-37); `real>512` ⇒ no pad ⇒ width=real.
- **Peer pacing (RELY ground).** `run_episodes_wire_pipelined` (the N>1 path) gathers ALL ready slots
  into ONE message (`issue_one`, runner_wire_batched.cpp:437-451), `K = N·ceil(pool_batch/T)`
  (:286, runtime_config.hpp:12-15), `recv_batch` per loop (:458), refill while `inflight_msgs<D`
  (:456,474). **Decisive sub-fact (Z3-confirmed UNSAT for inflight==2, `inflight_le1_check.py`):**
  because `issue_one` gathers *all* ready slots and readiness is only created inside a `recv_batch`
  completion loop, after any single issue there is no second ready slot to form a concurrent message, so
  **`inflight_msgs ∈ {0,1}` per thread** — never climbs to D. The real per-peer wire depth is ≤ 1, hence
  the cross-peer coalescing ceiling is **T**, not T·D.

This last fact is the lens-relevant pivot, and it cuts the *opposite* way to my lens for the server
models: every server model bounds queue depth / coalescing by **`≤ T·D`** (Model 1 DOF-1; Model 3 DOF-2
"g saturates at ~T*D"). Since the true bound is T ≤ T·D, `≤T·D` is *over-permissive*, not
too-constrained. So the dominant inflight≤1 finding belongs to the too-PERMISSIVE verifier; under my
lens it produces **no** finding (the models do not forbid any reachable depth — they admit more than is
reachable). I record this explicitly so the gap is not silently papered over.

## What I looked for and ruled OUT as too-constrained (with code refs)

These are candidate forced-orderings/pins I checked and found faithful (NOT defects), so they are not
charged as findings:

1. **Forwards never overlap** (all models). Faithful: one server thread, serve_forever:222-225.
2. **Production S independent of real B** (Model 1 sink_service; Model 3 Exec B). Faithful: `pad_to=
   max_batch` (:198) + pad-when-`pad_to>B` (:58) make the fed shape constant for `B≤max_batch`; the
   model does NOT forbid the B-dependent over-fill case (Model 1 DOF-4, Model 2 G6, Model 3 DOF-7 all
   keep `B>max_batch` ⇒ width=B). Not collapsed.
3. **Bench leaf = one forward per drained MESSAGE (not per row).** Model 2 DOF-2 states this precisely
   (stage_a:57 `[[d] for d in drained]`, each `d` may carry `B_msg>1`). Models 1 and 3 phrase it as "per
   request" but their service-time math sums per-message rows, so they do not forbid fat leaf forwards.
   Not too-constrained.
4. **Bench has no weight reload** (stage_a:56 `current()`; StaticParamsSource.poll→None, :110-111). All
   models forbid a bench reload transition — correct, it is genuinely impossible there (faithful
   restriction, not over-constraint).
5. **Over-fill past max_batch admitted** (Model 1 DOF-4; Model 2 §5.4/Exec; Model 3 DOF-7). All keep
   `B∈[1,max_batch+K-1]` and width=B unpadded — Z3-confirmed admissible, and a `width≤M` clamp is UNSAT
   (`z3_bucket_drain_check.py`). None of the models impose the clamp, so none is too-constrained here.
6. **100ms poll is not added latency** (all models): poll wakes early on POLLIN (:165). Modeling it as a
   liveness/stop bound rather than per-request latency is faithful; treating it as added latency would
   be the over-constraint, and no model does that.

## FINDINGS (too-constrained), per model

### Model A — `model-server-greedy-drain.md`

**Verdict: faithful** (with one minor mischaracterization that does not narrow the admitted set).

- A1 (minor, mis-mapped wording, not set-narrowing). The Z3 witness encodes drain membership as
  `in_c0[i] == (arr[i] <= c0_wake)` over a single global timeline (`check_server_greedy_drain.py:77`).
  ZeroMQ ROUTER delivers across peers by **fair-queuing**, not global arrival-time order
  (inference_server.py:173). For pure *membership* this is harmless (the drain takes all queued up to
  cap), and the witness caps both cycles below `max_batch` so no over-fill split is exercised; the model
  separately leaves the over-fill split free in prose (DOF-4: "depends on … arrival order",
  inference_server.py:171). So no reachable drain SET is forbidden. Charged as a minor wording defect,
  not a set-narrowing one — corrected below for precision.

No fatal/major too-constrained defect. The model's bounds (`≤T*K` coalescing, `B∈[1,max_batch+K-1]`,
service-band-not-duration) are at-or-wider-than reachable.

### Model B — `model-server-bucket-drain.md`

**Verdict: faithful** (one minor too-constrained mischaracterization).

- B1 (minor, forbids-too-much in wording). Audit OC-1 asserts the drain "pulls strict **FIFO socket
  order** and stops at the FIRST zmq.Again; it cannot skip a queued message"
  (inference_server.py:171-185). "Strict FIFO socket order" is inaccurate: a ROUTER multiplexes T DEALER
  peers with **fair-queuing**, so the pull sequence is an interleaving across peers, not a single FIFO.
  The drained SET (all queued ≤ cap) is unaffected, but the *pull order* — which determines which message
  straddles the `total_rows>=max_batch` cut (:171) and thus which one becomes the over-fill / which
  messages are deferred to the next cycle — has cross-peer interleaving freedom that "strict FIFO" reads
  out. Because the model elsewhere (DOF-1 "over-fill split point depends on … arrival order") leaves the
  split free, this is a localized wording over-constraint, not a global set-narrowing. Minor.

The "cannot skip a queued message" half is correct and faithful (the NOBLOCK loop does pull contiguously
until Again). Over-M overshoot, bucket clamp, leaf-per-message, no-reload-in-bench are all faithful.

### Model C — `model-server-transport.md`

**Verdict: faithful** (no too-constrained defect of consequence).

- C1 (minor, illustrative narration, not a guard). Exec D step 1 narrates bench-leaf drained rows as
  "each B_i small/often 1" (stage_a_server.py:57; runner_wire_batched.cpp:437-451). Taken as a *guard*
  this would forbid the fat-per-message leaf forwards the code produces once `K=N·base` is large; but it
  is an example, and the same Exec's `n_dependence` explicitly states each `b_j` grows to K with N, and
  DOF-5 states leaf is "one forward per drained request" without pinning row counts. So the admitted set
  is not narrowed. Charged minor for the potentially-misleading example only.

Model C is the most careful of the three on the inflight coupling: its causal constraint "message m+1
for a submitted slot cannot precede … resume_with" and the D-cap (runner_wire_batched.cpp:466-471) are
stated, and `check_server_selfbatch.py` confirms the self-batching execution with one message per peer
(consistent with inflight≤1). It does not forbid any reachable server input.

## Cross-model note

The single dominant fidelity issue on this boundary — that per-thread `inflight_msgs ∈ {0,1}`
(Z3-UNSAT for ==2, `inflight_le1_check.py`; grounded at runner_wire_batched.cpp:437-451,456,474) so the
true cross-peer coalescing ceiling is **T**, not **T·D** — makes all three server models *over-permissive*
on their depth bounds (Model 1 DOF-1 "≤T*D"; Model 3 DOF-2 "g saturates at ~T*D"), which is the
too-PERMISSIVE verifier's charge, not mine. Under the **too-constrained** lens the three server models
are essentially clean: their quantitative bounds are at-or-wider-than reachable, their timing is kept as
bounded nondeterminism (no service time collapsed to a constant; the only constants are the structural
`pad_to=max_batch` shape and the literal 100ms poll, both faithful), and the few defects are localized
mischaracterizations of ZeroMQ ROUTER pull *ordering* ("strict FIFO" / global-arrival-order) that do not
narrow the admitted drained SET because the drain takes everything queued up to the row cap. Net: no
fatal or major too-constrained findings; three minor wording-level over-constraints, corrected above.

## Validation

I re-ran the three referenced server Z3 checks under `nice -n 19 timeout 90`
(z3 4.16, `/home/bork/w/vdc/venvs/generic/bin/python`):
- `check_server_greedy_drain.py` → sat (cycle-0 B=1, cycle-1 coalesced B=45, both padded to 256):
  confirms the central coalescing latitude is admissible, i.e. NOT forbidden — no too-constrained defect
  there.
- `z3_bucket_drain_check.py` → faithful sat (K=600,width=600,pad=0), `width≤M` clamp unsat: confirms a
  too-constrained clamp would be wrong, and no model imposes it.
- `inflight_le1_check.py` → unsat for inflight==2: confirms inflight≤1, the pivot that makes the models'
  `≤T·D` bounds over-permissive (not too-constrained).

These are confirmation only; the rigorous reading above is the source of trust.
