# Adversarial fidelity audit — C++ producer side, TOO-CONSTRAINED lens

Verifier role: find any execution the real code CAN produce that a model FORBIDS.
Two producer-side models audited: `model-producer-pacing.md` (Model 1) and
`model-producer-transport.md` (Model 2).

All cited code read end to end (ADR-0002):
- `cpp/include/chocofarm/wire_leaf_pool.hpp` (1-243)
- `cpp/src/runner_wire_batched.cpp` (1-630)
- `cpp/include/chocofarm/inference_wire.hpp` (1-226)
- `cpp/include/chocofarm/wire_spec.hpp` (1-58)
- `cpp/include/chocofarm/fiber_tree.hpp` (1-111)
- `cpp/include/chocofarm/fiber_leaf.hpp` (1-57)
- `cpp/include/chocofarm/runtime_config.hpp` (1-46)
- `cpp/include/chocofarm/runner_wire_batched.hpp` (1-100)
- `chocofarm/az/inference_server.py` (1-457)
- `chocofarm/az/inference_wire.py` (1-185)
- `chocofarm/az/forward.py` (1-63)
- `docs/design/zmq-inference-service.md` (1-367)

Independently verified the single most fidelity-critical fact by grep over the WHOLE
cpp tree: the producer DEALER (`wire_leaf_pool.hpp:82-83`) sets ONLY `ZMQ_LINGER=0` and
`ZMQ_RCVTIMEO=timeout_ms`. No `SNDHWM`/`RCVHWM`/`SNDTIMEO`/context option exists anywhere
in the producer path (`runner_wire_batched.cpp` sets none; the only other `zmq_setsockopt`
sites are `zmq_net_client.cpp`, `wire_parallel_bench.cpp`, `dealer_probe.cpp` — not this
path). Server ROUTER (`inference_server.py:315-318`) sets NO socket option at all → ROUTER
SNDHWM/RCVHWM at libzmq default 1000.

---

## Summary verdicts

- **Model 1 (`model-producer-pacing.md`): FAITHFUL.** No execution the code admits is
  forbidden. The timing is bounded-nondeterministic in both directions; the one collapse
  (σ independent of B) is the correct, code-justified direction. One MINOR over-constraint
  candidate examined and cleared. Two MINOR notes recorded as latent (not defects).
- **Model 2 (`model-producer-transport.md`): FAITHFUL** with one MINOR genuine
  over-constraint in a single transition guard (the `SCATTERING → IDLE` collapse hides the
  `SCATTERING → PARKED-fresh-episode` transition the code produces) and one MINOR
  over-constraint in the warmup-cost narrative. Neither removes a real schedule from the
  representable set in a load-bearing way; both are precision defects in the state-machine
  labeling, not in the admitted-execution set.

The too-constrained lens found NO fatal or major hole in either model. Both leave the
timing latitude the code leaves and pin only what the code pins. The findings below are
the precise, code-anchored places where a constraint MIGHT have narrowed the behavior set,
each adjudicated.

---

## Adjudicated constraint-by-constraint audit

### C-1. Source think-time δ — collapsed to a constant? (the highest-stakes too-constrained risk)

**Code:** `fiber_tree.hpp:99,106-107` (`running = ch.at_leaf` after a fiber resume);
`fiber_leaf.hpp:46-48` (`predict` parks + yields); no interval is fixed anywhere in the
driver (`runner_wire_batched.cpp` reacts only to `sl.ts->running`, never to a clock).

**Both models:** δ is a positive, otherwise-unconstrained nondeterministic interval
(Model 1 DOF-2; Model 2 DOF-1, `RUNNING_FIBER` state). NEITHER pinned it. **Cleared — not
a defect.** Collapsing δ would pin the coalescing degree S and forbid the staggered-arrival
executions (the entire point of `is_ready` snapshotting at the gather instant,
`runner_wire_batched.cpp:554-560`). Both models correctly leave it free.

### C-2. Sink service-time σ — collapsed to a constant or made batch-dependent?

**Code:** `inference_server.py:171-172` (`pad_to=max_batch` pads every batch UP to one
fixed `(max_batch,in_dim)` shape) + `:385` (`_serve_batch` passes `pad_to=self._max_batch`)
+ `:95-115` (`jit_forward_core` compiles ONE executable). So once warmed, the matmul shape
is constant in B.

**Both models:** σ is a positive bounded nondeterministic interval, made INDEPENDENT of
instantaneous B (the padding consequence), but NOT pinned to a numeric constant and NOT
forced to grow with B. **Cleared — and this is the CORRECT direction for the too-constrained
lens.** The over-constraint here would be to make σ grow with B (that would forbid the real
constant-shape forward). Both models avoid it. Model 1 additionally allows a cold-JIT spike
(`:389-426` warmup docstring) as a σ_max allowance; Model 2 folds the same into "constant in
B, not constant across calls." Both faithful.

### C-3. Reply arrival order across the D outstanding messages — forced FIFO?

**Code:** `wire_leaf_pool.hpp:170-196` (`recv_batch` decodes whatever arrives and routes by
`inflight_.find(corr)`, NOT by submit position); `runner_wire_batched.cpp:584` (drain
processes that corr-id's slot list). DEALER fair-queues; the server's `_drain`
(`inference_server.py:348-363`) + `_serve_batch` reply loop (`:384-387`) reply in drain
order, which is NOT the producer's submit order.

**Both models:** any permutation admitted (Model 1 DOF-3 / E2; Model 2 DOF-4 / E3).
**Cleared — not a defect.** A FIFO assumption would forbid E2/E3, which the code admits.
Both avoid it. (Confirmed SAT both ways by the bounded Z3 check, see Validation.)

### C-4. Pipeline depth D and the strict-barrier specialization — over-collapsed?

**Code:** `runner_wire_batched.cpp:392` (`D = max(1, wcfg.max_inflight_msgs)`); the strict
driver is a SEPARATE function body (`:60-354`) dispatched at `:66-67`, structurally D=1.

**Model 1:** treats the strict barrier as the D=1 specialization of the pipelined driver
(DOF-4). **This is a fidelity SIMPLIFICATION worth scrutiny, since it could forbid a real
execution.** Adjudication: the strict-barrier body gathers ALL parked into one message,
blocks, resumes all (`:310-337`) — it issues exactly ONE message, blocks for its reply, never
holds 2. That IS the D=1 behavior of the pipelined loop's `while(inflight<D)` (`:578,596`).
The two bodies produce the SAME representable execution set at D=1 (same `spawn/advance/fill`
lambdas, asserted line-for-line `:441-443`). So Model 1's collapse does NOT forbid any
strict-barrier execution. **Cleared.** Model 2 models BOTH bodies explicitly (E4 for the
strict barrier, E1-E3 for the pipeline) — strictly more faithful at the structural-labeling
level, but the admitted-execution set is identical. See cross-model note.

### C-5. Coalescing degree S — pinned below 1..K?

**Code:** `runner_wire_batched.cpp:554-561` (`issue_one` gathers EVERY ready slot,
`gathered.size()` ranges 1..#ready ≤ K; empty-guard at `:561` excludes S=0).

**Both models:** S ∈ [1,K] free (Model 1 DOF-1; Model 2 DOF-2). **Cleared.** Neither pins
S to a constant nor to a function of the round. The strict barrier's S=#all-parked-each-round
is correctly noted as the D=1 consequence, not a separate pinning.

### C-6. The send-block latitude (SNDHWM=1000, SNDTIMEO=-1) — dropped?

**Code:** `wire_leaf_pool.hpp:139-144` (`zmq_send` twice, no SNDHWM/SNDTIMEO set anywhere →
default SNDHWM=1000, SNDTIMEO=-1, so a send blocks indefinitely once the DEALER send queue
hits 1000).

**Model 1:** lists this as a fidelity-audit boundary — "If SNDHWM WERE reached (D·T≥1000)
the send would block (SNDTIMEO=-1), but that is unreachable under the code's geometry —
noted as a boundary, not admitted." It does NOT carry a `BLOCKED_SEND` state.
**Model 2:** carries an explicit `BLOCKED_SEND` state + the `REFILLING → BLOCKED_SEND`
transition (DOF-6), representing the indefinite send-block.

Adjudication of the TOO-CONSTRAINED question for **Model 1**: does omitting `BLOCKED_SEND`
forbid a real execution? Under the code's geometry, D ≤ `max_inflight_msgs` (default 8) per
thread and a thread issues at most D messages before it must `recv` (`runner_wire_batched.cpp
:578-580,596-597` — the loop recvs whenever `inflight>0` and refills only up to D). So per
thread at most D ≤ 8 messages are ever unacknowledged-and-enqueued; the DEALER send queue
cannot reach 1000 by the resume-gated cap. The execution where `zmq_send` blocks is therefore
**genuinely unreachable under the code's own control flow** (it would require the producer to
enqueue ≥1000 messages without recving, which the `inflight<D` gate forbids). So Model 1's
omission does NOT forbid a real execution — the execution does not exist in this code.
**Cleared as a legitimate non-modeling of an unreachable corner, NOT an over-constraint.**
Model 2's inclusion is also faithful (it represents the option-state honestly and marks it
practically unreachable). Both are defensible; this is a matter of where to draw the
reachability line. Recorded as the one place the two models DISAGREE (cross-model note).

### C-7. Within-thread fiber serialization — over-constrained to lockstep?

**Code:** boost.context is cooperative (`fiber_tree.hpp:90-100,103-107` — one fiber runs
between `resume()` calls); the drain processes Completions one at a time
(`runner_wire_batched.cpp:584-594`).

**Both models:** at most one fiber of a thread runs at a wall-clock instant; across T OS
threads true parallelism is admitted and left code-unbounded (Model 1 causal_constraints;
Model 2 RUNNING_FIBER + DOF-7). **Cleared.** Neither bounds cross-thread parallelism to the
4-vCPU host (correctly — the code spawns T independent `std::thread`s, `:342/:604`, with no
such cap; the host wall is an operational fact, not a code constraint). This is the RIGHT
call for the too-constrained lens: bounding T-parallelism to 4 would forbid the interleavings
the code's threading admits.

### C-8. Out-of-order reply WITHIN the same server forward — forbidden?

**Code:** the server can drain SEVERAL of one producer's outstanding messages into ONE
forward (`inference_server.py:348-363` — `_drain` pulls all currently-queued frames up to
`max_batch` rows, with NO per-identity limit), then replies to all of them in one
`_serve_batch` loop (`:384-387`). So two of a single producer's D messages can have their
replies released at the SAME forward-completion instant, back-to-back on the wire.

**Both models:** neither REQUIRES separate forwards per message, and the producer recvs ONE
corr-id reply per `recv_batch` call regardless of how they were batched server-side
(`wire_leaf_pool.hpp:170-173`). So the same-forward co-batch of one producer's own messages
is admitted by both (they just arrive together and are recv'd one at a time). **Cleared — not
forbidden by either.** This is the subtlest too-constrained trap and both models pass it
because they model the producer's recv as corr-id-keyed, order-agnostic, one-reply-per-call.

### C-9. The first leaf of a fresh episode is NOT reply-gated — forbidden by a blanket reply-dependence?

**Code:** `fill → spawn_ply` (`runner_wire_batched.cpp:516-532,278/532`) parks the FIRST leaf
of a fresh episode WITHOUT any prior reply (it runs off the slot's rng + world draw, no recv).
Only the (r+1)-th leaf of an ONGOING episode is reply-gated (it is computed inside
`resume_with(c.pred)`, `:589`).

**Model 1:** its reply-dependence constraint is scoped "per-slot reply-dependence — the
(r+1)-th park cannot precede the r-th reply" — correctly the (r+1)-th, leaving the FIRST park
ungated. **Cleared.**
**Model 2:** explicitly states "The FIRST leaf of a fresh episode (fill->spawn_ply,
:516-532) is NOT reply-gated" in its `source_emission`. **Cleared — both correctly exempt the
first leaf.** A blanket "every park follows a reply" would forbid the priming phase
(`:572,578` issue messages before ANY reply exists) — neither model commits that error.

### C-10. Telemetry / the wire_summary line — does either model forbid the trailing-line execution?

**Code:** `runner_wire_batched.cpp:618-625` emits a trailing `wire_summary` JSON line when
`stats_out` is present, after the drain. Neither model needs to represent this at the
transport level (it is post-drain bookkeeping). Not a transport behavior; no constraint
either way. **N/A.**

---

## Model 1 (`model-producer-pacing.md`) — findings

**Findings: none rising to a defect.** Two latent notes recorded for completeness (defect_type
`none`):

- **N1 (IDLE terminality, latent-correct).** `IDLE` is described as "Terminal for that slot
  (a slot that exhausted its subset during priming is never re-filled)." Verified against the
  code: a slot that returns false from `fill` during priming (`:572`, `next_idx≥episodes`) has
  `active=false`, `submitted=0`, so `is_ready` is false (`:543`), it is never gathered, never
  recv'd, never reaches the `fill(s)` at `:593`. The "terminal" claim is CORRECT, not an
  over-constraint. Listed only to confirm it was checked. Severity minor / defect none.

- **N2 (D=1 strict-barrier collapse, cleared).** See C-4. The collapse of the strict barrier
  to "the D=1 specialization" does not forbid any strict-barrier execution because the two
  driver bodies share the per-slot lambdas line-for-line (`:441-443`) and the strict body
  issues exactly one message per round (`:321-323`). The admitted-execution sets coincide at
  D=1. Severity minor / defect none.

**Trace check (Model 1 representative executions E1-E5):** all five are genuine schedules the
code admits, each step enabled by a real transition:
- E1 (S=K barrier, T=1,K=4,D=2): step 2 "second issue_one finds nothing ready → false" is
  exactly `:561` empty-guard after all 4 became `submitted`; D>1 buying nothing is correct
  (`:596` finds nothing to refill). Admissible.
- E2 (staggered, out-of-order c1<c0, D=3): every step maps to `:551-568` (issue), `:580-590`
  (recv+resume), `:596` (refill). The out-of-order recv is `wire_leaf_pool.hpp:179` corr-id
  routing. Admissible (Z3-confirmed).
- E3 (RCVTIMEO abort): `:580-581` set_error/break; `wire_leaf_pool.hpp:217-221` EAGAIN;
  `:609-614` whole-pass unexpected. Admissible.
- E4 (non-parking chain): `fill:527-535`, `advance:502-510` drain a degenerate chain.
  Admissible.
- E5 (mixed-S steady state, T=4,K=8,D=8): the design's intended operating point;
  `issue_one` mixed-S + server cross-thread `_drain` coalescing. Admissible.

**Timing fidelity (Model 1):** faithful. Source δ and sink σ are both bounded-nondeterministic
positive intervals; the ONLY collapse (σ ⊥ B) is the code-justified faithful direction
(`inference_server.py:171-172`), and Model 1 even allows the cold-JIT σ_max spike
(`:389-400`). Neither timing is pinned to a constant or an instant.

**Fidelity verdict (Model 1): FAITHFUL.**

---

## Model 2 (`model-producer-transport.md`) — findings

**Finding M2-1 (MINOR over-constraint — a real transition collapsed/mislabeled).**
- model_element: the `SCATTERING → IDLE` transition (guard "ts->running false (Decision) and
  advance finalizes (TERMINATE/horizon/empty belief); action: finalize_and_write then fill
  (next episode or idle)").
- expected_code_ref: `runner_wire_batched.cpp:591-593` — after `advance(s)` returns false, the
  code calls `fill(s)` (`:593`), which, if `next_idx < episodes`, starts the NEXT episode and
  parks its first leaf → the slot becomes PARKED with a FRESH episode, NOT idle.
- defect_type: mis-mapped-transition.
- severity: minor.
- explanation: Model 2 folds two distinct post-finalize outcomes into one `→ IDLE` edge. The
  code's `fill(s)` at `:593` has TWO results: (a) `next_idx < episodes` → a new episode parks
  → the slot lands in PARKED (its `→ PARKED-fresh` is the common case in steady state), or
  (b) `next_idx ≥ episodes` → IDLE. By labeling the edge only `→ IDLE`, Model 2's state
  graph FORBIDS the `SCATTERING → PARKED (fresh episode)` transition that the code produces on
  every episode boundary that is not the thread's last. Model 1 handles this correctly with a
  distinct `FINALIZED → PARKED` edge (its `:593 fill(s)` transition). NOTE: Model 2's PROSE
  guard says "next episode or idle," so the author KNEW both outcomes — the defect is the
  state-graph edge target, which names only IDLE. The admitted-EXECUTION set is not actually
  reduced (the `fill→spawn_ply→park` path is reachable via the IDLE→PARKED edge the model also
  has), so this is a labeling/altitude defect, not a removed schedule — hence minor, not major.
- correction: split the edge into `SCATTERING → PARKED` (guard `next_idx < episodes`: fill
  parks a fresh episode's first leaf, `:593→:511-533`) and `SCATTERING → IDLE` (guard
  `next_idx ≥ episodes`: fill returns false, `:537`), mirroring Model 1's `FINALIZED→PARKED`
  / `FINALIZED→IDLE` split.

**Finding M2-2 (MINOR over-constraint in the σ narrative — superseded-comment over-read).**
- model_element: the `sink_service` claim that `warmup` "removes the cold-compile confound"
  and the framing that under current code σ has no cold spike.
- expected_code_ref: `inference_server.py:389-426` (`warmup` is OPTIONAL — it is a public
  method, NOT called inside `serve_forever` `:428-439`); `:171-172` pad-to-one-shape.
- defect_type: timing-collapsed.
- severity: minor.
- explanation: Model 2's σ interval is "constant in B, not constant across calls (XLA/OS
  jitter)" — faithful. But it leans on warmup having "removed the cold-compile confound,"
  whereas `warmup` is not invoked by `serve_forever`; whether a cold spike occurs on the FIRST
  real forward depends on the caller having run `warmup` (an out-of-band choice the producer
  cannot observe). Because padding fixes ONE shape (`:171-172`), there is at most ONE cold
  compile total (not per-B), so the residual σ_max-spike-on-first-forward is small — but it
  EXISTS when warmup was not called. Model 1's σ_max "one-time cold-JIT spike on the first few
  distinct shapes" is the more faithful allowance here; Model 2 slightly under-allows σ_max by
  treating the cold spike as removed. This narrows σ's upper tail below what an un-warmed
  server can produce. Minor because RCVTIMEO=15000ms (`runner_wire_batched.hpp:69`) bounds it
  either way and the producer only observes "reply within RCVTIMEO."
- correction: state σ_max includes a one-time cold-XLA-compile allowance on the first forward
  of the (single, padded) shape WHEN warmup was not pre-run, since `serve_forever` does not
  call `warmup` itself.

**Trace check (Model 2 representative executions E1-E6):** all six are genuine schedules the
code admits.
- E1 (steady D=8 drain): `:572,578` prime; `:580` recv; `:582,588-590` scatter; `:596`
  refill. Admissible.
- E2 (source-bound underflow, pipe below D): `:561` `gathered.empty()→false` at refill when
  the source is slow; `:596` leaves `inflight` below D. Admissible — exactly the latitude
  DOF-1/DOF-3 leave.
- E3 (out-of-order c2 then c1): `wire_leaf_pool.hpp:179-196` corr-id routing. Admissible.
- E4 (strict-barrier round): maps to the SEPARATE strict body `:310-337`. The step refs cite
  `:310-336` correctly (NB: one trace step cites `:326-336` for scatter — the strict body's
  scatter loop IS `:326-336`; consistent). Admissible.
- E5 (RCVTIMEO loud abort): `wire_leaf_pool.hpp:65-70,217-220` lazy connect + EAGAIN;
  `:581,609-614`. Admissible.
- E6 (desync: unknown corr-id / count mismatch): `wire_leaf_pool.hpp:180-188`. Admissible
  (unreachable against a correct server, correctly flagged as the fail-loud net).
The M2-1 mislabeling does not break any traced execution (the traces never claim a
`SCATTERING→IDLE` where the code would `→PARKED`; E1 step 3 says "ts->running true → re-park,"
i.e. the re-park path, which is fine).

**Timing fidelity (Model 2):** faithful, with the M2-2 minor under-allowance of σ_max. Source
δ (`delta_search`) and sink σ (`tau_fwd`) are positive bounded nondeterministic, neither
pinned; the σ⊥B padding consequence is correctly derived (`:171-172,385`). Model 2 ALSO
correctly resolves the misleading warmup comment (`:397` "ONCE PER EXACT BATCH SIZE B") as the
PRE-padding rationale superseded by `pad_to=max_batch` — a careful, faithful read.

**Fidelity verdict (Model 2): MIXED (faithful execution-set, two minor state-label/timing
defects).** No fatal/major over-constraint; the admitted-execution set is not materially
narrowed. The verdict is "too-constrained" only at the minor state-graph-labeling altitude
(M2-1), so for the schema's per-model verdict I record `mixed`.

---

## Cross-model note

The two models DISAGREE on exactly one load-bearing structural choice and one minor one:

1. **The strict-barrier driver (D=1).** Model 1 collapses it into "the D=1 specialization of
   the pipelined driver" (one state machine, DOF-4). Model 2 models BOTH driver bodies
   explicitly (E4 is the strict-barrier round; E1-E3 the pipeline). For the TOO-CONSTRAINED
   lens, BOTH are faithful: the admitted-execution sets coincide at D=1 (the per-slot lambdas
   are line-for-line identical, `runner_wire_batched.cpp:441-443`, and the strict body issues
   exactly one message per round, `:321-323`). Model 2 is more faithful at the structural-
   labeling altitude (it does not assert an equivalence it would have to defend); Model 1 is
   more economical and its equivalence claim is TRUE. Neither forbids a real execution. On
   balance Model 2 is marginally more faithful HERE (it represents the two code paths as the
   two code paths), but the difference is presentational, not a fidelity hole in either.

2. **The send-block corner (SNDHWM=1000, SNDTIMEO=-1).** Model 2 carries an explicit
   `BLOCKED_SEND` state; Model 1 records it as an unreachable boundary and omits the state.
   The execution is genuinely unreachable under the resume-gated `inflight<D` cap (D≤8 ≪ 1000),
   so NEITHER forbids a real execution. Model 2's explicit option-state is slightly more
   faithful to the SOCKET-OPTION semantics (the latitude exists in the option settings even if
   the control flow never reaches it); Model 1's omission is defensible as not-modeling-an-
   unreachable-corner. This is a where-to-draw-the-reachability-line judgment, not a defect in
   either.

On the M2-1 state-edge defect specifically, **Model 1 is more faithful**: its `FINALIZED →
PARKED` (fresh episode via `fill`, `:593`) vs `FINALIZED → IDLE` (subset exhausted) split
captures the post-finalize fork the code actually produces, which Model 2 collapses to a
single `→ IDLE` edge.

Net: both models are faithful in the too-constrained direction (the set of representable
executions is not narrowed below the code's in any load-bearing way). Model 1 is the cleaner
state machine for the post-finalize fork; Model 2 is the more explicit one for the two driver
bodies and the send-block option-state. The only concrete fix is M2-1 (split Model 2's
`SCATTERING→IDLE` edge) and the M2-2 σ_max allowance.

---

## Validation

The producer-side bounded Z3 confirmations already in this directory were inspected (NOT
re-derived from):
- `check_e2_admissible.py` — confirms E2's out-of-order pipelined interleaving (c1 reply
  before c0) is SAT under positivity, round-trip, reply-after-forward, totally-ordered
  forwards; AND that FIFO is separately SAT (so DOF-3 reply order is genuine free latitude,
  not pinned). This directly substantiates C-3 (no forced FIFO) — the too-constrained risk
  most likely to be a real hole, here shown absent.
- `producer_transport_check.smt2.py` — Model 2's E1+E3 out-of-order encoding, reported SAT.

Both checks confirm the models do NOT over-constrain reply ordering. I did not run the
C++/Python system and did not run a new solver sweep (the host is shared); the existing
checks suffice to confirm the one ordering latitude at issue. No new bounded check was needed:
the findings are state-label/timing-narrative precision defects (M2-1, M2-2), not
admitted-execution-set holes, so an SMT witness would not discriminate them.
