<!--
docs/design/cpp-eval-wire-formal-diagnosis.md
Purpose: formal-methods diagnosis of the C++<->Python leaf-eval flow-control "stall", plus the protocol
  design guidance it earned. Recovered final report of the formal-methods agent ac9094fce3a8cf204 (reaped
  before it could re-report). The Z3 bounded-model-checking artifacts + the self-audit live under
  ~/w/vdc/chocobo/runs/formal-stall/ (model*.py, convoy*.py, sweep.py, HACK_AUDIT.md). Companion to the
  EMPIRICAL diagnosis (the stall is a metastable coalescing LIVELOCK, not a deadlock) and to
  cpp-eval-transport-adapter.md sec 6/7. Cross-validated verdict: the protocol is correct + deadlock-free
  (deadlock query UNSAT across the config sweep), the bug is a throughput livelock (the 1:1
  message<->forward convoy, SAT-admissible), fixed by enforcing a minimum coalescing degree — server-side
  (max_queue_delay/preferred_batch_size, a tuning surface) or producer-side (never issue a sub-threshold
  message while inflight<D, a CLOSED structural fix). Provenance note: the convoy leg was guided by the
  empirical finding (handed in), so the independent leg is the deadlock-UNSAT; the convoy-SAT is a sound
  formalization, not a blind rediscovery.
Public Domain (The Unlicense).
-->

# Formal diagnosis: the C++↔Python leaf-eval flow-control protocol

## 1. The model

I built the control/flow state machine in **Z3 bounded model checking** (TLA+/SPIN weren't installed; z3-solver installed into the generic venv). Five models, escalating in faithfulness, all under `~/w/vdc/chocobo/runs/formal-stall/`:

- `model.py` — single producer thread, K slots, in-flight **message** cap D, request/reply channels, server greedy-drain. Deadlock query.
- `model2.py` — adds **T producer threads** + the server's **max_batch ROW cap** with indivisible requests.
- `model3.py` — adds **finite channel capacity (HWM)** and the missing **blocking-send** producer state (DEALER has RCVTIMEO but no SNDTIMEO).
- `convoy.py`/`convoy2.py`/`convoy3.py` — retargeted to a **throughput/liveness** property (rows-per-forward), with progressively correct queue modeling.
- `convoy4.py` — the decisive staggered-arrival convoy witness.

**What it captures (transitions mirror these code lines):**
- Slot lifecycle idle→parked→in-flight→resumed: `is_ready`/`issue_one`/`recv_batch` resume loop (`runner_wire_batched.cpp:541-596`).
- `issue_one` coalesces **all** ready slots into ONE message, `++inflight_msgs` (`:551-569`); prime loop `while inflight<D && issue_one()` (`:578`); refill `:596`.
- Transport corr-id↔ordered-slot map, 1:1 request↔reply (`wire_leaf_pool.hpp:129-196`).
- Server greedy `_drain` (block for ≥1, drain all queued up to max_batch **rows**, `inference_server.py:322-363`) + group-wakeup one-forward-per-drain with **one response per request** (`stage_a_server.py:97-120`).

**Abstractions / assumptions (the faithfulness ledger):** leaf features/NN values/search internals → "a leaf is requested / a result returns" (one slot = one row per ply); ZMQ pipes are reliable (no message loss); the server's bounded poll and the producer's RCVTIMEO are true waits (a *permanent* wait surfaces as a loud timeout-abort, not a silent hang — `wire_leaf_pool.hpp:70`); one producer thread suffices for the deadlock query (more senders can only help the server); max_batch ≥ K so the row cap doesn't bind at small K. `convoy4.py` abstracts the *staggering itself* into a free schedule Z3 chooses, rather than deriving it from per-slot search-latency variance — this is the model's main unfaithfulness (see confidence).

## 2. The result

**There is NO reachable deadlock.** Across the full sweep — T∈{1,2}, K(slots)∈1..5, D∈{1,2,4,8}, plies∈{1,2}, max_rows∈{1,2,4}, plus HWM=1 backpressure on both channels — the deadlock query is **UNSAT** to unroll depth 8–16. The reason is structural and clean: **request↔reply is strictly 1:1 by corr-id, and the greedy drain replies to every queued request whenever any is queued**, so the server can never be permanently blocked while a request is outstanding. This matches the field observation that the run *always eventually completes*. The message accounting (`inflight_msgs` ↔ one reply per `recv_batch`) is balanced; even blocking-send under HWM=1 cannot wedge it.

**The actual pathology is a metastable LIVELOCK, not a deadlock** — a recoverable throughput collapse. `convoy4.py` returns **SAT**: a sustained `rows/forward == 1` schedule with the pipe full (inflight=D=8) and work remaining is an admissible interleaving. The counterexample is an alternating **ARRIVE → FORWARD** schedule where each server wake sees exactly one arrived message, so it forwards 1 row, while D messages stay outstanding each carrying ~1 row.

**Root cause (the guard that produces it):** the server's `_drain` forwards **whatever is queued at its wake instant** with no minimum — and the producer's `issue_one` coalesces **whatever is ready at issue instant** with no minimum. Neither actor enforces a floor on the coalescing degree, so the degree is set entirely by arrival *timing*. Crucially, **the SAME protocol also admits a high-coalescing schedule** (`convoy4.py` healthy mode is also SAT, rows/forward=4). Both regimes are reachable; nothing in the protocol *forces* the good one. That non-forcing is the root cause. The convoy is the design's pre-identified failure mode: leaf turnaround ≈ inter-arrival spacing, so the pipe degenerates into lockstep 1:1.

**N-dependence:** the deadlock query is N-independent (UNSAT everywhere). The convoy needs the pipe to *stay* full (D outstanding) while each message carries ~1 row; higher per-thread slot count K makes the 1:1 schedule easier to sustain (more frequent single-slot frees), consistent with the empirical N=4 repro. I can state the **mechanism** (degree collapses when reply turnaround < inter-arrival spacing) but my models did **not** derive a clean critical N(D) arithmetic — `convoy4` is parameterized on D and total work and abstracts K. So treat N=4 as empirical, not formally derived.

**Invariant check (ADR-0012):** the P9/P7 invariants — every submitted leaf gets exactly one reply, no slot re-submitted while in-flight (`submitted[]` flag, `:437,564,588`), corr-id map consistency — show **no reachable violation** in any model. The protocol is correct; it is only slow in the convoy regime.

## 3. Proposed fix (invariant to add — diagnosis only, do not implement)

> **A forward must not fire below a minimum coalescing degree that is independent of arrival timing.**

Concretely, enforced by *either* writer of the coalescing degree (there are exactly two; both confirmed by enumeration):
- **Server-side (preferred, more robust):** add a Triton-style `preferred_batch_size` + `max_queue_delay` to `_drain` — after the first message arrives, wait up to a small bounded delay *or* until ≥ a row threshold before forwarding. It is throughput-only, so the wait is free latency. This caps under-coalescing regardless of producer timing.
- **Producer-side (complementary):** have the pipelined driver hold a parked slot briefly before `issue_one`, so S rarely degrades to 1.

Either independently breaks the convoy. **Caveat surfaced by the audit:** this converts a metastable failure into a *tuning surface* (the delay/threshold), not a closed structural invariant. A genuinely closed fix makes under-coalescing **unrepresentable** (e.g., the producer never issues a sub-threshold message while `inflight < D`).

## 4. Confidence and what it abstracts away

**High confidence** on the negative (no deadlock): the BMC is exhaustive to the bound across a wide config sweep, and the 1:1 corr-id structure makes the result robust. **Medium confidence** on the convoy as the *operative* cause: `convoy4` proves the bad schedule is **admissible**, not that real ZMQ/OS timing **enters** it or that it is **sticky** — the empirical bimodal-and-sticky evidence (server rows/forward 1.4 vs 55–177; 6 s vs 70 s) supplies the "entered + metastable" leg the model does not. The earlier `convoy.py`/`2`/`3` UNSATs are from an over-constrained (lockstep) model and must **not** be miscited as "convoy unreachable." The proposed fix's efficacy is **unverified-on-paper**.

Bounded runs only (timeouts on every invocation, no monitors/background loops); the chocofarm repo was **not** edited; all artifacts — five models, the sweep, and `HACK_AUDIT.md` — are under `~/w/vdc/chocobo/runs/formal-stall/`.

---

## How I would design this protocol (the part you asked for)

Now an expert on this seam, here's the design guidance the formal work earned:

**1. Never let the coalescing degree be a function of arrival timing alone.** This is *the* lesson. Greedy-drain ("forward whatever is queued now") is liveness-correct and deadlock-free, but its batch size is set by a race between leaf-turnaround and inter-arrival spacing. When those are comparable you get the metastable lockstep convoy — admissible, self-sustaining, and invisible to a deadlock checker. **Make a minimum batch a first-class server contract** (`max_queue_delay` + `preferred_batch_size`). The delay is pure latency you spend to buy throughput; in a throughput-bound generator it is free.

**2. Separate the liveness property from the performance property, and check both.** The protocol here is *correct* (no deadlock, exactly-once replies) and *slow* (convoy). A deadlock model checker will give a clean bill of health and miss the entire bug. Design-time, write down two properties: a safety/liveness one ("every leaf gets one reply; the run terminates") and a **quantitative throughput invariant** ("rows/forward ≥ θ in steady state"). The second is the one that actually governs your DPS.

**3. Decouple the producer's in-flight cap from the coalescing degree.** The convoy is sharpened by `issue_one` collapsing all-ready into one message *and* the cap D counting messages, so the pipe stays "full" at depth D while each message is trivially small. If you instead cap **outstanding rows** (or decouple "how many in flight" from "how big each is"), depth-fullness stops masking degree-collapse. Count the thing you care about (rows), not a proxy (messages).

**4. Keep the corr-id transport exactly as-is — it is the part that's right.** The 1:1 corr-id↔ordered-slot map, the opaque envelope round-trip (serialization ⊥ transport), the per-thread single-writer DEALER, and fail-loud on unknown corr-id / count-mismatch are precisely why there's no deadlock and no wrong-slot apply. That discipline (ADR-0012 P7/P9) is load-bearing; the throughput bug lives entirely in the *scheduling* layer above it, not the wire. Don't refactor the wire to chase DPS.

**5. Make the failure mode observable by construction.** The convoy was invisible until someone instrumented **rows/forward** on the server. Bake that counter (and mean-S on the producer) into the protocol's telemetry as a permanent SLI, with a logged warning when steady-state rows/forward drops below θ. A metastable regime you can't see is one you'll rediscover by gdb every few months.

**6. Prefer "unrepresentable" over "tuned" when you can afford it.** A `max_queue_delay` fixes the convoy but adds a knob that can be mis-set in both directions. Where the structure allows, make the bad state impossible to express (producer refuses to emit a sub-threshold message while it still has in-flight headroom) rather than merely unlikely. Closed invariants don't drift; thresholds do.

The meta-point: this protocol's hard parts (the IoC seam, the corr-id discipline, fail-loud) were done right, and the bug hid in the one place that *looks* trivial — "when does the server decide to run a forward." Adaptive/greedy batching is where throughput protocols go to die quietly, because the failure is a performance livelock that every correctness check passes.
