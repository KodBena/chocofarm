# Adversarial fidelity audit — Python server side (ROUTER InferenceServer + MLP forward sink)

**Lens: TOO-PERMISSIVE.** Find any execution a model ADMITS that the real code cannot produce.

Verifier read end to end (ADR-0002), independently of the models' citations:

- `chocofarm/az/inference_server.py` (1-457)
- `chocofarm/az/forward.py` (1-64)
- `chocofarm/az/inference_wire.py` (1-185)
- `chocofarm/az/wire_spec.py` (1-84)
- `cpp/include/chocofarm/wire_leaf_pool.hpp` (1-243)
- `cpp/src/runner_wire_batched.cpp` (1-630, both drivers)
- `cpp/include/chocofarm/inference_wire.hpp` (1-226, the C++ codec — to ground the wire RELYs)
- `docs/design/zmq-inference-service.md` (1-367)
- `chocofarm/config.py` (1-70, the XLA/OMP pin)

Installed transport: **pyzmq 27.1.0 / libzmq 4.3.5** (confirmed at runtime).

---

## The decisive socket-option fact (settles a cross-model contradiction)

The server ROUTER has **no `setsockopt` call anywhere** (`inference_server.py:315-318` is
`socket(zmq.ROUTER)` → `bind` → register the poller; nothing else). So every option is at the
libzmq-4.3.5 default: `RCVHWM=1000`, `SNDHWM=1000`, `SNDTIMEO=-1`, and **`ZMQ_ROUTER_MANDATORY=0`**.

The two models disagree, load-bearingly, on what a scatter `send_multipart` (`:387`) does when a
KNOWN peer's outbound pipe is full (at SNDHWM):

- **Model A (model-server-drain):** silent drop (DOF-7).
- **Model B (model-server-transport):** **blocks indefinitely** (E7 / DOF-7 / a dedicated
  `SCATTER→SCATTER` "send blocks indefinitely (SNDTIMEO=-1)" transition).

The authoritative libzmq-4.3.5 `zmq_setsockopt(3)` text for `ZMQ_ROUTER_MANDATORY` resolves it:

> "A value of **0 is the default** and **discards the message silently when it cannot be routed or
> the peers SNDHWM is reached**. A value of 1 returns an EHOSTUNREACH error code if the message
> cannot be routed or EAGAIN error code if the SNDHWM is reached and ZMQ_DONTWAIT was used. Without
> ZMQ_DONTWAIT it will block until the SNDTIMEO is reached or a spot in the send queue opens up."

The blocking-on-SNDHWM behavior is a **consequence of setting ROUTER_MANDATORY=1**. Under the
default (=0), the SNDHWM-reached case **drops silently**; it does not block, and `SNDTIMEO=-1` is
never reached because the send never enters the blocking path. The `zmq_socket(3)` summary
"Action in mute state: Block" is the *generic* per-type characteristic; the ROUTER_MANDATORY=0
default overrides it for ROUTER, dropping on BOTH unroutable AND SNDHWM-reached.

**Therefore Model B's E7/DOF-7 indefinite-block wedge is an execution the code under its actual
(default) options cannot produce — a too-permissive admission.** Model A's silent-drop
characterization is faithful.

---

## Model A — `model-server-drain.md`

### Findings

**A-1 (minor, admits-too-much, possibly over-permissive but defensible).** DOF-1's "recv order
among simultaneous arrivals is free" is correct *as a server-side model* (`:350` fair-queues across
peer pipes; the server cannot pin it). Faithful — flagged only to confirm it is latitude the server
code genuinely leaves (libzmq cross-pipe fair-queue), not invented freedom. No correction.

**A-2 (minor, unfaithful-rely scoping).** R4 bounds the free batch range at `T` (strict) or `T·D`
(pipelined) outstanding. This is correct for the modeled peer (`runner_wire_batched.cpp:392,438,
578-596`), but the server-side drain coalesces ACROSS threads, so a single drain's `total_rows`
free range is bounded by `Σ_threads` outstanding ≈ `T·D·(mean rows/msg)`, not a single thread's
`T·D`. Model A's own E3/E4 use multi-thread coalescing correctly; the R4 prose just under-states the
aggregate. The cap (`max_batch` + soft overrun) is the operative server-side bound regardless, so
this does not admit any impossible execution. Correction: state the aggregate `Σ`-over-threads
bound, not the per-thread one, when grounding the batch-size free range.

Otherwise: A's state machine, the soft-cap overrun (DOF-2, faithful to the `total_rows < max_batch`
pre-test at `:348` + whole-request append at `:361-362`), the unwarmed-overrun JIT spike (DOF-4/E4,
faithful — warmup at `:426` only ever compiles the single `(max_batch,in_dim)` padded shape via
`pad_to=max_batch`, so a 70-row overrun is genuinely unwarmed), the single-thread serialization
(C-svc-5), the silent scatter-drop (DOF-7, faithful per the man page above), the reject path
(`total_rows` unchanged on `continue` at `:360` before `:362`), and the between-batch reload
(DOF-6) are all derived forward by line and admit exactly what the code admits.

### Trace check
All six representative executions (E1-E6) are genuine schedules. Every step maps to an enabled
transition: E1/E3 self-clocking (`:339-363`), E2 burst-split via the cap pre-test, E4 cap-overrun →
unpadded unwarmed forward, E5 reject + empty-drain re-poll (`:438,:339`), E6 between-batch reload
(`:381,:283-288`). The Z3 witness (batch sizes 4 then 1, serialized drains `t1≥f0`, `svc>0`,
reply-after-forward) is admissible. No step relies on an over-permissive move.

### Timing fidelity
Faithful. Service time is a bounded-positive duration, never instantaneous (`:177` `np.asarray`
blocks on XLA). The derived near-constancy across B≤max_batch (C-svc-3) is grounded in the single
compiled padded shape (`:171-172`, `:95-115`) and the warmup, and is correctly broken by the
overrun regime (C-svc-4). Arrivals are free positive intervals constrained only by the reply-gated
closed loop (C-arr-2 / R3, grounded in the driver's `submitted[]`/strict-barrier discipline). No
timing collapsed to a constant; no impossible ordering admitted.

### Verdict: **faithful.**

---

## Model B — `model-server-transport.md`

### Findings

**B-1 (FATAL, admits-too-much / missing-blocking-semantics — the headline too-permissive defect).**
Model element: the `SCATTER→SCATTER` transition "send to a KNOWN identity whose outbound pipe is at
SNDHWM=1000 (RELY R3 violated) → send blocks indefinitely (SNDTIMEO=-1)"; DOF-7 "a send to a
HWM-full KNOWN peer pipe blocks indefinitely"; and E7 (the whole "send_multipart stall under RELY
violation" execution, "server parked indefinitely … absorbing wedge").
Expected code ref: `inference_server.py:387` + libzmq-4.3.5 `ZMQ_ROUTER_MANDATORY` default
(=0, no setsockopt at `:315-318`).
Defect: with ROUTER_MANDATORY=0, the man page is explicit that a send is "discard[ed] … silently …
[when] the peers SNDHWM is reached"; blocking is the ROUTER_MANDATORY=**1** behavior. So the
indefinite-block-in-SCATTER execution **cannot be produced** by the code under its actual options.
The model invents a wedge (and an entire representative execution E7) that the default ROUTER
removes. The `SNDTIMEO=-1` default is real but irrelevant: the send never enters the blocking path.
Correction: replace the block transition/DOF/E7 with the silent-drop behavior (Model A's DOF-7) — a
send to a full OR vanished peer is silently dropped, the loop continues, and the peer's finite
RCVTIMEO (R6) surfaces the lost reply as a loud timeout. The only ROUTER_MANDATORY-related blocking
execution would require a `setsockopt(ZMQ_ROUTER_MANDATORY,1)` the code never makes.

**B-2 (major, unfaithful-guarantee — downstream of B-1).** G5 ("the only unbounded wait
(send_multipart at SNDHWM) is excluded by R3") and the timing-model causal frame both rest on the
false premise that send-on-full BLOCKS. Since it DROPS, there is no unbounded send-wait to exclude;
the server's only waits are the bounded 100ms poll and the between-batch reload. The guarantee's
*conclusion* (bounded liveness under R1-R4) is actually STRONGER than B states — the SCATTER step is
unconditionally non-blocking under default options — but the stated *mechanism* (R3 excludes a real
block) is unfaithful to the code. Correction: liveness holds because the ROUTER never blocks on
scatter at all (drop semantics), independent of R3; R3's real role is only that a dropped reply
surfaces loudly at the peer (R6), not that it prevents a server-side stall.

**B-3 (minor, unjustified-free-choice labeling).** B labels `POLL_WAIT→DRAINING` and
`DRAINING→DRAINING(append)` as `is_free_choice: true` "free (timing)/(arrival count)". The *count*
is genuinely free (arrival schedule); the *transition action* is determined. This is the same
freedom Model A localizes more precisely (the freedom lives in the arrival timeline / drained
membership, the per-step action is fixed). Not an admitted-impossible execution; a presentation
imprecision. Correction: locate the free choice in the (arrival schedule, service duration) input
pair, as B's own stance says, and mark the per-step recv action determined.

Otherwise B is strong: the pad-to-fixed-shape service-time shape-invariance (S4, grounded in
`:171-172` + the one compiled executable, `forward.py:50-63` row-independence), the cold-compile
tail kept distinct (S5), the soft-cap overrun ADMITTED in the self-audit ("B exceeding max_batch …
pad_to>B false → larger executable"), and the reject `total_rows`-unchanged accounting are all
faithful and well-cited.

### Trace check
E1-E6 are genuine schedules and every step maps to an enabled transition (the drain/forward/scatter
loop, the cold→warm tail E3, the between-batch reload E4, the reject E5, the bounded-poll shutdown
E6). **E7 is NOT an admissible schedule under the actual code**: its load-bearing step ("send to a
full known pipe → BLOCK") is disabled by ROUTER_MANDATORY=0 (silent drop instead). `traces_admissible`
is therefore false for the representative set as a whole, on account of E7.

### Timing fidelity
Faithful on the source/sink axes (emission e_r free positive post-reply, service s_f free positive
with a cold tail, non-overlap S3, shape-invariance S4 correctly derived not collapsed). The timing
DEFECT is not in the duration model but in the discrete blocking semantics of the scatter step
(B-1): a too-permissive *control-flow* latitude (an unbounded stall) that the code forbids, dressed
as a timing/DOF freedom.

### Verdict: **mixed** (the duration/state model is faithful; the scatter-blocking transition, DOF-7,
and E7 are too-permissive — they admit an indefinite-wedge execution the default ROUTER cannot
produce).

---

## Cross-model note

Both models read the same code and agree on the core self-clocking automaton, the pad-to-one-shape
service-time near-constancy, the soft-cap overrun, the single-thread serialization, and the reject
path. They disagree on exactly one load-bearing fact: what a scatter `send_multipart` to a full
KNOWN peer pipe does. **Model A (silent drop) is faithful; Model B (indefinite block) is
too-permissive.** The libzmq-4.3.5 `ZMQ_ROUTER_MANDATORY` default (=0) discards silently on
SNDHWM-reached; blocking is the MANDATORY=1 behavior the code never enables (`:315-318` has no
setsockopt). Model B even builds a whole representative execution (E7) and a guarantee mechanism
(G5) on the false block premise. Model A's only blemish is an under-stated aggregate bound (A-2),
which admits no impossible execution.
