# RECONCILE — fresh parametric models vs the prior N=1 baseline

**Role: Reconciler (pre-synthesizer).** This document cross-checks the four fresh
(parametric, both-drains) side-models and their four audits against the PRIOR faithful model
(`prior-model/n1-baseline-model.md`, derived blind at N=1, production greedy drain only). The
objective is to "capture up" anything the fresh models missed by differential comparison —
without rubber-stamping the prior. Citations are cross-referenced by SYMBOL / function / code
structure (the two line bases do not correspond: the cleanroom has comments stripped).

Every claim below is grounded in cleanroom source read end to end for this reconciliation:
`runner_wire_batched.cpp` (1-506), `wire_leaf_pool.hpp` (1-173), `inference_server.py` (1-239),
`stage_a_server.py` (1-131), `inference_wire.py` (1-107), `wire_spec.hpp` (1-25), `forward.py`
(1-19), `fiber_tree.hpp` (1-66), `fiber_leaf.hpp` (1-36), `runner_wire_batched.hpp` (1-37),
`runtime_config.hpp` (1-26).

---

## 0. The headline differential

**The prior's single most load-bearing structural conclusion — per-thread in-flight depth is
identically ONE, the "D knob is dead" — SURVIVES PARAMETRICALLY. It holds for every N, not just
N=1.** The fresh PRIMARY producer models (both) REGRESS on this: they reintroduce the exact
depth-toward-D / out-of-order-per-thread latitude the prior had already killed, and assert it
GROWS with N. The fresh AUDITS (too-permissive and too-constrained) independently re-derived the
prior's depth-1 fact and correctly flagged the primary models — so the correction is already
present in the fresh corpus, but it lives in the audits, not the headline models. A synthesizer
must adopt the AUDIT verdict over the PRIMARY model verdict on this axis.

I re-verified the depth-1 fact line-by-line against `run_episodes_wire_pipelined` and confirmed
it parametrically with a bounded Z3 check at K ∈ {2, 8, 24} (stand-ins for N·base): `inflight==2`
is UNSAT at every K under a model strictly LOOSER than the code
(`out/reconcile_depth_parametric.py`; the prior's own UNSAT was at N=1 only). The control-flow
proof is N-independent: `issue_one` (cleanroom 434-452) gathers EVERY `is_ready` slot into ONE
message and sets `submitted[s]=1` for all of them (447); a slot regains `is_ready` (429:
`active && running && !submitted`) ONLY inside the post-`recv_batch` completion loop (462-472,
via `resume_with`/`advance`/`fill`); so between two `issue_one` calls with no intervening recv no
slot becomes newly ready, the second `issue_one` hits `gathered.empty()` → returns false (444),
and PRIME (456) and every REFILL (474) issue exactly ONE message. K = N·base (286) changes the
ROW count inside that one message, never the message count.

---

## 1. CARRY_FORWARD — each load-bearing prior structural finding

| # | Prior finding (N=1) | Status | Code-grounded note |
|---|---|---|---|
| CF-1 | **Per-thread in-flight depth ≡ 1; D is a dead knob** (prior §1.1, §7.1, §8-B). | **confirmed** | Holds for ALL N. Gather-all `issue_one` (434-452) + readiness only inside the recv completion loop (462-472) ⇒ `inflight_msgs ∈ {0,1}`. K=N·base (286) scales rows/msg, not msg count. Z3 UNSAT for `inflight==2` at K∈{2,8,24} (`reconcile_depth_parametric.py`). The fresh too-constrained audit (cross_model_note) and too-permissive audit (F1.1/F2.1) both re-derive this; the fresh PRIMARY models contradict it. |
| CF-2 | **All batch growth / all server-visible concurrency is CROSS-thread** (prior §0, §5-R2, §7.2). | **holds-with-caveat** | Mechanism confirmed: one message/thread, ≤T messages on the wire, server batch = #threads whose message arrived during the prior forward. CAVEAT the prior could not state: at N=1 the per-message row count is fixed at S∈[1,base]; for N>1 each of those ≤T messages is FATTER (up to K=N·base rows). So cross-thread coalescing is still the only mechanism, but its per-message payload now scales with N — the fresh models' one genuine parametric contribution. |
| CF-3 | **DEALER sets only LINGER=0 and RCVTIMEO=timeout_ms; SNDHWM/RCVHWM/SNDTIMEO at defaults** (prior §8). | **confirmed** | Exact: `wire_leaf_pool.hpp` create() sets `ZMQ_LINGER=0` and `ZMQ_RCVTIMEO=timeout_ms` only (cleanroom 39-41). No SNDHWM/RCVHWM/SNDTIMEO/context opts anywhere. N-invariant. |
| CF-4 | **ROUTER sets NO socket options ⇒ ROUTER_MANDATORY=0 ⇒ scatter DROPS, never blocks** (prior §1.2, §7.6). | **confirmed** | Exact: `inference_server.py` __init__ does `socket(ROUTER)`+`bind`+poller-register, no `setsockopt` (cleanroom 153-156). Scatter `send_multipart` (200) silently drops to a full/vanished peer. The fresh server too-permissive audit independently re-killed the greedy-drain model's phantom send-block (its Finding 1.1). N-invariant. |
| CF-5 | **RCVTIMEO-bounded recv is the SOLE producer blocking point and sole liveness backstop** (prior §3-6, §7.7). | **holds-with-caveat** | The structure is confirmed (one blocking `zmq_msg_recv`, 147; SNDHWM unreachable at D·T≪1000). CAVEAT the prior assumed `timeout_ms=15000>0` and stated "never a silent hang" categorically. The fresh too-constrained audit (P2/Q2) correctly notes `timeout_ms` is an UNVALIDATED int handed straight to `ZMQ_RCVTIMEO` (41; default 15000 at `runner_wire_batched.hpp:22`): `timeout_ms<0` ⇒ block forever = silent hang; `timeout_ms==0` ⇒ instant EAGAIN. The prior's "never a silent hang" is true only for the default positive value. N-independent. |
| CF-6 | **Forward service σ is positive, never instant, shape-INVARIANT in B under pad-to-max** (prior §1.3, §3, §7.3). | **holds-with-caveat** | Confirmed for the PRODUCTION greedy drain (pads to `max_batch`, 198; one compiled shape via cached `jax.jit`, 22-34; `forward_core` row-independent, forward.py:3-18). CAVEAT: the prior modeled ONLY this drain. Under the BENCH drain (out of prior scope), σ is NOT shape-invariant — bucket E-policy gives a 3-step σ over {64,256,512} then UNPADDED for real>512, and leaf wakeup gives multiple forwards/drain. See DRAIN_CONTRAST §4. |
| CF-7 | **Soft-cap overrun: cap tested on PRE-request total ⇒ a message crosses max_batch whole; second compiled shape** (prior §1.3, §3-σ4, §5-R5, §7.10). | **confirmed and SHARPENED** | Cap checked at loop TOP (`inference_server.py:171`), last message appended whole (184-185); `pad_to>B` false ⇒ unpadded larger shape (58). Prior: reachable "only if a single thread's K > max_batch" — at N=1, K=base=8 ≪ 256, UNREACHABLE under defaults. SHARPENED parametrically: K=N·base, so overshoot becomes reachable once **N > max_batch/base**, and the overshoot magnitude is K−1 = N·base−1 (grows linearly in N). This is the one place N directly enlarges the forward SHAPE. The fresh models (bucket-drain Exec §5.4, transport DOF-7) derive this correctly and Z3-witness it (K=600>512). |
| CF-8 | **Reachable EXCEPTIONAL_TERMINATION server terminal** (ragged in_dim / bad forward shape / reload raise), RELY-gated, distinct from clean `_stop` (prior §1.2, §2.2, §7.9). | **holds-with-caveat** | Confirmed reachable: `run_microbatch` raises uncaught on ragged in_dim (51-53) and bad output shape (62-63); `RedisParamsSource.poll`→`read_weights` can raise; none caught → kills the server thread. CAVEAT: the BENCH server uses `StaticParamsSource` (`current()` only, stage_a:56), so the reload-raise sub-case is impossible there; only ragged/bad-shape remain. The fresh server models note the malformed-frame `_reject` path but UNDER-DERIVE the uncaught-`ValueError` thread-death terminal — see GAPS §6. |
| CF-9 | **Loud bilateral abort, never silent wrong-slot apply; corr-id matched, B-exact checked** (prior §4.1-PG5, §7.8). | **confirmed** | `wire_leaf_pool.hpp` recv_batch hard-errors on unknown corr (116-118) and size mismatch (121-124); globally-unique corr via shared `corr_seq.fetch_add` (84). N-invariant (a protocol property). |
| CF-10 | **Strict-barrier driver is the PRODUCTION DEFAULT; pipelined arm is behind the mode flag** (prior §1.1, §8-attestation). | **confirmed, and NOT ADDRESSED by the fresh models** | `WireMode::StrictBarrier` is the struct default (`runner_wire_batched.hpp:23`); `run_episodes_wire_batched` dispatches to `run_episodes_wire_pipelined` only when `mode==PipelinedBucket` (cleanroom 44-45). The strict-barrier driver (39-268) is ALSO per-thread depth-1 BY CONSTRUCTION (one `submit_batch`/one `recv_batch`/resume-all per loop, 235-237) and has NO `trees_per_thread` at all (its K=base, not N·base). The fresh models ALL model only the pipelined arm. Consequence for N-dependence: **N is a pipelined-driver-only parameter; the production-default driver is entirely N-invariant.** This is a CARRY_FORWARD the fresh corpus dropped — see GAPS §6. |
| CF-11 | **Self-clocking / negative-feedback batch-size fixed point (server self-batches via single-thread off-socket accumulation)** (prior §3-σ5, §5-R2 stability). | **confirmed and EXTENDED** | The single-threaded server does no recv during a forward, so requests accumulate for the next drain (`inference_server.py` serialized serve_forever 219-225). The fresh server models all reproduce this (greedy E1/E2, transport Exec C self-batching, bucket "FORWARDING-window accumulation") and Z3-confirm it. EXTENDED parametrically: larger N ⇒ fatter messages ⇒ longer σ (under bucket) ⇒ wider accumulation window ⇒ fatter next drain — a stronger feedback gain than the prior's N=1 view, saturating at the max_batch/512 ceiling. |
| CF-12 | **corr-id reorder tolerance offered to peer but VACUOUS per-thread on one socket (slack, not a gap)** (prior §4.3-R3, §4.4-2). | **confirmed** | Because depth≡1, a thread never holds 2 outstanding messages, so it never observes per-socket reorder; `inflight_.find(corr)` always hits the single entry. The fresh primary models WRONGLY make this reorder a live, N-growing DOF (DOF-4/DOF-T6); the fresh too-permissive audit (F1.1/F2.1) correctly restricts reorder to ACROSS DEALERs (T≥2). Cross-thread reorder is real and N-independent in mechanism. |

---

## 2. DISCREPANCIES — fresh vs prior

| Topic | Prior claim | Fresh finding | More faithful | N-dependent divergence? |
|---|---|---|---|---|
| **Per-thread in-flight depth** | Identically 1; D dead (§1.1). | PRIMARY producer models: depth rises 1→D as N grows ("staggered waves fill D", DOF-3/DOF-T3, rows-in-flight ~D·B). AUDITS: depth≡1 for all N, D dead (UNSAT for inflight==2). | **prior** (and the fresh AUDITS, which agree with it). The fresh PRIMARY models are wrong. | The DIVERGENCE is itself N-framed: the primary models claim the regime APPEARS/GROWS with N (escapes a "K≤D corner" as N passes D/base). It does not — depth is N-flat at 1. The primary models manufacture an N-dependence that the code forecloses. |
| **What N actually scales** | (Out of scope at N=1; prior notes K=ceil(pool_batch/T) with no N factor.) | rows-per-message up to K=N·base; message count capped at D (really 1); payload bytes ~N; mean_rows_per_msg rises, total_msgs falls. | **fresh** — this is the fresh corpus's correct and valuable parametric contribution (the producer-transport model and both server models state it cleanly). | YES — this is the true N-axis: N fattens each message linearly; it does NOT add messages or pipeline depth. |
| **Single-thread reply order** | FIFO per pipe; per-thread reorder vacuous (§4.4-2). | PRIMARY: any permutation of a thread's outstanding replies (DOF-4/DOF-T6), Z3-"confirmed" drain order 1,3,2 on one socket. AUDIT: FIFO per pipe; reorder only across DEALERs. | **prior** / fresh audit. The primary models' Z3 scripts certify an inadmissible single-socket reorder (the audit's `verify_producer_fifo_check.py`: SAT as-written, UNSAT with FIFO-per-pipe). | N-independent in mechanism; the primary models' "more reorder per unit work as N grows" is doubly wrong (reorder is unreachable at any N within a thread). |
| **leaf-wakeup forward count** | (Bench out of scope.) | greedy-drain server model: "leaf multiplies forward count with N." bucket/transport models + too-permissive audit: forward count per drain bounded by #messages drained (≤ T·D, really ≤T given depth-1), N-INDEPENDENT; N fattens each forward, not their count. | **fresh majority** (bucket model, transport model, too-permissive audit). The greedy-drain server model overstates. | The greedy model invents an N-growth in forward COUNT; the real N-effect under leaf is rows-per-forward / bucket climb. |
| **σ vs B** | σ shape-invariant in B (pad-to-max), constant shape (§1.3). | Production: same (shape-invariant). Bench bucket: σ is a 3-step function of real rows, then unpadded for >512; bench leaf: multiple σ per drain. | **both-partial**: prior is faithful for the production drain it modeled; fresh extends correctly to the bench drain the prior never saw. | YES under bench: N climbs the bucket ladder (σ steps up); N-flat under production until the overshoot threshold. |
| **timeout_ms sign** | "never a silent hang"; reply within 15000ms (§3-6, R4). | too-constrained audit: `timeout_ms` unvalidated; <0 ⇒ silent hang, ==0 ⇒ instant EAGAIN. | **fresh** (the audit). The prior over-constrained by assuming the default positive value. | N-independent. |
| **drain message/row ceiling** | (N=1: ≤T messages, T·base rows ≤ pool_batch.) | greedy-drain server model: "saturation ceiling T·K"; transport/bucket: message count ≤ T·D (≤T), rows ≤ max_batch+(K−1). | **fresh majority** (the T·K message ceiling is the same slot-vs-message conflation; the correct row ceiling is max_batch+K−1). | The conflation grows with N (T·K = T·N·base), making the greedy model's error N-amplified. |

---

## 3. N_DEPENDENCE_SUMMARY — where the prior's N=1 conclusions stop holding

The prior's N=1 conclusions are **structurally robust to N**: every load-bearing structural
fact (depth-1, cross-thread-only batching, the two socket-option facts, the single bounded
blocking point, loud-abort, the self-clocking feedback, the EXCEPTIONAL_TERMINATION terminal)
holds verbatim for all N. **N is purely a per-message ROW-COUNT / throughput-utilization knob,
not a structural one.** The places where N changes something the prior could not see (because at
N=1, K=base and the quantity is small/unreachable):

1. **Per-message payload scales linearly: rows/msg ≤ K = N·base** (286 × gather-all 437-444).
   The prior fixed S∈[1,base]; for N>1 each of the ≤T cross-thread messages carries up to N·base
   rows. This is the ONLY first-order N effect. (CF-2 caveat.)
2. **Soft-cap overrun crosses from unreachable to reachable at N > max_batch/base**, with
   overshoot magnitude K−1 = N·base−1 growing linearly (CF-7). At N=1, K=8≪256 (prod) / ≪512
   (bench) ⇒ the prior correctly called R5 unreachable under defaults; it becomes reachable and
   then routine as N grows. This is the one place N enlarges the forward SHAPE (and can trip a
   cold compile on an unwarmed over-cap shape).
3. **The self-clocking feedback gain rises with N** (CF-11): fatter messages → longer σ (bench
   bucket) or wider accumulation → the batch-size fixed point climbs the bucket ladder / toward
   max_batch faster; saturation reached at lower wall-time, after which N converts to queueing
   latency rather than larger forwards.
4. **The RCVTIMEO firing PROBABILITY (not its bound) rises with N** via the peer: larger N ⇒
   fatter coalesced server batches ⇒ longer σ ⇒ closer to `timeout_ms`. The producer-side
   mechanism is N-invariant; only the latency that triggers it grows. (Prior R6, parametrized.)
5. **The episode-tail ramp-down lengthens to ~N·base episodes/thread** (producer DOF-6): more
   slots to drain to EMPTY at end-of-run, so declining-coalescing tail is longer. (Not in prior
   scope; a faithful fresh corollary.)

**Where the prior conclusions DO NOT stop holding (the central differential answer):**
the load regime is STILL self-correcting (CF-11 negative feedback) for all N under the production
drain — N raises the fixed-point batch size and the saturation ceiling, but the feedback remains
negative (a slower forward → larger batch → same padded cost amortized over more rows → higher
throughput → backlog drains). The prior's "self-correcting" conclusion SURVIVES parametrically
under production. The fresh server models corroborate this (negative-feedback fixed point in
greedy E1/E2 and transport Exec C). It is NOT destabilized by N. The depth-1 backpressure
(≤ pool_batch·N... no: ≤ T messages, each ≤K rows, so ≤ T·K = T·N·base rows offered at once)
still bounds the standing backlog, so no unbounded queue forms at any N.

---

## 4. DRAIN_CONTRAST — production greedy drain vs bench bucketed-group drain

The prior modeled ONLY the production greedy drain. The fresh corpus adds the bench. The two
share the SAME transport spine — crucially, `StageAServer` overrides ONLY `_serve_batch` and
INHERITS `_drain`, `serve_forever`, `run_microbatch` (stage_a:39,54). So the GREEDY NON-BLOCKING
DRAIN (loop-top row cap, over-fill, fair-queued ROUTER pull) is IDENTICAL in both. The reachable
regimes differ ONLY in the SHAPING (`_serve_batch`):

| Axis | Production greedy (`inference_server._serve_batch`) | Bench (`StageAServer._serve_batch`) |
|---|---|---|
| Forwards per drain | Exactly ONE (one `run_microbatch` over all drained, 196-198). | `wakeup=group`: ONE. `wakeup=leaf`: one per drained MESSAGE (`[[d] for d in drained]`, 57) — NOT per row; each message is up to K=N·base rows. Count ≤ #messages drained (≤T given depth-1), N-INDEPENDENT in count. |
| Forward shape (σ regime) | ALWAYS pad to `max_batch` (198) ⇒ one compiled shape ⇒ σ N-INVARIANT (until overshoot). | `padmax`: same as production. `bucket`: snap real up to {64,256,512} (`_bucket_for`, 32-37), UNPADDED for real>512 ⇒ σ is a 3-step-then-rising function of real rows ⇒ σ STEPS UP with N. |
| Weight reload | `poll()` each batch (194) ⇒ mid-stream weight swap possible (production-only DOF). | `StaticParamsSource` → `current()` only (56) ⇒ NEVER reloads. |
| max_batch default | 256 (`inference_server.py:145`). | 512 (`stage_a_server.py:89`). Shifts the overshoot threshold (N>512/base vs N>256/base) and the bucket clamp. |
| Warmup | Caller-supplied; `serve_forever` does NOT warm ⇒ cold-compile tail possible on first forward. | `build()` warms {64,256,512,max_batch} (82) ⇒ no cold compile for those shapes (but overshoot/over-512 shapes are unwarmed). |
| EXCEPTIONAL_TERMINATION sub-cases | ragged in_dim, bad output shape, AND reload-raise. | ragged in_dim, bad output shape only (no reload). |

**Reachable-regime contrast in one line:** the production drain can ONLY coalesce-up
(one fixed-shape forward per drain, σ flat in N until overshoot) — it has no anti-coalescing or
σ-stepping latitude; the bench drain can additionally (a) DE-coalesce (leaf: one forward per
message, the deliberately pessimal anti-batching baseline), and (b) STEP σ with load (bucket
ladder, converging to production only when the bench `max_batch=512` equals the top bucket). For
both drains the in-flight depth is 1, the batching is cross-thread, and the drain itself is the
same greedy non-blocking pull — the prior's transport conclusions transfer to the bench unchanged;
only the SHAPING/σ conclusions are production-specific and must be re-derived per E-policy/wakeup.

---

## 5. Where the fresh models are MORE faithful than the prior (genuine captures)

- The **rows/msg = K = N·base** parametrization (producer-transport, both server models) is the
  correct N-axis and is cleanly derived; the prior could not see it at N=1.
- The **bench drain** (both server models) is wholly outside prior scope and faithfully derived
  (bucket ladder, leaf de-coalescing, the over-512 unpadded forward, StaticParamsSource no-reload).
- The **timeout_ms sign** edge (too-constrained audit) is a real refinement of the prior's
  categorical "never a silent hang."
- The **episode-tail ramp-down ~N·base** (producer DOF-6) is a faithful N-corollary the prior
  had no occasion to state.

## 6. GAPS_TO_INVESTIGATE — focused, code-answerable, left open or under-derived

| # | Question | Why it matters | Target files |
|---|---|---|---|
| G-1 | **Does the production greedy drain's negative-feedback fixed point remain stable for ALL N, or does a fast enough σ-step under any reachable config flip it?** The fresh server models ASSERT self-reinforcing stability (greedy E2, transport Exec C) but do not derive a stability CONDITION; the prior asserted self-correction at N=1. Under production σ is N-flat (good), but verify no config (e.g. K>max_batch overshoot recompiles) injects a positive-feedback σ spike that grows the next batch faster than it drains. | The prior's "self-correcting load regime" is the central differential claim; it must be shown to survive N, not assumed. | `inference_server.py` (_drain 160-186, _serve_batch 192-200, run_microbatch 40-73); `runner_wire_batched.cpp` (456,474). |
| G-2 | **Which driver is actually in scope — strict-barrier (production default) or pipelined?** The fresh models ALL model only `run_episodes_wire_pipelined`; the strict-barrier `run_episodes_wire_batched` is the `WireMode` default (`runner_wire_batched.hpp:23`) and has NO `trees_per_thread` (K=base, N-invariant). A faithful composed model must state whether N-dependence is even reachable in the default configuration. | If production runs strict-barrier, the entire N-axis is dormant unless `mode==PipelinedBucket` is set; the prior flagged this (CF-10) and the fresh corpus dropped it. | `runner_wire_batched.cpp` (39-45 dispatch, 213-251 strict loop); `runner_wire_batched.hpp` (16-26). |
| G-3 | **Is the uncaught-`ValueError` server-thread-death terminal reachable under the conforming C++ peer, parametrically?** The ragged-in_dim raise (run_microbatch 51-53) fires only if two drained messages carry DIFFERENT in_dim. Within one net in_dim is fixed (`fb.dim()`), so it is RELY-excluded — but the fresh server models model `_reject` (malformed frame, no reply) and UNDER-DERIVE the distinct uncaught-exception thread-death the prior named (CF-8). Confirm whether any N makes a mixed-in_dim drain reachable (it should not, but the fresh models don't close it). | The prior carries EXCEPTIONAL_TERMINATION as a reachable terminal; the fresh models conflate/omit it. A synthesizer needs the distinction (drop-one-reply vs kill-the-server). | `inference_server.py` (run_microbatch 50-63, _serve_batch 192-200, serve_forever 219-225); `runner_wire_batched.cpp` (in_dim from `fb.dim()` 275,325). |
| G-4 | **Exact over-fill/over-512 cold-compile cost as a function of N.** Both server models Z3-witness the over-cap unpadded forward (K=600>512) but do not derive whether each distinct overshoot width N·base−ε compiles AFRESH (jax.jit per-shape, `_jit_forward_cache` 20-34) — i.e. whether large-N runs pay a recurring compile tax at varied overshoot widths, or a one-time tax per width. | Determines whether the N>max_batch/base regime degrades throughput transiently (one-time) or persistently (per-width). The prior called R5 transient at N=1; parametrically it could recur. | `inference_server.py` (jit_forward_core 22-34, run_microbatch pad guard 58); `stage_a_server.py` (_bucket_for 32-37, warmup set 82). |
| G-5 | **Does ROUTER fair-queue starvation interact with N?** The drain pulls in libzmq fair-queue order across T peers (not pinned by code). At large N each peer's single message is fatter, so a drain may hit the `max_batch` cap after FEWER peers' messages, deferring the rest. Confirm whether this can starve a slow producer thread of forward slots under sustained saturation as N grows. | The prior left this open (its open-Q 5) at N=1; N fattening messages changes how many peers fit under the cap per drain, a new parametric angle. | `inference_server.py` (_drain loop-top cap 171, recv_multipart 173); `runner_wire_batched.cpp` (gather-all message size 437-444). |
| G-6 | **mean_rows_per_msg telemetry as a closed-form check.** The pipelined driver emits `mean_rows_per_msg = total_leaves/total_msgs` (496-500). Since depth≡1 and each issue gathers all ready, derive the expected value as a function of (N, base, the park/reply timing) to give the synthesizer a falsifiable N-prediction (rises toward K=N·base as parking concentrates). | Turns the depth-1 fact into a quantitative, code-emitted, N-parametric prediction the prior could not make at N=1. | `runner_wire_batched.cpp` (my_leaves/my_msgs 449-450, summary 494-500). |

---

## 7. Bottom line for the synthesizer

Adopt the prior's structural spine VERBATIM — it is parametrically robust. Specifically: take the
prior's depth-1 / dead-D / cross-thread-batching / two-socket-option / single-bounded-block /
loud-abort / self-clocking conclusions as confirmed for all N, and OVERRIDE the fresh PRIMARY
producer models wherever they reintroduce depth-toward-D or per-thread reorder (the fresh AUDITS
already supply the override). Then GRAFT the fresh corpus's two genuine parametric additions:
(a) rows/msg = K = N·base as the sole first-order N-axis (fattening the ≤T cross-thread messages),
and (b) the BENCH drain's σ-stepping / de-coalescing latitude (DRAIN_CONTRAST §4), neither of
which the prior could see. Apply the prior's `timeout_ms` over-constraint fix (CF-5 caveat).
Close G-2 (which driver) and G-3 (EXCEPTIONAL_TERMINATION reachability) before declaring the
composed model complete.

**Bounded confirmation (theory's check, not its source).** `out/reconcile_depth_parametric.py`
under `nice -n 19 timeout 90` (z3 4.16): `inflight_msgs==2` UNSAT at K∈{2,8,24} (stand-ins for
N·base), under a model strictly looser than the code ⇒ per-thread depth ≡ 1 is N-INDEPENDENT,
extending the prior's N=1 UNSAT to all N. Confirmation only; trust is in the §0 control-flow
derivation.

*Public Domain (The Unlicense).*
