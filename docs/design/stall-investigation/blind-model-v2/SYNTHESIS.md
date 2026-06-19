# SYNTHESIS — the authoritative parametric model of the leaf-evaluation transport boundary

(path: out/SYNTHESIS.md — Public Domain, The Unlicense)

Synthesizer role. One faithful model composed from the two fresh producer models +
two fresh server models, their too-permissive / too-constrained audits, the
reconciliation against the prior N=1 model, and five targeted capture-up derivations.
For each element I adopt the version that survives BOTH fidelity lenses (admits no
execution the code cannot produce; forbids no execution the code can), correct the
fatal/major defects, compose the two sides via assume-guarantee, and then — as neutral
analysis — characterize the global behavior parametrically with explicit N-dependence.

All `file:line` refer to the cleanroom tree under
`/home/bork/w/vdc/chocobo/runs/leaf-eval-model-2/cleanroom`; these line numbers
correspond to the real source. Every claim below was re-grounded against the code read
end to end (all 14 cleanroom files, including the strict-barrier driver the fresh models
dropped). Bounded Z3 confirmations are confirmation only, never the source of trust.

---

## 0. The single most important structural fact (the spine of the whole model)

**Per-thread in-flight MESSAGE depth is identically 1, for ALL N, T, D.** This is the
load-bearing fact the reconciliation rescued: the two fresh PRIMARY producer models
regressed by re-asserting a depth-toward-D growth with N (DOF-3/DOF-5, DOF-T3); the
fresh AUDITS independently re-derived depth-1; the prior had it at N=1. The synthesis
adopts depth-1, parametrically, as proven.

Mechanism (N-independent, exact):
- `issue_one` (runner_wire_batched.cpp:434-452) gathers **every** `is_ready` slot into
  **one** message (`:437-444`, a loop with no subset-selection / no early break) and
  sets `submitted[s]=1` for **all** of them (`:447`).
- A slot regains `is_ready` (`active && ts && running && !submitted`, `:427-430`)
  **only inside the post-recv completion loop** (`:462-472`: `resume_with` re-parks, or
  `advance`/`fill` re-park a fresh leaf/episode).
- Therefore no slot becomes newly ready between two consecutive `issue_one` calls
  **without an intervening `recv_batch`**: the second `issue_one` hits
  `gathered.empty()` and returns `false` (`:444`). The prime loop (`:456`) and **every**
  refill loop (`:474`) issue **exactly one** message; `inflight_msgs ∈ {0,1}`.
- `K = N·base` (`:286`) scales **rows per message**, never message count.

Z3 (confirmation): `inflight_le1_check.py` (encoded **strictly looser** than the code —
RECV may re-park any subset, `ready` left free) returns **UNSAT for `inflight==2`**;
`reconcile_depth_parametric.py` re-runs it at `K ∈ {2,8,24}` (stand-ins for `N·base`):
UNSAT at every K. Depth is N-flat at 1. **Consequence: `D = max_inflight_msgs` is a dead
parameter on the pipelined path** (clamped `≥1` at `:287`, never binds the `<D` gate).

Everything downstream is a corollary of this fact crossed with two free
nondeterministic inputs (per-slot park interval; per-thread reply timing) and the
server's shape-dependent service time.

---

## 1. Scope: which driver, and is the N-axis even live? (capture-up CF-10)

The single entry point `run_episodes_wire_batched` dispatches on `wcfg.mode`
(runner_wire_batched.cpp:44): `PipelinedBucket` → tail-call the pipelined loop
(`:270-503`, the only body that reads `N` at `:285` and `D` at `:287`); any other value
of the two-valued enum → fall through to the **strict-barrier** body (`:47-268`).

The in-class default is `WireMode::StrictBarrier` (runner_wire_batched.hpp:23), with
`trees_per_thread = 1` (`:25`) and `max_inflight_msgs = 8` (`:24`). The cleanroom
contains **no caller** that mutates a `WireRunnerConfig`, so the default initializer is
the only authority on what runs.

**The composed model carries BOTH arms as a top-level discrete regime selected by `mode`:**

- **Strict-barrier (default).** `K = base = ceil(pool_batch/T)` (runner_wire_batched.cpp:57,
  runtime_config.hpp:12-15) — **no N factor**; `trees_per_thread` and `max_inflight_msgs`
  are **dead code** (absent from `:47-268`). The loop is a hard barrier: `any_parked()`
  → gather all parked slots into one message → one blocking `recv_batch` → resume all →
  repeat (`:224-251`). **Exactly one message in flight per thread (an implicit D=1
  barrier).** The N-axis and D-axis are **non-representable** here; a model that indexes
  any state by N or D in this regime is too permissive for the default config.
- **Pipelined (opt-in `PipelinedBucket`).** `K = N·base` (`:286`), `submitted[]`
  book-keeping (`:327`), the `<D` gate — but depth is still 1 (§0), so D is dead and the
  only live N effect is rows/message. **This is the only regime where the N-axis is live.**

Faithfulness clause: presenting the N-parameterized pipelined behavior as "production"
without conditioning on `mode == PipelinedBucket` admits N>1 executions the default-config
code cannot produce. The whole N-analysis below is **dormant unless `mode ==
PipelinedBucket`**. Confidence on the control flow: high (one `if` over a two-valued enum +
grep-verified parameter absence). Confidence that deployed production literally uses the
default vs. an out-of-tree launcher that flips it: medium (no launcher in the cleanroom).

---

## 2. The composed two-party protocol (assume-guarantee)

The boundary is a two-party protocol: **T producer threads, each one DEALER**
(wire_leaf_pool.hpp:35) ↔ **one single-threaded ROUTER server** (inference_server.py:153).
I state each side's RELY and GUARANTEE and discharge every RELY against the peer's code.

### 2.1 Socket options (determined set-vs-default; blocking depends on exactly these)

| Socket | Option | Set? | Value | Source | Effect |
|---|---|---|---|---|---|
| DEALER (producer) | `ZMQ_LINGER` | set | 0 | wire_leaf_pool.hpp:39-40 | teardown discards unsent, no flush block |
| DEALER | `ZMQ_RCVTIMEO` | set | `timeout_ms` | wire_leaf_pool.hpp:41 | bounds the sole blocking recv |
| DEALER | `ZMQ_SNDHWM`/`SNDTIMEO`/`RCVHWM` | **unset** | dflt 1000 / −1 / 1000 | — | send never HWM-blocks for D·T ≪ 1000 |
| ROUTER (server) | (none) | **unset** | — | inference_server.py:153-156 | see below |
| ROUTER | `ZMQ_ROUTER_MANDATORY` | **unset** | dflt 0 | inference_server.py:153-156 | **scatter silently DROPS** to a dead/unknown peer; never blocks, never errors on send |
| ROUTER | `ZMQ_SNDHWM`/`RCVHWM` | **unset** | dflt 1000 | — | never approached (per-peer depth ≤1, §0) |
| ROUTER | `ZMQ_LINGER` | only at close | 0 | inference_server.py:236 | drops queued frames at shutdown |

The only producer setsockopt calls in the whole tree are the two DEALER lines
(`:39-41`); the server sets nothing at bind. These two facts (MANDATORY-off silent-drop;
per-peer depth ≤1 so no HWM) are N-invariant.

### 2.2 Producer side (DEALER): RELY ← discharged by server GUARANTEE

- **R1 reply envelope.** Every reply is `[corr(8B), payload]`, ≥2 frames, leading frame
  exactly 8 bytes, payload a valid wire-v2 response. *Discharged:* ROUTER strips the
  ident it prepended; server sends `[ident, *envelope, resp]` (inference_server.py:200),
  DEALER receives `[corr, resp]`; checked at wire_leaf_pool.hpp:157-163. **GUARANTEE G2(srv).**
- **R2 corr echo & uniqueness.** The reply corr equals a corr this DEALER sent and not
  yet retired. *Discharged:* server treats `envelope=frames[1:-1]` as opaque and echoes
  it verbatim (inference_server.py:177,197-200); never invents a corr. Producer
  hard-errors on unknown corr (wire_leaf_pool.hpp:116-118). **G1(srv).**
- **R3 row-count match.** Reply prediction count == request row count B. *Discharged:*
  `run_microbatch` slices `v[off:off+n]` per request by its own row count
  (inference_server.py:64-72); padding rows are appended after real rows (`:59`) and
  never sliced into a reply. Producer hard-errors on mismatch (wire_leaf_pool.hpp:121-124).
  **G3(srv).**
- **R4 eventual reply / liveness.** Every sent corr is answered within `timeout_ms`.
  *Discharged conditionally:* server drains ≤ max_batch rows then runs one bounded forward
  and scatters before the next drain (serve_forever 219-225); S>0 finite. **Caveat (see
  §5 FAILED regime):** if S exceeds `timeout_ms`, or the server thread dies, R4 is broken
  and the producer fails loud.
- **R5 service-time character.** S>0, never an instant; ~row-invariant under the
  production padmax drain; 3-step then rising under the bench bucket drain. Producer
  depends on S only for liveness (R4) and coalescing statistics, not correctness. **G5(srv).**

### 2.3 Producer side (DEALER): GUARANTEE → discharges server RELY

- **G1(prod) well-formed request.** Two frames `[corr(8B)][payload]`, payload =
  9-byte header (ver=2, B≥1, in_dim≥1) + B·in_dim f32, `flat.size()==B·in_dim`.
  Enforced by `encode_request` refusing B==0 / in_dim==0 / size-mismatch
  (inference_wire.hpp:53-62) and `issue_one`'s `gathered.empty()→return` (`:444`).
- **G2(prod) corr uniqueness & monotone.** Every corr from one global atomic
  `corr_seq.fetch_add` shared across all T threads (wire_leaf_pool.hpp:84;
  runner_wire_batched.cpp:298): globally distinct, never reused (erased on reply, never
  re-emplaced). No two outstanding requests across all T threads share a corr.
- **G3(prod) depth bound.** At most 1 message per thread outstanding (§0); with default
  SNDHWM=1000 ≫ D·T the producer never blocks on send.
- **G4(prod) reply-causal pacing.** The producer never sends a request whose features
  depend on a not-yet-received reply: the slot holds exactly one fiber, blocked inside
  `predict` (fiber_leaf.hpp:24-29) until `resume_with` feeds `ch.value` (fiber_tree.hpp:58);
  leaf k+1 cannot be reached before reply k is resumed.
- **G5(prod) one consumer per socket.** Each move-only non-copyable `WireLeafPool`
  (wire_leaf_pool.hpp:55-69) is owned by exactly one worker thread.
- **G6(prod) eventual recv.** While `inflight_msgs>0` the producer is in a blocking
  `recv_batch` (runner_wire_batched.cpp:457-458); it does not send-only and stall.

### 2.4 Server side (ROUTER): RELY ← discharged by producer GUARANTEE

- **A1 frame shape** `[corr(8B), payload]` (← G1/G2 prod). `envelope=frames[1:-1]=[corr]`.
- **A2 payload validity** decodes under `decode_request`: ver==2, B≥1, in_dim≥1, exact
  length, finite (inference_wire.py:42-61) (← G1 prod via inference_wire.hpp). On
  violation → `_reject` drops with no reply (the open-peer path; see §5).
- **A3 D-capped pacing** (← G3 prod): per-peer in-flight depth ≤1, so the per-peer
  ROUTER receive-queue depth is ≤1 and HWM is never approached.
- **A4 reply consumed** (← G6 prod): the producer recvs promptly, so the ROUTER outbound
  pipe does not stay full; SNDHWM never approached.
- **A5 corr matching is the peer's job** (← producer owns the `inflight_` map,
  wire_leaf_pool.hpp:115-124): the server treats corr as opaque and never matches.
- **A6 uniform in_dim** (← capture-up): every frame in one run carries the same
  `in_dim = feat_dim = fb.dim()`, a single Environment constant shared across all T
  threads (runner_wire_batched.cpp:275,325,445; wire_leaf_pool.hpp:80;
  inference_wire.hpp:65-67). N only sets the slot count K, never the per-row width.
- **A7 width agreement** (← deployment, not a per-message act): `in_dim == W1.shape[0]`,
  built from the same Environment (stage_a_server.py:73-77 vs `fb.dim()`).

### 2.5 Server side (ROUTER): GUARANTEE → discharges producer RELY

- **G1(srv) envelope fidelity** (→ R1/R2): echoes the corr frame verbatim, routes to the
  originating ident (inference_server.py:197-200); never reorders frames within a message.
- **G2(srv) per-request cardinality & order** (→ R3): each ident receives exactly its
  `r_j` predictions sliced in request row order (`:64-72`); padding rows never returned.
- **G3(srv) one forward per drained group** (→ R5): production = exactly one forward per
  non-empty drain (`:198,219-225`); bench group = one per drain, bench leaf = one per
  drained MESSAGE (stage_a_server.py:57).
- **G4(srv) no spurious empty forward**: `_drain` returns ≥1 request before `_serve_batch`
  (`:224`); `run_microbatch` fail-loud-rejects an empty batch (`:44-45`).
- **G5(srv) malformed isolation** (→ partial R4): a request failing decode is rejected in
  isolation (`:181-183`) and excluded; it gets **no reply** (only a print, `:190`), so its
  corr stays in the producer's `inflight_` until that DEALER's RCVTIMEO fires. **G5 is
  "don't poison siblings", NOT "always reply."**
- **G6(srv) fail-loud terminal**: ragged in_dim (`:51-53`), too-few output rows
  (`:62-63`), or a reload raise propagate uncaught and **kill the server thread**
  (the EXCEPTIONAL_TERMINATION terminal; §5). No silent degrade.

### 2.6 Composition gaps (RELYs not fully discharged by a peer GUARANTEE)

1. **R4 (liveness) is conditional, not unconditional.** The producer's R4 ("answered
   within `timeout_ms`") is discharged by the server's bounded forward only while the
   server thread lives and S < `timeout_ms`. The server's G5 explicitly does **not**
   reply to a malformed request, and G6 **kills the thread** on a contract violation.
   Under either, R4 is breached and surfaces as the producer's RCVTIMEO. The gap is
   real and intentional (fail-loud), but it means **liveness is a joint property
   contingent on A2/A6/A7 holding**, not a unilateral server guarantee.
2. **`timeout_ms` is an unvalidated int** (runner_wire_batched.hpp:22, default 15000)
   handed straight to `ZMQ_RCVTIMEO` (wire_leaf_pool.hpp:41). For `timeout_ms < 0` the
   recv blocks forever (a **silent hang** under a dead peer — R4's backstop vanishes);
   for `== 0` it is instant EAGAIN → FAILED even with a reply pending. The "never a
   silent hang" guarantee holds **only for the default positive value**. N-independent.
3. **A6/A7 are RELYs the conforming C++ peer discharges by construction, but the server
   does not cross-check.** `decode_request` accepts any self-consistent `in_dim`
   (inference_wire.py:46-58) without comparing it to the net's W1 width. So a
   **non-conforming/foreign peer** writing a different `in_dim` header breaches A6 and
   trips the server-thread-death terminal (G6) — a gap that is closed for the two
   in-scope C++ peers and open for any other peer on the same ROUTER endpoint.
4. **The two-frame DEALER↔ROUTER pipe is FIFO; the producer's corr-keyed match is
   order-agnostic.** No gap, but note the slack: depth-1 means a thread never holds two
   outstanding messages, so its corr-keyed tolerance is **vacuous per-thread** (§4 DOF-R).

---

## 3. The composed timing model (source and sink nondeterminism, never collapsed)

Neither the source park interval nor the sink service time is collapsed to a constant or
an instant. The model leaves exactly the latitude the code leaves, constrained only by
causality.

### 3.1 SOURCE emission (per slot)

A slot parks at a leaf when `policy.run_search` calls `predict` (sets `ch.at_leaf=true`,
yields; fiber_leaf.hpp:24-28, surfaced as `running=ch.at_leaf`, fiber_tree.hpp:55,61). The
interval `δ_park(s,k) > 0` between a slot's resume and its next park is set by the
search's internal progress, which **this code does not fix** (run_search body absent,
fiber_tree.hpp:50). Modeled as positive bounded nondeterminism. Constraints the code
imposes:
- **C1 positivity.** No two parks of one slot are simultaneous (a coroutine runs work
  between leaves).
- **C2 reply-causality.** Leaf k cannot be reached, parked, or sent before reply k−1 has
  been resumed (`resume_with` at runner_wire_batched.cpp:467 feeds `ch.value` consumed
  before the search descends). This is the per-slot, depth-1 pacing.
- **C3 one-fiber.** A single reply yields **at most one** new eligible from that slot
  (one fiber/slot), possibly zero (`advance` may finalize the episode); a fresh episode
  via `fill` re-parks the **same slot index** on a new fiber — admitted (DOF-S churn),
  distinct from C3.
- The **first** leaf of a fresh episode (`fill`→`spawn_ply`, `:419`) is **not**
  reply-dependent — at prime time, with zero replies, all K first-leaf parks can be
  eligible at once (the maximal first wave).

Aggregate per thread: the offered process is the superposition of up to K = N·base
independent `δ_park` streams; the simultaneously-parked subset at any instant is a
nondeterministic 0..K-sized set — exactly what `issue_one` snapshots.

### 3.2 SINK service (per forward)

Service time S = wall time of `np.asarray(forward_fn(params, Xb, …))` (inference_server.py:61
— `forward_fn` returns a lazy JAX array; `np.asarray` blocks until XLA materializes it).
**Not an instant.** Positive nondeterministic duration S(shape) from a band determined by
the **compiled padded shape**. `forward_core` (forward.py:3-18) is a value-pure 2-layer
MLP (+ optional residual + value/logits heads), monotone non-decreasing in padded row
count, single-thread-pinned by default (config.py:5-6 `setdefault` — overridable).
Compilation discontinuity: one cached `jax.jit` (inference_server.py:20,33-34) but JAX
re-specializes per distinct input **shape**; first sight of a shape pays
`S_cold(R) ≫ S_warm(R)`.

- **Production / padmax (greedy drain):** `pad_to = max_batch` always (inference_server.py:198)
  ⇒ one steady compiled shape ⇒ S is **~independent of real row count** for B ≤ max_batch.
  This is a *derived* structural fact (`pad_to > B` guard, `:58`), not a modeling
  shortcut; the duration itself stays free in its band.
- **Bench bucket:** `pad_to = _bucket_for(real)` ∈ {64,256,512} (stage_a_server.py:32-37,
  63-64) ⇒ S a **3-step** function of real rows; for real > 512 `pad_to=512 ≤ real` so
  `run_microbatch` adds no padding and forwards at **width = real** with S(real).
- **Bench leaf:** one forward per drained MESSAGE (stage_a_server.py:57) ⇒ G serial
  forwards/drain, each padded independently; service per drain = sum over the drained
  messages.
- **Overshoot (both drains):** the last drained message can push B to max_batch + K − 1
  (cap at loop top, `:171`); `pad_to ≤ B` ⇒ unpadded **fresh shape** ⇒ a one-time
  `S_cold` per distinct overshoot width.

### 3.3 Causal constraints binding source to sink (the closed loop)

- **No reply before its forward:** `recv(corr) > fwd_end(corr) > fwd_start(corr) >
  drain_that_fed(corr)`; the server scatters (`:200`) strictly after the forward returns.
- **Single-thread serialization:** drain k+1 cannot begin until scatter k completes
  (serve_forever 219-225); forwards never overlap. **This is the coalescing engine.**
- **No recv during a forward:** while in FORWARD the server issues no recv, so producer
  frames accumulate in the ROUTER queue and are first observed at the **next** drain —
  the **self-batching feedback** (§4 DOF-F, §5 regime SELF_BATCH).
- **Depth-1 closed loop:** a submitted slot cannot re-emit until its reply clears
  `submitted` (runner_wire_batched.cpp:447,466), so the server's output timing feeds back
  into its input timing; the standing offered backlog is bounded by ≤ T·K = T·N·base rows.
- **Poll quantum** 100 ms (`_POLL_INTERVAL_MS`, inference_server.py:142,165) is a
  liveness/stop re-check bound only; `zmq_poll` wakes early on POLLIN, so a real arrival
  incurs **no** added latency. Not collapsed into per-request latency.

Z3 (confirmation, `synthesis_composed_check.py`): the **composed canonical regime** —
depth-1 per thread, FIFO-per-pipe (no intra-thread reorder), cross-thread coalescing
(threads B,C arriving mid-forward-0 coalesced into forward 1), forward-causality,
reply-causal pacing, **and** an inter-DEALER reorder (`recv B0 < recv A1`) — is **SAT**.
A concrete schedule: forward 0 [0,2] serves A alone; forward 1 [2,3] coalesces B,C;
`recv A0=3 < recv B0=recv C0=4`; `send A1=4 > recv A0`. The composed model is non-vacuous.

---

## 4. Degrees of freedom (composed), the constraint that removes each, and what becomes unrepresentable

Numbered DOFs span both sides. For each: side, what it admits, the design constraint
that removes it, and which regimes become unrepresentable.

| DOF | Side | Admits | Removing constraint | Made unrepresentable | Cost |
|---|---|---|---|---|---|
| **DOF-park** (δ_park nondeterminism) | producer | any C1–C3 eligible-set evolution, from all-K-eligible (lockstep) to one-at-a-time (staggered) | collapse δ_park to a constant (lockstep) | every staggered eligible-set, hence all B<K coalescing; falsely forces mean_rows_per_msg=K | erases the entire eligible-set dynamics & the only N-statistic |
| **DOF-B** (coalescing degree B/message) | producer | one message of 1..K rows; B = `|eligible-set|` at the `issue_one` instant | treat B as a free scheduler choice in [1,K] | "send only some eligible slots" — `issue_one` gathers EVERY ready slot (no subset, `:437-444`) | over-permits a non-producible message |
| **DOF-depth** (in-flight depth) | producer | **identically 1** (D dead, §0) | — (it is already pinned) | depth ≥2 / D-utilization / cross-message intra-thread reorder | adopting depth-1 forbids the fresh primaries' depth-toward-D regime, which is **unreachable** |
| **DOF-R** (reply arrival order) | both | across distinct DEALERs (T≥2): any interleaving; within ONE thread: **FIFO per pipe** (recv order == send order) | force global FIFO | cross-thread reorder (real: ROUTER drains/replies in arrival order across peers) | over-constrains |
| | | | admit intra-thread reorder | nothing real — depth-1 makes it **vacuous** (a thread never holds 2 outstanding); the only intra-thread freedom is the row scatter within one coalesced reply (fixed ascending-slot, wire_leaf_pool.hpp:126-130) | the fresh primaries' DOF-4/DOF-T6 admit a phantom |
| **DOF-drain** (drain boundary / batch B) | server | drained = contiguous ROUTER fair-queued arrival-prefix, 1 msg .. ≤T msgs, rows capped at max_batch + (last-msg − 1) | remove the `total_rows<max_batch` cap (`:171`) or the `Again` break (`:174`) | bounds: unbounded single-forward batches; or a blocking drain that waits for a 2nd message (the drain is strictly non-blocking, `:173`) | both directions unfaithful |
| **DOF-overshoot** (past max_batch) | server | B ∈ (max_batch, max_batch+K−1]; unwarmed fresh shape, no pad | check the cap per-row instead of loop-top | the genuine whole-last-message overshoot the code produces (`:171,184-185`) | over-constrains; loses the cold-compile sub-case |
| **DOF-S** (service-time duration) | server | any positive S in the shape band + one-time S_cold per unseen shape | collapse S to an instant | zero-latency replies, unbounded throughput, **no coalescing window** | erases the coalescing mechanism |
| | | | drop the production pad (S depends on real B) | a 1-row drain cheaper than a 256-row drain | misstates production service as B-dependent |
| **DOF-F** (self-batching feedback) | server | a long forward lets the queue grow ⇒ a fatter next drain (negative feedback) | let the server recv during a forward (multi-threaded server) | the self-batching coupling vanishes; batch size goes N-flat | unfaithful for this single-threaded server |
| **DOF-E** (E-policy {padmax,bucket}) | server-bench | forward shape ∈ {max_batch} vs {64,256,512}→unpadded | fix padmax (= production) | the bucket 3-step service ladder | bench-only latitude |
| **DOF-W** (wakeup {group,leaf}) | server-bench | 1 forward/drain vs G forwards/drain (G = #drained messages ≤T) | fix group (= production) | the leaf de-coalescing baseline | bench-only latitude |
| **DOF-reload** (weight reload) | server-prod | a forward under freshly-swapped weights at any drain boundary | force `current()` (= bench) | the production mid-stream weight swap (`:194`, RedisParamsSource) | bench (StaticParamsSource) genuinely cannot reload |
| **DOF-drop** (scatter silent drop) | server | a reply to a vanished/unknown ident dropped silently (no error) | set `ROUTER_MANDATORY=1` | the silent-drop execution becomes a raise out of SCATTER | the unset option is exactly what admits the drop |
| **DOF-pin** (XLA thread pin) | server | single-thread (default) or multi-thread (env override survives `setdefault`) | hard-set XLA single-thread | the multi-thread band the override path permits | an env fact, not an in-run latitude |
| **DOF-churn** (episode-boundary slot churn) | producer | slot oscillates ELIGIBLE↔OUTSTANDING↔ADVANCING then EMPTY in the tail | let one reply produce ≥2 new eligibles from one slot | impossible — one fiber/slot (`advance` drives to one park, `:389-397`) | over-permits |
| **DOF-tmo** (RCVTIMEO sign) | producer | bounded-then-FAILED (timeout_ms>0); unbounded silent hang (<0); instant EAGAIN (==0) | validate/clamp `timeout_ms>0` | the silent-hang and instant-EAGAIN outcomes the unvalidated int permits | the "never a silent hang" claim holds only for the default |

---

## 5. Global behavior — every qualitatively distinct regime (DERIVED stability, with N-dependence)

Stability is DERIVED, not assumed. "Reachable" is against the **conforming closed system**
(the two in-scope C++ peers); regimes reachable only via a breached RELY are marked.
All regimes below are within the **pipelined arm**; the strict-barrier arm has only the
depth-1 transport with no N-axis (§1) and its own STRICT-BARRIER regime.

### R0. STRICT-BARRIER (the default arm)
- **Conditions:** `mode == StrictBarrier`. K = base, N-invariant; one message in flight
  per thread (implicit D=1 barrier, runner_wire_batched.cpp:224-251).
- **Reachable:** yes (the default). **Stability:** self-reinforcing (a fixed loop).
- **Progress:** every parked-slot wave is submitted, recv'd, resumed; episodes drain to
  completion; FAILED on transport/redis error.
- **N-dependence:** **none** — N is dead code here. (This regime is the reason the whole
  N-axis is conditional.)

### R1. PIPELINED STEADY COALESCING (the modal regime; central latitude)
- **Conditions:** `mode == PipelinedBucket`; moderate load; per thread one fat message
  (B up to K = N·base rows) in flight; the single-threaded server coalesces, across
  threads, the ≤T messages that arrived during the prior forward.
- **Reachable:** yes. **Stability:** **self-reinforcing** — the negative-feedback
  batch-size fixed point (G1 capture-up, see §6). Larger offered load ⇒ fatter next drain
  ⇒ (production) same padded cost amortized over more rows ⇒ higher throughput ⇒ backlog
  drains. The feedback sign is **negative**.
- **Progress:** every drained request is answered in its cycle before the next drain
  (serve_forever 219-225); per-thread leaf k+1 paced after reply k.
- **N-dependence:** N fattens each of the ≤T cross-thread messages linearly (rows/msg ↑
  toward K = N·base); message COUNT stays ≤T (N-independent). mean_rows_per_msg ↑,
  total_msgs ↓. The server slides from arrival-bound toward service-bound as N grows,
  the batch-size fixed point climbs toward the cap, the saturation ceiling lifts to
  max_batch + N·base − 1. **Stability does NOT flip sign with N** — it stays
  self-reinforcing/negative-feedback for all N (G1: divergence UNSAT at every tested N).

### R2. PIPELINED SATURATION (service-bound, large N·T)
- **Conditions:** every drain fills to ~max_batch; throughput pinned at max_batch/S;
  excess queues. Sub-case **OVERSHOOT**: the last whole message pushes B to
  (max_batch, max_batch+K−1], an unwarmed shape ⇒ one-time S_cold.
- **Reachable:** yes once offered load suffices. The overshoot sub-case is reachable
  exactly when **T·K > max_batch** (with K>1) — i.e. as N grows past ~max_batch/(T·base);
  at N=1, K=base=8, T·K=32 ≪ 256/512, **unreachable** (the prior's R5). Z3
  (g1_greedy_stability_check.py): overshoot SAT exactly at the T·K>max_batch configs.
- **Stability:** self-reinforcing on batch size (bounded by the absorbing ceiling), but
  the overshoot **compile spike** is **one-time per distinct width**, non-recurring
  (single cached jax.jit, inference_server.py:20,33-34), ≤ K−1 distinct widths total, and
  **cannot lift the ceiling it is already under** ⇒ no positive-feedback divergence.
- **Progress:** unimpaired (each spike is a finite latency, fully amortized).
- **N-dependence:** overshoot magnitude = K−1 = N·base−1 grows **linearly**; the **set**
  of distinct overshoot widths is O(N), so a large-N run pays an O(N)-sized **bundle of
  one-time** compiles smeared across the run (not a startup spike, not a per-call tax).
  This is the **one place N directly enlarges the forward SHAPE**.

### R3. PIPELINED STARVED / ARRIVAL-BOUND (light load, small N)
- **Conditions:** poll often times out; single-row drains; high padding fraction
  (max_batch−1)/max_batch.
- **Reachable:** yes. **Stability:** transient — N manufactures concurrency that fills
  windows and pushes the system into R1/R2.
- **Progress:** unimpaired (every request answered, just under-batched).
- **N-dependence:** **less** reachable as N grows; escaped monotonically.

### R4. BENCH LEAF DE-COALESCING (bench-only, deliberately pessimal)
- **Conditions:** `wakeup == leaf`; a drain that coalesced G messages produces **G serial
  forwards**, each padded independently; re-pays the fixed per-forward overhead G times.
- **Reachable:** bench only (production has no W knob). **Stability:** self-reinforcing as
  the anti-batching baseline.
- **Progress:** unimpaired but throughput-degraded (rows/s falls vs group).
- **N-dependence:** forward COUNT per drain = G = #drained messages ≤ T, **N-INDEPENDENT
  in count** (the slot-count-vs-message-count fact; the greedy-drain model's "leaf
  multiplies forward count with N" was the audit-flagged conflation). The real N-effect
  is **rows per forward** (each leaf message carries up to K = N·base rows, so a leaf
  forward is itself a fat batch); leaf and group converge in rows/forward as N grows.
  Divergence (extra forwards + per-message padding) is largest at **small N / thin
  messages**.

### R5. BENCH BUCKET LADDER (bench-only)
- **Conditions:** `e_policy == bucket`; real rows climb 64→256→512→(unpadded >512).
- **Reachable:** bench only. **Stability:** self-reinforcing (monotone climb).
- **Progress:** unimpaired.
- **N-dependence:** N drives real rows up the {64,256,512} ladder, **stepping S up** and
  pad-fraction down at each threshold; past 512 the over-bucket forward is unpadded
  (width=real, S(real)) and unwarmed unless max_batch>512 covered it. Converges to
  production padmax only when bench max_batch = 512 = the top bucket.

### R6. FAILED — producer RCVTIMEO abort (liveness RELY R4 violated)
- **Conditions:** a sent corr unanswered within `timeout_ms`: a slow/huge forward, a
  dead server, or (capture-up) the server thread already dead. `zmq_msg_recv` → EAGAIN
  (wire_leaf_pool.hpp:147-150) → `set_error` → all T threads observe `failed` and abort.
- **Reachable:** yes (via the peer's latency or death). **Stability:** transient
  (terminates the run). For `timeout_ms<0` it is replaced by a **silent hang** (no FAILED
  transition); for `==0`, instant EAGAIN even with a reply pending (DOF-tmo).
- **Progress:** none past the abort (fail-loud, ADR-0002).
- **N-dependence:** the bound `timeout_ms` is N-invariant, but the **firing probability**
  rises with N (and T): larger N ⇒ fatter server batches / longer single-threaded forward
  queue ⇒ higher per-reply latency, pushing toward the fixed timeout. The producer-side
  mechanism is N-invariant; the latency that trips it grows with N via the server's
  serialization. **This is the only place a CORRECT execution can flip to an abort as N
  grows** — but it is a liveness/latency effect (the server batch-size feedback stays
  negative, §6), not a batch-size divergence.

### R7. EXCEPTIONAL_TERMINATION — uncaught server-thread death (RELY-gated; distinct from _reject)
- **Conditions:** an **uncaught** `ValueError` inside `run_microbatch` — ragged in_dim
  (`:51-53`), bad forward shape (`:62-63`), empty batch (`:44-45`) — or a width-mismatch
  matmul raise inside `forward_core` (`X@W1`, forward.py:5), or a reload raise
  (production only). The chain `run_microbatch ← _serve_batch ← serve_forever` is
  **unguarded** (no try/except), so the exception unwinds past `serve_forever` and **kills
  the server thread**.
- **Reachable:** **NO under the conforming closed system, at any N.** Ragged in_dim is
  blocked by A6 (uniform in_dim = feat_dim across all T threads, capture-up; Z3
  z3_in_dim_ragged.py: UNSAT under uniform in_dim). Bad forward shape is blocked by
  row-count algebra (B≥1, `pad_to≥B`, forward preserves rows). Width-mismatch is blocked
  by A7 (matched config). The empty-batch guard is redundant (`_serve_batch` only on
  truthy drained, `:224`). **Reachable only via a breached RELY** — a non-conforming/foreign
  peer writing a different in_dim (→ ragged, SAT only with a rogue in_dim) or a
  feat_dim-vs-W1 misconfiguration (→ matmul raise). Bench cannot hit the reload sub-case
  (StaticParamsSource, stage_a_server.py:56,78).
- **Stability:** terminal (the server stops forever). **Progress:** none.
- **Distinctness (critical for the model):** **`_reject` ≠ EXCEPTIONAL_TERMINATION.**
  `_reject` (decode failure) is **CAUGHT** (try/except around `decode_request`,
  inference_server.py:181-183), prints, continues, sends **no reply for one corr-id** —
  blast radius = the **single** peer that sent it → its one RCVTIMEO. The uncaught
  ValueError **kills the server thread** — blast radius = **ALL** peers → every pending/next
  recv → all-T RCVTIMEO. The fresh server models modeled only `_reject` and under-derived
  this terminal; the composed model keeps both, with the right blast radii.
- **N-dependence:** none in reachability (RELY-gated); N-independent.

---

## 6. The central differential answer: does any regime's stability flip sign as N grows?

**No structural regime flips sign with N.** The prior's N=1 conclusions are robust:

- The **self-correcting (negative-feedback) batch-size fixed point** (R1) **survives
  parametrically for all N** under the production greedy drain. Derived condition (G1
  capture-up, confirmed UNSAT-for-divergence at N ∈ {1,8,33,75,200}): every batch is
  bounded by an **absorbing ceiling** `min(T·K, max_batch + K − 1)`, reached in one cycle,
  via two S-independent clamps — (Clamp 1) the loop-top cap admits exactly one message
  past max_batch so B ≤ max_batch + K − 1; (Clamp 2) depth-1 backpressure means a thread
  in recv emits no rows, so total offered ≤ T·K independent of S — plus the production
  pad making the sub-cap forward SHAPE constant (no interior σ-feedback). The posited
  divergent loop (B↑⇒S↑⇒A↑⇒B↑ unbounded) is foreclosed link by link: B↑⇒S↑ only across
  the cap boundary; S↑⇒A↑ saturates at T·K (S-independent); A↑⇒B↑ clamped at
  max_batch+K−1. N moves the fixed point **up** and lifts the ceiling **linearly**; it
  **never flips the feedback sign**. The overshoot compile spike (R2) is one-time per
  width and cannot lift its own ceiling. Confidence: **high** on non-divergence/
  boundedness (S-independent); medium on the interior fixed point's exact location
  (S left free by the code).

- **No structural regime is destabilized by N.** Depth-1, cross-thread-only batching, the
  two socket-option facts, the single bounded blocking point, loud-abort, the
  self-clocking negative-feedback fixed point, and the EXCEPTIONAL_TERMINATION terminal
  all hold **verbatim** for all N. N is purely a **per-message ROW-COUNT / throughput-
  utilization** knob, not a structural one.

**The one sign-relevant N-effect is a liveness threshold, not a stability flip:** R6
(RCVTIMEO abort) becomes more *probable* as N grows because fatter batches lengthen the
single-threaded forward queue toward the fixed `timeout_ms` (via the peer's serialization,
RELY R4). A correct execution can be **replaced** by a loud abort as N·T grows past where
S approaches `timeout_ms`. This is a backpressure/latency effect, and it is itself
**further-negative feedback** (a timed-out producer stops offering ⇒ A shrinks ⇒ next
batch shrinks) — it terminates the run rather than diverging the batch size. So even this
"flip" is from correct to fail-loud, never to silent runaway.

Five places where N changes something the prior could not see at N=1 (all
throughput/utilization, none structural): (1) per-message payload scales linearly,
rows/msg ≤ K = N·base; (2) the overshoot regime crosses from unreachable to reachable at
T·K > max_batch, magnitude K−1 = N·base−1, the one place N enlarges the forward SHAPE and
can trip a (one-time-per-width) cold compile; (3) the self-clocking feedback **gain** rises
with N (fatter messages ⇒ longer S under bench bucket / wider accumulation ⇒ the fixed
point climbs faster, saturating sooner) while the sign stays negative; (4) the RCVTIMEO
firing **probability** rises with N; (5) the episode-tail ramp-down lengthens to ~N·base
episodes/thread (the declining active-slot pool in the tail).

**Falsifiable N-prediction (mean_rows_per_msg telemetry, capture-up).** Because depth==1
and each issue gathers all ready slots, the driver-emitted
`mean_rows_per_msg = total_leaves/total_msgs` (runner_wire_batched.cpp:449-450,494-500)
has the closed form `P_total / (Σ_t ⌈P_t/(N·base)⌉ + Σ_t r_t)`, monotone increasing in N,
envelope ≤ K = N·base, → K as plies-per-slot → ∞; r_t (0 ≤ r_t ≤ ℓ_max−1) is a straggler
term that grows ∝ K, so the fraction-of-envelope falls as N rises even as the absolute
mean climbs (Z3 check_mean_rows.py: N=1→8.00, N=2→15.38, N=4→28.57 at base=8). It is
**invariant** to D, to all park/reply timing, and to the drain variant — a concrete,
code-emitted, N-parametric check on the depth-1 fact. (Confidence medium: rests on the
depth==1-per-slot premise; the `run_search` body that fixes plies-per-slot is absent from
the cleanroom.)

---

## 7. Drain contrast (production greedy vs bench bucketed-group): which regimes each admits

**Shared spine.** `StageAServer` overrides **only** `_serve_batch` and **inherits**
`_drain`, `serve_forever`, `run_microbatch` (stage_a_server.py:39,54). So the greedy
**non-blocking** drain — loop-top row cap (`:171`), whole-last-message overshoot
(`:184-185`), ROUTER fair-queued pull — is **identical** in both. For BOTH: in-flight
depth is 1, batching is cross-thread, the drain is the same greedy non-blocking pull, the
socket options are the same, the self-batching negative feedback is the same.
**Consequence: every TRANSPORT regime (R0–R3, R6, R7) transfers to the bench unchanged.**
The two drains differ **only in SHAPING** (`_serve_batch`):

| Axis | Production greedy | Bench (StageAServer) |
|---|---|---|
| Forwards/drain | **always 1** (`:198`, all drained → one forward) | group: 1; **leaf**: 1 per drained MESSAGE (`:57`), count ≤T, N-independent |
| Forward shape / S | **always pad max_batch** (`:198`) ⇒ one compiled shape ⇒ S N-invariant until overshoot | padmax: identical; **bucket**: {64,256,512}→unpadded>512 ⇒ S **steps up with N** (`:32-37,63-64`) |
| Weight reload | `poll()` each batch ⇒ mid-stream swap (`:194`) | `current()` only ⇒ **never reloads** (StaticParamsSource, `:56,78`) |
| max_batch default | 256 | 512 |
| Overshoot threshold | T·K > 256 | T·K > 512 |
| Warmup | caller-supplied; serve_forever does not warm ⇒ cold-compile tail possible | `build()` warms {64,256,512,max_batch} (`:82`) ⇒ no cold-compile for those shapes |

**Regime admission.** The production drain can **only coalesce-up**: one fixed-shape
forward per drain, S flat in N until overshoot, with no anti-coalescing and no
σ-stepping latitude — it admits R0–R3, R6, R7, and **DOF-reload** (its own latitude).
The bench drain admits **all** of the production transport regimes **plus** two
shaping-only regimes the production drain cannot produce: **R4 (leaf de-coalescing**, the
deliberately pessimal anti-batching baseline**)** and **R5 (bucket ladder σ-stepping)**;
and it **subtracts** DOF-reload (StaticParamsSource cannot reload).

**Is any pathology drain-specific or shared?** Every genuine pathology is **shared**, not
drain-specific:
- R6 (RCVTIMEO abort) — shared; both drains serialize through one thread, and the bench's
  larger default max_batch (512) only shifts the overshoot/latency thresholds, not the
  mechanism.
- R7 (EXCEPTIONAL_TERMINATION) — shared spine (`run_microbatch` is inherited), and
  unreachable in BOTH under a conforming peer; the **only** drain-specific difference is
  that the bench cannot hit the **reload-raise sub-case** (StaticParamsSource), so the
  production drain has one extra (still RELY-gated, production-only) entry into R7.
- The overshoot cold-compile — shared mechanism (inherited `run_microbatch` pad guard),
  thresholds differ by max_batch only.
- The auditor-flagged **phantom send-block** (a SCATTER that blocks under HWM) is **not a
  real pathology in either drain**: ROUTER_MANDATORY=0 ⇒ scatter silently drops, never
  blocks (the greedy-drain model's over-permission, corrected here).

R4 (leaf de-coalescing) and R5 (bucket ladder) are **bench-only by construction**, but
they are **performance regimes, not pathologies** — they degrade throughput by design (the
anti-batching baseline and the σ-step ladder) and never break progress or liveness.

**One-line contrast.** The production greedy drain can only coalesce-up (one fixed-shape
forward per drain, σ flat in N until the soft-cap overshoot); the bench drain shares that
identical transport and can additionally DE-coalesce (leaf) and STEP σ with load (bucket),
converging to production only when bench max_batch=512 equals the top bucket — so the
prior's transport conclusions transfer to the bench unchanged, and only the shaping/σ
conclusions are production-specific and must be re-derived per E-policy/wakeup.

---

## 8. Minimal fidelity requirements

A model of this boundary is faithful iff it preserves **all** of:

1. **Depth-1 per thread, parametrically (D dead on the pipelined path).** Any model that
   makes in-flight message depth a live, N-growing, or D-bounded quantity is both too
   permissive (admits unreachable depth≥2) and too constrained (forbids the universal
   depth-1 executions at large N). The N effect is **rows/message ≤ K=N·base only**.
2. **Mode conditioning.** The N-axis exists only under `mode==PipelinedBucket`; the
   default strict-barrier arm is N-invariant (K=base). A model that presents the
   N-parameterized behavior as unconditional "production" is too permissive for the default.
3. **Cross-thread-only batching with FIFO-per-pipe.** Reorder is admitted **across**
   DEALERs (T≥2), forbidden **within** one thread (FIFO per pipe; intra-thread reorder is
   vacuous under depth-1). The only intra-thread reply-ordering freedom is the fixed
   ascending-slot scatter within one coalesced reply.
4. **Timing as bounded nondeterminism, never collapsed.** δ_park>0 (free), S>0 (free in a
   shape-determined band). The production padmax SHAPE-constancy is a *derived structural*
   fact (pad-to-max_batch), not a collapse of S to a constant; the bench bucket is a
   3-step-then-rising σ; both leave S's duration free.
5. **The two socket-option facts.** DEALER sets only LINGER=0 + RCVTIMEO=timeout_ms;
   ROUTER sets nothing ⇒ MANDATORY=0 ⇒ scatter silently DROPS (never blocks/errors on
   send), HWM never approached (depth-1).
6. **RCVTIMEO is the sole producer blocking point, with sign-dependent semantics.**
   `timeout_ms>0` ⇒ bounded-then-FAILED; `<0` ⇒ silent hang; `==0` ⇒ instant EAGAIN. The
   "never a silent hang" claim is conditional on the default positive value.
7. **Soft-cap overshoot.** Cap tested on the pre-request total ⇒ one whole message crosses
   max_batch ⇒ B ≤ max_batch+K−1, an unpadded fresh shape ⇒ one-time-per-width cold
   compile; reachable once T·K>max_batch (i.e. as N grows).
8. **The self-clocking negative-feedback fixed point, bounded for all N.** Batch size is
   bounded by the absorbing ceiling `min(T·K, max_batch+K−1)`; the feedback sign is
   negative for all N; it never diverges.
9. **Two distinct fail-loud terminals with distinct blast radii.** `_reject` (caught,
   drop-one-reply, one-peer RCVTIMEO) vs EXCEPTIONAL_TERMINATION (uncaught, server thread
   death, all-peer RCVTIMEO). Both must be representable; EXCEPTIONAL_TERMINATION is
   RELY-gated (unreachable under a conforming peer, reachable in the open system).
10. **Loud bilateral abort, never silent wrong-slot apply** — corr-id matched, B-exact
    checked, globally-unique corr (wire_leaf_pool.hpp:84,116-124).

---

## 9. Independence note (how this was produced)

The boundary was first modeled **blind** from the cleanroom: two producer-side models
(pacing-centric and transport-centric) and two server-side models (greedy-drain and
bucket-drain), each derived forward from the code with its own Z3 confirmation, with no
access to the prior N=1 model. Two adversarial audits (too-permissive and too-constrained)
were then run against the fresh corpus, **independently re-deriving** two facts the prior
held: that single-thread reply reorder is unreachable (FIFO-per-pipe;
verify_producer_fifo_check.py SAT-as-written, UNSAT-with-FIFO), and that the server's
phantom send-block is impossible (ROUTER_MANDATORY=0 silent-drop).

The **reconciliation against the prior N=1 model** added the decisive correction: the two
fresh PRIMARY producer models had **regressed** on the prior's load-bearing structural fact
— per-thread in-flight depth — by manufacturing an N-dependence (depth climbing toward D as
N passes D/base) that the code forecloses. The reconciliation showed (and bounded Z3
re-confirmed at K∈{2,8,24}) that depth is identically 1 for **all** N, that the fresh
audits — not the fresh primaries — were the version to adopt, and that the prior's entire
suite of N=1 structural conclusions (cross-thread-only batching, the socket-option facts,
the single bounded blocking point, loud-abort, the self-clocking negative-feedback fixed
point, the EXCEPTIONAL_TERMINATION terminal) survives **parametrically** for all N — with N
re-cast as a pure per-message row-count knob, the one axis the prior could not see at N=1.

Five **targeted capture-up derivations** then closed the reconciliation's open questions
with code-grounded proofs + bounded Z3: (G1) the greedy fixed point is **unconditionally**
non-divergent for all N (two S-independent clamps); (CF-10) the default driver is
strict-barrier and the N-axis is opt-in; the EXCEPTIONAL_TERMINATION terminal is distinct
from `_reject` and unreachable under a conforming peer; over-cap compiles are one-time
**per distinct width** with an O(N)-sized width set; and ROUTER fair-queue + depth-1 makes
starvation impossible at all N (deferral bounded by ≤T drains). This synthesis composes the
audit-surviving version of each element, discharges every RELY against the peer's
GUARANTEE (naming the four composition gaps in §2.6), and re-grounded all 14 cleanroom
files end to end — including the **strict-barrier driver every fresh model dropped**.

Bounded confirmations re-run for this synthesis (confirmation only, never the source of
trust): `inflight_le1_check.py` (UNSAT inflight==2), `reconcile_depth_parametric.py`
(UNSAT at K∈{2,8,24}), `g1_greedy_stability_check.py` (divergence UNSAT at N∈{1,8,33,75,200};
overshoot SAT exactly at T·K>max_batch), `z3_fairqueue_starvation_N.py` (starvation UNSAT,
adversarial control SAT), `z3_in_dim_ragged.py` (ragged UNSAT conforming, SAT rogue), and a
new `synthesis_composed_check.py` (the composed canonical regime — depth-1 + FIFO-per-pipe
+ cross-thread coalescing + forward-causality + reply-causal pacing + inter-DEALER reorder
— SAT). All pass.

---

## 10. Open questions

1. **Which arm does deployed production actually run?** The cleanroom shows the
   strict-barrier default and no launcher that flips `mode`. If an out-of-tree launcher
   sets `PipelinedBucket`, the N-axis is live; otherwise it is dead. The composed model
   carries both, but the *weight* of the N-analysis depends on a file outside my world.
   (Confidence on the conditional: high; on the deployed value: medium.)
2. **The interior fixed point's exact location and approach (monotone vs oscillatory).**
   The code leaves S(shape) as bounded nondeterminism (`forward_core` is value-pure;
   wall-clock duration is free), so boundedness and non-divergence are pinned but the
   precise fixed-point batch size and the trajectory's approach are not, absent an S model
   the synthesis deliberately does not assume.
3. **The plies-per-slot distribution behind `mean_rows_per_msg`.** The closed-form
   N-prediction (§6) rests on the depth==1-per-slot premise and on `P_t` (plies/thread),
   set by `policy.run_search` whose body is absent from the cleanroom (fiber_tree.hpp:50).
   The prediction is falsifiable against the driver telemetry but its inputs are
   not fully observable here.
4. **ROUTER fair-queue at scale × N.** Starvation is proven impossible (≤T-drain
   deferral) under the libzmq 4.3.5 fair-queue, which is **external** to the cleanroom.
   chocofarm provably does not disable it (no overriding setsockopt), but the residual
   risk lives entirely in that library contract; a libzmq version that weakened the
   per-pipe round-robin would reopen the question.
5. **The XLA single-thread pin under host contention.** `config.py:5-6` uses
   `os.environ.setdefault`, so an externally-set `XLA_FLAGS`/`OMP_NUM_THREADS` survives
   and admits a multi-thread forward with a different S band contending for the 4 vCPUs.
   The deployment fixes it, but the override path the code permits is not characterized
   quantitatively.
6. **RedisClient write blocking as a hidden source-timing input.** A finished slot's
   successor becomes `is_ready` only after `finalize_and_write`'s redis write
   (runner_wire_batched.cpp:359); a redis stall delays the next leaf offered. Judged off
   the leaf-eval transport boundary, but if the boundary is taken to include "time until
   the next leaf is offered," this is an unmodeled source-timing input.
