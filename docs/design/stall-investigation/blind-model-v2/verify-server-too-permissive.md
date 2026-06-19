# Adversarial fidelity verification — Python server side (TOO-PERMISSIVE lens)

Role: adversarial verifier for the Python single-threaded ROUTER inference server
(production greedy drain `chocofarm/az/inference_server.py` + bench bucketed/group/leaf
drain `cpp/stage_a/stage_a_server.py`).

Lens: **too permissive** — find executions a model ADMITS that the code cannot produce
(unjustified free choice, RELY stronger than the peer guarantees, ordering/timing the
protocol or physics forbids, N-dependence that overstates what grows).

All file:line refs are into `/home/bork/w/vdc/chocobo/runs/leaf-eval-model-2/cleanroom`.
I read every file in that tree end to end before judging (the 15 files listed by the
models, all present and read).

---

## Ground facts re-derived from the code (used to adjudicate)

- **D-cap / per-peer in-flight bound.** Each producer thread runs
  `while (inflight_msgs < D && issue_one()) {}` at prime (runner_wire_batched.cpp:456)
  and after each reply (runner_wire_batched.cpp:474). `issue_one` does exactly one
  `submit_batch` = one wire message and `++inflight_msgs` (runner_wire_batched.cpp:445-448);
  `inflight_msgs` is decremented only on a received reply
  (runner_wire_batched.cpp:460). Therefore a thread holds **at most D messages**
  outstanding at any instant, and the number of request messages sitting in the
  ROUTER receive queue from that peer is ≤ inflight_msgs ≤ D.
  D = `max_inflight_msgs`, default 8 (runner_wire_batched.hpp:24); T = `pool_threads`,
  default 4 (runner_wire_batched.hpp:20). **D and T do not depend on N.**

- **Slot ceiling vs message ceiling.** K = N·base, base = `fibers_per_thread()`
  = ceil(pool_batch/T) (runner_wire_batched.cpp:286; runtime_config.hpp:12-15). A single
  message carries up to K rows (issue_one gathers all `is_ready` slots over s∈[0,K),
  runner_wire_batched.cpp:437-443). So **rows/message scales with N (≤K)** but the
  **count of distinct messages queued at the ROUTER is ≤ T·D, which is N-independent.**
  A single `_drain` pulls a contiguous FIFO prefix of the queue and so captures
  ≤ T·D messages (inference_server.py:171-185).

- **ZeroMQ ROUTER scatter with ROUTER_MANDATORY unset (= 0, OS default; the code never
  sets it, inference_server.py:153-156).** Documented libzmq semantics: a ROUTER socket
  that has reached SNDHWM (default 1000, also never set) for a peer's outbound pipe
  **silently DROPS** the message when MANDATORY=0; it **blocks (or returns EAGAIN) only
  when MANDATORY=1**. So `send_multipart` (inference_server.py:200 / stage_a_server.py:70)
  on this socket **cannot block on a full peer pipe** — it drops.

These three facts decide the findings below.

---

## MODEL 1 — `model-server-greedy-drain.md`

### Finding 1.1 (MAJOR, too-permissive timing/blocking) — DOF-6 sub-case (a) "scatter send blocks under SNDHWM"

- model_element: DOF-6 "Send blocking / silent drop on scatter", behavior (a)
  "block if a peer outbound pipe is full"; and the timing/DOF claim that an in-flight
  scatter can stall on HWM.
- expected_code_ref: inference_server.py:200 (send_multipart), :153-156 (ROUTER created,
  ROUTER_MANDATORY/SNDHWM never set → both at OS default 0 / 1000).
- defect_type: timing-over-permissive / admits-too-much.
- The code's ROUTER with MANDATORY=0 does **not** block a send when a peer's outbound
  pipe is full; it silently drops (documented libzmq behavior; MANDATORY-on is the only
  mode that blocks/EAGAINs). Model 1 admits a "transient send-block under HWM" execution
  the socket cannot produce. (Models 2 and 3 state this correctly: "send is blocking but
  cannot HWM-block under MANDATORY-off"; Model 1 is the outlier.)
- correction: Remove sub-case (a) "block if a peer outbound pipe is full." Under
  MANDATORY=0 the only HWM outcome is the silent drop already captured in sub-case (b).
  The server SCATTER state never blocks and never errors on send; it either delivers or
  silently drops. Keep (b); delete (a).

### Finding 1.2 (MAJOR, wrong-N-dependence / too-permissive) — DOF-8 leaf "forwards-per-drain rises with N", and E4 "more messages per drain" with N

- model_element: DOF-8 n_dependence ("forwards-per-drain = message multiplicity, which
  rises with N ... leaf multiplies forward count with N"), and E4 n_dependence
  ("larger N → fatter/more messages per drain → larger G → more serial forwards").
- expected_code_ref: runner_wire_batched.cpp:456,474 (D-cap bounds messages/thread by D);
  :286 (K=N·base scales rows/message, not message count); inference_server.py:171-185
  (one drain pulls ≤ T·D messages).
- defect_type: wrong-n-dependence.
- Under leaf the forward count per drain = |drained| = number of messages drained, which
  is bounded by the messages queued at the ROUTER = ≤ T·D, an **N-independent** ceiling.
  Growing N fattens each message (up to K rows) but does **not** raise the message count;
  if anything fatter messages mean fewer messages are needed to carry the same slot
  supply. So "leaf multiplies forward count with N" overstates: the per-drain forward
  count saturates at T·D regardless of N. The genuine N-effect under leaf is that each
  forward gets FATTER / climbs the bucket ladder, not that there are MORE forwards.
  (Models 2 and 3 get this right: "leaf does NOT defeat batching for large N — each
  message is already fat"; "g saturates at ~T·D".)
- correction: State the leaf forward count per drain as bounded by T·D (N-independent
  ceiling); the N-effect under leaf is rising rows-per-forward (bucket climb), not rising
  forward count. The "inverse of production coalescing" framing holds only at small
  N / thin messages, not asymptotically in N.

### Finding 1.3 (MINOR, too-permissive N-claim) — DOF-1 drain "saturation ceiling T*K"

- model_element: DOF-1 n_dependence "saturation ceiling T*K".
- expected_code_ref: runner_wire_batched.cpp:456,474 (per-thread ≤ D messages);
  inference_server.py:171 (drain row-capped at max_batch, +K-1 overshoot).
- defect_type: wrong-n-dependence.
- DOF-1's own description correctly bounds drain size in messages by T·D, but the
  n_dependence line then writes the ceiling as **T·K** (= T·N·base, the total *slot*
  count). As a message-count ceiling that is wrong (should be T·D); as a row ceiling it
  is also wrong (rows/drain are capped at max_batch + K − 1, not T·K). T·K conflates the
  N-linear slot population with the N-independent queue depth — the same conflation as
  Finding 1.2.
- correction: Drain message-count ceiling = T·D (N-independent); drain row ceiling =
  max_batch + K − 1 (per inference_server.py:171,184-185). Drop the "T·K" ceiling.

### Trace check (Model 1)
The representative executions E1–E5 are each genuinely enabled by the code given the
ground facts above, with one caveat already covered by Finding 1.2: E4's narrative
("more messages per drain" as N grows) leans on the over-stated N-dependence, but the
trace's *concrete* steps (drain 3 messages → 3 leaf forwards, stage_a_server.py:57-65)
are admissible. E1's coalescing (lone cycle-0 request; cycle-1 batch from streams that
emit during cycle-0 service; single-thread serialization, no recv during forward,
inference_server.py:61) is admissible. E2 saturation + overshoot B∈(max_batch,max_batch+K-1]
is admissible (loop-top cap, inference_server.py:171; no pad when pad_to≤B, :58). E3
idle-poll is admissible. E5 bucket ladder + over-512 unpadded forward is admissible
(_bucket_for clamps to 512, stage_a_server.py:34-37; run_microbatch pad guard false at
:58). traces_admissible = true (the defects are in DOF n-dependence prose, not in the
step sequences).

### Timing fidelity (Model 1)
Source inter-emission and sink service are kept as bounded positive nondeterminism, not
collapsed. Sink service correctly tied to the COMPILED padded shape (constant in
production, stepped in bench bucket). The one timing over-permission is the phantom
HWM-block in SCATTER (Finding 1.1); otherwise faithful.

### N-dependence check (Model 1)
N enters only through the peer via K = N·base (runner_wire_batched.cpp:286): correct.
The fattening of messages and the slide arrival-bound→service-bound are correctly
N-derived. Two N-claims overstate growth by conflating the N-linear slot count (T·K)
with the N-independent message/queue count (T·D): the leaf forward-count claim
(Finding 1.2) and the DOF-1 "T·K" ceiling (Finding 1.3).

### Fidelity verdict (Model 1): too-permissive (repairable)
One major blocking-semantics over-permission (1.1) and one major N-overstatement (1.2),
plus a minor related N-claim (1.3). The state machine, guards, and traces are otherwise
faithful.

---

## MODEL 2 — `model-server-bucket-drain.md`

### Trace check (Model 2)
All five representative executions are genuinely enabled. The over-M single-message
overshoot (base=200, N=3 → K=600>M=512; whole 600-row message admitted by the loop-top
test at inference_server.py:171; _bucket_for(600)=512 via stage_a_server.py:34-37;
pad_to=512≤600 → no pad at inference_server.py:58 → forward width=600) is a genuine
code-permitted execution — confirmed reachable because a priming `issue_one` can gather
all K ready slots into one message (runner_wire_batched.cpp:437-444) before any of them
become `submitted`. The self-batching accumulation, bucket ladder, leaf fan-out, and
production padmax+reload traces are all admissible. traces_admissible = true.

### Findings (Model 2)
No too-permissive defect found with a load-bearing code line. Model 2 explicitly and
correctly: (i) bounds queued message count by T·D, N-independent (DOF-1, DOF-2 "g
saturates at ~T·D"); (ii) states "send is blocking but cannot HWM-block under
MANDATORY-off" (stance), avoiding Model 1's phantom block; (iii) flags in its self-audit
that DOF-1 must be a contiguous FIFO arrival-prefix, not an arbitrary subset
(inference_server.py:171-185), pre-empting the only subset-skip over-permission;
(iv) flags the MANDATORY-off silent drop (G1 self-audit) rather than asserting guaranteed
delivery. The only items in its self-audit (unbounded-S, arity-1 RELY, warmup set) are
correctly characterized as RELY-gated or per-run-fixed, not in-run latitudes.
[defect_type: none.]

### Timing fidelity (Model 2)
Faithful. tau_park and S(r) bounded positive nondeterminism; S monotone non-decreasing in
width; affine band reasonable for the fixed-width MLP (forward.py:5-17); compilation
modeled as a one-time per-shape penalty with warmup pre-paying {64,256,512,M}
(stage_a_server.py:82); 100ms poll quantum is a literal, not a collapsed service time.
Nothing collapsed to an instant.

### N-dependence check (Model 2)
Faithfully derived. The single load-bearing N-fact — offered batch WIDTH grows linearly
because one message carries up to K=N·base rows (runner_wire_batched.cpp:286,437-451),
while message COUNT stays ≤ T·D independent of N — is stated correctly and is exactly the
fact Model 1 gets wrong. Bucket-ladder climb-then-clamp and the over-M threshold
(N > M/base) are correctly N-derived.

### Fidelity verdict (Model 2): faithful
No admit-too-much execution found. (Lens is too-permissive; a separate too-constrained
pass would re-examine the affine-S band and the FIFO-prefix assertion, neither of which
over-permits.)

---

## MODEL 3 — `model-server-transport.md`

### Trace check (Model 3)
Executions A–F are each genuinely enabled. Exec B/C (multi-peer coalescing and the
self-batching feedback where a long forward lets the queue grow so the next drain is
fatter) rest on single-thread mutual exclusion + the libzmq IO thread accumulating frames
during the forward (inference_server.py:61; no recv during FORWARD) — admissible. Exec D
(leaf pessimal: g singleton groups → g forwards) is admissible with the caveat that each
"singleton group" is one MESSAGE (up to K rows), which Model 3 states correctly. Exec E
(REJECT) and Exec F (stop during poll) are admissible and correctly RELY-/shutdown-gated.
traces_admissible = true.

### Finding 3.1 (MINOR, self-flagged over-permission) — G4 "no ordering guarantee across requests"

- model_element: G4 "no ordering guarantee across requests: replies scatter in
  ROUTER-arrival order, NOT request emission order"; self-audit OP-4.
- expected_code_ref: inference_server.py:197-200 (replies scattered in drained order,
  deterministic given the queue); run_microbatch:66-72 (within a request, per-row order
  is fixed); wire_leaf_pool.hpp:126-129 (peer maps reply[i]→slots[i], so WITHIN a message
  order is load-bearing and IS guaranteed by G2).
- defect_type: missing-blocking-semantics / admits-too-much (benign).
- G4 as written is slightly more permissive than the code: the server's reply order is
  deterministic in drained order, and the within-message per-row order is REQUIRED by the
  peer and guaranteed by G2. Model 3 already flags this in self-audit OP-4 and notes the
  peer matches by corr so cross-message order is immaterial. The over-permission is
  therefore harmless (no peer relies on cross-message order; within-message order is
  separately guaranteed). Kept as MINOR and self-disclosed.
- correction: Scope G4 to "cross-MESSAGE reply order is unconstrained (peer keys by
  corr); WITHIN a message, per-row order is preserved (G2)." This removes the appearance
  that the server could permute rows inside a single reply (which it cannot, :66-72).

### Findings — no fatal/major over-permission
Model 3 correctly: bounds per-peer ROUTER queue depth ≤ D, N-independent (RELY-R3,
runner_wire_batched.cpp:456,474); states scatter "cannot HWM-block or error under
MANDATORY-off; may silently drop to a dead peer" (SCATTER state) — i.e. it does NOT make
Model 1's phantom-block error; correctly derives no HWM is approached for any N
(SNDHWM/RCVHWM=1000 default vs depth ≤ T·D); and gates REJECT and MANDATORY-off drop as
RELY-only.

### Timing fidelity (Model 3)
Faithful. Source emission an external positive-interval point process with finite
positive wire delay dwire (loopback); sink service S(R,warm?) affine + one-time cold
compile per unseen shape; padmax → N-invariant shape; bucket → N-stepped; 100ms a literal
poll bound, not added per-request latency. Single-thread mutual exclusion correctly named
as the coalescing engine. Nothing collapsed to an instant.

### N-dependence check (Model 3)
Faithfully derived and the most explicit of the three: server is structurally N-blind,
N enters only through the wire; per-peer queue depth ≤ D and message count ≤ T·D are
N-independent; the only place N enters the forward SHAPE directly is the overshoot
K−1 = N·ceil(pool_batch/T)−1 (DOF-7), correctly N-linear. Matches the code.

### Fidelity verdict (Model 3): faithful (one benign, self-flagged minor)
No fatal or major admit-too-much. The single minor item (3.1) is already disclosed in the
model's own self-audit and is harmless under the peer's corr-keyed matching.

---

## Cross-model note

The decisive discriminator across the three models is the **ROUTER receive-queue message
count** versus the **slot population**. The D-cap (runner_wire_batched.cpp:456,474) bounds
per-peer in-flight messages by D, so the messages a single `_drain` can pull
(inference_server.py:171-185) are bounded by T·D — an **N-independent** ceiling — while
rows-per-message scale with K = N·base (runner_wire_batched.cpp:286,437-443). Models 2 and
3 encode this correctly ("g saturates at ~T·D"; "message COUNT bounded by T·D independent
of N"). **Model 1 conflates the two**, claiming the leaf forward count per drain "rises
with / multiplies with N" (Finding 1.2) and a drain "saturation ceiling T·K"
(Finding 1.3); both overstate N-growth by substituting the N-linear slot count for the
N-independent queue depth.

The second discriminator is **ROUTER scatter blocking semantics**. With
ROUTER_MANDATORY=0 and SNDHWM=1000 (both OS defaults, never set — inference_server.py:153-156),
a full peer pipe causes a **silent drop, not a block**. Models 2 and 3 state this
correctly; **Model 1 alone** admits a phantom "send blocks under HWM" execution
(Finding 1.1) the socket cannot produce.

Net: Model 1 is **too-permissive** (one major blocking over-permission, one major
N-overstatement, one related minor). Models 2 and 3 are **faithful** under the
too-permissive lens (Model 3 carries one benign, self-flagged minor on cross-message reply
ordering). No model collapses source or sink timing to constants/instants; all three keep
service time tied to the compiled shape and tau_park as positive nondeterminism.

(Confirmation: the two load-bearing facts behind the findings — the T·D message ceiling
and the MANDATORY-off drop-not-block semantics — are a counting invariant over the D-cap
and a documented libzmq socket-option behavior, respectively, both decidable by
inspection; no Z3 scheduling search adds trust beyond the code-line derivation, so per the
"if in doubt, deliver the rigorous derivation" instruction none was run.)
