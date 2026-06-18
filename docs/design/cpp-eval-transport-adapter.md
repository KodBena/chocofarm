<!--
docs/design/cpp-eval-transport-adapter.md
Purpose: the design point + validation plan for the C++->JAX leaf-evaluation transport adapter (the
  predict PORT and the wire between the C++ Gumbel search and the in-process JAX InferenceServer). Output
  of a bounded prior-art survey workflow (transport-adapter-lit-survey, run wf_66209f19-914): 6 domain
  surveys (self-play eval architectures, production batching servers, parallel MCTS, producer->accelerator
  transport, centralized-inference RL, CPU small-MLP batching) -> a completeness critic -> this synthesis,
  grounded in the session's MEASURED per-core ceilings. Scope: the transport/batching discipline ONLY; no
  search-algorithm change. PROPOSAL — not yet validated; Stage A/B (sec 4) gate any lift into the boundary.
Public Domain (The Unlicense).
-->

# Design Note: the C++->JAX leaf-evaluation transport adapter

Status: **proposal**, pending the Stage A/B validation in §4. Literature claims are flagged PREDICTED;
the per-core ceilings are MEASURED this session (the serve/gen roofline + the S=1/S=K adapter dead-ends).

## 0. The problem in one paragraph

Three knobs the current code conflates into one: **S** = leaves per wire message (send-batch), **D** =
outstanding in-flight messages (overlap depth), **E** = the server's XLA forward shape (eval-batch). Our
two existing adapters sit at the two degenerate corners — strict-barrier (`S=K, D=1`, serialization-bound,
~49 dps) and greedy-async (`S=1, D=K`, frame-overhead-bound, ~37 dps, flat in D). The literature is
unanimous that the throughput design point is the third corner neither occupies: **S>1 AND D>1 with E
decoupled and chosen server-side from whatever has arrived, run at its true row count (no pad-to-max)** —
a server-side queue-drain batched evaluator (the obvious, standard pattern), generalized across a wire à la
SEED RL, made pad-safe by Triton-style shape bucketing.

## 1. The recommended design

### 1a. Set-points for S, D, E

- **E is decoupled and server-chosen.** The server runs a pop-up-to-N coalescing drain: each forward
  takes whatever leaves accumulated up to a cap, snapped **up to the nearest of a small set of
  AOT-compiled bucket shapes** — never pad-to-max. This is the single change that stops partial drains
  paying the ~2x max-pad that killed the half-batch attempt (#1).
- **The bucket set is FEW and LARGE.** Our 241->256->65 MLP is a trivial GEMM; the forward is dominated by
  fixed per-call dispatch overhead (the "framework tax"), NOT matmul FLOPs — which is why our serve curve
  still gains ~40% from B=64 (190k) to B=512 (264k leaves/s) though the FLOPs barely move. Each *distinct*
  forward pays that fixed tax, so use the fewest distinct forwards that absorb the arrival stream. Start at
  **{64, 256, 512}** (three AOT executables), biased to the largest the drain can fill.
- **S>1, sized to amortize the ZMQ frame, NOT to decide E.** A C++ drain thread coalesces parked leaves
  into ONE multipart message via zero-copy (`zmq_msg_init_data`): one envelope frame of correlation ids +
  N body frames of raw 241-float rows, no serialization copy. The search stops deciding the eval batch
  (strict-barrier's mistake); it only decides how many leaves ride one envelope.
- **D>1, non-blocking.** The drainer issues a new coalesced frame whenever leaves re-accumulate, without
  blocking on the prior reply; replies return OUT OF ORDER keyed by correlation id. D>1 only pays once S>1
  amortizes the frame (D alone was flat at 64..256 *because* S=1 framing was the wall) — they land together.

So: **S moderate (amortize the frame), D high (never stall the search), E server-drained + bucketed
(decoupled, no max-pad)** — three independent mechanisms, matching ADR-0012 P7 (serialization ⊥ transport).

### 1b. The single-threaded-server constraint -> start P=1, scale OUT by processes only if profiling demands

The XLA-single-thread invariant means cross-request parallelism comes only from replicas, never from
threading the forward. But the arithmetic: serve per-core (190k-264k leaves/s) is ~2.5-3.5x aggregate gen
(4x76k = 304k, with ~3 cores free for gen if one runs the server) — so **one well-fed coalescing server
can plausibly saturate the remaining gen cores. Start P=1.** Add a second server PROCESS only if the
microbench shows the single server's recv+drain+forward loop is the wall; the fan-out primitive is lc0's
`demux` backend with `minimum-split-size` (one drainer demuxes to P identical replicas, never splitting a
sub-batch below the bucket floor).

### 1c. Wakeup: signal once per drained group, not per leaf

A literal per-leaf condvar is a caveat at our rate: condvar wakeup is a known wall (futex ~30x
faster on x64). Latency is free and CPU is the constrained resource on 4 vCPUs, so **signal once per
drained group** (one eventfd/futex wake per multipart reply), or a brief bounded spin on the corr slot.

### 1d. Keep ZMQ `inproc` first; shared-memory ring is the ceiling, not step one

The server is in-process: the cheapest correct baseline is `inproc://` (no copy on large messages) with
the zero-copy multipart of §1a. Escalate to a vLLM/Arrow-style mmap SPSC ring ONLY if the microbench shows
the ZMQ envelope still dominates after S>1 amortization — the ring is folklore-grade for our exact case
(no published MCTS->JAX engine uses one) and a build cost we should not pay speculatively.

### 1e. What we explicitly do NOT build

No vLLM/TRT-LLM continuous-batching scheduler: in-flight/iteration-level batching needs a token loop, and
our forward is SINGLE-SHOT (one leaf -> one value/policy), so iteration-level retirement solves a
ragged-completion problem we structurally do not have. The single-shot specialization of continuous
batching *is* this single-shot queue-drain — that is all we need.

## 2. Fit to constraints and ceilings

**ADR-0012:** P7 — the correlation id stitching reply->leaf (and enabling out-of-order D>1 returns) is a
TRANSPORT-envelope frame, never a codec field; the batch/bucket/delay policy lives entirely server-side
and never touches the codec; the frame layout keeps one authoritative home both sides derive (drift-test).
P9 — the queue + drain thread + ZMQ is the imperative shell; leaf coalescing + bucket-selection are pure
core over a `std::span` of leaves with `std::expected` framing. P2 — `predict(features) ->
expected<{value,logits}>` is unchanged; the whole scheme lives BEHIND the port, and P=2 is a config change
against the port, not a rewrite.

**Throughput (MEASURED vs PREDICTED):** MEASURED — gen 152 dps/core (76k leaves/s), 4.0x linear core
scaling; serve 190k@B64 -> 264k@B>=512; system ~49 dps/core (strict-barrier), dropping to ~30 at 2
threads. PREDICTED — removing the D=1 serialization idle (search idles the whole round-trip each batch)
while the next leaf cohort descends should move per-core throughput toward the **gen ceiling of ~152
dps/core**, since serve has 2.5-3.5x headroom and is never the wall; the S=1 frame wall is removed by the
zero-copy multipart. Honest ceiling: a SINGLE tree's in-flight leaves cap below serve's fast region (Lc0's
measured ~1200-1800 collision ceiling) — but we do not need B>=512 to win, only to stop idling. Reaching
serve's 264k needs ELF-style overcommit (multiple trees per server) — a LATER lever, not this adapter.

## 3. Ranked shortlist

1. **(ADOPT)** server-side queue-drain + bucketed E {64,256,512} + zero-copy multipart ZMQ-inproc,
   P=1, group-wakeup. Best expected throughput, cleanest ADR-0012 fit, every piece is shipped production
   prior art. Tradeoff: a real drain loop both sides + corr-id bookkeeping.
2. **(RUNNER-UP)** Same drain+bucketing but P=2 server processes (lc0 demux/`minimum-split-size`) from day
   one. Only wins if the single server's loop is the wall — the arithmetic (serve >> gen) says it is not.
   Reserve for when §4 shows P=1 saturated.
3. **(RUNNER-UP)** Drain+bucketing over an in-process shared-memory SPSC ring instead of ZMQ-inproc.
   Removes serialization to literal zero, but unproven for this case + a real build/drift cost. Only if §4
   shows ZMQ-inproc envelope still dominates after S>1 amortization.

## 4. Validation plan (rigorous, BEFORE any lift into the boundary)

### Stage A — pure-transport microbench (no MCTS, synthetic leaves)

Map the throughput surface over **S x D x E**, isolating transport from search. Rig: a C++ producer
emitting pre-baked random 241-float leaf rows at a rate >= the gen ceiling (so the producer never bounds),
through the real ZMQ-inproc + multipart + zero-copy path, to the real single-threaded JAX server running
the real MLP forward; replies discarded after corr-id match.

- Sweep: S ∈ {1,4,16,64}; D ∈ {1,2,8,32,128}; E-policy ∈ {pad-to-max(512), bucket{64,256,512},
  true-ragged}; wakeup ∈ {per-leaf condvar, per-group futex}.
- Primary metric: aggregate leaves/s (implied dps); secondary: forwards/s, mean rows/forward (confirm E
  decouples from S), per-core CPU occupancy.
- PRE-REGISTERED predictions (so we can be wrong loudly, ADR-0002): S=1 corner frame-bound regardless of D
  (~37 dps, flat in D); D=1 corner serialization-bound regardless of S (~49 dps); (S>=16, D>=8, bucket-E)
  beats both and is the global max; pad-to-max underperforms bucket on any partial drain by ~the pad
  ratio; true-ragged is impossible under XLA (recompile thrash) -> confirm bucket ≈ ragged-in-throughput
  while staying compiled; per-group futex >= per-leaf condvar, widening as leaves/s rises.
- Variance: >=5 runs/cell; report median + min-max; a cell wins only if its MIN beats the other's MAX.

### Stage B — e2e A/B against the real search

Only after Stage A picks a single (S,D,E,wakeup) point. Arms: (1) strict-barrier, (2) greedy-async, (3)
the new adapter. Same MLP, n_sims=256, m=24, `--cores 0,1,2,3`. Metric: decisions/s/core at 1 thread AND 2
threads (confirm we cured the per-core DROP under threading). Protocol: >=5 iters/arm, mean +/- stddev +
min-max; output under `~/w/vdc/chocobo/runs/`, never `/tmp`. Acceptance: arm 3 beats 49 dps/core at 1
thread with NON-OVERLAPPING variance bands AND does not drop per-core at 2 threads; record the gap to 152
as the residual for a later overcommit lever. Escalate: if forward-idle >> forward time at the chosen
point -> runner-up #3 (shm ring); if the single server's loop is CPU-saturated while gen cores starve ->
runner-up #2 (P=2 demux).

## 5. Stage B — executed results (2026-06-19, append-only record)

Stage B was run e2e on the REAL Gumbel-AZ search (the unchanged `run_search` / fiber-mux), every leaf
resolved remotely on the bucketed-E + group-wakeup server (the Stage A `StageAServer`, a server flag —
NOT the production default). Arm 3 (`pipelined-bucket`) is a SELECTABLE runner transport mode behind a
flag; the strict-barrier default (`run_episodes_wire_batched`), the production server, and the wire_spec
are UNCHANGED. Rig: n_sims=256, m=24, hidden=256, `taskset -c 0,1,2,3`, 5 iters/arm, 8s/iter,
pool_batch=64, D(in-flight msgs)=8. Harness: `cpp/src/wire_ab_bench.cpp` + `cpp/stage_a/stage_b_ab.py`;
raw output under `~/w/vdc/chocobo/runs/stage_b_ab/`.

**A/B table (decisions/s/core, mean ± stddev [min–max]):**

| arm | 1 thread | 2 threads |
| — | — | — |
| arm 1 (strict-barrier) | 69.86 ± 3.33 [65.59–73.78] | 47.44 ± 0.27 [46.98–47.79] |
| arm 3 (pipelined-bucket) | 72.65 ± 0.20 [72.36–72.93] | 47.27 ± 0.29 [46.70–47.50] |

(Arm 2 greedy-async ~37 dps from `wire_pool_bench` is the cited reference, not re-run.)

**Findings (raw, ADR-0009 — pre-registered predictions met/missed loudly):**

- **Arm 3 does NOT beat arm 1 with non-overlapping bands at 1 thread** (arm3 min 72.36 ≤ arm1 max
  73.78 — overlapping; the means are within ~4%). Arm 3 is *tighter* (σ=0.20 vs 3.33) but not a
  separated win. The Stage B acceptance gate ("non-overlapping bands at 1 thread") is **NOT met**.
- **Per-core drop at 2 threads:** both arms sit at ~47/core — neither exhibits the feared 49→30 drop;
  the multi-fiber pool already overlaps the RTT under both schedules, so the threading drop the design
  worried about is a strict-barrier-with-low-K artifact, not present at pool_batch=64. Arm 3 does not
  cure a drop arm 1 also does not suffer here.
- **Measured in-flight depth (the key overcommit number):** the single-tree-per-thread search sustains a
  server mean **rows/forward ≈ 53.7 at 1 thread**, **≈ 30.5 at 2 threads** (the queue splits across two
  per-thread DEALERs). Both are **far below the serve fast region B≈192** — an overcommit gap of ~3.6×
  (1t) to ~6× (2t). This is exactly the §2/§4 caveat: K=64 fibers across ONE thread fill ~54 rows, but
  reaching B≥192 needs many more concurrent trees per server (the ELF overcommit of §4), which a single
  per-thread pool does not supply.
- **Why 1-thread arm1 ≈ arm3:** at T=1, K=ceil(64/1)=64 parked fibers, the strict barrier ALREADY
  gathers ~54 rows into one batch (S≈54, D=1) — so the D>1 pipelining adds little when one round's gather
  is already large. The pipelining lever pays only when a single round's gather is SMALL (few fibers
  ready at once), which is the deep-overcommit regime Stage B's single per-thread pool does not reach.

**Byte-identity (Gate 2):** HELD. `CHOCO_RUN_ZMQ=1 CHOCO_RUN_CPP=1 pytest tests/test_zmq_net_cpp.py
tests/test_cpp_runner.py -k "gumbel or wire"` → 4 passed. The forward is row-independent and replies
route per corr-id, so per-leaf predictions and the search are unchanged across the schedule.

**Recommendation:** the single-tree (single per-thread pool) gain is **NOT worth lifting arm 3 as the
default on its own** — it is a ~4% mean improvement with overlapping bands, exactly the "modest gain"
the model predicted. The transport adapter's value is gated on the **overcommit phase** (§4: many
concurrent trees per server to push rows/forward from ~54 toward B≈192, where the serve curve's 235–264k
fast region lives and where D>1 non-blocking pipelining stops the search idling). Arm 3 is correct,
non-regressing, and ready to carry that phase; it should ride as a selectable flag until overcommit
makes the per-round gather small enough that D>1 separates from the strict barrier. The adapter is
validated as CORRECT (parity held) but its throughput payoff is deferred to overcommit, as predicted.

## Key sources

SEED RL (arXiv:1910.06591) · Triton dynamic batcher ·
lc0 demux/`minimum-split-size` · Lc0 collision ceiling · framework tax (arXiv:2302.06117) · CUDA-graph/AOT
replay (PyTorch blog) · ZMQ zero-copy multipart (zguide ch.2) · condvar/futex wakeup cost · Packrat
multi-instance (arXiv:2311.18174) · in-flight==iteration-level (TRT-LLM) · ELF overcommit (arXiv:1707.01067).
Full per-domain survey + the completeness critic: workflow run `wf_66209f19-914`.
