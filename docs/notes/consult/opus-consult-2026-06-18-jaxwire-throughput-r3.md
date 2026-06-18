<!--
docs/notes/consult/opus-consult-2026-06-18-jaxwire-throughput-r3.md
Purpose: ROUND 3 of the profile→independent-Opus-consult methodology (rounds 1/2 + the reusable prompt in
  the round-1 doc). Same two profiles (C++ perf T=1/B=64, Python cProfile T=2/B=64), run on main 5fe3ef2
  (batched wire #2 + jax.jit forward #1 + clairvoyant cache #3), clairvoyant pre-cached so it does not
  confound. Records the verified r2→r3 deltas, the firewall's read, and the RESOLVED Eigen-thread question.
Public Domain (The Unlicense).
-->

# Opus consult R3 — JAX-wire throughput, post #1/#2/#3 (2026-06-18)

Same prompt as rounds 1/2 (the round-1 template), pointed at `e2e-cpp-b64t1-r3.perf.data` (T=1/B=64) and
`e2e-py-b64t2-r3.prof` (T=2/B=64), unbiased — plus an explicit ask to determine the forward's threading.

## Verified r2 → r3 deltas (measured — all three fixes landed hard)

cProfile diff (Python parent, T=2/B=64):

| metric | r2 (post-B/C/D) | r3 (post #1/#2/#3) | fix |
|---|---|---|---|
| total wall | 97.6s | **40.1s (−59%)** | |
| per-leaf ZMQ `recv_multipart` | 832k calls / 8.45s | **62k / 1.18s** | #2 wire-batch (~13× fewer frames) |
| eager `_operator_matmul` | 70,872 / 2.07s | **8 / 0.00s** | #1 jit (eager dispatch collapsed) |
| `route_time` (clairvoyant) | 2.6M / 5.23s | **absent** | #3 cache (no recompute) |
| dps (Python profile iter1) | 27.3 | **51.5** | combined ~2× |

## The Eigen-thread question — RESOLVED (round 2 had it wrong)

The forward runs `CpuPjRtRawLoadedExecutable::Execute → ThunkExecutor::ExecuteSequential → YnnFusionThunk
→ ynn::dot_fp32` inside ONE Eigen `WorkerLoop` thread, **thunks SEQUENTIAL** (not a parallel executor).
So `--xla_cpu_multi_thread_eigen=false` (config.py) DID take effect — the matmul is single-threaded; XLA's
PjRt runtime merely **hops execution onto one worker thread** (`ThreadPoolAsyncWorkRunner::Execute`), a
thread-hop, not a parallel fan-out. Round 2's "24.5% Eigen spin" was a MISREAD: that time is the
single-threaded matmul compute running on the worker, not wasted work-stealing. **Single-threaded is
CORRECT here** (a 27×256×256 GEMM on a 4-vCPU VM would lose to thread-pool sync) — there is no win from
changing the thread count. The config.py pin stays; the C-fix was effective. Question closed.

## Firewall round-3 read (independent)

- **The loop SERIALIZES the two halves.** The strict gather-barrier (`runner_wire_batched.cpp:306-333`)
  submits then immediately recvs — ≤1 request in flight per thread — so C++ search (~40% of the C++-profile
  tree samples) and the JAX forward (~57%) never overlap on the wall clock. This is the deepest limiter now.
- **The Python forward is dispatch/transfer-bound, not compute-bound.** Per-microbatch fixed overhead
  dominates the ~27-wide matmul: `numpy.asarray` device→host pull 9.7s (321µs × 30,173 microbatches),
  an eager `convert_element_type` dispatch ~2.5s (the `jnp.asarray(Xb)` cast + de-standardize, OUTSIDE the
  jit), ZMQ poll/recv + codec ~6.6s. rows/microbatch ≈ 27 (the gather batches; padding waste ~2.4×, not 64×).
- **C++:** `belief_features` 9.88% self (hottest leaf) + per-leaf/per-round alloc churn (`_int_malloc`
  ~4% + memmove) ~6% self.

## Ranked next fixes (firewall) — for the next round

- **#1 — overlap C++ search with the JAX forward (pipeline / double-buffer the gather-barrier):** two
  corr-ids in flight, or split K into alternating half-batches, so a thread submits round n+1 while round
  n's reply is outstanding. The BIGGEST lever (~1.4–1.7×). This is the design's own async direction AND the
  "rejected alternative" recorded at `runner_wire_batched.cpp:294-296` (the open question we defaulted away
  from in #2). Respect P6 (parallel≈serial aggregate-determinism is a recorded behavioral-equivalence
  judgment — Revisit-when #5) and P9 (keep the in-flight slot state typed, one writer).
- **#2 — fold the host→device cast + de-standardization into the jit:** kills the eager
  `convert_element_type` (~2.5s, per-call) and shrinks the pull. P1 (one `forward_core` home) / P6.
- **#3 — cut the per-microbatch host↔device round-trip** (`numpy.asarray` 9.7s): compounds with #1
  (fatter microbatches → fewer pulls) and #2 (only the final result is pulled). ~3–5s, overlaps #1/#2.
- **#4 — trim C++ per-leaf alloc churn** (~6% of the C++ half) via a typed `Workspace&` (P9 rule 4 — the
  perf data IS the required allocation measurement). Secondary; only matters once #1 rebalances the poles.
- **Do NOT touch the XLA thread count** — single-threaded `ExecuteSequential` is confirmed correct at this
  batch/host.
