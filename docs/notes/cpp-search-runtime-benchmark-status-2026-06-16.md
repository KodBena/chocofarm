<!-- docs/notes/cpp-search-runtime-benchmark-status-2026-06-16.md -->

# Session status: goal 1 (the 3-way benchmark) COMPLETE + verified; goal 2 (AZ loop) scoped

**Status:** Autonomous-session record. Everything below is on `feat/cpp-search-runtime-serial` (off
`main` = `c85b97a`), each commit built green at `-std=c++23 -Wall -Wextra` and gated on a passing
check (opt-in `CHOCO_RUN_CPP=1` for the binary-dependent ones). No end-of-world blockers.

## The Python batched server is ready (the opening question)

`chocofarm/az/inference_server.py` — `InferenceServer(ParamsSource, bind, max_batch)` + the greedy-drain
microbatch loop + `serve_forever()`, with `StaticParamsSource` (params injected, no redis). Driven end
to end below against the real C++ search over the wire.

## GOAL 1 — the three-way benchmark: DONE

| axis | what | measured (4-vCPU host, deterministic/representative net) |
| — | — | — |
| **C++-native MLP (local)** | `SerialRuntime` + `PoolRuntime` (`c7a5c40`) | **3.55×** at the real budget; **1.00 / 1.99 / 3.25×** at 1/2/4 workers; **bit-identical** to serial (exact, not aggregate) |
| **over-the-wire synchronous** | `SerialRuntime` + a blocking `ZmqNetClient` (`5725d89`) | **9.8 dec/s, ~4–6 ms/leaf** (the un-batched single-row JAX forward + RTT) |
| **over-the-wire parallel** | K boost.context tree-fibers + batched DEALER (`02d0606`) | **13.97 dec/s, 1.43× over sync** (same 390 leaves; first batch = 16) |

Plus the load-bearing foundations, each proven in isolation and mechanized:
- **Option A (fiber) proven** (`c109755`): the UNCHANGED `run_search` runs inside a boost.context fiber, a
  `YieldingNetEvaluator` yields at each leaf — result **bit-identical** to the direct run (executed /
  improved-π argmax / n_spent), at the real budget (94 leaves through the fiber). Fidelity preserved by
  construction; no continuation rewrite of the 1a/1b search.
- **DEALER batched transport verified** (`chocofarm-dealer-probe`): 32 concurrent submits, server batches,
  positional FIFO holds (32/32).

### Honest reads (not to be mistaken for limits of the approach)
- The wire-parallel **1.43×** is the **round-synchronous MVP** (a barrier per round: submit-all, recv-all,
  resume-all). The win is capped by per-round RTT, NOT the batched forward. The continuous **greedy-async
  work-stealing pool** (per-tree corr-id, no barrier — the production design) is the path to the full
  batching win; the MVP proves the mechanism end to end and that parallel > sync.
- Fair local-vs-wire needs `NetForward` on the **same** weights (the local bench uses a cheap `DetNet`,
  the wire benches the real `ValueMLP`/JAX). A small refinement.
- P1 cleanup: `YieldCtx`/`YieldingNetEvaluator` are inlined in both `fiber_proto.cpp` and
  `wire_parallel_bench.cpp` — extract to a shared fiber-leaf header when the production pool lands.

## GOAL 2 — a test AZ loop: scoped (the actor scaffold already exists)

`cpp/src/runner.cpp` ALREADY runs E self-play episodes via an injected `Policy` and writes the four
(X, PI, M, Y) AZ transition blocks to redis (mirroring `worker.py`'s `generate_episode`) — it is the
actor, wired today to `RandomPolicy` with the Gumbel search "deferred." The concrete remaining work:

1. **The full `Decision` (`improved_pi` + `n_spent`)** — expose the production `RngGumbelSource` (move it
   from `gumbel.cpp`'s anonymous namespace to the header; behaviour-preserving, re-verified by
   `gumbel_logic.py` + `gumbel_precision.py`) so the runtime/runner drives `run_search` and captures the
   improved-π **as the PI target** (today the runner uses `policy.decide()`, which discards it).
2. **Wire `GumbelAZPolicy` + a `NetForward`** (local, from published weights) into `run_episode` as the
   actor policy, emitting improved-π into the PI block.
3. **One turn of the loop:** the C++ actor generates Gumbel transitions → the existing redis transport →
   the existing Python learner (`worker.py`/`exit_loop`) trains a step → publishes weights → the C++ side
   reloads (the version-gated seam already exists). A short run proves the loop turns.

## Locked decisions
Option A (fiber) + boost.context (installed, linked via `find_package(Boost COMPONENTS context)` + a
find_library fallback); the unified work-stealing pool is the spine; the pool is born-clean structure,
only the DEALER transport is the measure-first-gated optimization; the greedy-async (no-barrier) pool is
the wire-parallel production refinement past the round-synchronous MVP.

## The session's commits (feat/cpp-search-runtime-serial)
`6831601` seam+SerialRuntime · `9d9167d` Option-A decision · `c7a5c40` PoolRuntime+local bench ·
`5725d89` wire-sync bench · `c109755` fiber proof · `02d0606` wire-parallel bench + DEALER ·
(`af67596` → this memo)

*Public Domain (The Unlicense).*
