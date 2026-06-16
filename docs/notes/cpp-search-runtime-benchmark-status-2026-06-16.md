<!-- docs/notes/cpp-search-runtime-benchmark-status-2026-06-16.md -->

# Session status: the C++ SearchRuntime benchmark — 2 of 3 axes landed, verified

**Status:** Autonomous-session record. Everything below is on `feat/cpp-search-runtime-serial` (off
current `main` = `c85b97a`), each commit built green at `-std=c++23 -Wall -Wextra` and gated on a
passing check. No end-of-world blockers — the two unbuilt pieces are sized work, not blocked work.

## The question answered: is the Python batched server ready? YES.

`chocofarm/az/inference_server.py` is built and tested: `InferenceServer(ParamsSource, bind, max_batch)`
+ the greedy-drain microbatch loop + `serve_forever()`, with `StaticParamsSource` (params injected, no
redis). `test_zmq_net_cpp.py` already spins it in-process and drives the C++ `ZmqNetClient` against it.
The wire benchmark below confirms it end to end against the real search.

## What landed (verified, pushed)

| commit | what | verification |
| — | — | — |
| `6831601` | `SearchRuntime` seam + `SerialRuntime` (wraps the unchanged `decide`, zero edits to the 1a/1b search) | `chocofarm-serial-runtime-check`: SerialRuntime ≡ per-task `decide`, 6/6 |
| `9d9167d` | the A-vs-B decision memo → **Option A (fiber)**, now confirmed by you + boost.context approved | — (decision record) |
| `c7a5c40` | `PoolRuntime` — local task-parallel tree search + the serial-vs-parallel benchmark | bit-identical to serial; **3.55× at the real budget**, scaling 1.00/1.99/3.25× at 1/2/4 workers |
| `5725d89` | over-the-wire **synchronous** benchmark (SerialRuntime + ZmqNetClient vs the batched server) | end to end: **6.6 decisions/s, ~6.2 ms/leaf** |

## The three benchmark axes (your goal 1)

- ✅ **C++-native MLP (local)** — serial *and* parallel. The "parallel tree descent + backprop"
  abstraction you named is real and measured: near-linear scaling, **bit-identical** to serial (exact,
  not aggregate — independent deterministic trees), and it needs **no fibers** (a local leaf never
  blocks). On the 4-vCPU host it clears the ~1.9× Python-substrate ceiling because it's a fresh C++
  pool over CPU-bound trees.
- ✅ **Over-the-wire synchronous** — 6.6 dps, ~6.2 ms/leaf. That 6.2 ms is the **un-batched single-row
  JAX forward + RTT**: exactly the cost the wire-parallel config amortizes by batching. The wire path
  (C++ encode → server `forward_core` → C++ decode) runs faithfully end to end.
- ⏳ **Over-the-wire parallel** — needs the fiber + DEALER work-stealing pool (below). Not built.

**Honest fairness caveat:** the local bench uses a cheap deterministic net; the wire bench uses the
real `ValueMLP`/JAX. A fully apples-to-apples local-vs-wire needs `NetForward` on the **same** weights
(read via the manifest) — a small refinement, noted, not yet done.

## What is NOT built, and the precise plan (sized, not blocked)

**1. Over-the-wire parallel = the fiber + DEALER work-stealing pool (the 3rd axis).** Per the decision
memo, Option A + boost.context:
  - (a) `YieldingNetEvaluator` + a boost.context fiber wrapper, proven in isolation: run the **unchanged**
    `run_search` in a fiber, the leaf yields, assert fiber-driven ≡ direct (near-trivial under A).
  - (b) the unified work-stealing pool over `{SELECT, BACKPROP, FAIL}`, workers running trees as fibers,
    **single-writer-per-tree gated by a TSan test before it's trusted** (§8.2).
  - (c) the `DealerRendezvous` (non-blocking submit/poll) + the echoed-`u64`-corr_id wire amendment
    (a P7-disciplined `wire_spec` SSOT bump).
  - (d) wire it in as the third `--net` mode of the benchmark.
  This is the careful systems chunk — deliberately not rushed unverified overnight; the boost
  integration is worth a fresh, focused pass.

**2. A test AZ loop (your goal 2).** Prerequisite: the full `Decision` (`improved_pi` the trainer
target + `n_spent`), which needs the production `RngGumbelSource` exposed (move it from `gumbel.cpp`'s
anonymous namespace to the header — behaviour-preserving, re-verified by `gumbel_logic.py` +
`gumbel_precision.py`) so the runtime drives `run_search` directly. Then: a C++ Gumbel-AZ episode that
emits AZ transitions (features, improved-π, value target) → the existing redis transport → the existing
Python learner does a training step → publishes weights → the C++ side reloads. The Python AZ stack
(`exit_loop`, `worker`, `train`) already exists; this is the actor-side wiring + one turn of the loop.

## Locked decisions
Option A (fiber) for the resumable search; boost.context as the fiber mechanism; the work-stealing
pool is the spine (transport + scheduling swappable beneath it); the pool is born-clean structure,
only the DEALER transport is the measure-first-gated optimization.

*Public Domain (The Unlicense).*
