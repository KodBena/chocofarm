<!--
docs/notes/consult/opus-consult-2026-06-18-jaxwire-throughput.md
Purpose: point-in-time record of an independent Opus performance consultation on the C++-search /
  JAX-inference-server wire actor — the analyst was given ONLY the two e2e profiles, the source, the
  throughput goal, and ADR-0012 (no prior conclusions, to keep the read independent). Preserved as a
  REUSABLE METHODOLOGY: the prompt below is the template for the next profile→consult iteration.
Public Domain (The Unlicense).
-->

# Opus consult — JAX-wire generation throughput (2026-06-18)

Methodology: profile the real e2e generation two ways (a `perf` capture of the C++ runner; a `cProfile`
of the Python parent), hand an independent Opus analyst **only** the artifacts + source + the throughput
goal + ADR-0012 (deliberately unbiased — no hypotheses from us), and have it interpret + rank fixes.
Re-run this loop after each round of fixes. Artifacts profiled: `~/w/vdc/chocobo/profiles/e2e-cpp-b128t1.perf.data`,
`e2e-py-b64t2.{prof,svg}` (e2e `exit_loop`, n_sims=256, m=24, hidden=256, I=2; cpp run T1/B128, py run T2/B64).

## The prompt (reusable template)

> You are an independent performance analyst. Read the evidence yourself and form your own
> interpretation — no prior conclusions are supplied.
>
> System: a C++ Monte-Carlo-tree-search self-play actor (Gumbel-AlphaZero). The search runs in a C++
> subprocess (`chocofarm-cpp-runner`); every tree leaf needs a neural-net forward, which the C++ search
> submits over a ZMQ DEALER socket to an in-process Python JAX `InferenceServer` that batches leaves and
> returns value+policy. Generation (self-play) is the throughput target — decisions/sec; current regime
> ~30–40 dps.
>
> Two profiles of the SAME e2e generation loop (`exit_loop`, n_sims=256, m=24, hidden=256, 2 iters) are
> on disk — the two runs use different pool configs, so don't naively diff absolute numbers:
> [cpp `perf record -g` T1/B128 → e2e-cpp-b128t1.perf.data, read with `perf report --stdio` (TUI bug),
> runner comm `chocofarm-cpp-r`] [py `cProfile` T2/B64 → e2e-py-b64t2.prof, read with `pstats`].
> Source: runner_wire_batched.cpp, wire_leaf_pool.hpp, gumbel.cpp, features.cpp, inference_server.py,
> cpp_executor.py, exit_loop.py.
> Deliver: (1) where time actually goes + the binding bottleneck(s); (2) concrete, ranked changes to
> raise throughput, each tied to profile/code evidence + rough headroom. Single design constraint:
> **ADR-0012** (read it). Analyze and recommend only; do not modify files.

## The analysis (verbatim)

**The C++ search is not the bottleneck** — `chocofarm-cpp-r` is only **14.8% of all cycles**; no dominant
hotspot (`belief_features` 3.5%, child-key/geometry hashtables ~1.1%, `evaluate`/`descend`/`puct_select`
~1.4%, `filter_bits`/`rank_or_abort` ~0.9%, malloc ~2%, `WireLeafPool::poll` 0.18%). A well-optimized
search; effort here moves throughput very little.

**The Python parent is 81.6% of cycles, almost none of it the NN math.** Three buckets:

1. **XLA/LLVM JIT recompilation — the single largest cost.** `backend_compile_and_load` = 9.9s self over
   **660 compiles** (warmup pre-compiles only 128). `run_microbatch` stacks `(B,241)`, so each distinct
   batch size `B` is a fresh XLA executable; varying greedy-drain `B` + version reloads recompile during
   the run. `PjRtCompiler::CompileAndLoad` ~7.4% cum in perf; MLIR `StorageUniquer` 5.2%. Compilation,
   not inference.
2. **Eigen CPU threadpool spin — ~24.5% cumulative.** `ThreadPoolTempl::WorkerLoop`/`GlobalSteal`/
   `RunQueue::PopBack` self-cost = XLA's intra-op pool spinning/work-stealing on matmuls too small to
   fill it. Real kernel (`dot_fp32 fma3`) only 2.77%; `_aligned_contig_cast_double_to_float` 1.63%
   (f64 weights recast to f32 per forward). Batches are tiny: 819,891 leaf sends / 17,812 forwards ≈
   **46 leaves/forward**; a 46×241 @ 241×256 matmul is microscopic, so threadpool overhead dwarfs FLOPs.
3. **ZMQ per-leaf transport — ~26s in cProfile.** `send` 6.1s + `send_multipart` 2.75s self/15.9s cum +
   `recv_multipart` 8.5s across ~820K leaf round-trips. **6.7s of it is `enum.__or__`** — `zmq.SNDMORE`
   re-OR'd inside 1.64M frame-sends. `decode_request` 3.5s self/9.6s cum (820K calls), each doing
   `np.all(np.isfinite)` per leaf; `encode_response` 1.5s.

**Separable non-generation cost in the py profile:** the in-process Python `GumbelPolicy` **eval**
(`forward_core` called 22,375× via `mlp._predict_both_f32`, `env.simulate`) and a **24.7s clairvoyant-
ceiling %VoI** brute-force (`combinations × permutations`, 2.6M `route_time`) in a side thread — eval/
metrics, not the gen path; they inflate the 113.7s cProfile wall.

**Binding bottleneck:** the inference server's **per-forward fixed overhead at tiny batch sizes** —
(a) XLA recompilation across varying `B`, (b) Eigen-threadpool spin on tiny matmuls — not the C++ search
and not the FLOPs. Root cause: **starvation** — `poll()`-one-then-resubmit keeps in-flight (and thus `B`)
far below T×K, so the server pays fixed cost over ~46 leaves instead of hundreds.

**Ranked recommendations** (all ADR-0012-clean; no wire-format change, P7 untouched; #1/#2 are P9/P6
measured-perf, behavioral-equivalence not byte-identity):
1. **Pad every forward to a fixed `B`** (or a few buckets) — kill batch-size-varying recompilation.
   P9 hot-path Workspace covers the fixed-shape scratch; pad `(B,241)` with zero rows, slice back.
   **Headroom: large** (most of the ~7–15% compile + the warmup).
2. **Pin XLA intra-op threadpool to 1** on the 4-vCPU host. **Headroom: large** (much of the 24.5%);
   numerics within the accepted f32 ABS_TOL.
3. **Raise achieved batch** — submit/await many leaves per worker before draining so `B`→T×K; fewer,
   fatter forwards amortize all per-forward + ZMQ cost. Behind the P2/P7 seams. **Headroom: medium-large.**
4. **Cut per-leaf codec overhead** — hoist `zmq.SNDMORE` to a constant; do `isfinite` on the stacked
   batch, not per row. Loud-reject boundary preserved (P9/ADR-0002). **Headroom: medium.**
5. **Cast f64→f32 weights once at reload**, not per forward (`_aligned_contig_cast` 1.63%). P1/P6.
   **Headroom: small but free.**
6. **(Hygiene)** exclude the in-process eval + the 24.7s clairvoyant-%VoI from the generation-dps headline.

Files: `inference_server.py` (#1,#3,#4,#5), `inference_wire.py:90,112` (#4), `forward.py` (#2,#5),
`runner_wire_batched.cpp` + `wire_leaf_pool.hpp` (#3 drain loop), `references.py` + `exit_loop.py:471-487` (#6).
