<!--
docs/notes/consult/opus-consult-2026-06-18-jaxwire-throughput-r2.md
Purpose: ROUND 2 of the profile→independent-Opus-consult methodology (round 1 + the reusable prompt:
  opus-consult-2026-06-18-jaxwire-throughput.md). Same prompt template, same two profiles (C++ perf
  T=1/B=64, Python cProfile T=2/B=64), run on the POST-fix code (main 08fb264: fixed-shape batch /
  XLA-pin SSOT / f32-at-load). Records the verified round-1→round-2 deltas + the firewall's round-2 read.
Public Domain (The Unlicense).
-->

# Opus consult R2 — JAX-wire throughput, post-B/C/D (2026-06-18)

Same prompt as round 1 (template in the round-1 doc), pointed at `e2e-cpp-b64t1-r2.perf.data` (T=1/B=64)
and `e2e-py-b64t2-r2.prof` (T=2/B=64), unbiased (no mention of the changes).

## Verified round-1 → round-2 deltas (the loop's payoff — measured, not the firewall's claim)

cProfile diff (Python parent, T=2/B=64), round-1 `e2e-py-b64t2.prof` vs round-2 `e2e-py-b64t2-r2.prof`:

| metric | round 1 (pre-fix) | round 2 (post-fix) | verdict |
|---|---|---|---|
| total wall | 113.7s | **97.6s (−14%)** | |
| XLA compiles (`backend_compile_and_load`) | **660 / 9.90s** | **30 / 0.45s** | fixed-shape batch **worked** ✓ |
| matmul (`_operator_matmul`) | 4.24s (f64) | **2.07s (f32)** | f32-at-load **worked** ✓ |
| per-leaf ZMQ (`recv_multipart`) | 8.46s | 8.45s | untouched (next target) |
| eager dispatch (`apply_primitive`) | 1.34s | 1.38s | untouched (forward not jit'd) |

The XLA **single-thread pin (C) is UNCONFIRMED**: the round-2 perf still shows ~35% in XLA/Eigen
worker-thread forward execution, so `--xla_cpu_multi_thread_eigen=false` may not control this XLA build's
matmul-thread backend. Needs a direct check.

## Firewall round-2 read (independent)

- **C++ runner ≈ 25% of cycles** (not the bottleneck): feature build (`belief_features` ~7%) +
  per-descent malloc/`std::variant` churn (~3.5%); search arithmetic <1% each.
- **Python parent ≈ 75%**, two centers:
  1. **Clairvoyant/VoI reference ≈ 24.6s (25% of wall)** — one-time brute-force `itertools.permutations`
     memo (`references.py` `clairvoyant_rate`), NOT generation; amortizes to ~0 on long runs, poisons
     short/CI timings.
  2. **Generation ≈ 60.9s**, bound by **per-leaf round-trip overhead, not compute**: ZMQ Python layer
     ~18.85s (820k leaves, one frame each), codec ~7s, **eager-JAX dispatch ~6s (server runs bare
     `forward_core`, not `jax.jit`'d)**, enum/`IntFlag` hot-path ~5s. Served batch **B≈46** (near the 64
     cap) → XLA batching works, the ~7s forward is well-amortized; the cost is **messaging granularity**.
- **Binding bottleneck:** per-leaf round-trip overhead in the Python server (ZMQ frames + codec + eager
  dispatch, ~33s), dwarfing the forward (~7s) and the C++ search.

## Ranked next fixes (firewall) — execution order chosen by the maintainer: #2 → #1 → #3

- **#2 — batch the leaf submissions over the wire** (~30%+, the biggest lever): send K parked leaves in
  one DEALER message, receive K predictions in one reply, instead of one frame per leaf. **Delicate:**
  P7 — the batched frame must be a NEW versioned shape in the single-authored `wire_spec` SSOT (both
  sides derive, drift-tested), NOT a hand-mirrored second codec; P9 — typed framing; the corr-id→slot
  routing generalizes to a vector.
- **#1 — `jax.jit` the forward** (~10%): collapses the eager per-primitive dispatch into one compiled
  executable call. P6 (numeric-equivalence bar, not byte-identity) / P7-P1 (still the one `forward_core`).
- **#3 — the VoI reference off the hot/short path** (~25% short-run, amortizes to ~0 long-run — a
  LATENCY/CI win, not steady-state dps): make it an optimistic cache keyed by an instance hash on the
  persistent (6379, noeviction) redis — fetch if present, else compute+store; pre-populate out-of-band
  before profiling so it does not confound. Fail-loud on an instance-hash mismatch (P2 ACL).
- (#4 C++ per-descent alloc — few %, after the wire; #5 enum hot-path — ~5%, low-confidence locus.)
