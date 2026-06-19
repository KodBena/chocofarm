<!--
docs/design/stall-investigation/blind-model-v2/README.md
Purpose: provenance + findings for the BLIND, N-parametric, cleanroom re-run of the leaf-eval transport
  model (2026-06-19). Supersedes ../blind-model/ (which modeled only the N=1 baseline). Companion to
  ../formal/, ../../cpp-eval-wire-formal-diagnosis.md, and ../../cpp-eval-transport-adapter.md secs 6/7.
  Point-in-time record — amend by append (ADR-0005 Rule 8).
Public Domain (The Unlicense).
-->

# Blind N-parametric model of the leaf-eval transport boundary (2026-06-19, re-run)

Output of a multi-agent **blind** workflow (run `wf_5b8a1d86-930`) that re-modeled the C++↔Python
leaf-evaluation transport boundary against the **overcommit geometry** (`trees_per_thread = N`,
`K = N·base`) and **both server drains** (production greedy + bench bucketed-group), then reconciled the
fresh models against the prior N=1 model (`../blind-model/SYNTHESIS.md`). This supersedes `../blind-model/`,
which modeled only the N=1 baseline.

## Methodology (why this is trustworthy)

- **Blind on a cleanroom.** The modeling/verifying agents read ONLY a comment-stripped, whitelist source
  tree built by `tools/cleanroom.py` (comments/docstrings removed — source headers literally state the
  throughput target; blank runs compacted ≤1; `verify` lint passed 15/15). They were never told a defect
  exists, the empirical fingerprint, the mechanism, or the prior model.
- **Structure:** 5 modelers (2 producer, 3 server: greedy-drain / bucket-drain / ROUTER-transport) → 4
  fidelity verifiers (too-permissive / too-constrained per side) → a **reconciler** (cross-check vs the
  prior N=1 model, by symbol not line) → 5 targeted capture-up derivations → synthesizer. 17 agents.
- The **empirical cross-check and the redirect below are NOT the workflow's** — they were added afterward
  by the orchestrator (who may see the §7 results; the blind agents may not).

## Central finding (high confidence — derived + bounded-Z3, not asserted)

**The transport boundary is stable for all N. There is no transport coalescing livelock.**
- Per-thread in-flight message depth is **identically 1 for all N, T, D** (`issue_one` coalesces all ready
  slots into one message; depth never reaches 2 — Z3 UNSAT at K∈{2,8,24}). `D = max_inflight_msgs` is a
  **dead parameter** on the pipelined path. N scales **rows per message only** (≤ K = N·base), never
  message count, depth, or feedback sign.
- The greedy-drain batch-size fixed point is **bounded by an absorbing ceiling** `min(T·K, max_batch+K−1)`
  via two **service-time-independent** clamps; divergence is **UNSAT at N ∈ {1,8,33,75,200}**. The
  self-clocking feedback is **negative for all N and never flips sign**. (This is the rigor `convoy4`
  lacked: stability *derived*, not asserted.)
- The reconciler **caught both fresh primary producer models regressing** (re-asserting depth→D growth, the
  prior work's error) and rescued depth-1 from the audits — the pre-synthesizer stage earned its place.
- Both fail-loud terminals are distinguished: `_reject` (caught; drops one reply; one-peer RCVTIMEO) vs
  `EXCEPTIONAL_TERMINATION` (uncaught `ValueError` → server thread death → all-peer RCVTIMEO; RELY-gated,
  unreachable under a conforming peer). The scatter is non-blocking (ROUTER_MANDATORY=0 → drop).

## The redirect (orchestrator cross-check vs §7 — strong hypothesis, pending empirical confirm)

The empirical §7 stall ("nondeterministic, N≳4, works once / wedges the next") is **not** a transport
livelock. Combining the faithful model with the bench facts — `RCVTIMEO=60s` (`wire_ab_bench.cpp:119`),
the **N-aware subprocess timeout** that `overcommit_sweep.py` notes "under-estimates high-N", and
`warmup` covering only `{64,256,512}` — the stall localizes to the model's **R2 OVERSHOOT**:

> At high N the bucket drain forwards **unpadded shapes >512** (overshoot width up to `max_batch+K−1` ≈ 709
> at N=9), a range that grows linearly with N (~197 distinct widths) and that **warmup never covers**. Each
> first-seen width pays a cold XLA compile (~seconds); the nondeterministic batch sizes hit a *cluster* of
> new unwarmed shapes → a transient compile tax the wave-time estimate doesn't budget → it overruns the
> subprocess timeout. "Works once / wedges the next."

The model *has* this (R2: "unwarmed fresh shape", "O(N) distinct widths") but characterized the compiles as
**"benign / fully amortized"** — true for steady-state throughput, **false for an estimate-based harness
timeout**. The fix therefore redirects from a **min-batch / coalescing floor** (the prior convoy framing) to
a **closed** fix: pre-warm the overshoot widths, or hard-cap the batch so no over-cap unwarmed shape is
produced, or bucket the >512 region. **To confirm:** measure compile count / distinct >512 widths per wave
at high N, and whether pre-warming/hard-capping makes the N=8,9 wedge vanish.

## Standing caveats (from the model)

- The whole N-axis is conditional on `mode == PipelinedBucket`; the in-tree default is **strict-barrier**
  (N-invariant). No in-cleanroom launcher flips it (open Q1).
- `finalize_and_write`'s redis write is an **off-boundary** source-timing input the model did not cover
  (open Q6) — a secondary wedge candidate if the compile hypothesis under-accounts.

## Files

`SYNTHESIS.md` (the authoritative parametric model) · `RECONCILE.md` (the pre-synthesizer) ·
`model-*.md` (5 side-models) · `verify-*.md` (4 audits) · `derive-G1-greedy-stability.md`,
`mean_rows_per_msg_derivation.md`, `Q-*.md` (capture-up) · `*.py` (bounded Z3 confirmations).
Live (gitignored) output: `~/w/vdc/chocobo/runs/leaf-eval-model-2/`.
