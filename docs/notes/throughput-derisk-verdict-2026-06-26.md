# Throughput de-risk — combined verdict + reframed roadmap (2026-06-26)

Both #1 and #3 de-risked (no GPU), prototypes on feat/tlab-batch-jax (464a2f3) + feat/tlab-batch-insrc
(0eceddf), independently re-verified. The de-risk SURFACED A LEVER not in the original consultation.

## The reframed picture (biggest → smallest, by measured evidence)
1. **AVX2 vpshufb popcount primitive swap: +74%** on the ~55% `belief_features` hotspot. NO seam, NO wire,
   NO batching — just replace scalar `POPCNT` (port-1-bound, 1 word/cyc) in `popcount_and` with an AVX2
   vpshufb popcount (4+ words/instr). Bit-identical. **The biggest + cheapest lever, hiding in plain sight.**
   It's a production-core kernel change (gated by the bit-identity oracle + an A/B), contained to popcount_and.
2. **BatchPredict seam (#3): ~+30% ON TOP of the SIMD primitive** (mask-resident tiling reuses each mask load
   across the belief tile). The headline prototype +124% mostly belongs to #1's primitive swap; the seam's
   TRUE marginal value, measured against an already-SIMD baseline, is ~+30%. Worth the refactor iff that +30%
   justifies it. Idea #3 as a pure cache/loop-reorder is NULL (refuted — masks already L2-resident).
3. **#1 fused-JAX matmul featurization: numerically GO, urgency SHIFTED.** The matmul reframe is
   denotationally EXACT (JAX f64 == C++ double oracle, 0.0); f32 within tolerance (worst 9.5e-7 ≪ 1e-4 bar);
   the legal-mask invariant bit-exact in f32 (nworlds<2^24 guard). CPU-feasible (~41-49 us/belief). BUT 2.02x
   wire (1944 vs 964 B) AND — now that the local producer popcount can be +74% faster — offloading
   featurization is LESS urgent (the producer is less of a bottleneck; the 2x wire is harder to justify).
   #1's value now hinges on the e2e bottleneck (GPU) AFTER the +74% lands.

## Recommended order
1. Land the **AVX2 popcount swap** (+74%, bit-identical, no architecture change) — clear first move.
2. Re-measure the producer; build the **BatchPredict seam** if its now-~+30% marginal still earns the refactor.
3. Re-evaluate **#1** with the GPU + the faster producer (the seam makes it a clean A/B either way).

## Honesty notes (ADR-0013)
- The #3 agent's FIRST draft concluded "NO-GO ≤0%" from the scalar loop-reorder alone; an out-of-frame
  hack-rationalization audit (independent subagent) flagged it UNDISCHARGED-HACK (the natural SIMD layout was
  untested), built a 2.2x kernel, and the deliverable was corrected. The audit caught the real win.
- PROCESS SLIP (mine): the worktrees were branched at 9324a01 BEFORE I committed the design note (3b92ec5),
  so docs/notes/batchpredict-throughput-design-2026-06-26.md was absent on the de-risk branches (the agents
  worked from the brief). Branch after committing shared context next time.

Public Domain (The Unlicense).
