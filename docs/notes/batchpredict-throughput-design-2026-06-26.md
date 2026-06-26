# BatchPredict throughput work — design + de-risk plan (2026-06-26)

Both levers attack the SAME target: `belief_features`, the ~55% producer hotspot. Neither needs the GPU to
*develop or de-risk* (the GPU only speeds the net later). Roadmap-before-code; these are de-risk PROTOTYPES
(measure + parity), not production rewires — the real builds follow the evidence.

## The seam (the abstraction both impls plug into)
`BatchPredict`: a batch of `(loc, belief, collected)` (the B leaves the cursor already parks per RTT)
→ `B × prediction`. Two implementations behind ONE interface, A/B'd head-to-head once both exist:
- **#3 in-process**: featurize the batch in C++ (batched, cache-friendly) → net (JAX, as today). No wire change.
- **#1 fused-JAX**: ship the raw belief batch to JAX; featurize (as a matmul) + net there. Wire change.

This is why #3 is the foundation: once the seam exists, #1 is a second impl, A/B'd behind it — not a rewrite.

## #1 — featurization as a matmul, fused into the JAX forward
Reframe: `marginals = belief_indicator · world_feature_matrix` (belief_indicator = the nb-bit live-world mask;
world_feature_matrix = the env-static nb×(N+nD) bit matrix). Batched over B = a matmul → JAX/XLA-native,
rides the batch, GPU-amortizes later. Ship belief instead of features.
**De-risk questions (CPU JAX, no GPU):** (a) does the matmul reproduce C++ `belief_features`
(marginals/p_pos/informative) within float tolerance? — NB this is a TOLERANCE bar, not bit-exact (C++ double
vs JAX f32): a real provability cost to name. (b) wire payload: belief (~bitset 1944 B) vs feature vector
(~241 f32 ≈ 964 B) — the compute-for-bandwidth trade. (c) CPU matmul timing — feasibility.

## #3 — in-process batched featurization
Hypothesis: featurizing the B parked beliefs *together* (transpose the loops so the masks/treasure columns
stay hot across the batch) amortizes mask loads + improves cache/SIMD vs B separate per-leaf sweeps.
**De-risk questions:** (a) bit-identity — batched features == per-leaf features byte-for-byte. (b) is there a
measurable cache/SIMD win, and how big? Ceiling is modest (stays CPU-bound, no offload) but it's the
foundation + a free clarity/locality win if positive.

## Sequencing
De-risk both now (this pass) → if #3 wins, build the seam + in-process featurizer → add #1 as the second
impl → A/B behind the seam (GPU makes #1's net faster, but the seam comparison is fair either way).
Branches: `feat/tlab-batch-insrc` (#3), `feat/tlab-batch-jax` (#1), off `feat/tlab-phantom-counts`.

Public Domain (The Unlicense).
