# AZ MLP — JAX/optax training, kept numpy inference, pinned by a float32 equivalence test (2026-06-14)

The MLP **training** moves to JAX (autodiff via `jax.value_and_grad` + optax-Adam); **inference
stays numpy float32** (it is faster at batch-1). The two forward implementations are pinned to each
other by a **numpy-vs-jax-JIT float32 equivalence test** — the load-bearing safeguard.

Worktree `feat/az-jax-train` (off `feat/az-residual`), committed but **not pushed**. This is the
machinery build + a bounded smoke — NOT a training-quality result. The hours-scale real run is the
orchestrator's.

---

## Why

Manual backprop is fragile — a whole investigation went into the hand-derived residual backward +
finite-difference gradient-check (`docs/results/az-residual.md` §1). `jax.value_and_grad` makes the
gradient **correct-by-construction**: no hand-derived backward to re-derive, and an architecture
change (a second residual block, a different head) is a one-line edit to the forward with nothing to
re-derive on the backward side.

But jax-jit **batch-1 inference is materially slower than numpy** at single-row dispatch (the
dispatch + host↔device transfer tax; the negative result `docs/results/az-jax-perf.md` already
established and which this pass re-confirms). The search calls the leaf forward one belief at a
time with fresh numpy arrays, so numpy wins there. **So inference stays numpy.**

The risk of keeping two forward implementations (numpy `_forward` for inference, `_forward_jax` for
training) is bounded by the equivalence test, which compares numpy against the **jit'd** jax forward
(not eager jax): XLA fuses/reorders, so jit numerics differ from eager, and the weights are trained
under the jit'd forward — numpy inference must match *that* to float32, or the search would run a
subtly different net than training optimized.

## What changed

| site | change |
|---|---|
| `chocofarm/az/mlp_jax_train.py` (new) | functional `_forward_jax` (params-pytree → (v_std, logits)) mirroring `ValueMLP._forward` exactly (trunk 241→256→256 ReLU, no-outer-ReLU residual block, linear value head, 65-logit policy head); `forward_jax_jit = jax.jit(_forward_jax)` (the equivalence reference); the AZ loss `α·CE(masked) + β·MSE(v_std) + ½·l2·‖W‖²`; `JaxTrainer` (optax-Adam over `value_and_grad`, jit'd step, writes weights back into the net) |
| `chocofarm/az/mlp.py` | manual `_residual_backward` / `train_step` / `train_step_value` / `_adam_apply` / `_init_adam` marked **SUPERSEDED** (kept, not deleted — they document the loss the JAX path reproduces, and the equivalence test needs the numpy `_forward` they were built around) |
| `chocofarm/az/exit_loop.py` | builds a `JaxTrainer` once (Adam moments persist across iterations); `train_epochs` steps the trainer; generation/eval inference unchanged (numpy) |
| `chocofarm/az/train_value.py` | Stage-1 value-net training also via `JaxTrainer` (value-only loss) |
| `tests/test_az_loop.py` | dropped the manual residual gradient-check (`_residual_grad_check` / `test_residual_gradient_check`) — gradients are now correct-by-construction; added `test_jax_train_step_reduces_loss`, `test_jax_train_writes_back_numpy_inference`; kept the numpy-inference / cache-coherence / save-load tests |
| `tests/test_jax_equivalence.py` (new) | THE equivalence safeguard + batch-1 latency benchmark |

Inference is untouched: `predict_both` / `predict_value` / the search leaf path all run the numpy
float32 fast path. The float32 inference cache stays coherent because the trainer **rebinds** the
net's weight arrays (`net.W1 = np.asarray(...)`), which the cache's identity check catches — the
same invariant the numpy path relied on (no new invalidation gate needed). The redis worker
transport (`parallel.pack_net`/`unpack_net`) and checkpoint save/load round-trip the JAX-trained
params unchanged (verified).

## The numpy-vs-jax-JIT float32 equivalence numbers (the maintainer's explicit requirement)

`numpy float64 _forward` vs `jax.jit(_forward_jax)` float32, residual ON, random W and X. Max
absolute / relative difference for BOTH the value (standardized scalar) and the policy logits:

| case | value max abs Δ | value max rel Δ | logits max abs Δ | logits max rel Δ |
|---|---|---|---|---|
| batched (B=37) | 2.29e-07 | 2.83e-06 | 4.39e-07 | 1.63e-04 |
| single-row (B=1) | 1.60e-08 | 7.41e-08 | 2.96e-07 | 8.07e-05 |
| **worst over 5 seeds × residual ON/OFF** | **2.18e-07** | — | **7.71e-07** | — |

The absolute differences are **float32 representable-precision** over a ~241→256→256→head matmul
chain (~1e-7). The bar is `< 1e-4` abs — comfortably above the float32 floor while still catching
any real algebraic divergence (a wrong residual skip, a missing ReLU, a transposed weight) by
orders of magnitude. The logits' larger *relative* diff is dominated by near-zero logit entries
(rel = |Δ| / (|x| + 1e-6)), so abs is the meaningful bar there. **Numpy inference matches the jit'd
training forward to float32.**

## Batch-1 inference latency — numpy vs jax.jit (justifies keeping numpy)

Real-shaped model (hidden=256, residual ON), the search's actual per-leaf dispatch shape — **fresh
numpy array in, numpy out** (the host↔device round-trip the leaf eval pays), `taskset -c 3`, warmed:

| path | µs / call | ratio vs numpy |
|---|---|---|
| **numpy f32 `predict_both`** | **~73** | 1.0× |
| `jax.jit` forward (fresh-array per-leaf dispatch) | ~226 | **3.1×** |
| `mlp_jax.MlpJaxForward.predict_both` (the full inference drop-in) | ~468 | 6.5× |

jax-jit batch-1 is **3–6× slower** than numpy at the search's single-row dispatch — same regime as
the perf doc's ~10× (the gap narrowed because numpy's f32 path is itself fast and this forward is
lean; the *direction* — numpy wins at batch-1 — is decisive). For contrast, a pre-built on-device
jnp array reused every call is ~1× — but that is the misleading microbench the perf doc warns
against; the search never has the input already on device.

## Smoke (machinery, not quality)

`exit_loop -I 3 -E 40 --residual -W 2 --m 8 --n-sims 24 --eval-n 60 --workers 0 --seed 7`,
`taskset -c 3`:

```
training: JAX/optax Adam (lr=0.001 l2=0.0001); inference: numpy float32
iter 0/3  rate=0.0480 (%VoI=-63)  CE=3.060 vMSE=0.943 R²=0.126  [759 tr | gen 8s train 2s eval 7s]
iter 1/3  rate=0.0550 (%VoI=-51)  CE=3.012 vMSE=0.858 R²=0.232  [369 tr | gen 5s train 0s eval 9s]
iter 2/3  rate=0.0529 (%VoI=-54)  CE=2.893 vMSE=0.976 R²=0.133  [505 tr | gen 6s train 0s eval 6s]
best eval rate 0.0550 at iter 1
```

- JAX/optax training runs; `train 2s` first iter (XLA trace+compile), `0s` after (one cached kernel).
- **Loss decreases**: CE 3.060 → 3.012 → 2.893 (monotone); value R² moves off zero (0.126 → 0.232).
- **Rate climbs** then wobbles (0.0480 → 0.0550 → 0.0529, best 0.0550). 3 iters of 40 episodes at
  m=8/n=24 is a machinery smoke, not a quality run — the climb-then-noise is the expected shape at
  this tiny budget (cf. the perf-doc smoke).
- numpy inference produced the eval rate (eval uses the numpy `GumbelPolicy`); optax/jax import clean.

## Tests

`tests/` — **34 passed**. The equivalence test (3 forward-equivalence cases + the latency bench),
the two JAX-train tests, and the kept numpy-inference / cache-coherence / save-load / Danihelka-
fidelity / feature / value-target tests all green.

## Honest caveats / what didn't port cleanly

- **Training precision moved float64 → float32.** The manual numpy path trained in float64; the JAX
  path trains in float32 (the inference precision). This is deliberate and *closes* a gap rather than
  opening one: the weights numpy inference reads are now exactly the weights the jit'd forward
  optimized, with no f64→f32 truncation step between training and inference. The cost is float32
  optimizer arithmetic; at this net size the JAX f32 and numpy f64 training trajectories track
  closely (over 200 steps on a fixed batch: identical CE to 4 d.p., value MSE 0.0145 jax vs 0.026
  numpy — same descent, minor f32/f64 + optax-vs-manual-Adam differences). If float32 training ever
  proves unstable on the real run, `CHOCO_AZ_DTYPE=float64` flips both training and inference to
  float64 (the equivalence then holds at float64).

- **L2 is COUPLED weight decay folded into the loss** (`½·l2·‖W‖²` on weight matrices only, so its
  gradient flows through Adam's preconditioner), reproducing the numpy path's `g + l2·W` gradient
  exactly — NOT optax's *decoupled* `add_decayed_weights` (which is a different update and would
  decay biases too). Same coefficient, same weights-only scope, same effect. (The earlier draft
  mislabeled this "decoupled"; the math always matched the numpy path — coupled is the correct
  term.)

- **The equivalence diff is float32 roundoff, not zero.** numpy f64 and jax-jit f32 differ at ~1e-7
  abs (value) / ~1e-7–1e-6 abs (logits) — pure float32 representable precision, nowhere near an
  algebraic divergence. No place diverges *beyond* float32 roundoff. (The near-zero-logit relative
  diff up to ~1.6e-4 is the rel-metric's small-denominator artifact, not a real disagreement; abs is
  the meaningful bar.)

- **The ~10× from the perf doc shows as ~3–6× here.** Not a contradiction: the perf doc's numpy
  baseline was ~49µs (an earlier f32 path) and its jax number ~500µs; this pass's numpy is ~73µs and
  the lean jit forward ~226µs (fresh-array). Both say numpy wins decisively at batch-1; the exact
  multiple depends on the numpy baseline and the wrapper overhead. The latency test soft-asserts
  `> 1.5×` to confirm the direction without flaking on host noise.

- **The manual numpy training methods are kept, marked SUPERSEDED, not deleted.** They are off the
  training path (no caller) but document the exact loss/optimizer the JAX path reproduces and the
  equivalence test needs the numpy `_forward` they sit beside. Deleting working, tested code
  mid-migration is a larger blast radius than marking it; a follow-up can excise them once the JAX
  path has a real-run track record.

- **Smoke only.** No long run, no quality claim — the orchestrator drives experiments.

### Out-of-frame hack-rationalization audit — findings + triage

An out-of-frame `hack-rationalization-detector` pass (separate agent, did not see the
implementer's reasoning) reviewed the diff. **Verdict: general** (the f32 cache-coherence
invariant holds with `JaxTrainer` as the new writer: it writes by REBIND — `setattr(net, k,
np.asarray(...))` — which the cache's identity check catches with no new gate; all five enumerated
weight writers — `__init__`, `load`, `_adam_apply`, the exit_loop warm-start, and
`JaxTrainer._write_params` — are caught, four by id, one by `_w_revision`). The equivalence test was
confirmed to compare the **jit'd** (not eager) forward, exercise residual ON and OFF, and the L2 was
confirmed weights-only with the same coefficient as the numpy path. Triage of the findings beyond
the verdict:

- **APPLIED — the equivalence test pinned the wrong numpy forward.** The forward-equivalence cases
  compared numpy float64 `_forward` against the jit'd jax forward, but PRODUCTION inference runs
  `_predict_both_f32` (a third, hand-written float32 forward). `_forward` ↔ `_predict_both_f32` was
  trusted by inspection. Added `test_production_f32_forward_matches_jax_jit`, which pins the float32
  path the search actually runs (de-standardized value + masked softmax) against the jit'd forward,
  closing the gap (numpy-f32 ≈ jax-f32, not just numpy-f64 ≈ jax-f32).
- **DECLINED (intended) — the dropped numpy residual gradient-check.** The brief explicitly directed
  dropping it (autodiff makes the gradient correct-by-construction). The numpy `_residual_backward`
  it covered is now superseded dead code; its correctness no longer gates anything that ships.
- **NOTED — two training implementations coexist, no test asserts they agree.** The numpy
  `train_step` is marked SUPERSEDED and kept; tests still exercise it for "loss reduces." Nothing
  pins the two trainings' *gradients* to each other (the equivalence test pins the two *forwards*).
  This is acceptable because the numpy path has no production caller; the honest follow-up is to
  delete it once the JAX path has a real-run track record (the deferral the doc already names).
- **NOTED — in-code "an audit already blessed this" comments** (in `mlp.py`, pre-existing from the
  residual/perf arcs) are self-certification to be re-checked, not treated as a discharge. This
  audit re-derived the invariant independently rather than trusting those comments.

## Files

- `chocofarm/az/mlp_jax_train.py` — JAX forward + AZ/value loss + `JaxTrainer` (new).
- `chocofarm/az/mlp.py` — manual training methods marked SUPERSEDED (inference unchanged).
- `chocofarm/az/exit_loop.py` — trains via `JaxTrainer`; inference unchanged.
- `chocofarm/az/train_value.py` — Stage-1 value training via `JaxTrainer`.
- `tests/test_jax_equivalence.py` — the float32 equivalence safeguard + batch-1 latency bench (new).
- `tests/test_az_loop.py` — dropped manual grad-check; added the two JAX-train tests.
