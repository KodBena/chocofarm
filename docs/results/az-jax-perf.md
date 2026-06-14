# AZ search hot-path — float32 + numba + JAX pass (2026-06-14)

An aggressive perf pass on the per-decision Gumbel-AZ search hot path
(`generate_episode` → `decide_with_target` → `_descend` → `_evaluate` → {`features.build`,
`predict_both`}), succeeding the conservative 1.41× bit-exact pass (`docs/results/az-perf.md`).
Mandate: cut per-episode latency hard with float32, numba, and JAX; bit-exactness explicitly NOT
required — only **aggregate behavioral equivalence**.

## Result (end-to-end)

Cold seeded net (hidden=256), m=12 n_sims=48, λ₀=0.0855, 20 episodes, profiler off, warmed
(compile + numba JIT excluded), pinned `taskset -c 2`, single core.

| | ms / episode | speedup vs baseline |
|---|---|---|
| baseline (committed HEAD code, float64 numpy, no numba) | ~640 (621–660, 648 ref) | 1.00× |
| **optimized default** (numba belief + float32 features + float32-numpy MLP) | **~335** (326–348) | **~1.9×** |
| optimized + JAX-single MLP | 1372 | 0.47× (JAX single-eval LOSES — see below) |
| MLP-free floor (Python tree control flow only) | 216 | 3.0× (the ceiling) |

(The baseline was independently re-derived from the committed HEAD code in a clean checkout —
621–660 ms/ep — so the ~1.9× is against verified-baseline, not a remembered number. Run-to-run
variance on a single shared-host core is ±~5%; both bands are reported.)

The pass lands **~1.9×** wall. The ≥3× target is **not** reached on this hot path: an
upper-bound experiment (replace the MLP forward with a zero-cost stub) shows the e2e floor is
**216 ms/ep = 3.00×** — i.e. even an infinitely-fast MLP could not exceed 3×, because the
remaining 216 ms is the **Python tree control flow** (the SH/PUCT recursion, per-edge dict
bookkeeping, `env.apply`/filter), which neither float32 nor numba nor JAX touches. The honest
gap between ~1.9× and 3× is the MLP forward (≈28% of decide time); see the batching analysis.

## Per-hotspot before/after (bench_hotpath.py, median per-call µs, 4001 captured states)

| component | baseline (float64) | optimized | factor | route |
|---|---|---|---|---|
| **belief_reductions** (the nb×44 detector reduction, prior #1) | 145.9 | **15.9** | **9.2×** | numba fused kernel |
| **env.marginals** (per-treasure bit reduction) | 70.4 | **7.3** | **9.6×** | numba kernel |
| **predict_both** (MLP fwd + masked softmax) | 64.4 | **52.9** (f32-numpy) | 1.22× | float32 numpy |
| _evaluate (full leaf: build+mask+fwd+legal) | 103.2 | 94.4 | 1.09× | (gated by MLP) |
| features.build | 13.5 | 16.3 | — | (unchanged; numba folds marginals in) |
| _puct_select | 15.0 | 17.1 | — | unchanged (Python bookkeeping) |
| predict_both **JAX single-eval** | — | 585.3 | **0.11×** | rejected (dispatch) |
| predict_both **JAX batch-48 / item** | — | 12.1 | 5.3× | only batched (see below) |

The two belief reductions — formerly the dominant 216 µs of the leaf — collapse to 23 µs
combined (~9.4×). They are now off the critical list; the leaf is gated by the MLP forward.

Note the numba kernel **fuses** `env.marginals` and the detector reduction into ONE pass over
`bw`, so `features.build` no longer needs a separate `env.marginals` call — the 7.3 µs and
15.9 µs are not additive in the real path (the fused `belief_marg_cover` does both at once;
`marginals_kernel` is the standalone fast path for `env.marginals`'s other callers).

## Which hotspot went jax vs numba vs numpy — and why

- **belief reductions / marginals → numba `@njit`** (`chocofarm/az/kernels.py`). A reduction over
  int64 world-bitmasks (bit tests + integer counts) is exactly what JAX does *not* help —
  there's no large dense matmul, the win is avoiding the `(nb×N)`/`(nb×nD)` temporaries. A numba
  scalar loop fuses both reductions into one pass with no allocation, ~9–10× across the whole
  `|bw|` distribution (15,504 → 1). The kernel is **integer-exact** with the numpy reduction
  (the float dtype does not touch it; only the downstream `cnt/nb` division does). This is the
  maintainer's call exactly: "if jax is no good for `_belief_feats`, numba probably is."

- **MLP forward → float32 numpy** (`ValueMLP._predict_both_f32`), NOT JAX, as the default.
  This is the load-bearing negative result. JAX-CPU jit looks great in a tight microbench
  (~34 µs single-eval, reusing the same on-device array) but the search calls `predict_both`
  **one leaf at a time with fresh numpy arrays**, and there JAX costs **~500 µs/call** — ~10×
  slower than float32-numpy (49 µs). Decomposition (CPU-only backend, `jax.default_backend() ==
  'cpu'`, so no PCIe/device transfer):
    - first call (compile): 328 ms — paid once by `warmup()`, excluded from timing. NOT the cause.
    - steady-state on-device + `block_until_ready`, zero numpy conversion: **204 µs** — pure
      XLA eager-dispatch overhead on a sub-microsecond compute.
    - a trivial jit `add` on-device is 8 µs/call; a single on-device `1×256@256×256` matmul is
      17 µs — vs numpy's 0.5 µs. The dispatch tax, not compute, not transfer, not compilation.
  The crossover where batched JAX beats float32-numpy per item is **batch ≥ 8** (40 µs/item @
  B=8, 10 µs/item @ B=48); the dispatch floor is ~290 µs regardless of batch.

  float32 numpy is *also* the consistency win: it's the parametric `DTYPE` (below), a clean
  ~1.2–1.8× over float64 BLAS at single-row dispatch, with zero new failure surface.

- **The JAX implementation is kept, not deleted** (`chocofarm/az/mlp_jax.py`), and is selectable
  via `GumbelAZSearch(..., use_jax_mlp=True)`. It is the right tool the moment a batched-eval
  seam exists (it wins decisively at batch ≥ 8). The bench measures both single and batch-48 so
  the crossover stays visible.

- **Tree control flow (recursion / PUCT / dict bookkeeping) stays Python.** It does not vectorize
  (each PUCT step depends on stats updated by the previous simulation's backup) and a fully
  `scan`'d tree is not worth it. It is now the e2e ceiling (the 216 ms floor).

## Numeric types — single parametric dtype (`chocofarm/az/dtypes.py`)

A module-level `DTYPE` (default **float32**, `CHOCO_AZ_DTYPE=float64` to switch) threads through
`features.py` (feature vector), `mlp.py` (`predict_both` fast path), and `gumbel_search.py`. One
consistent, switchable precision for the whole hot path. Integer bit-mechanics (the world-set,
cover masks) are always int64 — the knob governs only the real-valued feature/net arithmetic.
Training (`train_step`) stays float64 manual backprop — it is off the per-leaf hot path (once per
iteration over a batch, not per decision), and keeping the optimizer in float64 avoids any
training-stability question. An unrecognised precision request fails loudly (ADR-0002).

## Behavioral-equivalence evidence

The bar (not bit-, not per-decision equality): the optimized policy's fixed-λ₀ rate, mean E[T],
and action distribution statistically indistinguishable from the float64 baseline over N≥300
episodes across ≥2 seeds, within MC CI.

Harness: `chocofarm/az/bench/bench_equivalence.py` — greedy GumbelPolicy rollouts at the ambient
`CHOCO_AZ_DTYPE`, reporting rate = ΣR/ΣT, E[T], and the executed-action histogram.

| metric | float64 (baseline) | float32 (optimized default) |
|---|---|---|
| N (2 seeds × 300) | 600 | 600 |
| fixed-λ₀ rate (ΣR/ΣT) | 0.06180 | 0.06180 |
| mean E[T] (± MC SE) | 76.486 ± 0.531 | 76.486 ± 0.531 |
| action dist (collect / sense / term) | 64.5% / 30.2% / 5.3% | 64.5% / 30.2% / 5.3% |
| per-seed rate (0, 1) | 0.06266, 0.06096 | 0.06266, 0.06096 |
| cross-check seeds 2, 3 (rate) | 0.06192, 0.06156 | 0.06192, 0.06156 |

The float32 and float64 results are **identical** across all four seeds (1200 episodes) — not
merely within CI. At this net and seed set the greedy argmax / Sequential-Halving decisions are
well-separated enough that float32 rounding never flipped one. This is *stronger* than the bar
(indistinguishable-within-CI) requires. float32 does NOT measurably degrade the rate, so it is
kept as the default (the parametric `DTYPE` lets any future regime that *does* degrade fall back
to float64 piecewise).

- **Logic invariants:** the 10 `tests/test_az_loop.py` checks (Danihelka SH budget-exactness /
  executed-action = SH survivor / prior-weighted v_mix, masking, slot bijection) stay **green**
  under both float32 and float64. One assertion was loosened: `test_masked_softmax_zero_on_illegal`
  bounded the probability *sum* deviation at `<1e-9` — a float64-era tolerance. float32 softmax
  sums to 1 within ~1.2e-7 (exactly float32 epsilon); the bound is now `<1e-6`. The LOGIC
  invariant it guards — **exactly zero mass on illegal slots** — is asserted exactly (`== 0.0`)
  and holds bit-for-bit; only the float-precision tolerance moved.
- **Loop smoke:** a short ExIt loop (I=3, E=20, W=2, eval-n=60, hidden=256, seed=7) runs clean
  and learns under float32: eval rate climbs 0.0518 → 0.0596 → 0.0587 (best 0.0596), value R²
  moves off zero, CE/vMSE finite and well-behaved — the numeric/compilation changes don't break
  learning. The same config under `CHOCO_AZ_DTYPE=float64` produces **identical** per-iter rates
  and metrics (0.0518 / 0.0596 / 0.0587), confirming the climb is unperturbed by precision. (Not
  a quality run — 3 iters of 20 episodes is a machinery smoke; the real loop is the orchestrator's.)

## Honest caveats / what didn't pay

- **3× not reached (~1.9×).** The hard ceiling on this hot path is 3.00× (the MLP-free floor):
  216 ms/ep is Python tree control flow that none of float32/numba/JAX addresses. Reaching even
  the 3× ceiling would require the MLP to be ~free; the realistic batched-JAX best (~12 µs/item)
  would land e2e around ~2.4–2.6× *if* a batched seam existed without behavioral regression.

- **JAX single-eval is a net loss and is OFF by default** — the single most important finding.
  Kept selectable (the maintainer's steer) because it wins the moment evals batch (≥8).

- **Batched-forward restructure (approved, NOT taken in this pass).** The only batch-≥8 seam in
  SH+PUCT is **virtual-loss lockstep** (descend K sims against current+virtual-loss stats, batch
  the K leaf evals, back them all up) — the production-AlphaZero technique. It is a genuine
  *algorithmic* change (leaves chosen against stale stats), not a numeric one: it perturbs SH's
  exact per-phase budget accounting and the "executed action = SH survivor" invariant the tests
  pin as the Danihelka-fidelity immune system. Given the Amdahl ceiling (MLP is 28% of decide;
  batching it buys ≤1.55× more, realistically ~1.3×) weighed against re-deriving SH budget
  accounting under virtual loss and re-validating fidelity, this pass leaves it as a documented,
  scoped follow-up rather than rushing a fidelity-risking change. The JAX path is in place so the
  follow-up is wiring, not a rewrite.

- **GPU not pursued.** A ~100k-param MLP at batch 1 is the worst possible GPU case (all dispatch,
  no compute); the CPU-JAX dispatch result above is the same lesson amplified. GPU only helps the
  batched regime, which we don't yet expose.

- **A further safe lever exists in the Python floor:** a numba `assemble` kernel for
  `features.build`'s output-vector assembly prototypes at 1.3 µs vs ~10 µs of numpy-slice Python
  per call (~8 µs/build × ~16k builds). Behavior-preserving, no fidelity risk. Scoped as a
  follow-up to keep this pass's surface coherent.

## Files

- `chocofarm/az/dtypes.py` — parametric `DTYPE` (new).
- `chocofarm/az/kernels.py` — numba belief kernels: `belief_marg_cover` (fused), `marginals_kernel` (new).
- `chocofarm/az/mlp_jax.py` — JAX-jit inference forward, kept + selectable (new).
- `chocofarm/az/features.py` — numba belief reduction; float32 output (`DTYPE`).
- `chocofarm/az/mlp.py` — float32 inference fast path (`_predict_both_f32`); the cache validity
  key is an **identity+revision signature over all weight writers** (`id(W)` catches a rebind from
  `load`/warm-start; `_w_revision` catches the in-place Adam mutation; `y_mean`/`y_std` included),
  so no producer can serve stale weights.
- `chocofarm/az/gumbel_search.py` — leaf eval routes to float32-numpy (default) or JAX (`use_jax_mlp`).
- `chocofarm/az/bench/bench_hotpath.py` — numba/f32/JAX-single/JAX-batch variants added.
- `chocofarm/az/bench/bench_equivalence.py` — behavioral-equivalence rollout harness (new).
- `tests/test_az_loop.py` — one float-tolerance bound loosened for float32 (logic invariant
  intact); new `test_predict_both_cache_coherent_across_writers` pins the cache-coherence invariant.

## Appendix — out-of-frame hack-rationalization audit (verbatim)

An out-of-frame `hack-rationalization-detector` pass (separate agent, did not see the
implementer's reasoning) reviewed the diff. Verdict: **narrower-but-justified, with one
undischarged invariant claim** — it reproduced a *latent* stale-cache bug: the first cut of the
float32 cache invalidated only on the Adam step, so a post-populate weight **rebind**
(`load`/warm-start) would serve stale weights (the auditor reproduced `max|Δp| = 0.0082`). Safe in
today's call order only by accident (rebinds happen before any predict). The audit's named general
fix — *every weight writer invalidates the cache, via an invariant not a per-writer gate* — was
applied: the cache now keys on an identity+revision signature, and
`test_predict_both_cache_coherent_across_writers` pins it (rebind + in-place Adam + y-scale). The
auditor's other findings (the overclaiming comment — corrected; no coherence test — added; the
f64 tolerance no longer asserted tight — accepted as a minor conscious trade since the f32 path is
what ships) are addressed or noted. The auditor confirmed the test-tolerance loosening is a
legitimate float32 consequence with the logic invariant (`== 0.0` illegal mass) preserved exactly,
and that the two deferrals (batched restructure, assemble kernel) cite concrete costs.
