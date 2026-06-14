# AZ search hot-path optimization — 1.41× wall, bit-identical (2026-06-14)

A behavior-preserving optimization pass on the per-decision Gumbel-AZ search hot path
(`generate_episode` → `decide_with_target` → `_descend` → `_evaluate` → {`features.build`,
`predict_both`}). Built by a delegated optimization agent; the agent stopped itself early
(it began foreground-polling a measurement, which the brief forbade) **without committing or
writing this report**, so the coordinator independently verified the working-tree changes,
confirmed they are correct and beneficial, and committed the verified subset + authored this
note.

## Result

| | base (`feat/az-exit-loop` @7fec13e) | optimized | |
|---|---|---|---|
| wall, 20 cold-net episodes (no profiler) | 38.50 s | **27.21 s** | **1.41×** |
| trajectory equivalence (same seed) | — | `max\|ΔG\|=0.0`, `max\|Δπʹ\|=0.0` over 381 decisions | **bit-identical** |
| `tests/test_az_loop.py` | 10/10 | 10/10 | green |

The speedup is real wall-clock with the profiler **off**. The earlier *profiled* comparison
(20.6 s → 23.5 s) was a **cProfile artifact**: the optimizations cut function-call count 63 %
(12.4 M → 4.6 M), so the baseline's profiled time carried far more per-call profiler tax —
profiled times are not comparable across a call-count change this large. (Absolute per-episode
time here is higher than the original profile because this comparison uses a **cold seeded net**
for determinism, not the trained net; opt-vs-base is apples-to-apples regardless.)

## Equivalence (how it was verified)

Identical RNG seed + identical (cold, seed-0) net under base vs optimized → the search is
deterministic, so a correct structural optimization must reproduce **every float**. It does:
the improved-policy targets (πʹ) and value targets (G) are **bit-identical** across all 381
decisions of 20 episodes (`max abs diff = 0.0`), not merely within tolerance. This is a global
proof — it validates every changed file, including `gumbel_search.py`/`actions.py`, not just the
ones inspected by eye. The 10 logic/invariant tests (incl. the Danihelka Sequential-Halving /
v_mix / executed-action invariants) stay green.

The one float-exactness risk flagged during review — that a precomputed distance table built the
"natural" vectorized way (`scipy.pdist`/`cdist`, `np.sqrt(dx²+dy²)`, even `np.hypot`) would
differ from `math.hypot` in the last ULPs — was **avoided**: the table caches the *literal*
`math.hypot(x1−x2, y1−y2)` scalar outputs and `env.d` falls back to a live compute, so it is
bit-identical by construction. The belief-derived feature cache verifies every hit with
`np.array_equal` (no false-positive cache hits).

## What changed

- **`chocofarm/model/env.py`** — precomputed inter-node distance table (cached `math.hypot`,
  bit-exact), eliminating ~2.08 M `math.hypot` calls per profiled batch.
- **`chocofarm/az/features.py`** — per-`loc` distance block memo + per-belief-key cache of the
  belief-derived block (`marg`, `p_pos`, `informative`, sharpness), `np.array_equal`-verified;
  `informative`/`p_pos` via integer `count_nonzero` (exact equivalent of the boolean
  `any`/`mean`).
- **`chocofarm/az/actions.py`** — precomputed slot↔action lookup arrays (was 3.5 M per-edge
  function calls).
- **`chocofarm/az/gumbel_search.py`**, **`chocofarm/az/exit_loop.py`** — consume the above; no
  logic change (proven by the bit-identical trajectories).

## Post-optimization profile (bench harness, median per-call µs)

`chocofarm/az/bench/bench_hotpath.py` times each hot-path component in isolation on captured
representative states (regenerate `states.npz` via `bench/capture_states.py`; not committed,
~30 MB):

```
belief_reductions  158   _puct_select    15
env.marginals       76   features.build  15   (was the ~40% bottleneck)
predict_both        66   filter_treasure  5
                         slot_conversions 0.4  env.d  0.2
```

The win came from collapsing feature-build + bookkeeping (`features.build`, `env.d`,
`slot_conversions` now trivial). The remaining ceiling is the **belief reductions** (`marginals`
+ the nb×44 `p_pos`/`informative` over the world-set) and the **NN** — neither addressed here.

## Honest caveats / what remains

- **Partial pass.** The agent stopped early; this is the *verified-correct, beneficial subset*,
  not a finished optimization campaign. The dominant remaining hotspot (belief reductions over
  up to 15,504 worlds) is untouched — further wins there need belief-statistic caching /
  incremental updates, or a representation change, not micro-tuning.
- The NN is now ~relatively larger but still modest; batching/GPU remains Amdahl-capped until the
  belief reductions are also cut (see the hot-path discussion).
- Speedup measured on cold-net trajectories; the structural wins (distance/slot/feature-build)
  are net-independent, so ~1.4× should hold on the trained-net distribution, though the exact
  factor may shift with cache-hit rates.
