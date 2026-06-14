# AZ MLP — toggleable residual block before the output heads (2026-06-14)

A residual block inserted between the trunk output and the value+policy heads of the pure-numpy
MLP (`chocofarm/az/mlp.py`), behind a constructor toggle (`residual: bool = False`) and a
`--residual` CLI flag (`chocofarm/az/exit_loop.py`). Default OFF → numerically identical to the
pre-residual net, so `residual` is a clean ablation axis. **This is the machinery build + a bounded
gradient-check/smoke — NOT a training result.** The hours-scale real run is the orchestrator's; the
exact command is in §Run.

Worktree `feat/az-residual`, committed but **not pushed**.

---

## Block design

The trunk is unchanged: `in → H → ReLU → H → ReLU`, producing `h` (= `a2`). With the block ON, a
two-layer residual block sits between `h` and the two heads:

```
zr1 = h @ Wr1 + br1
ar1 = ReLU(zr1)
zr2 = ar1 @ Wr2 + br2
head_in = ReLU(h + zr2)        # skip dimension matches: Wr1, Wr2 are H×H
```

The value head and policy head then read `head_in` instead of `h`. With the block OFF,
`head_in is h` (the literal same array object) and there are no block params — the forward is
byte-for-byte the pre-residual matmul chain.

New params: `Wr1 (H×H)`, `br1 (H,)`, `Wr2 (H×H)`, `br2 (H,)`. They are He-initialised and drawn
from the constructor rng **after** the heads, so the trunk and head weights draw the identical rng
sequence whether or not the block is built — that is what makes `residual=False` bit-identical (the
block draws are skipped entirely when OFF; asserted in `test_residual_off_bit_identical_to_baseline`).

### Param count added

`2·(H² + H)` parameters (two H×H matrices + two length-H biases).

| hidden H | params added |
|---|---|
| live net (H=256) | **131 584** = 2·(256² + 256) |
| toy test (H=16) | 544 = 2·(16² + 16) |

### Backward

The block backward (`_residual_backward`) flows the head-input gradient `∂L/∂head_in` through both
paths into `∂L/∂a2`: the skip (`head_in = ReLU(h + zr2)`, so `h` gets the post-ReLU gradient
directly) and the block path (`zr1 = h @ Wr1 + …`). It returns `∂L/∂a2` plus the four block param
grads, which `train_step` / `train_step_value` fold into the existing Adam update — the block params
get Adam state for free via the param registry (`_params()`). With the block OFF the helper is the
identity (returns `dhead` unchanged, no block grads).

## Integration points (all driven by the single param registry `_params()`)

| site | change |
|---|---|
| `_params()` | residual params added when `residual` ON → Adam state + L2 cover them automatically |
| `_forward` | applies the block; cache carries `(…, head_in, res_cache)`; `res_cache=None` when OFF |
| `train_step` / `train_step_value` | both heads' grads sum into `dhead`, then through `_residual_backward` into the trunk |
| f32 inference cache (`_f32_weights` / `_rebuild_f32_cache` / `_predict_both_f32`) | validity gate + rebuild + forward extended to the residual params; coherence stays an invariant over all writers |
| `save` / `load` | `_meta` grows a 4th field (residual flag); `load` rebuilds the block iff flag-ON AND params present, validates Wr*/br* shapes at setup (ADR-0002 fail-loud), and loads a legacy 3-field npz with the block OFF + a log line |
| `parallel.pack_net` / `unpack_net` | now enumerate `net._params().keys()` (not a hardcoded tuple) and carry the `residual` flag in the manifest, set before binding arrays — so the redis-transported worker net rebuilds the block correctly |
| `exit_loop.py` | `--residual` flag, wired into both the cold-net and warm-start (`--init-weights`) paths |

## Verification (bounded, `taskset -c 3`, no training loop)

### 1. Finite-difference gradient check (the load-bearing check)

Analytic grads (the exact computation `train_step` performs) vs central finite differences, on a
small random batch, under the float64 train path. ReLU-kink-crossing entries are excluded (a
pre-activation that flips sign across ±eps — the central difference is not a valid gradient there).
`eps=1e-4` is the float64 sweet spot: an eps sweep (1e-7→1e-4) showed the relative error shrinking
monotonically as eps **grows**, the unambiguous signature of roundoff cancellation in
`f(+eps)−f(−eps)`, not a backprop bug (a real bug is eps-insensitive or grows from truncation).

| config | max relative error |
|---|---|
| residual **ON**, l2=0 | **3.67e-6** |
| residual ON, l2=1e-3 | 1.68e-7 |
| residual OFF, l2=0 | 1.17e-7 |
| residual OFF, l2=1e-3 | 1.14e-7 |

All comfortably under the ~1e-5 bar. The eps sweep that diagnosed the roundoff floor:

| eps | 1e-4 | 3e-5 | 1e-5 | 1e-6 | 1e-7 |
|---|---|---|---|---|---|
| max rel err (residual ON) | 3.7e-6 | 5.3e-6 | 1.5e-5 | 1.1e-4 | 1.4e-3 |

### 2. residual=False ≡ baseline (bit-identical)

- `head_in is a2` (the literal trunk-output object; the identity skip-less path), `res_cache is None`,
  no `Wr*`/`br*` params exist.
- The forward is **byte-identical** (`np.array_equal`) to the explicit pre-residual matmul chain for
  both value and policy logits.
- Two residual=OFF nets at the same seed train bit-identically over multiple `train_step` calls.
- Every trunk/head weight (`W1 b1 W2 b2 Wv bv Wp bp`) is byte-identical between a residual-OFF and a
  residual-ON net at the same seed — the block draws come after the heads and are skipped when OFF,
  so they do not perturb the rng stream.

(An out-of-frame audit additionally confirmed bit-identity against the actual pre-change `main` code,
across seeds — not just against in-file re-derived math.)

### 3. residual=True smoke

- A few `train_step` calls reduce the value MSE (e.g. 0.98 → 0.11) with CE+MSE both finite.
- `predict_both` returns correct shapes (batched and single-row), sums to 1, zero mass on illegal
  slots — via the default float32 fast path, with the residual block applied.
- The float32 cache stays coherent with residual ON across a residual-param rebind.

### 4. Test suite

`tests/test_az_loop.py` — all green (22 checks). Added/strengthened:
`test_residual_gradient_check` (ON+OFF × l2), `test_residual_off_bit_identical_to_baseline` (incl.
the draw-order guard), `test_residual_on_train_step_finite_and_reduces`,
`test_residual_on_cache_coherent_across_writers`, `test_residual_save_load_roundtrip_and_old_npz`
(round-trip + legacy-npz OFF fallback + corrupt-shape fail-loud).

## Run

The orchestrator drives the experiment. To turn the block ON, add `--residual` to the existing
`exit_loop` invocation:

```
python -m chocofarm.az.exit_loop --residual -I 40 -E 300 -W 5 --epochs 2 --m 12 --n-sims 48 \
    --lam 0.0855 --seed 7 --ckpt-dir ckpt
```

Omit `--residual` for the OFF baseline (bit-identical to the pre-change net) — the two together are
the clean A/B ablation. The flag works on both the cold-start and `--init-weights` warm-start paths.
Checkpoints written with `--residual` carry the block params and the residual flag in the npz; a
checkpoint written without it loads block-OFF on any build.
