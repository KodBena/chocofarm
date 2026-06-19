# Open question: does each over-cap / over-512 overshoot width compile afresh per-shape (recurring tax) or one-time-per-width?

Role: focused derivation of one reconciliation-flagged question for the leaf-eval transport-boundary model.
Public Domain (The Unlicense).

All file:line references are to `/home/bork/w/vdc/chocobo/runs/leaf-eval-model-2/cleanroom`.

---

## 1. Statement of the question

When the drained batch's real row count `B` exceeds the server's pad target (the production
`max_batch`, or the bench bucket policy's largest bucket `512`), the forward runs at an *unpadded,
oversized* row dimension. The question: does each *distinct* such overshoot width recompile via
`jax.jit` (so a large-N run, which produces many different overshoot widths, pays a **recurring**
compile tax across widths), or is the tax **one-time per width**?

The distinction matters for the regime model: if one-time-per-width, the over-cap regime is a
transient throughput dip (the prior's "R5 at N=1" framing) bounded by the *number of distinct widths*;
if per-occurrence, it would degrade throughput persistently.

---

## 2. The single compiled object and its cache key

`jit_forward_core` (`chocofarm/az/inference_server.py:22-34`) holds **one** module-level cache list
`_jit_forward_cache` (`:20`). On first call it builds **one** `jax.jit(_fwd)` object and appends it
(`:33`); every subsequent call reuses `_jit_forward_cache[0]` (`:34`). There is exactly one jitted
function for the whole process lifetime.

`_fwd` (`:29-32`) closes over `_FORWARD_CORE` = `forward_core`
(`:18`, `chocofarm/az/forward.py:3-18`). The row dimension `B` enters as `X.shape[0]` and flows
through `X @ W1` (`forward.py:5`) into every downstream shape, including the output rows
(`jnp.reshape(v_std,(-1,1))`, `jnp.concatenate(...,axis=1)`, `inference_server.py:31-32`). So the
output's abstract shape is a function of `B`.

Crucially, the `jax.jit` wrapper at `:33` is built with **no** `static_argnums`, **no**
`in_shardings`, and **no** shape-polymorphism / `jax.export` wrapping. It is a plain
`jax.jit(_fwd)`. Under JAX's default semantics, a `jax.jit` function specializes (traces +
lowers + compiles) once per distinct *abstract input signature*, where an array argument's
abstract value `ShapedArray(shape, dtype, weak_type)` includes its **concrete shape** — the leading
row dimension `B` is part of the key, not abstracted away. Distinct `B` ⇒ distinct cache key ⇒ fresh
trace + XLA compile; a repeated `B` ⇒ cache hit, no recompile. The compiled executable for each `B`
is retained in the per-`jax.jit`-object cache for the process lifetime.

This is the load-bearing JAX fact. It is confirmed empirically in §6 (a property of `jax.jit`, run
in isolation; the solver and the busy host paths are untouched).

`params` are passed as runtime arguments, not closed over, so a weight reload
(`RedisParamsSource.poll`, `inference_server.py:129-138`) does **not** change input *shapes* and does
**not** invalidate the shape-keyed compile cache; weights of a fixed layout keep the same
`ShapedArray` signature. (Weight *dtype/shape* is fixed by the manifest, `params_from_manifest_blob`
`:75-88`, `.astype(np.float32)` at `:85`.) So reloads are orthogonal to this question.

---

## 3. What row dimension `B'` actually reaches `jax.jit`

The array fed to `forward_fn` is `Xb` in `run_microbatch`
(`inference_server.py:55-61`). Its row count after the pad guard is:

```
B  = total real rows in the drained group            (inference_server.py:56)
B' = pad_to   if (pad_to is not None and pad_to > B) (the pad branch, :58-59)
   = B        otherwise                              (no pad: pad_to is None, or pad_to <= B)
```

The pad guard `:58` pads **only when `pad_to > B`**. When `B >= pad_to` (the over-cap / over-bucket
case), padding is skipped and the **raw** `B' = B` reaches `jax.jit`. This is the exact mechanism by
which a varied overshoot width becomes a varied jit shape.

### 3a. Production server (`InferenceServer._serve_batch`, `:192-200`)

Always calls `run_microbatch(..., pad_to=self._max_batch)` (`:198`). Therefore
`B' = max(B, max_batch)`.

- **In-cap regime** `B <= max_batch`: `B' = max_batch` — a single constant shape. One compile,
  one-time, shared by every in-cap drain.
- **Over-cap regime** `B > max_batch`: `B' = B`, the raw oversized width. The drain
  (`_drain`, `:160-186`) accumulates with the cap checked *before* each `recv_multipart`
  (`while total_rows < self._max_batch:`, `:171`), and a single request adds **all** its rows at once
  (`total_rows += X.shape[0]`, `:185`). So the loop can overshoot: it stops only once `total_rows`
  has crossed `max_batch`, and the last admitted request can carry up to `K` rows (see §4). Distinct
  over-cap `B` ⇒ distinct `B'` ⇒ §2 ⇒ fresh compile.

### 3b. Bench server bucket policy (`StageAServer._serve_batch`, `:54-70`; `_bucket_for`, `:32-37`)

`pad_to` is set per group:
- `e_policy == "padmax"` (`:61-62`): `pad_to = max_batch` — identical to §3a.
- `e_policy == "bucket"` (`:63-64`): `pad_to = _bucket_for(real)`.

`_bucket_for` (`:32-37`) returns the **smallest** bucket `b in {64,256,512}` with `real <= b`; if
`real > 512` it returns `BUCKETS[-1] = 512` (`:37`). Then in `run_microbatch` the guard `:58` sees
`pad_to = 512 <= B = real`, so **no padding** — `B' = real`, the raw over-512 width.

- **In-bucket regime** `real <= 512`: `B'` snaps to one of `{64,256,512}` — three constant shapes,
  three one-time compiles total (the warmup, §5, front-loads all three).
- **Over-512 regime** `real > 512`: `B' = real`, raw. Distinct over-512 `real` ⇒ distinct `B'` ⇒
  fresh compile per distinct width.

The `wakeup` knob (`:57`) changes grouping (`group`: one forward per whole drain; `leaf`: one forward
per queued request) and therefore changes *which* `real` values occur, but each forward still goes
through the same `_bucket_for`/pad-guard logic, so the per-shape compile semantics are identical.

---

## 4. How `B` (and hence the set of over-cap widths) scales with N

The source caps a single message's row count at `K`:

- Per-thread slot count `K = N * fibers_per_thread()` (`runner_wire_batched.cpp:286`), with
  `fibers_per_thread() = ceil(max(1,pool_batch) / max(1,T))`
  (`runtime_config.hpp:12-15`). So `K = N * ceil(pool_batch / T)`.
- `issue_one()` (`runner_wire_batched.cpp:434-452`) gathers **all** currently-ready slots
  (`is_ready`, `:427-430`) into **one** request (one corr-id, `wire_leaf_pool.hpp:76-94`). The gather
  is bounded by `K` (loop `for s in [0,K)`, `:437`). So a single request carries `B_req in [1, K]`
  rows.

Server-side, the drained `B` is the sum of the requests admitted by the drain
(`inference_server.py:171-185`):

- Production: `B in [1, max_batch + (K - 1)]`. Over-cap occurs when accumulated rows cross
  `max_batch`; the overshoot width `B` lands in `(max_batch, max_batch + K - 1]`. As **N grows**, `K`
  grows linearly (`K ∝ N`), so:
  - the **range** of possible over-cap widths widens linearly in N — the worst-case overshoot is
    `max_batch + K - 1 = max_batch + N*ceil(pool_batch/T) - 1`;
  - the **number of distinct integer widths** that can occur is at most `K - 1 = N*ceil(pool_batch/T) - 1`,
    i.e. **O(N)** distinct over-cap shapes.
- Bench bucket: over-512 occurs when `real > 512`; with `wakeup=group` a single drained group's
  `real` can reach the same `B` envelope, so distinct over-512 widths are likewise **O(N)** in count
  (bounded by how far past 512 the gathered `real` can reach, ≤ `max_batch + K - 1` given the same
  drain).

So the *count of distinct widths* is O(N), but each width is hit by the §2 cache the first time and is
free thereafter.

---

## 5. Warmup front-loads only the fixed shapes, never the overshoot widths

`StageAServer.build` calls `server.warmup(sorted(set(BUCKETS) | {max_batch}))`
(`stage_a_server.py:82`), i.e. warmup over `{64,256,512,max_batch}`. `InferenceServer.warmup`
(`:202-217`) runs `run_microbatch(..., pad_to=self._max_batch)` for each `b`, so it pre-compiles the
shapes `{max(b, max_batch) : b in {64,256,512,max_batch}}` — for `max_batch=512` that is just the
single shape `512`. The production server has no automatic warmup (it is opt-in).

Either way, warmup compiles **only fixed in-cap/bucket shapes**. No overshoot width is ever in the
warmup set (the set is constant, independent of N). Therefore every over-cap / over-512 width pays its
compile **at first occurrence during serving**, not at startup.

---

## 6. Z3-free empirical confirmation of the JAX cache semantics (§2)

The claim "distinct `B` ⇒ one fresh compile; repeats ⇒ cache hit; cache persists" is a property of
`jax.jit`, independent of the system under load. Verified in isolation (jax 0.10.1, the project
interpreter), mirroring the one-jitted-object pattern of `jit_forward_core`:

```
seq     [257, 300, 257, 511, 600, 600, 300]
retrace [  1,   1,   0,   1,   1,   0,   0]   (1=fresh compile, 0=cache hit)
total distinct B : 4
total retraces   : 4
jit cache size   : 4
```

The Python trace body ran exactly once per *distinct* leading dimension (first 257, first 300, first
511, first 600), and the repeats (`257`, `600`, `300`) were cache hits — `jit cache size == 4 ==
|distinct B|`. This is the over-cap/over-512 width set behaving as O(N) distinct, one-time-each
compiles.

(No Z3 model was needed for this sub-question: the recurrence is a deterministic property of the
compile cache, not a concurrency interleaving. The two server models' existing Z3 witnesses of the
over-cap forward remain valid; this derivation supplies the compile-cost recurrence they did not
derive.)

---

## 7. Assume-guarantee framing

- **RELY** (on the C++ peer): a single request carries `B_req in [1, K]` rows with
  `K = N*ceil(pool_batch/T)` (`runner_wire_batched.cpp:286`, `runtime_config.hpp:12-15`,
  `wire_leaf_pool.hpp:76-94`); the peer never sends `B_req = 0` (`is_ready` + non-empty gather guard,
  `runner_wire_batched.cpp:444`). This bounds the server's drained `B` and hence the over-cap width
  envelope. Checkable against the cited peer lines.
- **GUARANTEE** (server side): the server runs exactly one `jax.jit` forward per drained group at row
  dimension `B' = max(B, pad_to)` for `pad_to > B`, else `B' = B`
  (`inference_server.py:55-61`); each distinct `B'` triggers exactly one compile and is cached for the
  process lifetime (§2, §6). No per-occurrence recompilation for a width already seen.

---

## 8. Answer

**One-time tax per distinct width — not per occurrence — but the *set* of distinct over-cap /
over-512 widths grows O(N), so a large-N run pays an O(N)-sized bundle of one-time compile taxes,
amortized across the run.**

Mechanism, end to end:
1. The pad guard `inference_server.py:58` skips padding whenever `pad_to <= B`, so an over-cap
   (`B > max_batch`, production) or over-512 (`real > 512`, bucket) batch reaches `jax.jit` at its
   **raw** row dimension `B' = B`.
2. There is exactly one `jax.jit(_fwd)` object (`:20,:33,:34`) and its compile cache is keyed on the
   concrete leading shape; distinct `B'` ⇒ one fresh trace+compile, repeats ⇒ cache hit, retained for
   the process lifetime (§2, confirmed §6).
3. The width `B'` scales with `K = N*ceil(pool_batch/T)` (`runner_wire_batched.cpp:286`); the count of
   distinct reachable over-cap widths is **O(N)** (§4).
4. Warmup never covers overshoot widths (`stage_a_server.py:82`; the set is N-independent), so each
   width's one-time compile lands on its first serving occurrence (§5).

Throughput consequence (forward to the regime model): the over-cap regime is **transiently**
degrading per width (each width amortizes its compile over all later reuses of that exact width), but
because the width set is O(N) and is sampled by nondeterministic source pacing, a large-N run sees a
*prolonged warmup smear* — a sequence of one-time compile spikes spread across the run as new widths
are first hit — rather than a single startup spike (the N=1 "R5" case) or a persistent per-call
penalty. It is one-time-per-width with an O(N) number of widths: not the persistent per-occurrence
tax, and not the bounded single-spike of small N either.

**Confidence: high.** The pad-skip mechanism, the single-cached-jit object, and the O(N) width
envelope are all direct from cited cleanroom lines; the per-shape one-time compile semantics of
`jax.jit` are confirmed empirically (§6) on the project's own jax 0.10.1. The only residual is the
exact *count* of widths actually realized at runtime (bounded above by `K-1`, but the realized subset
depends on the nondeterministic source pacing the model deliberately leaves free) — that affects the
*magnitude* of the smear, not the one-time-per-width conclusion.
