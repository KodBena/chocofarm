# AZ ExIt loop — parallelism + lower-variance target + belief-resolution features (2026-06-14)

Three changes to the Gumbel-AZ Expert-Iteration loop (`chocofarm/az/exit_loop.py`), aimed at
breaking past the ~0.094 decomp-parity plateau while using all 4 cores. This is an **experiment**,
not a behavior-preserving pass: Parts B and C deliberately change learning. The bar met here is:
the loop trains stably (rate climbs in the smoke), the parallelism is statistically equivalent to
serial (in fact bit-identical aggregate data across worker counts), and the logic-invariant tests
stay green. **This is the machinery build + a bounded smoke — NOT the result.** The hours-scale real
run is the orchestrator's; the exact command is in §E.

Built on the `feat/az-jax-perf` fast code (numba belief kernel, float32 parametric `DTYPE`, the
Gumbel ExIt loop). Worktree `feat/az-parallel-exp`, committed but **not pushed**.

---

## Module map

| module | change |
|---|---|
| `chocofarm/az/parallel.py` | **NEW (Part A).** Persistent core-pinned `multiprocessing` pool; weights + transitions transported over **redis as raw bytes** (no pickle); per-worker numba warmup; worker-count-invariant per-episode seeding. |
| `chocofarm/az/value_target.py` | **NEW (Part B).** The value-target rule extracted to ONE place: `suffix_returns_to_go` (pure MC) + `blended_returns_to_go` (TD(λ)/n-step over the realized λ-return and the search root-value bootstrap). |
| `chocofarm/az/exit_loop.py` | Part A wiring (`--workers`/`--cores`, parallel generate+eval, the serial in-process baseline kept); Part B wiring (`--td-lambda`/`--n-step`, per-iter `target_var` watch); routes the value target through `value_target.py`. |
| `chocofarm/az/gumbel_search.py` | Part B seam: `decide_with_value` returns the search's ~n_sims-averaged root-value bootstrap (`_root_search_value`); `decide_with_target` is now a thin wrapper over the shared `_decide_root`. |
| `chocofarm/az/features.py` | **Part C.** Per-treasure `unc[i] = marg[i]·(1−marg[i])` + global `Σ_uncollected unc[i]`. `feature_dim` 220 → **241**. |
| `chocofarm/az/actions.py` | `legal_mask_from_features` slice offsets updated for the 4N→5N per-treasure block (Part C). |
| `chocofarm/az/feature_response.py` | feature-name map mirrors the new `unc` sub-block + global `sum_unc`. |
| `chocofarm/az/dataset.py` | routes its (decomp-teacher) value target through `value_target.suffix_returns_to_go` — closes the §(f)-audit-flagged two-writer duplication BEFORE adding the blend. |
| `chocofarm/az/bench/bench_value_target.py` | **NEW.** Part B variance probe: rolls episodes once, recomputes the target under several blends, reports var/mc + the E[mean-target] optimism watch + a probe-R². |
| `tests/test_az_loop.py` | +6 tests (Part C: dim/unc/resolved-zero; Part B: MC bit-identity, TD(λ) limits, `decide_with_value` finite bootstrap). All 17 green. |

---

## Part A — 4-core actor/learner episode parallelism

**Design.** Each outer iteration is generate → train → eval → checkpoint. The two episode fan-outs
(E generation, N eval) are embarrassingly parallel — independent rollouts under a frozen net —
while TRAIN stays central in the parent. A persistent pool of `--workers` processes, each pinned to
a distinct core via `os.sched_setaffinity` in the pool initializer (cores `--cores`, default
0,1,2,3), runs the rollouts; each worker JIT-compiles the numba belief kernel once at init
(per-process, amortized). Processes not threads, because the search is GIL-bound pure-Python tree
control flow (the 216ms/ep floor, `az-jax-perf.md`) — threads would serialize. `--workers 0` keeps
the in-process serial path (the A/B baseline); `--workers 1` runs the single-worker pool (the
parallel code path).

**Transport: redis raw bytes, not pickle.** multiprocessing's default result return pickles the
worker's Python objects (per-episode transition records — lists of float32 arrays) back through ONE
pipe; for E=300 that is a large, slow pickle funnelled through a single serialization point that
caps scaling. So neither weights nor results travel as pickle:
- **weights** are packed as raw `ndarray.tobytes()` + a tiny JSON shape/dtype manifest into one
  redis key (`az:w:<run>:<version>`); workers reconstruct via `np.frombuffer` and rebuild the net
  only when the version changes (load is byte-exact, ~ms, no disk).
- **generation results** are packed into contiguous float32 blocks (feats/pis/masks/targets) under a
  per-task redis key; the task returns only `(idx, n_rows)` through the pipe; the parent reads the
  raw bytes back with `np.frombuffer` and deletes the keys. **Zero pickle of array data.**
- **eval results** are scalars `(R, T)` — they ride the pipe directly (no blob needed).

Redis is the memory-cache instance at `127.0.0.1:6380` (1GB, `allkeys-lru`); env-overridable
(`CHOCO_REDIS_HOST/PORT/DB`). Redis unreachable is a loud failure (ADR-0002 posture) — no silent
fallback to a slow path. Weights carry a 1h TTL and a missing payload is a loud `RuntimeError`
(never a silent stale-net serve); result blobs are read+deleted in the same iteration, so the
eviction window is tiny (≤ a few MB live vs 1GB).

### Parallel ≈ serial — bit-identical aggregate data (stronger than the bar)

A task's RNG is folded from `(base_seed, weight_version, kind_tag, episode_idx)`, and the parent
draws the per-episode worlds, so the SAME logical episode draws the SAME stream regardless of worker
count or scheduling. Verified directly: generating the same 24-world batch under **workers=1 and
workers=4** produces **bit-identical aggregate transition multisets** (439 transitions both; target
mean −0.815498, var 0.304852 — identical to 6 digits, over redis transport). The parallelism
perturbs nothing but wall-clock. (This is stronger than the "statistically equivalent within MC
noise" bar — it is exact.) `pack_net`/`unpack_net` are byte-exact (test + standalone check).

### Throughput — and an honest host-contention ceiling

Warmed steady-state (numba compile excluded), full budget m=12/n=48, H=256, cold seeded net,
λ_blend=0.5:

| fan-out | serial (1 core, in-process) | parallel (4 workers, redis) | speedup |
|---|---|---|---|
| E=48 generation | 355 ms/ep | 181 ms/ep | **1.96×** |
| E=120 generation | 355 ms/ep | 183 ms/ep | **1.94×** |

**The ~1.9× does not reach the 3–4× target — and the cause is the host, not the code.** This is an
i5-6600 (4 physical cores) exposed through **libvirt as 4 vCPUs**, and the physical cores are
contended by the host (other VMs, the paused solver daemons' VM, TensorBoard) at run time
(`load average` ~2.2 with only this work). A pure-Python CPU spinner makes the ceiling explicit: 1
instance 5.52s, **4 instances pinned one-per-core 8.59s** — the 4 vCPUs deliver only
`5.52×4/8.59 ≈ 2.6×` aggregate throughput, not 4×, because they time-slice over fewer free physical
cores. The measured 1.9× tracks that ~2.6× hardware ceiling discounted by per-iteration overhead
(weight publish + result gather) and episode-length imbalance over only ~12–30 tasks/worker.

The parallel CODE is correct and confirmed: affinity pins 4 **distinct** vCPUs (probed:
PoolWorker-1..4 → cores 0,1,2,3, one each), the data is worker-count-invariant, redis transport is
byte-exact. On an **uncontended 4-core host** (or off-peak on this one) the same code approaches the
core count — the orchestrator should re-measure the serial-vs-parallel ratio in its run environment
and not assume 1.9× is the code's limit. Stacking the merged 1.9× compute pass with even the
contended 1.9× actor parallelism is already ~3.7× end-to-end; an uncontended host would push the
actor term toward 3–4× (≈7× end-to-end, the original target).

---

## Part B — lower-variance value target (the mechanism lever)

**Problem.** The prior target is the single-rollout realized λ-return-to-go — high variance, which
(per the feature-response finding) collapsed the value to a progress counter that never learned the
geometry/belief structure. **Lever:** blend the realized λ-return with the **search's root-value
bootstrap** — the Gumbel search already produces a ~48-sim-averaged root value at every decision
(`_root_search_value`: the visit-weighted mean Σ_a W[a]/Σ_a N[a] of the root actions' simulated
returns), so the bootstrap is free (no extra rollouts). Two parametrizations of the one idea, in
`value_target.blended_returns_to_go`:
- **TD(λ)** (`--td-lambda ℓ`): forward-view geometric average of all n-step returns, computed by the
  O(D) backward recurrence `G_j = (r_j − λ·dt_j) + ℓ·G_{j+1} + (1−ℓ)·boot[j+1]`, boundary
  `boot[D] = −λ·exit_c`. ℓ=1 → pure MC (prior behavior); ℓ→0 → pure 1-step bootstrap.
- **n-step** (`--n-step n`): realized reward for n steps then bootstrap; n=∞ → pure MC.

The pure-MC limit (ℓ=1 / n=None) is **bit-identical** to the prior suffix rule (asserted by test),
so Part B is opt-in. The suffix rule was extracted to `value_target.py` and `dataset.py` rerouted
through it FIRST, per the `az-exit-loop.md` §(f) audit's own prescription (don't add the blend until
the two-writer duplication is collapsed to one).

### Variance evidence (real net, 568 real decisions, `bench_value_target.py`)

Same episodes, target recomputed under each blend (so the change is the rule, not the trajectory):

| blend | mean | std | var | **var/mc** | probe R² |
|---|---|---|---|---|---|
| mc (ℓ=1) | −0.765 | 0.823 | 0.677 | **1.000** | 0.492 |
| td0.7 | −1.032 | 0.547 | 0.300 | **0.443** | 0.499 |
| td0.5 | −1.076 | 0.488 | 0.239 | **0.353** | 0.481 |
| td0.3 | −1.097 | 0.457 | 0.209 | **0.308** | 0.463 |
| n3 | −1.040 | 0.669 | 0.447 | 0.661 | 0.422 |
| n2 | −1.072 | 0.574 | 0.330 | 0.487 | 0.414 |
| n1 | −1.113 | 0.436 | 0.190 | 0.281 | 0.432 |

The blended targets carry **2–3.5× less variance** than pure MC, monotone in the blend — the
mechanism works. The probe-R² holds (td0.7 edges MC, 0.499 vs 0.492), so the lower-variance target
is at least as fittable.

### HONEST RISK — bootstrap optimism (E[T] / E[mean-target] watch)

The bootstrap reintroduces some of the optimism the pure-MC target (design F4) was chosen to avoid:
the search can over-value under-sampled deep-sensing beliefs, which inflates the target and shows up
as **over-collection (rising E[T])**. Two watches are wired:
- the loop logs **`target_var`** (`yVar`) AND **E[T]** (`ET`) every iteration (history.json + TB);
- the bench reports **E[mean-target] drift vs MC** — a mean drifting UP toward the search estimate
  is the optimism signature.

On the smoke net the bootstrap is **not** optimistic — the mean target drifts DOWN (more negative:
mc −0.765 → td0.3 −1.097), i.e. the early net's λ₀ search-root value is conservative, and the smoke
loop's E[T] does not run away (58 → 42 → 45). This is reassuring but **must be re-watched as the net
trains** — the risk is real in principle and the watches exist precisely so a regression is visible.
Keep the knob tunable: ℓ=1 recovers the un-optimistic pure-MC target exactly. Note the blend
changes the target's MEAN (it estimates a slightly different object — the value via bootstrap vs the
return via MC; equal in expectation only if the bootstrap is unbiased), which the per-iteration value
standardization (`set_value_scale`) absorbs. The bias/variance tradeoff is the whole point: accept a
little bootstrap bias to buy the variance the value head needs to fit geometry/belief structure.

---

## Part C — belief-resolution features (known-vs-unknown encoding)

**Design.** Two features added (threaded through the parametric `DTYPE`):
- per-treasure **`unc[i] = marg[i]·(1−marg[i])`** — the Bernoulli variance of treasure i's presence:
  0 when resolved (marg 0 or 1), 0.25 at maximum doubt (marg 0.5). The bare marginal cannot carry
  this: a resolved-absent treasure (marg 0, unc 0) and a split one (marg 0.5, unc 0.25) both look
  "not here" to a value head reading marg alone.
- global **`Σ_{uncollected} unc[i]`** — the expected number of treasures still in question; a scalar
  "how much belief structure remains to resolve" the value head can read directly (the
  belief-dependent component the high-variance MC target collapsed away from). Sums only over
  UNCOLLECTED treasures (a collected one carries no remaining decision-relevant doubt).

**Wiring.** The per-treasure block grows 4N→5N (`unc` is the 5th sub-block, after `dist`); the
global block grows 5→6 (`Σunc` after the nonempty flag). `feature_dim` 220 → **241** (env-derived,
nothing hardcoded). This re-inits the net's input layer (no warm-start of W1 — fine; the richer
input is re-learned). `legal_mask_from_features` slice offset updated (per-detector block now starts
at 5N not 4N; `available` stays at 2N..3N) — the two legal-mask paths still agree (test). Bench
(`bench/`) is feature-dim-agnostic (derives from env); `feature_response` name map updated; tests
that compute `feature_dim(env)` adapt automatically.

Verified at the root: every marginal is K/N = 5/20 = 0.25 → unc = 0.1875 exactly, Σunc = 20·0.1875
= 3.75 exactly; after a sense chain, resolved treasures (marg 0 or 1) carry unc exactly 0.

---

## Smoke (run THIS only — 4 cores, bounded; NOT the result)

Parallel loop, cold net, all three changes active, I=4, E=40, W=3, full budget m=12/n=48, H=256,
β=2.0, **TD(λ_blend=0.6)**, 4 workers over redis, eval N=40:

```
env: feat_dim=241 action_slots=65
value target: TD(λ_blend=0.6)
parallel actor/learner: 4 workers pinned to cores [0,1,2,3]; weights+transitions over redis (no pickle)
iter 0/4  rate=0.0601 (%VoI=-42) ET=58.2  CE=3.156 vMSE=0.952 R²=0.142 H=0.20 yVar=0.314  [747 tr | gen 8s eval 6s]
iter 1/4  rate=0.0662 (%VoI=-32) ET=49.1  CE=2.952 vMSE=0.929 R²=0.117 H=0.19 yVar=0.267  [605 tr | gen 6s eval 6s]
iter 2/4  rate=0.0693 (%VoI=-27) ET=42.2  CE=2.760 vMSE=0.921 R²=0.124 H=0.21 yVar=0.231  [570 tr | gen 6s eval 5s]
iter 3/4  rate=0.0651 (%VoI=-34) ET=44.9  CE=2.595 vMSE=0.916 R²=0.101 H=0.20 yVar=0.253  [458 tr | gen 5s eval 5s]
best eval rate 0.0693 at iter 2
```

Proves: (a) 4 cores used (the run pins workers 0–3; throughput A/B is §A); (b) **rate climbs**
0.0601 → 0.0693 (best), policy **CE drops monotonically** 3.16 → 2.60, value MSE drops 0.952 →
0.916 — the loop trains stably with the new target + 241-dim features; (c) **lower target variance
is visible** (`yVar` 0.31 → 0.23–0.25 as π′ sharpens; the controlled var/mc table is §B); (d)
feat_dim=241, masks intact, **all 17 logic tests green**. The rate is a 4-iter cold-start smoke —
meaningless in absolute terms; it shows the machinery learns, not where it plateaus.

To (re)generate the bench artifacts (a net + states + the variance probe), pinned + bounded:
```
PY=/home/bork/w/vdc/venvs/generic/bin/python
taskset -c 2 timeout 300 $PY -m chocofarm.az.bench.bench_value_target \
    --net /tmp/az_parexp_smoke_final/latest_net.npz --episodes 60 --seed 11 \
    --blends mc,td0.7,td0.5,td0.3,n3,n2,n1
```

---

## E — exact command for the orchestrator's REAL run (hours-scale; do NOT push)

Warm-start the value head from the E-DECIDE net (or omit for cold), I=40, E=300, W=5, full budget,
β=2.0, **4 workers over redis**, **TD(λ_blend=0.6)** (the smoke's stable setting; ℓ=1 falls back to
pure MC if the bootstrap optimism regresses — watch E[T]). The pool pins cores 0–3; if E-DECIDE/TB
need a dedicated core, set `--cores` and `--workers` to leave it free.

```
PY=/home/bork/w/vdc/venvs/generic/bin/python

timeout 30000 $PY -m chocofarm.az.exit_loop \
    --workers 4 --cores 0,1,2,3 \
    --init-weights /tmp/az_value.npz \
    -I 40 -E 300 -W 5 --epochs 2 --batch 256 \
    --m 12 --n-sims 48 --lr 1e-3 --l2 1e-4 --alpha 1.0 --beta 2.0 \
    --lam 0.0855 --explore-plies 4 --eval-n 200 --eval-seed 12345 --seed 7 \
    --td-lambda 0.6 \
    --tb-logdir tb/az_parexp --ckpt-dir /tmp/az_parexp_ckpt
```

Notes for the orchestrator:
- **Do NOT `taskset` the parent** — the pool pins the workers; let the parent float (it is mostly
  idle during the fan-outs). Re-measure the serial-vs-parallel ratio in YOUR host-contention state
  (§A: the 1.9× here is a contended-vCPU ceiling, not the code's limit).
- redis must be up at `127.0.0.1:6380` (it is the memory-cache instance). The run fails loud at
  pool start if it isn't (no silent slow fallback).
- Knob sweep if the plateau holds: `--td-lambda {1.0,0.6,0.3}` (1.0 = the pre-change pure-MC
  control) and `--n-step {1,2,3}` are the Part B axis; watch `value_R2` (fit) and `ET`
  (over-collection / bootstrap optimism) in history.json / TB.
- Headline = the unbiased Dinkelbach rate of the best checkpoint:
  ```
  $PY -c "from chocofarm.model.env import Environment; from chocofarm.az.mlp import ValueMLP
  from chocofarm.az.gumbel_search import GumbelPolicy
  env=Environment(); net=ValueMLP.load('/tmp/az_parexp_ckpt/latest_net.npz')
  print(env.dinkelbach_rate(GumbelPolicy(net, env, m=12, n_sims=48), final_runs=3000))"
  ```

---

## Honest caveats

- **The smoke decides nothing** (I=4/E=40/N=40, cold net). It is a stability + machinery check; the
  rate, %VoI, R² are tiny-scale artifacts. Whether the three changes actually break the ~0.094
  plateau is the real run's question.
- **~1.9× not 3–4×, due to host vCPU contention** (§A) — a hardware ceiling on this libvirt host
  right now (4 vCPUs delivering ~2.6× pure-CPU), not a code defect. Re-measure on an uncontended
  host. The parallel-≈-serial correctness (bit-identical data) is independent of this and holds.
- **Bootstrap optimism** (Part B, §B) is the named risk: the search bootstrap can over-value
  under-sampled deep-sensing beliefs and inflate E[T]. Not seen on the smoke net (mean drifts
  conservative, E[T] stable), but it can emerge as the net trains — the per-iter `target_var`/`ET`
  watches and the ℓ=1 escape hatch are there for it.
- **numba per-worker warmup**: each of the 4 worker processes JIT-compiles the belief kernel once at
  pool init (~seconds, parallel, one-time per process). Amortized over a 40-iteration run;
  negligible. The first iteration's gen time includes it.
- **Part B changes the estimand's mean** (value-via-bootstrap vs return-via-MC); absorbed by the
  per-iter value standardization, but it means MC and TD runs aren't estimating an identical target
  — the bias/variance tradeoff is deliberate.
- **redis `allkeys-lru` eviction**: weights have a 1h TTL + loud missing-key check; result blobs are
  read+deleted same-iteration. Eviction window is small (≤ a few MB live vs 1GB), but a pathological
  memory-pressure event on the shared cache could in principle evict a weight key mid-iteration — it
  would fail loud (`RuntimeError`), not corrupt silently.
- **Single instance, uncalibrated time model** (the standing caveat): everything is conditioned on
  TELE_OH=12 and symmetric Euclidean travel.

---

## Out-of-frame audit (hack-rationalization-detector)

An out-of-frame `hack-rationalization-detector` pass (separate agent, did not see the
implementer's reasoning) reviewed the diff and ran the deterministic scripts. **Verdict: general**
— correctness is established by invariants over all writers (per-process ownership of the worker
`_W` dict; redis key-space partitioned by `(run, version)` / `(res_token, idx)`; one value-target
module both callers route through; layout-derived mask offsets), not per-writer gates. The prior
two-writer hazard (`az-exit-loop.md` §(f): the duplicated return-to-go rule diverging when a TD(λ)
blend is added) is discharged by **extraction-first** — exactly the general fix the audit looked
for. The auditor independently verified: the legal-mask offset agrees with the env-call mask over
215 belief states; the pure-MC limit is byte-equal to the suffix rule over 2000 episodes; the seed
fold is worker-count-invariant and collision-free at the configured scale; eviction/missing-key
paths fail loud, not silent.

The auditor's findings beyond the verdict, and their disposition:
- **`--init-weights` × Part-C dim mismatch (the one untested seam) — FIXED.** The warm-start path
  copied the old E-DECIDE net's 220-dim `W1` into a 241-dim net, which would crash on the first
  forward. Now detected at setup: if the warm net's `in_dim` ≠ the current `in_dim`, `W1` keeps its
  fresh random init (Part C's "no warm-start of the input layer") with an explicit log line, rather
  than failing opaquely later (ADR-0002 posture). Verified against both a 220→241 and a 241→241
  checkpoint.
- **"n=∞ ⇒ pure MC" is FP-close, not bit-identical — DOC QUALIFIED.** A finite large `n_step` is
  mathematically MC but accumulates term-by-term (~1 ULP off the suffix rule). The bit-identical
  path is `n_step=None`, which dispatches to `suffix_returns_to_go` verbatim. The docstring now says
  so; the asserted/tested bit-identity was always correctly scoped to `n_step is None` / `ℓ=1`.
- **"Serial baseline" A/B is statistical, not bit-identical — ACCEPTED, already precise in the
  report.** `--workers 0` (in-process, `gen_rng` direct) and the pool path (`_task_rng` fold) are
  different streams; only workers=1-vs-4 within the pool is bit-identical (which is what §A claims).
  Worth re-stating here so the two guarantees aren't conflated.
- **`widx` from process name is brittle but fail-soft — NOTED.** The core-pin index is parsed from
  `PoolWorker-N`; a future stdlib rename would collapse all workers to core 0 (a throughput, not
  correctness, regression — affinity is wrapped fail-soft). The probe (workers→cores 0,1,2,3, one
  each) is the standing check; re-probe if the throughput ever looks single-core.
- **Seed-fold collision-freedom is verified-for-config, not proven-invariant; redis read/delete are
  separate non-transactional pipes (harmless under LRU) — NOTED, no action.** Neither is a
  correctness risk (a fold collision would only mildly correlate two episodes, not break
  worker-count invariance; a delete-of-evicted-key is a no-op).

---

## Real-run result (coordinator-driven, 2026-06-14) — FIRST METHOD PAST DECOMP

Full parallel run: cold-start, TD λ_blend=0.6, belief features (dim 241), 4-worker/redis,
I=40 E=300. **Broke past the ~0.094 decomp-parity plateau.**

| | rate | %VoI | E[T] | value R² (train) |
|---|---|---|---|---|
| static floor | 0.0855 | 0% | — | — |
| decomp (exact) | 0.0941 | +14% | — | — |
| prior AZ (plain-MC, 220-d) | ~0.0925 | +12% | ~45 | ~0.62 |
| **this run (TD0.6 + unc feats)** | **0.0973–0.0978** | **+20%** | ~43 | **0.90** |
| clairvoyant ceiling | 0.1454 | +100% | — | — |

- Steady last-10 mean 0.0978; final-net **N=400 confirm 0.09734** (~2σ above decomp +
  sustained 5 consecutive iters); best single eval 0.0994 (+23%).
- E[T] ~43 stable — the TD bootstrap did NOT reintroduce over-collection.
- Per-iter wall ~55s (4-worker/redis + numba/float32) vs ~7min serial original (~7-8×).

### Mechanism (feature-response, final 241-d net; value head)
baseline R² 0.54 (fresh loop-dist, MC target). Importance shift vs the prior 220-d AZ:
`treasure/collected` 0.43→0.50 (still dominant); `detector/dist` ~0.003→**0.028** and
`treasure/dist` ~0.002→0.012 (geometry rose OFF ZERO); `informative` 0.027→0.060,
`p_pos` 0.026→0.042; **`treasure/unc` (the new feature) = 0.002 — near-inert in the value.**

- **Part B (lower-variance TD target) is the cause**: it made the value far more predictive
  and modestly de-blinded it to geometry/sensing — enough for +5 VoI past decomp.
- **Part C (uncertainty feature) free-rode** in the value head (0.002). It MAY aid the policy's
  sense-selection (untested — feature-response is value-head-only), and there was no B-only vs
  B+C ablation, so the gain is not cleanly attributed.

### Caveats
- Modest exceed: +0.0032 over decomp, ~2σ at N=400; still +20% of the +70% ceiling (far below).
- 4-vCPU libvirt host caps the parallel win.
- No B-vs-C ablation; `unc` policy-head effect untested.
- Parallel runs require redis (raw-bytes transport) up at 127.0.0.1:6380; `redis-py` + `numba`
  are now runtime deps of the parallel/fast path.
