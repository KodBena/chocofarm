<!--
docs/notes/leaf-eval-loop/step-6-corrected-model.md
Purpose: the CAPSTONE of the leaf-eval impl->model loop. Synthesizes steps 0-5 into the corrected model: the
  implementation underperforms min(stages) because the producer and server are SERIALIZED (depth-1,
  msgs/forward~=1), not pipelined. The impl-faithful bound is the serialized round-trip serve+gen+wire,
  validated vs the lab eventlog and implemented as model_cycletime.serialized_roundtrip_dps.
ADR-0005 point-in-time record; ADR-0006 header; ADR-0009 substantiation (validated); ADR-0012 (aux bound,
  does not touch the driver's f / INPUT_NAMES); claims-measured-vs-interpreted. Public Domain (The Unlicense).
-->

# Step 6 — the corrected model: the serialized producer-server round-trip (2026-06-23)

The impl→model loop's capstone. **Question:** the implementation does ~96–168 dps where the model's
`min(stages)` says ~456 — *why?* Steps 0–5 walked it to a measured, grounded answer and a validated fix.

## The answer (the loop, in one line per step)

- **Not the operating point** (glossary correction): the 456 is the full-bucket idealization; the realized B
  is lower (95–277), but the model evaluated at the realized B *still* over-predicts.
- **Not a padding form-fault** (step 2): the forward computes on the padded width (`max_batch`), which the
  model's `serve_sawtooth` already handles; my "285 dps form fault" was a mis-evaluation.
- **Not the wire** (step 5): the ZMQ round-trip is ~113 µs + 0.3 µs/row (perf: ZMQ ipc's background-IO-thread
  architecture), only **10–17%** of the per-forward gap.
- **The producer↔server SERIALIZATION** (steps 4/5): at `msgs/forward ≈ 1` the producer issues a leaf-batch and
  **waits for the serve reply** before searching on (the leaf eval is on the search's critical path). The two
  stages do **not** overlap, so the realized cycle is the **serialized sum** of the stages, not `min`'s max.

## The corrected model (validated, ADR-0009)

`min(stages)` is the **pipelined ceiling** (depth-∞, full overlap). The implementation-faithful operating
bound is the **serialized round-trip** (depth-1):

```
cycle_us(B) = serve(W) + gen(B) + wire(B)
    serve = T_disp + W·t_row        # padded forward compute (W = max_batch under padmax)
    gen   = 1e6·B / (N_gen·g_core)   # producer's aggregate search time for B leaves
    wire  = 113 + 0.3·B             # ZMQ round-trip (tools/zmq-wire-bench, step-5)
dps = 1e6·B / (cycle·L)
```

| B | serve | gen | wire | period µs | **corrected dps** | real dps | `min(stages)` |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 154 | 2279 | 675 | 159 | 3113 | **98.9** | 99.3 | 134 |
| 276 | 2279 | 1211 | 195 | 3684 | **149.8** | 167.6 | 242 |

The corrected bound reproduces the realized dps (**N=2 within 0.4%, N=4 within 11%**) where `min(stages)`
over-predicts by 35–45%. Implemented as `model_cycletime.serialized_roundtrip_dps` — an **auxiliary** bound
(it does *not* touch the driver's `f` / `INPUT_NAMES`, so equivalence + conformance tests stay green: 26/26).

## The two bounds and the headroom between them

- `serialized_roundtrip_dps` (depth-1): the **impl-faithful operating bound** — the producer waits per batch.
- `min(stages)` / `serve_sawtooth` (depth-∞): the **pipelined ceiling** — full overlap.
- The gap between them is the **overcommit / pipelining headroom**: the depth (overcommit N) interpolates
  depth-1 → depth-∞. The N=4 residual (real 168 > serialized 150) *is* partial overlap kicking in at higher N.
  A **depth-aware MVA** (closed-queueing over the serialized stages) is the refinement that captures the
  N-dependence — the natural next step, and the principled form of the overcommit lever.

## Honest accounting

- **Measured/validated (M):** the per-forward periods (eventlog, step 4), the wire RTT (zmq-wire-bench + perf,
  step 5), the realized dps (sweep, noisy). The corrected model reproduces them.
- **Provisional (I):** the depth/overlap N-dependence is first-order (two clean eventlog points); the MVA
  refinement firms it. `gen(B)` uses the *sole-workload* `g_core`, validated to ~3% at N=2 / ~25% at N=4 —
  i.e. the producer's effective rate degrades a little under overcommit (the same partial-overlap residual).
- **Grounding TODO** (ADR-0002/0008 marked at the site): the wire constants (113, 0.3) belong grounded in
  `grounding.py` (a runnable bench exists), not pinned in the function — a documented deferral, not a silent one.

## What the loop delivered

It turned "the implementation underperforms by ~4×" into: a measured operating-point correction (padmax
compute width), a refuted coordination-idle hypothesis (the wire is 10–17%, perf-attributed), and **the
answer** — the serialized producer↔server round-trip — implemented as a validated, grounded model bound. The
model's optimism was the *pipelining assumption*; the fix is the serialized operating bound plus the overcommit
headroom it exposes. (The dynamic *control* of that overcommit/queue remains `control_lab`'s domain, as scoped.)
