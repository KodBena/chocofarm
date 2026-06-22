<!--
docs/notes/leaf-eval-loop/step-0-implementation-map.md
Purpose: Step 0 of the implementation->model diagnostic loop (docs/design/leaf-eval-impl-to-model-
  diagnostic-loop.md §6): the stage->locus->observation map, grounded FIRST-HAND against the live code,
  re-deriving the consultation's §3 (which leaned on a now-stale source the maintainer flagged). The
  maintainer GATES this before Step 1. Read-only — no measurement was taken.
ADR-0005 (a point-in-time run record); ADR-0006 header; ADR-0002 fail-loud; ADR-0012 (cite code:line).
Public Domain (The Unlicense).
-->

# Step 0 — implementation map (gated checkpoint, 2026-06-23)

The first step of the implementation→model loop: before anything is measured, the model's terms are
mapped to *where they actually live in the running code* and *where each could be observed*. Produced
by a commissioned mapping agent that read the live sources first-hand and re-derived the consultation's
§3 against current code (treating the blind-model SYNTHESIS as the stale reference the maintainer
flagged, not authority). Read-only.

## Verification stamp (orchestrator spot-check)

The single most frame-changing finding — **deployed = `StrictBarrier`, and `N` (overcommit) is
structurally dead in production** — I confirmed first-hand (it overturns both the consultation's open
question #2 and the blind-model's "N=1" framing):

- `serve.cpp:178-185` (the `--serve` wire path) builds `WireRunnerConfig` setting only
  `pool_threads`/`pool_batch`, leaving `mode` at its struct default `WireMode::StrictBarrier`
  (`runner_wire_batched.hpp:107`); `main.cpp:111-124` (`--serve`) parses no wire-mode / overcommit flag.
- `trees_per_thread` (N) is read ONLY at `runner_wire_batched.cpp:451` (the *pipelined* driver) and in
  `wire_ab_bench.cpp` (a bench); the strict driver never reads it. `WireMode::PipelinedBucket` is *set*
  only at `wire_ab_bench.cpp:123`, never in `serve.cpp`.
- ⇒ production runs the strict gather-barrier; N is a bench-only knob. **Confirmed.**

---

## 1. The stage→locus→observation table

Model terms from `model_cycletime.py` (Design-B: `cycle_us = T_disp + T_io + B·t_row`;
`serve = 1e6·B/(cycle_us·L)`; `producer = N_gen·R_gen`) and `model_capacity.py` (Design-A: adds `iota`,
`tmsg`). Loci verified first-hand unless noted.

| Term | Locus (file:func) | What it realizes / pays | Observation point (exists? / to-add) | Clean-boundary? | Existence / notes |
| --- | --- | --- | --- | --- | --- |
| **`N_gen`** | `serve.cpp:184` (`wcfg.pool_threads`) for wire; `cpp_executor.py --pool-threads`. NOT cores — pool **threads** T. | producer parallelism. Model: "generator cores"; impl: wire-pool OS threads T (default 4). | `RuntimeConfig::from_env` / `--pool-threads` (a config fact, exists). | Yes (a config value). | **FORM MISMATCH:** model `n_gen=3` "generator cores" (CONSTANT) vs impl `pool_threads=T`; serial `cpp_actor_loop` has no pool at all (single-threaded). "3 cores" is a pinning assumption, not a realized quantity. |
| **`R_gen`** | Prod: `runner.cpp:104 run_episode` (serial) **or** the per-slot machine `runner_wire_batched.cpp:64` (wire). Measured at: `search_runtime_bench.cpp:135 SerialRuntime::run` over `DetNet`. | one core's decision rate, **eval mocked**. | `bench_r_gen.py` (`serial_dps`, exists & built). No live counter on either production driver. | **Confounded / wrong path.** The bench measures the `SearchRuntime` seam with in-process `DetNet` — a THIRD code path, neither production driver. | Bench is shrinkable (median) but does NOT observe the production producer at its real leaf boundary. |
| **`T_disp`** | `inference_server.py:123 jit_forward_core` / `build_staged_forward` — pjit/XLA dispatch floor. | irreducible per-forward XLA dispatch. | `bench_t_disp.py` (fit intercept, exists). Live `FWD dt_us` (`:283`) is the WHOLE forward, not the dispatch split. | Partially. The floor is isolable only offline; the live `FWD` lumps disp+compute. | Bench measures the Python forward graph (same `forward_core` SSOT), not the live in-loop `run_microbatch`. |
| **`tau_io`** | **No single locus.** Drain `inference_server.py:505 _drain` (recv_multipart×T), decode `:264`, encode/scatter `:685` (send_multipart×T). | server drain/decode/encode/scatter, serial between forwards. | `bench_tau_io.py` is a **synthetic** microbench (not the live server). `SERVE_IO_US` is an UNMEASURED prior (20µs). | **No clean boundary.** `DRAIN` event (`:572`) carries `msgs/rows/floor` but **no duration**; `tau_io` = `(DRAIN end → FWD start) − dispatch`, derivable only by adding timestamps. | **FORM FINDING (headline):** `tau_io` is a term the *modeler ADDED* ("Critique A's missing Stage-4"); the impl exposes no such quantity. Binding stage, no clean live observation → NAMED LIMITATION (ADR-0002). |
| **`t_row`** | `inference_server.py:234 run_microbatch` → the `(N_total,in_dim)@W` matmul. | per-row marginal forward cost. | `bench_t_row.py` (fit slope, exists, shrinkable). Live: regress `FWD dt_us` on `width`. | Partially (a fit artifact; observable live only by regression). | Bench is the staged `forward_core` fit, faithful to the matmul by construction. |
| **`B` / `B_op`** | `inference_server.py:548 _drain` (sum of drained `X.shape[0]`), forwarded `:679` (`real`), padded `:637`. Producer: `runner_wire_batched.cpp:317-323` (strict). | rows per forward (serve batch width). | **`FWD real=` field (`:283`) — the live realized rows/forward, EXISTS under `CHOCO_EVENTLOG`.** `mean_rows_per_msg` is rows/**message**, pipelined-path only. | **Yes — `FWD real=` is a clean in-system counter** (least-confounded; it's in the running server). | `bench_b_op` is a PRIOR pin (256). Realized strict-path `B` ≈ T threads × K rows (K=ceil(32/4)=8 ⇒ ~32 at defaults) — far below `B_op=256`. The B-gap is the central operating-point question; `FWD real=` resolves it. |
| **`L` / `LPD`** | Prod wire: the Gumbel tree `fiber_tree.hpp:42 TreeState` (parks one leaf at a time; **no per-decision count**). Measured at: `search_runtime_bench.cpp:160` (`Decision::leaf_requests`). | leaves (net forwards) per recorded decision. | `bench_lpd.py` (`leaf_requests_per_task`, exists). **NO per-decision counter on the production `TreeState`/wire path.** | **Confounded / wrong path** (the `SearchRuntime` seam, not the production `fiber_tree` driver). | `LEAVES_PER_DECISION=500` is an explicit DESIGN PIN (the 76000/152 tautology), not a histogram. |
| **`tmsg`** | `wire_leaf_pool.hpp:183 encode_request` + `:241 decode_response`; codec in `inference_wire` (not read line-by-line). | per-leaf wire-framing cost (TRANSPORT, non-binding). | `bench_tmsg.py` (codec /S, exists, shrinkable). Live: `SUBMIT`/`RECV` spacing under `CHOCO_EVENTLOG_CPP`. | Yes (codec microbench faithful; provably non-binding). | `a_i ≈ 0` — exists and measured; the loop should not spend tokens here. |

## 2. §3 corrections (what the consultant's agent-sourced §3 had wrong or stale)

1. **Deployed `WireMode` is StrictBarrier (§3 hedged it open).** Settled: `serve.cpp:178-186` leaves
   `mode` at the `StrictBarrier` default; `ServeOptions` (`serve.hpp:33-37`) has no `mode` field;
   `main.cpp:114-134` parses none. `PipelinedBucket`/`trees_per_thread` are set only in `wire_ab_bench.cpp`.
   **`N` is structurally dead in production** (only the pipelined driver reads it, `runner_wire_batched.cpp:451`).
2. **`mean_rows_per_msg` is NOT on the production path** (§3 calls it "gold, already emitted"). It is
   emitted ONLY by `run_episodes_wire_pipelined`'s `wire_summary` JSON (`runner_wire_batched.cpp:868-879`),
   consumed only by `cpp/stage_a/*.py` sweeps. The strict production driver emits no `wire_summary`. The
   live rows/forward that DOES exist on production is the Python `FWD real=` field (under `CHOCO_EVENTLOG`).
3. **§3 conflates THREE code paths.** Two production drivers — `cpp_actor_loop.py` → serial local-`NetForward`
   (no wire/server); `cpp_executor.py` → wire (`run_episodes_wire_batched` + `inference_server.py`). The
   model's SERVE cycle describes only the wire path. `R_gen`/`L` are measured on a THIRD path — the
   `SearchRuntime`+`DetNet` bench (`search_runtime_bench`) — neither production driver.
4. **`R_gen` is the `SearchRuntime` seam, not the deployed producer.** Sole-workload-clean for *that* path,
   but not the production producer at its real leaf boundary.
5. **`FWD`/`DRAIN` fields differ from §3.** `FWD` carries `width/real/cold/dt_us` where `dt_us` is the WHOLE
   forward (disp+compute, not split); `DRAIN` carries `msgs/rows/floor` with NO duration. `tau_io` needs
   added timestamps (a passive port), as §7a anticipated.
6. **`mean_rows_per_msg` is rows/MESSAGE, not rows/FORWARD** (`runner_wire_batched.cpp:866-870` distinguishes
   them); the model's `B` is rows/forward = `FWD real=`. §3 equated them.

## 3. Form findings (model stages with no impl locus, or impl costs the model omits)

- **`tau_io` has no implementation locus as a named quantity** (headline). A modeler-added additive term;
  the server realizes it as un-timed serial cost *around* the forward; no clean live observation → NAMED
  LIMITATION (ADR-0002). Its grounding is an UNMEASURED prior; `bench_tau_io` reconstructs it synthetically.
- **The model omits the server-side coalescing floor + `min_forward_rows` (θ) machinery** (`_drain:505-573`).
  Realized `B` is emergent / load-dependent — the additive `cycle_us` with a fixed `B_op` cannot express it.
  (This is where the blind-model's "negative-feedback batch-size fixed point" — the §7a coupling — lives.)
- **`N_gen` is a category mismatch:** model "cores," impl "pool threads" (wire) or "nothing" (serial one-shot).
- **GENERATION and the production producer are on different code paths:** `R_gen`/`L` from the
  `SearchRuntime`+`DetNet` bench; production from `run_episode` (serial) or the wire per-slot machine. A
  fidelity exposure Step 2/3 catches only if it reads the producer at the production boundary — which has no counter.
- **No per-decision `L` counter on the production `fiber_tree`/`TreeState` path** (`fiber_tree.hpp:42-115`).
  L is observable only on the bench `SearchRuntime` path. Adding a per-decision leaf counter = a passive port.

## 4. Deployed-WireMode finding (consultation open question #2)

**The deployed default is StrictBarrier, settled from the code** (the chain in the verification stamp).
**Implication:** the SERVE terms must be evaluated at the StrictBarrier wire path (`cpp_executor`); `N` is
not live (any model treatment of overcommit-N describes a bench config, not production). Realized `B` =
the server's cross-thread coalescing of T strict messages, each ≈ K=`ceil(pool_batch/pool_threads)` rows
(default K=8 ⇒ ~32 rows/forward at the defaults) — far below the model's `B_op=256` full bucket. The
GLOSSARY's "~54 rows/forward at 1 thread" is a *different-config* (1-thread) number. The `B_op=256` vs
realized-`B` gap is the live operating-point question, and `FWD real=` is the counter that resolves it.
*Caveat (measured-vs-interpreted): the K/T arithmetic is DERIVED from `RuntimeConfig::fibers_per_thread`
+ the default `pool_batch=32`/`pool_threads=4`; the actual deployed args (what `cpp_executor.py` passes)
were not traced, and the realized rows/forward is to be READ off `FWD real=` in Step 2, not asserted here.*

## 5. Honest accounting (ADR-0002)

**Read end to end, first-hand:** the diagnostic-loop design + `GLOSSARY.md` (from the loop branch ref);
`model_cycletime.py`, `model_capacity.py`; `inference_server.py` (full); `runner_wire_batched.{hpp,cpp}`;
`serve.{hpp,cpp}`; `runner.cpp`; `main.cpp`; `search_runtime_bench.cpp`; `event_log.hpp`;
`wire_leaf_pool.hpp`; `runtime_config.hpp`; `grounding.py`; `bench_r_gen.py`; `fiber_tree.hpp` TreeState
struct; the docstring heads of the other benches; relevant slices of `cpp_actor_loop.py` / `cpp_executor.py`.

**Read partially / flagged:** `bench_tau_io/b_op/lpd/tmsg.py` bodies (docstrings only); `cpp_executor.py`
(docstring + greps, not full control flow); `cpp_actor_loop.py` (gen loop only).

**Did NOT read (named gaps):** `gumbel.{cpp,hpp}` search internals (so the production per-decision
leaf-count mechanism / tree distinct-node semantics behind `L` rest on `search_runtime.hpp:69` + grounding,
not the search core); `transport.cpp`, `inference_wire.{hpp,py}` codecs (so `tmsg`'s byte-cost is from
grounding, not first-hand); `MANUAL.md`; `search_runtime.cpp`. None affect the locus/observation/WireMode
findings (which rest on files read in full); they would refine the `tmsg` byte-cost and search-internal `L`.

---

## The gate — what the maintainer decides

Per §6 Step-0 gate: *confirm each model stage maps to a real implementation locus and a real observation
point — or flag a stage the implementation does not have (a form finding).* Concretely, this map puts
four decisions to you:

1. **Ratify the form findings** (§3): chiefly that **`tau_io` is a modeler-added term with no clean live
   observation** (a named limitation), and the **`N_gen` core-vs-thread / three-code-path** mismatches. Are
   these the right findings, or do you read any differently?
2. **Accept the §3 corrections** (§2): deployed = StrictBarrier, **N dead in production**;
   `mean_rows_per_msg` not on production; `FWD real=` (not `mean_rows_per_msg`) is the live `B` counter.
3. **The observation reality for Step 2:** the *only* clean existing production counter is `FWD real=` (for
   `B`), and it is **off by default** (`CHOCO_EVENTLOG`). `tau_io`, `L`, and `R_gen` at the production
   boundary each need a passive port or sit on a different path. So Step 2 is not "parse existing telemetry"
   — it needs instrumentation (passive ports), which raises its cost. Confirm that is acceptable, or scope it.
4. **Process:** the loop's docs live on `docs/leaf-eval-impl-to-model-loop`, the code on `feat/issue-control-lab`
   (the loop branch has both, since it forks from feat). Which branch should the loop run from, and should the
   loop docs merge into the main line?

**On your gate:** proceed to **Step 1** (pin the implementation's top-line DPS under a named, frozen config —
which path, which `--pool-threads`/`--pool-batch`), or send this back.
