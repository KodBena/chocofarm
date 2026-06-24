# throughput-lab: real-generator integration, the fiber verdict, and the scheduling win

**Date:** 2026-06-24. **Branch:** `feat/tlab-real-generators`. **Status:** investigation note (ADR-0005 — a dated, slowly-aging record of what was found and why; not a live task queue).

## Why

The `throughput-lab/` testbed was built with a **synthetic** load generator (a calibrated `x += 1` spin emitting fixed-shape feature batches) to study the producer→boundary→server *transport* in isolation, after the old C++ inference-server coupling proved too tangled to reason about. The open question: does the lab's near-optimal **static** coupling hold up under the **real** generators — the chocofarm Gumbel-AZ search — and where is the actual bottleneck? This note records the answer.

## What was built (the integration)

The chocofarm search is already ADR-0012-clean: it drives leaf evaluation through the injected **`NetEvaluator` port** (`cpp/include/chocofarm/net_evaluator.hpp`), with `SerialRuntime` / `PoolRuntime` and the boost.context fiber machinery (`fiber_tree.hpp`, `wire_parallel_bench`) already present. So integrating it needed only a thin ACL, not a rewrite:

- **`BoundaryNetEvaluator`** (`throughput-lab/cpp/boundary_net_evaluator.hpp`) — bridges `chocofarm::NetEvaluator::predict(x)` to *our* `tlab::Boundary` (B=1 send→recv). We keep our transport; we do **not** reuse the old wire client.
- **`tlab-real-producer`** (`throughput-lab/cpp/real_producer.cpp`) — runs real Gumbel decisions as load. Non-fiber baseline (one `SerialRuntime`/thread, blocking) **and** the fiber multiplexer (`--fibers K`, K `TreeState` fibers/thread; `--driver round-sync|greedy`).
- **Build coupling** — gated behind `-DTLAB_REAL_GENERATOR=ON`; links the prebuilt `chocofarm_core` + boost.context. The default synthetic build stays standalone (ADR-0012 compose). The search emits real 241-wide features (matching the lab `in_dim`); no shape adaptation.

## Findings

### 1. Fibers help, never hurt (the maintainer's open question, answered by data)
Round-synchronous fiber multiplexer, server gathers across the K-in-flight (interleaved replicates, IQR ~1%):

| K (fibers/thread) | 0 | 1 | 4 | 16 | 64 | 128 |
|---|--:|--:|--:|--:|--:|--:|
| leaf-rows/s | 2,718 | 3,029 | 6,729 | 12,776 | 14,745 | 15,825 |

- **K=1 ≈ K=0**: no overhead at matched concurrency — fibers are not a detriment.
- **K↑ → 5.8× by K=128**, server mean batch 1.5 → ~8 — fibers are the lever that unlocks batching.
- **Asymptote, not a mode**: throughput rises and flattens through K=128 (no peak-and-fall); high K is safe.
- **Greedy-async is *not* better** than round-sync (marginally worse at every K) — the overlap hypothesis is refuted; the bottleneck is search compute, not pipeline idle, so pipeline shape can't help. `--driver greedy` kept optional for the record.

### 2. The real workload is generator-bound
At the asymptote the inference server sits at **~58% compute-util — starved** — while the 3 search cores saturate. Real generation is ~16k leaf-rows/s, **~20× below** what the server can serve (~300k synthetic). **The bottleneck is generation, not our transport** — which vindicates the premise (harden the static coupling, distrust the old dynamic coupling).

### 3. Process topology, enumerated then swept
`topology_enum.py` (CP-SAT) is the **single home of the config space** (ADR-0012 P1): it places server / 3 generators / optional surplus onto the 4 vCPUs with scheduling policies, under reasonableness constraints, and emits 40 **orbit-correct** configs (canonical-key dedup over Sym(isolated cores)×Sym(generators); core 0, the housekeeping/IRQ core, never permuted — the boundary that distinguishes "surplus shares the gen on core 0" from "…on an isolated core"). `topology_sweep.py` runs them by composition (`taskset` + `sched_wrap`, no recompiles). Single-rep sweep over all 40:

- **Server on housekeeping core 0 ≈ server on an isolated core (~2–3%)** — the host's `isolcpus`/`irqaffinity` isolation does **not** translate to an in-guest benefit (re-confirms the prior A/B). Keep the server on core 0.
- **The surplus is best placed sharing the *server's* core** (where the idle slack is), not a generator's.
- **`--slice` (server latency), gen BATCH vs OTHER: marginal** — the server isn't latency-starved in this generator-bound regime.

### 4. The win — `SCHED_IDLE` surplus reclaims +18% (the decisive control)
The fragmented ~42% idle on the server's core *is* reclaimable — but only by the right policy. One server@0, 3 gens@1,2,3, surplus@0, policy swept (interleaved replicates, IQR ~0.3%):

| surplus policy | leaf-rows/s | vs no-surplus |
|---|--:|--:|
| none | 15,168 | baseline |
| **`SCHED_IDLE`** | **17,907** | **+18.1%** |
| `nice +19` | 15,936 | +5.1% |
| `SCHED_BATCH` | 15,014 | −1.0% |

This vindicates the ADR-0014 kernel consult's EEVDF diagnosis (`docs/consults/2026-06-24-linux-scheduler-fragmented-slack.md`): on EEVDF, `nice` is a **share weight** that can't gate fragmented slack (+5% only), and `SCHED_BATCH` *removes* wakeup-preempt credit so it contends with the server's forwards (−1%). Only a **runnability gate** — `SCHED_IDLE` (run in true idle, yield instantly) — converts the slack to generation. The path that found it: consult → CP-SAT enumeration → control.

## Privilege

All of the above runs **unprivileged**: `SCHED_IDLE`/`SCHED_BATCH`/positive-nice are self-settable; the EEVDF custom `--slice` is accepted without a capability; negative nice works via an `/etc/security/limits.d` RLIMIT_NICE bump. Only `SCHED_FIFO`/`SCHED_DEADLINE` need `CAP_SYS_NICE`, confined to the audited **`sched_wrap`** helper (`setcap cap_sys_nice+ep`) — *not* the interpreter, *not* root. This kernel (6.19, EEVDF) has **no `latency_nice` field** (`sched_wrap --latency-nice` returns `E2BIG`, fail-loud); the latency lever is the custom slice.

## Recommended configuration (mechanized)

For the real generator load on this 4-vCPU host: **server on core 0; 3 generators on cores 1,2,3; a `SCHED_IDLE` surplus generator also on core 0** (`--fibers ≈ 64`, round-sync). Encoded in `throughput-lab/harness/run_real_best.sh` so the +18% is not rediscovered by hand (ADR-0011 — mechanize the finding).

## Artifacts

- Binaries/code: `real_producer.cpp`, `boundary_net_evaluator.hpp`, `sched_wrap.cpp` (`throughput-lab/cpp/`).
- Harness: `topology_enum.py`, `topology_sweep.py`, `surplus_policy_control.py`, `fiber_sweep.py`, `run_real_best.sh` (`throughput-lab/harness/`).
- Consult: `docs/consults/2026-06-24-linux-scheduler-fragmented-slack.md`.
- Runs (preserved): `~/w/vdc/chocobo/runs/tlab/{fiber-robust,topo-sweep,surplus-policy-ctl}-*`.

## Open / next

The throughput question is answered (generator-bound; `SCHED_IDLE` surplus +18%). The remaining frontier is **dynamic coupling control**: now that the real generators produce realistic `ready`/`inflight` dynamics, wire the server's gather/coalesce seam to expose that state so a `control_lab` gate (static `S_min` floor, then bang-bang / a learned policy) can act on it — the open "does dynamic control beat our near-optimal static coupling?" question. The static coupling characterized here is the baseline that question is measured against.
