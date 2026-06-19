# Which driver is in scope, and is the N-axis reachable in the default config?

(path: out/driver-scope-and-N-reachability.md — Public Domain, The Unlicense)

Focused derivation of one reconciliation-flagged open question (CF-10), purely from
the cleanroom code. All `file:line` refer to the cleanroom tree under
`/home/bork/w/vdc/chocobo/runs/leaf-eval-model-2/cleanroom`.

## The question

Two C++ drivers share one entry point and one config struct:

- `run_episodes_wire_batched` — the **strict-barrier** loop (the body that runs when
  the entry point is *not* dispatched away).
- `run_episodes_wire_pipelined` — the **pipelined** loop, the only one that reads
  `trees_per_thread` (the N-axis) and the in-flight cap `max_inflight_msgs` (D).

The fresh corpus modeled *only* the pipelined arm. If production runs strict-barrier,
the entire N-dependence is **dormant** unless `mode == PipelinedBucket` is explicitly
set. This note settles which arm the composed model must treat as in scope and whether
the N-axis is reachable under the default configuration.

## The dispatch and the default (mechanical facts)

`WireMode` and the config struct:

```
runner_wire_batched.hpp:16   enum class WireMode { StrictBarrier, PipelinedBucket };
runner_wire_batched.hpp:18   struct WireRunnerConfig {
runner_wire_batched.hpp:23       WireMode mode = WireMode::StrictBarrier;   // <-- default
runner_wire_batched.hpp:24       int max_inflight_msgs = 8;
runner_wire_batched.hpp:25       int trees_per_thread = 1;                  // <-- default N = 1
```

The single entry point is `run_episodes_wire_batched`. Its first executable statement
is the dispatch:

```
runner_wire_batched.cpp:44   if (wcfg.mode == WireMode::PipelinedBucket)
runner_wire_batched.cpp:45       return run_episodes_wire_pipelined(env, fb, gc, redis, cfg, wcfg, stats_out);
```

So the control flow is exactly:

- `mode == PipelinedBucket`  → tail-call into the pipelined loop (lines 270–503).
- `mode == StrictBarrier` (or any non-`PipelinedBucket` value of the 2-valued enum)
  → fall through and execute the strict-barrier body (lines 47–268).

The enum has two values (hpp:16); `StrictBarrier` is the in-class default initializer
(hpp:23). Therefore **the default-constructed `WireRunnerConfig` selects the
strict-barrier body**, and the pipelined function is reached *only* when a caller
explicitly assigns `mode = WireMode::PipelinedBucket`.

## Is the N-axis reachable in the strict-barrier body? (No — derive it)

K (per-thread slot count) in the two bodies:

- **Strict-barrier** (cpp:53–57):
  ```
  RuntimeConfig rc;
  rc.thread_pool_size = wcfg.pool_threads;
  rc.batch_size       = wcfg.pool_batch;
  const int T = std::max(1, rc.thread_pool_size);
  const int K = rc.fibers_per_thread();              // cpp:57
  ```
  with `fibers_per_thread() = max(1, ceil(batch_size / T))` (runtime_config.hpp:12–15).
  So `K_strict = ceil(pool_batch / T)` = **base**, with **no N factor**.

- **Pipelined** (cpp:285–286):
  ```
  const int N = std::max(1, wcfg.trees_per_thread);  // cpp:285
  const int K = N * rc.fibers_per_thread();          // cpp:286
  const int D = std::max(1, wcfg.max_inflight_msgs); // cpp:287
  ```
  So `K_pipe = N · base`, and N enters K linearly.

Lexical reach of the N/D parameters (verified by grep over the whole cleanroom, the
only `.cpp`/`.hpp` hits are):

- `wcfg.trees_per_thread` — used **only** at cpp:285 (pipelined). It appears **nowhere**
  in the strict-barrier body (cpp:47–268).
- `wcfg.max_inflight_msgs` (D) — used **only** at cpp:287 (pipelined). Absent from the
  strict-barrier body.

Hence in strict-barrier:

- K, and therefore the per-thread parked-slot population, the gather width, the
  per-msg batch row count, and every downstream transport quantity, are **invariant in
  N**. Changing `trees_per_thread` cannot alter any strict-barrier execution.
- There is no in-flight cap D and no `submitted[]` book-keeping. The loop is a hard
  barrier: gather all parked slots → one `submit_batch` → one blocking `recv_batch` →
  scatter (cpp:224–251). Exactly one message is in flight per thread at a time.

In pipelined:

- K = N·base, D bounds concurrent messages, `submitted[]` (cpp:327) tracks per-slot
  outstanding state, and `issue_one` greedily forms a message from every *ready,
  not-yet-submitted* slot up to the D-message cap (cpp:434–456, 474). N is the axis the
  whole pipelined model is parameterized on.

**Conclusion on reachability:** the N-axis (and the D cap) are reachable **only** on the
pipelined arm. In the default configuration (`mode = StrictBarrier`) the N-dependence is
*structurally dormant*: not merely set to N=1, but never read. Even forcing
`trees_per_thread > 1` while leaving `mode` at its default has **zero** effect, because
the parameter is dead code on the path taken.

## What is "production"? (Honest scoping from the cleanroom)

The cleanroom is self-contained for the transport task but contains **no caller** that
constructs or mutates a `WireRunnerConfig`. Grepping the entire tree for `WireMode`,
`trees_per_thread`, `max_inflight`, `.mode`, `run_episodes_wire*` yields only: the enum
and struct declaration (hpp:16,23,24,25), the two function declarations (hpp:28,32), the
dispatch + the two definitions (cpp:44,45,285,287, plus the function headers/error
strings). The Python files are the *sink* (`stage_a_server.py`, `inference_server.py`)
and a model import; they do not select the C++ driver.

Therefore, *within the cleanroom's evidence*, the only authority on which arm runs is the
struct's in-class default initializer. That default is `StrictBarrier` (hpp:23). I cannot
observe an external production launcher that flips it, so the strongest defensible claim
is:

- **Default-config production runs the strict-barrier driver** (the WireMode default,
  K=base, N-invariant). This is what executes unless a caller outside the cleanroom
  assigns `mode = PipelinedBucket`.
- **The pipelined driver is opt-in**, reached only by that explicit assignment. The
  bench harness named in the task (the E-policy / wakeup knobs in `stage_a_server.py`)
  is the *server* side and is orthogonal to the C++ `mode`; it does not by itself put the
  pipelined arm in scope.

Confidence on the C++ control flow and on N being unreachable in strict-barrier:
**high** — it is a single `if` against a two-valued enum plus a grep-verified absence of
the parameter from the other body. Confidence that *deployed* production literally uses
the default (rather than a launcher outside the cleanroom that sets `PipelinedBucket`):
**medium** — the cleanroom shows no such launcher, so I rely on the in-class default as
the only available signal; an out-of-tree caller could override it, and that file is not
in my world.

## Consequence for the composed model (what a faithful model must state)

1. The composed model must carry **both** arms as distinct regimes selected by the
   single discrete parameter `mode`, not silently assume the pipelined one.
2. **Strict-barrier regime (default):** K = ceil(pool_batch/T), N-invariant; exactly one
   in-flight message per thread (an implicit D=1 barrier); no `submitted[]` /
   `max_inflight_msgs`. The N-axis and D-axis collapse out — any model state or regime
   indexed by N or D is *non-representable* here and including it would make the model
   too permissive for the default config.
3. **Pipelined regime (opt-in `PipelinedBucket`):** K = N·base, D-bounded pipelining,
   `submitted[]` tracking. This is the only regime where the fresh N-parameterized models
   are faithful.
4. A model that presents the N-parameterized pipelined behavior as *the* model of
   "production" without conditioning on `mode == PipelinedBucket` is **unfaithful in the
   too-permissive direction** for the default configuration: it admits N>1 executions the
   default-config code cannot produce. CF-10 must be reinstated as an explicit
   mode-conditioning clause in the composed model.
