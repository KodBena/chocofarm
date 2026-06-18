<!--
cpp/../docs/design/cpp-wire-generation-roadmap.md
Purpose: the FINAL authoritative implementation roadmap for wiring the C++ `--serve` actor's
  GENERATION step to the JAX batched InferenceServer over ZMQ (Option A fiber-mux leaf transport),
  synthesized from the SUGGEST roadmap + the adversarial CRITIQUE, with every conflict adjudicated by
  reading the code (ADR-0002). Hand this to the implementation engineer; execute phases in order.
Public Domain (The Unlicense).
-->

# FINAL ROADMAP — Wiring the C++ `--serve` Actor's GENERATION to the JAX InferenceServer over ZMQ

**Status:** implementation-ready. Branch `cpp-actor-online-reconfig`. Build dir `cpp/build` (Release, `-O3 -march=native`; do **not** flip `CHOCO_BELIEF_ZDD` there — ODR hazard, `cpp/CMakeLists.txt:43-49`). Read end to end before implementing (ADR-0002). This document is the synthesis of a SUGGEST roadmap and an adversarial CRITIQUE; **every point where the two conflicted was adjudicated by reading the actual code**, and the resolution is stated inline with the file:line that settled it. The CRITIQUE was correct on all of its CRITICAL findings — they are adopted here and the SUGGEST roadmap's optimistic framings are corrected.

---

## 0. MAINTAINER OVERRIDES (2026-06-18 — these SUPERSEDE any conflicting text anywhere below)

Two hard rules from the maintainer. Where the synthesized text below conflicts with these, **THESE WIN.**

**O-1 — The corr-id is a TRANSPORT-envelope frame, NEVER a codec field (ADR-0012 P7: serialization ⊥ transport).**
The leading `[u64 corr-id]` ZMQ frame ahead of `wire::encode_request(features)` (`wire_pool_bench.cpp:191-198`),
round-tripped **opaquely** by the server as the transport envelope (`inference_server.py:283`, `frames[1:-1]`), IS the
mechanism — it never enters the value codec. Do **NOT** add a corr-id field to the value codec. Do **NOT** touch
`inference_wire.*`, `wire_spec`, `test_wire_drift.py`, or the hp schema. The SUGGEST roadmap's codec-field idea is
REJECTED (the critique was right); §4.1's recommended codec-field embodiment is **not** taken.

**O-2 — The local C++ batched runtime is OFF-LIMITS: do not read, `#include`, lift from, depend on, or wire into serve.**
`cpp/src/runner_batched.cpp` (`run_episodes_batched`), `NetForward::predict_batch` (`net.cpp:253`, `net.hpp:93`), and
`docs/design/cpp-local-batched-runtime.md` are the discarded **"C++ does ML ops locally"** dead end — NOT a foundation.
The wire driver MUST source its episode logic from the LEGIT pieces only:
  - **Episode orchestration + record-assembly + pure-MC λ-return** ← the SERIAL `run_episode` (`runner.cpp:40-119`):
    record feat/π/mask + the TERMINATE branch; `env.apply` stepping; per-episode seeding `fold_seed(cfg.seed, idx)`;
    the suffix λ-return `g_j = Σr − λ(Σdt + exit_c)`. Re-derive this as a resumable K-slot structure.
  - **Per-decision search (park/resume at a leaf)** ← the fiber engine `fiber_tree.hpp` (the **RngGumbelSource** ctor,
    `:65-66`) + `fiber_leaf.hpp` (`YieldingNetEvaluator`), driving the UNCHANGED `GumbelAZPolicy::run_search`.
  - **DEALER submit/poll/corr-id transport** ← `wire_pool_bench.cpp:172-236` (lifted into `WireLeafPool`, Phase A).
The leaf is resolved REMOTELY over the DEALER to the JAX `InferenceServer`; the wire driver calls **no** local forward.
**Serve dispatch is BINARY:** `--infer-endpoint` set → `run_episodes_wire_batched`; else → serial `run_episodes`.
There is **NO** `run_episodes_batched` branch in serve. (Phase B below says "lift from `runner_batched.cpp`"; per O-2
that is overridden — re-derive the identical episode logic from the serial `run_episode`; only the source differs.)

---

## 1. Overview & target architecture

**Goal:** replace the serve actor's serial, local `NetForward`-per-leaf generation (~13 dps) with a **DEALER-multiplexed fiber pool feeding the existing JAX batched `InferenceServer` over ZMQ** — the proven `wire_pool_bench.cpp` transport (~50 dps in the composed T×K regime), wired into the `--serve` actor's episode driver. **Option A** (stackful boost.context tree-fibers + `YieldingNetEvaluator`, `fiber_leaf.hpp`/`fiber_tree.hpp`): the real Gumbel search runs **UNCHANGED** inside each fiber and the leaf predict *yields*; the §3.2 Option-B continuation refactor was **not taken and is not needed**. Eval stays in-process Python (ADR-0008). The InferenceServer is the **SSOT batched leaf evaluator** — there is no C++-local batched forward on this path.

**The honest framing (correcting the SUGGEST roadmap, per CRITIQUE A2/B1):** this is **not** a "lift two engines verbatim and merge a single line." Two pre-built pieces exist, but they are **incompatible by construction** and only one half of each is reusable:

- `wire_pool_bench.cpp` is a **throughput bench**, not an episode runner. It drives the **scripted** `TreeState` ctor (`CyclicGumbelSource`, `wire_pool_bench.cpp:203`) over a **constant root state** (`:155-157`, never `env.apply`), decides once, discards (`:228-230`), and **builds no `EpisodeBlocks` — it only counts leaves/decisions** (`:234-235`). Its **only liftable part is the ~45-line DEALER transport loop** (`:172-236`): the corr-id frame, the greedy-async drain, the per-thread disjoint task subset.
- `runner_batched.cpp` (`run_episodes_batched`) is the **single-threaded** local episode driver: K slots, one `EpisodeSlot` per slot owning the persistent rng + world + stable loc/bw/coll + live `EpisodeBuilder`, a **barrier** gather→`net.predict_batch`→resume loop (`:204-242`), with `next_idx`/`written` as **shared single-thread state captured by reference** in `fill`/`on_decided`/`spawn_ply` (`:78-79,86,101`).

**The integration is therefore a genuine new driver** (`run_episodes_wire_batched`) that takes the **episode/RNG/stepping state-machine** (`EpisodeSlot`, `spawn_ply`, `on_decided`, `EpisodeBuilder::finalize`) from `runner_batched.cpp` and re-homes it under a **multi-threaded, greedy-async** control structure with the **DEALER transport** from `wire_pool_bench.cpp` substituted for the local `predict_batch`. The seam between *greedy-async transport* and the *episode driver* is the real new logic — it is not a one-line swap (CRITIQUE B1; see §3 Phase B).

### Data-flow diagram (generation)

```
exit_loop.run  ──generate(net, version, worlds, lam, …)──▶  CppActorExecutor.generate  (cpp_executor.py:102)
                                                              │
  [NEW] in-process JAX InferenceServer thread                │
        endpoint ipc:///tmp/choco-infer-<run>.sock           │  (the SSOT batched leaf eval — JAX/XLA)
        RedisParamsSource(version_supplier=lambda:_pub_ver)   │  single daemon thread, one forward owner
        built in _ensure_actor; closed in close()    ◀────────┤
                                                              │
  publish_weights(net,"gen",version)  THEN  _pub_ver = version   ← (CRITIQUE B2: publish-then-bump)
  reconfigure(cfg)  [HOT search knobs only: m/n_sims/c_*]     │
  generate(GenerateRequest{epoch,version,seed,lam,episodes})  │
                                                              ▼
                              SubprocessActorTransport (JSON-line control over stdin/stdout)
                                                              │
       chocofarm-cpp-runner --serve --run R --infer-endpoint ipc://… --pool-threads T --pool-batch B
                                                              │
                              serve.handle_generate (serve.cpp:122)
                                ├─ [wire path]  --infer-endpoint set → run_episodes_wire_batched(...)
                                │     (NO C++ NetForward reload :158-169; NO local policy build :172-175)
                                └─ [serial]     → run_episodes(...)   ← BINARY dispatch (Override O-2: no local-batched branch)
                                                              │
                  run_episodes_wire_batched  (NEW: cpp/src/runner_wire_batched.cpp)
                    T threads, each: own DEALER socket, K EpisodeSlots, own corr-id→slot map,
                    own disjoint episode subset {tid, tid+T, …}; shared atomics: written, failed, corr_seq
                    per thread greedy-async loop:
                      gather parked slots → dealer.submit(corr_id, EpisodeSlot.feat-leaf) ──────────┐
                      poll() one reply → route by corr-id → resume_with(pred)                       │
                      if ts->running: resubmit;  else on_decided(slot) → fill(slot)                 │
                    on_decided/spawn_ply/EpisodeBuilder LIFTED from runner_batched.cpp (RNG-exact)  │
                    → write_results (immediate on finalize, idx-keyed, redis 6380)                  │
                                                                                                    │
  in-process InferenceServer  ◀──ipc── corr-id frame [u64] + wire::encode_request(features) ────────┘
        opaque envelope round-trip (frames[1:-1], inference_server.py:283) → batched JAX forward → reply
```

---

## 2. Resolved decisions on all 10 integration questions

### Q1 — Where the JAX InferenceServer lives in production
**Decision: `cpp_executor.py` stands up an in-process `InferenceServer` daemon thread over the live net it holds, on an `ipc://` endpoint, replacing BOTH the C++ local `NetForward` AND the role of the redis weight-publish as a C++ read-path.** Why: the executor already holds `self.env`, derives `in_dim`/`n_slots`, and publishes the net every generate (`cpp_executor.py:82-83,131`); standing the server there is the minimal-touch home and keeps eval (also Python, in-process) symmetric. Endpoint: **`ipc:///tmp/choco-infer-<run>.sock`** (namespaced by `self.run`). `inproc` is impossible (the C++ DEALER is a different process); `ipc` beats `tcp` on a single host (skips the loopback TCP stack — exactly the per-RTT cost this change attacks). A transient socket under `/tmp` is fine — the "never `/tmp`" rule (CLAUDE.md) governs *experiment data*, not sockets. **Lifecycle:** built in `_ensure_actor` (`cpp_executor.py:163-174`) **before** the actor's first `ping`; one daemon `threading.Thread(target=server.serve_forever)`, **single-threaded** server (JAX/XLA owns the forward — the R14 / `jaxtrain-deadlock-rca` invariant; no XLA in a worker thread). **Teardown** in `close()` (`cpp_executor.py:250-261`): `server.stop()` → `thread.join(timeout)` → `server.close()`, then reap the actor subprocess. (NB per CRITIQUE B4: lock-step on the control channel already precludes an in-flight generate at `close()` time, so the precise teardown-order rationale is "tidy shutdown," not "avoid mid-generate block"; do not over-justify it.)

### Q2 — Weight versioning: how a new gen version reaches the server
**Decision: a live `RedisParamsSource` driven by a `_published_version` supplier the executor bumps — NOT a frozen `StaticParamsSource` (that is a bench control only).** Build the server over `RedisParamsSource(self._conn, self.run, "gen", version_supplier=lambda: self._published_version, initial_version=…)` (`inference_server.py:184-219`). The server's between-batch `poll()` reloads when the supplier advances (`:210-219`).

**CRITICAL ordering correction (CRITIQUE B2, adopted — the SUGGEST roadmap had this BACKWARDS):** `RedisParamsSource.poll()` calls `read_weights(want)` which **raises loudly on a missing payload** (`inference_server.py:191,204,215`). Therefore the executor MUST **publish the blob first, then bump the supplier**:
```python
self.transport.publish_weights(net, "gen", version, self.run)   # blob exists in redis
self._published_version = version                                # only now can poll() want it
```
The reverse order (bump-then-publish) opens a window where the server's `poll()` wants version V before its blob is written → a loud reload-abort mid-generate. This sequencing replaces `cpp_executor.py:131`'s single `publish_weights` line.

**The two-gate config_epoch/version discipline is PRESERVED unchanged.** The control protocol still carries `version`/`config_epoch` (`serve.cpp:131-148`, `actor_transport.py:229-246`); the executor still asserts the echoed epoch/version (`actor_transport.py:238-244`). The change to the C++ actor: gate-2's *local NetForward reload* (`serve.cpp:158-169`) is **skipped on the wire path** (the leaf is remote); the actor still stamps `version` into its reply token (`serve.cpp:194`). **Net-version straddle:** in production the server reloads only between generates, and the actor's generate is **lock-step on the control channel** — the generate reply is sent only after `run_episodes_wire_batched` returns, i.e. after every leaf of generate N resolves (`serve.cpp:187-195`, `actor_transport.py:206-217`). With publish-then-bump (above), there is exactly one reload boundary per generate and no in-flight straddle. **This is tighter than `cpp-batched-search.md §3.7` worried** — but see Open Risk OR-3 (weight-blob LRU eviction) for the residual hazard.

### Q3 — Result blocks: where the episode loop lives (the genuine gap)
**Decision: lift the `EpisodeSlot` / `spawn_ply` / `on_decided` / `EpisodeBuilder::finalize` machinery from `runner_batched.cpp:39-190` into the new wire driver, and substitute ONLY the leaf-resolution step.** The honest gap (CRITIQUE A2): `wire_pool_bench.cpp` builds **no** `EpisodeBlocks`. The per-episode loop — record decision (feat/π/mask, the TERMINATE branch), `env.apply`, the horizon/empty-belief guard, `finalize`'s pure-MC λ-return suffix target — lives in `on_decided` (`runner_batched.cpp:101-157`) + `EpisodeBuilder` (`runner.hpp:78`, `runner.cpp:51-102`), lifted **unchanged in logic** but re-homed (its closures' capture set changes under multi-threading — see Phase B). The substitution: `runner_batched.cpp:204-242`'s "gather → `net.predict_batch` → resume" **barrier** becomes a per-thread **greedy-async** submit/poll/resume loop over the DEALER.

### Q4 — Corr-id routing: echoed opaque envelope frame, already implemented
**Decision: an echoed `u64` corr-id as a leading ZMQ *transport-envelope* frame — ALREADY IMPLEMENTED on both sides; NO wire-codec change.** C++ stamps a globally-unique `u64` corr-id ahead of the payload (`wire_pool_bench.cpp:191-198`: `zmq_send(corr, SNDMORE); zmq_send(payload)`) and matches replies by it, failing loud on an unknown corr-id (`:212-220`). The server round-trips it **opaquely as the transport envelope** (`inference_server.py:82-85,283,312-314`: `envelope = frames[1:-1]`, echoed verbatim, **never parsed**). The corr-id is a *transport* concern kept OUT of the value codec (`inference_wire.*` unchanged) — ADR-0012 P7 serialization⊥transport.

**Provenance correction (CRITIQUE A3, adopted):** `cpp-search-runtime.md §4.1` *recommended* a **codec field** (a new `wire_spec` SSOT field covered by `test_wire_drift.py`). The implementers **diverged from that recommended embodiment** and used the transport-envelope mechanism instead. State it precisely: the implementation **diverged from §4.1's recommended codec-field embodiment and closed §8.1 with a transport envelope** — do **not** write "reached the §4.1 mechanism" (it didn't) nor "supersedes §4.1" (imprecise). The envelope is genuinely better (zero codec surface) and **this is P7 (don't re-author the wire codec): leave `wire_spec`/`inference_wire`/`test_wire_drift` untouched.**

### Q5 — Per-tree-in-flight==1, failure routing, abort granularity
- **Per-tree-in-flight==1:** structural and free — `TreeState` parks at exactly one leaf and cannot submit a second until `resume_with` (`fiber_tree.hpp:21,103-107`); the driver resubmits a slot only after resuming it (`wire_pool_bench.cpp:226-227`). No runtime check.
- **Failure routing:** fail loud, **NEVER** a zero/stale leaf substitution (ADR-0002). `wire_pool_bench.cpp` already sets `failed` and breaks on a recv error, malformed envelope, decode failure, or unknown corr-id (`:215-219`). In the production driver this becomes a `std::expected<int,Error>` **whole-generate abort** (matching `run_episodes_batched`'s contract, `runner_batched.cpp:223,150`) propagated up as `ERR_GENERATE_FAILED` (`serve.cpp:188-189`) — never a partial write.
- **Whole-generate-abort, not per-task-expected:** loudest (ADR-0002); matches the established `run_episodes`/`run_episodes_batched` contract (`runner.cpp:214`, `runner_batched.cpp:150,223`); keeps `written` all-or-nothing so the executor's written-vs-read reconciliation (`cpp_executor.py:156-161`) holds. The `§8.2` "promote to per-task for long self-play" is **deferred, flagged**.
- **Timeout granularity (honest statement, CRITIQUE D1):** the per-thread DEALER sets `ZMQ_RCVTIMEO` (`wire_pool_bench.cpp:176`). The **actual behavior is: one timed-out leaf kills the WHOLE generate** after `timeout_ms` (the thread blocking in `recv` with K−1 other slots unservable → `failed` → abort). This is acceptable under ADR-0002 (loud, no hang) but it is **not** per-tree attribution. A per-outstanding-corr-id submit-timestamp for *attribution* is a deferred refinement; do not claim per-tree timeout exists. Default `timeout_ms=15000`.

### Q6 — Control protocol / ActorConfig changes
**Decision (CRITIQUE C4/C5, adjudicated by reading `runtime_config.hpp` + `hp/schema.py`): pass the endpoint AND the pool knobs (threads, batch) as `--serve` STARTUP ARGS — do NOT add them to `ActorConfig`.** Why, decisively:
- Every `ActorConfig` field's Mut class is **read from the hp schema** (`SearchConfig`/`EnvConfig` metadata via `_mut_of`, `actor_config.py:90-108`). `batch`/`threads` are **runtime/parallelism knobs whose ONE home is `RuntimeConfig`** (`runtime_config.hpp:23-24`), **not** search knobs. Routing them through `SearchConfig` to satisfy the Mut lookup is exactly the "second vocabulary" **ADR-0012 P1 forbids**, and there is no `RuntimeConfig` group in the schema (only a `ParallelConfig` at `schema.py:218`, which governs the *Python* pool, not the C++ env-sourced knobs).
- `RuntimeConfig` **already** reads `CHOCO_POOL_THREADS`/`CHOCO_POOL_BATCH` from the env at construction (`runtime_config.hpp:38-43`) — the live-source seam already exists. The `--serve` args simply override those (the same pattern `wire_pool_bench.cpp:144-145` uses for `--threads`/`--batch`).
- The endpoint is **fixed for the server's life** (a new endpoint is a new server) — INSTANCE-like, so a startup arg is its correct home, not a HOT/reconfigurable field.

**Concretely:** add to `main.cpp` (`:121-138`) three `--serve` args — `--infer-endpoint <ipc://…>`, `--pool-threads N`, `--pool-batch N` — parsed alongside `--run`, threaded into `serve(...)` (Q8). `cpp_executor.py` passes them via `extra_args` (`cpp_executor.py:171`, `actor_transport.py:131-144`). **This touches ZERO drift-net surface — `ActorConfig`, `actor_config.hpp`, `test_wire_drift.py`, and the schema are UNCHANGED.** Online K-retuning (riding `ActorConfig` HOT) is **deferred until Phase F proves it worth the drift-net surface** (ADR-0009: do not build the reconfigure hook before the measure says T×K composition even helps). The config_epoch gate and the 7 HOT search knobs (`m/n_sims/c_*`) ride `ActorConfig` exactly as today.

### Q7 — Parity/test gate: the acceptance bar
**Decision: aggregate BEHAVIORAL equivalence (`cpp-search-runtime.md §7.3`), NOT per-decision byte-identity.** The wire path makes this *mandatory*: batch-composition roundoff (`cpp-batched-search.md §2.2`) means *which* trees co-batch a leaf depends on arrival timing under the greedy drain, so a near-tie (`§1.3`) can legitimately flip an SH survivor — per-decision byte-identity across runtimes is **wrong to require**. Four composing layers:
1. **Net-forward parity (inherited):** the wire forward is pinned max|Δ|<1e-4, measured ~e-7 (`tests/test_zmq_net_cpp.py`, `cpp/parity/wire_bench.py`). Unchanged — the runtime does not touch leaf numerics.
2. **Structural determinism (single-tree, canned leaves):** **MUST use the RNG `TreeState` ctor with a fixed seed** (`fiber_tree.hpp:65-66`), **NOT** `fiber_proto`'s scripted source (CRITIQUE F1 — the wire driver drives the RNG arm; the scripted check validates the wrong source-selection branch, `fiber_tree.hpp:80-82`). Assert the `NeedsLeaf` feature-row sequence + final `Decision.action` matches the in-process serial reference for matched seed + canned `NetPrediction`s.
3. **The three Danihelka invariants:** `test_executed_action_is_sh_survivor`, `test_vmix_prior_weighted`, `test_sequential_halving_spends_full_budget` (`chocofarm-gumbel-dump`, `cpp/parity/gumbel_*.py`) — `run_search` is byte-untouched (Option A), so they must still pass.
4. **Aggregate behavioral equivalence (the cross-runtime bar):** N≥300 decisions, ≥2 seeds; action-distribution + improved-π statistics statistically indistinguishable across {serial `run_episodes`, local `run_episodes_batched`, wire `run_episodes_wire_batched`} within Monte-Carlo CI (report the MC standard error). Plus a **batch-composition stress** (vary K/threads, inject arrival jitter; assert aggregate stays inside CI).

**Structural cross-check correction (CRITIQUE A1, adopted — the SUGGEST roadmap cited a phantom field):** `GumbelAZPolicy::Decision` (`gumbel.hpp:231-236`) has **only `action`, `improved`, `n_spent`, `survivor_slot` — there is NO `leaf_requests`.** `leaf_requests` lives on `SearchRuntime::Decision` (`search_runtime.hpp:69`), a **different** struct the fiber/wire path does **not** produce. The "leaf count per decision must match across runtimes" structural check is still valid and valuable, but it MUST be derived **driver-side**: the wire driver owns the submit loop, so it counts the leaves it submits per slot per decision directly (a per-slot counter reset at `spawn_ply`). A mismatched count for matched seeds is a **driver bug**, not roundoff (`cpp-search-runtime.md §7.2`). Do not read a `Decision` field for this. (Note: `CountingNetEvaluator` at `search_runtime.hpp:78` is the SearchRuntime mechanism for the same observable — not available here because the fiber path bypasses `SearchRuntime`; the driver-side counter is the fiber-path equivalent.)

**Acceptance bar:** layers 1-3 green (exact where exact); layer 4 within MC CI; the driver-side per-decision leaf count identical across runtimes for matched seeds. A per-decision divergence at a near-tie is a *correct* result, not a failure.

### Q8 — Build/CMake (see §5 for the diff)
The runner gains the mode via a **dispatch in `serve.handle_generate`** keyed on the `--infer-endpoint` startup arg being present (additive, ADR-0004 minimal-touch; serial/local/one-shot paths untouched). `serve()` gains the endpoint + pool knobs as parameters (a real signature change to `serve.hpp:31`, with `main.cpp:125` updated in lockstep — its only caller). **zmq is already linked PUBLIC into `chocofarm_core` (`CMakeLists.txt:147`), so it is transitive — the SUGGEST roadmap's "undefined-reference if you skip the zmq link" claim is FALSE (CRITIQUE E1).** The serve/runner target (`chocofarm-cpp-runner`, `CMakeLists.txt:161`) needs only **`${BOOST_CONTEXT_LINK}`** added (boost.context is NOT in core, `:344-348`); zmq comes free. The new `runner_wire_batched.cpp` is compiled **per-executable** (kept out of core, mirroring `runner_batched.cpp` at `:353`).

### Q9 — Sequencing
**Re-sequenced per CRITIQUE C2/G1 to put a cheap throughput probe BEFORE the expensive driver build (ADR-0009 measure-first):** (P0) throughput probe — does T×K wire beat ~13 dps at all, using the EXISTING `wire_pool_bench` against a server stood up over the *production-hidden-256* net? → GO/NO-GO-lite → (A) extract the reusable wire leaf-resolver → (B) `run_episodes_wire_batched` (the multi-thread + greedy-async + episode-driver merge) → (C) the cross-runtime parity check (layer 4) → (D+E together — they are interdependent, CRITIQUE G2) executor server standup + serve dispatch + startup args → (F) exit_loop end-to-end + the 4-iter measurement. Detailed in §3.

### Q10 — What stays
- **Eval in-process Python (ADR-0008):** `CppActorExecutor.evaluate` (`cpp_executor.py:227-248`) untouched — pure-Python `GumbelPolicy`, no subprocess, no server. The InferenceServer serves **generation only**.
- **The Part-B-blend and explore-plies>0 fail-loud guards STAY** (`cpp_executor.py:114-129`) — unchanged.
- **The pure-MC value target STAYS:** `EpisodeBuilder::finalize`'s suffix-return math (`runner.cpp:51-102`) lifted verbatim.
- **The written-vs-read reconciliation STAYS** (`cpp_executor.py:156-161`) — the whole-generate-abort contract (Q5) keeps `written` all-or-nothing.

---

## 3. Phased plan

Each phase lands and is independently verifiable (ADR-0009). The measure-first probe (P0) gates the expensive build.

### Phase P0 — throughput probe (the measure-first gate; CRITIQUE C2/G1)
**Goal:** before building any new driver, confirm T×K wire generation can beat the serial ~13 dps using the **existing** `wire_pool_bench` + `wire_server.py`, but with the server stood up over the **production net geometry** (`--hidden 256`, the real instance/faces). This is cheap (both binaries exist) and is the GO/NO-GO-lite.
**Files touched:** none (or a tiny shim to stand `wire_server.py` over the hidden-256 net). 
**Checkpoint:** `wire-pool-bench --instance … --faces … --endpoint ipc://… --threads {1,2,4} --batch {16,32,64} --m 24 --n-sims 256` reports `pool_dps` materially above ~13 at some (T,B) on the 4-vCPU host, with the achieved server batch `B` reported. **GO** → proceed to Phase A. **NO-GO** (the 4-vCPU ~1.9× wall + ipc RTT cap it at/below serial) → record honestly (ADR-0009), stop; do not build the driver.
**Risk:** the probe drives the scripted/constant-root source, so its dps is an *upper* bound on the real episode driver (no `env.apply`, no per-ply respawn). A marginal probe result is a NO-GO. Mitigation: require a comfortable margin, not a tie.

### Phase A — extract the reusable wire leaf-resolver
**Goal:** lift the DEALER + corr-id submit/poll/route mechanics out of `wire_pool_bench.cpp`'s `worker` lambda into a reusable header so the bench and the production driver share one home (P1).
**Files created:** `cpp/include/chocofarm/wire_leaf_pool.hpp` — a **per-thread** DEALER resolver, RAII/move-only, `create()`-factory (a throwing ctor cannot return a value — P9):
```cpp
class WireLeafPool final {
  [[nodiscard]] static std::expected<WireLeafPool,Error>
      create(void* zctx, const std::string& endpoint, int timeout_ms, std::atomic<uint64_t>& corr_seq);
  // stamp a unique corr-id, send [corr | wire::encode_request(features)], track corr→slot (wire_pool_bench.cpp:191-198):
  [[nodiscard]] std::expected<void,Error> submit(int slot, std::span<const float> features);
  // block up to timeout for ONE reply; decode; route by corr-id to its slot (wire_pool_bench.cpp:212-220):
  [[nodiscard]] std::expected<Completion,Error> poll();   // Completion{ int slot; NetPrediction pred; }
  [[nodiscard]] bool any_outstanding() const;
};
```
**Files edited:** `cpp/src/wire_pool_bench.cpp` — re-expressed over `WireLeafPool` (behaviour-preserving). **Mark the header's rationale HONESTLY (CRITIQUE C1):** `WireLeafPool` is **per-thread** (its `inflight` map is single-thread state); the `wire_pool_bench.cpp:168-170` "a future shared registry / work-stealing keys on the global corr-id unchanged" is an **aspiration, not delivered here** — do not build the cross-thread migration hook (ADR-0009: not before the measure says T×K helps). The corr-id atomic stays process-global (passed by reference) so a future shared registry *could* key on it.
**Checkpoint:** `cmake --build cpp/build --target chocofarm-wire-pool-bench` clean under `-Wall -Wextra`; re-run reproduces the prior `pool_dps`/`leaves`/`decided` (extraction changed nothing).
**Risk:** low.

### Phase B — `run_episodes_wire_batched` (the genuine integration; CRITIQUE B1)
**Goal:** the core — drive K episodes per thread, T threads, over the DEALER, emitting `EpisodeBlocks` per episode exactly as `run_episodes`. **This is a genuine new control structure, not a one-line swap.**
**Files created:** `cpp/include/chocofarm/runner_wire_batched.hpp`, `cpp/src/runner_wire_batched.cpp`:
```cpp
struct WireRunnerConfig { std::string endpoint; int pool_threads = 4; int pool_batch = 32; int timeout_ms = 15000; };
// Same contract / redis writes / `written` semantics as run_episodes (runner.cpp:179). Leaf eval is REMOTE
// (the JAX server) — NO NetEvaluator argument; each TreeState's policy holds the YieldingNetEvaluator.
[[nodiscard]] std::expected<int, Error>
run_episodes_wire_batched(const Environment& env, const FeatureBuilder& fb, const GumbelConfig& gc,
                          RedisClient& redis, const RunnerConfig& cfg, const WireRunnerConfig& wcfg,
                          std::ostream* stats_out = nullptr);
```
**Source the episode logic from the SERIAL `run_episode` (`runner.cpp:40-119`), NOT from `runner_batched.cpp` (Override O-2).** Re-derive the per-ply record-assembly (feat/π/mask + the TERMINATE branch), the `env.apply` stepping, the per-episode seeding `fold_seed(cfg.seed, idx)`, and the pure-MC λ-return suffix target directly from `run_episode`'s inline logic, expressed as a resumable per-slot `EpisodeSlot` structure with its own persistent rng + stable loc/bw/coll + a record accumulator. The logic is identical to what the off-limits `runner_batched.cpp` holds; only the source differs — do not read or depend on that file. **The genuinely new control structure (the parts the SUGGEST roadmap waved at):**
- **Multi-thread state.** `run_episodes_batched`'s `next_idx`/`written` are shared single-thread locals captured by reference (`runner_batched.cpp:78-79`). Here: each thread owns a **disjoint episode subset** `{tid, tid+T, …}` (its own `next_idx` over that subset, `wire_pool_bench.cpp:182-184`), its own slot array, its own `WireLeafPool`. `written` is a shared `std::atomic<int>` summed across threads (`wire_pool_bench.cpp:165,234`); any thread's `Error` sets a shared `std::atomic<bool> failed` and the whole pass returns `std::unexpected` (Q5). **No tree migrates between threads** — single-writer-per-slot is structural.
- **Greedy-async vs barrier.** `run_episodes_batched` uses a **barrier**: gather ALL parked → one `predict_batch` → resume ALL (`runner_batched.cpp:204-242`), and `on_decided`'s "re-gather next flush" assumption (`:233`) relies on that barrier. Under greedy-async **there is no "next flush"** — the driver resumes ONE slot per `poll()` and immediately resubmits (`wire_pool_bench.cpp:224-231`). **MUST specify:** when `on_decided` returns `true` (the slot re-parked at the next ply's leaf via `spawn_ply`), the slot is **resubmitted into the greedy loop** (one `pool.submit(slot, slot.ts->ch.features)`), not enqueued for a barrier flush. The per-thread loop is: prime K slots (`fill` each, submit its first leaf); then `while pool.any_outstanding() && !failed`: `c = pool.poll()`; `slots[c.slot].ts->resume_with(c.pred)`; if `running` → `submit(c.slot, …)`; else `on_decided(slot)` (which may `spawn_ply`+`submit`, or finalize+write+`fill`).
**RNG contract (lifted, the load-bearing determinism invariant):** each `EpisodeSlot` owns ONE persistent `std::mt19937_64` seeded `fold_seed(cfg.seed, idx)`; world-pick once before first start; per-ply-fresh `RngGumbelSource` off that same slot rng (`runner_batched.cpp:39-44,87,170-180`, `fiber_tree.hpp:65-66`). Distinct per-slot rngs ⇒ the thread/fiber interleave is irrelevant to per-tree draw order. Unchanged by the transport.
**Checkpoint:** new TU compiles + links (`${BOOST_CONTEXT_LINK}`; zmq transitive via core); a standalone smoke (8 episodes at T=1,K=4 against a warm `wire_server.py`) writes 8 `EpisodeBlocks` to redis with sane shapes/lengths, and at T=2,K=4 still writes exactly `n_eps` blocks (multi-thread disjoint-subset correctness).
**Risk:** medium-high — the most new code, and the greedy-async/episode-driver seam is the genuine integration. Mitigation: the episode driver's *logic* is proven (local layer-4 check); the transport is proven (`wire-pool-bench`); isolate the new seam with the T=1 smoke first (no multi-thread, pure greedy-async-vs-barrier), then add T>1.

### Phase C — cross-runtime parity check (layer 4)
**Goal:** prove `run_episodes_wire_batched` is aggregate-equivalent to serial `run_episodes` and local `run_episodes_batched` (Q7 bar).
**Files created:** `cpp/src/wire_batched_runtime_check.cpp` (+ CMake target `chocofarm-wire-batched-runtime-check`, linking `runner_wire_batched.cpp` + `${BOOST_CONTEXT_LINK}`). Stands up a warm in-process `InferenceServer` (via a Python sidecar / `wire_bench.py`-style server) over a fixed net; runs the SAME task/episode corpus through {serial, local-batched, wire-batched} at K∈{1,4,32}, T∈{1,2,4}; asserts:
- layer 2: single-tree `NeedsLeaf` sequence vs direct, **using the RNG ctor + fixed seed + canned leaves** (CRITIQUE F1 — not `fiber_proto`'s scripted arm).
- layer 3: the three Danihelka invariants green.
- layer 4: action-distribution + improved-π aggregate within MC CI over N≥300, ≥2 seeds; batch-composition stress (jitter K/T).
- structural: the **driver-side per-decision leaf count** (Q7 correction) identical across runtimes for matched seeds.
**Checkpoint:** `ctest -R wire-batched` green at the aggregate bar; MC standard error reported. The K=1,T=1 wire case (no batch-composition variation) should be near-bit-exact to local K=1 (only JAX-vs-numpy forward roundoff <1e-4).
**Risk:** medium — distinguishing a real aggregate divergence (a driver bug) from legitimate batch-composition roundoff. Mitigation: the driver-side leaf count is the structural discriminator (a different count is a bug, not roundoff).

### Phase D+E — executor server standup + serve dispatch + startup args (interdependent; CRITIQUE G2)
**Goal:** stand up the in-process JAX server; wire the runner to select the wire path; carry the startup args. **D and E are done together** — the executor's end-to-end test needs the serve dispatch to exist (the SUGGEST roadmap's D-before-E was circular).
**Files edited — Python (`chocofarm/az/cpp_executor.py`):**
- `_ensure_actor` (`:163-174`): build `server = InferenceServer(...)` over `RedisParamsSource(self._conn, self.run, "gen", lambda: self._published_version, initial_version)` on `ipc:///tmp/choco-infer-<run>.sock`; start a daemon `serve_forever` thread; **derive `in_dim`/`n_actions` from the SAME `self.env` (same instance/faces) the actor loads** (CRITIQUE B3 — assert this invariant explicitly; a mismatch is a ragged-batch loud reject, `inference_server.py:113`); store the endpoint; pass `--infer-endpoint ipc://… --pool-threads … --pool-batch …` via `extra_args` (`:171`).
- `generate` (`:131`): **publish-then-bump** (Q2/CRITIQUE B2) — `publish_weights(...)` FIRST, then `self._published_version = version`.
- `close` (`:250-261`): `server.stop()` → `thread.join(timeout)` → `server.close()` before reaping the actor.
**Files edited — C++:**
- `cpp/src/main.cpp` (`:121-138`): parse `--infer-endpoint`, `--pool-threads`, `--pool-batch`; thread them into `serve(...)`.
- `cpp/include/chocofarm/serve.hpp` (`:31`) + `cpp/src/serve.cpp` (`serve()` impl + `handle_generate` `:122-199`): add the endpoint + pool knobs as `serve()` params (update the only caller `main.cpp:125`); in `handle_generate`, when the endpoint is set → `run_episodes_wire_batched(...)` (skip the `NetForward` reload `:158-169` and local policy build `:172-175`); build `WireRunnerConfig` from the args (env-overridable defaults via `RuntimeConfig::from_env`, `runtime_config.hpp:38`); else if `pool_batch>1` → `run_episodes_batched`; else `run_episodes`.
- `cpp/CMakeLists.txt`: add `${BOOST_CONTEXT_LINK}` to `chocofarm-cpp-runner` (`:161-162`); add `src/runner_wire_batched.cpp` to that target's sources (per-executable, out of core). **No zmq add (transitive via core, `:147`).**
**No drift-net / ActorConfig / schema changes** (Q6).
**Checkpoint:** `pytest tests/ -q` green (including a new unit test that stands up `CppActorExecutor`, publishes a net, and drives an end-to-end generate through the serve dispatch — the two-gate config_epoch/version echoes correctly, `actor_transport.py:238-244`); `cmake --build cpp/build` clean under `-Wall -Wextra`; `test_wire_drift.py` green (unchanged, as a regression guard).
**Risk:** medium — server-thread lifecycle vs the actor subprocess. Failure mode: server dies → actor blocks at a leaf. Mitigation: the DEALER `ZMQ_RCVTIMEO` bounds the wait → loud `ERR_GENERATE_FAILED`; the executor's bounded `_recv` (`actor_transport.py:186-204`) bounds the control reply. **If a hang appears, get a `kill -ABRT` traceback — do NOT infer from the symptom (CLAUDE.md).** Note (CRITIQUE D2): a dead `ipc://` endpoint does NOT fail `zmq_connect` (lazy connect) — the failure surfaces only at first `recv` after `timeout_ms` (15s default), once per worker thread; this latency is expected, not a hang.

### Phase F — exit_loop end-to-end + the 4-iter measurement (the ADR-0009 gate)
**Goal:** the measure-first GO/NO-GO — does JAX-over-wire generation beat the serial local `NetForward` actor end-to-end, with fidelity intact?
**Measurement (this session's protocol, mirrored exactly):**
```
-I 4 -E 64 --m 24 --n-sims 256 --explore-plies 0 --td-lambda 1.0 --eval-n 8 --hidden 256 \
   --cpp-runner <serve-binary> --cores 0,1,2,3
```
Run gen-isolated, with the wire path on (endpoint set, `--pool-threads`/`--pool-batch` swept to the saturating point) vs the serial actor (no endpoint), **under matched memory pressure** (the LRU-eviction exposure, OR-3).
**Gate criteria:**
1. **Throughput:** wire generation dps materially exceeds the serial ~13 dps. The ~50 dps lives in the **composed T×K** regime (`cpp-local-batched-runtime.md §6` — ~50 is the composed number, not single-fiber). Sweep K∈{8,16,32,64}, T∈{1,2,4}; report achieved server batch `B`, dps, core utilization vs the ~1.9× wall. A dps win with no `B` increase is suspicious and must be explained.
2. **Fidelity:** the 4-iter training/eval curve is statistically indistinguishable from the serial-generation run (aggregate-within-CI, Q7 layer 4); the only difference is wall-clock. Eval is in-process Python, unaffected.
3. **No new reconciliation failures** under matched memory pressure (`cpp_executor.py:156-161`).
**Checkpoint:** GO ⇒ the wire path is the default for C++ generation, recorded with measured dps/B/utilization (ADR-0009). NO-GO ⇒ recorded honestly (e.g. ipc RTT or T×K contention on the 4-vCPU wall eats the batch win); no further wiring committed.
**Risk:** the genuine kill risk — the 4-vCPU ~1.9× ceiling may cap T×K composition and the ipc RTT may not amortize as the probe suggested. This is *why* P0 and F are measure-first gates. Do not pre-commit the wire path as default before F.

---

## 4. Parity & test plan

**The aggregate-behavioral bar (NOT per-decision byte-identity)** — see Q7. Batch-composition roundoff legitimately moves the float at a near-tie; requiring per-decision byte-identity across runtimes is wrong. The bar is: layers 1-3 exact where exact, layer 4 within Monte-Carlo CI (MC standard error reported), and the **driver-side per-decision leaf count** identical across runtimes for matched seeds (the structural discriminator — Q7 correction, NOT a `Decision` field).

**Harnesses to reuse/extend:**
- `tests/test_zmq_net_cpp.py`, `cpp/parity/wire_bench.py` — net-forward parity (layer 1), unchanged.
- `chocofarm-gumbel-dump` + `cpp/parity/gumbel_*.py` — the three Danihelka invariants (layer 3), re-run against the wrapped search.
- `fiber_proto` — reused for the fiber≡direct structural idea, **but the new layer-2 check uses the RNG ctor**, not `fiber_proto`'s scripted source (CRITIQUE F1).
- New `chocofarm-wire-batched-runtime-check` — layers 2 & 4 + the structural leaf-count cross-check (Phase C).
- `test_wire_drift.py` — unchanged, kept green as a regression guard (we add NO ActorConfig fields).

**The final 4-ExIt-iteration cpp-arm measurement** mirrors this session's protocol exactly (Phase F command above), gen-isolated, wire-on vs serial, under matched memory pressure — proving the dps lift with fidelity intact.

---

## 5. Build / CMake changes

- **`chocofarm-cpp-runner`** (`CMakeLists.txt:161-162`): add `${BOOST_CONTEXT_LINK}` to its link line; add `src/runner_wire_batched.cpp` to its sources. **Do NOT add zmq** — it is PUBLIC in `chocofarm_core` (`:147`) and thus transitive (CRITIQUE E1). Adding boost.context pulls boost onto the production runner binary (the one-shot CLI path links it too but does not use fibers — acceptable, stated).
- **New target `chocofarm-wire-batched-runtime-check`**: `add_executable(... src/wire_batched_runtime_check.cpp src/runner_wire_batched.cpp)`; `target_link_libraries(... PRIVATE chocofarm_core ${BOOST_CONTEXT_LINK})`; `-Wall -Wextra`. Mirror `chocofarm-batched-runtime-check` (`:353-355`).
- **Optionally `chocofarm-wire-batched-bench`** (the ADR-0009 driver-level measure), same link shape. (Phase F can instead reuse the existing `wire-pool-bench` for the throughput axis and the runtime-check for the episode axis — prefer not adding a target unless the bench is needed.)
- `runner_wire_batched.cpp` stays **OUT of `chocofarm_core`** (boost off the core link surface — `:344-348`, the established discipline).
- **Do NOT** flip `CHOCO_BELIEF_ZDD` in `cpp/build` (`:43-49`).

---

## 6. What NOT to do

- **P7 — do NOT re-author the wire codec.** The corr-id is an opaque ZMQ transport-envelope frame (`wire_pool_bench.cpp:191-198`, `inference_server.py:283`), not a codec field. Leave `wire_spec`, `inference_wire.*`, and `test_wire_drift.py` untouched. (And state the §4.1 provenance precisely: the implementation *diverged from* §4.1's recommended codec-field embodiment — do not write "reached §4.1.")
- **No C++-local forward reimplementation.** The JAX `InferenceServer` is the SSOT batched leaf evaluator. The wire path does NOT call `NetForward`/`predict_batch` in C++; the C++ NetForward reload (`serve.cpp:158-169`) is *skipped* on this path, not duplicated.
- **Eval STAYS Python (ADR-0008).** `CppActorExecutor.evaluate` (`cpp_executor.py:227-248`) is untouched; the server serves generation only.
- **NEVER substitute a zero/stale leaf (ADR-0002).** A failed/timed-out/unknown-corr-id leaf aborts the whole generate loudly (`std::unexpected` → `ERR_GENERATE_FAILED`); never a partial write, never a stale prediction.
- **Do NOT over-provision K past the ~1.9× host wall.** The 4-vCPU VM caps real composition; K/T are swept to the saturating point in P0/F, not maximized blindly.
- **Do NOT add the pool knobs to `ActorConfig`/`SearchConfig`** (P1 — their home is `RuntimeConfig`; route them as `--serve` startup args). Do NOT build the online-K-reconfigure hook before Phase F proves T×K composition helps (ADR-0009).
- **Do NOT bump `_published_version` before `publish_weights`** (CRITIQUE B2 — it opens a missing-blob reload-abort window). Publish first, then bump.
- **Do NOT use `fiber_proto`'s scripted source for the layer-2 structural check** — use the RNG ctor (the arm the wire driver actually runs).
- **Do NOT lift `wire_pool_bench`'s episode handling "verbatim"** — it builds no `EpisodeBlocks` and drives a constant root; only its ~45-line transport loop is reusable.

---

## 7. Open risks / unknowns the implementer must watch

- **OR-1 (the kill risk): the 4-vCPU ~1.9× ceiling + ipc RTT.** The ~50 dps lives in composed T×K; on this host the wall + per-leaf ipc RTT may cap it at/below the serial ~13 dps once the *real* episode driver (with `env.apply` and per-ply respawn, unlike the probe) runs. P0 and F are the measure-first gates. A marginal probe is a NO-GO.
- **OR-2 (the greedy-async/episode-driver seam): `on_decided`'s barrier assumption.** `run_episodes_batched`'s `on_decided` was written for a barrier flush ("re-gather next flush", `runner_batched.cpp:233`); under greedy-async the re-parked slot must be *resubmitted into the greedy loop*, not enqueued for a flush. This is the genuine new logic (Phase B); a subtle bug here silently changes which leaves co-batch. Validate with the T=1 smoke (pure greedy-async, no multi-thread) before adding T>1.
- **OR-3 (weight-blob LRU eviction on 6380 — CRITIQUE D3, NOT addressed by the SUGGEST roadmap):** `publish_weights` writes the net blob to the **transport redis 6380 (`volatile-lru`)**, and `RedisParamsSource.poll()` reads it back from the SAME instance (`inference_server.py:204`, via `transport.read_weights`). Under T×K memory pressure the **net blob itself can LRU-evict** between publish and the server's reload poll → `read_weights(want)` raises and the reload aborts (the server keeps stale params, `inference_server.py:192-193`). This is worse than a *result* blob evicting (which the executor's reconciliation catches). The registry is on 6379 (noeviction); **weights are on 6380 (volatile-lru)** (CLAUDE.md). Watch for it under the matched-pressure Phase F run; if it bites, the mitigation is to publish weights with a short TTL refresh or move the gen-weight key to the noeviction instance (a real decision the implementer must surface, not silently make).
- **OR-4 (server geometry invariant — CRITIQUE B3):** the server's `in_dim`/`n_actions` MUST equal the C++ actor's `fb.dim()`/`n_slots`, i.e. the Python `self.env` and the C++ actor must load the SAME instance/faces. Assert this at server standup (Phase D); a mismatch is a ragged-batch loud reject or silent dimension corruption.
- **OR-5 (failure granularity — CRITIQUE D1/D2):** one timed-out leaf kills the WHOLE generate after a 15s `ZMQ_RCVTIMEO`; a dead `ipc://` endpoint does not fail `connect` (lazy) and surfaces only at first `recv`. This is loud and correct (ADR-0002) but the 15s latency per failure is expected — do not mistake it for a hang; get a `kill -ABRT` traceback if uncertain.

---

## 8. Documentation obligations (ADR-0005 — part of the delivery)

- `docs/design/cpp-search-runtime.md` — amend-by-append (Rule 8, dated): the wire case **closes §8.1 via a transport envelope (diverging from §4.1's recommended codec-field embodiment)**; adopts **Option A**; takes **whole-generate-abort** (§8.2 deferred); server-standup lives in `cpp_executor.py`; **publish-then-bump** weight ordering; pool knobs ride **startup args, not ActorConfig** (the §8.x ActorConfig-knob assumption is superseded).
- `docs/design/cpp-local-batched-runtime.md` — cross-reference the wire sibling and the local/wire fork.
- `docs/STATUS.md` — if it describes the C++ generation leaf-eval regime, update it (the leaf can now be remote-batched on the wire path).
- ADR-0006 headers on every new file (`runner_wire_batched.{hpp,cpp}`, `wire_leaf_pool.hpp`, `wire_batched_runtime_check.cpp`).
- `cpp_executor.py` module docstring (`:1-44`) — the redis weight seam now feeds the **in-process JAX server**, not a C++ local `NetForward`.
