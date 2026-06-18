I now have everything needed. I have read all key files end to end. Note one critical finding: `run_episodes_batched` already exists and gathers leaves into `net.predict_batch` (a **local** batched forward via `NetForward::predict_batch`), and `TreeState` already has both a scripted and an RNG-backed ctor. The wire path needs to replace the `net.predict_batch` local call with a DEALER-to-InferenceServer submit/poll loop, multi-threaded per `wire_pool_bench`, while preserving the `EpisodeSlot`/`EpisodeBuilder` episode-driver machinery.

Let me write the roadmap.

# Wiring the C++ `--serve` Actor's GENERATION Step to the JAX InferenceServer over ZMQ — Implementation Roadmap

**Status:** Implementation-ready design record (contracts-first; no code committed by this artifact). Branch: `cpp-actor-online-reconfig`. Build dir `cpp/build` (Release `-O3 -march=native`; do **not** flip `CHOCO_BELIEF_ZDD` there — ODR hazard, `cpp/CMakeLists.txt:46-49`). Read end to end before implementation (ADR-0002).

This roadmap targets the **wire** regime: replace the serve actor's serial local `NetForward`-per-leaf generation with a **DEALER-multiplexed fiber pool feeding the JAX batched `InferenceServer` over ZMQ** — the proven `wire_pool_bench.cpp` engine (ADR-0012 P7: lift it, do not re-author). It is the wire-sibling of the already-landed **local** batched runtime (`docs/design/cpp-local-batched-runtime.md`, `run_episodes_batched`), and it deliberately reuses that runtime's `EpisodeSlot`/`EpisodeBuilder` episode-driver verbatim — **the wire change is a leaf-transport swap, not a search or episode-loop rewrite.**

---

## 0. What already exists (the load-bearing inventory — verified by reading, not grep)

The session's framing ("this is an INTEGRATION, not a search rewrite") is correct and stronger than stated, because **two** pre-built engines exist:

1. **The fiber search core** — `TreeState` (`fiber_tree.hpp:42-108`) drives the **UNCHANGED** `GumbelAZPolicy::run_search` (`gumbel.hpp:237-239`) inside a `boost.context` fiber via `YieldingNetEvaluator` (`fiber_leaf.hpp:41-54`). It has **two ctors**: scripted/`CyclicGumbelSource` (`fiber_tree.hpp:55-56`) and the **production `RngGumbelSource`** off a persistent `std::mt19937_64&` (`fiber_tree.hpp:65-66`). `start()`/`resume_with()` (`:88-107`) park at one leaf and resume. Per-tree-in-flight==1 is structural (`fiber_tree.hpp:21`).

2. **The wire DEALER engine** — `wire_pool_bench.cpp` already multiplexes `T` threads × `K` fibers over per-thread DEALER sockets to the JAX `InferenceServer`, with **echoed `u64` corr-id routing** (`wire_pool_bench.cpp:84-103,191-198,212-232`), greedy-async drain (resume-and-immediately-resubmit, `:224-231`), fail-loud on unknown corr-id (`:219`) and on transport/decode error (`:215-217`). It drives `TreeState` directly. **It only counts leaves/decisions — it does NOT build `EpisodeBlocks`** (this is the genuine gap, Q3).

3. **The local episode driver** — `run_episodes_batched` (`runner_batched.cpp:67-245`) is the wire engine's missing half: `EpisodeSlot` (`:45-63`) owns the persistent per-slot rng, stable `loc/bw/coll`, a live `EpisodeBuilder`, and the current ply's `TreeState`; `fill()`/`spawn_ply()`/`on_decided()` (`:86-190`) re-express `run_episode`'s control flow (record decision → `env.apply` → finalize → `write_results`) around the fiber park/resume, gathering parked leaves into **one local `net.predict_batch`** (`:222`).

4. **The server** — `InferenceServer` (`inference_server.py:222-345`) already round-trips the DEALER corr-id envelope **opaquely** (`frames[1:-1]` echoed verbatim, `:283,312-314`) — so **no server change is needed** for corr-id routing. `wire_server.py:build_server` stands one up over a `StaticParamsSource`.

5. **`EpisodeBuilder`** (`runner.hpp:78-119`, `runner.cpp:18-102`) — the per-episode record-assembly + pure-MC λ-return suffix target, **decoupled from the serial ply loop** precisely so a fiber-mux driver can drive it. Already used by `run_episodes_batched`.

**The integration is therefore: cross-breed (2) and (3).** Take the wire DEALER transport from `wire_pool_bench.cpp` and the episode-driver state-machine from `runner_batched.cpp`, producing a new `run_episodes_wire_batched` that gathers parked leaves and resolves them over the DEALER instead of a local `predict_batch`. Stand the server up in `cpp_executor.py`. This is far less new code than the framing implies.

---

## 1. Resolution of the 10 integration questions (concrete decisions + citations)

### Q1 — Where the JAX InferenceServer lives in production

**Decision: `cpp_executor.py` stands up an in-process `InferenceServer` thread over the live net it already holds, on an `ipc://` endpoint, replacing BOTH the redis weight-publish AND the C++ local `NetForward`.**

- **Home:** `CppActorExecutor.__init__` (`cpp_executor.py:70-100`) gains a server built exactly as `wire_server.py:build_server` (`ParamsSource` + `InferenceServer` + a daemon `serve_forever` thread). The executor already holds `self.env`, `in_dim`, `n_slots` (`:82-83`), and publishes the net every generate (`:131`). The server replaces the C++ leaf forward; the redis weight-publish becomes the server's *param feed* (see Q2), not a C++ read path.
- **Endpoint:** **`ipc://` (Unix domain socket), not `tcp`, not `inproc`.** `inproc` is impossible — the C++ DEALER is in a *different process* (the `--serve` subprocess), and `inproc` is intra-process only. `ipc` beats `tcp` because both peers are on the same host (the 4-vCPU VM) — it skips the TCP/loopback stack for lower per-RTT latency, which is exactly the cost this whole change attacks. Endpoint string: `ipc:///tmp/choco-infer-<run>.sock` (namespaced by `self.run`, `cpp_executor.py:85`, so concurrent runs don't collide; under `/tmp` is acceptable for a *socket* — the "never `/tmp` for experiment output" rule, CLAUDE.md, is about *data*, not transient sockets). The endpoint is passed to the C++ actor via the control protocol (Q6) and as a `--serve` startup arg.
- **Lifecycle/teardown:** server thread started lazily in `_ensure_actor` (`cpp_executor.py:163-174`) **before** the actor's first `ping`, so the actor's first `generate` finds a live server. `close()` (`cpp_executor.py:250-261`) gains, in order: `server.stop()` → `serve_thread.join(timeout)` → `server.close()` (the exact `wire_server.py:main` shutdown sequence, mirroring `InferenceServer.stop→join→close`'s documented discipline, `inference_server.py:329-344`) — done **before** reaping the actor subprocess, so the actor never blocks on a dead server mid-generate.
- **Thread/daemon:** one daemon `threading.Thread(target=server.serve_forever)`, single-threaded server (JAX/XLA owns the forward — the `jaxtrain-deadlock-rca` / R14 invariant, `inference_server.py:24-28`). No XLA in a worker thread.

### Q2 — Weight versioning: how a new gen version reaches the server

**Decision: a live `ParamsSource` swap driven by the executor, NOT a frozen `StaticParamsSource`. Production must follow the net per generation; `StaticParamsSource` is a benchmark control only (`cpp-search-runtime.md §7.1, §8.3`).**

The server holds the net now, so the existing C++ version-gated `NetForward` reload (`serve.cpp:158-169`) is **deleted from the wire path** (the actor no longer reads weights for the leaf). Two viable mechanisms; pick **(A)**:

- **(A) `RedisParamsSource` + a version-supplier the executor bumps (chosen).** `cpp_executor.py:131` already calls `self.transport.publish_weights(net, "gen", version, self.run)` every generate. Build the server over `RedisParamsSource(conn, run, "gen", version_supplier=lambda: self._published_version, initial_version=0)` (`inference_server.py:184-219`). The executor sets `self._published_version = version` *before* publishing, so the server's between-batch `poll()` (`inference_server.py:210-219,309`) reloads on the next drain. This reuses the proven version-gated reload verbatim and keeps the redis weight seam (now feeding the *server* instead of the C++ actor).
- **(B) direct in-process param injection** (call `params_from_manifest_blob(pack_net(net))` and swap into a custom `ParamsSource`) — avoids the redis round-trip but re-authors the reload seam. Rejected: (A) reuses an already-tested seam; the redis hop is off the hot path (between-batch only).

**The two-gate config_epoch/version discipline is PRESERVED unchanged.** The control protocol still carries `version` and `config_epoch` (`serve.cpp:131-148`, `actor_transport.py:229-246`); the executor still asserts the echoed epoch/version round-trips (`actor_transport.py:238-244`). The change is purely: gate-2 in the C++ actor (`serve.cpp:158-169`) no longer reloads a *local* net — it instead becomes a **fail-loud assertion that the actor knows which version it is generating under** (it stamps `version` into the result-token reply, `serve.cpp:194`, exactly as today). The runner knows the server is serving the right version because the executor sequences `self._published_version = version; publish_weights(...)` **before** sending the `generate` control message — and the server reloads between batches before serving any of that generate's leaves. **The net-version-straddle within one episode's ~48 leaves stays open** (`cpp-batched-search.md §3.7`, `cpp-search-runtime.md §8.3`): in production the server only reloads between generates (the version-supplier is constant within one generate), so this is actually *tighter* than the design note's worry — there is exactly one reload boundary, before the generate's first leaf. **Risk:** if a previous generate's late leaves are still in flight when the next `publish_weights` lands, the server could reload mid-flight. Mitigation: the actor's `generate` is lock-step (`actor_transport.py:206-217` — one request in flight), so the actor fully drains generate N (all leaves resolved, all episodes written) before the executor publishes version N+1. Named, not hand-waved.

### Q3 — Result blocks: the genuine gap, and where the episode loop lives

**The honest gap: `wire_pool_bench.cpp` builds NO result blocks — it only counts leaves/decisions (`:225,234-235`). The episode driver does not exist in the wire engine.** It exists in `run_episodes_batched` (local). The integration **lifts the `EpisodeSlot`/`EpisodeBuilder`/`fill`/`spawn_ply`/`on_decided` machinery from `runner_batched.cpp:45-190` into the new wire driver**, and replaces only the leaf-resolution step:

- `runner_batched.cpp:204-242` gathers parked leaves → `net.predict_batch(batch, in_dim)` (one local call) → resumes each slot.
- The wire driver instead: gathers parked leaves → `dealer.submit(corr_id, features)` per slot, tracking `corr_id → (thread, slot)` → greedy-async `poll()`/`recv` loop → `slot.ts->resume_with(pred)` → `on_decided` (verbatim from `runner_batched.cpp:101-157`).

So **the per-episode loop (step env, collect feat/pi/mask, value targets, the TERMINATE decision, the pure-MC suffix target) lives in `EpisodeSlot` + `on_decided` + `EpisodeBuilder::finalize`, lifted unchanged** (`runner_batched.cpp:101-157`, `runner.cpp:51-102`). The wire engine contributes only the DEALER transport + corr-id routing + the greedy-async drain. **This is the single largest piece of genuinely new code, and it is mostly a merge of two existing files.**

The **T-thread disjoint-subset** structure (`wire_pool_bench.cpp:182-184`) carries over: each thread owns episode indices `tid, tid+T, …`, its own DEALER socket, its own `K` slots, its own corr-id→slot map — so single-writer-per-slot is structural (no migration). `written` is summed across threads via an atomic (`wire_pool_bench.cpp:165,234`).

### Q4 — Corr-id routing: echoed corr-id, already present

**Decision: echoed `u64` corr-id frame — and it is ALREADY IMPLEMENTED on both sides. No wire-codec amendment is needed.** This supersedes `cpp-search-runtime.md §4.1`/`§8.1`, which priced a codec bump as an open cost. The reality:

- C++ stamps a globally-unique `u64` corr-id as a **leading ZMQ frame** ahead of the payload (`wire_pool_bench.cpp:191-198`: `zmq_send(corr, SNDMORE); zmq_send(payload)`), and matches replies by it (`:212-220`). Unknown corr-id ⇒ loud fail (`:219`).
- The server round-trips it **opaquely as the transport envelope** (`inference_server.py:82-85,283,312-314`): `envelope = frames[1:-1]` echoed verbatim, **never parsed**. The corr-id is a *transport* concern kept OUT of the value codec (`inference_wire.*` is unchanged) — ADR-0012 P7 serialization⊥transport, stated verbatim at `inference_server.py:84` and `wire_pool_bench.cpp:16-18`.

This is the **robust** mechanism the design note recommended (§4.1), reached without touching `wire_spec`/`inference_wire`/`test_wire_drift`. The note's §8.1 open question is **closed: echoed-id, already paid, zero wire change.**

### Q5 — Per-tree-in-flight==1, failure routing, batch-abort granularity

- **Cap (a) per-tree-in-flight==1 (`cpp-search-runtime.md §6a`):** structural and free — `TreeState` parks at exactly one leaf and cannot submit a second until `resume_with` (`fiber_tree.hpp:21,103-107`). The driver submits a slot's next leaf only after resuming it (`wire_pool_bench.cpp:226-227`). No runtime check.
- **Failure routing (`§5`):** fail loud, **never** a zero/stale leaf substitution. `wire_pool_bench.cpp` already does this: a recv error, malformed envelope, decode failure, or unknown corr-id sets `failed` and breaks the worker loop (`:215-219`). For the production driver this must become a **typed `std::expected<int,Error>` abort of the whole generate** (matching `run_episodes_batched`'s contract, `runner_batched.cpp:223,150`), propagated up as `ERR_GENERATE_FAILED` (`serve.cpp:188-189`) — never a partial write. **Per-tree timeout:** the DEALER socket sets `ZMQ_RCVTIMEO` (`wire_pool_bench.cpp:176`); a poll timeout (server-down/overloaded) returns the loud non-hang path. Add a per-outstanding-corr-id submit timestamp so a stuck single leaf is attributable (the `§5` per-tree-timeout refinement); at minimum the socket-level RCVTIMEO bounds the whole-batch hang.
- **Whole-batch-abort vs per-task-expected (`§1`, `§8.2`):** **whole-generate-abort** for the self-play run. Rationale: (a) loudest (ADR-0002); (b) `run_episodes`/`run_episodes_batched` already abort the whole pass on any leaf/write error (`runner.cpp:214`, `runner_batched.cpp:150,223`), so this matches the established contract and `cpp_executor.py`'s written-vs-found reconciliation (Q10) expects an all-or-nothing `written`; (c) a per-episode `expected` would desync the `written` count the executor reconciles (`cpp_executor.py:156-161`). The `§8.2` "promote to per-task for long self-play" is **deferred**, flagged, not taken here — losing one episode silently is worse than a loud abort the executor retries at the generate level.

### Q6 — Control protocol / ActorConfig changes

**Decision: add THREE knobs as a new Mut class beyond the 7 HOT search knobs: `endpoint` (the ipc address — INSTANCE-like, fixed per actor), `max_batch` and `batch_k`/`K`+`threads` (HOT pool knobs).** Concretely the minimal, drift-safe addition:

- `endpoint: str` — the ipc server address. Classified **INSTANCE** (fixed for the actor's life, like instance/faces — changing it is a new server, a loud reject), OR passed as a `--serve` startup arg (simpler, see below).
- `batch_k: int`, `pool_threads: int` — the in-flight target and thread count, classified **HOT** (ride `ActorConfig`, retune without respawn — the online-reconfigure win this branch exists for). These map to `RuntimeConfig{thread_pool_size, batch_size}` (`runtime_config.hpp:24-26`), from which `fibers_per_thread` derives (`:32-35`).

**Where they're added (drift-netted, ADR-0012 P7):**
- Python: `actor_config.py` `ActorConfig` dataclass (`:43-59`) + `_SCHEMA_SOURCE` (`:67-78`) + the `FIELD_NAMES`/`MUT_CLASSES` derivations (`:81,108`). New schema fields needed in `hp/schema.py` for `batch_k`/`pool_threads` (HOT) so they have a home (`_mut_of`, `:93-102`).
- C++ mirror: `actor_config.hpp` `ACTOR_CONFIG_FIELDS`/`ACTOR_CONFIG_MUT` literals (`:60-71`) — bump array sizes from 9, add the fields. `actor_config_from_json` (`:55`, impl `actor_config.cpp`) parses them.
- `tests/test_wire_drift.py` is the backstop — it parses the C++ literals as text and asserts equality (`actor_config.hpp:7-9`).
- `cpp_executor.py:_actor_config` (`:182-192`) adds the knobs to its KeyError-guarded bag (a missing K is a loud failure, never a silent serial fallback — same standard as the existing 7).

**Configure-on-change semantics + the config_epoch gate are UNCHANGED.** `cpp_executor.py:143-145` already sends `configure` only when the projected `ActorConfig` changes; the runner assigns `config_epoch` (`serve.cpp:117-119`); gate-1 (`serve.cpp:145-148`) still refuses a stale-epoch generate. Adding HOT pool knobs means a K-sweep re-tunes without a respawn.

**Simpler alternative for `endpoint`:** pass it as a **`--serve` CLI startup arg** (`--infer-endpoint ipc://…`, parsed in `main.cpp:121-138` alongside `--run`) rather than through `ActorConfig`. This keeps `ActorConfig`/the drift net touching only the two HOT pool knobs, and matches `endpoint`'s actually-fixed-for-life nature. **Recommended.** `cpp_executor.py` passes it via `extra_args` (`cpp_executor.py:171`, `actor_transport.py:131-144`).

### Q7 — Parity/test gate: the acceptance bar

**Decision: aggregate behavioral equivalence (`cpp-search-runtime.md §7.3`), NOT per-decision byte-identity.** This is the load-bearing fidelity subtlety, and the wire path makes it *mandatory* (unlike the local path, which could reach bit-exactness): batch-composition roundoff (`cpp-batched-search.md §2.2`) means *which* trees co-batch a leaf depends on arrival timing under the greedy drain, so a near-tie (`§1.3`) can legitimately flip an SH survivor — **per-decision byte-identity across runtimes is WRONG to require.** The four composing layers (`§7.3`):

1. **Net-forward parity (inherited):** the wire path is pinned at max|Δ|<1e-4, measured ~e-7 (`tests/test_zmq_net_cpp.py`, `cpp/parity/wire_bench.py`). Unchanged — the runtime does not touch the leaf numerics.
2. **Structural-determinism (single-tree, canned leaves):** feed canned byte-identical `NetPrediction`s and assert the `NeedsLeaf` feature-row sequence + final `Decision` matches the in-process serial reference. `fiber_proto.cpp` already proves fiber-driven ≡ direct (`cpp-local-batched-runtime.md §1`). Re-used as-is.
3. **The three Danihelka invariants:** `test_executed_action_is_sh_survivor`, `test_vmix_prior_weighted`, `test_sequential_halving_spends_full_budget` (`chocofarm-gumbel-dump`, `cpp/parity/gumbel_*.py`) — re-run against the wrapped search; `run_search` is byte-untouched (Option A), so they must still pass.
4. **Aggregate behavioral equivalence (the cross-runtime bar):** N≥300 decisions across ≥2 seeds, action-distribution + improved-π statistics statistically indistinguishable across {serial `run_episodes`, local `run_episodes_batched`, wire `run_episodes_wire_batched`} within Monte-Carlo CI, MC standard error reported. Plus a **batch-composition stress** (vary `K`/`threads`, inject arrival jitter; assert aggregate stays inside CI). Harness: `cpp/parity/` analog of `wire_bench.py` driving the wire driver against a warm `InferenceServer`.

**Acceptance bar:** layers 1-3 green (byte/exact where they are exact), layer 4 within MC CI. A per-decision divergence at a near-tie is a *correct* result, not a failure.

### Q8 — Build/CMake

- **How the runner gains the mode:** **an `ActorConfig`/startup-arg selector in `serve.handle_generate`, identical in shape to the local-batched selector the local roadmap defined** (`cpp-local-batched-runtime.md §2`, Chunk 5). `serve.cpp:187` currently calls `run_episodes(...)`. The dispatch becomes: if a wire endpoint is configured (the `--infer-endpoint` startup arg is set) → `run_episodes_wire_batched(...)`; else if `batch_k>1` → `run_episodes_batched(...)` (local); else `run_episodes(...)` (serial). **The wire path is selected by the endpoint being present, not always-on** — so the one-shot CLI and local paths are untouched (ADR-0004 minimal-touch). When the wire path is taken, the C++ `NetForward` reload (`serve.cpp:158-169`) and policy build (`:172-175`) are **skipped** (the leaf is remote; the per-tree `GumbelAZPolicy` is built inside each `TreeState` over the `YieldingNetEvaluator`, `fiber_tree.hpp:56`, exactly as `wire_pool_bench.cpp:203`).
- **Deps already in CMake:** `boost.context` (`${BOOST_CONTEXT_LINK}`, `CMakeLists.txt:294-315`) and libzmq (linked by `wire-pool-bench`, `:355`) are **confirmed present** — `wire-pool-bench` builds and links both. The new `runner_wire_batched.cpp` TU is kept **OUT of `chocofarm_core`** (boost stays off the core link surface — `cpp-local-batched-runtime.md` finding #8, `CMakeLists.txt:344-350`) and compiled per-executable; the **serve/runner executable target gains `${BOOST_CONTEXT_LINK}` + the zmq link** (mirroring `wire-pool-bench`, `:355`). An implementer who skips this hits an undefined-reference at link.
- **New targets:** `chocofarm-wire-batched-runtime-check` (the layer-4 parity check against a warm in-process server) and `chocofarm-wire-batched-bench` (the ADR-0009 measure), both linking `runner_wire_batched.cpp` + `${BOOST_CONTEXT_LINK}` + zmq.

### Q9 — Sequencing (each step independently verifiable, ADR-0009)

(a) extract a reusable **wire leaf-resolver** component from `wire_pool_bench.cpp`'s DEALER+corr-id loop → (b) build `run_episodes_wire_batched` (result-block emission through it, reusing `EpisodeSlot`/`EpisodeBuilder`) → (c) the parity check (layer 4) against a warm in-process server → (d) `cpp_executor` server standup + lifecycle → (e) serve dispatch + ActorConfig knobs → (f) the exit_loop end-to-end + the 4-iter measurement. Detailed in §3.

### Q10 — What stays

- **Eval in-process Python (ADR-0008):** `CppActorExecutor.evaluate` (`cpp_executor.py:227-248`) is untouched — a pure-Python `GumbelPolicy` over the trained net, no subprocess, no server. The InferenceServer serves **generation only**.
- **The Part-B-blend and explore-plies>0 fail-loud guards STAY:** `cpp_executor.py:114-129` — the runner still emits the pure-MC λ-return only and plays temperature-0 every ply; both guards raise loudly. Unchanged.
- **The pure-MC value target STAYS:** `EpisodeBuilder::finalize`'s suffix-return math (`runner.cpp:51-102`) is lifted verbatim into the wire driver.
- **The cpp_executor written-vs-read reconciliation STAYS:** `cpp_executor.py:156-161` — `n_found != result.written` is a loud failure. The whole-generate-abort contract (Q5) keeps `written` all-or-nothing, so reconciliation is unaffected. **New exposure (inherited from the local roadmap, finding #4):** concurrent T×K episodes change the write→read timeline; the transport redis is `volatile-lru` (6380), so early-finishing episodes' blobs sit longer before the parent reads — enlarging the eviction window. Mitigation: write each episode immediately on finalize (`runner_batched.cpp:148`, lifted), and run the §4 integration measure under the same memory pressure as the serial baseline.

---

## 2. Target architecture

```
exit_loop.run  ──generate(net, version, worlds, lam, …)──▶  CppActorExecutor.generate  (cpp_executor.py:102)
                                                              │
  [NEW] in-process InferenceServer thread  ◀──ipc://choco-infer-<run>.sock──┐
        (RedisParamsSource, version-supplier=_published_version)            │  (JAX batched leaf eval — SSOT)
        built in __init__/_ensure_actor; closed in close()                  │
                                                              │             │
  self._published_version = version; publish_weights(net,…)   │             │
  reconfigure(cfg)  [HOT: m/n_sims/c_* + batch_k/pool_threads] │             │
  generate(GenerateRequest{epoch,version,seed,lam,episodes,…})│             │
                                                              ▼             │
                              SubprocessActorTransport (JSON-line control over stdin/stdout)
                                                              │             │
                                  chocofarm-cpp-runner --serve --run R --infer-endpoint ipc://…
                                                              │             │
                              serve.handle_generate (serve.cpp:122)         │
                                ├─ [wire path] endpoint set → run_episodes_wire_batched(...)
                                │     (NO NetForward reload; NO local policy build)
                                ├─ [local path] batch_k>1 → run_episodes_batched(...)
                                └─ [serial]     → run_episodes(...)
                                                              │             │
                  run_episodes_wire_batched  (NEW: runner_wire_batched.cpp) │
                    T threads × K fiber-slots, lifted from wire_pool_bench + runner_batched:
                    EpisodeSlot{ts, eb, rng, world, loc/bw/coll} (runner_batched.cpp:45)
                    gather parked leaves → dealer.submit(corr_id, features) ──────────────┘
                    greedy-async recv → resume_with(pred) → on_decided → EpisodeBuilder
                    → write_results (immediate, idx-keyed, redis 6380)
```

---

## 3. Phased roadmap

Each phase lands and tests independently (ADR-0009). Phases touching validated code (`gumbel.cpp`, `runner.cpp`, `fiber_tree.hpp`): **none beyond what the local roadmap already paid** — `fiber_tree.hpp`'s `RngGumbelSource` ctor exists (`:65-66`); `EpisodeBuilder` exists (`runner.hpp:78`); `run_search` is touched zero lines (Option A).

### Phase A — extract the reusable wire leaf-resolver component

**Goal:** lift the DEALER+corr-id submit/poll/route mechanics out of `wire_pool_bench.cpp`'s `worker` lambda into a reusable, testable header so the production driver and the bench share one home (ADR-0012 P1).

**Files created:**
- `cpp/include/chocofarm/wire_leaf_pool.hpp` — a per-thread DEALER resolver:
  ```cpp
  // One thread's DEALER socket + its outstanding-leaf registry. RAII, move-only, create()-factory
  // (a throwing ctor cannot return a value — P9 rule 5). Reuses inference_wire verbatim (P7).
  class WireLeafPool final {
    [[nodiscard]] static std::expected<WireLeafPool,Error>
        create(void* zctx, const std::string& endpoint, int timeout_ms, std::atomic<uint64_t>& corr_seq);
    // submit features for slot s; stamp+track a unique corr-id (wire_pool_bench.cpp:191-198):
    [[nodiscard]] std::expected<void,Error> submit(int slot, std::span<const float> features);
    // block up to timeout for ONE reply; decode; route by corr-id to its slot (wire_pool_bench.cpp:212-220):
    [[nodiscard]] std::expected<Completion,Error> poll();   // Completion{int slot; NetPrediction pred;}
    [[nodiscard]] bool any_outstanding() const;
  };
  ```
**Files edited:** `cpp/src/wire_pool_bench.cpp` — re-expressed over `WireLeafPool` (its `submit`/`recv_corr_payload`/`inflight` logic, `:84-103,191-232`, moves into the header). Behaviour-preserving.

**Checkpoint:** `cmake --build cpp/build --target chocofarm-wire-pool-bench` clean under `-Wall -Wextra`; `wire-pool-bench` re-run against `wire_server.py` reproduces its prior `pool_dps`/`leaves`/`decided` numbers (the extraction changed nothing — it still drives `TreeState` with the cyclic source).

**Risk:** low. The corr-id atomic must stay process-global across threads (`wire_pool_bench.cpp:170` — passed by reference into each pool) so a future shared registry / work-stealing keys on it unchanged (`:18-26`).

### Phase B — `run_episodes_wire_batched`: the result-block-emitting wire driver

**Goal:** the integration's core — drive K episodes per thread over the DEALER, emitting `EpisodeBlocks` per episode exactly as `run_episodes`. This is the merge of `runner_batched.cpp`'s episode driver and `WireLeafPool`.

**Files created:**
- `cpp/include/chocofarm/runner_wire_batched.hpp`, `cpp/src/runner_wire_batched.cpp`:
  ```cpp
  struct WireRunnerConfig { std::string endpoint; int batch_k = 32; int pool_threads = 4; int timeout_ms = 15000; };
  // Same contract / redis writes / `written` count as run_episodes (runner.cpp:179). T threads, each
  // owning a disjoint episode subset (wire_pool_bench.cpp:182-184), K EpisodeSlots, one WireLeafPool.
  // Per-episode EpisodeBlocks == run_episodes at the §7 aggregate bar. Leaf eval is REMOTE (the JAX
  // server) — NO net argument (the policy inside each TreeState holds the YieldingNetEvaluator).
  [[nodiscard]] std::expected<int,Error>
  run_episodes_wire_batched(const Environment& env, const FeatureBuilder& fb, const GumbelConfig& gc,
                            RedisClient& redis, const RunnerConfig& cfg, const WireRunnerConfig& wcfg,
                            std::ostream* stats_out = nullptr);
  ```
**Reused verbatim (lifted, not re-authored):** `EpisodeSlot` (`runner_batched.cpp:45-63`), `spawn_ply`/`on_decided`/`fill` (`:86-190`), `EpisodeBuilder` (`runner.hpp:78`). **The one substitution:** `runner_batched.cpp:204-242`'s "gather → `net.predict_batch` → resume each" loop becomes a per-thread greedy-async loop: gather parked slots → `pool.submit(slot, features)` per slot → `pool.poll()` → `resume_with` → if running re-submit, else `on_decided` + `fill` (the `wire_pool_bench.cpp:211-232` drain shape, but resuming an episode-driving slot instead of a bare task).

**Threading:** T `std::thread`s, each its own `WireLeafPool`, its own slot array, its own `next_idx` over `tid, tid+T, …`. `written` accumulated via `std::atomic<int>` (`wire_pool_bench.cpp:165,234`); any thread's `Error` sets a shared `failed` atomic and the whole pass returns `std::unexpected` (Q5 whole-generate-abort). **No tree migrates between threads** (single-writer-per-slot structural, `wire_pool_bench.cpp:11-13`).

**RNG contract (lifted, the load-bearing determinism invariant):** each `EpisodeSlot` owns ONE persistent `std::mt19937_64` seeded `fold_seed(cfg.seed, idx)`; world-pick once before first start; per-ply-fresh `RngGumbelSource` off that same slot rng (`runner_batched.cpp:39-44,170-180`, `fiber_tree.hpp:65-66`). Distinct per-slot rngs ⇒ the fiber/thread interleave is irrelevant to per-tree draw order. **This is unchanged from the local driver — the wire transport does not touch it.**

**Checkpoint:** the new TU compiles + links (`${BOOST_CONTEXT_LINK}` + zmq); a standalone smoke (drive 8 episodes at K=4,T=1 against a warm `wire_server.py`) writes 8 `EpisodeBlocks` to redis with sane shapes/lengths.

**Risk:** medium — this is the most new code. The episode-driver/transport interleave is the genuine integration. Mitigated by: the episode driver is lifted verbatim (proven by the local layer-4 check); the transport is lifted verbatim (proven by `wire-pool-bench`); only their seam is new.

### Phase C — the cross-runtime parity check (layer 4)

**Goal:** prove `run_episodes_wire_batched` is aggregate-equivalent to serial `run_episodes` and the local `run_episodes_batched` (Q7 bar).

**Files created:** `cpp/src/wire_batched_runtime_check.cpp` (+ CMake target `chocofarm-wire-batched-runtime-check`, linking `runner_wire_batched.cpp` + `${BOOST_CONTEXT_LINK}` + zmq). It stands up an **in-process warm `InferenceServer`** (the test harness builds it via a Python sidecar or `cpp/parity/wire_bench.py`-style server) over a fixed net, runs the SAME task/episode corpus through {serial, local-batched, wire-batched} at K∈{1,4,32}, T∈{1,2,4}, and asserts:
- layer 2: single-tree `NeedsLeaf` sequence vs direct (canned leaves) — reuse `fiber_proto`.
- layer 3: the three Danihelka invariants (`chocofarm-gumbel-dump`) green.
- layer 4: action-distribution + improved-π aggregate within MC CI over N≥300, ≥2 seeds; batch-composition stress (jitter K/T).

**Checkpoint:** `ctest -R wire-batched` green at the aggregate bar; MC standard error reported. The K=1,T=1 wire case is the tightest (no batch-composition variation) — it should be near-bit-exact to local K=1 (only JAX-vs-numpy forward roundoff, <1e-4, `inference_server.py:30-34`).

**Risk:** medium — distinguishing a *real* aggregate divergence (a driver bug) from legitimate batch-composition roundoff. Mitigation: the leaf-request count per decision (`Decision::leaf_requests`, `search_runtime.hpp:69`) is a *structural* cross-check that must be identical across runtimes for matched seeds — a different count is a bug, not roundoff (`cpp-search-runtime.md §7.2`).

### Phase D — `cpp_executor` server standup + lifecycle

**Goal:** stand up the in-process JAX `InferenceServer` over the live net, replacing the redis-weight-only feed of the C++ local forward.

**Files edited:** `chocofarm/az/cpp_executor.py`:
- `__init__`/`_ensure_actor` (`:70-100,163-174`): build `server, in_dim, n_actions = build_server-analog(self.env, hidden, endpoint=ipc://…-<run>.sock, max_batch)` over a `RedisParamsSource(self._conn, self.run, "gen", lambda: self._published_version, 0)`; start a daemon `serve_forever` thread; store the endpoint.
- `generate` (`:131`): set `self._published_version = version` **before** `publish_weights` so the server's between-batch `poll()` reloads to the new version before serving that generate's first leaf.
- `close` (`:250-261`): `server.stop()` → `thread.join(timeout)` → `server.close()` **before** reaping the actor (Q1).
- pass `--infer-endpoint ipc://…` via `extra_args` (`:171`).

**Checkpoint:** a unit test stands up `CppActorExecutor`, publishes a net, and asserts the server serves the right version (a direct DEALER probe, or the wire driver's smoke from Phase B run through the executor). `pytest tests/ -q` green.

**Risk:** medium — server thread lifecycle vs the actor subprocess. Failure mode: server dies, actor blocks at a leaf forever. Mitigation: the DEALER `ZMQ_RCVTIMEO` (`wire_pool_bench.cpp:176`) bounds the leaf wait → loud `ERR_GENERATE_FAILED`; the executor's bounded `_recv` (`actor_transport.py:186-204`) bounds the control reply. Per CLAUDE.md, if a hang appears, get a `kill -ABRT` traceback rather than inferring from the symptom.

### Phase E — serve dispatch + ActorConfig knobs

**Goal:** wire the runner to select the wire path; carry the HOT pool knobs.

**Files edited:**
- `cpp/src/main.cpp` (`:121-138`): parse `--infer-endpoint`; thread it into `serve(...)` (a new param on `serve()`, `serve.hpp:31`).
- `cpp/src/serve.cpp` `handle_generate` (`:122-199`): when the endpoint is set, dispatch to `run_episodes_wire_batched` (skip the `NetForward` reload `:158-169` and policy build `:172-175`); build `WireRunnerConfig` from the endpoint + `st.gc`-adjacent HOT pool knobs.
- `cpp/include/chocofarm/actor_config.hpp` (`:44-71`) + `cpp/src/actor_config.cpp`: add `batch_k`, `pool_threads` (HOT) to the struct + the drift literals (bump array sizes).
- `chocofarm/az/actor_config.py` (`:43-78,108`) + `chocofarm/hp/schema.py`: add the two HOT fields with `mut=HOT` metadata.
- `chocofarm/az/cpp_executor.py:_actor_config` (`:182-192`): add the two knobs to the KeyError-guarded bag.
- `cpp/CMakeLists.txt`: add `${BOOST_CONTEXT_LINK}` + zmq to the runner/serve executable target; new `runner_wire_batched.cpp` per-executable (OUT of `chocofarm_core`).

**Checkpoint:** `tests/test_wire_drift.py` green (the C++ literals match `FIELD_NAMES`/`MUT_CLASSES`); `cmake --build cpp/build` clean under `-Wall -Wextra`; `pytest tests/ -q` green; a `configure`→`generate` round-trip through the serve path drives the wire driver end-to-end (the config_epoch/version two-gate echoes correctly, `actor_transport.py:238-244`).

**Risk:** low-medium — drift net is the backstop for the config additions. The dispatch is additive (ADR-0004); serial/local/one-shot paths untouched.

### Phase F — exit_loop end-to-end + the 4-iter measurement (the ADR-0009 gate)

**Goal:** the measure-first GO/NO-GO. Does JAX-over-wire generation beat the serial local `NetForward` actor (~13 dps) end-to-end?

**Measurement (the session protocol):**
```
-I 4 -E 64 --m 24 --n-sims 256 --explore-plies 0 --td-lambda 1.0 --eval-n 8 --hidden 256 \
   --cpp-runner <serve-binary> --cores 0,1,2,3
```
run with the wire path on (endpoint set, `batch_k`/`pool_threads` HOT-tuned to the saturating point) vs the serial actor (`batch_k=1`, no endpoint), **under matched memory pressure** (the LRU-eviction exposure, Q10/finding #4).

**Gate criteria:**
1. **Throughput:** wire generation dps materially exceeds the serial ~13 dps — the proven ~50 dps target lives in the **composed T×K** regime (`cpp-batched-search.md`-style; `cpp-local-batched-runtime.md §6` finding #6 — ~50 is the *composed* number, not single-fiber). Sweep K∈{8,16,32,64}, T∈{1,2,4}; report the achieved server batch `B`, dps, core utilization vs the ~1.9× wall (CLAUDE.md). A dps win with no `B` increase is suspicious and must be explained (`cpp-search-runtime.md §7.2`).
2. **Fidelity:** the 4-iter training/eval curve is statistically indistinguishable from the serial-generation run (aggregate-within-CI, Q7 layer 4) — the only difference is wall-clock. Eval is in-process Python (`cpp_executor.py:227-248`), unaffected.
3. **No new reconciliation failures** under matched memory pressure (`cpp_executor.py:156-161`).

**Checkpoint:** GO ⇒ the wire path is the default for C++ generation, recorded with the measured dps/B/utilization (ADR-0009). NO-GO ⇒ recorded honestly (e.g. ipc RTT or T×K contention on the 4-vCPU wall eats the batch win), no further wiring committed.

**Risk:** the genuine kill risk — the 4-vCPU wall (~1.9× ceiling) may cap T×K composition, and the ipc RTT per leaf may not amortize as the `wire_pool_bench` numbers (run against a possibly-differently-loaded server) suggest. This is *why* it is a measure-first gate. Do not pre-commit the wire path as default before this gate.

---

## 4. Documentation obligations (ADR-0005, part of the delivery)

- **`docs/design/cpp-search-runtime.md`** — amend-by-append (Rule 8, dated): record that the wire case **closes §8.1** (echoed-id is already implemented, zero wire change — `wire_pool_bench.cpp:191-198` + the server's opaque envelope `inference_server.py:283`), adopts **Option A** (the fiber substrate, not the §3.2 Option-B continuation), takes **whole-generate-abort** (§8.2 deferred), and that the server-standup lives in `cpp_executor.py`.
- **`docs/design/cpp-local-batched-runtime.md`** — cross-reference the new wire sibling (the §7 "composition with thread parallelism" and the wire/local fork).
- **`docs/STATUS.md`** — if it describes the C++ generation path's leaf-eval regime, update it (the leaf can now be remote-batched).
- **ADR-0006 headers** on every new file (`runner_wire_batched.{hpp,cpp}`, `wire_leaf_pool.hpp`, the two new check/bench TUs).
- **`cpp_executor.py` module docstring** (`:1-44`) — update the "publishes the frozen net to redis … the runner reads via read_weights/NetForward" description: the redis weight seam now feeds the **server**, not the C++ local forward.

---

## 5. The genuine hard parts / gaps (named, not papered over)

1. **The result-block gap (Q3):** `wire_pool_bench.cpp` builds no `EpisodeBlocks` — it is a throughput bench. The episode driver must be lifted from `runner_batched.cpp` and married to the wire transport. This is real new code (Phase B), though it is a merge of two proven files, not novel logic.
2. **The 4-vCPU ceiling (Phase F):** the ~50 dps lives in composed T×K; on the 4-vCPU VM the ~1.9× wall + ipc RTT may cap it. This is the measure-first kill risk, and the entire reason Phase F is a gate.
3. **LRU eviction window (Q10):** T×K concurrent episodes enlarge the write→read window for early-finishing episodes on the `volatile-lru` 6380 instance → intermittent loud reconciliation failure under memory pressure. Mitigated by immediate-on-finalize writes + matched-pressure measurement.
4. **Net-version straddle (Q2):** production reloads only between generates and the actor is lock-step, so there is one reload boundary per generate — tighter than the design note's worry, but the "late leaves of generate N in flight when version N+1 publishes" case must be confirmed impossible by the lock-step drain (it is, given `actor_transport.py:206-217`).
5. **Server lifecycle vs actor subprocess (Phase D):** a dead server must surface as a loud bounded-timeout leaf failure, not a hang. The DEALER `ZMQ_RCVTIMEO` is the bound; verify with a `kill -ABRT` traceback if a hang ever appears (CLAUDE.md), never infer from the symptom.

**Relevant file paths (all absolute):**
- `/home/bork/w/vdc/1/chocofarm/cpp/src/wire_pool_bench.cpp` (the engine to lift; echoed corr-id at :191-198, drain at :212-232)
- `/home/bork/w/vdc/1/chocofarm/cpp/src/runner_batched.cpp` + `/home/bork/w/vdc/1/chocofarm/cpp/include/chocofarm/runner_batched.hpp` (the episode driver to lift)
- `/home/bork/w/vdc/1/chocofarm/cpp/include/chocofarm/fiber_tree.hpp`, `/home/bork/w/vdc/1/chocofarm/cpp/include/chocofarm/fiber_leaf.hpp` (the fiber core + RngGumbelSource ctor at :65)
- `/home/bork/w/vdc/1/chocofarm/cpp/src/serve.cpp` (:122-199 handle_generate; dispatch site :187)
- `/home/bork/w/vdc/1/chocofarm/cpp/src/main.cpp` (:121-138 --serve startup; add --infer-endpoint)
- `/home/bork/w/vdc/1/chocofarm/chocofarm/az/cpp_executor.py` (server standup home; :131 publish, :250 close)
- `/home/bork/w/vdc/1/chocofarm/chocofarm/az/inference_server.py` (opaque corr-id envelope at :283,312-314; RedisParamsSource :184)
- `/home/bork/w/vdc/chocobo/profiles/wire_server.py` (build_server pattern)
- `/home/bork/w/vdc/1/chocofarm/chocofarm/az/actor_config.py` + `/home/bork/w/vdc/1/chocofarm/cpp/include/chocofarm/actor_config.hpp` (the HOT pool knobs + drift net)
- `/home/bork/w/vdc/1/chocofarm/cpp/CMakeLists.txt` (:294-315 boost link, :344-355 wire/batched targets)
- New: `/home/bork/w/vdc/1/chocofarm/cpp/include/chocofarm/wire_leaf_pool.hpp`, `/home/bork/w/vdc/1/chocofarm/cpp/{include/chocofarm,src}/runner_wire_batched.{hpp,cpp}`, `/home/bork/w/vdc/1/chocofarm/cpp/src/wire_batched_runtime_check.cpp`