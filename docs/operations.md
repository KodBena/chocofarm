<!-- docs/operations.md
     Purpose: the operations runbook — how to build, test, sweep, profile, and run every experiment across chocofarm.
     Public Domain (The Unlicense). -->

# chocofarm — Operations Runbook

This is the copy-pasteable operations runbook for chocofarm: how to build the C++ runner/search stack, run the bit-exact and behavioral-parity gates, drive the profiling/sweep harnesses, launch AlphaZero training (including the C++ actor loop), and operate the supporting infrastructure (TensorBoard, the two redis instances, experiment records, and hang debugging). Every literal command and every `file:line` citation is preserved verbatim from the subsystem sources. Unless a command states otherwise, run it from the repo root `/home/bork/w/vdc/1/chocofarm`, with the project interpreter `/home/bork/w/vdc/venvs/generic/bin/python`.

## Table of Contents

- [C++ builds & targets](#c-builds--targets)
- [Correctness, equality & parity tests](#correctness-equality--parity-tests)
- [Search profiling, benchmarks & sweeps](#search-profiling-benchmarks--sweeps)
- [AlphaZero training & the C++ actor loop](#alphazero-training--the-c-actor-loop)
- [Infrastructure: TensorBoard, redis, experiment records & debugging](#infrastructure-tensorboard-redis-experiment-records--debugging)
- [Conventions cheat-sheet](#conventions-cheat-sheet)

## C++ builds & targets

The C++ runner/search stack lives under `cpp/`, built with CMake. It is **C++23** (`set(CMAKE_CXX_STANDARD 23)` — needs `std::expected`; `cpp/CMakeLists.txt:16`) and defaults to a **Release** build (`cpp/CMakeLists.txt:20-22`). All commands below run from the **repo root** `/home/bork/w/vdc/1/chocofarm` (the `-S cpp` flag points CMake at the source tree; the `-B <dir>` flag names the out-of-tree build dir).

**System dependencies (build fails loudly without them).** Three are required:
- `hiredis` — Debian/Ubuntu `libhiredis-dev`, Fedora/openSUSE `hiredis-devel` (`cpp/CMakeLists.txt:93-97`).
- `libzmq` — Debian/Ubuntu `libzmq3-dev`, Fedora/openSUSE `zeromq-devel`/`libzmq5` (the C API `zmq.h` only; cppzmq is **not** used) (`cpp/CMakeLists.txt:121-126`).
- `boost.context` (`libboost_context`) — only needed by the fiber targets, located via `find_package(Boost COMPONENTS context)` with a `find_library(boost_context)` fallback (`cpp/CMakeLists.txt:279-289`).

`nlohmann/json` is **not** a system dep — it is fetched/pinned via FetchContent (`v3.11.3`), preferring an installed `nlohmann_json >= 3.2.0` if present (`cpp/CMakeLists.txt:59-69`).

### Build variants

**(a) Default build — `cpp/build` (live flat+bitset belief, ZDD OFF):**

```bash
cd /home/bork/w/vdc/1/chocofarm
cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Release
cmake --build cpp/build -j4
```

This is the canonical build the CMakeLists header documents (`cpp/CMakeLists.txt:8`). `CHOCO_BELIEF_ZDD` defaults OFF, so the belief surface is flat+bitset only and `belief_zdd_engine.cpp`/`env_zdd.cpp` are **not** compiled (`cpp/CMakeLists.txt:49,156-159`).

**(b) Opt-in ZDD arm — a SEPARATE dir `cpp/build-zdd-on`:**

```bash
cd /home/bork/w/vdc/1/chocofarm
cmake -S cpp -B cpp/build-zdd-on -DCMAKE_BUILD_TYPE=Release -DCHOCO_BELIEF_ZDD=ON
cmake --build cpp/build-zdd-on -j4
```

`-DCHOCO_BELIEF_ZDD=ON` adds `ZddBelief` as a third `Belief` variant arm, compiles `belief_zdd_engine.cpp` + `env_zdd.cpp` into `chocofarm_core`, and propagates `CHOCO_BELIEF_ZDD` as a **PUBLIC** compile-definition (`cpp/CMakeLists.txt:49,156-159`). The option doc text mandates a separate build dir so the default dir stays the live flat+bitset (`cpp/CMakeLists.txt:47-48`).

> **Gotcha (ODR hazard).** The flag changes the size/alternative-count of the `Belief` variant across **every** TU that links `chocofarm_core`. Never mix object files or a build dir between ZDD-ON and ZDD-OFF configurations — that is exactly the ODR desync the PUBLIC definition exists to prevent (`cpp/CMakeLists.txt:45-47,154-158`). Reconfigure a clean dir, do not flip the flag in `cpp/build`.

**(c) Profiling build — `cpp/build-profile` (frame pointers for `perf`):**

```bash
cd /home/bork/w/vdc/1/chocofarm
cmake -S cpp -B cpp/build-profile \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCMAKE_CXX_FLAGS=-fno-omit-frame-pointer
cmake --build cpp/build-profile -j4
```

This variant is **not** a CMakeLists option — it is an ad-hoc configure, reconstructed verbatim from the existing `cpp/build-profile/CMakeCache.txt` (`CMAKE_BUILD_TYPE:STRING=RelWithDebInfo` → `-O2 -g -DNDEBUG`; `CMAKE_CXX_FLAGS:STRING=-fno-omit-frame-pointer`; `CHOCO_MARCH_NATIVE:BOOL=ON`). The `-g` (debug info, from RelWithDebInfo) plus `-fno-omit-frame-pointer` give `perf record`/`perf report` resolvable, walkable stacks while keeping `-O2 -march=native` so the profile reflects optimized, vectorized code. Combine with core-pinning when recording (4-vCPU host):

```bash
taskset -c 0,1,2,3 perf record -g -- cpp/build-profile/chocofarm-wire-pool-bench
```

> **Gotcha.** `RelWithDebInfo` is `-O2`, not the `-O3` of the default `Release` build — a profile here is not bit-identical in codegen to the `cpp/build` binary. For the closest-to-production profile keep `-march=native` ON (the default).

**The `CHOCO_MARCH_NATIVE` option (default ON).** `option(CHOCO_MARCH_NATIVE ... ON)` adds `-march=native` (`cpp/CMakeLists.txt:31-34`). It unlocks AVX2/FMA/BMI2 so the branchless integer belief sweep auto-vectorizes (~1.94× isolated, ~1.29× end-to-end at K=32). It pins the binary to **this** host's ISA and may `SIGILL` on an older CPU.

> **Set `-DCHOCO_MARCH_NATIVE=OFF` when** producing a binary that must run on a different/older CPU than the build host (`cpp/CMakeLists.txt:27-29`):
> ```bash
> cmake -S cpp -B cpp/build-portable -DCMAKE_BUILD_TYPE=Release -DCHOCO_MARCH_NATIVE=OFF
> cmake --build cpp/build-portable -j4
> ```

### Targets / executables

One `chocofarm_core` static library aggregates the seam (`instance/env/features/feature_layout/transport/net/zmq_net_client/policy/nmcs/ismcts/gumbel/search_runtime/runner/actor_config/serve`; `cpp/CMakeLists.txt:129-148`); every executable below links it. Per ADR-0012 P3 (one-owner), the runner is the only production binary — every `*-dump`/`*-check`/`*-bench`/`*-probe`/`*-proto` is a separate single-purpose fixture, **not** the runner.

| Target | One-line purpose | Source TU |
|---|---|---|
| `chocofarm-cpp-runner` | The production runner (wire + episode loop; the only shipping binary) | `src/main.cpp` (+ `chocofarm_core`) |
| `chocofarm-mask-dump` | Parity fixture: replay actions, dump the legality mask (bit-identical to Python's); no redis | `src/mask_dump.cpp` |
| `chocofarm-net-dump` | Parity fixture: build C++ `NetForward` off the redis manifest seam, dump value+policy logits for stdin features (max\|Δ\|<1e-4) | `src/net_dump.cpp` |
| `chocofarm-nmcs-dump` | Parity fixture: `NMCSPolicy::search` with a scripted RNG-free `WorldSource`, print selected action (deterministic nesting check) | `src/nmcs_dump.cpp` |
| `chocofarm-ismcts-dump` | Parity fixture: `ISMCTSPolicy::run_search` with scripted RNG-free source, print selected action | `src/ismcts_dump.cpp` |
| `chocofarm-gumbel-dump` | Parity fixture: `GumbelAZPolicy::run_search` with scripted gumbel/world/leaf, print executed action + improved-π argmax; no redis | `src/gumbel_dump.cpp` |
| `chocofarm-zmq-net-probe` | Parity fixture: `ZmqNetClient` RPCs the running Python inference service, prints returned (value, logits) | `src/zmq_net_probe.cpp` |
| `chocofarm-serial-runtime-check` | Correctness check: `SerialRuntime` over a task batch == direct `GumbelAZPolicy::decide` per seed (seam-faithfulness); no redis/weights/stdin | `src/serial_runtime_check.cpp` |
| `chocofarm-belief-cache-check` | Correctness check: force a `(count,first,last)` fingerprint collision, assert the memo's full-equality guard (ADR-0011); pure FeatureBuilder | `src/belief_cache_check.cpp` |
| `chocofarm-belief-sweep-oracle-check` | Correctness check: production fused sweep vs naive `env.observe` count, assert byte-equality over sampled beliefs | `src/belief_sweep_oracle_check.cpp` |
| `chocofarm-belief-sweep-bench` | Microbench: time `belief_features` alone across belief sizes (clean per-world sweep signal) | `src/belief_sweep_bench.cpp` |
| `chocofarm-belief-filter-bench` | Microbench + A/B: branchless `filter_inplace` vs branchy `erase(remove_if)`, assert byte-identical (~27.6% of K=32 profile) | `src/belief_filter_bench.cpp` |
| `chocofarm-search-runtime-bench` | Throughput bench: `SerialRuntime` vs `PoolRuntime` over Gumbel-AZ decisions; reports decisions/s + parallel speedup; links Threads | `src/search_runtime_bench.cpp` |
| `chocofarm-wire-bench` | Synchronous over-the-wire bench: `SerialRuntime` + remote `ZmqNetClient` vs the Python `InferenceServer` (RTT + server-forward) | `src/wire_bench.cpp` |
| `chocofarm-local-mlp-bench` | C++-native local-MLP axis: same batch with in-process `NetForward` off the redis weight seam (vs wire RTT) | `src/local_mlp_bench.cpp` |
| `chocofarm-fiber-proto` | Option-A proof: run unchanged `run_search` inside a boost.context fiber, assert fiber-driven == direct; links boost.context | `src/fiber_proto.cpp` |
| `chocofarm-dealer-probe` | Transport probe: N concurrent DEALER↔ROUTER requests, proves framing + greedy-drain server batching | `src/dealer_probe.cpp` |
| `chocofarm-wire-parallel-bench` | Wire-parallel bench: K tree-fibers on one thread, batch-submit parked leaves over DEALER; boost.context + libzmq | `src/wire_parallel_bench.cpp` |
| `chocofarm-wire-pool-bench` | Production-shaped wire-parallel POOL bench: T threads × K fibers, greedy-async DEALER drain (batch/threads from `runtime_config` SSOT); boost.context + libzmq + threads | `src/wire_pool_bench.cpp` |

Source for the table: `cpp/CMakeLists.txt:161-324`.

> **Gotcha.** The wire/parallel/probe targets that talk to the Python inference service (`chocofarm-wire-bench`, `-wire-parallel-bench`, `-wire-pool-bench`, `-dealer-probe`, `-zmq-net-probe`) require a running server (e.g. spun by `cpp/parity/wire_bench.py`, cited at `cpp/CMakeLists.txt:262`); the redis-seam targets (`-net-dump`, `-local-mlp-bench`) require published weights on **redis 6379** (the hp-registry / weight-read seam). The `*-dump`/`*-check` parity fixtures and the isolated `belief-*` benches need neither redis nor a server.

### Build a single target

```bash
cd /home/bork/w/vdc/1/chocofarm
cmake --build cpp/build --target chocofarm-wire-pool-bench -j4
```

The `--target <name>` argument takes any target from the table above. Swap the build dir (`cpp/build` / `cpp/build-zdd-on` / `cpp/build-profile`) to build that target in the corresponding variant — e.g. `cmake --build cpp/build-profile --target chocofarm-cpp-runner -j4` for a frame-pointer profiling runner.

## Correctness, equality & parity tests

This section is the copy-pasteable runbook for the **bit-exact / behavioral-parity** gates that pin the C++ port against its Python single-source-of-truth. Two kinds of check live here: **in-language bit-exact A/Bs** (two independent computations must be byte-identical) and **cross-language parity** (C++ vs Python, either byte-exact for serialization contracts or `max|Δ| < 1e-4` for ML float math — the ADR-0012 P6 behavioral bar).

Shared facts used throughout:

- **Python interpreter:** `/home/bork/w/vdc/venvs/generic/bin/python`
- **cwd for every command below:** `/home/bork/w/vdc/1/chocofarm` (the repo root — the C++ tools resolve `chocofarm/data/...` and `FeatureBuilder`'s `feature_layout.json` relative to it; `tests/test_cpp_runner.py:108-110,133-135` pass `cwd=REPO` + `PYTHONPATH=REPO`).
- **Default-build binaries:** `cpp/build/` (`tests/test_cpp_runner.py:69-88`).
- **Instance / faces fixtures:** `chocofarm/data/instance.json`, `chocofarm/data/faces.json` (`tests/test_cpp_runner.py:89-90`).
- **Opt-in gate:** the binary-dependent legs run **only** under `CHOCO_RUN_CPP=1` *and* a freshly built binary — a **stale binary fails rather than skips** (`tests/test_cpp_runner.py:92-98`). Always rebuild first: `cmake --build cpp/build`.

### The full C++ parity gate

```bash
cd /home/bork/w/vdc/1/chocofarm
cmake --build cpp/build                     # MANDATORY: a stale binary reds, it does not skip
CHOCO_RUN_CPP=1 PYTHONPATH=. /home/bork/w/vdc/venvs/generic/bin/python \
  -m pytest tests/test_cpp_runner.py -q
# expected: 25 passed, 2 skipped
```

Source: the file-level run recipe is `tests/test_cpp_runner.py:96`; the gate env var `CHOCO_RUN_CPP` is `tests/test_cpp_runner.py:97`.

**What it proves** — 27 test functions; the **2 skips are the retired NMCS pair** (`@pytest.mark.skip`, NMCS parity retired until nmcs-init resumes — `tests/test_cpp_runner.py:273,288`), unconditionally skipped regardless of env. The 25 that pass cover, in layers:

- **Always-on seam invariants** (no C++, no redis): each ported search is a `Policy` subclass registered in `SOLVERS` — `random`, `nmcs`, `ismcts`, gumbel (`tests/test_cpp_runner.py:147-179`); plus the `RandomPolicy` legality-mask invariants and λ-threading (`:182-228`).
- **Deterministic logic parity** (C++ dump binary vs Python, RNG abstracted behind a scripted source, **no redis**): ISMCTS selects the same action (`tests/test_cpp_runner.py:305-317`, `cpp/parity/ismcts_logic.py`); Gumbel-AZ **structure** (1a) executes the same action + improved-π argmax and pins the two Danihelka invariants (`:320-340`, `cpp/parity/gumbel_logic.py`); Gumbel **mixed-precision near-tie** (1b) reproduces Python's float32-prior × float64-Q exactly while the all-float64 control diverges (`:343-359`, `cpp/parity/gumbel_precision.py`).
- **Aggregate behavioral parity** (runner + redis; skips if redis down): ISMCTS aggregates within Monte-Carlo CI (`tests/test_cpp_runner.py:362-376`). The full P6/P7 harness (`:243-254`, `cpp/parity/parity.py`) and net-forward parity (`:257-270`, `cpp/parity/net_parity.py`, `max|Δvalue|` and `max|Δlogit| < 1e-4`).
- **Runtime/runner traces**: SerialRuntime seam-faithfulness (`tests/test_cpp_runner.py:379-391`); PoolRuntime **bit-identical** to SerialRuntime per task (`:394-408`); the wire benchmark both axes (`:411-424`); the fiber prototype bit-identical to direct `run_search` (`:427-440`); the C++ actor loop / exit_loop swap / online-reconfig serve path (`:443-658`).
- **It shells out to the two belief checks** documented below: `test_cpp_belief_cache_collision_guard` (`tests/test_cpp_runner.py:101-113`) and `test_cpp_belief_sweep_oracle` (`:116-141`), each asserting `RESULT: PASS` in the subprocess stdout.

**Gotchas:**
- Redis must be up for the redis-gated legs to actually run rather than skip — the **worker-transport instance `127.0.0.1:6380`** (`_redis_up()` connects via `chocofarm.az.transport`, `tests/test_cpp_runner.py:234-240`). If 6380 is down those legs **skip silently**, so a green run with 6380 down is **not** the full 25.
- The oracle leg (`tests/test_cpp_runner.py:116-141`) additionally asserts `RESULT: PASS flat-vs-bitset A/B` and `RESULT: SKIP not in stdout` — i.e. it pins that the live instance gates the **bitset** arm ON and did not silently fall back to flat.

### The belief equality A/Bs — `chocofarm-belief-sweep-oracle-check`

This is the in-language **bit-exact** oracle for the belief sweep (`cpp/src/belief_sweep_oracle_check.cpp`). Protocol: `--instance <p> --faces <p>` (`belief_sweep_oracle_check.cpp:23,443-446`). Pure compute — **no redis, no net, no layout file** (`:24-25`), but run from REPO for parity with the other gates.

#### Default build — the flat oracle + the flat-vs-bitset A/B

```bash
cd /home/bork/w/vdc/1/chocofarm
cmake --build cpp/build
./cpp/build/chocofarm-belief-sweep-oracle-check \
  --instance chocofarm/data/instance.json \
  --faces    chocofarm/data/faces.json
```

**What each `RESULT:` line asserts:**

- `RESULT: PASS belief-sweep bit-exact oracle (...)` — the production fused branchless `chocofarm::belief_features` (over contiguous `env.face_masks()`) is **byte-for-byte** equal to an independent naive `env.observe`-count reference, same `*inv` convention, over the sampled beliefs (empty, prefixes 1…|worlds|, and two strided subsets). Asserted by `equal_features` (`==` on the `double` vectors/scalars is exact here — counts ≥ 0, `inv > 0`, so no NaN/−0.0). Source: `belief_sweep_oracle_check.cpp:470-488`.
- `GATE: kW64=… mask_bytes=… budget=… inline_cap=… => use_bitset=true` — the gate decision printed so a flip is diagnosable (`belief_sweep_oracle_check.cpp:490-506`). On the live instance the bitset arm gates **ON** (`mask_bytes ≈ 121.5 KiB ≤ 128 KiB`).
- `RESULT: PASS flat-vs-bitset A/B byte-identical (...)` — builds a `FlatBelief` and a `BitsetBelief` **directly** (bypassing the gate) for each sampled belief and asserts byte-identity across **every** env seam op — `nb`, `empty`, `belief_key`, `marginals`, `informative` (per detector), `legal_actions`, `belief_features`, `world_at_rank` (all `r`), `sample_world` (256 draws over a fixed RNG) — both static and re-asserted after a full filter sequence (all detectors + all treasures). Flat is the reference; any divergence is a bitset bug (ADR-0002). Source: `belief_sweep_oracle_check.cpp:508-532` (`ab_identical` at `:168-196`).
  - **Gotcha:** if the env gated bitset **OFF** it would print `RESULT: SKIP flat-vs-bitset A/B (...)` instead (`belief_sweep_oracle_check.cpp:512-516`). The pytest gate (the full parity gate) explicitly fails on a `SKIP` — so a SKIP here means a dim change pushed `mask_bytes`/`kW64` past the cache/inline budget and the arm silently fell back.

#### ZDD-enabled build — the flat-vs-ZDD FEATURE A/B + the operator== construction-order net

The ZDD arm is compiled only under `-DCHOCO_BELIEF_ZDD=ON` (CMake option default **OFF** — `cpp/CMakeLists.txt:49,150-158`; the Part-3/4 code is `#ifdef CHOCO_BELIEF_ZDD`, `belief_sweep_oracle_check.cpp:198,534-606`). A pre-built ZDD binary already exists at `cpp/build-zdd-on/`:

```bash
cd /home/bork/w/vdc/1/chocofarm
# Build the ZDD-on variant if needed (separate build dir; keeps the default build flat+bitset):
cmake -S cpp -B cpp/build-zdd-on -DCHOCO_BELIEF_ZDD=ON
cmake --build cpp/build-zdd-on
# Run the ZDD-on oracle (prints Parts 1-2 AND Parts 3-4):
./cpp/build-zdd-on/chocofarm-belief-sweep-oracle-check \
  --instance chocofarm/data/instance.json \
  --faces    chocofarm/data/faces.json
```

**Gotcha:** the full pytest gate runs only the **default** `cpp/build/` binary (`tests/test_cpp_runner.py:88`), which is built ZDD-**off**, so the ZDD Parts never appear there — run the `build-zdd-on` binary by hand to exercise them. Additional `RESULT:` lines:

- `GATE-ZDD: use_zdd=… (...)` — the opt-in arm gate (`belief_sweep_oracle_check.cpp:541-542`).
- `RESULT: PASS flat-vs-ZDD FEATURE A/B byte-identical (...)` — every **FEATURE** op (`nb`, `empty`, `marginals`, `informative`, `legal_actions`, `belief_features`) is byte-identical flat-vs-ZDD **and** `members(Z)` is set-equal to the flat belief, static + after a full restrict-op filter sequence (`restrict_cover`/`restrict_var`) + the four empty-RESULT restrict→BOT nets. **Asymmetry (by design):** the **sampling** trio (`sample_world`/`world_at_rank`/`belief_key`) RE-BASELINES — the ZDD's canonical member order ≠ `worlds()`-rank order — and is **NOT** asserted equal (`belief_sweep_oracle_check.cpp:198-204,217-246,534-586`).
- `RESULT: PASS ZDD operator== construction-order invariance (...)` — falsifies the canonical-layout assumption behind the O(|Z|) structural `==` directly: the **same** world-family built two different ways (build-from-worlds vs `full_belief()`+`restrict_var`, and again vs `full_belief()`+`restrict_cover`) must compare structurally **equal**, while **disjoint** families (treasure present/absent, cover hold/fail) must compare **not-equal** (no false positives). Source: `belief_sweep_oracle_check.cpp:299-435,588-605`.

Any failure prints `RESULT: FAIL <op> ...` and exits nonzero (`belief_sweep_oracle_check.cpp:51,479-485,521-525,546-550,596-599`).

### The belief-memo collision-guard — `chocofarm-belief-cache-check`

```bash
cd /home/bork/w/vdc/1/chocofarm
cmake --build cpp/build
PYTHONPATH=. ./cpp/build/chocofarm-belief-cache-check \
  --instance chocofarm/data/instance.json \
  --faces    chocofarm/data/faces.json
```

Source binary: `cpp/src/belief_cache_check.cpp`; usage `--instance <p> --faces <p>` (`belief_cache_check.cpp:38-42`). In the pytest gate it runs `cwd=REPO`, `PYTHONPATH=REPO` so `FeatureBuilder` finds `feature_layout.json` (`tests/test_cpp_runner.py:101-110`).

**What it proves** — the `FeatureBuilder` belief-memo keys on the `(count, first, last)` `belief_key` fingerprint, which is collision-**resistant**, not collision-free; correctness rests on the full bw-equality guard. The tool **forces a collision** — two distinct flat beliefs `{1,3,5}` and `{1,4,5}` (same `count=3`, `first=1`, `last=5`, different middle world) that share a fingerprint (`belief_cache_check.cpp:54-59`) — and asserts:
- **(a)** the colliding beliefs get **distinct** features (the guard is not mis-served — `f1 != f2`, `belief_cache_check.cpp:61-63`);
- **(b)** a true cache **HIT** is bit-identical to the belief's first build, for both `bw1` and `bw2` (`belief_cache_check.cpp:65-68`);
- **(c)** the warm-cache value equals a **cold** recompute on a fresh `FeatureBuilder` — hit == miss (`belief_cache_check.cpp:70-73`).

On success: `RESULT: PASS belief-memo collision guard + hit-exactness (dim=…, shared fingerprint=(…))` (`belief_cache_check.cpp:75-80`). Pure `FeatureBuilder` — **no redis, no net** (`belief_cache_check.cpp:9`).

### Wire / result drift & ZMQ round-trip equality

#### `tests/test_wire_drift.py` — the Python↔C++ wire/config drift net

Four legs are **always-on** (parse the C++ mirror headers as text — **no C++ binary, no redis**, run in the default suite); one leg is opt-in (`CHOCO_RUN_CPP`, needs `g++`/`clang++`). Source file-level doc: `tests/test_wire_drift.py:1-51`; the opt-in gate `_RUN_CPP` is `tests/test_wire_drift.py:80`.

```bash
cd /home/bork/w/vdc/1/chocofarm
# always-on legs only (no build, no redis needed):
PYTHONPATH=. /home/bork/w/vdc/venvs/generic/bin/python -m pytest tests/test_wire_drift.py -q

# + the opt-in cross-language golden round-trip (compiles cpp/parity/wire_golden.cpp with bare g++):
CHOCO_RUN_CPP=1 PYTHONPATH=. /home/bork/w/vdc/venvs/generic/bin/python \
  -m pytest tests/test_wire_drift.py -q
```

**What it proves:**
- **Leg 1 — layout agreement:** the C++ mirror header `constexpr` literals equal the Python SSOTs — `PROTOCOL_VERSION`, field byte widths (`VERSION_BYTES`/`COUNT_BYTES`/`FLOAT_BYTES`), the wire float dtype `<f4`, the result block dtype `<f4`/itemsize 4, and the canonical block `ORDER (X,PI,M,Y)` + per-block `RANKS (2,2,2,1)` (`tests/test_wire_drift.py:134-189`).
- **Leg 1b — codec derives from spec:** the real `inference_wire` encode/decode emit/read exactly the spec's little-endian-f32 bytes (independent reference encoder), and the real result codec round-trips X/PI/M/Y through an in-memory fake redis with exact values (`tests/test_wire_drift.py:201-313`).
- **Leg 2 — drift-catch self-checks (negative proofs):** a one-sided perturbation of a constant (either side) makes the agreement assertion raise — so the net is demonstrated non-vacuous (`tests/test_wire_drift.py:334-369,444-461,503-518`).
- **Leg 3 — weight-blob dtype invariant:** the one non-self-describing cross-language literal, `<f8` (float64), is pinned on both the Python packer and the C++ `parse_manifest` reject (`tests/test_wire_drift.py:375-392`).
- **Leg 4 — actor-config + control-protocol agreement:** the C++ `ACTOR_CONFIG_FIELDS`/`ACTOR_CONFIG_MUT` equal the Python `FIELD_NAMES`/`MUT_CLASSES`, the `actor_config.cpp` `j.at("…")` parse keys equal the field set, and the control `MSG_TYPES`/`ERROR_TAGS` agree (`tests/test_wire_drift.py:403-518`).
- **Leg 5 (opt-in) — cross-language golden round-trip:** Python encodes golden request/response (incl. the `n_actions=0` value-only edge) and result blocks → a standalone `cpp/parity/wire_golden.cpp` (compiled `g++ -std=c++23`, mirror headers only) decodes & re-encodes → **byte-for-byte** identity (a serialization contract is exact, not float-tolerant). Source: `tests/test_wire_drift.py:561-609`. **Gotcha:** skips (not fails) without `CHOCO_RUN_CPP` or a c++23-capable compiler (`tests/test_wire_drift.py:534-547,561,585`).

#### `tests/test_zmq_net_cpp.py` — the cross-language inference round-trip

The C++ `ZmqNetClient` encodes a request → the in-process Python `InferenceServer` runs the one `forward_core` → the C++ client decodes `(value, logits)`; asserted to match the local float32 `forward_core` within `1e-4` (transitively pins it to C++ `NetForward`). **No redis** (server spun in-process with `StaticParamsSource`). Opt-in: needs `CHOCO_RUN_CPP=1` + a built `chocofarm-zmq-net-probe` + `pyzmq`; skips otherwise. Source: `tests/test_zmq_net_cpp.py:1-25`.

**Gotcha — zmq needs a real context, so disable the sandbox** (per the file's own recipe, `tests/test_zmq_net_cpp.py:18-21`):

```bash
cd /home/bork/w/vdc/1/chocofarm
cmake --build cpp/build
CHOCO_RUN_CPP=1 PYTHONPATH=. /home/bork/w/vdc/venvs/generic/bin/python \
  -m pytest tests/test_zmq_net_cpp.py -q -s
```

#### `tests/test_zmq_inference.py` — the Python-side wire codec + batching round-trip

Always-on legs (no server, no redis, no network): the wire codec round-trips `encode∘decode == identity` (including the value-only `logits=None` case) and **rejects malformed frames loudly** (bad protocol byte, wrong length, NaN feature — ADR-0002); the `Net` Protocol is satisfied structurally by both impls; the greedy-drain microbatch logic collapses B requests to one forward with per-request scatter. The opt-in leg spins the server in a thread (`StaticParamsSource`, no redis) for the full client parity harness. Source: `tests/test_zmq_inference.py:1-25`.

```bash
cd /home/bork/w/vdc/1/chocofarm
# always-on codec/batching legs (no build, no redis):
PYTHONPATH=. /home/bork/w/vdc/venvs/generic/bin/python -m pytest tests/test_zmq_inference.py -q
# + the opt-in in-process server parity leg (real zmq context -> -s to disable sandbox):
CHOCO_RUN_CPP=1 PYTHONPATH=. /home/bork/w/vdc/venvs/generic/bin/python \
  -m pytest tests/test_zmq_inference.py -q -s
```

### General gotchas across this section

- Always `cmake --build cpp/build` (or the relevant build dir) **before** any `CHOCO_RUN_CPP=1` run — the opt-in C++ legs **fail on a stale binary, they do not skip** (`tests/test_cpp_runner.py:92-98`).
- The redis-gated parity legs use the **worker-transport** instance `127.0.0.1:6380` (`volatile-lru`), not the hp-registry instance `127.0.0.1:6379`; with 6380 down they skip rather than fail, so confirm it is up before claiming the full 25-pass run.
- These checks are CPU-bound and single/short — for the longer aggregate-parity or runtime-bench legs, pin to the 4 vCPUs with `taskset -c 0,1,2,3 …` and launch hang-prone runs under `PYTHONFAULTHANDLER=1` so a `kill -ABRT` yields thread tracebacks (`ptrace_scope=1` blocks `py-spy` attach).

## Search profiling, benchmarks & sweeps

These harnesses profile the C++ Gumbel search's wire path (the greedy-async `chocofarm-wire-pool-bench` driver) and the isolated belief primitives. The **critical structural fact**: every Python harness stands up its **own in-process `InferenceServer`** via `wire_server.build_server` (`/home/bork/w/vdc/chocobo/profiles/wire_server.py:32`) on a daemon thread, runs the timed C++ client as a subprocess, then tears the server down. **Do not hand-roll a `nohup`'d server + a foreground `sleep` to wait for it** — the sandbox blocks foreground `sleep`, and these harnesses are self-contained precisely to avoid that. The server holds one frozen `StaticParamsSource` net (no redis; `hidden` is the only knob that affects forward cost — `wire_server.py:32-35`).

Shared facts for the harnesses below:
- Python: `/home/bork/w/vdc/venvs/generic/bin/python`; harnesses `sys.path.insert(0, REPO)` themselves, so no `PYTHONPATH` export is needed (`fiber_sweep.py:23-24`, `wire_server.py:26-27`).
- Client binary the sweep/grid/h2h drive: `/home/bork/w/vdc/1/chocofarm/cpp/build/chocofarm-wire-pool-bench` (hardcoded, `fiber_sweep.py:26`, `fiber_grid.py:24`). Built via `add_executable(chocofarm-wire-pool-bench …)` (`cpp/CMakeLists.txt:313`).
- Instance/faces data: `chocofarm/data/instance.json`, `chocofarm/data/faces.json` (`fiber_sweep.py:27-28`).
- 4-vCPU host: pin with `taskset -c 0,1,2,3`. The h2h harness pins internally per-process (`headtohead_profile.py:13,47`), so pin its *parent* to the server cores only.
- `pool_dps` = decisions/second = `n_tasks / wall` — completed search trees per wall-second, the throughput figure of merit (`wire_pool_bench.cpp:251`).

### Single-threaded fiber-count sweep (`fiber_sweep.py`)

Pins `--threads 1`, so `fibers_per_thread = batch = K` — "number of fibers" is literally `--batch K` (`fiber_sweep.py:6-8`; derived `fibers_per_thread = ceil(batch/threads)` in `wire_pool_bench.cpp:147`). One warm server (hidden=256), warms two throwaway batch shapes, then sweeps `--ks`, best-of-`--reps` (noise / one-time JIT only drags a sample DOWN — `fiber_sweep.py:9-10`). Prints a table and `SWEETSPOT: K=… (best pool_dps=…)`.

The `sims256` regime (the production runner's `hidden=256` n-sims-256 regime, the one the local profile flagged at 41% matvec — `wire_server.py:15-16`) is exactly the default flags, but spell them out for the record:

```bash
taskset -c 0,1,2,3 /home/bork/w/vdc/venvs/generic/bin/python \
  /home/bork/w/vdc/chocobo/profiles/fiber_sweep.py \
  --hidden 256 --n-sims 256 --m 24 --max-depth 24 --lam 0.0855 \
  --tasks 256 --reps 3 \
  --ks 1,2,4,8,12,16,24,32,48,64,96,128 \
  2>&1 | tee ~/w/vdc/chocobo/profiles/fiber-sweep-sims256.log
```

- cwd: irrelevant (subprocess cwd is forced to `REPO` — `fiber_sweep.py:40`). PYTHONPATH: not needed.
- Defaults already match the above (`fiber_sweep.py:52-59`); the explicit form is the copy-pasteable record.
- Endpoint `tcp://127.0.0.1:5762` (`fiber_sweep.py:29`). The C++ per-leaf RPC timeout is hardcoded `--timeout-ms 30000` inside the harness (`fiber_sweep.py:39`); the subprocess wall timeout is 180s/point (`fiber_sweep.py:33`).
- **Gotcha:** a non-`RESULT: PASS` at any K aborts that rep loudly (ADR-0002 — `fiber_sweep.py:41-43`); the server is rebuilt per-process, never restarted *between* K-points (design-note §7.1 — `fiber_sweep.py:8-9`).

### The 2-D (threads × fibers/thread) grid (`fiber_grid.py`)

Sibling of the 1-D sweep, same sleep-free `build_server` plumbing. Sweeps `T ∈ --ts` × `F ∈ --fs`, runs each cell with `--batch = T*F` (T worker threads each multiplexing F fibers — `fiber_grid.py:7-9,77`). **Cells where `T*F > --tasks` are SKIPPED** (the pool can't fill — underfilled is not a valid throughput point — `fiber_grid.py:9,78-79`). Best-of-`--reps`; prints the grid and `OPTIMAL: threads=… fibers/thread=… (batch=…) -> … decisions/s` (`fiber_grid.py:110-112`).

```bash
taskset -c 0,1,2,3 /home/bork/w/vdc/venvs/generic/bin/python \
  /home/bork/w/vdc/chocobo/profiles/fiber_grid.py \
  --hidden 256 --n-sims 256 --m 24 --max-depth 24 --lam 0.0855 \
  --tasks 256 --reps 2 \
  --ts 1,2,4 --fs 16,32,64,128 \
  2>&1 | tee ~/w/vdc/chocobo/profiles/grid-2026-06-17.log
```

- Defaults match (`fiber_grid.py:46-54`). Same endpoint `5762` (`fiber_grid.py:27`); subprocess wall timeout 300s/cell (`fiber_grid.py:30`).
- **Gotcha:** because `T*F > tasks` cells are skipped, raising `--fs` without raising `--tasks` silently shrinks the grid. To probe larger batches, bump `--tasks` accordingly. On the 4-vCPU host the realized parallel ceiling is ~1.9×, so `--ts` past 4 is not informative.

### Head-to-head bitset-vs-ZDD perf profile (`headtohead_profile.py`)

Spins one in-process server (hidden=256, endpoint `tcp://127.0.0.1:5764`, distinct from the sweep's 5762 to avoid a leftover-server clash — `headtohead_profile.py:38,84`), **warms at the SAME `--K`** so the JAX JIT compiles for those batch shapes and the recorded run is steady-state (`headtohead_profile.py:8-9,92-93`), then `perf record`s the client. **Self-time only — no `--call-graph`** (the question is "which rep is faster + where does the ZDD arm spend", which self-time answers without boost.context fiber-fp call-graph noise — `headtohead_profile.py:17-21`). The parent is pinned to the server cores; the profiled client is pinned to core 0 *inside* the harness (`taskset -c 0 perf record … --` — `headtohead_profile.py:47`), separating server and client.

Run once per arm (`--label bitset` against the `cpp/build` binary; `--label zdd` against the ZDD-on binary `cpp/build-zdd-on/chocofarm-wire-pool-bench`):

```bash
# bitset arm
taskset -c 1,2 /home/bork/w/vdc/venvs/generic/bin/python \
  /home/bork/w/vdc/chocobo/headtohead_profile.py \
  --bin /home/bork/w/vdc/1/chocofarm/cpp/build/chocofarm-wire-pool-bench \
  --out ~/w/vdc/chocobo/profiles/h2h-bitset.perf.data \
  --label bitset --tasks 2048 --warm-tasks 512 --K 512

# zdd arm (ZDD-on build dir)
taskset -c 1,2 /home/bork/w/vdc/venvs/generic/bin/python \
  /home/bork/w/vdc/chocobo/headtohead_profile.py \
  --bin /home/bork/w/vdc/1/chocofarm/cpp/build-zdd-on/chocofarm-wire-pool-bench \
  --out ~/w/vdc/chocobo/profiles/h2h-zdd.perf.data \
  --label zdd --tasks 2048 --warm-tasks 512 --K 512
```

- The fixed inner flags are `--n-sims 256 --m 24 --max-depth 24 --threads 1 --batch K` (`headtohead_profile.py:43-45`); `--bin/--out/--label` are required (`headtohead_profile.py:59-61`).
- Then read each `.perf.data` separately. **CRITICAL read caveat:** use `perf report --stdio -i <file>`, **NOT** the interactive TUI. perf 7.0.12 (confirmed on this host) has a TUI bug on `FINISHED_ROUND`; `--stdio` parses the same file fine (`headtohead_profile.py:21`):

```bash
perf report --stdio -i ~/w/vdc/chocobo/profiles/h2h-bitset.perf.data \
  > ~/w/vdc/chocobo/profiles/h2h-bitset-self.txt
perf report --stdio -i ~/w/vdc/chocobo/profiles/h2h-zdd.perf.data \
  > ~/w/vdc/chocobo/profiles/h2h-zdd-self.txt
```

- Self-time (no `--call-graph` recorded) means the report is a flat per-symbol self-time table — read `belief_features` / alloc / `ZddBelief::operator==` directly off it (`headtohead_profile.py:17-20`). Do not pass `-g`/`--call-graph` to the report; nothing was recorded for it.
- **Gotcha:** `ptrace_scope=1` on this host blocks `py-spy` attach, which is why this uses `perf record` on a subprocess launched *under* perf rather than attaching. The warmup at the same K is load-bearing: profile without it and you measure JIT compilation, not steady state.

### RSS / no-OOM probe (`zdd_mem.py`)

Escalates `n_sims`/`max_depth` at small `--tasks 32 --batch 16` to find the copy-bloat-vs-depth point where the ZDD-on build OOMs, each step wrapped in `/usr/bin/time -v` to read "Maximum resident set size" (`zdd_mem.py:18-24`). Drives the **ZDD-on** binary `cpp/build-zdd-on/chocofarm-wire-pool-bench` (`zdd_mem.py:15`), own server on endpoint `tcp://127.0.0.1:5766` (`zdd_mem.py:9`). Breaks the escalation loop on the first non-zero return code (`zdd_mem.py:25-26`).

```bash
taskset -c 0,1,2,3 /home/bork/w/vdc/venvs/generic/bin/python \
  /home/bork/w/vdc/chocobo/zdd_mem.py \
  2>&1 | tee ~/w/vdc/chocobo/profiles/zdd-mem-probe.log
```

- No CLI flags — the (tasks, batch, n_sims, max_depth) escalation ladder is hardcoded (`zdd_mem.py:18`). Subprocess wall timeout 600s/step (`zdd_mem.py:21`).
- **Gotcha:** requires `/usr/bin/time` (the GNU binary, present on this host) — the shell builtin `time` does not accept `-v`. Requires the `cpp/build-zdd-on/` build to exist; it is a *separate* build dir from `cpp/build/`.

### Standalone belief micro-benchmarks (no server, no net)

These are pure C++ executables timing the belief primitives in isolation — no `InferenceServer`, no redis, no wire (`belief_sweep_bench.cpp:6-8`, `belief_filter_bench.cpp:14`). They take only `--instance`/`--faces` and self-time, so run them directly (cwd irrelevant; pin to keep a clean core).

**belief-sweep-bench** — times `chocofarm::belief_features` (the popcount-marg + observe-cover sweep, ~81% of the client thread at K=16) across belief sizes `nb`; reports ns/call, ns/world, Mworlds/s, Mcalls/s (slope = per-world cost, intercept = fixed per-call cost — `belief_sweep_bench.cpp:6-12`):

```bash
taskset -c 0 /home/bork/w/vdc/1/chocofarm/cpp/build/chocofarm-belief-sweep-bench \
  --instance /home/bork/w/vdc/1/chocofarm/chocofarm/data/instance.json \
  --faces    /home/bork/w/vdc/1/chocofarm/chocofarm/data/faces.json \
  --budget-s 0.3 \
  | tee ~/w/vdc/chocobo/profiles/belief-sweep-baseline.txt
```

`--budget-s` defaults to 0.3s/point (`belief_sweep_bench.cpp:48-49`). Target `add_executable(chocofarm-belief-sweep-bench …)` (`cpp/CMakeLists.txt:241`).

**belief-filter-bench** — the filter-compaction A/B gate: times the production `chocofarm::filter_inplace` idiom against a hand-branchless candidate on *realistic* observation-filtered beliefs (NOT a prefix), filter-isolated by subtracting a restore-only baseline, and **asserts bit-exactness** (`RESULT: PASS`/`FAIL`). `speedup` column = candidate/idiom, `>1 ⇒ idiom faster` (`belief_filter_bench.cpp:16-18`):

```bash
taskset -c 0 /home/bork/w/vdc/1/chocofarm/cpp/build/chocofarm-belief-filter-bench \
  --instance /home/bork/w/vdc/1/chocofarm/chocofarm/data/instance.json \
  --faces    /home/bork/w/vdc/1/chocofarm/chocofarm/data/faces.json \
  --budget-s 0.3 --trials 5 \
  | tee ~/w/vdc/chocobo/profiles/belief-filter.txt
```

`--budget-s` 0.3 (per point, split across trials), `--trials` 5 are the defaults (`belief_filter_bench.cpp:81-84`). Target `add_executable(chocofarm-belief-filter-bench …)` (`cpp/CMakeLists.txt:249`).

- **Gotcha:** both are self-timing loops that run *to a time budget*, not a fixed iteration count — a contended core inflates ns/world. Pin to an idle core (`taskset -c 0`) and don't co-run the sweep/grid against them. The filter-bench returns exit 1 + `RESULT: FAIL` if the candidate ever disagrees byte-for-byte with the idiom (`belief_filter_bench.cpp:139`); a non-zero exit there is a correctness failure, not a perf result.

## AlphaZero training & the C++ actor loop

The ExIt training loop is `chocofarm.az.exit_loop` (`chocofarm/az/exit_loop.py:531` `main()` → `:217` `run()`). Every iteration runs GENERATE → TRAIN → EVALUATE → CHECKPOINT (docstring `exit_loop.py:9-18`). `--ckpt-dir` is the only required flag (`exit_loop.py:588`). All commands run from the repo root with the registry-seeding env in place.

### Three executor modes (how GENERATION is fanned out)

The executor is selected once at launch (`exit_loop.py:348-369`); eval/train/replay/checkpoint always stay in-process Python regardless of mode.

| Mode | Selecting flag | Source | Cores actually used |
|---|---|---|---|
| C++ Gumbel actor | `--cpp-runner <path>` | `exit_loop.py:349-356` | **~1** (serial actor) |
| Python process pool | `--workers N` (`N>0`) | `exit_loop.py:357-367` | up to N (one per pinned core) |
| Serial in-process | neither | `exit_loop.py:368-369` | 1 |

**`--cpp-runner` takes precedence over `--workers`.** The `if args.cpp_runner:` branch is tested first (`exit_loop.py:349`); the `elif cfg0.par.workers ...` pool branch (`exit_loop.py:357`) is only reached when `--cpp-runner` is unset. This is documented at `exit_loop.py:346-347` and in the `--cpp-runner` help text (`exit_loop.py:568-574`: "Overrides --workers"). `--workers` defaults to **4** (`exit_loop.py:561`), so the pool is the default multi-core path; pass `--workers 0` for the true serial A/B baseline (`exit_loop.py:562-563`).

The multi-core path is **`--workers N`** with `--cores` (default `0,1,2,3`, `exit_loop.py:564-565`); each worker is pinned to a distinct core inside `ParallelExecutor` (`exit_loop.py:363-364`).

### C++ actor: hard fail-loud constraints (and the 1-core caveat)

The C++ actor (`CppActorExecutor`, `cpp_executor.py:63`) plays the Sequential-Halving survivor at **temperature 0 every ply** and emits the **pure-MC λ-return** only. Two generation-shaping knobs cannot cross the C++ wire and **fail loud** (ADR-0002):

- **`explore_plies` must be 0.** `generate()` raises `RuntimeError` if `explore_plies > 0` (`cpp_executor.py:123-129`). The loop default is `--explore-plies 4` (`exit_loop.py:545-546`), so you **must** pass `--explore-plies 0`.
- **Value target must be pure-MC.** `generate()` raises if `n_step is not None` or `lam_blend < 1.0` (`cpp_executor.py:114-118`). The loop defaults `--td-lambda 1.0` / `--n-step None` (`exit_loop.py:581-586`) are already pure-MC, so the default is fine — but **never** combine `--cpp-runner` with `--td-lambda <1` or `--n-step`. Use `--workers>0` for Part-B blends or an exploration prefix (`cpp_executor.py:24-39`).

**GOTCHA — the `--serve` actor generates episodes SERIALLY (single thread).** `CppActorExecutor.cores` is the empty list — "the runner is one subprocess running episodes serially; no per-worker core pin" (`cpp_executor.py:86-87`). The threads/fibers batched sweep runtime is **not** wired into the actor. So a `--cpp-runner` run consumes **~1 core regardless of `taskset -c 0,1,2,3`**, while a `--workers 4` run uses up to 4. A naive `--cpp-runner`-vs-Python-pool wall-clock comparison is therefore apples-to-oranges (1 core vs N) — pin cores and executor consistently and compare per-core throughput, not raw wall-clock. (This is exactly why the reference `launch-sims256-2026-06-17.sh:5-8` runs the two experiments **sequentially** under the same `taskset`.)

### Smoke test (validates the C++ actor end-to-end in ~minutes)

The persistent `--serve` runner is spawned lazily on first `generate` and pinged for readiness (`cpp_executor.py:163-174`); a missing binary or dead runner raises loudly there. Binary path and instance/faces JSON are taken from the reference script (`launch-sims256-2026-06-17.sh:14-16`); the `--cpp-instance` / `--cpp-faces` defaults are `chocofarm/data/{instance,faces}.json` (`exit_loop.py:575-579`). Use a **fresh** `--ckpt-dir` basename (see registry gotcha below).

```bash
cd /home/bork/w/vdc/1/chocofarm
PYTHONFAULTHANDLER=1 PYTHONPATH=. taskset -c 0,1,2,3 \
  /home/bork/w/vdc/venvs/generic/bin/python -m chocofarm.az.exit_loop \
  -I 2 -E 8 --m 8 --n-sims 16 --hidden 64 --eval-n 8 \
  --explore-plies 0 --td-lambda 1.0 \
  --cpp-runner /home/bork/w/vdc/1/chocofarm/cpp/build/chocofarm-cpp-runner \
  --cpp-instance /home/bork/w/vdc/1/chocofarm/chocofarm/data/instance.json \
  --cpp-faces    /home/bork/w/vdc/1/chocofarm/chocofarm/data/faces.json \
  --ckpt-dir  /home/bork/w/vdc/chocobo/runs/cpp-smoke-$(date +%Y%m%d-%H%M%S) \
  --tb-logdir /home/bork/w/vdc/chocobo/tb/az/cpp-smoke-$(date +%Y%m%d-%H%M%S)
```

Flag sources: `-I/-E/--m/--n-sims` (`exit_loop.py:533-539`), `--hidden` (`:550`), `--eval-n` (`:547`), `--explore-plies`/`--td-lambda` (`:545,:581`), `--cpp-runner`/`--cpp-instance`/`--cpp-faces` (`:568,:575,:578`), `--tb-logdir`/`--ckpt-dir` (`:587,:588`). `taskset -c 0,1,2,3` is pinning convention (note: the actor still uses ~1 core); `PYTHONFAULTHANDLER=1` is the hang-debug launch convention. The runner is reaped on every exit path via `close()` (`cpp_executor.py:250-261`).

**PASS criteria** (cf. the recorded smoke `cpp-smoke-2026-06-17.log:9-13`):
- Two `iter N/2` lines on stdout with finite `rate=` and finite `gen/train/eval` seconds (printed `exit_loop.py:514-517`).
- Checkpoints in the ckpt dir: `net_iter000.npz`, `net_iter001.npz`, `latest_net.npz`, `history.json` (written every iteration, `exit_loop.py:488-498`).
- A non-zero TensorBoard event file `events.out.tfevents.*` in the tb dir (`exit_loop.py:306-308,500-512`).

### hp-registry gotcha: `--experiment-id` defaults to the ckpt-dir basename

`experiment_id = args.experiment_id or os.path.basename(os.path.normpath(args.ckpt_dir))` (`exit_loop.py:238`); `--experiment-id` defaults to `None` (`exit_loop.py:593-595`). On launch `seed_registry` writes the config blob **only if the key does not already exist** — it is idempotent and a re-launch **re-binds to the existing blob rather than clobbering it** (`registry.py:475-507`). **Reusing a ckpt-dir basename therefore re-binds the OLD seeded config blob** (and adopts any operator overrides recorded against it), silently ignoring most of your new CLI flags. Always use a **fresh basename** for a fresh run (the smoke command above appends a timestamp). To override explicitly without changing the ckpt dir, pass `--experiment-id <fresh-id>`.

Inspect a config read-only via the operator CLI or directly (registry lives on the noeviction **6379** instance, keyed `choco:hp:<id>`, `registry.py:11-24,94-101`):

```bash
# operator CLI (validated pretty-print):
PYTHONPATH=. /home/bork/w/vdc/venvs/generic/bin/python -m chocofarm.hp.registry \
  get --experiment-id cpp-sims256-2026-06-17b
# raw blob, no decode:
redis-cli -p 6379 get 'choco:hp:cpp-sims256-2026-06-17b'
```

CLI source: `get` subcommand `registry.py:812-814` / `_cli_get` `registry.py:745-748`. (`set`/`init` exist too, `registry.py:816-829`.)

### sims256 reference config

The throughput-probe launcher is `~/w/vdc/chocobo/runs/launch-sims256-2026-06-17.sh` (Exp A = C++ actor lines 19-23, Exp B = matched 4-worker Python pool lines 25-28; both `--explore-plies 0 --td-lambda 1.0`, run sequentially under one `taskset`). The CLI line is `COMMON="-I 40 -E 300 --m 24 --n-sims 256 --seed 7 --explore-plies 0 --td-lambda 1.0"` (`launch-sims256-2026-06-17.sh:17`); everything else falls to schema defaults. The fully resolved registry blob `choco:hp:cpp-sims256-2026-06-17b` (read via `redis-cli -p 6379 get`) is:

```text
search:  m=24  n_sims=256  c_puct=1.25  c_visit=50.0  c_scale=1.0  c_outcome=2  max_depth=24  use_jax_mlp=false
loop:    iters(I)=40  episodes(E)=300  window(W)=5  lam=0.0855  explore_plies=0  seed=7
train:   epochs=2  batch=256  lr=0.001  l2=0.0001  alpha=1.0  beta=1.0  beta1=0.9  beta2=0.999  eps=1e-08
arch:    hidden=256  residual=false  init_seed=7  in_dim=241  n_actions=65  dtype=float32
eval:    eval_n=200  eval_seed=12345
value:   td_lambda=1.0  n_step=null   (pure-MC)
env:     entry=CSNE  max_steps=40  present_k=5  teleport_overhead=12.0
par:     workers=4  cores=0,1,2,3  redis_host=127.0.0.1  redis_port=6380  redis_db=0
feat:    per_treasure_channels=5  per_detector_channels=3  global_channels=6
```

To reproduce the **C++-actor** arm (uses ~1 core; ~330s gen + ~360s eval per iter per `cpp-sims256-2026-06-17b.log:9-23`):

```bash
cd /home/bork/w/vdc/1/chocofarm
PYTHONFAULTHANDLER=1 PYTHONPATH=. taskset -c 0,1,2,3 \
  /home/bork/w/vdc/venvs/generic/bin/python -m chocofarm.az.exit_loop \
  -I 40 -E 300 --m 24 --n-sims 256 --seed 7 --explore-plies 0 --td-lambda 1.0 \
  --cpp-runner /home/bork/w/vdc/1/chocofarm/cpp/build/chocofarm-cpp-runner \
  --cpp-instance /home/bork/w/vdc/1/chocofarm/chocofarm/data/instance.json \
  --cpp-faces    /home/bork/w/vdc/1/chocofarm/chocofarm/data/faces.json \
  --ckpt-dir  /home/bork/w/vdc/chocobo/runs/cpp-sims256-FRESHID \
  --tb-logdir /home/bork/w/vdc/chocobo/tb/az/cpp-sims256-FRESHID
```

To reproduce the **matched Python 4-worker pool** arm (`launch-sims256-2026-06-17.sh:25-28`), drop the three `--cpp-*` flags and add `--workers 4 --cores 0,1,2,3`. Replace `FRESHID` with an unused basename (registry gotcha above). The reference binary, instance, and faces paths are fixed at `launch-sims256-2026-06-17.sh:14-16`; the runner binary is `/home/bork/w/vdc/1/chocofarm/cpp/build/chocofarm-cpp-runner` (built; confirm it exists before launching).

## Infrastructure: TensorBoard, redis, experiment records & debugging

This section is copy-pasteable. Throughout, `<py>` is the project interpreter `/home/bork/w/vdc/venvs/generic/bin/python` (CLAUDE.md:155-156). All experiment paths live under `~/w/vdc/chocobo/`.

### TensorBoard daemon

Runs streamed by training/eval loops are written under `~/w/vdc/chocobo/tb/az/`; the daemon serves `--logdir tb/az` on port 6006 (CLAUDE.md:159-161). Start it (detached, logging to a file you can tail):

```bash
cd /home/bork/w/vdc/chocobo && \
nohup /home/bork/w/vdc/venvs/generic/bin/python -m tensorboard.main \
  --logdir tb/az --port 6006 --bind_all \
  > tb/tb-daemon.log 2>&1 &
```

- **Source:** the logdir/port facts are CLAUDE.md:159-161 (`--logdir tb/az` on 6006). `cd /home/bork/w/vdc/chocobo` makes `tb/az` resolve correctly; `--bind_all` exposes it on the VM's interface.
- **How new runs appear:** every loop that takes `--tb-logdir` writes a `tensorboardX.SummaryWriter` to that path; pointing it at a subdir of `tb/az` makes it a new run under the daemon. Loops with a `--tb-logdir` flag: `chocofarm/az/exit_loop.py:587`, `chocofarm/az/cpp_actor_loop.py:89` (example target `/home/bork/w/vdc/chocobo/tb/az/cpp-actor-loop`, cpp_actor_loop.py:30), `chocofarm/az/train_value.py:112`, `chocofarm/eval/eval_az.py:123`. The daemon auto-discovers new subdirs; no restart needed.
- **Check it's up:**

```bash
pgrep -af tensorboard.main          # PID + full argv if running; empty if not
curl -fsS http://127.0.0.1:6006/ >/dev/null && echo "TB up on 6006"
```

- **Gotcha:** the daemon only sees subdirs *under* the `--logdir` it was started with. A `--tb-logdir` outside `tb/az` (e.g. a bare `tb/az_exit_loop`, exit_loop.py:29) will not show up on this daemon — write under `tb/az/<name>` or start a second daemon on another port.

### The two redis instances

chocofarm runs **two** redis instances by design, one per role; `chocofarm/config.py` is the single owner of which-redis-for-what (config.py:5-21).

| Port | Role | Policy | Persistence | Source |
|------|------|--------|-------------|--------|
| **6380** db 0 | TRANSPORT — AZ parallel-loop weight + result blobs (`az/transport.py`, `az/parallel.py`) | `allkeys-lru`, `maxmemory` cap | EPHEMERAL; keys carry **1h TTLs**, read+deleted within the iteration; nothing needs to survive a restart | config.py:9-14, 34-37 |
| **6379** db 0 | REGISTRY — hp config blobs (`hp/registry.py`) | `noeviction`, no `maxmemory` cap | DISK-PERSISTED (RDB `save`); registry blobs carry **NO TTL**, must survive a restart and never be evicted | config.py:16-21, 39-42 |

Check each is up:

```bash
redis-cli -p 6380 ping     # transport  -> PONG
redis-cli -p 6379 ping     # hp registry -> PONG
```

- **WARNING — do NOT flush either instance.** `FLUSHALL`/`FLUSHDB` on **6379** destroys the hp registry, which is the disk-persisted system of record (no TTL, must never be evicted — config.py:16-21). Flushing **6380** mid-run drops the live weight/result blobs an iteration is mid-read on. Neither flush is part of any normal operation.
- **Note (code vs. shorthand):** the worker-transport instance is `allkeys-lru` per config.py:11-13, not `volatile-lru`; its keys are short-lived under 1h TTLs and the LRU policy is only the safety-net evicting anything left behind.
- **Override (don't edit code):** each role's host/port/db is env-overridable via distinct families — `CHOCO_TRANSPORT_REDIS_{HOST,PORT,DB}` (config.py:50-59) and `CHOCO_REGISTRY_REDIS_{HOST,PORT,DB}` (config.py:62-71). Timeouts are shared: `CHOCO_REDIS_SOCKET_TIMEOUT` (default 60s), `CHOCO_REDIS_CONNECT_TIMEOUT` (default 10s) (config.py:46-47, 74-82).

### Experiment-record locations

| Path | Holds |
|------|-------|
| `~/w/vdc/chocobo/runs/` | checkpoints (npz) + `.log` files (CLAUDE.md:158-159) |
| `~/w/vdc/chocobo/tb/az/` | TensorBoard event files (CLAUDE.md:159-161) |
| `~/w/vdc/chocobo/profiles/` | perf data + sweep logs |

- **Source:** `runs/` and `tb/az/` are CLAUDE.md:158-161; all three dirs exist on disk under `~/w/vdc/chocobo/`. Both `runs/` and `tb/` are gitignored (CLAUDE.md:160-161).
- **Rule — NEVER discard experiment output; keep it under `~/w/vdc`, never `/tmp`** (CLAUDE.md:162). Checkpointing happens *every* iteration so a timeout/restart loses nothing (exit_loop.py:17-18) — point `--ckpt-dir` at `~/w/vdc/chocobo/runs/<run>`, not `/tmp` (the `/tmp/...` ckpt-dirs in `docs/results/*.md` are illustrative, not the durable convention).
- **Gotcha:** `--ckpt-dir`/`--tb-logdir` are relative to the **process cwd**. Either pass absolute paths under `~/w/vdc/chocobo/` (as cpp_actor_loop.py:30 does), or launch from `~/w/vdc/chocobo` so `runs/...` and `tb/az/...` resolve there.

### Core-pinning

The host is a 4-vCPU libvirt VM; pin execution to cores 0–3, parallel ceiling ~1.9× (CLAUDE.md:163-164).

- **Loops with a built-in `--cores` flag** (preferred — the worker pool reads it and pins each worker): default is already `0,1,2,3`:

```bash
PYTHONPATH=. /home/bork/w/vdc/venvs/generic/bin/python -m chocofarm.az.exit_loop \
  --cores 0,1,2,3   # ... remaining args
```

  **Source:** `--cores` default `0,1,2,3` at exit_loop.py:564-565; cores parsed and handed to `ParallelExecutor` at exit_loop.py:363-364.

- **Single-process / no `--cores` flag** — wrap with `taskset`:

```bash
PYTHONPATH=. taskset -c 0,1,2,3 /home/bork/w/vdc/venvs/generic/bin/python -m <module> ...
# e.g. the dual-bound validator, pinned to one core:
PYTHONPATH=. timeout 600 taskset -c 3 /home/bork/w/vdc/venvs/generic/bin/python \
  -m chocofarm.bounds.eval_bound --validate
```

  **Source:** `taskset -c` usage shown in eval_bound.py:19-20.

- **Gotcha:** ~1.9× is the realistic parallel ceiling on 4 vCPUs (CLAUDE.md:164) — do not expect ~4×; oversubscribing past the 4 cores degrades throughput.

### Debugging a hang / wedge

`ptrace_scope=1` on this host means **`py-spy` cannot attach to a running process** — you must launch under faulthandler and signal it (CLAUDE.md:164-166).

- **Launch hang-prone code with faulthandler armed:**

```bash
PYTHONPATH=. PYTHONFAULTHANDLER=1 taskset -c 0,1,2,3 \
  /home/bork/w/vdc/venvs/generic/bin/python -m <module> ...
```

- **Dump every thread's traceback** of a wedged process:

```bash
kill -ABRT <pid>     # faulthandler prints all thread tracebacks to the process's stderr
```

  **Source:** the `PYTHONFAULTHANDLER=1` + `kill -ABRT` recipe and the `ptrace_scope=1` / py-spy constraint are CLAUDE.md:130, 164-166. When investigating a hang, ask for the `kill -ABRT` traceback rather than inferring from the symptom (CLAUDE.md:128-131; the cautionary instance is `docs/notes/jaxtrain-deadlock-rca.md`).

- **Sandbox gotcha — a foreground `sleep` is blocked.** To wait on a condition (e.g. "redis is up", "the daemon bound 6006"), do **not** put a bare `sleep` in a foreground command. Use an until-loop with a sub-second sleep inside a **backgrounded** task, or a self-contained harness. Example — wait for the TB daemon to answer, then report:

```bash
( until curl -fsS http://127.0.0.1:6006/ >/dev/null 2>&1; do sleep 0.2; done; \
  echo "TB ready" ) &
```

  The same pattern waits on `redis-cli -p 6380 ping` returning `PONG` before kicking off a parallel run.

## Conventions cheat-sheet

- **Python interpreter:** `/home/bork/w/vdc/venvs/generic/bin/python` — the shared scratch venv (JAX/optax/numba). Most modules need `PYTHONPATH=.` from the repo root; the profiling harnesses `sys.path.insert` themselves.
- **Two redis instances:** `127.0.0.1:6380` db 0 = **transport** (`allkeys-lru`, ephemeral 1h-TTL weight/result blobs — `az/transport.py`/`az/parallel.py`); `127.0.0.1:6379` db 0 = **hp registry** (`noeviction`, disk-persisted, no-TTL config blobs keyed `choco:hp:<id>`). Check with `redis-cli -p 6380 ping` / `redis-cli -p 6379 ping`. **Never `FLUSHALL`/`FLUSHDB` either.**
- **Core-pinning:** 4-vCPU host, parallel ceiling ~1.9×. Prefer a loop's `--cores 0,1,2,3` flag; otherwise wrap with `taskset -c 0,1,2,3`. Do not oversubscribe past 4 cores.
- **Hang debugging:** `ptrace_scope=1` blocks `py-spy` attach — launch under `PYTHONFAULTHANDLER=1`, then `kill -ABRT <pid>` to dump every thread's traceback to the process's stderr. Ask for the `kill -ABRT` traceback rather than inferring from the symptom.
- **`perf report` caveat:** read `.perf.data` with `perf report --stdio -i <file>`, **NOT** the interactive TUI — perf 7.0.12 on this host has a TUI bug on `FINISHED_ROUND`; `--stdio` parses the same file fine.
- **Foreground-sleep sandbox pitfall:** a bare foreground `sleep` is blocked. Never hand-roll a `nohup`'d server + foreground `sleep` to wait for it (the profiling harnesses are self-contained precisely to avoid this); to wait on a condition, use an until-loop with a sub-second `sleep` inside a **backgrounded** task, e.g. `( until curl -fsS http://127.0.0.1:6006/ >/dev/null 2>&1; do sleep 0.2; done; echo "TB ready" ) &`.
