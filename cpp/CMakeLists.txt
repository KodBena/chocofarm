# cpp/CMakeLists.txt
# Purpose: build the chocofarm C++ search/sim runner (the dumb-random seam-proof MVP, ADR-0012's
#   C++ section / scaling-and-cpp-seam.md Shape A). C++23 (for std::expected — ADR-0012 P9 rule 5).
#   Dependencies are self-contained:
#   nlohmann/json is fetched via FetchContent (no system install needed); hiredis is found via
#   pkg-config (a system package: libhiredis-dev / hiredis-devel — the one documented system dep).
#
# Build:   cmake -S cpp -B cpp/build && cmake --build cpp/build
# Runner:  cpp/build/chocofarm-cpp-runner --instance chocofarm/data/instance.json \
#              --faces chocofarm/data/faces.json --run R --res-token T --episodes 300 --lam 0.1
#
# Public Domain (The Unlicense).
cmake_minimum_required(VERSION 3.16)
project(chocofarm_cpp_runner LANGUAGES CXX)

# Emit compile_commands.json so clangd resolves each TU with its EXACT flags (-std=c++23, the -I dirs) —
# the LSP's single source of truth, no hand-duplicated flags to drift. clangd auto-discovers it in this
# build/ subdir. (A repo-root .clangd adds a -std floor for headers opened with no including TU in view.)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# Build for the HOST CPU (-march=native): unlocks AVX2/FMA/BMI2 on this i5-6600 (Skylake) so the
# branchless integer belief sweep (features.cpp belief_features, the §A.4 form) AUTO-VECTORIZES — ~1.94x
# on the isolated sweep, ~1.29x end-to-end at the K=32 wire-pool sweetspot, with no code change. Pins the
# binary to this host's ISA (it may SIGILL on an older CPU) — correct for this build-and-run-locally
# scratch VM; pass -DCHOCO_MARCH_NATIVE=OFF for a portable build. FMA fusion shifts the FEATURE vector by
# ~1 ULP (within the ADR-0012 P6 behavioral bar vs Python; the legal MASK stays bit-exact — it is sign /
# integer tests), netted by the gumbel parity.
option(CHOCO_MARCH_NATIVE "Compile for the host CPU (-march=native): AVX2/FMA auto-vectorization" ON)
if(CHOCO_MARCH_NATIVE)
  add_compile_options(-march=native)
endif()

# The OPT-IN belief-as-diagram (ZDD) arm — the §B.4(b) graduation
# (belief_features_and_decision_diagram_note.md Part B; docs/design/cpp-belief-zdd-onramp.md;
# docs/design/cpp-belief-rep-scoping.md). DEFAULT OFF: the default build's belief surface is
# flat+bitset ONLY (the live two arms), byte-for-byte unchanged — the ZDD engine TU + the env-op ZDD
# bodies are NOT compiled, the Belief variant has two alternatives, and the env-op visits' third arm
# vanishes (the empty CHOCO_ZDD_ELSE macro). ON: -DCHOCO_BELIEF_ZDD=ON adds ZddBelief as a THIRD
# Belief variant arm, compiles the engine (belief_zdd_engine.cpp) + the ZDD seam bodies (env_zdd.cpp)
# into chocofarm_core, and the gate SELECTS the ZDD arm (full_belief returns a ZddBelief) so the whole
# search runs on the maintained diagram — for the head-to-head profile vs the bitset (and the large-N
# hedge). The flag is propagated as a PUBLIC compile-definition so every consumer of chocofarm_core
# (the runner, the benches, the A/B oracle, the tests' tools) sees the SAME Belief variant shape (else
# the variant size/alternatives desync across TUs — an ODR hazard). Pass -DCHOCO_BELIEF_ZDD=ON to a
# SEPARATE build dir (e.g. cpp/build-zdd) so the default build dir stays the live flat+bitset.
option(CHOCO_BELIEF_ZDD "Opt-in: add the maintained ZDD belief arm (§B.4(b)) as a third Belief variant arm" OFF)

# Threads (pthread): the SearchRuntime's PoolRuntime spawns std::thread workers (src/search_runtime.cpp
# is in chocofarm_core), so the core links Threads::Threads PUBLIC and every consumer gets it.
find_package(Threads REQUIRED)

include(FetchContent)

# ---- nlohmann/json (vendored via FetchContent — the build is self-contained) ----
# Prefer an installed package if present (offline / packaged builds); otherwise fetch a pinned tag.
find_package(nlohmann_json 3.2.0 QUIET)
if(NOT nlohmann_json_FOUND)
  FetchContent_Declare(
    nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG        v3.11.3
    GIT_SHALLOW    TRUE
  )
  set(JSON_BuildTests OFF CACHE INTERNAL "")
  FetchContent_MakeAvailable(nlohmann_json)
endif()

# ---- hiredis (system dependency — documented in cpp/README.md) ----
# hiredis ships no CMake config on most distros, so locate it via pkg-config, falling back to a
# manual find_path/find_library. This is the one system dependency the runner declares.
find_package(PkgConfig QUIET)
set(HIREDIS_FOUND FALSE)
if(PkgConfig_FOUND)
  pkg_check_modules(HIREDIS IMPORTED_TARGET hiredis)
  if(HIREDIS_FOUND)
    set(HIREDIS_LINK PkgConfig::HIREDIS)
  endif()
endif()
if(NOT HIREDIS_FOUND)
  find_path(HIREDIS_INCLUDE_DIR hiredis/hiredis.h)
  find_library(HIREDIS_LIBRARY hiredis)
  if(HIREDIS_INCLUDE_DIR AND HIREDIS_LIBRARY)
    add_library(hiredis_manual INTERFACE)
    target_include_directories(hiredis_manual INTERFACE ${HIREDIS_INCLUDE_DIR})
    target_link_libraries(hiredis_manual INTERFACE ${HIREDIS_LIBRARY})
    set(HIREDIS_LINK hiredis_manual)
    set(HIREDIS_FOUND TRUE)
  endif()
endif()
if(NOT HIREDIS_FOUND)
  message(FATAL_ERROR
    "hiredis not found. Install it (Debian/Ubuntu: libhiredis-dev; Fedora/openSUSE: hiredis-devel) "
    "or point CMAKE_PREFIX_PATH at it. It is the runner's one system dependency (see cpp/README.md).")
endif()

# ---- libzmq (system dependency — the Shape B inference client's transport) ----
# The C++ ZmqNetClient (the remote NetEvaluator, docs/design/zmq-inference-service.md) speaks the libzmq
# C API (zmq.h). Located via pkg-config exactly like hiredis (cppzmq's zmq.hpp is NOT used / required),
# with a manual find_path/find_library fallback. This is the second documented system dependency.
set(ZMQ_FOUND FALSE)
if(PkgConfig_FOUND)
  pkg_check_modules(ZMQ IMPORTED_TARGET libzmq)
  if(ZMQ_FOUND)
    set(ZMQ_LINK PkgConfig::ZMQ)
  endif()
endif()
if(NOT ZMQ_FOUND)
  find_path(ZMQ_INCLUDE_DIR zmq.h)
  find_library(ZMQ_LIBRARY zmq)
  if(ZMQ_INCLUDE_DIR AND ZMQ_LIBRARY)
    add_library(zmq_manual INTERFACE)
    target_include_directories(zmq_manual INTERFACE ${ZMQ_INCLUDE_DIR})
    target_link_libraries(zmq_manual INTERFACE ${ZMQ_LIBRARY})
    set(ZMQ_LINK zmq_manual)
    set(ZMQ_FOUND TRUE)
  endif()
endif()
if(NOT ZMQ_FOUND)
  message(FATAL_ERROR
    "libzmq not found. Install it (Debian/Ubuntu: libzmq3-dev; Fedora/openSUSE: zeromq-devel / "
    "libzmq5) or point CMAKE_PREFIX_PATH at it. It is the inference client's system dependency "
    "(the C API zmq.h; cppzmq is NOT required — see cpp/README.md).")
endif()

# ---- the runner core (a library so tests/tools can link it; main is a thin shell) ----
add_library(chocofarm_core
  src/instance.cpp
  src/env.cpp
  src/features.cpp
  src/batch_predict.cpp
  src/feature_layout.cpp
  src/transport.cpp
  src/net.cpp
  src/zmq_net_client.cpp
  src/policy.cpp
  src/nmcs.cpp
  src/ismcts.cpp
  src/gumbel.cpp
  src/gumbel_cursor.cpp
  src/search_runtime.cpp
  src/runner.cpp
  src/actor_config.cpp
  src/serve.cpp
)
target_include_directories(chocofarm_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(chocofarm_core PUBLIC nlohmann_json::nlohmann_json ${HIREDIS_LINK} ${ZMQ_LINK} Threads::Threads)
target_compile_options(chocofarm_core PRIVATE -Wall -Wextra)

# The OPT-IN ZDD belief arm (§B.4(b)): when CHOCO_BELIEF_ZDD=ON, compile the engine + the env-op ZDD
# bodies INTO chocofarm_core and define CHOCO_BELIEF_ZDD PUBLICly (so every consumer's Belief variant has
# the same three-arm shape — an ODR consistency requirement). When OFF neither TU is compiled and the
# definition is absent, so the default build is byte-for-byte the live flat+bitset (the two arms + the
# two-alternative variant). The TUs are added here (not as a separate library) so the runner + the benches
# + the A/B oracle all link the SAME core with the SAME variant shape.
if(CHOCO_BELIEF_ZDD)
  target_sources(chocofarm_core PRIVATE src/belief_zdd_engine.cpp src/env_zdd.cpp)
  target_compile_definitions(chocofarm_core PUBLIC CHOCO_BELIEF_ZDD)
endif()

# boost.context — the Option-A fiber mechanism (docs/notes/cpp-continuation-refactor-decision-2026-06-16.md).
# The resumable-search prototype runs the UNCHANGED run_search inside a stackful fiber. Located via
# find_package(Boost) with a find_library fallback (boost ships no pkg-config; the lib is libboost_context).
# Resolved HERE (before the first consumer, chocofarm-cpp-runner) so ${BOOST_CONTEXT_LINK} is set when the
# runner's wire generation path (src/runner_wire_batched.cpp) links it (the runner is the production --serve
# binary; the fiber benches/checks below reuse the same ${BOOST_CONTEXT_LINK}).
find_package(Boost QUIET COMPONENTS context)
if(Boost_FOUND)
  set(BOOST_CONTEXT_LINK Boost::context)
else()
  find_library(BOOST_CONTEXT_LIBRARY boost_context REQUIRED)
  find_path(BOOST_CONTEXT_INCLUDE boost/context/fiber.hpp REQUIRED)
  add_library(boost_context_manual INTERFACE)
  target_include_directories(boost_context_manual INTERFACE ${BOOST_CONTEXT_INCLUDE})
  target_link_libraries(boost_context_manual INTERFACE ${BOOST_CONTEXT_LIBRARY})
  set(BOOST_CONTEXT_LINK boost_context_manual)
endif()

# The runner gains the WIRE generation path (serve.handle_generate dispatches on --infer-endpoint to
# run_episodes_wire_batched, docs/design/cpp-wire-generation-roadmap.md Phase D+E). That driver uses
# TreeState -> boost/context/fiber.hpp, so src/runner_wire_batched.cpp is compiled INTO this executable
# (kept out of chocofarm_core — boost off the core link surface, the established discipline) and
# ${BOOST_CONTEXT_LINK} is added. zmq is PUBLIC/transitive via chocofarm_core (CMakeLists:147) — NOT
# re-added (CRITIQUE E1). The serial/one-shot CLI path links boost too but does not use fibers (acceptable).
add_executable(chocofarm-cpp-runner src/main.cpp src/runner_wire_batched.cpp)
target_link_libraries(chocofarm-cpp-runner PRIVATE chocofarm_core ${BOOST_CONTEXT_LINK})
target_compile_options(chocofarm-cpp-runner PRIVATE -Wall -Wextra)

# A tiny parity fixture (NOT the runner): replays an action sequence and dumps the legality mask, so
# the harness can assert the C++ mask is bit-identical to Python's for the same (loc, belief). No
# redis. Kept separate from the runner (P3, one-owner).
add_executable(chocofarm-mask-dump src/mask_dump.cpp)
target_link_libraries(chocofarm-mask-dump PRIVATE chocofarm_core)
target_compile_options(chocofarm-mask-dump PRIVATE -Wall -Wextra)

# A second tiny parity fixture (NOT the runner): reads a published net off redis via the manifest
# (the SAME weight-read seam, no hardcoded offsets — P1), builds the C++ NetForward, and dumps the
# value + policy logits for feature vectors on stdin, so the harness can hold the C++ forward to the
# Python forward_core under the ADR-0012 P6 behavioral bar (max|Δ| < 1e-4). Separate from the runner
# (P3, one-owner): the runner owns the wire + episode loop, this tool owns the forward parity fixture.
add_executable(chocofarm-net-dump src/net_dump.cpp)
target_link_libraries(chocofarm-net-dump PRIVATE chocofarm_core)
target_compile_options(chocofarm-net-dump PRIVATE -Wall -Wextra)

# A third tiny parity fixture (NOT the runner): runs NMCSPolicy::search with a SCRIPTED, RNG-free
# WorldSource (sample_world -> bw[0]; playout_value -> a stdin FIFO) and prints the selected action,
# so the harness can feed the C++ NMCS and the Python NMCS the SAME leaf returns on a fixed
# (loc, belief) and assert the SAME action — the DETERMINISTIC nesting+selection logic check (ADR-0012
# P6), independent of RNG. Separate from the runner (P3, one-owner).
add_executable(chocofarm-nmcs-dump src/nmcs_dump.cpp)
target_link_libraries(chocofarm-nmcs-dump PRIVATE chocofarm_core)
target_compile_options(chocofarm-nmcs-dump PRIVATE -Wall -Wextra)

# A fourth tiny parity fixture (NOT the runner): runs ISMCTSPolicy::run_search with a SCRIPTED,
# RNG-free ISMCTSSource (sample_world -> bw[0]; expand_index -> a stdin FIFO mod n; leaf_value -> a
# stdin FIFO) and prints the selected action, so the harness can feed the C++ ISMCTS and the Python
# ISMCTS the SAME world/expansion/leaf draws on a fixed (loc, belief) and assert the SAME action —
# the DETERMINISTIC selection+nesting logic check (ADR-0012 P6), independent of RNG. Separate from
# the runner (P3, one-owner).
add_executable(chocofarm-ismcts-dump src/ismcts_dump.cpp)
target_link_libraries(chocofarm-ismcts-dump PRIVATE chocofarm_core)
target_compile_options(chocofarm-ismcts-dump PRIVATE -Wall -Wextra)

# A sixth tiny parity fixture (NOT the runner): runs GumbelAZPolicy::run_search with a SCRIPTED,
# RNG-free NetEvaluator leaf (coarse, well-separated (value, logits) per call — precision-INSENSITIVE,
# 1a) + a SCRIPTED GumbelSource (gumbel -> a stdin FIFO; sample_world -> bw[0] or a stdin world-index
# FIFO) and prints the executed action slot + the improved-π argmax slot, so the harness can feed the
# C++ Gumbel and the Python Gumbel the SAME gumbel/world draws + the SAME scripted leaf on a fixed
# (loc, belief) and assert the SAME executed action AND improved-π argmax — the DETERMINISTIC structure
# + selection logic check (ADR-0012 P6), independent of RNG. No redis (the scripted leaf is in-process).
# Separate from the runner (P3, one-owner).
add_executable(chocofarm-gumbel-dump src/gumbel_dump.cpp)
target_link_libraries(chocofarm-gumbel-dump PRIVATE chocofarm_core)
target_compile_options(chocofarm-gumbel-dump PRIVATE -Wall -Wextra)

# OPTION B foundation proof (NOT the runner): drive the Gumbel-AZ search through the explicit-state
# TreeCursor (gumbel_cursor.hpp) and assert BIT-IDENTITY (executed action + improved-π argmax + n_spent +
# survivor + the full leaf-request sequence) to a DIRECT run_search fed the same scripted leaves/draws —
# the Option-B analogue of fiber_proto.cpp's Option-A proof (docs/design/cpp-search-runtime.md §7.1/§7.2
# layer 2). Deterministic DetNet + CyclicGumbelSource, in-process; no redis, no weights. (P3, one-owner.)
add_executable(chocofarm-gumbel-cursor-proto src/gumbel_cursor_proto.cpp)
target_link_libraries(chocofarm-gumbel-cursor-proto PRIVATE chocofarm_core)
target_compile_options(chocofarm-gumbel-cursor-proto PRIVATE -Wall -Wextra)

# The SearchRuntime seam's self-contained SEAM-FAITHFULNESS check (NOT the runner): runs SerialRuntime
# over a batch of independent tasks against a deterministic, stateless, in-process net, and asserts each
# task's executed action matches a direct GumbelAZPolicy::decide with the same seed — proving the seam
# does not perturb the search (docs/design/cpp-search-runtime.md §7.1). No redis, no weights, no stdin.
# Separate from the runner (P3, one-owner).
add_executable(chocofarm-serial-runtime-check src/serial_runtime_check.cpp)
target_link_libraries(chocofarm-serial-runtime-check PRIVATE chocofarm_core)
target_compile_options(chocofarm-serial-runtime-check PRIVATE -Wall -Wextra)

# The FeatureBuilder belief-memo collision-guard check (NOT the runner): forces a (count,first,last)
# fingerprint COLLISION between two distinct beliefs and asserts the memo's full-equality guard serves
# each its OWN features + that a cache hit is bit-identical to a recompute (ADR-0011: net the guard,
# don't trust it). No redis, no net — pure FeatureBuilder. Separate from the runner (P3, one-owner).
add_executable(chocofarm-belief-cache-check src/belief_cache_check.cpp)
target_link_libraries(chocofarm-belief-cache-check PRIVATE chocofarm_core)
target_compile_options(chocofarm-belief-cache-check PRIVATE -Wall -Wextra)

# The belief-sweep BIT-EXACT oracle (NOT the runner): recomputes chocofarm::belief_features two independent
# ways — the production contiguous-mask/branchless/fused sweep vs a naive env.observe count, same `* inv`
# spec — and asserts byte-equality over a sample of beliefs. The in-language regression net the §A.4 rewrite
# and every later rung (SIMD/pos-popcount, the Part B diagram) diff against (ADR-0011). Separate (P3, one-owner).
add_executable(chocofarm-belief-sweep-oracle-check src/belief_sweep_oracle_check.cpp)
target_link_libraries(chocofarm-belief-sweep-oracle-check PRIVATE chocofarm_core)
target_compile_options(chocofarm-belief-sweep-oracle-check PRIVATE -Wall -Wextra)

# The ISOLATED belief-sweep microbenchmark (NOT the runner): times chocofarm::belief_features alone (no
# search / wire / cache / geometry) across belief sizes, so a sweep optimizer (pospopcount / inline-masks
# / decision-diagram) iterates with a clean per-world signal. Drives the compute via feature_compute.hpp.
# Separate from the runner (P3, one-owner).
add_executable(chocofarm-belief-sweep-bench src/belief_sweep_bench.cpp)
target_link_libraries(chocofarm-belief-sweep-bench PRIVATE chocofarm_core)
target_compile_options(chocofarm-belief-sweep-bench PRIVATE -Wall -Wextra)

# The BatchPredict (lever #3, IN-PROCESS) component bench + bit-identity gate (NOT the runner): the
# BatchFeaturizer's mask-resident tiled AVX2 belief sweep + the production assembly. The gate asserts every
# batched feature ROW == B per-leaf FeatureBuilder::build byte-for-byte; the A/B times the batched featurizer
# vs B per-leaf belief_features across B in {8,32,64} (interleaved/paired, bootstrap CI). Separate executable
# (ADR-0012 P3, one-owner). No redis/net — pure compute.
add_executable(chocofarm-batch-predict-bench src/batch_predict_bench.cpp)
target_link_libraries(chocofarm-batch-predict-bench PRIVATE chocofarm_core)
target_compile_options(chocofarm-batch-predict-bench PRIVATE -Wall -Wextra)

# The ISOLATED belief-FILTER microbench + bit-exact A/B (NOT the runner): times the branchless
# chocofarm::filter_inplace against the former branchy erase(remove_if) across belief sizes, and asserts
# byte-identical output. The belief UPDATE side of the same O(nb) belief problem (the K=32 native profile's
# ~27.6%). Separate executable (ADR-0012 P3, one-owner). No redis/net.
add_executable(chocofarm-belief-filter-bench src/belief_filter_bench.cpp)
target_link_libraries(chocofarm-belief-filter-bench PRIVATE chocofarm_core)
target_compile_options(chocofarm-belief-filter-bench PRIVATE -Wall -Wextra)

# The SearchRuntime throughput benchmark (NOT the runner): SerialRuntime vs PoolRuntime over a batch of
# independent Gumbel-AZ decisions against a local deterministic net, asserting bit-identical per-task
# results then reporting decisions/s + the parallel speedup (the C++-native-MLP / local-parallel config).
# Threads, so link Threads::Threads. Separate from the runner (P3, one-owner).
add_executable(chocofarm-search-runtime-bench src/search_runtime_bench.cpp)
target_link_libraries(chocofarm-search-runtime-bench PRIVATE chocofarm_core Threads::Threads)
target_compile_options(chocofarm-search-runtime-bench PRIVATE -Wall -Wextra)

# The over-the-wire SYNCHRONOUS benchmark (NOT the runner): SerialRuntime with a remote ZmqNetClient
# leaf against the running Python InferenceServer (spun by cpp/parity/wire_bench.py), measuring the
# wire RTT + server-forward cost per leaf. The wire-PARALLEL axis (fiber + DEALER pool) is a later chunk.
add_executable(chocofarm-wire-bench src/wire_bench.cpp)
target_link_libraries(chocofarm-wire-bench PRIVATE chocofarm_core)
target_compile_options(chocofarm-wire-bench PRIVATE -Wall -Wextra)

# The C++-NATIVE LOCAL MLP axis (the third leaf-eval regime): the SAME SerialRuntime batch as wire-bench,
# but with an in-process NetForward (read off the SAME redis weight seam, on weights the harness publishes
# from the SAME pack_net the wire server is seeded from) — so it compares "local forward" vs "wire RTT +
# server forward" on the IDENTICAL net (float32-equivalent). No socket, no boost (local + serial).
add_executable(chocofarm-local-mlp-bench src/local_mlp_bench.cpp)
target_link_libraries(chocofarm-local-mlp-bench PRIVATE chocofarm_core)
target_compile_options(chocofarm-local-mlp-bench PRIVATE -Wall -Wextra)

# The Option-A foundation proof (NOT the runner): run the unchanged run_search inside a boost.context
# fiber with a YieldingNetEvaluator, and assert fiber-driven == direct (the §7.1 validity precondition,
# near-trivial under A). Separate from the runner (P3, one-owner).
add_executable(chocofarm-fiber-proto src/fiber_proto.cpp)
target_link_libraries(chocofarm-fiber-proto PRIVATE chocofarm_core ${BOOST_CONTEXT_LINK})
target_compile_options(chocofarm-fiber-proto PRIVATE -Wall -Wextra)

# The non-blocking DEALER transport probe (NOT the runner): submits N concurrent requests to the ROUTER
# server and recvs N replies, proving the DEALER<->ROUTER framing + the server's greedy-drain batching
# (the wire-parallel leaf-rendezvous foundation). Reuses the shared inference_wire codec (P7).
add_executable(chocofarm-dealer-probe src/dealer_probe.cpp)
target_link_libraries(chocofarm-dealer-probe PRIVATE chocofarm_core)
target_compile_options(chocofarm-dealer-probe PRIVATE -Wall -Wextra)

# The over-the-wire PARALLEL benchmark (NOT the runner): K tree-fibers on one thread, batch-submitting
# parked leaves over the DEALER so the server batches them — the third §6-Q5 axis. boost.context + libzmq.
add_executable(chocofarm-wire-parallel-bench src/wire_parallel_bench.cpp)
target_link_libraries(chocofarm-wire-parallel-bench PRIVATE chocofarm_core ${BOOST_CONTEXT_LINK})
target_compile_options(chocofarm-wire-parallel-bench PRIVATE -Wall -Wextra)

# The production-shaped wire-PARALLEL POOL benchmark (NOT the runner): T threads x K fibers, greedy-async
# DEALER drain (no per-round barrier), batch/threads from runtime_config (SSOT). boost.context + libzmq + threads.
add_executable(chocofarm-wire-pool-bench src/wire_pool_bench.cpp)
target_link_libraries(chocofarm-wire-pool-bench PRIVATE chocofarm_core ${BOOST_CONTEXT_LINK} Threads::Threads)
target_compile_options(chocofarm-wire-pool-bench PRIVATE -Wall -Wextra)

# The Stage A PURE-TRANSPORT microbench (NOT the runner): pre-baked synthetic 241-float rows driven over
# the real ZMQ + corr-id WireLeafPool path to the bench-scoped JAX server, isolating the S (leaves/msg)
# and D (in-flight depth) transport knobs from the search (no MCTS). Maps the throughput surface for the
# eval-transport-adapter design's Stage A (docs/design/cpp-eval-transport-adapter.md §4). Reuses the
# shared WireLeafPool corr-id transport + inference_wire codec (P1/P7); no boost (no fibers/search).
# zmq is PUBLIC/transitive via chocofarm_core. Separate from the runner (P3, one-owner).
add_executable(chocofarm-stage-a-transport-bench src/stage_a_transport_bench.cpp)
target_link_libraries(chocofarm-stage-a-transport-bench PRIVATE chocofarm_core)
target_compile_options(chocofarm-stage-a-transport-bench PRIVATE -Wall -Wextra)

# A fifth tiny parity fixture (NOT the runner): constructs a ZmqNetClient (the REMOTE NetEvaluator),
# RPCs the running Python inference service with feature vectors on stdin, and prints the returned
# (value, logits) — so the cross-language round-trip test (tests/test_zmq_net_cpp.py) can drive the C++
# client against the Python InferenceServer and hold the result to the local C++ NetForward (< 1e-4).
# Separate from the runner (P3, one-owner): this tool owns the inference-client round-trip fixture.
add_executable(chocofarm-zmq-net-probe src/zmq_net_probe.cpp)
target_link_libraries(chocofarm-zmq-net-probe PRIVATE chocofarm_core)
target_compile_options(chocofarm-zmq-net-probe PRIVATE -Wall -Wextra)

# ---- the WIRE-batched generation driver (docs/design/cpp-wire-generation-roadmap.md, Phases A-F) ----
# run_episodes_wire_batched (runner_wire_batched.cpp) is the T×K greedy-async fiber-pool driver that
# resolves every Gumbel-AZ leaf REMOTELY on the JAX InferenceServer over a DEALER socket (no local
# forward). It uses TreeState -> boost/context/fiber.hpp, so it is kept OUT of chocofarm_core and compiled
# per-executable with ${BOOST_CONTEXT_LINK} added to every target that links it (mirroring wire-pool-bench).
# zmq is PUBLIC/transitive via chocofarm_core (CMakeLists:147) — NOT re-added (CRITIQUE E1).

# The Phase B SMOKE (NOT the runner): runs --episodes self-play episodes through run_episodes_wire_batched
# against a warm Python InferenceServer (stood up by cpp/parity/_wire_smoke_server.py), writing the four
# EpisodeBlocks per non-empty episode to redis — the Python harness reads them back + asserts sane shapes.
add_executable(chocofarm-wire-batched-smoke src/wire_batched_smoke.cpp src/runner_wire_batched.cpp)
target_link_libraries(chocofarm-wire-batched-smoke PRIVATE chocofarm_core ${BOOST_CONTEXT_LINK})
target_compile_options(chocofarm-wire-batched-smoke PRIVATE -Wall -Wextra)

# The Phase C CROSS-RUNTIME parity check (NOT the runner): drives the SAME corpus through serial
# run_episodes (local NetForward leaf) and wire run_episodes_wire_batched (the JAX server leaf), asserting
# layer-2 single-tree NeedsLeaf-sequence (RNG ctor + canned leaf, fibered==direct) + layer-4 aggregate
# action-dist/improved-π within a Monte-Carlo CI (the Q7 behavioral bar — NOT byte-identity). Uses redis
# (read-back). boost.context (the fiber mux). Separate (P3, one-owner).
add_executable(chocofarm-wire-batched-runtime-check src/wire_batched_runtime_check.cpp src/runner_wire_batched.cpp)
target_link_libraries(chocofarm-wire-batched-runtime-check PRIVATE chocofarm_core ${BOOST_CONTEXT_LINK})
target_compile_options(chocofarm-wire-batched-runtime-check PRIVATE -Wall -Wextra)

# The Stage B e2e A/B THROUGHPUT bench (NOT the runner): drives the REAL Gumbel-AZ search (every leaf
# remote on the JAX server) for a wall budget under ONE selectable transport mode (--wire-mode
# strict-barrier = arm 1, pipelined-bucket = arm 3), reporting decisions/s/core — the eval-transport-adapter
# Stage B A/B (docs/design/cpp-eval-transport-adapter.md §4). Links run_episodes_wire_batched (which carries
# both modes) + boost.context (the fiber mux). Separate from the runner (P3, one-owner).
add_executable(chocofarm-wire-ab-bench src/wire_ab_bench.cpp src/runner_wire_batched.cpp)
target_link_libraries(chocofarm-wire-ab-bench PRIVATE chocofarm_core ${BOOST_CONTEXT_LINK})
target_compile_options(chocofarm-wire-ab-bench PRIVATE -Wall -Wextra)

# ---- the Quantity<Tag,Rep> ZERO-COST elision A/B oracle (the maintainer's hard gate, mechanized) ----
# Compiles the SAME hot kernel (cpp/parity/quantity_elision_ab.cpp) twice, -O3 -march=native: once over the
# strong type Quantity<ProbeTag,uint32_t> (CHOCO_AB_TYPED=1) and once over the bare uint32_t the phantom
# wraps (=0), then objdump-disassembles the `hot` symbol in each and DIFFS them. A non-empty diff FAILS the
# build — the phantom is not zero-cost. This is the runnable substantiation behind the elision claim
# (ADR-0009; ADR-0000: net the property, don't assert it in prose). The static side lives in quantity.hpp's
# static_asserts (sizeof / trivially-copyable / standard-layout); this is the empirical instruction-identity
# side. Forced -O3 -march=native regardless of CMAKE_BUILD_TYPE (the gate is about THESE flags).
add_library(chocofarm-quantity-elision-typed OBJECT parity/quantity_elision_ab.cpp)
target_compile_definitions(chocofarm-quantity-elision-typed PRIVATE CHOCO_AB_TYPED=1)
target_compile_options(chocofarm-quantity-elision-typed PRIVATE -O3 -march=native -DNDEBUG)
target_include_directories(chocofarm-quantity-elision-typed PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_library(chocofarm-quantity-elision-raw OBJECT parity/quantity_elision_ab.cpp)
target_compile_definitions(chocofarm-quantity-elision-raw PRIVATE CHOCO_AB_TYPED=0)
target_compile_options(chocofarm-quantity-elision-raw PRIVATE -O3 -march=native -DNDEBUG)
target_include_directories(chocofarm-quantity-elision-raw PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_custom_target(chocofarm-quantity-elision-check ALL
  COMMAND ${CMAKE_COMMAND}
          -DTYPED_OBJ=$<TARGET_OBJECTS:chocofarm-quantity-elision-typed>
          -DRAW_OBJ=$<TARGET_OBJECTS:chocofarm-quantity-elision-raw>
          -P ${CMAKE_CURRENT_SOURCE_DIR}/parity/quantity_elision_check.cmake
  DEPENDS chocofarm-quantity-elision-typed chocofarm-quantity-elision-raw
  VERBATIM
  COMMENT "Quantity<Tag,Rep> zero-cost gate: objdump A/B of `hot` (typed vs raw) must be byte-identical")
