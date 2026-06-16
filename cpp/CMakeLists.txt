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

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

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
  src/transport.cpp
  src/net.cpp
  src/zmq_net_client.cpp
  src/policy.cpp
  src/nmcs.cpp
  src/ismcts.cpp
  src/gumbel.cpp
  src/search_runtime.cpp
  src/runner.cpp
  src/actor_config.cpp
  src/serve.cpp
)
target_include_directories(chocofarm_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(chocofarm_core PUBLIC nlohmann_json::nlohmann_json ${HIREDIS_LINK} ${ZMQ_LINK} Threads::Threads)
target_compile_options(chocofarm_core PRIVATE -Wall -Wextra)

add_executable(chocofarm-cpp-runner src/main.cpp)
target_link_libraries(chocofarm-cpp-runner PRIVATE chocofarm_core)
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

# The SearchRuntime seam's self-contained SEAM-FAITHFULNESS check (NOT the runner): runs SerialRuntime
# over a batch of independent tasks against a deterministic, stateless, in-process net, and asserts each
# task's executed action matches a direct GumbelAZPolicy::decide with the same seed — proving the seam
# does not perturb the search (docs/design/cpp-search-runtime.md §7.1). No redis, no weights, no stdin.
# Separate from the runner (P3, one-owner).
add_executable(chocofarm-serial-runtime-check src/serial_runtime_check.cpp)
target_link_libraries(chocofarm-serial-runtime-check PRIVATE chocofarm_core)
target_compile_options(chocofarm-serial-runtime-check PRIVATE -Wall -Wextra)

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

# boost.context — the Option-A fiber mechanism (docs/notes/cpp-continuation-refactor-decision-2026-06-16.md).
# The resumable-search prototype runs the UNCHANGED run_search inside a stackful fiber. Located via
# find_package(Boost) with a find_library fallback (boost ships no pkg-config; the lib is libboost_context).
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

# A fifth tiny parity fixture (NOT the runner): constructs a ZmqNetClient (the REMOTE NetEvaluator),
# RPCs the running Python inference service with feature vectors on stdin, and prints the returned
# (value, logits) — so the cross-language round-trip test (tests/test_zmq_net_cpp.py) can drive the C++
# client against the Python InferenceServer and hold the result to the local C++ NetForward (< 1e-4).
# Separate from the runner (P3, one-owner): this tool owns the inference-client round-trip fixture.
add_executable(chocofarm-zmq-net-probe src/zmq_net_probe.cpp)
target_link_libraries(chocofarm-zmq-net-probe PRIVATE chocofarm_core)
target_compile_options(chocofarm-zmq-net-probe PRIVATE -Wall -Wextra)
