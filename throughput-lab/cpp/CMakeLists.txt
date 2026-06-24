# throughput-lab/cpp/CMakeLists.txt
# Purpose: build scaffolding for the clean-room throughput-lab PRODUCER + BOUNDARY (C++23, libzmq).
#   Defines the `tlab-producer` executable and links libzmq (-lzmq). The wire codec (wire.hpp) and the
#   boundary/producer contracts (boundary.hpp, producer.hpp) are header-only SSOTs; the build agents
#   supply the .cpp implementations listed below. ADR-0012 P3 one-owner: one TU per topology impl, so
#   no impl can be hand-copied into another and silently diverge.
# Public Domain (The Unlicense).

cmake_minimum_required(VERSION 3.16)
project(throughput_lab_producer CXX)

# Emit compile_commands.json for clangd (exact per-TU flags: -std=c++23 + the -I dirs). The chocofarm
# include path only enters this DB when configured with -DTLAB_REAL_GENERATOR=ON (that option adds it),
# so configure with the option ON for the real_gen_* TUs to resolve. clangd auto-discovers it here.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# C++23 — the contracts (wire.hpp, boundary.hpp, producer.hpp) use the modern types ADR-0012 P9
# mandates: std::span (C++20) for bounds-carrying views, std::expected (C++23) for typed failure,
# std::optional (C++17) for typed absence. The brief said "C++17", but P9 — which the brief itself
# invokes — requires std::span/std::expected, so the standard that honors the contracts is C++23
# (which is also what chocofarm's own cpp/ uses, so the testbed matches the parent toolchain). The
# wire codec + the contract headers are VERIFIED to compile clean here under -std=c++23 -Wall -Wextra.
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "build type" FORCE)
endif()

# -Wall -Wextra is the floor (ADR-0012 P9: the compiler is the first mechanization surface). A future
# clang-tidy modernize-*/cppcoreguidelines-* config is the per-class net when a reliquary form recurs.
add_compile_options(-Wall -Wextra)

# libzmq: prefer pkg-config (the usual libzmq install path), fall back to a plain find_library so the
# build works on a host without the .pc file. The target the executable links is `zmq_lib`.
find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
  pkg_check_modules(ZMQ QUIET libzmq)
endif()

if(ZMQ_FOUND)
  add_library(zmq_lib INTERFACE)
  target_include_directories(zmq_lib INTERFACE ${ZMQ_INCLUDE_DIRS})
  target_link_libraries(zmq_lib INTERFACE ${ZMQ_LIBRARIES})
  target_link_directories(zmq_lib INTERFACE ${ZMQ_LIBRARY_DIRS})
else()
  find_path(ZMQ_INCLUDE_DIR zmq.h)
  find_library(ZMQ_LIBRARY NAMES zmq libzmq)
  if(NOT ZMQ_INCLUDE_DIR OR NOT ZMQ_LIBRARY)
    message(FATAL_ERROR "libzmq not found (install libzmq3-dev / zeromq-devel, or set ZMQ_INCLUDE_DIR/ZMQ_LIBRARY)")
  endif()
  add_library(zmq_lib INTERFACE)
  target_include_directories(zmq_lib INTERFACE ${ZMQ_INCLUDE_DIR})
  target_link_libraries(zmq_lib INTERFACE ${ZMQ_LIBRARY})
endif()

find_package(Threads REQUIRED)

# The producer executable. One TU per concern (ADR-0012 P3 one-owner: one TU per topology impl, so no
# impl can be hand-copied into another and silently diverge):
#   main.cpp                 — the imperative shell (CLI ACL -> ProducerConfig -> run_producer -> report)
#   producer.cpp             — run_producer + the calibration spin + the two ProducerModes (decoupled/coupled)
#   boundary_per_thread.cpp  — Topology A (one DEALER per producer thread)
#   boundary_coalescing.cpp  — Topology B (producer threads -> one coalescing thread -> one DEALER)
#   boundary_factory.cpp     — make_boundary() dispatch on BoundaryTopology
# Header-only SSOTs (no source entry): wire.hpp (the codec), boundary.hpp / producer.hpp (the seams),
# zmq_dealer.hpp (the shared Layer-2 DEALER framing), zmq_context.hpp (the process-global ZMQ context).
add_executable(tlab-producer
  main.cpp
  producer.cpp
  boundary_per_thread.cpp
  boundary_coalescing.cpp
  boundary_factory.cpp
)
target_include_directories(tlab-producer PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(tlab-producer PRIVATE zmq_lib Threads::Threads)

# ================================================================================================
#  THE REAL-GENERATOR LEAF SOURCE (opt-in: -DTLAB_REAL_GENERATOR=ON) — the fork-in-the-road work.
#  Couples the lab to the PARENT chocofarm_core (env / gumbel / search_runtime / fiber primitives) so
#  the producer can drive REAL Gumbel-AZ searches as the load, routing each leaf through OUR
#  tlab::Boundary (NOT the parent's wire client — we keep our near-optimal static coupling and test the
#  real DYNAMIC leaf-production against it). Gated so the default synthetic build stays standalone /
#  clean-room (ADR-0012 compose: the real source is an ADDITIVE leaf-source plug, not a replacement).
#  chocofarm_core is the PREBUILT static lib in the parent cpp/build (rebuild it first if stale:
#  `cmake --build cpp/build --target chocofarm_core`); boost.context backs the fiber multiplexer.
# ================================================================================================
option(TLAB_REAL_GENERATOR "Build the real-generator (chocofarm search) leaf source" OFF)
if(TLAB_REAL_GENERATOR)
  get_filename_component(CHOCO_REPO_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../.. ABSOLUTE)
  set(CHOCO_CORE_INCLUDE ${CHOCO_REPO_ROOT}/cpp/include)
  find_library(CHOCO_CORE_LIB chocofarm_core PATHS ${CHOCO_REPO_ROOT}/cpp/build NO_DEFAULT_PATH)
  if(NOT CHOCO_CORE_LIB)
    message(FATAL_ERROR "TLAB_REAL_GENERATOR=ON but libchocofarm_core not found under "
                        "${CHOCO_REPO_ROOT}/cpp/build — build it: cmake --build cpp/build --target chocofarm_core")
  endif()
  # chocofarm_core links nlohmann_json + hiredis + zmq + threads transitively; resolve the ones the
  # lab does not already carry (hiredis for redis — linked though the lab never uses it; json header-only).
  find_library(HIREDIS_LIB hiredis)
  find_package(Boost QUIET COMPONENTS context)
  if(Boost_FOUND)
    set(TLAB_BOOST_CONTEXT Boost::context)
  else()
    find_library(BOOST_CONTEXT_LIBRARY boost_context REQUIRED)
    set(TLAB_BOOST_CONTEXT ${BOOST_CONTEXT_LIBRARY})
  endif()

  # The build-coupling + routing PROOF smoke: run one real SerialRuntime decision over a local DetNet
  # (no endpoint) OR route each leaf through tlab::Boundary to a live server (endpoint given). The
  # boundary impls + factory are linked in so make_boundary() resolves (the same TUs tlab-producer uses).
  add_executable(tlab-real-gen-smoke
    real_gen_smoke.cpp
    boundary_per_thread.cpp
    boundary_coalescing.cpp
    boundary_factory.cpp)
  target_include_directories(tlab-real-gen-smoke PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CHOCO_CORE_INCLUDE})
  target_link_libraries(tlab-real-gen-smoke PRIVATE
    ${CHOCO_CORE_LIB} ${TLAB_BOOST_CONTEXT} ${HIREDIS_LIB} zmq_lib Threads::Threads)

  # The NON-FIBER real-generator load driver (N threads x SerialRuntime, B=1 blocking round-trips ->
  # server batch ~= N). The baseline the fiber multiplexer is measured against.
  add_executable(tlab-real-producer
    real_producer.cpp
    boundary_per_thread.cpp
    boundary_coalescing.cpp
    boundary_factory.cpp)
  target_include_directories(tlab-real-producer PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CHOCO_CORE_INCLUDE})
  target_link_libraries(tlab-real-producer PRIVATE
    ${CHOCO_CORE_LIB} ${TLAB_BOOST_CONTEXT} ${HIREDIS_LIB} zmq_lib Threads::Threads)
endif()
