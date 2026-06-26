# throughput-lab/fused_jax/CMakeLists.txt
# Purpose: build the C++ side of the fused-JAX BatchPredict COMPONENT (lever #1,
#   docs/notes/batchpredict-throughput-design-2026-06-26.md). Three targets:
#     belief-wire-test        — the standalone codec unit test (no env, no link to core).
#     belief-batch-encode     — the SEND side (reads env, writes setup/request belief-wire bytes +
#                               the parity oracle JSON). Links chocofarm_core + nlohmann_json.
#     belief-response-decode  — the RECEIVE side (decodes the JAX-produced response frame). Standalone.
#
#   STANDALONE configure (codec test + response decode only; no env needed):
#     cmake -S throughput-lab/fused_jax -B throughput-lab/fused_jax/build -DCMAKE_BUILD_TYPE=Release
#   WITH the env send side (needs the prebuilt parent chocofarm_core static lib):
#     cmake --build cpp/build --target chocofarm_core      # build the core first if stale
#     cmake -S throughput-lab/fused_jax -B throughput-lab/fused_jax/build \
#         -DCMAKE_BUILD_TYPE=Release -DFUSED_WITH_ENV=ON
#
#   This is an ADDITIVE component build: it does NOT touch throughput-lab/cpp (the producer) or its
#   build/sched_wrap. ADR-0012 P3 one-owner — one TU per concern.
# Public Domain (The Unlicense).

cmake_minimum_required(VERSION 3.16)
project(tlab_fused_jax CXX)

set(CMAKE_CXX_STANDARD 23)           # std::span / std::expected, matching the parent toolchain (P9).
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "build type" FORCE)
endif()
add_compile_options(-Wall -Wextra)

# The parent chocofarm cpp/include carries the Band-1 phantom machinery (chocofarm/quantity.hpp) the
# belief wire reuses to phantom-split its B/kW64 count domains (header-only reuse, no link). Same as
# throughput-lab/cpp/CMakeLists.txt's quantity-only include.
get_filename_component(CHOCO_CORE_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/../../cpp/include ABSOLUTE)

# ---- the codec unit test (no env, no link) -----------------------------------------------------
add_executable(belief-wire-test belief_wire_test.cpp)
target_include_directories(belief-wire-test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CHOCO_CORE_INCLUDE})

# ---- the response-decode receive side (standalone codec consumer) -------------------------------
add_executable(belief-response-decode belief_response_decode.cpp)
target_include_directories(belief-response-decode PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CHOCO_CORE_INCLUDE})

# ================================================================================================
#  THE ENV SEND SIDE (opt-in: -DFUSED_WITH_ENV=ON) — couples to the PREBUILT parent chocofarm_core for
#  env + belief_features (the parity oracle). Gated so the codec test builds with no parent dependency.
# ================================================================================================
option(FUSED_WITH_ENV "Build belief-batch-encode (needs the prebuilt parent chocofarm_core)" OFF)
if(FUSED_WITH_ENV)
  get_filename_component(CHOCO_REPO_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../.. ABSOLUTE)
  find_library(CHOCO_CORE_LIB chocofarm_core PATHS ${CHOCO_REPO_ROOT}/cpp/build NO_DEFAULT_PATH)
  if(NOT CHOCO_CORE_LIB)
    message(FATAL_ERROR "FUSED_WITH_ENV=ON but libchocofarm_core not found under "
                        "${CHOCO_REPO_ROOT}/cpp/build — build it: cmake --build cpp/build --target chocofarm_core")
  endif()
  # chocofarm_core links nlohmann_json + hiredis + zmq + threads transitively; the export needs the
  # json header (the oracle blob) + hiredis/zmq to satisfy the static lib's transitive symbols.
  find_package(nlohmann_json QUIET)
  find_library(HIREDIS_LIB hiredis)
  find_package(PkgConfig QUIET)
  if(PkgConfig_FOUND)
    pkg_check_modules(ZMQ QUIET libzmq)
  endif()

  add_executable(belief-batch-encode belief_batch_encode.cpp)
  target_include_directories(belief-batch-encode PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CHOCO_CORE_INCLUDE})
  if(nlohmann_json_FOUND)
    target_link_libraries(belief-batch-encode PRIVATE nlohmann_json::nlohmann_json)
  else()
    # fall back to the core build's vendored nlohmann include (FetchContent location).
    target_include_directories(belief-batch-encode PRIVATE
      ${CHOCO_REPO_ROOT}/cpp/build/_deps/nlohmann_json-src/include)
  endif()
  target_link_libraries(belief-batch-encode PRIVATE ${CHOCO_CORE_LIB})
  if(HIREDIS_LIB)
    target_link_libraries(belief-batch-encode PRIVATE ${HIREDIS_LIB})
  endif()
  if(ZMQ_FOUND)
    target_link_libraries(belief-batch-encode PRIVATE ${ZMQ_LIBRARIES})
    target_link_directories(belief-batch-encode PRIVATE ${ZMQ_LIBRARY_DIRS})
  endif()
endif()
