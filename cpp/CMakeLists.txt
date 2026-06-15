# cpp/CMakeLists.txt
# Purpose: build the chocofarm C++ search/sim runner (the dumb-random seam-proof MVP, ADR-0012's
#   C++ section / scaling-and-cpp-seam.md Shape A). C++20. Dependencies are self-contained:
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

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

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

# ---- the runner core (a library so tests/tools can link it; main is a thin shell) ----
add_library(chocofarm_core
  src/instance.cpp
  src/env.cpp
  src/features.cpp
  src/transport.cpp
  src/net.cpp
  src/runner.cpp
)
target_include_directories(chocofarm_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(chocofarm_core PUBLIC nlohmann_json::nlohmann_json ${HIREDIS_LINK})
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
