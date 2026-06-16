// cpp/include/chocofarm/result_spec.hpp
// Purpose: the C++ MIRROR of the redis RESULT blob's byte format (the worker→parent training-record
//   transport). The ONE authoritative declaration of that format is chocofarm/az/result_spec.py
//   (ADR-0012 P1/P7: a cross-boundary fact has one home; every side DERIVES its view). This header
//   declares the SAME block order + dtype + ranks so the C++ write_results (transport.cpp) derives
//   them — never re-authors "the result blocks are float32, in order X/PI/M/Y". The values here are
//   DRIFT-CHECKED against the Python SSOT in the default Python suite (tests/test_wire_drift.py parses
//   these literals and asserts equality), so a one-sided change (a fifth block, a reorder, a float64
//   widening) reds the default suite instead of silently corrupting a reshape (ADR-0002 / R4).
//
//   ── DERIVED FROM chocofarm/az/result_spec.py — DO NOT EDIT EITHER SIDE WITHOUT THE OTHER. ──
//
//   The format (one task `idx`, under keys az:res:<token>:<idx>:{X,PI,M,Y}):
//       X  : (n, feat_dim)  float32   — feature rows
//       PI : (n, n_slots)   float32   — policy targets
//       M  : (n, n_slots)   float32   — legality mask
//       Y  : (n,)           float32   — scalar λ-return targets
//   Each block is contiguous, little-endian, row-major float32; np.frombuffer(...).reshape(...)
//   decodes it byte-for-byte (no second encoder — ADR-0012 P7).
//
// Public Domain (The Unlicense).
#pragma once

#include <array>
#include <cstddef>
#include <string_view>

namespace chocofarm::result {

// The per-block element type + its byte size (mirror result_spec.RESULT_DTYPE / RESULT_ITEMSIZE).
// All four blocks are float32; write_results spans `const float`, so this IS the dtype both sides
// commit to. A widening (float64) is a one-line edit here the drift test reconciles with the Python
// SSOT's '<f4'.
using block_t = float;                                   // IEEE-754 binary32, matching numpy '<f4'
inline constexpr std::size_t BLOCK_ITEMSIZE = 4;         // bytes per float32 (== sizeof(block_t))
inline constexpr std::string_view BLOCK_DTYPE_STR = "<f4";   // numpy dtype string (little-endian f32)

// The canonical block ORDER + ranks (mirror result_spec.BLOCK_ORDER / BLOCK_RANK). The key suffixes
// X/PI/M/Y in this order; X/PI/M are 2-D (n, dim), Y is 1-D (n,). transport.cpp::write_results emits
// the four spans in exactly this order; this header names the order so the C++ side does not hardcode
// a private sequence.
inline constexpr std::array<std::string_view, 4> BLOCK_ORDER = {"X", "PI", "M", "Y"};
inline constexpr std::array<int, 4> BLOCK_RANK = {2, 2, 2, 1};   // X, PI, M 2-D; Y 1-D — the reshape rule

static_assert(sizeof(block_t) == BLOCK_ITEMSIZE, "result block_t width must match BLOCK_ITEMSIZE");

}  // namespace chocofarm::result
