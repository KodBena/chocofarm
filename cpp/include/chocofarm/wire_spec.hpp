// cpp/include/chocofarm/wire_spec.hpp
// Purpose: the C++ MIRROR of the Shape B ZeroMQ inference wire frame's byte layout. The ONE
//   authoritative declaration of that layout is chocofarm/az/wire_spec.py (ADR-0012 P1/P7: a
//   cross-boundary fact has one home; every side DERIVES its view, none re-authors it). This header
//   declares the SAME constants so the deferred C++ ZmqNetClient (the docs/design/zmq-inference-
//   service.md §9 P9-pass impl) derives its codec from them — never from re-read prose. The values
//   here are DRIFT-CHECKED against the Python SSOT in the default Python suite
//   (tests/test_wire_drift.py parses these literals and asserts equality), so a one-sided layout
//   change (a PROTOCOL_VERSION bump, a wider count field, a byte-order change) reds the default suite
//   instead of silently misreading floats on the wire (ADR-0002 / ADR-0011 Rule 4).
//
//   ── DERIVED FROM chocofarm/az/wire_spec.py — DO NOT EDIT EITHER SIDE WITHOUT THE OTHER. ──
//   The drift test is the mechanical net that makes that instruction enforced, not advisory.
//
//   The frame (the §2 contract):
//       Request  : [ver:u8][in_dim   :u32 LE][X      : f32×in_dim   LE]
//       Response : [ver:u8][n_actions:u32 LE][value:f32 LE][logits : f32×n_actions LE]
//   n_actions == 0 ⇒ value-only (empty logits block). All multi-byte fields little-endian. Bump
//   PROTOCOL_VERSION on ANY layout change so an old pairing fails loudly at decode (unknown byte).
//
// Public Domain (The Unlicense).
#pragma once

#include <cstddef>
#include <cstdint>

namespace chocofarm::wire {

// The protocol-version header byte (mirrors wire_spec.PROTOCOL_VERSION). Bump on ANY layout change.
inline constexpr std::uint8_t PROTOCOL_VERSION = 1;

// Fixed-field byte widths (mirror wire_spec.VERSION_BYTES / COUNT_BYTES / FLOAT_BYTES). The version
// byte is u8; the length prefix (in_dim / n_actions) is u32; the payload/value floats are f32. These
// ARE the layout — the codec reads/writes exactly these many bytes per field, little-endian.
inline constexpr std::size_t VERSION_BYTES = 1;   // u8  — the protocol-version header byte
inline constexpr std::size_t COUNT_BYTES = 4;     // u32 — the length prefix (in_dim, n_actions)
inline constexpr std::size_t FLOAT_BYTES = 4;     // f32 — the payload float (and the response value)

// The on-wire integer/float types the widths above name, so the codec's reads are typed (P9): a u8
// version, a u32 length prefix, an f32 payload element. (Little-endian is asserted host-side; these
// are the value types the bytes decode to.)
using version_t = std::uint8_t;
using count_t = std::uint32_t;
using float_t = float;   // IEEE-754 binary32, matching numpy '<f4'

// Fixed-header byte size = the version byte + the u32 count (the same for request and response — both
// are [version][count]). Derived from the widths, never a separate literal.
inline constexpr std::size_t HEADER_BYTES = VERSION_BYTES + COUNT_BYTES;   // 5

static_assert(sizeof(version_t) == VERSION_BYTES, "wire version_t width must match VERSION_BYTES");
static_assert(sizeof(count_t) == COUNT_BYTES, "wire count_t width must match COUNT_BYTES");
static_assert(sizeof(float_t) == FLOAT_BYTES, "wire float_t width must match FLOAT_BYTES");

}  // namespace chocofarm::wire
