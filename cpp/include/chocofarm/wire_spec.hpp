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
//   The BATCHED frame (the §2 contract; B=1 is the degenerate single-leaf case, so the batched frame
//   SUBSUMES single-leaf — there is no dual-mode):
//       Request  : [ver:u8][B:u32 LE][in_dim:u32 LE][X : f32×(B·in_dim) LE]   (row-major)
//       Response : [ver:u8][B:u32 LE][n_actions:u32 LE][ B × (value:f32 LE, logits:f32×n_actions LE) ]
//   n_actions == 0 ⇒ value-only (every prediction's logits block empty). All multi-byte fields
//   little-endian. Bump PROTOCOL_VERSION on ANY layout change so an old pairing fails loudly at decode
//   (unknown byte) — the single-leaf → batched migration is exactly such a bump (v1 → v2).
//
// Public Domain (The Unlicense).
#pragma once

#include <cstddef>
#include <cstdint>

#include "chocofarm/quantity.hpp"  // Quantity<Tag, Rep> — the in-memory phantom split (NOT the on-wire bytes)

namespace chocofarm::wire {

// The protocol-version header byte (mirrors wire_spec.PROTOCOL_VERSION). Bump on ANY layout change.
// v2: the BATCHED frame (a B:u32 count ahead of in_dim/n_actions). v1 was the single-leaf frame.
inline constexpr std::uint8_t PROTOCOL_VERSION = 2;

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

// Fixed-header byte size = the version byte + the TWO u32 counts (the same for request and response —
// both are [version][B][in_dim|n_actions]). Derived from the widths, never a separate literal.
inline constexpr std::size_t HEADER_BYTES = VERSION_BYTES + COUNT_BYTES + COUNT_BYTES;   // 9

static_assert(sizeof(version_t) == VERSION_BYTES, "wire version_t width must match VERSION_BYTES");
static_assert(sizeof(count_t) == COUNT_BYTES, "wire count_t width must match COUNT_BYTES");
static_assert(sizeof(float_t) == FLOAT_BYTES, "wire float_t width must match FLOAT_BYTES");

// ============================================================================================
//  IN-MEMORY PHANTOM SPLIT of the shared count_t (ADR-0000 / ADR-0012 P6 BIT-IDENTITY GATE)
// ============================================================================================
// The wire carries ONE 4-byte u32 length-prefix type (count_t) in three semantically-DISTINCT roles: B
// (batch row count), in_dim (feature width), n_actions (policy action count). On the wire they are
// indistinguishable bytes (P6: the byte layout is the Python SSOT, NOT free to re-motivate here — a wider
// field is a PROTOCOL_VERSION bump). But IN MEMORY mixing B with in_dim or n_actions is a category error
// the wire layout itself cannot catch. The phantom split below makes the swap a COMPILE error while
// serializing BIT-IDENTICALLY: each phantom wraps the SAME count_t, and the codec does .value() at the
// exact byte read/write (the named wire ACL) so no on-wire byte changes. All three are additive (a row
// count + a row count is a row count) AND affine (a loop index over [0, B) / [0, in_dim) / [0, n_actions)).

// B — the number of leaf feature ROWS in one message (>= 1; B==1 is the degenerate single-leaf case).
struct RowCountTag {};
using RowCount = Quantity<RowCountTag, count_t>;

// in_dim — the feature width per row (241 on Stage-A). A WIDTH, not a row count: the classic B<->in_dim
// swap is the bug this distinct tag forbids. (Named WireFeatureDim to avoid colliding with the in-memory
// chocofarm::FeatureDim of domains.hpp — this is the on-wire u32 width, a different rep and home.)
struct WireFeatureDimTag {};
using WireFeatureDim = Quantity<WireFeatureDimTag, count_t>;

// n_actions — the policy action count for the whole response batch (0 ⇒ value-only, a MEANINGFUL typed
// zero, not a sentinel; ADR-0002). DISTINCT tag from B/in_dim though the same count_t carrier.
struct ActionCountTag {};
using ActionCount = Quantity<ActionCountTag, count_t>;

// The protocol-version header byte as a CLOSED tag (ADR-0000: an unknown version unrepresentable-by-
// construction at the type level; the on-wire byte width stays version_t = u8, P6). `using enum` lets the
// codec name `ProtocolVersion::Batched` without re-spelling the literal; the decode crosses the wire ACL
// with static_cast<version_t> and fails loud on a byte outside this set (ADR-0002).
enum class ProtocolVersion : version_t {
    SingleLeaf = 1,  // v1: the single-leaf frame (superseded)
    Batched = 2,     // v2: the batched frame (current; B:u32 ahead of in_dim/n_actions)
};
static_assert(static_cast<version_t>(ProtocolVersion::Batched) == PROTOCOL_VERSION,
              "ProtocolVersion::Batched must equal the SSOT PROTOCOL_VERSION byte (one home, P1).");

}  // namespace chocofarm::wire

namespace chocofarm {
// Opt the wire count tags into additive + affine (a row count + a row count is a row count; a loop index
// over [0,B) / [0,in_dim) / [0,n_actions)). The traits live in this (the machinery's) namespace; the tags
// are in the nested ::wire namespace, so they are qualified here.
template <> struct quantity_additive<wire::RowCountTag> : std::true_type {};
template <> struct quantity_affine<wire::RowCountTag> : std::true_type {};
template <> struct quantity_additive<wire::WireFeatureDimTag> : std::true_type {};
template <> struct quantity_affine<wire::WireFeatureDimTag> : std::true_type {};
template <> struct quantity_additive<wire::ActionCountTag> : std::true_type {};
template <> struct quantity_affine<wire::ActionCountTag> : std::true_type {};
}  // namespace chocofarm
