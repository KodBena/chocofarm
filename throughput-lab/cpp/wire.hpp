// throughput-lab/cpp/wire.hpp
// Purpose: the ONE authoritative byte-level spec of the producer<->server wire for this clean-room
//   testbed, AND the header-only codec the producer uses. It is a faithful copy of chocofarm's live
//   inference wire (chocofarm/az/wire_spec.py + inference_wire.py + cpp/include/chocofarm/wire_spec.hpp
//   + inference_wire.hpp), so this testbed's server is byte-for-byte comparable with the production
//   serving path (the whole point of the lab). The Python side (server/wire.py) derives the SAME
//   layout from the SAME constants spelled here — there is exactly one definition of every byte
//   (ADR-0012 P7: a cross-boundary fact has one home; every side derives, none re-authors).
// Public Domain (The Unlicense).
//
// ============================================================================================
//  THE WIRE — two distinct layers, kept apart (ADR-0012 P7: serialization ⊥ transport)
// ============================================================================================
//
//  LAYER 1 — THE VALUE FRAME (the serialization contract; what the codec below encodes/decodes)
//  -------------------------------------------------------------------------------------------
//  A length-prefixed little-endian float32 frame, fronted by a one-byte protocol-version header so
//  a codec mismatch fails LOUDLY (ADR-0002) rather than silently misreading floats. B leaves per
//  message (B=1 is the degenerate single-leaf case; the batched frame SUBSUMES single-leaf — there
//  is no dual-mode). All multi-byte fields LITTLE-ENDIAN.
//
//      Request  : [ver:u8][B:u32 LE][in_dim:u32 LE][X : f32 x (B*in_dim) LE]   (X row-major)
//      Response : [ver:u8][B:u32 LE][n_actions:u32 LE][ B x (value:f32 LE, logits:f32 x n_actions LE) ]
//
//   - ver       : PROTOCOL_VERSION (currently 2). An old/new mismatch is a loud decode error.
//   - B         : number of leaf rows in this message (>= 1).
//   - in_dim    : feature width per row (241 on the live Stage-A env; see server/lifted/forward).
//   - X         : B*in_dim float32, ROW-MAJOR (byte (r*in_dim + c)*4 is row r, column c).
//   - n_actions : policy action count for the WHOLE batch (one field — every row of one net shares
//                 it). n_actions == 0 => value-only (each prediction's logits block is empty).
//   - response  : B records, each [value:f32][logits:f32 x n_actions]. The value is DE-STANDARDIZED
//                 (v = v_std*y_std + y_mean) by the server; the logits are RAW (not softmaxed).
//
//   Fixed-header byte size = VERSION_BYTES + COUNT_BYTES + COUNT_BYTES = 1 + 4 + 4 = 9 (both
//   directions). Request body = B*in_dim*4 bytes. Response body = B*(1 + n_actions)*4 bytes.
//
//  LAYER 2 — THE ZMQ TRANSPORT ENVELOPE (DEALER producer <-> ROUTER server; the lab's matched wire)
//  -------------------------------------------------------------------------------------------
//  The producer holds a ZMQ_DEALER socket; the server binds a ZMQ_ROUTER. The producer sends the
//  value frame as a MULTIPART message led by an 8-byte correlation id:
//
//      producer DEALER sends :  [ corr-id : u64 (8 raw native-endian bytes) ] [ <Layer-1 request> ]
//                               (two ZMQ frames: corr-id first with ZMQ_SNDMORE, then the payload)
//
//  ZMQ's ROUTER prepends the producer's connection IDENTITY as a leading frame on receipt, so the
//  server's recv_multipart yields:
//
//      server ROUTER recv    :  [ identity ] [ corr-id ] [ <Layer-1 request> ]
//                               frames[0]    frames[1]    frames[-1]
//                               (the server treats frames[1:-1] as an OPAQUE envelope it echoes back
//                                verbatim — here that is exactly the single [corr-id] frame)
//
//  The server replies on the ROUTER addressed to that identity, echoing the envelope unchanged:
//
//      server ROUTER sends   :  [ identity ] [ corr-id ] [ <Layer-1 response> ]
//      producer DEALER recv  :  [ corr-id ] [ <Layer-1 response> ]
//                               (ZMQ strips the identity; the producer matches the reply to its
//                                outstanding request by the echoed corr-id)
//
//  The correlation id is a TRANSPORT concern (a u64 the producer stamps and the server round-trips
//  byte-for-byte, NEVER parsing it) — it stays OUT of the Layer-1 value codec. Its bytes are the
//  raw native-endian u64 (matching chocofarm's WireLeafPool, which memcpy's a uint64_t straight onto
//  the wire); the server never interprets them, so endianness is irrelevant to correctness.
//
//  WHY THIS EXACT SHAPE: it is byte-identical to chocofarm's production DEALER<->ROUTER path
//  (cpp/include/chocofarm/wire_leaf_pool.hpp submit_batch + inference_server.py _drain/_scatter), so
//  a throughput measured here is a throughput of the SAME serving wire.
//
// ============================================================================================

#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "chocofarm/quantity.hpp"  // the zero-cost phantom machinery (Band-1, reused across the boundary)

namespace tlab::wire {

// ---- the protocol-version header byte (the codec-mismatch tripwire) ----------------------------
// Mirrors chocofarm wire_spec.PROTOCOL_VERSION. v2 is the BATCHED frame (a B:u32 ahead of
// in_dim/n_actions). An old/new pairing fails loudly at the version byte instead of misreading the
// next field as a float (ADR-0002). Bump on ANY Layer-1 layout change.
inline constexpr std::uint8_t PROTOCOL_VERSION = 2;

// ---- fixed-field byte widths (these ARE the layout) --------------------------------------------
inline constexpr std::size_t VERSION_BYTES = 1;   // u8  — the protocol-version header byte
inline constexpr std::size_t COUNT_BYTES   = 4;   // u32 — a length prefix (B, in_dim, n_actions)
inline constexpr std::size_t FLOAT_BYTES   = 4;   // f32 — a payload float (and the response value)

// On-wire value types the widths name (typed reads, P9): a u8 version, a u32 count, an f32 element.
using version_t = std::uint8_t;
using count_t   = std::uint32_t;
using float_t   = float;          // IEEE-754 binary32, matching numpy '<f4'

// Fixed-header byte size = version + the TWO u32 counts (same for request and response).
inline constexpr std::size_t HEADER_BYTES = VERSION_BYTES + COUNT_BYTES + COUNT_BYTES;   // 9

// The Stage-A feature width on the live env (feat_dim = 5N + 3nD + 6 + n_tel = 241). The producer
// emits rows of this width; documented here so the synthetic load matches the real payload size.
// (This is a payload-size fact, NOT a wire-layout field — in_dim travels on the wire per message.)
inline constexpr count_t STAGE_A_IN_DIM = 241;

// The correlation-id field width: an 8-byte native-endian u64, the LEADING ZMQ frame the producer
// stamps and the server round-trips opaquely. A transport concern, not part of the Layer-1 codec.
using corr_t = std::uint64_t;
inline constexpr std::size_t CORR_BYTES = sizeof(corr_t);   // 8

static_assert(sizeof(version_t) == VERSION_BYTES, "version_t width must match VERSION_BYTES");
static_assert(sizeof(count_t)   == COUNT_BYTES,   "count_t width must match COUNT_BYTES");
static_assert(sizeof(float_t)   == FLOAT_BYTES,   "float_t width must match FLOAT_BYTES");
static_assert(sizeof(corr_t)    == CORR_BYTES,    "corr_t width must match CORR_BYTES");

// ============================================================================================
//  IN-MEMORY PHANTOM SPLIT (ADR-0000 / ADR-0012 P6) — reuses chocofarm's Quantity<Tag,Rep> machinery
// ============================================================================================
// The same three-roles-one-u32 fusion as chocofarm/wire_spec.hpp (B / in_dim / n_actions all ride
// count_t) plus the two-corr-namespaces fusion (producer-corr vs wire-corr both ride corr_t). The phantom
// split makes the in-memory mix a COMPILE error while keeping the on-wire bytes BIT-IDENTICAL: each
// phantom wraps the same count_t/corr_t and the codec .value()-unwraps at the exact byte read/write
// (the named wire ACL). The traits live in namespace chocofarm (the machinery's home), so they are
// specialized with explicit qualification on the tlab-local tags below.

// B — leaf ROWS per message (>= 1). A row count, NOT a width.
struct RowCountTag {};
using RowCount = chocofarm::Quantity<RowCountTag, count_t>;

// in_dim — feature WIDTH per row (241 on Stage-A). The B<->in_dim swap is the bug a distinct tag forbids.
struct FeatureDimTag {};
using FeatureDim = chocofarm::Quantity<FeatureDimTag, count_t>;

// n_actions — policy action count for the whole batch (0 ⇒ value-only; a meaningful typed zero, ADR-0002).
struct ActionCountTag {};
using ActionCount = chocofarm::Quantity<ActionCountTag, count_t>;

// CorrId — the OPAQUE 8-byte transport correlation token (memcpy'd raw native-endian; never parsed). Two
// namespaces share corr_t: the PRODUCER corr (per-thread, stamped by the generator) and the WIRE corr
// (stamped by the coalescing thread), split/joined in Topology B. Distinct tags make the coalescing
// scatter map's two id-spaces unmixable. NOT additive (an id), NOT a quantity — but the monotonic ++
// generation is the one affine op (next = corr + 1), so the increment crossing is named, not implicit.
struct ProducerCorrTag {};
using ProducerCorr = chocofarm::Quantity<ProducerCorrTag, corr_t>;
struct WireCorrTag {};
using WireCorr = chocofarm::Quantity<WireCorrTag, corr_t>;

// The protocol-version header byte as a CLOSED tag (an unknown version unrepresentable; the on-wire byte
// width stays version_t = u8, P6). The decode crosses the wire ACL with static_cast and fails loud
// (ADR-0002) on a byte outside this set.
enum class ProtocolVersionTag : version_t {
    SingleLeaf = 1,
    Batched = 2,
};
static_assert(static_cast<version_t>(ProtocolVersionTag::Batched) == PROTOCOL_VERSION,
              "ProtocolVersionTag::Batched must equal the SSOT PROTOCOL_VERSION byte (one home, P1).");

}  // namespace tlab::wire

namespace chocofarm {
// Opt the tlab wire count tags into additive + affine (a row count + a row count is a row count; a loop
// index over [0,B)). The corr tags are affine ONLY (monotonic ++ generation), never additive.
template <> struct quantity_additive<tlab::wire::RowCountTag> : std::true_type {};
template <> struct quantity_affine<tlab::wire::RowCountTag> : std::true_type {};
template <> struct quantity_additive<tlab::wire::FeatureDimTag> : std::true_type {};
template <> struct quantity_affine<tlab::wire::FeatureDimTag> : std::true_type {};
template <> struct quantity_additive<tlab::wire::ActionCountTag> : std::true_type {};
template <> struct quantity_affine<tlab::wire::ActionCountTag> : std::true_type {};
template <> struct quantity_affine<tlab::wire::ProducerCorrTag> : std::true_type {};
template <> struct quantity_affine<tlab::wire::WireCorrTag> : std::true_type {};
}  // namespace chocofarm

namespace tlab::wire {

// =============================================================================================
//  THE LAYER-1 CODEC (header-only, transport-free — no zmq include here). The build agent for the
//  producer USES encode_request; the round-trip helpers (decode) are provided for tests/parity.
//  These signatures mirror chocofarm/cpp/include/chocofarm/inference_wire.hpp byte-for-byte.
// =============================================================================================

// One decoded prediction: the de-standardized value + the raw policy logits (empty when value-only).
struct ResponseFields {
    float value = 0.0f;
    std::vector<float> logits;   // raw logits over n_actions slots (empty when n_actions == 0)
};

// A decoded BATCHED request: B feature rows of width in_dim, row-major in `flat`.
struct RequestFields {
    count_t B = 0;
    count_t in_dim = 0;
    std::vector<float> flat;     // B*in_dim floats, row-major (row r, col c at flat[r*in_dim + c])
};

// ---- little-endian field helpers (host is LE — wire_spec's standing assumption) ----------------
inline void put_count(std::vector<unsigned char>& out, count_t v) {
    unsigned char b[COUNT_BYTES];
    std::memcpy(b, &v, COUNT_BYTES);
    out.insert(out.end(), b, b + COUNT_BYTES);
}
inline void put_f32(std::vector<unsigned char>& out, float_t v) {
    unsigned char b[FLOAT_BYTES];
    std::memcpy(b, &v, FLOAT_BYTES);
    out.insert(out.end(), b, b + FLOAT_BYTES);
}
[[nodiscard]] inline count_t read_count(std::span<const unsigned char> bytes, std::size_t at) {
    count_t v = 0;
    std::memcpy(&v, bytes.data() + at, COUNT_BYTES);
    return v;
}
[[nodiscard]] inline float_t read_f32(std::span<const unsigned char> bytes, std::size_t at) {
    float_t v = 0.0f;
    std::memcpy(&v, bytes.data() + at, FLOAT_BYTES);
    return v;
}

// ---- request codec ------------------------------------------------------------------------------
// Encode a BATCHED feature matrix into a request frame [ver][B][in_dim][X:f32 x (B*in_dim)] (X
// row-major). `flat` is the B*in_dim contiguous rows; B/in_dim are passed explicitly and validated
// against the span length (never silently re-derived in a way that could disagree — P1). B=1 is the
// degenerate single-leaf case.
//
// NOTE FOR THE PRODUCER BUILD AGENT: this throws std::invalid_argument on a contract violation
// (an empty batch / ragged flat). That is the lab-side ergonomic choice for a self-contained
// testbed; chocofarm's production codec returns std::expected instead. The BYTES produced are
// identical either way — only the failure channel differs. Keep encode total on a well-typed input.
[[nodiscard]] inline std::vector<unsigned char> encode_request(
        std::span<const float> flat, count_t B, count_t in_dim) {
    if (B == 0)      throw std::invalid_argument("tlab wire: encode_request B is 0 (empty batch)");
    if (in_dim == 0) throw std::invalid_argument("tlab wire: encode_request in_dim is 0 (no features)");
    const std::size_t want = static_cast<std::size_t>(B) * in_dim;
    if (flat.size() != want)
        throw std::invalid_argument("tlab wire: encode_request flat size != B*in_dim");
    std::vector<unsigned char> out;
    out.reserve(HEADER_BYTES + flat.size() * FLOAT_BYTES);
    out.push_back(static_cast<unsigned char>(PROTOCOL_VERSION));
    put_count(out, B);
    put_count(out, in_dim);
    for (float v : flat) put_f32(out, v);
    return out;
}

// Decode a batched request frame back to (B, in_dim, flat). BOUNDARY validation (ADR-0002): an
// unknown protocol byte, a too-short frame, a B/in_dim of 0, or a wrong-length body throws.
[[nodiscard]] inline RequestFields decode_request(std::span<const unsigned char> frame) {
    if (frame.size() < HEADER_BYTES)
        throw std::invalid_argument("tlab wire: request frame too short for header");
    if (frame[0] != PROTOCOL_VERSION)
        throw std::invalid_argument("tlab wire: request protocol-byte mismatch");
    count_t B = read_count(frame, VERSION_BYTES);
    count_t in_dim = read_count(frame, VERSION_BYTES + COUNT_BYTES);
    if (B == 0)      throw std::invalid_argument("tlab wire: request B is 0");
    if (in_dim == 0) throw std::invalid_argument("tlab wire: request in_dim is 0");
    const std::size_t n = static_cast<std::size_t>(B) * in_dim;
    if (frame.size() != HEADER_BYTES + n * FLOAT_BYTES)
        throw std::invalid_argument("tlab wire: request body length != B*in_dim*f32");
    RequestFields r;
    r.B = B; r.in_dim = in_dim; r.flat.resize(n);
    for (std::size_t i = 0; i < n; ++i) r.flat[i] = read_f32(frame, HEADER_BYTES + i * FLOAT_BYTES);
    return r;
}

// ---- response codec (the producer DECODES replies; the server ENCODES them in Python) -----------
// Decode a batched response frame back to B ResponseFields. BOUNDARY validation (ADR-0002).
[[nodiscard]] inline std::vector<ResponseFields> decode_response(std::span<const unsigned char> frame) {
    if (frame.size() < HEADER_BYTES)
        throw std::invalid_argument("tlab wire: response frame too short for header");
    if (frame[0] != PROTOCOL_VERSION)
        throw std::invalid_argument("tlab wire: response protocol-byte mismatch");
    count_t B = read_count(frame, VERSION_BYTES);
    count_t n_actions = read_count(frame, VERSION_BYTES + COUNT_BYTES);
    if (B == 0) throw std::invalid_argument("tlab wire: response B is 0");
    const std::size_t per = static_cast<std::size_t>(1 + n_actions);   // value + logits per record
    if (frame.size() != HEADER_BYTES + static_cast<std::size_t>(B) * per * FLOAT_BYTES)
        throw std::invalid_argument("tlab wire: response body length != B*(1+n_actions)*f32");
    std::vector<ResponseFields> out(B);
    for (count_t r = 0; r < B; ++r) {
        std::size_t base = HEADER_BYTES + static_cast<std::size_t>(r) * per * FLOAT_BYTES;
        out[r].value = read_f32(frame, base);
        out[r].logits.resize(n_actions);
        for (count_t c = 0; c < n_actions; ++c)
            out[r].logits[c] = read_f32(frame, base + static_cast<std::size_t>(1 + c) * FLOAT_BYTES);
    }
    return out;
}

}  // namespace tlab::wire
