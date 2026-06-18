// cpp/include/chocofarm/inference_wire.hpp
// Purpose: the C++ wire CODEC for the Shape B batched ZeroMQ inference frame — the C++ twin of
//   chocofarm/az/inference_wire.py. It is header-only and DERIVES every byte width / version / order
//   from the SSOT mirror chocofarm/wire_spec.hpp (ADR-0012 P1/P7: a cross-boundary fact has one home;
//   every side derives its view, none re-authors the `[ver][count][f32…]` layout). The Python codec
//   derives the SAME layout from chocofarm/az/wire_spec.py, and the two specs are drift-checked in the
//   default Python suite (tests/test_wire_drift.py) — so the codecs cannot silently diverge.
//
//   This header has NO transport dependency (no zmq, no hiredis): it is pure byte-array encode/decode
//   over std::span / std::vector. The two consumers are:
//     * the ZmqNetClient (cpp/src/zmq_net_client.cpp) — encodes a request, decodes the reply;
//     * the standalone golden round-trip (cpp/parity/wire_golden.cpp) — so the #23 drift net's opt-in
//       C++ leg exercises THIS codec (it compiles with a bare `g++ -std=c++23` over only the mirror
//       headers + this one — still dependency-free).
//
//   The BATCHED frame (the docs/design/zmq-inference-service.md §2 contract; B=1 is the degenerate
//   single-leaf case, so the batched frame SUBSUMES single-leaf — there is no dual-mode):
//       Request  : [ver:u8][B:u32 LE][in_dim:u32 LE][X : f32×(B·in_dim) LE]   (row-major)
//       Response : [ver:u8][B:u32 LE][n_actions:u32 LE][ B × (value:f32 LE, logits:f32×n_actions LE) ]
//   n_actions == 0 ⇒ value-only (every prediction's logits block empty, mirroring forward_core's
//   logits=None).
//
//   ADR-0012 P9 / ADR-0002 (translate-and-validate, never coerce): ENCODE is total (a well-typed input
//   always produces a frame) and returns the bytes by value. DECODE is a BOUNDARY — an unknown protocol
//   byte, a frame too short for its header, or a length-prefix that disagrees with the byte count is a
//   typed [[nodiscard]] std::expected<…, Error>, NEVER a zero-filled or truncated forward. Little-endian
//   is the host standing assumption (wire_spec); the bytes are read/written native-LE.
//
// Public Domain (The Unlicense).
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <expected>
#include <span>
#include <string>
#include <vector>

#include "chocofarm/error.hpp"
#include "chocofarm/wire_spec.hpp"

namespace chocofarm::wire {

// One decoded prediction: the de-standardized value + the raw policy logits (empty when value-only,
// mirroring the Python (value, logits=None) and the NetPrediction shape net.hpp defines).
struct ResponseFields {
    float value = 0.0f;
    std::vector<float> logits;   // raw logits over n_actions slots (empty when n_actions == 0)
};

// A decoded BATCHED request: B feature rows of width in_dim, row-major in `flat` (so row r, column c is
// flat[r*in_dim + c]). The server-side decode; the driver only ENCODES requests.
struct RequestFields {
    count_t B = 0;
    count_t in_dim = 0;
    std::vector<float> flat;     // B·in_dim floats, row-major
};

// ---- little-endian field helpers (host is LE — wire_spec's standing assumption) ----
// Append a u32 count to `out` (little-endian). memcpy, not a reinterpret store, so it is well-defined.
inline void put_count(std::vector<unsigned char>& out, count_t v) {
    unsigned char b[COUNT_BYTES];
    std::memcpy(b, &v, COUNT_BYTES);
    out.insert(out.end(), b, b + COUNT_BYTES);
}

// Append one f32 to `out` (little-endian) by its raw bytes (the codec carries f32 bytes verbatim).
inline void put_f32(std::vector<unsigned char>& out, float_t v) {
    unsigned char b[FLOAT_BYTES];
    std::memcpy(b, &v, FLOAT_BYTES);
    out.insert(out.end(), b, b + FLOAT_BYTES);
}

// Read a u32 count at byte offset `at` (caller has bounds-checked `bytes`).
[[nodiscard]] inline count_t read_count(std::span<const unsigned char> bytes, std::size_t at) {
    count_t v = 0;
    std::memcpy(&v, bytes.data() + at, COUNT_BYTES);
    return v;
}

// Read one f32 at byte offset `at` (caller has bounds-checked `bytes`).
[[nodiscard]] inline float_t read_f32(std::span<const unsigned char> bytes, std::size_t at) {
    float_t v = 0.0f;
    std::memcpy(&v, bytes.data() + at, FLOAT_BYTES);
    return v;
}

// ---- request codec ----
// Encode a BATCHED feature matrix into a request frame [ver][B][in_dim][X:f32×(B·in_dim)] (X row-major:
// row r, column c at flat[r*in_dim + c]). `flat` is the B·in_dim contiguous rows; `B`/`in_dim` are
// passed explicitly (the matrix shape the driver gathers) and validated against the span length —
// never silently re-derived in a way that could disagree with the payload (P1). A flat.size() that is
// not exactly B·in_dim is the typed Error arm (ADR-0002 — never a ragged encode). B=1 is the
// degenerate single-leaf case. Finiteness is the server's boundary (the C++ NetForward / driver feeds
// real finite features), exactly as the Python client lets the server reject non-finite; encode here
// mirrors the byte layout only.
[[nodiscard]] inline std::expected<std::vector<unsigned char>, Error> encode_request(
    std::span<const float> flat, count_t B, count_t in_dim) {
    if (B == 0)
        return std::unexpected(make_error("chocofarm wire: encode_request B is 0 (empty batch)"));
    if (in_dim == 0)
        return std::unexpected(make_error("chocofarm wire: encode_request in_dim is 0 (no features)"));
    std::size_t want = static_cast<std::size_t>(B) * in_dim;
    if (flat.size() != want)
        return std::unexpected(make_error("chocofarm wire: encode_request flat has " +
                                          std::to_string(flat.size()) + " floats, expected " +
                                          std::to_string(want) + " (= B " + std::to_string(B) +
                                          " × in_dim " + std::to_string(in_dim) + ")"));
    std::vector<unsigned char> out;
    out.reserve(HEADER_BYTES + flat.size() * FLOAT_BYTES);
    out.push_back(static_cast<unsigned char>(PROTOCOL_VERSION));
    put_count(out, B);
    put_count(out, in_dim);
    for (float v : flat) put_f32(out, v);
    return out;
}

// Decode a BATCHED request frame back to its (B, in_dim, flat) fields. BOUNDARY validation (ADR-0002):
// an unknown protocol byte, a frame too short for its header, a B or in_dim of 0, or a payload whose
// byte count is not exactly `B·in_dim × f32` is a typed Error — never a zero-filled/truncated matrix.
[[nodiscard]] inline std::expected<RequestFields, Error> decode_request(
    std::span<const unsigned char> frame) {
    if (frame.size() < HEADER_BYTES)
        return std::unexpected(make_error("chocofarm wire: request frame too short (" +
                                          std::to_string(frame.size()) + " bytes) for its " +
                                          std::to_string(HEADER_BYTES) + "-byte header"));
    version_t ver = frame[0];
    if (ver != PROTOCOL_VERSION)
        return std::unexpected(make_error("chocofarm wire: request protocol byte " +
                                          std::to_string(ver) + " != supported " +
                                          std::to_string(PROTOCOL_VERSION) + " (codec mismatch)"));
    count_t B = read_count(frame, VERSION_BYTES);
    count_t in_dim = read_count(frame, VERSION_BYTES + COUNT_BYTES);
    if (B == 0)
        return std::unexpected(make_error("chocofarm wire: request B is 0 (empty batch)"));
    if (in_dim == 0)
        return std::unexpected(make_error("chocofarm wire: request in_dim is 0 (no feature vector)"));
    std::size_t n = static_cast<std::size_t>(B) * in_dim;
    std::size_t want = HEADER_BYTES + n * FLOAT_BYTES;
    if (frame.size() != want)
        return std::unexpected(make_error("chocofarm wire: request payload is " +
                                          std::to_string(frame.size()) + " bytes, expected " +
                                          std::to_string(want) + " (= B " + std::to_string(B) +
                                          " × in_dim " + std::to_string(in_dim) + " × f32 + header)"));
    RequestFields r;
    r.B = B;
    r.in_dim = in_dim;
    r.flat.resize(n);
    for (std::size_t i = 0; i < n; ++i)
        r.flat[i] = read_f32(frame, HEADER_BYTES + i * FLOAT_BYTES);
    return r;
}

// ---- response codec ----
// Encode B predictions into a batched response frame [ver][B][n_actions][ B × (value, logits:f32) ].
// `values` is the length-B de-standardized scalars; `logits_flat` is the B·n_actions raw (NOT
// softmaxed) logits row-major (empty ⇒ n_actions=0, value-only, mirroring forward_core's logits=None).
// `n_actions` is one field for the whole batch (every row of one net has the same action count); it is
// validated against logits_flat.size()==B·n_actions (ADR-0002 — never a ragged scatter). Masking is
// client-side (§2). The server is the only producer of responses; the C++ side (wire_golden) re-encodes
// what it decoded, so this completes the round-trip.
[[nodiscard]] inline std::expected<std::vector<unsigned char>, Error> encode_response(
    std::span<const float> values, std::span<const float> logits_flat, count_t n_actions) {
    count_t B = static_cast<count_t>(values.size());
    if (B == 0)
        return std::unexpected(make_error("chocofarm wire: encode_response B is 0 (no predictions)"));
    std::size_t want = static_cast<std::size_t>(B) * n_actions;
    if (logits_flat.size() != want)
        return std::unexpected(make_error("chocofarm wire: encode_response logits_flat has " +
                                          std::to_string(logits_flat.size()) + " floats, expected " +
                                          std::to_string(want) + " (= B " + std::to_string(B) +
                                          " × n_actions " + std::to_string(n_actions) + ")"));
    std::vector<unsigned char> out;
    out.reserve(HEADER_BYTES + static_cast<std::size_t>(B) * (1 + n_actions) * FLOAT_BYTES);
    out.push_back(static_cast<unsigned char>(PROTOCOL_VERSION));
    put_count(out, B);
    put_count(out, n_actions);
    for (count_t r = 0; r < B; ++r) {
        put_f32(out, values[r]);
        for (count_t c = 0; c < n_actions; ++c)
            put_f32(out, logits_flat[static_cast<std::size_t>(r) * n_actions + c]);
    }
    return out;
}

// Decode a batched response frame back to B ResponseFields (each value + its n_actions raw logits).
// BOUNDARY validation (ADR-0002): an unknown protocol byte, a B of 0, a frame too short for the header,
// or a body whose byte count is not exactly `B·(1 + n_actions) × f32` is a typed Error. n_actions==0 ⇒
// every prediction has empty logits (value-only).
[[nodiscard]] inline std::expected<std::vector<ResponseFields>, Error> decode_response(
    std::span<const unsigned char> frame) {
    if (frame.size() < HEADER_BYTES)
        return std::unexpected(make_error("chocofarm wire: response frame too short (" +
                                          std::to_string(frame.size()) + " bytes) for its " +
                                          std::to_string(HEADER_BYTES) + "-byte header"));
    version_t ver = frame[0];
    if (ver != PROTOCOL_VERSION)
        return std::unexpected(make_error("chocofarm wire: response protocol byte " +
                                          std::to_string(ver) + " != supported " +
                                          std::to_string(PROTOCOL_VERSION) + " (codec mismatch)"));
    count_t B = read_count(frame, VERSION_BYTES);
    count_t n_actions = read_count(frame, VERSION_BYTES + COUNT_BYTES);
    if (B == 0)
        return std::unexpected(make_error("chocofarm wire: response B is 0 (no predictions)"));
    std::size_t per = static_cast<std::size_t>(1 + n_actions);   // floats per prediction (value + logits)
    std::size_t want = HEADER_BYTES + static_cast<std::size_t>(B) * per * FLOAT_BYTES;
    if (frame.size() != want)
        return std::unexpected(make_error("chocofarm wire: response body makes the frame " +
                                          std::to_string(frame.size()) + " bytes, expected " +
                                          std::to_string(want) + " (= B " + std::to_string(B) +
                                          " × (value + n_actions " + std::to_string(n_actions) +
                                          ") × f32 + header)"));
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

}  // namespace chocofarm::wire
