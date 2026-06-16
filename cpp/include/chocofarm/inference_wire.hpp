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
//   The frame (the docs/design/zmq-inference-service.md §2 contract):
//       Request  : [ver:u8][in_dim   :u32 LE][X      : f32×in_dim   LE]
//       Response : [ver:u8][n_actions:u32 LE][value:f32 LE][logits : f32×n_actions LE]
//   n_actions == 0 ⇒ value-only (empty logits block, mirroring forward_core's logits=None).
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

// One decoded response: the de-standardized value + the raw policy logits (empty when value-only,
// mirroring the Python (value, logits=None) and the NetPrediction shape net.hpp defines).
struct ResponseFields {
    float value = 0.0f;
    std::vector<float> logits;   // raw logits over n_actions slots (empty when n_actions == 0)
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
// Encode one feature vector X into a request frame [ver][in_dim][X:f32]. in_dim is DERIVED from the
// vector — never a separate argument that could disagree with the payload (P1). Total: a well-formed
// span always yields a frame (the caller is responsible for non-empty / finite, matching the C++
// NetForward's own predict contract — finiteness is the server's boundary, exactly as the Python
// client lets the server reject non-finite; encode here mirrors the byte layout only).
[[nodiscard]] inline std::vector<unsigned char> encode_request(std::span<const float> X) {
    std::vector<unsigned char> out;
    out.reserve(HEADER_BYTES + X.size() * FLOAT_BYTES);
    out.push_back(static_cast<unsigned char>(PROTOCOL_VERSION));
    put_count(out, static_cast<count_t>(X.size()));
    for (float v : X) put_f32(out, v);
    return out;
}

// Decode a request frame back to the feature vector X (float32, length in_dim). BOUNDARY validation
// (ADR-0002): an unknown protocol byte, a frame too short for its header, an in_dim of 0, or a payload
// whose byte count is not exactly `in_dim × f32` is a typed Error — never a zero-filled/truncated row.
[[nodiscard]] inline std::expected<std::vector<float>, Error> decode_request(
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
    count_t in_dim = read_count(frame, VERSION_BYTES);
    if (in_dim == 0)
        return std::unexpected(make_error("chocofarm wire: request in_dim is 0 (no feature vector)"));
    std::size_t want = HEADER_BYTES + static_cast<std::size_t>(in_dim) * FLOAT_BYTES;
    if (frame.size() != want)
        return std::unexpected(make_error("chocofarm wire: request payload is " +
                                          std::to_string(frame.size()) + " bytes, expected " +
                                          std::to_string(want) + " (= in_dim " + std::to_string(in_dim) +
                                          " × f32 + header)"));
    std::vector<float> X(in_dim);
    for (count_t i = 0; i < in_dim; ++i)
        X[i] = read_f32(frame, HEADER_BYTES + static_cast<std::size_t>(i) * FLOAT_BYTES);
    return X;
}

// ---- response codec ----
// Encode a NetPrediction into a response frame [ver][n_actions][value][logits:f32]. An EMPTY logits
// span ⇒ n_actions=0 (value-only, mirroring forward_core's logits=None). value is the de-standardized
// scalar; the logits are raw (NOT softmaxed) — masking is client-side (§2). Total.
[[nodiscard]] inline std::vector<unsigned char> encode_response(float value,
                                                                std::span<const float> logits) {
    std::vector<unsigned char> out;
    out.reserve(HEADER_BYTES + FLOAT_BYTES + logits.size() * FLOAT_BYTES);
    out.push_back(static_cast<unsigned char>(PROTOCOL_VERSION));
    put_count(out, static_cast<count_t>(logits.size()));
    put_f32(out, value);
    for (float v : logits) put_f32(out, v);
    return out;
}

// Decode a response frame back to (value, logits). BOUNDARY validation (ADR-0002): an unknown protocol
// byte, a frame too short for the header+value, or a logits block whose byte count is not exactly
// `n_actions × f32` is a typed Error. n_actions==0 ⇒ empty logits (value-only).
[[nodiscard]] inline std::expected<ResponseFields, Error> decode_response(
    std::span<const unsigned char> frame) {
    std::size_t fixed = HEADER_BYTES + FLOAT_BYTES;   // header + the value scalar
    if (frame.size() < fixed)
        return std::unexpected(make_error("chocofarm wire: response frame too short (" +
                                          std::to_string(frame.size()) + " bytes) for its " +
                                          std::to_string(fixed) + "-byte header+value"));
    version_t ver = frame[0];
    if (ver != PROTOCOL_VERSION)
        return std::unexpected(make_error("chocofarm wire: response protocol byte " +
                                          std::to_string(ver) + " != supported " +
                                          std::to_string(PROTOCOL_VERSION) + " (codec mismatch)"));
    count_t n_actions = read_count(frame, VERSION_BYTES);
    std::size_t want = fixed + static_cast<std::size_t>(n_actions) * FLOAT_BYTES;
    if (frame.size() != want)
        return std::unexpected(make_error("chocofarm wire: response logits block makes the frame " +
                                          std::to_string(frame.size()) + " bytes, expected " +
                                          std::to_string(want) + " (= n_actions " +
                                          std::to_string(n_actions) + " × f32 + header+value)"));
    ResponseFields r;
    r.value = read_f32(frame, HEADER_BYTES);
    r.logits.resize(n_actions);
    for (count_t i = 0; i < n_actions; ++i)
        r.logits[i] = read_f32(frame, fixed + static_cast<std::size_t>(i) * FLOAT_BYTES);
    return r;
}

}  // namespace chocofarm::wire
