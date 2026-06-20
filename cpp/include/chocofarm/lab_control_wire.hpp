// cpp/include/chocofarm/lab_control_wire.hpp
// Purpose: the ONE authoritative byte layout for the issue-gate CONTROL LAB's PER-FORWARD on-wire
//   decision transport (the Batch-0 harness, cpp/stage_a/control_lab/). It is the C++ twin and the
//   AUTHORITATIVE home of the lab control frame; the Python side derives the same layout in
//   cpp/stage_a/control_lab/lab_wire.py (a magic + length runtime parity check is the floor —
//   ADR-0012 P1/P7: a cross-boundary fact has ONE home, every side derives its view, none re-authors it).
//
//   WHAT IT IS (and is NOT). This is NOT a change to the VALUE codec (inference_wire.hpp/.py stays
//   BYTE-UNCHANGED) and NOT a change to the corr-id transport envelope frame. The decision epoch is the
//   eval server's FORWARD, synchronous: a producer thread rides its own feature snapshot ALONGSIDE its
//   batched leaf request, and the server rides that thread's next issue-gate bit back ALONGSIDE the
//   reply. The ride is a SECOND transport-envelope frame (the corr-id is the first), so the wire frame is
//       [corr-id u64][LAB-CONTROL?][value-payload]
//   The server round-trips the envelope (frames[1:-1]) opaquely on the NON-lab path; the lab server
//   variant PARSES frame[1] (the FEATURE frame) and REWRITES it as the GATE frame on the reply. When the
//   lab is OFF no LAB-CONTROL frame is attached — the envelope is `[corr-id]` and every byte on the wire
//   is identical to the production bench (ADR-0004 minimal-touch / lab-gated; P7 additive+versioned).
//
//   This SUPERSEDES the async issue_control_bridge.hpp FOR THE LAB (the bridge is a slow out-of-band ZMQ
//   REQ/REP loop on its own cadence; the lab's decision epoch is the forward itself). The bridge stays for
//   its own callers — this header does not touch it. The ACTUATION reuses the EXISTING IssueController hub:
//   the producer writes the gate bit through set_allow(tid, bit) and refill() reads may_issue(tid) on the
//   hot path exactly as before — ONE actuation cell, no second path (P3 one-owner).
//
//   ADR-0012 P9 / ADR-0002 (translate-and-validate, never coerce): ENCODE is total and returns bytes by
//   value; DECODE is a BOUNDARY returning a typed [[nodiscard]] std::expected — a bad magic, a wrong
//   version byte, or a length that disagrees with the field count is a typed Error, never a silently
//   misread snapshot. Little-endian is the host standing assumption (x86_64), asserted by the magic check.
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

namespace chocofarm::lab {

// The protocol version of the LAB control frame (independent of the value codec's PROTOCOL_VERSION —
// this frame is a separate transport-envelope contract). Bump on ANY layout change so an old pairing
// fails loudly at the version byte (ADR-0002) instead of misreading a field.
inline constexpr std::uint8_t LAB_WIRE_VERSION = 1;

// The two frame magics (a 32-bit tag fronting each frame so a stray/legacy frame is rejected, not
// misread — the P7 runtime parity floor). Distinct per direction so a request frame can never be
// mistaken for a reply frame.
inline constexpr std::uint32_t LAB_FEAT_MAGIC = 0x1AB0F0A1u;   // FEATURE frame (producer -> server)
inline constexpr std::uint32_t LAB_GATE_MAGIC = 0x1AB0F0A2u;   // GATE frame    (server -> producer)

// One producer thread's per-forward feature snapshot — the decision-epoch observation the server's
// Controller consumes for THIS thread. tid is the thread id (the gate routes back to it). The five
// counters mirror the IssueController metrics surface (inflight/ready cumulative-msgs/cumulative-leaves/
// recent-rtt) so the lab Observation is the same feature vocabulary the async path marshalled.
struct LabFeature {
    std::int32_t tid = 0;
    std::int32_t inflight = 0;   // outstanding (submitted, unanswered) messages this thread holds
    std::int32_t ready = 0;      // ready (parked-at-leaf, unsubmitted) slots this thread holds
    std::int64_t msgs = 0;       // cumulative messages this thread has issued
    std::int64_t leaves = 0;     // cumulative leaves this thread has sent
    std::int64_t rtt_us = 0;     // recent mean reply RTT for this thread (microseconds; 0 until measured)
    std::int64_t decisions = 0;  // cumulative recorded DECISIONS (completed Gumbel searches) this thread —
                                 // the SCORING numerator: the server sums served threads' decisions so the
                                 // harness can delta true dps over a wall window (the warm pool persists, so
                                 // the producer is one continuous process; this rides the throughput count).
};

// FEATURE frame byte size = magic(4) + ver(1) + tid(4) + inflight(4) + ready(4) + msgs(8) + leaves(8)
// + rtt_us(8) + decisions(8) — derived from the field widths, never a separate literal (P1).
inline constexpr std::size_t LAB_FEAT_BYTES = 4 + 1 + 4 + 4 + 4 + 8 + 8 + 8 + 8;   // 49
// GATE frame byte size = magic(4) + ver(1) + tid(4) + allow(1).
inline constexpr std::size_t LAB_GATE_BYTES = 4 + 1 + 4 + 1;                   // 10

// ---- little-endian field helpers (host is x86_64 LE; memcpy, never a reinterpret store) ----
inline void put_u32(std::vector<unsigned char>& out, std::uint32_t v) {
    unsigned char b[4];
    std::memcpy(b, &v, 4);
    out.insert(out.end(), b, b + 4);
}
inline void put_i32(std::vector<unsigned char>& out, std::int32_t v) {
    unsigned char b[4];
    std::memcpy(b, &v, 4);
    out.insert(out.end(), b, b + 4);
}
inline void put_i64(std::vector<unsigned char>& out, std::int64_t v) {
    unsigned char b[8];
    std::memcpy(b, &v, 8);
    out.insert(out.end(), b, b + 8);
}
[[nodiscard]] inline std::uint32_t read_u32(std::span<const unsigned char> b, std::size_t at) {
    std::uint32_t v = 0;
    std::memcpy(&v, b.data() + at, 4);
    return v;
}
[[nodiscard]] inline std::int32_t read_i32(std::span<const unsigned char> b, std::size_t at) {
    std::int32_t v = 0;
    std::memcpy(&v, b.data() + at, 4);
    return v;
}
[[nodiscard]] inline std::int64_t read_i64(std::span<const unsigned char> b, std::size_t at) {
    std::int64_t v = 0;
    std::memcpy(&v, b.data() + at, 8);
    return v;
}

// ---- FEATURE frame codec (producer -> server) ----
// Encode one thread's snapshot into a FEATURE frame. Total (a well-typed input always produces a frame),
// returns by value.
[[nodiscard]] inline std::vector<unsigned char> encode_feature(const LabFeature& f) {
    std::vector<unsigned char> out;
    out.reserve(LAB_FEAT_BYTES);
    put_u32(out, LAB_FEAT_MAGIC);
    out.push_back(LAB_WIRE_VERSION);
    put_i32(out, f.tid);
    put_i32(out, f.inflight);
    put_i32(out, f.ready);
    put_i64(out, f.msgs);
    put_i64(out, f.leaves);
    put_i64(out, f.rtt_us);
    put_i64(out, f.decisions);
    return out;
}

// Decode a FEATURE frame. BOUNDARY (ADR-0002): a wrong magic, a wrong version, or a wrong length is a
// typed Error — never a silently misread snapshot.
[[nodiscard]] inline std::expected<LabFeature, Error> decode_feature(
    std::span<const unsigned char> frame) {
    if (frame.size() != LAB_FEAT_BYTES)
        return std::unexpected(make_error("lab control wire: FEATURE frame is " +
                                          std::to_string(frame.size()) + " bytes, expected " +
                                          std::to_string(LAB_FEAT_BYTES)));
    if (read_u32(frame, 0) != LAB_FEAT_MAGIC)
        return std::unexpected(make_error("lab control wire: bad FEATURE magic (wire-contract drift, P7)"));
    if (frame[4] != LAB_WIRE_VERSION)
        return std::unexpected(make_error("lab control wire: FEATURE version " +
                                          std::to_string(frame[4]) + " != supported " +
                                          std::to_string(LAB_WIRE_VERSION)));
    LabFeature f;
    f.tid = read_i32(frame, 5);
    f.inflight = read_i32(frame, 9);
    f.ready = read_i32(frame, 13);
    f.msgs = read_i64(frame, 17);
    f.leaves = read_i64(frame, 25);
    f.rtt_us = read_i64(frame, 33);
    f.decisions = read_i64(frame, 41);
    return f;
}

// ---- GATE frame codec (server -> producer) ----
// Encode one thread's next issue-gate bit (1 = allow the next discretionary issue, 0 = deny).
[[nodiscard]] inline std::vector<unsigned char> encode_gate(std::int32_t tid, bool allow) {
    std::vector<unsigned char> out;
    out.reserve(LAB_GATE_BYTES);
    put_u32(out, LAB_GATE_MAGIC);
    out.push_back(LAB_WIRE_VERSION);
    put_i32(out, tid);
    out.push_back(allow ? 1 : 0);
    return out;
}

// Decode a GATE frame to (tid, allow). BOUNDARY (ADR-0002): a wrong magic/version/length is a typed Error.
struct LabGate {
    std::int32_t tid = 0;
    bool allow = true;
};
[[nodiscard]] inline std::expected<LabGate, Error> decode_gate(std::span<const unsigned char> frame) {
    if (frame.size() != LAB_GATE_BYTES)
        return std::unexpected(make_error("lab control wire: GATE frame is " +
                                          std::to_string(frame.size()) + " bytes, expected " +
                                          std::to_string(LAB_GATE_BYTES)));
    if (read_u32(frame, 0) != LAB_GATE_MAGIC)
        return std::unexpected(make_error("lab control wire: bad GATE magic (wire-contract drift, P7)"));
    if (frame[4] != LAB_WIRE_VERSION)
        return std::unexpected(make_error("lab control wire: GATE version " +
                                          std::to_string(frame[4]) + " != supported " +
                                          std::to_string(LAB_WIRE_VERSION)));
    LabGate g;
    g.tid = read_i32(frame, 5);
    g.allow = frame[9] != 0;
    return g;
}

}  // namespace chocofarm::lab
