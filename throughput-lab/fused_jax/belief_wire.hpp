// throughput-lab/fused_jax/belief_wire.hpp
// Purpose: the ONE authoritative byte-level spec + header-only codec of the FUSED-JAX BatchPredict
//   wire (lever #1, docs/notes/batchpredict-throughput-design-2026-06-26.md). Where the production
//   inference wire (throughput-lab/cpp/wire.hpp) ships FEATURES (X : f32 x B*in_dim — the producer
//   featurizes, then sends the feature matrix), THIS wire ships the raw BELIEF BATCH: B leaves, each a
//   (loc, belief-rank-bitset, collected) triple. The JAX side featurizes (the belief_indicator @
//   world_feature_matrix matmul, de-risked 2026-06-26) + runs the net, FUSED. The env-static
//   world_feature_matrix is NOT per-leaf: it is a SETUP fact sent ONCE (encode_setup), not in the
//   per-batch frame.
//
//   This is a CLEAN-ROOM COMPONENT codec, mirroring wire.hpp's Layer-1 discipline byte-for-byte in
//   spirit (a one-byte protocol-version tripwire fronting a length-prefixed little-endian frame; all
//   multi-byte fields LE; boundary validation that fails LOUDLY per ADR-0002; the COUNT/header widths
//   spelled as constants that the layout derives from, one home — ADR-0012 P7). It is a SEPARATE
//   protocol from wire.hpp (a distinct PROTOCOL_VERSION namespace) so a belief frame fed to the
//   feature decoder (or vice versa) fails at the version byte instead of misreading a bitset word as a
//   float. It does NOT touch the production producer/server path.
//
// Public Domain (The Unlicense).
//
// ============================================================================================
//  THE BELIEF WIRE — two frames (a SETUP frame sent once, a per-batch REQUEST frame), one RESPONSE
// ============================================================================================
//
//  SETUP frame (env-static, sent ONCE at boundary establishment, NOT per leaf)
//  -------------------------------------------------------------------------------------------
//      Setup : [bver:u8][N:u32][nD:u32][nworlds:u32][kW64:u32]
//              [ world_feature_matrix : u64 x ((N+nD)*kW64) LE ]   (column-major rank bitsets)
//
//   - bver       : BELIEF_PROTOCOL_VERSION (the codec-mismatch tripwire; distinct from wire.hpp's).
//   - N / nD     : treasure / detector counts (the matrix has N+nD columns).
//   - nworlds    : |worlds()| — the rank space; the matmul's contracted (world) axis.
//   - kW64       : ceil(nworlds/64) — the per-column rank-bitset word count (env.kW64()).
//   - matrix     : N+nD columns, each kW64 u64 words. column t (t<N)   = treasure_mask(t);
//                  column N+j           = detector_mask(j). bit r of a column set iff the
//                  rank-r world is in that column's set (the de-risk's world_feature_matrix).
//
//  REQUEST frame (the per-batch belief leaves)
//  -------------------------------------------------------------------------------------------
//      Request : [bver:u8][B:u32][kW64:u32]
//                [ B records, each : (loc:u32 LE)(collected:u32 LE)(belief : u64 x kW64 LE) ]
//
//   - B          : leaf rows in this batch (>= 1). B=1 is the degenerate single-leaf case.
//   - kW64       : echoed (MUST equal the setup kW64; a per-frame guard, validated on decode — the
//                  belief bitset width has ONE home, the env, ADR-0012 P1).
//   - per record :
//       * loc       : the standing point, an opaque u32 the JAX side passes to the net's loc-embedding
//                     (the component's stand-in net does not use it; named so the real net can).
//       * collected : a 32-bit collected-treasure bitmask (treasure t collected iff bit t; N<=32 on the
//                     live env). Opaque to the matmul (it gates the `available` mask, a net-side map);
//                     carried so the leaf is COMPLETE (loc, belief, collected) per the seam.
//       * belief    : the live-world rank bitset (kW64 u64 words) — bit r set iff the rank-r world is
//                     live. This IS the belief_indicator the matmul multiplies. FIXED kW64 words
//                     regardless of nb (it spans the full rank space) — the 2x-wire planning number.
//
//  RESPONSE frame (predictions back — the fused featurize+net output)
//  -------------------------------------------------------------------------------------------
//      Response : [bver:u8][B:u32][n_actions:u32]
//                 [ B records, each : (value:f32 LE)(logits : f32 x n_actions LE) ]
//
//   Identical SHAPE to wire.hpp's response (value + raw logits per row) — the predictions a BatchPredict
//   returns are the same TYPE regardless of which impl produced them (#1 fused-JAX vs #3 in-process), so
//   the response codec is deliberately the SAME layout. n_actions == 0 => value-only.
//
//  Fixed header (all three frames lead with bver:u8). Setup body = (N+nD)*kW64*8. Request body =
//  B*(4 + 4 + kW64*8). Response body = B*(1 + n_actions)*4.
//
// ============================================================================================

#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <stdexcept>
#include <vector>

#include "chocofarm/quantity.hpp"  // the zero-cost phantom machinery (Band-1, reused across the boundary)

namespace tlab::bwire {

// ---- the protocol-version header byte (the codec-mismatch tripwire) ----------------------------
// DISTINCT from tlab::wire::PROTOCOL_VERSION: this is the BELIEF wire (ships beliefs, not features). A
// belief frame fed to the feature decoder (or vice versa) fails at this byte rather than misreading a
// bitset word as a float (ADR-0002). Bump on ANY belief-frame layout change.
inline constexpr std::uint8_t BELIEF_PROTOCOL_VERSION = 1;

// ---- fixed-field byte widths (these ARE the layout; mirror wire.hpp's COUNT/FLOAT discipline) ---
inline constexpr std::size_t VERSION_BYTES = 1;   // u8  — the belief-protocol-version header byte
inline constexpr std::size_t COUNT_BYTES   = 4;   // u32 — a length/index prefix (B, N, nD, nworlds, kW64)
inline constexpr std::size_t WORD_BYTES    = 8;   // u64 — a rank-bitset word (the belief / matrix words)
inline constexpr std::size_t FLOAT_BYTES   = 4;   // f32 — a response float (value / logit)

using version_t = std::uint8_t;
using count_t   = std::uint32_t;
using word_t    = std::uint64_t;   // a rank-bitset word, native-LE on the host (the standing assumption)
using float_t   = float;           // IEEE-754 binary32, matching numpy '<f4'

static_assert(sizeof(version_t) == VERSION_BYTES, "version_t width must match VERSION_BYTES");
static_assert(sizeof(count_t)   == COUNT_BYTES,   "count_t width must match COUNT_BYTES");
static_assert(sizeof(word_t)    == WORD_BYTES,    "word_t width must match WORD_BYTES");
static_assert(sizeof(float_t)   == FLOAT_BYTES,   "float_t width must match FLOAT_BYTES");

// ============================================================================================
//  PHANTOM SPLIT (ADR-0000 / ADR-0012 P6) — reuses chocofarm's Quantity<Tag,Rep> machinery, same as
//  wire.hpp. The on-wire bytes stay BIT-IDENTICAL (each phantom wraps count_t and .value()-unwraps at
//  the byte read/write). Distinct tags make the B<->kW64 swap (a row count vs a bitset word count) a
//  COMPILE error in any in-memory mix — the exact class of bug ADR-0000 forbids representing.
// ============================================================================================

// B — leaf ROWS per batch (>= 1). A row count, NOT a width.
struct RowCountTag {};
using RowCount = chocofarm::Quantity<RowCountTag, count_t>;

// kW64 — the rank-bitset WORD count per column/belief. The B<->kW64 swap is the bug a distinct tag forbids.
struct WordCountTag {};
using WordCount = chocofarm::Quantity<WordCountTag, count_t>;

}  // namespace tlab::bwire

namespace chocofarm {
// A row count + a row count is a row count; a loop index over [0,B). A word count likewise indexes the
// bitset words [0,kW64). Both additive + affine; never inter-mix (distinct tags enforce it).
template <> struct quantity_additive<tlab::bwire::RowCountTag> : std::true_type {};
template <> struct quantity_affine<tlab::bwire::RowCountTag>   : std::true_type {};
template <> struct quantity_additive<tlab::bwire::WordCountTag> : std::true_type {};
template <> struct quantity_affine<tlab::bwire::WordCountTag>   : std::true_type {};
}  // namespace chocofarm

namespace tlab::bwire {

// ---- little-endian field helpers (host is LE — the codec's standing assumption, as wire.hpp) ----
inline void put_count(std::vector<unsigned char>& out, count_t v) {
    unsigned char b[COUNT_BYTES];
    std::memcpy(b, &v, COUNT_BYTES);
    out.insert(out.end(), b, b + COUNT_BYTES);
}
inline void put_word(std::vector<unsigned char>& out, word_t v) {
    unsigned char b[WORD_BYTES];
    std::memcpy(b, &v, WORD_BYTES);   // native-LE words, matching the env's in-memory bitset layout
    out.insert(out.end(), b, b + WORD_BYTES);
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
[[nodiscard]] inline word_t read_word(std::span<const unsigned char> bytes, std::size_t at) {
    word_t v = 0;
    std::memcpy(&v, bytes.data() + at, WORD_BYTES);
    return v;
}
[[nodiscard]] inline float_t read_f32(std::span<const unsigned char> bytes, std::size_t at) {
    float_t v = 0.0f;
    std::memcpy(&v, bytes.data() + at, FLOAT_BYTES);
    return v;
}

// ============================================================================================
//  SETUP FRAME — the env-static world_feature_matrix, sent ONCE.
// ============================================================================================

struct SetupFields {
    count_t N = 0, nD = 0, nworlds = 0, kW64 = 0;
    std::vector<word_t> matrix;   // (N+nD)*kW64 words, column-major: column c at matrix[c*kW64 .. +kW64]
};

// Encode the setup frame. `matrix` is the (N+nD)*kW64 column-major rank-bitset words (column t<N =
// treasure_mask(t); column N+j = detector_mask(j)). Validates the length against (N+nD)*kW64 — never
// silently re-derives a disagreeing length (ADR-0012 P1).
[[nodiscard]] inline std::vector<unsigned char> encode_setup(
        count_t N, count_t nD, count_t nworlds, count_t kW64, std::span<const word_t> matrix) {
    if (kW64 == 0)    throw std::invalid_argument("belief wire: encode_setup kW64 is 0 (env not enumerable)");
    if (nworlds == 0) throw std::invalid_argument("belief wire: encode_setup nworlds is 0");
    const std::size_t want = static_cast<std::size_t>(N + nD) * kW64;
    if (matrix.size() != want)
        throw std::invalid_argument("belief wire: encode_setup matrix size != (N+nD)*kW64");
    std::vector<unsigned char> out;
    out.reserve(VERSION_BYTES + 4 * COUNT_BYTES + matrix.size() * WORD_BYTES);
    out.push_back(static_cast<unsigned char>(BELIEF_PROTOCOL_VERSION));
    put_count(out, N); put_count(out, nD); put_count(out, nworlds); put_count(out, kW64);
    for (word_t w : matrix) put_word(out, w);
    return out;
}

// Decode a setup frame. BOUNDARY validation (ADR-0002): unknown version byte, too-short frame, zero
// dims, or a body length != (N+nD)*kW64*8 throws.
[[nodiscard]] inline SetupFields decode_setup(std::span<const unsigned char> frame) {
    const std::size_t header = VERSION_BYTES + 4 * COUNT_BYTES;
    if (frame.size() < header) throw std::invalid_argument("belief wire: setup frame too short for header");
    if (frame[0] != BELIEF_PROTOCOL_VERSION)
        throw std::invalid_argument("belief wire: setup protocol-byte mismatch");
    SetupFields s;
    s.N       = read_count(frame, VERSION_BYTES);
    s.nD      = read_count(frame, VERSION_BYTES + COUNT_BYTES);
    s.nworlds = read_count(frame, VERSION_BYTES + 2 * COUNT_BYTES);
    s.kW64    = read_count(frame, VERSION_BYTES + 3 * COUNT_BYTES);
    if (s.kW64 == 0)    throw std::invalid_argument("belief wire: setup kW64 is 0");
    if (s.nworlds == 0) throw std::invalid_argument("belief wire: setup nworlds is 0");
    const std::size_t n = static_cast<std::size_t>(s.N + s.nD) * s.kW64;
    if (frame.size() != header + n * WORD_BYTES)
        throw std::invalid_argument("belief wire: setup body length != (N+nD)*kW64*u64");
    s.matrix.resize(n);
    for (std::size_t i = 0; i < n; ++i) s.matrix[i] = read_word(frame, header + i * WORD_BYTES);
    return s;
}

// ============================================================================================
//  REQUEST FRAME — the per-batch belief leaves.
// ============================================================================================

// One decoded leaf: (loc, collected, belief rank-bitset).
struct BeliefLeaf {
    count_t loc = 0;
    count_t collected = 0;          // a 32-bit collected-treasure bitmask (treasure t iff bit t)
    std::vector<word_t> belief;     // kW64 rank-bitset words (the belief_indicator)
};

struct RequestFields {
    count_t B = 0;
    count_t kW64 = 0;
    std::vector<BeliefLeaf> leaves;  // B leaves
};

// Encode the per-batch request. `leaves` are the B (loc, collected, belief) triples; each belief MUST
// be exactly kW64 words (validated — the bitset width has one home, ADR-0012 P1).
[[nodiscard]] inline std::vector<unsigned char> encode_request(
        std::span<const BeliefLeaf> leaves, count_t kW64) {
    if (leaves.empty()) throw std::invalid_argument("belief wire: encode_request empty batch (B=0)");
    if (kW64 == 0)      throw std::invalid_argument("belief wire: encode_request kW64 is 0");
    const auto B = static_cast<count_t>(leaves.size());
    const std::size_t rec = 2 * COUNT_BYTES + static_cast<std::size_t>(kW64) * WORD_BYTES;
    std::vector<unsigned char> out;
    out.reserve(VERSION_BYTES + 2 * COUNT_BYTES + static_cast<std::size_t>(B) * rec);
    out.push_back(static_cast<unsigned char>(BELIEF_PROTOCOL_VERSION));
    put_count(out, B); put_count(out, kW64);
    for (const BeliefLeaf& lf : leaves) {
        if (lf.belief.size() != kW64)
            throw std::invalid_argument("belief wire: encode_request leaf belief size != kW64");
        put_count(out, lf.loc); put_count(out, lf.collected);
        for (word_t w : lf.belief) put_word(out, w);
    }
    return out;
}

// Decode a request frame. BOUNDARY validation (ADR-0002): unknown version, too-short frame, B/kW64 of
// 0, or a body length != B*(8 + kW64*8) throws.
[[nodiscard]] inline RequestFields decode_request(std::span<const unsigned char> frame) {
    const std::size_t header = VERSION_BYTES + 2 * COUNT_BYTES;
    if (frame.size() < header) throw std::invalid_argument("belief wire: request frame too short for header");
    if (frame[0] != BELIEF_PROTOCOL_VERSION)
        throw std::invalid_argument("belief wire: request protocol-byte mismatch");
    RequestFields r;
    r.B    = read_count(frame, VERSION_BYTES);
    r.kW64 = read_count(frame, VERSION_BYTES + COUNT_BYTES);
    if (r.B == 0)    throw std::invalid_argument("belief wire: request B is 0");
    if (r.kW64 == 0) throw std::invalid_argument("belief wire: request kW64 is 0");
    const std::size_t rec = 2 * COUNT_BYTES + static_cast<std::size_t>(r.kW64) * WORD_BYTES;
    if (frame.size() != header + static_cast<std::size_t>(r.B) * rec)
        throw std::invalid_argument("belief wire: request body length != B*(loc+collected+kW64*u64)");
    r.leaves.resize(r.B);
    for (count_t i = 0; i < r.B; ++i) {
        const std::size_t base = header + static_cast<std::size_t>(i) * rec;
        BeliefLeaf& lf = r.leaves[i];
        lf.loc       = read_count(frame, base);
        lf.collected = read_count(frame, base + COUNT_BYTES);
        lf.belief.resize(r.kW64);
        const std::size_t wbase = base + 2 * COUNT_BYTES;
        for (count_t w = 0; w < r.kW64; ++w)
            lf.belief[w] = read_word(frame, wbase + static_cast<std::size_t>(w) * WORD_BYTES);
    }
    return r;
}

// ============================================================================================
//  RESPONSE FRAME — predictions back (same SHAPE as wire.hpp's response — a BatchPredict prediction is
//  one type regardless of impl). The JAX side ENCODES it; this C++ side DECODES (the round-trip demo).
// ============================================================================================

struct ResponseFields {
    float value = 0.0f;
    std::vector<float> logits;   // raw logits over n_actions slots (empty when n_actions == 0)
};

// Encode predictions. `values`/`logits_flat` are B values and B*n_actions row-major logits.
[[nodiscard]] inline std::vector<unsigned char> encode_response(
        std::span<const float> values, std::span<const float> logits_flat, count_t n_actions) {
    if (values.empty()) throw std::invalid_argument("belief wire: encode_response B is 0");
    const auto B = static_cast<count_t>(values.size());
    const std::size_t want = static_cast<std::size_t>(B) * n_actions;
    if (logits_flat.size() != want)
        throw std::invalid_argument("belief wire: encode_response logits size != B*n_actions");
    std::vector<unsigned char> out;
    out.reserve(VERSION_BYTES + 2 * COUNT_BYTES + static_cast<std::size_t>(B) * (1 + n_actions) * FLOAT_BYTES);
    out.push_back(static_cast<unsigned char>(BELIEF_PROTOCOL_VERSION));
    put_count(out, B); put_count(out, n_actions);
    for (count_t r = 0; r < B; ++r) {
        put_f32(out, values[r]);
        for (count_t c = 0; c < n_actions; ++c)
            put_f32(out, logits_flat[static_cast<std::size_t>(r) * n_actions + c]);
    }
    return out;
}

// Decode a response frame back to B ResponseFields. BOUNDARY validation (ADR-0002).
[[nodiscard]] inline std::vector<ResponseFields> decode_response(std::span<const unsigned char> frame) {
    const std::size_t header = VERSION_BYTES + 2 * COUNT_BYTES;
    if (frame.size() < header) throw std::invalid_argument("belief wire: response frame too short for header");
    if (frame[0] != BELIEF_PROTOCOL_VERSION)
        throw std::invalid_argument("belief wire: response protocol-byte mismatch");
    count_t B = read_count(frame, VERSION_BYTES);
    count_t n_actions = read_count(frame, VERSION_BYTES + COUNT_BYTES);
    if (B == 0) throw std::invalid_argument("belief wire: response B is 0");
    const std::size_t per = static_cast<std::size_t>(1 + n_actions);
    if (frame.size() != header + static_cast<std::size_t>(B) * per * FLOAT_BYTES)
        throw std::invalid_argument("belief wire: response body length != B*(1+n_actions)*f32");
    std::vector<ResponseFields> out(B);
    for (count_t r = 0; r < B; ++r) {
        const std::size_t base = header + static_cast<std::size_t>(r) * per * FLOAT_BYTES;
        out[r].value = read_f32(frame, base);
        out[r].logits.resize(n_actions);
        for (count_t c = 0; c < n_actions; ++c)
            out[r].logits[c] = read_f32(frame, base + static_cast<std::size_t>(1 + c) * FLOAT_BYTES);
    }
    return out;
}

}  // namespace tlab::bwire
