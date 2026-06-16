// cpp/parity/wire_golden.cpp
// Purpose: the C++ HALF of the cross-language golden-vector round-trip for the two #23-mechanized
//   raw-binary contracts — the Shape B ZeroMQ inference wire frame and the redis RESULT blob. It is a
//   standalone, dependency-FREE program (it includes ONLY the SSOT mirror headers
//   chocofarm/wire_spec.hpp + chocofarm/result_spec.hpp; no hiredis, no zmq, no nlohmann, no cmake
//   build) so the Python drift test can compile it with a bare `g++ -std=c++23` and prove the C++
//   codec — DERIVED from the mirror constants, never re-authoring the `[ver][count][f32…]` /
//   X·PI·M·Y·f32 layout — decodes a Python-encoded golden frame byte-for-byte and re-encodes it
//   identically (ADR-0012 P6 bar for a BYTE format is byte-exactness, not float tolerance).
//
//   This is the OPT-IN cross-language leg of tests/test_wire_drift.py (gated CHOCO_RUN_CPP, mirroring
//   tests/test_cpp_runner.py). The ALWAYS-ON leg is the pure-Python constant-agreement test that needs
//   no compiler; this program is the stronger end-to-end check when a C++ toolchain is present.
//
//   The wire round-trip drives the SHARED C++ codec (chocofarm/inference_wire.hpp) — the SAME
//   encode/decode the ZmqNetClient uses — so this golden leg covers the production codec, not a
//   bespoke passthrough (ADR-0012 P1: one codec, derived from the wire_spec SSOT). The codec header is
//   itself dependency-free (no zmq), so this program still compiles with a bare `g++ -std=c++23`.
//
//   Protocol (stdin → stdout, all little-endian, length-prefixed by an explicit count so the harness
//   needs no shared framing of its own):
//
//     argv[1] == "wire":
//       DECODE ONE inference REQUEST frame to its (in_dim, X) fields and ONE RESPONSE frame to its
//       (value, logits) fields via the shared codec, then RE-ENCODE each from those decoded fields and
//       emit the two re-encoded frames. The Python side asserts the bytes returned == the bytes sent
//       (exact inverse through the production codec — decode∘encode is identity, no drift).
//         stdin : [u32 req_len][req_bytes][u32 resp_len][resp_bytes]
//         stdout: [u32 req_len][req_bytes][u32 resp_len][resp_bytes]   (re-encoded)
//
//     argv[1] == "result":
//       Decode ONE result-blob group (the four X/PI/M/Y float32 blocks) using the result_spec mirror's
//       block order + dtype width, then re-emit each block's bytes in BLOCK_ORDER. The Python side
//       asserts equality block-for-block.
//         stdin : for each name in result::BLOCK_ORDER: [u32 nbytes][block_bytes]
//         stdout: for each name in result::BLOCK_ORDER: [u32 nbytes][block_bytes]   (round-tripped)
//
//   Every read is bounds-checked; a malformed/over-short stream is a loud nonzero exit (ADR-0002),
//   never a silent truncated echo.
//
// Public Domain (The Unlicense).
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "chocofarm/inference_wire.hpp"
#include "chocofarm/result_spec.hpp"
#include "chocofarm/wire_spec.hpp"

namespace {

// Read EXACTLY n bytes from stdin into `out` (appended). Loud false on short read (ADR-0002).
bool read_exact(std::vector<unsigned char>& out, std::size_t n) {
    std::size_t base = out.size();
    out.resize(base + n);
    std::size_t got = std::fread(out.data() + base, 1, n, stdin);
    return got == n;
}

// Read a u32 length prefix (native little-endian; the host is LE — wire_spec's standing assumption).
bool read_u32(std::uint32_t& v) {
    unsigned char b[4];
    if (std::fread(b, 1, 4, stdin) != 4) return false;
    std::memcpy(&v, b, 4);
    return true;
}

// Write a u32 length prefix + the bytes back to stdout.
void write_framed(std::span<const unsigned char> bytes) {
    std::uint32_t n = static_cast<std::uint32_t>(bytes.size());
    std::fwrite(&n, 1, 4, stdout);
    std::fwrite(bytes.data(), 1, bytes.size(), stdout);
}

// Read a u32-length-prefixed chunk from stdin.
bool read_framed(std::vector<unsigned char>& out) {
    std::uint32_t n = 0;
    if (!read_u32(n)) return false;
    out.clear();
    return read_exact(out, n);
}

// ---- inference wire: DECODE→RE-ENCODE through the SHARED codec (chocofarm/inference_wire.hpp) ----
// Decode a REQUEST [ver:u8][in_dim:u32][X:f32×in_dim] to its X field via the shared codec, then
// RE-ENCODE from that field. decode∘encode is identity, so the harness asserting bytes-out==bytes-in
// proves the production codec (the one ZmqNetClient uses) round-trips the Python-encoded golden frame.
// Returns the re-encoded frame, or empty on a decode error (loud caller exit — layout drift).
std::vector<unsigned char> roundtrip_request(std::span<const unsigned char> frame) {
    auto decoded = chocofarm::wire::decode_request(frame);
    if (!decoded) return {};   // unknown protocol byte / wrong length is loud (caller sees empty)
    return chocofarm::wire::encode_request(*decoded);
}

// Decode a RESPONSE [ver:u8][n_actions:u32][value:f32][logits:f32×n_actions] to its (value, logits)
// fields via the shared codec, then re-encode from them — same exact-inverse proof for the response.
std::vector<unsigned char> roundtrip_response(std::span<const unsigned char> frame) {
    auto decoded = chocofarm::wire::decode_response(frame);
    if (!decoded) return {};
    return chocofarm::wire::encode_response(decoded->value, decoded->logits);
}

int run_wire() {
    std::vector<unsigned char> req, resp;
    if (!read_framed(req) || !read_framed(resp)) {
        std::fprintf(stderr, "wire_golden: short read on request/response frames\n");
        return 2;
    }
    std::vector<unsigned char> req_out = roundtrip_request(req);
    std::vector<unsigned char> resp_out = roundtrip_response(resp);
    if (req_out.empty() || resp_out.empty()) {
        std::fprintf(stderr, "wire_golden: frame failed mirror-derived decode (layout drift?)\n");
        return 3;
    }
    write_framed(req_out);
    write_framed(resp_out);
    return 0;
}

int run_result() {
    namespace r = chocofarm::result;
    // Read the four blocks in the mirror's BLOCK_ORDER; re-emit each block's raw bytes in that order.
    // (block_t is float; BLOCK_ITEMSIZE the width — we copy bytes through, so the round-trip is exact,
    // but we VALIDATE each block length is a whole number of float32 elements by the mirror width.)
    std::vector<std::vector<unsigned char>> blocks(r::BLOCK_ORDER.size());
    for (std::size_t i = 0; i < r::BLOCK_ORDER.size(); ++i) {
        if (!read_framed(blocks[i])) {
            std::fprintf(stderr, "wire_golden: short read on result block %zu\n", i);
            return 2;
        }
        if (blocks[i].size() % r::BLOCK_ITEMSIZE != 0) {
            std::fprintf(stderr, "wire_golden: result block %zu is not a whole number of f32 elements\n", i);
            return 3;
        }
    }
    for (const auto& blk : blocks) write_framed(blk);
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: wire_golden <wire|result>\n");
        return 64;
    }
    std::string_view mode = argv[1];
    if (mode == "wire") return run_wire();
    if (mode == "result") return run_result();
    std::fprintf(stderr, "wire_golden: unknown mode '%s'\n", argv[1]);
    return 64;
}
