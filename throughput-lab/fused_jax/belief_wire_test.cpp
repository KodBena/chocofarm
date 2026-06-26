// throughput-lab/fused_jax/belief_wire_test.cpp
// Purpose: a self-contained (no env, no redis, no JAX) unit test of the belief-batch wire codec
//   (belief_wire.hpp). It nets encode -> decode round-trips for the setup / request / response frames
//   AND exercises the boundary-validation paths (ADR-0002 fail-loud: a truncated frame, a wrong
//   protocol byte, a B/kW64 of 0, a ragged body MUST throw). Pure-logic; runs in the standalone build.
//   Exit 0 = all pass; nonzero + a message = first failure (fail loudly).
// Public Domain (The Unlicense).
#include <cstdint>
#include <cstdio>
#include <span>
#include <stdexcept>
#include <vector>

#include "belief_wire.hpp"

namespace bw = tlab::bwire;

namespace {
int failures = 0;
void check(bool ok, const char* what) {
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", what); ++failures; }
}
// Assert that calling `fn` throws std::invalid_argument (the loud boundary failure).
template <class F>
void check_throws(F fn, const char* what) {
    bool threw = false;
    try { fn(); } catch (const std::invalid_argument&) { threw = true; }
    check(threw, what);
}
}  // namespace

int main() {
    // ---- SETUP round-trip ----
    {
        const bw::count_t N = 3, nD = 2, nworlds = 130, kW64 = 3;  // ceil(130/64)=3
        std::vector<bw::word_t> matrix((N + nD) * kW64);
        for (std::size_t i = 0; i < matrix.size(); ++i)
            matrix[i] = 0x0123456789abcdefULL ^ (static_cast<bw::word_t>(i) * 0x100000001ULL);
        auto frame = bw::encode_setup(N, nD, nworlds, kW64, matrix);
        auto s = bw::decode_setup(frame);
        check(s.N == N && s.nD == nD && s.nworlds == nworlds && s.kW64 == kW64, "setup dims round-trip");
        check(s.matrix == matrix, "setup matrix round-trip (bit-exact)");
    }

    // ---- REQUEST round-trip ----
    {
        const bw::count_t kW64 = 3;
        std::vector<bw::BeliefLeaf> leaves;
        for (bw::count_t i = 0; i < 4; ++i) {
            bw::BeliefLeaf lf;
            lf.loc = 10 + i;
            lf.collected = (bw::count_t{1} << i) | 0x5u;
            lf.belief.assign(kW64, 0);
            for (bw::count_t w = 0; w < kW64; ++w) lf.belief[w] = 0xdeadbeef00000000ULL + i * 7 + w;
            leaves.push_back(std::move(lf));
        }
        auto frame = bw::encode_request(leaves, kW64);
        auto r = bw::decode_request(frame);
        check(r.B == 4 && r.kW64 == kW64, "request B/kW64 round-trip");
        bool eq = (r.leaves.size() == leaves.size());
        for (std::size_t i = 0; eq && i < leaves.size(); ++i)
            eq = (r.leaves[i].loc == leaves[i].loc) &&
                 (r.leaves[i].collected == leaves[i].collected) &&
                 (r.leaves[i].belief == leaves[i].belief);
        check(eq, "request leaves round-trip (bit-exact)");
    }

    // ---- RESPONSE round-trip (value-only and with logits) ----
    {
        std::vector<float> values{0.5f, -1.25f, 3.0f};
        std::vector<float> none;
        auto f0 = bw::encode_response(values, none, 0);
        auto d0 = bw::decode_response(f0);
        check(d0.size() == 3 && d0[1].value == -1.25f && d0[1].logits.empty(), "response value-only round-trip");

        const bw::count_t nA = 4;
        std::vector<float> logits(values.size() * nA);
        for (std::size_t i = 0; i < logits.size(); ++i) logits[i] = static_cast<float>(i) * 0.5f - 1.0f;
        auto f1 = bw::encode_response(values, logits, nA);
        auto d1 = bw::decode_response(f1);
        bool eq = (d1.size() == 3);
        for (std::size_t r = 0; eq && r < 3; ++r) {
            eq = (d1[r].value == values[r]) && (d1[r].logits.size() == nA);
            for (bw::count_t c = 0; eq && c < nA; ++c) eq = (d1[r].logits[c] == logits[r * nA + c]);
        }
        check(eq, "response value+logits round-trip (bit-exact)");
    }

    // ---- BOUNDARY validation (ADR-0002 fail-loud) ----
    {
        // a valid request frame to mutilate
        std::vector<bw::BeliefLeaf> leaves(2);
        for (auto& lf : leaves) lf.belief.assign(2, 1);
        auto good = bw::encode_request(leaves, 2);

        // each lambda's codec call is EXPECTED to throw before returning; (void)-cast the [[nodiscard]]
        // result so the deliberate discard is not a -Wunused-result warning.
        check_throws([&] { (void)bw::decode_request(std::span<const unsigned char>(good.data(), 3)); },
                     "decode_request rejects a too-short frame");
        auto badver = good; badver[0] = 0xFF;
        check_throws([&] { (void)bw::decode_request(badver); }, "decode_request rejects a wrong protocol byte");
        auto truncated = good; truncated.pop_back();
        check_throws([&] { (void)bw::decode_request(truncated); }, "decode_request rejects a ragged body");

        check_throws([&] { (void)bw::encode_request(std::span<const bw::BeliefLeaf>{}, 2); },
                     "encode_request rejects an empty batch");
        std::vector<bw::BeliefLeaf> ragged(1); ragged[0].belief.assign(1, 0);  // 1 word but kW64=2
        check_throws([&] { (void)bw::encode_request(ragged, 2); },
                     "encode_request rejects a leaf whose belief != kW64 words");

        std::vector<bw::word_t> m(2);
        check_throws([&] { (void)bw::encode_setup(1, 1, 100, 0, m); }, "encode_setup rejects kW64=0");
        check_throws([&] { (void)bw::encode_setup(1, 1, 100, 2, std::span<const bw::word_t>(m.data(), 3)); },
                     "encode_setup rejects matrix size != (N+nD)*kW64");

        // decode_setup negative paths (it is on the live round-trip path; cover its boundary code too).
        auto good_setup = bw::encode_setup(2, 1, 130, 2, std::vector<bw::word_t>(6));  // (2+1)*2 words
        check_throws([&] { (void)bw::decode_setup(std::span<const unsigned char>(good_setup.data(), 5)); },
                     "decode_setup rejects a too-short frame");
        auto bad_setup_ver = good_setup; bad_setup_ver[0] = 0xFF;
        check_throws([&] { (void)bw::decode_setup(bad_setup_ver); }, "decode_setup rejects a wrong version");
        auto trunc_setup = good_setup; trunc_setup.pop_back();
        check_throws([&] { (void)bw::decode_setup(trunc_setup); }, "decode_setup rejects a ragged body");

        // decode_response negative paths (also on the round-trip path: the C++ receive side decodes it).
        std::vector<float> rv{1.0f, 2.0f};
        std::vector<float> rl(2 * 3);
        auto good_resp = bw::encode_response(rv, rl, 3);
        check_throws([&] { (void)bw::decode_response(std::span<const unsigned char>(good_resp.data(), 4)); },
                     "decode_response rejects a too-short frame");
        auto bad_resp_ver = good_resp; bad_resp_ver[0] = 0xFF;
        check_throws([&] { (void)bw::decode_response(bad_resp_ver); }, "decode_response rejects a wrong version");
        auto trunc_resp = good_resp; trunc_resp.pop_back();
        check_throws([&] { (void)bw::decode_response(trunc_resp); }, "decode_response rejects a ragged body");

        // decode_request with an in-frame B=0 / kW64=0 (a header that decodes to a zero count, not just
        // an encode-side guard): hand-craft a frame whose B field is 0.
        auto zeroB = good;  // a valid 2-leaf kW64=2 request
        zeroB[bw::VERSION_BYTES] = 0; zeroB[bw::VERSION_BYTES + 1] = 0;  // B (u32 LE) -> 0
        zeroB[bw::VERSION_BYTES + 2] = 0; zeroB[bw::VERSION_BYTES + 3] = 0;
        check_throws([&] { (void)bw::decode_request(zeroB); }, "decode_request rejects an in-frame B=0");
    }

    if (failures == 0) { std::printf("belief_wire_test: ALL PASS\n"); return 0; }
    std::fprintf(stderr, "belief_wire_test: %d FAILURE(S)\n", failures);
    return 1;
}
