// throughput-lab/fused_jax/belief_response_decode.cpp
// Purpose: the C++ RECEIVE SIDE that closes the fused-JAX BatchPredict round-trip. Reads a RESPONSE
//   frame (belief_wire.hpp, produced by the JAX featurize+predict side) and prints the decoded
//   predictions (value + a logits summary per row) so the demo can show C++ encode -> JAX predict ->
//   C++ decode end-to-end. It is the symmetric partner of belief_batch_encode (ADR-0012 P3: the encode
//   side and the decode side share ONE codec header, neither re-authors the layout).
//
//   Run:  belief-response-decode --response /tmp/response.bin
//
// Public Domain (The Unlicense).
#include <fstream>
#include <iostream>
#include <iterator>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "belief_wire.hpp"

namespace bw = tlab::bwire;

namespace {
[[nodiscard]] std::optional<std::string_view> opt(std::span<const std::string_view> args,
                                                  std::string_view name) {
    for (size_t i = 1; i + 1 < args.size(); ++i)
        if (args[i] == name) return args[i + 1];
    return std::nullopt;
}
}  // namespace

int main(int argc, char** argv) {
    std::vector<std::string_view> args(argv, argv + argc);
    auto resp_p = opt(args, "--response");
    if (!resp_p) { std::cerr << "usage: belief-response-decode --response <path>\n"; return 2; }

    std::ifstream f(std::string(*resp_p), std::ios::binary);
    if (!f) { std::cerr << "belief-response-decode: FATAL: cannot open " << *resp_p << "\n"; return 1; }
    std::vector<unsigned char> frame((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());

    // decode_response fails LOUDLY on a malformed frame (ADR-0002) — let it propagate as a nonzero exit.
    std::vector<bw::ResponseFields> preds = bw::decode_response(frame);

    std::cout << "belief-response-decode: B=" << preds.size()
              << " n_actions=" << (preds.empty() ? 0 : preds[0].logits.size()) << "\n";
    for (size_t r = 0; r < preds.size(); ++r) {
        const bw::ResponseFields& p = preds[r];
        double lmax = p.logits.empty() ? 0.0 : p.logits[0];
        for (float l : p.logits) lmax = std::max<double>(lmax, l);
        std::cout << "  row " << r << ": value=" << p.value
                  << "  logits=" << p.logits.size() << "  max_logit=" << lmax << "\n";
    }
    return 0;
}
