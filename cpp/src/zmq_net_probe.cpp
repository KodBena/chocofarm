// cpp/src/zmq_net_probe.cpp
// Purpose: a tiny PARITY tool (NOT the runner) — constructs a ZmqNetClient (the REMOTE NetEvaluator,
//   docs/design/zmq-inference-service.md §1), RPCs the running Python InferenceServer with the feature
//   vectors on stdin, and prints the returned (de-standardized value, raw logits). The round-trip test
//   (tests/test_zmq_net_cpp.py) spins the Python InferenceServer in-process with StaticParamsSource (NO
//   redis), feeds N≥100 random float32 feature vectors through BOTH this probe and the local C++
//   NetForward on the SAME weights, and asserts the ADR-0012 P6 behavioral bar (max|Δ| < 1e-4) — proving
//   the full path (C++ encode → server forward_core → C++ decode) is faithful end-to-end across
//   languages, residual ON and OFF.
//
//   It is a SEPARATE executable from the runner (P3, one-owner): the runner's job is the wire +
//   RandomPolicy episode loop, this tool's job is the inference-client round-trip fixture. The
//   ZmqNetClient encodes/decodes via the SHARED wire codec (chocofarm/inference_wire.hpp), derived from
//   the wire_spec SSOT (no second hand-written frame — ADR-0012 P1/P7).
//
//   ADR-0012 P9: the imperative shell. argv is decoded once into typed views; `opt` returns a
//   std::optional<std::string_view>; the boundary failures (a failed connect, a recv timeout / server-
//   down, a malformed reply) arrive as typed std::expected and are reported loudly, never thrown.
//
//   Protocol:
//     argv:  --endpoint <tcp://host:port>  [--timeout-ms <int>]  [--probe-down]
//            --probe-down: do ONE RPC and EXPECT it to FAIL (the loud-failure path — a server-down OR a
//            malformed/wrong-length reply must return a typed Error, not hang, not a silent fallback);
//            prints `DOWN_OK <message>` and exits 0 iff predict() errored, else exits 5. (Used by the
//            test's server-down AND wrong-length-reply demonstrations — both take this same error arm.)
//     stdin: one feature vector per line, `in_dim` space-separated float values
//     stdout: per input line, `value logit0 logit1 ...` (value-only nets print just `value`), full
//             float precision (so the float32-equivalence comparison is honest).
//
// Public Domain (The Unlicense).
#include <cstdlib>
#include <iostream>
#include <optional>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include "chocofarm/zmq_net_client.hpp"

namespace {
[[nodiscard]] std::optional<std::string_view> opt(std::span<const std::string_view> args,
                                                  std::string_view name) {
    for (size_t i = 1; i + 1 < args.size(); ++i)
        if (args[i] == name) return args[i + 1];
    return std::nullopt;
}
[[nodiscard]] bool has_flag(std::span<const std::string_view> args, std::string_view name) {
    for (size_t i = 1; i < args.size(); ++i)
        if (args[i] == name) return true;
    return false;
}
}  // namespace

int main(int argc, char** argv) {
    std::vector<std::string_view> args(argv, argv + argc);
    std::optional<std::string_view> endpoint = opt(args, "--endpoint");
    if (!endpoint) {
        std::cerr << "usage: zmq-net-probe --endpoint <tcp://host:port> [--timeout-ms <int>] "
                     "[--probe-down]  (feature vectors on stdin)\n";
        return 2;
    }
    int timeout_ms = 5000;
    if (auto t = opt(args, "--timeout-ms")) timeout_ms = std::atoi(std::string(*t).c_str());
    const bool probe_down = has_flag(args, "--probe-down");

    auto client = chocofarm::ZmqNetClient::create(std::string(*endpoint), timeout_ms);
    if (!client) { std::cerr << "zmq-net-probe: " << client.error().message << "\n"; return 1; }

    // ---- server-down demonstration: ONE RPC that MUST return an Error (a loud non-hang) ----
    if (probe_down) {
        std::vector<float> x(8, 1.0f);  // a small dummy vector; the point is the RPC fails, not its dims
        auto pred = client->predict(x);
        if (pred) {
            std::cerr << "zmq-net-probe: --probe-down but predict SUCCEEDED (expected a typed failure)\n";
            return 5;
        }
        std::cout << "DOWN_OK " << pred.error().message << "\n";
        return 0;
    }

    std::cout.precision(9);  // float32 has ~7 sig digits; 9 round-trips the value exactly

    std::string line;
    while (std::getline(std::cin, line)) {
        if (line.empty()) continue;
        std::istringstream iss(line);
        std::vector<float> X;
        float v;
        while (iss >> v) X.push_back(v);
        if (X.empty()) continue;
        auto pred = client->predict(X);   // std::vector<float> binds to std::span
        if (!pred) { std::cerr << "zmq-net-probe: " << pred.error().message << "\n"; return 3; }
        std::cout << pred->value;
        for (float l : pred->logits) std::cout << ' ' << l;
        std::cout << "\n";
    }
    return 0;
}
