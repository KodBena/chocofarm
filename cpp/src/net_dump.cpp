// cpp/src/net_dump.cpp
// Purpose: a tiny PARITY tool (NOT the runner) — reads a published net off redis via the manifest
//   (the SAME weight-read seam the runner exercises), builds a C++ NetForward, and for each feature
//   vector on stdin prints the de-standardized value + the policy logits. The parity harness
//   (cpp/parity/net_parity.py) feeds N≥1000 feature vectors through BOTH this tool and the Python
//   forward_core on the SAME weights and asserts the ADR-0012 P6 behavioral bar (max|Δ| < 1e-4).
//
//   It is a SEPARATE executable from the runner (P3, one-owner): the runner's job is the wire +
//   RandomPolicy episode loop, this tool's job is the NetForward parity fixture. It uses the SAME
//   transport.read_weights seam (no hardcoded offsets — P1) so the forward runs on the real
//   manifest-bound weights, residual ON or OFF per what the manifest carries.
//
//   ADR-0012 P9: the imperative shell. argv is decoded once into typed views; `opt` returns a
//   std::optional<std::string_view>; the boundary failures (a dead redis, a missing payload, a
//   malformed manifest) arrive as typed std::expected and are reported loudly, never thrown.
//
//   Protocol:
//     argv:  --run <run> --phase <gen|eval> --version <int>
//     stdin: one feature vector per line, `in_dim` space-separated float values
//     stdout: per input line, `value logit0 logit1 ...` (value-only nets print just `value`),
//             full float precision (so the float32-equivalence comparison is honest).
//   The first stdout line is a header: `# in_dim=<d> n_actions=<a> residual=<0|1>` so the harness
//   can sanity-check the derived dims match the Python net.
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

#include "chocofarm/net.hpp"
#include "chocofarm/transport.hpp"

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
    std::optional<std::string_view> run = opt(args, "--run");
    std::optional<std::string_view> phase = opt(args, "--phase");
    std::optional<std::string_view> version_s = opt(args, "--version");
    if (!run || !phase || !version_s) {
        std::cerr << "usage: net-dump --run <run> --phase <gen|eval> --version <int>  "
                     "(feature vectors on stdin)\n";
        return 2;
    }
    int version = std::atoi(std::string(*version_s).c_str());

    auto cli = chocofarm::RedisClient::create();  // the CHOCO_TRANSPORT_REDIS_* contract (6380)
    if (!cli) { std::cerr << "net-dump: " << cli.error().message << "\n"; return 1; }
    // read the published net via the manifest (no hardcoded offsets — P1), then build the forward.
    auto payload = cli->read_weights(*run, *phase, version);
    if (!payload) { std::cerr << "net-dump: " << payload.error().message << "\n"; return 1; }
    auto net = chocofarm::NetForward::create(*payload);
    if (!net) { std::cerr << "net-dump: " << net.error().message << "\n"; return 1; }

    std::cout << "# in_dim=" << net->in_dim() << " n_actions=" << net->n_actions()
              << " residual=" << (net->residual() ? 1 : 0) << "\n";
    std::cout.precision(9);  // float32 has ~7 sig digits; 9 round-trips the value exactly

    std::string line;
    while (std::getline(std::cin, line)) {
        if (line.empty()) continue;
        std::istringstream iss(line);
        std::vector<float> X;
        float x;
        while (iss >> x) X.push_back(x);
        if (static_cast<int>(X.size()) != net->in_dim()) {
            std::cerr << "net-dump: input row has " << X.size() << " values, expected in_dim "
                      << net->in_dim() << "\n";
            return 3;
        }
        // predict() now returns the NetEvaluator port's std::expected (shared with the remote
        // ZmqNetClient). NetForward's LOCAL compute is total, so this never takes the error arm — but
        // P9 [[nodiscard]] forces us to handle it, so a future fallible swap can't silently drop it.
        auto pred = net->predict(X);  // std::vector<float> binds to std::span
        if (!pred) { std::cerr << "net-dump: " << pred.error().message << "\n"; return 4; }
        std::cout << pred->value;
        for (float l : pred->logits) std::cout << ' ' << l;
        std::cout << "\n";
    }
    return 0;
}
