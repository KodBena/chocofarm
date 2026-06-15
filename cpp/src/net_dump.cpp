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
//   Protocol:
//     argv:  --run <run> --phase <gen|eval> --version <int>
//     stdin: one feature vector per line, `in_dim` space-separated float values
//     stdout: per input line, `value logit0 logit1 ...` (value-only nets print just `value`),
//             full float precision (so the float32-equivalence comparison is honest).
//   The first stdout line is a header: `# in_dim=<d> n_actions=<a> residual=<0|1>` so the harness
//   can sanity-check the derived dims match the Python net.
//
// Public Domain (The Unlicense).
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "chocofarm/net.hpp"
#include "chocofarm/transport.hpp"

static const char* opt(int argc, char** argv, const char* name) {
    for (int i = 1; i + 1 < argc; ++i)
        if (std::strcmp(argv[i], name) == 0) return argv[i + 1];
    return nullptr;
}

int main(int argc, char** argv) {
    const char* run = opt(argc, argv, "--run");
    const char* phase = opt(argc, argv, "--phase");
    const char* version_s = opt(argc, argv, "--version");
    if (!run || !phase || !version_s) {
        std::cerr << "usage: net-dump --run <run> --phase <gen|eval> --version <int>  "
                     "(feature vectors on stdin)\n";
        return 2;
    }
    int version = std::atoi(version_s);

    try {
        chocofarm::RedisClient cli;  // the CHOCO_TRANSPORT_REDIS_* contract (6380), fail-loud on connect
        // read the published net via the manifest (no hardcoded offsets — P1), then build the forward.
        chocofarm::WeightPayload payload = cli.read_weights(run, phase, version);
        chocofarm::NetForward net(payload);

        std::cout << "# in_dim=" << net.in_dim() << " n_actions=" << net.n_actions()
                  << " residual=" << (net.residual() ? 1 : 0) << "\n";
        std::cout.precision(9);  // float32 has ~7 sig digits; 9 round-trips the value exactly

        std::string line;
        while (std::getline(std::cin, line)) {
            if (line.empty()) continue;
            std::istringstream iss(line);
            std::vector<float> X;
            float x;
            while (iss >> x) X.push_back(x);
            if (static_cast<int>(X.size()) != net.in_dim()) {
                std::cerr << "net-dump: input row has " << X.size() << " values, expected in_dim "
                          << net.in_dim() << "\n";
                return 3;
            }
            chocofarm::NetPrediction pred = net.predict(X);
            std::cout << pred.value;
            for (float l : pred.logits) std::cout << ' ' << l;
            std::cout << "\n";
        }
    } catch (const std::exception& e) {
        std::cerr << "net-dump: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
