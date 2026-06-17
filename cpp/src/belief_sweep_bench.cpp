// cpp/src/belief_sweep_bench.cpp
// Purpose: an ISOLATED microbenchmark of the belief sweep (chocofarm::belief_features) — the
//   O(nb·(N+nD)) popcount-marg + observe-cover loop over the world-set that the K=16 perf profile
//   measured at ~81% of the single client thread. NO search, NO wire, NO cache, NO geometry/assemble:
//   just the sweep, timed across belief sizes nb (subsets of env.worlds()), so an optimizer (pospopcount
//   / inline detector masks / decision-diagram) iterates on the REAL function with a clean signal — the
//   per-WORLD cost is the slope (ns/world), the fixed per-call cost (alloc + the O(N+nD) tail) is the
//   intercept (read it off the smallest nb). Separate executable (ADR-0012 P3, one-owner). No redis/net.
//
//   Protocol:  belief-sweep-bench --instance <p> --faces <p> [--budget-s 0.3]
//   Output:    the env dims, then a table over nb: ns/call, ns/world, Mworlds/s, Mcalls/s.
//
// Public Domain (The Unlicense).
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "chocofarm/env.hpp"
#include "chocofarm/feature_compute.hpp"
#include "chocofarm/instance.hpp"

namespace {
[[nodiscard]] std::optional<std::string_view> opt(std::span<const std::string_view> args,
                                                  std::string_view name) {
    for (size_t i = 1; i + 1 < args.size(); ++i)
        if (args[i] == name) return args[i + 1];
    return std::nullopt;
}
volatile double g_sink = 0.0;  // defeat dead-code elimination of the timed call
}  // namespace

int main(int argc, char** argv) {
    std::vector<std::string_view> args(argv, argv + argc);
    std::optional<std::string_view> instance = opt(args, "--instance");
    std::optional<std::string_view> faces = opt(args, "--faces");
    if (!instance || !faces) {
        std::cerr << "usage: belief-sweep-bench --instance <p> --faces <p> [--budget-s 0.3]\n";
        return 2;
    }
    const double budget = opt(args, "--budget-s")
        ? std::atof(std::string(*opt(args, "--budget-s")).c_str()) : 0.3;

    auto inst = chocofarm::load_instance(*instance, *faces);
    if (!inst) { std::cerr << "belief-sweep-bench: FATAL: " << inst.error().message << "\n"; return 1; }
    chocofarm::Environment env(*inst);
    const int N = env.N();
    const int nD = env.n_detectors();
    const std::vector<uint32_t> bw_full = env.worlds();
    const size_t nworlds = bw_full.size();
    const double log_nworlds = std::log(static_cast<double>(nworlds));

    std::cout << "belief-sweep-bench: N=" << N << " nD=" << nD << " |worlds|=" << nworlds
              << " budget=" << budget << "s/point  (timing chocofarm::belief_features in isolation)\n";
    std::cout << std::setw(8) << "nb" << std::setw(12) << "ns/call"
              << std::setw(12) << "ns/world" << std::setw(13) << "Mworlds/s"
              << std::setw(12) << "Mcalls/s" << "\n";

    // belief sizes: powers of two up to |worlds|, plus |worlds| itself.
    std::vector<size_t> sizes;
    for (size_t s = 1; s < nworlds; s *= 2) sizes.push_back(s);
    sizes.push_back(nworlds);

    for (size_t nb : sizes) {
        const std::span<const uint32_t> bw(bw_full.data(), nb);
        using clk = std::chrono::steady_clock;
        long calls = 0;
        double sink = 0.0;
        const auto t0 = clk::now();
        double el = 0.0;
        do {
            for (int i = 0; i < 16; ++i) {
                const chocofarm::BeliefFeatures bf = chocofarm::belief_features(env, bw, N, nD, log_nworlds);
                sink += bf.marg_sum + bf.sharpness;
            }
            calls += 16;
            el = std::chrono::duration<double>(clk::now() - t0).count();
        } while (el < budget);
        g_sink += sink;
        const double ns_call = el * 1e9 / static_cast<double>(calls);
        const double ns_world = ns_call / static_cast<double>(nb);
        const double mworlds_s = static_cast<double>(nb) / ns_call * 1e3;   // nb worlds per ns_call ns
        const double mcalls_s = 1e3 / ns_call;
        std::cout << std::fixed << std::setprecision(2)
                  << std::setw(8) << nb
                  << std::setw(12) << ns_call
                  << std::setw(12) << ns_world
                  << std::setw(13) << mworlds_s
                  << std::setw(12) << mcalls_s << "\n";
    }
    return 0;
}
