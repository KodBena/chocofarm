// cpp/src/local_mlp_bench.cpp
// Purpose: the C++-NATIVE LOCAL MLP benchmark fixture (NOT the runner) — the THIRD axis of the leaf-eval
//   comparison. It runs the SAME SerialRuntime batch as the over-the-wire SYNCHRONOUS bench
//   (wire_bench.cpp), but with a LOCAL NetForward leaf evaluator (the in-process C++ MLP forward) in place
//   of the remote ZmqNetClient. No socket, no server, no batching: every leaf is a direct local matmul.
//   It is apples-to-apples with the wire-sync axis (both drive ONE in-flight leaf at a time through
//   SerialRuntime), so the throughput delta is exactly "local forward" vs "wire RTT + server forward" —
//   the cost the over-the-wire batched configs (wire-parallel / wire-pool) exist to amortize.
//
//   FAIRNESS (ADR-0012 P6 / P1): the net is read off the SAME manifest+blob weight-read seam
//   (RedisClient::read_weights -> NetForward::create) the C++ runner uses, on weights the harness
//   (cpp/parity/wire_bench.py) publishes to redis via the SAME pack_net the Python InferenceServer is
//   seeded from. So the local NetForward and the wire server compute the SAME net (float32-equivalent
//   < 1e-4, NOT byte-identity — the only gap is the server's float64 weights vs NetForward's float32
//   cast). The benchmark therefore compares transport regimes, never two different nets.
//
//   ADR-0012 P9: the imperative shell. argv is decoded once into typed views; RedisClient::create /
//   read_weights / NetForward::create failures arrive as typed std::expected and are reported loudly (a
//   missing weight payload is a typed Error, not a silent default — ADR-0002), as is a published-net
//   dimension mismatch (caught early here, not as a deep assert inside predict()).
//
//   Protocol:  local-mlp-bench --instance <p> --faces <p> --run R --phase P --version V
//                  [--tasks N --n-sims N --m N --max-depth N --c-outcome N --lam f]
//   Output:    a config line, then "RESULT: PASS local_mlp_dps=<n> leaf_requests=<n> wall=<s>
//              us_per_leaf=<n>" + exit 0, or a loud Error + nonzero (a dead redis / a missing weight
//              payload / a dimension mismatch — aborts loudly, ADR-0002).
//
// Public Domain (The Unlicense).
#include <chrono>
#include <cstdint>
#include <iostream>
#include <optional>
#include <set>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "chocofarm/env.hpp"
#include "chocofarm/features.hpp"
#include "chocofarm/gumbel.hpp"
#include "chocofarm/instance.hpp"
#include "chocofarm/net.hpp"
#include "chocofarm/search_runtime.hpp"
#include "chocofarm/transport.hpp"

namespace {
[[nodiscard]] std::optional<std::string_view> opt(std::span<const std::string_view> args,
                                                  std::string_view name) {
    for (size_t i = 1; i + 1 < args.size(); ++i)
        if (args[i] == name) return args[i + 1];
    return std::nullopt;
}
[[nodiscard]] int to_int(std::string_view s) { return std::atoi(std::string(s).c_str()); }
[[nodiscard]] double to_double(std::string_view s) { return std::atof(std::string(s).c_str()); }
[[nodiscard]] double secs(std::chrono::steady_clock::time_point a,
                          std::chrono::steady_clock::time_point b) {
    return std::chrono::duration<double>(b - a).count();
}
}  // namespace

int main(int argc, char** argv) {
    std::vector<std::string_view> args(argv, argv + argc);
    std::optional<std::string_view> instance = opt(args, "--instance");
    std::optional<std::string_view> faces = opt(args, "--faces");
    std::optional<std::string_view> run = opt(args, "--run");
    std::optional<std::string_view> phase = opt(args, "--phase");
    std::optional<std::string_view> version = opt(args, "--version");
    if (!instance || !faces || !run || !phase || !version) {
        std::cerr << "usage: local-mlp-bench --instance <p> --faces <p> --run R --phase P --version V "
                     "[--tasks N --n-sims N --m N --max-depth N --c-outcome N --lam f]\n";
        return 2;
    }

    const int n_tasks = opt(args, "--tasks") ? to_int(*opt(args, "--tasks")) : 8;
    const int ver = to_int(*version);
    const double lam = opt(args, "--lam") ? to_double(*opt(args, "--lam")) : 0.1;
    chocofarm::GumbelConfig cfg;
    cfg.n_sims = 12;   // match wire-bench's default so the axes are compared at the SAME search budget
    cfg.max_depth = 8;
    if (auto v = opt(args, "--m")) cfg.m = to_int(*v);
    if (auto v = opt(args, "--n-sims")) cfg.n_sims = to_int(*v);
    if (auto v = opt(args, "--max-depth")) cfg.max_depth = to_int(*v);
    if (auto v = opt(args, "--c-outcome")) cfg.c_outcome = to_int(*v);

    auto inst = chocofarm::load_instance(*instance, *faces);
    if (!inst) {
        std::cerr << "local-mlp-bench: FATAL: " << inst.error().message << "\n";
        return 1;
    }
    chocofarm::Environment env(*inst);
    chocofarm::FeatureBuilder fb(env);

    // the LOCAL leaf evaluator: read the published weights off the SAME seam the runner uses, build the
    // in-process NetForward. A dead redis / missing payload / malformed manifest is a typed Error (loud).
    auto redis = chocofarm::RedisClient::create();
    if (!redis) {
        std::cerr << "local-mlp-bench: FATAL: RedisClient::create failed: " << redis.error().message << "\n";
        return 1;
    }
    auto wp = redis->read_weights(*run, *phase, ver);
    if (!wp) {
        std::cerr << "local-mlp-bench: FATAL: read_weights failed: " << wp.error().message << "\n";
        return 1;
    }
    auto net = chocofarm::NetForward::create(*wp);
    if (!net) {
        std::cerr << "local-mlp-bench: FATAL: NetForward::create failed: " << net.error().message << "\n";
        return 1;
    }
    // fail loud + early on a dimension mismatch (else it surfaces as a deep assert inside predict()).
    if (net->in_dim() != fb.dim()) {
        std::cerr << "local-mlp-bench: FATAL: published net in_dim=" << net->in_dim()
                  << " != env feature_dim=" << fb.dim() << " (dimension mismatch — wrong weights?)\n";
        return 1;
    }

    chocofarm::Loc root_loc{env.entry_point()};
    chocofarm::Belief root_bw = env.full_belief();   // the seam's belief construction entry
    std::set<int> root_collected;
    std::vector<chocofarm::SearchTask> tasks;
    tasks.reserve(static_cast<size_t>(n_tasks));
    for (int i = 0; i < n_tasks; ++i) {
        chocofarm::SearchTask t;
        t.loc = root_loc;
        t.bw = root_bw;
        t.collected = root_collected;
        t.lam = lam;
        t.seed = static_cast<std::uint64_t>(i) + 1;
        t.cfg = cfg;
        tasks.push_back(std::move(t));
    }

    std::cout << "config: tasks=" << n_tasks << " m=" << cfg.m << " n_sims=" << cfg.n_sims
              << " max_depth=" << cfg.max_depth << " c_outcome=" << cfg.c_outcome << " lam=" << lam
              << " run=" << *run << " phase=" << *phase << " version=" << ver
              << " in_dim=" << net->in_dim() << " n_actions=" << net->n_actions()
              << " n_slots=" << chocofarm::n_action_slots(env) << "\n";

    chocofarm::SerialRuntime serial(*net);
    auto t0 = std::chrono::steady_clock::now();
    auto result = serial.run(env, tasks);
    auto t1 = std::chrono::steady_clock::now();
    if (!result) {
        // a local forward cannot fail; an Error here means a search-runtime boundary failure — loud.
        std::cerr << "local-mlp-bench: FATAL: SerialRuntime.run returned an Error: "
                  << result.error().message << "\n";
        return 1;
    }

    long leaf_total = 0;
    for (const chocofarm::Decision& d : *result) leaf_total += d.leaf_requests;
    const double wall = secs(t0, t1);
    const double dps = static_cast<double>(n_tasks) / wall;
    const double us_per_leaf = leaf_total > 0 ? (wall * 1e6 / static_cast<double>(leaf_total)) : 0.0;
    std::cout.precision(5);
    std::cout << "RESULT: PASS local_mlp_dps=" << dps << " leaf_requests=" << leaf_total
              << " wall=" << wall << " us_per_leaf=" << us_per_leaf << "\n";
    return 0;
}
