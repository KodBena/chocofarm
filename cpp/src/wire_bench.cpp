// cpp/src/wire_bench.cpp
// Purpose: the OVER-THE-WIRE SYNCHRONOUS benchmark fixture (NOT the runner) — runs a batch of
//   independent Gumbel-AZ decisions through SerialRuntime where the leaf is a REMOTE ZmqNetClient
//   (a blocking REQ to the Python Shape-B InferenceServer), times the batch, and reports the wire-sync
//   throughput (decisions/s) + the per-leaf round-trip cost. This is the "over-the-wire synchronous"
//   axis of docs/design/cpp-search-runtime.md's §6-Q5 benchmark: one in-flight leaf at a time, no
//   batching benefit, so it measures the wire RTT + server-forward cost the multiplexed wire-parallel
//   config (the fiber + DEALER pool — the next chunk) exists to hide.
//
//   The server is spun separately (the Python driver cpp/parity/wire_bench.py injects a dimension-matched
//   ValueMLP via StaticParamsSource and runs serve_forever); this binary only connects a ZmqNetClient to
//   --endpoint and drives the search. PoolRuntime is deliberately NOT used here: the blocking REQ client
//   is not thread-safe and a parallel wire config needs per-worker clients (or the fiber multiplexer) —
//   that is the wire-PARALLEL chunk, not this one.
//
//   ADR-0012 P9: the imperative shell. argv decoded once into typed views; ZmqNetClient::create + the
//   per-leaf predict failures arrive as typed std::expected and are reported loudly (a server-down is a
//   typed timeout, not a hang — design §5), never a silent fallback.
//
//   Protocol:  wire-bench --instance <p> --faces <p> --endpoint <tcp://host:port>
//                  [--tasks N --n-sims N --m N --max-depth N --c-outcome N --lam f --timeout-ms N]
//   Output:    a config line, then "RESULT: PASS wire_sync_dps=<n> leaf_requests=<n> wall=<s>
//              us_per_leaf=<n>" + exit 0, or a loud Error + nonzero (a failed connect / a leaf RPC
//              timeout — the search aborts loudly, ADR-0002).
//
// Public Domain (The Unlicense).
#include <chrono>
#include <cstdint>
#include <iostream>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "chocofarm/env.hpp"
#include "chocofarm/features.hpp"
#include "chocofarm/gumbel.hpp"
#include "chocofarm/instance.hpp"
#include "chocofarm/search_runtime.hpp"
#include "chocofarm/zmq_net_client.hpp"

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
    std::optional<std::string_view> endpoint = opt(args, "--endpoint");
    if (!instance || !faces || !endpoint) {
        std::cerr << "usage: wire-bench --instance <p> --faces <p> --endpoint <tcp://host:port> "
                     "[--tasks N --n-sims N --m N --max-depth N --c-outcome N --lam f --timeout-ms N]\n";
        return 2;
    }

    const int n_tasks = opt(args, "--tasks") ? to_int(*opt(args, "--tasks")) : 8;
    const int timeout_ms = opt(args, "--timeout-ms") ? to_int(*opt(args, "--timeout-ms")) : 10000;
    const double lam = opt(args, "--lam") ? to_double(*opt(args, "--lam")) : 0.1;
    chocofarm::GumbelConfig cfg;
    cfg.n_sims = 12;   // a modest default: each leaf is a wire RPC, so keep the RPC count bounded
    cfg.max_depth = 8;
    if (auto v = opt(args, "--m")) cfg.m = to_int(*v);
    if (auto v = opt(args, "--n-sims")) cfg.n_sims = to_int(*v);
    if (auto v = opt(args, "--max-depth")) cfg.max_depth = to_int(*v);
    if (auto v = opt(args, "--c-outcome")) cfg.c_outcome = to_int(*v);

    auto inst = chocofarm::load_instance(*instance, *faces);
    if (!inst) {
        std::cerr << "wire-bench: FATAL: " << inst.error().message << "\n";
        return 1;
    }
    chocofarm::Environment env(*inst);

    // connect the remote leaf evaluator (blocking REQ to the running server).
    auto client = chocofarm::ZmqNetClient::create(std::string(*endpoint), timeout_ms);
    if (!client) {
        std::cerr << "wire-bench: FATAL: ZmqNetClient::create failed: " << client.error().message << "\n";
        return 1;
    }

    chocofarm::Loc root_loc{env.entry_point()};
    chocofarm::Belief root_bw = env.full_belief();   // the seam's belief construction entry
    chocofarm::CollectedSet root_collected;
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
              << " endpoint=" << *endpoint << " n_slots=" << chocofarm::n_action_slots(env) << "\n";

    chocofarm::SerialRuntime serial(*client);
    auto t0 = std::chrono::steady_clock::now();
    auto result = serial.run(env, tasks);
    auto t1 = std::chrono::steady_clock::now();
    if (!result) {
        // a leaf RPC failed (server-down / timeout / malformed reply) — loud, not a silent fallback.
        std::cerr << "wire-bench: FATAL: SerialRuntime.run returned an Error (a leaf RPC failed): "
                  << result.error().message << "\n";
        return 1;
    }

    long leaf_total = 0;
    for (const chocofarm::Decision& d : *result) leaf_total += d.leaf_requests;
    const double wall = secs(t0, t1);
    const double dps = static_cast<double>(n_tasks) / wall;
    const double us_per_leaf = leaf_total > 0 ? (wall * 1e6 / static_cast<double>(leaf_total)) : 0.0;
    std::cout.precision(5);
    std::cout << "RESULT: PASS wire_sync_dps=" << dps << " leaf_requests=" << leaf_total
              << " wall=" << wall << " us_per_leaf=" << us_per_leaf << "\n";
    return 0;
}
