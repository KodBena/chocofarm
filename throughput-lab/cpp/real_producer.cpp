// throughput-lab/cpp/real_producer.cpp
// Purpose: the REAL-generator load driver (NON-FIBER baseline) — N producer threads, each running real
//   Gumbel-AZ decisions back-to-back through its OWN tlab::Boundary (a per-thread DEALER), each leaf a
//   B=1 blocking round-trip to the live server. With N threads each holding one leaf in flight, the
//   server gathers up to N concurrent leaves per forward (batch ~= N). This is the NON-FIBER data point
//   the fiber multiplexer (K leaves/thread -> batch ~= N*K) is measured against: the open question is
//   whether the fiber model helps or hurts throughput vs this baseline (the maintainer's investigation,
//   neither prior trusted). All rates MEASURED (leaves/wall, decisions/wall), never assumed (ADR-0009).
//
//   Built only under -DTLAB_REAL_GENERATOR=ON (links chocofarm_core). The synthetic tlab-producer stays
//   a standalone clean-room binary; this is the additive real-generator sibling (ADR-0012 compose).
// Public Domain (The Unlicense).
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include "chocofarm/env.hpp"
#include "chocofarm/gumbel.hpp"
#include "chocofarm/instance.hpp"
#include "chocofarm/search_runtime.hpp"

#include "boundary.hpp"
#include "boundary_net_evaluator.hpp"

namespace {
using SteadyClock = std::chrono::steady_clock;
[[nodiscard]] double secs_since(SteadyClock::time_point t0) {
    return std::chrono::duration<double>(SteadyClock::now() - t0).count();
}
[[nodiscard]] std::optional<std::string_view> opt(std::span<const std::string_view> a, std::string_view k) {
    for (size_t i = 1; i + 1 < a.size(); ++i)
        if (a[i] == k) return a[i + 1];
    return std::nullopt;
}

struct ThreadStat {
    std::uint64_t decisions = 0;
    std::uint64_t leaves = 0;   // predict() round-trips issued (the leaf-eval count)
    bool failed = false;
    std::string err;
};

// One producer thread: build its own boundary + bridge + SerialRuntime, then run real decisions from the
// root state (varying the seed so trees differ) until the wall deadline. Each decision's leaf_requests is
// the count of B=1 round-trips it drove through the boundary.
void run_thread(int idx, const chocofarm::Environment& env, const std::string& endpoint,
                const chocofarm::GumbelConfig& cfg, double run_seconds, int in_dim, ThreadStat& out) {
    tlab::BoundaryConfig bcfg;
    bcfg.endpoint = endpoint;
    bcfg.recv_timeout_ms = 10000;     // generous: the server may be busy gathering other threads' leaves
    bcfg.n_producer_threads = 1;
    bcfg.rows = 1;
    bcfg.in_dim = in_dim;
    auto b = tlab::make_boundary(tlab::BoundaryTopology::PerThread, bcfg);
    if (!b) { out.failed = true; out.err = "boundary: " + b.error().message; return; }
    std::unique_ptr<tlab::Boundary> boundary = std::move(*b);
    tlab::BoundaryNetEvaluator bridge(*boundary);
    chocofarm::SerialRuntime serial(bridge);

    const chocofarm::Loc loc{env.entry_point()};
    const chocofarm::Belief bw = env.full_belief();
    const chocofarm::CollectedSet coll;
    const auto start = SteadyClock::now();
    std::uint64_t seed = static_cast<std::uint64_t>(idx) * 1'000'003ull + 1ull;
    while (secs_since(start) < run_seconds) {
        chocofarm::SearchTask t;
        t.loc = loc; t.bw = bw; t.collected = coll; t.lam = 0.1; t.seed = seed++; t.cfg = cfg;
        std::vector<chocofarm::SearchTask> tasks{t};
        auto dec = serial.run(env, std::span<const chocofarm::SearchTask>(tasks));
        if (!dec) { out.failed = true; out.err = "runtime: " + dec.error().message; return; }
        out.decisions += 1;
        out.leaves += static_cast<std::uint64_t>((*dec)[0].leaf_requests);
    }
}
}  // namespace

int main(int argc, char** argv) {
    std::vector<std::string_view> args(argv, argv + argc);
    auto inst_p = opt(args, "--instance"), faces_p = opt(args, "--faces"), ep = opt(args, "--endpoint");
    if (!inst_p || !faces_p || !ep) {
        std::cerr << "usage: tlab-real-producer --instance <p> --faces <p> --endpoint <ipc://...> "
                     "[--threads N --seconds S --n-sims K --m M --in-dim D]\n";
        return 2;
    }
    const int threads = opt(args, "--threads") ? std::atoi(std::string(*opt(args, "--threads")).c_str()) : 3;
    const double seconds = opt(args, "--seconds") ? std::atof(std::string(*opt(args, "--seconds")).c_str()) : 5.0;
    const int in_dim = opt(args, "--in-dim") ? std::atoi(std::string(*opt(args, "--in-dim")).c_str()) : 241;
    chocofarm::GumbelConfig cfg;
    if (auto v = opt(args, "--n-sims")) cfg.n_sims = std::atoi(std::string(*v).c_str());
    if (auto v = opt(args, "--m")) cfg.m = std::atoi(std::string(*v).c_str());

    auto inst = chocofarm::load_instance(*inst_p, *faces_p);
    if (!inst) { std::cerr << "tlab-real-producer: FATAL: " << inst.error().message << "\n"; return 1; }
    chocofarm::Environment env(*inst);

    std::cout << "tlab-real-producer: generator=real(non-fiber) threads=" << threads
              << " seconds=" << seconds << " n_sims=" << cfg.n_sims << " m=" << cfg.m
              << " n_slots=" << chocofarm::n_action_slots(env) << " endpoint=" << *ep << "\n";

    std::vector<ThreadStat> stats(static_cast<size_t>(threads));
    std::vector<std::thread> pool;
    const std::string endpoint(*ep);
    const auto t0 = SteadyClock::now();
    for (int i = 0; i < threads; ++i)
        pool.emplace_back([&, i] { run_thread(i, env, endpoint, cfg, seconds, in_dim, stats[static_cast<size_t>(i)]); });
    for (auto& th : pool) th.join();
    const double wall = secs_since(t0);

    std::uint64_t dec = 0, leaves = 0; bool any_fail = false;
    for (const auto& s : stats) {
        dec += s.decisions; leaves += s.leaves;
        if (s.failed) { any_fail = true; std::cerr << "  thread failed: " << s.err << "\n"; }
    }
    const double dps = wall > 0 ? static_cast<double>(dec) / wall : 0.0;
    const double lps = wall > 0 ? static_cast<double>(leaves) / wall : 0.0;
    std::cout << "REAL-AGG threads=" << threads << " wall_s=" << wall
              << " decisions=" << dec << " leaves=" << leaves
              << " decisions_per_sec=" << dps << " leaves_per_sec=" << lps
              << " any_fail=" << (any_fail ? 1 : 0) << "\n";
    return any_fail ? 1 : 0;
}
