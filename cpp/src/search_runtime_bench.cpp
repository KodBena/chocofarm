// cpp/src/search_runtime_bench.cpp
// Purpose: a throughput benchmark for the SearchRuntime seam (NOT the runner) — runs a batch of
//   independent Gumbel-AZ decisions through SerialRuntime and PoolRuntime over a LOCAL, in-process net,
//   times each, and reports decisions/s + the parallel speedup, AFTER asserting the two produce
//   BIT-IDENTICAL per-task results (same executed action AND leaf-request count) — the exact-parallelism
//   proof that independent deterministic trees give (docs/design/cpp-search-runtime.md: the C++-native
//   MLP config; the "parallel tree descent + backprop" abstraction is valuable independent of where the
//   net runs, and for a local net it needs no fibers). The net is a DETERMINISTIC, STATELESS in-process
//   evaluator (no redis, no weights, no RNG) whose per-leaf cost is cheap — representative of the
//   tiny-MLP regime the host actually targets, where the tree descent + backprop, not the forward,
//   dominates, so the measured speedup is the speedup of the parallel SEARCH. (Swapping in NetForward /
//   ZmqNetClient — the real-MLP and over-the-wire configs — is a construction-site change; this binary
//   measures the local axis.)
//
//   ADR-0012 P9: the imperative shell. argv decoded once into typed views; load_instance returns a
//   typed std::expected reported loudly.
//
//   Protocol:  search-runtime-bench --instance <p> --faces <p>
//                  [--tasks N --n-sims N --m N --max-depth N --c-outcome N --lam f --workers N --reps N]
//   Output:    a header line of the config, then per-rep timings, then a summary:
//              "RESULT: PASS speedup=<x> serial_dps=<n> parallel_dps=<n>" + exit 0, or
//              "RESULT: FAIL (<m> mismatches between serial and parallel)" + exit 3.
//
// Public Domain (The Unlicense).
#include <chrono>
#include <cmath>
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
#include "chocofarm/net_evaluator.hpp"
#include "chocofarm/search_runtime.hpp"

namespace {
[[nodiscard]] std::optional<std::string_view> opt(std::span<const std::string_view> args,
                                                  std::string_view name) {
    for (size_t i = 1; i + 1 < args.size(); ++i)
        if (args[i] == name) return args[i + 1];
    return std::nullopt;
}
[[nodiscard]] int to_int(std::string_view s) { return std::atoi(std::string(s).c_str()); }
[[nodiscard]] double to_double(std::string_view s) { return std::atof(std::string(s).c_str()); }

// A deterministic, stateless leaf (a pure function of the feature vector) — cheap, finite, varied,
// thread-safe (no shared mutable state), so SerialRuntime and PoolRuntime see byte-identical leaves and
// the per-task decisions are bit-identical. Cost is representative of the tiny-MLP regime.
class DetNet final : public chocofarm::NetEvaluator {
  public:
    explicit DetNet(int n_slots) : n_slots_(n_slots) {}
    std::expected<chocofarm::NetPrediction, chocofarm::Error> predict(
        std::span<const float> x) const override {
        double s = 0.0;
        for (float v : x) s += static_cast<double>(v);
        chocofarm::NetPrediction pred;
        pred.value = static_cast<float>(0.01 * s);
        pred.logits.resize(static_cast<size_t>(n_slots_));
        for (int i = 0; i < n_slots_; ++i)
            pred.logits[static_cast<size_t>(i)] =
                static_cast<float>(std::sin(0.5 * static_cast<double>(i) + 0.001 * s));
        return pred;
    }

  private:
    int n_slots_;
};

[[nodiscard]] double secs(std::chrono::steady_clock::time_point a,
                          std::chrono::steady_clock::time_point b) {
    return std::chrono::duration<double>(b - a).count();
}
}  // namespace

int main(int argc, char** argv) {
    std::vector<std::string_view> args(argv, argv + argc);
    std::optional<std::string_view> instance = opt(args, "--instance");
    std::optional<std::string_view> faces = opt(args, "--faces");
    if (!instance || !faces) {
        std::cerr << "usage: search-runtime-bench --instance <p> --faces <p> [--tasks N --n-sims N "
                     "--m N --max-depth N --c-outcome N --lam f --workers N --reps N]\n";
        return 2;
    }

    const int n_tasks = opt(args, "--tasks") ? to_int(*opt(args, "--tasks")) : 32;
    const int workers = opt(args, "--workers") ? to_int(*opt(args, "--workers")) : 4;
    const int reps = opt(args, "--reps") ? to_int(*opt(args, "--reps")) : 3;
    const double lam = opt(args, "--lam") ? to_double(*opt(args, "--lam")) : 0.1;
    chocofarm::GumbelConfig cfg;
    if (auto v = opt(args, "--m")) cfg.m = to_int(*v);
    if (auto v = opt(args, "--n-sims")) cfg.n_sims = to_int(*v);
    if (auto v = opt(args, "--max-depth")) cfg.max_depth = to_int(*v);
    if (auto v = opt(args, "--c-outcome")) cfg.c_outcome = to_int(*v);

    auto inst = chocofarm::load_instance(*instance, *faces);
    if (!inst) {
        std::cerr << "search-runtime-bench: FATAL: " << inst.error().message << "\n";
        return 1;
    }
    chocofarm::Environment env(*inst);
    DetNet net(chocofarm::n_action_slots(env));

    // a batch of independent root-state tasks, one RNG seed each.
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

    std::cout << "config: tasks=" << n_tasks << " workers=" << workers << " reps=" << reps
              << " m=" << cfg.m << " n_sims=" << cfg.n_sims << " max_depth=" << cfg.max_depth
              << " c_outcome=" << cfg.c_outcome << " lam=" << lam
              << " n_slots=" << chocofarm::n_action_slots(env) << "\n";

    chocofarm::SerialRuntime serial(net);
    chocofarm::PoolRuntime pool(net, workers);

    // correctness FIRST (independent of timing): serial and pool must agree per task, bit-for-bit.
    auto serial_ref = serial.run(env, tasks);
    auto pool_ref = pool.run(env, tasks);
    if (!serial_ref || !pool_ref) {
        std::cerr << "search-runtime-bench: FATAL: a runtime returned an Error\n";
        return 1;
    }
    int mismatches = 0;
    long leaf_total = 0;
    for (size_t i = 0; i < tasks.size(); ++i) {
        const chocofarm::Decision& s = (*serial_ref)[i];
        const chocofarm::Decision& p = (*pool_ref)[i];
        if (!(s.executed == p.executed) || s.leaf_requests != p.leaf_requests) ++mismatches;
        leaf_total += s.leaf_requests;
    }
    if (mismatches != 0) {
        std::cout << "RESULT: FAIL (" << mismatches << " mismatches between serial and parallel)\n";
        return 3;
    }

    // timing: best-of-`reps` wall time for each runtime (best-of reduces scheduler noise).
    double best_serial = 1e300, best_pool = 1e300;
    for (int r = 0; r < reps; ++r) {
        auto t0 = std::chrono::steady_clock::now();
        auto a = serial.run(env, tasks);
        auto t1 = std::chrono::steady_clock::now();
        auto b = pool.run(env, tasks);
        auto t2 = std::chrono::steady_clock::now();
        (void)a;
        (void)b;
        double ds = secs(t0, t1), dp = secs(t1, t2);
        best_serial = std::min(best_serial, ds);
        best_pool = std::min(best_pool, dp);
        std::cout << "rep " << r << ": serial=" << ds << "s parallel=" << dp << "s\n";
    }

    const double serial_dps = static_cast<double>(n_tasks) / best_serial;
    const double parallel_dps = static_cast<double>(n_tasks) / best_pool;
    const double speedup = best_serial / best_pool;
    std::cout.precision(4);
    std::cout << "leaf_requests_total=" << leaf_total
              << " best_serial=" << best_serial << "s best_parallel=" << best_pool << "s\n";
    std::cout << "RESULT: PASS speedup=" << speedup << " serial_dps=" << serial_dps
              << " parallel_dps=" << parallel_dps << "\n";
    return 0;
}
