// throughput-lab/scratch/leaf_cpu_microbench.cpp
// Purpose: measure the PRODUCER's own per-leaf CPU cost (feature build + belief reductions + search
//   control flow) so the fiber-machinery overhead from fiber_switch_microbench.cpp can be put over a
//   MEASURED denominator (ADR-0009), not the Python/numba-era "~16 us/leaf" the prior header cited.
//
//   The real net forward happens on the SERVER, not the producer, so a trivial total DetNet leaf (the
//   same one fiber_proto.cpp uses) is a FAIR stand-in for the producer-side per-leaf cost MINUS the wire
//   encode/decode (which a CPU-only probe cannot include). It therefore UNDER-states the real producer
//   per-leaf wall slightly (no zmq codec) and OVER-states the fiber-overhead ratio — i.e. it is the
//   conservative direction for the soundness question.
//
//   Times N direct run_search() decisions at the production config (m=24, n_sims=256, c_outcome=2,
//   max_depth=24) on the SAME env/FeatureBuilder a producer fiber drives, reports us/decision and the
//   per-leaf cost (us/decision / leaves_per_decision). 3 interleaved replicates, median.
// Public Domain (The Unlicense).
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "chocofarm/cyclic_gumbel.hpp"
#include "chocofarm/env.hpp"
#include "chocofarm/fiber_tree.hpp"
#include "chocofarm/gumbel.hpp"
#include "chocofarm/gumbel_cursor.hpp"  // OPTION B — the explicit-state resumable cursor (head-to-head)
#include "chocofarm/instance.hpp"
#include "chocofarm/net_evaluator.hpp"
#include <variant>

using Clock = std::chrono::steady_clock;

namespace {
[[nodiscard]] std::optional<std::string_view> opt(std::span<const std::string_view> args,
                                                  std::string_view name) {
    for (size_t i = 1; i + 1 < args.size(); ++i)
        if (args[i] == name) return args[i + 1];
    return std::nullopt;
}
[[nodiscard]] int to_int(std::string_view s) { return std::atoi(std::string(s).c_str()); }

// The SAME total deterministic leaf fiber_proto.cpp uses (a pure function of the features) — stands in
// for the server-side forward, which is not a producer cost.
class DetNet final : public chocofarm::NetEvaluator {
  public:
    explicit DetNet(int n_slots) : n_slots_(n_slots) {}
    std::expected<chocofarm::NetPrediction, chocofarm::Error> predict(
        std::span<const float> x) const override {
        double s = 0.0;
        for (float v : x) s += static_cast<double>(v);
        chocofarm::NetPrediction p;
        p.value = static_cast<float>(0.01 * s);
        p.logits.resize(static_cast<size_t>(n_slots_));
        for (int i = 0; i < n_slots_; ++i)
            p.logits[static_cast<size_t>(i)] =
                static_cast<float>(std::sin(0.5 * static_cast<double>(i) + 0.001 * s));
        return p;
    }

  private:
    int n_slots_;
};
}  // namespace

int main(int argc, char** argv) {
    std::vector<std::string_view> args(argv, argv + argc);
    std::optional<std::string_view> instance = opt(args, "--instance");
    std::optional<std::string_view> faces = opt(args, "--faces");
    if (!instance || !faces) {
        std::cerr << "usage: leaf-cpu-microbench --instance <p> --faces <p> [--decisions N]\n";
        return 2;
    }
    const int decisions = opt(args, "--decisions") ? to_int(*opt(args, "--decisions")) : 2000;
    const std::string_view mode = opt(args, "--mode").value_or("both");  // direct | fibered | both

    chocofarm::GumbelConfig cfg;
    cfg.m = 24;
    cfg.n_sims = 256;
    cfg.c_outcome = 2;
    cfg.max_depth = 24;
    const double lam = 0.1;

    auto inst = chocofarm::load_instance(*instance, *faces);
    if (!inst) {
        std::cerr << "FATAL: " << inst.error().message << "\n";
        return 1;
    }
    chocofarm::Environment env(*inst);
    DetNet net(chocofarm::n_action_slots(env));
    chocofarm::Loc loc{env.entry_point()};
    chocofarm::Belief bw = env.full_belief();
    chocofarm::CollectedSet collected;
    std::vector<double> gtable{0.40, -0.65, 1.10, 0.05, -0.30, 0.85, -1.20, 0.55,
                               0.20, -0.45, 0.95, -0.10, 0.70};

    // one decision to learn leaves/decision (and warm caches/pages)
    chocofarm::GumbelAZPolicy warm_policy(cfg, net, env);
    {
        chocofarm::CyclicGumbelSource s(env, gtable);
        auto d = warm_policy.run_search(loc, bw, collected, lam, s);
        (void)d;
    }

    // DIRECT: pure producer per-leaf CPU (normal thread stack — no fiber, no per-decision mmap, no
    // stack-page re-faulting). The denominator.
    auto run_direct = [&](int n) -> double {
        volatile std::int64_t sink = 0;
        const auto t0 = Clock::now();
        for (int i = 0; i < n; ++i) {
            chocofarm::CyclicGumbelSource s(env, gtable);
            chocofarm::GumbelAZPolicy policy(cfg, net, env);
            auto d = policy.run_search(loc, bw, collected, lam, s);
            sink += d.n_spent;
        }
        const auto t1 = Clock::now();
        (void)sink;
        return std::chrono::duration<double, std::micro>(t1 - t0).count() / static_cast<double>(n);
    };

    // FIBERED: the production path — TreeState::start() mmaps a FRESH 512 KiB protected_fixedsize_stack
    // EVERY decision, the descent demand-faults its stack pages, and each leaf is a round-trip swap. This
    // is the ALL-IN fiber overhead (alloc + guard + switches + descent page-faults + munmap), end-to-end,
    // NOT a sum of isolated micro-costs. A fresh TreeState per decision mirrors a fiber slot taking a new
    // decision (start() re-allocs the fiber either way).
    // FIBERED-POOLED: drive the search through a fiber whose 512 KiB stack is allocated ONCE and REUSED
    // every decision (a pool of 1). This isolates the fresh-stack-per-decision cost (mmap+munmap+page
    // re-residency) from the intrinsic in-fiber execution cost (context switches + running on a fiber
    // stack at all). If pooled ≈ direct + switch, the overhead is the per-decision alloc and a pool fixes
    // it WITHIN Option A; if pooled still carries it, it is intrinsic to executing inside a fiber.
    struct PooledStack {
        boost::context::stack_context sc;
        boost::context::stack_context allocate() { return sc; }
        void deallocate(boost::context::stack_context&) noexcept {}  // reused; freed once at teardown
    };
    boost::context::protected_fixedsize_stack proto_alloc(512 * 1024);
    PooledStack pool{proto_alloc.allocate()};
    auto run_fibered_pooled = [&](int n) -> double {
        volatile std::int64_t sink = 0;
        const auto t0 = Clock::now();
        for (int i = 0; i < n; ++i) {
            chocofarm::FiberLeafChannel ch;
            chocofarm::YieldingNetEvaluator ynet(ch);
            chocofarm::GumbelAZPolicy policy(cfg, ynet, env);
            chocofarm::CyclicGumbelSource s(env, gtable);
            chocofarm::GumbelAZPolicy::Decision dec;
            boost::context::fiber fib{
                std::allocator_arg, pool,
                [&](boost::context::fiber&& caller) {
                    ch.caller = std::move(caller);
                    dec = policy.run_search(loc, bw, collected, lam, s);
                    ch.at_leaf = false;
                    return std::move(ch.caller);
                }};
            fib = std::move(fib).resume();
            while (ch.at_leaf) {
                auto pred = net.predict(ch.features);
                ch.value = pred.value();
                fib = std::move(fib).resume();
            }
            sink += dec.n_spent;
        }
        const auto t1 = Clock::now();
        (void)sink;
        return std::chrono::duration<double, std::micro>(t1 - t0).count() / static_cast<double>(n);
    };

    auto run_fibered = [&](int n) -> double {
        volatile std::int64_t sink = 0;
        const auto t0 = Clock::now();
        for (int i = 0; i < n; ++i) {
            chocofarm::TreeState ts(cfg, env, gtable);
            ts.start(loc, bw, collected, lam);
            while (ts.running) {
                auto pred = net.predict(ts.ch.features);
                ts.resume_with(pred.value());
            }
            sink += ts.decision.n_spent;
        }
        const auto t1 = Clock::now();
        (void)sink;
        return std::chrono::duration<double, std::micro>(t1 - t0).count() / static_cast<double>(n);
    };

    // OPTION B: the explicit-state cursor — the SAME resumable search WITHOUT a fiber. advance()/resume()
    // run STRAIGHT-LINE on the normal thread stack (no boost.context, no per-decision mmap'd stack, no
    // context switch); each leaf is a value-returned NeedsLeaf the driver feeds back via resume(). This
    // is the conjecture under test: does B recover the ~1% intrinsic fiber tax (run at ≈ the `direct`
    // cost)? A fresh GumbelAZPolicy + CyclicGumbelSource per decision mirrors run_direct / run_fibered's
    // per-decision construction (a fiber slot / a tree taking a new decision), so the comparison is fair.
    auto run_cursor = [&](int nn) -> double {
        volatile std::int64_t sink = 0;
        const auto t0 = Clock::now();
        for (int i = 0; i < nn; ++i) {
            chocofarm::CyclicGumbelSource s(env, gtable);
            chocofarm::GumbelAZPolicy policy(cfg, net, env);
            chocofarm::TreeCursor cur(policy, loc, bw, collected, lam, s);
            chocofarm::Step st = cur.advance();
            while (std::holds_alternative<chocofarm::CursorNeedsLeaf>(st)) {
                const auto& need = std::get<chocofarm::CursorNeedsLeaf>(st);
                auto pred = net.predict(need.features);
                st = cur.resume(pred.value());
            }
            sink += std::get<chocofarm::CursorDecided>(st).decision.n_spent;
        }
        const auto t1 = Clock::now();
        (void)sink;
        return std::chrono::duration<double, std::micro>(t1 - t0).count() / static_cast<double>(nn);
    };

    // --mode direct|fibered: run ONE path only (so /usr/bin/time -v attributes minor page-faults to it).
    if (mode == "direct") {
        double t = run_direct(decisions);
        std::printf("direct-only: %.2f us/decision over %d decisions\n", t, decisions);
        return 0;
    }
    if (mode == "fibered") {
        double t = run_fibered(decisions);
        std::printf("fibered-only: %.2f us/decision over %d decisions\n", t, decisions);
        return 0;
    }
    if (mode == "pooled") {
        double t = run_fibered_pooled(decisions);
        std::printf("pooled-only: %.2f us/decision over %d decisions\n", t, decisions);
        return 0;
    }
    if (mode == "cursor") {
        double t = run_cursor(decisions);
        std::printf("cursor-only: %.2f us/decision over %d decisions\n", t, decisions);
        return 0;
    }

    auto med3 = [](double a, double b, double c) {
        double hi = a > b ? a : b, lo = a > b ? b : a;
        return c > hi ? hi : (c < lo ? lo : c);
    };
    // interleaved replicates (direct, pooled, fibered, cursor) ×3 — robust-benchmark-statistics. The
    // cursor (Option B) is interleaved with the others so a thermal/scheduler drift hits all four arms
    // alike (the head-to-head is the per-rep relative ordering, not an absolute single reading).
    double rd[3], rp[3], rf[3], rc[3];
    for (int k = 0; k < 3; ++k) {
        rd[k] = run_direct(decisions);
        rp[k] = run_fibered_pooled(decisions);
        rf[k] = run_fibered(decisions);
        rc[k] = run_cursor(decisions);
    }
    double us_direct = med3(rd[0], rd[1], rd[2]);
    double us_pool = med3(rp[0], rp[1], rp[2]);
    double us_fiber = med3(rf[0], rf[1], rf[2]);
    double us_cursor = med3(rc[0], rc[1], rc[2]);
    const double L = 497.0;  // leaves/decision (fiber_proto, this config)
    std::printf("direct_us_per_decision        =%.2f  (reps: %.2f %.2f %.2f)\n", us_direct, rd[0], rd[1], rd[2]);
    std::printf("pooled_us_per_decision        =%.2f  (reps: %.2f %.2f %.2f)\n", us_pool, rp[0], rp[1], rp[2]);
    std::printf("freshstack_us_per_decision    =%.2f  (reps: %.2f %.2f %.2f)\n", us_fiber, rf[0], rf[1], rf[2]);
    std::printf("cursor_us_per_decision  (B)   =%.2f  (reps: %.2f %.2f %.2f)\n", us_cursor, rc[0], rc[1], rc[2]);
    std::printf("--- decomposition (us/decision, %%of direct) ---\n");
    std::printf("intrinsic fiber (pooled-direct): %.2f us = %.3f%%  (context-switch + run-on-fiber-stack)\n",
                us_pool - us_direct, 100.0 * (us_pool - us_direct) / us_direct);
    std::printf("fresh-stack tax (fresh-pooled):  %.2f us = %.3f%%  (per-decision mmap/munmap+residency; a POOL removes this)\n",
                us_fiber - us_pool, 100.0 * (us_fiber - us_pool) / us_direct);
    std::printf("ALL-IN fiber (fresh-direct):     %.2f us = %.2f ns/leaf = %.3f%% of producer per-leaf CPU\n",
                us_fiber - us_direct, (us_fiber - us_direct) * 1000.0 / L, 100.0 * (us_fiber - us_direct) / us_direct);
    std::printf("--- OPTION B vs A (the conjecture: does B recover the fiber tax, running at ≈ direct?) ---\n");
    std::printf("cursor (B) over direct:          %.2f us = %.3f%%  (the cursor's OWN overhead vs the bare recursion)\n",
                us_cursor - us_direct, 100.0 * (us_cursor - us_direct) / us_direct);
    std::printf("cursor (B) vs fresh-fiber (A):   %.2f us = %.3f%% of direct  (the head-to-head: A_allin - B_overhead)\n",
                us_fiber - us_cursor, 100.0 * (us_fiber - us_cursor) / us_direct);
    std::printf("producer per-leaf CPU (direct) = %.4f us/leaf\n", us_direct / L);
    return 0;
}
