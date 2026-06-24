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
#include <sys/resource.h>   // setpriority / PRIO_PROCESS — per-thread nice (Linux: nice is per-task)
#include <sys/syscall.h>    // SYS_gettid
#include <unistd.h>         // syscall

#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <vector>

#include "chocofarm/env.hpp"
#include "chocofarm/fiber_tree.hpp"
#include "chocofarm/gumbel.hpp"
#include "chocofarm/instance.hpp"
#include "chocofarm/search_runtime.hpp"

#include "boundary.hpp"
#include "boundary_net_evaluator.hpp"

namespace {
// A small fixed Gumbel script (the scripted CyclicGumbelSource path, as wire_parallel_bench uses): the
// RNG-free source produces a faithful search STRUCTURE / leaf pattern without the production RNG slot-fill
// — sufficient for a LOAD generator (throughput depends on the search's leaf-dependency + matmul shape,
// not on the gumbel draws being random). Each fiber rotates the table so the K trees differ.
const std::vector<double> kGumbelTable{0.40, -0.65, 1.10, 0.05, -0.30, 0.85, -1.20, 0.55,
                                       0.20, -0.45, 0.95, -0.10, 0.70};
constexpr double kLam = 0.1;
}  // namespace

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

// Renice THIS generator thread DOWN iff it is the designated one (Linux per-task nice via setpriority on
// the thread's gettid). In the generator-bound regime, the inference server's core has idle slack; a 4th
// generator sharing that core, reniced low, soaks the slack but yields the instant the server has a batch.
// A no-op unless low_prio_thread names this index. Failure is non-fatal + loud-ish (ADR-0002).
void apply_thread_priority(int thread_index, int low_prio_thread, int low_prio_nice) {
    if (low_prio_thread < 0 || thread_index != low_prio_thread || low_prio_nice == 0) return;
    const auto tid = static_cast<id_t>(::syscall(SYS_gettid));
    errno = 0;
    if (::setpriority(PRIO_PROCESS, tid, low_prio_nice) != 0)
        std::fprintf(stderr, "[tlab-real-producer] WARN: could not renice thread %d to nice %d (errno=%d)\n",
                     thread_index, low_prio_nice, errno);
}

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

// One FIBER producer thread: multiplex K TreeState fibers over its own Boundary, ROUND-SYNCHRONOUS
// (wire_parallel_bench's discipline): each round, submit every parked fiber's leaf (B=1) into the DEALER,
// let the SERVER gather the K concurrent requests into one forward, then recv the K replies and resume
// each fiber. K leaves in flight per thread -> the server's per-forward batch grows with K (and with N
// threads, ~N*K). A finished fiber is restarted on a fresh decision to keep K in flight for the window.
// This is the fiber arm of the investigation; run_thread (above) is the non-fiber baseline it is measured
// against. (Greedy-async -- keep the pipe full across rounds -- is the next refinement.)
void run_thread_fiber(int idx, const chocofarm::Environment& env, const std::string& endpoint,
                      const chocofarm::GumbelConfig& cfg, double run_seconds, int in_dim, int fibers_k,
                      ThreadStat& out) {
    tlab::BoundaryConfig bcfg;
    bcfg.endpoint = endpoint;
    bcfg.recv_timeout_ms = 10000;
    bcfg.n_producer_threads = 1;
    bcfg.rows = 1;            // B=1 per submitted leaf; the SERVER coalesces across the K in flight
    bcfg.in_dim = in_dim;
    auto b = tlab::make_boundary(tlab::BoundaryTopology::PerThread, bcfg);
    if (!b) { out.failed = true; out.err = "boundary: " + b.error().message; return; }
    std::unique_ptr<tlab::Boundary> boundary = std::move(*b);

    // Root state — kept alive for every fiber's whole life (TreeState::start captures loc/bw/coll BY
    // REFERENCE and re-reads them on every leaf across all resume_with calls).
    const chocofarm::Loc loc{env.entry_point()};
    const chocofarm::Belief bw = env.full_belief();
    const chocofarm::CollectedSet coll;

    // K independent tree-fibers (scripted source, per-tree rotated table so the trees differ).
    std::vector<std::unique_ptr<chocofarm::TreeState>> trees;
    trees.reserve(static_cast<size_t>(fibers_k));
    for (int i = 0; i < fibers_k; ++i) {
        std::vector<double> table(kGumbelTable.size());
        for (size_t j = 0; j < kGumbelTable.size(); ++j)
            table[j] = kGumbelTable[(j + static_cast<size_t>(i)) % kGumbelTable.size()];
        trees.push_back(std::make_unique<chocofarm::TreeState>(cfg, env, std::move(table)));
    }
    for (auto& t : trees) t->start(loc, bw, coll, kLam);   // advance each to its first parked leaf

    tlab::wire::corr_t corr = static_cast<tlab::wire::corr_t>(idx) * 1'000'000'000ull + 1ull;
    const auto t_start = SteadyClock::now();
    while (secs_since(t_start) < run_seconds) {
        // Collect parked fibers; restart any that finished (count the completed decision) to keep K busy.
        std::vector<int> active;
        active.reserve(static_cast<size_t>(fibers_k));
        for (int i = 0; i < fibers_k; ++i) {
            if (!trees[static_cast<size_t>(i)]->running) {
                out.decisions += 1;
                trees[static_cast<size_t>(i)]->start(loc, bw, coll, kLam);
            }
            if (trees[static_cast<size_t>(i)]->running) active.push_back(i);
        }
        if (active.empty()) break;

        // Submit every parked leaf (B=1). corr->fiber so the (unordered) DEALER replies route home.
        std::unordered_map<tlab::wire::corr_t, int> corr_to_fiber;
        corr_to_fiber.reserve(active.size());
        for (int i : active) {
            const std::span<const float> feats = trees[static_cast<size_t>(i)]->ch.features;
            const tlab::wire::corr_t cc = corr++;
            corr_to_fiber.emplace(cc, i);
            const tlab::LeafBatch lb{cc, 1, static_cast<tlab::wire::count_t>(feats.size()), feats};
            if (auto s = boundary->send(lb); !s) { out.failed = true; out.err = "send: " + s.error().message; return; }
        }
        // Recv exactly |active| replies, routing each to its fiber and resuming it.
        for (size_t r = 0; r < active.size(); ++r) {
            auto reply = boundary->recv();
            if (!reply) { out.failed = true; out.err = "recv: " + reply.error().message; return; }
            auto it = corr_to_fiber.find(reply->corr);
            if (it == corr_to_fiber.end() || reply->preds.empty()) {
                out.failed = true; out.err = "unmatched/empty reply corr=" + std::to_string(reply->corr); return;
            }
            chocofarm::NetPrediction pred;
            pred.value = reply->preds[0].value;
            pred.logits = std::move(reply->preds[0].logits);
            trees[static_cast<size_t>(it->second)]->resume_with(pred);
            out.leaves += 1;
        }
    }
}

// One GREEDY-ASYNC fiber producer thread (wire_pool_bench's discipline): keep ~K leaves CONTINUOUSLY in
// flight and process replies as they land — recv ONE, resume that fiber (it computes its next leaf), and
// re-submit it immediately, rather than the round-synchronous barrier (submit all, wait for all). The
// difference is overlap: in round-sync the search cores idle while the whole round's replies are awaited;
// here a thread is always either receiving or computing a fiber's next leaf, so the search cores stay
// busy across the RTT. Same corr->fiber routing and finished-fiber restart as the round-sync arm; the
// ONLY change is the pipeline shape. Optional (selected by --driver greedy) so the round-sync semantics
// are preserved for comparison.
void run_thread_fiber_greedy(int idx, const chocofarm::Environment& env, const std::string& endpoint,
                             const chocofarm::GumbelConfig& cfg, double run_seconds, int in_dim,
                             int fibers_k, ThreadStat& out) {
    tlab::BoundaryConfig bcfg;
    bcfg.endpoint = endpoint;
    bcfg.recv_timeout_ms = 10000;
    bcfg.n_producer_threads = 1;
    bcfg.rows = 1;
    bcfg.in_dim = in_dim;
    auto b = tlab::make_boundary(tlab::BoundaryTopology::PerThread, bcfg);
    if (!b) { out.failed = true; out.err = "boundary: " + b.error().message; return; }
    std::unique_ptr<tlab::Boundary> boundary = std::move(*b);

    const chocofarm::Loc loc{env.entry_point()};
    const chocofarm::Belief bw = env.full_belief();
    const chocofarm::CollectedSet coll;

    std::vector<std::unique_ptr<chocofarm::TreeState>> trees;
    trees.reserve(static_cast<size_t>(fibers_k));
    for (int i = 0; i < fibers_k; ++i) {
        std::vector<double> table(kGumbelTable.size());
        for (size_t j = 0; j < kGumbelTable.size(); ++j)
            table[j] = kGumbelTable[(j + static_cast<size_t>(i)) % kGumbelTable.size()];
        trees.push_back(std::make_unique<chocofarm::TreeState>(cfg, env, std::move(table)));
    }
    for (auto& t : trees) t->start(loc, bw, coll, kLam);

    std::unordered_map<tlab::wire::corr_t, int> corr_to_fiber;
    tlab::wire::corr_t corr = static_cast<tlab::wire::corr_t>(idx) * 1'000'000'000ull + 1ull;
    int in_flight = 0;
    // Submit fiber i's current parked leaf, restarting it on a fresh decision if it had finished. Returns
    // false only on a send error (out.failed set) or if a restarted fiber yielded no leaf (exhausted).
    auto submit = [&](int i) -> bool {
        if (!trees[static_cast<size_t>(i)]->running) {
            out.decisions += 1;
            trees[static_cast<size_t>(i)]->start(loc, bw, coll, kLam);
        }
        if (!trees[static_cast<size_t>(i)]->running) return false;  // produced no leaf (won't happen for n_sims>=1)
        const std::span<const float> feats = trees[static_cast<size_t>(i)]->ch.features;
        const tlab::wire::corr_t cc = corr++;
        const tlab::LeafBatch lb{cc, 1, static_cast<tlab::wire::count_t>(feats.size()), feats};
        if (auto s = boundary->send(lb); !s) { out.failed = true; out.err = "send: " + s.error().message; return false; }
        corr_to_fiber.emplace(cc, i);
        ++in_flight;
        return true;
    };

    for (int i = 0; i < fibers_k; ++i) submit(i);   // prime: K leaves in flight
    if (out.failed) return;

    const auto t_start = SteadyClock::now();
    while (secs_since(t_start) < run_seconds && in_flight > 0) {
        auto reply = boundary->recv();
        if (!reply) { out.failed = true; out.err = "recv: " + reply.error().message; return; }
        auto it = corr_to_fiber.find(reply->corr);
        if (it == corr_to_fiber.end() || reply->preds.empty()) {
            out.failed = true; out.err = "unmatched/empty reply corr=" + std::to_string(reply->corr); return;
        }
        const int i = it->second;
        corr_to_fiber.erase(it);
        --in_flight;
        chocofarm::NetPrediction pred;
        pred.value = reply->preds[0].value;
        pred.logits = std::move(reply->preds[0].logits);
        trees[static_cast<size_t>(i)]->resume_with(pred);   // advances to its next leaf (or finishes)
        out.leaves += 1;
        submit(i);                                          // re-arm this fiber -> back to ~K in flight
        if (out.failed) return;
    }
}
}  // namespace

int main(int argc, char** argv) {
    std::vector<std::string_view> args(argv, argv + argc);
    auto inst_p = opt(args, "--instance"), faces_p = opt(args, "--faces"), ep = opt(args, "--endpoint");
    if (!inst_p || !faces_p || !ep) {
        std::cerr << "usage: tlab-real-producer --instance <p> --faces <p> --endpoint <ipc://...> "
                     "[--threads N --fibers K --driver round-sync|greedy --seconds S --n-sims K --m M --in-dim D]\n"
                     "  --fibers 0 (default) = non-fiber baseline; K>=1 = K fibers/thread (the fiber model)\n"
                     "  --driver round-sync (default) | greedy (keep ~K leaves continuously in flight)\n";
        return 2;
    }
    const int threads = opt(args, "--threads") ? std::atoi(std::string(*opt(args, "--threads")).c_str()) : 3;
    const double seconds = opt(args, "--seconds") ? std::atof(std::string(*opt(args, "--seconds")).c_str()) : 5.0;
    const int in_dim = opt(args, "--in-dim") ? std::atoi(std::string(*opt(args, "--in-dim")).c_str()) : 241;
    // --fibers K: 0 (default) = NON-FIBER baseline (one SerialRuntime/thread, B=1 blocking); K>=1 = the
    // FIBER model (K TreeState fibers/thread multiplexed, K leaves in flight -> server batch grows with K).
    const int fibers = opt(args, "--fibers") ? std::atoi(std::string(*opt(args, "--fibers")).c_str()) : 0;
    // --driver: how the FIBER arm pipelines leaves. round-sync (default, preserves the committed
    // semantics) submits a whole round then awaits it; greedy keeps ~K leaves continuously in flight.
    const std::string driver = opt(args, "--driver") ? std::string(*opt(args, "--driver")) : "round-sync";
    if (driver != "round-sync" && driver != "greedy") {
        std::cerr << "tlab-real-producer: --driver must be round-sync|greedy, got " << driver << "\n";
        return 2;
    }
    // Renice ONE generator thread DOWN (the generator-bound core-sharing lever): a 4th generator on the
    // server's core, reniced, soaks its idle slack but yields to the server's forward.
    const int low_prio_thread = opt(args, "--low-prio-thread")
        ? std::atoi(std::string(*opt(args, "--low-prio-thread")).c_str()) : -1;
    const int low_prio_nice = opt(args, "--low-prio-nice")
        ? std::atoi(std::string(*opt(args, "--low-prio-nice")).c_str()) : 0;
    chocofarm::GumbelConfig cfg;
    if (auto v = opt(args, "--n-sims")) cfg.n_sims = std::atoi(std::string(*v).c_str());
    if (auto v = opt(args, "--m")) cfg.m = std::atoi(std::string(*v).c_str());

    auto inst = chocofarm::load_instance(*inst_p, *faces_p);
    if (!inst) { std::cerr << "tlab-real-producer: FATAL: " << inst.error().message << "\n"; return 1; }
    chocofarm::Environment env(*inst);

    std::cout << "tlab-real-producer: generator=real(" << (fibers > 0 ? "fiber" : "non-fiber")
              << ") driver=" << (fibers > 0 ? driver : std::string("n/a"))
              << " threads=" << threads << " fibers_per_thread=" << fibers
              << " seconds=" << seconds << " n_sims=" << cfg.n_sims << " m=" << cfg.m
              << " n_slots=" << chocofarm::n_action_slots(env) << " endpoint=" << *ep << "\n";

    std::vector<ThreadStat> stats(static_cast<size_t>(threads));
    std::vector<std::thread> pool;
    const std::string endpoint(*ep);
    const auto t0 = SteadyClock::now();
    for (int i = 0; i < threads; ++i)
        pool.emplace_back([&, i] {
            apply_thread_priority(i, low_prio_thread, low_prio_nice);   // renice iff designated
            if (fibers > 0 && driver == "greedy")
                run_thread_fiber_greedy(i, env, endpoint, cfg, seconds, in_dim, fibers, stats[static_cast<size_t>(i)]);
            else if (fibers > 0)
                run_thread_fiber(i, env, endpoint, cfg, seconds, in_dim, fibers, stats[static_cast<size_t>(i)]);
            else
                run_thread(i, env, endpoint, cfg, seconds, in_dim, stats[static_cast<size_t>(i)]);
        });
    for (auto& th : pool) th.join();
    const double wall = secs_since(t0);

    std::uint64_t dec = 0, leaves = 0; bool any_fail = false;
    for (const auto& s : stats) {
        dec += s.decisions; leaves += s.leaves;
        if (s.failed) { any_fail = true; std::cerr << "  thread failed: " << s.err << "\n"; }
    }
    const double dps = wall > 0 ? static_cast<double>(dec) / wall : 0.0;
    const double lps = wall > 0 ? static_cast<double>(leaves) / wall : 0.0;
    std::cout << "REAL-AGG threads=" << threads << " fibers=" << fibers
              << " driver=" << (fibers > 0 ? driver : std::string("n/a")) << " wall_s=" << wall
              << " decisions=" << dec << " leaves=" << leaves
              << " decisions_per_sec=" << dps << " leaves_per_sec=" << lps
              << " any_fail=" << (any_fail ? 1 : 0) << "\n";
    return any_fail ? 1 : 0;
}
