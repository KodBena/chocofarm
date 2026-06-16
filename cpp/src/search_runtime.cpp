// cpp/src/search_runtime.cpp
// Purpose: SerialRuntime — the one-tree-at-a-time SearchRuntime impl (see search_runtime.hpp). It is a
//   faithful wrapper over the existing GumbelAZPolicy::decide: it adds the batch loop, the per-task RNG
//   seeding, and the leaf-request count, and touches nothing in the validated search.
//
// Public Domain (The Unlicense).
#include "chocofarm/search_runtime.hpp"

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <random>
#include <thread>

#include "chocofarm/gumbel.hpp"

namespace chocofarm {

namespace {
// Map the search's full Decision (action + improved-π in double + n_spent) onto the runtime Decision
// (improved-π narrowed to float32, the wire/trainer dtype), stamping the per-decision leaf-request count.
// Shared by both runtimes so the mapping has one home (P1).
[[nodiscard]] Decision to_decision(const GumbelAZPolicy::Decision& dec, int leaf_requests) {
    Decision d;
    d.executed = dec.action;
    d.improved_pi.assign(dec.improved.begin(), dec.improved.end());  // double -> float32
    d.n_spent = dec.n_spent;
    d.leaf_requests = leaf_requests;
    return d;
}
}  // namespace

std::expected<std::vector<Decision>, Error>
SerialRuntime::run(const Environment& env, std::span<const SearchTask> tasks) const {
    std::vector<Decision> out;
    out.reserve(tasks.size());
    for (const SearchTask& task : tasks) {
        // One counting decorator + one seeded RNG per task — a task is a fully independent tree (its own
        // stream). The decorator delegates to the held net unchanged, so the search sees exactly the
        // leaves it would without the runtime; the count is the structural observable.
        CountingNetEvaluator counter(net_);
        GumbelAZPolicy policy(task.cfg, counter, env);
        std::mt19937_64 rng(task.seed);
        // decide_with_target() builds the production RngGumbelSource off `rng` internally and returns the
        // FULL Decision (executed action + improved-π + n_spent) — the UNCHANGED search entry point.
        // SerialRuntime over a LOCAL total net cannot fail here (the leaf is total); the seam's error arm
        // is the contract a future remote-leaf pool needs, so run() returns the fallible expected even
        // though this impl always takes the value arm.
        GumbelAZPolicy::Decision dec =
            policy.decide_with_target(env, task.loc, task.bw, task.collected, task.lam, rng);
        out.push_back(to_decision(dec, counter.count()));
    }
    return out;
}

std::expected<std::vector<Decision>, Error>
PoolRuntime::run(const Environment& env, std::span<const SearchTask> tasks) const {
    // Pre-size the result so each worker writes its own DISJOINT indices (no contention, no lock on the
    // output). A shared atomic cursor hands out the next task index; a worker runs that whole tree
    // exactly as SerialRuntime does (a fresh CountingNetEvaluator + a seeded RNG per task), so its
    // per-task Decision is bit-identical to the serial one. Independent trees + a thread-safe net mean
    // no shared mutable state across workers (see the header's thread-safety note).
    std::vector<Decision> out(tasks.size());
    std::atomic<std::size_t> cursor{0};

    auto worker = [&]() {
        std::size_t i;
        while ((i = cursor.fetch_add(1, std::memory_order_relaxed)) < tasks.size()) {
            const SearchTask& task = tasks[i];
            CountingNetEvaluator counter(net_);
            GumbelAZPolicy policy(task.cfg, counter, env);
            std::mt19937_64 rng(task.seed);
            GumbelAZPolicy::Decision dec =
                policy.decide_with_target(env, task.loc, task.bw, task.collected, task.lam, rng);
            out[i] = to_decision(dec, counter.count());
        }
    };

    // Never spawn more threads than tasks (an idle thread does nothing useful); at least one.
    int nw = std::max(1, std::min(n_workers_, static_cast<int>(tasks.size())));
    if (nw == 1) {  // degenerate: run inline (the serial path), no thread spawn
        worker();
        return out;
    }
    std::vector<std::thread> threads;
    threads.reserve(static_cast<std::size_t>(nw));
    for (int t = 0; t < nw; ++t) threads.emplace_back(worker);
    for (std::thread& th : threads) th.join();
    return out;
}

}  // namespace chocofarm
