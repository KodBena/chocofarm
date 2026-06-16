// cpp/src/search_runtime.cpp
// Purpose: SerialRuntime — the one-tree-at-a-time SearchRuntime impl (see search_runtime.hpp). It is a
//   faithful wrapper over the existing GumbelAZPolicy::decide: it adds the batch loop, the per-task RNG
//   seeding, and the leaf-request count, and touches nothing in the validated search.
//
// Public Domain (The Unlicense).
#include "chocofarm/search_runtime.hpp"

#include <random>

#include "chocofarm/gumbel.hpp"

namespace chocofarm {

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
        // decide() builds the production RngGumbelSource off `rng` internally and returns the executed
        // action (the SH survivor, temperature 0) — the UNCHANGED search entry point. SerialRuntime over
        // a LOCAL total net cannot fail here (the leaf is total); the seam's error arm is the contract a
        // future remote-leaf pool needs, so run() returns the fallible expected even though this impl
        // always takes the value arm.
        Action executed = policy.decide(env, task.loc, task.bw, task.collected, task.lam, rng);
        out.push_back(Decision{executed, counter.count()});
    }
    return out;
}

}  // namespace chocofarm
