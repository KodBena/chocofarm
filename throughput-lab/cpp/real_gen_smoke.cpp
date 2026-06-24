// throughput-lab/cpp/real_gen_smoke.cpp
// Purpose: a BUILD-COUPLING PROOF for the real-generator integration — link the throughput-lab against
//   the parent chocofarm_core (env / gumbel / search_runtime / features) and run ONE real Gumbel-AZ
//   decision through SerialRuntime over a local DetNet, printing its leaf-request count. This isolates
//   the foundational risk (does the lab compile + link + run against chocofarm_core under C++23 — ODR,
//   redis/boost transitive deps, toolchain match?) from the adapter logic (BoundaryNetEvaluator) that
//   follows. NOT a benchmark and NOT the integration itself; the smoke that proves the seam is reachable.
//   Built only when -DTLAB_REAL_GENERATOR=ON (the synthetic-only lab build stays standalone, ADR-0012).
// Public Domain (The Unlicense).
#include <iostream>
#include <span>
#include <string_view>
#include <vector>

#include "chocofarm/env.hpp"
#include "chocofarm/gumbel.hpp"
#include "chocofarm/instance.hpp"
#include "chocofarm/net_evaluator.hpp"
#include "chocofarm/search_runtime.hpp"

namespace {
// The same deterministic, stateless local leaf the parent's search_runtime_bench uses — a pure function
// of the feature vector, so the smoke needs no net weights / redis / wire. Stands in for the real net;
// the REAL integration replaces this with a BoundaryNetEvaluator that round-trips the lab transport.
class DetNet final : public chocofarm::NetEvaluator {
  public:
    explicit DetNet(int n_slots) : n_slots_(n_slots) {}
    std::expected<chocofarm::NetPrediction, chocofarm::Error> predict(
        std::span<const float> x) const override {
        double s = 0.0;
        for (float v : x) s += static_cast<double>(v);
        chocofarm::NetPrediction pred;
        pred.value = static_cast<float>(0.01 * s);
        pred.logits.assign(static_cast<size_t>(n_slots_), 0.0f);
        return pred;
    }

  private:
    int n_slots_;
};
}  // namespace

int main(int argc, char** argv) {
    std::vector<std::string_view> args(argv, argv + argc);
    if (args.size() < 3) {
        std::cerr << "usage: tlab-real-gen-smoke <instance.json> <faces.json> [n_sims]\n";
        return 2;
    }
    const int n_sims = args.size() > 3 ? std::atoi(std::string(args[3]).c_str()) : 16;

    auto inst = chocofarm::load_instance(args[1], args[2]);
    if (!inst) {
        std::cerr << "tlab-real-gen-smoke: FATAL: " << inst.error().message << "\n";
        return 1;
    }
    chocofarm::Environment env(*inst);
    const int n_slots = chocofarm::n_action_slots(env);
    DetNet net(n_slots);

    chocofarm::GumbelConfig cfg;
    cfg.n_sims = n_sims;
    chocofarm::SearchTask t;
    t.loc = chocofarm::Loc{env.entry_point()};
    t.bw = env.full_belief();
    t.collected = chocofarm::CollectedSet{};
    t.lam = 0.1;
    t.seed = 1;
    t.cfg = cfg;

    chocofarm::SerialRuntime serial(net);
    std::vector<chocofarm::SearchTask> tasks{t};
    auto out = serial.run(env, std::span<const chocofarm::SearchTask>(tasks));
    if (!out) {
        std::cerr << "tlab-real-gen-smoke: FATAL: runtime error: " << out.error().message << "\n";
        return 1;
    }
    std::cout << "tlab-real-gen-smoke: OK  n_slots=" << n_slots
              << "  n_sims=" << cfg.n_sims
              << "  leaf_requests=" << (*out)[0].leaf_requests << "\n";
    return 0;
}
