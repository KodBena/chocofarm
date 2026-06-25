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

#include <memory>

#include "chocofarm/env.hpp"
#include "chocofarm/gumbel.hpp"
#include "chocofarm/instance.hpp"
#include "chocofarm/net_evaluator.hpp"
#include "chocofarm/search_runtime.hpp"

#include "boundary.hpp"
#include "boundary_net_evaluator.hpp"

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
        std::cerr << "usage: tlab-real-gen-smoke <instance.json> <faces.json> [n_sims] [endpoint]\n"
                     "  no endpoint -> local DetNet (build-coupling proof);\n"
                     "  endpoint (ipc://...) -> route each leaf through tlab::Boundary to a live server.\n";
        return 2;
    }
    // CLI ACL (ADR-0002 fail-loud at the boundary): the optional n_sims is a core search-shape int consumed
    // straight into GumbelConfig.n_sims (the core's raw-int field — its typedef is the core's home, not
    // tlab's). A non-positive value is rejected here rather than fed onward as a meaningless budget.
    int n_sims = 16;
    if (args.size() > 3) {
        n_sims = std::atoi(std::string(args[3]).c_str());
        if (n_sims < 1) {
            std::cerr << "tlab-real-gen-smoke: n_sims must be an integer >= 1\n";
            return 2;
        }
    }
    const bool route = args.size() > 4;   // a 4th arg = the server endpoint -> route through the boundary

    auto inst = chocofarm::load_instance(args[1], args[2]);
    if (!inst) {
        std::cerr << "tlab-real-gen-smoke: FATAL: " << inst.error().message << "\n";
        return 1;
    }
    chocofarm::Environment env(*inst);
    // n_action_slots returns a typed SlotCount; this smoke driver sizes DetNet's logits from a raw int —
    // unwrap at the crossing (ADR-0000 item 5: a named, visible raw<->domain crossing).
    const int n_slots = static_cast<int>(chocofarm::n_action_slots(env).value());

    // The leaf evaluator: a local DetNet (no transport — the build-coupling proof), OR a
    // BoundaryNetEvaluator that round-trips each leaf through OUR tlab::Boundary to a live server (the
    // routing proof — the real generator driving load over the real transport).
    DetNet det(n_slots);
    std::unique_ptr<tlab::Boundary> boundary;
    std::unique_ptr<tlab::BoundaryNetEvaluator> bridge;
    const chocofarm::NetEvaluator* net = &det;
    if (route) {
        // BoundaryConfig speaks the typed tlab domains (Milliseconds/ThreadCount/RowCount/FeatureDim,
        // boundary.hpp/proc_domains.hpp); construct the literal knobs in-domain (ADR-0000 rule 1).
        tlab::BoundaryConfig bcfg;
        bcfg.endpoint = std::string(args[4]);
        bcfg.recv_timeout_ms = tlab::Milliseconds{5000};
        bcfg.n_producer_threads = tlab::ThreadCount{1};
        bcfg.rows = tlab::wire::RowCount{1};
        bcfg.in_dim = tlab::wire::FeatureDim{241};
        auto b = tlab::make_boundary(tlab::BoundaryTopology::PerThread, bcfg);
        if (!b) {
            std::cerr << "tlab-real-gen-smoke: FATAL: boundary: " << b.error().message << "\n";
            return 1;
        }
        boundary = std::move(*b);
        bridge = std::make_unique<tlab::BoundaryNetEvaluator>(*boundary);
        net = bridge.get();
    }

    chocofarm::GumbelConfig cfg;
    cfg.n_sims = n_sims;
    chocofarm::SearchTask t;
    t.loc = chocofarm::Loc{env.entry_point()};
    t.bw = env.full_belief();
    t.collected = chocofarm::CollectedSet{};
    t.lam = 0.1;
    t.seed = chocofarm::RngSeed{1};  // opaque 64-bit seed bit-pattern (RngSeed domain), not a count/index
    t.cfg = cfg;

    chocofarm::SerialRuntime serial(*net);
    std::vector<chocofarm::SearchTask> tasks{t};
    auto out = serial.run(env, std::span<const chocofarm::SearchTask>(tasks));
    if (!out) {
        std::cerr << "tlab-real-gen-smoke: FATAL: runtime error: " << out.error().message << "\n";
        return 1;
    }
    std::cout << "tlab-real-gen-smoke: OK  n_slots=" << n_slots
              << "  n_sims=" << cfg.n_sims
              << "  leaf_requests=" << (*out)[0].leaf_requests.value() << "\n";  // SimBudget -> raw for print
    return 0;
}
