// cpp/src/serial_runtime_check.cpp
// Purpose: a self-contained, deterministic SEAM-FAITHFULNESS check for SerialRuntime (NOT the runner).
//   It proves the SearchRuntime seam does not perturb the search: SerialRuntime.run(env, tasks)[i] must
//   produce the SAME executed action as a direct GumbelAZPolicy::decide on task i with the same seeded
//   RNG, for a batch of independent tasks. The leaf is a DETERMINISTIC, STATELESS net (a pure function
//   of the feature vector — no redis, no weights, no RNG), so the whole check is reproducible and
//   in-process; the only randomness is the search's own std::mt19937_64, seeded per task. Exit 0 iff
//   every task's executed action matches and every decision issued ≥1 leaf request, else nonzero.
//
//   This is the runtime-level analogue of the gumbel logic check (cpp/parity/gumbel_logic.py): the
//   gumbel check pins the SEARCH against Python; this pins the SEAM against the un-wrapped search. It is
//   a regression guard for SerialRuntime's plumbing (the batch loop, the per-task seed -> RNG -> source,
//   the CountingNetEvaluator delegating faithfully, the input-order result alignment) — a divergence
//   here means the runtime changed the search, which the work-stealing pool's parity precondition
//   (docs/design/cpp-search-runtime.md §7.1) forbids.
//
//   ADR-0012 P9: the imperative shell. argv is decoded once into typed views; `opt` returns a
//   std::optional<std::string_view>; load_instance returns a typed std::expected reported loudly.
//
//   Protocol:  serial-runtime-check --instance <p> --faces <p>   (no stdin; the net is in-process)
//   Output:    one line per task ("task i: serial=<slot> direct=<slot> leaves=<n> OK|MISMATCH"), then
//              "PASS (K/K)" + exit 0, or "FAIL (m mismatches)" + exit 3.
//
// Public Domain (The Unlicense).
#include <cmath>
#include <cstdint>
#include <iostream>
#include <optional>
#include <random>
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

// A DETERMINISTIC, STATELESS leaf: predict(x) is a pure function of the feature vector — finite, varied
// per slot, and independent of call order — so SerialRuntime and the direct reference see byte-identical
// leaves for the same belief. Realism is irrelevant here (the check is about the SEAM, not search
// quality); determinism and statelessness are what make the comparison exact.
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

// The executed action's slot (for printing / comparison), mirroring gumbel_dump's slot encoding.
[[nodiscard]] int exec_slot(const chocofarm::Environment& env, const chocofarm::Action& a) {
    if (a.kind == chocofarm::ActionKind::Terminate) return chocofarm::term_slot(env);
    if (a.kind == chocofarm::ActionKind::Treasure) return a.i;
    return env.N() + a.i;
}
}  // namespace

int main(int argc, char** argv) {
    std::vector<std::string_view> args(argv, argv + argc);
    std::optional<std::string_view> instance = opt(args, "--instance");
    std::optional<std::string_view> faces = opt(args, "--faces");
    if (!instance || !faces) {
        std::cerr << "usage: serial-runtime-check --instance <p> --faces <p>\n";
        return 2;
    }

    auto inst = chocofarm::load_instance(*instance, *faces);
    if (!inst) {
        std::cerr << "serial-runtime-check: FATAL: " << inst.error().message << "\n";
        return 1;
    }
    chocofarm::Environment env(*inst);
    DetNet net(chocofarm::n_action_slots(env));

    // A batch of independent tasks at the root state, varying the seed AND the budget, so the seam is
    // exercised across distinct RNG streams and distinct cfgs. A reduced budget keeps the check fast
    // (the full C(N,K) belief makes a 48-sim search heavy; seam faithfulness needs neither a big budget
    // nor a realistic net).
    chocofarm::Loc root_loc{env.entry_point()};
    chocofarm::Belief root_bw = env.full_belief();   // the seam's belief construction entry
    chocofarm::CollectedSet root_collected;

    chocofarm::GumbelConfig small;
    small.m = 6;
    small.n_sims = 12;
    small.max_depth = 6;
    small.c_outcome = 1;
    chocofarm::GumbelConfig smaller = small;
    smaller.m = 4;
    smaller.n_sims = 8;

    std::vector<chocofarm::SearchTask> tasks;
    for (std::uint64_t seed = 1; seed <= 6; ++seed) {
        chocofarm::SearchTask t;
        t.loc = root_loc;
        t.bw = root_bw;
        t.collected = root_collected;
        t.lam = 0.1;
        t.seed = seed;
        t.cfg = (seed % 2 == 0) ? small : smaller;  // alternate the budget across tasks
        tasks.push_back(std::move(t));
    }

    // run the whole batch through SerialRuntime.
    chocofarm::SerialRuntime runtime(net);
    auto result = runtime.run(env, tasks);
    if (!result) {
        std::cerr << "serial-runtime-check: FATAL: SerialRuntime.run returned an Error: "
                  << result.error().message << "\n";
        return 1;
    }
    const std::vector<chocofarm::Decision>& decisions = *result;
    if (decisions.size() != tasks.size()) {
        std::cerr << "serial-runtime-check: FATAL: result size " << decisions.size() << " != task count "
                  << tasks.size() << "\n";
        return 4;
    }

    // the direct reference: run each task the un-wrapped way (a fresh policy + a freshly-seeded RNG),
    // and assert the EXECUTED action matches the seam's, in input order.
    int mismatches = 0;
    for (size_t i = 0; i < tasks.size(); ++i) {
        const chocofarm::SearchTask& t = tasks[i];
        chocofarm::GumbelAZPolicy direct(t.cfg, net, env);
        std::mt19937_64 rng(t.seed);
        chocofarm::Action ref = direct.decide(env, t.loc, t.bw, t.collected, t.lam, rng);

        int serial = exec_slot(env, decisions[i].executed);
        int reference = exec_slot(env, ref);
        bool ok = (decisions[i].executed == ref) && (decisions[i].leaf_requests > 0);
        if (!ok) ++mismatches;
        std::cout << "task " << i << ": serial=" << serial << " direct=" << reference
                  << " leaves=" << decisions[i].leaf_requests << " " << (ok ? "OK" : "MISMATCH") << "\n";
    }

    if (mismatches == 0) {
        std::cout << "PASS (" << tasks.size() << "/" << tasks.size() << ")\n";
        return 0;
    }
    std::cout << "FAIL (" << mismatches << " mismatches)\n";
    return 3;
}
