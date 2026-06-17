// cpp/src/fiber_proto.cpp
// Purpose: the Option-A foundation proof (NOT the runner) — run the UNCHANGED GumbelAZPolicy::run_search
//   inside a boost.context stackful fiber, with a YieldingNetEvaluator whose predict() YIELDS the fiber
//   at each leaf and is resumed with the leaf value, and assert the result is BIT-IDENTICAL to a direct
//   synchronous run_search fed the same leaves. This is item 1a of the continuation-refactor decision
//   (docs/notes/cpp-continuation-refactor-decision-2026-06-16.md): Option A makes the search resumable
//   for the wire-parallel work-stealing pool WITHOUT touching the 1a/1b-validated search — the fiber's
//   stack holds the recursion, the injected yielding net does the suspension, and the search code is
//   oblivious. Proving fiber-driven ≡ direct here is the §7.1 validity precondition made near-trivial
//   (there is no refactor — only WHEN predict returns changes, not WHAT it returns).
//
//   The leaf and the RNG are deterministic + scripted (a DetNet — a pure function of the features; a
//   scripted GumbelSource), so the two runs (direct vs fibered) see identical leaves + draws and MUST
//   agree exactly. No redis, no weights, no real net.
//
//   ADR-0012 P9: the imperative shell. The fiber + the yield ARE the effect, confined to this driver and
//   the YieldingNetEvaluator; the search core stays a pure value-function of its inputs. The yielding
//   net's predict returns the value arm (the driver feeds a total DetNet leaf); a real remote leaf's
//   typed failure routes through the same expected the port already carries.
//
//   Protocol:  fiber-proto --instance <p> --faces <p> [--m N --n-sims N --max-depth N --c-outcome N --lam f]
//   Output:    "RESULT: PASS (executed/argmax/n_spent identical, <k> leaves via fiber)" + exit 0, or
//              "RESULT: FAIL ..." + exit 3.
//
// Public Domain (The Unlicense).
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
#include "chocofarm/features.hpp"
#include "chocofarm/fiber_tree.hpp"
#include "chocofarm/gumbel.hpp"
#include "chocofarm/instance.hpp"
#include "chocofarm/net_evaluator.hpp"

namespace {
[[nodiscard]] std::optional<std::string_view> opt(std::span<const std::string_view> args,
                                                  std::string_view name) {
    for (size_t i = 1; i + 1 < args.size(); ++i)
        if (args[i] == name) return args[i + 1];
    return std::nullopt;
}
[[nodiscard]] int to_int(std::string_view s) { return std::atoi(std::string(s).c_str()); }
[[nodiscard]] double to_double(std::string_view s) { return std::atof(std::string(s).c_str()); }

// A deterministic, stateless leaf (a pure function of the features) — so the direct run and the fibered
// run see byte-identical leaf values for the same belief.
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

// The scripted Gumbel source + the fiber-leaf primitives + the per-tree fiber state are now the ONE-home
// shared types (ADR-0012 P1): CyclicGumbelSource (cyclic_gumbel.hpp), FiberLeafChannel +
// YieldingNetEvaluator (fiber_leaf.hpp), TreeState (fiber_tree.hpp). Driving this proof through the SAME
// TreeState the wire benches multiplex means it validates the real shared primitive, not a proof-only
// copy — the §7.1 validity precondition now bears directly on the type the pool/parallel benches run.

[[nodiscard]] int argmax(const std::vector<double>& v) {
    int best = 0;
    double bv = v.empty() ? 0.0 : v[0];
    for (int i = 1; i < static_cast<int>(v.size()); ++i)
        if (v[static_cast<size_t>(i)] > bv) { bv = v[static_cast<size_t>(i)]; best = i; }
    return best;
}
}  // namespace

int main(int argc, char** argv) {
    std::vector<std::string_view> args(argv, argv + argc);
    std::optional<std::string_view> instance = opt(args, "--instance");
    std::optional<std::string_view> faces = opt(args, "--faces");
    if (!instance || !faces) {
        std::cerr << "usage: fiber-proto --instance <p> --faces <p> [--m N --n-sims N --max-depth N "
                     "--c-outcome N --lam f]\n";
        return 2;
    }
    const double lam = opt(args, "--lam") ? to_double(*opt(args, "--lam")) : 0.1;
    chocofarm::GumbelConfig cfg;
    cfg.n_sims = 24;
    cfg.max_depth = 8;
    if (auto v = opt(args, "--m")) cfg.m = to_int(*v);
    if (auto v = opt(args, "--n-sims")) cfg.n_sims = to_int(*v);
    if (auto v = opt(args, "--max-depth")) cfg.max_depth = to_int(*v);
    if (auto v = opt(args, "--c-outcome")) cfg.c_outcome = to_int(*v);

    auto inst = chocofarm::load_instance(*instance, *faces);
    if (!inst) {
        std::cerr << "fiber-proto: FATAL: " << inst.error().message << "\n";
        return 1;
    }
    chocofarm::Environment env(*inst);
    DetNet net(chocofarm::n_action_slots(env));

    chocofarm::Loc loc{env.entry_point()};
    chocofarm::Belief bw = env.full_belief();   // the seam's belief construction entry
    chocofarm::CollectedSet collected;
    // a fixed, varied gumbel script (cycled to fill the n_slots draw) — identical for both runs.
    std::vector<double> gtable{0.40, -0.65, 1.10, 0.05, -0.30, 0.85, -1.20, 0.55,
                               0.20, -0.45, 0.95, -0.10, 0.70};

    // --- direct (synchronous) run: the reference ---
    chocofarm::CyclicGumbelSource src_direct(env, gtable);
    chocofarm::GumbelAZPolicy direct_policy(cfg, net, env);
    chocofarm::GumbelAZPolicy::Decision direct = direct_policy.run_search(loc, bw, collected, lam, src_direct);

    // --- fibered run: the SAME unchanged run_search, driven through the shared TreeState (a fiber + the
    //     yielding leaf), feeding each parked leaf through the total DetNet. A stack local that never moves
    //     (the fiber captures `this`); one tree, so no heap/vector is needed. ---
    chocofarm::TreeState ts(cfg, env, gtable);
    ts.start(loc, bw, collected, lam);  // advance to the first parked leaf (or finish)
    int leaves = 0;
    while (ts.running) {
        auto pred = net.predict(ts.ch.features);  // DetNet is total → the value arm always
        ts.resume_with(pred.value());
        ++leaves;
    }
    const chocofarm::GumbelAZPolicy::Decision fib = ts.decision;

    // --- compare: executed action slot, improved-pi argmax, n_spent ---
    const bool exec_ok = (direct.action == fib.action);
    const bool argmax_ok = (argmax(direct.improved) == argmax(fib.improved));
    const bool nspent_ok = (direct.n_spent == fib.n_spent);
    std::cout << "direct:  survivor_slot=" << direct.survivor_slot << " argmax=" << argmax(direct.improved)
              << " n_spent=" << direct.n_spent << "\n";
    std::cout << "fibered: survivor_slot=" << fib.survivor_slot << " argmax=" << argmax(fib.improved)
              << " n_spent=" << fib.n_spent << " (leaves via fiber=" << leaves << ")\n";

    if (exec_ok && argmax_ok && nspent_ok) {
        std::cout << "RESULT: PASS (executed/argmax/n_spent identical, " << leaves
                  << " leaves driven through the fiber)\n";
        return 0;
    }
    std::cout << "RESULT: FAIL (exec_ok=" << exec_ok << " argmax_ok=" << argmax_ok
              << " nspent_ok=" << nspent_ok << ")\n";
    return 3;
}
