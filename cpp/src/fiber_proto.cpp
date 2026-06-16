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
#include <boost/context/fiber.hpp>

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

namespace ctxb = boost::context;

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

// A scripted, RNG-free Gumbel source (a fixed gumbel FIFO cycled mod its length; sample_world -> bw[0]),
// identical across the two runs by construction (mirrors gumbel_dump's scripted source).
class ScriptedGumbelSource final : public chocofarm::GumbelSource {
  public:
    explicit ScriptedGumbelSource(std::vector<double> table) : table_(std::move(table)) {}
    uint32_t sample_world(const std::vector<uint32_t>& bw) override { return bw.empty() ? 0u : bw[0]; }
    std::vector<double> gumbel(int n) override {
        std::vector<double> out(static_cast<size_t>(n));
        for (int i = 0; i < n; ++i) out[static_cast<size_t>(i)] = table_[(idx_++) % table_.size()];
        return out;
    }

  private:
    std::vector<double> table_;
    size_t idx_ = 0;
};

// The fiber<->driver channel: the yielding net writes the leaf row + a flag and yields; the driver writes
// the evaluated leaf value back and resumes.
struct YieldCtx {
    ctxb::fiber caller;                    // continuation to yield back to (updated each ping-pong)
    std::span<const float> leaf_features;  // OUT: the row predict() wants evaluated
    chocofarm::NetPrediction leaf_value;   // IN: the evaluated leaf, fed back to predict()
    bool at_leaf = false;                  // OUT: true when predict() yielded (vs the search finished)
};

// The leaf evaluator the FIBERED search holds: its predict() does NOT compute — it parks the feature row
// and YIELDS the fiber to the driver, returning the driver-supplied value on resume. To the unchanged
// search this looks like an ordinary (if slow) predict() returning a value — the whole point of Option A.
class YieldingNetEvaluator final : public chocofarm::NetEvaluator {
  public:
    explicit YieldingNetEvaluator(YieldCtx& ctx) : ctx_(ctx) {}
    std::expected<chocofarm::NetPrediction, chocofarm::Error> predict(
        std::span<const float> x) const override {
        ctx_.leaf_features = x;
        ctx_.at_leaf = true;
        ctx_.caller = std::move(ctx_.caller).resume();  // yield to the driver; resumes here when it returns
        return ctx_.leaf_value;                         // the driver set the evaluated leaf
    }

  private:
    YieldCtx& ctx_;
};

// Run `policy.run_search` (the policy already holds a YieldingNetEvaluator) inside a fiber, feeding each
// yielded leaf through `real_net` (the total DetNet). Returns the Decision the fibered search produced.
chocofarm::GumbelAZPolicy::Decision run_fibered(
    const chocofarm::GumbelAZPolicy& policy, const chocofarm::NetEvaluator& real_net,
    const chocofarm::Loc& loc, const std::vector<uint32_t>& bw, const std::set<int>& collected,
    double lam, chocofarm::GumbelSource& src, YieldCtx& ctx, int& leaves) {
    chocofarm::GumbelAZPolicy::Decision decision;
    leaves = 0;
    ctxb::fiber fib{std::allocator_arg, ctxb::fixedsize_stack(512 * 1024),
                    [&](ctxb::fiber&& caller) {
                        ctx.caller = std::move(caller);
                        decision = policy.run_search(loc, bw, collected, lam, src);
                        ctx.at_leaf = false;  // the search finished
                        return std::move(ctx.caller);
                    }};
    fib = std::move(fib).resume();  // run to the first leaf-yield (or finish)
    while (ctx.at_leaf) {
        auto pred = real_net.predict(ctx.leaf_features);  // the driver evaluates the parked leaf
        ctx.leaf_value = pred.value();                    // DetNet is total — value arm always
        ++leaves;
        fib = std::move(fib).resume();  // resume the search; on return ctx.at_leaf marks next leaf vs finish
    }
    return decision;
}

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
    std::vector<uint32_t> bw = env.worlds();
    std::set<int> collected;
    // a fixed, varied gumbel script (cycled to fill the n_slots draw) — identical for both runs.
    std::vector<double> gtable{0.40, -0.65, 1.10, 0.05, -0.30, 0.85, -1.20, 0.55,
                               0.20, -0.45, 0.95, -0.10, 0.70};

    // --- direct (synchronous) run: the reference ---
    ScriptedGumbelSource src_direct(gtable);
    chocofarm::GumbelAZPolicy direct_policy(cfg, net, env);
    chocofarm::GumbelAZPolicy::Decision direct = direct_policy.run_search(loc, bw, collected, lam, src_direct);

    // --- fibered run: the same unchanged run_search, driven through a fiber + a yielding leaf ---
    YieldCtx ctx;
    YieldingNetEvaluator ynet(ctx);
    chocofarm::GumbelAZPolicy fiber_policy(cfg, ynet, env);
    ScriptedGumbelSource src_fiber(gtable);
    int leaves = 0;
    chocofarm::GumbelAZPolicy::Decision fib =
        run_fibered(fiber_policy, net, loc, bw, collected, lam, src_fiber, ctx, leaves);

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
