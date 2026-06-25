// cpp/src/gumbel_cursor_proto.cpp
// Purpose: the OPTION-B foundation proof (NOT the runner) — drive the Gumbel-AZ search through the
//   explicit-state TreeCursor (gumbel_cursor.hpp) and assert the result is BIT-IDENTICAL to a DIRECT
//   synchronous run_search fed the same scripted leaves + draws. This is the Option-B analogue of
//   fiber_proto.cpp's Option-A proof (fiber-driven ≡ direct): it is the docs/design/cpp-search-runtime.md
//   §7.1 validity precondition for B — the explicit-cursor reification preserved the per-tree RNG draw
//   order, the four 1b precision seams, and the Danihelka invariants (re-entry resumes at exactly the draw
//   the recursion was about to make), so the cursor must produce the SAME decision the recursion does.
//
//   Unlike Option A (where there is NO refactor — the fiber only changes WHEN predict returns), Option B
//   IS a refactor of the recursion into an explicit cursor, so this proof is load-bearing: it is the
//   witness that the refactor did not perturb the search. It asserts the executed action, the improved-π
//   argmax, n_spent, the survivor slot, AND the full per-leaf REQUEST SEQUENCE (the sequence of feature
//   rows the two paths forward, in order) are identical — the strongest structural-determinism check
//   (§7.2 layer 2: the leaf-request count + order is the discriminator that the scheduling/reification
//   did not change the search).
//
//   The leaf + RNG are deterministic + scripted (a DetNet — a pure function of the features; a
//   CyclicGumbelSource), so the two runs (direct vs cursor) see identical leaves + draws and MUST agree
//   exactly. No redis, no weights, no real net. (Same fixture shape as fiber_proto.cpp.)
//
//   ADR-0012 P9: the imperative shell. The cursor's advance/resume are the functional core (total
//   value-functions returning a Step by value, no I/O, no throw); the leaf forward + the resume loop are
//   the shell here. Bit-identity is achieved by REUSE not re-derivation — the cursor calls the policy's
//   validated precision helpers (ADR-0000: the parked-search state is a representable typed value).
//
//   Protocol:  gumbel-cursor-proto --instance <p> --faces <p> [--m N --n-sims N --max-depth N
//                                   --c-outcome N --lam f]
//   Output:    "RESULT: PASS (executed/argmax/n_spent/survivor + leaf-sequence identical, <k> leaves)"
//              + exit 0, or "RESULT: FAIL ..." + exit 3.
//
// Public Domain (The Unlicense).
#include <cmath>
#include <cstdint>
#include <iostream>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

#include "chocofarm/cyclic_gumbel.hpp"
#include "chocofarm/env.hpp"
#include "chocofarm/features.hpp"
#include "chocofarm/gumbel.hpp"
#include "chocofarm/gumbel_cursor.hpp"
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

// The SAME total deterministic leaf fiber_proto.cpp / leaf_cpu_microbench.cpp use (a pure function of the
// features) — so the direct run and the cursor run see byte-identical leaf values for the same belief.
// To capture the per-leaf REQUEST SEQUENCE it also records each feature row's checksum (a sum of the
// row), so the two paths' leaf streams can be compared element-by-element, not just by count.
class DetNet final : public chocofarm::NetEvaluator {
  public:
    explicit DetNet(int n_slots) : n_slots_(n_slots) {}
    std::expected<chocofarm::NetPrediction, chocofarm::Error> predict(
        std::span<const float> x) const override {
        double s = 0.0;
        for (float v : x) s += static_cast<double>(v);
        seq_.push_back(s);  // record the leaf-request checksum in call order (the request sequence)
        chocofarm::NetPrediction p;
        p.value = static_cast<float>(0.01 * s);
        p.logits.resize(static_cast<size_t>(n_slots_));
        for (int i = 0; i < n_slots_; ++i)
            p.logits[static_cast<size_t>(i)] =
                static_cast<float>(std::sin(0.5 * static_cast<double>(i) + 0.001 * s));
        return p;
    }
    [[nodiscard]] const std::vector<double>& sequence() const { return seq_; }
    void reset() { seq_.clear(); }

  private:
    int n_slots_;
    mutable std::vector<double> seq_;  // per-leaf feature-row checksum, in call order
};

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
        std::cerr << "usage: gumbel-cursor-proto --instance <p> --faces <p> [--m N --n-sims N "
                     "--max-depth N --c-outcome N --lam f]\n";
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
        std::cerr << "gumbel-cursor-proto: FATAL: " << inst.error().message << "\n";
        return 1;
    }
    chocofarm::Environment env(*inst);
    // n_action_slots returns the typed SlotCount; DetNet (a proto-local test net) sizes its logit vector
    // from a raw int — unwrap at this crossing (ADR-0000 item 5: a named, visible raw<->domain crossing).
    DetNet net(static_cast<int>(chocofarm::n_action_slots(env).value()));

    chocofarm::Loc loc{env.entry_point()};
    chocofarm::Belief bw = env.full_belief();
    chocofarm::CollectedSet collected;
    std::vector<double> gtable{0.40, -0.65, 1.10, 0.05, -0.30, 0.85, -1.20, 0.55,
                               0.20, -0.45, 0.95, -0.10, 0.70};

    // --- direct (synchronous) run: the reference ---
    chocofarm::CyclicGumbelSource src_direct(env, gtable);
    chocofarm::GumbelAZPolicy direct_policy(cfg, net, env);
    net.reset();
    chocofarm::GumbelAZPolicy::Decision direct =
        direct_policy.run_search(loc, bw, collected, lam, src_direct);
    const std::vector<double> direct_seq = net.sequence();  // the direct leaf-request sequence

    // --- cursor run: the SAME search through the explicit-state TreeCursor, feeding each parked leaf
    //     through the SAME total DetNet. ---
    chocofarm::CyclicGumbelSource src_cursor(env, gtable);
    chocofarm::GumbelAZPolicy cursor_policy(cfg, net, env);
    net.reset();
    chocofarm::TreeCursor cur(cursor_policy, loc, bw, collected, lam, src_cursor);
    int leaves = 0;
    chocofarm::Step st = cur.advance();
    while (std::holds_alternative<chocofarm::CursorNeedsLeaf>(st)) {
        const auto& need = std::get<chocofarm::CursorNeedsLeaf>(st);
        auto pred = net.predict(need.features);  // DetNet is total → the value arm always
        st = cur.resume(pred.value());
        ++leaves;
    }
    const chocofarm::GumbelAZPolicy::Decision cdec = std::get<chocofarm::CursorDecided>(st).decision;
    const std::vector<double> cursor_seq = net.sequence();  // the cursor leaf-request sequence

    // --- compare: executed action, improved-π argmax, n_spent, survivor slot, AND the leaf sequence ---
    const bool exec_ok = (direct.action == cdec.action);
    const bool argmax_ok = (argmax(direct.improved) == argmax(cdec.improved));
    const bool nspent_ok = (direct.n_spent == cdec.n_spent);
    const bool survivor_ok = (direct.survivor_slot == cdec.survivor_slot);
    bool seq_ok = (direct_seq.size() == cursor_seq.size());
    if (seq_ok)
        for (size_t i = 0; i < direct_seq.size(); ++i)
            if (direct_seq[i] != cursor_seq[i]) { seq_ok = false; break; }

    std::cout << "direct:  survivor_slot=" << direct.survivor_slot << " argmax=" << argmax(direct.improved)
              << " n_spent=" << direct.n_spent << " leaves=" << direct_seq.size() << "\n";
    std::cout << "cursor:  survivor_slot=" << cdec.survivor_slot << " argmax=" << argmax(cdec.improved)
              << " n_spent=" << cdec.n_spent << " leaves=" << cursor_seq.size() << " (driven via cursor="
              << leaves << ")\n";

    if (exec_ok && argmax_ok && nspent_ok && survivor_ok && seq_ok) {
        std::cout << "RESULT: PASS (executed/argmax/n_spent/survivor + leaf-request sequence identical, "
                  << leaves << " leaves driven through the explicit-state cursor)\n";
        return 0;
    }
    std::cout << "RESULT: FAIL (exec_ok=" << exec_ok << " argmax_ok=" << argmax_ok
              << " nspent_ok=" << nspent_ok << " survivor_ok=" << survivor_ok << " seq_ok=" << seq_ok
              << ")\n";
    return 3;
}
