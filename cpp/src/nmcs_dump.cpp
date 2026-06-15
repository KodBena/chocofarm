// cpp/src/nmcs_dump.cpp
// Purpose: a tiny PARITY tool (NOT the runner) — runs the C++ NMCSPolicy::search with a SCRIPTED,
//   deterministic WorldSource so the parity harness can feed the C++ NMCS and the Python NMCS the
//   SAME leaf playout returns on a fixed (loc, belief, collected) and assert they SELECT THE SAME
//   ACTION (ADR-0012 P6: the nesting + selection logic, the part that must be exact, validated
//   independent of RNG). It is a SEPARATE executable from the runner (P3, one-owner): the runner
//   owns the wire + episode loop, this tool owns the deterministic-logic parity fixture. No redis.
//
//   The scripted source is RNG-FREE and identical across languages by construction:
//     * sample_world(bw)  -> bw[0]  (the lowest-bitmask world; itertools/combinations order is the
//                            same on both sides, so the forward-played world is identical);
//     * playout_value(...) -> the next value from a FIFO read off stdin (consumed in call order;
//                            the recursion is identical on both sides, so the call order matches).
//   The Python reference (cpp/parity/nmcs_logic.py) monkeypatches NMCSPolicy._playout and
//   env.sample_world to the SAME FIFO + bw[0] rule, runs _search, and asserts the same first action.
//
//   ADR-0012 P9: the imperative shell. argv is decoded once into typed views; `opt` returns a
//   std::optional<std::string_view>; load_instance returns a typed std::expected reported loudly.
//   The scripted table being non-empty is the fixture's own invariant (checked at parse, then an
//   assert in playout_value) — a programmer/operator precondition, not a recoverable boundary Error.
//
//   Protocol:
//     argv: --instance <p> --faces <p> --level <int> [--cand-det N --cand-tre N --step-samples N
//           --max-steps N --lam <f> --prefix "s s s"]  (--prefix advances the real (loc,bw,coll) by a
//           deterministic slot sequence against the true world bw[0] before the search, so the fixed
//           input state can be non-trivial);
//     stdin: one line of space-separated playout values (the scripted leaf returns FIFO).
//     stdout: one line — the selected action slot index (TERMINATE = term_slot), then the search score.
//
// Public Domain (The Unlicense).
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <optional>
#include <set>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include "chocofarm/env.hpp"
#include "chocofarm/features.hpp"
#include "chocofarm/instance.hpp"
#include "chocofarm/nmcs.hpp"

namespace {
[[nodiscard]] std::optional<std::string_view> opt(std::span<const std::string_view> args,
                                                  std::string_view name) {
    for (size_t i = 1; i + 1 < args.size(); ++i)
        if (args[i] == name) return args[i + 1];
    return std::nullopt;
}
[[nodiscard]] int to_int(std::string_view s) { return std::atoi(std::string(s).c_str()); }
[[nodiscard]] double to_double(std::string_view s) { return std::atof(std::string(s).c_str()); }

// The scripted, RNG-free world source: sample_world -> bw[0]; playout_value -> the next value off a
// fixed table consumed in CALL ORDER and CYCLED modulo its length (so a level-2 search, which can
// consume far more leaf values than the table holds, never exhausts it — the Python reference cycles
// the SAME table the SAME way, so the value delivered at each call index is identical on both sides).
class ScriptedSource final : public chocofarm::NMCSWorldSource {
  public:
    explicit ScriptedSource(std::vector<double> vals) : vals_(std::move(vals)) {}
    uint32_t sample_world(const std::vector<uint32_t>& bw) override { return bw[0]; }
    double playout_value(const chocofarm::Loc&, const std::vector<uint32_t>&,
                         const std::set<int>&, double) override {
        // The fixture guarantees a non-empty table (checked in main before constructing the source);
        // an empty table here would be a programmer bug (ADR-0012 P9: an invariant, an assert).
        assert(!vals_.empty() && "nmcs_dump: empty scripted playout table");
        return vals_[(idx_++) % vals_.size()];
    }

  private:
    std::vector<double> vals_;
    size_t idx_ = 0;
};
}  // namespace

int main(int argc, char** argv) {
    std::vector<std::string_view> args(argv, argv + argc);
    std::optional<std::string_view> instance = opt(args, "--instance");
    std::optional<std::string_view> faces = opt(args, "--faces");
    if (!instance || !faces) {
        std::cerr << "usage: nmcs-dump --instance <p> --faces <p> [--level N --cand-det N "
                     "--cand-tre N --step-samples N --max-steps N --lam f --prefix \"s s\"] "
                     "(playout values on stdin)\n";
        return 2;
    }

    chocofarm::NMCSConfig cfg;
    cfg.level = opt(args, "--level") ? to_int(*opt(args, "--level")) : 1;
    cfg.cand_det = opt(args, "--cand-det") ? to_int(*opt(args, "--cand-det")) : 4;
    cfg.cand_tre = opt(args, "--cand-tre") ? to_int(*opt(args, "--cand-tre")) : 4;
    cfg.step_samples = opt(args, "--step-samples") ? to_int(*opt(args, "--step-samples")) : 2;
    cfg.max_steps = opt(args, "--max-steps") ? to_int(*opt(args, "--max-steps")) : 24;
    // playout_samples is irrelevant here (the scripted source ignores it), but kept default.
    double lam = opt(args, "--lam") ? to_double(*opt(args, "--lam")) : 0.1;

    auto inst = chocofarm::load_instance(*instance, *faces);
    if (!inst) { std::cerr << "nmcs-dump: FATAL: " << inst.error().message << "\n"; return 1; }
    chocofarm::Environment env(*inst);
    chocofarm::NMCSPolicy policy(cfg);

    chocofarm::Loc loc{env.entry_point()};
    std::vector<uint32_t> bw = env.worlds();
    std::set<int> collected;

    // optionally advance the real (loc, bw, collected) by a prefix slot sequence against the
    // true world bw[0] (the same deterministic world both languages advance by), so the fixed
    // search input can be a mid-episode state, not just the root.
    if (auto pref = opt(args, "--prefix")) {
        uint32_t world = bw.empty() ? 0u : bw[0];
        std::istringstream iss{std::string(*pref)};
        int slot;
        while (iss >> slot) {
            if (bw.empty()) break;
            if (slot >= env.N() + env.n_detectors()) break;  // TERMINATE in prefix: stop
            chocofarm::Action a = (slot < env.N())
                ? chocofarm::Action{chocofarm::ActionKind::Treasure, slot}
                : chocofarm::Action{chocofarm::ActionKind::Detector, slot - env.N()};
            env.apply(loc, bw, collected, a, world);
        }
    }

    // read the scripted playout values (FIFO) from stdin
    std::vector<double> vals;
    {
        std::string line;
        std::getline(std::cin, line);
        std::istringstream iss(line);
        double v;
        while (iss >> v) vals.push_back(v);
    }
    if (vals.empty()) {
        std::cerr << "nmcs-dump: FATAL: empty scripted playout table on stdin\n";
        return 1;
    }
    ScriptedSource src(std::move(vals));

    int level = std::max(1, cfg.level);
    auto [score, action] = policy.search(env, loc, bw, collected, lam, level, src);

    int slot;
    if (action.kind == chocofarm::ActionKind::Terminate) slot = chocofarm::term_slot(env);
    else if (action.kind == chocofarm::ActionKind::Treasure) slot = action.i;
    else slot = env.N() + action.i;

    std::cout.precision(17);
    std::cout << slot << " " << score << "\n";
    return 0;
}
