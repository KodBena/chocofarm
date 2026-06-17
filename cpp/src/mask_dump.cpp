// cpp/src/mask_dump.cpp
// Purpose: a tiny PARITY tool (NOT the runner) — replays a deterministic action sequence through
//   the C++ env and prints the legality mask M at each step, so the parity harness can assert the
//   C++ mask is BIT-IDENTICAL to Python's legal_mask for the SAME (loc, belief) (ADR-0012 P6/P7:
//   the mask is a logic invariant float32 cannot perturb). It is kept a SEPARATE executable from
//   the runner (P3, one-owner): the runner's job is the wire, this tool's job is the mask-replay
//   parity fixture; neither carries the other's concern. No redis: pure env + features.
//
//   ADR-0012 P9: the imperative shell. argv is decoded once into typed views; `opt` returns a
//   std::optional<std::string_view>; load_instance returns a typed std::expected reported loudly.
//
//   Protocol: argv gives --instance / --faces / --world <uint32>; stdin gives one line of
//   space-separated slot indices (the action sequence, TERMINATE = the last slot ends it). For each
//   action, BEFORE applying it, print the mask as `n_slots` space-separated 0/1 values on one line;
//   then apply the action (advancing the belief) so the next mask reflects the new state.
//
// Public Domain (The Unlicense).
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

namespace {
[[nodiscard]] std::optional<std::string_view> opt(std::span<const std::string_view> args,
                                                  std::string_view name) {
    for (size_t i = 1; i + 1 < args.size(); ++i)
        if (args[i] == name) return args[i + 1];
    return std::nullopt;
}
[[nodiscard]] bool has_flag(std::span<const std::string_view> args, std::string_view name) {
    for (size_t i = 1; i < args.size(); ++i)
        if (args[i] == name) return true;
    return false;
}
}  // namespace

int main(int argc, char** argv) {
    std::vector<std::string_view> args(argv, argv + argc);
    std::optional<std::string_view> instance = opt(args, "--instance");
    std::optional<std::string_view> faces = opt(args, "--faces");
    std::optional<std::string_view> world_s = opt(args, "--world");
    if (!instance || !faces || !world_s) {
        std::cerr << "usage: mask-dump --instance <p> --faces <p> --world <uint32>  (seq on stdin)\n";
        return 2;
    }
    uint32_t world = static_cast<uint32_t>(std::strtoul(std::string(*world_s).c_str(), nullptr, 10));

    const bool dump_feats = has_flag(args, "--features");

    auto inst = chocofarm::load_instance(*instance, *faces);
    if (!inst) { std::cerr << "mask-dump: FATAL: " << inst.error().message << "\n"; return 1; }
    chocofarm::Environment env(*inst);
    chocofarm::FeatureBuilder fb(env);
    const int n_slots = chocofarm::n_action_slots(env);
    const int term = chocofarm::term_slot(env);

    // read the action sequence (slot indices) from stdin
    std::vector<int> seq;
    {
        std::string line;
        std::getline(std::cin, line);
        std::istringstream iss(line);
        int s;
        while (iss >> s) seq.push_back(s);
    }

    chocofarm::Loc loc{env.entry_point()};
    chocofarm::Belief bw = env.full_belief();   // the seam's belief construction entry
    std::set<int> collected;

    std::cout.precision(17);
    auto print_mask = [&]() {
        if (dump_feats) {
            // emit the §2.2 feature vector (float64, full precision) for X-port parity
            std::vector<double> f = fb.build(loc.pt, bw, collected);
            for (size_t i = 0; i < f.size(); ++i) std::cout << (i ? " " : "") << f[i];
        } else {
            std::vector<float> m = chocofarm::legal_mask(env, bw, collected);
            for (int i = 0; i < n_slots; ++i) std::cout << (i ? " " : "") << static_cast<int>(m[i]);
        }
        std::cout << "\n";
    };

    for (int slot : seq) {
        if (env.empty(bw)) break;
        print_mask();                       // mask BEFORE applying (the (loc, belief) at this step)
        if (slot == term) break;            // TERMINATE: no step
        chocofarm::Action a;
        if (slot < env.N()) a = chocofarm::Action{chocofarm::ActionKind::Treasure, slot};
        else a = chocofarm::Action{chocofarm::ActionKind::Detector, slot - env.N()};
        env.apply(loc, bw, collected, a, world);
    }
    return 0;
}
