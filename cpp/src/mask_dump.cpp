// cpp/src/mask_dump.cpp
// Purpose: a tiny PARITY tool (NOT the runner) — replays a deterministic action sequence through
//   the C++ env and prints the legality mask M at each step, so the parity harness can assert the
//   C++ mask is BIT-IDENTICAL to Python's legal_mask for the SAME (loc, belief) (ADR-0012 P6/P7:
//   the mask is a logic invariant float32 cannot perturb). It is kept a SEPARATE executable from
//   the runner (P3, one-owner): the runner's job is the wire, this tool's job is the mask-replay
//   parity fixture; neither carries the other's concern. No redis: pure env + features.
//
//   Protocol: argv gives --instance / --faces / --world <uint32>; stdin gives one line of
//   space-separated slot indices (the action sequence, TERMINATE = the last slot ends it). For each
//   action, BEFORE applying it, print the mask as `n_slots` space-separated 0/1 values on one line;
//   then apply the action (advancing the belief) so the next mask reflects the new state.
//
// Public Domain (The Unlicense).
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "chocofarm/env.hpp"
#include "chocofarm/features.hpp"
#include "chocofarm/instance.hpp"

static const char* opt(int argc, char** argv, const char* name) {
    for (int i = 1; i + 1 < argc; ++i)
        if (std::strcmp(argv[i], name) == 0) return argv[i + 1];
    return nullptr;
}

static bool has_flag(int argc, char** argv, const char* name) {
    for (int i = 1; i < argc; ++i)
        if (std::strcmp(argv[i], name) == 0) return true;
    return false;
}

int main(int argc, char** argv) {
    const char* instance = opt(argc, argv, "--instance");
    const char* faces = opt(argc, argv, "--faces");
    const char* world_s = opt(argc, argv, "--world");
    if (!instance || !faces || !world_s) {
        std::cerr << "usage: mask-dump --instance <p> --faces <p> --world <uint32>  (seq on stdin)\n";
        return 2;
    }
    uint32_t world = static_cast<uint32_t>(std::strtoul(world_s, nullptr, 10));

    const bool dump_feats = has_flag(argc, argv, "--features");

    chocofarm::Instance inst = chocofarm::load_instance(instance, faces);
    chocofarm::Environment env(inst);
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
    std::vector<uint32_t> bw = env.worlds();
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
        if (bw.empty()) break;
        print_mask();                       // mask BEFORE applying (the (loc, belief) at this step)
        if (slot == term) break;            // TERMINATE: no step
        chocofarm::Action a;
        if (slot < env.N()) a = chocofarm::Action{chocofarm::ActionKind::Treasure, slot};
        else a = chocofarm::Action{chocofarm::ActionKind::Detector, slot - env.N()};
        env.apply(loc, bw, collected, a, world);
    }
    return 0;
}
