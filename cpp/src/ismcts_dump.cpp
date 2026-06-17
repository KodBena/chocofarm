// cpp/src/ismcts_dump.cpp
// Purpose: a tiny PARITY tool (NOT the runner) — runs the C++ ISMCTSPolicy::run_search with a
//   SCRIPTED, deterministic ISMCTSSource so the parity harness can feed the C++ ISMCTS and the
//   Python ISMCTS the SAME world draws, expansion-index draws, AND leaf returns on a fixed
//   (loc, belief, collected) and assert they SELECT THE SAME ACTION (ADR-0012 P6: the selection +
//   nesting logic, the part that must be exact, validated independent of RNG). It is a SEPARATE
//   executable from the runner (P3, one-owner): the runner owns the wire + episode loop, this tool
//   owns the deterministic-logic parity fixture. No redis.
//
//   The scripted source is RNG-FREE and identical across languages by construction:
//     * sample_world(bw)   -> bw[0]  (the lowest-bitmask world; itertools/combinations order is the
//                             same on both sides, so each iteration's determinization is identical);
//     * expand_index(n)    -> the next value from an EXPANSION-INDEX FIFO, taken modulo n (so a
//                             scripted index is always in [0, n); both sides apply the SAME mod n at
//                             the SAME call, so they expand the SAME untried action);
//     * leaf_value(...)    -> the next value from a LEAF FIFO, consumed in call order (the recursion
//                             is identical on both sides, so the call order matches).
//   The Python reference (cpp/parity/ismcts_logic.py) monkeypatches env.sample_world (-> bw[0]),
//   rng.integers (-> the SAME expansion-index FIFO mod n) and _base_value (-> the SAME leaf FIFO),
//   runs decide, and asserts the same selected action.
//
//   Both FIFOs are CYCLED modulo their length (a 300-iteration search consumes far more draws than a
//   small table holds; the Python reference cycles the SAME tables the SAME way, so the value
//   delivered at each call index is identical on both sides).
//
//   ADR-0012 P9: the imperative shell. argv is decoded once into typed views; `opt` returns a
//   std::optional<std::string_view>; load_instance returns a typed std::expected reported loudly.
//   The scripted tables being non-empty is the fixture's own invariant (checked at parse, then an
//   assert in the source) — a programmer/operator precondition, not a recoverable boundary Error.
//
//   Protocol:
//     argv: --instance <p> --faces <p> [--iterations N --max-depth N --c <f> --lam <f>
//           --prefix "s s s"]  (--prefix advances the real (loc,bw,coll) by a deterministic slot
//           sequence against the true world bw[0] before the search, so the fixed input state can be
//           non-trivial);
//     stdin: line 1 = space-separated expansion indices (ints, the expand-index FIFO);
//            line 2 = space-separated leaf values (doubles, the leaf FIFO).
//     stdout: one line — the selected action slot index (TERMINATE = term_slot).
//
// Public Domain (The Unlicense).
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
#include "chocofarm/ismcts.hpp"

namespace {
[[nodiscard]] std::optional<std::string_view> opt(std::span<const std::string_view> args,
                                                  std::string_view name) {
    for (size_t i = 1; i + 1 < args.size(); ++i)
        if (args[i] == name) return args[i + 1];
    return std::nullopt;
}
[[nodiscard]] int to_int(std::string_view s) { return std::atoi(std::string(s).c_str()); }
[[nodiscard]] double to_double(std::string_view s) { return std::atof(std::string(s).c_str()); }

// The scripted, RNG-free ISMCTS source. Three deterministic draws, each identical across languages:
//   sample_world -> bw[0]; expand_index(n) -> the next expand-index value mod n; leaf_value -> the
// next leaf value. Both FIFOs are consumed in CALL ORDER and CYCLED modulo their length.
class ScriptedISMCTSSource final : public chocofarm::ISMCTSSource {
  public:
    // The scripted source threads `const Environment&` so it resolves its `bw[idx]` pokes through the
    // seam (env.world_at_rank, L4) — byte-identical to the former direct `bw[idx]` (the flat belief is a
    // worlds()-RANK-ordered subsequence, so the r-th element == the r-th rank).
    ScriptedISMCTSSource(const chocofarm::Environment& env, std::vector<int> idxs,
                         std::vector<double> leaves, std::vector<int> world_idxs = {})
        : env_(env), idxs_(std::move(idxs)), leaves_(std::move(leaves)),
          world_idxs_(std::move(world_idxs)) {}

    // sample_world: an EMPTY world-index FIFO reproduces the original collapsed behavior (bw[0]),
    // keeping cpp/parity/ismcts_logic.py byte-compatible. A NON-EMPTY FIFO cycles distinct worlds of
    // bw by scripted index (bw[idx mod bw.size()]) so the SAME action at a node resolves to DIFFERENT
    // observation outcomes across iterations -> multiple (action, belief_key) children: the
    // multi-determinization sub-child split (the ISMCTS-defining property) is then exercised. The
    // Python multi-world fixture pops the SAME world-index FIFO against the SAME bw ordering, so the
    // determinizations are identical on both sides. The `bw[idx]` poke routes through env.world_at_rank
    // (L4), preserving the EXACT index (rank 0, or raw % nb).
    uint32_t sample_world(const chocofarm::Belief& bw) override {
        if (world_idxs_.empty()) return env_.world_at_rank(bw, 0);
        int raw = world_idxs_[(widx_++) % world_idxs_.size()];
        int n = env_.nb(bw);
        return env_.world_at_rank(bw, ((raw % n) + n) % n);  // non-negative modulo; legal rank into bw
    }

    int expand_index(int n) override {
        // The fixture guarantees a non-empty index table (checked in main); an empty one here would
        // be a programmer bug (ADR-0012 P9: an invariant, an assert). The scripted value is reduced
        // mod n so it is always a legal untried-list index (the Python reference applies the SAME
        // mod n at the SAME call, so both sides expand the SAME action).
        assert(!idxs_.empty() && "ismcts_dump: empty scripted expand-index table");
        int raw = idxs_[(iidx_++) % idxs_.size()];
        int m = ((raw % n) + n) % n;  // non-negative modulo (a scripted index may be authored >= n)
        return m;
    }

    double leaf_value(const chocofarm::Loc&, const chocofarm::Belief&, const std::set<int>&,
                      uint32_t, double) override {
        assert(!leaves_.empty() && "ismcts_dump: empty scripted leaf table");
        return leaves_[(lidx_++) % leaves_.size()];
    }

  private:
    const chocofarm::Environment& env_;
    std::vector<int> idxs_;
    std::vector<double> leaves_;
    std::vector<int> world_idxs_;   // OPTIONAL world-index FIFO (empty -> sample_world = bw[0])
    size_t iidx_ = 0;
    size_t lidx_ = 0;
    size_t widx_ = 0;
};
}  // namespace

int main(int argc, char** argv) {
    std::vector<std::string_view> args(argv, argv + argc);
    std::optional<std::string_view> instance = opt(args, "--instance");
    std::optional<std::string_view> faces = opt(args, "--faces");
    if (!instance || !faces) {
        std::cerr << "usage: ismcts-dump --instance <p> --faces <p> [--iterations N --max-depth N "
                     "--c f --lam f --prefix \"s s\"] "
                     "(expand-index FIFO on stdin line 1, leaf FIFO on line 2)\n";
        return 2;
    }

    chocofarm::ISMCTSConfig cfg;
    cfg.iterations = opt(args, "--iterations") ? to_int(*opt(args, "--iterations")) : 300;
    cfg.max_depth = opt(args, "--max-depth") ? to_int(*opt(args, "--max-depth")) : 24;
    cfg.c = opt(args, "--c") ? to_double(*opt(args, "--c")) : chocofarm::UCB_C;
    double lam = opt(args, "--lam") ? to_double(*opt(args, "--lam")) : 0.1;

    auto inst = chocofarm::load_instance(*instance, *faces);
    if (!inst) { std::cerr << "ismcts-dump: FATAL: " << inst.error().message << "\n"; return 1; }
    chocofarm::Environment env(*inst);
    chocofarm::ISMCTSPolicy policy(cfg);

    chocofarm::Loc loc{env.entry_point()};
    chocofarm::Belief bw = env.full_belief();   // the seam's belief construction entry
    std::set<int> collected;

    // optionally advance the real (loc, bw, collected) by a prefix slot sequence against the true
    // world bw[0] (the same deterministic world both languages advance by), so the fixed search input
    // can be a mid-episode state, not just the root.
    if (auto pref = opt(args, "--prefix")) {
        uint32_t world = env.empty(bw) ? 0u : env.world_at_rank(bw, 0);  // rank-0 world (L4)
        std::istringstream iss{std::string(*pref)};
        int slot;
        while (iss >> slot) {
            if (env.empty(bw)) break;
            if (slot >= env.N() + env.n_detectors()) break;  // TERMINATE in prefix: stop
            chocofarm::Action a = (slot < env.N())
                ? chocofarm::Action{chocofarm::ActionKind::Treasure, slot}
                : chocofarm::Action{chocofarm::ActionKind::Detector, slot - env.N()};
            env.apply(loc, bw, collected, a, world);
        }
    }

    // read the two scripted FIFOs from stdin: line 1 = expand indices, line 2 = leaf values.
    std::vector<int> idxs;
    std::vector<double> leaves;
    {
        std::string line;
        std::getline(std::cin, line);
        std::istringstream iss(line);
        int v;
        while (iss >> v) idxs.push_back(v);
    }
    {
        std::string line;
        std::getline(std::cin, line);
        std::istringstream iss(line);
        double v;
        while (iss >> v) leaves.push_back(v);
    }
    // OPTIONAL line 3: the world-index FIFO. Absent/empty -> sample_world = bw[0] (back-compatible
    // with ismcts_logic.py). Present -> cycle bw[idx mod bw.size()] to exercise the multi-belief split.
    std::vector<int> world_idxs;
    {
        std::string line;
        if (std::getline(std::cin, line)) {
            std::istringstream iss(line);
            int v;
            while (iss >> v) world_idxs.push_back(v);
        }
    }
    if (idxs.empty() || leaves.empty()) {
        std::cerr << "ismcts-dump: FATAL: need a non-empty expand-index FIFO (line 1) AND a "
                     "non-empty leaf FIFO (line 2) on stdin\n";
        return 1;
    }
    ScriptedISMCTSSource src(env, std::move(idxs), std::move(leaves), std::move(world_idxs));

    chocofarm::Action action = policy.run_search(env, loc, bw, collected, lam, src);

    int slot;
    if (action.kind == chocofarm::ActionKind::Terminate) slot = chocofarm::term_slot(env);
    else if (action.kind == chocofarm::ActionKind::Treasure) slot = action.i;
    else slot = env.N() + action.i;

    std::cout << slot << "\n";
    return 0;
}
