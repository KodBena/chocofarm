// cpp/src/gumbel_dump.cpp
// Purpose: a tiny PARITY tool (NOT the runner) — runs the C++ GumbelAZPolicy::run_search with a
//   SCRIPTED, RNG-free NetEvaluator leaf + a SCRIPTED, RNG-free GumbelSource so the parity harness can
//   feed the C++ Gumbel search and the Python Gumbel search the SAME gumbel/world draws + the SAME
//   coarse, well-separated scripted leaf (value, logits) on a fixed (loc, belief, collected) and assert
//   they EXECUTE THE SAME ACTION and produce the SAME improved-π argmax (ADR-0012 P6: the discrete
//   structure + selection logic, the part that must be exact, validated independent of RNG). It is a
//   SEPARATE executable from the runner (P3, one-owner). No redis (the scripted leaf is in-process).
//
//   *** PHASE 1a, precision-INSENSITIVE by construction ***
//   The scripted leaf returns COARSE, well-separated (value, logits) with NO near-ties, so the discrete
//   outcome (the SH survivor + the improved-π argmax) is identical regardless of float32-vs-float64.
//   This proves the STRUCTURE is faithful WITHOUT chasing the float32-prior/float64-Q mixed precision
//   (that is 1b). The near-tie / fine-input parity is explicitly OUT of scope here.
//
//   The scripted seam is RNG-free and identical across languages by construction:
//     * GumbelSource::gumbel(n)  -> the next n values off a GUMBEL FIFO, cycled mod its length (so the
//                                   root logit+g top-k AND every SH cut key use the SAME perturbations
//                                   on both sides; Python feeds the SAME table to rng.gumbel);
//     * GumbelSource::sample_world(bw) -> bw[0] by default, OR a scripted world-index FIFO cycling
//                                   bw[idx mod |bw|] (so a root action's c_outcome determinizations /
//                                   the per-sim world resolve identically across languages);
//     * the leaf (value, logits) -> the next entry off a LEAF FIFO, consumed in CALL ORDER (the descent
//                                   is structurally identical on both sides, so the call order matches).
//                                   logits[s] = leaf_logit_base + s·0.25 (a coarse, well-separated ramp
//                                   computed IDENTICALLY on both sides — distinct per slot by ≥0.25, so
//                                   the masked-softmax prior has no near-tie; precision-insensitive).
//   All three FIFOs are CYCLED modulo their length (a 48-sim search consumes more leaves than a small
//   table holds; the Python reference cycles the SAME tables the SAME way).
//
//   ADR-0012 P9: the imperative shell. argv is decoded once into typed views; `opt` returns a
//   std::optional<std::string_view>; load_instance returns a typed std::expected reported loudly. The
//   scripted tables being non-empty is the fixture's own invariant (checked at parse, then asserted).
//
//   Protocol:
//     argv: --instance <p> --faces <p> [--m N --n-sims N --c-puct f --c-visit f --c-scale f
//           --c-outcome N --max-depth N --lam f --prefix "s s s"]  (--prefix advances the real
//           (loc,bw,coll) by a deterministic slot sequence against the true world bw[0] before the
//           search, so the fixed input state can be mid-episode);
//     stdin: line 1 = space-separated gumbel values (doubles, the gumbel FIFO);
//            line 2 = space-separated leaf value/logit-base pairs flattened as "v0 lb0 v1 lb1 ..."
//                     (doubles, the leaf FIFO: each consecutive pair is one leaf's (value, logit_base));
//            line 3 = OPTIONAL space-separated world indices (ints; absent/empty -> sample_world=bw[0]).
//     stdout: three ints — the executed action slot, the improved-π argmax slot, then the total
//             root-action sims spent (n_spent; = n_sims on a faithful non-empty-belief search — the
//             SH full-budget invariant the harness checks against the Python side).
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

// The scripted, RNG-free leaf evaluator (a NetEvaluator). Consumed in CALL ORDER, cycled modulo its
// length. Each call delivers one (value, logit_base) pair; the returned logits ramp logits[s] =
// logit_base + s·0.25 over the full slot space (coarse, well-separated -> no near-tie). The Python
// reference builds the SAME ramp from the SAME (value, logit_base) table, so the masked-softmax priors
// are identical (precision-insensitive).
class ScriptedNet final : public chocofarm::NetEvaluator {
  public:
    ScriptedNet(std::vector<double> values, std::vector<double> logit_bases, int n_slots)
        : values_(std::move(values)), logit_bases_(std::move(logit_bases)), n_slots_(n_slots) {}

    std::expected<chocofarm::NetPrediction, chocofarm::Error> predict(
        std::span<const float> x) const override {
        (void)x;  // the scripted leaf ignores the features (it is keyed only on call order)
        assert(!values_.empty() && !logit_bases_.empty() && "gumbel_dump: empty scripted leaf table");
        size_t i = call_++;
        double v = values_[i % values_.size()];
        double lb = logit_bases_[i % logit_bases_.size()];
        chocofarm::NetPrediction pred;
        pred.value = static_cast<float>(v);
        pred.logits.resize(static_cast<size_t>(n_slots_));
        for (int s = 0; s < n_slots_; ++s)
            pred.logits[static_cast<size_t>(s)] = static_cast<float>(lb + s * 0.25);  // coarse ramp
        return pred;
    }

  private:
    std::vector<double> values_;
    std::vector<double> logit_bases_;
    int n_slots_;
    mutable size_t call_ = 0;  // call-order index (mutable: predict is const per the port)
};

// The scripted, RNG-free Gumbel source. gumbel(n) cycles the gumbel FIFO; sample_world cycles a
// world-index FIFO (empty -> bw[0]). Identical across languages by construction.
class ScriptedGumbelSource final : public chocofarm::GumbelSource {
  public:
    ScriptedGumbelSource(std::vector<double> gumbels, std::vector<int> world_idxs)
        : gumbels_(std::move(gumbels)), world_idxs_(std::move(world_idxs)) {}

    uint32_t sample_world(const std::vector<uint32_t>& bw) override {
        if (world_idxs_.empty()) return bw[0];
        int raw = world_idxs_[(widx_++) % world_idxs_.size()];
        int n = static_cast<int>(bw.size());
        return bw[static_cast<size_t>(((raw % n) + n) % n)];  // non-negative modulo
    }

    std::vector<double> gumbel(int n) override {
        assert(!gumbels_.empty() && "gumbel_dump: empty scripted gumbel table");
        std::vector<double> out(static_cast<size_t>(n));
        for (int i = 0; i < n; ++i) out[static_cast<size_t>(i)] = gumbels_[(gidx_++) % gumbels_.size()];
        return out;
    }

  private:
    std::vector<double> gumbels_;
    std::vector<int> world_idxs_;
    size_t gidx_ = 0;
    size_t widx_ = 0;
};
}  // namespace

int main(int argc, char** argv) {
    std::vector<std::string_view> args(argv, argv + argc);
    std::optional<std::string_view> instance = opt(args, "--instance");
    std::optional<std::string_view> faces = opt(args, "--faces");
    if (!instance || !faces) {
        std::cerr << "usage: gumbel-dump --instance <p> --faces <p> [--m N --n-sims N --c-puct f "
                     "--c-visit f --c-scale f --c-outcome N --max-depth N --lam f --prefix \"s s\"] "
                     "(gumbel FIFO on stdin line 1, leaf value/logit-base pairs on line 2, optional "
                     "world-index FIFO on line 3)\n";
        return 2;
    }

    chocofarm::GumbelConfig cfg;
    if (auto v = opt(args, "--m")) cfg.m = to_int(*v);
    if (auto v = opt(args, "--n-sims")) cfg.n_sims = to_int(*v);
    if (auto v = opt(args, "--c-puct")) cfg.c_puct = to_double(*v);
    if (auto v = opt(args, "--c-visit")) cfg.c_visit = to_double(*v);
    if (auto v = opt(args, "--c-scale")) cfg.c_scale = to_double(*v);
    if (auto v = opt(args, "--c-outcome")) cfg.c_outcome = to_int(*v);
    if (auto v = opt(args, "--max-depth")) cfg.max_depth = to_int(*v);
    double lam = opt(args, "--lam") ? to_double(*opt(args, "--lam")) : 0.1;

    auto inst = chocofarm::load_instance(*instance, *faces);
    if (!inst) { std::cerr << "gumbel-dump: FATAL: " << inst.error().message << "\n"; return 1; }
    chocofarm::Environment env(*inst);

    chocofarm::Loc loc{env.entry_point()};
    std::vector<uint32_t> bw = env.worlds();
    std::set<int> collected;

    // optionally advance the real (loc, bw, collected) by a prefix slot sequence against the true world
    // bw[0] (the same deterministic world both languages advance by), so the fixed search input can be
    // a mid-episode state, not just the root.
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

    // read the scripted FIFOs from stdin.
    std::vector<double> gumbels;
    {
        std::string line;
        std::getline(std::cin, line);
        std::istringstream iss(line);
        double v;
        while (iss >> v) gumbels.push_back(v);
    }
    // line 2: flattened (value, logit_base) pairs.
    std::vector<double> leaf_flat;
    {
        std::string line;
        std::getline(std::cin, line);
        std::istringstream iss(line);
        double v;
        while (iss >> v) leaf_flat.push_back(v);
    }
    std::vector<int> world_idxs;
    {
        std::string line;
        if (std::getline(std::cin, line)) {
            std::istringstream iss(line);
            int v;
            while (iss >> v) world_idxs.push_back(v);
        }
    }
    if (gumbels.empty() || leaf_flat.size() < 2) {
        std::cerr << "gumbel-dump: FATAL: need a non-empty gumbel FIFO (line 1) AND at least one "
                     "(value, logit_base) leaf pair (line 2) on stdin\n";
        return 1;
    }
    // split the flattened pairs into the value FIFO + the logit-base FIFO.
    std::vector<double> values, logit_bases;
    for (size_t i = 0; i + 1 < leaf_flat.size(); i += 2) {
        values.push_back(leaf_flat[i]);
        logit_bases.push_back(leaf_flat[i + 1]);
    }

    ScriptedNet net(std::move(values), std::move(logit_bases), chocofarm::n_action_slots(env));
    chocofarm::GumbelAZPolicy policy(cfg, net, env);
    ScriptedGumbelSource src(std::move(gumbels), std::move(world_idxs));

    chocofarm::GumbelAZPolicy::Decision dec = policy.run_search(loc, bw, collected, lam, src);

    // executed action slot.
    int exec_slot;
    if (dec.action.kind == chocofarm::ActionKind::Terminate) exec_slot = chocofarm::term_slot(env);
    else if (dec.action.kind == chocofarm::ActionKind::Treasure) exec_slot = dec.action.i;
    else exec_slot = env.N() + dec.action.i;

    // improved-π argmax slot (first-wins tie over slot order, mirroring numpy argmax).
    int argmax_slot = 0;
    double best = dec.improved.empty() ? 0.0 : dec.improved[0];
    for (int s = 1; s < static_cast<int>(dec.improved.size()); ++s) {
        if (dec.improved[static_cast<size_t>(s)] > best) {
            best = dec.improved[static_cast<size_t>(s)];
            argmax_slot = s;
        }
    }

    std::cout << exec_slot << " " << argmax_slot << " " << dec.n_spent << "\n";
    return 0;
}
