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
//   Protocol (TWO leaf modes — COARSE 1a vs FINE 1b — selected by --leaf-logits-rows):
//     argv: --instance <p> --faces <p> [--m N --n-sims N --c-puct f --c-visit f --c-scale f
//           --c-outcome N --max-depth N --lam f --prefix "s s s" --leaf-logits-rows N]  (--prefix
//           advances the real (loc,bw,coll) by a deterministic slot sequence against bw[0] before the
//           search; --leaf-logits-rows N>0 selects the FINE per-slot-logits mode — see line 4);
//     stdin: line 1 = space-separated gumbel values (doubles, the gumbel FIFO);
//            line 2 = the leaf VALUE FIFO. In COARSE mode (--leaf-logits-rows absent/0) this is the
//                     flattened value/logit-base pairs "v0 lb0 v1 lb1 ..." (each pair = one leaf's
//                     (value, logit_base) for the s·0.25 ramp). In FINE mode it is one value per leaf
//                     "v0 v1 v2 ..." (the per-slot logits come from line 4 instead);
//            line 3 = OPTIONAL space-separated world indices (ints; absent/empty -> sample_world=bw[0]);
//            line 4 = (FINE mode only) the flattened FULL-PRECISION per-slot logits table, n_slots
//                     doubles per leaf row, --leaf-logits-rows rows total: "r0c0 r0c1 ... r1c0 ..." (the
//                     near-tie inputs the float32 seam discriminates on — gumbel_precision.py).
//     stdout: three ints — the executed action slot, the improved-π argmax slot, then the total
//             root-action sims spent (n_spent; = n_sims on a faithful non-empty-belief search — the
//             SH full-budget invariant the harness checks against the Python side).
//
// Public Domain (The Unlicense).
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <optional>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include "chocofarm/env.hpp"
#include "chocofarm/features.hpp"
#include "chocofarm/gumbel.hpp"
#include "chocofarm/gumbel_cursor.hpp"  // OPTION B: the explicit-state resumable driver (CHOCO_GUMBEL_DRIVER)
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
// length. TWO leaf modes share one class:
//   * COARSE (1a, gumbel_logic.py): each call delivers one (value, logit_base) pair; the returned logits
//     ramp logits[s] = logit_base + s·0.25 over the full slot space (well-separated -> NO near-tie). The
//     Python reference builds the SAME ramp from the SAME (value, logit_base) table (precision-
//     insensitive: the discrete outcome is identical float32-vs-float64).
//   * FINE (1b, gumbel_precision.py): each call delivers one value off the value FIFO + one FULL-
//     PRECISION per-slot logits ROW off a logits table (n_slots doubles per row). The rows carry tiny
//     per-slot deltas at the float32-epsilon scale, so the masked-softmax prior near-ties and the
//     float32 storage (the 1b prior seam) rounds it differently than float64 — the precision hazard.
// The mode is selected by which table is non-empty: a non-empty `logit_rows_` (one or more length-
// n_slots rows) takes the FINE path; otherwise the COARSE (logit_bases_) ramp.
class ScriptedNet final : public chocofarm::NetEvaluator {
  public:
    // COARSE constructor (the 1a (value, logit_base) ramp).
    ScriptedNet(std::vector<double> values, std::vector<double> logit_bases, int n_slots)
        : values_(std::move(values)), logit_bases_(std::move(logit_bases)), n_slots_(n_slots) {}

    // FINE constructor (the 1b full-precision per-slot logits table). `logit_rows` is a list of length-
    // n_slots rows, consumed in call order (cycled). The COARSE logit_bases_ is left empty.
    ScriptedNet(std::vector<double> values, std::vector<std::vector<double>> logit_rows, int n_slots)
        : values_(std::move(values)), logit_rows_(std::move(logit_rows)), n_slots_(n_slots) {}

    std::expected<chocofarm::NetPrediction, chocofarm::Error> predict(
        std::span<const float> x) const override {
        (void)x;  // the scripted leaf ignores the features (it is keyed only on call order)
        assert(!values_.empty() && "gumbel_dump: empty scripted value table");
        size_t i = call_++;
        chocofarm::NetPrediction pred;
        pred.value = static_cast<float>(values_[i % values_.size()]);
        pred.logits.resize(static_cast<size_t>(n_slots_));
        if (!logit_rows_.empty()) {
            // FINE: copy the full-precision per-slot logits row (narrowed to float32, the net's logit
            // dtype — the Python reference's leaf_logits_table rows are float64, softmaxed in float64 and
            // the PRIOR narrowed to float32; the C++ logits narrow to float32 here as the wire dtype, but
            // the masked softmax that builds the prior runs in float64 in evaluate(), as in Python).
            const std::vector<double>& row = logit_rows_[i % logit_rows_.size()];
            assert(row.size() == static_cast<size_t>(n_slots_) &&
                   "gumbel_dump: fine logits row width != n_slots");
            for (int s = 0; s < n_slots_; ++s)
                pred.logits[static_cast<size_t>(s)] = static_cast<float>(row[static_cast<size_t>(s)]);
        } else {
            // COARSE: the well-separated ramp logits[s] = logit_base + s·0.25.
            assert(!logit_bases_.empty() && "gumbel_dump: empty scripted leaf table");
            double lb = logit_bases_[i % logit_bases_.size()];
            for (int s = 0; s < n_slots_; ++s)
                pred.logits[static_cast<size_t>(s)] = static_cast<float>(lb + s * 0.25);
        }
        return pred;
    }

  private:
    std::vector<double> values_;
    std::vector<double> logit_bases_;               // COARSE mode: per-leaf logit base (ramp)
    std::vector<std::vector<double>> logit_rows_;   // FINE mode: per-leaf full-precision logits rows
    int n_slots_;
    mutable size_t call_ = 0;  // call-order index (mutable: predict is const per the port)
};

// The scripted, RNG-free Gumbel source. gumbel(n) cycles the gumbel FIFO; sample_world cycles a
// world-index FIFO (empty -> bw[0]). Identical across languages by construction.
class ScriptedGumbelSource final : public chocofarm::GumbelSource {
  public:
    // The scripted source threads `const Environment&` so it resolves its `bw[idx]` pokes through the
    // seam (env.world_at_rank, L4) — byte-identical to the former direct `bw[idx]` (the flat belief is a
    // worlds()-RANK-ordered subsequence, so the r-th element == the r-th rank).
    ScriptedGumbelSource(const chocofarm::Environment& env, std::vector<double> gumbels,
                         std::vector<int> world_idxs)
        : env_(env), gumbels_(std::move(gumbels)), world_idxs_(std::move(world_idxs)) {}

    uint32_t sample_world(const chocofarm::Belief& bw) override {
        if (world_idxs_.empty()) return env_.world_at_rank(bw, 0);
        int raw = world_idxs_[(widx_++) % world_idxs_.size()];
        int n = env_.nb(bw);
        return env_.world_at_rank(bw, ((raw % n) + n) % n);  // non-negative modulo
    }

    std::vector<double> gumbel(int n) override {
        assert(!gumbels_.empty() && "gumbel_dump: empty scripted gumbel table");
        std::vector<double> out(static_cast<size_t>(n));
        for (int i = 0; i < n; ++i) out[static_cast<size_t>(i)] = gumbels_[(gidx_++) % gumbels_.size()];
        return out;
    }

  private:
    const chocofarm::Environment& env_;
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
    // HPO/BENCHMARK-ONLY no-early-exit (default false). A presence flag (no value): when present the
    // executed-action Terminate substitution in run_search is exercised, so the parity test can compare
    // flag-off vs flag-on executed action on the SAME scripted input (gumbel.hpp GumbelConfig::no_early_exit).
    for (const std::string_view& a : args)
        if (a == "--no-early-exit") cfg.no_early_exit = true;
    double lam = opt(args, "--lam") ? to_double(*opt(args, "--lam")) : 0.1;
    // FINE leaf mode (1b): --leaf-logits-rows N>0 means line 2 is one value per leaf and line 4 carries
    // the full-precision per-slot logits table (N rows of n_slots doubles). Absent/0 = COARSE 1a mode.
    int leaf_logits_rows = opt(args, "--leaf-logits-rows") ? to_int(*opt(args, "--leaf-logits-rows")) : 0;

    auto inst = chocofarm::load_instance(*instance, *faces);
    if (!inst) { std::cerr << "gumbel-dump: FATAL: " << inst.error().message << "\n"; return 1; }
    chocofarm::Environment env(*inst);

    chocofarm::Loc loc{env.entry_point()};
    chocofarm::Belief bw = env.full_belief();   // the seam's belief construction entry
    chocofarm::CollectedSet collected;

    // optionally advance the real (loc, bw, collected) by a prefix slot sequence against the true world
    // bw[0] (the same deterministic world both languages advance by), so the fixed search input can be
    // a mid-episode state, not just the root.
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

    // read the scripted FIFOs from stdin.
    std::vector<double> gumbels;
    {
        std::string line;
        std::getline(std::cin, line);
        std::istringstream iss(line);
        double v;
        while (iss >> v) gumbels.push_back(v);
    }
    // line 2: in COARSE mode, flattened (value, logit_base) pairs; in FINE mode, one value per leaf.
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
    // line 4 (FINE mode only): the flattened full-precision per-slot logits table.
    std::vector<double> logits_flat;
    if (leaf_logits_rows > 0) {
        std::string line;
        if (std::getline(std::cin, line)) {
            std::istringstream iss(line);
            double v;
            while (iss >> v) logits_flat.push_back(v);
        }
    }

    // n_action_slots / term_slot now return typed SlotCount / SlotIndex; this dump tool's scripting stays
    // raw int (out of the retyping slice's scope) — unwrap at the crossing (ADR-0000 item 5).
    int n_slots = static_cast<int>(chocofarm::n_action_slots(env).value());
    // build the scripted net: FINE mode (--leaf-logits-rows N>0) uses line 2 as the value FIFO + line 4
    // as the per-slot logits table; COARSE mode splits line 2 into (value, logit_base) pairs.
    std::unique_ptr<ScriptedNet> net;
    if (leaf_logits_rows > 0) {
        if (leaf_flat.empty() ||
            logits_flat.size() != static_cast<size_t>(leaf_logits_rows) * static_cast<size_t>(n_slots)) {
            std::cerr << "gumbel-dump: FATAL: FINE mode needs a non-empty value FIFO (line 2) AND a "
                         "line-4 logits table of exactly leaf-logits-rows*n_slots doubles (got "
                      << logits_flat.size() << ", expected "
                      << static_cast<size_t>(leaf_logits_rows) * static_cast<size_t>(n_slots) << ")\n";
            return 1;
        }
        std::vector<std::vector<double>> rows;
        rows.reserve(static_cast<size_t>(leaf_logits_rows));
        for (int r = 0; r < leaf_logits_rows; ++r) {
            std::vector<double> row(logits_flat.begin() + static_cast<long>(r) * n_slots,
                                    logits_flat.begin() + static_cast<long>(r + 1) * n_slots);
            rows.push_back(std::move(row));
        }
        net = std::make_unique<ScriptedNet>(std::move(leaf_flat), std::move(rows), n_slots);
    } else {
        if (gumbels.empty() || leaf_flat.size() < 2) {
            std::cerr << "gumbel-dump: FATAL: need a non-empty gumbel FIFO (line 1) AND at least one "
                         "(value, logit_base) leaf pair (line 2) on stdin\n";
            return 1;
        }
        std::vector<double> values, logit_bases;
        for (size_t i = 0; i + 1 < leaf_flat.size(); i += 2) {
            values.push_back(leaf_flat[i]);
            logit_bases.push_back(leaf_flat[i + 1]);
        }
        net = std::make_unique<ScriptedNet>(std::move(values), std::move(logit_bases), n_slots);
    }
    if (gumbels.empty()) {
        std::cerr << "gumbel-dump: FATAL: need a non-empty gumbel FIFO (line 1)\n";
        return 1;
    }

    chocofarm::GumbelAZPolicy policy(cfg, *net, env);
    ScriptedGumbelSource src(env, std::move(gumbels), std::move(world_idxs));

    // DRIVER SELECTION (test seam; default = the validated Option-A path run_search). CHOCO_GUMBEL_DRIVER
    // =optionb drives the SAME search through the Option-B explicit-state cursor (gumbel_cursor.hpp),
    // feeding each parked leaf through the SAME ScriptedNet. The parity harnesses (gumbel_logic.py /
    // gumbel_precision.py) run UNCHANGED with this env var set, re-proving the crown jewels against B
    // (they pass env through). Bit-identity is the gate: B must produce the SAME (exec, argmax, n_spent).
    chocofarm::GumbelAZPolicy::Decision dec;
    const char* drv = std::getenv("CHOCO_GUMBEL_DRIVER");
    if (drv != nullptr && std::string_view(drv) == "optionb") {
        chocofarm::TreeCursor cur(policy, loc, bw, collected, lam, src);
        chocofarm::Step st = cur.advance();
        while (std::holds_alternative<chocofarm::CursorNeedsLeaf>(st)) {
            const auto& need = std::get<chocofarm::CursorNeedsLeaf>(st);
            auto pred = net->predict(need.features);  // the SAME scripted leaf the direct path forwards
            assert(pred.has_value() && "gumbel-dump(optionb): scripted net failed");
            st = cur.resume(*pred);
        }
        dec = std::get<chocofarm::CursorDecided>(st).decision;
    } else {
        dec = policy.run_search(loc, bw, collected, lam, src);
    }

    // executed action slot.
    int exec_slot;
    if (dec.action.kind == chocofarm::ActionKind::Terminate) exec_slot = static_cast<int>(chocofarm::term_slot(env).value());
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

    std::cout << exec_slot << " " << argmax_slot << " " << dec.n_spent.value() << "\n";  // .value() at the ostream boundary
    return 0;
}
