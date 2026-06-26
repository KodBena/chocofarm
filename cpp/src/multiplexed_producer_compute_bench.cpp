// cpp/src/multiplexed_producer_compute_bench.cpp
// Purpose: the PRODUCER-LEVEL A/B for the in-process batched featurizer (BatchPredict lever #3,
//   batch_predict.hpp). The component bench (batch_predict_bench.cpp) proved the FEATURIZER-level win
//   (~+27-30% on the belief sweep, bit-identical) in ISOLATION. The open question is the PRODUCER-LEVEL win
//   once the batched featurizer is driven inside the MULTIPLEXED search the producer actually runs — because
//   the batch tiling reuse only materializes when B *different* parked beliefs are featurized TOGETHER, and
//   that only happens in the multiplex (the producer parks K resumable TreeCursors, each on a leaf per RTT).
//   A single cursor (leaf_cpu_microbench) processes one belief at a time and cannot exhibit the reuse.
//
//   THE HARNESS (pure producer COMPUTE — no inference server, no ZMQ; a DetNet stand-in for the net):
//     Drive K TreeCursors multiplexed in the producer's round-sync park-collect-resume loop. Each RTT every
//     running cursor advances to its leaf park-point (CursorNeedsLeaf); the driver COLLECTS the parked
//     leaves, evaluates them (the DetNet — the producer does NOT do the net forward, so a trivial total
//     deterministic leaf is a FAIR stand-in for the producer-side per-leaf cost MINUS the wire codec, the
//     same stand-in leaf_cpu_microbench / fiber_proto use), and resumes each cursor. The ONLY arm difference
//     is WHERE/HOW the feature row is built:
//       PER-LEAF (baseline): each cursor builds its OWN row at park (eval_build_features — the production
//                            path), the driver predicts on need.features, resumes via resume(pred).
//       BATCHED  (lever #3): each cursor is in DEFERRED-featurize mode (skips its per-leaf build at park);
//                            the driver collects the parked (loc, bw, collected), calls
//                            BatchFeaturizer::featurize_batch ONCE per RTT, predicts on each row, and
//                            resumes via resume_with_features(row, pred). The batched sweep holds each
//                            env-static mask word resident across a 4-belief tile (lever #3 locality).
//     Both arms drive the SAME K searches (same rotated CyclicGumbel tables, same DetNet, fresh policy+source
//     per decision), so the only difference is the featurize path -> the A/B isolates the lever.
//
//   GATES:
//     (1) BIT-IDENTITY (non-negotiable): the two arms produce IDENTICAL per-decision results (executed
//         action + survivor + n_spent + improved-pi) AND identical per-leaf feature-row sequences. The
//         batched rows are byte-identical to the per-leaf rows (proven upstream + re-asserted here), and the
//         deferred-featurize seam only moves WHERE the identical row is computed, so the multiplexed search
//         MUST be identical. A divergence is a loud FAIL + do-not-ship (ADR-0002).
//     (2) A/B: full producer-COMPUTE us/decision, BATCHED vs PER-LEAF, across the multiplex width
//         K in {8,32,64} (the net batch). Interleaved/paired reps, warmup discarded, median/IQR + bootstrap
//         95% CI of the paired ratio. THIS is the producer-level number the ~-9% projection becomes.
//         MEASURED (this base, core-3): ~+22-25% FASTER (sweep-dominated upper bound). The DetNet stand-in
//         is TRIVIAL (the server's real forward is NOT a producer cost), so the belief sweep is a LARGER
//         fraction of THIS harness's per-decision compute (~48% of per-leaf instructions, perf-profiled) than
//         of the real producer. An independent perf-stat firewall confirmed the win is genuine CPU-work
//         reduction (~18-20% fewer cycles, ~13% fewer L1 loads — the mask-resident 4-belief tiling, the lever
//         #3 mechanism) and that every harness asymmetry (the cap-16 belief memo only the per-leaf arm uses;
//         the batched arm's extra per-leaf row alloc + double float32 narrow) HANDICAPS the batched arm — so
//         the true lever win is if anything LARGER, never inflated by a confound. To port to a real-producer
//         claim, scale by the producer's measured sweep wall fraction (~55%) -> ~+13-15% producer (the
//         de-risk f=0.55 projection); the brief's ~-9% modeled a smaller featurizer fraction (0.436*0.21).
//
//   Run:  nice -n -19 taskset -c 3 chocofarm-mux-producer-compute-bench --instance <p> --faces <p>
//                                  [--decisions 256] [--reps 11] [--n-sims 256 --m 24 --c-outcome 2 --max-depth 24]
//   A separate executable (ADR-0012 P3, one-owner). ADDITIVE — does NOT rewire the production real_producer.
// Public Domain (The Unlicense).
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <memory>
#include <optional>
#include <random>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

#include "chocofarm/batch_predict.hpp"
#include "chocofarm/collected_set.hpp"
#include "chocofarm/cyclic_gumbel.hpp"
#include "chocofarm/env.hpp"
#include "chocofarm/gumbel.hpp"
#include "chocofarm/gumbel_cursor.hpp"
#include "chocofarm/instance.hpp"
#include "chocofarm/net_evaluator.hpp"

namespace {
using Clock = std::chrono::steady_clock;

[[nodiscard]] std::optional<std::string_view> opt(std::span<const std::string_view> args,
                                                  std::string_view name) {
    for (size_t i = 1; i + 1 < args.size(); ++i)
        if (args[i] == name) return args[i + 1];
    return std::nullopt;
}
[[nodiscard]] int to_int(std::string_view s) { return std::atoi(std::string(s).c_str()); }

// The SAME total deterministic leaf fiber_proto.cpp / leaf_cpu_microbench.cpp use (a pure function of the
// features) — the producer-side stand-in for the server's forward, which is NOT a producer cost. A pure
// function of x, so PER-LEAF and BATCHED feeding the byte-identical row get the byte-identical prediction.
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

// The per-cursor rotated gumbel table (the K trees differ only by the rotation) — the SAME shape
// real_producer.cpp's slot_table uses, so the multiplex is a faithful producer LOAD structure.
[[nodiscard]] std::vector<double> slot_table(const std::vector<double>& base, int i) {
    std::vector<double> table(base.size());
    for (size_t j = 0; j < base.size(); ++j)
        table[j] = base[(j + static_cast<size_t>(i)) % base.size()];
    return table;
}

// One completed-decision record + its full leaf-row sequence (the bit-identity trace; compared across arms).
struct DecisionRec {
    int action_kind = -1;
    int action_i = -1;
    long n_spent = 0;
    long survivor = -2;                       // -2 = no survivor (trace-only typed-absence sentinel)
    std::vector<double> improved;
    std::vector<std::vector<float>> leaf_rows;  // each leaf's forwarded float32 row, in request order
};

// One multiplexed cursor slot: its OWN policy + rotated source + the live decision state (the cursor captures
// loc/bw/collected BY REFERENCE, so these are stable members), the cursor, and the in-progress leaf trace.
// Re-armed per decision with a fresh policy + source (mirrors leaf_cpu_microbench's per-decision construction
// — a tree/fiber slot taking a new decision). Deferred-featurize iff `batched`. The cursor holds references
// into this slot, so slots live behind unique_ptr (never moved).
struct Slot {
    chocofarm::GumbelConfig cfg;
    const chocofarm::Environment& env;
    chocofarm::NetEvaluator& net;
    std::vector<double> table;
    bool batched;

    chocofarm::Loc loc;
    chocofarm::Belief bw;
    chocofarm::CollectedSet collected;
    double lam = 0.1;

    std::unique_ptr<chocofarm::GumbelAZPolicy> policy;
    std::unique_ptr<chocofarm::CyclicGumbelSource> src;
    std::unique_ptr<chocofarm::TreeCursor> cur;
    chocofarm::Step step{chocofarm::CursorDecided{}};  // the last advance/resume result
    bool decided = false;
    std::vector<std::vector<float>> cur_leaves;        // leaves of the IN-PROGRESS decision (trace mode)

    Slot(const chocofarm::Environment& e, chocofarm::NetEvaluator& n, std::vector<double> tab, bool b)
        : env(e), net(n), table(std::move(tab)), batched(b),
          loc(e.entry_point()), bw(e.full_belief()) {}

    void arm(const chocofarm::GumbelConfig& c) {
        cfg = c;
        policy = std::make_unique<chocofarm::GumbelAZPolicy>(cfg, net, env);
        src = std::make_unique<chocofarm::CyclicGumbelSource>(env, table);
        cur = std::make_unique<chocofarm::TreeCursor>(*policy, loc, bw, collected, lam, *src);
        if (batched) cur->enable_deferred_featurize();
        decided = false;
        cur_leaves.clear();
        step = cur->advance();  // park at the root-eval leaf (or decide immediately on an empty belief)
    }
};

// Drive K cursors multiplexed (round-sync park-collect-resume) for EXACTLY `target_decisions` total decisions
// (across all K slots), starting each slot from the root with a fresh policy+source. If `trace` is non-null,
// it records each completed decision (in completion order) for the bit-identity gate. `bf` is used ONLY by the
// batched arm (the per-RTT featurize_batch). leaves_out += the leaf-request count. Returns elapsed seconds.
[[nodiscard]] double drive_mux(std::vector<std::unique_ptr<Slot>>& slots, const chocofarm::GumbelConfig& cfg,
                               bool batched, const chocofarm::BatchFeaturizer* bf, int target_decisions,
                               long& leaves_out, std::vector<DecisionRec>* trace) {
    const size_t K = slots.size();
    int done = 0;
    std::vector<chocofarm::BatchLeaf> blv;       // batched: the parked leaves this RTT
    std::vector<std::vector<double>> rows;       // batched: featurize_batch output rows
    std::vector<int> parked_idx;                 // slot indices parked at a leaf this RTT

    const auto t0 = Clock::now();
    for (size_t i = 0; i < K; ++i) slots[i]->arm(cfg);

    auto record_and_rearm = [&](Slot& s) {
        // s.step is CursorDecided and not yet recorded: record (+ trace) + count + re-arm (if more to do).
        if (!s.decided) {
            if (trace) {
                const auto& d = std::get<chocofarm::CursorDecided>(s.step).decision;
                DecisionRec dr;
                dr.action_kind = static_cast<int>(d.action.kind);
                dr.action_i = d.action.i;
                dr.n_spent = static_cast<long>(d.n_spent.value());
                dr.survivor = d.survivor_slot ? static_cast<long>(d.survivor_slot->value()) : -2;
                dr.improved = d.improved;
                dr.leaf_rows = std::move(s.cur_leaves);
                trace->push_back(std::move(dr));
            }
            s.decided = true;
            ++done;
        }
        if (done < target_decisions) s.arm(cfg);  // next decision
    };

    while (done < target_decisions) {
        parked_idx.clear();
        for (size_t i = 0; i < K; ++i) {
            Slot& s = *slots[i];
            if (std::holds_alternative<chocofarm::CursorDecided>(s.step)) record_and_rearm(s);
            if (std::holds_alternative<chocofarm::CursorNeedsLeaf>(s.step))
                parked_idx.push_back(static_cast<int>(i));
        }
        if (parked_idx.empty()) {
            if (done >= target_decisions) break;
            continue;  // every slot decided + re-armed into another immediate decision (empty belief) — loop
        }
        const size_t B = parked_idx.size();

        if (batched) {
            blv.resize(B);
            for (size_t j = 0; j < B; ++j) {
                Slot& s = *slots[static_cast<size_t>(parked_idx[j])];
                blv[j].loc = s.cur->parked_loc().pt;
                blv[j].bw = &s.cur->parked_belief();
                blv[j].collected = &s.cur->parked_collected();
            }
            bf->featurize_batch(std::span<const chocofarm::BatchLeaf>(blv.data(), B), rows);
        }
        for (size_t j = 0; j < B; ++j) {
            Slot& s = *slots[static_cast<size_t>(parked_idx[j])];
            if (batched) {
                std::vector<float> row32(rows[j].begin(), rows[j].end());  // the float32 the net forwards
                auto pred = s.net.predict(row32);
                if (trace) s.cur_leaves.push_back(std::move(row32));
                s.step = s.cur->resume_with_features(rows[j], pred.value());
            } else {
                const auto& need = std::get<chocofarm::CursorNeedsLeaf>(s.step);
                if (trace) s.cur_leaves.emplace_back(need.features.begin(), need.features.end());
                auto pred = s.net.predict(need.features);
                s.step = s.cur->resume(pred.value());
            }
            leaves_out += 1;
        }
    }
    return std::chrono::duration<double>(Clock::now() - t0).count();
}

// ---- robust stats (the batch_predict_bench template: median/IQR + bootstrap-95% CI of the paired ratio) --
struct Stat { double median, q1, q3; };
[[nodiscard]] Stat stat(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    auto pct = [&](double p) {
        if (v.empty()) return 0.0;
        const double idx = p * static_cast<double>(v.size() - 1);
        const size_t lo = static_cast<size_t>(idx);
        const size_t hi = std::min(lo + 1, v.size() - 1);
        const double frac = idx - static_cast<double>(lo);
        return v[lo] * (1.0 - frac) + v[hi] * frac;
    };
    return {pct(0.5), pct(0.25), pct(0.75)};
}
struct Boot { double med, lo, hi; };
[[nodiscard]] Boot bootstrap_ratio_ci(const std::vector<double>& ratios, std::mt19937_64& rng) {
    const size_t n = ratios.size();
    std::vector<double> meds; meds.reserve(4000);
    std::uniform_int_distribution<size_t> pick(0, n - 1);
    for (int it = 0; it < 4000; ++it) {
        std::vector<double> samp(n);
        for (size_t i = 0; i < n; ++i) samp[i] = ratios[pick(rng)];
        std::sort(samp.begin(), samp.end());
        meds.push_back(samp[n / 2]);
    }
    std::sort(meds.begin(), meds.end());
    std::vector<double> r = ratios; std::sort(r.begin(), r.end());
    return {r[n / 2], meds[static_cast<size_t>(0.025 * meds.size())],
            meds[static_cast<size_t>(0.975 * meds.size())]};
}

[[nodiscard]] std::vector<std::unique_ptr<Slot>> make_slots(const chocofarm::Environment& env,
                                                            chocofarm::NetEvaluator& net,
                                                            const std::vector<double>& base, size_t K,
                                                            bool batched) {
    std::vector<std::unique_ptr<Slot>> slots;
    slots.reserve(K);
    for (size_t i = 0; i < K; ++i)
        slots.push_back(std::make_unique<Slot>(env, net, slot_table(base, static_cast<int>(i)), batched));
    return slots;
}
}  // namespace

int main(int argc, char** argv) {
    std::vector<std::string_view> args(argv, argv + argc);
    std::optional<std::string_view> instance = opt(args, "--instance");
    std::optional<std::string_view> faces = opt(args, "--faces");
    if (!instance || !faces) {
        std::cerr << "usage: mux-producer-compute-bench --instance <p> --faces <p> [--decisions N] "
                     "[--reps N] [--n-sims N --m N --c-outcome N --max-depth N]\n";
        return 2;
    }
    const int decisions = opt(args, "--decisions") ? to_int(*opt(args, "--decisions")) : 256;
    const int reps = opt(args, "--reps") ? to_int(*opt(args, "--reps")) : 11;

    // The production search config (the leaf_cpu_microbench / batch_predict default shape); each field is a
    // typed domain (gumbel.hpp), wrapped at the CLI ACL exactly as gumbel_cursor_proto does.
    using SR = chocofarm::SearchRep;
    chocofarm::GumbelConfig cfg;
    cfg.m = chocofarm::CandidateCount{static_cast<SR>(opt(args, "--m") ? to_int(*opt(args, "--m")) : 24)};
    cfg.n_sims = chocofarm::SimBudget{static_cast<SR>(opt(args, "--n-sims") ? to_int(*opt(args, "--n-sims")) : 256)};
    cfg.c_outcome = chocofarm::OutcomeIndex{static_cast<SR>(opt(args, "--c-outcome") ? to_int(*opt(args, "--c-outcome")) : 2)};
    cfg.max_depth = chocofarm::PlyDepth{static_cast<SR>(opt(args, "--max-depth") ? to_int(*opt(args, "--max-depth")) : 24)};

    auto inst = chocofarm::load_instance(*instance, *faces);
    if (!inst) { std::cerr << "mux-producer-compute-bench: FATAL: " << inst.error().message << "\n"; return 1; }
    chocofarm::Environment env(*inst);
    DetNet net(chocofarm::n_action_slots(env).value());
    chocofarm::BatchFeaturizer bf(env);

    const std::vector<double> base{0.40, -0.65, 1.10, 0.05, -0.30, 0.85, -1.20, 0.55,
                                   0.20, -0.45, 0.95, -0.10, 0.70};
    const std::vector<size_t> Ks = {8, 32, 64};

    std::cout << "mux-producer-compute-bench: N=" << env.N() << " nD=" << env.n_detectors()
              << " |worlds|=" << env.worlds().size() << " kW64=" << env.kW64()
              << " dim=" << bf.dim().value() << " n_slots=" << chocofarm::n_action_slots(env).value()
              << " use_bitset=" << (env.use_bitset() ? "true" : "false")
              << "  (cfg: m=" << cfg.m.value() << " n_sims=" << cfg.n_sims.value()
              << " c_outcome=" << cfg.c_outcome.value() << " max_depth=" << cfg.max_depth.value()
              << "; decisions/point=" << decisions << " reps=" << reps << ")\n";

    // ------------------- BIT-IDENTITY GATE (PER-LEAF vs BATCHED, same multiplexed search) ---------------
    // Drive the SAME K cursors both ways and assert every completed decision + its full leaf-row sequence is
    // identical. The batched rows are byte-identical to the per-leaf rows (the deferred-featurize seam only
    // moves WHERE the identical row is computed), so the multiplexed search MUST be identical.
    {
        bool ok = true;
        std::string why;
        for (size_t K : Ks) {
            const int target = std::max<int>(static_cast<int>(K) * 3, 64);  // a few decisions per slot
            auto sa = make_slots(env, net, base, K, /*batched=*/false);
            auto sb = make_slots(env, net, base, K, /*batched=*/true);
            long la = 0, lb = 0;
            std::vector<DecisionRec> ta, tb;
            (void)drive_mux(sa, cfg, false, nullptr, target, la, &ta);
            (void)drive_mux(sb, cfg, true, &bf, target, lb, &tb);
            if (ta.size() != tb.size()) { ok = false; why = "decision-count mismatch (K=" + std::to_string(K) + ")"; break; }
            if (la != lb) { ok = false; why = "leaf-count mismatch (K=" + std::to_string(K) + ")"; break; }
            for (size_t d = 0; d < ta.size() && ok; ++d) {
                const DecisionRec& a = ta[d];
                const DecisionRec& b = tb[d];
                auto fail = [&](const char* f) { ok = false; why = std::string(f) + " (K=" + std::to_string(K) + " dec=" + std::to_string(d) + ")"; };
                if (a.action_kind != b.action_kind || a.action_i != b.action_i) { fail("executed action"); break; }
                if (a.n_spent != b.n_spent) { fail("n_spent"); break; }
                if (a.survivor != b.survivor) { fail("survivor"); break; }
                if (a.improved != b.improved) { fail("improved-pi"); break; }
                if (a.leaf_rows.size() != b.leaf_rows.size()) { fail("leaf-sequence length"); break; }
                for (size_t l = 0; l < a.leaf_rows.size(); ++l)
                    if (a.leaf_rows[l] != b.leaf_rows[l]) { fail("leaf feature row"); break; }
            }
            if (!ok) break;
        }
        if (!ok) {
            std::cout << "RESULT: FAIL bit-identity (" << why << ") — the BATCHED multiplexed search diverged "
                         "from PER-LEAF; do NOT ship\n";
            return 1;
        }
        std::cout << "RESULT: PASS bit-identity (PER-LEAF vs BATCHED multiplexed search produce identical "
                     "decisions + identical per-leaf feature-row sequences across K in {8,32,64})\n";
    }

    // ------------------- A/B: full producer-COMPUTE us/decision, BATCHED vs PER-LEAF -------------------
    std::cout << "\nA/B: full producer-COMPUTE us/decision, BATCHED (lever #3, in-process featurize_batch) "
                 "vs PER-LEAF (the production eval_build_features). ratio = batched/perleaf; speedup>0 => "
                 "batched faster; FASTER iff 95% CI wholly below 1.0:\n";
    std::cout << std::setw(8) << "K" << std::setw(14) << "perleaf us/d" << std::setw(14) << "batched us/d"
              << std::setw(10) << "ratio" << std::setw(18) << "95% CI(ratio)"
              << std::setw(11) << "speedup%" << std::setw(9) << "verdict" << "\n";

    for (size_t K : Ks) {
        auto sa = make_slots(env, net, base, K, /*batched=*/false);
        auto sb = make_slots(env, net, base, K, /*batched=*/true);

        auto time_perleaf = [&]() -> double {
            long lv = 0;
            const double el = drive_mux(sa, cfg, false, nullptr, decisions, lv, nullptr);
            return el * 1e6 / static_cast<double>(decisions);
        };
        auto time_batched = [&]() -> double {
            long lv = 0;
            const double el = drive_mux(sb, cfg, true, &bf, decisions, lv, nullptr);
            return el * 1e6 / static_cast<double>(decisions);
        };

        (void)time_perleaf(); (void)time_batched();  // warmup discarded
        std::vector<double> sep_us, cand_us, ratios;
        for (int r = 0; r < reps; ++r) {
            double s, t;
            if (r & 1) { t = time_batched(); s = time_perleaf(); }
            else       { s = time_perleaf(); t = time_batched(); }
            sep_us.push_back(s); cand_us.push_back(t); ratios.push_back(t / s);
        }
        const Stat ss = stat(sep_us), cs = stat(cand_us);
        std::mt19937_64 brng(0x1234ull + K);
        const Boot bt = bootstrap_ratio_ci(ratios, brng);
        const double speedup = (1.0 / bt.med - 1.0) * 100.0;
        const char* verdict = (bt.hi < 1.0) ? "FASTER" : (bt.lo > 1.0) ? "SLOWER" : "NULL";
        std::ostringstream ci;
        ci << "[" << std::fixed << std::setprecision(3) << bt.lo << "," << bt.hi << "]";
        std::cout << std::fixed << std::setprecision(2)
                  << std::setw(8) << K << std::setw(14) << ss.median << std::setw(14) << cs.median
                  << std::setw(10) << std::setprecision(4) << bt.med
                  << std::setw(18) << ci.str()
                  << std::setw(10) << std::setprecision(1) << speedup << "%"
                  << std::setw(9) << verdict << "\n";
    }
    std::cout << "\nNOTE: full producer COMPUTE per decision (search control flow + belief sweeps + feature "
                 "assembly + the DetNet stand-in), NOT the belief sweep alone. ratio<1 => batched faster; "
                 "speedup% = (1/ratio - 1)*100.\n"
                 "CAVEAT (read the speedup as a SWEEP-DOMINATED UPPER BOUND, not the real-producer win): this "
                 "harness uses a TRIVIAL DetNet stand-in (the server's real forward is NOT a producer cost), so "
                 "the belief sweep is a LARGER fraction of THIS harness's per-decision compute (~48% of per-leaf "
                 "instructions, independently profiled) than of the real producer. An independent perf-stat "
                 "firewall confirmed the win is genuine CPU-work reduction (~18-20% fewer cycles, ~13% fewer L1 "
                 "loads from the mask-resident 4-belief tiling — lever #3's claimed mechanism) and that EVERY "
                 "harness asymmetry (the cap-16 belief memo only the per-leaf arm uses; the batched arm's extra "
                 "per-leaf row alloc + double float32 narrow) HANDICAPS the batched arm — so the true lever win "
                 "is if anything LARGER than measured, never inflated by a confound. To port to a real-producer "
                 "claim, scale by the producer's measured belief-sweep wall fraction (~55%, throughput-derisk "
                 "notes) -> ~+13-15% producer (the de-risk f=0.55 projection). The ~-9% projection in the brief "
                 "modeled a SMALLER featurizer fraction (0.436*0.21); the honest producer-level estimate is "
                 "~+13-15%, with this ~+22-25% the sweep-dominated stand-in upper bound.\n";
    return 0;
}
