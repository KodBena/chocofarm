// cpp/src/batch_predict_bench.cpp
// Purpose: the COMPONENT bench + BIT-IDENTITY gate for the BatchPredict IN-PROCESS featurizer (lever #3,
//   batch_predict.hpp / batch_predict.cpp). Two parts, both pure compute (no redis, no net):
//
//   (1) BIT-IDENTITY gate (non-negotiable): for a spread of batches (B in {8,32,64} x several batches, a
//       realistic density/loc/collected mix), assert BatchFeaturizer::featurize_batch's row[b] ==
//       FeatureBuilder::build(loc_b, bw_b, collected_b) BYTE-FOR-BYTE — the FULL feature row, not just the
//       belief block. The per-leaf production build is the reference (the belief-sweep oracle's style). Also
//       net belief_features_batch row == belief_features(env, bw) directly (the sweep alone). A failure is a
//       loud RESULT: FAIL + do-not-ship (ADR-0002).
//
//   (2) A/B TIMING: the batched featurizer's belief SWEEP (BatchFeaturizer::belief_features_batch — the
//       lever-#3 mask-resident tiled AVX2 kernel) vs B per-leaf chocofarm::belief_features (the production
//       per-leaf sweep — which on this base ALREADY uses the AVX2 popcount primitive, so this isolates the
//       BATCH-SPECIFIC increment, the de-risk note's ~+30%). Interleaved/paired reps, warmup discarded,
//       median/IQR + bootstrap-95% CI of the paired ratio. The baseline-to-beat is the already-AVX2 per-leaf
//       sweep (belief_features over the bitset arm == belief_features_bitset == popcount_and = AVX2 here);
//       the batched arm's win over it IS the tiling increment.
//
//   Run:  nice -n -19 taskset -c 3 chocofarm-batch-predict-bench --instance <p> --faces <p>
//                                  [--budget-s 0.25] [--reps 13]
//   A separate executable (ADR-0012 P3, one-owner). Public Domain (The Unlicense).
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <optional>
#include <random>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include "chocofarm/batch_predict.hpp"
#include "chocofarm/collected_set.hpp"
#include "chocofarm/env.hpp"
#include "chocofarm/feature_compute.hpp"
#include "chocofarm/features.hpp"
#include "chocofarm/instance.hpp"

namespace {
[[nodiscard]] std::optional<std::string_view> opt(std::span<const std::string_view> args,
                                                  std::string_view name) {
    for (size_t i = 1; i + 1 < args.size(); ++i)
        if (args[i] == name) return args[i + 1];
    return std::nullopt;
}
volatile double g_sink = 0.0;  // defeat dead-code elimination of the timed work

// Build a BitsetBelief DIRECTLY over env.worlds()' RANK space from a flat world-subset (the oracle's
// to_bitset: bypass the gate so we hold a real bitset arm regardless). Each world's rank sets its bit.
[[nodiscard]] chocofarm::BitsetBelief to_bitset(const chocofarm::Environment& env,
                                                const std::vector<uint32_t>& flat_ranks) {
    chocofarm::BitsetBelief b;  // bits{} zero-initialized; tail stays 0
    b.kw64_ = chocofarm::WordCount{static_cast<chocofarm::WordRep>(env.kW64())};
    for (uint32_t r : flat_ranks) b.bits[r >> 6] |= (uint64_t{1} << (r & 63u));
    b.count_ = chocofarm::WorldCount{static_cast<chocofarm::WorldCountRep>(flat_ranks.size())};
    return b;
}

// A spread of REALISTIC belief densities (the belief-sweep oracle/bench style): the full prior + rank-
// strided subsets of varied size (different cover mixes via a pseudo-random phase). The caller cycles this
// pool to fill a batch.
[[nodiscard]] std::vector<std::vector<uint32_t>> belief_pool(size_t nworlds, std::mt19937_64& rng) {
    std::vector<std::vector<uint32_t>> pool;
    auto full = [&]() { std::vector<uint32_t> v(nworlds); std::iota(v.begin(), v.end(), 0u); return v; };
    auto strided = [&](size_t target) {
        std::vector<uint32_t> v;
        if (target == 0 || nworlds == 0) return v;
        const size_t step = std::max<size_t>(1, nworlds / target);
        const size_t phase = rng() % step;
        for (size_t i = phase; i < nworlds && v.size() < target; i += step)
            v.push_back(static_cast<uint32_t>(i));
        return v;
    };
    pool.push_back(full());
    for (size_t tgt : {nworlds / 2, nworlds / 4, nworlds / 8, nworlds / 16,
                       size_t{1000}, size_t{256}, size_t{64}, size_t{8}})
        pool.push_back(strided(std::min(tgt, nworlds)));
    return pool;
}

// One batch of B leaves: each leaf = a pool belief (cycled+shuffled), a named env coordinate as its loc
// (cycled over treasure/face/teleport points — the build()'s per-loc memo key contract), and a collected
// set (a varied subset of treasures). Returns the beliefs (owned), the locs, the collected sets, so the
// BatchLeaf views can point into them (they must outlive the call).
struct Batch {
    std::vector<chocofarm::BitsetBelief> beliefs;
    std::vector<chocofarm::Point> locs;
    std::vector<chocofarm::CollectedSet> collected;
};
[[nodiscard]] Batch make_batch(const chocofarm::Environment& env,
                               const std::vector<std::vector<uint32_t>>& pool, size_t B,
                               std::mt19937_64& rng) {
    using chocofarm::TreasureId;
    using chocofarm::TreasureRep;
    Batch out;
    out.beliefs.reserve(B); out.locs.reserve(B); out.collected.reserve(B);
    const int N = env.N(), nD = env.n_detectors(), nT = env.n_teleports();
    for (size_t i = 0; i < B; ++i) {
        out.beliefs.push_back(to_bitset(env, pool[(rng() + i) % pool.size()]));
        // a named coordinate: rotate through treasure / face / teleport rep_points (all are build()-valid loc keys)
        const size_t pick = (rng() + i) % static_cast<size_t>(N + nD + std::max(1, nT));
        chocofarm::Point pt;
        if (pick < static_cast<size_t>(N)) pt = env.treasure_pt(static_cast<int>(pick));
        else if (pick < static_cast<size_t>(N + nD)) pt = env.face_pt(static_cast<int>(pick - static_cast<size_t>(N)));
        else if (nT > 0) pt = env.teleport_pt(static_cast<int>((pick - static_cast<size_t>(N + nD)) % static_cast<size_t>(nT)));
        else pt = env.treasure_pt(0);
        out.locs.push_back(pt);
        // a varied collected subset: every (i mod 3 + 1)-th treasure up to i (a deterministic mix per leaf)
        chocofarm::CollectedSet cs;
        const int stride = static_cast<int>(i % 3) + 1;
        for (int t = 0; t < N; t += stride)
            if ((static_cast<size_t>(t) + i) % 2 == 0) cs = cs.with(TreasureId{static_cast<TreasureRep>(t)});
        out.collected.push_back(cs);
    }
    return out;
}

[[nodiscard]] bool equal_features(const chocofarm::BeliefFeatures& a, const chocofarm::BeliefFeatures& b,
                                  std::string& why) {
    auto note = [&](const char* f) { why = f; return false; };
    if (a.marg != b.marg) return note("marg");
    if (a.p_pos != b.p_pos) return note("p_pos");
    if (a.informative != b.informative) return note("informative");
    if (a.marg_sum != b.marg_sum) return note("marg_sum");
    if (a.sharpness != b.sharpness) return note("sharpness");
    if (a.nonempty != b.nonempty) return note("nonempty");
    return true;
}

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
}  // namespace

int main(int argc, char** argv) {
    std::vector<std::string_view> args(argv, argv + argc);
    std::optional<std::string_view> instance = opt(args, "--instance");
    std::optional<std::string_view> faces = opt(args, "--faces");
    if (!instance || !faces) {
        std::cerr << "usage: batch-predict-bench --instance <p> --faces <p> [--budget-s 0.25] [--reps 13]\n";
        return 2;
    }
    const double budget = opt(args, "--budget-s")
        ? std::atof(std::string(*opt(args, "--budget-s")).c_str()) : 0.25;
    const int reps = opt(args, "--reps") ? std::atoi(std::string(*opt(args, "--reps")).c_str()) : 13;

    auto inst = chocofarm::load_instance(*instance, *faces);
    if (!inst) { std::cerr << "batch-predict-bench: FATAL: " << inst.error().message << "\n"; return 1; }
    chocofarm::Environment env(*inst);
    chocofarm::FeatureBuilder fb(env);          // the per-leaf production reference
    chocofarm::BatchFeaturizer bf(env);         // the component under test
    const int N = env.N(), nD = env.n_detectors();
    const size_t nworlds = env.worlds().size();
    const size_t mask_bytes =
        static_cast<size_t>(N + nD) * static_cast<size_t>(env.kW64()) * sizeof(uint64_t);

    std::cout << "batch-predict-bench: N=" << N << " nD=" << nD << " |worlds|=" << nworlds
              << " kW64=" << env.kW64() << " dim=" << bf.dim().value()
              << " mask_matrix=" << (mask_bytes / 1024.0) << " KiB"
              << " use_bitset=" << (env.use_bitset() ? "true" : "false")
              << "  (budget=" << budget << "s/point reps=" << reps << ")\n";
    if (!env.use_bitset()) {
        std::cout << "RESULT: SKIP (env gates OFF the bitset arm — the batched kernel sweeps the env-static "
                     "bitset masks; nothing to bench)\n";
        return 0;
    }

    std::mt19937_64 rng(0xC0FFEEull);
    const std::vector<std::vector<uint32_t>> pool = belief_pool(nworlds, rng);
    const std::vector<size_t> Bs = {8, 32, 64};

    // ------------------- BIT-IDENTITY GATE (the FULL feature ROW + the belief sweep alone) -------------
    {
        size_t row_checks = 0, sweep_checks = 0;
        for (size_t B : Bs) {
            for (int batch_i = 0; batch_i < 6; ++batch_i) {
                Batch batch = make_batch(env, pool, B, rng);
                // Build the BatchLeaf views (bw points at the variant-wrapped bitset belief). We wrap each
                // BitsetBelief in a Belief variant (owned alongside the batch) so the view is stable.
                std::vector<chocofarm::Belief> bel_var; bel_var.reserve(B);
                for (size_t b = 0; b < B; ++b) bel_var.emplace_back(batch.beliefs[b]);
                std::vector<chocofarm::BatchLeaf> lv(B);
                for (size_t b = 0; b < B; ++b) {
                    lv[b].loc = batch.locs[b]; lv[b].bw = &bel_var[b]; lv[b].collected = &batch.collected[b];
                }

                // (a) the FULL feature row == per-leaf build, byte-for-byte.
                std::vector<std::vector<double>> rows;
                bf.featurize_batch(std::span<const chocofarm::BatchLeaf>(lv.data(), B), rows);
                for (size_t b = 0; b < B; ++b) {
                    const std::vector<double> ref = fb.build(batch.locs[b], bel_var[b], batch.collected[b]);
                    if (rows[b] != ref) {
                        // name the first diverging index for diagnosis
                        size_t k = 0; for (; k < ref.size() && k < rows[b].size(); ++k) if (rows[b][k] != ref[k]) break;
                        std::cout << "RESULT: FAIL bit-identity ROW (B=" << B << " batch=" << batch_i
                                  << " b=" << b << " idx=" << k << " got=" << (k < rows[b].size() ? rows[b][k] : 0.0)
                                  << " ref=" << (k < ref.size() ? ref[k] : 0.0)
                                  << ") — batched row != per-leaf build; do NOT ship\n";
                        return 1;
                    }
                    ++row_checks;
                }

                // (b) the belief SWEEP alone == per-leaf belief_features, byte-for-byte.
                std::vector<const chocofarm::Belief*> bptr(B);
                for (size_t b = 0; b < B; ++b) bptr[b] = &bel_var[b];
                std::vector<chocofarm::BeliefFeatures> got;
                bf.belief_features_batch(std::span<const chocofarm::Belief* const>(bptr.data(), B), got);
                for (size_t b = 0; b < B; ++b) {
                    const chocofarm::BeliefFeatures ref = chocofarm::belief_features(env, bel_var[b]);
                    std::string why;
                    if (!equal_features(got[b], ref, why)) {
                        std::cout << "RESULT: FAIL bit-identity SWEEP (B=" << B << " batch=" << batch_i
                                  << " b=" << b << " field=" << why << ") — batched sweep != per-leaf "
                                     "belief_features; do NOT ship\n";
                        return 1;
                    }
                    ++sweep_checks;
                }
            }
        }
        std::cout << "RESULT: PASS bit-identity (" << row_checks << " full-row (featurize_batch == per-leaf "
                     "build) + " << sweep_checks << " belief-sweep (belief_features_batch == per-leaf "
                     "belief_features) byte-for-byte comparisons across B in {8,32,64} x 6 batches)\n";
    }

    // ------------------- A/B: batched belief sweep vs B per-leaf belief_features (already-AVX2) ---------
    std::cout << "\nA/B: BatchFeaturizer::belief_features_batch vs B per-leaf chocofarm::belief_features "
                 "(the already-AVX2 bitset arm). ratio = batched/per-leaf; speedup>0 => batched faster; "
                 "FASTER iff 95% CI wholly below 1.0:\n";
    std::cout << std::setw(8) << "B" << std::setw(12) << "perleaf us" << std::setw(12) << "batched us"
              << std::setw(9) << "ratio" << std::setw(18) << "95% CI(ratio)"
              << std::setw(10) << "speedup%" << std::setw(9) << "verdict" << "\n";

    for (size_t B : Bs) {
        Batch batch = make_batch(env, pool, B, rng);
        std::vector<chocofarm::Belief> bel_var; bel_var.reserve(B);
        for (size_t b = 0; b < B; ++b) bel_var.emplace_back(batch.beliefs[b]);
        std::vector<const chocofarm::Belief*> bptr(B);
        for (size_t b = 0; b < B; ++b) bptr[b] = &bel_var[b];
        std::vector<chocofarm::BeliefFeatures> out_b, out_s(B);

        auto time_perleaf = [&]() -> double {
            using clk = std::chrono::steady_clock;
            long it = 0; double sink = 0.0; const auto t0 = clk::now(); double el = 0.0;
            do {
                for (size_t b = 0; b < B; ++b) {
                    out_s[b] = chocofarm::belief_features(env, bel_var[b]);
                    sink += out_s[b].marg_sum + out_s[b].sharpness;
                }
                ++it; el = std::chrono::duration<double>(clk::now() - t0).count();
            } while (el < budget);
            g_sink += sink; return el * 1e6 / static_cast<double>(it);
        };
        auto time_batched = [&]() -> double {
            using clk = std::chrono::steady_clock;
            long it = 0; double sink = 0.0; const auto t0 = clk::now(); double el = 0.0;
            do {
                bf.belief_features_batch(std::span<const chocofarm::Belief* const>(bptr.data(), B), out_b);
                for (size_t b = 0; b < B; ++b) sink += out_b[b].marg_sum + out_b[b].sharpness;
                ++it; el = std::chrono::duration<double>(clk::now() - t0).count();
            } while (el < budget);
            g_sink += sink; return el * 1e6 / static_cast<double>(it);
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
        std::mt19937_64 brng(0x1234ull + B);
        const Boot bt = bootstrap_ratio_ci(ratios, brng);
        const double speedup = (1.0 / bt.med - 1.0) * 100.0;
        const char* verdict = (bt.hi < 1.0) ? "FASTER" : (bt.lo > 1.0) ? "SLOWER" : "NULL";
        std::ostringstream ci;
        ci << "[" << std::fixed << std::setprecision(3) << bt.lo << "," << bt.hi << "]";
        std::cout << std::fixed << std::setprecision(2)
                  << std::setw(8) << B << std::setw(12) << ss.median << std::setw(12) << cs.median
                  << std::setw(9) << std::setprecision(4) << bt.med
                  << std::setw(18) << ci.str()
                  << std::setw(9) << std::setprecision(1) << speedup << "%"
                  << std::setw(9) << verdict << "\n";
    }
    std::cout << "\nNOTE: the per-leaf baseline ALREADY uses the AVX2 popcount primitive on this base "
                 "(belief_features over the bitset arm => popcount_and => vpshufb), so this A/B isolates the "
                 "BATCH-SPECIFIC mask-resident-tiling increment (the de-risk note's ~+30%), NOT the no-seam "
                 "primitive swap. ratio<1 => batched faster; speedup% = (1/ratio - 1)*100.\n";
    return 0;
}
