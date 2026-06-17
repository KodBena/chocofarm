// cpp/src/belief_filter_bench.cpp
// Purpose: the belief-FILTER compaction A/B gate (NOT the runner) — times the PRODUCTION compaction
//   (chocofarm::filter_inplace, the idiomatic std::erase_if) against candidate alternatives across the
//   belief sizes the search actually filters, on REALISTIC beliefs (the full world-set narrowed by a random
//   observation sequence consistent with a sampled true world, then a fresh INFORMATIVE detector — NOT a
//   prefix, whose predicate is perfectly predictable and meaningless here). The belief UPDATE side of the
//   same O(nb) belief problem the K=32 native profile put at ~27.6% of the client. Filter-ISOLATED by
//   subtracting a restore-only baseline; bit-exact asserted (same kept set + order; ADR-0011).
//
//   The current candidate is a hand-written BRANCHLESS stream-compaction. Result on the i5-6600 native
//   build: the idiom WINS (branchless is ~1.4-1.5x slower) — the belief predicate predicts well and the
//   idiom's scan auto-vectorizes while the branchless serial out-pointer does not. This is the gate the
//   expert's "SIMD-compress only if the filter is hot, scalar fallback" rung would re-run (drop the AVX2
//   vpcompress candidate in beside branchless_ref). Separate executable (ADR-0012 P3, one-owner). No net.
//
//   Protocol:  belief-filter-bench --instance <p> --faces <p> [--budget-s 0.3] [--trials 5]
//   Output:    a table over realistic belief sizes nb (median over trials of idiom / candidate ns/world +
//              speedup = candidate/idiom, >1 ⇒ idiom faster), then RESULT: PASS (bit-exact) / FAIL.
//
// Public Domain (The Unlicense).
#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <optional>
#include <random>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "chocofarm/env.hpp"
#include "chocofarm/instance.hpp"

namespace {
[[nodiscard]] std::optional<std::string_view> opt(std::span<const std::string_view> args,
                                                  std::string_view name) {
    for (size_t i = 1; i + 1 < args.size(); ++i)
        if (args[i] == name) return args[i + 1];
    return std::nullopt;
}
volatile std::size_t g_sink = 0;  // defeat dead-code elimination of the timed work

// The hand-written BRANCHLESS stream-compaction candidate (unconditional store + advance-by-keep) — what
// this bench measures against the production idiom. Kept here, NOT in env.cpp, because the measurement
// REJECTS it (it is slower); it lives on only as the A/B reference + the slot a future SIMD-compress joins.
std::size_t branchless_ref(std::vector<uint32_t>& bw, uint32_t mask, bool want) {
    uint32_t* out = bw.data();
    for (uint32_t w : bw) {
        *out = w;
        out += static_cast<std::ptrdiff_t>(((w & mask) != 0) == want);
    }
    bw.resize(static_cast<std::size_t>(out - bw.data()));
    return bw.size();
}

// informative over bw: the mask splits the belief (both a hit and a miss exist) — the realistic non-trivial
// filter the search applies (a detector worth sensing).
[[nodiscard]] bool informative_over(std::span<const uint32_t> bw, uint32_t mask) {
    bool hit = false, miss = false;
    for (uint32_t w : bw) {
        if ((w & mask) != 0) hit = true; else miss = true;
        if (hit && miss) return true;
    }
    return false;
}
}  // namespace

int main(int argc, char** argv) {
    std::vector<std::string_view> args(argv, argv + argc);
    std::optional<std::string_view> instance = opt(args, "--instance");
    std::optional<std::string_view> faces = opt(args, "--faces");
    if (!instance || !faces) {
        std::cerr << "usage: belief-filter-bench --instance <p> --faces <p> [--budget-s 0.3] [--trials 5]\n";
        return 2;
    }
    const double budget = opt(args, "--budget-s")
        ? std::atof(std::string(*opt(args, "--budget-s")).c_str()) : 0.3;
    const int trials = opt(args, "--trials")
        ? std::atoi(std::string(*opt(args, "--trials")).c_str()) : 5;

    auto inst = chocofarm::load_instance(*instance, *faces);
    if (!inst) { std::cerr << "belief-filter-bench: FATAL: " << inst.error().message << "\n"; return 1; }
    chocofarm::Environment env(*inst);
    const int nD = env.n_detectors();
    const std::vector<uint32_t> all = env.worlds();
    const size_t nworlds = all.size();
    const std::span<const uint32_t> masks = env.face_masks();
    std::mt19937 rng(0xC0FFEEu);

    // A realistic filter EVENT: the full world-set narrowed by random observations (consistent with a
    // sampled true world) down to ~target size, plus a fresh informative detector mask to time filtering by.
    auto make_event = [&](size_t target, std::vector<uint32_t>& bw, uint32_t& mask, bool& want) -> bool {
        std::vector<int> order(static_cast<size_t>(nD));
        std::iota(order.begin(), order.end(), 0);
        std::shuffle(order.begin(), order.end(), rng);
        const uint32_t wstar = all[rng() % nworlds];
        bw = all;
        size_t oi = 0;
        while (bw.size() > target && oi < order.size()) {
            const int d = order[oi++];
            chocofarm::filter_inplace(bw, masks[static_cast<size_t>(d)], (wstar & masks[static_cast<size_t>(d)]) != 0);
        }
        for (; oi < order.size(); ++oi) {
            const uint32_t m = masks[static_cast<size_t>(order[oi])];
            if (informative_over(bw, m)) { mask = m; want = (wstar & m) != 0; return bw.size() >= 2; }
        }
        return false;  // no informative detector left for this trajectory at this size
    };

    std::cout << "belief-filter-bench: nD=" << nD << " |worlds|=" << nworlds << " budget=" << budget
              << "s/point trials=" << trials << "  (idiom=production filter_inplace vs hand-branchless; "
              << "REALISTIC observation-filtered beliefs)\n";
    std::cout << std::setw(9) << "nb" << std::setw(8) << "keep%"
              << std::setw(13) << "idiom" << std::setw(13) << "branchless"
              << std::setw(10) << "speedup" << "   (median ns/world, filter-isolated; >1 = idiom faster)\n";

    std::vector<size_t> targets;
    for (size_t s = nworlds; s >= 8; s /= 2) targets.push_back(s);

    using clk = std::chrono::steady_clock;
    for (size_t target : targets) {
        std::vector<double> id_samples, bl_samples, keep_samples, nb_samples;
        for (int t = 0; t < trials; ++t) {
            std::vector<uint32_t> master;
            uint32_t mask = 0;
            bool want = true;
            if (!make_event(target, master, mask, want)) continue;
            const size_t nb = master.size();

            // bit-exact net: the production idiom and the branchless candidate must agree byte-for-byte.
            std::vector<uint32_t> a = master, b = master;
            chocofarm::filter_inplace(a, mask, want);
            branchless_ref(b, mask, want);
            if (a != b) { std::cout << "RESULT: FAIL idiom != branchless at nb=" << nb << "\n"; return 1; }

            std::vector<uint32_t> work;
            work.reserve(nb);
            auto time_loop = [&](int which) {
                long calls = 0;
                const auto t0 = clk::now();
                double el = 0.0;
                do {
                    for (int r = 0; r < 8; ++r) {
                        work.assign(master.begin(), master.end());   // restore the belief (the baseline)
                        if (which == 1) chocofarm::filter_inplace(work, mask, want);
                        else if (which == 2) branchless_ref(work, mask, want);
                        g_sink += work.size();
                    }
                    calls += 8;
                    el = std::chrono::duration<double>(clk::now() - t0).count();
                } while (el < budget / static_cast<double>(trials));
                return el * 1e9 / static_cast<double>(calls);   // ns/call
            };
            const double base_ns = time_loop(0);
            const double id_ns = time_loop(1);   // production idiom (filter_inplace)
            const double bl_ns = time_loop(2);   // hand branchless candidate
            id_samples.push_back((id_ns - base_ns) / static_cast<double>(nb));
            bl_samples.push_back((bl_ns - base_ns) / static_cast<double>(nb));
            keep_samples.push_back(100.0 * static_cast<double>(a.size()) / static_cast<double>(nb));
            nb_samples.push_back(static_cast<double>(nb));
        }
        if (id_samples.empty()) continue;
        auto median = [](std::vector<double> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; };
        const double id = median(id_samples), bl = median(bl_samples);
        std::cout << std::fixed
                  << std::setw(9) << static_cast<long>(median(nb_samples))
                  << std::setw(7) << std::setprecision(1) << median(keep_samples) << "%"
                  << std::setw(13) << std::setprecision(2) << id
                  << std::setw(13) << bl
                  << std::setw(9) << std::setprecision(2) << (id > 0.0 ? bl / id : 0.0) << "x" << "\n";
    }

    std::cout << "RESULT: PASS belief-filter idiom == branchless byte-for-byte (bit-exact compaction)\n";
    return 0;
}
