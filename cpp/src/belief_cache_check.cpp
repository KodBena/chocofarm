// cpp/src/belief_cache_check.cpp
// Purpose: NET the FeatureBuilder belief-memo's full-equality collision guard (ADR-0011 — net the
//   guard, do not trust it). The (count, first, last) belief_key fingerprint is collision-RESISTANT,
//   not collision-FREE; the memo distinguishes two distinct beliefs that share a fingerprint only via
//   the full bw-equality check on a bucket hit. This FORCES such a collision and asserts:
//     (a) each belief gets ITS OWN features (the guard works — a collision is not mis-served),
//     (b) a true cache HIT is bit-identical to the belief's first build, and
//     (c) the warm-cache value equals a cold recompute on a fresh builder (hit == miss).
//   No redis, no net — pure FeatureBuilder. A separate executable (ADR-0012 P3, one-owner): this tool
//   owns the belief-memo correctness fixture. Public Domain (The Unlicense).
#include <cstdint>
#include <iostream>
#include <optional>
#include <set>
#include <span>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>

#include "chocofarm/belief_key.hpp"
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
[[nodiscard]] bool fail(const char* msg) { std::cout << "RESULT: FAIL " << msg << "\n"; return false; }
}  // namespace

int main(int argc, char** argv) {
    std::vector<std::string_view> args(argv, argv + argc);
    std::optional<std::string_view> instance = opt(args, "--instance");
    std::optional<std::string_view> faces = opt(args, "--faces");
    if (!instance || !faces) {
        std::cerr << "usage: belief-cache-check --instance <p> --faces <p>\n";
        return 2;
    }
    auto inst = chocofarm::load_instance(*instance, *faces);
    if (!inst) { std::cerr << "belief-cache-check: FATAL: " << inst.error().message << "\n"; return 1; }
    chocofarm::Environment env(*inst);
    chocofarm::FeatureBuilder fb(env);
    const chocofarm::Point loc = env.entry_point();
    const std::set<int> coll;

    // Two DISTINCT beliefs that SHARE a (count, first, last) fingerprint (different MIDDLE world) — the
    // exact collision the full-equality guard must distinguish. front()=1, back()=5, size=3 for both.
    const std::vector<uint32_t> bw1 = {1u, 3u, 5u};
    const std::vector<uint32_t> bw2 = {1u, 4u, 5u};
    bool ok = true;
    if (chocofarm::belief_key(bw1) != chocofarm::belief_key(bw2))
        ok = fail("test setup: bw1/bw2 do not share a fingerprint");

    const std::vector<double> f1 = fb.build(loc, bw1, coll);   // miss: compute + store
    const std::vector<double> f2 = fb.build(loc, bw2, coll);   // miss: same fingerprint, full-eq fails -> own bucket entry
    if (ok && f1 == f2) ok = fail("colliding beliefs returned IDENTICAL features (the guard is broken)");

    const std::vector<double> f1b = fb.build(loc, bw1, coll);  // HIT for bw1 (must find bw1's entry, NOT bw2's)
    if (ok && f1b != f1) ok = fail("cache hit for bw1 is not bit-identical to its first build");
    const std::vector<double> f2b = fb.build(loc, bw2, coll);  // HIT for bw2
    if (ok && f2b != f2) ok = fail("cache hit for bw2 is not bit-identical to its first build");

    // a cold recompute on a FRESH builder (always a miss) must equal the warm-cache value (hit == miss).
    chocofarm::FeatureBuilder fb_cold(env);
    if (ok && fb_cold.build(loc, bw1, coll) != f1) ok = fail("warm-cache value != cold recompute (bw1)");
    if (ok && fb_cold.build(loc, bw2, coll) != f2) ok = fail("warm-cache value != cold recompute (bw2)");

    if (!ok) return 1;
    const auto k = chocofarm::belief_key(bw1);
    std::cout << "RESULT: PASS belief-memo collision guard + hit-exactness (dim=" << fb.dim()
              << ", shared fingerprint=(" << std::get<0>(k) << "," << std::get<1>(k) << ","
              << std::get<2>(k) << "))\n";
    return 0;
}
