// cpp/src/belief_sweep_oracle_check.cpp
// Purpose: the BIT-EXACT oracle for the belief sweep (chocofarm::belief_features) — the regression net
//   the §A.4 rewrite and every later rung (SIMD/pos-popcount, the Part B decision diagram) diff against
//   (belief_features_and_decision_diagram_note.md §A.5/B.3; ADR-0011: net the rewrite, do not trust it).
//   It computes each sample belief's BeliefFeatures TWO independent ways and asserts they are byte-equal:
//     (production) chocofarm::belief_features — contiguous env.face_masks(), branchless integer fused sweep
//     (reference)  a dead-simple naive count via env.observe (the array-of-structs path), same `* inv` spec
//   The two share ONLY the math spec, not the implementation: matching counts therefore prove the
//   contiguous-mask derivation (face_masks()[j] == faces[j].bitmask) and the branchless/fused transcription
//   are exact. The reference fixes the `* inv` convention (the settled re-baseline — marg AND p_pos use
//   `* inv`), so the oracle is the home of "the *inv sweep IS the reference." Cross-language vs Python stays
//   at the P6 behavioral bar (the gumbel parity); THIS is the in-language bit-exact bar.
//
//   Protocol:  belief-sweep-oracle-check --instance <p> --faces <p>
//   A separate executable (ADR-0012 P3, one-owner): this tool owns the belief-sweep bit-exactness fixture.
//   No redis, no net — pure compute. Public Domain (The Unlicense).
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

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
[[nodiscard]] bool fail(const std::string& msg) { std::cout << "RESULT: FAIL " << msg << "\n"; return false; }

// The INDEPENDENT naive reference: env.observe (the array-of-structs path the production replaces with a
// contiguous span), the simplest scalar loops, the SAME `* inv` spec. Deliberately NOT branchless / fused
// so it shares no code path with the production beyond the math definition.
[[nodiscard]] chocofarm::BeliefFeatures reference(const chocofarm::Environment& env,
                                                  std::span<const uint32_t> bw,
                                                  int N, int nD, double log_nworlds) {
    chocofarm::BeliefFeatures bf;
    bf.marg.assign(N, 0.0);
    bf.p_pos.assign(nD, 0.0);
    bf.informative.assign(nD, 0.0);
    const size_t nb = bw.size();
    if (nb == 0) return bf;  // empty: all derived quantities 0 (matches belief_features_empty)
    std::vector<int64_t> bc(N, 0), dc(nD, 0);
    for (uint32_t w : bw) {
        for (int t = 0; t < N; ++t) if ((w >> t) & 1u) bc[t] += 1;
        for (int j = 0; j < nD; ++j) if (env.observe(j, w)) dc[j] += 1;   // <- the independent path
    }
    const double inv = 1.0 / static_cast<double>(nb);
    for (int t = 0; t < N; ++t) { bf.marg[t] = static_cast<double>(bc[t]) * inv; bf.marg_sum += bf.marg[t]; }
    for (int j = 0; j < nD; ++j) {
        bf.p_pos[j] = static_cast<double>(dc[j]) * inv;
        bf.informative[j] = (dc[j] > 0 && dc[j] < static_cast<int64_t>(nb)) ? 1.0 : 0.0;
    }
    bf.sharpness = std::log(static_cast<double>(nb)) / log_nworlds;
    bf.nonempty = 1.0;
    return bf;
}

// Byte-equal every field of two BeliefFeatures (== on double vectors/scalars: the values are produced by
// identical float ops on identical integer counts, so == is the exact bit comparison — no NaN/-0.0 arise
// from counts >= 0 and inv > 0). On a mismatch, name the field for the failing belief.
[[nodiscard]] bool equal_features(const chocofarm::BeliefFeatures& a, const chocofarm::BeliefFeatures& b,
                                  size_t nb, std::string& why) {
    auto note = [&](const char* f) { why = std::string(f) + " (nb=" + std::to_string(nb) + ")"; return false; };
    if (a.marg != b.marg) return note("marg");
    if (a.p_pos != b.p_pos) return note("p_pos");
    if (a.informative != b.informative) return note("informative");
    if (a.marg_sum != b.marg_sum) return note("marg_sum");
    if (a.sharpness != b.sharpness) return note("sharpness");
    if (a.nonempty != b.nonempty) return note("nonempty");
    return true;
}
}  // namespace

int main(int argc, char** argv) {
    std::vector<std::string_view> args(argv, argv + argc);
    std::optional<std::string_view> instance = opt(args, "--instance");
    std::optional<std::string_view> faces = opt(args, "--faces");
    if (!instance || !faces) {
        std::cerr << "usage: belief-sweep-oracle-check --instance <p> --faces <p>\n";
        return 2;
    }
    auto inst = chocofarm::load_instance(*instance, *faces);
    if (!inst) { std::cerr << "belief-sweep-oracle-check: FATAL: " << inst.error().message << "\n"; return 1; }
    chocofarm::Environment env(*inst);
    const int N = env.N();
    const int nD = env.n_detectors();
    const std::span<const uint32_t> masks = env.face_masks();
    const std::vector<uint32_t>& all = env.worlds();
    const size_t nworlds = all.size();
    const double log_nworlds = std::log(static_cast<double>(nworlds));

    // Sample beliefs: the empty belief, prefixes spanning small -> full (varied per-detector cover counts),
    // and a strided subset (every 13th world) for a cover mix the prefixes do not produce.
    std::vector<std::vector<uint32_t>> beliefs;
    beliefs.emplace_back();  // nb == 0
    for (size_t n : {size_t{1}, size_t{2}, size_t{3}, size_t{5}, size_t{16}, size_t{100}, size_t{1000},
                     nworlds / 2, nworlds}) {
        const size_t k = std::min(n, nworlds);
        beliefs.emplace_back(all.begin(), all.begin() + static_cast<std::ptrdiff_t>(k));
    }
    { std::vector<uint32_t> strided; for (size_t i = 0; i < nworlds; i += 13) strided.push_back(all[i]);
      beliefs.push_back(std::move(strided)); }

    bool ok = true;
    std::string why;
    size_t checked = 0;
    for (const std::vector<uint32_t>& bw : beliefs) {
        const chocofarm::BeliefFeatures prod = chocofarm::belief_features(bw, masks, N, nD, log_nworlds);
        const chocofarm::BeliefFeatures ref = reference(env, bw, N, nD, log_nworlds);
        if (!equal_features(prod, ref, bw.size(), why)) {
            ok = fail("production belief_features != naive reference at field " + why);
            break;
        }
        ++checked;
    }

    if (!ok) return 1;
    std::cout << "RESULT: PASS belief-sweep bit-exact oracle (N=" << N << " nD=" << nD
              << " |worlds|=" << nworlds << "; " << checked << " beliefs, production == naive reference"
              << " byte-for-byte, *inv convention)\n";
    return 0;
}
