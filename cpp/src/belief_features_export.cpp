// cpp/src/belief_features_export.cpp
// Purpose: a tiny DE-RISK EXPORT tool (NOT the runner) for de-risk idea #1 (the fused-JAX matmul
//   featurization — docs/notes/batchpredict-throughput-design-2026-06-26.md / derisk-jax-matmul-
//   featurization-2026-06-26.md). It dumps, as one JSON blob:
//     (1) the ENV-STATIC world_feature_matrix in its native COLUMN-BITSET form — the env already
//         builds it: treasure_mask(t) is the rank-bitset of worlds containing treasure t (column t),
//         detector_mask(j) is the rank-bitset of worlds whose detector-j cover hits (column N+j). Each
//         column is kW64 u64 words over the worlds()-rank space. So the matmul's right operand
//         (nb x (N+nD), bit-valued) is exactly the unpack of these N+nD column bitsets.
//     (2) a spread of REFERENCE (belief_indicator, C++ belief_features) pairs: for each sampled belief
//         we emit its live-world rank bitset (kW64 u64 words) AND chocofarm::belief_features's output
//         (marg, p_pos, informative, marg_sum, sharpness, nonempty) in full double precision — the
//         PARITY ORACLE the JAX f32 matmul is diffed against.
//
//   This is an ADDITIVE prototype: it touches no production path, only READS env + belief_features. It
//   is a SEPARATE executable from the runner (ADR-0012 P3, one-owner): the runner owns the wire, this
//   tool owns the de-risk export fixture. No redis, no net — pure env + features.
//
//   Protocol:  belief-features-export --instance <p> --faces <p>   (writes JSON to stdout)
//
// Public Domain (The Unlicense).
#include <cmath>
#include <cstdint>
#include <iostream>
#include <map>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include <nlohmann/json.hpp>

#include "chocofarm/env.hpp"
#include "chocofarm/feature_compute.hpp"  // chocofarm::belief_features (the production sweep entry)
#include "chocofarm/features.hpp"
#include "chocofarm/instance.hpp"

namespace {
[[nodiscard]] std::optional<std::string_view> opt(std::span<const std::string_view> args,
                                                  std::string_view name) {
    for (size_t i = 1; i + 1 < args.size(); ++i)
        if (args[i] == name) return args[i + 1];
    return std::nullopt;
}

// world value -> RANK (its position in env.worlds(): combinations order, NOT numeric — see FlatBelief).
// The rank space is the matmul's world axis: column bit r is set iff the rank-r world is in the column's
// set; a belief_indicator bit r is set iff the rank-r world is live. Mirrors belief_sweep_oracle_check.
[[nodiscard]] std::map<uint32_t, size_t> rank_of(const chocofarm::Environment& env) {
    std::map<uint32_t, size_t> m;
    const std::vector<uint32_t>& worlds = env.worlds();
    for (size_t r = 0; r < worlds.size(); ++r) m.emplace(worlds[r], r);
    return m;
}

// Pack a flat world-set (a SUBSET of env.worlds(), any order) into a kW64-word rank bitset: bit r set
// iff the rank-r world is in the set. This is the wire encoding the de-risk weighs (the belief_indicator
// the JAX side multiplies). Returns kW64 u64 words.
[[nodiscard]] std::vector<uint64_t> pack_rank_bits(const std::map<uint32_t, size_t>& rank,
                                                   const std::vector<uint32_t>& flat, int kW64) {
    std::vector<uint64_t> bits(static_cast<size_t>(kW64), 0);
    for (uint32_t w : flat) {
        const size_t r = rank.at(w);  // every belief here is a subset of worlds() (invariant)
        bits[r >> 6] |= (uint64_t{1} << (r & 63u));
    }
    return bits;
}

// A column bitset (treasure_mask/detector_mask) is a span<const uint64_t> of kW64 words over rank space.
// Copy it to a json array of u64 (decimal — nlohmann emits unsigned 64-bit exactly).
[[nodiscard]] nlohmann::json words_to_json(std::span<const uint64_t> words) {
    nlohmann::json a = nlohmann::json::array();
    for (uint64_t w : words) a.push_back(w);
    return a;
}
}  // namespace

int main(int argc, char** argv) {
    std::vector<std::string_view> args(argv, argv + argc);
    std::optional<std::string_view> instance = opt(args, "--instance");
    std::optional<std::string_view> faces = opt(args, "--faces");
    if (!instance || !faces) {
        std::cerr << "usage: belief-features-export --instance <p> --faces <p>  (JSON to stdout)\n";
        return 2;
    }
    auto inst = chocofarm::load_instance(*instance, *faces);
    if (!inst) { std::cerr << "belief-features-export: FATAL: " << inst.error().message << "\n"; return 1; }
    chocofarm::Environment env(*inst);

    const int N = env.N();
    const int nD = env.n_detectors();
    const std::vector<uint32_t>& all = env.worlds();
    const size_t nworlds = all.size();
    const int kW64 = env.kW64();
    if (kW64 <= 0) {
        // The bitset arm (and thus the rank-bitset column masks) requires an enumerable, gated env.
        // A non-enumerable env has no kW64/column masks to export — fail loud (ADR-0002), do not emit a
        // degenerate blob the JAX side would silently mis-shape.
        std::cerr << "belief-features-export: FATAL: env kW64=" << kW64 << " (<=0): no rank-bitset column "
                     "masks (env not enumerable / bitset arm not built). The matmul export needs the "
                     "column masks treasure_mask/detector_mask, built only when the bitset arm is gated.\n";
        return 1;
    }

    nlohmann::json out;
    out["N"] = N;
    out["nD"] = nD;
    out["nworlds"] = nworlds;
    out["kW64"] = kW64;
    out["log_nworlds"] = std::log(static_cast<double>(nworlds));

    // (1) the env-static world_feature_matrix, COLUMN-MAJOR as rank bitsets. columns 0..N-1 are the
    // treasure columns (bit r = rank-r world contains treasure t); columns N..N+nD-1 are the detector
    // cover columns (bit r = rank-r world's detector-j cover hits). The JAX side unpacks these kW64-word
    // columns into a dense nworlds x (N+nD) bit matrix (the matmul's right operand).
    nlohmann::json cols = nlohmann::json::array();
    for (int t = 0; t < N; ++t) cols.push_back(words_to_json(env.treasure_mask(t)));
    for (int j = 0; j < nD; ++j) cols.push_back(words_to_json(env.detector_mask(j)));
    out["columns"] = std::move(cols);  // length N+nD; each a kW64-word rank bitset

    // (2) the reference beliefs. A spread mirroring the oracle's: prefixes small->full (varied per-detector
    // cover counts) + two strides (cover mixes the prefixes do not produce). For each, the belief_indicator
    // (rank bitset) AND the C++ belief_features (double precision) — the parity oracle.
    std::vector<std::vector<uint32_t>> beliefs;
    for (size_t n : {size_t{1}, size_t{2}, size_t{3}, size_t{5}, size_t{16}, size_t{100}, size_t{1000},
                     nworlds / 2, nworlds}) {
        const size_t k = std::min(n, nworlds);
        beliefs.emplace_back(all.begin(), all.begin() + static_cast<std::ptrdiff_t>(k));
    }
    for (size_t step : {size_t{7}, size_t{13}}) {
        std::vector<uint32_t> strided;
        for (size_t i = 0; i < nworlds; i += step) strided.push_back(all[i]);
        beliefs.push_back(std::move(strided));
    }

    const std::map<uint32_t, size_t> rank = rank_of(env);
    nlohmann::json refs = nlohmann::json::array();
    for (const std::vector<uint32_t>& bw : beliefs) {
        // C++ reference features via the production sweep (the parity oracle). FlatBelief over the subset.
        const chocofarm::BeliefFeatures bf = chocofarm::belief_features(env, chocofarm::FlatBelief{bw});
        nlohmann::json r;
        r["nb"] = bw.size();
        r["indicator"] = words_to_json(pack_rank_bits(rank, bw, kW64));  // kW64-word rank bitset
        r["marg"] = bf.marg;              // N doubles
        r["p_pos"] = bf.p_pos;            // nD doubles
        r["informative"] = bf.informative;// nD doubles (0/1)
        r["marg_sum"] = bf.marg_sum;
        r["sharpness"] = bf.sharpness;
        r["nonempty"] = bf.nonempty;
        refs.push_back(std::move(r));
    }
    out["beliefs"] = std::move(refs);

    // Full double precision on stdout (the parity bar reads these back as the oracle).
    std::cout << out.dump() << "\n";
    return 0;
}
