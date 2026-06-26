// throughput-lab/fused_jax/belief_batch_encode.cpp
// Purpose: the C++ SEND SIDE of the fused-JAX BatchPredict component (lever #1). Reads the env and
//   writes the ACTUAL belief-wire bytes (belief_wire.hpp) the JAX side decodes:
//     --setup   <path>  : the env-static SETUP frame (world_feature_matrix column bitsets, sent ONCE).
//     --request <path>  : a per-batch REQUEST frame (B belief leaves: loc, collected, rank-bitset).
//     --oracle  <path>  : a JSON PARITY ORACLE — per leaf, the C++ chocofarm::belief_features output in
//                         FULL DOUBLE precision (marg / p_pos / informative / marg_sum / sharpness /
//                         nonempty). The JAX featurization is diffed against this (re-confirm the
//                         de-risk; ADR-0012: the C++ sweep is the cross-language SSOT, JAX must agree).
//
//   The matmul's right operand (world_feature_matrix) is env-static -> it rides the SETUP frame, NOT the
//   per-leaf request (the design's "sent ONCE at setup"). The belief leaves are the per-batch payload.
//   --B selects the batch size; the beliefs are a spread of nb (prefixes of worlds() + strides), the
//   same population the de-risk used, so the parity numbers are comparable on this base.
//
//   This is an ADDITIVE COMPONENT tool. It touches NO production path (it READS env + belief_features,
//   writes its own files). It is a SEPARATE executable from the producer (ADR-0012 P3 one-owner: the
//   producer owns the feature wire, this tool owns the belief wire export).
//
//   Run:
//     belief-batch-encode --instance chocofarm/data/instance.json --faces chocofarm/data/faces.json
//         --B 32 --setup /tmp/setup.bin --request /tmp/request.bin --oracle /tmp/oracle.json
//
// Public Domain (The Unlicense).
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <map>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include <nlohmann/json.hpp>

#include "belief_wire.hpp"
#include "chocofarm/env.hpp"
#include "chocofarm/feature_compute.hpp"  // chocofarm::belief_features (the production sweep entry)
#include "chocofarm/features.hpp"
#include "chocofarm/instance.hpp"

namespace bw = tlab::bwire;

namespace {
[[nodiscard]] std::optional<std::string_view> opt(std::span<const std::string_view> args,
                                                  std::string_view name) {
    for (size_t i = 1; i + 1 < args.size(); ++i)
        if (args[i] == name) return args[i + 1];
    return std::nullopt;
}

// world value -> RANK (its position in env.worlds(): combinations order). The rank space is the matmul's
// world axis. Mirrors belief_features_export / belief_sweep_oracle_check.
[[nodiscard]] std::map<uint32_t, size_t> rank_of(const chocofarm::Environment& env) {
    std::map<uint32_t, size_t> m;
    const std::vector<uint32_t>& worlds = env.worlds();
    for (size_t r = 0; r < worlds.size(); ++r) m.emplace(worlds[r], r);
    return m;
}

// Pack a flat world-set (a subset of env.worlds()) into a kW64-word rank bitset: bit r set iff the
// rank-r world is in the set. This IS the belief_indicator the JAX side multiplies.
[[nodiscard]] std::vector<uint64_t> pack_rank_bits(const std::map<uint32_t, size_t>& rank,
                                                   const std::vector<uint32_t>& flat, int kW64) {
    std::vector<uint64_t> bits(static_cast<size_t>(kW64), 0);
    for (uint32_t w : flat) {
        const size_t r = rank.at(w);
        bits[r >> 6] |= (uint64_t{1} << (r & 63u));
    }
    return bits;
}

void write_bytes(std::string_view path, const std::vector<unsigned char>& bytes) {
    std::ofstream f(std::string(path), std::ios::binary);
    if (!f) { std::cerr << "belief-batch-encode: FATAL: cannot open " << path << " for write\n"; std::exit(1); }
    f.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
}
}  // namespace

int main(int argc, char** argv) {
    std::vector<std::string_view> args(argv, argv + argc);
    auto instance = opt(args, "--instance");
    auto faces    = opt(args, "--faces");
    auto setup_p  = opt(args, "--setup");
    auto req_p    = opt(args, "--request");
    auto oracle_p = opt(args, "--oracle");
    auto B_s      = opt(args, "--B");
    if (!instance || !faces || !setup_p || !req_p || !oracle_p) {
        std::cerr << "usage: belief-batch-encode --instance <p> --faces <p> --B <n> "
                     "--setup <out> --request <out> --oracle <out.json>\n";
        return 2;
    }
    const int Bwant = B_s ? std::stoi(std::string(*B_s)) : 32;
    if (Bwant <= 0) { std::cerr << "belief-batch-encode: FATAL: --B must be >= 1\n"; return 1; }

    auto inst = chocofarm::load_instance(*instance, *faces);
    if (!inst) { std::cerr << "belief-batch-encode: FATAL: " << inst.error().message << "\n"; return 1; }
    chocofarm::Environment env(*inst);

    const int N = env.N();
    const int nD = env.n_detectors();
    const std::vector<uint32_t>& all = env.worlds();
    const auto nworlds = static_cast<int>(all.size());
    const int kW64 = env.kW64();
    if (kW64 <= 0) {
        // No rank-bitset column masks without an enumerable, gated env (ADR-0002 — do not emit a
        // degenerate frame the JAX side would silently mis-shape).
        std::cerr << "belief-batch-encode: FATAL: env kW64=" << kW64 << " (<=0): no rank-bitset column "
                     "masks (env not enumerable / bitset arm not built).\n";
        return 1;
    }
    if (N > 32) {
        // collected is a 32-bit mask on the wire; N<=32 on the live env. Fail loud if an instance
        // violates that rather than silently truncating the collected set (ADR-0002).
        std::cerr << "belief-batch-encode: FATAL: N=" << N << " > 32 (collected mask is u32 on the wire).\n";
        return 1;
    }

    // ---- (1) the SETUP frame: the env-static world_feature_matrix, column-major rank bitsets. ----
    std::vector<uint64_t> matrix;
    matrix.reserve(static_cast<size_t>(N + nD) * kW64);
    for (int t = 0; t < N; ++t) { auto c = env.treasure_mask(t); matrix.insert(matrix.end(), c.begin(), c.end()); }
    for (int j = 0; j < nD; ++j){ auto c = env.detector_mask(j); matrix.insert(matrix.end(), c.begin(), c.end()); }
    write_bytes(*setup_p, bw::encode_setup(static_cast<bw::count_t>(N), static_cast<bw::count_t>(nD),
                                           static_cast<bw::count_t>(nworlds), static_cast<bw::count_t>(kW64),
                                           matrix));

    // ---- (2) a belief population (same spread the de-risk used) cycled up to B rows. ----
    std::vector<std::vector<uint32_t>> pop;
    for (size_t n : {size_t{1}, size_t{2}, size_t{3}, size_t{5}, size_t{16}, size_t{100}, size_t{1000},
                     static_cast<size_t>(nworlds) / 2, static_cast<size_t>(nworlds)}) {
        const size_t k = std::min(n, static_cast<size_t>(nworlds));
        pop.emplace_back(all.begin(), all.begin() + static_cast<std::ptrdiff_t>(k));
    }
    for (size_t step : {size_t{7}, size_t{13}}) {
        std::vector<uint32_t> strided;
        for (size_t i = 0; i < static_cast<size_t>(nworlds); i += step) strided.push_back(all[i]);
        pop.push_back(std::move(strided));
    }

    const std::map<uint32_t, size_t> rank = rank_of(env);

    // ---- (2a) the REQUEST frame (B leaves) AND (3) the parity oracle, in lock-step over the SAME
    // beliefs (so oracle row i corresponds to request leaf i). loc/collected are deterministic synthetic
    // values per leaf (the matmul ignores them; carried to make the leaf complete + the wire size honest).
    std::vector<bw::BeliefLeaf> leaves;
    nlohmann::json oracle;
    oracle["N"] = N; oracle["nD"] = nD; oracle["nworlds"] = nworlds; oracle["kW64"] = kW64;
    oracle["log_nworlds"] = std::log(static_cast<double>(nworlds));
    nlohmann::json refs = nlohmann::json::array();
    for (int i = 0; i < Bwant; ++i) {
        const std::vector<uint32_t>& world_set = pop[static_cast<size_t>(i) % pop.size()];

        bw::BeliefLeaf lf;
        lf.loc = static_cast<bw::count_t>(i);                     // synthetic standing point
        lf.collected = static_cast<bw::count_t>((i * 2654435761u) & ((N == 32) ? 0xffffffffu : ((1u << N) - 1u)));
        lf.belief = pack_rank_bits(rank, world_set, kW64);
        leaves.push_back(std::move(lf));

        // the C++ belief_features double-precision oracle for this belief (the parity SSOT).
        const chocofarm::BeliefFeatures bf = chocofarm::belief_features(env, chocofarm::FlatBelief{world_set});
        nlohmann::json r;
        r["nb"] = world_set.size();
        r["marg"] = bf.marg;
        r["p_pos"] = bf.p_pos;
        r["informative"] = bf.informative;
        r["marg_sum"] = bf.marg_sum;
        r["sharpness"] = bf.sharpness;
        r["nonempty"] = bf.nonempty;
        refs.push_back(std::move(r));
    }
    oracle["leaves"] = std::move(refs);

    write_bytes(*req_p, bw::encode_request(leaves, static_cast<bw::count_t>(kW64)));

    const std::string oracle_path(*oracle_p);
    std::ofstream of(oracle_path);
    if (!of) { std::cerr << "belief-batch-encode: FATAL: cannot open " << *oracle_p << "\n"; return 1; }
    of << oracle.dump() << "\n";

    std::cerr << "belief-batch-encode: N=" << N << " nD=" << nD << " nworlds=" << nworlds
              << " kW64=" << kW64 << " B=" << Bwant
              << "  setup=" << *setup_p << "  request=" << *req_p << "  oracle=" << *oracle_p << "\n";
    return 0;
}
