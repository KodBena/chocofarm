// cpp/src/batched_featurization_proto.cpp
// Purpose: a PROTOTYPE de-risk bench (idea #3, docs/notes/derisk-batched-featurization-2026-06-26.md) —
//   does featurizing B parked beliefs TOGETHER, in a cache-restructured (mask-major) pass, beat B separate
//   per-leaf chocofarm::belief_features (the BITSET arm) calls? The producer's cursor parks B leaves per
//   RTT (the same B batched to the net); the belief-features sweep (popcount-AND over the env-static
//   treasure/detector masks) is ~55% of producer compute. The mask matrix is (N+nD)*kW64*8 ≈ 121.5 KiB —
//   it does NOT fit L1 (32 KiB) and ~fills half of L2 (256 KiB). B separate calls RE-STREAM that whole
//   matrix B times. The batched pass transposes the loop nest to MASK-MAJOR: load each mask row ONCE and
//   popcount-AND it against ALL B beliefs while it is hot, so the mask matrix streams ~once instead of B
//   times (the belief matrix, B*kW64*8 ≈ B*1.9 KiB, is the smaller operand — it re-streams instead).
//
//   This is an ADDITIVE prototype: it does NOT touch the production search / feature core. It reuses
//   chocofarm::belief_features (the BITSET arm) as the bit-identity REFERENCE — the batched kernel here
//   re-derives the same Σ_w bit_t(w) / Σ_w [(w&mask_j)!=0] integer counts and the IDENTICAL Phase-2 `* inv`
//   maps, and asserts byte-for-byte equality (the belief-sweep oracle's equal_features style).
//
//   It is NOT wired into the search and does NOT decide anything: it answers ONE question (is the batched
//   sweep faster, by how much) BEFORE the BatchPredict seam is built. A null result is a valid finding.
//
//   Protocol:  batched-featurization-proto --instance <p> --faces <p>
//                                          [--budget-s 0.25] [--reps 12]
//   A separate executable (ADR-0012 P3, one-owner). No redis, no net — pure compute. Public Domain.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <optional>
#include <random>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include <immintrin.h>  // AVX2 vpshufb popcount (the stronger batched primitive; -march=native => AVX2)

#include "chocofarm/belief_bitset_ops.hpp"  // popcount_and (the same ACL the production arm uses)
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

// ---------------------------------------------------------------------------------------------------
// The BATCHED (mask-major) belief-features kernel — the PROTOTYPE under test.
//
// Inputs: `beliefs` is B BitsetBeliefs (each kw64_ live words, count_ = nb). `env` carries the env-static
// treasure_mask(t) / detector_mask(j) rows (each kW64 words, contiguous). Output: `out[b]` is the
// BeliefFeatures for beliefs[b] — byte-identical to belief_features(env, beliefs[b]) for every b.
//
// LAYOUT (the cache restructure): the production per-leaf arm loops MASK rows in the INNER position (for
// one belief, sweep all N+nD masks); B leaves => the (N+nD)*kW64-word mask matrix is re-streamed B times.
// Here the loop nest is TRANSPOSED to MASK-MAJOR: the OUTER loop walks the N+nD mask rows; the INNER loop
// walks the B beliefs. Each mask row's kW64 words are read once from memory and reused across all B
// popcount_and calls while still hot in L1/registers; only the belief words (the smaller B*kW64 operand)
// re-stream. The per-(mask,belief) primitive is the SAME popcount_and the production arm uses — so the
// Phase-1 integer counts are identical by construction; only the ORDER of memory access changes. Phase 2
// (the `* inv` maps + informative split + sharpness/nonempty) is byte-for-byte the production Phase 2.
//
// Counts are staged in a B-major scratch (cnt_marg[b*N + t], cnt_det[b*nD + j]) so the mask-major inner
// loop writes contiguously per mask column across beliefs; Phase 2 then reads them back per belief.
// AVX2 vpshufb (nibble-LUT) popcount of 4×uint64 lanes, horizontal-summed per 64-bit lane (sad_epu8).
[[gnu::target("avx2")]] inline __m256i popcnt256(__m256i v) {
    const __m256i lut = _mm256_setr_epi8(
        0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4, 0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4);
    const __m256i lo_mask = _mm256_set1_epi8(0x0f);
    const __m256i lo = _mm256_and_si256(v, lo_mask);
    const __m256i hi = _mm256_and_si256(_mm256_srli_epi16(v, 4), lo_mask);
    const __m256i pc = _mm256_add_epi8(_mm256_shuffle_epi8(lut, lo), _mm256_shuffle_epi8(lut, hi));
    return _mm256_sad_epu8(pc, _mm256_setzero_si256());  // 4 partial counts, one per 64-bit lane
}
// AVX2 popcount(b & m) over W words, 4 words/iter; scalar tail. Returns the SAME integer count as
// popcount_and (popcount is order-independent), so the staged counts stay bit-identical.
[[gnu::target("avx2")]] inline uint64_t pc_and_avx2(const uint64_t* b, const uint64_t* m, size_t W) {
    __m256i acc = _mm256_setzero_si256();
    size_t w = 0;
    for (; w + 4 <= W; w += 4)
        acc = _mm256_add_epi64(acc, popcnt256(_mm256_and_si256(
            _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + w)),
            _mm256_loadu_si256(reinterpret_cast<const __m256i*>(m + w)))));
    uint64_t tmp[4]; _mm256_storeu_si256(reinterpret_cast<__m256i*>(tmp), acc);
    uint64_t s = tmp[0] + tmp[1] + tmp[2] + tmp[3];
    for (; w < W; ++w) s += static_cast<uint64_t>(std::popcount(b[w] & m[w]));
    return s;
}

// Which Phase-1 (count-staging) kernel to run. Phase 2 is shared + byte-identical across all three.
enum class Kern {
    Scalar,     // mask-major loop transpose, SAME scalar popcount_and primitive (the original prototype)
    Avx2,       // mask-major, AVX2 vpshufb primitive — a PRIMITIVE swap (also speeds the per-leaf path)
    Avx2Tiled,  // mask-major + AVX2 + the mask word kept resident across a 4-belief tile — BATCH-SPECIFIC
};

// Shared Phase 2 (the IDENTICAL `* inv` pointwise maps the production arm runs) over the staged B-major
// counts; byte-for-byte the production Phase 2.
void phase2(const std::vector<chocofarm::WorldCountRep>& cnt_marg,
            const std::vector<chocofarm::WorldCountRep>& cnt_det,
            const std::vector<chocofarm::WorldCountRep>& nb, size_t Nn, size_t nDn, size_t B,
            double log_nworlds, std::vector<chocofarm::BeliefFeatures>& out) {
    using chocofarm::WorldCountRep;
    out.resize(B);
    for (size_t b = 0; b < B; ++b) {
        chocofarm::BeliefFeatures& bf = out[b];
        bf.marg.assign(Nn, 0.0);
        bf.p_pos.assign(nDn, 0.0);
        bf.informative.assign(nDn, 0.0);
        bf.marg_sum = 0.0;
        const WorldCountRep nbb = nb[b];
        const double inv = 1.0 / static_cast<double>(nbb);
        for (size_t t = 0; t < Nn; ++t) {
            bf.marg[t]   = static_cast<double>(cnt_marg[b * Nn + t]) * inv;
            bf.marg_sum += bf.marg[t];  // treasure-id order (P6) — matches the production arm exactly
        }
        for (size_t j = 0; j < nDn; ++j) {
            const WorldCountRep dc = cnt_det[b * nDn + j];
            bf.p_pos[j]       = static_cast<double>(dc) * inv;
            bf.informative[j] = (dc > 0 && dc < nbb) ? 1.0 : 0.0;
        }
        bf.sharpness = std::log(static_cast<double>(nbb)) / log_nworlds;
        bf.nonempty  = 1.0;
    }
}

// Stage one mask column's counts across the B beliefs into `dst[b*stride + col]`, per the chosen kernel.
// All three produce the SAME integer counts (popcount is order-independent) — the bit-identity gate proves it.
void stage_column(Kern k, const uint64_t* mask, size_t W,
                  const std::vector<const uint64_t*>& belptr, size_t B, size_t col, size_t stride,
                  std::vector<chocofarm::WorldCountRep>& dst) {
    using chocofarm::WorldCountRep;
    if (k == Kern::Scalar) {
        for (size_t b = 0; b < B; ++b) {
            uint64_t s = 0; const uint64_t* bp = belptr[b];
            for (size_t w = 0; w < W; ++w) s += static_cast<uint64_t>(std::popcount(bp[w] & mask[w]));
            dst[b * stride + col] = static_cast<WorldCountRep>(s);
        }
    } else if (k == Kern::Avx2) {
        for (size_t b = 0; b < B; ++b)
            dst[b * stride + col] = static_cast<WorldCountRep>(pc_and_avx2(belptr[b], mask, W));
    } else {  // Avx2Tiled: mask word held resident in registers across a 4-belief tile
        size_t b = 0;
        for (; b + 4 <= B; b += 4) {
            const uint64_t* b0 = belptr[b]; const uint64_t* b1 = belptr[b + 1];
            const uint64_t* b2 = belptr[b + 2]; const uint64_t* b3 = belptr[b + 3];
            __m256i a0 = _mm256_setzero_si256(), a1 = a0, a2 = a0, a3 = a0;
            size_t w = 0;
            for (; w + 4 <= W; w += 4) {
                const __m256i mv = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(mask + w));
                a0 = _mm256_add_epi64(a0, popcnt256(_mm256_and_si256(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(b0 + w)), mv)));
                a1 = _mm256_add_epi64(a1, popcnt256(_mm256_and_si256(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(b1 + w)), mv)));
                a2 = _mm256_add_epi64(a2, popcnt256(_mm256_and_si256(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(b2 + w)), mv)));
                a3 = _mm256_add_epi64(a3, popcnt256(_mm256_and_si256(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(b3 + w)), mv)));
            }
            const uint64_t* bs[4] = {b0, b1, b2, b3};
            const __m256i accs[4] = {a0, a1, a2, a3};
            for (int kk = 0; kk < 4; ++kk) {
                uint64_t tmp[4]; _mm256_storeu_si256(reinterpret_cast<__m256i*>(tmp), accs[kk]);
                uint64_t s = tmp[0] + tmp[1] + tmp[2] + tmp[3];
                for (size_t ww = w; ww < W; ++ww) s += static_cast<uint64_t>(std::popcount(bs[kk][ww] & mask[ww]));
                dst[(b + static_cast<size_t>(kk)) * stride + col] = static_cast<WorldCountRep>(s);
            }
        }
        for (; b < B; ++b)
            dst[b * stride + col] = static_cast<WorldCountRep>(pc_and_avx2(belptr[b], mask, W));
    }
}

void batched_belief_features_k(Kern k, const chocofarm::Environment& env,
                               const std::vector<chocofarm::BitsetBelief>& beliefs,
                               std::vector<chocofarm::BeliefFeatures>& out) {
    using chocofarm::TreasureRep;
    using chocofarm::GeometryIdRep;
    using chocofarm::WorldCountRep;
    const TreasureRep N = static_cast<TreasureRep>(env.N());
    const GeometryIdRep nD = static_cast<GeometryIdRep>(env.n_detectors());
    const size_t Nn = static_cast<size_t>(N);
    const size_t nDn = static_cast<size_t>(nD);
    const size_t B = beliefs.size();
    const size_t W = static_cast<size_t>(env.kW64());
    const double log_nworlds = std::log(static_cast<double>(env.worlds().size()));

    // Cache the B belief live-word pointers once (avoid the per-(mask,belief) .live() indirection).
    std::vector<const uint64_t*> belptr(B);
    std::vector<WorldCountRep> nb(B);
    for (size_t b = 0; b < B; ++b) { belptr[b] = beliefs[b].live().data(); nb[b] = beliefs[b].count_.value(); }

    // B-major count scratch (column = treasure/detector, row = belief). Sized once.
    std::vector<WorldCountRep> cnt_marg(B * Nn, 0), cnt_det(B * nDn, 0);

    // Phase 1, MASK-MAJOR: each mask row read once, counted against all B beliefs (per the chosen kernel).
    for (size_t t = 0; t < Nn; ++t)
        stage_column(k, env.treasure_mask(static_cast<int>(t)).data(), W, belptr, B, t, Nn, cnt_marg);
    for (size_t j = 0; j < nDn; ++j)
        stage_column(k, env.detector_mask(static_cast<int>(j)).data(), W, belptr, B, j, nDn, cnt_det);

    phase2(cnt_marg, cnt_det, nb, Nn, nDn, B, log_nworlds, out);
}


// Byte-equal compare (the oracle's equal_features: == on doubles produced by identical int->double->*inv
// ops IS the exact bit comparison; no NaN/-0.0 arise from counts>=0, inv>0). Names the diverging field.
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

// A spread of REALISTIC beliefs (the belief-sweep oracle/bench style): the full belief + filtered beliefs
// of varied sizes. A real producer parks beliefs that have been narrowed by some sense/collect steps, so
// we build a range of densities by taking rank-strided subsets of [0, nworlds). The FULL belief (all ranks)
// is included; the rest are strided to span small->large nb with varied per-mask cover counts. Returns the
// rank-subsets; the caller turns each into a BitsetBelief. We cycle through this pool to fill a batch of B.
[[nodiscard]] std::vector<std::vector<uint32_t>> belief_pool(size_t nworlds, std::mt19937_64& rng) {
    std::vector<std::vector<uint32_t>> pool;
    auto full = [&]() { std::vector<uint32_t> v(nworlds); std::iota(v.begin(), v.end(), 0u); return v; };
    // strided subset of a target size: pick `target` distinct ranks spread across [0,nworlds) with a
    // pseudo-random phase so different pool entries have different cover mixes (not nested prefixes).
    auto strided = [&](size_t target) {
        std::vector<uint32_t> v;
        if (target == 0 || nworlds == 0) return v;
        const size_t step = std::max<size_t>(1, nworlds / target);
        const size_t phase = rng() % step;
        for (size_t i = phase; i < nworlds && v.size() < target; i += step)
            v.push_back(static_cast<uint32_t>(i));
        return v;
    };
    pool.push_back(full());                          // the full prior (nb = nworlds)
    for (size_t tgt : {nworlds / 2, nworlds / 4, nworlds / 8, nworlds / 16,
                       size_t{1000}, size_t{256}, size_t{64}, size_t{8}})
        pool.push_back(strided(std::min(tgt, nworlds)));
    return pool;
}

// One batch of B beliefs drawn from the pool (cycled), shuffled so the batch is a realistic density mix.
[[nodiscard]] std::vector<chocofarm::BitsetBelief> make_batch(
        const chocofarm::Environment& env, const std::vector<std::vector<uint32_t>>& pool,
        size_t B, std::mt19937_64& rng) {
    std::vector<chocofarm::BitsetBelief> batch;
    batch.reserve(B);
    for (size_t i = 0; i < B; ++i) batch.push_back(to_bitset(env, pool[(rng() + i) % pool.size()]));
    return batch;
}

// median + IQR of a sample (copies + sorts).
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

// Bootstrap 95% CI of the median of the PAIRED per-rep ratio (batched_ns / separate_ns). A ratio < 1 is a
// speedup; the CI excluding 1.0 is the significance bar. Paired: each rep times BOTH arms back-to-back, so
// the ratio cancels shared drift. Returns {median_ratio, ci_lo, ci_hi}.
struct Boot { double med, lo, hi; };
[[nodiscard]] Boot bootstrap_ratio_ci(const std::vector<double>& ratios, std::mt19937_64& rng) {
    const size_t n = ratios.size();
    std::vector<double> meds;
    meds.reserve(4000);
    std::uniform_int_distribution<size_t> pick(0, n - 1);
    for (int it = 0; it < 4000; ++it) {
        std::vector<double> samp(n);
        for (size_t i = 0; i < n; ++i) samp[i] = ratios[pick(rng)];
        std::sort(samp.begin(), samp.end());
        meds.push_back(samp[n / 2]);
    }
    std::sort(meds.begin(), meds.end());
    std::vector<double> r = ratios;
    std::sort(r.begin(), r.end());
    return {r[n / 2], meds[static_cast<size_t>(0.025 * meds.size())],
            meds[static_cast<size_t>(0.975 * meds.size())]};
}
}  // namespace

int main(int argc, char** argv) {
    std::vector<std::string_view> args(argv, argv + argc);
    std::optional<std::string_view> instance = opt(args, "--instance");
    std::optional<std::string_view> faces = opt(args, "--faces");
    if (!instance || !faces) {
        std::cerr << "usage: batched-featurization-proto --instance <p> --faces <p> "
                     "[--budget-s 0.25] [--reps 12]\n";
        return 2;
    }
    const double budget = opt(args, "--budget-s")
        ? std::atof(std::string(*opt(args, "--budget-s")).c_str()) : 0.25;
    const int reps = opt(args, "--reps") ? std::atoi(std::string(*opt(args, "--reps")).c_str()) : 12;

    auto inst = chocofarm::load_instance(*instance, *faces);
    if (!inst) { std::cerr << "batched-featurization-proto: FATAL: " << inst.error().message << "\n"; return 1; }
    chocofarm::Environment env(*inst);
    const int N = env.N();
    const int nD = env.n_detectors();
    const size_t nworlds = env.worlds().size();
    const size_t mask_bytes =
        static_cast<size_t>(N + nD) * static_cast<size_t>(env.kW64()) * sizeof(uint64_t);

    std::cout << "batched-featurization-proto: N=" << N << " nD=" << nD << " |worlds|=" << nworlds
              << " kW64=" << env.kW64() << " mask_matrix=" << (mask_bytes / 1024.0) << " KiB"
              << " use_bitset=" << (env.use_bitset() ? "true" : "false")
              << "  (budget=" << budget << "s/point reps=" << reps << ")\n";
    if (!env.use_bitset()) {
        std::cout << "RESULT: SKIP (env gates OFF the bitset arm — the production hot path under test is "
                     "the bitset masked-AND+popcount kernel; nothing to prototype)\n";
        return 0;
    }

    std::mt19937_64 rng(0xC0FFEEull);
    const std::vector<std::vector<uint32_t>> pool = belief_pool(nworlds, rng);
    const std::vector<size_t> Bs = {8, 16, 32, 64};

    // ---------------- BIT-IDENTITY GATE (ALL THREE batched kernels x all B x a spread of batches) ----
    // For each kernel, build several batches and assert: batched output[b] == belief_features(env,
    // batch[b]) byte-for-byte for every b. belief_features (the BITSET arm) is the reference (the oracle
    // nets IT against the naive count; here we net every batched kernel against belief_features). Counts
    // are order-independent integers, so the AVX2 vpshufb / tiled kernels must equal the scalar arm exactly.
    {
        const std::pair<Kern, const char*> kerns[] = {
            {Kern::Scalar, "scalar-transpose"}, {Kern::Avx2, "avx2"}, {Kern::Avx2Tiled, "avx2-tiled"}};
        size_t checked = 0;
        for (auto [k, name] : kerns) {
            for (size_t B : Bs) {
                for (int batch_i = 0; batch_i < 8; ++batch_i) {
                    std::vector<chocofarm::BitsetBelief> batch = make_batch(env, pool, B, rng);
                    std::vector<chocofarm::BeliefFeatures> got;
                    batched_belief_features_k(k, env, batch, got);
                    for (size_t b = 0; b < B; ++b) {
                        const chocofarm::BeliefFeatures ref =
                            chocofarm::belief_features(env, chocofarm::Belief{batch[b]});
                        std::string why;
                        if (!equal_features(got[b], ref, why)) {
                            std::cout << "RESULT: FAIL bit-identity (kernel=" << name << " B=" << B
                                      << " batch=" << batch_i << " b=" << b << " field=" << why
                                      << ") — batched kernel != per-leaf belief_features; do NOT ship\n";
                            return 1;
                        }
                        ++checked;
                    }
                }
            }
        }
        std::cout << "RESULT: PASS bit-identity (" << checked << " (batched row == per-leaf "
                     "belief_features) byte-for-byte comparisons across 3 kernels x B in {8,16,32,64} x 8 "
                     "batches)\n";
    }

    // ---------------- A/B TIMING (interleaved, paired, median/IQR + bootstrap CI) ----------------
    // The baseline is the PRODUCTION per-leaf scalar arm (B independent belief_features). Each candidate is
    // ratioed against it, paired per rep (the ratio cancels shared drift). The five candidates separate:
    //   sep-avx2     : per-leaf, AVX2 primitive, NO batching   — isolates the PRIMITIVE-swap win (no seam).
    //   bat-scalar   : mask-major loop transpose, scalar       — the original #3 layout (loop order only).
    //   bat-avx2     : mask-major, AVX2 primitive              — primitive swap, in the batched shape.
    //   bat-avx2-tile: mask-major, AVX2, mask-resident 4-tile  — the BATCH-SPECIFIC locality increment.
    // The decomposition that matters: (bat-avx2-tile vs sep-avx2) is the increment that needs the SEAM;
    // (sep-avx2 vs separate) is the primitive swap that helps the per-leaf path WITHOUT a seam.
    enum class Arm { SepAvx2, BatScalar, BatAvx2, BatAvx2Tile };
    struct Cand { Arm arm; const char* name; };
    const Cand cands[] = {
        {Arm::SepAvx2, "sep-avx2"}, {Arm::BatScalar, "bat-scalar"},
        {Arm::BatAvx2, "bat-avx2"}, {Arm::BatAvx2Tile, "bat-avx2-tile"}};

    std::cout << "\nA/B vs the production per-leaf SCALAR arm (ratio = candidate/separate; speedup>0 => "
                 "faster; FASTER iff 95% CI wholly below 1.0):\n";
    std::cout << std::setw(16) << "candidate" << std::setw(5) << "B"
              << std::setw(11) << "sep us" << std::setw(11) << "cand us"
              << std::setw(9) << "ratio" << std::setw(18) << "95% CI(ratio)"
              << std::setw(10) << "speedup%" << std::setw(9) << "verdict" << "\n";

    for (const Cand& c : cands) {
        for (size_t B : Bs) {
            std::vector<chocofarm::BitsetBelief> batch = make_batch(env, pool, B, rng);
            std::vector<chocofarm::BeliefFeatures> out_b, out_s(B);

            auto time_separate = [&]() -> double {  // production per-leaf scalar
                using clk = std::chrono::steady_clock;
                long it = 0; double sink = 0.0; const auto t0 = clk::now(); double el = 0.0;
                do {
                    for (size_t b = 0; b < B; ++b) {
                        out_s[b] = chocofarm::belief_features(env, chocofarm::Belief{batch[b]});
                        sink += out_s[b].marg_sum + out_s[b].sharpness;
                    }
                    ++it; el = std::chrono::duration<double>(clk::now() - t0).count();
                } while (el < budget);
                g_sink += sink; return el * 1e6 / static_cast<double>(it);
            };
            auto time_cand = [&]() -> double {
                using clk = std::chrono::steady_clock;
                long it = 0; double sink = 0.0; const auto t0 = clk::now(); double el = 0.0;
                do {
                    switch (c.arm) {
                        case Arm::SepAvx2: {
                            // per-leaf AVX2: one belief, all masks, AVX2 primitive — a batch-of-1 over each
                            // belief separately (NO mask-resident cross-belief tiling), the no-seam variant.
                            for (size_t b = 0; b < B; ++b) {
                                std::vector<chocofarm::BitsetBelief> one{batch[b]};
                                batched_belief_features_k(Kern::Avx2, env, one, out_b);
                                sink += out_b[0].marg_sum;
                            }
                            break;
                        }
                        case Arm::BatScalar:   batched_belief_features_k(Kern::Scalar, env, batch, out_b); break;
                        case Arm::BatAvx2:     batched_belief_features_k(Kern::Avx2, env, batch, out_b); break;
                        case Arm::BatAvx2Tile: batched_belief_features_k(Kern::Avx2Tiled, env, batch, out_b); break;
                    }
                    if (c.arm != Arm::SepAvx2) for (size_t b = 0; b < B; ++b) sink += out_b[b].marg_sum;
                    ++it; el = std::chrono::duration<double>(clk::now() - t0).count();
                } while (el < budget);
                g_sink += sink; return el * 1e6 / static_cast<double>(it);
            };

            (void)time_separate(); (void)time_cand();  // warmup discarded
            std::vector<double> sep_us, cand_us, ratios;
            for (int r = 0; r < reps; ++r) {
                double s, t;
                if (r & 1) { t = time_cand(); s = time_separate(); }
                else       { s = time_separate(); t = time_cand(); }
                sep_us.push_back(s); cand_us.push_back(t); ratios.push_back(t / s);
            }
            const Stat ss = stat(sep_us), cs = stat(cand_us);
            std::mt19937_64 brng(0x1234ull + B + static_cast<size_t>(c.arm) * 101);
            const Boot bt = bootstrap_ratio_ci(ratios, brng);
            const double speedup = (1.0 / bt.med - 1.0) * 100.0;  // how much faster the candidate is
            const char* verdict = (bt.hi < 1.0) ? "FASTER" : (bt.lo > 1.0) ? "SLOWER" : "NULL";
            std::ostringstream ci;
            ci << "[" << std::fixed << std::setprecision(3) << bt.lo << "," << bt.hi << "]";
            std::cout << std::fixed << std::setprecision(2)
                      << std::setw(16) << c.name << std::setw(5) << B
                      << std::setw(11) << ss.median << std::setw(11) << cs.median
                      << std::setw(9) << std::setprecision(4) << bt.med
                      << std::setw(18) << ci.str()
                      << std::setw(9) << std::setprecision(1) << speedup << "%"
                      << std::setw(9) << verdict << "\n";
        }
    }
    std::cout << "\nDECOMPOSITION: (sep-avx2 vs separate) is the PRIMITIVE-swap win (helps the per-leaf path, "
                 "NO seam). (bat-avx2-tile vs sep-avx2) is the BATCH-SPECIFIC increment (needs the seam). "
                 "ratio<1 => candidate faster; speedup% = (1/ratio - 1)*100.\n";
    return 0;
}
