// cpp/src/belief_zdd_probe.cpp
// Purpose: the standalone STAGED MEASURE-FIRST probe for the §B.4(a) belief decision-diagram on-ramp
//   (belief_features_and_decision_diagram_note.md Part B; docs/design/cpp-belief-zdd-onramp.md). It
//   mirrors belief_sweep_oracle_check.cpp's structure (opt()/fail()/reference()/equal_features,
//   --instance/--faces, RESULT: PASS|FAIL) and links chocofarm_core for Environment / belief_features
//   / BeliefFeatures / load_instance, exactly as the oracle does. The hand-rolled ZDD engine
//   (BeliefDiagram, beliefzdd::) lives in THE PROBE'S OWN TU (cpp/probe/belief_zdd.{hpp,cpp}), NOT in
//   chocofarm_core — a WIP engine must not be able to red the runner build (§11 scope discipline).
//
//   STAGE 1 — THE DECISION GATE. For each belief: assert bw is duplicate-free (§13 trap 8), build Z,
//   run the FAITHFUL-REPRESENTATION triple (§5.4: set(members(Z))==set(bw), |members|==|bw|,
//   count(Z)==|bw|) so |Z| is trustworthy, then record (nb, |Z|, |Z|/nb). The headline beliefs are
//   REALISTIC search information sets — worlds() narrowed by random CONSISTENT observation sequences
//   (informative-only steps, outcomes from a sampled true world w*; §6) — NOT random subsets. A
//   RANDOM-SUBSET CONTROL arm of the same nb is carried alongside (§7) to prove the win is structure,
//   not small nb. The t=0 full-set symmetric outlier is reported separately, EXCLUDED from the
//   headline aggregate (§7). The median |Z|/nb per depth IS the (a)->(b) decision number — the probe
//   produces it, it does NOT decide (§B.4).
//
//   STAGE 2 — THE BIT-EXACT NET (§10). For each belief (the realistic grid + the control arm + the
//   explicit edge cases): the diagram's integer bit_cnt (all_marginals) / det_cnt (non-constructing
//   disjoint-count) EQUAL chocofarm::belief_features's integer counts (the §B.3 logic invariant, vs
//   an independent naive reference()); the Σ_t bit_cnt[t] == 5·nb canary + count()==members().size();
//   the popcount-1 mask shortcut (det_cnt[j]==bit_cnt[b]) and the multi-bit brute-disjoint cross-check;
//   then the identical Phase-2 *inv makes the WHOLE BeliefFeatures byte-identical (equal_features
//   verbatim). On any mismatch: RESULT: FAIL naming the belief + the field/index. On success: PASS.
//
//   A separate executable (ADR-0012 P3, one-owner): this tool owns the ZDD on-ramp gate. No redis, no
//   net — pure compute. Public Domain (The Unlicense).
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <optional>
#include <random>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "belief_zdd.hpp"

#include "chocofarm/env.hpp"
#include "chocofarm/feature_compute.hpp"
#include "chocofarm/features.hpp"
#include "chocofarm/instance.hpp"

namespace {

using chocofarm::beliefzdd::BeliefDiagram;

// --- CLI ACL (mirrors the oracle's typed opt) ---
[[nodiscard]] std::optional<std::string_view> opt(std::span<const std::string_view> args,
                                                  std::string_view name) {
    for (size_t i = 1; i + 1 < args.size(); ++i)
        if (args[i] == name) return args[i + 1];
    return std::nullopt;
}
// Print the FAIL verdict and return the process exit code (1), so a caller writes `return fail(msg);`.
[[nodiscard]] int fail(const std::string& msg) {
    std::cout << "RESULT: FAIL " << msg << "\n";
    return 1;
}

// --- The INDEPENDENT naive reference (the oracle's reference() shape): env.observe (the
// array-of-structs path the diagram replaces), the SAME `* inv` spec, deliberately NOT branchless /
// fused so it shares no code path with the production sweep beyond the math definition. §10.2. ---
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
        for (int j = 0; j < nD; ++j) if (env.observe(j, w)) dc[j] += 1;  // <- the independent path
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

// Byte-equal every BeliefFeatures field (== on doubles: identical float ops on identical integer
// counts -> == is the exact bit comparison; no NaN/-0.0 from counts>=0, inv>0). Verbatim from the
// oracle's equal_features. §10.2 step 3.
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

// --- The §10.1 Phase-2 helper: BeliefFeatures from the diagram's integer counts via the IDENTICAL
// Phase-2 *inv of features.cpp (one home for the byte-identity claim). int64_t counts, informative
// via the EXACT features.cpp form det_cnt>0 && det_cnt<(int64_t)nb (§13 trap 9). ---
[[nodiscard]] chocofarm::BeliefFeatures belief_features_from_diagram(
    const BeliefDiagram& z, std::span<const uint32_t> masks, int N, int nD, double log_nworlds) {
    const int64_t nb = z.count();
    chocofarm::BeliefFeatures bf;
    bf.marg.assign(N, 0.0);
    bf.p_pos.assign(nD, 0.0);
    bf.informative.assign(nD, 0.0);
    if (nb == 0) return bf;  // == belief_features_empty
    const std::vector<int64_t> bit_cnt = z.all_marginals();
    const std::vector<int64_t> det_cnt = z.all_detector_counts(masks);
    const double inv = 1.0 / static_cast<double>(nb);
    for (int t = 0; t < N; ++t) { bf.marg[t] = static_cast<double>(bit_cnt[t]) * inv; bf.marg_sum += bf.marg[t]; }
    for (int j = 0; j < nD; ++j) {
        bf.p_pos[j] = static_cast<double>(det_cnt[j]) * inv;
        bf.informative[j] = (det_cnt[j] > 0 && det_cnt[j] < static_cast<int64_t>(nb)) ? 1.0 : 0.0;
    }
    bf.sharpness = std::log(static_cast<double>(nb)) / log_nworlds;
    bf.nonempty = 1.0;
    return bf;
}

// --- A naive disjoint-count over the explicit bw (the §10.3 multi-bit cross-check): #{w in bw :
// (w & mask) == 0}. Independent of the diagram. ---
[[nodiscard]] int64_t brute_disjoint(std::span<const uint32_t> bw, uint32_t mask) {
    int64_t c = 0;
    for (uint32_t w : bw) if ((w & mask) == 0) ++c;
    return c;
}

// --- bw duplicate-free precondition (§13 trap 8, fail-loud ADR-0002). Returns true iff every world
// in `sorted` (already sorted) is distinct. ---
[[nodiscard]] bool is_duplicate_free(const std::vector<uint32_t>& sorted) {
    for (size_t i = 1; i < sorted.size(); ++i) if (sorted[i] == sorted[i - 1]) return false;
    return true;
}

// --- The REALISTIC-belief generator (§6): worlds() narrowed by a random CONSISTENT observation
// sequence. Outcomes are read from a sampled true world w* (so the belief is always non-empty and is
// a genuine information set the search could occupy). Candidate actions are INFORMATIVE-ONLY
// (detectors with env.informative; treasures with 0 < bit-count < nb) — mirroring env.legal_actions,
// so every step is a real search move and the depth axis is monotone. Returns the sorted, narrowed bw
// AND the anchor w* so the caller can assert the §6 consistency invariant (w* in bw, fail-loud). ---
[[nodiscard]] std::pair<std::vector<uint32_t>, uint32_t> generate_realistic_belief(
    const chocofarm::Environment& env, int depth, std::mt19937_64& rng) {
    const int N = env.N();
    const int nD = env.n_detectors();
    const std::vector<uint32_t>& all = env.worlds();
    std::uniform_int_distribution<size_t> pick_world(0, all.size() - 1);
    const uint32_t wstar = all[pick_world(rng)];  // the consistency anchor
    std::vector<uint32_t> bw = all;               // the search's t=0 state (copy)

    for (int step = 1; step <= depth; ++step) {
        // assemble informative-only candidates: ("d", j) then ("t", i), mirroring legal_actions order.
        std::vector<std::pair<bool, int>> cands;  // (is_detector, index)
        for (int j = 0; j < nD; ++j) if (env.informative(j, bw)) cands.emplace_back(true, j);
        const int64_t nb = static_cast<int64_t>(bw.size());
        for (int i = 0; i < N; ++i) {
            int64_t ci = 0;
            for (uint32_t w : bw) ci += (w >> i) & 1u;
            if (ci > 0 && ci < nb) cands.emplace_back(false, i);
        }
        if (cands.empty()) break;  // belief fully determined; stop (record the actual depth reached)
        std::uniform_int_distribution<size_t> pick(0, cands.size() - 1);
        auto [is_det, idx] = cands[pick(rng)];
        if (is_det) env.filter_detector(bw, idx, env.observe(idx, wstar));
        else        env.filter_treasure(bw, idx, ((wstar >> idx) & 1u) != 0);
    }
    std::sort(bw.begin(), bw.end());  // canonical order for set-compares
    return {std::move(bw), wstar};
}

// --- A RANDOM-SUBSET control belief (§7): a distinct random subset of worlds() of the given size,
// sorted. The control arm — expected |Z| ~ nb (no shared substructure), proving the realistic win is
// STRUCTURE, not small nb. ---
[[nodiscard]] std::vector<uint32_t> random_subset(const chocofarm::Environment& env, size_t nb,
                                                 std::mt19937_64& rng) {
    const std::vector<uint32_t>& all = env.worlds();
    nb = std::min(nb, all.size());
    std::vector<uint32_t> idx(all.size());
    for (size_t i = 0; i < idx.size(); ++i) idx[i] = static_cast<uint32_t>(i);
    // partial Fisher-Yates: draw nb distinct indices.
    for (size_t i = 0; i < nb; ++i) {
        std::uniform_int_distribution<size_t> d(i, idx.size() - 1);
        std::swap(idx[i], idx[d(rng)]);
    }
    std::vector<uint32_t> out;
    out.reserve(nb);
    for (size_t i = 0; i < nb; ++i) out.push_back(all[idx[i]]);
    std::sort(out.begin(), out.end());
    return out;
}

// --- median of an int vector (sorted copy) ---
[[nodiscard]] double median_i(std::vector<int64_t> v) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    size_t m = v.size() / 2;
    return (v.size() & 1) ? static_cast<double>(v[m])
                          : 0.5 * (static_cast<double>(v[m - 1]) + static_cast<double>(v[m]));
}
[[nodiscard]] double median_d(std::vector<double> v) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    size_t m = v.size() / 2;
    return (v.size() & 1) ? v[m] : 0.5 * (v[m - 1] + v[m]);
}

// --- The full STAGE-2 bit-exact net for ONE belief (§10). Returns true on PASS; on FAIL sets `why`
// and prints nothing (the caller prints the RESULT: FAIL). `label` names the belief for diagnostics.
// ---
[[nodiscard]] bool stage2_check(const chocofarm::Environment& env, const std::vector<uint32_t>& bw,
                                std::span<const uint32_t> masks, int N, int nD, double log_nworlds,
                                const std::string& label, std::string& why) {
    // The family is the SET of distinct worlds: take a sorted copy for the duplicate-free precondition
    // and the faithful-rep SET comparison, so the check is correct for any input order (the realistic /
    // control generators already sort; the full-set edge case passes worlds() in combinations order).
    std::vector<uint32_t> sbw(bw.begin(), bw.end());
    std::sort(sbw.begin(), sbw.end());

    // bw duplicate-free precondition (asserted on every belief; fail-loud). §13 trap 8.
    if (!is_duplicate_free(sbw)) { why = label + ": bw is NOT duplicate-free"; return false; }

    BeliefDiagram z(bw, N);
    const int64_t nb = z.count();

    // faithful-rep (§5.4: set(members(Z))==set(bw), |members|==|bw|, count(Z)==|bw|) — kept active in
    // Stage 2 (§10.2 step 4).
    {
        std::vector<uint32_t> m = z.members();
        std::sort(m.begin(), m.end());
        if (static_cast<size_t>(nb) != sbw.size()) {
            why = label + ": count(Z)=" + std::to_string(nb) + " != |bw|=" + std::to_string(sbw.size());
            return false;
        }
        if (m.size() != sbw.size()) {
            why = label + ": |members(Z)|=" + std::to_string(m.size()) + " != |bw|=" + std::to_string(sbw.size());
            return false;
        }
        if (m != sbw) { why = label + ": set(members(Z)) != set(bw)"; return false; }
    }

    // integer bit_cnt / det_cnt vs the independent naive reference (§10.2 step 1). NEVER the
    // llround(marg*nb) round-trip — the naive integer reference is the oracle.
    std::vector<int64_t> bc(N, 0), dc(nD, 0);
    for (uint32_t w : bw) {
        for (int t = 0; t < N; ++t) if ((w >> t) & 1u) bc[t] += 1;
        for (int j = 0; j < nD; ++j) if (env.observe(j, w)) dc[j] += 1;
    }
    const std::vector<int64_t> bit_cnt = z.all_marginals();
    const std::vector<int64_t> det_cnt = z.all_detector_counts(masks);
    for (int t = 0; t < N; ++t)
        if (bit_cnt[t] != bc[t]) {
            why = label + ": bit_cnt[" + std::to_string(t) + "]=" + std::to_string(bit_cnt[t]) +
                  " != naive " + std::to_string(bc[t]);
            return false;
        }
    for (int j = 0; j < nD; ++j)
        if (det_cnt[j] != dc[j]) {
            why = label + ": det_cnt[" + std::to_string(j) + "]=" + std::to_string(det_cnt[j]) +
                  " != naive " + std::to_string(dc[j]);
            return false;
        }

    // the §9 canary + count==members (§10.2 step 2). NB: K is the constant present-bit count per
    // world (every world is a K-subset), so Σ_t bit_cnt[t] == K·nb.
    int64_t sum_bits = 0;
    for (int t = 0; t < N; ++t) sum_bits += bit_cnt[t];
    if (sum_bits != static_cast<int64_t>(env.K()) * nb) {
        why = label + ": Σ bit_cnt=" + std::to_string(sum_bits) + " != K·nb=" +
              std::to_string(static_cast<int64_t>(env.K()) * nb);
        return false;
    }

    // the §B.2 through-line: a popcount-1 mask 1<<b gives det_cnt == bit_cnt[b] (Part A's shortcut as
    // the |mask|=1 disjoint-count special case); a multi-bit mask matches a brute disjoint over bw.
    // (§10.3 edge rows, asserted as extra nets on every belief that has such masks present.)
    for (int j = 0; j < nD; ++j) {
        uint32_t mask = masks[j];
        if (mask == 0) continue;
        if ((mask & (mask - 1)) == 0) {  // popcount 1
            int b = 0;
            while (((mask >> b) & 1u) == 0) ++b;
            if (det_cnt[j] != bit_cnt[b]) {
                why = label + ": popcount-1 det_cnt[" + std::to_string(j) + "] != bit_cnt[" +
                      std::to_string(b) + "]";
                return false;
            }
        } else {  // multi-bit
            int64_t want = nb - brute_disjoint(bw, mask);
            if (det_cnt[j] != want) {
                why = label + ": multi-bit det_cnt[" + std::to_string(j) + "]=" +
                      std::to_string(det_cnt[j]) + " != nb-brute_disjoint " + std::to_string(want);
                return false;
            }
        }
    }

    // feature byte-identity: belief_features_from_diagram == chocofarm::belief_features (the production
    // sweep) AND == the naive reference. The only float op is *inv over exact integers, so == is exact
    // (§10.2 step 3, same justification as the oracle).
    const chocofarm::BeliefFeatures from_diag =
        belief_features_from_diagram(z, masks, N, nD, log_nworlds);
    const chocofarm::BeliefFeatures prod =
        chocofarm::belief_features(bw, masks, N, nD, log_nworlds);
    const chocofarm::BeliefFeatures ref = reference(env, bw, N, nD, log_nworlds);
    std::string sub;
    if (!equal_features(from_diag, prod, bw.size(), sub)) {
        why = label + ": diagram features != production belief_features at " + sub;
        return false;
    }
    if (!equal_features(from_diag, ref, bw.size(), sub)) {
        why = label + ": diagram features != naive reference at " + sub;
        return false;
    }
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    std::vector<std::string_view> args(argv, argv + argc);
    std::optional<std::string_view> instance = opt(args, "--instance");
    std::optional<std::string_view> faces = opt(args, "--faces");
    if (!instance || !faces) {
        std::cerr << "usage: belief-zdd-probe --instance <p> --faces <p>\n";
        return 2;
    }
    auto inst = chocofarm::load_instance(*instance, *faces);
    if (!inst) { std::cerr << "belief-zdd-probe: FATAL: " << inst.error().message << "\n"; return 1; }
    chocofarm::Environment env(*inst);
    const int N = env.N();
    const int nD = env.n_detectors();
    const std::span<const uint32_t> masks = env.face_masks();
    const std::vector<uint32_t>& all = env.worlds();
    const size_t nworlds = all.size();
    const double log_nworlds = std::log(static_cast<double>(nworlds));

    std::string why;

    // ========================== STAGE 2 edge cases (the §10.3 net) ==========================
    // Run the explicit edge cases first (they pin BOT / TOP / the chain / the full set / both mask
    // edge cases) — a failure here is the cheapest, clearest diagnostic.
    {
        std::vector<std::vector<uint32_t>> edges;
        edges.emplace_back();                                   // empty (nb=0): exercises BOT
        edges.push_back({all[0]});                              // single (nb=1): chain + TOP
        edges.push_back(all);                                   // full set (nb=15504): symmetric stress
        const char* names[] = {"edge:empty", "edge:single", "edge:full"};
        for (size_t e = 0; e < edges.size(); ++e) {
            if (!stage2_check(env, edges[e], masks, N, nD, log_nworlds, names[e], why))
                return fail(why);
        }
        // extra empty/single structural pins not covered by stage2_check's count path.
        {
            BeliefDiagram z0(edges[0], N);
            if (z0.count() != 0 || z0.node_count() != 0 || !z0.members().empty())
                return fail("edge:empty: count/node_count/members not all zero");
            for (int64_t d : z0.all_detector_counts(masks))
                if (d != 0) return fail("edge:empty: a det_cnt != 0");
        }
        {
            BeliefDiagram z1(edges[1], N);
            if (z1.node_count() != env.K())  // a K-of-N world is a K-node chain (zero-suppression)
                return fail("edge:single: node_count=" + std::to_string(z1.node_count()) +
                            " != K=" + std::to_string(env.K()));
        }
        // full-set marginal symmetry: bit_cnt[t] == C(N-1, K-1) for every t.
        {
            BeliefDiagram zf(all, N);
            // C(N-1,K-1) via the env count: #{worlds with bit t} = nworlds*K/N (== C(19,4)=3876).
            const int64_t want = static_cast<int64_t>(nworlds) * env.K() / N;
            for (int64_t bcv : zf.all_marginals())
                if (bcv != want)
                    return fail("edge:full: bit_cnt != C(N-1,K-1)=" + std::to_string(want));
        }
    }

    // ========================== STAGE 1 + STAGE 2 over the realistic grid ==========================
    // Depths D ∈ {1,2,3,5,8,12,20}; S samples per depth, seeded deterministically (reproducible). Per
    // sample: generate a realistic belief, run the full stage2_check (which embeds the faithful-rep
    // gate), and record (depth, nb, |Z|). A random-subset CONTROL of the same nb is built and checked
    // alongside. The t=0 full unfiltered set is a SEPARATE, clearly-labelled row, EXCLUDED from the
    // headline aggregate (§7).
    const std::vector<int> depths = {1, 2, 3, 5, 8, 12, 20};
    const int S = 64;
    std::mt19937_64 rng(0x9E3779B97F4A7C15ull);  // deterministic seed (ADR-0009 reproducible)

    struct Row {
        int depth;
        std::vector<int64_t> nb_real, z_real, nb_ctrl, z_ctrl;
        std::vector<double> ratio_real, ratio_ctrl;
        int below_half_real = 0, below_half_ctrl = 0;  // #{|Z| < nb/2}
    };
    std::vector<Row> rows;

    for (int depth : depths) {
        Row r;
        r.depth = depth;
        for (int s = 0; s < S; ++s) {
            auto [bw, wstar] = generate_realistic_belief(env, depth, rng);
            const std::string lbl = "real(D=" + std::to_string(depth) + ",s=" + std::to_string(s) + ")";
            // §6 consistency invariant (fail-loud, ADR-0002): the anchor w* must survive every filter,
            // so the belief is a genuine NON-EMPTY information set (nb >= 1). bw is sorted, so binary
            // search suffices.
            if (!std::binary_search(bw.begin(), bw.end(), wstar))
                return fail(lbl + ": w* not in bw (the §6 consistency invariant failed)");
            if (!stage2_check(env, bw, masks, N, nD, log_nworlds, lbl, why)) return fail(why);
            BeliefDiagram z(bw, N);
            const int64_t nb = z.count();
            const int64_t zc = z.node_count();
            r.nb_real.push_back(nb);
            r.z_real.push_back(zc);
            r.ratio_real.push_back(nb > 0 ? static_cast<double>(zc) / static_cast<double>(nb) : 0.0);
            if (nb > 0 && 2 * zc < nb) ++r.below_half_real;

            // control: a random subset of the SAME size nb.
            std::vector<uint32_t> cw = random_subset(env, static_cast<size_t>(nb), rng);
            const std::string clbl =
                "ctrl(D=" + std::to_string(depth) + ",s=" + std::to_string(s) + ")";
            if (!stage2_check(env, cw, masks, N, nD, log_nworlds, clbl, why)) return fail(why);
            BeliefDiagram cz(cw, N);
            const int64_t cnb = cz.count();
            const int64_t czc = cz.node_count();
            r.nb_ctrl.push_back(cnb);
            r.z_ctrl.push_back(czc);
            r.ratio_ctrl.push_back(cnb > 0 ? static_cast<double>(czc) / static_cast<double>(cnb) : 0.0);
            if (cnb > 0 && 2 * czc < cnb) ++r.below_half_ctrl;
        }
        rows.push_back(std::move(r));
    }

    // ---- the table (stdout, parseable). Per depth: nb min/median/max, |Z| min/median/max, median
    // |Z|/nb (realistic vs control), and the fraction with |Z| < nb/2. The median |Z|/nb at realistic
    // depths IS the (a)->(b) decision number (§7). ----
    std::cout << "\n=== |Z| vs nb (the §B.4 (a)->(b) decision measurement) ===\n";
    std::cout << "env: N=" << N << " K=" << env.K() << " nD=" << nD << " |worlds|=" << nworlds
              << "; S=" << S << " samples/depth; REALISTIC = worlds() narrowed by random CONSISTENT"
              << " observation sequences (informative-only); CONTROL = random subset of same nb.\n\n";
    std::cout << std::left
              << std::setw(5) << "D"
              << std::setw(22) << "nb (min/med/max)"
              << std::setw(22) << "|Z| (min/med/max)"
              << std::setw(13) << "med Z/nb"
              << std::setw(13) << "ctrl Z/nb"
              << std::setw(11) << "Z<nb/2"
              << std::setw(11) << "ctrl<nb/2"
              << "\n";
    std::cout << std::string(97, '-') << "\n";
    auto minv = [](std::vector<int64_t> v) { return v.empty() ? int64_t{0} : *std::min_element(v.begin(), v.end()); };
    auto maxv = [](std::vector<int64_t> v) { return v.empty() ? int64_t{0} : *std::max_element(v.begin(), v.end()); };
    for (const Row& r : rows) {
        std::string nbcol = std::to_string(minv(r.nb_real)) + "/" +
                            std::to_string(static_cast<int64_t>(median_i(r.nb_real))) + "/" +
                            std::to_string(maxv(r.nb_real));
        std::string zcol = std::to_string(minv(r.z_real)) + "/" +
                           std::to_string(static_cast<int64_t>(median_i(r.z_real))) + "/" +
                           std::to_string(maxv(r.z_real));
        std::cout << std::left << std::setw(5) << r.depth
                  << std::setw(22) << nbcol
                  << std::setw(22) << zcol
                  << std::setw(13) << std::fixed << std::setprecision(4) << median_d(r.ratio_real)
                  << std::setw(13) << std::fixed << std::setprecision(4) << median_d(r.ratio_ctrl)
                  << std::setw(11) << (std::to_string(r.below_half_real) + "/" + std::to_string(S))
                  << std::setw(11) << (std::to_string(r.below_half_ctrl) + "/" + std::to_string(S))
                  << "\n";
    }

    // the t=0 full unfiltered set — the SYMMETRIC OUTLIER, reported separately, NOT in the headline.
    {
        BeliefDiagram zf(all, N);
        const int64_t nb = zf.count();
        const int64_t zc = zf.node_count();
        std::cout << std::string(97, '-') << "\n";
        std::cout << "t=0 unfiltered (NOT a decision point — symmetric outlier, excluded from headline): "
                  << "nb=" << nb << " |Z|=" << zc << " Z/nb=" << std::fixed << std::setprecision(6)
                  << (static_cast<double>(zc) / static_cast<double>(nb)) << "\n";
    }

    std::cout << "\nNOTE: the probe PRODUCES the number; the human reads it (§B.4). |Z| << nb (realistic"
              << " << control) => the belief decision-diagram pays (push to B.4(b)); |Z| ~ nb"
              << " (realistic ~ control) => it does not (the SIMD sweep wins; shelve for features).\n";

    // ========================== the verdict ==========================
    std::cout << "\nRESULT: PASS belief-zdd on-ramp (N=" << N << " nD=" << nD << " |worlds|=" << nworlds
              << "; faithful-rep + Stage-2 bit-exact over " << (depths.size() * S * 2)
              << " realistic+control beliefs + 3 edge cases; diagram counts == belief_features"
              << " byte-for-byte, Σ bit_cnt==K·nb canary, *inv convention)\n";
    return 0;
}
