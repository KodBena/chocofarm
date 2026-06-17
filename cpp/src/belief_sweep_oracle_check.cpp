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
//   STEP 2 (the bitset arm, docs/design/cpp-belief-rep-scoping.md §5 step 6) EXTENDS this into the
//   FLAT-vs-BITSET A/B harness: for each sampled belief it builds BOTH a FlatBelief and a BitsetBelief
//   DIRECTLY (bypassing the gate) and asserts they are BYTE-IDENTICAL across every env seam op — marginals,
//   informative (per detector), legal_actions, belief_features, nb, belief_key, world_at_rank(r) for all r,
//   and sample_world over a fixed RNG stream (the same draws ⇒ the same world). The flat arm is the
//   REFERENCE; any divergence is a bitset bug (ADR-0002). This is the strongest P6 tier (exact integer
//   counts) and pins flat↔bitset bit-exactness — the basis on which the end-to-end search (gate ON) matches
//   Python exactly as the flat arm did.
//
//   Protocol:  belief-sweep-oracle-check --instance <p> --faces <p>
//   A separate executable (ADR-0012 P3, one-owner): this tool owns the belief-sweep bit-exactness fixture
//   AND the flat-vs-bitset A/B. No redis, no net — pure compute. Public Domain (The Unlicense).
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <map>
#include <optional>
#include <random>
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

// world value -> RANK (its position in env.worlds(), i.e. combinations order — NOT numeric order: the
// world-set is built by combinations(range(N), K) which is NOT numerically ascending; see the §1B note
// correction in the Step-2 report). Built once from env.worlds(); used to set the rank bit for each world.
[[nodiscard]] std::map<uint32_t, size_t> rank_of(const chocofarm::Environment& env) {
    std::map<uint32_t, size_t> m;
    const std::vector<uint32_t>& worlds = env.worlds();
    for (size_t r = 0; r < worlds.size(); ++r) m.emplace(worlds[r], r);
    return m;
}

// Build a BitsetBelief over env.worlds()' RANK space from a flat world-set (a SUBSET of env.worlds(), in
// any order). Each world's rank (its position in env.worlds(), via the rank map — combinations order, NOT
// numeric) sets its bit. This bypasses the env's gate (it constructs the bitset directly regardless of
// use_bitset_), so the A/B can run flat-vs-bitset for the SAME belief. count_ = the set-bit count.
[[nodiscard]] chocofarm::BitsetBelief to_bitset(const chocofarm::Environment& env,
                                                const std::map<uint32_t, size_t>& rank,
                                                const std::vector<uint32_t>& flat) {
    chocofarm::BitsetBelief b;  // bits{} zero-initialized inline (the fixed-capacity array); tail stays 0
    b.kw64_ = env.kW64();       // the runtime word count (the env gates ON so kW64 <= kBitsetMaxWords)
    for (uint32_t w : flat) {
        const auto it = rank.find(w);
        // every belief here is a subset of env.worlds(), so the world is always found (invariant).
        const size_t r = it->second;
        b.bits[r >> 6] |= (uint64_t{1} << (r & 63u));  // only live words [0,kw64_) are ever written here
    }
    b.count_ = static_cast<int>(flat.size());
    return b;
}

// Compare a FLAT and a BITSET belief (the SAME world-set in two reps) across EVERY env seam read op and
// assert byte-identity. `tag` prefixes the op name (so a filter-sequence step can be located). Returns true
// on agreement; on a mismatch names the diverging op in `why`.
[[nodiscard]] bool ops_identical(const chocofarm::Environment& env, const chocofarm::Belief& fb,
                                 const chocofarm::Belief& bb, const std::string& tag, std::string& why) {
    const int nb = env.nb(fb);
    auto note = [&](const std::string& f) {
        why = tag + f + " (nb=" + std::to_string(nb) + ")"; return false;
    };
    if (env.nb(fb) != env.nb(bb)) return note("nb");
    if (env.empty(fb) != env.empty(bb)) return note("empty");
    if (env.belief_key(fb) != env.belief_key(bb)) return note("belief_key");  // the fingerprint triple, §6 risk 5
    if (env.marginals(fb) != env.marginals(bb)) return note("marginals");
    for (int j = 0; j < env.n_detectors(); ++j)
        if (env.informative(j, fb) != env.informative(j, bb)) return note("informative[" + std::to_string(j) + "]");
    for (const chocofarm::CollectedSet& coll :
         {chocofarm::CollectedSet{}, chocofarm::CollectedSet{}.with(0),
          chocofarm::CollectedSet{}.with(0).with(3).with(7)}) {
        if (env.legal_actions(fb, coll) != env.legal_actions(bb, coll)) return note("legal_actions");
    }
    {
        const chocofarm::BeliefFeatures ff = chocofarm::belief_features(env, fb);
        const chocofarm::BeliefFeatures bf2 = chocofarm::belief_features(env, bb);
        std::string fwhy;
        if (!equal_features(ff, bf2, static_cast<size_t>(nb), fwhy)) return note("belief_features." + fwhy);
    }
    for (int r = 0; r < nb; ++r)
        if (env.world_at_rank(fb, r) != env.world_at_rank(bb, r)) return note("world_at_rank[" + std::to_string(r) + "]");
    if (nb > 0) {
        // sample_world over a FIXED RNG stream: the SAME draws ⇒ the SAME world (the byte-identity basis).
        std::mt19937_64 rf(0xA5A5A5A5ull), rb(0xA5A5A5A5ull);
        for (int draw = 0; draw < 256; ++draw)
            if (env.sample_world(fb, rf) != env.sample_world(bb, rb))
                return note("sample_world[draw=" + std::to_string(draw) + "]");
    }
    return true;
}

// The flat-vs-bitset A/B for ONE belief: build BOTH arms over the same world-set, assert every read op is
// byte-identical (ops_identical), THEN drive a SEQUENCE of in-place filters through BOTH arms (the mutation
// seam, §6 risk 8 — the one place count_ is written) consistent with a fixed true world, re-asserting every
// op after each filter. This exercises filter_treasure/filter_detector/the count_ recompute, which the
// static (directly-built) beliefs do not. Returns true on agreement; names the diverging op in `why`.
[[nodiscard]] bool ab_identical(const chocofarm::Environment& env,
                                const std::map<uint32_t, size_t>& rank,
                                const std::vector<uint32_t>& flat, std::string& why) {
    chocofarm::Belief fb = chocofarm::FlatBelief{flat};
    chocofarm::Belief bb = to_bitset(env, rank, flat);

    // (a) the static belief: every read op byte-identical across the two reps.
    if (!ops_identical(env, fb, bb, "", why)) return false;
    if (flat.empty()) return true;  // no filter sequence to run on the empty belief

    // (b) a filter SEQUENCE consistent with a fixed true world (the first live world), applied to BOTH arms
    // in lockstep; re-assert every op after each step. Detector filters by the world's true reading; a
    // treasure filter by the world's presence bit. The two arms must track byte-identically through the
    // narrowing (the in-place mutation + count_ recompute).
    const uint32_t wstar = flat.front();  // the true world for this scripted trajectory
    for (int j = 0; j < env.n_detectors(); ++j) {
        const bool pos = env.observe(j, wstar);
        env.filter_detector(fb, j, pos);
        env.filter_detector(bb, j, pos);
        if (!ops_identical(env, fb, bb, "after filter_detector[" + std::to_string(j) + "] ", why)) return false;
    }
    for (int t = 0; t < env.N(); ++t) {
        const bool present = ((wstar >> t) & 1u) != 0;
        env.filter_treasure(fb, t, present);
        env.filter_treasure(bb, t, present);
        if (!ops_identical(env, fb, bb, "after filter_treasure[" + std::to_string(t) + "] ", why)) return false;
    }
    return true;
}

#ifdef CHOCO_BELIEF_ZDD
// ---- the OPT-IN flat-vs-ZDD FEATURE A/B (the §B.4(b) net) ----
// The ASYMMETRY vs the flat-vs-bitset A/B: the ZDD arm is BIT-EXACT on counts/marginals/det-counts/
// features + members set-equal, but the SAMPLING trio (sample_world / world_at_rank / belief_key) RE-
// BASELINES (the ZDD's canonical member order != worlds()-rank order). So zdd_ops_identical asserts the
// FEATURE ops byte-identical + members(Z) SET-EQUAL the flat belief, and DOES NOT assert sampling equal.

// Build a ZddBelief DIRECTLY from a flat world-set (bypassing the gate, so the A/B can run flat-vs-ZDD
// for the same belief regardless of use_zdd_). The diagram is the family of exactly `flat`'s worlds.
[[nodiscard]] chocofarm::ZddBelief to_zdd(const chocofarm::Environment& env,
                                          const std::vector<uint32_t>& flat) {
    chocofarm::ZddBelief b;
    b.z = chocofarm::beliefzdd::BeliefDiagram(std::span<const uint32_t>(flat), env.N());
    b.cached_count_ = b.z.count();
    return b;
}

// Assert every FEATURE op is byte-identical flat-vs-ZDD, AND members(Z) is set-equal to the flat belief.
// Sampling (sample_world / world_at_rank / belief_key) is NOT compared — it re-baselines (by design).
[[nodiscard]] bool zdd_ops_identical(const chocofarm::Environment& env, const chocofarm::Belief& fb,
                                     const chocofarm::Belief& zb, const std::string& tag, std::string& why) {
    const int nb = env.nb(fb);
    auto note = [&](const std::string& f) { why = tag + f + " (nb=" + std::to_string(nb) + ")"; return false; };
    if (env.nb(fb) != env.nb(zb)) return note("nb");
    if (env.empty(fb) != env.empty(zb)) return note("empty");
    if (env.marginals(fb) != env.marginals(zb)) return note("marginals");
    for (int j = 0; j < env.n_detectors(); ++j)
        if (env.informative(j, fb) != env.informative(j, zb))
            return note("informative[" + std::to_string(j) + "]");
    for (const chocofarm::CollectedSet& coll :
         {chocofarm::CollectedSet{}, chocofarm::CollectedSet{}.with(0),
          chocofarm::CollectedSet{}.with(0).with(3).with(7)})
        if (env.legal_actions(fb, coll) != env.legal_actions(zb, coll)) return note("legal_actions");
    {
        const chocofarm::BeliefFeatures ff = chocofarm::belief_features(env, fb);
        const chocofarm::BeliefFeatures zf = chocofarm::belief_features(env, zb);
        std::string fwhy;
        if (!equal_features(ff, zf, static_cast<size_t>(nb), fwhy)) return note("belief_features." + fwhy);
    }
    // members(Z) SET-EQUAL the flat belief (the restrict-op faithful-rep witness): sort both and compare.
    {
        std::vector<uint32_t> zm = std::get<chocofarm::ZddBelief>(zb).z.members();
        std::vector<uint32_t> fm = std::get<chocofarm::FlatBelief>(fb).worlds;
        std::sort(zm.begin(), zm.end());
        std::sort(fm.begin(), fm.end());
        if (zm != fm) return note("members(Z) != set(flat belief)");
    }
    return true;
}

// The flat-vs-ZDD A/B for ONE belief: static feature ops byte-identical + members set-equal, THEN a filter
// SEQUENCE (the restrict ops — the B.4(b) maintenance) applied to BOTH arms in lockstep, re-asserting
// every feature op + the members set-equality after each step.
[[nodiscard]] bool zdd_ab_identical(const chocofarm::Environment& env,
                                    const std::vector<uint32_t>& flat, std::string& why) {
    chocofarm::Belief fb = chocofarm::FlatBelief{flat};
    chocofarm::Belief zb = to_zdd(env, flat);
    if (!zdd_ops_identical(env, fb, zb, "", why)) return false;
    if (flat.empty()) return true;  // no filter sequence on the empty belief
    const uint32_t wstar = flat.front();  // the true world for this scripted trajectory
    for (int j = 0; j < env.n_detectors(); ++j) {
        const bool pos = env.observe(j, wstar);
        env.filter_detector(fb, j, pos);  // flat erase_if
        env.filter_detector(zb, j, pos);  // ZDD restrict_cover
        if (!zdd_ops_identical(env, fb, zb, "after filter_detector[" + std::to_string(j) + "] ", why)) return false;
    }
    for (int t = 0; t < env.N(); ++t) {
        const bool present = ((wstar >> t) & 1u) != 0;
        env.filter_treasure(fb, t, present);  // flat erase_if
        env.filter_treasure(zb, t, present);  // ZDD restrict_var
        if (!zdd_ops_identical(env, fb, zb, "after filter_treasure[" + std::to_string(t) + "] ", why)) return false;
    }
    // (c) drive the belief to EMPTY (a restrict -> BOT transition the wstar-consistent trajectory never
    // reaches): contradict a still-live treasure. After the full trajectory the belief is {wstar} (single
    // world); restrict_var on a treasure to the OPPOSITE of wstar's bit empties BOTH arms. Re-assert: both
    // empty (the restrict->BOT / erase_if->{} path — nb=0, members empty). Nets the empty-result restrict.
    if (env.nb(fb) > 0) {
        const uint32_t w = env.world_at_rank(fb, 0);          // a still-live world (flat rank 0)
        const int t0 = 0;
        const bool opposite = ((w >> t0) & 1u) == 0;          // the polarity that w does NOT satisfy
        env.filter_treasure(fb, t0, opposite);                // flat -> {} (w dropped, and it was the only one if single)
        env.filter_treasure(zb, t0, opposite);               // ZDD -> BOT (restrict_var to empty)
        if (!zdd_ops_identical(env, fb, zb, "after filter-to-empty ", why)) return false;
    }
    return true;
}

// Apply ONE filter (detector or treasure, given polarity) to a freshly-built flat+ZDD pair over `flat`,
// and assert flat==ZDD afterward. Used to drive the restrict ops to EMPTY through paths the wstar-
// consistent trajectory never reaches (cover_hold->BOT / cover_fail->BOT / with_var->BOT / without_var->
// BOT — finding from the out-of-frame review). `tag` names the case.
[[nodiscard]] bool zdd_one_filter_ok(const chocofarm::Environment& env, const std::vector<uint32_t>& flat,
                                     bool is_detector, int idx, bool polarity,
                                     const std::string& tag, std::string& why) {
    chocofarm::Belief fb = chocofarm::FlatBelief{flat};
    chocofarm::Belief zb = to_zdd(env, flat);
    if (is_detector) { env.filter_detector(fb, idx, polarity); env.filter_detector(zb, idx, polarity); }
    else             { env.filter_treasure(fb, idx, polarity); env.filter_treasure(zb, idx, polarity); }
    return zdd_ops_identical(env, fb, zb, tag, why);
}

// ---- the CONSTRUCTION-ORDER-INVARIANCE net for ZddBelief::operator== (the canonical-layout crux) ----
// The structural == (z == o.z compares n_/root_/nodes_) is EXACT only because compact()'s post-order DFS
// renumber is CANONICAL — a reduced/ordered/hash-consed ZDD's node layout is determined by the DAG, NOT by
// construction history. This net FALSIFIES that claim directly (ADR-0011 — net the guard, do not trust it):
// for a target world-set, build a ZddBelief TWO ways that reach the SAME family but via different
// construction orders, and assert structural == is TRUE. If it ever fails, the canonical-layout assumption
// is false — make it RESULT: FAIL loud (ADR-0002). Also assert two DIFFERENT families compare FALSE.
//
//   way (i)  build-from-worlds DIRECTLY (the bw ctor over `target`'s worlds).
//   way (ii) full_belief() (the family of EVERY world over N vars) then restrict_* DOWN to the same
//            world-set, by intersecting with each treasure's present/absent constraint that `target` shares
//            (a sequence of restrict_var ops — a wholly different construction path: zunion-fold from chains
//            vs restrict-applies on the full diagram, both ending in compact()). The two reach the same
//            family iff `target` is exactly the set of worlds satisfying that conjunction of bit-constraints.
//
// To make way (ii) reach a clean family we drive it on the worlds that AGREE with a fixed reference world on
// a chosen treasure bit (the subfamily "treasure t present" or "absent"): full_belief() then restrict_var(t,
// present) is exactly the set of K-subsets that (do/don't) contain t — and the same family built directly
// from those worlds is way (i). For a few t we assert way(i) == way(ii) (same family, different order) AND
// way(i) for "present" != way(i) for "absent" (disjoint families → structural !=).
[[nodiscard]] bool zdd_ctor_order_invariant(const chocofarm::Environment& env, std::string& why) {
    const std::vector<uint32_t>& all = env.worlds();
    const int N = env.N();
    // A handful of treasures spread across the universe (each partitions worlds into present/absent).
    for (int t : {0, 1, N / 2, N - 1}) {
        if (t < 0 || t >= N) continue;
        // The two target families: worlds WITH treasure t, worlds WITHOUT it (a clean bit-constraint family).
        std::vector<uint32_t> with_t, without_t;
        for (uint32_t w : all) (((w >> t) & 1u) ? with_t : without_t).push_back(w);

        for (bool present : {true, false}) {
            const std::vector<uint32_t>& target = present ? with_t : without_t;
            if (target.empty()) continue;  // (no such family on this instance — skip)

            // way (i): build the family DIRECTLY from its worlds (the bw ctor / zunion-fold).
            chocofarm::beliefzdd::BeliefDiagram zi(std::span<const uint32_t>(target), N);
            // way (ii): full_belief() then restrict DOWN to the same family (a different construction path).
            chocofarm::beliefzdd::BeliefDiagram zii(std::span<const uint32_t>(all), N);
            zii.restrict_var(t, present);

            // FAMILY pre-check (members set-equal): both ways must represent `target` exactly (else the test
            // itself is malformed — a loud failure, not a silent skip).
            {
                std::vector<uint32_t> mi = zi.members(), mii = zii.members(), tg = target;
                std::sort(mi.begin(), mi.end());
                std::sort(mii.begin(), mii.end());
                std::sort(tg.begin(), tg.end());
                if (mi != tg || mii != tg) {
                    why = "construction-order net malformed: members(way i/ii) != target family (t=" +
                          std::to_string(t) + " present=" + (present ? "1" : "0") + ")";
                    return false;
                }
            }

            // THE CRUX: same family, two construction orders → structural == MUST be TRUE (canonical layout).
            if (!(zi == zii)) {
                why = "CANONICAL-LAYOUT FALSIFIED: same family via different construction orders compared "
                      "NOT-EQUAL under structural == (t=" + std::to_string(t) + " present=" +
                      (present ? "1" : "0") + " |target|=" + std::to_string(target.size()) +
                      ") — compact()'s post-order renumber is non-canonical; the ZDD operator== fix is unsound";
                return false;
            }
        }

        // DIFFERENT families → structural == MUST be FALSE (no false positives). worlds-with-t vs
        // worlds-without-t are disjoint non-empty families (when both exist); their canonical layouts differ.
        if (!with_t.empty() && !without_t.empty()) {
            chocofarm::beliefzdd::BeliefDiagram za(std::span<const uint32_t>(with_t), N);
            chocofarm::beliefzdd::BeliefDiagram zb(std::span<const uint32_t>(without_t), N);
            if (za == zb) {
                why = "FALSE POSITIVE: two DIFFERENT families (with/without treasure " + std::to_string(t) +
                      ") compared EQUAL under structural == — the canonical layout is not injective";
                return false;
            }
        }
    }

    // ALSO net the restrict_COVER construction path (the PRODUCTION filter_detector path: env_zdd.cpp routes
    // filter_detector -> restrict_cover -> cover_hold/cover_fail — the diagrams the belief cache actually
    // compares are mutated by BOTH restrict ops, not just restrict_var). The restrict_var loop above leaves
    // the cover path un-netted; this exercises it the same two-ways: a face's COVER-defined subfamily built
    // (i) directly from its worlds vs (ii) full_belief() then restrict_cover(mask, positive). The cover ops
    // create nodes only via mk + end in compact() exactly like restrict_var, so the canonical-layout claim
    // covers them — and now the net falsifies it on the cover path too (a future cover edit that stranded a
    // non-canonical arena would make the same-family case fail loud here, ADR-0002).
    const std::span<const uint32_t> masks = env.face_masks();
    for (int j : {0, env.n_detectors() / 2, env.n_detectors() - 1}) {
        if (j < 0 || j >= env.n_detectors()) continue;
        const uint32_t mask = masks[static_cast<size_t>(j)];
        std::vector<uint32_t> hold, fail;  // cover-disjunction holds (>=1 mask bit) / fails (none)
        for (uint32_t w : all) ((w & mask) != 0 ? hold : fail).push_back(w);

        for (bool positive : {true, false}) {
            const std::vector<uint32_t>& target = positive ? hold : fail;
            if (target.empty()) continue;  // (no such cover subfamily on this instance — skip)

            // way (i): build the cover subfamily DIRECTLY from its worlds (the bw ctor / zunion-fold).
            chocofarm::beliefzdd::BeliefDiagram zi(std::span<const uint32_t>(target), N);
            // way (ii): full_belief() then restrict_cover DOWN to the same family (the production cover path).
            chocofarm::beliefzdd::BeliefDiagram zii(std::span<const uint32_t>(all), N);
            zii.restrict_cover(mask, positive);

            {
                std::vector<uint32_t> mi = zi.members(), mii = zii.members(), tg = target;
                std::sort(mi.begin(), mi.end());
                std::sort(mii.begin(), mii.end());
                std::sort(tg.begin(), tg.end());
                if (mi != tg || mii != tg) {
                    why = "construction-order net malformed (cover): members(way i/ii) != target family (face=" +
                          std::to_string(j) + " positive=" + (positive ? "1" : "0") + ")";
                    return false;
                }
            }

            // THE CRUX (cover path): same family, two construction orders -> structural == MUST be TRUE.
            if (!(zi == zii)) {
                why = "CANONICAL-LAYOUT FALSIFIED (cover): same family via build-from-worlds vs full_belief()+"
                      "restrict_cover compared NOT-EQUAL under structural == (face=" + std::to_string(j) +
                      " positive=" + (positive ? "1" : "0") + " |target|=" + std::to_string(target.size()) +
                      ") — compact()'s post-order renumber is non-canonical on the cover path; the fix is unsound";
                return false;
            }
        }

        // DIFFERENT cover families (hold vs fail are a disjoint partition) -> structural == MUST be FALSE.
        if (!hold.empty() && !fail.empty()) {
            chocofarm::beliefzdd::BeliefDiagram za(std::span<const uint32_t>(hold), N);
            chocofarm::beliefzdd::BeliefDiagram zb(std::span<const uint32_t>(fail), N);
            if (za == zb) {
                why = "FALSE POSITIVE (cover): the cover-hold and cover-fail families of face " +
                      std::to_string(j) + " compared EQUAL under structural == — layout not injective";
                return false;
            }
        }
    }
    return true;
}
#endif  // CHOCO_BELIEF_ZDD
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
    for (size_t step : {size_t{7}, size_t{13}}) {  // two strides => two cover mixes the prefixes do not produce
        std::vector<uint32_t> strided; for (size_t i = 0; i < nworlds; i += step) strided.push_back(all[i]);
        beliefs.push_back(std::move(strided));
    }

    // ---- Part 1: the belief-sweep bit-exact oracle (production §A.4 sweep == naive reference) ----
    bool ok = true;
    std::string why;
    size_t checked = 0;
    for (const std::vector<uint32_t>& bw : beliefs) {
        // STEP 2 folds the masks/dims into the env argument; the production call now takes (env, belief).
        // The naive reference() keeps the raw vector — its math is identical (the bit-exact net is unchanged).
        const chocofarm::BeliefFeatures prod = chocofarm::belief_features(env, chocofarm::FlatBelief{bw});
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

    // The gate decision, printed so a flip is DIAGNOSABLE (the §4 derived-quantity-vs-machine-constant): a
    // dim change that pushes mask_bytes past the budget silently drops the live instance to flat, which the
    // bare 'RESULT: PASS in stdout' grep would miss — the visible numbers + the A/B PASS/SKIP line below
    // (the test asserts PASS, not SKIP) are the regression surface (ADR-0011 measure-first / Rule 1).
    const std::size_t mask_bytes =
        static_cast<std::size_t>(N + nD) * static_cast<std::size_t>(env.kW64()) * sizeof(uint64_t);
    // The GATE: line now also prints the inline-buffer fit (the THIRD conjunct, env.cpp): kW64 must be
    // <= kBitsetMaxWords (the BitsetBelief's fixed inline-array capacity) or the bitset arm cannot hold the
    // belief and the env falls to flat. A dim change that pushes kW64 past the cap (or mask_bytes past the
    // budget) silently drops to flat, which this line + the A/B PASS/SKIP gate below make DIAGNOSABLE.
    std::cout << "GATE: kW64=" << env.kW64() << " mask_bytes=" << mask_bytes << " ("
              << (static_cast<double>(mask_bytes) / 1024.0) << " KiB) budget="
              << chocofarm::kTargetMaskCacheBudgetBytes << " ("
              << (static_cast<double>(chocofarm::kTargetMaskCacheBudgetBytes) / 1024.0) << " KiB) inline_cap="
              << chocofarm::kBitsetMaxWords << " words (kW64<=cap: "
              << ((env.kW64() <= chocofarm::kBitsetMaxWords) ? "yes" : "no") << ") => use_bitset="
              << (env.use_bitset() ? "true" : "false") << "\n";

    // ---- Part 2: the flat-vs-bitset A/B (every env seam op byte-identical across the two reps) ----
    // The A/B builds a BitsetBelief DIRECTLY (bypassing the gate), so it needs the env's bitset masks,
    // which the ctor builds only when the gate is ON (use_bitset_). The live instance gates ON
    // (mask_bytes ≈ 121.5 KiB <= 128 KiB); a gate-OFF env has no bitset arm to A/B, so report that and skip.
    if (!env.use_bitset()) {
        std::cout << "RESULT: SKIP flat-vs-bitset A/B (this env gates OFF the bitset arm — no masks built; "
                  << "N=" << N << " nD=" << nD << " |worlds|=" << nworlds << ")\n";
        return 0;
    }
    const std::map<uint32_t, size_t> rank = rank_of(env);  // world value -> rank (combinations order)
    size_t ab_checked = 0;
    for (const std::vector<uint32_t>& bw : beliefs) {
        std::string ab_why;
        if (!ab_identical(env, rank, bw, ab_why)) {
            (void)fail("flat-vs-bitset A/B DIVERGES at op " + ab_why + " — the bitset arm is the bug (flat "
                       "is the reference, ADR-0002)");  // (void): fail() is [[nodiscard]]; we return 1 next
            return 1;
        }
        ++ab_checked;
    }
    std::cout << "RESULT: PASS flat-vs-bitset A/B byte-identical (kW64=" << env.kW64()
              << "; " << ab_checked << " beliefs x {nb, empty, belief_key, marginals, informative(per-det), "
              << "legal_actions, belief_features, world_at_rank(all r), sample_world(256 draws)} static + a "
              << "full filter SEQUENCE (all " << env.n_detectors() << " detectors + " << env.N()
              << " treasures, re-asserted after each) — flat == bitset for every op)\n";

#ifdef CHOCO_BELIEF_ZDD
    // ---- Part 3 (OPT-IN): the flat-vs-ZDD FEATURE A/B (the §B.4(b) net) ----
    // The ZDD arm is BIT-EXACT on counts/marginals/det-counts/features + members set-equal; the SAMPLING
    // trio re-baselines, so it is NOT asserted equal here (the gumbel/ismcts-dump parity will diverge on
    // the ZDD arm — EXPECTED). The restrict ops (filter_treasure -> restrict_var, filter_detector ->
    // restrict_cover) are exercised by the filter SEQUENCE; members(Z) is set-equal to the flat belief
    // after each step (the restrict-op faithful-rep witness, the design-§4 requirement).
    std::cout << "GATE-ZDD: use_zdd=" << (env.use_zdd() ? "true" : "false")
              << " (the opt-in §B.4(b) arm; full_belief returns a ZddBelief when on)\n";
    size_t zdd_checked = 0;
    for (const std::vector<uint32_t>& bw : beliefs) {
        std::string zab_why;
        if (!zdd_ab_identical(env, bw, zab_why)) {
            (void)fail("flat-vs-ZDD FEATURE A/B DIVERGES at op " + zab_why + " — the ZDD arm is the bug "
                       "(flat is the reference, ADR-0002)");
            return 1;
        }
        ++zdd_checked;
    }

    // Targeted empty-RESULT restrict nets (the out-of-frame review's coverage finding): force EACH restrict
    // op to BOT through a path the wstar-consistent trajectory cannot reach. Build beliefs from worlds()
    // partitioned by a chosen detector's cover / a chosen treasure's bit, then filter to the empty subfamily.
    {
        const int j = 0;  // a detector whose cover is non-trivial (face 0)
        std::vector<uint32_t> hitters, missers;  // cover holds / fails
        for (uint32_t w : all) (env.observe(j, w) ? hitters : missers).push_back(w);
        const int t = 0;  // a treasure
        std::vector<uint32_t> with_t, without_t;
        for (uint32_t w : all) (((w >> t) & 1u) ? with_t : without_t).push_back(w);
        struct Case { bool is_det; int idx; bool pol; const std::vector<uint32_t>* bw; const char* tag; };
        const Case cases[] = {
            {true,  j, true,  &missers,   "cover_hold->BOT (all-miss belief, filter_detector +)"},
            {true,  j, false, &hitters,   "cover_fail->BOT (all-hit belief, filter_detector -)"},
            {false, t, true,  &without_t, "with_var->BOT (no-treasure-t belief, filter_treasure +)"},
            {false, t, false, &with_t,    "without_var->BOT (all-treasure-t belief, filter_treasure -)"},
        };
        for (const Case& c : cases) {
            if (c.bw->empty()) continue;  // (cannot construct this empty-path case on this instance — skip)
            std::string ewhy;
            if (!zdd_one_filter_ok(env, *c.bw, c.is_det, c.idx, c.pol, std::string("empty-path ") + c.tag + " ", ewhy)) {
                (void)fail("flat-vs-ZDD empty-RESULT restrict DIVERGES: " + ewhy);
                return 1;
            }
        }
    }
    std::cout << "RESULT: PASS flat-vs-ZDD FEATURE A/B byte-identical (" << zdd_checked
              << " beliefs x {nb, empty, marginals, informative(per-det), legal_actions, belief_features} "
              << "+ members(Z) set-equal the flat belief, static + a full filter SEQUENCE (all "
              << env.n_detectors() << " detectors + " << env.N() << " treasures via restrict_cover/restrict_var, "
              << "re-asserted after each) + the empty-RESULT restrict nets (cover_hold/cover_fail/with_var/"
              << "without_var each driven to BOT) — flat == ZDD on every FEATURE op; SAMPLING (sample_world/"
              << "world_at_rank/belief_key) RE-BASELINES and is NOT asserted equal — the design-§4 asymmetry)\n";

    // ---- Part 4 (OPT-IN): the construction-order-invariance net for the structural ZddBelief::operator== ----
    // The structural == (z == o.z: canonical n_/root_/nodes_ compare) replaces the O(nb) members()==members()
    // enumerate-both-sides equality. It is EXACT iff compact()'s post-order renumber is CANONICAL. This nets
    // that crux directly (ADR-0011): same family via DIFFERENT construction orders → == TRUE; different
    // families → == FALSE. A false on the same-family case falsifies the canonical-layout claim (RESULT: FAIL,
    // ADR-0002 — do not paper over).
    {
        std::string cwhy;
        if (!zdd_ctor_order_invariant(env, cwhy)) {
            (void)fail("ZDD operator== construction-order-invariance net: " + cwhy);
            return 1;
        }
    }
    std::cout << "RESULT: PASS ZDD operator== construction-order invariance (same family via build-from-worlds "
              << "vs full_belief()+restrict_var AND vs full_belief()+restrict_cover → structural == TRUE; "
              << "disjoint families (treasure present/absent + cover hold/fail) → structural == FALSE; the "
              << "canonical-layout assumption behind the O(|Z|) structural == is netted on BOTH restrict paths, "
              << "not trusted)\n";
#endif
    return 0;
}
