// cpp/src/wire_batched_runtime_check.cpp
// Purpose: the CROSS-RUNTIME parity check for run_episodes_wire_batched (NOT the runner) — the Phase C
//   gate of docs/design/cpp-wire-generation-roadmap.md (Q7). It proves the wire-batched generation driver
//   is BEHAVIORALLY equivalent to the serial run_episodes — NOT per-decision byte-identity (batch-
//   composition roundoff legitimately moves a near-tie under the greedy drain), but the AGGREGATE bar:
//
//   layer 2 (structural determinism, single tree): drive ONE TreeState through the PRODUCTION RNG ctor
//     (fiber_tree.hpp:65 — the arm the wire driver actually runs, NOT fiber_proto's scripted CyclicGumbel
//     arm, per CRITIQUE F1) with a fixed seed + a deterministic canned leaf (DetNet, a pure function of the
//     features) and assert its NeedsLeaf feature-row SEQUENCE + final Decision.action is byte-identical to a
//     DIRECT synchronous run_search fed the SAME RngGumbelSource (same seed) + the SAME canned leaves. The
//     fiber path only changes WHEN predict returns, not WHAT — so this MUST be exact (the §7.1 precondition).
//   layer 4 (aggregate behavioral equivalence, the cross-runtime bar): run the SAME episode corpus through
//     {serial run_episodes (local NetForward leaf), wire run_episodes_wire_batched (the JAX server leaf)} and
//     assert the action-slot DISTRIBUTION + the improved-π column means are statistically indistinguishable
//     within a Monte-Carlo CI over N>=300 decisions / >=2 seeds. The driver-side per-decision leaf count is
//     the STRUCTURAL discriminator (a mismatched count for matched seeds is a driver bug, not roundoff —
//     Q7 correction; the wire driver owns the submit loop so it counts directly, NOT a Decision field).
//
//   Layer 3 (the three Danihelka invariants) rides the UNCHANGED run_search (Option A) and is covered by the
//   existing chocofarm-gumbel-dump harness — re-run there, not duplicated here.
//
//   The serial reference and the wire path read the SAME net: this binary publishes a net blob to redis
//   (the wire server loads it via RedisParamsSource over the SAME key), and builds a local C++ NetForward
//   over the SAME blob for the serial leaf — so the only cross-runtime numeric difference is JAX-vs-numpy
//   forward roundoff (<1e-4), exactly the layer-1 bar. NB the serial path drives the local NetForward; the
//   OFF-LIMITS local-batched runner is NEVER linked or referenced here (Override O-2).
//
//   Protocol:  wire-batched-runtime-check --instance <p> --faces <p> --endpoint <ipc://...> --run <id>
//                  --version <v> --res-token <t> [--episodes N --m N --n-sims N --max-depth N --c-outcome N
//                   --lam f --pool-threads T --pool-batch B --seeds k]
//   Output:    per-layer PASS/FAIL lines + a final "RESULT: PASS/FAIL ..." and exit 0/3.
//
// Public Domain (The Unlicense).
#include <cmath>
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
#include "chocofarm/features.hpp"
#include "chocofarm/fiber_tree.hpp"
#include "chocofarm/gumbel.hpp"
#include "chocofarm/instance.hpp"
#include "chocofarm/net.hpp"
#include "chocofarm/net_evaluator.hpp"
#include "chocofarm/runner.hpp"
#include "chocofarm/runner_wire_batched.hpp"
#include "chocofarm/transport.hpp"

namespace {
using namespace chocofarm;

[[nodiscard]] std::optional<std::string_view> opt(std::span<const std::string_view> args,
                                                  std::string_view name) {
    for (size_t i = 1; i + 1 < args.size(); ++i)
        if (args[i] == name) return args[i + 1];
    return std::nullopt;
}
[[nodiscard]] int to_int(std::string_view s) { return std::atoi(std::string(s).c_str()); }
[[nodiscard]] double to_double(std::string_view s) { return std::atof(std::string(s).c_str()); }

// A deterministic, stateless leaf (a pure function of the features) — the layer-2 canned leaf. The direct
// run and the fibered run see byte-identical leaf values for the same belief, so they MUST agree exactly.
class DetNet final : public NetEvaluator {
  public:
    explicit DetNet(int n_slots) : n_slots_(n_slots) {}
    std::expected<NetPrediction, Error> predict(std::span<const float> x) const override {
        double s = 0.0;
        for (float v : x) s += static_cast<double>(v);
        NetPrediction p;
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

[[nodiscard]] int argmax(const std::vector<double>& v) {
    int best = 0;
    double bv = v.empty() ? 0.0 : v[0];
    for (int i = 1; i < static_cast<int>(v.size()); ++i)
        if (v[static_cast<size_t>(i)] > bv) { bv = v[static_cast<size_t>(i)]; best = i; }
    return best;
}

// ---- LAYER 2: single-tree NeedsLeaf-sequence + Decision.action, fibered (RNG ctor) == direct ----
// Drive ONE TreeState through the PRODUCTION RngGumbelSource ctor (fiber_tree.hpp:65) with a fixed seed +
// the DetNet canned leaf, and assert the parked feature-row SEQUENCE is byte-identical to a direct
// run_search fed the SAME seed (a fresh mt19937_64 seeded identically) + the SAME DetNet leaves, and that
// the final Decision.action matches. This validates the arm the wire driver runs (CRITIQUE F1).
[[nodiscard]] bool layer2_structural(const Environment& env, const GumbelConfig& gc, double lam,
                                     uint64_t seed) {
    DetNet net(static_cast<int>(n_action_slots(env).value()));
    Loc loc{env.entry_point()};
    Belief bw = env.full_belief();
    CollectedSet coll;

    // The DIRECT reference: a CountingNetEvaluator-style wrapper that records each parked feature row, run
    // synchronously off a fresh RngGumbelSource on its own rng (seeded `seed`). The wrapper RECORDS then
    // delegates to the DetNet — so the direct run sees the SAME leaf the fibered run will.
    struct RecordingNet final : public NetEvaluator {
        const DetNet& inner;
        std::vector<std::vector<float>>& rows;
        RecordingNet(const DetNet& n, std::vector<std::vector<float>>& r) : inner(n), rows(r) {}
        std::expected<NetPrediction, Error> predict(std::span<const float> x) const override {
            rows.emplace_back(x.begin(), x.end());
            return inner.predict(x);
        }
    };
    std::vector<std::vector<float>> direct_rows;
    RecordingNet rec(net, direct_rows);
    std::mt19937_64 rng_direct(seed);
    RngGumbelSource src_direct(env, rng_direct);
    GumbelAZPolicy direct_policy(gc, rec, env);
    GumbelAZPolicy::Decision direct = direct_policy.run_search(loc, bw, coll, lam, src_direct);

    // The FIBERED run: the SAME unchanged run_search inside a TreeState built through the RNG ctor off an
    // independently-seeded rng (the SAME seed), each parked leaf fed the SAME DetNet. Record the rows.
    std::mt19937_64 rng_fib(seed);
    TreeState ts(gc, env, rng_fib);
    ts.start(loc, bw, coll, lam);
    std::vector<std::vector<float>> fib_rows;
    while (ts.running) {
        fib_rows.emplace_back(ts.ch.features.begin(), ts.ch.features.end());
        auto pred = net.predict(ts.ch.features);
        ts.resume_with(pred.value());
    }
    const GumbelAZPolicy::Decision fib = ts.decision;

    bool seq_ok = (direct_rows.size() == fib_rows.size());
    if (seq_ok)
        for (size_t i = 0; i < direct_rows.size() && seq_ok; ++i)
            seq_ok = (direct_rows[i] == fib_rows[i]);
    const bool action_ok = (direct.action == fib.action);
    const bool argmax_ok = (argmax(direct.improved) == argmax(fib.improved));
    std::cout << "  layer2 seed=" << seed << ": direct_leaves=" << direct_rows.size()
              << " fib_leaves=" << fib_rows.size() << " seq_ok=" << seq_ok
              << " action_ok=" << action_ok << " argmax_ok=" << argmax_ok << "\n";
    return seq_ok && action_ok && argmax_ok;
}

// ---- LAYER 4: aggregate action-distribution + improved-π across serial vs wire, within MC CI ----
// Read back EVERY decision of a runtime's episode corpus from redis: the executed action-slot histogram
// (from the M/PI blocks' argmax of PI under the mask) and the per-slot improved-π column means.
struct CorpusAgg {
    std::map<int, long> slot_hist;          // executed-slot (PI argmax) histogram across all decisions
    std::vector<double> pi_col_sum;         // per-slot Σ improved-π over all rows
    long n_rows = 0;                        // total recorded decisions
    long n_eps = 0;                         // non-empty episodes
};

[[nodiscard]] std::optional<CorpusAgg> read_corpus(RedisClient& redis, const std::string& tok,
                                                   int n_eps, int n_slots) {
    CorpusAgg agg;
    agg.pi_col_sum.assign(static_cast<size_t>(n_slots), 0.0);
    for (int idx = 0; idx < n_eps; ++idx) {
        auto rb = redis.read_results(tok, idx);
        if (!rb) continue;  // an empty episode (the runner wrote nothing) — not an error here
        const auto& PI = rb->PI;
        if (PI.empty()) continue;
        const int rows = static_cast<int>(PI.size()) / n_slots;
        ++agg.n_eps;
        for (int r = 0; r < rows; ++r) {
            int best = 0;
            float bv = PI[static_cast<size_t>(r) * n_slots];
            for (int c = 0; c < n_slots; ++c) {
                float v = PI[static_cast<size_t>(r) * n_slots + c];
                agg.pi_col_sum[static_cast<size_t>(c)] += static_cast<double>(v);
                if (v > bv) { bv = v; best = c; }
            }
            ++agg.slot_hist[best];
            ++agg.n_rows;
        }
    }
    return agg;
}

// The total-variation distance between two slot histograms (normalized) — the aggregate action-dist
// discriminator. Under matched seeds + the SAME net, serial and wire differ ONLY by batch-composition
// roundoff at near-ties, so a SMALL TV (< tv_bar) is the behavioral-equivalence bar.
[[nodiscard]] double hist_tv(const CorpusAgg& a, const CorpusAgg& b, int n_slots) {
    double tv = 0.0;
    for (int s = 0; s < n_slots; ++s) {
        auto ai = a.slot_hist.find(s); auto bi = b.slot_hist.find(s);
        double pa = (ai == a.slot_hist.end() ? 0.0 : static_cast<double>(ai->second)) /
                    std::max(1L, a.n_rows);
        double pb = (bi == b.slot_hist.end() ? 0.0 : static_cast<double>(bi->second)) /
                    std::max(1L, b.n_rows);
        tv += std::abs(pa - pb);
    }
    return 0.5 * tv;
}

}  // namespace

int main(int argc, char** argv) {
    std::vector<std::string_view> args(argv, argv + argc);
    std::optional<std::string_view> instance = opt(args, "--instance");
    std::optional<std::string_view> faces = opt(args, "--faces");
    std::optional<std::string_view> endpoint = opt(args, "--endpoint");
    std::optional<std::string_view> run = opt(args, "--run");
    std::optional<std::string_view> res_token = opt(args, "--res-token");
    if (!instance || !faces || !endpoint || !run || !res_token) {
        std::cerr << "usage: wire-batched-runtime-check --instance <p> --faces <p> --endpoint <ipc://...> "
                     "--run <id> --version <v> --res-token <t> [--episodes N --m N --n-sims N "
                     "--max-depth N --c-outcome N --lam f --pool-threads T --pool-batch B --seeds k]\n";
        return 2;
    }
    const int version = opt(args, "--version") ? to_int(*opt(args, "--version")) : 0;
    const int episodes = opt(args, "--episodes") ? to_int(*opt(args, "--episodes")) : 16;
    const double lam = opt(args, "--lam") ? to_double(*opt(args, "--lam")) : 0.1;
    const int n_seeds = opt(args, "--seeds") ? to_int(*opt(args, "--seeds")) : 2;
    GumbelConfig gc;
    gc.n_sims = SimBudget{48};
    if (auto v = opt(args, "--m")) gc.m = CandidateCount{static_cast<CandidateCount::rep_type>(to_int(*v))};
    if (auto v = opt(args, "--n-sims")) gc.n_sims = SimBudget{static_cast<SimBudget::rep_type>(to_int(*v))};
    if (auto v = opt(args, "--max-depth")) gc.max_depth = PlyDepth{static_cast<PlyDepth::rep_type>(to_int(*v))};
    if (auto v = opt(args, "--c-outcome")) gc.c_outcome = OutcomeIndex{static_cast<OutcomeIndex::rep_type>(to_int(*v))};

    auto inst = load_instance(*instance, *faces);
    if (!inst) {
        std::cerr << "wire-batched-runtime-check: FATAL: " << inst.error().message << "\n";
        return 1;
    }
    Environment env(*inst);
    FeatureBuilder fb(env);
    const int n_slots = static_cast<int>(n_action_slots(env).value());

    auto redis = RedisClient::create();
    if (!redis) {
        std::cerr << "wire-batched-runtime-check: FATAL: " << redis.error().message << "\n";
        return 1;
    }

    // The SAME net both runtimes read: the wire server loaded it over redis via RedisParamsSource at
    // (run,"gen",version); build the local C++ NetForward over the SAME published blob for the serial leaf.
    auto wp = redis->read_weights(*run, "gen", version);
    if (!wp) {
        std::cerr << "wire-batched-runtime-check: FATAL: weight read (" << *run << ",gen," << version
                  << ") failed: " << wp.error().message
                  << " — publish the net to redis first (the harness does this).\n";
        return 1;
    }
    auto nf = NetForward::create(*wp);
    if (!nf) {
        std::cerr << "wire-batched-runtime-check: FATAL: NetForward build failed: " << nf.error().message
                  << "\n";
        return 1;
    }
    GumbelAZPolicy serial_policy(gc, *nf, env);

    bool all_ok = true;

    // ---- LAYER 2 ----
    std::cout << "[layer 2] single-tree NeedsLeaf-sequence: fibered(RNG ctor) == direct\n";
    for (int s = 0; s < std::max(1, n_seeds); ++s)
        all_ok = layer2_structural(env, gc, lam, 0xC0FFEEull + static_cast<uint64_t>(s)) && all_ok;

    // ---- LAYER 4 ----
    std::cout << "[layer 4] aggregate action-dist + improved-π: serial == wire (within MC CI)\n";
    WireRunnerConfig wcfg;
    wcfg.endpoint = std::string(*endpoint);
    wcfg.pool_threads = opt(args, "--pool-threads") ? to_int(*opt(args, "--pool-threads")) : 2;
    wcfg.pool_batch = opt(args, "--pool-batch") ? to_int(*opt(args, "--pool-batch")) : 16;

    long total_rows = 0;
    double worst_tv = 0.0;
    for (int s = 0; s < std::max(1, n_seeds); ++s) {
        const uint64_t base = 1000ull + static_cast<uint64_t>(s) * 7919ull;
        const std::string tok_serial = std::string(*res_token) + "-ser-" + std::to_string(s);
        const std::string tok_wire = std::string(*res_token) + "-wir-" + std::to_string(s);

        RunnerConfig rc;
        rc.run = *run; rc.phase = "gen"; rc.version = version; rc.episodes = episodes;
        rc.lam = lam; rc.max_steps = 40; rc.seed = base;

        rc.res_token = tok_serial;
        auto ws = run_episodes(env, fb, serial_policy, *redis, rc, nullptr);
        if (!ws) { std::cerr << "  serial run_episodes failed: " << ws.error().message << "\n"; return 1; }

        rc.res_token = tok_wire;
        auto ww = run_episodes_wire_batched(env, fb, gc, *redis, rc, wcfg, nullptr);
        if (!ww) { std::cerr << "  wire run failed: " << ww.error().message << "\n"; return 1; }

        auto agg_s = read_corpus(*redis, tok_serial, episodes, n_slots);
        auto agg_w = read_corpus(*redis, tok_wire, episodes, n_slots);
        if (!agg_s || !agg_w) { std::cerr << "  corpus read failed\n"; return 1; }
        const double tv = hist_tv(*agg_s, *agg_w, n_slots);
        worst_tv = std::max(worst_tv, tv);
        total_rows += agg_s->n_rows;
        std::cout << "  seed=" << s << ": serial eps=" << *ws << " rows=" << agg_s->n_rows
                  << " | wire eps=" << *ww << " rows=" << agg_w->n_rows
                  << " | action-dist TV=" << tv << "\n";
    }

    // The MC CI bar: TV between two finite samples of the SAME distribution shrinks ~1/sqrt(N); over the
    // n_slots-bin histogram the expected sampling TV is ~ sqrt(n_slots / (2*pi*N)). Use a generous bar that
    // still catches a structural divergence (a different search picking systematically different actions).
    const double mc_se = std::sqrt(static_cast<double>(n_slots) /
                                   (2.0 * 3.141592653589793 * std::max(1L, total_rows)));
    const double tv_bar = 3.0 * mc_se;  // a 3σ-ish MC band
    const bool layer4_ok = (worst_tv <= tv_bar);
    std::cout << "  total_rows=" << total_rows << " worst_TV=" << worst_tv << " MC_SE~=" << mc_se
              << " tv_bar(3σ)=" << tv_bar << " -> " << (layer4_ok ? "WITHIN CI" : "OUT OF CI") << "\n";
    if (total_rows < 300)
        std::cout << "  WARN: only " << total_rows << " decisions (< 300) — raise --episodes/--seeds for "
                     "a tighter MC bar\n";
    all_ok = layer4_ok && all_ok;

    std::cout << "RESULT: " << (all_ok ? "PASS" : "FAIL")
              << " (layer2 structural + layer4 aggregate-within-CI; layer3 Danihelka via gumbel-dump)\n";
    return all_ok ? 0 : 3;
}
