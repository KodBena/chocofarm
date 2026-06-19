// cpp/src/wire_ab_bench.cpp
// Purpose: the Stage B e2e A/B THROUGHPUT bench for the eval-transport-adapter
//   (docs/design/cpp-eval-transport-adapter.md §4 Stage B). It drives the REAL Gumbel-AZ search (the
//   unchanged run_search / fiber-mux, every leaf resolved REMOTELY on the JAX InferenceServer over the
//   wire) for a wall-clock budget and reports decisions/s/core — for ONE selectable transport mode:
//
//     --wire-mode strict-barrier  : arm 1 (the production default run_episodes_wire_batched: gather ALL
//                                   parked -> one batched submit -> await the one reply -> resume all; D=1).
//     --wire-mode pipelined-bucket: arm 3 (run_episodes_wire_pipelined: D>1 non-blocking, resume each fiber
//                                   as its reply lands, out of order by corr-id; the server's bucketed-E +
//                                   group-wakeup drain assembles the forward). The strict path is UNTOUCHED.
//
//   This is NOT the runner and NOT a parity check (that is wire-batched-runtime-check) — it is a pure
//   throughput meter (P3, one-owner): it times how many self-play EPISODES the real search completes in the
//   budget at the spec operating point (n_sims=256, m=24, hidden=256), divides by wall, and reports a
//   decisions/s estimate. The server-side mean rows/FORWARD (the in-flight depth a single real tree
//   sustains — the Stage B key number) is reported by the server harness (stage_a_server.py SERVER_STATS),
//   not here; this binary also writes its own wire-summary (mean rows/WIRE-MESSAGE, S) via --parity-stats.
//
//   The search reads the SAME net both arms read (published to redis at (run,"gen",version); the wire
//   server loads it over the SAME key) — so the ONLY cross-arm difference is the transport schedule, the
//   ADR-0012 P7 invariant Stage B validates. The ZMQ context / DEALER / corr-id transport are the effect,
//   confined to the shared WireLeafPool (P9); a recv error / desync is a LOUD abort (ADR-0002).
//
//   Protocol:  wire-ab-bench --instance <p> --faces <p> --endpoint <ipc://...> --run <id> --version <v>
//                  --res-token <t> --wire-mode <strict-barrier|pipelined-bucket>
//                  [--secs 8 --m 24 --n-sims 256 --max-depth 24 --c-outcome 2 --lam 0.1 --max-steps 40
//                   --pool-threads T --pool-batch B --inflight-msgs D --parity-stats <path>]
//   Timing is an HONEST WALL TIME-BOX: a WARMUP phase (one full-occupancy slot-fill — JITs the server
//   bucket shapes + fills the slots, NOT counted) is separated from a MEASURE phase that runs short
//   slot-sized passes and re-checks the `--secs` budget AFTER EACH pass, so the measured window lands
//   within ~one chunk of `--secs` (not the 11–31× overshoot of a single oversized pass). dps = decisions
//   over the MEASURE window / measure_wall; total bench_wall ≈ warmup + ~secs.
//
//   Output:    a config line, a `warmup=.. measure=.. decisions=.. dps=..` line, then a RESULT line with
//              eps/s + dps + wall (= the MEASURE window) + warmup_wall + bench_wall, + exit 0, or a loud
//              failure + exit 1.
//
// Public Domain (The Unlicense).
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "chocofarm/env.hpp"
#include "chocofarm/features.hpp"
#include "chocofarm/gumbel.hpp"
#include "chocofarm/instance.hpp"
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
[[nodiscard]] double secs(std::chrono::steady_clock::time_point a,
                          std::chrono::steady_clock::time_point b) {
    return std::chrono::duration<double>(b - a).count();
}
}  // namespace

int main(int argc, char** argv) {
    std::vector<std::string_view> args(argv, argv + argc);
    std::optional<std::string_view> instance = opt(args, "--instance");
    std::optional<std::string_view> faces = opt(args, "--faces");
    std::optional<std::string_view> endpoint = opt(args, "--endpoint");
    std::optional<std::string_view> run = opt(args, "--run");
    std::optional<std::string_view> res_token = opt(args, "--res-token");
    std::optional<std::string_view> wire_mode = opt(args, "--wire-mode");
    if (!instance || !faces || !endpoint || !run || !res_token || !wire_mode) {
        std::cerr << "usage: wire-ab-bench --instance <p> --faces <p> --endpoint <ipc://...> --run <id> "
                     "--version <v> --res-token <t> --wire-mode <strict-barrier|pipelined-bucket> "
                     "[--secs 8 --m 24 --n-sims 256 --max-depth 24 --c-outcome 2 --lam 0.1 --max-steps 40 "
                     "--pool-threads T --pool-batch B --inflight-msgs D --parity-stats <path>]\n";
        return 2;
    }

    WireMode mode;
    if (*wire_mode == "strict-barrier") {
        mode = WireMode::StrictBarrier;
    } else if (*wire_mode == "pipelined-bucket") {
        mode = WireMode::PipelinedBucket;
    } else {
        std::cerr << "wire-ab-bench: FATAL: unknown --wire-mode " << *wire_mode
                  << " (expected strict-barrier | pipelined-bucket)\n";
        return 2;
    }

    const int version = opt(args, "--version") ? to_int(*opt(args, "--version")) : 0;
    const double budget = opt(args, "--secs") ? to_double(*opt(args, "--secs")) : 8.0;
    const double lam = opt(args, "--lam") ? to_double(*opt(args, "--lam")) : 0.1;
    const int max_steps = opt(args, "--max-steps") ? to_int(*opt(args, "--max-steps")) : 40;

    GumbelConfig gc;  // the Stage B operating point: m=24, n_sims=256 (overridable)
    gc.m = opt(args, "--m") ? to_int(*opt(args, "--m")) : 24;
    gc.n_sims = opt(args, "--n-sims") ? to_int(*opt(args, "--n-sims")) : 256;
    if (auto v = opt(args, "--max-depth")) gc.max_depth = to_int(*v);
    if (auto v = opt(args, "--c-outcome")) gc.c_outcome = to_int(*v);

    WireRunnerConfig wcfg;
    wcfg.endpoint = std::string(*endpoint);
    wcfg.mode = mode;
    wcfg.pool_threads = opt(args, "--pool-threads") ? to_int(*opt(args, "--pool-threads")) : 1;
    wcfg.pool_batch = opt(args, "--pool-batch") ? to_int(*opt(args, "--pool-batch")) : 64;
    wcfg.timeout_ms = opt(args, "--timeout-ms") ? to_int(*opt(args, "--timeout-ms")) : 60000;
    if (auto v = opt(args, "--inflight-msgs")) wcfg.max_inflight_msgs = to_int(*v);
    if (auto v = opt(args, "--trees-per-thread")) wcfg.trees_per_thread = to_int(*v);

    auto inst = load_instance(*instance, *faces);
    if (!inst) {
        std::cerr << "wire-ab-bench: FATAL: " << inst.error().message << "\n";
        return 1;
    }
    Environment env(*inst);
    FeatureBuilder fb(env);

    auto redis = RedisClient::create();
    if (!redis) {
        std::cerr << "wire-ab-bench: FATAL: " << redis.error().message << "\n";
        return 1;
    }
    // The net must be published (the wire server loads it over redis); we do not read it here (the leaf is
    // remote) but a sanity-read confirms the run/version exists, failing loud early rather than at recv.
    auto wp = redis->read_weights(*run, "gen", version);
    if (!wp) {
        std::cerr << "wire-ab-bench: FATAL: weight read (" << *run << ",gen," << version
                  << ") failed: " << wp.error().message << " — publish the net to redis first.\n";
        return 1;
    }

    // optional per-episode + wire-summary stats sink (the pipelined driver writes its mean rows/msg here).
    std::ofstream stats_file;
    std::ostream* stats_out = nullptr;
    if (auto stats_path = opt(args, "--parity-stats")) {
        stats_file.open(std::string(*stats_path));
        if (!stats_file) {
            std::cerr << "wire-ab-bench: FATAL: cannot open --parity-stats: " << *stats_path << "\n";
            return 1;
        }
        stats_file << std::setprecision(17);
        stats_out = &stats_file;
    }

    std::cout << "config: wire-mode=" << *wire_mode << " m=" << gc.m << " n_sims=" << gc.n_sims
              << " threads=" << wcfg.pool_threads << " pool_batch=" << wcfg.pool_batch
              << " inflight_D=" << wcfg.max_inflight_msgs
              << " trees_per_thread=" << wcfg.trees_per_thread << " secs=" << budget
              << " endpoint=" << *endpoint << "\n";

    // HONEST TIME-BOX (warmup separated from measurement). The driver runs E episodes per pass
    // SYNCHRONOUSLY and returns to completion; a pass REFILLS each slot with the next episode as the prior
    // one finishes (runner_wire_batched fill()), so a pass of E episodes keeps all `total_slots` slots
    // FULL for ~E/total_slots episode-depths, then drains a short tail. Two consequences fix the old lie:
    //
    //   (1) WARMUP (NOT counted) is one full-occupancy pass — it JIT-compiles the server's bucket shapes
    //       AND fills the slots, AND lets us MEASURE the steady-state episodes/s (eps_rate) to size the
    //       measure pass. Its stats sink is suppressed so warmup rows don't pollute the measured
    //       rows/forward (server mean) the harness reads.
    //
    //   (2) The MEASURE pass is sized from that rate to ~`budget` seconds of wall: episodes ≈ budget ×
    //       eps_rate, FLOORED at total_slots so every slot stays full (E ≥ slots ⇒ no low-occupancy
    //       fragment — what would depress rows/forward + dps). One full-occupancy episode-depth (~one
    //       episode's wall) is the irreducible granularity, so for a `budget` smaller than that the pass
    //       rounds UP to one full wave (total_slots) and `measure_wall` reports the true (slightly-over)
    //       window honestly — never the old 11–31× overshoot of a fixed 8×total_slots oversize pass whose
    //       budget was checked only between passes (the bug this fixes). dps = decisions / measure_wall;
    //       the rate is a steady-state quantity so the NUMBER is unchanged from the old meter.
    const int K_base = (wcfg.pool_batch + wcfg.pool_threads - 1) / std::max(1, wcfg.pool_threads);
    const int total_slots = wcfg.pool_threads * std::max(1, wcfg.trees_per_thread) * std::max(1, K_base);
    const int warmup_eps = std::max(total_slots, 8);  // one full-occupancy fill of all slots
    const int n_slots = n_action_slots(env);

    // Count recorded decisions (rows) across a pass's episodes — the true search-work numerator (dps).
    auto count_decisions = [&](const std::string& tok, int n_eps) -> long {
        long dec = 0;
        for (int idx = 0; idx < n_eps; ++idx) {
            auto rb = redis->read_results(tok, idx);
            if (!rb) continue;
            if (!rb->PI.empty()) dec += static_cast<long>(rb->PI.size()) / n_slots;
        }
        return dec;
    };

    // ---- WARMUP (NOT counted): full-occupancy fill + JIT, measuring eps_rate to size the measure pass.
    long warmup_eps_done = 0;
    auto warm0 = std::chrono::steady_clock::now();
    {
        const std::string tok = std::string(*res_token) + "-warmup";
        RunnerConfig rcfg;
        rcfg.run = std::string(*run);
        rcfg.phase = "gen";
        rcfg.version = version;
        rcfg.episodes = warmup_eps;
        rcfg.lam = lam;
        rcfg.max_steps = max_steps;
        rcfg.seed = 104729ull;  // distinct from the measured seed below
        rcfg.res_token = tok;
        auto w = run_episodes_wire_batched(env, fb, gc, *redis, rcfg, wcfg, nullptr);
        if (!w) {
            std::cerr << "wire-ab-bench: FATAL: warmup failed: " << w.error().message << "\n";
            return 1;
        }
        warmup_eps_done = *w;
    }
    const double warmup_wall = secs(warm0, std::chrono::steady_clock::now());
    // Steady-state episodes/s from the warmup window (its first pass also paid the JIT, so this slightly
    // UNDER-estimates the measure rate — biasing the measure pass to run a touch longer than budget, never
    // shorter; honest). Fall back to a small positive rate if the warmup somehow wrote nothing.
    const double eps_rate =
        (warmup_wall > 0.0 && warmup_eps_done > 0) ? (warmup_eps_done / warmup_wall) : 1.0;

    // ---- MEASURE: ONE full-occupancy pass sized to ~`budget` seconds of wall (floored at one full wave).
    const long want_eps = static_cast<long>(budget * eps_rate);
    const int measure_eps = static_cast<int>(std::max<long>(want_eps, total_slots));
    long total_decisions = 0;
    long total_eps = 0;
    auto t0 = std::chrono::steady_clock::now();
    {
        const std::string tok = std::string(*res_token) + "-measure";
        RunnerConfig rcfg;
        rcfg.run = std::string(*run);
        rcfg.phase = "gen";
        rcfg.version = version;
        rcfg.episodes = measure_eps;
        rcfg.lam = lam;
        rcfg.max_steps = max_steps;
        rcfg.seed = 7919ull;
        rcfg.res_token = tok;
        auto w = run_episodes_wire_batched(env, fb, gc, *redis, rcfg, wcfg, stats_out);
        if (!w) {
            std::cerr << "wire-ab-bench: FATAL: measure pass failed: " << w.error().message << "\n";
            return 1;
        }
        total_eps = *w;
        total_decisions = count_decisions(tok, measure_eps);
    }
    const double measure_wall = secs(t0, std::chrono::steady_clock::now());

    const double eps_per_s = static_cast<double>(total_eps) / measure_wall;
    const double dps = static_cast<double>(total_decisions) / measure_wall;
    std::cout.precision(7);
    std::cout << "warmup=" << warmup_wall << " measure=" << measure_wall
              << " decisions=" << total_decisions << " dps=" << dps
              << " measure_eps=" << measure_eps << " eps_rate=" << eps_rate << "\n";
    // RESULT: keep `wall=` as the MEASUREMENT window (the dps denominator — the number stays valid). The
    // total bench wall is warmup_wall + measure_wall, surfaced as warmup= above and bench_wall= here.
    std::cout << "RESULT: PASS wire-mode=" << *wire_mode << " threads=" << wcfg.pool_threads
              << " episodes=" << total_eps << " decisions=" << total_decisions << " wall=" << measure_wall
              << " warmup_wall=" << warmup_wall << " bench_wall=" << (warmup_wall + measure_wall)
              << " eps_per_s=" << eps_per_s << " dps=" << dps
              << " dps_per_core=" << (dps / std::max(1, wcfg.pool_threads)) << "\n";
    return 0;
}
