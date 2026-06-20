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
//                   --pool-threads T --pool-batch B --inflight-msgs D --trees-per-thread N
//                   --min-coalesce S_min --parity-stats <path>]
//
//   --min-coalesce S_min (PipelinedBucket only) is the producer's minimum coalescing degree — the closed
//   convoy fix (cpp-eval-wire-formal-diagnosis.md §3): the driver never issues a sub-S_min message while
//   replies are outstanding, so ready slots pool into a fuller batch instead of collapsing to B=1. Default
//   32; sweep it to confirm it raises rows/forward without capping the B≈192 fast region.
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
#include <sstream>
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
// Read a numeric field from the runner's wire_summary line (the fixed-decision measure reports the measured
// window's decision count + nanosecond wall there — abandoned episodes never reach redis). Returns the LAST
// occurrence's value, or 0 if absent.
[[nodiscard]] long long read_stat_ll(const std::string& path, const std::string& key) {
    std::ifstream f(path);
    std::string line;
    long long val = 0;
    while (std::getline(f, line)) {
        auto p = line.find(key);
        if (p != std::string::npos) val = std::atoll(line.c_str() + static_cast<long>(p + key.size()));
    }
    return val;
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
                     "--pool-threads T --pool-batch B --inflight-msgs D --trees-per-thread N "
                     "--min-coalesce S_min --gen-chunk-floor <0|1> --measure-decisions M "
                     "--settle-decisions S --pool-plies P --pool-seed K --parity-stats <path>]\n";
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
    // FIXED-DECISION measure budget (PipelinedBucket only; keeps the live search). 0 = the legacy
    // episode-wave measure. When > 0, the measure stops at this many recorded decisions (dps =
    // decisions/wall over a fixed amount of real search work); --warmup-decisions is an optional small
    // uncounted pre-pass (a standalone cold server; 0 when the harness pre-warms the server's buckets).
    const long measure_decisions =
        opt(args, "--measure-decisions") ? std::atol(std::string(*opt(args, "--measure-decisions")).c_str()) : 0;
    // The reproducible WARM POOL (fixes the cold-start lockstep + all-openers bias of a naive budget): each
    // slot is advanced 0..--pool-plies random legal actions (seeded by --pool-seed) so the population spans a
    // wide range of search dynamics, regenerated bit-identically per config (comparable numbers). --settle-
    // decisions run UNCOUNTED first (pipeline desync) before the measured window. pool-plies=0 ⇒ fresh openers.
    const int pool_plies = opt(args, "--pool-plies") ? to_int(*opt(args, "--pool-plies")) : 0;
    const std::uint64_t pool_seed = opt(args, "--pool-seed")
        ? static_cast<std::uint64_t>(std::atoll(std::string(*opt(args, "--pool-seed")).c_str())) : 20260620ull;
    const long settle_decisions =
        opt(args, "--settle-decisions") ? std::atol(std::string(*opt(args, "--settle-decisions")).c_str()) : 0;

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
    if (auto v = opt(args, "--min-coalesce")) wcfg.min_coalesce = to_int(*v);
    // The runnable gen-side batch-floor (the "final bolt"): --gen-chunk-floor <0|1>, default 0 (drain-all).
    if (auto v = opt(args, "--gen-chunk-floor")) wcfg.chunk_floor = (to_int(*v) != 0);

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

    // ---- IN-PROCESS HPO SWEEP (the faithful warm pool). --sweep-configs "cf:S:D,cf:S:D,..." builds ONE
    // warm pool ONCE — a real run staggered over --warmup-decisions and SNAPSHOTTED (snapshot_out) — then
    // REPLAYS it per producer config (chunk_floor:S_min:D), measuring --measure-decisions after
    // --settle-decisions. So the expensive warmup (reaching a faithful staggered, representative-belief
    // population) is amortized ONCE across every config, and every config measures the SAME reproducible
    // population (comparable numbers). The benchmark search runs with no_early_exit (default ON here) so
    // the population spans the real belief distribution without early-exit truncation (gumbel.cpp). The
    // server θ is fixed for the whole run (the harness sets it); the bench sweeps the PRODUCER knobs. Each
    // config prints one SWEEP_RESULT line.
    if (auto sweep = opt(args, "--sweep-configs")) {
        gc.no_early_exit = !opt(args, "--no-early-exit") ||
                           std::atoi(std::string(*opt(args, "--no-early-exit")).c_str()) != 0;   // default ON
        const int sweep_kbase = (wcfg.pool_batch + wcfg.pool_threads - 1) / std::max(1, wcfg.pool_threads);
        const int sweep_slots =
            wcfg.pool_threads * std::max(1, wcfg.trees_per_thread) * std::max(1, sweep_kbase);
        const long warm_dec = opt(args, "--warmup-decisions")
            ? std::atol(std::string(*opt(args, "--warmup-decisions")).c_str())
            : std::max<long>(2000, static_cast<long>(sweep_slots) * 4);
        const long meas = measure_decisions > 0 ? measure_decisions : 1000;

        std::vector<SlotSnapshot> pool;
        {
            RunnerConfig rc;
            rc.run = std::string(*run); rc.phase = "gen"; rc.version = version;
            rc.episodes = 1'000'000'000; rc.lam = lam; rc.max_steps = max_steps;
            rc.seed = pool_seed; rc.res_token = std::string(*res_token) + "-pool";
            WireRunnerConfig wcp = wcfg; wcp.decision_budget = warm_dec;
            auto w = run_episodes_wire_pipelined(env, fb, gc, *redis, rc, wcp, nullptr, nullptr, 0, &pool);
            if (!w) { std::cerr << "wire-ab-bench: FATAL: pool warmup failed: " << w.error().message << "\n"; return 1; }
        }
        std::cout << "POOL built slots=" << pool.size() << " warmup_decisions=" << warm_dec
                  << " no_early_exit=" << (gc.no_early_exit ? 1 : 0)
                  << " measure_decisions=" << meas << " settle_decisions=" << settle_decisions << "\n" << std::flush;

        std::string s(*sweep);
        size_t start = 0;
        while (start < s.size()) {
            const size_t comma = s.find(',', start);
            std::string tok = s.substr(start, comma == std::string::npos ? std::string::npos : comma - start);
            start = comma == std::string::npos ? s.size() : comma + 1;
            if (tok.empty()) continue;
            int cf = 0, S = 32, D = 8;   // chunk_floor : S_min : D
            const size_t c1 = tok.find(':');
            if (c1 != std::string::npos) {
                cf = std::atoi(tok.substr(0, c1).c_str());
                const size_t c2 = tok.find(':', c1 + 1);
                if (c2 != std::string::npos) {
                    S = std::atoi(tok.substr(c1 + 1, c2 - c1 - 1).c_str());
                    D = std::atoi(tok.substr(c2 + 1).c_str());
                }
            }
            WireRunnerConfig wcm = wcfg;
            wcm.chunk_floor = (cf != 0); wcm.min_coalesce = S; wcm.max_inflight_msgs = D;
            wcm.decision_budget = meas;
            RunnerConfig rc;
            rc.run = std::string(*run); rc.phase = "gen"; rc.version = version;
            rc.episodes = 1'000'000'000; rc.lam = lam; rc.max_steps = max_steps;
            rc.seed = 7919ull; rc.res_token = std::string(*res_token) + "-m";
            std::ostringstream oss;
            auto w = run_episodes_wire_pipelined(env, fb, gc, *redis, rc, wcm, &oss, &pool,
                                                 settle_decisions, nullptr);
            if (!w) { std::cerr << "wire-ab-bench: FATAL: sweep config " << tok << " failed: "
                                << w.error().message << "\n"; return 1; }
            const std::string sm = oss.str();
            auto getll = [&](const std::string& key) -> long long {
                const auto p = sm.rfind(key);
                return p == std::string::npos ? 0 : std::atoll(sm.c_str() + static_cast<long>(p + key.size()));
            };
            const long long md = getll("\"measure_decisions\":");
            const long long mns = getll("\"measure_wall_ns\":");
            const double dps = mns > 0 ? static_cast<double>(md) / (static_cast<double>(mns) * 1e-9) : 0.0;
            double rpm = 0.0;
            const auto pr = sm.rfind("\"mean_rows_per_msg\":");
            if (pr != std::string::npos) rpm = std::atof(sm.c_str() + static_cast<long>(pr + 20));
            std::cout << "SWEEP_RESULT cf=" << cf << " S=" << S << " D=" << D
                      << " measure_decisions=" << md << " measure_wall_s=" << (static_cast<double>(mns) * 1e-9)
                      << " dps=" << dps << " wire_rows_per_msg=" << rpm << "\n" << std::flush;
        }
        return 0;
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
              << " trees_per_thread=" << wcfg.trees_per_thread
              << " min_coalesce_Smin=" << wcfg.min_coalesce
              << " gen_chunk_floor=" << (wcfg.chunk_floor ? 1 : 0)
              << " measure_decisions=" << measure_decisions << " settle_decisions=" << settle_decisions
              << " pool_plies=" << pool_plies << " secs=" << budget
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

    long total_decisions = 0, total_eps = 0, measure_eps = 0;
    double warmup_wall = 0.0, measure_wall = 0.0, eps_rate = 0.0;

    if (measure_decisions > 0) {
        // ---- FIXED-DECISION measure over a reproducible WARM POOL (keeps the LIVE Gumbel search +
        // inference). build_warm_pool advances each of the total_slots slots a varied (seeded) number of
        // random legal actions, so the population spans a WIDE RANGE of belief sizes / plies (representative,
        // not all-openers — server-gen-floor-result.md follow-up); the runner runs --settle-decisions
        // UNCOUNTED (pipeline desync) then measures wcfg.decision_budget decisions, reporting the measured
        // WINDOW's decisions + nanosecond wall in the wire_summary (the window excludes settle, so dps is
        // steady-state). Abandoned episodes never reach redis, so the numbers are read from the wire_summary
        // — --parity-stats REQUIRED (ADR-0002 fail-loud). Reproducible in --pool-seed (comparable numbers).
        if (!stats_out) {
            std::cerr << "wire-ab-bench: FATAL: --measure-decisions requires --parity-stats (the window "
                         "count + wall are read from the wire_summary; abandoned episodes are not in redis).\n";
            return 2;
        }
        const std::string stats_path = std::string(*opt(args, "--parity-stats"));
        std::vector<SlotSnapshot> pool;
        if (pool_plies > 0) pool = build_warm_pool(env, total_slots, pool_seed, pool_plies);
        RunnerConfig rcfg;
        rcfg.run = std::string(*run); rcfg.phase = "gen"; rcfg.version = version;
        rcfg.episodes = 1'000'000'000;   // effectively unbounded — the decision budget bounds the work
        rcfg.lam = lam; rcfg.max_steps = max_steps; rcfg.seed = 7919ull;
        rcfg.res_token = std::string(*res_token) + "-measure";
        WireRunnerConfig wcm = wcfg; wcm.decision_budget = measure_decisions;
        auto t0 = std::chrono::steady_clock::now();
        auto w = run_episodes_wire_pipelined(env, fb, gc, *redis, rcfg, wcm, stats_out,
                                             pool_plies > 0 ? &pool : nullptr, settle_decisions);
        if (!w) { std::cerr << "wire-ab-bench: FATAL: measure pass failed: " << w.error().message << "\n"; return 1; }
        const double outer_wall = secs(t0, std::chrono::steady_clock::now());
        total_eps = *w;
        stats_file.flush();
        const long long meas_dec = read_stat_ll(stats_path, "\"measure_decisions\":");
        const long long meas_ns = read_stat_ll(stats_path, "\"measure_wall_ns\":");
        total_decisions = static_cast<long>(meas_dec);
        // dps = measured decisions / the measured WINDOW wall (steady-state, settle excluded). Fall back to
        // the outer wall if the window never opened (the budget was never reached — a wedge).
        measure_wall = meas_ns > 0 ? static_cast<double>(meas_ns) * 1e-9 : outer_wall;
        measure_eps = measure_decisions;
    } else {
        // ---- LEGACY episode-wave measure (full self-play episodes; dps from redis result blocks). WARMUP
        // (NOT counted): one full-occupancy fill + JIT, measuring eps_rate to size the measure pass.
        const int warmup_eps = std::max(total_slots, 8);
        long warmup_eps_done = 0;
        auto warm0 = std::chrono::steady_clock::now();
        {
            RunnerConfig rcfg;
            rcfg.run = std::string(*run); rcfg.phase = "gen"; rcfg.version = version;
            rcfg.episodes = warmup_eps; rcfg.lam = lam; rcfg.max_steps = max_steps; rcfg.seed = 104729ull;
            rcfg.res_token = std::string(*res_token) + "-warmup";
            auto w = run_episodes_wire_batched(env, fb, gc, *redis, rcfg, wcfg, nullptr);
            if (!w) { std::cerr << "wire-ab-bench: FATAL: warmup failed: " << w.error().message << "\n"; return 1; }
            warmup_eps_done = *w;
        }
        warmup_wall = secs(warm0, std::chrono::steady_clock::now());
        eps_rate = (warmup_wall > 0.0 && warmup_eps_done > 0) ? (warmup_eps_done / warmup_wall) : 1.0;
        const int meas_eps_i = static_cast<int>(std::max<long>(static_cast<long>(budget * eps_rate), total_slots));
        measure_eps = meas_eps_i;
        const std::string tok = std::string(*res_token) + "-measure";
        auto t0 = std::chrono::steady_clock::now();
        {
            RunnerConfig rcfg;
            rcfg.run = std::string(*run); rcfg.phase = "gen"; rcfg.version = version;
            rcfg.episodes = meas_eps_i; rcfg.lam = lam; rcfg.max_steps = max_steps; rcfg.seed = 7919ull;
            rcfg.res_token = tok;
            auto w = run_episodes_wire_batched(env, fb, gc, *redis, rcfg, wcfg, stats_out);
            if (!w) { std::cerr << "wire-ab-bench: FATAL: measure pass failed: " << w.error().message << "\n"; return 1; }
            total_eps = *w;
            total_decisions = count_decisions(tok, meas_eps_i);
        }
        measure_wall = secs(t0, std::chrono::steady_clock::now());
    }

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
