// cpp/src/wire_batched_smoke.cpp
// Purpose: the Phase B SMOKE for run_episodes_wire_batched (NOT the runner) — docs/design/
//   cpp-wire-generation-roadmap.md Phase B checkpoint. It runs `--episodes` self-play episodes through the
//   wire-batched generation driver at (--pool-threads, --pool-batch) against a WARM Python InferenceServer
//   on `--endpoint` (stood up by cpp/parity/_wire_smoke_server.py over the production hidden-256 geometry),
//   writing the four (X, PI, M, Y) float32 EpisodeBlocks per non-empty episode to redis under `--res-token`.
//   It prints the written count; the Python smoke harness reads the blocks back and asserts sane shapes
//   (X (n,feat_dim), PI/M (n,n_slots), Y (n,)) and the expected episode count.
//
//   The net is read off redis at (run,"gen",version) — the SAME blob the server loaded — but the wire
//   driver itself calls NO local forward (the leaf is remote, on the JAX server); the read here is only the
//   weight-seam exercise the smoke wants (it does not build a NetForward). The OFF-LIMITS local-batched
//   runner is NEVER linked or referenced (Override O-2).
//
//   Protocol:  wire-batched-smoke --instance <p> --faces <p> --endpoint <ipc://...> --run <id> --res-token
//                  <t> [--version v --episodes N --m N --n-sims N --max-depth N --c-outcome N --lam f
//                   --max-steps N --seed S --pool-threads T --pool-batch B]
//   Output:    a config line, then "RESULT: PASS wrote=<n> ..." + exit 0, or a loud failure + exit 1/3.
//
// Public Domain (The Unlicense).
#include <cstdint>
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
[[nodiscard]] uint64_t to_u64(std::string_view s) {
    return std::strtoull(std::string(s).c_str(), nullptr, 10);
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
        std::cerr << "usage: wire-batched-smoke --instance <p> --faces <p> --endpoint <ipc://...> --run "
                     "<id> --res-token <t> [--version v --episodes N --m N --n-sims N --max-depth N "
                     "--c-outcome N --lam f --max-steps N --seed S --pool-threads T --pool-batch B]\n";
        return 2;
    }
    RunnerConfig rc;
    rc.run = std::string(*run);
    rc.phase = "gen";
    rc.res_token = std::string(*res_token);
    rc.version = opt(args, "--version") ? to_int(*opt(args, "--version")) : 0;
    rc.episodes = opt(args, "--episodes") ? to_int(*opt(args, "--episodes")) : 8;
    rc.lam = opt(args, "--lam") ? to_double(*opt(args, "--lam")) : 0.1;
    rc.max_steps = opt(args, "--max-steps") ? to_int(*opt(args, "--max-steps")) : 40;
    rc.seed = opt(args, "--seed") ? to_u64(*opt(args, "--seed")) : 0;

    GumbelConfig gc;
    gc.n_sims = SimBudget{48};
    if (auto v = opt(args, "--m")) gc.m = CandidateCount{static_cast<CandidateCount::rep_type>(to_int(*v))};
    if (auto v = opt(args, "--n-sims")) gc.n_sims = SimBudget{static_cast<SimBudget::rep_type>(to_int(*v))};
    if (auto v = opt(args, "--max-depth")) gc.max_depth = PlyDepth{static_cast<PlyDepth::rep_type>(to_int(*v))};
    if (auto v = opt(args, "--c-outcome")) gc.c_outcome = OutcomeIndex{static_cast<OutcomeIndex::rep_type>(to_int(*v))};

    WireRunnerConfig wcfg;
    wcfg.endpoint = std::string(*endpoint);
    wcfg.pool_threads = opt(args, "--pool-threads") ? to_int(*opt(args, "--pool-threads")) : 1;
    wcfg.pool_batch = opt(args, "--pool-batch") ? to_int(*opt(args, "--pool-batch")) : 4;

    auto inst = load_instance(*instance, *faces);
    if (!inst) {
        std::cerr << "wire-batched-smoke: FATAL: " << inst.error().message << "\n";
        return 1;
    }
    Environment env(*inst);
    FeatureBuilder fb(env);

    auto redis = RedisClient::create();
    if (!redis) {
        std::cerr << "wire-batched-smoke: FATAL: " << redis.error().message << "\n";
        return 1;
    }
    // exercise the weight-read seam (the published net the server also loaded); the driver itself does NOT
    // build a NetForward (the leaf is remote). A missing payload is a loud abort.
    auto wp = redis->read_weights(rc.run, "gen", rc.version);
    if (!wp) {
        std::cerr << "wire-batched-smoke: FATAL: weight read (" << rc.run << ",gen," << rc.version
                  << ") failed: " << wp.error().message << "\n";
        return 1;
    }

    std::cout << "config: episodes=" << rc.episodes << " endpoint=" << wcfg.endpoint
              << " pool_threads=" << wcfg.pool_threads << " pool_batch=" << wcfg.pool_batch
              << " m=" << gc.m.value() << " n_sims=" << gc.n_sims.value() << " feat_dim=" << fb.dim().value()
              << " n_slots=" << n_action_slots(env).value() << " res_token=" << rc.res_token << "\n";

    auto written = run_episodes_wire_batched(env, fb, gc, *redis, rc, wcfg, nullptr);
    if (!written) {
        std::cerr << "wire-batched-smoke: FATAL: " << written.error().message << "\n";
        return 3;
    }
    std::cout << "RESULT: PASS wrote=" << *written << " (of " << rc.episodes << " episodes) endpoint="
              << wcfg.endpoint << "\n";
    return 0;
}
