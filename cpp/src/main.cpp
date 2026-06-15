// cpp/src/main.cpp
// Purpose: the chocofarm-cpp-runner entrypoint. Loads the instance geometry (instance.json +
//   the DERIVED faces.json), builds the env + feature builder, connects to redis via the
//   CHOCO_REDIS_* env contract, reads weights for (run, phase, version) — exercising the weight-
//   read seam (P7) — and runs E episodes of the INJECTED RandomPolicy (the env<->Policy seam, P2),
//   writing the (X, PI, M, Y) result blocks. lam / m-as-episodes / max_steps arrive as LIVE CLI
//   scalars (P4), never baked in. The Gumbel search and MLP forward are deferred (this is the
//   dumb-random seam-proof MVP, ADR-0012's C++ section + scaling-and-cpp-seam.md Shape A).
//
// Public Domain (The Unlicense).
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

#include "chocofarm/env.hpp"
#include "chocofarm/features.hpp"
#include "chocofarm/instance.hpp"
#include "chocofarm/policy.hpp"
#include "chocofarm/runner.hpp"
#include "chocofarm/transport.hpp"

namespace {

void usage(const char* prog) {
    std::cerr <<
        "usage: " << prog << " [options]\n"
        "  --instance <path>   path to data/instance.json (required)\n"
        "  --faces <path>      path to data/faces.json (the DERIVED cover; required)\n"
        "  --run <id>          weight namespace run id (required)\n"
        "  --phase <gen|eval>  weight phase (default gen)\n"
        "  --version <int>     weight version (default 0)\n"
        "  --episodes <int>    number of random episodes E (default 1)\n"
        "  --lam <float>       live rate target λ (default 0.0)\n"
        "  --max-steps <int>   live episode horizon (default 40)\n"
        "  --seed <uint>       per-episode RNG seed base (default 0)\n"
        "  --res-token <id>    result-key namespace token (required)\n"
        "  --parity-stats <p>  ALSO write per-episode aggregate stats (JSON lines) to <p>\n"
        "Connection: CHOCO_REDIS_HOST/PORT/DB env (default 127.0.0.1:6379 db0).\n";
}

const char* opt(int argc, char** argv, const char* name) {
    for (int i = 1; i + 1 < argc; ++i)
        if (std::strcmp(argv[i], name) == 0) return argv[i + 1];
    return nullptr;
}

}  // namespace

int main(int argc, char** argv) {
    const char* instance = opt(argc, argv, "--instance");
    const char* faces = opt(argc, argv, "--faces");
    const char* run = opt(argc, argv, "--run");
    const char* res_token = opt(argc, argv, "--res-token");
    if (!instance || !faces || !run || !res_token) {
        usage(argv[0]);
        return 2;
    }
    chocofarm::RunnerConfig cfg;
    cfg.run = run;
    cfg.res_token = res_token;
    cfg.phase = opt(argc, argv, "--phase") ? opt(argc, argv, "--phase") : "gen";
    cfg.version = opt(argc, argv, "--version") ? std::atoi(opt(argc, argv, "--version")) : 0;
    cfg.episodes = opt(argc, argv, "--episodes") ? std::atoi(opt(argc, argv, "--episodes")) : 1;
    cfg.lam = opt(argc, argv, "--lam") ? std::atof(opt(argc, argv, "--lam")) : 0.0;
    cfg.max_steps = opt(argc, argv, "--max-steps") ? std::atoi(opt(argc, argv, "--max-steps")) : 40;
    cfg.seed = opt(argc, argv, "--seed")
                   ? std::strtoull(opt(argc, argv, "--seed"), nullptr, 10) : 0ULL;

    try {
        chocofarm::Instance inst = chocofarm::load_instance(instance, faces);
        chocofarm::Environment env(inst);
        chocofarm::FeatureBuilder fb(env);
        chocofarm::RandomPolicy policy;          // the trivial composable Policy (P2 drop-in)
        chocofarm::RedisClient redis;            // CHOCO_REDIS_* contract (no hardcoded port)
        const char* stats_path = opt(argc, argv, "--parity-stats");
        std::ofstream stats_file;
        std::ostream* stats_out = nullptr;
        if (stats_path) {
            stats_file.open(stats_path);
            if (!stats_file) throw std::runtime_error(std::string("cannot open --parity-stats: ") + stats_path);
            stats_file << std::setprecision(17);  // full float64 round-trip for the harness
            stats_out = &stats_file;
        }
        int n = chocofarm::run(env, fb, policy, redis, cfg, stats_out);
        std::cerr << "chocofarm-cpp-runner: wrote " << n << " episode(s) under res_token="
                  << cfg.res_token << " (run=" << cfg.run << " phase=" << cfg.phase
                  << " version=" << cfg.version << " feat_dim=" << fb.dim()
                  << " n_slots=" << chocofarm::n_action_slots(env) << ")\n";
        return 0;
    } catch (const std::exception& e) {
        // ADR-0002 / P5: a missing weight payload, an unreachable redis, or a malformed instance is
        // a LOUD abort (non-zero exit + diagnostic), never a silent partial run.
        std::cerr << "chocofarm-cpp-runner: FATAL: " << e.what() << "\n";
        return 1;
    }
}
