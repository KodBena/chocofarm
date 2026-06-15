// cpp/src/main.cpp
// Purpose: the chocofarm-cpp-runner entrypoint. Loads the instance geometry (instance.json +
//   the DERIVED faces.json), builds the env + feature builder, connects to redis via the
//   CHOCO_TRANSPORT_REDIS_* env contract, reads weights for (run, phase, version) — exercising the weight-
//   read seam (P7) — and runs E episodes of an INJECTED Policy (the env<->Policy seam, P2), writing
//   the (X, PI, M, Y) result blocks. lam / m-as-episodes / max_steps arrive as LIVE CLI scalars (P4),
//   never baked in. The policy is a clean strategy selection over `--policy random|nmcs`: RandomPolicy
//   (the seam-proof baseline) or NMCSPolicy (the nested Monte-Carlo search, nmcs.hpp). The runner
//   never names a concrete Policy — adding NMCS is ZERO runner-core edits (the P2 seam). The Gumbel
//   search and MLP forward remain deferred (ADR-0012's C++ section + scaling-and-cpp-seam.md Shape A).
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
#include "chocofarm/nmcs.hpp"
#include "chocofarm/policy.hpp"
#include "chocofarm/runner.hpp"
#include "chocofarm/transport.hpp"

#include <memory>

namespace {

void usage(const char* prog) {
    std::cerr <<
        "usage: " << prog << " [options]\n"
        "  --instance <path>   path to data/instance.json (required)\n"
        "  --faces <path>      path to data/faces.json (the DERIVED cover; required)\n"
        "  --run <id>          weight namespace run id (required)\n"
        "  --phase <gen|eval>  weight phase (default gen)\n"
        "  --version <int>     weight version (default 0)\n"
        "  --episodes <int>    number of episodes E (default 1)\n"
        "  --lam <float>       live rate target λ (default 0.0)\n"
        "  --max-steps <int>   live episode horizon (default 40)\n"
        "  --seed <uint>       per-episode RNG seed base (default 0)\n"
        "  --res-token <id>    result-key namespace token (required)\n"
        "  --policy <name>     search policy: random | nmcs (default random)\n"
        "  --nmcs-level <int>      NMCS nesting level (default 1; 2 is the milestone)\n"
        "  --nmcs-playouts <int>   worlds per level-0 playout (default 3)\n"
        "  --nmcs-step-samples <i> worlds per per-move eval (default 2)\n"
        "  --nmcs-cand-det <int>   nearest informative detectors kept (default 4)\n"
        "  --nmcs-cand-tre <int>   nearest uncollected treasures kept (default 4)\n"
        "  --nmcs-max-steps <int>  hard cap on a search line (default 24)\n"
        "  --parity-stats <p>  ALSO write per-episode aggregate stats (JSON lines) to <p>\n"
        "Connection: CHOCO_TRANSPORT_REDIS_HOST/PORT/DB env (default 127.0.0.1:6380 db0).\n";
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

    // strategy selection over the env<->Policy seam: --policy random|nmcs (P2 — the runner core never
    // names a concrete Policy; this is the ONE place a policy is chosen). NMCS knobs are live CLI
    // scalars too (P4), defaulting to NMCSConfig's.
    const char* policy_name = opt(argc, argv, "--policy") ? opt(argc, argv, "--policy") : "random";

    try {
        chocofarm::Instance inst = chocofarm::load_instance(instance, faces);
        chocofarm::Environment env(inst);
        chocofarm::FeatureBuilder fb(env);

        std::unique_ptr<chocofarm::Policy> policy;
        if (std::strcmp(policy_name, "random") == 0) {
            policy = std::make_unique<chocofarm::RandomPolicy>();  // the trivial composable Policy
        } else if (std::strcmp(policy_name, "nmcs") == 0) {
            chocofarm::NMCSConfig nc;  // defaults match NMCSConfig (level=1, ps=3, ss=2, 4/4, 24)
            if (opt(argc, argv, "--nmcs-level")) nc.level = std::atoi(opt(argc, argv, "--nmcs-level"));
            if (opt(argc, argv, "--nmcs-playouts"))
                nc.playout_samples = std::atoi(opt(argc, argv, "--nmcs-playouts"));
            if (opt(argc, argv, "--nmcs-step-samples"))
                nc.step_samples = std::atoi(opt(argc, argv, "--nmcs-step-samples"));
            if (opt(argc, argv, "--nmcs-cand-det"))
                nc.cand_det = std::atoi(opt(argc, argv, "--nmcs-cand-det"));
            if (opt(argc, argv, "--nmcs-cand-tre"))
                nc.cand_tre = std::atoi(opt(argc, argv, "--nmcs-cand-tre"));
            if (opt(argc, argv, "--nmcs-max-steps"))
                nc.max_steps = std::atoi(opt(argc, argv, "--nmcs-max-steps"));
            policy = std::make_unique<chocofarm::NMCSPolicy>(nc);  // nested Monte-Carlo search (P2 drop-in)
        } else {
            throw std::runtime_error(std::string("unknown --policy: ") + policy_name +
                                     " (expected random | nmcs)");
        }
        chocofarm::RedisClient redis;            // CHOCO_TRANSPORT_REDIS_* contract (no hardcoded port)
        const char* stats_path = opt(argc, argv, "--parity-stats");
        std::ofstream stats_file;
        std::ostream* stats_out = nullptr;
        if (stats_path) {
            stats_file.open(stats_path);
            if (!stats_file) throw std::runtime_error(std::string("cannot open --parity-stats: ") + stats_path);
            stats_file << std::setprecision(17);  // full float64 round-trip for the harness
            stats_out = &stats_file;
        }
        int n = chocofarm::run(env, fb, *policy, redis, cfg, stats_out);
        std::cerr << "chocofarm-cpp-runner: wrote " << n << " episode(s) under res_token="
                  << cfg.res_token << " (policy=" << policy_name << " run=" << cfg.run
                  << " phase=" << cfg.phase << " version=" << cfg.version << " feat_dim=" << fb.dim()
                  << " n_slots=" << chocofarm::n_action_slots(env) << ")\n";
        return 0;
    } catch (const std::exception& e) {
        // ADR-0002 / P5: a missing weight payload, an unreachable redis, or a malformed instance is
        // a LOUD abort (non-zero exit + diagnostic), never a silent partial run.
        std::cerr << "chocofarm-cpp-runner: FATAL: " << e.what() << "\n";
        return 1;
    }
}
