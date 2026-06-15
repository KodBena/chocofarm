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
//   ADR-0012 P9: the imperative shell. argv (the untyped char** the OS hands main) is decoded ONCE
//   into a typed std::vector<std::string_view> — the Port/ACL translate-at-the-edge (P2) — and the
//   CLI helper `opt` returns a [[nodiscard]] std::optional<std::string_view> (a missing flag is
//   routine ABSENCE, not a failure or a nullable raw pointer — rules 1 & 5). Boundary failures
//   (a malformed instance, an unreachable redis, a missing weight payload) arrive as typed
//   std::expected and are reported loudly here (ADR-0002), never thrown.
//
// Public Domain (The Unlicense).
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "chocofarm/env.hpp"
#include "chocofarm/features.hpp"
#include "chocofarm/instance.hpp"
#include "chocofarm/nmcs.hpp"
#include "chocofarm/policy.hpp"
#include "chocofarm/runner.hpp"
#include "chocofarm/transport.hpp"

namespace {

void usage(std::string_view prog) {
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

// The CLI flag lookup (ADR-0012 P9 rules 1 & 5): typed bounds-carrying input (a span of views), a
// [[nodiscard]] std::optional<std::string_view> output — a missing flag is routine absence carried
// in the type, never a nullable raw pointer whose missed check is undefined behavior.
[[nodiscard]] std::optional<std::string_view> opt(std::span<const std::string_view> args,
                                                  std::string_view name) {
    for (size_t i = 1; i + 1 < args.size(); ++i)
        if (args[i] == name) return args[i + 1];
    return std::nullopt;
}

// Parse helpers over the typed view, preserving the as-merged numeric behavior exactly (std::ato*
// over the flag's C-string value, which argv elements always are). A std::string_view from argv is
// null-terminated; building a std::string keeps the same atoi/atof/strtoull conversion.
[[nodiscard]] int to_int(std::string_view s) { return std::atoi(std::string(s).c_str()); }
[[nodiscard]] double to_double(std::string_view s) { return std::atof(std::string(s).c_str()); }
[[nodiscard]] uint64_t to_u64(std::string_view s) {
    return std::strtoull(std::string(s).c_str(), nullptr, 10);
}

}  // namespace

int main(int argc, char** argv) {
    // The Port/ACL (P2): decode the untyped argv ONCE into typed views; every signature downstream
    // is typed (ADR-0012 P9 — not an excuse to keep raw pointers flowing inward).
    std::vector<std::string_view> args(argv, argv + argc);
    std::string_view prog = args.empty() ? "chocofarm-cpp-runner" : args[0];

    std::optional<std::string_view> instance = opt(args, "--instance");
    std::optional<std::string_view> faces = opt(args, "--faces");
    std::optional<std::string_view> run = opt(args, "--run");
    std::optional<std::string_view> res_token = opt(args, "--res-token");
    if (!instance || !faces || !run || !res_token) {
        usage(prog);
        return 2;
    }
    chocofarm::RunnerConfig cfg;
    cfg.run = std::string(*run);
    cfg.res_token = std::string(*res_token);
    cfg.phase = std::string(opt(args, "--phase").value_or("gen"));
    if (auto v = opt(args, "--version")) cfg.version = to_int(*v);
    if (auto v = opt(args, "--episodes")) cfg.episodes = to_int(*v);
    else cfg.episodes = 1;
    if (auto v = opt(args, "--lam")) cfg.lam = to_double(*v);
    if (auto v = opt(args, "--max-steps")) cfg.max_steps = to_int(*v);
    if (auto v = opt(args, "--seed")) cfg.seed = to_u64(*v);

    // strategy selection over the env<->Policy seam: --policy random|nmcs (P2 — the runner core never
    // names a concrete Policy; this is the ONE place a policy is chosen). NMCS knobs are live CLI
    // scalars too (P4), defaulting to NMCSConfig's.
    std::string_view policy_name = opt(args, "--policy").value_or("random");

    std::unique_ptr<chocofarm::Policy> policy;
    if (policy_name == "random") {
        policy = std::make_unique<chocofarm::RandomPolicy>();  // the trivial composable Policy
    } else if (policy_name == "nmcs") {
        chocofarm::NMCSConfig nc;  // defaults match NMCSConfig (level=1, ps=3, ss=2, 4/4, 24)
        if (auto v = opt(args, "--nmcs-level")) nc.level = to_int(*v);
        if (auto v = opt(args, "--nmcs-playouts")) nc.playout_samples = to_int(*v);
        if (auto v = opt(args, "--nmcs-step-samples")) nc.step_samples = to_int(*v);
        if (auto v = opt(args, "--nmcs-cand-det")) nc.cand_det = to_int(*v);
        if (auto v = opt(args, "--nmcs-cand-tre")) nc.cand_tre = to_int(*v);
        if (auto v = opt(args, "--nmcs-max-steps")) nc.max_steps = to_int(*v);
        policy = std::make_unique<chocofarm::NMCSPolicy>(nc);  // nested Monte-Carlo search (P2 drop-in)
    } else {
        // ADR-0002 / P5: an unknown policy is a loud abort at the boundary (a CLI misuse).
        std::cerr << prog << ": FATAL: unknown --policy: " << policy_name
                  << " (expected random | nmcs)\n";
        return 1;
    }

    // ---- the boundary: every fallible step returns a typed Error reported loudly here (P9 / ADR-0002) ----
    auto inst = chocofarm::load_instance(*instance, *faces);
    if (!inst) {
        std::cerr << prog << ": FATAL: " << inst.error().message << "\n";
        return 1;
    }
    chocofarm::Environment env(*inst);
    chocofarm::FeatureBuilder fb(env);

    auto redis = chocofarm::RedisClient::create();  // CHOCO_TRANSPORT_REDIS_* contract (no hardcoded port)
    if (!redis) {
        std::cerr << prog << ": FATAL: " << redis.error().message << "\n";
        return 1;
    }

    std::ofstream stats_file;
    std::ostream* stats_out = nullptr;
    if (auto stats_path = opt(args, "--parity-stats")) {
        stats_file.open(std::string(*stats_path));
        if (!stats_file) {
            std::cerr << prog << ": FATAL: cannot open --parity-stats: " << *stats_path << "\n";
            return 1;
        }
        stats_file << std::setprecision(17);  // full float64 round-trip for the harness
        stats_out = &stats_file;
    }

    auto written = chocofarm::run(env, fb, *policy, *redis, cfg, stats_out);
    if (!written) {
        // ADR-0002 / P5: a missing weight payload, an unreachable redis, or a failed write is a LOUD
        // abort (non-zero exit + diagnostic), never a silent partial run.
        std::cerr << prog << ": FATAL: " << written.error().message << "\n";
        return 1;
    }
    std::cerr << prog << ": wrote " << *written << " episode(s) under res_token="
              << cfg.res_token << " (policy=" << policy_name << " run=" << cfg.run
              << " phase=" << cfg.phase << " version=" << cfg.version << " feat_dim=" << fb.dim()
              << " n_slots=" << chocofarm::n_action_slots(env) << ")\n";
    return 0;
}
