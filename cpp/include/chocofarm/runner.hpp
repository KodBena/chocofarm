// cpp/include/chocofarm/runner.hpp
// Purpose: the C++ runner — runs E self-play episodes via an INJECTED Policy and writes the four
//   (X, PI, M, Y) result blocks to redis, mirroring chocofarm/az/worker.py's generate_episode flow.
//   It is a composition of the already-clean seams: it reads weights for (run, phase, version) over
//   the wire (P7, exercising the weight-read seam even though RandomPolicy ignores them), runs the
//   episode through the env<->Policy seam (P2), and emits the result bytes (no second encoder, P7).
//
//   `lam` / `max_steps` arrive as LIVE scalars (ADR-0012 P4), not baked into the runner object. The
//   value target is the pure-MC λ-penalized return-to-go (suffix_returns_to_go), the
//   lam_blend=1/n_step=None limit generate_episode produces by default; PI is the RandomPolicy's own
//   uniform-over-legal action distribution (the natural improved-policy target for a search-free
//   policy). The Gumbel search and the MLP forward are deferred to a later slice.
//
// Public Domain (The Unlicense).
#pragma once

#include <cstdint>
#include <ostream>
#include <random>
#include <string>

#include "chocofarm/env.hpp"
#include "chocofarm/features.hpp"
#include "chocofarm/policy.hpp"
#include "chocofarm/transport.hpp"

namespace chocofarm {

// One episode's per-decision records (mirrors generate_episode's (feat, pi, mask, g) list), already
// stacked into the four contiguous float32 blocks the wire expects.
struct EpisodeBlocks {
    int n = 0;              // number of recorded decisions
    int feat_dim = 0;
    int n_slots = 0;
    std::vector<float> X;   // (n, feat_dim) row-major
    std::vector<float> PI;  // (n, n_slots) row-major
    std::vector<float> M;   // (n, n_slots) row-major
    std::vector<float> Y;   // (n,)

    // ---- per-episode aggregate stats (for the ADR-0012 P6 behavioral-parity harness) ----
    // These are the float-sensitive / RNG-driven quantities the harness compares vs the Python
    // RandomPolicy reference within Monte-Carlo CI (NOT byte-identity). Computed during the run.
    int ep_length = 0;        // number of EXECUTED (non-TERMINATE) decisions
    double lam_return = 0.0;  // the full-episode λ-return ΣR − λ(ΣT + exit_c)
    int n_collect = 0;        // # executed ("t", i) collect actions
    int n_sense = 0;          // # executed ("d", j) sense actions
    int n_terminate = 0;      // 1 if the episode ended on a TERMINATE decision, else 0
    double belief_shrinkage = 0.0;  // 1 − |bw_final| / |bw_initial| (how much belief was resolved)
};

// Run ONE episode against `world` under `policy`, building the per-decision records. `rng` is the
// per-episode RNG (seeded by the caller). `max_steps` is the live horizon (P4). Mirrors
// generate_episode: record (feat, pi, mask, g) per decision incl. a trailing TERMINATE decision,
// with g the pure-MC suffix return-to-go.
EpisodeBlocks run_episode(const Environment& env, const FeatureBuilder& fb, const Policy& policy,
                          uint32_t world, double lam, std::mt19937_64& rng, int max_steps);

// Configuration for a runner pass (all live scalars; P4). `run`/`phase`/`version` select the weights
// to read; `episodes` is E; `lam`/`max_steps` are the live knobs; `seed` seeds the per-episode RNG
// stream; `res_token` namespaces the result keys.
struct RunnerConfig {
    std::string run;
    std::string phase = "gen";
    int version = 0;
    int episodes = 0;
    double lam = 0.0;
    int max_steps = 40;
    uint64_t seed = 0;
    std::string res_token;
};

// The runner entrypoint: read weights (exercising the weight-read seam), then for each episode draw
// a world from the belief and run+write it. Returns the number of episodes written. The episode i
// world + RNG are drawn from a seed fold over (cfg.seed, i) so the harness can match worlds.
//
// `stats_out` (optional): when non-null, the runner ALSO writes one JSON-object line per episode to
// it — {"length","lam_return","n_collect","n_sense","n_terminate","belief_shrinkage"} — the
// aggregate-stat sink the ADR-0012 P6 parity harness reads. This is purely additive to the redis
// result write (the wire proof); it does not change what crosses the wire.
int run(const Environment& env, const FeatureBuilder& fb, const Policy& policy,
        RedisClient& redis, const RunnerConfig& cfg, std::ostream* stats_out = nullptr);

}  // namespace chocofarm
