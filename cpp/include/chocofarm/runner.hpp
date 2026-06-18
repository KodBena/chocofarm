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
#include <expected>
#include <ostream>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "chocofarm/env.hpp"
#include "chocofarm/error.hpp"
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

    // ---- the exact episode trace (for the WIRE-CONTENT cross-impl parity check) ----
    // The true world + the executed-slot sequence (TERMINATE recorded as the last slot when the
    // episode terminated on it). Given (world, slots) the env is deterministic, so the harness can
    // replay the SAME episode in Python and value-compare the wire PI / Y / X / M bytes against an
    // INDEPENDENT Python computation (not just illegal-mass + shape). This closes the wire-content
    // parity gap ADR-0012 P7 flags as "review-only until a manifest-round-trip parity test
    // mechanizes it".
    uint32_t world = 0;
    std::vector<int> exec_slots;  // executed action slots in order (+ TERMINATE slot if it ended so)
};

// EpisodeBuilder — the per-episode RECORD-ASSEMBLY accumulator extracted from run_episode (the ONE home
// for the float-sensitive value-target suffix math, ADR-0012 P1/P3 / docs/design/
// cpp-local-batched-runtime.md Chunk 3). It owns the (feat, pi, mask) record accumulation and the
// pure-MC λ-penalized return-to-go finalization, WITHOUT the serial ply loop — so the K-fiber-mux
// batched driver (run_episodes_batched) can drive K episodes' record accumulation + finalization
// concurrently while run_episode keeps its serial loop. The suffix math at finalize() is moved VERBATIM
// from the former run_episode body (behaviour-preserving extraction): byte-identical EpisodeBlocks
// before/after (the §5 layer-3 wire-content parity gate).
//
// Value semantics: built by the static factory create(), fed record_decision()/record_step() per ply,
// consumed by finalize() && (rvalue-qualified — the builder is spent). It holds the per-episode aggregate
// counters (n_collect/n_sense/n_terminate) and the world + bw0 for the stats; lam is captured at create
// (the live per-decision λ, P4 — constant within one episode).
class EpisodeBuilder {
  public:
    // Build the accumulator for one episode. `feat_dim`/`n_slots` size the output blocks; `world`/`bw0`
    // are stamped into the stats; `lam` is this episode's live Dinkelbach penalty (constant within the
    // episode). The env/fb are NOT held (the builder is pure record assembly); the caller owns dynamics.
    [[nodiscard]] static EpisodeBuilder create(uint32_t world, double lam, int feat_dim, int n_slots,
                                               int bw0);

    // Record one DECIDED ply (mirrors run_episode's record block, runner.cpp:54-67): the feature row,
    // the improved-π PI row, and the legality mask. `is_terminate` marks the trailing TERMINATE decision
    // (which executes no step); `exec_slot` is the executed action slot (or the TERMINATE slot). Moves
    // the vectors in (the caller is done with them).
    void record_decision(std::vector<double> feat, std::vector<float> pi, std::vector<float> mask,
                         bool is_terminate, bool is_collect, int exec_slot);

    // Record the env.apply result for a NON-TERMINATE step (mirrors run_episode's step push,
    // runner.cpp:68-69): the immediate (reward, dt) the value-target suffix sums. Call exactly once per
    // record_decision whose is_terminate==false, AFTER that record_decision.
    void record_step(double reward, double dt);

    // The suffix-return value target + the aggregate stats (the verbatim runner.cpp:72-117 math),
    // CONSUMED (rvalue-qualified). `exit_cost` is env.exit_cost(loc.pt) at episode end (the exit toll in
    // every value suffix). `nb_final` is env.nb(bw) at episode end (for the belief-shrinkage stat).
    [[nodiscard]] EpisodeBlocks finalize(double exit_cost, int nb_final) &&;

  private:
    EpisodeBuilder() = default;

    uint32_t world_ = 0;
    double lam_ = 0.0;
    int feat_dim_ = 0;
    int n_slots_ = 0;
    int bw0_ = 0;
    std::vector<std::vector<double>> feats_;
    std::vector<std::vector<float>> pis_;
    std::vector<std::vector<float>> masks_;
    std::vector<std::pair<double, double>> step_rt_;  // (r, dt) per EXECUTED (non-TERMINATE) step
    std::vector<int> exec_slots_;
    int n_collect_ = 0;
    int n_sense_ = 0;
    int n_terminate_ = 0;
};

// Run ONE episode against `world` under `policy`, building the per-decision records. `rng` is the
// per-episode RNG (seeded by the caller). `max_steps` is the live horizon (P4). Mirrors
// generate_episode: record (feat, pi, mask, g) per decision incl. a trailing TERMINATE decision,
// with g the pure-MC suffix return-to-go. Re-expressed (Chunk 3) as EpisodeBuilder + the serial ply
// loop — behaviour-preserving (byte-identical EpisodeBlocks before/after).
[[nodiscard]] EpisodeBlocks run_episode(const Environment& env, const FeatureBuilder& fb,
                                        const Policy& policy, uint32_t world, double lam,
                                        std::mt19937_64& rng, int max_steps);

// A splitmix64-style fold over (cfg.seed, episode idx) — the C++ runner's OWN per-episode seeding (NOT
// the Python worker's numpy seed fold; the RNGs differ across the language boundary by design, so parity
// is the ADR-0012 P6 behavioral bar). The ONE home (P1): both run_episodes and the LOCAL batched driver
// (run_episodes_batched) seed each episode's persistent rng with THIS fold, so they pick the SAME world
// + draw the SAME stream per idx (the byte-identity basis for the batched↔serial parity).
[[nodiscard]] uint64_t fold_seed(uint64_t seed, int idx);

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
// a world from the belief and run+write it. Returns the number of episodes written, OR a typed Error
// (ADR-0012 P9 rule 5: a missing weight payload / a failed redis write is a recoverable boundary
// failure returned by value, never a throw). The episode i world + RNG are drawn from a seed fold
// over (cfg.seed, i) so the harness can match worlds.
//
// `stats_out` (optional): when non-null, the runner ALSO writes one JSON-object line per episode to
// it — {"length","lam_return","n_collect","n_sense","n_terminate","belief_shrinkage"} — the
// aggregate-stat sink the ADR-0012 P6 parity harness reads. This is purely additive to the redis
// result write (the wire proof); it does not change what crosses the wire.
[[nodiscard]] std::expected<int, Error> run(const Environment& env, const FeatureBuilder& fb,
                                            const Policy& policy, RedisClient& redis,
                                            const RunnerConfig& cfg, std::ostream* stats_out = nullptr);

// The episode loop, factored out of run() (P1 — one loop, two entry points). For each of cfg.episodes
// episodes it folds the per-episode seed over (cfg.seed, idx), draws the world, and run_episode +
// write_results. It does NOT read weights — the CALLER owns the net lifecycle: the one-shot run() reads
// once up front and delegates here, while the persistent --serve loop (serve.hpp) reloads the net only
// when `version` advances and then calls this with the already-rebuilt policy. `stats_out` is the same
// optional P6 aggregate-stat sink run() forwards.
[[nodiscard]] std::expected<int, Error> run_episodes(const Environment& env, const FeatureBuilder& fb,
                                                     const Policy& policy, RedisClient& redis,
                                                     const RunnerConfig& cfg,
                                                     std::ostream* stats_out = nullptr);

}  // namespace chocofarm
