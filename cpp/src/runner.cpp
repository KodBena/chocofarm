// cpp/src/runner.cpp
// Purpose: the C++ runner (see runner.hpp). Mirrors chocofarm/az/worker.py's generate_episode flow,
//   but with the injected RandomPolicy in place of the Gumbel search and the MLP forward (deferred).
//   It reads weights over the wire (exercising the weight-read seam, P7), runs E episodes through
//   the env<->Policy seam (P2) with lam/max_steps as live scalars (P4), and writes the four float32
//   result blocks (no second encoder, P7). The value target is the pure-MC λ-penalized return-to-go
//   (the lam_blend=1/n_step=None limit generate_episode produces by default).
//
// Public Domain (The Unlicense).
#include "chocofarm/runner.hpp"

#include <cmath>
#include <utility>
#include <vector>

namespace chocofarm {

EpisodeBuilder EpisodeBuilder::create(uint32_t world, double lam, int feat_dim, int n_slots, int bw0) {
    EpisodeBuilder b;
    b.world_ = world;
    b.lam_ = lam;
    b.feat_dim_ = feat_dim;
    b.n_slots_ = n_slots;
    b.bw0_ = bw0;
    return b;
}

void EpisodeBuilder::record_decision(std::vector<double> feat, std::vector<float> pi,
                                     std::vector<float> mask, bool is_terminate, bool is_collect,
                                     int exec_slot) {
    // mirrors run_episode's record block (runner.cpp:54-67): push the (feat, pi, mask) row + the trace
    // slot; bump the aggregate counters. The TERMINATE decision executes no step (record_step is NOT
    // called for it).
    feats_.push_back(std::move(feat));
    pis_.push_back(std::move(pi));
    masks_.push_back(std::move(mask));
    exec_slots_.push_back(exec_slot);
    if (is_terminate) {
        n_terminate_ = 1;
    } else if (is_collect) {
        ++n_collect_;
    } else {
        ++n_sense_;
    }
}

void EpisodeBuilder::record_step(double reward, double dt) {
    step_rt_.emplace_back(reward, dt);  // the executed (r, dt) the value-target suffix sums (runner.cpp:69)
}

EpisodeBlocks EpisodeBuilder::finalize(double exit_c, int nb_final) && {
    // ---- the verbatim runner.cpp:72-117 value-target + aggregate-stats math (Chunk 3 extraction) ----
    int n_dec = static_cast<int>(step_rt_.size());  // executed (non-TERMINATE) decisions
    int n_rec = static_cast<int>(feats_.size());    // all recorded decisions (incl. trailing TERMINATE)

    // value targets: pure-MC λ-penalized return-to-go (suffix_returns_to_go). g_j = Σ_{t>=j} r_t -
    // λ·(Σ_{t>=j} dt_t + exit_c). The exit toll is in every suffix (charged once at episode end).
    std::vector<double> g_steps(static_cast<size_t>(n_dec), 0.0);
    double suffix_r = 0.0, suffix_t = 0.0;
    for (int j = n_dec - 1; j >= 0; --j) {
        suffix_r += step_rt_[static_cast<size_t>(j)].first;
        suffix_t += step_rt_[static_cast<size_t>(j)].second;
        g_steps[static_cast<size_t>(j)] = suffix_r - lam_ * (suffix_t + exit_c);
    }

    EpisodeBlocks out;
    out.n = n_rec;
    out.feat_dim = feat_dim_;
    out.n_slots = n_slots_;
    // ---- aggregate stats (P6 behavioral parity) ----
    out.ep_length = n_dec;
    // the full-episode λ-return = the return-to-go from the FIRST executed decision; with no
    // executed step the episode is a bare exit, value = the exit toll −λ·exit_c.
    out.lam_return = (n_dec > 0) ? g_steps[0] : (-lam_ * exit_c);
    out.n_collect = n_collect_;
    out.n_sense = n_sense_;
    out.n_terminate = n_terminate_;
    out.belief_shrinkage = (bw0_ > 0) ? (1.0 - static_cast<double>(nb_final) /  // L6, via the seam
                                               static_cast<double>(bw0_)) : 0.0;
    out.world = world_;
    out.exec_slots = std::move(exec_slots_);
    out.X.resize(static_cast<size_t>(n_rec) * feat_dim_);
    out.PI.resize(static_cast<size_t>(n_rec) * n_slots_);
    out.M.resize(static_cast<size_t>(n_rec) * n_slots_);
    out.Y.resize(static_cast<size_t>(n_rec));
    for (int j = 0; j < n_rec; ++j) {
        for (int c = 0; c < feat_dim_; ++c)
            out.X[static_cast<size_t>(j) * feat_dim_ + c] =
                static_cast<float>(feats_[static_cast<size_t>(j)][static_cast<size_t>(c)]);
        for (int c = 0; c < n_slots_; ++c) {
            out.PI[static_cast<size_t>(j) * n_slots_ + c] =
                pis_[static_cast<size_t>(j)][static_cast<size_t>(c)];
            out.M[static_cast<size_t>(j) * n_slots_ + c] =
                masks_[static_cast<size_t>(j)][static_cast<size_t>(c)];
        }
        // the trailing TERMINATE decision (j >= n_dec) has no step: its target is the exit toll
        // continuation, -λ·exit_c (mirrors generate_episode's terminal-decision g).
        double g = (j < n_dec) ? g_steps[static_cast<size_t>(j)] : (-lam_ * exit_c);
        out.Y[static_cast<size_t>(j)] = static_cast<float>(g);
    }
    return out;
}

EpisodeBlocks run_episode(const Environment& env, const FeatureBuilder& fb, const Policy& policy,
                          uint32_t world, double lam, std::mt19937_64& rng, int max_steps) {
    const int n_slots = n_action_slots(env);
    const int feat_dim = fb.dim();

    // Live episode state (mirrors generate_episode's loc/bw/collected init: start at the entry).
    Loc loc{env.entry_point()};
    Belief bw = env.full_belief();   // belief = full world-set (the seam's construction entry)
    CollectedSet collected;

    const int bw0 = env.nb(bw);  // initial belief size (for belief-shrinkage stat) — via the seam (L6)

    // Re-expressed (Chunk 3) as the EpisodeBuilder accumulator + the serial ply loop — the suffix math
    // now lives in EpisodeBuilder::finalize (the ONE home; behaviour-preserving extraction).
    EpisodeBuilder eb = EpisodeBuilder::create(world, lam, feat_dim, n_slots, bw0);

    for (int ply = 0; ply < max_steps; ++ply) {
        if (env.empty(bw)) break;  // mirrors generate_episode's len(bw)==0 break (L6, via the seam)

        // decide + the improved-policy target (the env<->Policy seam): a SEARCH policy (Gumbel) returns
        // its real σ-transformed improved-π; a search-free policy returns the uniform-over-legal default
        // (illegal-slot mass == 0.0, the M invariant). lam is a live per-decision scalar (P4). PI is now
        // the policy's improved-π, so the C++ Gumbel actor emits a CORRECT AZ training target (not the
        // uniform fallback the runner used while the search was deferred).
        ActionAndPi ap = policy.decide_target(env, loc, bw, collected, lam, rng);
        Action action = ap.action;

        // the §2.2 feature vector + the legality mask for THIS belief (mirrors generate_episode:
        // fb.build then legal_mask_from_features). The mask is the logic-exact M (bit-exact parity).
        std::vector<double> feat = fb.build(loc.pt, bw, collected);
        std::vector<float> mask = legal_mask(env, bw, collected);
        std::vector<float> pi = std::move(ap.pi);  // the improved-policy target (the PI block)

        if (action.kind == ActionKind::Terminate) {
            // the TERMINATE decision executes no step (mirrors generate_episode): record it, break.
            eb.record_decision(std::move(feat), std::move(pi), std::move(mask),
                               /*is_terminate=*/true, /*is_collect=*/false, term_slot(env));
            break;
        }
        const bool is_collect = (action.kind == ActionKind::Treasure);
        eb.record_decision(std::move(feat), std::move(pi), std::move(mask),
                           /*is_terminate=*/false, is_collect, action_to_slot(env, action));
        StepResult sr = env.apply(loc, bw, collected, action, world);
        eb.record_step(sr.reward, sr.dt);
    }

    const double exit_c = env.exit_cost(loc.pt);
    const int nb_final = env.nb(bw);  // L6, via the seam (the belief-shrinkage stat reads this)
    return std::move(eb).finalize(exit_c, nb_final);
}

// A small splitmix64-style fold over (seed, idx) to seed each episode's RNG and pick its world. The ONE
// home (P1 — declared in runner.hpp): both run_episodes and the LOCAL batched driver seed off THIS, so
// they pick the SAME world + stream per idx. (It is NOT the Python worker's numpy seed fold — the RNGs
// differ across the language boundary by design, so cross-language parity is the ADR-0012 P6 behavioral
// bar; the harness reproduces THIS fold to match worlds episode-for-episode.)
uint64_t fold_seed(uint64_t seed, int idx) {
    uint64_t z = seed + static_cast<uint64_t>(idx) * 0x9E3779B97F4A7C15ULL;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

std::expected<int, Error> run(const Environment& env, const FeatureBuilder& fb, const Policy& policy,
                              RedisClient& redis, const RunnerConfig& cfg, std::ostream* stats_out) {
    // EXERCISE the weight-read seam (P7) even though RandomPolicy ignores the weights: this proves
    // the manifest-driven read path (parse by manifest, abort loud on missing). A missing payload is
    // a typed Error here (mirrors read_weights), the loud abort the shell reports — not a stale serve.
    auto wp = redis.read_weights(cfg.run, cfg.phase, cfg.version);
    if (!wp) return std::unexpected(wp.error());  // RandomPolicy is search-free; the read is the seam proof
    // delegate to the shared episode loop (P1). The persistent --serve loop reuses run_episodes directly,
    // reloading the net itself on a version change rather than re-reading here every generate.
    return run_episodes(env, fb, policy, redis, cfg, stats_out);
}

std::expected<int, Error> run_episodes(const Environment& env, const FeatureBuilder& fb,
                                       const Policy& policy, RedisClient& redis,
                                       const RunnerConfig& cfg, std::ostream* stats_out) {
    const std::vector<uint32_t>& worlds = env.worlds();
    int written = 0;
    for (int idx = 0; idx < cfg.episodes; ++idx) {
        std::mt19937_64 rng(fold_seed(cfg.seed, idx));
        // draw the true world for this episode uniformly from the prior world-set
        std::uniform_int_distribution<size_t> wpick(0, worlds.size() - 1);
        uint32_t world = worlds[wpick(rng)];

        EpisodeBlocks ep = run_episode(env, fb, policy, world, cfg.lam, rng, cfg.max_steps);
        if (stats_out) {
            // one JSON-object line per episode for the P6 behavioral-parity harness (additive to the
            // wire write below; does not change what crosses the wire). It carries the aggregate
            // stats AND the exact episode trace (idx, world, executed slots) so the harness can
            // replay the SAME episode in Python and value-compare the wire PI/Y/X/M bytes against an
            // INDEPENDENT computation (the wire-content parity check).
            (*stats_out) << "{\"idx\":" << idx
                         << ",\"world\":" << ep.world
                         << ",\"length\":" << ep.ep_length
                         << ",\"lam_return\":" << ep.lam_return
                         << ",\"n_collect\":" << ep.n_collect
                         << ",\"n_sense\":" << ep.n_sense
                         << ",\"n_terminate\":" << ep.n_terminate
                         << ",\"belief_shrinkage\":" << ep.belief_shrinkage
                         << ",\"exec_slots\":[";
            for (size_t s = 0; s < ep.exec_slots.size(); ++s)
                (*stats_out) << (s ? "," : "") << ep.exec_slots[s];
            (*stats_out) << "]}\n";
        }
        if (ep.n == 0) continue;  // no records (empty belief immediately) — nothing to write
        // spans bind from the EpisodeBlocks vectors (bounds-carrying, no raw pointer/len pair — P9).
        auto wr = redis.write_results(cfg.res_token, idx, ep.X, ep.n, ep.feat_dim, ep.PI, ep.M, ep.Y,
                                      ep.n_slots);
        if (!wr) return std::unexpected(wr.error());
        ++written;
    }
    return written;
}

}  // namespace chocofarm
