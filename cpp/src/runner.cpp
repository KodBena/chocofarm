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
#include <vector>

namespace chocofarm {

EpisodeBlocks run_episode(const Environment& env, const FeatureBuilder& fb, const Policy& policy,
                          uint32_t world, double lam, std::mt19937_64& rng, int max_steps) {
    const int n_slots = n_action_slots(env);
    const int feat_dim = fb.dim();

    // Live episode state (mirrors generate_episode's loc/bw/collected init: start at the entry).
    Loc loc{env.entry_point()};
    std::vector<uint32_t> bw = env.worlds();   // belief = full world-set
    std::set<int> collected;

    const size_t bw0 = bw.size();  // initial belief size (for belief-shrinkage stat)

    // per-decision records (feat, pi, mask) + the executed-step (r, dt) list for the value target.
    std::vector<std::vector<double>> feats;
    std::vector<std::vector<float>> pis, masks;
    std::vector<std::pair<double, double>> step_rt;  // (r, dt) for each EXECUTED (non-TERMINATE) step
    int n_collect = 0, n_sense = 0, n_terminate = 0;
    std::vector<int> exec_slots;  // the executed-action slot trace (for wire-content replay parity)

    for (int ply = 0; ply < max_steps; ++ply) {
        if (bw.empty()) break;  // mirrors generate_episode's len(bw)==0 break

        // decide (the env<->Policy seam): lam is a live per-decision scalar (P4).
        Action action = policy.decide(env, loc, bw, collected, lam, rng);

        // the §2.2 feature vector + the legality mask for THIS belief (mirrors generate_episode:
        // fb.build then legal_mask_from_features). The mask is the logic-exact M (bit-exact parity).
        std::vector<double> feat = fb.build(loc.pt, bw, collected);
        std::vector<float> mask = legal_mask(env, bw, collected);

        // PI: the policy's own action distribution. RandomPolicy is uniform over the legal action
        // set (the legal_actions + the always-legal TERMINATE slot) — the natural improved-policy
        // target for a search-free policy. Normalized over exactly the mask's 1-slots, 0 elsewhere
        // (so illegal-slot mass is == 0.0, the same invariant M carries).
        std::vector<float> pi(n_slots, 0.0f);
        std::vector<Action> legal = env.legal_actions(bw, collected);
        int n_choices = static_cast<int>(legal.size()) + 1;  // + TERMINATE
        float u = 1.0f / static_cast<float>(n_choices);
        for (const Action& a : legal) pi[action_to_slot(env, a)] = u;
        pi[term_slot(env)] = u;  // TERMINATE share

        if (action.kind == ActionKind::Terminate) {
            // the TERMINATE decision executes no step (mirrors generate_episode): record it, break.
            feats.push_back(std::move(feat));
            pis.push_back(std::move(pi));
            masks.push_back(std::move(mask));
            n_terminate = 1;
            exec_slots.push_back(term_slot(env));   // record the TERMINATE decision in the trace
            break;
        }
        feats.push_back(std::move(feat));
        pis.push_back(std::move(pi));
        masks.push_back(std::move(mask));
        if (action.kind == ActionKind::Treasure) ++n_collect; else ++n_sense;
        exec_slots.push_back(action_to_slot(env, action));
        StepResult sr = env.apply(loc, bw, collected, action, world);
        step_rt.emplace_back(sr.reward, sr.dt);
    }

    double exit_c = env.exit_cost(loc.pt);
    int n_dec = static_cast<int>(step_rt.size());  // executed (non-TERMINATE) decisions
    int n_rec = static_cast<int>(feats.size());    // all recorded decisions (incl. trailing TERMINATE)

    // value targets: pure-MC λ-penalized return-to-go (suffix_returns_to_go). g_j = Σ_{t>=j} r_t -
    // λ·(Σ_{t>=j} dt_t + exit_c). The exit toll is in every suffix (charged once at episode end).
    std::vector<double> g_steps(n_dec, 0.0);
    double suffix_r = 0.0, suffix_t = 0.0;
    for (int j = n_dec - 1; j >= 0; --j) {
        suffix_r += step_rt[j].first;
        suffix_t += step_rt[j].second;
        g_steps[j] = suffix_r - lam * (suffix_t + exit_c);
    }

    EpisodeBlocks out;
    out.n = n_rec;
    out.feat_dim = feat_dim;
    out.n_slots = n_slots;
    // ---- aggregate stats (P6 behavioral parity) ----
    out.ep_length = n_dec;
    // the full-episode λ-return = the return-to-go from the FIRST executed decision; with no
    // executed step the episode is a bare exit, value = the exit toll −λ·exit_c.
    out.lam_return = (n_dec > 0) ? g_steps[0] : (-lam * exit_c);
    out.n_collect = n_collect;
    out.n_sense = n_sense;
    out.n_terminate = n_terminate;
    out.belief_shrinkage = (bw0 > 0) ? (1.0 - static_cast<double>(bw.size()) /
                                              static_cast<double>(bw0)) : 0.0;
    out.world = world;
    out.exec_slots = std::move(exec_slots);
    out.X.resize(static_cast<size_t>(n_rec) * feat_dim);
    out.PI.resize(static_cast<size_t>(n_rec) * n_slots);
    out.M.resize(static_cast<size_t>(n_rec) * n_slots);
    out.Y.resize(n_rec);
    for (int j = 0; j < n_rec; ++j) {
        for (int c = 0; c < feat_dim; ++c)
            out.X[static_cast<size_t>(j) * feat_dim + c] = static_cast<float>(feats[j][c]);
        for (int c = 0; c < n_slots; ++c) {
            out.PI[static_cast<size_t>(j) * n_slots + c] = pis[j][c];
            out.M[static_cast<size_t>(j) * n_slots + c] = masks[j][c];
        }
        // the trailing TERMINATE decision (j >= n_dec) has no step: its target is the exit toll
        // continuation, -λ·exit_c (mirrors generate_episode's terminal-decision g).
        double g = (j < n_dec) ? g_steps[j] : (-lam * exit_c);
        out.Y[j] = static_cast<float>(g);
    }
    return out;
}

// A small splitmix64-style fold over (seed, idx) to seed each episode's RNG and pick its world.
// (This is the C++ runner's OWN per-episode seeding; it is NOT the Python worker's numpy seed fold —
// the RNGs differ across the language boundary by design, so parity is the ADR-0012 P6 behavioral
// bar, not byte-identity. The harness reproduces THIS fold to match worlds episode-for-episode.)
static uint64_t fold_seed(uint64_t seed, int idx) {
    uint64_t z = seed + static_cast<uint64_t>(idx) * 0x9E3779B97F4A7C15ULL;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

int run(const Environment& env, const FeatureBuilder& fb, const Policy& policy,
        RedisClient& redis, const RunnerConfig& cfg, std::ostream* stats_out) {
    // EXERCISE the weight-read seam (P7) even though RandomPolicy ignores the weights: this proves
    // the manifest-driven read path (parse by manifest, abort loud on missing). A missing payload
    // throws std::runtime_error here (mirrors read_weights), which is the loud abort, not a stale serve.
    WeightPayload wp = redis.read_weights(cfg.run, cfg.phase, cfg.version);
    (void)wp;  // RandomPolicy is search-free; the read is the seam proof, not a consumer

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
        redis.write_results(cfg.res_token, idx, ep.X, ep.n, ep.feat_dim, ep.PI, ep.M, ep.Y,
                            ep.n_slots);
        ++written;
    }
    return written;
}

}  // namespace chocofarm
