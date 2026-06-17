// cpp/include/chocofarm/nmcs.hpp
// Purpose: the C++ NMCS Policy — Nested Monte-Carlo Search (Cazenave, IJCAI 2009) ported behind the
//   composable env<->Policy seam, mirroring chocofarm/solvers/nmcs.py EXACTLY (ADR-0012 P7: derive
//   from the ONE authority, reimplement, behavioral parity NOT byte-identity). It is a drop-in
//   `Policy` alongside RandomPolicy: the runner takes `const Policy&` and never names this class, so
//   adding it is ZERO edits to the search/env core (the P2 seam).
//
//   This unit holds ONLY the NMCS-specific surface. The base.py-mirroring primitives it builds on —
//   GreedyBase (its default leaf base), base_value (the determinized leaf utility), candidate_actions
//   (the bounded-branching pruner), and the generic WorldSource sample_world seam — live in the
//   shared base home (policy.hpp), exactly as nmcs.py IMPORTS them from solvers.base (ADR-0012 P1:
//   one home, derive-don't-duplicate). NMCS's ONLY extension is its leaf-value method: the level-0
//   determinized playout value, added on top of the shared WorldSource (the search-specific part).
//
//   The algorithm is faithful to nmcs.py's three parts:
//     * the level-k nested recursion (`search`): walk the line forward; at each step evaluate every
//       candidate by a level-(k-1) search of its result, take the argmax (strict `>`, first-wins on
//       ties — matching Python's `if q > best_q`), play it in a determinized world, continue;
//       memorize-and-replay the best complete line's first action;
//     * the level-0 determinized base playout (`playout`): mean over `playout_samples` sampled worlds
//       of GreedyPolicy played deterministically to the end (`base_value`), scored by the
//       λ-penalized return sum(value) − λ·(travel + exit);
//     * the per-move evaluation (`eval_move`): mean over `step_samples` determinizations of
//       (immediate λ-step + the nested level-(k-1) continuation).
//   The base played at the leaf is the λ-rational GreedyBase (mirrors nmcs.py's default
//   `base=GreedyPolicy()`); candidate pruning is the shared nearest-few-detectors/treasures + always
//   TERMINATE (the shared candidate_actions with include_terminate=true).
//
//   RNG note (ADR-0012 P6): a C++ reimplementation with std::mt19937_64 does NOT match numpy's
//   stream, so aggregate parity is the behavioral bar, not byte-identity. The world-sampling seam is
//   injectable (via the shared WorldSource) so a DETERMINISTIC logic check can feed BOTH languages
//   the same leaf playout returns on fixed (loc, belief, collected) inputs and assert the SAME action
//   — the nesting + selection logic, the part that must be exact, validated independent of RNG.
//
// Public Domain (The Unlicense).
#pragma once

#include <cstdint>
#include <random>
#include <set>
#include <utility>
#include <vector>

#include "chocofarm/env.hpp"
#include "chocofarm/policy.hpp"

namespace chocofarm {

// The frozen scalar hyperparameters (mirrors solvers.nmcs.NMCSConfig — audit item I). The base
// Policy (a live object, not a scalar) stays a separate construction param, exactly as Python keeps
// `base` out of the frozen config. Defaults match NMCSConfig's.
struct NMCSConfig {
    int level = 1;            // NMCS nesting level (1 or 2 in this project's tests)
    int playout_samples = 3;  // worlds averaged per level-0 playout score (variance reduction)
    int step_samples = 2;     // worlds averaged when evaluating a candidate at a level-n step
    int cand_det = 4;         // nearest informative detectors kept as candidates
    int cand_tre = 4;         // nearest uncollected-possible treasures kept as candidates
    int max_steps = 24;       // hard cap on any search line (matches NMCSConfig.max_steps)
};

// NMCS's leaf-value extension of the shared WorldSource. The generic `sample_world` draw lives in
// the base; NMCS adds `playout_value` — the level-0 determinized base playout value at
// (loc, bw, collected) under λ (the leaf of the recursion). The production source computes the real
// mean-over-`playout_samples` GreedyPolicy playout; an injectable source (the logic-check fixture)
// returns a scripted value, so both languages run identical nesting on identical leaf returns.
struct NMCSWorldSource : public WorldSource {
    virtual double playout_value(const Loc& loc, const Belief& bw,
                                 const std::set<int>& collected, double lam) = 0;
};

// Nested Monte-Carlo Search as a pluggable Policy. Construction takes the scalar config (level etc.)
// and the level-0 base Policy (defaults to GreedyBase), mirroring NMCSPolicy.__init__.
class NMCSPolicy final : public Policy {
  public:
    explicit NMCSPolicy(const NMCSConfig& cfg = {}, const Policy* base = nullptr);

    // The Policy contract. Builds the production NMCSWorldSource off `rng` and runs the level-`level`
    // search from the current observed state, returning the FIRST action of the best line (mirrors
    // nmcs.py's decide). λ is the live Dinkelbach penalty threaded through every score (P4).
    Action decide(const Environment& env, const Loc& loc, const Belief& bw,
                  const std::set<int>& collected, double lam, std::mt19937_64& rng) const override;

    // The pure search core, parameterized by an injected NMCSWorldSource (the seam the logic check
    // exploits). Returns (score_of_best_line, first_action). Mirrors nmcs.py's _search exactly.
    std::pair<double, Action> search(const Environment& env, const Loc& loc,
                                     const Belief& bw, const std::set<int>& collected,
                                     double lam, int level, NMCSWorldSource& src) const;

    // The level-0 determinized base playout averaged over `playout_samples` sampled worlds (mirrors
    // nmcs.py's _playout): mean GreedyPolicy base_value, or −λ·exit_cost when the belief is empty.
    double playout(const Environment& env, const Loc& loc, const Belief& bw,
                   const std::set<int>& collected, double lam, NMCSWorldSource& src) const;

    const NMCSConfig& config() const { return cfg_; }

  private:
    // Per-move evaluation: mean over `step_samples` determinizations of (immediate λ-step + nested
    // level-(level-1) continuation). At level<=1 the continuation is a base playout (mirrors
    // nmcs.py's _eval_move).
    double eval_move(const Environment& env, const Loc& loc, const Belief& bw,
                     const std::set<int>& collected, const Action& a, double lam, int level,
                     NMCSWorldSource& src) const;

    NMCSConfig cfg_;
    GreedyBase default_base_;
    const Policy* base_;  // the level-0 leaf base (default_base_ unless overridden)
};

}  // namespace chocofarm
