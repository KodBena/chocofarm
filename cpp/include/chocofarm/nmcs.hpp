// cpp/include/chocofarm/nmcs.hpp
// Purpose: the C++ NMCS Policy — Nested Monte-Carlo Search (Cazenave, IJCAI 2009) ported behind the
//   composable env<->Policy seam, mirroring chocofarm/solvers/nmcs.py EXACTLY (ADR-0012 P7: derive
//   from the ONE authority, reimplement, behavioral parity NOT byte-identity). It is a drop-in
//   `Policy` alongside RandomPolicy: the runner takes `const Policy&` and never names this class, so
//   adding it is ZERO edits to the search/env core (the P2 seam).
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
//   The base played at the leaf is the λ-rational GreedyPolicy (mirrors nmcs.py's default
//   `base=GreedyPolicy()`); candidate pruning is the shared nearest-few-detectors/treasures + always
//   TERMINATE (mirrors solvers.base.candidate_actions with include_terminate=True).
//
//   RNG note (ADR-0012 P6): a C++ reimplementation with std::mt19937_64 does NOT match numpy's
//   stream, so aggregate parity is the behavioral bar, not byte-identity. The world-sampling seam
//   (`WorldSource`) is injectable so a DETERMINISTIC logic check can feed BOTH languages the same
//   leaf playout returns on fixed (loc, belief, collected) inputs and assert the SAME action — the
//   nesting + selection logic, the part that must be exact, validated independent of RNG.
//
// Public Domain (The Unlicense).
#pragma once

#include <functional>
#include <random>
#include <set>
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

// The world-sampling seam. NMCS's ONLY RNG use is sampling a concrete world from the current belief
// (env.sample_world / rng.choice(bw)) — to determinize a playout or a forward step. Routing every
// such draw through one functor lets the production policy use the real RNG while the deterministic
// logic check (cpp/parity/nmcs_logic.cpp) injects a SCRIPTED leaf-value source so both languages run
// identical nesting/selection on identical leaf returns. `sample_world(bw)` returns the chosen world
// bitmask; `playout_value(loc, bw, collected, lam)` returns the level-0 leaf value (the production
// source computes the real determinized GreedyPolicy playout; a scripted source returns the next
// canned number). Production wiring builds both off a single std::mt19937_64.
struct WorldSource {
    virtual ~WorldSource() = default;
    // Sample one concrete world from the belief `bw` (mirrors env.sample_world(bw, rng)).
    virtual uint32_t sample_world(const std::vector<uint32_t>& bw) = 0;
    // The level-0 determinized base playout value at (loc, bw, collected) under λ (the leaf of the
    // recursion). The production source computes the real mean-over-`playout_samples` GreedyPolicy
    // playout; an injectable source returns a scripted value (the logic check).
    virtual double playout_value(const Loc& loc, const std::vector<uint32_t>& bw,
                                 const std::set<int>& collected, double lam) = 0;
};

// GreedyPolicy as the level-0 leaf base (mirrors solvers.base.GreedyPolicy): myopic argmax over the
// treasure with best λ-adjusted expected value marg[i]·value[i] − λ·d(loc, ("t", i)); init
// (best, act) = (0.0, TERMINATE), strict `>` so the first treasure wins a tie — the SAME selection
// rule the Python GreedyPolicy uses. Detector-blind (a deliberately weak base). λ-threaded (P4).
class GreedyBase final : public Policy {
  public:
    Action decide(const Environment& env, const Loc& loc, const std::vector<uint32_t>& bw,
                  const std::set<int>& collected, double lam, std::mt19937_64& rng) const override;
};

// Nested Monte-Carlo Search as a pluggable Policy. Construction takes the scalar config (level etc.)
// and the level-0 base Policy (defaults to GreedyBase), mirroring NMCSPolicy.__init__.
class NMCSPolicy final : public Policy {
  public:
    explicit NMCSPolicy(const NMCSConfig& cfg = {}, const Policy* base = nullptr);

    // The Policy contract. Builds the production WorldSource off `rng` and runs the level-`level`
    // search from the current observed state, returning the FIRST action of the best line (mirrors
    // nmcs.py's decide). λ is the live Dinkelbach penalty threaded through every score (P4).
    Action decide(const Environment& env, const Loc& loc, const std::vector<uint32_t>& bw,
                  const std::set<int>& collected, double lam, std::mt19937_64& rng) const override;

    // The pure search core, parameterized by an injected WorldSource (the seam the logic check
    // exploits). Returns (score_of_best_line, first_action). Mirrors nmcs.py's _search exactly.
    std::pair<double, Action> search(const Environment& env, const Loc& loc,
                                     const std::vector<uint32_t>& bw, const std::set<int>& collected,
                                     double lam, int level, WorldSource& src) const;

    // The shared bounded-branching candidate set (mirrors solvers.base.candidate_actions with
    // include_terminate=True): nearest `cand_det` informative detectors + nearest `cand_tre`
    // uncollected-possible treasures by env.d, then TERMINATE. Exposed for the logic-check fixture.
    std::vector<Action> candidates(const Environment& env, const Loc& loc,
                                   const std::vector<uint32_t>& bw,
                                   const std::set<int>& collected) const;

    // The level-0 determinized base playout averaged over `playout_samples` sampled worlds (mirrors
    // nmcs.py's _playout): mean GreedyPolicy base_value, or −λ·exit_cost when the belief is empty.
    double playout(const Environment& env, const Loc& loc, const std::vector<uint32_t>& bw,
                   const std::set<int>& collected, double lam, WorldSource& src) const;

    const NMCSConfig& config() const { return cfg_; }

  private:
    // Per-move evaluation: mean over `step_samples` determinizations of (immediate λ-step + nested
    // level-(level-1) continuation). At level<=1 the continuation is a base playout (mirrors
    // nmcs.py's _eval_move).
    double eval_move(const Environment& env, const Loc& loc, const std::vector<uint32_t>& bw,
                     const std::set<int>& collected, const Action& a, double lam, int level,
                     WorldSource& src) const;

    NMCSConfig cfg_;
    GreedyBase default_base_;
    const Policy* base_;  // the level-0 leaf base (default_base_ unless overridden)
};

// Play a deterministic base policy to the end in a fixed `world`; return its λ-value
// sum(value) − λ·(travel + exit) (mirrors solvers.base._base_value). The hot inner loop of a playout.
double base_value(const Environment& env, const Policy& base, Loc loc, std::vector<uint32_t> bw,
                  std::set<int> collected, uint32_t world, double lam);

}  // namespace chocofarm
