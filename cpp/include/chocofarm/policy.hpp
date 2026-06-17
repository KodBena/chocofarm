// cpp/include/chocofarm/policy.hpp
// Purpose: the SHARED C++ base unit — the one home for the solvers.base.py-mirroring primitives,
//   mirroring the Python LAYOUT (chocofarm/solvers/base.py holds the shared policy primitives;
//   nmcs.py and ismcts.py each IMPORT from it). It holds the composable env<->Policy seam
//   (ADR-0012 P2: the env owns dynamics, a Policy is injected and decides) PLUS the shared,
//   search-agnostic building blocks every search reuses:
//     * Policy           — the abstract decision contract (mirrors base.Policy).
//     * RandomPolicy     — the trivial composable instance (mirrors base.RandomPolicy).
//     * GreedyBase       — the myopic λ-rational greedy base (mirrors base.GreedyPolicy).
//     * GreedyStopBase   — the stop-cleanly greedy base, default ISMCTS leaf (mirrors
//                          base.GreedyStopBase): nets the exit relocation into the step value.
//     * UCB_C            — the one home for the UCB1 exploration constant (mirrors base.UCB_C=0.7).
//     * candidate_actions— the shared bounded-branching pruner (mirrors base.candidate_actions).
//     * base_value       — play a deterministic base to the end in a fixed world (mirrors
//                          base._base_value): the leaf utility every search's playout reuses.
//     * WorldSource      — the generic world-sampling seam (the `sample_world` draw, mirrors
//                          env.sample_world / rng.choice(bw)); each search adds its OWN leaf-value
//                          extension on top, the search-specific part.
//
//   These were FIRST consumed by NMCS and so were authored inside nmcs.{hpp,cpp}, but they are not
//   NMCS-specific — they are base.py primitives. Hoisting them here (ADR-0012 P1: one home, derive-
//   don't-duplicate) lets nmcs AND ismcts include ONE shared base, exactly as the Python solvers
//   import from solvers.base; neither search re-authors a base/sampling/leaf, and ismcts does NOT
//   include nmcs.hpp to borrow them.
//
//   `lam` arrives as a live per-decision scalar (ADR-0012 P4), never baked into a policy object.
//   RNG note (ADR-0012 P6): std::mt19937_64 does NOT match numpy's stream, so parity on the
//   float-sensitive / RNG-driven aggregates is the BEHAVIORAL bar, not byte-identity.
//
// Public Domain (The Unlicense).
#pragma once

#include <cstdint>
#include <random>
#include <set>
#include <vector>

#include "chocofarm/env.hpp"

namespace chocofarm {

// The UCB1 exploration constant, held fixed across the search policies for a fair comparison — ONE
// home (mirrors solvers.base.UCB_C). A constexpr (ADR-0012 P9 modern-C++: a typed compile-time
// constant, not a #define), referenced by the search configs' defaults rather than re-typed.
inline constexpr double UCB_C = 0.7;

// One decision + its improved-policy target (the AZ training PI row): the executed action and a
// (n_slots,) probability target over the action slots, 0.0 (exactly) on illegal slots. Mirrors the
// Python search's `(executed_action, improved_pi)`.
struct ActionAndPi {
    Action action;
    std::vector<float> pi;  // (n_action_slots,) the improved-policy target; illegal-slot mass == 0.0
};

// The injected decision contract. `decide` mirrors Python's
// Policy.decide(env, loc, bw, collected, lam, rng): returns ("t", i) / ("d", i) / TERMINATE.
class Policy {
  public:
    virtual ~Policy() = default;
    virtual Action decide(const Environment& env, const Loc& loc,
                          const Belief& bw, const std::set<int>& collected,
                          double lam, std::mt19937_64& rng) const = 0;

    // Decide AND return the improved-policy target (the PI block the AZ learner trains on). The DEFAULT
    // (search-free policies — RandomPolicy/NMCS/ISMCTS) is decide() + a UNIFORM distribution over the
    // legal action set + the always-legal TERMINATE slot (the natural target for a non-search policy;
    // illegal-slot mass == 0.0). A SEARCH policy (GumbelAZPolicy) OVERRIDES this with its real
    // σ-transformed improved-π. The runner records this per decision as the PI row (mirrors Python's
    // generate_episode using the search's improved_pi). Not pure — the default is defined in policy.cpp.
    [[nodiscard]] virtual ActionAndPi decide_target(const Environment& env, const Loc& loc,
                                                    const Belief& bw,
                                                    const std::set<int>& collected, double lam,
                                                    std::mt19937_64& rng) const;
};

// Uniform over the legal action set (+ always-legal TERMINATE). Uses ONLY the env's own dynamics
// primitive (legal_actions); lam does not enter the choice. Mirrors the Python RandomPolicy's
// behavioral contract — NOT its RNG (numpy's Generator != std::mt19937_64), so parity on the
// float-sensitive / RNG-driven aggregates is the ADR-0012 P6 behavioral bar, not byte-identity.
class RandomPolicy final : public Policy {
  public:
    Action decide(const Environment& env, const Loc& loc, const Belief& bw,
                  const std::set<int>& collected, double lam, std::mt19937_64& rng) const override {
        (void)loc;  // RandomPolicy is position-blind; loc stays in the seam signature (P2 contract)
        (void)lam;  // P4: threaded through the seam, ignored by this dumb-random policy
        std::vector<Action> acts = env.legal_actions(bw, collected);
        acts.push_back(terminate_action());  // TERMINATE always legal (matches actions.term_slot)
        std::uniform_int_distribution<size_t> pick(0, acts.size() - 1);
        return acts[pick(rng)];
    }
};

// GreedyPolicy as a myopic λ-rational leaf base (mirrors solvers.base.GreedyPolicy): myopic argmax
// over the treasure with best λ-adjusted expected value marg[i]·value[i] − λ·d(loc, ("t", i)); init
// (best, act) = (0.0, TERMINATE), strict `>` so the first treasure wins a tie — the SAME selection
// rule the Python GreedyPolicy uses. Detector-blind (a deliberately weak base). λ-threaded (P4).
class GreedyBase final : public Policy {
  public:
    Action decide(const Environment& env, const Loc& loc, const Belief& bw,
                  const std::set<int>& collected, double lam, std::mt19937_64& rng) const override;
};

// GreedyStopBase: the default ISMCTS/UCT playout base (mirrors solvers.base.GreedyStopBase). Plain
// GreedyPolicy over-collects under a renewal-reward penalty (it ignores that reaching a treasure
// RELOCATES the exit); this base nets the exit relocation into the step value: move to the best
// treasure only when marg·value − λ·(go_there + exit(there) − exit(here)) > 0, else TERMINATE. Same
// init (best=0.0, act=TERMINATE) and strict `>` first-wins tie as GreedyPolicy. λ-threaded (P4).
class GreedyStopBase final : public Policy {
  public:
    Action decide(const Environment& env, const Loc& loc, const Belief& bw,
                  const std::set<int>& collected, double lam, std::mt19937_64& rng) const override;
};

// The shared bounded-branching candidate set (mirrors solvers.base.candidate_actions): nearest
// `n_det` still-informative detectors + nearest `n_tre` uncollected-possible treasures by env.dist,
// then optionally TERMINATE. Detectors and treasures are stable-sorted on distance (Python's
// `sorted` is stable, so a distance tie keeps ascending-id order). The order is detectors, then
// treasures, then (if requested) TERMINATE. Used by NMCS (include_terminate=true); a free function
// so every consumer derives the SAME pruning, not a per-search member (ADR-0012 P1).
[[nodiscard]] std::vector<Action> candidate_actions(const Environment& env, const Loc& loc,
                                                    const Belief& bw,
                                                    const std::set<int>& collected, int n_det,
                                                    int n_tre, bool include_terminate);

// Play a deterministic base policy to the end in a fixed `world`; return its λ-value
// sum(value) − λ·(travel + exit) (mirrors solvers.base._base_value). The leaf utility every search's
// determinized playout reuses — one home, so NMCS and ISMCTS score a leaf identically (P1). `bw` is
// taken BY VALUE — _base_value mutates a playout COPY of the belief in place (the seam's apply filters
// it through the run), so the copy is deliberate (a Belief value, not a borrowed ref).
[[nodiscard]] double base_value(const Environment& env, const Policy& base, Loc loc,
                                Belief bw, std::set<int> collected, uint32_t world,
                                double lam);

// The GENERIC world-sampling seam: the part of a search's RNG use that is NOT search-specific.
// Every determinized search draws a concrete world from the current belief the SAME way
// (mirrors env.sample_world / rng.choice(bw)) — to determinize a playout or a forward step. Routing
// the draw through one functor lets the production policy use the real RNG while a DETERMINISTIC
// logic check injects a SCRIPTED, RNG-free sampler so both languages run identical
// nesting/selection on identical worlds. Each search adds its OWN leaf-value method (the NMCS
// playout value, the ISMCTS expansion-index draw + scripted leaf) by deriving from this base — the
// generic draw is shared (ADR-0012 P1), only the leaf is search-specific.
struct WorldSource {
    virtual ~WorldSource() = default;
    // Sample one concrete world from the belief `b` (mirrors env.sample_world(b, rng)).
    virtual uint32_t sample_world(const Belief& b) = 0;
};

// The production world sampler: a uniform draw from the belief off a single std::mt19937_64
// (mirrors env.sample_world -> rng.choice(bw)). Searches that need only the generic draw use this
// directly; a search needing a leaf-value extension DERIVES from WorldSource and reuses this draw.
// The uniform draw now lives ON the env (env.sample_world, L1 — the seam owns the read of `.worlds`),
// so this borrows the env to route through it; the RNG stream is byte-identical (the same
// std::uniform_int_distribution<size_t>(0, nb-1)(rng) draw, just homed on the env).
class RngWorldSource : public WorldSource {
  public:
    RngWorldSource(const Environment& env, std::mt19937_64& rng) : env_(env), rng_(rng) {}
    uint32_t sample_world(const Belief& b) override { return env_.sample_world(b, rng_); }

  protected:
    const Environment& env_;  // borrowed: the home of the uniform draw (the seam, L1)
    std::mt19937_64& rng_;    // shared so a derived leaf-value source draws off the SAME stream
};

}  // namespace chocofarm
