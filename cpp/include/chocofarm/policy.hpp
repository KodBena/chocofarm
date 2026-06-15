// cpp/include/chocofarm/policy.hpp
// Purpose: the composable C++ Policy interface (ADR-0012 P2, mirroring the Python env<->Policy
//   seam). A Policy maps the observable state (loc, belief, collected) + the rate target lam to an
//   action; the env owns all dynamics, the policy is injected and decides. A new C++ capability —
//   a search policy, an MLP policy — is a NEW Policy subclass with ZERO edits to the env core or
//   the runner: the runner takes `const Policy&` and never names a concrete subclass.
//
//   RandomPolicy is the trivial composable instance (mirrors chocofarm/solvers/base.RandomPolicy):
//   uniform-random over the legal action set (collects + informative senses + TERMINATE). It proves
//   the seam end-to-end before any search is ported.
//
//   `lam` arrives as a live per-decision scalar (ADR-0012 P4), never baked into the policy object —
//   RandomPolicy ignores it (a dumb-random runner), but it is threaded through the signature
//   unchanged so a value-aware policy is a drop-in with no signature change.
//
// Public Domain (The Unlicense).
#pragma once

#include <random>
#include <set>
#include <vector>

#include "chocofarm/env.hpp"

namespace chocofarm {

// The injected decision contract. `decide` mirrors Python's
// Policy.decide(env, loc, bw, collected, lam, rng): returns ("t", i) / ("d", i) / TERMINATE.
class Policy {
  public:
    virtual ~Policy() = default;
    virtual Action decide(const Environment& env, const Loc& loc,
                          const std::vector<uint32_t>& bw, const std::set<int>& collected,
                          double lam, std::mt19937_64& rng) const = 0;
};

// Uniform over the legal action set (+ always-legal TERMINATE). Uses ONLY the env's own dynamics
// primitive (legal_actions); lam does not enter the choice. Mirrors the Python RandomPolicy's
// behavioral contract — NOT its RNG (numpy's Generator != std::mt19937_64), so parity on the
// float-sensitive / RNG-driven aggregates is the ADR-0012 P6 behavioral bar, not byte-identity.
class RandomPolicy final : public Policy {
  public:
    Action decide(const Environment& env, const Loc& loc, const std::vector<uint32_t>& bw,
                  const std::set<int>& collected, double lam, std::mt19937_64& rng) const override {
        (void)loc;  // RandomPolicy is position-blind; loc stays in the seam signature (P2 contract)
        (void)lam;  // P4: threaded through the seam, ignored by this dumb-random policy
        std::vector<Action> acts = env.legal_actions(bw, collected);
        acts.push_back(terminate_action());  // TERMINATE always legal (matches actions.term_slot)
        std::uniform_int_distribution<size_t> pick(0, acts.size() - 1);
        return acts[pick(rng)];
    }
};

}  // namespace chocofarm
