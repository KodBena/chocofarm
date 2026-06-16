// cpp/src/policy.cpp
// Purpose: the SHARED base-unit implementations (see policy.hpp) — the C++ mirror of
//   chocofarm/solvers/base.py's search-agnostic primitives: the GreedyBase / GreedyStopBase leaf
//   policies, the candidate_actions bounded-branching pruner, and the base_value determinized leaf
//   utility. These were first consumed by NMCS but are base.py primitives, hoisted here so NMCS and
//   ISMCTS share ONE home (ADR-0012 P1: derive-don't-duplicate; the Python layout where the searches
//   import from solvers.base). Behavior-preserving move, not a rewrite.
//
// Public Domain (The Unlicense).
#include "chocofarm/policy.hpp"

#include <algorithm>

#include "chocofarm/features.hpp"

namespace chocofarm {

// ---- Policy::decide_target default — decide() + a uniform-over-legal improved-policy target ---------
// The search-free default (RandomPolicy/NMCS/ISMCTS): the executed action + a UNIFORM PI over the legal
// action set + the always-legal TERMINATE slot, 0.0 (exactly) on illegal slots (the M invariant — the
// same uniform target the runner built inline for RandomPolicy). A search policy (GumbelAZPolicy)
// overrides this with its real σ-transformed improved-π.
ActionAndPi Policy::decide_target(const Environment& env, const Loc& loc, const std::vector<uint32_t>& bw,
                                  const std::set<int>& collected, double lam,
                                  std::mt19937_64& rng) const {
    Action action = decide(env, loc, bw, collected, lam, rng);
    std::vector<float> pi(static_cast<size_t>(n_action_slots(env)), 0.0f);
    std::vector<Action> legal = env.legal_actions(bw, collected);
    const float u = 1.0f / static_cast<float>(legal.size() + 1);  // legal + TERMINATE
    for (const Action& a : legal) pi[static_cast<size_t>(action_to_slot(env, a))] = u;
    pi[static_cast<size_t>(term_slot(env))] = u;
    return ActionAndPi{action, std::move(pi)};
}

// ---- GreedyBase: the λ-rational myopic leaf base (mirrors solvers.base.GreedyPolicy) -------------
Action GreedyBase::decide(const Environment& env, const Loc& loc, const std::vector<uint32_t>& bw,
                          const std::set<int>& collected, double lam, std::mt19937_64& rng) const {
    (void)rng;  // GreedyPolicy is deterministic (Python calls it with rng=None)
    std::vector<double> marg = env.marginals(bw);
    double best = 0.0;
    Action act = terminate_action();
    for (int i = 0; i < env.N(); ++i) {
        if (collected.count(i) != 0 || marg[i] <= 0.0) continue;
        double s = marg[i] * env.value(i) - lam * env.dist(loc.pt, env.treasure_pt(i));
        if (s > best) {  // strict >: first treasure wins a tie (matches Python's `if s > best`)
            best = s;
            act = Action{ActionKind::Treasure, i};
        }
    }
    return act;
}

// ---- GreedyStopBase: the stop-cleanly greedy base (mirrors solvers.base.GreedyStopBase) ----------
// Nets the exit RELOCATION into the step value: move to the best treasure only when
// marg·value − λ·(go_there + exit(there) − exit(here)) > 0, else TERMINATE. Same init (best=0.0,
// act=TERMINATE) and strict `>` first-wins tie as GreedyPolicy. The default ISMCTS playout base.
Action GreedyStopBase::decide(const Environment& env, const Loc& loc,
                              const std::vector<uint32_t>& bw, const std::set<int>& collected,
                              double lam, std::mt19937_64& rng) const {
    (void)rng;  // deterministic (Python calls it with rng=None, mirrors _base_value)
    std::vector<double> marg = env.marginals(bw);
    double cur_exit = env.exit_cost(loc.pt);  // exit(here)
    double best = 0.0;
    Action act = terminate_action();
    for (int i = 0; i < env.N(); ++i) {
        if (collected.count(i) != 0 || marg[i] <= 0.0) continue;
        double go = env.dist(loc.pt, env.treasure_pt(i));            // go_there = d(loc, ("t", i))
        // exit(there) = exit_cost(("t", i)) = exit_cost from the treasure's coordinate.
        double net = marg[i] * env.value(i) - lam * (go + env.exit_cost(env.treasure_pt(i)) - cur_exit);
        if (net > best) {  // strict >: first treasure wins a tie (matches Python's `if net > best`)
            best = net;
            act = Action{ActionKind::Treasure, i};
        }
    }
    return act;
}

// ---- candidate_actions (mirrors solvers.base.candidate_actions) -----------------------------------
std::vector<Action> candidate_actions(const Environment& env, const Loc& loc,
                                      const std::vector<uint32_t>& bw,
                                      const std::set<int>& collected, int n_det, int n_tre,
                                      bool include_terminate) {
    std::vector<double> marg = env.marginals(bw);

    // nearest `n_det` still-informative detectors by env.dist(loc, ("d", i)); stable on face id
    // (Python's `sorted` is stable, so a distance tie keeps ascending-id order).
    std::vector<int> dets;
    for (int i = 0; i < env.n_detectors(); ++i)
        if (env.informative(i, bw)) dets.push_back(i);
    std::stable_sort(dets.begin(), dets.end(), [&](int a, int b) {
        return env.dist(loc.pt, env.face_pt(a)) < env.dist(loc.pt, env.face_pt(b));
    });
    if (static_cast<int>(dets.size()) > n_det) dets.resize(n_det);

    // nearest `n_tre` uncollected, marg>0 treasures by env.dist(loc, ("t", i)); stable on treasure id.
    std::vector<int> tres;
    for (int i = 0; i < env.N(); ++i)
        if (collected.count(i) == 0 && marg[i] > 0.0) tres.push_back(i);
    std::stable_sort(tres.begin(), tres.end(), [&](int a, int b) {
        return env.dist(loc.pt, env.treasure_pt(a)) < env.dist(loc.pt, env.treasure_pt(b));
    });
    if (static_cast<int>(tres.size()) > n_tre) tres.resize(n_tre);

    // order: detectors, then treasures, then (optionally) TERMINATE (matches candidate_actions).
    std::vector<Action> cands;
    cands.reserve(dets.size() + tres.size() + (include_terminate ? 1u : 0u));
    for (int i : dets) cands.push_back(Action{ActionKind::Detector, i});
    for (int i : tres) cands.push_back(Action{ActionKind::Treasure, i});
    if (include_terminate) cands.push_back(terminate_action());
    return cands;
}

// ---- base_value: play the base to the end in a fixed world (mirrors solvers.base._base_value) -----
double base_value(const Environment& env, const Policy& base, Loc loc, std::vector<uint32_t> bw,
                  std::set<int> collected, uint32_t world, double lam) {
    double R = 0.0, T = 0.0;
    // env.max_steps() is the single episode-horizon home (mirrors _base_value's range(env.max_steps)),
    // read from the env so a playout's horizon cannot silently desync from the Python env's.
    std::mt19937_64 unused(0);  // the base is deterministic; rng is part of the seam signature only
    for (int step = 0; step < env.max_steps(); ++step) {
        Action a = base.decide(env, loc, bw, collected, lam, unused);
        if (a.kind == ActionKind::Terminate) break;
        StepResult sr = env.apply(loc, bw, collected, a, world);
        R += sr.reward;
        T += sr.dt;
    }
    return R - lam * (T + env.exit_cost(loc.pt));
}

}  // namespace chocofarm
