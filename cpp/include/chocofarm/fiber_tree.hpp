// cpp/include/chocofarm/fiber_tree.hpp
// Purpose: TreeState — ONE Gumbel-AZ tree running inside a boost.context stackful fiber, advanceable
//   leaf-by-leaf (the ONE home, ADR-0012 P1). It composes the Option-A primitives (FiberLeafChannel +
//   YieldingNetEvaluator, fiber_leaf.hpp) with a GumbelAZPolicy whose UNCHANGED run_search runs inside the
//   fiber and a CyclicGumbelSource: start() advances the search to its first parked leaf (or to finish);
//   resume_with() feeds the evaluated leaf back and advances to the next. The search core is oblivious to
//   the fiber (P9) — only WHEN predict() returns changes, not WHAT.
//
//   This is the single primitive THREE drivers multiplex/drive: the wire-parallel bench (round-synchronous
//   K trees/thread), the wire-pool bench (greedy-async T×K, corr-id routed), and the Option-A proof
//   (fiber_proto.cpp, driven against a local DetNet and asserted bit-identical to a direct run). Driving
//   the proof through THIS type means the proof validates the real shared primitive, not a proof-only copy.
//
//   Two lifetime contracts the fiber's captures impose (compile-enforced where possible, else documented):
//   (1) NEVER relocate a TreeState after start() — the entry lambda captures `this` and ynet/policy hold
//   references into it; the copy/move special members are =deleted so a relocation is a hard compile error,
//   not a silent dangle (ADR-0002). Hold it as a never-moved local or behind a unique_ptr. (2) start()
//   captures loc/bw/coll BY REFERENCE; the search re-reads them on every leaf across ALL resume_with()
//   calls — not just during start() — so the caller must keep them alive until `running` becomes false:
//   pass named lvalues, NEVER temporaries (`lam` is captured by value and is exempt). `running` is true
//   iff the fiber is parked at a leaf (vs the search having returned its Decision).
//
// Public Domain (The Unlicense).
#pragma once

#include <boost/context/fiber.hpp>

#include <set>
#include <utility>
#include <vector>

#include "chocofarm/cyclic_gumbel.hpp"
#include "chocofarm/env.hpp"
#include "chocofarm/fiber_leaf.hpp"
#include "chocofarm/gumbel.hpp"
#include "chocofarm/net_evaluator.hpp"

namespace chocofarm {

struct TreeState {
    FiberLeafChannel ch;
    YieldingNetEvaluator ynet;
    GumbelAZPolicy policy;
    CyclicGumbelSource src;
    GumbelAZPolicy::Decision decision;
    boost::context::fiber fib;
    bool running = false;

    TreeState(const GumbelConfig& cfg, const Environment& env, std::vector<double> table)
        : ynet(ch), policy(cfg, ynet, env), src(std::move(table)) {}

    // The fiber captures `this` and ynet/policy hold references INTO this object, so the only correct
    // relocation semantics are "do not relocate". Make that compile-enforced (a silent dangle becomes a
    // hard error). Copy is already implicitly deleted by the move-only `fib`; spelling all four out states
    // the intent and matches the declare-relocation-explicitly precedent (RedisClient, ZmqNetClient).
    TreeState(const TreeState&) = delete;
    TreeState& operator=(const TreeState&) = delete;
    TreeState(TreeState&&) = delete;
    TreeState& operator=(TreeState&&) = delete;

    // advance the UNCHANGED search to its first parked leaf (or to finish); `running` == parked-at-a-leaf.
    // LIFETIME: loc/bw/coll are captured BY REFERENCE into the fiber and re-read on every leaf across ALL
    // later resume_with() calls — keep them alive until `running` is false; pass named lvalues, never
    // temporaries (e.g. NOT start(loc, env.worlds(), {}, lam) — that world-set dies at the `;`).
    void start(const Loc& loc, const std::vector<uint32_t>& bw, const std::set<int>& coll, double lam) {
        fib = boost::context::fiber{
            std::allocator_arg, boost::context::fixedsize_stack(512 * 1024),
            [this, &loc, &bw, &coll, lam](boost::context::fiber&& caller) {
                ch.caller = std::move(caller);
                decision = policy.run_search(loc, bw, coll, lam, src);
                ch.at_leaf = false;  // the search returned — no more leaves
                return std::move(ch.caller);
            }};
        fib = std::move(fib).resume();  // run to the first leaf-yield (or finish)
        running = ch.at_leaf;
    }

    // feed the driver-evaluated leaf back and advance to the next leaf (or finish).
    void resume_with(const NetPrediction& pred) {
        ch.value = pred;
        fib = std::move(fib).resume();
        running = ch.at_leaf;
    }
};

}  // namespace chocofarm
