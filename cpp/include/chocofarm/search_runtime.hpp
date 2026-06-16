// cpp/include/chocofarm/search_runtime.hpp
// Purpose: the SearchRuntime seam — the swappable boundary over HOW a batch of independent Gumbel-AZ
//   decisions is scheduled, per docs/design/cpp-search-runtime.md. The durable contract is
//   {a batch of SearchTasks -> their Decisions}; the swappable mechanism beneath it is the scheduling
//   discipline (this file ships SerialRuntime; a unified work-stealing pool is the next impl) and,
//   later, the leaf transport. A new scheduling embodiment is a new SearchRuntime subclass with ZERO
//   edits to the search (gumbel.{hpp,cpp}) or to its callers (ADR-0012 P2/P3 — the same inversion of
//   control the NetEvaluator port already realizes).
//
//   *** CHUNK 1 SCOPE (this file) ***
//   SerialRuntime — one tree at a time, against the UNCHANGED GumbelAZPolicy::decide. It is (a) the
//   born-clean seam every later runtime reuses, (b) the B==1 / no-concurrency fidelity reference the
//   work-stealing pool is measured against, and (c) the driver that needs no continuation refactor, so
//   it touches nothing the just-landed 1a/1b search validated (minimal-touch). It drives the search via
//   the existing decide() entry point, so the production RNG GumbelSource (file-local to gumbel.cpp) is
//   not yet exposed — hence Decision carries the EXECUTED action + the leaf-request count now;
//   `improved_pi`/`n_spent` land when SerialRuntime drives run_search directly (a small, behaviour-
//   preserving exposure of the production GumbelSource — the documented next step, NOT forced into the
//   validated file here).
//
//   ADR-0012 P9: SearchRuntime::run takes a bounds-carrying std::span<const SearchTask> and the borrowed
//   env (factored out of the task so SearchTask stays trivially vector-storable — no reference member),
//   and RETURNS its results by value as a [[nodiscard]] std::expected<std::vector<Decision>, Error>. The
//   fallible-by-contract return is the port shape the future remote-leaf pool needs (a timed-out leaf is
//   a typed Error routed to the owning tree); SerialRuntime over a LOCAL total net always returns the
//   value arm, exactly as NetForward's total predict shares the fallible NetEvaluator port.
//
// Public Domain (The Unlicense).
#pragma once

#include <cstdint>
#include <expected>
#include <set>
#include <span>
#include <vector>

#include "chocofarm/env.hpp"
#include "chocofarm/error.hpp"
#include "chocofarm/gumbel.hpp"
#include "chocofarm/net_evaluator.hpp"

namespace chocofarm {

// One unit of work the runtime schedules: make ONE Gumbel-AZ decision for one independent problem
// instance, from this observed state, under this live λ, off this RNG seed. Each task is a fully
// independent tree (its own RNG stream, its own _Node graph — the cross-tree independence that makes a
// batched leaf row-independent, the Axis-A regime). The live per-decision scalars (λ, the budget cfg,
// the seed) ride the task (ADR-0012 P4), never baked into the runtime. The env is NOT held here — it is
// borrowed once by run() and shared by every task — so SearchTask carries no reference member and is
// trivially vector-storable behind the std::span run() consumes.
struct SearchTask {
    Loc loc;                       // observed agent location
    std::vector<uint32_t> bw;      // observed belief world-set (bitmasks over treasure ids)
    std::set<int> collected;       // treasures already collected
    double lam = 0.0;              // the live Dinkelbach penalty (per-decision, not frozen)
    std::uint64_t seed = 0;        // the per-tree RNG seed (seeds the std::mt19937_64 the source draws off)
    GumbelConfig cfg{};            // the frozen budget for this decision (m, n_sims, c_puct, ...)
};

// One decision result, returned by value. CHUNK 1: the EXECUTED action (the SH survivor at temperature
// 0) + the count of net forwards this decision issued (the structural observable — equal to the leaf
// request sequence length; a cross-check that a re-scheduling did not change the search). `improved_pi`
// (the trainer's policy target) and `n_spent` are added when the runtime drives run_search directly;
// they are deliberately absent rather than present-but-empty (no lying field).
struct Decision {
    Action executed{};             // the executed action (the SH survivor, temperature 0)
    int leaf_requests = 0;         // net forwards this decision issued (the structural sequence length)
};

// A NetEvaluator decorator that counts predict() calls and delegates to an inner evaluator unchanged.
// The leaf-request count (Decision::leaf_requests) is the structural observable the runtime reads; a
// counting wrapper keeps that instrumentation OUT of the search (the search calls net_.predict exactly
// as before — zero edits) and is reused by every runtime. predict() is const on the port, so the count
// is a mutable member (the only mutation, named and local — the imperative-shell counter, not search
// state). Borrows the inner evaluator; one counter per decision (constructed per task by the runtime).
class CountingNetEvaluator final : public NetEvaluator {
  public:
    explicit CountingNetEvaluator(const NetEvaluator& inner) : inner_(inner) {}

    [[nodiscard]] std::expected<NetPrediction, Error> predict(std::span<const float> x) const override {
        ++count_;
        return inner_.predict(x);
    }

    [[nodiscard]] int count() const { return count_; }

  private:
    const NetEvaluator& inner_;
    mutable int count_ = 0;
};

// The runtime seam: turn a batch of independent SearchTasks into their Decisions, IN INPUT ORDER, or a
// typed boundary failure. The result vector is positionally aligned with `tasks`. The search the
// runtime drives is identical across impls; only the scheduling (and later the leaf transport) differs.
// Polymorphic — held by base reference at the call site; impls are `final`.
class SearchRuntime {
  public:
    virtual ~SearchRuntime() = default;

    // Drive `tasks` to completion against the borrowed `env` and return one Decision per task, in input
    // order (P9 rules 1, 2, 5). A leaf failure on SOME task is a typed Error aborting the whole batch
    // (it never returns a partial vector with a silent hole). SerialRuntime over a local total net
    // always returns the value arm; the error arm is the contract the future remote-leaf pool needs.
    [[nodiscard]] virtual std::expected<std::vector<Decision>, Error>
    run(const Environment& env, std::span<const SearchTask> tasks) const = 0;
};

// SerialRuntime: one tree at a time, no concurrency, no leaf transport. The B==1 fidelity reference and
// the born-clean seam. It constructs, per task, a fresh CountingNetEvaluator over the held net and a
// std::mt19937_64 seeded by task.seed, then runs the UNCHANGED GumbelAZPolicy::decide — so a
// re-scheduling cannot perturb the search (there is no scheduling). Holds the net (the leaf evaluator);
// the env is borrowed per run().
class SerialRuntime final : public SearchRuntime {
  public:
    explicit SerialRuntime(const NetEvaluator& net) : net_(net) {}

    [[nodiscard]] std::expected<std::vector<Decision>, Error>
    run(const Environment& env, std::span<const SearchTask> tasks) const override;

  private:
    const NetEvaluator& net_;
};

}  // namespace chocofarm
