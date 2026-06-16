// cpp/include/chocofarm/fiber_leaf.hpp
// Purpose: the Option-A fiber-leaf primitives (the ONE home, ADR-0012 P1) — the fiber<->driver channel
//   and the YieldingNetEvaluator that let an UNCHANGED GumbelAZPolicy::run_search run inside a
//   boost.context stackful fiber, parking at each leaf instead of blocking. The search holds the
//   yielding net as its NetEvaluator and is oblivious: its predict() looks like an ordinary call
//   returning a value, but it suspends the fiber to the driver (which evaluates the leaf — locally, or
//   by batching it over a DEALER to the inference server) and resumes with the value. This is what
//   decouples "many trees in flight (a big MLP-eval batch)" from "many OS threads": one thread
//   multiplexes K fibers. (fiber_proto.cpp + wire_parallel_bench.cpp inline equivalents predating this
//   header — retrofit to it on touch.)
//
// Public Domain (The Unlicense).
#pragma once

#include <boost/context/fiber.hpp>

#include <expected>
#include <span>

#include "chocofarm/error.hpp"
#include "chocofarm/net_evaluator.hpp"

namespace chocofarm {

// The fiber<->driver channel. The yielding net writes `features` + sets `at_leaf` and yields to
// `caller`; the driver writes the evaluated `value` and resumes the fiber. `at_leaf` is false once the
// fiber's run_search returns (the decision is done).
struct FiberLeafChannel {
    boost::context::fiber caller;   // the continuation to yield back to (updated each ping-pong)
    std::span<const float> features;  // OUT: the leaf feature row predict() parked on (valid while parked)
    NetPrediction value;            // IN: the evaluated leaf the driver fed back
    bool at_leaf = false;           // OUT: predict() yielded at a leaf (vs the search finished)
};

// A NetEvaluator whose predict() does not compute — it parks the feature row and YIELDS the fiber to the
// driver, returning the driver-supplied value on resume. To the unchanged search this is just a predict()
// returning a value (the leaf evaluator port, satisfied; the suspension is invisible to the search core —
// the effect lives in the driver, P9). Returns the value arm; a real remote-leaf failure would be a typed
// Error the driver routes, but the driver feeds a value here so this returns the value arm.
class YieldingNetEvaluator final : public NetEvaluator {
  public:
    explicit YieldingNetEvaluator(FiberLeafChannel& ch) : ch_(ch) {}

    [[nodiscard]] std::expected<NetPrediction, Error> predict(std::span<const float> x) const override {
        ch_.features = x;
        ch_.at_leaf = true;
        ch_.caller = std::move(ch_.caller).resume();  // yield to the driver; resumes here when it returns
        return ch_.value;
    }

  private:
    FiberLeafChannel& ch_;
};

}  // namespace chocofarm
