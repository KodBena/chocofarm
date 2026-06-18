// cpp/include/chocofarm/net_evaluator.hpp
// Purpose: the NetEvaluator PORT — the C++ leaf-evaluator boundary the search compiles against (not a
//   concrete net). The C++ twin of the Python `Net` Protocol (chocofarm/az/net_port.py) and the
//   docs/design/zmq-inference-service.md §1 zero-cost ACL: the search holds the net as an injected
//   dependency and calls only `predict(features) -> {value, logits}`; swapping the impl is a
//   construction-site choice, the search does not change a line.
//
//   Two impls satisfy it (design §1, the table):
//     * NetForward    (cpp/include/chocofarm/net.hpp) — the forward runs LOCALLY, in-process (interim;
//                      the service not up — parity, smoke). Its predict cannot fail, so it always
//                      returns a value-`expected`.
//     * ZmqNetClient  (cpp/include/chocofarm/zmq_net_client.hpp) — the forward runs REMOTELY on the
//                      Python SSOT batched service (the production path). Its predict CAN fail (a
//                      timeout / a server-down / a malformed reply), so the `expected` error arm is
//                      live there — propagated to the caller, never a silent fallback (design §5).
//
//   ADR-0012 P9 (the port shape): `predict` takes a typed, bounds-carrying input
//   (std::span<const float>) and RETURNS by value a [[nodiscard]] std::expected<NetPrediction, Error>
//   (rules 1, 2, 5) — a transport/validation failure is a typed return value the caller MUST handle,
//   never an exception (an untyped control-flow escape) and never a sentinel. This is the
//   fallible-by-contract surface the §5 failure semantics need: the remote impl's error path is
//   declared in the type the local impl shares, so the search holds ONE port whether the leaf is local
//   or remote.
//
// Public Domain (The Unlicense).
#pragma once

#include <cassert>
#include <cstddef>
#include <expected>
#include <span>
#include <utility>
#include <vector>

#include "chocofarm/error.hpp"

namespace chocofarm {

// One forward result: the de-standardized leaf value + the policy logits over the action slots (empty
// when the net is value-only — mirroring forward_core's `logits=None` / the Python (value, None)). The
// shared return type of every NetEvaluator impl (local NetForward, remote ZmqNetClient) and the wire
// `NetPrediction` contract chocofarm/az/inference_wire.py round-trips.
struct NetPrediction {
    float value = 0.0f;             // de-standardized: v_std*y_std + y_mean (the λ-penalized return scale)
    std::vector<float> logits;      // raw policy logits over n_actions slots (NOT softmaxed; empty if none)
};

// The leaf-evaluator port: a raw forward `X -> (value, logits)` over a single feature vector. The
// search compiles against THIS, not a concrete net (the zero-cost ACL, design §1). Polymorphic by
// design (called through a base reference at the leaf), so it carries a virtual destructor; the
// concrete impls are `final`.
class NetEvaluator {
  public:
    virtual ~NetEvaluator() = default;

    // Run one forward over a length-`in_dim` float32 feature vector and return the de-standardized
    // value + raw logits, OR a typed boundary failure (ADR-0012 P9 rule 5 / design §5). The local
    // NetForward always returns a value (its compute is total); the remote ZmqNetClient returns the
    // Error arm on a timeout / server-down / malformed reply. The input is a typed bounds-carrying
    // view (a std::vector<float> binds implicitly); the result is returned by value.
    [[nodiscard]] virtual std::expected<NetPrediction, Error> predict(std::span<const float> x) const = 0;
};

}  // namespace chocofarm
