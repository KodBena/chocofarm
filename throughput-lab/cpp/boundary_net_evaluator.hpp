// throughput-lab/cpp/boundary_net_evaluator.hpp
// Purpose: the ACL bridging the chocofarm search's NetEvaluator PORT to OUR lab Boundary — the seam that
//   lets the REAL generator (chocofarm::SearchRuntime, calling predict() per leaf) drive load through the
//   throughput-lab's near-optimal STATIC transport, instead of the parent's old wire client. predict(x)
//   ships ONE leaf (B=1) through tlab::Boundary (a send -> recv round-trip) and returns the decoded
//   value+logits as a chocofarm::NetPrediction. This is the COUPLED B=1 shape — the NON-FIBER baseline a
//   per-thread SerialRuntime drives; the fiber multiplexer keeps K leaves in flight per thread instead
//   (and routes the parked leaves through the same Boundary).
//
//   THREADING: one BoundaryNetEvaluator per producer thread, each owning its own Boundary (a tlab DEALER
//   is single-writer, per boundary.hpp's Topology-A discipline). predict() is `const` on the port, but a
//   reference member's referent is not const-qualified by the enclosing const method, so the non-const
//   send/recv are legal; the corr counter is the one `mutable` member (the imperative-shell counter, not
//   search state — mirroring CountingNetEvaluator's mutable count_). Not thread-safe by itself; the
//   per-thread ownership is what makes that safe.
//
//   ADR-0012 P9: a transport failure is a TYPED chocofarm::Error on the expected error arm (the port's
//   fallible contract — the same arm ZmqNetClient's remote predict uses), never a thrown escape.
// Public Domain (The Unlicense).
#pragma once

#include <expected>
#include <span>
#include <utility>

#include "chocofarm/error.hpp"
#include "chocofarm/net_evaluator.hpp"

#include "boundary.hpp"
#include "wire.hpp"

namespace tlab {

class BoundaryNetEvaluator final : public chocofarm::NetEvaluator {
  public:
    explicit BoundaryNetEvaluator(Boundary& boundary) : boundary_(boundary) {}

    [[nodiscard]] std::expected<chocofarm::NetPrediction, chocofarm::Error> predict(
        std::span<const float> x) const override {
        const wire::corr_t corr = next_corr_++;
        const auto in_dim = static_cast<wire::count_t>(x.size());
        const LeafBatch lb{corr, /*B=*/1, in_dim, x};
        if (auto sent = boundary_.send(lb); !sent)
            return std::unexpected(chocofarm::make_error(
                "BoundaryNetEvaluator: send failed: " + sent.error().message));
        auto reply = boundary_.recv();
        if (!reply)
            return std::unexpected(chocofarm::make_error(
                "BoundaryNetEvaluator: recv failed: " + reply.error().message));
        // B=1 COUPLED is strict request->reply on one thread's DEALER, so the reply is THIS corr's; a
        // mismatch or empty payload is a loud transport/decode fault, never a silent wrong-leaf value.
        if (reply->corr != corr || reply->preds.empty())
            return std::unexpected(chocofarm::make_error(
                "BoundaryNetEvaluator: corr mismatch or empty reply (expected corr=" +
                std::to_string(corr) + ", got corr=" + std::to_string(reply->corr) +
                ", preds=" + std::to_string(reply->preds.size()) + ")"));
        chocofarm::NetPrediction pred;
        pred.value = reply->preds[0].value;
        pred.logits = std::move(reply->preds[0].logits);
        return pred;
    }

  private:
    Boundary& boundary_;
    mutable wire::corr_t next_corr_ = 1;
};

}  // namespace tlab
