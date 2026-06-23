// throughput-lab/cpp/boundary_per_thread.cpp
// Purpose: Topology A — BoundaryPerThread: ONE outbound DEALER socket per producer thread (today's
//   chocofarm shape, the WireLeafPool-per-thread layout). Each producer thread owns its own
//   BoundaryPerThread instance, so the socket is single-writer / single-reader by that one thread —
//   NO locks, the simplest honest impl of the Boundary seam. send() writes the matched [corr-id][payload]
//   frame to this thread's DEALER; recv()/poll() drain a reply from the same DEALER and decode it.
//   This is the transparent baseline the maintainer reads first.
// Public Domain (The Unlicense).
//
//   THREADING CONTRACT (Topology A's): each instance is touched by exactly ONE thread (the producer
//   thread that owns it). ZMQ sockets are not thread-safe; Topology A honors that structurally by giving
//   each thread its own socket. There is no shared mutable state here, hence no synchronization.
//
//   OUTSTANDING-CORR TRACKING: a small unordered_set of corr-ids this instance has sent but not yet had a
//   reply matched for. any_outstanding() reads it; recv()/poll() erase the matched corr. An UNKNOWN corr
//   on a reply is a loud error (ADR-0002 — a desynchronized wire), never a silent accept. (The set is the
//   lab's analogue of WireLeafPool's inflight_ map; the lab does not re-scatter to slots, so it tracks
//   only the id, not an ordered slot list.)

#include <algorithm>
#include <expected>
#include <optional>
#include <string>
#include <unordered_set>

#include "boundary.hpp"
#include "zmq_context.hpp"
#include "zmq_dealer.hpp"

namespace tlab {

namespace {

// Topology A boundary: one owned DEALER + the set of corr-ids awaiting a reply. Single-thread; no locks.
class BoundaryPerThread final : public Boundary {
  public:
    explicit BoundaryPerThread(ZmqDealer dealer) : dealer_(std::move(dealer)) {}

    [[nodiscard]] std::expected<void, BoundaryError> send(const LeafBatch& batch) override {
        auto sent = dealer_.send_batch(batch);
        if (!sent) return std::unexpected(sent.error());
        outstanding_.insert(batch.corr);
        return {};
    }

    // BLOCK up to RCVTIMEO for the next reply. A timeout (nullopt from the dealer) becomes a typed
    // is_timeout error here (recv()'s contract is "deliver a reply or fail"; absence-as-success is poll()'s
    // job, not recv()'s). An unknown corr is a loud desynchronization error.
    [[nodiscard]] std::expected<BoundaryReply, BoundaryError> recv() override {
        auto got = dealer_.recv_one();
        if (!got) return std::unexpected(got.error());
        if (!got->has_value())
            return std::unexpected(BoundaryError{
                "BoundaryPerThread::recv: timed out waiting for a reply (slow/absent server)", true});
        return match_and_take(std::move(**got));
    }

    // NON-BLOCKING poll: the dealer's nullopt (RCVTIMEO elapsed with nothing) maps straight to this
    // boundary's nullopt (a legitimately-absent reply), drawn apart from a transport/decode failure (the
    // error arm) per P9. NOTE: this relies on the configured recv_timeout_ms being SMALL for a true
    // non-blocking poll — see the run note in producer.cpp (the decoupled loop uses a short timeout).
    [[nodiscard]] std::expected<std::optional<BoundaryReply>, BoundaryError> poll() override {
        auto got = dealer_.recv_one();
        if (!got) return std::unexpected(got.error());
        if (!got->has_value()) return std::optional<BoundaryReply>{std::nullopt};
        auto matched = match_and_take(std::move(**got));
        if (!matched) return std::unexpected(matched.error());
        return std::optional<BoundaryReply>{std::move(*matched)};
    }

    [[nodiscard]] bool any_outstanding() const override { return !outstanding_.empty(); }

  private:
    // Erase the reply's corr from the outstanding set (loud if it was never sent — a desynchronized wire,
    // ADR-0002), then hand the reply on.
    [[nodiscard]] std::expected<BoundaryReply, BoundaryError> match_and_take(BoundaryReply reply) {
        auto it = outstanding_.find(reply.corr);
        if (it == outstanding_.end())
            return std::unexpected(BoundaryError{
                "BoundaryPerThread::recv: reply for unknown corr-id " + std::to_string(reply.corr) +
                    " (a desynchronized wire)",
                false});
        outstanding_.erase(it);
        return reply;
    }

    ZmqDealer dealer_;
    std::unordered_set<wire::corr_t> outstanding_;
};

}  // namespace

// Factory leg for Topology A (dispatched to by make_boundary in boundary_factory.cpp). Each call opens a
// FRESH DEALER on the shared context and connects it to cfg.endpoint, returning one owned boundary for
// ONE producer thread. n_producer_threads is ignored here (Topology A sizes its concurrency by how many
// times the producer calls this — one boundary per thread).
[[nodiscard]] std::expected<std::unique_ptr<Boundary>, BoundaryError> make_boundary_per_thread(
        const BoundaryConfig& cfg) {
    auto ctx = shared_zmq_context();
    if (!ctx) return std::unexpected(ctx.error());
    // The send timeout is the recv timeout with a 1s floor: a wedged wire (full send queue against a dead
    // peer) surfaces as a loud bounded send error within at most ~1s, while normal backpressure (a live
    // server drains the queue in microseconds) never trips it. Independent of the recv timeout because the
    // decoupled mode drives the recv timeout to 0, which must NOT also make sends non-blocking-fragile.
    const int send_timeout_ms = std::max(cfg.recv_timeout_ms, 1000);
    // One DEALER per producer thread, so split the TOTAL outstanding-send byte budget across the threads;
    // the byte budget -> a per-dealer message-count SNDHWM that bounds memory regardless of row size.
    const std::size_t per_dealer_bytes =
        cfg.send_queue_bytes / static_cast<std::size_t>(std::max(cfg.n_producer_threads, 1));
    const int send_hwm = send_hwm_for_budget(per_dealer_bytes, cfg.rows, cfg.in_dim);
    auto dealer = ZmqDealer::create(*ctx, cfg.endpoint, cfg.recv_timeout_ms, send_timeout_ms, send_hwm);
    if (!dealer) return std::unexpected(dealer.error());
    return std::make_unique<BoundaryPerThread>(std::move(*dealer));
}

}  // namespace tlab
