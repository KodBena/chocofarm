// throughput-lab/cpp/boundary_coalescing.cpp
// Purpose: Topology B — BoundaryCoalescing: the producer threads feed ONE separate COALESCING THREAD that
//   holds the ONLY DEALER socket and MERGES their concurrent submissions into one wire frame before
//   sending. This is the topology that asks "does pushing the producer threads' batches through one
//   coalescing point (one socket, larger frames) beat one-socket-per-thread?". One BoundaryCoalescing is
//   SHARED by all producer threads; send() is the thread-safe handoff to the coalescing thread's intake
//   queue; recv()/poll() read THIS calling thread's reply mailbox.
// Public Domain (The Unlicense).
//
//   THE COALESCING THREAD (the sole owner of the DEALER — ZMQ sockets are not thread-safe, so exactly one
//   thread ever touches it):
//     intake  : producer threads push (producer-corr, owning-thread mailbox, COPIED rows, B, in_dim) onto
//               a mutex+condvar queue. The rows are COPIED because LeafBatch.flat is only valid for the
//               duration of the send() call (the seam's contract) — the coalescing thread outlives it.
//     coalesce: drain up to all currently-queued submissions, concatenate their rows row-major into ONE
//               (sum B, in_dim) matrix, stamp ONE fresh WIRE corr-id W, send [W][encode_request(...)],
//               and record W -> the ordered list of (owning mailbox, producer-corr, B) so the reply can
//               be split back. One coalesced wire message == one server forward over (sum B) rows.
//     scatter : on a reply for W, split the (sum B) predictions by the recorded per-submission B's, rebuild
//               each producer's BoundaryReply{corr = its producer-corr, preds = its slice}, and push it
//               into THAT producer's reply mailbox (keyed by the submitting thread). FAIL LOUD (ADR-0002):
//               an unknown W, a prediction-count mismatch, or a transport/decode error aborts the thread
//               and is surfaced to every waiting producer (the shared error latch below).
//
//   WHY in_dim must agree across coalesced submissions: a single wire frame carries ONE in_dim header, so
//   only rows of the SAME width can ride one frame. The lab runs a single in_dim (Stage-A 241) across all
//   threads, so this always holds; a mismatched submission is a loud error rather than a silent reshape.
//
//   ROUTING BY SUBMITTING THREAD: the Boundary seam's recv()/poll() carry no thread index, so each producer
//   thread is identified by std::this_thread::get_id(); on its first send() it lazily registers a reply
//   MAILBOX, and its replies are routed there. recv()/poll() read the calling thread's mailbox. This keeps
//   the seam unchanged (P8) while letting one shared boundary fan replies back to the right thread.

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <expected>
#include <iterator>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "boundary.hpp"
#include "proc_domains.hpp"   // tlab::ByteCount / HwmMessages / Milliseconds (the coalescing-thread shape)
#include "zmq_context.hpp"
#include "zmq_dealer.hpp"

namespace tlab {

namespace {

// ---- OutstandingCount: sent-but-not-yet-replied tally (per mailbox + the wire-frame tally) --------------
// A non-negative count of in-flight submissions/frames. Minted HERE (its single home — the two outstanding
// tallies are the only sites) rather than in the process-shape SSOT, because it is a coalescing-thread-local
// bookkeeping quantity. u64 preserves the original std::uint64_t width (a long-running run's cumulative
// in-flight peak comfortably fits, and the width must not narrow under behaviour preservation). Additive so
// the ++/-- read as count arithmetic; the decrement is GUARDED (never wraps below zero — ADR-0002).
struct OutstandingCountTag {};
using OutstandingCount = chocofarm::Quantity<OutstandingCountTag, std::uint64_t>;

}  // namespace
}  // namespace tlab

namespace chocofarm {
// Opt the (TU-local) OutstandingCount tag into additive (a count + a count is a count — the increment) AND
// affine (count - raw 1 -> count — the GUARDED decrement; the machinery offers no Q-=Q, so the dec is the
// affine Q - Rep crossing). Both are the meaningful tally ops; neither is the cross-domain mix it forbids.
template <> struct quantity_additive<tlab::OutstandingCountTag> : std::true_type {};
template <> struct quantity_affine<tlab::OutstandingCountTag> : std::true_type {};
}  // namespace chocofarm

namespace tlab {
namespace {

// One producer thread's reply MAILBOX: a queue of completed BoundaryReplies the coalescing thread routes
// here, plus the synchronization the owning thread blocks on in recv(). One per producer thread (keyed by
// thread id). The shared_ptr lets the coalescing thread hold a stable handle even as the map mutates.
struct Mailbox {
    std::mutex mu;
    std::condition_variable cv;
    std::deque<BoundaryReply> replies;
    OutstandingCount outstanding{0};   // sent-but-not-yet-replied for THIS thread (for any_outstanding)
};

// One producer submission waiting to be coalesced: the rows (COPIED — the seam's flat is transient), the
// producer's own corr-id (echoed back in its BoundaryReply), and the mailbox to route the reply to.
struct Submission {
    wire::ProducerCorr producer_corr{0};     // the producer's own corr-id (echoed back in its BoundaryReply)
    wire::RowCount B{0};
    wire::FeatureDim in_dim{0};
    std::vector<float> rows;                 // B*in_dim floats, row-major (owned copy)
    std::shared_ptr<Mailbox> mailbox;        // where this submission's reply is delivered
};

// What the coalescing thread records for one OUTGOING wire message, so it can split the reply: the ordered
// list of (mailbox, producer-corr, B) for the submissions packed into this frame.
struct PackedPart {
    std::shared_ptr<Mailbox> mailbox;
    wire::ProducerCorr producer_corr{0};
    wire::RowCount B{0};
};

class BoundaryCoalescing final : public Boundary {
  public:
    BoundaryCoalescing(ZmqDealer dealer, Milliseconds recv_timeout_ms, HwmMessages intake_cap)
        : dealer_(std::move(dealer)),
          recv_timeout_ms_(recv_timeout_ms),
          // intake_cap is a byte-budgeted message count (>= 4 by send_hwm_for_budget); the > 0 guard is kept
          // as a defensive floor (behaviour-preserving) in case a future caller hands a zero.
          intake_cap_(intake_cap.value() > 0 ? intake_cap : HwmMessages{1}),
          coalescer_([this] { coalesce_loop(); }) {}

    ~BoundaryCoalescing() override {
        // Signal the coalescing thread to stop and join it (RAII shutdown). Outstanding replies in flight
        // are abandoned at teardown — the producer is expected to drain before destroying the boundary.
        {
            std::lock_guard<std::mutex> lk(intake_mu_);
            stop_ = true;
            intake_closed_ = true;   // wake any producer blocked in send() back-pressure so it bails
        }
        intake_cv_.notify_all();
        intake_space_cv_.notify_all();
        if (coalescer_.joinable()) coalescer_.join();
    }

    // THREAD-SAFE handoff: copy the rows, attach the calling thread's mailbox, enqueue for the coalescing
    // thread. A prior fatal error on the coalescing thread (a dead socket / desynchronized wire) is
    // surfaced here so the producer stops feeding a broken wire (ADR-0002).
    [[nodiscard]] std::expected<void, BoundaryError> send(const LeafBatch& batch) override {
        if (auto err = fatal_error()) return std::unexpected(*err);
        auto mailbox = mailbox_for_this_thread();
        Submission sub;
        sub.producer_corr = batch.corr;
        sub.B = batch.B;
        sub.in_dim = batch.in_dim;
        sub.rows.assign(batch.flat.begin(), batch.flat.end());   // COPY (flat is transient per the seam)
        sub.mailbox = mailbox;
        {
            std::lock_guard<std::mutex> mlk(mailbox->mu);
            mailbox->outstanding += OutstandingCount{1};
        }
        {
            std::unique_lock<std::mutex> lk(intake_mu_);
            // BACK-PRESSURE: block until the coalescing thread has drained the intake below its cap, so a
            // DECOUPLED free-run cannot pile unbounded COPIES of rows here. This intake is the SECOND
            // unbounded buffer that OOM'd the producer — the DEALER SNDHWM only bounds the WIRE queue
            // DOWNSTREAM of the coalescing thread; back-pressure there just pushes the backlog up into this
            // queue. The cap is byte-budgeted (send_hwm_for_budget), so intake memory is bounded regardless
            // of rows. A live coalescing thread drains in milliseconds; only stop/fatal ends the wait early.
            intake_space_cv_.wait(lk, [this] {
                // ACL: the deque .size() (size_t) is compared against the message-count cap via .value().
                return intake_.size() < static_cast<std::size_t>(intake_cap_.value()) || intake_closed_;
            });
            if (intake_closed_) {
                lk.unlock();
                if (auto err = fatal_error()) return std::unexpected(*err);
                return std::unexpected(BoundaryError{"BoundaryCoalescing::send: boundary shutting down", false});
            }
            intake_.push_back(std::move(sub));
        }
        intake_cv_.notify_one();
        return {};
    }

    // BLOCK up to recv_timeout_ms for THIS thread's next reply. A coalescing-thread fatal error is raised
    // here (so a waiting producer does not hang on a broken wire); a genuine timeout is a typed is_timeout.
    [[nodiscard]] std::expected<BoundaryReply, BoundaryError> recv() override {
        auto mailbox = mailbox_for_this_thread();
        std::unique_lock<std::mutex> lk(mailbox->mu);
        // ACL: Milliseconds -> the chrono duration's rep at the deadline computation. recv_timeout_ms_ is
        // unsigned, so the legacy "<= 0 = block forever" is the .value() == 0 case (handled below).
        const auto deadline = std::chrono::steady_clock::now() +
                              std::chrono::milliseconds(recv_timeout_ms_.value());
        for (;;) {
            if (!mailbox->replies.empty()) {
                BoundaryReply r = std::move(mailbox->replies.front());
                mailbox->replies.pop_front();
                return r;
            }
            if (auto err = fatal_error()) return std::unexpected(*err);
            if (recv_timeout_ms_.value() == 0) {
                mailbox->cv.wait(lk);   // block forever (config opted out of a bound — not recommended)
            } else {
                if (mailbox->cv.wait_until(lk, deadline) == std::cv_status::timeout &&
                    mailbox->replies.empty()) {
                    if (auto err = fatal_error()) return std::unexpected(*err);
                    return std::unexpected(BoundaryError{
                        "BoundaryCoalescing::recv: timed out waiting for a reply (slow/absent server)", true});
                }
            }
        }
    }

    // NON-BLOCKING poll of THIS thread's mailbox: a reply if one is queued, nullopt if none yet (drawn
    // apart from a fatal error, which is the error arm — P9).
    [[nodiscard]] std::expected<std::optional<BoundaryReply>, BoundaryError> poll() override {
        if (auto err = fatal_error()) return std::unexpected(*err);
        auto mailbox = mailbox_for_this_thread();
        std::lock_guard<std::mutex> lk(mailbox->mu);
        if (mailbox->replies.empty()) return std::optional<BoundaryReply>{std::nullopt};
        BoundaryReply r = std::move(mailbox->replies.front());
        mailbox->replies.pop_front();
        return std::optional<BoundaryReply>{std::move(r)};
    }

    // True iff THIS calling thread has a submission still awaiting its reply. (Per-thread, matching the
    // per-thread mailbox; a coupled producer asks about its own outstanding batch.)
    [[nodiscard]] bool any_outstanding() const override {
        auto mailbox = mailbox_for_this_thread();
        std::lock_guard<std::mutex> lk(mailbox->mu);
        return mailbox->outstanding > OutstandingCount{0};
    }

  private:
    // ---- per-thread mailbox registry (lazy: a thread gets a mailbox on its first send/recv/poll) -------
    std::shared_ptr<Mailbox> mailbox_for_this_thread() const {
        const std::thread::id tid = std::this_thread::get_id();
        std::lock_guard<std::mutex> lk(mailboxes_mu_);
        auto it = mailboxes_.find(tid);
        if (it != mailboxes_.end()) return it->second;
        auto mb = std::make_shared<Mailbox>();
        mailboxes_.emplace(tid, mb);
        return mb;
    }

    // ---- the shared fatal-error latch (set once by the coalescing thread; read by producers) -----------
    void set_fatal_error(const BoundaryError& e) {
        {
            std::lock_guard<std::mutex> lk(fatal_mu_);
            if (!fatal_) fatal_ = e;   // first error wins (the originating cause); keep it stable
        }
        // Wake every producer that might be blocked in recv() so they observe the error and stop.
        {
            std::lock_guard<std::mutex> mlk(mailboxes_mu_);
            for (auto& [tid, mb] : mailboxes_) {
                std::lock_guard<std::mutex> block(mb->mu);
                mb->cv.notify_all();
            }
        }
        // Wake every producer blocked in send() back-pressure so it observes the dead wire and stops
        // feeding it (otherwise, with the coalescing thread gone, the intake never drains -> a hang).
        {
            std::lock_guard<std::mutex> lk(intake_mu_);
            intake_closed_ = true;
        }
        intake_space_cv_.notify_all();
    }
    [[nodiscard]] std::optional<BoundaryError> fatal_error() const {
        std::lock_guard<std::mutex> lk(fatal_mu_);
        return fatal_;
    }

    // ---- the coalescing thread's loop --------------------------------------------------------------
    void coalesce_loop() {
        // Bound each blocking phase so the loop interleaves "send what's queued" with "drain replies" and
        // notices stop_ promptly. A short poll bound (a few ms) keeps both directions live without a busy
        // spin. The DEALER's own RCVTIMEO (set at create from recv_timeout_ms) bounds the recv leg.
        while (!should_stop()) {
            // PHASE 1 — gather all currently-queued submissions (block briefly if none, to avoid a spin).
            std::vector<Submission> batch = drain_intake();
            if (!batch.empty()) {
                if (auto err = send_coalesced(batch); err) {
                    set_fatal_error(*err);
                    return;
                }
            }
            // PHASE 2 — drain ALL replies that have landed right now and scatter them, so reply collection
            // keeps pace with a high send rate (not one reply per outer iteration). recv_one() with the
            // boundary's RCVTIMEO returns nullopt the instant nothing more is queued; that ends the inner
            // drain and we loop back to PHASE 1. (In decoupled mode RCVTIMEO=0, so this is a non-blocking
            // spin-drain; in coupled mode it blocks up to RCVTIMEO for the first reply, which is fine since
            // the coupled producer is itself waiting on that round-trip.)
            while (outstanding_wire_ > OutstandingCount{0}) {
                auto got = dealer_.recv_one();
                if (!got) {
                    set_fatal_error(got.error());
                    return;
                }
                if (!got->has_value()) break;   // nothing more ready -> back to PHASE 1
                if (auto err = scatter_reply(std::move(**got)); err) {
                    set_fatal_error(*err);
                    return;
                }
            }
        }
    }

    [[nodiscard]] bool should_stop() {
        std::lock_guard<std::mutex> lk(intake_mu_);
        // Stop once asked AND the intake is drained. We do NOT also wait for outstanding_wire_ == 0: by the
        // time the destructor sets stop_, the producer threads have already joined and run their own tail-
        // drain (collecting every reply they could under their deadline), so any STILL-outstanding wire
        // reply is one the server never returned. Waiting on it here would HANG the destructor's join()
        // against a slow/dead server (a teardown wedge — the very failure ADR-0002 forbids). So we make a
        // bounded best-effort: finish sending what's queued, then stop, abandoning replies the server never
        // sent. (The producer already has its honest sent/recv counts; the gap is reported, not hidden.)
        return stop_ && intake_.empty();
    }

    // Block up to a short bound for at least one submission, then take EVERYTHING currently queued (the
    // coalescing: many producer submissions -> one wire frame). Returns empty on a timeout with nothing
    // queued (the loop then goes to drain replies).
    [[nodiscard]] std::vector<Submission> drain_intake() {
        std::unique_lock<std::mutex> lk(intake_mu_);
        if (intake_.empty()) {
            // Short wait so PHASE 2 (reply drain) still runs promptly when nothing is being produced.
            // ACL: Milliseconds -> the chrono duration's rep at the bounded wait.
            intake_cv_.wait_for(lk, std::chrono::milliseconds(kIntakeWaitMs.value()),
                                [this] { return stop_ || !intake_.empty(); });
        }
        std::vector<Submission> out;
        out.reserve(intake_.size());
        while (!intake_.empty()) {
            out.push_back(std::move(intake_.front()));
            intake_.pop_front();
        }
        // Room freed -> wake producers blocked in send() back-pressure (the intake is now empty).
        if (!out.empty()) intake_space_cv_.notify_all();
        return out;
    }

    // Concatenate the batch's rows into one matrix, stamp one wire corr-id, send it, and record the split.
    [[nodiscard]] std::optional<BoundaryError> send_coalesced(std::vector<Submission>& batch) {
        // All coalesced rows must share in_dim (one frame carries one in_dim header). Validate loudly.
        const wire::FeatureDim in_dim = batch.front().in_dim;
        wire::RowCount total_B{0};
        for (const auto& s : batch) {
            if (s.in_dim != in_dim)
                return BoundaryError{"BoundaryCoalescing: coalesced submissions disagree on in_dim (" +
                                         std::to_string(s.in_dim.value()) + " vs " +
                                         std::to_string(in_dim.value()) + ")",
                                     false};
            total_B += s.B;   // RowCount additive: a sum of row counts is a row count
        }
        std::vector<float> flat;
        // ACL: the count->element-extent crossing (total_B * in_dim) unwraps both typed counts into the
        // size_t reserve arithmetic at the allocation site.
        flat.reserve(static_cast<std::size_t>(total_B.value()) * in_dim.value());
        std::vector<PackedPart> parts;
        parts.reserve(batch.size());
        for (auto& s : batch) {
            flat.insert(flat.end(), s.rows.begin(), s.rows.end());
            parts.push_back(PackedPart{std::move(s.mailbox), s.producer_corr, s.B});
        }
        const wire::WireCorr wire_corr = next_wire_corr_;
        next_wire_corr_ = next_wire_corr_ + wire::corr_t{1};   // affine ++ (the named monotonic generation)
        LeafBatch lb;
        // ACL (the two-corr-namespace fusion): the coalescing thread stamps a WIRE corr, but LeafBatch.corr
        // is the seam's ProducerCorr slot — so the wire corr crosses into it via .value() + the explicit
        // ProducerCorr ctor. The bytes are identical (both ride corr_t); the dealer round-trips them opaquely
        // and recv_one hands the echoed corr back as a ProducerCorr, which scatter_reply re-reads as the wire
        // corr (the inverse crossing). The packed_ map keeps the authoritative WireCorr key.
        lb.corr = wire::ProducerCorr{wire_corr.value()};
        lb.B = total_B;
        lb.in_dim = in_dim;
        lb.flat = std::span<const float>(flat.data(), flat.size());
        auto sent = dealer_.send_batch(lb);
        if (!sent) return sent.error();
        packed_.emplace(wire_corr.value(), std::move(parts));   // ACL: WireCorr -> raw corr_t map key
        outstanding_wire_ += OutstandingCount{1};
        return std::nullopt;
    }

    // Split one wire reply's predictions back to the producer mailboxes per the recorded parts.
    [[nodiscard]] std::optional<BoundaryError> scatter_reply(BoundaryReply reply) {
        // ACL (the inverse of send_coalesced's stamp): recv_one handed the echoed corr back as a ProducerCorr
        // (the dealer is corr-namespace-agnostic), but on THIS leg it is the WIRE corr the coalescing thread
        // stamped. Re-read it as a WireCorr, then unwrap to the raw corr_t the packed_ map is keyed by.
        const wire::WireCorr wire_corr{reply.corr.value()};
        auto it = packed_.find(wire_corr.value());
        if (it == packed_.end())
            return BoundaryError{"BoundaryCoalescing: reply for unknown wire corr-id " +
                                     std::to_string(wire_corr.value()) + " (a desynchronized wire)",
                                 false};
        std::vector<PackedPart> parts = std::move(it->second);
        packed_.erase(it);
        // GUARDED decrement (OutstandingCount is non-negative; never wrap below zero — ADR-0002). An entry was
        // just found in packed_, so this is always > 0 here, but the guard makes the invariant structural.
        // GUARDED decrement via the affine Q - Rep crossing (the machinery has no Q-=Q for an additive tag).
        if (outstanding_wire_ > OutstandingCount{0})
            outstanding_wire_ = outstanding_wire_ - OutstandingCount::rep_type{1};

        std::size_t expected = 0;
        for (const auto& p : parts) expected += p.B.value();   // ACL: RowCount -> size_t prediction-count sum
        if (reply.preds.size() != expected)
            return BoundaryError{"BoundaryCoalescing: wire reply carried " +
                                     std::to_string(reply.preds.size()) + " predictions for " +
                                     std::to_string(expected) + " coalesced rows (a desynchronized wire)",
                                 false};

        // Walk the parts in pack order, slicing the prediction vector and delivering each producer's slice
        // to its mailbox under that producer's OWN corr-id (so its recv() sees the corr it stamped).
        std::size_t off = 0;
        for (auto& part : parts) {
            BoundaryReply out;
            out.corr = part.producer_corr;   // the producer's OWN corr (so its recv() sees the corr it stamped)
            // ACL: RowCount -> size_t slice width at the prediction-vector iterator arithmetic.
            const std::size_t part_B = part.B.value();
            out.preds.assign(std::make_move_iterator(reply.preds.begin() + off),
                             std::make_move_iterator(reply.preds.begin() + off + part_B));
            off += part_B;
            {
                std::lock_guard<std::mutex> lk(part.mailbox->mu);
                part.mailbox->replies.push_back(std::move(out));
                // GUARDED decrement (non-negative; never wrap below zero, ADR-0002) via the affine Q - Rep.
                if (part.mailbox->outstanding > OutstandingCount{0})
                    part.mailbox->outstanding =
                        part.mailbox->outstanding - OutstandingCount::rep_type{1};
            }
            part.mailbox->cv.notify_one();
        }
        return std::nullopt;
    }

    // How long the coalescing thread blocks for new intake before falling through to drain replies. A few
    // ms keeps both directions live without a busy spin; it does NOT cap throughput (the moment a batch is
    // queued the wait returns immediately via the predicate).
    static constexpr Milliseconds kIntakeWaitMs{2};

    ZmqDealer dealer_;
    Milliseconds recv_timeout_ms_{5000};   // mirrors the dealer's RCVTIMEO; bounds producer recv() too

    // ---- intake (producer threads -> coalescing thread), BYTE-BUDGET-BOUNDED for back-pressure ----
    std::mutex intake_mu_;
    std::condition_variable intake_cv_;          // coalescing thread waits here for work
    std::condition_variable intake_space_cv_;    // producers wait here for room (the back-pressure signal)
    std::deque<Submission> intake_;
    HwmMessages intake_cap_{1};                    // max queued submissions (= byte budget / per-message size)
    bool intake_closed_ = false;                  // stop/fatal -> wake blocked producers so they bail, not hang
    bool stop_ = false;

    // ---- coalescing-thread-private state (touched only by the coalescing thread) ----
    wire::WireCorr next_wire_corr_{1};            // the coalescing thread's monotonic WIRE corr generation
    std::unordered_map<wire::corr_t, std::vector<PackedPart>> packed_;   // wire corr (raw key) -> its split plan
    OutstandingCount outstanding_wire_{0};                               // outgoing frames awaiting a reply

    // ---- per-thread mailboxes (the reply fan-out) ----
    mutable std::mutex mailboxes_mu_;
    mutable std::unordered_map<std::thread::id, std::shared_ptr<Mailbox>> mailboxes_;

    // ---- the fatal-error latch ----
    mutable std::mutex fatal_mu_;
    std::optional<BoundaryError> fatal_;

    // The coalescing thread (started last so all members above are constructed before it runs).
    std::thread coalescer_;
};

}  // namespace

// Factory leg for Topology B. ONE BoundaryCoalescing is created (the producer threads SHARE it). It opens
// the single DEALER on the shared context and spawns its coalescing thread. n_producer_threads is intake
// sizing only (the impl handles any count via the per-thread mailbox registry); the boundary itself is one.
[[nodiscard]] std::expected<std::unique_ptr<Boundary>, BoundaryError> make_boundary_coalescing(
        const BoundaryConfig& cfg) {
    auto ctx = shared_zmq_context();
    if (!ctx) return std::unexpected(ctx.error());
    // Send timeout = recv timeout with a 1s floor (see boundary_per_thread.cpp for the rationale): a wedged
    // wire surfaces loudly within ~1s; normal backpressure against a live server never trips it.
    const Milliseconds send_timeout_ms{std::max(cfg.recv_timeout_ms.value(), 1000u)};
    // ONE shared DEALER for all threads -> the whole outstanding-send byte budget is its queue cap; the
    // byte budget -> a message-count SNDHWM that bounds memory regardless of row size.
    const HwmMessages send_hwm = send_hwm_for_budget(cfg.send_queue_bytes, cfg.rows, cfg.in_dim);
    auto dealer = ZmqDealer::create(*ctx, cfg.endpoint, OptMilliseconds{cfg.recv_timeout_ms},
                                    OptMilliseconds{send_timeout_ms}, send_hwm);
    if (!dealer) return std::unexpected(dealer.error());
    // The intake queue (producer threads -> coalescing thread) gets the SAME byte-budget cap as the wire
    // send queue, so BOTH buffers in the coalescing path are bounded (total outstanding ~2x the budget).
    return std::make_unique<BoundaryCoalescing>(std::move(*dealer), cfg.recv_timeout_ms, send_hwm);
}

}  // namespace tlab
