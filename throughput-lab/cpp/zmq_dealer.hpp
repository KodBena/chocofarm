// throughput-lab/cpp/zmq_dealer.hpp
// Purpose: ZmqDealer — the ONE home for the Layer-2 ZMQ DEALER transport this lab's boundary rides on
//   (ADR-0012 P1: a cross-boundary mechanism has one home; both topology impls derive their wire from
//   HERE, neither re-authors the corr-id framing). It owns one ZMQ_DEALER socket, sends a leaf-batch as
//   the matched multipart frame [corr-id : u64][<Layer-1 request>] (wire.hpp Layer 2 — byte-identical to
//   chocofarm's WireLeafPool::submit_batch), and receives ONE reply as [corr-id][<Layer-1 response>],
//   splitting it into the echoed corr-id (leading frame) and the value payload (last frame). It NEVER
//   computes the forward and NEVER interprets the corr-id (a transport concern it round-trips opaquely).
// Public Domain (The Unlicense).
//
//   WHY A SEPARATE HEADER (not folded into a topology .cpp): Topology A (one dealer per producer thread)
//   and Topology B (one dealer behind a coalescing thread) are TWO impls of the Boundary seam, but they
//   ride the SAME wire. Putting the socket lifecycle + the exact frame bytes in one place means the two
//   topologies cannot silently diverge on the wire (ADR-0012 P1) — they differ only in HOW MANY dealers
//   exist and WHO drains them, never in WHAT bytes cross.
//
//   This header DOES include <zmq.h> (the transport boundary lives here, exactly as chocofarm's
//   wire_leaf_pool.hpp has it); the value codec (wire.hpp) stays transport-free.
//
//   ADR-0012 P9: RAII move-only (the raw void* socket is a unique owned resource, closed in the dtor /
//   on move-from); a create() factory over a private ctor (a connect failure is the error arm, not a
//   throw — a failing ctor cannot return a value); std::span<const float> not a raw pointer+len.

#pragma once

#include <zmq.h>

#include <cstdint>
#include <cstring>
#include <expected>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "boundary.hpp"   // tlab::BoundaryError, boundary_err, BoundaryReply, LeafBatch
#include "wire.hpp"        // tlab::wire — encode_request / decode_response, corr_t, CORR_BYTES

namespace tlab {

// The owned DEALER socket + the matched corr-id framing. Move-only. Build via create().
//
// THREADING: a single ZmqDealer is single-writer / single-reader — ONE thread sends and ONE thread
// recvs (they may be the SAME thread). Topology A gives each producer thread its own ZmqDealer (so the
// owning thread both sends and recvs — fully single-thread). Topology B keeps ONE ZmqDealer behind a
// dedicated coalescing thread that is the sole caller of BOTH send_batch and recv_one. ZMQ sockets are
// NOT thread-safe, so neither topology ever touches one ZmqDealer from two threads concurrently.
class ZmqDealer final {
  public:
    // Open a ZMQ_DEALER on `zctx`, set LINGER=0, RCVTIMEO=recv_timeout_ms, and SNDTIMEO=send_timeout_ms,
    // then connect to `endpoint`. Mirrors chocofarm WireLeafPool::create (same LINGER/RCVTIMEO, same lazy
    // connect) and ADDS a bounded SNDTIMEO — the lab's discipline (ADR-0002) demands a full send queue
    // against a dead/slow server become a LOUD bounded error, not an infinite block. (chocofarm omits
    // SNDTIMEO because its server is always alive to drain; the lab must survive a missing/slow server.)
    // A null context, a socket-open failure, or a connect failure is the typed error arm (P9: a failing
    // ctor cannot return a value). NB: zmq_connect over a not-yet-bound ipc:// endpoint is LAZY and does
    // NOT fail here — a dead/absent server surfaces at the first send (SNDTIMEO) or recv (RCVTIMEO).
    //
    // WHY SNDTIMEO matters: a DEALER's send queue (ZMQ_SNDHWM, default 1000) fills when the peer does not
    // drain; the DEFAULT zmq_send then BLOCKS once the queue is full. Against a dead endpoint that block is
    // a permanent hang (the very wedge this lab must not introduce). A bounded SNDTIMEO turns it into a
    // typed is_timeout error the producer reports honestly.
    [[nodiscard]] static std::expected<ZmqDealer, BoundaryError> create(
            void* zctx, const std::string& endpoint, int recv_timeout_ms, int send_timeout_ms,
            int send_hwm) {
        if (zctx == nullptr)
            return std::unexpected(BoundaryError{"ZmqDealer::create: null zmq context", false});
        void* sock = zmq_socket(zctx, ZMQ_DEALER);
        if (sock == nullptr)
            return std::unexpected(BoundaryError{
                std::string("ZmqDealer::create: zmq_socket failed: ") + zmq_strerror(zmq_errno()), false});
        int linger = 0;
        zmq_setsockopt(sock, ZMQ_LINGER, &linger, sizeof(linger));
        // recv_timeout_ms <= 0 means "block forever" to ZMQ (a -1 RCVTIMEO). We pass the value through:
        // the boundary config documents that <= 0 blocks forever (not recommended — an absent server then
        // hangs instead of producing a loud timeout). For the standard bounded case it is a real timeout.
        zmq_setsockopt(sock, ZMQ_RCVTIMEO, &recv_timeout_ms, sizeof(recv_timeout_ms));
        // SNDTIMEO bounds a full-queue send so a wedged wire fails loudly rather than hanging (ADR-0002).
        zmq_setsockopt(sock, ZMQ_SNDTIMEO, &send_timeout_ms, sizeof(send_timeout_ms));
        // BOUND the DEALER send queue (ZMQ_SNDHWM) so a DECOUPLED free-run BACK-PRESSURES instead of
        // buffering without limit. `send_hwm` is computed by the caller from a BYTE budget and the message
        // size (boundary.hpp send_hwm_for_budget), so outstanding-send memory is capped REGARDLESS of row
        // count — an unbounded (1'000'000-deep) queue let a producer outrunning a slow / 1-core server OOM
        // at ~60 GB (this lab really did OOM-kill the producer that way). Once the queue fills, `zmq_send`
        // blocks up to SNDTIMEO; a LIVE-but-slow server drains it within that window, so the producer simply
        // throttles to the server's serve rate — achieved-rate then measures the true serving CEILING. Only
        // a genuinely DEAD peer (no drain for the full SNDTIMEO = max(recv_timeout_ms, 1000), 5 s default)
        // trips the loud is_timeout send error (ADR-0002).
        int sndhwm = send_hwm;
        zmq_setsockopt(sock, ZMQ_SNDHWM, &sndhwm, sizeof(sndhwm));
        if (zmq_connect(sock, endpoint.c_str()) != 0) {
            std::string msg = std::string("ZmqDealer::create: zmq_connect(") + endpoint +
                              ") failed: " + zmq_strerror(zmq_errno());
            zmq_close(sock);
            return std::unexpected(BoundaryError{std::move(msg), false});
        }
        return ZmqDealer(sock);
    }

    ~ZmqDealer() {
        if (sock_ != nullptr) zmq_close(sock_);
    }

    ZmqDealer(const ZmqDealer&) = delete;
    ZmqDealer& operator=(const ZmqDealer&) = delete;
    ZmqDealer(ZmqDealer&& o) noexcept : sock_(std::exchange(o.sock_, nullptr)) {}
    ZmqDealer& operator=(ZmqDealer&& o) noexcept {
        if (this != &o) {
            if (sock_ != nullptr) zmq_close(sock_);
            sock_ = std::exchange(o.sock_, nullptr);
        }
        return *this;
    }

    // Send ONE leaf-batch as [corr-id : u64 native bytes (ZMQ_SNDMORE)][<Layer-1 request>] — the exact
    // two-frame DEALER message chocofarm's WireLeafPool::submit_batch emits. The corr-id bytes are the
    // raw native-endian u64 (the server round-trips them opaquely, so endianness is irrelevant). A bad
    // batch shape (encode_request throws std::invalid_argument) or a send failure is the typed error arm
    // (ADR-0002 — never a silent partial send).
    //
    // BACKPRESSURE: ZMQ buffers a multipart message frame-by-frame and dispatches it on the FINAL frame,
    // so the SNDHWM/SNDTIMEO bite there. If the send times out (EAGAIN — the send queue is full because
    // the peer is not draining), this returns a typed is_timeout error. The producer treats ANY send
    // failure as a wedged wire and ABORTS the thread (it does not try to continue on a possibly half-sent
    // multipart) — the honest ADR-0002 response, not a silent stream-corrupting retry. The SNDHWM is
    // BOUNDED (byte-budgeted, set in create()), so a healthy DECOUPLED run that outruns the server fills
    // the queue and BLOCKS here (back-pressure) until the server drains a slot — milliseconds on a live
    // server; only a DEAD peer that never drains for the full SNDTIMEO trips the abort.
    [[nodiscard]] std::expected<void, BoundaryError> send_batch(const LeafBatch& batch) {
        std::vector<unsigned char> payload;
        try {
            payload = wire::encode_request(batch.flat, batch.B, batch.in_dim);
        } catch (const std::exception& e) {
            return std::unexpected(BoundaryError{
                std::string("ZmqDealer::send_batch: encode_request failed: ") + e.what(), false});
        }
        // frame 1: the corr-id (opaque u64, echoed back verbatim) — native bytes, ZMQ_SNDMORE.
        const wire::corr_t corr = batch.corr;
        if (zmq_send(sock_, &corr, wire::CORR_BYTES, ZMQ_SNDMORE) < 0) {
            const int err = zmq_errno();
            return std::unexpected(BoundaryError{
                std::string("ZmqDealer::send_batch: zmq_send(corr) failed: ") + zmq_strerror(err),
                err == EAGAIN});
        }
        // frame 2 (last): the batched value payload (the message dispatches here; HWM/SNDTIMEO bite here).
        if (zmq_send(sock_, payload.data(), payload.size(), 0) < 0) {
            const int err = zmq_errno();
            return std::unexpected(BoundaryError{
                std::string("ZmqDealer::send_batch: zmq_send(payload) failed (queue full / peer not "
                            "draining?): ") + zmq_strerror(err),
                err == EAGAIN});
        }
        return {};
    }

    // Receive ONE reply, blocking up to the socket RCVTIMEO. Returns:
    //   * a fully-decoded BoundaryReply (corr + decoded predictions) on success;
    //   * std::nullopt when the recv timed out with NO message available (RCVTIMEO elapsed — a legitimate
    //     "nothing yet", drawn apart from a hard failure per P9);
    //   * the error arm on a hard transport error, a malformed envelope (<2 frames or a non-8-byte
    //     leading frame), or a decode failure (ADR-0002 — a desynchronized wire is never papered over).
    // Mirrors chocofarm WireLeafPool::recv_corr_payload: frames.front() is the 8-byte corr-id, frames.back()
    // is the payload; >=2 frames required. (The lab server replies with exactly [corr-id][payload].)
    [[nodiscard]] std::expected<std::optional<BoundaryReply>, BoundaryError> recv_one() {
        std::vector<std::vector<unsigned char>> frames;
        int more = 1;
        while (more) {
            zmq_msg_t m;
            zmq_msg_init(&m);
            int rc = zmq_msg_recv(&m, sock_, 0);
            if (rc < 0) {
                int err = zmq_errno();
                zmq_msg_close(&m);
                if (err == EAGAIN) {
                    // RCVTIMEO elapsed before the FIRST frame -> nothing available yet (not a failure).
                    // This can only happen on the leading frame: once ZMQ delivers a multipart message it
                    // delivers ALL its frames atomically, so a mid-message EAGAIN cannot occur.
                    if (frames.empty()) return std::optional<BoundaryReply>{std::nullopt};
                    return std::unexpected(BoundaryError{
                        "ZmqDealer::recv_one: EAGAIN mid-multipart (impossible envelope)", false});
                }
                return std::unexpected(BoundaryError{
                    std::string("ZmqDealer::recv_one: zmq_msg_recv failed: ") + zmq_strerror(err), false});
            }
            const auto* d = static_cast<const unsigned char*>(zmq_msg_data(&m));
            frames.emplace_back(d, d + zmq_msg_size(&m));
            more = zmq_msg_more(&m);
            zmq_msg_close(&m);
        }
        if (frames.size() < 2 || frames.front().size() != wire::CORR_BYTES)
            return std::unexpected(BoundaryError{
                "ZmqDealer::recv_one: malformed reply envelope (" + std::to_string(frames.size()) +
                    " frames, leading " + std::to_string(frames.empty() ? 0 : frames.front().size()) +
                    " bytes; want >=2 frames + 8-byte corr-id)",
                false});
        BoundaryReply reply;
        std::memcpy(&reply.corr, frames.front().data(), wire::CORR_BYTES);  // opaque round-trip: native bytes
        try {
            reply.preds = wire::decode_response(frames.back());
        } catch (const std::exception& e) {
            return std::unexpected(BoundaryError{
                std::string("ZmqDealer::recv_one: decode_response failed: ") + e.what(), false});
        }
        return std::optional<BoundaryReply>{std::move(reply)};
    }

  private:
    explicit ZmqDealer(void* sock) noexcept : sock_(sock) {}
    void* sock_ = nullptr;   // the owned DEALER socket (closed in the dtor / on move-from)
};

}  // namespace tlab
