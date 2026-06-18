// cpp/include/chocofarm/wire_leaf_pool.hpp
// Purpose: WireLeafPool — the reusable PER-THREAD DEALER leaf-resolver lifted out of
//   wire_pool_bench.cpp's worker lambda (the ONE home for the corr-id transport, ADR-0012 P1). It owns
//   one ZMQ DEALER socket over which K parked tree-fibers' leaf forwards are multiplexed to the batched
//   JAX InferenceServer: submit() stamps a globally-unique u64 correlation id (a SHARED process-global
//   atomic, passed by reference), carries it as the LEADING zmq frame ahead of wire::encode_request(X)
//   ([corr-id][payload]), and tracks corr-id -> slot; poll() blocks up to the socket RCVTIMEO for ONE
//   reply, validates the envelope, decodes via wire::decode_response, and routes the reply to ITS slot
//   by the echoed corr-id (the server round-trips that leading frame OPAQUELY — frames[1:-1] in
//   inference_server.py — so the corr-id is a TRANSPORT-envelope concern that NEVER enters the value
//   codec, ADR-0012 P7 serialization⊥transport).
//
//   FAIL LOUD (ADR-0002): a recv error, a malformed envelope (<2 frames or a non-8-byte leading frame),
//   a decode failure, or an UNKNOWN corr-id is a typed std::unexpected — NEVER a silent wrong-slot apply
//   and NEVER a zero/stale leaf substitution. The driver propagates that to a whole-pass abort.
//
//   SCOPE (CRITIQUE C1, honestly marked): WireLeafPool is PER-THREAD — its `inflight_` map is single-
//   thread state. The corr-id atomic is process-GLOBAL (passed by reference) so that a FUTURE shared
//   tree-registry / work-stealing layer could key on it to route ANY reply to ANY tree regardless of
//   which thread submitted its leaf — but that cross-thread migration is NOT built here (ADR-0009: not
//   before the measure says T×K composition helps). Today a slot is single-writer-per-thread, structural.
//
//   ADR-0012 P9: RAII, move-only (the raw `void* sock_` is a unique owning resource, closed in the dtor /
//   on move-from), a create() factory over a private ctor (a throwing/failing ctor cannot return a value
//   — the connect failure is the create() error arm), std::span<const float> not a raw pointer/len pair.
//   This header DOES depend on <zmq.h> (the transport boundary lives here, exactly as wire_pool_bench
//   had it); the value codec (inference_wire.hpp) stays transport-free.
//
// Public Domain (The Unlicense).
#pragma once

#include <zmq.h>

#include <atomic>
#include <cstdint>
#include <cstring>
#include <expected>
#include <span>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "chocofarm/error.hpp"
#include "chocofarm/inference_wire.hpp"
#include "chocofarm/net_evaluator.hpp"

namespace chocofarm {

// One resolved leaf: the slot whose outstanding leaf this reply answers + the decoded NetPrediction.
struct Completion {
    int slot = -1;
    NetPrediction pred;
};

// A per-thread DEALER leaf-resolver. Move-only (it owns the socket); construct via create().
class WireLeafPool final {
  public:
    // Open a DEALER on `zctx`, set LINGER=0 + RCVTIMEO=timeout_ms, connect to `endpoint`. A connect
    // failure is the typed error arm (a throwing/failing ctor cannot return a value — ADR-0012 P9). The
    // `corr_seq` atomic is borrowed by reference and outlives the pool (the driver owns it, process-
    // global so corr-ids are unique across ALL pools/threads). NB (CRITIQUE D2): zmq_connect over a
    // not-yet-bound ipc:// endpoint is LAZY and does NOT fail here — a dead endpoint surfaces only at the
    // first poll() recv after timeout_ms, as a loud recv-timeout error, not a hang.
    [[nodiscard]] static std::expected<WireLeafPool, Error> create(void* zctx,
                                                                   const std::string& endpoint,
                                                                   int timeout_ms,
                                                                   std::atomic<uint64_t>& corr_seq) {
        if (zctx == nullptr)
            return std::unexpected(make_error("WireLeafPool::create: null zmq context"));
        void* sock = zmq_socket(zctx, ZMQ_DEALER);
        if (sock == nullptr)
            return std::unexpected(make_error(std::string("WireLeafPool::create: zmq_socket failed: ") +
                                              zmq_strerror(zmq_errno())));
        int linger = 0;
        zmq_setsockopt(sock, ZMQ_LINGER, &linger, sizeof(linger));
        zmq_setsockopt(sock, ZMQ_RCVTIMEO, &timeout_ms, sizeof(timeout_ms));
        if (zmq_connect(sock, endpoint.c_str()) != 0) {
            std::string msg = std::string("WireLeafPool::create: zmq_connect(") + endpoint +
                              ") failed: " + zmq_strerror(zmq_errno());
            zmq_close(sock);
            return std::unexpected(make_error(std::move(msg)));
        }
        return WireLeafPool(sock, corr_seq);
    }

    ~WireLeafPool() {
        if (sock_ != nullptr) zmq_close(sock_);
    }

    WireLeafPool(const WireLeafPool&) = delete;
    WireLeafPool& operator=(const WireLeafPool&) = delete;
    WireLeafPool(WireLeafPool&& o) noexcept
        : sock_(std::exchange(o.sock_, nullptr)),
          corr_seq_(o.corr_seq_),
          inflight_(std::move(o.inflight_)) {}
    WireLeafPool& operator=(WireLeafPool&& o) noexcept {
        if (this != &o) {
            if (sock_ != nullptr) zmq_close(sock_);
            sock_ = std::exchange(o.sock_, nullptr);
            corr_seq_ = o.corr_seq_;
            inflight_ = std::move(o.inflight_);
        }
        return *this;
    }

    // Submit slot `slot`'s outstanding leaf: stamp a unique corr-id, send [corr-id][encode_request(X)],
    // and record corr-id -> slot. A send failure (the socket died) is the typed error arm (ADR-0002).
    // `features` is a bounds-carrying view valid for the duration of this call (the fiber's parked row).
    [[nodiscard]] std::expected<void, Error> submit(int slot, std::span<const float> features) {
        std::vector<unsigned char> req = wire::encode_request(features);
        const uint64_t corr = corr_seq_->fetch_add(1, std::memory_order_relaxed);
        // frame 1: the corr-id (opaque u64, echoed back verbatim). frame 2: the value payload.
        if (zmq_send(sock_, &corr, sizeof(corr), ZMQ_SNDMORE) < 0)
            return std::unexpected(make_error(std::string("WireLeafPool::submit: zmq_send(corr) failed: ") +
                                              zmq_strerror(zmq_errno())));
        if (zmq_send(sock_, req.data(), req.size(), 0) < 0)
            return std::unexpected(make_error(std::string("WireLeafPool::submit: zmq_send(payload) failed: ") +
                                              zmq_strerror(zmq_errno())));
        inflight_.emplace(corr, slot);
        return {};
    }

    // Block up to the socket RCVTIMEO for ONE reply, decode it, and route it to its slot by the echoed
    // corr-id. A recv error/timeout, a malformed envelope (<2 frames or a non-8-byte leading frame), a
    // decode failure, or an UNKNOWN corr-id is the loud error arm (ADR-0002): the wire is desynchronized
    // and the driver MUST abort the whole pass — never a silent wrong-slot apply, never a zero/stale leaf.
    [[nodiscard]] std::expected<Completion, Error> poll() {
        uint64_t corr = 0;
        std::vector<unsigned char> payload;
        auto rcv = recv_corr_payload(corr, payload);
        if (!rcv) return std::unexpected(rcv.error());
        auto decoded = wire::decode_response(payload);
        if (!decoded)
            return std::unexpected(make_error("WireLeafPool::poll: malformed response payload: " +
                                              decoded.error().message));
        auto it = inflight_.find(corr);
        if (it == inflight_.end())
            return std::unexpected(make_error("WireLeafPool::poll: unknown correlation id " +
                                              std::to_string(corr) + " (a desynchronized wire)"));
        Completion c;
        c.slot = it->second;
        inflight_.erase(it);
        c.pred.value = decoded->value;
        c.pred.logits = std::move(decoded->logits);
        return c;
    }

    // True iff at least one submitted leaf has not yet been resolved by a poll().
    [[nodiscard]] bool any_outstanding() const { return !inflight_.empty(); }

  private:
    WireLeafPool(void* sock, std::atomic<uint64_t>& corr_seq) noexcept
        : sock_(sock), corr_seq_(&corr_seq) {}

    // Receive ONE reply and split it into its echoed correlation id (the LEADING frame — an opaque u64
    // the server round-tripped) and the response payload (the LAST frame). The error arm is taken on a
    // recv error/timeout or a malformed envelope (<2 frames, or a leading frame that is not 8 bytes) —
    // ADR-0002: a desynchronized wire is never silently papered over. (Lifted verbatim in logic from
    // wire_pool_bench.cpp's recv_corr_payload; expressed as a typed std::expected here.)
    [[nodiscard]] std::expected<void, Error> recv_corr_payload(uint64_t& corr,
                                                               std::vector<unsigned char>& payload) {
        std::vector<std::vector<unsigned char>> frames;
        int more = 1;
        while (more) {
            zmq_msg_t m;
            zmq_msg_init(&m);
            if (zmq_msg_recv(&m, sock_, 0) < 0) {
                std::string err = zmq_strerror(zmq_errno());
                zmq_msg_close(&m);
                return std::unexpected(make_error("WireLeafPool::poll: zmq_msg_recv failed: " + err));
            }
            const auto* d = static_cast<const unsigned char*>(zmq_msg_data(&m));
            frames.emplace_back(d, d + zmq_msg_size(&m));
            more = zmq_msg_more(&m);
            zmq_msg_close(&m);
        }
        if (frames.size() < 2 || frames.front().size() != sizeof(uint64_t))
            return std::unexpected(make_error("WireLeafPool::poll: malformed reply envelope (" +
                                              std::to_string(frames.size()) + " frames, leading " +
                                              std::to_string(frames.empty() ? 0 : frames.front().size()) +
                                              " bytes; want >=2 frames + 8-byte corr-id)"));
        std::memcpy(&corr, frames.front().data(), sizeof(uint64_t));  // opaque round-trip: native bytes
        payload = std::move(frames.back());
        return {};
    }

    void* sock_ = nullptr;                            // the owned DEALER socket (closed in dtor / on move)
    std::atomic<uint64_t>* corr_seq_ = nullptr;       // borrowed process-global corr-id source (P1)
    std::unordered_map<uint64_t, int> inflight_;      // corr-id -> slot of its outstanding leaf (per-thread)
};

}  // namespace chocofarm
