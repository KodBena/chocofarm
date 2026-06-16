// cpp/src/zmq_net_client.cpp
// Purpose: the ZmqNetClient implementation — the REMOTE NetEvaluator that RPCs the Shape B inference
//   service over a blocking ZMQ_REQ socket (see zmq_net_client.hpp). Built on the libzmq C API
//   (zmq.h) wrapped in RAII; cppzmq (zmq.hpp) is NOT a dependency. The request/reply frames are the
//   SHARED codec (chocofarm/inference_wire.hpp), derived from the wire_spec SSOT (ADR-0012 P1/P7).
//
//   ADR-0012 P9 / ADR-0002 (translate-and-validate; design §5): every libzmq C error is translated
//   into a typed Error and returned via std::expected — a failed ctx/socket/connect aborts create()
//   loudly; a recv TIMEOUT (errno == EAGAIN, the ZMQ_RCVTIMEO bound firing on a server-down / dropped
//   reply) or any transport error is a typed predict() failure, NOT a hang and NOT a silent fallback;
//   a malformed reply is rejected by the codec's typed decode. No exceptions cross the boundary.
//
// Public Domain (The Unlicense).
#include "chocofarm/zmq_net_client.hpp"

#include <zmq.h>

#include <cerrno>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>

#include "chocofarm/inference_wire.hpp"

namespace chocofarm {

namespace {
// Translate the live libzmq C errno into a readable suffix (zmq_strerror covers the zmq-specific codes
// EFSM/ETERM/... plus the system errnos). Captured at the call site so the message names the cause.
std::string zmq_err() { return std::string(zmq_strerror(zmq_errno())); }
}  // namespace

std::expected<ZmqNetClient, Error> ZmqNetClient::create(const std::string& endpoint,
                                                        int recv_timeout_ms) {
    void* ctx = zmq_ctx_new();
    if (ctx == nullptr)
        return std::unexpected(make_error("chocofarm ZmqNetClient: zmq_ctx_new failed: " + zmq_err()));

    void* sock = zmq_socket(ctx, ZMQ_REQ);
    if (sock == nullptr) {
        Error e = make_error("chocofarm ZmqNetClient: zmq_socket(ZMQ_REQ) failed: " + zmq_err());
        zmq_ctx_term(ctx);   // no socket to close; reclaim the context before returning the error
        return std::unexpected(std::move(e));
    }

    // Bound the receive (ADR-0002 / design §5): a server-down or a dropped (malformed-request) reply
    // must become a loud timeout, not a forever-block at the leaf. A non-positive timeout means block
    // forever (zmq's -1 default) — left available, but the loud-failure path wants a finite bound.
    if (recv_timeout_ms >= 0) {
        if (zmq_setsockopt(sock, ZMQ_RCVTIMEO, &recv_timeout_ms, sizeof(recv_timeout_ms)) != 0) {
            Error e = make_error("chocofarm ZmqNetClient: zmq_setsockopt(ZMQ_RCVTIMEO) failed: " + zmq_err());
            zmq_close(sock);
            zmq_ctx_term(ctx);
            return std::unexpected(std::move(e));
        }
    }
    // LINGER 0: drop unsent frames immediately on close so a dead peer cannot wedge ctx termination
    // (the same discipline the Python client sets — mirrors transport.py's deadlock-avoidance).
    int linger = 0;
    if (zmq_setsockopt(sock, ZMQ_LINGER, &linger, sizeof(linger)) != 0) {
        Error e = make_error("chocofarm ZmqNetClient: zmq_setsockopt(ZMQ_LINGER) failed: " + zmq_err());
        zmq_close(sock);
        zmq_ctx_term(ctx);
        return std::unexpected(std::move(e));
    }

    if (zmq_connect(sock, endpoint.c_str()) != 0) {
        Error e = make_error("chocofarm ZmqNetClient: zmq_connect('" + endpoint + "') failed: " + zmq_err());
        zmq_close(sock);
        zmq_ctx_term(ctx);
        return std::unexpected(std::move(e));
    }

    return ZmqNetClient(ctx, sock, endpoint, recv_timeout_ms);
}

void ZmqNetClient::destroy() noexcept {
    // Close the socket BEFORE terminating the context (zmq_ctx_term blocks until all sockets close;
    // LINGER 0 means no wait on pending sends). Idempotent — a moved-from client has null handles.
    if (sock_ != nullptr) {
        zmq_close(sock_);
        sock_ = nullptr;
    }
    if (ctx_ != nullptr) {
        zmq_ctx_term(ctx_);
        ctx_ = nullptr;
    }
}

ZmqNetClient::~ZmqNetClient() { destroy(); }

ZmqNetClient::ZmqNetClient(ZmqNetClient&& o) noexcept
    : ctx_(o.ctx_), sock_(o.sock_), endpoint_(std::move(o.endpoint_)),
      recv_timeout_ms_(o.recv_timeout_ms_) {
    o.ctx_ = nullptr;   // leave the source EMPTY so its dtor frees nothing (no double-close)
    o.sock_ = nullptr;
}

ZmqNetClient& ZmqNetClient::operator=(ZmqNetClient&& o) noexcept {
    if (this != &o) {
        destroy();   // free our own handles first
        ctx_ = o.ctx_;
        sock_ = o.sock_;
        endpoint_ = std::move(o.endpoint_);
        recv_timeout_ms_ = o.recv_timeout_ms_;
        o.ctx_ = nullptr;
        o.sock_ = nullptr;
    }
    return *this;
}

std::expected<NetPrediction, Error> ZmqNetClient::predict(std::span<const float> x) const {
    if (sock_ == nullptr)
        return std::unexpected(make_error("chocofarm ZmqNetClient: predict on a moved-from/closed client"));

    // Validate the feature vector at the NEAREST boundary (ADR-0002 / Port/ACL — mirroring the Python
    // client's encode_request): an empty vector or a non-finite (NaN/Inf) entry is a malformed request
    // rejected BEFORE it touches the wire, a precise typed Error rather than a server-side drop that the
    // client would only see as a recv timeout. (The server re-validates at its own decode boundary.)
    if (x.empty())
        return std::unexpected(make_error("chocofarm ZmqNetClient: feature vector is empty (in_dim ≥ 1)"));
    for (float v : x)
        if (!std::isfinite(v))
            return std::unexpected(make_error(
                "chocofarm ZmqNetClient: feature vector has a non-finite (NaN/Inf) entry — refusing to RPC"));

    // ---- encode → send (the shared codec; the byte layout has one home — wire_spec) ----
    std::vector<unsigned char> req = wire::encode_request(x);
    int sent = zmq_send(sock_, req.data(), req.size(), 0);
    if (sent < 0)
        return std::unexpected(make_error(
            "chocofarm ZmqNetClient: zmq_send to '" + endpoint_ + "' failed: " + zmq_err() +
            " — NOT falling back to a local net (ADR-0002)"));

    // ---- recv (timeout-bounded) → decode ----
    // A zmq_msg_t recv gets the EXACT reply length (a fixed buffer would truncate an over-long frame
    // and hide drift — the codec must see the true byte count). On the ZMQ_RCVTIMEO bound firing
    // (server-down / dropped reply) zmq_msg_recv returns -1 with errno == EAGAIN — a typed timeout
    // failure, the loud non-hang path (design §5).
    zmq_msg_t msg;
    if (zmq_msg_init(&msg) != 0)
        return std::unexpected(make_error("chocofarm ZmqNetClient: zmq_msg_init failed: " + zmq_err()));

    int got = zmq_msg_recv(&msg, sock_, 0);
    if (got < 0) {
        int err = zmq_errno();
        std::string detail = std::string(zmq_strerror(err));
        zmq_msg_close(&msg);
        if (err == EAGAIN)
            return std::unexpected(make_error(
                "chocofarm ZmqNetClient: inference RPC to '" + endpoint_ + "' timed out after " +
                std::to_string(recv_timeout_ms_) + " ms (service down, overloaded, or it rejected the "
                "request) — NOT falling back to a local net (ADR-0002)"));
        return std::unexpected(make_error(
            "chocofarm ZmqNetClient: zmq_msg_recv from '" + endpoint_ + "' failed at the transport: " +
            detail));
    }

    // Copy the reply bytes out into a typed span for the codec, then release the zmq message.
    const auto* data = static_cast<const unsigned char*>(zmq_msg_data(&msg));
    std::size_t len = zmq_msg_size(&msg);
    std::vector<unsigned char> reply(data, data + len);
    zmq_msg_close(&msg);

    auto decoded = wire::decode_response(reply);
    if (!decoded)
        return std::unexpected(decoded.error());   // malformed reply: the codec's typed rejection (§5)

    NetPrediction out;
    out.value = decoded->value;
    out.logits = std::move(decoded->logits);
    return out;
}

}  // namespace chocofarm
