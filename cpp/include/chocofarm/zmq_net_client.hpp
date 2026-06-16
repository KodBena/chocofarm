// cpp/include/chocofarm/zmq_net_client.hpp
// Purpose: ZmqNetClient — the REMOTE NetEvaluator impl that RPCs the Shape B batched inference service
//   (docs/design/zmq-inference-service.md §1, §5, §6). The C++ twin of chocofarm/az/zmq_net_client.py:
//   a worker holds one at the leaf and calls `predict(X) -> {value, logits}`, the SAME NetEvaluator
//   port (net_evaluator.hpp) the local NetForward satisfies — so a search swaps local-for-remote with
//   zero call-site change (the zero-cost ACL, §1). The forward runs REMOTELY on the SSOT batched
//   service; this client only encodes the request, round-trips it, and decodes the NetPrediction
//   (de-standardized value + RAW logits — masking stays client-side, §2).
//
//   Transport: a blocking ZMQ_REQ socket (the lock-step request→reply peer of the server's ROUTER),
//   built on the libzmq C API (zmq.h) wrapped in small RAII types — cppzmq (zmq.hpp) is NOT a
//   dependency. The codec is the SHARED one (chocofarm/inference_wire.hpp), derived from the wire_spec
//   SSOT — no second hand-written frame (ADR-0012 P1/P7).
//
//   Failure semantics (ADR-0002 / ADR-0012 P9 rule 5 — design §5): `predict` returns
//   std::expected<NetPrediction, Error>. A receive TIMEOUT (the ctor sets ZMQ_RCVTIMEO so a
//   server-down / dropped reply becomes a loud timeout, NOT a forever-block at the leaf), a transport
//   error, or a MALFORMED reply (bad protocol byte / wrong-length frame, rejected by the codec) is a
//   TYPED failure propagated to the caller — never a silent fallback to a local net (that would mask
//   the SSOT path being down, the exact silent failure ADR-0002 forbids) and never a thrown exception.
//
//   Lifetime (P9 RAII): the zmq context + socket are owned by RAII members; the type is MOVE-ONLY (a
//   socket/context is not copyable). Construction can fail (zmq_ctx_new / zmq_socket / zmq_connect all
//   return error), so it is a static factory create() over a private ctor — a throwing ctor cannot
//   return a value (rule 5), and REQ is strict send→recv lock-step so one client is NOT thread-safe
//   (give each worker its own).
//
// Public Domain (The Unlicense).
#pragma once

#include <expected>
#include <span>
#include <string>
#include <utility>

#include "chocofarm/error.hpp"
#include "chocofarm/net_evaluator.hpp"

namespace chocofarm {

class ZmqNetClient final : public NetEvaluator {
  public:
    // Connect a blocking REQ socket to `endpoint` (e.g. "tcp://127.0.0.1:5599") with a receive timeout
    // of `recv_timeout_ms` (so a server-down becomes a loud timeout, not a hang — design §5). Returns a
    // connected client OR a typed Error if the context/socket/connect fails — never throws on a failed
    // construction (ADR-0012 P9 rule 5: a throwing ctor cannot return a value). A non-positive timeout
    // means block forever on recv (NOT recommended — the loud-failure path needs a bound).
    [[nodiscard]] static std::expected<ZmqNetClient, Error> create(const std::string& endpoint,
                                                                   int recv_timeout_ms = 5000);

    ~ZmqNetClient() override;
    ZmqNetClient(const ZmqNetClient&) = delete;
    ZmqNetClient& operator=(const ZmqNetClient&) = delete;
    ZmqNetClient(ZmqNetClient&& o) noexcept;
    ZmqNetClient& operator=(ZmqNetClient&& o) noexcept;

    // Blocking forward RPC over one feature vector `x` (length in_dim): encode → send → recv → decode →
    // {value, logits}. The value is de-standardized and the logits RAW (NOT softmaxed) — the consumer
    // masks (design §2). The NetEvaluator port override (P9). A timeout / transport failure / malformed
    // reply is a typed Error (design §5), never a silent fallback. The input is a typed bounds-carrying
    // view (a std::vector<float> binds implicitly); the result is returned by value.
    [[nodiscard]] std::expected<NetPrediction, Error> predict(std::span<const float> x) const override;

  private:
    // void* (not zmq.hpp / a typedef) so the header carries NO libzmq include — the C API types stay in
    // the .cpp. They are the zmq context + socket handles, owned and freed in the dtor (RAII). nullptr
    // marks a moved-from (empty) client.
    ZmqNetClient(void* ctx, void* sock, std::string endpoint, int recv_timeout_ms) noexcept
        : ctx_(ctx), sock_(sock), endpoint_(std::move(endpoint)), recv_timeout_ms_(recv_timeout_ms) {}

    void destroy() noexcept;   // close the socket + terminate the context (idempotent); used by dtor/move

    void* ctx_ = nullptr;
    void* sock_ = nullptr;
    std::string endpoint_;
    int recv_timeout_ms_ = 0;
};

}  // namespace chocofarm
