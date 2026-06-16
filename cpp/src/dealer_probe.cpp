// cpp/src/dealer_probe.cpp
// Purpose: verify the non-blocking DEALER transport (the wire-PARALLEL leaf rendezvous foundation, NOT
//   the runner) — connect a ZMQ_DEALER to the Python Shape-B ROUTER server, submit N feature-vector
//   requests WITHOUT waiting (many outstanding at once — the property the blocking REQ lacks), then
//   receive N replies and decode them. This proves (a) the DEALER<->ROUTER multipart framing against the
//   server's [identity][...][payload] expectation (the server reads frames[-1] as the payload), and (b)
//   that the server's greedy drain BATCHES the concurrently-submitted requests into one forward — the
//   throughput lever the wire-parallel fiber multiplexer exploits. Positional FIFO: with one DEALER peer
//   submitting all N before receiving, the server drains them in arrival order and replies in order, so
//   the i-th reply is the i-th request (no echoed id needed for this barrier pattern — that is only for
//   the continuous-async case, docs/design/cpp-search-runtime.md §4.1).
//
//   Reuses the SHARED inference_wire codec (P7 — no re-authored frame). The codec stays the same; only
//   the socket type and the submit/recv discipline differ from the blocking ZmqNetClient.
//
//   Protocol:  dealer-probe --endpoint <tcp://host:port> [--n N --in-dim D --timeout-ms N]
//   Output:    "RESULT: PASS got=<n>/<N> all-finite first_value=<v>" + exit 0, or a loud failure.
//
// Public Domain (The Unlicense).
#include <zmq.h>

#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "chocofarm/inference_wire.hpp"

namespace {
[[nodiscard]] std::optional<std::string_view> opt(std::span<const std::string_view> args,
                                                  std::string_view name) {
    for (size_t i = 1; i + 1 < args.size(); ++i)
        if (args[i] == name) return args[i + 1];
    return std::nullopt;
}
[[nodiscard]] int to_int(std::string_view s) { return std::atoi(std::string(s).c_str()); }

// Receive ONE full multipart message and return its LAST frame (the payload): the ROUTER replies
// [empty][payload] to a DEALER, so the last frame is the response bytes. Returns empty on error/timeout.
[[nodiscard]] std::vector<unsigned char> recv_payload(void* sock, bool& ok) {
    std::vector<unsigned char> last;
    ok = false;
    int more = 1;
    while (more) {
        zmq_msg_t m;
        zmq_msg_init(&m);
        int n = zmq_msg_recv(&m, sock, 0);
        if (n < 0) {
            zmq_msg_close(&m);
            return {};
        }
        const auto* d = static_cast<const unsigned char*>(zmq_msg_data(&m));
        last.assign(d, d + zmq_msg_size(&m));
        more = zmq_msg_more(&m);
        zmq_msg_close(&m);
    }
    ok = true;
    return last;
}
}  // namespace

int main(int argc, char** argv) {
    std::vector<std::string_view> args(argv, argv + argc);
    std::optional<std::string_view> endpoint = opt(args, "--endpoint");
    if (!endpoint) {
        std::cerr << "usage: dealer-probe --endpoint <tcp://host:port> [--n N --in-dim D --timeout-ms N]\n";
        return 2;
    }
    const int n = opt(args, "--n") ? to_int(*opt(args, "--n")) : 16;
    const int in_dim = opt(args, "--in-dim") ? to_int(*opt(args, "--in-dim")) : 241;
    const int timeout_ms = opt(args, "--timeout-ms") ? to_int(*opt(args, "--timeout-ms")) : 10000;

    void* ctx = zmq_ctx_new();
    void* sock = zmq_socket(ctx, ZMQ_DEALER);
    int linger = 0;
    zmq_setsockopt(sock, ZMQ_LINGER, &linger, sizeof(linger));
    zmq_setsockopt(sock, ZMQ_RCVTIMEO, &timeout_ms, sizeof(timeout_ms));
    if (zmq_connect(sock, std::string(*endpoint).c_str()) != 0) {
        std::cerr << "dealer-probe: FATAL: connect failed: " << zmq_strerror(zmq_errno()) << "\n";
        return 1;
    }

    // submit N requests WITHOUT waiting for replies (many outstanding — the DEALER property).
    for (int i = 0; i < n; ++i) {
        std::vector<float> x(static_cast<size_t>(in_dim));
        for (int j = 0; j < in_dim; ++j)
            x[static_cast<size_t>(j)] = 0.01f * static_cast<float>((i * 7 + j) % 13 - 6);
        std::vector<unsigned char> req = chocofarm::wire::encode_request(x);
        // send as ONE frame; the ROUTER prepends identity -> [identity][req], server reads frames[-1]=req.
        if (zmq_send(sock, req.data(), req.size(), 0) < 0) {
            std::cerr << "dealer-probe: FATAL: send " << i << " failed: " << zmq_strerror(zmq_errno())
                      << "\n";
            return 1;
        }
    }

    // receive N replies (positional FIFO: the i-th reply is the i-th request).
    int got = 0;
    bool all_finite = true;
    float first_value = 0.0f;
    for (int i = 0; i < n; ++i) {
        bool ok = false;
        std::vector<unsigned char> payload = recv_payload(sock, ok);
        if (!ok) {
            std::cerr << "dealer-probe: recv " << i << " timed out/failed: " << zmq_strerror(zmq_errno())
                      << "\n";
            break;
        }
        auto decoded = chocofarm::wire::decode_response(payload);
        if (!decoded) {
            std::cerr << "dealer-probe: decode " << i << " failed: " << decoded.error().message << "\n";
            break;
        }
        if (i == 0) first_value = decoded->value;
        if (!std::isfinite(decoded->value)) all_finite = false;
        for (float l : decoded->logits)
            if (!std::isfinite(l)) all_finite = false;
        ++got;
    }

    zmq_close(sock);
    zmq_ctx_term(ctx);

    if (got == n && all_finite) {
        std::cout << "RESULT: PASS got=" << got << "/" << n << " all-finite first_value=" << first_value
                  << "\n";
        return 0;
    }
    std::cout << "RESULT: FAIL got=" << got << "/" << n << " all_finite=" << all_finite << "\n";
    return 3;
}
