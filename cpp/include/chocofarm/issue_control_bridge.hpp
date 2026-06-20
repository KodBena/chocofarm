// cpp/include/chocofarm/issue_control_bridge.hpp
// Purpose: the C++ side of the ONLINE ISSUE CONTROL LOOP — the ZeroMQ bridge between the in-process
//   actuation hub (IssueController) and the external Python POLICY ENGINE. A dedicated control thread,
//   on a slow cadence, snapshots the hub's marshalled features, SENDS them over a ZMQ REQ socket to the
//   engine, RECEIVES the per-thread issue-allow bits back, and writes them into the hub's actuation cells.
//   The runner's refill() reads those cells (one relaxed atomic) on the hot path — untouched by this.
//
//   WHY ZMQ + BINARY (not redis-key-polling, not JSON). The control loop is COORDINATION/STREAMING, which
//   ADR-0012 P7 assigns to a MESSAGING FABRIC (ZeroMQ), NOT a bytes-store used as a sync primitive (the P7
//   smell). And the control PATH is latency/jitter sensitive — its realtime behaviour feeds back into the
//   policy's own prediction quality — against a per-batch forward of a handful of microseconds, so the wire
//   is a PACKED BINARY frame (the codec below is its ONE authoritative P7 definition; the Python engine
//   derives the same layout, with a magic + length runtime parity check as the floor). FAIL LOUD (ADR-0002):
//   a recv timeout / malformed reply stops the bridge with an error the harness checks — never a silent hang
//   nor a stale gate.
// Public Domain (The Unlicense).
#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <zmq.h>

#include "chocofarm/issue_controller.hpp"

namespace chocofarm {

// ---- THE CONTROL WIRE (the one authoritative P7 layout; Python issue_engine derives it) ------------------
// All little-endian, packed field-by-field (no struct padding); the host is x86_64 (LE), asserted by the
// magic check. FEATURES frame (C++ -> engine):
//   u32 magic=FEAT_MAGIC | u32 T | u32 D | f64 server_rows_per_forward | T*{ i32 inflight; i32 ready;
//   i64 msgs; i64 leaves; i64 rtt_us }
// GATES frame (engine -> C++):  u32 magic=GATE_MAGIC | u32 T | T*{ u8 allow }
inline constexpr std::uint32_t FEAT_MAGIC = 0x15C0F1A1u;
inline constexpr std::uint32_t GATE_MAGIC = 0x15C0F1A2u;

class IssueControlBridge {
public:
    // ctl: the actuation hub (snapshot_features / set_allow); endpoint: the ZMQ control socket the Python
    // engine binds (e.g. ipc:///tmp/...); cadence_ms: the control-tick period; timeout_ms: the engine-reply
    // deadline (fail-loud past it).
    IssueControlBridge(IssueController* ctl, std::string endpoint, double cadence_ms, int timeout_ms = 2000)
        : ctl_(ctl), endpoint_(std::move(endpoint)),
          cadence_ms_(cadence_ms > 0.0 ? cadence_ms : 5.0), timeout_ms_(timeout_ms) {}
    ~IssueControlBridge() { stop(); }
    IssueControlBridge(const IssueControlBridge&) = delete;
    IssueControlBridge& operator=(const IssueControlBridge&) = delete;

    [[nodiscard]] bool failed() const { return failed_.load(std::memory_order_relaxed); }
    [[nodiscard]] const std::string& error() const { return error_; }

    void start() {
        if (running_ || ctl_ == nullptr) return;
        running_ = true;
        stop_.store(false, std::memory_order_relaxed);
        thread_ = std::thread([this] { run_(); });
    }

    void stop() {
        if (!running_) return;
        { std::lock_guard<std::mutex> lk(mu_); stop_.store(true, std::memory_order_relaxed); }
        cv_.notify_all();
        if (thread_.joinable()) thread_.join();
        running_ = false;
    }

private:
    static void put_u32(std::vector<char>& b, std::uint32_t v) {
        const size_t o = b.size(); b.resize(o + 4); std::memcpy(b.data() + o, &v, 4);
    }
    static void put_i32(std::vector<char>& b, std::int32_t v) {
        const size_t o = b.size(); b.resize(o + 4); std::memcpy(b.data() + o, &v, 4);
    }
    static void put_i64(std::vector<char>& b, std::int64_t v) {
        const size_t o = b.size(); b.resize(o + 8); std::memcpy(b.data() + o, &v, 8);
    }
    static void put_f64(std::vector<char>& b, double v) {
        const size_t o = b.size(); b.resize(o + 8); std::memcpy(b.data() + o, &v, 8);
    }

    void fail_(std::string msg) {
        std::lock_guard<std::mutex> lk(mu_);
        if (!failed_.load(std::memory_order_relaxed)) { error_ = std::move(msg); failed_.store(true, std::memory_order_relaxed); }
        stop_.store(true, std::memory_order_relaxed);
    }

    void run_() {
        void* ctx = zmq_ctx_new();
        if (ctx == nullptr) { fail_("issue-bridge: zmq_ctx_new failed"); return; }
        void* sock = zmq_socket(ctx, ZMQ_REQ);
        if (sock == nullptr) { zmq_ctx_term(ctx); fail_("issue-bridge: zmq_socket failed"); return; }
        const int rcvto = timeout_ms_, sndto = timeout_ms_, linger = 0;
        zmq_setsockopt(sock, ZMQ_RCVTIMEO, &rcvto, sizeof(rcvto));
        zmq_setsockopt(sock, ZMQ_SNDTIMEO, &sndto, sizeof(sndto));
        zmq_setsockopt(sock, ZMQ_LINGER, &linger, sizeof(linger));
        if (zmq_connect(sock, endpoint_.c_str()) != 0) {
            zmq_close(sock); zmq_ctx_term(ctx); fail_("issue-bridge: zmq_connect failed: " + endpoint_); return;
        }
        const int T = ctl_->n_threads();
        std::vector<char> fbuf;
        std::vector<unsigned char> gbuf(static_cast<size_t>(8 + T));
        std::unique_lock<std::mutex> lk(mu_);
        while (!stop_.load(std::memory_order_relaxed)) {
            cv_.wait_for(lk, std::chrono::duration<double, std::milli>(cadence_ms_),
                         [this] { return stop_.load(std::memory_order_relaxed); });
            if (stop_.load(std::memory_order_relaxed)) break;
            lk.unlock();

            // ---- snapshot -> pack FEATURES ----
            const IssueFeatures f = ctl_->snapshot_features();
            fbuf.clear();
            put_u32(fbuf, FEAT_MAGIC);
            put_u32(fbuf, static_cast<std::uint32_t>(T));
            put_u32(fbuf, static_cast<std::uint32_t>(ctl_->d_ceiling()));
            put_f64(fbuf, f.server_rows_per_forward);
            for (int t = 0; t < T; ++t) {
                const size_t i = static_cast<size_t>(t);
                put_i32(fbuf, static_cast<std::int32_t>(f.inflight[i]));
                put_i32(fbuf, static_cast<std::int32_t>(f.ready[i]));
                put_i64(fbuf, static_cast<std::int64_t>(f.msgs[i]));
                put_i64(fbuf, static_cast<std::int64_t>(f.leaves[i]));
                put_i64(fbuf, static_cast<std::int64_t>(f.mean_rtt_ms[i] * 1000.0));
            }
            if (zmq_send(sock, fbuf.data(), fbuf.size(), 0) < 0) { fail_("issue-bridge: zmq_send (features) failed/timeout"); lk.lock(); break; }

            // ---- recv GATES -> set_allow ----
            const int n = zmq_recv(sock, gbuf.data(), gbuf.size(), 0);
            if (n < 0) { fail_("issue-bridge: zmq_recv (gates) failed/timeout — Python engine not responding"); lk.lock(); break; }
            if (static_cast<size_t>(n) < 8) { fail_("issue-bridge: gates frame too short"); lk.lock(); break; }
            std::uint32_t magic = 0, gt = 0;
            std::memcpy(&magic, gbuf.data(), 4);
            std::memcpy(&gt, gbuf.data() + 4, 4);
            if (magic != GATE_MAGIC || static_cast<int>(gt) != T || n != 8 + T) {
                fail_("issue-bridge: gates frame malformed (magic/T/len mismatch — wire-contract drift, P7)"); lk.lock(); break;
            }
            for (int t = 0; t < T; ++t) ctl_->set_allow(t, gbuf[static_cast<size_t>(8 + t)] != 0);
            lk.lock();
        }
        lk.unlock();
        zmq_close(sock);
        zmq_ctx_term(ctx);
    }

    IssueController* ctl_;
    std::string endpoint_;
    double cadence_ms_;
    int timeout_ms_;
    bool running_ = false;
    std::atomic<bool> stop_{false};
    std::atomic<bool> failed_{false};
    std::string error_;
    std::mutex mu_;
    std::condition_variable cv_;
    std::thread thread_;
};

}  // namespace chocofarm
