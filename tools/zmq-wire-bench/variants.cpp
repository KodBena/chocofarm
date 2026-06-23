// tools/zmq-wire-bench/variants.cpp — adversarial transport-variant RTT probe (NOT part of the committed bench).
// Measures the production-shaped DEALER/ROUTER (or SERVER/CLIENT, or inproc PAIR) round-trip RTT under
// several transport configs, to test whether the ~115us ipc fixed cost is irreducible or a usage/config
// artifact. Same payload shape as producer.cpp: request B*in_dim floats, reply B*out_dim floats.
//
// mode:
//   ipc-cpp    : DEALER(producer thread) <-> ROUTER(echo consumer thread), ipc://, both in C++ (no GIL).
//   ipc-busy   : same, but the producer uses a NON-BLOCKING recv spin (busy-poll) instead of blocking recv.
//   tcp-cpp    : same as ipc-cpp but tcp://127.0.0.1:port.
//   inproc     : PAIR<->PAIR over inproc:// (no IO thread for the transit, no syscall) — the T_io~=0 floor.
//   server     : draft SERVER(consumer) <-> CLIENT(producer), ipc:// (thread-safe sockets, no envelope).
//
// args: <mode> <B> <in_dim> <out_dim> <T_secs> <io_threads> [endpoint_or_port]
// Reports the same percentiles as producer.cpp. Single producer (P=1) — the regime the claim is about.
// Public Domain (The Unlicense).
#define ZMQ_BUILD_DRAFT_API 1
#include <zmq.h>
#include <atomic>
#include <thread>
#include <vector>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

static double pct(std::vector<double>& v, double q) {
    if (v.empty()) return 0.0;
    size_t i = (size_t)(q * (v.size() - 1) + 0.5);
    if (i >= v.size()) i = v.size() - 1;
    return v[i];
}

int main(int argc, char** argv) {
    if (argc < 6) {
        fprintf(stderr, "usage: %s <mode> B in_dim out_dim T_secs [io_threads] [endpoint_or_port]\n", argv[0]);
        return 2;
    }
    const std::string mode = argv[1];
    const int    B       = atoi(argv[2]);
    const int    in_dim  = atoi(argv[3]);
    const int    out_dim = atoi(argv[4]);
    const double T       = atof(argv[5]);
    const int    io_threads = argc > 6 ? atoi(argv[6]) : 1;
    const std::string ep_arg = argc > 7 ? argv[7] : "";

    const size_t req_bytes = (size_t)B * in_dim * sizeof(float);
    const size_t rep_bytes = (size_t)B * out_dim * sizeof(float);

    void* ctx = zmq_ctx_new();
    zmq_ctx_set(ctx, ZMQ_IO_THREADS, io_threads);

    std::vector<double> rtts;
    rtts.reserve(1 << 20);
    std::atomic<bool> consumer_ready{false};
    std::atomic<bool> stop{false};

    // ---------- consumer (echo) ----------
    auto echo_router = [&](const char* endpoint) {
        void* s = zmq_socket(ctx, ZMQ_ROUTER);
        int rc = zmq_bind(s, endpoint);
        if (rc != 0) { fprintf(stderr, "consumer bind failed: %s\n", zmq_strerror(zmq_errno())); std::exit(1); }
        consumer_ready = true;
        std::vector<char> id(256), buf(1u << 22);
        while (!stop.load(std::memory_order_relaxed)) {
            zmq_pollitem_t it = {s, 0, ZMQ_POLLIN, 0};
            if (zmq_poll(&it, 1, 50) <= 0) continue;
            int idn = zmq_recv(s, id.data(), id.size(), 0);            // identity
            int more; size_t msz = sizeof(more);
            zmq_getsockopt(s, ZMQ_RCVMORE, &more, &msz);
            // drain remaining request frames (corr + payload), keep last 2 for echo shape
            uint64_t corr = 0; bool have_corr = false;
            while (more) {
                int n = zmq_recv(s, buf.data(), buf.size(), 0);
                zmq_getsockopt(s, ZMQ_RCVMORE, &more, &msz);
                if (!more) break;            // payload is the LAST frame; the frame before it is corr
                if (n == (int)sizeof(uint64_t)) { memcpy(&corr, buf.data(), sizeof(corr)); have_corr = true; }
            }
            (void)have_corr;
            zmq_send(s, id.data(), idn, ZMQ_SNDMORE);
            zmq_send(s, &corr, sizeof(corr), ZMQ_SNDMORE);
            zmq_send(s, buf.data(), rep_bytes, 0);                     // reply payload (zeros, right size)
        }
        zmq_close(s);
    };

    auto echo_pair_inproc = [&](const char* endpoint) {        // inproc PAIR: pure in-process, no IO thread
        void* s = zmq_socket(ctx, ZMQ_PAIR);
        if (zmq_bind(s, endpoint) != 0) { fprintf(stderr, "pair bind failed: %s\n", zmq_strerror(zmq_errno())); std::exit(1); }
        consumer_ready = true;
        std::vector<char> buf(1u << 22);
        while (!stop.load(std::memory_order_relaxed)) {
            zmq_pollitem_t it = {s, 0, ZMQ_POLLIN, 0};
            if (zmq_poll(&it, 1, 50) <= 0) continue;
            uint64_t corr = 0;
            zmq_recv(s, &corr, sizeof(corr), 0);            // corr
            zmq_recv(s, buf.data(), buf.size(), 0);         // payload
            zmq_send(s, &corr, sizeof(corr), ZMQ_SNDMORE);
            zmq_send(s, buf.data(), rep_bytes, 0);
        }
        zmq_close(s);
    };

    // ---------- producer ----------
    auto producer_dealer = [&](const char* endpoint, bool busy) {
        void* s = zmq_socket(ctx, ZMQ_DEALER);
        if (zmq_connect(s, endpoint) != 0) { fprintf(stderr, "producer connect failed: %s\n", zmq_strerror(zmq_errno())); std::exit(1); }
        std::vector<float> payload((size_t)B * in_dim, 1.0f);
        std::vector<char> rbuf(1u << 22);
        uint64_t corr = 0xA1B2C3D4ull;
        auto t_start = std::chrono::steady_clock::now();
        for (;;) {
            auto t0 = std::chrono::steady_clock::now();
            zmq_send(s, &corr, sizeof(corr), ZMQ_SNDMORE);
            zmq_send(s, payload.data(), req_bytes, 0);
            uint64_t ce = 0;
            if (busy) {
                while (zmq_recv(s, &ce, sizeof(ce), ZMQ_DONTWAIT) < 0) { /* spin */ }
            } else {
                zmq_recv(s, &ce, sizeof(ce), 0);
            }
            zmq_recv(s, rbuf.data(), rbuf.size(), 0);
            auto t1 = std::chrono::steady_clock::now();
            rtts.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
            corr++;
            if (std::chrono::duration<double>(t1 - t_start).count() >= T) break;
        }
        zmq_close(s);
    };

    // TRUE-DEPTH pipelined DEALER: keep `depth` requests of FIXED size B in flight (issue `depth`, then
    // for each reply received issue one more). RTT here = time from a request's send to its matching reply
    // (corr-id ordered, but DEALER+echo preserves order under one consumer). Tests whether deeper pipelining
    // at FIXED message size is BAD (it should be neutral/better — the lab's "depth bad" is the size confound).
    auto producer_pipelined = [&](const char* endpoint, int depth) {
        void* s = zmq_socket(ctx, ZMQ_DEALER);
        zmq_setsockopt(s, ZMQ_RCVHWM, &depth, sizeof(depth));
        zmq_setsockopt(s, ZMQ_SNDHWM, &depth, sizeof(depth));
        if (zmq_connect(s, endpoint) != 0) { fprintf(stderr, "pipe connect failed: %s\n", zmq_strerror(zmq_errno())); std::exit(1); }
        std::vector<float> payload((size_t)B * in_dim, 1.0f);
        std::vector<char> rbuf(1u << 22);
        std::vector<std::chrono::steady_clock::time_point> sent(1 << 20);
        uint64_t corr = 0;
        auto t_start = std::chrono::steady_clock::now();
        auto issue = [&]() {
            sent[corr & 0xFFFFF] = std::chrono::steady_clock::now();
            zmq_send(s, &corr, sizeof(corr), ZMQ_SNDMORE);
            zmq_send(s, payload.data(), req_bytes, 0);
            corr++;
        };
        for (int i = 0; i < depth; i++) issue();
        for (;;) {
            uint64_t ce = 0;
            zmq_recv(s, &ce, sizeof(ce), 0);
            zmq_recv(s, rbuf.data(), rbuf.size(), 0);
            auto t1 = std::chrono::steady_clock::now();
            rtts.push_back(std::chrono::duration<double, std::micro>(t1 - sent[ce & 0xFFFFF]).count());
            if (std::chrono::duration<double>(t1 - t_start).count() >= T) break;
            issue();   // keep `depth` in flight
        }
        zmq_close(s);
    };

    auto producer_pair = [&](const char* endpoint) {
        void* s = zmq_socket(ctx, ZMQ_PAIR);
        if (zmq_connect(s, endpoint) != 0) { fprintf(stderr, "pair connect failed: %s\n", zmq_strerror(zmq_errno())); std::exit(1); }
        std::vector<float> payload((size_t)B * in_dim, 1.0f);
        std::vector<char> rbuf(1u << 22);
        uint64_t corr = 1;
        auto t_start = std::chrono::steady_clock::now();
        for (;;) {
            auto t0 = std::chrono::steady_clock::now();
            zmq_send(s, &corr, sizeof(corr), ZMQ_SNDMORE);
            zmq_send(s, payload.data(), req_bytes, 0);
            uint64_t ce = 0;
            zmq_recv(s, &ce, sizeof(ce), 0);
            zmq_recv(s, rbuf.data(), rbuf.size(), 0);
            auto t1 = std::chrono::steady_clock::now();
            rtts.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
            corr++;
            if (std::chrono::duration<double>(t1 - t_start).count() >= T) break;
        }
        zmq_close(s);
    };

    std::thread cons, prod;
    std::string ep;
    if (mode == "inproc") {
        ep = "inproc://wirevar";
        cons = std::thread(echo_pair_inproc, ep.c_str());
        while (!consumer_ready.load()) std::this_thread::sleep_for(std::chrono::milliseconds(1));
        prod = std::thread(producer_pair, ep.c_str());
    } else if (mode == "tcp-cpp") {
        ep = ep_arg.empty() ? "tcp://127.0.0.1:5599" : ep_arg;
        cons = std::thread(echo_router, ep.c_str());
        while (!consumer_ready.load()) std::this_thread::sleep_for(std::chrono::milliseconds(1));
        prod = std::thread(producer_dealer, ep.c_str(), false);
    } else if (mode.rfind("pipe", 0) == 0) {   // pipe:<depth> — true depth at FIXED size B over ipc
        int depth = 1;
        const size_t colon = mode.find(':');
        if (colon != std::string::npos) depth = atoi(mode.c_str() + colon + 1);
        if (depth < 1) depth = 1;
        ep = ep_arg.empty() ? "ipc:///tmp/wirevar_pipe.ipc" : ep_arg;
        cons = std::thread(echo_router, ep.c_str());
        while (!consumer_ready.load()) std::this_thread::sleep_for(std::chrono::milliseconds(1));
        prod = std::thread(producer_pipelined, ep.c_str(), depth);
    } else {  // ipc-cpp, ipc-busy
        ep = ep_arg.empty() ? "ipc:///tmp/wirevar.ipc" : ep_arg;
        bool busy = (mode == "ipc-busy");
        cons = std::thread(echo_router, ep.c_str());
        while (!consumer_ready.load()) std::this_thread::sleep_for(std::chrono::milliseconds(1));
        prod = std::thread(producer_dealer, ep.c_str(), busy);
    }
    prod.join();
    stop = true;
    cons.join();

    std::sort(rtts.begin(), rtts.end());
    double sum = 0; for (double x : rtts) sum += x;
    long long total = (long long)rtts.size();
    printf("VARIANT mode=%s B=%d in=%d out=%d io_threads=%d msgs=%lld thr_msgs_s=%.1f "
           "mean_us=%.2f p50_us=%.2f p90_us=%.2f p99_us=%.2f\n",
           mode.c_str(), B, in_dim, out_dim, io_threads, total, total / T,
           total ? sum / total : 0.0, pct(rtts, 0.5), pct(rtts, 0.9), pct(rtts, 0.99));
    zmq_ctx_destroy(ctx);
    return 0;
}
