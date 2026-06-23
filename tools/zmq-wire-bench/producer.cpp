/*
 * tools/zmq-wire-bench/producer.cpp — isolated ZMQ wire round-trip benchmark (producer side).
 *
 * Purpose: measure the RAW ZMQ DEALER<->ROUTER round-trip cost per message — the WIRE alone, with the real
 * Gumbel-AZ search replaced by an optional usleep and the real net replaced by an echo consumer
 * (consumer.py) — to settle whether the leaf-eval lab's per-forward "gap" (905-1864 us, step-4) is the wire
 * or the producer's search-wait. Reflects the production producer: P C++ threads, each its own DEALER
 * socket, a [corr-id][float-payload] frame, ONE in-flight request per thread (send -> recv -> repeat).
 * Sweeps message width B and thread count P (driven by run-sweep.py); the driver runs R interleaved
 * replicates per cell and regresses RTT vs B with a CI (robust statistics).
 *
 * Per run it reports WITHIN-RUN percentiles (median/p90), not just the mean, because RTTs are right-skewed.
 *
 * args: <endpoint> <B rows> <in_dim> <out_dim> <P threads> <T secs> <usleep_us>
 *   B*in_dim floats per request (production in_dim=241); the consumer echoes B*out_dim floats (out_dim=66).
 *   usleep_us=0 => saturated wire (max throughput); >0 mimics per-message producer compute.
 *
 * Public Domain (The Unlicense).
 */
#include <zmq.h>
#include <thread>
#include <vector>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

static double pct(std::vector<double>& v, double q) {   // q in [0,1]; v is sorted on first call by caller
    if (v.empty()) return 0.0;
    size_t i = (size_t)(q * (v.size() - 1) + 0.5);
    if (i >= v.size()) i = v.size() - 1;
    return v[i];
}

int main(int argc, char** argv) {
    if (argc < 8) {
        fprintf(stderr, "usage: %s endpoint B in_dim out_dim P T_secs usleep_us\n", argv[0]);
        return 2;
    }
    const char* endpoint = argv[1];
    const int    B       = atoi(argv[2]);
    const int    in_dim  = atoi(argv[3]);
    const int    out_dim = atoi(argv[4]);
    const int    P       = atoi(argv[5]);
    const double T       = atof(argv[6]);
    const int    usleep_us = atoi(argv[7]);

    void* ctx = zmq_ctx_new();
    std::vector<std::vector<double>> rtts(P);               // per-thread per-message RTTs (us)

    auto worker = [&](int tid) {
        void* sock = zmq_socket(ctx, ZMQ_DEALER);
        zmq_connect(sock, endpoint);
        const size_t req_bytes = (size_t)B * in_dim * sizeof(float);
        std::vector<float> payload((size_t)B * in_dim, 1.0f);
        std::vector<char>  rbuf(1u << 22);                 // 4 MiB reply scratch (>= any B*out_dim*4)
        rtts[tid].reserve(1 << 16);
        uint64_t corr = ((uint64_t)0xA1B2C3D4u << 32) | (uint32_t)tid;
        const auto t_start = std::chrono::steady_clock::now();
        for (;;) {
            const auto t0 = std::chrono::steady_clock::now();
            zmq_send(sock, &corr, sizeof(corr), ZMQ_SNDMORE);   // [corr-id]
            zmq_send(sock, payload.data(), req_bytes, 0);       // [float payload]
            uint64_t corr_echo = 0;
            zmq_recv(sock, &corr_echo, sizeof(corr_echo), 0);   // reply [corr-id]
            zmq_recv(sock, rbuf.data(), rbuf.size(), 0);        // reply [payload]
            const auto t1 = std::chrono::steady_clock::now();
            rtts[tid].push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
            corr++;
            if (usleep_us > 0)
                std::this_thread::sleep_for(std::chrono::microseconds(usleep_us));
            if (std::chrono::duration<double>(t1 - t_start).count() >= T) break;
        }
        zmq_close(sock);
    };

    std::vector<std::thread> ths;
    for (int t = 0; t < P; t++) ths.emplace_back(worker, t);
    for (auto& th : ths) th.join();

    std::vector<double> all;
    for (int t = 0; t < P; t++) all.insert(all.end(), rtts[t].begin(), rtts[t].end());
    const long long total = (long long)all.size();
    double sum = 0.0; for (double x : all) sum += x;
    std::sort(all.begin(), all.end());
    const double thr      = (T > 0) ? total / T : 0.0;        // messages / second (aggregate over P threads)
    const double mean_rtt = total ? sum / total : 0.0;        // us / message (per-thread RTT)
    printf("RESULT B=%d in_dim=%d out_dim=%d P=%d usleep_us=%d msgs=%lld throughput_msgs_s=%.1f "
           "mean_rtt_us=%.2f median_rtt_us=%.2f p90_rtt_us=%.2f p99_rtt_us=%.2f\n",
           B, in_dim, out_dim, P, usleep_us, total, thr, mean_rtt,
           pct(all, 0.50), pct(all, 0.90), pct(all, 0.99));
    zmq_ctx_destroy(ctx);
    return 0;
}
