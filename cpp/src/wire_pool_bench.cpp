// cpp/src/wire_pool_bench.cpp
// Purpose: the production-shaped wire-PARALLEL pool benchmark (NOT the runner) — T worker threads, each
//   multiplexing K boost.context tree-fibers over its OWN DEALER socket in a GREEDY-ASYNC loop (no
//   per-round barrier): keep K leaves continuously in flight, and the moment a reply lands, resume that
//   tree and immediately submit its next leaf, so the server's greedy drain always has up to T×K leaves
//   to batch. This is the maintainer's ask made concrete — grow the MLP-eval batch by adding FIBERS per
//   thread, not OS threads — and it supersedes the round-synchronous wire_parallel_bench MVP (whose
//   1.43× was capped by the barrier, not the batched forward).
//
//   Concurrency model (why it is race-free without a TSan-gated single-writer protocol): each thread
//   OWNS a disjoint subset of the tasks, its own DEALER socket, its own K fiber slots, and its own
//   counters — no tree migrates between threads YET, so there is no shared per-tree state and single-
//   writer-per-tree is structural. Reply→tree correlation is by an ECHOED u64 CORRELATION ID, not
//   positional FIFO: each submit stamps the request with a globally-unique corr-id (a shared atomic
//   counter), carries it as a leading zmq frame `[corr-id][payload]`, and the server echoes that frame
//   verbatim in the reply (it round-trips the envelope opaquely — the corr-id is a TRANSPORT concern, it
//   never enters the value codec, ADR-0012 P7 serialization⊥transport). The worker looks the reply's
//   corr-id up in its `inflight` map to find the slot; an unknown corr-id is a LOUD failure (ADR-0002),
//   not a silent wrong-slot apply. This buys two things the positional FIFO could not: (1) it is robust
//   to any reply-reorder, a server-side drop, or a reconnect — correlation no longer rides on submit
//   order; (2) it keeps WORK-STEALING / tree MIGRATION open as a performance lever — the corr-id is
//   globally unique, so promoting `inflight` to a shared registry lets ANY worker route ANY reply to a
//   tree regardless of which thread submitted its leaf (the positional FIFO structurally pinned a tree
//   to its submitting thread). The coming Zobrist-hashed eval cache (a mutex-pool transposition table)
//   introduces shared state regardless, so paying for correlation now is not premature — it is the
//   foundation both that cache and migration build on. The batch_size / thread_pool_size knobs come from
//   the ONE home runtime_config.hpp (fibers_per_thread = ceil(batch/threads) is derived there).
//
//   ADR-0012 P9: the fibers + the DEALERs are the effect, confined to this driver; the search core is
//   unchanged + oblivious (the YieldingNetEvaluator, fiber_leaf.hpp). A leaf RPC failure aborts that
//   thread loudly.
//
//   Protocol:  wire-pool-bench --instance <p> --faces <p> --endpoint <tcp://h:p>
//                  [--tasks N --threads T --batch B --n-sims N --m N --max-depth N --c-outcome N
//                   --lam f --timeout-ms N]   (--threads/--batch override runtime_config's env/defaults)
//   Output:    a config line, then "RESULT: PASS tasks=N threads=T batch=B fibers_per_thread=K
//              pool_dps=<n> leaves=<n> wall=<s>" + exit 0, or a loud failure.
//
// Public Domain (The Unlicense).
#include <zmq.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <deque>
#include <iostream>
#include <memory>
#include <optional>
#include <set>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <vector>

#include "chocofarm/env.hpp"
#include "chocofarm/features.hpp"
#include "chocofarm/fiber_tree.hpp"
#include "chocofarm/gumbel.hpp"
#include "chocofarm/inference_wire.hpp"
#include "chocofarm/instance.hpp"
#include "chocofarm/net_evaluator.hpp"
#include "chocofarm/runtime_config.hpp"

namespace {
[[nodiscard]] std::optional<std::string_view> opt(std::span<const std::string_view> args,
                                                  std::string_view name) {
    for (size_t i = 1; i + 1 < args.size(); ++i)
        if (args[i] == name) return args[i + 1];
    return std::nullopt;
}
[[nodiscard]] int to_int(std::string_view s) { return std::atoi(std::string(s).c_str()); }
[[nodiscard]] double to_double(std::string_view s) { return std::atof(std::string(s).c_str()); }
[[nodiscard]] double secs(std::chrono::steady_clock::time_point a,
                          std::chrono::steady_clock::time_point b) {
    return std::chrono::duration<double>(b - a).count();
}

// Receive one reply and split it into its echoed correlation id (the LEADING frame, an opaque u64 the
// server round-tripped) and the response payload (the LAST frame). Fails loud (returns false) on a
// recv error or a malformed envelope (fewer than 2 frames, or a leading frame that is not 8 bytes) —
// ADR-0002: a desynchronized wire is never silently papered over.
[[nodiscard]] bool recv_corr_payload(void* sock, uint64_t& corr, std::vector<unsigned char>& payload) {
    std::vector<std::vector<unsigned char>> frames;
    int more = 1;
    while (more) {
        zmq_msg_t m;
        zmq_msg_init(&m);
        if (zmq_msg_recv(&m, sock, 0) < 0) {
            zmq_msg_close(&m);
            return false;
        }
        const auto* d = static_cast<const unsigned char*>(zmq_msg_data(&m));
        frames.emplace_back(d, d + zmq_msg_size(&m));
        more = zmq_msg_more(&m);
        zmq_msg_close(&m);
    }
    if (frames.size() < 2 || frames.front().size() != sizeof(uint64_t)) return false;
    std::memcpy(&corr, frames.front().data(), sizeof(uint64_t));   // opaque round-trip: native bytes
    payload = std::move(frames.back());
    return true;
}

// The fiber<->driver channel, the YieldingNetEvaluator, the scripted Gumbel source, and the per-tree
// TreeState are the ONE-home shared primitives — chocofarm::{FiberLeafChannel, YieldingNetEvaluator}
// (fiber_leaf.hpp), CyclicGumbelSource (cyclic_gumbel.hpp), TreeState (fiber_tree.hpp). This bench is now
// only the GREEDY-ASYNC T×K DRIVER over those primitives (the corr-id transport + the work loop).

const std::vector<double> kGtable{0.40, -0.65, 1.10, 0.05, -0.30, 0.85, -1.20, 0.55,
                                  0.20, -0.45, 0.95, -0.10, 0.70};
[[nodiscard]] std::vector<double> script_for(int task_index) {
    std::vector<double> t(kGtable.size());
    for (size_t j = 0; j < kGtable.size(); ++j)
        t[j] = kGtable[(j + static_cast<size_t>(task_index)) % kGtable.size()];
    return t;
}
}  // namespace

int main(int argc, char** argv) {
    std::vector<std::string_view> args(argv, argv + argc);
    std::optional<std::string_view> instance = opt(args, "--instance");
    std::optional<std::string_view> faces = opt(args, "--faces");
    std::optional<std::string_view> endpoint = opt(args, "--endpoint");
    if (!instance || !faces || !endpoint) {
        std::cerr << "usage: wire-pool-bench --instance <p> --faces <p> --endpoint <tcp://h:p> "
                     "[--tasks N --threads T --batch B --n-sims N --m N --max-depth N --c-outcome N "
                     "--lam f --timeout-ms N]\n";
        return 2;
    }
    const int n_tasks = opt(args, "--tasks") ? to_int(*opt(args, "--tasks")) : 64;
    const int timeout_ms = opt(args, "--timeout-ms") ? to_int(*opt(args, "--timeout-ms")) : 15000;
    const double lam = opt(args, "--lam") ? to_double(*opt(args, "--lam")) : 0.1;
    chocofarm::GumbelConfig cfg;
    cfg.n_sims = 12;
    cfg.max_depth = 8;
    if (auto v = opt(args, "--m")) cfg.m = to_int(*v);
    if (auto v = opt(args, "--n-sims")) cfg.n_sims = to_int(*v);
    if (auto v = opt(args, "--max-depth")) cfg.max_depth = to_int(*v);
    if (auto v = opt(args, "--c-outcome")) cfg.c_outcome = to_int(*v);

    // the SSOT parallelism knobs (env defaults), with CLI overrides.
    chocofarm::RuntimeConfig rc = chocofarm::RuntimeConfig::from_env();
    if (auto v = opt(args, "--threads")) rc.thread_pool_size = std::max(1, to_int(*v));
    if (auto v = opt(args, "--batch")) rc.batch_size = std::max(1, to_int(*v));
    const int T = rc.thread_pool_size;
    const int K = rc.fibers_per_thread();

    auto inst = chocofarm::load_instance(*instance, *faces);
    if (!inst) {
        std::cerr << "wire-pool-bench: FATAL: " << inst.error().message << "\n";
        return 1;
    }
    chocofarm::Environment env(*inst);
    chocofarm::Loc loc{env.entry_point()};
    std::vector<uint32_t> bw = env.worlds();
    std::set<int> coll;

    std::cout << "config: tasks=" << n_tasks << " threads=" << T << " batch=" << rc.batch_size
              << " fibers_per_thread=" << K << " m=" << cfg.m << " n_sims=" << cfg.n_sims
              << " max_depth=" << cfg.max_depth << " endpoint=" << *endpoint
              << " n_slots=" << chocofarm::n_action_slots(env) << "\n";

    void* zctx = zmq_ctx_new();
    std::atomic<long> leaf_total{0};
    std::atomic<int> decided_total{0};
    std::atomic<bool> failed{false};
    // globally-unique correlation ids across ALL worker threads — the per-thread `inflight` maps key on
    // these now, and a future shared tree-registry (work-stealing/migration) keys on them unchanged.
    std::atomic<uint64_t> corr_seq{0};

    auto worker = [&](int tid) {
        void* sock = zmq_socket(zctx, ZMQ_DEALER);
        int linger = 0;
        zmq_setsockopt(sock, ZMQ_LINGER, &linger, sizeof(linger));
        zmq_setsockopt(sock, ZMQ_RCVTIMEO, &timeout_ms, sizeof(timeout_ms));
        if (zmq_connect(sock, std::string(*endpoint).c_str()) != 0) {
            failed.store(true);
            zmq_close(sock);
            return;
        }
        // this thread's disjoint task subset: tid, tid+T, tid+2T, ...
        std::deque<int> my_tasks;
        for (int i = tid; i < n_tasks; i += T) my_tasks.push_back(i);

        std::vector<std::unique_ptr<chocofarm::TreeState>> slots(static_cast<size_t>(K));
        std::unordered_map<uint64_t, int> inflight;  // corr-id -> slot id of its outstanding leaf
        long my_leaves = 0;
        int my_decided = 0;

        auto submit = [&](int s) {
            std::vector<unsigned char> req = chocofarm::wire::encode_request(slots[static_cast<size_t>(s)]->ch.features);
            uint64_t corr = corr_seq.fetch_add(1, std::memory_order_relaxed);
            // frame 1: the corr-id (opaque u64, the server echoes it back verbatim). frame 2: the payload.
            if (zmq_send(sock, &corr, sizeof(corr), ZMQ_SNDMORE) < 0) { failed.store(true); return; }
            if (zmq_send(sock, req.data(), req.size(), 0) < 0) { failed.store(true); return; }
            inflight.emplace(corr, s);
        };
        // (re)fill slot s with the next task; submit its first leaf. Returns true if a leaf was submitted.
        auto fill = [&](int s) -> bool {
            while (!my_tasks.empty()) {
                int ti = my_tasks.front(); my_tasks.pop_front();
                slots[static_cast<size_t>(s)] = std::make_unique<chocofarm::TreeState>(cfg, env, script_for(ti));
                slots[static_cast<size_t>(s)]->start(loc, bw, coll, lam);
                if (slots[static_cast<size_t>(s)]->running) { submit(s); return true; }
                ++my_decided;  // finished immediately (degenerate); count + try the next task
            }
            return false;
        };

        for (int s = 0; s < K && !failed.load(); ++s) fill(s);
        while (!inflight.empty() && !failed.load()) {
            uint64_t corr = 0;
            std::vector<unsigned char> payload;
            if (!recv_corr_payload(sock, corr, payload)) { failed.store(true); break; }
            auto decoded = chocofarm::wire::decode_response(payload);
            if (!decoded) { failed.store(true); break; }
            auto it = inflight.find(corr);
            if (it == inflight.end()) { failed.store(true); break; }  // unknown corr-id: a desync, loud
            int s = it->second; inflight.erase(it);  // corr-id: this reply is for slot s's outstanding leaf
            chocofarm::NetPrediction pred;
            pred.value = decoded->value;
            pred.logits = std::move(decoded->logits);
            slots[static_cast<size_t>(s)]->resume_with(pred);
            ++my_leaves;
            if (slots[static_cast<size_t>(s)]->running) {
                submit(s);  // parked at the next leaf — keep the pipe full
            } else {
                ++my_decided;
                fill(s);  // finished — start the next task in this slot
            }
        }
        zmq_close(sock);
        leaf_total.fetch_add(my_leaves);
        decided_total.fetch_add(my_decided);
    };

    auto t0 = std::chrono::steady_clock::now();
    std::vector<std::thread> threads;
    threads.reserve(static_cast<size_t>(T));
    for (int t = 0; t < T; ++t) threads.emplace_back(worker, t);
    for (std::thread& th : threads) th.join();
    auto t1 = std::chrono::steady_clock::now();
    zmq_ctx_term(zctx);

    if (failed.load()) {
        std::cout << "RESULT: FAIL (a leaf RPC failed in some thread)\n";
        return 1;
    }
    const double wall = secs(t0, t1);
    const double dps = static_cast<double>(n_tasks) / wall;
    std::cout.precision(5);
    std::cout << "RESULT: PASS tasks=" << n_tasks << " threads=" << T << " batch=" << rc.batch_size
              << " fibers_per_thread=" << K << " pool_dps=" << dps << " leaves=" << leaf_total.load()
              << " decided=" << decided_total.load() << " wall=" << wall << "\n";
    return 0;
}
