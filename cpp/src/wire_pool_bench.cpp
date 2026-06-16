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
//   counters — no tree migrates between threads, so there is no shared per-tree state. Single-writer-
//   per-tree is structural. Reply→tree correlation is positional FIFO PER SOCKET: one DEALER peer
//   submits-then-the-server-replies in submit order (per-peer ordering, design §4.1), and each slot has
//   exactly one leaf outstanding at a time, so a FIFO of slot ids matches replies to slots with no
//   echoed-id. (The flip side, acknowledged: with no corr-id, a violation of one-peer-in-order — a
//   shared socket, a mid-run reconnect that changes identity, or a server-side drop of a malformed
//   frame — would silently apply a reply to the WRONG slot, caught only by the eventual recv timeout
//   (a value misalignment, not a crash). This bench never triggers it (one socket/thread, no malformed
//   sends); the production pool adds the echoed u64 corr-id — design §4.1.) The batch_size /
//   thread_pool_size knobs come from the ONE home runtime_config.hpp
//   (fibers_per_thread = ceil(batch/threads) is derived there).
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
#include <boost/context/fiber.hpp>
#include <zmq.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <deque>
#include <iostream>
#include <memory>
#include <optional>
#include <set>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include "chocofarm/env.hpp"
#include "chocofarm/features.hpp"
#include "chocofarm/fiber_leaf.hpp"
#include "chocofarm/gumbel.hpp"
#include "chocofarm/inference_wire.hpp"
#include "chocofarm/instance.hpp"
#include "chocofarm/runtime_config.hpp"

namespace ctxb = boost::context;

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

[[nodiscard]] std::vector<unsigned char> recv_payload(void* sock, bool& ok) {
    std::vector<unsigned char> last;
    ok = false;
    int more = 1;
    while (more) {
        zmq_msg_t m;
        zmq_msg_init(&m);
        if (zmq_msg_recv(&m, sock, 0) < 0) {
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

class ScriptedGumbelSource final : public chocofarm::GumbelSource {
  public:
    explicit ScriptedGumbelSource(std::vector<double> table) : table_(std::move(table)) {}
    uint32_t sample_world(const std::vector<uint32_t>& bw) override { return bw.empty() ? 0u : bw[0]; }
    std::vector<double> gumbel(int n) override {
        std::vector<double> out(static_cast<size_t>(n));
        for (int i = 0; i < n; ++i) out[static_cast<size_t>(i)] = table_[(idx_++) % table_.size()];
        return out;
    }

  private:
    std::vector<double> table_;
    size_t idx_ = 0;
};

// One tree in a fiber. Heap-allocated for a STABLE address (the fiber captures references to its members).
struct TreeState {
    chocofarm::FiberLeafChannel ch;
    chocofarm::YieldingNetEvaluator ynet;
    chocofarm::GumbelAZPolicy policy;
    ScriptedGumbelSource src;
    chocofarm::GumbelAZPolicy::Decision decision;
    ctxb::fiber fib;
    bool running = false;

    TreeState(const chocofarm::GumbelConfig& cfg, const chocofarm::Environment& env,
              std::vector<double> table)
        : ynet(ch), policy(cfg, ynet, env), src(std::move(table)) {}

    void start(const chocofarm::Loc& loc, const std::vector<uint32_t>& bw, const std::set<int>& coll,
               double lam) {
        fib = ctxb::fiber{std::allocator_arg, ctxb::fixedsize_stack(512 * 1024),
                          [this, &loc, &bw, &coll, lam](ctxb::fiber&& caller) {
                              ch.caller = std::move(caller);
                              decision = policy.run_search(loc, bw, coll, lam, src);
                              ch.at_leaf = false;
                              return std::move(ch.caller);
                          }};
        fib = std::move(fib).resume();
        running = ch.at_leaf;
    }
    void resume_with(const chocofarm::NetPrediction& pred) {
        ch.value = pred;
        fib = std::move(fib).resume();
        running = ch.at_leaf;
    }
};

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

        std::vector<std::unique_ptr<TreeState>> slots(static_cast<size_t>(K));
        std::deque<int> fifo;  // slot ids, submit order (per-peer reply order matches)
        long my_leaves = 0;
        int my_decided = 0;

        auto submit = [&](int s) {
            std::vector<unsigned char> req = chocofarm::wire::encode_request(slots[static_cast<size_t>(s)]->ch.features);
            if (zmq_send(sock, req.data(), req.size(), 0) < 0) { failed.store(true); return; }
            fifo.push_back(s);
        };
        // (re)fill slot s with the next task; submit its first leaf. Returns true if a leaf was submitted.
        auto fill = [&](int s) -> bool {
            while (!my_tasks.empty()) {
                int ti = my_tasks.front(); my_tasks.pop_front();
                slots[static_cast<size_t>(s)] = std::make_unique<TreeState>(cfg, env, script_for(ti));
                slots[static_cast<size_t>(s)]->start(loc, bw, coll, lam);
                if (slots[static_cast<size_t>(s)]->running) { submit(s); return true; }
                ++my_decided;  // finished immediately (degenerate); count + try the next task
            }
            return false;
        };

        for (int s = 0; s < K && !failed.load(); ++s) fill(s);
        while (!fifo.empty() && !failed.load()) {
            bool ok = false;
            std::vector<unsigned char> payload = recv_payload(sock, ok);
            if (!ok) { failed.store(true); break; }
            auto decoded = chocofarm::wire::decode_response(payload);
            if (!decoded) { failed.store(true); break; }
            int s = fifo.front(); fifo.pop_front();  // FIFO: this reply is for slot s's outstanding leaf
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
