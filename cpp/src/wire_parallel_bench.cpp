// cpp/src/wire_parallel_bench.cpp
// Purpose: the OVER-THE-WIRE PARALLEL benchmark (NOT the runner) — the third §6-Q5 axis. K independent
//   Gumbel-AZ trees run as boost.context fibers on ONE multiplexer thread; each advances (UNCHANGED
//   run_search) to its leaf and YIELDS; the multiplexer batch-submits all parked leaves over a
//   non-blocking DEALER socket so the Python server's greedy drain BATCHES them into one forward, then
//   recvs the replies (positional FIFO within a round) and resumes each fiber. This amortizes the
//   ~6 ms un-batched single-row JAX forward the wire-SYNCHRONOUS axis pays per leaf — the whole point of
//   the wire-parallel regime.
//
//   This combines the two foundations proven separately: the Option-A fiber (fiber_proto.cpp — the
//   unchanged search runs in a fiber, yielding at the leaf) and the batched DEALER transport
//   (dealer_probe.cpp — many outstanding, server batches, positional FIFO). It is the ROUND-SYNCHRONOUS
//   MVP (a barrier per round: submit all parked leaves, recv all, resume all) — which keeps the wire
//   contract unchanged (no echoed id needed: one peer, submit-then-recv, the server replies in submit
//   order, design §4.1). The continuous greedy-async work-stealing pool (per-tree corr-id, no barrier) is
//   the production refinement; this MVP measures the batching throughput win first (ADR-0009 measure-first).
//
//   ADR-0012 P9: the fiber + the DEALER are the effect, confined to this driver; the search core stays a
//   pure value-function. A leaf RPC timeout/decode failure aborts loudly (ADR-0002). The fiber-leaf
//   primitives + the per-tree fixture are now the ONE-home shared types (ADR-0012 P1): FiberLeafChannel +
//   YieldingNetEvaluator (fiber_leaf.hpp), CyclicGumbelSource (cyclic_gumbel.hpp), TreeState
//   (fiber_tree.hpp) — the SAME types the wire-pool bench and the Option-A proof use.
//
//   Protocol:  wire-parallel-bench --instance <p> --faces <p> --endpoint <tcp://h:p>
//                  [--trees K --n-sims N --m N --max-depth N --c-outcome N --lam f --timeout-ms N]
//   Output:    a config line, then "RESULT: PASS trees=K wire_parallel_dps=<n> rounds=<r>
//              first_batch=<b> leaves=<n> wall=<s>" + exit 0, or a loud failure.
//
// Public Domain (The Unlicense).
#include <zmq.h>

#include <chrono>
#include <cstdint>
#include <iostream>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "chocofarm/env.hpp"
#include "chocofarm/features.hpp"
#include "chocofarm/fiber_tree.hpp"
#include "chocofarm/gumbel.hpp"
#include "chocofarm/inference_wire.hpp"
#include "chocofarm/instance.hpp"
#include "chocofarm/net_evaluator.hpp"

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

// recv ONE multipart reply, return its last frame (the payload); ok=false on timeout/error.
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

// The fiber<->driver channel, the YieldingNetEvaluator, the scripted Gumbel source, and the per-tree
// TreeState are the ONE-home shared primitives — chocofarm::{FiberLeafChannel, YieldingNetEvaluator}
// (fiber_leaf.hpp), CyclicGumbelSource (cyclic_gumbel.hpp), TreeState (fiber_tree.hpp). This bench is now
// only the ROUND-SYNCHRONOUS multiplexer DRIVER over those primitives (batch-submit, recv, resume).
}  // namespace

int main(int argc, char** argv) {
    std::vector<std::string_view> args(argv, argv + argc);
    std::optional<std::string_view> instance = opt(args, "--instance");
    std::optional<std::string_view> faces = opt(args, "--faces");
    std::optional<std::string_view> endpoint = opt(args, "--endpoint");
    if (!instance || !faces || !endpoint) {
        std::cerr << "usage: wire-parallel-bench --instance <p> --faces <p> --endpoint <tcp://h:p> "
                     "[--trees K --n-sims N --m N --max-depth N --c-outcome N --lam f --timeout-ms N]\n";
        return 2;
    }
    const int K = opt(args, "--trees") ? to_int(*opt(args, "--trees")) : 16;
    const int timeout_ms = opt(args, "--timeout-ms") ? to_int(*opt(args, "--timeout-ms")) : 10000;
    const double lam = opt(args, "--lam") ? to_double(*opt(args, "--lam")) : 0.1;
    chocofarm::GumbelConfig cfg;
    cfg.n_sims = 12;
    cfg.max_depth = 8;
    if (auto v = opt(args, "--m")) cfg.m = to_int(*v);
    if (auto v = opt(args, "--n-sims")) cfg.n_sims = to_int(*v);
    if (auto v = opt(args, "--max-depth")) cfg.max_depth = to_int(*v);
    if (auto v = opt(args, "--c-outcome")) cfg.c_outcome = to_int(*v);

    auto inst = chocofarm::load_instance(*instance, *faces);
    if (!inst) {
        std::cerr << "wire-parallel-bench: FATAL: " << inst.error().message << "\n";
        return 1;
    }
    chocofarm::Environment env(*inst);
    chocofarm::Loc loc{env.entry_point()};
    chocofarm::Belief bw = env.full_belief();   // the seam's belief construction entry
    chocofarm::CollectedSet coll;
    std::vector<double> gtable{0.40, -0.65, 1.10, 0.05, -0.30, 0.85, -1.20, 0.55,
                               0.20, -0.45, 0.95, -0.10, 0.70};

    // the non-blocking DEALER leaf transport.
    void* zctx = zmq_ctx_new();
    void* sock = zmq_socket(zctx, ZMQ_DEALER);
    int linger = 0;
    zmq_setsockopt(sock, ZMQ_LINGER, &linger, sizeof(linger));
    zmq_setsockopt(sock, ZMQ_RCVTIMEO, &timeout_ms, sizeof(timeout_ms));
    if (zmq_connect(sock, std::string(*endpoint).c_str()) != 0) {
        std::cerr << "wire-parallel-bench: FATAL: connect failed: " << zmq_strerror(zmq_errno()) << "\n";
        return 1;
    }

    std::cout << "config: trees=" << K << " m=" << cfg.m << " n_sims=" << cfg.n_sims
              << " max_depth=" << cfg.max_depth << " c_outcome=" << cfg.c_outcome << " lam=" << lam
              << " endpoint=" << *endpoint << " n_slots=" << chocofarm::n_action_slots(env) << "\n";

    // K independent tree-fibers (per-tree rotated gumbel script so the trees differ).
    std::vector<std::unique_ptr<chocofarm::TreeState>> trees;
    trees.reserve(static_cast<size_t>(K));
    for (int i = 0; i < K; ++i) {
        std::vector<double> table(gtable.size());
        for (size_t j = 0; j < gtable.size(); ++j)
            table[j] = gtable[(j + static_cast<size_t>(i)) % gtable.size()];
        trees.push_back(std::make_unique<chocofarm::TreeState>(cfg, env, std::move(table)));
    }

    auto t0 = std::chrono::steady_clock::now();
    for (auto& t : trees) t->start(loc, bw, coll, lam);  // advance each to its first leaf

    int rounds = 0, first_batch = 0;
    long leaf_total = 0;
    bool failed = false;
    while (!failed) {
        // collect the parked leaves of all still-running trees + batch-submit them.
        std::vector<int> active;
        for (int i = 0; i < K; ++i)
            if (trees[static_cast<size_t>(i)]->running) active.push_back(i);
        if (active.empty()) break;
        if (rounds == 0) first_batch = static_cast<int>(active.size());
        ++rounds;
        for (int i : active) {
            std::span<const float> feats = trees[static_cast<size_t>(i)]->ch.features;
            // a per-leaf submit is the degenerate B=1 batched frame.
            auto req_e = chocofarm::wire::encode_request(
                feats, /*B=*/1, static_cast<chocofarm::wire::count_t>(feats.size()));
            if (!req_e) {
                std::cerr << "wire-parallel-bench: FATAL: encode failed: " << req_e.error().message << "\n";
                failed = true;
                break;
            }
            const std::vector<unsigned char>& req = *req_e;
            if (zmq_send(sock, req.data(), req.size(), 0) < 0) {
                std::cerr << "wire-parallel-bench: FATAL: send failed: " << zmq_strerror(zmq_errno())
                          << "\n";
                failed = true;
                break;
            }
        }
        if (failed) break;
        // recv the replies in submit order (positional FIFO within the round) + resume each fiber.
        for (int i : active) {
            bool ok = false;
            std::vector<unsigned char> payload = recv_payload(sock, ok);
            if (!ok) {
                std::cerr << "wire-parallel-bench: FATAL: recv timed out/failed\n";
                failed = true;
                break;
            }
            auto decoded = chocofarm::wire::decode_response(payload);
            if (!decoded) {
                std::cerr << "wire-parallel-bench: FATAL: decode failed: " << decoded.error().message
                          << "\n";
                failed = true;
                break;
            }
            if (decoded->size() != 1) {
                std::cerr << "wire-parallel-bench: FATAL: reply carried " << decoded->size()
                          << " predictions, expected 1 (B=1 per-leaf)\n";
                failed = true;
                break;
            }
            chocofarm::NetPrediction pred;
            pred.value = (*decoded)[0].value;
            pred.logits = std::move((*decoded)[0].logits);
            trees[static_cast<size_t>(i)]->resume_with(pred);
            ++leaf_total;
        }
    }
    auto t1 = std::chrono::steady_clock::now();

    zmq_close(sock);
    zmq_ctx_term(zctx);
    if (failed) {
        std::cout << "RESULT: FAIL (a leaf RPC failed mid-run)\n";
        return 1;
    }

    const double wall = secs(t0, t1);
    const double dps = static_cast<double>(K) / wall;
    std::cout.precision(5);
    std::cout << "RESULT: PASS trees=" << K << " wire_parallel_dps=" << dps << " rounds=" << rounds
              << " first_batch=" << first_batch << " leaves=" << leaf_total << " wall=" << wall << "\n";
    return 0;
}
