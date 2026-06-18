// cpp/src/stage_a_transport_bench.cpp
// Purpose: the Stage A PURE-TRANSPORT microbench for the eval-transport-adapter design
//   (docs/design/cpp-eval-transport-adapter.md §4). NO MCTS, NO search, NO belief — it isolates the
//   transport (S/D) from the search by driving PRE-BAKED random 241-float synthetic leaf rows from C++
//   through the real ZMQ-inproc + zero-copy-multipart + corr-id path to the real single-threaded JAX
//   InferenceServer running the real MLP forward; replies are discarded after the corr-id match. It
//   exists ONLY to map the throughput surface over the three knobs the design separates:
//
//     S = leaves coalesced into ONE wire message (send-batch): S synthetic rows ride one corr-id
//         envelope frame + one batched value-codec body (encode_request packs the (S,in_dim) matrix).
//         The corr-id is a TRANSPORT-envelope frame (the WireLeafPool's leading [corr-id] frame), the
//         codec carries only feature rows — ADR-0012 P7 serialization⊥transport.
//     D = in-flight depth: keep D coalesced messages outstanding without blocking on prior replies;
//         replies return OUT OF ORDER keyed by corr-id (the WireLeafPool inflight_ map routes them).
//     E = the server eval shape — chosen SERVER-SIDE (stage_a_server.py's --e-policy / --wakeup),
//         DECOUPLED from S. This producer does not set E; it only sets S and D.
//
//   The producer never bounds the throughput: it always has D coalesced messages of S pre-baked rows
//   ready, so the wall is the transport + server, exactly the Stage A rig the design specifies.
//
//   ADR-0012 P9: the DEALER socket + the corr-id transport are the effect, confined to the shared
//   WireLeafPool (wire_leaf_pool.hpp) — the SAME submit_batch/recv_batch the production wire driver
//   uses (no second codec, no raw pointers; std::span<const float>, std::expected error arm). A recv
//   error / unknown corr-id / desynchronized reply is a LOUD abort (ADR-0002), never a silent miscount.
//
//   Protocol:  stage-a-transport-bench --endpoint <ipc://...|tcp://...>
//                  --S <leaves/msg> --D <in-flight depth> --in-dim 241 --secs <wall budget>
//                  [--warmup-secs 1.0 --timeout-ms 30000 --seed 17]
//   Output:    a config line, then a RESULT line:
//                "RESULT: PASS S=.. D=.. in_dim=.. leaves=N msgs=M wall=W leaves_per_s=.. msgs_per_s=.."
//              + exit 0, or a loud failure + exit 1.
//
// Public Domain (The Unlicense).
#include <zmq.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <optional>
#include <random>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "chocofarm/wire_leaf_pool.hpp"

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
}  // namespace

int main(int argc, char** argv) {
    std::vector<std::string_view> args(argv, argv + argc);
    std::optional<std::string_view> endpoint = opt(args, "--endpoint");
    if (!endpoint) {
        std::cerr << "usage: stage-a-transport-bench --endpoint <tcp://h:p|ipc://...> "
                     "--S <n> --D <n> [--in-dim 241 --secs 4 --warmup-secs 1 --timeout-ms 30000 "
                     "--seed 17]\n";
        return 2;
    }
    const int S = opt(args, "--S") ? std::max(1, to_int(*opt(args, "--S"))) : 1;
    const int D = opt(args, "--D") ? std::max(1, to_int(*opt(args, "--D"))) : 1;
    const int in_dim = opt(args, "--in-dim") ? to_int(*opt(args, "--in-dim")) : 241;
    const double budget = opt(args, "--secs") ? to_double(*opt(args, "--secs")) : 4.0;
    const double warmup = opt(args, "--warmup-secs") ? to_double(*opt(args, "--warmup-secs")) : 1.0;
    const int timeout_ms = opt(args, "--timeout-ms") ? to_int(*opt(args, "--timeout-ms")) : 30000;
    const unsigned seed = opt(args, "--seed") ? static_cast<unsigned>(to_int(*opt(args, "--seed"))) : 17u;

    // Pre-bake one pool of synthetic random rows (NUM_ROWS S-row blocks), so the producer never spends
    // time generating features inside the timed loop — it cycles through the baked blocks. Finite random
    // floats in a feature-like range (the codec rejects non-finite; the forward is value-independent of
    // the actual numbers, only the shape matters — ADR-0009: the rows are synthetic by design).
    constexpr int NUM_BLOCKS = 256;
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
    std::vector<std::vector<float>> blocks(NUM_BLOCKS);
    for (auto& blk : blocks) {
        blk.resize(static_cast<size_t>(S) * in_dim);
        for (float& x : blk) x = dist(rng);
    }
    // Synthetic "slots" for one coalesced message: S leaf ids (their values are irrelevant — the reply
    // is discarded after the corr-id match; they only satisfy submit_batch's ordered-slot bookkeeping).
    std::vector<int> slots(static_cast<size_t>(S));
    for (int i = 0; i < S; ++i) slots[static_cast<size_t>(i)] = i;

    std::cout << "config: S=" << S << " D=" << D << " in_dim=" << in_dim << " secs=" << budget
              << " warmup=" << warmup << " endpoint=" << *endpoint << " blocks=" << NUM_BLOCKS << "\n";

    void* zctx = zmq_ctx_new();
    std::atomic<uint64_t> corr_seq{0};
    auto pool_e = chocofarm::WireLeafPool::create(zctx, std::string(*endpoint), timeout_ms, corr_seq);
    if (!pool_e) {
        std::cerr << "stage-a-transport-bench: FATAL: " << pool_e.error().message << "\n";
        zmq_ctx_term(zctx);
        return 1;
    }
    // The measured tallies the inner pool scope produces. `pool` is destroyed at the close of the inner
    // block (its RAII dtor zmq_close()s the DEALER) BEFORE zmq_ctx_term below — zmq_ctx_term blocks until
    // every socket on the context is closed, so terminating the context while the pool's socket is still
    // open would deadlock (the as-built bug this scope fixes). Result printing happens after the term.
    long leaves = 0, msgs = 0;
    double wall = 0.0;
    bool ok = true;
    {
        chocofarm::WireLeafPool pool = std::move(*pool_e);

        int next_block = 0;
        auto submit_one = [&]() -> bool {
            const auto& blk = blocks[static_cast<size_t>(next_block)];
            next_block = (next_block + 1) % NUM_BLOCKS;
            auto r = pool.submit_batch(std::span<const int>(slots),
                                       std::span<const float>(blk.data(), blk.size()),
                                       static_cast<chocofarm::wire::count_t>(in_dim));
            if (!r) {
                std::cerr << "stage-a-transport-bench: submit failed: " << r.error().message << "\n";
                return false;
            }
            return true;
        };

        // Prime the pipe: D coalesced messages in flight.
        for (int i = 0; i < D && ok; ++i)
            if (!submit_one()) ok = false;

        auto run_phase = [&](double phase_budget, bool count) -> bool {
            const long leaves0 = leaves, msgs0 = msgs;
            auto t0 = std::chrono::steady_clock::now();
            while (secs(t0, std::chrono::steady_clock::now()) < phase_budget) {
                auto batch = pool.recv_batch();   // ONE reply (D-deep, out-of-order by corr-id)
                if (!batch) {
                    std::cerr << "stage-a-transport-bench: recv failed: " << batch.error().message << "\n";
                    return false;
                }
                // Replies are DISCARDED after the corr-id match (Stage A: pure transport, no search
                // consumes the predictions) — count the S leaves resolved, then re-submit to hold D.
                leaves += static_cast<long>(batch->size());
                ++msgs;
                if (!submit_one()) return false;
            }
            if (!count) { leaves = leaves0; msgs = msgs0; }  // discard warmup tallies
            return true;
        };

        // Warmup phase (untimed) lets the server JIT every reachable bucket/pad shape and the pipe reach
        // steady D-depth before the measured window — ADR-0009 (the cold-compile confound the server's
        // warmup() also guards; the producer-side warmup covers steady-state pipe fill too).
        if (ok && warmup > 0.0) ok = run_phase(warmup, /*count=*/false);

        auto m0 = std::chrono::steady_clock::now();
        if (ok) ok = run_phase(budget, /*count=*/true);
        wall = secs(m0, std::chrono::steady_clock::now());

        // Drain the D still-outstanding messages so the DEALER closes clean (not counted).
        while (ok && pool.any_outstanding()) {
            auto batch = pool.recv_batch();
            if (!batch) { ok = false; break; }
            (void)batch;
        }
    }  // pool dtor closes the socket HERE, before zmq_ctx_term
    zmq_ctx_term(zctx);

    if (!ok) {
        std::cout << "RESULT: FAIL (a transport RPC failed)\n";
        return 1;
    }
    const double lps = static_cast<double>(leaves) / wall;
    const double mps = static_cast<double>(msgs) / wall;
    std::cout.precision(7);
    std::cout << "RESULT: PASS S=" << S << " D=" << D << " in_dim=" << in_dim
              << " leaves=" << leaves << " msgs=" << msgs << " wall=" << wall
              << " leaves_per_s=" << lps << " msgs_per_s=" << mps << "\n";
    return 0;
}
