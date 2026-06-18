// cpp/include/chocofarm/runner_wire_batched.hpp
// Purpose: run_episodes_wire_batched — the WIRE-batched episode driver (the production --serve GENERATION
//   path when an inference endpoint is set). It runs E self-play episodes EXACTLY as the serial
//   run_episodes (runner.cpp) — same per-episode seed fold, same world draw, same per-ply record-assembly
//   (feat/π/mask + the TERMINATE branch), same env.apply stepping, same pure-MC λ-return suffix target,
//   same redis-write / `written` all-or-nothing semantics — but resolves every Gumbel-AZ search LEAF
//   REMOTELY on the batched JAX InferenceServer over a DEALER socket instead of locally per leaf.
//
//   The control structure is a STRICT GATHER-BARRIER fiber pool: T worker threads, each owning a
//   disjoint episode subset {tid, tid+T, …}, K EpisodeSlots, its own WireLeafPool (wire_leaf_pool.hpp),
//   and its own per-slot rng. Each slot parks ONE tree at exactly one leaf (TreeState, fiber_tree.hpp);
//   each round the thread gathers ALL its parked slots' feature rows into ONE batched wire request
//   (submit_batch — one corr-id), awaits the ONE batched reply, scatters the B predictions back to the
//   slots in order (recv_batch), and resumes all — so the server's drain assembles a big batch. The
//   episode/RNG/stepping logic is RE-DERIVED from the SERIAL run_episode (runner.cpp:40-119) — NOT lifted
//   from the discarded local-batched runner — and re-homed as a resumable per-slot state machine; only the
//   leaf-resolution step differs (remote, over the wire). NO local NetForward / predict_batch is called.
//
//   FAIL LOUD (ADR-0002): any leaf failure (a recv timeout, a malformed reply, an unknown corr-id, a redis
//   write error) sets a shared `failed` flag and the WHOLE pass returns std::unexpected — never a partial
//   write, never a zero/stale leaf. This matches the run_episodes contract (a missing weight / a failed
//   write is a typed Error) so the executor's written-vs-read reconciliation stays all-or-nothing.
//
//   ADR-0012 P9: a typed value-function (std::expected<int,Error>); the WireRunnerConfig carries the
//   endpoint + pool knobs (their ONE home is RuntimeConfig, threaded in via --serve startup args, NOT
//   ActorConfig — P1). The leaf is the JAX server's SSOT batched eval (ADR-0008 keeps eval Python).
//
// Public Domain (The Unlicense).
#pragma once

#include <expected>
#include <ostream>
#include <string>

#include "chocofarm/env.hpp"
#include "chocofarm/error.hpp"
#include "chocofarm/features.hpp"
#include "chocofarm/gumbel.hpp"
#include "chocofarm/runner.hpp"
#include "chocofarm/transport.hpp"

namespace chocofarm {

// The wire-path runtime knobs (the ONE home is RuntimeConfig; these arrive as --serve startup args, never
// ActorConfig — P1). `endpoint` is the ipc:// (or tcp://) the in-process JAX InferenceServer binds;
// `pool_threads` = T OS worker threads; `pool_batch` = the in-flight leaf target across the pool
// (fibers_per_thread K = ceil(pool_batch/pool_threads), derived in RuntimeConfig). `timeout_ms` bounds the
// per-leaf DEALER recv (a timed-out leaf aborts the whole generate loudly — Q5/OR-5).
struct WireRunnerConfig {
    std::string endpoint;
    int pool_threads = 4;
    int pool_batch = 32;
    int timeout_ms = 15000;
};

// Run cfg.episodes self-play episodes over the wire-batched driver, writing the four (X, PI, M, Y) result
// blocks per non-empty episode to redis (idx-keyed, exactly as run_episodes). Returns the number of
// episodes written, OR a typed Error on ANY leaf/transport/write failure (whole-pass abort — never a
// partial write). The leaf eval is REMOTE (the JAX server at wcfg.endpoint) — there is NO NetEvaluator
// argument; each tree's policy holds the YieldingNetEvaluator that parks at the leaf. `gc` is the HOT
// Gumbel search config; `cfg` carries the live lam/max_steps/seed/res_token (P4). `stats_out` (optional)
// is the same per-episode JSON aggregate-stat sink run_episodes forwards (additive to the wire write).
[[nodiscard]] std::expected<int, Error> run_episodes_wire_batched(
    const Environment& env, const FeatureBuilder& fb, const GumbelConfig& gc, RedisClient& redis,
    const RunnerConfig& cfg, const WireRunnerConfig& wcfg, std::ostream* stats_out = nullptr);

}  // namespace chocofarm
