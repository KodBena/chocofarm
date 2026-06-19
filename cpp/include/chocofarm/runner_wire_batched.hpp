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

// The runner's wire transport scheduling mode (the SELECTABLE arm — docs/design/cpp-eval-transport-adapter.md
// §4 Stage B). This is a TRANSPORT-SCHEDULING knob only (it changes WHEN/HOW MANY leaf messages are
// outstanding, never the wire frame / codec / wire_spec — ADR-0012 P7), so both modes produce
// behaviorally-equivalent search (the forward is row-independent; replies route per corr-id):
//   * StrictBarrier — the DEFAULT, untouched production path: each round gather ALL parked slots into ONE
//     batched submit (one corr-id, S=#parked), await the ONE reply, resume all (D=1 outstanding/thread).
//   * PipelinedBucket — arm 3: keep MULTIPLE coalesced messages outstanding (D>1, non-blocking), resume
//     each fiber as ITS reply lands (out of order by corr-id), re-submit immediately to hold D. Pairs with
//     the server's bucketed-E + group-wakeup drain (the Stage A StageAServer behavior, a server flag) — the
//     server, not the runner, decides the forward shape. The strict-barrier path stays the production default.
enum class WireMode { StrictBarrier, PipelinedBucket };

// The wire-path runtime knobs (the ONE home is RuntimeConfig; these arrive as --serve startup args, never
// ActorConfig — P1). `endpoint` is the ipc:// (or tcp://) the in-process JAX InferenceServer binds;
// `pool_threads` = T OS worker threads; `pool_batch` = the in-flight leaf target across the pool
// (fibers_per_thread K = ceil(pool_batch/pool_threads), derived in RuntimeConfig). `timeout_ms` bounds the
// per-leaf DEALER recv (a timed-out leaf aborts the whole generate loudly — Q5/OR-5). `mode` selects the
// transport scheduling arm (StrictBarrier = production default; PipelinedBucket = arm 3, behind a flag);
// `max_inflight_msgs` is the per-thread in-flight message cap D for PipelinedBucket (ignored under
// StrictBarrier, which is structurally D=1). NB — D is currently a DEAD knob: the driver gathers ALL ready
// slots into ONE message (drain-all), so a second issue with no intervening recv finds nothing ready and
// per-thread in-flight depth is identically 1 (SYNTHESIS §0). The depth>1 chunked-pipeline that would make D
// live was reverted (see the min_coalesce note).
//
// `trees_per_thread` is the OVERCOMMIT multiplier N (docs/design/cpp-eval-transport-adapter.md §6 M1):
// each PipelinedBucket producer thread owns N × K INDEPENDENT EpisodeSlots (each a self-contained
// TreeState parked at one leaf), not K, so its concurrent in-flight leaves SUM toward the server's fast
// region (the Stage-B measure found one tree parks ~1 leaf; K slots ~54 rows/forward; N multiplies the
// slot count to push rows/forward toward B≈192). P9 holds structurally: each thread OWNS its N×K trees
// + its DEALER socket (single-writer-per-tree) — no shared/stolen tree state. The corr-id transport
// already routes an out-of-order reply to the right (slot=tree-fiber); more slots need no routing change.
// N=1 reproduces the pre-overcommit slot count exactly. Ignored under StrictBarrier (production default
// untouched — the strict path keeps its K = ceil(pool_batch/pool_threads) slots).
//
// `min_coalesce` is the producer-side minimum coalescing degree S_min — an EMPIRICALLY-REFUTED experiment,
// RETAINED but INERT (kept as reviewed scaffolding; it does nothing on the current drain-all path).
// Intent (cpp-eval-wire-formal-diagnosis.md §3 / "How I would design this protocol" item 6): floor the
// per-message coalescing degree so the cross-thread COALESCING-COLLAPSE convoy becomes unrepresentable.
// issue() holds a sub-threshold (< S_min) gather unless forced; the forced flush (inflight_msgs == 0 with
// ready slots remaining) is the termination tail. BUT on the drain-all path per-thread depth is ≈1, so a
// refill that finds < S_min ready forced-flushes immediately — the floor never holds (INERT; measured:
// identical B and dps at S_min=1 vs 32). The depth>1 chunked-pipeline that would have made it bind
// (commit 89d6984) was REVERTED: it capped per-message degree at S_min (below the natural drain-all ~74)
// and flooded the single-threaded server with tiny messages it under-coalesces → a deep cross-thread convoy
// (wedge + growing producer RSS). The producer's per-message degree was never the bottleneck — the server's
// cross-thread rows/forward is (the fixed per-forward cost amortized over the batch), so the real fix is
// server-side (increment (ii): preferred_batch_size / max_queue_delay). S_min=1 reproduces the pre-fix
// drain-all behavior. Ignored under StrictBarrier.
struct WireRunnerConfig {
    std::string endpoint;
    int pool_threads = 4;
    int pool_batch = 32;
    int timeout_ms = 15000;
    WireMode mode = WireMode::StrictBarrier;
    int max_inflight_msgs = 8;
    int trees_per_thread = 1;
    int min_coalesce = 32;
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

// Arm 3 (docs/design/cpp-eval-transport-adapter.md §4 Stage B): the NON-BLOCKING, HIGH-D pipelined driver.
// Same contract as run_episodes_wire_batched (same per-episode seed fold / world draw / record-assembly /
// λ-return target / redis write / whole-pass-abort semantics — RE-DERIVED from the SAME serial run_episode
// and the SAME per-slot state machine), but the transport schedule differs: it keeps up to
// wcfg.max_inflight_msgs coalesced messages outstanding per thread WITHOUT a strict per-round barrier,
// resumes each fiber as its own reply lands (OUT OF ORDER by corr-id), and re-submits immediately to hold D
// — pairing with the server's bucketed-E + group-wakeup drain (which assembles the forward shape). The
// wire frame / codec / wire_spec are UNCHANGED (ADR-0012 P7 — transport scheduling, not a codec change).
// run_episodes_wire_batched delegates here when wcfg.mode == WireMode::PipelinedBucket, so the production
// StrictBarrier path stays byte-untouched and this arm is reachable only behind the mode flag.
[[nodiscard]] std::expected<int, Error> run_episodes_wire_pipelined(
    const Environment& env, const FeatureBuilder& fb, const GumbelConfig& gc, RedisClient& redis,
    const RunnerConfig& cfg, const WireRunnerConfig& wcfg, std::ostream* stats_out = nullptr);

}  // namespace chocofarm
