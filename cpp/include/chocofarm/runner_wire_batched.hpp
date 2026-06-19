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
// StrictBarrier, which is structurally D=1) — the non-blocking driver holds up to D coalesced messages
// outstanding before it blocks on a reply. D is a LIVE knob: the driver emits each non-forced message as a
// bounded chunk of S_min rows (not a drain-all gather), so a ready wave of W slots stacks ⌊W/S_min⌋
// distinct outstanding messages up to D — genuine in-flight depth > 1, the §6 D-pipeline. (This overturns
// the as-built drain-all-into-one driver SYNTHESIS §0 modeled, under which depth was identically 1 and D
// was dead; see the min_coalesce note below.)
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
// `min_coalesce` is the producer-side MINIMUM coalescing degree S_min (the closed fix for the cross-thread
// COALESCING-COLLAPSE convoy — docs/design/cpp-eval-wire-formal-diagnosis.md §3 / "How I would design this
// protocol" item 6). S_min is ALSO the per-message CHUNK SIZE: the PipelinedBucket driver emits each
// non-forced message as exactly S_min ready rows (it stops gathering at S_min and leaves the rest ready),
// so a ready wave of W slots issues ⌊W/S_min⌋ distinct outstanding messages up to D — which is what makes
// the floor BIND. The driver refuses to issue a chunk of fewer than S_min ready rows WHILE replies are
// still outstanding (inflight headroom exists, so blocking on the next reply will free more slots that pool
// into a full chunk — the wait is free under the overcommit's N fibers, NOT an added timer/sleep). A
// sub-threshold message is representable ONLY as a FORCED FLUSH — when nothing is outstanding
// (inflight_msgs == 0) and ready slots remain — the single state where waiting can gather no more and
// progress/termination demand the partial send. This makes under-coalescing-while-productive STRUCTURALLY
// UNREPRESENTABLE (a closed invariant in the control flow, not a tunable that can re-open the convoy):
// S_min only raises the floor; the forced flush always drains the tail (so a high S_min can never
// deadlock), and S_min=1 degrades to the pre-fix per-row behavior exactly (no regression). It NEVER caps
// the achievable batch upward — the held remainder stays ready and flies on the next chunk (and the server
// coalesces the D outstanding chunks × T threads), so a full overcommit wave still reaches the server's
// B≈192 fast region. Mechanism note (overturning SYNTHESIS §0 for THIS arm): the as-built driver drained
// ALL ready slots into one message, so per-thread in-flight DEPTH was identically 1, D was dead, and the
// S_min floor was INERT (depth-1 forces a flush after every reply, bypassing it). Chunking at S_min restores
// genuine depth > 1 (⌊ready/S_min⌋ outstanding chunks up to D), so the floor now BINDS as a closed invariant
// — every steady-state message carries exactly S_min rows, the forced flush is reserved for the terminal
// tail (inflight==0). This floors the PER-THREAD PER-MESSAGE degree at S_min; the server's cross-thread
// rows/forward (the other writer of the coalescing degree, §3) is unchanged — the producer floor lifts the
// server's input but does not itself force rows/forward ≥ θ (a server-side increment (ii) lever if the
// cross-thread convoy survives). Default S_min=32: a PER-THREAD chunk floor large enough to defeat the B=1
// lockstep convoy yet small enough that an overcommit wave stacks several chunks (depth > 1) toward D; it is
// NOT a server bucket bound (those are cross-thread) — sweep `--min-coalesce` against rows/forward to tune
// rather than treat 32 as derived. Ignored under StrictBarrier (the strict barrier is structurally D=1 and
// gathers all parked at once, so it has no convoy to floor).
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
