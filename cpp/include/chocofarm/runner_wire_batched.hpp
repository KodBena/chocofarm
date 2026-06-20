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

#include <cstdint>
#include <expected>
#include <ostream>
#include <random>
#include <string>
#include <vector>

#include "chocofarm/env.hpp"
#include "chocofarm/error.hpp"
#include "chocofarm/features.hpp"
#include "chocofarm/gumbel.hpp"
#include "chocofarm/runner.hpp"
#include "chocofarm/transport.hpp"

namespace chocofarm {

class IssueController;   // the online overcommit controller fixture (issue_controller.hpp); injected by ptr

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
// StrictBarrier, which is structurally D=1). D is LIVE only when `chunk_floor` is on: the chunk break makes
// each non-forced message carry S_min rows and leave the rest ready, so up to D distinct messages stay
// outstanding (depth>1). With `chunk_floor` off the driver gathers ALL ready into ONE message (drain-all),
// a second issue finds nothing ready, and per-thread depth is identically 1 (SYNTHESIS §0) — D is then dead.
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
// `min_coalesce` is the producer-side minimum coalescing degree S_min (cpp-eval-wire-formal-diagnosis.md §3
// / "How I would design this protocol" item 6): the per-message coalescing floor. It BINDS only when
// `chunk_floor` is on (see below); on the drain-all path (chunk_floor==false) it is INERT (a refill finding
// < S_min ready forced-flushes immediately at depth≈1, bypassing the hold — measured identical B/dps at
// S_min=1 vs 32). Clamped to [1, K]. S_min=1 reproduces the pre-fix drain-all behavior. Ignored under
// StrictBarrier.
//
// `chunk_floor` is the RUNNABLE generation-side batch-floor option (the "final bolt" — reverted in e6d2c41,
// re-instated here behind a flag, default OFF). When TRUE, issue() emits each non-forced message as a
// BOUNDED CHUNK of exactly S_min ready rows (it stops gathering at S_min and leaves the rest READY), so a
// ready wave of W slots issues ⌊W/S_min⌋ DISTINCT outstanding messages up to D — genuine in-flight DEPTH > 1,
// the §6 D-pipeline (overcommit on the wire). The held remainder pools with later-freed slots into the next
// chunk (no deadlock: the forced flush at inflight==0 always drains the tail). This makes the floor BIND
// (every steady-state message carries S_min rows) WITHOUT capping aggregate throughput — every ready row
// still flies, across D chunks. ALONE it floods the single-threaded server with small messages it
// under-coalesces (the revert's convoy); the design intent is to pair it with the SERVER-side coalescing
// floor (inference_server min_forward_rows θ, increment ii) that re-assembles the D×T chunks into one large
// forward — so the producer supplies the overcommit DEPTH while the server controls the forward WIDTH. When
// FALSE, issue() drains ALL ready into ONE message (depth≈1, the production path). Ignored under StrictBarrier.
struct WireRunnerConfig {
    std::string endpoint;
    int pool_threads = 4;
    int pool_batch = 32;
    int timeout_ms = 15000;
    WireMode mode = WireMode::StrictBarrier;
    int max_inflight_msgs = 8;
    int trees_per_thread = 1;
    int min_coalesce = 32;
    bool chunk_floor = false;   // gen-side depth>1 chunk-at-S_min (the runnable "final bolt"); default OFF
    // CONTROL-LAB per-forward on-wire decision transport (the Batch-0 harness; PipelinedBucket only),
    // default OFF (the production/bench path is byte-unchanged when off). When ON *and* an IssueController
    // is injected, each producer thread rides its per-forward feature snapshot in the request's
    // LAB-CONTROL envelope frame (lab_control_wire.hpp) and reads its next issue-gate bit off the reply,
    // actuating through the SAME IssueController::set_allow / may_issue cell (no second actuation path).
    // The decision epoch is the eval server's forward (the lab StageAServer runs the Controller there).
    // This SUPERSEDES the async issue_control_bridge FOR THE LAB; with `lab_decision` off the bridge path
    // is unchanged. Ignored under StrictBarrier and when `controller` is null.
    bool lab_decision = false;
    // BENCH-ONLY fixed-DECISION measure budget (PipelinedBucket only). 0 = unlimited (run cfg.episodes to
    // completion — the production/default path, byte-unchanged). When > 0, the pipelined driver stops as
    // soon as it has RECORDED this many decisions (completed Gumbel searches) ACROSS the pool and abandons
    // the in-flight episodes — so the measured window is a fixed amount of real search work at full
    // occupancy (dps = decisions / wall), not a full episode wave. The count is reported in the wire_summary
    // (`decisions`); abandoned episodes are NOT written to redis, so the bench reads the count from there,
    // not from result blocks. Keeps the LIVE Gumbel search + inference (unlike the synthetic transport bench).
    long decision_budget = 0;
};

// A reproducible WARM-POOL snapshot of ONE slot's copyable episode state (everything EXCEPT the live
// search fiber, which is non-copyable — TreeState deletes copy/move; the search is re-spawned on load).
// `build_warm_pool` advances each slot a varied (seeded) number of random legal actions so the pool spans
// a WIDE RANGE of search dynamics (belief sizes, locations, plies) — fixing the cold-start lockstep + the
// all-openers sampling bias the naive decision-budget measure suffers (server-gen-floor-result.md follow-up).
// Belief is a copyable value-variant, so the snapshot copies in-memory; the pool is regenerated bit-
// identically per config from the same `pool_seed`, so every config measures the SAME population (comparable
// numbers). Public Domain.
struct SlotSnapshot {
    int idx = -1;
    std::mt19937_64 rng;
    std::uint32_t world = 0;
    Loc loc{};
    Belief bw;
    CollectedSet collected;
    int bw0 = 0;
    int ply = 0;
};

// Build a reproducible diverse warm pool of `n_slots` slot snapshots: each slot draws a world from
// `pool_seed`+slot, then advances 0..`max_div_plies` random NON-terminate legal actions (env.apply,
// filtering the belief by the drawn world) so beliefs/locations span the natural distribution. Deterministic
// in `pool_seed` (the comparable-numbers guarantee). No search, no transport — cheap (sub-second).
[[nodiscard]] std::vector<SlotSnapshot> build_warm_pool(
    const Environment& env, int n_slots, std::uint64_t pool_seed, int max_div_plies);

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
// `pool` (optional): a warm-pool prime (build_warm_pool) — when non-null, each slot loads its snapshot's
// copyable episode state and re-spawns the search (instead of priming a fresh ply-0 opener), and an ended
// episode reloads its snapshot (the population stays diverse). `settle_budget`: decisions to run UNCOUNTED
// before the measured window opens (pipeline desync) — with `wcfg.decision_budget` the MEASURE budget, the
// driver runs settle_budget + decision_budget decisions and reports the measured window's decisions +
// wall (steady-state, excluding settle) in the wire_summary. Both default off (the legacy path is unchanged).
[[nodiscard]] std::expected<int, Error> run_episodes_wire_pipelined(
    const Environment& env, const FeatureBuilder& fb, const GumbelConfig& gc, RedisClient& redis,
    const RunnerConfig& cfg, const WireRunnerConfig& wcfg, std::ostream* stats_out = nullptr,
    const std::vector<SlotSnapshot>* warm_pool = nullptr, long settle_budget = 0,
    std::vector<SlotSnapshot>* snapshot_out = nullptr, IssueController* controller = nullptr);

}  // namespace chocofarm
