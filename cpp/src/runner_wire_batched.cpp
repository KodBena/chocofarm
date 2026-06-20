// cpp/src/runner_wire_batched.cpp
// Purpose: run_episodes_wire_batched (see runner_wire_batched.hpp) — the wire-batched generation driver.
//   T worker threads, each multiplexing K resumable EpisodeSlots over its own WireLeafPool DEALER socket
//   in a STRICT GATHER-BARRIER loop, feeding the batched JAX InferenceServer: gather ALL currently-parked
//   slots' feature rows into ONE batched request (one corr-id, one send), await the ONE batched reply,
//   resume all, repeat. The per-ply episode logic (record-assembly, env.apply stepping, per-episode
//   seeding + world draw, the pure-MC λ-return suffix target) is RE-DERIVED from the serial run_episode
//   (runner.cpp:40-119) — same draws, same records, same EpisodeBlocks — and re-homed as a per-slot
//   state machine; only the leaf is remote (over the wire).
//
// Public Domain (The Unlicense).
#include "chocofarm/runner_wire_batched.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <mutex>
#include <random>
#include <span>
#include <thread>
#include <vector>

#include <zmq.h>

#include "chocofarm/fiber_tree.hpp"
#include "chocofarm/issue_controller.hpp"
#include "chocofarm/runtime_config.hpp"
#include "chocofarm/wire_leaf_pool.hpp"

namespace chocofarm {
namespace {

// One resumable episode in flight in a worker thread's slot. It owns the per-episode RNG (seeded
// fold_seed(seed, idx)), the world drawn once off that rng, the LIVE (loc, bw, collected) the search
// re-reads on every leaf, the bw0 for the belief-shrinkage stat, the record accumulator, and the current
// ply's TreeState (parked at one leaf until its search returns its Decision). Mirrors the serial
// run_episode's per-episode locals, lifted into a slot so K episodes advance concurrently per thread.
//
// LIFETIME (the fiber's captures, fiber_tree.hpp): TreeState::start captures loc/bw/coll BY REFERENCE and
// re-reads them on every resume — they MUST stay alive (named, stable) for the tree's whole life. They are
// members here, so the slot owns them and they never move while `ts` is live (a slot is single-writer per
// thread; no migration). The slot rng is likewise a stable member the RngGumbelSource borrows.
struct EpisodeSlot {
    int idx = -1;
    bool active = false;                       // a tree is in flight (vs the slot is idle/exhausted)
    std::mt19937_64 rng;
    uint32_t world = 0;
    Loc loc{};
    Belief bw;
    CollectedSet collected;
    int bw0 = 0;
    int ply = 0;                               // plies executed so far this episode (the max_steps cap)
    EpisodeBuilder eb;
    std::unique_ptr<TreeState> ts;             // the current ply's search tree (parked at a leaf when active)

    EpisodeSlot() : eb(EpisodeBuilder::create(0, 0.0, 0, 0, 0)) {}  // placeholder; reset per episode
};

}  // namespace

std::expected<int, Error> run_episodes_wire_batched(
    const Environment& env, const FeatureBuilder& fb, const GumbelConfig& gc,
    [[maybe_unused]] RedisClient& redis, const RunnerConfig& cfg, const WireRunnerConfig& wcfg,
    std::ostream* stats_out) {
    // Arm-3 dispatch (ADR-0012 P7 transport-scheduling flag): a PipelinedBucket request delegates to the
    // non-blocking high-D driver, leaving this strict-barrier body the byte-untouched production default.
    if (wcfg.mode == WireMode::PipelinedBucket)
        return run_episodes_wire_pipelined(env, fb, gc, redis, cfg, wcfg, stats_out);
    // NB `redis` is the API-symmetry seam with run_episodes (the serve dispatch passes it uniformly), but
    // the WIRE driver does NOT use this shared connection: hiredis's redisContext is not thread-safe, so
    // each worker thread creates its OWN RedisClient (below) for its result writes (single-writer-per-
    // connection). Marked [[maybe_unused]] so the unused shared handle does not warn under -Wextra.
    const int n_slots = n_action_slots(env);
    const int feat_dim = fb.dim();
    const std::vector<uint32_t>& worlds = env.worlds();
    if (worlds.empty())
        return std::unexpected(make_error("run_episodes_wire_batched: empty world-set (no prior)"));

    // the pool geometry: T threads, K = ceil(batch / threads) slots per thread (the ONE home derivation,
    // RuntimeConfig — the --serve args already overrode the env defaults upstream; here they arrive as the
    // WireRunnerConfig fields). At least 1 each.
    RuntimeConfig rc;
    rc.thread_pool_size = wcfg.pool_threads;
    rc.batch_size = wcfg.pool_batch;
    const int T = std::max(1, rc.thread_pool_size);
    const int K = rc.fibers_per_thread();

    void* zctx = zmq_ctx_new();
    if (zctx == nullptr)
        return std::unexpected(make_error("run_episodes_wire_batched: zmq_ctx_new failed"));

    std::atomic<int> written{0};
    std::atomic<bool> failed{false};
    Error first_error;                         // the first observed failure's diagnostic (set under a flag)
    std::atomic<bool> have_error{false};
    std::mutex err_mu;
    // globally-unique correlation ids across ALL worker threads (the per-thread WireLeafPool inflight maps
    // key on these; a future shared registry would key on them unchanged — P1).
    std::atomic<uint64_t> corr_seq{0};

    auto set_error = [&](Error e) {
        failed.store(true);
        std::lock_guard<std::mutex> lk(err_mu);
        if (!have_error.load()) {
            first_error = std::move(e);
            have_error.store(true);
        }
    };

    auto worker = [&](int tid) {
        auto pool_e = WireLeafPool::create(zctx, wcfg.endpoint, wcfg.timeout_ms, corr_seq);
        if (!pool_e) {
            set_error(pool_e.error());
            return;
        }
        WireLeafPool pool = std::move(*pool_e);

        // PER-THREAD FeatureBuilder for the RECORD build (fb.build / the belief memo are mutable + single-
        // thread-owned, features.hpp): sharing the passed `fb` across T threads would race its memo, so each
        // thread owns its own (the synchronization analysis the header demands at a sharing site — we do NOT
        // share). The search's own leaf featurizer is already per-TreeState (its policy's fb_). The passed
        // `fb` is read ONLY for dim() (pure const) above.
        FeatureBuilder rec_fb(env);

        // PER-THREAD RedisClient for the result WRITE (the SAME single-writer-per-connection discipline):
        // hiredis's redisContext is NOT thread-safe — concurrent redisCommand() on ONE shared connection
        // interleaves the synchronous request/reply byte stream on the socket and corrupts the protocol (a
        // "redis SET failed" the driver then fails loud on, ADR-0002). So each worker owns its OWN connection
        // (the CHOCO_TRANSPORT_REDIS_* env contract, same as the parent's), exactly as it owns its own
        // WireLeafPool + rec_fb. The passed `redis` param is the API-symmetry seam with run_episodes; the
        // writes go through `wredis` here. A connect failure is the loud error arm (the redis is unreachable).
        auto wredis_e = RedisClient::create();
        if (!wredis_e) {
            set_error(wredis_e.error());
            return;
        }
        RedisClient wredis = std::move(*wredis_e);

        // this thread's disjoint episode subset: tid, tid+T, tid+2T, ...
        int next_idx = tid;
        std::vector<EpisodeSlot> slots(static_cast<size_t>(K));
        const wire::count_t in_dim = static_cast<wire::count_t>(feat_dim);

        // spawn the current ply's search tree on slot `s` for its LIVE (loc, bw, collected). The ONE
        // place a tree transitions into "parked". Builds the RNG-ctor TreeState off the slot's persistent
        // rng (the same RngGumbelSource decide_target builds, byte-identical draw order per tree), starts
        // it. Leaves the slot PARKED (sl.ts->running) at its first leaf, OR finished (degenerate empty-
        // belief guard inside run_search returns without parking). NB: unlike the prior greedy-async
        // driver, spawn does NOT submit — the STRICT GATHER-BARRIER gathers all parked slots into ONE
        // batched submit below, so submission is decoupled from spawning.
        auto spawn_ply = [&](int s) {
            EpisodeSlot& sl = slots[static_cast<size_t>(s)];
            sl.ts = std::make_unique<TreeState>(gc, env, sl.rng);  // the PRODUCTION RNG ctor (fiber_tree.hpp:65)
            sl.ts->start(sl.loc, sl.bw, sl.collected, cfg.lam);
        };

        // finalize the episode in slot `s` and write its EpisodeBlocks (idx-keyed redis, same as
        // run_episodes). Sets `failed` on a write error. Then deactivates the slot (the caller refills it).
        auto finalize_and_write = [&](int s) {
            EpisodeSlot& sl = slots[static_cast<size_t>(s)];
            const double exit_c = env.exit_cost(sl.loc.pt);
            const int nb_final = env.nb(sl.bw);
            EpisodeBlocks ep = std::move(sl.eb).finalize(exit_c, nb_final);
            if (stats_out) {
                // one JSON-object line per episode (the P6 behavioral-parity sink, additive to the wire
                // write) — byte-shaped like run_episodes (runner.cpp:197-208). Guarded by a mutex so the T
                // threads do not interleave a line mid-write.
                std::lock_guard<std::mutex> lk(err_mu);  // reuse the one mutex (rare path, contention-free)
                (*stats_out) << "{\"idx\":" << sl.idx
                             << ",\"world\":" << ep.world
                             << ",\"length\":" << ep.ep_length
                             << ",\"lam_return\":" << ep.lam_return
                             << ",\"n_collect\":" << ep.n_collect
                             << ",\"n_sense\":" << ep.n_sense
                             << ",\"n_terminate\":" << ep.n_terminate
                             << ",\"belief_shrinkage\":" << ep.belief_shrinkage
                             << ",\"exec_slots\":[";
                for (size_t k = 0; k < ep.exec_slots.size(); ++k)
                    (*stats_out) << (k ? "," : "") << ep.exec_slots[k];
                (*stats_out) << "]}\n";
            }
            sl.active = false;
            sl.ts.reset();
            if (ep.n == 0) return;  // no records (empty belief immediately) — nothing to write
            auto wr = wredis.write_results(cfg.res_token, sl.idx, ep.X, ep.n, ep.feat_dim, ep.PI, ep.M,
                                           ep.Y, ep.n_slots);
            if (!wr) {
                set_error(wr.error());
                return;
            }
            written.fetch_add(1, std::memory_order_relaxed);
        };

        // Apply ONE finished ply's Decision (ts->running is false): read its Decision, assemble the record
        // EXACTLY as serial run_episode (feat/π/mask + the TERMINATE branch), env.apply step the non-
        // TERMINATE action, advance the ply. Returns true iff the episode continues to a NEXT ply (the
        // caller spawns it), false iff this ply finalized the episode (TERMINATE / horizon / empty belief).
        // Pure record/step/guard logic — no spawn, no submit (the caller owns the next-ply transition).
        auto apply_decision = [&](int s) -> bool {
            EpisodeSlot& sl = slots[static_cast<size_t>(s)];
            // run_search's Decision == decide_target's (executed action + float32 improved-π).
            const GumbelAZPolicy::Decision& dec = sl.ts->decision;
            Action action = dec.action;

            // the §2.2 feature row + the legality mask for THIS belief, and the improved-π PI row narrowed
            // to float32 — exactly decide_target + run_episode's record block (runner.cpp:128-135).
            std::vector<double> feat = rec_fb.build(sl.loc.pt, sl.bw, sl.collected);
            std::vector<float> mask = legal_mask(env, sl.bw, sl.collected);
            std::vector<float> pi(dec.improved.begin(), dec.improved.end());

            if (action.kind == ActionKind::Terminate) {
                sl.eb.record_decision(std::move(feat), std::move(pi), std::move(mask),
                                      /*is_terminate=*/true, /*is_collect=*/false, term_slot(env));
                finalize_and_write(s);
                return false;
            }
            const bool is_collect = (action.kind == ActionKind::Treasure);
            sl.eb.record_decision(std::move(feat), std::move(pi), std::move(mask),
                                  /*is_terminate=*/false, is_collect, action_to_slot(env, action));
            StepResult sr = env.apply(sl.loc, sl.bw, sl.collected, action, sl.world);
            sl.eb.record_step(sr.reward, sr.dt);
            ++sl.ply;

            // the next ply (mirrors run_episode's loop guards: max_steps cap + the empty-belief break at the
            // TOP of the loop, BEFORE any record). A break finalizes; otherwise the caller spawns the next ply.
            if (sl.ply >= cfg.max_steps || env.empty(sl.bw)) {
                finalize_and_write(s);
                return false;
            }
            return true;
        };

        // Drive slot `s`'s episode forward from a JUST-FINISHED search (ts->running false): apply the
        // decision, then spawn the next ply — and KEEP draining any chain of non-parking plies (a
        // degenerate empty-belief guard inside run_search returns without parking) until the slot is
        // either PARKED at a leaf (returns true — its row will be gathered into the next barrier) or the
        // episode finalized (returns false). Does NOT submit (the strict barrier gathers parked slots).
        // Stops on `failed`.
        auto advance = [&](int s) -> bool {
            EpisodeSlot& sl = slots[static_cast<size_t>(s)];
            while (!failed.load()) {
                if (!apply_decision(s)) return false;     // the episode finalized this ply
                spawn_ply(s);
                if (sl.ts->running) return true;          // parked at the next leaf (to be gathered)
                // a non-parking next ply (degenerate): loop to apply ITS immediate decision + spawn again.
            }
            return false;
        };

        // (re)fill slot `s` with the next episode in this thread's subset; start its first ply's tree.
        // Returns true iff the slot is now PARKED at a leaf (its row to be gathered). Mirrors
        // run_episodes' per-episode seed fold + world draw (runner.cpp:185-188), then run_episode's
        // first-ply spawn. Skips immediately-finalizing episodes (empty belief / a search that returns
        // without parking), trying the next idx — so a returned true always means a leaf is parked. Does
        // NOT submit (the strict barrier gathers parked slots into one batched send).
        auto fill = [&](int s) -> bool {
            EpisodeSlot& sl = slots[static_cast<size_t>(s)];
            while (next_idx < cfg.episodes && !failed.load()) {
                const int idx = next_idx;
                next_idx += T;
                sl.idx = idx;
                sl.rng.seed(fold_seed(cfg.seed, idx));
                std::uniform_int_distribution<size_t> wpick(0, worlds.size() - 1);
                sl.world = worlds[wpick(sl.rng)];  // the SAME world draw as serial run_episodes (rng-exact)
                sl.loc = Loc{env.entry_point()};
                sl.bw = env.full_belief();
                sl.collected = CollectedSet{};
                sl.bw0 = env.nb(sl.bw);
                sl.ply = 0;
                sl.eb = EpisodeBuilder::create(sl.world, cfg.lam, feat_dim, n_slots, sl.bw0);
                sl.active = true;

                // run_episode's first guard: an immediately-empty belief writes nothing (n==0); skip it.
                if (env.empty(sl.bw)) {
                    finalize_and_write(s);   // n==0 path: deactivates, no write
                    if (failed.load()) return false;
                    continue;
                }
                spawn_ply(s);
                if (sl.ts->running) return true;        // parked at the first leaf (to be gathered)
                // degenerate: the first ply's search returned without parking (empty-belief guard inside
                // run_search). Drive it forward; advance() drains any non-parking chain to a park/finalize.
                if (advance(s)) return true;            // parked somewhere down the chain
                if (failed.load()) return false;
                // the episode finalized without ever parking a leaf — try the next idx in this slot.
            }
            return false;
        };

        // prime K slots: each fill spawns the first ply (or finalizes a degenerate episode + moves on),
        // leaving the slot PARKED at a leaf (no submit yet — the barrier gathers them below).
        for (int s = 0; s < K && !failed.load(); ++s) fill(s);

        // ---- the STRICT GATHER-BARRIER drain ----
        // Each round: gather ALL currently-parked slots' feature rows into ONE batched request, send it
        // (one corr-id), await the ONE batched reply, then resume EACH parked slot in order — re-parking
        // it (still running → its row joins the next gather) or advancing its episode (decision done →
        // record/step/finalize, then re-spawn / refill). Loop until no slot is parked.
        //
        // (Rejected alternative: flush when ≥ a partial-parked threshold, keeping more RTT overlap. Left
        //  as an open question; we defaulted to the strict barrier.)
        std::vector<float> gather;            // the B parked rows, row-major (B·in_dim), rebuilt per round
        std::vector<int> gathered_slots;      // the parked slots, in gather order (the scatter order)
        auto any_parked = [&]() -> bool {
            for (int s = 0; s < K; ++s)
                if (slots[static_cast<size_t>(s)].active && slots[static_cast<size_t>(s)].ts &&
                    slots[static_cast<size_t>(s)].ts->running)
                    return true;
            return false;
        };
        while (any_parked() && !failed.load()) {
            gather.clear();
            gathered_slots.clear();
            for (int s = 0; s < K; ++s) {
                EpisodeSlot& sl = slots[static_cast<size_t>(s)];
                if (sl.active && sl.ts && sl.ts->running) {
                    std::span<const float> feats = sl.ts->ch.features;
                    gather.insert(gather.end(), feats.begin(), feats.end());
                    gathered_slots.push_back(s);
                }
            }
            auto sub = pool.submit_batch(gathered_slots, gather, in_dim);
            if (!sub) { set_error(sub.error()); break; }
            auto reply = pool.recv_batch();
            if (!reply) { set_error(reply.error()); break; }   // recv/decode/corr-id/count: loud abort
            // scatter the B predictions to their slots IN ORDER, resume each, then re-park or advance.
            for (const Completion& c : *reply) {
                if (failed.load()) break;
                const int s = c.slot;
                EpisodeSlot& sl = slots[static_cast<size_t>(s)];
                sl.ts->resume_with(c.pred);
                if (sl.ts->running) continue;   // parked at the next leaf — its row joins the next gather
                // the search returned its Decision: apply it + spawn/advance to the next leaf or finalize.
                if (advance(s)) continue;       // parked again (down the chain) — joins the next gather
                if (failed.load()) break;
                fill(s);                        // finalized — start the next episode in this slot (or idle)
            }
        }
    };

    std::vector<std::thread> threads;
    threads.reserve(static_cast<size_t>(T));
    for (int t = 0; t < T; ++t) threads.emplace_back(worker, t);
    for (std::thread& th : threads) th.join();

    zmq_ctx_term(zctx);

    if (failed.load()) {
        std::lock_guard<std::mutex> lk(err_mu);
        return std::unexpected(have_error.load()
                                   ? first_error
                                   : make_error("run_episodes_wire_batched: a leaf/transport/write failed"));
    }
    return written.load();
}

// ============================================================================================
// Arm 3 — the NON-BLOCKING, HIGH-D PIPELINED driver (docs/design/cpp-eval-transport-adapter.md §4 Stage B).
//
// Same per-slot EpisodeSlot state machine, same per-episode seed fold / world draw / record-assembly /
// λ-return target / redis write / whole-pass-abort semantics as the strict-barrier driver above (RE-DERIVED
// from the SAME serial run_episode) — only the TRANSPORT SCHEDULE differs:
//
//   STRICT BARRIER (above): gather ALL parked -> ONE submit -> await the ONE reply -> resume all. D=1
//     message outstanding per thread; the search idles the whole round-trip each round.
//   PIPELINED (here): keep up to D = wcfg.max_inflight_msgs COALESCED messages outstanding per thread; on
//     each reply (ONE corr-id, recv_batch), resume just those slots, advance them, and RE-ISSUE messages
//     to refill back to D — so the search never idles the full RTT and the server's group-wakeup drain
//     coalesces across the D outstanding messages (and across threads) into one big bucketed forward. A
//     slot is single-writer-per-thread (its row stays alive in the slot until its reply resumes it), so an
//     out-of-order reply routes to the right slot by corr-id with no extra bookkeeping (wire_leaf_pool.hpp
//     already maps corr-id -> ordered slot list).
//
// COALESCING FLOOR (S_min) — RETAINED BUT INERT (the producer-side experiment, REVERTED). issue() gathers
// ALL ready slots into ONE message (drain-all); since a slot regains readiness only inside the post-recv
// completion loop, a second issue with no intervening recv finds nothing ready, so per-thread in-flight DEPTH
// is identically 1 (SYNTHESIS §0) and D is a DEAD knob. On this depth-1 path the S_min floor is INERT (a
// refill finding < S_min ready forced-flushes immediately at inflight==0, bypassing the hold). The depth>1
// chunk break that made the floor bind (committed in 89d6984) was REVERTED: chunking caps per-message degree
// at S_min (below the natural drain-all ~74), floods the single-threaded server with tiny messages it
// under-coalesces, and drives a deep cross-thread convoy (wedge + growing producer RSS). The producer was
// never the bottleneck; the lever is server-side rows/forward amortization. The wire frame / codec /
// wire_spec are UNCHANGED (P7): submit_batch/recv_batch over the SAME corr-id transport.
// ============================================================================================
std::vector<SlotSnapshot> build_warm_pool(
    const Environment& env, int n_slots, std::uint64_t pool_seed, int max_div_plies) {
    const std::vector<uint32_t>& worlds = env.worlds();
    std::vector<SlotSnapshot> pool;
    if (worlds.empty() || n_slots <= 0) return pool;
    pool.reserve(static_cast<size_t>(n_slots));
    for (int s = 0; s < n_slots; ++s) {
        SlotSnapshot snap;
        snap.idx = s;
        snap.rng.seed(fold_seed(pool_seed, s));
        std::uniform_int_distribution<size_t> wpick(0, worlds.size() - 1);
        snap.world = worlds[wpick(snap.rng)];
        snap.loc = Loc{env.entry_point()};
        snap.bw = env.full_belief();
        snap.collected = CollectedSet{};
        snap.bw0 = env.nb(snap.bw);
        snap.ply = 0;
        // Advance a varied (seeded) number of random NON-terminate legal actions so the pool spans a wide
        // range of belief sizes / locations / plies (mirrors RandomPolicy's pick over env.legal_actions).
        const int target = (max_div_plies > 0)
            ? static_cast<int>(snap.rng() % static_cast<std::uint64_t>(max_div_plies + 1)) : 0;
        for (int p = 0; p < target; ++p) {
            std::vector<Action> acts = env.legal_actions(snap.bw, snap.collected);
            int n_nonterm = 0;
            for (const Action& a : acts) n_nonterm += (a.kind != ActionKind::Terminate) ? 1 : 0;
            if (n_nonterm == 0) break;   // only terminate is legal — a valid late state, stop here
            int pick = static_cast<int>(snap.rng() % static_cast<std::uint64_t>(n_nonterm));
            Action chosen = terminate_action();
            int seen = 0;
            for (const Action& a : acts) {
                if (a.kind == ActionKind::Terminate) continue;
                if (seen++ == pick) { chosen = a; break; }
            }
            env.apply(snap.loc, snap.bw, snap.collected, chosen, snap.world);
            ++snap.ply;
            if (env.empty(snap.bw)) break;   // belief collapsed (episode would terminate) — stop
        }
        pool.push_back(std::move(snap));
    }
    return pool;
}

std::expected<int, Error> run_episodes_wire_pipelined(
    const Environment& env, const FeatureBuilder& fb, const GumbelConfig& gc,
    [[maybe_unused]] RedisClient& redis, const RunnerConfig& cfg, const WireRunnerConfig& wcfg,
    std::ostream* stats_out, const std::vector<SlotSnapshot>* warm_pool, long settle_budget,
    std::vector<SlotSnapshot>* snapshot_out, IssueController* controller) {
    const int n_slots = n_action_slots(env);
    const int feat_dim = fb.dim();
    const std::vector<uint32_t>& worlds = env.worlds();
    if (worlds.empty())
        return std::unexpected(make_error("run_episodes_wire_pipelined: empty world-set (no prior)"));

    RuntimeConfig rc;
    rc.thread_pool_size = wcfg.pool_threads;
    rc.batch_size = wcfg.pool_batch;
    const int T = std::max(1, rc.thread_pool_size);
    // OVERCOMMIT (§6 M1): N independent TreeStates per thread, on top of the historical per-thread slot
    // derivation. K = N × ceil(pool_batch/pool_threads). Each slot is a self-contained, independently-
    // seeded episode (a distinct idx in this thread's stride-T subset) holding its own TreeState parked at
    // one leaf — so the N×K_base slots' in-flight leaves SUM onto this thread's ONE DEALER socket, routed
    // back out-of-order by corr-id (no per-tree interaction under virtual loss). P9: single-writer-per-tree
    // — each slot owns its TreeState, mutated only by this thread; no cross-thread / global tree pool.
    const int N = std::max(1, wcfg.trees_per_thread);
    const int K = N * rc.fibers_per_thread();
    const int D = std::max(1, wcfg.max_inflight_msgs);  // per-thread in-flight message cap
    // The producer-side MINIMUM coalescing degree S_min (the CLOSED fix for the convoy — see the header,
    // cpp-eval-wire-formal-diagnosis.md §3, SYNTHESIS §0). Clamped into [1, K]: S_min ≤ K because a single
    // round can offer at most K ready rows, so a floor above K could never be met by a non-forced issue and
    // would degenerate to forced-flush-only (B=1 again); clamping keeps the floor a meaningful threshold the
    // overcommit wave can actually clear. S_min=1 reproduces the pre-fix behavior exactly.
    const int S_min = std::clamp(wcfg.min_coalesce, 1, K);
    // The runnable gen-side batch-floor option (the "final bolt"): when on, issue() chunks each non-forced
    // message at S_min rows (depth>1 overcommit); when off, drain-all (depth≈1). See the header.
    const bool chunk_floor = wcfg.chunk_floor;

    void* zctx = zmq_ctx_new();
    if (zctx == nullptr)
        return std::unexpected(make_error("run_episodes_wire_pipelined: zmq_ctx_new failed"));

    std::atomic<int> written{0};
    std::atomic<bool> failed{false};
    Error first_error;
    std::atomic<bool> have_error{false};
    std::mutex err_mu;
    std::atomic<uint64_t> corr_seq{0};
    // In-flight-depth telemetry (the key Stage B number — the rows/forward a single real tree sustains).
    // Accumulated across threads: total leaves coalesced into messages / total messages issued = mean S
    // (rows per WIRE message); the SERVER reports rows/FORWARD (its drain coalesces across messages). Both
    // are reported — the server's mean rows/forward is the in-flight depth the design's overcommit phase needs.
    std::atomic<long> total_leaves{0};
    std::atomic<long> total_msgs{0};
    // BENCH fixed-decision measure (wcfg.decision_budget > 0): count recorded decisions across the pool and
    // stop once the budget is reached (abandoning in-flight episodes). budget_done is a CLEAN stop (not an
    // error like `failed`): the threads observe it and return, the count is reported in wire_summary.
    const long decision_budget = wcfg.decision_budget;
    // settle_budget decisions run UNCOUNTED first (pipeline desync), then decision_budget MEASURED — the
    // window [t0,t1] excludes the settle ramp. total stop = settle + measure.
    const long total_budget = (decision_budget > 0) ? settle_budget + decision_budget : 0;
    std::atomic<long> total_decisions{0};
    std::atomic<bool> budget_done{false};
    std::atomic<long long> measure_t0_ns{0};   // captured when the settle phase completes (window opens)
    std::atomic<long long> measure_t1_ns{0};   // captured when the measure budget is reached (window closes)
    auto now_ns = [] {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(
                   std::chrono::steady_clock::now().time_since_epoch()).count();
    };

    auto set_error = [&](Error e) {
        failed.store(true);
        std::lock_guard<std::mutex> lk(err_mu);
        if (!have_error.load()) {
            first_error = std::move(e);
            have_error.store(true);
        }
    };

    auto worker = [&](int tid) {
        auto pool_e = WireLeafPool::create(zctx, wcfg.endpoint, wcfg.timeout_ms, corr_seq);
        if (!pool_e) { set_error(pool_e.error()); return; }
        WireLeafPool pool = std::move(*pool_e);

        FeatureBuilder rec_fb(env);  // per-thread (the fb memo is single-thread state) — same as strict

        auto wredis_e = RedisClient::create();  // per-thread connection (hiredis ctx not thread-safe)
        if (!wredis_e) { set_error(wredis_e.error()); return; }
        RedisClient wredis = std::move(*wredis_e);

        int next_idx = tid;
        std::vector<EpisodeSlot> slots(static_cast<size_t>(K));
        const wire::count_t in_dim = static_cast<wire::count_t>(feat_dim);
        // A slot is "submitted" while its leaf is outstanding to the server (awaiting a reply); it must NOT
        // be re-gathered into another message until that reply resumes it. The corr-id transport routes the
        // reply to the slot; this flag prevents double-submitting a parked-but-already-in-flight slot.
        std::vector<char> submitted(static_cast<size_t>(K), 0);
        int inflight_msgs = 0;  // messages this thread has outstanding (== D cap)
        long my_leaves = 0, my_msgs = 0;

        // ---- the per-slot episode state machine: IDENTICAL to the strict driver's (re-derived from the
        // SAME serial run_episode). spawn_ply / finalize_and_write / apply_decision / advance / fill are
        // line-for-line the strict-barrier driver's lambdas — only the OUTER drain (below) differs. ----
        auto spawn_ply = [&](int s) {
            EpisodeSlot& sl = slots[static_cast<size_t>(s)];
            sl.ts = std::make_unique<TreeState>(gc, env, sl.rng);
            sl.ts->start(sl.loc, sl.bw, sl.collected, cfg.lam);
        };
        auto finalize_and_write = [&](int s) {
            EpisodeSlot& sl = slots[static_cast<size_t>(s)];
            const double exit_c = env.exit_cost(sl.loc.pt);
            const int nb_final = env.nb(sl.bw);
            EpisodeBlocks ep = std::move(sl.eb).finalize(exit_c, nb_final);
            if (stats_out) {
                std::lock_guard<std::mutex> lk(err_mu);
                (*stats_out) << "{\"idx\":" << sl.idx
                             << ",\"world\":" << ep.world
                             << ",\"length\":" << ep.ep_length
                             << ",\"lam_return\":" << ep.lam_return
                             << ",\"n_collect\":" << ep.n_collect
                             << ",\"n_sense\":" << ep.n_sense
                             << ",\"n_terminate\":" << ep.n_terminate
                             << ",\"belief_shrinkage\":" << ep.belief_shrinkage
                             << ",\"exec_slots\":[";
                for (size_t k = 0; k < ep.exec_slots.size(); ++k)
                    (*stats_out) << (k ? "," : "") << ep.exec_slots[k];
                (*stats_out) << "]}\n";
            }
            sl.active = false;
            sl.ts.reset();
            if (ep.n == 0) return;
            auto wr = wredis.write_results(cfg.res_token, sl.idx, ep.X, ep.n, ep.feat_dim, ep.PI, ep.M,
                                           ep.Y, ep.n_slots);
            if (!wr) { set_error(wr.error()); return; }
            written.fetch_add(1, std::memory_order_relaxed);
        };
        auto apply_decision = [&](int s) -> bool {
            EpisodeSlot& sl = slots[static_cast<size_t>(s)];
            // BENCH fixed-decision budget: every apply_decision is exactly one recorded decision (a
            // completed Gumbel search — both the Terminate and the normal branch below record_decision). The
            // first settle_budget run UNCOUNTED (pipeline desync); decision number settle_budget+1 opens the
            // measured window; reaching settle+measure closes it and flips the clean stop (threads drain out,
            // abandoning in-flight episodes). The exact-equality captures fire on a single thread each.
            if (decision_budget > 0) {
                const long v = total_decisions.fetch_add(1, std::memory_order_relaxed) + 1;
                if (v == settle_budget + 1)
                    measure_t0_ns.store(now_ns(), std::memory_order_relaxed);
                if (v >= total_budget) {
                    long long zero = 0;
                    measure_t1_ns.compare_exchange_strong(zero, now_ns(), std::memory_order_relaxed);
                    budget_done.store(true, std::memory_order_relaxed);
                }
            }
            const GumbelAZPolicy::Decision& dec = sl.ts->decision;
            Action action = dec.action;
            std::vector<double> feat = rec_fb.build(sl.loc.pt, sl.bw, sl.collected);
            std::vector<float> mask = legal_mask(env, sl.bw, sl.collected);
            std::vector<float> pi(dec.improved.begin(), dec.improved.end());
            if (action.kind == ActionKind::Terminate) {
                sl.eb.record_decision(std::move(feat), std::move(pi), std::move(mask),
                                      /*is_terminate=*/true, /*is_collect=*/false, term_slot(env));
                finalize_and_write(s);
                return false;
            }
            const bool is_collect = (action.kind == ActionKind::Treasure);
            sl.eb.record_decision(std::move(feat), std::move(pi), std::move(mask),
                                  /*is_terminate=*/false, is_collect, action_to_slot(env, action));
            StepResult sr = env.apply(sl.loc, sl.bw, sl.collected, action, sl.world);
            sl.eb.record_step(sr.reward, sr.dt);
            ++sl.ply;
            if (sl.ply >= cfg.max_steps || env.empty(sl.bw)) {
                finalize_and_write(s);
                return false;
            }
            return true;
        };
        auto advance = [&](int s) -> bool {
            EpisodeSlot& sl = slots[static_cast<size_t>(s)];
            while (!failed.load() && !budget_done.load(std::memory_order_relaxed)) {
                if (!apply_decision(s)) return false;
                spawn_ply(s);
                if (sl.ts->running) return true;
            }
            return false;
        };
        auto fill = [&](int s) -> bool {
            EpisodeSlot& sl = slots[static_cast<size_t>(s)];
            while (next_idx < cfg.episodes && !failed.load()
                   && !budget_done.load(std::memory_order_relaxed)) {
                const int idx = next_idx;
                next_idx += T;
                sl.idx = idx;
                sl.rng.seed(fold_seed(cfg.seed, idx));
                std::uniform_int_distribution<size_t> wpick(0, worlds.size() - 1);
                sl.world = worlds[wpick(sl.rng)];
                sl.loc = Loc{env.entry_point()};
                sl.bw = env.full_belief();
                sl.collected = CollectedSet{};
                sl.bw0 = env.nb(sl.bw);
                sl.ply = 0;
                sl.eb = EpisodeBuilder::create(sl.world, cfg.lam, feat_dim, n_slots, sl.bw0);
                sl.active = true;
                if (env.empty(sl.bw)) {
                    finalize_and_write(s);
                    if (failed.load()) return false;
                    continue;
                }
                spawn_ply(s);
                if (sl.ts->running) return true;
                if (advance(s)) return true;
                if (failed.load()) return false;
            }
            return false;
        };

        // POOL prime (build_warm_pool): load a slot's copyable episode snapshot and RE-SPAWN the search (the
        // fiber is non-copyable, so it is rebuilt, not copied). Thread `tid` owns the contiguous pool slice
        // [tid*K, tid*K+K). Used ONLY for the INITIAL diverse prime — an ended episode refills via fill() (a
        // fresh opener, real steady-state dynamics), never re-loading a possibly-terminal snapshot.
        auto load_from_pool = [&](int s) -> bool {
            EpisodeSlot& sl = slots[static_cast<size_t>(s)];
            const size_t pidx = static_cast<size_t>(tid) * static_cast<size_t>(K) + static_cast<size_t>(s);
            if (warm_pool == nullptr || pidx >= warm_pool->size()) return false;
            const SlotSnapshot& snap = (*warm_pool)[pidx];
            sl.idx = snap.idx;
            sl.rng = snap.rng;            // copy the rng STATE (reproducible search continuation)
            sl.world = snap.world;
            sl.loc = snap.loc;
            sl.bw = snap.bw;              // Belief value-copy
            sl.collected = snap.collected;
            sl.bw0 = snap.bw0;
            sl.ply = snap.ply;
            sl.eb = EpisodeBuilder::create(sl.world, cfg.lam, feat_dim, n_slots, sl.bw0);  // fresh (rows unused)
            sl.active = true;
            if (env.empty(sl.bw)) { sl.active = false; return false; }
            spawn_ply(s);
            if (sl.ts->running) return true;
            return advance(s);
        };
        // Prime/refill dispatch: warm-pool load when a pool is given, else the fresh ply-0 fill.
        auto prime_slot = [&](int s) -> bool {
            return warm_pool != nullptr ? load_from_pool(s) : fill(s);
        };

        // A slot is "ready" iff it is parked at a leaf AND not already outstanding to the server.
        auto is_ready = [&](int s) -> bool {
            EpisodeSlot& sl = slots[static_cast<size_t>(s)];
            return sl.active && sl.ts && sl.ts->running && !submitted[static_cast<size_t>(s)];
        };

        // Count the currently-ready slots (parked at a leaf, not already outstanding). The producer's
        // coalescing-floor decision (issue vs. wait-for-more) reads this snapshot.
        auto ready_count = [&]() -> int {
            int n = 0;
            for (int s = 0; s < K; ++s)
                if (is_ready(s)) ++n;
            return n;
        };

        // Issue ONE coalesced message. The gather shape depends on `chunk_floor` (the runnable gen-side
        // batch-floor option — the "final bolt", default OFF):
        //   * chunk_floor OFF → drain-all: gather ALL currently-ready slots into one submit_batch, so a second
        //     issue with no intervening recv finds nothing ready and per-thread depth is ≈1 (SYNTHESIS §0); D
        //     is dead and the S_min floor is inert.
        //   * chunk_floor ON → chunk-at-S_min: a non-forced message carries exactly S_min ready rows and leaves
        //     the rest READY, so the refill loop holds ⌊W/S_min⌋ distinct messages outstanding up to D —
        //     genuine depth>1 (the overcommit the server-side floor re-coalesces). The floor then BINDS: a
        //     sub-S_min non-forced gather is HELD (not sent) because more replies will arrive to fatten it.
        //
        // `force` governs the S_min floor guard either way:
        //   * `force == false`: refuse to issue when FEWER than S_min slots are ready (return false, leave them
        //     ready) — the floor guard (binds under chunk_floor; bypassed at depth-1 when off).
        //   * `force == true` (the FORCED FLUSH): emit ALL ready rows regardless — used only when nothing is
        //     outstanding (inflight_msgs == 0) and ready slots remain, the one state where waiting can gather
        //     no more (no reply will ever arrive), so progress / TERMINATION demand the partial send.
        //
        // Returns false on a submit error (loud abort), when nothing was ready, OR (force==false) when the
        // ready degree is below S_min.
        std::vector<float> gather;
        std::vector<int> gathered;
        auto issue = [&](bool force) -> bool {
            gather.clear();
            gathered.clear();
            for (int s = 0; s < K; ++s) {
                if (is_ready(s)) {
                    std::span<const float> feats = slots[static_cast<size_t>(s)].ts->ch.features;
                    gather.insert(gather.end(), feats.begin(), feats.end());
                    gathered.push_back(s);
                    // THE "FINAL BOLT" (runnable, gated): when chunk_floor is on, a non-forced message is a
                    // BOUNDED CHUNK of exactly S_min rows — stop gathering and leave the rest READY, so the
                    // refill loop holds ⌊W/S_min⌋ DISTINCT messages outstanding up to D (genuine depth>1, the
                    // overcommit the server-side floor then re-coalesces). chunk_floor OFF → drain-all (depth≈1).
                    if (chunk_floor && !force && static_cast<int>(gathered.size()) >= S_min) break;
                }
            }
            if (gathered.empty()) return false;  // nothing ready to send
            // The S_min floor guard (inert at depth-1, see above): hold a sub-threshold gather unless forced.
            if (!force && static_cast<int>(gathered.size()) < S_min) return false;
            auto sub = pool.submit_batch(gathered, gather, in_dim);
            if (!sub) { set_error(sub.error()); return false; }
            for (int s : gathered) submitted[static_cast<size_t>(s)] = 1;
            ++inflight_msgs;
            my_leaves += static_cast<long>(gathered.size());
            ++my_msgs;
            return true;
        };

        // Refill: issue a (drain-all) message while there is headroom (< D) and ≥ S_min ready. Because a
        // non-forced issue drains ALL ready into one message, this normally issues at most once per call (the
        // next issue finds nothing ready) → depth ≈ 1, D unused (SYNTHESIS §0). Then the FORCED-FLUSH backstop:
        // if NOTHING is outstanding yet ready slots remain (< S_min, so the floor guard held), issue them
        // anyway — the termination guarantee (the drain can never wedge holding a partial batch with an empty
        // pipe). At depth-1 this backstop fires on every sub-threshold refill, which is why the floor is inert
        // here (see the function header).
        auto refill = [&]() {
            const bool stop = budget_done.load(std::memory_order_relaxed);   // bench budget: issue no more
            // The non-forced ISSUE is gated by the controller's per-thread predicate (the IssueController
            // fixture; null ⇒ always allow ⇒ byte-unchanged). `D` stays the runner's fixed SAFETY ceiling —
            // the controller decides ISSUE allow/deny, NOT "overcommit" (depth>1 is the downstream emergent
            // effect). The forced-flush below stays UNGATED — the liveness floor (a denied thread still
            // drains at depth-1, never deadlocks).
            const bool may_issue = controller == nullptr || controller->may_issue(tid);
            while (!stop && inflight_msgs < D && may_issue && !failed.load() && issue(/*force=*/false)) {}
            if (!stop && inflight_msgs == 0 && !failed.load() && ready_count() > 0)
                issue(/*force=*/true);  // forced flush: nothing outstanding ⇒ waiting gathers nothing more
        };

        // prime K slots (warm-pool load or fresh fill — each leaves the slot parked at a leaf).
        for (int s = 0; s < K && !failed.load(); ++s) prime_slot(s);

        // ---- the PIPELINED drain: drain-all one message at a time (depth ≈ 1; D unused — SYNTHESIS §0) ----
        // Prime: issue the initial ready wave (forced-flush the remainder so the loop runs even if the whole
        // prime wave is sub-threshold). Then loop: recv ONE reply (out-of-order-tolerant by corr-id), resume +
        // advance just those slots, and refill. Continue while any message is outstanding — the refill's
        // forced-flush backstop guarantees we never exit with inflight_msgs == 0 while ready slots remain (no
        // deadlock), so the loop runs exactly until every episode drains.
        refill();  // prime under the floor (with the forced-flush termination backstop)
        while (inflight_msgs > 0 && !failed.load() && !budget_done.load(std::memory_order_relaxed)) {
            auto reply = pool.recv_batch();
            if (!reply) { set_error(reply.error()); break; }  // recv/decode/corr-id/count: loud abort
            --inflight_msgs;  // this corr-id's message is resolved
            // Resume each slot this message answered, advance its episode, then re-park or refill.
            for (const Completion& c : *reply) {
                if (failed.load()) break;
                const int s = c.slot;
                EpisodeSlot& sl = slots[static_cast<size_t>(s)];
                submitted[static_cast<size_t>(s)] = 0;  // no longer outstanding
                sl.ts->resume_with(c.pred);
                if (sl.ts->running) continue;  // re-parked at the next leaf — will be re-gathered on issue
                if (advance(s)) continue;      // parked again down the chain — re-gathered on issue
                if (failed.load()) break;
                fill(s);                       // finalized — start a FRESH episode (real steady-state refill;
                                               // the warm pool only stages the INITIAL diverse population, so
                                               // an ended episode does not re-load a possibly-terminal snapshot)
            }
            refill();  // hold the floor: full messages back up to depth D, then the forced-flush backstop
            // Marshal this thread's metrics to the controller (benchmark-only — null in production ⇒ zero
            // cost; single-writer per tid ⇒ race-free). The ready_count() scan is paid only under a controller.
            if (controller != nullptr)
                controller->publish(tid, inflight_msgs, ready_count(), my_msgs, my_leaves);
        }
        // Capture this thread's slots' copyable episode state into the warm pool (snapshot_out) — the slots
        // sit at varied plies (the real staggered run), giving a faithful diverse population. The fiber is
        // NOT captured (non-copyable); load_from_pool re-spawns the search. Disjoint slice → no lock.
        if (snapshot_out != nullptr) {
            for (int s = 0; s < K; ++s) {
                const EpisodeSlot& sl = slots[static_cast<size_t>(s)];
                SlotSnapshot& snap =
                    (*snapshot_out)[static_cast<size_t>(tid) * static_cast<size_t>(K) + static_cast<size_t>(s)];
                snap.idx = sl.idx;
                snap.rng = sl.rng;
                snap.world = sl.world;
                snap.loc = sl.loc;
                snap.bw = sl.bw;
                snap.collected = sl.collected;
                snap.bw0 = sl.bw0;
                snap.ply = sl.ply;
            }
        }
        total_leaves.fetch_add(my_leaves, std::memory_order_relaxed);
        total_msgs.fetch_add(my_msgs, std::memory_order_relaxed);
    };

    // Warm-pool CAPTURE target (snapshot_out): pre-size to T*K so each worker writes its disjoint slice
    // [tid*K, tid*K+K) lock-free at the budget. The slots are at varied plies (the real run staggers them),
    // so the captured population is a faithful diverse warm pool to replay per HPO config.
    if (snapshot_out != nullptr)
        snapshot_out->assign(static_cast<size_t>(T) * static_cast<size_t>(K), SlotSnapshot{});

    std::vector<std::thread> threads;
    threads.reserve(static_cast<size_t>(T));
    for (int t = 0; t < T; ++t) threads.emplace_back(worker, t);
    for (std::thread& th : threads) th.join();

    zmq_ctx_term(zctx);

    if (failed.load()) {
        std::lock_guard<std::mutex> lk(err_mu);
        return std::unexpected(have_error.load()
                                   ? first_error
                                   : make_error("run_episodes_wire_pipelined: a leaf/transport/write failed"));
    }
    // Emit the wire-side coalescing telemetry to stats_out (one trailing JSON line) when a sink is present —
    // the harness reads it to report mean rows/WIRE-MESSAGE (S); the SERVER reports mean rows/FORWARD (the
    // in-flight depth the overcommit phase needs). Guarded so it does not interleave with episode lines.
    if (stats_out) {
        const long lv = total_leaves.load(), ms = total_msgs.load();
        const double mean_s = ms ? static_cast<double>(lv) / static_cast<double>(ms) : 0.0;
        std::lock_guard<std::mutex> lk(err_mu);
        (*stats_out) << "{\"wire_summary\":1,\"leaves\":" << lv << ",\"msgs\":" << ms
                     << ",\"mean_rows_per_msg\":" << mean_s << ",\"inflight_cap_D\":" << D
                     << ",\"min_coalesce_Smin\":" << S_min
                     << ",\"decisions\":" << total_decisions.load()   // total recorded (settle + measure)
                     << ",\"measure_decisions\":" << decision_budget   // the MEASURED window (excl. settle)
                     << ",\"measure_wall_ns\":" << (measure_t1_ns.load() - measure_t0_ns.load())
                     << ",\"threads\":" << T << ",\"fibers_per_thread\":" << K << "}\n";
    }
    return written.load();
}

}  // namespace chocofarm
