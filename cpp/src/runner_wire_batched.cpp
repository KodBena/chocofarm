// cpp/src/runner_wire_batched.cpp
// Purpose: run_episodes_wire_batched (see runner_wire_batched.hpp) — the wire-batched generation driver.
//   T worker threads, each multiplexing K resumable EpisodeSlots over its own WireLeafPool DEALER socket
//   in a greedy-async loop, feeding the batched JAX InferenceServer. The per-ply episode logic (record-
//   assembly, env.apply stepping, per-episode seeding + world draw, the pure-MC λ-return suffix target)
//   is RE-DERIVED from the serial run_episode (runner.cpp:40-119) — same draws, same records, same
//   EpisodeBlocks — and re-homed as a per-slot state machine; only the leaf is remote (over the wire).
//
// Public Domain (The Unlicense).
#include "chocofarm/runner_wire_batched.hpp"

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <random>
#include <thread>
#include <vector>

#include <zmq.h>

#include "chocofarm/fiber_tree.hpp"
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

        // submit slot `s`'s currently-parked leaf into the pool (the corr-id DEALER send). A submit failure
        // is a loud whole-pass abort (ADR-0002). Called ONLY when sl.ts->running (parked at a leaf) —
        // per-tree-in-flight==1 is structural (a slot is resubmitted only after its resume).
        auto submit_parked = [&](int s) -> bool {
            if (!pool.submit(s, slots[static_cast<size_t>(s)].ts->ch.features)) {
                set_error(make_error("run_episodes_wire_batched: leaf submit failed"));
                return false;
            }
            return true;
        };

        // spawn the current ply's search tree on slot `s` for its LIVE (loc, bw, collected) AND submit its
        // first leaf IF it parked — the ONE place a tree transitions into "parked", so "parked ⟹ submitted"
        // holds BY CONSTRUCTION at the sole spawn producer (a caller cannot park-and-forget-to-submit; the
        // original 0-write bug's whole class is closed here). Builds the RNG-ctor TreeState off the slot's
        // persistent rng (the same RngGumbelSource decide_target builds, byte-identical draw order per tree),
        // starts it, and submits iff `running`. If the search returns WITHOUT parking (degenerate empty-
        // belief guard inside run_search), there is no leaf to submit and the caller drives the decision
        // through on_search_done (`ts->running` is false). Returns false ONLY on a submit failure (failed
        // set); a non-parking spawn returns true (no error — the slot just is not outstanding).
        auto spawn_ply = [&](int s) -> bool {
            EpisodeSlot& sl = slots[static_cast<size_t>(s)];
            sl.ts = std::make_unique<TreeState>(gc, env, sl.rng);  // the PRODUCTION RNG ctor (fiber_tree.hpp:65)
            sl.ts->start(sl.loc, sl.bw, sl.collected, cfg.lam);
            if (sl.ts->running) return submit_parked(s);  // parked ⟹ submit (the invariant, at its producer)
            return true;                                  // degenerate non-parking spawn: nothing to submit
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
        // decision, then spawn the next ply — and KEEP draining any chain of non-parking plies (a degenerate
        // empty-belief guard inside run_search returns without parking) until the slot is either parked-AND-
        // submitted (returns true, a leaf is outstanding) or the episode finalized (returns false). spawn_ply
        // submits on park, so "parked ⟹ submitted" holds without a separate submit here. Stops on `failed`.
        auto advance = [&](int s) -> bool {
            EpisodeSlot& sl = slots[static_cast<size_t>(s)];
            while (!failed.load()) {
                if (!apply_decision(s)) return false;     // the episode finalized this ply
                if (!spawn_ply(s)) return false;          // spawn submitted-on-park, or hit a submit error
                if (sl.ts->running) return true;          // parked at the next leaf (already submitted)
                // a non-parking next ply (degenerate): loop to apply ITS immediate decision + spawn again.
            }
            return false;
        };

        // (re)fill slot `s` with the next episode in this thread's subset; start its first ply's tree and
        // submit the first leaf. Returns true iff a leaf is now in flight on this slot. Mirrors run_episodes'
        // per-episode seed fold + world draw (runner.cpp:185-188), then run_episode's first-ply spawn.
        // Skips immediately-finalizing episodes (empty belief / a search that returns without parking),
        // trying the next idx — so a returned true always means a leaf is outstanding.
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
                if (!spawn_ply(s)) return false;        // spawn submits-on-park (or a submit error -> false)
                if (sl.ts->running) return true;        // a leaf is outstanding (spawn_ply already submitted)
                // degenerate: the first ply's search returned without parking (empty-belief guard inside
                // run_search). Drive it forward; advance() submits-on-park and drains any non-parking chain.
                if (advance(s)) return true;            // parked-AND-submitted somewhere down the chain
                if (failed.load()) return false;
                // the episode finalized without ever parking a leaf — try the next idx in this slot.
            }
            return false;
        };

        // prime K slots: each fill spawns + submits its first leaf (or finalizes a degenerate episode + moves on).
        for (int s = 0; s < K && !failed.load(); ++s) fill(s);

        // the greedy-async drain: resume ONE slot per reply, then resubmit it (still parked) or advance its
        // episode (decision done → record/step/finalize, then re-park or refill the slot). Per-tree-in-
        // flight==1 is structural (a slot is resubmitted only AFTER its resume).
        while (pool.any_outstanding() && !failed.load()) {
            auto c = pool.poll();
            if (!c) {                       // recv error / decode fail / unknown corr-id: loud abort
                set_error(c.error());
                break;
            }
            const int s = c->slot;
            EpisodeSlot& sl = slots[static_cast<size_t>(s)];
            sl.ts->resume_with(c->pred);
            if (sl.ts->running) {
                if (!submit_parked(s)) break;   // parked at the next leaf — keep the pipe full
                continue;
            }
            // the search returned its Decision: apply it + spawn the next ply (advance submits-on-park and
            // drains any non-parking chain). True -> a leaf is ALREADY outstanding again (advance submitted
            // it — do NOT submit again); false -> the episode finalized, so refill the slot.
            if (failed.load()) break;
            if (advance(s)) continue;          // a leaf is outstanding (advance already submitted on park)
            if (failed.load()) break;
            // the episode finalized — start the next episode in this slot (or leave it idle if exhausted).
            fill(s);
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

}  // namespace chocofarm
