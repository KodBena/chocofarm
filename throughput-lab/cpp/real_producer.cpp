// throughput-lab/cpp/real_producer.cpp
// Purpose: the REAL-generator load driver (NON-FIBER baseline) — N producer threads, each running real
//   Gumbel-AZ decisions back-to-back through its OWN tlab::Boundary (a per-thread DEALER), each leaf a
//   B=1 blocking round-trip to the live server. With N threads each holding one leaf in flight, the
//   server gathers up to N concurrent leaves per forward (batch ~= N). This is the NON-FIBER data point
//   the fiber multiplexer (K leaves/thread -> batch ~= N*K) is measured against: the open question is
//   whether the fiber model helps or hurts throughput vs this baseline (the maintainer's investigation,
//   neither prior trusted). All rates MEASURED (leaves/wall, decisions/wall), never assumed (ADR-0009).
//
//   Built only under -DTLAB_REAL_GENERATOR=ON (links chocofarm_core). The synthetic tlab-producer stays
//   a standalone clean-room binary; this is the additive real-generator sibling (ADR-0012 compose).
// Public Domain (The Unlicense).
#include <sys/resource.h>   // setpriority / PRIO_PROCESS — per-thread nice (Linux: nice is per-task)
#include <sys/syscall.h>    // SYS_gettid
#include <unistd.h>         // syscall

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <vector>

#include "chocofarm/env.hpp"
#include "chocofarm/fiber_tree.hpp"
#include "chocofarm/gumbel.hpp"
#include "chocofarm/instance.hpp"
#include "chocofarm/issue_control_bridge.hpp"   // consolidation Gate A: REUSE the control plane (one home,
#include "chocofarm/issue_controller.hpp"       //   ADR-0012 P1) — the same headers runner_wire_batched uses
#include "chocofarm/search_runtime.hpp"

#include "boundary.hpp"
#include "boundary_net_evaluator.hpp"
#include "cursor_slot.hpp"   // OPTION B: the explicit-state CursorSlot engine (selectable, default fiber)

namespace {
// A small fixed Gumbel script (the scripted CyclicGumbelSource path, as wire_parallel_bench uses): the
// RNG-free source produces a faithful search STRUCTURE / leaf pattern without the production RNG slot-fill
// — sufficient for a LOAD generator (throughput depends on the search's leaf-dependency + matmul shape,
// not on the gumbel draws being random). Each fiber rotates the table so the K trees differ.
const std::vector<double> kGumbelTable{0.40, -0.65, 1.10, 0.05, -0.30, 0.85, -1.20, 0.55,
                                       0.20, -0.45, 0.95, -0.10, 0.70};
constexpr double kLam = 0.1;
}  // namespace

namespace {
using SteadyClock = std::chrono::steady_clock;
[[nodiscard]] double secs_since(SteadyClock::time_point t0) {
    return std::chrono::duration<double>(SteadyClock::now() - t0).count();
}
[[nodiscard]] std::optional<std::string_view> opt(std::span<const std::string_view> a, std::string_view k) {
    for (size_t i = 1; i + 1 < a.size(); ++i)
        if (a[i] == k) return a[i + 1];
    return std::nullopt;
}

// MemAvailable from /proc/meminfo (the kernel's own estimate of allocatable-without-swap bytes), as a
// std::optional: present iff the field was read. Absence is a typed value, NOT a sentinel (ADR-0012 P9 /
// ADR-0002): on a kernel/container without MemAvailable the admission guard SKIPS rather than guessing.
[[nodiscard]] std::optional<std::uint64_t> mem_available_bytes() {
    std::ifstream f("/proc/meminfo");
    if (!f) return std::nullopt;
    std::string key;
    std::uint64_t kb = 0;
    std::string unit;
    while (f >> key >> kb >> unit) {
        if (key == "MemAvailable:") return kb * 1024ull;
        f.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    }
    return std::nullopt;
}

// Estimated RESIDENT bytes one fiber holds while parked mid-decision. The driver keeps EVERY fiber's tree
// live simultaneously (each parked on a leaf across the multiplexed RTT), so the producer's resident floor
// is threads*fibers*this. This model is RE-CALIBRATED to the post-structural-fix footprint (RCA
// tlab_finding #26; the O(fibers)-resident dissolution: the per-fiber belief memo is now BOUNDED to a small
// ring (FeatureBuilder::kDefaultBeliefCacheCap, features.hpp), the node arena's overflow now RETURNS to the
// OS per decision (releasing_arena.hpp), and the fiber stack is a demand-paged guard-page mmap (fiber_tree.
// hpp) — so the former dominant unbounded term, the belief memo, no longer scales with tree size). MEASURED
// (ADR-0009, the structural-fix per-fiber sweep at default cap, /usr/bin/time peak RSS / K, both n_sims):
//   K 256/512/1024 @ n_sims=256 -> ~1489/1475/1314 KiB per fiber;  @ n_sims=48 -> ~411/395/389 KiB per
//   fiber. A two-point fit gives base ~256 KiB + ~5 KiB/sim (the residual is the live TreeState + policy +
//   node arena; the bounded belief memo is cap*~2.6 KiB ~= 42 KiB, no longer the dominant term). The
//   constants below are rounded slightly CONSERVATIVE (over- not under-estimate) so the admission guard
//   stays a true fail-loud backstop — it ADMITS the full config (1024/256: ~1536 KiB*1024 ~= 1.5 GiB/
//   producer, well under 50% of an 8 GiB box) yet still rejects a genuinely-too-big ask.
// Non-fiber (--fibers 0) runs ONE tree per thread, so the live count is `threads`, not threads*fibers.
[[nodiscard]] std::uint64_t est_fiber_resident_bytes(int n_sims) {
    constexpr std::uint64_t kFiberBaseBytes = 256ull * 1024;   // measured per-fiber floor (TreeState+policy+
                                                               //   stack-resident+bounded belief memo)
    constexpr std::uint64_t kBytesPerSim = 5ull * 1024;        // measured live node storage per sim (post-fix)
    return kFiberBaseBytes
           + static_cast<std::uint64_t>(std::max(n_sims, 0)) * kBytesPerSim;
}

struct ThreadStat {
    std::uint64_t decisions = 0;
    std::uint64_t leaves = 0;   // predict() round-trips issued (the leaf-eval count)
    bool failed = false;
    std::string err;
};

// Renice THIS generator thread DOWN iff it is the designated one (Linux per-task nice via setpriority on
// the thread's gettid). In the generator-bound regime, the inference server's core has idle slack; a 4th
// generator sharing that core, reniced low, soaks the slack but yields the instant the server has a batch.
// A no-op unless low_prio_thread names this index. Failure is non-fatal + loud-ish (ADR-0002).
void apply_thread_priority(int thread_index, int low_prio_thread, int low_prio_nice) {
    if (low_prio_thread < 0 || thread_index != low_prio_thread || low_prio_nice == 0) return;
    const auto tid = static_cast<id_t>(::syscall(SYS_gettid));
    errno = 0;
    if (::setpriority(PRIO_PROCESS, tid, low_prio_nice) != 0)
        std::fprintf(stderr, "[tlab-real-producer] WARN: could not renice thread %d to nice %d (errno=%d)\n",
                     thread_index, low_prio_nice, errno);
}

// One producer thread: build its own boundary + bridge + SerialRuntime, then run real decisions from the
// root state (varying the seed so trees differ) until the wall deadline. Each decision's leaf_requests is
// the count of B=1 round-trips it drove through the boundary.
void run_thread(int idx, const chocofarm::Environment& env, const std::string& endpoint,
                const chocofarm::GumbelConfig& cfg, double run_seconds, int in_dim, ThreadStat& out) {
    tlab::BoundaryConfig bcfg;
    bcfg.endpoint = endpoint;
    bcfg.recv_timeout_ms = 10000;     // generous: the server may be busy gathering other threads' leaves
    bcfg.n_producer_threads = 1;
    bcfg.rows = 1;
    bcfg.in_dim = in_dim;
    auto b = tlab::make_boundary(tlab::BoundaryTopology::PerThread, bcfg);
    if (!b) { out.failed = true; out.err = "boundary: " + b.error().message; return; }
    std::unique_ptr<tlab::Boundary> boundary = std::move(*b);
    tlab::BoundaryNetEvaluator bridge(*boundary);
    chocofarm::SerialRuntime serial(bridge);

    const chocofarm::Loc loc{env.entry_point()};
    const chocofarm::Belief bw = env.full_belief();
    const chocofarm::CollectedSet coll;
    const auto start = SteadyClock::now();
    std::uint64_t seed = static_cast<std::uint64_t>(idx) * 1'000'003ull + 1ull;
    while (secs_since(start) < run_seconds) {
        chocofarm::SearchTask t;
        t.loc = loc; t.bw = bw; t.collected = coll; t.lam = 0.1; t.seed = seed++; t.cfg = cfg;
        std::vector<chocofarm::SearchTask> tasks{t};
        auto dec = serial.run(env, std::span<const chocofarm::SearchTask>(tasks));
        if (!dec) { out.failed = true; out.err = "runtime: " + dec.error().message; return; }
        out.decisions += 1;
        out.leaves += static_cast<std::uint64_t>((*dec)[0].leaf_requests);
    }
}

// Drive ONE round-synchronous batch: submit the `active` fibers' parked leaves COALESCED into messages of
// up to `coalesce_rows` leaves each (B<=M) -- K leaves become ceil(K/M) requests, fewer per-request decodes
// server-side and fewer/bigger forwards (the static S_min coalescing floor; M=1 = the per-leaf B=1 path).
// corr -> the group's fibers (submit order) so each B=G reply's G preds route home in order; then recv all
// groups and resume each fiber. Returns false (out.failed set) on a transport/decode fault. Shared by the
// root and episodic drivers (ADR-0012 P1: one home for the coalescing send/recv).
//
// TEMPLATED over the per-slot engine TYPE (ADR-0012 P1, ENGINE-PARAMETERIZED one-home): `Slot` is
// chocofarm::TreeState (Option A, the fiber) or tlab::CursorSlot (Option B, the explicit-state cursor).
// Both expose the SAME surface this body reads — `slot->ch.features` (the parked leaf row) and
// `slot->resume_with(pred)` — so the coalescing send/recv is byte-identical across engines and the fiber
// instantiation compiles to the exact code it did before (deduced on TreeState). Only the slot type
// differs; the pipe + routing are one home.
template <class Slot>
[[nodiscard]] bool drive_round(tlab::Boundary& boundary,
                               std::vector<std::unique_ptr<Slot>>& trees,
                               const std::vector<int>& active, int coalesce_rows, int in_dim,
                               tlab::wire::corr_t& corr, ThreadStat& out) {
    std::unordered_map<tlab::wire::corr_t, std::vector<int>> corr_to_group;
    std::vector<float> buf;
    int n_msgs = 0;
    for (size_t off = 0; off < active.size(); off += static_cast<size_t>(coalesce_rows)) {
        const size_t g_end = std::min(active.size(), off + static_cast<size_t>(coalesce_rows));
        buf.clear();
        std::vector<int> group;
        group.reserve(g_end - off);
        for (size_t k = off; k < g_end; ++k) {
            const std::span<const float> feats = trees[static_cast<size_t>(active[k])]->ch.features;
            buf.insert(buf.end(), feats.begin(), feats.end());   // concat the group's feature rows
            group.push_back(active[k]);
        }
        const tlab::wire::corr_t cc = corr++;
        const auto G = static_cast<tlab::wire::count_t>(group.size());
        corr_to_group.emplace(cc, std::move(group));
        const tlab::LeafBatch lb{cc, G, static_cast<tlab::wire::count_t>(in_dim),
                                 std::span<const float>(buf.data(), buf.size())};
        if (auto s = boundary.send(lb); !s) { out.failed = true; out.err = "send: " + s.error().message; return false; }
        ++n_msgs;   // buf is reused next group: send copies the bytes into the wire frame before returning
    }
    for (int r = 0; r < n_msgs; ++r) {
        auto reply = boundary.recv();
        if (!reply) { out.failed = true; out.err = "recv: " + reply.error().message; return false; }
        auto it = corr_to_group.find(reply->corr);
        if (it == corr_to_group.end() || reply->preds.size() != it->second.size()) {
            out.failed = true; out.err = "unmatched/size-mismatch reply corr=" + std::to_string(reply->corr); return false;
        }
        for (size_t j = 0; j < it->second.size(); ++j) {
            chocofarm::NetPrediction pred;
            pred.value = reply->preds[j].value;
            pred.logits = std::move(reply->preds[j].logits);
            trees[static_cast<size_t>(it->second[j])]->resume_with(pred);
            out.leaves += 1;
        }
    }
    return true;
}

// One FIBER producer thread: multiplex K TreeState fibers over its own Boundary, ROUND-SYNCHRONOUS
// (wire_parallel_bench's discipline): each round, submit every parked fiber's leaf (B=1) into the DEALER,
// let the SERVER gather the K concurrent requests into one forward, then recv the K replies and resume
// each fiber. K leaves in flight per thread -> the server's per-forward batch grows with K (and with N
// threads, ~N*K). A finished fiber is restarted on a fresh decision to keep K in flight for the window.
// This is the fiber arm of the investigation; run_thread (above) is the non-fiber baseline it is measured
// against. (Greedy-async -- keep the pipe full across rounds -- is the next refinement.)
void run_thread_fiber(int idx, const chocofarm::Environment& env, const std::string& endpoint,
                      const chocofarm::GumbelConfig& cfg, double run_seconds, int in_dim, int fibers_k,
                      int coalesce_rows, ThreadStat& out) {
    if (coalesce_rows < 1) coalesce_rows = 1;
    tlab::BoundaryConfig bcfg;
    bcfg.endpoint = endpoint;
    bcfg.recv_timeout_ms = 10000;
    bcfg.n_producer_threads = 1;
    bcfg.rows = coalesce_rows;   // up to coalesce_rows leaves per message -> sizes the send HWM budget
    bcfg.in_dim = in_dim;
    auto b = tlab::make_boundary(tlab::BoundaryTopology::PerThread, bcfg);
    if (!b) { out.failed = true; out.err = "boundary: " + b.error().message; return; }
    std::unique_ptr<tlab::Boundary> boundary = std::move(*b);

    // Root state — kept alive for every fiber's whole life (TreeState::start captures loc/bw/coll BY
    // REFERENCE and re-reads them on every leaf across all resume_with calls).
    const chocofarm::Loc loc{env.entry_point()};
    const chocofarm::Belief bw = env.full_belief();
    const chocofarm::CollectedSet coll;

    // K independent tree-fibers (scripted source, per-tree rotated table so the trees differ).
    std::vector<std::unique_ptr<chocofarm::TreeState>> trees;
    trees.reserve(static_cast<size_t>(fibers_k));
    for (int i = 0; i < fibers_k; ++i) {
        std::vector<double> table(kGumbelTable.size());
        for (size_t j = 0; j < kGumbelTable.size(); ++j)
            table[j] = kGumbelTable[(j + static_cast<size_t>(i)) % kGumbelTable.size()];
        trees.push_back(std::make_unique<chocofarm::TreeState>(cfg, env, std::move(table)));
    }
    for (auto& t : trees) t->start(loc, bw, coll, kLam);   // advance each to its first parked leaf

    tlab::wire::corr_t corr = static_cast<tlab::wire::corr_t>(idx) * 1'000'000'000ull + 1ull;
    const auto t_start = SteadyClock::now();
    while (secs_since(t_start) < run_seconds) {
        // Collect parked fibers; restart any that finished (count the completed decision) to keep K busy.
        std::vector<int> active;
        active.reserve(static_cast<size_t>(fibers_k));
        for (int i = 0; i < fibers_k; ++i) {
            if (!trees[static_cast<size_t>(i)]->running) {
                out.decisions += 1;
                trees[static_cast<size_t>(i)]->start(loc, bw, coll, kLam);
            }
            if (trees[static_cast<size_t>(i)]->running) active.push_back(i);
        }
        if (active.empty()) break;

        if (!drive_round(*boundary, trees, active, coalesce_rows, in_dim, corr, out)) return;
    }
}

// EPISODIC fiber driver: like run_thread_fiber, but each slot runs a SEQUENCE of decisions forming an
// EPISODE. On a completed decision the executed action steps the slot's OWN (loc, bw, collected) via
// env.apply against a per-episode sampled true world (mirrors runner.cpp's run_episode), and the next
// decision starts from the EVOLVED state, not the root. With cfg.no_early_exit on, a Terminate is
// substituted unless the belief is exhausted, so episodes run to a genuine terminal (or env.max_steps).
// This is the production-shape workload; DPS = decisions/wall. Scripted gumbel source for now (incremental
// fidelity); a per-slot RNG samples each episode's world uniformly from the prior worlds. The per-slot
// state vectors are RESERVED (never reallocated) so TreeState::start's by-reference captures stay valid.
//
// The episode STATE MACHINE (the `advance` lambda) is defined ONCE; `driver` selects only the PIPE SHAPE
// (ADR-0012 P1 -- one home for the episode logic, one branch for the overlap):
//   round-sync : submit every parked fiber's leaf (coalesced into coalesce_rows-row messages), then BLOCK
//                recv'ing the WHOLE round before resuming any -> the search cores idle across the round RTT.
//   greedy     : keep up to inflight_msgs coalesced messages CONTINUOUSLY in flight; recv ONE group, resume
//                + re-arm its fibers immediately and re-send -> producer compute overlaps the server forward.
// Coalescing (coalesce_rows) is held IDENTICAL across both, so an A/B isolates the pipe shape, not the
// batch width -- the within-stack driver attribution the ours/overcommit bridge needs (ADR-0013).
//
// TEMPLATED over the per-slot engine (ADR-0012 P1): `Slot` is chocofarm::TreeState (Option A, fiber) or
// tlab::CursorSlot (Option B, explicit-state cursor); `make_slot(i)` builds slot i (the per-slot rotated
// gumbel table differs only in the slot TYPE, not the rotation). EVERYTHING else — the per-slot episode
// state (ep_loc/ep_bw/ep_coll/ep_world/ep_step/ep_rng), new_episode, the advance() state machine, the
// greedy + round-sync pipe shapes, the control-plane gate — is REUSED VERBATIM across engines, because
// both slot types expose the SAME surface (.start/.running/.ch.features/.resume_with/.decision). The
// engine is the ONLY thing that differs; the episode logic is one home. run_thread_fiber_episodic /
// run_thread_cursor_episodic are one-line wrappers that bind `Slot` + the factory (below).
template <class Slot, class MakeSlot>
void run_episodic(int idx, const chocofarm::Environment& env, const std::string& endpoint,
                  double run_seconds, int in_dim,
                  int fibers_k, int coalesce_rows, const std::string& driver,
                  int inflight_msgs, chocofarm::IssueController* ctl, MakeSlot make_slot,
                  ThreadStat& out) {
    // NOTE: the GumbelConfig is NOT a parameter here — it is baked into each slot by `make_slot` (the
    // engine factory), exactly where it is needed (the per-slot policy ctor). The episode body reads only
    // env (max_steps/empty/apply) + the slot surface, so threading cfg through here would be a dead param.
    if (coalesce_rows < 1) coalesce_rows = 1;
    tlab::BoundaryConfig bcfg;
    bcfg.endpoint = endpoint; bcfg.recv_timeout_ms = 10000; bcfg.n_producer_threads = 1;
    bcfg.rows = coalesce_rows; bcfg.in_dim = in_dim;
    auto b = tlab::make_boundary(tlab::BoundaryTopology::PerThread, bcfg);
    if (!b) { out.failed = true; out.err = "boundary: " + b.error().message; return; }
    std::unique_ptr<tlab::Boundary> boundary = std::move(*b);

    const int max_steps = env.max_steps();
    const auto& worlds = env.worlds();
    const std::size_t n_worlds = worlds.size();

    std::vector<chocofarm::Loc> ep_loc; ep_loc.reserve(static_cast<size_t>(fibers_k));
    std::vector<chocofarm::Belief> ep_bw; ep_bw.reserve(static_cast<size_t>(fibers_k));
    std::vector<chocofarm::CollectedSet> ep_coll; ep_coll.reserve(static_cast<size_t>(fibers_k));
    std::vector<std::uint32_t> ep_world(static_cast<size_t>(fibers_k), 0);
    std::vector<int> ep_step(static_cast<size_t>(fibers_k), 0);
    std::vector<std::mt19937_64> ep_rng; ep_rng.reserve(static_cast<size_t>(fibers_k));
    std::vector<std::unique_ptr<Slot>> trees; trees.reserve(static_cast<size_t>(fibers_k));

    auto new_episode = [&](int i) {
        ep_loc[static_cast<size_t>(i)] = chocofarm::Loc{env.entry_point()};
        ep_bw[static_cast<size_t>(i)] = env.full_belief();
        ep_coll[static_cast<size_t>(i)] = chocofarm::CollectedSet{};
        ep_world[static_cast<size_t>(i)] = static_cast<std::uint32_t>(
            worlds[ep_rng[static_cast<size_t>(i)]() % n_worlds]);
        ep_step[static_cast<size_t>(i)] = 0;
    };

    for (int i = 0; i < fibers_k; ++i) {
        ep_loc.emplace_back(env.entry_point());
        ep_bw.emplace_back(env.full_belief());
        ep_coll.emplace_back();
        ep_rng.emplace_back(static_cast<std::uint64_t>(idx) * 1'000'003ull + static_cast<std::uint64_t>(i) + 1ull);
        trees.push_back(make_slot(i));   // the engine-specific slot factory (fiber TreeState or CursorSlot)
    }
    for (int i = 0; i < fibers_k; ++i) {
        new_episode(i);
        trees[static_cast<size_t>(i)]->start(ep_loc[static_cast<size_t>(i)], ep_bw[static_cast<size_t>(i)],
                                             ep_coll[static_cast<size_t>(i)], kLam);
    }

    // advance(i): slot i's decision completed -> count it, step the episode (env.apply the executed action,
    // or start a fresh episode at a terminal: Terminate, max_steps, or exhausted belief), then start the
    // next decision so the fiber is parked on its first leaf again. The SINGLE home for the episode state
    // machine -- BOTH pipe shapes below call it (ADR-0012 P1); `driver` selects only the overlap shape.
    auto advance = [&](int i) {
        const size_t si = static_cast<size_t>(i);
        out.decisions += 1;
        const chocofarm::Action act = trees[si]->decision.action;
        ep_step[si] += 1;
        const bool terminal = (act.kind == chocofarm::ActionKind::Terminate)
                              || ep_step[si] >= max_steps || env.empty(ep_bw[si]);
        if (terminal) new_episode(i);                                          // fresh episode
        else env.apply(ep_loc[si], ep_bw[si], ep_coll[si], act, ep_world[si]); // step the env
        trees[si]->start(ep_loc[si], ep_bw[si], ep_coll[si], kLam);            // next decision
    };

    tlab::wire::corr_t corr = static_cast<tlab::wire::corr_t>(idx) * 1'000'000'000ull + 1ull;
    const auto t_start = SteadyClock::now();

    if (driver == "greedy") {
        // GREEDY-ASYNC pipe: keep up to `budget` coalesced messages CONTINUOUSLY in flight. Each loop, fill
        // the in-flight budget from the `ready` fibers (groups of up to coalesce_rows rows), then recv ONE
        // group, resume + advance its fibers, and return them to `ready` so they re-arm immediately. The
        // producer's search compute thus overlaps the server's forward (vs round-sync's whole-round barrier).
        const int budget = inflight_msgs > 0 ? inflight_msgs : 8;
        std::unordered_map<tlab::wire::corr_t, std::vector<int>> corr_to_group;
        std::vector<int> ready; ready.reserve(static_cast<size_t>(fibers_k));
        for (int i = 0; i < fibers_k; ++i) ready.push_back(i);   // all parked on a leaf after start
        std::vector<float> buf;
        int in_flight = 0;
        auto send_group = [&]() -> bool {
            const size_t g = std::min(ready.size(), static_cast<size_t>(coalesce_rows));
            buf.clear();
            std::vector<int> group; group.reserve(g);
            for (size_t k = ready.size() - g; k < ready.size(); ++k) {
                const std::span<const float> feats = trees[static_cast<size_t>(ready[k])]->ch.features;
                buf.insert(buf.end(), feats.begin(), feats.end());   // concat the group's feature rows
                group.push_back(ready[k]);
            }
            ready.resize(ready.size() - g);
            const tlab::wire::corr_t cc = corr++;
            const auto G = static_cast<tlab::wire::count_t>(group.size());
            corr_to_group.emplace(cc, std::move(group));
            const tlab::LeafBatch lb{cc, G, static_cast<tlab::wire::count_t>(in_dim),
                                     std::span<const float>(buf.data(), buf.size())};
            if (auto s = boundary->send(lb); !s) { out.failed = true; out.err = "send: " + s.error().message; return false; }
            ++in_flight;
            return true;
        };
        while (secs_since(t_start) < run_seconds) {
            // Gate A (consolidation): the overcommit controller gates issuance — mirror
            // runner_wire_batched.cpp's `may_issue(tid)` refill gate (one actuation path). ctl==nullptr (no
            // --control-endpoint) leaves the loop byte-unchanged, so the control-off arm is the exact baseline.
            while (!ready.empty() && in_flight < budget && (!ctl || ctl->may_issue(idx)))
                if (!send_group()) return;
            // FORCED-FLUSH LIVENESS FLOOR (ADR-0002 / the lab's gate-everything contract). The gate above
            // is the controller's DISCRETIONARY issue point; like runner_wire_batched.cpp's ungated forced
            // flush, a thread that is fully drained (in_flight==0) yet still has ready work MUST push one
            // group regardless of the gate, so a deny-everything controller cannot starve the thread to
            // exit (the depth-1 floor: a denied thread never deadlocks/terminates). Without this, the
            // greedy episodic pipe — unlike the batched runner — had no floor and a hard-gating method
            // drove in_flight to 0 and broke the loop (the thread exited rc=0 mid-trial). One forced group
            // re-primes the pipe; the gate resumes throttling once in_flight>0 again.
            if (in_flight == 0 && !ready.empty()) {
                if (!send_group()) return;
            }
            if (in_flight == 0) break;   // genuinely no work left (every fiber finished) -> end the window
            auto reply = boundary->recv();
            if (!reply) { out.failed = true; out.err = "recv: " + reply.error().message; return; }
            auto it = corr_to_group.find(reply->corr);
            if (it == corr_to_group.end() || reply->preds.size() != it->second.size()) {
                out.failed = true; out.err = "unmatched/size-mismatch reply corr=" + std::to_string(reply->corr); return;
            }
            for (size_t j = 0; j < it->second.size(); ++j) {
                const int i = it->second[j];
                chocofarm::NetPrediction pred;
                pred.value = reply->preds[j].value;
                pred.logits = std::move(reply->preds[j].logits);
                trees[static_cast<size_t>(i)]->resume_with(pred);            // advance to next leaf (or finish)
                out.leaves += 1;
                if (!trees[static_cast<size_t>(i)]->running) advance(i);     // decision done -> step + re-park
                ready.push_back(i);                                          // running again -> ready to send
            }
            corr_to_group.erase(it);
            --in_flight;
            // Gate A: publish this thread's telemetry — the bridge thread reads it each cadence and REQs the
            // policy engine (issue_engine.py). The identity policy ignores the values (allow-all); a real
            // control policy consumes them. Cheap relaxed atomics, off the forward's critical path.
            if (ctl) ctl->publish(idx, in_flight, static_cast<int>(ready.size()), 0,
                                  static_cast<long>(out.leaves), 0.0);
        }
    } else {
        // ROUND-SYNC pipe: submit every parked fiber's leaf (coalesced into coalesce_rows-row messages),
        // then BLOCK recv'ing the whole round before resuming any -> the search cores idle across the
        // round's RTT. The committed baseline the greedy pipe is measured against.
        while (secs_since(t_start) < run_seconds) {
            std::vector<int> active; active.reserve(static_cast<size_t>(fibers_k));
            for (int i = 0; i < fibers_k; ++i) {
                if (!trees[static_cast<size_t>(i)]->running) advance(i);   // completed -> step the episode
                if (trees[static_cast<size_t>(i)]->running) active.push_back(i);
            }
            if (active.empty()) break;
            if (!drive_round(*boundary, trees, active, coalesce_rows, in_dim, corr, out)) return;
        }
    }
}

// The per-slot rotated gumbel table (the K trees differ only by the rotation) — shared by both engine
// factories so the scripted source is IDENTICAL across A and B at the same slot (ADR-0012 P1).
[[nodiscard]] std::vector<double> slot_table(int i) {
    std::vector<double> table(kGumbelTable.size());
    for (size_t j = 0; j < kGumbelTable.size(); ++j)
        table[j] = kGumbelTable[(j + static_cast<size_t>(i)) % kGumbelTable.size()];
    return table;
}

// OPTION A wrapper: bind run_episodic to the fiber TreeState (the DEFAULT engine — byte-untouched). The
// factory reproduces the original per-slot TreeState construction (cfg, env, rotated table).
void run_thread_fiber_episodic(int idx, const chocofarm::Environment& env, const std::string& endpoint,
                               const chocofarm::GumbelConfig& cfg, double run_seconds, int in_dim,
                               int fibers_k, int coalesce_rows, const std::string& driver,
                               int inflight_msgs, chocofarm::IssueController* ctl, ThreadStat& out) {
    auto make = [&](int i) {
        return std::make_unique<chocofarm::TreeState>(cfg, env, slot_table(i));
    };
    run_episodic<chocofarm::TreeState>(idx, env, endpoint, run_seconds, in_dim, fibers_k,
                                       coalesce_rows, driver, inflight_msgs, ctl, make, out);
}

// OPTION B wrapper: bind run_episodic to the explicit-state CursorSlot (CHOCO_PRODUCER_ENGINE=cursor /
// --engine cursor). The factory builds a CursorSlot per slot (the SAME cfg/env/rotated table) — the
// cursor adapter exposes the SAME .start/.running/.ch.features/.resume_with surface, so the entire
// episodic body (state machine + both pipe shapes + the control gate) is reused (ADR-0012 P1). The
// search the cursor runs is bit-identical to the fiber's run_search (cpp/parity + gumbel_cursor_proto).
void run_thread_cursor_episodic(int idx, const chocofarm::Environment& env, const std::string& endpoint,
                                const chocofarm::GumbelConfig& cfg, double run_seconds, int in_dim,
                                int fibers_k, int coalesce_rows, const std::string& driver,
                                int inflight_msgs, chocofarm::IssueController* ctl, ThreadStat& out) {
    auto make = [&](int i) {
        return std::make_unique<tlab::CursorSlot>(cfg, env, slot_table(i));
    };
    run_episodic<tlab::CursorSlot>(idx, env, endpoint, run_seconds, in_dim, fibers_k,
                                   coalesce_rows, driver, inflight_msgs, ctl, make, out);
}

// One GREEDY-ASYNC fiber producer thread (wire_pool_bench's discipline): keep ~K leaves CONTINUOUSLY in
// flight and process replies as they land — recv ONE, resume that fiber (it computes its next leaf), and
// re-submit it immediately, rather than the round-synchronous barrier (submit all, wait for all). The
// difference is overlap: in round-sync the search cores idle while the whole round's replies are awaited;
// here a thread is always either receiving or computing a fiber's next leaf, so the search cores stay
// busy across the RTT. Same corr->fiber routing and finished-fiber restart as the round-sync arm; the
// ONLY change is the pipeline shape. Optional (selected by --driver greedy) so the round-sync semantics
// are preserved for comparison.
void run_thread_fiber_greedy(int idx, const chocofarm::Environment& env, const std::string& endpoint,
                             const chocofarm::GumbelConfig& cfg, double run_seconds, int in_dim,
                             int fibers_k, ThreadStat& out) {
    tlab::BoundaryConfig bcfg;
    bcfg.endpoint = endpoint;
    bcfg.recv_timeout_ms = 10000;
    bcfg.n_producer_threads = 1;
    bcfg.rows = 1;
    bcfg.in_dim = in_dim;
    auto b = tlab::make_boundary(tlab::BoundaryTopology::PerThread, bcfg);
    if (!b) { out.failed = true; out.err = "boundary: " + b.error().message; return; }
    std::unique_ptr<tlab::Boundary> boundary = std::move(*b);

    const chocofarm::Loc loc{env.entry_point()};
    const chocofarm::Belief bw = env.full_belief();
    const chocofarm::CollectedSet coll;

    std::vector<std::unique_ptr<chocofarm::TreeState>> trees;
    trees.reserve(static_cast<size_t>(fibers_k));
    for (int i = 0; i < fibers_k; ++i) {
        std::vector<double> table(kGumbelTable.size());
        for (size_t j = 0; j < kGumbelTable.size(); ++j)
            table[j] = kGumbelTable[(j + static_cast<size_t>(i)) % kGumbelTable.size()];
        trees.push_back(std::make_unique<chocofarm::TreeState>(cfg, env, std::move(table)));
    }
    for (auto& t : trees) t->start(loc, bw, coll, kLam);

    std::unordered_map<tlab::wire::corr_t, int> corr_to_fiber;
    tlab::wire::corr_t corr = static_cast<tlab::wire::corr_t>(idx) * 1'000'000'000ull + 1ull;
    int in_flight = 0;
    // Submit fiber i's current parked leaf, restarting it on a fresh decision if it had finished. Returns
    // false only on a send error (out.failed set) or if a restarted fiber yielded no leaf (exhausted).
    auto submit = [&](int i) -> bool {
        if (!trees[static_cast<size_t>(i)]->running) {
            out.decisions += 1;
            trees[static_cast<size_t>(i)]->start(loc, bw, coll, kLam);
        }
        if (!trees[static_cast<size_t>(i)]->running) return false;  // produced no leaf (won't happen for n_sims>=1)
        const std::span<const float> feats = trees[static_cast<size_t>(i)]->ch.features;
        const tlab::wire::corr_t cc = corr++;
        const tlab::LeafBatch lb{cc, 1, static_cast<tlab::wire::count_t>(feats.size()), feats};
        if (auto s = boundary->send(lb); !s) { out.failed = true; out.err = "send: " + s.error().message; return false; }
        corr_to_fiber.emplace(cc, i);
        ++in_flight;
        return true;
    };

    for (int i = 0; i < fibers_k; ++i) submit(i);   // prime: K leaves in flight
    if (out.failed) return;

    const auto t_start = SteadyClock::now();
    while (secs_since(t_start) < run_seconds && in_flight > 0) {
        auto reply = boundary->recv();
        if (!reply) { out.failed = true; out.err = "recv: " + reply.error().message; return; }
        auto it = corr_to_fiber.find(reply->corr);
        if (it == corr_to_fiber.end() || reply->preds.empty()) {
            out.failed = true; out.err = "unmatched/empty reply corr=" + std::to_string(reply->corr); return;
        }
        const int i = it->second;
        corr_to_fiber.erase(it);
        --in_flight;
        chocofarm::NetPrediction pred;
        pred.value = reply->preds[0].value;
        pred.logits = std::move(reply->preds[0].logits);
        trees[static_cast<size_t>(i)]->resume_with(pred);   // advances to its next leaf (or finishes)
        out.leaves += 1;
        submit(i);                                          // re-arm this fiber -> back to ~K in flight
        if (out.failed) return;
    }
}
}  // namespace

int main(int argc, char** argv) {
    std::vector<std::string_view> args(argv, argv + argc);
    auto inst_p = opt(args, "--instance"), faces_p = opt(args, "--faces"), ep = opt(args, "--endpoint");
    if (!inst_p || !faces_p || !ep) {
        std::cerr << "usage: tlab-real-producer --instance <p> --faces <p> --endpoint <ipc://...> "
                     "[--threads N --fibers K --msg-rows M --driver round-sync|greedy --engine fiber|cursor --seconds S --n-sims K --m M --in-dim D]\n"
                     "  --fibers 0 (default) = non-fiber baseline; K>=1 = K fibers/thread (the fiber model)\n"
                     "  --engine fiber (default) | cursor = Option A (stackful fiber) vs Option B (explicit-state TreeCursor); cursor requires --fibers K>=1 + --episodic (also via CHOCO_PRODUCER_ENGINE)\n"
                     "  --msg-rows M (default 1) = coalesce up to M round leaves per message (round-sync; the S_min floor)\n"
                     "  --driver round-sync (default) | greedy (keep ~K leaves continuously in flight)\n"
                     "  --inflight-msgs N (default 8) = greedy-episodic pipe depth (coalesced msgs in flight; round-sync ignores)\n"
                     "  --episodic = run real episodes (step the env per executed action); DPS = decisions/s\n"
                     "  --no-early-exit = substitute Terminate so episodes run full-length (cfg.no_early_exit)\n"
                     "  --control-endpoint <ipc://...> = inject the overcommit control plane (issue_engine.py peer);\n"
                     "      --controller-cadence-ms M (default 5) = control-loop tick. Absent = control off (baseline).\n";
        return 2;
    }
    const int threads = opt(args, "--threads") ? std::atoi(std::string(*opt(args, "--threads")).c_str()) : 3;
    const double seconds = opt(args, "--seconds") ? std::atof(std::string(*opt(args, "--seconds")).c_str()) : 5.0;
    const int in_dim = opt(args, "--in-dim") ? std::atoi(std::string(*opt(args, "--in-dim")).c_str()) : 241;
    // --fibers K: 0 (default) = NON-FIBER baseline (one SerialRuntime/thread, B=1 blocking); K>=1 = the
    // FIBER model (K TreeState fibers/thread multiplexed, K leaves in flight -> server batch grows with K).
    const int fibers = opt(args, "--fibers") ? std::atoi(std::string(*opt(args, "--fibers")).c_str()) : 0;
    // --msg-rows M: coalesce up to M of a fiber round's parked leaves into ONE B<=M message (the static
    // coalescing floor). M=1 (default) = one leaf per message (the historical B=1 path). Round-sync only.
    const int msg_rows = opt(args, "--msg-rows") ? std::atoi(std::string(*opt(args, "--msg-rows")).c_str()) : 1;
    // --inflight-msgs N: the greedy-episodic pipe depth -- up to N coalesced messages kept CONTINUOUSLY in
    // flight (the overlap budget; ignored by round-sync and by the root drivers). Default 8 (the overcommit
    // reference's --inflight-msgs). The within-stack greedy-vs-round-sync A/B holds coalesce_rows fixed and
    // moves only this (and the driver) so the pipe shape is attributed in isolation.
    const int inflight_msgs = opt(args, "--inflight-msgs") ? std::atoi(std::string(*opt(args, "--inflight-msgs")).c_str()) : 8;
    // --episodic: each fiber runs a SEQUENCE of decisions (env stepped by the executed action) instead of
    // repeated root decisions -> the production-shape workload; DPS = decisions/wall. --no-early-exit sets
    // cfg.no_early_exit so a Terminate is substituted (episodes run full-length, not short-circuited).
    bool episodic = false, no_early_exit = false;
    for (const auto& a : args) { if (a == "--episodic") episodic = true; if (a == "--no-early-exit") no_early_exit = true; }
    // --driver: how the FIBER arm pipelines leaves. round-sync (default, preserves the committed
    // semantics) submits a whole round then awaits it; greedy keeps ~K leaves continuously in flight.
    const std::string driver = opt(args, "--driver") ? std::string(*opt(args, "--driver")) : "round-sync";
    if (driver != "round-sync" && driver != "greedy") {
        std::cerr << "tlab-real-producer: --driver must be round-sync|greedy, got " << driver << "\n";
        return 2;
    }
    // --engine fiber (DEFAULT) | cursor: which per-slot resumable-search ENGINE multiplexes the K trees.
    //   fiber  = Option A (chocofarm::TreeState, a boost.context stackful fiber — the byte-untouched baseline).
    //   cursor = Option B (tlab::CursorSlot, the explicit-state TreeCursor — no fiber, runs on the thread
    //            stack). Bit-identical search to the fiber (cpp/parity + gumbel_cursor_proto). The engine is
    //            ADDITIVE and selectable; the default is fiber so the baseline is unchanged (ADR-0004). Also
    //            settable via CHOCO_PRODUCER_ENGINE (the env overrides the default, the flag overrides the env).
    //   B is wired ONLY into the EPISODIC path (the production-shape workload the e2e measures); the root /
    //   non-episodic drivers stay fiber-only (a loud reject if cursor is asked for there — never a silent
    //   fallback, ADR-0002).
    std::string engine = "fiber";
    if (const char* e = std::getenv("CHOCO_PRODUCER_ENGINE")) engine = e;
    if (auto v = opt(args, "--engine")) engine = std::string(*v);
    if (engine != "fiber" && engine != "cursor") {
        std::cerr << "tlab-real-producer: --engine must be fiber|cursor, got " << engine << "\n";
        return 2;
    }
    if (engine == "cursor" && !(opt(args, "--fibers") && episodic)) {
        std::cerr << "tlab-real-producer: --engine cursor requires --fibers K>=1 AND --episodic (the "
                     "cursor engine is wired into the EPISODIC multiplexed path only — the production-shape "
                     "workload). Got fibers/episodic absent. Refusing rather than silently falling back to "
                     "the fiber engine (ADR-0002).\n";
        return 2;
    }
    // Renice ONE generator thread DOWN (the generator-bound core-sharing lever): a 4th generator on the
    // server's core, reniced, soaks its idle slack but yields to the server's forward.
    const int low_prio_thread = opt(args, "--low-prio-thread")
        ? std::atoi(std::string(*opt(args, "--low-prio-thread")).c_str()) : -1;
    const int low_prio_nice = opt(args, "--low-prio-nice")
        ? std::atoi(std::string(*opt(args, "--low-prio-nice")).c_str()) : 0;
    chocofarm::GumbelConfig cfg;
    if (auto v = opt(args, "--n-sims")) cfg.n_sims = std::atoi(std::string(*v).c_str());
    if (auto v = opt(args, "--m")) cfg.m = std::atoi(std::string(*v).c_str());
    cfg.no_early_exit = no_early_exit;   // HPO/benchmark-only: substitute Terminate so episodes run full

    auto inst = chocofarm::load_instance(*inst_p, *faces_p);
    if (!inst) { std::cerr << "tlab-real-producer: FATAL: " << inst.error().message << "\n"; return 1; }
    chocofarm::Environment env(*inst);

    std::cout << "tlab-real-producer: generator=real(" << (fibers > 0 ? "fiber" : "non-fiber")
              << ") engine=" << engine
              << " driver=" << (fibers > 0 ? driver : std::string("n/a"))
              << " threads=" << threads << " fibers_per_thread=" << fibers << " msg_rows=" << msg_rows
              << " episodic=" << (episodic ? 1 : 0) << " no_early_exit=" << (cfg.no_early_exit ? 1 : 0)
              << " inflight_msgs=" << inflight_msgs
              << " seconds=" << seconds << " n_sims=" << cfg.n_sims << " m=" << cfg.m
              << " n_slots=" << chocofarm::n_action_slots(env) << " endpoint=" << *ep << "\n";

    // Gate A (consolidation): optionally inject the control plane — a process-shared IssueController + an
    // IssueControlBridge that REQs the Python policy engine (issue_engine.py) every cadence and applies the
    // returned per-thread allow bits. REUSES the cpp/include/chocofarm headers runner_wire_batched uses (one
    // home, ADR-0012 P1). No --control-endpoint => ctl stays null => the producer is byte-unchanged (the
    // control-off baseline the perf-hold A/B measures against; prereg gateA-control-plane-perf-hold).
    const auto control_ep = opt(args, "--control-endpoint");
    const double cadence_ms = opt(args, "--controller-cadence-ms")
        ? std::atof(std::string(*opt(args, "--controller-cadence-ms")).c_str()) : 5.0;
    std::unique_ptr<chocofarm::IssueController> ctl;
    std::unique_ptr<chocofarm::IssueControlBridge> bridge;
    if (control_ep) {
        ctl = std::make_unique<chocofarm::IssueController>(threads, inflight_msgs);
        bridge = std::make_unique<chocofarm::IssueControlBridge>(ctl.get(), std::string(*control_ep), cadence_ms);
        bridge->start();
        std::cout << "tlab-real-producer: control plane ON (endpoint=" << *control_ep
                  << " cadence_ms=" << cadence_ms << ")\n";
    }

    // ADMISSION GUARD (RCA tlab_finding #23; ADR-0000 make-the-illegal-state-unrepresentable + ADR-0002
    // a config the receiver can't honor must not be silently accepted). The fiber driver keeps EVERY fiber's
    // Gumbel tree live simultaneously, so the producer's resident floor is threads*fibers*est_tree_bytes
    // (non-fiber: threads*est_tree_bytes — one tree per thread). At --fibers 1024 --n-sims 256 that is
    // ~3.4 GiB for ONE producer; four concurrent producers exceed an 8 GiB box and the OOM killer fires an
    // opaque rc=-9 mid-run (and a single producer's throughput halves under the memory-pressure regime as
    // its RSS climbs through 3+ GiB). Reject that footprint LOUDLY up front — naming the estimate and the
    // available memory — instead of letting it SIGKILL. Only a SINGLE producer's own footprint is checked
    // here (a process can't see its siblings); the >50% bar leaves headroom for the server + a second
    // producer, and the operator running N producers divides accordingly. Skipped only if MemAvailable is
    // unreadable (a typed absence, never a guess).
    const std::uint64_t live_trees =
        static_cast<std::uint64_t>(threads) * (fibers > 0 ? static_cast<std::uint64_t>(fibers) : 1ull);
    const std::uint64_t per_fiber = est_fiber_resident_bytes(cfg.n_sims);
    const std::uint64_t est_resident = live_trees * per_fiber;
    std::cout << "tlab-real-producer: est_resident=" << (est_resident >> 20) << " MiB ("
              << live_trees << " live trees x " << (per_fiber >> 10) << " KiB)\n";
    if (const auto avail = mem_available_bytes()) {
        if (est_resident > *avail / 2) {
            std::cerr << "tlab-real-producer: FATAL: estimated resident " << (est_resident >> 20)
                      << " MiB (" << live_trees << " live trees x " << (per_fiber >> 10)
                      << " KiB/tree at n_sims=" << cfg.n_sims << ") exceeds 50% of MemAvailable "
                      << (*avail >> 20) << " MiB. Reduce --fibers, --threads, or --n-sims (resident scales "
                         "linearly with each). Refusing rather than risk an OOM SIGKILL mid-run.\n";
            return 1;
        }
    }

    std::vector<ThreadStat> stats(static_cast<size_t>(threads));
    std::vector<std::thread> pool;
    const std::string endpoint(*ep);
    const auto t0 = SteadyClock::now();
    for (int i = 0; i < threads; ++i)
        pool.emplace_back([&, i] {
            apply_thread_priority(i, low_prio_thread, low_prio_nice);   // renice iff designated
            if (fibers > 0 && episodic && engine == "cursor")
                run_thread_cursor_episodic(i, env, endpoint, cfg, seconds, in_dim, fibers, msg_rows, driver, inflight_msgs, ctl.get(), stats[static_cast<size_t>(i)]);
            else if (fibers > 0 && episodic)
                run_thread_fiber_episodic(i, env, endpoint, cfg, seconds, in_dim, fibers, msg_rows, driver, inflight_msgs, ctl.get(), stats[static_cast<size_t>(i)]);
            else if (fibers > 0 && driver == "greedy")
                run_thread_fiber_greedy(i, env, endpoint, cfg, seconds, in_dim, fibers, stats[static_cast<size_t>(i)]);
            else if (fibers > 0)
                run_thread_fiber(i, env, endpoint, cfg, seconds, in_dim, fibers, msg_rows, stats[static_cast<size_t>(i)]);
            else
                run_thread(i, env, endpoint, cfg, seconds, in_dim, stats[static_cast<size_t>(i)]);
        });
    for (auto& th : pool) th.join();
    if (bridge) {
        bridge->stop();   // join the control thread before reporting (a failed bridge is loud, ADR-0002)
        if (bridge->failed())
            std::cerr << "tlab-real-producer: control bridge FAILED: " << bridge->error() << "\n";
    }
    const double wall = secs_since(t0);

    std::uint64_t dec = 0, leaves = 0; bool any_fail = false;
    for (const auto& s : stats) {
        dec += s.decisions; leaves += s.leaves;
        if (s.failed) { any_fail = true; std::cerr << "  thread failed: " << s.err << "\n"; }
    }
    const double dps = wall > 0 ? static_cast<double>(dec) / wall : 0.0;
    const double lps = wall > 0 ? static_cast<double>(leaves) / wall : 0.0;
    std::cout << "REAL-AGG threads=" << threads << " fibers=" << fibers << " engine=" << engine
              << " msg_rows=" << msg_rows
              << " episodic=" << (episodic ? 1 : 0) << " no_early_exit=" << (cfg.no_early_exit ? 1 : 0)
              << " driver=" << (fibers > 0 ? driver : std::string("n/a")) << " wall_s=" << wall
              << " control=" << (control_ep ? 1 : 0)
              << " decisions=" << dec << " leaves=" << leaves
              << " decisions_per_sec=" << dps << " leaves_per_sec=" << lps
              << " any_fail=" << (any_fail ? 1 : 0) << "\n";
    return any_fail ? 1 : 0;
}
