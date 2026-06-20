// cpp/include/chocofarm/issue_controller.hpp
// Purpose: the ONLINE ISSUE CONTROLLER fixture — the in-process ACTUATION + MARSHALLING HUB (HPO/benchmark;
//   default-OFF => the wire path is byte-unchanged when no controller is injected). Its primitive decision
//   is a per-thread ISSUE allow/deny predicate — `may_issue(tid)` — consulted by
//   run_episodes_wire_pipelined's refill() at the single non-forced-issue point: the gate is
//   `inflight_msgs < D && may_issue(tid)`. The fixed `D` stays the runner's own SAFETY ceiling; the
//   controller only allows or denies the next discretionary issue. "Overcommit" (in-flight depth > 1) is a
//   DOWNSTREAM emergent property of allowed issues + the in-flight history — NOT the action (ADR-0008
//   precise vocabulary / ADR-0012 honest naming).
//
//   POLICY LIVES IN PYTHON; C++ ONLY ACTUATES. This hub is deliberately PASSIVE — no control thread, no
//   policy here. The slow-and-smart POLICY ENGINE is an external Python process; the fast-and-dumb GATE is
//   these in-process atomics. The two are bridged by issue_control_bridge.hpp over a ZeroMQ control socket
//   (ADR-0012 P7: a MESSAGING FABRIC carries coordination/streaming — a live control loop is exactly the
//   "bytes-store as a sync primitive" smell P7 forbids, so NOT redis-key-polling). The hub exposes
//   snapshot_features() (the bridge reads, serialises, sends) and set_allow() (the bridge writes from the
//   engine's reply); the runner's refill reads may_issue() (one relaxed atomic load — the hot path).
//
//   Discipline (ADR-0012): P2 — the issue decision is an injected port; the runner depends on this hub's
//   actuation (may_issue) + metrics sink (publish), never on the policy. P3 — one owner: this hub owns ONLY
//   the actuation+marshalling cells; the ZMQ coordination is issue_control_bridge's, the policy is Python's.
//   P5 / ADR-0002 — the controller gates ONLY the non-forced refill path; the forced-flush backstop stays
//   UNGATED (the depth-1 liveness floor, a denied thread never deadlocks). P9 — single-writer-per-cell (each
//   worker writes only its own metrics slot; the bridge writes only the allow bits), race-free by
//   construction. The CONTROL PATH (snapshot->send->policy->recv->set_allow) is latency/jitter sensitive —
//   its realtime behaviour feeds back into the policy's own prediction quality — hence binary frames + a
//   low-latency fabric, never JSON.
// Public Domain (The Unlicense).
#pragma once

#include <atomic>
#include <memory>
#include <vector>

namespace chocofarm {

// The marshalled observation the policy sees (the "given features"). Per-thread vectors (size T) + a couple
// aggregate/const scalars. This is the FEATURE SURFACE the policy work extends; fields whose source channel
// does not exist yet ride a documented sentinel (server_rows_per_forward — the server->producer metrics
// channel is the one gap — and per-thread mean_rtt_ms, both 0 until wired).
struct IssueFeatures {
    int n_threads = 0;
    int d_ceiling = 1;                     // the runner's fixed D safety ceiling — read-only context (the
                                           // controller does NOT set/micromanage it)
    std::vector<int> inflight;             // outstanding (submitted, unanswered) messages per thread
    std::vector<int> ready;                // ready (parked-at-leaf, unsubmitted) slots per thread
    std::vector<long> msgs;                // cumulative messages issued per thread
    std::vector<long> leaves;              // cumulative leaves sent per thread
    std::vector<double> mean_rtt_ms;       // recent mean reply RTT per thread (0 until wired)
    double server_rows_per_forward = 0.0;  // (metrics-channel gap; 0 until the server->producer channel exists)
};

class IssueController {
public:
    // T = producer threads; d_ceiling = the runner's fixed D (a read-only FEATURE for the policy; the
    // controller does not enforce it — the runner's `inflight < D` does). Default actuation = ALL ALLOW
    // (=> the gate reduces to `inflight < D`, byte-identical to the fixed-D runner, until the engine acts).
    IssueController(int n_threads, int d_ceiling)
        : T_(n_threads > 0 ? n_threads : 1),
          d_ceiling_(d_ceiling > 0 ? d_ceiling : 1),
          slots_(std::make_unique<Slot[]>(static_cast<size_t>(T_))),
          allow_(std::make_unique<std::atomic<int>[]>(static_cast<size_t>(T_))) {
        for (int t = 0; t < T_; ++t) allow_[static_cast<size_t>(t)].store(1, std::memory_order_relaxed);
    }
    IssueController(const IssueController&) = delete;
    IssueController& operator=(const IssueController&) = delete;

    [[nodiscard]] int n_threads() const { return T_; }
    [[nodiscard]] int d_ceiling() const { return d_ceiling_; }

    // HOT path (the runner's refill non-forced gate): may thread `tid` issue the next discretionary message?
    // One relaxed load. A stale value is a benign control signal. Does NOT gate the forced flush (the
    // ungated liveness floor).
    [[nodiscard]] bool may_issue(int tid) const {
        return allow_[static_cast<size_t>(tid)].load(std::memory_order_relaxed) != 0;
    }

    // The worker publishes its per-thread metrics — single-writer per `tid`, no lock, no race (relaxed).
    void publish(int tid, int inflight, int ready, long msgs, long leaves, double mean_rtt_ms = 0.0) {
        Slot& s = slots_[static_cast<size_t>(tid)];
        s.inflight.store(inflight, std::memory_order_relaxed);
        s.ready.store(ready, std::memory_order_relaxed);
        s.msgs.store(msgs, std::memory_order_relaxed);
        s.leaves.store(leaves, std::memory_order_relaxed);
        s.rtt_us.store(static_cast<long long>(mean_rtt_ms * 1000.0), std::memory_order_relaxed);
    }

    // The bridge (control thread) reads the published metrics into an IssueFeatures to serialise + send.
    [[nodiscard]] IssueFeatures snapshot_features() const {
        IssueFeatures f;
        f.n_threads = T_;
        f.d_ceiling = d_ceiling_;
        f.inflight.resize(static_cast<size_t>(T_));
        f.ready.resize(static_cast<size_t>(T_));
        f.msgs.resize(static_cast<size_t>(T_));
        f.leaves.resize(static_cast<size_t>(T_));
        f.mean_rtt_ms.resize(static_cast<size_t>(T_));
        for (int t = 0; t < T_; ++t) {
            const Slot& s = slots_[static_cast<size_t>(t)];
            const size_t i = static_cast<size_t>(t);
            f.inflight[i] = s.inflight.load(std::memory_order_relaxed);
            f.ready[i] = s.ready.load(std::memory_order_relaxed);
            f.msgs[i] = s.msgs.load(std::memory_order_relaxed);
            f.leaves[i] = s.leaves.load(std::memory_order_relaxed);
            f.mean_rtt_ms[i] = static_cast<double>(s.rtt_us.load(std::memory_order_relaxed)) / 1000.0;
        }
        return f;
    }

    // The bridge writes the engine's per-thread allow decision back into the actuation cell (any nonzero
    // => allow). Single-writer (the bridge thread); relaxed.
    void set_allow(int tid, bool allow) {
        allow_[static_cast<size_t>(tid)].store(allow ? 1 : 0, std::memory_order_relaxed);
    }

private:
    struct Slot {
        std::atomic<int> inflight{0};
        std::atomic<int> ready{0};
        std::atomic<long> msgs{0};
        std::atomic<long> leaves{0};
        std::atomic<long long> rtt_us{0};
    };

    int T_;
    int d_ceiling_;
    std::unique_ptr<Slot[]> slots_;
    std::unique_ptr<std::atomic<int>[]> allow_;   // the actuation surface: per-thread issue-allow bits
};

}  // namespace chocofarm
