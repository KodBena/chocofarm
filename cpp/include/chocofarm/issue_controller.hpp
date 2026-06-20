// cpp/include/chocofarm/issue_controller.hpp
// Purpose: the ONLINE ISSUE CONTROLLER fixture (HPO/benchmark; default-OFF → the wire path is
//   byte-unchanged when no controller is injected). Its primitive decision is a per-thread ISSUE
//   allow/deny predicate — `may_issue(tid)` — consulted by run_episodes_wire_pipelined's refill() at the
//   single non-forced-issue point: the gate becomes `inflight_msgs < D && may_issue(tid)`. The fixed `D`
//   stays the runner's own SAFETY ceiling (never micromanaged here); the controller only allows or denies
//   the next discretionary issue. The aggregate "overcommit" (in-flight depth > 1) is therefore a
//   DOWNSTREAM, emergent property — a function of the allowed issues AND the in-flight history — not a
//   thing the controller decides directly. Naming it an *issue* decision (not an "overcommit" decision) is
//   the ADR-0008 precise-vocabulary / ADR-0012 honest-signature discipline: when the gate branch is taken
//   the thread issues a message, and that issue is an overcommit ONLY if a message was already in flight —
//   a fact the predicate does not (and should not) assert.
//
//   Each control tick the controller MARSHALS the per-thread + aggregate metrics into an `IssueFeatures`
//   observation, calls a swappable POLICY, and writes back the per-thread allow bits. The POLICY (the
//   boolean-per-thread classifier vs softmax-regression decision over the features) is designed SEPARATELY;
//   this header is only the plumbing it plugs into. The DEFAULT policy is identity (every thread allowed ⇒
//   the gate reduces to `inflight < D`, byte-identical to the fixed-D runner — the regression baseline of
//   the seam, the way RandomPolicy validates env↔Policy).
//
//   Discipline (ADR-0012): P2 — the issue decision is an injected port (the controller); the runner depends
//   on the actuation surface (the allow read) + the metrics sink (publish), never on the policy
//   implementation. P5 / ADR-0002 — the controller gates ONLY the non-forced refill path; the forced-flush
//   backstop stays UNGATED (the liveness floor — a denied thread can never deadlock, its worst case is the
//   depth-1 the forced flush guarantees). P9 — single-writer-per-cell (each worker writes only its own
//   metrics slot; the control thread writes only the allow bits), so the marshalling is race-free by
//   construction, mirroring single-writer-per-tree. Hot path is one relaxed atomic load.
// Public Domain (The Unlicense).
#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace chocofarm {

// The marshalled observation the policy sees each control tick (the "given features"). Per-thread vectors
// (size T) + aggregate scalars. This is the FEATURE SURFACE the policy work extends; fields whose source
// channel does not exist yet ride a documented sentinel (server_rows_per_forward — the server→producer
// metrics channel is the one gap from the design — and per-thread mean_rtt_ms, both 0 until wired).
struct IssueFeatures {
    int n_threads = 0;
    int d_ceiling = 1;                     // the runner's fixed D safety ceiling — read-only context for the
                                           // policy (it is NOT set/micromanaged by the controller)
    std::vector<int> inflight;             // outstanding (submitted, unanswered) messages per thread
    std::vector<int> ready;                // ready (parked-at-leaf, unsubmitted) slots per thread
    std::vector<long> msgs;                // cumulative messages issued per thread
    std::vector<long> leaves;              // cumulative leaves sent per thread
    std::vector<double> mean_rtt_ms;       // recent mean reply RTT per thread (0 until wired)
    double server_rows_per_forward = 0.0;  // (metrics-channel gap; 0 until the server→producer channel exists)
};

// The policy seam: features in, per-thread ISSUE-allow bits out (nonzero = allow the next discretionary
// issue, 0 = deny it). DEFAULT = identity (all allow ⇒ byte-unchanged). How the policy derives the bits —
// a boolean-per-thread classifier, or a softmax regression thresholded into bits — is the policy's own
// concern (the later workflow); the actuation the runner reads is always the per-thread allow bit.
using IssuePolicy = std::function<void(const IssueFeatures&, std::vector<char>& allow_out)>;

class IssueController {
public:
    // T = producer threads; d_ceiling = the runner's fixed D (passed only as a read-only FEATURE for the
    // policy — the controller does not enforce it; the runner's `inflight < D` does). cadence_ms = the SLOW
    // control-tick period (the hot allow-read is independent of it). A null policy ⇒ identity (all allow).
    IssueController(int n_threads, int d_ceiling, double cadence_ms, IssuePolicy policy = {})
        : T_(n_threads > 0 ? n_threads : 1),
          d_ceiling_(d_ceiling > 0 ? d_ceiling : 1),
          cadence_ms_(cadence_ms > 0.0 ? cadence_ms : 5.0),
          policy_(std::move(policy)),
          slots_(std::make_unique<Slot[]>(static_cast<size_t>(T_))),
          allow_(std::make_unique<std::atomic<int>[]>(static_cast<size_t>(T_))) {
        for (int t = 0; t < T_; ++t) allow_[static_cast<size_t>(t)].store(1, std::memory_order_relaxed);
    }
    ~IssueController() { stop(); }
    IssueController(const IssueController&) = delete;
    IssueController& operator=(const IssueController&) = delete;

    // HOT path (the runner's refill non-forced gate): may thread `tid` issue the next discretionary message?
    // One relaxed load. A stale value is a benign control signal (the predicate is advisory, re-read every
    // refill). This does NOT gate the forced flush — that stays the ungated liveness floor.
    [[nodiscard]] bool may_issue(int tid) const {
        return allow_[static_cast<size_t>(tid)].load(std::memory_order_relaxed) != 0;
    }

    // The worker publishes its per-thread metrics — single-writer per `tid`, so no lock and no race
    // (relaxed stores; the control thread reads them on its own cadence).
    void publish(int tid, int inflight, int ready, long msgs, long leaves, double mean_rtt_ms = 0.0) {
        Slot& s = slots_[static_cast<size_t>(tid)];
        s.inflight.store(inflight, std::memory_order_relaxed);
        s.ready.store(ready, std::memory_order_relaxed);
        s.msgs.store(msgs, std::memory_order_relaxed);
        s.leaves.store(leaves, std::memory_order_relaxed);
        s.rtt_us.store(static_cast<long long>(mean_rtt_ms * 1000.0), std::memory_order_relaxed);
    }

    // Spawn the control thread (tick = marshal features → policy → write allow bits, every cadence_ms). An
    // identity policy leaves every bit at allow, so start()/stop() around a run is a no-op on behaviour.
    void start() {
        if (running_) return;
        running_ = true;
        stop_.store(false, std::memory_order_relaxed);
        thread_ = std::thread([this] {
            std::unique_lock<std::mutex> lk(mu_);
            while (!stop_.load(std::memory_order_relaxed)) {
                cv_.wait_for(lk, std::chrono::duration<double, std::milli>(cadence_ms_),
                             [this] { return stop_.load(std::memory_order_relaxed); });
                if (stop_.load(std::memory_order_relaxed)) break;
                lk.unlock();
                tick_();
                lk.lock();
            }
        });
    }

    void stop() {
        if (!running_) return;
        {
            std::lock_guard<std::mutex> lk(mu_);
            stop_.store(true, std::memory_order_relaxed);
        }
        cv_.notify_all();
        if (thread_.joinable()) thread_.join();
        running_ = false;
    }

private:
    struct Slot {
        std::atomic<int> inflight{0};
        std::atomic<int> ready{0};
        std::atomic<long> msgs{0};
        std::atomic<long> leaves{0};
        std::atomic<long long> rtt_us{0};
    };

    // One control iteration: read the published slots into an IssueFeatures, run the policy (identity if
    // none), and store the per-thread allow bits (any nonzero ⇒ allow). The slow loop — never on the hot path.
    void tick_() {
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
        std::vector<char> out(static_cast<size_t>(T_), char{1});   // identity baseline: all allow
        if (policy_) policy_(f, out);
        for (int t = 0; t < T_; ++t) {
            const bool allow = (static_cast<size_t>(t) < out.size()) ? (out[static_cast<size_t>(t)] != 0) : true;
            allow_[static_cast<size_t>(t)].store(allow ? 1 : 0, std::memory_order_relaxed);
        }
    }

    int T_;
    int d_ceiling_;
    double cadence_ms_;
    IssuePolicy policy_;
    std::unique_ptr<Slot[]> slots_;
    std::unique_ptr<std::atomic<int>[]> allow_;   // the actuation surface: per-thread issue-allow bits
    bool running_ = false;
    std::atomic<bool> stop_{false};
    std::mutex mu_;
    std::condition_variable cv_;
    std::thread thread_;
};

}  // namespace chocofarm
