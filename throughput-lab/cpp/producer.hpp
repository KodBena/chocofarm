// throughput-lab/cpp/producer.hpp
// Purpose: the PRODUCER seam — the calibrated synthetic-load generator and its two MODES
//   (DECOUPLED / COUPLED, a plug per ADR-0012 P8, not the only modes the typed seam admits). Each
//   producer thread CALIBRATES its own compute rate (a timed x+=1 spin -> ops/sec), then emits
//   synthetic leaf-batches (rows of 241 float32 — the Stage-A in_dim) at a KNOWN, DIALABLE rate by
//   doing a calibrated amount of x+=1 busy-work per production (hardware-calibrated, NOT a fragile
//   usleep). This header defines the calibration protocol + the producer-mode contract + the config;
//   it implements NOTHING (the build agent supplies producer.cpp).
// Public Domain (The Unlicense).
//
// ================================================================================================
//  THE CALIBRATION PROTOCOL (per thread, at start — the "rate is hardware-calibrated, not a sleep")
// ================================================================================================
//  GOAL: convert a desired EMISSION RATE (target leaf-batches per second) into a COUNT of x+=1
//  busy-work iterations to burn between emissions, so the rate holds across hardware without relying
//  on usleep granularity/jitter (a usleep floor of ~50-100us would itself cap throughput and add
//  jitter — exactly the artifact this testbed must not introduce).
//
//  STEP 1 — CALIBRATE ops/sec. Spin a timed loop of the literal "task": `x += 1` on a volatile (or
//    otherwise un-elided) integer accumulator, for a fixed wall-clock window (e.g. ~200 ms) measured
//    with std::chrono::steady_clock. ops_per_sec = iterations_completed / elapsed_seconds. Run a
//    short warmup spin first (discarded) so the CPU is at frequency before the timed window. Use a
//    sink the optimizer cannot prove dead (volatile, an atomic, or a value folded into the emitted
//    payload) so the loop is not deleted — a deleted spin calibrates to "infinity" and the rate
//    control collapses (ADR-0002: a silently-elided calibration is a silently-wrong rate).
//
//  STEP 2 — DERIVE busy-work-per-production. For a target emission rate R (batches/sec) on a thread
//    whose calibrated rate is ops_per_sec, the inter-emission compute budget is:
//        seconds_between_emissions = 1.0 / R
//        ops_between_emissions     = ops_per_sec * seconds_between_emissions
//    Subtract a measured estimate of the fixed per-emission overhead (the feature-fill + encode +
//    send) so the spin fills only the REMAINING budget — otherwise the achieved rate undershoots R.
//    If ops_between_emissions <= 0 after subtracting overhead, the thread is overhead-bound at R
//    (the requested rate exceeds what one thread can emit) — report that honestly (the achieved rate
//    is then the overhead-bound ceiling), do not silently spin zero and pretend R was met.
//
//  STEP 3 — EMIT AT RATE. Each production: fill the (B, in_dim) row(s), submit through the Boundary,
//    then burn `ops_between_emissions` x+=1 iterations (the calibrated spin) before the next. The
//    achieved rate is MEASURED (productions / wall-time) and reported alongside the requested R, so a
//    gap between requested and achieved is visible, never hidden (ADR-0009 measure-honesty).
//
//  The calibration is PER THREAD because per-core frequency/scheduling differ; each thread carries
//  its own ops_per_sec and its own derived spin count.

#pragma once

#include <cstdint>
#include <expected>
#include <string>

#include "boundary.hpp"   // tlab::Boundary, BoundaryError, BoundaryTopology, BoundaryConfig
#include "wire.hpp"        // tlab::wire — STAGE_A_IN_DIM, count_t

namespace tlab {

// ---- the producer mode plug (ADR-0012 P8: the closed vocabulary of modes; BOTH are built) -------
//  DECOUPLED: free-run at the calibrated rate. Replies ARE received and their latency measured, but
//    they do NOT gate production (the next batch is emitted on the calibrated clock regardless of
//    whether prior replies have landed). Models a producer that is NOT on the reply's critical path.
//  COUPLED: wait for each batch's reply before producing more (request -> reply -> produce), emulating
//    the real search's leaf-eval-ON-the-critical-path. Production rate is then bounded by the
//    round-trip, and the calibrated spin sits BETWEEN receiving a reply and emitting the next batch
//    (the "think time" budget), so the requested rate caps but the RTT can lower the achieved rate.
//  enum class (P9 — scoped).
enum class ProducerMode {
    Decoupled,   // free-run at the calibrated rate; replies measured but non-gating
    Coupled,     // block on each reply before the next production (leaf-eval on the critical path)
};

// ---- the result of STEP 1 calibration (a thread's measured compute rate) ------------------------
struct Calibration {
    double ops_per_sec = 0.0;       // measured x+=1 iterations per second (STEP 1)
    std::uint64_t warmup_ops = 0;   // iterations burned in the discarded warmup (for the record)
    std::uint64_t timed_ops = 0;    // iterations in the timed window (the basis of ops_per_sec)
    double timed_seconds = 0.0;     // the timed window's measured wall-clock duration
};

// ---- per-thread emission + latency measurements (reported at shutdown; ADR-0009 honesty) --------
// All rates MEASURED, never assumed. `requested_rate_hz` is what was asked; `achieved_rate_hz` is
// what was delivered (productions / wall-time) — a gap is visible. Latency stats are over the
// received replies (every reply is measured in BOTH modes; in DECOUPLED they just don't gate).
struct ProducerStats {
    Calibration calib;
    double requested_rate_hz = 0.0;
    double achieved_rate_hz = 0.0;     // productions / wall-seconds (the honest delivered rate)
    std::uint64_t batches_sent = 0;
    std::uint64_t replies_recv = 0;
    double mean_reply_latency_us = 0.0;   // mean producer-side send->reply round-trip, microseconds
    double p50_reply_latency_us = 0.0;
    double p99_reply_latency_us = 0.0;
    bool overhead_bound = false;       // STEP 2: requested rate exceeded one thread's emit ceiling
};

// ---- the producer run configuration (the dialable knobs) ----------------------------------------
struct ProducerConfig {
    int n_threads = 1;                          // producer threads (each self-calibrates, STEP 1)
    ProducerMode mode = ProducerMode::Decoupled;
    BoundaryTopology topology = BoundaryTopology::PerThread;
    double target_rate_hz = 1000.0;             // requested per-thread emission rate R (batches/sec)
    wire::count_t rows_per_batch = 1;           // B per submitted batch (1 = degenerate single-leaf)
    wire::count_t in_dim = wire::STAGE_A_IN_DIM;// feature width per row (241 on the live env)
    double run_seconds = 5.0;                   // measured run duration (STEP 3 emission window)
    double calib_window_seconds = 0.2;          // STEP 1 timed-spin window
    std::string endpoint = "ipc:///tmp/tlab-infer.sock";   // the server's ZMQ ipc endpoint
    int recv_timeout_ms = 5000;                 // bounds Boundary recv()/poll() (loud timeout, P5)
    std::size_t send_queue_bytes = 256ull << 20;// TOTAL outstanding-send byte budget (back-pressure cap; <=1G)

    // ---- per-thread scheduling priority (the "renice ONE generator thread" lever) ----------------
    // A single designated generator thread can be run at a LOWER scheduling priority than its peers (and
    // than the nice-0 inference server). The intent: when that generator SHARES a core with the inference
    // server (or with its higher-priority peers), it YIELDS — so the forward finishes and its reply is
    // read, and the other generators run on unaffected. On Linux `nice` is PER-TASK, so this is one
    // thread's nice via setpriority(gettid()), not the whole process (that would be a uniform process
    // nice — a different, coarser lever). nice is graceful/weighted: the reniced thread still runs in the
    // slack, it just cedes under contention (vs SCHED_IDLE's binary starve, which could collapse the feed).
    int low_prio_thread = -1;                   // index of the ONE generator thread to renice (-1 = none)
    int low_prio_nice = 0;                       // its nice value (>0 = lower priority; 0 = no-op)
};

// Run the producer: stand up the Boundary (per `cfg.topology`/`cfg.endpoint`), spawn `cfg.n_threads`
// producer threads (each calibrates per STEP 1, derives its spin per STEP 2, emits per STEP 3 in
// `cfg.mode`), run for `cfg.run_seconds`, then join and aggregate. Returns one ProducerStats per
// thread, or a typed BoundaryError if the transport could not be established (P9: fallible -> a typed
// return, never a thrown escape across this seam). The build agent IMPLEMENTS this in producer.cpp.
[[nodiscard]] std::expected<std::vector<ProducerStats>, BoundaryError> run_producer(
        const ProducerConfig& cfg);

}  // namespace tlab
