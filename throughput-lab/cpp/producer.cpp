// throughput-lab/cpp/producer.cpp
// Purpose: run_producer — the calibrated synthetic-load generator (producer.hpp's seam). It stands up the
//   Boundary per cfg.topology, spawns cfg.n_threads producer threads, and each thread: (STEP 1) calibrates
//   its own compute rate with a timed x+=1 spin, (STEP 2) derives the busy-work iterations to burn between
//   emissions to hit cfg.target_rate_hz, and (STEP 3) emits synthetic (rows_per_batch, in_dim) leaf-batches
//   at that rate in cfg.mode (DECOUPLED free-run / COUPLED block-on-reply), measuring the ACHIEVED rate and
//   per-reply latency. Returns one ProducerStats per thread (all rates MEASURED, never assumed — ADR-0009).
// Public Domain (The Unlicense).
//
//   THE CALIBRATED SPIN (the "rate is hardware-calibrated, not a usleep"): spin_ops(n) runs n iterations of
//   a volatile-sink x+=1 so the optimizer cannot delete it (a deleted spin calibrates to "infinity" and the
//   rate control collapses — ADR-0002). STEP 1 times a fixed window of it to recover ops/sec; STEP 3 burns
//   ops_between_emissions of it after each emission. Because the SAME spin primitive both calibrates and
//   paces, the pacing is in the thread's own measured ops, immune to usleep granularity/jitter.
//
//   TOPOLOGY WIRING:
//     PerThread (A): each producer thread creates its OWN Boundary (one DEALER per thread) — single-thread
//       send+recv, no contention. The boundary is created INSIDE the thread so the socket lives on the
//       thread that uses it.
//     Coalescing (B): ONE shared Boundary is created up front; every thread calls send/recv/poll on it and
//       the boundary routes replies back per submitting thread (its coalescing thread holds the one socket).
//
//   DECOUPLED vs COUPLED:
//     DECOUPLED — emit on the calibrated clock; poll() replies without gating (latency measured, not waited
//       on). A SHORT boundary recv-timeout makes poll() return promptly so the free-run is not stalled.
//     COUPLED — emit one, recv() its reply (block), THEN spin the think-time budget, then emit the next. The
//       round-trip bounds the achieved rate; the requested rate caps the think-time only.

#include "producer.hpp"

#include <sys/resource.h>   // setpriority / PRIO_PROCESS — per-thread nice (Linux: nice is per-task)
#include <sys/syscall.h>    // SYS_gettid — this thread's kernel task id
#include <unistd.h>         // syscall

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <expected>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <span>
#include <thread>
#include <unordered_map>
#include <vector>

#include "boundary.hpp"
#include "wire.hpp"

namespace tlab {

namespace {

// Renice the ONE designated generator thread DOWN (cfg.low_prio_thread), on Linux where `nice` is
// per-task: setpriority(PRIO_PROCESS, gettid(), n) sets THIS thread's nice, not the process's. A no-op
// unless this thread is the designated one and a non-zero nice is asked. Renicing DOWN (positive nice)
// needs no privilege; a failure is surfaced loud-ish (a warning) and is non-fatal — the run continues at
// the default priority rather than aborting the measurement (ADR-0002: surface, do not silently swallow).
void apply_thread_priority(const ProducerConfig& cfg, int thread_index) {
    if (cfg.low_prio_thread < 0 || thread_index != cfg.low_prio_thread || cfg.low_prio_nice == 0)
        return;
    const auto tid = static_cast<id_t>(::syscall(SYS_gettid));
    errno = 0;
    if (::setpriority(PRIO_PROCESS, tid, cfg.low_prio_nice) != 0) {
        std::fprintf(stderr, "[tlab-producer] WARN: could not renice generator thread %d to nice %d "
                             "(errno=%d) — running at default priority\n",
                     thread_index, cfg.low_prio_nice, errno);
    }
}

// A steady (monotonic) clock alias — the right clock for a duration, never wall time. Named SteadyClock to
// avoid colliding with the C ::clock_t typedef from <ctime>.
using SteadyClock = std::chrono::steady_clock;

[[nodiscard]] double seconds_since(SteadyClock::time_point t0) {
    return std::chrono::duration<double>(SteadyClock::now() - t0).count();
}

// ---- THE CALIBRATED SPIN -------------------------------------------------------------------------
// Burn `iters` iterations of the literal "task" (x += 1) such that the optimizer CANNOT fold the loop to
// a single `x += iters` (that fold is the silent-elision failure producer.hpp STEP 1 and ADR-0002 name:
// a folded loop calibrates to "infinity" and the rate control collapses). The guard is a per-iteration
// compiler barrier — an empty `asm volatile("" : "+r"(x))` — that forces x to be materialized in a
// register on EVERY iteration, so the add genuinely executes `iters` times. This is a pure barrier (no
// instructions emitted, no memory traffic), so the timed cost is exactly the integer-add loop, not the
// barrier. `g_spin_sink` (a volatile) is also published so the result is observably live across TUs.
//
// The barrier is the standard, transparent way to defeat the fold (it is what google/benchmark's
// DoNotOptimize does); it keeps the "task" the literal `x += 1` rather than substituting a heavier op.
volatile std::uint64_t g_spin_sink = 0;

[[gnu::noinline]] std::uint64_t spin_ops(std::uint64_t iters) {
    std::uint64_t x = g_spin_sink;
    for (std::uint64_t i = 0; i < iters; ++i) {
        x += 1;                              // the literal task
        asm volatile("" : "+r"(x));          // per-iteration barrier: x is live each step, no fold to +iters
    }
    g_spin_sink = x;   // publish to the volatile sink so the loop is not dead code
    return x;
}

// STEP 1 — calibrate ops/sec for THIS thread. A short warmup spin (discarded) brings the core to frequency,
// then a fixed wall-clock window of the spin recovers ops_per_sec = iters / elapsed. The window is grown
// adaptively until it spans at least `window_seconds` so a too-fast first guess does not under-time.
[[nodiscard]] Calibration calibrate(double window_seconds) {
    Calibration c;
    // Warmup: a fixed chunk, timed only to record warmup_ops (discarded from the rate).
    constexpr std::uint64_t kWarmupIters = 5'000'000;
    spin_ops(kWarmupIters);
    c.warmup_ops = kWarmupIters;

    // Timed window: start with a guess and double until we've spanned >= window_seconds, so the measured
    // window is long enough to be stable (a sub-millisecond window would be clock-noise dominated).
    std::uint64_t iters = 10'000'000;
    for (;;) {
        const auto t0 = SteadyClock::now();
        spin_ops(iters);
        const double elapsed = seconds_since(t0);
        if (elapsed >= window_seconds || iters >= (std::uint64_t(1) << 40)) {
            c.timed_ops = iters;
            c.timed_seconds = elapsed;
            c.ops_per_sec = elapsed > 0.0 ? static_cast<double>(iters) / elapsed : 0.0;
            return c;
        }
        // Scale the next guess to roughly hit the window (with a 2x floor so we always make progress).
        const double scale = elapsed > 0.0 ? (window_seconds / elapsed) * 1.2 : 2.0;
        iters = static_cast<std::uint64_t>(static_cast<double>(iters) * std::max(2.0, scale));
    }
}

// A reusable percentile over a copy-sorted sample (nearest-rank, p in [0,1]). Empty -> 0.
[[nodiscard]] double percentile(std::vector<double> xs, double p) {
    if (xs.empty()) return 0.0;
    std::sort(xs.begin(), xs.end());
    std::size_t idx = static_cast<std::size_t>(p * (static_cast<double>(xs.size()) - 1.0) + 0.5);
    if (idx >= xs.size()) idx = xs.size() - 1;
    return xs[idx];
}

// ---- one producer thread's run ------------------------------------------------------------------
// `boundary` is the seam this thread sends through (its own in Topology A, the shared one in Topology B).
// `corr_seq` is the process-global unique corr-id source (so corr-ids are unique across all threads, exactly
// as chocofarm's WireLeafPool shares one atomic). Fills the result into `out` (one slot per thread).
void run_one_thread(const ProducerConfig& cfg, Boundary& boundary, std::atomic<std::uint64_t>& corr_seq,
                    ProducerStats& out) {
    // STEP 1 — calibrate this thread's compute rate.
    const Calibration calib = calibrate(cfg.calib_window_seconds);
    out.calib = calib;
    out.requested_rate_hz = cfg.target_rate_hz;

    // STEP 2 — derive ops to burn between emissions. We do NOT subtract a modeled per-emission overhead a
    // priori; instead the loop MEASURES the achieved rate and the overhead is whatever it is (ADR-0009 —
    // measure, do not model). The spin target is the full inter-emission budget; if the budget is already
    // smaller than one emission's cost the thread is overhead_bound (the achieved rate then sits at the
    // overhead ceiling and we report it honestly).
    const double target_rate = cfg.target_rate_hz;
    const double seconds_between = target_rate > 0.0 ? 1.0 / target_rate : 0.0;
    const double ops_between_f = calib.ops_per_sec * seconds_between;
    // A guard: if the requested rate is so high that even ZERO spin cannot keep up, the loop will simply
    // run flat-out (spin 0) and the achieved rate reports the true ceiling. We mark overhead_bound after the
    // run by comparing achieved vs requested (the honest, measured test), not from this a-priori number.
    std::uint64_t ops_between = ops_between_f > 0.0 ? static_cast<std::uint64_t>(ops_between_f) : 0;

    // Pre-build one batch of synthetic feature rows (rows_per_batch x in_dim). The VALUES are arbitrary
    // (throughput depends on the matmul SHAPES, not the contents); we fill a simple deterministic ramp so
    // the bytes are not all-zero (which a compiler/allocator might special-case) and are reproducible.
    const wire::count_t B = cfg.rows_per_batch;
    const wire::count_t in_dim = cfg.in_dim;
    std::vector<float> rows(static_cast<std::size_t>(B) * in_dim);
    for (std::size_t i = 0; i < rows.size(); ++i)
        rows[i] = static_cast<float>((i % 97)) * 0.01f - 0.5f;

    // Latency samples (producer-side send->reply round-trip, microseconds). Reserve a generous capacity.
    std::vector<double> latencies_us;
    latencies_us.reserve(1u << 16);

    // corr -> send timestamp, so a reply's latency is now - send_time. A small map suffices in COUPLED
    // (at most one outstanding); DECOUPLED may have many in flight, so we use an unordered_map.
    std::unordered_map<wire::corr_t, SteadyClock::time_point> send_times;

    std::uint64_t batches_sent = 0;
    std::uint64_t replies_recv = 0;

    auto note_reply = [&](const BoundaryReply& reply) {
        auto it = send_times.find(reply.corr);
        if (it != send_times.end()) {
            const double us = std::chrono::duration<double, std::micro>(SteadyClock::now() - it->second).count();
            latencies_us.push_back(us);
            send_times.erase(it);
        }
        replies_recv += 1;
    };

    const auto run_start = SteadyClock::now();
    const double run_seconds = cfg.run_seconds;

    if (cfg.mode == ProducerMode::Coupled) {
        // COUPLED — emit one, block for its reply, spin the think-time budget, repeat. The achieved rate is
        // bounded by the round-trip; the calibrated spin is the think-time BETWEEN reply and next emit.
        while (seconds_since(run_start) < run_seconds) {
            const wire::corr_t corr = corr_seq.fetch_add(1, std::memory_order_relaxed);
            LeafBatch lb{corr, B, in_dim, std::span<const float>(rows.data(), rows.size())};
            send_times[corr] = SteadyClock::now();
            auto sent = boundary.send(lb);
            if (!sent) break;   // a dead wire: stop this thread (the error is reported via achieved < requested)
            batches_sent += 1;
            auto reply = boundary.recv();   // BLOCK for this batch's reply (the critical-path emulation)
            if (!reply) break;              // timeout/transport error -> stop
            note_reply(*reply);
            if (ops_between > 0) spin_ops(ops_between);   // think-time budget
        }
    } else {
        // DECOUPLED — free-run at the calibrated rate; poll() replies without gating. We emit, drain any
        // ready replies (non-blocking), then burn the inter-emission spin. The boundary was created with a
        // SHORT recv-timeout (see run_producer) so poll() returns promptly and does not stall the free-run.
        while (seconds_since(run_start) < run_seconds) {
            const wire::corr_t corr = corr_seq.fetch_add(1, std::memory_order_relaxed);
            LeafBatch lb{corr, B, in_dim, std::span<const float>(rows.data(), rows.size())};
            send_times[corr] = SteadyClock::now();
            auto sent = boundary.send(lb);
            if (!sent) break;
            batches_sent += 1;
            // Drain whatever replies are ready right now (non-gating). Bounded by however many are queued.
            for (;;) {
                auto polled = boundary.poll();
                if (!polled) { sent = std::unexpected(polled.error()); break; }   // transport error -> stop
                if (!polled->has_value()) break;   // nothing ready -> stop draining, keep producing
                note_reply(**polled);
            }
            if (!sent) break;
            if (ops_between > 0) spin_ops(ops_between);   // pace to the calibrated rate
        }
    }

    const double elapsed = seconds_since(run_start);

    // DECOUPLED tail-drain: collect replies still in flight so the latency sample is not truncated and the
    // reply count is honest. The free-run loop uses a NON-BLOCKING poll() (boundary RCVTIMEO=0 in this
    // mode), so the drain CANNOT block on a still-arriving reply; we poll repeatedly under a WALL-CLOCK
    // deadline (a few RTTs past the last send) and stop when nothing remains outstanding OR the deadline
    // passes. A genuinely lost reply (server never answered) thus bounds the drain by time, never hangs.
    if (cfg.mode == ProducerMode::Decoupled) {
        const auto drain_deadline =
            SteadyClock::now() + std::chrono::milliseconds(std::max(cfg.recv_timeout_ms, 100));
        while (boundary.any_outstanding() && SteadyClock::now() < drain_deadline) {
            auto polled = boundary.poll();
            if (!polled) break;                  // transport/decode error -> stop; report what we have
            if (polled->has_value()) {
                note_reply(**polled);
            }
            // else: nothing ready this instant — keep polling until drained or the deadline passes.
        }
    }

    out.batches_sent = batches_sent;
    out.replies_recv = replies_recv;
    out.achieved_rate_hz = elapsed > 0.0 ? static_cast<double>(batches_sent) / elapsed : 0.0;
    out.mean_reply_latency_us =
        latencies_us.empty()
            ? 0.0
            : std::accumulate(latencies_us.begin(), latencies_us.end(), 0.0) /
                  static_cast<double>(latencies_us.size());
    out.p50_reply_latency_us = percentile(latencies_us, 0.50);
    out.p99_reply_latency_us = percentile(latencies_us, 0.99);
    // overhead_bound (the HONEST, MEASURED test — STEP 2 reported by outcome, not by an a-priori model):
    // the thread could not emit at the requested rate because the per-emission cost (feature-fill + encode
    // + send + drain) consumed the inter-emission budget — i.e. achieved fell meaningfully below requested.
    // This is true whether the calibrated spin was driven to zero (budget < one emission) OR the spin was
    // non-zero but the emission overhead alone exceeded the requested period; both are the same honest fact
    // ("this thread is at its emit ceiling, not at the requested rate"). A small shortfall from jitter is
    // NOT flagged (the 0.95 band); a real ceiling (achieved well under requested) is. The COUPLED mode's
    // shortfall is the round-trip, not emission overhead — there overhead_bound reads "rate not met", which
    // is still the honest "requested rate not delivered" signal the report pairs with the RTT latency.
    out.overhead_bound = out.achieved_rate_hz < 0.95 * cfg.target_rate_hz;
}

}  // namespace

[[nodiscard]] std::expected<std::vector<ProducerStats>, BoundaryError> run_producer(
        const ProducerConfig& cfg) {
    if (cfg.n_threads < 1)
        return std::unexpected(BoundaryError{"run_producer: n_threads must be >= 1", false});
    if (cfg.in_dim == 0 || cfg.rows_per_batch == 0)
        return std::unexpected(BoundaryError{"run_producer: in_dim and rows_per_batch must be >= 1", false});

    // The boundary's recv-timeout governs how long a single socket recv blocks:
    //   DECOUPLED — poll() must be TRULY non-blocking so an empty reply queue does NOT throttle the
    //     free-run (a 1ms-per-empty-poll cost would itself cap throughput — the very artifact this lab
    //     must not introduce). RCVTIMEO=0 makes ZMQ return EAGAIN immediately when nothing is queued.
    //     (Topology A: poll() is one such immediate socket read. Topology B: the COALESCING THREAD's
    //     recv leg becomes a non-blocking spin-drain, and the producer's poll() reads a mailbox — also
    //     immediate. The decoupled tail-drain is deadline-bounded, not recv-blocked; see run_one_thread.)
    //   COUPLED — recv() legitimately waits for the round-trip, so we honor cfg.recv_timeout_ms (a slow
    //     or absent server then surfaces as a loud bounded timeout, ADR-0002).
    const int boundary_recv_timeout_ms =
        (cfg.mode == ProducerMode::Decoupled) ? 0 : cfg.recv_timeout_ms;

    BoundaryConfig bcfg;
    bcfg.endpoint = cfg.endpoint;
    bcfg.recv_timeout_ms = boundary_recv_timeout_ms;
    bcfg.n_producer_threads = cfg.n_threads;
    bcfg.rows = static_cast<int>(cfg.rows_per_batch);     // sizes the per-message memory for the send HWM
    bcfg.in_dim = static_cast<int>(cfg.in_dim);
    bcfg.send_queue_bytes = cfg.send_queue_bytes;          // TOTAL outstanding-send byte budget (back-pressure)

    std::atomic<std::uint64_t> corr_seq{1};   // process-global unique corr-id source (shared by all threads)
    std::vector<ProducerStats> stats(static_cast<std::size_t>(cfg.n_threads));

    if (cfg.topology == BoundaryTopology::Coalescing) {
        // Topology B — ONE shared boundary; all threads send/recv/poll on it (it routes per thread).
        auto boundary = make_boundary(BoundaryTopology::Coalescing, bcfg);
        if (!boundary) return std::unexpected(boundary.error());
        Boundary& shared = **boundary;
        std::vector<std::thread> threads;
        threads.reserve(static_cast<std::size_t>(cfg.n_threads));
        for (int t = 0; t < cfg.n_threads; ++t)
            threads.emplace_back([&, t] {
                apply_thread_priority(cfg, t);   // renice this thread DOWN iff it is the designated one
                run_one_thread(cfg, shared, corr_seq, stats[t]);
            });
        for (auto& th : threads) th.join();
        return stats;
    }

    // Topology A — each thread creates and owns its OWN boundary (one DEALER per thread). The boundary is
    // created INSIDE the thread so the socket is opened/used/closed all on the owning thread. A per-thread
    // creation failure is latched and surfaced as the run's error after the join.
    std::vector<std::thread> threads;
    threads.reserve(static_cast<std::size_t>(cfg.n_threads));
    std::mutex err_mu;
    std::optional<BoundaryError> first_err;
    for (int t = 0; t < cfg.n_threads; ++t) {
        threads.emplace_back([&, t] {
            apply_thread_priority(cfg, t);   // renice this thread DOWN iff it is the designated one
            auto boundary = make_boundary(BoundaryTopology::PerThread, bcfg);
            if (!boundary) {
                std::lock_guard<std::mutex> lk(err_mu);
                if (!first_err) first_err = boundary.error();
                return;
            }
            run_one_thread(cfg, **boundary, corr_seq, stats[t]);
        });
    }
    for (auto& th : threads) th.join();
    if (first_err) return std::unexpected(*first_err);
    return stats;
}

}  // namespace tlab
