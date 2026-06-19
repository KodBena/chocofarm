// cpp/include/chocofarm/event_log.hpp
// Purpose: optional, gated, best-effort PROTOCOL EVENT LOG for the leaf-eval transport. When the
//   environment names CHOCO_EVENTLOG_CPP, `CHOCO_EV(kind, fields)` appends a high-resolution
//   monotonic-timestamped line `<mono_ns> PRD-<tid> <kind> <fields>` to that file. The Python server
//   logs to its OWN file (CHOCO_EVENTLOG) on the SAME monotonic timebase — std::chrono::steady_clock and
//   Python's time.monotonic both read CLOCK_MONOTONIC, shared across processes on one Linux host — so
//   tools/event_merge.py orders both sides into one correlated timeline despite OS jitter (the timestamp
//   is taken AT the event, not at write time).
//
//   OFF by default (env unset) => zero behaviour change and one cached-bool branch per call site. This is
//   BEST-EFFORT observability, deliberately NOT a fail-loud path: ADR-0002 governs the correctness of the
//   transport, not a debug log, so a logging failure silently disables the stream and never aborts a
//   send/recv (the alternative — letting an instrumentation error perturb or abort the protocol — would
//   corrupt the very measurement it exists to take).
//
// Public Domain (The Unlicense).
#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <ios>
#include <mutex>
#include <sstream>
#include <string_view>

namespace chocofarm::evlog {

// Cached once: is the C++ event log enabled? (env read a single time.)
inline bool enabled() {
    static const bool on = (std::getenv("CHOCO_EVENTLOG_CPP") != nullptr);
    return on;
}

// Nanoseconds on the monotonic clock. libstdc++/libc++ implement steady_clock over CLOCK_MONOTONIC, the
// same clock Python's time.monotonic_ns() reads — so values from both processes are directly comparable.
inline std::uint64_t mono_ns() {
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch())
            .count());
}

// A small per-thread id (0,1,2,… in first-emit order) so producer threads are distinguishable in the log.
inline int tid() {
    static std::atomic<int> next{0};
    thread_local int id = next.fetch_add(1, std::memory_order_relaxed);
    return id;
}

// Append one event. `t` is captured by the caller BEFORE this call (so the recorded time is the event
// time, not the lock-acquire time). A single shared sink guarded by a mutex; opened lazily once.
inline void emit(std::uint64_t t, std::string_view kind, std::string_view fields) {
    static std::mutex m;
    static std::ofstream* const f = []() -> std::ofstream* {
        const char* p = std::getenv("CHOCO_EVENTLOG_CPP");
        if (p == nullptr) return nullptr;
        auto* s = new std::ofstream(p, std::ios::app);  // process-lifetime singleton; intentionally leaked
        return (s != nullptr && s->is_open()) ? s : nullptr;
    }();
    if (f == nullptr) return;
    std::lock_guard<std::mutex> lk(m);
    (*f) << t << " PRD-" << tid() << ' ' << kind << ' ' << fields << '\n';
    f->flush();  // per-event flush: the producer may be SIGKILLed mid-wedge — never lose trailing events
}

}  // namespace chocofarm::evlog

// CHOCO_EV(kind, fields_stream) — e.g. CHOCO_EV("SUBMIT", "corr=" << corr << " B=" << B);
// Timestamp captured first; the fields string is built ONLY when enabled.
#define CHOCO_EV(kind, fields_stream)                                  \
    do {                                                               \
        if (::chocofarm::evlog::enabled()) {                           \
            const std::uint64_t _ev_t = ::chocofarm::evlog::mono_ns(); \
            std::ostringstream _ev_o;                                  \
            _ev_o << fields_stream;                                    \
            ::chocofarm::evlog::emit(_ev_t, (kind), _ev_o.str());      \
        }                                                              \
    } while (0)
