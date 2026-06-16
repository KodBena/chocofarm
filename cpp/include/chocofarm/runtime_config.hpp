// cpp/include/chocofarm/runtime_config.hpp
// Purpose: the ONE home for the C++ search-runtime parallelism knobs (ADR-0012 P1: single source of
//   truth, no re-typed literals across the pool / the benchmark / the actor):
//     * thread_pool_size — T OS worker threads;
//     * batch_size       — the IN-FLIGHT leaf-eval target = the number of leaf forwards concurrently
//                          outstanding to the batched inference server (= T x fibers-per-thread), i.e.
//                          how big a batch the server's greedy drain can assemble across the T peers.
//   `fibers_per_thread` is DERIVED from those two (batch / threads, rounded up), never re-typed — this is
//   the whole point of decoupling batch from OS threads: you grow the MLP-eval batch by adding FIBERS
//   per thread, not by adding context-switching threads (the maintainer's ask). Defaults are sized for
//   the 4-vCPU host; env-overridable (CHOCO_POOL_THREADS / CHOCO_POOL_BATCH) so a run tunes them without
//   a rebuild (ADR-0012 P4 — read at construction, the live source), and any consumer reads the SAME
//   home rather than hardcoding 4 / 32 at its own call site.
//
// Public Domain (The Unlicense).
#pragma once

#include <algorithm>
#include <cstdlib>

namespace chocofarm {

struct RuntimeConfig {
    int thread_pool_size = 4;  // T OS worker threads (the 4-vCPU host wall; CHOCO_POOL_THREADS)
    int batch_size = 32;       // in-flight leaf-eval target across the pool (CHOCO_POOL_BATCH)

    // Fibers (parked trees) each worker thread multiplexes — DERIVED so batch and threads stay decoupled
    // and the relation has one home. ceil(batch / threads), at least 1.
    [[nodiscard]] int fibers_per_thread() const {
        const int t = std::max(1, thread_pool_size);
        return std::max(1, (std::max(1, batch_size) + t - 1) / t);
    }

    // Read the two knobs from the environment, falling back to the host-sized defaults above. A consumer
    // may further override (e.g. a --threads / --batch CLI flag) after this — but the DEFAULTS + the
    // derivation live only here.
    [[nodiscard]] static RuntimeConfig from_env() {
        RuntimeConfig c;
        if (const char* t = std::getenv("CHOCO_POOL_THREADS")) c.thread_pool_size = std::max(1, std::atoi(t));
        if (const char* b = std::getenv("CHOCO_POOL_BATCH")) c.batch_size = std::max(1, std::atoi(b));
        return c;
    }
};

}  // namespace chocofarm
