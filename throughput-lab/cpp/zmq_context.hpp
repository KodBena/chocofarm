// throughput-lab/cpp/zmq_context.hpp
// Purpose: the ONE process-global ZMQ context the lab's dealers share (ADR-0012 P1: one home). A ZMQ
//   context is the only ZMQ object that IS thread-safe and is explicitly meant to be shared across a
//   process's sockets (zmq_ctx_new / "one context per process"); every DEALER this lab opens (Topology
//   A's per-thread dealers, Topology B's single coalescing dealer) is created on THIS context, and its
//   IO threads carry the traffic. Owning it once (a function-local static, constructed on first use,
//   torn down at process exit) keeps the IO-thread pool singular and avoids a per-socket context churn.
// Public Domain (The Unlicense).
//
//   ADR-0002: a context-creation failure (zmq_ctx_new returning null — an OOM/resource-exhaustion) is a
//   loud typed failure at first use, never a null silently propagated into zmq_socket.

#pragma once

#include <zmq.h>

#include <expected>
#include <string>

#include "boundary.hpp"   // tlab::BoundaryError

namespace tlab {

// Borrow the process-global ZMQ context (created on first call, destroyed at process exit via the
// Holder's dtor). Returns the borrowed void* context (NOT owned by the caller — never zmq_ctx_term it),
// or a typed BoundaryError if the one-time creation failed. Thread-safe: the function-local static's
// initialization is guaranteed thread-safe by the C++ standard (a "magic static").
[[nodiscard]] inline std::expected<void*, BoundaryError> shared_zmq_context() {
    struct Holder {
        void* ctx = nullptr;
        Holder() { ctx = zmq_ctx_new(); }     // null on resource exhaustion — checked by the caller below
        ~Holder() {
            if (ctx != nullptr) zmq_ctx_term(ctx);   // graceful teardown at process exit
        }
        Holder(const Holder&) = delete;
        Holder& operator=(const Holder&) = delete;
    };
    static Holder holder;
    if (holder.ctx == nullptr)
        return std::unexpected(BoundaryError{"shared_zmq_context: zmq_ctx_new failed (resource exhaustion)",
                                             false});
    return holder.ctx;
}

}  // namespace tlab
