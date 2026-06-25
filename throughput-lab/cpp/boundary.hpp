// throughput-lab/cpp/boundary.hpp
// Purpose: the ABSTRACT producer->server boundary seam (the typed SSOT, ADR-0012 P8) — the interface
//   the producer sends leaf-batches THROUGH, with two implementations the build agent supplies:
//   Topology A (one outbound DEALER socket per producer thread — today's shape) and Topology B (the
//   producer threads feed a SEPARATE coalescing thread holding ONE socket). The interface is the
//   SSOT; A and B are two impls of it, and the seam admits MORE topologies than these two (P8: the
//   typed signature is the contract, not the current count of impls). This header defines the
//   contract + the value types; it implements NOTHING.
// Public Domain (The Unlicense).
//
//  WHERE THIS SITS (producer -> BOUNDARY -> server):
//    producer threads --(submit a leaf-batch)--> Boundary --(ZMQ DEALER -> ROUTER)--> Python server
//    server --(reply: corr-id + response frame)--> Boundary --(deliver to the submitting thread)-->
//
//  The Boundary owns the ZMQ transport (the DEALER socket[s], the corr-id stamping, the multipart
//  framing defined in wire.hpp Layer 2). A producer thread hands it a typed request (the flat
//  feature rows + their shape) and a corr-id, and later collects the matched reply. The Boundary
//  NEVER computes the forward and NEVER interprets the corr-id (it round-trips it opaquely).
//
//  ADR-0012 P9 (honest signatures): every fallible op returns [[nodiscard]] std::expected<T, Error>;
//  a "maybe nothing yet" poll returns [[nodiscard]] std::optional<T>; inputs are bounds-carrying
//  std::span<const T>, not raw pointer+len; construction that can fail is a static create() factory
//  over a private ctor (a throwing ctor cannot return a value). No untyped-effectful-void, no
//  nullable-pointer/sentinel optionals, no thrown control flow across this seam.

#pragma once

#include <cstddef>
#include <cstdint>
#include <expected>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "proc_domains.hpp"  // tlab — ThreadCount/ByteCount/HwmMessages/Milliseconds (process-shape domains)
#include "wire.hpp"   // tlab::wire — the Layer-1 codec + RowCount/FeatureDim/ProducerCorr (+ corr_t/count_t)

namespace tlab {

// ---- the typed error carried across the boundary seam (ADR-0012 P9 rule 5) ----------------------
// A boundary failure is a TYPED return value, never a thrown exception across the seam and never a
// sentinel. `message` is human-readable; `is_timeout` lets the caller distinguish a recv timeout (a
// slow/absent server) from a hard transport error (a dead socket) without string-matching.
struct BoundaryError {
    std::string message;
    bool is_timeout = false;
};

[[nodiscard]] inline std::unexpected<BoundaryError> boundary_err(std::string msg, bool is_timeout = false) {
    return std::unexpected(BoundaryError{std::move(msg), is_timeout});
}

// ---- a leaf-batch to send (the producer's value-level submission) -------------------------------
// `flat` is B*in_dim contiguous float32 rows, row-major (row r, col c at flat[r*in_dim + c]); B and
// in_dim name its shape. `corr` is the correlation id the producer stamps so it can match the reply
// (the Boundary frames it as the LEADING ZMQ frame per wire.hpp Layer 2, and round-trips it opaquely
// — it NEVER parses it). A bounds-carrying view, not a raw pointer (P9).
struct LeafBatch {
    wire::ProducerCorr corr{0};       // the producer-stamped correlation id (opaque round-trip)
    wire::RowCount B{0};              // # leaf ROWS in this batch (>= 1)
    wire::FeatureDim in_dim{0};       // feature WIDTH per row (Stage-A 241)
    std::span<const float> flat;   // B*in_dim floats, valid for the duration of the send call
};

// ---- a reply the boundary delivered back (matched to a submitted corr) --------------------------
// `corr` is the echoed correlation id (the producer matches it to its outstanding LeafBatch);
// `preds` is the decoded B predictions (each a de-standardized value + raw logits) in submit order.
struct BoundaryReply {
    wire::ProducerCorr corr{0};   // the echoed PRODUCER corr (Topology B rewrites the wire corr back to it)
    std::vector<wire::ResponseFields> preds;
};

// ================================================================================================
//  THE ABSTRACT BOUNDARY (the SSOT seam). Two impls: BoundaryPerThread (Topology A) and
//  BoundaryCoalescing (Topology B). A producer thread calls send() to push a leaf-batch and
//  recv()/poll() to collect a matched reply. The contract is the same for both topologies; what
//  differs is how many sockets exist and whether a coalescing thread sits between (P8: one seam,
//  many impls).
// ================================================================================================
class Boundary {
  public:
    virtual ~Boundary() = default;

    Boundary(const Boundary&) = delete;
    Boundary& operator=(const Boundary&) = delete;

    // Send one leaf-batch toward the server. Returns {} on a successful enqueue/transmit, or a typed
    // BoundaryError on a transport failure. NON-BLOCKING with respect to the server's reply: send
    // returns once the batch is handed to the transport (Topology A: written to this thread's DEALER;
    // Topology B: handed to the coalescing thread's queue). The reply is collected later via
    // recv()/poll(). The corr in `batch` is the producer's to choose and to match on.
    //
    // THREADING CONTRACT (the seam's, honored differently per topology): in Topology A each producer
    // thread owns its own Boundary (a per-thread DEALER is single-writer) — `send` is called only by
    // that owning thread. In Topology B the Boundary is SHARED and `send` is the thread-safe handoff
    // to the coalescing thread. The build agent documents which discipline its impl assumes.
    [[nodiscard]] virtual std::expected<void, BoundaryError> send(const LeafBatch& batch) = 0;

    // BLOCK up to the configured receive timeout for the NEXT matched reply, then deliver it. Returns
    // the BoundaryReply, or a typed BoundaryError (BoundaryError::is_timeout == true on a timeout — a
    // slow/absent server; a hard transport/decode error otherwise). The reply may be for ANY
    // outstanding corr this Boundary submitted (DEALER replies are not ordered across corrs — the
    // caller routes by reply.corr). FAIL LOUD (ADR-0002): a malformed envelope, a decode failure, or
    // an unknown corr is the error arm, never a silent wrong-batch delivery.
    [[nodiscard]] virtual std::expected<BoundaryReply, BoundaryError> recv() = 0;

    // NON-BLOCKING poll for a ready reply. Returns std::nullopt when no reply is currently available
    // (a legitimately-absent result — P9 typed optional, NOT a sentinel), the BoundaryReply when one
    // is, or a typed BoundaryError on a transport/decode failure. The two-layer return
    // (expected<optional<...>>) draws ABSENCE (nullopt: nothing yet, valid) apart from FAILURE
    // (the error arm) precisely (ADR-0012 P9). Used by the DECOUPLED producer mode's free-run loop.
    [[nodiscard]] virtual std::expected<std::optional<BoundaryReply>, BoundaryError> poll() = 0;

    // True iff at least one submitted batch has not yet been matched by a recv()/poll() reply. Lets a
    // coupled producer know whether to block on recv() before producing the next batch, and lets a
    // decoupled producer drain remaining replies at shutdown.
    [[nodiscard]] virtual bool any_outstanding() const = 0;

  protected:
    Boundary() = default;
};

// ---- the boundary topology plug (ADR-0012 P8: the closed vocabulary of impls of the seam) --------
// A is one DEALER socket PER producer thread (today's chocofarm shape); B is the producer threads
// feeding ONE coalescing thread that holds ONE socket. The interface admits more, but these are the
// two the lab builds. enum class (P9 — scoped, not an unscoped int).
enum class BoundaryTopology {
    PerThread,    // Topology A: one outbound DEALER per producer thread
    Coalescing,   // Topology B: producer threads -> one coalescing thread -> one DEALER
};

// Configuration the Boundary factory needs. `endpoint` is the ZMQ ipc:// (or tcp://) the server
// binds (the lab default is an ipc:// unix socket — see harness/ and the README). `recv_timeout_ms`
// bounds recv()/poll() so an absent server becomes a loud timeout, not a hang (ADR-0002). For
// Topology B, `n_producer_threads` sizes the coalescing thread's intake.
struct BoundaryConfig {
    std::string endpoint;            // e.g. "ipc:///tmp/tlab-infer.sock"
    Milliseconds recv_timeout_ms{5000};   // bounds recv()/poll(); the empty-optional case (block forever) is
                                          // handled at the ZMQ ACL (zmq_dealer.hpp), not modeled in the config
    ThreadCount n_producer_threads{1};    // Topology A: # DEALERs (one per thread); Topology B: intake sizing
    wire::RowCount rows{1};               // B per message — sizes per-message memory for the byte-budgeted HWM
    wire::FeatureDim in_dim{241};         // feature width per row — sizes per-message memory for the send HWM
    ByteCount send_queue_bytes{256ull << 20};  // TOTAL outstanding-send byte budget across all dealers (cap)
};

// The send-queue high-water mark (in MESSAGES) that holds outstanding-send memory under `budget_bytes`
// for messages of `rows` x `in_dim` float32. The DEALER buffers at most SNDHWM messages before it
// back-pressures; bounding by BYTES (not a fixed message count) keeps the memory cap honest as `rows`
// grows — a fixed count lets large-row messages OOM the producer (exactly what the old 1'000'000-deep
// HWM did, ~60 GB). Overestimates per-message overhead so the budget is a CEILING, not a floor.
[[nodiscard]] inline HwmMessages send_hwm_for_budget(ByteCount budget_bytes, wire::RowCount rows,
                                                     wire::FeatureDim in_dim) {
    // ACL: the count->bytes crossing (B*in_dim*FLOAT_BYTES) is the explicit multiply at the sizing site —
    // the typed counts .value()-unwrap into the size_t byte arithmetic here (the named count->bytes ACL).
    const std::size_t per_msg = static_cast<std::size_t>(17)              // corr(8) + req header(9)
        + static_cast<std::size_t>(rows.value()) * static_cast<std::size_t>(in_dim.value())
              * wire::FLOAT_BYTES                                          // B*in_dim float32 payload
        + static_cast<std::size_t>(512);                                  // generous ZMQ bookkeeping slack
    std::size_t hwm = budget_bytes.value() / (per_msg > 0 ? per_msg : 1);
    if (hwm < 4) hwm = 4;                  // always allow a little pipelining
    if (hwm > 1'000'000) hwm = 1'000'000;  // sane ceiling (also guards the message-count narrowing)
    return HwmMessages{static_cast<std::uint32_t>(hwm)};  // ACL: size_t->u32 message count (clamped above)
}

// The boundary factory: build the impl named by `topology`, connected per `cfg`. Returns the owned
// Boundary or a typed BoundaryError if the transport (context/socket/connect) fails — never throws
// on a failed construction (P9 rule 5: a failing ctor cannot return a value). The build agent
// IMPLEMENTS this in boundary_per_thread.cpp / boundary_coalescing.cpp (one TU per impl, P3
// one-owner) and this factory dispatches on the enum.
[[nodiscard]] std::expected<std::unique_ptr<Boundary>, BoundaryError> make_boundary(
        BoundaryTopology topology, const BoundaryConfig& cfg);

}  // namespace tlab
