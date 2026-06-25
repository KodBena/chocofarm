// throughput-lab/cpp/proc_domains.hpp
// Purpose: the ONE home (ADR-0012 P1) for the throughput-lab PROCESS / OS / pipeline-shape integer-domain
//   phantom types — the ones that are NOT wire bytes (those live in wire.hpp) and NOT chocofarm-core
//   search shape (those live in chocofarm/domains.hpp, consumed via the typed GumbelConfig/Environment).
//   Minted over the reused Quantity<Tag, Rep> machinery (chocofarm/quantity.hpp): the producer thread
//   COUNT vs INDEX split (the pervasive bare `int` the inventory most wants separated, with the -1
//   "low-prio thread = none" sentinel dissolved into std::optional<ThreadIndex>), the calibration
//   op-iteration count, the high-water-mark message count, the byte-budget domain, and the millisecond
//   duration domain. Sign + width MOTIVATED at each declaration (ADR-0000 rule 1).
//
//   The C-API-FORCED domains (ZMQ SNDHWM / RCVTIMEO take a signed int; setpriority takes a signed nice
//   in [-20,19]; setpriority's tid is id_t) are documented as ACL boundaries: the typed domain narrows to
//   the primitive ONLY at the setsockopt/setpriority call (ADR-0012 P2: the ACL conforms to the port), so
//   the int is forced at the leaf, not free-floating through the code.
//
//   A leaf header (only <cstdint>/<optional> + the machinery) so producer.hpp / real_producer.cpp /
//   main.cpp / boundary*.cpp include it with no cycle.
//
// Public Domain (The Unlicense).
#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>

#include "chocofarm/quantity.hpp"

namespace tlab {

// ---- thread COUNT vs INDEX (the count-vs-index pair the inventory most wants split) ----
// A thread COUNT (n_threads / n_producer_threads): how many producer threads. >= 1 (validated); the
// budget divisor (per_dealer_bytes = send_queue_bytes / n) — the >= 1 invariant dissolves the std::max
// guard and the divide-by-zero hazard (ADR-0008). Tiny (a 4-vCPU box). Additive (a count).
struct ThreadCountTag {};
using ThreadCount = chocofarm::Quantity<ThreadCountTag, std::uint32_t>;

// A 0-based thread INDEX (idx / thread_index / the t/i loop var). DISTINCT from the count. The legacy
// -1 "no designated low-prio thread" sentinel becomes std::optional<ThreadIndex> (ADR-0002 typed
// absence) so the index itself is unsigned and never carries a negative. Affine (a 0-based index).
struct ThreadIndexTag {};
using ThreadIndex = chocofarm::Quantity<ThreadIndexTag, std::uint32_t>;
using OptThreadIndex = std::optional<ThreadIndex>;  // the "no designated thread" case, typed not -1

// ---- calibration op-iteration count ----
// The x+=1 busy-work iteration count (warmup_ops, timed_ops, ops_between_emissions, the adaptive `iters`
// window up to 1<<40, the spin accumulator). u64 BECAUSE the count exceeds 2^32 in a sub-second window
// on modern hardware (the adaptive loop grows to 1<<40) and a wrapped count calibrates to a WRONG rate
// (ADR-0002). Unsigned (a count). Additive. The volatile sink stays a raw u64 the optimizer can't fold.
struct OpCountTag {};
using OpCount = chocofarm::Quantity<OpCountTag, std::uint64_t>;

// ---- ZMQ send high-water mark, in MESSAGES (not bytes) ----
// A queue depth in whole messages, derived from a byte budget / per-message size, clamped to
// [4, 1'000'000]. The DOMAIN is an unsigned message count; it narrows to the signed int the
// zmq_setsockopt(ZMQ_SNDHWM) C API demands ONLY at that call (the named ACL — the clamp also guards the
// size_t->int cast). A count, additive.
struct HwmMessagesTag {};
using HwmMessages = chocofarm::Quantity<HwmMessagesTag, std::uint32_t>;

// ---- byte-count / byte-offset (memory & wire sizing) ----
// A non-negative byte extent or buffer offset: send_queue_bytes / per_dealer_bytes / budget_bytes /
// per_msg / frame offsets / the B*in_dim*4 products / est_resident / MemAvailable. size_t is the
// MOTIVATED width here (the one place a reflexive size_t IS right): it indexes/sizes raw byte buffers,
// matches sizeof / pointer arithmetic / std::span, and is what /proc MemAvailable and the std allocators
// speak. Distinct from an element COUNT (a row count is not a byte count; the *FLOAT_BYTES is the
// count->bytes ACL). Additive + affine (a size, and an offset into a buffer). Shared by core + tlab.
struct ByteCountTag {};
using ByteCount = chocofarm::Quantity<ByteCountTag, std::size_t>;

// ---- millisecond durations ----
// A duration in ms (recv_timeout_ms, send_timeout_ms, kIntakeWaitMs). The ZMQ RCVTIMEO/SNDTIMEO C API
// takes a signed int with -1/<=0 = "block forever" — that sign is API-load-bearing, so the value narrows
// to int ONLY at the setsockopt ACL, and the "block forever" case is modeled as an EMPTY
// std::optional<Milliseconds> (ADR-0002 typed absence) rather than a sign sentinel threaded through. A
// non-negative duration; affine (durations add / a deadline + a delta).
struct MillisecondsTag {};
using Milliseconds = chocofarm::Quantity<MillisecondsTag, std::uint32_t>;
using OptMilliseconds = std::optional<Milliseconds>;  // empty = "block forever" (ZMQ -1), typed not <=0

}  // namespace tlab

namespace chocofarm {
template <> struct quantity_additive<tlab::ThreadCountTag> : std::true_type {};
template <> struct quantity_affine<tlab::ThreadIndexTag> : std::true_type {};
template <> struct quantity_additive<tlab::OpCountTag> : std::true_type {};
template <> struct quantity_additive<tlab::HwmMessagesTag> : std::true_type {};
template <> struct quantity_additive<tlab::ByteCountTag> : std::true_type {};
template <> struct quantity_affine<tlab::ByteCountTag> : std::true_type {};
template <> struct quantity_affine<tlab::MillisecondsTag> : std::true_type {};
}  // namespace chocofarm
