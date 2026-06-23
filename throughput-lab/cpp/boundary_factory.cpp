// throughput-lab/cpp/boundary_factory.cpp
// Purpose: make_boundary() — the dispatch over BoundaryTopology declared in boundary.hpp. It is the ONE
//   place that maps the topology enum to its impl's factory leg (make_boundary_per_thread /
//   make_boundary_coalescing), each living in its own TU (ADR-0012 P3 one-owner: one TU per impl, so no
//   impl can be hand-copied into another and silently diverge). Adding a third topology adds an enum case
//   here + a new TU — the seam (boundary.hpp) is untouched (P8: the typed signature is the SSOT).
// Public Domain (The Unlicense).

#include <expected>
#include <memory>

#include "boundary.hpp"

namespace tlab {

// The per-impl factory legs (defined in boundary_per_thread.cpp / boundary_coalescing.cpp). Declared here
// (not in boundary.hpp) because they are an INTERNAL dispatch detail — the public seam is make_boundary().
[[nodiscard]] std::expected<std::unique_ptr<Boundary>, BoundaryError> make_boundary_per_thread(
        const BoundaryConfig& cfg);
[[nodiscard]] std::expected<std::unique_ptr<Boundary>, BoundaryError> make_boundary_coalescing(
        const BoundaryConfig& cfg);

[[nodiscard]] std::expected<std::unique_ptr<Boundary>, BoundaryError> make_boundary(
        BoundaryTopology topology, const BoundaryConfig& cfg) {
    switch (topology) {
        case BoundaryTopology::PerThread:
            return make_boundary_per_thread(cfg);
        case BoundaryTopology::Coalescing:
            return make_boundary_coalescing(cfg);
    }
    // Unreachable for a valid enum; a loud typed failure rather than undefined fall-off (ADR-0002).
    return std::unexpected(BoundaryError{"make_boundary: unknown BoundaryTopology", false});
}

}  // namespace tlab
