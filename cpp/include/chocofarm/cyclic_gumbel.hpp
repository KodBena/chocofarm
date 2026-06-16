// cpp/include/chocofarm/cyclic_gumbel.hpp
// Purpose: CyclicGumbelSource — the RNG-free, deterministic GumbelSource the fiber benches + the Option-A
//   proof share (the ONE home, ADR-0012 P1). Every draw cycles a fixed gumbel table (mod its length) and
//   the sampled world is bw[0], so two runs fed the SAME table see byte-identical draws — the property
//   fiber_proto.cpp relies on to assert fiber-driven ≡ direct, and the wire benches use to make K
//   independent trees differ only by a rotated table. Deliberately trivial: it carries no RNG and no
//   real prior, only a scripted draw sequence.
//
//   (Distinct from gumbel_dump.cpp's richer file-local scripted source, which ALSO scripts world
//   selection via a `(gumbels, world_idxs)` script — a different fixture that merely shares the bare
//   "scripted" idea, not this type.)
//
// Public Domain (The Unlicense).
#pragma once

#include <cstdint>
#include <utility>
#include <vector>

#include "chocofarm/gumbel.hpp"

namespace chocofarm {

class CyclicGumbelSource final : public GumbelSource {
  public:
    explicit CyclicGumbelSource(std::vector<double> table) : table_(std::move(table)) {}
    uint32_t sample_world(const std::vector<uint32_t>& bw) override { return bw.empty() ? 0u : bw[0]; }
    std::vector<double> gumbel(int n) override {
        std::vector<double> out(static_cast<size_t>(n));
        for (int i = 0; i < n; ++i) out[static_cast<size_t>(i)] = table_[(idx_++) % table_.size()];
        return out;
    }

  private:
    std::vector<double> table_;
    size_t idx_ = 0;
};

}  // namespace chocofarm
