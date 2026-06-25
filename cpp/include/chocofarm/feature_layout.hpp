// cpp/include/chocofarm/feature_layout.hpp
// Purpose: the C++ runtime READER for the cross-language feature-layout SSOT. The Python FeatureLayout
//   (chocofarm/az/features.py) is the one owner of the §2.2 ordered block table; FeatureLayout.spec()
//   emits chocofarm/data/feature_layout.json — the ordered (key, width) blocks + dim — netted against
//   the owner by tests/test_feature_layout.py (the same fail-loud SSOT-net idiom as wire_spec /
//   control_spec). This loads that artifact at runtime so FeatureBuilder::build assembles by NAMED
//   block rather than re-encoding the layout as a positional `o += N` offset ladder (ADR-0012 P7: the
//   C++ re-derives nothing — the block order + widths are the owner's). The spec path is
//   CHOCO_FEATURE_LAYOUT (default chocofarm/data/feature_layout.json), read by the FeatureBuilder ctor.
//
// Public Domain (The Unlicense).
#pragma once

#include <expected>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "chocofarm/domains.hpp"  // FeatureDim — the feature-vector dimension/offset/width domain (P1)
#include "chocofarm/error.hpp"

namespace chocofarm {

// One layout block: a named contiguous span [start, start+width) of the feature vector. `start` and
// `width` are both FeatureDim (the same vector-space offset domain; FeatureDim is additive so
// start + width -> the next start, the contiguous-partition arithmetic load() does — ADR-0012 P1).
struct FeatureBlock {
    std::string key;
    FeatureDim start{0};
    FeatureDim width{0};
};

// The feature-vector layout, loaded from the Python-emitted SSOT (feature_layout.json). Built by the
// load() factory — a throwing ctor cannot return a value (P9 rule 5), and a missing / malformed /
// dim-inconsistent spec is a typed boundary Error returned by value. Serves a named block's start
// offset; a lookup of a key the spec does not carry is an INVARIANT violation (the C++ writer's keys
// and the spec are one contract, enforced by the drift net + the FeatureBuilder ctor's key check) ->
// a loud abort, not a boundary Error.
class FeatureLayoutSpec {
  public:
    // An empty, INVALID layout — the placeholder a FeatureLayoutSpec member holds until its owner
    // move-assigns a load()ed one. Never queried before that assignment (FeatureBuilder loads in its
    // ctor body, before any build()).
    FeatureLayoutSpec() = default;

    // Load + validate the spec at `path`. `expected_dim` is the env-derived dim the caller computed
    // (5N+3nD+6+n_tel); BOTH the spec's `dim` field AND Σwidth must equal it — a desync between the
    // shipped spec and this env is a loud boundary Error (never a silent mislabel).
    [[nodiscard]] static std::expected<FeatureLayoutSpec, Error>
    load(std::string_view path, FeatureDim expected_dim);

    [[nodiscard]] FeatureDim dim() const { return dim_; }
    // block_count is the cardinality of the block table — a generic container count netted against
    // kWritten.size() (NOT a feature-vector dimension), so it stays a raw int (no domain crossing).
    [[nodiscard]] int block_count() const { return static_cast<int>(blocks_.size()); }
    [[nodiscard]] bool contains(std::string_view key) const {
        return index_.find(std::string(key)) != index_.end();
    }
    // The start offset of the named block. Aborts loudly (ADR-0002) if `key` is absent — an invariant
    // (the writer names only keys the FeatureBuilder ctor verified the spec carries).
    [[nodiscard]] FeatureDim start(std::string_view key) const;

  private:
    FeatureDim dim_{0};
    std::vector<FeatureBlock> blocks_;
    std::unordered_map<std::string, int> index_;  // key -> blocks_ index (a vector position, not a dim)
};

}  // namespace chocofarm
