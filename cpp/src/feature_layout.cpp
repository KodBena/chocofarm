// cpp/src/feature_layout.cpp
// Purpose: load + validate the feature-layout SSOT (see feature_layout.hpp). Reuses the project's
//   load-json-at-the-boundary discipline (mirrors instance.cpp): a missing / malformed / inconsistent
//   spec is a typed Error (P9 rule 5); nlohmann's accessor throws (a missing key / a type mismatch)
//   are caught HERE at the edge and translated, so the public contract is throw-free.
//
//   Plain nlohmann::json (NOT ordered_json): `blocks` is a JSON ARRAY, so its order is the array's
//   order regardless; `dim` / `key` / `width` are object lookups (order-independent). The layout block
//   ORDER — the thing whose drift would silently mislabel the vector — rides the array, not object keys.
//
// Public Domain (The Unlicense).
#include "chocofarm/feature_layout.hpp"

#include <cstdlib>
#include <iostream>
#include <string>

#include <fstream>
#include <nlohmann/json.hpp>

namespace chocofarm {

std::expected<FeatureLayoutSpec, Error>
FeatureLayoutSpec::load(std::string_view path, FeatureDim expected_dim) {
    std::ifstream f{std::string(path)};
    if (!f)
        return std::unexpected(make_error("cannot open feature-layout spec: " + std::string(path)));
    nlohmann::json j = nlohmann::json::parse(f, nullptr, /*allow_exceptions=*/false);
    if (j.is_discarded())
        return std::unexpected(make_error("malformed JSON in feature-layout spec: " + std::string(path)));
    try {
        FeatureLayoutSpec spec;
        // JSON loader ACL (ADR-0002): nlohmann get<int> parses signed; FAIL-LOUD on a negative dim at the
        // boundary BEFORE the explicit FeatureDim (unsigned u16) crossing — a negative dim is a malformed
        // spec, never a wrapped huge unsigned. (Mirrors instance.cpp's signed->unsigned-id boundary.)
        const int dim_raw = j.at("dim").get<int>();
        if (dim_raw < 0)
            return std::unexpected(make_error("feature-layout: negative dim (" + std::to_string(dim_raw) + ")"));
        spec.dim_ = FeatureDim{static_cast<LayoutRep>(dim_raw)};
        FeatureDim o{0};  // running Σwidth (cumulative offset); FeatureDim is additive (offset + width)
        for (const auto& b : j.at("blocks")) {
            FeatureBlock blk;
            blk.key = b.at("key").get<std::string>();
            // JSON loader ACL: fail-loud on a negative width at the boundary before the unsigned crossing.
            const int width_raw = b.at("width").get<int>();
            if (width_raw < 0)
                return std::unexpected(make_error("feature-layout: negative width for block '" + blk.key + "'"));
            blk.width = FeatureDim{static_cast<LayoutRep>(width_raw)};
            blk.start = o;
            o = o + blk.width;  // additive FeatureDim: the contiguous-partition cumulative start
            auto [it, inserted] = spec.index_.emplace(blk.key, static_cast<int>(spec.blocks_.size()));
            if (!inserted)
                return std::unexpected(make_error("feature-layout: duplicate block key '" + blk.key + "'"));
            spec.blocks_.push_back(std::move(blk));
        }
        // Cross-language consistency (ADR-0002): the shipped spec's `dim`, its Σwidth, and THIS env's
        // derived dim must ALL agree. Σwidth==dim with cumulative starts makes the blocks a contiguous
        // partition of [0, dim) by construction; ==expected_dim ties the shipped spec to this env. Any
        // disagreement is a silent-mislabel hazard, so it is a loud boundary Error here. Same-domain
        // FeatureDim == (the defaulted three-way over the rep); .value() at the diagnostic-string crossing.
        if (o != spec.dim_)
            return std::unexpected(make_error("feature-layout: Σwidth (" + std::to_string(o.value()) +
                                              ") != spec dim (" + std::to_string(spec.dim_.value()) + ")"));
        if (spec.dim_ != expected_dim)
            return std::unexpected(make_error("feature-layout: spec dim (" + std::to_string(spec.dim_.value()) +
                                              ") != env-derived dim (" + std::to_string(expected_dim.value()) +
                                              ") — the spec does not match this env"));
        return spec;
    } catch (const std::exception& e) {
        return std::unexpected(make_error(std::string("feature-layout: malformed spec: ") + e.what()));
    }
}

FeatureDim FeatureLayoutSpec::start(std::string_view key) const {
    auto it = index_.find(std::string(key));
    if (it == index_.end()) {
        // ADR-0012 P9: the writer names only keys the FeatureBuilder ctor verified the spec carries,
        // so a miss here is an impossible state (a programmer bug) — a loud abort, not a boundary Error.
        std::cerr << "chocofarm: FATAL invariant: feature-layout has no block '" << key << "'\n";
        std::abort();
    }
    return blocks_[it->second].start;
}

}  // namespace chocofarm
