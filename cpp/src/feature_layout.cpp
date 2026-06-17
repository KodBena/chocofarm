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
FeatureLayoutSpec::load(std::string_view path, int expected_dim) {
    std::ifstream f{std::string(path)};
    if (!f)
        return std::unexpected(make_error("cannot open feature-layout spec: " + std::string(path)));
    nlohmann::json j = nlohmann::json::parse(f, nullptr, /*allow_exceptions=*/false);
    if (j.is_discarded())
        return std::unexpected(make_error("malformed JSON in feature-layout spec: " + std::string(path)));
    try {
        FeatureLayoutSpec spec;
        spec.dim_ = j.at("dim").get<int>();
        int o = 0;
        for (const auto& b : j.at("blocks")) {
            FeatureBlock blk;
            blk.key = b.at("key").get<std::string>();
            blk.width = b.at("width").get<int>();
            if (blk.width < 0)
                return std::unexpected(make_error("feature-layout: negative width for block '" + blk.key + "'"));
            blk.start = o;
            o += blk.width;
            auto [it, inserted] = spec.index_.emplace(blk.key, static_cast<int>(spec.blocks_.size()));
            if (!inserted)
                return std::unexpected(make_error("feature-layout: duplicate block key '" + blk.key + "'"));
            spec.blocks_.push_back(std::move(blk));
        }
        // Cross-language consistency (ADR-0002): the shipped spec's `dim`, its Σwidth, and THIS env's
        // derived dim must ALL agree. Σwidth==dim with cumulative starts makes the blocks a contiguous
        // partition of [0, dim) by construction; ==expected_dim ties the shipped spec to this env. Any
        // disagreement is a silent-mislabel hazard, so it is a loud boundary Error here.
        if (o != spec.dim_)
            return std::unexpected(make_error("feature-layout: Σwidth (" + std::to_string(o) +
                                              ") != spec dim (" + std::to_string(spec.dim_) + ")"));
        if (spec.dim_ != expected_dim)
            return std::unexpected(make_error("feature-layout: spec dim (" + std::to_string(spec.dim_) +
                                              ") != env-derived dim (" + std::to_string(expected_dim) +
                                              ") — the spec does not match this env"));
        return spec;
    } catch (const std::exception& e) {
        return std::unexpected(make_error(std::string("feature-layout: malformed spec: ") + e.what()));
    }
}

int FeatureLayoutSpec::start(std::string_view key) const {
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
