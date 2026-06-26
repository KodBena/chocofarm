// cpp/src/actor_config.cpp
// Purpose: actor_config_from_json — the Port/ACL parse of a `configure` message's "config" object into a
//   validated ActorConfig (ADR-0012 P2 / P9 rule 5: validate-don't-coerce — a typed Error returned by
//   value, never a throw or a coerced default). The field set it reads is the one drift-netted against
//   the Python actor_config.py (tests/test_wire_drift.py); the domain checks mirror schema.check_invariants.
//   nlohmann's accessor exceptions are translated into a typed Error at this boundary (the same
//   discipline instance.cpp uses), so the search/serve core stays throw-free.
//
// Public Domain (The Unlicense).
#include "chocofarm/actor_config.hpp"

#include <string>

namespace chocofarm {

std::expected<ActorConfig, Error> actor_config_from_json(const nlohmann::json& j) {
    if (!j.is_object())
        return std::unexpected(make_error("config must be a JSON object"));
    ActorConfig c;
    try {
        c.instance_path = j.at("instance_path").get<std::string>();
        c.faces_path = j.at("faces_path").get<std::string>();
        // JSON get<int>() is the boundary decode; wrap into the config's typed domains at this ACL (P2).
        c.gumbel.m = CandidateCount{static_cast<SearchRep>(j.at("m").get<int>())};
        c.gumbel.n_sims = SimBudget{static_cast<SearchRep>(j.at("n_sims").get<int>())};
        c.gumbel.c_puct = j.at("c_puct").get<double>();
        c.gumbel.c_visit = j.at("c_visit").get<double>();
        c.gumbel.c_scale = j.at("c_scale").get<double>();
        c.gumbel.c_outcome = OutcomeIndex{static_cast<SearchRep>(j.at("c_outcome").get<int>())};
        c.gumbel.max_depth = PlyDepth{static_cast<SearchRep>(j.at("max_depth").get<int>())};
    } catch (const nlohmann::json::exception& e) {
        // a missing required field (out_of_range) or a wrong type (type_error) — translate into a typed
        // Error at the boundary (P9: the core stays throw-free; the shell reports loudly, ADR-0002).
        return std::unexpected(make_error(std::string("config field error: ") + e.what()));
    }
    // domain validation (validate-don't-coerce; mirrors schema.check_invariants' positive-count + depth
    // domains): a zero/negative budget or an empty geometry path is a typed Error, never a silently-wrong
    // search (an m=0 SH bracket, a missing instance file).
    if (c.instance_path.empty())
        return std::unexpected(make_error("config.instance_path is empty"));
    if (c.faces_path.empty())
        return std::unexpected(make_error("config.faces_path is empty"));
    // domain checks against the typed minima; .value() at the std::to_string diagnostic boundary.
    if (c.gumbel.m < CandidateCount{1})
        return std::unexpected(make_error("config.m must be >= 1, got " + std::to_string(c.gumbel.m.value())));
    if (c.gumbel.n_sims < SimBudget{1})
        return std::unexpected(make_error("config.n_sims must be >= 1, got " + std::to_string(c.gumbel.n_sims.value())));
    if (c.gumbel.c_outcome < OutcomeIndex{1})
        return std::unexpected(make_error("config.c_outcome must be >= 1, got " + std::to_string(c.gumbel.c_outcome.value())));
    if (c.gumbel.max_depth < PlyDepth{1})
        return std::unexpected(make_error("config.max_depth must be >= 1, got " + std::to_string(c.gumbel.max_depth.value())));
    return c;
}

}  // namespace chocofarm
