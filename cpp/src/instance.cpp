// cpp/src/instance.cpp
// Purpose: load the instance geometry (treasures / teleports / K from data/instance.json) and the
//   DERIVED arrangement faces (cover + rep_point from data/faces.json). See instance.hpp. The
//   fossil `overlaps` / `delta_treasures` arrays in instance.json are NEVER read — the disjunctive
//   cover comes only from faces.json (the geometry-derived intersection-refinement). ADR-0012 P1.
//
//   ADR-0012 P9 (rule 5): a missing/malformed file is a typed boundary failure returned as
//   std::expected<Instance, Error> — the shell prints it loudly (ADR-0002), never a throw. The
//   nlohmann json accessors (at/get) throw on a malformed document; the loader catches them at THIS
//   boundary and translates them into the typed Error, so the public contract is total (throw-free).
//
// Public Domain (The Unlicense).
#include "chocofarm/instance.hpp"

#include <fstream>
#include <stdexcept>   // std::invalid_argument (from std::stoi on a non-integer treasure key)
#include <string>

#include <nlohmann/json.hpp>

namespace chocofarm {

// IMPORTANT: use ordered_json (insertion-order-preserving) — the per-teleport feature block (dist_w)
// is ordered by the teleports' dict-INSERTION order in instance.json (Python dicts preserve it), and
// plain nlohmann::json sorts object keys alphabetically, which would permute the teleport block and
// silently diverge the feature vector (caught by the parity harness). treasures are re-keyed by
// integer id so their order is order-independent; teleports are NOT.
using json = nlohmann::ordered_json;

namespace {

// Load + parse a JSON file. A missing file or a parse error is a typed Error (P9 rule 5).
std::expected<json, Error> load_json(std::string_view path) {
    std::ifstream f{std::string(path)};
    if (!f) {
        // ADR-0002 / P5: a missing instance file is a loud failure, not a silent default.
        return std::unexpected(make_error("chocofarm: cannot open instance file: " + std::string(path)));
    }
    json j = json::parse(f, nullptr, /*allow_exceptions=*/false);
    if (j.is_discarded())
        return std::unexpected(make_error("chocofarm: malformed JSON in " + std::string(path)));
    return j;
}

// The actual parse, factored so the boundary try/catch translates nlohmann's accessor exceptions
// (a missing key, a type mismatch) into a typed Error rather than letting them escape (P9 rule 5:
// the public contract is throw-free; the JSON library's exceptions are caught HERE at the edge).
std::expected<Instance, Error> parse_instance(const json& inst, const json& faces) {
    Instance out;
    // JSON boundary ACL (ADR-0002 / domains.hpp): nlohmann get<int> parses K SIGNED; a negative K is a
    // malformed instance — fail loud HERE, before the explicit PresentCount(K) crossing into the unsigned
    // domain (a silent signed->unsigned wrap would smuggle a garbage K past the count domain).
    const int k_raw = inst.at("K").get<int>();
    if (k_raw < 0)
        return std::unexpected(make_error("chocofarm: negative K in instance.json"));
    out.K = PresentCount{static_cast<TreasureRep>(k_raw)};

    // treasures: {id-string -> [x, y]}. Ids are 0..N-1; place each at its integer id index so bit
    // position == treasure id (mirrors the Python loader's {int(i): tuple(xy)} + the bit layout).
    const json& tj = inst.at("treasures");
    // .size() ACL (domains.hpp): std::map .size() is size_t, non-negative by construction -> the named
    // size_t->TreasureRep narrowing into the TreasureCount domain via the explicit ctor.
    out.N = TreasureCount{static_cast<TreasureRep>(tj.size())};
    // K <= N is the structural exactly-K-of-N invariant (|worlds| = C(N,K)); a K>N instance is malformed.
    // Fail loud at the boundary (ADR-0002) — same-domain compare is the typed Quantity <=> (K and N share
    // the TreasureRep family, but are DISTINCT tags, so unwrap to the shared rep for the cardinality check).
    if (out.K.value() > out.N.value())
        return std::unexpected(make_error("chocofarm: K > N in instance.json (exactly-K-of-N violated)"));
    out.treasures.resize(out.N.value());  // count -> container size: the count's raw extent
    for (auto it = tj.begin(); it != tj.end(); ++it) {
        // std::stoi parses the treasure-id key SIGNED (JSON boundary ACL); range-check against [0,N) before
        // the explicit TreasureId crossing (a negative or out-of-range id is a malformed instance).
        const int id_raw = std::stoi(it.key());
        if (id_raw < 0 || id_raw >= static_cast<int>(out.N.value()))
            return std::unexpected(make_error("chocofarm: treasure id out of range in instance.json"));
        const TreasureId id{static_cast<TreasureRep>(id_raw)};
        const json& xy = it.value();
        out.treasures[id.value()] = Point{xy.at(0).get<double>(), xy.at(1).get<double>()};
    }

    // teleports: {name -> [x, y]} (insertion order, as Python dict preserves it).
    const json& wj = inst.at("teleports");
    for (auto it = wj.begin(); it != wj.end(); ++it) {
        out.teleport_names.push_back(it.key());
        const json& xy = it.value();
        out.teleports.push_back(Point{xy.at(0).get<double>(), xy.at(1).get<double>()});
    }

    // faces: the DERIVED cover (a sorted list of treasure ids) + rep_point. Bitmask = sum(1<<j).
    const json& fj = faces.at("faces");
    out.faces.reserve(fj.size());
    for (const auto& f : fj) {
        Face face;
        for (const auto& j : f.at("cover")) {
            // JSON boundary ACL: a cover entry is a SIGNED get<int> treasure id; range-check [0,N) before
            // the TreasureId crossing, then set bit t in the World mask (the world.hpp domain Face::bitmask
            // now names). The 1<<t shift stays on the raw World rep (a bit position, not a count).
            const int t_raw = j.get<int>();
            if (t_raw < 0 || t_raw >= static_cast<int>(out.N.value()))
                return std::unexpected(make_error("chocofarm: face cover bit out of range [0,N) in faces.json"));
            const TreasureId t{static_cast<TreasureRep>(t_raw)};
            face.bitmask |= (World{1} << t.value());  // bit t of the per-treasure World mask
        }
        const json& rp = f.at("rep_point");
        face.rep_point = Point{rp.at(0).get<double>(), rp.at(1).get<double>()};
        out.faces.push_back(face);
    }
    return out;
}

}  // namespace

std::expected<Instance, Error> load_instance(std::string_view instance_json,
                                             std::string_view faces_json) {
    auto inst = load_json(instance_json);
    if (!inst) return std::unexpected(inst.error());
    auto faces = load_json(faces_json);
    if (!faces) return std::unexpected(faces.error());
    try {
        return parse_instance(*inst, *faces);
    } catch (const json::exception& e) {
        // Translate a malformed-document accessor failure (missing key / wrong type / bad treasure
        // id string) into the typed boundary Error — the JSON library's only escape, caught at the
        // edge so the public load_instance contract stays total (ADR-0012 P9 rule 5).
        return std::unexpected(make_error(std::string("chocofarm: malformed instance/faces JSON: ") + e.what()));
    } catch (const std::invalid_argument& e) {
        // std::stoi on a non-integer treasure key.
        return std::unexpected(make_error(std::string("chocofarm: bad treasure id key in instance.json: ") + e.what()));
    }
}

}  // namespace chocofarm
