// cpp/src/instance.cpp
// Purpose: load the instance geometry (treasures / teleports / K from data/instance.json) and the
//   DERIVED arrangement faces (cover + rep_point from data/faces.json). See instance.hpp. The
//   fossil `overlaps` / `delta_treasures` arrays in instance.json are NEVER read — the disjunctive
//   cover comes only from faces.json (the geometry-derived intersection-refinement). ADR-0012 P1.
//
// Public Domain (The Unlicense).
#include "chocofarm/instance.hpp"

#include <fstream>
#include <stdexcept>

#include <nlohmann/json.hpp>

namespace chocofarm {

// IMPORTANT: use ordered_json (insertion-order-preserving) — the per-teleport feature block (dist_w)
// is ordered by the teleports' dict-INSERTION order in instance.json (Python dicts preserve it), and
// plain nlohmann::json sorts object keys alphabetically, which would permute the teleport block and
// silently diverge the feature vector (caught by the parity harness). treasures are re-keyed by
// integer id so their order is order-independent; teleports are NOT.
using json = nlohmann::ordered_json;

static json load_json(const std::string& path) {
    std::ifstream f(path);
    if (!f) {
        // ADR-0002 / P5: a missing instance file is a loud failure, not a silent default.
        throw std::runtime_error("chocofarm: cannot open instance file: " + path);
    }
    json j;
    f >> j;
    return j;
}

Instance load_instance(const std::string& instance_json, const std::string& faces_json) {
    json inst = load_json(instance_json);
    json faces = load_json(faces_json);

    Instance out;
    out.K = inst.at("K").get<int>();

    // treasures: {id-string -> [x, y]}. Ids are 0..N-1; place each at its integer id index so bit
    // position == treasure id (mirrors the Python loader's {int(i): tuple(xy)} + the bit layout).
    const json& tj = inst.at("treasures");
    out.N = static_cast<int>(tj.size());
    out.treasures.resize(out.N);
    for (auto it = tj.begin(); it != tj.end(); ++it) {
        int id = std::stoi(it.key());
        if (id < 0 || id >= out.N) {
            throw std::runtime_error("chocofarm: treasure id out of range in instance.json");
        }
        const json& xy = it.value();
        out.treasures[id] = Point{xy.at(0).get<double>(), xy.at(1).get<double>()};
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
            int t = j.get<int>();
            if (t < 0 || t >= out.N) {
                throw std::runtime_error("chocofarm: face cover bit out of range [0,N) in faces.json");
            }
            face.bitmask |= (uint32_t{1} << t);
        }
        const json& rp = f.at("rep_point");
        face.rep_point = Point{rp.at(0).get<double>(), rp.at(1).get<double>()};
        out.faces.push_back(face);
    }
    return out;
}

}  // namespace chocofarm
