// cpp/include/chocofarm/instance.hpp
// Purpose: the parsed chocofarm instance geometry — treasures, teleports, K, and the
//   geometry-DERIVED arrangement faces (the sense actions' cover + rep_point). Mirrors the
//   Python loader (chocofarm/model/instance.py + arrangement.py + facemodel.py): treasures /
//   teleports / K come from data/instance.json; the disjunctive cover structure comes from
//   data/faces.json (the intersection-refinement of the atomic detectors, DERIVED from the
//   geometry by scripts/chocobo_geometry.py + arrangement.arrangement()). This loader reads
//   ONLY treasures / teleports / K from instance.json and DERIVES the cover from faces.json —
//   it never touches instance.json's fossil `overlaps` / `delta_treasures` arrays (those encode
//   the superseded per-region cover_mask the face model replaced; ADR-0012 P1 derive-don't-
//   duplicate, and the maintainer's geometric-derivability constraint).
//
// Public Domain (The Unlicense).
#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace chocofarm {

// A single (x, y) coordinate. Distances use std::hypot (mirroring env.py's math.hypot), so the
// distance numbers are float-sensitive-but-equivalent across the language boundary (ADR-0012 P6).
struct Point {
    double x = 0.0;
    double y = 0.0;
};

// One arrangement-face sense action (mirrors facemodel.SenseAction). `bitmask` is the cover as
// bits (bit j set <=> treasure j in the face's cover) — built from the DERIVED face cover, never
// from instance.json's fossil arrays. The face id is the index in `faces`; the ("d", id) action
// shape is the env's legacy shape (env.py / actions.py).
struct Face {
    uint32_t bitmask = 0;   // sum over j in cover of (1 << j) — the disjunction read at rep_point
    Point rep_point;        // where you stand to read this face (det_pt[id])
};

// The Tier-1 instance geometry (one home; ADR-0012 P1). N is derived from the treasure count,
// never stored (mirrors Instance.N).
struct Instance {
    int N = 0;                       // number of treasures (= treasures.size())
    int K = 0;                       // exactly-K-of-N present count
    std::vector<Point> treasures;    // treasures[i] = coord of treasure id i (ids are 0..N-1)
    std::vector<std::string> teleport_names;  // teleport order as in instance.json (insertion order)
    std::vector<Point> teleports;    // teleports[k] = coord of teleport teleport_names[k]
    std::string entry = "CSNE";      // entry teleport name (env default)
    double teleport_overhead = 12.0; // the fixed exit/teleport surcharge (env default tp)
    std::vector<Face> faces;         // the DERIVED sense actions (44 on the live instance)
};

// Load the instance geometry. `instance_json` is data/instance.json (treasures / teleports / K);
// `faces_json` is data/faces.json (the DERIVED face cover + rep_point). A missing/malformed file or
// a face cover bit >= N is a loud std::runtime_error (ADR-0002 / P5: never silently coerce).
Instance load_instance(const std::string& instance_json, const std::string& faces_json);

}  // namespace chocofarm
