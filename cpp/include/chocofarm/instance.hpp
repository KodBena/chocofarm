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
#include <expected>
#include <string>
#include <string_view>
#include <vector>

#include "chocofarm/domains.hpp"  // TreasureCount(N)/PresentCount(K)/TreasureId + World — the typed instance domains (P1)
#include "chocofarm/error.hpp"

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
    World bitmask = 0;      // sum over j in cover of (1 << j) — the disjunction read at rep_point. `World`
                            // (= the uint32 per-treasure mask, world.hpp) names what this bitfield IS:
                            // a treasure-id bitset, the SAME domain env.observe ANDs against (ADR-0000;
                            // the former bare uint32 was a silent re-author of the world-mask type).
    Point rep_point;        // where you stand to read this face (det_pt[id])
};

// The Tier-1 instance geometry (one home; ADR-0012 P1). N is derived from the treasure count,
// never stored (mirrors Instance.N).
struct Instance {
    // N is the treasure-universe cardinality (= treasures.size()); K the exactly-K-of-N present count.
    // Typed as the count domains (domains.hpp): N a TreasureCount, K a PresentCount — DISTINCT tags, so a
    // K used where an N is owed no longer compiles (ADR-0000; the former bare int pair conflated them).
    // Both default to 0 (the zero-init Quantity ctor) — an unloaded instance has no treasures (mirrors
    // Instance.N's derive-from-count).
    TreasureCount N{};               // number of treasures (= treasures.size())
    PresentCount K{};                // exactly-K-of-N present count
    std::vector<Point> treasures;    // treasures[i] = coord of treasure id i (ids are 0..N-1)
    std::vector<std::string> teleport_names;  // teleport order as in instance.json (insertion order)
    std::vector<Point> teleports;    // teleports[k] = coord of teleport teleport_names[k]
    std::string entry = "CSNE";      // entry teleport name (env default)
    double teleport_overhead = 12.0; // the fixed exit/teleport surcharge (env default tp)
    std::vector<Face> faces;         // the DERIVED sense actions (44 on the live instance)
};

// Load the instance geometry. `instance_json` is data/instance.json (treasures / teleports / K);
// `faces_json` is data/faces.json (the DERIVED face cover + rep_point). A missing/malformed file or
// a face cover bit >= N is a typed boundary failure (ADR-0002 / P5 + ADR-0012 P9 rule 5: a
// [[nodiscard]] std::expected<Instance, Error> returned by value — the loud failure is the Error
// the shell prints, never a silent coerce and never a thrown exception).
[[nodiscard]] std::expected<Instance, Error> load_instance(std::string_view instance_json,
                                                           std::string_view faces_json);

}  // namespace chocofarm
