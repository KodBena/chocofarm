#!/usr/bin/env python3
"""
Stage 2a -- parse the real chocobo.ggb geometry and extract the model the solver needs.

Pulls from the GeoGebra construction:
  * treasures tau_0..tau_19 (coords),
  * teleports: CSNE (point W), CSCE (point Z), and tau_4 (per the maintainer),
  * detection regions Delta_i (polygons captioned 'Delta_{i}', possibly multi-part:
    Delta_6 = main+aux, Delta_7 = main+aux_0+aux_1) -> region detects tau_i,
  * delta singularities: treasures with NO polygon (3, 4, 16, 19) -> observe==collect
    at the treasure point (tau_3/tau_4 per the maintainer; 16/19 have no Delta region).

Then, via shapely, the DISJUNCTIVE faces: pairs (i,j) whose regions overlap (standing
there detects tau_i OR tau_j), plus delta-points that fall inside another region (the
tau_16-in-Delta_1 overlap the maintainer flagged).  Approximations, stated loudly:
travel is Euclidean+symmetric (real terrain is asymmetric); map coords are approximate.
"""
import zipfile
import os
import re
import json
import xml.etree.ElementTree as ET
import math
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union

GGB = "/home/bork/chocobo.ggb"
DELTA_RE = re.compile(r"Δ\s*_?\{?(\d+)")          # 'Δ' then first integer
TAU_RE = re.compile(r"τ_\{?(\d+)\}?")            # 'τ_{n}'
DELTAPT_RE = re.compile(r"δ_\{?(\d+)")           # 'δ_{n,..}'


def load():
    xml = zipfile.ZipFile(GGB).read("geogebra.xml").decode("utf-8")
    con = ET.fromstring(xml).find("construction")

    pts = {}            # label -> (x, y)   (dehomogenised)
    caps = {}           # label -> caption
    polys = []          # (poly_label, [vertex labels])
    for el in con:
        if el.tag == "element":
            lab = el.get("label")
            cap = el.find("caption")
            if cap is not None:
                caps[lab] = cap.get("val")
            c = el.find("coords")
            if el.get("type") == "point" and c is not None:
                x, y, z = float(c.get("x")), float(c.get("y")), float(c.get("z"))
                pts[lab] = (x / z, y / z)
        elif el.tag == "command" and el.get("name") == "Polygon":
            inp, out = el.find("input"), el.find("output")
            verts = [inp.attrib[k] for k in sorted(inp.attrib)]
            poly_label = out.attrib["a0"]
            polys.append((poly_label, verts))

    treasures = {}
    for lab, xy in pts.items():
        m = TAU_RE.fullmatch(lab) or TAU_RE.match(lab)
        if m:
            treasures[int(m.group(1))] = xy

    teleports = {"CSNE": pts["W"], "CSCE": pts["Z"], "tau_4": treasures[4]}

    # detection regions Delta_i  (union the multi-part ones)
    region_parts = {}   # i -> list of shapely Polygons
    for poly_label, verts in polys:
        cap = caps.get(poly_label, "")
        m = DELTA_RE.search(cap)
        if not m:
            continue
        i = int(m.group(1))
        ring = [pts[v] for v in verts if v in pts]
        if len(ring) >= 3:
            region_parts.setdefault(i, []).append(Polygon(ring).buffer(0))  # repair non-simple rings
    regions = {i: unary_union(parts).buffer(0) for i, parts in region_parts.items()}

    # delta singularity points (for the tau_16 / tau_1 style overlaps)
    delta_pts = {}      # i -> list of points
    for lab, cap in caps.items():
        m = DELTAPT_RE.search(cap or "")
        if m and lab in pts:
            delta_pts.setdefault(int(m.group(1)), []).append(pts[lab])

    return treasures, teleports, regions, delta_pts, caps


def main():
    treasures, teleports, regions, delta_pts, caps = load()

    have_region = sorted(regions)
    delta_treasures = sorted(set(treasures) - set(regions))

    print("=" * 74)
    print("Stage 2a -- parsed chocobo.ggb geometry")
    print("=" * 74)
    print(f"treasures              : {len(treasures)}  (tau_0 .. tau_{max(treasures)})")
    print(f"detection regions D_i  : {len(regions)}  for tau in {have_region}")
    print(f"delta-singularity tau  : {delta_treasures}   "
          f"(no advance region -> observe==collect on arrival)")
    print(f"delta marker points    : "
          + ", ".join(f"delta_{i}:{len(p)}pt" for i, p in sorted(delta_pts.items())))
    print(f"teleports              : "
          + ", ".join(f"{k}=({x:.2f},{y:.2f})" for k, (x, y) in teleports.items()))

    # ---- disjunctive faces: overlapping detection regions ----
    print("\n--- DISJUNCTIVE faces (overlapping regions -> detection is tau_i OR tau_j) ---")
    ids = sorted(regions)
    overlaps = []
    for a in range(len(ids)):
        for b in range(a + 1, len(ids)):
            i, j = ids[a], ids[b]
            inter = regions[i].intersection(regions[j])
            if inter.area > 1e-6:
                frac_i = inter.area / regions[i].area
                frac_j = inter.area / regions[j].area
                overlaps.append((i, j, inter.area, frac_i, frac_j))
    if overlaps:
        for i, j, area, fi, fj in sorted(overlaps, key=lambda t: -t[2]):
            print(f"  D_{i} & D_{j}: overlap area {area:.3f}  "
                  f"({fi*100:.0f}% of D_{i}, {fj*100:.0f}% of D_{j})")
    else:
        print("  (none -- all detection regions are disjoint)")

    # ---- delta points landing inside another region (the tau_16-in-D_1 case) ----
    print("\n--- delta-point detections inside another region ---")
    found = False
    for i, ptlist in sorted(delta_pts.items()):
        for (x, y) in ptlist:
            for j, reg in regions.items():
                if reg.contains(Point(x, y)):
                    print(f"  delta_{i} point ({x:.2f},{y:.2f}) lies inside D_{j}  "
                          f"-> being in D_{j} can also detect tau_{i}  (tau_{i} OR tau_{j})")
                    found = True
    if not found:
        print("  (no delta marker falls strictly inside a detection region)")

    # ---- teleport geometry ----
    print("\n--- teleport geometry ---")
    tk = list(teleports)
    for a in range(len(tk)):
        for b in range(a + 1, len(tk)):
            (x1, y1), (x2, y2) = teleports[tk[a]], teleports[tk[b]]
            print(f"  d({tk[a]}, {tk[b]}) = {math.hypot(x1-x2, y1-y2):.2f}")
    cx = sum(x for x, y in treasures.values()) / len(treasures)
    cy = sum(y for x, y in treasures.values()) / len(treasures)
    print(f"  treasure centroid ~ ({cx:.2f},{cy:.2f})")
    for k, (x, y) in teleports.items():
        near = sorted(treasures, key=lambda t: math.hypot(treasures[t][0]-x, treasures[t][1]-y))[:4]
        print(f"  {k}: dist-to-centroid {math.hypot(x-cx, y-cy):.2f}, nearest treasures {near}")

    # ---- persist for the solver stage ----
    out = {
        "treasures": {str(i): xy for i, xy in treasures.items()},
        "teleports": teleports,
        "regions_wkt": {str(i): regions[i].wkt for i in regions},
        "delta_treasures": delta_treasures,
        "overlaps": [[i, j] for i, j, *_ in overlaps],
    }
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "..", "chocofarm", "data", "instance.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved parsed instance -> {os.path.abspath(out_path)}")


if __name__ == "__main__":
    main()
