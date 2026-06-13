#!/usr/bin/env python3
"""
verify_faces.py — self-checks for the arrangement face model (consult-002 §4).

Re-derives every figure in docs/design/face-model-verification.md from the
frozen instance, and validates the chocobo_faces.ggb round-trip. Bounded, read
only against the WKT; no solver runs. Run under `timeout`.
"""
import collections
import re
import zipfile
import xml.etree.ElementTree as ET
import json
from shapely import wkt
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union
from chocofarm.model import arrangement as A

GGB = "chocobo_faces.ggb"


def model_checks(regions, faces):
    print("=" * 68)
    print("FACE MODEL — self-check vs consult-002")
    print("=" * 68)

    # old (buggy) cover_mask: {i} ∪ overlap-neighbours
    data = json.load(open(A.INSTANCE))
    nbr = {i: {i} for i in regions}
    for a, b in data["overlaps"]:
        nbr[int(a)].add(int(b)); nbr[int(b)].add(int(a))

    print(f"\n[1] total faces = {len(faces)}   (consult-002 ~44)")

    by_k = collections.defaultdict(float)
    for f in faces:
        by_k[len(f.cover)] += f.area
    sigma = sum(f.area for f in faces)
    union_area = unary_union(list(regions.values())).area
    print(f"\n[2] cover-cardinality area distribution (Σ={sigma:.3f}):")
    for k in sorted(by_k):
        print(f"    k={k}: {by_k[k]:7.3f}  = {100 * by_k[k] / sigma:4.1f}%")
    print("    consult-002: 70.3 / 24.2 / 5.4 / 0.1 / 0")

    print(f"\n[3] tiling: Σ face area {sigma:.4f} vs union {union_area:.4f} "
          f"(Δ {abs(sigma - union_area):.1e})")

    ok = True
    for i in sorted(regions):
        cov = set()
        for f in faces:
            if regions[i].contains(Point(*f.rep_point)):
                cov |= f.cover
        ok &= (cov == nbr[i])
    print(f"\n[4] per-region ⋃(face covers) == old cover_mask, all 16: {ok}")

    k4 = [f for f in faces if len(f.cover) == 4]
    print(f"\n[5] k=4 faces: {len(k4)} -> {[sorted(f.cover) for f in k4]} "
          f"area {[round(f.area, 4) for f in k4]}  (consult-002: {{8,9,11,12}} ~0.052)")
    print(f"\n[6] max cover cardinality = {max(len(f.cover) for f in faces)}  (consult-002: 4)")


def ggb_checks(faces):
    print("\n" + "=" * 68)
    print("GGB ROUND-TRIP — chocobo_faces.ggb")
    print("=" * 68)
    z = zipfile.ZipFile(GGB)
    print(f"\n[zip] testzip (None=ok): {z.testzip()}  members: {len(z.namelist())}")
    con = ET.fromstring(z.read("geogebra.xml").decode("utf-8")).find("construction")

    pts, defined = {}, set()
    for el in con:
        if el.tag == "element":
            defined.add(el.get("label"))
            if el.get("type") == "point":
                c = el.find("coords"); zz = float(c.get("z"))
                pts[el.get("label")] = (float(c.get("x")) / zz, float(c.get("y")) / zz)
        elif el.tag == "command":
            out = el.find("output")
            for k in out.attrib:
                defined.add(out.attrib[k])

    img = [e for e in con if e.tag == "element" and e.get("type") == "image"]
    tau = sum(1 for e in con if e.tag == "element" and re.fullmatch(r"τ_\{?\d+\}?", e.get("label", "")))
    delta = face = 0
    cmd_pts, caps = {}, {}
    for el in con:
        if el.tag == "element" and el.get("type") == "polygon":
            cap = el.find("caption"); v = cap.get("val") if cap is not None else ""
            if "Δ" in v: delta += 1
            if el.get("label", "").startswith("Face_"):
                face += 1; caps[el.get("label")] = v
        if el.tag == "command" and el.get("name") == "Polygon":
            o = el.find("output"); i = el.find("input")
            cmd_pts[o.attrib["a0"]] = [i.attrib[k] for k in sorted(i.attrib, key=lambda s: int(s[1:]))]

    bad_inp = [v for el in con if el.tag == "command"
               for v in (el.find("input").attrib.values() if el.find("input") is not None else [])
               if v not in defined]

    print(f"[xml] image Bild1: {len(img)}   treasures: {tau}   Δ remaining: {delta}   Face: {face}")
    print(f"[xml] commands with undefined inputs: {len(bad_inp)}")

    # geometric identity of every face polygon
    def cover_of_cap(c):
        return frozenset(int(x) for x in re.findall(r"\d+", c))
    worst = 0.0
    for f in faces:
        cands = [Polygon([pts[v] for v in cmd_pts[pl]])
                 for pl, cp in caps.items() if cover_of_cap(cp) == f.cover]
        cg = wkt.loads(f.polygon_wkt)
        worst = max(worst, min((cg.symmetric_difference(g).area for g in cands), default=1.0))
    print(f"[geo] max symmetric-difference (ggb face vs computed), all 44: {worst:.2e}")


if __name__ == "__main__":
    regions = A.detection_regions()
    faces = A.load()
    model_checks(regions, faces)
    ggb_checks(faces)
