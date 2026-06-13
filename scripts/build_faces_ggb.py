#!/usr/bin/env python3
"""
build_faces_ggb.py — the verification round-trip (consult-002 deliverable 2).

Copies the original chocobo.ggb and, in its geogebra.xml, REPLACES the 19 Δ
detection polygons with the 44 computed arrangement faces.  Each face becomes a
GeoGebra Polygon whose vertices are fresh points, captioned by its cover set
(e.g. {8,9}, {8,9,11,12}).  Everything else — the treasure points τ_*, the
teleports W/Z/τ_4, the background map image Bild1 and its embedded jpg, every
named annotation point — is kept, so the faces overlay the real map.

What is removed: each Δ Polygon command, its polygon element, and the segment
elements that command outputs (leaving them orphaned would invalidate the
construction).  Vertex/annotation points are kept (many are the maintainer's
own map labels; orphaned points are harmless dots).

The result is re-zipped as chocobo_faces.ggb with the image and thumbnail
preserved.  We cannot run GeoGebra, so validation checks: the zip is well
formed, the XML parses, and every face polygon references defined points.
"""
import os
import re
import zipfile
import xml.etree.ElementTree as ET
from shapely import wkt
from chocofarm.model import arrangement as A

SRC = "/home/bork/chocobo.ggb"
DST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chocobo_faces.ggb")
DELTA_CAP = re.compile(r"Δ")        # any Δ-captioned polygon is a detection region

# face fill: a translucent steel-blue, distinct from the Δ brown, so the maintainer
# can tell faces from any residual map ink.
FACE_COLOR = dict(r="0", g="102", b="204", alpha="0.18")


def _face_point_label(face_i: int, k: int) -> str:
    return f"FP_{{{face_i}_{k}}}"


def _face_poly_label(face_i: int) -> str:
    return f"Face_{{{face_i}}}"


def strip_delta_polygons(con: ET.Element) -> int:
    """Remove every Δ Polygon command, its polygon element, and its output
    segments.  Returns the number of Δ polygons removed."""
    # 1. find Polygon commands whose output polygon is Δ-captioned
    poly_caption = {}
    for el in con:
        if el.tag == "element" and el.get("type") == "polygon":
            cap = el.find("caption")
            poly_caption[el.get("label")] = cap.get("val") if cap is not None else ""

    drop_poly_labels = set()      # polygon element labels to drop
    drop_seg_labels = set()       # segment element labels to drop
    drop_cmds = []                # Polygon command elements to drop
    for el in con:
        if el.tag == "command" and el.get("name") == "Polygon":
            out = el.find("output")
            plabel = out.attrib["a0"]
            if DELTA_CAP.search(poly_caption.get(plabel, "")):
                drop_cmds.append(el)
                drop_poly_labels.add(plabel)
                # outputs a1.. are the edge segments
                for key in sorted(out.attrib):
                    if key != "a0":
                        drop_seg_labels.add(out.attrib[key])

    removed = 0
    for el in list(con):
        if el in drop_cmds:
            con.remove(el); removed += 1
        elif el.tag == "element" and el.get("type") == "polygon" and el.get("label") in drop_poly_labels:
            con.remove(el)
        elif el.tag == "element" and el.get("type") == "segment" and el.get("label") in drop_seg_labels:
            con.remove(el)

    # Helper commands (Intersect/Point) that consumed a removed Δ segment now
    # dangle.  Their *output* points carry GeoGebra's cached coords, so dropping
    # only the command demotes them to free points at their original location —
    # nothing visible moves, and the construction stays consistent.
    _drop_dangling_commands(con, drop_seg_labels | drop_poly_labels)
    return removed


def _drop_dangling_commands(con: ET.Element, gone: set) -> None:
    for el in list(con):
        if el.tag != "command" or el.get("name") == "Polygon":
            continue
        inp = el.find("input")
        if inp is not None and any(v in gone for v in inp.attrib.values()):
            con.remove(el)


def _point_element(label: str, x: float, y: float) -> ET.Element:
    el = ET.Element("element", {"type": "point", "label": label})
    ET.SubElement(el, "show", {"object": "false", "label": "false"})
    ET.SubElement(el, "objColor", {"r": "97", "g": "97", "b": "97", "alpha": "0"})
    ET.SubElement(el, "layer", {"val": "0"})
    ET.SubElement(el, "labelMode", {"val": "0"})
    ET.SubElement(el, "coords", {"x": repr(x), "y": repr(y), "z": "1"})
    return el


def _polygon_command(verts: list, poly_label: str, seg_labels: list) -> ET.Element:
    cmd = ET.Element("command", {"name": "Polygon"})
    inp = ET.SubElement(cmd, "input")
    for k, v in enumerate(verts):
        inp.set(f"a{k}", v)
    out = ET.SubElement(cmd, "output")
    out.set("a0", poly_label)
    for k, s in enumerate(seg_labels):
        out.set(f"a{k+1}", s)
    return cmd


def _polygon_element(poly_label: str, cover: frozenset) -> ET.Element:
    el = ET.Element("element", {"type": "polygon", "label": poly_label})
    ET.SubElement(el, "lineStyle", {"thickness": "2", "type": "0", "typeHidden": "1", "opacity": "178"})
    ET.SubElement(el, "show", {"object": "true", "label": "true"})
    ET.SubElement(el, "objColor", FACE_COLOR)
    ET.SubElement(el, "layer", {"val": "0"})
    ET.SubElement(el, "labelMode", {"val": "3"})        # 3 == caption
    ET.SubElement(el, "caption", {"val": "{" + ",".join(map(str, sorted(cover))) + "}"})
    return el


def _segment_element(seg_label: str) -> ET.Element:
    el = ET.Element("element", {"type": "segment", "label": seg_label})
    ET.SubElement(el, "show", {"object": "true", "label": "false"})
    ET.SubElement(el, "objColor", FACE_COLOR)
    ET.SubElement(el, "layer", {"val": "0"})
    ET.SubElement(el, "labelMode", {"val": "0"})
    ET.SubElement(el, "lineStyle", {"thickness": "2", "type": "0", "typeHidden": "1", "opacity": "178"})
    return el


def append_faces(con: ET.Element, faces: list) -> None:
    """Append, for each face: its vertex points, a Polygon command, the polygon
    element (captioned by cover), and the edge segment elements."""
    for fi, f in enumerate(faces):
        ring = list(wkt.loads(f.polygon_wkt).exterior.coords)[:-1]   # drop closing dup
        vlabels = []
        for k, (x, y) in enumerate(ring):
            lab = _face_point_label(fi, k)
            con.append(_point_element(lab, x, y))
            vlabels.append(lab)
        plabel = _face_poly_label(fi)
        slabels = [f"fe_{{{fi}_{k}}}" for k in range(len(vlabels))]
        con.append(_polygon_command(vlabels, plabel, slabels))
        con.append(_polygon_element(plabel, f.cover))
        for s in slabels:
            con.append(_segment_element(s))


def build() -> dict:
    faces = A.load()
    raw = zipfile.ZipFile(SRC).read("geogebra.xml").decode("utf-8")
    tree = ET.ElementTree(ET.fromstring(raw))
    root = tree.getroot()
    con = root.find("construction")

    removed = strip_delta_polygons(con)
    append_faces(con, faces)

    new_xml = ET.tostring(root, encoding="unicode")
    new_xml = '<?xml version="1.0" encoding="utf-8"?>\n' + new_xml

    # re-zip: replace geogebra.xml, copy every other member byte-for-byte
    with zipfile.ZipFile(SRC) as zin, zipfile.ZipFile(DST, "w", zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            if item.filename == "geogebra.xml":
                zout.writestr(item, new_xml)
            else:
                zout.writestr(item, zin.read(item.filename))

    return {"faces": len(faces), "deltas_removed": removed, "out": DST}


if __name__ == "__main__":
    r = build()
    print(f"removed {r['deltas_removed']} Δ polygons; wrote {r['faces']} faces -> "
          f"{os.path.basename(r['out'])}")
