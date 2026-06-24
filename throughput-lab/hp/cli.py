#!/usr/bin/env python3
"""
throughput-lab/hp/cli.py — the command-line front end for the HP config-space compiler.

    python -m hp.cli --select <surface> [--pin k=v ...] [--variant real|synthetic]
                     [--verify] [--json out.json] [--gens G --cores C]

Selects a sub-space of the SSOT (spec.py), compiles it to a ConfigSpace (compile.py), enumerates
the feasible symmetry-reduced candidate set (backends/cpsat.py), and writes the candidate set + a
PROVENANCE LEDGER (which axes are measured / hypothesized / unknown — effects ANNOTATE, never prune;
DESIGN.md §0). `--verify` runs both oracles + the inertness self-check and returns NON-ZERO on any
divergence, refusing to emit (ADR-0002) — mirroring topology_enum.py --verify returning 3.

Public Domain (The Unlicense).
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Optional

from . import compile as cc
from . import spec, verify
from . import topology_materialize as tm
from .backends import cpsat
from .relations import TopologyParams
from .spec import Surface


_SURFACES = {s.value: s for s in Surface}


def _decode(rec: dict[str, int], cs) -> dict:
    """Decode a projection record's index-coded values back to human-facing values."""
    out: dict = {}
    for v in cs.vars:
        if v.id not in cs.projection:
            continue
        raw = rec[v.id]
        if v.enum_values:
            out[v.id] = v.enum_values[raw]
        elif v.kind.value == "bool":
            out[v.id] = bool(raw)
        else:
            out[v.id] = raw
    return out


def _parse_pin(items: list[str], reg: spec.Registry) -> dict[str, object]:
    pins: dict[str, object] = {}
    for it in items:
        if "=" not in it:
            raise SystemExit(f"--pin expects k=v, got {it!r}")
        k, v = it.split("=", 1)
        if k not in reg:
            raise SystemExit(f"--pin: unknown HP {k!r}")
        dom = reg[k].domain
        # coerce v to the domain's value type.
        if isinstance(dom, spec.Bool):
            pins[k] = v.lower() in ("1", "true", "yes", "on")
        elif isinstance(dom, (spec.IntRange, spec.IntSet)):
            pins[k] = int(v)
        else:
            pins[k] = v
    return pins


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--select", required=True,
                    help="surface to compile: " + " | ".join(_SURFACES) +
                         " (comma-separate to compose)")
    ap.add_argument("--include", default="",
                    help="explicit comma-separated HP names to add to the selection")
    ap.add_argument("--pin", action="append", default=[], metavar="k=v",
                    help="finalize an axis (repeatable)")
    ap.add_argument("--variant", default=None, help="producer variant: real | synthetic")
    ap.add_argument("--gens", type=int, default=3, help="topology: number of generators")
    ap.add_argument("--cores", type=int, default=4, help="topology: number of vCPUs")
    ap.add_argument("--housekeeping-core", type=int, default=0)
    ap.add_argument("--verify", action="store_true",
                    help="run both oracles + the inertness self-check; non-zero on divergence")
    ap.add_argument("--json", default=None, help="write the candidate set + ledger here")
    ap.add_argument("--no-table", action="store_true")
    args = ap.parse_args(argv)

    reg = spec.registry()
    surfaces = set()
    for s in args.select.split(","):
        s = s.strip()
        if s not in _SURFACES:
            print(f"error: unknown surface {s!r}; choices: {', '.join(_SURFACES)}", file=sys.stderr)
            return 2
        surfaces.add(_SURFACES[s])

    include = frozenset(n.strip() for n in args.include.split(",") if n.strip()) or None
    pins = _parse_pin(args.pin, reg)
    topo = TopologyParams(n_cores=args.cores, n_gens=args.gens,
                          housekeeping_core=args.housekeeping_core)
    target = cc.Target(surfaces=frozenset(surfaces), include=include, pin=pins,
                       variant=args.variant, topo=topo)

    try:
        cs = cc.compile(reg, target)
    except ValueError as e:
        print(f"error: selection refused: {e}", file=sys.stderr)
        return 2

    emitted = cpsat.enumerate_configs(cs)

    if args.verify:
        ok, msgs = verify.verify_all(cs, emitted)
        for m in msgs:
            print(m, file=sys.stderr)
        if not ok:
            print("ERROR: config-space reduction is unsound — refusing to emit (ADR-0002)",
                  file=sys.stderr)
            return 3

    # decode + (for TOPOLOGY) materialize the runnable record.
    is_topo_only = surfaces == {Surface.TOPOLOGY}
    configs = []
    for rec in emitted:
        decoded = _decode(rec, cs)
        entry = {"config": decoded}
        if is_topo_only:
            entry["topology"] = tm.materialize(rec, topo)
        configs.append(entry)

    ledger = {name: prov for name, prov in cs.provenance.items()}

    payload = {
        "schema": "throughput-lab/hp/candidate-set/v1",
        "selection": {"surfaces": sorted(s.value for s in surfaces),
                      "include": sorted(include) if include else None,
                      "pin": {k: (v if not isinstance(v, bool) else bool(v)) for k, v in pins.items()},
                      "variant": args.variant,
                      "topology": {"n_cores": topo.n_cores, "n_gens": topo.n_gens,
                                   "housekeeping_core": topo.housekeeping_core}},
        "n_configs": len(configs),
        "provenance_ledger": ledger,
        "configs": configs,
    }

    if args.json:
        with open(args.json, "w") as fh:
            json.dump(payload, fh, indent=2)
        print(f"wrote {len(configs)} configs -> {args.json}", file=sys.stderr)

    if not args.no_table:
        print(f"{len(configs)} feasible configs for selection {sorted(s.value for s in surfaces)}")
        for i, c in enumerate(configs):
            if is_topo_only:
                print(f"[{i:>3}] {c['topology']['config_id']}  ({c['topology']['tag']})")
            else:
                print(f"[{i:>3}] " + "  ".join(f"{k}={v}" for k, v in c["config"].items()))
        print("\n-- provenance ledger (effects ANNOTATE, never prune) --")
        for name, prov in sorted(ledger.items()):
            print(f"  {name:<22} {prov.get('effect','?'):<12} {prov.get('note','')[:60]}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
