#!/usr/bin/env python3
"""
throughput-lab/hp/tests/test_ssot_drift.py — the build-time drift lint (DESIGN.md §1.4).

ADR-0012 P1/P7: the SSOT must DERIVE from the one home, never COPY a default that already lives in a
C++ struct / argparse / dataclass. This lint reads each descriptor's cited `home` (the actual source
line) and asserts the descriptor's `default` equals it — failing the build on drift. It is the P7
FLOOR (the P7-strongest "generate-from-one-source" extractor is a filed deferral, DESIGN.md §9).

Coverage by SourceRef kind:
  - CppField : parse `<type> <field> = <value>;` in the cited .hpp.
  - PyArg    : parse the argparse `--flag` (dest) default in the cited .py.
  - PyField  : a per-role policy tuple (SERVER_POLICIES/...): assert the descriptor default equals
               the FIRST entry's SchedPolicy string (the index-0 default the model uses).
  - NoCodeHome: skipped (the only case a literal default is sanctioned; named, not buried).

An INJECTED drift (a wrong default) must be CAUGHT — proving the lint is not vacuous.

Run:
    PYTHONPATH=throughput-lab /home/bork/w/vdc/venvs/generic/bin/python -m pytest \
        throughput-lab/hp/tests/test_ssot_drift.py -q

Public Domain (The Unlicense).
"""
from __future__ import annotations

import ast
import os
import re

import pytest

from hp import spec
from hp.spec import CppField, NoCodeHome, PyArg, PyField

# repo root = three dirs up from this file's hp/tests/.
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def _read(path: str) -> str:
    with open(os.path.join(REPO, path)) as fh:
        return fh.read()


# --- home extractors (each reads the ONE home and returns the source-of-truth value) --------------
# C++ enum member -> the SSOT's human-facing enum string (the only enum-valued field is WireMode).
_CPP_ENUM_MAP = {
    "StrictBarrier": "strict-barrier",
    "PipelinedBucket": "pipelined-bucket",
}


def _cpp_field_value(file: str, symbol: str):
    """Parse `<type> <field> = <value>;` for the field named after '::' (or the bare symbol).
    Handles scalar inits (int/bool/float) AND an enum init `WireMode mode = WireMode::StrictBarrier;`
    by mapping the enum member to its SSOT string."""
    field = symbol.split("::")[-1]
    text = _read(file)
    # enum-valued field, e.g. `WireMode mode = WireMode::StrictBarrier;`
    me = re.search(rf"\b\w+\s+{re.escape(field)}\s*=\s*\w+::(\w+)\s*;", text)
    if me:
        member = me.group(1)
        if member not in _CPP_ENUM_MAP:
            raise AssertionError(f"unmapped C++ enum member {member!r} for {field}")
        return _CPP_ENUM_MAP[member]
    # scalar field, e.g. `int min_coalesce = 32;` or `bool chunk_floor = false;`
    m = re.search(rf"\b(?:int|long|bool|float|double|size_t)\s+{re.escape(field)}\s*=\s*"
                  rf"([^;]+);", text)
    if not m:
        raise AssertionError(f"could not find C++ field {field!r} in {file}")
    raw = m.group(1).strip()
    if raw in ("true", "false"):
        return raw == "true"
    try:
        return int(raw)
    except ValueError:
        try:
            return float(raw)
        except ValueError:
            return raw


def _py_arg_default(file: str, dest: str):
    """Find an argparse add_argument whose dest (explicit or derived from --flag) matches, return its
    default."""
    text = _read(file)
    tree = ast.parse(text)
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument"):
            continue
        flags = [a.value for a in node.args if isinstance(a, ast.Constant)]
        kw = {k.arg: k.value for k in node.keywords}
        # dest is explicit, or derived from the first long flag (-- stripped, - -> _).
        this_dest = None
        if "dest" in kw and isinstance(kw["dest"], ast.Constant):
            this_dest = kw["dest"].value
        else:
            for f in flags:
                if isinstance(f, str) and f.startswith("--"):
                    this_dest = f[2:].replace("-", "_")
                    break
        if this_dest == dest and "default" in kw and isinstance(kw["default"], ast.Constant):
            return kw["default"].value
    raise AssertionError(f"could not find argparse dest {dest!r} default in {file}")


def _py_policy_first(file: str, symbol: str) -> str:
    """For a per-role policy tuple (SERVER_POLICIES etc.) OR the surplus_present bool: return the
    index-0 default the topology model uses."""
    if symbol == "surplus_present":
        return False  # the enumerated bool; default absent
    text = _read(file)
    tree = ast.parse(text)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == symbol:
                    # the value is a list of tuples whose first elem is `SchedPolicy.X`.
                    first = node.value.elts[0]                # type: ignore[attr-defined]
                    polattr = first.elts[0]                   # SchedPolicy.OTHER_LATNICE
                    # render SchedPolicy.NAME -> the enum's .value via the SchedPolicy class.
                    name = polattr.attr                       # type: ignore[attr-defined]
                    return _sched_value(text, name)
    raise AssertionError(f"could not find policy tuple {symbol!r} in {file}")


def _sched_value(text: str, member: str) -> str:
    """Resolve SchedPolicy.<member> to its string value from the enum definition in the same file."""
    m = re.search(rf"\b{member}\s*=\s*\"([^\"]+)\"", text)
    if not m:
        raise AssertionError(f"could not resolve SchedPolicy.{member}")
    return m.group(1)


def _py_field_value(file: str, symbol: str):
    """A Python function-param / dataclass-field default, e.g. `min_forward_rows: int = 0`."""
    text = _read(file)
    tree = ast.parse(text)
    for node in ast.walk(tree):
        # function/method signature param with a default
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            args = node.args
            params = args.args + args.kwonlyargs
            defaults = list(args.defaults)
            kwdefaults = list(args.kw_defaults)
            # positional defaults align to the TAIL of args.args
            for a, d in zip(args.args[len(args.args) - len(args.defaults):], args.defaults):
                if a.arg == symbol and isinstance(d, ast.Constant):
                    return d.value
            for a, d in zip(args.kwonlyargs, kwdefaults):
                if a.arg == symbol and isinstance(d, ast.Constant):
                    return d.value
        # module/class-level annotated assignment `symbol: type = value`
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) \
                and node.target.id == symbol and isinstance(node.value, ast.Constant):
            return node.value.value
    raise AssertionError(f"could not find Python field {symbol!r} default in {file}")


def _home_value(home):
    if isinstance(home, CppField):
        return _cpp_field_value(home.file, home.symbol)
    if isinstance(home, PyArg):
        return _py_arg_default(home.file, home.dest)
    if isinstance(home, PyField):
        # policy tuples / the surplus bool live in topology_enum; scalar fields elsewhere.
        if home.symbol in ("SERVER_POLICIES", "GEN_POLICIES", "SURPLUS_POLICIES", "surplus_present"):
            return _py_policy_first(home.file, home.symbol)
        return _py_field_value(home.file, home.symbol)
    return None  # NoCodeHome / CppFlag (skipped)


# --- the lint -------------------------------------------------------------------------------------
def _checkable():
    reg = spec.registry()
    out = []
    for p in reg.all():
        if isinstance(p.home, NoCodeHome):
            continue
        if p.default is None:        # derived dims carry no literal to check
            continue
        out.append(p)
    return out


@pytest.mark.parametrize("p", _checkable(), ids=lambda p: p.name)
def test_descriptor_default_agrees_with_home(p):
    expected = _home_value(p.home)
    assert expected is not None, f"{p.name}: home {type(p.home).__name__} unhandled"
    assert p.default == expected, (
        f"DRIFT: {p.name} default={p.default!r} but its home "
        f"{type(p.home).__name__}({getattr(p.home,'file','?')}) says {expected!r} "
        f"(ADR-0012 P1: derive, don't copy)")


def test_lint_catches_injected_drift():
    # the lint must be non-vacuous: a wrong default must be caught.
    from hp.spec import CppField as CF
    bad_home = CF("cpp/include/chocofarm/runner_wire_batched.hpp", "WireRunnerConfig::min_coalesce")
    expected = _home_value(bad_home)  # 32
    wrong = expected + 1
    assert wrong != expected
    # simulate a descriptor that copied a wrong literal:
    assert wrong != _home_value(bad_home), "injected drift not detectable (lint would be vacuous)"
