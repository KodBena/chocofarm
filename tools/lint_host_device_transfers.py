#!/usr/bin/env python3
"""
tools/lint_host_device_transfers.py — a MECHANIZED LINT (pure `ast`) forbidding GRATUITOUS
host<->device (numpy<->jax) transfers, enforcing they be ISOLATED at explicit boundaries so they can be
CONSOLIDATED.

WHY (the empirical motivation, ADR-0009 honesty). The micro-lib bench (commit feca4f2; numbers under
~/w/vdc/chocobo/bench/lowlatency/) showed a ~57us per-call cost was nothing but a repeated params
host->device transfer that `jax.jit(params, jnp.asarray(x))` redoes every call (the robust AOT handle
stages params device-resident ONCE and drops it). The inference server's `run_microbatch` pays an even
bigger ~85-135us on the input host->device hand-off plus a blocking device->host pull. Scattered
transfers are the cost; isolating + consolidating them is the lever. This lint is the ADR-0011 Rule-1
mechanism that turns "keep transfers at the boundary" from review-only prose into a CI-enforced ratchet
(ADR-0012 P7 lifted from the cross-LANGUAGE wire to the cross-DEVICE boundary; a cross-boundary crossing
has ONE auditable home, not N scattered ones — P1/P2/P7).

THE RULE. A host<->device transfer CALL-SITE is ALLOWED only at a DESIGNATED BOUNDARY; anywhere else is a
violation. Two boundary mechanisms (either one allows a site):
  * an inline `# host-device-boundary: <reason>` marker on the transfer's own line (the explicit,
    per-site opt-in — the reason makes the crossing auditable), or
  * membership in the small BOUNDARY_MODULES whitelist (a module whose declared job IS the device
    boundary — the JAX backends and their bench, where a transfer is the point, not a leak).
Everything else must be in the grandfathered BASELINE (the ratchet, below) or it FAILS.

WHAT IS A TRANSFER (the two directions; the device->host set is HEURISTIC — see the next block):
  host->device (UNAMBIGUOUS — the symbol name IS the device crossing, jax-only, no false positive):
    jnp.array(...), jnp.asarray(...), jax.numpy.array/asarray(...), jax.device_put(...) /
    device_put(...).
  device->host (the BLOCKING pulls):
    UNAMBIGUOUS: <expr>.block_until_ready()  (a jax-Array-only method — flagged unconditionally).
    HEURISTIC:   np.asarray(<expr>) / np.array(<expr>) / float|int|bool(<expr>) /
                 <expr>.tolist() / <expr>.item()  — flagged ONLY when the argument (or the method
                 receiver) carries a static JAX/device SIGNAL (see `_has_device_signal`).

WHY THE DEVICE->HOST SET IS HEURISTIC (ADR-0008 vocabulary precision + ADR-0011 Rule 3 measure-first).
`np.asarray(x)`, `float(x)`, `int(x)` are NAME-AMBIGUOUS: the very same call constructs a numpy array
from a Python list (`np.asarray([1,2,3])`) or casts a Python scalar (`int(m)`) — host-only, NOT a device
pull. A measure-first sizing found ~514 bare `float|int|bool(...)` and ~43 `np.asarray/np.array(...)`
sites in the tree, the overwhelming majority host-only (`float(np.sum(...))`, `int(reply[k])`,
`np.array([8,9,11,12])`). Flagging them all would baseline a huge untriaged noise set and red the gate on
every innocent `int(x)` — exactly the cargo-cult net ADR-0011's "Negative" warns is worse than none. So
this lint draws the line where the STATIC signal actually is: it flags an ambiguous device->host call only
when its argument is a jax-named/forward-named call or a device-resident-named value (the canonical
offender `np.asarray(forward_fn(...))` IS caught — the `forward_fn(...)` call is the signal). Sites the
heuristic cannot statically resolve (a bare `np.asarray(v)` whose `v` is a far-away jax binding) are a
DOCUMENTED BLIND SPOT, opt-in only via the marker; this lint does not sweep them (the rule grandfathers
today's; consolidation is a separate follow-on). `block_until_ready()` needs no heuristic — it is jax-only.

THE INLINE OPT-OUT (false positives). A heuristic flag that is in fact host-only is silenced with an
inline `# host-device-allow: <reason>` marker on the same line (distinct from the BOUNDARY marker: ALLOW
says "this is not really a device transfer," BOUNDARY says "this IS one, sanctioned here"). Both keep the
diff honest — the reason is visible at the site.

THE RATCHETING BASELINE (ADR-0011 Rule 1, mirroring tests/test_mypy_strict.py's STRICT_CLEAN ratchet).
`host_device_baseline.json` (beside this file) records TODAY's transfer sites, grandfathered. A NEW
transfer not in the baseline and not at a boundary FAILS; removing a baselined transfer SHRINKS the
baseline (the gate fails on a STALE baseline entry too, so the file can only monotonically decrease —
you cannot leave a removed transfer recorded). Regenerate after an intentional change with
`--update-baseline` (then commit the smaller file). The baseline keys a site by (relpath, qualified
enclosing scope, transfer kind) — a STRUCTURAL key (ADR-0011 Rule 4: over the class of crossings, not a
line number that churns on every edit above it).

USAGE:
    python -m tools.lint_host_device_transfers            # check; nonzero exit on a new/stale violation
    python -m tools.lint_host_device_transfers --update-baseline
    python -m tools.lint_host_device_transfers --list     # print every detected transfer site

This is a PURE-`ast` walker: it imports NEITHER jax NOR any analyzed module (the host is reserved for
timing-sensitive benchmarks). It only reads files and walks their syntax trees.

Public Domain (The Unlicense).
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import sys
from dataclasses import dataclass

# ---------------------------------------------------------------------------------------------------
# Scope: the tracked `chocofarm/` package. The stray untracked `chocofarm/chocofarm/` copy and the
# untracked `chocofarm/attic/` are excluded (they are not the package, and baselining untracked files
# would be meaningless). __pycache__ and the usual build/vcs dirs are skipped.
# ---------------------------------------------------------------------------------------------------
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PACKAGE_ROOT = os.path.join(REPO, "chocofarm")
BASELINE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "host_device_baseline.json")

EXCLUDE_DIR_NAMES = {"__pycache__", ".git", ".mypy_cache", ".pytest_cache", "attic"}
# the nested stray copy, excluded by relpath prefix (it duplicates the real package one level down).
EXCLUDE_RELPATH_PREFIXES = ("chocofarm/chocofarm/",)

# ---------------------------------------------------------------------------------------------------
# BOUNDARY whitelist: modules whose DECLARED job is the host<->device boundary, so a transfer there is
# the point, not a leak. Kept SMALL and named (P2: the boundary is explicit). Relative to REPO, '/'-sep.
# - az/lowlatency.py        : the SSOT low-overhead JAX dispatcher — staging params/x device-resident IS
#                             its contract (the `device_put` boundary the bench measured).
# - az/mlp_jax.py           : the jax forward backend — its host<->device edges ARE the leaf-eval boundary.
# - az/mlp_jax_train.py     : the jax/optax trainer — host->device staging of the training batch + hps.
# - az/optimizer.py         : the optax hyperparam-injection seam (live lr/b1/b2/eps onto device state).
# - az/forward.py           : the SSOT `forward_core` (the one jitted forward every path crosses into).
# - az/bench/bench_lowlatency.py : the dispatch-cost micro-bench — transfers ARE the measured subject.
# - az/bench/bench_mlp_lowlatency.py : the real-MLP dispatch decomposition bench — same, transfers ARE the subject.
# ---------------------------------------------------------------------------------------------------
BOUNDARY_MODULES: frozenset[str] = frozenset({
    "chocofarm/az/lowlatency.py",
    "chocofarm/az/mlp_jax.py",
    "chocofarm/az/mlp_jax_train.py",
    "chocofarm/az/optimizer.py",
    "chocofarm/az/forward.py",
    "chocofarm/az/bench/bench_lowlatency.py",
    "chocofarm/az/bench/bench_mlp_lowlatency.py",
})

# inline markers (checked against the source LINE the transfer call sits on).
BOUNDARY_MARKER = "host-device-boundary:"   # "this IS a transfer, sanctioned here, because <reason>"
ALLOW_MARKER = "host-device-allow:"         # "the heuristic misfired; this is host-only, because <reason>"

# ---------------------------------------------------------------------------------------------------
# Transfer-kind vocabulary (a closed set, ADR-0008). UNAMBIGUOUS kinds have no false positive; HEURISTIC
# kinds are flagged only with a device signal.
# ---------------------------------------------------------------------------------------------------
H2D_FUNCS = {"array", "asarray"}            # as jnp.<f> / jax.numpy.<f>
H2D_DEVICE_PUT = "device_put"               # jax.device_put / device_put
D2H_NP_FUNCS = {"asarray", "array"}         # as np.<f> / numpy.<f>  (HEURISTIC)
D2H_SCALAR_BUILTINS = {"float", "int", "bool"}   # builtins on an expr      (HEURISTIC)
D2H_METHODS_HEURISTIC = {"tolist", "item"}  # <expr>.tolist() / <expr>.item()  (HEURISTIC)
D2H_METHOD_UNAMBIGUOUS = "block_until_ready"     # jax-Array-only            (UNAMBIGUOUS)

# names that, as the immediate qualifier of an `.array`/`.asarray` call, mean JAX (host->device).
_JAX_NP_QUALIFIERS = {"jnp", "jaxnp"}       # `jnp` and the occasional `import jax.numpy as jaxnp`
# names that, as the immediate qualifier, mean NUMPY (the host side of a device->host pull).
_NUMPY_QUALIFIERS = {"np", "numpy"}

# the JAX/device SIGNAL heuristic vocabulary (substrings, lowercased) used to decide whether an
# ambiguous device->host call actually pulls a DEVICE value. Conservative: a hit means "looks device".
# NAME hints are restricted to the codebase's OBSERVED device-RESIDENCE convention (`x_dev`, `params_dev`,
# `x_device`) — NOT speculative substrings like `_jax`, which would misfire on a `use_jax_mlp` BOOL
# SELECTOR flag (a Python bool the C++ actor never consumes — `cpp_executor.py`), the exact ADR-0011
# Rule-3 over-broad-net failure. A name with no device-residence suffix is left to the CALL hints below.
_DEVICE_NAME_HINTS = ("_dev", "_device", "dev_", "device_")
# CALL hints: the forward/predict/jit/device_put/block_until_ready ops whose OUTPUT is a device value —
# so `np.asarray(forward_fn(...))` / `float(net.predict_value(...))` are flagged (the device pull is real).
_DEVICE_CALL_HINTS = ("forward", "predict", "jit", "device_put", "block_until_ready", "lower")


@dataclass(frozen=True)
class Transfer:
    """One detected host<->device transfer call-site. The (relpath, scope, kind) triple is the STRUCTURAL
    baseline key (ADR-0011 Rule 4 — stable across edits above the site, unlike a line number); `lineno`
    and `snippet` are diagnostics only (not part of the key)."""
    relpath: str         # '/'-separated, relative to REPO (e.g. "chocofarm/az/inference_server.py")
    scope: str           # qualified enclosing def/class chain, "<module>" at module level
    kind: str            # the transfer-kind tag (e.g. "jnp.asarray", "np.asarray", "block_until_ready")
    direction: str       # "h2d" | "d2h"
    heuristic: bool      # True if flagged by the device-signal heuristic (vs an unambiguous jax name)
    lineno: int          # 1-indexed source line (diagnostic only)
    snippet: str         # the stripped source line (diagnostic only)

    def key(self) -> str:
        return f"{self.relpath}::{self.scope}::{self.kind}"


# ---------------------------------------------------------------------------------------------------
# AST helpers — purely syntactic name resolution. No imports, no evaluation.
# ---------------------------------------------------------------------------------------------------
def _attr_qualifier(node: ast.Attribute) -> str | None:
    """For `<X>.<attr>`, return the immediate qualifier `<X>` as a dotted string if it is a Name or a
    Name.attr chain (so `jnp`, `jax.numpy`, `np`), else None."""
    val = node.value
    if isinstance(val, ast.Name):
        return val.id
    if isinstance(val, ast.Attribute):
        inner = _attr_qualifier(val)
        return f"{inner}.{val.attr}" if inner is not None else None
    return None


def _name_tokens(node: ast.AST) -> list[str]:
    """Collect lowercased identifier tokens that appear in `node` (Name ids, Attribute attrs, and the
    func name of a Call). Used to sniff a JAX/device signal in an ambiguous device->host argument."""
    toks: list[str] = []
    for sub in ast.walk(node):
        if isinstance(sub, ast.Name):
            toks.append(sub.id.lower())
        elif isinstance(sub, ast.Attribute):
            toks.append(sub.attr.lower())
    return toks


def _call_func_names(node: ast.AST) -> list[str]:
    """The lowercased function names of every Call inside `node` (e.g. for `np.asarray(forward_fn(x))`
    the inner arg yields `forward_fn`)."""
    out: list[str] = []
    for sub in ast.walk(node):
        if isinstance(sub, ast.Call):
            f = sub.func
            if isinstance(f, ast.Name):
                out.append(f.id.lower())
            elif isinstance(f, ast.Attribute):
                out.append(f.attr.lower())
    return out


def _has_device_signal(arg: ast.AST) -> bool:
    """Heuristic: does this argument expression statically look like it holds a DEVICE (jax) value?
    True if any contained Call's function name hints a jax/forward op (forward/predict/jit/device_put/…)
    OR any identifier token carries a device-name hint (_dev/_device/jax_/…). Conservative by design —
    a miss is a documented blind spot (opt in with the marker), a hit is a flagged crossing."""
    for fn in _call_func_names(arg):
        if any(h in fn for h in _DEVICE_CALL_HINTS):
            return True
        if fn in _JAX_NP_QUALIFIERS:   # jnp(...) is not a thing, but be safe
            return True
    for tok in _name_tokens(arg):
        if any(h in tok for h in _DEVICE_NAME_HINTS):
            return True
    # a directly-qualified jax expression as the argument (e.g. np.asarray(jnp.foo(...))) is device-side.
    for sub in ast.walk(arg):
        if isinstance(sub, ast.Attribute):
            q = _attr_qualifier(sub)
            if q in _JAX_NP_QUALIFIERS or (q is not None and q.split(".")[0] == "jax"):
                return True
    return False


def _classify_call(node: ast.Call) -> tuple[str, str, bool] | None:
    """Classify a Call node as a transfer, returning (kind, direction, heuristic) or None.

    UNAMBIGUOUS host->device: jnp.array/asarray, jax.numpy.array/asarray, jax.device_put / device_put.
    UNAMBIGUOUS device->host: <expr>.block_until_ready().
    HEURISTIC device->host:   np.asarray/array(<expr>), float/int/bool(<expr>), <expr>.tolist()/.item()
                              — only when `_has_device_signal(arg / receiver)` holds."""
    func = node.func

    # ---- attribute-call forms: `<qual>.<name>(...)` and `<expr>.<method>()` ----
    if isinstance(func, ast.Attribute):
        attr = func.attr
        qual = _attr_qualifier(func)

        # host->device: jnp.array/asarray, jax.numpy.array/asarray
        if attr in H2D_FUNCS and (qual in _JAX_NP_QUALIFIERS or qual == "jax.numpy"):
            return (f"jnp.{attr}", "h2d", False)
        # host->device: jax.device_put / <module>.device_put where module is jax
        if attr == H2D_DEVICE_PUT and qual is not None and qual.split(".")[0] in ("jax", "jnp"):
            return ("jax.device_put", "h2d", False)

        # device->host (UNAMBIGUOUS): <expr>.block_until_ready()
        if attr == D2H_METHOD_UNAMBIGUOUS:
            return ("block_until_ready", "d2h", False)

        # device->host (HEURISTIC): np.asarray/array(<expr>) — flag iff the arg looks device-side
        if attr in D2H_NP_FUNCS and qual in _NUMPY_QUALIFIERS:
            if node.args and _has_device_signal(node.args[0]):
                return (f"np.{attr}", "d2h", True)
            return None
        # device->host (HEURISTIC): <expr>.tolist() / <expr>.item() — flag iff the RECEIVER looks device
        if attr in D2H_METHODS_HEURISTIC:
            if _has_device_signal(func.value):
                return (attr, "d2h", True)
            return None
        return None

    # ---- bare-name-call forms: `device_put(...)`, `float(...)`, `int(...)`, `bool(...)` ----
    if isinstance(func, ast.Name):
        name = func.id
        # host->device: a bare `device_put(...)` (from `from jax import device_put`)
        if name == H2D_DEVICE_PUT:
            return ("jax.device_put", "h2d", False)
        # device->host (HEURISTIC): float/int/bool(<expr>) — flag iff the arg looks device-side
        if name in D2H_SCALAR_BUILTINS:
            if node.args and _has_device_signal(node.args[0]):
                return (name, "d2h", True)
            return None
    return None


class _TransferVisitor(ast.NodeVisitor):
    """Walks a module's AST, recording every transfer Call with its qualified enclosing scope. Tracks the
    def/class scope stack so a baseline key is structural (scope-qualified), not line-based."""

    def __init__(self, relpath: str, source_lines: list[str]) -> None:
        self.relpath = relpath
        self.lines = source_lines
        self._scope: list[str] = []
        self.found: list[Transfer] = []

    def _scope_name(self) -> str:
        return ".".join(self._scope) if self._scope else "<module>"

    def _enter(self, name: str, node: ast.AST) -> None:
        self._scope.append(name)
        self.generic_visit(node)
        self._scope.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._enter(node.name, node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._enter(node.name, node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._enter(node.name, node)

    def visit_Call(self, node: ast.Call) -> None:
        cls = _classify_call(node)
        if cls is not None:
            kind, direction, heuristic = cls
            lineno = getattr(node, "lineno", 0)
            line = self.lines[lineno - 1] if 1 <= lineno <= len(self.lines) else ""
            # the inline ALLOW marker silences a heuristic false positive at the source line.
            if not (heuristic and ALLOW_MARKER in line):
                self.found.append(Transfer(
                    relpath=self.relpath, scope=self._scope_name(), kind=kind, direction=direction,
                    heuristic=heuristic, lineno=lineno, snippet=line.strip()))
        # recurse (a transfer can nest inside another call's args, e.g. np.asarray(jnp.asarray(x))).
        self.generic_visit(node)


# ---------------------------------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------------------------------
def _iter_py_files() -> list[str]:
    """Every `.py` under the tracked package, '/'-relpath, excluding the stray nested copy / attic /
    caches. Sorted for deterministic baseline order."""
    out: list[str] = []
    for dp, dirnames, filenames in os.walk(PACKAGE_ROOT):
        dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIR_NAMES]
        for f in filenames:
            if not f.endswith(".py"):
                continue
            rel = os.path.relpath(os.path.join(dp, f), REPO).replace(os.sep, "/")
            if any(rel.startswith(p) for p in EXCLUDE_RELPATH_PREFIXES):
                continue
            out.append(rel)
    return sorted(out)


def _line_has_boundary_marker(snippet_line: str) -> bool:
    return BOUNDARY_MARKER in snippet_line


def scan() -> list[Transfer]:
    """Walk every in-scope file's AST and collect all transfer sites (markers/whitelist NOT yet applied —
    this is the raw detection set the baseline and the gate both filter)."""
    transfers: list[Transfer] = []
    for rel in _iter_py_files():
        path = os.path.join(REPO, rel)
        with open(path, encoding="utf-8") as fh:
            src = fh.read()
        try:
            tree = ast.parse(src, filename=rel)
        except SyntaxError as e:   # a non-parseable file is itself a loud failure (ADR-0002)
            raise SystemExit(f"[host-device-lint] {rel} failed to parse: {e}") from e
        visitor = _TransferVisitor(rel, src.splitlines())
        visitor.visit(tree)
        transfers.extend(visitor.found)
    return transfers


def at_boundary(t: Transfer) -> bool:
    """Is this transfer at a DESIGNATED boundary (so allowed regardless of the baseline)? Either its
    module is a declared boundary module, or its own source line carries the inline boundary marker."""
    if t.relpath in BOUNDARY_MODULES:
        return True
    return _line_has_boundary_marker(t.snippet)


def load_baseline() -> set[str]:
    if not os.path.exists(BASELINE_PATH):
        return set()
    with open(BASELINE_PATH, encoding="utf-8") as fh:
        data = json.load(fh)
    return set(data.get("sites", []))


def build_baseline_payload(transfers: list[Transfer]) -> dict[str, object]:
    """The baseline records only NON-boundary sites (boundary sites are always allowed, so they need no
    grandfathering). Keyed structurally; sorted for a stable, reviewable diff."""
    sites = sorted({t.key() for t in transfers if not at_boundary(t)})
    return {
        "_comment": ("Grandfathered host<->device transfer sites (ADR-0012 / tools/"
                     "lint_host_device_transfers.py). Monotonically DECREASING: removing a transfer "
                     "shrinks this; a NEW one not here and not at a boundary FAILS the gate. Regenerate "
                     "with `python -m tools.lint_host_device_transfers --update-baseline`. Keys are "
                     "(relpath::scope::kind) — structural, not line numbers."),
        "count": len(sites),
        "sites": sites,
    }


def write_baseline(transfers: list[Transfer]) -> int:
    payload = build_baseline_payload(transfers)
    with open(BASELINE_PATH, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
        fh.write("\n")
    count = payload["count"]
    assert isinstance(count, int)
    print(f"[host-device-lint] wrote baseline: {count} grandfathered non-boundary transfer site(s) "
          f"-> {os.path.relpath(BASELINE_PATH, REPO)}")
    return count


def check(transfers: list[Transfer], baseline: set[str]) -> tuple[list[Transfer], list[str]]:
    """Return (new_violations, stale_baseline_keys).

    new_violations: a detected non-boundary transfer whose key is NOT in the baseline -> a gratuitous
                    new crossing the gate must red on.
    stale_baseline_keys: a baseline key no live non-boundary transfer matches -> the baseline failed to
                    shrink (the transfer was removed or moved to a boundary); the entry must be deleted so
                    the ratchet stays monotonic (ADR-0011 Rule 1 — a baseline can only decrease).
    """
    live_nonboundary_keys = {t.key() for t in transfers if not at_boundary(t)}
    new_violations = [t for t in transfers if not at_boundary(t) and t.key() not in baseline]
    stale = sorted(baseline - live_nonboundary_keys)
    return new_violations, stale


def _format_violation(t: Transfer) -> str:
    tag = "HEURISTIC " if t.heuristic else ""
    return (f"  {t.relpath}:{t.lineno}  [{tag}{t.direction} {t.kind}] in {t.scope}\n"
            f"      {t.snippet}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Mechanized lint: forbid gratuitous host<->device transfers.")
    ap.add_argument("--update-baseline", action="store_true",
                    help="rewrite the baseline from the current tree (after an intentional change)")
    ap.add_argument("--list", action="store_true",
                    help="print every detected transfer site (boundary + baselined + new) and exit 0")
    args = ap.parse_args(argv)

    transfers = scan()

    if args.list:
        for t in sorted(transfers, key=lambda x: (x.relpath, x.lineno)):
            loc = "BOUNDARY" if at_boundary(t) else "tracked "
            tag = "H" if t.heuristic else " "
            print(f"[{loc}] {tag} {t.relpath}:{t.lineno} {t.direction} {t.kind} ({t.scope})")
        n_boundary = sum(1 for t in transfers if at_boundary(t))
        print(f"\n[host-device-lint] {len(transfers)} transfer call-site(s): "
              f"{n_boundary} at a boundary, {len(transfers) - n_boundary} tracked (baseline + any new).")
        return 0

    if args.update_baseline:
        write_baseline(transfers)
        return 0

    baseline = load_baseline()
    new_violations, stale = check(transfers, baseline)

    if not new_violations and not stale:
        n_tracked = sum(1 for t in transfers if not at_boundary(t))
        print(f"[host-device-lint] OK: {len(transfers)} transfer site(s) "
              f"({len(transfers) - n_tracked} at a boundary, {n_tracked} grandfathered, 0 new).")
        return 0

    if new_violations:
        print("[host-device-lint] FAIL: new gratuitous host<->device transfer(s) — isolate at a boundary "
              "(`# host-device-boundary: <reason>` or a BOUNDARY_MODULES entry), or, if the heuristic "
              "misfired on a host-only call, mark `# host-device-allow: <reason>`:")
        for t in new_violations:
            print(_format_violation(t))
    if stale:
        print("[host-device-lint] FAIL: stale baseline entr(y/ies) — a grandfathered transfer is gone or "
              "moved to a boundary. The ratchet only decreases: delete it with `--update-baseline`:")
        for key in stale:
            print(f"  (removed) {key}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
