#!/usr/bin/env python3
"""
tests/test_no_gratuitous_transfers.py — the MECHANICAL NET (pure `ast`) against GRATUITOUS host<->device
(numpy<->jax) transfers, asserting they stay ISOLATED at explicit boundaries so they can be CONSOLIDATED
(ADR-0012 P1/P2/P7 lifted from the cross-LANGUAGE wire to the cross-DEVICE boundary; ADR-0011 Rule 1 — the
CI gate that converts "keep transfers at the boundary" from review-only prose into a ratchet).

This drives `tools/lint_host_device_transfers.py` (the checker) over the tracked `chocofarm/` package and
asserts the ratchet holds. It is PURE `ast`: it imports the CHECKER (stdlib only) but NEVER jax and NEVER
an analyzed module — the host is reserved for timing-sensitive benchmarks, and a name-pattern AST walk
needs neither. It runs in the default `pytest tests/ -q`.

Legs (all always-on — no jax, no C++ binary, no redis):

  1. NO NEW VIOLATIONS (the gate). The current tree's non-boundary transfers are exactly the
     grandfathered baseline — a NEW transfer not at a boundary and not baselined reds CI. This is the
     leg that catches a gratuitous transfer a contributor adds (the empirical lever: the feca4f2 bench
     found a ~57us params host->device repeat and run_microbatch's ~85-135us input/output crossing —
     scattered transfers are the cost, so a new scattered one must be surfaced loudly, ADR-0002).

  2. NO STALE BASELINE (the ratchet is monotonic). Every baselined site still resolves to a live
     non-boundary transfer — a baseline can only DECREASE (ADR-0011 Rule 1). A transfer that was removed
     or moved to a boundary but left in the baseline reds, forcing the file to shrink.

  3. THE CANONICAL OFFENDER IS CAUGHT (the detector is not vacuous on the real tree). `run_microbatch`'s
     `np.asarray(forward_fn(...))` device->host pull (chocofarm/az/inference_server.py) is in the
     detected set — proof the heuristic resolves the documented canonical case, not just synthetic ones.

  4. DRIFT-CATCH SELF-CHECK (the negative/mutation proof — the proportionate verification, mirroring
     tests/test_wire_drift.py leg 2). A SYNTHETIC new transfer injected into a parsed module's AST text
     is asserted to FAIL the gate, and the SAME shape carrying a `# host-device-boundary:` marker is
     asserted to PASS — so the net is demonstrated to actually catch a new crossing AND to honor the
     boundary opt-out, not merely pass when nothing is wrong. Operates on in-memory strings; touches no
     real source.

  5. THE HEURISTIC IS DOCUMENTED-CONSERVATIVE (the ADR-0011 Rule-3 measure-first guard). A bare
     `int(x)` / `np.asarray([1,2,3])` with NO device signal is asserted NOT flagged — pinning that the
     gate did not regress into the 514-scalar-cast cargo-cult net (worse than none) the checker's
     docstring rejects; and a `# host-device-allow:` marker is asserted to silence a heuristic hit.

Public Domain (The Unlicense).
"""
import ast
import importlib.util
import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKER_PATH = os.path.join(REPO, "tools", "lint_host_device_transfers.py")


def _load_checker():
    """Import the checker module by path (tools/ is not an installed package). Skips gracefully if the
    file is absent, mirroring how tests/test_cpp_runner.py skips without its binary."""
    if not os.path.exists(CHECKER_PATH):
        pytest.skip("tools/lint_host_device_transfers.py not present")
    spec = importlib.util.spec_from_file_location("lint_host_device_transfers", CHECKER_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    # register in sys.modules BEFORE exec so the module's own @dataclass(frozen=True) annotation
    # resolution (dataclasses._is_type reads sys.modules[cls.__module__].__dict__) can find it.
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


LINT = _load_checker()


# ---------------------------------------------------------------------------------------------------
# A small helper that runs the checker's CLASSIFIER over an in-memory snippet (no file I/O) — the basis
# for the synthetic drift-catch / heuristic legs. Reuses the checker's own _TransferVisitor so the test
# exercises the REAL classification logic, not a reimplementation.
# ---------------------------------------------------------------------------------------------------
def _detect(src: str, relpath: str = "chocofarm/az/_probe.py") -> list:
    tree = ast.parse(src)
    v = LINT._TransferVisitor(relpath, src.splitlines())
    v.visit(tree)
    return v.found


# ===================================================================================================
# LEG 1 — NO NEW VIOLATIONS (the gate over the real tree).
# ===================================================================================================
def test_no_new_gratuitous_transfers():
    """The tracked tree introduces NO host<->device transfer outside a designated boundary that is not in
    the grandfathered baseline. A new gratuitous transfer reds here (ADR-0011 gate)."""
    transfers = LINT.scan()
    baseline = LINT.load_baseline()
    new_violations, _stale = LINT.check(transfers, baseline)
    assert not new_violations, (
        "new gratuitous host<->device transfer(s) outside a boundary and not baselined — isolate at a "
        "boundary (`# host-device-boundary: <reason>` / a BOUNDARY_MODULES entry) or, if the heuristic "
        "misfired, `# host-device-allow: <reason>`:\n"
        + "\n".join(LINT._format_violation(t) for t in new_violations))


# ===================================================================================================
# LEG 2 — NO STALE BASELINE (the ratchet only decreases).
# ===================================================================================================
def test_baseline_has_no_stale_entries():
    """Every baselined key still resolves to a live non-boundary transfer. A stale entry (a removed /
    boundary-moved transfer left recorded) reds — the baseline is monotonically DECREASING (ADR-0011
    Rule 1); regenerate with `--update-baseline` to shrink it."""
    transfers = LINT.scan()
    baseline = LINT.load_baseline()
    _new, stale = LINT.check(transfers, baseline)
    assert not stale, (
        "stale baseline entr(y/ies) — a grandfathered transfer is gone or moved to a boundary; the "
        "ratchet only decreases. Delete with `python -m tools.lint_host_device_transfers "
        f"--update-baseline`:\n  " + "\n  ".join(stale))


def test_baseline_count_matches_recorded():
    """The baseline file's `count` equals the number of recorded sites (an internal-consistency pin so a
    hand-edit that drops a site but not the count, or vice versa, is caught)."""
    baseline = LINT.load_baseline()
    import json
    with open(LINT.BASELINE_PATH, encoding="utf-8") as fh:
        data = json.load(fh)
    assert data["count"] == len(baseline) == len(data["sites"])


# ===================================================================================================
# LEG 3 — THE CANONICAL OFFENDER IS CAUGHT (the detector is not vacuous on the real tree).
# ===================================================================================================
def test_run_microbatch_device_pull_is_detected():
    """`run_microbatch`'s `np.asarray(forward_fn(...))` device->host pull (the documented canonical
    offender, chocofarm/az/inference_server.py) is in the detected set — so the heuristic resolves the
    real case, and the always-on gate is genuinely guarding it (not passing vacuously)."""
    transfers = LINT.scan()
    keys = {t.key() for t in transfers}
    assert "chocofarm/az/inference_server.py::run_microbatch::np.asarray" in keys, (
        "the canonical run_microbatch device->host pull is no longer detected — the heuristic regressed "
        "or the site moved; re-confirm the checker before trusting the gate")


def test_host_to_device_jax_names_are_detected_unambiguously():
    """The unambiguous host->device jax names (jnp.asarray / jax.device_put) are detected wherever they
    appear (the jax backends) — the no-false-positive core the gate rests on. At least the known
    mlp_jax / lowlatency staging sites are present."""
    transfers = LINT.scan()
    h2d = {t.relpath for t in transfers if t.direction == "h2d"}
    assert "chocofarm/az/mlp_jax.py" in h2d
    assert "chocofarm/az/lowlatency.py" in h2d


# ===================================================================================================
# LEG 4 — DRIFT-CATCH SELF-CHECK (the negative/mutation proof; in-memory, touches no real source).
# ===================================================================================================
def test_synthetic_new_transfer_would_fail_the_gate():
    """NEGATIVE proof: a synthetic non-boundary `jnp.asarray(...)` is (a) detected and (b) NOT in the
    baseline — i.e. it WOULD red the gate. If this didn't hold, leg 1's pass would be vacuous."""
    src = "import jax.numpy as jnp\n\ndef f(x):\n    return jnp.asarray(x)\n"
    found = _detect(src, relpath="chocofarm/az/_probe.py")  # _probe is NOT a boundary module
    assert any(t.kind == "jnp.asarray" and t.direction == "h2d" for t in found), found
    baseline = LINT.load_baseline()
    # none of the synthetic probe's keys are grandfathered -> they are new violations the gate catches.
    assert all(t.key() not in baseline for t in found)
    assert all(not LINT.at_boundary(t) for t in found), "the _probe module must not be a boundary"


def test_boundary_marker_makes_a_transfer_pass():
    """The `# host-device-boundary: <reason>` inline marker silences a transfer on its own line (the
    explicit per-site opt-in). The SAME shape WITHOUT the marker is NOT at a boundary — proving the
    marker is load-bearing, not incidentally green."""
    marked = "import jax.numpy as jnp\n\ndef f(x):\n    return jnp.asarray(x)  # host-device-boundary: the one staging point\n"
    unmarked = "import jax.numpy as jnp\n\ndef f(x):\n    return jnp.asarray(x)\n"
    t_marked = _detect(marked, relpath="chocofarm/az/_probe.py")[0]
    t_unmarked = _detect(unmarked, relpath="chocofarm/az/_probe.py")[0]
    assert LINT.at_boundary(t_marked), "the boundary marker did not silence the transfer"
    assert not LINT.at_boundary(t_unmarked), "the unmarked transfer must NOT be at a boundary (vacuity guard)"


def test_boundary_module_whitelist_makes_a_transfer_pass():
    """A transfer in a declared BOUNDARY_MODULES file (the jax backends, whose job IS the device edge) is
    at a boundary without any inline marker — the module-level opt-in. A non-listed module is not."""
    src = "import jax.numpy as jnp\n\ndef f(x):\n    return jnp.asarray(x)\n"
    # mlp_jax.py is a declared boundary module; _probe.py is not.
    t_boundary = _detect(src, relpath="chocofarm/az/mlp_jax.py")[0]
    t_plain = _detect(src, relpath="chocofarm/az/_probe.py")[0]
    assert LINT.at_boundary(t_boundary)
    assert not LINT.at_boundary(t_plain)
    assert "chocofarm/az/mlp_jax.py" in LINT.BOUNDARY_MODULES


# ===================================================================================================
# LEG 5 — THE HEURISTIC IS DOCUMENTED-CONSERVATIVE (the ADR-0011 Rule-3 measure-first guard).
# ===================================================================================================
def test_bare_scalar_casts_are_not_flagged():
    """A bare `int(x)` / `float(s)` / `bool(flag)` with NO device signal is NOT flagged — pinning that the
    gate did not regress into netting the ~514 host-only scalar casts (the cargo-cult net worse than
    none, ADR-0011 'Negative'). The device boundary, not every scalar conversion, is the subject."""
    src = ("def f(m, s, flag):\n"
           "    a = int(m)\n"
           "    b = float(s)\n"
           "    c = bool(flag)\n"
           "    return a, b, c\n")
    found = _detect(src)
    assert found == [], f"bare scalar casts with no device signal must not be flagged, got {found}"


def test_numpy_construction_from_list_is_not_flagged():
    """`np.array([...])` / `np.asarray(list_literal)` constructs a numpy array from host data — NOT a
    device pull — so with no device signal it is NOT flagged. (The name `np.asarray` is ambiguous; the
    checker flags it only when the argument looks device-side.)"""
    src = ("import numpy as np\n"
           "def f(rows):\n"
           "    a = np.array([8, 9, 11, 12])\n"
           "    b = np.asarray(rows, dtype=np.int64)\n"
           "    return a, b\n")
    found = _detect(src)
    assert found == [], f"numpy construction from host data must not be flagged, got {found}"


def test_device_signaled_pull_is_flagged():
    """The other side of the heuristic: `np.asarray(forward_fn(...))` and `float(net.predict_value(...))`
    DO carry a device signal (a forward/predict call) and ARE flagged — the canonical-offender shape,
    confirmed on a synthetic so the heuristic's POSITIVE direction is pinned too."""
    src = ("import numpy as np\n"
           "def f(forward_fn, net, X, feat):\n"
           "    out = np.asarray(forward_fn(X))\n"
           "    v = float(net.predict_value(feat))\n"
           "    return out, v\n")
    found = _detect(src)
    kinds = {(t.kind, t.heuristic) for t in found}
    assert ("np.asarray", True) in kinds, found
    assert ("float", True) in kinds, found


def test_allow_marker_silences_a_heuristic_false_positive():
    """The `# host-device-allow: <reason>` marker silences a heuristic hit the contributor knows is
    host-only (distinct from the boundary marker: ALLOW says 'not really a device transfer'). The same
    shape WITHOUT the marker IS flagged — the marker is load-bearing."""
    silenced = ("import numpy as np\n"
                "def f(forward_fn, X):\n"
                "    return np.asarray(forward_fn(X))  # host-device-allow: forward_fn is a host stub here\n")
    flagged = ("import numpy as np\n"
               "def f(forward_fn, X):\n"
               "    return np.asarray(forward_fn(X))\n")
    assert _detect(silenced) == [], "the allow marker did not silence the heuristic hit"
    assert _detect(flagged), "the unmarked heuristic shape must be flagged (vacuity guard)"


def test_block_until_ready_is_unconditionally_flagged():
    """`<expr>.block_until_ready()` is a jax-Array-only method — flagged with NO heuristic (no false
    positive possible), in EITHER a device-named or plain receiver. The unambiguous device->host method."""
    src = ("def f(y, result):\n"
           "    y.block_until_ready()\n"
           "    return result.block_until_ready()\n")
    found = _detect(src)
    assert len(found) == 2, found
    assert all(t.kind == "block_until_ready" and not t.heuristic and t.direction == "d2h" for t in found)
