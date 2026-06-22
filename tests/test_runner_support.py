"""
tests/test_runner_support.py
============================

Run-free unit tests for `runner_support.delta_method` — the shared numpy first-order
delta-method bound the leaf-eval runners use as their openturns-ABSENT fallback (the
responsibility-refactor's move 5, numpy-bound half;
`docs/design/leaf-eval-bound-responsibility-refactor.md` §2.6/§3). The recipe was hand-copied
across `throughput_bound` (`_numpy_bound`) and `transport_sweep` (two inlined fallbacks),
each re-deriving `a_i=(grad·σ)²`, `var=Σa_i`, `ci=z·√var` and its own `_Z95`; this pins the
single home directly (the end-to-end byte-identical-output check on the runners is the
complementary behavioral oracle).

Fast: the `delta_method` tests are numpy-only by construction — `delta_method` composes the
`alloc.gradient.fd_gradient_dict` seam (which imports openturns lazily) with the σ-weighting, so they
need no openturns. ONE further test (`test_Z95_agrees_with_the_driver_z_quantile`) cross-checks the
`Z95` constant against the driver's z-quantile via a LOCAL `neyman_driver` import (that one does pull in
openturns) — pinning the hack-audit's "latent z re-divergence" finding.

Public Domain (The Unlicense).
"""
from __future__ import annotations

import math
import os
import sys

import pytest

_OT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "tools", "analysis", "OpenTURNS",
)
if _OT not in sys.path:
    sys.path.insert(0, _OT)

import runner_support as rs  # noqa: E402


def _linear(d: dict[str, float]) -> float:
    """A stand-in model f with a CONSTANT gradient (∂/∂a = 3, ∂/∂b = −1.5), so the delta-method
    numbers are exact closed forms independent of the evaluation point."""
    return 3.0 * d["a"] - 1.5 * d["b"]


def test_delta_method_matches_closed_form() -> None:
    """∇(3a − 1.5b) = {a: 3, b: −1.5}; with σ = {a: 2, b: 4}: a_i = (grad·σ)² = {a: 36, b: 36},
    var = Σa_i = 72, ci = Z95·√72. The single home reproduces the recipe the runners inlined."""
    names = ["a", "b"]
    dm = rs.delta_method(_linear, names, {"a": 1.0, "b": 1.0}, {"a": 2.0, "b": 4.0})
    assert dm.grad["a"] == pytest.approx(3.0, rel=1e-4)
    assert dm.grad["b"] == pytest.approx(-1.5, rel=1e-4)
    assert dm.a["a"] == pytest.approx(36.0, rel=1e-4)
    assert dm.a["b"] == pytest.approx(36.0, rel=1e-4)
    assert dm.var == pytest.approx(72.0, rel=1e-4)
    assert dm.ci == pytest.approx(rs.Z95 * math.sqrt(72.0), rel=1e-4)


def test_delta_method_internal_invariants() -> None:
    """The bundle is self-consistent: var == Σ a_i and ci == Z95·√max(var, 0) — the diagonal delta-method
    identities (the OT-absent twin of the driver's gᵀΣg → z·√var). Holds for any inputs/σ."""
    names = ["a", "b"]
    dm = rs.delta_method(_linear, names, {"a": 7.0, "b": -3.0}, {"a": 1.5, "b": 0.25})
    assert dm.var == pytest.approx(sum(dm.a.values()), rel=1e-12)
    assert dm.ci == pytest.approx(rs.Z95 * math.sqrt(max(dm.var, 0.0)), rel=1e-12)
    # the returned object is the typed NamedTuple bundle (grad, a, var, ci) the runners destructure.
    assert isinstance(dm, rs.DeltaMethod)
    assert set(dm.grad) == set(names) == set(dm.a)


def test_delta_method_zero_spread_gives_zero_ci() -> None:
    """A zero-σ input contributes no variance; an all-zero-σ set gives var = ci = 0 (a degenerate but
    honest bound — no spread, no interval), never a NaN from √ of a negative (the max(var, 0) guard)."""
    dm = rs.delta_method(_linear, ["a", "b"], {"a": 1.0, "b": 1.0}, {"a": 0.0, "b": 0.0})
    assert dm.var == 0.0
    assert dm.ci == 0.0


def test_Z95_agrees_with_the_driver_z_quantile() -> None:
    """Consistency tie (the hack-audit's finding 1): the 95% z-multiplier is computed in TWO places —
    `runner_support.Z95` (the runners' fixed constant) and `neyman_driver._z_from_confidence` (the
    general confidence→z quantile, whose openturns-absent fallback hard-codes the same 95% literal).
    They are distinct responsibilities with nothing structurally tying them, so this pins that at 95%
    they evaluate to the same number — a future edit diverging one from the other fails loudly
    (ADR-0002) instead of silently. The `neyman_driver` import is LOCAL (it requires openturns) so the
    rest of this file's delta_method tests stay numpy-only."""
    import neyman_driver as nd
    assert nd._z_from_confidence(0.95) == pytest.approx(rs.Z95, abs=1e-12)
