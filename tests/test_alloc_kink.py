"""
tests/test_alloc_kink.py
========================

Isolated unit tests for `alloc.kink.assess_min_kink` — the Clark-1961 min()-kink closed form (the
responsibility-refactor's move 4; `docs/design/leaf-eval-bound-responsibility-refactor.md`) —
exercised DIRECTLY on synthetic arms + a synthetic Σ (no `NeymanDriver`, no `step()`, no live
measurement). This is the unit-testability the lift was for: the §8 reproduction targets
(E[min] / sd / Φ(−t)) are pinned at the pure-function level, so a regression in the Clark math
surfaces here without the whole driver. (The end-to-end equivalence — that the driver's `step()` is
unchanged by the lift — is the separate oracle in `test_neyman_driver_phase2.py`; the gradient
backend `alloc.gradient.jax_gradient` is covered by `tests/test_jax_f_equivalence.py`.)

Run-free / fast: pure numpy + scipy on hand-built arrays. No timed bench, no DB, no network, no
OpenTURNS.

Public Domain (The Unlicense).
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest

_OT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "tools", "analysis", "leaf_eval_bound",
)
if _OT not in sys.path:
    sys.path.insert(0, _OT)

from alloc import kink as K  # noqa: E402


# --------------------------------------------------------------------------- #
# The operating-point arms (the §8 production anchor, mirrored from
# test_neyman_driver_phase2._kink_driver(sigma_R=8.0)): a min(producer, serve) model with
#   producer = N_gen·R_gen,  N_gen=3±0.05,  R_gen=152±8   -> cap 456, σ₁ propagates to 25.17
#   serve    = serve_cap,    serve_cap=428.28±2           -> cap 428.28 (the binding arm)
# The arms are the POST-`_model_arms` shape the driver hands to assess_min_kink:
#   [(capacity, ∇capacity_over_inputs), …]  with inputs ordered [N_gen, R_gen, serve_cap].
# Σ is the diagonal input covariance the Fixed/NORMAL estimates assemble (each input's σ²).
# --------------------------------------------------------------------------- #
def _operating_arms() -> list:
    N, R, S = 3.0, 152.0, 428.28
    producer = (N * R, np.array([R, N, 0.0]))   # ∂(N·R)/∂N=R, /∂R=N, /∂S=0
    serve = (S, np.array([0.0, 0.0, 1.0]))       # ∂S/∂S = 1
    return [producer, serve]


def _sigma(sigma_R: float) -> np.ndarray:
    return np.diag([0.05 ** 2, sigma_R ** 2, 2.0 ** 2]).astype(float)


def test_assess_min_kink_reproduces_operating_point() -> None:
    """The §8 OPERATING anchor at σ_R=8 (propagated producer σ₁=25.17): Clark gives E[min]=426.5,
    sd=6.2, Φ(t)=P(producer is min)=0.136 — pinned at the pure-function level (the same numbers
    test_clark_kink_reproduces_operating_sigma1_25 asserts through the full driver)."""
    res = K.assess_min_kink(_operating_arms(), _sigma(8.0))
    assert res is not None
    assert res["E_min"] == pytest.approx(426.5, abs=0.1)
    assert math.sqrt(res["var_min"]) == pytest.approx(6.2, abs=0.05)
    assert res["p_nonbinding_max"] == pytest.approx(0.136, abs=0.002)
    # binding = serve (cap 428.28, index 1 in the arms list); contender = producer (index 0).
    assert (res["binding"], res["contender"]) == (1, 0)
    # the Φ(±t)-weighted both-arm gradient has the input dimension (3) and sums the criticality weights to 1.
    assert np.asarray(res["grad_weighted"]).shape == (3,)


def test_assess_min_kink_reproduces_stress_point() -> None:
    """The §8 STRESS figure at producer σ₁=60: E[min]=415.68, sd=25.58, Φ(t)=0.322. σ_R is chosen so
    √((152·0.05)²+(3·σ_R)²)=60 — sourced from the covariance exactly as the driver test derives it."""
    sigma_R = math.sqrt(60.0 ** 2 - (152 * 0.05) ** 2) / 3.0  # -> propagated σ₁ = 60
    res = K.assess_min_kink(_operating_arms(), _sigma(sigma_R))
    assert res is not None
    assert res["E_min"] == pytest.approx(415.68, abs=0.05)
    assert math.sqrt(res["var_min"]) == pytest.approx(25.58, abs=0.05)
    assert res["p_nonbinding_max"] == pytest.approx(0.322, abs=0.002)


def test_assess_min_kink_none_for_absent_or_single_arm() -> None:
    """The honest default (a non-visible-min model): None arms, or fewer than two arms, is the smooth
    regime — assess_min_kink returns None, never a fabricated tie."""
    assert K.assess_min_kink(None, _sigma(8.0)) is None
    assert K.assess_min_kink([], _sigma(8.0)) is None
    assert K.assess_min_kink([(456.0, np.array([152.0, 3.0, 0.0]))], _sigma(8.0)) is None


def test_assess_min_kink_smooth_far_from_a_tie() -> None:
    """A comfortably-bound contender (serve far BELOW producer) → Φ(t)→0 < the floor → None (smooth):
    the analytic single-arm gradient is honest, exactly today's behavior away from a tie."""
    N, R = 3.0, 152.0
    arms = [(N * R, np.array([R, N, 0.0])), (200.0, np.array([0.0, 0.0, 1.0]))]  # serve 200 ≪ producer 456
    assert K.assess_min_kink(arms, _sigma(8.0)) is None


def test_assess_min_kink_none_when_difference_spread_is_zero() -> None:
    """The measure-zero guard: with no input spread (Σ=0) the SD of (binding − contender) is ~0 — no
    resolvable tie scale — so assess_min_kink returns None rather than dividing by ~0 (ADR-0002)."""
    assert K.assess_min_kink(_operating_arms(), np.zeros((3, 3))) is None


def test_assess_min_kink_pfloor_gates_the_regime() -> None:
    """The `pfloor` parameter is the binding-margin trigger: at the operating point P(flip)=0.136, so a
    floor below it ENTERS the kink regime and a floor above it stays smooth (None). This is `KINK_PFLOOR`
    made an explicit, testable knob by the lift (it was a module-private constant inside the driver)."""
    arms, Sigma = _operating_arms(), _sigma(8.0)
    assert K.assess_min_kink(arms, Sigma, pfloor=0.10) is not None   # 0.136 > 0.10 -> fires
    assert K.assess_min_kink(arms, Sigma, pfloor=0.20) is None       # 0.136 < 0.20 -> smooth
    assert K.KINK_PFLOOR == 1e-3                                     # the default is the driver's old floor
