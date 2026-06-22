"""
tests/test_jax_f_equivalence.py
===============================

The OpenTURNS→JAX migration's equivalence proof (`docs/design/leaf-eval-bound-responsibility-refactor.md`
§5): each leaf-eval model's single JAX-traceable `throughput_jax` — the OT→JAX one-home for `f` — evaluates
IDENTICALLY to `throughput_numpy`, and `alloc.gradient.jax_gradient` (jax.grad) reproduces the gradient,
checked against an INLINE central-difference oracle (NOT the production FD functions this migration removes,
so the test survives their deletion). This is the evidence the driver swap (the next increment) rests on.

The gradient agrees because OT itself fell back to central FD through the model `min()` (the
"WRN - Switch to finite difference"), so jax.grad — analytic, exact through `min()` — gives the same
binding-arm gradient away from a tie (and the symmetric 0.5/0.5 subgradient AT a tie, which the driver never
uses for the bound: a plausible tie routes to the Clark closed form, `alloc.kink`, not the linearization).

x64 is enabled via `alloc.jax_backend` (float32 would drift the bound ~1e-6). Public Domain (The Unlicense).
"""
from __future__ import annotations

import importlib
import os
import sys
from typing import Any

import numpy as np
import pytest

_OT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "tools", "analysis", "OpenTURNS",
)
if _OT not in sys.path:
    sys.path.insert(0, _OT)

from alloc.jax_backend import jnp  # noqa: E402 — importing it ENABLES x64 (before the first trace)
import alloc.gradient as G  # noqa: E402

_MODELS = ["model_capacity", "model_cycletime", "model_zmq_baseline", "model_shm_spin_poll",
           "model_futex_wake", "model_lockfree_mpsc", "model_cpp_inproc_port"]


def _x0(M: Any) -> dict:
    """The model's grounded initial point — manifest models take `trust=True`, the static ones take no arg."""
    return M.initial_point(trust=True) if "trust" in M.initial_point.__code__.co_varnames else M.initial_point()


def _central_fd(fn_dict: Any, names: list, x0: dict, h: float = 1e-6) -> np.ndarray:
    """An INLINE central-difference gradient over `throughput_numpy` (dict→float) — the oracle, computed here
    so the test does NOT depend on the FD functions the migration deletes."""
    g = np.empty(len(names))
    for i, nm in enumerate(names):
        step = h * max(abs(x0[nm]), 1.0)
        xp = dict(x0); xp[nm] += step
        xm = dict(x0); xm[nm] -= step
        g[i] = (fn_dict(xp) - fn_dict(xm)) / (2.0 * step)
    return g


def test_jax_backend_is_x64() -> None:
    """The migration REQUIRES float64 (the tool is float64; float32 drifts the bound ~1e-6). `alloc.jax_backend`
    enables x64 on import — pin it, so a regression (jax defaulting to float32) fails loudly HERE, not as a
    silent ~1e-6 drift in the bound."""
    assert jnp.asarray(1.0).dtype == jnp.float64


@pytest.mark.parametrize("modname", _MODELS)
def test_throughput_jax_evaluates_identically_to_numpy(modname: str) -> None:
    """The single JAX `f` reproduces `throughput_numpy` at the grounded point (the eval the bound rests on),
    in x64 — exactly (the formulae are identical; the cross-check that the muParser/numpy/JAX homes agree)."""
    M = importlib.import_module(modname)
    x0 = _x0(M)
    arr = jnp.array([x0[nm] for nm in M.INPUT_NAMES])
    assert float(M.throughput_jax(arr)) == pytest.approx(M.throughput_numpy(x0), rel=1e-9)


@pytest.mark.parametrize("modname", _MODELS)
def test_jax_gradient_reproduces_the_fd_gradient(modname: str) -> None:
    """jax.grad (via `alloc.gradient.jax_gradient`) reproduces the central-difference gradient at the grounded
    point — the evidence that swapping OT/FD for jax.grad is lossless (OT itself FD'd through the `min()`)."""
    M = importlib.import_module(modname)
    x0 = _x0(M)
    arr = np.array([x0[nm] for nm in M.INPUT_NAMES])
    g_jax = G.jax_gradient(M.throughput_jax, arr)
    g_fd = _central_fd(M.throughput_numpy, M.INPUT_NAMES, x0)
    assert g_jax == pytest.approx(g_fd, abs=1e-5)
