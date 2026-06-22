"""
tools/analysis/OpenTURNS/alloc/gradient.py
==========================================

The gradient-backend SEAM — the ONE home of "the gradient of the scalar model function `f`
at a point" (the responsibility refactor §5;
`docs/design/leaf-eval-bound-responsibility-refactor.md`). After the OpenTURNS→JAX migration
the gradient is JAX autodiff:

  * `jax_gradient` — `jax.grad` of a JAX-traceable scalar `f(x_array)` (the driver's form, a
    model's `throughput_jax`). Analytic reverse-mode autodiff; THE gradient backend. (It
    replaced the OpenTURNS analytic `f.gradient()` + central-FD fallback that lived here — OT
    itself fell back to FD through the model's `min()`, so jax.grad loses nothing; the arm-tie
    is handled by the Clark closed form `alloc.kink`, never the linearization.)
  * `fd_gradient_dict` — a central finite-difference gradient of the model's `throughput_numpy`
    (a `dict→float` numpy callable). The runners' numpy delta-method path
    (`runner_support.delta_method`) still uses it; it retires with the numpy fallback (migration
    J4), leaving `jax_gradient` as the sole gradient.

numpy + jax only — this module imports NO OpenTURNS (the OT-function gradient forms `gradient`
/ `fd_gradient` and their lazy `_ot()` import were removed in migration J3).

Public Domain (The Unlicense).
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np


def fd_gradient_dict(
    fn: Callable[[dict[str, float]], float], names: list[str], x0: dict[str, float], rel: float = 1e-5,
) -> dict[str, float]:
    """Central finite-difference gradient of a numpy callable `fn(dict) -> float` at `x0`, returned as
    `{input_name: ∂fn/∂input}`. The runners' numpy delta-method gradient over `model.throughput_numpy`
    (`runner_support.delta_method`). Pending retirement with the numpy fallback (migration J4), after
    which `jax_gradient` is the sole gradient. numpy-only."""
    g: dict[str, float] = {}
    for nm in names:
        h = rel * max(abs(x0[nm]), 1.0)
        xp = dict(x0); xp[nm] += h
        xm = dict(x0); xm[nm] -= h
        g[nm] = (fn(xp) - fn(xm)) / (2.0 * h)
    return g


def jax_gradient(f: Callable[..., Any], point: np.ndarray) -> np.ndarray:
    """The gradient of a JAX-traceable scalar `f` at `point` (ordered by the model's INPUT_NAMES), via
    `jax.grad` — analytic reverse-mode autodiff. THE gradient backend (the OT→JAX migration, §5). It
    replaced the OpenTURNS analytic `f.gradient()` + central-FD fallback (OT itself fell back to FD
    through the model's `min()`, so it was never truly analytic). `jax.grad` differentiates through
    `min()` exactly: away from an arm-tie it returns the binding arm's gradient (== FD, validated ~1e-9);
    at a tie the symmetric 0.5/0.5 subgradient (== central FD). The arm-TIE bound is NOT a linearization
    — it is the Clark-1961 closed form (`alloc.kink`) the driver routes to near a tie, so the subgradient
    is never the bound there. No FD step-size truncation. x64 is enforced via `jax_backend` (float32 would
    drift the bound ~1e-6)."""
    from alloc.jax_backend import grad, jnp
    g = grad(f)(jnp.asarray(point, dtype=float))
    return np.asarray(g, dtype=float)
