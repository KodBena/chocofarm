"""
tools/analysis/leaf_eval_bound/alloc/gradient.py
==========================================

The gradient-backend SEAM — the ONE home of "the gradient of the scalar model function `f` at a
point" (the responsibility refactor §5; `docs/design/leaf-eval-bound-responsibility-refactor.md`).
After the OpenTURNS→JAX migration the gradient is JAX autodiff, and it is now the ONLY gradient:

  * `jax_gradient` — `jax.grad` of a JAX-traceable scalar `f(x_array)` (the driver's form, a model's
    `throughput_jax`). Analytic reverse-mode autodiff. It replaced the OpenTURNS analytic
    `f.gradient()` + central-FD fallback (migration J3) and, with the single-f collapse (J4), the
    numpy `fd_gradient_dict` that backed the runners' delta-method fallback — both retired.

numpy + jax only — this module imports NO OpenTURNS.

Public Domain (The Unlicense).
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np


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
