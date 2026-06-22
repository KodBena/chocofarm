"""
tools/analysis/OpenTURNS/alloc/gradient.py
==========================================

The gradient-backend SEAM — lifted out of `neyman_driver.py` as the ONE home of "the
gradient of the scalar model function `f` at a point" (the responsibility-refactor's §5
gradient Port; `docs/design/leaf-eval-bound-responsibility-refactor.md` §3
`alloc/gradient.py` / §5). Today it is OpenTURNS: the analytic `f.gradient()` with a central
finite-difference fallback. It is deliberately the SINGLE site the planned OpenTURNS→JAX
swap replaces — `gradient()` becomes `jax.grad(f)` in ONE file, and every caller (the
driver's `step()`, and later the runners' numpy-FD copies — refactor move 5) derives from
this one definition rather than re-implementing it (§5 / ADR-0012 P7: one authoritative
definition, every caller derives, no silent divergence between three hand-rolled gradients).

It is parameterized by `f` (an OpenTURNS scalar `Function`), NOT bound to the driver's
state, so the same seam serves the driver and — after move 5 — the runners. Depends only on
numpy + openturns.

NOTE on the third OT/autodiff site (§5): the `TaylorExpansionMoments` curvature DIAGNOSTIC
(`NeymanDriver._second_order_mean`) is deliberately NOT here. The note records it as
drop-not-port for the JAX swap (a smooth-region diagnostic, blind to the kink), and it is a
moments computation over the driver's sample pools — a different concern from the gradient.
It stays in the driver until the swap drops it; this module is the gradient ALONE.

Public Domain (The Unlicense).
"""
from __future__ import annotations

from typing import Optional

import numpy as np

try:
    import openturns as ot
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "alloc.gradient requires openturns: pip install openturns"
    ) from exc


def gradient(
    f: "ot.Function", point: np.ndarray, *, dim: Optional[int] = None, fd_rel_step: float = 1e-5
) -> np.ndarray:
    """The gradient of scalar `f` at `point`: the OpenTURNS analytic `f.gradient()` if available, else
    central finite differences on `f` (`fd_gradient`). THIS is the gradient-backend seam (§5) — the JAX
    swap replaces this body with `jax.grad(f)`. `dim` defaults to `f.getInputDimension()` (pass the
    driver's cached `self.d` to avoid the round-trip)."""
    d = int(dim) if dim is not None else int(f.getInputDimension())
    try:
        g = f.gradient(ot.Point(point))
        return np.array([g[i, 0] for i in range(d)], dtype=float)
    except Exception:
        return fd_gradient(f, point, dim=d, fd_rel_step=fd_rel_step)


def fd_gradient(
    f: "ot.Function", point: np.ndarray, *, dim: Optional[int] = None, fd_rel_step: float = 1e-5
) -> np.ndarray:
    """Central finite-difference gradient of scalar `f` at `point` — the analytic fallback (used when
    `f.gradient()` raises, e.g. a PythonFunction with no analytic gradient). Relative step
    `fd_rel_step·max(|x_i|, 1)` per input. `dim` defaults to `f.getInputDimension()`."""
    d = int(dim) if dim is not None else int(f.getInputDimension())
    g = np.empty(d)
    f0_pt = point.copy()
    for i in range(d):
        h = fd_rel_step * max(abs(point[i]), 1.0)
        xp = f0_pt.copy(); xp[i] += h
        xm = f0_pt.copy(); xm[i] -= h
        yp = float(f(ot.Point(xp))[0])
        ym = float(f(ot.Point(xm))[0])
        g[i] = (yp - ym) / (2.0 * h)
    return g
