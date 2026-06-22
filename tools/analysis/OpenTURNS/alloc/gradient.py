"""
tools/analysis/OpenTURNS/alloc/gradient.py
==========================================

The gradient-backend SEAM — the ONE home of "the gradient of the scalar model function
`f` at a point", lifted out of `neyman_driver.py` (refactor increment 1) and the two
runner copies (increment 2) of the responsibility refactor
(`docs/design/leaf-eval-bound-responsibility-refactor.md` §5). Today it is OpenTURNS /
numpy finite differences; it is deliberately the SINGLE site the planned OpenTURNS→JAX
swap replaces — `jax.grad(f)` lands here once and every caller derives (ADR-0012 P7),
instead of the three hand-rolled gradients (driver + throughput_bound + transport_sweep)
that drifted apart before this.

It holds the gradient in BOTH forms the tool uses, because the model's `f` exists today in
two representations (the §2.4/§5 dual-home the JAX swap dissolves):

  * `gradient` / `fd_gradient` — over an OpenTURNS scalar `Function` (the driver's `step()`
    form): analytic `f.gradient()` with a central-FD fallback; point as a numpy array.
  * `fd_gradient_dict` — over the model's `throughput_numpy` (a `dict→float` numpy
    callable): point + result as `{input_name: value}` dicts. This is the runners' OT-ABSENT
    fallback gradient — the verbatim copy that lived in BOTH `throughput_bound` and
    `transport_sweep` (refactor move 5). numpy-only.

The two forms differentiate the SAME `f` by two routes; the JAX migration collapses them to
one (`jax.grad` of a single traceable `f`), which is why they share this home now.

OpenTURNS is imported LAZILY (only inside the OT forms, via `_ot()`): this module must
import — and `fd_gradient_dict` must RUN — on a host WITHOUT openturns, because that is
exactly the runners' numpy-only fallback path (`_HAS_OT=False`). The OT forms raise loudly
(ADR-0002) if openturns is genuinely absent when one of them is CALLED.

NOTE on the third OT/autodiff site (§5): the `TaylorExpansionMoments` curvature DIAGNOSTIC
(`NeymanDriver._second_order_mean`) is NOT here — the note records it as drop-not-port for
the JAX swap, and it is a moments computation over the driver's pools, a different concern.

Public Domain (The Unlicense).
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Optional

import numpy as np

if TYPE_CHECKING:  # for the annotations only; never imported at runtime (the numpy path must run sans OT)
    import openturns as ot


def _ot() -> Any:
    """Lazily import openturns for the OT gradient forms. ADR-0002: a loud, HELPFUL error if absent — but
    only when an OT form is actually CALLED, so this module (and `fd_gradient_dict`) import without OT."""
    try:
        import openturns as ot
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "alloc.gradient.{gradient,fd_gradient} require openturns: pip install openturns "
            "(the numpy-only fd_gradient_dict does NOT — it is the openturns-absent fallback)."
        ) from exc
    return ot


def gradient(
    f: "ot.Function", point: np.ndarray, *, dim: Optional[int] = None, fd_rel_step: float = 1e-5
) -> np.ndarray:
    """The gradient of scalar `f` at `point`: the OpenTURNS analytic `f.gradient()` if available, else
    central finite differences on `f` (`fd_gradient`). THIS is the gradient-backend seam (§5) — the JAX
    swap replaces this body with `jax.grad(f)`. `dim` defaults to `f.getInputDimension()` (pass the
    driver's cached `self.d` to avoid the round-trip)."""
    ot = _ot()
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
    ot = _ot()
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


def fd_gradient_dict(
    fn: Callable[[dict[str, float]], float], names: list[str], x0: dict[str, float], rel: float = 1e-5,
) -> dict[str, float]:
    """Central finite-difference gradient of a numpy callable `fn(dict) -> float` at `x0`, returned as
    `{input_name: ∂fn/∂input}`. The runners' OT-ABSENT fallback gradient over `model.throughput_numpy`
    (it was duplicated VERBATIM in throughput_bound + transport_sweep — refactor move 5). numpy-only by
    construction: it must run when openturns is absent, which is the entire point of the fallback."""
    g: dict[str, float] = {}
    for nm in names:
        h = rel * max(abs(x0[nm]), 1.0)
        xp = dict(x0); xp[nm] += h
        xm = dict(x0); xm[nm] -= h
        g[nm] = (fn(xp) - fn(xm)) / (2.0 * h)
    return g
