"""
tools/analysis/OpenTURNS/runner_support.py
==========================================

Shared RUNNER support — the numpy first-order delta-method bound the leaf-eval runners
(`throughput_bound`, `transport_sweep`) use as their openturns-ABSENT fallback (and, in
`throughput_bound`, as the always-on lockstep cross-check of the OT path). The
responsibility-refactor's move 5, numpy-bound half
(`docs/design/leaf-eval-bound-responsibility-refactor.md` §2.6/§3): the delta-method
recipe was hand-copied across the two runners — a full `_numpy_bound` in `throughput_bound`,
two inlined fallback recipes in `transport_sweep` — each re-deriving `a_i=(df/dx·σ)²`,
`var=Σa_i`, `ci=z·√var` and its own `_Z95`. This single-homes that math (ADR-0012 P1).

It is the numpy DIAGONAL twin of the driver's general `gᵀΣg` (`neyman_driver.step`): the
openturns-absent path computes the same bound for an independent (diagonal-Σ) input set —
which is exactly the grounded leaf-eval suite. The gradient itself is NOT redefined here;
it is the shared `alloc.gradient.fd_gradient_dict` seam (move 5, gradient half), which this
module composes with the σ-weighting into the delta-method bundle the runners consume. So
the layering is runners → `runner_support` → `alloc.gradient` (a clean DAG, no cycle).

This is the top-level, flat-layout form of the note's `runners/support.py`; when the
runners are relocated into a `runners/` package (a later refactor increment) this file
moves there. It does NOT yet hold the model-dialect shims (`_model_sigmas` /
`_registry_qname` / `_untrusted` / `_model_estimates`) — those are move 3's
model-interface concern, deliberately left in the runners for now.

Public Domain (The Unlicense).
"""
from __future__ import annotations

import math
import os
import sys
from typing import Callable, NamedTuple

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
from alloc import gradient as _grad  # noqa: E402 — the shared gradient-backend seam (numpy-dict form);
# it imports openturns LAZILY, so this module (and the numpy delta-method) load on an openturns-absent host.

# The two-sided 95% normal CI multiplier z_{0.975}. The single home of the literal both runners hard-coded
# as `_Z95` (the OT path takes its z from the driver's `_z_from_confidence`; this numpy fallback fixes 95%
# by construction — the runners report a 95% delta-method CI half-width).
Z95 = 1.959963984540054


class DeltaMethod(NamedTuple):
    """The numpy first-order delta-method bundle at n=1/input (the OT-absent diagonal twin of `gᵀΣg`):
    `grad` (∂f/∂input, by name), `a` (a_i=(grad·σ)² per input — the bound-tightening ranking key),
    `var` (Σ a_i = Var(E[f]) with one reading per input), `ci` (z·√var, the 95% half-width)."""
    grad: dict[str, float]
    a: dict[str, float]
    var: float
    ci: float


def delta_method(
    model_fn: Callable[[dict[str, float]], float], names: list[str],
    x0: dict[str, float], sigmas: dict[str, float],
) -> DeltaMethod:
    """The numpy delta-method bound of `model_fn` at `x0`: central-FD gradient (via the shared
    `alloc.gradient` seam), then `a_i=(grad·σ)²`, `var=Σa_i`, `ci=Z95·√var`. The runners' openturns-absent
    fallback — byte-for-byte the recipe they each inlined before move 5, so the bound numbers are
    unchanged (ADR-0009)."""
    grad = _grad.fd_gradient_dict(model_fn, names, x0)
    a = {nm: (grad[nm] * sigmas[nm]) ** 2 for nm in names}
    var = float(sum(a.values()))
    ci = float(Z95 * math.sqrt(max(var, 0.0)))
    return DeltaMethod(grad=grad, a=a, var=var, ci=ci)
