"""
tools/analysis/OpenTURNS/neyman_driver.py
=========================================

The GENERIC, model-agnostic Neyman optimal-allocation driver — a benchmark-budget
allocator for functional uncertainty-propagation models. It owns NO specific model
(ADR-0012 P1 single-home / P2 separation of the allocator-transport from the thing
allocated): a concrete throughput model is a SEPARATE module (e.g.
`examples/demo_msgpass.py`, `model_capacity.py`, `model_cycletime.py`) that builds an
`ot.Function` and hands it to `NeymanDriver`. The synthetic demo that previously lived
in this file (the `_demo()` impurity) was extracted to
`tools/analysis/OpenTURNS/examples/demo_msgpass.py` per the ADR-0012 purification.

Public Domain (The Unlicense).

A benchmark-budget allocator for functional uncertainty-propagation models.

You have a deterministic expression  y = f(x_1, ..., x_d)  (e.g. a crude
throughput model) whose inputs x_i are physical quantities you can *sample*
from a real system at some per-sample cost c_i (message-passing latency,
context-switch cost, inference throughput, ...). You estimate E[f] by f(mu_hat),
where each input mean mu_hat_i is the average of n_i collected samples.

The variance of that estimate, by the delta method (inputs independent), is

        Var(f(mu_hat)) ~= sum_i (df/dx_i)^2 * sigma_i^2 / n_i
                        =  sum_i a_i / n_i,      a_i := (df/dx_i)^2 sigma_i^2.

Minimising total cost  sum_i c_i n_i  subject to  sum_i a_i / n_i <= V*
gives Neyman allocation:

        n_i*  proportional to  sqrt(a_i / c_i),

scaled to hit V*. Because a_i depends on sigma_i and the gradient at mu --
which you only know after sampling -- the procedure is iterative: pilot,
estimate a_i, allocate a top-up, repeat. This module runs that loop.

OpenTURNS supplies f and its gradient (with a finite-difference fallback), and
an optional TaylorExpansionMoments diagnostic flags when the linearisation the
allocation rests on is no longer trustworthy (Jensen / curvature).

Dependencies: numpy, openturns (>=1.17 or so). TaylorExpansionMoments and the
JointDistribution/ComposedDistribution naming are version-sensitive and are
both guarded.

Author's note: this allocates effort to shrink the CI on E[f] (the expected
ceiling). It does NOT shrink sigma_f = sqrt(sum_i a_i), the genuine run-to-run
spread of f itself, which is physical and irreducible by input sampling.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

try:
    import openturns as ot
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "neyman_driver requires openturns: pip install openturns"
    ) from exc


# --------------------------------------------------------------------------- #
# Result containers
# --------------------------------------------------------------------------- #
@dataclass
class PrimitiveState:
    """Per-input bookkeeping for one iteration."""
    index: int
    name: str
    n: int                       # current pool size
    mean: float                  # mu_hat_i
    sigma: float                 # sample std of the pool
    grad: float                  # df/dx_i at current mean point
    a: float                     # (df/dx_i)^2 * sigma_i^2
    cost: float                  # c_i, cost per sample
    var_contribution: float      # a_i / n_i  (this input's share of Var(E[f]))
    recommend: int               # additional samples suggested this round


@dataclass
class Recommendation:
    """The output of one driver step()."""
    iteration: int
    converged: bool
    estimate: float              # current f(mu_hat)
    estimate_second_order: Optional[float]  # curvature-corrected mean, if available
    var_estimate: float          # Var(E[f]) ~= sum a_i / n_i
    ci_halfwidth: float          # z * sqrt(var_estimate)
    target_halfwidth: float      # requested tolerance h
    shadow_price: float          # d(cost)/dV*, marginal price of variance
    primitives: List[PrimitiveState] = field(default_factory=list)

    def where_to_spend(self) -> List[PrimitiveState]:
        """Primitives ranked by recommended additional samples (desc)."""
        return sorted(self.primitives, key=lambda p: p.recommend, reverse=True)

    def report(self) -> str:
        lines = []
        status = "CONVERGED" if self.converged else "continue"
        lines.append(
            f"[iter {self.iteration}] {status}  "
            f"E[f]={self.estimate:.4g}"
            + (
                f" (2nd-order {self.estimate_second_order:.4g})"
                if self.estimate_second_order is not None
                else ""
            )
        )
        lines.append(
            f"  CI half-width = {self.ci_halfwidth:.4g}  "
            f"(target {self.target_halfwidth:.4g})   "
            f"shadow price lambda = {self.shadow_price:.4g}"
        )
        header = (
            f"  {'primitive':<16}{'n':>8}{'sigma':>12}"
            f"{'|df/dx|':>12}{'a_i':>12}{'a_i/n_i':>12}{'+samples':>10}"
        )
        lines.append(header)
        lines.append("  " + "-" * (len(header) - 2))
        for p in self.where_to_spend():
            lines.append(
                f"  {p.name:<16}{p.n:>8d}{p.sigma:>12.4g}"
                f"{abs(p.grad):>12.4g}{p.a:>12.4g}"
                f"{p.var_contribution:>12.4g}{p.recommend:>10d}"
            )
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
class NeymanDriver:
    """
    Iterative optimal-allocation driver for an OpenTURNS scalar function f.

    Parameters
    ----------
    f : ot.Function
        Scalar-output function of d inputs. Built however you like
        (SymbolicFunction, PythonFunction, composite). Its .gradient() is used
        if available; otherwise central finite differences on f are used.
    costs : sequence of float
        Per-sample cost c_i of benchmarking each primitive. Relative scale is
        all that matters. Use wall-clock seconds, or 1.0 for all if uniform.
    tolerance : float
        Target CI half-width h on E[f]. The loop stops when z*sqrt(Var(E[f])) <= h.
    names : sequence of str, optional
        Human-readable primitive names (defaults to x0, x1, ...).
    confidence : float
        Two-sided confidence level for the CI (default 0.95 -> z=1.96).
    growth_cap : float
        Max multiplicative growth of any pool per round (damping; default 3.0).
    max_batch : int, optional
        Hard cap on additional samples per primitive per round.
    fd_rel_step : float
        Relative step for the finite-difference gradient fallback.
    """

    def __init__(
        self,
        f: "ot.Function",
        costs: Sequence[float],
        tolerance: float,
        names: Optional[Sequence[str]] = None,
        confidence: float = 0.95,
        growth_cap: float = 3.0,
        max_batch: Optional[int] = None,
        fd_rel_step: float = 1e-5,
    ):
        self.f = f
        self.d = f.getInputDimension()
        if f.getOutputDimension() != 1:
            raise ValueError("f must have scalar output (output dimension 1).")
        self.costs = np.asarray(costs, dtype=float)
        if self.costs.shape != (self.d,):
            raise ValueError(f"costs must have length {self.d}.")
        if np.any(self.costs <= 0):
            raise ValueError("costs must be positive.")
        self.tolerance = float(tolerance)
        self.names = list(names) if names is not None else [f"x{i}" for i in range(self.d)]
        self.z = float(_z_from_confidence(confidence))
        self.growth_cap = float(growth_cap)
        self.max_batch = max_batch
        self.fd_rel_step = float(fd_rel_step)

        # pools[i] is a 1-D numpy array of collected samples for primitive i
        self.pools: List[np.ndarray] = [np.empty(0) for _ in range(self.d)]
        self.iteration = 0

    # ------------------------------------------------------------------ #
    # Pool management
    # ------------------------------------------------------------------ #
    def add_samples(self, new: Dict[int, np.ndarray]) -> None:
        """Merge freshly benchmarked samples. Keys are primitive indices."""
        for i, vals in new.items():
            vals = np.asarray(vals, dtype=float).ravel()
            self.pools[i] = np.concatenate([self.pools[i], vals])

    def add_samples_by_name(self, new: Dict[str, np.ndarray]) -> None:
        idx = {name: i for i, name in enumerate(self.names)}
        self.add_samples({idx[k]: v for k, v in new.items()})

    # ------------------------------------------------------------------ #
    # Gradient (OT analytic if present, else central FD on f)
    # ------------------------------------------------------------------ #
    def _gradient(self, point: np.ndarray) -> np.ndarray:
        try:
            g = self.f.gradient(ot.Point(point))
            return np.array([g[i, 0] for i in range(self.d)], dtype=float)
        except Exception:
            return self._fd_gradient(point)

    def _fd_gradient(self, point: np.ndarray) -> np.ndarray:
        g = np.empty(self.d)
        f0_pt = point.copy()
        for i in range(self.d):
            h = self.fd_rel_step * max(abs(point[i]), 1.0)
            xp = f0_pt.copy(); xp[i] += h
            xm = f0_pt.copy(); xm[i] -= h
            yp = float(self.f(ot.Point(xp))[0])
            ym = float(self.f(ot.Point(xm))[0])
            g[i] = (yp - ym) / (2.0 * h)
        return g

    # ------------------------------------------------------------------ #
    # Optional curvature diagnostic via TaylorExpansionMoments
    # ------------------------------------------------------------------ #
    def _second_order_mean(self) -> Optional[float]:
        """
        Build a kernel-smoothed joint distribution from the current pools and
        ask OpenTURNS for the second-order (curvature-corrected) mean. Returns
        None if anything in this version-sensitive path is unavailable.
        """
        try:
            marginals = []
            for i in range(self.d):
                sample = ot.Sample(self.pools[i].reshape(-1, 1))
                marginals.append(ot.KernelSmoothing().build(sample))
            dist = _make_joint(marginals)
            inputRV = ot.RandomVector(dist)
            outputRV = ot.CompositeRandomVector(self.f, inputRV)
            taylor = ot.TaylorExpansionMoments(outputRV)
            return float(taylor.getMeanSecondOrder()[0])
        except Exception:
            return None

    # ------------------------------------------------------------------ #
    # The allocation step
    # ------------------------------------------------------------------ #
    def step(self, second_order_check: bool = True) -> Recommendation:
        """
        Re-estimate a_i from current pools and recommend additional samples.

        Requires every pool to have >= 2 samples (run a pilot first). Returns a
        Recommendation; collect its suggested samples, call add_samples(), and
        call step() again until .converged is True.
        """
        self.iteration += 1
        n = np.array([len(p) for p in self.pools], dtype=float)
        if np.any(n < 2):
            missing = [self.names[i] for i in range(self.d) if n[i] < 2]
            raise RuntimeError(
                f"Need a pilot of >=2 samples for every primitive first. "
                f"Missing/short: {missing}"
            )

        mu = np.array([p.mean() for p in self.pools])
        sigma = np.array([p.std(ddof=1) for p in self.pools])
        grad = self._gradient(mu)
        a = (grad * sigma) ** 2                      # a_i
        var_contrib = np.where(n > 0, a / n, np.inf)  # a_i / n_i
        var_est = float(var_contrib.sum())            # Var(E[f])
        ci_half = self.z * math.sqrt(max(var_est, 0.0))
        estimate = float(self.f(ot.Point(mu))[0])

        V_target = (self.tolerance / self.z) ** 2
        converged = var_est <= V_target

        # Neyman optimal TOTAL allocation to hit V_target.
        # n_i* = sqrt(a_i/c_i) * (sum_j sqrt(a_j c_j)) / V_target
        sqrt_ac = np.sqrt(a * self.costs)
        S = float(sqrt_ac.sum())
        with np.errstate(divide="ignore", invalid="ignore"):
            n_star = np.sqrt(a / self.costs) * (S / V_target)
        n_star = np.where(a > 0, n_star, n)           # don't sample dead inputs

        # Incremental top-up toward the optimal totals. The raw target is
        # (n_star - n); we damp the *whole vector* by a single scalar so the
        # Neyman proportions between primitives are preserved (clipping each
        # primitive independently would flatten the allocation toward uniform,
        # which is exactly wrong -- it would fund cheap, irrelevant inputs).
        topup = np.maximum(n_star - n, 0.0)
        if not converged:
            scale = 1.0
            # Damp 1: no pool grows by more than growth_cap * n in one round.
            grow_lim = self.growth_cap * np.maximum(n, 1.0)
            binding = topup > grow_lim
            if np.any(binding):
                scale = min(scale, float(np.min(grow_lim[binding] / topup[binding])))
            # Damp 2: respect an absolute per-round batch ceiling.
            if self.max_batch is not None:
                over = topup > self.max_batch
                if np.any(over):
                    scale = min(scale, float(self.max_batch) / float(topup.max()))
            topup = np.floor(topup * scale)
            # Forward progress if rounding zeroed the vector.
            if topup.sum() == 0:
                worst = int(np.argmax(var_contrib))
                topup[worst] = max(1.0, math.ceil(0.5 * n[worst]))
        else:
            topup[:] = 0.0
        topup = topup.astype(int)

        # Shadow price of variance: d(cost)/dV* = -(sum sqrt(a c))^2 / V*^2.
        shadow = (S ** 2) / (V_target ** 2) if V_target > 0 else float("inf")

        second_order = (
            self._second_order_mean() if (second_order_check and not converged) else None
        )

        prims = [
            PrimitiveState(
                index=i, name=self.names[i], n=int(n[i]), mean=float(mu[i]),
                sigma=float(sigma[i]), grad=float(grad[i]), a=float(a[i]),
                cost=float(self.costs[i]), var_contribution=float(var_contrib[i]),
                recommend=int(topup[i]),
            )
            for i in range(self.d)
        ]

        return Recommendation(
            iteration=self.iteration, converged=converged, estimate=estimate,
            estimate_second_order=second_order, var_estimate=var_est,
            ci_halfwidth=ci_half, target_halfwidth=self.tolerance,
            shadow_price=shadow, primitives=prims,
        )

    # ------------------------------------------------------------------ #
    # Autonomous loop (when you can sample programmatically)
    # ------------------------------------------------------------------ #
    def run(
        self,
        samplers: Dict[int, Callable[[int], np.ndarray]],
        pilot: int = 256,
        max_rounds: int = 25,
        verbose: bool = True,
    ) -> Recommendation:
        """
        Drive the whole loop when each primitive can be sampled by a callback
        samplers[i](k) -> array of k samples. For manual benchmarking, ignore
        this and use step()/add_samples() by hand.
        """
        # Pilot.
        self.add_samples({i: samplers[i](pilot) for i in samplers})
        rec = self.step()
        if verbose:
            print(rec.report(), "\n")
        rounds = 0
        while not rec.converged and rounds < max_rounds:
            batch = {
                p.index: samplers[p.index](p.recommend)
                for p in rec.primitives if p.recommend > 0
            }
            self.add_samples(batch)
            rec = self.step()
            rounds += 1
            if verbose:
                print(rec.report(), "\n")
        return rec


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _z_from_confidence(confidence: float) -> float:
    try:
        return float(ot.Normal().computeQuantile(0.5 + confidence / 2.0)[0])
    except Exception:
        return 1.959963984540054  # 95% fallback


def _make_joint(marginals):
    """JointDistribution (newer OT) or ComposedDistribution (older)."""
    if hasattr(ot, "JointDistribution"):
        return ot.JointDistribution(marginals)
    return ot.ComposedDistribution(marginals)


# This module owns NO model and has no __main__ (ADR-0012 P1/P2): the synthetic
# message-passing demo that previously lived here was extracted to
# tools/analysis/OpenTURNS/examples/demo_msgpass.py. Run a concrete model via its own
# module (examples/demo_msgpass.py, model_capacity.py, model_cycletime.py) or the
# throughput_bound.py runner.
