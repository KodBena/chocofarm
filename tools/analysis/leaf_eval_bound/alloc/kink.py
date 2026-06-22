"""
tools/analysis/leaf_eval_bound/alloc/kink.py
======================================

The Clark-1961 min()-KINK machinery — lifted out of `neyman_driver.py` as a PURE,
self-contained module (the responsibility-refactor's move 4;
`docs/design/leaf-eval-bound-responsibility-refactor.md` §2.3-D / §3 move 4). It owns the
§4.1 binding-margin diagnostic + the Clark-1961 closed-form `E[min]` / `Var[min]`: given a
model's two tightest min()-arms (each `(capacity, ∇capacity)`) and the joint input
covariance Σ, it returns the deterministic (O(1), NO Monte-Carlo) min-moments the
allocation driver uses near an arg-min tie — or None in the smooth regime.

It depends on NOTHING in the driver but the arm capacities + their covariances (numpy + a
lazy `scipy.stats.norm` for Φ/φ): the driver supplies the arms (it owns the `arms_fn`
adapter, `NeymanDriver._model_arms`, that reads them off the model) and Σ; this module
supplies the statistics. That is what makes it unit-testable on synthetic arm covariances
without an OpenTURNS `f`, and what keeps it INDEPENDENT of the gradient backend: the
planned OpenTURNS→JAX swap (§5) touches the smooth-gradient path only — the kink consumes
arm covariances, NOT `f.gradient()`, so the backend change cannot perturb it.

Public Domain (The Unlicense).
"""
from __future__ import annotations

import math
from typing import Any, Optional

import numpy as np

# §4.1: the kink regime fires only when a non-binding min()-arm has at least this probability of being
# the realized min (Φ(−t) ≥ floor) — a statistically-plausible tie, not numerical noise. Below it the
# contender is effectively never the min and the analytic single-arm gradient is honest (today's
# behavior). 1e-3 keeps the seed 8.6%-margin tie (Φ(−t)≈0.136) firing while not triggering on a
# comfortably-bound arm.
KINK_PFLOOR = 1e-3


def assess_min_kink(
    arms: Optional[list[tuple[float, np.ndarray]]], Sigma: np.ndarray, *, pfloor: float = KINK_PFLOOR
) -> Optional[dict[str, Any]]:
    """§4.1 — the `min()`-kink binding-margin diagnostic + the Clark-1961 closed form. Returns None
    (the smooth regime — today's behavior) unless the model's SECOND-tightest arm is within a
    statistically-plausible tie of the binding arm, in which case it returns the Clark moments
    (deterministic, O(1), NO Monte-Carlo) the kink path uses.

    `arms` is the model's min()-arms at the evaluation point, as `[(capacity, ∇capacity_ndarray), …]`
    — the driver extracts them via its `arms_fn` adapter (`NeymanDriver._model_arms`) and passes them
    here; absent a visible-min model the driver passes None (and a single arm is < 2), so this returns
    None — the honest default, never a fabricated tie. `Sigma` is the joint input covariance (the §2.2
    Σ). `pfloor` is the binding-margin trigger: the contender's arg-min probability `Φ(t)` must reach it
    to enter the kink regime.

    Each arm is linearized to `Normal(μ_k, σ_k²)` with `μ_k = capacity_k(μ̂)`, `σ_k² = ∇c_kᵀ Σ ∇c_k`,
    cross-covariance `∇c_aᵀ Σ ∇c_b`. The two tightest arms drive Clark's exact `min`-moments:
    `a = SD(c_bind − c_contender)`, `t = (μ_bind − μ_contender)/a`,
    `E[min] = μ_bind·Φ(−t) + μ_contender·Φ(t) − a·φ(t)`, and `Var[min]` from the second moment. An arm
    is the realized min iff it is the smaller draw, so `P(binding is min) = Φ(−t)` (the larger weight)
    and `P(contender is min) = Φ(t)` (the arg-min-flip probability). The kink regime fires when
    `P(contender is min) = Φ(t)` exceeds `pfloor` (a live arg-min flip). The both-arm allocation
    gradient is `Φ(±t)`-weighted (the SSTA criticality weights, summing to 1).

    Returns the dict `{E_min, var_min, p_nonbinding_max, t, a, grad_weighted, binding, contender}`
    (binding/contender are indices INTO `arms`), or None outside the kink regime.
    """
    if arms is None or len(arms) < 2:
        return None  # not a visible-min model -> smooth regime (today's behavior)

    # Each arm: (capacity μ_k, σ_k = sqrt(∇c_kᵀ Σ ∇c_k)).  Cross-cov via the same Σ.
    caps = np.array([c for (c, _g) in arms], dtype=float)
    arm_grads = [np.asarray(g, dtype=float) for (_c, g) in arms]
    arm_var = np.array([float(g @ Sigma @ g) for g in arm_grads])
    arm_sd = np.sqrt(np.maximum(arm_var, 0.0))

    # The binding arm is the realized min; the contender is the next-tightest capacity.
    order = np.argsort(caps)
    b, s = int(order[0]), int(order[1])  # binding, second
    mu_b, mu_s = float(caps[b]), float(caps[s])
    sd_b, sd_s = float(arm_sd[b]), float(arm_sd[s])
    cov_bs = float(arm_grads[b] @ Sigma @ arm_grads[s])

    # Degenerate / measure-zero guards (ADR-0002 honest about the one pathological case): if the
    # spread of (binding − contender) is ~0 there is no resolvable tie scale; treat as smooth.
    a_kink = math.sqrt(max(sd_b ** 2 + sd_s ** 2 - 2.0 * cov_bs, 0.0))
    if a_kink <= 1e-12:
        return None

    from scipy.stats import norm  # deterministic Φ, φ — no draws (ADR-0002 loud if scipy absent)
    # Clark's min-moments via min(X,Y) = −max(−X,−Y), arms (binding=b, contender=s) standardized
    # by the SD of their difference. t = (μ_b − μ_s)/a. An arm is the realized min iff it is the
    # SMALLER draw, so P(b is min) = P(b < s) = Φ((μ_s−μ_b)/a) = Φ(−t) (the binding arm, smaller
    # mean, usually wins); P(s is min) = Φ(t) (the contender's arg-min-flip probability). The
    # criticality weights Φ(−t), Φ(t) sum to 1.
    t = (mu_b - mu_s) / a_kink
    P_b_min = float(norm.cdf(-t))   # P(binding arm is the min)   = Φ(−t)   (the larger weight)
    P_s_min = float(norm.cdf(t))    # P(contender is the min)     = Φ(t)    (the flip probability)
    phi_t = float(norm.pdf(t))
    E_min = mu_b * P_b_min + mu_s * P_s_min - a_kink * phi_t
    E_sq = ((mu_b ** 2 + sd_b ** 2) * P_b_min
            + (mu_s ** 2 + sd_s ** 2) * P_s_min
            - (mu_b + mu_s) * a_kink * phi_t)
    var_min = max(E_sq - E_min ** 2, 0.0)

    # The binding-margin trigger: enter the kink regime only when the contender has a live chance
    # of being the realized min (Φ(t) = P(contender is min) above a small floor — a statistically-
    # plausible tie). Far from a tie this →0 and we return None (smooth; the analytic single-arm
    # gradient is honest, the non-binding arm's df/dx = 0 is correct, behavior is exactly today's).
    p_nonbinding_max = P_s_min
    if p_nonbinding_max < pfloor:
        return None

    # The Φ(±t)-weighted both-arm allocation gradient (SSTA criticality; weights sum to 1). Only
    # the two tightest arms carry weight; the rest are far and drop out (Φ→0). This funds the
    # previously-zero-weighted contender arm's inputs near the tie, curing the dead-gradient.
    grad_weighted = P_b_min * arm_grads[b] + P_s_min * arm_grads[s]
    return {
        "E_min": float(E_min),
        "var_min": float(var_min),
        "p_nonbinding_max": float(p_nonbinding_max),
        "t": float(t),
        "a": float(a_kink),
        "grad_weighted": grad_weighted,
        "binding": b,
        "contender": s,
    }
