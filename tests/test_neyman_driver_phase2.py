"""
tests/test_neyman_driver_phase2.py
==================================

§6 Phase 2 of the harmonized-estimator migration
(docs/design/harmonized-estimator-interface.md §6, the Phase-2 bullet): the DRIVER
consumes the `Estimate`. This is the behavior-CHANGING phase — it replaces what the
allocator computes (the diagonal variance sum -> the §2.2 `gᵀΣg` quadratic form) and
how convergence is decided (the §4.1 Clark-1961 `min()`-kink path + the §4.3 per-family
CI multiplier + the binding-margin convergence guard), while staying additive on the
all-mean / diagonal case (no regression).

The tests cover the §6 Phase-2 deliverables and the §8 EXECUTED verification targets:

  * DUAL-MODE input — `set_estimate`/`set_estimates_by_name` beside `add_samples`; `step()`
    prefers the Estimate, else wraps the pool as a `Poolwise` Estimate. A pool-fed and an
    Estimate-fed driver AGREE on the mean case (the confirmed fixed point).
  * `gᵀΣg` — equals today's diagonal sum on an all-mean model (no regression), and folds
    in the off-diagonal cross-term a declared `cross` carries.
  * the SOCP allocation (§2.3) — reduces to the closed form `n_i* ∝ √(a_i/c_i)` on the
    diagonal (rel diff ~1e-5), hits `gᵀΣ(n*)g = V*` exactly on a non-diagonal Σ, and the
    sign-safe Q-form does NOT silently misallocate on mixed-sign gradients (the §8 corr-3
    trap), guarded by the fail-loud `gᵀΣ(n*)g ≈ V*` assertion.
  * the Clark `kink_regime` path (§4.1) — reproduces `E[min]/sd/Φ(−t)` = 415.68/25.58/0.322
    at the σ₁=60 stress case AND 426.5/6.2/0.136 at the propagated σ₁≈25.17 operating point
    (σ sourced from the `Estimate.cov`), funds both arms by the Φ(±t) weights, and REFUSES
    convergence while the arg-min-flip probability exceeds α.
  * the per-family CI multiplier (§4.3) — NORMAL→z, STUDENT_T(dof)→t_dof, the mixed-family
    conservative multiplier.

The `estimate`/`neyman_driver` modules live under tools/analysis/OpenTURNS/ (no
__init__.py — imported by sys.path the way manifest.py imports them), so this test
prepends that directory. openturns + cvxpy + scipy are required (the driver's deps); the
tests skip loudly if a dep is genuinely absent rather than asserting a false pass.

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
    "tools", "analysis", "OpenTURNS",
)
if _OT not in sys.path:
    sys.path.insert(0, _OT)

ot = pytest.importorskip("openturns", reason="neyman_driver requires openturns")
pytest.importorskip("scipy", reason="the Clark closed form needs scipy.stats.norm")

import estimate as E  # noqa: E402  — the Estimate contract
import neyman_driver as ND  # noqa: E402  — the Phase-2 driver under test
from neyman_driver import NeymanDriver, _t_multiplier  # noqa: E402

_HAS_CVXPY = __import__("importlib").util.find_spec("cvxpy") is not None


# --------------------------------------------------------------------------- #
# Helpers — build the two Estimate kinds the tests need.
# --------------------------------------------------------------------------- #
def _poolwise(name: str, mean: float, per_sample_var: float, n: int, cross=None) -> E.Estimate:
    """A k=1 Poolwise mean Estimate: cov = s²/n (already divided), per_sample_var = s²."""
    return E.Estimate(
        theta_hat=np.array([mean], dtype=float),
        cov=np.array([[per_sample_var / n]], dtype=float),
        names=(name,),
        shrink=E.Poolwise(per_sample_var=np.array([per_sample_var], dtype=float)),
        support=(E.Support.POSITIVE,),
        family=(E.CIFamily.NORMAL,),
        cross=(cross or {}),
        kind="mean",
    )


def _fixed(name: str, mean: float, sigma: float) -> E.Estimate:
    """A k=1 Fixed (declared-spread) Estimate: cov = sigma² (un-divided, un-shrinkable)."""
    return E.Estimate(
        theta_hat=np.array([mean], dtype=float),
        cov=np.array([[sigma * sigma]], dtype=float),
        names=(name,),
        shrink=E.Fixed(),
        support=(E.Support.POSITIVE,),
        family=(E.CIFamily.NORMAL,),
        kind="declared_spread",
    )


def _legacy_diagonal_step(f, costs, tol, names, pools, z=1.959963984540054):
    """The EXACT pre-Phase-2 step() math, recomputed standalone for the no-regression comparison."""
    n = np.array([len(p) for p in pools], dtype=float)
    mu = np.array([p.mean() for p in pools])
    sigma = np.array([p.std(ddof=1) for p in pools])
    g = f.gradient(ot.Point(mu))
    grad = np.array([g[i, 0] for i in range(len(names))])
    a = (grad * sigma) ** 2
    var_est = float((a / n).sum())
    ci = z * math.sqrt(max(var_est, 0.0))
    return dict(a=a, var_contrib=a / n, var=var_est, ci=ci)


# --------------------------------------------------------------------------- #
# 1. Dual-mode input + the pool/Estimate fixed point.
# --------------------------------------------------------------------------- #
def test_set_estimate_rejects_non_estimate_and_bad_index() -> None:
    """ADR-0002: set_estimate validates the input contract — a non-Estimate, or an out-of-range index,
    is a loud error, never a silent accept."""
    f = ot.SymbolicFunction(["x0", "x1"], ["x0 + x1"])
    d = NeymanDriver(f, costs=[1.0, 1.0], tolerance=1.0, names=["x0", "x1"])
    with pytest.raises(TypeError):
        d.set_estimate(0, {"mean": 1.0})  # a bespoke dict is exactly what the contract forbids
    with pytest.raises(IndexError):
        d.set_estimate(5, _poolwise("x0", 1.0, 1.0, 10))


def test_set_estimates_by_name_unknown_name_raises() -> None:
    f = ot.SymbolicFunction(["x0", "x1"], ["x0 + x1"])
    d = NeymanDriver(f, costs=[1.0, 1.0], tolerance=1.0, names=["x0", "x1"])
    with pytest.raises(KeyError):
        d.set_estimates_by_name({"not_an_input": _poolwise("q", 1.0, 1.0, 10)})


def test_pool_and_estimate_fed_drivers_agree_on_the_mean_case() -> None:
    """THE confirmed fixed point (§6 Phase-2 deliverable 1): a pool-fed driver and an Estimate-fed
    driver produce the SAME var_estimate, ci, per-input a, and recommendation on an all-means model —
    because step() wraps a raw pool as exactly the Poolwise Estimate the Estimate-fed driver is handed."""
    f = ot.SymbolicFunction(["x0", "x1"], ["3*x0 - 1.5*x1"])
    costs, tol = [1.0, 2.0], 0.5
    rng = np.random.default_rng(11)
    p0, p1 = rng.normal(10, 3, 60), rng.normal(20, 2, 45)

    dp = NeymanDriver(f, costs=costs, tolerance=tol, names=["x0", "x1"], confidence=0.95)
    dp.add_samples({0: p0, 1: p1})
    rp = dp.step(second_order_check=False)

    de = NeymanDriver(f, costs=costs, tolerance=tol, names=["x0", "x1"], confidence=0.95)
    for i, pool in ((0, p0), (1, p1)):
        de.set_estimate(i, _poolwise(["x0", "x1"][i], float(pool.mean()), float(pool.var(ddof=1)),
                                     len(pool)))
    re = de.step(second_order_check=False)

    assert math.isclose(rp.var_estimate, re.var_estimate, rel_tol=1e-12, abs_tol=1e-15)
    assert rp.ci_halfwidth == re.ci_halfwidth
    assert rp.estimate == re.estimate
    pa = sorted(rp.primitives, key=lambda p: p.index)
    ea = sorted(re.primitives, key=lambda p: p.index)
    assert [p.a for p in pa] == [p.a for p in ea]
    assert [p.recommend for p in pa] == [p.recommend for p in ea]


# --------------------------------------------------------------------------- #
# 2. gᵀΣg — no regression on the diagonal, cross-term on the non-diagonal.
# --------------------------------------------------------------------------- #
def test_gtsigmag_equals_legacy_diagonal_sum_no_regression() -> None:
    """§2.2 exactness: on an all-means diagonal model `gᵀΣg` equals today's `sum a_i/n_i` to machine
    epsilon (the matrix form reorders the float summation — a 1-ULP round-off, not a semantic change),
    the per-sample `p.a = (g·σ)²` is byte-for-byte the legacy field, and `p.var_contribution` is the
    legacy `a/n`. This is the no-regression guarantee for every all-mean model."""
    names = ["x0", "x1", "x2"]
    f = ot.SymbolicFunction(names, ["3*x0 - 1.5*x1 + 0.7*x2"])
    costs, tol = [1.0, 2.5, 0.8], 0.5
    rng = np.random.default_rng(7)
    pools = [rng.normal(10, 3, 60), rng.normal(20, 2, 45), rng.normal(5, 4, 80)]
    leg = _legacy_diagonal_step(f, costs, tol, names, pools)

    d = NeymanDriver(f, costs=costs, tolerance=tol, names=names, confidence=0.95)
    d.add_samples({i: pools[i] for i in range(3)})
    rec = d.step(second_order_check=False)

    assert math.isclose(rec.var_estimate, leg["var"], rel_tol=1e-12, abs_tol=1e-15)
    assert math.isclose(rec.ci_halfwidth, leg["ci"], rel_tol=1e-12, abs_tol=1e-15)
    prims = sorted(rec.primitives, key=lambda p: p.index)
    assert [p.a for p in prims] == list(leg["a"])  # per-sample a EXACT
    np.testing.assert_allclose([p.var_contribution for p in prims], leg["var_contrib"], atol=1e-12)


def test_gtsigmag_folds_in_the_cross_term() -> None:
    """§4.2: a declared `cross` makes Σ non-diagonal, and `gᵀΣg` picks up the `2·g_i·g_j·Σ_ij` cross-term
    the diagonal sum drops — materially changing the variance (here a negative off-diagonal with
    same-sign gradients LOWERS it). The assembled Σ carries the off-diagonal symmetrically."""
    f = ot.SymbolicFunction(["x0", "x1"], ["2*x0 + 3*x1"])  # same-sign gradients g=[2,3]
    n0, n1, s0, s1, corr = 50, 50, 9.0, 4.0, -0.81
    Sig00, Sig11 = s0 / n0, s1 / n1
    Sig01 = corr * math.sqrt(Sig00 * Sig11)
    e0 = _poolwise("x0", 10.0, s0, n0, cross={"x1": Sig01})
    e1 = _poolwise("x1", 5.0, s1, n1, cross={"x0": Sig01})
    d = NeymanDriver(f, costs=[1.0, 1.5], tolerance=0.5, names=["x0", "x1"])
    Sigma = d._assemble_sigma([e0, e1])
    assert Sigma[0, 1] == Sigma[1, 0]
    assert math.isclose(Sigma[0, 1], Sig01, rel_tol=1e-12)
    g = np.array([2.0, 3.0])
    var_cov = float(g @ Sigma @ g)
    var_diag = g[0] ** 2 * Sig00 + g[1] ** 2 * Sig11
    assert var_cov < var_diag  # the negative cross-term with same-sign g LOWERS the variance
    assert math.isclose(var_cov, var_diag + 2 * g[0] * g[1] * Sig01, rel_tol=1e-12)


def test_assemble_sigma_rejects_disagreeing_cross_homes() -> None:
    """ADR-0002 / P1: a cross coupling declared by BOTH sides with DIFFERENT values is two homes for one
    number that disagree — a loud fault, not a silent pick."""
    f = ot.SymbolicFunction(["x0", "x1"], ["x0 + x1"])
    e0 = _poolwise("x0", 1.0, 1.0, 10, cross={"x1": -0.05})
    e1 = _poolwise("x1", 1.0, 1.0, 10, cross={"x0": -0.09})  # disagrees with e0's -0.05
    d = NeymanDriver(f, costs=[1.0, 1.0], tolerance=1.0, names=["x0", "x1"])
    with pytest.raises(ValueError):
        d._assemble_sigma([e0, e1])


# --------------------------------------------------------------------------- #
# 3. The SOCP allocation (§2.3).
# --------------------------------------------------------------------------- #
def test_socp_reduces_to_closed_form_on_the_diagonal() -> None:
    """§2.3 / §8(b): on a diagonal Σ the allocation is the closed form `n_i* ∝ √(a_i/c_i)`. The driver
    dispatches to the closed form on the diagonal (exact + robust to scaling); we check it reproduces
    the textbook Neyman ratio to a tight relative tolerance."""
    f = ot.SymbolicFunction(["x0", "x1", "x2"], ["3*x0 - 1.5*x1 + 0.7*x2"])
    costs = np.array([1.0, 2.5, 0.8])
    rng = np.random.default_rng(7)
    pools = [rng.normal(10, 3, 60), rng.normal(20, 2, 45), rng.normal(5, 4, 80)]
    d = NeymanDriver(f, costs=list(costs), tolerance=0.5, names=["x0", "x1", "x2"])
    d.add_samples({i: pools[i] for i in range(3)})
    ests = [d._estimate_for(i) for i in range(3)]
    Sigma = d._assemble_sigma(ests)
    mu = np.array([float(e.theta_hat[0]) for e in ests])
    grad = d._gradient(mu)
    ncur = np.array([d._effective_n(i, ests[i]) for i in range(3)], dtype=float)
    z = d.z
    V_target = (0.5 / z) ** 2

    n_star = d._socp_allocation(grad, Sigma, d.costs, V_target, ncur, ests=ests)

    # The closed-form Neyman ratio scaled to hit V*.
    sigma = np.array([float(p.std(ddof=1)) for p in pools])
    a = (grad * sigma) ** 2
    S = float(np.sqrt(a * costs).sum())
    n_cf = np.sqrt(a / costs) * (S / V_target)
    rel = float(np.max(np.abs(n_star - n_cf) / np.abs(n_cf)))
    assert rel < 1e-3  # closed form reproduces the ratio (the driver IS the closed form on the diagonal)


@pytest.mark.skipif(not _HAS_CVXPY, reason="the non-diagonal SOCP needs cvxpy (CLARABEL)")
def test_socp_hits_v_star_on_a_nondiagonal_sigma() -> None:
    """§2.3 / §8(b): on a NON-diagonal Σ the SOCP fires (not the closed form) and the returned `n*`
    realizes `gᵀΣ(n*)g = V*` EXACTLY — the case the closed form cannot express. Same-sign gradients."""
    f = ot.SymbolicFunction(["x0", "x1"], ["2*x0 + 3*x1"])
    n0, n1, s0, s1, corr = 50, 50, 9.0, 4.0, -0.81
    Sig00, Sig11 = s0 / n0, s1 / n1
    Sig01 = corr * math.sqrt(Sig00 * Sig11)
    e0 = _poolwise("x0", 10.0, s0, n0, cross={"x1": Sig01})
    e1 = _poolwise("x1", 5.0, s1, n1, cross={"x0": Sig01})
    d = NeymanDriver(f, costs=[1.0, 1.5], tolerance=0.5, names=["x0", "x1"])
    Sigma = d._assemble_sigma([e0, e1])
    grad = np.array([2.0, 3.0])
    V = 5.0
    n_star = d._socp_allocation(grad, Sigma, d.costs, V, np.array([n0, n1], float), ests=[e0, e1])

    sig2 = np.array([s0, s1]) / n_star
    R = np.array([[1.0, corr], [corr, 1.0]])
    Sig_star = np.outer(np.sqrt(sig2), np.sqrt(sig2)) * R
    var_real = float(grad @ Sig_star @ grad)
    assert math.isclose(var_real, V, rel_tol=1e-4)


@pytest.mark.skipif(not _HAS_CVXPY, reason="the sign-safe Q-form SOCP needs cvxpy (CLARABEL)")
def test_socp_sign_safe_on_mixed_sign_gradients() -> None:
    """§8 correction 3: the sign-safe Q-form returns a CORRECT allocation on MIXED-SIGN gradients (the
    naive v=u/√n form silently misallocates, claiming `optimal` while the true Var≠V*). The driver's
    fail-loud `gᵀΣ(n*)g ≈ V*` assertion is what guarantees this — a returned allocation always realizes
    V* or the call raises. `model_capacity` HAS mixed-sign gradients, so this is the live case."""
    f = ot.SymbolicFunction(["x0", "x1"], ["x0 + x1"])  # f arbitrary; we drive _socp_allocation directly
    n0, n1, s0, s1, corr = 50, 50, 9.0, 4.0, -0.81
    Sig00, Sig11 = s0 / n0, s1 / n1
    Sig01 = corr * math.sqrt(Sig00 * Sig11)
    e0 = _poolwise("x0", 10.0, s0, n0, cross={"x1": Sig01})
    e1 = _poolwise("x1", 5.0, s1, n1, cross={"x0": Sig01})
    d = NeymanDriver(f, costs=[1.0, 1.5], tolerance=0.5, names=["x0", "x1"])
    Sigma = d._assemble_sigma([e0, e1])
    grad = np.array([-2.0, 3.0])  # MIXED sign — the trap input
    V = 5.0
    n_star = d._socp_allocation(grad, Sigma, d.costs, V, np.array([n0, n1], float), ests=[e0, e1])
    sig2 = np.array([s0, s1]) / n_star
    R = np.array([[1.0, corr], [corr, 1.0]])
    Sig_star = np.outer(np.sqrt(sig2), np.sqrt(sig2)) * R
    var_real = float(grad @ Sig_star @ grad)
    assert math.isclose(var_real, V, rel_tol=1e-4)  # the Q-form realizes V* on mixed signs


def test_fixed_pin_drops_out_of_allocation() -> None:
    """§2.3: a Fixed/declared-spread pin has irreducible variance, so it gets NO allocation (its n is
    unchanged) — but it still contributes its a_i to the bound (via gᵀΣg). The 'don't sample dead
    inputs' branch, now for the right reason (irreducible, not merely a==0)."""
    f = ot.SymbolicFunction(["x0", "x1"], ["2*x0 + 3*x1"])
    d = NeymanDriver(f, costs=[1.0, 1.0], tolerance=0.5, names=["x0", "x1"])
    d.set_estimate(0, _poolwise("x0", 10.0, 9.0, 50))   # shrinkable mean
    d.set_estimate(1, _fixed("x1", 5.0, 2.0))           # un-shrinkable pin
    rec = d.step(second_order_check=False)
    prims = {p.name: p for p in rec.primitives}
    assert prims["x1"].recommend == 0       # the pin is never funded
    assert prims["x1"].a > 0                 # but it DOES contribute to the bound (a_i > 0)


# --------------------------------------------------------------------------- #
# 4. The Clark min()-kink path (§4.1) — the §8 reproduction targets.
# --------------------------------------------------------------------------- #
def _kink_driver(sigma_R: float) -> NeymanDriver:
    """A min(producer, serve) driver whose producer arm reads {N_gen, R_gen} (input-disjoint from the
    serve arm). producer = N_gen·R_gen with N_gen=3±0.05, R_gen=152±sigma_R; serve=428.28±2. σ₁ (the
    producer spread) propagates through N·R from the Estimate covs. The arms hook supplies the per-arm
    capacities + gradients the Clark path linearizes."""
    names = ["N_gen", "R_gen", "serve_cap"]
    f = ot.SymbolicFunction(names, ["min(N_gen*R_gen, serve_cap)"])
    d = NeymanDriver(f, costs=[0.5, 30.0, 8.0], tolerance=5.0, names=names, confidence=0.95)
    d.set_estimate(0, _fixed("N_gen", 3.0, 0.05))
    d.set_estimate(1, _fixed("R_gen", 152.0, sigma_R))
    d.set_estimate(2, _fixed("serve_cap", 428.28, 2.0))

    def arms_fn(x):
        N, R, S = x["N_gen"], x["R_gen"], x["serve_cap"]
        producer = (N * R, {"N_gen": R, "R_gen": N, "serve_cap": 0.0})
        serve = (S, {"N_gen": 0.0, "R_gen": 0.0, "serve_cap": 1.0})
        return [producer, serve]

    d.arms_fn = arms_fn
    return d


def test_clark_kink_reproduces_stress_sigma1_60() -> None:
    """§8(a) STRESS: at producer σ₁=60 the Clark closed form reproduces E[min]=415.68, sd=25.58,
    Φ(−t)=P(producer is min)=0.322 — deterministically, no Monte-Carlo. σ₁ is sourced from the
    Estimate covs (σ_R chosen so √((R·σ_N)²+(N·σ_R)²)=60), NOT a literal."""
    sigma_R = math.sqrt(60.0 ** 2 - (152 * 0.05) ** 2) / 3.0  # -> propagated σ₁ = 60
    d = _kink_driver(sigma_R)
    rec = d.step(second_order_check=False)
    assert rec.kink_regime is True
    assert rec.estimate_kink == pytest.approx(415.68, abs=0.05)
    assert math.sqrt(rec.var_estimate) == pytest.approx(25.58, abs=0.05)
    assert rec.p_nonbinding_max == pytest.approx(0.322, abs=0.002)
    # the de-biased E[min] is BELOW the hard min (the −a·φ(t) Jensen correction, min concave).
    assert rec.estimate_kink < rec.estimate


def test_clark_kink_reproduces_operating_sigma1_25() -> None:
    """§8(a) OPERATING (the production anchor): with the SEED σ_R=8 the producer σ₁ propagates to
    √((152·0.05)²+(3·8)²)=25.17, and Clark gives E[min]=426.5, sd=6.2, Φ(−t)=0.136 — the real operating
    point, NOT the dramatic σ₁=60 stress figure. This is the production rule: source σ from the cov."""
    d = _kink_driver(sigma_R=8.0)  # the seed -> propagated σ₁ = 25.17
    rec = d.step(second_order_check=False)
    assert rec.kink_regime is True
    assert rec.estimate_kink == pytest.approx(426.5, abs=0.1)
    assert math.sqrt(rec.var_estimate) == pytest.approx(6.2, abs=0.05)
    assert rec.p_nonbinding_max == pytest.approx(0.136, abs=0.002)


def test_clark_kink_funds_both_arms_and_guard_refuses_convergence() -> None:
    """§4.1 mechanisms 1+3: in the kink regime BOTH contending arms' inputs get funded (the Φ(±t)-weighted
    gradient cures the dead-gradient on the non-binding arm), and convergence is REFUSED while the
    arg-min-flip probability Φ(−t) exceeds α (here 0.136 > 0.05) — the false-SAT the guard forbids."""
    d = _kink_driver(sigma_R=8.0)
    rec = d.step(second_order_check=False)
    assert rec.converged is False                 # guard refuses: P(flip)=0.136 > alpha=0.05
    # both arms are live: the producer inputs and the serve input both register (the contender no longer
    # has the hard-min's df/dx = 0). At least one producer input gets funded OR the forward-progress
    # nudge fires on a contending input — the dead-gradient pathology is cured.
    funded = {p.name: p.recommend for p in rec.primitives}
    assert sum(funded.values()) > 0


def test_no_kink_regime_without_the_arms_hook() -> None:
    """§4.1 honest default: absent the model arms hook the driver CANNOT see the min structure (OT cannot
    differentiate through min() anyway), so it stays in the smooth regime — it never fabricates a tie. A
    min() model WITHOUT arms_fn behaves exactly as today (kink_regime False, single-arm gᵀΣg)."""
    names = ["N_gen", "R_gen", "serve_cap"]
    f = ot.SymbolicFunction(names, ["min(N_gen*R_gen, serve_cap)"])
    d = NeymanDriver(f, costs=[0.5, 30.0, 8.0], tolerance=5.0, names=names)
    d.set_estimate(0, _fixed("N_gen", 3.0, 0.05))
    d.set_estimate(1, _fixed("R_gen", 152.0, 8.0))
    d.set_estimate(2, _fixed("serve_cap", 428.28, 2.0))
    # no d.arms_fn set
    rec = d.step(second_order_check=False)
    assert rec.kink_regime is False
    assert rec.estimate_kink is None


def test_kink_collapses_to_smooth_far_from_a_tie() -> None:
    """§4.1: away from a tie (a comfortably-bound contender) Φ(−t)→0 and the driver returns to the smooth
    regime — the analytic single-arm gradient is honest, the non-binding arm's df/dx=0 is correct. Here
    the producer (456) is far above a much-lower serve (200), so no kink fires."""
    names = ["N_gen", "R_gen", "serve_cap"]
    f = ot.SymbolicFunction(names, ["min(N_gen*R_gen, serve_cap)"])
    d = NeymanDriver(f, costs=[0.5, 30.0, 8.0], tolerance=5.0, names=names)
    d.set_estimate(0, _fixed("N_gen", 3.0, 0.05))
    d.set_estimate(1, _fixed("R_gen", 152.0, 8.0))
    d.set_estimate(2, _fixed("serve_cap", 200.0, 2.0))  # serve far below producer -> no tie

    def arms_fn(x):
        N, R, S = x["N_gen"], x["R_gen"], x["serve_cap"]
        return [(N * R, {"N_gen": R, "R_gen": N, "serve_cap": 0.0}),
                (S, {"N_gen": 0.0, "R_gen": 0.0, "serve_cap": 1.0})]

    d.arms_fn = arms_fn
    rec = d.step(second_order_check=False)
    assert rec.kink_regime is False


# --------------------------------------------------------------------------- #
# 5. The per-family CI multiplier (§4.3).
# --------------------------------------------------------------------------- #
def test_family_multiplier_normal_is_z() -> None:
    """§4.3: an all-NORMAL set uses the z multiplier (today's behavior)."""
    f = ot.SymbolicFunction(["x0", "x1"], ["3*x0 - 1.5*x1"])
    d = NeymanDriver(f, costs=[1.0, 2.0], tolerance=0.5, names=["x0", "x1"])
    d.set_estimate(0, _poolwise("x0", 10.0, 9.0, 50))
    d.set_estimate(1, _poolwise("x1", 20.0, 4.0, 50))
    rec = d.step(second_order_check=False)
    assert rec.ci_multiplier == pytest.approx(d.z)
    assert rec.ci_multiplier_label == "z"


def test_family_multiplier_student_t_widens() -> None:
    """§4.3: a STUDENT_T(dof=5) fit-coefficient input widens the multiplier to t_{5,0.975}≈2.571 (vs
    z=1.96, a 31% wider CI honestly reported). The mixed-family case is LABELLED conservative."""
    assert _t_multiplier(5, 0.95) == pytest.approx(2.5706, abs=1e-3)
    f = ot.SymbolicFunction(["x0", "x1"], ["3*x0 - 1.5*x1"])
    d = NeymanDriver(f, costs=[1.0, 2.0], tolerance=0.5, names=["x0", "x1"])
    d.set_estimate(0, E.Estimate(
        theta_hat=np.array([10.0]), cov=np.array([[0.5]]), names=("x0",),
        shrink=E.RegressionLaw(resid_var=1.0, XtX_inv=np.array([[1.0]]),
                               design=np.array([[1.0], [2.0]])),
        support=(E.Support.POSITIVE,), family=(E.StudentT(dof=5),), kind="ols_fit"))
    d.set_estimate(1, _poolwise("x1", 20.0, 4.0, 50))  # NORMAL
    rec = d.step(second_order_check=False)
    assert rec.ci_multiplier == pytest.approx(2.5706, abs=1e-3)
    assert "t(dof=5)" in rec.ci_multiplier_label


# --------------------------------------------------------------------------- #
# 6. run() dual-mode (measurers / samplers).
# --------------------------------------------------------------------------- #
def test_run_requires_exactly_one_of_measurers_or_samplers() -> None:
    """ADR-0002: run() takes EXACTLY ONE input contract — both, or neither, is a loud error."""
    f = ot.SymbolicFunction(["x0"], ["x0"])
    d = NeymanDriver(f, costs=[1.0], tolerance=1.0, names=["x0"])
    with pytest.raises(ValueError):
        d.run()  # neither
    with pytest.raises(ValueError):
        d.run(measurers={0: lambda b: _poolwise("x0", 1.0, 1.0, 10)},
              samplers={0: lambda k: np.zeros(int(k))})  # both


def test_run_measurers_form_converges() -> None:
    """The §6 Phase-2 `measurers[i](budget) -> Estimate` form drives the loop to convergence on a smooth
    all-mean model (the form the migrated runners move to in Phase 4)."""
    f = ot.SymbolicFunction(["x0", "x1"], ["x0 + 2*x1"])
    d = NeymanDriver(f, costs=[1.0, 1.0], tolerance=2.0, names=["x0", "x1"], confidence=0.95)
    rng = np.random.default_rng(1)
    state = {0: [], 1: []}

    def make_measurer(i, mean, sd):
        def m(budget):
            state[i].extend(rng.normal(mean, sd, int(budget)).tolist())
            pool = np.array(state[i])
            return _poolwise(["x0", "x1"][i], float(pool.mean()), float(pool.var(ddof=1)), len(pool))
        return m

    rec = d.run(measurers={0: make_measurer(0, 10, 3), 1: make_measurer(1, 5, 1)},
                pilot=64, max_rounds=20, verbose=False)
    assert rec.converged is True
    assert rec.ci_halfwidth <= 2.0


def test_run_legacy_samplers_form_still_works() -> None:
    """Backward compat: the legacy `samplers[i](k) -> array` form (which untrusted_drive used pre-Phase-4,
    before it moved to `measurers` -> Estimate) still drives the loop — Phase 2 is additive, the
    `add_samples` shim is kept, not a breaking change."""
    f = ot.SymbolicFunction(["x0", "x1"], ["x0 + 2*x1"])
    d = NeymanDriver(f, costs=[1.0, 1.0], tolerance=2.0, names=["x0", "x1"], confidence=0.95)
    rng = np.random.default_rng(0)
    rec = d.run(samplers={0: lambda k: rng.normal(10, 3, int(k)),
                          1: lambda k: rng.normal(5, 1, int(k))},
                pilot=64, max_rounds=20, verbose=False)
    assert rec.converged is True
