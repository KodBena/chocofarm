"""
tools/analysis/leaf_eval_bound/benchmarks/bench_common.py
===================================================

Shared scaffolding for the leaf-eval transport benchmark modules (`bench_<name>.py`), so
each bench owns ONLY its measurement loop + its v1 seed — never the register/connect/log
boilerplate (ADR-0012 P3 one-owner: this module owns the bench<->store glue; bench_store.py
owns the SQL; each bench owns its physics). A bench module is a thin object:

    SEED = ...                       # the v1 Grounded fallback (get_seed() returns it)
    def get_seed(): return SEED
    def register_self(): return register_quantity(NAME, quantity=…, units=…, …)
    def run(...): -> measure, then `with logged_run(NAME, config) as log: log(values)`

`logged_run` is the one helper the run() bodies use: it registers the definition (idempotent),
opens an instance with the repo git_sha + host + config, OPTIONALLY persists a harmonized
`Estimate` as the instance's `estimate` jsonb (the §5.1 SSOT), hands back a `log(values,
sample_size)` callable for the raw-reading provenance, and on exit leaves a populated instance. A
bench that measures NOTHING during the parallel workflow (timing-sensitive) still exposes run() —
it just must not be CALLED then (the manifest gates rerun behind an explicit operator action).

`fit_estimate` is the §6 Phase-3 FIT helper: it turns a `time = intercept + slope·rows` OLS fit
over per-width medians into ONE k=2 harmonized `Estimate` — the full 2×2 `cov = resid_var·(AᵀA)⁻¹`
carrying the slope/intercept −0.81 off-diagonal (docs/design/harmonized-estimator-interface.md
§4.2), `RegressionLaw`/`StudentT(dof=n_pts−2)`. It is computed HERE (not by `_fit_line`, which
discards the design+residuals) so the fit slice needs NO shared-helper change / base-method-override
audit (ADR-0004 minimal-touch).

`median_estimate` / `pin_estimate` are the §6 Phase-3 MEDIAN and PIN helpers (the second slice).
`median_estimate` turns a raw latency/cost pool (headline `np.median`) into ONE k=1 `QuantileLaw`
`Estimate` with a BOOTSTRAP median SE (§7.A — not the small-sample-fragile asymptotic `p(1−p)/(n·f̂²)`),
`family=EMPIRICAL`, `kind='median'`. `pin_estimate` turns a declared constant / declared spread into
ONE k=1 `Fixed` `Estimate` — `cov=[[σ²]]` UN-DIVIDED (recovering e.g. B_op's σ=64 that the §5
`stddev_samp`-over-one-value bug discards), `family=DEGENERATE` for a true constant or `NORMAL` for a
declared-spread prior (§3). Both mirror `fit_estimate`'s shape: the bench computes its `Estimate` in
`run()` from `measure()`'s pool/seed and passes it to `logged_run(..., estimate=…)`.

`collect_pool` is the race-collector POOL FLOOR (RCA fix #2): a `len(pool) >= min_readings` guarantee for
a producer/consumer bench whose realized reading count is DECOUPLED from the requested effort (it
coalesces edges + drops torn reads, so a tiny allocator budget yields < 2 readings -> `median_estimate`
RAISES — the ~/shm_spin_poll_fail wakeup crash). It re-runs the batch at growing effort until the floor is
met (the floor binds on readings COLLECTED, not effort), then `median_estimate` consumes the floored pool;
ONE home for the guarantee (P1) so a new race bench inherits it, never re-deriving a retry loop.

`window_pool` is the DETERMINISTIC counterpart (RCA fix #2, the DRY half): the ONE home of the
`for _ in range(N): pool.append(measure_one_window())` idiom the median benches hand-copied (the tau_io
family, gather, req_drain, zmq_baseline_wakeup, the tmsg family). It takes the per-window measurement as a
closure + the window count and runs the loop `max(min_windows, count)` times — so unlike the race collector
(whose count it cannot promise, hence collect_pool's retry), a window loop's count IS the budget and the
floor is owned STRUCTURALLY (`len(pool) >= 2` by construction, P1/P3). At count >= 2 it is a pure refactor
of the inline loop (ADR-0009 behavioral equivalence); the only change is the >= 2 floor at a tiny budget.

FAIL LOUD (ADR-0002). A registration/insert error propagates as a typed psycopg error; a degenerate
fit (too few / collinear design points) RAISES in `fit_estimate`, never a padded low-info estimate.
The git_sha read is best-effort (a bench may run outside a checkout) and degrades to None, which is
a recorded provenance gap, not a swallowed failure.

Public Domain (The Unlicense).
"""
from __future__ import annotations

import contextlib
import os
import subprocess
import sys
import uuid
from typing import Any, Callable, Iterator, Mapping, Optional, Sequence

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)  # the OpenTURNS dir (holds bench_store, leaf_eval_grounding)
for _p in (_PARENT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import bench_store  # noqa: E402
import estimate as _est  # noqa: E402  — the harmonized Estimate contract (the type SSOT, ADR-0012 P8)


def repo_git_sha() -> Optional[str]:
    """The repo HEAD short-SHA for sample provenance, or None outside a checkout (best-effort — a missing
    SHA is a recorded gap, not a failure: a sole-workload bench run on a detached tree still logs)."""
    try:
        out = subprocess.run(
            ["git", "-C", _HERE, "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip() or None if out.returncode == 0 else None
    except Exception:
        return None


def register_quantity(
    name: str, *, quantity: str, units: str, description: str, module_path: str
) -> uuid.UUID:
    """Idempotently register this bench's quantity (a `benchmark_definition` row) and return its id. The
    `module_path` SHOULD be the dotted import path of the bench module (e.g. `benchmarks.bench_t_row`) so
    the manifest can re-import it for get_seed()/run(). Loud if postgres is down (registration is the
    registry write)."""
    return bench_store.register_definition(
        name, quantity=quantity, units=units, description=description, module_path=module_path)


# ============================================================================================
# The §6 Phase-3 FIT slice: turn a `time = intercept + slope·rows` OLS fit over per-width medians
# into ONE k=2 harmonized `Estimate` (docs/design/harmonized-estimator-interface.md §4.2/§5).
#
# WHY THE COVARIANCE IS COMPUTED HERE (not in `_fit_line`). `_fit_line`
# (chocofarm/az/bench/bench_mlp_lowlatency.py) returns only (intercept, slope, r2) and DISCARDS
# the design matrix A and the residuals — so `Cov(slope, intercept)` (the −0.81 off-diagonal the
# whole §4.2 slice is about) is not even computed today (§5). Per ADR-0004 minimal-touch, the
# LESS-INVASIVE route the spec names (§4.2/§5) is to recompute `(AᵀA)⁻¹` and `resid_var` HERE from
# the bench's OWN `(rows, medians)` — no shared-helper change, so NO base-method-override audit and
# NO cpp/ behavioral re-verification. The fit numbers (intercept/slope) are byte-identical to
# `_fit_line`'s lstsq because the design `A = [1, rows]` is the same; this adds only the cov/SE the
# `lstsq` threw away.
#
# THE COMPONENT ORDERING (the §4.2 + driver contract, the load-bearing subtlety). The driver
# (neyman_driver.set_estimate / _assemble_sigma / manifest._project_estimate) reads a multi-component
# estimate's FIRST component as the input's marginal (theta_hat[0], cov[0,0]) and the off-diagonal
# via `cross[partner_name]`. The 8 live `manifest.value("t_row_us")` consumers read the SLOPE mean,
# and `value("T_disp_us")` reads the INTERCEPT mean — so the Estimate a fit bench logs must order its
# OWN quantity as component 0 (else the first-component projection hands the slope-reader the
# intercept — a silent wrong number, ADR-0002). So `own_role` selects which of {intercept, slope} is
# component 0; the full 2×2 cov (carrying the −0.81) is row/col-permuted to match, and the SAME
# off-diagonal is ALSO placed in `cross={partner_name: off}` (the form `_assemble_sigma` reads — §4.2
# "the slope/intercept pairing a Phase-3 bench populates"). Both fit partners thus log the SAME fit
# (same correlation, same information), each ordered so ITS quantity projects correctly.
# ============================================================================================
def fit_estimate(
    rows: Sequence[float],
    medians_us: Sequence[float],
    *,
    own_name: str,
    own_role: str,
    partner_name: str,
) -> "_est.Estimate":
    """Build the k=2 OLS-fit `Estimate` for a `time = intercept + slope·rows` fit, with `own_role`'s
    component (∈ {'intercept', 'slope'}) placed FIRST so the manifest's first-component projection
    gives THIS quantity its right mean (§4.2). Carries the full 2×2 `cov = resid_var·(AᵀA)⁻¹` (the
    −0.81 off-diagonal), `shrink=RegressionLaw(resid_var, XtX_inv, design)`, `family=StudentT(dof=
    n_pts−2)` per component, POSITIVE support, `cross={partner_name: off_diagonal}`, `kind='ols_fit'`.

    FAIL LOUD (ADR-0002): a degenerate fit RAISES — fewer than 3 design points (dof = n−2 < 1, no
    Student-t), a collinear/zero-spread x-design (`Sxx == 0`, `(AᵀA)` singular), or any non-finite
    input. It NEVER returns a padded low-information estimate."""
    if own_role not in ("intercept", "slope"):
        raise ValueError(f"fit_estimate: own_role must be 'intercept' or 'slope'; got {own_role!r}")
    x = np.asarray(rows, dtype=np.float64)
    y = np.asarray(medians_us, dtype=np.float64)
    if x.ndim != 1 or y.ndim != 1 or x.shape != y.shape:
        raise ValueError(
            f"fit_estimate: rows {x.shape} and medians_us {y.shape} must be 1-D and the same length")
    n = int(x.shape[0])
    if n < 3:
        raise ValueError(
            f"fit_estimate({own_name!r}): need >= 3 design points for an OLS fit with a Student-t SE "
            f"(dof = n_pts - 2 >= 1); got n={n} (ADR-0002: a degenerate fit raises, it is not padded).")
    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
        raise ValueError(f"fit_estimate({own_name!r}): rows/medians must be finite (ADR-0002).")
    Sxx = float(np.sum((x - x.mean()) ** 2))
    if not (Sxx > 0.0):
        raise ValueError(
            f"fit_estimate({own_name!r}): the x-design has zero spread (Sxx={Sxx}); the slope SE is "
            f"undefined and (AᵀA) is singular (ADR-0002: a collinear design is a loud fault).")

    # The SAME design `_fit_line` uses: columns [1, rows] -> coeffs [intercept, slope]. So intercept is
    # index 0 and slope is index 1 in the NATURAL (fit) ordering. theta_hat/cov are computed here, then
    # permuted to put `own_role` first.
    A = np.vstack([np.ones_like(x), x]).T            # (n, 2): [1, x]
    AtA = A.T @ A
    XtX_inv_nat = np.linalg.inv(AtA)                 # (2, 2) symmetric — (AᵀA)⁻¹, the unscaled cov
    (intercept, slope), *_ = np.linalg.lstsq(A, y, rcond=None)
    yhat = intercept + slope * x
    ss_res = float(np.sum((y - yhat) ** 2))
    # The unbiased residual variance s² = SS_res / (n − p), p = 2 (intercept + slope). cov = s²·(AᵀA)⁻¹
    # is the textbook OLS sampling covariance (already an SE², the contract's invariant — §1 D1).
    resid_var = ss_res / float(n - 2)
    cov_nat = resid_var * XtX_inv_nat                # (2, 2): the FULL sampling cov, off-diagonal carried

    theta_nat = np.array([intercept, slope], dtype=np.float64)
    names_nat = ("intercept", "slope")               # role labels for the permutation, not registry names

    # Permute so `own_role` is component 0 (the marginal the manifest/driver project). A swap when the
    # bench's own quantity is the SLOPE (t_row / t_row_bare); identity when it is the INTERCEPT
    # (iota / T_disp). The permutation reorders theta_hat, cov (rows AND cols), the design columns, and
    # the per-component (name, units, family) together so every per-component field stays aligned.
    perm = (0, 1) if names_nat[0] == own_role else (1, 0)
    theta = theta_nat[list(perm)]
    cov = cov_nat[np.ix_(perm, perm)]
    XtX_inv = XtX_inv_nat[np.ix_(perm, perm)]
    design = A[:, list(perm)]                         # design columns match the (now-permuted) coeff order
    off = float(cov[0, 1])                            # Cov(own, partner) — the −0.81-correlated cross-term
    dof = n - 2                                       # the OLS-coefficient Student-t dof (§4.3)
    # `own` is component 0 by construction (the permutation above), the partner is component 1; so
    # `names = (own_name, partner_name)` regardless of perm.
    return _est.Estimate(
        theta_hat=theta,
        cov=cov,
        names=(own_name, partner_name),
        shrink=_est.RegressionLaw(resid_var=resid_var, XtX_inv=XtX_inv, design=design),
        support=(_est.Support.POSITIVE, _est.Support.POSITIVE),
        family=(_est.StudentT(dof=dof), _est.StudentT(dof=dof)),
        cross={partner_name: off},
        kind="ols_fit",
    )


# ============================================================================================
# ============================================================================================
# The race-collector POOL FLOOR (RCA fix #2, docs/notes/leaf-eval-estimator-pin-cascade-rca.md): a
# SHARED `len(pool) >= min_readings` guarantee for a RACE-BASED collector whose realized reading count
# is DECOUPLED from the requested effort. A producer/consumer wakeup bench (shm_spin_poll, futex_wake,
# lockfree_mpsc, cpp_inproc_port) coalesces edges it polls past and drops torn reads, so a batch returns
# FEWER readings than the effort asked — and at a small allocator budget can return < 2, which
# `median_estimate` then RAISES on (the shm_spin_poll wakeup crash: budget 6 -> 1 reading -> raise). The
# floor binds on READINGS COLLECTED, not on the requested effort: re-run the batch at growing effort
# until the accumulated pool reaches the floor. ONE home for the guarantee (ADR-0012 P1) so a new race
# bench inherits it by calling this, not by re-deriving a retry loop (ADR-0011 Rule 4: a structural net
# over the class, not a per-bench patch). It NEVER fabricates a reading (ADR-0002): an un-yielding
# collector RAISES at the attempt cap.
# ============================================================================================
def collect_pool(
    collect_batch: Callable[[int], Sequence[float]],
    *,
    name: str,
    budget: int,
    min_readings: int = 8,
    max_attempts: int = 12,
) -> list[float]:
    """Accumulate a latency/cost pool to a floor of `min_readings` readings from a RACE-BASED collector.
    `collect_batch(effort) -> Sequence[float]` runs ONE batch at the given effort and returns its
    (possibly short) pool; `collect_pool` runs it at `effort = max(min_readings, budget)` and, while the
    accumulated pool is under the floor, RE-RUNS it at doubled effort — so the floor binds on readings
    COLLECTED, not on the requested effort (the count a race collector cannot promise). Returns the
    accumulated pool (`len >= min_readings`).

    `min_readings` defaults to 8 — comfortably above `median_estimate`'s HARD minimum of 2 so the bootstrap
    median SE is non-degenerate (2 readings risk a zero-spread pool, the OTHER median_estimate raise). The
    normal path (a real allocator budget) yields hundreds in the first batch and never retries, so the
    floor binds only at the pathological tiny budget that produced the crash.

    FAIL LOUD (ADR-0002): if `max_attempts` batches (effort up to `budget·2^(max_attempts-1)`) still
    under-yield, RAISE — a collector that cannot reach `min_readings` is a real fault (a wedged producer, a
    pathological over-coalescing), never a sub-floor pool padded into a fake median."""
    if min_readings < 2:
        raise ValueError(
            f"collect_pool({name!r}): min_readings must be >= 2 (a bootstrap median SE needs >= 2 "
            f"readings); got {min_readings} (ADR-0002).")
    pool: list[float] = []
    effort = max(int(min_readings), int(budget))
    for _ in range(int(max_attempts)):
        pool.extend(float(x) for x in collect_batch(effort))
        if len(pool) >= min_readings:
            return pool
        effort *= 2
    raise ValueError(
        f"collect_pool({name!r}): only {len(pool)} reading(s) after {max_attempts} batches (final effort "
        f"{effort}) — the race collector under-yields below the floor {min_readings}; a real fault (a "
        f"wedged/over-coalescing producer), not a sub-floor pool to pad (ADR-0002).")


# ============================================================================================
# The DETERMINISTIC WINDOW-LOOP pool builder (RCA fix #2, the DRY half;
# docs/notes/leaf-eval-estimator-pin-cascade-rca.md §5.1 "factor the window-loop idiom" / §5.2c):
# the deterministic COUNTERPART to `collect_pool`. The leaf-eval median benches whose `_measure_raw`
# runs a `for _ in range(N): pool.append(measure_one_window())` loop (the tau_io family, gather,
# req_drain, zmq_baseline_wakeup, the tmsg family) hand-copied that loop ≈12 times, each differing
# only in the per-window measurement + setup — the audit's cancer D (copy-paste) / P1 (no single
# home). `window_pool` is the ONE home (ADR-0012 P1/P3: a parameterized collaborator, the per-window
# measurement injected as a closure), so a new deterministic window bench inherits the loop by
# calling this, never re-deriving it.
#
# WHY A SEPARATE HELPER FROM `collect_pool` (the deterministic↔race asymmetry). A window loop has a
# KNOWN reading count — exactly the budget, ONE reading per window iteration — because the loop body
# is timed deterministically (no edge-coalescing, no torn-read drops), unlike a race collector whose
# realized count is decoupled from the effort. So `collect_pool`'s RETRY-until-floor machinery
# (re-run the batch at growing effort) is the wrong shape here: there is nothing to retry, the count
# is the loop bound. `window_pool` instead OWNS THE FLOOR STRUCTURALLY — it runs the loop
# `max(min_windows, count)` times, so `len(pool) >= min_windows >= 2` BY CONSTRUCTION. This makes the
# deterministic benches EXPLICITLY safe (a 1-window pool RAISES in `median_estimate`, ADR-0002):
# before this, each bench leaned on the driver's `max(2, …)` budget floor (untrusted_drive
# `_make_measurer`) plus its own ad-hoc `n_windows = max(2, …)` — a per-bench guard the audit names
# as the right instinct applied per-bench (RCA §5.2c); making the floor a PROPERTY OF THE CONTRACT
# closes the gap symmetrically with `collect_pool`.
#
# BEHAVIORAL EQUIVALENCE (ADR-0009). At any `count >= min_windows` (the normal operating regime — a
# real allocator budget is hundreds), `max(min_windows, count) == count`, so the loop runs EXACTLY
# `count` times: the migration is a pure refactor (same closure body, same readings, same pool), the
# `min_windows` default of 2 reproducing the benches' existing `max(2, …)` floor byte-for-byte. The
# ONLY intended change is at a tiny `count < 2` (the floor lifts it to 2) — the same safety
# improvement `collect_pool` made for the race family. `window_pool` owns ONLY the count/floor
# guarantee (the finiteness / zero-spread checks stay single-homed in `median_estimate`, its sole
# gate — this helper does not duplicate them).
# ============================================================================================
def window_pool(
    measure_window: Callable[[], float],
    *,
    name: str,
    count: int,
    min_windows: int = 2,
) -> list[float]:
    """Build a deterministic latency/cost pool by calling `measure_window()` once per window — the
    shared home of the `for _ in range(N): pool.append(one_window())` idiom. `measure_window() ->
    float` times ONE window (the per-window measurement the bench injects; its setup/warmup/teardown
    stay in the bench, around the call) and returns that window's reading; `window_pool` runs it
    `n = max(min_windows, count)` times and returns the `n` readings (so `len(pool) >= min_windows`,
    the >= 2 the bootstrap median SE needs — the floor is structural, not per-bench).

    `min_windows` defaults to 2 — `median_estimate`'s HARD minimum (a 1-reading pool has no bootstrap
    spread and RAISES, ADR-0002), and exactly the floor the deterministic benches carried inline as
    `max(2, …)`. The normal path (a real allocator budget) passes `count` in the hundreds, so the
    floor never binds and the loop runs `count` times unchanged; the floor binds only at the
    pathological tiny budget, lifting it to 2.

    FAIL LOUD (ADR-0002): `min_windows < 2` is itself a contract violation (a bootstrap median SE
    needs >= 2 readings) and RAISES — symmetric with `collect_pool`. It NEVER fabricates a reading;
    each window's value is whatever `measure_window()` returns (the finiteness / zero-spread gate is
    `median_estimate`'s, the single home for that check)."""
    if min_windows < 2:
        raise ValueError(
            f"window_pool({name!r}): min_windows must be >= 2 (a bootstrap median SE needs >= 2 "
            f"readings); got {min_windows} (ADR-0002).")
    n = max(int(min_windows), int(count))
    return [float(measure_window()) for _ in range(n)]


# The §6 Phase-3 MEDIAN/LATENCY slice: turn a raw pool of per-cycle/per-trial/per-window readings
# (whose headline is `np.median(pool)`) into ONE k=1 harmonized `Estimate`
# (docs/design/harmonized-estimator-interface.md §3 MEDIAN/QUANTILE row, §7.A).
#
# WHY THE SE IS A BOOTSTRAP, NOT THE ASYMPTOTIC `p(1−p)/(n·f̂²)` (§7.A, the review refinement).
# The order-statistic variance of a sample median IS `p(1−p)/(n·f̂(median)²)`, but `f̂(median)` — the
# density AT the quantile — is small-sample-FRAGILE (a kernel-density estimate at a single point is
# the noisiest thing one can read off a finite latency pool, and timing data is right-skewed/heavy-
# tailed: the benches compute a median precisely BECAUSE the mean is tail-poisoned). So the spec
# §7.A says PREFER a bootstrap median SE — steadier at the bench `n` — and declare `family=EMPIRICAL`
# (a sample quantile's CI is NOT a t-interval, §4.3; fabricating an asymptotic Normal SE here would
# be the over-claim ADR-0009 forbids). The bootstrap SE is the VARIANCE AUTHORITY in `cov`.
#
# WHY `QuantileLaw.f_at_q` IS BACK-DERIVED FROM THE BOOTSTRAP SE (not an independent KDE). The
# contract's `shrink=QuantileLaw(p, f_at_q, n)` carries `f_at_q` for the driver's allocation
# MARGINAL hook (`cov(n) = p(1−p)/(n·f_at_q²)` — how one more reading shrinks the variance). To keep
# the law SELF-CONSISTENT with the `cov` it ships beside (P1 single-home: the variance has ONE home,
# the bootstrap, and the law must agree with it rather than assert a second number), `f_at_q` is the
# value that makes the law's implied variance EQUAL the bootstrap variance at the current `n`:
# `f̂ = sqrt(p(1−p)/(n·Var_boot))`. So `cov` is the bootstrap SSOT and `QuantileLaw` reproduces it
# (NOT a fabricated independent asymptotic) — and the marginal `d cov/d n = −cov/n` is the honest
# 1/n shrink the order-statistic law gives, with the bootstrap-calibrated density. (If a bench ever
# prefers a true KDE `f̂`, it passes it; absent one, the bootstrap-calibrated value is the least-bad
# honest choice and is labelled EMPIRICAL, never NORMAL.)
# ============================================================================================
def median_estimate(
    pool: Sequence[float],
    *,
    name: str,
    n_boot: int = 2000,
    boot_seed: int = 0,
) -> "_est.Estimate":
    """Build the k=1 median `Estimate` for a raw latency/cost pool: `theta_hat=[median(pool)]`,
    `cov=[[median_SE²]]` with a BOOTSTRAP median SE (§7.A — NOT the small-sample-fragile asymptotic
    `p(1−p)/(n·f̂²)`), `shrink=QuantileLaw(p=0.5, f_at_q=[f̂], n=len(pool))` (the `f̂` back-derived from
    the bootstrap SE so the law's implied variance equals `cov`), `family=(EMPIRICAL,)`, POSITIVE
    support, `kind='median'`. The bootstrap resamples the pool WITH REPLACEMENT `n_boot` times and
    takes the std (ddof=1) of the resampled medians — a deterministic SE under the fixed `boot_seed`.

    FAIL LOUD (ADR-0002): a pool that cannot yield a defensible variance RAISES, never pads — an empty
    pool, fewer than 2 readings (no bootstrap spread), any non-finite reading, or a DEGENERATE pool
    whose bootstrap medians are all identical (SE == 0, so the order-statistic density `f̂` would be
    infinite — a real fault: a zero-spread latency pool is not a measured median, it is a constant
    masquerading as one, which the spec's MEDIAN row does not cover; surface it, do not fabricate a
    QuantileLaw the data cannot support)."""
    p = np.asarray(pool, dtype=np.float64)
    if p.ndim != 1:
        raise ValueError(f"median_estimate({name!r}): pool must be 1-D; got shape {p.shape}")
    n = int(p.shape[0])
    if n < 2:
        raise ValueError(
            f"median_estimate({name!r}): need >= 2 readings for a bootstrap median SE; got n={n} "
            f"(ADR-0002: a 1-sample 'median' has no defensible variance — it raises, it is not padded).")
    if not np.all(np.isfinite(p)):
        raise ValueError(f"median_estimate({name!r}): pool readings must be finite (ADR-0002).")
    med = float(np.median(p))
    # Bootstrap the median SE: n_boot resamples of size n, with replacement; the SE is the std of the
    # resampled medians. Steadier than the asymptotic density estimate at the bench n (§7.A).
    rng = np.random.default_rng(boot_seed)
    idx = rng.integers(0, n, size=(int(n_boot), n))
    boot_meds = np.median(p[idx], axis=1)
    median_se = float(np.std(boot_meds, ddof=1))
    if not (median_se > 0.0) or not np.isfinite(median_se):
        raise ValueError(
            f"median_estimate({name!r}): the bootstrap median SE is {median_se} (the pool's bootstrap "
            f"medians are degenerate — a zero-spread pool is a constant, not a measured median). "
            f"(ADR-0002: a median with no defensible spread raises; use a Fixed pin for a true constant.)")
    var_med = median_se * median_se
    # Back-derive the density-at-median f̂ so QuantileLaw's implied variance `p(1−p)/(n·f̂²)` EQUALS the
    # bootstrap var (the law agrees with the cov SSOT, it does not assert an independent asymptotic).
    f_at_q = float(np.sqrt(0.25 / (n * var_med)))  # p(1−p) = 0.25 at p=0.5
    return _est.Estimate(
        theta_hat=np.array([med], dtype=np.float64),
        cov=np.array([[var_med]], dtype=np.float64),
        names=(name,),
        shrink=_est.QuantileLaw(p=0.5, f_at_q=np.array([f_at_q], dtype=np.float64), n=n),
        support=(_est.Support.POSITIVE,),
        family=(_est.CIFamily.EMPIRICAL,),
        kind="median",
    )


# ============================================================================================
# The §6 Phase-3 PIN slice: a declared constant / declared engineering-judgement spread into ONE k=1
# `Fixed` `Estimate` (docs/design/harmonized-estimator-interface.md §3 PIN rows, §1 D2/D4).
#
# THE TWO PIN FLAVORS (the §3 distinction the family encodes). A TRUE CONSTANT (n_gen, a deployment
# pinning fact, σ tiny) is `family=DEGENERATE` (no sampling interval; a_i≈0, drops out of allocation)
# and `kind='pin'`. A DECLARED-SPREAD PRIOR (B_op's σ=64, LPD's σ=25, R_gen's σ=8 — an engineering-
# judgement 1-sigma the manifest seeds) is `family=NORMAL` (a prior the models consume as
# `Normal(mean, sigma)` — matching `manifest._estimate_from_seed`) and `kind='declared_spread'`: it
# CONTRIBUTES its a_i to the bound (so the CI honestly rests on the prior) but is un-shrinkable by
# sampling (`shrink=Fixed()`; no finite budget reduces an engineering-judgement prior). EITHER way the
# variance is the declared spread UN-DIVIDED — `cov=[[σ²]]`, NOT σ²/n (a prior has no n) — which is
# the §5 store-bug fix: `stddev_samp` over one logged value returns NULL→0, DISCARDING the declared σ
# (B_op's 64 today lives only in `Grounded`/the seed path, never reaching the instance row's variance).
# ============================================================================================
def pin_estimate(
    value: float,
    sigma: float,
    *,
    name: str,
    constant: bool = False,
) -> "_est.Estimate":
    """Build the k=1 `Fixed` `Estimate` for a pin: `theta_hat=[value]`, `cov=[[sigma²]]` (the declared
    spread UN-DIVIDED — un-shrinkable, a prior has no n), `shrink=Fixed()`, POSITIVE support. A TRUE
    CONSTANT (`constant=True`, e.g. n_gen's layout fact) is `family=DEGENERATE` / `kind='pin'`; a
    DECLARED-SPREAD PRIOR (`constant=False`, the default — B_op σ=64, LPD σ=25, …) is `family=NORMAL`
    / `kind='declared_spread'` (the prior the models treat as `Normal(mean, sigma)`).

    FAIL LOUD (ADR-0002): a non-finite value/sigma, or a negative sigma, RAISES. A declared-spread pin
    with `sigma == 0` is also a fault (`family=NORMAL` with zero variance is a contradiction — a prior
    with no spread is a true constant, so pass `constant=True`); a true constant MAY carry `sigma == 0`
    (a `DEGENERATE` zero-variance point is valid). It NEVER coerces a bad spread into a plausible one."""
    v, s = float(value), float(sigma)
    if not (np.isfinite(v) and np.isfinite(s)):
        raise ValueError(f"pin_estimate({name!r}): value and sigma must be finite; got ({v}, {s}) (ADR-0002).")
    if s < 0.0:
        raise ValueError(f"pin_estimate({name!r}): sigma must be >= 0 (a spread); got {s} (ADR-0002).")
    if not constant and not (s > 0.0):
        raise ValueError(
            f"pin_estimate({name!r}): a declared-spread prior (family=NORMAL) needs sigma > 0; got {s}. "
            f"A spread-less prior is a true constant — pass constant=True for family=DEGENERATE (ADR-0002).")
    return _est.Estimate(
        theta_hat=np.array([v], dtype=np.float64),
        cov=np.array([[s * s]], dtype=np.float64),
        names=(name,),
        shrink=_est.Fixed(),
        support=(_est.Support.POSITIVE,),
        family=(_est.CIFamily.DEGENERATE if constant else _est.CIFamily.NORMAL,),
        kind=("pin" if constant else "declared_spread"),
    )


@contextlib.contextmanager
def logged_run(
    name: str,
    *,
    quantity: str,
    units: str,
    description: str,
    module_path: str,
    config: Optional[Mapping[str, Any]] = None,
    estimate: Optional["_est.Estimate"] = None,
) -> Iterator[Callable[..., None]]:
    """Open a measurement RUN for `name`: (1) register the definition (idempotent), (2) open an instance
    stamped with the repo git_sha + host + `config`, (3) if an `estimate` is given, persist it as the
    instance's `estimate` jsonb (the §5.1 SSOT of the measured object — the harmonized `Estimate`),
    (4) yield a `log(values, sample_size=None, seq=None)` callable the run() body calls with its raw-reading
    PROVENANCE rows. On normal exit the instance is populated; an exception propagates (ADR-0002 — a
    half-measured run is surfaced, the partial samples already committed stay as provenance of the attempt).

    The `estimate` is the §6 Phase-3 path: a migrated bench computes its `Estimate` in `measure()` and passes
    it here, and the raw `log(...)` rows become PROVENANCE only — the variance authority is the jsonb, so the
    headline scalar must NOT be re-logged as a sample row (the §5.2 de-dup obligation, which corrupts
    `latest_aggregate`'s count). Usage:

        # legacy (mean/median bench): no estimate yet, raw pool is the authority
        with logged_run(NAME, quantity=…, units=…, description=…, module_path=…, config={…}) as log:
            log(per_op_us_list, sample_size=iters)        # bulk readings
            log(single_reading)                            # one reading
        # Phase-3 (fit bench): the Estimate is the SSOT; log only raw-design-point provenance
        with logged_run(NAME, …, estimate=est) as log:
            log(per_width_medians, sample_size=iters)     # provenance (NOT the headline scalar)
    """
    def_id = register_quantity(
        name, quantity=quantity, units=units, description=description, module_path=module_path)
    inst_id = bench_store.open_instance(
        def_id, git_sha=repo_git_sha(), config=dict(config) if config else None)
    if estimate is not None:
        # The harmonized Estimate is the SSOT (§5.1); persist it onto the instance up-front so the jsonb is
        # present even if a later raw-provenance log() raises. Validated at construction (ADR-0002).
        bench_store.set_estimate(inst_id, estimate)

    def log(values: Any, sample_size: Optional[int] = None, seq: Optional[int] = None) -> None:
        if isinstance(values, (list, tuple)):
            bench_store.log_samples(inst_id, list(values), sample_size=sample_size)
        else:
            bench_store.log_sample(inst_id, float(values), seq=seq, sample_size=sample_size)

    yield log


# The recognized "how many units of work" keyword a bench's measure()/run() may expose so a caller can
# SIZE one call: the Neyman drive's _make_measurer passes the allocated budget through it, warm() below
# passes the burn-in count. A bench names its sizing knob ONE of these; the caller introspects measure()'s
# signature and uses the first match (None => no sizing knob, e.g. a pin). SINGLE HOME (ADR-0012 P1):
# untrusted_drive._ITERS_KW ALIASES this — the names are NOT re-listed anywhere else. `budget` is the
# drive's own canonical term for the lever (its measurer wrapper is `def measure(budget)`); `leaves` is the
# cpp-inproc tmsg bench's honest per-leaf-count knob. Both size a SHRINKABLE quantity; omitting them left
# those benches showing budget-kw "None" in the drive (shrinkable-but-un-sizable — silently de-funded).
SIZING_KWARGS = ("cycles", "trials", "iters", "n_trials", "reps", "rounds", "samples", "n",
                 "budget", "leaves")


def warm(mod: Any, **kwargs: Any) -> None:
    """The harness WARMUP PHASE for a registered bench, run ONCE before the measured phase so a cold
    transient (a first cold JAX forward at ~hundreds of us/row vs the ~few-us/row steady state; a cache
    fill; socket setup) never poisons the recorded samples. OPT-IN: a bench advertises EITHER a
    `warmup(**kwargs)` callable (its OWN warmup — the harness calls it and does NOT care what it does)
    OR a module-level `WARMUP` int (the harness runs `measure()` for that many discarded iterations, a
    generic burn-in). A bench advertising NEITHER gets no warmup phase. Everything a warmup produces is
    DISCARDED — never logged, never returned."""
    fn = getattr(mod, "warmup", None)
    if callable(fn):
        fn(**kwargs)
        return
    n = int(getattr(mod, "WARMUP", 0) or 0)
    if n > 0 and hasattr(mod, "measure"):
        import inspect
        params = inspect.signature(mod.measure).parameters
        iters_kw = next((k for k in SIZING_KWARGS if k in params), None)
        mod.measure(**({iters_kw: n} if iters_kw else {}))  # discarded — the generic burn-in
