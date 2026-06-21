"""
tools/analysis/OpenTURNS/benchmarks/bench_common.py
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
        iters_kw = next((k for k in ("cycles", "trials", "iters", "n_trials", "reps", "rounds",
                                     "samples", "n") if k in params), None)
        mod.measure(**({iters_kw: n} if iters_kw else {}))  # discarded — the generic burn-in
