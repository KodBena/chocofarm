"""
tools/analysis/leaf_eval_bound/untrusted_drive.py
===========================================

Test-drive the Neyman allocation loop with EVERY input fed LIVE from its benchmark — the
analog of the original neyman_driver demo (examples/demo_msgpass.py), except the synthetic
measurers are replaced by the real per-quantity benches and nothing is trusted/seeded: every
value the loop consumes comes from running a benchmark right now.

WHAT IT DOES (§6 Phase 4 — the harmonized `Estimate` path). For a chosen transport variant
(model_<slug>.py), it builds that model's `NeymanDriver` (its f + manifest-grounded costs),
wires one MEASURER per input — `measurer[i](budget)` runs input i's bench `measure()` and
returns the input's harmonized `Estimate` (the bench DECLARES it — §1; the driver
`set_estimate`s it directly) — and calls `driver.run(measurers=…, pilot, max_rounds)`. The loop
pilots every input off its live bench, reads each input's already-divided sampling variance off
its `Estimate.cov`, the eval point off `theta_hat`, the CI multiplier off `family`, allocates the
next batch to the inputs that most tighten the bound's CI, and re-measures — exactly the demo
loop, now driven by the real measurements through the typed contract.

NO COERCION (the §6 Phase-4 deletion). The pre-Phase-4 driver took a SAMPLER per input
(`sampler[i](k) -> raw array`) and a generic `_per_sample` heuristic guessed which list in a
bench's dict was the per-sample pool (the LONGEST numeric list). For a fit bench that grabbed the
ROW-COUNT x-axis (`batches=[32…512]`, ~224 mean) instead of the slope, so the driver multiplied
`df/dt_row ≈ −91.7` against ~224 and the bound CRATERED to a nonsense `E[f] ≈ 11.9` (the original
symptom). Under the contract there is NO guessing: the bench RETURNS its `Estimate` and the driver
consumes it (P2 reject-don't-guess). A pin is a `Fixed`/`DEGENERATE` `Estimate`, not a faked
2-sample zero-spread pool; a bench exposing no valid `Estimate` is a loud `is_valid()` failure
(ADR-0002).

NOT POSTGRES. It uses `measure()` (compute only), NOT `run()` (which would log to the metric
store) — so a test-drive never persists numbers into `control_research`. The manifest's
trusted-flag plumbing is unaffected; this is the loop mechanism, in memory.

TWO HONEST CAVEATS (this is a MECHANISM test, not a trustworthy measurement):
  1. SOLE WORKLOAD. The benches are timing-sensitive; run nothing else. Pin if you like
     (`taskset -c 0-3 ...`), but co-scheduling other work corrupts every reading.
  2. THE NUMBERS ARE CONFOUNDED. The current benches are Python; the cross-thread wakeup benches
     carry GIL handoff in the timing path (the bench's own number: ~34us GIL on a ~0.1-2us
     signal). So the bound this prints is NOT a faithful throughput floor — it is proof the
     LOOP runs end to end on live data. The native (C++) rebuild is what makes the values
     honest; this driver is unchanged by it (same samplers, real numbers).

Run (sole-workload):
    /home/bork/w/vdc/venvs/generic/bin/python tools/analysis/leaf_eval_bound/untrusted_drive.py [slug]
        slug in {zmq_baseline, shm_spin_poll, futex_wake, lockfree_mpsc, cpp_inproc_port}
        default: zmq_baseline
    Env knobs: UD_PILOT (pilot budget/input, default 32), UD_ROUNDS (max rounds, default 4),
               UD_ITERS_CAP (max iters/bench call, default 1500), UD_TOL (dps CI target, default 5).
    The per-input budget sizes each bench's measurement work (its cycles/trials/iters knob, capped at
    UD_ITERS_CAP); the bench returns its `Estimate` whatever the budget. Some inputs (t_row, B_op)
    measure the real JAX forward — expect a warm-up and a few minutes; lower UD_PILOT/UD_ITERS_CAP for
    a faster spin.

Public Domain (The Unlicense).
"""
from __future__ import annotations

import importlib
import inspect
import os
import sys
import time
from typing import Any, Callable, Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import estimate as _est  # noqa: E402  — the harmonized Estimate contract (the typed value measurers return)
import manifest  # noqa: E402  — the SSOT registry: name -> bench module_path
sys.path.insert(0, os.path.join(_HERE, "benchmarks"))
import harness  # noqa: E402  — the harness warmup phase (warm())
import leaf_eval_grounding as _G  # noqa: E402  — seed iota/t_row, to predict the slow JAX-fit ETAs

# The recognized "how many units" keyword a bench's measure() exposes so the loop can size the batch to
# the allocator's k by passing it. SINGLE HOME (ADR-0012 P1) is harness.SIZING_KWARGS — aliased here,
# never re-listed (the duplicate that left `budget`/`leaves`-named tmsg benches showing budget-kw "None").
_ITERS_KW = harness.SIZING_KWARGS

_VARIANTS = ("zmq_baseline", "shm_spin_poll", "futex_wake", "lockfree_mpsc", "cpp_inproc_port")


def _bench_module(qname: str) -> Any:
    """Import the bench module that OWNS quantity `qname`, via its registered definition's module_path
    (the manifest's SSOT resolution). Loud if the quantity is unregistered or its bench cannot load
    (ADR-0002 — a quantity with no live bench is a real gap for a loop that claims to feed off benches)."""
    defs = manifest.discover()
    d = defs.get(qname)
    if not d:
        raise KeyError(
            f"untrusted_drive: quantity {qname!r} is not registered — cannot feed the loop from a "
            f"bench that does not exist. (Is postgres up + the bench definition inserted?)")
    return manifest._import_bench_module(d["module_path"])  # noqa: SLF001 — the documented resolver


# --------------------------------------------------------------------------- #
# Progress + ETA. A loop step can be a long, mostly-silent wait — a JAX-fit bench at a big budget is
# MINUTES — so every measurement announces WHERE it is and, when it can, an expected duration. The
# estimate's source (in preference order): (1) EMPIRICAL — this bench's own prior wall-clock, scaled
# by the budget (the honest one, used once a bench has run); (2) for a slow fit bench's FIRST run, the
# current iota/t_row estimates × the fit's work shape (iters × repeat × Σ_widths(iota + t_row·width) —
# the "sourced from the quantities themselves" estimate); (3) none — a fast median/pin, timed live in ms.
# --------------------------------------------------------------------------- #
_FIT_QUANTITIES = frozenset({"t_row_us", "iota_us", "T_disp_us", "cpp_inproc_port_t_row_bare_us"})
_FIT_REPEAT = 30                                   # bench_t_row/iota/t_disp measure() default `repeat`
_FIT_WIDTHS = (32, 64, 128, 192, 256, 384, 512)    # the staged-forward batch sweep each fit times

_TIMINGS: dict[str, tuple[int, float]] = {}        # qname -> (last budget, last wall-clock seconds)
_T0 = 0.0                                           # the loop wall-clock origin (set at the start of main())


def _fmt_dur(s: float) -> str:
    if s < 1.0:
        return f"{s * 1000:.0f}ms"
    if s < 90.0:
        return f"{s:.1f}s"
    if s < 5400.0:
        return f"{s / 60:.1f}min"
    return f"{s / 3600:.1f}h"


def _eta_seconds(qname: str, budget: int) -> Optional[float]:
    """Predicted seconds for one `measure(budget)` call (None if a fast bench with no prior — just
    timed live). Empirical (a prior timing scaled by the budget) is preferred; for a slow JAX-fit
    bench's first run it falls back to the current iota/t_row estimates × the fit's work shape."""
    prior = _TIMINGS.get(qname)
    if prior is not None:
        pb, pt = prior
        return pt * (budget / pb) if pb > 0 else pt
    if qname in _FIT_QUANTITIES:
        try:
            iota = float(_G.SERVE_INTERCEPT_US.mean)
            trow = float(_G.SERVE_SLOPE_US.mean)
        except Exception:                          # grounding unavailable -> the v1 seed fit
            iota, trow = 94.58, 4.317
        per_iter_us = _FIT_REPEAT * sum(iota + trow * w for w in _FIT_WIDTHS)
        return budget * per_iter_us / 1e6


def _announce(qname: str, n: int, unit: Optional[str], eta: Optional[float]) -> None:
    slow = "  [SLOW — JAX fit]" if qname in _FIT_QUANTITIES else ""
    est = f" — est ~{_fmt_dur(eta)}" if eta is not None else " — timing live (fast)"
    print(f"  > benchmarking {qname} [{n} {unit or 'calls'}]{slow}{est} …", flush=True)


def _done(qname: str, n: int, dt: float) -> None:
    _TIMINGS[qname] = (n, dt)
    total = (time.perf_counter() - _T0) if _T0 else dt
    print(f"    done: {qname} in {_fmt_dur(dt)}   ·   total elapsed {_fmt_dur(total)}", flush=True)


def _make_measurer(qname: str, iters_cap: int) -> Callable[[int], "_est.Estimate"]:
    """A driver MEASURER for one input (the §6 Phase-4 contract): `measure(budget)` runs the bench's
    `measure()` (compute only, no postgres) sized toward `budget` units of work (capped at iters_cap so a
    big allocation does not hang a test-drive) and returns the input's harmonized `Estimate` — the typed
    contract the driver `set_estimate`s directly. NO `_per_sample` heuristic, NO 2-sample pad: the bench
    DECLARES its `Estimate` (P2 reject-don't-guess), so there is nothing to guess. The returned value MUST
    be an `Estimate` and MUST pass `is_valid()` — a bench that returns anything else, or an invalid
    estimate, is a loud failure at this seam (ADR-0002), never silently coerced into a plausible-looking
    bound (the cratered `E[f]≈11.9` that motivated this phase)."""
    mod = _bench_module(qname)
    harness.warm(mod)   # run the bench's advertised warmup phase ONCE (opt-in; no-op if none)
    fn = getattr(mod, "measure", None)
    if fn is None:
        raise AttributeError(
            f"untrusted_drive: bench for {qname!r} exposes no measure() — the §6 Phase-4 contract is that "
            f"measure() returns the input's Estimate (run() persists to postgres and is not the test-drive "
            f"path). A bench without measure() cannot feed the un-trusted loop (ADR-0002).")
    params = inspect.signature(fn).parameters
    iters_kw = next((k for k in _ITERS_KW if k in params), None)

    def measure(budget: int) -> "_est.Estimate":
        n = max(2, min(int(budget), iters_cap))
        kwargs = {iters_kw: n} if iters_kw else {}
        _announce(qname, n, iters_kw, _eta_seconds(qname, n))
        _t0 = time.perf_counter()
        est = fn(**kwargs)        # the (possibly long) live measurement — the WHERE was announced above
        _done(qname, n, time.perf_counter() - _t0)
        if not isinstance(est, _est.Estimate):
            raise TypeError(
                f"untrusted_drive: bench {qname!r} measure() returned {type(est).__name__}, not an "
                f"estimate.Estimate — the §6 Phase-4 contract is a typed Estimate, never a bespoke dict the "
                f"driver must guess a pool out of (ADR-0002 / P2 reject-don't-guess).")
        if not est.is_valid():
            raise ValueError(
                f"untrusted_drive: bench {qname!r} measure() returned an Estimate that fails is_valid() — "
                f"a bench with no defensible Estimate is a loud failure, never a padded pool (ADR-0002).")
        return est

    measure._meta = (qname, getattr(mod, "NAME", qname), iters_kw)  # for the banner
    return measure


def main() -> int:
    slug = sys.argv[1] if len(sys.argv) > 1 else "zmq_baseline"
    if slug not in _VARIANTS:
        print(f"untrusted_drive: unknown slug {slug!r}; choose one of {_VARIANTS}", file=sys.stderr)
        return 2
    pilot = int(os.environ.get("UD_PILOT", "32"))
    rounds = int(os.environ.get("UD_ROUNDS", "4"))
    iters_cap = int(os.environ.get("UD_ITERS_CAP", "1500"))
    tol = float(os.environ.get("UD_TOL", "5.0"))

    model = importlib.import_module(f"model_{slug}")
    driver, x0 = model.build_driver(tolerance=tol, trust=True)
    names = list(model.INPUT_NAMES)
    measurers = {i: _make_measurer(model.registry_qname(nm), iters_cap) for i, nm in enumerate(names)}

    print("=" * 88)
    print(f"UNTRUSTED LIVE-BENCH NEYMAN DRIVE — model_{slug}  (pilot={pilot}, rounds={rounds}, "
          f"iters_cap={iters_cap}, tol={tol} dps)")
    print("=" * 88)
    print("MECHANISM TEST — every input is fed LIVE from its bench as an Estimate; NOTHING is trusted/seeded.")
    print("  ! sole-workload only (timing-sensitive benches)")
    print("  ! numbers are CONFOUNDED (Python/GIL substrate); this proves the LOOP runs, not the floor")
    print("  ! measure() path = in-memory, no postgres write; the bench DECLARES its Estimate (no coercion)\n")
    print(f"  {'input':<8}{'registry quantity':<26}{'budget-kw':<12}")
    print("  " + "-" * 48)
    for i, nm in enumerate(names):
        q, bench_nm, itk = measurers[i]._meta
        print(f"  {nm:<8}{q:<26}{str(itk):<12}")
    print()

    global _T0
    _T0 = time.perf_counter()
    _fe = _eta_seconds("t_row_us", max(2, min(pilot, iters_cap)))
    print("  PROGRESS — each measurement announces below as it runs. The JAX-fit benches are the slow")
    print(f"  ones (~{_fmt_dur(_fe) if _fe else '?'} each at this budget, ∝ iters); the conflation fix measures a floored")
    print("  fit ONCE (in the pilot) then de-funds it, so the rounds are the fast median benches.\n")

    final = driver.run(measurers=measurers, pilot=pilot, max_rounds=rounds, verbose=True)

    if _TIMINGS:
        print(f"\n  TIMING — total {_fmt_dur(time.perf_counter() - _T0)} across {len(_TIMINGS)} benches this run "
              "(slowest first):")
        for _q, (_b, _t) in sorted(_TIMINGS.items(), key=lambda kv: -kv[1][1]):
            print(f"    {_q:<30} {_fmt_dur(_t):>9}   ({_b} units)")

    print("=" * 88)
    print(f"FINAL  E[f] = {final.estimate:.1f} dps   CI half-width = {final.ci_halfwidth:.2f} dps "
          f"(target {final.target_halfwidth:.2f})   converged={final.converged}")
    print("  Every value above was measured live this run (no seed, no postgres). The loop works;")
    print("  the NUMBER is confounded until the benches are native — that is the next step, not this one.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
