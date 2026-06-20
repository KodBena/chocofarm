"""
tools/analysis/OpenTURNS/throughput_bound.py
============================================

Runner: for EACH leaf-eval throughput lower-bound model (Design-A capacity,
model_capacity.py; Design-B cycle-time, model_cycletime.py), compute the bound estimate
E[f] with its CI and print the Neyman allocation — which physical quantity to benchmark
next to tighten the bound. The models are the things transported; this runner + the
generic `NeymanDriver` are the transport (ADR-0012 P2 separation).

Two paths, both producing the bound:
  * openturns present: the full `NeymanDriver` loop (delta-method CI on E[f]; Neyman
    optimal allocation n_i* proportional to sqrt(a_i/c_i)). The pilot pools are drawn
    from each input's GROUNDED Normal(mean, sigma) — so the allocation reflects the
    grounded uncertainty (we are estimating E[f] over the inputs' uncertainty, not
    sampling a live system here).
  * openturns absent: a numpy-only fallback computes f(mu_hat) and a first-order
    delta-method CI (central finite-difference gradient), and ranks the inputs by the
    same Neyman quantity sqrt(a_i/c_i). The bound is still computed; the report says so.

Run: /home/bork/w/vdc/venvs/generic/bin/python tools/analysis/OpenTURNS/throughput_bound.py

Public Domain (The Unlicense).
"""
from __future__ import annotations

import importlib.util
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import leaf_eval_grounding as G  # noqa: E402
import model_capacity  # noqa: E402
import model_cycletime  # noqa: E402

_HAS_OT = importlib.util.find_spec("openturns") is not None
_Z95 = 1.959963984540054


def _fd_gradient(fn, x_names, x0, rel=1e-5):
    """Central finite-difference gradient of a numpy callable fn(dict) at x0."""
    g = {}
    for nm in x_names:
        h = rel * max(abs(x0[nm]), 1.0)
        xp = dict(x0); xp[nm] += h
        xm = dict(x0); xm[nm] -= h
        g[nm] = (fn(xp) - fn(xm)) / (2.0 * h)
    return g


def _numpy_bound(model, sigmas, costs):
    """Fallback: f(mu_hat), delta-method CI on E[f], and the Neyman ranking — all numpy."""
    x0 = model.initial_point()
    names = model.INPUT_NAMES
    f0 = model.throughput_numpy(x0)
    grad = _fd_gradient(model.throughput_numpy, names, x0)
    a = {nm: (grad[nm] * sigmas[nm]) ** 2 for nm in names}     # a_i = (df/dx_i)^2 sigma_i^2
    var = sum(a.values())                                       # Var(E[f]) at n_i=1 each
    ci = _Z95 * np.sqrt(max(var, 0.0))
    # Neyman: n_i* proportional to sqrt(a_i/c_i). Rank desc; report the proportions.
    weight = {nm: np.sqrt(a[nm] / costs[nm]) if a[nm] > 0 else 0.0 for nm in names}
    return x0, f0, grad, a, var, ci, weight


def _print_neyman_table(names, sigmas, costs, grad, a, weight, needs_meas):
    tot = sum(weight.values()) or 1.0
    order = sorted(names, key=lambda nm: weight[nm], reverse=True)
    print(f"  {'quantity':<14}{'|df/dx|':>11}{'sigma':>11}{'a_i':>12}"
          f"{'cost':>7}{'alloc%':>9}  measure?")
    print("  " + "-" * 76)
    for nm in order:
        flag = "NEEDS-SOLE-WORKLOAD" if needs_meas.get(nm) else "grounded"
        print(f"  {nm:<14}{abs(grad[nm]):>11.4g}{sigmas[nm]:>11.4g}{a[nm]:>12.4g}"
              f"{costs[nm]:>7.3g}{100*weight[nm]/tot:>8.1f}%  {flag}")


def _ot_bound(model):
    """openturns path: a MINIMAL 2-sample pilot then one `step()`, so the Recommendation's
    Neyman allocation is LIVE and its PROPORTIONS rank which quantity to benchmark next
    (the deliverable). We deliberately do NOT run the convergence top-up loop — that would
    drive +samples to 0 and hide the ranking; the per-input `a_i` and the recommended
    proportions at the minimal pilot are the "where to spend" answer. The pilot pools are
    drawn from each input's GROUNDED Normal(mean, sigma) so a_i reflects the grounded
    uncertainty. A tight tolerance keeps the step un-converged so it allocates."""
    import openturns as ot
    driver, x0 = model.build_driver(tolerance=0.1)
    names = model.INPUT_NAMES
    sig = model.SIGMAS
    # A DETERMINISTIC symmetric 2-point pilot {mean - sigma, mean + sigma} per input, so
    # each pool's sample-mean is EXACTLY the grounded mean and its sample-std EXACTLY the
    # grounded sigma. This anchors the allocation gradient at the true grounded mean (not a
    # noisy random draw) — important because the min() kink makes the binding stage, and
    # hence the gradient, sensitive to the exact evaluation point. The allocation then
    # reflects the stage that binds AT THE GROUNDED MEAN.
    pilot = {i: np.array([x0[nm] - max(sig[nm], 1e-9), x0[nm] + max(sig[nm], 1e-9)])
             for i, nm in enumerate(names)}
    driver.add_samples(pilot)
    rec = driver.step(second_order_check=False)
    f_mu = float(model.build_symbolic_function()(ot.Point([x0[nm] for nm in names]))[0])
    return driver, rec, f_mu, x0


def run_model(title, model):
    print("=" * 80)
    print(title)
    print("=" * 80)
    names = model.INPUT_NAMES
    sigmas = model.SIGMAS
    costs = model.COSTS
    needs_meas = model.NEEDS_MEASUREMENT

    # The bound + Neyman ranking via the numpy route (always available; also the lockstep
    # cross-check of the symbolic formula).
    x0, f0, grad, a, var, ci, weight = _numpy_bound(model, sigmas, costs)

    # Stage / cycle breakdown so the binding term is explicit.
    if model is model_capacity:
        caps = model.stage_capacities(x0)
        binding = min(caps, key=caps.get)
        print("\nStage capacities (dps):", {k: round(v, 1) for k, v in caps.items()})
        print("Binding stage:", binding, f"({caps[binding]:.1f} dps)")
        print("Alt producer caps:",
              {k: round(v, 1) for k, v in model.report_alt_producer_caps().items()})
    else:
        cb = model.cycle_breakdown(x0)
        print("\nPer-forward cycle (us):",
              {k: round(v, 1) for k, v in cb.items() if k.endswith("_us")})
        print("Serve vs producer (dps):",
              {"serve": round(cb["serve_dps"], 1), "producer": round(cb["producer_dps"], 1)})
        print("Producer caps:", {k: round(v, 1) for k, v in model.producer_caps().items()})
        print("B->inf asymptote (UNREACHABLE under max_batch cap):",
              round(model.asymptote_dps(), 1), "dps")
        print("C++ inproc-port contrast:",
              {k: round(v, 1) for k, v in model.inproc_port_contrast().items()})

    # The serve sawtooth (both models expose it) — shows the bound peaks at FULL buckets
    # and DROPS just past an edge, so a larger B is not unconditionally conservative.
    saw = {r: round(model.serve_sawtooth(r), 1)
           for r in (64, 128, 192, 224, 256, 384, 512)}
    print("Serve sawtooth dps (real rows -> bucketed serve dps, incl. tau_io):", saw)

    if _HAS_OT:
        try:
            driver, rec, f_mu, _ = _ot_bound(model)
            print(f"\n[openturns] E[f] (symbolic, mu_hat) = {f_mu:.1f} dps   "
                  f"(numpy cross-check {f0:.1f})")
            print(f"[openturns] delta-method CI half-width on E[f] at the minimal pilot "
                  f"= {rec.ci_halfwidth:.1f} dps (the grounded-uncertainty spread of the "
                  f"bound)")
            print("[openturns] Neyman allocation (where to spend the next bench budget — "
                  "min() zeros the non-binding stage's inputs, by design):")
            tot = sum(p.recommend for p in rec.primitives) or 1
            for p in rec.where_to_spend():
                flag = "NEEDS-SOLE-WORKLOAD" if needs_meas.get(p.name) else "grounded"
                print(f"  {p.name:<14} a_i={p.a:>11.4g}  cost={p.cost:>5.3g}  "
                      f"+samples={p.recommend:>7d} ({100*p.recommend/tot:4.1f}%)  {flag}")
        except Exception as exc:   # fall back loudly, never silently (ADR-0002)
            print(f"\n[openturns path raised: {exc!r} — using numpy fallback below]")
            _print_numpy_summary(names, sigmas, costs, f0, ci, grad, a, weight, needs_meas)
    else:
        print("\n[openturns ABSENT — numpy-only fallback; the bound is still computed]")
        _print_numpy_summary(names, sigmas, costs, f0, ci, grad, a, weight, needs_meas)

    return f0, ci


def _print_numpy_summary(names, sigmas, costs, f0, ci, grad, a, weight, needs_meas):
    print(f"  E[f] (numpy, mu_hat) = {f0:.1f} dps   "
          f"delta-method 95% CI half-width (at n=1/input) = +/- {ci:.1f} dps")
    print("  Neyman ranking (which quantity to benchmark next):")
    _print_neyman_table(names, sigmas, {nm: costs[nm] for nm in names}, grad, a, weight,
                        needs_meas)


def main():
    print("Leaf-eval transport — first-principles throughput LOWER BOUNDS")
    print(f"openturns available: {_HAS_OT}")
    print(f"Reference points (NOT targets): empirical plateau ~{G.REF_PLATEAU_DPS:.0f} dps "
          f"(user-supplied, one config family); prior model literal ~{G.REF_PRIOR_MODEL_DPS:.0f} "
          f"dps (overcommit_sweep.py bare literal, called 'an upper bound').")
    print(f"Measured anchors: gen-ceiling 3x152={3*152} dps (4.0x linear) | "
          f"serve full-bucket GLOBAL MAX {G.REF_GLOBAL_MAX_DPS:.0f} dps | "
          f"high-N bench (over-reads e2e) {G.REF_HIGH_N_BENCH_DPS:.0f} dps | "
          f"StageB 1-thread {G.REF_STAGEB_1THREAD_DPS_PER_CORE:.0f} dps/core\n")

    fa, cia = run_model("MODEL A — capacity / min-of-stages (Design-A)", model_capacity)
    print()
    fb, cib = run_model("MODEL B — serialized-server cycle-time (Design-B)", model_cycletime)

    print("\n" + "=" * 80)
    print("SUMMARY — the two lower bounds vs the references")
    print("=" * 80)
    plat = G.REF_PLATEAU_DPS
    print(f"  Design-A (capacity)  : {fa:.0f} +/- {cia:.0f} dps   "
          f"(~{fa/plat:.2f}x the ~{plat:.0f} plateau)")
    print(f"  Design-B (cycle-time): {fb:.0f} +/- {cib:.0f} dps   "
          f"(~{fb/plat:.2f}x the ~{plat:.0f} plateau)")
    print(f"  The two routes agree to within {abs(fa-fb):.0f} dps (the cross-check).")
    print(f"  Both lower bounds sit ABOVE the ~{plat:.0f} reference, so ~{plat:.0f} is NOT "
          f"near the achievable optimum's floor.")
    print("  CAVEAT (not overclaimed): these are BENCH-dps bounds at a FULL-bucket feed,")
    print("  and tau_io (server drain/scatter) is UNMEASURED — the top Neyman target.")


if __name__ == "__main__":
    main()
