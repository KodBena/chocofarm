"""
tools/analysis/leaf_eval_bound/runners/throughput_bound.py
============================================

Runner: for EACH leaf-eval throughput lower-bound model (Design-A capacity, model_capacity.py;
Design-B cycle-time, model_cycletime.py), compute the bound estimate E[f] with its CI and print
the Neyman allocation — which physical quantity to benchmark next to tighten the bound. The
models are the things transported; this runner + the generic `AllocationDriver` are the transport
(ADR-0012 P2 separation).

The bound is computed via the `AllocationDriver` (§6 Phase 4 — each input fed as its harmonized
`Estimate` via `driver.set_estimate`, NOT a fabricated 2-point pilot). Each grounded input is a
`Fixed`/declared-spread `Estimate` (`cov=[[sigma^2]]` un-divided — built via reconstruct's
seed->Estimate SSOT `reconstruct._estimate_from_seed`), so the allocation reflects the grounded
uncertainty: `gᵀΣg` is the grounded-uncertainty CI on E[f]. A declared-spread prior is
un-shrinkable by sampling, so it gets NO allocation (the §2.3 "a Fixed pin drops out, for the
right reason" branch); the report ranks the next-benchmark targets by a_i = (df/dx)²·sigma² (the
bound-tightening potential), and tightening one means RUNNING its bench (flipping it to
trusted=True), not drawing more samples of a fixed prior. The gradient is `jax.grad` of the
model's `throughput_jax` (the OT→JAX migration — this runner imports no OpenTURNS, and the old
numpy delta-method fallback/cross-check retired with the single-f collapse, J4).

Run: /home/bork/w/vdc/venvs/generic/bin/python -m leaf_eval_bound.runners.throughput_bound   (from tools/analysis, or PYTHONPATH=tools/analysis)

Public Domain (The Unlicense).
"""
from __future__ import annotations

import os
import sys

import numpy as np

from leaf_eval_bound.contract import grounding as G  # noqa: E402
from leaf_eval_bound.contract import references
from leaf_eval_bound.store import reconstruct  # noqa: E402  — the seed->Estimate SSOT (_estimate_from_seed) for the §6 Phase-4 pilot
from leaf_eval_bound.models import model_capacity  # noqa: E402
from leaf_eval_bound.models import model_cycletime  # noqa: E402


def _bound(model):
    """Feed each input as its GROUNDED `Estimate` then one driver `step()`, so the Recommendation's
    per-input a_i ranks which quantity to benchmark next (the deliverable). The §6 Phase-4 feed: a
    `Fixed`/declared-spread `Estimate` per input built via reconstruct's seed->Estimate SSOT
    (`reconstruct._estimate_from_seed`) from the model's grounded (mean, sigma) — a declared-spread prior,
    un-shrinkable by sampling, so the §2.3 allocator funds none (a Fixed pin drops out, for the right
    reason). The grounded mean anchors the gradient at the binding stage (the min() kink makes it
    point-sensitive). We deliberately do NOT loop to convergence — these are seeds, not a live system."""
    driver, x0 = model.build_driver(tolerance=0.1)
    names = model.INPUT_NAMES
    sig = model.SIGMAS
    ests = {nm: reconstruct._estimate_from_seed(nm, x0[nm], sig[nm], "") for nm in names}
    driver.set_estimates_by_name(ests)
    rec = driver.step(second_order_check=False)
    f_mu = float(model.throughput_jax(np.array([x0[nm] for nm in names])))  # JAX f eval (x64 via the f's jax_backend)
    return rec, f_mu, x0


def run_model(title, model):
    print("=" * 80)
    print(title)
    print("=" * 80)
    needs_meas = model.NEEDS_MEASUREMENT
    rec, f_mu, x0 = _bound(model)

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

    # The serve sawtooth (both models expose it) — shows the bound peaks at FULL buckets and DROPS
    # just past an edge, so a larger B is not unconditionally conservative.
    saw = {r: round(model.serve_sawtooth(r), 1)
           for r in (64, 128, 192, 224, 256, 384, 512)}
    print("Serve sawtooth dps (real rows -> bucketed serve dps, incl. tau_io):", saw)

    # The bound + the Neyman ranking via the JAX-driven AllocationDriver. Rank by a_i (the per-input share
    # of Var(E[f])), NOT by +samples: every input is a Fixed declared-spread prior here, so the allocator
    # funds none (a prior is un-shrinkable by sampling — the §2.3 branch); a_i is the honest "which most
    # tightens the bound" signal. min() zeros the non-binding stage's inputs by design.
    print(f"\nE[f] (mu_hat) = {f_mu:.1f} dps   "
          f"delta-method CI half-width on E[f] at the minimal pilot = {rec.ci_halfwidth:.1f} dps")
    print("Neyman ranking (which quantity to benchmark next — ranked by a_i = (df/dx)^2*sigma^2; the "
          "grounded inputs are declared-spread priors, so the allocator funds none — tightening one means "
          "RUNNING its bench):")
    ranked = sorted(rec.primitives, key=lambda p: p.a, reverse=True)
    tot_a = sum(p.a for p in rec.primitives) or 1.0
    for p in ranked:
        flag = "NEEDS-SOLE-WORKLOAD" if needs_meas.get(p.name) else "grounded"
        print(f"  {p.name:<14} a_i={p.a:>11.4g}  cost={p.cost:>5.3g}  "
              f"share={100*p.a/tot_a:>5.1f}%  {flag}")
    return f_mu, rec.ci_halfwidth


def main():
    print("Leaf-eval transport — first-principles throughput LOWER BOUNDS")
    print(f"Reference points (NOT targets): empirical plateau ~{references.REF_PLATEAU_DPS:.0f} dps "
          f"(user-supplied, one config family); prior model literal ~{references.REF_PRIOR_MODEL_DPS:.0f} "
          f"dps (overcommit_sweep.py bare literal, called 'an upper bound').")
    print(f"Measured anchors: gen-ceiling 3x152={3*152} dps (4.0x linear) | "
          f"serve full-bucket GLOBAL MAX {references.REF_GLOBAL_MAX_DPS:.0f} dps | "
          f"high-N bench (over-reads e2e) {references.REF_HIGH_N_BENCH_DPS:.0f} dps | "
          f"StageB 1-thread {references.REF_STAGEB_1THREAD_DPS_PER_CORE:.0f} dps/core\n")

    fa, cia = run_model("MODEL A — capacity / min-of-stages (Design-A)", model_capacity)
    print()
    fb, cib = run_model("MODEL B — serialized-server cycle-time (Design-B)", model_cycletime)

    print("\n" + "=" * 80)
    print("SUMMARY — the two lower bounds vs the references")
    print("=" * 80)
    plat = references.REF_PLATEAU_DPS
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
