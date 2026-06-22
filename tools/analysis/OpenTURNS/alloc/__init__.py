"""
tools/analysis/OpenTURNS/alloc/__init__.py
==========================================

`alloc` — the generic, model-agnostic optimal-allocation engine (Band-2 OR-general) for
the leaf-eval throughput-bound tool. The responsibility-refactor
(`docs/design/leaf-eval-bound-responsibility-refactor.md` §3) groups the allocation
machinery under this package; this is its FIRST increment — the two pure sub-modules
lifted out of the `neyman_driver.py` god-object:

  * `kink`     — the Clark-1961 min()-kink closed-form moments (§2.3-D / §3 move 4):
                 pure, unit-testable on synthetic arm covariances, INDEPENDENT of the
                 autodiff backend (the planned OpenTURNS→JAX swap cannot perturb it).
  * `gradient` — the gradient-backend seam (§5): OpenTURNS analytic `f.gradient()` with a
                 finite-difference fallback today; the ONE site the JAX swap replaces.

`neyman_driver.py` (the driver — to become `alloc/driver.py` in a later increment of the
refactor) imports these. The driver owns NO model (ADR-0012 P1/P2 — a model injects its
`arms_fn` onto the driver; `alloc` never imports `models`), so this package sits strictly
above `estimate` (the contract) and below the models and runners — the clean import DAG of
§3 (`alloc/` imports the contract only; nothing in `alloc/` imports a model or a runner).

Public Domain (The Unlicense).
"""
