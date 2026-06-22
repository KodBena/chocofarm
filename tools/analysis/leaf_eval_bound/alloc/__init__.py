"""
tools/analysis/leaf_eval_bound/alloc/__init__.py
==========================================

`alloc` — the generic, model-agnostic optimal-allocation engine (Band-2 OR-general) for
the leaf-eval throughput-bound tool. The responsibility-refactor
(`docs/design/leaf-eval-bound-responsibility-refactor.md` §3) groups the allocation
machinery under this package:

  * `driver`   — the `AllocationDriver`: the cost-constrained c-optimal allocation engine
                 (solved as a SOCP, §2.3). Strict Neyman `n_i ∝ √(a_i/c_i)` is its diagonal
                 special case; the Clark-1961 min()-kink path is no part of Neyman at all. The
                 §4 rename `neyman_driver.py` → `alloc/driver.py` has landed — the engine is
                 named for its responsibility (allocation), not the one branch (ADR-0008).
  * `kink`     — the Clark-1961 min()-kink closed-form moments (§2.3-D / §3 move 4):
                 pure, unit-testable on synthetic arm covariances, INDEPENDENT of the
                 autodiff backend.
  * `gradient` — the gradient-backend seam (§5): the JAX `jax.grad` backend. The OpenTURNS→JAX
                 swap is done — `alloc.gradient` imports no OpenTURNS; it is the ONE site that
                 backend lives at.
  * `jax_backend` — the x64-enabled JAX handle (`jnp`) the driver evaluates `f` at a point with.

`driver` owns NO model (ADR-0012 P1/P2 — a model injects its `arms_fn` onto the driver;
`alloc` never imports `models`), so this package sits strictly above `estimate` (the contract)
and below the models and runners — the clean import DAG of §3 (`alloc/` imports the contract
only; nothing in `alloc/` imports a model or a runner).

Public Domain (The Unlicense).
"""
