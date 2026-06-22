"""
tools/analysis/leaf_eval_bound/alloc/jax_backend.py
=============================================

The x64-enabled JAX backend handle for the leaf-eval bound tool (the OpenTURNS→JAX
migration; `docs/design/leaf-eval-bound-responsibility-refactor.md` §5). The tool is float64
THROUGHOUT — its bound, its grounded constants, its `Estimate` covariances — but jax
defaults to float32, which would drift both the evaluated bound and the gradient ~1e-6 from
the numpy/OT path (and break the byte-for-byte equivalence the migration rests on). So this
module enables x64 ONCE, on import, BEFORE the first jnp trace, and re-exports the `jnp` +
`grad` handles every model `f` and the gradient seam use. Import jnp from HERE, never
`jax.numpy` directly, so x64 is guaranteed enabled first.

(The tool runs as its own analysis process; this process-global flag does not touch the
`az/` training stack, which runs in a separate process and chooses its own precision.)

Public Domain (The Unlicense).
"""
from __future__ import annotations

import jax

# jax.config.update is unstubbed in jax's types; the targeted ignore below keeps this module --strict clean.
jax.config.update("jax_enable_x64", True)  # type: ignore[no-untyped-call]

import jax.numpy as jnp  # noqa: E402  — imported AFTER x64 is enabled, so jnp is float64 by default

grad = jax.grad
