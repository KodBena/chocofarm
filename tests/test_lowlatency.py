#!/usr/bin/env python3
"""
tests/test_lowlatency.py — pins for the phantom-typed low-overhead JAX dispatcher
(chocofarm/az/lowlatency.py).

Always-on (no redis, no socket, no network) — runs in every `pytest tests/ -q`, but SKIPS GRACEFULLY
if jax is unimportable in the interpreter (mirroring test_cpp_runner.py / test_zmq_inference.py's
skip-without-its-dependency posture), so the default suite stays green on a box without jax.

What is pinned:
  * EQUIVALENCE (the feasibility gate's correctness half): the low-overhead call returns the SAME
    result as plain `jax.jit(fn)(params, x)` (allclose) — for BOTH the robust AOT `Compiled` path AND
    the unsafe loaded-executable path, single- and multi-row, value-only and two-output fns.
  * SSOT FAIL-LOUD (ADR-0002 / ADR-0012 P2 translate-and-validate): the constructor rejects a
    non-array call-arg leaf at construction; a per-call shape/dtype mismatch raises loudly on the
    robust path; `prefer_unsafe=True` with an unbuildable unsafe path refuses (never silently
    downgrades); `run(unsafe=True)` on a handle without the unsafe plumbing refuses.
  * IMMUTABILITY (ADR-0012 P9 — the handle is a value): the frozen dataclass rejects attribute set.
  * PHANTOM TYPE (zero runtime cost): the type parameters live only in `__class_getitem__` /
    annotations; no `In`/`Out` value is stored on the handle (the metadata fields are the avals only).

The mypy --strict cleanliness of the module is enforced separately by tests/test_mypy_strict.py once
the module is added to its STRICT_CLEAN set (the gate, ADR-0011); this file pins runtime behavior.

Public Domain (The Unlicense).
"""
import dataclasses
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _jax_available() -> bool:
    try:
        import jax  # noqa: F401
        return True
    except ImportError:
        return False


pytestmark = pytest.mark.skipif(not _jax_available(),
                                reason="jax not importable in this interpreter")


# --- toy dispatch-bound fns (the regime the library targets) ------------------------------------
def _mlp_value(p, x):  # type: ignore[no-untyped-def]
    """A 2-layer MLP value head — `(params, x) -> (B, 1)`. A single-output fn (the inference shape)."""
    import jax.numpy as jnp
    a1 = jnp.maximum(x @ p["W1"] + p["b1"], 0.0)
    a2 = jnp.maximum(a1 @ p["W2"] + p["b2"], 0.0)
    return a2 @ p["Wv"] + p["bv"]


def _mlp_value_policy(p, x):  # type: ignore[no-untyped-def]
    """A 2-layer MLP with value AND logits heads — `(params, x) -> (value, logits)`. A two-output fn
    (the multi-leaf-output path: the robust call returns the pytree, not a bare array)."""
    import jax.numpy as jnp
    a1 = jnp.maximum(x @ p["W1"] + p["b1"], 0.0)
    a2 = jnp.maximum(a1 @ p["W2"] + p["b2"], 0.0)
    v = a2 @ p["Wv"] + p["bv"]
    logits = a2 @ p["Wp"] + p["bp"]
    return v, logits


def _toy_params(in_dim: int, hidden: int, n_actions: int | None, seed: int = 0):  # type: ignore[no-untyped-def]
    import jax.numpy as jnp
    rng = np.random.default_rng(seed)

    def mk(a: int, b: int, s: float = 1.0):  # type: ignore[no-untyped-def]
        return jnp.asarray((rng.standard_normal((a, b)) * s).astype(np.float32))
    p = {"W1": mk(in_dim, hidden), "b1": jnp.zeros(hidden, jnp.float32),
         "W2": mk(hidden, hidden), "b2": jnp.zeros(hidden, jnp.float32),
         "Wv": mk(hidden, 1, 0.1), "bv": jnp.zeros(1, jnp.float32)}
    if n_actions is not None:
        p["Wp"] = mk(hidden, n_actions, 0.1)
        p["bp"] = jnp.zeros(n_actions, jnp.float32)
    return p


# ===========================================================================
# EQUIVALENCE — the low-overhead call matches plain jit(fn)(x)
# ===========================================================================
@pytest.mark.parametrize("B", [1, 4, 16])
@pytest.mark.parametrize("in_dim,hidden", [(8, 16), (80, 64)])
def test_robust_matches_plain_jit_value(B, in_dim, hidden):
    """The robust AOT path == `jax.jit(fn)(params, x)` (allclose), value-only fn, across batch sizes."""
    import jax
    import jax.numpy as jnp
    from chocofarm.az.lowlatency import compile_lowlatency, run
    p = _toy_params(in_dim, hidden, None)
    x = np.random.default_rng(B + in_dim).standard_normal((B, in_dim)).astype(np.float32)
    ref = np.asarray(jax.jit(_mlp_value)(p, jnp.asarray(x)))
    h = compile_lowlatency(_mlp_value, p, x)
    got = np.asarray(run(h, x))
    assert got.shape == ref.shape
    assert np.allclose(got, ref, atol=1e-5), f"max|d|={np.abs(got - ref).max()}"
    # __call__ is the same as run on a robust-default handle
    assert np.allclose(np.asarray(h(x)), ref, atol=1e-5)


@pytest.mark.parametrize("B", [1, 8])
def test_unsafe_matches_plain_jit_value(B):
    """The unsafe loaded-executable path == `jax.jit(fn)(params, x)` (allclose). The unsafe path is the
    measured low-level call; it must still be NUMERICALLY identical to the robust/plain path (it runs
    the SAME compiled executable — only the Python dispatch wrapper differs)."""
    import jax
    import jax.numpy as jnp
    from chocofarm.az.lowlatency import compile_lowlatency, run
    p = _toy_params(32, 48, None)
    x = np.random.default_rng(B).standard_normal((B, 32)).astype(np.float32)
    ref = np.asarray(jax.jit(_mlp_value)(p, jnp.asarray(x)))
    hu = compile_lowlatency(_mlp_value, p, x, prefer_unsafe=True)
    assert hu.use_unsafe is True
    assert np.allclose(np.asarray(run(hu, x, unsafe=True)), ref, atol=1e-5)
    # __call__ routes to unsafe for a prefer_unsafe handle
    assert np.allclose(np.asarray(hu(x)), ref, atol=1e-5)


def test_two_output_fn_returns_pytree_matching_plain_jit():
    """A two-output fn (value, logits): the robust call returns the output pytree (not a bare array),
    each leaf allclose to plain jit. Exercises `_single_output=False`."""
    import jax
    import jax.numpy as jnp
    from chocofarm.az.lowlatency import compile_lowlatency, run
    p = _toy_params(20, 32, n_actions=7)
    x = np.random.default_rng(3).standard_normal((5, 20)).astype(np.float32)
    v_ref, l_ref = jax.jit(_mlp_value_policy)(p, jnp.asarray(x))
    h = compile_lowlatency(_mlp_value_policy, p, x)
    assert h._single_output is False
    v_got, l_got = run(h, x)
    assert np.allclose(np.asarray(v_got), np.asarray(v_ref), atol=1e-5)
    assert np.allclose(np.asarray(l_got), np.asarray(l_ref), atol=1e-5)


def test_handle_metadata_matches_staged_shapes():
    """The handle's staged in/out avals (read off the compiled executable, P1 derive-don't-duplicate)
    reflect the example shapes/dtypes — the contract a downstream caller introspects without the
    device."""
    from chocofarm.az.lowlatency import compile_lowlatency, AvalSpec
    p = _toy_params(8, 16, None)
    x = np.zeros((4, 8), np.float32)
    h = compile_lowlatency(_mlp_value, p, x)
    # the LAST input leaf is x (params flatten first): its aval is (4, 8) float32
    assert h.in_avals[-1] == AvalSpec(shape=(4, 8), dtype="float32")
    assert h.out_avals == (AvalSpec(shape=(4, 1), dtype="float32"),)
    assert all(a.dtype == "float32" for a in h.in_avals)


# ===========================================================================
# SSOT FAIL-LOUD — the constructor / call validate, never coerce (ADR-0002)
# ===========================================================================
def test_constructor_rejects_non_array_call_leaf():
    """A non-array leaf among the call args (a Python scalar threaded as part of x) is rejected AT
    CONSTRUCTION — the Port/ACL boundary translate-and-validate, not a per-call surprise."""
    from chocofarm.az.lowlatency import compile_lowlatency
    p = _toy_params(8, 16, None)
    # x as a pytree carrying a python int leaf the fn would consume — not an array.
    with pytest.raises((TypeError, ValueError)):
        compile_lowlatency(lambda pp, xx: xx[0] @ pp["W1"], p, (np.zeros((2, 8), np.float32), 3))


def test_robust_call_raises_on_shape_mismatch():
    """A per-call shape mismatch on the robust path raises loudly inside jax (the AOT executable was
    compiled for one shape; a different one cannot run) — ADR-0002 at the jax boundary."""
    from chocofarm.az.lowlatency import compile_lowlatency, run
    p = _toy_params(8, 16, None)
    h = compile_lowlatency(_mlp_value, p, np.zeros((4, 8), np.float32))
    with pytest.raises(Exception):
        run(h, np.zeros((4, 9), np.float32))   # wrong in_dim


def test_prefer_unsafe_refuses_when_unbuildable():
    """`prefer_unsafe=True` with a multi-array `x` (the unsafe leaf-splice is ill-defined) REFUSES at
    construction — a caller asking for the unsafe path is never silently downgraded to robust."""
    from chocofarm.az.lowlatency import compile_lowlatency
    p = _toy_params(8, 16, None)
    x = (np.zeros((2, 8), np.float32), np.zeros((2, 8), np.float32))
    with pytest.raises(ValueError):
        compile_lowlatency(lambda pp, xx: xx[0] @ pp["W1"] + xx[1] @ pp["W1"], p, x, prefer_unsafe=True)


def test_run_unsafe_refuses_on_robust_only_handle():
    """`run(unsafe=True)` on a handle whose unsafe path is not constructible (multi-array x, built
    robust) refuses loudly rather than silently taking the robust path."""
    from chocofarm.az.lowlatency import compile_lowlatency, run
    p = _toy_params(8, 16, None)
    x = (np.zeros((2, 8), np.float32), np.zeros((2, 8), np.float32))
    h = compile_lowlatency(lambda pp, xx: xx[0] @ pp["W1"] + xx[1] @ pp["W1"], p, x)
    assert h._unsafe_call is None
    with pytest.raises(ValueError):
        run(h, x, unsafe=True)


# ===========================================================================
# IMMUTABILITY + PHANTOM TYPE — the handle is a value with zero-cost type params
# ===========================================================================
def test_handle_is_immutable():
    """The handle is a frozen dataclass (ADR-0012 P9 — a value): setting an attribute raises."""
    from chocofarm.az.lowlatency import compile_lowlatency
    h = compile_lowlatency(_mlp_value, _toy_params(8, 16, None), np.zeros((4, 8), np.float32))
    with pytest.raises(dataclasses.FrozenInstanceError):
        h.use_unsafe = True  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        h.in_avals = ()  # type: ignore[misc]


def test_phantom_type_has_zero_runtime_cost():
    """The `In`/`Out` type parameters are PHANTOM — they appear in no field, so subscripting the
    generic produces the same runtime behavior and stores no type object on the instance. The handle's
    fields are exactly the executable + metadata, never an `In`/`Out` value."""
    from chocofarm.az.lowlatency import compile_lowlatency, LowLatencyFn
    h = compile_lowlatency(_mlp_value, _toy_params(8, 16, None), np.zeros((4, 8), np.float32))
    field_names = {f.name for f in dataclasses.fields(h)}
    # no field named for the phantom parameters; the stored state is the executable + staged metadata
    assert "In" not in field_names and "Out" not in field_names
    assert {"_compiled", "_params", "in_avals", "out_avals", "_single_output",
            "_unsafe_call", "_static_leaves", "_x_leaf_index", "use_unsafe"} == field_names
    # subscripting the generic is a no-op at runtime (the alias is the same class)
    assert LowLatencyFn[int, float] is not None
