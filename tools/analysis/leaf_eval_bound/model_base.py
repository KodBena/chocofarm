"""
tools/analysis/leaf_eval_bound/model_base.py
============================================

The TYPED contract a leaf-eval TRANSPORT-VARIANT model satisfies — the `TransportModel` Protocol
(the responsibility-refactor note's move 3, ADR-0012 P8 typed-signature-is-SSOT). The 5 variant
models (`model_zmq_baseline`, `model_shm_spin_poll`, `model_futex_wake`, `model_lockfree_mpsc`,
`model_cpp_inproc_port`) satisfy it STRUCTURALLY — they neither import nor inherit it (they are
MODULES, not classes); the Protocol NAMES the duck-typed contract they already share and the two
transport runners (`transport_sweep`, `untrusted_drive`) consume, so "what a transport model is" has
ONE home instead of being convention scattered across 5 copy-paste templates.

WHY A PROTOCOL OF CALLABLE ATTRIBUTES (not methods-with-self). A model is a MODULE: its `f`,
`build_driver`, etc. are module-level FUNCTIONS (no `self`). A Protocol of `Callable` attributes
types a module faithfully (a module's function IS a callable attribute), and `@runtime_checkable`
lets the conformance test assert `isinstance(variant_module, TransportModel)` — the structural net
(ADR-0011) that stops the GROWING variant family from diverging: a new variant missing a member
fails `tests/test_transport_model_conformance.py`, not silently at a runner call site (the §2.4 "no
shared interface" divergence this closes). The tool is not mypy-gated, so the test IS the enforcement.

SCOPE (honest). The earlier OT->JAX migration + move-3b already grew the canonical per-model methods
(`registry_qname` / `sigmas` / `trusted_flags`) and collapsed the runners' getattr-shims, so this move
is the typed contract + the net, NOT shim-deletion (already done). `bound` / `SIGMAS` / `COSTS` are
deliberately NOT in the contract — only 3 of the 5 variants expose them (model-specific diagnostics,
not the shared interface); the optional contrast helpers (`gather_contrast`, …) stay getattr'd in the
runner (model-specific, legitimately optional). The static-grounded models (`model_capacity` /
`model_cycletime`, a different dialect consumed by `throughput_bound`) are out of scope here (2 stable
models, not the growing family).

Public Domain (The Unlicense).
"""
from __future__ import annotations

from typing import Any, Callable, Mapping, Protocol, Sequence, runtime_checkable


@runtime_checkable
class TransportModel(Protocol):
    """The structural contract every transport-variant model satisfies + the two transport runners
    consume. Declared as Callable ATTRIBUTES (a module's functions carry no `self`); `@runtime_checkable`
    so the conformance test can `isinstance`-check a variant MODULE (presence of every member)."""

    INPUT_NAMES: Sequence[str]                                              # the input order (x is ordered by this)
    SLUG: str                                                              # the variant's registry slug
    throughput_jax: Callable[[Any], Any]                                   # the driver's f: x_array -> scalar dps
    throughput_numpy: Callable[[Mapping[str, float]], float]               # the headline f: dict -> dps
    registry_qname: Callable[[str], str]                                   # input name -> manifest quantity name
    initial_point: Callable[..., Mapping[str, float]]                      # (trust=...) -> grounded x0
    sigmas: Callable[..., Mapping[str, float]]                             # (trust=...) -> per-input 1-sigma
    trusted_flags: Callable[..., Mapping[str, bool]]                       # (trust=...) -> per-input trusted?
    build_driver: Callable[..., Any]                                       # (tolerance=..., trust=...) -> (driver, x0)
    cycle_breakdown: Callable[[Mapping[str, float]], Mapping[str, float]]  # x -> per-forward us breakdown
    serve_sawtooth: Callable[[int], float]                                 # real rows -> bucketed serve dps


# The contract's members, derived from the Protocol's OWN annotations (P1 single-home: the conformance
# test reads THIS, never a hand-retyped list — a member added to the Protocol is automatically enforced).
TRANSPORT_MODEL_MEMBERS: tuple[str, ...] = tuple(
    m for m in TransportModel.__annotations__ if not m.startswith("_"))
