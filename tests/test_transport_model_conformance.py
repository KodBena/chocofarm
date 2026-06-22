"""
tests/test_transport_model_conformance.py
=========================================

The structural NET for the transport-variant model contract (responsibility-refactor move 3,
ADR-0011 mechanization): every transport-variant model MUST satisfy the `TransportModel` Protocol
(`model_base.py`) — the typed SSOT of "what a transport model is." A new variant that omits a member
(`registry_qname`, `sigmas`, `trusted_flags`, …) fails HERE, not silently at a runner call site (the
§2.4 "no shared interface" divergence this closes). The tool is not mypy-gated, so this test is the
enforcement, not mypy.

Run-free: pure introspection (isinstance on the imported module + inspect.signature on the trust
methods). No driver step, no timed bench, no postgres, no OpenTURNS.

Public Domain (The Unlicense).
"""
from __future__ import annotations

import importlib
import inspect
import os
import sys

import pytest

_LEB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "tools", "analysis")
if _LEB not in sys.path:
    sys.path.insert(0, _LEB)

from leaf_eval_bound.models.model_base import TRANSPORT_MODEL_MEMBERS, TransportModel  # noqa: E402

# The transport-variant family (the GROWING one). A new variant is added here AND must conform.
_VARIANTS = ["leaf_eval_bound.models.model_zmq_baseline", "leaf_eval_bound.models.model_shm_spin_poll", "leaf_eval_bound.models.model_futex_wake",
             "leaf_eval_bound.models.model_lockfree_mpsc", "leaf_eval_bound.models.model_cpp_inproc_port"]


@pytest.mark.parametrize("modname", _VARIANTS)
def test_variant_satisfies_the_transport_model_protocol(modname: str) -> None:
    """Every transport variant satisfies `TransportModel` structurally — presence of every contract
    member (the anti-divergence net). Checked two ways: an explicit per-member hasattr (names the
    missing one) AND the runtime_checkable `isinstance` (the typed surface)."""
    mod = importlib.import_module(modname)
    missing = [m for m in TRANSPORT_MODEL_MEMBERS if not hasattr(mod, m)]
    assert not missing, f"{modname} is missing TransportModel member(s): {missing}"
    assert isinstance(mod, TransportModel), f"{modname} does not satisfy TransportModel"


@pytest.mark.parametrize("modname", _VARIANTS)
def test_variant_trust_methods_accept_trust(modname: str) -> None:
    """The manifest-dialect contract: `initial_point` / `sigmas` / `trusted_flags` / `build_driver` each
    take a `trust` parameter (the static dialect's no-`trust` signature would silently mis-resolve here —
    a finer divergence than mere presence, so it is asserted explicitly)."""
    mod = importlib.import_module(modname)
    for fn in ("initial_point", "sigmas", "trusted_flags", "build_driver"):
        params = inspect.signature(getattr(mod, fn)).parameters
        assert "trust" in params, f"{modname}.{fn} must accept `trust` (the manifest contract)"


def test_transport_model_members_are_the_documented_contract() -> None:
    """`TRANSPORT_MODEL_MEMBERS` is derived from the Protocol's annotations (P1 single-home); this pins
    the contract surface, so a member silently dropped from the Protocol fails here (alerting the author
    that the runners' required surface changed)."""
    assert set(TRANSPORT_MODEL_MEMBERS) == {
        "INPUT_NAMES", "SLUG", "throughput_jax", "throughput_numpy", "registry_qname",
        "initial_point", "sigmas", "trusted_flags", "build_driver", "cycle_breakdown", "serve_sawtooth"}
