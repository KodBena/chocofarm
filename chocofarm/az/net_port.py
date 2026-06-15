#!/usr/bin/env python3
"""
chocofarm/az/net_port.py — the `Net` PORT: the raw value+policy forward as an injected dependency
(docs/design/zmq-inference-service.md §1, §6; the Python mirror of the C++ `NetEvaluator` port).

The search holds the net as an injected dependency (seam 2 of scaling-and-cpp-seam.md §0) and, at the
leaf, needs only one thing: a raw forward `X → (value, logits)`. `Net` is that port as a
`typing.Protocol` — STRUCTURAL, so any object with the right `predict` shape satisfies it without
inheriting. Two impls satisfy it interchangeably (the zero-cost ACL, §1):

    * the LOCAL forward — `ValueMLPNet`, a thin adapter over `ValueMLP` (matmul in-process), and
    * the REMOTE forward — `ZmqNetClient` (zmq_net_client.py), the SSOT batched service round-trip.

so a Python search (or the parity harness) uses local-or-remote with zero call-site change.

What `predict` returns — and what it does NOT. `predict(X) -> (value, logits)` is the RAW forward
BENEATH the masked/de-standardized conveniences (ADR design §2):

    * `value`  — the DE-STANDARDIZED scalar (v = v_std·y_std + y_mean, the λ-penalized-return scale),
                 exactly `ValueMLP.predict_value`.
    * `logits` — the RAW policy logits over the fixed n_actions slots (NOT softmaxed), or `None` for a
                 value-only net (mirroring `forward.forward_core`'s `logits=None`).

The masked softmax (`ValueMLP._masked_softmax`, the search's prior) is NOT part of this port: the legal
mask is per-node search state, and masking is a pure function of `(raw logits, legal_mask)`, so it
stays at the CONSUMER. `ValueMLP.predict_both`/`predict_policy` remain the masked/de-standardized
conveniences composed ON TOP of this raw forward; the port is the forward beneath them, which is exactly
what the wire (inference_wire.py) carries — so local and remote return the same `NetPrediction` shape.

This module imports `ValueMLP` LAZILY (inside the adapter ctor / a TYPE_CHECKING block), so importing
the port itself does not pull the jax/numba kernel boundary into a consumer or the mypy gate.

Public Domain (The Unlicense).
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from chocofarm.az.mlp import ValueMLP


@runtime_checkable
class Net(Protocol):
    """The raw leaf-evaluator forward as a structural port: `predict(X) -> (value, logits)`.

    `runtime_checkable` so a structural `isinstance(x, Net)` check (the always-on Protocol-satisfaction
    test) can assert both impls conform; the real contract is the SIGNATURE below (the type checker
    enforces the argument/return shapes; `runtime_checkable` only sees the method's presence)."""

    def predict(self, X: npt.NDArray[np.floating]) -> tuple[float, npt.NDArray[np.float32] | None]:
        """One forward over a single feature vector `X` (shape (in_dim,)). Returns `(value, logits)`:
        `value` the DE-STANDARDIZED scalar, `logits` the RAW (non-softmaxed) policy logits over the
        n_actions slots as float32, or `None` for a value-only net. Masking is the caller's."""
        ...


class ValueMLPNet:
    """The LOCAL `Net` impl: a thin adapter exposing `ValueMLP`'s raw forward beneath its masked/de-std
    conveniences. `predict(X)` runs the trunk once and returns `(de-standardized value, RAW logits)` —
    the SAME quantity the service returns over the wire, so a search can swap this for `ZmqNetClient`
    with no call-site change (the zero-cost ACL, design §1).

    It reuses `ValueMLP`'s own forward (`_forward` → `forward_core` → de-standardize), so this adapter
    introduces NO second transcription of the graph (R11 / ADR-0012 P1 — there is one forward). It is
    constructed from a `ValueMLP`; the `ValueMLP` import is deferred to call/type time so importing the
    port does not pull the jax/numba boundary."""

    def __init__(self, net: ValueMLP) -> None:
        self._net = net

    def predict(self, X: npt.NDArray[np.floating]) -> tuple[float, npt.NDArray[np.float32] | None]:
        """Raw forward over one feature vector `X` (shape (in_dim,)). Returns `(value, logits)` with
        `value` de-standardized (exactly `ValueMLP.predict_value`) and `logits` the RAW head output as
        float32 (or `None` for a value-only net) — NOT masked, NOT softmaxed (the consumer's job)."""
        x = np.ascontiguousarray(X)
        if x.ndim != 1:
            raise ValueError(f"ValueMLPNet.predict expects a 1-D feature vector, got shape {x.shape}")
        # _forward returns (None, v_std (B,), logits (B, n_actions)|None) on a (1, in_dim) batch; reuse
        # the ONE forward and de-standardize exactly as predict_value does, so this matches the wire.
        _, v_std, logits = self._net._forward(x[None, :])
        value = float(v_std[0]) * self._net.y_std + self._net.y_mean
        if logits is None:
            return value, None
        return value, np.ascontiguousarray(logits[0], dtype=np.float32)
