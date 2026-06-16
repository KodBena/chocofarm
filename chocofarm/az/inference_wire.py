#!/usr/bin/env python3
"""
chocofarm/az/inference_wire.py — the ONE wire CODEC for the Shape B batched ZeroMQ inference service
(docs/design/zmq-inference-service.md §2). Both the server (inference_server.py) and the client
(zmq_net_client.py) import THIS — there is exactly one place the request/response frame is encoded, so
a Python↔Python (and, later, a Python↔C++ ZmqNetClient) codec cannot silently drift (ADR-0012 P7: a
cross-boundary fact has one authoritative home; every side derives its view, none re-authors it).

The frame's BYTE LAYOUT is NOT spelled here — it is DERIVED from the single-source-of-truth
`chocofarm/az/wire_spec.py` (the protocol version, the byte order, the u8/u32 header widths, the f32
dtype). This codec composes those constants into its `struct.Struct` formats; the C++ side derives the
SAME layout from `cpp/include/chocofarm/wire_spec.hpp`, whose constants are drift-checked against the
Python spec in the default test suite (tests/test_wire_drift.py). So a layout change has ONE edit point
(wire_spec.py) and a mechanical net catches a one-sided change (ADR-0012 P1/P7, ADR-0011 Rule 4).

The frame is the `NetPrediction` contract `cpp/include/chocofarm/net.hpp` already defines, on the wire:
length-prefixed LITTLE-ENDIAN float32, fronted by a one-byte protocol-version header so a codec
mismatch fails LOUDLY (ADR-0002) rather than silently misreading floats.

    Request  : [ver:u8][in_dim:u32 LE][X : f32×in_dim LE]
    Response : [ver:u8][n_actions:u32 LE][value:f32 LE][logits : f32×n_actions LE]

`n_actions == 0` ⇒ value-only: the logits block is empty, mirroring `forward.forward_core`'s
`logits=None` (the value-only Stage-1 net). The response value is DE-STANDARDIZED
(v = v_std·y_std + y_mean) and the logits are RAW (not softmaxed) — masking is per-node search state
the server does not hold, so it stays client-side (§2). The codec carries floats and counts only; the
net's internal shape never reaches the wire, so no consumer recompiles when the architecture changes.

Failure semantics (ADR-0002 / ADR-0012 P9, Port/ACL: translate-and-validate, never coerce). Decode is
a BOUNDARY: an unknown protocol byte, a truncated/over-long frame, a length-prefix that does not match
the byte count, or a NaN/Inf float is a LOUD `WireError`, never a zero-filled or truncated forward. The
two encode/decode pairs are exact inverses over finite float32 (the always-on round-trip test pins it).

Public Domain (The Unlicense).
"""
from __future__ import annotations

import struct

import numpy as np
import numpy.typing as npt

from chocofarm.az import wire_spec

# The protocol-version header byte — DERIVED from the wire_spec SSOT (ADR-0012 P1), re-exported so
# existing importers (the codec tests, the server/client) keep `inference_wire.PROTOCOL_VERSION`. Bump
# it in wire_spec.py so an old client/server pairing fails loudly at decode (unknown byte) instead of
# misreading the next field as a float, and the C++ mirror is reconciled by the drift test.
PROTOCOL_VERSION = wire_spec.PROTOCOL_VERSION

# The fixed-header / value struct formats + the f32 dtype are all DERIVED from wire_spec.py — there is
# no `"<BI"` / `"<f4"` literal here (that would be a second author of the layout, ADR-0012 P7). A
# request header is [ver:u8][in_dim:u32]; a response header is [ver:u8][n_actions:u32]; the value is a
# single LE f32. f32 is `wire_spec.FLOAT_BYTES` (4) bytes, all little-endian.
_REQ_HEADER = struct.Struct(wire_spec.REQ_HEADER_FMT)     # ver (u8), in_dim (u32)
_RESP_HEADER = struct.Struct(wire_spec.RESP_HEADER_FMT)   # ver (u8), n_actions (u32)
_F32 = np.dtype(wire_spec.FLOAT_DTYPE)                     # explicit little-endian float32 (both directions)
_F32_BYTES = wire_spec.FLOAT_BYTES


class WireError(ValueError):
    """A malformed inference frame: an unknown protocol byte, a truncated/over-long frame, a
    length-prefix that disagrees with the byte count, or a non-finite float. A loud BOUNDARY rejection
    (ADR-0002 fail-loud / ADR-0012 P9 translate-and-validate) — the codec never coerces a bad frame
    into a plausible forward (no zero-fill, no truncation)."""


def _as_finite_f32_row(X: npt.NDArray[np.floating] | npt.NDArray[np.integer]) -> npt.NDArray[np.float32]:
    """Validate-and-translate an inbound feature vector into a contiguous 1-D little-endian float32 row
    (the wire's input contract). Rejects (ADR-0002): a non-1-D array, an empty vector, or any non-finite
    entry — a NaN/Inf feature is a malformed request, never something to silently forward."""
    a = np.ascontiguousarray(X, dtype=_F32)
    if a.ndim != 1:
        raise WireError(f"feature vector must be 1-D, got shape {a.shape}")
    if a.size == 0:
        raise WireError("feature vector is empty (in_dim must be ≥ 1)")
    if not np.all(np.isfinite(a)):
        raise WireError("feature vector has a non-finite (NaN/Inf) entry — refusing to forward")
    return a


def encode_request(X: npt.NDArray[np.floating] | npt.NDArray[np.integer]) -> bytes:
    """Encode one feature vector `X` into a request frame `[ver][in_dim][X:f32]`. `X` is cast to
    little-endian float32 and validated finite (a NaN/Inf is a loud `WireError`). `in_dim` is derived
    from the vector — never a separate argument that could disagree with the payload (P1)."""
    a = _as_finite_f32_row(X)
    return _REQ_HEADER.pack(PROTOCOL_VERSION, a.shape[0]) + a.tobytes()


def decode_request(frame: bytes) -> npt.NDArray[np.float32]:
    """Decode a request frame back to the feature vector `X` (little-endian float32, shape (in_dim,)).
    BOUNDARY validation (ADR-0002): an unknown protocol byte, a frame too short for its header, a
    payload whose byte count is not exactly `in_dim` floats, or a non-finite entry is a loud
    `WireError` — never a zero-filled or truncated forward."""
    if len(frame) < _REQ_HEADER.size:
        raise WireError(f"request frame too short ({len(frame)} bytes) for its {_REQ_HEADER.size}-byte header")
    ver, in_dim = _REQ_HEADER.unpack_from(frame)
    if ver != PROTOCOL_VERSION:
        raise WireError(f"request protocol byte {ver} != supported {PROTOCOL_VERSION} (codec mismatch)")
    if in_dim == 0:
        raise WireError("request in_dim is 0 (no feature vector)")
    body = frame[_REQ_HEADER.size:]
    want = in_dim * _F32_BYTES
    if len(body) != want:
        raise WireError(f"request payload is {len(body)} bytes, expected {want} (= in_dim {in_dim} × f32)")
    a = np.frombuffer(body, dtype=_F32)
    if not np.all(np.isfinite(a)):
        raise WireError("request feature vector has a non-finite (NaN/Inf) entry — refusing to forward")
    return a


def encode_response(value: float, logits: npt.NDArray[np.floating] | None) -> bytes:
    """Encode a `NetPrediction` into a response frame `[ver][n_actions][value][logits:f32]`. `value` is
    the DE-STANDARDIZED scalar; `logits` are the RAW (non-softmaxed) policy logits, or `None` for the
    value-only net (`n_actions == 0`, empty logits block — mirroring `forward_core`'s `logits=None`)."""
    if logits is None:
        n_actions = 0
        logit_bytes = b""
    else:
        la = np.ascontiguousarray(logits, dtype=_F32).ravel()
        n_actions = int(la.shape[0])
        logit_bytes = la.tobytes()
    return _RESP_HEADER.pack(PROTOCOL_VERSION, n_actions) + struct.pack(wire_spec.VALUE_FMT, float(value)) + logit_bytes


def decode_response(frame: bytes) -> tuple[float, npt.NDArray[np.float32] | None]:
    """Decode a response frame back to `(value, logits)`: `value` a Python float (de-standardized),
    `logits` a little-endian float32 array of length `n_actions`, or `None` when `n_actions == 0`
    (value-only). BOUNDARY validation (ADR-0002): an unknown protocol byte, a frame too short for the
    header+value, or a logits block whose byte count is not exactly `n_actions` floats is a loud
    `WireError`."""
    fixed = _RESP_HEADER.size + _F32_BYTES
    if len(frame) < fixed:
        raise WireError(f"response frame too short ({len(frame)} bytes) for its {fixed}-byte header+value")
    ver, n_actions = _RESP_HEADER.unpack_from(frame)
    if ver != PROTOCOL_VERSION:
        raise WireError(f"response protocol byte {ver} != supported {PROTOCOL_VERSION} (codec mismatch)")
    (value,) = struct.unpack_from(wire_spec.VALUE_FMT, frame, _RESP_HEADER.size)
    body = frame[fixed:]
    want = n_actions * _F32_BYTES
    if len(body) != want:
        raise WireError(f"response logits block is {len(body)} bytes, expected {want} "
                        f"(= n_actions {n_actions} × f32)")
    logits = np.frombuffer(body, dtype=_F32) if n_actions > 0 else None
    return float(value), logits
