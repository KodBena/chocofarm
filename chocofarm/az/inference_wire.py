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

The frame is the `NetPrediction` contract `cpp/include/chocofarm/net.hpp` defines, BATCHED B leaves per
message, on the wire: length-prefixed LITTLE-ENDIAN float32, fronted by a one-byte protocol-version
header so a codec mismatch fails LOUDLY (ADR-0002) rather than silently misreading floats. B=1 is the
degenerate single-leaf case (a single-leaf caller passes a (1, in_dim) matrix), so the batched frame
SUBSUMES single-leaf — there is no dual-mode.

    Request  : [ver:u8][B:u32 LE][in_dim:u32 LE][X : f32×(B·in_dim) LE]   (row-major)
    Response : [ver:u8][B:u32 LE][n_actions:u32 LE][ B × (value:f32 LE, logits:f32×n_actions LE) ]

`encode_request` takes a `(B, in_dim)` float32 matrix; `decode_request` returns it. `encode_response`
takes B values + B logits rows (a `(B, n_actions)` matrix or `None`); `decode_response` returns them.
`n_actions == 0` ⇒ value-only: every prediction's logits block is empty, mirroring
`forward.forward_core`'s `logits=None` (the value-only Stage-1 net). The response values are
DE-STANDARDIZED (v = v_std·y_std + y_mean) and the logits are RAW (not softmaxed) — masking is per-node
search state the server does not hold, so it stays client-side (§2). The codec carries floats and
counts only; the net's internal shape never reaches the wire, so no consumer recompiles when the
architecture changes.

Failure semantics (ADR-0002 / ADR-0012 P9, Port/ACL: translate-and-validate, never coerce). Decode is
a BOUNDARY: an unknown protocol byte, a truncated/over-long frame, a length-prefix that does not match
the byte count, a ragged batch, or a NaN/Inf float is a LOUD `WireError`, never a zero-filled or
truncated forward. The two encode/decode pairs are exact inverses over finite float32 (the always-on
round-trip test pins it).

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
# no `"<BII"` / `"<f4"` literal here (that would be a second author of the layout, ADR-0012 P7). A
# request header is [ver:u8][B:u32][in_dim:u32]; a response header is [ver:u8][B:u32][n_actions:u32];
# the value is a single LE f32. f32 is `wire_spec.FLOAT_BYTES` (4) bytes, all little-endian.
_REQ_HEADER = struct.Struct(wire_spec.REQ_HEADER_FMT)     # ver (u8), B (u32), in_dim (u32)
_RESP_HEADER = struct.Struct(wire_spec.RESP_HEADER_FMT)   # ver (u8), B (u32), n_actions (u32)
_VALUE = struct.Struct(wire_spec.VALUE_FMT)               # one prediction's value scalar (LE f32)
_F32 = np.dtype(wire_spec.FLOAT_DTYPE)                     # explicit little-endian float32 (both directions)
_F32_BYTES = wire_spec.FLOAT_BYTES


class WireError(ValueError):
    """A malformed inference frame: an unknown protocol byte, a truncated/over-long frame, a
    length-prefix that disagrees with the byte count, or a non-finite float. A loud BOUNDARY rejection
    (ADR-0002 fail-loud / ADR-0012 P9 translate-and-validate) — the codec never coerces a bad frame
    into a plausible forward (no zero-fill, no truncation)."""


def _as_finite_f32_matrix(X: npt.NDArray[np.floating] | npt.NDArray[np.integer]) -> npt.NDArray[np.float32]:
    """Validate-and-translate an inbound feature batch into a contiguous (B, in_dim) little-endian
    float32 matrix (the wire's input contract). A 1-D input is promoted to a single row (B=1, the
    degenerate single-leaf case) so single-leaf callers pass one vector. Rejects (ADR-0002): a rank ≠
    1-or-2 array, an empty batch (B==0 or in_dim==0), or any non-finite entry — a NaN/Inf feature is a
    malformed request, never something to silently forward."""
    a = np.ascontiguousarray(X, dtype=_F32)
    if a.ndim == 1:
        a = a.reshape(1, -1)
    if a.ndim != 2:
        raise WireError(f"feature batch must be a (B, in_dim) matrix (or a 1-D row), got shape {a.shape}")
    if a.shape[0] == 0:
        raise WireError("feature batch is empty (B must be ≥ 1)")
    if a.shape[1] == 0:
        raise WireError("feature batch in_dim is 0 (no features)")
    if not np.all(np.isfinite(a)):
        raise WireError("feature batch has a non-finite (NaN/Inf) entry — refusing to forward")
    return np.ascontiguousarray(a)


def encode_request(X: npt.NDArray[np.floating] | npt.NDArray[np.integer]) -> bytes:
    """Encode a `(B, in_dim)` feature matrix `X` into a batched request frame `[ver][B][in_dim][X:f32]`
    (X row-major). `X` is cast to little-endian float32 and validated finite (a NaN/Inf is a loud
    `WireError`). A 1-D input is the degenerate B=1 single-leaf case. `B` and `in_dim` are DERIVED from
    the matrix — never separate arguments that could disagree with the payload (P1)."""
    a = _as_finite_f32_matrix(X)
    B, in_dim = a.shape
    return _REQ_HEADER.pack(PROTOCOL_VERSION, B, in_dim) + a.tobytes()


def decode_request(frame: bytes) -> npt.NDArray[np.float32]:
    """Decode a batched request frame back to the `(B, in_dim)` feature matrix (little-endian float32).
    BOUNDARY validation (ADR-0002): an unknown protocol byte, a frame too short for its header, a B or
    in_dim of 0, a payload whose byte count is not exactly `B·in_dim` floats, or a non-finite entry is a
    loud `WireError` — never a zero-filled or truncated forward."""
    if len(frame) < _REQ_HEADER.size:
        raise WireError(f"request frame too short ({len(frame)} bytes) for its {_REQ_HEADER.size}-byte header")
    ver, B, in_dim = _REQ_HEADER.unpack_from(frame)
    if ver != PROTOCOL_VERSION:
        raise WireError(f"request protocol byte {ver} != supported {PROTOCOL_VERSION} (codec mismatch)")
    if B == 0:
        raise WireError("request B is 0 (empty batch)")
    if in_dim == 0:
        raise WireError("request in_dim is 0 (no feature vector)")
    body = frame[_REQ_HEADER.size:]
    want = B * in_dim * _F32_BYTES
    if len(body) != want:
        raise WireError(f"request payload is {len(body)} bytes, expected {want} "
                        f"(= B {B} × in_dim {in_dim} × f32)")
    a = np.frombuffer(body, dtype=_F32).reshape(B, in_dim)
    if not np.all(np.isfinite(a)):
        raise WireError("request feature batch has a non-finite (NaN/Inf) entry — refusing to forward")
    return a


def encode_response(values: npt.NDArray[np.floating],
                    logits: npt.NDArray[np.floating] | None) -> bytes:
    """Encode B predictions into a batched response frame
    `[ver][B][n_actions][ B × (value, logits:f32) ]`. `values` is a length-B 1-D array of
    DE-STANDARDIZED scalars; `logits` is a `(B, n_actions)` matrix of RAW (non-softmaxed) policy logits,
    or `None` for the value-only net (`n_actions == 0`, empty per-row logits block — mirroring
    `forward_core`'s `logits=None`). `B` is derived from `values`; `logits` (when present) must have B
    rows (a loud `WireError` otherwise — never a ragged scatter)."""
    v = np.ascontiguousarray(values, dtype=_F32).ravel()
    B = int(v.shape[0])
    if B == 0:
        raise WireError("encode_response: B is 0 (no predictions)")
    if logits is None:
        n_actions = 0
        logit_rows: npt.NDArray[np.float32] | None = None
    else:
        la = np.ascontiguousarray(logits, dtype=_F32)
        if la.ndim != 2:
            raise WireError(f"encode_response: logits must be a (B, n_actions) matrix, got shape {la.shape}")
        if la.shape[0] != B:
            raise WireError(f"encode_response: logits has {la.shape[0]} rows, expected B={B}")
        n_actions = int(la.shape[1])
        logit_rows = la
    out = bytearray(_RESP_HEADER.pack(PROTOCOL_VERSION, B, n_actions))
    for i in range(B):
        out += _VALUE.pack(float(v[i]))
        if logit_rows is not None:
            out += logit_rows[i].tobytes()
    return bytes(out)


def decode_response(frame: bytes) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32] | None]:
    """Decode a batched response frame back to `(values, logits)`: `values` a length-B little-endian
    float32 array (de-standardized), `logits` a `(B, n_actions)` little-endian float32 matrix, or `None`
    when `n_actions == 0` (value-only). BOUNDARY validation (ADR-0002): an unknown protocol byte, a B of
    0, a frame too short for the header, or a body whose byte count is not exactly
    `B·(1 + n_actions)` floats is a loud `WireError`."""
    if len(frame) < _RESP_HEADER.size:
        raise WireError(f"response frame too short ({len(frame)} bytes) for its {_RESP_HEADER.size}-byte header")
    ver, B, n_actions = _RESP_HEADER.unpack_from(frame)
    if ver != PROTOCOL_VERSION:
        raise WireError(f"response protocol byte {ver} != supported {PROTOCOL_VERSION} (codec mismatch)")
    if B == 0:
        raise WireError("response B is 0 (no predictions)")
    body = frame[_RESP_HEADER.size:]
    want = B * (1 + n_actions) * _F32_BYTES   # each prediction: one value + n_actions logits
    if len(body) != want:
        raise WireError(f"response body is {len(body)} bytes, expected {want} "
                        f"(= B {B} × (value + n_actions {n_actions}) × f32)")
    # The body is B records of (value:f32, logits:f32×n_actions). Read it as a (B, 1+n_actions) matrix:
    # column 0 is the values, columns 1.. are the logits — one np.frombuffer, no per-row Python copy.
    rec = np.frombuffer(body, dtype=_F32).reshape(B, 1 + n_actions)
    values = np.ascontiguousarray(rec[:, 0])
    logits = np.ascontiguousarray(rec[:, 1:]) if n_actions > 0 else None
    return values, logits
