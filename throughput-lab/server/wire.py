#!/usr/bin/env python3
"""
throughput-lab/server/wire.py — the Python view of the producer<->server wire for the clean-room
throughput testbed. It DERIVES the SAME byte layout the C++ side (cpp/wire.hpp) derives: the two
files are two views of one truth, not two authors of it (ADR-0012 P7). The layout is a faithful
copy of chocofarm's live inference wire (chocofarm/az/wire_spec.py + inference_wire.py), so this
testbed's server is byte-for-byte comparable with the production serving path.

TWO LAYERS, KEPT APART (ADR-0012 P7: serialization ⊥ transport)
---------------------------------------------------------------
LAYER 1 — THE VALUE FRAME (what the codec in this module encodes/decodes). Length-prefixed
little-endian float32, fronted by a one-byte protocol version (a codec mismatch fails LOUDLY,
ADR-0002). B leaves per message (B=1 is the degenerate single-leaf case; the batched frame
SUBSUMES single-leaf — no dual-mode). All multi-byte fields little-endian:

    Request  : [ver:u8][B:u32 LE][in_dim:u32 LE][X : f32 x (B*in_dim) LE]   (X row-major)
    Response : [ver:u8][B:u32 LE][n_actions:u32 LE][ B x (value:f32 LE, logits:f32 x n_actions LE) ]

  - ver       : PROTOCOL_VERSION (2). A mismatch is a loud WireError, never a misread float.
  - B         : leaf rows in this message (>= 1).
  - in_dim    : feature width per row (241 on the live Stage-A env).
  - X         : B*in_dim float32, ROW-MAJOR (byte (r*in_dim + c)*4 is row r, column c).
  - n_actions : policy action count for the WHOLE batch. 0 => value-only (empty logits blocks).
  - response  : B records [value:f32][logits:f32 x n_actions]; value DE-STANDARDIZED, logits RAW.

  Fixed header = VERSION_BYTES + COUNT_BYTES + COUNT_BYTES = 9 bytes (both directions).

LAYER 2 — THE ZMQ TRANSPORT ENVELOPE (DEALER producer <-> ROUTER server). The server binds a
ZMQ_ROUTER; each producer thread connects a ZMQ_DEALER and sends a multipart message led by an
8-byte correlation id:

    producer DEALER sends :  [ corr-id : u64 (8 raw native-endian bytes) ] [ <Layer-1 request> ]

ZMQ's ROUTER prepends the producer's connection IDENTITY, so the server's recv_multipart yields:

    server ROUTER recv    :  [ identity ] [ corr-id ] [ <Layer-1 request> ]
                              frames[0]    frames[1]    frames[-1]

The server treats frames[1:-1] as an OPAQUE envelope it echoes back VERBATIM (here exactly the
single [corr-id] frame — the server NEVER parses the corr-id), and replies addressed to that
identity:

    server ROUTER sends   :  [ identity ] [ corr-id ] [ <Layer-1 response> ]
    producer DEALER recv  :  [ corr-id ] [ <Layer-1 response> ]   (ZMQ strips the identity)

The corr-id is a TRANSPORT concern (a u64 the producer stamps, the server round-trips byte-for-byte
without interpreting) — it stays OUT of the Layer-1 value codec. This is byte-identical to
chocofarm's production DEALER<->ROUTER path (cpp wire_leaf_pool.hpp + az/inference_server.py).

THE REFINED BOUNDARY TYPE (see BoundedBatch / decode_bounded below). decode_request validates a
frame's INTERNAL self-consistency, but only the SERVER knows its own geometry — so this module also
owns the row/col/dtype LAW (1 <= rows <= max_batch, cols == in_dim, float32) as a refined type whose
validator is the single door a decoded request passes through. Stated once here (wire.py owns the
bytes, so it owns the law); every downstream consumer assumes it without re-checking (ADR-0012). A
violation is a loud per-identity WireError at the boundary, never a crash at the forward / the bucket
ladder / np.concatenate.

Public Domain (The Unlicense).
"""
from __future__ import annotations

import struct

import attrs
import deal
import numpy as np
import numpy.typing as npt

# ---- the protocol version (the codec-mismatch tripwire) ----------------------------------------
# Mirrors chocofarm wire_spec.PROTOCOL_VERSION. v2 is the BATCHED frame. Bump on ANY Layer-1 change.
PROTOCOL_VERSION: int = 2

# ---- byte order + field widths (these ARE the layout) ------------------------------------------
BYTE_ORDER: str = "<"          # struct/numpy little-endian — the one pin that makes the frame portable
VERSION_FMT: str = "B"         # u8  — the protocol-version header byte
COUNT_FMT: str = "I"           # u32 — a length prefix (B, in_dim, n_actions)
FLOAT_FMT: str = "f"           # f32 — a payload float (and the response value scalar)

FLOAT_DTYPE: str = BYTE_ORDER + "f4"   # '<f4' — the wire dtype both directions

VERSION_BYTES: int = struct.calcsize(VERSION_FMT)   # 1
COUNT_BYTES: int = struct.calcsize(COUNT_FMT)       # 4
FLOAT_BYTES: int = struct.calcsize(FLOAT_FMT)       # 4

# The two fixed-header struct formats (derived, NOT a second '<BII' literal). Both are
# [version][count][count]; named separately so a future response-only field has an obvious edit point.
REQ_HEADER_FMT: str = BYTE_ORDER + VERSION_FMT + COUNT_FMT + COUNT_FMT     # '<BII'  ver, B, in_dim
RESP_HEADER_FMT: str = BYTE_ORDER + VERSION_FMT + COUNT_FMT + COUNT_FMT    # '<BII'  ver, B, n_actions
VALUE_FMT: str = BYTE_ORDER + FLOAT_FMT                                    # '<f'    one value scalar

# The Stage-A feature width on the live env (feat_dim = 5N + 3nD + 6 + n_tel = 241). A payload-size
# fact (in_dim travels per message on the wire), surfaced so the synthetic load matches the real one.
STAGE_A_IN_DIM: int = 241

# The correlation-id field: an 8-byte native-endian u64, the LEADING ZMQ frame on the DEALER side.
# A transport concern (round-tripped opaquely), not part of the Layer-1 codec below.
CORR_BYTES: int = 8

_REQ_HEADER = struct.Struct(REQ_HEADER_FMT)
_RESP_HEADER = struct.Struct(RESP_HEADER_FMT)
_VALUE = struct.Struct(VALUE_FMT)
_F32 = np.dtype(FLOAT_DTYPE)


class WireError(ValueError):
    """A malformed value frame: an unknown protocol byte, a truncated/over-long frame, a length
    prefix that disagrees with the byte count, or a ragged batch. A loud BOUNDARY rejection
    (ADR-0002 fail-loud) — the codec never coerces a bad frame into a plausible forward."""


# =================================================================================================
#  THE LAYER-1 CODEC. The server DECODES requests and ENCODES responses; the round-trip
#  encode_request / decode_response are provided for tests, parity checks, and a Python producer.
#  Signatures mirror chocofarm/az/inference_wire.py.
# =================================================================================================

def encode_request(X: npt.NDArray[np.floating]) -> bytes:
    """Encode a (B, in_dim) float32 feature matrix into a request frame
    [ver][B][in_dim][X:f32] (X row-major). A 1-D input is the degenerate B=1 case. B/in_dim are
    DERIVED from the matrix — never separate args that could disagree with the payload (P1)."""
    a = np.ascontiguousarray(X, dtype=_F32)
    if a.ndim == 1:
        a = a.reshape(1, -1)
    if a.ndim != 2:
        raise WireError(f"feature batch must be (B, in_dim) or 1-D, got shape {a.shape}")
    B, in_dim = a.shape
    if B == 0 or in_dim == 0:
        raise WireError(f"empty feature batch (B={B}, in_dim={in_dim})")
    return _REQ_HEADER.pack(PROTOCOL_VERSION, B, in_dim) + a.tobytes()


def decode_request(frame: bytes) -> npt.NDArray[np.float32]:
    """Decode a request frame back to the (B, in_dim) float32 matrix. BOUNDARY validation
    (ADR-0002): an unknown protocol byte, a too-short frame, a B/in_dim of 0, or a body whose byte
    count is not exactly B*in_dim floats is a loud WireError — never a zero-filled forward."""
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
    want = B * in_dim * FLOAT_BYTES
    if len(body) != want:
        raise WireError(f"request payload is {len(body)} bytes, expected {want} (= B {B} x in_dim {in_dim} x f32)")
    return np.frombuffer(body, dtype=_F32).reshape(B, in_dim)


# =================================================================================================
#  THE REFINED BOUNDARY TYPE (BoundedBatch / decode_bounded). decode_request validates a frame's
#  INTERNAL self-consistency (its body byte count matches its OWN [B][in_dim] header) — but it cannot
#  know the SERVER's geometry. So a frame declaring in_dim=100 against a 241-wide server, or B rows
#  beyond the server's max_batch, decodes "successfully" and only detonates DOWNSTREAM: an in_dim
#  mismatch at the forward, an oversize batch with no covering bucket at the ladder, or a column
#  mismatch inside np.concatenate that poisons EVERY co-batched identity (and, uncaught, wedges the
#  whole server). BoundedBatch closes that gap. It is the ONE door a decoded request passes through,
#  and its validator — the only constructor path — makes the shape legal BY CONSTRUCTION: 2-D,
#  1 <= rows <= max_batch, cols == in_dim, dtype == float32. A violation is a loud per-identity
#  WireError at the BOUNDARY (ADR-0002), not a crash three layers down. wire.py owns the bytes, so it
#  owns the row/col/dtype law: stated ONCE here, every downstream consumer (the gather, the concat,
#  the bucket-ladder `pack`) may assume it WITHOUT re-checking (ADR-0012 — the typed value IS the
#  contract; the illegal state is unrepresentable, not merely rejected at a call site).
# =================================================================================================

@attrs.frozen(kw_only=True)
class BoundedBatch:
    """A decoded request whose shape is legal BY CONSTRUCTION (2-D; 1 <= rows <= max_batch;
    cols == in_dim; little-endian float32). The validator is the ONLY way to build one, so a
    BoundedBatch in hand IS a proof of the invariant — the server's gather, concatenation, and the
    forward's `pack` consume it without re-validating (ADR-0012). `max_batch` and `in_dim` are
    declared BEFORE `X` so they are already bound when X's validator runs (attrs runs field
    validators in definition order)."""

    max_batch: int = attrs.field()
    in_dim: int = attrs.field()
    X: "npt.NDArray[np.float32]" = attrs.field()

    @max_batch.validator
    def _check_max_batch(self, _attribute, value) -> None:
        if value < 1:
            raise WireError(f"BoundedBatch max_batch must be >= 1, got {value}")

    @in_dim.validator
    def _check_in_dim(self, _attribute, value) -> None:
        if value < 1:
            raise WireError(f"BoundedBatch in_dim must be >= 1, got {value}")

    @X.validator
    def _check_X(self, _attribute, X) -> None:
        if X.ndim != 2:
            raise WireError(f"feature matrix must be 2-D (rows, in_dim), got {X.ndim}-D shape {X.shape}")
        rows, cols = int(X.shape[0]), int(X.shape[1])
        if not (1 <= rows <= self.max_batch):
            raise WireError(f"rows {rows} outside [1, {self.max_batch}] — oversize batch, no covering bucket")
        if cols != self.in_dim:
            raise WireError(f"cols {cols} != server in_dim {self.in_dim} — geometry mismatch (would poison the batch)")
        if X.dtype != _F32:
            raise WireError(f"dtype {X.dtype} != wire dtype {_F32} (expected little-endian float32)")


@deal.raises(WireError)
@deal.post(lambda result: 1 <= int(result.X.shape[0]) <= result.max_batch
           and int(result.X.shape[1]) == result.in_dim)
def decode_bounded(payload: bytes, *, max_batch: int, in_dim: int) -> BoundedBatch:
    """Decode a request frame AND narrow it to the SERVER's geometry in one boundary step: the
    Layer-1 codec checks the frame's self-consistency (`decode_request`), then `BoundedBatch` enforces
    the server's law (1 <= rows <= max_batch, cols == in_dim, float32). BOTH raise `WireError`, so a
    single `except WireError` at the caller turns an oversize / wrong-width / wrong-dtype request into
    one loud per-identity reject (ADR-0002) — never a crash at the forward, the ladder, or inside
    np.concatenate. The `deal` contract is the machine-checkable SPEC (discharged by the property
    suite; stripped via `deal.disable()` on the serving hot-path); the `attrs` validator is the
    always-on BOUNDARY GUARD (it must stay on — it is what actually rejects a bad frame)."""
    return BoundedBatch(max_batch=max_batch, in_dim=in_dim, X=decode_request(payload))


def encode_response(values: npt.NDArray[np.floating],
                    logits: npt.NDArray[np.floating] | None) -> bytes:
    """Encode B predictions into a response frame [ver][B][n_actions][ B x (value, logits) ].
    `values` is a length-B 1-D array of DE-STANDARDIZED scalars; `logits` is a (B, n_actions)
    matrix of RAW (non-softmaxed) logits, or None for the value-only net (n_actions == 0). B is
    derived from `values`; `logits` (when present) must have B rows (a loud WireError otherwise)."""
    v = np.ascontiguousarray(values, dtype=_F32).ravel()
    B = int(v.shape[0])
    if B == 0:
        raise WireError("encode_response: B is 0 (no predictions)")
    if logits is None:
        n_actions = 0
        rows: npt.NDArray[np.float32] | None = None
    else:
        la = np.ascontiguousarray(logits, dtype=_F32)
        if la.ndim != 2:
            raise WireError(f"encode_response: logits must be (B, n_actions), got {la.shape}")
        if la.shape[0] != B:
            raise WireError(f"encode_response: logits has {la.shape[0]} rows, expected B={B}")
        n_actions = int(la.shape[1])
        rows = la
    out = bytearray(_RESP_HEADER.pack(PROTOCOL_VERSION, B, n_actions))
    for i in range(B):
        out += _VALUE.pack(float(v[i]))
        if rows is not None:
            out += rows[i].tobytes()
    return bytes(out)


def decode_response(frame: bytes) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32] | None]:
    """Decode a response frame back to (values, logits). BOUNDARY validation (ADR-0002): an
    unknown protocol byte, a B of 0, a too-short frame, or a body whose byte count is not exactly
    B*(1 + n_actions) floats is a loud WireError. n_actions == 0 => logits is None (value-only)."""
    if len(frame) < _RESP_HEADER.size:
        raise WireError(f"response frame too short ({len(frame)} bytes) for its {_RESP_HEADER.size}-byte header")
    ver, B, n_actions = _RESP_HEADER.unpack_from(frame)
    if ver != PROTOCOL_VERSION:
        raise WireError(f"response protocol byte {ver} != supported {PROTOCOL_VERSION} (codec mismatch)")
    if B == 0:
        raise WireError("response B is 0 (no predictions)")
    body = frame[_RESP_HEADER.size:]
    want = B * (1 + n_actions) * FLOAT_BYTES
    if len(body) != want:
        raise WireError(f"response body is {len(body)} bytes, expected {want} (= B {B} x (value + n_actions {n_actions}) x f32)")
    rec = np.frombuffer(body, dtype=_F32).reshape(B, 1 + n_actions)
    values = np.ascontiguousarray(rec[:, 0])
    logits = np.ascontiguousarray(rec[:, 1:]) if n_actions > 0 else None
    return values, logits
