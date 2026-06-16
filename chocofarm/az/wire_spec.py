#!/usr/bin/env python3
"""
chocofarm/az/wire_spec.py — the ONE authoritative declaration of the Shape B ZeroMQ inference wire
frame's byte layout (docs/design/zmq-inference-service.md §2). This is the single source of truth
(ADR-0012 P1 / P7: a cross-boundary fact has ONE home; every side DERIVES its view, none re-authors
it). The Python codec (`inference_wire.py`) derives its `struct.Struct` formats and dtype from THESE
constants, and the C++ `ZmqNetClient` (deferred to the P9 `cpp/` pass) will include the mirror header
`cpp/include/chocofarm/wire_spec.hpp`, whose constants are DRIFT-CHECKED against this module in the
default test suite (`tests/test_wire_drift.py`) — so the two codecs cannot silently diverge.

Why a spec module separate from the codec
-----------------------------------------
Before this module the frame's layout (the protocol version, the LE byte order, the u8/u32 header
field widths, the f32 dtype) lived ONLY inside `inference_wire.py`'s `struct.Struct("<BI")` literals.
There was no place the C++ side could derive from — it would have re-authored the same `[ver:u8]
[count:u32][f32…]` layout from the prose spec, and a Python-side change (a version bump, a wider count
field, a different byte order) would not have been mechanically reconciled against it. That is exactly
the two-writers-of-one-truth sin ADR-0012 P7 names. This module is the one writer; the codec and the
C++ header are derivers, and the drift test is the net.

The frame (the §2 contract), spelled once here and nowhere else:

    Request  : [ver:u8][in_dim   :u32 LE][X      : f32×in_dim   LE]
    Response : [ver:u8][n_actions:u32 LE][value:f32 LE][logits : f32×n_actions LE]

`n_actions == 0` ⇒ value-only (empty logits block, mirroring forward_core's `logits=None`). All
multi-byte fields are LITTLE-ENDIAN (the `<` byte-order pin) so x86↔ARM↔C++ agree. Bump
`PROTOCOL_VERSION` on ANY layout change so an old client/server pairing fails LOUDLY at decode
(unknown protocol byte) instead of misreading the next field as a float (ADR-0002).

Public Domain (The Unlicense).
"""
from __future__ import annotations

from typing import Final

# ---- the protocol version (the header byte that fails a codec mismatch loudly) ----
# Bump on ANY frame-layout change. The C++ wire_spec.hpp mirror declares the SAME value; the drift
# test asserts they are equal, so a bump on one side that is not mirrored fails the default suite.
PROTOCOL_VERSION: Final[int] = 1

# ---- byte order (the one pin that makes the frame cross-architecture) ----
# `struct`/numpy little-endian sigil. The frame is little-endian end to end; the C++ side is built on
# little-endian hosts (x86/ARM little-endian) and reads the fields as native LE. Spelled here so a
# future byte-order change is a one-line edit the drift test reconciles against the C++ mirror.
BYTE_ORDER: Final[str] = "<"            # struct/numpy little-endian

# ---- field widths (the struct format characters for the fixed header fields) ----
# The version byte is an unsigned 8-bit int; the length prefix (in_dim / n_actions) an unsigned 32-bit
# int. The payload/value floats are 32-bit. These three widths ARE the layout; the codec builds its
# `struct.Struct` formats from them, and the C++ mirror declares the matching byte counts.
VERSION_FMT: Final[str] = "B"           # u8  — the protocol-version header byte
COUNT_FMT: Final[str] = "I"             # u32 — the length prefix (in_dim, n_actions)
FLOAT_FMT: Final[str] = "f"             # f32 — the payload float (and the response value scalar)

# Numpy dtype string for the f32 payload arrays (the wire dtype, both directions). Derived from the
# byte order + the float width so there is no second place "little-endian float32" is spelled.
FLOAT_DTYPE: Final[str] = BYTE_ORDER + "f4"   # '<f4'

# ---- derived byte sizes (computed from the format chars, never hardcoded — P1) ----
import struct as _struct  # noqa: E402  (after the constants it derives the sizes from)

VERSION_BYTES: Final[int] = _struct.calcsize(VERSION_FMT)   # 1
COUNT_BYTES: Final[int] = _struct.calcsize(COUNT_FMT)       # 4
FLOAT_BYTES: Final[int] = _struct.calcsize(FLOAT_FMT)       # 4

# The two fixed-header struct formats, derived from the field-width constants above (NOT a second
# `"<BI"` literal). A request header is [ver:u8][in_dim:u32]; a response header is
# [ver:u8][n_actions:u32]. Both are `[version][count]`, so they share a format — but each is named so
# a future divergence (e.g. a response-only field) has an obvious edit point.
REQ_HEADER_FMT: Final[str] = BYTE_ORDER + VERSION_FMT + COUNT_FMT    # '<BI'
RESP_HEADER_FMT: Final[str] = BYTE_ORDER + VERSION_FMT + COUNT_FMT   # '<BI'
VALUE_FMT: Final[str] = BYTE_ORDER + FLOAT_FMT                       # '<f' — the response value scalar
