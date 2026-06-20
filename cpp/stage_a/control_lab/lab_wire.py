#!/usr/bin/env python3
"""
cpp/stage_a/control_lab/lab_wire.py — the Python DERIVATION of the issue-gate control lab's per-forward
on-wire decision frame. The ONE authoritative definition lives in
cpp/include/chocofarm/lab_control_wire.hpp (ADR-0012 P1/P7: a cross-boundary fact has ONE home; every
side derives its view, none re-authors it). This module derives the SAME layout; the magic + length
checks below are the runtime parity floor (a one-sided change is caught loudly at decode, ADR-0002).

THE FRAME IS A TRANSPORT-ENVELOPE FRAME, NOT A VALUE-CODEC CHANGE. The value codec
(chocofarm/az/inference_wire.py) is byte-unchanged; the corr-id frame is byte-unchanged. The wire frame
on the DEALER socket is `[corr-id u64][LAB-CONTROL?][value-payload]`. The decision epoch is the eval
server's FORWARD: a producer thread rides its feature snapshot in the request's LAB-CONTROL frame, and
the server rides that thread's next issue-gate bit back in the reply's LAB-CONTROL frame. When the lab
is OFF no LAB-CONTROL frame is attached (envelope = `[corr-id]`), so the wire is byte-identical to the
production bench.

    FEATURE (producer -> server):  u32 magic=LAB_FEAT | u8 ver | i32 tid | i32 inflight | i32 ready
                                   | i64 msgs | i64 leaves | i64 rtt_us | i64 decisions   (49 bytes)
    GATE    (server -> producer):  u32 magic=LAB_GATE | u8 ver | i32 tid | u8 allow        (10 bytes)

All little-endian (the host is x86_64 LE; asserted by the magic check). Bump LAB_WIRE_VERSION in the
authoritative header (and here) on any layout change.

Public Domain (The Unlicense).
"""
from __future__ import annotations

import struct
from dataclasses import dataclass

# ---- DERIVED from cpp/include/chocofarm/lab_control_wire.hpp — keep the two sides in lockstep. ----
LAB_WIRE_VERSION = 1
LAB_FEAT_MAGIC = 0x1AB0F0A1
LAB_GATE_MAGIC = 0x1AB0F0A2

# magic, ver, tid, inflight, ready (u32,u8,i32,i32,i32) then msgs, leaves, rtt_us, decisions (i64×4).
# struct cannot mix a u8 then i32 without padding under native alignment, so the format is explicit
# little-endian ("<") which packs field-by-field with NO padding (matching the C++ memcpy layout).
_FEAT = struct.Struct("<IBiiiqqqq")   # 4+1+4+4+4+8+8+8+8 = 49 bytes
_GATE = struct.Struct("<IBiB")        # 4+1+4+1 = 10 bytes

LAB_FEAT_BYTES = _FEAT.size
LAB_GATE_BYTES = _GATE.size
assert LAB_FEAT_BYTES == 49, f"lab FEATURE frame size drift: {LAB_FEAT_BYTES} != 49 (P7)"
assert LAB_GATE_BYTES == 10, f"lab GATE frame size drift: {LAB_GATE_BYTES} != 10 (P7)"


@dataclass(frozen=True)
class LabFeature:
    """One producer thread's per-forward feature snapshot (the decision-epoch observation for THIS
    thread). Mirrors chocofarm::lab::LabFeature."""
    tid: int
    inflight: int
    ready: int
    msgs: int
    leaves: int
    rtt_us: int
    decisions: int = 0   # cumulative recorded decisions this thread (the dps numerator)


def encode_feature(f: LabFeature) -> bytes:
    return _FEAT.pack(LAB_FEAT_MAGIC, LAB_WIRE_VERSION, f.tid, f.inflight, f.ready,
                      f.msgs, f.leaves, f.rtt_us, f.decisions)


def decode_feature(frame: bytes) -> LabFeature:
    """Decode a FEATURE frame. BOUNDARY (ADR-0002): a wrong magic/version/length is a loud ValueError —
    never a silently misread snapshot."""
    if len(frame) != LAB_FEAT_BYTES:
        raise ValueError(f"lab control wire: FEATURE frame is {len(frame)} bytes, expected {LAB_FEAT_BYTES}")
    magic, ver, tid, inflight, ready, msgs, leaves, rtt_us, decisions = _FEAT.unpack(frame)
    if magic != LAB_FEAT_MAGIC:
        raise ValueError(f"lab control wire: bad FEATURE magic {magic:#x} (wire-contract drift, P7)")
    if ver != LAB_WIRE_VERSION:
        raise ValueError(f"lab control wire: FEATURE version {ver} != supported {LAB_WIRE_VERSION}")
    return LabFeature(tid=tid, inflight=inflight, ready=ready, msgs=msgs, leaves=leaves,
                      rtt_us=rtt_us, decisions=decisions)


def encode_gate(tid: int, allow: bool) -> bytes:
    return _GATE.pack(LAB_GATE_MAGIC, LAB_WIRE_VERSION, tid, 1 if allow else 0)


def decode_gate(frame: bytes) -> "tuple[int, bool]":
    """Decode a GATE frame to (tid, allow). BOUNDARY (ADR-0002): a wrong magic/version/length is loud."""
    if len(frame) != LAB_GATE_BYTES:
        raise ValueError(f"lab control wire: GATE frame is {len(frame)} bytes, expected {LAB_GATE_BYTES}")
    magic, ver, tid, allow = _GATE.unpack(frame)
    if magic != LAB_GATE_MAGIC:
        raise ValueError(f"lab control wire: bad GATE magic {magic:#x} (wire-contract drift, P7)")
    if ver != LAB_WIRE_VERSION:
        raise ValueError(f"lab control wire: GATE version {ver} != supported {LAB_WIRE_VERSION}")
    return tid, bool(allow)
