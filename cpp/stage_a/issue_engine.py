#!/usr/bin/env python3
"""
cpp/stage_a/issue_engine.py — the PYTHON POLICY ENGINE for the online issue controller (HPO/benchmark).

The slow-and-smart half of the control loop: it BINDS a ZeroMQ REP socket, and for each control tick it
receives the C++ producer's marshalled features (issue_control_bridge.hpp sends them over a ZMQ REQ socket
every cadence_ms), runs a swappable POLICY, and replies with the per-thread issue-allow bits the producer's
refill() then reads. The policy lives HERE (Python) so it can be any model — iterate it without recompiling
C++; the C++ side only actuates the returned bits. Default policy = identity (all-allow ⇒ byte-unchanged).

The wire is the PACKED BINARY contract authored in cpp/include/chocofarm/issue_control_bridge.hpp (its ONE
authoritative P7 definition); this module DERIVES the same layout (a magic + length check is the runtime
parity floor). NO JSON — the control path is latency/jitter sensitive (its realtime behaviour feeds back
into the policy's own prediction quality) against a per-batch forward of a handful of microseconds.

Public Domain (The Unlicense).
"""
from __future__ import annotations

import struct
import threading
from typing import Callable

import zmq

# The ONE authoritative layout lives in issue_control_bridge.hpp; these magics + formats DERIVE it.
FEAT_MAGIC = 0x15C0F1A1
GATE_MAGIC = 0x15C0F1A2
_FEAT_HDR = struct.Struct("<IIId")     # magic, n_threads, d_ceiling, server_rows_per_forward (20 bytes)
_FEAT_ROW = struct.Struct("<iiqqq")    # inflight, ready, msgs, leaves, rtt_us (32 bytes) per thread
_GATE_HDR = struct.Struct("<II")       # magic, n_threads


def decode_features(buf: bytes) -> dict:
    """Decode a FEATURES frame into a dict (the observation a policy consumes). Loud on a contract mismatch
    (ADR-0002 / P7 parity floor)."""
    magic, T, D, srv = _FEAT_HDR.unpack_from(buf, 0)
    if magic != FEAT_MAGIC:
        raise ValueError(f"issue_engine: bad features magic {magic:#x} (wire-contract drift, P7)")
    if len(buf) != _FEAT_HDR.size + T * _FEAT_ROW.size:
        raise ValueError(f"issue_engine: features frame length {len(buf)} != expected for T={T}")
    inflight, ready, msgs, leaves, rtt_us = [], [], [], [], []
    off = _FEAT_HDR.size
    for _ in range(T):
        i, r, m, l, rt = _FEAT_ROW.unpack_from(buf, off)
        off += _FEAT_ROW.size
        inflight.append(i); ready.append(r); msgs.append(m); leaves.append(l); rtt_us.append(rt)
    return {"n_threads": T, "d_ceiling": D, "server_rows_per_forward": srv,
            "inflight": inflight, "ready": ready, "msgs": msgs, "leaves": leaves, "rtt_us": rtt_us}


def encode_gates(n_threads: int, allow: list[int]) -> bytes:
    """Encode the per-thread issue-allow bits (1 = allow the next discretionary issue, 0 = deny)."""
    return _GATE_HDR.pack(GATE_MAGIC, n_threads) + bytes(1 if a else 0 for a in allow)


# The default (identity) policy: allow every thread ⇒ the runner gate reduces to `inflight < D` ⇒ the
# fixed-D runner, byte-unchanged. This is the seam where a real policy (features -> per-thread {0,1}) plugs.
def identity_policy(f: dict) -> list[int]:
    return [1] * f["n_threads"]


Policy = Callable[[dict], list]


class IssueEngine:
    """Runs the policy as a ZMQ REP service on its own thread. start() binds + serves; stop() joins.
    `on_features` (optional) is called with each decoded observation (for logging / data collection)."""

    def __init__(self, endpoint: str, policy: Policy = identity_policy,
                 on_features: Callable[[dict], None] | None = None) -> None:
        self.endpoint = endpoint
        self.policy = policy
        self.on_features = on_features
        self.ticks = 0
        self.last_features: dict | None = None
        self._stop = False
        self._thread: threading.Thread | None = None
        self._err: BaseException | None = None

    def _run(self) -> None:
        ctx = zmq.Context()
        sock = ctx.socket(zmq.REP)
        sock.setsockopt(zmq.RCVTIMEO, 500)   # wake to check _stop; REP stays in recv state on a timeout
        sock.setsockopt(zmq.LINGER, 0)
        sock.bind(self.endpoint)
        try:
            while not self._stop:
                try:
                    buf = sock.recv()
                except zmq.Again:
                    continue
                f = decode_features(buf)
                self.last_features = f
                if self.on_features is not None:
                    self.on_features(f)
                allow = self.policy(f)
                sock.send(encode_gates(f["n_threads"], allow))
                self.ticks += 1
        except BaseException as exc:   # noqa: BLE001 — surface any engine error to the harness (ADR-0002)
            self._err = exc
        finally:
            sock.close(0)
            ctx.term()

    def start(self) -> "IssueEngine":
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop = True
        if self._thread is not None:
            self._thread.join(timeout=3.0)
        if self._err is not None:
            raise self._err
