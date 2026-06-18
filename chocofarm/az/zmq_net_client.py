#!/usr/bin/env python3
"""
chocofarm/az/zmq_net_client.py — `ZmqNetClient`: the REMOTE `Net` impl that RPCs the Shape B inference
service (docs/design/zmq-inference-service.md §1, §5, §6).

It is the drop-in the parity harness checks and the REFERENCE the future C++ `ZmqNetClient` mirrors:
a worker holds one of these at the leaf and calls a blocking `predict(X) -> (value, logits)` — the same
raw `Net` port (net_port.py) a local `ValueMLPNet` satisfies, so a Python search uses local-or-remote
interchangeably with zero call-site change (the zero-cost ACL, design §1). The forward runs REMOTELY,
on the SSOT batched service; this client only encodes the request, round-trips it, and decodes the
`NetPrediction` (de-standardized value + RAW logits — masking stays client-side, design §2).

Transport: a ZeroMQ REQ socket (the lock-step request→reply peer of the server's ROUTER). REQ enforces
the strict send→recv alternation a blocking leaf RPC wants; the server's greedy-drain batches whatever
REQ requests are concurrently in-flight across many clients. The codec is the shared one
(inference_wire.py) — there is no second hand-written frame here (ADR-0012 P7).

Failure semantics (ADR-0002 / ADR-0012 P9 — design §5): a receive timeout (`RCVTIMEO`) or a transport
error is a LOUD raise (`InferenceClientError`), NOT a silent fallback to a local net — falling back
would mask the SSOT path being down, the exact silent failure ADR-0002 forbids. A malformed reply
(bad protocol byte, wrong length) is a loud `WireError` from the codec. There is no degraded-quiet mode.
(The Python client RAISES; the C++ `ZmqNetClient` will instead return `std::expected<…, Error>` per
ADR-0012 P9 rule 5 — same loud-typed-failure semantics in each language's idiom.)

Public Domain (The Unlicense).
"""
from __future__ import annotations

from types import TracebackType
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from chocofarm.az.inference_wire import decode_response, encode_request

if TYPE_CHECKING:
    import zmq


class InferenceClientError(RuntimeError):
    """A failed inference RPC: a receive timeout, the service unreachable/down, or a transport error.
    A LOUD typed failure (ADR-0002) — the client never silently falls back to a local net, because that
    would mask the SSOT service being down. The C++ register of this is `std::expected<…, Error>`."""


class ZmqNetClient:
    """The remote `Net` impl (a blocking leaf-RPC client). Satisfies the `Net` port (net_port.py):
    `predict(X) -> (value, logits)` where `value` is the DE-STANDARDIZED scalar and `logits` are the RAW
    (non-softmaxed) policy logits, or `None` for a value-only net. Masking is the caller's (design §2).

    Construct with the service endpoint and a receive timeout; one client owns one REQ socket and is
    NOT thread-safe (REQ is strict send→recv lock-step — give each worker its own client). Usable as a
    context manager (closes the socket on exit) or explicitly via `close()`."""

    def __init__(self, endpoint: str = "tcp://127.0.0.1:5599", *, recv_timeout_ms: int = 5000,
                 context: "zmq.Context[zmq.Socket[bytes]] | None" = None) -> None:
        import zmq
        self._endpoint = endpoint
        self._recv_timeout_ms = int(recv_timeout_ms)
        self._owns_context = context is None
        self._ctx: zmq.Context[zmq.Socket[bytes]] = context if context is not None else zmq.Context()
        self._sock: zmq.Socket[bytes] = self._ctx.socket(zmq.REQ)
        # Bound the receive (ADR-0002 / the transport.py deadlock-fix discipline): a server-down or a
        # dropped (malformed-request) reply must become a loud timeout, not a forever-block at the leaf.
        self._sock.setsockopt(zmq.RCVTIMEO, self._recv_timeout_ms)
        self._sock.setsockopt(zmq.LINGER, 0)
        self._sock.connect(endpoint)

    def predict(self, X: npt.NDArray[np.floating]) -> tuple[float, npt.NDArray[np.float32] | None]:
        """Blocking forward RPC over one feature vector `X` (shape (in_dim,)): encode → send → recv →
        decode → `(value, logits)`. The request is the degenerate B=1 batched frame (the batched frame
        subsumes single-leaf); the reply carries one prediction, unwrapped here to `(value, logits)`.
        The value is de-standardized and the logits RAW (NOT softmaxed) — the consumer masks. A NaN/Inf
        feature is rejected by the codec before it ever hits the wire (ADR-0002). A timeout / transport
        failure raises `InferenceClientError` LOUDLY — no silent local fallback (design §5)."""
        import zmq
        req = encode_request(X)   # codec validates finite + 1-D (→ B=1) before anything touches the socket
        try:
            self._sock.send(req)
            reply = self._sock.recv()
        except zmq.Again as exc:
            raise InferenceClientError(
                f"inference RPC to {self._endpoint} timed out after {self._recv_timeout_ms} ms "
                f"(service down, overloaded, or it rejected the request) — NOT falling back to a "
                f"local net (ADR-0002)") from exc
        except zmq.ZMQError as exc:
            raise InferenceClientError(
                f"inference RPC to {self._endpoint} failed at the transport: {exc}") from exc
        values, logits = decode_response(reply)
        # a single-leaf RPC expects exactly ONE prediction back (B=1) — a different count is a
        # desynchronized wire, a loud failure rather than a silent wrong-row read (ADR-0002).
        if values.shape[0] != 1:
            from chocofarm.az.inference_wire import WireError
            raise WireError(f"single-leaf RPC reply carried {values.shape[0]} predictions, expected 1 (B=1)")
        return float(values[0]), (None if logits is None else logits[0])

    def close(self) -> None:
        """Close the REQ socket and (if we created it) terminate the context. Idempotent-safe."""
        self._sock.close(linger=0)
        if self._owns_context:
            self._ctx.term()

    def __enter__(self) -> "ZmqNetClient":
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc: BaseException | None,
                 tb: TracebackType | None) -> None:
        self.close()
