#!/usr/bin/env python3
"""
chocofarm/az/inference_server.py — the Shape B batched ZeroMQ inference SERVICE
(docs/design/zmq-inference-service.md §3; scaling-and-cpp-seam.md §2 — Axis A cross-episode batching).

The production leaf evaluator (design §0): a single Python process that holds the weights, batches
leaf-evaluation requests from N independent workers, and runs ONE `forward.forward_core` over the
stacked `(B, in_dim)` matrix — the SAME SSOT every Python path runs (R11; there is no second
transcription here). The per-leaf cost amortizes over the batch; the net's architecture stays free to
change because the wire (inference_wire.py) carries only float vectors and `(value, logits)`.

Three parts, separated so the batching LOGIC is testable without a socket and the weight reload is
mockable without redis:

  1. `run_microbatch(...)` — the PURE core: stack the drained requests' feature rows into
     `(B, in_dim)` float32, run ONE `forward_core(params, Xb, jnp)`, DE-STANDARDIZE the value
     (v = v_std·y_std + y_mean), and SCATTER `(value, logits[i])` back per request. No socket, no
     redis — a deterministic function of `(forward_fn, params, y-scale, drained rows)`, so the
     drain/stack/scatter is unit-tested directly (tests/test_zmq_inference.py).
  2. `ParamsSource` — the version-gated weight RELOAD hook (seam 4). `RedisParamsSource` watches the
     published `(phase, version)` via `transport.read_weights` and reloads `params` when it changes
     (the server is the ONE holder of weights — design §3). `StaticParamsSource` injects params
     DIRECTLY so a parity test needs NO redis and NO broadcast (the default test path).
  3. `InferenceServer` — the imperative SHELL: a ZeroMQ ROUTER, the self-clocking greedy-drain loop
     (block until ≥1 request, drain ALL currently-queued up to a max-batch cap, one forward, scatter
     to each request's identity frame), and a reload check between batches. SINGLE-THREADED: JAX/XLA
     owns the forward, no shared-state concurrency and no XLA-in-a-worker-thread (the failure mode the
     jaxtrain-deadlock-rca arc fought, design §3).

Fidelity (design §4): a row of the batched `(B,in)@W` matmul is the same row-wise-independent dot
product as the single-row call — it carries only the forward-roundoff non-exactness the project
already accepts (test_jax_equivalence ABS_TOL=1e-4, ADR-0012 P6 — behavioral float32-equivalence, NOT
byte-identity). This is Axis A (cross-episode) batching; it never touches a search's
Sequential-Halving budget or RNG order (Axis C, deferred).

Failure semantics (ADR-0002 / ADR-0012 P9, Port/ACL: translate-and-validate, never coerce): a
malformed request (wrong length, NaN, bad protocol byte) is rejected LOUDLY at the codec boundary
(inference_wire decode raises `WireError`), never a zero-filled or truncated forward. A reload whose
payload is missing/shape-inconsistent is a loud abort of the reload, not a silent run on stale weights.

Hot import discipline (ADR-0012 P8 / the mypy gate): this module's hot path imports `forward_core` +
numpy + jax.numpy ONLY — it does NOT import `ValueMLP`/`mlp_jax` (which carry the held-out jax/numba
kernel boundary). Params are reconstructed from the transport manifest+blob into the flat dict
`forward_core` consumes, so the server stays out of the deferred boundary and inside the strict gate.

Public Domain (The Unlicense).
"""
from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Callable, Protocol

import numpy as np
import numpy.typing as npt

from chocofarm.az.forward import forward_core
from chocofarm.az.inference_wire import decode_request, encode_response

if TYPE_CHECKING:
    import zmq

# The forward the server runs: `forward_core(params, Xb, xp) -> (v_std (B,), logits (B, n_actions)|None)`.
# Typed as a Callable so a FAKE forward can be injected (the always-on batching-logic test) without a
# real JAX import. The array slots (`Xb`, the two returns) are `Any`, NOT `NDArray`: forward_core is
# backend-polymorphic — the server hands it a jax `Array` (`jnp.asarray(Xb)`) and gets jax arrays back,
# while the stub test hands numpy. That is the SAME documented backend-polymorphism seam forward_core's
# `xp` parameter rides (ADR-0012 P8: an honest `Any` at a real either-backend boundary, not a
# convenience relaxation — the value genuinely IS "numpy-or-jax array"). `params` stays `NDArray[Any]`
# (forward_core indexes it by key; the weight arrays are numpy on both paths).
ForwardFn = Callable[[dict[str, "npt.NDArray[Any]"], Any, Any],
                     tuple[Any, "Any | None"]]

# A drained request: (identity_frame, feature_row) — the ROUTER identity bytes to scatter the response
# back to, and the decoded float32 feature vector (shape (in_dim,)).
DrainedRequest = tuple[bytes, "npt.NDArray[np.float32]"]


def run_microbatch(forward_fn: ForwardFn, params: dict[str, npt.NDArray[Any]],
                   y_mean: float, y_std: float,
                   requests: list[DrainedRequest]) -> list[tuple[bytes, bytes]]:
    """The PURE microbatch core (design §3): STACK the drained requests' feature rows into one
    `(B, in_dim)` float32 matrix, run ONE `forward_fn(params, Xb, jnp)`, DE-STANDARDIZE the value
    (v = v_std·y_std + y_mean), and SCATTER `(value, logits[i])` back per request as an encoded
    response frame. Returns `[(identity, response_bytes), …]` aligned 1:1 with `requests`.

    This is the whole batching contract as a deterministic function of its inputs — no socket, no
    redis — so the always-on test asserts B concurrent requests collapse to ONE `forward_fn` call and
    each request gets ITS OWN row back (the drain/stack/scatter logic), against a stub forward.

    `forward_fn` is the injected forward (the real `forward_core` under JAX in production; a stub in the
    test). The response carries the RAW logits (NOT softmaxed) + the de-standardized value — masking is
    client-side (design §2). A value-only net (`logits is None`) scatters `n_actions=0` to every
    request. Refuses an empty batch (ADR-0002 — the loop only calls this with ≥1 drained request)."""
    if not requests:
        raise ValueError("run_microbatch called with an empty batch (the drain guarantees ≥1 request)")
    identities = [ident for ident, _ in requests]
    rows = [row for _, row in requests]
    in_dim = rows[0].shape[0]
    for i, r in enumerate(rows):
        if r.shape != (in_dim,):
            # A ragged batch (mixed in_dim) is a malformed mix the server must not silently pad/truncate
            # (ADR-0002 fail-loud). Every leaf of one net has the same feature dim by construction.
            raise ValueError(f"batched request {i} has shape {r.shape}, expected ({in_dim},) — ragged batch")
    Xb = np.stack(rows).astype(np.float32, copy=False)   # (B, in_dim) float32, the one stacked input
    import jax.numpy as jnp                               # local import: the JAX backend lives in the shell
    v_std, logits = forward_fn(params, jnp.asarray(Xb), jnp)
    v = np.asarray(v_std, dtype=np.float32).ravel() * np.float32(y_std) + np.float32(y_mean)
    logits_np = None if logits is None else np.asarray(logits, dtype=np.float32)
    out: list[tuple[bytes, bytes]] = []
    for i, ident in enumerate(identities):
        row_logits = None if logits_np is None else logits_np[i]
        out.append((ident, encode_response(float(v[i]), row_logits)))
    return out


# ---- params reconstruction (jax-free) — the manifest+blob into the flat dict forward_core consumes ----
def params_from_manifest_blob(manifest_json: str, blob: bytes) -> tuple[dict[str, npt.NDArray[Any]], float, float]:
    """Reconstruct `(params, y_mean, y_std)` from the transport's `(manifest, blob)` wire payload —
    WITHOUT constructing a `ValueMLP` (so the server stays off the held-out jax/numba boundary). The
    manifest's `layout` carries each weight's `(name, shape, dtype, offset, len)` and the scalar meta
    carries `y_mean`/`y_std`; this binds the float64 weight bytes by `np.frombuffer` into a flat dict
    keyed EXACTLY like `ValueMLP._params()` (W1 b1 W2 b2 [Wr1 br1 Wr2 br2] Wv bv [Wp bp]), which is what
    `forward_core` consumes. The residual block / policy head ride along automatically iff the manifest
    lists `Wr1` / `Wp` (P1: derive from the one authority, never re-author the layout — same derivation
    the C++ NetForward does). A malformed manifest is a loud failure (ADR-0002)."""
    m = json.loads(manifest_json)
    layout = m["layout"]
    params: dict[str, npt.NDArray[Any]] = {}
    for e in layout:
        shape = tuple(int(s) for s in e["shape"])
        count = int(np.prod(shape)) if shape else 1
        arr = np.frombuffer(blob, dtype=np.dtype(e["dtype"]), count=count, offset=int(e["off"]))
        params[e["name"]] = arr.reshape(shape).copy()   # copy: own writable arrays, not a blob view
    y_mean = float(m["y_mean"])
    y_std = float(m["y_std"])
    return params, y_mean, y_std


class ParamsSource(Protocol):
    """The version-gated weight RELOAD hook (seam 4) as a port, so the reload is MOCKABLE without redis.
    `current()` returns the live `(params, y_mean, y_std)`; `poll()` returns a FRESH triple iff the
    published version changed since the last load (else `None`, meaning keep the current params). The
    server calls `poll()` between batches and swaps in any non-None result — the ONE holder of weights,
    reloading only on a version change (design §3)."""

    def current(self) -> tuple[dict[str, npt.NDArray[Any]], float, float]:
        """The live `(params, y_mean, y_std)` — must be available before the loop serves any request."""
        ...

    def poll(self) -> tuple[dict[str, npt.NDArray[Any]], float, float] | None:
        """A fresh `(params, y_mean, y_std)` iff the published version changed since the last load, else
        `None`. Called between batches; a non-None result is swapped in as the new live params."""
        ...


class StaticParamsSource:
    """A `ParamsSource` holding ONE fixed param set injected directly — NO redis, NO broadcast. The
    default test path: the parity harness constructs the server with this so it serves a known net with
    no transport at all. `poll()` always returns `None` (the version never changes), so the loop never
    reloads."""

    def __init__(self, params: dict[str, npt.NDArray[Any]], y_mean: float, y_std: float) -> None:
        self._params = params
        self._y_mean = float(y_mean)
        self._y_std = float(y_std)

    def current(self) -> tuple[dict[str, npt.NDArray[Any]], float, float]:
        return self._params, self._y_mean, self._y_std

    def poll(self) -> tuple[dict[str, npt.NDArray[Any]], float, float] | None:
        return None


class RedisParamsSource:
    """The PRODUCTION `ParamsSource`: the version-gated weight broadcast (seam 4). Reads the published
    weights for `(run, phase, version)` via `transport.read_weights` and reconstructs the flat params
    (jax-free, `params_from_manifest_blob`). The server polls a live version supplier (e.g. the hp
    registry's published version) and reloads only when it advances — so one reload serves all leaves
    and every leaf in a batch sees one consistent net version (design §3).

    `read_weights` raises loudly on a missing payload (ADR-0002 — never a silent stale serve); a
    version that fails to load aborts the reload and the loop keeps serving the last-good params (a loud
    log, the operator can republish). The redis client is `Any` (the documented duck-typed bytes-store
    stub-gap, ADR-0012 P7 — same seam transport.py declares)."""

    def __init__(self, conn: Any, run: str, phase: str,
                 version_supplier: Callable[[], int], initial_version: int) -> None:
        self._conn = conn
        self._run = run
        self._phase = phase
        self._version_supplier = version_supplier
        self._loaded_version = initial_version
        from chocofarm.az import transport
        manifest, blob = transport.read_weights(conn, run, phase, initial_version)
        self._params, self._y_mean, self._y_std = params_from_manifest_blob(manifest, blob)

    def current(self) -> tuple[dict[str, npt.NDArray[Any]], float, float]:
        return self._params, self._y_mean, self._y_std

    def poll(self) -> tuple[dict[str, npt.NDArray[Any]], float, float] | None:
        want = int(self._version_supplier())
        if want == self._loaded_version:
            return None
        from chocofarm.az import transport
        manifest, blob = transport.read_weights(self._conn, self._run, self._phase, want)
        params, y_mean, y_std = params_from_manifest_blob(manifest, blob)
        self._params, self._y_mean, self._y_std = params, y_mean, y_std
        self._loaded_version = want
        return params, y_mean, y_std


class InferenceServer:
    """The imperative SHELL: a ZeroMQ ROUTER socket + the self-clocking greedy-drain loop + the
    version-gated reload between batches (design §3). Single-threaded by construction (JAX owns the
    forward; no XLA-in-a-worker-thread). Workers connect a REQ/DEALER socket and make a blocking
    `predict` RPC each; the SERVER batches whatever is concurrently in-flight.

    `forward_fn` defaults to the SSOT `forward_core`; the always-on test injects a stub to assert the
    drain/scatter without a real forward. `max_batch` caps the greedy drain so an unbounded burst can't
    build an oversized matmul; B self-scales with load below the cap (no latency timer to tune)."""

    # The bounded first-request poll interval (ms): the wakeup cadence at which an idle loop re-checks
    # `_stop`. A wakeup-to-recheck, not a spin — an idle server still parks at ~0 CPU between wakeups.
    _POLL_INTERVAL_MS = 100

    def __init__(self, params_source: ParamsSource, *, bind: str = "tcp://127.0.0.1:5599",
                 max_batch: int = 256, forward_fn: ForwardFn = forward_core,
                 context: "zmq.Context[zmq.Socket[bytes]] | None" = None) -> None:
        import zmq
        self._params_source = params_source
        self._max_batch = int(max_batch)
        self._forward_fn = forward_fn
        self._owns_context = context is None
        self._ctx: zmq.Context[zmq.Socket[bytes]] = context if context is not None else zmq.Context()
        self._sock: zmq.Socket[bytes] = self._ctx.socket(zmq.ROUTER)
        self._sock.bind(bind)
        self._poller: zmq.Poller = zmq.Poller()
        self._poller.register(self._sock, zmq.POLLIN)
        self._stop = False
        self._closed = False

    def _drain(self) -> list[DrainedRequest]:
        """Greedy-drain (design §3): BLOCK until ≥1 request is queued, then drain ALL currently-queued
        requests non-blocking up to `max_batch`. Each ROUTER frame is `[identity][empty][payload]`;
        the payload is decoded at the BOUNDARY (a malformed frame is rejected loudly — its identity is
        skipped from the batch, never zero-filled into the forward). Returns `[(identity, X), …]`, or an
        EMPTY list if it woke on the stop-check interval with nothing queued (the loop then re-checks
        `_stop` and re-blocks — a clean shutdown path, no socket killed from another thread).

        The block is a BOUNDED poll re-issued until a request arrives or `stop()` is observed: blocking
        forever on `timeout=None` would mean `stop()` could not wake the loop without closing the socket
        out from under a polling thread (a band-aid the bounded poll removes — ADR-0002/P5: fix the
        root, do not race the socket). The interval is long enough that an idle server still parks at
        ~0 CPU (it is a wakeup-to-recheck, not a spin)."""
        import zmq
        while not self._stop:
            # Bounded block for the FIRST request — self-clocks the batch to the load (B≈1 when idle),
            # and wakes every _POLL_INTERVAL_MS so a flipped `_stop` is observed promptly.
            if self._poller.poll(timeout=self._POLL_INTERVAL_MS):
                break
        if self._stop:
            return []
        drained: list[DrainedRequest] = []
        while len(drained) < self._max_batch:
            try:
                frames = self._sock.recv_multipart(flags=zmq.NOBLOCK)
            except zmq.Again:
                break   # nothing more currently queued — drain whatever accumulated, run the batch
            ident = frames[0]
            payload = frames[-1]
            try:
                X = decode_request(payload)
            except Exception as exc:   # malformed request: loud reject of THIS frame, batch unaffected
                self._reject(ident, exc)
                continue
            drained.append((ident, X))
        return drained

    def _reject(self, ident: bytes, exc: Exception) -> None:
        """A malformed request is rejected LOUDLY (ADR-0002): the server does not coerce it into a
        zero-filled forward. It logs the rejection; the client's RPC sees no valid response frame and
        raises on its own receive (a timeout / a decode failure), so the failure is not silent at either
        end. (The protocol carries no error frame, so the reject is a server-side drop + log.)"""
        print(f"[InferenceServer] rejecting malformed request: {exc}", flush=True)

    def _serve_batch(self, drained: list[DrainedRequest]) -> None:
        """Run ONE microbatch over the drained requests and scatter each encoded response back to its
        identity frame (`[identity][empty][response]`, the ROUTER reply envelope). Reloads params first
        if the published version changed (between-batch reload, design §3)."""
        reloaded = self._params_source.poll()
        params, y_mean, y_std = reloaded if reloaded is not None else self._params_source.current()
        for ident, resp in run_microbatch(self._forward_fn, params, y_mean, y_std, drained):
            self._sock.send_multipart([ident, b"", resp])

    def serve_forever(self) -> None:
        """The greedy-drain loop: block for ≥1 request, drain to the cap, run one forward, scatter,
        repeat. While batch K's forward runs, batch K+1 queues up — so B tracks demand automatically
        (no microbatch timer). Runs until `stop()` flips the flag (the in-process test spins this on a
        thread and stops it; a standalone process runs it forever)."""
        # Ensure params are available before serving (RedisParamsSource loads at construction;
        # StaticParamsSource holds the injected set) — a loud failure here beats a first-request stall.
        self._params_source.current()
        while not self._stop:
            drained = self._drain()
            if not self._stop and drained:
                self._serve_batch(drained)

    def stop(self) -> None:
        """Flip the stop flag so the loop exits at its next bounded-poll wakeup (≤ `_POLL_INTERVAL_MS`).
        The bounded poll in `_drain` observes this without the socket being closed from another thread —
        so the clean shutdown sequence is `stop()`, then `join()` the serve thread, then `close()`."""
        self._stop = True

    def close(self) -> None:
        """Close the ROUTER socket and (if we created it) terminate the context. Idempotent. Call AFTER
        `stop()` + joining the serve thread, so the loop is no longer touching the socket (the bounded
        poll guarantees the loop is between polls within `_POLL_INTERVAL_MS`, not mid-recv)."""
        if self._closed:
            return
        self._closed = True
        self._sock.close(linger=0)
        if self._owns_context:
            self._ctx.term()
