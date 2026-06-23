#!/usr/bin/env python3
"""
chocofarm/az/inference_server.py — the Shape B batched ZeroMQ inference SERVICE
(docs/design/zmq-inference-service.md §3; scaling-and-cpp-seam.md §2 — Axis A cross-episode batching).

The production leaf evaluator (design §0): a single Python process that holds the weights, batches
leaf-evaluation requests from N independent workers, and runs ONE `forward.forward_core` over the
stacked `(N_total, in_dim)` matrix — the SAME SSOT every Python path runs (R11; there is no second
transcription here). The per-leaf cost amortizes over the batch; the net's architecture stays free to
change because the wire (inference_wire.py) carries only float vectors and `(value, logits)`.

BATCHED WIRE (the firewall #2 lever): each request frame now carries B leaves (the batched
inference_wire frame; B=1 is the degenerate single-leaf case). The driver's strict gather-barrier sends
ALL its parked leaves in ONE message; the server decodes the `(B, in_dim)` matrix, stacks it with any
OTHER concurrently-queued requests' rows into one forward, and scatters each request's OWN B
predictions back as one batched response frame. This collapses the ~820k one-leaf frames into one
frame per gather-barrier — the per-leaf ZMQ+codec overhead that was the binding bottleneck.

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
from typing import TYPE_CHECKING, Any, Callable, Iterable, Protocol

import numpy as np
import numpy.typing as npt

from chocofarm.az.forward import forward_core
from chocofarm.az.inference_wire import decode_request, encode_response

# ---- optional protocol event log (gated by CHOCO_EVENTLOG; unset => zero behaviour change) ------------
# Injectable observability: when CHOCO_EVENTLOG names a path, the server appends monotonic-timestamped
# events `<mono_ns> SRV <kind> <fields>` at the protocol-affecting points (FWD, DRAIN). The C++ producer
# logs to its OWN file (CHOCO_EVENTLOG_CPP) on the SAME monotonic timebase (Python time.monotonic and C++
# std::chrono::steady_clock both read CLOCK_MONOTONIC, shared across processes on one Linux host), so
# tools/event_merge.py orders both into one timeline. BEST-EFFORT debug observability, deliberately NOT a
# fail-loud path (ADR-0002 governs transport correctness, not a debug log): a log error disables the stream,
# it never perturbs a forward. `_FWD_SEEN` records every forwarded row count, so a width unseen before (an
# overshoot / un-warmed XLA shape => a cold compile) is flagged `cold=1` — the XLA-churn signal.
import os
import time

_EVLOG_PATH = os.environ.get("CHOCO_EVENTLOG") or None
_evf = None
_FWD_SEEN: "set[int]" = set()


def _ev(kind: str, fields: str) -> None:
    global _evf
    if _EVLOG_PATH is None:
        return
    try:
        if _evf is None:
            _evf = open(_EVLOG_PATH, "a", buffering=1)
        _evf.write(f"{time.monotonic_ns()} SRV {kind} {fields}\n")
    except OSError:
        pass   # best-effort: a logging failure must never perturb the forward path

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
# The injected forward's contract: `(params, Xb_host, y_mean, y_std) -> one array`. The returned array is
# `(B, 1+n_actions)` — column 0 the DE-STANDARDIZED value, columns 1.. the raw logits — or `(B, 1)` for a
# value-only net. This single-homes the array path: ONE host→device hand-off (the cast happens INSIDE the
# forward, no separate eager `jnp.asarray` convert), the value de-standardized ON-device, and ONE
# device→host pull in `run_microbatch` (a single `np.asarray` over the whole block) — replacing the old
# `(v_std, logits)` two-pull tuple + a numpy de-standardize. The always-on test injects a stub returning
# the same shape over numpy.
ForwardFn = Callable[[dict[str, "npt.NDArray[Any]"], Any, float, float], Any]

# forward_core's OWN signature (params, X, xp) -> (v_std, logits|None) — the backend-polymorphic SSOT the
# jitted production forward composes. A typed alias so calling it in a typed context is not a
# no-untyped-call (forward_core itself carries no annotations).
ForwardCore = Callable[[dict[str, "npt.NDArray[Any]"], Any, Any], "tuple[Any, Any | None]"]
_FORWARD_CORE: ForwardCore = forward_core

_jit_forward_cache: list[Any] = []


def jit_forward_core(params: "dict[str, npt.NDArray[Any]]", Xb: Any, y_mean: float, y_std: float) -> Any:
    """The PRODUCTION forward (the InferenceServer default forward_fn): `forward_core` wrapped in ONE
    `jax.jit` that ALSO casts the host batch, de-standardizes the value, and packs `[v | logits]` into one
    device array — so the whole forward is a single compiled-graph call with ONE host→device crossing (the
    cast, folded in — no separate eager convert) and (via run_microbatch) ONE device→host pull. The server
    pads every batch to one shape, so it compiles a SINGLE executable. ADR-0012 P6: a numerically-
    equivalent reordering of the SAME `forward_core` (the ABS_TOL=1e-4 wire-parity bar holds, NOT byte
    identity); P1/P7: still the one `forward_core`, only wrapped — no second transcription. Built LAZILY so
    the module import stays jax-free (the hot-import discipline below); `Xb`/`y_mean`/`y_std` ride as traced
    ARGS so a same-shape weight+scale reload reuses the executable, no recompile."""
    if not _jit_forward_cache:
        import chocofarm.config  # noqa: F401 — XLA/OMP thread pin (SSOT) applied before jax initializes
        import jax
        import jax.numpy as jnp

        def _fwd(p: Any, x: Any, ym: Any, ys: Any) -> Any:
            v_std, logits = _FORWARD_CORE(p, x, jnp)
            v = jnp.reshape(v_std, (-1, 1)) * ys + ym            # de-standardize ON-device → (B, 1)
            return v if logits is None else jnp.concatenate([v, logits], axis=1)
        _jit_forward_cache.append(jax.jit(_fwd))
    return _jit_forward_cache[0](params, Xb, y_mean, y_std)


def build_staged_forward(params: "dict[str, npt.NDArray[Any]]", y_mean: float, y_std: float,
                         pad_to: int) -> ForwardFn:
    """Build the server's forward as a low-overhead `LowLatencyFn` whose PARAMS are staged DEVICE-RESIDENT
    ONCE (lowlatency.py — the SSOT dispatcher), returning a `ForwardFn`-shaped callable run_microbatch can
    call EXACTLY like `jit_forward_core`, but which re-transfers only `Xb` per call instead of the whole
    weight dict. This consolidates the ~45–53 µs per-forward params host→device re-transfer the
    `jit_forward_core` path repeats every call (the lever the real-MLP decomposition isolated — ADR-0012
    P7 cross-DEVICE / `bench_mlp_lowlatency.py`; bench fb9cfbc): the robust AOT handle holds the weights as
    device buffers and re-passes them with no re-transfer, so each forward pays only the (in-scope) input
    host→device + the one device→host pull run_microbatch already owns.

    The staged graph is the SAME production `[v | logits]` forward `jit_forward_core` jits — the ONE
    `forward_core` (P1/P7: no second transcription, only composed), de-standardized ON-device, packed
    `[v | logits]` — with the y-scale scalars FOLDED into the staged params closure (`p["_ym"]`/`p["_ys"]`)
    rather than ridden as the two traced args, because the handle stages one params pytree and varies only
    `x` (the `fn(params, x)` contract). Folding the scalars is a numerically-identical de-standardize (the
    same multiply-add), so the staged forward is allclose (ABS_TOL=1e-4, the project's forward bar; in
    practice byte-identical) to the `jit_forward_core` path on the same `(params, Xb)`. The folded params
    are a SHALLOW COPY (rebind-not-mutate, ADR-0001) — the caller's weight dict is never mutated.

    `pad_to` is the server's `max_batch`: run_microbatch pads every batch to `(pad_to, in_dim)` before the
    forward, so the handle compiles for that ONE fixed shape (one XLA executable, like jit_forward_core's
    single padded shape). `in_dim` is DERIVED from `params["W1"]` (P1 — the feature dim's one home; fail
    loud if absent). The lowlatency `device_put` lives INSIDE the dispatcher (the designated cross-DEVICE
    boundary, ADR-0012 P7) — this builder adds NO transfer call-site to the server's hot path.

    REBUILT per net reload (the version-gated swap rebinds `params`): the returned handle stages THIS
    version's weights, so the server rebuilds it whenever the params identity changes (see
    `InferenceServer._effective_forward`). A rebuild is a warm XLA-cache hit (~2.7 ms — the fixed-shape graph
    is already compiled; only the params re-stage), amortized over that version's many forwards. Built
    LAZILY (jax/lowlatency imported in-body) so the module import stays jax-free (the hot-import discipline
    above)."""
    import chocofarm.config  # noqa: F401 — XLA/OMP thread pin (SSOT) applied before jax initializes
    import jax.numpy as jnp

    from chocofarm.az.lowlatency import LowLatencyFn, compile_lowlatency, run

    if "W1" not in params:
        raise KeyError(
            "build_staged_forward: params has no 'W1' weight to derive in_dim from — cannot compile the "
            "staged forward for the fixed (pad_to, in_dim) shape (ADR-0002, refusing to guess the width).")
    in_dim = int(params["W1"].shape[0])

    def _fwd(p: Any, x: Any) -> Any:
        # The SAME production graph jit_forward_core jits, with the y-scale folded off the params closure
        # (p["_ym"]/p["_ys"]) so the de-standardize stays on-device while the signature fits fn(params, x).
        v_std, logits = _FORWARD_CORE(p, x, jnp)
        v = jnp.reshape(v_std, (-1, 1)) * p["_ys"] + p["_ym"]   # de-standardize ON-device → (B, 1)
        return v if logits is None else jnp.concatenate([v, logits], axis=1)

    # Fold the y-scale scalars into a SHALLOW copy of the weights (rebind-not-mutate — never touch the
    # caller's dict). f32 to match the inference precision the weights already carry (params_from_manifest_blob).
    staged_params: dict[str, Any] = dict(params)
    staged_params["_ym"] = np.float32(y_mean)
    staged_params["_ys"] = np.float32(y_std)

    # AOT-compile for the ONE fixed padded shape, staging the params device-resident at construction (the
    # ~170 ms cold compile is paid once here — at warmup/reload, off the per-forward path). The example x
    # pins the (pad_to, in_dim) signature run_microbatch always pads up to.
    example_x = np.zeros((pad_to, in_dim), dtype=np.float32)
    handle: "LowLatencyFn[Any, Any]" = compile_lowlatency(_fwd, staged_params, example_x)

    def staged_forward_fn(p: "dict[str, npt.NDArray[Any]]", Xb: Any, ym: float, ys: float) -> Any:
        # The ForwardFn contract: run_microbatch passes the host (params, Xb, y_mean, y_std); the staged
        # handle already holds THIS version's weights+scale device-resident, so the host `p`/`ym`/`ys` are
        # ignored and only `Xb` (the padded (pad_to, in_dim) host batch) is transferred — the params
        # re-transfer is gone. Returns the device `[v | logits]` block run_microbatch pulls once.
        return run(handle, Xb)

    return staged_forward_fn


# A batch row: (identity_frame, feature_matrix) — the ROUTER identity bytes to scatter the response
# back to, and the decoded float32 feature MATRIX (shape (B_i, in_dim) — the request's B leaves). This
# is `run_microbatch`'s VALUE-LEVEL input: the pure batching core knows only identities and matrices,
# never the transport envelope.
BatchRow = tuple[bytes, "npt.NDArray[np.float32]"]

# A drained request: (identity_frame, envelope_frames, feature_matrix). `envelope_frames` are the
# ROUTER frames BETWEEN the identity and the payload (`frames[1:-1]`) — the transport-routing envelope
# the server echoes back VERBATIM. For a REQ client this is the single empty delimiter `[b""]` (so the
# reply is byte-identical to the legacy `[identity][b""][resp]`); for a DEALER client carrying an
# 8-byte correlation id it is `[corr_id]`; for a bare single-frame DEALER it is `[]`. The server never
# PARSES the envelope — it round-trips it opaquely — so the correlation id needs no cross-language wire
# format (it is a transport concern, kept OUT of the value codec; ADR-0012 P7 serialization⊥transport).
DrainedRequest = tuple[bytes, list[bytes], "npt.NDArray[np.float32]"]


def run_microbatch(forward_fn: ForwardFn, params: dict[str, npt.NDArray[Any]],
                   y_mean: float, y_std: float,
                   requests: list[BatchRow], pad_to: int | None = None) -> list[tuple[bytes, bytes]]:
    """The PURE microbatch core (design §3): CONCATENATE the drained requests' feature MATRICES (each a
    request's B_i leaves) into one `(N_total, in_dim)` float32 matrix, run ONE `forward_fn(params, Xb,
    y_mean, y_std)` — which casts, runs the net, and DE-STANDARDIZES the value ON-device — then make ONE
    device→host pull of the returned `(rows, 1+n_actions)` block (column 0 the de-standardized value,
    columns 1.. the raw logits; `(rows, 1)` value-only) and SCATTER each request's OWN B_i predictions back
    as ONE batched response frame. Returns `[(identity, response_bytes), …]` aligned 1:1 with `requests`.

    This is the whole batching contract as a deterministic function of its inputs — no socket, no
    redis — so the always-on test asserts that requests collapse to ONE `forward_fn` call and each
    request gets ITS OWN rows back (the drain/concat/scatter logic), against a stub forward.

    `forward_fn` is the injected forward (the jitted `forward_core` in production; a stub in the test).
    The response carries the RAW logits (NOT softmaxed) + the de-standardized value — masking is
    client-side (design §2). A value-only net (no logits column) scatters `n_actions=0`. Refuses an
    empty batch (ADR-0002 — the loop only calls this with ≥1 drained request)."""
    if not requests:
        raise ValueError("run_microbatch called with an empty batch (the drain guarantees ≥1 request)")
    identities = [ident for ident, _ in requests]
    mats = [np.atleast_2d(m) for _, m in requests]
    in_dim = mats[0].shape[1]
    counts: list[int] = []
    for i, m in enumerate(mats):
        if m.ndim != 2 or m.shape[1] != in_dim:
            # A ragged batch (mixed in_dim) is a malformed mix the server must not silently pad/truncate
            # (ADR-0002 fail-loud). Every leaf of one net has the same feature dim by construction.
            raise ValueError(f"batched request {i} has shape {m.shape}, expected (B_i, {in_dim}) — ragged batch")
        counts.append(m.shape[0])
    Xb = np.concatenate(mats, axis=0).astype(np.float32, copy=False)   # (N_total, in_dim), the one input
    B = Xb.shape[0]
    # PAD to a single fixed shape (pad_to, in_dim) so XLA compiles ONE executable instead of one per
    # drained B (the per-B recompile that dominated the server profile). Padded rows are zero and the
    # forward is row-independent, so the real rows' outputs are byte-identical to the unpadded forward
    # (ADR-0012 P6) — only the first B are read back below. pad_to is the server's max_batch and the drain
    # caps the total at max_batch, so this only ever pads UP.
    if pad_to is not None and pad_to > B:
        Xb = np.concatenate([Xb, np.zeros((pad_to - B, in_dim), dtype=np.float32)], axis=0)
    # ONE host→device hand-off + de-standardize, folded into the forward; ONE device→host pull here. The
    # production forward casts Xb (no eager pre-`jnp.asarray`), de-standardizes on-device, and returns the
    # combined `(rows, 1+n_actions)` block; a value-only net returns `(rows, 1)`. The stub returns the same
    # over numpy. (The XLA pin + jax import live inside the jitted forward, ahead of its first jax touch.)
    _t0 = time.monotonic_ns() if _EVLOG_PATH is not None else 0
    out_arr = np.asarray(forward_fn(params, Xb, float(y_mean), float(y_std)), dtype=np.float32)
    if _EVLOG_PATH is not None:
        _w = int(Xb.shape[0])                        # the forwarded shape's row count (post-pad)
        _cold = _w not in _FWD_SEEN                  # unseen width => an XLA recompile this call
        _FWD_SEEN.add(_w)
        _ev("FWD", f"width={_w} real={B} cold={int(_cold)} dt_us={(time.monotonic_ns() - _t0) // 1000}")
    if out_arr.ndim != 2 or out_arr.shape[0] < B:
        raise ValueError(f"forward returned shape {out_arr.shape}, expected (>={B}, 1+n_actions)")
    v = out_arr[:, 0]
    has_logits = out_arr.shape[1] > 1
    out: list[tuple[bytes, bytes]] = []
    off = 0
    for ident, n in zip(identities, counts):
        v_rows = v[off:off + n]
        l_rows = out_arr[off:off + n, 1:] if has_logits else None
        out.append((ident, encode_response(v_rows, l_rows)))
        off += n
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
        # cast to f32 ONCE at load (ADR-0012 P1/P6): inference is float32 (the SSOT bar the C++/numpy
        # paths use), so the server must not carry f64 weights that force a per-forward f64->f32 recast
        # (and an f64 matmul). astype yields an owned, writable, contiguous f32 array.
        params[e["name"]] = arr.reshape(shape).astype(np.float32)
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

    `forward_fn` defaults to `jit_forward_core` (the SSOT `forward_core` wrapped in one `jax.jit`); the
    always-on test injects a stub to assert the
    drain/scatter without a real forward. `max_batch` caps the greedy drain so an unbounded burst can't
    build an oversized matmul; B self-scales with load below the cap (no latency timer to tune).

    PARAMS-STAGING (the cross-DEVICE consolidation — ADR-0012 P7 / bench fb9cfbc). With the DEFAULT
    forward, the server runs each forward through a `LowLatencyFn` (`build_staged_forward`) whose weights
    are staged DEVICE-RESIDENT for the live net version, so a forward re-transfers only the input batch,
    eliminating the ~45–53 µs/call weight-dict host→device re-transfer the plain `jit_forward_core` path
    repeats (measured ~79 µs/forward saved in the real run_microbatch path — the fixed-cost intercept
    drops, the per-row slope is unchanged; numbers under ~/w/vdc/chocobo/bench/run_microbatch_staging/).
    The staged handle is REBUILT on every version-gated reload (the reload rebinds a fresh params dict —
    ADR-0001 rebind-not-mutate — and `_effective_forward` re-stages on the identity change), so a forward
    never runs against a stale-version staged net (ADR-0002); the rebuild is a warm ~2.7 ms XLA-cache hit
    amortized over the version's forwards. A non-default injected forward_fn (a test stub) bypasses staging
    and is called directly — the pure run_microbatch seam is untouched.

    EXTENSION CONTRACT (the ADR-0012 P3 template-method split — lab-staging-divergence-rca.md §6). A serve
    is `_serve_batch` = the SEALED `_run_forward` (the WHOLE forward dispatch — params, the fixed-pad-gated
    `_effective_forward`, the pad shape, `run_microbatch`) then the OVERRIDABLE `_scatter` boundary hook. A
    subclass (`StageAServer`, `LabServer`) extends behaviour through `_scatter` (the serve/scatter — counters,
    the lab's Controller call + gate-frame tagging) and the two FOCUSED dispatch hooks `_pad_shape`
    (per-forward pad policy) + `_forward_groups` (the drained→forwards partition); it does **NOT** override
    `_serve_batch`/`_run_forward`. So a subclass never sees `run_microbatch` and CANNOT re-author (and
    silently diverge) the forward — the override-divergence bug class that motivated this split is
    unrepresentable by construction.

    `min_forward_rows` (θ) and `max_queue_delay_ms` arm the optional server-side coalescing FLOOR
    (server-floor-design.md — increment ii of cpp-eval-transport-adapter.md §6), DEFAULT OFF (θ=0): with
    θ>0 the drain accumulates across producer threads until ≥θ rows or the bounded delay, lifting
    cross-thread rows/forward to amortize the fixed per-forward cost. Off by default the drain is the
    byte-unchanged greedy production path (see `_drain`)."""

    # The bounded first-request poll interval (ms): the wakeup cadence at which an idle loop re-checks
    # `_stop`. A wakeup-to-recheck, not a spin — an idle server still parks at ~0 CPU between wakeups.
    _POLL_INTERVAL_MS = 100

    # The FIXED-PAD STAGING PREDICATE (the structural invariant the lab-staging-divergence RCA distilled).
    # The staged single-shape AOT handle (`build_staged_forward`) is AOT-compiled for ONE fixed
    # `(pad_to=max_batch, in_dim)` shape, so it is valid ONLY for a server that pads EVERY forward to that
    # one fixed shape (pad-to-max). A server that snaps the pad per-forward to a bucket (StageAServer's
    # bucket-E) feeds the handle `[64,…]`/`[256,…]` and the single-shape executable rejects it
    # (`TypeError: compiled with float32[512,241] and called with float32[64,241]` — the interim re-align's
    # crash, 12b27bf). This class flag encodes "you can only stage a FIXED-pad server" as a structural
    # invariant `_effective_forward`/`_run_forward` check: True here (InferenceServer pads to max_batch),
    # overridden False by a bucketing subclass. So the lab-alignment NEXT step is a CLEAN FLIP — a bucketing
    # server that adopts pad-to-max sets this True and is auto-staged, no crash — and a bucketing server can
    # never accidentally run the single-shape handle against a mismatched bucket (the divergence becomes
    # unrepresentable by construction, not caught by a runtime crash).
    _uses_fixed_pad: bool = True

    def __init__(self, params_source: ParamsSource, *, bind: str = "tcp://127.0.0.1:5599",
                 max_batch: int = 256, forward_fn: ForwardFn = jit_forward_core,
                 min_forward_rows: int = 0, max_queue_delay_ms: float = 0.0,
                 context: "zmq.Context[zmq.Socket[bytes]] | None" = None) -> None:
        self._max_batch = int(max_batch)
        # The server-side coalescing floor (server-floor-design.md, increment ii of
        # cpp-eval-transport-adapter.md §6). Default OFF (θ=0): the drain stays the byte-unchanged greedy
        # drain (production path untouched). When θ>0 the drain ACCUMULATES across producer threads until
        # ≥θ rows or `max_queue_delay_ms` elapses — lifting cross-thread rows/forward so the server
        # amortizes its large fixed per-forward cost over a fuller batch. Live cells read per-drain on
        # `self` (ADR-0012 P4: a swept tunable lives where it can breathe — read at the point of use, never
        # baked into a closure), so a sweep need only set the attribute. Fail loud on a value the drain
        # cannot honor (ADR-0002 / P2: a θ above the max_batch cap is a lying knob — the cap would forbid
        # ever reaching it; reject before binding the socket).
        if min_forward_rows < 0:
            raise ValueError(f"min_forward_rows must be ≥ 0 (θ=0 disables the floor), got {min_forward_rows}")
        if min_forward_rows > self._max_batch:
            raise ValueError(
                f"min_forward_rows={min_forward_rows} exceeds max_batch={self._max_batch}: the drain cap "
                f"forbids ever reaching θ, so the floor would only ever fire at the delay (ADR-0002 — a "
                f"config the receiver cannot honor must not be silently accepted; keep θ ≤ max_batch).")
        if max_queue_delay_ms < 0:
            raise ValueError(f"max_queue_delay_ms must be ≥ 0, got {max_queue_delay_ms}")
        self._min_forward_rows = int(min_forward_rows)
        self._max_queue_delay_ms = float(max_queue_delay_ms)
        import zmq
        self._params_source = params_source
        self._forward_fn = forward_fn
        # Params-staging (the cross-DEVICE consolidation — ADR-0012 P7 / bench fb9cfbc): when the default
        # production forward (`jit_forward_core`) is in use, the server runs the forward through a
        # `LowLatencyFn` (build_staged_forward) that holds THIS net version's weights device-resident, so a
        # forward re-transfers only the input batch, not the ~45–53 µs/call weight dict. `_staged_fn` is the
        # built handle's ForwardFn (None until warmup/first serve builds it); `_staged_params_id` is the
        # IDENTITY of the params object it was staged from, so the version-gated reload (which REBINDS a new
        # params dict — ADR-0001 rebind-not-mutate) is detected by `is` and the handle rebuilt (a warm
        # ~2.7 ms XLA-cache hit, amortized over the version's forwards) — never a forward against a stale
        # staged net (ADR-0002: a stale-net serve is a loud-failure class, here closed by rebuild-on-rebind).
        # A non-default injected forward_fn (a test stub) bypasses staging: `_stages_params` is False and the
        # injected fn is called directly, exactly as before (the pure run_microbatch seam is untouched).
        self._stages_params = forward_fn is jit_forward_core
        self._staged_fn: ForwardFn | None = None
        self._staged_params_id: int | None = None
        self._owns_context = context is None
        self._ctx: zmq.Context[zmq.Socket[bytes]] = context if context is not None else zmq.Context()
        self._sock: zmq.Socket[bytes] = self._ctx.socket(zmq.ROUTER)
        self._sock.bind(bind)
        self._poller: zmq.Poller = zmq.Poller()
        self._poller.register(self._sock, zmq.POLLIN)
        self._stop = False
        self._closed = False
        # A request held over from a drain that would have straddled the max_batch cap (set in `_drain`,
        # consumed at the next drain's start) — so the cap is honored ACROSS drains without dropping it.
        self._pending: "DrainedRequest | None" = None

    def _drain(self) -> list[DrainedRequest]:
        """Greedy-drain (design §3): BLOCK until ≥1 request is queued, then drain ALL currently-queued
        requests non-blocking up to `max_batch`. Each ROUTER frame is `[identity][envelope…][payload]`;
        the identity is `frames[0]`, the payload is `frames[-1]`, and the ENVELOPE is everything between
        (`frames[1:-1]` — a REQ delimiter, a DEALER correlation id, or nothing — captured opaquely and
        echoed VERBATIM in the reply, never parsed). The payload is decoded at the BOUNDARY (a malformed
        frame is rejected loudly — its identity is skipped from the batch, never zero-filled into the
        forward). Returns `[(identity, envelope, X), …]`, or an
        EMPTY list if it woke on the stop-check interval with nothing queued (the loop then re-checks
        `_stop` and re-blocks — a clean shutdown path, no socket killed from another thread).

        The block is a BOUNDED poll re-issued until a request arrives or `stop()` is observed: blocking
        forever on `timeout=None` would mean `stop()` could not wake the loop without closing the socket
        out from under a polling thread (a band-aid the bounded poll removes — ADR-0002/P5: fix the
        root, do not race the socket). The interval is long enough that an idle server still parks at
        ~0 CPU (it is a wakeup-to-recheck, not a spin).

        The server-side coalescing FLOOR (server-floor-design.md — increment ii): with `min_forward_rows`
        (θ) > 0, after the first request lands the drain KEEPS accumulating across producer threads —
        re-draining whatever is queued, then briefly waiting for the next arrival — until ≥θ rows are
        gathered OR `max_queue_delay_ms` elapses, then runs one forward. This lifts cross-thread
        rows/forward so the server amortizes its large fixed per-forward cost over a fuller batch. θ=0
        (the default) degenerates to a SINGLE greedy NOBLOCK pass — byte-identical to the production
        drain, no clock read taken (the production path is untouched). `max_queue_delay_ms` is a HARD
        bound (the escape hatch): the forward fires within it of the first request whether or not θ is
        reached, so there is no "wait forever for θ rows that never come" wedge — the brief waits are
        bounded by the remaining delay, never an unbounded block (the producer-floor refutation's lesson:
        no new wedge, ADR-0002/P5)."""
        import zmq
        while not self._stop:
            # Bounded block for the FIRST request — self-clocks the batch to the load (B≈1 when idle),
            # and wakes every _POLL_INTERVAL_MS so a flipped `_stop` is observed promptly.
            if self._poller.poll(timeout=self._POLL_INTERVAL_MS):
                break
        if self._stop:
            return []
        drained: list[DrainedRequest] = []
        total_rows = 0   # the concatenated row count across drained requests (each carries B_i leaves)
        # A request held over from the PRIOR drain (it would have straddled the max_batch cap) leads this
        # batch — so the cap is honored ACROSS drains and the request is never dropped, only deferred one
        # forward (it keeps its 1:1 reply). This restores the invariant the docstring and run_microbatch both
        # assert ("the drain caps the total at max_batch, so this only ever pads UP"): overcommit beyond the
        # cap coalesces across forwards instead of building one oversized, uncompiled forward (ADR-0002).
        if self._pending is not None:
            drained.append(self._pending)
            total_rows += int(self._pending[2].shape[0])
            self._pending = None
        theta = self._min_forward_rows                            # live floor target (P4 — read per-drain)
        deadline_ns = (time.monotonic_ns() + int(self._max_queue_delay_ms * 1_000_000)) if theta > 0 else 0
        deferred = False   # a straddling request was held over — forward the capped batch now, do not wait
        while True:
            # Drain everything CURRENTLY queued (NOBLOCK), up to the max_batch cap. The cap GENUINELY bounds
            # the matmul: a request that would push past it is deferred whole (below), so even an unbounded
            # overcommit burst cannot build an oversized forward.
            while total_rows < self._max_batch:
                try:
                    frames = self._sock.recv_multipart(flags=zmq.NOBLOCK)
                except zmq.Again:
                    break   # nothing more currently queued — drain whatever accumulated
                ident = frames[0]
                envelope = frames[1:-1]   # transport-routing frames (REQ delimiter / DEALER corr-id / none)
                payload = frames[-1]
                try:
                    X = decode_request(payload)   # a (B_i, in_dim) matrix (B_i ≥ 1; B_i=1 is single-leaf)
                except Exception as exc:   # malformed request: loud reject of THIS frame, batch unaffected
                    self._reject(ident, exc)
                    continue
                rows = int(X.shape[0])
                if rows > self._max_batch:
                    # A SINGLE request wider than the AOT-compiled forward cannot be padded down or split
                    # here — reject it LOUDLY (ADR-0002) rather than hand the fixed-shape forward an oversized
                    # matmul (the cryptic XLA shape-crash). Chunked-forward handling for this case: BACKLOG.
                    self._reject(ident, ValueError(
                        f"request of {rows} rows exceeds max_batch={self._max_batch}: the forward is compiled "
                        f"for one fixed width and cannot take an oversized single request"))
                    continue
                if drained and total_rows + rows > self._max_batch:
                    # Would straddle the cap: defer it WHOLE to the next drain, forward the capped batch now.
                    self._pending = (ident, envelope, X)
                    deferred = True
                    break
                drained.append((ident, envelope, X))
                total_rows += rows
            # Floor decision: forward now if a request was deferred (the batch is capped), the floor is OFF, θ
            # is reached, or the cap is hit; otherwise block BRIEFLY (bounded by the remaining delay) for the
            # next producer's RTT to land, re-drain.
            if deferred or theta <= 0 or total_rows >= theta or total_rows >= self._max_batch:
                break
            remaining_ns = deadline_ns - time.monotonic_ns()
            if remaining_ns < 1_000_000:   # <1ms of the hard delay left — forward now, never spin the tail
                break
            self._poller.poll(timeout=remaining_ns // 1_000_000)   # ≤ remaining delay; the deadline caps it
        if _EVLOG_PATH is not None and drained:
            _ev("DRAIN", f"msgs={len(drained)} rows={total_rows} floor={theta}")
        return drained

    def _reject(self, ident: bytes, exc: Exception) -> None:
        """A malformed request is rejected LOUDLY (ADR-0002): the server does not coerce it into a
        zero-filled forward. It logs the rejection; the client's RPC sees no valid response frame and
        raises on its own receive (a timeout / a decode failure), so the failure is not silent at either
        end. (The protocol carries no error frame, so the reject is a server-side drop + log.)"""
        print(f"[InferenceServer] rejecting malformed request: {exc}", flush=True)

    def _effective_forward(self, params: dict[str, npt.NDArray[Any]],
                           y_mean: float, y_std: float) -> ForwardFn:
        """The forward `run_microbatch` should call for THIS set of live params — the params-staging seam.

        When staging is on (the default `jit_forward_core` forward), return a `LowLatencyFn`-backed forward
        whose weights are staged DEVICE-RESIDENT for this net version, REBUILDING it whenever the params
        object identity has changed since it was last built. The version-gated reload rebinds a FRESH params
        dict (ADR-0001 rebind-not-mutate; `RedisParamsSource.poll/current` return a new object on reload and
        the same object between reloads), so `id(params) != self._staged_params_id` is exactly "the net
        reloaded" — the trigger to re-stage. (Keying on the object identity is sound under the single-
        threaded sequential serve: the prior version's dict is held live by the source as `self._params`
        when the reload allocates the new one, so adjacent versions never collide on a reused id — the same
        identity-coherence pattern ADR-0001 uses for the f32 inference cache.) This closes the
        stale-staged-net hazard structurally (ADR-0002:
        a forward never runs against a previous version's staged weights — the rebuild is wired into the
        same call that fetches the live params). The rebuild compiles for the ONE fixed padded shape
        `(max_batch, in_dim)` run_microbatch always pads to, so it is a warm XLA-cache hit (~2.7 ms), not a
        cold compile; that cost is paid once per version and amortized over the version's many forwards.

        When staging is off (a non-default injected forward_fn — a test stub) OR the server does NOT pad to
        a FIXED shape (`_uses_fixed_pad` is False — a bucketing subclass, whose per-forward pad varies and so
        cannot use the single-shape AOT handle), return the injected `self._forward_fn` unchanged: the pure
        run_microbatch seam is honored exactly as before, no staging interposed. The `_uses_fixed_pad` clause
        is the structural invariant the lab-staging-divergence RCA distilled — a bucketing server feeding the
        single-shape staged handle a non-max bucket is the crash the interim re-align hit; gating staging on
        the fixed-pad predicate makes that mismatch unrepresentable (the seam decides correctly by
        construction, not by a runtime shape check)."""
        if not (self._stages_params and self._uses_fixed_pad):
            return self._forward_fn
        if self._staged_fn is None or self._staged_params_id != id(params):
            self._staged_fn = build_staged_forward(params, y_mean, y_std, pad_to=self._max_batch)
            self._staged_params_id = id(params)
        return self._staged_fn

    # ---- the template-method split (ADR-0012 P3 / lab-staging-divergence-rca.md §6.1) ----
    # `_serve_batch` formerly WELDED two orthogonal concerns into one overridable method: (a) the FORWARD
    # DISPATCH (which params, staged-or-not, the pad shape, `run_microbatch`) and (b) the SERVE/SCATTER
    # boundary (envelope echo; the lab's Controller call + gate-frame tagging). A subclass that legitimately
    # needs to extend (b) was COMPELLED to override the whole method and HAND-COPY (a) — the divergence
    # lineage the RCA traced (the consolidation upgraded the base's dispatch but not the two hand-copies).
    # The split makes the dispatch a SEALED seam (`_run_forward`) no subclass overrides and the boundary an
    # OVERRIDABLE hook (`_scatter`): a subclass extends (b) WITHOUT ever seeing `run_microbatch`, so the
    # override-divergence bug class becomes UNREPRESENTABLE (there is no dispatch to re-author). The two
    # focused hooks the dispatch reads — `_pad_shape` (the per-forward pad policy) and `_forward_groups`
    # (how the drained batch partitions into forwards) — let a subclass vary the pad/grouping WITHOUT
    # touching the dispatch. Confirm: the ONE remaining `run_microbatch(...)` call-site is inside
    # `_run_forward` below.

    def _pad_shape(self, real: int) -> int:
        """The per-forward PAD shape: pad a forward of `real` concatenated rows up to this width before the
        matmul (one fixed XLA shape so the executable is reused). The base server pads to `max_batch`
        (pad-to-max — the production policy; with `_uses_fixed_pad` True this is the ONE fixed shape the
        staged AOT handle compiles for). A FOCUSED hook: a bucketing subclass overrides ONLY this (snap up
        to a compiled bucket), never the whole dispatch (ADR-0012 P3 — the pad policy is one axis, the
        dispatch another)."""
        return self._max_batch

    def _forward_groups(self, drained: list[DrainedRequest]) -> list[list[DrainedRequest]]:
        """How the drained batch PARTITIONS into forwards. The base (and the production path) runs ONE
        forward over ALL drained rows — `[drained]`, a single group — so a drained burst self-clocks into one
        matmul. A FOCUSED hook returning a PARTITION (data, not dispatch logic): a bench subclass that wants
        per-leaf forwards overrides ONLY this to return `[[d] for d in drained]`, and the SEALED dispatch
        runs the identical forward over each part — the subclass still cannot re-author which-forward /
        staged-or-not / `run_microbatch` (the divergence the split forbids)."""
        return [drained]

    def _run_forward(
        self, drained: list[DrainedRequest]
    ) -> "tuple[list[tuple[bytes, bytes]], list[tuple[int, int]]]":
        """The SEALED FORWARD-DISPATCH seam — the ONE owner of the whole forward dispatch, which NO subclass
        overrides. It: (1) reloads params iff the published version changed, else reads the live params
        (between-batch reload, design §3); (2) resolves the forward for THIS version's params via
        `_effective_forward`, re-staging the device-resident weights iff the reload rebound a new params
        object (the stale-staged-net guard, ADR-0002) AND the server pads to a fixed shape (the
        `_uses_fixed_pad` invariant — a bucketing server gets the un-staged `self._forward_fn`, since the
        single-shape AOT handle is valid only for the fixed pad); (3) for each forward group (`_forward_groups`)
        runs ONE `run_microbatch` at that forward's pad shape (`_pad_shape`).

        Returns `(responses, forwards)`: `responses` is the encoded `(identity, response_bytes)` list 1:1
        with `drained` in drained order (so `_scatter` re-pairs each with its envelope by position — both the
        one-group and the per-leaf partition preserve drained order); `forwards` is the per-forward
        `(real_rows, pad)` metadata (one entry per group — a single entry for the production one-group path)
        the boundary hook needs for its counters / reward / observe (the small metadata contract the RCA
        §6.1 sanctions — a reason to SPLIT the dispatch from the boundary, not to keep them fused). The host
        params/scale stay LOCAL to this seam (the lab reads them only through this return), so a subclass
        never re-fetches them to hand-roll a forward."""
        reloaded = self._params_source.poll()
        params, y_mean, y_std = reloaded if reloaded is not None else self._params_source.current()
        # Resolve the forward for THIS version's params, re-staging the device-resident weights iff the
        # reload rebound a new params object AND this server pads to a fixed shape (the stale-staged-net guard
        # + the fixed-pad invariant both live here, in the same call that fetched the live params — ADR-0002).
        # Off staging (or on a bucketing server) this is just the injected forward_fn.
        forward_fn = self._effective_forward(params, y_mean, y_std)
        responses: list[tuple[bytes, bytes]] = []
        forwards: list[tuple[int, int]] = []
        for group in self._forward_groups(drained):
            rows = [(ident, X) for ident, _envelope, X in group]
            real = int(sum(int(X.shape[0]) for _ident, X in rows))
            pad = self._pad_shape(real)
            responses.extend(run_microbatch(forward_fn, params, y_mean, y_std, rows, pad_to=pad))
            forwards.append((real, pad))
        return responses, forwards

    def _scatter(self, drained: list[DrainedRequest], responses: "list[tuple[bytes, bytes]]",
                 forwards: "list[tuple[int, int]]") -> None:
        """The OVERRIDABLE SERVE/SCATTER boundary hook. The base scatters each encoded response back to its
        identity, ECHOING that request's transport envelope verbatim (`[identity][envelope…][response]`). The
        envelope is opaque routing the server round-trips unchanged: for a REQ client it is the empty
        delimiter (so the reply is byte-identical to the legacy `[identity][b""][resp]`), for a DEALER client
        it carries the correlation id the C++ pool matches replies on. `responses` come back 1:1 in drained
        order and re-pair with their envelopes by position. A subclass OVERRIDES THIS (and only this) to
        extend the boundary — bench counters, the lab's Controller call + gate-frame tagging — WITHOUT ever
        touching the sealed forward dispatch (`forwards` carries the per-forward `(real, pad)` it needs)."""
        for (ident, resp), (_ident, envelope, _X) in zip(responses, drained):
            self._sock.send_multipart([ident, *envelope, resp])

    def _serve_batch(self, drained: list[DrainedRequest]) -> None:
        """Serve ONE drained batch: run the SEALED forward dispatch, then the OVERRIDABLE scatter boundary.
        FINAL by convention (ADR-0012 P3) — no subclass overrides `_serve_batch` or `_run_forward`; a
        subclass extends behaviour through `_scatter` (+ `_pad_shape`/`_forward_groups`). This is the
        structural fix the lab-staging-divergence RCA named: the forward dispatch has ONE home, so it cannot
        be hand-copied (and silently diverge) into a subclass override."""
        responses, forwards = self._run_forward(drained)
        self._scatter(drained, responses, forwards)

    def warmup(self, batch_sizes: Iterable[int]) -> None:
        """PRE-COMPILE the XLA kernels for each batch size B the wire path can produce, BEFORE the loop
        serves a single real request (ADR-0009 measure-honesty).

        Why this exists (the confound it removes): `run_microbatch` does `np.stack(rows) -> (B, in_dim)`
        and runs the jitted `forward_core`, so XLA compiles ONCE PER EXACT BATCH SIZE B. The greedy drain
        (`_drain`) yields B in 1..max_batch depending on instantaneous load, so a COLD server JIT-compiles
        each new B *inside the first timed iterations* — a per-B compile latency the host/VM are otherwise
        idle for. That cold-compile cost has been mis-read as run-to-run "jitter" and has poisoned the
        before/after DPS numbers. Forcing every reachable B to compile up front makes the first real
        generation's throughput equal to steady state, so any measured delta is a REAL property of the
        code under test, not a JIT artifact.

        Mechanism (P1/P7 — REUSE, do not re-author the forward): for each B this runs the SAME forward
        path the loop runs — `run_microbatch(self._forward_fn, params, y_mean, y_std, [rows…])` over a
        dummy `(B, in_dim)` zero matrix — and blocks on the result by reading it back (`run_microbatch`
        already does `np.asarray` on the forward outputs, so its return forces XLA compilation to
        COMPLETE, not merely enqueue). No socket, no scatter to a client — the responses are discarded.

        `in_dim` is DERIVED from the live params' `W1` first dim via `_params_source.current()` (P1 — the
        feature dim has one home, the net's first weight; never a hardcoded width). Fail-loud on any error
        (ADR-0002): a missing `W1`, a bad B, or a forward failure raises here at standup, where it is a
        loud pre-flight abort, not a silent first-request stall."""
        params, y_mean, y_std = self._params_source.current()
        if "W1" not in params:
            raise KeyError(
                "InferenceServer.warmup: params has no 'W1' weight to derive in_dim from — cannot build a "
                "dummy batch matching the forward's input shape (ADR-0002, refusing to guess the width).")
        in_dim = int(params["W1"].shape[0])
        # Build (and cache) the staged forward for the current params BEFORE timing, so the ~170 ms cold
        # AOT compile + the device params staging happen HERE at standup, off the first real request's path
        # (with staging off this returns the injected forward_fn unchanged). The serve loop reuses this same
        # staged handle (same params identity) until a reload rebinds new weights.
        forward_fn = self._effective_forward(params, y_mean, y_std)
        for b in batch_sizes:
            b = int(b)
            if b < 1:
                raise ValueError(f"InferenceServer.warmup: batch size {b} < 1 (the drain produces B≥1).")
            # ONE request carrying a (b, in_dim) matrix — concatenated + padded to (max_batch, in_dim).
            rows: list[BatchRow] = [(b"", np.zeros((b, in_dim), dtype=np.float32))]
            # run_microbatch returns encoded response frames built from np.asarray(forward outputs) — the
            # asarray blocks until XLA has actually compiled+run shape (max_batch, in_dim). Discard them.
            run_microbatch(forward_fn, params, y_mean, y_std, rows, pad_to=self._max_batch)

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
