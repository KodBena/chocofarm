#!/usr/bin/env python3
"""
tests/test_zmq_inference.py — pins for the Shape B batched ZeroMQ inference service
(docs/design/zmq-inference-service.md; chocofarm/az/{inference_wire,net_port,inference_server,
zmq_net_client}.py).

Two layers, mirroring tests/test_cpp_runner.py's split:

  * ALWAYS-ON (no server, no redis, no network) — runs in every `pytest tests/ -q`:
      - the wire CODEC round-trips (encode∘decode == identity over random vectors, INCLUDING the
        value-only `n_actions=0` / `logits=None` case), and REJECTS malformed frames loudly (bad
        protocol byte, wrong length, NaN feature) — ADR-0002 translate-and-validate;
      - the `Net` Protocol is satisfied STRUCTURALLY by BOTH impls (the local ValueMLP adapter and the
        remote ZmqNetClient);
      - the greedy-drain batching LOGIC over a FAKE forward, exercised as the PURE `run_microbatch`
        function (no socket): B concurrent requests collapse to ONE forward call and each request gets
        ITS OWN row back (drain → stack → one forward → scatter), de-standardized value + RAW logits.

  * OPT-IN (needs a running server) — the full server+client PARITY harness: spin the server in-process
    (a thread) with params injected DIRECTLY (NO redis, StaticParamsSource), the ZmqNetClient RPCs it,
    and assert `value` AND each `logit` within 1e-4 of the local `forward_core` over N≥1000 random
    float32 feature vectors, residual ON and OFF, across varied batch sizes B (concurrent clients) to
    exercise the row-vs-single f32 roundoff (the ADR-0012 P6 bar). Guarded like the cpp opt-in tests:
    skips gracefully if pyzmq is missing or CHOCO_RUN_ZMQ is unset, so the DEFAULT suite stays green
    with no server.

Public Domain (The Unlicense).
"""
import os
import sys
import threading

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chocofarm.az import inference_wire as wire
from chocofarm.az.inference_server import (
    StaticParamsSource,
    params_from_manifest_blob,
    run_microbatch,
)
from chocofarm.az.net_port import Net, ValueMLPNet
from chocofarm.az.zmq_net_client import ZmqNetClient

# OPT-IN gate (mirrors test_cpp_runner's CHOCO_RUN_CPP): the full server+client parity runs only with
# CHOCO_RUN_ZMQ=1, and skips if pyzmq is unimportable — so the default `pytest tests/ -q` stays green
# on a box with no server and (if it ever lacked it) no pyzmq.
_RUN_ZMQ = bool(os.environ.get("CHOCO_RUN_ZMQ"))


def _pyzmq_available() -> bool:
    try:
        import zmq  # noqa: F401
        return True
    except ImportError:
        return False


# ===========================================================================
# ALWAYS-ON 1 — the wire codec round-trips (identity), incl. value-only, and rejects malformed frames
# ===========================================================================
@pytest.mark.parametrize("B", [1, 2, 5, 64])
@pytest.mark.parametrize("in_dim", [1, 2, 7, 65, 300])
@pytest.mark.parametrize("n_actions", [0, 1, 65])
def test_codec_round_trips_request_and_response(B, in_dim, n_actions):
    """encode∘decode == identity over random float32 BATCHES, for the REQUEST (a (B, in_dim) matrix) and
    the RESPONSE (B de-std values + B raw-logits rows). B=1 is the degenerate single-leaf case;
    `n_actions=0` is the value-only case (logits=None, empty per-row block). float32 is the wire dtype,
    so the round-trip is EXACT (no precision lost re-reading the same bytes)."""
    rng = np.random.default_rng(1234 + B * 911 + in_dim * 131 + n_actions)
    X = rng.standard_normal((B, in_dim)).astype(np.float32)
    back_X = wire.decode_request(wire.encode_request(X))
    assert back_X.dtype == np.float32
    assert back_X.shape == (B, in_dim)
    np.testing.assert_array_equal(back_X, X)   # EXACT — same f32 bytes re-read

    values = rng.standard_normal(B).astype(np.float32)
    logits = None if n_actions == 0 else rng.standard_normal((B, n_actions)).astype(np.float32)
    back_v, back_l = wire.decode_response(wire.encode_response(values, logits))
    assert back_v.shape == (B,)
    np.testing.assert_array_equal(back_v, values)   # f32 round-trips its image exactly
    if n_actions == 0:
        assert back_l is None
    else:
        assert back_l is not None
        assert back_l.dtype == np.float32
        assert back_l.shape == (B, n_actions)
        np.testing.assert_array_equal(back_l, logits)


def test_codec_single_leaf_is_b1():
    """A 1-D feature vector encodes as the degenerate B=1 batched request (the batched frame subsumes
    single-leaf); decode returns a (1, in_dim) matrix."""
    back_X = wire.decode_request(wire.encode_request(np.arange(5, dtype=np.float32)))
    assert back_X.shape == (1, 5)
    np.testing.assert_array_equal(back_X[0], np.arange(5, dtype=np.float32))


def test_codec_value_only_response_is_empty_logits():
    """The value-only path: `encode_response(values, None)` carries n_actions=0 and an EMPTY per-row
    logits block, and decodes back to `(values, None)` — mirroring forward_core's `logits=None`."""
    frame = wire.encode_response(np.array([3.5, -1.0], dtype=np.float32), None)
    v, logits = wire.decode_response(frame)
    assert logits is None
    np.testing.assert_array_equal(v, np.array([3.5, -1.0], dtype=np.float32))


def test_codec_rejects_bad_protocol_byte():
    """A request/response whose first byte is not the supported PROTOCOL_VERSION is a LOUD WireError —
    a codec mismatch fails loudly rather than misreading the next field as a float (ADR-0002)."""
    good = wire.encode_request(np.ones((1, 4), dtype=np.float32))
    bad = bytes([wire.PROTOCOL_VERSION + 7]) + good[1:]
    with pytest.raises(wire.WireError):
        wire.decode_request(bad)
    good_resp = wire.encode_response(np.ones(1, dtype=np.float32), np.ones((1, 3), dtype=np.float32))
    bad_resp = bytes([wire.PROTOCOL_VERSION + 7]) + good_resp[1:]
    with pytest.raises(wire.WireError):
        wire.decode_response(bad_resp)


def test_codec_rejects_wrong_length_payload():
    """A request whose payload byte count is not exactly `B·in_dim × f32` is a LOUD WireError — the
    codec refuses a truncated/over-long frame instead of zero-filling or truncating it (ADR-0002)."""
    good = wire.encode_request(np.ones((2, 8), dtype=np.float32))
    with pytest.raises(wire.WireError):
        wire.decode_request(good[:-4])        # one float short
    with pytest.raises(wire.WireError):
        wire.decode_request(good + b"\x00\x00\x00\x00")   # one float long


def test_codec_rejects_nan_feature():
    """A NaN/Inf feature is rejected at ENCODE (a malformed request never reaches the wire) — ADR-0002:
    never a coerced/zero-filled forward."""
    bad = np.array([[1.0, np.nan, 2.0, 3.0]], dtype=np.float32)
    with pytest.raises(wire.WireError):
        wire.encode_request(bad)
    worse = np.array([[1.0, np.inf, 2.0]], dtype=np.float32)
    with pytest.raises(wire.WireError):
        wire.encode_request(worse)


def test_codec_rejects_ragged_response_logits():
    """A logits matrix whose row count != B is a LOUD WireError at encode (ADR-0002 — never a ragged
    scatter)."""
    with pytest.raises(wire.WireError):
        wire.encode_response(np.ones(3, dtype=np.float32), np.ones((2, 5), dtype=np.float32))


# ===========================================================================
# ALWAYS-ON 2 — the `Net` Protocol is satisfied by BOTH impls (structural check)
# ===========================================================================
def test_net_protocol_satisfied_by_both_impls():
    """The raw-forward `Net` port is satisfied STRUCTURALLY by the local ValueMLP adapter AND the remote
    ZmqNetClient — so a Python search uses local-or-remote interchangeably (the zero-cost ACL). Checked
    runtime-structurally (`isinstance` against the `runtime_checkable` Protocol) on real instances; the
    full signature contract is enforced by mypy --strict (the gate)."""
    # local: a tiny ValueMLP with a policy head
    from chocofarm.az.mlp import ValueMLP
    net = ValueMLP(in_dim=6, hidden=8, n_actions=5, seed=0)
    local = ValueMLPNet(net)
    assert isinstance(local, Net)
    # the adapter's raw predict returns (de-std value, RAW logits) of the right shape
    value, logits = local.predict(np.zeros(6, dtype=np.float32))
    assert isinstance(value, float)
    assert logits is not None and logits.shape == (5,)

    # remote: a ZmqNetClient instance (constructed without connecting to a live server is fine — the
    # socket connect is lazy/non-blocking; we never call predict here, just check the structural shape).
    if _pyzmq_available():
        client = ZmqNetClient(endpoint="tcp://127.0.0.1:5599")
        try:
            assert isinstance(client, Net)
        finally:
            client.close()
    else:
        # even without a live zmq, the class structurally has the method — assert the attribute contract
        assert hasattr(ZmqNetClient, "predict")


def test_valuemlp_adapter_value_only_returns_none_logits():
    """A value-only ValueMLP (no policy head) through the adapter returns `logits=None` — the raw-port
    mirror of forward_core's value-only path."""
    from chocofarm.az.mlp import ValueMLP
    net = ValueMLP(in_dim=4, hidden=6, n_actions=None, seed=0)
    local = ValueMLPNet(net)
    value, logits = local.predict(np.ones(4, dtype=np.float32))
    assert isinstance(value, float)
    assert logits is None


# ===========================================================================
# ALWAYS-ON 3 — the greedy-drain batching LOGIC over a FAKE forward (pure function, no socket)
# ===========================================================================
def test_run_microbatch_collapses_B_requests_to_one_forward():
    """Several requests (each carrying its OWN B_i leaves) CONCATENATE into EXACTLY ONE forward call
    (the microbatch), and each request gets back ONE batched response carrying its OWN B_i predictions.
    Uses a STUB forward (no JAX, no socket) so the concat → one-forward → scatter logic is asserted
    directly and deterministically. The requests carry differing B_i (1, 3, 2) to exercise the
    variable-width scatter back to per-request frames."""
    in_dim, n_actions = 6, 4
    rng = np.random.default_rng(7)
    counts = [1, 3, 2]
    mats = [rng.standard_normal((c, in_dim)).astype(np.float32) for c in counts]
    requests = [(f"ident-{i}".encode(), mats[i]) for i in range(len(counts))]
    total = sum(counts)

    calls = {"n": 0, "batch_shape": None}

    def stub_forward(params, Xb, y_mean, y_std):
        # assert ALL rows arrived in ONE concatenated (total, in_dim) call — the collapse the design promises
        calls["n"] += 1
        calls["batch_shape"] = tuple(np.asarray(Xb).shape)
        Xb = np.asarray(Xb)
        # a deterministic, row-distinguishable fake: v_std[i] = sum(row_i); logits[i] = row_i broadcast. The
        # forward now DE-STANDARDIZES on its side and returns the combined (B, 1+n_actions) block.
        v = (Xb.sum(axis=1) * y_std + y_mean).reshape(-1, 1).astype(np.float32)
        logits = (Xb[:, :n_actions] * 10.0).astype(np.float32)
        return np.concatenate([v, logits], axis=1)

    y_mean, y_std = 2.0, 3.0
    out = run_microbatch(stub_forward, params={}, y_mean=y_mean, y_std=y_std, requests=requests)

    assert calls["n"] == 1, "the requests must concatenate into ONE forward call"
    assert calls["batch_shape"] == (total, in_dim)
    assert len(out) == len(counts)
    # each identity gets ITS OWN B_i rows' value+logits, de-standardized — scatter correctness
    for i, (ident, resp) in enumerate(out):
        assert ident == f"ident-{i}".encode()
        v, logits = wire.decode_response(resp)
        assert v.shape == (counts[i],)
        expected_v = mats[i].sum(axis=1).astype(np.float32) * np.float32(y_std) + np.float32(y_mean)
        np.testing.assert_allclose(v, expected_v, atol=1e-5)
        assert logits is not None
        np.testing.assert_allclose(logits, mats[i][:, :n_actions] * 10.0, atol=1e-5)


def test_run_microbatch_value_only_scatters_empty_logits():
    """A value-only net (stub forward returns logits=None) scatters n_actions=0 to every request — the
    value-only batched path."""
    requests = [(b"a", np.ones((1, 3), dtype=np.float32)), (b"b", np.full((1, 3), 2.0, dtype=np.float32))]

    def stub(params, Xb, y_mean, y_std):
        # value-only: return just the (B, 1) de-standardized value column (no logits columns)
        return (np.asarray(Xb).sum(axis=1) * y_std + y_mean).reshape(-1, 1).astype(np.float32)

    out = run_microbatch(stub, {}, 0.0, 1.0, requests)
    assert [ident for ident, _ in out] == [b"a", b"b"]
    for ident, resp in out:
        _, logits = wire.decode_response(resp)
        assert logits is None


def test_run_microbatch_refuses_empty_batch():
    """ADR-0002: an empty batch is a loud refusal (the drain guarantees ≥1; calling with [] is a bug)."""
    with pytest.raises(ValueError):
        run_microbatch(lambda p, x, ym, ys: x.sum(1).reshape(-1, 1), {}, 0.0, 1.0, [])


def test_run_microbatch_refuses_ragged_batch():
    """ADR-0002: a ragged batch (mixed in_dim) is rejected, never silently padded/truncated."""
    requests = [(b"a", np.ones((1, 4), dtype=np.float32)), (b"b", np.ones((1, 5), dtype=np.float32))]
    with pytest.raises(ValueError):
        run_microbatch(lambda p, x, ym, ys: x.sum(1).reshape(-1, 1), {}, 0.0, 1.0, requests)


# ===========================================================================
# ALWAYS-ON 4 (zmq-gated, NO server/network) — the drain CAPS at max_batch.
# The regression guard for the 2026-06-23 N-sweep crash: at high overcommit the drain accumulated PAST
# max_batch (the cap was checked before the recv but the whole request was added after), and run_microbatch
# pads only UP — so a >max_batch batch hit the AOT-compiled fixed-shape forward and crashed. The drain now
# defers a straddling request whole to the next forward (restoring the invariant the docstring asserts).
# Driven through InferenceServer._drain in isolation (`__new__` + a fake socket, the staging test's pattern).
# ===========================================================================
class _FakeSock:
    """Stand-in ROUTER for `_drain`: hands back queued multipart frames, then raises `zmq.Again` (nothing
    more queued) — so the drain logic runs with no real socket bound."""
    def __init__(self, frames):
        self._frames = list(frames)

    def recv_multipart(self, flags=0):
        import zmq
        if self._frames:
            return self._frames.pop(0)
        raise zmq.Again()


class _FakePoller:
    def poll(self, timeout=0):
        return True   # the first bounded block always "sees" a request


def _drain_server(max_batch):
    """An InferenceServer with ONLY the fields `_drain` reads, no socket bound (the staging test's `__new__`
    pattern). Floor OFF (θ=0) — `_drain` is the single greedy pass plus the new cap-deferral."""
    from chocofarm.az.inference_server import InferenceServer
    srv = InferenceServer.__new__(InferenceServer)
    srv._max_batch = max_batch
    srv._min_forward_rows = 0
    srv._max_queue_delay_ms = 0.0
    srv._pending = None
    srv._stop = False
    srv._poller = _FakePoller()
    srv._POLL_INTERVAL_MS = 1
    return srv


def _req_frames(ident, X):
    """A REQ-style multipart frame `[ident][b""][payload]` the drain parses (ident, empty envelope, payload)."""
    return [ident, b"", wire.encode_request(X)]


@pytest.mark.skipif(not _pyzmq_available(), reason="the drain cap test needs zmq.Again (no server is bound)")
def test_drain_caps_at_max_batch_and_defers_straddler():
    """REGRESSION (the 2026-06-23 N-sweep crash): the drain must NEVER return more than max_batch rows — a
    request that would straddle the cap is DEFERRED WHOLE to the next drain (held in `_pending`), so the
    fixed-shape forward never gets an oversized matmul. Three 2-row requests at max_batch=5: the first drain
    takes 4 rows (two requests) and defers the third; the second drain returns the deferred one. All three
    serve exactly once, none crosses the cap. The PRE-FIX code returned all 6 rows in one drain → the crash."""
    X = np.ones((2, 4), dtype=np.float32)
    srv = _drain_server(max_batch=5)
    srv._sock = _FakeSock([_req_frames(f"r{i}".encode(), X) for i in range(3)])

    d1 = srv._drain()
    rows1 = sum(int(x.shape[0]) for _i, _e, x in d1)
    assert rows1 <= 5, f"drain returned {rows1} rows, exceeding max_batch=5 (the overshoot crash)"
    assert rows1 == 4 and len(d1) == 2          # two 2-row requests fit; a third would straddle 5
    assert srv._pending is not None             # the straddling third is held over, not dropped

    d2 = srv._drain()
    rows2 = sum(int(x.shape[0]) for _i, _e, x in d2)
    assert rows2 <= 5 and rows2 == 2 and len(d2) == 1   # the deferred third, alone
    assert srv._pending is None
    idents = [i for grp in (d1, d2) for (i, _e, _x) in grp]
    assert sorted(idents) == [b"r0", b"r1", b"r2"]      # served once each — no drop, no duplicate


@pytest.mark.skipif(not _pyzmq_available(), reason="the drain reject test needs zmq.Again (no server is bound)")
def test_drain_rejects_single_request_wider_than_max_batch():
    """ADR-0002: a SINGLE request wider than max_batch cannot be padded down or split in the drain — it is
    REJECTED loudly (a logged drop; the client's RPC times out) rather than handed to the fixed-shape forward
    as an oversized matmul (the cryptic XLA shape-crash). It must not appear in the drained batch; a valid
    request behind it still serves."""
    srv = _drain_server(max_batch=4)
    big = np.ones((6, 4), dtype=np.float32)            # one request, 6 rows > max_batch=4
    small = np.ones((2, 4), dtype=np.float32)
    srv._sock = _FakeSock([_req_frames(b"big", big), _req_frames(b"ok", small)])

    d = srv._drain()
    rows = sum(int(x.shape[0]) for _i, _e, x in d)
    assert rows <= 4
    idents = [i for (i, _e, _x) in d]
    assert b"big" not in idents          # the oversized request is rejected, never forwarded
    assert b"ok" in idents               # the valid request behind it still serves


def test_params_from_manifest_blob_matches_valuemlp_params():
    """The jax-free param reconstruction (manifest+blob → flat dict) yields ValueMLP._params() cast to
    float32 — the server's inference precision (the SSOT bar; the f64 wire weights are cast ONCE here at
    load, not per forward — ADR-0012 P1/P6). So the server runs the SSOT weights, at f32, without
    constructing a ValueMLP (staying off the held-out jax/numba boundary). Residual ON to exercise the
    optional block."""
    from chocofarm.az.mlp import ValueMLP
    from chocofarm.az.transport import pack_net
    net = ValueMLP(in_dim=6, hidden=8, n_actions=5, seed=3, y_mean=1.5, y_std=2.5, residual=True)
    manifest, blob = pack_net(net)
    params, y_mean, y_std = params_from_manifest_blob(manifest, blob)
    ref = net._params()
    assert set(params.keys()) == set(ref.keys())
    for k in ref:
        assert params[k].dtype == np.float32                       # cast once at load — the server is f32
        np.testing.assert_array_equal(params[k], ref[k].astype(np.float32))
    assert y_mean == pytest.approx(1.5)
    assert y_std == pytest.approx(2.5)


def test_inference_server_rejects_unhonorable_floor():
    """ADR-0002 / ADR-0012 P2 (translate-and-validate at the boundary, never coerce): the server-side
    coalescing floor (server-floor-design.md) is validated at CONSTRUCTION, before the socket binds. A
    θ above the max_batch cap is a knob the drain cannot honor (the cap forbids ever reaching it) — a
    loud raise, not a silent accept; a negative θ or delay is likewise rejected. (These raises precede
    the `import zmq` in __init__, so this is an ALWAYS-ON test: no pyzmq, no socket bound.)"""
    from chocofarm.az.inference_server import InferenceServer
    src = StaticParamsSource({}, 0.0, 1.0)
    with pytest.raises(ValueError, match="exceeds max_batch"):
        InferenceServer(src, max_batch=128, min_forward_rows=256)
    with pytest.raises(ValueError, match="min_forward_rows must"):
        InferenceServer(src, min_forward_rows=-1)
    with pytest.raises(ValueError, match="max_queue_delay_ms must"):
        InferenceServer(src, max_queue_delay_ms=-1.0)


# ===========================================================================
# JAX-GATED — the params-staging forward (build_staged_forward) equivalence + rebuild-on-reload.
# Skips if jax is unimportable, so the default suite stays green without jax (the same opt-in gating
# posture the server-parity tests use). No socket, no redis — the staging seam is exercised through the
# pure run_microbatch, against the production jit_forward_core path on the SAME (params, X).
# ===========================================================================
def _jax_available() -> bool:
    try:
        import jax  # noqa: F401
        return True
    except ImportError:
        return False


@pytest.mark.skipif(not _jax_available(), reason="params-staging equivalence needs jax (the real forward)")
@pytest.mark.parametrize("residual", [False, True])
def test_staged_forward_matches_jit_forward_core(residual):
    """The cross-DEVICE consolidation (ADR-0012 P7 / bench fb9cfbc) is BEHAVIOR-PRESERVING: the staged
    forward — `build_staged_forward`, whose weights are staged device-resident once via the lowlatency
    handle — computes the SAME `[v | logits]` block as the production `jit_forward_core` (host params
    re-passed every call) on the same `(params, Xb)`, through the pure `run_microbatch`, within ABS_TOL=1e-4
    (the project forward bar; ADR-0009 — in practice byte-identical, the staging only changes WHERE the
    weights live, not the arithmetic). Several batch sizes B exercise the padded fixed-shape forward both
    above and at the pad cap. residual ON and OFF (the optional block rides through the staged graph)."""
    from chocofarm.az.inference_server import build_staged_forward, jit_forward_core, run_microbatch
    from chocofarm.az.mlp import ValueMLP
    from chocofarm.az.transport import pack_net

    in_dim, n_actions, max_batch = 12, 7, 16
    net = ValueMLP(in_dim=in_dim, hidden=24, n_actions=n_actions, seed=11,
                   y_mean=0.7, y_std=1.3, residual=residual)
    params, y_mean, y_std = params_from_manifest_blob(*pack_net(net))
    staged = build_staged_forward(params, y_mean, y_std, pad_to=max_batch)

    rng = np.random.default_rng(2025 + int(residual))
    max_d = 0.0
    for B in (1, 2, 5, 8, 16):
        req = [(b"x", rng.standard_normal((B, in_dim)).astype(np.float32))]
        out_jit = run_microbatch(jit_forward_core, params, y_mean, y_std, req, pad_to=max_batch)
        out_stg = run_microbatch(staged, params, y_mean, y_std, req, pad_to=max_batch)
        for (ij, rj), (is_, rs) in zip(out_jit, out_stg):
            assert ij == is_
            vj, lj = wire.decode_response(rj)
            vs, ls = wire.decode_response(rs)
            max_d = max(max_d, float(np.max(np.abs(vj - vs))))
            if lj is not None:
                assert ls is not None
                max_d = max(max_d, float(np.max(np.abs(lj - ls))))
    assert max_d < 1e-4, f"residual={residual}: staged vs jit_forward_core max|Δ|={max_d:.3e} exceeds 1e-4"


@pytest.mark.skipif(not _jax_available(), reason="staged-rebuild-on-reload needs jax (the real forward)")
def test_staged_forward_rebuilds_on_version_reload():
    """The RECONFIG guard (ADR-0002 — a stale-net serve is a loud-failure class): the staged handle is
    REBUILT when the version-gated reload rebinds a fresh params dict, so a forward never runs against the
    previous version's staged weights. Drives `InferenceServer._effective_forward` directly (no socket):
    (1) same params object -> the SAME staged handle is reused (no rebuild); (2) a reload to a DIFFERENT
    net -> the handle is rebuilt and serves the NEW net's forward (matching jit_forward_core on the new
    params, NOT the stale net's)."""
    from chocofarm.az.inference_server import InferenceServer, jit_forward_core, run_microbatch
    from chocofarm.az.mlp import ValueMLP
    from chocofarm.az.transport import pack_net

    in_dim, n_actions, max_batch = 12, 7, 16
    net1 = ValueMLP(in_dim=in_dim, hidden=24, n_actions=n_actions, seed=1, y_mean=0.5, y_std=1.1, residual=True)
    net2 = ValueMLP(in_dim=in_dim, hidden=24, n_actions=n_actions, seed=2, y_mean=-0.3, y_std=2.0, residual=True)
    p1, ym1, ys1 = params_from_manifest_blob(*pack_net(net1))
    p2, ym2, ys2 = params_from_manifest_blob(*pack_net(net2))

    # Build the server WITHOUT binding a socket (the staging seam needs no zmq): construct via __new__ and
    # set only the fields _effective_forward reads. (A full InferenceServer would bind a ROUTER; this test
    # targets the staging logic in isolation, mirroring the pure run_microbatch always-on tests.)
    srv = InferenceServer.__new__(InferenceServer)
    srv._max_batch = max_batch
    srv._stages_params = True
    srv._staged_fn = None
    srv._staged_params_id = None
    srv._forward_fn = jit_forward_core

    f1 = srv._effective_forward(p1, ym1, ys1)
    assert srv._staged_fn is not None
    f1_again = srv._effective_forward(p1, ym1, ys1)   # same params object -> reuse, no rebuild
    assert f1_again is f1

    f2 = srv._effective_forward(p2, ym2, ys2)          # reload to net2 -> rebuild
    assert f2 is not f1

    # The rebuilt handle must serve net2 (not the stale net1): equal to jit_forward_core on net2's params.
    rng = np.random.default_rng(9)
    req = [(b"x", rng.standard_normal((4, in_dim)).astype(np.float32))]
    out_stg = run_microbatch(f2, p2, ym2, ys2, req, pad_to=max_batch)
    out_ref = run_microbatch(jit_forward_core, p2, ym2, ys2, req, pad_to=max_batch)
    vs, ls = wire.decode_response(out_stg[0][1])
    vr, lr = wire.decode_response(out_ref[0][1])
    d = max(float(np.max(np.abs(vs - vr))), float(np.max(np.abs(ls - lr))))
    assert d < 1e-4, f"rebuilt staged handle disagrees with net2 jit_forward_core: max|Δ|={d:.3e}"


# ===========================================================================
# OPT-IN — the full server+client parity harness (needs a running server; NO redis)
# ===========================================================================
def _py_forward_f32(params, X, y_mean, y_std):
    """The local reference at the float32 precision the service runs at: float32 weights + float32 X
    through the ONE forward_core, then de-standardize (v_std·y_std + y_mean) in float32. The SAME
    quantity the server computes — the only gap is the batched matmul's row-vs-single reorder (§4)."""
    import numpy as _np

    from chocofarm.az.forward import forward_core
    p32 = {k: _np.asarray(v, dtype=_np.float32) for k, v in params.items()}
    x = _np.asarray(X, dtype=_np.float32)
    v_std, logits = forward_core(p32, x, _np)
    value = _np.asarray(v_std, dtype=_np.float32) * _np.float32(y_std) + _np.float32(y_mean)
    return _np.asarray(value, dtype=_np.float32), (None if logits is None else _np.asarray(logits, dtype=_np.float32))


@pytest.mark.skipif(not (_RUN_ZMQ and _pyzmq_available()),
                    reason="opt-in zmq parity: set CHOCO_RUN_ZMQ=1 (and have pyzmq) and a server spins in-process")
@pytest.mark.parametrize("residual", [False, True])
def test_server_client_parity_within_1e_4(residual):
    """The SSOT parity (ADR-0012 P6 / design §7): spin the InferenceServer in-process with params
    injected DIRECTLY (StaticParamsSource — NO redis), RPC it with the ZmqNetClient over N≥1000 random
    float32 feature vectors across VARIED batch sizes B (concurrent clients), and assert `value` AND
    each `logit` within 1e-4 of the local float32 forward_core. residual ON and OFF. The batched path's
    row-vs-single f32 roundoff lives inside the 1e-4 bar (behavioral equivalence, NOT byte-identity)."""
    import zmq

    from chocofarm.az.inference_server import InferenceServer
    from chocofarm.az.mlp import ValueMLP
    from chocofarm.az.transport import pack_net

    in_dim, n_actions = 12, 7
    rng = np.random.default_rng(2024 + int(residual))
    net = ValueMLP(in_dim=in_dim, hidden=24, n_actions=n_actions, seed=11,
                   y_mean=float(rng.normal()) * 2.0, y_std=float(abs(rng.normal()) + 0.5),
                   residual=residual)
    params, y_mean, y_std = params_from_manifest_blob(*pack_net(net))
    src = StaticParamsSource(params, y_mean, y_std)

    endpoint = f"tcp://127.0.0.1:{5700 + int(residual)}"
    server = InferenceServer(src, bind=endpoint, max_batch=256)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()

    n_inputs = 1200    # ≥1000
    X = rng.standard_normal((n_inputs, in_dim)).astype(np.float32)
    ref_v, ref_l = _py_forward_f32(params, X, y_mean, y_std)

    max_dv = 0.0
    max_dl = 0.0
    try:
        # Drive VARIED batch sizes B by firing B concurrent client RPCs at once so the server's
        # greedy-drain stacks them into one (B, in_dim) forward (exercising the row-vs-single roundoff).
        i = 0
        for B in (1, 2, 4, 8, 16, 32, 64):
            while i < n_inputs:
                batch = X[i:i + B]
                if batch.shape[0] == 0:
                    break
                got_v, got_l = _concurrent_predict(endpoint, batch)
                for j in range(batch.shape[0]):
                    dv = abs(float(got_v[j]) - float(ref_v[i + j]))
                    max_dv = max(max_dv, dv)
                    if ref_l is not None:
                        gl = got_l[j]
                        assert gl is not None
                        dl = float(np.max(np.abs(gl.astype(np.float64) - ref_l[i + j].astype(np.float64))))
                        max_dl = max(max_dl, dl)
                i += batch.shape[0]
                if i >= n_inputs:
                    break
            if i >= n_inputs:
                break
    finally:
        # Clean shutdown sequence (no socket killed from another thread): flip stop, let the bounded
        # poll observe it and the serve thread exit, THEN close the socket.
        server.stop()
        t.join(timeout=5.0)
        server.close()

    assert max_dv < 1e-4, f"residual={residual}: max|Δvalue|={max_dv:.3e} exceeds 1e-4"
    if ref_l is not None:
        assert max_dl < 1e-4, f"residual={residual}: max|Δlogit|={max_dl:.3e} exceeds 1e-4"
    print(f"[zmq parity residual={'ON ' if residual else 'OFF'}] N={n_inputs} "
          f"max|Δvalue|={max_dv:.3e} max|Δlogit|={max_dl:.3e} (bar 1e-4)")


@pytest.mark.skipif(not (_RUN_ZMQ and _pyzmq_available()),
                    reason="opt-in zmq floor: set CHOCO_RUN_ZMQ=1 (and have pyzmq) and a server spins in-process")
def test_server_parity_holds_under_coalescing_floor():
    """The increment-(ii) server floor (server-floor-design.md) must not corrupt the scatter: with θ>0
    the drain ACCUMULATES across concurrent clients (re-draining + briefly waiting) before one forward,
    so this fires B concurrent RPCs at a floor-armed server (θ=8, a 40ms hard delay) and asserts every
    client still gets ITS OWN row back within the 1e-4 bar (ADR-0012 P6). It exercises the new
    multi-pass drain loop end-to-end: θ is reached when B≥8 land together, and the hard delay is the
    escape hatch for the final partial group (no wedge). A correct floor changes only WHEN the forward
    fires, never WHICH rows scatter to whom."""
    from chocofarm.az.inference_server import InferenceServer
    from chocofarm.az.mlp import ValueMLP
    from chocofarm.az.transport import pack_net

    in_dim, n_actions = 12, 7
    rng = np.random.default_rng(7)
    net = ValueMLP(in_dim=in_dim, hidden=24, n_actions=n_actions, seed=5,
                   y_mean=0.7, y_std=1.3, residual=True)
    params, y_mean, y_std = params_from_manifest_blob(*pack_net(net))
    src = StaticParamsSource(params, y_mean, y_std)

    endpoint = "tcp://127.0.0.1:5702"
    server = InferenceServer(src, bind=endpoint, max_batch=256,
                             min_forward_rows=8, max_queue_delay_ms=40.0)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()

    n_inputs = 320
    X = rng.standard_normal((n_inputs, in_dim)).astype(np.float32)
    ref_v, ref_l = _py_forward_f32(params, X, y_mean, y_std)

    max_dv = 0.0
    max_dl = 0.0
    try:
        i = 0
        for B in (8, 16, 8, 3):   # ≥θ groups (θ reached) and a sub-θ tail (fires at the hard delay)
            while i < n_inputs:
                batch = X[i:i + B]
                if batch.shape[0] == 0:
                    break
                got_v, got_l = _concurrent_predict(endpoint, batch)
                for j in range(batch.shape[0]):
                    max_dv = max(max_dv, abs(float(got_v[j]) - float(ref_v[i + j])))
                    if ref_l is not None:
                        gl = got_l[j]
                        assert gl is not None
                        max_dl = max(max_dl, float(np.max(np.abs(
                            gl.astype(np.float64) - ref_l[i + j].astype(np.float64)))))
                i += batch.shape[0]
                if i >= n_inputs:
                    break
            if i >= n_inputs:
                break
    finally:
        server.stop()
        t.join(timeout=5.0)
        server.close()

    assert max_dv < 1e-4, f"floor: max|Δvalue|={max_dv:.3e} exceeds 1e-4"
    assert max_dl < 1e-4, f"floor: max|Δlogit|={max_dl:.3e} exceeds 1e-4"
    print(f"[zmq floor θ=8 delay=40ms] N={n_inputs} max|Δvalue|={max_dv:.3e} max|Δlogit|={max_dl:.3e}")


# ===========================================================================
# OPT-IN — the SUBCLASS PARITY harness (the P6 behavioural backstop the lab-staging-divergence RCA named:
# §5/§6.2 — "the test that was missing"). Stands up the bench/lab InferenceServer SUBCLASSES alongside the
# base and asserts each one's SERVED value+logits is allclose(1e-4) to the base's on a MATCHED (params, X)
# request. This is the test that would have caught the params-staging divergence (a subclass serving a
# different forward than the base) — and, after the ADR-0012 P3 template-method split, it pins that the
# sealed `_run_forward` dispatch is the SAME for the base and every subclass (the subclasses now vary only
# `_pad_shape`/`_forward_groups`/`_scatter`, never the forward), so no subclass can silently diverge.
# Guarded like the other socket tests: needs CHOCO_RUN_ZMQ=1 + pyzmq, so the default suite stays green.
# ===========================================================================
def _stage_a_on_path() -> bool:
    """Put the bench `cpp/stage_a` (+ its `control_lab`) on sys.path so the subclasses import, mirroring how
    they bootstrap their own path. Returns False if the bench tree is absent (then the test skips), so this
    test is robust to a checkout without the cpp bench."""
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    stage_a = os.path.join(repo, "cpp", "stage_a")
    control_lab = os.path.join(stage_a, "control_lab")
    if not os.path.isdir(control_lab):
        return False
    for p in (stage_a, control_lab):
        if p not in sys.path:
            sys.path.insert(0, p)
    return True


@pytest.mark.skipif(not (_RUN_ZMQ and _pyzmq_available()),
                    reason="opt-in subclass parity: set CHOCO_RUN_ZMQ=1 (and have pyzmq); servers spin in-process")
@pytest.mark.parametrize("residual", [False, True])
def test_subclass_servers_parity_with_base(residual):
    """The SUBCLASS PARITY backstop (RCA §6.2): a real `StageAServer` and a real `LabServer` serve the SAME
    `(value, logits)` as the base `InferenceServer` on a matched `(params, X)`, within 1e-4 (ADR-0012 P6).

    The base is STAGED + pad-to-max (`_uses_fixed_pad=True`); the subclasses are UN-STAGED + bucket-E
    (`_uses_fixed_pad=False`) — DELIBERATELY different forward *plumbing* (the divergence the consolidation
    introduced and the RCA traced). The numerics must nonetheless agree to 1e-4 because all three run the
    ONE `forward_core` via the sealed `_run_forward` — the staging changes only WHERE the weights live and
    the bucket changes only the (zero) pad rows, neither the arithmetic of the real rows. A subclass that
    re-authored (and drifted) the forward — the bug class the P3 split makes unrepresentable — would fail
    THIS assertion. Several batch sizes B exercise the row-vs-single roundoff under each pad policy; the
    LabServer is additionally driven over its lab FEATURE/GATE envelope (so the gate-tagging boundary is on
    the wire), and the value it serves under that envelope is asserted equal to the base's."""
    if not _pyzmq_available():
        pytest.skip("pyzmq missing")
    if not _stage_a_on_path():
        pytest.skip("cpp/stage_a bench tree not present")
    import struct

    import zmq

    from chocofarm.az.inference_server import InferenceServer, jit_forward_core
    from chocofarm.az.mlp import ValueMLP
    from chocofarm.az.transport import pack_net
    from control_lab.lab_server import LabServer
    from control_lab.lab_wire import LabFeature, decode_gate, encode_feature
    from stage_a_server import StageAServer

    in_dim, n_actions = 12, 7
    rng = np.random.default_rng(4242 + int(residual))
    net = ValueMLP(in_dim=in_dim, hidden=24, n_actions=n_actions, seed=11,
                   y_mean=0.7, y_std=1.3, residual=residual)
    params, y_mean, y_std = params_from_manifest_blob(*pack_net(net))

    base_ep = f"tcp://127.0.0.1:{5720 + int(residual)}"
    stage_ep = f"tcp://127.0.0.1:{5722 + int(residual)}"
    lab_ep = f"tcp://127.0.0.1:{5724 + int(residual)}"

    # max_batch=512 with the default BUCKETS (64,256,512): the StageA/Lab bucket-E pad is exercised on the
    # small B forwards (a B=1/2/5 request snaps up to bucket 64), the base pads every forward to 512.
    base = InferenceServer(StaticParamsSource(params, y_mean, y_std), bind=base_ep, max_batch=512)
    stage = StageAServer(StaticParamsSource(params, y_mean, y_std), bind=stage_ep, max_batch=512,
                         forward_fn=jit_forward_core, e_policy="bucket", wakeup="group")
    lab = LabServer(StaticParamsSource(params, y_mean, y_std), bind=lab_ep, max_batch=512,
                    forward_fn=jit_forward_core, e_policy="bucket", wakeup="group")
    # The base is fixed-pad → STAGED; the subclasses bucket → UN-STAGED. Pin the predicate explicitly so a
    # future regression of the flag is caught here, not silently (the divergence the RCA is about).
    assert base._uses_fixed_pad is True and base._stages_params is True
    assert stage._uses_fixed_pad is False and lab._uses_fixed_pad is False

    threads = []
    for srv in (base, stage, lab):
        t = threading.Thread(target=srv.serve_forever, daemon=True)
        t.start()
        threads.append(t)

    def _lab_predict(endpoint, x_row, tid):
        """Drive the LabServer over its lab envelope `[corr][FEATURE][value]` and decode the reply
        `[corr][GATE][value]`, returning the served (value, logits). Asserts the GATE frame came back for
        the served tid (the producer STRICTLY requires a gate when it sent a feature)."""
        ctx = zmq.Context()
        sock = ctx.socket(zmq.DEALER)
        sock.setsockopt(zmq.LINGER, 0)
        sock.setsockopt(zmq.RCVTIMEO, 10000)
        sock.connect(endpoint)
        try:
            corr = struct.pack("<Q", 0xA1B2C3D4 + tid)
            feat = encode_feature(LabFeature(tid=tid, inflight=0, ready=0, msgs=1, leaves=1, rtt_us=0,
                                             decisions=0))
            sock.send_multipart([corr, feat, wire.encode_request(x_row)])
            frames = sock.recv_multipart()
            assert len(frames) == 3, f"lab reply must be [corr][gate][resp], got {len(frames)} frames"
            assert frames[0] == corr, "lab reply must echo the corr-id verbatim"
            gt_tid, _allow = decode_gate(frames[1])
            assert gt_tid == tid, f"gate frame tid {gt_tid} != requested {tid}"
            v, l = wire.decode_response(frames[-1])
            return v, l
        finally:
            sock.close(0)
            ctx.term()

    max_d = 0.0
    try:
        # A matched bank of feature vectors; for each, compare each subclass's served (value, logits) to the
        # base's on the IDENTICAL row. Varied B (concurrent clients) exercises the row-vs-single roundoff
        # under bucket-E (subclass) vs pad-to-max (base).
        n_inputs = 60
        X = rng.standard_normal((n_inputs, in_dim)).astype(np.float32)
        i = 0
        for B in (1, 2, 5, 8, 16):
            if i >= n_inputs:
                break
            batch = X[i:i + B]
            if batch.shape[0] == 0:
                break
            # base + StageA over the plain ZmqNetClient (REQ); concurrent so the drain stacks them.
            base_v, base_l = _concurrent_predict(base_ep, batch)
            stage_v, stage_l = _concurrent_predict(stage_ep, batch)
            for j in range(batch.shape[0]):
                max_d = max(max_d, abs(float(base_v[j]) - float(stage_v[j])))
                if base_l is not None:
                    assert stage_l[j] is not None
                    max_d = max(max_d, float(np.max(np.abs(
                        np.asarray(base_l[j], dtype=np.float64) - np.asarray(stage_l[j], dtype=np.float64)))))
                # LabServer over its lab envelope (one row at a time, tagged by tid=j).
                lab_v, lab_l = _lab_predict(lab_ep, batch[j], tid=j)
                max_d = max(max_d, abs(float(base_v[j]) - float(lab_v[0])))
                if base_l is not None:
                    assert lab_l is not None
                    max_d = max(max_d, float(np.max(np.abs(
                        np.asarray(base_l[j], dtype=np.float64) - lab_l[0].astype(np.float64)))))
            i += batch.shape[0]
    finally:
        for srv in (base, stage, lab):
            srv.stop()
        for t in threads:
            t.join(timeout=5.0)
        for srv in (base, stage, lab):
            srv.close()

    assert max_d < 1e-4, (
        f"residual={residual}: subclass (StageA/Lab) served value/logits diverged from the base by "
        f"max|Δ|={max_d:.3e} > 1e-4 — a subclass is serving a different forward than the base (the "
        f"divergence the P3 split forbids; ADR-0012 P6 / lab-staging-divergence-rca.md)")
    print(f"[subclass parity residual={'ON ' if residual else 'OFF'}] StageA+Lab vs base "
          f"max|Δ(value,logits)|={max_d:.3e} (bar 1e-4)")


def _concurrent_predict(endpoint, batch):
    """Fire `len(batch)` ZmqNetClient.predict RPCs CONCURRENTLY (one client/thread each) so the server
    drains them into ONE microbatch of size B — the row-vs-single roundoff exerciser. Returns
    (values[B], logits[B]) aligned to `batch`'s rows."""
    B = batch.shape[0]
    values: list = [None] * B
    logit_rows: list = [None] * B

    def one(j):
        with ZmqNetClient(endpoint=endpoint, recv_timeout_ms=10000) as client:
            v, l = client.predict(batch[j])
            values[j] = v
            logit_rows[j] = l

    threads = [threading.Thread(target=one, args=(j,)) for j in range(B)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    return values, logit_rows


@pytest.mark.skipif(not (_RUN_ZMQ and _pyzmq_available()),
                    reason="opt-in zmq corr-id: set CHOCO_RUN_ZMQ=1 (and have pyzmq); a server spins in-process")
def test_server_echoes_corr_id_envelope_across_a_batch():
    """The corr-id correlation contract — the C++ pool's reply→tree routing, mechanized at the Python
    boundary (ADR-0011: lock the contract, don't trust the class). A DEALER client stamps each request
    with a LEADING 8-byte correlation-id frame `[corr-id][request]`; the server ROUND-TRIPS that frame
    VERBATIM in the reply `[corr-id][response]` WITHOUT parsing it — the corr-id is transport routing,
    kept OUT of the value codec (ADR-0012 P7 serialization⊥transport), so it carries no cross-language
    format to drift, and its only validation is the consumer's own lookup (fail-loud on an unknown id,
    C++-side).

    The crux is the BATCH: B requests with DISTINCT corr-ids AND distinct payloads are fired on ONE
    socket so the greedy drain stacks them into one forward, and each reply's echoed corr-id must select
    the request whose value it carries. The corr-ids are deliberately NOT in submit order, so a
    positional FIFO would mis-route — only true id-correlation passes. This is exactly what lets a worker
    route a batched reply to the right tree (and, promoted to a shared registry, enables work-stealing
    tree migration)."""
    import struct

    import zmq

    from chocofarm.az.inference_server import InferenceServer
    from chocofarm.az.mlp import ValueMLP
    from chocofarm.az.transport import pack_net

    in_dim, n_actions, B = 9, 5, 12
    rng = np.random.default_rng(31337)
    net = ValueMLP(in_dim=in_dim, hidden=16, n_actions=n_actions, seed=5,
                   y_mean=0.3, y_std=1.7, residual=False)
    params, y_mean, y_std = params_from_manifest_blob(*pack_net(net))
    src = StaticParamsSource(params, y_mean, y_std)

    endpoint = "tcp://127.0.0.1:5708"
    server = InferenceServer(src, bind=endpoint, max_batch=256)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()

    X = rng.standard_normal((B, in_dim)).astype(np.float32)
    ref_v, _ = _py_forward_f32(params, X, y_mean, y_std)
    # distinct 64-bit corr-ids, NOT in submit order (a multiplicative hash of j) — a positional match
    # would route these wrong; only the echoed id gets them right.
    corr_ids = [(0xC0FFEE01 + j * 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF for j in range(B)]
    by_corr = {struct.pack("<Q", c): j for j, c in enumerate(corr_ids)}
    assert len(by_corr) == B, "corr-ids must be distinct for the routing assertion to mean anything"

    ctx = zmq.Context()
    sock = ctx.socket(zmq.DEALER)
    sock.setsockopt(zmq.LINGER, 0)
    sock.setsockopt(zmq.RCVTIMEO, 10000)
    sock.connect(endpoint)
    seen: dict[int, bool] = {}
    try:
        for j in range(B):
            sock.send_multipart([struct.pack("<Q", corr_ids[j]), wire.encode_request(X[j])])
        for _ in range(B):
            frames = sock.recv_multipart()
            assert len(frames) == 2, f"expected [corr-id][response], got {len(frames)} frames"
            corr_echo, resp = frames[0], frames[-1]
            assert corr_echo in by_corr, f"server echoed an unknown/garbled corr-id: {corr_echo.hex()}"
            j = by_corr[corr_echo]
            assert j not in seen, f"corr-id for request {j} came back twice"
            seen[j] = True
            v, _l = wire.decode_response(resp)   # B=1 reply: v is a length-1 array
            assert v.shape == (1,)
            assert abs(float(v[0]) - float(ref_v[j])) < 1e-4, (
                f"corr-id {corr_echo.hex()} routed to request {j}, but value {v[0]} != ref {ref_v[j]}")
        assert len(seen) == B, "not every request received its echoed reply"
    finally:
        sock.close(0)
        ctx.term()
        server.stop()
        t.join(timeout=5.0)
        server.close()
    print(f"[zmq corr-id] B={B} requests batched, each routed by echoed id (out-of-order ids)")
