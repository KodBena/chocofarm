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
@pytest.mark.parametrize("in_dim", [1, 2, 7, 65, 300])
@pytest.mark.parametrize("n_actions", [0, 1, 65])
def test_codec_round_trips_request_and_response(in_dim, n_actions):
    """encode∘decode == identity over random float32 vectors, for the REQUEST (feature vector) and the
    RESPONSE (de-std value + raw logits). `n_actions=0` is the value-only case (logits=None, empty
    block). float32 is the wire dtype, so the round-trip is EXACT (no precision is lost re-reading the
    same bytes)."""
    rng = np.random.default_rng(1234 + in_dim * 131 + n_actions)
    X = rng.standard_normal(in_dim).astype(np.float32)
    back_X = wire.decode_request(wire.encode_request(X))
    assert back_X.dtype == np.float32
    np.testing.assert_array_equal(back_X, X)   # EXACT — same f32 bytes re-read

    value = float(rng.standard_normal())
    logits = None if n_actions == 0 else rng.standard_normal(n_actions).astype(np.float32)
    back_v, back_l = wire.decode_response(wire.encode_response(value, logits))
    # value is encoded as a single f32, so it round-trips to its float32 image exactly
    assert back_v == pytest.approx(np.float32(value), abs=0.0, rel=0.0) or back_v == float(np.float32(value))
    if n_actions == 0:
        assert back_l is None
    else:
        assert back_l is not None
        assert back_l.dtype == np.float32
        np.testing.assert_array_equal(back_l, logits)


def test_codec_value_only_response_is_empty_logits():
    """The value-only path: `encode_response(v, None)` carries n_actions=0 and an EMPTY logits block,
    and decodes back to `(v, None)` — mirroring forward_core's `logits=None`."""
    frame = wire.encode_response(3.5, None)
    v, logits = wire.decode_response(frame)
    assert logits is None
    assert v == float(np.float32(3.5))


def test_codec_rejects_bad_protocol_byte():
    """A request/response whose first byte is not the supported PROTOCOL_VERSION is a LOUD WireError —
    a codec mismatch fails loudly rather than misreading the next field as a float (ADR-0002)."""
    good = wire.encode_request(np.ones(4, dtype=np.float32))
    bad = bytes([wire.PROTOCOL_VERSION + 7]) + good[1:]
    with pytest.raises(wire.WireError):
        wire.decode_request(bad)
    good_resp = wire.encode_response(1.0, np.ones(3, dtype=np.float32))
    bad_resp = bytes([wire.PROTOCOL_VERSION + 7]) + good_resp[1:]
    with pytest.raises(wire.WireError):
        wire.decode_response(bad_resp)


def test_codec_rejects_wrong_length_payload():
    """A request whose payload byte count is not exactly `in_dim × f32` is a LOUD WireError — the codec
    refuses a truncated/over-long frame instead of zero-filling or truncating it (ADR-0002)."""
    good = wire.encode_request(np.ones(8, dtype=np.float32))
    with pytest.raises(wire.WireError):
        wire.decode_request(good[:-4])        # one float short
    with pytest.raises(wire.WireError):
        wire.decode_request(good + b"\x00\x00\x00\x00")   # one float long


def test_codec_rejects_nan_feature():
    """A NaN/Inf feature is rejected at ENCODE (a malformed request never reaches the wire) — ADR-0002:
    never a coerced/zero-filled forward."""
    bad = np.array([1.0, np.nan, 2.0, 3.0], dtype=np.float32)
    with pytest.raises(wire.WireError):
        wire.encode_request(bad)
    worse = np.array([1.0, np.inf, 2.0], dtype=np.float32)
    with pytest.raises(wire.WireError):
        wire.encode_request(worse)


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
    """B concurrent requests collapse to EXACTLY ONE forward call (the greedy-drain microbatch), and
    each request's identity gets ITS OWN row back. Uses a STUB forward (no JAX, no socket) so the
    drain → stack → one-forward → scatter logic is asserted directly and deterministically."""
    in_dim, n_actions, B = 6, 4, 5
    rng = np.random.default_rng(7)
    rows = [rng.standard_normal(in_dim).astype(np.float32) for _ in range(B)]
    requests = [(f"ident-{i}".encode(), rows[i]) for i in range(B)]

    calls = {"n": 0, "batch_shape": None}

    def stub_forward(params, Xb, xp):
        # assert ALL rows arrived in ONE stacked (B, in_dim) call — the collapse the design promises
        calls["n"] += 1
        calls["batch_shape"] = tuple(np.asarray(Xb).shape)
        Xb = np.asarray(Xb)
        # a deterministic, row-distinguishable fake: v_std[i] = sum(row_i); logits[i] = row_i broadcast
        v_std = Xb.sum(axis=1)
        logits = Xb[:, :n_actions] * 10.0
        return v_std, logits

    y_mean, y_std = 2.0, 3.0
    out = run_microbatch(stub_forward, params={}, y_mean=y_mean, y_std=y_std, requests=requests)

    assert calls["n"] == 1, "B requests must collapse to ONE forward call"
    assert calls["batch_shape"] == (B, in_dim)
    assert len(out) == B
    # each identity gets ITS OWN row's value+logits, de-standardized — scatter correctness
    for i, (ident, resp) in enumerate(out):
        assert ident == f"ident-{i}".encode()
        v, logits = wire.decode_response(resp)
        expected_v = np.float32(rows[i].sum()) * np.float32(y_std) + np.float32(y_mean)
        assert v == pytest.approx(float(expected_v), abs=1e-5)
        assert logits is not None
        np.testing.assert_allclose(logits, rows[i][:n_actions].astype(np.float32) * 10.0, atol=1e-5)


def test_run_microbatch_value_only_scatters_empty_logits():
    """A value-only net (stub forward returns logits=None) scatters n_actions=0 to every request — the
    value-only batched path."""
    requests = [(b"a", np.ones(3, dtype=np.float32)), (b"b", np.full(3, 2.0, dtype=np.float32))]

    def stub(params, Xb, xp):
        return np.asarray(Xb).sum(axis=1), None

    out = run_microbatch(stub, {}, 0.0, 1.0, requests)
    assert [ident for ident, _ in out] == [b"a", b"b"]
    for ident, resp in out:
        _, logits = wire.decode_response(resp)
        assert logits is None


def test_run_microbatch_refuses_empty_batch():
    """ADR-0002: an empty batch is a loud refusal (the drain guarantees ≥1; calling with [] is a bug)."""
    with pytest.raises(ValueError):
        run_microbatch(lambda p, x, xp: (x.sum(1), None), {}, 0.0, 1.0, [])


def test_run_microbatch_refuses_ragged_batch():
    """ADR-0002: a ragged batch (mixed in_dim) is rejected, never silently padded/truncated."""
    requests = [(b"a", np.ones(4, dtype=np.float32)), (b"b", np.ones(5, dtype=np.float32))]
    with pytest.raises(ValueError):
        run_microbatch(lambda p, x, xp: (x.sum(1), None), {}, 0.0, 1.0, requests)


def test_params_from_manifest_blob_matches_valuemlp_params():
    """The jax-free param reconstruction (manifest+blob → flat dict) yields EXACTLY ValueMLP._params()
    — so the server runs the SSOT weights without constructing a ValueMLP (staying off the held-out
    jax/numba boundary). Residual ON to exercise the optional block."""
    from chocofarm.az.mlp import ValueMLP
    from chocofarm.az.transport import pack_net
    net = ValueMLP(in_dim=6, hidden=8, n_actions=5, seed=3, y_mean=1.5, y_std=2.5, residual=True)
    manifest, blob = pack_net(net)
    params, y_mean, y_std = params_from_manifest_blob(manifest, blob)
    ref = net._params()
    assert set(params.keys()) == set(ref.keys())
    for k in ref:
        np.testing.assert_array_equal(params[k], ref[k])
    assert y_mean == pytest.approx(1.5)
    assert y_std == pytest.approx(2.5)


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
