#!/usr/bin/env python3
"""
tests/test_zmq_net_cpp.py — the cross-language round-trip pin for the C++ ZmqNetClient (the REMOTE
NetEvaluator, docs/design/zmq-inference-service.md §1/§5/§7; cpp/src/zmq_net_client.cpp).

It proves the FULL inference path round-trips faithfully across languages: the C++ ZmqNetClient encodes
a request (shared wire codec, derived from the wire_spec SSOT) → the Python InferenceServer runs the ONE
`forward_core` (the SSOT batched forward) → the C++ client decodes the reply into (value, logits). The
reference is the LOCAL float32 `forward_core` — the SAME quantity the C++ `NetForward` is pinned to
(cpp/parity/net_parity.py holds C++ NetForward ≡ Python float32 forward_core < 1e-4). So asserting the
C++ client matches the local float32 forward_core within 1e-4 transitively asserts it matches the local
C++ NetForward on the same weights+inputs — redis-free.

OPT-IN, mirroring tests/test_cpp_runner.py (CHOCO_RUN_CPP) and tests/test_zmq_inference.py (the
in-process server spin): the binary-dependent leg runs only with CHOCO_RUN_CPP=1 AND a freshly built
`chocofarm-zmq-net-probe` AND pyzmq present. It SKIPS (never fails) when any of those is absent, so the
DEFAULT `pytest tests/ -q` stays green on a box without the C++ build / libzmq / a server. NO redis: the
server is spun in-process with StaticParamsSource (params injected directly).

  ── Run the round-trip with the sandbox DISABLED (zmq needs a real context):
       cmake --build cpp/build && \
       CHOCO_RUN_CPP=1 PYTHONPATH=. python -m pytest tests/test_zmq_net_cpp.py -q -s

Two opt-in legs:
  * the round-trip parity: N≥100 random float32 feature vectors through the C++ client vs local
    float32 forward_core, max|Δvalue| AND max|Δlogit| < 1e-4, residual ON and OFF;
  * the loud-failure path: a ZmqNetClient pointed at a DEAD endpoint → predict() returns an Error (a
    bounded recv-timeout, NOT a hang), and a wrong-length reply is rejected (exercised via the probe's
    --probe-down mode against an unbound endpoint).

Public Domain (The Unlicense).
"""
import os
import subprocess
import sys
import threading

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROBE_BIN = os.path.join(REPO, "cpp", "build", "chocofarm-zmq-net-probe")

# OPT-IN gate (mirrors test_cpp_runner's CHOCO_RUN_CPP): the binary-dependent round-trip runs only with
# CHOCO_RUN_CPP=1 and a freshly built probe. A stale binary should skip, not red the default suite.
_RUN_CPP = bool(os.environ.get("CHOCO_RUN_CPP"))
_CPP_SKIP = ("opt-in cpp zmq round-trip: set CHOCO_RUN_CPP=1, build the probe fresh "
             "(cmake --build cpp/build), and run with the sandbox DISABLED (zmq needs a real context)")

ABS_TOL = 1e-4


def _pyzmq_available() -> bool:
    try:
        import zmq  # noqa: F401
        return True
    except ImportError:
        return False


def _py_forward_f32(params, X, y_mean, y_std):
    """The local reference at the float32 precision the service runs at: float32 weights + float32 X
    through the ONE forward_core, then de-standardize (v_std·y_std + y_mean) in float32 — the SAME
    quantity the C++ NetForward computes (cpp/parity/net_parity.py pins C++ NetForward to THIS). The
    only gap to the server is the batched matmul's row-vs-single reorder (§4), inside the 1e-4 bar."""
    from chocofarm.az.forward import forward_core
    p32 = {k: np.asarray(v, dtype=np.float32) for k, v in params.items()}
    x = np.asarray(X, dtype=np.float32)
    v_std, logits = forward_core(p32, x, np)
    value = np.asarray(v_std, dtype=np.float32) * np.float32(y_std) + np.float32(y_mean)
    return (np.asarray(value, dtype=np.float32),
            None if logits is None else np.asarray(logits, dtype=np.float32))


def _probe_predict(endpoint, X, timeout_ms=10000):
    """Drive the C++ probe over the running server: feed every row of X on stdin, parse (value, logits)
    per output line. Returns (values[N], logits[N]|None)."""
    # one feature vector per LINE (space-separated floats), one line per row of X
    stdin = "\n".join(" ".join(repr(float(v)) for v in row) for row in X) + "\n"
    out = subprocess.run([PROBE_BIN, "--endpoint", endpoint, "--timeout-ms", str(timeout_ms)],
                         input=stdin, capture_output=True, text=True, timeout=120)
    if out.returncode != 0:
        raise RuntimeError(f"zmq-net-probe failed (rc={out.returncode}): {out.stderr}")
    rows = [ln for ln in out.stdout.splitlines() if ln.strip()]
    assert len(rows) == len(X), (len(rows), len(X), out.stderr)
    values = np.empty(len(rows), dtype=np.float32)
    logits = None
    for i, ln in enumerate(rows):
        parts = ln.split()
        values[i] = np.float32(float(parts[0]))
        rest = [np.float32(float(p)) for p in parts[1:]]
        if logits is None:
            logits = np.empty((len(rows), len(rest)), dtype=np.float32) if rest else None
        if rest:
            logits[i] = rest
    return values, logits


@pytest.mark.skipif(not (_RUN_CPP and os.path.exists(PROBE_BIN) and _pyzmq_available()),
                    reason=_CPP_SKIP)
@pytest.mark.parametrize("residual", [False, True])
def test_cpp_zmq_client_round_trip_within_1e_4(residual):
    """The cross-language SSOT round-trip (design §7): spin the InferenceServer in-process with params
    injected DIRECTLY (StaticParamsSource — NO redis), drive the C++ ZmqNetClient probe with N≥100
    random float32 feature vectors, and assert `value` AND each `logit` within 1e-4 of the local float32
    forward_core (the reference the C++ NetForward is pinned to). residual ON and OFF — proving the C++
    encode → server forward_core → C++ decode path is faithful end-to-end."""
    from chocofarm.az.inference_server import (
        InferenceServer,
        StaticParamsSource,
        params_from_manifest_blob,
    )
    from chocofarm.az.mlp import ValueMLP
    from chocofarm.az.transport import pack_net

    in_dim, n_actions = 12, 7
    rng = np.random.default_rng(4096 + int(residual))
    net = ValueMLP(in_dim=in_dim, hidden=24, n_actions=n_actions, seed=17,
                   y_mean=float(rng.normal()) * 2.0, y_std=float(abs(rng.normal()) + 0.5),
                   residual=residual)
    params, y_mean, y_std = params_from_manifest_blob(*pack_net(net))
    src = StaticParamsSource(params, y_mean, y_std)

    endpoint = f"tcp://127.0.0.1:{5740 + int(residual)}"
    server = InferenceServer(src, bind=endpoint, max_batch=256)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()

    n_inputs = 200   # ≥100
    X = rng.standard_normal((n_inputs, in_dim)).astype(np.float32)
    ref_v, ref_l = _py_forward_f32(params, X, y_mean, y_std)

    try:
        got_v, got_l = _probe_predict(endpoint, X)
    finally:
        # Clean shutdown (no socket killed from another thread): flip stop, let the bounded poll observe
        # it and the serve thread exit, THEN close.
        server.stop()
        t.join(timeout=5.0)
        server.close()

    max_dv = float(np.max(np.abs(got_v.astype(np.float64) - ref_v.astype(np.float64))))
    assert max_dv < ABS_TOL, f"residual={residual}: max|Δvalue|={max_dv:.3e} exceeds {ABS_TOL:.0e}"
    max_dl = 0.0
    if ref_l is not None:
        assert got_l is not None
        max_dl = float(np.max(np.abs(got_l.astype(np.float64) - ref_l.astype(np.float64))))
        assert max_dl < ABS_TOL, f"residual={residual}: max|Δlogit|={max_dl:.3e} exceeds {ABS_TOL:.0e}"
    print(f"[cpp zmq round-trip residual={'ON ' if residual else 'OFF'}] N={n_inputs} "
          f"max|Δvalue|={max_dv:.3e} max|Δlogit|={max_dl:.3e} (bar {ABS_TOL:.0e})")


@pytest.mark.skipif(not (_RUN_CPP and os.path.exists(PROBE_BIN) and _pyzmq_available()),
                    reason=_CPP_SKIP)
def test_cpp_zmq_client_server_down_is_typed_error_not_hang():
    """The loud-failure path (design §5 / ADR-0012 P9 rule 5): a ZmqNetClient pointed at a DEAD endpoint
    (nothing bound) must return a TYPED Error from predict() — a bounded recv-timeout, NOT a forever
    hang and NOT a silent fallback. Driven via the probe's --probe-down mode with a SHORT timeout; the
    subprocess timeout (well above the recv timeout) would itself fail the test if predict() hung."""
    # an endpoint with no server bound — connect succeeds (lazy), the RPC's recv must time out loudly.
    dead = "tcp://127.0.0.1:5777"
    out = subprocess.run([PROBE_BIN, "--endpoint", dead, "--timeout-ms", "800", "--probe-down"],
                         capture_output=True, text=True, timeout=15)  # >> 800ms; a hang fails here
    assert out.returncode == 0, f"server-down probe rc={out.returncode}: {out.stdout}\n{out.stderr}"
    assert out.stdout.startswith("DOWN_OK"), out.stdout
    assert "timed out" in out.stdout.lower() or "NOT falling back" in out.stdout, out.stdout
    print(f"[cpp zmq server-down] {out.stdout.strip()}")


@pytest.mark.skipif(not (_RUN_CPP and os.path.exists(PROBE_BIN) and _pyzmq_available()),
                    reason=_CPP_SKIP)
def test_cpp_zmq_client_rejects_wrong_length_reply():
    """The malformed-reply boundary (design §5 / ADR-0002): a server that replies with a WRONG-LENGTH
    frame (a valid response truncated by one float) must be REJECTED by the C++ client's codec —
    predict() returns a typed Error (the loud-failure arm), never a misread/zero-filled NetPrediction.
    A bare Python ROUTER replies with a deliberately truncated frame; the probe's --probe-down mode
    expects predict() to fail."""
    import zmq

    from chocofarm.az import inference_wire as wire

    ctx = zmq.Context()
    sock = ctx.socket(zmq.ROUTER)
    endpoint = "tcp://127.0.0.1:5779"
    sock.bind(endpoint)
    # a VALID response frame, then truncated by one f32 — the codec must reject the byte-count mismatch.
    good = wire.encode_response(1.5, np.ones(3, dtype=np.float32))
    truncated = good[:-4]
    stop = threading.Event()

    def serve_one_truncated():
        poller = zmq.Poller()
        poller.register(sock, zmq.POLLIN)
        while not stop.is_set():
            if poller.poll(timeout=100):
                frames = sock.recv_multipart()      # [identity][empty][request]
                ident = frames[0]
                sock.send_multipart([ident, b"", truncated])

    t = threading.Thread(target=serve_one_truncated, daemon=True)
    t.start()
    try:
        out = subprocess.run([PROBE_BIN, "--endpoint", endpoint, "--timeout-ms", "3000", "--probe-down"],
                             capture_output=True, text=True, timeout=15)
    finally:
        stop.set()
        t.join(timeout=5.0)
        sock.close(linger=0)
        ctx.term()

    assert out.returncode == 0, f"malformed-reply probe rc={out.returncode}: {out.stdout}\n{out.stderr}"
    assert out.stdout.startswith("DOWN_OK"), out.stdout
    # the codec's wrong-length rejection (not a timeout) — the message names the byte-count mismatch.
    assert "logits block" in out.stdout or "expected" in out.stdout, out.stdout
    print(f"[cpp zmq wrong-length reply] {out.stdout.strip()}")
