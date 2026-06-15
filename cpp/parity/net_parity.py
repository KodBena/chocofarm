#!/usr/bin/env python3
"""
cpp/parity/net_parity.py — the ADR-0012 P6/P7 forward-parity harness for the C++ NetForward.

It validates that the C++ `NetForward` (cpp/src/net.cpp) reimplements the ONE Python forward
`chocofarm/az/forward.forward_core` to behavioral equivalence — the `test_jax_equivalence` bar
(ABS_TOL=1e-4), NOT byte-identity (float32 is not associative; a C++ reorder of the same math will
move the last bits — ADR-0012 P6).

Method:
  1. Build a `ValueMLP` (with a policy head) in Python — once with residual OFF, once with residual
     ON — and PUBLISH its weights to redis via the transport (the SAME manifest+blob the C++ reads).
  2. Feed N≥1000 float32 feature vectors (random, the right `in_dim`) through BOTH:
       * the C++ `chocofarm-net-dump` (reads the published net off redis via the manifest, runs the
         C++ NetForward), and
       * the Python reference forward — run on the SAME weights at the SAME float32 precision the C++
         uses (the `ValueMLP._predict_both_f32` regime: float32 weights + float32 X through
         `forward_core`, then de-standardize v_std*y_std + y_mean in float32), so the two sides are
         computing the SAME quantity in the SAME precision and the only gap is the C++ reorder.
  3. Assert max|Δvalue| < 1e-4 AND max|Δlogit| < 1e-4, residual ON and OFF; REPORT the actual max|Δ|.

Needs the C++ binary built (cpp/build/chocofarm-net-dump) and redis up (the CHOCO_TRANSPORT_REDIS_*
contract, 6380). Run (from repo root):
    PYTHONPATH=. /home/bork/w/vdc/venvs/generic/bin/python cpp/parity/net_parity.py

Public Domain (The Unlicense).
"""
from __future__ import annotations

import os
import subprocess
import sys
import uuid

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

from chocofarm.az import transport
from chocofarm.az.actions import n_action_slots
from chocofarm.az.features import feature_dim
from chocofarm.az.forward import forward_core
from chocofarm.az.mlp import ValueMLP
from chocofarm.model.env import Environment

NET_BIN = os.path.join(REPO, "cpp", "build", "chocofarm-net-dump")
ABS_TOL = 1e-4


def py_forward_f32(net: ValueMLP, X: np.ndarray):
    """The Python reference at the SAME float32 precision the C++ runs at — the `_predict_both_f32`
    regime: float32 weights + float32 X through the ONE `forward_core`, then de-standardize the value
    (v_std*ys + ym) in float32. Returns (value (n,), logits (n, n_actions)) as float32, with the value
    de-standardized exactly as `ValueMLP.predict_value`/`_predict_both_f32` (and so as the C++)."""
    c = net._f32_weights()           # float32 copies of the weights (the parametric hot-path cache)
    params = net._f32_params(c)      # the flat params dict keyed like forward_core consumes
    x = np.asarray(X, dtype=np.float32)
    v_std, logits = forward_core(params, x, np)
    value = v_std * c["ys"] + c["ym"]   # de-standardize in float32 (predict_value's inverse)
    return np.asarray(value, dtype=np.float32), np.asarray(logits, dtype=np.float32)


def cpp_forward(run_id, version, X: np.ndarray):
    """Feed every row of X through the C++ net-dump and parse (value, logits) per line. The first
    stdout line is a `# in_dim=.. n_actions=.. residual=..` header (asserted against the Python net)."""
    lines = []
    for row in X:
        lines.append(" ".join(repr(float(v)) for v in row))
    stdin = "\n".join(lines) + "\n"
    out = subprocess.run([NET_BIN, "--run", run_id, "--phase", "gen", "--version", str(version)],
                         input=stdin, capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(f"net-dump failed (rc={out.returncode}): {out.stderr}")
    rows = [ln for ln in out.stdout.splitlines() if ln.strip()]
    header = rows[0]
    assert header.startswith("#"), header
    meta = dict(tok.split("=") for tok in header[1:].split())
    body = rows[1:]
    assert len(body) == len(X), (len(body), len(X))
    values = np.empty(len(body), dtype=np.float32)
    logits = None
    for i, ln in enumerate(body):
        parts = ln.split()
        values[i] = np.float32(float(parts[0]))
        rest = [np.float32(float(p)) for p in parts[1:]]
        if logits is None:
            logits = np.empty((len(body), len(rest)), dtype=np.float32) if rest else None
        if rest:
            logits[i] = rest
    return values, logits, meta


def run_case(env, residual: bool, n_inputs: int, seed: int):
    """Build a residual-{ON,OFF} ValueMLP, publish it, push n_inputs random float32 feature vectors
    through both sides, and return (max|Δvalue|, max|Δlogit|, n_inputs, meta)."""
    fd = feature_dim(env)
    ns = n_action_slots(env)
    rng = np.random.default_rng(seed)
    # a real-ish net: nonzero y-scale (de-standardization is exercised), a policy head, residual toggle
    net = ValueMLP(fd, hidden=48, n_actions=ns, seed=seed,
                   y_mean=float(rng.normal()) * 2.0, y_std=float(abs(rng.normal()) + 0.5),
                   residual=residual)
    run_id = "netparity-" + uuid.uuid4().hex[:10]
    version = 0
    conn = transport.connect()
    transport.RedisTransport(conn).publish_weights(net, "gen", version, run_id)

    # N≥1000 random float32 feature vectors of the right dim (the leaf evaluator's input contract).
    X = rng.standard_normal((n_inputs, fd)).astype(np.float32)

    py_v, py_l = py_forward_f32(net, X)
    cpp_v, cpp_l, meta = cpp_forward(run_id, version, X)

    # the C++ must have DERIVED the same dims/toggle from the manifest (P1)
    assert int(meta["in_dim"]) == fd, (meta, fd)
    assert int(meta["n_actions"]) == ns, (meta, ns)
    assert bool(int(meta["residual"])) == residual, (meta, residual)

    dv = float(np.max(np.abs(py_v.astype(np.float64) - cpp_v.astype(np.float64))))
    dl = float(np.max(np.abs(py_l.astype(np.float64) - cpp_l.astype(np.float64))))
    return dv, dl, n_inputs, meta


def main():
    if not os.path.exists(NET_BIN):
        print(f"FAIL: C++ binary not built at {NET_BIN}\n"
              f"      build it: cmake -S cpp -B cpp/build && cmake --build cpp/build")
        return 1

    env = Environment()
    fd = feature_dim(env)
    ns = n_action_slots(env)
    n_inputs = 1500   # ≥1000

    print("=== ADR-0012 P6/P7 forward parity: C++ NetForward vs Python forward_core ===")
    print(f"in_dim={fd} n_actions={ns}  N={n_inputs} random float32 feature vectors/case  "
          f"(bar ABS_TOL={ABS_TOL:.0e}, the test_jax_equivalence bar)\n")

    ok = True
    for residual in (False, True):
        dv, dl, n, meta = run_case(env, residual, n_inputs, seed=1234 + int(residual))
        v_ok = dv < ABS_TOL
        l_ok = dl < ABS_TOL
        case_ok = v_ok and l_ok
        ok = ok and case_ok
        tag = "residual ON " if residual else "residual OFF"
        print(f"[{tag}] over {n} inputs (C++-derived: in_dim={meta['in_dim']} "
              f"n_actions={meta['n_actions']} residual={meta['residual']}):")
        print(f"    max|Δvalue| = {dv:.3e}  -> {'OK' if v_ok else 'DIVERGE'}")
        print(f"    max|Δlogit| = {dl:.3e}  -> {'OK' if l_ok else 'DIVERGE'}")

    print()
    if ok:
        print("RESULT: PASS — C++ NetForward matches Python forward_core to < 1e-4 on value AND "
              "logits, residual ON and OFF")
    else:
        print("RESULT: FAIL — a value or logit max|Δ| exceeded ABS_TOL=1e-4")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
