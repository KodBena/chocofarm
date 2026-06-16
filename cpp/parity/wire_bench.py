#!/usr/bin/env python3
"""
cpp/parity/wire_bench.py — the over-the-wire SYNCHRONOUS benchmark driver (Shape B).

Spins the Python InferenceServer in-process (StaticParamsSource — NO redis) with a dimension-matched
ValueMLP (in_dim = feature_dim(env), n_actions = n_action_slots(env), the same instance the C++ env
loads), then runs the C++ `chocofarm-wire-bench` against it: SerialRuntime driving the Gumbel-AZ search
where every leaf is a blocking REQ round-trip to the server. It reports the wire-synchronous throughput
(decisions/s) and the per-leaf round-trip cost — the "over-the-wire synchronous" axis of the §6-Q5
benchmark (one in-flight leaf at a time, the cost the wire-PARALLEL fiber+DEALER pool exists to hide).

Prints "RESULT: PASS ..." + exit 0 on a clean run, or a loud failure + nonzero. Opt-in: needs the C++
binary built and pyzmq present (the wrapping test in tests/test_cpp_runner.py gates on CHOCO_RUN_CPP=1).

Public Domain (The Unlicense).
"""
from __future__ import annotations

import os
import subprocess
import sys
import threading

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

WIRE_BENCH_BIN = os.path.join(REPO, "cpp", "build", "chocofarm-wire-bench")
DATA_INSTANCE = os.path.join(REPO, "chocofarm", "data", "instance.json")
DATA_FACES = os.path.join(REPO, "chocofarm", "data", "faces.json")
ENDPOINT = "tcp://127.0.0.1:5762"


def main() -> int:
    if not os.path.exists(WIRE_BENCH_BIN):
        print(f"RESULT: SKIP (binary not built: {WIRE_BENCH_BIN})")
        return 0
    try:
        import zmq  # noqa: F401
    except ImportError:
        print("RESULT: SKIP (pyzmq not available)")
        return 0

    from chocofarm.az.actions import n_action_slots
    from chocofarm.az.features import feature_dim
    from chocofarm.az.inference_server import (
        InferenceServer,
        StaticParamsSource,
        params_from_manifest_blob,
    )
    from chocofarm.az.mlp import ValueMLP
    from chocofarm.az.transport import pack_net
    from chocofarm.model.env import Environment

    # the live instance the C++ instance.json mirrors — so feature_dim / n_action_slots match the C++ env.
    env = Environment()
    in_dim, n_actions = feature_dim(env), n_action_slots(env)
    net = ValueMLP(in_dim, hidden=24, n_actions=n_actions, seed=17,
                   y_mean=0.0, y_std=1.0, residual=False)
    params, y_mean, y_std = params_from_manifest_blob(*pack_net(net))
    src = StaticParamsSource(params, y_mean, y_std)

    server = InferenceServer(src, bind=ENDPOINT, max_batch=256)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    print(f"[wire_bench] server up: in_dim={in_dim} n_actions={n_actions} endpoint={ENDPOINT}", flush=True)

    rc = 1
    try:
        out = subprocess.run(
            [WIRE_BENCH_BIN, "--instance", DATA_INSTANCE, "--faces", DATA_FACES,
             "--endpoint", ENDPOINT, "--tasks", "8", "--n-sims", "12", "--max-depth", "8"],
            cwd=REPO, capture_output=True, text=True, timeout=300)
        sys.stdout.write(out.stdout)
        if out.returncode != 0 or "RESULT: PASS" not in out.stdout:
            sys.stderr.write(out.stderr)
            print(f"RESULT: FAIL (wire-bench rc={out.returncode})")
            rc = 3
        else:
            rc = 0
    finally:
        # clean shutdown (no socket killed cross-thread): flip stop, let the bounded poll see it, then close.
        server.stop()
        t.join(timeout=5.0)
        server.close()
    return rc


if __name__ == "__main__":
    sys.exit(main())
