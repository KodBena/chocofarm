#!/usr/bin/env python3
"""
cpp/parity/wire_bench.py — the over-the-wire benchmark driver (Shape B), BOTH axes.

Spins the Python InferenceServer in-process (StaticParamsSource — NO redis) with a dimension-matched
ValueMLP (in_dim = feature_dim(env), n_actions = n_action_slots(env), the same instance the C++ env
loads), then runs, against that ONE server:

  * the over-the-wire SYNCHRONOUS bench (chocofarm-wire-bench): SerialRuntime, one in-flight leaf at a
    time — the wire RTT + un-batched single-row forward cost; and
  * the over-the-wire PARALLEL bench (chocofarm-wire-parallel-bench): K tree-fibers on one thread,
    batch-submitting parked leaves over a DEALER so the server batches them into one forward.

It reports both throughputs + the parallel/sync speedup — the §6-Q5 comparison. The parallel bench is the
ROUND-SYNCHRONOUS MVP (a barrier per round), so its win is modest and capped by per-round latency; the
continuous greedy-async work-stealing pool is the production refinement (a bigger win). Prints
"RESULT: PASS ..." + exit 0, or a loud failure / SKIP (pyzmq or a binary absent).

Public Domain (The Unlicense).
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import threading

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

BUILD = os.path.join(REPO, "cpp", "build")
SYNC_BIN = os.path.join(BUILD, "chocofarm-wire-bench")
PAR_BIN = os.path.join(BUILD, "chocofarm-wire-parallel-bench")
POOL_BIN = os.path.join(BUILD, "chocofarm-wire-pool-bench")
LOCAL_BIN = os.path.join(BUILD, "chocofarm-local-mlp-bench")

# The redis (run, phase, version) the harness publishes the SAME net under, so the C++-native
# local-mlp-bench reads the IDENTICAL weights the wire server is seeded from (the fairness anchor).
LOCAL_RUN, LOCAL_PHASE, LOCAL_VER = "wire-bench", "eval", 0
DATA_INSTANCE = os.path.join(REPO, "chocofarm", "data", "instance.json")
DATA_FACES = os.path.join(REPO, "chocofarm", "data", "faces.json")
ENDPOINT = "tcp://127.0.0.1:5762"


def _dps(text: str, key: str) -> float | None:
    m = re.search(key + r"=([0-9.eE+-]+)", text)
    return float(m.group(1)) if m else None


def main() -> int:
    if not os.path.exists(SYNC_BIN):
        print(f"RESULT: SKIP (binary not built: {SYNC_BIN})")
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

    env = Environment()
    in_dim, n_actions = feature_dim(env), n_action_slots(env)
    net = ValueMLP(in_dim, hidden=24, n_actions=n_actions, seed=17,
                   y_mean=0.0, y_std=1.0, residual=False)
    manifest, blob = pack_net(net)
    params, y_mean, y_std = params_from_manifest_blob(manifest, blob)
    server = InferenceServer(StaticParamsSource(params, y_mean, y_std), bind=ENDPOINT, max_batch=256)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    print(f"[wire_bench] server up: in_dim={in_dim} n_actions={n_actions} endpoint={ENDPOINT}", flush=True)

    # Publish the SAME (manifest, blob) to redis so the C++-native local-mlp-bench reads the IDENTICAL net
    # off the runner's weight-read seam — the only difference from the server is float64 weights vs the
    # NetForward's float32 cast (ADR-0012 P6, < 1e-4). A redis hiccup only SKIPS the local axis, never
    # fails the wire comparison.
    local_published = False
    if os.path.exists(LOCAL_BIN):
        try:
            import redis as _redis

            from chocofarm.az.transport import weight_keys
            from chocofarm.config import transport_redis_params
            mk, bk = weight_keys(LOCAL_RUN, LOCAL_PHASE, LOCAL_VER)
            conn = _redis.Redis(**transport_redis_params())
            conn.set(mk, manifest)
            conn.set(bk, blob)
            local_published = True
            print(f"[wire_bench] published weights to redis: {LOCAL_RUN}/{LOCAL_PHASE}/{LOCAL_VER} "
                  f"(for the C++-native local axis)", flush=True)
        except Exception as e:   # transport redis down: skip the local axis, keep the wire comparison
            print(f"[wire_bench] could not publish weights for local axis (skipping it): {e}", flush=True)

    common = ["--instance", DATA_INSTANCE, "--faces", DATA_FACES, "--endpoint", ENDPOINT,
              "--n-sims", "12", "--max-depth", "8"]
    rc = 0
    try:
        sync = subprocess.run([SYNC_BIN, *common, "--tasks", "16"],
                              cwd=REPO, capture_output=True, text=True, timeout=300)
        sys.stdout.write(sync.stdout)
        if sync.returncode != 0 or "RESULT: PASS" not in sync.stdout:
            sys.stderr.write(sync.stderr)
            print("RESULT: FAIL (wire-sync bench)")
            return 3
        sync_dps = _dps(sync.stdout, "wire_sync_dps")

        # the C++-NATIVE LOCAL axis: the SAME SerialRuntime batch, in-process NetForward on the redis
        # weights we just published — apples-to-apples with wire-sync (the only delta is local forward
        # vs wire RTT + server forward). Independent of the server; run it on the same 16-task budget.
        local_dps = None
        if local_published and os.path.exists(LOCAL_BIN):
            local = subprocess.run(
                [LOCAL_BIN, "--instance", DATA_INSTANCE, "--faces", DATA_FACES,
                 "--run", LOCAL_RUN, "--phase", LOCAL_PHASE, "--version", str(LOCAL_VER),
                 "--tasks", "16", "--n-sims", "12", "--max-depth", "8"],
                cwd=REPO, capture_output=True, text=True, timeout=300)
            sys.stdout.write(local.stdout)
            if local.returncode != 0 or "RESULT: PASS" not in local.stdout:
                sys.stderr.write(local.stderr)
                print("RESULT: FAIL (local-mlp bench)")
                return 3
            local_dps = _dps(local.stdout, "local_mlp_dps")

        if os.path.exists(PAR_BIN):
            par = subprocess.run([PAR_BIN, *common, "--trees", "16"],
                                 cwd=REPO, capture_output=True, text=True, timeout=300)
            sys.stdout.write(par.stdout)
            if par.returncode != 0 or "RESULT: PASS" not in par.stdout:
                sys.stderr.write(par.stderr)
                print("RESULT: FAIL (wire-parallel bench)")
                return 3
            par_dps = _dps(par.stdout, "wire_parallel_dps")
            pool_dps = None
            if os.path.exists(POOL_BIN):
                # the production greedy-async pool: T threads x K fibers, batch via fibers not threads.
                pool = subprocess.run([POOL_BIN, *common, "--tasks", "32", "--threads", "2", "--batch", "16"],
                                      cwd=REPO, capture_output=True, text=True, timeout=300)
                sys.stdout.write(pool.stdout)
                if pool.returncode != 0 or "RESULT: PASS" not in pool.stdout:
                    sys.stderr.write(pool.stderr)
                    print("RESULT: FAIL (wire-pool bench)")
                    return 3
                pool_dps = _dps(pool.stdout, "pool_dps")
            if sync_dps and par_dps:
                pool_x = f" wire_pool_dps={pool_dps:.3f} pool_speedup={pool_dps / sync_dps:.3f}" if pool_dps else ""
                local_x = (f" local_mlp_dps={local_dps:.3f} local_speedup={local_dps / sync_dps:.3f}"
                           if local_dps else "")
                print(f"RESULT: PASS wire_sync_dps={sync_dps:.3f} wire_parallel_dps={par_dps:.3f} "
                      f"speedup={par_dps / sync_dps:.3f}{pool_x}{local_x}")
            else:
                print("RESULT: PASS (ran; dps parse incomplete)")
        else:
            local_x = (f" local_mlp_dps={local_dps:.3f} local_speedup={local_dps / sync_dps:.3f}"
                       if (local_dps and sync_dps) else "")
            print(f"RESULT: PASS (wire-sync only; parallel binary not built: {PAR_BIN}){local_x}")
    finally:
        server.stop()
        t.join(timeout=5.0)
        server.close()
    return rc


if __name__ == "__main__":
    sys.exit(main())
