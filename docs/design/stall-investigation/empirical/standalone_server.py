#!/usr/bin/env python3
"""Standalone StageAServer launcher for stall diagnosis (mirrors overcommit_sweep's
build_and_publish + start_server). Publishes the net to redis, stands up the bucketed
group-wakeup server, prints SERVER_READY, then serves forever until SIGTERM/SIGINT.
Run under PYTHONFAULTHANDLER=1 so kill -QUIT dumps Python tracebacks.

Public Domain (The Unlicense)."""
import argparse, os, signal, sys, threading, time

REPO = "/home/bork/w/vdc/1/chocofarm-wt-stall"
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "cpp", "stage_a"))

import chocofarm.config  # noqa
from chocofarm.az.actions import n_action_slots
from chocofarm.az.features import feature_dim
from chocofarm.az.inference_server import (StaticParamsSource, jit_forward_core,
                                           params_from_manifest_blob)
from chocofarm.az.mlp import ValueMLP
from chocofarm.az.transport import RedisTransport, connect, pack_net
from chocofarm.model.env import Environment
from stage_a_server import BUCKETS, StageAServer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--version", type=int, default=0)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--max-batch", type=int, default=512)
    ap.add_argument("--server-core", type=int, default=0)
    a = ap.parse_args()

    env = Environment()
    in_dim, n_actions = feature_dim(env), n_action_slots(env)
    net = ValueMLP(in_dim, hidden=a.hidden, n_actions=n_actions, seed=17,
                   y_mean=0.0, y_std=1.0, residual=False)
    manifest, blob = pack_net(net)
    RedisTransport(connect()).publish_weights(net, phase="gen", version=a.version, run=a.run)
    params, y_mean, y_std = params_from_manifest_blob(manifest, blob)
    src = StaticParamsSource(params, y_mean, y_std)

    server = StageAServer(src, bind=a.endpoint, max_batch=a.max_batch,
                          forward_fn=jit_forward_core, e_policy="bucket", wakeup="group")
    server.warmup(sorted(set(BUCKETS) | {a.max_batch}))

    def _serve():
        try:
            os.sched_setaffinity(0, {a.server_core})
        except (OSError, AttributeError):
            pass
        server.serve_forever()

    t = threading.Thread(target=_serve, daemon=True)
    t.start()
    print(f"SERVER_READY endpoint={a.endpoint} run={a.run} v={a.version} in_dim={in_dim}", flush=True)

    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    while not stop.is_set():
        stop.wait(1.0)
        print(f"SERVER_HEARTBEAT forwards={server.n_forwards} rows={server.n_real_rows}", flush=True)
    server.stop(); t.join(timeout=5.0); server.close()


if __name__ == "__main__":
    main()
