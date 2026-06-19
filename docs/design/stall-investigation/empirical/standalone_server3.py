#!/usr/bin/env python3
"""Instrumented standalone server for stall discrimination.
 - ROUTER_MANDATORY=1: a reply that cannot be routed to its DEALER raises (EHOSTUNREACH)
   instead of being silently dropped -> turns a reply-drop into a LOUD log line.
 - counts recv messages and sent replies; prints them in the heartbeat.
This tells us: reply-drop (mandatory raises) vs send-drop (recv count lags, no raise).
Public Domain (The Unlicense)."""
import argparse, os, signal, sys, threading, time

REPO = "/home/bork/w/vdc/1/chocofarm-wt-stall"
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "cpp", "stage_a"))

import chocofarm.config  # noqa
import zmq
from chocofarm.az.actions import n_action_slots
from chocofarm.az.features import feature_dim
from chocofarm.az.inference_server import (StaticParamsSource, jit_forward_core,
                                           params_from_manifest_blob, run_microbatch)
from chocofarm.az.mlp import ValueMLP
from chocofarm.az.transport import RedisTransport, connect, pack_net
from chocofarm.model.env import Environment
from stage_a_server import BUCKETS, StageAServer, _bucket_for


class CountingServer(StageAServer):
    _POLL_INTERVAL_MS = 2  # stall-test: drop the 100ms server idle-poll cadence
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.recv_msgs = 0
        self.sent_msgs = 0
        self.route_errors = 0
        # ROUTER_MANDATORY: surface an unroutable reply as a raise, not a silent drop.
        self._sock.setsockopt(zmq.ROUTER_MANDATORY, 1)

    def _drain(self):
        d = super()._drain()
        self.recv_msgs += len(d)
        return d

    def _serve_batch(self, drained):  # type: ignore[override]
        params, y_mean, y_std = self._params_source.current()
        rows = [(ident, X) for ident, _e, X in drained]
        real = int(sum(X.shape[0] for _i, X in rows))
        pad_to = _bucket_for(real)
        responses = run_microbatch(self._forward_fn, params, y_mean, y_std, rows, pad_to=pad_to)
        self.n_forwards += 1
        self.n_real_rows += real
        for (ident, resp), (_ident, envelope, _X) in zip(responses, drained):
            try:
                self._sock.send_multipart([ident, *envelope, resp])
                self.sent_msgs += 1
            except zmq.ZMQError as e:
                self.route_errors += 1
                print(f"ROUTE_ERROR (reply drop would-have-been-silent): {e} "
                      f"ident={ident!r} recv={self.recv_msgs} sent={self.sent_msgs}", flush=True)


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

    server = CountingServer(src, bind=a.endpoint, max_batch=a.max_batch,
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
    print(f"SERVER_READY endpoint={a.endpoint} run={a.run} v={a.version}", flush=True)

    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    while not stop.is_set():
        stop.wait(1.0)
        print(f"SERVER_HEARTBEAT fwd={server.n_forwards} recv_msgs={server.recv_msgs} "
              f"sent_msgs={server.sent_msgs} route_errors={server.route_errors}", flush=True)
    server.stop(); t.join(timeout=5.0); server.close()


if __name__ == "__main__":
    main()
