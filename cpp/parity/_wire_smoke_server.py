#!/usr/bin/env python3
"""
cpp/parity/_wire_smoke_server.py — a THROWAWAY in-process InferenceServer harness for the
wire-driver Phase A/B smoke (NOT a committed parity fixture; a session scratch helper).

Stands up the Python InferenceServer on a daemon thread over the PRODUCTION geometry
(hidden=256, chocofarm/data/instance.json + faces.json) using wire_server.build_server, runs a
subprocess command (the C++ wire-pool-bench OR the wire-batched smoke binary) against it on the
given endpoint, tears the server down (server.stop/thread.join/server.close — never a nohup'd
foreground sleep), and forwards the subprocess stdout/stderr + return code. The server is the SSOT
batched leaf evaluator; the corr-id frame is opaque transport round-trip (frames[1:-1]).

Usage:
    python _wire_smoke_server.py --endpoint <ipc://...|tcp://...> [--hidden 256] -- <cmd...>

Public Domain (The Unlicense).
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import threading

REPO = "/home/bork/w/vdc/1/chocofarm"
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(os.path.expanduser("~"), "w", "vdc", "chocobo", "profiles"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--max-batch", type=int, default=256)
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    cmd = a.cmd
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]
    if not cmd:
        print("no subprocess command given (after --)", file=sys.stderr)
        return 2

    from wire_server import build_server

    server, in_dim, n_actions = build_server(a.hidden, a.endpoint, a.max_batch)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    print(f"[smoke_server] up hidden={a.hidden} in_dim={in_dim} n_actions={n_actions} "
          f"endpoint={a.endpoint}", flush=True)
    try:
        proc = subprocess.run(cmd, cwd=REPO, text=True, timeout=600)
        rc = proc.returncode
    finally:
        server.stop()
        t.join(timeout=5.0)
        server.close()
        print("[smoke_server] stopped", flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
