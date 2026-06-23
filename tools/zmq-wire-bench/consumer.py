#!/usr/bin/env python3
"""
tools/zmq-wire-bench/consumer.py — isolated ZMQ wire benchmark (consumer side): a minimal ROUTER ECHO.

Receives `[ident][corr][payload]` and replies `[ident][corr][B*out_dim zero-floats]` — NO codec, NO real net,
so the measured round-trip is the WIRE (ZMQ transit + framing) ALONE, not the forward. It deliberately echoes
the production-shaped ASYMMETRIC sizes (request B*in_dim, reply B*out_dim) so the return transit matches. The
point is the counterfactual to the lab's step-4 gap: if this raw wire is tens of us while the lab gap is ~1 ms,
the lab gap is the producer's search-wait, not the wire.

args: <endpoint> <in_dim> <out_dim>
Public Domain (The Unlicense).
"""
import sys

import zmq

endpoint, in_dim, out_dim = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
ctx = zmq.Context()
sock = ctx.socket(zmq.ROUTER)
sock.bind(endpoint)
req_row_bytes = in_dim * 4
try:
    while True:
        frames = sock.recv_multipart()                 # [ident][corr][payload]
        ident, corr, payload = frames[0], frames[1], frames[-1]
        B = len(payload) // req_row_bytes              # rows in this request
        sock.send_multipart([ident, corr, bytes(B * out_dim * 4)])   # B*out_dim zero floats
except KeyboardInterrupt:
    pass
finally:
    sock.close(0)
    ctx.term()
