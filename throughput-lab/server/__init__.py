#!/usr/bin/env python3
# throughput-lab/server/__init__.py — the throughput-lab Python server package.
#
# A self-contained, clean-room receive/serve loop for the producer->boundary->server throughput
# testbed. Only the MLP forward + the phantom-typed jax/numpy ACL are LIFTED verbatim from chocofarm
# (server/lifted/); everything else (the wire codec view in wire.py, the decoupled receive/serve
# loop in server.py) is re-implemented fresh here.
#
# Per ADR-0006, __init__.py is exempt from the module-docstring header convention; this comment is
# orientation only. Public Domain (The Unlicense).
