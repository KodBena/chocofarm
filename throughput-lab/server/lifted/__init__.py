#!/usr/bin/env python3
# throughput-lab/server/lifted/__init__.py — the TRUSTED parts COPIED VERBATIM from chocofarm.
#
# This subpackage holds the ONLY code lifted from the parent (the testbed is otherwise clean-room):
#   * the MLP forward graph (forward_core — chocofarm/az/forward.py), and
#   * the phantom-typed jax/numpy ACL it runs over (the `xp` array-module seam + the dtype pin).
# These are COPIED, not imported, so throughput-lab is self-contained (it does not depend on the
# chocofarm package being importable). Each lifted file records its chocofarm provenance in its header
# so a drift between the copy and the original is traceable (ADR-0005/0006).
#
# Per ADR-0006, __init__.py is exempt from the header convention. Public Domain (The Unlicense).
