#!/usr/bin/env python3
"""
throughput-lab/server/lifted/dtypes.py — the single parametric float dtype for the forward, COPIED
(condensed) from chocofarm/az/dtypes.py. Part of the phantom-typed jax/numpy ACL lifted into the
clean-room testbed: it pins ONE switchable precision (float32 default) so the testbed's forward runs
at the SAME precision the production server runs.

PROVENANCE: copied (condensed) from chocofarm/az/dtypes.py @ d30fe8e (the DTYPE/DTYPE_NAME pin + the
fail-loud allow-list, verbatim; the parent's prose-only usage block was trimmed). Default float32
(the production default — a real
matmul-bandwidth win and the consistency fix at once). Set CHOCO_AZ_DTYPE=float64 before import for
the float64 variant. Read ONCE at import (ADR-0002 fail-loud on an unrecognized request).

Public Domain (The Unlicense).
"""
from __future__ import annotations

import os

import numpy as np

_NAME = os.environ.get("CHOCO_AZ_DTYPE", "float32").strip().lower()
_ALLOWED = {"float32": np.float32, "float64": np.float64,
            "f32": np.float32, "f64": np.float64}
if _NAME not in _ALLOWED:
    # ADR-0002 fail-loud: an unrecognised precision request is a configuration error, not a thing to
    # silently coerce to a default.
    raise ValueError(f"CHOCO_AZ_DTYPE={_NAME!r} is not a recognised precision (allowed: {sorted(_ALLOWED)})")

DTYPE = np.dtype(_ALLOWED[_NAME])
DTYPE_NAME = DTYPE.name


def is_float32() -> bool:
    return DTYPE == np.dtype(np.float32)
