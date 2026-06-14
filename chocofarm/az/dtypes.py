#!/usr/bin/env python3
"""
chocofarm AZ — the single parametric float dtype for the search hot path.

The hot path (features.py / mlp.py / gumbel_search.py) used to mix float64 implicitly. This
module pins ONE switchable precision so the whole pipeline is consistent and scannable. The
default is float32 — a real perf win (matmul bandwidth, the belief-feature blocks) and the
consistency fix at once — but float64 stays selectable per the maintainer's "keep float64 for
any piece that measurably degrades the rate" steer.

Usage
-----
    from chocofarm.az.dtypes import DTYPE
    out = np.empty(dim, dtype=DTYPE)

`DTYPE` is a numpy dtype object (np.float32 by default). To run the float64 variant — for the
behavioral-equivalence comparison, or if float32 ever measurably hurts the fixed-λ₀ rate — set
the env var before import:

    CHOCO_AZ_DTYPE=float64   (or float32; default float32)

The choice is read ONCE at import. The bench harness flips it via a small reload helper so each
precision is measured in isolation. Integer bit-mechanics (the world-set, cover masks) are
ALWAYS int64 — this knob governs only the real-valued feature / net arithmetic.
"""
from __future__ import annotations

import os

import numpy as np

_NAME = os.environ.get("CHOCO_AZ_DTYPE", "float32").strip().lower()
_ALLOWED = {"float32": np.float32, "float64": np.float64,
            "f32": np.float32, "f64": np.float64}
if _NAME not in _ALLOWED:
    # ADR-0002 fail-loud: an unrecognised precision request is a configuration error, not a
    # thing to silently coerce to a default.
    raise ValueError(
        f"CHOCO_AZ_DTYPE={_NAME!r} is not a recognised precision "
        f"(allowed: {sorted(_ALLOWED)})")

DTYPE = np.dtype(_ALLOWED[_NAME])
DTYPE_NAME = DTYPE.name


def is_float32() -> bool:
    return DTYPE == np.dtype(np.float32)
