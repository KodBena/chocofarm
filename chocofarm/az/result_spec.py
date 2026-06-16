#!/usr/bin/env python3
"""
chocofarm/az/result_spec.py — the ONE authoritative declaration of the redis RESULT blob's byte
format (the worker→parent training-record transport, `transport.py` write/read + `worker.py` stack +
the C++ `transport.cpp::write_results`). This is the single source of truth (ADR-0012 P1/P7) for the
four float32 blocks the worker emits and the parent decodes — the format most prone to silent drift,
because BOTH the Python `np.frombuffer(..., dtype=np.float32)` read and the C++ `std::span<const
float>` / `sizeof(float)` write hardcoded "float32" independently, with nothing reconciling them.

The format (the worker-transition record on the wire), spelled once here and nowhere else:

    For one task `idx`, four CONTIGUOUS little-endian float32 blocks, row-major, under the keys
    az:res:<token>:<idx>:{X, PI, M, Y}:

        X  : (n, feat_dim)  float32   — the feature rows
        PI : (n, n_slots)   float32   — the policy targets (per action slot)
        M  : (n, n_slots)   float32   — the legality mask (per action slot)
        Y  : (n,)           float32   — the scalar λ-penalized-return targets

`n` is the record count for the task; `feat_dim` / `n_slots` are the net's feature/action dims. Each
block is the raw `tobytes()` of a contiguous float32 array — `np.frombuffer(blob, np.float32).reshape
(...)` decodes it byte-for-byte. The C++ `cpp/include/chocofarm/result_spec.hpp` mirror declares the
SAME block order + dtype + ranks; the constants are DRIFT-CHECKED against this module in the default
suite (tests/test_wire_drift.py), so a one-sided change (a fifth block, a reorder, a float64 widening)
reds `pytest tests/ -q` rather than corrupting a reshape silently (ADR-0002 / ADR-0011 Rule 4).

Why a spec module
-----------------
`transport.py` and `worker.py` (Python) and `transport.cpp` (C++) each hardcoded the dtype `float32`
and the X/PI/M/Y order. A dtype widening (float32→float64) or a block reorder on one side would have
been a SILENT reshape corruption — read floats, no exception, wrong numbers — exactly the
silent-serialization defect ADR-0002 forbids and the #23 result-blob target. This module is the one
writer of those facts; the readers/writers DERIVE the dtype and the block order from it, and the C++
mirror + drift test are the net.

Public Domain (The Unlicense).
"""
from __future__ import annotations

from typing import Final

import numpy as np

# ---- the per-block float dtype (the ONE place "the result blocks are float32" is spelled) ----
# Little-endian float32 (numpy '<f4'). All four blocks share it. A widening to float64 is a one-line
# edit here that the drift test reconciles against the C++ mirror (where it is `float` / sizeof 4).
RESULT_DTYPE: Final[np.dtype] = np.dtype("<f4")
RESULT_DTYPE_STR: Final[str] = RESULT_DTYPE.str       # '<f4'
RESULT_ITEMSIZE: Final[int] = RESULT_DTYPE.itemsize   # 4 (bytes per float32)

# ---- the block ORDER + ranks (the canonical X/PI/M/Y sequence the keys + reshape commit to) ----
# The four result keys are `az:res:<token>:<idx>:{X,PI,M,Y}` (transport.result_keys spells the key
# strings; THIS spells the block semantics: the order, the rank, the second-dim source). A block is
# either 2-D (its row is `n` long, its column the named feature/slot dim) or 1-D (the scalar target).
BLOCK_X: Final[str] = "X"      # (n, feat_dim) — feature rows
BLOCK_PI: Final[str] = "PI"    # (n, n_slots)  — policy targets
BLOCK_M: Final[str] = "M"      # (n, n_slots)  — legality mask
BLOCK_Y: Final[str] = "Y"      # (n,)          — scalar λ-return targets

# The canonical block order (the sequence write_results emits and read_and_delete_results decodes). A
# tuple so it is immutable and a reorder is a deliberate edit. Mirrored in result_spec.hpp.
BLOCK_ORDER: Final[tuple[str, ...]] = (BLOCK_X, BLOCK_PI, BLOCK_M, BLOCK_Y)

# Per-block rank (2 = two-dimensional (n, dim); 1 = one-dimensional (n,)) — the reshape rule. X is
# (n, feat_dim); PI and M are (n, n_slots); Y is (n,). Carried so the C++ size sanity-check and the
# Python reshape both derive the rank from one source, not a hardcoded `.reshape(n, ...)` each.
BLOCK_RANK: Final[dict[str, int]] = {BLOCK_X: 2, BLOCK_PI: 2, BLOCK_M: 2, BLOCK_Y: 1}
