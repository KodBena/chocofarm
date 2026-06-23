#!/usr/bin/env python3
"""
throughput-lab/tests/test_wire_boundary.py — Hypothesis discharge of the refined boundary type
(server/wire.py BoundedBatch / decode_bounded) and the bucket-ladder totality (server/lifted/
mlp_forward.py pack). This is Annex B of the ratified resolution: the contracts the type+deal carry
are GATES, and a property suite is what turns "the type makes the illegal state unrepresentable" from
a claim into a discharged fact — feeding rows in [0, 2*max_batch] and off-in_dim widths and asserting
the boundary rejects exactly the illegal ones, and that `pack` is total (no raise, no recompile, the
pad tail sliced back) on exactly the legal ones.

Run (deal contracts ON — tests are where the spec is enforced; the serving path strips them):
    PYTHONPATH=throughput-lab /home/bork/w/vdc/venvs/generic/bin/python -m pytest \
        throughput-lab/tests/test_wire_boundary.py -q

Public Domain (The Unlicense).
"""
from __future__ import annotations

import struct

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

from server.wire import (
    PROTOCOL_VERSION,
    BoundedBatch,
    WireError,
    decode_bounded,
)
from server.lifted.mlp_forward import MlpForward

# A small geometry so the warmed XLA kernels compile fast; the law under test is shape-agnostic. The
# top of the ladder MUST equal MAX_BATCH (the server builds it that way), so a BoundedBatch's
# rows <= max_batch guarantee chains into pack's rows <= top-bucket precondition (pack stays total).
IN_DIM = 8
MAX_BATCH = 16
LADDER = [1, 4, 16]


@pytest.fixture(scope="module")
def forward() -> MlpForward:
    f = MlpForward.random_net(in_dim=IN_DIM, hidden=16, n_actions=0, seed=0)
    f.warmup(LADDER, IN_DIM)
    return f


# ---- BoundedBatch: legal BY CONSTRUCTION, illegal LOUDLY rejected (clauses 2/4/5) ---------------

@given(rows=st.integers(min_value=0, max_value=2 * MAX_BATCH),
       cols=st.integers(min_value=1, max_value=IN_DIM + 4))
@settings(max_examples=150, deadline=None)
def test_bounded_batch_legal_iff_in_bounds(rows: int, cols: int) -> None:
    """A BoundedBatch constructs iff the shape is legal (1 <= rows <= max_batch and cols == in_dim);
    every illegal shape is a loud WireError at the door, never a half-built object."""
    X = np.zeros((rows, cols), np.float32)
    legal = (1 <= rows <= MAX_BATCH) and (cols == IN_DIM)
    if legal:
        b = BoundedBatch(max_batch=MAX_BATCH, in_dim=IN_DIM, X=X)
        assert b.X.shape == (rows, cols)
    else:
        with pytest.raises(WireError):
            BoundedBatch(max_batch=MAX_BATCH, in_dim=IN_DIM, X=X)


def test_bounded_batch_rejects_non_2d_and_bad_dtype() -> None:
    """The two refinements the row/col sweep does not reach: rank and dtype."""
    with pytest.raises(WireError):
        BoundedBatch(max_batch=MAX_BATCH, in_dim=IN_DIM, X=np.zeros((IN_DIM,), np.float32))   # 1-D
    with pytest.raises(WireError):
        BoundedBatch(max_batch=MAX_BATCH, in_dim=IN_DIM, X=np.zeros((2, IN_DIM), np.float64))  # f64


# ---- decode_bounded: the same law enforced at the WIRE boundary, against the SERVER's geometry ---

@given(B=st.integers(min_value=0, max_value=2 * MAX_BATCH),
       frame_in_dim=st.integers(min_value=1, max_value=IN_DIM + 4))
@settings(max_examples=150, deadline=None)
def test_decode_bounded_boundary(B: int, frame_in_dim: int) -> None:
    """A self-consistent frame (its byte count matches its own [B][in_dim]) is accepted iff it ALSO
    satisfies the SERVER's law (1 <= B <= max_batch, in_dim == server in_dim) — the gap decode_request
    alone cannot close. Everything else is a loud WireError at the boundary, never a downstream crash."""
    body = np.zeros((B, frame_in_dim), "<f4").tobytes() if B > 0 else b""
    frame = struct.pack("<BII", PROTOCOL_VERSION, B, frame_in_dim) + body
    server_legal = (1 <= B <= MAX_BATCH) and (frame_in_dim == IN_DIM)
    if server_legal:
        bb = decode_bounded(frame, max_batch=MAX_BATCH, in_dim=IN_DIM)
        assert bb.X.shape == (B, frame_in_dim)
    else:
        with pytest.raises(WireError):
            decode_bounded(frame, max_batch=MAX_BATCH, in_dim=IN_DIM)


# ---- pack: TOTAL on the ladder — no raise, no recompile, the pad tail sliced back (clause 5) -----

@given(rows=st.integers(min_value=1, max_value=MAX_BATCH))
@settings(max_examples=40, deadline=None)
def test_pack_is_total_on_the_ladder(forward: MlpForward, rows: int) -> None:
    """For every legal row count (1..max_batch), a BoundedBatch's matrix packs into a WARMED bucket and
    forwards without raising or recompiling, and the result is sliced back to the real row count. This
    is what 'pack is total' means once the boundary guarantees rows <= max_batch == top-of-ladder; the
    deal @pre/@ensure on pack are exercised (ON in tests) on every example."""
    X = np.zeros((rows, IN_DIM), np.float32)
    b = BoundedBatch(max_batch=MAX_BATCH, in_dim=IN_DIM, X=X)
    packed = forward.pack(b.X)                 # must not raise (deal @pre holds) ...
    assert packed.x.shape[0] in set(LADDER)    # ... and lands on a warmed bucket (deal @ensure) ...
    assert packed.n_real == rows
    values, logits = forward.forward_batch(packed)
    assert values.shape[0] == rows             # pad tail sliced off — real rows returned
    assert logits is None                      # value-only net (n_actions == 0)
