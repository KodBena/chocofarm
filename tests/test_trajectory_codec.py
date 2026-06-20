#!/usr/bin/env python3
"""
tests/test_trajectory_codec.py — the round-trip + fail-loud net for the control-lab trajectory codec
(cpp/stage_a/control_lab/trajectory_codec.py). The codec is a BESPOKE columnar, per-feature-compressed
serialization of the (Observation, action, reward) decision stream; this suite pins the two things that
matter: (1) decode(encode(x)) is EXACT on synthetic data matching the real lab shape (T in {1,2,3,4},
D=8, K≈54, monotone counters, mostly-stable gates, sentinel-0 rtt_us/server_rows_per_forward), including
property-based (hypothesis) round-trips over the whole shape space; and (2) every decode BOUNDARY fails
LOUD on a corrupt/drifted blob (bad magic, wrong version, schema-hash mismatch, codec drift, truncation)
— ADR-0002, never a silently misread trajectory.

The bound math is checked too: inflight is BITPACK'd at ceil(log2(D+1)) bits, ready at ceil(log2(K+1))
bits — the absolute-value bounds provable from TrialContext — and a value that overflows the declared
bound is a loud violation (the `check=True` append assert), not a silent truncation.

Public Domain (The Unlicense).
"""
from __future__ import annotations

import os
import struct
import sys
from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_STAGE_A = os.path.join(REPO, "cpp", "stage_a")
for _p in (REPO, _STAGE_A):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from control_lab import trajectory_codec as tc          # noqa: E402
from control_lab.adapter import Observation, TrialContext  # noqa: E402

try:
    from hypothesis import HealthCheck, given, settings
    from hypothesis import strategies as st
    _HAS_HYPOTHESIS = True
except ImportError:                                       # pragma: no cover - hypothesis optional
    _HAS_HYPOTHESIS = False


# ============================================================================================
# A reference trajectory generator that matches the REAL lab shape (the same surface
# lab_server._run_controller builds: length-T inflight/ready/msgs/leaves/rtt_us, a variable served subset,
# a mostly-stable gate vector, sentinel-0 rtt_us + server_rows_per_forward). Returns the buffer + the
# ground-truth arrays so a test asserts exact equality column-by-column.
# ============================================================================================
@dataclass
class _Truth:
    inflight: np.ndarray
    ready: np.ndarray
    msgs: np.ndarray
    leaves: np.ndarray
    rtt_us: np.ndarray
    served: np.ndarray
    action: np.ndarray
    forward_rows: np.ndarray
    reward: np.ndarray
    t_monotonic: np.ndarray


def _make_trajectory(T: int, D: int, K: int, n: int, seed: int, *,
                     gate_flip_p: float = 0.02, rtt_sentinel: bool = True,
                     srv_sentinel: bool = True, initial_cap: int = 1 << 12,
                     check: bool = True) -> tuple[tc.TrajectoryBuffer, _Truth]:
    """Build a realistic trajectory: monotone per-thread cumulative msgs/leaves, inflight in [0,D], ready in
    [0,K], a variable served subset, a gate vector that flips rarely (gate_flip_p), sentinel-0 rtt_us +
    server_rows_per_forward. Returns the appended buffer + the ground-truth columns."""
    rng = np.random.default_rng(seed)
    ctx = TrialContext(n_threads=T, d_ceiling=D, k_per_thread=K, s_min=32, chunk_floor=False, seed=seed)
    buf = tc.TrajectoryBuffer(ctx, initial_cap=initial_cap, check=check)

    inflight = np.zeros((n, T), dtype=np.int64)
    ready = np.zeros((n, T), dtype=np.int64)
    msgs = np.zeros((n, T), dtype=np.int64)
    leaves = np.zeros((n, T), dtype=np.int64)
    rtt_us = np.zeros((n, T), dtype=np.int64)
    served = np.zeros((n, T), dtype=np.uint8)
    action = np.zeros((n, T), dtype=np.uint8)
    forward_rows = np.zeros(n, dtype=np.int64)
    reward = np.zeros(n, dtype=np.float64)
    t_mono = np.zeros(n, dtype=np.float64)

    msgs_cum = np.zeros(T, dtype=np.int64)
    leaves_cum = np.zeros(T, dtype=np.int64)
    gate = np.ones(T, dtype=np.int64)           # all-allow start (matches the server's reset)
    t = 1_000.0 + rng.random() * 10.0
    # server_rows_per_forward is a CONSTANT column (fixed across the trial): sentinel 0, or a FIXED non-zero
    # value chosen once (NOT per-decision — a per-decision value would violate the const-column contract,
    # which the codec's check=True append rightly rejects).
    srv_rows_const = 0.0 if srv_sentinel else float(int(rng.integers(1, 256)))
    for i in range(n):
        # monotone cumulative counters; the per-forward growth is small but NOT cleanly bounded by D/K.
        msgs_cum = msgs_cum + rng.integers(0, D + 1, size=T)
        leaves_cum = leaves_cum + rng.integers(0, 4, size=T)
        inf = rng.integers(0, D + 1, size=T)    # in [0, D]
        rdy = rng.integers(0, K + 1, size=T)    # in [0, K]
        rt = np.zeros(T, dtype=np.int64) if rtt_sentinel else rng.integers(0, 5000, size=T)
        # a variable served subset (at least one thread served, like a real group forward).
        k_served = int(rng.integers(1, T + 1))
        srv_ids = sorted(rng.choice(T, size=k_served, replace=False).tolist())
        # the gate flips rarely -> long stable runs (the RLE/zstd-friendly real shape).
        flips = rng.random(T) < gate_flip_p
        gate = np.where(flips, 1 - gate, gate)
        fr = int(rng.integers(1, 256))

        feats = {
            "n_threads": T, "d_ceiling": D, "server_rows_per_forward": srv_rows_const,
            "inflight": inf.tolist(), "ready": rdy.tolist(),
            "msgs": msgs_cum.tolist(), "leaves": leaves_cum.tolist(), "rtt_us": rt.tolist(),
        }
        t += float(rng.random() * 1e-3)         # monotone-increasing clock (microsecond-ish steps)
        obs = Observation(features=feats, served=srv_ids, forward_rows=fr, t_monotonic=t)
        rew = float(fr)
        buf.append(obs, gate.tolist(), rew)

        inflight[i] = inf
        ready[i] = rdy
        msgs[i] = msgs_cum
        leaves[i] = leaves_cum
        rtt_us[i] = rt
        for tid in srv_ids:
            served[i, tid] = 1
        action[i] = gate
        forward_rows[i] = fr
        reward[i] = rew
        t_mono[i] = t

    truth = _Truth(inflight, ready, msgs, leaves, rtt_us, served, action, forward_rows, reward, t_mono)
    return buf, truth


def _assert_exact(dec: tc.DecodedTrajectory, truth: _Truth, T: int, D: int, K: int) -> None:
    """Every column decodes byte/bit-exact to the appended ground truth (the round-trip contract)."""
    assert dec.n_decisions == truth.inflight.shape[0]
    assert dec.n_threads == T and dec.d_ceiling == D and dec.k_per_thread == K
    assert np.array_equal(dec.inflight, truth.inflight), "inflight column drifted"
    assert np.array_equal(dec.ready, truth.ready), "ready column drifted"
    assert np.array_equal(dec.msgs, truth.msgs), "msgs (delta) column drifted"
    assert np.array_equal(dec.leaves, truth.leaves), "leaves (delta) column drifted"
    assert np.array_equal(dec.rtt_us, truth.rtt_us), "rtt_us (RLE) column drifted"
    assert np.array_equal(dec.served, truth.served), "served mask drifted"
    assert np.array_equal(dec.action, truth.action), "action mask drifted"
    assert np.array_equal(dec.forward_rows, truth.forward_rows), "forward_rows drifted"
    # reward + t_monotonic are float64 — EXACT (bit-identical), not tolerance: a serialization contract.
    assert np.array_equal(dec.reward, truth.reward), "reward (float64-raw) drifted"
    assert dec.t_monotonic.dtype == np.float64
    assert np.array_equal(dec.t_monotonic, truth.t_monotonic), "t_monotonic (mono-bits) not bit-exact"
    # dtypes are part of the contract (the consumers index these).
    assert dec.inflight.dtype == np.int64 and dec.served.dtype == np.uint8


# ============================================================================================
# Round-trip exactness on the real shape (the point of the task).
# ============================================================================================
@pytest.mark.parametrize("T", [1, 2, 3, 4])
def test_roundtrip_exact_real_shape(T: int) -> None:
    """decode(encode(x)) is EXACT for every T in {1,2,3,4} at the real D=8/K=54 geometry, over a few
    thousand decisions (enough to span several geometric-growth doublings from a small initial cap)."""
    buf, truth = _make_trajectory(T=T, D=8, K=54, n=5000, seed=100 + T, initial_cap=512)
    dec = tc.decode(buf.encode())
    _assert_exact(dec, truth, T=T, D=8, K=54)


def test_roundtrip_empty_trajectory() -> None:
    """A zero-decision trajectory round-trips (the degenerate edge: encode an unused buffer -> decode to 0
    rows). The const columns + geometry still decode; every data column is shape (0, T) / (0,)."""
    ctx = TrialContext(n_threads=3, d_ceiling=8, k_per_thread=54, s_min=32, chunk_floor=False, seed=1)
    buf = tc.TrajectoryBuffer(ctx, initial_cap=16)
    dec = tc.decode(buf.encode())
    assert dec.n_decisions == 0
    assert dec.n_threads == 3 and dec.d_ceiling == 8 and dec.k_per_thread == 54 and dec.s_min == 32
    assert dec.inflight.shape == (0, 3) and dec.msgs.shape == (0, 3)
    assert dec.forward_rows.shape == (0,) and dec.reward.shape == (0,)


def test_roundtrip_single_decision() -> None:
    """The n=1 edge (delta base row with no successor; cumsum of a single delta)."""
    buf, truth = _make_trajectory(T=2, D=8, K=54, n=1, seed=7, initial_cap=8)
    dec = tc.decode(buf.encode())
    _assert_exact(dec, truth, T=2, D=8, K=54)


def test_roundtrip_nonsentinel_rtt_and_srv() -> None:
    """rtt_us and server_rows_per_forward are sentinel-0 TODAY, but the codec must round-trip them when
    they are wired (rtt_us a real per-thread RLE, server_rows_per_forward a non-zero constant). This pins
    that the RLE codec handles a multi-run column and the CONSTANT_F64 a non-zero value."""
    # server_rows_per_forward stays constant within a trial (the const-column contract), so srv_sentinel
    # False uses a FIXED non-zero value per trial; rtt_us gets real (multi-run) per-thread values.
    buf, truth = _make_trajectory(T=3, D=8, K=54, n=800, seed=42, rtt_sentinel=False, srv_sentinel=False,
                                  initial_cap=256)
    blob = buf.encode()
    dec = tc.decode(blob)
    _assert_exact(dec, truth, T=3, D=8, K=54)
    # the non-sentinel server_rows_per_forward is a FIXED non-zero constant; assert it round-tripped as one.
    assert dec.server_rows_per_forward != 0.0 and float(dec.server_rows_per_forward).is_integer()


def test_all_zero_columns_roundtrip() -> None:
    """A fully sentinel trajectory (rtt_us=0, gates never flip, a degenerate K bound) collapses to tiny
    runs but must still round-trip exactly — the all-zero BITPACK (1 bit) + single-run RLE edges."""
    rng = np.random.default_rng(0)
    ctx = TrialContext(n_threads=4, d_ceiling=8, k_per_thread=54, s_min=32, chunk_floor=False, seed=0)
    buf = tc.TrajectoryBuffer(ctx, initial_cap=128, check=True)
    n = 500
    t = 1000.0
    for i in range(n):
        t += 1e-4
        feats = {"n_threads": 4, "d_ceiling": 8, "server_rows_per_forward": 0.0,
                 "inflight": [0, 0, 0, 0], "ready": [0, 0, 0, 0], "msgs": [0, 0, 0, 0],
                 "leaves": [0, 0, 0, 0], "rtt_us": [0, 0, 0, 0]}
        obs = Observation(features=feats, served=[0, 1, 2, 3], forward_rows=0, t_monotonic=t)
        buf.append(obs, [1, 1, 1, 1], 0.0)
    dec = tc.decode(buf.encode())
    assert dec.n_decisions == n
    assert np.array_equal(dec.inflight, np.zeros((n, 4), dtype=np.int64))
    assert np.array_equal(dec.action, np.ones((n, 4), dtype=np.uint8))
    assert np.array_equal(dec.rtt_us, np.zeros((n, 4), dtype=np.int64))


# ============================================================================================
# The provable BITPACK bound (the maintainer's key ask): width = ceil(log2(bound+1)).
# ============================================================================================
@pytest.mark.parametrize("hi,expected_bits", [(0, 1), (1, 1), (2, 2), (7, 3), (8, 4), (54, 6), (63, 6),
                                              (64, 7), (255, 8)])
def test_bits_for_range(hi: int, expected_bits: int) -> None:
    """ceil(log2(hi+1)) — the minimal fixed width for [0, hi]. D=8 -> 4 bits (holds 0..8); K=54 -> 6 bits
    (holds 0..63 ⊇ 0..54). The width is DERIVED from the bound, never a separate literal (P1)."""
    assert tc._bits_for_range(hi) == expected_bits


def test_bitpack_at_provable_width_roundtrips_full_range() -> None:
    """inflight pinned to its bound D and ready to K (the extreme values) round-trip exactly through the
    BITPACK width — proving ceil(log2(bound+1)) bits actually hold the whole declared range."""
    D, K, T = 8, 54, 3
    ctx = TrialContext(n_threads=T, d_ceiling=D, k_per_thread=K, s_min=32, chunk_floor=False, seed=5)
    buf = tc.TrajectoryBuffer(ctx, initial_cap=64, check=True)
    t = 1000.0
    inf_vals = [0, D, D // 2]          # includes the upper bound D
    rdy_vals = [K, 0, K // 3]          # includes the upper bound K
    n = 64
    for i in range(n):
        t += 1e-4
        feats = {"n_threads": T, "d_ceiling": D, "server_rows_per_forward": 0.0,
                 "inflight": inf_vals, "ready": rdy_vals, "msgs": [i, i, i], "leaves": [i, i, i],
                 "rtt_us": [0, 0, 0]}
        obs = Observation(features=feats, served=[0, 1, 2], forward_rows=10, t_monotonic=t)
        buf.append(obs, [1, 0, 1], 10.0)
    dec = tc.decode(buf.encode())
    assert np.all(dec.inflight == np.array(inf_vals))
    assert np.all(dec.ready == np.array(rdy_vals))


def test_check_flag_catches_bound_violation() -> None:
    """A value OVER the declared bound (inflight > D) is a LOUD violation under check=True (ADR-0002), not
    a silent BITPACK truncation. The hot bench path runs check=False (no overhead); tests run check=True."""
    ctx = TrialContext(n_threads=2, d_ceiling=8, k_per_thread=54, s_min=32, chunk_floor=False, seed=0)
    buf = tc.TrajectoryBuffer(ctx, initial_cap=8, check=True)
    feats = {"n_threads": 2, "d_ceiling": 8, "server_rows_per_forward": 0.0,
             "inflight": [99, 0], "ready": [0, 0], "msgs": [0, 0], "leaves": [0, 0], "rtt_us": [0, 0]}
    obs = Observation(features=feats, served=[0, 1], forward_rows=1, t_monotonic=1.0)
    with pytest.raises(AssertionError, match="inflight exceeds D"):
        buf.append(obs, [1, 1], 1.0)


# ============================================================================================
# DECODE BOUNDARIES — every one fails LOUD (ADR-0002), never a silent misread.
# ============================================================================================
def _good_blob() -> bytes:
    buf, _ = _make_trajectory(T=3, D=8, K=54, n=200, seed=11, initial_cap=128)
    return buf.encode()


def test_decode_rejects_short_blob() -> None:
    with pytest.raises(tc.CodecError, match="too short for the header"):
        tc.decode(b"\x00\x01")


def test_decode_rejects_bad_magic() -> None:
    blob = bytearray(_good_blob())
    blob[0:8] = b"XXXXXXXX"
    with pytest.raises(tc.CodecError, match="bad trajectory magic"):
        tc.decode(bytes(blob))


def test_decode_rejects_wrong_format_version() -> None:
    blob = bytearray(_good_blob())
    blob[8] = tc.FORMAT_VERSION + 9    # the format-version byte sits right after the 8-byte magic
    with pytest.raises(tc.CodecError, match="format version"):
        tc.decode(bytes(blob))


def test_decode_rejects_wrong_column_count() -> None:
    blob = bytearray(_good_blob())
    blob[9] = len(tc.COLUMNS) + 1       # the n_columns byte
    with pytest.raises(tc.CodecError, match="declares .* columns"):
        tc.decode(bytes(blob))


def test_decode_rejects_corrupt_compressed_body() -> None:
    """A blob whose compressed body is garbage fails loud at zstd decompression, not as a misread."""
    blob = bytearray(_good_blob())
    # keep the valid sniff header, replace the compressed body with noise.
    head = bytes(blob[: tc._HEADER.size])
    with pytest.raises(tc.CodecError, match="zstd decompression"):
        tc.decode(head + b"not a valid zstd frame at all, definitely")


def test_decode_rejects_schema_hash_mismatch() -> None:
    """If the column SCHEMA differs from what wrote the blob (a re-declared codec / added column), the
    schema-hash check rejects it — a stale decoder never silently misreads a re-schema'd blob (ADR-0002).
    We simulate by decoding a blob written under a perturbed SCHEMA_HASH."""
    buf, _ = _make_trajectory(T=2, D=8, K=54, n=50, seed=3, initial_cap=64)
    # Encode normally, then rewrite the schema_hash field inside the (decompressed) body and recompress —
    # the surgical equivalent of "a different schema wrote this blob".
    import zstandard as zstd
    blob = buf.encode()
    body = bytearray(zstd.ZstdDecompressor().decompress(blob[tc._HEADER.size:]))
    # schema_hash is the last 8 bytes of the _GEOM block (HIIIQ Q -> the trailing Q).
    bad_hash = (tc.SCHEMA_HASH ^ 0xDEADBEEF) & 0xFFFFFFFFFFFFFFFF
    struct.pack_into("<Q", body, tc._GEOM.size - 8, bad_hash)
    recompressed = zstd.ZstdCompressor(level=1).compress(bytes(body))
    bad_blob = blob[: tc._HEADER.size] + recompressed
    with pytest.raises(tc.CodecError, match="schema hash"):
        tc.decode(bad_blob)


def test_decode_rejects_codec_id_drift_in_directory() -> None:
    """A directory entry whose codec id disagrees with the schema's codec for that column is codec drift —
    fail loud (ADR-0002), not decode-with-the-wrong-codec. We flip one directory entry's codec id."""
    import zstandard as zstd
    buf, _ = _make_trajectory(T=2, D=8, K=54, n=50, seed=4, initial_cap=64)
    blob = buf.encode()
    body = bytearray(zstd.ZstdDecompressor().decompress(blob[tc._HEADER.size:]))
    # the first directory entry sits right after the _GEOM block: (name_len u8, codec_id u8, payload_len u32).
    codec_id_off = tc._GEOM.size + 1     # name_len is byte 0 of the entry; codec_id is byte 1
    orig = body[codec_id_off]
    body[codec_id_off] = (orig + 1) % 8  # a different (but in-enum) codec id -> drift vs the schema
    recompressed = zstd.ZstdCompressor(level=1).compress(bytes(body))
    with pytest.raises(tc.CodecError, match="codec id"):
        tc.decode(blob[: tc._HEADER.size] + recompressed)


def test_schema_hash_is_stable_and_nonzero() -> None:
    """The schema hash is deterministic (a value, not a per-run nonce) and non-zero — it is the cross-batch
    contract's schema fingerprint, recomputed identically by any decoder build of this module."""
    assert tc.SCHEMA_HASH != 0
    assert tc._schema_hash() == tc.SCHEMA_HASH       # deterministic
    assert tc._schema_hash() == tc.SCHEMA_HASH       # idempotent


def test_column_codec_table_covers_every_column() -> None:
    """The introspection table names every declared column with its codec + bound (the dashboard/docs read
    this). One row per COLUMNS entry; the bounded columns name their TrialContext bound source."""
    table = tc.column_codec_table()
    assert [r["column"] for r in table] == [c.name for c in tc.COLUMNS]
    by_name = {r["column"]: r for r in table}
    assert by_name["inflight"]["codec"] == "BITPACK" and by_name["inflight"]["bound"] == "d_ceiling"
    assert by_name["ready"]["codec"] == "BITPACK" and by_name["ready"]["bound"] == "k_per_thread"
    assert by_name["msgs"]["codec"] == "DELTA_ZIGZAG_VARINT"
    assert by_name["t_monotonic"]["codec"] == "FLOAT64_MONO_BITS"
    assert by_name["rtt_us"]["codec"] == "RLE_I64"


# ============================================================================================
# PROPERTY-BASED round-trip (hypothesis) over the whole shape space — the strongest exactness net.
# ============================================================================================
if _HAS_HYPOTHESIS:

    @st.composite
    def _trajectories(draw: Any) -> tuple[tc.TrajectoryBuffer, _Truth, int, int, int]:
        T = draw(st.integers(min_value=1, max_value=4))
        D = draw(st.integers(min_value=1, max_value=16))
        K = draw(st.integers(min_value=1, max_value=64))
        n = draw(st.integers(min_value=0, max_value=300))
        seed = draw(st.integers(min_value=0, max_value=2**31 - 1))
        rtt_sentinel = draw(st.booleans())
        srv_sentinel = draw(st.booleans())
        gate_flip_p = draw(st.floats(min_value=0.0, max_value=0.5))
        buf, truth = _make_trajectory(T=T, D=D, K=K, n=n, seed=seed, gate_flip_p=gate_flip_p,
                                      rtt_sentinel=rtt_sentinel, srv_sentinel=srv_sentinel,
                                      initial_cap=draw(st.sampled_from([1, 8, 64, 512])), check=True)
        return buf, truth, T, D, K

    @settings(max_examples=150, deadline=None, suppress_health_check=[HealthCheck.too_slow,
                                                                      HealthCheck.data_too_large])
    @given(_trajectories())
    def test_roundtrip_property(case: tuple[tc.TrajectoryBuffer, _Truth, int, int, int]) -> None:
        """For ANY trajectory in the shape space (T,D,K,n, sentinel flags, gate-flip rate, initial cap),
        decode(encode(x)) reproduces every column EXACTLY. This is the codec's central invariant."""
        buf, truth, T, D, K = case
        dec = tc.decode(buf.encode())
        _assert_exact(dec, truth, T=T, D=D, K=K)

    @st.composite
    def _monotone_floats(draw: Any) -> np.ndarray:
        n = draw(st.integers(min_value=1, max_value=400))
        base = draw(st.floats(min_value=0.0, max_value=1e6, allow_nan=False, allow_infinity=False))
        steps = draw(st.lists(st.floats(min_value=0.0, max_value=1e3, allow_nan=False,
                                        allow_infinity=False), min_size=n, max_size=n))
        return np.cumsum(np.array([base] + steps[:-1], dtype=np.float64))

    @settings(max_examples=100, deadline=None)
    @given(_monotone_floats())
    def test_mono_float_codec_bit_exact(vals: np.ndarray) -> None:
        """The FLOAT64_MONO_BITS codec round-trips a monotone-increasing non-negative float64 sequence
        BIT-EXACTLY (the t_monotonic contract): reinterpret-as-u64 + integer delta is exactly invertible,
        so no quantization error — `==`, not a tolerance."""
        payload = tc._enc_float64_mono_bits(vals.astype(np.float64))
        back = tc._dec_float64_mono_bits(payload, len(vals))
        assert np.array_equal(back, vals), "mono-float64 codec is not bit-exact"
