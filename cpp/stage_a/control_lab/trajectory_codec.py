#!/usr/bin/env python3
"""
cpp/stage_a/control_lab/trajectory_codec.py — a BESPOKE COLUMNAR, PER-FEATURE-COMPRESSED serialization
for the issue-gate control lab's controller-trajectory stream: the (Observation, action, reward) tuples
the LabServer logs per server-forward (lab_server.py / adapter.py — the FROZEN contract). One 4-second
experiment emits ~1e5-1e6 decision tuples that must compress HARD (the blob egresses to postgres as a
compressed bytea; VM disk is scarce), so this trades a hot allocation-free append for a deferred,
between-trials columnar encode + zstd.

WHY COLUMNAR (struct-of-arrays, NOT array-of-structs). Each feature is its own contiguous column across
the decision sequence. Column HOMOGENEITY is what makes delta-encoding and zstd effective: a column of
monotone per-thread `msgs` deltas is a run of small integers; a column of mostly-stable gate bits is a
run of identical bytes; the high bytes of a column of `t_monotonic` float64s are near-constant. An
array-of-structs interleaves these and destroys the runs.

THE THREE SEAMS (ADR-0012 P2/P8/P9):
  * ColumnSpec — a per-column codec DECLARATION (name + Codec id + bound source). Adding a feature is
    declaring its column + codec id in COLUMNS, nothing else (the seam the supervised-batch and dashboard
    consumers read by name). The codec ids are a CLOSED vocabulary (the Codec IntEnum); a header naming an
    unknown id fails loud at decode (ADR-0002).
  * TrajectoryBuffer — the columnar struct-of-arrays accumulator with a CHEAP, allocation-free steady-path
    `append(obs, action, reward)` (preallocated numpy column arrays grown geometrically; NO compression at
    append time). `encode() -> bytes` runs BETWEEN trials.
  * encode/decode — the pure (bytes in / bytes out) functional core. decode is a BOUNDARY: a bad magic,
    an unsupported format version, an unknown codec id, or a schema-hash mismatch is a loud CodecError,
    never a silently misread trajectory (ADR-0002 fail-loud, the strongest decode surface).

THE PROVABLE DELTA-BOUND (the maintainer's key ask: "delta-encode where it makes sense, bound the
accumulator where a bound on the delta is provable"). Two distinct facts, drawn apart:
  * ABSOLUTE-VALUE bounds that ARE provable from TrialContext, used to pick a MINIMAL FIXED bit width:
      - inflight in [0, D]  (D = TrialContext.d_ceiling — the runner's `inflight < D` ceiling,
        issue_controller.hpp) -> BITPACK at ceil(log2(D+1)) bits.
      - ready    in [0, K]  (K = TrialContext.k_per_thread, the capacity normalizer)
        -> BITPACK at ceil(log2(K+1)) bits.
      - served / action are T-bit-per-decision masks -> BITPACK at exactly T bits.
  * The cumulative counters msgs/leaves and the clock t_monotonic are monotone-WHEN-SERVED, so consecutive
    DIFFERENCES are small -> DELTA. But the per-forward delta of a cumulative counter is NOT cleanly
    bounded by D/K from TrialContext alone: an arbitrary number of forwards can elapse between a thread's
    appearances, a thread can issue-and-drain many messages across that gap, AND the logged Observation
    re-zeros a thread absent from a forward (lab_server._run_controller builds length-T vectors with absent
    tids = 0), so a per-thread column can step 5 -> 0 -> 7 and the delta is signed and unbounded above by
    any TrialContext field. Where the bound is NOT clean we therefore FALL BACK to ZIGZAG + VARINT (LEB128)
    on the delta — small in the common all-served monotone case, correct in the worst case. This is the
    honest split: a provable bound buys a fixed minimal width; an unprovable one buys a self-sizing varint.

THE BLOB IS A CROSS-BATCH CONTRACT (the deferred supervised batch + the dashboard decode it). The format is
SELF-DESCRIBING: a fixed header (magic, format version, schema hash, T/D/K/s_min, n_decisions) + a column
directory (per-column name, codec id, byte length) precede the zstd-compressed columnar payload, so decode
is unambiguous and forward-compatible. This module is PURE serialization (bytes in / bytes out); it does
NOT touch the DB — the harness egress (built separately) does the psycopg3 insert of these bytes.

Public Domain (The Unlicense).
"""
from __future__ import annotations

import enum
import hashlib
import struct
from dataclasses import dataclass
from typing import Any, Final, Mapping, Sequence

import numpy as np
import numpy.typing as npt
import zstandard as zstd

# ============================================================================================
# Format constants (the cross-batch contract). FORMAT_VERSION is bumped on ANY layout change so an old
# decoder paired with a new blob fails loudly at the version byte (ADR-0002), never misreads a column.
# ============================================================================================
MAGIC: Final[bytes] = b"CHTRAJ01"          # 8-byte container magic (control-lab TRAJectory, fmt family 01)
FORMAT_VERSION: Final[int] = 1
# zstd level 1 — a MEASURED choice (bench_trajectory_codec.py / ADR-0009), not a default reflex. The
# per-column codecs (delta/bitpack/RLE) already strip the structural redundancy, leaving a near-
# incompressible varint+float residue, so zstd here only mops up byte-level repetition (mask runs, varint
# patterns). On the 1e6-row synthetic worst case level 1 gives 10.7x overall in ~55 ms, while level 19
# spends ~10 s for 10.8x — a 0.7% ratio gain for ~190x the time. Level 1 meets the <100 ms inter-trial
# target with margin AND ties the best ratio. (A trained cross-trial dictionary or threads=N would buy a
# higher level inside the budget on more-compressible real data — NOTED, not built.)
ZSTD_LEVEL: Final[int] = 1

_HEADER = struct.Struct("<8sBB")             # magic, format_version, n_columns  (the fixed sniff prefix)
# Per-trial geometry block (the bounds the codec exploits + n_decisions), all little-endian.
#   n_threads(T) u16 | d_ceiling(D) u32 | k_per_thread(K) u32 | s_min u32 | n_decisions u64 | schema_hash u64
_GEOM = struct.Struct("<HIIIQQ")
# Per-column directory entry: name_len u8 | codec_id u8 | payload_len u32  (name bytes + payload follow).
_COLENT = struct.Struct("<BBI")


class CodecError(ValueError):
    """A decode-boundary failure: bad magic, unsupported version, unknown codec id, schema-hash mismatch,
    or a length disagreement (ADR-0002 — a loud, typed failure, never a silently misread trajectory)."""


class Codec(enum.IntEnum):
    """The CLOSED per-column codec vocabulary (ADR-0008 — a closed set; a header naming an id outside this
    enum fails loud at decode). Each id pins ONE column encoding; the directory carries the id per column."""
    CONSTANT = 0          # one value repeated across the trial (int) — stored once, replayed n_decisions times
    CONSTANT_F64 = 1      # CONSTANT for a float64 column (server_rows_per_forward, sentinel/fixed)
    BITPACK = 2           # n_decisions values, each in [0, 2^bits-1], packed at `bits` bits (bound from ctx)
    DELTA_ZIGZAG_VARINT = 3   # per-element consecutive delta -> zigzag -> LEB128 varint (cumulative counters)
    RLE_I64 = 4           # run-length of (value:i64, count:varint) pairs (long sentinel/constant runs)
    MASK_BITS = 5         # n_decisions T-bit masks (served / action), packed T bits per decision
    FLOAT64_RAW = 6       # n_decisions raw little-endian float64 (exact; non-monotone floats — reward)
    FLOAT64_MONO_BITS = 7  # monotone float64: reinterpret f64->u64, delta, zigzag, varint (exact, t_monotonic)


# ============================================================================================
# Column declaration — the SEAM. A new feature is one row here (name + codec + how its column is filled +
# the bound source for a BITPACK width). `kind` routes how `append` fills the column; `bound` names the
# TrialContext field a BITPACK column's width is DERIVED from (P1 — the width is never a separate literal).
# ============================================================================================
@dataclass(frozen=True)
class ColumnSpec:
    """One trajectory column's declaration: its on-blob name, its codec, how `append` fills it, and (for a
    bounded column) which TrialContext bound sizes it. Frozen — the schema is a value, hashed into the
    header so a decoder verifies it is reading the same schema it was written with (ADR-0002)."""
    name: str
    codec: Codec
    kind: str           # 'scalar_const_int' | 'scalar_const_f64' | 'per_thread' | 'served_mask' |
    #                      'action_mask' | 'scalar_int' | 'scalar_f64'
    bound: str = ""     # for BITPACK: 'd_ceiling' (range [0,D]) | 'k_per_thread' ([0,K]) | 'n_threads' (T bits)


# The schema. ORDER is the on-blob column order; consumers read BY NAME from the directory, so a reorder is
# safe, but the schema HASH (over this tuple) pins the exact set+codecs a decoder must agree with.
COLUMNS: Final[tuple[ColumnSpec, ...]] = (
    # --- once-per-trial scalars (CONSTANT): fixed or sentinel across the whole trial ---
    ColumnSpec("n_threads", Codec.CONSTANT, "scalar_const_int"),
    ColumnSpec("d_ceiling", Codec.CONSTANT, "scalar_const_int"),
    ColumnSpec("server_rows_per_forward", Codec.CONSTANT_F64, "scalar_const_f64"),
    # --- bounded-range per-thread snapshots (BITPACK at a width PROVABLE from TrialContext) ---
    ColumnSpec("inflight", Codec.BITPACK, "per_thread", bound="d_ceiling"),   # in [0, D]
    ColumnSpec("ready", Codec.BITPACK, "per_thread", bound="k_per_thread"),    # in [0, K]
    # --- monotone-when-served cumulative counters (DELTA + zigzag + varint; delta not cleanly bounded) ---
    ColumnSpec("msgs", Codec.DELTA_ZIGZAG_VARINT, "per_thread"),
    ColumnSpec("leaves", Codec.DELTA_ZIGZAG_VARINT, "per_thread"),
    # --- sentinel-0-today per-thread counter (RLE: one long constant run while unwired) ---
    ColumnSpec("rtt_us", Codec.RLE_I64, "per_thread"),
    # --- the variable served subset + the gate vector (T-bit masks; gate changes rarely -> zstd-friendly) ---
    ColumnSpec("served", Codec.MASK_BITS, "served_mask", bound="n_threads"),
    ColumnSpec("action", Codec.MASK_BITS, "action_mask", bound="n_threads"),
    # --- per-decision scalars: forward_rows (small, delta) + reward (float, exact) + the clock (monotone) ---
    ColumnSpec("forward_rows", Codec.DELTA_ZIGZAG_VARINT, "scalar_int"),
    ColumnSpec("reward", Codec.FLOAT64_RAW, "scalar_f64"),
    ColumnSpec("t_monotonic", Codec.FLOAT64_MONO_BITS, "scalar_f64"),
)


def _schema_hash() -> int:
    """A 64-bit hash of the column schema (name, codec id, kind, bound — in declared order). Written into
    the header; a decoder recomputes it and rejects a blob whose schema differs (ADR-0002 — a stale
    decoder against a re-declared schema fails loud, never silently misreads)."""
    h = hashlib.sha256()
    for c in COLUMNS:
        h.update(f"{c.name}\0{int(c.codec)}\0{c.kind}\0{c.bound}\0".encode("utf-8"))
    return int.from_bytes(h.digest()[:8], "little")


SCHEMA_HASH: Final[int] = _schema_hash()


def _bits_for_range(hi_inclusive: int) -> int:
    """Minimal bit width to hold every value in [0, hi_inclusive]. ceil(log2(hi+1)); 0 collapses to 1 bit
    (a degenerate all-zero column still costs 1 bit/value, which zstd then crushes)."""
    if hi_inclusive < 0:
        raise CodecError(f"bitpack range upper bound is negative: {hi_inclusive}")
    if hi_inclusive == 0:
        return 1
    return int(hi_inclusive).bit_length()


# ============================================================================================
# The columnar buffer — cheap hot append, deferred encode.
# ============================================================================================
class TrajectoryBuffer:
    """A struct-of-arrays accumulator for the (Observation, action, reward) trajectory, appended to in the
    server's per-forward HOT loop (~1e5-1e6 appends per 4 s). The steady-path `append` is allocation-free:
    preallocated numpy column arrays, grown GEOMETRICALLY (amortized O(1), a doubling realloc only on the
    rare growth tick). NO compression at append time — `encode()` runs between trials, off the hot path.

    Columns held (all numpy, dtype chosen so a per-thread cumulative counter never overflows):
      * scalars per decision: forward_rows (i64), reward (f64), t_monotonic (f64)
      * per-thread (T columns each, shape (cap, T)): inflight/ready (i64), msgs/leaves/rtt_us (i64)
      * served mask + action: (cap, T) uint8 0/1
    The const scalars (n_threads, d_ceiling, server_rows_per_forward) are captured ONCE at reset from
    TrialContext / the first observation (CONSTANT columns), verified stable on append under a debug flag.
    """

    __slots__ = (
        "_T", "_D", "_K", "_s_min", "_n", "_cap",
        "_inflight", "_ready", "_msgs", "_leaves", "_rtt_us",
        "_served", "_action", "_forward_rows", "_reward", "_t_mono",
        "_srv_rows_const", "_srv_rows_seen", "_check",
    )

    def __init__(self, ctx: Any, initial_cap: int = 1 << 16, check: bool = False) -> None:
        """ctx is a TrialContext (adapter.TrialContext) — its n_threads/d_ceiling/k_per_thread/s_min are
        the geometry the codec exploits and writes into the header. `check=True` enables steady-path
        invariant asserts (bounds, length-T) for tests; the hot bench path leaves it False (zero overhead).
        `initial_cap` preallocates the column arrays; growth is geometric from there."""
        self._T = int(ctx.n_threads)
        self._D = int(ctx.d_ceiling)
        self._K = int(ctx.k_per_thread)
        self._s_min = int(ctx.s_min)
        if self._T <= 0:
            raise CodecError(f"TrajectoryBuffer: n_threads must be >= 1, got {self._T}")
        self._n = 0
        cap = max(1, int(initial_cap))
        self._cap = cap
        T = self._T
        # i64 throughout for the integer columns: a cumulative counter must not wrap over a long run, and a
        # uniform dtype keeps the append path branch-light. The masks are uint8 (one byte 0/1 per thread).
        self._inflight: npt.NDArray[np.int64] = np.zeros((cap, T), dtype=np.int64)
        self._ready: npt.NDArray[np.int64] = np.zeros((cap, T), dtype=np.int64)
        self._msgs: npt.NDArray[np.int64] = np.zeros((cap, T), dtype=np.int64)
        self._leaves: npt.NDArray[np.int64] = np.zeros((cap, T), dtype=np.int64)
        self._rtt_us: npt.NDArray[np.int64] = np.zeros((cap, T), dtype=np.int64)
        self._served: npt.NDArray[np.uint8] = np.zeros((cap, T), dtype=np.uint8)
        self._action: npt.NDArray[np.uint8] = np.zeros((cap, T), dtype=np.uint8)
        self._forward_rows: npt.NDArray[np.int64] = np.zeros(cap, dtype=np.int64)
        self._reward: npt.NDArray[np.float64] = np.zeros(cap, dtype=np.float64)
        self._t_mono: npt.NDArray[np.float64] = np.zeros(cap, dtype=np.float64)
        # server_rows_per_forward is a CONSTANT column today (sentinel/fixed). Captured on first append.
        self._srv_rows_const: float = 0.0
        self._srv_rows_seen: bool = False
        self._check = bool(check)

    def __len__(self) -> int:
        return self._n

    @property
    def n_decisions(self) -> int:
        return self._n

    def _grow(self) -> None:
        """Geometric (doubling) growth of every column array — the rare realloc on the append path. Amortizes
        the append to O(1); a 4 s run that overflows the initial cap pays log2(N/cap) doublings total."""
        new_cap = self._cap * 2

        def _g2(a: npt.NDArray[Any]) -> npt.NDArray[Any]:
            b = np.zeros((new_cap, self._T), dtype=a.dtype)
            b[: self._cap] = a
            return b

        def _g1(a: npt.NDArray[Any]) -> npt.NDArray[Any]:
            b = np.zeros(new_cap, dtype=a.dtype)
            b[: self._cap] = a
            return b

        self._inflight = _g2(self._inflight)
        self._ready = _g2(self._ready)
        self._msgs = _g2(self._msgs)
        self._leaves = _g2(self._leaves)
        self._rtt_us = _g2(self._rtt_us)
        self._served = _g2(self._served)
        self._action = _g2(self._action)
        self._forward_rows = _g1(self._forward_rows)
        self._reward = _g1(self._reward)
        self._t_mono = _g1(self._t_mono)
        self._cap = new_cap

    def append(self, obs: Any, action: Sequence[int], reward: float) -> None:
        """Append one decision tuple (Observation, action, reward) — the HOT per-forward call. Steady-path
        allocation-free: write into the preallocated row `self._n`, then increment. The ONLY allocation is
        the geometric `_grow` on a capacity tick (amortized away). `obs` is an adapter.Observation; `action`
        is the length-T gate list; `reward` the scalar.

        The per-thread feature vectors (obs.features['inflight'|'ready'|'msgs'|'leaves'|'rtt_us']) are
        length-T lists indexed by tid (lab_server builds them so); we copy each into the row. `served` is
        the variable subset of tids -> a one-hot row over T. The scalars ride straight in."""
        i = self._n
        if i >= self._cap:
            self._grow()
        f = obs.features
        # Per-thread columns: assign the length-T list straight into the row (numpy vectorizes the copy).
        # Direct dict access (NOT .get) — the frozen Observation contract guarantees these keys; a missing
        # one is a contract break that SHOULD raise loudly (ADR-0002), not be papered with a default.
        self._inflight[i, :] = f["inflight"]
        self._ready[i, :] = f["ready"]
        self._msgs[i, :] = f["msgs"]
        self._leaves[i, :] = f["leaves"]
        self._rtt_us[i, :] = f["rtt_us"]
        # action -> the gate row (length T, {0,1}); numpy copies the list.
        self._action[i, :] = action
        # served subset -> one-hot over T. Row i is ALWAYS freshly zeroed when reached: a row is written
        # exactly once (append only ever advances _n, never revisits), and both the initial alloc and
        # _grow zero the as-yet-unwritten region — so NO re-zero is needed here (just set the served bits).
        srow = self._served[i]
        for tid in obs.served:
            srow[tid] = 1
        # per-decision scalars
        self._forward_rows[i] = obs.forward_rows
        self._reward[i] = reward
        self._t_mono[i] = obs.t_monotonic
        # server_rows_per_forward: a CONSTANT column — captured ONCE (the steady path skips this work).
        if not self._srv_rows_seen:
            self._srv_rows_const = float(f["server_rows_per_forward"])
            self._srv_rows_seen = True
        if self._check:
            T = self._T
            assert len(action) == T, f"action length {len(action)} != T={T}"
            assert int(np.max(self._inflight[i])) <= self._D, "inflight exceeds D (BITPACK bound violated)"
            assert int(np.max(self._ready[i])) <= self._K, "ready exceeds K (BITPACK bound violated)"
            assert set(int(v) for v in action) <= {0, 1}, "action not binary"
            assert float(f["server_rows_per_forward"]) == self._srv_rows_const, \
                "server_rows_per_forward is not constant across the trial"
        self._n += 1

    # ---- the column views (n_decisions rows) the encoder consumes ----
    def _col_views(self) -> dict[str, npt.NDArray[Any]]:
        n = self._n
        return {
            "inflight": self._inflight[:n],
            "ready": self._ready[:n],
            "msgs": self._msgs[:n],
            "leaves": self._leaves[:n],
            "rtt_us": self._rtt_us[:n],
            "served": self._served[:n],
            "action": self._action[:n],
            "forward_rows": self._forward_rows[:n],
            "reward": self._reward[:n],
            "t_monotonic": self._t_mono[:n],
        }

    def encode(self) -> bytes:
        """Columnar-encode every column with its declared codec, concatenate into the self-describing
        container, and zstd-compress. Runs BETWEEN trials (off the hot path). Returns the bytea blob."""
        return _encode_columns(
            n=self._n, T=self._T, D=self._D, K=self._K, s_min=self._s_min,
            srv_rows_const=self._srv_rows_const, cols=self._col_views(),
        )


# ============================================================================================
# Per-column ENCODE/DECODE primitives (numpy-vectorized where it pays; the varint inner loop is the one
# explicit loop, kept tight). Each returns the column's payload bytes; the directory records the length.
# ============================================================================================
def _zigzag_encode(a: npt.NDArray[np.int64]) -> npt.NDArray[np.uint64]:
    """Map signed i64 -> unsigned u64 so small-magnitude (incl. negative) deltas get small varints:
    (n << 1) ^ (n >> 63). Vectorized."""
    return (a.astype(np.uint64) << np.uint64(1)) ^ (a >> np.int64(63)).astype(np.uint64)


def _zigzag_decode(u: npt.NDArray[np.uint64]) -> npt.NDArray[np.int64]:
    """Inverse of _zigzag_encode: (u >> 1) ^ -(u & 1)."""
    return ((u >> np.uint64(1)) ^ (-(u & np.uint64(1))).astype(np.uint64)).astype(np.int64)


_VARINT_SHIFTS: Final[npt.NDArray[np.uint64]] = np.arange(0, 70, 7, dtype=np.uint64)   # 10 groups of 7 bits


def _varint_max_len(maxval: int) -> int:
    """The LEB128 byte length of the LARGEST value in a u64 array — ceil(bit_length/7), min 1. Sizing the
    encode matrix to THIS (not the fixed 10) is the perf lever: the common monotone-delta column has tiny
    values, so L==1 and the matrix is (n, 1), not (n, 10) — a 10x smaller intermediate (ADR-0009)."""
    if maxval <= 0:
        return 1
    return (int(maxval).bit_length() + 6) // 7


def _varint_encode_u64(vals: npt.NDArray[np.uint64]) -> bytes:
    """LEB128 (unsigned, little-endian base-128) encode a u64 array — VECTORIZED (numpy), no Python
    per-element loop (the encode hot path: a 1e6-row trajectory has ~12M values across the delta/mono
    columns, so a Python loop here was the dominant encode cost — ADR-0009). 7 payload bits per byte, MSB
    = continuation; small values (the common monotone-delta case) cost 1 byte.

    Method: build only an (n, L) group matrix where L = max varint length over the array (NOT the fixed
    10 — for a small-delta column L==1, so the intermediate is 10x smaller and the encode is fast). Split
    each value into its L 7-bit groups, compute each value's byte LENGTH (highest nonzero group + 1, >=1),
    set the continuation bit on every group before the last, drop the unused trailing groups by a flat
    boolean mask, and emit the kept bytes row-major (little-endian). The kept order is exactly the scalar
    loop's emission order — verified byte-identical in the tests."""
    n = vals.shape[0]
    if n == 0:
        return b""
    L = _varint_max_len(int(vals.max()))                       # actual max length, 1..10
    shifts = _VARINT_SHIFTS[:L]
    groups = ((vals[:, None] >> shifts[None, :]) & np.uint64(0x7F)).astype(np.uint8)   # (n, L)
    grp_idx = np.arange(L, dtype=np.int64)[None, :]
    if L == 1:
        return groups.reshape(-1).tobytes()                    # every value is one byte; no continuation
    # byte length per value = 1 + index of the highest nonzero group (0 for value 0 -> length 1).
    highest = (np.arange(L, dtype=np.int64)[None, :] * (groups != 0)).max(axis=1)       # (n,)
    lengths = highest + 1
    groups = groups | ((grp_idx < (lengths[:, None] - 1)).astype(np.uint8) << np.uint8(7))  # continuation
    keep = grp_idx < lengths[:, None]
    kept: npt.NDArray[np.uint8] = groups[keep]
    return kept.tobytes()


def _varint_decode_u64(buf: bytes, count: int) -> tuple[npt.NDArray[np.uint64], int]:
    """Decode `count` LEB128 u64s from buf; return (array, n_bytes_consumed). VECTORIZED, no Python
    per-value loop: a byte terminates a value iff its MSB is clear, so a shifted cumulative count of
    terminator bytes assigns each byte to its value id; the within-value byte position (the 7-bit-group
    index) is a run-position that resets at each value boundary; the payload bits are then scattered into
    place by a single np.add.at. Loud on truncation / a value exceeding 64 bits (ADR-0002 — a blob that
    runs out mid-varint is corrupt, never a silent short read)."""
    if count == 0:
        return np.empty(0, dtype=np.uint64), 0
    raw = np.frombuffer(buf, dtype=np.uint8)
    nbytes = raw.shape[0]
    is_last = (raw & np.uint8(0x80)) == 0              # the terminating byte of each value
    if int(is_last.sum()) < count:
        raise CodecError("varint stream truncated (corrupt trajectory blob)")
    # value id of each byte = number of values COMPLETED strictly before it (so a value's own terminator
    # still carries that value's id).
    value_id = np.zeros(nbytes, dtype=np.int64)
    if nbytes > 1:
        value_id[1:] = np.cumsum(is_last[:-1].astype(np.int64))
    # bytes belonging to the first `count` values: up to (and including) value count-1's terminator.
    last_positions = np.nonzero(is_last)[0]
    end = int(last_positions[count - 1]) + 1
    value_id = value_id[:end]
    payload = (raw[:end] & np.uint8(0x7F)).astype(np.uint64)
    # within-value byte position: a run index that resets to 0 wherever value_id increments. Computed as
    # (flat index) - (flat index of this value's first byte); the first-byte index is broadcast by value id.
    flat = np.arange(end, dtype=np.int64)
    boundary = np.empty(end, dtype=bool)
    boundary[0] = True
    boundary[1:] = value_id[1:] != value_id[:-1]
    first_idx = np.maximum.accumulate(np.where(boundary, flat, 0))   # this value's first-byte flat index
    pos_in_val = flat - first_idx
    if int(pos_in_val.max(initial=0)) >= 10:
        raise CodecError("varint exceeds 64 bits (corrupt trajectory blob)")
    out = np.zeros(count, dtype=np.uint64)
    np.add.at(out, value_id, payload << (np.uint64(7) * pos_in_val.astype(np.uint64)))
    return out, end


def _enc_delta_zigzag_varint(col: npt.NDArray[np.int64]) -> bytes:
    """DELTA (consecutive per-element difference down the column) -> zigzag -> varint. For a per-thread
    column (2-D, (n, T)) the delta is taken DOWN each thread sub-column independently (axis 0), then the
    sub-columns are emitted thread-major so each thread's monotone run stays contiguous. For a 1-D scalar
    column the delta is along the single axis. The first row is stored as the absolute base (delta vs 0),
    so decode reconstructs by cumulative sum."""
    if col.ndim == 1:
        flat = col
        deltas = np.empty_like(flat)
        if flat.shape[0] > 0:
            deltas[0] = flat[0]
            deltas[1:] = np.diff(flat)
        return _varint_encode_u64(_zigzag_encode(deltas))
    # 2-D per-thread: delta down axis 0, emit thread-major (each column's run contiguous).
    n, _T = col.shape
    if n == 0:
        return b""
    deltas = np.empty_like(col)
    deltas[0, :] = col[0, :]
    deltas[1:, :] = np.diff(col, axis=0)
    flat = deltas.T.reshape(-1)   # thread-major: column t's n deltas, then column t+1's
    return _varint_encode_u64(_zigzag_encode(flat))


def _dec_delta_zigzag_varint(buf: bytes, n: int, T: int) -> npt.NDArray[np.int64]:
    """Inverse of _enc_delta_zigzag_varint. T==0 marks a 1-D scalar column (n values); T>0 a per-thread
    (n, T) column laid out thread-major."""
    count = n if T == 0 else n * T
    u, consumed = _varint_decode_u64(buf, count)
    if consumed != len(buf):
        raise CodecError(f"delta column has {len(buf) - consumed} trailing bytes (corrupt blob)")
    deltas = _zigzag_decode(u)
    if T == 0:
        return np.cumsum(deltas, dtype=np.int64)
    per_thread = deltas.reshape(T, n)   # un-flatten thread-major
    return np.cumsum(per_thread, axis=1, dtype=np.int64).T.copy()


def _enc_bitpack(col: npt.NDArray[np.int64], bits: int) -> bytes:
    """Pack a 2-D (n, T) or 1-D column of non-negative ints, each < 2^bits, at exactly `bits` bits/value,
    MSB-first within each value. The value count is recoverable from the header (n, T) + the codec, so no
    per-column count is stored.

    Fast path (bits <= 8, the only case our bounds reach — D=8 -> 4 bits, K=54 -> 6 bits, values < 256):
    unpackbits the uint8 values to their 8 MSB-first bit planes, slice the low `bits`, packbits. This is
    ~6x faster than a uint64 broadcast bit-matrix (ADR-0009 — measured 24 ms vs 148 ms on 3M values), and
    byte-identical. The general (bits > 8) path keeps the broadcast for a future wider bound."""
    flat = np.ascontiguousarray(col).reshape(-1)
    nvals = flat.shape[0]
    if nvals == 0:
        return b""
    if bits <= 8:
        as_u8 = flat.astype(np.uint8)                                   # values < 256 (asserted by bound)
        planes = np.unpackbits(as_u8).reshape(-1, 8)[:, 8 - bits:]      # low `bits` bits, MSB-first
        return np.packbits(planes.reshape(-1)).tobytes()
    flat64 = flat.astype(np.uint64)
    shifts = np.arange(bits - 1, -1, -1, dtype=np.uint64)               # [bits-1, ..., 1, 0]
    bitsmat = ((flat64[:, None] >> shifts[None, :]) & np.uint64(1)).astype(np.uint8)
    return np.packbits(bitsmat.reshape(-1)).tobytes()


def _dec_bitpack(buf: bytes, nvals: int, bits: int, shape: tuple[int, ...]) -> npt.NDArray[np.int64]:
    """Inverse of _enc_bitpack: unpack `nvals` values of `bits` bits each, reshape to `shape`. Fast path
    (bits <= 8): re-pad each value's `bits` planes to a full 8-bit byte (MSB-aligned) and packbits to
    recover the uint8 value directly — the inverse of the encode fast path, avoiding the (nvals, bits)
    uint64 weight-matrix multiply (ADR-0009 — the decode-side twin of the encode bitpack win)."""
    if nvals == 0:
        return np.zeros(shape, dtype=np.int64)
    total_bits = nvals * bits
    allbits = np.unpackbits(np.frombuffer(buf, dtype=np.uint8), count=total_bits).reshape(nvals, bits)
    if bits <= 8:
        padded = np.zeros((nvals, 8), dtype=np.uint8)
        padded[:, 8 - bits:] = allbits                         # MSB-align the `bits` planes in a byte
        vals = np.packbits(padded.reshape(-1)).astype(np.int64)
        return vals.reshape(shape)
    bitsmat = allbits.astype(np.uint64)
    weights = np.uint64(1) << np.arange(bits - 1, -1, -1, dtype=np.uint64)
    out: npt.NDArray[np.int64] = (bitsmat * weights[None, :]).sum(axis=1).astype(np.int64).reshape(shape)
    return out


def _enc_rle_i64(col: npt.NDArray[np.int64]) -> bytes:
    """Run-length encode a (possibly 2-D, flattened row-major) i64 column as (value, count) runs — for a
    long constant/sentinel column (rtt_us is sentinel-0 today) this is a handful of bytes. n_runs is
    varint-prefixed; the values are a zigzag-varint block, the counts a varint block."""
    flat = np.ascontiguousarray(col).reshape(-1)
    n = flat.shape[0]
    out = bytearray()
    if n == 0:
        out += _varint_encode_u64(np.zeros(1, dtype=np.uint64))   # 0 runs
        return bytes(out)
    change = np.nonzero(np.diff(flat))[0] + 1   # run boundaries: where the value changes
    starts = np.concatenate(([0], change))
    ends = np.concatenate((change, [n]))
    values = flat[starts].astype(np.int64)
    counts = (ends - starts).astype(np.uint64)
    out += _varint_encode_u64(np.array([len(values)], dtype=np.uint64))
    out += _varint_encode_u64(_zigzag_encode(values))
    out += _varint_encode_u64(counts)
    return bytes(out)


def _dec_rle_i64(buf: bytes, nvals_expected: int, shape: tuple[int, ...]) -> npt.NDArray[np.int64]:
    """Inverse of _enc_rle_i64. Reconstructs the flat column then reshapes; verifies the run counts sum to
    the expected element count (ADR-0002 — a run table that doesn't cover the column is corrupt)."""
    nruns_arr, pos = _varint_decode_u64(buf, 1)
    nruns = int(nruns_arr[0])
    if nruns == 0:
        if nvals_expected != 0:
            raise CodecError(f"RLE column declares 0 runs but {nvals_expected} values expected (corrupt)")
        return np.zeros(shape, dtype=np.int64)
    vz, c2 = _varint_decode_u64(buf[pos:], nruns)
    pos += c2
    counts, c3 = _varint_decode_u64(buf[pos:], nruns)
    pos += c3
    if pos != len(buf):
        raise CodecError(f"RLE column has {len(buf) - pos} trailing bytes (corrupt blob)")
    values = _zigzag_decode(vz)
    total = int(counts.sum())
    if total != nvals_expected:
        raise CodecError(f"RLE runs cover {total} values, expected {nvals_expected} (corrupt blob)")
    flat = np.repeat(values, counts.astype(np.int64))
    return flat.reshape(shape)


def _enc_mask_bits(col: npt.NDArray[np.uint8]) -> bytes:
    """Pack a (n, T) 0/1 mask at exactly T bits per decision, row-major MSB-first (numpy.packbits over the
    flattened rows). served + action use this; a mostly-stable gate row -> long identical-byte runs zstd
    then crushes. n,T are in the header, so no count is stored."""
    if col.shape[0] == 0:
        return b""
    return np.packbits(np.ascontiguousarray(col).reshape(-1)).tobytes()


def _dec_mask_bits(buf: bytes, n: int, T: int) -> npt.NDArray[np.uint8]:
    """Inverse of _enc_mask_bits: unpack n*T bits, reshape to (n, T)."""
    if n == 0:
        return np.zeros((0, T), dtype=np.uint8)
    bits = np.unpackbits(np.frombuffer(buf, dtype=np.uint8), count=n * T)
    return bits.reshape(n, T).astype(np.uint8)


def _enc_float64_raw(col: npt.NDArray[np.float64]) -> bytes:
    """Store raw little-endian float64 (EXACT for any value incl. NaN/inf). The column's high bytes are
    near-constant within a trial (reward ~ forward_rows magnitude), so zstd recovers the redundancy. Used
    for the non-monotone float column (reward)."""
    return np.ascontiguousarray(col, dtype="<f8").tobytes()


def _dec_float64_raw(buf: bytes, n: int) -> npt.NDArray[np.float64]:
    if len(buf) != n * 8:
        raise CodecError(f"float64-raw column is {len(buf)} bytes, expected {n * 8} (corrupt blob)")
    return np.frombuffer(buf, dtype="<f8").astype(np.float64).copy()


def _enc_float64_mono_bits(col: npt.NDArray[np.float64]) -> bytes:
    """Monotone-float64 codec (EXACT). Reinterpret each f64 as its u64 IEEE-754 bit pattern: for a
    MONOTONE-INCREASING sequence of non-negative doubles the bit patterns are themselves monotone as u64
    (the IEEE-754 total-order property for non-negative values), so delta -> zigzag -> varint is exact AND
    tiny. The reinterpretation + integer delta is exactly invertible, so t_monotonic round-trips to the bit
    — no quantization (a float-subtraction delta would NOT be invertible). A non-monotone blip still round-
    trips (zigzag handles the negative delta), just less compactly."""
    src = col if col.dtype == np.float64 else col.astype(np.float64)
    u = np.ascontiguousarray(src).view(np.uint64)
    as_i = u.astype(np.int64)            # bit patterns as signed for a signed delta
    deltas = np.empty_like(as_i)
    if as_i.shape[0] > 0:
        deltas[0] = as_i[0]
        deltas[1:] = np.diff(as_i)
    return _varint_encode_u64(_zigzag_encode(deltas))


def _dec_float64_mono_bits(buf: bytes, n: int) -> npt.NDArray[np.float64]:
    u, consumed = _varint_decode_u64(buf, n)
    if consumed != len(buf):
        raise CodecError(f"mono-float column has {len(buf) - consumed} trailing bytes (corrupt blob)")
    deltas = _zigzag_decode(u)
    as_i = np.cumsum(deltas, dtype=np.int64)
    return as_i.astype(np.uint64).view(np.float64).astype(np.float64).copy()


# ============================================================================================
# Container encode / decode — the self-describing blob (the cross-batch contract).
# ============================================================================================
def _encode_one_column(spec: ColumnSpec, col: npt.NDArray[Any], T: int, D: int, K: int,
                       srv_rows_const: float) -> bytes:
    """Encode a single column by its declared codec. Pure; returns the column payload bytes."""
    c = spec.codec
    if c is Codec.CONSTANT:
        val = {"n_threads": T, "d_ceiling": D}[spec.name]   # the geometry field this const column carries
        return struct.pack("<q", int(val))
    if c is Codec.CONSTANT_F64:
        return struct.pack("<d", float(srv_rows_const))
    if c is Codec.BITPACK:
        hi = {"d_ceiling": D, "k_per_thread": K, "n_threads": T}[spec.bound]
        return _enc_bitpack(col.astype(np.int64), _bits_for_range(hi))
    if c is Codec.DELTA_ZIGZAG_VARINT:
        return _enc_delta_zigzag_varint(col.astype(np.int64))
    if c is Codec.RLE_I64:
        return _enc_rle_i64(col.astype(np.int64))
    if c is Codec.MASK_BITS:
        return _enc_mask_bits(col.astype(np.uint8))
    if c is Codec.FLOAT64_RAW:
        return _enc_float64_raw(col.astype(np.float64))
    if c is Codec.FLOAT64_MONO_BITS:
        return _enc_float64_mono_bits(col.astype(np.float64))
    raise CodecError(f"no encoder for codec id {int(c)} (column {spec.name!r})")   # unreachable (closed enum)


def _encode_columns(n: int, T: int, D: int, K: int, s_min: int, srv_rows_const: float,
                    cols: Mapping[str, npt.NDArray[Any]]) -> bytes:
    """Assemble the self-describing container: header + geometry + column directory + per-column payloads,
    then zstd-compress the BODY. The header (magic, format version, n_columns) stays OUTSIDE compression so
    a reader sniffs magic+version without decompressing; everything else is compressed."""
    # 1) encode each declared column to its payload bytes (const columns ignore the empty data view).
    payloads: list[tuple[ColumnSpec, bytes]] = []
    for spec in COLUMNS:
        if spec.kind in ("scalar_const_int", "scalar_const_f64"):
            col: npt.NDArray[Any] = np.empty(0)   # const columns derive their value from geometry, not data
        else:
            col = cols[spec.name]
        payloads.append((spec, _encode_one_column(spec, col, T, D, K, srv_rows_const)))

    # 2) build the uncompressed body: geometry block + directory + concatenated payloads.
    body = bytearray()
    body += _GEOM.pack(T, D, K, s_min, n, SCHEMA_HASH)
    for spec, payload in payloads:
        name_b = spec.name.encode("ascii")
        if len(name_b) > 255:
            raise CodecError(f"column name too long for the directory: {spec.name!r}")
        body += _COLENT.pack(len(name_b), int(spec.codec), len(payload))
        body += name_b
    for _spec, payload in payloads:
        body += payload

    # 3) zstd-compress the body; prepend the uncompressed sniff header.
    comp = zstd.ZstdCompressor(level=ZSTD_LEVEL)
    compressed = comp.compress(bytes(body))
    return _HEADER.pack(MAGIC, FORMAT_VERSION, len(COLUMNS)) + compressed


# The decoded trajectory the dashboard / supervised consumers receive — a typed struct-of-arrays view.
@dataclass(frozen=True)
class DecodedTrajectory:
    """The decode result: the full struct-of-arrays trajectory + the per-trial geometry. Per-thread columns
    are (n, T) int64; masks are (n, T) uint8 0/1; the scalars are 1-D. Exact round-trip of what was
    appended (the cross-batch contract the dashboard + supervised batch read)."""
    n_decisions: int
    n_threads: int
    d_ceiling: int
    k_per_thread: int
    s_min: int
    server_rows_per_forward: float
    inflight: npt.NDArray[np.int64]
    ready: npt.NDArray[np.int64]
    msgs: npt.NDArray[np.int64]
    leaves: npt.NDArray[np.int64]
    rtt_us: npt.NDArray[np.int64]
    served: npt.NDArray[np.uint8]
    action: npt.NDArray[np.uint8]
    forward_rows: npt.NDArray[np.int64]
    reward: npt.NDArray[np.float64]
    t_monotonic: npt.NDArray[np.float64]


def _decode_one_column(spec: ColumnSpec, payload: bytes, n: int, T: int, D: int, K: int
                       ) -> npt.NDArray[Any] | int | float:
    """Decode a single column payload by its codec id. Const columns return the scalar value; data columns
    return the numpy array. Loud on any codec id the header carried that this build does not know."""
    c = spec.codec
    if c is Codec.CONSTANT:
        (val,) = struct.unpack("<q", payload)
        return int(val)
    if c is Codec.CONSTANT_F64:
        (val_f,) = struct.unpack("<d", payload)
        return float(val_f)
    if c is Codec.BITPACK:
        hi = {"d_ceiling": D, "k_per_thread": K, "n_threads": T}[spec.bound]
        return _dec_bitpack(payload, n * T, _bits_for_range(hi), (n, T))
    if c is Codec.DELTA_ZIGZAG_VARINT:
        is_scalar = spec.kind in ("scalar_int", "scalar_f64")
        return _dec_delta_zigzag_varint(payload, n, 0 if is_scalar else T)
    if c is Codec.RLE_I64:
        return _dec_rle_i64(payload, n * T, (n, T))
    if c is Codec.MASK_BITS:
        return _dec_mask_bits(payload, n, T)
    if c is Codec.FLOAT64_RAW:
        return _dec_float64_raw(payload, n)
    if c is Codec.FLOAT64_MONO_BITS:
        return _dec_float64_mono_bits(payload, n)
    raise CodecError(f"unknown codec id {int(c)} in directory (column {spec.name!r}) — "
                     f"a newer blob this decoder cannot read (ADR-0002 fail loud)")


def decode(blob: bytes) -> DecodedTrajectory:
    """Decode a trajectory blob back to the EXACT struct-of-arrays it was built from. BOUNDARY (ADR-0002):
    a wrong magic, an unsupported format version, a schema-hash mismatch, an unknown codec id, or any length
    disagreement is a loud CodecError — never a silently misread trajectory. The dashboard + supervised
    consumers call this; the contract is exact round-trip of what `append` saw."""
    if len(blob) < _HEADER.size:
        raise CodecError(f"trajectory blob is {len(blob)} bytes, too short for the header")
    magic, fmt_ver, n_columns = _HEADER.unpack_from(blob, 0)
    if magic != MAGIC:
        raise CodecError(f"bad trajectory magic {magic!r} (expected {MAGIC!r}; wire-contract drift)")
    if fmt_ver != FORMAT_VERSION:
        raise CodecError(f"trajectory format version {fmt_ver} != supported {FORMAT_VERSION} "
                         f"(an old decoder against a newer blob — ADR-0002 fail loud, never misread)")
    if n_columns != len(COLUMNS):
        raise CodecError(f"trajectory declares {n_columns} columns, this schema has {len(COLUMNS)} "
                         f"(schema drift; refuse to guess the layout)")
    dctx = zstd.ZstdDecompressor()
    try:
        body = dctx.decompress(blob[_HEADER.size:])
    except zstd.ZstdError as exc:
        raise CodecError(f"trajectory body failed zstd decompression: {exc!r}") from exc

    # geometry block
    if len(body) < _GEOM.size:
        raise CodecError("trajectory body too short for the geometry block (corrupt blob)")
    T, D, K, s_min, n_dec, schema_hash = _GEOM.unpack_from(body, 0)
    if schema_hash != SCHEMA_HASH:
        raise CodecError(f"trajectory schema hash {schema_hash:#x} != this build's {SCHEMA_HASH:#x} "
                         f"(the column schema/codecs differ from what wrote this blob — ADR-0002)")
    off = _GEOM.size

    # column directory: n_columns entries of (name_len, codec_id, payload_len) + name bytes.
    directory: list[tuple[str, int, int]] = []
    for _ in range(n_columns):
        if off + _COLENT.size > len(body):
            raise CodecError("trajectory directory truncated (corrupt blob)")
        name_len, codec_id, payload_len = _COLENT.unpack_from(body, off)
        off += _COLENT.size
        if off + name_len > len(body):
            raise CodecError("trajectory directory name truncated (corrupt blob)")
        name = body[off:off + name_len].decode("ascii")
        off += name_len
        directory.append((name, codec_id, payload_len))

    # read each payload by the directory length, decode by codec, key by name.
    spec_by_name = {c.name: c for c in COLUMNS}
    decoded: dict[str, Any] = {}
    srv_rows = 0.0
    for name, codec_id, payload_len in directory:
        if name not in spec_by_name:
            raise CodecError(f"trajectory directory names unknown column {name!r} (schema drift)")
        spec = spec_by_name[name]
        if codec_id != int(spec.codec):
            raise CodecError(f"column {name!r} codec id {codec_id} != schema {int(spec.codec)} "
                             f"(codec drift on one side — ADR-0002 fail loud)")
        if off + payload_len > len(body):
            raise CodecError(f"column {name!r} payload truncated (corrupt blob)")
        payload = body[off:off + payload_len]
        off += payload_len
        val = _decode_one_column(spec, payload, n_dec, T, D, K)
        if name == "server_rows_per_forward":
            assert isinstance(val, float)   # CONSTANT_F64 returns a float (narrows the union for P8)
            srv_rows = val
        elif spec.kind == "scalar_const_int":
            assert isinstance(val, int)     # CONSTANT returns an int
            decoded[name] = val
        else:
            decoded[name] = val

    # n_threads / d_ceiling const columns reconcile with the geometry block (one home; assert agreement).
    if int(decoded.get("n_threads", T)) != T or int(decoded.get("d_ceiling", D)) != D:
        raise CodecError("const column (n_threads/d_ceiling) disagrees with the geometry block (corrupt)")

    return DecodedTrajectory(
        n_decisions=int(n_dec), n_threads=int(T), d_ceiling=int(D), k_per_thread=int(K), s_min=int(s_min),
        server_rows_per_forward=srv_rows,
        inflight=decoded["inflight"], ready=decoded["ready"], msgs=decoded["msgs"],
        leaves=decoded["leaves"], rtt_us=decoded["rtt_us"], served=decoded["served"],
        action=decoded["action"], forward_rows=decoded["forward_rows"], reward=decoded["reward"],
        t_monotonic=decoded["t_monotonic"],
    )


def column_codec_table() -> list[dict[str, str]]:
    """Introspection for docs/dashboard: the declared per-column codec table (name -> codec -> bound).
    A new feature added to COLUMNS shows up here with zero extra wiring (the seam is self-describing)."""
    return [{"column": c.name, "codec": c.codec.name, "kind": c.kind, "bound": c.bound or "-"}
            for c in COLUMNS]
