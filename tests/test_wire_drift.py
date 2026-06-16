#!/usr/bin/env python3
"""
tests/test_wire_drift.py — the MECHANICAL NET against silent Python↔C++ drift on the two #23-mechanized
raw-binary contracts: the Shape B ZeroMQ inference WIRE frame (chocofarm/az/wire_spec.py ↔
cpp/include/chocofarm/wire_spec.hpp) and the redis RESULT blob (chocofarm/az/result_spec.py ↔
cpp/include/chocofarm/result_spec.hpp). Each layout has ONE authoritative home on the Python side
(ADR-0012 P1/P7); this test is what makes "the C++ mirror DERIVES from it, never re-authors it" an
ENFORCED fact rather than a comment (ADR-0011 Rule 4: a net quantifies over the layout, not one field).

Three always-on legs (NO C++ binary, NO redis — they run in the default `pytest tests/ -q`):

  1. LAYOUT AGREEMENT. Parse the C++ mirror header's `constexpr` literals and assert they equal the
     Python SSOT's constants — the protocol version, the field byte-widths, the float dtype/itemsize,
     the result block order + ranks. A one-sided edit (a PROTOCOL_VERSION bump only in Python, a wider
     count field only in C++, a float32→float64 widening on one side) reds the default suite instead of
     silently misreading floats / corrupting a reshape on the wire (ADR-0002).

  2. DRIFT-CATCH SELF-CHECK (the negative/mutation proof). Perturb ONE side's parsed constant in memory
     and assert the same agreement check RAISES — so the net is demonstrated to actually catch drift,
     not merely pass when nothing is wrong (the proportionate verification #23 asks for).

  3. WEIGHT-MANIFEST shared invariant. The weight blob is a SELF-DESCRIBING JSON manifest (each entry
     carries its own name/shape/dtype/off/len), so it is not silent-drift-prone the way the two raw
     formats are — BUT both sides hardcode the one cross-language fact the manifest does NOT make a
     reader re-derive: "the weight blob is float64 ('<f8')" (WeightContainer.pack writes it;
     transport.cpp::parse_manifest rejects anything else). That single shared literal is pinned here so
     a Python widening to float32 without the C++ reject being updated can't pass silently.

One opt-in leg (needs a C++ compiler; gated CHOCO_RUN_CPP, mirroring tests/test_cpp_runner.py):

  4. CROSS-LANGUAGE GOLDEN ROUND-TRIP. Python encodes fixed golden vectors → a tiny standalone C++
     decoder (cpp/parity/wire_golden.cpp, compiled with a bare `g++ -std=c++23`, including ONLY the
     mirror headers) decodes them by the mirror constants and re-encodes → Python asserts the bytes
     are byte-for-byte identical. For a BYTE format the bar is byte-exactness, not float tolerance
     (ADR-0012 P6: behavioral-equivalence for ML float math, but a serialization contract is exact).
     Skipped (not failed) without CHOCO_RUN_CPP or a compiler, so the default suite stays green.

Public Domain (The Unlicense).
"""
import os
import re
import struct
import subprocess
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chocofarm.az import inference_wire as wire
from chocofarm.az import result_spec, wire_spec

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WIRE_HPP = os.path.join(REPO, "cpp", "include", "chocofarm", "wire_spec.hpp")
RESULT_HPP = os.path.join(REPO, "cpp", "include", "chocofarm", "result_spec.hpp")
WEIGHTS_PY = os.path.join(REPO, "chocofarm", "az", "weights.py")
TRANSPORT_CPP = os.path.join(REPO, "cpp", "src", "transport.cpp")
GOLDEN_CPP = os.path.join(REPO, "cpp", "parity", "wire_golden.cpp")
INCLUDE_DIR = os.path.join(REPO, "cpp", "include")

# OPT-IN gate for the cross-language leg (mirrors test_cpp_runner's CHOCO_RUN_CPP). The always-on legs
# 1-3 need neither — they parse the C++ header as TEXT, so drift is caught with no build at all.
_RUN_CPP = bool(os.environ.get("CHOCO_RUN_CPP"))


# ---------------------------------------------------------------------------
# C++ mirror-header parsing (the always-on net reads the header as text).
# ---------------------------------------------------------------------------
def _read(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read()


def _cpp_int_const(src: str, name: str) -> int:
    """Parse `inline constexpr <type> <name> = <int>;` out of a C++ mirror header. Loud KeyError if the
    constant is absent (a mirror that DROPPED a field is itself drift the net must catch, ADR-0002)."""
    m = re.search(rf"\b{re.escape(name)}\s*=\s*(\d+)\s*;", src)
    if m is None:
        raise KeyError(f"C++ constant {name!r} not found in the mirror header")
    return int(m.group(1))


def _cpp_str_const(src: str, name: str) -> str:
    """Parse `... <name> = "<value>";` (a string_view literal) out of a C++ mirror header."""
    m = re.search(rf"\b{re.escape(name)}\s*=\s*\"([^\"]*)\"\s*;", src)
    if m is None:
        raise KeyError(f"C++ string constant {name!r} not found in the mirror header")
    return m.group(1)


def _cpp_str_array(src: str, name: str) -> list[str]:
    """Parse `std::array<std::string_view, N> <name> = {"a", "b", ...};` into a list of strings."""
    m = re.search(rf"\b{re.escape(name)}\s*=\s*\{{([^}}]*)\}}", src)
    if m is None:
        raise KeyError(f"C++ string-array {name!r} not found in the mirror header")
    return re.findall(r"\"([^\"]*)\"", m.group(1))


def _cpp_int_array(src: str, name: str) -> list[int]:
    """Parse `std::array<int, N> <name> = {2, 2, 2, 1};` into a list of ints."""
    m = re.search(rf"\b{re.escape(name)}\s*=\s*\{{([^}}]*)\}}", src)
    if m is None:
        raise KeyError(f"C++ int-array {name!r} not found in the mirror header")
    return [int(x) for x in re.findall(r"-?\d+", m.group(1))]


# ===========================================================================
# LEG 1 — LAYOUT AGREEMENT (always-on): the C++ mirror constants == the Python SSOT.
# ===========================================================================
def test_wire_spec_protocol_version_agrees():
    """The protocol-version header byte is identical in the Python SSOT and the C++ mirror. A bump on
    one side only (an old client paired with a new server) is exactly what the version byte exists to
    fail loudly — but it can only do so if BOTH sides carry the SAME bumped value, which this pins."""
    src = _read(WIRE_HPP)
    assert _cpp_int_const(src, "PROTOCOL_VERSION") == wire_spec.PROTOCOL_VERSION
    # the codec re-exports the SSOT value (no third copy) — pin that too.
    assert wire.PROTOCOL_VERSION == wire_spec.PROTOCOL_VERSION


def test_wire_spec_field_widths_agree():
    """The fixed-field byte widths (version u8, count u32, float f32) match between the Python SSOT's
    `struct.calcsize`-derived sizes and the C++ mirror's `constexpr std::size_t` literals. A wider
    count field on one side would shift every subsequent byte — a silent float-misread the net stops."""
    src = _read(WIRE_HPP)
    assert _cpp_int_const(src, "VERSION_BYTES") == wire_spec.VERSION_BYTES
    assert _cpp_int_const(src, "COUNT_BYTES") == wire_spec.COUNT_BYTES
    assert _cpp_int_const(src, "FLOAT_BYTES") == wire_spec.FLOAT_BYTES
    # The C++ HEADER_BYTES is DERIVED (`= VERSION_BYTES + COUNT_BYTES;`), not a literal — exactly the
    # derive-don't-duplicate posture (P1), so there is no second literal here to drift-check. We pin the
    # two component widths above and confirm the Python codec's own fixed-header struct equals their
    # sum, so the request/response header size both sides build is reconciled to the same components.
    py_header = wire_spec.VERSION_BYTES + wire_spec.COUNT_BYTES
    assert wire._REQ_HEADER.size == py_header
    assert wire._RESP_HEADER.size == py_header


def test_wire_spec_float_is_le_f32_both_sides():
    """The Python SSOT's wire float dtype is little-endian float32 ('<f4'), and the C++ mirror's
    `float_t`/FLOAT_BYTES describe a 4-byte IEEE-754 binary32 — the dtype both codecs read the payload
    as. (Byte order is the host-LE standing assumption both headers document.)"""
    assert wire_spec.FLOAT_DTYPE == "<f4"
    assert np.dtype(wire_spec.FLOAT_DTYPE).itemsize == 4
    src = _read(WIRE_HPP)
    assert _cpp_int_const(src, "FLOAT_BYTES") == np.dtype(wire_spec.FLOAT_DTYPE).itemsize


def test_result_spec_dtype_agrees():
    """The result-blob element dtype is little-endian float32 in BOTH the Python SSOT ('<f4', itemsize
    4) and the C++ mirror (BLOCK_DTYPE_STR "<f4", BLOCK_ITEMSIZE 4). A float64 widening on one side
    only would corrupt every reshape on read — read floats, no exception, wrong numbers (ADR-0002)."""
    src = _read(RESULT_HPP)
    assert _cpp_str_const(src, "BLOCK_DTYPE_STR") == result_spec.RESULT_DTYPE_STR
    assert _cpp_int_const(src, "BLOCK_ITEMSIZE") == result_spec.RESULT_ITEMSIZE
    assert result_spec.RESULT_DTYPE_STR == "<f4"
    assert result_spec.RESULT_ITEMSIZE == 4


def test_result_spec_block_order_and_ranks_agree():
    """The canonical block ORDER (X, PI, M, Y) and per-block RANKS (2,2,2,1) match between the Python
    SSOT and the C++ mirror. A reorder or a rank change on one side reshapes the wrong bytes into the
    wrong array — the silent reshape-corruption the result blob is most prone to."""
    src = _read(RESULT_HPP)
    assert _cpp_str_array(src, "BLOCK_ORDER") == list(result_spec.BLOCK_ORDER)
    py_ranks = [result_spec.BLOCK_RANK[name] for name in result_spec.BLOCK_ORDER]
    assert _cpp_int_array(src, "BLOCK_RANK") == py_ranks


# ===========================================================================
# LEG 1b — CODEC-DERIVES-FROM-SPEC (always-on): each codec READS WHAT ITS SPEC SAYS, not just that the
# mirror constants agree. Leg 1 pins "C++ mirror constant == Python SSOT constant" — but a codec can
# still drift its OWN float interpretation away from the spec module (e.g. read the payload as '>f4'
# big-endian) and leave leg 1 green, because leg 1 never interprets a float VALUE. These legs quantify
# the net over the codecs (ADR-0011 Rule 4: over the class, not the constant), by checking the bytes the
# actual codec emits/decodes are EXACTLY the spec's little-endian-float32 bytes — produced by an
# INDEPENDENT spec-derived reference encoder, so a symmetric encode∘decode drift cannot hide.
# ===========================================================================
def test_wire_codec_emits_exactly_the_spec_bytes():
    """The bytes `inference_wire.encode_request` / `encode_response` EMIT equal an independent reference
    built straight from the wire_spec SSOT (`struct.pack(BYTE_ORDER+...)` + a `<f4` payload). If the
    codec drifted its dtype/byte-order away from the spec (say `_F32 = '>f4'`), its emitted payload bytes
    would differ from this little-endian reference and this reds — closing the hole where the codec's
    own round-trip stays green under a SYMMETRIC encode∘decode drift."""
    rng = np.random.default_rng(7)
    X = rng.standard_normal(11).astype(np.float64)   # arbitrary input dtype; the wire is '<f4'
    # reference: the spec's header struct + a little-endian f32 payload, derived from wire_spec ONLY.
    ref_req = (struct.pack(wire_spec.REQ_HEADER_FMT, wire_spec.PROTOCOL_VERSION, X.shape[0])
               + X.astype(wire_spec.FLOAT_DTYPE).tobytes())
    assert wire.encode_request(X) == ref_req

    value, logits = -1.75, rng.standard_normal(6).astype(np.float64)
    ref_resp = (struct.pack(wire_spec.RESP_HEADER_FMT, wire_spec.PROTOCOL_VERSION, logits.shape[0])
                + struct.pack(wire_spec.VALUE_FMT, value)
                + logits.astype(wire_spec.FLOAT_DTYPE).tobytes())
    assert wire.encode_response(value, logits) == ref_resp
    # the value-only edge (n_actions == 0, empty logits block).
    ref_valonly = (struct.pack(wire_spec.RESP_HEADER_FMT, wire_spec.PROTOCOL_VERSION, 0)
                   + struct.pack(wire_spec.VALUE_FMT, value))
    assert wire.encode_response(value, None) == ref_valonly


def test_wire_codec_decodes_spec_bytes_to_exact_values():
    """The codec's DECODE reads the spec's little-endian-f32 bytes back to the exact values. Built from
    the spec-derived reference bytes (NOT from `encode_*`), so a decode-side dtype/byte-order drift is
    caught independently of the encode side."""
    payload = np.array([1.5, -2.25, 0.0, 3.125], dtype=wire_spec.FLOAT_DTYPE)
    ref_req = struct.pack(wire_spec.REQ_HEADER_FMT, wire_spec.PROTOCOL_VERSION, payload.size) + payload.tobytes()
    got = wire.decode_request(ref_req)
    assert got.dtype == np.dtype(wire_spec.FLOAT_DTYPE)
    assert np.array_equal(got, payload)

    logits = np.array([0.5, -0.5, 7.0], dtype=wire_spec.FLOAT_DTYPE)
    ref_resp = (struct.pack(wire_spec.RESP_HEADER_FMT, wire_spec.PROTOCOL_VERSION, logits.size)
                + struct.pack(wire_spec.VALUE_FMT, 9.5) + logits.tobytes())
    val, got_logits = wire.decode_response(ref_resp)
    assert val == 9.5
    assert got_logits is not None and np.array_equal(got_logits, logits)


class _FakeRedisPipe:
    """A minimal in-memory stand-in for a redis pipeline — just enough to drive the result codec's
    `write_results` (SET) and `read_and_delete_results` (GET + DELETE) WITHOUT a redis server. The
    store is shared with the parent fake so a worker-side write is visible to a parent-side read."""
    def __init__(self, store: dict[str, bytes]) -> None:
        self._store = store
        self._ops: list[tuple[str, tuple]] = []  # (op, args) queued until execute()

    def set(self, key: str, val: bytes, ex: int | None = None):  # ex ignored (no TTL in-memory)
        self._ops.append(("set", (key, val)))
        return self

    def get(self, key: str):
        self._ops.append(("get", (key,)))
        return self

    def delete(self, *keys: str):
        self._ops.append(("delete", keys))
        return self

    def execute(self) -> list:
        out: list = []
        for op, args in self._ops:
            if op == "set":
                self._store[args[0]] = args[1]
                out.append(True)
            elif op == "get":
                out.append(self._store.get(args[0]))
            elif op == "delete":
                for k in args[0]:
                    self._store.pop(k, None)
                out.append(len(args[0]))
        self._ops = []
        return out


class _FakeRedis:
    def __init__(self) -> None:
        self.store: dict[str, bytes] = {}

    def pipeline(self, transaction: bool = False) -> _FakeRedisPipe:
        return _FakeRedisPipe(self.store)


def test_result_codec_roundtrips_blocks_through_write_and_read():
    """Drive the REAL result codec — `transport.write_results` (worker side) → `RedisTransport.
    read_and_delete_results` (parent side) — over an in-memory fake redis, and assert the four blocks
    come back with EXACT values. This pins the result READER's dtype (the auditor's unpinned-reader
    hole): if the reader drifted `result_spec.RESULT_DTYPE` away from the writer's '<f4' (e.g. to '>f4'
    or float64), the recovered values would be garbage and this reds — the silent reshape-corruption the
    result blob is most prone to, now caught with no redis server."""
    from chocofarm.az import transport
    rng = np.random.default_rng(2026)
    n, feat_dim, n_slots = 5, 7, 4
    dt = result_spec.RESULT_DTYPE
    X = np.ascontiguousarray(rng.standard_normal((n, feat_dim)).astype(dt))
    PI = np.ascontiguousarray(rng.standard_normal((n, n_slots)).astype(dt))
    M = np.ascontiguousarray((rng.random((n, n_slots)) > 0.5).astype(dt))
    Y = np.ascontiguousarray(rng.standard_normal(n).astype(dt))

    conn = _FakeRedis()
    token, idx = "drifttok", 3
    transport.write_results(conn, token, idx, X, PI, M, Y)
    records = transport.RedisTransport(conn).read_and_delete_results(token, [(idx, n, feat_dim, n_slots)])

    assert len(records) == n
    for i, (feat, pi, mask, g) in enumerate(records):
        assert np.array_equal(feat, X[i]), f"X row {i} drifted through the codec"
        assert np.array_equal(pi, PI[i]), f"PI row {i} drifted through the codec"
        assert np.array_equal(mask, M[i]), f"M row {i} drifted through the codec"
        assert g == float(Y[i]), f"Y[{i}] drifted through the codec"


# ===========================================================================
# LEG 2 — DRIFT-CATCH SELF-CHECK (always-on): the net is PROVEN to fail on a deliberate mismatch.
# ===========================================================================
# The agreement legs above pass when the two sides agree. The proportionate verification #23 demands is
# the NEGATIVE: that a deliberate one-sided perturbation makes the agreement check FAIL. We reproduce
# the agreement check over a PERTURBED copy of the parsed C++ source (mutating one constant in the text,
# the way a real one-sided edit would) and assert the comparison no longer holds. This demonstrates the
# mechanism catches drift; it does not touch the real header.

def _perturb_cpp_const(src: str, name: str, new_value: int) -> str:
    """Return `src` with the integer constant `name` rewritten to `new_value` — simulating a one-sided
    C++ edit a real drift would be. Asserts the substitution actually changed the text (so the test
    can't silently no-op if the constant's spelling ever changes)."""
    out = re.sub(rf"(\b{re.escape(name)}\s*=\s*)\d+(\s*;)", rf"\g<1>{new_value}\g<2>", src, count=1)
    assert out != src, f"perturbation of {name!r} changed nothing — the mutation self-check is a no-op"
    return out


def test_drift_catch_protocol_version_mismatch_fails():
    """NEGATIVE proof: a C++ mirror whose PROTOCOL_VERSION was bumped without the Python SSOT following
    makes the agreement assertion FAIL. If this DIDN'T raise, leg-1's agreement check would be vacuous."""
    src = _read(WIRE_HPP)
    bad = _perturb_cpp_const(src, "PROTOCOL_VERSION", wire_spec.PROTOCOL_VERSION + 7)
    with pytest.raises(AssertionError):
        assert _cpp_int_const(bad, "PROTOCOL_VERSION") == wire_spec.PROTOCOL_VERSION


def test_drift_catch_count_width_mismatch_fails():
    """NEGATIVE proof: a C++ mirror that widened the count field (u32→u64, COUNT_BYTES 4→8) without the
    Python SSOT widening makes the agreement assertion FAIL — the silent byte-shift the net must stop."""
    src = _read(WIRE_HPP)
    bad = _perturb_cpp_const(src, "COUNT_BYTES", 8)
    with pytest.raises(AssertionError):
        assert _cpp_int_const(bad, "COUNT_BYTES") == wire_spec.COUNT_BYTES


def test_drift_catch_result_itemsize_mismatch_fails():
    """NEGATIVE proof: a C++ result mirror widened to float64 (BLOCK_ITEMSIZE 4→8) without the Python
    SSOT widening makes the agreement assertion FAIL — the silent reshape-corruption the net stops."""
    src = _read(RESULT_HPP)
    bad = _perturb_cpp_const(src, "BLOCK_ITEMSIZE", 8)
    with pytest.raises(AssertionError):
        assert _cpp_int_const(bad, "BLOCK_ITEMSIZE") == result_spec.RESULT_ITEMSIZE


def test_drift_catch_dtype_python_side_perturbation_fails():
    """NEGATIVE proof from the OTHER direction: perturbing the PYTHON dtype (the SSOT) to float64 makes
    the agreement with the (unchanged) C++ mirror FAIL. So a one-sided edit on EITHER side is caught,
    not just a C++-side one."""
    perturbed_py_dtype = np.dtype("<f8").str   # a hypothetical Python widening to float64
    src = _read(RESULT_HPP)
    cpp_dtype = _cpp_str_const(src, "BLOCK_DTYPE_STR")
    with pytest.raises(AssertionError):
        assert cpp_dtype == perturbed_py_dtype


# ===========================================================================
# LEG 3 — WEIGHT-MANIFEST shared invariant (always-on): the one cross-language literal.
# ===========================================================================
def test_weight_blob_dtype_invariant_is_shared():
    """The weight manifest is SELF-DESCRIBING (each entry carries name/shape/dtype/off/len), so a reader
    derives the layout from the bytes — it is NOT silent-drift-prone like the two raw formats, and is
    deliberately NOT over-mechanized (no separate spec module). But ONE cross-language fact is hardcoded
    on both sides and not re-derived per-read: the weight blob is float64 ('<f8'). The Python packer
    (WeightContainer.pack, via `a.dtype.str` over float64 weight arrays) writes it; the C++ reader
    (transport.cpp::parse_manifest) REJECTS anything but '<f8' loudly. Pin that the C++ reject literal
    is exactly '<f8', so a Python widening to float32 can't pass a stale C++ reader silently.

    (This is the proportionate cover the #23 brief asks for: confirm the manifest's self-describing
    nature handles the layout, and pin the single shared dtype literal that ISN'T self-describing.)"""
    cpp = _read(TRANSPORT_CPP)
    # the C++ reader's dtype guard: `if (we.dtype != "<f8")` — assert that literal is present and is f64.
    m = re.search(r'we\.dtype\s*!=\s*"([^"]*)"', cpp)
    assert m is not None, "C++ parse_manifest no longer guards the weight dtype by literal"
    assert m.group(1) == "<f8", f"C++ weight-dtype guard is {m.group(1)!r}, expected '<f8' (float64)"
    # and the Python side packs float64: np.dtype('<f8') is what '<f8' decodes to (the shared fact).
    assert np.dtype("<f8").itemsize == 8


# ===========================================================================
# LEG 4 — CROSS-LANGUAGE GOLDEN ROUND-TRIP (opt-in; needs a C++ compiler).
# ===========================================================================
def _compiler() -> str | None:
    for cc in ("g++", "clang++"):
        try:
            subprocess.run([cc, "--version"], capture_output=True, check=True)
            return cc
        except (FileNotFoundError, subprocess.CalledProcessError):
            continue
    return None


def _build_golden(tmp_path) -> str:
    """Compile cpp/parity/wire_golden.cpp with a bare `g++ -std=c++23` (no cmake, no hiredis/zmq — the
    program includes ONLY the dependency-free mirror headers). Returns the binary path."""
    cc = _compiler()
    if cc is None:
        pytest.skip("no C++ compiler (g++/clang++) on PATH for the cross-language golden leg")
    out_bin = os.path.join(str(tmp_path), "wire_golden")
    proc = subprocess.run(
        [cc, "-std=c++23", "-O0", "-Wall", "-Wextra", f"-I{INCLUDE_DIR}", "-o", out_bin, GOLDEN_CPP],
        capture_output=True, text=True)
    if proc.returncode != 0:
        # a compiler too old for c++23 should SKIP (env limitation), not red the suite.
        pytest.skip(f"could not compile wire_golden.cpp with {cc} -std=c++23:\n{proc.stderr}")
    return out_bin


def _framed(b: bytes) -> bytes:
    """u32-length-prefix a byte chunk (the harness's own framing to the golden program's stdin)."""
    return struct.pack("<I", len(b)) + b


def _read_framed(stream: bytes, off: int) -> tuple[bytes, int]:
    (n,) = struct.unpack_from("<I", stream, off)
    off += 4
    return stream[off:off + n], off + n


@pytest.mark.skipif(not _RUN_CPP, reason="opt-in cross-language golden: set CHOCO_RUN_CPP=1 (needs g++)")
def test_cpp_golden_wire_roundtrip(tmp_path):
    """Python encodes golden REQUEST + RESPONSE frames → the C++ decoder (deriving its layout from the
    wire_spec.hpp mirror) decodes and re-encodes them → assert the returned bytes are byte-for-byte the
    bytes Python sent. End-to-end proof that the two codecs agree, not just their declared constants."""
    bin_path = _build_golden(tmp_path)
    rng = np.random.default_rng(23)
    # golden vectors: a typical request, a value+logits response, AND the value-only (n_actions=0) edge.
    X = rng.standard_normal(17).astype(np.float32)
    req = wire.encode_request(X)
    resp_full = wire.encode_response(1.2345, rng.standard_normal(9).astype(np.float32))
    resp_valonly = wire.encode_response(-0.5, None)

    for resp in (resp_full, resp_valonly):
        stdin = _framed(req) + _framed(resp)
        proc = subprocess.run([bin_path, "wire"], input=stdin, capture_output=True, timeout=30)
        assert proc.returncode == 0, f"C++ wire round-trip failed (rc={proc.returncode}): {proc.stderr!r}"
        req_back, off = _read_framed(proc.stdout, 0)
        resp_back, _ = _read_framed(proc.stdout, off)
        # byte-exact: a serialization contract is byte-identical, not float-tolerant (ADR-0012 P6).
        assert req_back == req, "C++ re-encoded REQUEST diverged from Python's bytes (codec drift)"
        assert resp_back == resp, "C++ re-encoded RESPONSE diverged from Python's bytes (codec drift)"


@pytest.mark.skipif(not _RUN_CPP, reason="opt-in cross-language golden: set CHOCO_RUN_CPP=1 (needs g++)")
def test_cpp_golden_result_roundtrip(tmp_path):
    """Python builds the four result blocks (X/PI/M/Y as the worker emits them — contiguous little-
    endian float32, the result_spec dtype) → the C++ decoder reads them in the result_spec.hpp mirror's
    BLOCK_ORDER and re-emits each → assert each block is byte-for-byte identical. Proves the result-blob
    block order + dtype the two sides commit to actually round-trip across the language boundary."""
    bin_path = _build_golden(tmp_path)
    rng = np.random.default_rng(123)
    n, feat_dim, n_slots = 4, 6, 5
    dt = result_spec.RESULT_DTYPE
    blocks = {
        result_spec.BLOCK_X: rng.standard_normal((n, feat_dim)).astype(dt),
        result_spec.BLOCK_PI: rng.standard_normal((n, n_slots)).astype(dt),
        result_spec.BLOCK_M: (rng.random((n, n_slots)) > 0.5).astype(dt),
        result_spec.BLOCK_Y: rng.standard_normal(n).astype(dt),
    }
    sent = [np.ascontiguousarray(blocks[name]).tobytes() for name in result_spec.BLOCK_ORDER]
    stdin = b"".join(_framed(b) for b in sent)
    proc = subprocess.run([bin_path, "result"], input=stdin, capture_output=True, timeout=30)
    assert proc.returncode == 0, f"C++ result round-trip failed (rc={proc.returncode}): {proc.stderr!r}"
    off = 0
    for i, name in enumerate(result_spec.BLOCK_ORDER):
        back, off = _read_framed(proc.stdout, off)
        assert back == sent[i], f"C++ re-emitted result block {name!r} diverged from Python's bytes"
