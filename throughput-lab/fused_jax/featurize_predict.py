#!/usr/bin/env python
# throughput-lab/fused_jax/featurize_predict.py
# Purpose: the JAX (CPU) side of the fused-JAX BatchPredict COMPONENT (lever #1,
#   docs/notes/batchpredict-throughput-design-2026-06-26.md). Two responsibilities, kept apart:
#
#   (A) THE BELIEF WIRE CODEC (Python mirror of belief_wire.hpp — ADR-0012 P7: the byte layout has ONE
#       home, the .hpp; this side DERIVES the same layout from the same field widths, it does not
#       re-author it). decode_setup / decode_request unpack the C++-encoded belief-wire bytes;
#       encode_response packs predictions back. Little-endian, the .hpp's standing assumption.
#
#   (B) featurize_and_predict(setup, belief_batch) -> predictions: the FUSED path. It (1) featurizes via
#       the de-risked matmul `belief_indicator @ world_feature_matrix` + the §2.2 phase-2 pointwise maps
#       (mirroring cpp/src/features.cpp belief_features_nonempty), then (2) runs a NET forward over the
#       belief-feature blocks. featurize + net are ONE jitted XLA program (the design's intent — what
#       GPU-amortizes later). The net here is a STAND-IN (fixed pseudo-random dense weights): the
#       COMPONENT's point is the fused featurize+net PATH and its parity/timing, not the real weights
#       (real weights / GPU are the maintainer-owned integration).
#
#   The world_feature_matrix is env-static -> it arrives in the SETUP frame ONCE and is held resident
#   (the matmul's right operand, a device constant shared across the batch). The per-batch request
#   carries only the belief leaves.
#
#   PARITY (re-confirm the de-risk on this base): the matmul featurization reproduces C++
#   belief_features within tolerance; the legal-mask `informative` predicate is bit-exact in f32
#   (nworlds < 2^24 guard). Provability caveat: f32 marg/p_pos is a P6 BEHAVIORAL bar, not bit-exact;
#   f64 recovers the exact C++ double oracle (a provability fallback). See roundtrip_demo.py for the gate.
#
# Public Domain (The Unlicense).
import struct

import numpy as np

# ---- belief-wire field widths (mirror belief_wire.hpp; ONE home is the .hpp, these DERIVE it) -------
BELIEF_PROTOCOL_VERSION = 1
VERSION_BYTES = 1
COUNT_BYTES = 4   # u32 little-endian
WORD_BYTES = 8    # u64 little-endian (a rank-bitset word)
FLOAT_BYTES = 4   # f32 little-endian


def _u32(buf, at):
    return struct.unpack_from("<I", buf, at)[0]


# ============================================================================================
#  (A) THE BELIEF WIRE CODEC (Python mirror of belief_wire.hpp)
# ============================================================================================

def decode_setup(frame):
    """Decode a SETUP frame -> dict(N, nD, nworlds, kW64, columns) where `columns` is a list of N+nD
    rank-bitset word-lists (each kW64 u64). Fails LOUDLY (ValueError) on a malformed frame (ADR-0002)."""
    frame = bytes(frame)
    header = VERSION_BYTES + 4 * COUNT_BYTES
    if len(frame) < header:
        raise ValueError("belief wire: setup frame too short for header")
    if frame[0] != BELIEF_PROTOCOL_VERSION:
        raise ValueError(f"belief wire: setup protocol byte {frame[0]} != {BELIEF_PROTOCOL_VERSION}")
    N = _u32(frame, VERSION_BYTES)
    nD = _u32(frame, VERSION_BYTES + COUNT_BYTES)
    nworlds = _u32(frame, VERSION_BYTES + 2 * COUNT_BYTES)
    kW64 = _u32(frame, VERSION_BYTES + 3 * COUNT_BYTES)
    if kW64 == 0 or nworlds == 0:
        raise ValueError("belief wire: setup kW64/nworlds is 0")
    nwords = (N + nD) * kW64
    if len(frame) != header + nwords * WORD_BYTES:
        raise ValueError("belief wire: setup body length != (N+nD)*kW64*u64")
    words = np.frombuffer(frame, dtype="<u8", count=nwords, offset=header)
    columns = [words[c * kW64:(c + 1) * kW64] for c in range(N + nD)]
    return dict(N=N, nD=nD, nworlds=nworlds, kW64=kW64, columns=columns)


def decode_request(frame):
    """Decode a REQUEST frame -> dict(B, kW64, loc[B], collected[B], belief[B, kW64] u64). Fails LOUDLY
    on a malformed frame (ADR-0002)."""
    frame = bytes(frame)
    header = VERSION_BYTES + 2 * COUNT_BYTES
    if len(frame) < header:
        raise ValueError("belief wire: request frame too short for header")
    if frame[0] != BELIEF_PROTOCOL_VERSION:
        raise ValueError(f"belief wire: request protocol byte {frame[0]} != {BELIEF_PROTOCOL_VERSION}")
    B = _u32(frame, VERSION_BYTES)
    kW64 = _u32(frame, VERSION_BYTES + COUNT_BYTES)
    if B == 0 or kW64 == 0:
        raise ValueError("belief wire: request B/kW64 is 0")
    rec = 2 * COUNT_BYTES + kW64 * WORD_BYTES
    if len(frame) != header + B * rec:
        raise ValueError("belief wire: request body length != B*(loc+collected+kW64*u64)")
    loc = np.empty(B, dtype=np.uint32)
    collected = np.empty(B, dtype=np.uint32)
    belief = np.empty((B, kW64), dtype="<u8")
    for i in range(B):
        base = header + i * rec
        loc[i] = _u32(frame, base)
        collected[i] = _u32(frame, base + COUNT_BYTES)
        belief[i] = np.frombuffer(frame, dtype="<u8", count=kW64, offset=base + 2 * COUNT_BYTES)
    return dict(B=B, kW64=kW64, loc=loc, collected=collected, belief=belief)


def encode_response(values, logits, n_actions):
    """Encode predictions -> a RESPONSE frame (mirror belief_wire.hpp decode_response). `values` is
    length-B; `logits` is (B, n_actions) (ignored when n_actions==0). Little-endian f32."""
    values = np.asarray(values, dtype="<f4")
    B = values.shape[0]
    out = bytearray()
    out.append(BELIEF_PROTOCOL_VERSION)
    out += struct.pack("<II", B, n_actions)
    if n_actions == 0:
        for r in range(B):
            out += struct.pack("<f", float(values[r]))
    else:
        logits = np.asarray(logits, dtype="<f4").reshape(B, n_actions)
        for r in range(B):
            out += struct.pack("<f", float(values[r]))
            out += logits[r].tobytes()
    return bytes(out)


# ============================================================================================
#  (B) THE FUSED FEATURIZE + NET — the matmul featurization (de-risked) + a stand-in net, ONE program
# ============================================================================================

def unpack_columns_to_dense(columns, nworlds):
    """Unpack the N+nD column rank-bitsets (each kW64 u64 words) into a dense (nworlds x (N+nD)) bit
    matrix of 0/1 (the matmul's right operand). Mirrors the de-risk unpack."""
    ncol = len(columns)
    M = np.zeros((nworlds, ncol), dtype=np.uint8)
    ranks = np.arange(nworlds, dtype=np.uint64)
    word_idx = (ranks >> np.uint64(6)).astype(np.intp)
    bit_idx = ranks & np.uint64(63)
    for c, words in enumerate(columns):
        w = np.asarray(words, dtype=np.uint64)
        M[:, c] = ((w[word_idx] >> bit_idx) & np.uint64(1)).astype(np.uint8)
    return M


def unpack_beliefs_to_dense(belief_words, nworlds):
    """Unpack a (B x kW64) belief rank-bitset into a dense (B x nworlds) 0/1 indicator matrix (the
    matmul's left operand). bit r of belief i set iff the rank-r world is live in belief i."""
    B = belief_words.shape[0]
    ranks = np.arange(nworlds, dtype=np.uint64)
    word_idx = (ranks >> np.uint64(6)).astype(np.intp)
    bit_idx = ranks & np.uint64(63)
    out = np.empty((B, nworlds), dtype=np.uint8)
    bw = np.asarray(belief_words, dtype=np.uint64)
    for i in range(B):
        out[i] = ((bw[i][word_idx] >> bit_idx) & np.uint64(1)).astype(np.uint8)
    return out


class FusedBatchPredict:
    """The fused-JAX BatchPredict impl. Constructed from a decoded SETUP (the env-static
    world_feature_matrix held resident as a device constant). `predict(request)` featurizes the belief
    batch (matmul) + runs the stand-in net, ALL in one jitted XLA program, and returns predictions.

    dtype 'float32' is the realistic wire dtype; 'float64' (needs jax_enable_x64 set before import)
    recovers the exact C++ double oracle for the provability fallback."""

    def __init__(self, setup, n_actions=8, dtype="float32", net_seed=0):
        import jax
        import jax.numpy as jnp
        self._jax = jax
        self._jnp = jnp
        self.N = setup["N"]
        self.nD = setup["nD"]
        self.nworlds = setup["nworlds"]
        self.n_actions = n_actions
        self._dt = jnp.float64 if dtype == "float64" else jnp.float32

        # nworlds < 2^24 guard: the bit-exact-`informative` argument (exact integer matmul counts) holds
        # only while f32 represents every count exactly (<= 2^24). Fail loud otherwise (ADR-0002 /
        # model-bound-is-conjecture-not-witness: this is instance-contingent, re-measure past the guard).
        if dtype == "float32" and self.nworlds >= (1 << 24):
            raise ValueError(
                f"belief featurize: nworlds={self.nworlds} >= 2^24: f32 loses count-exactness, the "
                f"bit-exact `informative` guarantee no longer holds — re-measure or run x64.")

        Wdense = unpack_columns_to_dense(setup["columns"], self.nworlds)  # (nworlds x (N+nD)) uint8
        self._W = jnp.asarray(Wdense.astype(np.float32), dtype=self._dt)
        self._log_nworlds = float(np.log(self.nworlds))

        # The STAND-IN net: a single dense layer over the belief-feature blocks -> (value, logits). Fixed
        # pseudo-random weights (deterministic per net_seed) — a placeholder for the real net so the
        # FUSED featurize+net path is exercised end to end. feat_in = N (marg) + nD (p_pos) + nD
        # (informative) + 2 (marg_sum, sharpness). (nonempty is constant 1 here; omitted from the net in.)
        self._feat_in = self.N + 2 * self.nD + 2
        rng = np.random.default_rng(net_seed)
        self._Wv = jnp.asarray(rng.standard_normal((self._feat_in,)).astype(np.float32) * 0.01, dtype=self._dt)
        self._bv = self._dt(0.0)
        self._Wl = jnp.asarray(rng.standard_normal((self._feat_in, n_actions)).astype(np.float32) * 0.01,
                               dtype=self._dt)
        self._bl = jnp.asarray(np.zeros(n_actions, dtype=np.float32), dtype=self._dt)

        N, nD, dt, log_nw = self.N, self.nD, self._dt, self._log_nworlds
        Wv, bv, Wl, bl = self._Wv, self._bv, self._Wl, self._bl

        def _fused(ind_b, Wmat):
            # ---- phase 1: the featurization matmul (de-risked: == C++ per-world sweep) ----
            counts = ind_b @ Wmat                       # (B x (N+nD)) integer column counts
            nb = ind_b.sum(axis=1, keepdims=True)       # (B x 1)
            inv = 1.0 / nb
            bit_cnt = counts[:, :N]
            det_cnt = counts[:, N:]
            marg = bit_cnt * inv
            p_pos = det_cnt * inv
            informative = ((det_cnt > 0) & (det_cnt < nb)).astype(dt)   # the legal mask — bit-exact in f32
            marg_sum = marg.sum(axis=1, keepdims=True)
            sharpness = (jnp.log(nb) / log_nw)          # (B x 1)
            # ---- phase 2 -> net: assemble the belief-feature block, one fused forward ----
            feat = jnp.concatenate([marg, p_pos, informative, marg_sum, sharpness], axis=1)  # (B x feat_in)
            value = feat @ Wv + bv                      # (B,)
            logits = feat @ Wl + bl                     # (B x n_actions)
            return value, logits, marg, p_pos, informative, marg_sum, sharpness

        self._fused = jax.jit(_fused)

    def featurize_and_predict(self, request):
        """The fused path: a decoded REQUEST (belief batch) -> predictions + the featurization blocks.
        Returns dict(value[B], logits[B,n_actions]) plus marg/p_pos/informative/marg_sum/sharpness for
        the parity gate. The world_feature_matrix is the resident SETUP constant (sent ONCE)."""
        jnp = self._jnp
        ind = unpack_beliefs_to_dense(request["belief"], self.nworlds)  # (B x nworlds) 0/1
        ind_b = jnp.asarray(ind.astype(np.float32), dtype=self._dt)
        value, logits, marg, p_pos, informative, marg_sum, sharpness = self._fused(ind_b, self._W)
        self._jax.block_until_ready((value, logits))
        return dict(
            value=np.asarray(value), logits=np.asarray(logits),
            marg=np.asarray(marg), p_pos=np.asarray(p_pos),
            informative=np.asarray(informative),
            marg_sum=np.asarray(marg_sum)[:, 0], sharpness=np.asarray(sharpness)[:, 0],
        )
