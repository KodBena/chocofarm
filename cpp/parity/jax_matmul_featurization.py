#!/usr/bin/env python
# cpp/parity/jax_matmul_featurization.py
# Purpose: the JAX (CPU) side of de-risk idea #1 — the fused-JAX matmul featurization
#   (docs/notes/batchpredict-throughput-design-2026-06-26.md / derisk-jax-matmul-featurization-
#   2026-06-26.md). It:
#     (a) PARITY: reproduces C++ chocofarm::belief_features (marg / p_pos / informative / marg_sum /
#         sharpness / nonempty) as a batched matmul + the §2.2 phase-2 pointwise maps, and diffs it
#         against the C++ double-precision oracle (chocofarm-belief-features-export). NB: C++ computes
#         in double, JAX in float32 -> this is a TOLERANCE/behavioral bar, NOT bit-exact. We quantify
#         max abs/rel diff per feature block, in both f32 and f64 (X64) JAX, to isolate the float32
#         gap from any reframing error.
#     (b) WIRE: prints the wire-payload tradeoff (belief rank-bitset bytes vs the current feature-vector
#         payload bytes).
#     (c) TIMING: times the CPU JAX matmul featurization for realistic batch sizes B (8/32/64).
#
#   The matmul reframe: marginals[t] = sum over live worlds w of bit_t(w) = belief_indicator . column_t,
#   where belief_indicator is the nb-bit live-world (rank) mask and the world_feature_matrix is the
#   env-static nworlds x (N+nD) bit matrix (column t = worlds containing treasure t; column N+j = worlds
#   the detector-j cover hits). Batched over B beliefs it is a (B x nworlds) . (nworlds x (N+nD)) matmul.
#   Phase 2 (mirrors cpp/src/features.cpp belief_features_nonempty): * inv on both marg and p_pos;
#   informative = (0 < det_cnt < nb); marg_sum = sum_t marg[t]; sharpness = log(nb)/log(nworlds).
#
#   This is an ADDITIVE de-risk prototype: it does NOT wire anything into the producer/server. It reads
#   the export blob and reports numbers; no production path is touched.
#
#   Run (CPU JAX is fine, no GPU):
#     cpp/build/chocofarm-belief-features-export --instance chocofarm/data/instance.json \
#         --faces chocofarm/data/faces.json > /tmp/export.json
#     /home/bork/w/vdc/venvs/generic/bin/python cpp/parity/jax_matmul_featurization.py /tmp/export.json
#
# Public Domain (The Unlicense).
import json
import sys
import time

import numpy as np


def unpack_columns(columns, nworlds):
    """Unpack N+nD column rank-bitsets (each a list of kW64 u64 words) into a dense
    (nworlds x (N+nD)) bit matrix of 0/1. Column c bit r set iff rank-r world is in column c's set."""
    ncol = len(columns)
    M = np.zeros((nworlds, ncol), dtype=np.uint8)
    for c, words in enumerate(columns):
        w = np.array(words, dtype=np.uint64)
        # bit r of column = bit (r & 63) of word (r >> 6); vectorize over ranks.
        ranks = np.arange(nworlds, dtype=np.uint64)
        word_idx = (ranks >> np.uint64(6)).astype(np.intp)
        bit_idx = (ranks & np.uint64(63))
        M[:, c] = ((w[word_idx] >> bit_idx) & np.uint64(1)).astype(np.uint8)
    return M


def unpack_indicator(words, nworlds):
    """Unpack a belief rank-bitset (kW64 u64 words) into a dense length-nworlds 0/1 vector."""
    w = np.array(words, dtype=np.uint64)
    ranks = np.arange(nworlds, dtype=np.uint64)
    word_idx = (ranks >> np.uint64(6)).astype(np.intp)
    bit_idx = (ranks & np.uint64(63))
    return ((w[word_idx] >> bit_idx) & np.uint64(1)).astype(np.uint8)


def featurize_jax(jnp, ind_batch, Wmat, N, nD, log_nworlds):
    """The fused matmul featurization, mirroring features.cpp belief_features_nonempty phase 2.
    ind_batch: (B x nworlds) live-world indicator (the matmul left operand).
    Wmat:      (nworlds x (N+nD)) bit world_feature_matrix (the right operand).
    Returns a dict of feature blocks (B x ...). nb assumed >= 1 per row (caller filters empties)."""
    counts = ind_batch @ Wmat                       # (B x (N+nD)) integer column counts (in the active dtype)
    nb = ind_batch.sum(axis=1, keepdims=True)       # (B x 1)
    inv = 1.0 / nb
    bit_cnt = counts[:, :N]                          # marg_raw
    det_cnt = counts[:, N:]                          # detector cover counts
    marg = bit_cnt * inv
    p_pos = det_cnt * inv
    informative = ((det_cnt > 0) & (det_cnt < nb)).astype(marg.dtype)
    marg_sum = marg.sum(axis=1)                      # sum_t marg[t]
    sharpness = jnp.log(nb[:, 0]) / log_nworlds
    nonempty = jnp.ones((ind_batch.shape[0],), dtype=marg.dtype)
    return dict(marg=marg, p_pos=p_pos, informative=informative,
                marg_sum=marg_sum, sharpness=sharpness, nonempty=nonempty)


def block_diffs(jax_blocks, refs, names):
    """Per-block max abs/rel diff between the JAX batch output and the C++ oracle rows."""
    out = {}
    for name in names:
        j = np.asarray(jax_blocks[name], dtype=np.float64)
        if j.ndim == 1:
            r = np.array([ref[name] for ref in refs], dtype=np.float64)
        else:
            r = np.array([ref[name] for ref in refs], dtype=np.float64)
        absd = np.abs(j - r)
        denom = np.maximum(np.abs(r), 1e-30)
        reld = absd / denom
        out[name] = (float(absd.max()), float(reld.max()))
    return out


def main():
    if len(sys.argv) < 2:
        print("usage: jax_matmul_featurization.py <export.json>", file=sys.stderr)
        return 2
    blob = json.load(open(sys.argv[1]))
    N, nD, nworlds = blob["N"], blob["nD"], blob["nworlds"]
    kW64, log_nworlds = blob["kW64"], blob["log_nworlds"]
    refs = blob["beliefs"]

    print(f"# export: N={N} nD={nD} nworlds={nworlds} kW64={kW64} N+nD={N + nD}")
    print(f"# reference beliefs: {len(refs)} (nb = {[r['nb'] for r in refs]})")

    # Dense world_feature_matrix (nworlds x (N+nD)) + per-belief indicators (build once in numpy).
    t0 = time.perf_counter()
    Wmat = unpack_columns(blob["columns"], nworlds)              # uint8
    indicators = np.stack([unpack_indicator(r["indicator"], nworlds) for r in refs])  # (R x nworlds) uint8
    print(f"# unpacked matrix+indicators in {time.perf_counter() - t0:.3f}s "
          f"(Wmat {Wmat.shape} {Wmat.nbytes / 1024:.0f} KiB dense uint8)")

    # Sanity: the matmul count must equal the popcount-AND the C++ does. Net the indicator unpack against
    # the oracle nb (sum of indicator bits == nb).
    nb_from_ind = indicators.sum(axis=1)
    nb_ref = np.array([r["nb"] for r in refs])
    assert np.array_equal(nb_from_ind, nb_ref), f"indicator nb mismatch: {nb_from_ind} vs {nb_ref}"
    print("# indicator-unpack nb == oracle nb (OK)")

    names = ["marg", "p_pos", "informative", "marg_sum", "sharpness", "nonempty"]

    import jax
    import jax.numpy as jnp

    # ---- (a) PARITY: f32 and f64 ----
    for x64, tag in [(False, "float32"), (True, "float64")]:
        jax.config.update("jax_enable_x64", x64)
        dt = jnp.float32 if not x64 else jnp.float64
        ind_j = jnp.asarray(indicators.astype(np.float32), dtype=dt)
        W_j = jnp.asarray(Wmat.astype(np.float32), dtype=dt)
        out = featurize_jax(jnp, ind_j, W_j, N, nD, log_nworlds)
        out = {k: np.asarray(v) for k, v in out.items()}
        diffs = block_diffs(out, refs, names)
        print(f"\n## PARITY (JAX {tag} vs C++ double oracle) — max abs / max rel per block")
        worst_abs = 0.0
        for nm in names:
            a, rl = diffs[nm]
            worst_abs = max(worst_abs, a)
            print(f"  {nm:12s} max|abs|={a:.3e}  max|rel|={rl:.3e}")
        # informative is a logic predicate (0/1) — it MUST be exactly equal (no float gap allowed).
        info_abs = diffs["informative"][0]
        print(f"  -> informative exact-equal: {'YES' if info_abs == 0.0 else 'NO (DIVERGES)'} "
              f"(max abs {info_abs:.0e})")
        print(f"  -> worst abs across all blocks ({tag}): {worst_abs:.3e}")

    # ---- (c) TIMING: CPU JAX matmul for B = 8/32/64 (use f32, the realistic wire dtype) ----
    jax.config.update("jax_enable_x64", False)
    W_j = jnp.asarray(Wmat.astype(np.float32))
    # A representative non-trivial belief population: cycle the reference indicators to fill B rows
    # (timing is dominated by the (B x nworlds).(nworlds x (N+nD)) matmul, not which beliefs).
    base = indicators.astype(np.float32)

    @jax.jit
    def fused(ind_b, Wm):
        counts = ind_b @ Wm
        nb = ind_b.sum(axis=1, keepdims=True)
        inv = 1.0 / nb
        marg = counts[:, :N] * inv
        p_pos = counts[:, N:] * inv
        informative = ((counts[:, N:] > 0) & (counts[:, N:] < nb)).astype(marg.dtype)
        return marg, p_pos, informative, marg.sum(axis=1)

    print("\n## TIMING (CPU JAX fused matmul featurization, float32) — median of 50 runs after warmup")
    for B in (8, 32, 64):
        idx = np.arange(B) % base.shape[0]
        ind_b = jnp.asarray(base[idx])
        # warmup (jit compile for this B-shape)
        for _ in range(5):
            r = fused(ind_b, W_j)
            jax.block_until_ready(r)
        ts = []
        for _ in range(50):
            t0 = time.perf_counter()
            r = fused(ind_b, W_j)
            jax.block_until_ready(r)
            ts.append(time.perf_counter() - t0)
        ts = np.array(ts) * 1e3  # ms
        print(f"  B={B:3d}: median={np.median(ts):.3f} ms  IQR=[{np.percentile(ts, 25):.3f},"
              f"{np.percentile(ts, 75):.3f}]  per-belief={np.median(ts) / B * 1e3:.1f} us")

    # ---- (b) WIRE: payload tradeoff ----
    feat_dim = 5 * N + 3 * nD + 6  # + n_tel; n_tel small, the design note cites ~241 -> we print the belief blocks' contribution
    belief_wire = kW64 * 8                       # the rank bitset on the wire
    feat_wire = 241 * 4                           # the current feature vector (~241 f32), per the design note
    print("\n## WIRE payload tradeoff (per leaf)")
    print(f"  belief rank-bitset: kW64={kW64} words x 8 = {belief_wire} B")
    print(f"  feature vector:     ~241 f32 x 4    = {feat_wire} B")
    print(f"  ratio belief/feature = {belief_wire / feat_wire:.2f}x "
          f"({belief_wire - feat_wire:+d} B per leaf)")
    # A tighter belief encoding: only the LIVE words are non-empty for small beliefs, but kW64 is fixed
    # (full rank space). A sparse rank-index encoding (varint ranks) is smaller only for very small nb.
    print(f"  note: a sparse rank-list encoding would beat the bitset only for nb << nworlds/64 "
          f"(~{nworlds // 64} ranks); the bitset is fixed-size {belief_wire} B regardless of nb.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
