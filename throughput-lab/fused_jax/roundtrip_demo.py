#!/usr/bin/env python
# throughput-lab/fused_jax/roundtrip_demo.py
# Purpose: the CPU ROUND-TRIP + PARITY + TIMING driver for the fused-JAX BatchPredict component (lever
#   #1, docs/notes/batchpredict-throughput-design-2026-06-26.md). It closes the loop:
#
#     C++ belief-batch-encode  ->  {setup.bin, request.bin, oracle.json}     (the SEND side)
#       |                                                                     (FILE hand-off: a local,
#       v                                                                      in-host transport — the
#     this driver: decode the belief wire -> FusedBatchPredict.featurize_and_predict -> encode_response
#       |                                                                      socket round-trip needs the
#       v                                                                      server infra; see the GAP)
#     C++ belief-response-decode  <-  response.bin                            (the RECEIVE side closes it)
#
#   And it gates:
#     (a) PARITY — the JAX matmul featurization vs the C++ chocofarm::belief_features double oracle, in
#         f32 AND f64, per feature block (max abs / max rel). The legal-mask `informative` predicate MUST
#         be bit-exact (0.0). Re-confirms the de-risk on THIS base.
#     (b) TIMING — the CPU fused featurize+predict for B in {8, 32, 64} (median over reps after warmup).
#     (c) WIRE — the per-leaf belief payload vs the feature-vector payload (the 2x trade restated).
#
#   The FILE hand-off is the honest CPU round-trip: it exercises the SAME belief-wire bytes a socket
#   would carry (the C++ encoder writes them, the C++ decoder reads them back), with a local file as the
#   transport. The Layer-2 ZMQ envelope + the real server drain are maintainer-owned (the GAP, named in
#   the report) — this component proves the Layer-1 belief codec + the fused featurize+predict, not the
#   server rewire.
#
#   Run (the C++ side must have produced the .bin/.json first; the demo can also drive it via --encode):
#     /home/bork/w/vdc/venvs/generic/bin/python roundtrip_demo.py \
#         --setup /tmp/setup.bin --request /tmp/request.bin --oracle /tmp/oracle.json \
#         --response /tmp/response.bin
#
# Public Domain (The Unlicense).
import argparse
import json
import sys
import time

import numpy as np

import featurize_predict as fp


def parity(setup, request, oracle, dtype):
    """Featurize via JAX at `dtype`, diff each block vs the C++ double oracle. Returns (worst_abs,
    info_exact, per-block dict)."""
    bp = fp.FusedBatchPredict(setup, n_actions=8, dtype=dtype)
    out = bp.featurize_and_predict(request)
    refs = oracle["leaves"]
    names = ["marg", "p_pos", "informative", "marg_sum", "sharpness"]
    perblock = {}
    worst_abs = 0.0
    for nm in names:
        j = np.asarray(out[nm], dtype=np.float64)
        if j.ndim == 1:
            r = np.array([ref[nm] for ref in refs], dtype=np.float64)
        else:
            r = np.array([ref[nm] for ref in refs], dtype=np.float64)
        absd = np.abs(j - r)
        denom = np.maximum(np.abs(r), 1e-30)
        reld = absd / denom
        perblock[nm] = (float(absd.max()), float(reld.max()))
        worst_abs = max(worst_abs, float(absd.max()))
    info_exact = perblock["informative"][0] == 0.0
    return worst_abs, info_exact, perblock, out, bp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--setup", required=True)
    ap.add_argument("--request", required=True)
    ap.add_argument("--oracle", required=True)
    ap.add_argument("--response", required=True, help="where to write the response.bin for the C++ decoder")
    args = ap.parse_args()

    setup_bytes = open(args.setup, "rb").read()
    request_bytes = open(args.request, "rb").read()
    oracle = json.load(open(args.oracle))

    setup = fp.decode_setup(setup_bytes)
    request = fp.decode_request(request_bytes)
    print(f"# decoded setup: N={setup['N']} nD={setup['nD']} nworlds={setup['nworlds']} kW64={setup['kW64']}")
    print(f"# decoded request: B={request['B']} kW64={request['kW64']}")

    # Net the decoded belief nb (popcount of each rank-bitset) against the oracle nb — the wire carried
    # the right bitsets. np.unpackbits over the u8 view counts the live ranks per leaf.
    bits = np.unpackbits(request["belief"].view(np.uint8), axis=1)
    nb_wire = bits.sum(axis=1).astype(np.int64)
    nb_oracle = np.array([oracle["leaves"][i]["nb"] for i in range(request["B"])])
    assert np.array_equal(nb_wire, nb_oracle), f"wire nb != oracle nb: {nb_wire} vs {nb_oracle}"
    print(f"# wire-decoded belief popcounts == oracle nb (OK): nb = {nb_oracle.tolist()}")

    # ---- (a) PARITY: f32 and f64 ----
    print("\n## PARITY (JAX matmul featurization vs C++ belief_features double oracle)")
    for dtype in ("float32", "float64"):
        if dtype == "float64":
            import jax
            jax.config.update("jax_enable_x64", True)
        worst_abs, info_exact, perblock, _out, _bp = parity(setup, request, oracle, dtype)
        print(f"  [{dtype}] per-block max|abs| / max|rel|:")
        for nm, (a, rl) in perblock.items():
            print(f"    {nm:12s} max|abs|={a:.3e}  max|rel|={rl:.3e}")
        print(f"    -> informative bit-exact: {'YES' if info_exact else 'NO (DIVERGES)'}")
        print(f"    -> worst abs across blocks: {worst_abs:.3e}")
        if dtype == "float64":
            jax.config.update("jax_enable_x64", False)

    # ---- the ROUND-TRIP: featurize+predict at f32 (the wire dtype) -> encode response -> write file ----
    bp = fp.FusedBatchPredict(setup, n_actions=8, dtype="float32")
    out = bp.featurize_and_predict(request)
    resp = fp.encode_response(out["value"], out["logits"], bp.n_actions)
    open(args.response, "wb").write(resp)
    print(f"\n## ROUND-TRIP: wrote {len(resp)} B response ({request['B']} preds, "
          f"n_actions={bp.n_actions}) -> {args.response}")
    print(f"   (the C++ belief-response-decode reads this back to close C++->JAX->C++)")
    print(f"   sample value[0]={out['value'][0]:.6f}  logits[0]={np.asarray(out['logits'][0])}")

    # ---- (b) TIMING: the full path AND its FULLY-ATTRIBUTED breakdown (no unexplained remainder).
    # The full path is cpu-unpack (densify B x nworlds on the host) + host->device transfer + the jitted
    # matmul+net. We measure all three so the columns SUM to the full path (no hidden cost).
    import jax
    import jax.numpy as jnp
    print("\n## TIMING (CPU fused featurize+predict, float32) — median of 50 reps after warmup")
    print("   columns: full = cpu_unpack + host->device_xfer + jit(matmul+net); they SUM to full.")
    base = request["belief"]

    def _med(fn, warm=5, reps=50):
        for _ in range(warm):
            fn()
        ts = [(_t0 := time.perf_counter(), fn(), time.perf_counter() - _t0)[2] for _ in range(reps)]
        return float(np.median(np.array(ts) * 1e3))

    for B in (8, 32, 64):
        idx = np.arange(B) % base.shape[0]
        sub = dict(B=B, kW64=request["kW64"], loc=request["loc"][idx],
                   collected=request["collected"][idx], belief=base[idx])
        bw = sub["belief"]

        full = _med(lambda: bp.featurize_and_predict(sub))
        # cpu_unpack: the host densify alone.
        unpack = _med(lambda: fp.unpack_beliefs_to_dense(bw, bp.nworlds))
        # host->device transfer: jnp.asarray of a freshly-unpacked dense indicator, blocked.
        dense = fp.unpack_beliefs_to_dense(bw, bp.nworlds).astype(np.float32)
        xfer = _med(lambda: jax.block_until_ready(jnp.asarray(dense, dtype=bp._dt)))
        # jit(matmul+net): with the indicator already resident on device.
        ind_dev = jnp.asarray(dense, dtype=bp._dt)
        jit = _med(lambda: jax.block_until_ready(bp._fused(ind_dev, bp._W)))

        print(f"  B={B:3d}: full={full:6.3f} ms  (cpu_unpack={unpack:.3f} + xfer={xfer:.3f} + "
              f"jit={jit:.3f} = {unpack + xfer + jit:.3f})  per-leaf={full / B * 1e3:.1f} us")

    # ---- (c) WIRE: the per-leaf payload trade ----
    kW64 = setup["kW64"]
    belief_leaf = 4 + 4 + kW64 * 8        # loc + collected + the rank bitset (the belief-wire record)
    feat_leaf = 241 * 4                    # the feature-vector wire (~241 f32), per wire.hpp STAGE_A_IN_DIM
    print("\n## WIRE payload tradeoff (per leaf)")
    print(f"  belief-wire record: 4(loc)+4(collected)+kW64*8 = {belief_leaf} B")
    print(f"  feature-vector wire: 241 f32 x 4               = {feat_leaf} B")
    print(f"  ratio belief/feature = {belief_leaf / feat_leaf:.2f}x ({belief_leaf - feat_leaf:+d} B/leaf)")
    print(f"  note: the belief bitset is FIXED kW64*8={kW64 * 8} B regardless of nb (spans the full rank "
          f"space); the loc+collected add 8 B. Pay ~2x wire to move featurization off the producer.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
