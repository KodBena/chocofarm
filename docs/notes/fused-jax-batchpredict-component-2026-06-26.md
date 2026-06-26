# Fused-JAX BatchPredict component (lever #1) — CPU build + parity + round-trip (2026-06-26)

The working CPU component for de-risk idea #1 of `batchpredict-throughput-design-2026-06-26.md`: the
fused-JAX `BatchPredict` impl. Where the production inference wire ships FEATURES (the producer
featurizes, then sends the feature matrix), this path ships the raw BELIEF BATCH; the JAX side
featurizes (the `belief_indicator @ world_feature_matrix` matmul de-risked on `feat/tlab-batch-jax`'s
sibling) + runs the net, FUSED. This is the COMPONENT + parity + a CPU round-trip — NOT the production
server rewire (that integration + the GPU are maintainer-owned).

Frame: ADR-0000 (the per-world featurization loop is unrepresentable as a matmul; the wire's B↔kW64
swap is a compile error via the phantom split), ADR-0012 (the C++ `belief_features` is the cross-language
SSOT; the wire byte layout has one home, `belief_wire.hpp`, and every side derives it), ADR-0009 (measure,
don't assert).

## Artifacts (`throughput-lab/fused_jax/`)

- **`belief_wire.hpp`** — the belief-batch wire: a SETUP frame (the env-static world_feature_matrix
  column bitsets, sent ONCE), a per-batch REQUEST frame (B leaves: loc + collected + rank-bitset), a
  RESPONSE frame (value + logits, same shape as the feature wire's response). Mirrors `wire.hpp`'s
  Layer-1 discipline (a one-byte protocol-version tripwire, LE fields, loud boundary validation), with a
  DISTINCT version namespace so a belief frame fed to the feature decoder fails at the version byte.
  Reuses the `Quantity<Tag,Rep>` phantom machinery to split the B (row) and kW64 (word) count domains.
- **`belief_wire_test.cpp`** — standalone codec unit test: round-trips (setup/request/response,
  bit-exact) + the fail-loud boundary paths for ALL THREE decoders (setup/request/response: too-short,
  wrong version, ragged body) + the encode-side validators + an in-frame B=0 reject. ALL PASS.
- **`belief_batch_encode.cpp`** (`--FUSED_WITH_ENV=ON`) — the C++ SEND side: reads env, writes the
  actual belief-wire bytes (setup.bin, request.bin) + a JSON parity oracle (C++ `belief_features` in
  double). The matmul's env-static right operand rides the setup frame, not the per-leaf request.
- **`belief_response_decode.cpp`** — the C++ RECEIVE side that closes C++→JAX→C++.
- **`featurize_predict.py`** — the JAX side: a Python mirror of the belief-wire codec +
  `FusedBatchPredict.featurize_and_predict(request) → predictions`. featurize (matmul + the §2.2 phase-2
  maps) + a STAND-IN net (fixed pseudo-random dense weights — the point is the fused path, real weights
  + GPU later) in ONE jitted XLA program. Holds the world_feature_matrix resident as a device constant.
  Carries the `nworlds < 2^24` guard (fail-loud) that the bit-exact-`informative` argument is contingent on.
- **`roundtrip_demo.py`** — decode the C++ wire → featurize+predict → encode response (file hand-off) +
  the parity gate (f32 and f64) + the CPU timing + the wire restatement.

## (a) Parity — WITHIN TOLERANCE (a refactor cross-check, not an independent oracle)

Honest framing first (an independent review caught the overstatement): the JAX matmul and the C++ oracle
are **the same arithmetic in two arrangements, sharing their defining predicates**. The oracle's per-world
loop sums `(w>>t)&1` / `(w & mask_j)!=0` (`cpp/src/features.cpp`); the matmul contracts a `world_feature_matrix`
whose columns the env builds from the IDENTICAL predicates (`cpp/src/env.cpp` treasure_mask/detector_mask).
A matmul over a precomputed predicate bit-matrix IS definitionally that column-sum. So this is a
REFACTOR-AND-CROSS-CHECK (it catches transcription / indexing / normalization / float-width bugs, and it
re-confirmed the de-risk) — NOT an independent oracle: a shared error in the predicate DEFINITION would
pass parity silently. The C++ sweep is the cross-language SSOT (ADR-0012); the JAX path must agree with it,
and does.

Live env N=20, nD=44, nworlds=15504, kW64=243, B=64. JAX matmul featurization vs the C++ double oracle:

| block       | f32 max\|abs\| | f32 max\|rel\| | f64 max\|abs\| |
| ---         | ---            | ---            | ---            |
| marg        | 4.44e-08       | 8.24e-08       | 0              |
| p_pos       | 5.44e-08       | 1.00e-07       | 0              |
| informative | **0 (exact)**  | **0 (exact)**  | 0 (exact)      |
| marg_sum    | 9.54e-07       | 1.91e-07       | 0              |
| sharpness   | 7.27e-08       | 7.83e-08       | 1.11e-16       |

- **f32 worst 9.54e-7 ≪ the 1e-4 P6 bar** — within tolerance (identical to the de-risk numbers).
- **`informative` (the legal mask) is BIT-EXACT in f32** — the logic invariant ADR-0012 protects is not
  weakened (exact integer matmul counts; nworlds=15504 ≪ 2^24).
- **f64 reproduces the C++ double oracle** (worst 1.11e-16, one ULP in `log`) — the reframe is
  denotationally exact; an x64 provability fallback exists.

**Provability caveat:** moving featurization to JAX f32 forfeits the C++ bit-exact in-language oracle at
this boundary — marg/p_pos become a P6 *behavioral* bar (≤ ~1e-6), not bit-exact. `informative`/`available`
stay bit-exact *contingent on* nworlds < 2^24 (guarded, fail-loud); f64 recovers the exact oracle.

## (b) CPU round-trip — file hand-off (the honest CPU loop; socket is the gap)

C++ `belief-batch-encode` → {setup.bin, request.bin} → Python decode → `featurize_and_predict` →
`encode_response` → response.bin → C++ `belief-response-decode`. The C++ side decodes B=64 predictions;
`value[0] = -0.100199` matches the JAX output bit-for-bit. This exercises the SAME belief-wire bytes a
socket would carry, with a local FILE as the transport. **The GAP:** the Layer-2 ZMQ envelope + the live
server drain/scatter are maintainer-owned — this component proves the Layer-1 belief codec + the fused
featurize+predict, not the server rewire.

## (c) CPU timing — feasible; the full-path cost is attributed (with the remainder named)

Median of 50 reps after warmup, float32, `taskset -c 0`. The breakdown columns are measured ISOLATED
(unpack alone; transfer alone via `jnp.asarray`+block; jit with the indicator pre-resident):

| B   | full path | cpu_unpack | host→dev xfer | jit matmul+net | Σ of the three | remainder |
| --- | ---       | ---        | ---           | ---            | ---            | ---       |
| 8   | 1.26 ms   | 0.30 ms    | 0.13 ms       | 0.47 ms        | 0.90 ms        | 0.36 ms   |
| 32  | 3.28 ms   | 1.11 ms    | 0.25 ms       | 1.50 ms        | 2.86 ms        | 0.42 ms   |
| 64  | 6.54 ms   | 2.20 ms    | 0.63 ms       | 2.89 ms        | 5.71 ms        | 0.83 ms   |

- The **jitted featurize+net is ~2.89 ms at B=64**, matching the de-risk's bare-matmul 2.73 ms — the
  fused net adds negligible cost over the matmul (the stand-in net is one dense layer; the real net is
  what GPU-amortizes later).
- The breakdown columns are ISOLATED measurements; they do NOT exactly decompose the full path — a
  **remainder of ~0.8 ms at B=64** is per-call Python dispatch + output marshalling (`np.asarray` of the 7
  returned arrays, dict build, repeated `block_until_ready`) that the isolated timings do not incur. It is
  named here rather than buried (an independent review caught it being absorbed silently into the columns).
- The **full-path overhead beyond the kernel (~3.6 ms at B=64) is all host-side MARSHALLING** — the
  per-belief bit-twiddle unpack (densifying B×nworlds on the host), the host→device transfer of that dense
  ~4 MB indicator, and Python dispatch. This is the optimization the real build wants: ship the PACKED
  kW64 words to the device and unpack inside XLA (or vectorize the unpack), and use a persistent
  device-callable. It is NOT a featurization cost.

## Wire tradeoff (restated)

Per-leaf belief record = 4(loc) + 4(collected) + kW64·8 = **1952 B** vs the feature-vector wire 241·f32 =
**964 B** → **2.02×** (+988 B/leaf). The belief bitset is fixed 1944 B regardless of nb (full rank space).
The compute-for-bandwidth trade: pay ~2× the wire to move the O(nb·(N+nD)) featurization off the producer
and onto the (batched, GPU-amortizable) net side.

## GO / NO-GO

**GO** for the fused-JAX BatchPredict as a second impl behind the seam: parity holds (f32 ≪ 1e-4; legal
mask bit-exact; f64 fallback), the fused featurize+net is CPU-feasible (~3 ms at B=64), and the wire
codec round-trips bit-exactly. NOT a green light to wire production — this is the component + parity, not
the throughput A/B (the seam's head-to-head against impl #3 is what justifies the 2× wire).

## Open questions for the real server / GPU integration

1. **The marshalling cost.** Beyond the ~2.9 ms kernel, the rest of the full path (~3.6 ms at B=64) is
   host-side marshalling — the per-belief bitset unpack (densifying B×nworlds), the host→device transfer
   of that dense indicator, and per-call Python dispatch. Ship the PACKED kW64 words and unpack inside XLA
   (or vectorize the unpack); use a persistent device-callable. This is the dominant non-kernel cost.
2. **Layer-2 transport.** The file hand-off proves Layer-1; the real path needs the ZMQ DEALER↔ROUTER
   envelope (the corr-id frame) wrapping the belief request, and the server-side drain to recognize the
   belief protocol byte (vs the feature wire's).
3. **Setup-frame lifecycle.** The world_feature_matrix (124 KB here) is sent once; the server must hold
   it resident as a device constant per env and re-send on an env change. Where does that handshake live?
4. **Dense vs sparse matmul on GPU.** The dense nworlds×(N+nD) matmul is wasteful for narrow beliefs
   (the indicator is mostly zeros); a gather/segment-sum over live ranks may win on-device.
5. **The 2^24 guard.** The bit-exact-`informative` argument is instance-contingent; a larger nworlds must
   re-measure (or run the featurization in x64) — the guard fails loud, it does not silently degrade.

Public Domain (The Unlicense).
