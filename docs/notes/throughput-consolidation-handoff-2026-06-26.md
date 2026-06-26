# Throughput consolidation — handoff + GPU-session springboard (2026-06-26)

Path: `docs/notes/throughput-consolidation-handoff-2026-06-26.md`
Purpose: hand off the AVX2 + #3 throughput consolidation — what was built, the seams created,
the lessons (devops / measurement / tooling standardization per ADR-0009/0011), and the e2e plan
for the VM→host-GPU boundary crossing. Springboard for the next (GPU) session.
Public Domain (The Unlicense).

This is a point-in-time record (ADR-0005 Rule 8): the live state is the git log + the branch.

---

## 1. What was done / accomplished

The target was `belief_features`, the ~55% producer hotspot. Four levers were pursued:

| lever | result (% faster = higher better) | bit-identity | status |
| --- | --- | --- | --- |
| **AVX2 popcount** | producer compute ~25% less time (≈34% faster), scalar→vpshufb | byte-identical | consolidated, gated |
| **#3 batched featurizer** | featurizer +27%; producer-compute +22–25% (mux); ~+13–15% est; **e2e +5% vs CPU JAX (measured)** | byte-identical | consolidated + integrated |
| **#1 fused-JAX** | featurization parity f64-exact / f32 ≤1e-6; 2.02× wire | f64 exact, f32 behavioral | **staged** (not consolidated) |
| **ZDD belief engine** | full search 10.5× slower (construction-bound) | n/a | killed |

`AVX2 + #3` stack to roughly **two-thirds of the original producer compute** (different bottlenecks —
see §4). All on `feat/tlab-throughput`; `main` untouched.

## 2. Integration points & seams created  (the load-bearing part)

The work's durable value is the seams, not the numbers:

- **`BatchPredict` seam** — `cpp/include/chocofarm/batch_predict.hpp`. `BatchLeaf{loc, belief, collected}`
  (the B leaves the cursor parks per RTT) → `B × prediction`. `BatchFeaturizer::featurize_batch(span<BatchLeaf>)
  → B rows`. The ONE abstraction both impls plug into: **#3 = in-process featurize** (no wire change);
  **#1 = fused-JAX** (ship beliefs, featurize there). #1 is a second impl behind this seam, A/B'd — not a rewrite.

- **SSOT-preserving featurizer split** — `FeatureBuilder::build_into` was split into the belief-memo lookup
  + a shared `assemble_into(loc, bf, collected, out)` (the ONE assembly body, **unchanged float order**).
  Per-leaf and batched paths share it → no Phase-2 fork, no parity drift.

- **Cursor deferred-featurize seam** — `TreeCursor::enable_deferred_featurize()` /
  `parked_loc()/parked_belief()/parked_collected()` / `resume_with_features(row64, pred)`, backed by
  `GumbelAZPolicy::eval_legal_from_features(...)` (the byte-identical legal-slots tail, minus the belief
  sweep the batch already did). The contract: **park WITHOUT building the row → batch-featurize all parked
  → resume each**. Proven bit-identical by `cpp/src/multiplexed_producer_compute_bench.cpp`.

- **Production integration** — `throughput-lab/cpp/real_producer.cpp` + the ACL `cursor_slot.hpp`. The
  `CursorSlot` forwards the cursor seam (no search logic): `install_batched_row(row)` stores `row64` +
  its `row32` narrow (the **float64-tail / float32-wire split** — the cursor's legal tail consumes f64,
  the wire sends f32), `resume_with_batched(pred)`. BOTH drivers (`drive_round` round-sync, `send_group`
  greedy) collect the parked triples → `featurize_batch` ONCE per RTT → install → resume. New flag
  **`--featurize batched (default) | per-leaf`** (also `CHOCO_PRODUCER_FEATURIZE`); per-leaf is the
  byte-unchanged A/B baseline. The wire/net path is byte-identical across modes. An **install-before-read
  poison net** (ADR-0000/0002): `ch.features` is EMPTY at park, both drivers reject an uninstalled batched
  row loudly — "ship an uninstalled row" is structurally visible, not order-dependent.

- **Belief-wire protocol (#1, staged)** — `throughput-lab/fused_jax/belief_wire.hpp` (SETUP frame =
  env-static `world_feature_matrix` sent once; REQUEST = B rank-bitsets; RESPONSE) + `featurize_predict.py`
  (the `belief_indicator @ world_feature_matrix` matmul featurization + net, one jitted XLA program). The
  belief→JAX contract, parity-proven, ready when #1 is revisited.

- **AVX2 popcount, gated** — `belief_bitset_ops.hpp`: `#ifdef __AVX2__` (vpshufb nibble-LUT kernel) `#else`
  scalar `std::popcount`. Byte-identical fallback; non-AVX2 hardware compiles + runs (no SIGILL).

## 3. The e2e / GPU plan — and the "why not the existing CPU JAX server?" clarification

**You are right, and the in-session "pends the GPU" framing was imprecise.** The existing CPU JAX server
(`throughput-lab/server/server.py --forward jax`, XLA-jit MLP, bucket ladder) IS the e2e vehicle, and #3
ships feature vectors **unchanged**, so it runs against that server as-is — no GPU, no code change. The GPU
step is exactly as you envision: **copy the server, set the JAX default device to GPU, swap IPC→TCP ZMQ,
repoint URIs.**

Why numpy was used in-session: the agents ran `--forward numpy` for **smoke** (correctness — "does the
multiplexed park-collect-featurize_batch-resume loop run e2e"), where a trivial net is the right stand-in
(no XLA compile, no weights). That was appropriate for smoke; it is **not** a throughput vehicle. The
throughput number was then deferred to "the GPU" — which conflated two things: measuring #3 (doesn't need
the GPU) vs. maximizing #3's realized win (does benefit from a cheaper net). #1 is the only lever that
*needs* a server change (it ships beliefs, not features → the server must featurize); that's why #1 has its
own `featurize_predict.py`. #1 is staged, so it doesn't bear on the #3 e2e.

**Measured today (CPU JAX server, no GPU, `--net ""` random weights, core-0 server / core-3 producer):**

| featurize | dec/s (rep1, rep2) | any_fail |
| --- | --- | --- |
| batched (#3) | 134.1, 134.1 | 0 |
| per-leaf (baseline) | 128.7, 126.2 | 0 |

≈ **+5% e2e throughput** for #3. Coarse (2× 5 s runs, not the full interleaved-bootstrap protocol) but
directionally clear + consistent. This is BELOW the ~+13–15% pure-compute estimate → the run is **partially
server-bound** at this net/RTT balance (the CPU JAX forward + RTT consumes a slice of the cycle the
generator-compute saving can't reclaim). Interpretation marked provisional (measured = the +5%; the
boundedness reading is the conjecture it motivates).

**So the GPU's role is precise:** it makes the net forward cheaper → the run shifts toward generator-bound →
#3's realized e2e win widens from +5% toward the +13–15% compute ceiling. The GPU is not needed to *witness*
#3; it is needed to *expose* the generator as the bottleneck.

### The boundedness / linear-latency study (the next-session task)
The +5% (CPU JAX) vs +13–15% (pure compute) gap **quantifies current server-boundedness**. The plan:
1. Sample pure VM↔host ZMQ latency (TCP loopback vs cross-boundary RTT).
2. Inject the measured/simulated added latency and re-run the cycle under pure-VM.
3. Test whether cycle throughput falls **linearly** in the added latency (the server-bound prediction) or is
   absorbed (the generator-bound prediction) — and watch batched-vs-per-leaf at each latency point: the gap
   IS #3's realized win as a function of where on the bound the cycle sits.

Harness (verified working this session):
```
# server (CPU JAX). IPC now; TCP for VM↔host: --bind tcp://0.0.0.0:PORT
PYTHONPATH=throughput-lab taskset -c 0 python -m server --forward jax --net "" \
    --in-dim 241 --n-actions 65 --bind ipc:///tmp/tlab-infer.sock
# producer (core 3). --featurize batched|per-leaf; --endpoint tcp://HOST:PORT for cross-boundary
CHOCO_FEATURE_LAYOUT=chocofarm/data/feature_layout.json taskset -c 3 \
    throughput-lab/cpp/build/tlab-real-producer --instance chocofarm/data/instance.json \
    --faces chocofarm/data/faces.json --endpoint ipc:///tmp/tlab-infer.sock \
    --seconds S --fibers K --episodic --featurize batched --recv-timeout-ms 30000
```

## 4. Lessons — devops / operational efficiency / tooling / standardization

- **ALWAYS run CPU stats after a discriminating run** (perf TopdownL1 + cache counters + cycles/instructions).
  AVX2 and #3 are BOTH ~−22–25% cycles but by OPPOSITE mechanisms — only perf made it legible:
  AVX2 = **instruction-count collapse** (−39% instructions, IPC down, heavier ops); #3 = **memory-traffic
  collapse** (L1 −40%, L2 −48%, L1i −59%, dTLB −54%, IPC up). They compose *because* the bottlenecks differ.
- **Measure the bound, don't assume it.** "Pends the GPU" was an unmeasured assumption; one CPU-JAX run
  showed +5%. A denotational/model read MOTIVATES; only an operational run WITNESSES (the session's
  recurring discipline; cf. the AVX2 opacity in §the-avx2-RCA below).
- **Boundedness is the lens for any compute win:** it shows in throughput only when generator-bound. A smoke
  (numpy) is server-bound by construction → witnesses correctness, never the win. Pick the net for the
  question: trivial (numpy) for "does it run", real (JAX) for "how fast", GPU for "expose the generator".
- **Harness hygiene — the traps that cost time this session (standardize against them):**
  - **zsh scalar no-word-split:** `cmd $VAR` where `VAR="--a x --b y"` passes ONE arg → programs print
    `usage:` (misread as failure ≥2×). Use `${=VAR}`, an array, or explicit args.
  - **`--bind` must equal `--endpoint`** (dropped `--bind` once → server on default socket → recv timeout →
    `any_fail=1`). 
  - **server/producer dim handshake:** `--n-actions` MUST match `n_slots` (=65) and `--in-dim` the feature
    width (=241), else `eval_finish` reads OOB → **SIGSEGV**. The tell it's NOT a code bug: the per-leaf
    baseline crashes *identically* (gdb backtrace in `eval_finish` reading `NetPrediction` = shape mismatch).
  - **build-flag gates:** `tlab-real-producer` needs `-DTLAB_REAL_GENERATOR=ON` (OFF by default). "Tree
    builds green" must name WHICH tree + WHICH flags — my first "tlab green" silently omitted the real producer.
  - **JAX server warmup:** pre-compiles a bucket ladder `[1,8,64,512,4096]`; give a generous
    `--recv-timeout-ms` (30000) for first-shape jitter.
  - **`CHOCO_FEATURE_LAYOUT`** (or run from repo root) for the real producer's `FeatureBuilder`.
- **Verify the artifact, not the claim** (ADR-0013): independent reruns caught (a) a real base-build break
  the agents' "tree green" had masked, and (b) my own misconfigs. Both the green-exit AND my first crash
  were misleading until reproduced.
- **A typed-SSOT change must sweep ALL trees.** Phantom-typing `GumbelConfig` (`aa63507`) broke downstream
  call sites in `throughput-lab/cpp` (`real_producer`, `real_gen_smoke`) that the `cpp/`-only retrofit
  missed — the mandate branch still carries that break; it is repaired only here, in the consolidation.
- **Branch topology discipline:** the perf branches forked off a *pre-retrofit* base, so consolidation had
  to fold the retrofit back in (one `gumbel_dump.cpp` conflict, resolved to the canonical `::rep_type`).
  Fork perf work off the latest mandate HEAD, not an older one.

### The AVX2 RCA (ADR-0000, recorded for the registry)
The AVX2 win was a +74% lever sitting at the #1 hotspot, invisible to us. RCA: the compiler genuinely
*could not* find it (the box is AVX2-only — no AVX-512 VPOPCNTDQ; `std::popcount`→scalar `POPCNT` is its
ceiling; the vpshufb-LUT popcount is a hand-coded algorithm no auto-vectorizer synthesizes). The CODE was
not mis-architected (`std::popcount` is correct). The **opacity to us** was the mechanizable lapse: a
hand-vectorizable hot kernel (popcount over a bitset — a textbook vpshufb case) with **no primitive-level
A/B bench**, and a "no lever" verdict inferred from *reading* the disassembly rather than *measuring* the
alternative. Mechanization: hot kernels get a primitive A/B bench (now exist); a "no lever" verdict requires
the witness, not the asm-read. Not a type fix — a measurement-discipline fix.

## 5. Branch map / state

- **`feat/tlab-throughput` @ `ef49818`** — THE consolidation: mandate + DMZ lint + full retrofit +
  AVX2(gated) + #3 (seam + featurizer + cursor seam + production integration). All gates green
  (oracle, cursor-proto, gumbel_logic, gumbel_precision 144/144, mux + batch bit-identity); full `cpp/`
  and `throughput-lab/cpp` (incl. `tlab-real-producer`) build green; e2e clean. **Merge-ready.**
- `feat/tlab-fused-jax-impl` @ `78619ad` — #1 staged (revisit post-GPU with the wide-batch A/B vs #3; the
  2.02× wire pays only if the offload + GPU beats in-process #3).
- `feat/tlab-avx2-popcount`, `-seam-insrc`, `-seam-integ` — the staged pieces, now folded into `-throughput`.
- `feat/tlab-batch-insrc`, `-batch-jax` — the de-risk prototypes (kept for the record).

## 6. Next session (GPU) — springboard checklist

Start from `feat/tlab-throughput`.
1. Copy `server/server.py`; set the JAX default device to the RTX 2080Ti; IPC→TCP ZMQ; repoint URIs.
2. Sample pure VM↔host ZMQ latency (TCP loopback vs cross-boundary).
3. Run the boundedness / linear-latency study (§3) — batched vs per-leaf at each latency point.
4. Expectation to TEST (not assume): #3's e2e win grows from ~+5% (CPU JAX, partially server-bound) toward
   ~+13–15% (compute ceiling) as the GPU cheapens the net; added VM↔host latency pushes it back toward
   server-bound. Whether throughput falls *linearly* in added latency is the open question this study answers.
