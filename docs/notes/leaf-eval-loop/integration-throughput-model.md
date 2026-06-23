<!--
docs/notes/leaf-eval-loop/integration-throughput-model.md
Purpose: THE self-contained mathematical reference for the producer<->consumer integration throughput of the
  leaf-eval serving loop. Defines every quantity once, states the equations relating them, and proves (by
  direct measurement, R^2>0.99 / perf) the propositions about how they interact. Corrected against the
  analytic-firewall consult (firewall-zmq-ceiling-consult.md): the integration ceiling is the SERIAL CONSUMER,
  not the transport; the wire is hidden behind the forward; the dominant lever is pipelining the consumer.
  Supersedes the wire-on-the-critical-path framing of step-6-corrected-model.md (the binding gap term is the
  consumer per-message cost, not the wire).
ADR-0005; ADR-0006; ADR-0009 (every quantity substantiated); claims-measured-vs-interpreted (M vs I tagged).
Public Domain (The Unlicense).
-->

# The producer↔consumer integration throughput model (2026-06-23)

**Abstract.** The leaf-eval serving loop couples a 3-thread Gumbel-AZ search *producer* to a single-threaded
JAX inference *consumer* over ZMQ ipc. This note establishes the quantities governing its throughput and how
they interact, validated by direct measurement and corrected by an independent analytic firewall. The result:
the integration ceiling is set by the **serial consumer loop** (drain→forward→scatter), not by the transport;
the ZMQ wire (≈113 µs/round-trip) is **hidden behind the forward**, not on the critical path; and the dominant
lever is **pipelining the consumer** — overlapping forward K+1's drain/decode with forward K's matmul, which
JAX permits because it releases the GIL during the matmul — worth **+16–30%** throughput with the transport
untouched (vs +7% for an in-process port).

## 1. The system

The consumer serves in a **strictly serial loop**, per forward:
`DRAIN` (recv + decode the queued requests, coalescing `m` messages, up to `max_batch`) → `FORWARD` (one JAX
matmul, padded to 512 under padmax) → `SCATTER` (encode + send the replies). The next `DRAIN` cannot begin
until the current `SCATTER` completes. The producer's search **needs** each leaf's value to continue (the eval
is on the search's critical path), so producer and consumer are coupled, not free-running.

## 2. Quantities (each defined once; M = measured, I = inferred)

| symbol | meaning | unit | value | src |
| --- | --- | --- | --- | --- |
| `T_fwd` | the forward (matmul) wall time; constant under padmax (always 512 wide); **GIL-released** | µs | ≈ 2064 (M) | eventlog `dt_us` |
| `m` | messages coalesced per forward (`SRV DRAIN msgs=`) | — | 1.49 … 30.4 (M) | eventlog |
| `c_msg` | consumer per-message cost (`recv_multipart`+`decode`+`encode`+`send`, under the GIL) | µs/msg | ≈ 50 (M) | firewall regression |
| `T_consumer_fixed` | consumer per-forward overhead independent of `m` (codec + pyzmq setup) | µs | ≈ 110 (M) | measured even over inproc |
| `T_wire_rt` | round-trip wire transit charged into the gap (both directions) | µs | ≈ 150 (I) | decomposition |
| `T_prod_search` | producer search time for the next batch, charged into the gap | µs | ≈ 280 (I) | by subtraction |
| `T_floor` | `m`-independent gap floor `= T_consumer_fixed + T_wire_rt + T_prod_search` | µs | ≈ 540 (M) | regression intercept |
| `T_gap` | inter-forward gap (non-compute time) `= T_floor + c_msg·m` | µs | — | regression R²=0.995 |
| `T_period` | per-forward period `= T_fwd + T_gap` | µs | — | — |
| `RTT_ipc` | isolated ZMQ **ipc** round-trip (P=1 thread) | µs | ≈ 113 + 0.29·B (M) | wire bench |
| `RTT_inproc` | isolated **in-process** round-trip | µs | ≈ 30 + 0.10·B (M) | firewall variant |
| `B` (`real_per_fwd`) | useful rows (= leaves served) per forward | rows | ≈ 96 at `m`≈1.5 (M) | eventlog |
| `L` | **leaves per decision** — a *search* property, state/time-varying (~280–570, **not** the pinned 500) | leaves/dec | (M) | robust-L note |
| `util` | serve utilization `= T_fwd / T_period` | — | 0.783 at `m`≈1.5 (M) | eventlog |
| `leaf_rate` | useful leaves served per second `= B·1e6 / T_period` | leaves/s | 36214 at `m`≈1.5 (M) | eventlog |
| `dps` | decisions per second `= leaf_rate / L` | dec/s | — | — |

## 3. The model (the equations)

```
T_period  = T_fwd + T_gap
T_gap     = T_floor + c_msg · m
util      = T_fwd / (T_fwd + T_floor + c_msg · m)
leaf_rate = B · 1e6 / T_period
dps       = leaf_rate / L
```

**Validation (ADR-0009):** reconstructing `leaf_rate` from `B/T_period` matches the measured `fwds·B/secs`
within <1% across all six depth-sweep cells; the `T_gap = 540 + 50.5·m` regression has R²=0.9953.

## 4. Propositions (how the quantities interact — each substantiated)

- **P1 — the wire is hidden, not binding.** `RTT_ipc` (~113 µs) is identical across every depth-sweep cell, yet
  `T_gap` swings 615→2064 µs; the wire is overlapped behind `T_fwd`. The binding gap term is `c_msg·m`, not the
  wire. *(The firewall's gap-vs-`m` regression; the wire constant by construction.)* This **corrects step-6**,
  which charged the wire as a serialized cycle term — sound in form, wrong on the critical path.
- **P2 — the gap is the consumer's per-message CPU.** `T_gap` rises ~50 µs per coalesced message: the
  single-threaded Python consumer pays `recv+decode+encode+send` per message, ×`m`, serially. *(Regression
  slope `c_msg`≈50 µs/msg.)*
- **P3 — the "depth sweep" varied `m`, not depth.** `--inflight-msgs` was fixed at 64; the swept `S_min` drove
  `m ≈ W/S_min` (the coalescing granularity), so the monotonic-worse curve is the `c_msg·m` term, **a confound**,
  not a depth law. *(Direct read of the run configs + the `DRAIN msgs` field.)*
- **P4 — true depth HELPS.** Holding message size fixed and varying *in-flight depth*, throughput **rises**
  (8154→15323 msgs/s, depth 1→8) and plateaus at the consumer's service rate — the opposite of the confounded
  curve. *(Firewall's direct ipc test.)*
- **P5 — the consumer is the ceiling, and the GIL is free during the forward.** The loop is serial
  (`drain→forward→scatter→drain`), but JAX **releases the GIL** during the matmul (a spin thread runs at 96.5%
  of idle during `T_fwd`). So forward K+1's `DRAIN`/decode can run *during* forward K's matmul. *(Measured GIL
  occupancy.)*
- **P6 — the levers, ranked.** Pipelining the consumer collapses `T_gap → max(T_prod_search − T_fwd, 0) ≈ 0`
  (since the per-step search ≪ `T_fwd`), driving `util → 1` and `leaf_rate` **+16–30%** — *transport untouched*.
  An in-process port cuts only ~110 µs of wire from a ~2604 µs period → **+7%**. The two compose; the consumer
  pipeline dominates. *(Model extrapolation from the measured terms; magnitude is the one (I) result.)*

## 5. The corrected conclusion

**78% is a *serial-consumer* ceiling, not a ZMQ ceiling.** The transport (113 µs) is real but hidden; replacing
it buys little (+7%). The integration is bottlenecked by the consumer running `drain→forward→scatter` serially
while JAX leaves the GIL idle during the 2064 µs matmul. **Pipelining the consumer** (a one-deep prefetch of
K+1's drain/decode during K's forward) is the dominant, transport-free lever (+16–30%). This holds *regardless
of the leaf-production rate* because `util` and `leaf_rate` are rate-free; the search's `L` only converts
`leaf_rate` to `dps` and is the search's domain, out of the integration's scope.

## 6. Honest accounting (claims-measured-vs-interpreted)

- **High confidence (M, R²>0.99 / perf):** `T_fwd`, the `T_gap = 540 + 50.5·m` law, `util`/`leaf_rate` from raw
  eventlogs, the wire floor (115 µs) + its perf attribution, true-depth-helps, the GIL release, inproc = −65%.
- **Medium confidence (I):** the +16–30% pipelined-consumer gain is a *model extrapolation*; `T_prod_search`
  (~280 µs) is inferred by subtraction, not directly instrumented. If the real per-step search exceeds `T_fwd`,
  the producer genuinely binds and the ceiling lands lower (no evidence seen — depth=1 gaps were tight, 574 µs).
- **L is the search's, not ours:** `L` is state/time-varying (~280–570, the pinned 500 is mis-structured); the
  integration model targets `leaf_rate` (stationary), and `dps = leaf_rate / L` is reconstructed downstream.

## 7. The decisive next test

Add a **one-deep prefetch thread** to the serve loop (drain+decode forward K+1 while forward K matmuls), or a
C++ consumer, and measure `util` at `m`≈1.5. If `util` clears 78% toward ~1, the consumer was the ceiling
(P5/P6 confirmed). If it does **not**, a hidden serialization (XLA dispatch, the params-source poll) the GIL
micro-test missed is the real binder — the one falsifier to run. (Secondary, cheap: `decode_request`'s
`np.all(np.isfinite())` is ~6 µs of the 8 µs decode.)
