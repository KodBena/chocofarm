<!--
docs/notes/leaf-eval-loop/firewall-zmq-ceiling-consult.md
Purpose: the VERBATIM record of an independent analytic-firewall consult (an Opus subagent) commissioned to
  adversarially verify-or-refute the claim that ~78% serve utilization is the best producer<->consumer
  integration achievable over ZMQ ipc. The firewall REFUTED the claim's mechanism and prescription (the
  consumer, not the wire, is the ceiling; true depth helps; pipelining the consumer is the dominant lever).
  Captured verbatim per ADR-0005 (point-in-time record) and the surface-the-artifact-verbatim discipline.
ADR-0006 header. Public Domain (The Unlicense).
-->

# Analytic firewall — "is 78% the ZMQ ceiling?" (consult, 2026-06-23)

The maintainer did not believe a conclusion drawn this session — that ~78% serve utilization (one big message
per forward) is the best producer↔consumer integration achievable over ZMQ ipc, that small-message pipelining
is counterproductive because the 113 µs/message wire cost dominates, and that the path forward is a cheaper
transport (in-process). An independent Opus subagent was commissioned as an **analytic firewall** — mandate:
adversarially verify or refute on the merits, test empirically (it has the runnable wire bench + perf), do not
default to confirming, do not manufacture a refutation. It verified every artifact itself. **Its verdict broke
the claim's mechanism and prescription.** Recorded verbatim.

---

> # VERDICT: The "78% is the ZMQ ceiling" claim is PARTLY SOUND but its MECHANISM and PRESCRIPTION are WRONG. The depth-monotonic evidence is a confound, and ~30% more throughput is reachable without changing the transport.
>
> I verified every artifact myself (reproduced the wire bench, wrote variant transports, computed utilization from the raw eventlogs, perf-profiled the wire, measured the consumer codec and JAX's GIL behavior). Numbers below are mine, not the summary's.
>
> ## What I confirmed (the claim's true parts)
>
> 1. **The 113us ipc wire floor reproduces and is real.** My re-run: P=1 intercept **115.2us** (CI 113.8–116.5), slope 0.293us/row, R²=0.996 — matches `step-5-wire-isolation.md` exactly.
> 2. **The perf attribution is correct.** With full kernel symbols, `ZMQbg/IO/0` (the ZMQ background IO thread) burns **44.8%** of sampled CPU, dominated by syscall entry/return (`syscall_return_via_sysret` 8.9%, `entry_SYSCALL_64_after_hwframe` 7.7%, `entry_SYSRETQ_unsafe_stack` 5.4%); the kernel datagram copy (`rep_movs_alternative`, `unix_stream_sendmsg`, `__skb_datagram_iter`) is present but only ~3%. The 113us is genuinely ZMQ-ipc's IO-thread + 2-syscalls-per-direction architecture, not a harness bug.
> 3. **inproc is the right escape and it works:** my inproc PAIR round-trip is **38us @B=128 vs 112us** for the same payload over ipc (C++ both ends) — a ~73us (65%) cut. So "in-process, T_io≈0" is directionally right. (But T_io is NOT 0: inproc still costs ~30–54us, scaling with B — the memcpy + cross-thread signaler.)
> 4. **The 78% util / 36117 leaf/s at depth=1 is real.** Computing Σ(FWD dt_us)/wall directly from the eventlogs: **78.3% util, 36214 leaf/s** at s_min=64. And it IS monotonic in the swept parameter (51.8 → 58.3 → 66.0 → 71.0 → 75.4 → 78.3%).
>
> ## Where the claim BREAKS
>
> **A. The depth sweep is confounded three ways, and the binding cost is the CONSUMER, not the wire.**
> The sweep never varied in-flight depth: every cell ran `--inflight-msgs 64`. It varied `S_min` (min-coalesce) 1→64, which the harness itself documents drives effective depth = `floor(W/S_min)`. So smaller S_min couples {deeper, smaller messages, MORE messages coalesced per forward} together. From the eventlogs, the real driver is **messages-per-drain** (the consumer's `SRV DRAIN msgs=` field): 30.4 at "depth=64" → 1.49 at "depth=1". Regressing the inter-forward **gap** against it:
>
> > **gap_us ≈ 540 + 50.5 × (msgs_per_drain),  R² = 0.9953**
>
> The wire RTT (~113us, pipelined behind the forward, identical in all cells) does NOT change. What changes is the single-threaded Python consumer doing `recv_multipart` + `decode_request` + `encode_response` + `send_multipart` at **~50us/message** on the critical path, ×30 messages. The claim's stated mechanism — "small-message pipelining is counterproductive [because] the 113us/message wire overhead dominates" — is wrong: the wire is hidden, and the cost is consumer per-message CPU.
>
> **B. TRUE depth is GOOD, not bad — I tested it directly.** Fixed message size B=128, varying in-flight depth over ipc (C++ echo): throughput **rose** 8154 → 15323 msgs/s (depth 1→8), then plateaued at the consumer's service rate. Latency rises (Little's law), but throughput improves with depth and plateaus at the consumer — the exact opposite of "depth monotonically worse." The lab's monotonicity is 100% the size/msgs confound.
>
> **C. The "irreducible 22% producer-search-wait" is over-attributed.** The depth=1 gap (~574us, tight: p10–p90 = 409–713us, only 3% >800us, so structural not bursty) decomposes as: consumer fixed overhead (codec+pyzmq, measured **110us even over inproc**) + wire transit both directions (~150us) + producer search (~250–300us). The producer-wait is roughly **half** the residual; the rest is consumer + wire, both addressable.
>
> **D. The Python consumer is the real ceiling, and it's leaving a lever on the table: JAX releases the GIL during the forward.** I measured a spin thread running at **96.5%** of its idle rate during the matmul. So a second consumer thread (or C++ consumer) can drain/decode/scatter forward K+1 *during* forward K's 2064us matmul — collapsing the serial gap. The current loop is strictly serial (`drain → forward → scatter → drain`), forfeiting this overlap.
>
> ## (2) The cleanest WHY — the quantities as equations
>
> Per-forward period and the closed throughput model (all verified, leaf_rate reconstruction matches measured to <1% across all 6 cells):
>
> ```
> T_fwd   ≈ 2064 us                         # matmul, constant (padmax→512 every forward); GIL-released
> T_gap   ≈ T_floor + c_msg · m             # m = messages coalesced per forward
>           c_msg ≈ 50 us/msg  (consumer recv+decode+encode+send, under the GIL)
>           T_floor ≈ 540 us = T_consumer_fixed(~110) + T_wire_rt(~150) + T_prod_search(~280)
> T_period = T_fwd + T_gap
> util     = T_fwd / T_period
> leaf_rate = real_per_fwd · 1e6 / T_period
> ```
> Wire (isolated): `RTT_ipc ≈ 113 + 0.29·B us` (Python consumer adds ~43us → ~155); `RTT_inproc ≈ 30 + 0.10·B us`. The sweep confound: `m ≈ W/S_min` (so the swept axis moved `m`, the `c_msg·m` term, NOT depth). Pipelining a serial consumer: `T_gap → max(T_prod_search − T_fwd, 0) ≈ 0` since search-step ≪ T_fwd, driving util→1.
>
> ## (3) The single most important thing to test next
>
> **Make the consumer pipelined (overlap drain/decode/scatter of K+1 with the matmul of K), exploiting the GIL release I measured.** Modeled gain: util 78%→~90–100%, leaf_rate 36k→**41k–46k (+16% to +30%)** — without touching the transport. This is a bigger, cheaper win than the inproc port the claim prescribes (inproc alone modeled at only **+7%**, because it cuts ~110us of wire from a 2604us period). The two compose; the consumer pipeline is the dominant lever. The decisive test: add a one-deep prefetch thread to the serve loop (or a C++ consumer) and measure util at s_min=64.
>
> **Caveats worth one fixer:** `decode_request`'s `np.all(np.isfinite())` is ~6us of the per-message 8us decode (×30 msgs at high depth) — a cheap separate win. And the "depth" labels in `results.txt`/run.log are misleading (the inflight cap was fixed at 64); the corpus should be re-labeled as an S_min/coalesce sweep, or a TRUE depth sweep (vary `--inflight-msgs`, hold message size) should be run to get the actual depth axis.
>
> ## Confidence
>
> **High** on the structural findings (all are direct measurements with R²>0.99 or perf evidence): the confound (S_min not depth; gap∝msgs/drain), true-depth-is-good, GIL-released-during-forward, inproc=−65%, C++-consumer=−43us, util/leaf_rate reproduced from raw eventlogs. **Medium** on the exact magnitude of the pipelined-consumer win (+16–30%) — it's a model extrapolation; the producer-search component (~280us) is inferred by subtraction, not directly instrumented, so if the real producer step is slower than ~T_fwd the ceiling lands lower.
>
> **What would change my verdict:** (a) an instrumented producer showing the search-step latency actually exceeds T_fwd (~2064us) at depth=1 — then the producer genuinely IS the ceiling and the claim's residual-attribution stands (I saw no evidence of this: gaps were tight at 574us). (b) A real pipelined-consumer prototype that fails to beat 78% — which would mean a hidden serialization (e.g., XLA dispatch or the params-source poll) the GIL micro-test missed. Both are runnable; the second is the one to run.

---

**Disposition (maintainer's call):** the firewall leaves a reusable `tools/zmq-wire-bench/variants.cpp` probe (Public-Domain-headed; the binary + /tmp scratch were cleaned). The actionable corrections it mandates: (1) re-label the depth-sweep corpus as an S_min/coalesce sweep (the inflight cap was fixed); (2) the **decisive next test** is a one-deep prefetch thread in the serve loop, measuring util at s_min=64; (3) `serialized_roundtrip_dps`'s framing (wire as a serialized cycle term) is sound in form but the wire is *hidden*, not on the critical path — the binding gap term is `c_msg·m`, the consumer's per-message cost.
