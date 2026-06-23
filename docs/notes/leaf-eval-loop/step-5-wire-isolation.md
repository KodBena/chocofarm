<!--
docs/notes/leaf-eval-loop/step-5-wire-isolation.md
Purpose: Step 5 of the leaf-eval impl->model loop. Isolates the WIRE (ZMQ round-trip) from the per-forward
  GAP that step 4 found scales with B, to settle whether that gap is the wire or the producer's search-wait.
  An isolated ZMQ benchmark (tools/zmq-wire-bench/) + perf profiling: the wire is 10-17% of the gap and the
  ~113us fixed cost is ZMQ ipc's background-IO-thread architecture; the gap is dominated by the producer-wait.
ADR-0005 point-in-time record; ADR-0006 header; claims-measured-vs-interpreted; robust statistics (multiple
  interleaved replicates, median+IQR, bootstrap-CI regression). Public Domain (The Unlicense).
-->

# Step 5 — isolating the wire: the gap is the producer-wait, not the wire (2026-06-23)

Step 4 found the per-forward **gap** (905-1864 us) scales with B and lumped "drain/decode/scatter/producer-wait"
into the model's `T_io`. The maintainer (rightly) flagged: is that the **wire** (ZMQ) or the **producer's
search-wait**? An isolated benchmark settles it — `tools/zmq-wire-bench/`: a C++ DEALER producer (P threads,
production-shaped `[corr][B*241 float]` messages) ↔ a minimal Python ROUTER **echo** consumer (pure wire — no
codec, no net), swept over B × P with **R=8 interleaved replicates** (warmup discarded; median + IQR; bootstrap-CI
regression — robust statistics).

## The wire RTT (robust regression `median_rtt ≈ a + b·B`)

| P | intercept a (µs) | slope b (µs/row) | R² |
| --- | --- | --- | --- |
| 1 | 113.1 (CI 111–115) | 0.295 (CI 0.284–0.303) | 0.994 |
| 2 | 103.2 | 0.396 | 0.977 |
| 3 | 72.0 | 0.895 (queuing — the single-thread consumer saturates) | 0.921 |

A **~113 µs fixed per-round-trip overhead** + ~0.29 µs/row. IQR 1–7 µs (very stable).

## The wire is 10-17% of the lab gap — the rest is the producer-wait

| B | wire RTT (P=1, µs) | lab gap (µs) | wire share |
| --- | --- | --- | --- |
| 128 | 154 | 905 | 17% |
| 256 | 184 | 1864 | 10% |

The echo wire is a **lower bound** on the lab's real drain/decode/scatter, and even so it is 10-17% of the gap.
**So ~85% of the per-forward gap is the producer's search-wait** — the server idling while the producer
generates the next batch (∝ B because more leaves = more Gumbel-AZ search). **Step-4's "`T_io` scales with B"
was the producer's SEARCH, mislabeled as the wire.** The model's `T_io` (the wire) is real but small (~113 µs)
and is *not* the dominant unmodeled cost.

## What the 113 µs wire IS — perf-attributed (not a harness bug)

perf (saturated B=32 cell, system-wide): **~4.5 context-switches/message** (409k cs / ~91k msgs in 5 s); the top
on-CPU symbols are **all the ZMQ background IO thread `ZMQbg/IO/0` in unix-socket syscall entry/return**
(`syscall_return_via_sysret` 7.6%, `entry_SYSCALL_64_after_hwframe` 6.3%, `entry_SYSRETQ` 4.7%) + the kernel
datagram copy (`rep_movs_alternative` / `_copy_to_iter` / `unix_stream_recvmsg` 3.4%). So the 113 µs is ZMQ
ipc's **background-IO-thread architecture**: each round-trip is app→IO-thread→socket→IO-thread→app = several
context switches + syscalls + a kernel copy. The hot symbols are the **ZMQ IO thread and the kernel — not the
Python consumer, not the producer loop** — so it is inherent to the transport (any language pays it), not a
benchmark artifact, and not tunable in the harness. An **in-process port** (no IO thread, no syscalls — the
model's `inproc_port_contrast`, `T_io≈0`) is the thing that removes it.

## Conclusion

- **The lab's per-forward gap is dominated (~85%) by the producer-server pipeline idle** (the server waiting on
  the producer's search; ∝ B). The wire (ZMQ) is 10-17% and ~constant-ish (a ~113 µs fixed + small per-row).
- **The model correction is sharpened**: the unmodeled B-scaling cost is the **producer-wait** (poor producer↔
  server overlap — msgs/forward≈1), not the wire `T_io`. The right model term is a *coordination/overlap* term
  (or the producer's `R_gen` ceiling), not a bigger `T_io`.
- The 113 µs ZMQ wire is a documented, perf-attributed production-serve-path cost (the IO-thread architecture),
  removable only by changing the transport (in-process).

Artifacts: `tools/zmq-wire-bench/` (producer.cpp, consumer.py, run-sweep.py); robust sweep + perf output in the
session record. Statistics per [[robust-benchmark-statistics]].
