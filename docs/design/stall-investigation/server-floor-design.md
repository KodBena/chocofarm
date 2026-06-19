<!--
docs/design/stall-investigation/server-floor-design.md
Purpose: DESIGN (for review, not yet implemented) of the server-side coalescing floor — increment (ii) of
  cpp-eval-transport-adapter.md §6. The lever the empirical work points at (producer-floor-negative-result.md).
  Public Domain (The Unlicense).
-->

# Server-side coalescing floor (increment ii) — design for review

**Status: DESIGN ONLY — do not implement until reviewed.** This is the lever the empirical work converged
on after the producer side was twice-refuted (`producer-floor-negative-result.md`).

> **2026-06-19 — IMPLEMENTED (amend-by-append, ADR-0005 Rule 8).** The drain floor below is now in
> `InferenceServer._drain` (`chocofarm/az/inference_server.py`), default OFF (`min_forward_rows=0`) so the
> production greedy drain is byte-unchanged; the two knobs (`min_forward_rows` θ, `max_queue_delay_ms`)
> are constructor params validated fail-loud at construction (θ>max_batch raises — ADR-0002/P2), read
> live per-drain on `self` (ADR-0012 P4). Bench CLIs plumbed: `overcommit_sweep.py` and `stage_a_server.py`
> (`--min-forward-rows` / `--max-queue-delay-ms`). Tests: an always-on construction-validation pin and an
> opt-in (`CHOCO_RUN_ZMQ=1`) parity-under-floor pin in `tests/test_zmq_inference.py`; a direct check
> confirmed 16 concurrent leaves coalesce into ONE forward (vs 3 greedy) with exact parity.
>
> **2026-06-19 — MEASURED (the A/B verdict; full result in `server-gen-floor-result.md`).** The server
> floor ALONE is a NEGATIVE result: θ=192 lifts rows/forward ~2.2× (N=4 98→215) but LOWERS dps at every N
> (N=4 141→119, N=9 179→161) and every delay swept (even 1 ms: 130 < greedy 141), because the depth-1
> producers idle on recv during the accumulation wait. Pairing it with a re-instated GENERATION-side batch
> floor (the runnable `--gen-chunk-floor`) recovers the loss and then some — but the gain is VARIANCE
> reduction, not throughput: the gen-floor (S_min=32, D≥16, θ_server=0) holds N=9 dps at [183–184] vs
> greedy's [169–183], a +3.7% mean that ties greedy's ceiling. **The server-side θ floor is not a
> production lever** (θ>0 neutral-to-harmful); the live lever is generation-side. Both stay default-OFF.

## The lever (and why it's the opposite of what failed)

The convoy is the server forwarding too-few rows per forward, paying its large **fixed per-forward cost**
(~the bulk of `dt`; ~0.5 µs/row marginal) on near-empty batches. The fix is to **accumulate** before
forwarding:

> After the first request arrives, keep draining until either **≥ θ rows** are gathered **or** a bounded
> **`max_queue_delay`** elapses; then run one forward.

This *increases* `rows/forward` (more amortization) — the opposite of the producer chunking, which *reduced*
per-message degree. It accumulates **across all producer threads**, so it directly controls the
throughput-relevant quantity (cross-thread `rows/forward`) that the producer side cannot touch.

## Where it goes (and how production stays untouched)

`InferenceServer._drain` (`chocofarm/az/inference_server.py`) — inherited by the bench `StageAServer`.
Add two params, **defaulting to OFF** so the production default path is byte-unchanged:

- `min_forward_rows` (θ), default **0** → disabled → current greedy drain (production untouched).
- `max_queue_delay_ms`, default **0**.

`_drain` becomes (sketch, θ>0 path):
```
poll(block) until ≥1 request                      # unchanged: the first-request wait
deadline = monotonic_ms() + max_queue_delay_ms
loop:
    drain all currently-queued (NOBLOCK) into `drained`, total_rows += ...
    if total_rows >= θ:            break          # enough — forward now
    if monotonic_ms() >= deadline: break          # timed out — forward what we have
    poll(deadline - now)                          # block briefly for the next arrival
forward(drained)                                   # unchanged
```
`_serve_batch`/`run_microbatch` are unchanged. The bench (`overcommit_sweep.py`) plumbs
`--min-forward-rows` / `--max-queue-delay-ms` for the A/B; the C++ side is untouched (this is purely a
server-drain change).

## Termination / no-wedge (the lesson from the producer side)

`max_queue_delay` is a **hard** bound: the forward always fires within `max_queue_delay_ms` of the first
request, whether or not θ is reached. So there is no "wait forever for θ rows that never come" wedge — the
delay is the escape hatch (the server-side analogue of the forced flush, but time-bounded rather than
state-bounded). With θ=0 (default) the loop degenerates to the current greedy drain immediately.

## Candidate θ and max_queue_delay (derived; to be swept)

From the measured `dt`-vs-width curve (warm forwards): leaves/s = `rows/dt` rises monotonically with batch
(width 64 → ~121k, 256 → ~200k, 512 → ~365k), i.e. *bigger is strictly better* well past the bench's top
bucket — the forward is fixed-cost-dominated. So θ wants to be **as high as is reachably useful**, bounded
by what the producers can supply: **θ ≤ T·K** (else the floor never reaches θ and always fires at the
delay).

- **Candidate θ ≈ 192** (the design's stated fast region; bucketed to 256). Reachable with margin at N≥4
  (`T·K = 3·N·22`: 264 at N=4, 594 at N=9). At N=4 this lifts effective `rows/forward` from ~108 (42%
  fill of the 256-bucket) toward ~192–256 — roughly 1.8–2.4×.
- **Candidate `max_queue_delay_ms` ≈ 2–5 ms** — a few × the forward σ (~1.4 ms), enough for a couple more
  threads' RTTs to land, small enough not to waste time when θ is unmet.
- Both **swept** in the A/B; θ ideally expressed as a fraction of `min(max_batch, T·K)` so it scales.

## Latency trade-off

The delay lengthens each producer's RTT (they are depth-1, blocked on recv during the wait). For a
**throughput-bound generator with no latency SLA**, this is a good trade: a few ms of added latency buys a
~2× fuller batch. It is *not* free in the strict sense (the depth-1 producer threads do idle on recv during
the wait), but the generator cares about leaves/s, which rises.

## Risks / what to watch (it is a tuning surface, not a closed invariant)

- θ too high (≥ T·K) → never reached → fires at the delay every time → just adds latency. Keep θ ≤ ~0.7·T·K.
- `max_queue_delay` too large → wasted waiting when θ is unmet (light load / phase-lock). Too small → no
  accumulation. Sweep.
- This does **not** make the convoy *unrepresentable* (the diagnosis's caveat) — a mis-set θ/delay can still
  under-coalesce. It is the working lever, not the closed one; the closed one is unavailable here.

## Open question the A/B must answer first

Is it worth shipping? **N=9 (the fast operating point) is already at ~197 rows/forward / 146 dps** — the
floor barely changes it. The wins are: (a) lifting **N=4** (~108 → ~192+ rows/forward), and (b) preventing
the **rare deep-collapse wedge** by flooring `rows/forward`. The A/B (θ=0 vs θ≈192, RSS-monitored) measures
both: does N=4 dps rise, does the collapse tail vanish, and is there any throughput cost at N=9?

## Plan (gated on this review)

1. Implement the θ / `max_queue_delay` drain in `inference_server._drain` (default OFF) + bench CLI.
2. A/B with the RSS-monitored harness: N=4 and N=9, θ ∈ {0, 192} (and a sweep), `max_queue_delay` ∈ {2,5} ms.
3. Decide on the production default from the measured dps / latency / wedge-rate.
