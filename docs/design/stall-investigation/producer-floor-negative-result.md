<!--
docs/design/stall-investigation/producer-floor-negative-result.md
Purpose: the empirical NEGATIVE RESULT for the producer-side coalescing-floor fix — why it is inert at
  depth-1 and harmful when made to bind, and why the bottleneck is server-side. Point-in-time record
  (ADR-0005). Branch fix/producer-coalescing-floor; commits 89d6984 (attempt) -> e6d2c41 (revert).
Public Domain (The Unlicense).
-->

# Producer-side coalescing floor — empirical negative result

We tried to defeat the cross-thread coalescing-collapse convoy (`cpp-eval-wire-formal-diagnosis.md`)
from the **producer** side: enforce a minimum coalescing degree `S_min` so under-coalescing is
unrepresentable (the "closed" fix the diagnosis preferred over the server-side tuning surface). It does
not work in this architecture. This records why, so the dead-end is not re-walked.

## What was tried, and the verdict

| attempt | mechanism | benchmark verdict |
|---|---|---|
| **1. `S_min` floor** | `issue()` refuses a sub-threshold (`< S_min`) gather; forced-flush only at `inflight==0` | **INERT.** `--min-coalesce 1` vs `32` gave *identical* producer-B and dps. |
| **2. floor + depth>1 chunk** | also `break` the gather at `S_min` rows (S_min-row chunks) so the refill loop stacks D messages, making the floor bind | **HARMFUL.** Wedged; producer RSS grew unbounded. |

## Why attempt 1 is inert (forced-flush bypass at depth-1)

The blind model proved (SYNTHESIS §0) that `issue()` draining **all** ready slots into one message makes
per-thread in-flight depth **identically 1**. At depth-1, after every reply `inflight_msgs == 0`, so a
refill that finds `< S_min` ready hits the forced flush (which must exist for termination) and sends the
sub-threshold message anyway. The floor's *hold* path is never taken. Measured: with `--min-coalesce 32`,
producer B still reached **min 1** in ~12% of messages; the B distribution was byte-identical to
`--min-coalesce 1`.

## Why attempt 2 is harmful (the floor became a cap, and fed the server's weakness)

Making the floor bind required depth>1, achieved by `break`-ing the gather at `S_min` (S_min-row chunks).
But that **caps** per-message degree at `S_min` — *below* the producer's natural drain-all coalescing
(~74/88). So it shattered fat messages into small ones and bet the single-threaded server would reassemble
them — but the server's under-coalescing of small messages is the *very bug*. At `--min-coalesce 1`
(1-row chunks) it was catastrophic:

- **2.69M** producer messages vs **245k** server forwards → **~11 rows/forward** (deep convoy).
- The run **wedged past the subprocess timeout** (218 s) and the **producer RSS grew throughout**.

The RSS growth was **wedge-induced**, not a leak: the wire `inflight_` map is bounded at D (SUBMIT≈RECV,
balance ≤ D), and the gumbel arena is released per decision (`gumbel.cpp:571`). Reverting the chunk break
restored a **flat ~6.4 MB RSS, 114 dps, completes in 83 s** (`runs/revert-sanity-n4`).

## The conclusion (why producer-side cannot be the lever here)

The producer's per-message degree was **never** the bottleneck — at drain-all it is already healthy
(~74/88). Throughput is gated by the **server amortizing its fixed per-forward cost** over `rows/forward`.
That fixed cost is ~the bulk of the forward, not the matmul: measured `dt` is **64 rows → ~530 µs,
512 rows → ~1400 µs** (8× the rows, ~2.6× the time → fixed-overhead-dominated; the marginal is ~0.5 µs/row,
the XLA-dispatch/host-device/ZMQ framing is the fixed part). So leaves/s = `rows/forward / dt` rises with
batch (121k → 365k from width 64 → 512), and the convoy is catastrophic because it pays the full fixed cost
for ~1 row.

Anything producer-side that *reduces* per-message coalescing (chunking) moves the wrong way. The only lever
that raises `rows/forward` is **server-side accumulation** (increment ii: `preferred_batch_size` +
`max_queue_delay`) — see `server-floor-design.md`.

## Disposition

The `S_min` floor scaffolding and the `--min-coalesce` CLI are **retained** (inert at depth-1, `S_min=1`
reproduces drain-all) on branch `fix/producer-coalescing-floor` as the record; the harmful chunk break is
reverted (`e6d2c41`). Do not re-attempt a producer-side floor without first changing the depth-1 +
drain-all architecture *and* establishing that producer degree (not server `rows/forward`) is the binding
constraint — neither holds today.
