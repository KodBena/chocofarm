<!--
docs/design/stall-investigation/server-gen-floor-result.md
Purpose: the empirical result for the SERVER-side coalescing floor (increment ii) and its combination
  with the re-instated GENERATION-side batch floor (the runnable "final bolt"). Point-in-time record
  (ADR-0005 Rule 9). Refines — does NOT overturn — producer-floor-negative-result.md.
Public Domain (The Unlicense).
-->

# Server floor × generation floor — empirical result (2026-06-19)

Two levers were measured against the N=9 overcommit operating point (the leaf-eval transport; 3 producer
threads, 1:3 pin, pool_batch=64, the Gumbel-AZ search over `wire_ab_bench`):

1. **Server-side coalescing floor** (`inference_server.min_forward_rows` θ + `max_queue_delay_ms`) —
   accumulate ≥θ rows across producer threads before a forward (`server-floor-design.md`, increment ii).
2. **Generation-side batch floor** (the re-instated "final bolt": `wire_ab_bench --gen-chunk-floor`, with
   `--min-coalesce` S_min the generation minimum batch size) — chunk each producer message at S_min rows so
   the refill loop holds D messages outstanding (genuine in-flight depth > 1, overcommit on the wire).

## The headline

**The win is variance, not throughput, and it comes from the GENERATION floor — the server floor adds
nothing.** Best config: **gen-floor ON, θ_server=0 (no server floor), S_min=32, D∈{16,32}.**

| config (N=9) | dps mean | [min–max] | srv rows/fwd | wire rows/msg |
|---|---|---|---|---|
| greedy (gen off, θ=0) — the baseline | 177.2 | **[169–183]** | 195 | 165 |
| server floor alone (gen off, θ=128) | 175.9 | [174–179] | 206 | 165 |
| **gen floor, θ=0, S_min=32, D=32** | **183.7** | **[183–184]** | 108 | 28 |
| gen floor, θ=0, S_min=32, D=16 | 183.4 | [182–184] | 108 | 28 |
| gen floor, θ=0, S_min=64, D=8 | 182.4 | [182–183] | 101 | 50 |
| gen floor, θ=0, S_min=16, D=16 | 173.5 | [173–175] | 114 | 15 |
| gen floor, θ=0, S_min=16, D=8 | 156.0 | [155–157] | 85 | 15 |

(3 iters/config, the confirmation set; `runs/server_gen_floor_grid/refine-20260619-191306/`. The exploratory
24-config Sobol grid is `runs/server_gen_floor_grid/full-20260619-181303/`.)

- **It is a variance result, not a ceiling result.** Greedy swings **[169–183]** — the metastable
  collapse tail this whole investigation is about. The best gen-floor holds **[183–184]**, rock-steady at
  the *top* of greedy's range. The mean improves only **+3.7%** (183.7 vs 177.2) and its MIN (183) merely
  *ties* greedy's MAX (183) — so by the strict MIN-beats-MAX bar (ADR-0009) it does **not** raise the
  achievable ceiling. What it does is **remove the downside**: it never dips to greedy's 169 lows.
- **The mechanism is producer idle-time, not batch width.** The gen-floor wins with *smaller* forwards
  (108 vs 195 rows/fwd) and far smaller messages (wire rows/msg 28 vs 165). It is not "bigger server
  batch" (the server already reaches ~195 rows/fwd greedily). The depth>1 the chunk break creates keeps
  the otherwise depth-1 producer threads from blocking idle on a single outstanding reply.
- **Parameter shape.** S_min=16 under-coalesces (173, srv rows/fwd drops to ~114); **S_min=32 is the
  peak**; S_min=64 is within noise of it. **D must be ≥ 16** (S_min=16/D=8 collapses to 156). At S_min=32,
  D∈{8,16,32} are all ~182–184 (D-insensitive once ≥8 there). One run (S_min=64/D=32) returned an
  immediate RC1 (3 s) — a transient transport error, not a wedge; its D=8/16 siblings succeeded at ~182.
- **The server-side floor (θ>0) is neutral-to-harmful.** θ=128 ties greedy (175.9); combined with the
  gen-floor, raising θ_server *monotonically hurts* (the 24-config grid: θ≥384 → 123–148 dps), because the
  server floor's accumulation wait re-introduces the producer idle the gen-floor exists to remove. The
  prior server-floor-ALONE A/B (`runs/overcommit_sweep/ab-serverfloor-20260619-172343`) is the same
  finding at N∈{4,9}: θ=192 lifts rows/forward ~2.2× but lowers dps at every N and every delay (even 1 ms).

## Reconciliation with producer-floor-negative-result.md (refines, does not overturn)

`producer-floor-negative-result.md` found the chunk break **HARMFUL** (wedge, unbounded producer RSS,
~11 rows/forward). That verdict stands **for its regime**: it was measured at low overcommit with
`--min-coalesce 1` (1-row chunks), where the single-threaded server saw ~one tiny message at a time and
under-coalesced into the convoy. This result is a **different regime**:

- **N=9 overcommit.** Each thread owns K=198 slots, so chunking at S_min=32 issues ~6 concurrent chunks ×
  3 threads ≈ 18 messages the server *does* coalesce across (srv rows/fwd stays ~108, not ~11).
- **S_min ≥ 16, not 1.** The catastrophic 2.69M-message flood was specific to 1-row chunks; S_min=32
  keeps messages fat enough that the server forward stays in a healthy range.
- **RSS is bounded.** The harmful run's RSS growth was wedge-induced; here no config wedged (a 6 GiB
  producer-RSS kill-ceiling + 300 s timeout guarded the sweep), and the gen-floor configs ran ~1.3–1.75 GiB.

So the prior "producer-side moves the wrong way" holds where producer degree is the binding constraint and
the server cannot reassemble the chunks; in the overcommit regime the binding constraint is producer
*idle time*, which the chunk break (depth>1) reduces — the re-instatement is justified, and is a default-OFF
runnable option (`--gen-chunk-floor`), not an imposed behavior. The prior record is **not** retro-edited
(ADR-0005 Rule 8); this is the sibling result for the new regime.

## Disposition

- **Ship?** The throughput gain alone (+3.7% mean, ceiling-tied) does **not** justify shipping. The
  *variance* elimination ([169–183] → [183–184]) does, **iff** the metastable collapse is a real
  production cost (the investigation's premise: ~2–4% of windows, occasionally wedging). If so, the
  candidate is **gen-floor ON / S_min=32 / D=16 / θ_server=0**, behind the existing flags, with a
  longer-horizon RSS-monitored confirmation before a default flip. The server-side floor is **not** a
  production lever and stays default-OFF.
- The C++ `--gen-chunk-floor` option and the `inference_server` θ floor both stay **default-OFF**; the
  production path is byte-unchanged.
