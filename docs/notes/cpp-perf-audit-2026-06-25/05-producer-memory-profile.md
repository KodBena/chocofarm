# Producer memory profile — where the bytes actually live, 2026-06-25

**Stamp:** commit `17f1fbe` (feat/tlab-real-generators), CLEAN. Real producer (`tlab-real-producer`,
fiber-per-tree) against a single-thread JAX server, `nice -n -19 taskset -c 0`, msg_rows=256
inflight=8 n_sims=256 m=24 episodic, `CHOCO_BELIEF_CACHE_CAP=16` (post-OOM-fix default).

## Methodological finding first: heaptrack is BLIND here

heaptrack on a 768-decision / 424k-leaf run captured **371 allocations total**. Not a bug — the
producer's big memory is **mmap-based by design** (the `releasing_arena`/`MmapUpstream` node arena
and the `protected_fixedsize_stack` fiber stacks — the OOM-fix machinery), which **bypasses
malloc**, so malloc-interception sees almost nothing. The correct tool for mmap-resident memory is
**`/proc/<pid>/smaps`** (used below). (heaptrack would over-attribute to the few malloc-backed
things — e.g. the belief memo — and miss the arena entirely; keep this in mind reading the prior
`mem-structural-fix` heaptrack RCA.)

## Where the bytes live — K=1024 production config, at PEAK (smaps)

Peak resident **1313.7 MiB** (the run oscillates — see below; this is a near-peak snapshot):

| component | resident | share | note |
| --- | --- | --- | --- |
| **node arena** (`_Node` search-tree graphs, all parked trees) | **~1290 MiB** | **98%** | multiple mmap pool regions (334+220+151+143+80+70+65+… MiB); intrinsic to n_sims × concurrent trees |
| **fiber stacks** (1024 × `protected_fixedsize_stack` 512 KiB) | **16.8 MiB** | 1.3% | 512 MiB *virtual* reserved, demand-paged to **~17 KiB each** |
| libraries (libstdc++, libc, libm, libzmq) | ~5 MiB | 0.4% | — |

**The node arena IS the producer's memory** (98%). The fiber stacks — an Option-A-specific cost —
are negligible (16.8 MiB), fully demand-paged.

## RSS oscillates with the episode cycle (sawtooth)

K=1024 RSS over time (rollup): 1281 → 1012 → 658 → 305 → **1317** MiB. The arena fills across an
episode and **releases at episode boundaries** (`releasing_arena` munmaps → RSS drops), then
refills. Peak ≈ **1.3 GiB/producer**; trough ≈ 0.3 GiB. At 4 producers, if peaks coincide,
≈ 5.2 GiB — vs ~8 GiB MemAvailable. The OOM exposure is this **arena peak**, intrinsic to the
search budget; the belief-memo bound (finding #27) addressed a *different*, now-capped term.

## The `est_resident` admission guard's component model is wrong (total ~ok by cancellation)

`est_resident = 1024 × (512 KiB stack + 1024 KiB arena) = 1536 MiB`. Measured peak ≈ 1313 MiB:
- **stack component 512 MiB est vs 16.8 MiB real → ~30× OVER** (the 512 KiB/stack assumes fully
  resident; real is ~17 KiB demand-paged).
- **arena component 1024 MiB est vs ~1290 MiB real → ~26% UNDER**.
- Total est is ~17% conservative at K=1024 — but by the two component errors *cancelling*, not by
  being right. (ADR-0000: the guard's per-tree resident TYPE — 512 KiB stack + 1024 KiB arena, all
  resident — does not match reality: demand-paged stacks + a working-set arena pool. A sounder
  guard models the arena as the dominant calibrated term and stacks as ~17 KiB, or measures live
  RSS and fails loud approaching the limit.)

### Honest correction (measure to plateau, not mid-fill)
An earlier K=256 snapshot at 8–20s read 32 MiB resident and implied the guard over-estimates ~12×.
That was caught **mid-fill, before peak** — the K=1024 run to 45s shows the arena climbs to ~1.3 GiB.
The "12× over" read was premature; the corrected finding is the component-model split above.

## Implications
- **Option B has ~no memory advantage over A.** B saves the fiber stacks (16.8 MiB at K=1024,
  negligible); the arena (98%) is paid by BOTH (both hold the `_Node` graphs). B's case rests on
  the ~1% CPU (tlab_finding #30), not memory.
- **The K=512-vs-1024 memory saving (#29) is real and arena-driven** (halving concurrent parked
  trees ≈ halves the arena peak, ~hundreds of MiB), NOT stack-driven.
- The arena is the lever for any real producer-memory reduction (n_sims, or fewer concurrent
  parked trees), not the stacks or the (already-bounded) belief memo.

## Files
- `smaps_k1024_45s.txt` (peak attribution), `smaps_snap_{8,14,20}s.txt` (K=256 mid-fill),
  `producer_k{256,1024}.log`, `producer_heap.zst` (the near-empty heaptrack — kept as the
  methodological evidence). Tool: `throughput-lab/scratch/smaps_summary.py`.

Public Domain (The Unlicense).
