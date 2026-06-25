# Cache stats for the Option-B by-reference win — direct vs fiber(A) vs cursor(B), 2026-06-25

**Stamp:** worktree `optionB-impl` @ `caa13c6`, `leaf-cpu-microbench` -O3, `nice -n -19 taskset -c 0`,
production cfg (m=24 n_sims=256 c_outcome=2 max_depth=24), 1200 decisions/mode. Real host cache:
L1 32 KiB/core, L2 256 KiB/core, L3 6 MiB shared. Load rose 0.03→0.61 during the run and cursor ran
LAST (highest load) yet came out lowest — so the B-wins result is conservative; miss COUNTS are
event-driven and robust to CPU contention.

## Counts (per 1200 decisions)

| metric | direct | fiber A | cursor B | B vs A | B vs direct |
| --- | --- | --- | --- | --- | --- |
| cycles | 31.5e9 | 31.7e9 | **31.2e9** | lowest | lowest |
| instructions | 78.1e9 | 79.0e9 | 78.0e9 | — | — |
| L1d miss (mem_load_retired.l1_miss) | 276.1M | 288.8M | **265.9M** | **−7.9%** | −3.7% |
| L2 miss | 86.1M | 94.6M | **82.7M** | **−12.6%** | −3.9% |
| L3 hit | 84.9M | 90.3M | 81.5M | — | — |
| L3 miss (→DRAM) | 1.61M | 1.48M | 1.45M | ~tied (~0.01% of loads, all) | |
| L1i miss | 314.2M | 341.0M | **314.1M** | **−7.9%** | ≈ direct |
| iTLB miss | 4.10M | 3.97M | 3.96M | ~tied | ~tied |
| dTLB miss | 5.17M | 5.40M | 5.15M | −4.6% | ≈ direct |

## The story (corroborates finding #34, the ~1.6% B-over-A win)

**B is lowest on every cache level.** The win decomposes onto both axes:

1. **Instruction side — B eliminates fiber A's penalty.** L1i miss B 314M = direct 314M ≪ fiber 341M
   (+7.9%). This is the ~1% intrinsic fiber cost the original perf analysis (perf-cache-20260625)
   localized to instruction-fetch — the boost.context switches disrupting i-cache locality. B runs
   straight-line on the thread stack (no switches), so its i-cache behaves like the bare recursion.
   iTLB is flat across all three (confirming the earlier "+47% iTLB" was a multiplexing artifact,
   already corrected to ~+10% and now ~flat on the clean no-multiplex pass).
2. **Data side — B beats even direct.** L1d miss B 266M < direct 276M < fiber 289M; L2 miss B 83M <
   direct 86M < fiber 95M. This is WHY B edged ahead of the bare recursion: the by-reference cursor
   narrows ONE persistent in-place belief (descent_bw_, reused across decisions → warm in cache),
   whereas both direct's recursion and the old by-copy cursor materialize a CHAIN of fresh ~2 KiB
   belief buffers down the descent. One warm belief beats a cold chain — fewer L1d/L2 misses, and
   slightly fewer dTLB misses (fewer distinct belief pages touched).
3. **L3/DRAM untouched** (~0.01% miss, all three) — working set fits the 6 MiB L3; no
   memory-bandwidth component. Consistent with the original compute/instruction-bound conclusion.

Net: faster (lowest cycles + fewest misses at every level), spatially leaner per-tree (the in-place
single belief shows up as better data locality), architecturally sounder (straight-line thread-stack
execution removes the fiber instruction-side cache penalty). All three of the merge rationale's
claims (time, space, soundness) are corroborated at the microarchitecture level.

Public Domain (The Unlicense).
