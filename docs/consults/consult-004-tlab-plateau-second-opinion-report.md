# consult-004 — report: independent second opinion on the throughput-lab plateau

**Date:** 2026-06-24 · **Reviewer:** independent `general-purpose` agent, Opus tier (agentId a2be02e4f3ee3aa03) · **Recorded verbatim per ADR-0005 Rule 9.** Commission in `consult-004-tlab-plateau-second-opinion-prompt.md`.

**Integration note (added by the executor, NOT part of the verbatim report):** the report's load-bearing factual claims were independently verified before acting — DB query confirmed the two >120k readings used `producer_bin = chocofarm-wire-ab-bench` with `server_util_pct = NULL` (reference util never captured); all 31 `tlab-real-producer` readings average ~93.6k. The reframe was then tested with a pre-registered controlled experiment (synthetic max-rate flood → lab server), which confirmed the serve-loop-ceiling frame in bin B with margin (finding #10). The report is reproduced below exactly as returned.

---

Confirmed: the reference `InferenceServer`/`StageAServer`/`overcommit_sweep` **does not measure or report compute-busy/util at all.** The journal's "reference is 72-79% util" claim (finding #3) is therefore **not a measurement** — it's either an assumption or carried over from a different context. The lab measures util; the reference does not. This is the single decisive unresolved fact.

I have a complete, evidence-backed diagnosis. Let me write the report.

## Independent second opinion: the throughput-lab plateau

I read the journal (Witnesses 1-8) end to end, queried the DB, read both producers and both servers, and ran fresh probes (load was 0.2-0.9 throughout; commit `829eef4`, tree DIRTY from untracked `.claude/` only).

### Bottom line

The journal's **recent** line of attack (Witnesses 4-8: the residual ~1.5× is a server-side "forward envelope / XLA dispatch / serve-path inflation" problem) is **chasing a cost that my measurements show is not the binding constraint.** The lab server is **idle ~32% of the time** (67.9% compute-util at the banked config), and its per-forward compute (1.39 ms) is **identical** to what the reference's own forward must cost. The gap is not forward speed — it is that **the lab cannot keep its single-core server saturated, while the reference apparently can.** Whether the reference truly saturates is the one decisive fact nobody has measured.

### Evidence

**1. The "140k vs 100k" is two different producer binaries, and partly two different workloads.**
- DB readings 16/17 (the 140k figures) were recorded with `producer_bin = chocofarm-wire-ab-bench` — the *reference's own* producer driven by `overcommit_sweep.py --trees 9`. Every `tlab-real-producer` reading tops out at ~95-105k. The lab measured the reference binary with its own harness and reproduced 140k. So the comparison is not lab-vs-reference *server*; it's two producer/driver stacks.
- The workloads differ: the lab `--episodic` path (`real_producer.cpp:288-300`) runs **full episodes**, stepping the env to terminal and counting the cheap terminal TERMINATE decision; the reference replays a **warmed pool of mid-game beliefs** (`wire_ab_bench.cpp` `--pool-plies/--pool-seed`). This is why LPD differs structurally (lab ~634-687 vs reference 835) — different leaf-per-decision distributions, *not* a throughput difference. **DPS is not comparable across the two; only leaf-rows/s is** (the journal reaches this same conclusion in §3, correctly).

**2. leaf-rows/s IS a fair metric (rows the server forwarded / wall), and the real gap decomposes to forwards/s, not rows/forward.**
- Reference (reading 16): 8.9M rows / 63.3s = 140,578 leaf-rows/s; 45,270 forwards = **715 fwd/s** at 196.6 rows/fwd.
- Lab (my fresh probe, single-thread one-pull jax, K=256): 97,625 leaf-rows/s; 5,101 forwards / 10.46s = **488 fwd/s** at 191 rows/fwd, **67.9% util**, 1.39 ms/forward compute.
- Rows/forward are nearly equal (197 vs 191). The entire gap is **forwards/s: 715 vs 488 = 1.47×.**

**3. The lab server is starved, not forward-bound.**
- compute-busy 7.1s / 10.46s wall = 67.9%. At 1.39 ms/forward the server *could* do ~7,500 forwards in that window → ~144k leaf-rows/s, **if it were never idle.** It is idle ~32% of the wall (the 0.66 ms/forward of drain+pack+scatter+poll that, in `--single-thread` mode, does *not* overlap the forward — `server.py:11-12` states this explicitly: "while the forward runs, the socket is not being drained").
- I A/B'd it: two-thread mode (which *does* overlap draining) shows *higher* util (73.4%) but *lower* throughput (88k) — the IO thread steals cycles from compute on the single shared core. So **neither serve design is clean on one pinned core**: single-thread can't overlap drain/scatter; two-thread overlaps but contends. This is a real structural property of pinning the server to one vCPU.
- Deeper pipe (K=512, inflight=32) lifts leaf-rows/s to 105k but util *drops* to 64.8% — more fibers just fatten each forward, they don't cure the idle. Supply depth is not the lever.

**4. The decisive fact is UNMEASURED: the reference's server util.**
- The reference `InferenceServer`/`StageAServer`/`overcommit_sweep` **does not measure or report compute-busy/util at all** (verified by grep — no `% of wall` instrumentation). The journal's finding #3 "both servers ~72-79% util" is therefore **not a measurement on the reference side** — it is an assumption.
- This bifurcates the diagnosis with opposite conclusions:
  - If the reference runs at **~99% util** (715 fwd/s × 1.39 ms = 0.99), its forward is the *same speed* and the gap is **pure serve-loop overlap** — the reference's single-threaded loop simply spends near-zero non-compute time per forward (efficient C++-side coalescing feeding it, the OS socket buffer queuing during compute). Then the lab's fix is the **serve path's per-forward Python overhead** (the §1g profile finding the later witnesses abandoned), *not* the forward kernel.
  - If the reference runs at **~73% util**, its forward is genuinely ~1.02 ms — faster — and the journal's forward-envelope chase is right.

### The single highest-value next step

**Measure the reference's server compute-util directly.** Add the lab server's `compute-busy / % of wall` instrumentation to `StageAServer` (or run the reference producer against the *lab* server, which already reports util — readings 16/17 were close to this but the util column is NULL, so it wasn't captured). One number settles months of ambiguity: if the reference is ~99% util, the lab's entire remaining gap is **serve-loop / coupling overhead on the single core**, and the forward-envelope work (Witnesses 5-8, numpy/staged/one-pull) was optimizing a non-bottleneck. If it's ~73%, the forward chase is validated. Right now the team is optimizing the forward without having ever confirmed the forward is the constraint — exactly the "infer the bottleneck regime instead of measuring it" error the journal names as Lesson 1.

### Structural reframe the prior work missed

The plateau is most likely **the single-core serve loop's inability to overlap drain/scatter with compute**, a property of pinning the server to one vCPU — not the forward kernel. The journal *had* this right early (§1g: ~25% CPython serve-loop overhead, <1% IO) and then drifted to the forward envelope after Witness 4's single-thread win, never noticing that single-thread *wins by avoiding IO-thread contention but reintroduces the non-overlapped drain/scatter idle*. The ~32% idle I measured is that non-overlapped serve work. If true, the real levers are: (a) give the server a **second core** (the topology has a spare — 3 working gens + 1 idle-surplus core; the surplus generator could be dropped and the server given 2 cores to run a clean IO/compute split), or (b) shrink the per-forward serve-path cost (cheaper decode/scatter), not the matmul.

### Where I could not verify

- I could not measure the reference server's util (it isn't instrumented; DB util is NULL for readings 16/17). This is the load-bearing gap — my "starvation, not forward" diagnosis is strongly indicated by the lab-side 68% util + identical forward cost, but the reference-side confirmation is the missing piece.
- I did not re-run the full reference `overcommit_sweep` (it's a ~minutes-long sweep on a shared box); I relied on the DB readings 16/17, which are the journal's own most recent reference numbers.
