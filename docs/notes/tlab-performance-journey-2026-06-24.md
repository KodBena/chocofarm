# The throughput-lab performance journey: from a synthetic transport to a ~3.6× real-generator win — and how the bottleneck kept hiding

**Date:** 2026-06-24. **Branch:** `feat/tlab-real-generators`. **Status:** a narrative + lessons record of a multi-day performance investigation (ADR-0005 — a dated, slowly-aging account so the *science*, not just the result, survives). The terse measured reference is the sibling note `tlab-real-generators-2026-06-24.md`; this is the *why* and the *how we were wrong*.

---

## 0. The premise

The chocofarm leaf-eval path — the search asking an inference server to evaluate leaves — had a C++ coupling that was, in the maintainer's words, "too tacky to reason about." So we built `throughput-lab/`: a clean-room **synthetic** producer→boundary→server testbed to study the *transport* in isolation, with an eye toward an eventual **dynamic** coupling controller (`control_lab`). The static coupling was hardened first; this journey is what happened when we replaced the synthetic load with the **real** chocofarm Gumbel-AZ search and chased the throughput honestly.

The metric is **leaf-rows/s** (leaves evaluated per second). Its connection to the production metric, **decisions/s (DPS)**, is §7 — flagged up front because it's the point of the whole exercise.

---

## 1. The arc, with numbers

| stage | what changed | leaf-rows/s |
|---|---|---:|
| baseline | real generators, B=1, no surplus | ~16,000 |
| + fibers | round-sync K-fiber multiplexer | (the lever that made batching possible) |
| + `SCHED_IDLE` surplus | reclaim the server core's idle slack | ~20,000 |
| **+ `--msg-rows 64`** | **static producer coalescing** | **~55,000–58,000** |

**~3.6× over the baseline**, and — the punchline of the whole journey — **the dominant lever was found last.**

### 1a. Hardening the lab
A server flooding its stdout could wedge on a full pipe (the harness didn't drain it), holding the thread that runs the SIGINT handler. Fixed structurally: redirect the server's stdout/stderr to a **file**, whose `write()` never back-pressures — the wedge made *unrepresentable* (ADR-0000), witnessed by a re-armed 11 MB flood. Then the metric was set to **leaf-rows/s** (req/s favours tiny rows).

### 1b. Real generators integrated — the seam paid off
The search drives leaf eval through an injected `NetEvaluator` port. Because that seam was ADR-0012-clean, integrating the *real* search needed only a ~60-line ACL (`BoundaryNetEvaluator`) routing each leaf through *our* boundary — not a rewrite. The search emits real 241-wide features; no shape adaptation.

### 1c. Fibers: 5.8×, and they don't hurt
A single search is reply-bound (a tree blocks on each leaf's value). The fiber multiplexer runs K trees per thread, keeping K leaves in flight so the server can batch. Result (robust, IQR ~1%): K=1 ≈ K=0 (no overhead), K=128 → **5.8×**; a saturating asymptote, not a mode. The greedy-async refinement was *refuted* (round-sync ≥ greedy) — the bottleneck wasn't pipeline idle.

### 1d. The first (wrong) bottleneck read
We saw the server at ~58% matmul and a throughput plateau and concluded: **generator-bound** — the 3 search cores must be saturated, the server starved. **This was an inference, never a measurement.** (See §4, Lesson 1 — it cost the most.)

### 1e. The scheduling win — consult → enumerate → control
On a hunch the server's idle core slack was reclaimable, we commissioned a kernel-scheduler consult (ADR-0014). It diagnosed: this is **EEVDF**, where `nice` is a *share weight*, not a runnability gate — and predicted `SCHED_IDLE` (run only in true idle, yield instantly) would reclaim the slack where `nice` couldn't. We enumerated the process-topology space with a **CP-SAT** model (40 orbit-correct configs, a single-homed config space), swept it, and ran a controlled policy A/B. Verdict (IQR ~0.3%): a `SCHED_IDLE` surplus generator on the server's core, **+18–25%** — exactly as the consult predicted; `nice` +5%, `SCHED_BATCH` −1%. Unprivileged.

### 1f. The audit — refuting our own story
Going to "production levels" (`-march=native`), the system *looked* server-bound and I claimed `-march=native` had flipped the regime. The maintainer demanded an **auditable, scriptable toggle** isolating the cause. It refuted the claim: toggling `-march=native` does **not** flip the regime (both builds server-bound); it only speeds the search ~modestly. And it forced the deeper correction — **"generator-bound" was the misread from §1d.** Direct per-core measurement showed the generators *never* saturate (18–35%); they're reply-bound. The bottleneck was the **server core** all along (~92–98% busy, only ~49% matmul). There was no flip; there was a mislabel.

### 1g. The profile — the overhead is Python, not IO
A production-build `perf` profile (dot-SVG in `~/plots`) itemised the server's non-matmul half: **~25% CPython interpreter** (`_PyEval` + attr/call overhead — the Python serve loop), **<1% zmq syscalls**, ~16% the (already FMA3-vectorized) matmul. The serve-path cost is the *Python serve loop*, paid **per request and per forward**.

### 1h. The big lever, found last — static coalescing
The real-gen path sent each fiber round's K leaves as **K separate B=1 messages** — maximising the per-request CPython cost. `--msg-rows M` coalesces them into ceil(K/M) messages. Sweep (real config, surplus on): **~2.9× (20k→58k), saturating at M≈64.** The columns show the mechanism: server requests collapse **22–44×** → the per-request decode vanishes → the server core is freed from serve-path work (matmul **47% → 73%**), and in-server latency *halves* (9.4 → 2.8 ms). At the banked optimum: 55k leaf-rows/s, server 73.4% matmul, ~26.6% residual serve-path overhead.

### 1i. The dynamic-control verdict
The slack the profiles pointed at is real and large — but a **static** coalescing floor captures it, and the optimum is **flat (M=64–256)**: load-insensitive in steady load. So an adaptive gate has little to beat. **Dynamic control would only pay under bursty/variable load** (the deferred episodic/early-exit workload) — now a concrete, testable precondition, not an open assumption. The residual ~26.6% serve-path overhead is its theoretical ceiling (~+36% if driven to 100% matmul), realistically ~0 in steady load.

---

## 2. The compounding levers (final accounting)

Three orthogonal wins, multiplying: **fibers** (enable batching) × **`SCHED_IDLE` surplus** (+~25%, reclaim the server core's idle) × **`--msg-rows 64`** (+~190%, kill the per-request Python overhead) → **~16k → ~55–58k leaf-rows/s, ~3.6×.** All unprivileged; banked in `run_real_best.sh`.

---

## 3. The DPS question (lab leaf-rows/s ↔ episodic decisions/s)

Production cares about **DPS**. In the episodic scenario it was **190–210 in practice, with an estimated ceiling of 457** (the search-compute ceiling — DPS with leaf-eval infinitely fast). Do our wins translate?

The arithmetic (conjecture, to validate — §5): at the operating point ~47 leaves/decision, hitting the **457** ceiling needs the eval path to sustain ~457 × 47 ≈ **21.5k leaf-rows/s**; the **190** practice corresponds to only ~9k. Our banked **55k** is ~6× the latter and ~2.5× the ceiling's requirement — i.e. **the eval path is no longer the limiter**. *If* the production gap (190 → 457) was the eval/coupling overhead dragging the search below its compute ceiling — which is exactly the premise that motivated this lab ("the old coupling was tacky/inefficient") — then our ~3× coalescing + scheduling wins attack precisely that, and DPS should rise toward 457.

**This is an analysis, not a measurement.** It rests on two assumptions the episodic run must confirm: (a) the production bottleneck was the eval/transport (not the search compute or a different coupling fault), and (b) the LPD operating point. That measurement is §5.

---

## 4. Lessons (the part worth keeping)

1. **Don't infer a bottleneck regime — measure it directly.** "Generator-bound" was inferred from an underfed server's *matmul %*; we never measured generator-core util. It was wrong, and it framed days of work. The fix is one `cat /proc/stat` away. *A throughput plateau tells you there's a wall; it does not tell you which wall.*
2. **A surprising result needs an auditable, scriptable toggle that isolates ONE variable.** The maintainer's demand — a fixed `scenario_audit.py` flipping only `-march=native` — refuted my causal claim immediately and surfaced the §1 mislabel. Surprises must come with a reproducible proof, or they're lost to oblivion.
3. **Separate measured from interpreted; mark conjecture.** Every "X is the cause" here that wasn't a toggle was eventually wrong. State the number; flag the reading.
4. **Verify the artifact, not the claim.** A 20k-vs-15k "regression" scare was just the surplus on vs off (I'd changed two things and reported one). And: run the *actual* `run_real_best.sh`, not an equivalent inline config.
5. **The biggest lever can be found last.** We shipped fibers (5.8×) and a scheduling win (+25%) before noticing the dominant one — coalescing (~3×) — which had been *unused* (B=1) in every prior sweep. Keep looking after the first win; the bottleneck hides behind the one you just fixed.
6. **Profile to localize, don't guess.** "The serve-path overhead is IO" was the natural guess; the profile said CPython interpreter, <1% IO — which is what made coalescing (fewer requests) the right fix rather than transport tuning.
7. **For a stubborn problem: consult → enumerate → control.** The EEVDF consult (ADR-0014) → CP-SAT enumeration of the config space → a controlled A/B was the workflow that turned "try this pin and that" into a declarative, decided answer.
8. **Change one thing per measurement.** The cross-report confusion (§ Lesson 4) was two variables moving at once.

---

## 5. Open / next

**Test `control_lab` under the bursty (episodic) regime** — the one experiment that closes the dynamic-control question *and* validates the DPS translation:
- Build the episodic/no-early-exit workload (real episodes, state evolving, early-exit on/off).
- Re-run the coalescing sweep under it: if the optimum *shifts with load*, an adaptive gate (`control_lab` bang-bang) pays; if it stays flat, static M≈64 suffices and dynamic control is off the table. (Working hypothesis: bang-bang is the proxy — if it captures nothing, nothing will.)
- Measure **DPS** directly against the 190→457 gap, confirming (or refuting) §7.
- Integration caveat: `control_lab` is invasive; if it isn't ADR-0012-composable a compile-time toggle may be needed — itself a signal about its coupling.
