# The throughput-lab performance journey: from a synthetic transport to a ~3.6× real-generator win — and how the bottleneck kept hiding

**Date:** 2026-06-24. **Branch:** `feat/tlab-real-generators`. **Status:** a narrative + lessons record of a multi-day performance investigation (ADR-0005 — a dated, slowly-aging account so the *science*, not just the result, survives). The terse measured reference is the sibling note `tlab-real-generators-2026-06-24.md`; this is the *why* and the *how we were wrong*.

---

## 0. The premise

The chocofarm leaf-eval path — the search asking an inference server to evaluate leaves — had a C++ coupling that was, in the maintainer's words, "too tacky to reason about." So we built `throughput-lab/`: a clean-room **synthetic** producer→boundary→server testbed to study the *transport* in isolation, with an eye toward an eventual **dynamic** coupling controller (`control_lab`). The static coupling was hardened first; this journey is what happened when we replaced the synthetic load with the **real** chocofarm Gumbel-AZ search and chased the throughput honestly.

The metric is **leaf-rows/s** (leaves evaluated per second). Its connection to the production metric, **decisions/s (DPS)**, is §3 — flagged up front because it's the point of the whole exercise.

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

> **Correction (2026-06-24, same day — a worked instance of Lesson 1).** The arithmetic above used **~47 leaves/decision — the lab's n_sims=24 config**, while the production 190/457 numbers are at **sims256/m24**, where LPD is ~2×n_sims ≈ **~470 (≈10× higher)**. Redone at the right operating point: the 457 ceiling needs ~457 × 470 ≈ **215k leaf-rows/s**, so our banked 55k would be **eval-limited to ~110 DPS — *below* the 190 practice.** That **reverses the rosy conclusion**: at the real config our static coupling may *not* yet beat production, and the eval path is very plausibly still the limiter. I extrapolated from the wrong config — the exact "infer instead of measure" error this note's Lesson 1 warns against. The sign of the answer is now genuinely open, and **only the sims256 episodic measurement (§5) settles it** — which is why that measurement is the *baseline*, not a confirmation. (Caveat on the caveat: the 55k was itself measured at sims24; at sims256 the in-flight/batch dynamics differ, so the eval throughput there must also be measured, not assumed.)

> **Measured (2026-06-24, `episodic_dps.sh`).** The sims256/m24 episodic-static baseline, no-early-exit, banked optimum (server@0 + 3 gens@1,2,3 + `SCHED_IDLE` surplus@0), 4-vCPU host:
>
> | episodic config | leaf-rows/s | LPD | **DPS** |
> |---|---:|---:|---:|
> | M=1 (no coalescing) | 19,538 | 712 | **27** |
> | M=64 (banked) | 52,187 | 634 | **82** |
>
> **Coalescing translates to a clean 3.0× DPS win in the production-shape workload (27 → 82)** — not just the synthetic leaf-rows metric. But the deeper finding settles the correction's open sign: at 82 DPS the system is **server-compute-limited** (server 73% matmul; the ~58k leaf-rows/s server ceiling ÷ 634 LPD ≈ **~92 DPS max**), well below the **~138–184 DPS search ceiling** on this box (3–4 gen cores × the measured 46 DPS/core). So the residual bottleneck is the **server's compute, not the coupling** — meaning *any* coupling control, static or dynamic, has only ~11% headroom left (82 → ~92); the path to more DPS is a **faster server** (batch size already maxed; a GPU / lighter net / lower per-row CPython), not a smarter gate. **Production comparison — ATTRIBUTED (2026-06-24, archaeology; citations verified by hand).** An earlier same-day guess here — that the gap to (190–210, 457) was a **core-count** difference — was **retracted** (same 4-vCPU machine) and is now *attributed*, not replaced by another guess. The gap dissolves: neither figure is a like-for-like operational target on the same machine.

- **190–210 is a retracted conflation.** The maintainer's own loop notes already retracted it (`docs/notes/leaf-eval-loop/step-0-synthesis-and-path-forward.md:43–45, :125`): two *unmatched* early `control_lab` sessions read on the transient-inflated `dps_samp_mean` metric (`all_allow`=192.7 in one session, `bang_bang`=210.9 in another) — "*No artifact reproduces it head-to-head*," and it is "*not asserted here*."
- **457 (≈456) is a modeled, eval-free ceiling.** A bare literal `model_optimistic_dps: 456` (`cpp/stage_a/overcommit_sweep.py:307`) = `3 cores × 76 000 leaves/s ÷ 500 LPD` — and `76000/152 = 500` is a **tautology** (`grounding.py:84–94`: LPD=500 is a labelled DESIGN PIN, `76000 := 152·500`), so 456 reduces to `3 × 152` (producer DPS/core, *eval assumed free*), at "an operating point that exists nowhere as a runnable config." It is a denotational upper bound, not a witness — comparing our real-eval 82 to it is the model-bound-as-target error.
- **DPS was never the comparable metric; leaf-rows/s is.** DPS = leaf-rows/s ÷ LPD, and the two use *different* LPD (our **measured 634** vs the **pinned 500**) and *different* eval (real vs free) — so "82 vs 190/457" literally counts different things. On **leaf-rows/s at a matched static operating point there is no evidence we are behind**: the regime-correct serve bound at the drain-all point is ~83 DPS and the realized ~99.5 already *exceeds* it (`docs/notes/leaf-eval-loop/static-reconciliation.md`) — i.e. ~0 static-integration slack. Our own residual headroom is the ~11% to drive the server 73%→100% matmul (a faster server, not a coupling gate). This confirms the maintainer's hypothesis: leaf/s primary, server-utilization/slack secondary.
- **Decision counting (ours, verified):** the `--episodic` loop counts *every* completed decision including the terminal TERMINATE (`real_producer.cpp:284`); control_lab's counting is **unattributed**. Other unattributed gaps: the exact n_sims of the 192.7/210.9 sessions; whether the `189` ref (`tools/analysis/leaf_eval_bound/contract/references.py`) cross-contaminated the "190" recollection.

> **Correction 2 (2026-06-24, `overcommit_sweep` reconciliation — overturns the "server-compute-limited / ~11% headroom" claim above).** The maintainer ran `cpp/stage_a/overcommit_sweep.py` (same machine, sims256/m24, real eval): **pipelined-N9 = 180.7 DPS, 150,966 leaf-rows/s, LPD 835, 189.6 rows/fwd** (run `oc-20260624-151625`). On the comparable metric this is **~2.9× our 52k leaf-rows/s** — so we **are** behind, and the "~0 static slack / leaf/s-parity" reading two bullets up was **wrong** (it leaned on a *different* operating point in `static-reconciliation.md`, not this head-to-head). Attribution, with operational witnesses:
> - **Pad-tax ladder (confirmed).** Our server's bucket ladder `[1,8,64,512,4096]` has no bucket between 64 and 512, so ~124-row gathers pad to **512** (~4× waste); StageA's `{64,256,512}` pads a 189-row batch to 256 (~1.35×). Re-running episodic with `--warmup 1,8,32,64,128,256,512 --max-batch 512`: **52k→70k leaf-rows/s (+35%), 82→118 DPS**, and server matmul *dropped* 73%→61% (same real work, less pad). So the "58k ceiling" was a **pad-tax artifact, not a compute limit.**
> - **Round-sync barrier (attributed by util).** At the fine ladder, per-core util is gens **56–62%** and server matmul **61%** — *nothing saturated on useful work* → **coupling-limited**, not compute-limited. The remaining 70k→~151k is the round-sync submit-all/wait-all barrier vs overcommit's pipelined `inflight_msgs=8`. **Not yet witnessed** (our greedy/pipelined driver isn't wired for `--episodic`); the earlier "greedy ≈ round-sync" was real but *regime-specific* (it held when the server was saturated, which the coarse ladder caused).
>
> **Net:** the recoverable slack is **~2.9× in software** (finer ladder + pipelining), not the ~11%/"faster server only" I claimed. The static coupling is **not** exhausted. The wrong claim came, again, from generalizing one regime's measurement (coarse ladder → saturated server) without re-measuring when the regime changed — Lesson 1/8.

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

**Done:** the episodic/no-early-exit workload is built (`--episodic`, `episodic_dps.sh`); the static baseline is measured (82 DPS / 52k leaf-rows/s); and a same-machine reconciliation against `overcommit_sweep` (151k leaf-rows/s) **attributes a ~2.9× recoverable gap** to the bucket ladder + the round-sync barrier (§3 Correction 2). The earlier "static exhausted / ~11% headroom / faster-server-only" reading is **withdrawn**.

**Next — recover the attributed software slack, in order, each with an operational witness:**
1. **Finer bucket ladder (confirmed +35%).** Make the server's default ladder dense around the live operating range (e.g. `1,8,32,64,128,256,512`) and cap `--max-batch` at the real top (512), instead of `[1,8,64,512,4096]`. This is a server default change; bank it like `run_real_best`.
2. **Pipelined episodic driver (attributed, not yet witnessed).** Wire the greedy/`inflight_msgs`-style overlap into the `--episodic` path (the round-sync barrier is the remaining limit; both sides sit ~60% under-saturated). Measure: does it close 70k→~151k? The earlier "greedy ≈ round-sync" was regime-specific (saturated server) — re-measure here.
3. **THEN `control_lab` / dynamic control** — only once the static coupling is genuinely at its frontier (ladder + pipelining), so bang-bang is measured against an honest static optimum, not a pad-tax-throttled one. Integration caveat: `control_lab` is invasive; if it isn't ADR-0012-composable a compile-time toggle may be needed — itself a signal about its coupling.

---

> **Witness (2026-06-24, later same day — item 2 measured; over-attribution in Correction 2 corrected).** The greedy/`inflight_msgs` overlap is now wired into the `--episodic` path as a `--driver` toggle, with the episode state machine kept in one home and **coalescing (`--msg-rows 64`) held identical** across both drivers — so the A/B isolates the *pipe shape* from the batch width, **within one binary against one server** (no cross-stack confound). Three interleaved replicates (sims256/m24, K=128, 12 s; commit `567ec9d`):
>
> | driver | DPS (reps) | median |
> | --- | --- | --- |
> | round-sync | 77, 64, 71 | **71** |
> | greedy (inflight 8) | 96, 90, 93 | **93** |
>
> Greedy's **MIN (90) beats round-sync's MAX (77)** — a clean ADR-0009 win — at **+31%**, with markedly *tighter variance* (round-sync's submit-all/wait-all barrier is scheduling-jitter-sensitive; greedy's continuous overlap smooths it). At `--msg-rows 16` the two **tie**, so the lever lives at the banked coalescing point, not below it.
>
> **This corrects Correction 2's round-sync bullet:** that bullet attributed the *entire* 70k→~151k residual (~2.15×) to the barrier-vs-pipeline difference. Measured in isolation the pipe is worth **~+31%, not ~2.15×**. So the 93-vs-180 (and 70k-vs-151k) residual is **dominated by something other than the pipe** — the server (pad-tax ladder, +35% measured) plus the still-unbridged **workload/producer axis** (our `tlab-real-producer` + tlab server vs `overcommit_sweep`'s `wire-ab-bench` + `StageAServer`; the `93→180` of that paste also conflates N=1→N=9 concurrency + coalescing, *not* the pipe). The lesson repeats (Lesson 1/3): reading a lever's size off a cross-config printout (here, attributing the whole gap to the barrier) over-attributes; only the one-variable within-stack A/B sizes it. **Next:** bridge the server axis (apply StageA's `{64,256,512}` bucket policy to the tlab server, compose with greedy) and re-measure the composite before reaching for `control_lab`.

---

> **Witness 2 (2026-06-24, quiet box, stamped `commit=2ac1cef`) — the "+31%" RETRACTED; the driver×ladder *interaction* found.** Witness 1's "+31%" was a single unstamped session (a low round-sync + high greedy reading); it did **not** survive a controlled re-measurement. The server-axis bridge is reachable as pure config — `--warmup 64,256,512 --max-batch 512` reproduces StageA's snap-up policy exactly (the tlab server already reads its warmed ladder from the forward, one home — no code change). The full 2×2 (driver × ladder), 3 interleaved replicates, S=20, on a quiet box (load started 0.08; the 5–6 load@end is the benchmark's own 4 gens+server, identical across cells), median leaf-rows/s:
> | ladder | round-sync | greedy | driver Δ |
> | --- | ---: | ---: | --- |
> | `[1,8,64,512,4096]` (pad-tax) | 52,697 | 55,072 | +4.5% (NOT clean — ranges overlap) |
> | `{64,256,512}` (StageA, lean) | 65,907 | 75,705 | **+15% (CLEAN — greedy MIN 75,129 > round-sync MAX 65,920)** |
> | **ladder Δ (same driver)** | **+25%** | **+37%** | |
>
> **The corrected attribution:**
> - **Driver alone is ~+4.5% at the pad-tax ladder and NOT a clean win** — the +31% is retracted. It was Lesson 3 (a one-session number recorded as proven) compounded by no commit stamp (now mechanized: ADR-0011 amendment 2026-06-24, every reading carries `commit/tree`).
> - **The driver is a clean +15% — but only at the lean ladder.** This is a genuine **driver×ladder interaction**: the pad-tax ladder makes the server compute-bound on 4×-oversized matmuls, leaving *no idle* for greedy to overlap (greedy ≈ round-sync, the "saturated-server regime" Correction 2 named); remove the pad tax and the server gets fast enough that coupling/RTT idle dominates → greedy's overlap recovers it. Greedy stays the right default (wins at the lean ladder, ties at 4096), but its value is **regime-dependent, not a flat +31%.**
> - **The ladder is the dominant lever (+25–37%);** composite greedy + lean ladder vs round-sync + pad-tax ladder = **+44%** (52.7k→75.7k). In DPS terms (LPD≈648): ~81→~117.
>
> **Banked:** the lean-ladder bridge (`WARMUP=64,256,512 MAXBATCH=512` in `episodic_dps.sh`), a config-only win with one home. **Caveat on the stamp:** `tree=DIRTY` here is from unrelated untracked files (`.claude/`, etc.), not the benchmark code — the producer binary + harness are committed at `2ac1cef`; a follow-up should scope the dirty check to the relevant paths so the flag is not falsely tripped. **Next, honestly:** the 75.7k still trails `overcommit_sweep`'s 151k — the residual now lives in the **workload/producer axis** (the cross-stack difference the server-axis bridge does not touch), which is where attribution goes before `control_lab`.

---

> **Witness 3 (2026-06-24, quiet box, stamped `commit=5f95c2a`) — the 2× tlab-vs-overcommit gap, fully attributed and half-recovered.** Bridged the producer/server axes component-by-component at matched operating points. The headline gap: ours **70k** leaf-rows/s (greedy + lean ladder) vs overcommit's **140k** (`wire-ab-bench` + `StageAServer`, pipelined-N9). Decomposed as **2.0× = 1.51× (forwards/s) × 1.33× (real rows/forward)**, then each factor attributed by *running the artifact*, not inferring:
> - **Forward graph is IDENTICAL** — tlab's `MlpForward` is "condensed from `chocofarm/az/mlp_jax.py`," StageA's `jit_forward_core` is the production wrapper over the same 241→256→65 MLP. A B1 microbench (per-call wall at batch {64,256,512}) found tlab only **1.15× slower at 256** (1.16 vs 1.01 ms — fixed per-call host↔device overhead, *not* matmul). So the gap is **not** an irreducible compute deficit.
> - **Coupling is NOT the differentiator** — both servers run **~72–79% utilized** (neither saturated, neither starved).
> - **Factor A (batch fill) — dominant, producer-side, RECOVERED.** Banked K=128 *underfilled* the 256 bucket (147 real / 58%). A1 + the decisive test (3 interleaved replicates) showed **K=256 + `--max-batch 256`** fills it (210 real / 82%, util 78.8%), recovering **74k→95k = +27%**, landing in overcommit's regime (real rows/fwd 209.7 ≈ 196.6). The lever is *fibers*, not msg-rows (bigger messages at fixed K *hurt* — they cut pipeline granularity); and capping at 256 forbids the wasteful 512-spill (K256 max-512 reached 257 real but at 46% fill of a 2× slower bucket). **Banked** as `episodic_dps.sh`'s default (K=256, msg-rows 128, `WARMUP=64,256 MAXBATCH=256`).
> - **Factor B (the residual ~1.5×, 95k vs 140k) — server architecture, ATTRIBUTED not speculated.** After matching batch-fill *and beating* util, we still trail. The in-server forward implies ~1.75 ms (0.788 util ÷ 451 fwd/s) vs the 1.16 ms isolated microbench — the inflation is **our tlab server's two-thread IO/compute split contending on the single pinned core**. Confirmed by reading `chocofarm/az/inference_server.py:780`: the production `InferenceServer` is **single-threaded** (drain → forward → scatter in one loop; the next batch queues in the OS socket buffer *during* compute — no separate IO thread). Our "hard-won" decoupled-receiver design is a **net loss on one core**.
>
> **Net:** the 2× is *our-side and largely recoverable* — +27% banked (batch fill); the residual ~1.5× is the server's two-thread-on-one-core contention (the concrete next target: a single-threaded tlab serve path, or pin the two threads off each other). All readings now persisted in the `throughput_research` Postgres store (`tlab_config`/`tlab_reading`, via `throughput-lab/harness/exp_db.py`), keyed to this commit — the journal no longer the only home for the numbers.

---

> **Witness 4 (2026-06-24, quiet box) — the single-thread serve path: a clean +7.8%, but it REFUTES the "~1.5× residual is threading" hypothesis (finding #4).** Added a single-threaded serve mode to the tlab server (`--single-thread`: drain→forward→scatter on one thread, the production `InferenceServer` model) and A/B'd it against the two-thread IO/compute split at the banked config (K256/msg128/max256), 3 interleaved replicates:
> | arm | leaf-rows/s (median) | real rows/fwd | util |
> | --- | ---: | ---: | --- |
> | two-thread (the committed "hard-won" decoupled receiver) | 87,440 | ~201 | 77.5% |
> | **single-thread** | **94,288** | ~205 | **71%** |
>
> Single-thread is a **clean win (+7.8%, single MIN 92,168 > two-thread MAX 88,016)**, and the *lower* util at *higher* throughput is the confirmation: the two-thread server's "77.5%" was **inflated by IO/compute contention counted as compute**. So the contention is real — but it's worth **+7.8%, not the ~50% finding #4 hypothesized**. The residual to overcommit (94k vs 140k, still ~1.5×) is **dominated by the tlab serve-path per-forward overhead** (in-serve forward ~1.54 ms single-thread vs overcommit's ~0.99 ms — the concat/decode/encode/dispatch *around* the forward that the isolated 1.15× microbench did not capture), not the threading model. **Next target shifts** to the serve-path forward envelope (the host↔device staging the production server folds; ADR-0012's cross-DEVICE boundary). **Classification note:** `single_thread` is a *treatment arm* (a `server_impl` variant resolved by keeping the winner), **not** an HP — it stays out of the `hp/` SSOT and is recorded as provenance; it would earn promotion to a *guarded* HP only if it proves **topology-conditional** (e.g. two-thread wins at ≥2 server cores — untested). The verdict is a `tlab_finding` superseding the provisional #4, and every reading here auto-recorded to `throughput_research` via the now-wired `episodic_dps.sh`.

---

> **Witness 5 (2026-06-24, quiet box) — amortization to ~112k, and the residual localized to `forward_batch`'s in-serve dispatch (prep / XLA-threading / surplus all RULED OUT).** Instrumented the single-thread serve loop with a per-forward phase breakdown (`prep | compute | scatter`, now in `ServerStats`). Measured at the banked K256/max256: **prep 50 µs · compute 1628 µs · scatter 297 µs** — so the host serve-overhead (concat/pad/encode/send) is *small*; the cost is the forward itself, **1.63 ms in-serve vs ~1.1 ms isolated** (the microbench, which imports `chocofarm.config`'s XLA pin). The arithmetic: `leaf-rows/s = real_rows ÷ total_per_forward`, and ~1.6 ms of fixed forward overhead under-amortized at ~205 rows is the wall.
>
> - **Amortization works, capped by supply.** Single-thread + bigger well-filled buckets: K256→**K768** lifts **94k → ~112k** (mean batch 205→~340); but the batch plateaus at ~340 (the 3 producers can't supply more concurrent rows — K1024 grew the batch to 434 yet throughput *fell*, too many fibers), and in-flight depth (8/16/32) doesn't move it. So amortization alone tops out ~112k.
> - **The residual (the forward's in-serve inflation, 1.1→1.63 ms) is the last ~1.25× to 140k, and it is NOT** — *measured, not assumed*: **prep** (50 µs), **XLA threading** (`OMP_NUM_THREADS=1 XLA_FLAGS=--xla_cpu_multi_thread_eigen=false` on the server: no change), or **surplus-on-core-0 contention** (server-alone-on-core-0: no change). It is a genuine `MlpForward.forward_batch`-vs-production-`jit_forward_core` dispatch-efficiency gap (overcommit's whole per-forward wall, 1.4 ms, is *less than our forward's compute alone*).
>
> **State:** **70k → ~112k (+60%)**, banked levers (single-thread + amortization), gap precisely localized — **not yet overshot 140k.** **Next, decisive:** split `forward_batch`'s internal phases in-serve (h2d `jnp.asarray` | jit `_forward_both` | d2h `np.asarray`×2) to localize *where* the 1.63 ms goes, then either fix it or A/B the production `jit_forward_core` (single fused `[v|logits]` pull) directly in the tlab server. That is the lever that gets us *over* 140k.

---

> **Witness 6 (2026-06-24, quiet box) — the in-serve forward inflation IS XLA per-call dispatch overhead; a numpy forward is 2.5× cheaper in-serve (witnessed at single-gen). Full 4-gen overshoot witness BLOCKED by an environment limit this session.** Split `forward_batch`'s phases in-serve (`--profile-forward`): **h2d 331 µs · jit 1351 µs · d2h 31 µs** — the d2h "two-pull" idea is dead (31 µs), and the cost is the **jit (XLA execution): 1.35 ms in-serve vs ~0.88 ms tight-loop**. For a ~20 M-flop MLP, 1.35 ms is pure XLA-CPU *per-call launch latency*, not arithmetic. Since `forward_core(params, X, xp)` is backend-generic, I added a **numpy forward** (`NumpyMlpForward`, `--forward numpy`): the SAME graph in numpy — no XLA dispatch, and no pad (numpy forwards the exact gathered rows).
>
> **WITNESSED (single-gen, identical conditions, only the backend differs):**
> | forward | single-gen leaf-rows/s | server util | per-forward compute |
> | --- | ---: | ---: | ---: |
> | jax (`MlpForward`) | 44,526 | **50.8%** | **1.48 ms** |
> | numpy (`NumpyMlpForward`) | 45,886 | **20.1%** | **0.59 ms** |
>
> Equal throughput (producer-limited at single-gen) but **numpy's forward is 2.5× cheaper in-serve** (0.59 vs 1.48 ms; 20% vs 51% util) — the XLA-dispatch inflation, confirmed and sidestepped. **PROVISIONAL projection (NOT yet witnessed):** at 4 generators the server is the bottleneck, so jax (51% util → saturates ~88–94k) vs numpy (20% util → ~5× headroom) projects numpy to **~150–220k → overshooting 140k.** The 4-generator confirmation run was **blocked this session** — sustained 4-process benchmarks began dying with signal 16 (a sandbox/CPU-quota limit reached after the session's many heavy runs; jax *and* numpy both affected, the Bash tool itself fine), so the operational witness (`model-bound-is-conjecture-not-witness`) is **pending a fresh box** — one clean 4-gen run, `SINGLE_THREAD=1 FORWARD=numpy WARMUP=256 MAXBATCH=4096 bash episodic_dps.sh 512 16`. **State: lever found and proven 2.5× cheaper in-serve; the overshoot is projected, not yet banked.** `NumpyMlpForward` is the candidate fix; it shares one param-builder with the jax forward (ADR-0012) and runs the byte-identical `forward_core`.
