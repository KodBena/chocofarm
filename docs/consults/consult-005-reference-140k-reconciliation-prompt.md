# consult-005 — commission: reconcile the reference-stack throughput numbers (the "140k")

**Date:** 2026-06-24 · **Genre:** ADR-0014 second opinion (executor stumped) · **Recorded verbatim per ADR-0005 Rule 9.**

**Trigger:** measuring the reference stack's server util/throughput at its recorded 140k operating point produced THREE disagreeing numbers (server-side ~62k, bench-side dps→~120k, recorded ~140k; up to 2× apart) across three attempts (wrong operating point → a perturbing compute-timer that serialized the pipeline → non-invasive CPU). Per the maintainer's standing rule (criterion-not-met → immediately ADR-0014) and the rule's 3-partial-attempts trigger, escalated rather than launch a 4th self-attempt or speculate on the discrepancy. **Instrument:** an independent `general-purpose` reviewer, Opus tier, briefed for independence (Rule 3). **Model:** opus.

The commission prompt, verbatim:

---

You are an INDEPENDENT reviewer. Reason from the evidence and the code YOURSELF; treat the prior work's numbers and notes as claims to audit, not facts. Your value is that you have not walked this path — you are free to conclude any prior measurement is flawed, any recorded number is an artifact, or the question is mis-framed.

== THE PUZZLE ==
A "reference" inference stack supposedly does ~140,000 leaf-rows/s on this 4-vCPU host. Three measurements of what should be the same thing disagree by up to 2×, and the prior contributor could not reproduce the 140k. Reconcile them, and determine the reference server's true throughput and whether its server core is saturated.

== THE STACK ==
- Repo: /home/bork/w/vdc/1/chocofarm. Reference driver: `cpp/stage_a/overcommit_sweep.py` (helpers `build_and_publish`, `start_server`, `run_bench`); reference server `cpp/stage_a/stage_a_server.py` (`StageAServer`, subclass of `chocofarm/az/inference_server.py`'s `InferenceServer`); reference producer `cpp/build/chocofarm-wire-ab-bench` (source `cpp/src/wire_ab_bench.cpp`). Python: /home/bork/w/vdc/venvs/generic/bin/python. redis is up (6379/6380); overcommit_sweep publishes its own net.
- Operating point of the recorded 140k: pipelined-bucket wire-mode, trees_per_thread=9, threads=3 (producer cores 1,2,3), server core 0, pool_batch=64, inflight-msgs=8, min_coalesce=32, max_batch=512, m=24, n_sims=256, hidden=256. 196.6 rows/forward.

== THE THREE DISAGREEING NUMBERS (claims to audit) ==
1. **Recorded ~140k leaf-rows/s** — DB `throughput_research` (psql -h 192.168.122.1 -d throughput_research), tlab_reading rows 16/17, ~140,578 / 139,585 leaf-rows/s, 196.6 rows/fwd, producer_bin chocofarm-wire-ab-bench, "overcommit_sweep pipelined-N9 reference target", run oc-20260624-151625. (How was leaf-rows/s computed for these? It is NOT obvious the column was a direct server-side count — investigate.)
2. **Prior contributor's server-side reproduction ~62-76k leaf-rows/s** — counting `StageAServer.n_real_rows` (real rows forwarded) over wall, at the same N9 config. Script: ref_cpu.py (and ref_util.py). Server CPU measured (non-invasively, os.times user+system of the in-process server thread pinned to core 0) = 73.3% of one core. rows/fwd ~222.
3. **The bench's own RESULT line: dps_per_core ≈ 48** at the same run → ×3 producer cores × LPD(~835) ≈ ~120k leaf-rows/s.

So: bench-side ≈120k, server-side ≈62-76k, recorded ≈140k, server CPU ≈73%. Why do (2) and (3) disagree ~2×? Which is right? Is the server saturated or not?

== YOUR TASK ==
Independently determine: (a) the reference's TRUE leaf-rows/s at the N9 operating point, computed a way you can defend; (b) why the server-side count and the bench-side dps disagree by ~2× — is the prior server-side measurement undercounting (sampling/window artifact in ref_cpu.py?), or is the bench dps inflated (LPD definition? warmup? decisions that don't all become forwarded rows?), or is the recorded 140k itself suspect (how was that column derived)?; (c) whether the reference server CORE is saturated at the operating point (is 73% CPU the real steady-state, or a measurement artifact?). Reconcile to ONE coherent account.

Do real verification: read the code (how wire-ab-bench counts decisions/LPD; how overcommit_sweep derives any leaf-rows/s; how n_real_rows is incremented), query the DB, read ref_cpu.py/ref_util.py critically, and run overcommit_sweep / a probe yourself if it discriminates (you have a shell; load is shared — check loadavg, prefer short runs, the host has the build + venv). Note: a prior in-server "compute timer" that forced a device->host block was found to SERIALIZE the pipeline and halve throughput — beware any instrumentation that perturbs.

DELIVER: the reconciled account (one coherent story for all three numbers, with the arithmetic and the code/line or query evidence for each), the reference's defensible true throughput + whether its core saturates, and a flag of anything you could not verify. Where the prior method is wrong, say exactly where. (Project law in passing, do not let it lead you: ADR-0012 holds the typed signature is the single source of truth, and the project prizes measured fact over interpretation — apply that skepticism to your own conclusion too.)

---

(Report in `consult-005-reference-140k-reconciliation-report.md`. The report's central claim — the 140k is a whole-call-leaves ÷ measure-window-wall artifact — was SELF-VERIFIED by reproducing the exact leaf count 8,899,482 and the 2.00× wall ratio; finding #12.)
