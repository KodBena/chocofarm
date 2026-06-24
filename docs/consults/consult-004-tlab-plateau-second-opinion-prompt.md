# consult-004 — commission: independent second opinion on the throughput-lab plateau

**Date:** 2026-06-24 · **Genre:** ADR-0014 second opinion (executor stumped) · **Recorded verbatim per ADR-0005 Rule 9.**

**Trigger:** the maintainer invoked ADR-0014 after the throughput-lab effort hovered ~90–100k leaf-rows/s across ~8 turns against a 140k target, with a forward-optimization arc that barely moved the headline — an observable recurrence (Rule 2), not a feeling. **Instrument:** an independent `general-purpose` reviewer on the Opus tier, briefed for INDEPENDENCE (Rule 3: give the evidence, do not lead; prior work handed in as claims-to-interrogate, not facts). **Model:** opus.

The commission prompt, verbatim:

---

You are an INDEPENDENT reviewer giving a second opinion on a stalled performance-optimization effort. Reason from the evidence and the code YOURSELF. The prior work's journal and database of findings are available, but treat everything in them as a CLAIM TO AUDIT, not a fact to build on. Your entire value is that you have not walked this path — so do not adopt its frame. You are explicitly encouraged to REFRAME the problem, including concluding that the effort is chasing a mirage.

== CONTEXT (facts) ==
- Repo root: /home/bork/w/vdc/1/chocofarm
- `throughput-lab/` is a clean-room testbed: a real C++ producer (a search-tree leaf generator) sends batched feature rows over a ZMQ ROUTER socket to a Python inference server (a small MLP forward, in_dim=241, hidden=256, n_actions=65), which replies. It isolates the producer -> wire -> server -> reply path from the rest of the system.
- The goal driving the effort: match and then exceed a reference throughput. A separate, older script `cpp/stage_a/overcommit_sweep.py` (with `cpp/stage_a/stage_a_server.py`, built on `chocofarm/az/inference_server.py`) reportedly achieves ~140,000 "leaf-rows/s" on this host. The throughput-lab episodic path currently reaches ~95,000-101,000 leaf-rows/s. The effort to close/overshoot that gap has run for many sessions; the headline number has moved little recently (~95k -> ~101k).
- Metric in use: leaf-rows/s = (real rows per forward / forward wall-time) x server utilization.
- Host: a 4-vCPU libvirt VM. The server is pinned to core 0; producers to cores 1-3. Python: /home/bork/w/vdc/venvs/generic/bin/python (jax, numpy, pyzmq). Tests/scripts run with PYTHONPATH=throughput-lab.

== ARTIFACTS TO INTERROGATE (claims, not gospel) ==
- The investigation journal: docs/notes/tlab-performance-journey-2026-06-24.md (Witnesses 1-8). It records the prior diagnoses and fixes. Audit them: do the attributions actually hold? Is the leaf-rows/s decomposition sound? Are any "wins" measurement artifacts?
- The experiment DB (postgres, no password): `psql -h 192.168.122.1 -d throughput_research`. Tables: tlab_reading (measurements), tlab_finding (authored beliefs #1-9, append-only with supersede links), tlab_config. Query them to see what was measured vs interpreted. psycopg3 is available in the venv if you prefer Python.
- The code: throughput-lab/server/ (the lab server), throughput-lab/cpp/ (the producer), throughput-lab/harness/episodic_dps.sh (the benchmark harness), vs the REFERENCE: cpp/stage_a/overcommit_sweep.py, cpp/stage_a/stage_a_server.py, chocofarm/az/inference_server.py. The C++ producer source is under throughput-lab/cpp/ (built binary at throughput-lab/cpp/build/tlab-real-producer).

== YOUR TASK ==
Independently assess TWO things:
1. Why is throughput-lab plateaued at ~100k while the reference reports 140k? Is the prior work's attribution of the gap correct, or is the bottleneck somewhere it hasn't looked?
2. Is the current line of attack even targeting the right thing — or is the framing itself wrong?

You are free — encouraged — to reach any of these conclusions if the evidence supports them:
- The 140k vs 100k comparison is NOT apples-to-apples (different workload, producer count, batch policy, measurement window, or even a different DEFINITION of "leaf-rows/s" between the two harnesses) — so part or all of the "gap" is illusory.
- The plateau is STRUCTURAL — a property of the 4-vCPU host, the single-core pinned server, the producer's per-leaf eval cost, ZMQ/IPC overhead, or Python dispatch — and ~100k is at or near a real ceiling. If so, say what the ceiling is set by and roughly where it is.
- The bottleneck is not where the journal says (it has been focused on the server-side forward). Maybe it is the producer, the wire/coalescing, the IPC round-trip, GIL contention, or the topology (3 working gens + 1 idle surplus).
- The metric or the testbed design is subtly misleading.

Do real verification: read the relevant code, query the DB, and run a probe if it would discriminate between hypotheses (you have a shell; the venv and binaries are built). Note: the host is busy and shared, so prefer short/cheap probes and check loadavg before trusting timings.

DELIVER a concise report:
- Your independent diagnosis of the plateau (with the evidence you checked, cited by file/line or query).
- The SINGLE highest-value thing to investigate or try next — OR a clear statement that the gap is smaller/illusory/structural than framed and the effort should stop or change target.
- Any structural reframe the prior work missed.
- Where you could NOT verify something in the time available, say so plainly rather than guessing.

For your awareness (project law, mentioned in passing — do not let it lead your diagnosis): ADR-0012 holds that a typed signature is the single source of truth, and the project prizes honest measurement over interpretation (a measured plateau of one config is not a proven ceiling unless shown to be). Bring that skepticism to your OWN conclusions too.

---

(Report recorded in `consult-004-tlab-plateau-second-opinion-report.md`.)
