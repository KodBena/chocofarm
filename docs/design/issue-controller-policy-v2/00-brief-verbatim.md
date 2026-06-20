<!-- docs/design/issue-controller-policy-v2/00-brief-verbatim.md — Public Domain (The Unlicense) -->

# Phase brief (verbatim, for audit)

This is the **exact** shared-context block prepended to every one of the six agents in this run — reproduced verbatim so the framing can be audited for bias. Each agent also received a per-step task line (see the phase files) and, for steps 2–4, the prior steps' structured JSON appended verbatim (data, not framing).

```text
SHARED CONTEXT — the system and the control interface (these are facts; ground them by reading the cited files; do not treat any of it as a conclusion to confirm or refute).

You are designing one piece of a benchmark fixture in the chocofarm repo (working dir /home/bork/w/vdc/1/chocofarm).

THE TRANSPORT (the machine being controlled).
`cpp/build/chocofarm-wire-ab-bench` is a throughput benchmark for a leaf-evaluation transport. It runs T PRODUCER THREADS (--pool-threads T). Each producer thread owns N search trees (--trees-per-thread N), each tree with some slots; a slot that reaches a leaf needs one neural-net evaluation. A producer gathers parked (ready) leaves, packs some of them into a MESSAGE, and submits the message to a single shared EVALUATION SERVER over a ZeroMQ DEALER socket; when the reply arrives it applies the evaluations (advancing its trees, which produces new ready leaves) and issues more. `--inflight-msgs D` is the per-thread cap on outstanding (submitted-but-unanswered) messages; the runner's refill issues while inflight < D. The SERVER (cpp/stage_a/stage_a_server.py) is single-threaded: on each wakeup it drains the currently-queued messages and runs ONE batched forward over their concatenated rows, then routes the replies back by id. (Exact runner mechanics: cpp/src/runner_wire_batched.cpp, function run_episodes_wire_pipelined. The server: cpp/stage_a/stage_a_server.py.)

THE ACTION (what the controller decides).
A per-thread BINARY ISSUE GATE: allow[tid] in {0,1} for each producer thread. The runner consults it at the single discretionary-issue point — the effective gate is `inflight[tid] < D && allow[tid]`. So the controller may only DENY a thread's next discretionary issue; it cannot force an issue and it cannot change D. One liveness carve-out: when a thread has nothing outstanding but still has ready leaves it performs a FORCED FLUSH that is NOT gated (a denied thread always keeps making progress and can never deadlock). Default all-allow reproduces the plain fixed-D runner exactly. The gate is read on the hot path as one relaxed atomic load; a stale value is harmless.

THE CONTROL LOOP (how your policy runs).
The policy lives in an external PYTHON process (cpp/stage_a/issue_engine.py), bound to a ZeroMQ socket. On a slow cadence (a few milliseconds) the C++ side sends a FEATURES SNAPSHOT and receives the per-thread allow bits back (C++ bridge: cpp/include/chocofarm/issue_control_bridge.hpp; in-process actuation hub: cpp/include/chocofarm/issue_controller.hpp). A policy is any function `f(features: dict) -> [0 or 1] * T`. The policy is iterated in Python with no C++ recompile.

THE OBSERVATION (the features your policy receives each tick).
Defined by `struct IssueFeatures` in cpp/include/chocofarm/issue_controller.hpp and decoded in cpp/stage_a/issue_engine.py (decode_features). Per-thread vectors (length T): inflight, ready, msgs (cumulative), leaves (cumulative), mean_rtt_ms (0 until its channel is wired). Scalars: n_threads (T), d_ceiling (D), server_rows_per_forward (0 until its channel is wired).

THE OBJECTIVE.
Maximize the benchmark's throughput — leaves evaluated per unit wall-clock (the bench reports a number it calls dps).

SCOPE DISCIPLINE. Do the design step named in YOUR TASK below, and only that. Ground the facts above by reading the cited source files; do not point your analysis at anything else under docs/ (except where your task names a specific ADR). You are NOT asked to diagnose the transport, to identify a bottleneck, or to characterize any failure mode — only to carry out your assigned design step.
```
