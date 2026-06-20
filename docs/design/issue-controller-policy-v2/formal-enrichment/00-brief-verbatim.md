<!-- docs/design/issue-controller-policy-v2/formal-enrichment/00-brief-verbatim.md — Public Domain (The Unlicense) -->

# Enrichment brief (verbatim, for audit)

The exact shared-context block prepended to every agent in this enrichment pass — including the **critical caveat** that the formal model is an *approximate* substrate whose depth≡1 / D-dead spine is **superseded** (the chunk_floor/S_min floor un-pins DOF-depth; depth>1/D-live is the inquiry).

```text
SHARED CONTEXT — enriching an existing controller-design exploration with a formal model.

THE TASK FAMILY. You are refining a prior design exploration for the online per-thread ISSUE-GATE CONTROLLER in chocofarm's leaf-eval transport. That prior exploration (the "v2 design", committed under docs/design/issue-controller-policy-v2/) mapped candidate controllers but did so WITHOUT a formal model of the plant, so its algorithm choices were made largely by analogy. Your job is to supply that missing formal grounding (and, in a later stage, prior-art pointers).

THE CONTROL INTERFACE (unchanged; recap).
- Plant: T producer threads, each owning N search trees; a thread packs ready leaves into a MESSAGE and submits to a single shared single-threaded JAX eval server over ZeroMQ; the server drains queued messages into ONE batched forward; D = per-thread cap on outstanding messages.
- Action: a per-thread BINARY ISSUE GATE allow[tid] in {0,1}; effective gate = inflight[tid] < D && allow[tid]; DENY-only (cannot force an issue, cannot change D); the forced flush at inflight==0 is UNGATED (liveness floor). Default all-allow reproduces the plain runner.
- Observation: the IssueFeatures surface (per-thread inflight/ready/msgs/leaves/rtt_us; scalars T, D, server_rows_per_forward; the last two and rtt are sentinel-0 today). See docs/design/issue-controller-policy-v2/01-atomic-features.md and 02-derived-features.md.
- Objective: maximize throughput (the bench's dps). Control loop: a Python policy f(features:dict)->[0/1]*T over ZeroMQ on a slow cadence; the gate read is one relaxed atomic.

THE FORMAL MODEL (ADDITIONAL CONTEXT — an APPROXIMATE substrate to BOOST CREATIVITY, not a ground-truth oracle).
docs/design/stall-investigation/blind-model-v2/SYNTHESIS.md is a faithful, Z3-confirmed parametric model of this exact transport boundary, derived forward from the code: a two-party assume-guarantee protocol (section 2), source/sink timing nondeterminism (section 3), a DEGREES-OF-FREEDOM table (section 4), the qualitative regimes R0-R7 (section 5), and a self-clocking NEGATIVE-FEEDBACK batch-size fixed point bounded for all N (section 6). Read it end to end.

  ** CRITICAL CAVEAT — it is NO LONGER ENTIRELY TRUE. ** Section 0 (and DOF-depth in section 4, and fidelity-requirement #1) make "per-thread in-flight message depth is identically 1, D dead" the SPINE of the model. That holds only for the DRAIN-ALL default it was commissioned against (issue_one gathers EVERY ready slot into one message). Since then the code added a producer-side coalescing floor (chunk_floor / S_min: issue() emits a bounded S_min-row chunk and LEAVES the rest ready), which UN-PINS DOF-depth: up to D messages outstanding (depth>1), D becomes LIVE, intra-thread reorder becomes possible. THAT depth>1 / D-live regime is exactly what the issue-gate controller operates in and IS THE SUBJECT OF THIS INQUIRY. Therefore: the model is ACCURATE in the places that have NOT moved since it was commissioned (the protocol of section 2, the timing of section 3, the DOF table read as a MAP, the regimes, the section 6 feedback structure) and SUPERSEDED where the inquiry has moved (the depth/D spine, section 0). USE IT AS AN APPROXIMATE FORMAL SUBSTRATE to spark ideas and find reductions; do NOT treat the moved parts (depth==1, D-dead) as true, and do NOT re-derive its proofs into the depth>1 regime as if settled. Flag any model-based claim as the approximate/heuristic aid it is. The DOF table is most useful read FORWARD: each DOF the model pinned (especially DOF-depth/D, DOF-B coalescing degree, DOF-F self-batching feedback, DOF-S service time) is a degree of freedom the controller's gate now REACTIVATES and MOVES.

THE LENS (how to use the model).
Bolt algorithms onto this (approximate) formal structure. Where the v2 picks named an algorithm by loose analogy (e.g. "per-thread token-bucket" ~ Linux tc) with no formal target, SUPPLY THE MISSING REDUCTION: cast the gate's decision problem as a formal object via the model (a queueing / scheduling / Markov-decision / Petri-net / optimal-control / bandit / etc. formulation) and argue by ANALOGY / ISOMORPHIC TRANSFORMATION / KARP-STYLE REDUCTION which known algorithms apply AFTER a suitable transformation. The aim is to BROADEN the algorithmic space and find what genuinely fits the formal structure, not to confirm the existing picks. A formal model is no substitute for modeling flair; it is the springboard for it.

SCOPE. Do your assigned stage only. Ground model claims in SYNTHESIS.md (cite section / DOF / regime), respecting the caveat. You are NOT asked to diagnose, benchmark, or re-prove the model; you are enriching a design exploration with formal grounding and prior art.
```
