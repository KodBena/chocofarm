<!--
docs/design/stall-investigation/blind-model/README.md
Purpose: provenance + scope record for the BLIND faithful-model workflow run (2026-06-19) of the
  leaf-evaluation transport boundary. Companion to ../formal/ (the Z3 BMC artifacts) and
  ../../cpp-eval-wire-formal-diagnosis.md (the recovered formal diagnosis). Point-in-time record —
  amend by append, do not retro-edit (ADR-0005 Rule 8).
Public Domain (The Unlicense).
-->

# Blind faithful-model of the leaf-eval transport boundary (2026-06-19)

This directory is the verbatim output of a multi-agent **blind** assume-guarantee modeling workflow
(run `wf_45642a55-831`) of the C++↔Python leaf-evaluation transport boundary
(`cpp/include/chocofarm/wire_leaf_pool.hpp` DEALER producer ↔ `chocofarm/az/inference_server.py`
ROUTER server, driven by `run_episodes_wire_pipelined` in `cpp/src/runner_wire_batched.cpp`).

## Why "blind", and what that bought

The earlier formal pass (`../formal/`, `convoy*.py`, summarized in
`../../cpp-eval-wire-formal-diagnosis.md`) was, by its own admission, **guided by the empirically-found
convoy** ("the convoy leg was guided by the empirical finding handed in"). That makes a model that
reproduces the convoy weak as independent evidence. This run was therefore structured so the modeling
agents knew the **scope** but never the **target**: they were given the boundary to model and told to
model the source/sink nondeterministic timing faithfully, but were **not** told that any defect exists,
were given **no** empirical fingerprint (no `6 s/70 s`, no `rows/forward 1.4`), **no** prior artifacts,
and **no** mechanism. The test: does a model built with no knowledge of a defect exhibit the behavior on
its own under neutral analysis?

Structure: 4 modelers (2 producer-side, 2 server-side, diverse foci) → 4 fidelity verifiers
(per side: a *too-permissive* lens and a *too-constrained* lens, each mapping findings to code lines) →
1 synthesizer (`SYNTHESIS.md`) that composes the two halves via assume-guarantee and characterizes the
global behavior. Method was theoretical/rigorous; the `*.py` files are small bounded Z3 4.16 checks the
agents ran only to confirm a hand-derived execution (confirmation, not the source of trust).

## SCOPE CAVEAT — this run modeled the N=1 baseline, NOT the N≳4 stall

**Read this before citing any conclusion here.** The workflow modeled the code **checked out on
`docs/eval-transport-adapter-design`** (pre-overcommit) plus the production `inference_server.py`. On
that branch `K = fibers_per_thread()` and `trees_per_thread` (the overcommit multiplier **N**) does not
exist — so this run is effectively **N = 1**. The throughput **stall** under investigation is documented
in `../../cpp-eval-transport-adapter.md` §7 as a **nondeterministic wedge at N ≳ 4** on the
**bucketed-group *bench* server** (`cpp/stage_a/stage_a_server.py`), reached only via the overcommit
implementation on `overcommit-increment-i` (`K = N × fibers_per_thread()`). **That stall regime is out
of scope of this run** — its geometry never entered the high-N region where the wedge appears. A re-run
against the overcommit geometry (N×K slots, the bench server, high N) is required to model the stall
itself. This artifact is the faithful model of the **healthy N=1 baseline**, not of the failure.

## Load-bearing findings (verified against code, and they hold on the overcommit build too)

- **Per-thread in-flight depth is identically 1.** `issue_one` coalesces *every* ready slot into *one*
  message, so the `while (inflight < D && issue_one())` prime/refill loops cap at 1
  (`runner_wire_batched.cpp:540-597`). **`D` (`max_inflight_msgs`) is a dead knob.** The overcommit
  increment (i) changes only `K = N × fibers_per_thread()`, not this control flow — so it raises **S**
  (rows in the one coalesced message), **not D** (pipeline depth). This corrects `convoy4`'s premise
  ("D distinct outstanding messages per thread whose replies interleave"), which is unfaithful to the
  code at any N.
- **All coalescing is cross-thread** (≤ T messages outstanding, one per thread). Any per-thread "convoy"
  is a phase-locked cross-thread B≈1 rotation, not a per-thread pipeline collapse.
- **The scatter is non-blocking** — the ROUTER sets no socket options ⇒ `ROUTER_MANDATORY = 0` ⇒ a send
  to a full/vanished peer is *dropped, never blocked*. A send-wedge deadlock is a phantom.
- **The sole producer block and sole liveness backstop is the `RCVTIMEO = 15000 ms` recv.** A reachable
  `EXCEPTIONAL_TERMINATION` server terminal (uncaught `ValueError` on a ragged-`in_dim`/bad-shape batch,
  or a reload raise) would be a silent system-wide wedge if `RCVTIMEO` were unset — it is the one socket
  option that keeps that from being a true deadlock.
- Independently corroborates the recovered diagnosis's strongest leg: **no reachable deadlock** (1:1
  corr-id request↔reply + greedy drain).

## Files

- `SYNTHESIS.md` — the composed faithful model, assume-guarantee discharge table, regime characterization,
  degrees-of-freedom analysis, fidelity requirements, code-derivation attestation.
- `model-{producer,server}-{pacing,transport,drain}.md` — the four side-models (states, transitions,
  timing model, rely/guarantee), each derived blind.
- `verify-{producer,server}-{too-permissive,too-constrained}.md` — the four adversarial fidelity audits.
- `*.py` — the small bounded Z3 confirmation scripts (run under `nice -n 19 timeout 90`).

The live (gitignored) run output is under `~/w/vdc/chocobo/runs/leaf-eval-model/`.
