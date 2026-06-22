<!--
docs/notes/leaf-eval-loop/step-0-synthesis-and-path-forward.md
Purpose: the synthesis of Step 0 of the implementation->model diagnostic loop, after the maintainer
  re-grounded "production" to control_lab (the closed-loop issue-gate control lab). Ties together three
  parallel investigations (verified throughput numbers; the codebase-level SSOT violation; the doc
  review) and proposes how to move forward. Produced in an autonomous session the maintainer authorized;
  it is INPUT for their review, not a decision and not an executed change.
ADR-0005 (a point-in-time record); ADR-0006 header; ADR-0002 fail-loud; ADR-0012 SSOT lens;
  claims-measured-vs-interpreted throughout (every number is config-named + measured-vs-inferred).
Public Domain (The Unlicense).
-->

# Step 0 — synthesis & path forward (2026-06-23)

After three corrections to "what production is" (`--serve`/StrictBarrier → the fixed-N `overcommit_sweep`
→ the closed-loop **`control_lab`**), this synthesizes three parallel investigations of the re-grounded
Step 0 and proposes the way forward. Investigations (first-hand, cited): `acfa2641` (verified numbers),
`adb20a08` (the SSOT violation), `abf00404` (the doc review).

## What the three investigations found

### A. The implementation is a CONTROLLED system, not a pipeline

`control_lab` (`cpp/stage_a/control_lab/lab_harness.py`) is a closed-loop issue-gate control system: on
every server forward, a `Controller` reads each producer thread's `(ready, inflight, rtt_us,
server_rows_per_forward)` and returns a per-thread **allow/deny** bit — *issue your leaves now, or hold to
bank a fatter coalesced batch* — maximizing dps. The lab scores controller **methods** (`bang_bang`,
`contextual_bandit`, `a2c`, `reinforce`, …) back-to-back over one warm stream. **`B` (rows/forward) is a
CONTROLLED variable, not a fixed input.** `AllAllow` (allow everyone) is the baseline arm (byte-identical
to the fixed-D runner); the methods are what shape `B`.

### B. The value of control is REGIME-DEPENDENT (verified, with caveats)

There is no single control_lab number; it is config/regime-dependent (all `dps_window`, measured, from
the `lab_session-*.json` + `control_research` Postgres):

- **Drain-all (`chunk_floor` off — the default depth-1 path):** control is at **parity** with `AllAllow`.
  The flagship 16-method, `pool_batch=192`, 30 s run: `AllAllow` **95.9**, `bang_bang` 90.8,
  `contextual_bandit` 93.2 — gating does not beat all-allow (the drain already coalesces; no convoy to tame).
- **Convoy regime (`chunk_floor` / depth>1):** control **dominates**. `s_min=1`: `AllAllow` collapses to
  **11.0**, `contextual_bandit` **57.0**, `bang_bang` 53.0 (~5×, |t|≫2, decisively significant). A bucket
  N=32 convoy A/B: `AllAllow` **23** → `bang_bang` **163-172** (~7×).
- **The maintainer's "~190 → ~210" is a conflation** — two *different* early runs and the transient-inflated
  `dps_samp_mean` metric (`all_allow`=192.7 in one session; `bang_bang`=210.9 in another), not a matched
  pair. No artifact reproduces it head-to-head.
- **Model ≈456** is a *modeled arithmetic ceiling* (3 cores × ~76 000 leaves/s/core ÷ ~500 leaves/dec),
  explicitly an optimistic upper bound, at an operating point that **exists nowhere as a runnable config**.
- **Caveat (claims-measured-vs-interpreted):** the per-sample dps series are autocorrelated, so naive CIs
  understate variance; borderline results (|t|≈2) are NOT established. Every number above is config-named;
  the **direction** (convoy → control wins big; drain-all → parity) is robust, the magnitudes are provisional.

### C. The root: a codebase-level SSOT violation (the maintainer's insight, confirmed concrete)

"What production IS" took three corrections because **there is no single home for the operating point**:

- The operating **CONFIG is multiply-homed and DIVERGENT**: `pool_batch` is an independent literal at
  {**32** (`runner_wire_batched.hpp:105`/`runtime_config.hpp:25`), **64** (`overcommit_sweep.py:199`),
  **192** (`lab_harness.py:432`), **swept** (`stage_b_poolbatch_sweep.py:169`)}; the coalescing floor ∈ {1, 32};
  and the overcommit **mechanism `N` itself was silently REDEFINED** across sweeps — `trees_per_thread` (an
  explicit knob) in `overcommit_sweep`, **dropped and redefined as native `K=ceil(pool_batch/pool_threads)`**
  in `stage_b_poolbatch_sweep.py:11-13`. Two harnesses both calling themselves "production" realize overcommit
  by different code paths with different semantics.
- The model's operating point `{n_gen:3, B_op:256, LPD:500, tau_io:20}` is **assembled from idealizations**
  (a measured bench `R_gen`, a design-pin `LPD` self-labeled a "TAUTOLOGY", an "UNMEASURED" prior `tau_io`,
  a full-bucket `B_op`) and **joined to the harness only by PROSE comments** (`grounding.py` imports nothing
  from the harness; `B_op=256`↔harness `max_batch` and `n_gen=3`↔`--threads 3` are comment provenance only)
  — audit cancer-G ("load-bearing knowledge in unenforceable prose"), which ADR-0012 P7 forbids.
- The model-derived numbers (**456**, **192**) are re-typed as bare literals into 4+ homes; `references.py:16`
  even self-documents the sin ("`overcommit_sweep.py:307 BARE LITERAL`").
- Honest scope (ADR-0002): the leaf-eval *tool's internal* P1 discipline is clean (one `INPUT_NAMES`, one
  `throughput_jax`). The violation lives entirely at the **harness↔model seam** and **across the harness family**.

## The two findings

**Finding 1 — the model does not model control.** `f = min(stages)` with a *fixed* `B_op` denotes a *fixed*
cycle; the implementation *controls* `B` per forward against live state. This is neither a fidelity fault
(no benchmark lies) nor a classic form fault (no missing *cost* term) nor the consultation's §7a
coupling-example — it is a **missing control law**. The value-of-control (`AllAllow` → best method, a
measured, regime-dependent quantity) is structurally invisible to the model. This re-grounds the
consultation's §7.4 worked instance: `B` is not a fixed under-batched point (~54) the implementation
"happens to occupy" — it is the quantity the system *actively controls*.

**Finding 2 — the SSOT violation makes the comparison ill-defined (the root).** Until "the operating point
we are explaining" has a single home, the loop's form-vs-fidelity discriminator (consultation Step 3)
cannot fire honestly — it would attribute a "gap" that is really a **config-mismatch artifact** (different
harnesses at different `pool_batch`/floor/mechanism). The three "what is production" corrections are the
symptom; the missing `OperatingPoint` SSOT is the disease.

## The path forward (proposed — the maintainer decides; nothing here is executed)

**(0) Fix the SSOT first — it is the prerequisite, not a side-quest.** Establish ONE authoritative
`OperatingPoint` record (a typed SSOT, ADR-0012 P8): `{pool_threads, pool_batch, trees_per_thread,
wire_mode, min_coalesce/θ, n_sims, m, max_batch, server_cores, producer_cores, regime}`. Every harness
*constructs* its config from it (or a named delta off a shared base); the model grounding *derives* its
operating-point inputs from it (`n_gen`, the `B_op` *target*) instead of re-pinning literals joined by
comments; stop re-typing 456/192. Mechanize per ADR-0011 Rule 2/4 — an `OperatingPoint` that **fails loud**
when a harness's realized config diverges from the recorded one (the class-level net, quantifying over *any*
harness, not patching each sweep). **Without this, every gap the loop computes is suspect** — which is why
it is step 0 of moving forward, not an afterthought. (A Band-2/3 structural change — gated by your review.)

**(1) Re-scope the reconciliation around the controlled system.** Once the operating point is pinned: the
honest comparison is the model evaluated *at the controller's achieved operating point* (the realized `B`
distribution, the regime) vs the controller's achieved dps — and the **value-of-control** (`AllAllow` → best
method) is a first-class *measured* finding the model does not represent. "The model doesn't model control"
is the headline.

**(2) The witness is partly already built.** `control_lab` is an operational, instrumented, *clocked*
end-to-end cycle on the real stages with passive Postgres egress — exactly the "passive ports parsed offline"
the witness-lowering review prescribed. So the consultation's "build the witness" open question (§11.1) is
**cheaper than the review assumed**: control_lab is a down-payment / partial witness for the controlled
regime. The remaining gap to a full witness is the per-stage decomposition — and `tau_io` still has no
isolated live observation (the standing form finding).

**(3) The regime is the Step-3 lever.** Control's value lives in the convoy (`chunk_floor`/depth>1) regime;
the lab's `chunk_floor` A/B toggles exactly the regime where the gate bites — a gift for the form-vs-fidelity
discrimination, and a clean A/B the loop can run without new instrumentation.

## Honest accounting

- The numbers are regime-dependent + autocorrelation-caveated; the "190/210" recollection is **not** reproduced
  by any artifact and is **not** asserted here. State control_lab results per regime, config-named.
- The `OperatingPoint` SSOT is a **proposal** (`adb20a08`'s advisory), not executed — a Band-2/3 structural
  change for the maintainer's decision.
- The doc corrections this session applies (GLOSSARY §0/§5; the Step-0 dated amendment; the consultation dated
  addendum; the witness-review appended note) re-ground the artifacts to control_lab. The point-in-time records
  (the Step-0 original body, the consultation body, the witness verbatim quote) are amended by **dated append**,
  never retro-edited (ADR-0005 Rule 8). The MANUAL is untouched (it documents the tool's contract, not which
  implementation is production).
