<!--
docs/notes/leaf-eval-loop/step-4-tio-decomposition.md
Purpose: Step 4 of the leaf-eval impl->model loop. Instruments WHY dps is sub-linear in B (Step 3): an
  eventlog decomposition of the realized forward into its COMPUTE (dt_us) and the inter-forward GAP (the
  server's non-compute time, what the model lumps as T_io), at two batch widths. Finds the model's T_io is
  both ~50-90x too small AND scales with B -- the unmodeled B-scaling cost behind the sub-linearity.
ADR-0005 point-in-time record; ADR-0006 header; claims-measured-vs-interpreted (every number tagged).
Public Domain (The Unlicense).
-->

# Step 4 — the T_io decomposition: the inter-forward gap grows with B (2026-06-23)

> **Resolved by Step 5 (`step-5-wire-isolation.md`):** the gap is the **producer's search-wait** (~85%), NOT
> the wire `T_io`. An isolated ZMQ benchmark shows the pure wire is only **10-17%** of the gap (a ~113 µs fixed
> ZMQ-IO-thread cost + ~0.29 µs/row). So the B-scaling "`T_io`" below is the producer's *search* (more leaves →
> more Gumbel-AZ search), and the right model term is a **coordination/overlap** term (poor producer↔server
> pipelining, msgs/forward≈1), not a bigger `T_io`. The wire reading in this note is an upper-bound mislabel.

Step 3 found dps grows **sub-linearly** in B (filling the pad does not reach the model's ceiling). Step 4
instruments *why*: a `CHOCO_EVENTLOG` decomposition of the realized forward into its **COMPUTE** (`dt_us`) and
the **inter-forward GAP** (`period − dt_us` — the server's *non-compute* time per forward: drain / decode /
scatter / producer-wait, which the model lumps as **`T_io`**), at two batch widths. `--secs 12`, AllAllow,
padmax, watchdog-wrapped (`tools/shell/compute-watchdog.sh`; both runs `watchdog_rc=0`). Artifacts:
`~/w/vdc/chocobo/runs/control_lab/step4-eventlog/`.

| | B | `dt_us` (compute) | **gap** (T_io/idle) | period | dps |
| --- | --- | --- | --- | --- | --- |
| N=2 | 154 | 2125 | **905** | 3029 | 99.3 |
| N=4 | 276 | 2293 | **1864** | 4157 | 167.6 |

(means over ~3800 / ~2800 steady-state forwards — **stable**, unlike the noisy `dps_window`.)

## The finding: the model's `T_io` is both ×45–93 too small AND scales with B

- **Compute is ~flat** (`dt_us` 2125 → 2293, +8%): the padded-512 forward — ~constant in B, exactly Step 2's
  pad-aware result (the forward computes on `max_batch`, not B; `dt_us ≈ T_disp + 512·t_row`).
- **The inter-forward GAP roughly DOUBLES** (905 → 1864) as B goes 154 → 276 (1.8×) — **the gap scales with B**
  (≈6.3 µs/row; 905/154=5.9, 1864/276=6.8). This is the **non-compute** server time per forward, which the
  model represents as **`T_io`, grounded at 20 µs CONSTANT** (a PRIOR — the model's top *unmeasured* term).
- So the model's `T_io` is wrong **two ways**: **(a) fidelity** — it is 905–1864 µs, not 20 (×45–93); **(b)
  form** — it **scales with B**, the model holds it constant. **This B-scaling term is precisely why dps is
  sub-linear in B**: the cycle `T_disp + T_io(B) + W·t_row` grows with B through `T_io`, so filling B does not
  raise throughput proportionally and the model's full-fill ceiling is unreachable.

## The corrected serve form

`cycle_us = T_disp + T_io(B) + W·t_row` with **`T_io(B) ≈ k·B`** (k ≈ 6.3 µs/row from these two points) and
`W` = the padmax compute width (512) — replacing the constant `T_io=20`. By the throughput identity
(`dps = 1e6·B / (cycle·L)`) this *is* the measured period→dps relation: the measured `period` (3029, 4157)
already equals `dt_us + gap`, so a model carrying `T_io(B)=gap` reproduces the realized dps. The model fix is
therefore concrete: **make `T_io` a measured function of B, not a 20 µs prior.**

## Honest accounting (claims-measured-vs-interpreted)

- **Measured (M):** `dt_us`, `gap`, `period`, `B` at N=2, N=4 — means over thousands of forwards (stable).
- **Inferred (I):** `T_io ∝ B` *linear* — from **two points** (the DIRECTION, gap grows with B, is robust;
  the exact form/slope is provisional — more B values would pin linear-vs-curved).
- **Caveat:** the GAP is `T_io + idle` (drain/decode/scatter *work* AND any producer-*wait*), not purely the
  drain cost. The finer split — the `DRAIN`-event timing vs the FWD: is the growth drain-work or producer-wait?
  — is the optional **Step 5**. Either way the lumped non-compute term grows with B and dwarfs the constant 20.
- **Tooling:** ran under `compute-watchdog.sh` (maintainer-authored, verified this session) — no stall.

## Where the loop stands

The "why does the implementation underperform the model" arc now has its answer at the static operating point.
Not the operating point (corrected at the glossary), not a coordination idle (refuted at Step 2), not the fill
lever alone (Step 3) — but the **`T_io` term: the server's non-compute per-forward time grows with the
coalesced batch B and is ×45–93 the model's constant prior.** That is the concrete model fix (a B-scaling
`T_io`) and a direct measurement of the model's top unmeasured term. Optional refinement (Step 5): decompose
the gap (drain-work vs producer-wait) to ground `T_io(B)`'s functional form.
