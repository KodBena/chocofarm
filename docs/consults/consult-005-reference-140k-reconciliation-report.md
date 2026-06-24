# consult-005 — report: reconcile the reference-stack throughput numbers (the "140k")

**Date:** 2026-06-24 · **Reviewer:** independent `general-purpose` agent, Opus tier (agentId a05cb24c69f1e4fc3) · **Recorded verbatim per ADR-0005 Rule 9.** Commission in `consult-005-reference-140k-reconciliation-prompt.md`.

**Verification note (executor, NOT part of the verbatim report):** the report's central claim was SELF-VERIFIED per ADR-0013 Rule 5 before acting — a fresh `overcommit_sweep` N9 run at `be63062` reproduced the leaf count **8,899,482 bit-identically**, giving `÷ measure-wall(62.7s) = 141,985` (≈ recorded 140,578) and `÷ whole-call-wall(125.6s) = 70,851`, a **2.00×** ratio. Recorded as finding #12 + corrected reference reading #50. The report is reproduced below exactly as returned.

---

I have a complete, independently-verified reconciliation. Let me write up the final account.

## Reconciled account — the reference "140k" is a unit-mismatch artifact; true rate is ~62–69k, server core is NOT saturated

### The one coherent story

All three numbers describe the same N9 run; they diverge only in **which window the numerator and denominator are taken over**. The server's row counter (`StageAServer.n_real_rows`, incremented per-forward in `_scatter`, `stage_a_server.py:143-146`) runs the **entire time the server thread is alive during an iteration** — which spans the bench's *warmup pass* **plus** its *measure pass*. The bench's `dps` denominator covers **only the measure window**. Mixing the two inflates the rate by ~2.25×.

**The arithmetic (verified against DB rows 16/17 and a fresh run I ran at `be63062`):**

- Recorded reading 16: `leaves=8899482, forwards=45270, wall_s=63.31, leaf_rows_s=140578, dps=168.3, lpd=835`.
  - `leaf_rows_s = leaves / wall_s` → `8899482/63.31 = 140,570` ✓ (the recorded column is literally this division)
  - `= dps × lpd` → `168.3 × 835 = 140,530` ✓
  - `lpd = leaves / decisions` → `8899482/10653 = 835.4` ✓
- **The defect:** `leaves=8899482` is the `n_real_rows` delta over the **whole bench call**; `wall_s≈63.3s` is the **measure-window** wall (`168.3 dps = 10653 decisions / 63.30s`). My fresh run reproduced `n_real_rows` delta = **8899482 (identical)** but over a **whole-call wall of 142.5s** (warmup 70.8s + measure 71.8s). True rate = `8899482 / 142.5 = 62,452`.
- `LPD=835` is the same disease: `leaves(whole call) / decisions(measure window)`. Window-consistent LPD ≈ **421**, and `dps_measure(148.4) × 421 ≈ 62k` — matching the server side.

### Each claim adjudicated

1. **Recorded ~140k — WRONG (artifact).** `leaf_rows_s` was computed as whole-call leaves ÷ measure-window wall. Not a server-side steady-state count. The DB `lpd=835` column is contaminated the same way and should not be used as a multiplier.
2. **Prior server-side 62–76k — essentially RIGHT.** `ref_cpu.py` (os.times, non-invasive) is sound. My independent fine-grained probe gives whole-call **62,440** and steady-state plateau median **68,786 leaf-rows/s** — squarely in their band. (Caveat: `ref_util.py` is dead — it reads `server.compute_s`, which does not exist on the server; that was the removed perturbing block. Ignore it.)
3. **Bench dps→120k — WRONG, same artifact.** `48 dps/core × 3 × 835 LPD = 120,240` reused the inflated 835. With a window-consistent LPD (~421), `48×3×421 ≈ 61k`. The bench's own `dps` is fine; only the ×835 multiplication was bad.

### True throughput + saturation

- **Defensible true rate at N9: ~62,000 leaf-rows/s whole-call, ~65,000–69,000 steady-state** (server `n_real_rows` ÷ matching wall; consistent three ways — server counter rate, plateau median, and dps×true-LPD).
- **Server core is NOT saturated.** Steady-state server-thread CPU sits at **~73–79% of one core** across three independent measurements (prior 73.3%; my 0.5×peak window 75.0%; my 0.8×peak window 78.6%), never pinned at 100% during the plateau. ~21–27% idle headroom ⇒ the bottleneck is upstream (the real Gumbel search + wire round-trips on producer cores 1,2,3), not the JAX forward. The 73% is a real steady-state, not an artifact — it holds flat for ~115s of plateau in the timeseries.

### Could not verify / flags

- I could not locate the exact script that wrote DB rows 16/17 (no `leaf_rows_s`/`lpd` computation exists in the committed `cpp/stage_a/*` or `throughput-lab/harness/exp_db.py`; it was an ad-hoc scratchpad step). But the provenance is pinned by exact arithmetic match, so attribution is certain regardless: `leaf_rows_s = whole_call_leaves / measure_wall`.
- My fresh run's measure wall (71.8s) differs from the recorded 63.3s — ordinary run-to-run variance in the measure pass; it does not affect the structural diagnosis (the `leaves` numerator reproduced bit-identically at 8899482).
- Recommend the DB readings 16/17 be marked retracted/superseded and any `tlab_finding` citing 140k be corrected to **~62–69k leaf-rows/s, server core ~75% (unsaturated)**, with `lpd` recomputed window-consistently (~421, not 835).
