<!--
docs/notes/leaf-eval-loop/step-3-nsweep.md
Purpose: Step-3 of the leaf-eval impl->model loop, after the static-reconciliation correction. Sweeps the
  overcommit lever (--trees-per-thread N) to fill the serve pad and test the regime-correct model
  (serve_sawtooth, padmax pad=512). Records what is SOLID (the fill lever works; drain-all overflows the
  compiled max_batch at N>=3) vs PROVISIONAL (a dps-grows-slower-than-B signal, a T_io-scaling hypothesis).
ADR-0005 point-in-time record; ADR-0006 header; claims-measured-vs-interpreted (every number tagged).
Public Domain (The Unlicense).
-->

# Step-3 N-sweep — the fill lever, the max_batch wall, a provisional T_io lead (2026-06-23)

After the static-reconciliation **correction** (the model is sound; the implementation runs *under-filled* at
real≈96 of a 512 pad), the path was: overcommit (`--trees-per-thread N`) fills B toward the pad, and
`serve_sawtooth(B, pad=512) ≈ 0.87·B` predicts the throughput. This sweep tests it — `all_allow`, padmax,
drain-all (chunk_floor OFF), `--secs 8`.

## Result 1 — the fill lever works, but caps at the max_batch wall

| N | slots (192·N) | B (rows/fwd) | dps_window | model 0.87·B | dps/model |
| --- | --- | --- | --- | --- | --- |
| 1 | 192 | 88.7 | 86.5 | 77.2 | 1.12 |
| 2 | 384 | 150.5 | 102.4 | 130.9 | 0.78 |
| 3 | 576 | *CRASH* | — | — | — |

- **Overcommit fills B** (88.7 → 150.5 from N=1→2) — the lever works (M).
- **N≥3 CRASHES** (M): drain-all gathers *all* ready slots into one batch (576, 768 rows), exceeding the
  AOT-compiled `max_batch=512` → `TypeError: x compiled with float32[512,241], called with [576,241]` → the
  server forward dies, the producer + lab wedge (CPU idle). So **the drain-all fill is structurally capped
  below the 512 pad** — the model's full-fill ceiling (~445) is **not reachable** by overcommitting in the
  simple drain-all path.

## Result 2 — the structural finding: the fill ceiling is gated by the serve machinery / control

Reaching B→512 (the model's ceiling) needs ONE of:

- a **drain cap** — the server gathering `min(ready, max_batch)`, leaving the rest for the next forward — a
  serve-path fix that lets overcommit fill B→512 without overflow; OR
- the **chunk_floor (depth) path** — where the server caps width and overcommit adds depth — which is the
  **convoy regime** where `AllAllow` collapses (~11 dps, the prior finding) and the **controller** is what
  recovers it.

So the static fill ceiling is **entangled with the serve machinery / the control**: the simple
control-isolated drain-all path caps the fill below the pad; reaching the pad needs a drain-cap fix or the
control path. This sharpens the project's question — **the model's ceiling is not reachable by a naïve static
fill.** (My sweep design was wrong: drain-all assumes slots ≤ max_batch; high-N overcommit was always a
chunk_floor-path operation.)

## Result 3 — a PROVISIONAL lead on T_io (the top measurement target)

In the reachable range, **dps grew slower than B**: B rose 1.70× (88.7→150.5) but `dps_window` rose only 1.18×
(86.5→102.4), so the model's linear `serve∝B` over-predicts (dps/model went 1.12 → 0.78). `fwd/s` was ~constant
(339→360), so the cycle did **not** shrink — yet dps did not track B. **One hypothesis:** `T_io` (the
**UNMEASURED** drain/decode/encode/scatter term) **scales with the coalesced rows B** — more rows → more T_io →
the cycle grows with B → diminishing dps the model (constant T_io) misses. That would be a direct lead on the
model's **top Neyman target**.

**PROVISIONAL — not established.** Two points; `dps_window` carries ~±10% run-to-run noise (this N=1 read 86.5
vs the Step-2 N=1's 99.5 at the same config); the `dps_samp` variance is huge (±283 at N=2). The
dps-grows-slower-than-B *direction* is suggestive; the T_io-scaling is a *hypothesis*, one explanation among
several (redundant overcommit search; producer-feed limits), not a finding.

## Honest accounting (claims-measured-vs-interpreted)

- **Measured (M):** B(N=1)=88.7, B(N=2)=150.5 (means over ~2700 forwards — stable); the N≥3 overflow (the
  traceback). Artifacts: `~/w/vdc/chocobo/runs/control_lab/step3-nsweep/`.
- **Noisy (I):** `dps_window` (~±10% run-to-run); the dps-vs-B relationship is suggestive, not established.
- **Hypothesis (I):** `T_io` scaling with B — one candidate explanation of the provisional signal.

## Next steps (proposed — the maintainer decides)

1. **Firm the T_io signal** (cheap, existing instrument): re-run N=1 vs N=2 with `CHOCO_EVENTLOG` + longer
   windows; decompose `dt_us` (compute — should hold ~2204) vs the inter-forward gap (T_io/idle) — *does the
   gap grow with B?* If yes, T_io scales with B (a real form/grounding finding on the top measurement target).
2. **The drain-cap fix** (to reach the ceiling): cap the drain at `max_batch` so overcommit fills B→512 — a
   serve-path improvement that closes the throughput gap *and* makes the model's ceiling reachable/testable.
   A C++ change to the serve path (the maintainer's call).
3. Document and move on (the structural finding — the fill ceiling is gated/entangled — stands on its own).

## Step-3 (post-fix) — the ceiling test, RUN (2026-06-23)

The drain-overshoot was **fixed** (the drain caps at max_batch by deferring a straddling request; `_drain`,
commit `dd8fa21`, regression-tested), so the sweep runs past N=2. Re-run (`step3-nsweep-fixed/`, AllAllow,
padmax, `--secs 8`):

| N | slots | B (rows/fwd) | dps_window | model 0.87·B | dps/model | crash |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 192 | 95 | 98.6 | 82.6 | 1.19 | no |
| 2 | 384 | 160 | 65.2 | 139.1 | 0.47 | no |
| 3 | 576 | 210 | 86.6 | 182.4 | 0.47 | no |
| 4 | 768 | 277 | 140.4 | 240.8 | 0.58 | no |
| 6 | 1152 | — | — | — | — | **producer crash** |

- **The fill lever works** (M): overcommit fills B **95 → 277** (2.9×) across N=1→4 — clean, monotonic. N=3,4
  (576, 768 slots), which crashed pre-fix, now run clean.
- **But dps grows SUB-LINEARLY**: dps **99 → 140** (1.4×) — about *half* B's growth. The model's `serve ∝ 0.87·B`
  **over-predicts**: realized is ~0.47–0.58× the model at N≥2 (vs 1.19× at N=1). **The model's full-fill ceiling
  (~445 at B=512) is not reached** — at B=277 the realized 140 is far below the model's 241, and the trend is
  sub-linear, so B=512 would not reach it either.
- **This firms the provisional T_io lead** (Result 3 above): the 2-point hint (dps slower than B) now holds
  across **4 points**. The model (constant `T_io`) is missing a cost that **GROWS with the coalesced rows B** —
  the `T_io` (drain/decode/scatter, the UNMEASURED term) scaling hypothesis. **Direction robust (4 points);
  magnitude provisional** — `dps_window` is noisy (the N=2 dip to 65; `dps_samp` ±230–450; the 8 s window is
  transient-prone at high overcommit). The deferral my fix added is *not* the cause: the cap barely binds at
  B≤277 < 512, so most drains don't defer.
- **N=6 (1152 slots): a SEPARATE producer-side crash** — the C++ producer's pool warmup fails (`zmq_msg_recv …
  Resource temporarily unavailable`), not the server fix. It caps the static sweep at N≤4 (B≈277) for now
  (BACKLOG: "producer pool-warmup fails at very high overcommit").

**Honest accounting (claims-measured-vs-interpreted):** B (M, means over ~2000 forwards, clean). `dps_window`
(M but ~±10–15% run-to-run + transient-prone — the sub-linear DIRECTION is robust over 4 points, the exact
ratios provisional). The `T_io`-scaling MECHANISM is a hypothesis (candidates: `T_io ∝ B`; a producer-feed
limit at high B; redundant overcommit search). The N=6 crash is producer-side (verified: the producer log, not
a server `TypeError`).

**What it means for the project's question:** the model's optimistic ceiling is **not reached by filling the
pad** — static throughput grows sub-linearly in the batch width, so there is a real, *unmodeled* cost that
**scales with B**. That cost — not the operating point (corrected earlier) and not a coordination idle
(refuted at Step 2) — is where the model and reality now diverge, and it is the next thing to measure.

**Next (Step 4, to firm the mechanism):** an eventlog decomposition at N=2 vs N=4 (the existing `CHOCO_EVENTLOG`
`FWD`/`DRAIN` instrument) — does the inter-forward gap (the `T_io`/idle between forwards) grow with B? If yes,
`T_io ∝ B` is the form/grounding fix (and a direct measurement of the model's top unmeasured term). Longer
windows would also cut the `dps_window` noise.
