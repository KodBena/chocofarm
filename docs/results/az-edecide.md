# E-DECIDE — AlphaZero value-substitution probe (machinery + smoke, 2026-06-13)

The decisive first AlphaZero experiment from `docs/design/alphazero-surrogate-design.md` §9:
**does a learned λ-penalized value, dropped into the ISMCTS leaf in place of the determinized
playout, beat the playout leaf at matched budget?** (design §1 H-calibrate, §7 ablation #1).

This document records the **machinery build + a tiny correctness smoke**, plus the exact
commands the orchestrator should run for the full bounded experiment. It is **NOT the result** —
the smoke numbers are deliberately too small to decide anything.

---

## (a) Staleness-adaptation note — the honest env vs the doc's stale model

The design doc's §2.2/§3 dimensions (16 detectors, 37-slot action space, ~90-float feature
vector, "zero singleton detectors" / cover sizes 2–5) are from the **SUPERSEDED 16-region
detector model**. The honest `env.py` carries the arrangement-FACE sense model. Measured
directly from `env` (`taskset -c 3` probe, read-only):

| quantity | doc (stale) | honest env (measured) |
|---|---|---|
| treasures `env.N` | 20 | **20** |
| present `env.K` | 5 | **5** |
| detectors `len(env.detectors)` | 16 | **44 faces** |
| singleton-cover detectors | 0 | **21 of 44** |
| cover sizes (min/max/mean) | 2–5, mean 3.12 | **1–4, mean 1.66** |
| teleports | 3 | **3** (`CSNE`, `CSCE`, `tau_4`) |
| worlds C(20,5) | 15,504 | **15,504** |

Consequences, all handled by deriving dimensions from `env` (never hardcoded):

- **Feature dim = 220**, not 90: `env.N·4 + len(env.detectors)·3 + (5 + n_teleports)` =
  `20·4 + 44·3 + 8 = 80 + 132 + 8`. The larger size is expected — faces are far more numerous
  than the stale regions. `features.feature_dim(env)` reports it; nothing downstream assumes 90.
- **Action-space size = 65** (for a policy head, were one built): `N` collects (20) + `len(env.detectors)`
  senses (44) + 1 TERMINATE. The doc's "37 slots" is stale. The E-DECIDE probe uses **no
  policy head** (value is the load-bearing head, design §1/§9), so the action-space size is
  recorded for completeness, not exercised here.
- The "zero singletons" claim (F2) does NOT hold on the honest model — nearly half the faces
  are singleton covers (a singleton-cover face read == observing one treasure's presence). This
  does not change the featurization (the open-clause `informative[i]` indicator and `p_pos[i]`
  are computed per-face regardless of cover size); it is noted as an honest correction to the
  doc's F2.

---

## (b) Smoke results (NOT THE RESULT — too small to decide)

All runs pinned `taskset -c 3`, bounded, under `timeout`. Venv:
`/home/bork/w/vdc/venvs/generic/bin/python` (numpy only; no new deps).

**Stage-1 smoke (value learnability):** 40 decomp-teacher episodes → 592 transitions; value
head trained 120 epochs, batch 64, lr 1e-3, L2 1e-4, 25% held-out.

- **held-out R² ≈ 0.46–0.49, MAE ≈ 0.51** (target mean −0.24, std 0.95, on the λ-penalized
  return-to-go scale).
- Read: V_λ is **clearly learnable** from the §2.2 features on the honest 44-face env even from
  this tiny set — a positive signal for Decision Gate 1, and an empirical re-confirmation of the
  doc's F6 marginal-sufficiency claim **on the honest model** (F6 was measured on the stale
  16-region model). The full ~300-episode dataset (~4–5k transitions) should lift R²
  materially; do not read the smoke R² as the gate verdict.

**Stage-2 smoke (machinery runs):** `NetValueISMCTS` with the smoke weights, it=200, **N=15**,
baseline skipped:

- net-value leaf: dinkelbach rate **0.0616** (E[T]=68.2), fixed-λ₀ rate **0.0679** (E[T]=53.0),
  fixed-λ₀ %VoI = −30%.
- Read: the probe **produces a finite rate without error** — that is the whole point of the
  smoke. The rate itself is meaningless at N=15 with a 40-episode teacher (well below the design's
  no-small-N threshold and trained on almost no data). **Not a GO/NO-GO verdict.**

---

## (b-result) FULL E-DECIDE RESULT (orchestrator-driven — the actual verdict)

Run on a freed core (solver daemons paused), streamed to TensorBoard
(`tb/az_edecide_train`, `tb/az_edecide_eval`). Stage-1 dataset = 300 decomp episodes
(4,421 transitions); value head 200 epochs. Stage-2 = interleaved paired (same-seed)
net-value vs playout-leaf at fixed λ₀=0.0855, N=200, it=200.

**Stage 1 — Gate 1 (value learnability): PASS, moderate.** Held-out **R² = 0.413**,
MAE 0.562 (target mean −0.32, std 0.96). R² peaks ~0.44 by epoch 40 then plateaus — a
representational ceiling of the §2.2 features, not undertraining. The λ-value IS learnable
on the honest 44-face env (F6 holds), but the features leave ~59% of the return variance
unexplained.

**Stage 2 — the decisive comparison (fixed λ₀=0.0855, N=200):**

| leaf | rate | E[T] | %VoI |
|---|---|---|---|
| net-value (learned) | **0.0694** | 52.3 | −27% |
| playout (determinized) | 0.0665 | 42.5 | −32% |
| Δ (net − playout) | **+0.0030** | **+9.9** | +5 pts |

**Verdict: a marginal, mechanism-WRONG beat — a weak/provisional GO, not a clean one.**
The net-value leaf reliably (stable across the streamed run, not noise) edges the playout
leaf by +0.0030 / +5 VoI points — but it stays **well below the 0.0855 floor** (−27%), and
the over-collection signature goes the **wrong way**: net E[T] is *longer* (52.3 vs 42.5),
i.e. the learned value drives *more* over-collection, not less. The design's H-calibrate
mechanism (a calibrated value *cures* the F4 over-collection optimism → shorter E[T], higher
rate) is therefore **not confirmed** at iteration-0: the net gets its small edge by banking
more reward over a longer excursion, not by the hypothesized risk-calibration.

This matches the documented teacher-coverage caveat — decomp's myopic-macro policy never
visits the deep multi-detector sensing chains where the VoI lives, so the value net is
under-calibrated exactly there. Read this as **"the iteration-0 value from a non-exploring
teacher gives a marginal, mechanism-ambiguous lift"** — NOT proof a learned value cures F4
(it didn't here), and NOT a NO-GO (it didn't trail).

**Decision (per the maintainer's standing authorization — start the real run if reasonable,
restart later with more information):** value is learnable and net did not collapse → no red
flag → **proceed to the full Gumbel ExIt loop**, warm-started from this value net, flagged
**provisional**. The loop's Gumbel root exploration is the actual test of whether reaching
the deep-sensing beliefs converts this lukewarm iteration-0 into a real gain; the
feature-response diagnostic (`chocofarm/az/feature_response.py`) may redirect the
featurization first.

---

## (c) Exact commands for the FULL experiment (orchestrator runs these)

Each is a single bounded, core-3-pinned line. `PY` and `TS` for brevity:

```
PY=/home/bork/w/vdc/venvs/generic/bin/python
TS="taskset -c 3"
```

**Stage 1 — generate the full dataset (~300 decomp episodes ≈ 4–5k transitions; ~30–60s):**

```
$TS timeout 600 $PY -m chocofarm.az.dataset \
    --episodes 300 --out /tmp/az_data.npz --lam 0.0855 --seed 7
```

**Stage 1 — train the value head (report held-out R²/MAE = Decision Gate 1; ~1–2 min):**

```
$TS timeout 600 $PY -m chocofarm.az.train_value \
    --data /tmp/az_data.npz --out /tmp/az_value.npz \
    --epochs 200 --batch 256 --lr 1e-3 --l2 1e-4 --val-frac 0.2 --seed 0
```

**Stage 2 — measure net-value leaf vs playout leaf at matched budget (it=200, N=400 ≥ 300 for
<2% SE; this is the cost center — budget ~1–2 hours: ~800 episodes total across both policies ×
two rate readings × ISMCTS it=200 ≈ 0.9 s/decision):**

```
$TS timeout 7200 $PY -m chocofarm.eval.eval_az \
    --weights /tmp/az_value.npz --it 200 --n 400 --seed 7
```

(The Stage-2 line runs BOTH the net-value probe AND the playout-leaf ISMCTS baseline at the
matched it=200, plus the floor/ceiling reference lines and the read-out. Drop `--no-baseline`
for the full run — it is set only in the smoke.)

If you want even tighter SE or a second seed, re-run Stage 2 with `--seed 11 --n 400` and pool.

---

## (d) GO / NO-GO read-out criteria (design §9)

The eval prints the read-out automatically. Decide on the **fixed-λ₀ rows** (the operating point
the value was trained at, the apples-to-apples comparison):

- **GO** (full Gumbel ExIt loop justified) iff the **learned-value leaf strictly and
  significantly beats the playout leaf** at matched budget — ideally **clears the static floor
  0.0855** — AND shows the **ET-shrinking / less-over-collection signature** (net-value E[T] <
  playout-leaf E[T]). Even +5–10% of the +70% VoI gap clawed back *by the value swap alone* is a
  strong signal, since the full loop adds policy amortization on top. This is direct evidence
  H-calibrate cures F4 (calibrated value > optimistic playout).
- **NO-GO / rethink** iff the learned-value leaf **ties or trails** the playout leaf. Then the
  value is not the lever (it inherited the optimism, or the gate is genuinely search depth not
  the leaf), and a full loop is unlikely to pay — redirect to the consult's cheaper structural
  ideas (cluster-exact decomposition as a trusted anchor; recalibrated time model).

Reference lines for the headline %: static floor 0.0855, clairvoyant ceiling 0.1454, decomp
anchor ≈ 0.094 (the value teacher).

---

## Honest caveats

- **Value-label variant used:** ONLY the **honest realized λ-penalized return-to-go** of the
  decomp teacher's own episodes (design §4.5, the F4 cure). We did NOT additionally use an
  analytic decomp value-to-go — keeping the probe's calibration story clean and the labels a
  single, well-understood quantity. (Decomp does expose an exact λ-value via its micro tables;
  blending it as a lower-variance label is a deferred option, not exercised.)
- **Teacher substitution:** the design's Stage-1 names the **ISMCTS** policy as the dataset
  teacher; we used **decomp** instead. Decomp is both stronger (clears the floor, rate ~0.094 vs
  ISMCTS below it) and faster, so its honest realized returns are higher-quality, lower-
  over-collection labels for the *same* return-to-go quantity. This is a deliberate, documented
  deviation.
- **How the decomp teacher could bias the probe (the real risk):** the value net is trained on
  the *distribution of beliefs decomp visits*. Decomp's myopic-macro policy enters one best
  cluster, banks, and exits — it does NOT explore the deep multi-detector sensing chains where
  F3 says the VoI lives. So the value net may be **under-trained on exactly the deep-sensing
  beliefs the search most needs a calibrated leaf for**, and could extrapolate poorly (optimistic
  OR pessimistic) there. A NO-GO from this probe could therefore be a *teacher-coverage* artifact
  rather than a true H-calibrate failure. Mitigation for the full loop (design §6/§10): the real
  ExIt loop re-generates data under the net-guided search with Gumbel root exploration, which
  visits those beliefs; E-DECIDE only tests the *iteration-0* version (a value bootstrapped from
  the current expert). Read a NO-GO as "the bootstrapped value from a non-exploring teacher
  doesn't help," not "a learned value can never help."
- **Single instance, uncalibrated time model:** everything is conditioned on TELE_OH=12 and
  symmetric Euclidean travel; the whole conclusion could move under a recalibrated time model
  (design §10).
- **Smoke fidelity:** the smoke trained on 40 episodes and evaluated at N=15 — far below the
  thresholds the full commands use. The smoke proves correctness only.

---

## Module map

| module | role |
|---|---|
| `chocofarm/az/features.py` | §2.2 feature vector adapted to 44 faces; `feature_dim(env)`=220, env-derived; one cached `marginals` call per node |
| `chocofarm/az/mlp.py` | pure-numpy 2-layer MLP, value head (standardized targets, linear), manual Adam + backprop + L2, npz save/load; optional policy head (off) |
| `chocofarm/az/dataset.py` | (features, honest realized λ-return-to-go) pairs at fixed λ₀ from decomp-teacher episodes; CLI `--episodes --out --lam --seed` |
| `chocofarm/az/train_value.py` | trains the value head, reports held-out R²/MAE (Gate 1); CLI `--data --out --epochs --batch --lr --l2 --val-frac` |
| `chocofarm/az/netvalue_ismcts.py` | `NetValueISMCTS`: `ISMCTSPolicy` with ONLY the leaf eval swapped to `V_λ(features)`; same budget, byte-identical otherwise |
| `chocofarm/eval/eval_az.py` | Stage-2 eval: net-value vs playout leaf at matched budget, fixed-λ₀ + Dinkelbach rows, E[T], floor/ceiling/decomp lines, GO/NO-GO read-out |
