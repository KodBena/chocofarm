# AZ Gumbel Expert-Iteration loop — machinery + smoke (2026-06-13)

The full AlphaZero-style **Gumbel Expert-Iteration loop** from
`docs/design/alphazero-surrogate-design.md` §5 (Gumbel search) and §6 (ExIt loop) — the "real
run" the E-DECIDE probe gates. This document records the **machinery build + a tiny correctness
smoke**, plus the exact command the orchestrator should run for the hours-scale real loop. It is
**NOT the result** — the smoke (I=2, E=4) is far too small to decide anything; it proves the loop
closes end-to-end without error.

---

## (a) What was built

| module | role |
|---|---|
| `chocofarm/az/mlp.py` | the pure-numpy MLP — **policy head now completed**: masked-softmax inference (`predict_policy`/`predict_both`) + a combined AlphaZero `train_step` = CE(masked) + MSE(value) + L2. The value-only `train_step_value` is unchanged (E-DECIDE still uses it). |
| `chocofarm/az/actions.py` | the fixed **action↔slot mapping** (one documented place; §(b)) + the two legal-mask paths (`legal_mask` from `env.legal_actions`, `legal_mask_from_features` the hot-path slice). |
| `chocofarm/az/gumbel_search.py` | **Gumbel-AlphaZero search** (design §5): Gumbel-Top-k root sampling, Sequential Halving, PUCT interior with net prior+value, **net value at leaves** (no playout), chance-node outcome-averaging (c=2), the improved-policy target π′; `GumbelPolicy` for eval. |
| `chocofarm/az/exit_loop.py` | the **ExIt loop** (design §6): generate→train→eval→checkpoint, replay buffer, TensorBoard streaming, warm-start from an E-DECIDE value net. The CLI real-run entry point. |
| `chocofarm/az/feature_response.py` | the value-head **feature-response diagnostic**: permutation importance (ΔR² when each feature is shuffled) + 1-D partial dependence, grouped by §2.2 block. |
| `tests/test_az_loop.py` | bounded correctness gate (mapping bijection, mask agreement, masked-softmax legality, train_step finiteness/reduction, well-formed Gumbel target). |

---

## (b) Action space & slot scheme on the HONEST env (design doc is stale here)

The design doc's §3 "37-slot" action space assumed the **superseded 16-region detector model**.
The honest `env.py` carries the arrangement-FACE sense model. Derived from `env` (never
hardcoded), the fixed action space is `env.N + len(env.detectors) + 1`:

| | count | slot range | action |
|---|---|---|---|
| collects | `env.N` = 20 | `0 .. 19` | `("t", i)` for slot `i` |
| senses (faces) | `len(env.detectors)` = 44 | `20 .. 63` | `("d", slot − 20)` |
| TERMINATE | 1 | `64` | `("term", None)` (always legal) |
| **total** | **65** | | |

This matches the E-DECIDE staleness note (`docs/results/az-edecide.md` §a) and the prompt's
correction. Feature dim is **220** (`features.feature_dim(env)` = `20·4 + 44·3 + (5+3)`), also
env-derived. The **legal mask** is read straight off the feature vector's known blocks
(`available[i]` for collects, `informative[j]` for senses, +TERMINATE) — so building the mask
costs only array slicing beyond the feature build that already happened. `test_az_loop.py`
asserts this feature-slice mask matches the authoritative `env.legal_actions` mask at root and
after a sensing read.

---

## (c) Loop design as built

- **Search (design §5):** Gumbel-Top-k samples `m` root actions without replacement on
  `logit + Gumbel`; **Sequential Halving** (Danihelka §2) runs `⌈log2 m⌉` phases, each phase
  splitting an equal `n_sims/⌈log2 m⌉` share among the current survivors and halving the set by
  `g + logit + σ(q̂)`; the full `n_sims` budget is spent (a rounding remainder goes to the last
  phase's survivors). The **executed action at temperature 0 is the SH survivor** (paper §2), not
  an argmax over the full considered set. Interior nodes use **PUCT**
  `Q + c_puct·P·√(ΣN)/(1+N)` with the net's masked prior `P` and Q the running mean (unvisited Q
  completed by the node's net value). **Leaves are the net value** `V_λ(belief)` directly — no
  determinized playout (the F4 cure, design §5.2). Chance nodes (observation outcomes) are
  averaged over `c_outcome = 2` immediate determinizations on the SO-ISMCTS info-set tree
  (`_belief_key` reused). Params: `m=12, n_sims=48, c_puct=1.25, c_visit=50, c_scale=1.0,
  c_outcome=2` (design §5.4).
- **Improved-policy target (design §4.4/§5.1):** π′ = softmax over legal slots of
  `logit + σ(completedQ)`, `σ(q) = (c_visit + max_a N(a))·c_scale·q`, unvisited legal actions'
  Q completed by Danihelka's `v_mix = (v_net + ΣN·v̄)/(1 + ΣN)` where `v̄` is the **prior-weighted**
  mean of visited actions' Q (paper §3), NOT the visit-weighted mean — the two differ sharply
  because SH makes visit counts unequal. This is the apprentice's policy target (well-defined
  even at n=48). Returned alongside the executed action by `decide_with_target`. The three
  Danihelka invariants (full-budget SH, executed==survivor, prior-weighted v_mix) are locked by
  `tests/test_az_loop.py` — they regressed in the first cut and were caught by an out-of-frame
  hack-rationalization audit; see §(f).
- **Loss (design §6, Silver et al. 2017):** `L = α·CE(p_net, π′) + β·(v_net − v_target)² + l2·‖W‖²`,
  defaults `α=1.0, β=1.0` (β raisable per design §1's value-is-load-bearing inversion), `l2=1e-4`.
  CE is masked (illegal slots carry zero probability in both p and π′, so no gradient).
- **Value target (design §4.1/§4.5):** the **honest realized λ-penalized return-to-go** of the
  episode under true partial-observation dynamics (suffix accumulation; the single exit toll
  charged in every suffix). λ is **pinned to λ₀ = 0.0855** (the static-floor rate) for the whole
  run. The value-target standardization (`y_mean`/`y_std`) is re-pinned to the replay buffer's
  running target stats each iteration so the MSE stays O(1) as the return distribution drifts.
- **Replay (design §6):** last-W-iterations window (default W=5), concatenated for training.
- **Exploration (design §6):** Gumbel's root sampling is the exploration (no Dirichlet); the
  EXECUTED action is sampled from π′ for the first `--explore-plies` plies (temperature 1), argmax
  thereafter, to diversify trajectories. The improved-policy TARGET is always the full π′.
- **Eval (design §6 step 3):** the greedy (argmax-π′) `GumbelPolicy` rate on a held-out seed at
  fixed λ₀, reported as % of the +70% VoI gap clawed:
  `%VoI = (rate − 0.0855)/(0.1454 − 0.0855)·100`.
- **Checkpoint (every iteration):** `net_iterNNN.npz` + `latest_net.npz` + `history.json`
  (per-iter rate, %VoI, E[T], policy CE, value MSE, value R², entropy, transition/buffer counts,
  per-stage timings, λ) — a timeout/restart loses nothing and the rate-per-iteration is
  inspectable.
- **TensorBoard (tensorboardX):** `eval/rate`, `eval/voi_pct`, `eval/ET`,
  `ref/{static_floor=0.0855, clairvoyant_ceiling=0.1454, decomp_anchor=0.0941}` reference lines,
  `train/{policy_CE, value_MSE, value_R2}`, `gen/exec_policy_entropy`.
- **Warm start (design §6 init):** `--init-weights <e-decide-net.npz>` copies the trunk + value
  head from an E-DECIDE value net (the policy head stays random, since the value net has none).
  Without it, both heads start random.

### Honest simplifications / approximations vs §5/§6

- **Interior chance widening:** outcome-averaging (`c_outcome=2`) is applied at the **immediate
  (leaf) outcome only**, as design §5.2 specifies; interior chance nodes use one determinization
  per simulation (the SO-ISMCTS contract — one world threads the descent). Full progressive
  widening at every interior node is not done (the design does not ask for it).
- **No base-playout blend early:** design §5.2 keeps "an option to blend a short base playout in
  early iterations before the value net is trained." Not implemented — the value head is warm-
  startable from E-DECIDE (a trained leaf from iteration 0), which is the cleaner version of the
  same intent. A cold-start run leans on a random value head for the first iteration; the loop
  recovers as the buffer fills. If a cold start proves slow, blending is the documented add-on.
- **No outer Dinkelbach re-pin automated:** design §6 step 4 allows 0–2 outer λ re-pins if the
  rate drifts far from λ₀. The loop runs at fixed λ₀ and reports the rate; re-pinning is a manual
  re-launch with a new `--lam` (kept manual — it is a 0–2× event, not worth the in-loop
  machinery). The post-hoc unbiased Dinkelbach fixed point can be measured with
  `env.dinkelbach_rate(GumbelPolicy(net, env))` on the final checkpoint.
- **Eval is fixed-λ₀ rate, not the full Dinkelbach loop, per iteration** (the Dinkelbach loop is
  ~5× the episodes; too costly every iteration). The fixed-λ₀ rate is the apples-to-apples
  operating-point number; run `dinkelbach_rate` on the best checkpoint for the headline.

---

## (d) Smoke evidence (NOT THE RESULT — too small to decide)

All runs pinned `taskset -c 2`, bounded, under `timeout`. Venv
`/home/bork/w/vdc/venvs/generic/bin/python` (numpy only; tensorboardX already present).

**End-to-end loop smoke** — `exit_loop` at I=2, E=4, W=2, epochs=2, m=6, n=16, eval N=8, λ₀=0.0855,
cold net:

```
iter 0/2  rate=0.0317 (%VoI=-90) ET=43.3  CE=3.196 vMSE=0.977 R²=0.164 H=0.34  [84 tr | gen 1s train 0s eval 2s]
iter 1/2  rate=0.0438 (%VoI=-70) ET=48.5  CE=3.227 vMSE=0.852 R²=0.238 H=0.22  [34 tr | gen 1s train 0s eval 2s]
```

Proves: episodes generate; **both heads train** (CE finite ≈3.2; **value MSE finite and
decreasing 0.977→0.852**; value R² finite positive); **eval prints a finite rate** (0.032, 0.044);
a **checkpoint is written every iter** (`net_iter000.npz`, `net_iter001.npz`, `latest_net.npz`,
`history.json`); a **TB event file appears** (`events.out.tfevents.*`). The rate itself is
meaningless — a cold net trained on ~120 of its own transitions for 2 iters. Negative %VoI at
iter 0 is expected (random policy below the floor). The low executed-policy entropy (H≈0.2–0.3)
is the post-fix signature: the survivor-execution + prior-weighted v_mix produce a sharper, more
decisive improved policy, as Danihelka intends (the pre-fix cut had H≈0.65).

**feature_response smoke** — on the smoke net against a 181-transition decomp held-out set:

```
baseline value R² = -6.8020
rank   feature             block                 ΔR²
  1    t12.marg            treasure/marg          0.01407
  2    d34.informative     detector/informative   0.01192
  ...
block aggregate (Σ ΔR²): detector/informative dominates (most negative when its features removed)
partial dependence (top 5): printed value sweep per feature
```

Proves the diagnostic **produces a ranked table + per-block aggregate + partial-dependence
sweeps** without error. The negative baseline R² is expected and not a finding: the smoke net is
a 2-iteration policy+value net evaluated on the *decomp teacher's* distribution (a distribution
mismatch on a barely-trained net). On a real-loop net evaluated on its own held-out roll, R²
will be positive (cf. E-DECIDE's value-only R² ≈ 0.46–0.49 from a tiny set).

**Per-iteration wall-clock (measured, full budget m=12/n=48, H=256, one core):**
~0.84 s/episode → E=300 generation ≈ **4.2 min/iter**, N=200 eval ≈ 2.8 min/iter, training is
seconds → **≈ 7 min/outer-iteration**. So I=40 ≈ **4.7 hours**; I=60 ≈ 7 hours.

---

## (e) Exact command for the REAL loop (orchestrator runs this)

Recommended hours-scale first real run: warm-start the value head from the E-DECIDE net, I=40,
E=300, W=5, full budget m=12/n=48, β=2.0 (value emphasis, design §1), eval N=200. Pin to a free
core under a generous timeout (≈5 h expected; 8 h ceiling for headroom). **Do NOT push.**

```
PY=/home/bork/w/vdc/venvs/generic/bin/python
TS="taskset -c <free-core>"            # e.g. core 2 if E-DECIDE/TB are on 0/3

$TS timeout 30000 $PY -m chocofarm.az.exit_loop \
    --init-weights /tmp/az_value.npz \
    -I 40 -E 300 -W 5 --epochs 2 --batch 256 \
    --m 12 --n-sims 48 --lr 1e-3 --l2 1e-4 --alpha 1.0 --beta 2.0 \
    --lam 0.0855 --explore-plies 4 --eval-n 200 --eval-seed 12345 --seed 7 \
    --tb-logdir tb/az_exit_loop --ckpt-dir /tmp/az_exit_loop_ckpt
```

(`--init-weights` is the E-DECIDE Stage-1 value net from `docs/results/az-edecide.md` §c. Omit it
for a cold start — both heads random — which adds an iteration or two of warm-up. If the first
real run looks promising and budget allows, a longer I=60 run, or an outer Dinkelbach re-pin
`--lam <achieved-rate>` if the rate drifts far from 0.0855, are the natural follow-ups.)

Inspect progress live in TensorBoard (`tb/az_exit_loop`) or post-hoc via
`/tmp/az_exit_loop_ckpt/history.json`. The headline number is the unbiased Dinkelbach rate of the
best checkpoint:

```
$TS timeout 1200 $PY -c "
from chocofarm.model.env import Environment
from chocofarm.az.mlp import ValueMLP
from chocofarm.az.gumbel_search import GumbelPolicy
env=Environment(); net=ValueMLP.load('/tmp/az_exit_loop_ckpt/latest_net.npz')
print(env.dinkelbach_rate(GumbelPolicy(net, env, m=12, n_sims=48), final_runs=3000))"
```

---

## (f) Out-of-frame audit: three Danihelka-fidelity bugs found and fixed

Before committing, an out-of-frame hack-rationalization-detector pass (run by a subagent that
did not see the implementer's reasoning) checked the search against Danihelka et al. 2022's
actual formulas. It found the "honest simplifications" above to be legitimate narrowings (each
names a concrete cost), but caught **three correctness deviations** in the first cut of the
root-selection machinery — mislabeled as faithful to the paper:

1. **Executed action ≠ Sequential-Halving survivor.** `_sequential_halving` halved a local copy
   of the considered set, so the temperature-0 executed action was an argmax over the *full*
   top-m by `g+logit+σ·q`, not the SH survivor — diverging in ~52% of decisions and letting a
   lucky high-variance single-sample action win (the exact failure SH exists to prevent). This
   was the **eval policy's** decision rule, so it biased every reported rate. **Fixed:** SH now
   returns the survivor and the executed action is that survivor.
2. **v_mix visit-weighted instead of prior-weighted.** The unvisited-Q completion used the
   visit-weighted mean of visited Q; the paper uses the **prior-weighted** mean. Under SH visit
   counts are deliberately unequal, so this corrupted the apprentice's policy *target* π′ on
   every decision. **Fixed:** `_v_mix` now uses `Σπ(b)Q(b)/Σπ(b)` over visited `b`.
3. **Sequential Halving schedule not the paper's.** A decrementing-divisor-over-remaining-budget
   loop under-spent rounds and dumped the remainder on the survivor. **Fixed:** `⌈log2 m⌉`
   phases, equal `n_sims/⌈log2 m⌉` per phase split among survivors, remainder to the last phase.

The value-target suffix accumulation in `generate_episode` was audited and found **correct** in
all terminal cases (TERMINATE-terminal, belief-empty, max-steps), with no off-by-one. All three
fixes are now pinned by `tests/test_az_loop.py` (`test_executed_action_is_sh_survivor`,
`test_vmix_prior_weighted`, `test_sequential_halving_spends_full_budget`) — the test gap that let
the deviations pass a well-formedness suite is closed. The combined `train_step` gradient was
separately finite-difference-verified (policy and value heads, max error ~1e-10).

A residual the audit flagged and we accept: the return-to-go logic is **duplicated** across
`exit_loop.generate_episode` and `dataset.py:_episode_transitions` (two writers). They agree
today; a future edit to one (e.g. a TD(λ) value-target blend, design §4.5 leaves it open) would
silently diverge them. If that blend is built, the suffix rule should be extracted to one shared
function first.

---

## Honest caveats

- **The smoke decides nothing.** I=2/E=4/N=8 with a cold net is a correctness check only. The
  rate, %VoI, and feature-response R² are all artifacts of the tiny scale.
- **Value-target honesty (the design's spine, §4.5):** the value target is the realized MC
  return-to-go under true partial obs — never determinized. This is bootstrapped through the
  net's own search, so early targets are noisy-but-honest (low, not inflated — the asymmetry that
  is the whole reason to expect AZ to behave unlike NMCS/ISMCTS, design §6). The loop does not
  *guarantee* the value escapes the F4 optimism over many iterations (design §10's first risk —
  the value can drift optimistic on under-sampled deep-sensing beliefs); the loop streams value
  R² and entropy so that drift is observable, but watching it is a human-in-the-loop duty, not
  automated.
- **Cold start leans on a random value leaf for iteration 0.** The base-playout-blend option
  (design §5.2) is not implemented; the recommended mitigation is `--init-weights` (an E-DECIDE
  value net as the iteration-0 leaf). A cold run is correct but slower to warm up.
- **Single instance, uncalibrated time model** (design §10): everything is conditioned on
  TELE_OH=12 and symmetric Euclidean travel; a learned surrogate is only as meaningful as the env.
- **Eval cost dominates iteration time** (N=200 ≈ 40% of the ~7 min). If wall-clock is tight,
  `--eval-n 100` halves eval cost at the price of a wider SE on the per-iter rate curve (the
  headline still comes from the post-hoc 3000-run Dinkelbach on the best checkpoint).
