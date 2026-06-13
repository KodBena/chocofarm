# AlphaZero-surrogate design agent — report (verbatim)

> The agent's complete final message, reproduced verbatim. Commission in
> `alphazero-design-commission.md`; full design doc at
> `docs/design/alphazero-surrogate-design.md`. Recorded per the verbatim-record discipline.

---

Committed cleanly on `feat/alphazero-design` (898974a), not pushed, working tree clean, forbidden files untouched. The pinned processes on cores 0–2 were never disturbed (all my python ran under `taskset -c 3` and `timeout`).

Here is the complete record.

---

# AlphaZero-style learned surrogate for chocofarm — design investigation

**Deliverable:** `/home/bork/w/vdc/chocobo-azdesign/docs/design/alphazero-surrogate-design.md` (committed on `feat/alphazero-design`, not pushed).

## Verdict in one line
The design space is **narrow** and I produced a concrete implementer-ready spec. The idea is *plausible but not safe*: every prior search method sits below the static floor, and the entire approach rests on one falsifiable claim — that a **learned value calibrated on honest realized returns** cures the determinization optimism that sank NMCS/ISMCTS. I designed a ~4 core-hour first experiment that decides this before committing to a full loop.

## Key empirical findings (small read-only checks against `env.py`, all on core 3)
These reshaped the design more than the literature did:

- **The disjunctive structure is intact.** Detector cover sets are size 2–5, mean 3.12, **zero singletons** — the consult's feared geometry bug did NOT survive the fix. This is the regime the project is about.
- **Marginals(20) + collected-mask are nearly a sufficient statistic on-distribution.** Across 4,000 realistic mixed histories bucketed by rounded marginal vectors: **exactly 1 collision / 3,805 buckets**, and that one is resolved by adding the collected-mask. The naive "marginals are hopelessly lossy" worry is empirically overstated here — filtering bakes the disjunctive deduction *into* the marginals (drives them to 0/1).
- **Belief size spans 1 … 15,504** (median ~118, p90 ~7,260). This **kills DeepSets-over-worlds** as the primary encoder — infeasible at episode top AND unnecessary per the collision result.
- **Compute:** base playout 3.1 ms, `marginals` 1.2 ms (the per-node bottleneck — it's numpy, not the net), ISMCTS it=200 ≈ 0.9 s. **No GPU warranted.** torch absent from the venv but torch-CPU pip-installable; a numpy-MLP is the recommended zero-dep path.

## Recommended spec (the narrow answer)
- **Features:** ~90-float fixed vector — per-treasure (marg, collected, available, dist), per-detector (open-clause flag, p_pos, dist), globals (log|belief|, exit/teleport geometry). No set-encoder in the baseline; a DeepSets-over-*clauses* (not worlds) held in reserve only if an ablation shows a gap.
- **Architecture:** plain MLP 90→256→256, masked policy head over a fixed 37-slot action space, linear value head. ~100k params.
- **Value target (the question that's easy to get wrong):** the **λ-penalized differential value at a *fixed* λ** (pinned at the static-floor rate 0.0855), with the target being the **honest realized Monte-Carlo return-to-go under true partial observation** — never a determinized best-case. This is theoretically clean: the Dinkelbach λ *is* the average-reward gain, and the λ-penalized value *is* the differential value function. Direct-rate targets rejected (rate is non-additive, breaks bootstrapping). Outer Dinkelbach re-pin 0–2× if the rate drifts.
- **Search:** Gumbel-AlphaZero root (Gumbel-Top-k action sampling + Sequential Halving + completed-Q improved policy) — chosen specifically because it **guarantees policy improvement at low simulation budgets**, layered on the existing SO-ISMCTS information-set scaffold; net value replaces the optimistic determinized playout at leaves; observation outcomes are chance nodes.
- **Loop:** single-agent ExIt (no self-play); fresh i.i.d. world re-roll is the data augmentation; AlphaZero CE+MSE+L2 loss with value emphasis β≥1.

## The decisive first experiment (E-DECIDE, ~4 core-3 hours)
Isolate the load-bearing claim from the expensive loop: **swap a learned V_λ for the determinized-playout leaf in the *existing* ISMCTS at matched budget.** Stage 1 confirms V_λ is learnable from the features (F6 predicts yes); Stage 2 measures the rate. **GO** for the full Gumbel ExIt loop iff the learned-value leaf significantly beats the playout leaf (ideally clears static) with the ET-shrinking / less-over-collection signature — direct evidence the calibrated value cures the optimism. **NO-GO** → redirect to the consult's cheaper cluster-exact decomposition. A GO result is iteration-0 of the real loop, so it's not throwaway.

## Honest risks I flagged against the approach
The value can *inherit* the optimism if late-loop training data is dominated by the policy's own confident trajectories (it could quietly become "NMCS with a net"); the +70% clairvoyant ceiling is unattainable (it pays nothing for information, real sensing chains cost travel); and the whole conclusion is conditional on the uncalibrated time model (TELE_OH=12, symmetric Euclidean).

## Literature consulted
AlphaZero (Silver 2017 — loss/PUCT), **Gumbel AlphaZero** (Danihelka 2022 — the low-sim-budget fit, completed-Q improved policy), **ExIt** (Anthony/Tian/Barber 2017 — the no-self-play plan/generalize frame), SO-ISMCTS (Cowling 2012), POMDP belief-as-sufficient-statistic (Kaelbling 1998), DeepSets (Zaheer 2017), and average-reward/differential-value RL + Dinkelbach⟺average-cost-MDP equivalence (grounds the value-target choice).

Constraints honored: cores 0–2 untouched, all python `taskset -c 3` + `timeout`, belief space never enumerated, `env.py`/`policies.py`/`run.py` unchanged, staged by explicit path (no `git add -A`), not pushed.
