# Heterogeneous-value test — result (2026-06-13, point-in-time)

Test of the contention (STATUS.md §3) that adaptivity loses to static only because of
**unit values**, and that **heterogeneous gil values** would flip it. Method: synthetic
value vector, each policy's **true rate measured by unbiased Monte-Carlo** (`het_values_eval.py`).

Synthetic values: high(=10) at treasures {3, 9, 12, 17}, rest = 1.

| policy | rate | vs static | notes |
|---|---|---|---|
| value-aware static (greedy value/dist route, best prefix) | **0.2976** | — | visits {9,12,17} |
| greedy base | 0.2751 | −7.6% | E[R]=9.45, E[T]=34.35 (over-collects) |
| one-step rollout (own Dinkelbach λ) | 0.2852 | **−4.2%** | E[R]=7.91, E[T]=27.73, 2.28 detector-visits/run |

## Finding (honest)

Heterogeneous values **did not flip the outcome** — the adaptive policies still trail a
good value-aware static route. So "unit values mute the margin" is an **incomplete**
explanation. The recurring signature is over-collection + over-detouring: the adaptive
policies wander to detectors and low-value extras, netting a slightly lower rate than a
tight static route through the high-value cluster.

The most likely real bottleneck is **policy quality**, not value flatness:
- the rollout's **base policy is detector-blind greedy** — rolling out over a weak base
  caps the result. The obvious fix: **roll out over the static route as the base**, which
  is ≥ static by construction, so adaptivity can only add on top.
- shallow (one-step) lookahead may be too short for the multi-step contingencies that carry
  the human-obvious adaptive value (observe early → re-route → exit).

Consult-001 (`docs/consults/`) is examining exactly these implementation questions; the next
fix is deferred until its verdict lands, to avoid pre-empting the diagnosis.

## Caveat — τ_4

τ_4 was used heavily here (greedy 1732/2000) only because the synthetic weights placed a
high-value treasure (τ_3) next to the τ_4 teleport stone. **τ_4-dominance is
value-placement-dependent**, not the settled "never visited" seen under unit values.
