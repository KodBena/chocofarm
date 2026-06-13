# Pluggable policies vs the VoI ceiling (2026-06-13, point-in-time)

**Architecture (decoupled):** `env.py` — model, belief, dynamics, simulation, unbiased rate,
Dinkelbach (solver-agnostic) · `policies.py` — `Policy` ABC + `Greedy`, `CertaintyEquivalent`,
`Rollout`, `SparseSampling` · `run.py` — references + harness. A new method (NMCS, ISMCTS, a
learned policy) is a new `Policy` subclass; `env.py` does not change.

UNIT values; detectors disjunctive (the real 17 overlaps). static floor **0.0855**;
clairvoyant ceiling **0.1454** (**+70%** headroom).

| policy | rate | % of ceiling | VoI clawed |
|---|---|---|---|
| greedy | 0.0810 | 56% | −8% |
| certainty-equiv | 0.0751 | 52% | **−17%** |
| rollout(greedy) | 0.0892 | 61% | **+6%** |
| rollout(CE) | 0.0798 | 55% | −10% |
| sparse(d2, leaf=CE) | timed out at budget | — | — |

## Findings (honest)

- **The decoupling works.** All four policies run through one env/harness with no
  special-casing; NMCS/ISMCTS are now drop-in `Policy` subclasses.
- **Every current policy captures almost none of the +70%.** Best is rollout(greedy) at +6%.
- **The certainty-equivalent policy underperforms greedy (−17%).** It collapses the belief
  to a MAP present-set, but under the flat exactly-5-of-20 prior all marginals are 0.25, so
  the MAP is *arbitrary* — CE chases ~random treasures until the belief sharpens, and nothing
  sharpens it (CE gathers no information). It only helps with an already-sharp belief, which
  is the missing piece. Honest miss on the proposed base; rolling out over it is worse than
  over greedy.
- **The lesson, sharpened.** Capturing the +70% requires actively *gathering information*
  (cheap detectors to localize the present five) *then* routing — a multi-step contingent
  plan. One-step rollout over a belief-weak base can't see it; naive depth-2 sparse sampling
  is already too slow at a usable budget. The clairvoyant gets info free; a real policy must
  earn it through search depth.

## Next

This is the regime for deep contingent search: **NMCS** (single-agent nested rollouts) and
**ISMCTS** (information-set determinized MCTS), now implementable as `Policy` subclasses
against the unchanged env. Target: how much of the +70% does each claw back?
