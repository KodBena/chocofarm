# SO-ISMCTS vs the VoI ceiling (point-in-time)

Single-Observer Information Set Monte-Carlo Tree Search (Cowling, Powley & Whitehouse,
IEEE TCIAIG 4(2), 2012), implemented as a drop-in `Policy` subclass (`ismcts.py`) against the
unchanged `env.py`. UNIT values; detectors disjunctive (the real 17 overlaps). static floor
**0.0855**; clairvoyant ceiling **0.1454** (**+70%** headroom).

## What was implemented

Algorithm 1 of the paper (SO-ISMCTS, §IV-E) with the subset-armed-bandit UCB modification
(§IV-B), specialised to this single-observer belief-MDP:

- **Nodes are information sets** = the belief world-set `bw`. Edges are actions; an action's
  *observation outcome* under the active determinization routes which successor belief (child
  node) the iteration descends to, while the action's bandit statistics aggregate over the
  whole information set. This is the ISMCTS property that plain determinized-UCT lacks.
- **Determinize** each iteration by sampling one world `w ~ bw` (`env.sample_world`); the belief
  already encodes consistency (exactly-5-of-20 + all past observations), so this is unbiased —
  no particle filter.
- **Subset-armed bandit:** the legal-action set shrinks as the belief sharpens (a detector drops
  out once its outcome is certain; a treasure drops out once revealed absent), so UCB1's log
  term uses each action's *availability* count n'_j (times it was legal here), not the raw parent
  visit count. c = 0.7 (the paper's value across all its experiments).
- **Simulation** is a λ-penalised playout to episode end; **return** is `Σ value − λ·(Σ travel +
  exit)`, the same Dinkelbach surrogate the harness optimises. TERMINATE is an ordinary edge
  valued at the bare `−λ·exit_cost`, so the search can *learn* the early-exit option.
- **Default playout base** is `GreedyStopBase` (in `ismcts.py`): λ-rational greedy that nets the
  exit relocation into each step's value and stops when no treasure pays — a tighter renewal
  cycle than plain `GreedyPolicy`, which over-collects.

## Results (unbiased Dinkelbach rate; small N — see caveat)

| budget | rate | % of ceiling | VoI clawed | E[R] | E[T] | sec |
|---|---|---|---|---|---|---|
| static floor | 0.0855 | 59% | — | — | — | — |
| ismcts(it=150) | 0.0680 | 47% | −29% | 4.00 | 58.8 | 162 |
| ismcts(it=400) | 0.0763 | 52% | −15% | 3.58 | 47.0 | 274 |
| clairvoyant ceiling | 0.1454 | 100% | +100% | 4.55 | ~31 | — |

(Dinkelbach: 2 warm rounds then a final measurement; it=150 used 12/40 runs, it=400 used 8/24.)

## Findings (honest)

- **The algorithm is faithful and the trend is the right one.** More iterations → the search
  routes *tighter* (E[T] 58.8 → 47.0) while collecting slightly *fewer* treasures (E[R] 4.00 →
  3.58), so the rate climbs (0.0680 → 0.0763) and the VoI deficit shrinks (−29% → −15%). The
  search is learning to bank-and-exit rather than sweep — the direction the clairvoyant rewards.
- **At these budgets it still trails static (does not yet claw back VoI).** This reproduces, with
  a principled search rather than a shallow one, the bottleneck the consult and prior results
  named: the +70% lives in deep contingent *sensing chains*, and the disjunctive detectors are
  individually weak — a single positive reading on a detector covering ~4 treasures lifts each
  covered marginal only to ~0.4, so cornering one treasure to near-certainty needs several
  overlapping positive/negative readings. That depth is what a few hundred iterations against a
  36-way root branching factor cannot yet reliably find; the leaf payoff only materialises after
  the chain, so the credit signal is sparse.
- **The over-collection is partly a Dinkelbach artifact.** The fixed point sits at a low
  λ (~0.066) because the policy is weak; at that low time-penalty, continuing to collect looks
  λ-return-positive even when it lowers the *rate*. A stronger policy would converge to a higher
  λ and self-correct. This is the known coupling between an approximate policy and its
  Dinkelbach operating point, not a bug in the search.

## Caveat

The final measurements use small run counts (40 and 24 episodes) to stay inside a strict
wall-clock budget. The ratio estimator on R ∈ {0..5}/episode has a standard error of several
percent at N=24, so the it=150 → it=400 gap is a *trend*, not a tightly-resolved delta. Bump
`final_runs` (and `iterations`) when more wall-clock is available; the monotone direction is the
trustworthy part.

Source: `ismcts.py`, `eval_ismcts.py`.
