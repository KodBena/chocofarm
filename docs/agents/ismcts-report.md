# ISMCTS agent — report (verbatim)

> The implementation agent's complete final message, reproduced verbatim. Commission in
> `ismcts-commission.md`. Recorded per the verbatim-record discipline; not retro-edited.

---

Done. Committed on `feat/ismcts`, not pushed, working tree clean, `env.py`/`policies.py`/`run.py` untouched.

## What was delivered

Three files in `/home/bork/w/vdc/chocobo-ismcts` (branch `feat/ismcts`):
- `ismcts.py` — `ISMCTSPolicy(iterations=300, c=0.7)` plus the default playout base `GreedyStopBase`.
- `eval_ismcts.py` — measures the unbiased Dinkelbach rate at two budgets, reports rate / %ceiling / %VoI-clawed.
- `docs/results/ismcts-result.md` — results with honest caveats.

## Algorithm as implemented + literature

I fetched and read the canonical paper (Cowling, Powley & Whitehouse, "Information Set Monte Carlo Tree Search", IEEE TCIAIG 4(2), 2012) — specifically Algorithm 1 (SO-ISMCTS, §IV-E), the subset-armed-bandit modification (§IV-B), and the UCB1 form (eq. 7). I implemented the **single-observer** variant faithfully, specialised to this single-observer belief-MDP (no adversary — only an "environment" resolving observations):

- **Nodes are information sets** = the belief world-set `bw`. Edges are actions; the determinization's observation outcome *routes which successor-belief child* the descent enters, while each action's bandit statistics aggregate over the whole information set (the property plain determinized-UCT lacks).
- **Determinize** each iteration by sampling one world `w ~ bw` — unbiased per the env note, since the belief already encodes consistency (no particle filter).
- **Subset-armed bandit:** because the legal-action set shrinks as the belief sharpens (a detector drops out once its outcome is certain; a treasure once revealed absent), UCB1's log term uses each action's *availability* count `n'_j` rather than the raw parent visit count. `c=0.7` (the paper's value across all its experiments).
- **Simulation** is a λ-penalised playout scored as `Σvalue − λ·(Σtravel+exit)` — the same Dinkelbach surrogate the harness optimises. TERMINATE is an ordinary edge so the search can learn the early-exit option. Most-visited root action is returned.

## Measured results (UNIT values; static floor 0.0855, clairvoyant ceiling 0.1454, +70% headroom)

| budget | rate | % ceiling | VoI clawed | E[R] | E[T] | sec |
|---|---|---|---|---|---|---|
| ismcts(it=150) | 0.0680 | 47% | −29% | 4.00 | 58.8 | 162 |
| ismcts(it=400) | 0.0763 | 52% | −15% | 3.58 | 47.0 | 274 |

**Trend is monotone in the right direction:** more iterations → tighter routing (E[T] 58.8→47.0) with slightly fewer treasures (E[R] 4.00→3.58) → higher rate (0.0680→0.0763) → smaller VoI deficit (−29%→−15%). The search is learning to bank-and-exit rather than sweep.

Runtime: ~3.7 ms/iteration; a decision is ~0.55 s at it=150, ~1.5 s at it=400; each row above is a full Dinkelbach evaluation.

## Honest caveats

- **At these budgets ISMCTS still trails the static floor — it claws back none of the +70% yet** (it reached 47–52% of the ceiling, but that is below static's 59%). This reproduces, with a principled search, the bottleneck the project's consult-001 and prior results identified: the +70% lives in deep contingent *sensing chains*, and the disjunctive detectors are individually weak (one positive reading on a ~4-treasure cover lifts each marginal only to ~0.4, so cornering a treasure needs several overlapping readings). That depth is hard to reach with a few hundred iterations against a 36-way root branching factor — the leaf payoff only appears after the chain, so credit is sparse. I verified directly that after the belief is sharpened by ~8 detectors, base-playout returns flip strongly positive (+1.0 at λ=0.08), confirming the value is real but gated behind search depth.
- The over-collection is partly a **Dinkelbach artifact**: the fixed point sits at a low λ (~0.066) because the policy is weak, and at a low time-penalty continuing to collect looks λ-return-positive even when it lowers the *rate*. A stronger policy would converge to a higher λ and self-correct. Not a bug in the search.
- **Small N:** final measurements use 40 and 24 episodes to stay inside a strict wall-clock budget; the ratio estimator on R∈{0..5} has several-percent SE at N=24, so the it=150→it=400 gap is a trustworthy *trend*, not a tightly-resolved delta. `eval_ismcts.py` documents bumping `iterations`/`final_runs` when more wall-clock is available.

I worked exclusively in the worktree, kept all runs under `timeout`, ran no parallel solves, and never enumerated the belief space (the tree is bounded by the iteration budget). Network was available; I confirmed the algorithm against the original paper.

Source: [Information Set Monte Carlo Tree Search (Cowling, Powley & Whitehouse, 2012)](https://eprints.whiterose.ac.uk/id/eprint/75048/1/CowlingPowleyWhitehouse2012.pdf).
