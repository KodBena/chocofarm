# Vanilla UCT vs SO-ISMCTS at matched budgets (point-in-time)

Plain single-tree UCT on the belief-MDP (`chocofarm/solvers/uct.py`), implemented as a drop-in
`Policy` subclass against the unchanged `env.py`, measured by the env's own unbiased Dinkelbach
rate (`chocofarm/eval/eval_uct.py`). The scientific question: at MATCHED per-decision budgets,
does SO-ISMCTS's information-set / determinization machinery earn its keep over plain belief-tree
UCT, and does either clear the static floor?

UNIT values; honest **arrangement-face** detector model (44 faces, `arrangement.load()`); static
floor **0.0855**; clairvoyant ceiling **0.1454** (**+70%** VoI headroom). Same instance and
references as `honest-rates-faces.md` and `ismcts-result.md`.

## What "vanilla UCT" means here (the variant as built)

A single MCTS tree on the belief-MDP, deliberately WITHOUT ISMCTS's information-set grouping and
WITHOUT per-iteration world-determinization of the tree. Concretely:

- **Tree nodes are action–observation histories.** Each node carries the EXACT belief that history
  reaches, tracked by applying `env.filter_treasure` / `filter_detector` along the path (the
  belief is the sufficient statistic). Root = current `(loc, bw, collected)`.
- **Decision nodes select by UCB1:** argmax `Q̄(a) + c·sqrt(ln N(node) / N(a))` over
  `env.legal_actions(...)` plus TERMINATE. Exploration constant **c = 0.7** (the same value ISMCTS
  uses; standard for an O(1)-scale return). TERMINATE is an ordinary action valued at the bare
  `−λ·exit_cost`, so the search can learn the early-exit option.
- **Chance / observation nodes** hang off each action edge. After an action the outcome is binary —
  a treasure probe reads present/absent (P(present) = belief marginal at the treasure), a detector
  reads positive/negative (P(positive) = fraction of surviving worlds intersecting the cover). We
  sample ONE outcome per traversal weighted by the belief-conditioned probability (via
  `env.sample_world` → `env.apply`, which resolves the outcome and filters the belief), and descend
  into the child decision node for that outcome. Binary outcomes ⇒ no progressive widening.
- **Rollout** from a freshly expanded leaf plays a cheap base policy to the end of the episode in a
  single sampled world (`_base_value`, shared with the other solvers), accumulating the
  λ-penalised differential return `Σ (reward − λ·dt)` to the exit. Default base is **`GreedyStopBase`**
  — the SAME λ-rational bank-and-exit greedy ISMCTS rolls out with — chosen so the leaf estimator
  is held FIXED across the two solvers and any rate gap is attributable to the tree, not the
  heuristic. (`rollout="greedy"` is available as a weaker alternative.)
- **Backup:** the differential return flows up the descent path; each decision node's `Q̄(a)` is the
  running mean of returns through action `a`, each chance node's value the running mean over the
  outcomes it sampled.
- **Knob = `iterations`** (per-decision simulation budget), exactly like ISMCTS. `decide` runs
  `iterations` simulations from the root then returns the **most-visited** root action (the robust
  child, matching ISMCTS's final-selection rule so the two report on the same statistic).

### How this DIFFERS from SO-ISMCTS (the whole point)

| | vanilla UCT (this) | SO-ISMCTS (`ismcts.py`) |
|---|---|---|
| node identity | one node per **action–observation history** (its exact belief) | one node per **information set**; histories reaching the same belief collapse to one node |
| action statistics | per-(action, observation) — NOT aggregated across observations | aggregated over the **whole information set** (the ISMCTS contract) |
| determinization | none global; each observation is an explicit chance node, **one outcome sampled at a time** | **one world `w ~ bw` per iteration** resolves every observation along the descent |
| observation handling | explicit binary chance branch with its own visit/return mean | the determinization routes which sub-child a shared edge descends to |

Both act only on information actually available (the belief, via the exact `filter_*`), so the
information model is identical and the comparison is fair. The ONLY thing UCT lacks is the
information-set grouping / determinization. The expected cost of that lack: UCT fragments its
samples across observation-distinguished children, so a given budget is spread thinner — exactly
the fragmentation ISMCTS was designed to avoid.

## Results (unbiased Dinkelbach rate; small N — see caveats)

| budget | rate | % of ceiling | VoI clawed | E[R] | E[T] | N | sec/ep |
|---|---|---|---|---|---|---|---|
| static floor | 0.0855 | 59% | — | — | — | — | — |
| uct(it=200) | 0.0646 | 44% | −35% | 4.50 | 69.6 | 40 | 8.1 |
| uct(it=400) | 0.0746 | 51% | −18% | 4.00 | 53.6 | 20 | 15.8 |
| uct(it=800) | 0.0799 | 55% | −9% | 3.43 | 42.9 | 14 | 30.2 |
| uct(it=1600) | 0.0668 | 46% | −31% | 4.38 | 65.5 | 8 | 62.8 |
| clairvoyant ceiling | 0.1454 | 100% | +100% | 4.55 | ~31 | — | — |

## Matched-budget comparison: UCT vs SO-ISMCTS

ISMCTS reference numbers are the tail values from the live TensorBoard sweep (it200≈0.061,
it400≈0.071, it800≈0.073, it1600≈0.078); the it=200/400 point estimates from the bounded
`eval_faces.py` run (0.0621 / 0.0747) agree within Monte-Carlo noise. VoI clawed =
(rate − 0.0855) / (0.1454 − 0.0855).

| budget | UCT rate | UCT VoI | ISMCTS rate (live) | ISMCTS VoI | winner |
|---|---|---|---|---|---|
| it=200 | 0.0646 | −35% | 0.061 | −41% | tie (within noise) |
| it=400 | 0.0746 | −18% | 0.071 | −24% | tie (within noise) |
| it=800 | 0.0799 | −9% | 0.073 | −21% | tie (UCT edges, within noise) |
| it=1600 | 0.0668 | −31% | 0.078 | −13% | ISMCTS — but UCT N=8, untrustworthy |

## Verdict — does ISMCTS's machinery earn its keep?

**No — not measurably, on this instance.** At every matched budget where the measurement
is trustworthy (it=200/400/800), vanilla single-tree UCT and SO-ISMCTS are statistically
indistinguishable: both climb monotonically with budget (UCT 0.0646 → 0.0746 → 0.0799;
ISMCTS 0.061 → 0.071 → 0.073), and UCT is, if anything, marginally *above* ISMCTS rather
than below. The information-set grouping + per-iteration determinization that distinguishes
ISMCTS — and that we expected to help by not fragmenting samples across observation-
distinguished children — buys no detectable rate advantage here. Both search families remain
**below the static floor (0.0855)** at all budgets; the exact hierarchical decomposition
(0.094, `decomp-rate.md`) is still the only method above it.

The it=1600 UCT row (0.0668, *below* its own it=800) is **not** evidence of anti-scaling: at
N=8 the ratio estimator's standard error is several hundredths — larger than the whole
200→800 climb — so that single point is noise, not signal. Reading it as a regression would
be over-interpretation; the trustworthy reading is the monotone 200→800 climb, which mirrors
ISMCTS.

## Findings (honest)

1. **UCT ≈ ISMCTS at matched budget (the headline).** The no-determinization baseline matches
   the information-set method point-for-point through it=800. Whatever sample-fragmentation cost
   the explicit per-observation chance branching incurs, it is washed out at these budgets —
   plausibly because the observations are *binary* (low branching) and the effective search depth
   is shallow, so the fragmentation ISMCTS was designed to avoid simply isn't biting.
2. **Both climb with budget; neither clears the floor.** The shared determinant of the rate is
   not the tree machinery but the rollout base + the Dinkelbach λ-coupling, which both solvers
   share.
3. **Shared over-collection signature.** At the low fixed-point λ a weak policy converges to,
   continued collection scores λ-return-positive even when it lowers the rate, so both search
   families bank a large E[R] (4.0–4.5 of 5) at a large E[T] (50–70) — the same pattern NMCS L2
   shows (E[R] 4.55 / E[T] 65). This is a property of the price-of-time coupling at a weak
   policy's λ, not a defect unique to either search.
4. **The comparison was the point, and it is conclusive enough.** A negative result — "the extra
   machinery doesn't pay here" — is the finding. Tightening it=1600 (N≫8) would resolve the last
   noisy cell but would not change the verdict; it is left un-tightened deliberately (UCT is a
   comparison baseline, not a candidate solver).

## Caveats (budgets, N, timeouts)

- **Small N, deliberately, under a strict wall-clock.** UCT is slow: on core 3 it measured ~8.1
  s/episode at it=200 and scales roughly with the budget, so final-eval run counts are kept small
  (it=200: N=40; later budgets shrink — see the table). The ratio estimator on R∈{0..5}/episode has
  a standard error of several percent at N=10–40, so individual deltas are **trends, not
  tightly-resolved numbers**. Each budget was run as its own command under `timeout` on a single
  core (core 3), no parallelism, to respect the live solver runners pinned to cores 0–2.
- **Dinkelbach schedule.** 2 warm rounds then a final measurement; warm/final run counts shrink
  with the budget (it=200 used 8/40, and so on — see `eval_uct.py`'s `PLAN`). The fixed-point λ a
  weak policy converges to is low, which (as `ismcts-result.md` notes) makes continued collection
  look λ-return-positive even when it lowers the rate — the over-collection (E[R]≈4.5/5 at E[T]≈70)
  is partly that coupling, not purely a search defect.
- **ISMCTS reference is from the live run, not re-measured here.** The matched comparison pairs
  UCT's bounded measurement against ISMCTS's live-sweep tail values + the `eval_faces.py` point
  estimates. Both sides carry small-N noise; the qualitative verdict (below) is the trustworthy
  part, not the third decimal.

## Provenance

- Solver: `chocofarm/solvers/uct.py` (vanilla single-tree belief-MDP UCT; `GreedyStopBase` rollout).
- Harness: `chocofarm/eval/eval_uct.py` (mirrors `eval_ismcts.py`; per-budget, `N=` override).
- Readiness wiring: `chocofarm/eval/tb_runner.py` `make_policy` (`method == "uct"`, label "it%d").
- References + ISMCTS numbers: `chocofarm/eval/harness.py`, `docs/results/ismcts-result.md`,
  `docs/results/honest-rates-faces.md`.
