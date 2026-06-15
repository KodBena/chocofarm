# Simulation-phase parallelization / linearization — an algorithmic viability map (2026-06-15)

A research-and-design consult, analysis only. The question is **purely
algorithmic**: of the per-decision simulation phase of the Gumbel-AZ ExIt loop
(`generate_episode` → `decide_with_value` → Sequential-Halving + PUCT descent →
`_evaluate` → {`features.build`, `predict_both`}), **what reformulations admit
dense, branch-free, batched tensor execution that is bit-exact / numerically
equivalent to the current per-episode scalar computation**, and what would each
buy *structurally*. Compute and hardware are treated as unbounded and freely
marshallable — no conclusion below is scoped to a particular box, core count,
accelerator, or scheduler. "Run it on a GPU via Ray" and "reimplement in another
language" are assessed as **algorithmic avenues** (what the algorithm admits and
what it would buy), not against any specific machine.

The tone is the project's: name where exactness holds, name where it breaks,
prefer "this is only an approximate parallelization, and here is what the
approximation costs" over an optimistic listing. The headline up front, because
it governs everything below:

> **The simulation phase is dominated by a fundamentally sequential object — a
> tree search whose every step reads statistics written by the previous
> simulation's backup. The two embarrassingly-parallel axes (across episodes,
> across world-set rows of a belief reduction) are *exactly* parallelizable and
> already substantially exploited; they are the only places a bit-exact dense
> linearization is on the table. The one axis that would batch the *inner*
> simulation work — across tree leaves — is only reachable through an
> *approximate* algorithmic change (virtual-loss lockstep / batched-Gumbel
> `mctx`-style search) that perturbs the Sequential-Halving budget accounting and
> the Danihelka "executed action = SH survivor" invariant the test suite pins as
> the fidelity immune system. So the honest answer is: exact linearization is
> available for the *reductions and the net forward and the cross-episode
> fan-out*; an exact linearization of *the search itself* is not, and the
> language-reimplementation gain is real but is a constant-factor win on the
> sequential Python floor, not a new parallel axis.**

---

## 0. The cost structure we are reasoning about (load-bearing facts)

These are measured facts from the existing record
(`docs/results/az-perf.md`, `az-jax-perf.md`, `az-exit-loop.md`,
`az-parallel-exp.md`). Hardware specifics in those docs are deliberately *not*
imported here; only the **algorithmic cost shape** — which computations dominate
and *why structurally* — is used.

**C1 — After two optimization passes, the per-decision hot path decomposes into
three structurally distinct buckets** (`az-jax-perf.md` §per-hotspot):

| bucket | what it is | structural character |
|---|---|---|
| **belief reductions** (marginals + the `nb×nD` detector counts) | a reduction over the world-set `bw` (up to 15,504 int64 bitmasks) | **dense, data-parallel over rows** — already numba-fused to ~16 µs/leaf, ~9× over numpy |
| **net forward** (`predict_both`: trunk matmuls + masked softmax) | linear algebra | **already dense linear algebra**; ~28% of decide time; the de-risked exactness seam |
| **tree control flow** (SH/PUCT recursion, per-edge dict `W`/`N` bookkeeping, `env.apply`/filter, slot lookups) | data-dependent recursion with serial backups | **fundamentally sequential** — the MLP-free floor |

**C2 — The sequential floor is the binding constraint.** An upper-bound
experiment replacing the MLP forward with a zero-cost stub put the end-to-end
floor at the "MLP-free" level: even an infinitely fast forward and infinitely
fast reductions cannot remove the tree control flow, which is the dominant
remaining cost (`az-jax-perf.md` §result: "the remaining … is the Python tree
control flow … which neither float32 nor numba nor JAX touches"). **This is
Amdahl's law speaking: the parallelizable fraction of the *inner* simulation is
already small relative to the irreducibly sequential recursion.**

**C3 — Two exactness regimes already coexist in the codebase, and the project
has chosen them deliberately:**
- *bit-exact* structural memoizations (the `env._dist` table caches literal
  `math.hypot` outputs; the belief-feature cache verifies every hit with
  `np.array_equal`; the numba kernels are integer-exact with the numpy
  reduction). `az-perf.md` proves these reproduce **every float** over 381
  decisions (`max|ΔG| = 0.0`).
- *behavioral* (not bit) equivalence for the float32 + JAX path: the bar is
  "fixed-λ₀ rate, E[T], action distribution statistically indistinguishable over
  N≥300 episodes" (`bench_equivalence.py`). The numpy↔jit forward equivalence is
  pinned to **float32 roundoff** (`test_jax_equivalence.py`, `ABS_TOL=1e-4`), not
  bit-equality — XLA fuses/reorders, so it differs in the last bits and "may flip
  near-tied argmax/SH choices" (`mlp_jax.py` docstring).

The distinction in C3 is the whole game for question (1): **a dense batched
reformulation that changes reduction/accumulation order is at best
behaviorally-equivalent, never bit-exact**, and whether that matters depends on
whether a last-bit flip can change a discrete choice (an argmax, an SH survivor,
an RNG-consuming branch).

---

## 1. Exact linearization, component by component

The question asks whether each component can be reframed as **dense, branch-free
(masked), fixed-shape batched tensor ops that are bit-exact / numerically
equivalent to the current per-episode scalar computation**. Worked one component
at a time, with the exactness verdict made explicit each time.

### 1.1 Belief representation and its updates — **exactly vectorizable; already dense; the columnar form is the world-set matrix**

The belief is `bw`: a 1-D `int64` array of surviving-world bitmasks (each a
5-of-20 set, `C(20,5)=15,504` at the root). Two update primitives, both already
pure vectorized bitwise filters (`env.py`):

```
filter_treasure(bw, i, present): bit = (bw >> i) & 1; return bw[bit == present]
filter_detector(bw, i, pos):     hit = (bw & cover_mask[i]) != 0; return bw[hit if pos else ~hit]
```

These are **masked gathers over a columnar int64 vector** — the canonical
SIMD/GPU-friendly shape already. The belief-derived feature reductions
(marginals, per-detector positive counts) are a single pass over `bw`:

```
marg[t]  = mean over worlds of (w >> t) & 1
cnt[d]   = count over worlds of (w & cover[d]) != 0
```

This is exactly `kernels.belief_marg_cover`, an `@njit` fused loop, **integer-exact
with the numpy reduction** (the docstring and `az-jax-perf.md` certify it; the
float dtype never touches the integer counts — only the downstream `cnt/nb`
division does).

**Exactness verdict — bit-exact is *available and partly already realized*, with
one caveat.** The marginals/counts are integer sums; integer addition is
associative, so **any reduction order gives the identical integer result** — a
batched/segmented GPU reduction over `bw` rows is bit-exact with the scalar loop.
The only float operation is the final `marg = cnt * (1/nb)` and `p_pos = cnt/nb`;
a single reciprocal-multiply vs a divide can differ in the last ULP, but the
kernel already fixes a specific order (`inv = 1.0/nb; marg[t] *= inv`), so a dense
form that mirrors *that* order is bit-exact, and one that uses true division is
~1-ULP behavioral. Either is well inside the behavioral bar.

**The natural dense/columnar form for batched execution** is a `(n_beliefs ×
max_worlds)` padded int64 matrix with a validity mask (beliefs have ragged
world-counts, 1…15,504), reduced along the world axis with the pad lanes masked
out. Because the reduction is integer-additive, **masked pad lanes contribute
exactly zero and the result is bit-exact** regardless of padding — this is the
clean case. The cost is the padding waste: F5's belief-size distribution is
heavy-tailed (median ≈ 118, p90 ≈ 7,260, max 15,504), so a naive rectangular pad
to the max is enormously wasteful late-episode; a bucketed/ragged
(CSR-style, segment-id) layout recovers the density. None of this is a fidelity
risk — it is purely a layout decision.

**Honest structural note:** this component is *already* the cheap part. The two
belief reductions collapsed from the former dominant ~216 µs of the leaf to ~23 µs
combined (`az-jax-perf.md`). Vectorizing them further across leaves only helps if
the *leaves themselves* are batched — which runs into §1.2. In isolation the
belief update is the most exactly-linearizable component and also the one with the
least remaining payoff, precisely because it has already been linearized within a
leaf.

### 1.2 The search — **NOT exactly linearizable; batchable only via an approximate algorithmic change**

This is the crux and the honest "no." The search is Gumbel-root selection +
Sequential Halving over a PUCT-descended information-set tree (`gumbel_search.py`).
Its control flow is **data-dependent in a way that resists masking-to-exactness**,
for three composing reasons:

1. **Serial statistic dependence (the backup chain).** `_descend` selects the
   PUCT-argmax child using `node.N`/`node.W`, descends, and on return *writes back*
   `W[a] += ret; N[a] += 1`. The very next simulation of the same root action reads
   those updated stats. So simulation *k+1*'s descent path is a function of
   simulation *k*'s backup. This is a strict sequential dependence: you cannot
   evaluate simulation *k+1*'s leaf until simulation *k* has been backed up,
   **without changing what simulation *k+1* would have chosen.** Masking does not
   help — there is no fixed-shape batch of independent work here; the work is a
   dependency chain.

2. **Sequential-Halving's adaptive budget.** SH (`_sequential_halving`) runs
   `⌈log2 m⌉` phases; each phase allocates an equal share to the *current
   survivors*, then drops the worst half by `g + logit + σ(q̂)`. The survivor set
   of phase *p+1* depends on the realized `q̂` from phase *p*'s simulations. The
   phase structure, the per-phase budget arithmetic, and the survivor cut are
   data-dependent on the running statistics — again a chain, not a batch. The test
   suite pins three Danihelka invariants here
   (`test_executed_action_is_sh_survivor`, `test_vmix_prior_weighted`,
   `test_sequential_halving_spends_full_budget`); `az-exit-loop.md` §(f) records
   that the first implementation *broke* all three and an out-of-frame audit caught
   it. These invariants are the **fidelity immune system**, and any reformulation
   that perturbs the budget accounting must re-validate them.

3. **RNG-stream and tie-break determinism.** Within `_visit`, each simulation
   draws `w = env.sample_world(bw, rng)` (and the leaf outcome-averaging draws
   `c_outcome` more via `env.sample_world`), consuming the RNG stream in a
   **specific, position-dependent order**. The Gumbel root draw is one
   `rng.gumbel(size=n_slots)`. Parallel-≈-serial determinism in the existing actor
   pool is achieved *not* by parallelizing the RNG draws but by **folding a
   per-(iteration, kind, episode) seed** so the same logical episode draws the same
   stream regardless of worker count (`parallel.py:_task_rng`; verified
   bit-identical workers=1 vs workers=4, `az-parallel-exp.md` §"Parallel ≈
   serial"). That is the exactness lever for the cross-episode axis — it works
   because each episode is independent, not because the within-episode draws were
   reordered. The moment you batch *within* a search and draw the determinizations
   in a different order or all-at-once, you consume a different stream and the
   trajectory diverges — not wrong, but **not bit-exact**.

**What a batched search would require, and what it costs.** The only batch-≥8 seam
in SH+PUCT is the production-AlphaZero technique the `mctx` reference names: a
**batched-Gumbel / virtual-loss lockstep** search. You descend K simulations
together against the *current* stats plus a virtual-loss penalty (so the K
descents spread instead of collapsing onto one path), collect the K leaves into one
fixed-shape batch, evaluate the net once on the batch, and back all K up together.
This is genuinely batched (the K leaf forwards amortize the per-dispatch cost — the
regime where batched JAX wins at ~12 µs/item, `az-jax-perf.md`). But it is **an
algorithmic change, not a numeric one**: the K leaves are chosen against *stale*
stats (the virtual loss is a deliberate fiction), which:
- perturbs SH's exact per-phase budget accounting (K-at-a-time vs one-at-a-time
  changes which survivors get how many sims),
- can flip the "executed action = SH survivor" invariant,
- consumes the determinization RNG in a different order.

`az-jax-perf.md` explicitly weighs this and **defers it**: the Amdahl ceiling
(MLP is ~28% of decide time; perfectly batching it buys ≤1.55× more, realistically
~1.3×) does not justify re-deriving SH budget accounting under virtual loss and
re-validating Danihelka fidelity. That judgment stands under the
unbounded-hardware framing too, *because the ceiling is structural, not a hardware
limit* — the sequential recursion is ~72% of decide time on any machine, so even a
free, infinitely-wide accelerator for the batched 28% lands the inner search at
best ~1.3–1.55× faster, and at the cost of exactness.

There is a "fixed-shape padding" route worth naming honestly: you *can* pad the
tree to a fixed branching factor and depth and run a fully-`scan`'d masked descent
(this is what `mctx` does on TPU). It is mechanically possible. But `az-jax-perf.md`
records the verdict: "a fully `scan`'d tree is not worth it" — the per-step PUCT
selection still depends on the previous step's backup, so the `scan` is a
*sequential* `scan` (no parallel speedup along the recursion axis), and the masking
overhead to keep shapes fixed across the ragged real tree adds cost. It buys
hardware-portability of the control flow, not parallelism of it.

**Exactness verdict — exact linearization of the search is *not available*.** The
search batches only through an approximation (virtual-loss lockstep) whose cost is
(a) the Danihelka-fidelity re-derivation and (b) loss of bit-exactness *and*
behavioral drift in the discrete SH choices, against (c) an Amdahl-capped ≤1.55×
on the 28% slice. This is the gold-plating the project is right to defer.

### 1.3 Feature maps over the belief — **exactly vectorizable; the assembly is a known, deferred safe lever**

`features.build` assembles the fixed-dimension vector (241 floats on the live env:
per-treasure `[marg|collected|available|dist|unc]`×20, per-detector
`[informative|p_pos|dist]`×44, global×9). Mechanically it is: one fused belief
reduction (§1.1, integer-exact) + per-loc distance blocks served from a static
memo (bit-exact, `math.hypot` cached) + cheap elementwise derivations
(`unc = marg*(1-marg)`, `avail = (marg>0)&(coll==0)`, `Σunc`) + numpy-slice
assembly into the output array.

**Exactness verdict — bit-exact across leaves is available.** Every operation is
either an integer-exact reduction, a memoized-bit-exact distance lookup, or an
elementwise float op with a fixed order. Batching the build across N leaves is a
matter of stacking: the per-belief reductions batch as in §1.1 (integer-exact),
the per-loc blocks are gathers from the static memo (exact), and the elementwise
ops broadcast over the batch (exact, same order). The masked-form is trivial — the
legal mask itself is a *slice* of the feature vector (`legal_mask_from_features`
reads `available`/`informative` sub-blocks), so it composes with the dense layout
for free.

`az-jax-perf.md` names a concrete, **behavior-preserving, no-fidelity-risk** lever
already scoped here: a numba `assemble` kernel for the output-vector assembly
prototypes at 1.3 µs vs ~10 µs of numpy-slice Python per call. That is the same
kind of win as the reduction kernels and carries the same bit-exact guarantee. It
is a Python-floor constant-factor win (see §3), not a new parallel axis — but it is
*exact* and *cheap*, and it is the only feature-side lever not already taken.

### 1.4 The net forward — **already linear algebra; the de-risked exactness seam; bit-exactness is the one thing it does NOT have**

The forward is two 256×256 trunk matmuls + ReLUs (+ optional residual block) +
linear value head + policy logits + masked softmax (`mlp.py._forward` /
`_predict_both_f32`; `mlp_jax.py._forward_both`). This is *already* dense
branch-free linear algebra — there is nothing to linearize; it is the component
that is natively accelerator-shaped.

**The project maintains exactly the equivalence the question flags as the
de-risked seam.** `test_jax_equivalence.py` pins `numpy_forward(W,X) ≈
jax.jit(jax_forward)(W,X)` for both value and logits, residual ON/OFF, single-row
and batched. Critically it pins it at **three** levels: numpy-f64 `_forward` ↔
jit-f32 forward, *and* the production numpy-f32 `_predict_both_f32` ↔ jit-f32
forward (the `test_production_f32_forward_matches_jax_jit` case, added after an
out-of-frame audit noticed the safeguard certified f64≈f32 while the live path is
f32). So an accelerator forward is **wiring against a guarded contract**, not a
rewrite.

**Exactness verdict — *behaviorally* equivalent, explicitly *not* bit-exact, and
this is the one place where the non-exactness can bite a discrete choice.** The
equivalence bar is float32 roundoff (`ABS_TOL=1e-4`); XLA fuses/reorders, so the
forward's last bits differ from numpy's. `mlp_jax.py` is explicit: this "may flip
near-tied argmax/SH choices." In practice `bench_equivalence.py` found float32 and
float64 produced *identical* rates across 1200 episodes (the argmax/SH margins were
well-separated enough that no flip occurred) — so the behavioral bar holds
empirically and comfortably. But the honest statement is: **the net forward is the
exactly-the-component whose batched/accelerated form is numerically-equivalent but
not bit-exact, and the non-exactness propagates into the search only through
near-tied discrete choices, which are empirically rare here.** This is the
acceptable, measured, already-de-risked non-exactness.

The leverage point is that this seam only pays *when the forward is batched* — the
load-bearing negative result (`az-jax-perf.md`) is that single-leaf JAX dispatch is
~10× *slower* than float32-numpy because the per-call dispatch tax (~290 µs floor)
dwarfs the sub-microsecond compute. So the forward's accelerator value is
**entirely contingent on a batched leaf seam existing** — which is §1.2's
approximate-only restructure, or §2's leaf-batched / cross-episode axes.

### 1.5 Where exactness is at risk — the precise inventory

Pulling the four components together, exactness is at risk in exactly these
places, and each has a known disposition:

| risk site | mechanism | exact-preserving? | disposition |
|---|---|---|---|
| FP reduction/accumulation order in the **net forward** | XLA fuse/reorder; f32 vs f64 | **No** (behavioral only) | accepted, guarded by `test_jax_equivalence` + `bench_equivalence`; flips only near-tied choices, empirically none observed |
| FP order in **marginals/p_pos division** | `cnt*(1/nb)` vs `cnt/nb` | **Yes** if order mirrored; ~1 ULP otherwise | trivial; integer counts are exact regardless |
| **argmax tie-breaking** in PUCT (`v > best_v`, strict `>`) and SH survivor cut | a last-bit perturbation can flip a tie | **Yes** under bit-exact reductions; **at risk** under any reordered/f32 forward feeding Q | this is the *channel* through which forward non-exactness reaches the trajectory; rare because margins are wide |
| **RNG stream order** (`env.sample_world` per sim, `rng.gumbel` at root) | batching within a search consumes the stream in a different order | **Yes** across-episode (seed-fold); **No** within a batched search | the cross-episode axis is bit-exact by seed-fold; a within-search batch is not |

The clean reading: **everything except the net forward and a within-search batch
is recoverable to bit-exactness.** The net forward's non-exactness is the
deliberate, guarded, empirically-benign trade. A within-search batch's
non-exactness is the *approximate* search restructure of §1.2.

---

## 2. Batching axes and how they compose

Three independent axes along which the computation can be batched. The question is
which compose and what restructuring each demands.

### Axis A — across episodes (the embarrassingly-parallel axis) — **exact, already exploited, composes with everything**

Each generation/eval episode is an independent rollout under a frozen net. This is
the axis `parallel.py` already exploits: a process pool, each worker a distinct
episode stream, transitions returned over redis as raw bytes. **It is bit-exact
across worker counts** by the seed-fold (`_task_rng`): the same logical episode
draws the same RNG stream regardless of scheduling — verified workers=1 vs
workers=4 produce bit-identical aggregate transition multisets (`az-parallel-exp.md`).

Under unbounded hardware this axis is the **dominant structural payoff** and the
cleanest: episodes are independent, so throughput scales with the episode-fan-out
width with *zero* fidelity cost. The current realization is process-level
(GIL-bound Python search ⇒ processes not threads), and its measured ~1.9× is a
*host-contention* ceiling (4 vCPUs on a contended libvirt host delivering ~2.6×
pure-CPU), **not an algorithmic limit** — `az-parallel-exp.md` is explicit that an
uncontended host approaches the core count. Under the consult's
unbounded-hardware framing, **this axis scales to as many episodes as you can
afford to run concurrently, exactly.** It is the answer to "more/faster rollouts
per unit work" that carries no exactness tax.

What it demands of the structure: nothing new — it is built. The only algorithmic
note is that episode parallelism does not reduce *per-decision latency*; it
increases *throughput*. For a research program that wants "more calibration
experiments per unit work" (the dual-bound and consult-003 calibration agenda;
§4), throughput is exactly the right currency, so this axis is the one most aligned
with the program's actual need.

### Axis B — across world-set rows of a belief reduction (the intra-leaf data-parallel axis) — **exact (integer-additive), already realized within a leaf, composes with A**

The marginals/detector reduction over `bw` is a row-parallel reduction (§1.1).
It is bit-exact under any row partition (integer addition associative). It is
already realized as a fused numba pass within a leaf; under unbounded hardware it
could be a segmented GPU reduction over a batched ragged belief matrix.

It composes with Axis A trivially (different episodes reduce different beliefs
independently). **But on its own it has little remaining payoff** — it is already
~23 µs/leaf and off the critical path (C2: the floor is the tree control flow, not
the reductions). Its value re-emerges only *combined with Axis C* (batch many
leaves' reductions at once), which is where the payoff would be — but Axis C is the
approximate one.

### Axis C — across tree leaves (leaf-batched evaluation) — **approximate only; the one axis that would batch the inner forward + reductions; demands the virtual-loss restructure**

This is the axis that would let the net forward (§1.4) and the reductions (§1.1,
§1.3) batch enough to win on an accelerator (batch ≥ 8 is the JAX crossover). It is
*the* reason to want any of this. But as §1.2 establishes, **the SH+PUCT search
does not expose ≥8 independent leaves without virtual-loss lockstep**, which is an
approximate algorithmic change. So:

- Axis C **does not compose exactly** with the current search; it *replaces* the
  current search's selection rule with a batched-selection approximation.
- It composes with Axis A (you can run virtual-loss-batched searches across many
  episodes) and subsumes Axis B (the batched leaves' reductions batch together).
- Its payoff is Amdahl-capped: even free, infinitely-wide leaf batching only
  speeds the ~28% forward + the small reduction slice, leaving the ~72% sequential
  recursion — so the inner-search speedup ceiling is ~1.3–1.55×, *and* it costs
  exactness + the Danihelka re-validation.

**Composition summary.** A (episodes) and B (world rows) are exact and compose
freely; together they are the bit-exact parallel envelope, and A is the
high-payoff one. C (leaves) is the only axis that batches the inner forward, is
approximate-only, Amdahl-capped, and composes with A but replaces the exact search.
**The exact-and-high-payoff frontier is Axis A; the exact-but-low-remaining-payoff
is Axis B; the high-conceptual-appeal-but-approximate-and-capped is Axis C.**

---

## 3. Language reimplementation, framed structurally

What does moving the simulation's inner loop out of interpreted scalar Python buy
*structurally*, and which restructurings that **stay in Python** capture a
comparable share?

### What the language move buys, structurally

The MLP-free floor (C2) is the Python tree control flow: per-node `dict` `W`/`N`
bookkeeping, per-edge slot lookups, the recursive `_descend`/`_simulate_root_action`
call overhead, `env.apply`/filter dispatch, and the SH/PUCT Python arithmetic. A
compiled reimplementation of *this loop* (a tight C/Rust/numba-typed kernel over a
columnar node arena) would structurally eliminate:
- **per-node Python object/dict churn** — replace the `_Node` `__slots__` object
  with `W`/`N`/`children` dicts by a flat struct-of-arrays node arena (parallel
  arrays indexed by node id), so a "visit" is array writes, not dict inserts;
- **per-node interpreter dispatch** — the recursion becomes a compiled loop with
  no bytecode-dispatch tax per edge;
- **the `dict`-keyed children map** (`(action, belief_key) -> _Node`) — replace
  with integer-indexed child tables in the arena, eliminating hashing of
  `_belief_key` tuples on every descent;
- **tuple-action plumbing** — actions are already integer slots in the hot path
  (`_s2a`/`_a2s` hoisted); a compiled core works in slot-ints throughout.

This is a **constant-factor win on the sequential floor**, not a new parallel
axis. It is the single largest *exact* lever available, because the floor is the
binding constraint (C2) and the floor is exactly what a compiled reimplementation
attacks. Crucially it is *bit-exact-able*: the arithmetic (PUCT formula, SH cuts,
backups) is the same operations in the same order; only the dispatch and memory
layout change. `az-perf.md` already proved the structural memoizations of this kind
(distance table, slot tables) are bit-identical over 381 decisions — a compiled
node arena is the same discipline extended to the recursion itself.

### Which in-Python restructurings capture a comparable share

The honest framing the project prefers: most of the language-move's *structural*
gain is achievable without leaving Python, because the gain is about **layout and
dispatch, not about the language per se**:

1. **A columnar node arena in numba** (struct-of-arrays `W`/`N`/`prior`/`legal`
   indexed by node id, child edges as integer tables) — captures the per-node
   churn and dispatch elimination *inside* a `@njit` region. This is the same
   `@njit` lever already used for the reductions, extended to the tree
   bookkeeping. It is the highest-value in-Python restructure and is bit-exact-able.
   The obstacle is that the leaf eval (`predict_both`) and `env.apply` filters
   currently live in numpy/Python; a numba tree loop would need them callable from
   nopython mode (the filters are simple bitwise ops — numba-friendly; the MLP
   forward would be an object-mode boundary or a numba matmul).
2. **The `assemble` kernel for `features.build`** (§1.3) — already scoped, 1.3 µs
   vs ~10 µs, bit-exact, no fidelity risk. Cheap and immediate.
3. **Eliminate per-node object/array churn** — pool `_Node` allocations, avoid the
   `np.full(n_slots, -1e30)` + per-slot Python loop in `_improved_policy` and the
   root-logit construction by vectorizing over the legal-slot array. These are
   bit-exact numpy-vectorization wins inside the existing Python.
4. **Expand the numba kernels** to cover `legal_mask_from_features` slicing and the
   `filter_treasure`/`filter_detector` gathers as a fused step. Bit-exact.

The structural verdict: **a Python-resident columnar/numba restructure of the tree
loop captures the bulk of what a full-language port would, because the bottleneck
is layout + dispatch on a sequential recursion, and numba's `@njit` already gives
compiled scalar loops over columnar arrays.** A full port (Rust/C++) would capture
the *last* slice (the object-mode boundary at the MLP forward, the absolute
dispatch floor), but at a large engineering and licensing-surface cost and with no
new parallel axis to show for it. Under unbounded hardware the port still does not
beat Axis-A episode fan-out for *throughput* — it lowers per-decision latency,
which matters less than throughput for the calibration-experiment program (§4).

---

## 4. Overarching synthesis — ranked avenues, exactness × payoff

The currency, per the brief, is **simulation throughput of the experiment program
in the abstract** — more/faster rollouts per unit work, enabling more calibration
experiments (the program's actual frontier: the consult-003 calibration agenda and
the dual-bound's need for a calibrated `V̂_AZ`, where each calibration variant is a
fresh training run gated on rollout throughput). Ranked by (structural payoff) ×
(fidelity), highest first.

| rank | avenue | axis | payoff (throughput) | fidelity | demands |
|---|---|---|---|---|---|
| **1** | **Cross-episode fan-out, widened** (Axis A) | A | **High** — scales ~linearly with concurrent episodes under unbounded hardware; the measured 1.9× is a host ceiling, not an algorithmic one | **Bit-exact** (seed-fold, verified) | already built; under unbounded hardware just widen the pool / lift the host contention. No algorithm change. |
| **2** | **Compiled/columnar tree-loop core** (numba node arena), incl. the `assemble` kernel | (latency, not an axis) | **Medium** — a constant-factor cut on the binding sequential floor (C2); the largest *exact* per-decision lever | **Bit-exact-able** | a struct-of-arrays node arena callable from `@njit`; the MLP-forward boundary handled in object mode or a numba matmul. Real engineering, no fidelity risk. |
| **3** | **Leaf-batched / virtual-loss `mctx`-style search** (Axis C) | C | **Low-Medium and Amdahl-capped** — ≤1.3–1.55× on the inner search even with a free infinite accelerator, because ~72% is sequential recursion | **Approximate** — perturbs SH budget accounting + the executed-action=SH-survivor invariant; not bit-exact; behavioral drift in discrete choices | re-derive SH accounting under virtual loss, re-validate the three Danihelka invariants, accept behavioral (not bit) equivalence. The gold-plating candidate. |
| — | Intra-leaf belief-reduction vectorization (Axis B) standalone | B | **Negligible remaining** — already ~23 µs/leaf, off the critical path | Bit-exact | none; already realized within a leaf. Only re-emerges fused into Axis C. |
| — | Full non-Python port (Rust/C++) | (latency) | **Medium-Low marginal over #2** — captures the last dispatch slice beyond numba | Bit-exact-able | a full reimplementation + licensing-surface cost; no new parallel axis; loses to Axis A for throughput. |

### The honest which-1–3-to-pursue verdict

- **Pursue #1 (cross-episode fan-out) — it is the answer.** Under unbounded,
  freely-marshallable hardware, the simulation throughput of the experiment program
  is gated by how many independent episodes you can run concurrently, and that axis
  is **already bit-exact and already built** — the only thing between the current
  ~1.9× and a near-linear scale-out is host contention, which the unbounded-hardware
  framing dissolves. This is the avenue that most directly buys "more calibration
  experiments per unit work" with zero fidelity cost. It is the structurally correct
  place to spend, and the spend is operational (more concurrent workers / a less
  contended host / a distributed episode-fan-out à la Ray), not algorithmic.

- **Pursue #2 (compiled/columnar tree core) if and only if per-decision latency —
  not throughput — becomes the bottleneck**, e.g. for a latency-sensitive eval or
  an interactive use. It is the largest *exact* lever and attacks the genuine
  binding constraint (the sequential floor), and a Python-resident numba node arena
  captures most of a full-language port's gain. But for the *throughput* currency
  the program actually optimizes, #1 dominates it: widening episode fan-out scales
  unboundedly and exactly, whereas the tree-core is a bounded constant factor on one
  decision. So #2 is a *conditional* recommendation — worth it for latency, not the
  first thing to reach for under the throughput framing. The `assemble` kernel and
  the small vectorization cleanups (§3.2–3.3) are cheap, exact, and worth doing
  opportunistically regardless.

- **Do NOT pursue #3 (virtual-loss leaf batching) — it is the gold-plating.** It is
  the avenue with the most superficial appeal ("batch the leaves, hit the GPU"), but
  it is the only one that is *both* approximate *and* Amdahl-capped: the sequential
  recursion is ~72% of decide time on *any* hardware, so even a free infinite
  accelerator for the batched slice caps the inner-search win at ~1.3–1.55×, and it
  buys that capped win by trading away bit-exactness and re-opening the Danihelka
  fidelity surface the test suite was built to protect. The project already weighed
  and deferred this (`az-jax-perf.md`); the unbounded-hardware framing does **not**
  rehabilitate it, because its ceiling is structural (the dependency chain), not a
  hardware constraint. The only world where #3 becomes interesting is one where the
  *net itself* grows large enough that the forward dominates the recursion (a
  transformer/DeepSets-over-worlds encoder — explicitly rejected in the design,
  §2/§8 of `alphazero-surrogate-design.md`, because F6 shows the cheap MLP is
  near-sufficient). At the current tiny-MLP scale, the forward is 28% and batching
  it is not worth the fidelity surface.

### One-line synthesis

> The simulation phase admits **exact** parallel linearization on the
> cross-episode axis (high payoff, already built, throughput-aligned) and the
> intra-leaf reduction axis (exact, already realized, low remaining payoff), and an
> **exact** constant-factor compiled-tree-core latency win on the binding
> sequential floor; it admits **only an approximate, Amdahl-capped** linearization
> of the search itself (leaf-batched virtual-loss), whose ceiling is structural and
> whose cost is the Danihelka-fidelity surface — so the worth-pursuing avenues are
> **#1 widen the exact episode fan-out** (the throughput answer) and, conditionally
> for latency, **#2 a compiled columnar tree core**, while **#3 GPU leaf-batching is
> gold-plating** at this net scale.

---

## 5. Honest caveats on this analysis

- **This is a structural assessment from the record, not a fresh measurement.** No
  code was run, no benchmark re-measured (the brief forbids it). The cost-shape
  facts (C1/C2, the 28%/72% split, the batch-≥8 JAX crossover, the bit-exactness of
  the reductions) are taken from `az-perf.md` / `az-jax-perf.md` /
  `az-parallel-exp.md` / `test_jax_equivalence.py`, which are themselves
  measured-and-audited. Where I quote a speedup ceiling (~1.3–1.55× for Axis C) it
  is the record's own Amdahl arithmetic, not a new number.
- **The Amdahl split is net-scale-dependent.** The ~28%-forward / ~72%-recursion
  decomposition is for the current tiny ~100k-param MLP. A larger net would shift
  the forward's share up and make Axis C / the accelerator forward more attractive —
  but the design deliberately keeps the net tiny (F6: marginals near-sufficient), so
  this is a hypothetical, not a current lever.
- **"Bit-exact-able" is a claim about the operations, not a delivered artifact.**
  The compiled tree core (#2) *can* preserve bit-exactness because the arithmetic is
  unchanged — but only a bit-identical-trajectory check (the `az-perf.md`
  `max|ΔG|=0.0` discipline) would *certify* a given implementation. The analysis
  identifies where exactness is preservable; it does not pre-certify any code.
- **Throughput vs latency is the axis that decides the ranking, and it is a
  program-goal judgment.** I have ranked under the brief's stated currency
  (throughput of the experiment program). If the actual need were single-decision
  latency (it is not, for a calibration-experiment program), #2 would rise above #1.
- **Single instance, uncalibrated time model** — the standing chocofarm caveat: all
  of this is conditioned on the current env (TELE_OH=12, symmetric Euclidean
  travel). It does not affect the parallelization analysis (which is about the
  algorithm's structure, not the instance's numbers), but it is the frame within
  which "more calibration experiments" is worth wanting.

---

## Appendix A — commission prompt (verbatim)

> Recorded verbatim per the consult-record discipline (`docs/consults/consult-001-prompt.md`
> is the format reference).

---

This is a **research + design consult** — analysis only, NOT implementation. You will read code, then write ONE design document and commit it on a branch. Do not implement anything, do not modify source, do not run any code or job.

## Scope framing — READ THIS CAREFULLY (it governs the whole consult)

This is a **purely algorithmic** investigation: *what is on the table from an algorithmic perspective* for parallelizing / linearizing the simulation phase of the AlphaZero loop. **Treat compute and hardware as unbounded and freely marshallable.** Do **NOT** scope any conclusion to a particular machine, core count, accelerator model, memory budget, or cluster/scheduler setup; do **NOT** describe or even mention the current runtime environment or "what is wired in today." "Run it on a GPU via Ray" and "reimplement in another language" are named by the maintainer only as **examples of algorithmic avenues** — assess them for algorithmic viability and structural payoff, not against any specific box. The question is what reformulations the *algorithm* admits, and what they would buy *structurally*.

## The questions (the heart of the consult)

**(1) Exact linearization for parallel / accelerator execution.** Read the simulation algorithm and assess whether it can be reformulated as **dense, branch-free (masked), batched tensor operations that are bit-exact / numerically equivalent to the current per-episode scalar computation** — i.e. an *exact* linearization, not an approximation — and is therefore amenable to SIMD / GPU / large-batch execution. Work component by component:
  - **Belief representation and its updates** — the world-set and the bitwise filters that a sense/collect applies to it. How vectorizable is this; what is the natural dense/columnar form.
  - **The search** — Gumbel-root action selection + Sequential-Halving + tree expansion/descent. This is data-dependent control flow; assess batchability via masking and fixed-shape padding (the batched-Gumbel-search register — cf. DeepMind `mctx`). What is exactly preserved vs. what forces approximation.
  - **Feature maps** over the belief.
  - **The net forward** — already linear algebra. Note that the project maintains an exact equivalence between numpy inference and a compiled forward pass (a jax-equivalence invariant with a guarding benchmark); that equivalence is the de-risked seam for an accelerator forward — leverage it in the analysis.
  Identify precisely **where exactness is at risk** (floating-point reduction/accumulation order, argmax tie-breaking, the RNG stream and how stochastic choices are drawn) and how each can be preserved — or, honestly, where only an *approximate* parallelization is available and what the approximation costs.

**(2) Batching axes.** Enumerate the independent axes along which the computation can be batched — across episodes, across tree leaves (leaf-batched evaluation), across the world-set dimension — and the algorithmic restructuring each axis demands. Which axes compose.

**(3) Language-reimplementation gains, framed structurally.** What does moving the simulation's inner loop out of an interpreted scalar implementation buy **structurally** — eliminating per-node dispatch overhead, enabling tight columnar layouts, SIMD, branch-free kernels — and which **structural reformulations that remain in Python** (vectorized / leaf-batched search, a columnar belief representation, eliminating per-node object/array churn, expanding the numba/Cython kernels, etc.) capture a comparable share of that gain. Frame this as algorithmic restructuring and its expected structural payoff — not as a port for any particular target.

**(4) Overarching synthesis.** Rank the avenues by (expected structural payoff) × (fidelity: exact vs approximate to the current computation). For each, name what it demands of the algorithm's structure and whether it preserves exactness. Give an honest verdict on which 1–3 are worth pursuing and which are gold-plating. Tie the payoff to **simulation throughput of the experiment program in the abstract** (more / faster rollouts per unit of work, enabling more calibration experiments) — not to any deadline, machine, or capacity.

## Survey targets (read each fully before citing — do not act on grep fragments)

Simulation hot path (the algorithmic object of study):
- `chocofarm/az/gumbel_search.py` — the Gumbel + Sequential-Halving search.
- `chocofarm/model/env.py` — env transition and belief-set filtering.
- `chocofarm/az/features.py` — feature maps over the belief.
- `chocofarm/az/kernels.py` — existing numba kernels (what is already vectorized/compiled).
- `chocofarm/az/mlp.py` and `chocofarm/az/mlp_jax.py` — the numpy and compiled forward passes (the exactness seam).
- `chocofarm/az/value_target.py` — value-target computation in the loop.
- `chocofarm/az/parallel.py` — how generation is decomposed into independent work (read for the *algorithmic* decomposition only; ignore any host-specific detail).
- `chocofarm/az/bench/bench_hotpath.py`, `chocofarm/az/bench/bench_equivalence.py` — the existing hot-path characterization and the numpy↔compiled exactness invariant (directly relevant to "exact linearization").

Design + program context:
- `docs/design/alphazero-surrogate-design.md` (the algorithm's design), `docs/design/dual-bound.md` (why simulation throughput matters to the program), `docs/handoff-2026-06-15.md`, `docs/STATUS.md`.
- `docs/results/az-perf.md`, `az-jax-perf.md`, `az-exit-loop.md`, `az-parallel-exp.md` — read these to understand the **algorithmic cost structure** (which computations dominate and *why structurally*). Do NOT import their hardware specifics into your conclusions.

## Constraints (hard)

- Analysis/design ONLY. Do not modify code, do not run the simulation / training / any benchmark, do not touch any running process or any redis state.
- You are in an isolated git worktree. Create branch **`docs/sim-parallelization-viability`** and commit ONLY `docs/design/simulation-parallelization-viability.md` (**EXPLICIT PATH ONLY — never `git add -A`/`.`**). Commit message ends with exactly: `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`
- **Do NOT push.** The orchestrator handles the push.
- Append this entire commission prompt verbatim as "Appendix A — commission prompt" in the document (consult-record discipline; `docs/consults/consult-001-prompt.md` is the format reference). Match the existing design notes' style.
- Honest, mechanistic, scoped. Where uncertain, say so.

## Your final message back to me

Your returned message **IS the record** — make it a complete, self-contained rendering of the synthesis: the per-component exact-linearization verdict (belief updates / search / features / forward), where exactness holds vs. breaks, the batching axes and how they compose, the structural language-vs-Python analysis, and the ranked overarching synthesis with an honest which-1-3-to-pursue verdict. Also report the exact branch name, commit SHA, and file path. Render the substance; do not make it a pointer to the file.
