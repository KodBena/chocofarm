# C++ Gumbel-AZ search port — scoping (2026-06-15)

A scoping record (authored at decision time, per the documentation discipline),
produced by an assessment workflow before any porting. It maps the
`GumbelAZSearch` algorithm, the parity-critical `value_target` math, and the
validation strategy for a faithful C++ port.

**Framing.** The C++ runner is **policy-agnostic**: the search algorithm is a
swappable `Policy` behind the composable seam the MVP established
(`decide(env, loc, bw, collected, lam, rng)`), mirroring the Python `solvers/`
family. This note scopes the **Gumbel-AZ search policy** specifically — the
net-using AZ self-play search. It is one impl in the roster; the classical
policies (NMCS, ISMCTS) are simpler (no net, no Sequential-Halving, no
improved-policy σ-transform) and are ported first for the NMCS-initialize→ISMCTS
research milestone. Effort below is described by **complexity / risk / parity
surface**, not calendar time.

> **Not research-risky — faithful transcription plus a two-tier parity harness,
> not invention.** Every kernel is already a pure, unit-testable function
> (`value_target.improved_policy`/`v_mix`/`sigma_scale`/`blended_returns_to_go`,
> audit item C), and the four seams (env, the `NetEvaluator` leaf-eval port, the
> redis bytes transport, the version-gated weight broadcast) are clean. Risk
> concentrates in one place above all: the masked-softmax per-row argmax flipping
> on a near-tie, fed by the deliberate float32-prior / float64-Q mixed precision
> in `v_mix` — a one-ULP difference can move probability mass, and because the
> leaf forward is only behaviorally equal (~1e-4), it cannot be caught
> end-to-end; it needs a dedicated near-tied-logit kernel test.**

## 1. The algorithm (`az/gumbel_search.py`)

**`_Node`** (`__slots__`): `W` (action→summed λ-penalized return), `N`
(action→selection count), `children` ((action, belief_key)→`_Node`), `prior`
(masked-softmax prior, float32), `value` (scalar net V), `feat`, `mask`, `legal`.
`q(a) = W[a]/N[a]` or `0.0` unvisited.

**`_decide_root`** (one decision):
1. First-decision belief-cache reset iff `len(bw)==len(env.worlds)`.
2. `root=_Node(); _evaluate(root)` — one net forward sets prior/value/feat/mask/legal.
3. `legal_slots = [action_to_slot(a) for a in root.legal]`.
4. `logits[s] = log(max(prior[s], 1e-12))` for legal slots, `-1e30` elsewhere (Danihelka works in logit space).
5. `g = rng.gumbel(n_slots)`; `score0 = logits+g` (legal only); `considered = top-m of score0`, `m = min(self.m, #legal)` — Gumbel-Top-k sampling-without-replacement.
6. `survivor = _sequential_halving(root, considered, g, logits)` — spends all `n_sims`, returns the bracket winner.
7. `improved = _improved_policy(root, logits, legal_slots)` — the π′ target.
8. Executed action: `temperature==0 → slot_to_action(survivor)`; `>0 → rng.choice` over the temperature-reshaped improved.
9. Return `(action, improved, root)`.

**`_sequential_halving`**: `n_phases = ceil(log2 m)`, `per_phase = n_sims//n_phases` (≥1); each phase splits an equal share among survivors, then keeps the top half by `g[s]+logits[s]+sigma*q` (σ recomputed each phase as `max_a N(a)` grows); a remainder loop spends the **full** `n_sims` exactly.

**`_visit` → `_simulate_root_action`**: per sim, `w = sample_world(bw, rng)`; TERMINATE → `-λ·exit_cost`; else average over `c_outcome=2` leaf determinizations (k=0 reuses the threaded world), each `step = r - λ·dt` plus `_descend` continuation; accumulate `W[a]`, `N[a]`.

**`_descend`** (interior, ISMCTS — one threaded world per sim): depth/empty guards; an unexpanded leaf returns the **net value** (no rollout — the F4 cure); else `_puct_select` (`q + c_puct·p·√ΣN/(1+n)`, strict-`>` argmax, unvisited Q completed by the node's own net value), `apply`, recurse, backup.

**`_root_search_value` = ΣW/ΣN** (visit-weighted) — the value-target bootstrap.

## 2. The parity-critical math (`az/value_target.py`)

- **`sigma_scale`** = `(c_visit + max_a N(a))·c_scale`. Integer max-reduction → one add, one mul. **Robust** (no float tie).
- **`v_mix`** = `(v_net + ΣN·v̄)/(1+ΣN)`, `v̄ = Σ_{N>0} π(b)Q(b) / Σ_{N>0} π(b)` (**prior**-weighted, not visit-weighted). `sum_n` is forced to a Python `int` so `sum_n·v̄` keeps `v̄`'s dtype (the float32-promotion byte-identity contract). The prior-weighted accumulation is a **reduction over visited legal slots** — order- and dtype-sensitive.
- **`improved_policy`**: `completed[s] = logits[s] + sigma·q` (q = per-slot Q if visited, else the single scalar `vm` for every unvisited legal slot), then a **masked softmax** (`mlp.py:251-261`): subtract the per-row legal max, `exp`, zero illegal, normalize.

**Float-sensitivity (where a C++ reorder/dtype change can diverge):**
1. **The masked-softmax per-row max/argmax** — the classic near-tie hazard; two legal completed-logits within a few ULP can flip which is the row max, shifting every `exp` argument. *The single most reorder-fragile spot.*
2. The softmax `exp`/`sum`/normalize — `exp` is libm-dependent; the row-sum denom is an order-sensitive reduction.
3. `v_mix`'s prior-weighted `pw_num`/`pw_den` — two reductions whose ULP drift scales into **every** unvisited slot's completed Q (vm is broadcast), feeding hazard #1; the float32 weak-promotion makes ULP divergence *more* likely.
4. `completed[s] = logits[s] + sigma·q` — fma-vs-mul+add and float32-vs-float64 round differently in the last bit, which is exactly what can flip the argmax in #1.

**Robust:** `sigma_scale` (integer max), `sum_n` (exact int), the `v_mix` final combine, illegal-slot output (exactly `0.0` by mask + `-1e30` init), and the `N>0` branch predicates.

## 3. Parity strategy — two-tier (NOT byte-match)

**RNG: behavioral-only.** Bit-matching numpy's PCG64 + Gumbel + `choice` stream is feasible in principle but a large brittle reimplementation surface (PCG64 advance, `next_double`'s 53-bit assembly, the Gumbel `-log(-log(U))` transform with its `U==0` rejection, the Lemire bounded-int and cumsum+searchsorted `choice` paths, numpy's vectorized fill order) — and **explicitly not required**: the scaling note §3 records that a standalone actor *relaxes* the parallel≈serial bit-determinism, keeping only "each episode is valid self-play, per-episode reproducible from its own seed." A C++-native PCG64 (or any well-seeded RNG) with the same distributions suffices.

- **Tier 1 — deterministic, fixed inputs, no RNG.** Bit-exactable (same ops/order): env/belief mechanics, the `W`/`N` q-backup given a fixed realized-return+visit sequence, `_puct_select`, the SH cut key, the value-target pure-MC suffix path, `_root_search_value`. Tolerance-bounded (<1e-4): the leaf forward (behaviorally equal), and **the improved-policy/v_mix masked-softmax tail tested explicitly at near-tied completed-logits** (the parity-critical case). Drive the search-skeleton tests with an **injected fixed world/gumbel stream** so schedule, survivor, and backups validate without RNG.
- **Tier 2 — aggregate, RNG-driven.** No trajectory byte-compare. Validate **distributions** — the π′ target, the root-value-target, and the greedy eval rate (`dinkelbach_rate`, E[R]/E[T]) — over a large episode batch within Monte-Carlo CI, mirroring `bench_equivalence.py`. Pin the three Danihelka invariants per-episode structurally (executed==SH-survivor; v_mix prior-weighted; SH spends full budget).

## 4. Staged plan (by complexity / risk)

- **Stage 0 — NetForward (done).** The leaf-eval primitive, behavioral <1e-4. *Behind the `NetEvaluator` port*: the interim native impl; the SSOT-respecting impl is the `ZmqNetClient`→Python batched server. This is the parity FLOOR — end-to-end bit-match is impossible above it.
- **Stage 1 — env/belief mechanics.** `apply`/`sample_world`/`exit_cost`/`filter_*`/`marginals`/`legal_mask`, the slot↔action bijection, `_belief_key=(len,bw[0],bw[-1])`. **Bit-exactable, low-risk** — deterministic uint-world-set arithmetic, fully unit-testable. (Much already exists in the MVP env port.)
- **Stage 2 — the pure kernels.** `sigma_scale`/`v_mix`/`improved_policy` (with `mlp.py`'s exact masked-softmax, the float32/float64 precision seam) + `suffix_returns_to_go`/`blended_returns_to_go`. **Low conceptual risk, high attention-to-detail** on the float seam. Test to <1e-4 at near-tied inputs.
- **Stage 3 — tree + deterministic skeleton (no RNG).** `_Node`, `_puct_select`, `_descend`, `_simulate_root_action`, the q-backup, `_root_search_value`. **Moderate** (the dual q-completions, children keying). Validate by injecting a fixed world/draw sequence.
- **Stage 4 — Gumbel-Top-k + Sequential-Halving (RNG injected).** Root logits, the single `g` reused across top-k and every halving key, the SH budget arithmetic + remainder loop. **Moderate, error-prone** (the budget/sentinel dance). Drive with a fixed `g` stream first.
- **Stage 5 — real RNG + executed-action/temperature path** (behavioral-only).
- **Stage 6 — the episode runner** (`generate_episode` in C++): the `n_dec`/`n_rec`/trailing-TERMINATE bookkeeping, `blended_returns_to_go` value targets, the empty-belief guard. **Off-by-one-sensitive** — silently corrupts training data if wrong.
- **Stage 7 — transport integration**: read weight bytes (version-gated), write transition bytes over the result-format wire. **Gated on `#23`** (the result-format codegen) so the C++ writer targets the final schema.
- **Stage 8 — full two-tier parity harness + invariant pinning.** The near-tie kernel tests + the large-batch aggregate distribution runs.

## 5. The hard parts

1. **The masked-softmax near-tie argmax** (above) — caught only by a dedicated near-tied-completed-logit kernel test, never end-to-end.
2. **The float32/float64 mixed-precision seam** in `v_mix`/`improved_policy` — replicate per-operation (float prior-weighted product, double `sigma·q`, double `completed`).
3. **The two distinct q-completions that must NOT be unified**: PUCT completes unvisited interior Q with the node's **own net value**; `improved_policy` completes unvisited **root** Q with `v_mix` (prior-weighted); the root bootstrap is ΣW/ΣN (visit-weighted). Three reads of the same stats, three answers.
4. **The SH budget arithmetic + the two-threshold sentinel dance** (`-1e30` logits vs `-inf` score0; `g` drawn once; σ recomputed per phase; the remainder loop guaranteeing Σvisits == `n_sims`).
5. **The value-target form**: `blended_returns_to_go` with `boot[D]=-λ·exit_c`, the trailing-TERMINATE target forced to `-λ·exit_c` (not its boot), the `n_dec`/`n_rec` bookkeeping.

## 6. What the MVP + NetForward already give

The env port, the composable `Policy` seam, the redis raw-bytes transport, the version-gated broadcast, and the `NetForward` leaf eval. R8 collapsed the belief mechanics to one `Environment.restrict`; R11 collapsed the forward to one `forward_core` — so there is **one** of each surface to mirror. Crucially, the policy/value-target rules are **already extracted as pure unit-testable functions** (item C) precisely so they port and validate without a tree — the MVP did the decomposition that makes Tier-1 parity tractable.

## 7. Open decisions — status

- **RNG: behavioral-only** (recommended; the design sanctions the relaxation). *Settled.*
- **Standalone actor (Shape A)** first (the JAX learner unchanged; the async loop is separate). *Settled.*
- **`#23` result-format codegen before Stage 7** so the C++ writer targets the final wire. Stages 0-6 don't depend on it. *Settled.*
- **The MLP is Python, not C++.** Per the maintainer: leaf inference is a Python-hosted **batched ZeroMQ service** (Shape B — exact cross-episode batching, ~1e-4) so net architecture stays in JAX (flexible for architecture/hparam search). The C++ `NetForward` is the **interim** fast impl behind the zero-cost `NetEvaluator` port — a named, parity-pinned, temporary ADR-0012 deviation, retired when the ZMQ path suffices. *Settled (`#27`).*
- **Policy-agnostic roster.** NMCS (the milestone initializer, slowest-in-Python) → ISMCTS → this Gumbel-AZ search, all interchangeable `Policy` impls. The numba-vs-C++ question is moot — its draw was architecture flexibility, which the ZMQ MLP gives directly. *Settled.*

## 8. Status

Scoping only; no porting in this note. The classical policies (NMCS/ISMCTS) are
ported first per the research milestone; this Gumbel-AZ scoping stands for the
AZ-self-play policy, built behind the same `Policy` seam and the `NetEvaluator`
port.

*Public Domain (The Unlicense).*
