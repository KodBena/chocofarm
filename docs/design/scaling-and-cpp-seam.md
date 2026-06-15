# Deployment shapes and the C++ seam — what the architecture should make fall out (2026-06-15)

A forward-looking design record (authored at decision time, per the documentation
discipline), not yet built. It names three deployment shapes the maintainer intends
to keep reachable, and states — honestly — which fall out of the current architecture
for free and which need one named restructure. The point is to give the remaining
roadmap (`R12` RunConfig, `R14` numpy-only worker, and the Transport⊥Pool⊥Task split)
a target to be checked against, and to record the deliberate trades so a later reader
does not mistake a relaxation for a regression.

It composes with — does not contradict — `docs/design/simulation-parallelization-viability.md`
(the Axis A / B / C exactness-×-payoff analysis) and `docs/design/architecture-refactor-audit.md`
(§2.5/§3.6 Transport⊥Pool⊥Task, §5 the C++-sim seam). Where this note says "falls
out for free," it means "is a composition of seams that already exist, not new
plumbing."

> **The three shapes are compositions of four already-clean boundaries — the
> `env`↔`Policy` seam, the net-as-injected-port, the redis raw-bytes transport, and
> the version-gated weight broadcast. None needs a foundational change. The single
> asterisk is the synchronous `generate → train` loop: the continuous concurrent
> actor-learner shape needs that loop re-wired into decoupled learner + actors — a
> localized, R12/R14-enabled restructure, not a rewrite.**

---

## 0. The four seams these rest on

1. **`env` ↔ `Policy`** — the simulation surface (`apply`/`filter_treasure`/`filter_detector`/`marginals`/`d`/`exit_cost`/`simulate`) is a pure function of `(state, action, world) → numbers`; no training/optimizer/feature/target type crosses it. R8 collapsed it to *one* implementation (`Environment.restrict`, no `MiniEnv` copy), so there is one surface to reimplement.
2. **The net as an injected port** — the search holds the net as a dependency and uses only `predict_both` / `predict_value`. R11 collapsed the forward to *one* `forward_core(params, X, xp)`, so there is one function to reimplement and one interface to proxy.
3. **The redis raw-bytes transport** — weights cross as `tobytes()`, transitions return as `tobytes()`, the hot knobs (`m`/`n_sims`/`lam`/`max_steps`) cross as scalars. A key→number map + raw weight bytes is language-agnostic by construction.
4. **The version-gated weight broadcast** — `_ensure_net` reloads the net only when the published weight version changes. This *is* the actor-learner weight-publish mechanism, already in the tree.

---

## 1. Shape A — a C++ worker for search + sim

A worker that runs the Gumbel-AZ search and the belief mechanics in C++ (or numba),
reading weight bytes from redis and writing transition bytes back.

- **Falls out behind seams 1, 3, 4.** Nothing Python-specific crosses the worker boundary: it reads weight bytes, runs an episode, writes transition bytes. A C++ worker speaks the same redis protocol; the JAX parent (the learner) does not change a line. This is the prior audit's §5 claim, and R8/R11 shrank the surface a C++ core must reimplement (one belief-mechanics impl, one `forward_core`).
- **Prerequisite already on the roadmap:** R14's jax-free worker is the move that makes the worker a *clean numerical host* — a tight compiled inner loop cannot share a process with XLA's thread pool (that interaction is what the deadlock RCA fought). R14 is therefore the enabler, not a competitor, of the C++ ambition.
- **Per the sim-parallelization note, this is the #2-conditional-on-latency lever**, and a Python-resident numba columnar tree core captures most of a full C++/Rust port's gain (the win is layout + dispatch, not language). So the natural path is numba-first, C++ for the last dispatch slice — both behind the same four seams.

## 2. Shape B — inference over a Python-hosted batched ZeroMQ service

The C++ worker offloads leaf inference to a central Python server that batches
requests from many workers and runs one forward (where batched numpy/JAX is fast).

- **Falls out behind seam 2.** A `NetClient` that RPCs `predict_both`/`predict_value` over ZeroMQ is a drop-in for the local numpy net behind the same interface; the search is unchanged. The only formalization is making "the net" an explicit port with `local-numpy | zmq-client` implementations — R11's single `forward_core` and the prior audit's `net.forward()`-factory point (§2.6) are that shape already.
- **The batching is the *exact* kind, not the approximate kind.** A server batching leaves from N *independent* workers runs `forward_core` over a stacked `(B, in)` matrix; a row of a batched matmul is the same row-wise-independent dot product as the single-row call. This is cross-episode batching (the sim-note's **Axis A**), which never touches any search's Sequential-Halving budget, RNG order, or the Danihelka invariants. It carries only the *forward-roundoff* non-exactness the project already accepts (f32/jit, `test_jax_equivalence` ABS_TOL=1e-4) — **not** the *approximate-search* non-exactness of within-search leaf batching (**Axis C**), which the project rightly defers. So this gets "batched inference is fast in Python" without re-opening the fidelity surface.
- **Workers stay dumb.** Each worker makes a blocking `evaluate(X) → (v, p)` RPC; the *server* batches the concurrent in-flight requests in a small time window. No async machinery, no virtual loss, in the worker.

## 3. Shape C — continuous concurrent generation + training

Actors generate self-play continuously against the latest published weights and push
transitions to a streaming buffer; the learner trains continuously and periodically
publishes updated weights. The standard scalable actor-learner.

- **The primitives are there:** the version-gated weight broadcast (seam 4) is the weight-publish channel; the raw-bytes transition transport is the actor→buffer channel. Off-policy staleness (actors a few versions behind) is already the contract `_ensure_net` implements.
- **The one thing that does NOT fall out for free: the loop structure.** `exit_loop` is synchronous-orchestrated — `for it: generate → train → eval → checkpoint` — so actors idle during training and the learner idles during generation. Decoupling into a continuously-running learner and continuously-running actors over a streaming buffer is a genuine restructure, not a byproduct. *No boundary blocks it* — the seams compose straight into it — but the synchronous loop is the seam that is currently the wrong shape, and turning it is real work.
- **It is localized, roadmap-enabled work, not a rewrite.** R12 (`RunConfig`) gives learner and actor as first-class configurable units; the Transport⊥Pool⊥Task split gives the actor a legible boundary; R14 (jax-free `Worker` object + `(run, phase, version)` key namespacing) is the actor decoupled from the synchronous parent. The async actor-learner is then a *re-wiring* of components that already have the right boundaries.
- **Deliberate trade to record:** a continuous async loop **relaxes the parallel≈serial bit-determinism** the synchronous loop guarantees via the `_task_rng` seed-fold. Per-episode exactness is kept; bit-identical *aggregate* reproducibility is traded for throughput. This is the correct trade for a throughput-oriented continuous system — recorded here so it is not later mistaken for a regression in the determinism property `az-parallel-exp.md` verifies.

---

## 4. The enabler checklist

What fully discharges all three shapes — each a roadmap item, none foundational:

- **A `Net` port** (`predict_both`/`predict_value`) with `local-numpy | zmq-client` impls. *(Unblocks Shape B; ~the `net.forward()`-factory follow-up, near-free after R11.)*
- **Transport ⊥ Pool ⊥ Task split** (prior audit §3.6) — makes the worker boundary legible, so "swap the worker (or its inner loop) for C++" is a drop-in. *(Shapes A, C.)*
- **`RunConfig` (R12)** — learner/actor as configurable processes. *(Shape C.)*
- **Numpy-only `Worker` (R14)** — clean numerical host + decoupled actor. *(Shapes A, C.)*
- **The sync→async loop restructure** — the only genuinely new structure; everything above feeds it. *(Shape C.)*

---

## 5. Status and caveats

- **Forward-looking, not built.** No code change accompanies this note; it is a target.
- **Ranking, per the sim-parallelization note:** for the program's actual currency (rollout *throughput* → more calibration experiments), widening the exact cross-episode fan-out (**Axis A**) is #1; the C++ inner core is **#2, conditional on per-decision *latency*** becoming the bottleneck; GPU within-search leaf-batching (**Axis C**) is gold-plating at the current tiny-MLP scale. Shapes A/B serve throughput-via-batched-inference and latency; Shape C serves throughput directly. None argues for a foundational change.
- **Single instance, uncalibrated time model** — the standing chocofarm caveat; it does not affect this structural analysis (about boundaries, not instance numbers).

*Public Domain (The Unlicense).*
