<!-- docs/design/cpp-batched-search.md -->

# C++ batched/async Gumbel-AZ search: the cross-tree fan-out regime

**Status:** Design record (forward-looking). Reviewable artifact — contracts before code.
Composes from already-built seams (Shape B inference service, the env↔Policy and `Net`
ports, the `value_target` kernels); the only genuinely new structure is the worker-side
multiplexer + result router and (on the C++ side) a port of the not-yet-ported Gumbel-AZ
search. Read end to end before implementation.

**Scope is settled and is not re-opened here.** The regime: a pool of worker threads, each
advancing its **own independent search tree** for one problem instance. NOT a shared tree,
NOT root parallelization, NOTHING that perturbs a single tree's search distribution. A tree
carries **at most one outstanding leaf** and is not re-selected until that leaf has
backpropped (serial-per-tree). Trees rendezvous **only** at the central Python ZeroMQ
batched evaluator (the Shape B service: `predict(X) → (de-standardized value, RAW logits)`).
Workers multiplex across **many** trees to keep the batch full and hide inference latency.
A separate work-stealing backprop pool may apply returned results. The bar is **behavioral
float32-equivalence (~1e-4), not bit-identity**. Virtual-loss / shared-tree / root-parallel
are **out of scope** — they perturb the distribution.

---

## 0. The maintainer's idea, restated faithfully

> "Create a pool of worker threads, each of which may fetch a ground-truth, unseen, as
> context in the simulation [the determinization world]. It then follows the selection
> criterion to a leaf and pushes the leaf over ZeroMQ, then goes on to fetch the next work
> unit until a maximum number of in-flight work units. On the python side, they are batched,
> results are sent back to a waiting work-stealing pool of backproppers. Thus, the
> backpropagation workers do only that, and the tree selection/diving do only that (and
> submit over ZeroMQ)." Clarified: **independent trees** (not the same tree), no root
> parallelization, nothing that affects the search distribution.

This note grounds that idea in the **real** Gumbel-AZ search (`chocofarm/az/gumbel_search.py`,
read end to end), states precisely **why** the regime is the *exact* path (it is **Axis A**
— cross-tree/cross-episode batching — wearing a multiplexing coat), and lays out the
concrete architecture. It is honest about the one slot where the maintainer's framing needs
amendment: a *separate* selection pool and *separate* backprop pool is the weaker split; a
**single unified work-stealing pool over heterogeneous {SELECT, BACKPROP} tasks** with a
per-tree state word is the recommendation (§3). The "backprop workers do only that" intent
is preserved — any worker that picks up a BACKPROP task does only backprop on that task — but
the threads are not *partitioned* into two populations.

---

## 1. How THIS search actually works (where most readers will be wrong)

The vocabulary here ("Sequential-Halving phases", "m candidates", "backprop pool") is
MCTS/Gumbel-AlphaZero vocabulary. **That search is the deferred Shape A search**: per
`cpp/include/chocofarm/{env,net}.hpp`, the Gumbel-AZ search + MLP forward are *not yet*
ported to C++ — the only search currently in C++ is NMCS (a forward recursion with in-stack
memorize-and-replay, no per-node backprop). So this design composes built seams with a
search that, on the C++ side, is still to be written. The reference for its behavior is the
**Python** `gumbel_search.py`, and that is what the C++ port must match to ~1e-4. Everything
in §1 is grounded in that file (line citations are to `gumbel_search.py` unless noted).

### 1.1 One decision = Gumbel-Top-k → Sequential Halving → improved-π

`_decide_root` (gumbel_search.py:214) runs, in strict order:

1. **Root leaf eval.** `_evaluate(root, …)` (:235 → :146-166) does **one** net forward and
   caches `node.feat/mask/prior/value/legal`. `prior` is the **masked-softmax** over the raw
   logits; `value` is the de-standardized scalar leaf estimate (the F4 cure — net value at the
   leaf, no playout). This node-level cache is the F7 amortization: a belief reached again in
   the same tree is **not** re-evaluated (the `node.value is None` gate).

2. **Gumbel-Top-k.** One `g = rng.gumbel(size=n_slots)` over the **full** slot space (:250),
   then `considered = argsort(logits + g)[::-1][:m]` (:251-253), `m = min(self.m, n_legal)`.
   `logits = log(prior)` over legal slots (:245-247) — i.e. the logits the Gumbel machinery
   sorts on **are** `log` of the in-search float32 masked-softmax prior.

3. **Sequential Halving** (`_sequential_halving`, :282-327). `n_phases = ⌈log2 m⌉`,
   `per_phase = n_sims // n_phases`, `per_action = phase_budget // len(considered)`. Each
   phase runs `per_action` sims per surviving candidate, then **drops the worst half** by the
   key `g[s] + logits[s] + σ·root.q(a)` (:314-316), where `σ = (c_visit + max_a N(a))·c_scale`.
   Rounding remainder is spent round-robin on the last phase's survivors so the **full budget**
   is used (:320-327). Defaults: `m=12, n_sims=48, c_puct=1.25, c_visit=50, c_scale=1.0,
   c_outcome=2, max_depth=24`.

4. **Improved policy** `π′ = softmax(logit + σ(completedQ))` over the **full legal set**
   (`_improved_policy`, :476-485 → `value_target.improved_policy`). Unvisited actions have Q
   "completed" by `v_mix` (prior-weighted, Danihelka §3).

5. **Executed action.** At `temperature == 0` it is **the SH survivor** (:278). At
   `temperature > 0`, one `rng.choice(n_slots, p=probs)` samples from `π′` (:272). The
   improved-π **target is unaffected** by temperature.

The three Danihelka invariants — `test_executed_action_is_sh_survivor`,
`test_vmix_prior_weighted`, `test_sequential_halving_spends_full_budget` — are the project's
"fidelity immune system" and were broken by an earlier implementation and caught out-of-frame
(`az-exit-loop.md §(f)`). Any reformulation that perturbs SH budget accounting must
re-validate them.

### 1.2 The leaf-eval points and the per-tree serial chain

Every leaf is **one** net forward, blocked-on:

- **Root leaf** (:235): one forward sets `prior/value` before Gumbel-Top-k and SH begin. The
  whole decision waits on it.
- **Interior leaf** (`_descend`, :363-395): an unexpanded interior node returns the net value
  as the leaf return (:372-373). `_puct_select` (:397-426) reads the cached `node.prior` and
  the running `W/N` to pick the interior child via the strict-`>` argmax
  `q + c_puct·p·√(ΣN)/(1+n)` (:423-425).
- **Per-simulation outcome averaging** (`_simulate_root_action`, :339-360): each root-action
  realization averages the leaf over `c_outcome=2` immediate determinizations; each calls
  `_descend(depth=1)` which bottoms out at **one** leaf per net call.

**The load-bearing fact:** within ONE tree, simulation `k+1`'s PUCT descent and the SH halving
read the backprops of sims `0..k`. `_descend` writes `W[a]+=ret; N[a]+=1` *after* the leaf
value returns (:393-394); `_puct_select` (:420-421) and the SH key `root.q(a)` (:316) read
that running mean on the next step. So a tree is **strictly serial by construction**: it
issues exactly one leaf eval, blocks for the value, backprops, then selects the next. There is
**no** intra-tree parallelism over the `m` SH candidates — `_visit` runs one slot's `count`
sims fully before the next slot (:307-311). This is the **exactness mechanism**, and it is
exactly what the maintainer's "at most one outstanding leaf, re-select only after backprop"
rule names.

### 1.3 The float32 hazards — what the deep read established vs left open

Three things are **load-bearing precision contracts**, verified against the code:

- **The masked-softmax prior is computed IN-SEARCH, in float32 — NOT on the wire.**
  `_evaluate` → `net.predict_both` → `_predict_both_f32` (mlp.py:221) computes the value AND
  the masked-softmax prior in its **own float32 tail** (mlp.py:243-248: `np.float32(-1e30)`,
  per-row legal max subtract, `exp*legal`, normalize, denom-guard). The Shape B service
  returns **de-standardized value + RAW logits only** (net_port.py:16-28; mlp.py:240-241 is
  the value de-std). **A C++/batched search routing through the service must reproduce this
  masked-softmax client-side, in float32, to match the in-process reference.** This is the
  single most likely place a "looks exact" port silently diverges, and it is **orthogonal to
  batching** — the service never sees the mask.

- **The float32-prior / float64-Q mixed-precision weak-promotion seam.**
  `value_target.{v_mix, improved_policy, sigma_scale}` are byte-identical to the formerly
  welded rule **only** because the caller passes `visited_q`/`visited_n` as plain Python
  `float`/`int` (gumbel_search.py:442-448; value_target.py:209-213, 237-240). numpy's
  weak-scalar promotion keeps `prior[s]·visited_q[s]` at **float32** while `σ·q` in the
  completion keeps Q's **float64** magnitude; `v_mix` forces `sum_n = int(...)` so
  `sum_n·v_bar` stays at `v_bar`'s float32 dtype. A uniform-float32 or uniform-float64 C++
  port will diverge here, possibly **beyond 1e-4** at the improved-π target. Note a subtlety
  the deep read pinned: the improved-policy softmax tail uses `_masked_softmax` (mlp.py:264),
  which is **float64** internally (`np.float64(-1e30)`), whereas the leaf-eval prior softmax
  (`_predict_both_f32`) is **float32**. Both precisions are deliberate; transcribe each
  faithfully.

- **Near-tie discrete-choice flips — the headline hazard.** Three selections can flip under
  float32 roundoff: the Gumbel-Top-k `argsort(logit+g)`, the SH halving sort key, and the
  PUCT/SH-survivor strict-`>` argmax. A ≤1e-4 perturbation at the leaf value/logits **can**
  flip which root action survives SH or which interior child PUCT descends — a *legitimately
  different* trajectory.

**What the deep read left open / what is distributional, not bitwise:** the search's
randomness is numpy PCG64 (`rng.gumbel`, `rng.choice`/`env.sample_world`). These are **not**
naturally bit-reproducible in C++ (numpy PCG64 ≠ `std::mt19937_64`). So for a C++ tree,
"preserve the distribution" means **distributional** equivalence over many episodes (the
`parity.py` posture), not draw-for-draw identity — unless the port mirrors numpy's bitgen +
exact draw order. The **draw ORDER per tree** must be replicated regardless: one
`rng.gumbel(size=n_slots)` at root, then per-sim `sample_world` with `c_outcome-1` extra
`rng.choice` (:350), then optional temperature `rng.choice` (:272).

---

## 2. The regime is the EXACT path (Axis A), not the approximate one (Axis C)

### 2.1 Why cross-tree batching preserves each tree's distribution

The settled Axis framing (do **not** re-litigate): **Axis A** = cross-episode/cross-tree
batching = **exact** (the #1 throughput lever, Python half **already built**); **Axis B** =
across-world-set rows of a belief reduction = exact, already realized within a leaf,
negligible standalone payoff; **Axis C** = within-search leaf batching =
**approximate-only**, deferred, Amdahl-capped, and re-opens the Danihelka fidelity surface.
**Virtual-loss / shared-tree / root-parallel are OUT** — they perturb a single tree's
distribution.

This regime is **Axis A**. The argument, grounded in three invariants honored *by
construction*:

1. **Serial-per-tree is the exactness mechanism (§1.2).** Each tree blocks on its single
   outstanding leaf before re-selecting, so every PUCT/SH decision sees the same accumulated
   `W/N` it would in-process. The per-tree trajectory is identical to its serial trajectory
   **given identical leaf returns**.

2. **The batch is cross-tree, so it is row-independent.** `run_microbatch`
   (`inference_server.py`) stacks B rows from B **distinct independent trees** (each on its
   own numpy `Generator`) into `(B, in)` and runs **one** `forward_core`. A row of the batched
   matmul is the same row-wise dot product as the single-row call (the Axis-A justification in
   `zmq-inference-service.md §4`, measured `max|Δvalue|≈4.8e-7`, `max|Δlogit|≈2.4e-7` over
   N=1200, B∈{1..64}, residual on/off — two-plus orders inside the 1e-4 P6 bar). Batching
   never touches any tree's SH budget, RNG order, or the Danihelka invariants.

3. **Per-tree RNG order is untouched.** Cross-tree multiplexing interleaves **different**
   Generators' draws — independent streams. A single tree's `gumbel → sample_world →
   temperature` order on **its** Generator is never reordered, because the tree is serial.

Because **no tree ever has two leaves in flight**, the design **cannot silently slide into
Axis C** — the one forbidden failure mode is structurally unreachable, provided the
one-outstanding-leaf invariant is a hard assertion (§3). The moment a worker issues a second
leaf from the same tree before backprop (to fatten the batch), it has become virtual-loss /
Axis C and perturbs the distribution.

### 2.2 The one honest trade

Batched-vs-serial leaf numerics carry up to ~1e-4 of roundoff from three accepted sources:
(a) the service computes the value under JAX/XLA while the in-process reference uses numpy
sgemm (both float32, within `test_jax_equivalence` ABS_TOL=1e-4); (b) row-vs-single matmul
roundoff; (c) **batch-composition** roundoff — *which* other trees co-batch depends on arrival
timing under the greedy drain, so the low bits of a tree's leaf value depend on its batch
neighbours. At a **near-tie** (§1.3), that ≤1e-4 perturbation **can flip** the discrete choice
on an individual tree — a different SH survivor or PUCT descent.

**This is distribution-faithful, not bit-reproducible per decision.** It is explicitly **on
record**: `scaling-and-cpp-seam.md §2.5` (the Shape C trade) states a continuous async loop
**relaxes** the parallel≈serial bit-identical *aggregate* reproducibility that the synchronous
`_task_rng` seed-fold guarantees — **per-episode exactness is kept; aggregate bit-identity is
traded for throughput, recorded so it is not mistaken for a regression.** `zmq §4` binds the
batch-composition nondeterminism to this same trade. So this design is **sanctioned** to relax
aggregate bit-reproducibility but **must** preserve each tree's behavioral float32-equivalent
(~1e-4) distribution. The correctness bar is `parity.py`'s aggregate over N≥300 episodes within
Monte-Carlo CI, **not** per-decision identity — and the design **must not promise** per-decision
identity. A deterministic drain (fixed B + barrier) is the available-but-non-default escape
hatch when exact aggregate reproducibility is wanted (e.g. a single-tree determinism harness,
§5).

---

## 3. The concrete architecture

### 3.1 Per-tree state machine

Each tree owns: an atomic **state word** ∈ {READY, SELECTING, AWAITING_LEAF, BACKPROP,
DECIDED, FAILED}; its **own** numpy `Generator` (C++: its own `std::mt19937_64` / mirrored
PCG64); its **own** `_Node` graph; and a **parked-path** pointer (the descent awaiting a leaf
value). Transitions:

```
READY ──CAS──▶ SELECTING ──(advance recursion to a leaf, build feat row, park path)──▶
AWAITING_LEAF ──(submit feat row + correlation id; one leaf in flight)──▶
   [reply routed back] ──CAS──▶ BACKPROP ──(client-side float32 masked-softmax + de-std value
                                            + v_mix/improved_policy + W/N backup)──▶
READY ──(if sims/phases remain, enqueue SELECT) | DECIDED (emit survivor + improved-π)
FAILED ◀── (typed RPC failure → abort this tree's episode, loudly)
```

**The whole "no re-selection until backprop lands" guarantee is one assertion:** a tree is
present in the ready-queue **at most once** at any instant, and only when READY. Per-tree
in-flight is **strictly 1**, enforced because the tree is suspended at AWAITING_LEAF and
cannot issue a second `predict`.

### 3.2 Thread pools: UNIFIED work-stealing, not split

**Verdict: adopt a single unified work-stealing pool over heterogeneous tasks
`{SELECT(tree) | BACKPROP(tree, NetPrediction) | FAIL(tree, err)}`; reject the maintainer's
separate selection-pool / backprop-pool split.** Both models are CPU-side and neither touches
an exactness invariant — the choice is utilization and complexity, not fidelity. The case for
unified:

- **Backprop is cheap; selection (the descent) is expensive.** Backprop is a few `W/N`
  updates plus the client-side masked-softmax + `v_mix`; selection is the `_descend` PUCT loop
  over `node.legal`, `env.apply`, `_belief_key`, and a fresh feature build. A *dedicated*
  backprop pool is therefore mostly-idle threads competing for cores. On the 4-vCPU host
  (CLAUDE.md, parallel ceiling ~1.9×), you want `N = #cores` workers each doing the next ready
  task — a core never parks on an empty backprop queue while SELECT work piles up.

- **The invariant lives in the state word, not in which pool touches the tree.** In both
  models the serial-per-tree guarantee is "enqueued at most once, only when READY". The split
  needs the **same** state word **plus** a cross-pool handoff (selection pool → backprop pool
  on park; backprop pool → selection queue on re-ready) — two queues and an ownership transfer
  protecting the identical invariant the unified model protects with one queue. More moving
  parts, same guarantee, a wider surface for the "second leaf before backprop" bug that
  silently becomes Axis C.

- **Contention.** Tree state is per-tree, touched by one worker at a time by construction —
  no contention on tree state in either model. The contended structure is the task queue:
  unified has **one** (one lock, or per-worker deques with work-stealing); the split has
  **two** plus a handoff.

The maintainer's "backprop workers do only that" intent is **preserved**: a worker running a
BACKPROP task does only backprop on it. It is the fixed *partition* of threads into two
populations that is rejected.

**No XLA in a worker thread.** The server stays a single Python process, single-threaded
around one ROUTER (the `jaxtrain-deadlock-rca` / R14 invariant). All multiplexing and backprop
live worker-side. Scale-out is N stateless server instances behind a load balancer, **not**
threads in one process.

### 3.3 In-flight cap

Two caps, both load-bearing. **(a) Per-tree in-flight = 1**, structurally enforced (§3.1) —
the regime's defining invariant. **(b) Global concurrency = the number of trees with a leaf in
flight**, sized so the server's greedy drain stays near-full. Target steady-state batch
`B ≈ (#trees parked at a leaf per drain)`; the server `max_batch` caps the matmul. Steady-state
`B ≈ M × (round-trip fraction of a tree's cycle)`, so the number of in-flight trees must exceed
`(round-trip latency / per-leaf compute+backprop time)` to saturate. On the 4-vCPU host the
realistic ceiling is the ~1.9× host-contention wall, **not** the cap — do not over-provision
(each parked tree holds a full `_Node` heap).

### 3.4 ZMQ rendezvous + result routing (correlation IDs)

The built `ZmqNetClient` is a **blocking REQ** socket — strict `send → recv` lock-step,
**one in flight, not thread-safe** (zmq_net_client.py). It **cannot** multiplex many parked
trees on one socket. Two embodiments, **identical exactness**:

- **Embodiment 1 — thread-per-inflight (zero new code).** One OS worker per tree; at the leaf
  the worker **blocks** in its own `ZmqNetClient.predict`. The server's greedy drain batches
  whatever REQ requests are concurrently in flight. `B` is bounded by thread count and a blocked
  worker is a parked core — acceptable only when threads ≫ cores so IO-blocked threads overlap.
  **Build this first** if thread count, not the server, is the measured bottleneck.

- **Embodiment 2 — DEALER submit/poll multiplexer (the "multiplex MANY trees per thread"
  reading).** One multiplexer thread owns **one DEALER socket** (DEALER permits many
  outstanding sends — the parked-fibers shape) and K parked tree-fibers (stackful
  coroutines / greenlets in Python; boost.context-style or an explicit continuation refactor of
  the synchronous `WorldSource::playout_value` / `Net::predict` in C++ — the **one structural
  gap** the C++ map names). A fiber advances its tree to the leaf, hands the feature row to the
  DEALER with a **correlation id**, and **yields**; a small completion thread `recv`s replies
  and enqueues `BACKPROP(tree, prediction)` onto the **same** ready-queue. The DEALER client is
  the only genuinely new component — and the split threading model would need it too, so it is
  not a differentiator.

**Correlation, minimal-touch (ADR-0004).** The server is a ROUTER and already scatters by
identity frame; ROUTER↔DEALER is the natural pair and preserves **per-peer ordering**. So with
**one DEALER per multiplexer thread**, replies for that peer arrive in arrival order, and a
per-thread in-order queue of `(corr_id → fiber)` dequeued FIFO routes each reply correctly
**with no wire/codec change** (the current `inference_wire.py` frame is `[ver][in_dim][X] →
[ver][n_actions][value][logits]`, no id field; the ROUTER identity is the scatter key). Use
this. Only if replies could ever arrive out of order (they cannot with one single-threaded
ROUTER) add an explicit echoed `u32` request-id — a real codec amendment, fail-loud on
mismatch.

### 3.5 Reusing the seams (nothing new below the multiplexer)

- **`Net` port** (`net_port.py`): `predict(X) → (de-std value, RAW logits)`. `ValueMLPNet`
  (local) and `ZmqNetClient` (remote) satisfy it interchangeably — the zero-cost ACL. The
  masked-softmax prior is **not** in the port (stays at the consumer).
- **Shape B service** (`inference_server.py`): ROUTER + self-clocking greedy-drain (block for
  ≥1, drain all queued up to `max_batch`, one `forward_core`, scatter by identity — no latency
  timer, `B` tracks demand), version-gated reload between batches, mockable `ParamsSource`. The
  greedy drain already supports **either** caller shape (blocking-REQ-per-thread or DEALER
  multiplexer).
- **`inference_wire.py`**: the one length-prefixed LE-float32, version-headered codec;
  fail-loud `WireError` on malformed/non-finite.
- **`value_target.{improved_policy, v_mix, sigma_scale}`**: pure functions of explicit inputs,
  directly transcribable to C++, with the weak-promotion seam documented inline.
- **`FeatureBuilder` / `legal_mask_from_features` / `slot_action_tables`**: the float32
  feature vector and the **bit-exact** legal mask.
- **`worker.py` `_fold_seed`**: the per-tree/per-(version,kind,episode) seed fold (worker.py:120-140)
  — reuse the seeding discipline so each tree's stream is reproducible per logical episode.
- **C++ side**: the env↔Policy seam (`policy.hpp`), the `WorldSource` determinization seam
  (`nmcs.hpp` — the single injectable point for world sampling + the leaf value; a value-net
  leaf is a new `WorldSource` subclass with zero edits to `search()`/`eval_move()`), and the
  NMCS port as the worked example of a serial-per-tree search behind those seams.

### 3.6 Fail-loud at the ACL boundary (ADR-0002 / P9)

A timed-out / server-down RPC is a **typed** loud failure routed to the **owning** tree
(`ZmqNetClient` raises `InferenceClientError`; C++ returns `std::expected`). The multiplexer /
backprop pool **must not** silently drop a leaf or substitute a stale/zero value — that would
corrupt the tree's backup and its distribution. Embodiment 1 gets this free (the raise unwinds
the tree's thread). Embodiment 2 must route a failed correlation-id to a `FAIL(tree, err)` task
and abort **that** tree's episode, and must treat a missing/late reply as a per-fiber timeout
that resyncs by id — never let one bad frame desync the per-thread FIFO router (the one real
new failure mode the FIFO-correlation choice introduces).

### 3.7 Net version consistency

Version-gated reload is **between batches** (`inference_server.py`), so every leaf in one batch
sees one net version. A single tree's ~48 leaves **can** straddle a reload at batch boundaries
today (each `predict` uses whatever params are live). A mid-search version change perturbs that
tree's distribution. **Open decision (§6):** whether to pin one net version per tree-decision
(freeze weights during a generation phase). Orthogonal to threading.

---

## 4. The SH-phase intra-tree sub-batch — REJECTED as exact

The prompt invites combining cross-tree fan-out with an **intra-tree** sub-batch (the `m`
considered SH candidates' leaves evaluated together within one tree, to cut per-decision
latency). **This is rejected as a float32-exact optimization.** Grounded in the leaf-eval map
and the code:

- **The `m` SH candidates are evaluated SEQUENTIALLY today.** `_visit` runs one slot's `count`
  sims fully before the next (:307-311); `_simulate_root_action` and `_descend` read prior
  backprops within the same candidate (:393-394, :420-421). There is **no** set of ≥2
  independent leaves inside one tree without virtual-loss lockstep.

- **Grouping reorders the single RNG stream.** To submit the `m` candidates' leaves together
  you must pre-draw all candidates' `sample_world` calls up front. But a drawn world can be a
  TERMINATE world that short-circuits with `-λ·exit_cost` and **no net call** (:345-346), so
  pre-drawing **reorders** the single PCG64 Generator vs the serial baseline. This is **not**
  1e-4 roundoff — it is a **different RNG trajectory**: deterministic, but deterministically the
  *wrong* stream order. The map states it verbatim: "Batching/reordering draws WITHIN a tree
  breaks bit-exactness; the substrate may only interleave across independent per-tree streams."
  Labeling this "exact" is the load-bearing error to avoid.

- **It is Axis C — and Amdahl-capped.** Virtual-loss lockstep perturbs SH's per-phase budget
  accounting and can flip the executed-action = SH-survivor invariant. And even a *free*
  accelerator for the batched ~28% forward caps the inner-search win at ~1.3–1.55× (~72% of
  decide time is irreducibly-sequential tree recursion). Cross-tree fan-out (Axis A) gets
  ~linear scaling at **zero** fidelity cost; the SH sub-batch buys sub-1.55× by re-opening the
  fidelity surface.

**The one genuinely exact variant** is narrow: defer **only** the forward (keep serial draws +
serial backprop exactly as today, but accumulate the RNG-already-determined feature rows of a
phase and flush them as one batched request, resuming the serial backprop loop). This is exact
*only* where no candidate's later draws depend on an earlier candidate's leaf value — true for
`sample_world` and the post-phase halving key, but at defaults phase 1 has `per_action = 12//12
= 1` and later phases have `per_action ≥ 2`, so the intra-candidate sim chain serializes and
the exact-batchable width **shrinks** phase to phase (≈12 → 6 → 3 → 1 first-sims, minus
short-circuited candidates). It duplicates **inside** one tree the cross-tree drain the
evaluator already does, and does **not** compose additively with Axis A (a worker multiplexing
many trees already fills `max_batch` from independent streams).

**When it could help:** a **low-latency single-strong-search** corner (few trees, latency-bound,
remote evaluator) where the ~2.5–4× RTT-count cut on a *single decision* matters. **When it does
not:** the **many-tree self-play throughput** regime — the program's stated currency — where
Axis A already saturates the batch cross-tree and each tree staying strictly serial is free.
**Recommendation:** if ever built, scope it to an **opt-in eval-only low-latency mode** with a
**distinct "approximate / reordered-RNG" parity bar**, never the self-play default. The best
architecture spends all batch width on the cross-tree axis and keeps every tree's recursion
strictly serial.

---

## 5. Parity-testing implications

Prove a tree under the pipeline matches its serial self in **four composing layers**, reusing
the standing harnesses (the logic-vs-aggregate split the NMCS/ISMCTS ports already use):

1. **Net-forward parity (unchanged).** `cpp/parity/net_parity.py`: `max|Δ| < 1e-4` on
   value + logits, residual on/off. The ZMQ path is already measured at ~e-7 (N=1200,
   B∈{1..64}).

2. **Single-tree structural determinism.** Run **one** tree through the multiplexer against a
   **fixed-B deterministic-drain** server (the §2.2 escape hatch: fixed batch + barrier) with a
   **recording stub `Net`** so leaf returns are byte-identical on both sides. Assert the tree's
   sequence of `(loc, bw, collected, lam)` leaf **requests** and its executed action / improved-π
   are **identical** to the in-process serial `gumbel_search`. This isolates the **search
   structure** ("which leaf is requested next") from leaf numerics and proves the
   serial-per-tree guarantee directly.

3. **The three Danihelka invariants, per-episode, unchanged.**
   `test_executed_action_is_sh_survivor`, `test_vmix_prior_weighted`,
   `test_sequential_halving_spends_full_budget`. Each tree runs its own unmodified SH with its
   own private budget, so these must still pass; a failure means a tree's budget got coupled —
   you slid into Axis C.

4. **Aggregate behavioral equivalence.** `parity.py` / `bench_equivalence.py` over **N ≥ 300**
   episodes within Monte-Carlo CI with reported standard error, plus the **bit-exact** legal-mask
   and illegal-π-mass logic invariants. Add a **batch-composition stress test**: vary the number
   of in-flight trees and inject arrival jitter, and assert the aggregate stays inside CI — this
   directly pins that batch-composition roundoff stays inside the P6 envelope.

**The near-tie argmax hazard is named, not hidden:** layer 2 must use byte-identical injected
leaf returns precisely because, with **real** batched leaf numerics, a ≤1e-4 perturbation can
flip a near-tie and produce a *legitimately* different trajectory (§1.3, §2.2). That flip is a
distributional fact validated by layer 4, **not** a per-decision-identity claim. For a **C++**
tree, layers 2–4 are **distributional** (numpy PCG64 ≠ `std::mt19937_64`), so the bar is
aggregate, not draw-for-draw — unless the port mirrors numpy's bitgen.

**Two parity obligations the pipeline inherits but does not itself satisfy** (both at the
per-tree consumer, both orthogonal to batching — see §1.3): the **in-search float32
masked-softmax** and the **float32-prior/float64-Q weak-promotion seam**. A port using uniform
float32 or float64 will diverge possibly **beyond** 1e-4 at the improved-π target. These are the
first things a "looks exact" implementation gets wrong; pin each with a near-tied-logit kernel
test on `value_target` before integration.

---

## 6. Non-goals, deferred work, and open questions

**Non-goals (out of scope, would perturb the distribution):** virtual loss; a shared tree; root
parallelization; any within-tree leaf batching beyond the narrow exact defer-the-forward variant
of §4; moving masking server-side; adding XLA-bearing threads to the server.

**Deferred:** the C++ Gumbel-AZ search itself (only NMCS is ported today); the C++ `ZmqNetClient`
(ADR-0012 P9 `cpp/` pass); the DEALER submit/poll multiplexer (build Embodiment 1 first; build
the multiplexer only if measurement shows thread count, not the server, is the bottleneck); the
SH-phase sub-batch (eval-only, opt-in, distinct bar — if ever).

**Open questions for the maintainer:**

1. **Net version per tree-decision** (§3.7): pin one frozen version for a tree's whole search,
   or accept the off-policy straddle the current `_ensure_net` contract already allows?
2. **Client embodiment first cut** (§3.4): blocking-REQ-thread-per-tree (zero new code,
   OS-thread-bound `B`) vs. the DEALER multiplexer (fewer OS threads, genuinely new async
   client)? The exactness verdict is identical; recommendation is Embodiment 1 first, measured.
3. **Backprop-pool single-writer-per-tree** (§3.2): confirm the work-stealing pool guarantees
   one writer per tree's `_Node` dict at a time (the parked-tree-until-applied discipline). The
   Phase-1 map does not *prove* a multi-thread backprop pool is race-free per-tree; this needs an
   implementation guarantee + a targeted concurrency test, else `W/N` corruption is a correctness
   bug, not accepted slack.
4. **C++ RNG fidelity** (§1.3, §5): mirror numpy's PCG64 bitgen + exact draw order for
   draw-for-draw parity, or seed per-tree and accept distributional equivalence (the `parity.py`
   posture)? This sets whether the C++ parity bar is per-decision or aggregate.
5. **Earns-its-keep gate** (ADR-0009): on the 4-vCPU host with a tiny-MLP cheap forward, does the
   DEALER+fiber multiplexer actually beat M dumb blocking-REQ workers? Benchmark before committing
   to the multiplexer's complexity.

---

*Public Domain (The Unlicense).*
