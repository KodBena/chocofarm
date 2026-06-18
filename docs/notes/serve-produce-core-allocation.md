<!--
docs/notes/serve-produce-core-allocation.md

Purpose: An analytical (operations-research) note modeling the optimal partition of a
fixed pool of P CPU cores between SERVING (the batched JAX neural-net leaf evaluator)
and PRODUCING (the C++ MCTS search trees that emit leaf-eval requests) for the
Gumbel-AlphaZero self-play actor. Two lenses — a discrete integer-allocation/min-flow
solve, and a continuous bulk-service-queue / fluid-traffic-flow equilibrium — both
pointed at the same operating-point question. This is a MODELING exercise, not an
implementation; it changes no system code.

Public Domain (The Unlicense).
-->

# Serve/produce core allocation — an OR + queueing note

**Status:** Modeling note, branch `cpp-actor-online-reconfig`. The deliverable is the
written analysis; the only code is a ~30-line scratch solve (reproduced inline) used to
enumerate the integer splits and locate the equilibrium batch. No system code is
touched. All measured inputs are tagged **MEASURED**; all derived conclusions are tagged
**MODELED**.

The question the maintainer is asking: given **P cores** on the self-play host (P=4
vCPUs concretely), partition them into **S server cores** (each running ONE
single-threaded JAX inference-server process that batch-evaluates leaves) and **G
producer cores** (each running independent C++ MCTS trees that descend and emit
leaf-eval requests), with `S + G = P`. Pick `(S, G)` and the per-producer **overcommit
depth** (concurrent trees per core) to maximize aggregate steady-state throughput, in
leaves/s or equivalently decisions/s (dps). The two failure modes the optimum sits
between are **server starvation** (G too small → batches stay tiny → the server runs in
its slow per-call-dispatch-bound regime) and **producer backpressure** (G too large →
the queue grows without bound, trees block on values, the descent stalls).

---

## 1. Setup, the seam, and the measured primitives

The actor is the env/Policy seam run at scale: producers own the dynamics + belief +
descent (`Environment`), the server is the injected leaf evaluator (the batched
`NetEvaluator` over the wire). The two are decoupled by a **leaf request queue**: a
producer descending a tree reaches a leaf, posts `(features)` to the queue, and blocks
*that tree* until the matching `(value, policy)` returns. The server drains the queue in
microbatches, runs one forward per batch, and scatters the answers back.

The whole problem is the interaction of two measured curves across that queue.

### 1.1 Producer side (MEASURED)

One producer core, MCTS descent + a near-free local eval, on the production 241→256→65
MLP at `n_sims=256`:

- **152 decisions/s/core** (`DPS_PER_CORE`).
- **~500 leaves/decision** (`LEAVES_PER_DEC`; n_sims=256 with the tree's expansion
  factor) → **~76,000 leaves/s/core** (`GEN = 152 × 500`).
- Generation scales **4.0× LINEARLY** across the 4 cores — embarrassingly parallel,
  independent trees, no shared mutable state. So `G` producer cores emit

  ```
  prod_rate(G) = G × 76,000  leaves/s                                    (MODELED from MEASURED linear scaling)
  ```

### 1.2 Server side (MEASURED)

One single-threaded JAX server core, BLOCKING per forward (one forward at a time —
this is an **invariant**, not a tunable; cross-request parallelism comes ONLY from
adding server *processes*, never from threading a single forward). Leaves/s as a
function of the batch size B it gets to run:

| B    | 1   | 8   | 16  | 32  | 64  | 96  | 128 | 192 | 256 | 384 | 512 | 1024 |
|------|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|------|
| k/s  | 9.0 | 60  | 99  | 143 | 188 | 211 | 225 | 235 | 228 | 255 | 259 | 264  |

The shape is the whole story. The rate rises **steeply** from B=1 to B≈128 then goes
**~flat**: the forward is **per-call-dispatch-bound** ("framework tax" — Python/XLA
dispatch, pytree flattening, host↔device handoff, the C++↔Python wire), NOT FLOPs-bound.
Going B=64→512 buys ~40% more throughput on barely-changed arithmetic. Call this curve
`rate(B)`; I interpolate it linearly between the measured knots.

The forward is **single-shot** (leaf → value+policy, no recurrence), so batch members
are independent and a batch of B is exactly B amortized dispatches.

### 1.3 Assumptions, and their limits (be honest — this is first-order)

1. **`rate(B)` captures all server cost.** Per-leaf wire/serialize/scatter overhead
   beyond what's already baked into the MEASURED `rate(B)` is ignored. In reality the
   gather/scatter around the forward grows with B and erodes the flat region's top end;
   the model will slightly *over*-credit very large B.
2. **Perfect overcommit hides latency.** I assume a producer core with enough concurrent
   trees ALWAYS has a runnable tree to descend while others wait on values — i.e. the
   blocking-per-forward latency is fully amortized and producer cores never idle on the
   queue. This is the ELF Game / daemon-overcommit idealization. Section 4 quantifies
   the depth needed; below it the producer rate degrades from the 76k ceiling.
3. **Gen is load-independent.** `prod_rate(G) = 76k·G` regardless of queue pressure.
   Real descent slows under deep overcommit (cache pressure, scheduler overhead, the
   virtual-loss bookkeeping for more in-flight leaves). Treated as a constant ceiling.
4. **Steady state.** Transients (tree warm-up, batch fill latency) ignored; we model the
   fixed point.
5. **One shared queue per server.** All producers feeding a given server share one FIFO;
   the server drains greedily up to a max-batch cap. Multiple servers (S>1) partition
   the producer fleet's inflow evenly across S queues.

These limits all bias the same direction: the model is an **upper bound** on achievable
throughput. The recommended *split* is robust to them (it's an integer argmax with a
wide margin); the absolute dps ceiling is optimistic.

---

## 2. Discrete / OR lens — integer allocation + min-flow

### 2.1 Variables and objective

- `S, G ∈ ℤ≥0`, `S + G = P` (the partition).
- `c` — overcommit depth (concurrent trees per producer core); enters via Section 4,
  treated here as "large enough to fill B".
- `B` — the batch size the shared queue delivers to each server at equilibrium.

Throughput at the queue is a **min of two flows** (Lindley / bottleneck): the producers
push leaves in, the servers pull leaves out, and steady-state throughput is whichever
side is slower (the other side's slack shows up as queue growth or server idle):

```
Θ(S, G, B) = min( prod_rate(G),  S · rate(B) )            leaves/s         (MODELED)
           = min( G·76,000,       S · rate(B) )
```

with `S + G = P`. We want `B` as large as the producers can keep filled (more B only
helps up the rising part of `rate(B)`), then we pick the `(S,G)` split that maximizes the
min. Decisions/s is `Θ / 500`.

Degenerate corners: `(S=0)` — no evaluator, producers block forever, Θ=0;
`(S=P, G=0)` — no producers, nothing to evaluate, Θ=0. The interior splits are the only
candidates. For P=4: `(1,3)`, `(2,2)`, `(3,1)`.

### 2.2 The solve (P=4)

Scratch enumeration (run under the project interpreter; `rate(B)` is the MEASURED table
interpolated):

```python
import numpy as np
Bpts = np.array([1,8,16,32,64,96,128,192,256,384,512,1024])
Rpts = np.array([9,60,99,143,188,211,225,235,228,255,259,264])*1e3   # MEASURED leaves/s
rate = lambda B: np.interp(B, Bpts, Rpts)
GEN, LPD, P = 76_000., 500., 4
for S in range(1, P):
    G = P - S
    prod = G * GEN
    # best attainable server-limited rate, over feasible B:
    serv = max(S*rate(B) for B in Bpts)
    thru = min(prod, serv)
    print(f"(S={S},G={G})  prod={prod/1e3:6.1f}k  serv*={serv/1e3:6.1f}k  "
          f"Θ={thru/1e3:6.1f}k  dps={thru/LPD:6.0f}")
```

**MODELED output:**

| Split (S,G) | prod_rate | server ceiling (S·max rate) | Θ = min  | dps    | binding side |
|-------------|-----------|-----------------------------|----------|--------|--------------|
| **(1,3)**   | 228.0k    | 264.0k                      | **228.0k** | **456** | producers    |
| (2,2)       | 152.0k    | 528.0k                      | 152.0k   | 304    | producers    |
| (3,1)       | 76.0k     | 792.0k                      | 76.0k    | 152    | producers    |

Every interior split is **producer-bound** — the server side has slack in all three —
because even ONE server in its flat region (235–264k) out-produces three producer cores
(228k). So the objective reduces to **maximize G subject to S≥1 servers being able to
absorb `G·76k`**: spend the *minimum* cores on serving that keep the server off the
critical path, give the rest to production.

**The optimum is `(S=1, G=3)`** — one server, three producers. MODELED Θ ≈ **228k
leaves/s ≈ 456 dps**.

The check that S=1 suffices: the single server must sustain the 3-producer inflow of
228k leaves/s. From the MEASURED curve, `rate(B) ≥ 228k` first at **B≈192** (235k), and
holds for all B≥192. So one server in its flat region covers three producers with margin
to spare — S=1 is not just best, it's comfortable. (S=2 would leave the second server
~40% idle and steal a producer core: strictly worse.)

### 2.3 Integer effects and the co-located (fractional-S) option

The integer constraint is what makes this interesting. With P=4 the only knob is S∈{1,2,3}
and the curve's slack means the answer is "S as small as keeps the server uncritical" =
1. But the server is not *physically* a whole core — a single-threaded forward leaves the
core idle during the inter-batch gather/scatter and during producer-side stalls. That
opens **co-location**: run the server thread *on a producer core* (an in-process server
sharing core 0 with one tree fleet), making S effectively **fractional**. If the server
truly needs only, say, 0.7 of a core to sustain 228k, then co-locating reclaims ~0.3 core
of production — a few percent of dps. This is a real lever precisely because the (1,3)
split is producer-bound: any core-fraction clawed back from the (idle-ish) server
converts 1:1 into more leaves. The risk is scheduler interference (the forward and the
descent fighting for the same core's L2 / SMT siblings), which would erode the clean
linear gen scaling — an empirical question, flagged not resolved.

---

## 3. Continuous / queueing lens — bulk service + the fluid equilibrium

The discrete solve says *which split*; it doesn't explain *why the queue settles where it
does* or *how deep the overcommit must be*. For that, model the leaf queue as a
**bulk-service queue** and then take its fluid limit.

### 3.1 The bulk-service queue

Producers are a superposition of independent leaf streams → a roughly Poisson arrival
process of rate `λ = G·76,000` leaves/s into one queue (for the S=1 case; for S>1, λ/S
per server). The server is a **bulk server**: it waits/greedy-drains, grabs up to
`B_max` waiting leaves, runs ONE forward taking deterministic time `τ(B)`, returns all B.
This is an `M^[X]/D/1`-flavored batch-service queue — Poisson in, **deterministic
batch service** out, the batch size endogenous to the queue occupancy.

The key non-classical feature: the service *rate* is not fixed, it RISES with the batch
the server pulls, because `rate(B) = B/τ(B)` and `τ(B)` is dominated by the per-call
dispatch tax, nearly constant in B over the rising region. Bigger drains → bigger B →
higher effective service rate. The server is **faster when busier** — up to the knee.

### 3.2 Fluid / flow equilibrium

Let `q(t)` be queue length (leaves waiting). Inflow is constant `λ = 76k·G`. Outflow is
the server's drain rate, which depends on the batch it gets to assemble, which depends on
how much is waiting:

```
dq/dt = λ  −  μ(q)
```

where `μ(q)` is the effective service rate when the queue offers a batch of size
`B = min(q, B_max)`:

```
μ(q) = rate( min(q, B_max) )                                            (MODELED)
```

This is exactly a **traffic-flow fundamental-diagram** relation. In the highway analogy:

- **queue length q ↔ vehicle density** on a road segment,
- **service rate μ(q) ↔ traffic flow** (vehicles past a point per unit time),
- `rate(B)` rising then flattening ↔ the **flow-density curve** rising from free-flow,
  reaching **capacity at the knee**, then plateauing.

The equilibrium is `dq/dt = 0` ⇒ **`μ(q*) = λ`** ⇒ `rate(B*) = λ`. The system settles to
the queue occupancy whose batch makes the drain exactly match the inflow.

For the recommended `(1,3)` split, `λ = 228k`. Solving `rate(B*) = 228k` on the MEASURED
curve gives **`B* ≈ 192`** (the curve crosses 228k right around there; 192→235k, 128→225k,
so the crossing is ≈190). So the queue self-regulates to deliver batches of ~190 leaves —
squarely in the server's fast region, by construction, because that's the only batch at
which drain balances the 3-producer inflow.

### 3.3 Stability, the two branches, and the operating point

The fundamental-diagram structure gives the stability story directly. Plot `μ(q) = rate(min(q,B_max))`
against `λ` (a horizontal line):

- **Free-flow / rising branch (small q, small B).** Here `μ(q) < λ` would mean the queue
  *grows*, which *increases* B, which *increases* μ — a **self-correcting** push toward
  equilibrium. The equilibrium on the rising branch at `rate(B*)=λ` is **stable**: a
  perturbation that shortens the queue lowers μ below λ and refills it; one that lengthens
  it raises μ above λ and drains it. Good.
- **Capacity / flat branch (large q ≥ knee).** Once `q ≥ B_max` (or past the knee where
  `rate` flattens), μ stops rising. If `λ > rate(B_max)` — inflow exceeds the server's
  *peak* — there is **no equilibrium**: `dq/dt = λ − μ_max > 0` forever, the queue grows
  unboundedly, producers eventually all block on un-returned values → **backpressure
  collapse**. This is congested-branch traffic: density past capacity, flow can't keep up,
  jam propagates upstream (here: upstream = the producer trees stalling).

So the optimum lives **just below the capacity knee**: enough inflow (enough producer
cores G) to keep the equilibrium batch in the fast region (avoid the **starvation** mode
where the server idles on tiny batches — `λ` so small that `B*` lands at B=8/16 and the
server wastes its dispatch budget on near-empty forwards), but **not** so much inflow that
`λ` exceeds the single server's peak `rate(B_max) ≈ 264k` and tips onto the unstable
congested branch.

For P=4, `(1,3)` gives `λ=228k < 264k`: stable, with `B*≈192`, a ~14% capacity margin
(264/228). `(2,2)` would give each of two servers `λ=76k`, equilibrium `B*≈48` — *into the
slower part* of the curve and wasteful (two servers each loafing) — confirming from the
fluid side what the integer solve said: `(2,2)` strands server capacity at low batch.

### 3.4 The starvation ↔ backpressure window, concretely

The window of stable, efficient operation for **one** server is:

```
B_efficient ≲ λ ≲ rate(B_max)        i.e.   ~140k  ≲  λ  ≲  264k   leaves/s   (MODELED)
```

Lower bound: keep `B*` past ~B=32 (143k) so the server isn't dispatch-thrashing on tiny
batches. Upper bound: stay under peak so the queue is finite. In producer-core units
(76k/core), that's roughly **1.9 ≲ G ≲ 3.5 producers per server** — and **G=3** sits
almost dead-center, which is why the integer answer is so clean and so robust. One server
"wants" about three producers.

---

## 4. Overcommit depth — how deep to feed the queue

Everything above assumes producers can KEEP `B*≈192` worth of leaves in flight. Can they?

A single MCTS tree's concurrent in-flight leaves are **capped well below B*** by
virtual-loss collisions: once a tree has a handful of pending leaves, every further
descent collides on the same promising path (virtual loss steers it elsewhere, but only so
far), so a lone tree offers maybe a few-to-~10 in-flight leaves — far short of 192. This is
exactly the ELF observation: **you cannot fill a large batch from one tree; you fill it
from many trees.**

So the equilibrium batch must be assembled across the whole producer fleet's concurrent
trees. The overcommit requirement (MODELED):

```
total in-flight leaves needed  ≈  B*  +  (leaves consumed during one forward latency)
                               ≈  B*  +  λ·τ(B*)
```

With `B*≈192` and the forward latency `τ(B*) = B*/rate(B*) = 192/235k ≈ 0.82 ms`, the
in-flight-during-a-forward term is `λ·τ ≈ 228k · 0.00082 ≈ 187` — essentially another
batch. So the fleet must sustain **~B* on the order of ~2·B* ≈ 380 concurrent in-flight
leaves** to keep the pipeline full across the blocking forward (one batch executing, one
batch refilling — classic double-buffering).

Spread over G=3 producer cores, each core needs ~**130 concurrent in-flight leaves**. At
~5–10 in-flight per tree, that's **~15–30 concurrent trees per producer core** — the
overcommit factor `c`. This is the depth that makes Assumption 2 (perfect latency hiding)
true; below it, producers idle waiting on returns and `prod_rate` falls under 76k·G, the
equilibrium `B*` drops, and throughput slides down the curve. Above it, diminishing returns
plus the real-world gen slowdown (Assumption 3) of deep overcommit.

**MODELED recommendation: c ≈ 16–32 trees/core**, double-buffered, targeting ~2 batches of
in-flight leaves fleet-wide.

---

## 5. What this predicts for us

- **Recommended split: `S=1` server : `G=3` producers** (one batched JAX evaluator
  process, three MCTS producer cores). MODELED from the integer min-flow solve (§2.2) and
  independently confirmed by the fluid equilibrium landing `λ=228k` dead-center in the
  one-server stable window (§3.4).

- **Overcommit: ~16–32 concurrent trees per producer core** (≈380 in-flight leaves
  fleet-wide, double-buffered), to keep the self-regulated equilibrium batch at **B*≈192**
  in the server's fast region (§4). This is the load-bearing requirement — without it the
  server starves on small batches and the whole estimate degrades.

- **Predicted ceiling: Θ ≈ 228k leaves/s ≈ 456 dps** (MODELED, producer-bound). Against
  our current **~49–60 dps**, that's a **~7.5–9× headroom** the model says the (1,3) +
  deep-overcommit architecture leaves on the table — consistent with "current is
  serial/under-batched, not core-bound." Treat 456 as an optimistic upper bound (§1.3 all
  bias high); even at half, ~230 dps would be a 4–5× win.

- **Why not (2,2):** a second server is ~40% idle at this inflow and costs a producer core
  — strictly worse (304 dps MODELED). The serve curve's flat region is so much faster than
  one producer core (235k vs 76k) that one server amply covers three.

### Sensitivities (what moves the optimum)

| Change | Effect on optimum |
|--------|-------------------|
| **Faster server** (higher `rate(B_max)`, cheaper dispatch tax) | Pushes *further* toward more producers; the stable window's upper bound rises, so one server absorbs even more G. Reinforces (1,3); at P>4 would favor (1, P−1) longer. |
| **Slower server** / fatter net (FLOPs start to matter) | Shrinks the window; eventually one server can't cover 3 producers (`λ > rate(B_max)`) and the optimum tips to (2,2). The crossover is `rate(B_max) < 3·76k = 228k` — we're at 264k, ~14% margin, so a ~15% server slowdown flips it. |
| **More cores P** | Stays "minimum servers that keep S off the critical path." Each server covers ≈3 producers, so the rule of thumb is `S ≈ ⌈G/3⌉`: P=8 → (2,6) or (3,5)-ish; P=16 → ~(4,12). |
| **Cheaper dispatch tax** (the framework tax shrinks) | `rate(B)` flattens *earlier* (knee at smaller B) and the small-B penalty eases → less overcommit needed to reach the fast region, and B* can be smaller. Loosens the §4 requirement. |
| **Higher in-flight-per-tree** (weaker virtual loss) | Fewer trees needed per core to fill B* → smaller overcommit `c`. Doesn't move the split. |

### The one thing to measure next

The split (1,3) is robust; the **456 dps ceiling is not**. The single most informative
follow-up is the **co-location experiment** (§2.3): does the server actually need a whole
core, or can it share with a producer at 228k inflow without breaking the linear gen
scaling? If it co-locates cleanly, the effective split is `(0.7, 3.3)` and the ceiling
nudges up; if it interferes, (1,3) with a dedicated server core is the honest answer. Run
it before trusting any absolute dps number above.

---

*Model inputs (§1) are MEASURED on the 241→256→65 MLP at n_sims=256, single core; the
solve (§2.2), the equilibrium B*, the overcommit depth (§4), and the dps ceiling (§5) are
MODELED and inherit the §1.3 upper-bound bias.*
