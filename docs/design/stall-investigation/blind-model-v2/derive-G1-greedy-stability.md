# G-1 — Does the production greedy drain's negative-feedback batch-size fixed point survive ALL N?

**Role: focused derivation of one reconciliation gap (G-1).** This document answers, purely from the
cleanroom code and named causal necessities, the open question flagged by `out/RECONCILE.md §6 G-1`:

> Does the production greedy drain's negative-feedback batch-size fixed point remain stable for ALL N,
> or can any reachable config (e.g. a `K > max_batch` overshoot triggering a fresh XLA compile) inject a
> positive-feedback σ spike that grows the next batch faster than it drains?

The prior model and the fresh server models ASSERT "self-reinforcing stability" (greedy E1/E2, transport
Exec C) without deriving a stability CONDITION. The objective here is to DERIVE the condition and show it
survives N, not assume it.

All citations are to the cleanroom (`/home/bork/w/vdc/chocobo/runs/leaf-eval-model-2/cleanroom`),
read end to end for this derivation: `chocofarm/az/inference_server.py` (1–239), `chocofarm/az/forward.py`
(1–19), `chocofarm/az/inference_wire.py` (1–107), `cpp/src/runner_wire_batched.cpp` (1–506),
`cpp/include/chocofarm/wire_leaf_pool.hpp` (1–173), `cpp/include/chocofarm/runtime_config.hpp` (1–26),
`cpp/include/chocofarm/runner_wire_batched.hpp` (1–37), `cpp/stage_a/stage_a_server.py` (1–131),
`cpp/include/chocofarm/fiber_tree.hpp` (1–66), `cpp/include/chocofarm/fiber_leaf.hpp` (1–36),
`cpp/include/chocofarm/wire_spec.hpp` (1–25), `cpp/include/chocofarm/net_evaluator.hpp` (1–27),
`cpp/include/chocofarm/inference_wire.hpp` (1–164), `chocofarm/config.py` (1–44),
`cpp/include/chocofarm/error.hpp` (1–17).

---

## 0. Answer (headline)

**The production greedy drain's batch-size dynamics are UNCONDITIONALLY non-divergent for every N, T,
`max_batch`, `pool_batch`, and arrival timing the code permits.** No reachable config — including the
`K > max_batch` overshoot that triggers a fresh, unwarmed XLA compile — can inject a positive feedback that
grows the next batch *without bound*. The reason is structural, not a property of the (nondeterministic)
service-time function S:

1. **A hard, arrival-independent ceiling caps every batch:**
   `B_i ≤ max_batch + (m_i − 1) ≤ max_batch + K − 1`, with `K = N·base` (`runtime_config.hpp:12-15`,
   `runner_wire_batched.cpp:286`), because the drain's row cap is tested at the loop TOP and only one
   message is admitted past the cap (`inference_server.py:171,184-185`). This ceiling is reached in ONE
   cycle and is **absorbing**: no positive feedback can push a batch above it, no matter how large σ grows.

2. **The compile spike is a one-time, per-shape, NON-recurring transient** (`jit_forward_core:24-34`),
   so even the worst σ it can produce is paid at most once per distinct overshoot width and cannot
   compound across cycles.

3. **The standing offered-load is itself bounded by the depth-1 backpressure** (CF-1, reconcile §0):
   at most `T` messages are ever on the wire (one per DEALER), each ≤ K rows, so the queue the drain can
   ever pull from is `≤ T·K = T·N·base` rows — a fixed, finite quantity independent of how slow the server
   runs. A slow forward cannot manufacture *more* offered work than the producers structurally hold.

So the dangerous loop the question posits — "σ spike grows the next batch faster than it drains" — **cannot
occur**, because the next batch is bounded BELOW the cap-plus-one-message ceiling regardless of σ, and the
total offered work is bounded by `T·K` regardless of σ. The feedback from σ to batch size **saturates**
(monotone-bounded), which is the formal signature of negative feedback / a stable attractor, not of
divergence. The prior's informal "self-correcting" conclusion is therefore CORRECT and now has a derived
stability condition: **it holds unconditionally** (the condition is vacuous — there is no reachable
parameter regime that breaks it). Confidence: **high** for non-divergence/boundedness (it follows from the
two structural caps); **medium** for the finer monotone-contraction claim about the *interior* fixed point,
because S is left as bounded nondeterminism by the code (`forward_core` is shape-pure but the wall-clock
duration is not fixed) and the interior fixed point's exact location depends on the unmodeled S — but its
*existence within `[1, max_batch+K−1]`* and the *absence of divergence* do not.

---

## 1. The objects of the recurrence (all derived from code)

The server is a single thread running `serve_forever` (`inference_server.py:219-225`):

```
while not _stop:
    drained = self._drain()          # 223 — blocking poll, then non-blocking pull
    if not _stop and drained:
        self._serve_batch(drained)   # 225 — ONE forward, scatter
```

There is exactly one drain and (at most) one forward per loop iteration, executed **serially on one
thread**. No `recv` occurs during `_serve_batch` (`_drain` is the only receiver, 173). This is the
self-clocking premise (CF-11): everything that arrives during a forward queues in the ROUTER's incoming
buffer and is pulled by the *next* `_drain`.

Define a **cycle** `i` as one `(_drain → _serve_batch)` iteration that performed a forward. Let:

- `B_i` ∈ ℕ≥1 — rows fed to the forward in cycle `i` (`run_microbatch`'s `B`, `inference_server.py:56`).
- `shape_i` — the XLA input shape actually compiled/executed in cycle `i` (§3).
- `S_i = S(shape_i) > 0` — the **service time** (wall-clock) of cycle `i`'s forward, a positive,
  nondeterministic duration the code does NOT fix (forward.py is pure in *values* but says nothing about
  *wall time*; XLA execution time is left free, constrained only as §3 derives). `S_i > 0` by causal
  necessity (a reply cannot precede the forward that produced it).
- `A_i` ≥ 0 — rows that have ARRIVED into the ROUTER queue and are still unpulled at the instant
  `_drain` of cycle `i+1` begins its pull loop (i.e. accumulated during `S_i` plus any carried-over
  backlog the previous drain left because it hit the cap).
- `K = N·base`, `base = ceil(max(1,pool_batch)/max(1,T))` (`runtime_config.hpp:12-15`,
  `runner_wire_batched.cpp:285-286`). `K` is the **maximum rows a single producer message can carry**
  (gather-all `issue_one`, `runner_wire_batched.cpp:437-444`: it inserts every `is_ready` slot's features
  into ONE message; there are at most `K` slots per thread).
- `T` = `pool_threads` (`runner_wire_batched.cpp:283`); the number of DEALER sockets, hence the maximum
  number of distinct messages concurrently on the wire (one per thread, depth-1: CF-1, §2).
- `max_batch` — the server's row cap (`inference_server.py:149`, default 256; bench default 512 at
  `stage_a_server.py:89`).

---

## 2. RELY on the peer (checkable against the C++ producer)

The server's recurrence depends on what the producers can offer. The two RELY facts that bound the offered
load, each checkable against `runner_wire_batched.cpp`:

- **RELY-A (per-thread depth ≤ 1).** Each producer thread holds at most ONE message outstanding on its
  DEALER at a time. *Checkable:* `run_episodes_wire_pipelined` issues via `issue_one` (434-452), which
  gathers EVERY `is_ready` slot (`is_ready`, 427-430: `active && running && !submitted`) into one message
  and sets `submitted[s]=1` for all of them (447). A slot becomes ready again only inside the
  post-`recv_batch` completion loop (462-472). Between two `issue_one` calls with no intervening recv, no
  slot becomes newly ready, so the second `issue_one` finds `gathered.empty()` → returns false (444). The
  prime loop (456) and every refill (474) therefore issue exactly one message, and `inflight_msgs ∈ {0,1}`.
  This is `K`-independent, hence N-independent (reconcile §0, Z3-confirmed `reconcile_depth_parametric.py`).
  *Consequence for the server:* at most `T` messages are ever in flight ⇒ the ROUTER queue the server can
  pull from holds at most `T` messages.

- **RELY-B (per-message rows ≤ K).** Each producer message carries at most `K = N·base` feature rows.
  *Checkable:* `issue_one` gathers from `K` slots (the loop `for (int s=0; s<K; ++s)`, 437) and
  `submit_batch` encodes `B = slots.size()` rows (`wire_leaf_pool.hpp:79`, `inference_wire.hpp:51-70`).
  `K = N·rc.fibers_per_thread()` (286). So `1 ≤ rows_per_message ≤ K`.

These two give the **total offered-load ceiling**:

> **(OL)**  At any instant the rows queued in the ROUTER (in-flight + buffered) is `≤ T·K = T·N·base`.

(OL) is the linchpin: it is **independent of S**. No matter how slow a forward runs, the producers cannot
manufacture more than `T·K` outstanding rows, because each of the `T` threads blocks in `recv_batch`
(`runner_wire_batched.cpp:458`, the sole bounded blocking point CF-5) until its single outstanding message
is answered. A thread that is waiting for its reply emits NO new rows. This is the structural backpressure.

The server's **GUARANTEE** to the peer (the half it must uphold): it answers every well-formed request's
correlation id exactly once with a B-exact reply (scatter loop `inference_server.py:197-200`, paired with
`run_microbatch`'s per-identity split `:66-72`), or drops it silently on a vanished/full peer
(ROUTER_MANDATORY unset, CF-4) — never blocks the producer beyond `timeout_ms` (CF-5). Liveness of the
producer therefore depends on the server eventually replying; that is the assume-guarantee contract, but it
is orthogonal to the *batch-size* dynamics derived here.

---

## 3. The shape/σ function — exactly two branches, derived from code

`_serve_batch` (production, `inference_server.py:192-200`) calls
`run_microbatch(..., pad_to=self._max_batch)` (198). Inside `run_microbatch`:

- `B = Xb.shape[0]` is the concatenated real-row count (`:55-56`).
- Padding happens **only** when `pad_to > B` (`:58`): `Xb = concat([Xb, zeros(pad_to−B, in_dim)])` (`:59`).

So the forward's input shape is:

| regime | condition | `shape_i` (rows) | warmed? |
|---|---|---|---|
| **sub-cap** | `B_i ≤ max_batch` | `max_batch` (padded up) | YES if `max_batch` was warmed (`warmup` `:202-217`, bench warms `{64,256,512}∪{max_batch}` `stage_a_server.py:82`) |
| **overshoot** | `B_i > max_batch` | `B_i` (NOT padded; guard `:58` false) | only if that exact width was previously seen (one cached `jax.jit`, `:22-34`) |

**σ as a function of shape (the two bands per shape, derived):**

- `jit_forward_core` (`:22-34`) holds ONE cached `jax.jit` (`_jit_forward_cache`, `:20,33`). XLA compiles
  the executable **once per distinct input shape**; the first call at a new shape pays a one-time
  *compile+exec* duration (can be seconds), every subsequent call at that shape pays *exec-only*. So
  `S(shape)` is two-valued per shape: `S_compile(shape)` once, then `S_exec(shape)` thereafter.
- `forward_core` (`forward.py:3-18`) is a fixed sequence of dense matmuls + ReLU; it is **pure in the row
  count** (row `r` of the output depends only on row `r` of the input). Wall-clock `S_exec` is left free by
  the code but is causally **monotone non-decreasing in the row count** (more rows = at least as much
  arithmetic; this is a named causal necessity for dense matmul, not an outside performance assumption) and
  bounded for any bounded shape.

Two facts follow that are decisive for stability:

- **(SHAPE-FLAT) In the sub-cap regime the shape is CONSTANT at `max_batch`** regardless of `B_i ∈
  [1, max_batch]`. So `S_i` does not depend on `B_i` while `B_i ≤ max_batch` — the only variation is the
  one-time first-call compile of the `max_batch` shape. This is the prior's σ-shape-invariance (CF-6),
  and it means **within the entire sub-cap band there is NO σ-feedback on batch size at all.**
- **(SPIKE-ONCE) An overshoot width `w = B_i > max_batch` pays its compile spike at most ONCE** (cached
  forever after, `:20`), then is exec-only. The number of *distinct* overshoot widths is finite: `w ∈
  (max_batch, max_batch+K−1]`, i.e. at most `K−1` distinct values. So the total compile tax over the whole
  run is bounded by `(K−1)·S_compile_max` — a **one-time, non-recurring** cost, not a per-cycle one.

---

## 4. The recurrence and its absorbing ceiling

### 4.1 The drain's transfer function

`_drain` (`:160-186`): block on `poll` until ≥1 message (165), then pull non-blocking
(`recv_multipart(NOBLOCK)`, 173) in a loop guarded `while total_rows < max_batch` (171), adding
`X.shape[0]` per message (185), stopping on `Again` (174, queue empty) or when the **next** loop test sees
`total_rows ≥ max_batch`. Because the cap is tested at the loop TOP and a message is appended whole
(184-185) before the next test, the drain admits **one message past the cap**. Hence, with `A_i` rows
available and largest available message `m_i ≤ K`:

> **(DRAIN)**  `B_{i+1} = min( A_i ,  C_i )`, where `C_i ∈ [max_batch, max_batch + m_i − 1] ⊆
> [max_batch, max_batch + K − 1]` is the cap-with-overshoot reached on the available messages.
> Equivalently: `B_{i+1} ≤ min( A_i, max_batch + K − 1 )`, and `B_{i+1} ≤ A_i` ALWAYS (you cannot pull
> rows that have not arrived).

The two clamps in (DRAIN) are the whole argument:

- **Clamp 1 (cap):** `B_{i+1} ≤ max_batch + K − 1`. **Absorbing upper bound**, reached in one cycle, never
  exceeded, *independent of S and of A_i*. (`:171,184-185`, RELY-B.)
- **Clamp 2 (arrivals):** `B_{i+1} ≤ A_i ≤ T·K` (OL). You cannot drain more than was offered, and the
  offered total is itself capped by depth-1 backpressure — *independent of S*.

### 4.2 Why no positive feedback can diverge

The posited dangerous loop is: `B_i ↑ ⇒ S_i ↑ ⇒ A_i ↑ ⇒ B_{i+1} ↑ ⇒ …` without bound. Examine each link:

1. `B_i ↑ ⇒ S_i ↑`: TRUE only across the cap boundary (sub-cap shape is flat, SHAPE-FLAT). At worst a
   single overshoot cycle has a larger `S_i` (bigger exec + possible one-time compile).
2. `S_i ↑ ⇒ A_i ↑`: A longer service window lets MORE rows accumulate — but `A_i ≤ T·K` (OL) **regardless
   of how large `S_i` is.** The arrival integral saturates at the total work the `T` depth-1 producers can
   hold. A slower forward does NOT create new producers or lift the per-thread depth-1 cap; it only lets the
   *already-bounded* outstanding work finish arriving. So `A_i` is bounded by a constant independent of `S_i`.
3. `A_i ↑ ⇒ B_{i+1} ↑`: TRUE, but `B_{i+1} ≤ max_batch + K − 1` (Clamp 1) **regardless of `A_i`.**

Compose: `B_{i+1} ≤ min(T·K, max_batch + K − 1)` for ALL i, ALL S, ALL N. The right-hand side is a finite
constant in each fixed config. **The sequence `{B_i}` is bounded by an absorbing ceiling reached in one
step; it cannot diverge.** The σ→batch map is monotone and bounded above, i.e. it SATURATES — the formal
signature of negative feedback toward a fixed point, never of runaway positive feedback. ∎

### 4.3 The overshoot compile spike, specifically (the question's named worry)

The question singles out: a `K > max_batch` overshoot triggering a fresh XLA compile. Trace it precisely:

- **Reachability (sharpened by the bounded check):** overshoot (`B_i > max_batch`) requires a single drain
  to pull past the cap with the last message. Since one message is ≤ K rows and the cap admits one message
  past `max_batch`, two things must BOTH hold: (i) the queue can present strictly more than `max_batch`
  rows in aggregate, i.e. `A_i > max_batch`, which by (OL) needs `T·K > max_batch`; and (ii) the crossing
  message contributes `> 1` over-cap row, i.e. `K > 1`. The over-cap magnitude is then bounded by the
  single crossing message: `B_i ≤ max_batch + (m_i − 1) ≤ max_batch + K − 1`. So the **reachability
  condition is `T·K > max_batch` (with `K>1`)**, NOT merely `K > max_batch`: at `N=1`, `K=base` (e.g. 8),
  `T·K = 32 ≪ max_batch` (256/512) ⇒ overshoot UNREACHABLE (prior R5; Z3 `n/a`). As N grows, `T·K = T·N·base`
  crosses `max_batch` first — overshoot becomes reachable at `N > max_batch/(T·base)` — and a single
  message can carry the whole over-cap surplus once additionally `K = N·base > max_batch`
  (`N > max_batch/base`). Either way the over-cap WIDTH never exceeds `max_batch + K − 1`. This is the ONE
  place N enlarges the forward SHAPE. (Bounded check `g1_greedy_stability_check.py`: the overshoot witness is
  SAT exactly for the `T·K > max_batch` configs — N∈{33,75,200} — and `n/a` for N∈{1,8} where `T·K ≤
  max_batch`.)

- **σ effect:** the overshoot cycle's forward runs at an **unwarmed** width `w = B_i ∈ (max_batch,
  max_batch+K−1]` ⇒ pays `S_compile(w)` the first time `w` occurs (`:58` guard false ⇒ no pad ⇒ new shape
  ⇒ `jax.jit` traces+XLA compiles, `:22-34`). This is a genuine **σ spike** on that one cycle.

- **Does it grow the next batch faster than it drains?** NO, by Clamp 1: the very batch that paid the spike
  was *already at* the ceiling `B_i ≤ max_batch + K − 1`; the next batch is bounded by the SAME ceiling.
  The spike makes `S_i` larger, which by link (2) lets `A_i` rise — but only up to `T·K` (OL), and the
  next batch is still `≤ max_batch + K − 1` (Clamp 1). The spike cannot lift the ceiling it is already
  under. Moreover (SPIKE-ONCE) the SAME width never pays the compile again, so even the transient does not
  recur: subsequent overshoots at the same `w` are exec-only. The worst case is a finite, one-time
  per-width tax of total magnitude `≤ (K−1)·S_compile_max`, fully amortized.

  Two sub-mechanisms could in principle *sustain* a spike, and both are foreclosed:
  - *New widths each cycle.* If every cycle overshot to a *different* `w`, each would compile afresh. But
    there are only `K−1` distinct overshoot widths (`:171` cap + RELY-B), so after at most `K−1` distinct
    overshoots the entire over-cap band is warmed; thereafter overshoot is exec-only. Bounded, non-recurring.
  - *Spike → bigger batch → bigger spike.* Foreclosed by Clamp 1: a bigger `S_i` cannot produce
    `B_{i+1} > max_batch + K − 1`. The map `S_i ↦ B_{i+1}` is clamped, so its "gain" above the ceiling is
    exactly zero. There is no width above `max_batch+K−1` to escalate INTO.

### 4.4 The interior fixed point (the "self-correcting" regime), parametric in N

Within the absorbing interval `B ∈ [1, max_batch + K − 1]`, the closed loop has the structure the prior
called self-correcting, and it survives N:

- **Sub-cap band `[1, max_batch]`:** shape is flat at `max_batch` (SHAPE-FLAT) ⇒ `S` is constant in `B`
  (one band). The throughput `B/S` is INCREASING in `B` (fixed cost amortized over more rows). So if a
  cycle is small (`B_i` low), the next window's arrivals — bounded by OL but *driven up* by the fact that a
  busy window lets the depth-1 producers' replies clear and re-issue fatter messages — tend to be larger,
  filling toward `max_batch`. If a cycle is large, the wasted-padding fraction falls and throughput is
  maximal; arrivals during the (same-shape) window cannot exceed `max_batch` rows queued unless the offered
  load exceeds `max_batch/S`, in which case the system pins at the cap. **This is a stable attractor:**
  below the cap the batch grows toward it (more useful work per fixed-cost forward), at/above the cap it
  pins. The "feedback" is the desirable batch-FILL coalescing (a saturating positive feedback that stops at
  the cap), not a batch-GROWTH divergence (impossible by Clamp 1). The fixed point is `B* =
  min(offered-load-per-window, max_batch)`, and offered-load-per-window `≤ T·K` (OL).

- **N-dependence of the fixed point:** N raises `K = N·base`, which (a) raises the per-window arrival
  ceiling `T·K`, moving `B*` UP toward `max_batch` (so larger N reaches the efficient cap-pinned regime at
  lower wall-time — the prior's "feedback gain rises with N", CF-11), and (b) widens the over-cap band to
  `[max_batch+1, max_batch+K−1]`, so the absorbing ceiling itself rises *linearly* in N. Both are
  monotone, bounded effects: N moves the fixed point and the ceiling, it does NOT change the SIGN of the
  feedback (still saturating) nor introduce divergence (Clamp 1 holds at every N). **The self-correcting
  regime survives N; it does not flip.**

---

## 5. The stability CONDITION (what the prior asserted but did not derive)

Stated as a theorem with its proof obligations discharged:

> **Theorem (greedy-drain batch boundedness, parametric).** For the production greedy drain
> (`inference_server.serve_forever`/`_drain`/`_serve_batch`/`run_microbatch`) driven by the conforming
> pipelined C++ peer (`run_episodes_wire_pipelined`), for ALL `N ≥ 1`, `T ≥ 1`, `pool_batch ≥ 1`,
> `max_batch ≥ 1`, and ALL admissible positive service-time sequences `{S_i}` and admissible arrival
> timings, the offered batch sequence satisfies, for every cycle i,
> `1 ≤ B_i ≤ min( T·K , max_batch + K − 1 )`,  with `K = N·base`,
> and the total XLA compile cost over the run is `≤ (1 + #distinct overshoot widths)·S_compile_max ≤
> (1 + (K−1))·S_compile_max`. Hence `{B_i}` is bounded by an absorbing ceiling reached within one cycle,
> and admits NO divergent (positive-feedback) trajectory at any N.

**Proof obligations and where each is discharged:**

| obligation | discharged by | code_ref |
|---|---|---|
| `B_{i+1} ≤ A_i` (drain ≤ arrivals) | non-blocking pull stops on `Again` (empty queue) | `inference_server.py:172-174` |
| `B_{i+1} ≤ max_batch + K − 1` (cap clamp, absorbing) | loop-top cap test + one message past + RELY-B | `:171,184-185`; `runner_wire_batched.cpp:286,437-444` |
| `A_i ≤ T·K` (offered-load ceiling, S-independent) | RELY-A depth-1 + RELY-B rows≤K | `runner_wire_batched.cpp:447,456,474,458` (depth-1); `437` (gather-all) |
| sub-cap σ flat in B (no interior σ-feedback) | pad-to-`max_batch` ⇒ constant shape | `inference_server.py:198,58-59` |
| overshoot σ spike is one-time per width | single cached `jax.jit`; ≤ K−1 distinct widths | `:20,22-34`; `:171` (width range) |
| `S_i > 0`, reply ⊀ forward | causal necessity (yielding evaluator round-trip) | `fiber_leaf.hpp:24-29`; `fiber_tree.hpp:58-62` |

The CONDITION under which the fixed point could be destabilized is therefore the EMPTY SET: there is no
admissible `(N, T, max_batch, pool_batch, {S_i})` that produces divergence. The prior's "self-correcting"
claim holds unconditionally; the only thing N changes is the LOCATION of the fixed point (up toward the cap)
and the HEIGHT of the absorbing ceiling (`max_batch + N·base − 1`), both monotone and bounded.

### 5.1 The one honest caveat (where the code leaves latitude)

The code does NOT fix the service-time function `S(shape)` (it is bounded nondeterminism — `forward_core`
is value-pure but wall time is free, §3). Therefore the EXACT interior fixed point `B*` and whether the
approach to it is monotone vs. lightly oscillatory **cannot be pinned without choosing an S model**, and I
do not. What is INDEPENDENT of any S choice — and is all the question asks — is: (i) boundedness by the
absorbing ceiling, (ii) the one-time non-recurring nature of the compile spike, (iii) the S-independence of
the offered-load ceiling `T·K`, and therefore (iv) the impossibility of a divergent positive-feedback
trajectory at any N. These four are what the prior left underived; they are derived above and depend only
on the two structural clamps and depth-1 backpressure, none of which involve S.

A second caveat, orthogonal to batch *size*: a large-enough single overshoot compile spike `S_compile(w)`
could push the round-trip past the producer's `RCVTIMEO = timeout_ms` (`wire_leaf_pool.hpp:41`), firing the
producer's recv timeout (CF-5) — that is a LIVENESS/latency effect on the peer, not a batch-size
divergence, and it is itself self-limiting (a timed-out producer stops offering, REDUCING `A_i`, which
*shrinks* the next batch — a further negative feedback, not a positive one). It does not change the answer.

---

## 6. Bottom line for the synthesizer (closing G-1)

**G-1 is closed: YES, the production greedy drain's negative-feedback batch-size fixed point survives ALL
N.** No reachable config injects a positive-feedback σ spike that grows the next batch faster than it
drains, because (a) every batch is hard-clamped to the absorbing ceiling `max_batch + N·base − 1` reached in
one cycle (`inference_server.py:171,184-185`; RELY-B), (b) the total offered work is clamped to `T·N·base`
*independently of σ* by depth-1 producer backpressure (RELY-A; `runner_wire_batched.cpp:447,456,458,474`),
and (c) the overshoot compile spike is a one-time, ≤(K−1)-distinct-width, non-recurring transient
(`inference_server.py:20,22-34,58`). N moves the fixed point up toward the cap and lifts the ceiling
linearly, but never flips the feedback sign. The prior's central differential claim is upheld with a derived
condition (which is vacuously satisfiable-by-all, i.e. unconditional). Confidence **high** on
non-divergence/boundedness; **medium** on the interior fixed point's exact behavior (S left free by code).

A bounded Z3 confirmation of the absorbing-ceiling recurrence (that a σ-spike cycle cannot push the next
batch above `max_batch + K − 1`, at several N) is in `out/g1_greedy_stability_check.py`; it is confirmation
of the representative execution, not the source of trust — the trust is in §4–§5.

*Public Domain (The Unlicense).*
