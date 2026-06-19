# Closed-form expected value of `mean_rows_per_msg` (total_leaves / total_msgs)

Focused derivation for the leaf-eval transport boundary, pipelined-bucket runner.
All file:line citations are into the cleanroom tree
`/home/bork/w/vdc/chocobo/runs/leaf-eval-model-2/cleanroom`.

Public Domain (The Unlicense).

---

## 0. The question, restated mechanically

The driver emits, once at the end of `run_episodes_wire_pipelined`
(`cpp/src/runner_wire_batched.cpp:494-501`):

```
mean_rows_per_msg = total_leaves / total_msgs        (line 496)
```

where `total_leaves` / `total_msgs` are the cross-thread sums of the per-thread
counters `my_leaves` / `my_msgs` (declared 329, summed 476-477). The per-thread
counters are mutated in exactly one place — `issue_one()`:

```
my_leaves += gathered.size();   // 449
++my_msgs;                       // 450
```

and `issue_one()` is the ONLY emitter of a wire request in the pipelined runner
(it calls `pool.submit_batch`, 445). So:

* `total_leaves` = total number of leaf-rows ever placed into any request.
* `total_msgs`   = total number of non-empty requests ever submitted.

We want a closed form for their ratio as a function of
`N = trees_per_thread`, `T = pool_threads`, `base = fibers_per_thread`
(so `K = N·base`, line 286), `D = max_inflight_msgs` (line 287), and the
park/reply timing — under the question's premise **depth == 1** (each
`run_search` makes exactly one `predict()` call before returning a decision).

---

## 1. Conserved quantity: the numerator is timing/D/N-INVARIANT

### 1.1 One park = one row, gathered exactly once

A slot contributes a row to a message iff `is_ready(s)` holds at an `issue_one`
gather (437-444):

```
is_ready(s) := sl.active && sl.ts && sl.ts->running && !submitted[s]   (427-430)
```

`ts->running` is set true exactly when the slot's fiber is parked at a leaf:
`TreeState::start` and `resume_with` both set `running = ch.at_leaf`
(`fiber_tree.hpp:55, 61`), and `ch.at_leaf` is set true precisely inside
`YieldingNetEvaluator::predict` just before it yields to the caller
(`fiber_leaf.hpp:26-27`). So a slot is "ready" iff its search is suspended
inside a `predict()` call awaiting a network reply.

When a ready slot is gathered, `issue_one` sets `submitted[s] = 1` (447), which
makes `is_ready` false until the matching reply clears it: `submitted[s] = 0`
at 466, reached only after `recv_batch` returns that slot's completion. Hence
**each individual park (each `predict()` yield) is gathered into exactly one
message exactly once** — never zero (the slot stays ready and `any_parked`-style
progress requires draining it), never twice (`submitted` gate).

Therefore:

```
total_leaves = (total number of predict() yields executed over the whole run)
```

### 1.2 depth == 1 ⇒ numerator = total plies

Under depth == 1, one `run_search` invocation = one `predict()` yield. A new
search is spawned once per ply: `spawn_ply` (331-335) / the first ply via
`start` inside `fill` (419) and `advance` (393-394). So one ply = one search =
one `predict()` yield = one leaf row. Hence

```
total_leaves = P_total := Σ_{episodes e} ℓ_e          (ℓ_e = #plies/decisions in episode e)
```

`P_total` is fixed by the workload — `cfg.episodes`, `cfg.seed`, the env, and
`cfg.max_steps` (line 383) — and is **completely independent of N, T, D, the
drain variant, and all park/reply timing.** None of those parameters appears in
the count of plies; they only reorder *when* the plies are evaluated. This is
the first half of the closed form and it is exact.

> Honesty note. The body of `policy.run_search` (`fiber_tree.hpp:50`) is NOT in
> the cleanroom; "exactly one `predict()` per search" (depth == 1) is the
> premise handed by the question, not something I can prove from these files.
> If a search instead made `q` yields, §1.1 still gives
> `total_leaves = Σ_e Σ_{plies} q_{e,ply}`; everything below carries through with
> `P_total` reinterpreted as total predict-yields. The N-scaling of the *ratio*
> (§3) is unchanged because it lives entirely in the denominator.

---

## 2. The denominator: how many messages a wave splits into

All N/D/timing dependence of the ratio lives in `total_msgs`. The decisive
structural fact is that **each worker thread is strictly synchronous**: it owns
one DEALER socket (`wire_leaf_pool.hpp:35`), and advances a search ONLY inside
`resume_with` (467), which it calls itself in a serial for-loop. There is no
intra-thread concurrency between "issue" and "search progress."

### 2.1 The issue discipline forms maximal waves

`issue_one()` gathers EVERY currently-ready slot into ONE request (the `for s in
[0,K)` loop, 437-443) and emits a single message (445, 450). It is invoked in
two while-loops, the prime (456) and the post-reply re-issue (474), each of the
form `while (inflight < D && issue_one()) {}`.

Claim: **each such while-loop emits at most one non-empty message.** Proof: the
first `issue_one()` gathers all ready slots and marks them all `submitted`
(447); the immediately-following `issue_one()` therefore finds the ready set
empty, `gathered.empty()` is true, and it returns false WITHOUT counting (444,
before 449-450). New slots cannot become ready in between, because nothing runs
a fiber between the two calls — the thread is synchronous. ∎

So a message is emitted once per "wave," and it carries **all slots parked at
that instant.**

### 2.2 In-flight depth D is never exercised beyond 1 (depth == 1)

Trace the steady state:

* **Prime (456).** After the prefill loop (454) all K slots are parked at ply 0
  and ready. `issue_one()` #1 → one message of (up to) K rows; `inflight = 1`.
  #2 finds nothing → stop. The prime emits exactly ONE message, regardless of D.
* **Drain iteration (457-475).** `recv_batch` (458) blocks for ONE whole reply
  (one correlation id; `wire_leaf_pool.hpp:106-132` reads all frames of one
  multipart message and matches the corr-id), `--inflight` → 0 (460). The
  for-loop (462-472) processes ALL completions of that reply: clears `submitted`
  (466), `resume_with` (467) which — depth == 1 — completes the search
  (`running` becomes false), then `advance`/`fill` spawns the next ply's search,
  which re-parks at its first leaf (running true again) synchronously, in the
  same loop iteration. After the for-loop EVERY slot whose completion was in the
  reply is re-parked and ready (or has exhausted its episode stream and gone
  inactive). The re-issue (474) then gathers all of them into ONE message;
  `inflight = 1`; the second `issue_one()` finds nothing → stop.

So `inflight` oscillates `1 → 0 → 1`; **only one message is ever outstanding.**
`D ≥ 1` is required (clamped at 287) but `D > 1` is never reached. Consequence:

> **`mean_rows_per_msg` is independent of D for depth == 1.**

This is because a reply re-parks all of its own slots *before* the worker
re-issues, so there is never a second batch of ready slots to fill the second
in-flight slot. (D would matter only if a search re-parked WITHOUT being
immediately re-gatherable, or if replies could carry a strict subset of
outstanding slots — neither happens here: replies are atomic per request,
§2.3.)

### 2.3 Park/reply TIMING does not split a wave either

A reply is atomic: `recv_batch` returns exactly the rows of the one request it
matches by corr-id (`wire_leaf_pool.hpp:106-132`; size-equality check
121-124). The server — both the production greedy drain
(`chocofarm/az/inference_server.py:160-201`) and the bench variant
(`cpp/stage_a/stage_a_server.py:54-70`) — sends one reply frame-set per
identity it drained, carrying exactly that request's rows
(`inference_server.py:197-200`; `stage_a_server.py:69-70`). The server's
SERVICE TIME and its cross-thread batching change *when* a reply lands, but not
*which* rows it contains. Since the worker re-parks all of a reply's slots
before re-issuing (§2.2), the arrival time cannot scatter a wave across
messages. Hence:

> **`mean_rows_per_msg` is invariant to all park/reply timing** (server service
> time, batch-size-dependent forward cost, pad/bucket shape, group-vs-leaf
> wakeup) **and to the drain variant.** Timing governs wall-clock throughput,
> not this telemetry.

(The one place timing *could* in principle matter — a thread's own searches
parking at staggered moments so that an `issue_one` sees only some of them — is
foreclosed by synchrony: between the corr-reply and the re-issue, every slot the
worker touches is advanced to its next park by the worker itself, sequentially,
with no yield to anything that could change the ready set.)

---

## 3. Closed form for the ratio

Per thread, by §2: exactly one message per **wave**, and in each wave every
currently-active slot advances exactly one ply (consumes one leaf). Let, for
thread `t`:

* `K = N · base`  slots (`base = ⌈pool_batch / T⌉`, `runtime_config.hpp:12-15`),
* `P_t = Σ_{e ∈ E_t} ℓ_e`  the thread's total plies (`E_t` = episodes with
  `idx ≡ t (mod T)`, `idx < cfg.episodes`; assignment via `next_idx += T`,
  402/323),
* `a_w` = number of active slots in wave `w`, `W_t` = number of waves.

Then per thread `my_leaves = Σ_w a_w = P_t` and `my_msgs = W_t`, so:

```
mean_rows_per_msg = total_leaves / total_msgs
                  = (Σ_t P_t) / (Σ_t W_t)
                  = P_total / Σ_t W_t.                       (★)
```

`W_t` is the number of waves until thread `t`'s LAST slot empties. Because a
slot grabs the next episode the instant it finishes (`fill`, 398-425, dynamic
`next_idx`), per-slot work self-balances: each of the K slots processes
≈ `P_t / K` plies. Thus

```
W_t = max_j (plies processed by slot j)  ≈  P_t / K  +  (tail straggler).      (†)
```

The straggler term is bounded by the residual imbalance once episodes run out:
at most one extra in-progress episode per slot, i.e. `W_t ≤ ⌈P_t/K⌉ + (ℓ_max − 1)`
where `ℓ_max = max_e ℓ_e` (a slot that grabs the longest remaining episode after
others have emptied). Combining (★) and (†):

```
                                P_total
mean_rows_per_msg  =  ───────────────────────────────                        (‡)
                       Σ_t ( P_t/K + r_t ),   0 ≤ r_t ≤ ℓ_max − 1
```

### 3.1 Bulk (homogeneous) closed form

If episodes are balanced across threads (`P_t ≈ P_total/T`) and each thread runs
many plies per slot, the tail `r_t` is negligible relative to `P_t/K`, and (‡)
collapses to the clean prediction:

```
                          P_total
mean_rows_per_msg  ≈  ───────────────  =  K  =  N · base.
                       T · (P_total/(T·K))
```

i.e. **the mean rises monotonically toward `K = N·base`**, the per-thread slot
count. With the straggler kept (homogeneous lengths `ℓ`, `M` episodes/thread,
`P_t = Mℓ`):

```
                    Mℓ                    K
mean ≈ K · ───────────────────  =  K · ─────────── ,
            Mℓ + K·(r̄)               1 + K·r̄/(Mℓ)
```

a value strictly below K that → K as `Mℓ/K = P_t/K → ∞` (long runs / many
episodes per slot relative to slot count), and that *decreases* the more slots
K there are relative to the work `P_t` (more stragglers, deeper drain tail).

### 3.2 The N-parametric prediction (the point of the question)

Holding the workload (`episodes`, `seed`, env ⇒ `P_total`, `ℓ_max`) and
`base`, `T` fixed and varying `N`:

```
N = 1 :  mean ≈ base / (1 + base·r̄/(P_t))                      (the prior's only datum)
N > 1 :  mean ↑ toward  N · base,
         with the gap to N·base widening only through the tail term
         (Σ_t r_t grows ∝ K = N·base, while P_total is fixed),
```

so the prediction is the explicit, falsifiable, N-parametric curve

```
                              P_total
   mean_rows_per_msg(N)  =  ─────────────────────────────────────── ,   (FINAL)
                             Σ_t ⌈ P_t / (N·base) ⌉  +  Σ_t r_t(N)
```

monotone increasing in N, with envelope `mean ≤ N·base` (exact equality only
when every wave is full, i.e. no slot ever idles before the last). This is what
the depth-1 fact buys that an N=1 prior cannot: at N=1 the ceiling is `base`;
the claim "doubling N nearly doubles mean_rows_per_msg until the tail bites" is
directly checkable against the emitted `mean_rows_per_msg`, `leaves`, `msgs`,
`fibers_per_thread`(=K) fields (498-500).

### 3.3 Exact corner: N·base ≥ all the thread's leaves at once

If a thread has so few plies that all its episodes finish within one wave's
worth of slot-advances — concretely when `K ≥ |E_t|` and every episode is length
1 — then the thread emits exactly one message of `P_t` rows and `W_t = 1`,
giving the trivially exact `mean = P_t`. More generally `W_t ≥ ⌈P_t/K⌉ ≥ 1`, so
`mean ≤ K` always and `mean ≤ P_total/T`-flavored bounds hold in the
small-workload corner. The driver also guards the empty run: `ms ? lv/ms : 0.0`
(496), so `total_msgs = 0` (no work) emits `mean_rows_per_msg = 0`.

---

## 4. Assume–guarantee (this side = the producer/driver)

**GUARANTEE (driver/source side).**
* Every emitted request is non-empty (`gathered.empty() ⇒ no submit`, 444) and
  carries `B = gathered.size() ≥ 1` rows with one fresh, monotone corr-id
  (`corr_seq_->fetch_add`, `wire_leaf_pool.hpp:84`).
* At most ONE request is outstanding per thread under depth == 1 (§2.2); in
  general at most `D` (287, loop guards 456/474).
* `total_leaves` counts each predict-yield exactly once (§1.1); `total_msgs`
  counts each non-empty request exactly once. The emitted ratio is therefore an
  honest `Σrows/Σmsgs`.

**RELY (on the sink/peer over the wire).** Checkable against the server code:
* Exactly one reply per request, same corr-id echoed in the leading frame
  (`inference_server.py:197-200` send `[ident,*envelope,resp]`; the corr-id
  travels as the envelope frame the ROUTER preserves), carrying exactly `B` rows
  in request order (`run_microbatch` 66-73 preserves per-identity counts/order;
  `stage_a_server.py:69-70`). If violated, `recv_batch` fails loudly
  (`wire_leaf_pool.hpp:116-124`, unknown corr / size mismatch) — it would abort
  the run, not corrupt the telemetry.
* Replies are atomic (a request's rows arrive together), which is what makes
  §2.3's timing-invariance hold; this is guaranteed by the server emitting one
  multipart reply per drained identity.

Neither RELY clause depends on the server's batching shape (pad-to-max vs
bucket) or wakeup granularity — those change service time only, which §2.3
shows the telemetry does not see.

---

## 5. Faithfulness ledger (every claim → file:line or named necessity)

| Claim | Anchor |
|---|---|
| `K = N·base`, `base = ⌈pool_batch/T⌉` | 286 + `runtime_config.hpp:12-15` |
| numerator counts each park once | `submitted` gate 447/466, `is_ready` 427-430, 449 |
| park ⟺ search suspended in `predict` | `fiber_leaf.hpp:26-27`, `fiber_tree.hpp:55,61` |
| `total_leaves = P_total` (depth==1) | premise + one search/ply (331,393-394,419) |
| each while-loop emits ≤1 message | synchronous thread + `submitted` (444,447) |
| prime emits exactly 1 message | prefill 454, then 456 trace |
| D never exceeds 1 in-flight | drain trace 457-475, atomic reply 458 |
| timing/drain-variant invariance | atomic reply `wire_leaf_pool.hpp:106-132`; server one-reply-per-request `inference_server.py:197-200`, `stage_a_server.py:69-70` |
| `mean ≤ K`, → K as work/slot → ∞ | (★)(†)(‡), waves≥⌈P_t/K⌉ |
| empty-run guard `mean=0` | 496 `ms ? … : 0.0` |
| socket opts that make recv block | LINGER=0, RCVTIMEO set (`wire_leaf_pool.hpp:39-41`); SND/RCVHWM, ROUTER_MANDATORY at OS default |

---

## 6. Confidence

**Medium-high.** The numerator (`total_leaves = P_total`) and the D- and
timing-invariance of the ratio are derived rigorously and exactly from the
cleanroom files. The clean envelope `mean → K = N·base` and its monotone rise in
N are forced by the wave structure. The one irreducible looseness is the TAIL
term `r_t` and hence the exact value below the envelope: it depends on the
episode-length distribution `{ℓ_e}`, which is workload-determined and lives in
code (`policy.run_search`, the env step) that is NOT in the cleanroom. So the
closed form is *parametric in the workload* (`P_total`, `{ℓ_e}`, `ℓ_max`), which
is the most precise faithful statement the cleanroom permits. The depth==1
premise itself is unverifiable here (§1.2 note) and is the load-bearing
assumption; if a search makes `q>1` yields, the numerator generalizes and the
N-scaling of the ratio is unchanged.
