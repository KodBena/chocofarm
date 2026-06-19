# Faithful model of the Python inference server (ROUTER) — bench bucketed/group drain, contrasted with production greedy drain

Role: **server side** of the leaf-evaluation transport boundary.
Focus: the **bench** drain variant (`cpp/stage_a/stage_a_server.py` `StageAServer._serve_batch`),
contrasted with the **production greedy** drain (`chocofarm/az/inference_server.py`
`InferenceServer._serve_batch`).

All file:line references are to the cleanroom tree
`/home/bork/w/vdc/chocobo/runs/leaf-eval-model-2/cleanroom/`. The bench server is a *subclass* of the
production server and inherits the entire transport mechanism (`__init__`, `_drain`, `serve_forever`,
`run_microbatch`, sockets); it overrides ONLY `_serve_batch`. So the production drain is the same machine
with a different `_serve_batch`; modeling both is modeling one socket/loop machine parameterised by the
batch-shaping function.

Throughout, the configuration is treated as PARAMETERS:

| symbol | meaning | source |
|---|---|---|
| `T` | producer threads = `pool_threads` (one DEALER each) | `runner_wire_batched.cpp:283`, `runtime_config.hpp` |
| `N` | `trees_per_thread` | `runner_wire_batched.cpp:285` |
| `base` | `ceil(pool_batch / T)` = `RuntimeConfig::fibers_per_thread()` | `runtime_config.hpp:12-15` |
| `K` | per-thread slots = `N * base` | `runner_wire_batched.cpp:286` |
| `D` | per-thread in-flight cap = `max_inflight_msgs` | `runner_wire_batched.cpp:287` |
| `M` | server `max_batch` (row budget) | `inference_server.py:149` |
| `E` | E-policy ∈ {padmax, bucket} | `stage_a_server.py:48,61-64` |
| `W` | wakeup ∈ {group, leaf} | `stage_a_server.py:49,57` |
| `BUCKETS` | `(64,256,512)` | `stage_a_server.py:30` |

`max_batch` default differs by entrypoint: production constructor default 256 (`inference_server.py:145`);
bench CLI default 512 (`stage_a_server.py:89`). Keep it the symbol `M`.

---

## 0. One-paragraph operational summary (derived forward from the code)

A single Python thread runs `serve_forever` (`inference_server.py:219-226`): forever, it `_drain`s a
batch of queued requests off ONE ROUTER socket, and if non-empty calls `_serve_batch`. `_drain`
(`:160-186`) first *blocks* in a 100 ms-quantised poll loop until the socket is readable (or `stop`), then
greedily pulls every currently-queued multipart frame with `recv_multipart(NOBLOCK)` until either the
socket is momentarily empty (`zmq.Again`) or the accumulated **row** count reaches `M`. Each pulled frame
is split `ident=frames[0]`, `envelope=frames[1:-1]`, `payload=frames[-1]`, decoded; a decode failure is
printed and the frame is *silently dropped with no reply* (`:181-183`). `_serve_batch` then shapes those
requests into one or more forwards and scatters one reply frame-stack per request back through the ROUTER.
The two drains differ ONLY in the shaping function:

- **production greedy** (`inference_server.py:192-200`): ALL drained requests → ONE `run_microbatch` with
  `pad_to=M` → ONE forward of exactly `max(total_rows, M)` rows. (Padding only grows; never truncates.)
- **bench** (`stage_a_server.py:54-70`): partition the drained list by `W`. `group` → one group = all
  drained → one forward; `leaf` → one group *per request* → one forward each. Per group, `E` picks the
  padded width: `padmax`→`M`; `bucket`→smallest of `{64,256,512}` ≥ real rows, clamped to 512.

---

## 1. STATE MACHINE (operational, parametric)

The server thread is strictly sequential (single-threaded by construction:
`stage_a_server.py:97` runs `serve_forever` on one daemon thread; `config.py:5-6` pins XLA to a single
Eigen thread and OMP to 1, so even the forward is single-threaded — there is no intra-server concurrency
to model). The state is therefore a program counter over the drain/serve loop plus the socket's internal
input queue, which is *shared* with the kernel/zmq IO thread and is the only concurrent element.

### 1.1 States

| name | meaning |
|---|---|
| `BLOCKED_POLL` | inside `_drain`'s poll loop (`:163-166`); no request readable; waiting up to 100 ms per iteration |
| `DRAINING` | inside the NOBLOCK pull loop (`:171-185`); accumulating `drained` and `total_rows` |
| `SHAPING` | `_serve_batch` entered; partitioning into `groups` (bench `:57`) / building `rows` |
| `FORWARDING(g)` | executing `run_microbatch`→`forward_fn` for group `g` (`run_microbatch:61`, the JAX call) — the SINK SERVICE; consumes a positive nondeterministic duration |
| `SCATTERING(g)` | sending one reply stack per request of group `g` (`:69-70` bench / `:200` prod) |
| `IDLE_DISPATCH` | `serve_forever` top (`:222-225`): decide drain-again vs serve |
| `STOPPED` | `self._stop` true; `_drain` returns `[]`, loop exits; (`close` → `STOPPED_CLOSED`) |

The socket input queue `Q` is an auxiliary, concurrently-mutated multiset of queued request frames
(producer side enqueues via TCP+zmq IO thread; server dequeues in `DRAINING`). `Q` is bounded by RCVHWM
(default 1000 messages, since the code sets no `ZMQ_RCVHWM`).

### 1.2 Transitions (guard / action / code_ref / free-choice)

| # | from → to | guard | action | code_ref | free? |
|---|---|---|---|---|---|
| t1 | IDLE_DISPATCH → BLOCKED_POLL | not stop | enter `_drain` | inference_server.py:223,160-163 | no |
| t2 | BLOCKED_POLL → BLOCKED_POLL | poll timed out (no readable in 100 ms) AND not stop | re-loop | :163-166 | **yes** (whether a request arrives within the window is the producer's nondeterministic timing) |
| t3 | BLOCKED_POLL → DRAINING | `poll()` returned truthy (≥1 readable) | break poll loop | :165-166 | no |
| t4 | BLOCKED_POLL → STOPPED | `self._stop` set | `return []` | :163,167-168 | **yes** (stop is async via `stop()`/signal, `stage_a_server.py:110,124`) |
| t5 | DRAINING → DRAINING | `total_rows < M` AND `recv_multipart(NOBLOCK)` yields a frame AND decode OK | append `(ident,envelope,X)`; `total_rows += X.shape[0]` | :171-185 | **yes** (how many frames are queued *right now* is producer-timing dependent) |
| t6 | DRAINING → DRAINING | frame pulled but `decode_request` raised | `_reject(ident,exc)` (print); `continue` — NO reply, NOT counted | :179-183 | no (only on a malformed/short/non-finite/version-mismatch frame; RELY says peer never sends one) |
| t7 | DRAINING → SHAPING | `recv_multipart` raised `zmq.Again` (queue momentarily empty) OR `total_rows ≥ M` | break drain loop; return `drained` | :171,173-175 | **yes** (Again-vs-more is producer timing; the M cut is deterministic given arrivals) |
| t8 | SHAPING → FORWARDING(g0) | `drained` non-empty | build `groups`, pick `pad_to` for first group | bench stage_a_server.py:54-65 / prod inference_server.py:196-198 | no |
| t9 | FORWARDING(g) → SCATTERING(g) | forward returned `out_arr` with `shape[0] ≥ B` | slice per-request rows; `encode_response` | run_microbatch:61-71 | no |
| t10 | SCATTERING(g) → FORWARDING(g+1) | bench, `W==leaf`, more groups remain | next group | stage_a_server.py:58 | no |
| t11 | SCATTERING(g) → IDLE_DISPATCH | last group scattered | return from `_serve_batch` | :69-70 / :200 | no |
| t12 | IDLE_DISPATCH → BLOCKED_POLL | drained empty last cycle (only reachable if all frames were rejected → `_serve_batch` not called) | loop with empty drained | inference_server.py:224 | no |
| t13 | any non-FORWARDING → STOPPED | `_stop` true at a loop guard | exit loops | :163,222,224 | **yes** |

Notes on free choices:
- **t2/t5/t7** are the *injection points of the producer's nondeterministic pacing*. The server has no
  control over how many requests are in `Q` at the instant it drains; that is entirely a function of when
  the T×(≤D) in-flight producer messages were emitted and how fast TCP delivered them. The model leaves
  exactly this latitude: any `0 ≤ |drained| ≤ (number queued)` consistent with arrivals, subject to the
  row-cap stopping rule of t5/t7.
- The server NEVER blocks for a *second* message once it has one: the drain is NOBLOCK. So the batch it
  forms is "whatever happened to be queued at the moment the first poll fired, plus anything that raced in
  before the queue went momentarily empty." There is no linger/Nagle delay, no minimum-batch wait. This is
  load-following, not batch-accumulating.

### 1.3 Why FORWARDING is a state, not an instant

`run_microbatch:61` calls `forward_fn(params, Xb, ...)` which is `jit_forward_core`
(`inference_server.py:22-34`): a `jax.jit`-wrapped `forward_core` (`forward.py:3-18`) — two-or-four dense
GEMMs + ReLU + value/policy heads on a matrix of `rows × in_dim`. This is the SINK SERVICE TIME; it is
positive and batch-size-dependent (see §2). It is single-threaded (`config.py:5-6`). The server thread is
fully occupied here: while FORWARDING, it is NOT draining, so `Q` keeps growing from the producers — the
classic gather/compute alternation.

---

## 2. TIMING MODEL

### 2.1 Source emission (RELY on the producer; grounded in the peer code)

A producer thread `tid` (`runner_wire_batched.cpp:312`) owns one DEALER (`wire_leaf_pool.hpp:35`,
`ZMQ_DEALER`). It holds `K = N*base` slots. Its emission discipline (`run_episodes_wire_pipelined`):

1. Prime: `while inflight_msgs < D && issue_one()` (`:456`). `issue_one` (`:434-452`) gathers **every
   currently-ready slot** (`is_ready`: active, parked at a leaf, not already submitted — `:427-430`) into
   ONE message and `submit_batch`es it (`:445`), marks those slots `submitted`, `++inflight_msgs`.
2. Steady: `while inflight_msgs>0`: `recv_batch()` (blocking recv, one message) `:458`; `--inflight_msgs`;
   resume each returned slot, possibly `advance`/`fill` it (which re-parks it at a new leaf or retires it);
   then refill `while inflight_msgs<D && issue_one()` (`:474`).

Consequences for the **timing the server sees** (each checkable against peer code):

- **R-rate.** A thread has at most `D` messages in flight at once (`:456,474` gate on `inflight_msgs<D`).
  Across all threads the wire carries at most `T*D` in-flight request messages. So `Q` (server input queue)
  never exceeds `T*D` *un-served* request messages — RCVHWM (default 1000) is not hit unless `T*D > 1000`.
- **R-rows-per-msg.** One message batches all *simultaneously-ready* slots of a thread, so its row count
  `B_msg ∈ [1, K]`. Early in a run after priming, many slots are ready at once → large `B_msg` (up to `K`);
  in steady state each reply frees the slots it carried, and only those re-park, so `B_msg` trends toward
  the per-reply freed count. The mean is reported as `mean_rows_per_msg` (`:496-500`). **N-dependence:**
  `K=N*base`, so the *ceiling* on a single message's rows grows linearly in `N`; the first few messages
  after priming can be as wide as `K=N*base` rows.
- **R-park-interval (nondeterministic).** Between receiving a reply for a slot and that slot's next request,
  the producer does internal search work (`resume_with`→fiber runs the MCTS to the next leaf:
  `fiber_tree.hpp:58-62`, `fiber_leaf.hpp:24-29`) of an interval the code does NOT fix — model it as a
  positive nondeterministic duration `τ_park ∈ (0, ∞)`, possibly different per slot/visit. A retired slot
  that `fill`s a fresh episode takes a longer (still nondeterministic, positive) interval. **This is the
  irreducible source nondeterminism; do not collapse it.**
- **R-corr-blocking.** A producer that has hit `inflight_msgs==D` will not emit again until a reply lands
  (`:457-474`). So a slow server *throttles* the source: `Q` cannot grow without bound; the producers
  back-pressure themselves through the D cap. This is the only flow control; there is no SNDHWM stall on
  the server's send because ROUTER send is non-blocking-droppy by default (see §2.3).

### 2.2 Sink service (the forward; modeled, NOT collapsed)

For a forward over a matrix of `r` rows (`r = pad_to` when `pad_to>B`, else `r=B`; `run_microbatch:58-61`):

- Service time `S(r) > 0`, monotonic non-decreasing in `r` (more GEMM rows = more FLOPs;
  `forward.py:5-17`). Single-threaded CPU (`config.py:5-6`), so roughly affine: `S(r) ≈ c0 + c1·r` with
  `c0` a fixed per-call overhead (Python/JAX dispatch, the `np.concatenate` at `run_microbatch:55`, padding
  alloc `:59`) and `c1` the per-row GEMM cost. Model `S` as bounded nondeterministic within
  `[S_lo(r), S_hi(r)]`, both positive and non-decreasing in `r`; the only causal constraints are positivity
  and monotonicity. Do NOT fix `S` to a constant.
- **Compilation quantisation (the AOT/shape story).** `jit_forward_core` caches exactly ONE jitted callable
  in `_jit_forward_cache` (`inference_server.py:20,24,33-34`), but `jax.jit` *recompiles per distinct input
  shape* and caches each compilation internally. So the FIRST forward at a given row width pays a large
  one-time compile cost `S_compile(r) ≫ S(r)`; subsequent forwards at that *same* width are cheap.
  `warmup` (`inference_server.py:202-217`, called by bench `build` at `stage_a_server.py:82` with
  `sorted(set(BUCKETS)|{M})`) pre-pays the compiles for widths `{64,256,512,M}`. Therefore:
  - **padmax (E=padmax):** every forward is width `M` → exactly ONE compiled shape, warmed → steady
    `S(M)`. Padded rows are pure overhead but the *shape* is constant; service time is the constant-shape
    `S(M)` regardless of real load.
  - **bucket (E=bucket):** real rows snapped UP to one of `{64,256,512}` (`stage_a_server.py:32-37,64`),
    clamped to 512 for real>512. Three AOT shapes, all warmed (`build` warms `BUCKETS`). Service time is
    a *step function of real rows*: `S(64)` for `real≤64`, `S(256)` for `65≤real≤256`, `S(512)` for
    `real≥257`. Smaller real load ⇒ smaller bucket ⇒ smaller `S`. This is the bench drain's *point*:
    service time tracks load in three steps instead of always paying `S(M)`.
  - **un-warmed widths.** If `M ∉ {64,256,512}` (e.g. bench default `M=512` ∈ buckets — warmed; but a
    custom `M`, or production `M=256` ∈ buckets — warmed). In production, only width `M` ever occurs and
    `warmup` is whatever the caller passes; an un-warmed first forward pays `S_compile`. Model a one-time
    `S_compile(r)` penalty on the first occurrence of any width not in the warmed set.

### 2.3 Causal constraints (what the timing model MUST respect)

1. A reply for corr `c` cannot be SCATTERED before the FORWARD that produced its row completes
   (t9 precedes t10/t11; `run_microbatch` returns before `send_multipart`).
2. A producer cannot emit a request whose features depend on a prediction it has not yet received: a slot
   is re-`is_ready` only after `resume_with` consumed its reply (`runner_wire_batched.cpp:466-467,429`).
   So *per slot*, request_{k+1} strictly causally follows reply_k. (Across slots there is no such order.)
3. A reply cannot be received by the producer before the server sent it (TCP). Combined with (2):
   the per-slot cycle request→forward→reply→park→request has strictly positive total duration.
4. Durations are positive: `τ_park>0`, `S(r)>0`. The poll quantum is exactly 100 ms (`_POLL_INTERVAL_MS`,
   `inference_server.py:142`): an idle server checks `stop` and re-arms at most every 100 ms; a request
   that arrives mid-quantum is seen at the next `poll` return, which fires as soon as readable (poll
   returns early on readiness, so the 100 ms is a *ceiling on idle latency to notice stop*, not added
   latency to a real request — `zmq_poll` wakes on POLLIN immediately).

### 2.4 Was any timing collapsed to a constant?

No. `τ_park` and `S(r)` are both modeled as bounded nondeterminism (positive; `S` monotone in `r`). The
only constant is the structural poll quantum 100 ms, which is a code literal (`:142`), not a collapsed
service time. Compilation is modeled as a one-time per-shape penalty, not amortised away.

---

## 3. ASSUME–GUARANTEE CONTRACT (server as one party)

### 3.1 RELY (assumptions about the peer producer, each checkable against `wire_leaf_pool.hpp` /
`runner_wire_batched.cpp`)

- **A1 (frame shape).** Each request is a 2-frame DEALER message: `[corr(8 bytes), payload]`. On the ROUTER
  the server sees `[ident, corr, payload]`, so `envelope=frames[1:-1]=[corr]`, `payload=frames[-1]`.
  Ground: `submit_batch` sends `corr` with `ZMQ_SNDMORE` then `payload` (`wire_leaf_pool.hpp:86-91`); the
  DEALER prepends no identity of its own, ROUTER prepends one ident frame.
- **A2 (payload validity).** `payload` decodes under `decode_request`: protocol byte `==2`
  (`wire_spec.hpp:8`, `inference_wire.py:47`), `B≥1`, `in_dim≥1`, exact length `B*in_dim*4`, all finite.
  Ground: producer encodes via `wire::encode_request` (`inference_wire.hpp:51-70`), same protocol byte and
  layout; features come from a live search (`fiber_leaf.hpp:24` `ch.features`), finite by construction.
  ⇒ Under RELY, transition **t6 (reject)** is *never taken*; it exists only as a defensive ADR-0002 path.
- **A3 (in-flight bound).** A thread keeps ≤ `D` messages outstanding; the wire holds ≤ `T*D` unanswered
  request messages. Ground: `inflight_msgs<D` gates every `issue_one` (`runner_wire_batched.cpp:456,474`).
- **A4 (rows per message).** A message carries `B ∈ [1, K]` rows, `K=N*base`. Ground: `issue_one` gathers a
  subset of the `K` slots (`:437-443`); at least one (`gathered.empty()` ⇒ return false `:444`).
- **A5 (correlation echo expectation).** The producer matches replies by `corr` and requires the reply's
  predicted-row count to equal the request's slot count (`wire_leaf_pool.hpp:115-124`,
  `recv_batch:121-124`). So the server MUST return exactly `B` predictions for a `B`-row request, in row
  order, under the SAME `corr` and ROUTER ident.
- **A6 (liveness / no spurious extra frames).** Producer's `recv_corr_payload` expects ≥2 frames with an
  8-byte leading `corr` frame (`wire_leaf_pool.hpp:157-162`). The server echoes `[ident, *envelope, resp]`
  = `[ident, corr, resp]`; the DEALER strips `ident`, leaving `[corr, resp]` — exactly 2 frames, leading 8
  bytes. Matches.

### 3.2 GUARANTEE (what the server provides, checkable against server code)

- **G1 (reply identity + ordering).** For each accepted request `[ident, corr, payload]`, the server sends
  exactly one `[ident, corr, resp]` (`inference_server.py:200` / `stage_a_server.py:70`), preserving the
  ROUTER ident (routes back to the originating DEALER) and the corr frame verbatim (it is part of
  `envelope`, never inspected or rewritten).
- **G2 (prediction count + order).** `resp` encodes exactly `B` predictions for a `B`-row request, in the
  input row order. Ground: `run_microbatch` slices `v[off:off+n]` per request using the recorded `counts`
  (`:64-72`); padding rows are appended *after* all real rows (`:59`) and sliced off (`out_arr[:B]` via the
  per-request offsets, never reaching pad rows). ⇒ satisfies A5.
- **G3 (no reordering across requests).** `run_microbatch` preserves drained order in both `identities` and
  `counts` (`:46,68`); bench groups preserve order too (`groups` is a partition of `drained` in order,
  `stage_a_server.py:57`). So reply order matches request order — though DEALER/ROUTER correlation makes
  order immaterial to correctness (A5 uses corr, not order).
- **G4 (drop-on-malformed, never on valid).** A valid request is always answered (no path drops a decoded
  request); only an *undecodable* frame is dropped with no reply (`:181-183`). Under RELY (A2) this never
  fires. ⇒ The server provides at-most-once and (under RELY) exactly-once reply per request.
- **G5 (no head-of-line starvation beyond M-row batching).** Every drained request in a cycle is answered
  in that cycle's `_serve_batch` before the next `_drain` (`serve_forever:223-225`). The server cannot
  "forget" a drained-but-unanswered request: `_serve_batch` iterates the whole `drained` list.
- **G6 (bounded batch).** A drain accumulates until `total_rows ≥ M` (`:171`); but because the check is at
  loop *top*, ONE message can carry the total past `M` (a single `B_msg=K` message with `K>M` is accepted
  whole). So the realized forward width can EXCEED `M` when a single message's rows exceed the remaining
  budget. Production then pads to `max(B, M)=B` (no padding; `run_microbatch:58` `pad_to>B` false). Bench
  bucket clamps the *bucket* at 512 but `run_microbatch` still forwards all `B` real rows if `B>512`
  (`pad_to=512 < B` ⇒ no pad, width `B`). This is a real over-`M` execution the model must admit (§5.4).

---

## 4. DEGREES OF FREEDOM (each with code_ref, behaviors admitted, N-dependence)

**DOF-1 — drain batch composition (which/how-many requests form a cycle).**
Code: `_drain:171-185`. The NOBLOCK loop pulls whatever is queued *now*; `|drained|` and the per-request
row counts are set by producer arrival timing, not the server. Admits any `drained` consistent with the
arrival order and the `total_rows<M` stopping rule (with the loop-top check ⇒ possible overshoot past `M`).
N-dependence: as `N` grows, `K=N*base` grows, so the *maximum rows a single producer message contributes*
grows linearly; the early-run "fat" batches (post-priming, many ready slots) get fatter ∝ `N`. The *number
of distinct messages* queued is bounded by `T*D` independent of `N`.

**DOF-2 — wakeup granularity `W` (bench only).**
Code: `stage_a_server.py:57`. `group`→ one forward over the concatenation of all drained rows;
`leaf`→ one forward per drained *message* (NOT per row — a group is one `d`=one drained request, which may
itself carry `B_msg>1` rows from a multi-slot producer message). Admits: with `W=group`, forwards/cycle=1
and rows/forward = Σ over drained; with `W=leaf`, forwards/cycle = `|drained|` and rows/forward = each
message's own `B_msg`. N-dependence: `W` itself is N-independent, but its *effect* scales with N — under
`leaf` each forward is one producer message of up to `K=N*base` rows, so `leaf` does NOT defeat batching
when N is large (a single message is already a fat batch); under `group`, a cycle can fuse up to `T*D`
messages of up to `K` rows each, so group's batch can reach ~`min(T*D*K_effective, …)` rows, but is cut by
the `M`-row drain stop (DOF-1). Contrast: production has NO `W` knob — it is always `group`-equivalent
(one forward per drain).

**DOF-3 — E-policy `E` (bench) vs production fixed padmax.**
Code: bench `stage_a_server.py:61-64`; production `inference_server.py:198` (`pad_to=self._max_batch`,
i.e. permanently padmax). `padmax`→ width `M` always (one shape). `bucket`→ width
`_bucket_for(real)∈{64,256,512}` (`:32-37`), three shapes, step service time. Admits: different
(rows-forwarded, service-time-class) pairs for the same real load. N-dependence: as `N`↑, real rows per
forward rise, so `bucket` climbs its step ladder (64→256→512) and saturates at 512; once real>512 the
bucket clamps and bench bucket == "no padding, width = real", so for large `N` bucket and padmax converge
only if `M=512`. For real ≤ 64 (small N, or `leaf` with thin messages) bucket pays `S(64)` while padmax
pays `S(M)` — the divergence is largest at small N / thin load.

**DOF-4 — order of poll/serve interleaving with concurrent enqueues.**
Code: the `Q` mutation by the zmq IO thread is concurrent with `_drain`. Admits: a request that arrives
*during* FORWARDING is not seen until the *next* `_drain`; a request that arrives during DRAINING may or
may not be pulled depending on whether it lands before the `zmq.Again`. Free choice t5/t7. N-dependence:
larger `N` ⇒ longer FORWARDING (fatter batches, §2.2) ⇒ wider window during which arrivals accumulate ⇒
next drain is fatter — a *self-reinforcing* batching effect (see §5.2 stability).

**DOF-5 — stop timing.**
Code: `_stop` is set asynchronously (`stage_a_server.py:110` after the subprocess exits, or `:124` on
SIGINT/SIGTERM). Admits: stop observed at any loop guard (`:163,167,222,224`). t4/t13. N-independent.

**DOF-6 — param reload (production only).**
Code: production `_serve_batch:194` calls `self._params_source.poll()` *every cycle* (may swap weights
mid-run if a `RedisParamsSource` reports a new version, `inference_server.py:129-138`). Bench
`_serve_batch:56` calls only `current()` — **no reload**, weights are frozen (`StaticParamsSource`,
`stage_a_server.py:78`). Admits, production-only: a forward in cycle `i+1` may use different params than
cycle `i`; the *shape*/timing model is unaffected (params don't change matrix dims). N-independent. This is
a genuine production/bench divergence at the interface (bench never reloads).

---

## 5. REPRESENTATIVE EXECUTIONS (concrete traces; stability; N-scaling)

Notation: producer threads `T`, each ≤`D` in flight, `K=N*base` slots. `Q` = server input queue.

### 5.1 Bench, W=group, E=bucket, moderate load — the canonical bench cycle
1. IDLE_DISPATCH → t1 → BLOCKED_POLL. (serve_forever:223)
2. `Q` empty; t2 loops (≤100 ms idle) until producer priming lands `g` messages
   (`runner_wire_batched.cpp:456`). t3 → DRAINING.
3. t5×g: pull all `g` queued messages, `total_rows = Σ B_msg = real`. Suppose `real=180`. `total_rows<M`
   so loop continues until `zmq.Again` → t7 → SHAPING. (_drain:171-185)
4. SHAPING: `W=group` ⇒ `groups=[drained]` (one group). `E=bucket` ⇒ `_bucket_for(180)=256`
   (`stage_a_server.py:64,32-37`). (180≤256.)
5. FORWARDING(g0): `run_microbatch(..., pad_to=256)`. B=180<256 ⇒ pad 76 rows (`run_microbatch:59`).
   width 256, warmed ⇒ steady `S(256)`. counters: `n_forwards+=1`, `n_real_rows+=180`, `n_padded_rows+=76`
   (`stage_a_server.py:66-68`). t9.
6. SCATTERING(g0): one `[ident,corr,resp]` per drained message (≤g sends). t11 → IDLE_DISPATCH.
**Stability: self-reinforcing** in steady state — each reply frees its slots, they re-park after `τ_park`,
re-arrive ~together, next cycle is similarly sized. **N-scaling:** raise `N` ⇒ `real` rises (more slots) ⇒
`_bucket_for` steps 256→512 then clamps; once `real>512`, bucket pad=0 and width=`real` (>512), service
`S(real)`. So bench-bucket degenerates to "no padding, true width" for large N — efficient, but loses the
fixed-shape compile guarantee if `real` takes many distinct values >512.

### 5.2 Bench, W=group, E=bucket, FORWARDING-window accumulation (DOF-4 self-reinforcement)
1. Cycle `i` FORWARDS a fat batch (S large). During `S`, the server is off-socket; `Q` accumulates new
   producer messages (each thread that got a reply last cycle re-issues after `τ_park`).
2. Next `_drain` (t5) pulls *all* of them in one go → even fatter `real` → larger bucket → larger `S`.
3. Steady state: batch size converges to the fixed point where one service time `S(real)` ≈ the time for
   the offered load (`T` threads, ≤`D` msgs, `τ_park`) to refill `real` rows. **Stability: self-reinforcing
   / converges to a load-determined fixed point.** **N-scaling:** larger `N` raises the per-thread row
   supply (`K=N*base`), pushing the fixed point up the bucket ladder and saturating at the 512 clamp; the
   batching becomes *more* effective (fewer, fatter forwards) as `N` grows — until `real` exceeds 512 and
   the bucket stops helping.

### 5.3 Bench, W=leaf, E=bucket — no cross-message coalescing
1. Drain pulls `g` messages with row counts `[b1,…,bg]` (e.g. `[120, 8, 300]`).
2. SHAPING: `W=leaf` ⇒ `groups=[[d1],[d2],[d3]]` (`stage_a_server.py:57`).
3. FORWARDING(d1): bucket(120)=256 → `S(256)`; FORWARDING(d2): bucket(8)=64 → `S(64)`;
   FORWARDING(d3): bucket(300)=512 → `S(512)`. THREE forwards (`n_forwards+=3`), padded rows
   `136+56+212=404`. t10 chains them; t11 after the last.
**Contrast with group:** group would have done ONE forward of `real=428` → bucket 512 → `S(512)` once.
So `leaf` *multiplies* per-cycle service by the message count and pads each message up to its own bucket —
strictly more compute for the same work, UNLESS each message is already a fat batch. **N-scaling:** because
a single producer message can carry up to `K=N*base` rows, when `N` is large each `leaf` forward is itself
a fat batch (each `b_j` large) — so `leaf` and `group` converge in *rows/forward* as N grows; the
divergence (leaf's extra forwards + per-message padding) is **largest at small N / thin messages**, where
each message carries few rows and leaf pads every one of them up to 64.

### 5.4 Over-M single-message batch (G6 / DOF-1 overshoot) — both bench and production
1. After priming with large `N`, a thread's first `issue_one` gathers ALL `K=N*base` ready slots into ONE
   message of `B_msg=K` rows (`runner_wire_batched.cpp:437-451`). Say `K=600 > M=512`.
2. `_drain` t5: first pull adds 600 rows; `total_rows=600 ≥ M`, but the `<M` test was checked at loop top
   *before* this pull, so the 600-row message is accepted whole; next loop iteration's top test fails →
   t7 → SHAPING with `drained` = that one 600-row message. (`_drain:171-185`)
3. Bench group, E=bucket: `_bucket_for(600)=512` (clamp, `:36-37`), but `run_microbatch` sees
   `pad_to=512 < B=600` ⇒ NO padding (`:58`), forwards width **600**. Service `S(600)>S(512)`.
   Production: `pad_to=M=512<600` ⇒ no padding, width **600**, `S(600)`. Both forward >M rows.
**Stability: transient** (only the post-priming burst is this fat; steady state settles to §5.2's fixed
point unless N keeps every slot perpetually ready). **N-scaling:** the *existence* of this overshoot is
gated on `K=N*base > M`, i.e. it APPEARS once `N > M/base` and the overshoot magnitude grows ∝ `N`. For
small `N` (`K≤M`) it cannot occur. This is a genuine code-permitted execution that a "never exceed M" model
would wrongly forbid.

### 5.5 Production greedy, E≡padmax, W≡group, with param reload (DOF-6)
1. `_drain` identical to bench (inherited). Suppose `real=180`.
2. `_serve_batch:194`: `poll()` — if `RedisParamsSource` reports a new version, weights swap *now*
   (`inference_server.py:129-138`); else `current()`.
3. ONE `run_microbatch(pad_to=M)`; width `M` always (e.g. 256), pad 76. `S(M)` constant-shape.
4. SCATTER all. **Contrast:** production NEVER buckets and NEVER does per-leaf forwards; its width is
   *always* `M` (one compiled shape, constant service time regardless of real load) and it MAY reload
   weights mid-run. **N-scaling:** production's width is N-INVARIANT (`M`) until a single message's rows
   exceed `M` (§5.4), so its service time is flat in `N` up to the overshoot threshold — the opposite of
   bench-bucket, whose service climbs the bucket ladder with `N`.

### 5.6 Idle / stop
1. No producer load: BLOCKED_POLL loops t2 every 100 ms checking `_stop` (`:163-166`). Zero forwards.
2. Subprocess ends → `server.stop()` (`stage_a_server.py:110`) sets `_stop`; next poll-loop guard t4 →
   STOPPED, `_drain` returns `[]`, `serve_forever` exits (`:222`), `t.join`, `close(linger=0)`
   (`:236`). **N-independent.** **Stability: terminal.**

---

## 6. N-DEPENDENCE SUMMARY

The single most important way the SERVER's behavior changes as `N=trees_per_thread` grows: **the offered
batch width grows linearly** (a single producer message can carry up to `K=N*base` rows;
`runner_wire_batched.cpp:286,437-451`), which (a) for **bench bucket** climbs the `{64,256,512}` step
ladder and then *clamps* at 512 — beyond which `run_microbatch` forwards the true (>512) width with no
padding, so bucketing stops helping and service time `S(real)` rises with `N`; (b) for **bench padmax** and
**production** keeps service at the fixed-shape `S(M)` until a single message exceeds `M` rows
(`K=N*base>M`), past which the forward width (and service time) grow ∝ `N` via the loop-top overshoot
(§5.4); (c) makes the FORWARDING window longer, which *self-reinforces* batch accumulation (§5.2), so
larger `N` yields fewer, fatter forwards until the 512/`M` ceilings bite. Message *count* on the wire stays
bounded by `T*D` independent of `N`; only rows-per-message scale with `N`.

---

## 7. DOF-CONTROL NOTES (what removing each constraint would wrongly admit)

- **DOF-1 control (the `total_rows<M` loop-top test, `:171`).** If the model enforced a hard "batch ≤ M
  rows" cap, it would FORBID the real §5.4 overshoot (single message > M forwarded whole). Conversely, if
  the model let the drain block for more messages, it would ADMIT batches the NOBLOCK drain cannot form
  (the drain never waits for a *second* message). Both constraints must be exactly as coded:
  greedy-non-blocking with loop-top row check.
- **DOF-2 control (`groups` partition, `:57`).** Removing the W distinction (always group) would FORBID the
  `leaf` executions (per-message forwards, per-message padding, `n_forwards=|drained|`). Always-leaf would
  FORBID the group coalescing. Production has NO leaf mode at all — modeling production with a leaf option
  would ADMIT executions production cannot produce.
- **DOF-3 control (`pad_to` selection, `:61-64`).** Collapsing bucket to padmax (or vice versa) mis-models
  service time: bucket's *step* service vs padmax's *flat* `S(M)`. Letting bucket pad *below* real rows is
  impossible — `run_microbatch:58` pads only when `pad_to>B`, so bucket never truncates; a model that
  truncated would ADMIT wrong (too-short) forwards.
- **DOF-4 control (concurrent enqueue during FORWARDING).** Forbidding mid-forward arrivals would FORBID
  the self-reinforcing accumulation (§5.2) and make batch size N-flat — unfaithful. Admitting the server
  to *see* a mid-forward arrival within the same cycle would ADMIT an execution the single-threaded loop
  cannot produce (it is off-socket during the forward).
- **DOF-5 control (`_stop` async).** Forcing stop only at quiescence would FORBID the real mid-batch stop
  (t13 can fire at `:222/:224` between cycles; note `_drain` re-checks `_stop` at `:163,167` so a stop is
  honored before the next forward, never mid-forward).
- **DOF-6 control (production reload).** Allowing bench to reload weights would ADMIT executions bench
  cannot produce (`StaticParamsSource.poll()` returns `None`, `:111`, and bench calls only `current()`).
  Allowing production to *never* reload would FORBID a real mid-run weight swap.

---

## 8. FIDELITY SELF-AUDIT

**Possible over-permissions (model admits something the code may not):**
- The model treats `S(r)` and `τ_park` as free within positive/monotone bounds. The real system has a
  *deterministic* (if unknown) service curve; treating it as nondeterministic is intentional latitude
  (the prompt requires it), but a downstream check that assumed e.g. *unbounded* `S` could derive
  executions (server permanently behind) that, while causally admissible, the real bounded-FLOP forward
  would not sustain. Bound `S` by `S_hi(r)` if a tighter check is needed.
- I model t5/t7 as allowing ANY queued-subset to be drained. In reality the NOBLOCK drain pulls in strict
  FIFO socket order and stops only at the FIRST `zmq.Again`; it cannot "skip" a queued message. The model
  should constrain drained to be a *prefix* of the arrival order up to the first gap — I assert this in
  prose (G5, §1.2 note) but a naive reading of DOF-1 could over-permit non-prefix drains. Constraint:
  drained is a contiguous arrival-prefix bounded by the M-row loop-top rule.
- ROUTER send with `ROUTER_MANDATORY` unset *silently drops* a reply to a vanished/unknown ident
  (zmq default). I model G1 as "always sends one reply"; the send can be silently dropped if the producer
  disconnected. Under RELY (peer alive until run end) this doesn't fire, but the model should note the
  reply is *best-effort* at the socket layer, not guaranteed delivered.

**Possible over-constraints (model forbids something the code can do):**
- I assert exactly 2 envelope frames (`[corr]`). If a producer ever sent extra frames between corr and
  payload, `envelope=frames[1:-1]` would carry them and the server would echo them back verbatim
  (`:200/:70`) — the server is AGNOSTIC to envelope arity. The model is correct (it handles any arity via
  `*envelope`), but my RELY A1 pins arity=1; that is a RELY (justified by peer code `wire_leaf_pool.hpp:86-91`),
  not a server constraint. The server itself admits any envelope arity — noted so a downstream check does
  not treat arity=1 as a server invariant.
- The 100 ms poll quantum: I note `zmq_poll` returns early on POLLIN, so a real request incurs no 100 ms
  delay. If a downstream model treated the 100 ms as added per-request latency it would over-constrain
  (slow the server) — explicitly flagged in §2.3.
- I treat `warmup` as covering `{64,256,512,M}` for bench (`stage_a_server.py:82`). If `M ∉` that set the
  first width-`M` forward pays compile; for bench `M` is *always* in the warmed set by construction
  (`sorted(set(BUCKETS)|{max_batch})`), so bench never pays an un-warmed compile for any bucket OR for `M`.
  Production warmup is caller-supplied; a production model must NOT assume the bench warmup set.

**Net:** the model is faithful to the bench drain's mechanics (W×E shaping, step service time, greedy
non-blocking drain, the over-M overshoot) and to its divergences from production (no leaf/bucket knobs, no
reload). The principal residual latitude is the intentional timing nondeterminism (§2), which is required,
not a defect.

---

## 9. Z3 confirmation (single bounded check; confirmation only, not the source of trust)

A minimal encoding in `out/z3_bucket_drain_check.py` confirms one representative execution is admissible:
the §5.4 over-M overshoot under bench W=group/E=bucket — a single producer message of `K=N*base` rows with
`K>M` is drained whole (loop-top check) and forwarded at width `>512` with zero padding.

Run (`nice -n 19 timeout 90 python z3_bucket_drain_check.py`):

```
FAITHFUL model (no width<=M clamp): {'result': 'sat', 'K': '600', 'real': '600', 'pad_to': '512', 'width': '600', 'pad': '0', 'M': '512'}
WRONG model (forces width<=M)     : {'result': 'unsat'}
OK: §5.4 over-M overshoot admissible; width=600>M=512, bucket=512, pad=0; a width<=M clamp wrongly forbids it.
```

Reading: with `base=200, N=3` ⇒ `K=600 > M=512`, the loop-top `total_rows<M` test (`inference_server.py:171`)
admits the whole 600-row message; `_bucket_for(600)` clamps to 512 (`stage_a_server.py:36-37`); since
`pad_to=512 < real=600`, `run_microbatch:58` adds NO padding and forwards width 600. A model that clamped
the forward width at `M` is UNSAT here — it would wrongly forbid a code-permitted execution. This confirms
the over-M / bucket-clamp latitude is real, not an artifact of the prose.
