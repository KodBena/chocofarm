# Faithful parametric model — Python ROUTER inference-server side of the leaf-eval transport

**Side modeled:** the Python single-threaded ZeroMQ `ROUTER` inference server, in
both drain variants in scope:

- **production greedy drain** — `chocofarm/az/inference_server.py`
  (`InferenceServer._drain` / `_serve_batch` / `run_microbatch`).
- **bench bucketed/leaf drain** — `cpp/stage_a/stage_a_server.py`
  (`StageAServer._serve_batch`, which overrides only `_serve_batch` and adds the
  `e_policy ∈ {padmax, bucket}` × `wakeup ∈ {group, leaf}` knobs).

All file:line references are to the cleanroom tree
`/home/bork/w/vdc/chocobo/runs/leaf-eval-model-2/cleanroom`. The line numbers are
the real source lines.

This document derives the abstraction whose set of representable executions equals
the system's set of real executions — forward from the code's operational
semantics, with no imposed expectation of how the server "ought" to behave. The
configuration is treated as **parameters**:

| Param | Meaning | Where it enters the server's world |
|---|---|---|
| `T` = `pool_threads` | # C++ worker threads = # DEALER peers = # ROUTER peer-pipes | each is one `WireLeafPool` (`runner_wire_batched.cpp:312-315`) |
| `N` = `trees_per_thread` | independent searches per thread | `K = N · ceil(pool_batch/T)` slots (`runner_wire_batched.cpp:285-286`) |
| `pool_batch` | base fan-out | `fibers_per_thread() = ceil(max(1,batch)/max(1,T))` (`runtime_config.hpp:12-15`) |
| `K` = `N · ceil(pool_batch/T)` | slots per thread → **max rows in one message from one peer** | gather loop `runner_wire_batched.cpp:437-443` |
| `D` = `max_inflight_msgs` | per-thread in-flight message cap | gates `inflight_msgs < D` (`runner_wire_batched.cpp:287,456,474`) |
| `max_batch` | server row cap per drained group; pad target | `InferenceServer.__init__` (`inference_server.py:149`), `_drain` (`:171`), `_serve_batch` pad (`:198`) |
| drain variant | `production` vs `stage_a(e_policy,wakeup)` | `_serve_batch` override (`stage_a_server.py:54-70`) |

The server is **parameter-blind to N, T, K, D**: it never reads any of them. Its
entire dependence on them is *induced through the wire* — through how many rows a
peer packs into a message, how many messages a peer keeps queued, and how many
peers exist. This induced dependence is the spine of the model and is made explicit
at every state, DOF, and execution below.

---

## 0. Ground facts established from the code (socket options, codec, threading)

### 0.1 The server ROUTER socket — every option is OS/libzmq default except those listed

`InferenceServer.__init__` (`inference_server.py:147-156`):

```
self._ctx  = context or zmq.Context()          # :152  — default context (1 I/O thread)
self._sock = self._ctx.socket(zmq.ROUTER)      # :153
self._sock.bind(bind)                          # :154  default "tcp://127.0.0.1:5599"
self._poller = zmq.Poller(); register POLLIN   # :155-156
```

Options **explicitly set on the ROUTER**: *none* on the socket itself except the
implicit ones from `bind`. The only `setsockopt` anywhere on the server path is in
`close`: `self._sock.close(linger=0)` (`inference_server.py:236`).

Therefore, derived from "not set in code" ⇒ libzmq 4.3.5 default:

| Option | Value the server runs with | Source |
|---|---|---|
| `ZMQ_SNDHWM` | **1000** (default) | not set anywhere |
| `ZMQ_RCVHWM` | **1000** (default) | not set anywhere |
| `ZMQ_ROUTER_MANDATORY` | **0 / OFF** (default) | not set anywhere → **scatter to an unknown/dead identity is silently dropped, never blocks, never errors** |
| `ZMQ_RCVTIMEO` | **-1** (infinite) — *but never exercised*: every recv is `recv_multipart(flags=NOBLOCK)` (`inference_server.py:173`) | not set; NOBLOCK overrides |
| `ZMQ_SNDTIMEO` | **-1** (infinite) — relevant: `send_multipart` (`:200`,`stage_a:70`) is **blocking** | not set |
| `ZMQ_LINGER` | default **-1** during run; **0** only at `close()` (`:236`) | `close(linger=0)` |
| `ZMQ_ROUTER_HANDOVER`, `ZMQ_IMMEDIATE`, etc. | default | not set |

The peer (RELY surface) sets, on its DEALER (`wire_leaf_pool.hpp:39-41`):
`ZMQ_LINGER=0`, `ZMQ_RCVTIMEO=timeout_ms` (default 15000, `runner_wire_batched.hpp:22`).
It does **not** set SNDHWM/RCVHWM/SNDTIMEO ⇒ DEALER SNDHWM=RCVHWM=1000 default, DEALER
sends blocking.

**Consequence for the scatter (`send_multipart`, `inference_server.py:200`).** Because
`ROUTER_MANDATORY=0`:
- if the destination identity is **known and its pipe has < SNDHWM(1000) queued**, the
  frame is enqueued without blocking;
- if the pipe is **at SNDHWM**, ROUTER with MANDATORY off **drops the message
  silently** (does not block, does not raise `EAGAIN`);
- if the identity is **unknown/disconnected**, the message is **dropped silently**.

So the server's scatter **cannot block on HWM and cannot raise `EHOSTUNREACH`** — it
either delivers or silently drops. Blocking of `send_multipart` is therefore bounded
by the libzmq internal pipe/lock, not by application-visible backpressure; we model
it as a positive but unbounded-above service time that never deadlocks the server (a
"send" transition that always completes). This is the single most important transport
fact for the server: **the server never applies wire backpressure to a peer and never
fails on send** under this configuration.

### 0.2 The frame shape on the wire (correlation-id matching)

DEALER sends two parts (`wire_leaf_pool.hpp:86-91`):
`[ corr (8 bytes, ZMQ_SNDMORE) ][ payload (encode_request bytes) ]`.

ROUTER prepends the peer identity ⇒ server `recv_multipart` yields
`frames = [ ident, corr, payload ]`. In `_drain` (`inference_server.py:176-178`):
`ident=frames[0]`, `envelope=frames[1:-1]=[corr]`, `payload=frames[-1]`.

Scatter (`:200` / `stage_a:70`): `send_multipart([ident, *envelope, resp])
= [ident, corr, resp]`. ROUTER strips `ident`, delivers `[corr, resp]` to that DEALER,
which `recv_corr_payload` reads as `frames.front()=corr` (8 bytes),
`frames.back()=resp` (`wire_leaf_pool.hpp:157-163`). **Correlation matching is the
peer's job, not the server's**: the server echoes the `corr` frame verbatim and never
inspects it. The server is *correlation-transparent*. This is the GUARANTEE the peer
relies on (§5).

### 0.3 Single-thread compute pin (sink-service determinism premise)

`config.py:5-6` sets `XLA_FLAGS=--xla_cpu_multi_thread_eigen=false` and
`OMP_NUM_THREADS=1` via `os.environ.setdefault` (imported at
`stage_a_server.py:14` and transitively by `inference_server`'s `forward` import).
The forward runs single-threaded; service time is a deterministic function of the
*compiled shape* plus host scheduling noise — not of host parallelism. This grounds
the sink-service timing model (§3).

### 0.4 The forward (sink compute) — service-time shape, not math

`forward.py:3-18`: two dense ReLU layers, optional residual block (`"Wr1" in params`),
value head always, policy logits head iff `"Wp" in params`. Cost is
`O(B · in_dim · h + B · h² + B · h · n_actions)` — **affine in the row count `B`**
with a fixed per-call overhead. Wrapped by `jit_forward_core` (`inference_server.py:22-34`)
under a one-element JIT cache `_jit_forward_cache`: the **first** call at a given input
shape pays XLA trace+compile; subsequent calls at the **same shape** hit the compiled
executable. `jax.jit` specializes on `Xb.shape`, so **each distinct padded row-count is
a distinct compilation**. This is why padding policy is load-bearing for *service
time*, derived in §3.

`run_microbatch` (`inference_server.py:40-73`) is the shape-former:
- concatenate all drained matrices into `Xb` of `B = Σ Bᵢ` rows (`:55-56`);
- if `pad_to > B`, append `pad_to − B` zero rows (`:58-59`);
- one `forward_fn` call (`:61`); slice predictions back to each requester by its row
  count `Bᵢ` and `encode_response` per requester (`:66-72`).

So the forward always runs at exactly `max(B, pad_to)` rows; **padding rows are pure
service-time cost, never returned** (`out_arr[:, …]` sliced to the real `B`, `:62-72`).

---

## 1. Operational state machine (the single server thread)

The server is one thread running `serve_forever` (`inference_server.py:219-225`):
`current()` once for warmup-of-cache; then loop `{ drained = _drain(); if not stop and
drained: _serve_batch(drained) }`. There is exactly one logical thread of control;
all concurrency is *external* (T peers + the libzmq I/O thread fill the ROUTER's
receive queue while this thread computes). The state machine is over this one thread,
with the ROUTER queue as an environment input.

### States

| State | Meaning |
|---|---|
| `WARMUP` | one-time: `warmup(batch_sizes)` precompiles forwards at each requested shape (`stage_a_server.py:82` → `inference_server.py:202-217`); `serve_forever` not yet looping. Optional (production caller may skip). |
| `POLL_WAIT` | inside `_drain` blocking poll loop (`:163-166`): `poller.poll(timeout=100ms)` until ≥1 readable frame or stop. ROUTER receive-queue may be accumulating from peers. |
| `DRAINING` | non-blocking `recv_multipart(NOBLOCK)` loop (`:171-185`) pulling queued frames into `drained` until `Again` or `total_rows ≥ max_batch`. |
| `REJECT` | a drained payload failed `decode_request`; `_reject` prints and the frame is **discarded with no reply** (`:181-183`,`:188-190`). Transient sub-state of DRAINING. |
| `RELOAD_CHK` | `_serve_batch` start: `params_source.poll()` may swap weights (production, `:194-195`); stage_a always uses `current()` (`stage_a_server.py:56`). |
| `FORWARD` | one `forward_fn` call at `max(B,pad_to)` rows inside `run_microbatch` (`:61`). Sink-service time elapses here. Production: one forward per drain. stage_a `group`: one per drain. stage_a `leaf`: **one forward per drained request** (loop over singleton groups, `stage_a_server.py:57-66`). |
| `SCATTER` | `send_multipart([ident,*envelope,resp])` per requester (`:200` / `stage_a:69-70`). Cannot HWM-block / cannot error (§0.1); may silently drop to a dead peer. |
| `STOPPED` | `self._stop` observed true at a loop/poll boundary (`:163,167,222,224`); `serve_forever` returns; `close()` does `sock.close(linger=0)` (`:236`). |

### Transitions

| # | From → To | Guard | Action | code_ref | free? |
|---|---|---|---|---|---|
| t0 | (init)→WARMUP | caller invokes `warmup` | for each shape, one padded forward (compile) | `stage_a_server.py:82`; `inference_server.py:210-217` | no |
| t1 | WARMUP→POLL_WAIT | warmup returns (or skipped) | enter `serve_forever` loop | `:219-223` | no |
| t2 | POLL_WAIT→POLL_WAIT | `poll(100ms)` timed out **and** `not stop` | re-loop, re-check stop | `:163-166` | **yes** (whether a frame arrived in the window is environment-set) |
| t3 | POLL_WAIT→STOPPED | `stop` observed | `_drain` returns `[]`; loop exits | `:163,167-168,222` | no |
| t4 | POLL_WAIT→DRAINING | `poll` reports ≥1 readable | break to drain loop | `:165-166` | no |
| t5 | DRAINING→DRAINING | `recv` returns a frame, `total_rows < max_batch`, decode OK | append `(ident,[corr],X)`; `total_rows += X.shape[0]` | `:171-185` | **yes** (how many frames currently queued is environment-set) |
| t6 | DRAINING→REJECT→DRAINING | a frame decodes-fail | print, drop, **no reply**, continue | `:179-183,188-190` | **yes** (only if peer sends malformed/short/NaN/bad-version frame — RELY says it won't) |
| t7 | DRAINING→RELOAD_CHK | `recv` raises `Again` **or** `total_rows ≥ max_batch` | exit drain loop with `drained` (len ≥1) | `:171-174,186` | no (deterministic given queue) |
| t8 | RELOAD_CHK→FORWARD | always; weights = `poll() or current()` | choose params; build groups | `:194-196`; `stage_a:56-57` | no |
| t9 | FORWARD→FORWARD | stage_a `wakeup=leaf` and >1 group remaining | next singleton group → next forward | `stage_a_server.py:57-66` | no |
| t10 | FORWARD→SCATTER | a forward completed (sink-service elapsed) | slice predictions per requester | `:61-72` | no |
| t11 | SCATTER→SCATTER | more requesters in this group | `send_multipart` next reply | `:197-200`; `stage_a:69-70` | no |
| t12 | SCATTER→FORWARD | stage_a `leaf`: more groups | back to t9 | `stage_a:58-70` | no |
| t13 | SCATTER→POLL_WAIT | all groups scattered, `not stop` | next `serve_forever` iteration | `:222-225` | no |
| t14 | SCATTER→STOPPED | `stop` set | loop guard `while not self._stop` fails | `:222` | no |

**Free-choice transitions** (the model's bounded nondeterminism, all *environment*-set,
never server-set): **t2** (did a frame arrive in this 100 ms window), **t5** (how many
frames are queued right now / when the next becomes visible), **t6** (peer
malformedness — excluded by RELY), and the **duration** of **t10** (sink service) and
**t11/t10's** placement relative to peer activity. The server itself makes *no* free
choices — it is fully deterministic given (its config, the ROUTER queue contents at
each `recv`, the wall-clock at each `poll`). All latitude lives in *when peers' frames
become visible to `recv`* and *how long the forward takes*.

### The one structural subtlety: the drain is a snapshot, not a steady-state pull

`_drain` pulls **only what is already enqueued at `recv`-time**, up to `max_batch`
rows; the instant `recv` returns `Again`, the drain ends (`:171-186`) even if a peer is
microseconds from sending more. So the batch composition is exactly the set of frames
the libzmq I/O thread had delivered into the ROUTER's receive queue by the moment of
each `recv`. This is the join point where all timing nondeterminism enters.

---

## 2. Per-peer queue depth and HWM — does any HWM get approached as N, T grow?

This is the assignment's central transport question. Derive it from the producer's
own pacing (RELY, grounded in `runner_wire_batched.cpp`):

- A peer keeps `inflight_msgs` strictly `< D` (gates at `:456` and `:474`; increments
  at `:448`, decrements at `:460`). It **cannot emit message D+1 until it has received
  a reply** (the recv at `:458` is what frees a slot). So **at most `D` messages from
  one peer are unconsumed-by-the-server at any instant**, and the ROUTER's *incoming*
  per-peer pipe holds **≤ D** messages for that peer.
- Each message carries **≤ K rows** (`issue_one` gathers all *ready* slots into one
  message, `:437-443`; at most K slots exist).

Therefore, **per-peer incoming queue depth ≤ D messages**, independent of N. The
RCVHWM is 1000. Since `D = max_inflight_msgs` defaults to 8 and is a small int, and is
**not a function of N**, the per-peer pipe depth `D ≪ 1000`. **RCVHWM is never
approached on any per-peer pipe**, for any N.

What *does* grow with N is **rows per message** (up to K = N·ceil(pool_batch/T)) and
hence **`total_rows` accumulation speed inside one drain** — but the drain stops the
moment a single message would carry it past `max_batch` only *after* appending (the
loop checks `total_rows < max_batch` at the top, `:171`, then appends the whole next
message, so the last appended message can overshoot `max_batch`; the cap is a
*loose* upper bound, not exact — see DOF-7). N grows the *width* of each transport
unit, not the *count* of queued units.

Aggregate over T peers: total messages resident in the ROUTER ≤ `T · D`. HWM in
libzmq is **per-pipe** (per peer), not aggregate, so even `T·D` total never threatens
any single pipe's 1000-message HWM. **Conclusion: under the producer's `D`-capped
pacing, no server-side HWM (RCVHWM or SNDHWM) is ever approached, for any (N, T, D)
with `T·D` modest and `D ≪ 1000`.** The HWM would only matter under a peer that
violates RELY (a peer that sends without waiting for replies, or `D` configured
≥ 1000); the model's HWM-block branch is therefore *unreachable* under RELY and is
documented as a guarded, RELY-excluded edge (§6 over-permission audit).

**Send side (SCATTER).** Symmetric: the server enqueues one reply per requester. A
peer drains its replies promptly (its loop body `recv_batch` at `:458` runs as soon as
the server scatters). The ROUTER's *outgoing* per-peer pipe holds the replies not yet
pulled by that DEALER. Bounded by the same `D` discipline (the peer has ≤ D
outstanding ⇒ ≤ D replies can be in flight to it). SNDHWM(1000) never approached. And
even if it were, MANDATORY-off ROUTER drops rather than blocks (§0.1) — so the server
*still* never blocks on send. The producer would then observe its `ZMQ_RCVTIMEO`
(15 s) fire and fail loudly (`wire_leaf_pool.hpp:41,147-150`); that is a peer-visible
failure, not a server state.

---

## 3. Timing model

### 3.1 Source emission (when peers' frames become visible) — bounded nondeterminism

The server treats each peer's request arrivals as an **external arrival process** it
does not control. The arrival of the *next* request from a given slot is gated by that
slot's search progress, which the code does **not** fix:
- a slot parks at a leaf when `policy.run_search` calls `ynet.predict`, which sets
  `ch.at_leaf=true` and yields the fiber (`fiber_leaf.hpp:24-28`); the host thread sees
  `ts->running=true` (`fiber_tree.hpp:55-56,58-62`);
- the **interval between consecutive parks of one slot** is internal search work
  (`apply_decision`→`spawn_ply`→possibly several plies, `runner_wire_batched.cpp:364-425`)
  plus reply-handling — an **a-priori-unbounded positive interval**, modeled as a
  nondeterministic duration `τ_park ∈ (0, ∞)`.

So **source emission is bounded nondeterminism**: each peer emits messages at
times chosen freely subject only to causal constraints (§3.3). We do **not** collapse
it to a constant or to an instant. Concretely the model's free parameters are: for each
peer p and each message m, an emission time `e(p,m) > 0`, with `e(p,m+1)` causally after
that peer received reply `m` if message `m` saturated its `D` cap.

### 3.2 Sink service (the forward) — modeled, not collapsed; shape-dependent

The forward (`FORWARD` state) consumes `R = max(B, pad_to)` rows and produces
predictions after a **service time** `S(R, compiled?)`:

- **Affine in R** with fixed overhead: `S ≈ S₀ + c·R` (from `forward.py` arithmetic,
  single-threaded per §0.3). `S₀` = dispatch + Python/JAX call overhead; `c` =
  per-row compute. Both **positive**.
- **Compilation discontinuity.** The *first* forward at a given `R` (i.e. a given
  `Xb.shape`) pays an XLA compile `S_compile(R) ≫ S₀+c·R` (`jit_forward_core` caches
  one traced fn, but `jax.jit` re-specializes per shape, `:22-34`). Subsequent calls
  at that R are warm. `warmup` (`:202-217`, called by stage_a at `:82` for
  `{64,256,512,max_batch}`) **moves these compiles out of the serving loop** for
  exactly those shapes; any *unwarmed* R hits a cold compile mid-serve. This is a real,
  derivable timing effect and is left in the model as: `S(R) = S_warm(R)` if R is in the
  warmed set or has been seen before, else `S_cold(R)` on first sight.
- **Padding shape policy ⇒ set of distinct R values ⇒ set of distinct service times.**
  This is the whole point of the E-policy knob:
  - **production / `padmax`**: every forward runs at `R = max_batch` (pad_to=max_batch,
    `:198`; `stage_a:62`). Exactly **one** compiled shape ever ⇒ after one warm-up,
    `S` is **constant = S_warm(max_batch)** regardless of real B. Service time is
    *decoupled from real load*. As N grows (bigger real B per drain), service time does
    **not** change until B itself exceeds max_batch — which it cannot, because the drain
    caps at max_batch rows (loosely, DOF-7).
  - **`bucket`**: `pad_to = _bucket_for(real) ∈ {64,256,512}` (`stage_a:30-37,63-64`),
    snapping real row count up to the smallest covering bucket (overflow → 512). So
    **three** warm shapes; `S = S_warm(bucket(real))`, a **step function of real load**.
    As N grows, `real` per drain rises, climbing 64→256→512 steps ⇒ service time rises
    in discrete jumps, saturating at S_warm(512).
- **wakeup granularity ⇒ number of forwards per drain.**
  - **`group`** (and production): **one** forward per drained group, `R` covering the
    whole group's rows. Service time per drain = one `S(R)`.
  - **`leaf`**: **one forward per drained request** (`groups=[[d] for d in drained]`,
    `stage_a:57`; loop `:58-66`). If the drain held `g` requests, the server pays
    `Σ_{i<g} S(R_i)` ≈ `g·S₀ + c·ΣRᵢ` — i.e. it **re-pays the fixed overhead `S₀`
    `g` times** and (under `bucket`) pads each tiny request up to its own bucket
    (≥64 rows each). As N grows, `g` (requests per drain) grows ⇒ leaf-mode service
    time grows roughly linearly in `g` with the fixed-overhead multiplier — the
    deliberately *pessimal* sink-service regime, and the reason it is a bench knob.

We **do not collapse the forward to an instant**. We model `S` as a positive duration,
a function `S(R, warm?, policy, wakeup)` with the structure above, otherwise free
(host scheduling noise ⇒ `S` is a nondeterministic positive value in a
policy/shape-determined band, not a fixed constant — except `padmax`, where the *shape*
is constant so the only residual freedom is host noise).

### 3.3 Causal constraints (the only constraints on the free timings)

1. **Durations positive**: every `τ_park > 0`, every `S > 0`, every poll window
   > 0 (100 ms), every send completes in finite > 0 time.
2. **A reply cannot precede the forward that produced it**: `SCATTER(m)` strictly
   after `FORWARD(m)` completes (`run_microbatch` returns before `send_multipart`,
   `:197-200`).
3. **A peer cannot emit a reply-dependent request before its reply arrives**: peer
   message `m+1` for a slot that was `submitted` cannot be emitted before the server's
   scatter of `m` was received and `resume_with` ran (`runner_wire_batched.cpp:466-471`).
   Equivalently, the `D`-cap: message m+D cannot precede reply m.
4. **The drain only sees frames the libzmq I/O thread delivered by `recv`-time**: a
   frame emitted at `e(p,m)` is visible to a `recv` at server-time `r` only if
   `e(p,m) + δ_wire ≤ r`, where `δ_wire > 0` is the (positive, nondeterministic)
   socket/transport delay. No frame is visible instantly; none is delayed forever
   (loopback TCP, finite delivery).
5. **Single-thread mutual exclusion**: the server does exactly one of
   {poll, recv, forward, send} at a time; while in `FORWARD`/`SCATTER` it issues **no**
   `recv`, so peer frames accumulate in the ROUTER queue and are first observed at the
   *next* `_drain` (`:160-186`). This is the mechanism by which **a long forward batches
   more requests for the following drain** — the self-batching feedback (§4, exec C).

**Nothing else constrains the timings.** In particular the model leaves free: the
interleaving of the T peers' emissions, which peer's frame arrives first, how many
frames coincide in one drain, and the exact `S` within its band.

### 3.4 What was *not* collapsed

Nothing in source/sink timing was collapsed to a constant. `padmax` makes the
*forward shape* constant (a derived fact, not a modeling shortcut); the *service time*
still carries host-noise freedom, and the *arrival process* remains fully
nondeterministic. The poll timeout is the one genuine constant (100 ms,
`_POLL_INTERVAL_MS`, `:142`) — it is a literal in the code, so modeling it as the
constant 100 ms is faithful, not a collapse.

---

## 4. Regimes induced by the parameters (qualitative phase structure)

The server has no internal modes, but the *induced* arrival/service interplay has
distinct regimes, all parameterized:

- **Starved / poll-spinning** (low N·T or slow searches): drains usually find exactly
  one or zero-then-one message; the 100 ms poll often times out (t2 self-loop). Batch
  efficiency low; under `bucket`, most forwards at the 64-bucket. As N grows this regime
  *recedes*: more slots ⇒ more frequent parks ⇒ poll rarely times out.
- **Coalescing / self-batching** (moderate–high N·T): while one forward runs, peers
  enqueue; the next drain scoops a large multi-peer, multi-row batch up to `max_batch`.
  This is the regime the design targets. As N grows, mean rows/drain rises toward
  `max_batch`; `mean_rows_per_msg` in the C++ summary (`:496-500`) rises. Self-batching
  is **self-reinforcing**: a bigger batch ⇒ longer forward ⇒ even more accumulate
  (exec C).
- **Saturated** (N·T large enough that ≥ `max_batch` rows are always queued): every
  drain hits the `total_rows ≥ max_batch` exit (t7 via `:171`), every forward at
  `R=max_batch` (padmax) or 512 (bucket overflow). Service time pinned at its ceiling;
  excess requests simply wait in per-peer queues (still ≤ D each, §2). This is where N
  stops increasing per-drain rows and starts increasing *queueing latency* instead.

---

## 5. Assume-guarantee contract (server as one party)

### RELY (what the server assumes about the peer over the wire — each checkable in `wire_leaf_pool.hpp` / `runner_wire_batched.cpp`)

- **R1 — well-formed requests.** Each request payload is `encode_request` output:
  `[ver=2][B][in_dim][B·in_dim f32]`, B≥1, in_dim≥1, finite (`inference_wire.py:42-61`
  validates; peer guarantees via `wire/encode_request`, `inference_wire.hpp:51-70`).
  *If violated* → `decode_request` raises → server `_reject` drops with no reply
  (t6) → peer's `RCVTIMEO` (15 s) fires. So malformedness is **tolerated, not fatal**
  to the server, but starves that peer.
- **R2 — DEALER 2-part frame `[corr8][payload]`.** Grounds `frames[1:-1]=[corr]`
  envelope handling (`:176-178`). Peer: `wire_leaf_pool.hpp:86-91`.
- **R3 — `D`-capped pacing.** Peer keeps `inflight_msgs < D` and does not emit beyond
  its cap before receiving replies (`runner_wire_batched.cpp:287,456,474`). This is
  what bounds per-peer queue depth ≤ D (§2) and guarantees no HWM pressure / no server
  send-block.
- **R4 — peer drains replies promptly.** Peer's loop recvs (`:458`) right after the
  server scatters, so SNDHWM is never approached and `linger=0` on both ends means a
  dying peer's queued frames are dropped, not leaked.
- **R5 — stable identity for the lifetime of an outstanding request.** ZMQ DEALER
  identity is stable per socket; the server echoes whatever `ident` ROUTER attached, so
  even an auto-assigned identity is fine **as long as it does not change between request
  and reply**. (DEALER identity is fixed per connection ⇒ holds.)
- **R6 — correlation matching is the peer's responsibility.** Peer matches `corr` to
  its `inflight_` map (`wire_leaf_pool.hpp:115-124`) and validates reply cardinality
  vs slot count. Server treats `corr` as opaque.

### GUARANTEE (what the server provides, checkable in `inference_server.py` / `stage_a_server.py`)

- **G1 — correlation transparency.** For every well-formed request `[ident,corr,payload]`
  the server admits into a batch, it emits **exactly one** reply `[ident,corr,resp]`
  with the *same* `ident` and the *byte-identical* `corr` frame (`_serve_batch` zips
  responses with `drained` preserving envelope, `:197-200`; stage_a `:69-70`).
- **G2 — per-requester row fidelity.** A request with `Bᵢ` rows gets a response with
  exactly `Bᵢ` predictions, sliced at the correct offset (`run_microbatch:66-72`),
  regardless of how it was batched/padded with others. Padding rows are never returned.
- **G3 — `encode_response` well-formed.** Reply payload is
  `[ver=2][B][n_actions][B·(1+n_actions) f32]` (`inference_wire.py:63-86`), decodable by
  the peer (`wire_leaf_pool.hpp:111`).
- **G4 — no silent reordering within a peer's outstanding set is *imposed* by the
  server**, but the server gives **no ordering guarantee across requests**: replies are
  scattered in drain order per group, which is ROUTER-arrival order, **not** request
  emission order across peers. (Peer tolerates this — it matches by `corr`, R6.)
- **G5 — liveness modulo stop.** Every admitted well-formed request is eventually
  answered unless `stop` is set (which only happens at end-of-run, `stage_a:110/124`).
  No request is held indefinitely while the server is serving: each drain that admits a
  request runs its forward and scatters before the next drain (t8→t10→t13). The only
  non-answer is **G-exception**: a malformed request (R1 violated → t6 drop) or a
  scatter dropped because the peer died (MANDATORY-off drop, §0.1) — both peer-fault,
  surfaced to the peer as its 15 s `RCVTIMEO`.

**A-G composition note.** G5's liveness is *conditional* on R3/R4: if a peer violated
R3 and flooded the ROUTER past RCVHWM(1000), libzmq would apply receive backpressure on
*that pipe* (peer's blocking send stalls) — the server stays live, the peer self-limits.
So even a RELY violation degrades gracefully on the server side; it never wedges the
single server thread. This is the assignment's "blocking surface" answer: **the server's
only blocking points are `poll(100ms)` (always times out, bounded) and `send_multipart`
(never HWM-blocks under MANDATORY-off; bounded by an internal lock). The server has no
unbounded blocking call and cannot deadlock against a peer.**

---

## 6. Degrees of freedom (each with code_ref, behaviors admitted, N-dependence)

See the structured object for the machine-readable list; summarized here:

- **DOF-1 Arrival interleaving across T peers** (`runner_wire_batched.cpp:437-452`,
  server-side observed at `_drain:171-185`). Admits: any order/coincidence of the T
  peers' messages in a drain. **N-dependence:** as N grows, each peer parks more slots
  per unit time ⇒ more frames coincide per drain ⇒ the interleaving space *widens* and
  large multi-peer batches become *more* common (regime shift toward coalescing).
- **DOF-2 Drain batch size B (rows) and group count g** (`_drain:171-186`). Admits:
  any `1 ≤ g ≤ (#queued)`, `1 ≤ B ≤` (last-message-overshoot of `max_batch`).
  **N-dependence:** B and g both rise with N; B saturates at ~`max_batch`, g saturates
  at T·D.
- **DOF-3 Sink service time S(R, warm?)** (`run_microbatch:61`; `forward.py`;
  `jit_forward_core:22-34`). Admits: any positive S in the shape/policy band, plus a
  one-time cold-compile spike per unseen R. **N-dependence:** under `bucket`, S steps up
  with N (real rows climb buckets); under `padmax`, S is N-independent (always
  max_batch shape).
- **DOF-4 E-policy {padmax,bucket}** (`stage_a_server.py:61-64`). Admits: forward shape
  ∈ {max_batch} vs {64,256,512}. **N-dependence:** padmax — N-invariant shape; bucket —
  N selects the bucket; both converge to a single ceiling shape at saturation.
- **DOF-5 wakeup {group,leaf}** (`stage_a_server.py:57`). Admits: 1 forward/drain vs
  g forwards/drain. **N-dependence:** group — N-invariant *count* (1); leaf — forward
  count = g grows with N, so per-drain service time grows ~linearly in N's effect on g
  and re-pays S₀ g times.
- **DOF-6 Weight reload timing (production only)** (`inference_server.py:194`,
  `RedisParamsSource.poll:129-138`). Admits: a forward may run under freshly-swapped
  weights at any drain boundary; stage_a never reloads (`StaticParamsSource.poll`
  returns None, `:110-111`; stage_a uses `current()`). **N-dependence:** none — reload
  cadence is set by an external version supplier, orthogonal to N.
- **DOF-7 max_batch overshoot** (`_drain:171`). The cap is checked *before* appending
  the next whole message, so B can exceed max_batch by up to (rows in the last message
  − 1), i.e. up to K−1 rows of overshoot. Admits: `B ∈ [1, max_batch + K − 1]`.
  **N-dependence:** overshoot bound K−1 = N·ceil(pool_batch/T) − 1 **grows linearly with
  N** — the one place where N directly enlarges the server's per-forward row count and
  hence the worst-case forward shape (under padmax, B>max_batch still pads to max_batch
  only when pad_to>B, so if B>max_batch no padding and the forward runs at B>max_batch —
  an *unwarmed* shape ⇒ cold-compile risk; under bucket, B>512 still snaps to 512 via
  `_bucket_for`'s fallthrough `:36`, so rows are *truncated to the 512 shape only by
  padding logic when pad_to>B fails* — note pad_to=512 < B means **no padding**, forward
  runs at the true B>512, an unwarmed shape). This is the subtle N-driven cold-compile
  edge.
- **DOF-8 REJECT path** (`_drain:179-183`). Admits: drop-with-no-reply on malformed
  frame. **N-dependence:** none under RELY (R1 makes it unreachable); reachable only via
  a non-conforming peer.
- **DOF-9 Poll-window outcome** (`_drain:163-166`). Admits: timeout-and-respin
  (no frame in 100 ms) vs proceed. **N-dependence:** as N grows, timeout outcome becomes
  *rare* (more parks ⇒ frames almost always pending), so the server spends progressively
  more time in DRAIN/FORWARD and less spinning POLL_WAIT.

---

## 7. Representative executions (concrete traces of enabled transitions)

### Exec A — production / single-peer, single-row (starved regime, N small)
1. POLL_WAIT, `poll(100ms)` returns readable (t4) — peer p₀ sent 1 msg, B=1.
2. DRAINING: recv `[id₀,corr,payload]`, decode OK, drained=[(id₀,[corr],X₁)],
   total_rows=1 (t5); next recv → `Again` (t7).
3. RELOAD_CHK: `poll()`→None, use current (t8).
4. FORWARD: `run_microbatch(..., pad_to=max_batch)` ⇒ R=max_batch, S=S_warm(max_batch)
   (t10).
5. SCATTER: `send_multipart([id₀,corr,resp])` (t11→t13) → POLL_WAIT.
**Stability:** transient — exists only while load < 1 row/100 ms.
**N-dependence:** *less* reachable as N grows; at higher N step 2 drains many messages
instead of one.

### Exec B — production / coalescing, multi-peer self-batch (target regime)
1. POLL_WAIT→DRAINING (t4): at recv-time the ROUTER queue holds m frames from peers
   {p₀…p_{T-1}}, total rows Σ < max_batch.
2. DRAINING loops (t5×m) draining all m frames (g=m groups), until `Again` (t7).
3. FORWARD at R=max_batch (padmax always pads to max_batch) (t10), S=S_warm(max_batch).
4. SCATTER ×m, each `[id_i,corr_i,resp_i]` (t11×m), then POLL_WAIT (t13).
**Stability:** self-reinforcing — see Exec C for why.
**N-dependence:** m and Σrows both grow with N; this trace becomes the *modal* one.

### Exec C — self-batching feedback (coupling source pacing to sink service)
1. FORWARD(m₁) running at shape R₁ for duration S₁ (single-thread; no recv during this).
2. During S₁, peers park more slots and `issue_one` sends new messages; by causal
   constraint §3.3-5 these sit in the ROUTER queue (server isn't recv-ing).
3. FORWARD(m₁) ends → SCATTER → POLL_WAIT → DRAINING immediately finds the *accumulated*
   queue (t4 fires at once, no 100 ms wait), drains a **larger** batch m₂ > m₁ (t5×m₂).
4. Larger batch ⇒ (bucket) larger R₂ ⇒ longer S₂ ⇒ *even more* accumulates during S₂.
**Stability:** **self-reinforcing** up to the saturation ceiling (R capped at max_batch /
512); a positive feedback that *amplifies* batch size with N until the cap, then
converts to queueing latency.
**N-dependence:** the feedback gain rises with N (more slots ⇒ faster accumulation);
saturation reached at lower wall-time as N grows.

### Exec D — stage_a bucket + leaf (pessimal sink regime)
1. DRAINING gathers g requests (each B_i small, often 1) (t5×g), `Again` (t7).
2. groups = g singletons (`stage_a:57`). FORWARD(group₁) at pad_to=`_bucket_for(B₁)`
   (≥64) (t9→t10); SCATTER reply₁ (t11→t12); FORWARD(group₂)… repeated g times.
3. Server pays g·S₀ + Σ c·bucket(B_i) ≥ g·S_warm(64).
**Stability:** transient per-drain, but *systematically* costly; chosen as a bench
worst-case.
**N-dependence:** g grows with N ⇒ per-drain forward count and padded-row waste grow ~
linearly in N; the **opposite** scaling to group+padmax (where N is absorbed into one
forward). This trace is the empirical justification for the coalescing design.

### Exec E — REJECT (RELY-violation edge, unreachable under conforming peer)
1. DRAINING recvs a frame whose payload is short/NaN/bad-version → `decode_request`
   raises (t6) → `_reject` prints, **no reply**, continue.
2. The offending peer never gets a reply for that corr ⇒ its `RCVTIMEO`(15s) fires
   peer-side. Server proceeds normally for all other requests.
**Stability:** n/a (one-shot). **N-dependence:** none; gated only by peer conformance.

### Exec F — stop during poll (shutdown)
1. POLL_WAIT, `stop` set by main thread (`stage_a:110/124`); `_drain` sees `not
   self._stop` false → returns `[]` (t3); `serve_forever` loop guard fails → STOPPED;
   `close()` → `sock.close(linger=0)` drops any queued frames.
**Stability:** terminal. **N-dependence:** none (any in-flight requests at stop are
abandoned regardless of N; their peers see RCVTIMEO).

---

## 8. n_dependence_summary

The Python server is **structurally N-invariant** — it reads none of N, T, K, D — and
its behavior changes with N **only through the wire**, monotonically along one axis:
**as N grows, each drain scoops more rows from more peers, so the modal execution
shifts from "poll-spin on one small request" (Exec A) toward "coalesce a near-`max_batch`
multi-peer batch and run one ceiling-shape forward" (Exec B/C), with the
self-batching feedback (Exec C) growing in gain until the `max_batch`/512 saturation
ceiling, after which additional N converts to per-peer queueing latency rather than
larger forwards.** Crucially **per-peer queue depth stays ≤ D (N-independent)** so **no
HWM is ever approached for any N**; the only place N enters the *forward shape*
directly is the `max_batch`-overshoot of up to K−1 = N·ceil(pool_batch/T)−1 rows
(DOF-7), which can push a forward onto an *unwarmed* shape and trigger a cold compile.
Under `padmax` the forward shape is N-invariant (always `max_batch`); under `bucket` it
steps up with N (64→256→512); under `leaf` wakeup the *forward count per drain* grows
with N (one per request), the deliberately pessimal regime.

---

## 9. DOF-control notes (what removing each constraint would let in)

- **Remove the `D`-cap (RELY-R3):** the per-peer queue depth becomes unbounded by the
  application; RCVHWM(1000) becomes reachable; a peer's blocking DEALER `zmq_send`
  stalls when its pipe hits HWM (the producer self-throttles), and the server could
  then see very deep coalescing batches (B ≫ max_batch on the wire, still capped per
  drain). Constraint kept ⇒ those deep-queue executions are unrepresentable.
- **Remove single-thread mutual exclusion (§3.3-5):** if the server could recv during a
  forward, the self-batching feedback (Exec C) would vanish and batches would be smaller
  and more uniform. Keeping it makes "recv-while-forwarding" executions unrepresentable
  — which is correct, the server is single-threaded.
- **Remove MANDATORY-off drop semantics (set ROUTER_MANDATORY=1):** scatter to a dead
  peer would *raise* `EHOSTUNREACH` instead of silently dropping; that would add a
  server-visible error transition out of SCATTER. The code does **not** set it, so that
  transition is unrepresentable — the server cannot fault on send. Keeping MANDATORY off
  forbids any send-error execution.
- **Remove the 100 ms poll bound (use infinite RCVTIMEO recv):** the stop flag could not
  be observed between requests; the bounded poll exists precisely to re-check stop
  (t2/t3). Keeping it makes "server hangs forever ignoring stop" unrepresentable.
- **Remove warmup (production caller skips it):** every first-seen shape pays
  `S_cold` mid-serve; with `bucket` that is three cold compiles, with `padmax` one,
  plus any DOF-7 overshoot shape. Keeping warmup (stage_a always calls it, `:82`) makes
  the cold-compile-mid-serve executions for the warmed shapes unrepresentable.

---

## 10. Fidelity self-audit

### Possible over-permissions (executions the model admits that the code may forbid)
- **OP-1 — HWM-block / EAGAIN on send.** The model documents an HWM/MANDATORY edge but
  marks it **RELY-excluded and unreachable** under R3/R4 and MANDATORY-off. If a reader
  treats §2's HWM branch as reachable under normal config it would over-permit; it is
  explicitly gated to RELY-violation. Faithful only with that gate.
- **OP-2 — REJECT (t6) under a conforming peer.** Admitted as a transition but tagged
  unreachable while R1 holds (the C++ `encode_request` cannot produce a malformed frame).
  Including it is faithful *only* because the peer set is open (a future/foreign peer
  could send junk); for the exact two peers in scope it is dead code, and the model says
  so.
- **OP-3 — cold-compile mid-serve for warmed shapes.** Admitted only for *unwarmed*
  shapes (DOF-7 overshoot, or a production caller that skipped warmup). If a reader
  applied it to `{64,256,512,max_batch}` after stage_a's warmup it would over-permit.
- **OP-4 — reordering of replies (G4).** The model lets replies leave in any
  ROUTER-arrival order; the code actually fixes the order to drain order per group
  (deterministic given the queue), so cross-*drain* reordering is real but
  within-group order is fixed. Stated as "no ordering guarantee," which is slightly
  more permissive than the code's deterministic-within-group behavior — intentional,
  since the peer imposes no within-group ordering requirement and matches by `corr`.

### Possible over-constraints (executions the code can produce that the model may forbid)
- **OC-1 — partial drain below queued frames.** The model says the drain pulls all
  queued frames up to max_batch; but `recv_multipart(NOBLOCK)` can return `Again`
  *transiently* even when a frame is mid-delivery by the I/O thread, ending the drain
  early. The model captures this via DOF-2/§1's "snapshot, not steady-state" note and
  §3.3-4's `δ_wire` — so it is **not** over-constrained, but a reader who assumed
  "drain always empties the queue" would be. Flagged to be safe.
- **OC-2 — `max_batch` treated as an exact cap.** Naively one might forbid B>max_batch;
  the code permits overshoot by up to K−1 (DOF-7). The model includes the overshoot, so
  it is not over-constrained on this point.
- **OC-3 — assuming exactly one forward per drain.** True for production/group, **false**
  for `leaf` (g forwards). The state machine's t9/t12 self-loops encode the multi-forward
  case, so leaf executions are representable.
- **OC-4 — weight reload only in production.** The model forbids reload in stage_a
  (StaticParamsSource.poll→None). This is faithful: stage_a's `_serve_batch` calls
  `current()` directly and never `poll()` (`stage_a:56`), so a reload transition is
  genuinely impossible there.

### Net
The model is *tight* on the load-bearing transport facts (socket options all derived
from set-vs-default; HWM provably never approached under RELY; send never blocks/faults
under MANDATORY-off; single-thread exclusion drives self-batching). The only
deliberate looseness is keeping the RELY-violation edges (OP-1, OP-2) and the open-peer
reordering latitude (OP-4) in the model with explicit unreachability/looseness tags,
because faithfulness to *the open transport boundary* (not just the two in-scope peers)
warrants representing them as guarded, not deleting them.
