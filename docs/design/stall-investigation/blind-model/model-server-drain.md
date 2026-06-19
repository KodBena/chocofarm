# A faithful operational model of the Python server side of the leaf-evaluation transport boundary

**Side:** SERVER (sink). The single-threaded Python `InferenceServer` (ROUTER socket) + the
`run_microbatch` gather/pad/forward/scatter + the MLP forward sink (`forward_core`).

**Objective:** the abstraction whose set of representable executions equals the system's set of real
executions — no more, no fewer. Derived FORWARD from the code's operational semantics; no outside
expectation of how the service "ought" to behave is introduced.

**Files read end to end (ADR-0002):**

- `chocofarm/az/inference_server.py` (read in full, lines 1–457) — my side.
- `chocofarm/az/forward.py` (read in full, lines 1–64) — the sink compute.
- `chocofarm/az/inference_wire.py` (read in full, lines 1–185) — the value codec.
- `chocofarm/az/wire_spec.py` (read in full, lines 1–84) — the frame layout SSOT.
- `cpp/include/chocofarm/wire_leaf_pool.hpp` (read in full, lines 1–243) — the peer's DEALER wrapper.
- `cpp/src/runner_wire_batched.cpp` (read in full, lines 1–630) — the peer's driver (strict-barrier and pipelined).
- `docs/design/zmq-inference-service.md` (read in full, lines 1–367) — protocol design intent.
- `chocofarm/config.py` lines 34–53 — the XLA/OMP single-thread pin + the fixed inference batch size.

This model is the server's view. Cross-boundary facts about the peer are isolated into the RELY section;
everything else is grounded in the server's own code.

---

## 0. Orientation — what the server is, mechanically

`InferenceServer.serve_forever` (inference_server.py:428–439) is a single OS thread running an unbounded
loop of three operational phases:

```
serve_forever:                              # :436
  self._params_source.current()             # :435  — assert params exist (loud, before serving)
  while not self._stop:                      # :436
    drained = self._drain()                  # :437  — PHASE 1: block-then-greedy-drain
    if not self._stop and drained:           # :438
      self._serve_batch(drained)             # :439  — PHASE 2+3: poll-reload, ONE forward, scatter
```

There is exactly one thread. The class docstring (:291–300) and the module docstring (:31–35) pin this:
"SINGLE-THREADED: JAX/XLA owns the forward, no shared-state concurrency and no XLA-in-a-worker-thread."
There is therefore no intra-server concurrency to model: the only concurrency at this boundary is between
(a) the server thread's compute and (b) the *peers'* sends arriving asynchronously into the ROUTER
socket's OS/zmq receive buffer while the server thread is busy. **That buffer is where the system's
concurrency lives**, and the batch size the server forms is a pure function of how much of it has
accumulated at the instant `_drain` runs its non-blocking recv loop. Modeling that accumulation faithfully
is the heart of this document.

The socket is a `zmq.ROUTER` (:315) bound at `tcp://127.0.0.1:5599` by default (:306). A `zmq.Poller`
registered for `POLLIN` (:317–318) is the only thing the loop blocks on.

---

## 1. ZeroMQ socket options — determined from the code (governs blocking behavior)

The prompt requires every socket option be read off the code (set explicitly vs OS/library default),
because blocking behavior depends on exactly which options are set.

**Server ROUTER socket (`self._sock`, inference_server.py:315):**

| option | value | source |
| --- | --- | --- |
| socket type | `ROUTER` | :315 `self._ctx.socket(zmq.ROUTER)` — explicit |
| bind addr | `tcp://127.0.0.1:5599` (default) | :306, :316 `self._sock.bind(bind)` — explicit |
| `LINGER` (at open) | **not set** → library default (−1, infinite) | no `setsockopt` anywhere in the file |
| `LINGER` (at close) | **0** | :454 `self._sock.close(linger=0)` — explicit, close-time only |
| `RCVTIMEO` | **not set** → default −1 (infinite) | no setsockopt; the loop instead uses the *poller* timeout |
| `SNDTIMEO` | **not set** → default −1 (infinite) | no setsockopt; `send_multipart` (:387) is blocking-by-default |
| `RCVHWM` | **not set** → default **1000** | no setsockopt; libzmq default high-water mark |
| `SNDHWM` | **not set** → default **1000** | no setsockopt |
| `ROUTER_MANDATORY` | **not set** → default 0 (drop on unroutable) | no setsockopt — relevant to scatter, §4.4 |
| `IMMEDIATE` | **not set** → default 0 | no setsockopt |
| context `IO_THREADS` | **not set** → default 1 | `zmq.Context()` :314 with no `.set(zmq.IO_THREADS,…)` |
| context `MAX_SOCKETS` | **not set** → default | :314 |

I confirmed by `grep` over the file that the ONLY socket-affecting calls are `socket(ROUTER)`,
`bind`, `Poller.register`, the close-time `close(linger=0)`, and `Context.term()`. There is **no**
`setsockopt`/`set_hwm`/`set` on the ROUTER. So every HWM/timeout/linger above its open-time value is a
pyzmq/libzmq default. This is load-bearing for two model facts:

1. **The recv side never blocks inside the drain loop.** The drain does `recv_multipart(flags=zmq.NOBLOCK)`
   (:350) and catches `zmq.Again` (:351) to break. The blocking is done *separately* by the poller
   (:342). So `RCVTIMEO` being infinite is irrelevant — `NOBLOCK` overrides it per-call.
2. **The send side CAN block** in principle. `send_multipart` (:387) uses default `SNDTIMEO=−1` (infinite
   block) and `SNDHWM=1000`. On a ROUTER, if a peer's pipe is full (its `RCVHWM` reached) the ROUTER
   either drops or blocks depending on `ROUTER_MANDATORY`; with `ROUTER_MANDATORY` unset (the default,
   0), a ROUTER **silently drops** a message addressed to an unroutable/full peer rather than blocking.
   This is a real degree of freedom (DOF-7, §6) the code leaves and the design does not mention.

**Peer DEALER socket (`WireLeafPool`, wire_leaf_pool.hpp:77–83), for the RELY:**

| option | value | source |
| --- | --- | --- |
| socket type | `ZMQ_DEALER` | :77 — explicit |
| `ZMQ_LINGER` | **0** | :82 — explicit |
| `ZMQ_RCVTIMEO` | `timeout_ms` (a `WireRunnerConfig` field) | :83 — explicit |
| `ZMQ_SNDTIMEO` | not set → default −1 (infinite) | no setsockopt |
| `ZMQ_SNDHWM` / `ZMQ_RCVHWM` | not set → default 1000 each | no setsockopt |

This DEALER↔ROUTER pairing and its HWMs bound the RELY in §5.

---

## 2. The frame the server parses (value codec, grounded in code)

A ROUTER `recv_multipart` (:350) yields `frames = [identity][envelope…][payload]`:

- `ident = frames[0]` (:353) — the ROUTER identity prefix libzmq prepends, the routing key for the scatter.
- `envelope = frames[1:-1]` (:354) — opaque, echoed verbatim. For the production C++ DEALER peer this is
  exactly one frame: the 8-byte correlation id (`[corr-id][payload]`, wire_leaf_pool.hpp:138–144). The
  server **never parses** it (:329, :130). For a REQ peer it would be a single empty delimiter; for a bare
  DEALER, empty. The server is envelope-agnostic by construction.
- `payload = frames[-1]` (:355) — the value frame, decoded by `decode_request` (:357,
  inference_wire.py:105–127): `[ver:u8][B:u32 LE][in_dim:u32 LE][X:f32×(B·in_dim) LE]`, protocol version 2
  (wire_spec.py:49). Returns a `(B_i, in_dim)` float32 matrix; `B_i ≥ 1` (a B=0 frame is a loud
  `WireError`, inference_wire.py:115). The peer's strict-barrier driver sends `B_i = #parked slots` rows in
  one frame (runner_wire_batched.cpp:321); the pipelined driver sends `B_i = #ready slots` (:562); the
  degenerate per-leaf path sends `B_i=1` (wire_leaf_pool.hpp:119–122).

The codec validates at the boundary (Port/ACL translate-and-validate, inference_wire.py:34): unknown
protocol byte, header-too-short, `B==0`, `in_dim==0`, payload byte-count ≠ `B·in_dim·4`, or a non-finite
entry each raise `WireError` (:110–126). A raised `WireError` in `_drain` is caught (:358) and routed to
`_reject` (:365–370), which **logs and drops** — the malformed request's identity is omitted from the
batch, never zero-filled (:328). This is a genuine transition in the state machine (T-REJECT, §3).

---

## 3. The operational state machine

The server thread is a sequential process; its "state" is the loop's program point plus the contents of
the ROUTER receive buffer (which is mutated by the peers, asynchronously). I model the program point as the
explicit control state and the buffer as an environment variable `Q` (the multiset of fully-received,
not-yet-drained request frames sitting in the socket's receive buffer).

### 3.1 States

| name | meaning |
| --- | --- |
| `INIT` | constructed; `serve_forever` not yet entered. `current()` asserted (:435). |
| `POLL_BLOCK` | inside `_drain`'s bounded-poll loop (:339–343); blocked in `poller.poll(timeout=100ms)` waiting for `Q` to become non-empty or for the 100 ms wakeup. |
| `DRAINING` | poll returned readable; running the non-blocking recv loop (:348–362), moving frames out of `Q` into the local `drained` list until `Q` empties (`zmq.Again`, :351) or `total_rows` would reach `max_batch` (:348). |
| `RELOAD_CHK` | `_serve_batch` entered (:381); calling `params_source.poll()` to see if a new weight version is published. |
| `FORWARD` | inside `run_microbatch` (:134–189): concat→pad→ONE `forward_fn` call; the device→host pull (`np.asarray`, :177) blocks the thread until XLA finishes. This is the SINK SERVICE TIME. |
| `SCATTER` | `_serve_batch`'s send loop (:384–387): one `send_multipart([ident,*envelope,resp])` per drained request, in drained order. |
| `STOPPED` | `_stop` observed; loop exits (:436/:438) or `_drain` returns `[]` (:344–345). Terminal for `serve_forever`. |

`WARMUP` is a distinct pre-loop state (:389–426, §3.4) entered only if the operator calls `warmup(...)`
before `serve_forever`. It runs `run_microbatch` over dummy zero batches for a set of B's, forcing XLA to
compile the single padded shape, then discards the outputs (no socket touched).

### 3.2 Transitions (each with guard / action / code_ref / free-vs-determined)

| # | from → to | guard | action | code_ref | free? |
| --- | --- | --- | --- | --- | --- |
| T0 | INIT → POLL_BLOCK | `serve_forever` called, `not _stop` | assert `current()`; enter loop | :435–437 | determined |
| T1 | POLL_BLOCK → POLL_BLOCK | `poll(100ms)` returned empty (timeout) AND `not _stop` | re-issue poll | :339–342 | **determined** (the re-loop), but the *number* of self-loops is environment-driven (how long `Q` stays empty) |
| T2 | POLL_BLOCK → STOPPED | `_stop` is True at loop top, or after an empty poll | `_drain` returns `[]` | :339,:344–345 | determined (given `_stop`); *when* `_stop` flips is the peer/operator's free choice |
| T3 | POLL_BLOCK → DRAINING | `poll(100ms)` returned readable (≥1 frame in `Q`) AND `not _stop` | `break` out of poll loop | :342–344 | determined given `Q≠∅` at a wakeup; **`|Q|` at this instant is a FREE environment value** (DOF-1) |
| T4 | DRAINING → DRAINING | `total_rows < max_batch` AND `recv_multipart(NOBLOCK)` succeeded AND payload decoded OK | append `(ident,envelope,X)`; `total_rows += X.shape[0]` | :348–362 | determined per-frame; **which/how-many frames are present is free** (DOF-1) |
| T-REJECT | DRAINING → DRAINING | recv succeeded but `decode_request` raised | `_reject(ident,exc)`: log+drop; `continue` (frame NOT counted, identity NOT batched) | :356–360, :365 | determined given a malformed frame; whether a malformed frame *exists* is a peer free choice (RELY says it won't, §5) |
| T5 | DRAINING → RELOAD_CHK | `recv_multipart(NOBLOCK)` raised `zmq.Again` (Q empty) OR `total_rows ≥ max_batch` (cap hit) | exit recv loop; return `drained` (len ≥ 1 if any non-malformed arrived) | :348,:351–352,:363 | **determined** by `Q`/cap; the cap-vs-drain-empty distinction is a free environment outcome (DOF-2) |
| T5e | DRAINING → POLL_BLOCK | the only frame(s) in `Q` were all malformed → `drained == []` | `serve_forever`'s `if … drained` is False (:438); loop re-enters `_drain` | :438,:339 | determined |
| T6 | RELOAD_CHK → FORWARD | always (after `poll()` returns) | `params,y_mean,y_std = poll() or current()`; build `rows` | :381–383 | determined; **whether poll() returns new params is a peer/registry free choice** (DOF-6) |
| T7 | FORWARD → SCATTER | the `np.asarray(forward_fn(...))` returned (XLA done) | hold the `(rows,1+n_actions)` block; build per-identity responses | :177–189 | determined; **the duration of this transition (service time) is a bounded-nondeterministic FREE value** (DOF-3) |
| T8 | SCATTER → SCATTER | more drained requests remain | `send_multipart([ident,*envelope,resp])` for the next, in drained order | :384–387 | determined (order fixed); send may block on a full peer pipe or silently drop (DOF-7) |
| T9 | SCATTER → POLL_BLOCK | all drained responses sent AND `not _stop` | return to loop top | :387,:436–437 | determined |
| T10 | SCATTER → STOPPED | all sent AND `_stop` | loop condition fails | :436 | determined |
| TW | INIT → WARMUP → INIT/POLL_BLOCK | operator called `warmup` before `serve_forever` | compile the padded shape via dummy `run_microbatch`; discard | :389–426 | determined (a pre-flight) |

**The single genuine FREE input** driving every qualitatively distinct execution is the pair
`(arrival schedule of peer requests into Q, service duration of each forward)`. Everything inside the
server is a deterministic function of that input. The model's faithfulness rests on giving that input
exactly the latitude the code+causality leave (§4), no more, no fewer.

### 3.3 The drain cap arithmetic (exact, :347–363)

`total_rows` accumulates **rows**, not requests: `total_rows += X.shape[0]` (:362) where `X` is the
`(B_i, in_dim)` matrix of request i. The loop condition is `while total_rows < self._max_batch` (:348).
Critically, the cap is checked **before** the recv, and a request is appended **in full** once recv'd:

- A request is drained iff, *at the moment the while-condition is tested*, `total_rows < max_batch`. So the
  final drained request can push `total_rows` to **anywhere in `[max_batch, max_batch + (B_last − 1)]`** —
  the cap is a *soft* cap on the pre-request total, not a hard cap on the post-request total. Example:
  `max_batch=64`, already at `total_rows=63`, next request carries `B_i=10` → it is appended (63 < 64),
  `total_rows` becomes 73. **The concatenated matrix can exceed `max_batch` rows.**

This matters for the FORWARD: `run_microbatch` pads to `pad_to=max_batch` **only if `pad_to > B`**
(:171). When the drained total `B` already `≥ max_batch`, **no padding happens** and the forward runs over
a shape `(B, in_dim)` with `B > max_batch` — a *new* XLA shape not covered by warmup (DOF-4, a service-time
spike). The server-side comment (:170 "this only ever pads UP") is **slightly optimistic**: it pads up
*to* `max_batch` whenever `B < max_batch`, but when the cap-overrun above produces `B ≥ max_batch` it pads
not at all and the shape is `(B, in_dim) ≠ (max_batch, in_dim)`. This is a real, code-grounded degree of
freedom in the forward's service-time behavior (§6 DOF-4). With the production peers each B_i can be > 1
(batched/coalesced sends), so the overrun is reachable, not hypothetical.

### 3.4 Warmup (`warmup`, :389–426)

`warmup(batch_sizes)` runs, for each `b` in the supplied iterable, `run_microbatch(... [(b"", zeros(b,
in_dim))], pad_to=max_batch)`. Because `pad_to=max_batch`, **every** such call (for any `b ≤ max_batch`)
pads to the SAME `(max_batch, in_dim)` shape (:171–172) and `run_microbatch`'s `np.asarray` (:177) forces
XLA to compile-and-run that one shape. So warmup compiles **exactly one executable** — the padded shape —
regardless of how many distinct `b` it is given. `in_dim` is derived from `params["W1"].shape[0]` (:417),
fail-loud if `W1` absent (:413–416) or `b < 1` (:420–421). Warmup never touches the socket; outputs
discarded (:426). Consequence for the timing model: **after warmup, the steady-state forward over any
`B ≤ max_batch` is a single pre-compiled shape**, so its service time has no JIT-compile component — §4.2.

---

## 4. The timing model — the two nondeterministic inputs, constrained only as code+causality require

### 4.1 SOURCE timing (arrivals into `Q`) — a first-class nondeterministic input

The server cannot observe *why* a request arrives when it does; it observes only that, at the instant the
poller wakes and the recv loop runs, some multiset `Q` of complete frames is present. From the server's
code, the only constraints on arrivals are:

- **(C-arr-1) Positivity / monotonic time.** Each request's arrival timestamp `a_i ≥ 0`; the model orders
  events on a single real timeline. (Causal necessity; no code fixes a value.)
- **(C-arr-2) A request is reply-gated at the peer.** A peer cannot emit a request whose features depend on
  a prediction it has not yet received. This is a RELY about the peer (§5, R3): the C++ driver `resume_with`s a
  slot with its reply and only then re-parks/advances it (runner_wire_batched.cpp:330–331, 589–590) before
  that slot's row can be gathered into the *next* send. So for a given correlation chain on a given slot,
  `a_{next} > (server's send time of the reply for that slot)`. **This is the only causal link between the
  server's outputs and its future inputs**, and it is what makes the boundary a closed loop rather than an
  open arrival process.
- **(C-arr-3) Distinct slots are independent.** Different slots / different threads / different episodes
  emit on their own internal schedules, which the server's code does not constrain at all. So the *number*
  of requests queued at any drain instant is bounded only by the live concurrency (T·D outstanding messages
  for the pipelined driver, T·1 for the strict-barrier driver — RELY R4) and by the socket HWM (1000,
  §1) — and is otherwise a **free environment value** in `[0 … that bound]` at each instant.
- **(C-arr-4) ROUTER HWM back-pressure.** With `RCVHWM=1000` (default), `Q` cannot exceed ~1000 queued
  messages per peer pipe before libzmq stops reading from that peer's TCP socket. At the production scale
  (T ≤ 4 threads, D ≤ small) this is never reached, so I model `Q` as effectively unbounded by the HWM and
  bounded instead by RELY R4's outstanding-message count. (If a future config pushed T·D·K past 1000 this
  bound would bind — recorded, not pinned.)

**How I represent it:** arrivals are a bounded-nondeterministic sequence of timestamped frames; the model
admits *any* schedule satisfying C-arr-1..4. I do **not** pin inter-arrival to a constant or to an instant
(that would forbid real executions where the search's variable internal work spaces requests out, or where
a burst of N threads' replies all become re-emittable at once). I do **not** allow a frame to arrive before
the reply it depends on (C-arr-2) nor more outstanding than R4 permits. This is exactly the latitude the
producer-side search work leaves: "when a thread emits its next request is set by the search's own
progress, which the code does not fix."

### 4.2 SINK timing (the forward service time) — faithful, batch-size dependent, padding-shaped

The service time is the wall-clock duration of transition T7 (FORWARD): the time from entering
`run_microbatch`'s `forward_fn(...)` call to the `np.asarray` device→host pull returning (:177). Derived
from the code:

- **(C-svc-1) Strictly positive.** The forward is a real MLP matmul chain (forward.py:50–62: two dense
  layers + optional residual block + value/policy heads). `np.asarray` blocks until XLA completes (:177,
  warmup docstring :404–406). Duration `> 0` — never instantaneous/zero-cost. This is the single most
  important faithfulness constraint on the sink: a model that collapses T7 to an instant forbids every real
  execution (because while T7 runs, arrivals accumulate — §4.3).
- **(C-svc-2) Single-threaded compute.** `config.py:41` sets `XLA_FLAGS=--xla_cpu_multi_thread_eigen=false`
  and `:42` `OMP_NUM_THREADS=1`. So the forward runs on a single Eigen thread with no work-stealing pool.
  This *removes* one source of service-time nondeterminism (no thread-pool scheduling jitter) but does NOT
  make the duration a fixed constant (OS scheduling, cache state, and the VM's shared-host contention remain
  — the standing "uncalibrated time model" caveat, design §9). So service time is a *bounded* nondeterministic
  positive value, tighter than if the Eigen pool were on, but still not pinned.
- **(C-svc-3) Padded to ONE shape ⇒ service time is (almost) batch-size-independent.** `run_microbatch`
  pads the concatenated `(B, in_dim)` matrix to `(max_batch, in_dim)` whenever `B < max_batch` (:171–172),
  and `jit_forward_core` compiles a SINGLE executable for that one shape (:95–115, docstring "The server
  pads every batch to one shape, so it compiles a SINGLE executable"). The forward computes over all
  `max_batch` rows regardless of how many are real (padded rows are zeros; only the first B are read back,
  :178–189). **Therefore the steady-state service time is the cost of the `(max_batch, in_dim)` forward —
  essentially the SAME for any `B ∈ [1, max_batch]`.** This is a derived, non-obvious sink property: under
  this padding discipline, a batch of 1 real leaf costs the *same* forward as a batch of `max_batch` leaves.
  The batching wins amortization in *throughput per leaf*, not in per-forward latency.
- **(C-svc-4) Two service-time regimes from the code, not one.**
  - *Compiled-shape regime (the common case):* `B ≤ max_batch` ⇒ padded to `(max_batch, in_dim)` ⇒ the one
    warmed executable ⇒ a bounded positive duration `S_pad` ≈ constant across `B` (per C-svc-3).
  - *Overrun regime (the cap-overshoot case, §3.3):* the drained total `B ≥ max_batch` (a request whose
    rows pushed the total over the soft cap) ⇒ `pad_to > B` is false ⇒ **no padding** ⇒ a forward over
    `(B, in_dim)` with `B > max_batch`, a shape NOT covered by warmup. On a **cold** server (or any never-
    seen B) the first such forward additionally pays an XLA **JIT-compile** latency (warmup docstring
    :397–411 names exactly this: "a per-B compile latency"), making `S_overrun(B) = compile(B) + run(B)`
    on first sight of B and `run(B)` thereafter. `run(B) > S_pad` (more rows). So the model must admit a
    service-time *spike* for the first occurrence of each overrun shape, and a larger-than-`S_pad` duration
    for every overrun. Collapsing service time to one constant would forbid these.
- **(C-svc-5) One forward at a time (serialization).** The single thread runs exactly one `forward_fn` per
  loop iteration, in series. There is no overlap of two forwards. Any requests that arrive during T7
  accumulate in `Q` and are only seen by the *next* `_drain`. This is the engine of the batch-size feedback
  loop (§4.3).

**How I represent it:** each forward's service time is a free positive real drawn per-occurrence, subject
to (i) `> 0`; (ii) the same compiled shape ⇒ durations drawn from one bounded family `S_pad` (I do not
pin a value, only that it is positive and bounded by the host); (iii) an overrun shape ⇒ a larger family
`S_overrun(B)`, with a one-time additive compile term on first sight of each B; (iv) no two forwards
overlap (serialization). I deliberately do **not** model the forward as cost-proportional-to-B (the padding
makes it shape-fixed, C-svc-3), nor as a constant (the overrun regime + host contention forbid that).

### 4.3 The batch-size feedback loop — the central faithful behavior

Combine §4.1 and §4.2. The server's batch size at iteration k is `B_k = |Q at the instant _drain's recv
loop runs, capped|`. `Q` at that instant = (everything that arrived during the previous forward `T7_{k-1}`
and its scatter `SCATTER_{k-1}` and any poll-wait) minus what previous drains already took. Because the
forward is serialized (C-svc-5) and positive (C-svc-1):

> **Self-clocking, derived:** under light load, the loop spends most of its time in POLL_BLOCK; the first
> request to arrive wakes it (T3) and the recv loop usually finds just that one (or few) ⇒ `B_k ≈ 1`. Under
> heavy load, the previous forward `T7_{k-1}` takes long enough that many peers' requests pile into `Q`
> during it; the next drain scoops them all up to the cap ⇒ `B_k → max_batch`. **`B_k` is an emergent
> function of (offered load) × (service time of the previous forward), not a tunable** — design §3
> "B self-scales with load below the cap (no latency timer to tune)" and :299–300 confirm this is the
> intended self-clocking, and the code (:339–363) realizes it with no timer.

This is the qualitatively distinct behavior the bounded Z3 check (§7) witnesses: two consecutive batches of
*different* sizes arising purely because different numbers of requests were queued at the two serialized
drain instants.

### 4.4 Did I collapse any timing to a constant?

No. Both inputs are bounded-nondeterministic positive durations. The one place the code *itself* removes
latitude is C-svc-3 (padding makes per-forward latency shape-fixed across `B ≤ max_batch`) — and that is a
*derived* near-constancy of service time *across batch sizes*, not a collapse of the timeline: each forward
still has a free positive duration drawn from a bounded family, forwards do not overlap, and the overrun
regime (C-svc-4) explicitly breaks the across-B constancy. The arrival timeline is fully free within C-arr-1..4.

---

## 5. Assume–Guarantee contract

I model the SERVER as one party of a two-party protocol. Each RELY is checkable against the peer's code
(wire_leaf_pool.hpp / runner_wire_batched.cpp); each GUARANTEE is grounded in the server's code.

### 5.1 RELY (what the server assumes about the peer, observed over the wire)

- **R1 — well-formed value frames.** Each request payload the peer sends decodes under `decode_request`:
  version byte 2, `B≥1`, `in_dim≥1`, payload exactly `B·in_dim·4` bytes, all finite.
  *Checkable:* the peer builds payloads via `wire::encode_request(flat, B, in_dim)` (wire_leaf_pool.hpp:133)
  derived from the same `wire_spec.hpp` SSOT (drift-tested). If R1 is violated the server does NOT crash —
  it `_reject`s that frame (T-REJECT), so a RELY breach degrades gracefully to a dropped request, not a server fault.
- **R2 — the correlation-id envelope is opaque and round-trippable.** The peer sends `[corr-id (8 bytes)]
  [payload]` (wire_leaf_pool.hpp:138–144); on a ROUTER this becomes `[identity][corr-id][payload]`. The
  server's `frames[1:-1]` envelope capture (:354) and verbatim echo (:387) require only that the peer
  treats the leading frame as opaque and matches replies on it. *Checkable:* `recv_corr_payload`
  (wire_leaf_pool.hpp:210–235) reads `frames.front()` as the corr-id and `frames.back()` as the payload,
  requiring `≥2 frames` and an 8-byte leading frame — exactly what the server echoes.
- **R3 — replies gate the peer's next request on that slot (closed-loop).** The peer does not emit a
  follow-up request for a slot until it has received and applied that slot's reply (resume_with →
  re-park/advance → re-gather: runner_wire_batched.cpp:330–331, 589–590). *Checkable:* the strict-barrier
  driver issues exactly one outstanding message per thread and blocks on its reply (:321–324); the
  pipelined driver holds at most `D` outstanding and marks a slot `submitted` until its reply clears it
  (:543, :564, :588). This grounds C-arr-2.
- **R4 — bounded outstanding messages.** Total messages outstanding to the server at any instant is at most
  `T` (strict-barrier: one per thread, :321) or `T·D` (pipelined: `D=max_inflight_msgs` per thread,
  :392, :578–596). So `|Q| + (in-flight forwards' requests)` is bounded by this, far below `RCVHWM=1000` at
  production T,D. This bounds the per-drain batch size's free range.
- **R5 — the peer expects exactly one reply per request, B predictions for a B-row request, matched by
  corr-id, and will abort loudly on a mismatch.** *Checkable:* `recv_batch` aborts if `decoded->size() !=
  slots.size()` (wire_leaf_pool.hpp:185) or on an unknown corr-id (:180). So the server MUST return exactly
  `B_i` predictions per `B_i`-row request, echoing the corr-id — which constrains GUARANTEE G3.
- **R6 — the peer's recv has a finite RCVTIMEO.** `ZMQ_RCVTIMEO=timeout_ms` (wire_leaf_pool.hpp:83). So a
  request the server *drops* (a `_reject`, or a ROUTER-MANDATORY-off silent drop on scatter, DOF-7) surfaces
  at the peer as a recv timeout → a loud whole-pass abort (the driver's `set_error`), not a silent hang.
  This is what makes the server's drop-on-reject safe at the system level.

### 5.2 GUARANTEE (what the server guarantees to the peer, from its own code)

- **G1 — every accepted request gets exactly one reply, addressed to its identity, echoing its envelope
  verbatim.** `_serve_batch` sends `[ident, *envelope, resp]` (:387) for each drained request, 1:1 in
  drained order (`run_microbatch` returns responses aligned to `requests`, :142, :184–189; re-paired with
  envelopes by position via `zip`, :384). Grounded: :383–387.
- **G2 — the reply carries exactly `B_i` predictions for a `B_i`-row request.** `run_microbatch` scatters
  `v[off:off+n]` / `out_arr[off:off+n, 1:]` per request with `n = X.shape[0]` (:184–186), `encode_response`
  emits `B=n` records (inference_wire.py:138–158). So G2 holds row-for-row, satisfying R5.
- **G3 — predictions are de-standardized values + RAW logits.** The value is de-standardized on-device
  (`v = v_std·ys + ym`, :112) and logits are raw (forward.py:62; design §2 masking stays client-side).
  Grounded: jit_forward_core :110–113, run_microbatch :180–186.
- **G4 — a malformed request is rejected loudly (logged + dropped), never coerced into a zero/garbage
  forward, and never corrupts another request's reply.** `_reject` logs and drops (:365–370); the
  `continue` (:360) keeps `total_rows` and the rest of the batch unaffected. Grounded: :356–363.
- **G5 — every leaf in one batch sees one consistent net version.** Params are read once per batch in
  `_serve_batch` (`poll() or current()`, :381–382) and the entire `run_microbatch` runs under that single
  `params` snapshot; a version change is observed only *between* batches (T6/RELOAD_CHK, design §3
  "between batches"). Grounded: :381–385.
- **G6 — single forward per batch (no duplicate or partial evaluation).** Exactly one `forward_fn` call per
  `_serve_batch` (:177 inside one `run_microbatch`, :385). Grounded.
- **G7 — bounded shutdown latency, socket not closed mid-recv.** `stop()` flips a flag observed within
  `_POLL_INTERVAL_MS=100` by the bounded poll (:304, :339–342, :441–445); `close()` is called only after the
  loop is between polls (:447–456 contract). So a clean shutdown never closes the socket out from under a
  recv (design §3 / docstring :337). This is a self-guarantee, not peer-facing, but it bounds the worst-case
  reply latency a peer can see during shutdown to ≤100 ms + one forward.

---

## 6. Degrees of freedom the code leaves (each with code_ref and admitted behaviors)

- **DOF-1 — batch composition (which & how many requests at a drain instant).** `_drain`'s non-blocking
  recv loop (:348–362) takes *whatever is currently in `Q`* up to the cap; `Q`'s contents at that instant
  are a free environment value (§4.1). *Admits:* `B_k` anywhere in `[1, max_batch]` (and, via DOF-2, beyond),
  any partition of the offered load into batches, and — when multiple frames are simultaneously ready — any
  recv *order* among them (libzmq's fair-queuing across peer pipes is not pinned by the server code), so the
  drained-order (hence scatter order) among simultaneous arrivals is free.
- **DOF-2 — soft cap overrun.** The cap is tested on the pre-request total (`while total_rows < max_batch`,
  :348) and a request is appended whole (:361–362), so the post-batch row total ranges over
  `[max_batch, max_batch + B_last − 1]` when the cap is crossed by a multi-row request (§3.3). *Admits:*
  forwards over `B > max_batch` rows.
- **DOF-3 — forward service duration.** Free positive real per forward (§4.2, C-svc-1/2). *Admits:* any
  ordering of "arrivals during a forward" — i.e. how much `Q` grows before the next drain — and thus drives
  DOF-1's `B_{k+1}`.
- **DOF-4 — service-time regime / JIT spike.** The padded-shape vs overrun-shape branch (:171) and the
  first-sight-of-a-shape XLA compile (warmup docstring :397–411). *Admits:* a steady `S_pad` for all
  `B ≤ max_batch`, a larger `S_overrun(B)` for overruns, and a one-time compile spike on the first
  occurrence of each never-warmed shape. *Removed by warmup* (§3.4) only for the single padded shape, not
  for overrun shapes.
- **DOF-5 — poll-wakeup phase.** The 100 ms bounded poll (:304, :342) means an idle server can sit up to
  100 ms in POLL_BLOCK self-loops before noticing a flipped `_stop` (T1/T2), and a request that arrives
  just after a poll returns empty waits up to ~100 ms... *no* — correction grounded in code: the poll
  returns *immediately* when a frame is readable (`poll` is level-triggered on POLLIN), so an arriving
  request does NOT wait for the next 100 ms tick; the 100 ms only bounds the *idle re-check of `_stop`*
  (:341 comment "wakes every _POLL_INTERVAL_MS so a flipped `_stop` is observed promptly"). *Admits:* up to
  100 ms shutdown-observation latency; **no** added per-request latency under load.
- **DOF-6 — between-batch weight reload.** `poll()` may return new params (:381) iff the peer/registry
  advanced the version (RedisParamsSource.poll, :279–288). *Admits:* a params swap (and its `read_weights`
  cost added to that iteration's RELOAD_CHK) between any two batches; `StaticParamsSource.poll` always
  returns None (:249–250) so the test path never reloads.
- **DOF-7 — scatter drop on an unroutable/full peer.** With `ROUTER_MANDATORY` unset (§1), a
  `send_multipart` (:387) to a peer whose pipe is full or whose identity is gone is **silently dropped** by
  libzmq (no error raised, default SNDHWM=1000). *Admits:* a reply that never reaches its peer; the peer's
  finite RCVTIMEO (R6) then surfaces it as a loud abort. The server code does not detect this — a genuine,
  un-guarded latitude. (At production T·D ≪ HWM and with live peers this is not exercised, but the code
  admits it.)
- **DOF-8 — reject-only batch ⇒ empty drain ⇒ re-poll.** If every frame in a wakeup is malformed,
  `drained == []` and the loop re-enters `_drain` without a forward (:438, T5e). *Admits:* a wakeup that
  performs work (rejects) but issues no forward and no reply for those identities.

---

## 7. Representative executions (concrete traces of genuinely-enabled transitions)

Each trace is a sequence of the §3.2 transitions with code_refs; I state for each whether the regime it
exercises is self-reinforcing or transient.

### E1 — Idle → single-leaf (B≈1), self-clocking under light load. Exercises DOF-1, DOF-3.

1. POLL_BLOCK, `Q=∅`: `poll(100ms)` times out repeatedly (T1, :342) — self-loops at ~0 CPU.
2. One peer request arrives → `a_0`; next poll returns readable (T3, :342).
3. DRAINING: `recv_multipart(NOBLOCK)` gets request 0 (T4, :350,:361); `total_rows=B_0`; next recv raises
   `zmq.Again` (`Q` now empty) → exit (T5, :351). `drained=[req0]`.
4. RELOAD_CHK: `poll()` → None (T6, :381). FORWARD over `(B_0, in_dim)` padded to `(max_batch,in_dim)`
   (:172), service `S_pad` (T7, :177). SCATTER one reply to req0's identity+corr-id (T8/T9, :387).
5. Back to POLL_BLOCK. Because the peer is reply-gated (R3) and load is light, by the time it emits again
   `Q` is again ~1 deep → step repeats with `B≈1`.

*Stability:* **self-reinforcing while offered load < 1/`S_pad`.** Each batch is ~1 because the forward
finishes before a second request piles up. A faithful steady state of the light-load regime.

### E2 — Burst → cap-sized batch, then leftover. Exercises DOF-1, DOF-2, DOF-3, C-svc-5. (Z3-witnessed, §8.)

1. While forward `T7_{k-1}` runs (positive duration, C-svc-1), `max_batch+1`... say 5 single-row requests
   from different slots/threads pile into `Q` (R4 permits up to T·D outstanding) — serialization (C-svc-5)
   guarantees the server cannot touch them until `T7_{k-1}` returns.
2. POLL_BLOCK → T3 (Q non-empty at the next wakeup, :342).
3. DRAINING: recv loop scoops requests until `total_rows < max_batch` fails. With `max_batch=4` and five
   1-row requests queued: it takes 4 (T4×4), then on the 5th the while-condition `4 < 4` is false → exits
   (T5, :348) **leaving the 5th in `Q`**. `drained` has 4. (Note: among simultaneously-queued frames, *which*
   4 it takes is DOF-1 free; the Z3 model picks {0,1,2,4}.)
4. RELOAD_CHK→FORWARD over `(4, in_dim)` = `(max_batch,in_dim)`, no padding needed, service `S_pad` (T7).
   SCATTER 4 replies in drained order (T8×4, :387).
5. Loop top → `_drain` again. The 5th request is still in `Q` (plus anything new). Next batch `B=1`
   (the leftover) ⇒ a forward over `(1,in_dim)` padded to `(max_batch,in_dim)`, service `S_pad` again.

So two consecutive batches of sizes 4 and 1 from one burst — **distinct sizes purely from arrival timing
vs the serialized drain instants.** *Stability:* **transient** — it is the response to a burst; once the
backlog clears the loop relaxes toward E1 (or toward E3 if load is sustained).

### E3 — Sustained heavy load → batches pinned near the cap. Exercises DOF-1, DOF-3, C-svc-3.

1. Offered load (across T·D outstanding messages, R4) exceeds `1/S_pad`: more requests arrive per `S_pad`
   than one forward consumes... but each forward consumes *up to* `max_batch` rows. If offered rows/`S_pad`
   ≥ `max_batch`, then during each forward `Q` refills to ≥ `max_batch`.
2. Every drain hits the cap (T5 via `total_rows ≥ max_batch`, :348): `B_k = max_batch` (or just over, DOF-2)
   every iteration. The forward is the same `(max_batch,in_dim)` shape, service `S_pad` (C-svc-3 — the
   padded latency is the *same* whether real B is 1 or max_batch, so at the cap there is no extra cost).
3. Throughput = `max_batch / S_pad` leaves/sec, the amortization the design buys.

*Stability:* **self-reinforcing while offered load sustains ≥ max_batch rows per `S_pad`.** Because each
forward both (a) consumes max_batch and (b) lets max_batch more pile up, the cap is a fixed point. If load
dips, it relaxes to E1.

### E4 — Cap overrun ⇒ unpadded forward ⇒ service-time spike. Exercises DOF-2, DOF-4.

1. Production peers send multi-row requests (`B_i > 1`, the coalesced gather, runner_wire_batched.cpp:556).
   Suppose `max_batch=64`, `Q` holds requests of 30 + 30 + 10 rows.
2. DRAINING: take req(30) → total 30 (<64, T4); take req(30) → total 60 (<64, T4); test `60 < 64` true →
   take req(10) → total 70 (T4); test `70 < 64` false → exit (T5, :348). `drained` totals **70 rows**.
3. FORWARD: `pad_to=64`, `64 > 70` is **false** → no padding (:171). Forward over `(70, in_dim)`, a shape
   **not warmed**. On first sight, XLA compiles it (DOF-4): service `= compile(70) + run(70) > S_pad`
   (warmup docstring :397–411). On subsequent same-shape overruns, `run(70)` only, still `> S_pad`.
4. SCATTER 3 replies (10/30/30... in drained order, T8×3).

*Stability:* **transient per distinct overrun shape** (the compile term is one-time per shape); the
larger-than-`S_pad` run term recurs for every overrun. Faithful: the model must admit forwards over
`B > max_batch` and a first-sight JIT spike, both grounded in :171 and the warmup docstring.

### E5 — Malformed frame ⇒ reject, batch unaffected; reject-only ⇒ empty drain. Exercises T-REJECT, DOF-8.

1. DRAINING: req A decodes OK (T4); req B's payload byte-count ≠ `B·in_dim·4` → `decode_request` raises
   `WireError` (inference_wire.py:121) → caught (:358) → `_reject(identB)` logs+drops (T-REJECT, :365);
   `continue` (:360) — `total_rows` unchanged, A still batched.
2. If A existed, FORWARD over `[A]` only; B gets no reply (its peer times out, R6). If the *only* frame was
   B, `drained=[]` → `serve_forever`'s `if … drained` False (:438) → re-poll without a forward (T5e/DOF-8).

*Stability:* **transient** (depends on a RELY-R1 breach, which the contract says won't occur in steady
operation; included to pin the graceful-degradation behavior the code admits).

### E6 — Between-batch weight reload. Exercises DOF-6, G5.

1. Between batch k and k+1: RELOAD_CHK calls `poll()` (:381). With `RedisParamsSource`, `version_supplier()`
   advanced → `read_weights` + `params_from_manifest_blob` rebuild params (:283–288), returns the new triple
   → batch k+1 runs under the new params (T6). Every leaf in batch k+1 sees the new version (G5); batch k
   saw the old. *Stability:* transient (a one-time swap per version bump); with `StaticParamsSource`,
   `poll()` is always None (:250) so this transition never fires.

---

## 8. The bounded confirmation run (Z3) — confirmation, not source

I encoded E2's core latitude in `check_server_drain_admissible.py` (z3 4.16,
`/home/bork/w/vdc/venvs/generic/bin/python`) and ran ONE check under `nice -n 19 timeout 90`. The encoding
constrains: positive per-forward service times (`svc_k > 0`, C-svc-1); reply-after-forward
(`f_k = t_k + svc_k`, C-svc-1); **serialized drains** (`t_1 ≥ f_0`, C-svc-5); block-until-≥1 and
greedy-drain-up-to-cap with the soft cap (`size_k ≤ MAX_BATCH`, a queued request is taken unless the batch
is full — :348); and asks for two consecutive batches of *different* sizes arising from the arrival/drain
timing.

Result: **`sat`**. Witness (`MAX_BATCH=4`, 5 single-row requests):

```
batch0 size = 4   members [0,1,2,4]     (cap-sized: total_rows reached MAX_BATCH, the 5th left in Q)
batch1 size = 1   members [3]           (the leftover, drained next)
drain starts t = [0, 1/2]    svc = [1/2, 1]    forward fin f = [1/2, 3/2]
```

The second drain starts at `t=1/2 = f_0` — i.e. only after the first forward finished (serialization), and
the two batch sizes differ (4 vs 1) purely from how many requests were queued at each serialized drain
instant. This confirms E2/E3's qualitative latitude is jointly admissible with positivity, reply-after-
forward, and single-threaded serialization. (The membership {0,1,2,4} vs {3} among equal-time arrivals is
itself DOF-1's free recv-order — faithful, since the server code does not pin which of simultaneously-queued
frames recv_multipart returns first.) This is a confirmation of the derivation in §3–§4, not its source.

---

## 9. Fidelity self-audit

### 9.1 Could the model be TOO PERMISSIVE (admit executions the code cannot produce)?

- *Could `B_k` exceed what concurrency allows?* No — RELY R4 bounds outstanding messages to T or T·D, so
  the model caps `|Q|` accordingly; I did NOT let `B` range to `RCVHWM`=1000 freely (that would admit
  impossible batches at production T,D). The bound is the peer-grounded R4, not the raw HWM. *Possible
  over-permission:* if a future config raised T·D·K above the peer-side reasoning, R4's specific number
  would need re-deriving; I pinned it to the code's `D=max_inflight_msgs` and `T=pool_threads`.
- *Could two forwards overlap?* No — C-svc-5 / single-thread (:436 loop is sequential) forbids it; the Z3
  encoding enforces `t_1 ≥ f_0`.
- *Could a reply precede its forward?* No — `encode_response` is built from `out_arr` which is the
  `np.asarray` of the forward (:177–187); SCATTER (T8) strictly follows FORWARD (T7) in the same thread.
- *Could a request be answered twice or with another's prediction?* No — `run_microbatch` scatters by
  contiguous offset aligned 1:1 to drained order (:184–189), one send per request (:387). G1/G2.
- *Could the scatter reorder relative to drained order?* The model fixes SCATTER order = drained order
  (:384–387 `zip` over the aligned lists). I did NOT leave scatter order free — the code determines it.
  (Among *simultaneous arrivals* the drained order itself is free, DOF-1, but once drained, scatter order
  is determined.)

### 9.2 Could the model be TOO CONSTRAINED (forbid executions the code can produce)?

- *Did I pin service time to a constant?* No (§4.4) — it is a bounded positive free value, with two regimes
  (C-svc-4). Pinning it would forbid the host-contention variation (design §9 caveat) and the overrun JIT
  spike (E4).
- *Did I forbid the cap overrun?* No — DOF-2/E4 explicitly admit `B > max_batch` and the unpadded forward,
  grounded in :171's `pad_to > B` guard. A naive reading ("drain caps at max_batch") would have forbidden
  E4; the soft-cap arithmetic (§3.3) preserves it.
- *Did I forbid the reject / empty-drain paths?* No — T-REJECT and T5e/DOF-8 are in the machine
  (:356–363, :438).
- *Did I forbid simultaneous-arrival recv-order freedom?* No — DOF-1 explicitly leaves the recv order among
  simultaneously-ready frames free (the server code does not pin libzmq's cross-pipe fair-queue order).
- *Did I over-constrain arrivals?* The only arrival constraints are C-arr-1..4, all code/causality grounded;
  inter-arrival is otherwise fully free (§4.1). *Possible over-constraint to flag:* C-arr-2 (reply-gating)
  is a RELY about the peer; if a *different* (non-reply-gated) client connected, the server would accept its
  un-gated requests too. My model scopes arrivals to the documented C++ peer's reply-gated discipline (R3);
  a hypothetical fire-and-forget client is outside the modeled peer set. This is a scoping choice, named here.

### 9.3 Named residual risks

- **Over-permission risk:** DOF-7 (silent scatter drop with ROUTER_MANDATORY off) is admitted by the model
  because the code admits it, but in practice with live peers below HWM it is never exercised; I keep it in
  the machine because removing it would be an outside "ought," not a code fact.
- **Over-constraint risk:** I model `S_pad` as one bounded family across all `B ≤ max_batch` (C-svc-3). If
  XLA's padded forward had a measurable B-dependence *despite* the fixed shape (it should not — same shape,
  same executable, padded rows are real zeros that still get multiplied), that would be a slight
  over-constraint. The code (one compiled shape, :95–115, :171–172) supports the constancy; this is the
  honest reading, not a simplification.

---

## 10. Code-derivation attestation

Every state, transition, guard, free choice, and timing constraint in this model is derived FORWARD from
the operational semantics of the six required files (read end to end) plus `wire_spec.py` and the
`config.py` XLA/OMP pin, each cited by line. No behavior is asserted from an outside expectation of how an
inference service "ought" to work: the self-clocking batch size is read off `_drain`'s recv loop and the
cap arithmetic (:339–363), the soft-cap overrun off the `pad_to > B` guard (:171), the service-time regimes
off the padding + XLA-single-shape + warmup discipline (:95–115, :171–172, :389–426, config.py:41–47), the
single-thread serialization off the loop structure and the design's explicit single-threaded statement
(:291–300, :436), and every socket-blocking fact off the *absence* of `setsockopt` calls (§1, grep-verified)
versus the explicit `bind`/`close(linger=0)`/poller. Cross-boundary assumptions are isolated as RELYs, each
checked against a specific line of the peer's C++ code. The Z3 run (§8) is a confirmation of the derivation,
kept minimal, never its source.

*Public Domain (The Unlicense).*
